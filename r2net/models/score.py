"""Conditional residual score / displacement network (paper Eq. 11-14)."""
from __future__ import annotations

import torch
import torch.nn as nn


class ResnetBlockConv1d(nn.Module):
    def __init__(self, c_dim: int, size_in: int, size_h: int | None = None,
                 norm_method: str = "batch_norm"):
        super().__init__()
        size_h = size_in if size_h is None else size_h
        self.size_in = size_in
        self.size_h = size_h
        if norm_method == "batch_norm":
            norm = nn.BatchNorm1d
        elif norm_method == "sync_batch_norm":
            norm = nn.SyncBatchNorm
        else:
            raise ValueError(norm_method)
        self.bn_0 = norm(size_in)
        self.bn_1 = norm(size_h)
        self.fc_0 = nn.Conv1d(size_in, size_h, 1)
        self.fc_1 = nn.Conv1d(size_h, size_in, 1)
        self.fc_c = nn.Conv1d(c_dim, size_in, 1)
        self.actvn = nn.ReLU(inplace=True)
        self.shortcut = None
        nn.init.zeros_(self.fc_1.weight)
        nn.init.zeros_(self.fc_1.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        net = self.fc_0(self.actvn(self.bn_0(x)))
        dx = self.fc_1(self.actvn(self.bn_1(net)))
        return x + dx + self.fc_c(c)


class ScoreNet(nn.Module):
    """Predicts a 3-D displacement/score for each local-frame point."""

    def __init__(
        self,
        z_dim: int,
        dim: int = 3,
        out_dim: int = 3,
        hidden_size: int = 128,
        num_blocks: int = 4,
    ):
        super().__init__()
        self.z_dim = int(z_dim)
        self.dim = int(dim)
        self.out_dim = int(out_dim)
        self.hidden_size = int(hidden_size)
        self.num_blocks = int(num_blocks)
        c_dim = self.z_dim + self.dim
        self.conv_p = nn.Conv1d(c_dim, hidden_size, 1)
        self.blocks = nn.ModuleList(
            [ResnetBlockConv1d(c_dim, hidden_size) for _ in range(num_blocks)]
        )
        self.bn_out = nn.BatchNorm1d(hidden_size)
        self.actvn_out = nn.ReLU(inplace=True)
        self.conv_out = nn.Conv1d(hidden_size, out_dim, 1)
        nn.init.normal_(self.conv_out.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """x: (B,K,3); c: (B,F) -> (B,K,3)."""
        p = x.transpose(1, 2)                     # (B,3,K)
        c_expand = c.unsqueeze(2).expand(-1, -1, p.size(2))
        c_xyz = torch.cat([p, c_expand], dim=1)   # (B,3+F,K)
        net = self.conv_p(c_xyz)
        for block in self.blocks:
            net = block(net, c_xyz)
        return self.conv_out(self.actvn_out(self.bn_out(net))).transpose(1, 2)
