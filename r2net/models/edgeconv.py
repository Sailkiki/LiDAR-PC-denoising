"""OC-Conv: offset-corrected graph convolution + ACE calibration.

This is the corrected implementation of the module described in the paper.
Compared with the original public release it:
  * predicts a 3-D offset and applies it to coordinates, not latent channels;
  * keeps the attention-selected top-K neighbours (the old code overwrote
    them with raw kNN indices);
  * scores each candidate edge individually before top-K selection;
  * makes ACE optional for ablation studies.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..ops import knn_idx, knn_group
from .layers import Aggregator, FCLayer, FCLayerFirst


class ACE(nn.Module):
    """Adaptive Channel Enhancement (paper Eq. 6-10).

    Three parallel branches:
      * multi-pooling channel attention via a shared MLP;
      * spatial attention from channel-wise avg/max statistics;
      * SE-like global-context channel gate.
    The output is X * (W_channel * W_spatial * W_SE) + X.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=True),
        )
        self.spatial_gate = nn.Sequential(
            nn.Linear(2, 1, bias=True),
            nn.Sigmoid(),
        )
        self.se_gate = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, K, C)
        avg_n = x.mean(dim=2)                  # (B,N,C)
        max_n = x.max(dim=2).values            # (B,N,C)
        w_channel = torch.sigmoid(
            self.channel_mlp(avg_n) + self.channel_mlp(max_n)
        ).unsqueeze(2)                         # (B,N,1,C)
        spatial_in = torch.cat(
            [x.mean(dim=-1, keepdim=True), x.max(dim=-1, keepdim=True).values],
            dim=-1,
        )                                      # (B,N,K,2)
        w_spatial = self.spatial_gate(spatial_in)  # (B,N,K,1)
        w_se = self.se_gate(avg_n).unsqueeze(2)    # (B,N,1,C)
        weight = w_channel * w_spatial * w_se
        return x * weight + x


class OffsetPredictor(nn.Module):
    """Lightweight MLP f_offset(z_i) -> 3-D offset, bounded by 0.1*tanh."""

    def __init__(self, in_channels: int, hidden_dim: int = 64, scale: float = 0.1):
        super().__init__()
        self.scale = float(scale)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x) * self.scale


class DenseEdgeConv(nn.Module):
    """Dense edge convolution with optional OC-Conv and ACE."""

    def __init__(
        self,
        in_channels: int,
        num_fc_layers: int = 3,
        growth_rate: int = 12,
        knn: int = 16,
        aggr: str = "max",
        activation: str = "relu",
        use_oc: bool = True,
        use_ace: bool = True,
        candidate_ratio: float = 1.5,
        attention_hidden: int = 64,
        ace_reduction: int = 8,
    ):
        super().__init__()
        if num_fc_layers < 3:
            raise ValueError("num_fc_layers must be >= 3")
        self.in_channels = int(in_channels)
        self.knn = int(knn)
        self.num_fc_layers = int(num_fc_layers)
        self.growth_rate = int(growth_rate)
        self.use_oc = bool(use_oc)
        self.use_ace = bool(use_ace)
        self.candidate_k = int(round(self.knn * candidate_ratio))

        if self.use_oc:
            self.offset_predictor = OffsetPredictor(in_channels, hidden_dim=64, scale=0.1)
            self.edge_attention = nn.Sequential(
                nn.Linear(3 * in_channels + 3, attention_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(attention_hidden, 1),
            )

        self.layer_first = FCLayerFirst(3 * in_channels, 4 * in_channels, growth_rate, bias=True)
        self.layers = nn.ModuleList()
        for i in range(1, num_fc_layers - 1):
            self.layers.append(
                FCLayer(in_channels + i * growth_rate, growth_rate, bias=True,
                        activation=activation)
            )
        self.layer_last = FCLayer(
            in_channels + (num_fc_layers - 1) * growth_rate,
            growth_rate, bias=True, activation=None,
        )
        gate_channels = in_channels + 2 * growth_rate
        self.ace = ACE(gate_channels, reduction=ace_reduction) if self.use_ace else nn.Identity()
        self.aggr = Aggregator(aggr)

    @property
    def out_channels(self) -> int:
        return self.in_channels + self.num_fc_layers * self.growth_rate

    def get_edge_feature(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        # e_ij = concat(z_i, z_j, z_j - z_i)   (paper Eq. 2)
        knn_feat = knn_group(x, idx)
        x_tiled = x.unsqueeze(-2).expand_as(knn_feat)
        return torch.cat([x_tiled, knn_feat, knn_feat - x_tiled], dim=-1)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """x: (B,N,C) features; pos: (B,N,3) centred coordinates."""
        if self.use_oc:
            offsets = self.offset_predictor(x)                # (B,N,3)
            deformed = pos + offsets
            idx_cand = knn_idx(deformed, deformed, k=self.candidate_k, offset=1)
            edge_feat_cand = self.get_edge_feature(x, idx_cand)
            rel_pos = knn_group(deformed, idx_cand) - deformed.unsqueeze(2)
            logits = self.edge_attention(
                torch.cat([edge_feat_cand, rel_pos], dim=-1)
            ).squeeze(-1)                                      # (B,N,K')
            selected_logits, topk = torch.topk(logits, self.knn, dim=-1)
            idx = torch.gather(idx_cand, -1, topk)
            edge_weights = 2.0 * torch.sigmoid(selected_logits).unsqueeze(-1)
            edge_feat = self.get_edge_feature(x, idx) * edge_weights
        else:
            idx = knn_idx(pos, pos, k=self.knn, offset=1)
            edge_feat = self.get_edge_feature(x, idx)

        y = torch.cat(
            [self.layer_first(edge_feat),
             x.unsqueeze(-2).expand(-1, -1, self.knn, -1)],
            dim=-1,
        )
        for layer in self.layers:
            y = torch.cat([layer(y), y], dim=-1)
        y = self.ace(y)
        y = torch.cat([self.layer_last(y), y], dim=-1)
        return self.aggr(y, dim=-2)
