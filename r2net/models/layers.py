"""Small building blocks for the dense edge-convolution stream."""
from __future__ import annotations

import torch
import torch.nn as nn


class FCLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 activation: str | None = "relu"):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if activation is None:
            self.activation = nn.Identity()
        elif activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "elu":
            self.activation = nn.ELU(alpha=1.0)
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.1, inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x))


class FCLayerFirst(nn.Module):
    """Two-layer MLP used as the first dense-block transform."""

    def __init__(self, in_features: int, hidden_features: int, out_features: int,
                 bias: bool = True):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.linear2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.activation(self.linear1(x)))


class Aggregator(nn.Module):
    def __init__(self, oper: str = "max"):
        super().__init__()
        if oper not in ("mean", "sum", "max"):
            raise ValueError(oper)
        self.oper = oper

    def forward(self, x: torch.Tensor, dim: int = 2) -> torch.Tensor:
        if self.oper == "mean":
            return x.mean(dim=dim)
        if self.oper == "sum":
            return x.sum(dim=dim)
        return x.max(dim=dim).values
