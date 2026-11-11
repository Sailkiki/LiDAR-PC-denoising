"""Feature extractor: stacked OC-Conv blocks with ACE calibration."""
from __future__ import annotations

import torch
import torch.nn as nn

from .edgeconv import DenseEdgeConv
from .layers import FCLayer


class FeatureExtraction(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        conv_channels: int = 24,
        num_convs: int = 4,
        num_fc_layers: int = 3,
        growth_rate: int = 12,
        knn: int = 16,
        aggr: str = "max",
        activation: str = "relu",
        use_oc: bool = True,
        use_ace: bool = True,
        candidate_ratio: float = 1.5,
        ace_reduction: int = 8,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_convs = int(num_convs)
        self.transforms = nn.ModuleList()
        self.convs = nn.ModuleList()
        current = int(in_channels)
        for i in range(self.num_convs):
            trans = FCLayer(
                current, conv_channels, bias=True,
                activation=None if i == 0 else activation,
            )
            conv = DenseEdgeConv(
                conv_channels,
                num_fc_layers=num_fc_layers,
                growth_rate=growth_rate,
                knn=knn,
                aggr=aggr,
                activation=activation,
                use_oc=use_oc,
                use_ace=use_ace,
                candidate_ratio=candidate_ratio,
                ace_reduction=ace_reduction,
            )
            self.transforms.append(trans)
            self.convs.append(conv)
            current = conv.out_channels

    @property
    def out_channels(self) -> int:
        return self.convs[-1].out_channels

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        # Translation invariance: each patch is centred before graph kNN.
        pos = pos - pos.mean(dim=1, keepdim=True)
        x = pos
        for trans, conv in zip(self.transforms, self.convs):
            x = trans(x)
            x = conv(x, pos)
        return x
