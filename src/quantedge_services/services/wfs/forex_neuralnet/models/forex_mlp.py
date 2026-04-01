"""ForexMLP — 4-layer MLP for Forex regression."""
from __future__ import annotations

import torch
import torch.nn as nn


class ForexMLP(nn.Module):
    """4-layer MLP: Linear→BatchNorm→ReLU→Dropout, repeated hidden_sizes times, then output FC."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int = 1,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_size
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
