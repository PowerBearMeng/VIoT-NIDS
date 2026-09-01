"""Per-micro-bin reconstruction head for the TCN encoder."""

from __future__ import annotations

import torch
from torch import nn


class FlowDecoder(nn.Module):
    def __init__(self, hidden_channels: int, feature_dim: int = 6) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, feature_dim, kernel_size=1),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.network(hidden).transpose(1, 2)
