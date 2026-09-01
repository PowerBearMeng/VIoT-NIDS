"""MLP predicting an edge embedding from its endpoint contexts."""

from __future__ import annotations

import torch
from torch import nn


class SpatialContextPredictor(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, endpoint_contexts: torch.Tensor) -> torch.Tensor:
        return self.network(endpoint_contexts)
