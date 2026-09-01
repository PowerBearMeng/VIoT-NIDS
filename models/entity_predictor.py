"""GRU predictor for entity-level temporal normality."""

from __future__ import annotations

import torch
from torch import nn


class EntityGRUPredictor(nn.Module):
    def __init__(
        self, state_dim: int, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            state_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, state_dim))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(history)
        return self.output(hidden[-1])
