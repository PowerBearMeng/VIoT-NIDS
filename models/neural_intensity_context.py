"""Target-conditioned continuous normal-intensity model for Design V4."""

from __future__ import annotations

import torch
from torch import nn


class NeuralIntensityContext(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 64,
        latent_channels: int = 16,
        assignment_temperature: float = 0.5,
        min_log_scale: float = -3.0,
        max_log_scale: float = 2.0,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.latent_channels = int(latent_channels)
        self.assignment_temperature = float(assignment_temperature)
        self.min_log_scale = float(min_log_scale)
        self.max_log_scale = float(max_log_scale)
        self.behavior_gate = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.latent_channels),
        )
        self.pair_head = self._head(hidden_dim)
        self.entity_head = self._head(hidden_dim)

    def _head(self, hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def assignments(self, embeddings: torch.Tensor) -> torch.Tensor:
        logits = self.behavior_gate(embeddings)
        return torch.softmax(logits / self.assignment_temperature, dim=-1)

    def expected_parameters(
        self, embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pair = self.pair_head(embeddings)
        entity = self.entity_head(embeddings)
        pair_log_scale = pair[:, 1].clamp(self.min_log_scale, self.max_log_scale)
        entity_log_scale = entity[:, 1].clamp(
            self.min_log_scale, self.max_log_scale
        )
        return pair[:, 0], pair_log_scale, entity[:, 0], entity_log_scale

    @staticmethod
    def gaussian_nll(
        observed: torch.Tensor, mean: torch.Tensor, log_scale: torch.Tensor
    ) -> torch.Tensor:
        standardized = (observed - mean) * torch.exp(-log_scale)
        return 0.5 * standardized.square() + log_scale

    @staticmethod
    def excess_energy(
        observed: torch.Tensor, mean: torch.Tensor, log_scale: torch.Tensor
    ) -> torch.Tensor:
        """One-sided learned intensity energy; absent traffic is not anomalous."""

        excess = torch.relu((observed - mean) * torch.exp(-log_scale))
        return 0.5 * excess.square()
