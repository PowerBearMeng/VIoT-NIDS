"""V4.5 target-conditioned single-head sparse context model."""

from __future__ import annotations

import torch
from torch import nn


class SparseContextModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 64,
        attention_dim: int = 32,
        min_log_scale: float = -3.0,
        max_log_scale: float = 2.0,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.attention_dim = int(attention_dim)
        self.min_log_scale = float(min_log_scale)
        self.max_log_scale = float(max_log_scale)

        # Pair and entity relations have separate single heads.  IP identity is
        # absent; all parameters are shared across every transient scope.
        self.pair_query = nn.Linear(self.embedding_dim, self.attention_dim, bias=False)
        self.pair_key = nn.Linear(self.embedding_dim, self.attention_dim, bias=False)
        self.entity_query = nn.Linear(self.embedding_dim, self.attention_dim, bias=False)
        self.entity_key = nn.Linear(self.embedding_dim, self.attention_dim, bias=False)
        self.pair_null_head = self._scalar_head(hidden_dim)
        self.entity_null_head = self._scalar_head(hidden_dim)
        self.pair_distribution_head = self._distribution_head(hidden_dim)
        self.entity_distribution_head = self._distribution_head(hidden_dim)

    def _scalar_head(self, hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def _distribution_head(self, hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def attention_parameters(
        self, embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.pair_query(embeddings),
            self.pair_key(embeddings),
            self.pair_null_head(embeddings).squeeze(-1),
            self.entity_query(embeddings),
            self.entity_key(embeddings),
            self.entity_null_head(embeddings).squeeze(-1),
        )

    def expected_parameters(
        self, embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pair = self.pair_distribution_head(embeddings)
        entity = self.entity_distribution_head(embeddings)
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
