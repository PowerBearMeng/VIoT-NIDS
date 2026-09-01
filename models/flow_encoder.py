"""Lightweight dilated temporal convolutional flow encoders.

``v1`` is kept intact so that checkpoints produced before Design V2 remain
loadable.  ``v2`` keeps the same six traffic inputs, but removes sample-wise
normalization and replaces temporal mean pooling with learned attention plus
max pooling.  The latter makes short, sparse events visible without adding
hand-crafted protocol features.
"""

from __future__ import annotations

import torch
from torch import nn

from .flow_decoder import FlowDecoder


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.network = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.network(inputs)


class ResidualTCNBlockV2(nn.Module):
    """A residual block that preserves absolute train-scaled magnitudes."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.network = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.network(inputs)


class FlowEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int = 6,
        hidden_channels: int = 48,
        embedding_dim: int = 32,
        blocks: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # One explicit corruption indicator is appended. It is not a traffic
        # feature and is always zero during ordinary embedding extraction.
        self.stem = nn.Conv1d(feature_dim + 1, hidden_channels, kernel_size=1)
        self.blocks = nn.Sequential(
            *[
                ResidualTCNBlock(hidden_channels, kernel_size, 2**index, dropout)
                for index in range(blocks)
            ]
        )
        self.projection = nn.Linear(hidden_channels, embedding_dim)

    def forward(
        self, inputs: torch.Tensor, corruption_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if corruption_mask is None:
            corruption_mask = torch.zeros(inputs.shape[:2], dtype=torch.bool, device=inputs.device)
        indicator = corruption_mask.to(inputs.dtype).unsqueeze(-1)
        values = torch.cat([inputs, indicator], dim=-1).transpose(1, 2)
        hidden = self.blocks(self.stem(values))
        embedding = self.projection(hidden.mean(dim=-1))
        return embedding, hidden


class FlowEncoderV2(nn.Module):
    """Event-sensitive encoder over the unchanged ``[time, 6]`` input."""

    def __init__(
        self,
        feature_dim: int = 6,
        hidden_channels: int = 48,
        embedding_dim: int = 32,
        blocks: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.stem = nn.Conv1d(feature_dim + 1, hidden_channels, kernel_size=1)
        self.blocks = nn.Sequential(
            *[
                ResidualTCNBlockV2(hidden_channels, kernel_size, 2**index, dropout)
                for index in range(blocks)
            ]
        )
        self.attention = nn.Conv1d(hidden_channels, 1, kernel_size=1)
        self.projection = nn.Linear(hidden_channels * 2, embedding_dim)

    def forward(
        self, inputs: torch.Tensor, corruption_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if corruption_mask is None:
            corruption_mask = torch.zeros(inputs.shape[:2], dtype=torch.bool, device=inputs.device)
        indicator = corruption_mask.to(inputs.dtype).unsqueeze(-1)
        values = torch.cat([inputs, indicator], dim=-1).transpose(1, 2)
        hidden = self.blocks(self.stem(values))
        attention = torch.softmax(self.attention(hidden), dim=-1)
        attended = (attention * hidden).sum(dim=-1)
        strongest = hidden.amax(dim=-1)
        embedding = self.projection(torch.cat([attended, strongest], dim=-1))
        return embedding, hidden


class FlowAutoencoder(nn.Module):
    def __init__(self, architecture: str = "v1", **encoder_kwargs: object) -> None:
        super().__init__()
        self.architecture = str(architecture).lower()
        if self.architecture == "v1":
            self.encoder = FlowEncoder(**encoder_kwargs)
        elif self.architecture == "v2":
            self.encoder = FlowEncoderV2(**encoder_kwargs)
        else:
            raise ValueError(f"Unknown flow encoder architecture: {architecture}")
        self.decoder = FlowDecoder(
            hidden_channels=int(encoder_kwargs.get("hidden_channels", 48)),
            feature_dim=int(encoder_kwargs.get("feature_dim", 6)),
        )

    def forward(
        self, inputs: torch.Tensor, corruption_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        masked = inputs.masked_fill(corruption_mask.unsqueeze(-1), 0.0)
        embedding, hidden = self.encoder(masked, corruption_mask)
        return self.decoder(hidden), embedding

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        embedding, _ = self.encoder(inputs, None)
        return embedding
