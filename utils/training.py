"""Small common PyTorch training utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn


@dataclass
class EarlyStopping:
    patience: int
    best_loss: float = float("inf")
    best_epoch: int = -1
    bad_epochs: int = 0
    best_state: dict[str, torch.Tensor] | None = None

    def update(self, loss: float, epoch: int, model: nn.Module) -> bool:
        if loss < self.best_loss - 1e-9:
            self.best_loss = float(loss)
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
            self.best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def mean_batch_loss(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_fn: Callable[[tuple[torch.Tensor, ...]], torch.Tensor],
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for raw_batch in loader:
            batch = tuple(item.to(device) for item in raw_batch)
            loss = loss_fn(batch)
            batch_size = int(batch[0].shape[0])
            total += float(loss.item()) * batch_size
            count += batch_size
    return total / max(1, count)
