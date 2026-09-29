"""Single masked reconstruction objective with shared temporal memory."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class FlowMemory(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(25, 64), nn.LayerNorm(64),
                                     nn.GELU(), nn.Linear(64, 32))
        self.time_embedding = nn.Linear(1, 8)
        self.gru = nn.GRUCell(40, 32)
        self.decoder = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 25))

    def step(self, x: torch.Tensor, delta: torch.Tensor,
             hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(x)
        t = self.time_embedding(torch.log1p(delta).unsqueeze(-1))
        h = self.gru(torch.cat([embedding, t], dim=-1), hidden)
        return torch.cat([embedding, h], dim=-1), h


def channel_key(row: object, fields: list[str]) -> tuple[object, ...]:
    return tuple(getattr(row, field) for field in fields)


def channel_indices(frame: object, fields: list[str]) -> list[np.ndarray]:
    groups = frame.groupby(fields, sort=False).indices
    # read_flows has already sorted by (second, row_id); pandas indices retain
    # that order within every channel. Avoid per-row iloc calls on large scans.
    return [np.asarray(ix, dtype=np.int64) for ix in groups.values()]


class ChannelBalancedSampler:
    """Uniform channel, then uniform start index within that channel."""

    def __init__(self, channels: list[np.ndarray], seed: int = 42) -> None:
        self.channels = channels
        self.rng = np.random.default_rng(seed)

    def draw(self, batch_size: int, steps: int) -> np.ndarray:
        chosen = self.rng.integers(0, len(self.channels), size=batch_size)
        result = np.full((batch_size, steps), -1, dtype=np.int64)
        for j, channel_id in enumerate(chosen):
            indices = self.channels[channel_id]
            start = int(self.rng.integers(0, len(indices)))
            take = min(steps, len(indices) - start)
            result[j, :take] = indices[start:start + take]
        return result


def train_encoder(model: FlowMemory, x: np.ndarray, seconds: np.ndarray,
                  channels: list[np.ndarray], *, epochs: int, steps_per_epoch: int,
                  batch_size: int, sequence_length: int, learning_rate: float,
                  mask_ratio: float, seed: int, device: str) -> list[float]:
    torch.manual_seed(seed)
    sampler = ChannelBalancedSampler(channels, seed)
    model.to(device).train()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses: list[float] = []
    for epoch in range(epochs):
        epoch_losses = []
        for _ in range(steps_per_epoch):
            indices = sampler.draw(batch_size, sequence_length)
            valid = indices >= 0
            safe = np.maximum(indices, 0)
            values = torch.as_tensor(x[safe], dtype=torch.float32, device=device)
            # Subtract as int64 before float conversion: Unix epochs lose 1 s
            # precision in float32.
            offsets = seconds[safe] - seconds[safe][:, :1]
            timestamps = torch.as_tensor(offsets, dtype=torch.float32, device=device)
            hidden = torch.zeros(batch_size, 32, device=device)
            outputs = []
            for step in range(sequence_length):
                mask = torch.rand_like(values[:, step]) < mask_ratio
                masked = values[:, step].masked_fill(mask, 0)
                if step == 0:
                    delta = torch.zeros(batch_size, device=device)
                else:
                    delta = (timestamps[:, step] - timestamps[:, step - 1]).clamp_min(0)
                z, updated = model.step(masked, delta, hidden)
                hidden = torch.where(torch.as_tensor(valid[:, step], device=device).unsqueeze(1), updated, hidden)
                reconstruction = model.decoder(z)
                selected = mask & torch.as_tensor(valid[:, step], device=device).unsqueeze(1)
                if selected.any():
                    outputs.append(F.huber_loss(reconstruction[selected], values[:, step][selected]))
            loss = torch.stack(outputs).mean()
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
    model.eval()
    return losses


@torch.no_grad()
def embed(model: FlowMemory, x: np.ndarray, seconds: np.ndarray,
          channels: list[np.ndarray], device: str) -> np.ndarray:
    model.to(device).eval()
    result = np.empty((len(x), 64), dtype=np.float32)
    # Several hundred thousand five-tuples can collapse into one channel when
    # a flood changes source ports. nn.GRU uses the identical GRUCell weights
    # but executes that long recurrence in its optimized sequence kernel.
    long_channels = [indices for indices in channels if len(indices) > 1024]
    short_channels = [indices for indices in channels if len(indices) <= 1024]
    if long_channels:
        gru = nn.GRU(40, 32).to(device).eval()
        with torch.no_grad():
            gru.weight_ih_l0.copy_(model.gru.weight_ih)
            gru.weight_hh_l0.copy_(model.gru.weight_hh)
            gru.bias_ih_l0.copy_(model.gru.bias_ih)
            gru.bias_hh_l0.copy_(model.gru.bias_hh)
        for indices in long_channels:
            times = seconds[indices]
            delta = np.r_[0, np.maximum(0, np.diff(times))]
            hidden_sequence = torch.zeros(1, 1, 32, device=device)
            for start in range(0, len(indices), 4096):
                stop = min(start + 4096, len(indices))
                ids = indices[start:stop]
                values = torch.as_tensor(x[ids], dtype=torch.float32, device=device)
                dt = torch.as_tensor(delta[start:stop], dtype=torch.float32, device=device)
                e = model.encoder(values)
                t = model.time_embedding(torch.log1p(dt).unsqueeze(-1))
                h, hidden_sequence = gru(torch.cat([e, t], dim=-1).unsqueeze(1), hidden_sequence)
                result[ids] = torch.cat([e, h[:, 0]], dim=-1).cpu().numpy()
    hidden = torch.zeros(len(short_channels), 32, device=device)
    previous = np.full(len(short_channels), -1, dtype=np.int64)
    lengths = np.asarray([len(a) for a in short_channels])
    for position in range(int(lengths.max(initial=0))):
        active = np.flatnonzero(lengths > position)
        for subset in np.array_split(active, max(1, (len(active) + 2047) // 2048)):
            ids = np.asarray([short_channels[c][position] for c in subset], dtype=np.int64)
            now = seconds[ids]
            delta = np.where(previous[subset] < 0, 0, np.maximum(0, now - previous[subset]))
            values = torch.as_tensor(x[ids], dtype=torch.float32, device=device)
            dt = torch.as_tensor(delta, dtype=torch.float32, device=device)
            z, updated = model.step(values, dt, hidden[subset])
            hidden[subset] = updated
            result[ids] = z.cpu().numpy()
            previous[subset] = now
    return result
