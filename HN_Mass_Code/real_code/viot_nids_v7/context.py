"""Fixed behavior observation and strictly causal history construction."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler


@dataclass
class Scale:
    center: np.ndarray
    scale: np.ndarray
    log: bool = True

    @classmethod
    def fit(cls, values, floor=0.05, log=True):
        values = np.asarray(values, dtype=np.float64)
        transformed = np.log1p(values) if log else values
        fitted = RobustScaler().fit(transformed)
        return cls(fitted.center_, np.maximum(fitted.scale_, floor), log)

    def transform(self, values):
        values = np.asarray(values, dtype=np.float64)
        out = ((np.log1p(values) if self.log else values) - self.center) / self.scale
        if not np.isfinite(out).all():
            raise ValueError("Nonfinite transformed observations")
        return out.astype(np.float32)


@dataclass
class Behavior:
    scaler: Scale
    centers: np.ndarray
    widths: np.ndarray
    support: np.ndarray

    @classmethod
    def fit(cls, embedding, seconds, k=16, seed=42):
        scaler = Scale.fit(embedding, log=False)
        z = scaler.transform(embedding).astype(np.float64)
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(z)
        distances = np.linalg.norm(z - km.cluster_centers_[km.labels_], axis=1)
        floor = max(1e-3, 0.1 * float(np.median(distances)))
        widths = np.array([max(floor, float(np.median(distances[km.labels_ == i])))
                           if (km.labels_ == i).any() else floor for i in range(k)])
        blocks = (seconds - seconds.min()) // 10
        support = np.array([len(np.unique(blocks[km.labels_ == i])) for i in range(k)])
        return cls(scaler, km.cluster_centers_, widths, support)

    def transform(self, embedding):
        z = self.scaler.transform(embedding).astype(np.float64)
        squared = ((z[:, None] - self.centers[None]) ** 2).sum(2)
        log_response = -squared / (2 * self.widths[None] ** 2)
        log_response -= log_response.max(1, keepdims=True)
        response = np.exp(log_response)
        return response / np.linalg.norm(response, axis=1, keepdims=True)


def aggregate(phi, packets, byte_count, first_bin):
    k = phi.shape[1]
    u = np.zeros((10, k), dtype=np.float64)
    np.add.at(u, first_bin, phi)
    v = packets.T.astype(np.float64) @ phi
    b = byte_count.T.astype(np.float64) @ phi
    return np.stack([u, v, b], axis=0)  # channels, bin, behavior


def query(phi, totals, own=None):
    values = np.einsum("nk,cbk->nbc", phi, totals, optimize=True)
    if own is not None:
        packets, byte_count, first_bin = own
        values[np.arange(len(phi)), first_bin, 0] -= 1
        values[:, :, 1] -= packets
        values[:, :, 2] -= byte_count
    # Roundoff budgets must respect channel units: large byte totals must never
    # permit a material negative Flow count to be silently repaired.
    tolerance = 1e-10 * np.maximum(1.0, np.max(np.abs(totals), axis=(1, 2)))
    if (np.min(values, axis=(0, 1), initial=0) < -tolerance).any():
        raise ValueError("Context has materially negative self-excluded contribution")
    return np.maximum(values, 0).reshape(len(phi), 30)


def summary(totals, n_flows, packets, byte_count, first_bin):
    return np.r_[totals.sum(1).reshape(-1),
                 n_flows, packets.sum(), byte_count.sum(),
                 np.bincount(first_bin, minlength=10),
                 packets.sum(0), byte_count.sum(0)].astype(np.float64)


class History:
    def __init__(self, length=16, taus=(4, 16, 64)):
        self.length = length
        self.taus = np.asarray(taus, dtype=np.float64)
        self.recent = deque(maxlen=length)
        self.activity = np.zeros((len(taus), 48), dtype=np.float64)
        self.exposure = np.zeros(len(taus), dtype=np.float64)
        self.observed = 0

    def before(self):
        values = np.zeros((self.length, 81), dtype=np.float64)
        n = len(self.recent)
        if n:
            values[:n] = np.stack(self.recent)
        auxiliary = np.c_[self.activity, self.exposure].reshape(-1)
        return values, n, auxiliary

    def update(self, current):
        current = np.asarray(current, dtype=np.float64)
        self.activity *= np.exp(-1 / self.taus)[:, None]
        self.activity += current[:48][None]
        self.exposure *= np.exp(-1 / self.taus)
        self.exposure += 1
        self.recent.append(current.copy())
        self.observed += 1


def build_normal(data, phi, cfg):
    seconds = data["timestamp_second"]
    origin = int(seconds.min())
    span = int(seconds.max()) - origin + 1
    row_window = (seconds - origin).astype(np.int64)
    members = [np.flatnonzero(row_window == i) for i in range(span)]
    contexts = np.zeros((len(phi), 30), dtype=np.float64)
    totals = np.zeros((span, 3, 10, phi.shape[1]), dtype=np.float64)
    summaries = np.zeros((span, 81), dtype=np.float64)
    for w, ix in enumerate(members):
        if not len(ix):
            continue
        p, b, first = data["bin_packets"][ix], data["bin_bytes"][ix], data["first_bin"][ix]
        totals[w] = aggregate(phi[ix], p, b, first)
        contexts[ix] = query(phi[ix], totals[w], (p, b, first))
        summaries[w] = summary(totals[w], len(ix), p, b, first)
    history = History(cfg["history_seconds"], cfg["decay_seconds"])
    sequences, lengths, auxiliary = [], [], []
    for values in summaries:
        sequence, length, extra = history.before()
        sequences.append(sequence)
        lengths.append(length)
        auxiliary.append(extra)
        history.update(values)
    sequence_scale = Scale.fit(summaries)
    extra_scale = Scale.fit(auxiliary)
    sequences = sequence_scale.transform(sequences)
    lengths = np.asarray(lengths, dtype=np.int64)
    valid = np.arange(cfg["history_seconds"])[None] < lengths[:, None]
    sequences[~valid] = 0
    return dict(raw=contexts, row_window=row_window, members=members, totals=totals,
                sequences=sequences, lengths=lengths,
                auxiliary=extra_scale.transform(auxiliary),
                context_scale=Scale.fit(contexts), sequence_scale=sequence_scale,
                extra_scale=extra_scale, summaries=summaries)
