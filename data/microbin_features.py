"""Six payload-free directional features for fixed-duration micro-bins."""

from __future__ import annotations

import numpy as np


FEATURE_NAMES = (
    "a_to_b_packet_count",
    "b_to_a_packet_count",
    "a_to_b_byte_count",
    "b_to_a_byte_count",
    "a_to_b_mean_packet_length",
    "b_to_a_mean_packet_length",
)


class MicroBinAccumulator:
    def __init__(self, num_bins: int) -> None:
        self.counts = np.zeros((num_bins, 2), dtype=np.float32)
        self.bytes = np.zeros((num_bins, 2), dtype=np.float32)

    def add(self, bin_index: int, direction: int, packet_length: int) -> None:
        self.counts[bin_index, direction] += 1.0
        self.bytes[bin_index, direction] += float(packet_length)

    @property
    def packet_count(self) -> int:
        return int(self.counts.sum())

    @property
    def byte_count(self) -> int:
        return int(self.bytes.sum())

    def finalize(self) -> np.ndarray:
        means = np.divide(
            self.bytes,
            self.counts,
            out=np.zeros_like(self.bytes),
            where=self.counts > 0,
        )
        return np.column_stack(
            [
                self.counts[:, 0],
                self.counts[:, 1],
                self.bytes[:, 0],
                self.bytes[:, 1],
                means[:, 0],
                means[:, 1],
            ]
        ).astype(np.float32, copy=False)
