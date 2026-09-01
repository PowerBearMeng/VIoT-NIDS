"""Payload-free feature profiles for fixed-duration micro-bins."""

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

V4_DIRECTIONAL_IAT_FEATURE_NAMES = (
    "packet_count",
    "byte_count",
    "mean_packet_length",
    "std_packet_length",
    "mean_iat_ms",
    "std_iat_ms",
)


def feature_names(profile: str) -> tuple[str, ...]:
    if profile == "directional_iat_v1":
        return V4_DIRECTIONAL_IAT_FEATURE_NAMES
    if profile == "bidirectional_basic_v1":
        return FEATURE_NAMES
    raise ValueError(f"Unsupported micro-bin feature profile: {profile!r}")


class MicroBinAccumulator:
    def __init__(self, num_bins: int) -> None:
        self.counts = np.zeros((num_bins, 2), dtype=np.float32)
        self.bytes = np.zeros((num_bins, 2), dtype=np.float32)

    def add(
        self,
        bin_index: int,
        direction: int,
        packet_length: int,
        iat_seconds: float | None = None,
    ) -> None:
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


class DirectionalIATAccumulator:
    """Six non-degenerate features for a directional Flow.

    Inter-arrival times are supplied by the Flow builder so they remain
    continuous across adjacent micro-bins and 3-second windows. The first
    packet observed for a directed five-tuple has no fabricated IAT sample.
    """

    def __init__(self, num_bins: int) -> None:
        self.counts = np.zeros(num_bins, dtype=np.float64)
        self.bytes = np.zeros(num_bins, dtype=np.float64)
        self.length_sum = np.zeros(num_bins, dtype=np.float64)
        self.length_square_sum = np.zeros(num_bins, dtype=np.float64)
        self.iat_count = np.zeros(num_bins, dtype=np.float64)
        self.iat_sum_ms = np.zeros(num_bins, dtype=np.float64)
        self.iat_square_sum_ms = np.zeros(num_bins, dtype=np.float64)

    def add(
        self,
        bin_index: int,
        direction: int,
        packet_length: int,
        iat_seconds: float | None = None,
    ) -> None:
        if direction != 0:
            raise ValueError("directional_iat_v1 expects a directional Flow key")
        length = float(packet_length)
        self.counts[bin_index] += 1.0
        self.bytes[bin_index] += length
        self.length_sum[bin_index] += length
        self.length_square_sum[bin_index] += length * length
        if iat_seconds is not None:
            iat_ms = max(0.0, float(iat_seconds) * 1000.0)
            self.iat_count[bin_index] += 1.0
            self.iat_sum_ms[bin_index] += iat_ms
            self.iat_square_sum_ms[bin_index] += iat_ms * iat_ms

    @property
    def packet_count(self) -> int:
        return int(self.counts.sum())

    @property
    def byte_count(self) -> int:
        return int(self.bytes.sum())

    @staticmethod
    def _mean_std(
        count: np.ndarray, total: np.ndarray, square_total: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
        second = np.divide(
            square_total, count, out=np.zeros_like(square_total), where=count > 0
        )
        variance = np.maximum(0.0, second - mean * mean)
        return mean, np.sqrt(variance)

    def finalize(self) -> np.ndarray:
        mean_length, std_length = self._mean_std(
            self.counts, self.length_sum, self.length_square_sum
        )
        mean_iat, std_iat = self._mean_std(
            self.iat_count, self.iat_sum_ms, self.iat_square_sum_ms
        )
        return np.column_stack(
            [
                self.counts,
                self.bytes,
                mean_length,
                std_length,
                mean_iat,
                std_iat,
            ]
        ).astype(np.float32, copy=False)


def make_accumulator(num_bins: int, profile: str) -> MicroBinAccumulator | DirectionalIATAccumulator:
    if profile == "directional_iat_v1":
        return DirectionalIATAccumulator(num_bins)
    if profile == "bidirectional_basic_v1":
        return MicroBinAccumulator(num_bins)
    raise ValueError(f"Unsupported micro-bin feature profile: {profile!r}")
