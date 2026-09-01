"""Training-only feature transforms and calibration-only score scaling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass
class FeatureStandardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray) -> "FeatureStandardizer":
        values = np.log1p(np.asarray(features, dtype=np.float64))
        mean = values.mean(axis=(0, 1))
        scale = values.std(axis=(0, 1))
        scale = np.where(scale < 1e-6, 1.0, scale)
        return cls(mean.astype(np.float32), scale.astype(np.float32))

    def transform(self, features: np.ndarray) -> np.ndarray:
        values = np.log1p(np.asarray(features, dtype=np.float32))
        return ((values - self.mean) / self.scale).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "FeatureStandardizer":
        return cls(np.asarray(state["mean"], dtype=np.float32), np.asarray(state["scale"], dtype=np.float32))


@dataclass
class VectorStandardizer:
    """Ordinary train-only standardization for learned embedding vectors."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "VectorStandardizer":
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or len(array) == 0:
            raise ValueError("VectorStandardizer requires a nonempty 2D array")
        mean = array.mean(axis=0)
        scale = array.std(axis=0)
        scale = np.where(scale < 1e-6, 1.0, scale)
        return cls(mean.astype(np.float32), scale.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        return ((array - self.mean) / self.scale).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "VectorStandardizer":
        return cls(
            np.asarray(state["mean"], dtype=np.float32),
            np.asarray(state["scale"], dtype=np.float32),
        )


@dataclass
class MixedFeatureStandardizer:
    """Log-transform nonnegative count dimensions, preserve signed embeddings."""

    mean: np.ndarray
    scale: np.ndarray
    log_dimensions: int

    @classmethod
    def fit(cls, states: np.ndarray, log_dimensions: int) -> "MixedFeatureStandardizer":
        values = np.asarray(states, dtype=np.float64).copy()
        values[..., :log_dimensions] = np.log1p(np.maximum(values[..., :log_dimensions], 0.0))
        flattened = values.reshape(-1, values.shape[-1])
        mean = flattened.mean(axis=0)
        scale = flattened.std(axis=0)
        scale = np.where(scale < 1e-6, 1.0, scale)
        return cls(mean.astype(np.float32), scale.astype(np.float32), int(log_dimensions))

    def transform(self, states: np.ndarray) -> np.ndarray:
        values = np.asarray(states, dtype=np.float32).copy()
        values[..., : self.log_dimensions] = np.log1p(
            np.maximum(values[..., : self.log_dimensions], 0.0)
        )
        return ((values - self.mean) / self.scale).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "log_dimensions": self.log_dimensions,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "MixedFeatureStandardizer":
        return cls(
            np.asarray(state["mean"], dtype=np.float32),
            np.asarray(state["scale"], dtype=np.float32),
            int(state["log_dimensions"]),
        )


@dataclass
class QuantileScoreScaler:
    low: float
    high: float

    @classmethod
    def fit(
        cls, values: Iterable[float] | np.ndarray, low_quantile: float, high_quantile: float
    ) -> "QuantileScoreScaler":
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0 or not np.all(np.isfinite(array)):
            raise ValueError("Score scaler requires finite calibration values")
        low = float(np.quantile(array, low_quantile))
        high = float(np.quantile(array, high_quantile))
        if high - low < 1e-12:
            high = low + max(abs(low) * 1e-6, 1e-6)
        return cls(low, high)

    def transform(self, values: Iterable[float] | np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        # Do not upper-clip: unusually large attacks must remain distinguishable.
        return np.maximum(0.0, (array - self.low) / (self.high - self.low))

    def state_dict(self) -> dict[str, float]:
        return {"low": self.low, "high": self.high}

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "QuantileScoreScaler":
        return cls(float(state["low"]), float(state["high"]))


@dataclass
class EmpiricalUpperTail:
    """Normal-only conformal upper-tail evidence for arbitrary raw scores."""

    sorted_values: np.ndarray
    epsilon: float = 1e-12

    @classmethod
    def fit(cls, values: Iterable[float] | np.ndarray, epsilon: float = 1e-12) -> "EmpiricalUpperTail":
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0 or not np.all(np.isfinite(array)):
            raise ValueError("Empirical tail reference requires finite normal scores")
        return cls(np.sort(array), float(epsilon))

    def probabilities(self, values: Iterable[float] | np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        # Plus-one smoothing prevents a zero probability outside the finite
        # calibration sample while preserving upper-tail rank ordering.
        lower = np.searchsorted(self.sorted_values, array, side="left")
        count_ge = len(self.sorted_values) - lower
        return (count_ge + 1.0) / (len(self.sorted_values) + 1.0)

    def evidence(self, values: Iterable[float] | np.ndarray) -> np.ndarray:
        return -np.log(self.probabilities(values) + self.epsilon)

    def state_dict(self) -> dict[str, Any]:
        return {"sorted_values": self.sorted_values.tolist(), "epsilon": self.epsilon}

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "EmpiricalUpperTail":
        return cls(
            np.asarray(state["sorted_values"], dtype=np.float64),
            float(state.get("epsilon", 1e-12)),
        )
