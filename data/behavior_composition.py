"""Frozen normal behavior-composition profiles for Design V3.

The only categorical input is the normal prototype assignment (``mode_id``).
Profiles are keyed by IP entity or unordered IP pair.  Ports are deliberately
absent from both the reference keys and every anomaly calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import FlowDataset


PairKey = tuple[str, str]


def pair_key(endpoint_a: str, endpoint_b: str) -> PairKey:
    return tuple(sorted((str(endpoint_a), str(endpoint_b))))  # type: ignore[return-value]


@dataclass
class BehaviorCompositionReference:
    prototype_count: int
    use_log_count: bool
    epsilon: float
    pair_mean: dict[PairKey, np.ndarray]
    pair_std: dict[PairKey, np.ndarray]
    pair_samples: dict[PairKey, int]
    entity_mean: dict[str, np.ndarray]
    entity_std: dict[str, np.ndarray]
    entity_samples: dict[str, int]

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        pairs = sorted(self.pair_mean)
        entities = sorted(self.entity_mean)
        np.savez_compressed(
            output,
            format_version=np.asarray(1, dtype=np.int64),
            prototype_count=np.asarray(self.prototype_count, dtype=np.int64),
            use_log_count=np.asarray(int(self.use_log_count), dtype=np.int8),
            epsilon=np.asarray(self.epsilon, dtype=np.float64),
            pair_endpoint_a=np.asarray([key[0] for key in pairs]),
            pair_endpoint_b=np.asarray([key[1] for key in pairs]),
            pair_mean=np.stack([self.pair_mean[key] for key in pairs]).astype(np.float32),
            pair_std=np.stack([self.pair_std[key] for key in pairs]).astype(np.float32),
            pair_samples=np.asarray([self.pair_samples[key] for key in pairs], dtype=np.int64),
            entity_keys=np.asarray(entities),
            entity_mean=np.stack([self.entity_mean[key] for key in entities]).astype(np.float32),
            entity_std=np.stack([self.entity_std[key] for key in entities]).astype(np.float32),
            entity_samples=np.asarray([self.entity_samples[key] for key in entities], dtype=np.int64),
        )
        return output

    @classmethod
    def load(cls, path: str | Path) -> "BehaviorCompositionReference":
        with np.load(Path(path), allow_pickle=False) as archive:
            pair_a = archive["pair_endpoint_a"].astype(str).tolist()
            pair_b = archive["pair_endpoint_b"].astype(str).tolist()
            pair_keys = list(zip(pair_a, pair_b))
            entity_keys = archive["entity_keys"].astype(str).tolist()
            return cls(
                prototype_count=int(archive["prototype_count"]),
                use_log_count=bool(int(archive["use_log_count"])),
                epsilon=float(archive["epsilon"]),
                pair_mean={key: value for key, value in zip(pair_keys, archive["pair_mean"])},
                pair_std={key: value for key, value in zip(pair_keys, archive["pair_std"])},
                pair_samples={key: int(value) for key, value in zip(pair_keys, archive["pair_samples"])},
                entity_mean={key: value for key, value in zip(entity_keys, archive["entity_mean"])},
                entity_std={key: value for key, value in zip(entity_keys, archive["entity_std"])},
                entity_samples={key: int(value) for key, value in zip(entity_keys, archive["entity_samples"])},
            )


@dataclass
class BehaviorCompositionScores:
    pair_deviation: np.ndarray
    entity_a_deviation: np.ndarray
    entity_b_deviation: np.ndarray
    entity_deviation: np.ndarray
    context_deviation: np.ndarray
    pair_mode_count: np.ndarray
    entity_a_mode_count: np.ndarray
    entity_b_mode_count: np.ndarray


def _window_rows(dataset: FlowDataset, indices: np.ndarray) -> list[list[int]]:
    groups: dict[tuple[str, int], list[int]] = {}
    for local_row, raw_index in enumerate(indices.tolist()):
        key = (str(dataset.capture_ids[raw_index]), int(dataset.window_indices[raw_index]))
        groups.setdefault(key, []).append(local_row)
    return [rows for _, rows in sorted(groups.items(), key=lambda item: item[0])]


def _window_profiles(
    dataset: FlowDataset,
    mode_ids: np.ndarray,
    indices: np.ndarray,
    rows: list[int],
    prototype_count: int,
) -> tuple[dict[PairKey, np.ndarray], dict[str, np.ndarray]]:
    pair_counts: dict[PairKey, np.ndarray] = {}
    entity_counts: dict[str, np.ndarray] = {}
    for local_row in rows:
        raw_index = int(indices[local_row])
        mode = int(mode_ids[raw_index])
        if mode < 0 or mode >= prototype_count:
            raise ValueError(f"Invalid mode_id {mode}; expected [0,{prototype_count})")
        endpoint_a = str(dataset.endpoint_a_ips[raw_index])
        endpoint_b = str(dataset.endpoint_b_ips[raw_index])
        pair = pair_key(endpoint_a, endpoint_b)
        pair_counts.setdefault(pair, np.zeros(prototype_count, dtype=np.float64))[mode] += 1.0
        for entity in dict.fromkeys((endpoint_a, endpoint_b)):
            entity_counts.setdefault(entity, np.zeros(prototype_count, dtype=np.float64))[mode] += 1.0
    return pair_counts, entity_counts


def _fit_scope(
    observations: dict[Any, list[np.ndarray]],
    *,
    use_log_count: bool,
) -> tuple[dict[Any, np.ndarray], dict[Any, np.ndarray], dict[Any, int]]:
    means: dict[Any, np.ndarray] = {}
    stds: dict[Any, np.ndarray] = {}
    samples: dict[Any, int] = {}
    for key, rows in observations.items():
        values = np.stack(rows).astype(np.float64)
        if use_log_count:
            values = np.log1p(values)
        means[key] = values.mean(axis=0).astype(np.float32)
        stds[key] = values.std(axis=0).astype(np.float32)
        samples[key] = int(len(values))
    return means, stds, samples


def fit_behavior_reference(
    dataset: FlowDataset,
    mode_ids: np.ndarray,
    indices: np.ndarray,
    *,
    prototype_count: int,
    use_log_count: bool,
    epsilon: float,
) -> BehaviorCompositionReference:
    """Fit each known scope/mode over the complete normal-train window set.

    A known scope that is absent from a normal window contributes a zero-count
    vector.  A scope never seen during normal training has no reference and is
    explicitly neutral at inference.
    """

    indices = np.asarray(indices, dtype=np.int64)
    window_pair_counts: list[dict[PairKey, np.ndarray]] = []
    window_entity_counts: list[dict[str, np.ndarray]] = []
    for rows in _window_rows(dataset, indices):
        pair_counts, entity_counts = _window_profiles(
            dataset, mode_ids, indices, rows, prototype_count
        )
        window_pair_counts.append(pair_counts)
        window_entity_counts.append(entity_counts)
    zero = np.zeros(prototype_count, dtype=np.float64)
    pair_keys = set().union(*(counts.keys() for counts in window_pair_counts))
    entity_keys = set().union(*(counts.keys() for counts in window_entity_counts))
    pair_observations = {
        key: [counts.get(key, zero) for counts in window_pair_counts]
        for key in pair_keys
    }
    entity_observations = {
        key: [counts.get(key, zero) for counts in window_entity_counts]
        for key in entity_keys
    }
    pair_mean, pair_std, pair_samples = _fit_scope(
        pair_observations, use_log_count=use_log_count
    )
    entity_mean, entity_std, entity_samples = _fit_scope(
        entity_observations, use_log_count=use_log_count
    )
    return BehaviorCompositionReference(
        prototype_count=int(prototype_count),
        use_log_count=bool(use_log_count),
        epsilon=float(epsilon),
        pair_mean=pair_mean,
        pair_std=pair_std,
        pair_samples=pair_samples,
        entity_mean=entity_mean,
        entity_std=entity_std,
        entity_samples=entity_samples,
    )


def _positive_deviation(
    counts: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    use_log_count: bool,
    epsilon: float,
    positive_only: bool,
) -> np.ndarray:
    values = np.log1p(counts) if use_log_count else counts
    deviation = (values - mean) / (std + float(epsilon))
    return np.maximum(deviation, 0.0) if positive_only else np.abs(deviation)


def score_behavior_composition(
    dataset: FlowDataset,
    mode_ids: np.ndarray,
    indices: np.ndarray,
    reference: BehaviorCompositionReference,
    *,
    pair_enabled: bool,
    entity_enabled: bool,
    positive_deviation_only: bool,
) -> BehaviorCompositionScores:
    """Score each flow only against the deviation of its own behavior mode."""

    indices = np.asarray(indices, dtype=np.int64)
    size = len(indices)
    pair_score = np.zeros(size, dtype=np.float64)
    entity_a_score = np.zeros(size, dtype=np.float64)
    entity_b_score = np.zeros(size, dtype=np.float64)
    pair_count_out = np.zeros(size, dtype=np.int64)
    entity_a_count_out = np.zeros(size, dtype=np.int64)
    entity_b_count_out = np.zeros(size, dtype=np.int64)

    for rows in _window_rows(dataset, indices):
        pair_counts, entity_counts = _window_profiles(
            dataset, mode_ids, indices, rows, reference.prototype_count
        )
        pair_deviations: dict[PairKey, np.ndarray] = {}
        entity_deviations: dict[str, np.ndarray] = {}
        if pair_enabled:
            for key, counts in pair_counts.items():
                # An unseen pair is explicitly neutral; no novelty/rarity rule.
                if key not in reference.pair_mean:
                    pair_deviations[key] = np.zeros(reference.prototype_count, dtype=np.float64)
                else:
                    pair_deviations[key] = _positive_deviation(
                        counts,
                        reference.pair_mean[key],
                        reference.pair_std[key],
                        use_log_count=reference.use_log_count,
                        epsilon=reference.epsilon,
                        positive_only=positive_deviation_only,
                    )
        if entity_enabled:
            for key, counts in entity_counts.items():
                # Keep unseen entities neutral for the same reason as pairs.
                if key not in reference.entity_mean:
                    entity_deviations[key] = np.zeros(reference.prototype_count, dtype=np.float64)
                else:
                    entity_deviations[key] = _positive_deviation(
                        counts,
                        reference.entity_mean[key],
                        reference.entity_std[key],
                        use_log_count=reference.use_log_count,
                        epsilon=reference.epsilon,
                        positive_only=positive_deviation_only,
                    )
        for local_row in rows:
            raw_index = int(indices[local_row])
            mode = int(mode_ids[raw_index])
            endpoint_a = str(dataset.endpoint_a_ips[raw_index])
            endpoint_b = str(dataset.endpoint_b_ips[raw_index])
            pair = pair_key(endpoint_a, endpoint_b)
            pair_count_out[local_row] = int(pair_counts[pair][mode])
            entity_a_count_out[local_row] = int(entity_counts[endpoint_a][mode])
            entity_b_count_out[local_row] = int(entity_counts[endpoint_b][mode])
            if pair_enabled:
                pair_score[local_row] = float(pair_deviations[pair][mode])
            if entity_enabled:
                entity_a_score[local_row] = float(entity_deviations[endpoint_a][mode])
                entity_b_score[local_row] = float(entity_deviations[endpoint_b][mode])

    entity_score = np.maximum(entity_a_score, entity_b_score)
    context_score = np.maximum(pair_score, entity_score)
    return BehaviorCompositionScores(
        pair_deviation=pair_score,
        entity_a_deviation=entity_a_score,
        entity_b_deviation=entity_b_score,
        entity_deviation=entity_score,
        context_deviation=context_score,
        pair_mode_count=pair_count_out,
        entity_a_mode_count=entity_a_count_out,
        entity_b_mode_count=entity_b_count_out,
    )
