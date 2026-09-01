"""Spatial context construction for the original and historical V2 modes."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import Any

import numpy as np

from .dataset import FlowDataset


@dataclass
class SpatialSamples:
    indices: np.ndarray
    contexts: np.ndarray
    targets: np.ndarray
    reliability: np.ndarray
    endpoint_context_counts: np.ndarray

    def __len__(self) -> int:
        return int(self.indices.shape[0])


def initial_historical_state(global_mean: np.ndarray) -> dict[str, Any]:
    """Create a serializable state containing no endpoint or pair novelty rule."""

    return {
        "format_version": 1,
        "global_mean": np.asarray(global_mean, dtype=np.float32).tolist(),
        "entities": {},
        "pairs": {},
    }


def _pair_key(endpoint_a: str, endpoint_b: str) -> str:
    # Ports are deliberately absent.  Length prefixes make IPv4/IPv6 keys
    # unambiguous without imposing any application semantics.
    first, second = sorted((endpoint_a, endpoint_b))
    return f"{len(first)}:{first}{second}"


def _history_mean(entry: dict[str, Any] | None, fallback: np.ndarray) -> np.ndarray:
    if entry is None:
        return fallback
    return np.asarray(entry["mean"], dtype=np.float32)


def _ema_entry(
    table: dict[str, dict[str, Any]],
    key: str,
    value: np.ndarray,
    rate: float,
    fallback: np.ndarray,
) -> None:
    rate = float(np.clip(rate, 0.0, 1.0))
    existing = table.get(key)
    if existing is None:
        initial = (1.0 - rate) * fallback + rate * value
        table[key] = {"mean": np.asarray(initial, dtype=np.float32).tolist(), "observations": 1}
        return
    previous = np.asarray(existing["mean"], dtype=np.float32)
    existing["mean"] = ((1.0 - rate) * previous + rate * value).astype(np.float32).tolist()
    existing["observations"] = int(existing.get("observations", 0)) + 1


def build_historical_spatial_samples(
    dataset: FlowDataset,
    embeddings: np.ndarray,
    local_score: np.ndarray,
    indices: np.ndarray,
    *,
    initial_state: dict[str, Any],
    alpha: float,
    min_reliability: float,
    history_beta: float,
    state_update_rate: float,
    multiplicity_gamma: float,
    reset_each_capture: bool,
) -> tuple[SpatialSamples, dict[str, Any]]:
    """Build contexts using only state that existed before the current window.

    A previously unseen endpoint or endpoint pair falls back to the global
    normal reference with historical reliability one.  Therefore novelty is
    never an anomaly rule.  A pair is updated once per window, irrespective of
    how many port-level flows it produced.
    """

    indices = np.asarray(indices, dtype=np.int64)
    embedding_dim = embeddings.shape[1]
    contexts = np.zeros((len(indices), embedding_dim * 2), dtype=np.float32)
    reliability = np.zeros(len(indices), dtype=np.float64)
    context_counts = np.zeros((len(indices), 2), dtype=np.int32)
    base_state = deepcopy(initial_state)
    state = deepcopy(base_state)
    global_mean = np.asarray(base_state["global_mean"], dtype=np.float32)
    if global_mean.shape != (embedding_dim,):
        raise ValueError("Historical spatial state embedding dimension mismatch")

    windows: dict[tuple[str, int], list[int]] = {}
    for local_row, raw_index in enumerate(indices.tolist()):
        key = (str(dataset.capture_ids[raw_index]), int(dataset.window_indices[raw_index]))
        windows.setdefault(key, []).append(local_row)

    previous_capture: str | None = None
    for (capture_id, _window), rows in sorted(windows.items(), key=lambda item: item[0]):
        if reset_each_capture and previous_capture is not None and capture_id != previous_capture:
            state = deepcopy(base_state)
        previous_capture = capture_id

        # Read phase: every edge in this window sees exactly the same prior
        # state.  No target from the current window can enter its own context.
        pair_rows: dict[str, list[int]] = {}
        for row in rows:
            edge = int(indices[row])
            endpoint_a = str(dataset.endpoint_a_ips[edge])
            endpoint_b = str(dataset.endpoint_b_ips[edge])
            pair_key = _pair_key(endpoint_a, endpoint_b)
            pair_rows.setdefault(pair_key, []).append(row)
            pair_entry = state["pairs"].get(pair_key)
            pair_mean = _history_mean(pair_entry, global_mean)
            for side, entity in enumerate((endpoint_a, endpoint_b)):
                entity_entry = state["entities"].get(entity)
                entity_mean = _history_mean(entity_entry, global_mean)
                # The pair history refines endpoint history but does not encode
                # a port, application type, or a novelty bit.
                context = 0.5 * entity_mean + 0.5 * pair_mean if pair_entry else entity_mean
                contexts[row, side * embedding_dim : (side + 1) * embedding_dim] = context
                context_counts[row, side] = int(
                    0 if entity_entry is None else entity_entry.get("observations", 0)
                )
            local_reliability = np.exp(
                -float(alpha) * float(np.clip(local_score[edge], 0.0, 50.0))
            )
            if pair_entry is None:
                history_reliability = 1.0
            else:
                distance = float(np.mean((embeddings[edge] - pair_mean) ** 2))
                history_reliability = float(np.exp(-float(history_beta) * min(distance, 50.0)))
            reliability[row] = max(
                float(min_reliability), local_reliability * history_reliability
            )

        # Commit phase: aggregate a whole endpoint pair to a single update.
        # This prevents a port scan from writing thousands of times into the
        # shared camera/NVR state during one three-second window.
        for pair_key, pair_group in pair_rows.items():
            group_weights = reliability[pair_group]
            group_embeddings = embeddings[indices[pair_group]]
            weight_sum = float(group_weights.sum())
            if weight_sum <= 1e-12:
                continue
            group_mean = np.average(group_embeddings, axis=0, weights=group_weights).astype(np.float32)
            pair_entry = state["pairs"].get(pair_key)
            current_count = len(pair_group)
            if pair_entry is None:
                multiplicity_gate = 1.0
            else:
                expected = max(float(pair_entry.get("count_ema", 1.0)), 1.0)
                excess_ratio = max(0.0, current_count / expected - 1.0)
                multiplicity_gate = float(np.exp(-float(multiplicity_gamma) * excess_ratio))
            update_rate = float(state_update_rate) * float(group_weights.mean()) * multiplicity_gate
            endpoint_a = str(dataset.endpoint_a_ips[int(indices[pair_group[0]])])
            endpoint_b = str(dataset.endpoint_b_ips[int(indices[pair_group[0]])])
            old_count = 1.0 if pair_entry is None else float(pair_entry.get("count_ema", 1.0))
            _ema_entry(state["pairs"], pair_key, group_mean, update_rate, global_mean)
            updated_pair = state["pairs"][pair_key]
            updated_pair["count_ema"] = (1.0 - update_rate) * old_count + update_rate * current_count
            for entity in dict.fromkeys((endpoint_a, endpoint_b)):
                _ema_entry(state["entities"], entity, group_mean, update_rate, global_mean)

    samples = SpatialSamples(
        indices=indices,
        contexts=contexts,
        targets=embeddings[indices].astype(np.float32),
        reliability=reliability,
        endpoint_context_counts=context_counts,
    )
    return samples, state


def build_spatial_samples(
    dataset: FlowDataset,
    embeddings: np.ndarray,
    local_score: np.ndarray,
    indices: np.ndarray,
    *,
    alpha: float,
    min_reliability: float,
) -> SpatialSamples:
    indices = np.asarray(indices, dtype=np.int64)
    embedding_dim = embeddings.shape[1]
    reliability = np.maximum(
        min_reliability,
        np.exp(-float(alpha) * np.clip(local_score[indices], 0.0, 50.0)),
    )
    groups: dict[tuple[str, int], list[int]] = {}
    for local_row, raw_index in enumerate(indices.tolist()):
        key = (str(dataset.capture_ids[raw_index]), int(dataset.window_indices[raw_index]))
        groups.setdefault(key, []).append(local_row)

    contexts = np.zeros((len(indices), embedding_dim * 2), dtype=np.float32)
    context_counts = np.zeros((len(indices), 2), dtype=np.int32)
    for rows in groups.values():
        weighted_sums: dict[str, np.ndarray] = {}
        weight_sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for row in rows:
            edge = int(indices[row])
            weight = float(reliability[row])
            entities = dict.fromkeys(
                (str(dataset.endpoint_a_ips[edge]), str(dataset.endpoint_b_ips[edge]))
            )
            for entity in entities:
                weighted_sums.setdefault(entity, np.zeros(embedding_dim, dtype=np.float64))
                weighted_sums[entity] += weight * embeddings[edge]
                weight_sums[entity] = weight_sums.get(entity, 0.0) + weight
                counts[entity] = counts.get(entity, 0) + 1
        for row in rows:
            edge = int(indices[row])
            weight = float(reliability[row])
            for side, entity in enumerate(
                (str(dataset.endpoint_a_ips[edge]), str(dataset.endpoint_b_ips[edge]))
            ):
                denominator = weight_sums[entity] - weight
                context_counts[row, side] = max(0, counts[entity] - 1)
                if denominator > 1e-12:
                    context = (weighted_sums[entity] - weight * embeddings[edge]) / denominator
                    contexts[row, side * embedding_dim : (side + 1) * embedding_dim] = context
    return SpatialSamples(
        indices=indices,
        contexts=contexts,
        targets=embeddings[indices].astype(np.float32),
        reliability=reliability.astype(np.float64),
        endpoint_context_counts=context_counts,
    )
