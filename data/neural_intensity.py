"""Continuous, port-free scope aggregation for Design V4.

The aggregation is linear in the number of Flow segments.  A learned soft
behavior vector replaces V3's hard KMeans ``mode_id``.  Each target Flow reads
only the behavior-weighted mass similar to itself, so an unrelated entity-wide
burst is not copied indiscriminately to every Flow.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .dataset import FlowDataset


@dataclass(frozen=True)
class ScopeIndex:
    indices: np.ndarray
    pair_ids: np.ndarray
    entity_a_ids: np.ndarray
    entity_b_ids: np.ndarray
    entity_membership_edges: np.ndarray
    entity_membership_ids: np.ndarray
    pair_group_count: int
    entity_group_count: int

    def __len__(self) -> int:
        return len(self.indices)


@dataclass(frozen=True)
class TorchScopeIndex:
    pair_ids: torch.Tensor
    entity_a_ids: torch.Tensor
    entity_b_ids: torch.Tensor
    entity_membership_edges: torch.Tensor
    entity_membership_ids: torch.Tensor
    pair_group_count: int
    entity_group_count: int


def _intern(mapping: dict[tuple[object, ...], int], key: tuple[object, ...]) -> int:
    value = mapping.get(key)
    if value is None:
        value = len(mapping)
        mapping[key] = value
    return value


def build_scope_index(dataset: FlowDataset, indices: np.ndarray) -> ScopeIndex:
    """Build 3-second pair/entity groups without using ports as keys."""

    selected = np.asarray(indices, dtype=np.int64)
    pair_mapping: dict[tuple[object, ...], int] = {}
    entity_mapping: dict[tuple[object, ...], int] = {}
    pair_ids = np.empty(len(selected), dtype=np.int64)
    entity_a_ids = np.empty(len(selected), dtype=np.int64)
    entity_b_ids = np.empty(len(selected), dtype=np.int64)
    membership_edges: list[int] = []
    membership_ids: list[int] = []
    for row, index in enumerate(selected.tolist()):
        capture = str(dataset.capture_ids[index])
        window = int(dataset.window_indices[index])
        endpoint_a = str(dataset.endpoint_a_ips[index])
        endpoint_b = str(dataset.endpoint_b_ips[index])
        pair_a, pair_b = sorted((endpoint_a, endpoint_b))
        pair_ids[row] = _intern(
            pair_mapping, (capture, window, pair_a, pair_b)
        )
        entity_a_ids[row] = _intern(
            entity_mapping, (capture, window, endpoint_a)
        )
        entity_b_ids[row] = _intern(
            entity_mapping, (capture, window, endpoint_b)
        )
        membership_edges.append(row)
        membership_ids.append(int(entity_a_ids[row]))
        if entity_b_ids[row] != entity_a_ids[row]:
            membership_edges.append(row)
            membership_ids.append(int(entity_b_ids[row]))
    return ScopeIndex(
        indices=selected,
        pair_ids=pair_ids,
        entity_a_ids=entity_a_ids,
        entity_b_ids=entity_b_ids,
        entity_membership_edges=np.asarray(membership_edges, dtype=np.int64),
        entity_membership_ids=np.asarray(membership_ids, dtype=np.int64),
        pair_group_count=len(pair_mapping),
        entity_group_count=len(entity_mapping),
    )


def to_torch(scope: ScopeIndex, device: torch.device) -> TorchScopeIndex:
    return TorchScopeIndex(
        pair_ids=torch.as_tensor(scope.pair_ids, dtype=torch.long, device=device),
        entity_a_ids=torch.as_tensor(
            scope.entity_a_ids, dtype=torch.long, device=device
        ),
        entity_b_ids=torch.as_tensor(
            scope.entity_b_ids, dtype=torch.long, device=device
        ),
        entity_membership_edges=torch.as_tensor(
            scope.entity_membership_edges, dtype=torch.long, device=device
        ),
        entity_membership_ids=torch.as_tensor(
            scope.entity_membership_ids, dtype=torch.long, device=device
        ),
        pair_group_count=scope.pair_group_count,
        entity_group_count=scope.entity_group_count,
    )


def aggregate_soft_masses(
    assignments: np.ndarray, scope: ScopeIndex
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return leave-one-out pair/entity soft similarity masses."""

    weights = np.asarray(assignments, dtype=np.float64)
    if weights.ndim != 2 or len(weights) != len(scope):
        raise ValueError("Soft assignments must align with ScopeIndex rows")
    channels = weights.shape[1]
    pair_sums = np.zeros((scope.pair_group_count, channels), dtype=np.float64)
    entity_sums = np.zeros((scope.entity_group_count, channels), dtype=np.float64)
    for channel in range(channels):
        pair_sums[:, channel] = np.bincount(
            scope.pair_ids,
            weights=weights[:, channel],
            minlength=scope.pair_group_count,
        )
        entity_sums[:, channel] = np.bincount(
            scope.entity_membership_ids,
            weights=weights[scope.entity_membership_edges, channel],
            minlength=scope.entity_group_count,
        )
    pair_mass = np.sum(weights * (pair_sums[scope.pair_ids] - weights), axis=1)
    entity_a_mass = np.sum(
        weights * (entity_sums[scope.entity_a_ids] - weights), axis=1
    )
    entity_b_mass = np.sum(
        weights * (entity_sums[scope.entity_b_ids] - weights), axis=1
    )
    return tuple(
        np.log1p(np.maximum(values, 0.0)).astype(np.float32)
        for values in (pair_mass, entity_a_mass, entity_b_mass)
    )


def aggregate_soft_masses_torch(
    assignments: torch.Tensor, scope: TorchScopeIndex
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable counterpart used while fitting the normal intensity."""

    channels = assignments.shape[1]
    pair_sums = torch.zeros(
        (scope.pair_group_count, channels),
        dtype=assignments.dtype,
        device=assignments.device,
    )
    entity_sums = torch.zeros(
        (scope.entity_group_count, channels),
        dtype=assignments.dtype,
        device=assignments.device,
    )
    pair_sums.index_add_(0, scope.pair_ids, assignments)
    entity_sums.index_add_(
        0,
        scope.entity_membership_ids,
        assignments[scope.entity_membership_edges],
    )
    pair_mass = (assignments * (pair_sums[scope.pair_ids] - assignments)).sum(-1)
    entity_a_mass = (
        assignments * (entity_sums[scope.entity_a_ids] - assignments)
    ).sum(-1)
    entity_b_mass = (
        assignments * (entity_sums[scope.entity_b_ids] - assignments)
    ).sum(-1)
    return tuple(
        torch.log1p(values.clamp_min(0.0))
        for values in (pair_mass, entity_a_mass, entity_b_mass)
    )
