"""Port-free pair/entity scopes and target-conditioned sparse attention.

IP addresses are used only to construct transient groups inside one capture
and one three-second window.  They are never passed to the neural model and
are not persisted as identity-specific reference keys.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

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
    entity_a_memberships: np.ndarray
    entity_b_memberships: np.ndarray
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
    entity_a_memberships: torch.Tensor
    entity_b_memberships: torch.Tensor
    pair_group_count: int
    entity_group_count: int

    def __len__(self) -> int:
        return int(self.pair_ids.numel())


@dataclass(frozen=True)
class SparseAttentionStats:
    null_weight: torch.Tensor
    support_size: torch.Tensor
    relevance: torch.Tensor
    effective_neighbors: torch.Tensor
    observed_intensity: torch.Tensor

    def select(self, indices: torch.Tensor) -> "SparseAttentionStats":
        return SparseAttentionStats(
            null_weight=self.null_weight[indices],
            support_size=self.support_size[indices],
            relevance=self.relevance[indices],
            effective_neighbors=self.effective_neighbors[indices],
            observed_intensity=self.observed_intensity[indices],
        )


@dataclass(frozen=True)
class SparseContextObservations:
    pair: SparseAttentionStats
    entity_a: SparseAttentionStats
    entity_b: SparseAttentionStats


def _intern(mapping: dict[tuple[object, ...], int], key: tuple[object, ...]) -> int:
    value = mapping.get(key)
    if value is None:
        value = len(mapping)
        mapping[key] = value
    return value


def build_scope_index(dataset: FlowDataset, indices: np.ndarray) -> ScopeIndex:
    """Build pair/entity memberships without consulting port attributes."""

    selected = np.asarray(indices, dtype=np.int64)
    pair_mapping: dict[tuple[object, ...], int] = {}
    entity_mapping: dict[tuple[object, ...], int] = {}
    pair_ids = np.empty(len(selected), dtype=np.int64)
    entity_a_ids = np.empty(len(selected), dtype=np.int64)
    entity_b_ids = np.empty(len(selected), dtype=np.int64)
    entity_a_memberships = np.empty(len(selected), dtype=np.int64)
    entity_b_memberships = np.empty(len(selected), dtype=np.int64)
    membership_edges: list[int] = []
    membership_ids: list[int] = []
    for row, index in enumerate(selected.tolist()):
        capture = str(dataset.capture_ids[index])
        window = int(dataset.window_indices[index])
        endpoint_a = str(dataset.endpoint_a_ips[index])
        endpoint_b = str(dataset.endpoint_b_ips[index])
        pair_a, pair_b = sorted((endpoint_a, endpoint_b))
        pair_ids[row] = _intern(pair_mapping, (capture, window, pair_a, pair_b))
        entity_a_ids[row] = _intern(entity_mapping, (capture, window, endpoint_a))
        entity_b_ids[row] = _intern(entity_mapping, (capture, window, endpoint_b))

        entity_a_memberships[row] = len(membership_edges)
        membership_edges.append(row)
        membership_ids.append(int(entity_a_ids[row]))
        if entity_b_ids[row] == entity_a_ids[row]:
            entity_b_memberships[row] = entity_a_memberships[row]
        else:
            entity_b_memberships[row] = len(membership_edges)
            membership_edges.append(row)
            membership_ids.append(int(entity_b_ids[row]))
    return ScopeIndex(
        indices=selected,
        pair_ids=pair_ids,
        entity_a_ids=entity_a_ids,
        entity_b_ids=entity_b_ids,
        entity_membership_edges=np.asarray(membership_edges, dtype=np.int64),
        entity_membership_ids=np.asarray(membership_ids, dtype=np.int64),
        entity_a_memberships=entity_a_memberships,
        entity_b_memberships=entity_b_memberships,
        pair_group_count=len(pair_mapping),
        entity_group_count=len(entity_mapping),
    )


def to_torch(scope: ScopeIndex, device: torch.device) -> TorchScopeIndex:
    def tensor(values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.long, device=device)

    return TorchScopeIndex(
        pair_ids=tensor(scope.pair_ids),
        entity_a_ids=tensor(scope.entity_a_ids),
        entity_b_ids=tensor(scope.entity_b_ids),
        entity_membership_edges=tensor(scope.entity_membership_edges),
        entity_membership_ids=tensor(scope.entity_membership_ids),
        entity_a_memberships=tensor(scope.entity_a_memberships),
        entity_b_memberships=tensor(scope.entity_b_memberships),
        pair_group_count=scope.pair_group_count,
        entity_group_count=scope.entity_group_count,
    )


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax projection onto the probability simplex."""

    if logits.shape[dim] == 0:
        raise ValueError("sparsemax requires a nonempty dimension")
    shifted = logits - logits.amax(dim=dim, keepdim=True)
    sorted_logits, _ = torch.sort(shifted, dim=dim, descending=True)
    size = logits.shape[dim]
    ranks_shape = [1] * logits.ndim
    ranks_shape[dim] = size
    ranks = torch.arange(
        1, size + 1, dtype=logits.dtype, device=logits.device
    ).reshape(ranks_shape)
    cumulative = sorted_logits.cumsum(dim)
    support = 1.0 + ranks * sorted_logits > cumulative
    support_size = support.sum(dim=dim, keepdim=True).clamp_min(1)
    threshold_sum = cumulative.gather(dim, support_size - 1)
    threshold = (threshold_sum - 1.0) / support_size.to(logits.dtype)
    return torch.clamp(shifted - threshold, min=0.0)


def sparse_attention_by_scope(
    query: torch.Tensor,
    key: torch.Tensor,
    null_scores: torch.Tensor,
    membership_edges: torch.Tensor,
    membership_group_ids: torch.Tensor,
    *,
    epsilon: float = 1e-8,
    chunk_size: int = 512,
) -> SparseAttentionStats:
    """Attend from each membership target to other members of its scope.

    Results are aligned with ``membership_edges``.  Every membership occurs
    exactly once in a scope; self-attention is masked before sparsemax.  Target
    chunks bound temporary score-matrix memory while retaining exact sparsemax
    over all neighbors in the group.
    """

    if query.ndim != 2 or key.shape != query.shape:
        raise ValueError("query and key must have identical [N,d] shapes")
    if null_scores.shape != (len(query),):
        raise ValueError("null_scores must have shape [N]")
    if len(membership_edges) != len(membership_group_ids):
        raise ValueError("membership arrays must have equal length")
    if epsilon <= 0.0 or chunk_size <= 0:
        raise ValueError("epsilon and chunk_size must be positive")
    if not len(membership_edges):
        empty = query.new_empty((0,))
        return SparseAttentionStats(
            empty, empty.to(torch.long), empty, empty, empty
        )

    order = torch.argsort(membership_group_ids, stable=True)
    sorted_group_ids = membership_group_ids[order]
    _, counts = torch.unique_consecutive(sorted_group_ids, return_counts=True)
    scale = math.sqrt(float(query.shape[1]))
    null_parts: list[torch.Tensor] = []
    support_parts: list[torch.Tensor] = []
    relevance_parts: list[torch.Tensor] = []
    effective_parts: list[torch.Tensor] = []
    observed_parts: list[torch.Tensor] = []
    cursor = 0
    for count in counts.detach().cpu().tolist():
        positions = order[cursor : cursor + count]
        rows = membership_edges[positions]
        group_query = query[rows]
        group_key = key[rows]
        group_null = null_scores[rows]
        for start in range(0, count, chunk_size):
            stop = min(count, start + chunk_size)
            scores = group_query[start:stop] @ group_key.transpose(0, 1)
            scores = scores / scale
            self_mask = torch.zeros_like(scores, dtype=torch.bool)
            local_rows = torch.arange(stop - start, device=query.device)
            self_mask[local_rows, torch.arange(start, stop, device=query.device)] = True
            scores = scores.masked_fill(self_mask, -torch.inf)
            logits = torch.cat([group_null[start:stop, None], scores], dim=1)
            weights = sparsemax(logits, dim=1)
            null_weight = weights[:, 0]
            neighbor_weights = weights[:, 1:]
            relevance = 1.0 - null_weight
            square_sum = neighbor_weights.square().sum(dim=1)
            effective = relevance.square() / (square_sum + float(epsilon))
            support_size = (neighbor_weights > 0.0).sum(dim=1)
            observed = torch.log1p(relevance * effective)
            null_parts.append(null_weight)
            support_parts.append(support_size)
            relevance_parts.append(relevance)
            effective_parts.append(effective)
            observed_parts.append(observed)
        cursor += count

    def restore(parts: list[torch.Tensor]) -> torch.Tensor:
        sorted_values = torch.cat(parts, dim=0)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(len(order), device=order.device)
        return sorted_values[inverse]

    return SparseAttentionStats(
        null_weight=restore(null_parts),
        support_size=restore(support_parts),
        relevance=restore(relevance_parts),
        effective_neighbors=restore(effective_parts),
        observed_intensity=restore(observed_parts),
    )


def compute_sparse_observations(
    pair_query: torch.Tensor,
    pair_key: torch.Tensor,
    pair_null: torch.Tensor,
    entity_query: torch.Tensor,
    entity_key: torch.Tensor,
    entity_null: torch.Tensor,
    scope: TorchScopeIndex,
    *,
    epsilon: float,
    chunk_size: int,
) -> SparseContextObservations:
    pair = sparse_attention_by_scope(
        pair_query,
        pair_key,
        pair_null,
        torch.arange(len(scope), device=pair_query.device),
        scope.pair_ids,
        epsilon=epsilon,
        chunk_size=chunk_size,
    )
    entity_memberships = sparse_attention_by_scope(
        entity_query,
        entity_key,
        entity_null,
        scope.entity_membership_edges,
        scope.entity_membership_ids,
        epsilon=epsilon,
        chunk_size=chunk_size,
    )
    return SparseContextObservations(
        pair=pair,
        entity_a=entity_memberships.select(scope.entity_a_memberships),
        entity_b=entity_memberships.select(scope.entity_b_memberships),
    )
