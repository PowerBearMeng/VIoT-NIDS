"""Inference for Design V4.5 target-conditioned sparse context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from data.dataset import FlowDataset
from data.sparse_context import build_scope_index, compute_sparse_observations, to_torch
from models.sparse_context import SparseContextModel
from utils.config import resolve_path
from utils.io import load_torch_checkpoint, require_alignment
from utils.scaling import VectorStandardizer


@dataclass
class V45RawScores:
    indices: np.ndarray
    reconstruction_error: np.ndarray
    pair_observed_intensity: np.ndarray
    entity_a_observed_intensity: np.ndarray
    entity_b_observed_intensity: np.ndarray
    pair_expected_mean: np.ndarray
    pair_expected_scale: np.ndarray
    entity_expected_mean: np.ndarray
    entity_expected_scale: np.ndarray
    pair_null_weight: np.ndarray
    entity_a_null_weight: np.ndarray
    entity_b_null_weight: np.ndarray
    pair_relevance: np.ndarray
    entity_a_relevance: np.ndarray
    entity_b_relevance: np.ndarray
    pair_support_size: np.ndarray
    entity_a_support_size: np.ndarray
    entity_b_support_size: np.ndarray
    pair_effective_neighbors: np.ndarray
    entity_a_effective_neighbors: np.ndarray
    entity_b_effective_neighbors: np.ndarray
    pair_context_score: np.ndarray
    entity_a_context_score: np.ndarray
    entity_b_context_score: np.ndarray
    entity_context_score: np.ndarray
    context_score: np.ndarray


def smooth_max(values: list[np.ndarray], temperature: float) -> np.ndarray:
    """Normalized log-sum-exp that remains zero when all components are zero."""

    if not values:
        raise ValueError("smooth_max requires at least one component")
    if temperature <= 0:
        raise ValueError("smooth_max temperature must be positive")
    stacked = np.stack(values, axis=0).astype(np.float64)
    maximum = stacked.max(axis=0)
    shifted = np.exp((stacked - maximum) / temperature)
    return maximum + temperature * np.log(shifted.mean(axis=0))


def _excess_energy(
    observed: np.ndarray, mean: np.ndarray, log_scale: np.ndarray
) -> np.ndarray:
    standardized = np.maximum(0.0, (observed - mean) * np.exp(-log_scale))
    return (0.5 * np.square(standardized)).astype(np.float64)


def _numpy(values: torch.Tensor, dtype: Any = np.float64) -> np.ndarray:
    return values.detach().cpu().numpy().astype(dtype, copy=False)


def score_v45_components(
    config: dict[str, Any],
    dataset: FlowDataset,
    indices: np.ndarray,
    device: torch.device,
) -> V45RawScores:
    selected = np.asarray(indices, dtype=np.int64)
    embeddings_path = resolve_path(config, config["runtime"]["embeddings_path"])
    checkpoint_path = resolve_path(
        config, config["runtime"]["sparse_context_checkpoint"]
    )
    assert embeddings_path is not None and checkpoint_path is not None
    with np.load(embeddings_path, allow_pickle=False) as archive:
        require_alignment(
            dataset.segment_ids, archive["segment_ids"], "Embedding artifact"
        )
        embeddings = archive["embeddings"].astype(np.float32)
        reconstruction_error = archive["reconstruction_error"].astype(np.float64)
    checkpoint = load_torch_checkpoint(checkpoint_path, device)
    if str(checkpoint.get("format_version")) != "4.5" or str(
        checkpoint.get("context_mode")
    ) != "sparse_context":
        raise ValueError("V4.5 inference requires a sparse_context checkpoint")
    model = SparseContextModel(**checkpoint["model_parameters"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    scaler = VectorStandardizer.from_state_dict(checkpoint["embedding_scaler"])
    normalized = torch.as_tensor(
        scaler.transform(embeddings[selected]), dtype=torch.float32, device=device
    )
    scope = to_torch(build_scope_index(dataset, selected), device)
    section = config["context_model"]
    epsilon = float(
        checkpoint.get(
            "attention_epsilon", section.get("attention_epsilon", 1e-8)
        )
    )
    chunk_size = int(
        section.get(
            "attention_chunk_size", checkpoint.get("attention_chunk_size", 512)
        )
    )
    with torch.no_grad():
        observations = compute_sparse_observations(
            *model.attention_parameters(normalized),
            scope,
            epsilon=epsilon,
            chunk_size=chunk_size,
        )
        pair_mean, pair_log_scale, entity_mean, entity_log_scale = (
            model.expected_parameters(normalized)
        )
    pair_observed = _numpy(observations.pair.observed_intensity)
    entity_a_observed = _numpy(observations.entity_a.observed_intensity)
    entity_b_observed = _numpy(observations.entity_b.observed_intensity)
    pair_mean_np = _numpy(pair_mean)
    pair_log_scale_np = _numpy(pair_log_scale)
    entity_mean_np = _numpy(entity_mean)
    entity_log_scale_np = _numpy(entity_log_scale)
    pair_score = _excess_energy(pair_observed, pair_mean_np, pair_log_scale_np)
    entity_a_score = _excess_energy(
        entity_a_observed, entity_mean_np, entity_log_scale_np
    )
    entity_b_score = _excess_energy(
        entity_b_observed, entity_mean_np, entity_log_scale_np
    )
    scope_temperature = float(section.get("scope_temperature", 1.0))
    entity_score = smooth_max(
        [entity_a_score, entity_b_score], scope_temperature
    )
    active: list[np.ndarray] = []
    if bool(checkpoint.get("pair_enabled", True)):
        active.append(pair_score)
    if bool(checkpoint.get("entity_enabled", True)):
        active.extend([entity_a_score, entity_b_score])
    context_score = smooth_max(active, scope_temperature)
    return V45RawScores(
        indices=selected,
        reconstruction_error=reconstruction_error[selected],
        pair_observed_intensity=pair_observed,
        entity_a_observed_intensity=entity_a_observed,
        entity_b_observed_intensity=entity_b_observed,
        pair_expected_mean=pair_mean_np,
        pair_expected_scale=np.exp(pair_log_scale_np),
        entity_expected_mean=entity_mean_np,
        entity_expected_scale=np.exp(entity_log_scale_np),
        pair_null_weight=_numpy(observations.pair.null_weight),
        entity_a_null_weight=_numpy(observations.entity_a.null_weight),
        entity_b_null_weight=_numpy(observations.entity_b.null_weight),
        pair_relevance=_numpy(observations.pair.relevance),
        entity_a_relevance=_numpy(observations.entity_a.relevance),
        entity_b_relevance=_numpy(observations.entity_b.relevance),
        pair_support_size=_numpy(observations.pair.support_size, np.int64),
        entity_a_support_size=_numpy(
            observations.entity_a.support_size, np.int64
        ),
        entity_b_support_size=_numpy(
            observations.entity_b.support_size, np.int64
        ),
        pair_effective_neighbors=_numpy(
            observations.pair.effective_neighbors
        ),
        entity_a_effective_neighbors=_numpy(
            observations.entity_a.effective_neighbors
        ),
        entity_b_effective_neighbors=_numpy(
            observations.entity_b.effective_neighbors
        ),
        pair_context_score=pair_score,
        entity_a_context_score=entity_a_score,
        entity_b_context_score=entity_b_score,
        entity_context_score=entity_score,
        context_score=context_score,
    )
