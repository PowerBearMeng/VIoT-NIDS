"""Inference for V4 continuous neural behavior intensity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from data.dataset import FlowDataset
from data.neural_intensity import aggregate_soft_masses, build_scope_index
from models.neural_intensity_context import NeuralIntensityContext
from utils.config import resolve_path
from utils.io import load_torch_checkpoint, require_alignment
from utils.scaling import VectorStandardizer


@dataclass
class V4RawScores:
    indices: np.ndarray
    reconstruction_error: np.ndarray
    pair_log_mass: np.ndarray
    entity_a_log_mass: np.ndarray
    entity_b_log_mass: np.ndarray
    pair_expected_mean: np.ndarray
    pair_expected_scale: np.ndarray
    entity_expected_mean: np.ndarray
    entity_expected_scale: np.ndarray
    pair_context_score: np.ndarray
    entity_a_context_score: np.ndarray
    entity_b_context_score: np.ndarray
    entity_context_score: np.ndarray
    context_score: np.ndarray
    assignment_entropy: np.ndarray
    assignment_peak: np.ndarray


def smooth_max(values: list[np.ndarray], temperature: float) -> np.ndarray:
    """Normalized log-sum-exp that stays zero when every component is zero."""

    if not values:
        raise ValueError("smooth_max requires at least one component")
    if temperature <= 0:
        raise ValueError("smooth_max temperature must be positive")
    stacked = np.stack(values, axis=0).astype(np.float64)
    maximum = stacked.max(axis=0)
    shifted = np.exp((stacked - maximum) / temperature)
    return maximum + temperature * (
        np.log(shifted.mean(axis=0))
    )


def _predict_model(
    model: NeuralIntensityContext,
    embeddings: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assignments: list[np.ndarray] = []
    pair_means: list[np.ndarray] = []
    pair_log_scales: list[np.ndarray] = []
    entity_means: list[np.ndarray] = []
    entity_log_scales: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(embeddings), batch_size):
            stop = min(len(embeddings), start + batch_size)
            values = torch.as_tensor(
                embeddings[start:stop], dtype=torch.float32, device=device
            )
            assignments.append(model.assignments(values).cpu().numpy())
            pair_mean, pair_log_scale, entity_mean, entity_log_scale = (
                model.expected_parameters(values)
            )
            pair_means.append(pair_mean.cpu().numpy())
            pair_log_scales.append(pair_log_scale.cpu().numpy())
            entity_means.append(entity_mean.cpu().numpy())
            entity_log_scales.append(entity_log_scale.cpu().numpy())
    return tuple(
        np.concatenate(parts).astype(np.float32)
        for parts in (
            assignments,
            pair_means,
            pair_log_scales,
            entity_means,
            entity_log_scales,
        )
    )


def _excess_energy(
    observed: np.ndarray, mean: np.ndarray, log_scale: np.ndarray
) -> np.ndarray:
    standardized = np.maximum(0.0, (observed - mean) * np.exp(-log_scale))
    return (0.5 * standardized * standardized).astype(np.float64)


def score_v4_components(
    config: dict[str, Any],
    dataset: FlowDataset,
    indices: np.ndarray,
    device: torch.device,
) -> V4RawScores:
    selected = np.asarray(indices, dtype=np.int64)
    embeddings_path = resolve_path(config, config["runtime"]["embeddings_path"])
    checkpoint_path = resolve_path(
        config, config["runtime"]["neural_context_checkpoint"]
    )
    assert embeddings_path is not None and checkpoint_path is not None
    with np.load(embeddings_path, allow_pickle=False) as archive:
        require_alignment(dataset.segment_ids, archive["segment_ids"], "Embedding artifact")
        embeddings = archive["embeddings"].astype(np.float32)
        reconstruction_error = archive["reconstruction_error"].astype(np.float64)
    checkpoint = load_torch_checkpoint(checkpoint_path, device)
    if int(checkpoint.get("format_version", 0)) != 4:
        raise ValueError("V4 inference requires a format_version=4 context checkpoint")
    model = NeuralIntensityContext(**checkpoint["model_parameters"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    scaler = VectorStandardizer.from_state_dict(checkpoint["embedding_scaler"])
    normalized = scaler.transform(embeddings[selected])
    section = config["context_model"]
    assignments, pair_mean, pair_log_scale, entity_mean, entity_log_scale = (
        _predict_model(
            model,
            normalized,
            int(section.get("inference_batch_size", 8192)),
            device,
        )
    )
    scope = build_scope_index(dataset, selected)
    pair_mass, entity_a_mass, entity_b_mass = aggregate_soft_masses(
        assignments, scope
    )
    pair_score = _excess_energy(pair_mass, pair_mean, pair_log_scale)
    entity_a_score = _excess_energy(
        entity_a_mass, entity_mean, entity_log_scale
    )
    entity_b_score = _excess_energy(
        entity_b_mass, entity_mean, entity_log_scale
    )
    entity_score = smooth_max(
        [entity_a_score, entity_b_score],
        float(section.get("scope_temperature", 1.0)),
    )
    active: list[np.ndarray] = []
    if bool(checkpoint.get("pair_enabled", True)):
        active.append(pair_score)
    if bool(checkpoint.get("entity_enabled", True)):
        active.extend([entity_a_score, entity_b_score])
    context_score = smooth_max(
        active, float(section.get("scope_temperature", 1.0))
    )
    clipped = np.clip(assignments.astype(np.float64), 1e-12, 1.0)
    assignment_entropy = -np.sum(clipped * np.log(clipped), axis=1)
    return V4RawScores(
        indices=selected,
        reconstruction_error=reconstruction_error[selected],
        pair_log_mass=pair_mass.astype(np.float64),
        entity_a_log_mass=entity_a_mass.astype(np.float64),
        entity_b_log_mass=entity_b_mass.astype(np.float64),
        pair_expected_mean=pair_mean.astype(np.float64),
        pair_expected_scale=np.exp(pair_log_scale.astype(np.float64)),
        entity_expected_mean=entity_mean.astype(np.float64),
        entity_expected_scale=np.exp(entity_log_scale.astype(np.float64)),
        pair_context_score=pair_score,
        entity_a_context_score=entity_a_score,
        entity_b_context_score=entity_b_score,
        entity_context_score=entity_score,
        context_score=context_score,
        assignment_entropy=assignment_entropy,
        assignment_peak=assignments.max(axis=1).astype(np.float64),
    )
