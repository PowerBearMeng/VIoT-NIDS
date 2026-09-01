"""Load all three stages and produce raw anomaly components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from data.behavior_composition import (
    BehaviorCompositionReference,
    score_behavior_composition,
)
from data.dataset import FlowDataset
from data.entity_state import EntitySequences, build_entity_sequences
from data.spatial_context import (
    SpatialSamples,
    build_historical_spatial_samples,
    build_spatial_samples,
)
from models.entity_predictor import EntityGRUPredictor
from models.spatial_predictor import SpatialContextPredictor
from utils.config import resolve_path
from utils.io import load_torch_checkpoint, require_alignment
from utils.scaling import MixedFeatureStandardizer


@dataclass
class RawScores:
    indices: np.ndarray
    prototype_distance: np.ndarray
    prototype_score: np.ndarray
    reconstruction_error: np.ndarray
    reconstruction_score: np.ndarray
    combined_local_score: np.ndarray
    local_score: np.ndarray
    spatial_score: np.ndarray
    mode_ids: np.ndarray
    pair_context_score: np.ndarray
    entity_a_context_score: np.ndarray
    entity_b_context_score: np.ndarray
    entity_context_score: np.ndarray
    context_score: np.ndarray
    pair_mode_count: np.ndarray
    entity_a_mode_count: np.ndarray
    entity_b_mode_count: np.ndarray
    entity_a_score: np.ndarray
    entity_b_score: np.ndarray
    entity_score: np.ndarray
    reliability: np.ndarray
    context_counts: np.ndarray
    entity_sequences: EntitySequences
    entity_state_scores: np.ndarray


def _predict_batches(
    model: torch.nn.Module,
    inputs: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    errors: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            stop = min(len(inputs), start + batch_size)
            values = torch.as_tensor(inputs[start:stop], dtype=torch.float32, device=device)
            truth = torch.as_tensor(targets[start:stop], dtype=torch.float32, device=device)
            predicted = model(values)
            errors.append((predicted - truth).square().mean(dim=-1).cpu().numpy())
    return np.concatenate(errors) if errors else np.empty(0, dtype=np.float64)


def _legacy_spatial_scores(
    config: dict[str, Any],
    dataset: FlowDataset,
    embeddings: np.ndarray,
    local_score_all: np.ndarray,
    indices: np.ndarray,
    checkpoint_path: Any,
    device: torch.device,
) -> tuple[np.ndarray, SpatialSamples]:
    spatial_checkpoint = load_torch_checkpoint(checkpoint_path, device)
    spatial_model = SpatialContextPredictor(**spatial_checkpoint["model_parameters"]).to(device)
    spatial_model.load_state_dict(spatial_checkpoint["model_state"])
    embedding_scaler = MixedFeatureStandardizer.from_state_dict(spatial_checkpoint["embedding_scaler"])
    context_mode = str(spatial_checkpoint.get("context_mode", "current_window"))
    if context_mode == "historical":
        historical_parameters = spatial_checkpoint["historical_parameters"]
        spatial_samples, _ = build_historical_spatial_samples(
            dataset,
            embeddings,
            local_score_all,
            indices,
            initial_state=spatial_checkpoint["historical_reference_state"],
            alpha=float(spatial_checkpoint["alpha"]),
            min_reliability=float(spatial_checkpoint["min_reliability"]),
            history_beta=float(historical_parameters["history_beta"]),
            state_update_rate=float(historical_parameters["state_update_rate"]),
            multiplicity_gamma=float(historical_parameters["multiplicity_gamma"]),
            reset_each_capture=True,
        )
    else:
        spatial_samples = build_spatial_samples(
            dataset,
            embeddings,
            local_score_all,
            indices,
            alpha=float(spatial_checkpoint["alpha"]),
            min_reliability=float(spatial_checkpoint["min_reliability"]),
        )
    embedding_dim = embeddings.shape[1]
    spatial_inputs = embedding_scaler.transform(
        spatial_samples.contexts.reshape(-1, 2, embedding_dim)
    ).reshape(-1, embedding_dim * 2)
    spatial_targets = embedding_scaler.transform(spatial_samples.targets)
    scores = _predict_batches(
        spatial_model,
        spatial_inputs,
        spatial_targets,
        int(config["spatial_model"]["batch_size"]),
        device,
    )
    return scores.astype(np.float64), spatial_samples


def score_raw_components(
    config: dict[str, Any],
    dataset: FlowDataset,
    indices: np.ndarray,
    device: torch.device,
) -> RawScores:
    indices = np.asarray(indices, dtype=np.int64)
    embeddings_path = resolve_path(config, config["runtime"]["embeddings_path"])
    prototypes_path = resolve_path(config, config["runtime"]["prototypes_path"])
    entity_path = resolve_path(config, config["runtime"]["entity_checkpoint"])
    context_section = config.get("context_model", {})
    behavior_mode = str(context_section.get("mode", "legacy_spatial")) == "behavior_composition"
    needs_legacy_spatial = (not behavior_mode) or bool(
        context_section.get("legacy_spatial_ablation", False)
    )
    spatial_path = resolve_path(config, config["runtime"].get("spatial_checkpoint"))
    assert embeddings_path is not None and prototypes_path is not None and entity_path is not None
    if needs_legacy_spatial and spatial_path is None:
        raise ValueError("A spatial checkpoint is required for legacy spatial scoring")
    with np.load(embeddings_path, allow_pickle=False) as archive:
        require_alignment(dataset.segment_ids, archive["segment_ids"], "Embedding artifact")
        embeddings = archive["embeddings"].astype(np.float32)
        reconstruction_error_all = archive["reconstruction_error"].astype(np.float64)
    with np.load(prototypes_path, allow_pickle=False) as archive:
        require_alignment(dataset.segment_ids, archive["segment_ids"], "Prototype artifact")
        assignments = archive["assignments"].astype(np.int64)
        prototype_distance_all = archive["prototype_distance"].astype(np.float64)
        distance_scale = archive["distance_scale"].astype(np.float64)
        reconstruction_scale = archive["reconstruction_scale"].astype(np.float64)
        local_weights = archive["local_weights"].astype(np.float64)
        prototype_score_all = (
            archive["prototype_score"].astype(np.float64)
            if "prototype_score" in archive
            else np.maximum(
                0.0,
                (prototype_distance_all - float(distance_scale[0]))
                / max(float(distance_scale[1] - distance_scale[0]), 1e-12),
            )
        )
        reconstruction_score_all = (
            archive["reconstruction_score"].astype(np.float64)
            if "reconstruction_score" in archive
            else np.maximum(
                0.0,
                (reconstruction_error_all - float(reconstruction_scale[0]))
                / max(float(reconstruction_scale[1] - reconstruction_scale[0]), 1e-12),
            )
        )
        combined_local_score_all = (
            archive["combined_local_score"].astype(np.float64)
            if "combined_local_score" in archive
            else float(local_weights[0]) * prototype_score_all
            + float(local_weights[1]) * reconstruction_score_all
        )
        local_score_all = archive["local_score"].astype(np.float64)
        prototype_count = int(archive["centers"].shape[0])

    entity_checkpoint = load_torch_checkpoint(entity_path, device)
    if int(entity_checkpoint["prototype_count"]) != prototype_count:
        raise ValueError("Entity checkpoint prototype count differs from prototype artifact")
    entity_model = EntityGRUPredictor(**entity_checkpoint["model_parameters"]).to(device)
    entity_model.load_state_dict(entity_checkpoint["model_state"])
    entity_scaler = MixedFeatureStandardizer.from_state_dict(entity_checkpoint["state_scaler"])
    entity_sequences = build_entity_sequences(
        dataset,
        embeddings,
        assignments,
        prototype_count=prototype_count,
        indices=indices,
        history_windows=int(entity_checkpoint["history_windows"]),
        include_mean_embedding=bool(entity_checkpoint["include_mean_embedding"]),
    )
    scaled_histories = entity_scaler.transform(entity_sequences.histories)
    scaled_targets = entity_scaler.transform(entity_sequences.targets)
    entity_state_scores = _predict_batches(
        entity_model,
        scaled_histories,
        scaled_targets,
        int(config["entity_model"]["batch_size"]),
        device,
    )
    entity_a = entity_state_scores[entity_sequences.edge_entity_a[indices]]
    entity_b = entity_state_scores[entity_sequences.edge_entity_b[indices]]

    if needs_legacy_spatial:
        legacy_source = str(config["spatial_model"].get("local_score_source", "local"))
        legacy_local = (
            combined_local_score_all if legacy_source == "combined" else local_score_all
        )
        assert spatial_path is not None
        spatial_scores, spatial_samples = _legacy_spatial_scores(
            config, dataset, embeddings, legacy_local, indices, spatial_path, device
        )
    else:
        spatial_scores = np.zeros(len(indices), dtype=np.float64)
        spatial_samples = SpatialSamples(
            indices=indices,
            contexts=np.zeros((len(indices), embeddings.shape[1] * 2), dtype=np.float32),
            targets=embeddings[indices],
            reliability=np.ones(len(indices), dtype=np.float64),
            endpoint_context_counts=np.zeros((len(indices), 2), dtype=np.int32),
        )

    if behavior_mode:
        reference_path = resolve_path(config, config["runtime"]["context_reference_path"])
        assert reference_path is not None
        reference = BehaviorCompositionReference.load(reference_path)
        behavior = score_behavior_composition(
            dataset,
            assignments,
            indices,
            reference,
            pair_enabled=bool(context_section.get("pair_enabled", True)),
            entity_enabled=bool(context_section.get("entity_enabled", True)),
            positive_deviation_only=bool(
                context_section.get("positive_deviation_only", True)
            ),
        )
        pair_context = behavior.pair_deviation
        entity_a_context = behavior.entity_a_deviation
        entity_b_context = behavior.entity_b_deviation
        entity_context = behavior.entity_deviation
        context_score = behavior.context_deviation
        pair_mode_count = behavior.pair_mode_count
        entity_a_mode_count = behavior.entity_a_mode_count
        entity_b_mode_count = behavior.entity_b_mode_count
    else:
        pair_context = np.zeros(len(indices), dtype=np.float64)
        entity_a_context = np.zeros(len(indices), dtype=np.float64)
        entity_b_context = np.zeros(len(indices), dtype=np.float64)
        entity_context = np.zeros(len(indices), dtype=np.float64)
        context_score = spatial_scores.copy()
        pair_mode_count = np.zeros(len(indices), dtype=np.int64)
        entity_a_mode_count = np.zeros(len(indices), dtype=np.int64)
        entity_b_mode_count = np.zeros(len(indices), dtype=np.int64)
    return RawScores(
        indices=indices,
        prototype_distance=prototype_distance_all[indices],
        prototype_score=prototype_score_all[indices],
        reconstruction_error=reconstruction_error_all[indices],
        reconstruction_score=reconstruction_score_all[indices],
        combined_local_score=combined_local_score_all[indices],
        local_score=local_score_all[indices],
        spatial_score=spatial_scores.astype(np.float64),
        mode_ids=assignments[indices],
        pair_context_score=pair_context,
        entity_a_context_score=entity_a_context,
        entity_b_context_score=entity_b_context,
        entity_context_score=entity_context,
        context_score=context_score,
        pair_mode_count=pair_mode_count,
        entity_a_mode_count=entity_a_mode_count,
        entity_b_mode_count=entity_b_mode_count,
        entity_a_score=entity_a.astype(np.float64),
        entity_b_score=entity_b.astype(np.float64),
        entity_score=np.maximum(entity_a, entity_b).astype(np.float64),
        reliability=spatial_samples.reliability,
        context_counts=spatial_samples.endpoint_context_counts,
        entity_sequences=entity_sequences,
        entity_state_scores=entity_state_scores.astype(np.float64),
    )
