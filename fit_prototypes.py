#!/usr/bin/env python3
"""Fit normal MiniBatchKMeans prototypes and calibrate local components."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from data.dataset import FlowDataset
from utils.config import load_config, resolve_path
from utils.io import require_alignment
from utils.scaling import QuantileScoreScaler


def fit(config: dict[str, Any]) -> dict[str, Any]:
    dataset_path = resolve_path(config, config["data"]["processed_path"])
    metadata_path = resolve_path(config, config["data"]["metadata_path"])
    embeddings_path = resolve_path(config, config["runtime"]["embeddings_path"])
    output_path = resolve_path(config, config["runtime"]["prototypes_path"])
    assert dataset_path is not None and metadata_path is not None and embeddings_path is not None and output_path is not None
    dataset = FlowDataset(dataset_path, metadata_path)
    with np.load(embeddings_path, allow_pickle=False) as archive:
        require_alignment(dataset.segment_ids, archive["segment_ids"], "Embedding artifact")
        embeddings = archive["embeddings"].astype(np.float32)
        reconstruction_error = archive["reconstruction_error"].astype(np.float64)
    train_indices = dataset.require_normal("train", "normal prototype fitting")
    calibration_indices = dataset.require_normal("calibration", "local score calibration")
    requested = int(config["prototypes"]["count"])
    count = min(requested, len(train_indices))
    if count < requested:
        print(f"prototype count reduced from {requested} to {count}: only {len(train_indices)} normal training segments")
    model = MiniBatchKMeans(
        n_clusters=count,
        batch_size=max(count, int(config["prototypes"]["batch_size"])),
        max_iter=int(config["prototypes"]["max_iter"]),
        random_state=int(config["seed"]),
        n_init=3,
    )
    model.fit(embeddings[train_indices])
    # O(N*D), rather than materializing an O(N*K*D) distance tensor.
    assignments = model.predict(embeddings).astype(np.int64)
    assigned_centers = model.cluster_centers_[assignments]
    prototype_distance = np.linalg.norm(embeddings - assigned_centers, axis=1).astype(np.float64)
    scoring = config["scoring"]
    low_q = float(scoring["component_low_quantile"])
    high_q = float(scoring["component_high_quantile"])
    distance_scaler = QuantileScoreScaler.fit(prototype_distance[calibration_indices], low_q, high_q)
    reconstruction_scaler = QuantileScoreScaler.fit(reconstruction_error[calibration_indices], low_q, high_q)
    weights = scoring.get("local_weights", {"prototype": 0.5, "reconstruction": 0.5})
    prototype_weight = float(weights.get("prototype", 0.5))
    reconstruction_weight = float(weights.get("reconstruction", 0.5))
    weight_sum = prototype_weight + reconstruction_weight
    if weight_sum <= 0:
        raise ValueError("scoring.local_weights must have positive total weight")
    prototype_score = distance_scaler.transform(prototype_distance)
    reconstruction_score = reconstruction_scaler.transform(reconstruction_error)
    combined_local_score = (
        prototype_weight * prototype_score
        + reconstruction_weight * reconstruction_score
    ) / weight_sum
    anomaly_score = str(config["flow_model"].get("anomaly_score", "reconstruction_prototype"))
    if anomaly_score == "reconstruction_only":
        local_score = reconstruction_score
    elif anomaly_score == "reconstruction_prototype":
        local_score = combined_local_score
    else:
        raise ValueError(f"Unknown flow_model.anomaly_score: {anomaly_score}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        segment_ids=dataset.segment_ids,
        centers=model.cluster_centers_.astype(np.float32),
        assignments=assignments,
        prototype_distance=prototype_distance,
        prototype_score=prototype_score.astype(np.float64),
        reconstruction_score=reconstruction_score.astype(np.float64),
        combined_local_score=combined_local_score.astype(np.float64),
        local_score=local_score.astype(np.float64),
        distance_scale=np.asarray([distance_scaler.low, distance_scaler.high]),
        reconstruction_scale=np.asarray([reconstruction_scaler.low, reconstruction_scaler.high]),
        local_weights=np.asarray([prototype_weight / weight_sum, reconstruction_weight / weight_sum]),
        anomaly_score=np.asarray(anomaly_score),
    )
    result = {
        "path": str(output_path),
        "requested_prototypes": requested,
        "fitted_prototypes": count,
        "normal_train_segments": len(train_indices),
    }
    print(f"prototype fitting complete {result}")
    return result


def apply_existing(config: dict[str, Any], source_path: str | Path) -> dict[str, Any]:
    """Assign a test-only dataset to frozen normal prototypes without refitting."""
    dataset_path = resolve_path(config, config["data"]["processed_path"])
    metadata_path = resolve_path(config, config["data"]["metadata_path"])
    embeddings_path = resolve_path(config, config["runtime"]["embeddings_path"])
    output_path = resolve_path(config, config["runtime"]["prototypes_path"])
    assert dataset_path is not None and metadata_path is not None and embeddings_path is not None and output_path is not None
    dataset = FlowDataset(dataset_path, metadata_path)
    with np.load(embeddings_path, allow_pickle=False) as archive:
        require_alignment(dataset.segment_ids, archive["segment_ids"], "Embedding artifact")
        embeddings = archive["embeddings"].astype(np.float32)
        reconstruction_error = archive["reconstruction_error"].astype(np.float64)
    with np.load(Path(source_path), allow_pickle=False) as source:
        centers = source["centers"].astype(np.float32)
        distance_scale = source["distance_scale"].astype(np.float64)
        reconstruction_scale = source["reconstruction_scale"].astype(np.float64)
        local_weights = source["local_weights"].astype(np.float64)
    squared_norms = np.sum(embeddings * embeddings, axis=1, keepdims=True)
    center_norms = np.sum(centers * centers, axis=1)[None, :]
    squared_distances = np.maximum(0.0, squared_norms + center_norms - 2.0 * embeddings @ centers.T)
    assignments = squared_distances.argmin(axis=1).astype(np.int64)
    prototype_distance = np.sqrt(squared_distances[np.arange(len(dataset)), assignments]).astype(np.float64)
    distance_scaler = QuantileScoreScaler(float(distance_scale[0]), float(distance_scale[1]))
    reconstruction_scaler = QuantileScoreScaler(float(reconstruction_scale[0]), float(reconstruction_scale[1]))
    prototype_score = distance_scaler.transform(prototype_distance)
    reconstruction_score = reconstruction_scaler.transform(reconstruction_error)
    combined_local_score = (
        float(local_weights[0]) * prototype_score
        + float(local_weights[1]) * reconstruction_score
    )
    anomaly_score = str(config["flow_model"].get("anomaly_score", "reconstruction_prototype"))
    if anomaly_score == "reconstruction_only":
        local_score = reconstruction_score
    elif anomaly_score == "reconstruction_prototype":
        local_score = combined_local_score
    else:
        raise ValueError(f"Unknown flow_model.anomaly_score: {anomaly_score}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        segment_ids=dataset.segment_ids,
        centers=centers,
        assignments=assignments,
        prototype_distance=prototype_distance,
        prototype_score=prototype_score,
        reconstruction_score=reconstruction_score,
        combined_local_score=combined_local_score,
        local_score=local_score,
        distance_scale=distance_scale,
        reconstruction_scale=reconstruction_scale,
        local_weights=local_weights,
        anomaly_score=np.asarray(anomaly_score),
    )
    result = {"path": str(output_path), "segments": len(dataset), "frozen_prototypes": len(centers)}
    print(f"frozen prototype assignment complete {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--reuse-centers", default=None, help="Frozen training prototypes.npz")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.reuse_centers:
        apply_existing(config, args.reuse_centers)
    else:
        fit(config)


if __name__ == "__main__":
    main()
