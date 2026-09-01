#!/usr/bin/env python3
"""Score flow segments, write per-flow results, and compute NIDS metrics."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import numpy as np

from data.dataset import FlowDataset
from utils.config import load_config, resolve_path
from utils.full_inference import RawScores, score_raw_components
from utils.io import load_json, write_json
from utils.metrics import detection_metrics
from utils.scaling import EmpiricalUpperTail, QuantileScoreScaler
from utils.seed import choose_device, seed_everything


def _binary_labels(dataset: FlowDataset, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = np.isin(dataset.labels[indices], list(dataset.normal_label_ids))
    attack_ids = set(int(value) for value in dataset.metadata["attack_label_ids"])
    attack = np.isin(dataset.labels[indices], list(attack_ids))
    keep = normal | attack
    return keep, attack[keep].astype(np.int64)


def _write_scores(
    path: Path,
    dataset: FlowDataset,
    raw: RawScores,
    spatial: np.ndarray,
    entity_a: np.ndarray,
    entity_b: np.ndarray,
    entity: np.ndarray,
    final: np.ndarray,
    threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    label_names = dataset.metadata["label_names"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "segment_id", "flow_id", "capture", "window_index", "segment_start",
                "endpoint_a_ip", "endpoint_a_port", "endpoint_b_ip", "endpoint_b_port", "protocol",
                "label", "packet_count", "byte_count", "prototype_distance", "reconstruction_error",
                "local_anomaly", "spatial_anomaly", "entity_a_anomaly", "entity_b_anomaly",
                "entity_anomaly", "final_anomaly", "reliability", "context_edges_a", "context_edges_b",
                "deployment_prediction",
            ]
        )
        for row, index in enumerate(raw.indices.tolist()):
            label_id = int(dataset.labels[index])
            label = label_names[label_id] if label_id >= 0 else "__unlabeled__"
            writer.writerow(
                [
                    dataset.segment_ids[index], dataset.flow_ids[index], dataset.captures[index],
                    int(dataset.window_indices[index]), float(dataset.segment_starts[index]),
                    dataset.endpoint_a_ips[index], int(dataset.endpoint_a_ports[index]),
                    dataset.endpoint_b_ips[index], int(dataset.endpoint_b_ports[index]), dataset.protocols[index],
                    label, int(dataset.packet_counts[index]), int(dataset.byte_counts[index]),
                    float(raw.prototype_distance[row]), float(raw.reconstruction_error[row]),
                    float(raw.local_score[row]), float(spatial[row]), float(entity_a[row]),
                    float(entity_b[row]), float(entity[row]), float(final[row]),
                    float(raw.reliability[row]), int(raw.context_counts[row, 0]), int(raw.context_counts[row, 1]),
                    int(final[row] > threshold),
                ]
            )


def _write_entity_scores(
    path: Path, raw: RawScores, scaler: QuantileScoreScaler
) -> None:
    normalized = scaler.transform(raw.entity_state_scores)
    sequences = raw.entity_sequences
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["capture_id", "window_index", "entity_ip", "entity_anomaly_raw", "entity_anomaly"])
        for row in range(len(sequences)):
            writer.writerow(
                [
                    sequences.capture_ids[row], int(sequences.window_indices[row]), sequences.entity_ips[row],
                    float(raw.entity_state_scores[row]), float(normalized[row]),
                ]
            )


def _write_v3_scores(
    path: Path,
    dataset: FlowDataset,
    raw: RawScores,
    scores: dict[str, np.ndarray],
    threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    label_names = dataset.metadata["label_names"]
    columns = [
        "segment_id", "flow_id", "capture", "window_index", "segment_start",
        "endpoint_a_ip", "endpoint_a_port", "endpoint_b_ip", "endpoint_b_port", "protocol",
        "label", "packet_count", "byte_count", "mode_id", "prototype_distance",
        "prototype_score", "reconstruction_error", "reconstruction_score",
        "combined_local_score", "pair_mode_count", "entity_a_mode_count", "entity_b_mode_count",
        "pair_context_deviation", "entity_a_context_deviation", "entity_b_context_deviation",
        "entity_context_deviation", "context_deviation", "local_tail_evidence",
        "pair_context_tail_evidence", "entity_context_tail_evidence", "context_tail_evidence",
        "legacy_spatial_anomaly", "entity_temporal_anomaly", "old_weighted_fusion",
        "final_anomaly", "deployment_prediction",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row, index in enumerate(raw.indices.tolist()):
            label_id = int(dataset.labels[index])
            label = label_names[label_id] if label_id >= 0 else "__unlabeled__"
            writer.writerow(
                [
                    dataset.segment_ids[index], dataset.flow_ids[index], dataset.captures[index],
                    int(dataset.window_indices[index]), float(dataset.segment_starts[index]),
                    dataset.endpoint_a_ips[index], int(dataset.endpoint_a_ports[index]),
                    dataset.endpoint_b_ips[index], int(dataset.endpoint_b_ports[index]),
                    dataset.protocols[index], label, int(dataset.packet_counts[index]),
                    int(dataset.byte_counts[index]), int(raw.mode_ids[row]),
                    float(raw.prototype_distance[row]), float(raw.prototype_score[row]),
                    float(raw.reconstruction_error[row]), float(raw.reconstruction_score[row]),
                    float(raw.combined_local_score[row]), int(raw.pair_mode_count[row]),
                    int(raw.entity_a_mode_count[row]), int(raw.entity_b_mode_count[row]),
                    float(raw.pair_context_score[row]), float(raw.entity_a_context_score[row]),
                    float(raw.entity_b_context_score[row]), float(raw.entity_context_score[row]),
                    float(raw.context_score[row]), float(scores["local_only"][row]),
                    float(scores["pair_context_evidence"][row]),
                    float(scores["entity_context_evidence"][row]),
                    float(scores["context_only"][row]), float(scores["legacy_spatial"][row]),
                    float(scores["entity_temporal"][row]), float(scores["old_weighted_fusion"][row]),
                    float(scores["v3_normal_tail_max"][row]),
                    int(scores["v3_normal_tail_max"][row] > threshold),
                ]
            )


def _v3_scores(raw: RawScores, calibration: dict[str, Any]) -> dict[str, np.ndarray]:
    tails = {
        name: EmpiricalUpperTail.from_state_dict(state)
        for name, state in calibration["tail_references"].items()
    }
    a_local = tails["local"].evidence(raw.reconstruction_score)
    a_pair = tails["pair_context"].evidence(raw.pair_context_score)
    a_entity_context = tails["entity_context"].evidence(raw.entity_context_score)
    a_context = tails["context"].evidence(raw.context_score)
    legacy_spatial_scaler = QuantileScoreScaler.from_state_dict(
        calibration["legacy_spatial_scaler"]
    )
    legacy_spatial = legacy_spatial_scaler.transform(raw.spatial_score)
    entity_scaler = QuantileScoreScaler.from_state_dict(calibration["entity_scaler"])
    entity_temporal = entity_scaler.transform(raw.entity_score)
    weights = calibration["legacy_final_weights"]
    old_weighted = (
        float(weights["local"]) * raw.combined_local_score
        + float(weights["spatial"]) * legacy_spatial
    )
    return {
        "v2_reconstruction_only": raw.reconstruction_score,
        "v2_reconstruction_prototype": raw.combined_local_score,
        "v3_pair_mode_context_only": raw.pair_context_score,
        "v3_entity_mode_context_only": raw.entity_context_score,
        "v3_pair_entity_context": raw.context_score,
        "local_only": a_local,
        "context_only": a_context,
        "old_weighted_fusion": old_weighted,
        "v3_normal_tail_max": np.maximum(a_local, a_context),
        "pair_context_evidence": a_pair,
        "entity_context_evidence": a_entity_context,
        "legacy_spatial": legacy_spatial,
        "entity_temporal": entity_temporal,
    }


def _evaluate_v3(
    config: dict[str, Any],
    dataset: FlowDataset,
    indices: np.ndarray,
    raw: RawScores,
    calibration: dict[str, Any],
    scores_path: Path,
    metrics_path: Path,
    elapsed: float,
    device: Any,
    split: str,
) -> dict[str, Any]:
    scores = _v3_scores(raw, calibration)
    thresholds = calibration["thresholds"]
    final_threshold = float(thresholds["v3_normal_tail_max"])
    _write_v3_scores(scores_path, dataset, raw, scores, final_threshold)
    entity_scaler = QuantileScoreScaler.from_state_dict(calibration["entity_scaler"])
    _write_entity_scores(
        scores_path.with_name(scores_path.stem + "_entities.csv"), raw, entity_scaler
    )
    keep, labels = _binary_labels(dataset, indices)
    ablation_names = [
        "v2_reconstruction_only",
        "v2_reconstruction_prototype",
        "v3_pair_mode_context_only",
        "v3_entity_mode_context_only",
        "v3_pair_entity_context",
        "local_only",
        "context_only",
        "old_weighted_fusion",
        "v3_normal_tail_max",
    ]
    ablations = {
        name: detection_metrics(
            labels, scores[name][keep], float(thresholds[name])
        )
        for name in ablation_names
    }
    pair_metrics = detection_metrics(
        labels, scores["pair_context_evidence"][keep], float(thresholds["pair_context"])
    )
    entity_context_metrics = detection_metrics(
        labels,
        scores["entity_context_evidence"][keep],
        float(thresholds["entity_context"]),
    )
    entity_temporal_metrics = detection_metrics(
        labels, scores["entity_temporal"][keep], float(thresholds["entity"])
    )
    legacy_spatial_metrics = detection_metrics(
        labels, scores["legacy_spatial"][keep], float(thresholds["spatial"])
    )
    embeddings_path = resolve_path(config, config["runtime"]["embeddings_path"])
    assert embeddings_path is not None
    with np.load(embeddings_path, allow_pickle=False) as artifact:
        flow_seconds = float(artifact["inference_seconds"]) if "inference_seconds" in artifact else None
        flow_rate = float(artifact["segments_per_second"]) if "segments_per_second" in artifact else None
    metrics: dict[str, Any] = {
        "format_version": 3,
        "split": split,
        "threshold_source": "normal calibration empirical upper tail",
        "target_deployment_fpr": float(calibration["deployment_fpr"]),
        "fusion": "normal_tail_max",
        "final": ablations["v3_normal_tail_max"],
        "local": ablations["local_only"],
        "context": ablations["context_only"],
        "pair_context": pair_metrics,
        "entity_context": entity_context_metrics,
        "legacy_spatial": legacy_spatial_metrics,
        "entity_temporal": entity_temporal_metrics,
        "ablations": ablations,
        "runtime": {
            "scored_segments": int(len(indices)),
            "context_scoring_seconds": float(elapsed),
            "context_segments_per_second": float(len(indices) / max(elapsed, 1e-12)),
            "flow_model_inference_seconds_all_segments": flow_seconds,
            "flow_model_segments_per_second": flow_rate,
            "pcap_preprocessing": dataset.metadata.get("preprocessing_runtime"),
            "device": str(device),
        },
        "outputs": {
            "flow_scores": str(scores_path),
            "entity_scores": str(scores_path.with_name(scores_path.stem + "_entities.csv")),
        },
    }
    per_dataset: dict[str, Any] = {}
    selected_dataset_names = dataset.dataset_names[indices]
    for dataset_name in sorted(set(selected_dataset_names.tolist())):
        positions = np.flatnonzero(selected_dataset_names == dataset_name)
        subset_indices = indices[positions]
        subset_keep, subset_labels = _binary_labels(dataset, subset_indices)
        selected = positions[subset_keep]
        per_dataset[str(dataset_name)] = {
            "final": detection_metrics(
                subset_labels,
                scores["v3_normal_tail_max"][selected],
                float(thresholds["v3_normal_tail_max"]),
            ),
            "ablations": {
                name: detection_metrics(
                    subset_labels, scores[name][selected], float(thresholds[name])
                )
                for name in ablation_names
            },
        }
    metrics["per_dataset"] = per_dataset
    write_json(metrics_path, metrics)
    final_metrics = metrics["final"]
    print(
        f"V3 evaluation segments={len(indices)} AUROC={final_metrics['AUROC']} "
        f"AUPRC={final_metrics['AUPRC']} FPR={final_metrics['FPR']:.6f} "
        f"TPR={final_metrics['TPR']:.6f}"
    )
    print(f"metrics={metrics_path} scores={scores_path}")
    return metrics


def evaluate(
    config: dict[str, Any], split: str = "test", device_name: str | None = None
) -> dict[str, Any]:
    seed_everything(int(config["seed"]))
    device = choose_device(device_name or config["runtime"].get("device"))
    dataset_path = resolve_path(config, config["data"]["processed_path"])
    metadata_path = resolve_path(config, config["data"]["metadata_path"])
    calibration_path = resolve_path(config, config["runtime"]["calibration_path"])
    scores_path = resolve_path(config, config["runtime"]["scores_path"])
    metrics_path = resolve_path(config, config["runtime"]["metrics_path"])
    assert dataset_path is not None and metadata_path is not None and calibration_path is not None and scores_path is not None and metrics_path is not None
    dataset = FlowDataset(dataset_path, metadata_path)
    indices = dataset.indices(split)
    if not len(indices):
        raise ValueError(f"Split {split!r} contains no segments")
    calibration = load_json(calibration_path)
    started = time.perf_counter()
    raw = score_raw_components(config, dataset, indices, device)
    elapsed = time.perf_counter() - started
    if int(calibration.get("format_version", 1)) >= 3:
        return _evaluate_v3(
            config,
            dataset,
            indices,
            raw,
            calibration,
            scores_path,
            metrics_path,
            elapsed,
            device,
            split,
        )
    spatial_scaler = QuantileScoreScaler.from_state_dict(calibration["spatial_scaler"])
    entity_scaler = QuantileScoreScaler.from_state_dict(calibration["entity_scaler"])
    spatial = spatial_scaler.transform(raw.spatial_score)
    entity_a = entity_scaler.transform(raw.entity_a_score)
    entity_b = entity_scaler.transform(raw.entity_b_score)
    entity = np.maximum(entity_a, entity_b)
    weights = calibration["final_weights"]
    final = float(weights["local"]) * raw.local_score + float(weights["spatial"]) * spatial
    if float(weights["entity"]):
        final += float(weights["entity"]) * entity
    threshold = float(calibration["thresholds"]["final"])
    _write_scores(scores_path, dataset, raw, spatial, entity_a, entity_b, entity, final, threshold)
    _write_entity_scores(scores_path.with_name(scores_path.stem + "_entities.csv"), raw, entity_scaler)
    keep, labels = _binary_labels(dataset, indices)
    embeddings_path = resolve_path(config, config["runtime"]["embeddings_path"])
    assert embeddings_path is not None
    with np.load(embeddings_path, allow_pickle=False) as embedding_artifact:
        flow_seconds = float(embedding_artifact["inference_seconds"]) if "inference_seconds" in embedding_artifact else None
        flow_rate = float(embedding_artifact["segments_per_second"]) if "segments_per_second" in embedding_artifact else None
    metrics = {
        "split": split,
        "threshold_source": "normal calibration quantile",
        "target_deployment_fpr": float(calibration["deployment_fpr"]),
        "final": detection_metrics(labels, final[keep], threshold),
        "local": detection_metrics(labels, raw.local_score[keep], float(calibration["thresholds"]["local"])),
        "spatial": detection_metrics(labels, spatial[keep], float(calibration["thresholds"]["spatial"])),
        "entity": detection_metrics(labels, entity[keep], float(calibration["thresholds"]["entity"])),
        "runtime": {
            "scored_segments": int(len(indices)),
            "context_scoring_seconds": float(elapsed),
            "context_segments_per_second": float(len(indices) / max(elapsed, 1e-12)),
            "flow_model_inference_seconds_all_segments": flow_seconds,
            "flow_model_segments_per_second": flow_rate,
            "pcap_preprocessing": dataset.metadata.get("preprocessing_runtime"),
            "device": str(device),
            "note": "stage timings are separate; flow inference includes TCN embedding and masked reconstruction, context timing uses cached embeddings",
        },
        "outputs": {
            "flow_scores": str(scores_path),
            "entity_scores": str(scores_path.with_name(scores_path.stem + "_entities.csv")),
        },
    }
    per_dataset: dict[str, Any] = {}
    for dataset_name in sorted(set(dataset.dataset_names[indices].tolist())):
        positions = np.flatnonzero(dataset.dataset_names[indices] == dataset_name)
        subset_indices = indices[positions]
        subset_keep, subset_labels = _binary_labels(dataset, subset_indices)
        selected = positions[subset_keep]
        per_dataset[str(dataset_name)] = {
            "final": detection_metrics(subset_labels, final[selected], threshold),
            "local": detection_metrics(
                subset_labels,
                raw.local_score[selected],
                float(calibration["thresholds"]["local"]),
            ),
            "spatial": detection_metrics(
                subset_labels,
                spatial[selected],
                float(calibration["thresholds"]["spatial"]),
            ),
            "entity": detection_metrics(
                subset_labels,
                entity[selected],
                float(calibration["thresholds"]["entity"]),
            ),
        }
    metrics["per_dataset"] = per_dataset
    write_json(metrics_path, metrics)
    final_metrics = metrics["final"]
    print(
        f"evaluation split={split} segments={len(indices)} AUROC={final_metrics['AUROC']} "
        f"AUPRC={final_metrics['AUPRC']} FPR={final_metrics['FPR']:.6f} "
        f"TPR={final_metrics['TPR']:.6f} context_throughput={metrics['runtime']['context_segments_per_second']:.1f} segments/s"
    )
    print(f"metrics={metrics_path} scores={scores_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="test", choices=["train", "calibration", "validation", "test", "all"])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    evaluate(load_config(args.config), args.split, args.device)


if __name__ == "__main__":
    main()
