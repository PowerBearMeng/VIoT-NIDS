"""Design V4 continuous-context scoring and metric output."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data.dataset import FlowDataset
from utils.config import resolve_path
from utils.io import write_json
from utils.metrics import detection_metrics
from utils.scaling import QuantileScoreScaler
from utils.v4_inference import V4RawScores, score_v4_components, smooth_max


def _binary_labels(
    dataset: FlowDataset, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    normal = np.isin(dataset.labels[indices], list(dataset.normal_label_ids))
    attack_ids = set(int(value) for value in dataset.metadata["attack_label_ids"])
    attack = np.isin(dataset.labels[indices], list(attack_ids))
    keep = normal | attack
    return keep, attack[keep].astype(np.int64)


def _scaled_scores(
    raw: V4RawScores, calibration: dict[str, Any]
) -> dict[str, np.ndarray]:
    scalers = {
        name: QuantileScoreScaler.from_state_dict(state)
        for name, state in calibration["scalers"].items()
    }
    scores = {
        "local": scalers["local"].transform(raw.reconstruction_error),
        "pair_context": scalers["pair_context"].transform(
            raw.pair_context_score
        ),
        "entity_context": scalers["entity_context"].transform(
            raw.entity_context_score
        ),
        "context": scalers["context"].transform(raw.context_score),
    }
    scores["final"] = smooth_max(
        [scores["local"], scores["context"]],
        float(calibration["fusion_temperature"]),
    )
    return scores


def _write_scores(
    path: Path,
    dataset: FlowDataset,
    raw: V4RawScores,
    scores: dict[str, np.ndarray],
    threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    label_names = dataset.metadata["label_names"]
    columns = [
        "segment_id",
        "flow_id",
        "capture",
        "window_index",
        "segment_start",
        "endpoint_a_ip",
        "endpoint_a_port",
        "endpoint_b_ip",
        "endpoint_b_port",
        "protocol",
        "label",
        "packet_count",
        "byte_count",
        "reconstruction_error",
        "local_anomaly",
        "pair_soft_log_mass",
        "entity_a_soft_log_mass",
        "entity_b_soft_log_mass",
        "pair_expected_log_mass",
        "pair_expected_scale",
        "entity_expected_log_mass",
        "entity_expected_scale",
        "pair_context_raw",
        "entity_a_context_raw",
        "entity_b_context_raw",
        "entity_context_raw",
        "context_raw",
        "pair_context_anomaly",
        "entity_context_anomaly",
        "context_anomaly",
        "behavior_assignment_entropy",
        "behavior_assignment_peak",
        "final_anomaly",
        "deployment_prediction",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row, index in enumerate(raw.indices.tolist()):
            label_id = int(dataset.labels[index])
            label = label_names[label_id] if label_id >= 0 else "__unlabeled__"
            writer.writerow(
                [
                    dataset.segment_ids[index],
                    dataset.flow_ids[index],
                    dataset.captures[index],
                    int(dataset.window_indices[index]),
                    float(dataset.segment_starts[index]),
                    dataset.endpoint_a_ips[index],
                    int(dataset.endpoint_a_ports[index]),
                    dataset.endpoint_b_ips[index],
                    int(dataset.endpoint_b_ports[index]),
                    dataset.protocols[index],
                    label,
                    int(dataset.packet_counts[index]),
                    int(dataset.byte_counts[index]),
                    float(raw.reconstruction_error[row]),
                    float(scores["local"][row]),
                    float(raw.pair_log_mass[row]),
                    float(raw.entity_a_log_mass[row]),
                    float(raw.entity_b_log_mass[row]),
                    float(raw.pair_expected_mean[row]),
                    float(raw.pair_expected_scale[row]),
                    float(raw.entity_expected_mean[row]),
                    float(raw.entity_expected_scale[row]),
                    float(raw.pair_context_score[row]),
                    float(raw.entity_a_context_score[row]),
                    float(raw.entity_b_context_score[row]),
                    float(raw.entity_context_score[row]),
                    float(raw.context_score[row]),
                    float(scores["pair_context"][row]),
                    float(scores["entity_context"][row]),
                    float(scores["context"][row]),
                    float(raw.assignment_entropy[row]),
                    float(raw.assignment_peak[row]),
                    float(scores["final"][row]),
                    int(scores["final"][row] > threshold),
                ]
            )


def evaluate_v4(
    config: dict[str, Any],
    dataset: FlowDataset,
    indices: np.ndarray,
    calibration: dict[str, Any],
    scores_path: Path,
    metrics_path: Path,
    device: torch.device,
    split: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    raw = score_v4_components(config, dataset, indices, device)
    context_seconds = time.perf_counter() - started
    scores = _scaled_scores(raw, calibration)
    thresholds = calibration["thresholds"]
    _write_scores(
        scores_path, dataset, raw, scores, float(thresholds["final"])
    )
    keep, labels = _binary_labels(dataset, indices)
    component_names = ["local", "pair_context", "entity_context", "context", "final"]
    component_metrics = {
        name: detection_metrics(
            labels, scores[name][keep], float(thresholds[name])
        )
        for name in component_names
    }
    embeddings_path = resolve_path(config, config["runtime"]["embeddings_path"])
    assert embeddings_path is not None
    with np.load(embeddings_path, allow_pickle=False) as artifact:
        flow_seconds = (
            float(artifact["inference_seconds"])
            if "inference_seconds" in artifact
            else None
        )
        flow_rate = (
            float(artifact["segments_per_second"])
            if "segments_per_second" in artifact
            else None
        )
    metrics: dict[str, Any] = {
        "format_version": 4,
        "split": split,
        "threshold_source": "normal calibration continuous quantile scaling",
        "target_deployment_fpr": float(calibration["deployment_fpr"]),
        "context_model": "continuous_neural_intensity",
        "hard_mode_ids_used": False,
        "ports_used_for_scoring": False,
        "fusion": "robust_logsumexp",
        "final": component_metrics["final"],
        "local": component_metrics["local"],
        "pair_context": component_metrics["pair_context"],
        "entity_context": component_metrics["entity_context"],
        "context": component_metrics["context"],
        "ablations": component_metrics,
        "score_resolution": {
            name: {
                "unique_values": int(len(np.unique(values))),
                "largest_tie": int(
                    np.unique(values, return_counts=True)[1].max(initial=0)
                ),
            }
            for name, values in scores.items()
        },
        "runtime": {
            "scored_segments": int(len(indices)),
            "context_scoring_seconds": float(context_seconds),
            "context_segments_per_second": float(
                len(indices) / max(context_seconds, 1e-12)
            ),
            "flow_model_inference_seconds_all_segments": flow_seconds,
            "flow_model_segments_per_second": flow_rate,
            "pcap_preprocessing": dataset.metadata.get("preprocessing_runtime"),
            "device": str(device),
        },
        "outputs": {"flow_scores": str(scores_path)},
    }
    per_dataset: dict[str, Any] = {}
    selected_dataset_names = dataset.dataset_names[indices]
    for dataset_name in sorted(set(selected_dataset_names.tolist())):
        positions = np.flatnonzero(selected_dataset_names == dataset_name)
        subset_indices = indices[positions]
        subset_keep, subset_labels = _binary_labels(dataset, subset_indices)
        selected = positions[subset_keep]
        per_dataset[str(dataset_name)] = {
            name: detection_metrics(
                subset_labels,
                scores[name][selected],
                float(thresholds[name]),
            )
            for name in component_names
        }
    metrics["per_dataset"] = per_dataset
    write_json(metrics_path, metrics)
    final = metrics["final"]
    print(
        f"V4 evaluation segments={len(indices)} AUROC={final['AUROC']} "
        f"AUPRC={final['AUPRC']} EER={final['EER']} "
        f"FPR={final['FPR']:.6f} TPR={final['TPR']:.6f}"
    )
    print(f"metrics={metrics_path} scores={scores_path}")
    return metrics
