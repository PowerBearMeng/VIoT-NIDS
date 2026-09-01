#!/usr/bin/env python3
"""Fit all deployment score scales and thresholds on normal calibration only."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from data.dataset import FlowDataset
from utils.config import load_config, resolve_path
from utils.full_inference import score_raw_components
from utils.io import write_json
from utils.scaling import EmpiricalUpperTail, QuantileScoreScaler
from utils.seed import choose_device, seed_everything
from utils.v4_inference import V4RawScores, score_v4_components, smooth_max


def _quantile_threshold(values: np.ndarray, false_positive_rate: float) -> float:
    return float(np.quantile(values, 1.0 - false_positive_rate, method="higher"))


def _calibrate_v3(
    config: dict[str, Any], raw: Any, normal_segments: int
) -> dict[str, Any]:
    scoring = config["scoring"]
    low_q = float(scoring["component_low_quantile"])
    high_q = float(scoring["component_high_quantile"])
    fpr = float(scoring["deployment_fpr"])
    tail_epsilon = float(scoring.get("tail_epsilon", 1e-12))
    tails = {
        "local": EmpiricalUpperTail.fit(raw.reconstruction_score, tail_epsilon),
        "pair_context": EmpiricalUpperTail.fit(raw.pair_context_score, tail_epsilon),
        "entity_context": EmpiricalUpperTail.fit(raw.entity_context_score, tail_epsilon),
        "context": EmpiricalUpperTail.fit(raw.context_score, tail_epsilon),
    }
    a_local = tails["local"].evidence(raw.reconstruction_score)
    a_pair = tails["pair_context"].evidence(raw.pair_context_score)
    a_entity_context = tails["entity_context"].evidence(raw.entity_context_score)
    a_context = tails["context"].evidence(raw.context_score)
    final = np.maximum(a_local, a_context)

    legacy_spatial_scaler = QuantileScoreScaler.fit(raw.spatial_score, low_q, high_q)
    legacy_spatial = legacy_spatial_scaler.transform(raw.spatial_score)
    legacy_weights = scoring.get("legacy_final_weights", {"local": 0.7, "spatial": 0.3})
    legacy_local_weight = float(legacy_weights.get("local", 0.7))
    legacy_spatial_weight = float(legacy_weights.get("spatial", 0.3))
    legacy_total = legacy_local_weight + legacy_spatial_weight
    if legacy_total <= 0:
        raise ValueError("scoring.legacy_final_weights must have positive total")
    legacy_local_weight /= legacy_total
    legacy_spatial_weight /= legacy_total
    old_weighted = (
        legacy_local_weight * raw.combined_local_score
        + legacy_spatial_weight * legacy_spatial
    )

    entity_temporal_scaler = QuantileScoreScaler.fit(raw.entity_state_scores, low_q, high_q)
    entity_temporal = entity_temporal_scaler.transform(raw.entity_score)
    ablation_scores = {
        "v2_reconstruction_only": raw.reconstruction_score,
        "v2_reconstruction_prototype": raw.combined_local_score,
        "v3_pair_mode_context_only": raw.pair_context_score,
        "v3_entity_mode_context_only": raw.entity_context_score,
        "v3_pair_entity_context": raw.context_score,
        "local_only": a_local,
        "context_only": a_context,
        "old_weighted_fusion": old_weighted,
        "v3_normal_tail_max": final,
    }
    thresholds = {
        name: _quantile_threshold(values, fpr) for name, values in ablation_scores.items()
    }
    thresholds.update(
        {
            "local": thresholds["local_only"],
            "context": thresholds["context_only"],
            "pair_context": _quantile_threshold(a_pair, fpr),
            "entity_context": _quantile_threshold(a_entity_context, fpr),
            "spatial": _quantile_threshold(legacy_spatial, fpr),
            "entity": _quantile_threshold(entity_temporal, fpr),
            "final": thresholds["v3_normal_tail_max"],
        }
    )
    return {
        "format_version": 3,
        "normal_calibration_segments": int(normal_segments),
        "normal_calibration_entity_states": int(len(raw.entity_state_scores)),
        "attack_samples_used": 0,
        "context_mode": "behavior_composition",
        "history": "frozen_train_reference",
        "fusion": "normal_tail_max",
        "tail_epsilon": tail_epsilon,
        "tail_references": {name: tail.state_dict() for name, tail in tails.items()},
        "legacy_spatial_scaler": legacy_spatial_scaler.state_dict(),
        # Compatibility alias for readers expecting the V1/V2 key.
        "spatial_scaler": legacy_spatial_scaler.state_dict(),
        "entity_scaler": entity_temporal_scaler.state_dict(),
        "legacy_final_weights": {
            "local": legacy_local_weight,
            "spatial": legacy_spatial_weight,
            "entity": 0.0,
        },
        "deployment_fpr": fpr,
        "thresholds": thresholds,
    }


def _calibrate_v4(
    config: dict[str, Any], raw: V4RawScores, normal_segments: int
) -> dict[str, Any]:
    """Fit continuous normal-only scales without empirical-rank saturation."""

    scoring = config["scoring"]
    low_q = float(scoring["component_low_quantile"])
    high_q = float(scoring["component_high_quantile"])
    fpr = float(scoring["deployment_fpr"])
    fusion_temperature = float(scoring.get("fusion_temperature", 1.0))
    scalers = {
        "local": QuantileScoreScaler.fit(raw.reconstruction_error, low_q, high_q),
        "pair_context": QuantileScoreScaler.fit(
            raw.pair_context_score, low_q, high_q
        ),
        "entity_context": QuantileScoreScaler.fit(
            raw.entity_context_score, low_q, high_q
        ),
        "context": QuantileScoreScaler.fit(raw.context_score, low_q, high_q),
    }
    scores = {
        "local": scalers["local"].transform(raw.reconstruction_error),
        "pair_context": scalers["pair_context"].transform(raw.pair_context_score),
        "entity_context": scalers["entity_context"].transform(
            raw.entity_context_score
        ),
        "context": scalers["context"].transform(raw.context_score),
    }
    scores["final"] = smooth_max(
        [scores["local"], scores["context"]], fusion_temperature
    )
    thresholds = {
        name: _quantile_threshold(values, fpr) for name, values in scores.items()
    }
    return {
        "format_version": 4,
        "normal_calibration_segments": int(normal_segments),
        "attack_samples_used": 0,
        "context_mode": "neural_intensity",
        "hard_mode_ids_used": False,
        "ports_used": False,
        "fusion": "robust_logsumexp",
        "fusion_temperature": fusion_temperature,
        "component_low_quantile": low_q,
        "component_high_quantile": high_q,
        "scalers": {name: scaler.state_dict() for name, scaler in scalers.items()},
        "deployment_fpr": fpr,
        "thresholds": thresholds,
    }


def calibrate(config: dict[str, Any], device_name: str | None = None) -> dict[str, Any]:
    seed_everything(int(config["seed"]))
    device = choose_device(device_name or config["runtime"].get("device"))
    dataset_path = resolve_path(config, config["data"]["processed_path"])
    metadata_path = resolve_path(config, config["data"]["metadata_path"])
    output_path = resolve_path(config, config["runtime"]["calibration_path"])
    assert dataset_path is not None and metadata_path is not None and output_path is not None
    dataset = FlowDataset(dataset_path, metadata_path)
    indices = dataset.require_normal("calibration", "deployment calibration")
    context_mode = str(config.get("context_model", {}).get("mode", "legacy_spatial"))
    if context_mode == "neural_intensity":
        raw_v4 = score_v4_components(config, dataset, indices, device)
        calibration = _calibrate_v4(config, raw_v4, len(indices))
        write_json(output_path, calibration)
        print(
            f"V4 calibration complete normal_segments={len(indices)} "
            f"target_fpr={calibration['deployment_fpr']:.4f} "
            f"final_threshold={calibration['thresholds']['final']:.6f} "
            f"output={output_path}"
        )
        return calibration
    raw = score_raw_components(config, dataset, indices, device)
    if context_mode == "behavior_composition":
        calibration = _calibrate_v3(config, raw, len(indices))
        write_json(output_path, calibration)
        print(
            f"V3 calibration complete normal_segments={len(indices)} target_fpr={calibration['deployment_fpr']:.4f} "
            f"final_threshold={calibration['thresholds']['final']:.6f} output={output_path}"
        )
        return calibration
    scoring = config["scoring"]
    low_q = float(scoring["component_low_quantile"])
    high_q = float(scoring["component_high_quantile"])
    spatial_scaler = QuantileScoreScaler.fit(raw.spatial_score, low_q, high_q)
    entity_scaler = QuantileScoreScaler.fit(raw.entity_state_scores, low_q, high_q)
    spatial = spatial_scaler.transform(raw.spatial_score)
    entity = entity_scaler.transform(raw.entity_score)
    weights_raw = scoring["final_weights"]
    weights = {name: float(weights_raw.get(name, 0.0)) for name in ("local", "spatial", "entity")}
    weight_sum = sum(weights.values())
    weights = {name: value / weight_sum for name, value in weights.items()}
    final = weights["local"] * raw.local_score + weights["spatial"] * spatial
    if weights["entity"]:
        final += weights["entity"] * entity
    fpr = float(scoring["deployment_fpr"])
    calibration = {
        "format_version": 1,
        "normal_calibration_segments": int(len(indices)),
        "normal_calibration_entity_states": int(len(raw.entity_state_scores)),
        "attack_samples_used": 0,
        "component_low_quantile": low_q,
        "component_high_quantile": high_q,
        "spatial_scaler": spatial_scaler.state_dict(),
        "entity_scaler": entity_scaler.state_dict(),
        "final_weights": weights,
        "deployment_fpr": fpr,
        "thresholds": {
            "local": _quantile_threshold(raw.local_score, fpr),
            "spatial": _quantile_threshold(spatial, fpr),
            "entity": _quantile_threshold(entity, fpr),
            "final": _quantile_threshold(final, fpr),
        },
    }
    write_json(output_path, calibration)
    print(
        f"calibration complete normal_segments={len(indices)} target_fpr={fpr:.4f} "
        f"final_threshold={calibration['thresholds']['final']:.6f} output={output_path}"
    )
    return calibration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    calibrate(load_config(args.config), args.device)


if __name__ == "__main__":
    main()
