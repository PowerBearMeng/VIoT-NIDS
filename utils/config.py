"""YAML loading, validation, and project-relative path resolution."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")
    config = deepcopy(raw)
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(config_path.parent.parent)
    validate_config(config)
    return config


def resolve_path(config: dict[str, Any], value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config["_project_root"]) / path
    return path.resolve()


def validate_config(config: dict[str, Any]) -> None:
    for section in ("data", "flow_model", "scoring", "runtime"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Missing configuration section: {section}")
    behavior = config.get("context_model")
    if behavior is not None and not isinstance(behavior, dict):
        raise ValueError("context_model must be a mapping")
    behavior_mode = str(
        behavior.get("mode", "legacy_spatial")
        if isinstance(behavior, dict)
        else "legacy_spatial"
    ).lower()
    if behavior_mode != "neural_intensity":
        for section in ("prototypes", "entity_model", "spatial_model"):
            if section not in config or not isinstance(config[section], dict):
                raise ValueError(f"Missing configuration section: {section}")
    window = float(config["data"]["flow_window_seconds"])
    micro = float(config["data"]["micro_bin_seconds"])
    ratio = window / micro
    if window <= 0 or micro <= 0 or abs(ratio - round(ratio)) > 1e-8:
        raise ValueError("flow_window_seconds must be a positive multiple of micro_bin_seconds")
    if int(round(ratio)) < 2:
        raise ValueError("Each flow window must contain at least two micro-bins")
    flow_orientation = str(config["data"].get("flow_orientation", "bidirectional")).lower()
    if flow_orientation not in {"bidirectional", "directional"}:
        raise ValueError("data.flow_orientation must be bidirectional or directional")
    window_alignment = str(config["data"].get("window_alignment", "capture")).lower()
    if window_alignment not in {"capture", "epoch"}:
        raise ValueError("data.window_alignment must be capture or epoch")
    feature_profile = str(
        config["data"].get("microbin_feature_profile", "bidirectional_basic_v1")
    ).lower()
    if feature_profile not in {"bidirectional_basic_v1", "directional_iat_v1"}:
        raise ValueError(
            "data.microbin_feature_profile must be bidirectional_basic_v1 or "
            "directional_iat_v1"
        )
    if feature_profile == "directional_iat_v1" and flow_orientation != "directional":
        raise ValueError("directional_iat_v1 requires data.flow_orientation=directional")
    if not 0.0 < float(config["flow_model"]["mask_ratio"]) < 1.0:
        raise ValueError("flow_model.mask_ratio must be between zero and one")
    architecture = str(config["flow_model"].get("architecture", "v1")).lower()
    if architecture not in {"v1", "v2"}:
        raise ValueError("flow_model.architecture must be v1 or v2")
    anomaly_score = str(
        config["flow_model"].get("anomaly_score", "reconstruction_prototype")
    ).lower()
    if anomaly_score not in {"reconstruction_only", "reconstruction_prototype"}:
        raise ValueError(
            "flow_model.anomaly_score must be reconstruction_only or "
            "reconstruction_prototype"
        )
    if int(config["flow_model"].get("score_mask_rounds", 1)) <= 0:
        raise ValueError("flow_model.score_mask_rounds must be positive")
    active_weight = float(config["flow_model"].get("active_error_weight", 0.5))
    if not 0.0 <= active_weight <= 1.0:
        raise ValueError("flow_model.active_error_weight must be in [0,1]")
    if float(config["flow_model"].get("nonempty_loss_weight", 1.0)) < 1.0:
        raise ValueError("flow_model.nonempty_loss_weight must be at least one")
    if int(config["flow_model"]["kernel_size"]) < 1 or int(config["flow_model"]["kernel_size"]) % 2 == 0:
        raise ValueError("flow_model.kernel_size must be a positive odd integer")
    if behavior_mode != "neural_intensity":
        if int(config["prototypes"]["count"]) <= 0:
            raise ValueError("prototypes.count must be positive")
        spatial_context_mode = str(
            config["spatial_model"].get("context_mode", "current_window")
        ).lower()
        if spatial_context_mode not in {"current_window", "historical"}:
            raise ValueError(
                "spatial_model.context_mode must be current_window or historical"
            )
        if spatial_context_mode == "historical":
            if not 0.0 < float(
                config["spatial_model"].get("state_update_rate", 0.1)
            ) <= 1.0:
                raise ValueError("spatial_model.state_update_rate must be in (0,1]")
            if float(config["spatial_model"].get("history_beta", 1.0)) < 0.0:
                raise ValueError("spatial_model.history_beta cannot be negative")
            if float(
                config["spatial_model"].get("multiplicity_gamma", 0.05)
            ) < 0.0:
                raise ValueError("spatial_model.multiplicity_gamma cannot be negative")
    if behavior is not None:
        if behavior_mode not in {
            "legacy_spatial",
            "behavior_composition",
            "neural_intensity",
        }:
            raise ValueError(
                "context_model.mode must be legacy_spatial, behavior_composition, "
                "or neural_intensity"
            )
        if behavior_mode == "behavior_composition":
            if str(behavior.get("history", "frozen_train_reference")) != "frozen_train_reference":
                raise ValueError("V3.0 supports only context_model.history=frozen_train_reference")
            if not bool(behavior.get("positive_deviation_only", True)):
                raise ValueError("V3.0 requires positive_deviation_only=true")
            if float(behavior.get("epsilon", 1e-3)) <= 0.0:
                raise ValueError("context_model.epsilon must be positive")
        if behavior_mode == "neural_intensity":
            if flow_orientation != "directional" or feature_profile != "directional_iat_v1":
                raise ValueError(
                    "V4 neural_intensity requires directional Flow and "
                    "directional_iat_v1 features"
                )
            if int(behavior.get("latent_channels", 0)) < 2:
                raise ValueError("context_model.latent_channels must be at least two")
            if int(behavior.get("hidden_dim", 0)) <= 0:
                raise ValueError("context_model.hidden_dim must be positive")
            if float(behavior.get("assignment_temperature", 0.0)) <= 0.0:
                raise ValueError("context_model.assignment_temperature must be positive")
            if float(behavior.get("scope_temperature", 0.0)) <= 0.0:
                raise ValueError("context_model.scope_temperature must be positive")
            if not bool(behavior.get("pair_enabled", True)) and not bool(
                behavior.get("entity_enabled", True)
            ):
                raise ValueError("V4 requires pair_enabled or entity_enabled")
    fpr = float(config["scoring"]["deployment_fpr"])
    if not 0.0 < fpr < 1.0:
        raise ValueError("scoring.deployment_fpr must be between zero and one")
    fusion = str(config["scoring"].get("fusion", "weighted"))
    if fusion not in {"weighted", "normal_tail_max", "robust_logsumexp"}:
        raise ValueError(
            "scoring.fusion must be weighted, normal_tail_max, or robust_logsumexp"
        )
    if fusion == "normal_tail_max" and float(config["scoring"].get("tail_epsilon", 1e-12)) <= 0:
        raise ValueError("scoring.tail_epsilon must be positive")
    if fusion == "weighted":
        weights = config["scoring"].get("final_weights")
        if not isinstance(weights, dict):
            raise ValueError("weighted fusion requires scoring.final_weights")
        if any(
            float(weights.get(name, 0.0)) < 0
            for name in ("local", "spatial", "entity")
        ):
            raise ValueError("Final score weights cannot be negative")
        if sum(
            float(weights.get(name, 0.0))
            for name in ("local", "spatial", "entity")
        ) <= 0:
            raise ValueError("At least one final score weight must be positive")
    if isinstance(behavior, dict) and str(behavior.get("mode", "legacy_spatial")) == "behavior_composition":
        if fusion != "normal_tail_max":
            raise ValueError("V3 behavior_composition requires scoring.fusion=normal_tail_max")
        if float(config["scoring"].get("entity_weight", 0.0)) != 0.0:
            raise ValueError("V3.0 entity temporal score must not enter flow final score")
    if isinstance(behavior, dict) and str(
        behavior.get("mode", "legacy_spatial")
    ) == "neural_intensity":
        if fusion != "robust_logsumexp":
            raise ValueError(
                "V4 neural_intensity requires scoring.fusion=robust_logsumexp"
            )
        if float(config["scoring"].get("fusion_temperature", 0.0)) <= 0.0:
            raise ValueError("scoring.fusion_temperature must be positive")
