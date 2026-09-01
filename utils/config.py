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
    for section in (
        "data",
        "flow_model",
        "prototypes",
        "entity_model",
        "spatial_model",
        "scoring",
        "runtime",
    ):
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
    if int(config["prototypes"]["count"]) <= 0:
        raise ValueError("prototypes.count must be positive")
    context_mode = str(config["spatial_model"].get("context_mode", "current_window")).lower()
    if context_mode not in {"current_window", "historical"}:
        raise ValueError("spatial_model.context_mode must be current_window or historical")
    if context_mode == "historical":
        if not 0.0 < float(config["spatial_model"].get("state_update_rate", 0.1)) <= 1.0:
            raise ValueError("spatial_model.state_update_rate must be in (0,1]")
        if float(config["spatial_model"].get("history_beta", 1.0)) < 0.0:
            raise ValueError("spatial_model.history_beta cannot be negative")
        if float(config["spatial_model"].get("multiplicity_gamma", 0.05)) < 0.0:
            raise ValueError("spatial_model.multiplicity_gamma cannot be negative")
    behavior = config.get("context_model")
    if behavior is not None:
        if not isinstance(behavior, dict):
            raise ValueError("context_model must be a mapping")
        behavior_mode = str(behavior.get("mode", "legacy_spatial")).lower()
        if behavior_mode not in {"legacy_spatial", "behavior_composition"}:
            raise ValueError(
                "context_model.mode must be legacy_spatial or behavior_composition"
            )
        if behavior_mode == "behavior_composition":
            if str(behavior.get("history", "frozen_train_reference")) != "frozen_train_reference":
                raise ValueError("V3.0 supports only context_model.history=frozen_train_reference")
            if not bool(behavior.get("positive_deviation_only", True)):
                raise ValueError("V3.0 requires positive_deviation_only=true")
            if float(behavior.get("epsilon", 1e-3)) <= 0.0:
                raise ValueError("context_model.epsilon must be positive")
    fpr = float(config["scoring"]["deployment_fpr"])
    if not 0.0 < fpr < 1.0:
        raise ValueError("scoring.deployment_fpr must be between zero and one")
    weights = config["scoring"]["final_weights"]
    if any(float(weights.get(name, 0.0)) < 0 for name in ("local", "spatial", "entity")):
        raise ValueError("Final score weights cannot be negative")
    if sum(float(weights.get(name, 0.0)) for name in ("local", "spatial", "entity")) <= 0:
        raise ValueError("At least one final score weight must be positive")
    fusion = str(config["scoring"].get("fusion", "weighted"))
    if fusion not in {"weighted", "normal_tail_max"}:
        raise ValueError("scoring.fusion must be weighted or normal_tail_max")
    if fusion == "normal_tail_max" and float(config["scoring"].get("tail_epsilon", 1e-12)) <= 0:
        raise ValueError("scoring.tail_epsilon must be positive")
    if isinstance(behavior, dict) and str(behavior.get("mode", "legacy_spatial")) == "behavior_composition":
        if fusion != "normal_tail_max":
            raise ValueError("V3 behavior_composition requires scoring.fusion=normal_tail_max")
        if float(config["scoring"].get("entity_weight", 0.0)) != 0.0:
            raise ValueError("V3.0 entity temporal score must not enter flow final score")
