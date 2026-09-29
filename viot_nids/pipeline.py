"""Fit benign V5 components, persist them, and score unseen flows."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import RobustScaler

from . import FEATURE_NAMES
from .neural import FlowMemory, channel_indices, embed, train_encoder
from .statistics import (CONTEXT_NAMES, EmpiricalCDF, assign_local,
                         context_groups, context_raw, discover_roles, fit_context,
                         fit_regimes, group_roles, prototypes, responsibility)

LOG = logging.getLogger(__name__)


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    fields = config["channel_fields"]
    allowed = {"src_ip", "dst_ip", "protocol", "src_port", "dst_port"}
    if not fields or len(fields) != len(set(fields)) or set(fields) - allowed:
        raise ValueError("channel_fields must be distinct flow identity fields")
    return config


def read_flows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"src_ip": str, "dst_ip": str, "protocol": str})
    expected = {"row_id", "timestamp_second", "src_ip", "dst_ip", "src_port", "dst_port", "protocol", *FEATURE_NAMES}
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(f"Missing flow columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"No flow rows in {path}")
    return frame.sort_values(["timestamp_second", "row_id"]).reset_index(drop=True)


def preprocessing(frame: pd.DataFrame, scaler: RobustScaler | None = None,
                  min_scale: float = 0.0
                  ) -> tuple[np.ndarray, RobustScaler]:
    features = frame[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    if not np.isfinite(features).all() or (features < 0).any():
        raise ValueError("V5 features must be finite and nonnegative")
    log_features = np.log1p(features)
    if scaler is None:
        scaler = RobustScaler().fit(log_features)
        if min_scale > 0:
            # Near-constant microsecond IATs can have an IQR near 1e-6.
            # A floor prevents rare, still-normal gaps from dominating every
            # masked reconstruction batch.
            scaler.scale_ = np.maximum(scaler.scale_, min_scale)
    return scaler.transform(log_features).astype(np.float32), scaler


def _score_components(frame: pd.DataFrame, z: np.ndarray, regimes: dict,
                      context_scalers: dict, context_models: dict,
                      context_stats: dict,
                      local_cdf: EmpiricalCDF | None,
                      context_cdfs: dict[int, EmpiricalCDF] | None,
                      channel_fields: list[str],
                      seen: dict | None = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    roles, assigned_regimes, local_raw = assign_local(z, regimes)
    contexts, members, contributions = context_groups(frame, roles, seen)
    g_roles = group_roles(members, roles)
    g_raw = context_raw(contexts, g_roles, context_scalers, context_models)
    if local_cdf is None:
        local_scores = np.zeros(len(frame), dtype=np.float64)
    else:
        local_scores = local_cdf.transform(local_raw)
    group_scores = np.zeros(len(contexts), dtype=np.float64)
    if context_cdfs:
        for role, cdf in context_cdfs.items():
            ix = np.flatnonzero(g_roles == role)
            group_scores[ix] = cdf.transform(g_raw[ix])
    row_group_raw = np.zeros(len(frame), dtype=np.float64)
    row_group_scores = np.zeros(len(frame), dtype=np.float64)
    row_responsibility = np.zeros(len(frame), dtype=np.float64)
    for g, indices in enumerate(members):
        row_group_raw[indices] = g_raw[g]
        row_group_scores[indices] = group_scores[g]
        role = int(g_roles[g])
        if role in context_stats:
            mean, std = context_stats[role]
            row_responsibility[indices] = responsibility(contributions[indices], contexts[g], mean, std)
    context_flow = row_group_scores * row_responsibility
    scores = pd.DataFrame({
        "row_id": frame.row_id.to_numpy(),
        "timestamp_second": frame.timestamp_second.to_numpy(),
        "src_ip": frame.src_ip.to_numpy(), "dst_ip": frame.dst_ip.to_numpy(),
        "src_port": frame.src_port.to_numpy(), "dst_port": frame.dst_port.to_numpy(),
        "protocol": frame.protocol.to_numpy(),
        "channel_id": ["|".join(map(str, row)) for row in frame[channel_fields].itertuples(index=False, name=None)],
        "assigned_role": roles, "assigned_regime": assigned_regimes,
        "local_raw": local_raw, "local_score": local_scores,
        "context_group_raw": row_group_raw, "context_group_score": row_group_scores,
        "responsibility": row_responsibility, "context_flow_score": context_flow,
        "final_score": np.maximum(local_scores, context_flow),
    })
    return scores, g_raw, g_roles


def train(flows: Path, model_dir: Path, config_path: Path,
          device: str = "cpu") -> dict:
    config = load_config(config_path)
    seed = int(config["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(config.get("torch_threads", 4)))
    frame = read_flows(flows)
    seconds = np.unique(frame.timestamp_second.to_numpy())
    if len(seconds) < 2:
        raise ValueError("Need separate benign train and validation seconds")
    cutoff = seconds[max(1, int(np.floor(len(seconds) * (1 - config["validation_fraction"]))))]
    train_frame = frame[frame.timestamp_second < cutoff].reset_index(drop=True)
    val_frame = frame[frame.timestamp_second >= cutoff].reset_index(drop=True)
    LOG.info("benign train=%d validation=%d cutoff=%d", len(train_frame), len(val_frame), cutoff)
    x_train, scaler = preprocessing(train_frame, min_scale=float(config.get("min_feature_scale", 0)))
    x_val, _ = preprocessing(val_frame, scaler)
    fields = config["channel_fields"]
    train_channels = channel_indices(train_frame, fields)
    val_channels = channel_indices(val_frame, fields)
    network = FlowMemory()
    neural = config["neural"]
    losses = train_encoder(
        network, x_train, train_frame.timestamp_second.to_numpy(dtype=np.int64),
        train_channels, epochs=int(neural["epochs"]),
        steps_per_epoch=int(neural["steps_per_epoch"]),
        batch_size=int(neural["batch_size"]),
        sequence_length=int(neural["sequence_length"]),
        learning_rate=float(neural["learning_rate"]),
        mask_ratio=float(neural["mask_ratio"]), seed=seed, device=device,
    )
    LOG.info("masked reconstruction losses: %s", losses)
    z_train = embed(network, x_train, train_frame.timestamp_second.to_numpy(dtype=np.int64), train_channels, device)
    z_val = embed(network, x_val, val_frame.timestamp_second.to_numpy(dtype=np.int64), val_channels, device)
    proto = prototypes(z_train, train_frame.timestamp_second.to_numpy(dtype=np.int64), train_channels)
    proto_scaler, bgmm, train_roles, channel_counts = discover_roles(
        proto, train_channels, len(train_frame), int(config["roles"]["max_components"]), seed)
    regimes = fit_regimes(z_train, train_roles, int(config["regimes"]["max_components"]),
                          seed, float(config["regimes"]["reg_covar"]))
    # Use the same conditional-likelihood assignment for context fitting and inference.
    inferred_train_roles, _, _ = assign_local(z_train, regimes)
    train_contexts, train_members, _ = context_groups(train_frame, inferred_train_roles)
    train_group_roles = group_roles(train_members, inferred_train_roles)
    context_scalers, context_models, context_stats = fit_context(
        train_contexts, train_group_roles, int(config["context"]["max_components"]),
        seed, float(config["context"]["reg_covar"]))
    _, val_raw, val_group_roles = _score_components(
        val_frame, z_val, regimes, context_scalers, context_models, context_stats,
        None, None, fields)
    _, _, val_local_raw = assign_local(z_val, regimes)
    local_cdf = EmpiricalCDF(val_local_raw)
    context_cdfs = {}
    for role in context_models:
        selected = val_raw[val_group_roles == role]
        if len(selected):
            context_cdfs[role] = EmpiricalCDF(selected)
    LOG.info("roles=%s regimes=%s context=%s", channel_counts,
             {r: m.n_components for r, m in regimes.items()}, list(context_models))
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(network.state_dict(), model_dir / "encoder.pt")
    joblib.dump({"feature_scaler": scaler, "prototype_scaler": proto_scaler,
                 "role_bgmm": bgmm, "regimes": regimes,
                 "context_scalers": context_scalers, "context_models": context_models,
                 "context_stats": context_stats, "local_cdf": local_cdf,
                 "context_cdfs": context_cdfs}, model_dir / "statistics.joblib")
    (model_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    summary = {"feature_names": list(FEATURE_NAMES), "train_flows": len(train_frame),
               "validation_flows": len(val_frame), "train_channels": len(train_channels),
               "role_channel_counts": channel_counts,
               "regime_counts": {r: m.n_components for r, m in regimes.items()},
               "context_counts": {r: m.n_components for r, m in context_models.items()},
               "local_calibration_samples": len(local_cdf.values),
               "context_calibration_samples": {r: len(c.values) for r, c in context_cdfs.items()},
               "reconstruction_losses": losses,
               "encoder_parameters": sum(p.numel() for p in network.parameters())}
    (model_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def score(flows: Path, model_dir: Path, out: Path, device: str = "cpu") -> pd.DataFrame:
    config = load_config(model_dir / "config.yaml")
    torch.set_num_threads(int(config.get("torch_threads", 4)))
    frame = read_flows(flows)
    stats = joblib.load(model_dir / "statistics.joblib")
    model = FlowMemory()
    model.load_state_dict(torch.load(model_dir / "encoder.pt", map_location=device, weights_only=True))
    x, _ = preprocessing(frame, stats["feature_scaler"])
    channels = channel_indices(frame, config["channel_fields"])
    LOG.info("scoring %d flows across %d channels", len(frame), len(channels))
    z = embed(model, x, frame.timestamp_second.to_numpy(dtype=np.int64), channels, device)
    LOG.info("embedded %d flows", len(frame))
    scores, _, _ = _score_components(frame, z, stats["regimes"],
        stats["context_scalers"], stats["context_models"], stats["context_stats"],
        stats["local_cdf"], stats["context_cdfs"], config["channel_fields"])
    LOG.info("calculated local and context scores for %d flows", len(frame))
    if not np.isfinite(scores.final_score).all() or not scores.final_score.between(0, 1).all():
        raise ValueError("Nonfinite or out-of-range final score")
    out.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out, index=False)
    return scores


def inspect(model_dir: Path) -> dict:
    summary = json.loads((model_dir / "summary.json").read_text())
    stats = joblib.load(model_dir / "statistics.joblib")
    summary["encoder"] = "Linear(25,64) -> LayerNorm -> GELU -> Linear(64,32)"
    summary["temporal"] = "Linear(log1p(delta_t),8) -> GRUCell(40,32) -> z[64]"
    summary["decoder"] = "Linear(64,64) -> GELU -> Linear(64,25)"
    summary["active_roles"] = len(stats["regimes"])
    summary["context_models"] = {r: {"components": m.n_components, "covariance": m.covariance_type}
                                 for r, m in stats["context_models"].items()}
    return summary
