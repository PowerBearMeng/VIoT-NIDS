"""Frozen V5 local representation with V6 cohort context and relation evidence."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import RobustScaler

from viot_nids.neural import FlowMemory, channel_indices, embed
from viot_nids.pipeline import preprocessing, read_flows
from viot_nids.statistics import assign_local, context_groups, select_gmm

LOG = logging.getLogger(__name__)
RELATION = ["src_ip", "dst_ip", "protocol"]
CONTEXT_COLUMNS = [
    "flow_count", "total_bytes", "total_packets", "mean_bytes", "std_bytes",
    "mean_packets", "std_packets", "unique_peers", "unique_ports", "new_peers",
]


class TailEvidence:
    """Benign upper-tail evidence, with a finite unseen-tail floor."""

    def __init__(self, normal_raw: np.ndarray) -> None:
        if len(normal_raw) == 0 or not np.isfinite(normal_raw).all():
            raise ValueError("Tail calibration needs finite benign samples")
        self.values = np.sort(np.asarray(normal_raw, dtype=np.float64))

    def transform(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.float64)
        rank = np.searchsorted(self.values, raw, side="left")
        tail_p = (len(self.values) - rank + 1) / (len(self.values) + 1)
        return -np.log(tail_p)


def _load_base(model_dir: Path, device: str) -> tuple[FlowMemory, dict, dict]:
    stats = joblib.load(model_dir / "base_statistics.joblib")
    config = yaml.safe_load((model_dir / "base_config.yaml").read_text())
    net = FlowMemory()
    net.load_state_dict(torch.load(model_dir / "base_encoder.pt", map_location=device, weights_only=True))
    torch.set_num_threads(int(config.get("torch_threads", 4)))
    return net, stats, config


def _local(frame: pd.DataFrame, net: FlowMemory, stats: dict, base_config: dict,
           device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, _ = preprocessing(frame, stats["feature_scaler"])
    z = embed(net, x, frame.timestamp_second.to_numpy(dtype=np.int64),
              channel_indices(frame, base_config["channel_fields"]), device)
    return assign_local(z, stats["regimes"])


def _context(frame: pd.DataFrame, roles: np.ndarray,
             regimes: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    # The existing context aggregator implements the same 10 dimensions and
    # causal new-peer history. Distinct joint IDs scope it to role and regime.
    joint = roles.astype(np.int32) * 16 + regimes.astype(np.int32)
    contexts, members, _ = context_groups(frame, joint)
    codes = np.asarray([joint[ix[0]] for ix in members], dtype=np.int32)
    return contexts, members, codes


def _fit_context_models(train_contexts: np.ndarray, train_codes: np.ndarray,
                        calibration_codes: np.ndarray, cfg: dict
                        ) -> tuple[dict, dict]:
    contexts = np.log1p(train_contexts)
    seed = int(cfg["seed"])
    ccfg = cfg["context"]
    specs: dict[tuple, tuple[RobustScaler, object]] = {}

    def fit(key: tuple, block: np.ndarray) -> None:
        scaler = RobustScaler().fit(block)
        model = select_gmm(scaler.transform(block), int(ccfg["max_components"]),
                           seed, float(ccfg["reg_covar"]))
        specs[key] = (scaler, model)

    fit(("global",), contexts)
    for role in np.unique(train_codes // 16):
        mask = train_codes // 16 == role
        if int(mask.sum()) >= int(ccfg["min_role_train"]):
            fit(("role", int(role)), contexts[mask])
    choices = {}
    for code in np.unique(train_codes):
        role, regime = divmod(int(code), 16)
        train_count = int((train_codes == code).sum())
        calib_count = int((calibration_codes == code).sum())
        role_calib_count = int((calibration_codes // 16 == role).sum())
        cell = ("cell", role, regime)
        parent = ("role", role)
        if train_count >= int(ccfg["min_cell_train"]) and calib_count >= int(ccfg["min_cell_calibration"]):
            fit(cell, contexts[train_codes == code])
            choices[int(code)] = cell
        elif parent in specs and role_calib_count >= int(ccfg["min_role_calibration"]):
            choices[int(code)] = parent
        else:
            choices[int(code)] = ("global",)
    return specs, choices


def _context_raw(contexts: np.ndarray, codes: np.ndarray, specs: dict,
                 choices: dict) -> tuple[np.ndarray, list[tuple]]:
    keys = [choices.get(int(code), ("global",)) for code in codes]
    raw = np.empty(len(contexts), dtype=np.float64)
    transformed = np.log1p(contexts)
    for key in set(keys):
        ix = np.flatnonzero(np.asarray([candidate == key for candidate in keys]))
        scaler, model = specs[key]
        raw[ix] = -model.score_samples(scaler.transform(transformed[ix]))
    return raw, keys


def _relation_counts(frame: pd.DataFrame) -> dict:
    return {
        "pair": frame.groupby(RELATION, sort=False).size(),
        "src_port": frame.groupby(RELATION + ["src_port"], sort=False).size(),
        "dst_port": frame.groupby(RELATION + ["dst_port"], sort=False).size(),
        "protocol": frame.groupby(["protocol"], sort=False).size(),
        "global_src_port": frame.groupby(["protocol", "src_port"], sort=False).size(),
        "global_dst_port": frame.groupby(["protocol", "dst_port"], sort=False).size(),
    }


def _relation_raw(frame: pd.DataFrame, counts: dict) -> tuple[np.ndarray, np.ndarray]:
    pair = counts["pair"].reindex(pd.MultiIndex.from_frame(frame[RELATION]), fill_value=0).to_numpy()
    src = counts["src_port"].reindex(pd.MultiIndex.from_frame(frame[RELATION + ["src_port"]]), fill_value=0).to_numpy()
    dst = counts["dst_port"].reindex(pd.MultiIndex.from_frame(frame[RELATION + ["dst_port"]]), fill_value=0).to_numpy()
    global_count = counts["protocol"].reindex(frame.protocol, fill_value=0).to_numpy()
    global_src = counts["global_src_port"].reindex(
        pd.MultiIndex.from_frame(frame[["protocol", "src_port"]]), fill_value=0).to_numpy()
    global_dst = counts["global_dst_port"].reindex(
        pd.MultiIndex.from_frame(frame[["protocol", "dst_port"]]), fill_value=0).to_numpy()
    known = pair > 0
    reference_count = np.where(known, pair, global_count)
    familiar = np.where(known, np.maximum(src, dst), np.maximum(global_src, global_dst))
    # An unseen host pair backs off to benign protocol/service familiarity.
    # A new host using a common service need not look like a new service.
    raw = np.log1p(reference_count) - np.log1p(familiar)
    return raw, familiar


def _component_scores(frame: pd.DataFrame, roles: np.ndarray, regimes: np.ndarray,
                      local_raw: np.ndarray, model: dict) -> pd.DataFrame:
    contexts, members, codes = _context(frame, roles, regimes)
    group_raw, keys = _context_raw(contexts, codes, model["context_models"], model["context_choices"])
    group_evidence = np.empty(len(contexts), dtype=np.float64)
    for key in set(keys):
        ix = np.flatnonzero(np.asarray([candidate == key for candidate in keys]))
        group_evidence[ix] = model["context_calibrators"][key].transform(group_raw[ix])
    row_context_raw = np.empty(len(frame), dtype=np.float64)
    row_context_evidence = np.empty(len(frame), dtype=np.float64)
    for i, indices in enumerate(members):
        row_context_raw[indices] = group_raw[i]
        row_context_evidence[indices] = group_evidence[i]
    relation_raw, familiar = _relation_raw(frame, model["relation_counts"])
    local_evidence = model["local_calibrator"].transform(local_raw)
    relation_evidence = model["relation_calibrator"].transform(relation_raw)
    joint = local_evidence + row_context_evidence + relation_evidence
    if not np.isfinite(joint).all():
        raise ValueError("Non-finite V6 evidence")
    return pd.DataFrame({
        "row_id": frame.row_id.to_numpy(),
        "src_ip": frame.src_ip.to_numpy(), "dst_ip": frame.dst_ip.to_numpy(),
        "src_port": frame.src_port.to_numpy(), "dst_port": frame.dst_port.to_numpy(),
        "protocol": frame.protocol.to_numpy(), "timestamp_second": frame.timestamp_second.to_numpy(),
        "assigned_role": roles, "assigned_regime": regimes,
        "local_raw": local_raw, "local_evidence": local_evidence,
        "context_raw": row_context_raw, "context_evidence": row_context_evidence,
        "relation_raw": relation_raw, "relation_seen_count": familiar,
        "relation_evidence": relation_evidence,
        "final_score": joint,
        "final_percentile": 1 - np.exp(-model["joint_calibrator"].transform(joint))
            if model.get("joint_calibrator") is not None else np.zeros(len(joint)),
    })


def train(benign_flows: Path, base_model: Path, model_dir: Path,
          config_path: Path, device: str = "cpu") -> dict:
    cfg = yaml.safe_load(config_path.read_text())
    np.random.seed(int(cfg["seed"]))
    torch.manual_seed(int(cfg["seed"]))
    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(base_model / "encoder.pt", model_dir / "base_encoder.pt")
    shutil.copyfile(base_model / "statistics.joblib", model_dir / "base_statistics.joblib")
    shutil.copyfile(base_model / "config.yaml", model_dir / "base_config.yaml")
    net, base_stats, base_cfg = _load_base(model_dir, device)
    frame = read_flows(benign_flows)
    seconds = np.unique(frame.timestamp_second.to_numpy())
    cutoff = seconds[max(1, int(np.floor(len(seconds) * 0.8)))]
    fit_frame = frame[frame.timestamp_second < cutoff].reset_index(drop=True)
    heldout = frame[frame.timestamp_second >= cutoff].reset_index(drop=True)
    LOG.info("V6 benign fitting=%d heldout=%d", len(fit_frame), len(heldout))
    train_roles, train_regimes, _ = _local(fit_frame, net, base_stats, base_cfg, device)
    val_roles, val_regimes, val_local_raw = _local(heldout, net, base_stats, base_cfg, device)
    split_second = np.unique(heldout.timestamp_second.to_numpy())[len(np.unique(heldout.timestamp_second.to_numpy())) // 2]
    a = heldout.timestamp_second.to_numpy() < split_second
    b = ~a
    fit_contexts, _, fit_codes = _context(fit_frame, train_roles, train_regimes)
    cal_contexts, _, cal_codes = _context(heldout[a].reset_index(drop=True), val_roles[a], val_regimes[a])
    specs, choices = _fit_context_models(fit_contexts, fit_codes, cal_codes, cfg)
    cal_raw, cal_keys = _context_raw(cal_contexts, cal_codes, specs, choices)
    context_calibrators = {}
    for key in set(cal_keys):
        ix = np.asarray([candidate == key for candidate in cal_keys])
        context_calibrators[key] = TailEvidence(cal_raw[ix])
    if ("global",) not in context_calibrators:
        scaler, gmm = specs[("global",)]
        global_raw = -gmm.score_samples(scaler.transform(np.log1p(cal_contexts)))
        context_calibrators[("global",)] = TailEvidence(global_raw)
    counts = _relation_counts(fit_frame)
    relation_calibrator = TailEvidence(_relation_raw(heldout[a], counts)[0])
    local_calibrator = TailEvidence(val_local_raw[a])
    model = {
        "context_models": specs, "context_choices": choices,
        "context_calibrators": context_calibrators,
        "relation_counts": counts, "relation_calibrator": relation_calibrator,
        "local_calibrator": local_calibrator, "joint_calibrator": None,
    }
    val_b = heldout[b].reset_index(drop=True)
    calibration_scores = _component_scores(val_b, val_roles[b], val_regimes[b], val_local_raw[b], model)
    model["joint_calibrator"] = TailEvidence(calibration_scores.final_score.to_numpy())
    joblib.dump(model, model_dir / "v6.joblib")
    (model_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    summary = {
        "base_model": str(base_model), "base_encoder_frozen": True,
        "benign_fit_flows": len(fit_frame), "component_calibration_flows": int(a.sum()),
        "joint_calibration_flows": int(b.sum()),
        "fit_cohorts": len(fit_contexts), "component_calibration_cohorts": len(cal_contexts),
        "context_model_keys": {str(k): int(v[1].n_components) for k, v in specs.items()},
        "context_choices": {str(k): str(v) for k, v in choices.items()},
    }
    (model_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def score(flows: Path, model_dir: Path, out: Path, device: str = "cpu",
          base_scores: Path | None = None) -> pd.DataFrame:
    frame = read_flows(flows)
    model = joblib.load(model_dir / "v6.joblib")
    if base_scores is not None:
        cached = pd.read_csv(base_scores, usecols=["row_id", "timestamp_second", "src_ip", "dst_ip",
                                                     "src_port", "dst_port", "protocol",
                                                     "assigned_role", "assigned_regime", "local_raw"])
        fields = ["row_id", "timestamp_second", "src_ip", "dst_ip", "src_port", "dst_port", "protocol"]
        if len(cached) != len(frame) or any(not np.array_equal(cached[c].to_numpy(), frame[c].to_numpy()) for c in fields):
            raise ValueError("Cached V5 scores do not align with extracted Flows")
        roles = cached.assigned_role.to_numpy(dtype=np.int32)
        regimes = cached.assigned_regime.to_numpy(dtype=np.int32)
        local_raw = cached.local_raw.to_numpy(dtype=np.float64)
    else:
        net, base_stats, base_cfg = _load_base(model_dir, device)
        roles, regimes, local_raw = _local(frame, net, base_stats, base_cfg, device)
    scores = _component_scores(frame, roles, regimes, local_raw, model)
    out.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out, index=False)
    return scores


def rescore_components(previous_scores: Path, model_dir: Path, out: Path) -> pd.DataFrame:
    """Recompute only V6 relation/fusion after a frozen-context revision."""
    frame = pd.read_csv(previous_scores)
    expected = {"row_id", "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
                "local_evidence", "context_evidence"}
    if expected - set(frame.columns):
        raise ValueError("Missing frozen component scores or relation identities")
    model = joblib.load(model_dir / "v6.joblib")
    raw, familiar = _relation_raw(frame, model["relation_counts"])
    frame["relation_raw"] = raw
    frame["relation_seen_count"] = familiar
    frame["relation_evidence"] = model["relation_calibrator"].transform(raw)
    frame["final_score"] = (frame.local_evidence.to_numpy() +
                            frame.context_evidence.to_numpy() + frame.relation_evidence.to_numpy())
    frame["final_percentile"] = 1 - np.exp(-model["joint_calibrator"].transform(frame.final_score))
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    return frame
