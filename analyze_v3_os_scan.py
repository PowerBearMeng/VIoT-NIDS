#!/usr/bin/env python3
"""Generate the required Design V3 OS Scan mechanism audit.

Ports are read only after scoring to name benign application families in the
report.  They never participate in a reference key, deviation, or anomaly rule.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


PHASES = ("pre_attack", "attack_window", "post_attack")
SCORE_COLUMNS = (
    "local_tail_evidence",
    "context_tail_evidence",
    "final_anomaly",
)


def _phase(timestamp: float, start: float, end: float) -> str:
    # A context profile covers the complete 3-second window.  The one window
    # beginning shortly before the first attack packet therefore belongs to
    # the attack phase if it overlaps the attack interval.
    if timestamp + 3.0 <= start:
        return "pre_attack"
    if timestamp <= end:
        return "attack_window"
    return "post_attack"


def _service(row: dict[str, str]) -> str:
    """Post-evaluation naming only; never used by V3 scoring."""

    protocol = row["protocol"].lower()
    ports = {int(row["endpoint_a_port"]), int(row["endpoint_b_port"])}
    if protocol == "udp" and 8000 in ports:
        return "RTP"
    if protocol == "udp" and 8001 in ports:
        return "RTCP"
    if protocol == "tcp" and 8554 in ports:
        return "RTSP"
    if protocol == "tcp" and 8080 in ports:
        return "ONVIF"
    return "other"


def _distribution(values: list[float] | np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"n": 0, "min": None, "q25": None, "median": None, "q75": None, "q95": None, "max": None, "mean": None}
    return {
        "n": int(len(array)),
        "min": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _window_distribution(
    values: dict[tuple[str, int, int], float],
    windows: list[tuple[str, int]],
    mode: int,
) -> dict[str, Any]:
    array = np.asarray(
        [values.get((capture, window, mode), 0.0) for capture, window in windows],
        dtype=np.float64,
    )
    result = _distribution(array)
    result["sum"] = float(array.sum())
    result["nonzero_windows"] = int(np.count_nonzero(array))
    return result


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _normal(label: str) -> bool:
    return label.lower() in {"normal", "benign"}


def _read_rows(path: Path, start: float, end: float) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "capture", "window_index", "segment_start", "endpoint_a_ip", "endpoint_b_ip",
        "endpoint_a_port", "endpoint_b_port", "protocol", "label", "mode_id",
        "reconstruction_error", "reconstruction_score", "pair_mode_count",
        "entity_a_mode_count", "entity_b_mode_count", "pair_context_deviation",
        "entity_a_context_deviation", "entity_b_context_deviation",
        "local_tail_evidence", "context_tail_evidence", "final_anomaly",
        "deployment_prediction",
    }
    missing = required - set(rows[0] if rows else [])
    if missing:
        raise ValueError(f"V3 flow score CSV is missing columns: {sorted(missing)}")
    for row in rows:
        row["_phase"] = _phase(float(row["segment_start"]), start, end)
        row["_service"] = _service(row)
    return rows


def _scope_values(
    rows: list[dict[str, str]],
    source: str,
    target: str,
    *,
    count: bool,
) -> dict[str, dict[tuple[str, int, int], float]]:
    suffix = "mode_count" if count else "context_deviation"
    result: dict[str, dict[tuple[str, int, int], float]] = {
        "pair": {}, source: {}, target: {}
    }
    for row in rows:
        capture = row["capture"]
        window = int(row["window_index"])
        mode = int(row["mode_id"])
        key = (capture, window, mode)
        endpoints = {row["endpoint_a_ip"], row["endpoint_b_ip"]}
        if endpoints == {source, target}:
            result["pair"][key] = max(
                result["pair"].get(key, 0.0), _float(row, f"pair_{suffix}")
            )
        for entity, side in (
            (row["endpoint_a_ip"], "entity_a"),
            (row["endpoint_b_ip"], "entity_b"),
        ):
            if entity in {source, target}:
                result[entity][key] = max(
                    result[entity].get(key, 0.0), _float(row, f"{side}_{suffix}")
                )
    return result


def _mode_phase_profiles(
    rows: list[dict[str, str]], source: str, target: str, modes: list[int]
) -> list[dict[str, Any]]:
    windows_by_phase: dict[str, list[tuple[str, int]]] = {name: [] for name in PHASES}
    seen: set[tuple[str, int]] = set()
    for row in rows:
        key = (row["capture"], int(row["window_index"]))
        if key not in seen:
            windows_by_phase[row["_phase"]].append(key)
            seen.add(key)
    counts = _scope_values(rows, source, target, count=True)
    deviations = _scope_values(rows, source, target, count=False)
    output: list[dict[str, Any]] = []
    for mode in modes:
        for phase in PHASES:
            windows = windows_by_phase[phase]
            output.append(
                {
                    "mode_id": mode,
                    "phase": phase,
                    "windows": len(windows),
                    "pair_count": _window_distribution(counts["pair"], windows, mode),
                    "source_entity_count": _window_distribution(counts[source], windows, mode),
                    "target_entity_count": _window_distribution(counts[target], windows, mode),
                    "pair_deviation": _window_distribution(deviations["pair"], windows, mode),
                    "source_entity_deviation": _window_distribution(deviations[source], windows, mode),
                    "target_entity_deviation": _window_distribution(deviations[target], windows, mode),
                }
            )
    return output


def _service_audit(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for service in ("RTP", "RTCP", "RTSP", "ONVIF"):
        selected = [
            row for row in rows
            if _normal(row["label"])
            and row["_phase"] == "attack_window"
            and row["_service"] == service
        ]
        output.append(
            {
                "service": service,
                "normal_flows": len(selected),
                "false_positives": sum(int(row["deployment_prediction"]) for row in selected),
                "local": _distribution([_float(row, SCORE_COLUMNS[0]) for row in selected]),
                "context": _distribution([_float(row, SCORE_COLUMNS[1]) for row in selected]),
                "final": _distribution([_float(row, SCORE_COLUMNS[2]) for row in selected]),
                "mode_distribution": dict(
                    sorted(Counter(int(row["mode_id"]) for row in selected).items())
                ),
            }
        )
    return output


def _pair_normal_fp(
    rows: list[dict[str, str]], source: str, target: str
) -> list[dict[str, Any]]:
    output = []
    for phase in PHASES:
        selected = [
            row for row in rows
            if _normal(row["label"])
            and row["_phase"] == phase
            and {row["endpoint_a_ip"], row["endpoint_b_ip"]} == {source, target}
        ]
        false_positives = sum(int(row["deployment_prediction"]) for row in selected)
        output.append(
            {
                "phase": phase,
                "normal_flows": len(selected),
                "false_positives": false_positives,
                "false_positive_rate": false_positives / len(selected) if selected else 0.0,
                "by_service": {
                    service: {
                        "normal_flows": sum(row["_service"] == service for row in selected),
                        "false_positives": sum(
                            row["_service"] == service and int(row["deployment_prediction"])
                            for row in selected
                        ),
                    }
                    for service in ("RTP", "RTCP", "RTSP", "ONVIF", "other")
                },
            }
        )
    return output


def analyze(scores_path: Path, label_summary_path: Path, output_dir: Path) -> dict[str, Any]:
    summary = json.loads(label_summary_path.read_text(encoding="utf-8"))
    context = summary["captures"][0]["context"]
    attack_start = float(context["attack_start_epoch"])
    attack_end = float(context["attack_end_epoch"])
    source = str(context["attack_source"])
    target = str(context["target"])
    rows = _read_rows(scores_path, attack_start, attack_end)
    attack_rows = [row for row in rows if not _normal(row["label"])]
    normal_rows = [row for row in rows if _normal(row["label"])]
    mode_counter = Counter(int(row["mode_id"]) for row in attack_rows)
    major_modes = [mode for mode, _ in mode_counter.most_common(8)]
    normal_reconstruction_q95 = float(
        np.quantile([_float(row, "reconstruction_score") for row in normal_rows], 0.95)
    )
    sparse_like = [
        row for row in attack_rows
        if _float(row, "reconstruction_score") <= normal_reconstruction_q95
    ]
    sparse_like_context_positive = sum(
        _float(row, "context_tail_evidence") > 0.0 for row in sparse_like
    )
    benign_attack_period = [
        row for row in normal_rows if row["_phase"] == "attack_window"
    ]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "design": "V3.0",
        "dataset": "os_scan",
        "score_file": str(scores_path.resolve()),
        "label_summary": str(label_summary_path.resolve()),
        "attack_window": {
            "start_epoch": attack_start,
            "end_epoch": attack_end,
            "source": source,
            "target": target,
        },
        "audit_boundary": (
            "Ports are used only after evaluation to name RTP/RTCP/RTSP/ONVIF groups; "
            "they are absent from model inputs, reference keys, deviations, and anomaly rules."
        ),
        "attack_reconstruction_error": _distribution(
            [_float(row, "reconstruction_error") for row in attack_rows]
        ),
        "attack_reconstruction_score": _distribution(
            [_float(row, "reconstruction_score") for row in attack_rows]
        ),
        "attack_mode_distribution": [
            {
                "mode_id": mode,
                "flows": count,
                "fraction": count / len(attack_rows) if attack_rows else 0.0,
            }
            for mode, count in mode_counter.most_common()
        ],
        "major_mode_phase_profiles": _mode_phase_profiles(
            rows, source, target, major_modes
        ),
        "attack_period_normal_services": _service_audit(rows),
        "camera01_nvr_normal_fp_by_phase": _pair_normal_fp(rows, source, target),
        "hypothesis_checks": {
            "h1_sparse_individual_but_abnormal_multiplicity": {
                "normal_test_reconstruction_score_q95": normal_reconstruction_q95,
                "attack_flows_reconstruction_at_or_below_normal_q95": len(sparse_like),
                "attack_flows_total": len(attack_rows),
                "fraction_sparse_like": len(sparse_like) / len(attack_rows) if attack_rows else 0.0,
                "sparse_like_with_positive_context_evidence": sparse_like_context_positive,
                "fraction_sparse_like_with_positive_context_evidence": (
                    sparse_like_context_positive / len(sparse_like) if sparse_like else 0.0
                ),
            },
            "h2_mode_selective_no_blanket_entity_propagation": {
                "implementation_invariant": (
                    "Each flow reads pair/entity deviations only at its own mode_id."
                ),
                "attack_period_normal_flows": len(benign_attack_period),
                "attack_period_normal_flows_with_zero_context_evidence": sum(
                    _float(row, "context_tail_evidence") <= 0.0
                    for row in benign_attack_period
                ),
                "attack_period_normal_flows_flagged_by_final": sum(
                    int(row["deployment_prediction"]) for row in benign_attack_period
                ),
            },
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "os_scan_mechanism.json"
    md_path = output_dir / "os_scan_mechanism.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")
    print(f"OS Scan mechanism analysis complete json={json_path} markdown={md_path}")
    return payload


def _fmt(value: Any) -> str:
    return "-" if value is None else (f"{value:.4f}" if isinstance(value, float) else str(value))


def _markdown(payload: dict[str, Any]) -> str:
    rec = payload["attack_reconstruction_score"]
    h1 = payload["hypothesis_checks"]["h1_sparse_individual_but_abnormal_multiplicity"]
    h2 = payload["hypothesis_checks"]["h2_mode_selective_no_blanket_entity_propagation"]
    lines = [
        "# Design V3.0：OS Scan 机制分析",
        "",
        f"攻击源 Camera-01：`{payload['attack_window']['source']}`；目标 NVR：`{payload['attack_window']['target']}`。",
        "",
        f"> 审计边界：{payload['audit_boundary']}",
        "",
        "## 攻击 Flow 的 reconstruction 与 mode",
        "",
        f"攻击 Flow 数：{rec['n']}；reconstruction score 中位数 `{_fmt(rec['median'])}`，Q95 `{_fmt(rec['q95'])}`，最大值 `{_fmt(rec['max'])}`。",
        "",
        "| mode_id | 攻击 Flow | 占比 |",
        "|---:|---:|---:|",
    ]
    for row in payload["attack_mode_distribution"][:12]:
        lines.append(f"| {row['mode_id']} | {row['flows']} | {row['fraction']:.2%} |")
    lines += [
        "",
        "## 主要 mode 的阶段性 count/deviation",
        "",
        "下表的 count 为每个 3 秒窗口统计；deviation 为对应 scope/mode 的正向偏差。",
        "",
        "| mode | 阶段 | pair count 中位/最大 | Camera count 中位/最大 | NVR count 中位/最大 | pair dev 中位/最大 | Camera dev 中位/最大 | NVR dev 中位/最大 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["major_mode_phase_profiles"]:
        lines.append(
            f"| {row['mode_id']} | {row['phase']} | "
            f"{_fmt(row['pair_count']['median'])}/{_fmt(row['pair_count']['max'])} | "
            f"{_fmt(row['source_entity_count']['median'])}/{_fmt(row['source_entity_count']['max'])} | "
            f"{_fmt(row['target_entity_count']['median'])}/{_fmt(row['target_entity_count']['max'])} | "
            f"{_fmt(row['pair_deviation']['median'])}/{_fmt(row['pair_deviation']['max'])} | "
            f"{_fmt(row['source_entity_deviation']['median'])}/{_fmt(row['source_entity_deviation']['max'])} | "
            f"{_fmt(row['target_entity_deviation']['median'])}/{_fmt(row['target_entity_deviation']['max'])} |"
        )
    lines += [
        "",
        "## 攻击期间正常视频/控制 Flow",
        "",
        "| 业务 | 正常 Flow | FP | local 中位/Q95 | context 中位/Q95 | final 中位/Q95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["attack_period_normal_services"]:
        lines.append(
            f"| {row['service']} | {row['normal_flows']} | {row['false_positives']} | "
            f"{_fmt(row['local']['median'])}/{_fmt(row['local']['q95'])} | "
            f"{_fmt(row['context']['median'])}/{_fmt(row['context']['q95'])} | "
            f"{_fmt(row['final']['median'])}/{_fmt(row['final']['q95'])} |"
        )
    lines += [
        "",
        "## Camera-01/NVR 正常业务阶段 FP",
        "",
        "| 阶段 | 正常 Flow | FP | FPR |",
        "|---|---:|---:|---:|",
    ]
    for row in payload["camera01_nvr_normal_fp_by_phase"]:
        lines.append(
            f"| {row['phase']} | {row['normal_flows']} | {row['false_positives']} | {row['false_positive_rate']:.2%} |"
        )
    lines += [
        "",
        "## 假设检查",
        "",
        f"1. 攻击 Flow 中 `{h1['attack_flows_reconstruction_at_or_below_normal_q95']}/{h1['attack_flows_total']}` 的 reconstruction score 不超过正常 test Q95；其中 `{h1['sparse_like_with_positive_context_evidence']}` 条获得正 context evidence。该统计用于判断“单条可近似正常、同 mode 数量却异常增加”。",
        f"2. 实现层面保证 `{h2['implementation_invariant']}`，因此其他 mode 的 entity deviation 不会直接复制给当前 Flow。攻击期间正常 Flow 中有 `{h2['attack_period_normal_flows_with_zero_context_evidence']}/{h2['attack_period_normal_flows']}` 条 context evidence 为零；其余正值只能来自当前 Flow 自身 mode 的 count deviation。最终 FP 为 `{h2['attack_period_normal_flows_flagged_by_final']}`。",
        "",
        "完整分布、各 mode 三阶段 pair/entity count 与 deviation，以及业务 mode 分布见同目录 JSON。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--label-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.scores, args.label_summary, args.output_dir)


if __name__ == "__main__":
    main()
