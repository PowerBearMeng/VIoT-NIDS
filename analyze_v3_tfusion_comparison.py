#!/usr/bin/env python3
"""Compare trained Design V3 with TFusion using AUROC, EER, and OS Scan FPs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from utils.io import write_json
from utils.metrics import equal_error_rate


def _normal(label: str) -> bool:
    return label.lower() in {"normal", "benign"}


def _service(row: dict[str, str]) -> str:
    """Evaluation-only service naming; ports never enter V3 scoring."""

    protocol = row["protocol"].lower()
    ports = {int(row["endpoint_a_port"]), int(row["endpoint_b_port"])}
    packets = int(row["packet_count"])
    if protocol == "udp" and 123 in ports:
        return "NTP"
    if protocol == "udp" and 8000 in ports:
        return "RTP/UDP video"
    if protocol == "udp" and 8001 in ports:
        return "RTCP"
    if protocol == "tcp" and 8554 in ports:
        return "TCP interleaved video" if packets > 100 else "RTSP control"
    if protocol == "tcp" and 8080 in ports:
        return "HTTP/ONVIF"
    return "other"


def _phase(timestamp: float, start: float, end: float) -> str:
    if timestamp + 3.0 <= start:
        return "pre_attack"
    if timestamp <= end:
        return "attack_window"
    return "post_attack"


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _read_scores(path: Path, keep_rows: bool) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    labels: list[int] = []
    scores: list[float] = []
    retained: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            labels.append(0 if _normal(row["label"]) else 1)
            scores.append(float(row["final_anomaly"]))
            if keep_rows:
                retained.append(row)
    return np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float64), retained


def _counter_rows(population: Counter[str], false_positives: Counter[str]) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "normal_flows": int(population[name]),
            "false_positives": int(false_positives[name]),
            "false_positive_rate": _rate(false_positives[name], population[name]),
        }
        for name in sorted(population, key=lambda key: (-false_positives[key], key))
    ]


def _v3_os_false_positives(
    rows: list[dict[str, str]],
    labels: np.ndarray,
    scores: np.ndarray,
    eer: dict[str, float],
    label_summary: Path,
) -> dict[str, Any]:
    metadata = json.loads(label_summary.read_text(encoding="utf-8"))
    context = metadata["captures"][0]["context"]
    start = float(context["attack_start_epoch"])
    end = float(context["attack_end_epoch"])
    camera = str(context["attack_source"])
    nvr = str(context["target"])
    threshold = float(eer["threshold_upper"])
    predictions = scores >= threshold
    normal = labels == 0
    attack = labels == 1
    population_service: Counter[str] = Counter()
    fp_service: Counter[str] = Counter()
    population_phase: Counter[str] = Counter()
    fp_phase: Counter[str] = Counter()
    population_service_phase: Counter[tuple[str, str]] = Counter()
    fp_service_phase: Counter[tuple[str, str]] = Counter()
    camera_nvr_population: Counter[str] = Counter()
    camera_nvr_fp: Counter[str] = Counter()
    for index, row in enumerate(rows):
        if not normal[index]:
            continue
        service = _service(row)
        phase = _phase(float(row["segment_start"]), start, end)
        population_service[service] += 1
        population_phase[phase] += 1
        population_service_phase[(service, phase)] += 1
        is_pair = {row["endpoint_a_ip"], row["endpoint_b_ip"]} == {camera, nvr}
        if is_pair:
            camera_nvr_population[phase] += 1
        if predictions[index]:
            fp_service[service] += 1
            fp_phase[phase] += 1
            fp_service_phase[(service, phase)] += 1
            if is_pair:
                camera_nvr_fp[phase] += 1
    fp = int(np.sum(predictions & normal))
    fn = int(np.sum(~predictions & attack))
    return {
        "eer": eer,
        "threshold_policy": "threshold_upper; prediction is score >= threshold",
        "threshold": threshold,
        "normal_flows": int(normal.sum()),
        "attack_flows": int(attack.sum()),
        "false_positives": fp,
        "false_negatives": fn,
        "empirical_false_positive_rate": _rate(fp, int(normal.sum())),
        "empirical_false_negative_rate": _rate(fn, int(attack.sum())),
        "tie_warning": (
            "EER is interpolated. Large V3 score ties mean the realizable upper-bracket "
            "threshold can have empirical FPR/FNR far from the interpolated EER."
        ),
        "by_phase": _counter_rows(population_phase, fp_phase),
        "by_service": _counter_rows(population_service, fp_service),
        "by_service_phase": [
            {
                "service": service,
                "phase": phase,
                "normal_flows": int(population_service_phase[(service, phase)]),
                "false_positives": int(fp_service_phase[(service, phase)]),
                "false_positive_rate": _rate(
                    fp_service_phase[(service, phase)],
                    population_service_phase[(service, phase)],
                ),
            }
            for service, phase in sorted(
                population_service_phase,
                key=lambda key: (-fp_service_phase[key], key),
            )
        ],
        "camera01_nvr_by_phase": [
            {
                "phase": phase,
                "normal_flows": int(camera_nvr_population[phase]),
                "false_positives": int(camera_nvr_fp[phase]),
                "false_positive_rate": _rate(
                    camera_nvr_fp[phase], camera_nvr_population[phase]
                ),
            }
            for phase in ("pre_attack", "attack_window", "post_attack")
        ],
    }


def compare(
    v3_root: Path,
    tfusion_root: Path,
    tfusion_experiment: str,
    tfusion_fp_report: Path,
    label_summary: Path,
    output_dir: Path,
) -> dict[str, Any]:
    manifest_path = v3_root / "manifest_summary.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tfusion_fp = json.loads(tfusion_fp_report.read_text(encoding="utf-8"))
    tfusion_os = next(
        row for row in tfusion_fp["datasets"] if row["dataset"] == "os_scan"
    )
    comparisons: list[dict[str, Any]] = []
    os_rows: list[dict[str, str]] = []
    os_labels = np.empty(0, dtype=np.int64)
    os_scores = np.empty(0, dtype=np.float64)
    os_eer: dict[str, float] | None = None
    for dataset, v3_final in manifest["datasets"].items():
        scores_path = v3_root / "evaluation" / dataset / "flow_scores.csv"
        labels, scores, rows = _read_scores(scores_path, dataset == "os_scan")
        eer = equal_error_rate(labels, scores)
        v3_final["EER"] = float(eer["value"])
        v3_final["eer"] = eer
        metrics_path = v3_root / "evaluation" / dataset / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["final"]["EER"] = float(eer["value"])
        metrics["final"]["eer"] = eer
        if dataset in metrics.get("per_dataset", {}):
            metrics["per_dataset"][dataset]["final"]["EER"] = float(eer["value"])
            metrics["per_dataset"][dataset]["final"]["eer"] = eer
        write_json(metrics_path, metrics)
        tfusion_path = tfusion_root / "test" / dataset / tfusion_experiment / "metrics.json"
        tfusion = json.loads(tfusion_path.read_text(encoding="utf-8"))
        comparisons.append(
            {
                "dataset": dataset,
                "v3_unit": "3-second capture-relative bidirectional flow segment",
                "v3_auroc": float(v3_final["AUROC"]),
                "v3_eer": float(eer["value"]),
                "tfusion_unit": "3-second epoch-aligned directional flow segment",
                "tfusion_auroc": float(tfusion["roc_auc"]),
                "tfusion_eer": float(tfusion["eer"]["value"]),
                "auroc_delta_v3_minus_tfusion": float(
                    v3_final["AUROC"] - tfusion["roc_auc"]
                ),
                "eer_delta_v3_minus_tfusion": float(
                    eer["value"] - tfusion["eer"]["value"]
                ),
            }
        )
        if dataset == "os_scan":
            os_rows, os_labels, os_scores, os_eer = rows, labels, scores, eer
    write_json(manifest_path, manifest)
    assert os_eer is not None
    v3_os_fp = _v3_os_false_positives(
        os_rows, os_labels, os_scores, os_eer, label_summary
    )
    payload = {
        "schema_version": 1,
        "comparison_scope": (
            "Both methods use the same Gotham captures, packet ground truth, and 3-second "
            "Flow windows. TFusion uses epoch-aligned directional five-tuples; V3 uses "
            "capture-relative bidirectional canonical five-tuples. AUROC/EER are directly "
            "comparable baseline-native rates, while raw FP counts are not one-to-one."
        ),
        "tfusion_experiment": tfusion_experiment,
        "datasets": comparisons,
        "macro": {
            "v3_auroc": float(np.mean([row["v3_auroc"] for row in comparisons])),
            "tfusion_auroc": float(
                np.mean([row["tfusion_auroc"] for row in comparisons])
            ),
            "v3_eer": float(np.mean([row["v3_eer"] for row in comparisons])),
            "tfusion_eer": float(
                np.mean([row["tfusion_eer"] for row in comparisons])
            ),
        },
        "os_scan_false_positives": {
            "v3": v3_os_fp,
            "tfusion": tfusion_os,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "v3_vs_tfusion_auroc_eer_fp.json"
    markdown_path = output_dir / "v3_vs_tfusion_auroc_eer_fp.md"
    write_json(json_path, payload)
    markdown_path.write_text(_markdown(payload), encoding="utf-8")
    print(f"V3 vs TFusion comparison complete json={json_path} markdown={markdown_path}")
    return payload


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Design V3.0 与 TFusion：AUROC、EER 与 OS Scan 误报",
        "",
        "## 可比性说明",
        "",
        payload["comparison_scope"],
        "",
        "你的判断是对的：TFusion 与 V3 都按 3 秒切分 Flow。区别是 TFusion 按绝对时间边界切分有向五元组，而 V3 从每个 capture 的首包开始切分，并把两个方向合并为一个双向五元组。因此 AUROC/EER 可以直接作为各模型原生样本上的比较指标；原始 FP 条数则不是一一对应，应同时报告误报率、阶段、业务类型和实体位置。",
        "",
        "## AUROC 与 EER",
        "",
        "| 数据集 | V3 AUROC | TFusion AUROC | ΔAUROC | V3 EER | TFusion EER | ΔEER |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["datasets"]:
        lines.append(
            f"| {row['dataset']} | {row['v3_auroc']:.4f} | {row['tfusion_auroc']:.4f} | "
            f"{row['auroc_delta_v3_minus_tfusion']:+.4f} | {row['v3_eer']:.2%} | "
            f"{row['tfusion_eer']:.2%} | {row['eer_delta_v3_minus_tfusion']:+.2%} |"
        )
    macro = payload["macro"]
    lines.append(
        f"| **九数据集平均** | **{macro['v3_auroc']:.4f}** | **{macro['tfusion_auroc']:.4f}** | "
        f"**{macro['v3_auroc'] - macro['tfusion_auroc']:+.4f}** | **{macro['v3_eer']:.2%}** | "
        f"**{macro['tfusion_eer']:.2%}** | **{macro['v3_eer'] - macro['tfusion_eer']:+.2%}** |"
    )
    v3 = payload["os_scan_false_positives"]["v3"]
    tfusion = payload["os_scan_false_positives"]["tfusion"]
    lines += [
        "",
        "## OS Scan EER 误报",
        "",
        "| 方法 | EER | 原生正常 Flow | EER 上沿 FP | 经验 FPR | FN | 经验 FNR |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| V3 | {v3['eer']['value']:.2%} | {v3['normal_flows']} | {v3['false_positives']} | {v3['empirical_false_positive_rate']:.2%} | {v3['false_negatives']} | {v3['empirical_false_negative_rate']:.2%} |",
        f"| TFusion | {tfusion['eer']:.2%} | {tfusion['normal_flows']} | {tfusion['false_positives']} | {tfusion['empirical_false_positive_rate']:.2%} | {tfusion['false_negatives']} | {tfusion['empirical_false_negative_rate']:.2%} |",
        "",
        "V3 的 EER 是插值值。由于大量攻击 Flow 的 final score 并列，EER 上沿可实现阈值的经验 FPR/FNR 不对称；因此 EER 应与上表经验值一起解释。",
        "",
        "### V3 误报构成",
        "",
        "| 业务 | 正常 Flow | FP | FPR |",
        "|---|---:|---:|---:|",
    ]
    for row in v3["by_service"]:
        if row["false_positives"] or row["name"] != "other":
            lines.append(
                f"| {row['name']} | {row['normal_flows']} | {row['false_positives']} | {row['false_positive_rate']:.2%} |"
            )
    lines += [
        "",
        "| V3 阶段 | 正常 Flow | FP | FPR |",
        "|---|---:|---:|---:|",
    ]
    for row in v3["by_phase"]:
        lines.append(
            f"| {row['name']} | {row['normal_flows']} | {row['false_positives']} | {row['false_positive_rate']:.2%} |"
        )
    lines += [
        "",
        "| Camera-01/NVR 阶段 | 正常 Flow | FP | FPR |",
        "|---|---:|---:|---:|",
    ]
    for row in v3["camera01_nvr_by_phase"]:
        lines.append(
            f"| {row['phase']} | {row['normal_flows']} | {row['false_positives']} | {row['false_positive_rate']:.2%} |"
        )
    lines += [
        "",
        "### 与 TFusion 误报机制的差异",
        "",
        "TFusion 的 OS Scan EER 误报由两部分组成：NTP 是跨阶段稳定误报；RTP、RTCP、RTSP、ONVIF 则高度集中在攻击期 Camera-01/NVR，属于实体状态污染造成的连带误报。",
        "",
        "V3 不再把整个 entity anomaly 复制给所有 Flow，因此攻击期 Camera-01/NVR 的视频连带效应明显减弱。但在 EER 上沿，V3 仍会误报部分 RTSP、ONVIF、RTP 和 RTCP；这些误报来自对应 Flow 自身 mode 的 count deviation或 local reconstruction，而不是其他攻击 mode 的 entity 分数直接传播。",
        "",
        "V3 的剩余问题主要是分数离散和融合：大量攻击/正常 Flow 落在相同 empirical-tail evidence，导致 EER 阈值发生大跨度跳变。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3-root", type=Path, required=True)
    parser.add_argument("--tfusion-root", type=Path, required=True)
    parser.add_argument("--tfusion-experiment", required=True)
    parser.add_argument("--tfusion-fp-report", type=Path, required=True)
    parser.add_argument("--label-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    compare(
        args.v3_root,
        args.tfusion_root,
        args.tfusion_experiment,
        args.tfusion_fp_report,
        args.label_summary,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
