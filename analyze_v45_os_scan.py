"""OS Scan diagnostics for V4.5 sparse attention selectivity."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _normal(label: str) -> bool:
    return label.strip().lower() in {"normal", "benign"}


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "q95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def _phase(start: float, attack_start: float, attack_end: float) -> str:
    if start < attack_start:
        return "before"
    if start >= attack_end:
        return "after"
    return "attack"


def _row_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    fields = (
        "pair_null_weight", "pair_support_size", "pair_effective_neighbors",
        "entity_selected_null_weight", "entity_selected_support_size",
        "entity_selected_effective_neighbors", "context_anomaly", "final_anomaly",
    )
    return {
        "flows": len(rows),
        "false_positives": sum(int(row["deployment_prediction"]) for row in rows),
        **{
            field: _distribution([float(row[field]) for row in rows])
            for field in fields
        },
    }


def _select_entity_side(row: dict[str, str], entity: str) -> dict[str, str] | None:
    if row["endpoint_a_ip"] == entity:
        side = "entity_a"
    elif row["endpoint_b_ip"] == entity:
        side = "entity_b"
    else:
        return None
    copied = dict(row)
    copied["entity_selected_null_weight"] = row[f"{side}_null_weight"]
    copied["entity_selected_support_size"] = row[f"{side}_support_size"]
    copied["entity_selected_effective_neighbors"] = row[
        f"{side}_effective_neighbors"
    ]
    return copied


def analyze(scores_path: Path, label_summary_path: Path, output_dir: Path) -> dict[str, Any]:
    label_summary = json.loads(label_summary_path.read_text(encoding="utf-8"))
    context = label_summary["captures"][0]["context"]
    attack_start = float(context["attack_start_epoch"])
    attack_end = float(context["attack_end_epoch"])
    source = str(context["attack_source"])
    target = str(context["target"])
    with scores_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["_phase"] = _phase(
            float(row["segment_start"]), attack_start, attack_end
        )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "design": "V4.5 sparse_context",
        "dataset": "os_scan",
        "score_file": str(scores_path.resolve()),
        "attack_window": {
            "start_epoch": attack_start,
            "end_epoch": attack_end,
            "source": source,
            "target": target,
        },
        "normal_flow_attention_on_attack_entities": {},
        "normal_source_target_pair": {},
    }
    entity_output: dict[str, Any] = {}
    for name, entity in (("source", source), ("target", target)):
        selected = [
            enriched
            for row in rows
            if _normal(row["label"])
            for enriched in [_select_entity_side(row, entity)]
            if enriched is not None
        ]
        entity_output[name] = {
            phase: _row_summary(
                [row for row in selected if row["_phase"] == phase]
            )
            for phase in ("before", "attack", "after")
        }
    payload["normal_flow_attention_on_attack_entities"] = entity_output

    pair_rows: list[dict[str, str]] = []
    for row in rows:
        if not _normal(row["label"]):
            continue
        if {row["endpoint_a_ip"], row["endpoint_b_ip"]} != {source, target}:
            continue
        enriched = _select_entity_side(row, source)
        if enriched is not None:
            pair_rows.append(enriched)
    payload["normal_source_target_pair"] = {
        phase: _row_summary([row for row in pair_rows if row["_phase"] == phase])
        for phase in ("before", "attack", "after")
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "os_scan_sparse_attention.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"V4.5 OS Scan sparse-attention analysis complete output={output}")
    return payload
