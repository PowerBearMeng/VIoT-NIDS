#!/usr/bin/env python3
"""Evaluate Gotham manifest datasets one-by-one with frozen normal artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from analyze_v3_os_scan import analyze as analyze_os_scan
from evaluate_flow import evaluate
from extract_embeddings import extract
from fit_prototypes import apply_existing
from prepare_data import prepare
from utils.config import load_config, resolve_path
from utils.io import load_json, write_json


def dataset_config(base: dict[str, Any], dataset_name: str) -> dict[str, Any]:
    config = deepcopy(base)
    config["data"]["include_gotham_training"] = False
    config["data"]["include_gotham_evaluation"] = True
    config["data"]["evaluation_datasets"] = [dataset_name]
    config["data"]["processed_path"] = f"data/processed/gotham_eval/{dataset_name}.npz"
    config["data"]["metadata_path"] = f"data/processed/gotham_eval/{dataset_name}.metadata.json"
    output_root = str(base["runtime"].get("output_dir", "outputs/gotham")).rstrip("/")
    prefix = f"{output_root}/evaluation/{dataset_name}"
    config["runtime"]["embeddings_path"] = f"{prefix}/embeddings.npz"
    context_mode = str(config.get("context_model", {}).get("mode", "legacy_spatial"))
    if context_mode != "neural_intensity":
        config["runtime"]["prototypes_path"] = f"{prefix}/prototypes.npz"
    config["runtime"]["scores_path"] = f"{prefix}/flow_scores.csv"
    config["runtime"]["metrics_path"] = f"{prefix}/metrics.json"
    return config


def evaluation_names(config: dict[str, Any]) -> list[str]:
    manifest_path = resolve_path(config, config["data"]["gotham_manifest"])
    assert manifest_path is not None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [str(name) for name in manifest["evaluation"]["datasets"]]


def evaluate_one(
    base: dict[str, Any], dataset_name: str, device: str | None, keep_intermediates: bool
) -> dict[str, Any]:
    config = dataset_config(base, dataset_name)
    context_mode = str(config.get("context_model", {}).get("mode", "legacy_spatial"))
    training_prototypes = None
    if context_mode != "neural_intensity":
        training_prototypes = resolve_path(base, base["runtime"]["prototypes_path"])
        assert training_prototypes is not None
    print(f"===== Gotham dataset: {dataset_name} =====", flush=True)
    prepare(config)
    extract(config, device)
    if context_mode != "neural_intensity":
        assert training_prototypes is not None
        apply_existing(config, training_prototypes)
    metrics = evaluate(config, "test", device)
    if dataset_name == "os_scan" and int(metrics.get("format_version", 1)) == 3:
        manifest_path = resolve_path(base, base["data"]["gotham_manifest"])
        workspace_root = resolve_path(base, base["data"].get("gotham_workspace_root", ".."))
        scores_path = resolve_path(config, config["runtime"]["scores_path"])
        assert manifest_path is not None and workspace_root is not None and scores_path is not None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        capture_path = (workspace_root / manifest["datasets"][dataset_name]["pcap"]).resolve()
        label_summary = capture_path.parent / "labels" / "label_summary.json"
        analyze_os_scan(scores_path, label_summary, scores_path.parent / "analysis")
    if not keep_intermediates:
        cleanup_values = [
            config["data"]["processed_path"],
            config["runtime"]["embeddings_path"],
        ]
        if context_mode != "neural_intensity":
            cleanup_values.append(config["runtime"]["prototypes_path"])
        for value in cleanup_values:
            path = resolve_path(config, value)
            if path is not None:
                path.unlink(missing_ok=True)
    return metrics


def run_all(config_path: str, device: str | None, keep_intermediates: bool) -> dict[str, Any]:
    base = load_config(config_path)
    names = evaluation_names(base)
    script = Path(__file__).resolve()
    for name in names:
        command = [sys.executable, str(script), "--config", config_path, "--dataset", name, "--worker"]
        if device:
            command += ["--device", device]
        if keep_intermediates:
            command.append("--keep-intermediates")
        subprocess.run(command, check=True)
    summary: dict[str, Any] = {"datasets": {}, "ablations": {}, "count": len(names)}
    output_root = str(base["runtime"].get("output_dir", "outputs/gotham")).rstrip("/")
    for name in names:
        path = resolve_path(base, f"{output_root}/evaluation/{name}/metrics.json")
        assert path is not None
        metrics = load_json(path)
        summary["datasets"][name] = metrics["final"]
        if "ablations" in metrics:
            summary["ablations"][name] = metrics["ablations"]
    output = resolve_path(base, f"{output_root}/manifest_summary.json")
    assert output is not None
    write_json(output, summary)
    print(f"Gotham manifest evaluation complete summary={output}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gotham_train.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--keep-intermediates", action="store_true")
    args = parser.parse_args()
    base = load_config(args.config)
    if args.worker:
        if not args.dataset:
            raise ValueError("--worker requires --dataset")
        evaluate_one(base, args.dataset, args.device, args.keep_intermediates)
    else:
        run_all(args.config, args.device, args.keep_intermediates)


if __name__ == "__main__":
    main()
