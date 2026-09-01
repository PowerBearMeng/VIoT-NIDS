#!/usr/bin/env python3
"""Convert PCAP/PCAPNG captures into [segments, micro-bins, 6] tensors."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from data.flow_builder import (
    CaptureSpec,
    SegmentRecord,
    build_capture_segments,
    split_train_calibration,
)
from data.microbin_features import feature_names
from utils.config import load_config, resolve_path


def _normalize_split(value: str) -> str:
    split = value.strip().lower()
    aliases = {"validation": "calibration", "val": "calibration"}
    split = aliases.get(split, split)
    if split not in {"train", "calibration", "test", "train_calibration"}:
        raise ValueError(f"Unsupported split {value!r}")
    return split


def _configured_sources(config: dict[str, Any]) -> list[CaptureSpec]:
    specs: dict[Path, CaptureSpec] = {}
    gotham_manifest_value = config["data"].get("gotham_manifest")
    if gotham_manifest_value:
        manifest_path = resolve_path(config, gotham_manifest_value)
        workspace_root = resolve_path(
            config, config["data"].get("gotham_workspace_root", "..")
        )
        assert manifest_path is not None and workspace_root is not None
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        training_name = str(manifest_data["training"]["source_dataset"])
        requested = config["data"].get("evaluation_datasets")
        evaluation_names = (
            [str(name) for name in requested]
            if requested
            else [str(name) for name in manifest_data["evaluation"]["datasets"]]
        )
        include_training = bool(config["data"].get("include_gotham_training", True))
        include_evaluation = bool(config["data"].get("include_gotham_evaluation", True))
        selected_names = (
            ([training_name] if include_training else [])
            + (evaluation_names if include_evaluation else [])
        )
        for dataset_name in selected_names:
            if dataset_name not in manifest_data["datasets"]:
                raise ValueError(f"Dataset {dataset_name!r} is absent from Gotham manifest")
            row = manifest_data["datasets"][dataset_name]
            path = (workspace_root / row["pcap"]).resolve()
            is_training = dataset_name == training_name
            label_path = None if is_training else (workspace_root / row["labels"]).resolve()
            specs[path] = CaptureSpec(
                path=path,
                label="normal",
                split="train_calibration" if is_training else "test",
                packet_labels=label_path,
                dataset_name=dataset_name,
            )
    for source in config["data"].get("sources", []):
        pattern = resolve_path(config, source["glob"])
        assert pattern is not None
        for value in glob.glob(str(pattern), recursive=True):
            path = Path(value).resolve()
            if path.suffix.lower() in {".pcap", ".pcapng"}:
                specs[path] = CaptureSpec(
                    path, str(source["label"]).strip(), _normalize_split(source["split"])
                )
    manifest_value = config["data"].get("manifest")
    if manifest_value:
        manifest = resolve_path(config, manifest_value)
        assert manifest is not None
        with manifest.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"pcap", "label", "split"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError("Manifest must contain pcap,label,split columns")
            for row in reader:
                path = Path(row["pcap"]).expanduser()
                path = path.resolve() if path.is_absolute() else (manifest.parent / path).resolve()
                specs[path] = CaptureSpec(path, row["label"].strip(), _normalize_split(row["split"]))
    missing = [str(path) for path in specs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Capture files do not exist: {missing[:5]}")
    missing_labels = [
        str(spec.packet_labels)
        for spec in specs.values()
        if spec.packet_labels is not None and not spec.packet_labels.is_file()
    ]
    if missing_labels:
        raise FileNotFoundError(f"Packet label files do not exist: {missing_labels[:5]}")
    return sorted(specs.values(), key=lambda item: str(item.path))


def prepare(config: dict[str, Any], extra_pcaps: list[str] | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    gotham_manifest_value = config["data"].get("gotham_manifest")
    specs = _configured_sources(config)
    label_from_parent = bool(config["data"].get("label_from_parent", False))
    for value in extra_pcaps or []:
        path = Path(value).expanduser().resolve()
        label = path.parent.name if label_from_parent else "__unlabeled__"
        specs.append(CaptureSpec(path, label, "test"))
    if not specs:
        raise ValueError("No PCAP/PCAPNG files matched data.sources or data.manifest")

    reader_path = resolve_path(config, config["data"]["baseline_reader_path"])
    assert reader_path is not None
    records: list[SegmentRecord] = []
    for spec in specs:
        built = build_capture_segments(
            spec,
            reader_path=reader_path,
            allowed_protocols={str(x).lower() for x in config["data"]["allowed_protocols"]},
            window_seconds=float(config["data"]["flow_window_seconds"]),
            micro_bin_seconds=float(config["data"]["micro_bin_seconds"]),
            min_packets=int(config["data"]["min_packets_per_segment"]),
            flow_orientation=str(
                config["data"].get("flow_orientation", "bidirectional")
            ).lower(),
            window_alignment=str(
                config["data"].get("window_alignment", "capture")
            ).lower(),
            microbin_feature_profile=str(
                config["data"].get(
                    "microbin_feature_profile", "bidirectional_basic_v1"
                )
            ).lower(),
        )
        if spec.split == "train_calibration":
            built = split_train_calibration(
                built, float(config["data"].get("calibration_fraction", 0.2))
            )
        records.extend(built)
        split_summary = dict(Counter(row.split for row in built))
        label_summary = dict(Counter(row.label_name for row in built))
        print(
            f"capture={spec.path.name} dataset={spec.dataset_name or '-'} "
            f"splits={split_summary} labels={label_summary} segments={len(built)}"
        )
    if not records:
        raise ValueError("No eligible TCP/UDP flow segments were found")

    label_names = sorted({row.label_name for row in records if row.label_name != "__unlabeled__"})
    label_to_id = {name: index for index, name in enumerate(label_names)}
    normal_names = {str(name).lower() for name in config["data"]["normal_labels"]}
    attack_names = {str(name).lower() for name in config["data"]["attack_labels"]}
    unknown = [name for name in label_names if name.lower() not in normal_names | attack_names]
    if unknown:
        raise ValueError(
            f"Labels {unknown} are neither normal nor attack; update data.normal_labels/attack_labels"
        )
    normal_ids = [idx for name, idx in label_to_id.items() if name.lower() in normal_names]
    attack_ids = [idx for name, idx in label_to_id.items() if name.lower() in attack_names]
    split_ids = {"train": 0, "calibration": 1, "test": 2}

    def array(attribute: str) -> np.ndarray:
        return np.asarray([getattr(row, attribute) for row in records])

    output = resolve_path(config, config["data"]["processed_path"])
    metadata_path = resolve_path(config, config["data"]["metadata_path"])
    assert output is not None and metadata_path is not None
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=np.stack([row.features for row in records]).astype(np.float32),
        labels=np.asarray([label_to_id.get(row.label_name, -1) for row in records], dtype=np.int64),
        split=np.asarray([split_ids[row.split] for row in records], dtype=np.int8),
        segment_ids=array("segment_id"),
        flow_ids=array("flow_id"),
        captures=array("capture"),
        capture_ids=array("capture_id"),
        endpoint_a_ips=array("endpoint_a_ip"),
        endpoint_b_ips=array("endpoint_b_ip"),
        endpoint_a_ports=array("endpoint_a_port").astype(np.int32),
        endpoint_b_ports=array("endpoint_b_port").astype(np.int32),
        protocols=array("protocol"),
        window_indices=array("window_index").astype(np.int64),
        segment_starts=array("segment_start").astype(np.float64),
        packet_counts=array("packet_count").astype(np.int64),
        byte_counts=array("byte_count").astype(np.int64),
        dataset_names=array("dataset_name"),
    )
    metadata = {
        "format_version": 1,
        "num_segments": len(records),
        "num_captures": len(specs),
        "feature_shape": [int(round(float(config["data"]["flow_window_seconds"]) / float(config["data"]["micro_bin_seconds"]))), 6],
        "num_bins": int(round(float(config["data"]["flow_window_seconds"]) / float(config["data"]["micro_bin_seconds"]))),
        "feature_names": list(
            feature_names(
                str(
                    config["data"].get(
                        "microbin_feature_profile", "bidirectional_basic_v1"
                    )
                ).lower()
            )
        ),
        "microbin_feature_profile": str(
            config["data"].get(
                "microbin_feature_profile", "bidirectional_basic_v1"
            )
        ).lower(),
        "flow_window_seconds": float(config["data"]["flow_window_seconds"]),
        "micro_bin_seconds": float(config["data"]["micro_bin_seconds"]),
        "flow_orientation": str(
            config["data"].get("flow_orientation", "bidirectional")
        ).lower(),
        "window_alignment": str(
            config["data"].get("window_alignment", "capture")
        ).lower(),
        "direction_rule": (
            "A is packet source and B is packet destination; reverse traffic has a separate flow key"
            if str(config["data"].get("flow_orientation", "bidirectional")).lower()
            == "directional"
            else "A is the lexicographically smaller (IP binary value, port) endpoint"
        ),
        "window_rule": (
            "epoch-aligned half-open windows [kW, (k+1)W)"
            if str(config["data"].get("window_alignment", "capture")).lower() == "epoch"
            else "windows are aligned to the first eligible packet timestamp of each capture"
        ),
        "label_names": label_names,
        "label_to_id": label_to_id,
        "normal_label_ids": normal_ids,
        "attack_label_ids": attack_ids,
        "split_names": {"0": "train", "1": "calibration", "2": "test"},
        "split_counts": dict(Counter(row.split for row in records)),
        "label_counts": dict(Counter(row.label_name for row in records)),
        "captures": [str(spec.path) for spec in specs],
        "baseline_reader": str(reader_path),
        "gotham_manifest": str(resolve_path(config, gotham_manifest_value)) if gotham_manifest_value else None,
        "label_policy": "flow segment is attack when any eligible constituent packet has binary_label=1",
        "preprocessing_runtime": {
            "seconds": float(time.perf_counter() - started),
            "packets": int(sum(row.packet_count for row in records)),
        },
    }
    runtime = metadata["preprocessing_runtime"]
    assert isinstance(runtime, dict)
    seconds = float(runtime["seconds"])
    runtime["packets_per_second"] = float(int(runtime["packets"]) / max(seconds, 1e-12))
    runtime["segments_per_second"] = float(len(records) / max(seconds, 1e-12))
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"prepared segments={len(records)} shape={tuple(metadata['feature_shape'])} "
        f"packets_per_second={runtime['packets_per_second']:.1f} output={output}"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--pcap", action="append", default=[], help="Extra test PCAP; repeatable")
    args = parser.parse_args()
    prepare(load_config(args.config), args.pcap)


if __name__ == "__main__":
    main()
