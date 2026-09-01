#!/usr/bin/env python3
"""Fit the Design V3 frozen normal behavior-composition reference."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from data.behavior_composition import fit_behavior_reference
from data.dataset import FlowDataset
from utils.config import load_config, resolve_path
from utils.io import require_alignment


def train(config: dict[str, Any]) -> dict[str, Any]:
    dataset_path = resolve_path(config, config["data"]["processed_path"])
    metadata_path = resolve_path(config, config["data"]["metadata_path"])
    prototypes_path = resolve_path(config, config["runtime"]["prototypes_path"])
    output_path = resolve_path(config, config["runtime"]["context_reference_path"])
    assert dataset_path is not None and metadata_path is not None
    assert prototypes_path is not None and output_path is not None
    dataset = FlowDataset(dataset_path, metadata_path)
    train_indices = dataset.require_normal("train", "V3 behavior-composition reference")
    with np.load(prototypes_path, allow_pickle=False) as archive:
        require_alignment(dataset.segment_ids, archive["segment_ids"], "Prototype artifact")
        mode_ids = archive["assignments"].astype(np.int64)
        prototype_count = int(archive["centers"].shape[0])
    section = config["context_model"]
    reference = fit_behavior_reference(
        dataset,
        mode_ids,
        train_indices,
        prototype_count=prototype_count,
        use_log_count=bool(section.get("use_log_count", True)),
        epsilon=float(section.get("epsilon", 1e-3)),
    )
    reference.save(output_path)
    result = {
        "path": str(output_path),
        "normal_train_segments": int(len(train_indices)),
        "prototype_count": prototype_count,
        "known_pairs": len(reference.pair_mean),
        "known_entities": len(reference.entity_mean),
        "history": "frozen_train_reference",
        "ports_used": False,
    }
    print(f"behavior-composition reference complete {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
