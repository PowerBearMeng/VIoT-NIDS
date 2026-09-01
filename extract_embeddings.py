#!/usr/bin/env python3
"""Extract unmasked flow embeddings and deterministic masked errors."""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np

from data.dataset import FlowDataset
from utils.config import load_config, resolve_path
from utils.flow_runtime import infer_flow_batches, load_flow_model
from utils.seed import choose_device, seed_everything


def extract(config: dict[str, Any], device_name: str | None = None) -> dict[str, Any]:
    seed = int(config["seed"])
    seed_everything(seed)
    device = choose_device(device_name or config["runtime"].get("device"))
    dataset_path = resolve_path(config, config["data"]["processed_path"])
    metadata_path = resolve_path(config, config["data"]["metadata_path"])
    checkpoint_path = resolve_path(config, config["runtime"]["flow_checkpoint"])
    output_path = resolve_path(config, config["runtime"]["embeddings_path"])
    assert dataset_path is not None and metadata_path is not None and checkpoint_path is not None and output_path is not None
    dataset = FlowDataset(dataset_path, metadata_path)
    model, scaler, _ = load_flow_model(str(checkpoint_path), device)
    values = scaler.transform(dataset.features)
    occupancy = (dataset.features[:, :, 0] + dataset.features[:, :, 1]) > 0
    started = time.perf_counter()
    embeddings, errors = infer_flow_batches(
        model,
        values,
        dataset.segment_ids,
        batch_size=int(config["flow_model"]["batch_size"]),
        mask_ratio=float(config["flow_model"]["score_mask_ratio"]),
        seed=seed + 1009,
        device=device,
        occupancy=occupancy,
        score_mask_rounds=int(config["flow_model"].get("score_mask_rounds", 1)),
        active_error_weight=float(config["flow_model"].get("active_error_weight", 0.5)),
    )
    inference_seconds = time.perf_counter() - started
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        segment_ids=dataset.segment_ids,
        embeddings=embeddings.astype(np.float32),
        reconstruction_error=errors.astype(np.float64),
        inference_seconds=np.asarray(inference_seconds, dtype=np.float64),
        segments_per_second=np.asarray(len(dataset) / max(inference_seconds, 1e-12), dtype=np.float64),
    )
    result = {
        "path": str(output_path),
        "segments": len(dataset),
        "embedding_shape": list(embeddings.shape),
        "mean_reconstruction_error": float(errors.mean()),
        "inference_seconds": float(inference_seconds),
        "segments_per_second": float(len(dataset) / max(inference_seconds, 1e-12)),
    }
    print(f"embeddings complete {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    extract(load_config(args.config), args.device)


if __name__ == "__main__":
    main()
