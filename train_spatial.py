#!/usr/bin/env python3
"""Train reliability-aware spatial edge prediction on normal graphs."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data.dataset import FlowDataset
from data.spatial_context import (
    build_historical_spatial_samples,
    build_spatial_samples,
    initial_historical_state,
)
from models.spatial_predictor import SpatialContextPredictor
from utils.config import load_config, resolve_path
from utils.io import require_alignment, save_torch_checkpoint
from utils.scaling import MixedFeatureStandardizer
from utils.seed import choose_device, seed_everything
from utils.training import EarlyStopping, mean_batch_loss


def train(config: dict[str, Any], device_name: str | None = None) -> dict[str, Any]:
    seed_everything(int(config["seed"]))
    device = choose_device(device_name or config["runtime"].get("device"))
    dataset_path = resolve_path(config, config["data"]["processed_path"])
    metadata_path = resolve_path(config, config["data"]["metadata_path"])
    embeddings_path = resolve_path(config, config["runtime"]["embeddings_path"])
    prototypes_path = resolve_path(config, config["runtime"]["prototypes_path"])
    assert dataset_path is not None and metadata_path is not None and embeddings_path is not None and prototypes_path is not None
    dataset = FlowDataset(dataset_path, metadata_path)
    with np.load(embeddings_path, allow_pickle=False) as archive:
        require_alignment(dataset.segment_ids, archive["segment_ids"], "Embedding artifact")
        embeddings = archive["embeddings"].astype(np.float32)
    with np.load(prototypes_path, allow_pickle=False) as archive:
        require_alignment(dataset.segment_ids, archive["segment_ids"], "Prototype artifact")
        score_source = str(config["spatial_model"].get("local_score_source", "local"))
        if score_source == "combined" and "combined_local_score" in archive:
            local_score = archive["combined_local_score"].astype(np.float64)
        elif score_source == "local":
            local_score = archive["local_score"].astype(np.float64)
        else:
            raise ValueError(f"Unavailable spatial_model.local_score_source: {score_source}")
    section = config["spatial_model"]
    train_indices = dataset.require_normal("train", "spatial model training")
    calibration_indices = dataset.require_normal("calibration", "spatial early stopping")
    kwargs = {"alpha": float(section["alpha"]), "min_reliability": float(section["min_reliability"])}
    context_mode = str(section.get("context_mode", "current_window")).lower()
    historical_parameters: dict[str, float] = {}
    historical_reference_state: dict[str, Any] | None = None
    if context_mode == "historical":
        historical_parameters = {
            "history_beta": float(section.get("history_beta", 1.0)),
            "state_update_rate": float(section.get("state_update_rate", 0.1)),
            "multiplicity_gamma": float(section.get("multiplicity_gamma", 0.05)),
        }
        initial_state = initial_historical_state(embeddings[train_indices].mean(axis=0))
        train_samples, historical_reference_state = build_historical_spatial_samples(
            dataset,
            embeddings,
            local_score,
            train_indices,
            initial_state=initial_state,
            reset_each_capture=False,
            **kwargs,
            **historical_parameters,
        )
        calibration_samples, _ = build_historical_spatial_samples(
            dataset,
            embeddings,
            local_score,
            calibration_indices,
            initial_state=historical_reference_state,
            reset_each_capture=True,
            **kwargs,
            **historical_parameters,
        )
    elif context_mode == "current_window":
        train_samples = build_spatial_samples(dataset, embeddings, local_score, train_indices, **kwargs)
        calibration_samples = build_spatial_samples(dataset, embeddings, local_score, calibration_indices, **kwargs)
    else:
        raise ValueError(f"Unknown spatial context mode: {context_mode}")
    scaler = MixedFeatureStandardizer.fit(train_samples.targets, log_dimensions=0)
    embedding_dim = embeddings.shape[1]

    def scale_contexts(values: np.ndarray) -> np.ndarray:
        shaped = values.reshape(-1, 2, embedding_dim)
        return scaler.transform(shaped).reshape(-1, embedding_dim * 2)

    train_contexts = scale_contexts(train_samples.contexts)
    train_targets = scaler.transform(train_samples.targets)
    calibration_contexts = scale_contexts(calibration_samples.contexts)
    calibration_targets = scaler.transform(calibration_samples.targets)
    batch_size = int(section["batch_size"])
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_contexts), torch.from_numpy(train_targets)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(config["runtime"].get("num_workers", 0)),
    )
    calibration_loader = DataLoader(
        TensorDataset(torch.from_numpy(calibration_contexts), torch.from_numpy(calibration_targets)),
        batch_size=batch_size,
        shuffle=False,
    )
    parameters = {"embedding_dim": int(embedding_dim), "hidden_dim": int(section["hidden_dim"])}
    model = SpatialContextPredictor(**parameters).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(section["learning_rate"]), weight_decay=float(section["weight_decay"])
    )
    stopper = EarlyStopping(int(section["early_stopping_patience"]))

    def loss_fn(batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return (model(batch[0]) - batch[1]).square().mean()

    history: list[dict[str, float | int]] = []
    for epoch in range(1, int(section["epochs"]) + 1):
        model.train()
        total = 0.0
        seen = 0
        for raw_context, raw_target in train_loader:
            batch = (raw_context.to(device), raw_target.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(batch)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(raw_context)
            seen += len(raw_context)
        train_loss = total / max(1, seen)
        calibration_loss = mean_batch_loss(model, calibration_loader, device, loss_fn)
        history.append({"epoch": epoch, "train_loss": train_loss, "calibration_loss": calibration_loss})
        print(f"spatial epoch={epoch} train_loss={train_loss:.6f} calibration_loss={calibration_loss:.6f}")
        if stopper.update(calibration_loss, epoch, model):
            break
    if stopper.best_state is None:
        raise RuntimeError("Spatial model training did not produce a checkpoint")
    model.load_state_dict(stopper.best_state)
    checkpoint_path = resolve_path(config, config["runtime"]["spatial_checkpoint"])
    assert checkpoint_path is not None
    save_torch_checkpoint(
        checkpoint_path,
        {
            "format_version": 1,
            "model_state": model.state_dict(),
            "model_parameters": parameters,
            "embedding_scaler": scaler.state_dict(),
            "alpha": float(section["alpha"]),
            "min_reliability": float(section["min_reliability"]),
            "context_mode": context_mode,
            "historical_parameters": historical_parameters,
            "historical_reference_state": historical_reference_state,
            "best_epoch": stopper.best_epoch,
            "best_calibration_loss": stopper.best_loss,
            "normal_train_edges": len(train_samples),
            "history": history,
        },
    )
    result = {"checkpoint": str(checkpoint_path), "normal_train_edges": len(train_samples), "embedding_dim": embedding_dim}
    print(f"spatial training complete {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    train(load_config(args.config), args.device)


if __name__ == "__main__":
    main()
