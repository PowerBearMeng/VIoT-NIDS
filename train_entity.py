#!/usr/bin/env python3
"""Train the entity temporal GRU on normal entity states only."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data.dataset import FlowDataset
from data.entity_state import EntitySequences, build_entity_sequences
from models.entity_predictor import EntityGRUPredictor
from utils.config import load_config, resolve_path
from utils.io import require_alignment, save_torch_checkpoint
from utils.scaling import MixedFeatureStandardizer
from utils.seed import choose_device, seed_everything
from utils.training import EarlyStopping, mean_batch_loss


def _load_inputs(config: dict[str, Any]) -> tuple[FlowDataset, np.ndarray, np.ndarray, int]:
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
        assignments = archive["assignments"].astype(np.int64)
        prototype_count = int(archive["centers"].shape[0])
    return dataset, embeddings, assignments, prototype_count


def _sequences(
    config: dict[str, Any],
    dataset: FlowDataset,
    embeddings: np.ndarray,
    assignments: np.ndarray,
    prototype_count: int,
    indices: np.ndarray,
) -> EntitySequences:
    section = config["entity_model"]
    return build_entity_sequences(
        dataset,
        embeddings,
        assignments,
        prototype_count=prototype_count,
        indices=indices,
        history_windows=int(section["history_windows"]),
        include_mean_embedding=bool(section["include_mean_embedding"]),
    )


def train(config: dict[str, Any], device_name: str | None = None) -> dict[str, Any]:
    seed_everything(int(config["seed"]))
    device = choose_device(device_name or config["runtime"].get("device"))
    dataset, embeddings, assignments, prototype_count = _load_inputs(config)
    train_indices = dataset.require_normal("train", "entity model training")
    calibration_indices = dataset.require_normal("calibration", "entity early stopping")
    train_samples = _sequences(config, dataset, embeddings, assignments, prototype_count, train_indices)
    calibration_samples = _sequences(config, dataset, embeddings, assignments, prototype_count, calibration_indices)
    if not len(train_samples) or not len(calibration_samples):
        raise ValueError("Entity model requires nonempty normal train and calibration states")
    log_dimensions = 3 + prototype_count
    scaler = MixedFeatureStandardizer.fit(train_samples.targets, log_dimensions)
    train_history = scaler.transform(train_samples.histories)
    train_targets = scaler.transform(train_samples.targets)
    calibration_history = scaler.transform(calibration_samples.histories)
    calibration_targets = scaler.transform(calibration_samples.targets)
    section = config["entity_model"]
    batch_size = int(section["batch_size"])
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_history), torch.from_numpy(train_targets)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(config["runtime"].get("num_workers", 0)),
    )
    calibration_loader = DataLoader(
        TensorDataset(torch.from_numpy(calibration_history), torch.from_numpy(calibration_targets)),
        batch_size=batch_size,
        shuffle=False,
    )
    parameters = {
        "state_dim": int(train_targets.shape[1]),
        "hidden_dim": int(section["hidden_dim"]),
        "num_layers": int(section["num_layers"]),
        "dropout": float(section["dropout"]),
    }
    model = EntityGRUPredictor(**parameters).to(device)
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
        for raw_history, raw_target in train_loader:
            batch = (raw_history.to(device), raw_target.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(batch)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(raw_history)
            seen += len(raw_history)
        train_loss = total / max(1, seen)
        calibration_loss = mean_batch_loss(model, calibration_loader, device, loss_fn)
        history.append({"epoch": epoch, "train_loss": train_loss, "calibration_loss": calibration_loss})
        print(f"entity epoch={epoch} train_loss={train_loss:.6f} calibration_loss={calibration_loss:.6f}")
        if stopper.update(calibration_loss, epoch, model):
            break
    if stopper.best_state is None:
        raise RuntimeError("Entity model training did not produce a checkpoint")
    model.load_state_dict(stopper.best_state)
    checkpoint_path = resolve_path(config, config["runtime"]["entity_checkpoint"])
    assert checkpoint_path is not None
    save_torch_checkpoint(
        checkpoint_path,
        {
            "format_version": 1,
            "model_state": model.state_dict(),
            "model_parameters": parameters,
            "state_scaler": scaler.state_dict(),
            "prototype_count": prototype_count,
            "history_windows": int(section["history_windows"]),
            "include_mean_embedding": bool(section["include_mean_embedding"]),
            "feature_names": train_samples.feature_names,
            "best_epoch": stopper.best_epoch,
            "best_calibration_loss": stopper.best_loss,
            "normal_train_entity_states": len(train_samples),
            "history": history,
        },
    )
    result = {"checkpoint": str(checkpoint_path), "entity_states": len(train_samples), "state_dim": parameters["state_dim"]}
    print(f"entity training complete {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    train(load_config(args.config), args.device)


if __name__ == "__main__":
    main()
