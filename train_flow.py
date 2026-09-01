#!/usr/bin/env python3
"""Train the masked-reconstruction TCN strictly on normal flow segments."""

from __future__ import annotations

import argparse
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

from data.dataset import FlowDataset
from models.flow_encoder import FlowAutoencoder
from utils.config import load_config, resolve_path
from utils.flow_runtime import deterministic_masks, model_parameters
from utils.io import save_torch_checkpoint
from utils.scaling import FeatureStandardizer
from utils.seed import choose_device, seed_everything
from utils.training import EarlyStopping, mean_batch_loss


def _masked_loss(
    model: FlowAutoencoder,
    values: torch.Tensor,
    mask: torch.Tensor,
    occupancy: torch.Tensor | None = None,
    nonempty_loss_weight: float = 1.0,
) -> torch.Tensor:
    reconstruction, _ = model(values, mask)
    squared = (reconstruction - values).square().mean(dim=-1)
    weights = torch.ones_like(squared)
    if occupancy is not None:
        weights = torch.where(
            occupancy,
            torch.full_like(squared, float(nonempty_loss_weight)),
            weights,
        )
    selected = weights * mask
    return (squared * selected).sum() / selected.sum().clamp_min(1)


def _force_one_occupied_mask(mask: torch.Tensor, occupancy: torch.Tensor) -> torch.Tensor:
    """Ensure sparse flows teach the model about at least one real event."""

    missing = occupancy.any(dim=1) & ~(mask & occupancy).any(dim=1)
    for row in torch.nonzero(missing, as_tuple=False).flatten().tolist():
        first = int(torch.nonzero(occupancy[row], as_tuple=False).flatten()[0].item())
        mask[row, first] = True
    return mask


def train(config: dict[str, Any], device_name: str | None = None) -> dict[str, Any]:
    seed = int(config["seed"])
    seed_everything(seed)
    device = choose_device(device_name or config["runtime"].get("device"))
    dataset_path = resolve_path(config, config["data"]["processed_path"])
    metadata_path = resolve_path(config, config["data"]["metadata_path"])
    assert dataset_path is not None and metadata_path is not None
    dataset = FlowDataset(dataset_path, metadata_path)
    train_indices = dataset.require_normal("train", "flow model training")
    calibration_indices = dataset.require_normal("calibration", "flow early stopping")

    scaler = FeatureStandardizer.fit(dataset.features[train_indices])
    train_values = scaler.transform(dataset.features[train_indices])
    calibration_values = scaler.transform(dataset.features[calibration_indices])
    train_occupancy = (dataset.features[train_indices, :, 0] + dataset.features[train_indices, :, 1]) > 0
    calibration_occupancy = (
        dataset.features[calibration_indices, :, 0] + dataset.features[calibration_indices, :, 1]
    ) > 0
    section = config["flow_model"]
    architecture = str(section.get("architecture", "v1")).lower()
    event_aware = architecture == "v2"
    batch_size = int(section["batch_size"])
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_values), torch.from_numpy(train_occupancy)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(config["runtime"].get("num_workers", 0)),
    )
    calibration_masks = deterministic_masks(
        dataset.segment_ids[calibration_indices],
        calibration_values.shape[1],
        float(section["mask_ratio"]),
        seed,
        occupancy=calibration_occupancy,
        force_occupied=event_aware,
    )
    calibration_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(calibration_values),
            torch.from_numpy(calibration_masks),
            torch.from_numpy(calibration_occupancy),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    parameters = model_parameters(config)
    model = FlowAutoencoder(**parameters).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(section["learning_rate"]),
        weight_decay=float(section["weight_decay"]),
    )
    stopper = EarlyStopping(int(section["early_stopping_patience"]))
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    history: list[dict[str, float | int]] = []
    for epoch in range(1, int(section["epochs"]) + 1):
        model.train()
        total = 0.0
        seen = 0
        for raw_values, raw_occupancy in train_loader:
            values = raw_values.to(device)
            occupancy = raw_occupancy.to(device)
            mask = torch.rand(values.shape[:2], generator=generator, device=device) < float(section["mask_ratio"])
            empty = ~mask.any(dim=1)
            if empty.any():
                mask[empty, 0] = True
            if event_aware:
                mask = _force_one_occupied_mask(mask, occupancy)
            optimizer.zero_grad(set_to_none=True)
            loss = _masked_loss(
                model,
                values,
                mask,
                occupancy if event_aware else None,
                float(section.get("nonempty_loss_weight", 1.0)),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(section["gradient_clip"]))
            optimizer.step()
            total += float(loss.item()) * len(values)
            seen += len(values)
        train_loss = total / max(1, seen)
        calibration_loss = mean_batch_loss(
            model,
            calibration_loader,
            device,
            lambda batch: _masked_loss(
                model,
                batch[0],
                batch[1],
                batch[2] if event_aware else None,
                float(section.get("nonempty_loss_weight", 1.0)),
            ),
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "calibration_loss": calibration_loss})
        print(f"flow epoch={epoch} train_loss={train_loss:.6f} calibration_loss={calibration_loss:.6f}")
        if stopper.update(calibration_loss, epoch, model):
            break
    if stopper.best_state is None:
        raise RuntimeError("Flow model training did not produce a checkpoint")
    model.load_state_dict(stopper.best_state)
    checkpoint_path = resolve_path(config, config["runtime"]["flow_checkpoint"])
    assert checkpoint_path is not None
    save_torch_checkpoint(
        checkpoint_path,
        {
            "format_version": 1,
            "model_state": model.state_dict(),
            "model_parameters": parameters,
            "feature_scaler": scaler.state_dict(),
            "best_epoch": stopper.best_epoch,
            "best_calibration_loss": stopper.best_loss,
            "normal_train_segments": int(len(train_indices)),
            "history": history,
            "seed": seed,
        },
    )
    result = {"checkpoint": str(checkpoint_path), "best_epoch": stopper.best_epoch, "best_calibration_loss": stopper.best_loss}
    print(f"flow training complete {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    train(load_config(args.config), args.device)


if __name__ == "__main__":
    main()
