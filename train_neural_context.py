#!/usr/bin/env python3
"""Train the V4 continuous neural behavior-intensity model on normal traffic."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import torch

from data.dataset import FlowDataset
from data.neural_intensity import (
    TorchScopeIndex,
    aggregate_soft_masses_torch,
    build_scope_index,
    to_torch,
)
from models.neural_intensity_context import NeuralIntensityContext
from utils.config import load_config, resolve_path
from utils.io import require_alignment, save_torch_checkpoint
from utils.scaling import VectorStandardizer
from utils.seed import choose_device, seed_everything
from utils.training import EarlyStopping


def _loss(
    model: NeuralIntensityContext,
    embeddings: torch.Tensor,
    scope: TorchScopeIndex,
    *,
    pair_enabled: bool,
    entity_enabled: bool,
    balance_weight: float,
    entropy_weight: float,
    regularize_assignments: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    assignments = model.assignments(embeddings)
    pair_observed, entity_a_observed, entity_b_observed = (
        aggregate_soft_masses_torch(assignments, scope)
    )
    pair_mean, pair_log_scale, entity_mean, entity_log_scale = (
        model.expected_parameters(embeddings)
    )
    terms: list[torch.Tensor] = []
    pair_nll = model.gaussian_nll(
        pair_observed, pair_mean, pair_log_scale
    ).mean()
    entity_nll = 0.5 * (
        model.gaussian_nll(
            entity_a_observed, entity_mean, entity_log_scale
        ).mean()
        + model.gaussian_nll(
            entity_b_observed, entity_mean, entity_log_scale
        ).mean()
    )
    if pair_enabled:
        terms.append(pair_nll)
    if entity_enabled:
        terms.append(entity_nll)
    if not terms:
        raise ValueError("V4 neural context requires pair or entity scope")
    data_loss = torch.stack(terms).mean()
    usage = assignments.mean(dim=0).clamp_min(1e-8)
    balance = torch.sum(usage * torch.log(usage * assignments.shape[1]))
    entropy = -torch.sum(
        assignments.clamp_min(1e-8) * torch.log(assignments.clamp_min(1e-8)),
        dim=1,
    ).mean()
    loss = data_loss
    if regularize_assignments:
        loss = loss + balance_weight * balance + entropy_weight * entropy
    return loss, {
        "data_loss": float(data_loss.detach().item()),
        "pair_nll": float(pair_nll.detach().item()),
        "entity_nll": float(entity_nll.detach().item()),
        "assignment_balance": float(balance.detach().item()),
        "assignment_entropy": float(entropy.detach().item()),
    }


def train(
    config: dict[str, Any], device_name: str | None = None
) -> dict[str, Any]:
    seed_everything(int(config["seed"]))
    device = choose_device(device_name or config["runtime"].get("device"))
    dataset_path = resolve_path(config, config["data"]["processed_path"])
    metadata_path = resolve_path(config, config["data"]["metadata_path"])
    embeddings_path = resolve_path(config, config["runtime"]["embeddings_path"])
    checkpoint_path = resolve_path(
        config, config["runtime"]["neural_context_checkpoint"]
    )
    assert dataset_path is not None and metadata_path is not None
    assert embeddings_path is not None and checkpoint_path is not None
    dataset = FlowDataset(dataset_path, metadata_path)
    with np.load(embeddings_path, allow_pickle=False) as archive:
        require_alignment(dataset.segment_ids, archive["segment_ids"], "Embedding artifact")
        embeddings = archive["embeddings"].astype(np.float32)
    train_indices = dataset.require_normal("train", "V4 neural context training")
    calibration_indices = dataset.require_normal(
        "calibration", "V4 neural context early stopping"
    )
    scaler = VectorStandardizer.fit(embeddings[train_indices])
    train_embeddings = torch.as_tensor(
        scaler.transform(embeddings[train_indices]),
        dtype=torch.float32,
        device=device,
    )
    calibration_embeddings = torch.as_tensor(
        scaler.transform(embeddings[calibration_indices]),
        dtype=torch.float32,
        device=device,
    )
    train_scope = to_torch(build_scope_index(dataset, train_indices), device)
    calibration_scope = to_torch(
        build_scope_index(dataset, calibration_indices), device
    )
    section = config["context_model"]
    parameters = {
        "embedding_dim": int(embeddings.shape[1]),
        "hidden_dim": int(section["hidden_dim"]),
        "latent_channels": int(section["latent_channels"]),
        "assignment_temperature": float(section["assignment_temperature"]),
        "min_log_scale": float(section["min_log_scale"]),
        "max_log_scale": float(section["max_log_scale"]),
    }
    model = NeuralIntensityContext(**parameters).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(section["learning_rate"]),
        weight_decay=float(section["weight_decay"]),
    )
    stopper = EarlyStopping(int(section["early_stopping_patience"]))
    pair_enabled = bool(section.get("pair_enabled", True))
    entity_enabled = bool(section.get("entity_enabled", True))
    balance_weight = float(section.get("balance_weight", 0.1))
    entropy_weight = float(section.get("entropy_weight", 0.01))
    history: list[dict[str, float | int]] = []
    for epoch in range(1, int(section["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss, train_parts = _loss(
            model,
            train_embeddings,
            train_scope,
            pair_enabled=pair_enabled,
            entity_enabled=entity_enabled,
            balance_weight=balance_weight,
            entropy_weight=entropy_weight,
            regularize_assignments=True,
        )
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(section.get("gradient_clip", 5.0))
        )
        optimizer.step()
        model.eval()
        with torch.no_grad():
            calibration_loss, calibration_parts = _loss(
                model,
                calibration_embeddings,
                calibration_scope,
                pair_enabled=pair_enabled,
                entity_enabled=entity_enabled,
                balance_weight=balance_weight,
                entropy_weight=entropy_weight,
                regularize_assignments=False,
            )
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": float(train_loss.detach().item()),
            "calibration_loss": float(calibration_loss.item()),
            **{f"train_{key}": value for key, value in train_parts.items()},
            **{
                f"calibration_{key}": value
                for key, value in calibration_parts.items()
            },
        }
        history.append(record)
        print(
            f"neural-context epoch={epoch} "
            f"train_loss={record['train_loss']:.6f} "
            f"calibration_loss={record['calibration_loss']:.6f} "
            f"entropy={record['train_assignment_entropy']:.4f}"
        )
        if stopper.update(float(calibration_loss.item()), epoch, model):
            break
    if stopper.best_state is None:
        raise RuntimeError("V4 neural context training did not produce a checkpoint")
    model.load_state_dict(stopper.best_state)
    save_torch_checkpoint(
        checkpoint_path,
        {
            "format_version": 4,
            "model_state": model.state_dict(),
            "model_parameters": parameters,
            "embedding_scaler": scaler.state_dict(),
            "pair_enabled": pair_enabled,
            "entity_enabled": entity_enabled,
            "ports_used": False,
            "hard_mode_ids_used": False,
            "normal_train_segments": int(len(train_indices)),
            "normal_calibration_segments": int(len(calibration_indices)),
            "best_epoch": stopper.best_epoch,
            "best_calibration_loss": stopper.best_loss,
            "history": history,
        },
    )
    result = {
        "checkpoint": str(checkpoint_path),
        "normal_train_segments": int(len(train_indices)),
        "normal_calibration_segments": int(len(calibration_indices)),
        "latent_channels": int(parameters["latent_channels"]),
        "hard_mode_ids_used": False,
        "ports_used": False,
    }
    print(f"V4 neural context training complete {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gotham_v4_train.yaml")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    train(load_config(args.config), args.device)


if __name__ == "__main__":
    main()
