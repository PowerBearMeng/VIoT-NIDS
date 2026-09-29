#!/usr/bin/env python3
"""Train the V4.5 target-conditioned sparse context on normal traffic."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import torch

from data.dataset import FlowDataset
from data.sparse_context import (
    SparseContextObservations,
    TorchScopeIndex,
    build_scope_index,
    compute_sparse_observations,
    to_torch,
)
from models.sparse_context import SparseContextModel
from utils.config import load_config, resolve_path
from utils.io import require_alignment, save_torch_checkpoint
from utils.scaling import VectorStandardizer
from utils.seed import choose_device, seed_everything
from utils.training import EarlyStopping


def _mean_diagnostics(
    observations: SparseContextObservations,
) -> dict[str, float]:
    return {
        "pair_null_weight": float(observations.pair.null_weight.mean().detach().item()),
        "pair_support_size": float(
            observations.pair.support_size.to(torch.float32).mean().detach().item()
        ),
        "pair_effective_neighbors": float(
            observations.pair.effective_neighbors.mean().detach().item()
        ),
        "entity_null_weight": float(
            0.5
            * (
                observations.entity_a.null_weight.mean()
                + observations.entity_b.null_weight.mean()
            ).detach().item()
        ),
        "entity_support_size": float(
            0.5
            * (
                observations.entity_a.support_size.to(torch.float32).mean()
                + observations.entity_b.support_size.to(torch.float32).mean()
            ).detach().item()
        ),
        "entity_effective_neighbors": float(
            0.5
            * (
                observations.entity_a.effective_neighbors.mean()
                + observations.entity_b.effective_neighbors.mean()
            ).detach().item()
        ),
    }


def _loss(
    model: SparseContextModel,
    embeddings: torch.Tensor,
    scope: TorchScopeIndex,
    *,
    pair_enabled: bool,
    entity_enabled: bool,
    attention_epsilon: float,
    attention_chunk_size: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    attention_parameters = model.attention_parameters(embeddings)
    observations = compute_sparse_observations(
        *attention_parameters,
        scope,
        epsilon=attention_epsilon,
        chunk_size=attention_chunk_size,
    )
    pair_mean, pair_log_scale, entity_mean, entity_log_scale = (
        model.expected_parameters(embeddings)
    )
    pair_nll = model.gaussian_nll(
        observations.pair.observed_intensity, pair_mean, pair_log_scale
    ).mean()
    entity_nll = 0.5 * (
        model.gaussian_nll(
            observations.entity_a.observed_intensity,
            entity_mean,
            entity_log_scale,
        ).mean()
        + model.gaussian_nll(
            observations.entity_b.observed_intensity,
            entity_mean,
            entity_log_scale,
        ).mean()
    )
    terms: list[torch.Tensor] = []
    if pair_enabled:
        terms.append(pair_nll)
    if entity_enabled:
        terms.append(entity_nll)
    if not terms:
        raise ValueError("V4.5 sparse context requires pair or entity scope")
    loss = torch.stack(terms).mean()
    return loss, {
        "data_loss": float(loss.detach().item()),
        "pair_nll": float(pair_nll.detach().item()),
        "entity_nll": float(entity_nll.detach().item()),
        **_mean_diagnostics(observations),
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
        config, config["runtime"]["sparse_context_checkpoint"]
    )
    assert dataset_path is not None and metadata_path is not None
    assert embeddings_path is not None and checkpoint_path is not None
    dataset = FlowDataset(dataset_path, metadata_path)
    with np.load(embeddings_path, allow_pickle=False) as archive:
        require_alignment(
            dataset.segment_ids, archive["segment_ids"], "Embedding artifact"
        )
        embeddings = archive["embeddings"].astype(np.float32)
    train_indices = dataset.require_normal("train", "V4.5 sparse context training")
    calibration_indices = dataset.require_normal(
        "calibration", "V4.5 sparse context early stopping"
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
        "attention_dim": int(section["attention_dim"]),
        "min_log_scale": float(section["min_log_scale"]),
        "max_log_scale": float(section["max_log_scale"]),
    }
    model = SparseContextModel(**parameters).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(section["learning_rate"]),
        weight_decay=float(section["weight_decay"]),
    )
    stopper = EarlyStopping(int(section["early_stopping_patience"]))
    pair_enabled = bool(section.get("pair_enabled", True))
    entity_enabled = bool(section.get("entity_enabled", True))
    attention_epsilon = float(section.get("attention_epsilon", 1e-8))
    attention_chunk_size = int(section.get("attention_chunk_size", 512))
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
            attention_epsilon=attention_epsilon,
            attention_chunk_size=attention_chunk_size,
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
                attention_epsilon=attention_epsilon,
                attention_chunk_size=attention_chunk_size,
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
            f"sparse-context epoch={epoch} "
            f"train_loss={record['train_loss']:.6f} "
            f"calibration_loss={record['calibration_loss']:.6f} "
            f"pair_null={record['train_pair_null_weight']:.4f} "
            f"entity_null={record['train_entity_null_weight']:.4f}"
        )
        if stopper.update(float(calibration_loss.item()), epoch, model):
            break
    if stopper.best_state is None:
        raise RuntimeError("V4.5 sparse context training did not produce a checkpoint")
    model.load_state_dict(stopper.best_state)
    save_torch_checkpoint(
        checkpoint_path,
        {
            "format_version": "4.5",
            "context_mode": "sparse_context",
            "model_state": model.state_dict(),
            "model_parameters": parameters,
            "embedding_scaler": scaler.state_dict(),
            "pair_enabled": pair_enabled,
            "entity_enabled": entity_enabled,
            "attention_epsilon": attention_epsilon,
            "attention_chunk_size": attention_chunk_size,
            "ports_used": False,
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
        "attention_dim": int(parameters["attention_dim"]),
        "ports_used": False,
    }
    print(f"V4.5 sparse context training complete {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gotham_v45_train.yaml")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    train(load_config(args.config), args.device)


if __name__ == "__main__":
    main()
