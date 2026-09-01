"""Construction and inference helpers shared by flow-stage CLIs."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch

from models.flow_encoder import FlowAutoencoder
from utils.io import load_torch_checkpoint
from utils.scaling import FeatureStandardizer


def model_parameters(config: dict[str, Any]) -> dict[str, Any]:
    section = config["flow_model"]
    return {
        "architecture": str(section.get("architecture", "v1")),
        "feature_dim": 6,
        "hidden_channels": int(section["hidden_channels"]),
        "embedding_dim": int(section["embedding_dim"]),
        "blocks": int(section["tcn_blocks"]),
        "kernel_size": int(section["kernel_size"]),
        "dropout": float(section["dropout"]),
    }


def load_flow_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[FlowAutoencoder, FeatureStandardizer, dict[str, Any]]:
    checkpoint = load_torch_checkpoint(checkpoint_path, device)
    model = FlowAutoencoder(**checkpoint["model_parameters"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    scaler = FeatureStandardizer.from_state_dict(checkpoint["feature_scaler"])
    return model, scaler, checkpoint


def deterministic_masks(
    segment_ids: np.ndarray,
    num_bins: int,
    ratio: float,
    seed: int,
    occupancy: np.ndarray | None = None,
    force_occupied: bool = False,
) -> np.ndarray:
    count = max(1, min(num_bins - 1, int(round(num_bins * ratio))))
    result = np.zeros((len(segment_ids), num_bins), dtype=bool)
    for row, segment_id in enumerate(segment_ids.tolist()):
        digest = hashlib.sha1(f"{seed}|{segment_id}".encode("utf-8")).digest()
        local_seed = int.from_bytes(digest[:8], "little", signed=False)
        rng = np.random.default_rng(local_seed)
        result[row, rng.choice(num_bins, size=count, replace=False)] = True
        if force_occupied and occupancy is not None and occupancy[row].any():
            occupied = np.flatnonzero(occupancy[row])
            if not result[row, occupied].any():
                result[row, int(rng.choice(occupied))] = True
    return result


def complementary_masks(
    segment_ids: np.ndarray, num_bins: int, rounds: int, seed: int
) -> np.ndarray:
    """Assign every bin to exactly one deterministic reconstruction round."""

    rounds = max(1, min(int(rounds), num_bins))
    result = np.zeros((rounds, len(segment_ids), num_bins), dtype=bool)
    for row, segment_id in enumerate(segment_ids.tolist()):
        digest = hashlib.sha1(f"{seed}|all|{segment_id}".encode("utf-8")).digest()
        local_seed = int.from_bytes(digest[:8], "little", signed=False)
        permutation = np.random.default_rng(local_seed).permutation(num_bins)
        for round_index, bins in enumerate(np.array_split(permutation, rounds)):
            result[round_index, row, bins] = True
    return result


def _event_weighted_error(
    per_bin_error: torch.Tensor,
    occupancy: torch.Tensor,
    active_error_weight: float,
) -> torch.Tensor:
    occupied = occupancy.to(per_bin_error.dtype)
    empty = (~occupancy).to(per_bin_error.dtype)
    active_mean = (per_bin_error * occupied).sum(dim=-1) / occupied.sum(dim=-1).clamp_min(1.0)
    empty_mean = (per_bin_error * empty).sum(dim=-1) / empty.sum(dim=-1).clamp_min(1.0)
    has_active = occupancy.any(dim=-1)
    has_empty = (~occupancy).any(dim=-1)
    combined = float(active_error_weight) * active_mean + (1.0 - float(active_error_weight)) * empty_mean
    combined = torch.where(has_active & ~has_empty, active_mean, combined)
    combined = torch.where(~has_active & has_empty, empty_mean, combined)
    return combined


def infer_flow_batches(
    model: FlowAutoencoder,
    normalized_features: np.ndarray,
    segment_ids: np.ndarray,
    *,
    batch_size: int,
    mask_ratio: float,
    seed: int,
    device: torch.device,
    occupancy: np.ndarray | None = None,
    score_mask_rounds: int = 1,
    active_error_weight: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    architecture = getattr(model, "architecture", "v1")
    if occupancy is None:
        occupancy = np.ones(normalized_features.shape[:2], dtype=bool)
    if architecture == "v2":
        masks = complementary_masks(
            segment_ids, normalized_features.shape[1], score_mask_rounds, seed
        )
    else:
        masks = deterministic_masks(segment_ids, normalized_features.shape[1], mask_ratio, seed)[None, ...]
    embeddings: list[np.ndarray] = []
    errors: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(normalized_features), batch_size):
            stop = min(len(normalized_features), start + batch_size)
            values = torch.as_tensor(normalized_features[start:stop], dtype=torch.float32, device=device)
            embedding = model.encode(values)
            if architecture == "v2":
                per_bin_sum = torch.zeros(values.shape[:2], dtype=values.dtype, device=device)
                per_bin_count = torch.zeros_like(per_bin_sum)
                for round_masks in masks:
                    mask = torch.as_tensor(round_masks[start:stop], dtype=torch.bool, device=device)
                    reconstruction, _ = model(values, mask)
                    squared = (reconstruction - values).square().mean(dim=-1)
                    per_bin_sum += squared * mask
                    per_bin_count += mask
                per_bin_error = per_bin_sum / per_bin_count.clamp_min(1.0)
                occupied = torch.as_tensor(occupancy[start:stop], dtype=torch.bool, device=device)
                masked_error = _event_weighted_error(
                    per_bin_error, occupied, active_error_weight
                )
            else:
                mask = torch.as_tensor(masks[0, start:stop], dtype=torch.bool, device=device)
                reconstruction, _ = model(values, mask)
                squared = (reconstruction - values).square().mean(dim=-1)
                masked_error = (squared * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)
            embeddings.append(embedding.cpu().numpy())
            errors.append(masked_error.cpu().numpy())
    return np.concatenate(embeddings), np.concatenate(errors)
