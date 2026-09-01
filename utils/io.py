"""Safe artifact I/O and alignment checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def write_json(path: str | Path, value: Any) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def save_torch_checkpoint(path: str | Path, value: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, output)
    return output


def load_torch_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    try:
        value = torch.load(Path(path), map_location=device, weights_only=True)
    except TypeError:
        value = torch.load(Path(path), map_location=device)
    if not isinstance(value, dict):
        raise ValueError(f"Invalid checkpoint in {path}")
    return value


def require_alignment(expected_ids: np.ndarray, artifact_ids: np.ndarray, name: str) -> None:
    if expected_ids.shape != artifact_ids.shape or not np.array_equal(expected_ids, artifact_ids):
        raise ValueError(f"{name} is not aligned with the processed dataset; regenerate it")
