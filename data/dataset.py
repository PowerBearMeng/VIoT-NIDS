"""Persistent flow-segment dataset and strict normal-only split accessors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


SPLIT_TO_ID = {"train": 0, "calibration": 1, "validation": 1, "val": 1, "test": 2}


class FlowDataset:
    REQUIRED = {
        "features",
        "labels",
        "split",
        "segment_ids",
        "flow_ids",
        "captures",
        "capture_ids",
        "endpoint_a_ips",
        "endpoint_b_ips",
        "endpoint_a_ports",
        "endpoint_b_ports",
        "protocols",
        "window_indices",
        "segment_starts",
        "packet_counts",
        "byte_counts",
    }

    def __init__(self, npz_path: str | Path, metadata_path: str | Path) -> None:
        with np.load(Path(npz_path), allow_pickle=False) as archive:
            missing = self.REQUIRED - set(archive.files)
            if missing:
                raise ValueError(f"Processed dataset is missing arrays: {sorted(missing)}")
            for name in archive.files:
                setattr(self, name, archive[name])
        if not hasattr(self, "dataset_names"):
            self.dataset_names = np.asarray(
                [Path(str(value)).parent.name for value in self.captures]
            )
        self.metadata: dict[str, Any] = json.loads(
            Path(metadata_path).read_text(encoding="utf-8")
        )
        lengths = {len(getattr(self, name)) for name in self.REQUIRED}
        if len(lengths) != 1:
            raise ValueError("Processed dataset arrays have inconsistent lengths")
        expected = (int(self.metadata["num_bins"]), 6)
        if self.features.ndim != 3 or tuple(self.features.shape[1:]) != expected:
            raise ValueError(f"features must have shape [N,{expected[0]},6]")

    def __len__(self) -> int:
        return int(self.features.shape[0])

    @property
    def normal_label_ids(self) -> set[int]:
        return set(int(value) for value in self.metadata["normal_label_ids"])

    def indices(
        self,
        split: str | None = None,
        *,
        normal_only: bool = False,
        labeled_only: bool = False,
    ) -> np.ndarray:
        mask = np.ones(len(self), dtype=bool)
        if split and split != "all":
            if split not in SPLIT_TO_ID:
                raise ValueError(f"Unknown split: {split}")
            mask &= self.split == SPLIT_TO_ID[split]
        if normal_only:
            mask &= np.isin(self.labels, list(self.normal_label_ids))
        if labeled_only:
            mask &= self.labels >= 0
        return np.flatnonzero(mask)

    def require_normal(self, split: str, purpose: str) -> np.ndarray:
        indices = self.indices(split, normal_only=True)
        if not len(indices):
            raise ValueError(f"No normal samples in {split!r} split for {purpose}")
        rejected = self.indices(split)
        rejected = rejected[~np.isin(self.labels[rejected], list(self.normal_label_ids))]
        if len(rejected):
            # This is informational: the returned indices remain strictly normal.
            print(f"normal-only guard: excluded {len(rejected)} non-normal {split} segments from {purpose}")
        return indices
