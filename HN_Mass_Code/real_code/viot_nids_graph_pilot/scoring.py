"""Frozen inference; only here may encoded nodes be cached."""
from __future__ import annotations
import copy
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from viot_nids_v7.network import Pretrain, mask_table, affine
from viot_nids_v7.training import setup, embed
from viot_nids_v7.data import write_json
from .data import load
from .graph import GraphIndex, clocks, fixed_bank, subset_bank
from .network import Model, local_score, shape_nll, time_nll

LOG = logging.getLogger(__name__)
SCORING_POLICY = "event_seed_v1"


class Scorer:
    def __init__(self, root, cfg, variant, fit_scales=False):
        self.root, self.cfg, self.variant = Path(root), cfg, variant
        self.device = setup(cfg)
        self.artifacts = joblib.load(self.root / "artifacts.joblib")
        self.model = Model(cfg, variant).to(self.device)
        self.model.load_state_dict(torch.load(self.root / variant / "model.pt", map_location=self.device, weights_only=True))
        self.model.eval().requires_grad_(False)
        self.fixed = Pretrain().to(self.device)
        self.fixed.load_state_dict(torch.load(self.root / "pretrain.pt", map_location=self.device, weights_only=True))
        self.fixed.eval().requires_grad_(False)
        self.scales = None if fit_scales else json.loads((self.root / variant / "score_scales.json").read_text())
        self.singleton_model = copy.deepcopy(self.model).cpu().double().eval()
        self.singleton_values = {}

    def singleton(self, length):
        length = float(length)
        if length not in self.singleton_values:
            features = np.array([[1, length, 0, length, 0, length, length, length, length, length,
                                  *([0.] * 8), 1, length / 10, length * .3, length, .1, .3, 1]])
            x = torch.as_tensor(self.artifacts["input_scale"].transform(features), dtype=torch.float64)
            with torch.inference_mode():
                self.singleton_values[length] = float(local_score(self.singleton_model, x).item())
        return self.singleton_values[length]

    def prepare_data(self, data, bank=None):
        values = self.artifacts["input_scale"].transform(data["features"])
        fixed = embed(self.fixed, values, self.cfg, self.device)
        assignments = np.empty(len(values), np.int64)
        for start in range(0, len(values), 4096):
            end = min(len(values), start + 4096)
            assignments[start:end] = self.artifacts['behavior'].transform(fixed[start:end]).argmax(1)
        del fixed
        clock = clocks(data, assignments)
        if bank is None:
            index = GraphIndex(data, self.cfg)
            bank = fixed_bank(index, len(values), self.cfg)
        with torch.inference_mode():
            z = torch.as_tensor(embed(self.model, values, self.cfg, self.device), device=self.device)
        encoded = torch.cat([z, z.new_zeros((1, 32))])
        x = torch.as_tensor(values, device=self.device)
        return dict(data=data, x=x, encoded=encoded, bank=bank, assignments=assignments, clock=clock)

    def neural_bank(self, prepared, bank):
        n = len(prepared["data"]["row_id"])
        result = {}
        for name in ("private", "historical", "current"):
            ids = bank[name]
            safe = ids.copy()
            safe[ids < 0] = n
            result[name + "_z"] = prepared["encoded"][torch.as_tensor(safe, device=self.device)]
            result[name + "_valid"] = torch.as_tensor(ids >= 0, device=self.device)
            result[name + "_type"] = torch.as_tensor(bank[name + "_type"], device=self.device, dtype=torch.long)
            result[name + "_dt"] = torch.as_tensor(bank[name + "_dt"], device=self.device)
        return result

    def run(self, prepared, rows=None, bank=None, clock=None, details=False):
        data = prepared["data"]
        all_rows = rows is None
        rows = np.arange(len(data["row_id"])) if all_rows else np.asarray(rows, np.int64)
        bank = (prepared["bank"] if all_rows else subset_bank(prepared["bank"], rows)) if bank is None else bank
        clock = prepared["clock"] if clock is None else clock
        parts, detailed = [], []
        size = self.cfg["score_batch_size"]
        for start in range(0, len(rows), size):
            ids = rows[start:start + size]
            rb = subset_bank(bank, np.arange(start, min(len(rows), start + size)))
            nb = self.neural_bank(prepared, rb)
            x = prepared["x"][ids]
            age = np.minimum(64, data["timestamp_second"][ids] - data["timestamp_second"].min()) / 64
            flags = torch.as_tensor(np.stack([clock["new_connection"][ids], age], 1).astype(np.float32), device=self.device)
            gap = torch.as_tensor(self.artifacts["gap_scale"].transform(clock["gap"][ids, None]).ravel(), device=self.device)
            valid = clock["gap_valid"][ids]
            with torch.inference_mode():
                local = local_score(self.model, x).cpu().numpy().astype(np.float64)
                shape_values, timing_values, diagnostics = [], [], []
                for m in mask_table(self.device):
                    mask = m.expand_as(x)
                    dist, time_dist, diag, _ = self.model(x, mask, nb, flags)
                    shape_values.append(shape_nll(x, mask, dist).cpu().numpy().astype(np.float64))
                    timing_values.append(time_nll(gap, time_dist).cpu().numpy().astype(np.float64))
                    diagnostics.append({k: v.cpu().numpy().astype(np.float64) for k, v in diag.items()})
                    if details:
                        for p, row in enumerate(ids):
                            detailed.append(dict(row_id=int(row), mask_group=len(shape_values)-1,
                                masked_coordinates=np.flatnonzero(m.cpu().numpy()).tolist(),
                                means=dist[1][p].cpu().numpy().tolist(), scales=dist[2][p].cpu().numpy().tolist(),
                                weights=dist[0][p].exp().cpu().numpy().tolist(),
                                time_means=time_dist[1][p].cpu().numpy().tolist(),
                                time_scales=time_dist[2][p].cpu().numpy().tolist(),
                                private_rows=rb["private"][p][rb["private"][p]>=0].tolist(),
                                historical_rows=rb["historical"][p][rb["historical"][p]>=0].tolist(),
                                current_rows=rb["current"][p][rb["current"][p]>=0].tolist()))
            shape = np.mean(shape_values, axis=0)
            timing = np.mean(timing_values, axis=0)
            context = (shape + np.where(valid, timing, 0)) / (1 + valid.astype(float))
            packet_count = data["features"][ids, 0]
            byte_count = data["features"][ids, 1]
            single = packet_count == 1
            for length in np.unique(byte_count[single]):
                local[single & (byte_count == length)] = self.singleton(length)
            frame = pd.DataFrame(dict(row_id=data["row_id"][ids], timestamp_second=data["timestamp_second"][ids],
                local_raw=local, shape_raw=shape, timing_raw=np.where(valid, timing, np.nan),
                context_raw=context, history_seconds_available=np.minimum(64, data["timestamp_second"][ids]-data["timestamp_second"].min()),
                private_records=(rb["private"]>=0).sum(1), historical_records=(rb["historical"]>=0).sum(1),
                current_records=(rb["current"]>=0).sum(1) if self.variant == "HN" else 0,
                historical_relation_incidence_count=rb["historical_counts"].sum(1),
                current_relation_incidence_count=rb["current_counts"].sum(1),
                packet_count=packet_count, byte_count=byte_count, new_connection=clock["new_connection"][ids],
                gap_available=valid, gap_seconds=np.where(valid, clock["gap"][ids], np.nan),
                gap_predecessor_row_id=clock['gap_predecessor'][ids],
                behavior_anchor=prepared["assignments"][ids],
                behavior_support=self.artifacts["behavior"].support[prepared["assignments"][ids]]))
            for name in diagnostics[0]:
                frame[name] = np.mean([d[name] for d in diagnostics], 0)
            parts.append(frame)
        result = pd.concat(parts, ignore_index=True)
        if not np.isfinite(result[["local_raw", "shape_raw", "context_raw"]]).all().all():
            raise FloatingPointError("Nonfinite continuous scores")
        if self.scales is not None:
            self.apply_scales(result)
        return (result, detailed) if details else result

    def apply_scales(self, frame):
        for name in ("local", "context"):
            frame[name + "_scaled"] = affine(frame[name + "_raw"], self.scales[name])
        frame["final_score"] = frame.local_scaled + frame.context_scaled
        if not np.isfinite(frame.final_score).all():
            raise FloatingPointError("Nonfinite fusion")

    def score_cache(self, cache, out, normal=False):
        data, _ = load(Path(cache), normal=normal)
        LOG.info("%s scoring %d Flows from %s", self.variant, len(data["row_id"]), cache)
        prepared = self.prepare_data(data)
        frame = self.run(prepared)
        if self.scales is None:
            self.scales = {name: dict(zip(("q50", "q95"), map(float, frame[name + "_raw"].quantile([.5, .95]))))
                           for name in ("local", "context")}
            self.apply_scales(frame)
            self.scales["reference_q99"] = float(frame.final_score.quantile(.99))
            self.scales["reference_is_training_in_sample"] = True
            self.scales["scoring_policy"] = SCORING_POLICY
            write_json(self.root / self.variant / "score_scales.json", self.scales)
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=False)
        if normal:
            from .checks import check
            check(self, prepared)
        LOG.info("%s scores saved %s", self.variant, out)
        return frame
