"""Frozen detector and JSON/state-dict bundle independent of old project paths."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from .activity import Activity, affine, mass_batch, mass_tensors
from .network import LongMassModel, Pretrain
from .scaling import Behavior, Scale


def scale_dict(scale):
    return dict(center=scale.center.tolist(), scale=scale.scale.tolist(), log=scale.log)


def read_scale(value):
    return Scale(np.asarray(value["center"], np.float64), np.asarray(value["scale"], np.float64), value["log"])


def read_measurement(folder):
    folder = Path(folder)
    metadata = json.loads((folder / "bundle.json").read_text())
    value = metadata["behavior"]
    behavior = Behavior(read_scale(value["scaler"]), np.asarray(value["centers"], np.float64),
                        np.asarray(value["widths"], np.float64), np.asarray(value["support"], np.int64))
    encoder = Pretrain()
    encoder.load_state_dict(torch.load(folder / "flow_encoder.pt", map_location="cpu", weights_only=True))
    encoder.eval().requires_grad_(False)
    return metadata, read_scale(metadata["flow_scale"]), behavior, encoder


class PrototypeDistance:
    """Keep the original CPU float64 exact-input cache used in reported scores."""
    def __init__(self, encoder, flow_scale, behavior, batch_size=4096):
        self.encoder = encoder.double().cpu().eval().requires_grad_(False)
        self.flow_scale, self.behavior, self.batch_size = flow_scale, behavior, batch_size
        self.cache = {}

    def score(self, features):
        values = self.flow_scale.transform(features)
        scores = np.empty(len(values), np.float64)
        for begin in range(0, len(values), self.batch_size):
            stop = min(len(values), begin + self.batch_size)
            unique, inverse = np.unique(values[begin:stop], axis=0, return_inverse=True)
            keys = [row.tobytes() for row in unique]
            missing = [i for i, key in enumerate(keys) if key not in self.cache]
            if missing:
                with torch.inference_mode():
                    embedding = self.encoder.encoder(torch.as_tensor(unique[missing], dtype=torch.float64)).numpy()
                z = (embedding - self.behavior.scaler.center) / self.behavior.scaler.scale
                squared = ((z[:, None] - self.behavior.centers[None]) ** 2).sum(-1)
                distance = (squared / self.behavior.widths[None] ** 2).min(1)
                if not np.isfinite(distance).all():
                    raise FloatingPointError("Nonfinite prototype distance")
                for i, d in zip(missing, distance):
                    self.cache[keys[i]] = float(d)
            scores[begin:stop] = np.asarray([self.cache[key] for key in keys])[inverse]
        return scores


class Detector:
    def __init__(self, folder, device=None):
        self.folder = Path(folder)
        self.metadata, self.flow_scale, self.behavior, self.encoder = read_measurement(folder)
        self.cfg = self.metadata["config"]
        torch.set_num_threads(self.cfg["torch_threads"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        value = self.metadata["mass_artifact"]
        self.artifact = {k: np.asarray(value[k], np.float32) for k in ("lower", "upper", "mean_logits", "scale_logits")}
        self.artifact["input_scale"] = read_scale(value["input_scale"])
        self.artifact["target_scales"] = [read_scale(v) for v in value["target_scales"]]
        self.model = LongMassModel(self.cfg, "BASE_GRU", self.artifact).to(self.device)
        self.model.load_state_dict(torch.load(self.folder / "mass_model.pt", map_location=self.device, weights_only=True))
        self.model.eval().requires_grad_(False)

    def score_capture(self, data, align_scores=True):
        # This call starts independent capture history. Anonymous indices are
        # used to gather FPB, never passed as numerical inputs to a network.
        self.encoder.float().to(self.device)
        x = self.flow_scale.transform(data["features"])
        roles = np.empty(len(x), np.int64)
        with torch.inference_mode():
            for begin in range(0, len(x), self.cfg["score_batch_size"]):
                stop = min(len(x), begin + self.cfg["score_batch_size"])
                embedding = self.encoder.encoder(torch.as_tensor(x[begin:stop], device=self.device)).cpu().numpy()
                roles[begin:stop] = self.behavior.transform(embedding).argmax(1)
        a = Activity(data, roles, history=self.cfg["history_seconds"], behavior_count=self.cfg["behavior_k"])
        arrays = mass_tensors(a, self.artifact, "fpb", self.device)
        group_score = np.empty(len(a.active), np.float64)
        with torch.inference_mode():
            for begin in range(0, len(a.active), self.cfg["score_batch_size"]):
                ix = np.arange(begin, min(len(a.active), begin + self.cfg["score_batch_size"]))
                dist, _ = self.model(*mass_batch(a, arrays, ix, self.device))
                group_score[ix] = self.model.nll(arrays["target"][ix], dist).cpu().numpy()
        mass = group_score[a.source_pair]
        # Use a fresh state-dict copy: do not round-trip a double model back to
        # float32 for subsequent captures or change behavior-assignment weights.
        distance_encoder = Pretrain()
        distance_encoder.load_state_dict(torch.load(self.folder / "flow_encoder.pt", map_location="cpu", weights_only=True))
        distance = PrototypeDistance(distance_encoder, self.flow_scale, self.behavior, self.cfg["distance_batch_size"]).score(data["features"])
        result = dict(row_id=data["row_id"], timestamp_second=data["timestamp_second"], behavior=roles,
            prototype_raw=distance, mass_raw=mass, history_seconds_available=a.lengths[a.source_pair],
            group_flow_count=a.totals.reshape(-1, 3)[a.source_cell, 0],
            group_packet_count=a.totals.reshape(-1, 3)[a.source_cell, 1],
            group_byte_count=a.totals.reshape(-1, 3)[a.source_cell, 2])
        if align_scores:
            ref = self.metadata["score_reference"]
            result["prototype_scaled"] = affine(distance, ref["prototype"])
            result["mass_scaled"] = affine(mass, ref["mass"])
            result["final_score"] = result["prototype_scaled"] + result["mass_scaled"]
            result["above_training_q99"] = result["final_score"] > ref["fusion_q99"]
        for name, value in result.items():
            if not np.isfinite(value).all():
                raise FloatingPointError(f"Nonfinite score field {name}")
        return result
