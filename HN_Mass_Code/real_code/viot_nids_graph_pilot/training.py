"""Full normal traversal, fresh pretraining, then differentiable graph training."""
from __future__ import annotations
import csv
import logging
import time
from pathlib import Path
from importlib.metadata import version
import platform

import joblib
import numpy as np
import torch
from sklearn.cluster import KMeans

from viot_nids_v7.context import Scale, Behavior
from viot_nids_v7.network import Pretrain, masked_loss, mask_table, finite
from viot_nids_v7.training import setup, Balanced, epoch_batches, embed, save_torch, gradient_norm
from viot_nids_v7.data import write_json
from .data import load
from .graph import GraphIndex, clocks
from .network import Model, local_loss, shape_nll, time_nll

LOG = logging.getLogger(__name__)


def initialize(root, cfg, device):
    artifacts_file = root / "artifacts.joblib"
    if artifacts_file.exists():
        artifacts = joblib.load(artifacts_file)
        if artifacts["cfg"] != cfg:
            raise ValueError("Graph training configuration changed")
        return artifacts
    data, manifest = load(root / "normal_cache", normal=True)
    if manifest.get('slice_seconds', 1) != cfg.get('slice_seconds', 1):
        raise ValueError('Normal cache and training Slice lengths differ')
    scale = Scale.fit(data["features"], cfg["min_feature_scale"])
    x = scale.transform(data["features"])
    rng = np.random.default_rng(cfg["seed"])
    labels = KMeans(n_clusters=16, n_init=10, random_state=cfg["seed"]).fit_predict(x)
    sampler = Balanced(labels, data["timestamp_second"], rng)
    pretrained = Pretrain().to(device)
    optimizer = torch.optim.Adam(pretrained.parameters(), lr=cfg["pretrain_lr"])
    tensor = torch.as_tensor(x, device=device)
    with (root / "pretrain_log.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "local_loss", "gradient_norm", "seconds", "unique_rows", "steps"])
        writer.writeheader()
        for epoch in range(1, cfg["pretrain_epochs"] + 1):
            started, values = time.time(), []
            for ix in epoch_batches(len(x), cfg, sampler, rng):
                optimizer.zero_grad(set_to_none=True)
                masks = mask_table(device)[torch.as_tensor(rng.integers(3, size=len(ix)), device=device)]
                loss = masked_loss(pretrained, tensor[ix], masks)
                finite(loss, "pretrain loss")
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(pretrained.parameters(), cfg["gradient_clip"])
                finite(grad, "pretrain gradient")
                optimizer.step()
                values.append([float(loss.detach()), float(grad)])
            avg = np.mean(values, axis=0)
            writer.writerow(dict(epoch=epoch, local_loss=avg[0], gradient_norm=avg[1],
                                 seconds=time.time()-started, unique_rows=len(x), steps=len(values)))
            f.flush()
            LOG.info("Fresh pretrain %d/%d local=%.5f elapsed=%.1fs", epoch, cfg["pretrain_epochs"], avg[0], time.time()-started)
    pretrained.eval().requires_grad_(False)
    save_torch(root / "pretrain.pt", pretrained.state_dict())
    embeddings = embed(pretrained, x, cfg, device)
    behavior = Behavior.fit(embeddings, data["timestamp_second"], 16, cfg["seed"])
    assignments = behavior.transform(embeddings).argmax(1)
    clock = clocks(data, assignments)
    gap_scale = Scale.fit(clock["gap"][clock["gap_valid"], None], cfg["min_feature_scale"])
    artifacts = dict(cfg=cfg, input_scale=scale, behavior=behavior, gap_scale=gap_scale,
                     normal_manifest=manifest, assignments=assignments,
                     normal_flow_count=len(x), normal_gap_count=int(clock["gap_valid"].sum()))
    torch.manual_seed(cfg["seed"])
    prototype = Model(cfg, "HN").to(device)
    prototype.initialize(pretrained)
    save_torch(root / "initial_model.pt", prototype.state_dict())
    joblib.dump(artifacts, artifacts_file)
    write_json(root / "config.json", cfg)
    write_json(root / "environment.json", dict(python=platform.python_version(),
        **{p: version(p) for p in ("torch", "numpy", "pandas", "scikit-learn", "PyYAML", "joblib", "matplotlib")},
        device=str(device), gpu=torch.cuda.get_device_name() if device.type == "cuda" else None))
    write_json(root / "normal_preprocessing.json", dict(normal_flows=len(x), gap_samples=int(clock["gap_valid"].sum()),
        normal_new_connections=int(clock["new_connection"].sum()), behavior_support=behavior.support.tolist(),
        original_observation_alignment_verified=manifest["original_observation_alignment_verified"],
        frozen_measurement_only=True, main_encoder_frozen=False))
    return artifacts


class Arrays:
    def __init__(self, data, artifacts, cfg, device):
        self.data, self.cfg, self.device = data, cfg, device
        self.x = torch.as_tensor(artifacts["input_scale"].transform(data["features"]), device=device)
        self.assignments = artifacts["assignments"]
        self.clock = clocks(data, self.assignments)
        gaps = artifacts["gap_scale"].transform(self.clock["gap"][:, None]).ravel()
        self.gap = torch.as_tensor(gaps, device=device)
        self.valid = torch.as_tensor(self.clock["gap_valid"], device=device)
        age = np.minimum(64, data["timestamp_second"] - data["timestamp_second"].min()) / 64
        self.flags = torch.as_tensor(np.stack([self.clock["new_connection"], age], 1).astype(np.float32), device=device)
        self.index = GraphIndex(data, cfg, cache_candidates=True)
        _, counts = np.unique(data["timestamp_second"], return_counts=True)
        boundaries = np.quantile(counts, [.25, .5, .75])
        mapping = dict(zip(np.unique(data["timestamp_second"]), counts))
        self.quartiles = np.searchsorted(boundaries, [mapping[t] for t in data["timestamp_second"]], side="right")

    def bank(self, model, ix, rng, retain=False):
        raw = self.index.bank(ix, rng, include_current=True)
        used = np.unique(np.concatenate([raw[name][raw[name] >= 0] for name in ("private", "historical", "current")]))
        # This operation stays in autograd on every training batch.
        z = model.encoder(self.x[used])
        if retain and z.requires_grad:
            z.retain_grad()
        padded = torch.cat([z, z.new_zeros((1, 32))])
        bank = {"_neighbor_z": z}
        for name in ("private", "historical", "current"):
            ids = raw[name]
            position = np.searchsorted(used, np.maximum(ids, 0))
            position[ids < 0] = len(used)
            bank[name + "_z"] = padded[torch.as_tensor(position, device=self.device)]
            bank[name + "_valid"] = torch.as_tensor(ids >= 0, device=self.device)
            bank[name + "_type"] = torch.as_tensor(raw[name + "_type"], device=self.device)
            bank[name + "_dt"] = torch.as_tensor(raw[name + "_dt"], device=self.device)
        return bank

    def negatives(self, ix, rng):
        n = len(ix)
        if n < 2:
            return np.zeros(n, np.int64)
        result = np.empty(n, np.int64)
        for p in range(n):
            candidates = np.flatnonzero(self.data["timestamp_second"][ix] != self.data["timestamp_second"][ix[p]])
            if not len(candidates):
                raise ValueError("Relation resampling needs another normal window")
            if p % 2:
                q = self.quartiles[ix[p]]
                matching = candidates[self.quartiles[ix[candidates]] == q]
                if len(matching):
                    candidates = matching
            result[p] = rng.choice(candidates)
        return result


def losses(model, arrays, ix, rng, retain=False):
    bank = arrays.bank(model, ix, rng, retain)
    x, flags = arrays.x[ix], arrays.flags[ix]
    mask = mask_table(x.device)[torch.as_tensor(rng.integers(3, size=len(ix)), device=x.device)]
    shape, timing, diag, _ = model(x, mask, bank, flags)
    local = local_loss(model, x, mask)
    sl = shape_nll(x, mask, shape).mean()
    valid = arrays.valid[ix]
    tl = time_nll(arrays.gap[ix][valid], tuple(a[valid] for a in timing)).mean() if valid.any() else x.sum() * 0
    rl = model.relation_loss(x, mask, bank, flags, arrays.negatives(ix, rng))
    terms = [local, sl, tl, rl]
    for name, value in zip(("local", "shape", "timing", "relation"), terms):
        finite(value, name)
    return terms, diag, bank


def train(root: Path, cfg, variant):
    destination = root / variant
    if (destination / "model.pt").exists():
        raise FileExistsError(f"Model already trained: {destination}; choose a new output directory")
    destination.mkdir(parents=True, exist_ok=True)
    device = setup(cfg)
    artifacts = initialize(root, cfg, device)
    data, _ = load(root / "normal_cache", normal=True)
    arrays = Arrays(data, artifacts, cfg, device)
    model = Model(cfg, variant).to(device)
    model.load_state_dict(torch.load(root / "initial_model.pt", map_location=device, weights_only=True))
    encoder_initial = {k: v.detach().clone() for k, v in model.encoder.state_dict().items()}
    rng = np.random.default_rng(cfg["seed"] + 100)
    sampler = Balanced(arrays.assignments, data["timestamp_second"], rng)
    norms = []
    parameters = list(model.encoder.parameters())
    for _ in range(cfg["weight_batches"]):
        terms, _, _ = losses(model, arrays, sampler.draw(cfg["batch_size"]), rng)
        norms.append([gradient_norm(t, parameters) for t in terms])
    norms = np.asarray(norms)
    average = norms.mean(0)
    weights = np.r_[1., np.clip(average[0] * np.array([1, 1, .2]) / np.maximum(average[1:], 1e-8), .01, 10)]
    write_json(destination / "loss_weights.json", dict(weights=weights.tolist(), mean_encoder_gradient_norm=average.tolist(),
        batch_encoder_gradient_norm=norms.tolist(), target_ratio=[1, 1, 1, .2]))
    LOG.info("%s normal-only weights=%s gradient norms=%s", variant, weights, average)
    heads = [p for name, p in model.named_parameters() if not name.startswith("encoder.")]
    optimizer = torch.optim.Adam([dict(params=parameters, lr=cfg["encoder_lr"]), dict(params=heads, lr=cfg["head_lr"])])
    assert all(p.requires_grad for p in parameters)
    assert all(any(p is q for group in optimizer.param_groups for q in group["params"]) for p in parameters)
    rng = np.random.default_rng(cfg["seed"] + 100)
    sampler = Balanced(arrays.assignments, data["timestamp_second"], rng)
    fields = ["epoch", "local_loss", "shape_loss", "timing_loss", "relation_loss", "gradient_norm", "mean_sigma",
              "private_gate", "history_gate", "current_gate", "correction_sigma_fraction", "encoder_delta_l2",
              "neighbor_embedding_gradient_norm", "seconds", "unique_rows", "steps"]
    with (destination / "train_log.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, cfg["joint_epochs"] + 1):
            started, observations, seen = time.time(), [], set()
            model.train()
            neighbor_grad = 0.
            for step, ix in enumerate(epoch_batches(len(data["row_id"]), cfg, sampler, rng)):
                seen.update(map(int, ix))
                optimizer.zero_grad(set_to_none=True)
                terms, diag, bank = losses(model, arrays, ix, rng, retain=step == 0)
                total = sum(w * t for w, t in zip(weights, terms))
                finite(total, "joint loss")
                total.backward()
                if step == 0:
                    z = bank["_neighbor_z"]
                    if z.numel() and z.grad is not None:
                        neighbor_grad = float(z.grad.norm())
                        if not np.isfinite(neighbor_grad) or neighbor_grad == 0:
                            raise ValueError("Historical/neighbor embeddings did not receive a finite gradient")
                    else:
                        raise ValueError("Detached historical/neighbor embedding")
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
                finite(grad, "joint gradient")
                optimizer.step()
                observations.append([*[float(t.detach()) for t in terms], float(grad),
                    *[float(diag[k].detach().mean()) for k in ("mean_sigma", "private_gate", "history_gate", "current_gate", "correction_sigma_fraction")]])
            if len(seen) != len(data["row_id"]):
                raise ValueError("Epoch did not traverse every normal Flow")
            delta = float(torch.sqrt(sum((v - encoder_initial[k]).square().sum() for k, v in model.encoder.state_dict().items())))
            if delta == 0 or not np.isfinite(delta):
                raise ValueError("Main Flow encoder failed to update")
            avg = np.mean(observations, axis=0)
            record = dict(zip(fields[:11], [epoch, *avg]))
            record.update(encoder_delta_l2=delta, neighbor_embedding_gradient_norm=neighbor_grad,
                          seconds=time.time()-started, unique_rows=len(seen), steps=len(observations))
            writer.writerow(record)
            f.flush()
            LOG.info("%s epoch %d/%d local=%.4f shape=%.4f time=%.4f relation=%.4f encoder_delta=%.4f elapsed=%.1fs",
                     variant, epoch, cfg["joint_epochs"], *avg[:4], delta, time.time()-started)
    save_torch(destination / "model.pt", model.eval().state_dict())
    write_json(destination / "training_complete.json", dict(model=variant, normal_flows=len(data["row_id"]),
        epochs=cfg["joint_epochs"], checkpoint="last_complete_epoch", encoder_delta_l2=delta,
        end_to_end=True, weights=weights.tolist(), neighbor_gradient_verified=True))
    from .scoring import Scorer
    scorer = Scorer(root, cfg, variant, fit_scales=True)
    scorer.score_cache(root / "normal_cache", destination / "normal_scores.csv.gz", normal=True)
    write_json(destination / "complete.json", dict(model=variant, trained=True, normal_rescored=True,
        epochs=cfg["joint_epochs"], normal_flows=len(data["row_id"]), end_to_end=True))
