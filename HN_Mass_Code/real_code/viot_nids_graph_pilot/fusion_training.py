"""Fixed-budget normal-only fusion training; no attack imports or model selection."""
from __future__ import annotations
import csv
import json
import logging
import time
from pathlib import Path
import joblib
import numpy as np
import torch
from torch.nn import functional as F
from viot_nids_v7.data import write_json
from viot_nids_v7.network import mask_table, finite
from viot_nids_v7.training import setup, Balanced, epoch_batches, gradient_norm, save_torch
from .data import load
from .training import Arrays, losses
from .network import Model
from .fusion import FusionModel, composite

LOG = logging.getLogger(__name__)
FROZEN_VARIANTS = ('fixed', 'gated', 'gated-contrast')
JOINT_VARIANTS = ('continued-base', 'gated', 'gated-contrast')


def run_id(backbone, stage, variant):
    if backbone not in ('H', 'HN') or stage not in ('frozen', 'joint'):
        raise ValueError('Unknown fusion backbone/stage')
    allowed = FROZEN_VARIANTS if stage == 'frozen' else JOINT_VARIANTS
    if variant not in allowed:
        raise ValueError('Variant is not available in this stage')
    return f'{backbone}/{stage}/{variant}'


class Donors:
    def __init__(self, arrays):
        self.arrays = arrays
        d = arrays.data
        self.blocks = (d['timestamp_second'] - d['timestamp_second'].min()) // 10
        self.x = arrays.x.detach().cpu().numpy()
        self.has_private = np.array([sum(map(len, arrays.index.private(i))) > 0 for i in range(len(self.x))])
        width = arrays.cfg['same_connection_neighbors'] + arrays.cfg['reverse_connection_neighbors']
        self.private_rows = np.full((len(self.x), width), -1, np.int64)
        self.private_types = np.zeros_like(self.private_rows)
        start = arrays.cfg['same_connection_neighbors']
        for i in range(len(self.x)):
            own, reverse = arrays.index.private(i)
            self.private_rows[i, :len(own)] = own
            self.private_types[i, :len(own)] = 4
            self.private_rows[i, start:start + len(reverse)] = reverse
            self.private_types[i, start:start + len(reverse)] = 5
        valid = self.private_rows >= 0
        times = d['first_packet_time'][np.maximum(self.private_rows, 0)]
        if arrays.cfg.get('private_time_policy') == 'past_relative_v1':
            origin = np.max(np.where(valid, times, -np.inf), axis=1)
            origin[~valid.any(1)] = 0
            dt = np.maximum(0., origin[:, None] - times)
        else:
            dt = np.abs(d['first_packet_time'][:, None] - times)
        self.private_dt = (np.log1p(dt) / np.log(65)).astype(np.float32)
        self.private_dt[~valid] = 0
        check_rows = np.linspace(0, len(self.x) - 1, min(24, len(self.x))).astype(int)
        expected = arrays.index.bank(check_rows, np.random.default_rng(42), include_current=False)
        for name, values in [('private', self.private_rows), ('private_type', self.private_types), ('private_dt', self.private_dt)]:
            np.testing.assert_array_equal(values[check_rows], expected[name], err_msg='Private donor cache changed graph semantics')
        self.pools = {bool(flag): np.flatnonzero(self.has_private & (arrays.clock['new_connection'] == flag))
                      for flag in (False, True)}

    def bank(self, model, rows):
        # Cache indices only. Every donor node is encoded with current weights
        # and keeps its gradient in the joint stage.
        ids = self.private_rows[rows]
        used = np.unique(ids[ids >= 0])
        z = model.encoder(self.arrays.x[used])
        padded = torch.cat([z, z.new_zeros((1, 32))])
        pos = np.searchsorted(used, np.maximum(ids, 0))
        pos[ids < 0] = len(used)
        device = self.arrays.device
        return dict(private_z=padded[torch.as_tensor(pos, device=device)],
            private_valid=torch.as_tensor(ids >= 0, device=device),
            private_type=torch.as_tensor(self.private_types[rows], device=device),
            private_dt=torch.as_tensor(self.private_dt[rows], device=device))

    def draw(self, ix, mask_index, rng):
        donors = np.full((len(ix), 2), -1, np.int64)
        visible = 1 - mask_table('cpu')[mask_index].numpy()
        for pos, row in enumerate(ix):
            if not self.has_private[row]:
                continue
            pool = self.pools[bool(self.arrays.clock['new_connection'][row])]
            candidates = pool[self.blocks[pool] != self.blocks[row]]
            if not len(candidates):
                continue
            random = int(rng.choice(candidates))
            donors[pos, 0] = random
            remaining = candidates[candidates != random]
            if len(remaining):
                choices = rng.choice(remaining, min(32, len(remaining)), replace=False)
                distances = ((self.x[choices] - self.x[row]) ** 2 * visible).sum(1)
                donors[pos, 1] = choices[np.argmin(distances)]
        return donors


def pair_loss(model, arrays, ix, mask, u, bank, donors, donor_sampler):
    x, flags = arrays.x[ix], arrays.flags[ix]
    z = model.encoder(x)
    valid_target = (donors >= 0).any(1)
    if not valid_target.any():
        return u.sum() * 0
    positive = model.critic(torch.cat([u, z, mask], -1)).squeeze(-1)
    neg_sum = x.new_zeros(len(x))
    neg_count = x.new_zeros(len(x))
    for column in range(2):
        ok = donors[:, column] >= 0
        if not ok.any():
            continue
        db = donor_sampler.bank(model, donors[ok, column])
        # Only replace the private history; donor relative times remain intact.
        neg_bank = {k: v[ok] for k, v in bank.items() if not k.startswith('_')}
        for key in list(neg_bank):
            if key.startswith('private_'):
                neg_bank[key] = db[key]
        nu = model.represent(x[ok], mask[ok], neg_bank, flags[ok])[0]
        negative = model.critic(torch.cat([nu, z[ok], mask[ok]], -1)).squeeze(-1)
        neg_sum[ok] += F.softplus(negative)
        neg_count[ok] += 1
    selected = torch.as_tensor(valid_target, device=x.device)
    return .5 * (F.softplus(-positive[selected]) + neg_sum[selected] / neg_count[selected]).mean()


def fusion_losses(model, arrays, ix, rng, donor_sampler, retain=False):
    bank = arrays.bank(model, ix, rng, retain)
    x, flags = arrays.x[ix], arrays.flags[ix]
    fusion, relation, diags = [], [], []
    for group, m in enumerate(mask_table(x.device)):
        mask = m.expand_as(x)
        shape, timing, diag, u = model(x, mask, bank, flags)
        fusion.append(composite(x, mask, arrays.gap[ix], arrays.valid[ix], shape, timing)[0].mean())
        diags.append(diag)
        if model.variant == 'gated-contrast':
            donors = donor_sampler.draw(ix, group, rng)
            relation.append(pair_loss(model, arrays, ix, mask, u, bank, donors, donor_sampler))
    fl = torch.stack(fusion).mean()
    rl = torch.stack(relation).mean() if relation else fl * 0
    finite(fl, 'fusion likelihood')
    finite(rl, 'fusion source contrast')
    diag = {key: torch.stack([d[key].mean() for d in diags]).mean() for key in diags[0]}
    return fl, rl, diag, bank


def load_config(path):
    import yaml
    cfg = yaml.safe_load(Path(path).read_text())
    root = Path(cfg['output_root'])
    base_root = Path(cfg['base_root'])
    artifact = joblib.load(base_root / 'artifacts.joblib')
    merged = {**artifact['cfg'], **cfg}
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'config.json'
    if path.exists() and json.loads(path.read_text()) != merged:
        raise ValueError('Fusion configuration is already fixed; choose a new output root')
    write_json(path, merged)
    return root, base_root, merged, artifact


def train_fusion(config, backbone, stage, variant):
    root, base_root, cfg, artifacts = load_config(config)
    rid = run_id(backbone, stage, variant)
    destination = root / rid
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / 'training_complete.json').exists():
        LOG.info('Training already complete %s', rid)
        return rid
    resume = (destination / 'recovery.pt').exists()
    if (destination / 'train_log.csv').exists() and not resume:
        raise FileExistsError(f'Incomplete training found: {destination}; resume explicitly from its recovery checkpoint')
    device = setup(cfg)
    data, _ = load(base_root / 'normal_cache', normal=True)
    arrays = Arrays(data, artifacts, cfg, device)
    old = torch.load(base_root / backbone / 'model.pt', map_location=device, weights_only=True)
    if variant == 'continued-base':
        model = Model(cfg, backbone).to(device)
        model.load_state_dict(old)
    else:
        model = FusionModel(cfg, backbone, variant).to(device)
        model.initialize(old)
        if stage == 'joint':
            source = root / run_id(backbone, 'frozen', variant) / 'model.pt'
            model.load_state_dict(torch.load(source, map_location=device, weights_only=True))
        model.base.requires_grad_(stage == 'joint')
        if variant == 'fixed':
            model.gate.requires_grad_(False)
        if variant != 'gated-contrast':
            model.critic.requires_grad_(False)
    base = model if variant == 'continued-base' else model.base
    encoder_initial = {k: v.detach().clone() for k, v in base.encoder.state_dict().items()}
    base_initial = {k: v.detach().clone() for k, v in base.state_dict().items()} if stage == 'frozen' else None
    base_weights = json.loads((base_root / backbone / 'loss_weights.json').read_text())['weights']
    donors = Donors(arrays) if variant == 'gated-contrast' else None
    rng = np.random.default_rng(cfg['seed'] + 310)
    sampler = Balanced(arrays.assignments, data['timestamp_second'], rng)
    norms = []
    if resume:
        saved_weights = json.loads((destination / 'loss_weights.json').read_text())
        lf, lp = saved_weights['fusion_weight'], saved_weights['pair_weight']
        norms = saved_weights['normal_gradient_norms']
    elif variant != 'continued-base':
        parameters = list(model.delta.parameters()) if stage == 'frozen' else list(base.encoder.parameters())
        for _ in range(cfg['weight_batches']):
            ix = sampler.draw(cfg['batch_size'])
            fl, pl, _, _ = fusion_losses(model, arrays, ix, rng, donors)
            reference = fl if stage == 'frozen' else losses(base, arrays, ix, rng)[0][0]
            norms.append([gradient_norm(term, parameters) if term.requires_grad else 0 for term in (reference, fl, pl)])
        average = np.mean(norms, 0)
        lf = 1. if stage == 'frozen' else float(np.clip(average[0] / max(average[1], 1e-8), .01, 10))
        # Joint Gated and Gated+Contrast share the Gated fusion weight.
        if stage == 'joint' and variant == 'gated-contrast':
            shared = root / run_id(backbone, stage, 'gated') / 'loss_weights.json'
            lf = json.loads(shared.read_text())['fusion_weight']
        lp = float(np.clip(.2 * lf * average[1] / max(average[2], 1e-8), .01, 10)) if donors else 0.
    else:
        lf = lp = 0.
    write_json(destination / 'loss_weights.json', dict(base_weights=base_weights, fusion_weight=lf,
        pair_weight=lp, normal_gradient_norms=norms, computed_from_normal_only=True,
        calibration_rule_fixed_before_first_attack=True, fixed_before_this_run_attack_evaluation=True))
    encoder_params = [p for p in base.encoder.parameters() if p.requires_grad]
    encoder_ids = {id(p) for p in encoder_params}
    heads = [p for p in model.parameters() if p.requires_grad and id(p) not in encoder_ids]
    groups = [dict(params=heads, lr=cfg['head_lr'])]
    if encoder_params:
        groups.append(dict(params=encoder_params, lr=cfg['encoder_lr']))
    optimizer = torch.optim.Adam(groups)
    rng = np.random.default_rng(cfg['seed'] + 100)
    fusion_rng = np.random.default_rng(cfg['seed'] + 310)
    sampler = Balanced(arrays.assignments, data['timestamp_second'], rng)
    epochs = cfg['fusion_epochs'] if stage == 'frozen' else cfg['joint_epochs']
    fields = ['epoch', 'total_loss', 'local_loss', 'shape_loss', 'timing_loss', 'relation_loss',
              'fusion_loss', 'pair_loss', 'gradient_norm', 'fusion_gate_mean', 'fusion_delta_norm',
              'fusion_mean_sigma', 'fusion_floor_fraction', 'encoder_delta_l2',
              'neighbor_gradient_norm', 'unique_rows', 'steps', 'seconds']
    start_epoch = 1
    if resume:
        recovery = torch.load(destination / 'recovery.pt', map_location=device, weights_only=True)
        model.load_state_dict(recovery['model'])
        optimizer.load_state_dict(recovery['optimizer'])
        rng.bit_generator.state = recovery['target_rng']
        fusion_rng.bit_generator.state = recovery['fusion_rng']
        start_epoch = recovery['epoch'] + 1
        with (destination / 'train_log.csv').open(newline='') as f:
            previous_records = [row for row in csv.DictReader(f) if int(row['epoch']) < start_epoch]
        LOG.info('Resume %s at epoch %d; budgets and weights unchanged', rid, start_epoch)
    else:
        previous_records = []
    with (destination / 'train_log.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(previous_records)
        for epoch in range(start_epoch, epochs + 1):
            started, observations, seen = time.time(), [], set()
            model.train()
            neighbor_grad = 0.
            for step, ix in enumerate(epoch_batches(len(data['row_id']), cfg, sampler, rng)):
                seen.update(map(int, ix))
                optimizer.zero_grad(set_to_none=True)
                terms = [arrays.x.sum() * 0] * 4
                if stage == 'joint':
                    terms, _, base_bank = losses(base, arrays, ix, rng, retain=step == 0)
                fl = pl = arrays.x.sum() * 0
                diag = {}
                if variant != 'continued-base':
                    fl, pl, diag, fusion_bank = fusion_losses(model, arrays, ix, fusion_rng, donors, retain=step == 0)
                total = sum(w * t for w, t in zip(base_weights, terms)) + lf * fl + lp * pl
                finite(total, 'normal total fusion loss')
                total.backward()
                if step == 0 and stage == 'joint':
                    z = base_bank['_neighbor_z']
                    neighbor_grad = 0. if z.grad is None else float(z.grad.norm())
                    if neighbor_grad <= 0 or not np.isfinite(neighbor_grad):
                        raise ValueError('End-to-end neighbor gradient missing')
                grad = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg['gradient_clip'])
                finite(grad, 'fusion gradient')
                optimizer.step()
                observations.append([float(total.detach()), *[float(t.detach()) for t in terms],
                    float(fl.detach()), float(pl.detach()), float(grad),
                    *[float(diag.get(k, arrays.x.new_zeros(())).detach()) for k in
                      ('fusion_gate_mean', 'fusion_delta_norm', 'fusion_mean_sigma', 'fusion_floor_fraction')]])
            if len(seen) != len(data['row_id']):
                raise ValueError('Normal epoch coverage incomplete')
            delta = float(torch.sqrt(sum((v - encoder_initial[k]).square().sum() for k, v in base.encoder.state_dict().items())))
            if stage == 'frozen':
                for key, value in base.state_dict().items():
                    if not torch.equal(value, base_initial[key]):
                        raise ValueError(f'Frozen base changed: {key}')
            elif delta == 0 or not np.isfinite(delta):
                raise ValueError('Main encoder did not update')
            avg = np.mean(observations, 0)
            record = dict(zip(fields[:13], [epoch, *avg]))
            record.update(encoder_delta_l2=delta, neighbor_gradient_norm=neighbor_grad,
                          unique_rows=len(seen), steps=len(observations), seconds=time.time() - started)
            writer.writerow(record)
            f.flush()
            save_torch(destination / 'recovery.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                epoch=epoch, target_rng=rng.bit_generator.state, fusion_rng=fusion_rng.bit_generator.state))
            LOG.info('%s epoch %d/%d fusion=%.4f pair=%.4f sigma=%.4f seconds=%.1f',
                     rid, epoch, epochs, avg[5], avg[6], avg[10], record['seconds'])
    delta = float(torch.sqrt(sum((v - encoder_initial[k]).square().sum() for k, v in base.encoder.state_dict().items())))
    save_torch(destination / 'model.pt', model.eval().state_dict())
    write_json(destination / 'training_complete.json', dict(run_id=rid, backbone=backbone, stage=stage,
        variant=variant, epochs=epochs, normal_flows=len(data['row_id']), last_complete_epoch=True,
        base_frozen_verified=stage == 'frozen', encoder_delta_l2=delta,
        neighbor_gradient_verified=stage == 'joint', attack_labels_read=False))
    from .fusion_scoring import FusionScorer
    scorer = FusionScorer(config, rid, fit_scales=True)
    scorer.score_cache(base_root / 'normal_cache', destination / 'normal_scores.csv.gz', normal=True)
    return rid
