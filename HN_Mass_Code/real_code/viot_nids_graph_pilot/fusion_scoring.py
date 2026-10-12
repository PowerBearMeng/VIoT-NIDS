"""Frozen inference and actual-normal diagnostics for representation fusion."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from viot_nids_v7.data import write_json
from viot_nids_v7.network import affine, mask_table
from .scoring import Scorer
from .fusion import FusionModel, composite, density_diagnostics
from .fusion_training import load_config
from .network import shape_nll, time_nll
from torch.nn import functional as F


class FusionScorer(Scorer):
    def __init__(self, config, run, fit_scales=False):
        root, base_root, cfg, _ = load_config(config)
        backbone, stage, variant = run.split('/')
        super().__init__(base_root, cfg, backbone, fit_scales=True)
        self.root, self.variant, self.run_id = root, run, run
        self.backbone, self.stage, self.fusion_variant = backbone, stage, variant
        self.fusion_model = None
        if stage != 'original':
            state = torch.load(root / run / 'model.pt', map_location=self.device, weights_only=True)
            if variant == 'continued-base':
                self.model.load_state_dict(state)
            else:
                self.fusion_model = FusionModel(cfg, backbone, variant).to(self.device)
                self.fusion_model.load_state_dict(state)
                self.fusion_model.eval().requires_grad_(False)
                self.model = self.fusion_model.base
        self.model.eval().requires_grad_(False)
        self.singleton_model = copy.deepcopy(self.model).cpu().double().eval()
        self.singleton_values = {}
        self.legacy_scales = None
        if not fit_scales:
            self.legacy_scales = json.loads((root / run / 'score_scales.json').read_text())

    def run(self, prepared, rows=None, bank=None, clock=None, details=False):
        if self.fusion_model is None:
            raw = super().run(prepared, rows=rows, bank=bank, clock=clock, details=details)
            frame, predictions = raw if details else (raw, [])
            if self.legacy_scales is not None:
                self.apply_scales(frame)
            return (frame, predictions) if details else frame
        data = prepared['data']
        predictions, frames = [], []
        all_rows = rows is None
        rows = np.arange(len(data['row_id'])) if all_rows else np.asarray(rows, np.int64)
        from .graph import subset_bank
        raw_bank = (prepared['bank'] if all_rows else subset_bank(prepared['bank'], rows)) if bank is None else bank
        clock = prepared['clock'] if clock is None else clock
        parts, pits, sigmas, time_pits, time_sigmas = [], [], [], [], []
        for begin in range(0, len(rows), self.cfg['score_batch_size']):
            ids = rows[begin:begin + self.cfg['score_batch_size']]
            rb = subset_bank(raw_bank, np.arange(begin, begin + len(ids)))
            nb = self.neural_bank(prepared, rb)
            x = prepared['x'][ids]
            age = np.minimum(64, data['timestamp_second'][ids] - data['timestamp_second'].min()) / 64
            flags = torch.as_tensor(np.stack([clock['new_connection'][ids], age], 1).astype(np.float32), device=self.device)
            gap = torch.as_tensor(self.artifacts['gap_scale'].transform(clock['gap'][ids, None]).ravel(), device=self.device)
            valid = torch.as_tensor(clock['gap_valid'][ids], device=self.device)
            scores, shape_scores, time_scores, diagnostic = [], [], [], []
            base_shape, base_time, base_diag = [], [], []
            local_errors = torch.zeros_like(x)
            with torch.inference_mode():
                for group, m in enumerate(mask_table(self.device)):
                    mask = m.expand_as(x)
                    shape, timing, diag, _, legacy = self.fusion_model(x, mask, nb, flags, return_base=True)
                    bs, bt, lp, bd = legacy
                    local_errors += F.smooth_l1_loss(lp, x, reduction='none') * mask
                    base_shape.append(shape_nll(x, mask, bs).cpu().numpy().astype(np.float64))
                    base_time.append(time_nll(gap, bt).cpu().numpy().astype(np.float64))
                    base_diag.append({k: v.cpu().numpy().astype(np.float64) for k, v in bd.items()})
                    score, sn, tn = composite(x, mask, gap, valid, shape, timing)
                    ds = density_diagnostics(x, mask, shape)
                    diag['fusion_residual_mse'] = ds['residual_mse']
                    scores.append(score.cpu().numpy().astype(np.float64))
                    shape_scores.append(sn.cpu().numpy().astype(np.float64))
                    time_scores.append(tn.cpu().numpy().astype(np.float64))
                    diagnostic.append({k: v.cpu().numpy().astype(np.float64) for k, v in diag.items()})
                    if getattr(self, 'collect_normal', False):
                        pits.append(ds['pit'].cpu().numpy())
                        sigmas.append(ds['sigma'].cpu().numpy().ravel())
                        if valid.any():
                            tl, tm, ts = [v[valid] for v in timing]
                            tcdf = .5 * (1 + torch.erf((gap[valid, None] - tm) / (ts * np.sqrt(2))))
                            time_pits.append((tl.exp() * tcdf).sum(1).cpu().numpy())
                            time_sigmas.append(ts.cpu().numpy().ravel())
                    if details:
                        for p, row in enumerate(ids):
                            predictions.append(dict(row_id=int(data['row_id'][row]), mask_group=group,
                                predictor='fusion', masked_coordinates=np.flatnonzero(m.cpu().numpy()).tolist(),
                                means=shape[1][p].cpu().numpy().tolist(), scales=shape[2][p].cpu().numpy().tolist(),
                                weights=shape[0][p].exp().cpu().numpy().tolist(),
                                time_means=timing[1][p].cpu().numpy().tolist(),
                                time_scales=timing[2][p].cpu().numpy().tolist(),
                                private_rows=rb['private'][p][rb['private'][p]>=0].tolist(),
                                historical_rows=rb['historical'][p][rb['historical'][p]>=0].tolist(),
                                current_rows=rb['current'][p][rb['current'][p]>=0].tolist()))
            local = (local_errors / mask_table(self.device).sum(0)).mean(-1).cpu().numpy().astype(np.float64)
            packet_count, byte_count = data['features'][ids, 0], data['features'][ids, 1]
            single = packet_count == 1
            for length in np.unique(byte_count[single]):
                local[single & (byte_count == length)] = self.singleton(length)
            bshape, btime = np.mean(base_shape, 0), np.mean(base_time, 0)
            valid_np = valid.cpu().numpy()
            frame_part = pd.DataFrame(dict(row_id=data['row_id'][ids], timestamp_second=data['timestamp_second'][ids],
                local_raw=local, shape_raw=bshape, timing_raw=np.where(valid_np, btime, np.nan),
                context_raw=(bshape + np.where(valid_np, btime, 0)) / (1 + valid_np.astype(float)),
                history_seconds_available=np.minimum(64, data['timestamp_second'][ids] - data['timestamp_second'].min()),
                private_records=(rb['private'] >= 0).sum(1), historical_records=(rb['historical'] >= 0).sum(1),
                current_records=(rb['current'] >= 0).sum(1) if self.backbone == 'HN' else 0,
                historical_relation_incidence_count=rb['historical_counts'].sum(1),
                current_relation_incidence_count=rb['current_counts'].sum(1),
                packet_count=packet_count, byte_count=byte_count, new_connection=clock['new_connection'][ids],
                gap_available=valid_np, gap_seconds=np.where(valid_np, clock['gap'][ids], np.nan),
                gap_predecessor_row_id=clock['gap_predecessor'][ids],
                behavior_anchor=prepared['assignments'][ids],
                behavior_support=self.artifacts['behavior'].support[prepared['assignments'][ids]]))
            for key in base_diag[0]:
                frame_part[key] = np.mean([d[key] for d in base_diag], 0)
            frames.append(frame_part)
            extra = dict(fusion_raw=np.mean(scores, axis=0),
                fusion_shape_raw=np.mean(shape_scores, axis=0),
                fusion_timing_raw=np.where(valid.cpu().numpy(), np.mean(time_scores, axis=0), np.nan))
            for key in diagnostic[0]:
                extra[key] = np.mean([d[key] for d in diagnostic], axis=0)
            parts.append(pd.DataFrame(extra))
        frame = pd.concat(frames, ignore_index=True)
        values = pd.concat(parts, ignore_index=True)
        for key in values:
            frame[key] = values[key].to_numpy()
        frame['final_score'] = frame.fusion_raw
        if not np.isfinite(frame.final_score).all():
            raise FloatingPointError('Nonfinite composite fusion score')
        if pits:
            pit = np.concatenate(pits)
            sigma = np.concatenate(sigmas)
            self.normal_diagnostics = dict(training_in_sample=True,
                pit_histogram=np.histogram(pit, bins=np.linspace(0, 1, 11))[0].tolist(),
                pit_90_coverage=float(((pit >= .05) & (pit <= .95)).mean()),
                pit_95_coverage=float(((pit >= .025) & (pit <= .975)).mean()),
                sigma_quantiles=np.quantile(sigma, [.05, .5, .95]).tolist(),
                sigma_near_floor_fraction=float((sigma < self.cfg['sigma_floor'] * 1.1).mean()))
            if time_pits:
                tpit = np.concatenate(time_pits)
                self.normal_diagnostics.update(time_pit_histogram=np.histogram(tpit, bins=np.linspace(0, 1, 11))[0].tolist(),
                    time_pit_90_coverage=float(((tpit >= .05) & (tpit <= .95)).mean()),
                    time_sigma_quantiles=np.quantile(np.concatenate(time_sigmas), [.05, .5, .95]).tolist())
        if self.legacy_scales is not None:
            self.apply_scales(frame)
        return (frame, predictions) if details else frame

    def apply_scales(self, frame):
        for name in ('local', 'context'):
            frame[name + '_scaled'] = affine(frame[name + '_raw'], self.legacy_scales[name])
        frame['legacy_final_score'] = frame.local_scaled + frame.context_scaled
        frame['final_score'] = frame.fusion_raw if self.fusion_model is not None else frame.legacy_final_score

    def score_cache(self, cache, out, normal=False):
        from .data import load
        data, manifest = load(Path(cache), normal=normal)
        if manifest.get('slice_seconds', 1) != self.cfg.get('slice_seconds', 1):
            raise ValueError('Checkpoint and cache Slice lengths differ')
        prepared = self.prepare_data(data)
        self.collect_normal = normal
        frame = self.run(prepared)
        if self.legacy_scales is None:
            if not normal:
                raise ValueError('Fit score scales only on the all-normal reference')
            self.legacy_scales = {name: dict(zip(('q50', 'q95'), map(float,
                frame[name + '_raw'].quantile([.5, .95])))) for name in ('local', 'context')}
            self.apply_scales(frame)
            self.legacy_scales.update(reference_q99=float(frame.final_score.quantile(.99)),
                reference_is_training_in_sample=True, final_score_is_composite_nll=self.fusion_model is not None)
            write_json(self.root / self.run_id / 'score_scales.json', self.legacy_scales)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=False)
        if normal:
            if self.fusion_model is not None:
                write_json(self.root / self.run_id / 'normal_prediction_diagnostics.json', self.normal_diagnostics)
            from .checks import check
            check(self, prepared)
            self.check_predictors(prepared)
            write_json(self.root / self.run_id / 'complete.json', dict(run_id=self.run_id,
                normal_rescored=True, normal_flows=len(data['row_id']), prediction_isolation_verified=True))
        return frame

    def check_predictors(self, prepared):
        if self.fusion_model is None:
            return
        n = len(prepared['x'])
        ids = np.unique(np.r_[np.linspace(0, n - 1, min(16, n)).astype(int),
                              np.flatnonzero(prepared['clock']['gap_valid'])[:8]])
        from .graph import subset_bank
        bank = self.neural_bank(prepared, subset_bank(prepared['bank'], ids))
        x = prepared['x'][ids]
        clock = prepared['clock']
        age = np.minimum(64, prepared['data']['timestamp_second'][ids] - prepared['data']['timestamp_second'].min()) / 64
        flags = torch.as_tensor(np.stack([clock['new_connection'][ids], age], 1).astype(np.float32), device=self.device)
        with torch.inference_mode():
            for m in mask_table(self.device):
                mask = m.expand_as(x)
                original = self.fusion_model(x, mask, bank, flags)
                changed = self.fusion_model(x + mask * 3, mask, bank, flags)
                for a, b in zip(original[0], changed[0]):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                alternate = dict(bank)
                alternate['current_z'] = bank['current_z'] + 2
                alternate['current_valid'] = torch.ones_like(bank['current_valid'])
                current = self.fusion_model(x, mask, alternate, flags)
                for index in (0, 2):
                    torch.testing.assert_close(original[0][index], current[0][index], rtol=0, atol=0)
                for a, b in zip(original[1], current[1]):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                torch.testing.assert_close(original[3], current[3], rtol=0, atol=0)
        write_json(self.root / self.run_id / 'predictor_invariants.json', dict(
            real_normal_targets_checked=len(ids), masked_target_not_read=True,
            gap_not_in_predictor_signature=True, current_only_changes_shape_mean=True,
            current_cannot_change_gate=True, identity_is_lookup_only=True))
        from .graph import GraphIndex
        if self.cfg.get('private_time_policy') != 'past_relative_v1':
            raise ValueError('Arrival predictor needs history-only relative time')
        # Change target arrival in the actual graph, holding all observed past
        # nodes and Flow marks fixed. This catches gap copies in edge times.
        eligible = np.flatnonzero(prepared['clock']['gap_valid'] & ~prepared['clock']['new_connection'])
        probes = eligible[np.linspace(0, len(eligible)-1, min(8, len(eligible)), dtype=int)] if len(eligible) else ids[:8]
        for row in probes:
            row = int(row)
            changed_data = dict(prepared['data'])
            changed_data['first_packet_time'] = changed_data['first_packet_time'].copy()
            sec = changed_data['timestamp_second'][row]
            duration = self.cfg.get('slice_seconds', 1)
            relative_start = sec - int(prepared['data']['timestamp_second'].min())
            frac = changed_data['first_packet_time'][row] - relative_start
            changed_data['first_packet_time'][row] = relative_start + duration * (.75 if frac < .5*duration else .25)
            original_bank = subset_bank(prepared['bank'], np.asarray([row]))
            shifted_bank = GraphIndex(changed_data, self.cfg).bank(np.asarray([row]),
                np.random.default_rng(self.cfg['seed'] + 500), score_seed=self.cfg['seed'] + 500)
            np.testing.assert_array_equal(original_bank['private_dt'], shifted_bank['private_dt'])
            nb0 = self.neural_bank(prepared, original_bank)
            nb1 = self.neural_bank(prepared, shifted_bank)
            row_age = np.minimum(64, prepared['data']['timestamp_second'][row] - prepared['data']['timestamp_second'].min()) / 64
            row_flags = torch.as_tensor([[prepared['clock']['new_connection'][row], row_age]], dtype=torch.float32, device=self.device)
            with torch.inference_mode():
                for m in mask_table(self.device):
                    xx = prepared['x'][[row]]
                    mm = m.expand_as(xx)
                    tt0 = self.fusion_model(xx, mm, nb0, row_flags)[1]
                    tt1 = self.fusion_model(xx, mm, nb1, row_flags)[1]
                    for a, b in zip(tt0, tt1):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
        path = self.root / self.run_id / 'predictor_invariants.json'
        proof = json.loads(path.read_text())
        proof.update(target_arrival_graph_perturbation_checked=len(probes),
                     predicted_gap_absent_from_private_edge_times=True,
                     timing_parameters_unchanged_when_target_arrival_changes=True)
        write_json(path, proof)
