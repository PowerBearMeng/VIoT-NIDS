"""Normal-only representation fusion. Scores and labels are never model inputs."""
from __future__ import annotations
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from .network import Model, shape_nll, time_nll


class FusionModel(nn.Module):
    def __init__(self, cfg, backbone, variant):
        super().__init__()
        self.base = Model(cfg, backbone)
        self.variant = variant
        self.gate = nn.Sequential(nn.Linear(156, 64), nn.GELU(), nn.Linear(64, 64))
        self.delta = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 64))
        self.density = copy.deepcopy(self.base.density)
        self.timing = nn.Sequential(nn.Linear(162, 64), nn.GELU(), nn.Linear(64, 6))
        self.critic = nn.Sequential(nn.Linear(121, 64), nn.GELU(), nn.Linear(64, 1))
        for layer in (self.gate[-1], self.delta[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    @property
    def encoder(self):
        return self.base.encoder

    def initialize(self, state):
        self.base.load_state_dict(state)
        self.density.load_state_dict(self.base.density.state_dict())
        with torch.no_grad():
            self.timing[0].weight.zero_()
            self.timing[0].weight[:, :98].copy_(self.base.timing[0].weight)
            self.timing[0].bias.copy_(self.base.timing[0].bias)
            self.timing[2].load_state_dict(self.base.timing[2].state_dict())

    def represent(self, x, mask, bank, flags):
        b = self.base
        z = b.encoder(x * (1 - mask))
        local = b.local_decoder[:2](torch.cat([z, mask], -1))
        query = b.query(torch.cat([z, mask, flags], -1))
        private, pg, _, _ = b.private(query, bank['private_z'], bank['private_type'],
                                     bank['private_dt'], bank['private_valid'],
                                     quality=bank.get('private_quality'))
        private = self.private_reference(query, private, mask, bank)
        context = b.private_combine(torch.cat([query, private], -1))
        available = bank['private_valid'].any(1).to(x.dtype)[:, None]
        gate_input = torch.cat([local, context, mask, flags, available], -1)
        gate = torch.full_like(context, .5) if self.variant == 'fixed' else self.gate(gate_input).sigmoid()
        correction = gate * self.delta(torch.cat([local, context], -1))
        return context + correction, context, private, gate, correction, pg, local

    def private_reference(self, query, private, mask, bank):
        return private

    def forward(self, x, mask, bank, flags, return_base=False):
        b = self.base
        u, context, private, gate, correction, pg, local = self.represent(x, mask, bank, flags)
        historical, hg, _, _ = b.historical(context, bank['historical_z'], bank['historical_type'],
                                            bank['historical_dt'], bank['historical_valid'], True,
                                            quality=bank.get('historical_quality'))
        hd = b.history_delta(historical).reshape(-1, 2, 25) * hg[:, None, None]
        cd = torch.zeros_like(hd)
        cg = x.new_zeros(len(x))
        if b.variant == 'HN':
            current, cg, _, _ = b.current(context, bank['current_z'], bank['current_type'],
                                         bank['current_dt'], bank['current_valid'], True,
                                         quality=bank.get('current_quality'))
            cd = b.current_delta(current).reshape(-1, 2, 25) * cg[:, None, None]
        parameters = self.density(u)
        logits = F.log_softmax(parameters[:, :2], -1)
        mu, rho = parameters[:, 2:].reshape(-1, 2, 50).split(25, -1)
        sigma = F.softplus(rho) + b.sigma_floor
        bounded = torch.tanh(hd + cd)
        mu = mu + sigma * bounded
        full_z = b.encoder(x)
        timing = self.timing(torch.cat([full_z, private, flags, u], -1))
        time_dist = (F.log_softmax(timing[:, :2], -1), timing[:, 2:4],
                     F.softplus(timing[:, 4:]) + b.sigma_floor)
        diag = dict(fusion_gate_mean=gate.mean(1), fusion_gate_std=gate.std(1),
                    fusion_delta_norm=correction.norm(dim=1),
                    fusion_mean_sigma=sigma.mean((1, 2)),
                    fusion_floor_fraction=(sigma < b.sigma_floor * 1.1).float().mean((1, 2)),
                    fusion_correction_max=bounded.abs().amax((1, 2)),
                    fusion_private_gate=pg, fusion_history_gate=hg, fusion_current_gate=cg)
        result = ((logits, mu, sigma), time_dist, diag, u)
        if return_base:
            bp = b.density(context)
            bl = F.log_softmax(bp[:, :2], -1)
            bm, br = bp[:, 2:].reshape(-1, 2, 50).split(25, -1)
            bs = F.softplus(br) + b.sigma_floor
            bm = bm + bs * bounded
            bt = b.timing(torch.cat([full_z, private, flags], -1))
            btime = (F.log_softmax(bt[:, :2], -1), bt[:, 2:4], F.softplus(bt[:, 4:]) + b.sigma_floor)
            bdiag = dict(private_gate=pg, history_gate=hg, current_gate=cg,
                mean_sigma=bs.mean((1, 2)), correction_sigma_fraction=bounded.abs().mean((1, 2)),
                correction_max_sigma_fraction=bounded.abs().amax((1, 2)),
                mean_center=bm.mean((1, 2)), time_center=btime[1].mean(1), time_sigma=btime[2].mean(1))
            return (*result, ((bl, bm, bs), btime, b.local_decoder[-1](local), bdiag))
        return result


def composite(x, mask, gap, valid, shape, timing):
    # The first density is joint over masked coordinates; the second conditions
    # on the observed complete Flow. Missing gaps are marginalized, not zeros.
    sn = shape_nll(x, mask, shape)
    tn = torch.zeros_like(sn)
    if valid.any():
        tn[valid] = time_nll(gap[valid], tuple(v[valid] for v in timing))
    dims = mask.sum(-1)
    return (sn * dims + tn) / (dims + valid.to(x.dtype)), sn, tn


def density_diagnostics(x, mask, distribution):
    weights, means, scales = distribution
    probabilities = weights.exp()[..., None]
    center = (probabilities * means).sum(1)
    residual = (x - center).square()
    cdf = .5 * (1 + torch.erf((x[:, None] - means) / (scales * math.sqrt(2))))
    pit = (probabilities * cdf).sum(1)
    selected = mask.bool()
    return dict(residual_mse=(residual * mask).sum(1) / mask.sum(1),
                pit=pit[selected], sigma=scales[:, :, selected[0]])
