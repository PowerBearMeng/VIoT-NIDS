"""Shared trainable Flow encoder, private TCG and bounded graph corrections."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

from viot_nids_v7.network import Encoder, mask_table, finite


class RelationAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.types = nn.Embedding(6, 8)  # Relation types, never endpoint identities.
        self.time = nn.Sequential(nn.Linear(1, 16), nn.GELU())
        self.content = nn.Sequential(nn.Linear(56, 64), nn.GELU())
        self.query = nn.Linear(64, 64)
        self.key = nn.Linear(64, 64)
        self.value = nn.Linear(64, 64)
        self.gate = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
        self.output = nn.Linear(64, 64, bias=False)

    def forward(self, query, z, types, dt, valid, bounded_mean=False, quality=None):
        content = self.content(torch.cat([z, self.types(types), self.time(dt[..., None])], -1))
        q = self.query(query).reshape(-1, 4, 16)
        k = self.key(content).reshape(len(query), -1, 4, 16)
        v = self.value(content).reshape(len(query), -1, 4, 16)
        similarities = (q[:, None] * k).sum(-1) / 4
        gate_logits = self.gate(torch.cat([query[:, None].expand(-1, z.shape[1], -1), content], -1)).squeeze(-1)
        gates = torch.sigmoid(gate_logits) * valid
        if quality is not None:
            # Reliability attenuates messages; it never changes the denominator.
            gates = gates * quality.detach()
        if bounded_mean:
            # Average separately within each relation. Low gates do not get
            # renormalized back to unit weight, and multiplicity cannot amplify it.
            head_gates = torch.sigmoid(similarities) * gates[..., None]
            messages = []
            for rel in range(4):
                selected = valid & (types == rel)
                denom = selected.sum(1).clamp_min(1).to(z.dtype)
                messages.append((v * (head_gates * selected[..., None])[..., None]).sum(1) / denom[:, None, None])
            message = sum(messages) / 4
        else:
            logits = similarities.masked_fill(~valid[..., None], -1e9)
            weights = logits.softmax(1) * valid[..., None]
            weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
            message = (v * (weights * gates[..., None])[..., None]).sum(1)
        any_valid = valid.any(1).to(z.dtype)
        output = self.output(message.reshape(-1, 64)) * any_valid[:, None]
        count = valid.sum(1).clamp_min(1)
        return output, (gates.sum(1) / count), gate_logits, content


class Model(nn.Module):
    def __init__(self, cfg, variant):
        super().__init__()
        self.variant = variant
        self.encoder = Encoder()
        self.local_decoder = nn.Sequential(nn.Linear(57, 64), nn.GELU(), nn.Linear(64, 25))
        self.query = nn.Sequential(nn.Linear(59, 64), nn.GELU())  # masked z + mask + flags
        self.private = RelationAttention()
        self.historical = RelationAttention()
        self.current = RelationAttention()
        self.private_combine = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 64))
        self.density = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 102))
        self.history_delta = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 50))
        self.current_delta = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 50))
        nn.init.zeros_(self.current_delta[-1].weight)
        nn.init.zeros_(self.current_delta[-1].bias)
        self.timing = nn.Sequential(nn.Linear(98, 64), nn.GELU(), nn.Linear(64, 6))
        self.sigma_floor = cfg["sigma_floor"]

    def initialize(self, pretrained):
        self.encoder.load_state_dict(pretrained.encoder.state_dict())
        self.local_decoder.load_state_dict(pretrained.decoder.state_dict())

    def predict(self, x, mask):
        return self.local_decoder(torch.cat([self.encoder(x * (1 - mask)), mask], -1))

    def forward(self, x, mask, bank, flags):
        # No complete target representation enters its shape predictor.
        query = self.query(torch.cat([self.encoder(x * (1 - mask)), mask, flags], -1))
        private, pg, _, _ = self.private(query, bank["private_z"], bank["private_type"],
                                        bank["private_dt"], bank["private_valid"], quality=bank.get('private_quality'))
        state = self.private_combine(torch.cat([query, private], -1))
        historical, hg, _, _ = self.historical(state, bank["historical_z"], bank["historical_type"],
                                              bank["historical_dt"], bank["historical_valid"], True, quality=bank.get('historical_quality'))
        current = state.new_zeros(state.shape)
        cg = state.new_zeros(len(state))
        if self.variant == "HN":
            current, cg, _, _ = self.current(state, bank["current_z"], bank["current_type"],
                                             bank["current_dt"], bank["current_valid"], True, quality=bank.get('current_quality'))
        parameters = self.density(state)
        logits = F.log_softmax(parameters[:, :2], -1)
        mu, rho = parameters[:, 2:].reshape(-1, 2, 50).split(25, -1)
        sigma = F.softplus(rho) + self.sigma_floor
        hd = self.history_delta(historical).reshape(-1, 2, 25)
        hd = hd * hg[:, None, None]
        cd = self.current_delta(current).reshape(-1, 2, 25)
        cd = cd * cg[:, None, None] if self.variant == "HN" else torch.zeros_like(cd)
        correction = torch.tanh(hd + cd)
        mu = mu + sigma * correction
        # Arrival model has no current or historical cross-connection messages.
        # Target mark is allowed. GraphIndex must use past_relative_v1 to
        # exclude a copy of the target gap from private edge times; legacy
        # target-relative edge times do not satisfy this requirement.
        timing = self.timing(torch.cat([self.encoder(x), private, flags], -1))
        tlog = F.log_softmax(timing[:, :2], -1)
        tmu, trho = timing[:, 2:4], timing[:, 4:]
        tsigma = F.softplus(trho) + self.sigma_floor
        diagnostics = dict(private_gate=pg, history_gate=hg, current_gate=cg,
                           mean_sigma=sigma.mean((1, 2)),
                           correction_sigma_fraction=correction.abs().mean((1, 2)),
                           correction_max_sigma_fraction=correction.abs().amax((1, 2)),
                           mean_center=mu.mean((1, 2)), time_center=tmu.mean(1), time_sigma=tsigma.mean(1))
        return (logits, mu, sigma), (tlog, tmu, tsigma), diagnostics, state

    def relation_loss(self, x, mask, bank, flags, rng):
        query = self.query(torch.cat([self.encoder(x * (1 - mask)), mask, flags], -1))
        p, _, _, _ = self.private(query, bank["private_z"], bank["private_type"],
                                   bank["private_dt"], bank["private_valid"], quality=bank.get('private_quality'))
        state = self.private_combine(torch.cat([query, p], -1))
        terms = []
        for name in ("private", "historical", "current"):
            if name == "current" and self.variant != "HN":
                continue
            layer = getattr(self, name)
            q = query if name == "private" else state
            z, types, dt, valid = [bank[name + suffix] for suffix in ("_z", "_type", "_dt", "_valid")]
            quality = bank.get(name + '_quality')
            _, _, positive, _ = layer(q, z, types, dt, valid, name != "private", quality=quality)
            if not valid.any():
                continue
            # Re-pair with another normal target/history, leaving real counts intact.
            # Half random, half same normal window-count quartile; guaranteed different row.
            perm = torch.as_tensor(rng, device=x.device)
            _, _, negative, _ = layer(q, z[perm], types[perm], dt[perm], valid[perm], name != "private",
                                     quality=None if quality is None else quality[perm])
            if not valid[perm].any():
                # A rare relation can have a real edge but no edge among this
                # batch's resampled contexts. An empty reduction is not a loss.
                continue
            pos_loss = F.softplus(-positive)[valid].mean()
            neg_loss = F.softplus(negative)[valid[perm]].mean()
            terms.append(.5 * (pos_loss + neg_loss))
        return sum(terms) / len(terms) if terms else x.sum() * 0


def shape_nll(x, mask, distribution):
    logits, mu, sigma = distribution
    coordinate = -.5 * ((x[:, None] - mu) / sigma).square() - sigma.log() - .5 * math.log(2 * math.pi)
    return -torch.logsumexp(logits + (coordinate * mask[:, None]).sum(-1), -1) / mask.sum(-1)


def time_nll(target, distribution):
    logits, mu, sigma = distribution
    coordinate = -.5 * ((target[:, None] - mu) / sigma).square() - sigma.log() - .5 * math.log(2 * math.pi)
    return -torch.logsumexp(logits + coordinate, -1)


def local_loss(model, x, mask):
    error = F.smooth_l1_loss(model.predict(x, mask), x, reduction="none")
    coverage = mask_table(x.device).mean(0)
    return (error * mask / coverage).mean()


def local_score(model, x):
    masks = mask_table(x.device)
    errors = torch.zeros_like(x)
    for m in masks:
        mask = m.expand_as(x)
        errors += F.smooth_l1_loss(model.predict(x, mask), x, reduction="none") * mask
    return (errors / masks.sum(0)).mean(-1)
