"""Masked Flow prediction, causal history, conditional density and relation heads."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

MASK_GROUPS = (
    (1, 3, 4, 5, 6, 7, 8, 9, 19, 20, 21),
    (2, *range(10, 25)),
    (0, 1, 2, *range(10, 25)),
)


def mask_table(device):
    masks = torch.zeros(3, 25, device=device)
    for i, columns in enumerate(MASK_GROUPS):
        masks[i, list(columns)] = 1
    return masks


class Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(25, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 32))


class Pretrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = nn.Sequential(nn.Linear(57, 64), nn.GELU(), nn.Linear(64, 25))

    def predict(self, x, mask):
        return self.decoder(torch.cat([self.encoder(x * (1 - mask)), mask], -1))


class Historical(nn.Module):
    def __init__(self, auxiliary_size=147):
        super().__init__()
        self.gru = nn.GRU(81, 64, batch_first=True)
        self.combine = nn.Sequential(nn.Linear(64 + auxiliary_size, 64), nn.GELU(),
                                     nn.Linear(64, 64))

    def forward(self, sequence, lengths, auxiliary):
        hidden = sequence.new_zeros((len(sequence), 64))
        valid = lengths > 0
        if valid.any():
            packed = pack_padded_sequence(sequence[valid], lengths[valid].cpu(),
                                         batch_first=True, enforce_sorted=False)
            _, state = self.gru(packed)
            hidden = hidden.index_copy(0, valid.nonzero().flatten(), state[0])
        return self.combine(torch.cat([hidden, auxiliary], -1))


class Model(nn.Module):
    def __init__(self, cfg, variant, support):
        super().__init__()
        self.variant = variant
        self.context_enabled = variant != "A_local"
        self.history_enabled = variant in ("C_history", "D_contrastive")
        self.relation_enabled = variant == "D_contrastive"
        self.encoder = Encoder()
        self.decoder = nn.Sequential(nn.Linear(121, 64), nn.GELU(), nn.Linear(64, 25))
        self.history = Historical(len(cfg["decay_seconds"]) * 49)
        self.components = cfg["mixture_components"]
        self.sigma_floor = cfg["sigma_floor"]
        self.density = nn.Sequential(nn.Linear(96, 64), nn.GELU(),
                                    nn.Linear(64, self.components * 61))
        self.mean_bias = nn.Parameter(torch.zeros(cfg["anchors"], self.components, 30))
        support = torch.as_tensor(support, dtype=torch.float32)
        self.register_buffer("shrink_weight", (support.max() + 1) / (support + 1))
        self.context_encoder = nn.Sequential(nn.Linear(30, 64), nn.GELU(), nn.Linear(64, 32))
        self.critic = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def initialize(self, pretrained):
        self.encoder.load_state_dict(pretrained.encoder.state_dict())
        with torch.no_grad():
            self.decoder[0].weight.zero_()
            self.decoder[0].weight[:, :32].copy_(pretrained.decoder[0].weight[:, :32])
            self.decoder[0].weight[:, 96:].copy_(pretrained.decoder[0].weight[:, 32:])
            self.decoder[0].bias.copy_(pretrained.decoder[0].bias)
            self.decoder[2].load_state_dict(pretrained.decoder[2].state_dict())

    def predict(self, x, mask, h):
        z = self.encoder(x * (1 - mask))
        return self.decoder(torch.cat([z, h, mask], -1))

    def distribution(self, z, h, phi):
        parameters = self.density(torch.cat([z, h], -1))
        logits = parameters[:, :self.components]
        mu, rho = parameters[:, self.components:].reshape(-1, self.components, 60).split(30, -1)
        weight = phi / phi.sum(-1, keepdim=True)
        mu = mu + torch.einsum("nk,kcd->ncd", weight, self.mean_bias)
        sigma = F.softplus(rho) + self.sigma_floor
        return F.log_softmax(logits, -1), mu, sigma

    def nll(self, target, distribution):
        log_weights, mu, sigma = distribution
        coordinate = -0.5 * ((target[:, None] - mu) / sigma).square()
        coordinate -= sigma.log() + 0.5 * math.log(2 * math.pi)
        return -torch.logsumexp(log_weights + coordinate.sum(-1), dim=-1) / 30

    def shrinkage(self):
        return (self.mean_bias.square().mean((1, 2)) * self.shrink_weight).mean()

    def relation(self, z, h, context):
        return self.critic(torch.cat([z, h, self.context_encoder(context)], -1)).flatten()


def masked_loss(model, x, mask, h=None):
    predicted = model.predict(x, mask) if h is None else model.predict(x, mask, h)
    # Correct for coordinate coverage under uniform group sampling.
    coverage = mask_table(x.device).mean(0)
    errors = F.smooth_l1_loss(predicted, x, reduction="none")
    return (errors * mask / coverage).mean()


def local_score(model, x, h):
    masks = mask_table(x.device)
    error = torch.zeros_like(x)
    for mask in masks:
        selected = mask.expand_as(x)
        prediction = model.predict(x, selected, h)
        error += F.smooth_l1_loss(prediction, x, reduction="none") * selected
    return (error / masks.sum(0)).mean(-1)


def finite(value, name):
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"Nonfinite {name}; stopped before saving invalid parameters")


def affine(scores, reference):
    scores = np.asarray(scores, dtype=np.float64)
    result = (scores - reference["q50"]) / max(reference["q95"] - reference["q50"], 1e-6)
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite continuous score")
    return result
