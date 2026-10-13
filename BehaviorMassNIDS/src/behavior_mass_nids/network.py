"""Original Flow measurement architecture; see docs/SOURCE_MAP.csv."""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

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


def masked_loss(model, x, mask, h=None):
    predicted = model.predict(x, mask) if h is None else model.predict(x, mask, h)
    # Correct for coordinate coverage under uniform group sampling.
    coverage = mask_table(x.device).mean(0)
    errors = F.smooth_l1_loss(predicted, x, reduction="none")
    return (errors * mask / coverage).mean()


def finite(value, name):
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"Nonfinite {name}; stopped before saving invalid parameters")


class LongMassModel(nn.Module):
    """An identical bounded-state MDN, with optional jointly trained GRU/CL."""
    def __init__(self,cfg,variant,artifact):
        super().__init__();self.variant=variant;self.cfg=cfg
        self.k=int(cfg.get('behavior_k',16));self.states=int(cfg['normal_states'])
        self.gru=nn.GRU(3*self.k,64,batch_first=True)
        self.selector=nn.Sequential(nn.Linear(64+self.k+2,64),nn.GELU(),nn.Linear(64,self.states))
        self.mean_logits=nn.Parameter(torch.as_tensor(artifact['mean_logits']).clone())
        self.scale_logits=nn.Parameter(torch.as_tensor(artifact['scale_logits']).clone())
        self.register_buffer('lower',torch.as_tensor(artifact['lower']))
        self.register_buffer('upper',torch.as_tensor(artifact['upper']))
        self.query_projector=nn.Sequential(nn.Linear(64+self.k+2,64),nn.GELU(),nn.Linear(64,32))
        self.target_projector=nn.Sequential(nn.Linear(3,64),nn.GELU(),nn.Linear(64,32))
        # Nonzero shared random initialization gives the GRU a prediction gradient.
        nn.init.normal_(self.selector[-1].weight,std=.02);nn.init.zeros_(self.selector[-1].bias)
        if variant=='BASE':self.gru.requires_grad_(False)
        if variant!='BASE_GRU_CL':
            self.query_projector.requires_grad_(False);self.target_projector.requires_grad_(False)

    def forward(self,history,lengths,roles,directions):
        h=history.new_zeros((len(history),64))
        if self.variant!='BASE':
            ok=lengths>0
            if ok.any():
                packed=nn.utils.rnn.pack_padded_sequence(history[ok],lengths[ok].cpu(),batch_first=True,enforce_sorted=False)
                _,state=self.gru(packed);h[ok]=state[0]
        query=torch.cat([h,F.one_hot(roles,self.k).to(h.dtype),F.one_hot(directions,2).to(h.dtype)],-1)
        weight=self.selector(query).log_softmax(-1)
        center=self.lower+(self.upper-self.lower)*self.mean_logits.sigmoid()
        sigma=F.softplus(self.scale_logits)+self.cfg['mass_sigma_floor']
        return (weight,center[:,roles].permute(1,0,2),sigma[:,roles].permute(1,0,2)),query

    @staticmethod
    def nll(target,dist):
        w,mu,sigma=dist
        coord=-.5*((target[:,None]-mu)/sigma).square()-sigma.log()-.5*math.log(2*math.pi)
        return -torch.logsumexp(w+coord.sum(-1),-1)/3

    def contrastive(self,query,targets,negative,negative_valid,eligible):
        if not eligible.any():return query.sum()*0
        q=F.normalize(self.query_projector(query[eligible]),dim=-1)
        pos=F.normalize(self.target_projector(targets[eligible]),dim=-1)
        neg=F.normalize(self.target_projector(negative[eligible]),dim=-1)
        positive=(q*pos).sum(-1,keepdim=True)
        negatives=torch.einsum('nd,nkd->nk',q,neg)
        negatives=negatives.masked_fill(~negative_valid[eligible],-torch.inf)
        logits=torch.cat([positive,negatives],-1)/self.cfg['temperature']
        return F.cross_entropy(logits,torch.zeros(len(q),dtype=torch.long,device=q.device))
