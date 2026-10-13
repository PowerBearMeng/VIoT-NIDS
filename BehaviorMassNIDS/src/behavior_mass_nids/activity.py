"""Actual FPB totals and strictly past activity indices, without labels."""
from __future__ import annotations
import numpy as np
import torch
RESOURCE={"f":0,"p":1,"b":2}

class Activity:
    """Full actual mass, sparse queries; no sampled copies or NxN relations."""
    def __init__(self, data, roles, grouping='behavior', history=16, behavior_count=None):
        self.data = data; self.grouping = grouping
        self.k = (16 if behavior_count is None else int(behavior_count)) if grouping == 'behavior' else 1
        if self.k<1:raise ValueError('Behavior count must be positive')
        self.roles = np.asarray(roles, np.int64) if grouping=='behavior' else np.zeros(len(roles), np.int64)
        if self.roles.min()<0 or self.roles.max()>=self.k:raise ValueError('Behavior assignments outside declared K')
        self.origin = int(data['timestamp_second'].min())
        self.span = int(data['timestamp_second'].max()) - self.origin + 1
        self.seconds = data['timestamp_second'] - self.origin
        self.nodes = np.unique(np.r_[data['source_node'],data['destination_node']])
        self.units = 2*len(self.nodes); self.history = history
        self.sender = 2*np.searchsorted(self.nodes,data['source_node'])
        self.receiver = 2*np.searchsorted(self.nodes,data['destination_node'])+1
        self.source_cell = (self.sender*self.span+self.seconds)*self.k+self.roles
        self.destination_cell = (self.receiver*self.span+self.seconds)*self.k+self.roles
        codes = np.r_[self.source_cell,self.destination_cell]
        mass = np.stack([np.ones(len(roles)),data['features'][:,0],data['features'][:,1]],-1)
        totals = np.stack([np.bincount(codes,weights=np.tile(mass[:,j],2),minlength=self.units*self.span*self.k) for j in range(3)],-1)
        np.testing.assert_allclose(totals.sum(0),2*mass.sum(0),rtol=0,atol=0)
        self.totals = totals.reshape(self.units*self.span,self.k,3)
        self.active = np.flatnonzero(totals[:,0]>0)
        self.ut = self.active//self.k; self.role = self.active%self.k
        self.unit = self.ut//self.span; self.t = self.ut%self.span; self.direction = self.unit%2
        birth = np.full(self.units,self.span,np.int64)
        np.minimum.at(birth,self.sender,self.seconds); np.minimum.at(birth,self.receiver,self.seconds)
        self.birth = np.repeat(birth.reshape(-1,2).min(1),2)
        self.source_pair = np.searchsorted(self.active,self.source_cell)
        self.destination_pair = np.searchsorted(self.active,self.destination_cell)
        np.testing.assert_array_equal(self.active[self.source_pair],self.source_cell)
        np.testing.assert_array_equal(self.active[self.destination_pair],self.destination_cell)
        self.lengths = np.minimum(history,self.t-self.birth[self.unit]).clip(0).astype(np.int64)
        positions = self.t[:,None]-self.lengths[:,None]+np.arange(history)
        valid = np.arange(history)[None]<self.lengths[:,None]
        assert (positions[valid]<np.broadcast_to(self.t[:,None],positions.shape)[valid]).all()
        self.history_indices = self.unit[:,None]*self.span+positions
        self.history_indices[~valid] = self.units*self.span

    def view(self, resources):
        return self.totals[:,:,[RESOURCE[r] for r in resources]]


def mass_tensors(activity, artifact, resources, device):
    view=activity.view(resources); d=len(resources)
    h=artifact['input_scale'].transform(view.reshape(-1,activity.k*d))
    h=np.r_[h,np.zeros((1,activity.k*d),np.float32)]
    raw=view.reshape(-1,d)[activity.active]; target=np.empty_like(raw,dtype=np.float32)
    for role,scale in enumerate(artifact['target_scales']):
        ix=activity.role==role;target[ix]=scale.transform(raw[ix])
    return dict(history=torch.as_tensor(h,device=device),target=torch.as_tensor(target,device=device),
                role=torch.as_tensor(activity.role,device=device),direction=torch.as_tensor(activity.direction,device=device))


def mass_batch(activity, arrays, ix, device):
    return (arrays['history'][torch.as_tensor(activity.history_indices[ix],device=device)],
            torch.as_tensor(activity.lengths[ix],device=device),arrays['role'][ix],arrays['direction'][ix])


def quantiles(values):
    q50,q95=np.quantile(values,[.5,.95]);return dict(q50=float(q50),q95=float(q95),reference_q99=float(np.quantile(values,.99)))


def affine(values,scale):
    return (values-scale['q50'])/max(scale['q95']-scale['q50'],1e-6)
