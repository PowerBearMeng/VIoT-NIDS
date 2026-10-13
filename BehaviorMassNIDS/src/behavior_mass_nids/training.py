"""Original normal-state initialization and full-coverage batch plan."""
from __future__ import annotations
import math
import numpy as np
from .scaling import Scale
import csv
import json
import logging
import random
import time
from pathlib import Path
import torch
from sklearn.cluster import KMeans
from .activity import Activity, mass_tensors, mass_batch, quantiles, affine
from .network import Pretrain, LongMassModel, mask_table, masked_loss, finite
from .scaling import Behavior
from .model import scale_dict, read_measurement, Detector
from .io import load_capture, write_json

def fitted_artifact(a,cfg):
    raw=a.totals.reshape(-1,3)[a.active].astype(np.float32)
    observed=np.arange(a.span)[None]>=a.birth[:,None]
    input_scale=Scale.fit(a.totals.reshape(-1,3*a.k)[observed.ravel()],cfg['feature_scale_floor'])
    target_scales=[];lower=[];upper=[];means=[];widths=[]
    rng=np.random.default_rng(cfg['seed'])
    for role in range(a.k):
        scale=Scale.fit(raw[a.role==role],cfg['feature_scale_floor']);target_scales.append(scale)
        y=scale.transform(raw[a.role==role]);lo=y.min(0);hi=y.max(0)
        # Normal quantile initialization is fitted preprocessing, not old weights.
        mu=np.quantile(y,np.linspace(.05,.95,cfg['normal_states']),axis=0)
        fraction=(mu-lo)/np.maximum(hi-lo,1e-6)
        logits=np.log(np.clip(fraction,.001,.999)/np.clip(1-fraction,.001,.999))+rng.normal(0,.01,(cfg['normal_states'],3))
        width=np.maximum(y.std(0),.3)-cfg['mass_sigma_floor']
        raw_scale=width+np.log(-np.expm1(-width))
        lower.append(lo);upper.append(hi);means.append(logits);widths.append(np.tile(raw_scale,(cfg['normal_states'],1)))
    return dict(input_scale=input_scale,target_scales=target_scales,lower=np.asarray(lower,np.float32),
        upper=np.asarray(upper,np.float32),mean_logits=np.asarray(means,np.float32).transpose(1,0,2),
        scale_logits=np.asarray(widths,np.float32).transpose(1,0,2),resources='fpb',normal_pairs=len(a.active))


def batch_plan(a,cfg,epoch):
    rng=np.random.default_rng(cfg['seed']+10000+epoch);order=rng.permutation(len(a.active));n=cfg['batch_size']
    batches=[order[i:i+n] for i in range(0,len(order),n)]
    pools={k:[np.flatnonzero((a.role==k)&(a.t//10==b)) for b in np.unique(a.t[a.role==k]//10)] for k in range(a.k)}
    while len(batches)<max(cfg['steps_per_epoch'],math.ceil(len(order)/n)):
        draw=[]
        for _ in range(n):
            k=int(rng.integers(a.k));block=pools[k][int(rng.integers(len(pools[k])))];draw.append(rng.choice(block))
        batches.append(np.asarray(draw,np.int64))
    return batches


class Balanced:
    def __init__(self, assignments, seconds, rng):
        self.rng = rng
        blocks = (seconds - seconds.min()) // 10
        self.groups = []
        for a in np.unique(assignments):
            ix = np.flatnonzero(assignments == a)
            self.groups.append([ix[blocks[ix] == b] for b in np.unique(blocks[ix])])

    def draw(self, n):
        rows = []
        for _ in range(n):
            behavior = self.groups[self.rng.integers(len(self.groups))]
            block = behavior[self.rng.integers(len(behavior))]
            rows.append(block[self.rng.integers(len(block))])
        return np.asarray(rows, dtype=np.int64)


def epoch_batches(n, cfg, sampler, rng):
    permutation = rng.permutation(n)
    batch = cfg["batch_size"]
    full = math.ceil(n / batch)
    for begin in range(0, n, batch):
        yield permutation[begin:begin + batch]
    for _ in range(max(cfg["steps_per_epoch"], full) - full):
        yield sampler.draw(batch)


def setup(cfg):
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    torch.set_num_threads(cfg['torch_threads'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def measure(model, data, scale, cfg, device):
    result = np.empty((len(data['features']),32), np.float32)
    with torch.inference_mode():
        for begin in range(0,len(result),cfg['score_batch_size']):
            stop=min(len(result),begin+cfg['score_batch_size'])
            x=torch.as_tensor(scale.transform(data['features'][begin:stop]),device=device)
            result[begin:stop]=model.encoder(x).cpu().numpy()
    return result


def train(cache, output, cfg, measurement_bundle=None, reuse_initial=False):
    """All-normal measurement initialization, then fresh GRU/MDN training.

    A supplied measurement bundle shares f0/prototypes, not a trained detector.
    reuse_initial reads only initial_mass.pt (the original untrained state).
    """
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    if (root/'bundle.json').exists() or (root/'mass_model.pt').exists():
        raise FileExistsError('Use a fresh training output directory')
    if reuse_initial and not measurement_bundle:
        raise ValueError('An initial-state reproduction requires --measurement-bundle')
    data=load_capture(cache,normal=True);device=setup(cfg)
    if measurement_bundle:
        metadata,flow_scale,behavior,encoder=read_measurement(measurement_bundle)
        if len(behavior.centers)!=cfg['behavior_k']:
            raise ValueError('Measurement bundle and configured K differ')
    else:
        flow_scale=Scale.fit(data['features'],cfg['feature_scale_floor'])
        x=flow_scale.transform(data['features']);rng=np.random.default_rng(cfg['seed'])
        clusters=KMeans(n_clusters=16,n_init=10,random_state=cfg['seed']).fit_predict(x)
        sampler=Balanced(clusters,data['timestamp_second'],rng)
        encoder=Pretrain().to(device);tensor=torch.as_tensor(x,device=device)
        optimizer=torch.optim.Adam(encoder.parameters(),lr=cfg['pretrain_lr'])
        with (root/'pretrain_log.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=('epoch','local_loss','rows_covered','steps','seconds'));writer.writeheader()
            for epoch in range(1,cfg['pretrain_epochs']+1):
                start=time.perf_counter();values=[];seen=set()
                for ix in epoch_batches(len(x),cfg,sampler,rng):
                    seen.update(map(int,ix));optimizer.zero_grad(set_to_none=True)
                    masks=mask_table(device)[torch.as_tensor(rng.integers(3,size=len(ix)),device=device)]
                    loss=masked_loss(encoder,tensor[ix],masks);finite(loss,'measurement loss');loss.backward()
                    grad=torch.nn.utils.clip_grad_norm_(encoder.parameters(),cfg['gradient_clip']);finite(grad,'measurement gradient')
                    optimizer.step();values.append(float(loss.detach()))
                if len(seen)!=len(x):raise ValueError('Incomplete normal Flow coverage')
                writer.writerow(dict(epoch=epoch,local_loss=np.mean(values),rows_covered=len(seen),steps=len(values),seconds=time.perf_counter()-start));handle.flush()
                logging.info('Measurement epoch %d/%d loss=%.5f',epoch,cfg['pretrain_epochs'],np.mean(values))
        encoder.eval().requires_grad_(False)
        behavior=Behavior.fit(measure(encoder,data,flow_scale,cfg,device),data['timestamp_second'],cfg['behavior_k'],cfg['seed'])
    encoder=encoder.float().to(device).eval().requires_grad_(False)
    roles=behavior.transform(measure(encoder,data,flow_scale,cfg,device)).argmax(1)
    if len(np.unique(roles))!=cfg['behavior_k']:raise ValueError('A normal prototype has no support')
    a=Activity(data,roles,history=cfg['history_seconds'],behavior_count=cfg['behavior_k'])
    artifact=fitted_artifact(a,cfg);arrays=mass_tensors(a,artifact,'fpb',device)
    # Detector weights start fresh, independently of measurement pretraining.
    setup(cfg);model=LongMassModel(cfg,'BASE_GRU',artifact).to(device)
    if reuse_initial:
        model.load_state_dict(torch.load(Path(measurement_bundle)/'initial_mass.pt',map_location=device,weights_only=True))
    torch.save(model.state_dict(),root/'initial_mass.pt')
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'])
    with (root/'train_log.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=('epoch','nll','gradient_norm','groups_covered','steps','seconds'));writer.writeheader()
        for epoch in range(1,cfg['epochs']+1):
            model.train();seen=set();values=[];start=time.perf_counter()
            for ix in batch_plan(a,cfg,epoch):
                seen.update(map(int,ix));optimizer.zero_grad(set_to_none=True)
                dist,_=model(*mass_batch(a,arrays,ix,device));loss=model.nll(arrays['target'][ix],dist).mean()
                finite(loss,'Mass NLL');loss.backward()
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip']);finite(grad,'Mass gradient')
                optimizer.step();values.append((float(loss.detach()),float(grad)))
            if len(seen)!=len(a.active):raise ValueError('Incomplete normal group coverage')
            avg=np.mean(values,axis=0)
            writer.writerow(dict(epoch=epoch,nll=avg[0],gradient_norm=avg[1],groups_covered=len(seen),steps=len(values),seconds=time.perf_counter()-start));handle.flush()
            logging.info('Mass epoch %d/%d NLL=%.5f',epoch,cfg['epochs'],avg[0])
    torch.save(encoder.cpu().state_dict(),root/'flow_encoder.pt');torch.save(model.cpu().state_dict(),root/'mass_model.pt')
    serial={k:artifact[k].tolist() for k in ('lower','upper','mean_logits','scale_logits')}
    serial.update(input_scale=scale_dict(artifact['input_scale']),target_scales=[scale_dict(s) for s in artifact['target_scales']])
    bundle=dict(config=cfg,flow_scale=scale_dict(flow_scale),mass_artifact=serial,
        behavior=dict(scaler=scale_dict(behavior.scaler),centers=behavior.centers.tolist(),widths=behavior.widths.tolist(),support=behavior.support.tolist()),
        training=dict(normal_flows=len(roles),normal_groups=len(a.active),encoder_frozen_in_mass_training=True,
            detector_trained_from_initial_state=True,measurement_bundle_reused=bool(measurement_bundle),reuse_original_initial_state=reuse_initial,
            checkpoint='last_complete_epoch',validation_split=False,contrastive_enabled=False))
    write_json(root/'bundle.json',bundle)
    del model,arrays;torch.cuda.empty_cache()
    normal=Detector(root).score_capture(data,align_scores=False)
    dp=quantiles(normal['prototype_raw']);mp=quantiles(normal['mass_raw'])
    fusion=affine(normal['prototype_raw'],dp)+affine(normal['mass_raw'],mp)
    bundle['score_reference']=dict(prototype=dp,mass=mp,fusion_q99=float(np.quantile(fusion,.99)),
        threshold_comparison='score > training_q99',reference_only=True)
    write_json(root/'bundle.json',bundle)
    np.savez_compressed(root/'normal_scores.npz',**normal,final_score=fusion)
    write_json(root/'training_complete.json',bundle['training'])
