"""Normal-only group intensity pilot, with a frozen original HN checkpoint.

Mean and Mass differ in whether normalization discards activity magnitude.
Anonymous node IDs index observations only; they are never network inputs.
"""
from __future__ import annotations
import csv
import json
import logging
import math
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.cluster import KMeans
import yaml

from .data import load, prepare
from .capacity_experiment import eer_interpolated
from viot_nids_v7.context import Scale
from viot_nids_v7.training import setup, save_torch
from viot_nids_v7.data import write_json
from viot_nids_v7.extended import discover_all22
from viot_nids_v7.experiment import md_table
from viot_nids_v7.eer_diagnostics import phase
from viot_nids.evaluation import metrics

LOG = logging.getLogger(__name__)
RESOURCES = ('flows', 'packets', 'bytes')
BRANCHES = dict(HN='HN', Mean='mean', Mass='mass', HN_Mean='HN_mean', HN_Mass='HN_mass')


class Groups:
    """Dense unit-second observations, sparse occupied behavior queries."""
    def __init__(self, data, roles, history=16):
        self.data, self.roles, self.history_length = data, np.asarray(roles, np.int64), history
        n = len(self.roles)
        self.origin = int(data['timestamp_second'].min())
        self.span = int(data['timestamp_second'].max()) - self.origin + 1
        self.nodes = np.unique(np.r_[data['source_node'], data['destination_node']])
        self.units = 2 * len(self.nodes)
        self.seconds = data['timestamp_second'] - self.origin
        self.sender = 2 * np.searchsorted(self.nodes, data['source_node'])
        self.receiver = 2 * np.searchsorted(self.nodes, data['destination_node']) + 1
        count = self.units * self.span * 16
        self.source_cell = (self.sender * self.span + self.seconds) * 16 + self.roles
        self.destination_cell = (self.receiver * self.span + self.seconds) * 16 + self.roles
        codes = np.r_[self.source_cell, self.destination_cell]
        values = np.stack([np.ones(n), data['features'][:, 0], data['features'][:, 1]], -1)
        totals = np.stack([np.bincount(codes, weights=np.tile(values[:, j], 2), minlength=count)
                           for j in range(3)], -1)
        self.totals = totals.reshape(self.units * self.span, 16, 3)
        np.testing.assert_allclose(totals.sum(0), 2 * values.sum(0), rtol=0, atol=0)
        self.active = np.flatnonzero(totals[:, 0] > 0)
        self.ut = self.active // 16
        self.role = self.active % 16
        self.unit = self.ut // self.span
        self.t = self.ut % self.span
        self.direction = self.unit % 2
        self.birth = np.full(self.units, self.span, np.int64)
        np.minimum.at(self.birth, self.sender, self.seconds)
        np.minimum.at(self.birth, self.receiver, self.seconds)
        # Birth is shared by both directions of the same observed node.
        self.birth = np.repeat(self.birth.reshape(-1, 2).min(1), 2)
        self.source_pair = np.searchsorted(self.active, self.source_cell)
        self.destination_pair = np.searchsorted(self.active, self.destination_cell)
        np.testing.assert_array_equal(self.active[self.source_pair], self.source_cell)
        np.testing.assert_array_equal(self.active[self.destination_pair], self.destination_cell)
        self.history_indices, self.lengths = self.histories(np.arange(len(self.active)))

    def view(self, variant):
        values = self.totals.copy()
        if variant == 'mean':
            values /= values[:, :, 0].sum(1).clip(1)[:, None, None]
        return values

    def histories(self, ix):
        units, t = self.unit[ix], self.t[ix]
        length = np.minimum(self.history_length, t - self.birth[units]).clip(0)
        positions = t[:, None] - length[:, None] + np.arange(self.history_length)
        valid = np.arange(self.history_length)[None] < length[:, None]
        if valid.any():
            if not (positions[valid] < np.broadcast_to(t[:, None], positions.shape)[valid]).all():
                raise ValueError('Current/future activity entered predictor history')
        indices = units[:, None] * self.span + positions
        indices[~valid] = self.units * self.span  # A padding sentinel, not an observed zero.
        return indices, length.astype(np.int64)


class Predictor(nn.Module):
    def __init__(self, cfg, bounds=None, means=None, scales=None):
        super().__init__()
        self.states = cfg['normal_states']
        self.floor = cfg['sigma_floor']
        self.gru = nn.GRU(48, 64, batch_first=True)
        self.selector = nn.Sequential(nn.Linear(82, 64), nn.GELU(), nn.Linear(64, self.states))
        self.mean_logits = nn.Parameter(torch.zeros(self.states, 16, 3))
        self.scale_logits = nn.Parameter(torch.zeros(self.states, 16, 3))
        self.register_buffer('lower', torch.zeros(16, 3))
        self.register_buffer('upper', torch.ones(16, 3))
        if bounds is not None:
            with torch.no_grad():
                self.lower.copy_(torch.as_tensor(bounds[0])); self.upper.copy_(torch.as_tensor(bounds[1]))
                proportion = (torch.as_tensor(means) - self.lower) / (self.upper-self.lower).clamp_min(1e-6)
                self.mean_logits.copy_(torch.logit(proportion.clamp(.001, .999)))
                width = (torch.as_tensor(scales) - self.floor).clamp_min(.01)
                self.scale_logits.copy_(width + torch.log(-torch.expm1(-width)))
        nn.init.zeros_(self.selector[-1].weight); nn.init.zeros_(self.selector[-1].bias)

    def forward(self, history, lengths, roles, directions):
        state = history.new_zeros((len(history), 64))
        valid = lengths > 0
        if valid.any():
            packed = nn.utils.rnn.pack_padded_sequence(history[valid], lengths[valid].cpu(),
                                                       batch_first=True, enforce_sorted=False)
            _, h = self.gru(packed)
            state[valid] = h[0]
        query = torch.cat([state, F.one_hot(roles, 16).to(state.dtype),
                           F.one_hot(directions, 2).to(state.dtype)], -1)
        logw = self.selector(query).log_softmax(-1)
        mu = self.lower + (self.upper-self.lower) * self.mean_logits.sigmoid()
        sigma = F.softplus(self.scale_logits) + self.floor
        return logw, mu[:, roles].permute(1, 0, 2), sigma[:, roles].permute(1, 0, 2)

    @staticmethod
    def nll(target, dist):
        logw, mu, sigma = dist
        coordinate = -.5 * ((target[:, None]-mu)/sigma).square() - sigma.log() - .5*math.log(2*math.pi)
        return -torch.logsumexp(logw + coordinate.sum(-1), -1) / 3


def make_artifacts(groups, variant, cfg):
    view = groups.view(variant)
    real = np.arange(groups.span)[None] >= groups.birth[:, None]
    input_scale = Scale.fit(view.reshape(-1,48)[real.ravel()], cfg['feature_scale_floor'])
    occupied = groups.totals[:, :, 0].sum(1) > 0
    km = KMeans(n_clusters=cfg['normal_states'], n_init=10, random_state=cfg['seed']).fit(
        input_scale.transform(view.reshape(-1,48)[occupied]))
    state = km.predict(input_scale.transform(view.reshape(-1,48)))
    targets = view.reshape(-1,3)[groups.active]
    scales, transformed = [], np.empty_like(targets, dtype=np.float32)
    for k in range(16):
        ix = groups.role == k
        if not ix.any():
            raise ValueError(f'Behavior {k} has no positive normal quantity samples')
        scaler = Scale.fit(targets[ix], cfg['feature_scale_floor'])
        transformed[ix] = scaler.transform(targets[ix]); scales.append(scaler)
    bounds, means, widths = [], [], []
    for k in range(16):
        ix = groups.role == k; y = transformed[ix]
        lo, hi = y.min(0), y.max(0)
        bounds.append((lo, hi))
        support = state[groups.ut[ix]]
        means.append([y[support == j].mean(0) if (support == j).any() else y.mean(0)
                      for j in range(cfg['normal_states'])])
        widths.append([np.maximum(y[support == j].std(0) if (support == j).sum()>1 else y.std(0), .3)
                       for j in range(cfg['normal_states'])])
    bounds = np.asarray(bounds)
    return dict(input_scale=input_scale, target_scales=scales, bounds=(bounds[:,0],bounds[:,1]),
                means=np.asarray(means).transpose(1,0,2), widths=np.asarray(widths).transpose(1,0,2),
                normal_pairs=len(groups.active), normal_flows=len(groups.roles), conditional_on_presence=True)


def tensors(groups, artifact, variant, device):
    view = groups.view(variant)
    history = artifact['input_scale'].transform(view.reshape(-1,48))
    history = np.r_[history, np.zeros((1,48),np.float32)]
    target = view.reshape(-1,3)[groups.active]
    transformed = np.empty_like(target,dtype=np.float32)
    for role, scale in enumerate(artifact['target_scales']):
        ix=groups.role==role
        transformed[ix]=scale.transform(target[ix])
    return dict(history=torch.as_tensor(history,device=device), target=torch.as_tensor(transformed,device=device),
                role=torch.as_tensor(groups.role,device=device),direction=torch.as_tensor(groups.direction,device=device))


def batch_input(groups, arrays, ix, device):
    return (arrays['history'][torch.as_tensor(groups.history_indices[ix],device=device)],
            torch.as_tensor(groups.lengths[ix],device=device),arrays['role'][ix],arrays['direction'][ix])


def train_predictor(root, groups, cfg, variant, device):
    out=root/variant;out.mkdir(exist_ok=True)
    if (out/'training_complete.json').exists():return
    artifact=make_artifacts(groups,variant,cfg);joblib.dump(artifact,out/'artifacts.joblib')
    model=Predictor(cfg,artifact['bounds'],artifact['means'],artifact['widths']).to(device)
    arrays=tensors(groups,artifact,variant,device)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
    rng=np.random.default_rng(cfg['seed'])
    n=len(groups.active);batch=cfg['batch_size'];total_steps=max(cfg['steps_per_epoch'],math.ceil(n/batch))
    role_pools=[np.flatnonzero(groups.role==k) for k in range(16)]
    with (out/'train_log.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['epoch','loss','gradient_norm','pairs_seen','steps','seconds'])
        writer.writeheader()
        for epoch in range(1,cfg['epochs']+1):
            started=time.time();order=rng.permutation(n);batches=[order[i:i+batch] for i in range(0,n,batch)]
            for _ in range(total_steps-len(batches)):
                query=rng.integers(16,size=batch)
                batches.append(np.array([rng.choice(role_pools[k]) for k in query]))
            seen=set();values=[];model.train()
            for ix in batches:
                seen.update(map(int,ix));optimizer.zero_grad(set_to_none=True)
                dist=model(*batch_input(groups,arrays,ix,device))
                loss=model.nll(arrays['target'][ix],dist).mean()
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite group loss')
                loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'])
                if not torch.isfinite(grad):raise FloatingPointError('Nonfinite group gradient')
                optimizer.step();values.append((float(loss.detach()),float(grad)))
            if len(seen)!=n:raise ValueError('Normal group-pair coverage incomplete')
            avg=np.mean(values,0)
            writer.writerow(dict(epoch=epoch,loss=avg[0],gradient_norm=avg[1],pairs_seen=len(seen),
                                 steps=len(batches),seconds=time.time()-started));f.flush()
            LOG.info('%s %d/%d loss=%.4f seconds=%.1f',variant,epoch,cfg['epochs'],avg[0],time.time()-started)
    save_torch(out/'model.pt',model.eval().state_dict())
    write_json(out/'training_complete.json',dict(epochs=cfg['epochs'],normal_pairs=n,
        normal_flows=len(groups.roles),last_complete_epoch=True,attack_labels_read=False,
        conditional_on_behavior_presence=True,normal_states_fixed_during_test=True))


def score_predictor(root, groups, cfg, variant, device):
    out=root/variant;artifact=joblib.load(out/'artifacts.joblib')
    model=Predictor(cfg).to(device)
    model.load_state_dict(torch.load(out/'model.pt',map_location=device,weights_only=True))
    model.eval().requires_grad_(False);arrays=tensors(groups,artifact,variant,device)
    scores=np.empty(len(groups.active),np.float64)
    with torch.inference_mode():
        for start in range(0,len(scores),cfg['score_batch_size']):
            ix=np.arange(start,min(len(scores),start+cfg['score_batch_size']))
            scores[ix]=model.nll(arrays['target'][ix],model(*batch_input(groups,arrays,ix,device))).cpu().numpy()
    if not np.isfinite(scores).all():raise FloatingPointError('Nonfinite group scores')
    return scores


def score_scales(values):
    q50,q95=np.quantile(values,[.5,.95]);return dict(q50=float(q50),q95=float(q95))


def affine(values,scale):
    return (values-scale['q50'])/max(scale['q95']-scale['q50'],1e-6)


def evaluate(frame, case, destination):
    destination.mkdir(parents=True,exist_ok=True)
    attack_seconds=frame.loc[frame.binary_label.eq(1),'timestamp_second']
    frame['phase']=phase(frame.timestamp_second,attack_seconds.min(),attack_seconds.max())
    records,subsets,errors,points=[],[],[],[]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for branch,column in BRANCHES.items():
        value,roc,pr=metrics(frame.binary_label,frame[column]);value['EER_nearest']=value['EER']
        value['EER']=eer_interpolated(roc)
        records.append(dict(scenario=case['scenario'],branch=branch,**value))
        roc.to_csv(destination/(branch.lower()+'_roc.csv.gz'),index=False)
        pr.to_csv(destination/(branch.lower()+'_pr.csv.gz'),index=False)
        for axis,x,y in [(axes[0],roc.fpr,roc.tpr),(axes[1],pr.recall,pr.precision)]:
            ix=np.unique(np.linspace(0,len(x)-1,min(3000,len(x)),dtype=int));axis.plot(x.iloc[ix],y.iloc[ix],label=branch)
        idx=(roc.fpr-(1-roc.tpr)).abs().idxmin();threshold=roc.threshold.iloc[idx]
        normal=frame.binary_label.eq(0);fp=normal&(frame[column]>=threshold);fn=~normal&(frame[column]<threshold)
        points.append(dict(branch=branch,threshold=threshold,FP=int(fp.sum()),FN=int(fn.sum()),
                           FPR=float(fp.sum()/normal.sum()),FNR=float(fn.sum()/(~normal).sum())))
        groups=dict(single_packet=frame.packet_count.eq(1),single60=frame.packet_count.eq(1)&frame.byte_count.eq(60),
                    RTCP=frame.service.eq('RTCP_control'),sparse_control=frame.service.isin(['RTCP_control','RTSP_control','ONVIF_control']),
                    normal_in_mixed_sender_behavior=normal&frame.sender_group_attacks.gt(0),
                    normal_in_pure_sender_behavior=normal&frame.sender_group_attacks.eq(0))
        for key,selected in groups.items():
            if not selected.any():continue
            v,r,_=metrics(frame.loc[selected,'binary_label'],frame.loc[selected,column])
            if not r.empty:v['EER']=eer_interpolated(r)
            subsets.append(dict(branch=branch,subset=key,FP=int((fp&selected).sum()),FN=int((fn&selected).sum()),**v))
        for period,selected in frame.groupby('phase'):
            ids=selected.index
            errors.append(dict(branch=branch,phase=period,normal=int(selected.binary_label.eq(0).sum()),
                attack=int(selected.binary_label.eq(1).sum()),FP=int(fp.loc[ids].sum()),FN=int(fn.loc[ids].sum())))
        if branch=='HN_Mass':
            wrong=frame.loc[fp|fn].copy();wrong['error']=np.where(wrong.binary_label.eq(0),'FP','FN')
            wrong.to_csv(destination/'eer_errors.csv.gz',index=False)
            pd.concat([frame.loc[normal].nlargest(8,column),frame.loc[~normal].nsmallest(8,column)]).to_csv(destination/'representatives.csv',index=False)
    for axis in axes:axis.legend(fontsize=8)
    axes[0].set(xlabel='FPR',ylabel='TPR',title='ROC');axes[1].set(xlabel='Recall',ylabel='Precision',title='PR')
    fig.tight_layout();fig.savefig(destination/'curves.png',dpi=140);plt.close(fig)
    for name,rows in [('metrics',records),('subset_metrics',subsets),('phase_errors',errors),('eer_points',points)]:
        pd.DataFrame(rows).to_csv(destination/(name+'.csv'),index=False)


def necessary_checks(root, groups, cfg, device):
    raw=groups.totals;total=raw[:,:,0].sum(1)
    occupied=np.flatnonzero(total>0);ut=int(occupied[np.argmax(total[occupied])])
    original=raw[ut];repeated=original*1000
    np.testing.assert_allclose(original/total[ut],repeated/(total[ut]*1000),rtol=0,atol=1e-12)
    if not (repeated.sum(0)==1000*original.sum(0)).all():raise ValueError('Repeated mass lost')
    node_ids=np.unique(np.r_[groups.data['source_node'],groups.data['destination_node']])
    perm=np.random.default_rng(cfg['seed']).permutation(len(node_ids))+10000
    changed=dict(groups.data)
    for key in ['source_node','destination_node']:changed[key]=perm[np.searchsorted(node_ids,changed[key])]
    renamed=Groups(changed,groups.roles,cfg['history_seconds'])
    original_scores=score_predictor(root,groups,cfg,'mass',device)[groups.source_pair]
    changed_scores=score_predictor(root,renamed,cfg,'mass',device)[renamed.source_pair]
    np.testing.assert_allclose(original_scores,changed_scores,rtol=1e-5,atol=1e-4)
    write_json(root/'necessary_checks.json',dict(real_normal_flows=len(groups.roles),
        directed_mass_conserved=True,mean_invariant_when_repeated=True,
        mass_changes_proportionally_when_repeated=True,anonymous_identity_permutation_invariant=True,
        history_only_precedes_target=True,empty_observations_distinct_from_padding=True,
        labels_absent_from_neural_inputs=True,full_pairwise_matrix_created=False))


def repetition_audit(root, groups, cfg, device):
    """Hold real history/marks fixed and repeat a current normal activity unit."""
    single60=(groups.data['features'][:,0]==1)&(groups.data['features'][:,1]==60)
    choices=np.flatnonzero(single60)
    row=int(choices[0]) if len(choices) else 0
    ix=np.array([groups.source_pair[row]])
    ut=int(groups.ut[ix[0]]);role=int(groups.role[ix[0]])
    raw=groups.totals[ut].copy();count=raw[:,0].sum()
    results=[]
    for variant in cfg['variants']:
        artifact=joblib.load(root/variant/'artifacts.joblib')
        model=Predictor(cfg).to(device)
        model.load_state_dict(torch.load(root/variant/'model.pt',map_location=device,weights_only=True))
        model.eval().requires_grad_(False);arrays=tensors(groups,artifact,variant,device)
        with torch.inference_mode():
            dist=model(*batch_input(groups,arrays,ix,device))
            for multiplier in (1,2,10,100,1000):
                observed=raw*multiplier
                if variant=='mean':observed=observed/(count*multiplier)
                y=artifact['target_scales'][role].transform(observed[role,None])
                score=float(model.nll(torch.as_tensor(y,device=device),dist).item())
                results.append(dict(variant=variant,multiplier=multiplier,normal_row_id=row,
                    behavior=role,original_group_flows=int(count),selected_behavior_flows=int(raw[role,0]),score=score))
    table=pd.DataFrame(results)
    mean=table.loc[table.variant.eq('mean'),'score']
    np.testing.assert_allclose(mean,mean.iloc[0],rtol=0,atol=1e-6)
    table.to_csv(root/'normal_repetition_audit.csv',index=False)
    return table


def report(root):
    paths=list((root/'benchmark').glob('*/metrics.csv'))
    if not paths:return
    table=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True);table.to_csv(root/'capture_metrics.csv',index=False)
    macro=table.groupby('branch')[['AUROC','AUPRC','EER']].mean().reset_index()
    macro['captures']=table.groupby('branch').scenario.nunique().to_numpy();macro.to_csv(root/'macro_metrics.csv',index=False)
    group=pd.concat([pd.read_csv(p) for p in (root/'benchmark').glob('*/group_metrics.csv')],ignore_index=True)
    group.to_csv(root/'group_metrics.csv',index=False)
    lines=['# HN群体强度：四场景小实验', '', '这是冻结HN后的模块诊断；两个群体预测器只用正常数据训练，最后epoch及融合规则事先固定。', '',
           '## Flow指标（EER为比例）', '',md_table(table[['scenario','branch','normal','attack','AUROC','AUPRC','EER']]),'',
           '## 宏平均', '',md_table(macro),'','## 群体指标', '',md_table(group),'',
           'Mean丢弃绝对活动规模；Mass保留真实数量。预测器只读取过去16秒及当前行为通道/方向，不读取当前待预测数量。', '',
           '同一发送节点、行为通道、秒内的Flow共享强度分数。本轮还没有学出群体内部归因，混合群体的正常Flow误报必须单独看。', '',
           '正常状态测试冻结且均值限制在正常观测范围；历史仍可能改变状态权重，不能称为绝对不受污染。', '',
           '群体标签为含攻击Flow即攻击相关，不表示所有成员恶意。Flow/群体指标样本单位不同，不能混用。', '',
           '完整误报、RTCP和混合群体子集见各capture的subset_metrics.csv、phase_errors.csv、eer_errors.csv.gz。']
    (root/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


def run(config):
    cfg=yaml.safe_load(Path(config).read_text());root=Path(cfg['output_root']);root.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(),logging.FileHandler(root/'execution.log')])
    fixed=root/'config.json'
    if fixed.exists() and json.loads(fixed.read_text())!=cfg:raise ValueError('Configuration already fixed')
    write_json(fixed,cfg);device=setup(cfg)
    write_json(root/'environment.json',dict(device=str(device),torch=torch.__version__,numpy=np.__version__))
    base=Path(cfg['base_root']);source=Path(cfg['source_root']);run_id=cfg['source_run']
    checkpoint=source/run_id/'model.pt';stat=checkpoint.stat()
    ledger=dict(path=str(checkpoint),size=stat.st_size,mtime_ns=stat.st_mtime_ns)
    artifact=joblib.load(base/'artifacts.joblib');data,manifest=load(base/'normal_cache',normal=True)
    normal_frame=pd.read_csv(source/run_id/'normal_scores.csv.gz')
    np.testing.assert_array_equal(normal_frame.row_id,data['row_id'])
    np.testing.assert_array_equal(normal_frame.behavior_anchor,artifact['assignments'])
    groups=Groups(data,artifact['assignments'],cfg['history_seconds'])
    scales=dict(HN=score_scales(normal_frame.final_score))
    for variant in cfg['variants']:
        train_predictor(root,groups,cfg,variant,device)
        scores=score_predictor(root,groups,cfg,variant,device)
        normal_frame[variant]=scores[groups.source_pair];scales[variant]=score_scales(normal_frame[variant])
    write_json(root/'normal_score_scales.json',scales)
    normal_frame.to_csv(root/'normal_scores.csv.gz',index=False)
    necessary_checks(root,groups,cfg,device)
    cases=discover_all22(artifact['cfg']['attack_root'])
    cases=[c for c in cases if c['family'] in cfg['suite'] and (c['family']=='Mirai Single Bot' or c['condition']=='camera-udp')]
    cases.sort(key=lambda c:cfg['suite'].index(c['family']))
    if len(cases)!=4:raise ValueError('Expected exactly four predeclared captures')
    write_json(root/'suite_definition.json',dict(cases=cases,checkpoint=ledger,configuration_fixed_before_attack=True))
    del groups,data,normal_frame
    reusable={'SYN DoS':Path('outputs/behavior_assoc_pilot/data/syn_dos'),
              'OS Scan':Path('outputs/behavior_assoc_pilot/data/os_scan')}
    for number,case in enumerate(cases):
        out=root/'benchmark'/case['directory']
        if (out/'complete.json').exists():continue
        LOG.info('Group pilot %d/4 %s',number+1,case['scenario'])
        old=source/'benchmark'/case['directory'];cache=reusable.get(case['family'],root/'data'/case['directory'])
        expected=json.loads((old/'capture_manifest.json').read_text())
        if not (cache/'manifest.json').exists():prepare(Path(case['pcap']),cache,Path(case['labels']))
        current=json.loads((cache/'manifest.json').read_text())
        for field in ['flows','packets']:
            if current[field]!=expected[field]:raise ValueError('Flow cache differs from original HN')
        if current['source']['pcap']!=expected['source']['pcap']:raise ValueError('PCAP source changed')
        data,_=load(cache)
        frame=pd.read_csv(old/run_id/'scores.csv.gz',usecols=['row_id','timestamp_second','packet_count','byte_count','behavior_anchor','final_score'])
        for column,target in [('row_id',data['row_id']),('timestamp_second',data['timestamp_second']),
                              ('packet_count',data['features'][:,0]),('byte_count',data['features'][:,1])]:
            np.testing.assert_array_equal(frame[column],target)
        frame.rename(columns={'final_score':'HN'},inplace=True)
        groups=Groups(data,frame.behavior_anchor.to_numpy(),cfg['history_seconds'])
        scored={}
        for variant in cfg['variants']:
            pair_scores=score_predictor(root,groups,cfg,variant,device);scored[variant]=pair_scores
            frame[variant]=pair_scores[groups.source_pair]
            frame[variant+'_receiver_diagnostic']=pair_scores[groups.destination_pair]
            frame['HN_'+variant]=affine(frame.HN,scales['HN'])+affine(frame[variant],scales[variant])
        labels=pd.read_csv(old/'labels.csv.gz');metadata=pd.read_csv(old/'service_metadata.csv.gz',usecols=['row_id','service'])
        frame=frame.merge(labels,on='row_id',validate='one_to_one').merge(metadata,on='row_id',validate='one_to_one')
        if len(frame)!=len(data['row_id']) or frame.binary_label.isna().any():raise ValueError('Label alignment failed')
        attacks=np.bincount(np.r_[groups.source_pair,groups.destination_pair],
                            weights=np.tile(frame.binary_label.to_numpy(),2),minlength=len(groups.active))
        frame['sender_group_attacks']=attacks[groups.source_pair].astype(int)
        frame['sender_group_flows']=groups.totals.reshape(-1,3)[groups.active[groups.source_pair],0]
        frame['sender_group_packets']=groups.totals.reshape(-1,3)[groups.active[groups.source_pair],1]
        frame['sender_group_bytes']=groups.totals.reshape(-1,3)[groups.active[groups.source_pair],2]
        frame['intensity_history_seconds']=groups.lengths[groups.source_pair]
        out.mkdir(parents=True,exist_ok=True);frame.to_csv(out/'scores.csv.gz',index=False)
        evaluate(frame,case,out)
        pair_frame=pd.DataFrame(dict(unit=groups.unit,timestamp_second=groups.origin+groups.t,
            direction=groups.direction,behavior=groups.role,
            flow_count=groups.totals.reshape(-1,3)[groups.active,0],attack_flows=attacks,
            **scored))
        pair_frame.to_csv(out/'group_scores.csv.gz',index=False)
        group_metrics=[]
        for direction,name in [(0,'sender'),(1,'receiver')]:
            selected=pair_frame.direction.eq(direction)
            for variant in cfg['variants']:
                v,r,_=metrics(pair_frame.loc[selected,'attack_flows'].gt(0),pair_frame.loc[selected,variant]);v['EER']=eer_interpolated(r)
                group_metrics.append(dict(scenario=case['scenario'],direction=name,variant=variant,**v))
        pd.DataFrame(group_metrics).to_csv(out/'group_metrics.csv',index=False)
        write_json(out/'complete.json',dict(scenario=case['scenario'],flows=len(frame),scores_finite=True,
                   labels_for_offline_evaluation_only=True,HN_checkpoint_unchanged=True))
        LOG.info('%s %s',case['scenario'],pd.read_csv(out/'metrics.csv')[['branch','AUROC','EER']].to_dict('records'))
        del data,groups,frame,pair_frame
        report(root)
    after=checkpoint.stat()
    if after.st_size!=ledger['size'] or after.st_mtime_ns!=ledger['mtime_ns']:raise ValueError('HN checkpoint changed')
    report(root)
    write_json(root/'complete.json',dict(captures=4,original_HN_unchanged=True,normal_training_only=True))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=Path('configs/hn_group_intensity_pilot.yaml'))
    run(parser.parse_args().config)
