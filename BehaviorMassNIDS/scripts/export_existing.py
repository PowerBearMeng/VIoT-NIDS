"""Convert existing evaluated K8/K16 models to portable JSON/state dictionaries.

The old workspace is needed only for this conversion. No training or scoring
on attacks, and no model state change. Runtime does not load legacy joblib.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--k',type=int,choices=(8,16),default=8)
    args=parser.parse_args()
    sys.path.insert(0,str(args.workspace.resolve()))
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    import joblib
    import numpy as np
    import torch
    import yaml
    from behavior_mass_nids.model import scale_dict
    from behavior_mass_nids.io import write_json
    base=args.workspace/'outputs/graph_tcg_pilot'
    mass=args.workspace/f'outputs/mass_k_sensitivity_20261013/K{args.k}'
    scoring=args.workspace/f'outputs/mass_runtime_fusion_20261013/K{args.k}'
    artifact=joblib.load(mass/'artifacts.joblib');behavior=joblib.load(mass/'behavior.joblib')
    original=joblib.load(base/'artifacts.joblib')
    ref=json.loads((scoring/'distance_reference.json').read_text())
    cfg=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/k8.yaml').read_text());cfg['behavior_k']=args.k
    with np.load(scoring/'normal/scores.npz',allow_pickle=False) as f:
        d=f['prototype_distance'];m=f['mass_score']
    def affine(v,s):return (v-s['q50'])/max(s['q95']-s['q50'],1e-6)
    q99=float(np.quantile(affine(d,ref['normal_prototype_distance_scale'])+affine(m,ref['normal_mass_scale']),.99))
    serial={k:artifact[k].tolist() for k in ('lower','upper','mean_logits','scale_logits')}
    serial.update(input_scale=scale_dict(artifact['input_scale']),target_scales=[scale_dict(s) for s in artifact['target_scales']])
    bundle=dict(config=cfg,flow_scale=scale_dict(original['input_scale']),mass_artifact=serial,
        behavior=dict(scaler=scale_dict(behavior.scaler),centers=behavior.centers.tolist(),widths=behavior.widths.tolist(),support=behavior.support.tolist()),
        score_reference=dict(prototype=ref['normal_prototype_distance_scale'],mass=ref['normal_mass_scale'],fusion_q99=q99,reference_only=True),
        training=dict(normal_flows=12518,encoder_frozen_in_mass_training=True,contrastive_enabled=False,
            detector='BASE_GRU',history_seconds=192,normal_only=True,checkpoint='last_complete_epoch',
            exported_without_training=True,reported_results_scope='five captures; single seed'))
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'bundle.json').exists():raise FileExistsError('Use a fresh bundle output directory')
    for source,name in ((base/'pretrain.pt','flow_encoder.pt'),(mass/'model.pt','mass_model.pt'),(mass/'initial_model.pt','initial_mass.pt')):
        state=torch.load(source,map_location='cpu',weights_only=True)
        torch.save(state,args.out/name)
    write_json(args.out/'bundle.json',bundle)
    print(json.dumps({'K':args.k,'bundle':str(args.out),'fusion_training_q99':q99,'new_training':False}))


if __name__=='__main__':main()
