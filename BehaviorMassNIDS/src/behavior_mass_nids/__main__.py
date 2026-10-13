"""Portable one-second extraction, normal training, frozen scoring, evaluation."""
from __future__ import annotations
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from .io import prepare_pcap, load_capture
from .model import Detector


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--pcap',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--packet-labels',type=Path)
    p=sub.add_parser('train');p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--config',type=Path,default=Path('configs/k8.yaml'));p.add_argument('--measurement-bundle',type=Path);p.add_argument('--reuse-initial-mass',action='store_true')
    p=sub.add_parser('score');p.add_argument('--model',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',choices=('cpu','cuda'))
    p=sub.add_parser('evaluate');p.add_argument('--scores',type=Path,required=True);p.add_argument('--labels',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if args.command=='prepare':prepare_pcap(args.pcap,args.out,args.packet_labels)
    elif args.command=='train':
        from .training import train
        cfg=yaml.safe_load(args.config.read_text());train(args.cache,args.out,cfg,args.measurement_bundle,args.reuse_initial_mass)
    elif args.command=='score':
        result=Detector(args.model,args.device).score_capture(load_capture(args.cache))
        args.out.parent.mkdir(parents=True,exist_ok=True)
        if args.out.suffix=='.npz':np.savez_compressed(args.out,**result)
        else:pd.DataFrame(result).to_csv(args.out,index=False)
    else:
        from .evaluation import evaluate
        evaluate(args.scores,args.labels,args.out)


if __name__=='__main__':main()
