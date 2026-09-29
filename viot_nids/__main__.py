"""python -m viot_nids {extract,train,score,evaluate,inspect}."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .benchmark import benchmark
from .evaluation import evaluate
from .features import extract
from .pipeline import inspect, score, train


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m viot_nids")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("extract")
    p.add_argument("--pcap", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--packet-labels", type=Path)
    p.add_argument("--labels-out", type=Path)
    p = sub.add_parser("train")
    p.add_argument("--flows", required=True, type=Path)
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/v5.yaml")
    p.add_argument("--device", default="cpu")
    p = sub.add_parser("score")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--flows", type=Path)
    group.add_argument("--pcap", type=Path)
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", default="cpu")
    p = sub.add_parser("evaluate")
    p.add_argument("--scores", required=True, type=Path)
    p.add_argument("--labels", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--attack-name")
    p = sub.add_parser("inspect")
    p.add_argument("--model-dir", required=True, type=Path)
    p = sub.add_parser("benchmark")
    p.add_argument("--attack-root", required=True, type=Path)
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--family")
    p.add_argument("--only")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "extract":
        result = extract(args.pcap, args.out, args.labels_out, args.packet_labels)
    elif args.command == "train":
        result = train(args.flows, args.model_dir, args.config, args.device)
    elif args.command == "score":
        flows = args.flows
        if args.pcap:
            flows = args.out.with_name(args.out.stem + ".flows.csv")
            extract(args.pcap, flows)
        result = {"scored_flows": len(score(flows, args.model_dir, args.out, args.device)), "scores": str(args.out)}
        if args.pcap:
            flows.unlink()
    elif args.command == "evaluate":
        result = evaluate(args.scores, args.labels, args.out, args.attack_name)
    elif args.command == "benchmark":
        result = benchmark(args.attack_root, args.model_dir, args.out, args.device, args.family, args.only)
    else:
        result = inspect(args.model_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
