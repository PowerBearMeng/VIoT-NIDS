"""python -m viot_nids_v6 {train,score,benchmark}."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .benchmark import benchmark, rescore_benchmark
from .pipeline import rescore_components, score, train


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m viot_nids_v6")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train")
    p.add_argument("--benign-flows", type=Path, required=True)
    p.add_argument("--base-model", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/v6.yaml"))
    p.add_argument("--device", default="cpu")
    p = sub.add_parser("score")
    p.add_argument("--flows", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--base-scores", type=Path)
    p.add_argument("--device", default="cpu")
    p = sub.add_parser("benchmark")
    p.add_argument("--attack-root", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--v5-cache", type=Path)
    p.add_argument("--family")
    p.add_argument("--only")
    p.add_argument("--device", default="cpu")
    p = sub.add_parser("rescore")
    p.add_argument("--previous-scores", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("rescore-benchmark")
    p.add_argument("--previous-root", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "train":
        result = train(args.benign_flows, args.base_model, args.model_dir, args.config, args.device)
    elif args.command == "score":
        result = {"scored": len(score(args.flows, args.model_dir, args.out, args.device, args.base_scores)),
                  "scores": str(args.out)}
    elif args.command == "benchmark":
        result = benchmark(args.attack_root, args.model_dir, args.out, args.v5_cache,
                           args.family, args.only, args.device)
    elif args.command == "rescore":
        result = {"scored": len(rescore_components(args.previous_scores, args.model_dir, args.out)),
                  "scores": str(args.out)}
    else:
        result = rescore_benchmark(args.previous_root, args.model_dir, args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
