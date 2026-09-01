#!/usr/bin/env python3
"""Orchestrate the complete three-stage PCAP-to-score workflow."""

from __future__ import annotations

import argparse

from calibrate import calibrate
from evaluate_flow import evaluate
from extract_embeddings import extract
from fit_prototypes import fit
from prepare_data import prepare
from train_context import train as train_context
from train_entity import train as train_entity
from train_flow import train as train_flow
from train_neural_context import train as train_neural_context
from train_spatial import train as train_spatial
from utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mode", choices=["all", "prepare", "train", "evaluate"], default="all")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.mode in {"all", "prepare"}:
        prepare(config)
    if args.mode in {"all", "train"}:
        train_flow(config, args.device)
        extract(config, args.device)
        context_mode = str(config.get("context_model", {}).get("mode", "legacy_spatial"))
        if context_mode == "neural_intensity":
            train_neural_context(config, args.device)
        else:
            fit(config)
            train_entity(config, args.device)
        if context_mode == "behavior_composition":
            train_context(config)
            if bool(config["context_model"].get("legacy_spatial_ablation", False)):
                train_spatial(config, args.device)
        elif context_mode == "legacy_spatial":
            train_spatial(config, args.device)
        calibrate(config, args.device)
    if args.mode in {"all", "evaluate"}:
        evaluate(config, "test", args.device)


if __name__ == "__main__":
    main()
