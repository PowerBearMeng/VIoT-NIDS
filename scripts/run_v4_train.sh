#!/usr/bin/env bash
set -euo pipefail

project_dir="/home/mfh/Desktop/test/myModel"
python_bin="/home/mfh/miniconda3/envs/wxy/bin/python"

cd "$project_dir"
"$python_bin" run_pipeline.py \
    --config configs/gotham_v4_train.yaml \
    --mode prepare
"$python_bin" run_pipeline.py \
    --config configs/gotham_v4_train.yaml \
    --mode train \
    --device cuda

