#!/usr/bin/env bash
set -euo pipefail

project_dir="/home/mfh/Desktop/test/myModel"
python_bin="/home/mfh/miniconda3/envs/wxy/bin/python"

cd "$project_dir"
"$python_bin" evaluate_gotham_manifest.py \
    --config configs/gotham_v4_train.yaml \
    --device cuda

