#!/usr/bin/env bash
set -euo pipefail

project_dir="/home/mfh/Desktop/test/myModel"
python_bin="/home/mfh/miniconda3/envs/wxy/bin/python"
config="configs/gotham_v45_local_pair.yaml"
output_dir="$project_dir/outputs/gotham_v45_local_pair"

mkdir -p "$output_dir"
cd "$project_dir"

# Reuse the canonical V4.5 Local embedding artifact and retrain pair-only
# Sparse Context before normal-only calibration and manifest evaluation.
"$python_bin" -u train_sparse_context.py --config "$config" --device cuda
"$python_bin" -u calibrate.py --config "$config" --device cuda
"$python_bin" -u evaluate_gotham_manifest.py --config "$config" --device cuda
