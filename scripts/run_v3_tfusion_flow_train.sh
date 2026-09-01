#!/usr/bin/env bash
set -euo pipefail

project_dir="/home/mfh/Desktop/test/myModel"
python_bin="/home/mfh/miniconda3/envs/wxy/bin/python"
config_path="configs/gotham_v3_train.yaml"
output_dir="$project_dir/outputs/gotham_v3_tfusion_flow"

mkdir -p "$output_dir"
cd "$project_dir"

"$python_bin" run_pipeline.py --config "$config_path" --mode prepare
"$python_bin" run_pipeline.py --config "$config_path" --mode train --device cuda

