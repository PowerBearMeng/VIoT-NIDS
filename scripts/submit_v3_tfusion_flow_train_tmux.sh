#!/usr/bin/env bash
set -euo pipefail

session_name="v3_tfusion_flow_train"
project_dir="/home/mfh/Desktop/test/myModel"
output_dir="$project_dir/outputs/gotham_v3_tfusion_flow"
worker="$project_dir/scripts/run_v3_tfusion_flow_train.sh"
log_path="$output_dir/train.log"

if tmux has-session -t "$session_name" 2>/dev/null; then
    echo "tmux session already exists: $session_name"
    exit 1
fi

mkdir -p "$output_dir"
tmux new-session -d -s "$session_name" \
    "bash -lc 'set -o pipefail; \"$worker\" 2>&1 | tee \"$log_path\"'"
tmux set-option -t "$session_name" remain-on-exit on

echo "submitted tmux session: $session_name"
echo "output: $output_dir"
echo "log: $log_path"

