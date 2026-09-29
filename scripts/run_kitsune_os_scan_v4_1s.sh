#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/mfh/Desktop/test/myModel"
python_bin="/home/mfh/miniconda3/envs/wxy/bin/python"
config="configs/kitsune_os_scan_v4_1s.yaml"
stamp="${1:-$(date +%Y%m%d_%H%M%S)}"
log_root="$project_dir/outputs/kitsune_os_scan_v4_1s/runs/$stamp"
mkdir -p "$log_root"
printf '%s\n' "$$" >"$log_root/runner.pid"

run_stage() {
  local name="$1"
  shift
  printf '[%s] START %s\n' "$(date --iso-8601=seconds)" "$name" | tee -a "$log_root/status.log"
  if "$@" >>"$log_root/run.log" 2>&1; then
    printf '[%s] DONE %s\n' "$(date --iso-8601=seconds)" "$name" | tee -a "$log_root/status.log"
  else
    local code=$?
    printf '[%s] FAILED %s exit=%s\n' "$(date --iso-8601=seconds)" "$name" "$code" | tee -a "$log_root/status.log"
    exit "$code"
  fi
}

cd "$project_dir"
run_stage labels "$python_bin" scripts/convert_kitsune_packet_labels.py \
  --input "/home/mfh/Desktop/test/external/kitsune_data/OS Scan/OS_Scan_labels.csv" \
  --output "$project_dir/data/labels/kitsune_os_scan_packet_labels.csv"
run_stage prepare_train "$python_bin" run_pipeline.py --config "$config" --mode prepare
run_stage train "$python_bin" run_pipeline.py --config "$config" --mode train --device cuda
run_stage test "$python_bin" evaluate_gotham_manifest.py --config "$config" --device cuda --keep-intermediates

scores="$project_dir/outputs/kitsune_os_scan_v4_1s/evaluation/OS_Scan/flow_scores.csv"
metrics_dir="$project_dir/outputs/kitsune_os_scan_v4_1s/evaluation/OS_Scan/common_metrics"
mkdir -p "$metrics_dir"
run_stage common_metrics "$python_bin" /home/mfh/Desktop/test/scripts/evaluate_kitsune_data.py \
  --scores "$scores" --unit flow --score-column final_anomaly --label-column label \
  --fpr-targets 0.01,0.001,0.0001 \
  --output "$metrics_dir/metrics.json" --plot "$metrics_dir/roc_pr_inset.png" \
  --attack-sample-rate 0.05 --sample-seed 20260829 \
  --sampled-output "$metrics_dir/metrics_attack_sample_5pct.json" \
  --sampled-plot "$metrics_dir/roc_pr_attack_sample_5pct.png"

date --iso-8601=seconds >"$log_root/finished_at.txt"
