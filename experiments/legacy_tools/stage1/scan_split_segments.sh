#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/stage1_patch
HARNESS="$ROOT/tools/stage1/attention_harness.py"
CONFIG="$ROOT/tools/stage1/scan_config.json"
RESULT_DIR=${RESULT_DIR:-/workspace/logs/split_kv_segment_scan_repeat5_decodeprobe}

export PYTHONPATH="$ROOT/vllm-plugin-FL${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$RESULT_DIR"

for segments in 2 4 8 16; do
  for prefix in 8192 14336; do
    result="$RESULT_DIR/segments_${segments}_prefix_${prefix}.jsonl"
    python3 "$HARNESS" \
      --config "$CONFIG" \
      --device cuda:2 \
      --shape mixed \
      --prefix-token "$prefix" \
      --block-m 128 \
      --block-q 16 \
      --tile 16 \
      --warmup 2 \
      --iterations 10 \
      --repeats 5 \
      --split-kv-mixed \
      --split-kv-segments "$segments" \
      --result "$result"
  done
done
