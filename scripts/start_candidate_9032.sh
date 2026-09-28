#!/bin/sh
set -eu

export CUDA_VISIBLE_DEVICES=1
export VLLM_PLUGINS=fl
export PYTHONPATH=/workspace/split_kv_repair_20260928:/workspace/vllm-plugin-FL:/usr/local/lib/python3.12/site-packages
export ILUVATAR_USE_OPTIMIZED=1
export ILUVATAR_SPLIT_KV=1
export ILUVATAR_NUM_SPLITS="${ILUVATAR_NUM_SPLITS:-4}"
export ILUVATAR_BLOCK_M="${ILUVATAR_BLOCK_M:-64}"
export ILUVATAR_BLOCK_N="${ILUVATAR_BLOCK_N:-64}"
export ILUVATAR_NUM_WARPS="${ILUVATAR_NUM_WARPS:-4}"
export ILUVATAR_NUM_STAGES="${ILUVATAR_NUM_STAGES:-2}"

exec vllm serve /workspace/MiniCPM5-2B \
    --port 9032 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --served-model-name minicpm-diag \
    --gpu-memory-utilization 0.85 \
    --max-model-len 131072
