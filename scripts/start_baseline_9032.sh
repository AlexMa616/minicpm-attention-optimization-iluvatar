#!/bin/sh
set -eu

export CUDA_VISIBLE_DEVICES=1
export VLLM_PLUGINS=fl
export PYTHONPATH=/workspace/vllm-plugin-FL:/usr/local/lib/python3.12/site-packages
export ILUVATAR_USE_OPTIMIZED=0
export ILUVATAR_SPLIT_KV=0

exec vllm serve /workspace/MiniCPM5-2B \
    --port 9032 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --served-model-name minicpm-diag \
    --gpu-memory-utilization 0.85 \
    --max-model-len 131072
