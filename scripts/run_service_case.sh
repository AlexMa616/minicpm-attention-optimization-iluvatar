#!/bin/sh
set -eu

input_len="$1"
output_len="$2"
concurrency="$3"
num_prompts="$4"

cd /workspace/vllm-plugin-FL
export VLLM_PLUGINS=fl
export PYTHONPATH=/workspace/vllm-plugin-FL:/usr/local/lib/python3.12/site-packages

exec python3 -u /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
    --served-model-name minicpm-diag \
    --model /workspace/MiniCPM5-2B \
    --port 9032 \
    --test-cases "[[${input_len},${output_len},${concurrency},${num_prompts}]]"
