#!/bin/sh
set -eu

cd /workspace/vllm-plugin-FL
export VLLM_PLUGINS=fl
export PYTHONPATH=/workspace/vllm-plugin-FL:/usr/local/lib/python3.12/site-packages

run_case() {
    input_len="$1"
    python3 -u /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
        --served-model-name minicpm-diag \
        --model /workspace/MiniCPM5-2B \
        --port 9032 \
        --test-cases "[[${input_len},256,16,16]]"
}

run_case 4096
run_case 16384
