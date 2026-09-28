#!/bin/bash
# Benchmark script for attention optimization testing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TEST_DIR="$PROJECT_DIR/tests"
RESULT_DIR="$PROJECT_DIR/experiments/results"

# Default parameters
DEVICE="${DEVICE:-cuda:0}"
REPEATS="${REPEATS:-10}"
WARMUP="${WARMUP:-3}"
ITERATIONS="${ITERATIONS:-50}"

echo "==================================="
echo "MiniCPM Attention Benchmark"
echo "==================================="
echo "Device: $DEVICE"
echo "Repeats: $REPEATS"
echo "Warmup: $WARMUP"
echo "Iterations: $ITERATIONS"
echo ""

mkdir -p "$RESULT_DIR"

cd "$TEST_DIR"

# Test 1: Baseline (optimizations disabled)
echo "Running baseline test..."
ILUVATAR_USE_OPTIMIZED=0 \
python attention_harness.py \
    --config configs/attention_test.json \
    --device "$DEVICE" \
    --result "$RESULT_DIR/baseline.jsonl" \
    --repeats "$REPEATS" \
    --warmup "$WARMUP" \
    --iterations "$ITERATIONS"

# Test 2: Split-KV optimization
echo ""
echo "Running Split-KV optimization test..."
ILUVATAR_USE_OPTIMIZED=1 \
ILUVATAR_SPLIT_KV=1 \
VLLM_ILUVATAR_ATTN_SPLIT_KV_MIXED=1 \
python attention_harness.py \
    --config configs/attention_test.json \
    --device "$DEVICE" \
    --result "$RESULT_DIR/split_kv.jsonl" \
    --split-kv-mixed \
    --split-kv-segments 4 \
    --repeats "$REPEATS" \
    --warmup "$WARMUP" \
    --iterations "$ITERATIONS"

echo ""
echo "Running Split-KV configuration comparison..."
ILUVATAR_USE_OPTIMIZED=1 \
ILUVATAR_SPLIT_KV=1 \
ILUVATAR_NUM_SPLITS=8 \
VLLM_ILUVATAR_ATTN_SPLIT_KV_MIXED=1 \
python attention_harness.py \
    --config configs/attention_test.json \
    --device "$DEVICE" \
    --result "$RESULT_DIR/split_kv_8.jsonl" \
    --split-kv-mixed \
    --split-kv-segments 4 \
    --repeats "$REPEATS" \
    --warmup "$WARMUP" \
    --iterations "$ITERATIONS"

echo ""
echo "==================================="
echo "✅ Benchmark complete!"
echo "Results saved to: $RESULT_DIR"
echo ""
echo "To analyze results:"
echo "  cd $PROJECT_DIR/experiments"
echo "  # Use your preferred analysis tool"
echo "==================================="
