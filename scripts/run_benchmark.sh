#!/bin/bash
# Benchmark script for attention optimization testing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TEST_DIR="$PROJECT_DIR/tests"
RESULT_DIR="$PROJECT_DIR/experiments/results"

# Default parameters
DEVICE="${DEVICE:-cuda:0}"
REPEATS="${REPEATS:-3}"
WARMUP="${WARMUP:-3}"
ITERATIONS="${ITERATIONS:-10}"

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

# Each run records candidate and native vLLM timing on identical inputs.
echo ""
echo "Running repaired Split-KV optimization test..."
python attention_harness.py \
    --config configs/long_context.json \
    --device "$DEVICE" \
    --result "$RESULT_DIR/split_kv.jsonl" \
    --split-kv-mixed \
    --split-kv-segments 4 \
    --repeats "$REPEATS" \
    --warmup "$WARMUP" \
    --iterations "$ITERATIONS"

echo ""
echo "Running Split-KV configuration comparison..."
python attention_harness.py \
    --config configs/long_context.json \
    --device "$DEVICE" \
    --result "$RESULT_DIR/split_kv_8.jsonl" \
    --split-kv-mixed \
    --split-kv-segments 8 \
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
