#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace/vllm-plugin-FL}
OUT=${1:-/workspace/logs/native_2d_adaptive_audit_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT"

run_case() {
  local gpu=$1
  local shape=$2
  shift 2
  CUDA_VISIBLE_DEVICES="$gpu" python3 "$ROOT/tools/native_2d_harness.py" \
    --device cuda:0 --shape "$shape" --result "$OUT/gpu${gpu}_${shape}.json" \
    --warmup 3 --iterations 12 "$@" \
    >"$OUT/gpu${gpu}_${shape}.log" 2>&1
}

# GPU 2-5 are independent. GPU 0/1 remain reserved for the two services.
run_case 2 early --block-m 64 128 --tile 16 --warps 8 --stages 1 2 \
  --pipeline-stages 1 --scalar-block-lookup 0 1 &
p2=$!
run_case 3 pure --block-m 64 128 --tile 16 --warps 8 --stages 1 2 \
  --pipeline-stages 1 --scalar-block-lookup 0 &
p3=$!
run_case 4 mixed --block-m 64 128 --tile 16 --warps 4 8 --stages 1 2 \
  --pipeline-stages 1 --scalar-block-lookup 0 &
p4=$!
run_case 5 mixed-order --block-m 64 128 --tile 16 --warps 8 --stages 1 2 \
  --pipeline-stages 1 --scalar-block-lookup 0 &
p5=$!

rc=0
for pid in "$p2" "$p3" "$p4" "$p5"; do
  wait "$pid" || rc=1
done
printf '%s\n' "$rc" >"$OUT/exit"
exit "$rc"
