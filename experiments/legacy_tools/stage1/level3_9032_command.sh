#!/usr/bin/env bash
set -euo pipefail

# Diagnostic copy of the official Level 3 evaluation. It deliberately targets
# 9032 so the official 9031 service is not changed or consumed by this run.
OUT="/workspace/evalscope-datasets/level3_diag_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
exec evalscope eval \
  --model minicpm-diag \
  --api-url http://127.0.0.1:9032/v1/chat/completions \
  --api-key EMPTY \
  --eval-type openai_api \
  --datasets math_500 \
  --dataset-args '{"math_500": {"dataset_id": "/workspace/evalscope-datasets/math_500", "subset_list": ["Level 3"]}}' \
  --eval-batch-size 8 \
  --timeout 3600 \
  --generation-config '{"temperature": 1.0, "top_p": 0.95, "max_tokens": 32768}' \
  --work-dir "$OUT" \
  --ignore-errors
