# Stage 1 Attention Scan

This directory contains the reproducible scan specification and result schema
for the Iluvatar 2D prefill attention work. It is diagnostic material only;
it is not imported by the official service.

## Required result fields

Each row must include:

- `shape`: `pure_prefill` or `mixed`
- `prefix_tokens`, `query_tokens`, `decode_sequences`
- `block_m`, `block_q`, `tile_size`, `num_warps`, `num_stages`
- `compile_seconds`, `warmup_count`, `timed_iterations`
- `median_ms`, `batch_mean_samples_ms`, `max_batch_mean_ms`, `cuda_ms`,
  `wall_median_ms`, `timing_source`
- `effective_tflops`, `baseline_median_ms`, `speedup_vs_vllm`
- `baseline_effective_tflops`
- For mixed cases: `decode_only_ms`, `decode_cost_proxy_ratio`,
  `decode_cost_proxy_note`, `baseline_decode_only_ms`,
  `decode_speedup_vs_vllm`, `decode_segments_tested`
- `max_abs_error`, `mean_abs_error`, `status`, `failure_reason`

The causal attention FLOP estimate counts both QK and PV multiply-adds, with
two FLOPs per multiply-add, and counts only causally visible query/key pairs:

`4 * (query_len * prefix_len + query_len * (query_len + 1) / 2) * num_query_heads * head_size`

This is a math-work estimate, not measured hardware FLOPs. Use the same
estimate for baseline and candidate rows. Timing repeats are per-repeat means
over batched CUDA-event samples; with only five repeats, do not label an
empirical value as P95. The decode probe uses the tested mixed-path segment
count and compares it with the native 16-segment 3D path. Its ratio to full
mixed-call time is not a measured decoder tail fraction and must not be
presented as one. The probe is also not the service pure-decode default, which
remains 16 segments. Do not report a candidate as an optimization until it
passes the fp32 SDPA/paged-KV numerical comparison and separate service-level
decode regression checks.

Kernel time uses batched CUDA-event timing; wall time is also recorded to
expose host-launch overhead. Pure decode probes provide the same segmented
workspace shape as the vendor backend so they select its 3D decode path.

## Execution order

1. Run numerical correctness on one small shape.
2. Scan pure prefill positions `0..14k` in 2k increments.
3. Scan mixed shapes at `0/8k/14k`.
4. Retain the Pareto candidates and only then run a 9032 service A/B.

`scan_split_segments.sh` is the bounded GPU-2 screen for the current
128/16 mixed candidate. It compares 2/4/8/16 KV segments at 8k and 14k
prefixes and writes one JSONL result per case under `/workspace/logs`. Each
row records the requested segment count so results remain self-describing.
