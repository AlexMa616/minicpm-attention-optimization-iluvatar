# Stage 1 Attention Optimization Harness

This directory is reserved for diagnostics and reference material. It must not
be imported by the official service until numerical and service-level checks
are complete.

## Current baseline

- Platform: Iluvatar BI-V150
- Model attention shape: Q heads 16, KV heads 2, head size 128
- GQA ratio: 8
- Observed 2D launch: grid `[1025, 2, 1]`, block `[256, 1, 1]`
- Observed attention CUDA time in W1: 13.563 s across 336 launches

## Required checks before service integration

1. Copy the installed vLLM 0.24.0 unified-attention implementation into the
   Iluvatar vendor namespace without changing semantics.
2. Add an Iluvatar `AttentionBackend` class and keep unsupported combinations
   delegated to the original Triton backend.
3. Implement a reference comparison for BF16 paged KV, GQA=8, block size 16,
   causal attention, chunk boundaries, and mixed batches.
4. Scan only 2D parameters first. Keep the 3D decode path unchanged.
5. Use two harness families:
   - pure prefill: 8 x 2048 queries at prefix positions 0/2/4/6/8/10/12/14k;
   - mixed: 1 x 2048 prefill plus about 30 x 1 decode requests at 0/8/14k
     prefixes, with decoder share reported separately.
6. Scan `BLOCK_Q` candidates `{4, 8, 16, 32, 64, 128}` and the corresponding
   `BLOCK_M` candidates `{32, 64, 128}` only when the hardware resources allow
   compilation. Warm up 5 times and time 20 steady-state iterations.
7. Record kernel time, effective attention TFLOP/s, max absolute error, and
   decoder-tail ratio before any service-level A/B. The preferred entry gate is
   `>=15 TFLOP/s` or `>=4x` over the observed baseline, with mixed decoder tail
   below 20%; these are gates, not assumed results.

## Required call path in the report

`vllm serve` -> `vllm_fl.worker.worker:FLWorker.init_cache_engine` ->
`vllm_fl.dispatch.backends.loader` -> `iluvatar.yaml` ->
`vendor/iluvatar/iluvatar.py:IluvatarBackend.attention_backend()` ->
`IluvatarAttentionBackend` ->
`triton_unified_attention_optimized.py`.

The final report must show the old/new attention CUDA time and effective
TFLOP/s, so the change is demonstrably a kernel implementation rather than a
backend-selection switch. The final YAML must directly express the Iluvatar
path; do not leave a later revert commit that restores flagos-first priority.

## Qualification gates before service A/B

- Run the Level 3 accuracy baseline twice before parameter tuning begins and
  record per-question top-3 log-probability evidence when available.
- Confirm numerical agreement against an fp32 SDPA reference, including paged
  KV, GQA=8, block size 16, chunk boundaries, and mixed batches.
- Compare decode throughput with `FULL_DECODE_ONLY` cudagraphs enabled and
  disabled on 9032. This is a diagnostic only and must not alter 9031.

## Do not use as a submission yet

- Changing only `iluvatar.yaml` priority.
- Setting an environment variable to select an existing attention backend.
- Enlarging `BLOCK_Q` based only on pure-prefill measurements.

## Handoff status (2026-09-27)

- The installed vLLM 0.24.0 source has been exported to `/tmp/minicpm-vllm024/`.
- The remote diagnostic service has 621 parsed iterations: 32 context-only,
  381 generation-only, and 208 mixed; maximum observed generation concurrency
  was 8, so this is not a 64-request decode regression result.
- `vllm._C` is absent in the container. Treat this as an environment finding,
  not as proof of an attention bottleneck.
- No competition source files have been changed. The next implementation step
  must add a compatibility-preserving Iluvatar wrapper based on the installed
  vLLM source, then run numerical checks before any 9032 service A/B.
