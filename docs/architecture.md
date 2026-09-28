# Active Architecture

## Scope

The active implementation targets MiniCPM5-2B on Iluvatar BI-V150 and the
vLLM 0.24.0 paged-KV attention ABI. It is deliberately narrower than the
original FlashAttention-3-inspired design document: only code that is
implemented and testable on BI-V150 remains active.

## Execution Path

```text
vLLM AttentionImpl
  -> opt-in Iluvatar dispatcher
  -> flattened-query / paged-KV adapter
  -> Split-KV forward kernel
  -> online softmax reduction kernel
  -> output [total_query_tokens, num_query_heads, head_dim]
```

The forward grid is logically `(request, query_head, query_block, kv_split)`.
Because Triton exposes at most three grid axes, the implementation packs the
last two dimensions into one launch axis and decodes them with integer
division/modulo. Each program reads its request's query range from
`cu_seqlens_q` and maps every KV position through `block_table`; it does not
assume contiguous physical blocks or uniform query lengths.

## Online Softmax

For every query row and KV split, the forward kernel stores:

- `m`: the maximum logit;
- `l`: the exp-sum relative to `m`;
- `o`: the normalized partial output.

The reduction computes the global maximum, combines the split weights, and
normalizes the weighted partial outputs. Empty split tiles preserve the
running state and do not introduce NaNs.

## Supported Contract

The active kernel currently requires:

- causal attention;
- unquantized KV cache;
- no sliding window, ALiBi, sinks, soft cap, or tensor-descriptor path;
- `block_size` compatible with the paged-KV table;
- `num_splits` in `{2, 4, 8, 16}`.

Pure decode remains on native vLLM attention so the existing 3D segmented
decode path is not changed by the prefill/mixed experiment.

## Removed From Active Path

The earlier repository contained sketches labelled RoPE fusion, warp
specialization, ping-pong scheduling, and TMA. They are not active because the
current evidence does not establish a complete, correct BI-V150 implementation:

- RoPE fusion must be connected to the actual rotated-Q and KV-cache write
  semantics, otherwise it can double-rotate or use an unrotated Q;
- `num_stages` and barriers alone do not establish asynchronous
  producer/consumer warp specialization;
- ping-pong requires a verified device-side staging implementation;
- TMA is Hopper-specific and has no validated BI-V150 implementation.

The pre-audit code is retained as
`reference/experimental_unvalidated_attention.py`; it is not imported by the
active dispatcher or benchmark scripts.

## Validation Gates

1. Python syntax and Triton launch construction.
2. Randomized physical block-table correctness against an fp32 reference,
   including non-uniform query lengths and unaligned boundaries.
3. GPU 2 microbenchmark against native vLLM attention with identical inputs.
4. Isolated 9032 service A/B with profiler disabled.
5. Official 4k/16k benchmark and Level 3 accuracy, only after the short A/B
   passes. 9031 is never used for candidate traffic.

Microbenchmark latency or effective TFLOP/s is not an end-to-end service
throughput result.
