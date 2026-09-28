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
query-block and split dimensions into the three launch axes directly. The
query-block list is compacted per request, so a one-token decode request does
not inherit the prefill request's query-block count. Each program reads its
request's query range from
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
- `BLOCK_M` and `BLOCK_N` are multiples of 16; the active implementation has
  no independent `BLOCK_Q` parameter.

Pure decode remains on native vLLM attention so the existing 3D segmented
decode path is not changed by the prefill/mixed experiment.

## RoPE Prototype Boundary

`kernels/triton_rope_kv_cache.py` rotates Q in place and rotates K while
writing K/V to the NHD paged cache in one launch. It supports unquantized
BF16/FP16, NeoX/interleaved rotation, partial rotary dimensions and masked
slots. Four GPU 2 cases passed against an FP32 reference after computing
the rotation in FP32 before storing BF16 outputs.

This is a standalone prototype, not an active service optimization. MiniCPM
already rotates Q/K before attention and stores the rotated K in the cache.
The correct service integration point is vLLM's
`RopeKVCacheFusionPass`, but the pinned vLLM 0.24.0 configuration disables
that pass on non-ROCm platforms. The Iluvatar plugin does not override that
policy. A plugin-only RoPE flag or attention-side re-rotation would either be
unreachable or incorrect. No RoPE performance gain is claimed.

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

Hardware warp specialization and TMA are not implemented on BI-V150. Triton
`num_warps`/`num_stages` only configure a launch; their presence does not
establish a producer/consumer warp pipeline. A future software-pipelined
kernel would need generated-code evidence and an isolated device benchmark.

The vLLM integration must also remove the older untracked operator drafts
from the vendor directory. Only `impl/attention.py`,
`impl/ops/triton_split_kv_paged.py`, and
`impl/ops/triton_unified_attention_optimized.py` belong to the active
integration path.

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

The repaired mixed 8k Split-KV probe measured 75.89 ms versus 44.02 ms for
native attention (0.58x). This candidate does not pass the microbenchmark
gate. Earlier 2.15-2.43x results came from a superseded implementation and
must not be reported as the repaired kernel's performance.
