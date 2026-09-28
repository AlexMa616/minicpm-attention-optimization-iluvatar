# MiniCPM Attention Optimization for Iluvatar BI-V150

This repository contains the validated Split-KV attention work for
MiniCPM5-2B on Iluvatar BI-V150. It is a research and integration artifact;
numbers are not official competition results until they are reproduced with
the official service command and workload.

## Active Implementation

The active path contains only optimizations implemented against the vLLM
paged-KV ABI:

- Split-KV forward tiling over the KV axis;
- online softmax statistics `(m, l)` per split;
- numerically stable reduction over split outputs;
- flattened variable-length query addressing through `cu_seqlens_q`;
- physical block lookup for every KV lane through `block_table`;
- GQA mapping for MiniCPM5-2B (`16` query heads, `2` KV heads).

The active kernel is opt-in. Unsupported paths and pure decode remain on
vLLM's native unified attention implementation.

## Explicitly Not Active

The previous RoPE-fusion, warp-specialization, ping-pong, and TMA sketches are
kept under `reference/experimental_unvalidated_attention.py` for historical
comparison only. They are not imported by the active dispatcher, not enabled
by the benchmark scripts, and not included in performance claims:

- RoPE fusion needs integration with the actual Q/RoPE and KV-cache write
  path before it can be correct;
- Triton `num_stages` is not proof of warp specialization or asynchronous
  producer/consumer execution on BI-V150;
- TMA is Hopper-specific and BI-V150 is not a Hopper target;
- ping-pong scheduling has no independently validated device-side staging
  implementation in this repository.

## Repository Layout

```text
kernels/
  triton_split_kv_paged.py              # active paged Split-KV kernels
  triton_unified_attention_optimized.py # compatibility entry point
  attention.py                          # opt-in standalone dispatcher
reference/
  experimental_unvalidated_attention.py # archived pre-audit design
tests/
  attention_harness.py                  # correctness and timing harness
scripts/
  install_to_vllm.sh                    # copies active kernel files only
experiments/
  experiment-log.md                     # evidence and decisions
```

## Environment Variables

```bash
export ILUVATAR_USE_OPTIMIZED=1
export ILUVATAR_SPLIT_KV=1
export ILUVATAR_NUM_SPLITS=4       # 2, 4, 8, or 16
export ILUVATAR_BLOCK_M=64
export ILUVATAR_BLOCK_N=64
export ILUVATAR_NUM_WARPS=4
export ILUVATAR_NUM_STAGES=2
```

`BLOCK_M` is the query-row tile used by the active kernel and `BLOCK_N` is
the KV tile. There is no separate active `BLOCK_Q`, RoPE-fusion, warp-
specialization, ping-pong, or TMA switch. Those names were removed from the
active interface because the BI-V150 implementation and measurements did not
justify them.

The optimization must not be enabled for quantized KV cache, sliding-window
attention, ALiBi/sinks, non-causal attention, or pure decode. Those cases use
the native vLLM path in the plugin integration.

Both `ILUVATAR_USE_OPTIMIZED=1` and `ILUVATAR_SPLIT_KV=1` are required. This
prevents a benchmark that intends to run the native baseline from accidentally
using the candidate path.

## Validation Order

1. Run the harness correctness check against fp32 paged-SDPA.
2. Include a permuted physical `block_table`, non-uniform query lengths, and
   chunk boundaries that are not aligned to the tile size.
3. Compare Split-KV 4 and 8 against native vLLM attention on GPU 2, using
   the repaired launch mapping.
4. Run an isolated service A/B on 9032 with profiler disabled.
5. Only after the short A/B passes, run the official 4k/16k benchmark and
   Level 3 accuracy. The official 9031 service must remain untouched.

The harness reports CUDA-event latency, wall latency, numerical error, and
the causal FLOP estimate. Microbenchmark speedup is not service throughput.

## Installation

```bash
./scripts/install_to_vllm.sh
```

The script copies only active kernel files into an existing vLLM plugin
checkout. The Iluvatar backend registration and dispatch configuration are
kept as explicit plugin changes; the exact files and commit-local diff must
be reviewed and applied to the pinned `vllm-plugin-FL` checkout before a
service experiment. The installer does not silently monkey-patch vLLM.

## Evidence Status (September 28, 2026)

- Official Iluvatar baseline: `1983.45 tok/s` at 4k and `917.83 tok/s` at
  16k; Level 3: `102/105 = 97.1%`.
- Earlier Split-KV numbers came from a pre-audit harness and are exploratory;
  they must be re-run after the paged-KV and variable-length ABI fixes.
- No README target value is presented as a measured result.
- The repaired paged-KV kernel passed 6/6 correctness cases on remote GPU 2;
  repaired performance reruns and service A/B remain required before any
  throughput claim is promoted.

See [docs/architecture.md](docs/architecture.md) and
[experiments/experiment-log.md](experiments/experiment-log.md) for the
implementation contract and evidence trail.
