# Native 2D Attention Experiments

> Archive update, 2026-10-09: copied from the local `vllm-plugin-FL`
> experiment workspace, not an installable plugin checkout. See
> `../../docs/stage-summary-2026-10-09.md` for current routing findings.
> Statements below describe the historical candidate, not new acceptance.
> Paths under `vllm_fl/`, `benchmarks/`, and `tools/` refer to a separate
> `vllm-plugin-FL` checkout.

## Running Archived Tools

Tools may assume the old `/workspace/vllm-plugin-FL`, GPU, port, markers,
and explicit PID. Review and adapt them in an isolated checkout before use.
Never use an old PID on a new machine. Legacy installers and restart scripts
are not an automatic recovery recipe. The new Decode/queue/route tools and
`../patches/` remain experimental; do not start loads just to test this archive.

This directory contains diagnostic and rejected alternatives from the
MiniCPM5-2B Iluvatar optimization work. Nothing here is imported by the
production dispatch path.

## Production result

The submitted path is the Native 2D kernel in
`vllm_fl/dispatch/backends/vendor/iluvatar/impl/ops/triton_unified_attention_native.py`.
The production dispatcher keeps decode and unsupported attention features on
the upstream vLLM implementation.

The measured candidate configuration is:

- `BLOCK_M=128`
- `TILE_SIZE=16`
- `num_warps=8`
- `num_stages=2`
- `pipeline_stages=1`

## Rejected alternatives

`rejected_split_kv/` contains the Split-KV prototype. It was removed from the
production source tree after service-level tests showed no stable gain.

`service_and_profiler/` contains A/B, profiling, dual-launch, and parallel
scan scripts used during development. They are retained only for auditability
and are not required to run the final service.

The official benchmark harness remains under `benchmarks/`, and the focused
kernel harness remains under `tools/native_2d_harness.py`.

## Experimental parameters

The production dispatcher (`vllm_fl/dispatch/backends/vendor/iluvatar/impl/attention.py`)
only passes `block_m_override`, `tile_size_prefill_override`, `launch_num_warps_override`,
`launch_num_stages_override`, and `pipeline_stages_override=1`.

`scalar_block_lookup_override` and `mixed_dual_launch_override` are experimental-only
parameters used by `tools/native_2d_harness.py` and profiling scripts for ablation studies.
They remain disabled in production because no stable service-level gain was observed.
