#!/usr/bin/env python3
"""Profile the mixed dispatcher without starting a vLLM service.

The probe compares three paths on one identical paged-KV batch:

* native unified attention for the complete mixed batch;
* raw Split-KV for the prefill subset only;
* the diagnostic dispatcher (Split-KV prefill + native decode + scatter).

This is a diagnosis tool, not a service benchmark.  Run it only on an idle
GPU in the mllv container.  The profiler output is deliberately reduced to
JSON so a long Chrome trace is not required for the first bottleneck pass.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

import torch

from attention_harness import (
    HEAD_SIZE,
    NUM_Q_HEADS,
    build_batch,
    reference_output,
)
from dispatcher_mixed_probe import make_kwargs
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm_fl.dispatch.backends.vendor.iluvatar.impl import attention as iluvatar_attention
from vllm_fl.dispatch.backends.vendor.iluvatar.impl.attention import (
    _request_partition,
    _run_optimized_attention,
    _subset_attention_kwargs,
    _token_indices,
    _run_unified_attention,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--prefix", type=int, default=8192)
    parser.add_argument("--prefill-len", type=int, default=2048)
    parser.add_argument("--decode-requests", type=int, default=30)
    parser.add_argument("--splits", type=int, default=4)
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--profile-iterations", type=int, default=2)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    return parser.parse_args()


def timed(call: Callable[[], Any], warmup: int, iterations: int) -> dict[str, Any]:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    event_start = torch.cuda.Event(enable_timing=True)
    event_end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    event_start.record()
    for _ in range(iterations):
        call()
    event_end.record()
    event_end.synchronize()
    wall_seconds = time.perf_counter() - wall_start
    cuda_ms = event_start.elapsed_time(event_end) / iterations
    return {
        "cuda_ms": cuda_ms,
        "wall_ms": wall_seconds * 1000 / iterations,
    }


class CpuTimers:
    def __init__(self) -> None:
        self.samples: dict[str, list[float]] = {}

    def wrap(self, name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                self.samples.setdefault(name, []).append(
                    (time.perf_counter() - start) * 1000
                )

        return wrapped

    def summary(self) -> dict[str, dict[str, float]]:
        return {
            name: {
                "count": len(values),
                "total_ms": sum(values),
                "mean_ms": statistics.mean(values),
                "max_ms": max(values),
            }
            for name, values in sorted(self.samples.items())
        }


def profiler_events(profiler: torch.profiler.profile, limit: int = 40) -> list[dict[str, Any]]:
    rows = []
    for event in profiler.key_averages():
        rows.append(
            {
                "name": event.key,
                "count": event.count,
                "cpu_total_us": event.cpu_time_total,
                "cuda_total_us": event.device_time_total,
                "self_cpu_us": event.self_cpu_time_total,
                "self_cuda_us": event.device_time_total,
                "input_shapes": event.input_shapes,
            }
        )
    rows.sort(key=lambda row: row["cuda_total_us"], reverse=True)
    return rows[:limit]


def profile_call(
    call: Callable[[], Any], iterations: int, trace_dir: Path
) -> list[dict[str, Any]]:
    trace_dir.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)),
    ) as profiler:
        for _ in range(iterations):
            call()
            profiler.step()
    return profiler_events(profiler)


def main() -> None:
    args = parse_args()
    os.environ["ILUVATAR_USE_OPTIMIZED"] = "1"
    os.environ["ILUVATAR_SPLIT_KV"] = "1"
    os.environ["ILUVATAR_NUM_SPLITS"] = str(args.splits)
    os.environ["ILUVATAR_BLOCK_M"] = str(args.block_m)
    os.environ["ILUVATAR_BLOCK_N"] = str(args.block_n)

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    query_lengths = [args.prefill_len] + [1] * args.decode_requests
    prefix_lengths = [args.prefix] * len(query_lengths)
    generator = torch.Generator(device=device).manual_seed(20260928)
    query, cache, cu_q, seq_lens, block_table = build_batch(
        device, query_lengths, prefix_lengths, generator
    )
    reference = reference_output(
        query, cache, query_lengths, prefix_lengths, block_table
    )
    max_seqlen_k = int(seq_lens.max().item())

    native_out = torch.empty_like(query)
    native_kwargs = make_kwargs(query, cache, cu_q, seq_lens, block_table, native_out)
    native_kwargs["max_seqlen_k"] = max_seqlen_k

    candidate_out = torch.empty_like(query)
    candidate_kwargs = make_kwargs(
        query, cache, cu_q, seq_lens, block_table, candidate_out
    )
    candidate_kwargs["max_seqlen_k"] = max_seqlen_k
    candidate_kwargs["query_start_loc_cpu"] = cu_q.cpu()

    prefill_requests, decode_requests, starts, lengths = _request_partition(cu_q.cpu())
    prefill_indices = _token_indices(cu_q, prefill_requests, starts, lengths)
    prefill_kwargs = _subset_attention_kwargs(
        candidate_kwargs, prefill_requests, lengths, prefill_indices
    )

    native_timing = timed(
        lambda: unified_attention(**native_kwargs), args.warmup, args.iterations
    )
    raw_prefill_timing = timed(
        lambda: _run_optimized_attention(**prefill_kwargs),
        args.warmup,
        args.iterations,
    )

    timers = CpuTimers()
    wrapped_request_partition = timers.wrap(
        "request_partition", iluvatar_attention._request_partition
    )
    wrapped_token_indices = timers.wrap("token_indices", iluvatar_attention._token_indices)
    wrapped_subset_kwargs = timers.wrap(
        "subset_attention_kwargs", iluvatar_attention._subset_attention_kwargs
    )
    wrapped_optimized = timers.wrap(
        "run_optimized_attention", iluvatar_attention._run_optimized_attention
    )
    wrapped_native = timers.wrap(
        "native_unified_attention", iluvatar_attention.vllm_unified_attention
    )
    candidate_timing: dict[str, Any]
    with (
        patch.object(iluvatar_attention, "_request_partition", wrapped_request_partition),
        patch.object(iluvatar_attention, "_token_indices", wrapped_token_indices),
        patch.object(
            iluvatar_attention, "_subset_attention_kwargs", wrapped_subset_kwargs
        ),
        patch.object(iluvatar_attention, "_run_optimized_attention", wrapped_optimized),
        patch.object(iluvatar_attention, "vllm_unified_attention", wrapped_native),
    ):
        candidate_timing = timed(
            lambda: _run_unified_attention(**candidate_kwargs),
            args.warmup,
            args.iterations,
        )

    candidate_max_abs = (candidate_out.float() - native_out.float()).abs().max().item()
    candidate_sdpa_max_abs = (
        candidate_out.float() - reference.float()
    ).abs().max().item()

    profile_target = args.trace or args.result.with_suffix(".json.trace")
    profile_target.parent.mkdir(parents=True, exist_ok=True)
    profile_timers = CpuTimers()
    with (
        patch.object(
            iluvatar_attention,
            "_request_partition",
            profile_timers.wrap("request_partition", iluvatar_attention._request_partition),
        ),
        patch.object(
            iluvatar_attention,
            "_token_indices",
            profile_timers.wrap("token_indices", iluvatar_attention._token_indices),
        ),
        patch.object(
            iluvatar_attention,
            "_subset_attention_kwargs",
            profile_timers.wrap(
                "subset_attention_kwargs", iluvatar_attention._subset_attention_kwargs
            ),
        ),
        patch.object(
            iluvatar_attention,
            "_run_optimized_attention",
            profile_timers.wrap(
                "run_optimized_attention", iluvatar_attention._run_optimized_attention
            ),
        ),
        patch.object(
            iluvatar_attention,
            "vllm_unified_attention",
            profile_timers.wrap(
                "native_unified_attention", iluvatar_attention.vllm_unified_attention
            ),
        ),
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profile_target.parent)),
        ) as profiler,
    ):
        for _ in range(args.profile_iterations):
            _run_unified_attention(**candidate_kwargs)
            profiler.step()

    path_profiles = {
        "native_full_mixed": profile_call(
            lambda: unified_attention(**native_kwargs),
            args.profile_iterations,
            profile_target.parent / "native_full_mixed",
        ),
        "raw_split_kv_prefill": profile_call(
            lambda: _run_optimized_attention(**prefill_kwargs),
            args.profile_iterations,
            profile_target.parent / "raw_split_kv_prefill",
        ),
        "dispatcher_partitioned": profiler_events(profiler),
    }

    result = {
        "status": "ok",
        "device": str(device),
        "shape": {
            "query_lengths": query_lengths,
            "prefix_lengths": prefix_lengths,
            "prefill_requests": prefill_requests,
            "decode_requests": decode_requests,
            "splits": args.splits,
            "block_m": args.block_m,
            "block_n": args.block_n,
        },
        "timing": {
            "native_full_mixed": native_timing,
            "raw_split_kv_prefill": raw_prefill_timing,
            "dispatcher_partitioned": candidate_timing,
        },
        "cpu_timers_ms": timers.summary(),
        "profile_cpu_timers_ms": profile_timers.summary(),
        "correctness": {
            "max_abs_vs_native": candidate_max_abs,
            "max_abs_vs_sdpa": candidate_sdpa_max_abs,
        },
        "profiler_top_events": profiler_events(profiler),
        "path_profiler_top_events": path_profiles,
        "trace_directory": str(profile_target.parent),
    }
    if candidate_sdpa_max_abs > 0.05:
        result["status"] = "fail"
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if result["status"] != "ok":
        raise AssertionError("profile probe failed correctness gate")


if __name__ == "__main__":
    main()
