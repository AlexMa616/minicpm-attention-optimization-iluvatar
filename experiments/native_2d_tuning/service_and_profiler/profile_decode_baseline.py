#!/usr/bin/env python3
"""Measure the current decode attention paths without touching a service.

This is an evidence-only experiment for the q_len=1 path.  It compares the
version-pinned vLLM unified-attention kernel with the Native 2D implementation
using the same paged-KV tensors.  The Native 2D call is diagnostic only: the
production dispatcher still routes decode to the upstream implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from native_2d_harness import batch, kwargs
from vllm.v1.attention.ops.triton_unified_attention import (
    unified_attention as upstream_attention,
)
from vllm_fl.dispatch.backends.vendor.iluvatar.impl.ops.triton_unified_attention_native import (
    unified_attention as native_attention,
)


def measure(fn, kw, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        fn(**kw)
    torch.cuda.synchronize()
    values = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(**kw)
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)))
    return values


def profile_once(fn, kw) -> list[dict[str, object]]:
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn(**kw)
        torch.cuda.synchronize()
    rows = []
    for event in prof.key_averages():
        if "unified_attention" in event.key or "reduce_segments" in event.key:
            rows.append(
                {
                    "name": event.key,
                    "calls": event.count,
                    "cpu_us": event.cpu_time_total,
                    "cuda_us": getattr(event, "device_time_total", None),
                }
            )
    return rows


def run_case(device: torch.device, context: int, batch_size: int, seed: int,
             warmup: int, iterations: int) -> dict[str, object]:
    qs = [1] * batch_size
    prefixes = [context] * batch_size
    q, cache, cu, lens, table = batch(device, qs, prefixes, seed)
    out_upstream = torch.empty_like(q)
    out_native = torch.empty_like(q)
    kw_upstream = kwargs(q, cache, cu, lens, table, out_upstream)
    kw_native = kwargs(q, cache, cu, lens, table, out_native)
    kw_upstream["max_seqlen_q"] = 1
    kw_native["max_seqlen_q"] = 1

    kw_decode16 = dict(kw_native, decode_only_override=True,
                       decode_block_m_override=16)
    kw_decode8 = dict(kw_native, decode_only_override=True,
                      decode_block_m_override=8)

    # Compile and establish correctness before timing.
    upstream_attention(**kw_upstream)
    native_attention(**kw_native)
    native_attention(**kw_decode16)
    native_attention(**kw_decode8)
    torch.cuda.synchronize()
    error = (out_upstream.float() - out_native.float()).abs()

    upstream_ms = measure(upstream_attention, kw_upstream, warmup, iterations)
    native_ms = measure(native_attention, kw_native, warmup, iterations)
    decode16_ms = measure(native_attention, kw_decode16, warmup, iterations)
    decode8_ms = measure(native_attention, kw_decode8, warmup, iterations)
    decode16_error = (out_native.float() - out_native.float()).abs()
    # Reuse the output buffer for correctness checks one variant at a time.
    native_attention(**kw_decode16)
    decode16_out = kw_decode16["out"].clone()
    native_attention(**kw_decode8)
    decode8_out = kw_decode8["out"].clone()
    decode16_error = (out_upstream.float() - decode16_out.float()).abs()
    decode8_error = (out_upstream.float() - decode8_out.float()).abs()
    return {
        "context": context,
        "batch": batch_size,
        "upstream_ms": upstream_ms,
        "native_default_ms": native_ms,
        "upstream_median_ms": statistics.median(upstream_ms),
        "native_default_median_ms": statistics.median(native_ms),
        "decode16_median_ms": statistics.median(decode16_ms),
        "decode8_median_ms": statistics.median(decode8_ms),
        "native_over_upstream": statistics.median(native_ms)
        / statistics.median(upstream_ms),
        "decode16_over_upstream": statistics.median(decode16_ms)
        / statistics.median(upstream_ms),
        "decode8_over_upstream": statistics.median(decode8_ms)
        / statistics.median(upstream_ms),
        "max_abs_error": error.max().item(),
        "mean_abs_error": error.mean().item(),
        "decode16_max_abs_error": decode16_error.max().item(),
        "decode8_max_abs_error": decode8_error.max().item(),
        "upstream_profile": profile_once(upstream_attention, kw_upstream),
        "native_profile": profile_once(native_attention, kw_native),
        "decode16_profile": profile_once(native_attention, kw_decode16),
        "decode8_profile": profile_once(native_attention, kw_decode8),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", type=int, nargs="+", default=[4096, 16384])
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    records = []
    for context in args.contexts:
        record = run_case(
            device, context, args.batch, args.seed, args.warmup, args.iterations
        )
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    result = {
        "device": str(device),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
