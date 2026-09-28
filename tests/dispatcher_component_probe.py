#!/usr/bin/env python3
"""Attribute dispatcher subset construction to individual tensor operations."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch

from attention_harness import HEAD_SIZE, NUM_Q_HEADS, build_batch
from dispatcher_mixed_probe import make_kwargs
from vllm_fl.dispatch.backends.vendor.iluvatar.impl.attention import (
    _request_partition,
    _subset_attention_kwargs,
    _token_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--prefix", type=int, default=8192)
    parser.add_argument("--prefill-len", type=int, default=2048)
    parser.add_argument("--decode-requests", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def time_op(call: Callable[[], Any], warmup: int, repeats: int) -> dict[str, Any]:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    cuda_samples: list[float] = []
    wall_samples: list[float] = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start_event.record()
        call()
        end_event.record()
        end_event.synchronize()
        cuda_samples.append(start_event.elapsed_time(end_event))
        wall_samples.append((time.perf_counter() - wall_start) * 1000)
    return {
        "cuda_median_ms": statistics.median(cuda_samples),
        "wall_median_ms": statistics.median(wall_samples),
        "cuda_samples_ms": cuda_samples,
        "wall_samples_ms": wall_samples,
    }


def main() -> None:
    args = parse_args()
    os.environ["VLLM_PLUGINS"] = "fl"
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    query_lengths = [args.prefill_len] + [1] * args.decode_requests
    prefixes = [args.prefix] * len(query_lengths)
    generator = torch.Generator(device=device).manual_seed(20260929)
    query, cache, cu_q, seq_lens, block_table = build_batch(
        device, query_lengths, prefixes, generator
    )
    out = torch.empty_like(query)
    kwargs = make_kwargs(query, cache, cu_q, seq_lens, block_table, out)
    kwargs["query_start_loc_cpu"] = cu_q.cpu()
    prefill_ids, decode_ids, starts, lengths = _request_partition(cu_q.cpu())

    groups = {
        "prefill": prefill_ids,
        "decode": decode_ids,
    }
    result: dict[str, Any] = {
        "status": "ok",
        "shape": {
            "query_lengths": query_lengths,
            "prefix": args.prefix,
            "groups": {name: ids for name, ids in groups.items()},
        },
        "groups": {},
    }

    for name, request_ids in groups.items():
        indices = _token_indices(cu_q, request_ids, starts, lengths)
        selected_lengths = [lengths[i] for i in request_ids]
        request_index = torch.tensor(
            request_ids, device=device, dtype=torch.long
        )

        def make_selected_cu() -> torch.Tensor:
            selected_cu = torch.zeros(
                len(selected_lengths) + 1,
                device=device,
                dtype=cu_q.dtype,
            )
            if selected_lengths:
                selected_cu[1:] = torch.tensor(
                    selected_lengths,
                    device=device,
                    dtype=selected_cu.dtype,
                ).cumsum(0)
            return selected_cu

        def q_index_select() -> torch.Tensor:
            return query.index_select(0, indices)

        selected_q = q_index_select()

        def make_subset() -> dict[str, Any]:
            return _subset_attention_kwargs(kwargs, request_ids, lengths, indices)

        result["groups"][name] = {
            "request_count": len(request_ids),
            "token_count": int(indices.numel()),
            "ops": {
                "request_index_tensor": time_op(
                    lambda: torch.tensor(request_ids, device=device, dtype=torch.long),
                    args.warmup,
                    args.repeats,
                ),
                "selected_cu_and_lengths": time_op(
                    make_selected_cu, args.warmup, args.repeats
                ),
                "q_index_select": time_op(
                    q_index_select, args.warmup, args.repeats
                ),
                "out_empty_like": time_op(
                    lambda: torch.empty_like(selected_q), args.warmup, args.repeats
                ),
                "seqused_k_index_select": time_op(
                    lambda: kwargs["seqused_k"].index_select(0, request_index),
                    args.warmup,
                    args.repeats,
                ),
                "block_table_index_select": time_op(
                    lambda: kwargs["block_table"].index_select(0, request_index),
                    args.warmup,
                    args.repeats,
                ),
                "full_subset_helper": time_op(
                    make_subset, args.warmup, args.repeats
                ),
            },
        }

    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
