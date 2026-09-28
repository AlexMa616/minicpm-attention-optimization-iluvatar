#!/usr/bin/env python3
"""Probe the vendor dispatcher on a real mixed paged-KV batch."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from pathlib import Path

import torch

from attention_harness import (
    HEAD_SIZE,
    NUM_Q_HEADS,
    build_batch,
    reference_output,
)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import KVQuantMode
from vllm_fl.dispatch.backends.vendor.iluvatar.impl.attention import (
    _run_unified_attention,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--prefix", type=int, default=8192)
    parser.add_argument("--splits", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def make_kwargs(query, cache, cu_q, seq_lens, block_table, out):
    total_tokens = query.shape[0]
    head_size_padded = 1 << (HEAD_SIZE - 1).bit_length()
    segment_output = torch.empty(
        (total_tokens, NUM_Q_HEADS, 16, head_size_padded),
        device=query.device,
        dtype=torch.float32,
    )
    segment_max = torch.empty(
        (total_tokens, NUM_Q_HEADS, 16), device=query.device, dtype=torch.float32
    )
    return dict(
        q=query,
        k=cache[:, 0],
        v=cache[:, 1],
        out=out,
        cu_seqlens_q=cu_q,
        max_seqlen_q=2048,
        seqused_k=seq_lens,
        max_seqlen_k=int(seq_lens.max().item()),
        softmax_scale=1.0 / math.sqrt(HEAD_SIZE),
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        seq_threshold_3D=64,
        num_par_softmax_segments=16,
        softmax_segm_output=segment_output,
        softmax_segm_max=segment_max,
        softmax_segm_expsum=torch.empty_like(segment_max),
        alibi_slopes=None,
        output_scale=None,
        qq_bias=None,
        sinks=None,
        mm_prefix_range=None,
        use_alibi_sqrt=False,
        kv_quant_mode=KVQuantMode.NONE,
        k_scale_cache=None,
        v_scale_cache=None,
        chunk_lookback=-1,
        use_td=False,
    )


def timed(call, warmup, iterations, repeats):
    for _ in range(warmup):
        call()
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return statistics.median(samples), samples


def main() -> None:
    args = parse_args()
    os.environ["ILUVATAR_USE_OPTIMIZED"] = "1"
    os.environ["ILUVATAR_SPLIT_KV"] = "1"
    os.environ["ILUVATAR_NUM_SPLITS"] = str(args.splits)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    generator = torch.Generator(device=device).manual_seed(20260928)
    query_lengths = [2048] + [1] * 30
    prefix_lengths = [args.prefix] * len(query_lengths)
    query, cache, cu_q, seq_lens, block_table = build_batch(
        device, query_lengths, prefix_lengths, generator
    )
    reference = reference_output(
        query, cache, query_lengths, prefix_lengths, block_table
    )
    native_out = torch.empty_like(query)
    native_kwargs = make_kwargs(
        query, cache, cu_q, seq_lens, block_table, native_out
    )
    native_time, native_samples = timed(
        lambda: unified_attention(**native_kwargs),
        args.warmup,
        args.iterations,
        args.repeats,
    )
    candidate_out = torch.empty_like(query)
    candidate_kwargs = make_kwargs(
        query, cache, cu_q, seq_lens, block_table, candidate_out
    )
    _run_unified_attention(**candidate_kwargs)
    candidate_time, candidate_samples = timed(
        lambda: _run_unified_attention(**candidate_kwargs),
        args.warmup,
        args.iterations,
        args.repeats,
    )
    result = {
        "prefix": args.prefix,
        "splits": args.splits,
        "status": "ok",
        "candidate_ms": candidate_time,
        "native_ms": native_time,
        "speedup": native_time / candidate_time,
        "candidate_samples_ms": candidate_samples,
        "native_samples_ms": native_samples,
        "max_abs_vs_native": (candidate_out.float() - native_out.float()).abs().max().item(),
        "mean_abs_vs_native": (candidate_out.float() - native_out.float()).abs().mean().item(),
        "max_abs_vs_sdpa": (candidate_out.float() - reference.float()).abs().max().item(),
        "mean_abs_vs_sdpa": (candidate_out.float() - reference.float()).abs().mean().item(),
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
