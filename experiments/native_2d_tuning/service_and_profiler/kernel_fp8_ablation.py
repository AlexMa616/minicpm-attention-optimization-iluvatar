#!/usr/bin/env python3
"""Kernel-only FP8 per-token-head ablation for the Iluvatar Native 2D path.

This script does not start vLLM and does not touch the service path. It compares
the same Native 2D launch with BF16 KV, FP8 KV plus real per-token/head scales,
and FP8 KV plus unit scales. The latter keeps the scale-cache loads while
removing scale-value variation, which helps separate scale arithmetic from the
FP8 load/cast cost.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import KVQuantMode
from vllm_fl.dispatch.backends.vendor.iluvatar.impl.ops.triton_unified_attention_native import (
    unified_attention,
)


NQ = 16
NKV = 2
HEAD_SIZE = 128
BLOCK_SIZE = 16
BLOCK_M = 128
TILE_SIZE = 16
NUM_WARPS = 8
NUM_STAGES = 2
PIPELINE_STAGES = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261008)
    return parser.parse_args()


def make_batch(device: torch.device, seed: int):
    generator = torch.Generator(device=device).manual_seed(seed)
    query_lens = [2048] + [1] * 30
    prefixes = [8192] + [16384] * 30
    seq_lens = [q + p for q, p in zip(query_lens, prefixes)]
    blocks_per_seq = [math.ceil(length / BLOCK_SIZE) for length in seq_lens]
    total_blocks = sum(blocks_per_seq)

    cache = torch.randn(
        (total_blocks, 2, BLOCK_SIZE, NKV, HEAD_SIZE),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    block_table = torch.zeros(
        (len(seq_lens), max(blocks_per_seq)), device=device, dtype=torch.int32
    )
    cursor = 0
    for row, num_blocks in enumerate(blocks_per_seq):
        block_table[row, :num_blocks] = torch.arange(
            cursor, cursor + num_blocks, device=device
        )
        cursor += num_blocks

    query = torch.randn(
        (sum(query_lens), NQ, HEAD_SIZE),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    cu_seqlens_q = torch.tensor(
        [0, *torch.tensor(query_lens).cumsum(0).tolist()],
        device=device,
        dtype=torch.int32,
    )
    seq_lens_tensor = torch.tensor(seq_lens, device=device, dtype=torch.int32)
    return query, cache, cu_seqlens_q, seq_lens_tensor, block_table, query_lens


def make_fp8_cache(cache: torch.Tensor, fp8_dtype: torch.dtype):
    fp8_max = torch.finfo(fp8_dtype).max
    cache_f32 = cache.float()
    k_scale = cache_f32[:, 0].abs().amax(dim=-1).clamp_min(1e-6) / fp8_max
    v_scale = cache_f32[:, 1].abs().amax(dim=-1).clamp_min(1e-6) / fp8_max

    # The four tail bytes model the inline scale padding used by the service
    # cache. The kernel masks them out because HEAD_SIZE remains 128.
    padded = HEAD_SIZE + 4
    fp8_cache = torch.zeros(
        (*cache.shape[:-1], padded), device=cache.device, dtype=fp8_dtype
    )
    fp8_cache[:, 0, ..., :HEAD_SIZE] = (
        cache_f32[:, 0] / k_scale[..., None]
    ).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    fp8_cache[:, 1, ..., :HEAD_SIZE] = (
        cache_f32[:, 1] / v_scale[..., None]
    ).clamp(-fp8_max, fp8_max).to(fp8_dtype)

    # Match the service's interleaved K/V block stride: K uses even blocks and
    # V uses odd blocks in one backing allocation.
    scale_backing = torch.empty(
        (cache.shape[0] * 2, BLOCK_SIZE, NKV),
        device=cache.device,
        dtype=torch.float32,
    )
    scale_backing[0::2].copy_(k_scale)
    scale_backing[1::2].copy_(v_scale)
    k_scales = scale_backing[0::2]
    v_scales = scale_backing[1::2]
    unit_backing = torch.ones_like(scale_backing)
    return fp8_cache, k_scales, v_scales, unit_backing[0::2], unit_backing[1::2]


def make_int8_cache(cache: torch.Tensor):
    cache_f32 = cache.float()
    k_scale = cache_f32[:, 0].abs().amax(dim=-1).clamp_min(1e-6) / 127.0
    v_scale = cache_f32[:, 1].abs().amax(dim=-1).clamp_min(1e-6) / 127.0
    padded = HEAD_SIZE + 4
    int8_cache = torch.zeros(
        (*cache.shape[:-1], padded), device=cache.device, dtype=torch.int8
    )
    int8_cache[:, 0, ..., :HEAD_SIZE] = (
        cache_f32[:, 0] / k_scale[..., None]
    ).round().clamp(-128, 127).to(torch.int8)
    int8_cache[:, 1, ..., :HEAD_SIZE] = (
        cache_f32[:, 1] / v_scale[..., None]
    ).round().clamp(-128, 127).to(torch.int8)
    scale_backing = torch.empty(
        (cache.shape[0] * 2, BLOCK_SIZE, NKV),
        device=cache.device,
        dtype=torch.float32,
    )
    scale_backing[0::2].copy_(k_scale)
    scale_backing[1::2].copy_(v_scale)
    unit_backing = torch.ones_like(scale_backing)
    return (
        int8_cache,
        scale_backing[0::2],
        scale_backing[1::2],
        unit_backing[0::2],
        unit_backing[1::2],
    )


def common_kwargs(query, key, value, cu_seqlens_q, seq_lens, block_table, output):
    return dict(
        q=query,
        k=key,
        v=value,
        out=output,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=2048,
        seqused_k=seq_lens,
        max_seqlen_k=int(seq_lens.max()),
        softmax_scale=1.0 / math.sqrt(HEAD_SIZE),
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        seq_threshold_3D=None,
        num_par_softmax_segments=None,
        softmax_segm_output=None,
        softmax_segm_max=None,
        softmax_segm_expsum=None,
        alibi_slopes=None,
        output_scale=None,
        qq_bias=None,
        sinks=None,
        mm_prefix_range=None,
        use_alibi_sqrt=False,
        chunk_lookback=-1,
        use_td=False,
        block_m_override=BLOCK_M,
        tile_size_prefill_override=TILE_SIZE,
        launch_num_warps_override=NUM_WARPS,
        launch_num_stages_override=NUM_STAGES,
        pipeline_stages_override=PIPELINE_STAGES,
        scalar_block_lookup_override=True,
        mixed_dual_launch_override=False,
    )


def invoke(kwargs):
    unified_attention(**kwargs)


def measure(kwargs, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        invoke(kwargs)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        invoke(kwargs)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    query, cache, cu, seq_lens, block_table, query_lens = make_batch(
        device, args.seed
    )
    fp8_dtype = current_platform.fp8_dtype()
    fp8_cache, k_scales, v_scales, unit_k, unit_v = make_fp8_cache(
        cache, fp8_dtype
    )
    int8_cache, int8_k, int8_v, int8_unit_k, int8_unit_v = make_int8_cache(cache)

    bf16_out = torch.empty_like(query)
    bf16_scale_out = torch.empty_like(query)
    bf16_unit_out = torch.empty_like(query)
    actual_out = torch.empty_like(query)
    unit_out = torch.empty_like(query)
    bf16_kwargs = common_kwargs(
        query, cache[:, 0], cache[:, 1], cu, seq_lens, block_table, bf16_out
    )
    bf16_scale_kwargs = common_kwargs(
        query,
        cache[:, 0],
        cache[:, 1],
        cu,
        seq_lens,
        block_table,
        bf16_scale_out,
    )
    bf16_scale_kwargs.update(
        kv_quant_mode=KVQuantMode.FP8_PER_TOKEN_HEAD,
        k_scale_cache=k_scales,
        v_scale_cache=v_scales,
    )
    bf16_unit_kwargs = dict(bf16_scale_kwargs, out=bf16_unit_out)
    bf16_unit_kwargs.update(k_scale_cache=unit_k, v_scale_cache=unit_v)
    actual_kwargs = common_kwargs(
        query, fp8_cache[:, 0], fp8_cache[:, 1], cu, seq_lens, block_table, actual_out
    )
    actual_kwargs.update(
        kv_quant_mode=KVQuantMode.FP8_PER_TOKEN_HEAD,
        k_scale_cache=k_scales,
        v_scale_cache=v_scales,
    )
    unit_kwargs = dict(actual_kwargs, out=unit_out)
    unit_kwargs.update(k_scale_cache=unit_k, v_scale_cache=unit_v)
    int8_out = torch.empty_like(query)
    int8_unit_out = torch.empty_like(query)
    int8_kwargs = common_kwargs(
        query, int8_cache[:, 0], int8_cache[:, 1], cu, seq_lens, block_table, int8_out
    )
    int8_kwargs.update(
        kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
        k_scale_cache=int8_k,
        v_scale_cache=int8_v,
    )
    int8_unit_kwargs = dict(int8_kwargs, out=int8_unit_out)
    int8_unit_kwargs.update(k_scale_cache=int8_unit_k, v_scale_cache=int8_unit_v)

    # Compile and validate before timing.
    invoke(bf16_kwargs)
    invoke(bf16_scale_kwargs)
    invoke(bf16_unit_kwargs)
    invoke(actual_kwargs)
    invoke(unit_kwargs)
    invoke(int8_kwargs)
    invoke(int8_unit_kwargs)
    torch.cuda.synchronize()

    results = {
        "device": str(device),
        "fp8_dtype": str(fp8_dtype),
        "shape": {"query_lens": query_lens, "max_seq_len": int(seq_lens.max())},
        "config": {
            "block_m": BLOCK_M,
            "tile": TILE_SIZE,
            "warps": NUM_WARPS,
            "stages": NUM_STAGES,
            "pipeline_stages": PIPELINE_STAGES,
            "scalar_block_lookup": True,
        },
        "bf16_ms": measure(bf16_kwargs, args.warmup, args.iterations),
        "bf16_actual_scale_ms": measure(
            bf16_scale_kwargs, args.warmup, args.iterations
        ),
        "bf16_unit_scale_ms": measure(
            bf16_unit_kwargs, args.warmup, args.iterations
        ),
        "fp8_actual_scale_ms": measure(actual_kwargs, args.warmup, args.iterations),
        "fp8_unit_scale_ms": measure(unit_kwargs, args.warmup, args.iterations),
        "int8_actual_scale_ms": measure(int8_kwargs, args.warmup, args.iterations),
        "int8_unit_scale_ms": measure(
            int8_unit_kwargs, args.warmup, args.iterations
        ),
    }
    results["fp8_actual_vs_bf16"] = results["bf16_ms"] / results[
        "fp8_actual_scale_ms"
    ]
    results["fp8_unit_vs_bf16"] = results["bf16_ms"] / results["fp8_unit_scale_ms"]
    results["bf16_scale_overhead_ms"] = (
        results["bf16_actual_scale_ms"] - results["bf16_ms"]
    )
    results["fp8_cast_overhead_ms"] = (
        results["fp8_actual_scale_ms"] - results["bf16_actual_scale_ms"]
    )
    results["int8_vs_bf16"] = results["bf16_ms"] / results["int8_actual_scale_ms"]
    results["int8_scale_overhead_ms"] = (
        results["int8_actual_scale_ms"] - results["bf16_ms"]
    )
    results["unit_scale_delta_ms"] = (
        results["fp8_actual_scale_ms"] - results["fp8_unit_scale_ms"]
    )
    results["actual_unit_max_abs"] = (
        (actual_out.float() - unit_out.float()).abs().max().item()
    )
    results["bf16_fp8_actual_max_abs"] = float("nan")
    results["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
