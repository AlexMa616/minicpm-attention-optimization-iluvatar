#!/usr/bin/env python3
"""Direct Iluvatar unified-attention harness.

This script intentionally bypasses the vLLM service. It exercises the same
paged KV-cache ABI used by ``IluvatarAttentionImpl`` and is therefore suitable
for kernel screening before a service restart. It must run inside the mllv
container on a free GPU, with the plugin source on ``PYTHONPATH``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from vllm.v1.kv_cache_interface import KVQuantMode
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm_fl.dispatch.backends.vendor.iluvatar.impl.ops.triton_unified_attention_optimized import (
    unified_attention_optimized,
)


NUM_Q_HEADS = 16
NUM_KV_HEADS = 2
HEAD_SIZE = 128
BLOCK_SIZE = 16
DTYPE = torch.bfloat16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--shape", choices=("pure_prefill", "mixed", "all"), default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--block-m", type=int)
    parser.add_argument("--block-n", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warps", type=int)
    parser.add_argument("--stages", type=int)
    parser.add_argument("--prefix-token", type=int)
    parser.add_argument(
        "--split-kv-mixed",
        action="store_true",
        help="force the experimental 3D split-KV path for mixed batches",
    )
    parser.add_argument("--split-kv-segments", type=int, default=4)
    return parser.parse_args()


def read_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def set_device(device: str) -> torch.device:
    result = torch.device(device)
    torch.cuda.set_device(result)
    return result


def build_batch(
    device: torch.device,
    query_lengths: list[int],
    prefix_lengths: list[int],
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert len(query_lengths) == len(prefix_lengths)
    seq_lengths = [p + q for p, q in zip(prefix_lengths, query_lengths)]
    total_blocks = sum(math.ceil(seq_len / BLOCK_SIZE) for seq_len in seq_lengths)
    cache = torch.randn(
        total_blocks,
        2,
        BLOCK_SIZE,
        NUM_KV_HEADS,
        HEAD_SIZE,
        device=device,
        dtype=DTYPE,
        generator=generator,
    )
    max_blocks = max(math.ceil(seq_len / BLOCK_SIZE) for seq_len in seq_lengths)
    block_table = torch.empty(
        (len(seq_lengths), max_blocks), device=device, dtype=torch.int32
    )
    block_start = 0
    for row, seq_len in enumerate(seq_lengths):
        blocks = math.ceil(seq_len / BLOCK_SIZE)
        block_table[row].fill_(0)
        block_table[row, :blocks] = torch.arange(
            block_start, block_start + blocks, device=device, dtype=torch.int32
        )
        block_start += blocks

    query = torch.randn(
        sum(query_lengths), NUM_Q_HEADS, HEAD_SIZE,
        device=device, dtype=DTYPE, generator=generator
    )
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(query_lengths).cumsum(0).tolist()],
        device=device,
        dtype=torch.int32,
    )
    seq_lens = torch.tensor(seq_lengths, device=device, dtype=torch.int32)
    return query, cache, cu_seqlens, seq_lens, block_table


def reference_output(
    query: torch.Tensor,
    cache: torch.Tensor,
    query_lengths: list[int],
    prefix_lengths: list[int],
    block_table: torch.Tensor,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    offset = 0
    scale = 1.0 / math.sqrt(HEAD_SIZE)
    for row, (q_len, prefix_len) in enumerate(zip(query_lengths, prefix_lengths)):
        seq_len = q_len + prefix_len
        blocks = math.ceil(seq_len / BLOCK_SIZE)
        physical = block_table[row, :blocks].tolist()
        k = cache[physical, 0].reshape(-1, NUM_KV_HEADS, HEAD_SIZE)[:seq_len]
        v = cache[physical, 1].reshape(-1, NUM_KV_HEADS, HEAD_SIZE)[:seq_len]
        q = query[offset : offset + q_len]
        offset += q_len
        k = k.repeat_interleave(NUM_Q_HEADS // NUM_KV_HEADS, dim=1)
        v = v.repeat_interleave(NUM_Q_HEADS // NUM_KV_HEADS, dim=1)
        q_pos = torch.arange(prefix_len, seq_len, device=query.device)
        k_pos = torch.arange(seq_len, device=query.device)
        mask = k_pos[None, :] <= q_pos[:, None]
        logits = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * scale
        logits = logits.masked_fill(~mask[None, :, :], float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        outputs.append(torch.einsum("hqk,khd->qhd", probs, v.float()).to(DTYPE))
    return torch.cat(outputs, dim=0)


def run_attention(
    query: torch.Tensor,
    cache: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_query_len: int,
    block_m: int,
    block_n: int,
    optimized: bool = True,
    warps: int | None = None,
    stages: int | None = None,
    output: torch.Tensor | None = None,
    max_seqlen_k: int | None = None,
    segment_workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    decode_num_segments: int | None = None,
) -> torch.Tensor:
    if output is None:
        output = torch.empty_like(query)
    if max_seqlen_k is None:
        max_seqlen_k = int(seq_lens.max().item())
    if max_query_len == 1:
        num_segments = decode_num_segments or 16
        head_size_padded = 1 << (HEAD_SIZE - 1).bit_length()
        if segment_workspace is None:
            segment_output = torch.empty(
                (query.shape[0], NUM_Q_HEADS, num_segments, head_size_padded),
                device=query.device,
                dtype=torch.float32,
            )
            segment_max = torch.empty(
                (query.shape[0], NUM_Q_HEADS, num_segments),
                device=query.device,
                dtype=torch.float32,
            )
            segment_expsum = torch.empty_like(segment_max)
        else:
            segment_output, segment_max, segment_expsum = segment_workspace
        seq_threshold_3d = 64
    else:
        num_segments = None
        segment_output = segment_max = segment_expsum = None
        seq_threshold_3d = None
    window = (-1, -1)
    kwargs = dict(
        q=query,
        k=cache[:, 0],
        v=cache[:, 1],
        out=output,
        cu_seqlens_q=cu_seqlens,
        max_seqlen_q=max_query_len,
        seqused_k=seq_lens,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=1.0 / math.sqrt(HEAD_SIZE),
        causal=True,
        window_size=window,
        block_table=block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        seq_threshold_3D=seq_threshold_3d,
        num_par_softmax_segments=num_segments,
        softmax_segm_output=segment_output,
        softmax_segm_max=segment_max,
        softmax_segm_expsum=segment_expsum,
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
    if optimized:
        kwargs.update(
            block_m_override=block_m,
            block_n_override=block_n,
            launch_num_warps_override=warps,
            launch_num_stages_override=stages,
        )
    (unified_attention_optimized if optimized else unified_attention)(**kwargs)
    return output


def time_repeated(
    call: Any, iterations: int, repeats: int
) -> tuple[list[float], list[float]]:
    event_samples: list[float] = []
    wall_samples: list[float] = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start_event.record()
        for _ in range(iterations):
            call()
        end_event.record()
        end_event.synchronize()
        wall_elapsed = time.perf_counter() - wall_start
        event_elapsed = start_event.elapsed_time(end_event) / 1000
        event_samples.append(event_elapsed / iterations)
        wall_samples.append(wall_elapsed / iterations)
    return event_samples, wall_samples


def measure_case(
    device: torch.device,
    case_name: str,
    query_lengths: list[int],
    prefix_lengths: list[int],
    block_m: int,
    block_n: int,
    warmup: int,
    iterations: int,
    repeats: int,
    seed: int,
    warps: int | None = None,
    stages: int | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "shape": case_name,
        "prefix_tokens": prefix_lengths,
        "query_tokens": query_lengths,
        "decode_sequences": sum(q == 1 for q in query_lengths),
        "block_m": block_m,
        "block_n": block_n,
        "num_warps": warps,
        "num_stages": stages,
        "split_kv_mixed_enabled": os.getenv(
            "VLLM_ILUVATAR_ATTN_SPLIT_KV_MIXED", "0"
        ).lower() in ("1", "true", "yes"),
        "split_kv_segments_requested": (
            int(os.getenv("VLLM_ILUVATAR_ATTN_SPLIT_KV_SEGMENTS", "4"))
            if os.getenv("VLLM_ILUVATAR_ATTN_SPLIT_KV_MIXED", "0").lower()
            in ("1", "true", "yes")
            else None
        ),
        "status": "ok",
    }
    generator = torch.Generator(device=device).manual_seed(seed)
    query, cache, cu_seqlens, seq_lens, block_table = build_batch(
        device, query_lengths, prefix_lengths, generator
    )
    expected = reference_output(query, cache, query_lengths, prefix_lengths, block_table)
    compile_start = time.perf_counter()
    try:
        output = torch.empty_like(query)
        max_seqlen_k = int(seq_lens.max().item())
        segment_workspace = None
        baseline_segment_workspace = None
        candidate_decode_segments = 16
        if max(query_lengths) == 1:
            head_size_padded = 1 << (HEAD_SIZE - 1).bit_length()
            if os.getenv("VLLM_ILUVATAR_ATTN_SPLIT_KV_MIXED", "0").lower() in (
                "1", "true", "yes"
            ):
                candidate_decode_segments = max(
                    1,
                    min(
                        int(os.getenv("VLLM_ILUVATAR_ATTN_SPLIT_KV_SEGMENTS", "4")),
                        16,
                    ),
                )

            def make_decode_workspace(num_segments: int):
                segment_output = torch.empty(
                    (query.shape[0], NUM_Q_HEADS, num_segments, head_size_padded),
                    device=device,
                    dtype=torch.float32,
                )
                segment_max = torch.empty(
                    (query.shape[0], NUM_Q_HEADS, num_segments),
                    device=device,
                    dtype=torch.float32,
                )
                return segment_output, segment_max, torch.empty_like(segment_max)

            segment_workspace = make_decode_workspace(candidate_decode_segments)
            baseline_segment_workspace = make_decode_workspace(16)
        actual = run_attention(
            query, cache, cu_seqlens, seq_lens, block_table,
            max(query_lengths), block_m, block_n, True, warps, stages,
            output, max_seqlen_k, segment_workspace, candidate_decode_segments,
        )
        baseline_output = torch.empty_like(query)
        baseline = run_attention(
            query, cache, cu_seqlens, seq_lens, block_table,
            max(query_lengths), block_m, block_n, False, warps, stages,
            baseline_output, max_seqlen_k, baseline_segment_workspace, 16,
        )
        compile_seconds = time.perf_counter() - compile_start
        baseline_max_abs = (actual.float() - baseline.float()).abs().max().item()
        baseline_mean_abs = (actual.float() - baseline.float()).abs().mean().item()
        sdpa_max_abs = (actual.float() - expected.float()).abs().max().item()
        sdpa_mean_abs = (actual.float() - expected.float()).abs().mean().item()
        result.update(
            {
                "compile_seconds": compile_seconds,
                "max_abs_error": baseline_max_abs,
                "mean_abs_error": baseline_mean_abs,
                "sdpa_max_abs_error": sdpa_max_abs,
                "sdpa_mean_abs_error": sdpa_mean_abs,
            }
        )
        if sdpa_max_abs > 0.05 and sdpa_mean_abs > 0.001:
            result.update({"status": "fail", "failure_reason": "numerical mismatch"})
            return result
        for _ in range(warmup - 1):
            run_attention(
                query, cache, cu_seqlens, seq_lens, block_table,
                max(query_lengths), block_m, block_n, True, warps, stages,
                output, max_seqlen_k, segment_workspace, candidate_decode_segments,
            )
            run_attention(
                query, cache, cu_seqlens, seq_lens, block_table,
            max(query_lengths), block_m, block_n, False, warps, stages,
                baseline_output, max_seqlen_k, baseline_segment_workspace, 16,
            )
        samples, wall_samples = time_repeated(
            lambda: run_attention(
                query, cache, cu_seqlens, seq_lens, block_table,
                max(query_lengths), block_m, block_n, True, warps, stages,
                output, max_seqlen_k, segment_workspace, candidate_decode_segments,
            ),
            iterations,
            repeats,
        )
        baseline_samples, baseline_wall_samples = time_repeated(
            lambda: run_attention(
                query, cache, cu_seqlens, seq_lens, block_table,
                max(query_lengths), block_m, block_n, False, warps, stages,
                baseline_output, max_seqlen_k, baseline_segment_workspace, 16,
            ),
            iterations,
            repeats,
        )
        median_s = statistics.median(samples)
        baseline_median_s = statistics.median(baseline_samples)
        # Count both QK^T and PV multiply-adds, with the exact causal number
        # of query/key interactions for each sequence.
        causal_pairs = sum(
            q * p + q * (q + 1) // 2
            for q, p in zip(query_lengths, prefix_lengths)
        )
        flops = 4 * causal_pairs * NUM_Q_HEADS * HEAD_SIZE
        result.update(
            {
                "warmup_count": warmup,
                "timed_iterations": iterations,
                "median_ms": median_s * 1000,
                "batch_mean_samples_ms": [sample * 1000 for sample in samples],
                "max_batch_mean_ms": max(samples) * 1000,
                "cuda_ms": median_s * 1000,
                "timing_source": "cuda_event_batch_average",
                "timing_repeats": len(samples),
                "wall_median_ms": statistics.median(wall_samples) * 1000,
                "baseline_wall_median_ms": statistics.median(baseline_wall_samples) * 1000,
                "baseline_median_ms": baseline_median_s * 1000,
                "baseline_effective_tflops": flops / baseline_median_s / 1e12,
                "speedup_vs_vllm": baseline_median_s / max(median_s, 1e-9),
                "effective_tflops": flops / median_s / 1e12,
            }
        )
        if case_name == "mixed":
            decode_q = [q for q in query_lengths if q == 1]
            decode_result = measure_case(
                device=device,
                case_name="decode_probe",
                query_lengths=decode_q,
                prefix_lengths=[
                    p for q, p in zip(query_lengths, prefix_lengths) if q == 1
                ],
                block_m=block_m,
                block_n=block_n,
                warmup=max(2, warmup // 2),
                iterations=max(3, iterations // 2),
                repeats=repeats,
                seed=seed + 1,
                warps=warps,
                stages=stages,
            )
            if decode_result.get("status") == "ok":
                decode_only_ms = decode_result["median_ms"]
                result["decode_only_ms"] = decode_only_ms
                result["baseline_decode_only_ms"] = decode_result["baseline_median_ms"]
                result["decode_speedup_vs_vllm"] = decode_result[
                    "speedup_vs_vllm"
                ]
                result["decode_segments_tested"] = decode_result[
                    "split_kv_segments_requested"
                ]
                result["decode_cost_proxy_ratio"] = decode_only_ms / (
                    median_s * 1000
                )
                result["decode_cost_proxy_note"] = (
                    "isolated 3D decode probe using the mixed candidate segment "
                    "count, divided by full mixed-call time; not an attributable "
                    "decoder tail fraction or service pure-decode default"
                )
            else:
                result["decode_only_ms"] = None
                result["decode_cost_proxy_ratio"] = None
                result["decode_cost_proxy_note"] = decode_result.get(
                    "failure_reason", "decode probe failed"
                )
    except Exception as exc:  # keep one bad candidate from stopping the scan
        result.update({"status": "fail", "failure_reason": repr(exc)})
    return result


def main() -> None:
    args = parse_args()
    if args.split_kv_mixed:
        os.environ["VLLM_ILUVATAR_ATTN_SPLIT_KV_MIXED"] = "1"
        os.environ["VLLM_ILUVATAR_ATTN_SPLIT_KV_SEGMENTS"] = str(
            args.split_kv_segments
        )
    else:
        os.environ.pop("VLLM_ILUVATAR_ATTN_SPLIT_KV_MIXED", None)
    config = read_config(args.config)
    device = set_device(args.device)
    torch.manual_seed(args.seed)
    rows: list[dict[str, Any]] = []
    if args.block_m:
        candidates = [(args.block_m, args.block_n or 64, args.warps, args.stages)]
    else:
        candidates = [
            (m, n, None, None)
            for m in config["block_m_candidates"]
            for n in config.get("block_n_candidates", [64])
        ]
    cases: list[tuple[str, list[int], list[int]]] = []
    pure = config["pure_prefill"]
    for prefix in pure["prefix_lengths"]:
        cases.append(("pure_prefill", [pure["query_len"]] * pure["num_sequences"], [prefix] * pure["num_sequences"]))
    mixed = config["mixed"]
    for prefix in mixed["prefix_lengths"]:
        cases.append(("mixed", [mixed["prefill_query_len"]] + [1] * mixed["decode_sequences"], [prefix] + [prefix] * mixed["decode_sequences"]))
    if args.shape != "all":
        cases = [case for case in cases if case[0] == args.shape]
    if args.prefix_token is not None:
        cases = [case for case in cases if case[2][0] == args.prefix_token]
    if args.limit:
        cases = cases[:args.limit]
    args.result.parent.mkdir(parents=True, exist_ok=True)
    with args.result.open("w", encoding="utf-8") as output:
        for case_idx, (name, qlens, prefixes) in enumerate(cases):
            for cand_idx, (block_m, block_n, warps, stages) in enumerate(candidates):
                if args.check_only and case_idx > 0:
                    break
                row = measure_case(
                    device, name, qlens, prefixes, block_m, block_n,
                    args.warmup or config["warmup"],
                    args.iterations or config["iterations"],
                    args.repeats,
                    args.seed + case_idx * 1000 + cand_idx,
                    warps,
                    stages,
                )
                output.write(json.dumps(row, ensure_ascii=True) + "\n")
                output.flush()
                print(json.dumps(row, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
