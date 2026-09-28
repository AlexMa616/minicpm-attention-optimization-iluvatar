"""Correctness gate for the opt-in paged Split-KV attention path.

Run inside mllv with the plugin checkout on PYTHONPATH and a free GPU:
    python3 tests/test_paged_split_kv.py --device cuda:2
"""

from __future__ import annotations

import argparse
import math

import torch

from vllm_fl.dispatch.backends.vendor.iluvatar.impl.ops.triton_split_kv_paged import (
    paged_split_kv_attention,
)


def reference(
    q: torch.Tensor,
    cache: torch.Tensor,
    table: torch.Tensor,
    query_lengths: list[int],
    prefixes: list[int],
) -> torch.Tensor:
    results = []
    offset = 0
    heads = q.shape[1]
    kv_heads = cache.shape[3]
    dim = q.shape[2]
    block_size = cache.shape[2]
    for row, (q_len, prefix) in enumerate(zip(query_lengths, prefixes)):
        seq_len = prefix + q_len
        ids = table[row, : math.ceil(seq_len / block_size)].long()
        k = cache[ids, 0].reshape(-1, kv_heads, dim)[:seq_len].float()
        v = cache[ids, 1].reshape(-1, kv_heads, dim)[:seq_len].float()
        k = k.repeat_interleave(heads // kv_heads, dim=1)
        v = v.repeat_interleave(heads // kv_heads, dim=1)
        logits = torch.einsum(
            "qhd,khd->hqk", q[offset : offset + q_len].float(), k
        ) / math.sqrt(dim)
        causal = torch.arange(seq_len, device=q.device)[None, :] <= (
            prefix + torch.arange(q_len, device=q.device)[:, None]
        )
        logits.masked_fill_(~causal[None], float("-inf"))
        results.append(
            torch.einsum("hqk,khd->qhd", logits.softmax(-1), v).to(q.dtype)
        )
        offset += q_len
    return torch.cat(results)


def check_case(
    device: str,
    query_lengths: list[int],
    prefixes: list[int],
    segments: int,
) -> None:
    block_size, heads, kv_heads, dim = 16, 16, 2, 128
    seq_lengths = [p + q for p, q in zip(prefixes, query_lengths)]
    table_width = max(math.ceil(s / block_size) for s in seq_lengths)
    block_counts = [math.ceil(s / block_size) for s in seq_lengths]
    total_blocks = sum(block_counts)
    cache = torch.randn(
        total_blocks, 2, block_size, kv_heads, dim,
        device=device, dtype=torch.bfloat16,
    )
    # A permuted table catches kernels that assume consecutive physical blocks.
    permutation = torch.randperm(total_blocks, device=device, dtype=torch.int32)
    table = torch.zeros(len(query_lengths), table_width, device=device, dtype=torch.int32)
    next_block = 0
    for row, count in enumerate(block_counts):
        table[row, :count] = permutation[next_block : next_block + count]
        next_block += count
    q = torch.randn(
        sum(query_lengths), heads, dim, device=device, dtype=torch.bfloat16
    )
    starts = torch.tensor(
        [0, *torch.tensor(query_lengths).cumsum(0).tolist()],
        device=device, dtype=torch.int32,
    )
    seq_lens = torch.tensor(seq_lengths, device=device, dtype=torch.int32)
    out = torch.empty_like(q)
    paged_split_kv_attention(
        q, cache[:, 0], cache[:, 1], out, table, seq_lens, starts,
        max(query_lengths), max(seq_lengths), 1.0 / math.sqrt(dim),
        num_splits=segments, block_m=32, block_n=32,
    )
    torch.cuda.synchronize()
    expected = reference(q, cache, table, query_lengths, prefixes)
    difference = (out.float() - expected.float()).abs()
    max_error = difference.max().item()
    mean_error = difference.mean().item()
    assert torch.isfinite(out).all().item(), "non-finite output"
    assert max_error <= 0.05 and mean_error <= 0.001, (
        f"max_error={max_error}, mean_error={mean_error}"
    )
    print(
        f"PASS queries={query_lengths} prefixes={prefixes} "
        f"segments={segments} max={max_error:.6f} mean={mean_error:.6f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    torch.manual_seed(20260928)
    for segments in (4, 8):
        check_case(args.device, [1, 17, 65], [127, 111, 63], segments)
        check_case(args.device, [31, 1, 13], [0, 255, 1001], segments)
        check_case(args.device, [64, 1, 1], [2048, 4096, 8192], segments)


if __name__ == "__main__":
    main()
