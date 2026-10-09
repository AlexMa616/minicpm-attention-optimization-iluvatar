#!/usr/bin/env python3
"""Measure an isolated GPU exact-q-block metadata prototype.

This does not alter the production kernel. It compares the current launch
upper bound with an exact per-sequence block list and records the metadata
construction cost, so a future kernel change has an evidence-based gate.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def exact_mapping(q_lens: torch.Tensor, block_q: int):
    blocks = (q_lens + block_q - 1) // block_q
    starts = torch.cumsum(blocks, 0) - blocks
    total = int(blocks.sum().item())
    linear = torch.arange(total, device=q_lens.device, dtype=torch.int32)
    seq = torch.repeat_interleave(
        torch.arange(q_lens.numel(), device=q_lens.device, dtype=torch.int32), blocks
    )
    local = linear - torch.repeat_interleave(starts, blocks)
    return seq, local


def measure(device: torch.device, q_lens: list[int], block_q: int, repeats: int):
    lengths = torch.tensor(q_lens, device=device, dtype=torch.int32)
    upper = int(lengths.sum().item()) // block_q + len(q_lens)
    exact = sum((length + block_q - 1) // block_q for length in q_lens)

    for _ in range(10):
        exact_mapping(lengths, block_q)
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        seq, local = exact_mapping(lengths, block_q)
    end.record()
    end.synchronize()
    elapsed_ms = start.elapsed_time(end) / repeats
    assert seq.numel() == exact and local.numel() == exact
    return {
        "q_lens": q_lens,
        "block_q": block_q,
        "current_upper_bound": upper,
        "exact_blocks": exact,
        "avoided_ctas": upper - exact,
        "overlaunch_ratio": upper / exact if exact else None,
        "metadata_ms": elapsed_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    patterns = {
        "uniform_prefill": [2048] * 8,
        "mixed_prefill_decode": [2048] + [1] * 30,
        "short_variable": [1, 3, 7, 16, 17, 31, 63, 127] * 8,
        "chunked_16k": [512] * 32,
    }
    results = [
        {"name": name, **measure(device, q_lens, 16, args.repeats)}
        for name, q_lens in patterns.items()
    ]
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps({"device": str(device), "results": results}, indent=2))
    for row in results:
        print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
