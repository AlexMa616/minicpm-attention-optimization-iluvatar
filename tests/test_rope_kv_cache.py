"""Correctness gate for the fused Iluvatar RoPE+KV-cache kernel.

Run from the standalone repository root inside mllv:
    PYTHONPATH=. python3 tests/test_rope_kv_cache.py --device cuda:2
"""

from __future__ import annotations

import argparse
import torch

from kernels.triton_rope_kv_cache import (
    fused_rope_and_cache,
)


def reference_rope(x: torch.Tensor, cos_sin: torch.Tensor, positions: torch.Tensor, is_neox: bool, rotary_dim: int) -> torch.Tensor:
    out = x.clone().float()
    half = rotary_dim // 2
    cos, sin = cos_sin[positions].float().split(half, dim=-1)
    rot = out[..., :rotary_dim]
    if is_neox:
        first, second = rot[..., :half], rot[..., half:rotary_dim]
        rotated = torch.cat((first * cos[:, None, :] - second * sin[:, None, :],
                             first * sin[:, None, :] + second * cos[:, None, :]), dim=-1)
    else:
        even, odd = rot[..., 0::2], rot[..., 1::2]
        pair = torch.stack((even * cos[:, None, :] - odd * sin[:, None, :],
                            even * sin[:, None, :] + odd * cos[:, None, :]), dim=-1)
        rotated = pair.flatten(-2)
    out[..., :rotary_dim] = rotated
    return out.to(x.dtype)


def run_case(device: str, is_neox: bool, rotary_dim: int) -> None:
    torch.manual_seed(20260928 + int(is_neox) * 100 + rotary_dim)
    tokens, q_heads, kv_heads, head_dim = 9, 16, 2, 128
    block_size, blocks = 16, 8
    query = torch.randn(tokens, q_heads, head_dim, device=device, dtype=torch.bfloat16)
    key = torch.randn(tokens, kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    value = torch.randn(tokens, kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    positions = torch.tensor([0, 3, 17, 31, 63, 127, 255, 511, 1023], device=device, dtype=torch.int32)
    angles = torch.randn(2048, rotary_dim // 2, device=device, dtype=torch.float32)
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1).to(torch.bfloat16)
    slots = torch.tensor([0, 17, 34, 51, 68, 85, 102, 119, -1], device=device, dtype=torch.int32)
    key_cache = torch.full((blocks, block_size, kv_heads, head_dim), 7, device=device, dtype=torch.bfloat16)
    value_cache = torch.full_like(key_cache, -7)
    query_expected = reference_rope(query, cos_sin, positions, is_neox, rotary_dim)
    key_expected = reference_rope(key, cos_sin, positions, is_neox, rotary_dim)

    fused_rope_and_cache(
        query, key, value, positions, cos_sin, is_neox,
        key_cache, value_cache, slots, rotary_dim=rotary_dim,
    )
    torch.cuda.synchronize()

    query_error = (query.float() - query_expected.float()).abs().max().item()
    cache_key_error = 0.0
    cache_value_error = 0.0
    for token, slot in enumerate(slots.tolist()):
        if slot < 0:
            continue
        block, offset = divmod(slot, block_size)
        cache_key_error = max(cache_key_error, (key_cache[block, offset].float() - key_expected[token].float()).abs().max().item())
        cache_value_error = max(cache_value_error, (value_cache[block, offset].float() - value[token].float()).abs().max().item())
    assert query_error <= 0.02, query_error
    assert cache_key_error <= 0.02, cache_key_error
    assert cache_value_error <= 0.0, cache_value_error
    # Negative slots must not modify the cache.
    assert torch.equal(key_cache[-1, -1], torch.full_like(key_cache[-1, -1], 7))
    print(
        f"PASS neox={is_neox} rotary_dim={rotary_dim} "
        f"query_max={query_error:.6f} key_max={cache_key_error:.6f} "
        f"value_max={cache_value_error:.6f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    for is_neox in (True, False):
        for rotary_dim in (64, 128):
            run_case(args.device, is_neox, rotary_dim)


if __name__ == "__main__":
    main()
