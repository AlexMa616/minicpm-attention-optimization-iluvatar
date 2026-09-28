"""Fused RoPE and paged KV-cache update for Iluvatar BI-V150.

This kernel implements the vLLM ``fused_rope_and_unified_kv_cache_update``
ABI for the NHD cache layout used by MiniCPM5-2B.  It rotates Q in place,
rotates K while writing it to the paged cache, and copies V to the cache in
the same launch.  Unsupported cache layouts and quantized caches stay on the
native vLLM path.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_and_cache_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    cos_sin_ptr,
    positions_ptr,
    slot_mapping_ptr,
    query_cache_ptr,
    value_cache_ptr,
    query_stride_t,
    query_stride_h,
    query_stride_d,
    key_stride_t,
    key_stride_h,
    key_stride_d,
    value_stride_t,
    value_stride_h,
    value_stride_d,
    cos_sin_stride_t,
    cache_stride_b,
    cache_stride_t,
    cache_stride_h,
    cache_stride_d,
    num_query_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    block_size: tl.constexpr,
    TILE_D: tl.constexpr,
    IS_NEOX: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, TILE_D)
    d_valid = d < head_dim
    position = tl.load(positions_ptr + token).to(tl.int64)
    slot = tl.load(slot_mapping_ptr + token).to(tl.int64)
    active = slot >= 0

    # Q is written in place. The cache update is only needed for KV heads.
    q_head_active = head < num_query_heads
    q_ptrs = (
        query_ptr
        + token * query_stride_t
        + head * query_stride_h
        + d * query_stride_d
    )
    q = tl.load(q_ptrs, mask=d_valid & q_head_active, other=0.0).to(tl.float32)
    rot_valid = d < rotary_dim
    half = rotary_dim // 2
    if IS_NEOX:
        pair = d % half
        cos = tl.load(
            cos_sin_ptr + position * cos_sin_stride_t + pair,
            mask=rot_valid,
            other=1.0,
        ).to(tl.float32)
        sin = tl.load(
            cos_sin_ptr + position * cos_sin_stride_t + half + pair,
            mask=rot_valid,
            other=0.0,
        ).to(tl.float32)
        partner = tl.where(d < half, d + half, d - half)
        x_partner = tl.load(
            q_ptrs + (partner - d) * query_stride_d,
            mask=rot_valid & q_head_active,
            other=0.0,
        ).to(tl.float32)
        rotated = tl.where(
            d < half,
            q * cos - x_partner * sin,
            q * cos + x_partner * sin,
        ).to(tl.float32)
    else:
        pair = d // 2
        cos = tl.load(
            cos_sin_ptr + position * cos_sin_stride_t + pair,
            mask=rot_valid,
            other=1.0,
        ).to(tl.float32)
        sin = tl.load(
            cos_sin_ptr + position * cos_sin_stride_t + half + pair,
            mask=rot_valid,
            other=0.0,
        ).to(tl.float32)
        partner = tl.where((d % 2) == 0, d + 1, d - 1)
        x_partner = tl.load(
            q_ptrs + (partner - d) * query_stride_d,
            mask=rot_valid & q_head_active,
            other=0.0,
        ).to(tl.float32)
        rotated = tl.where(
            (d % 2) == 0,
            q * cos - x_partner * sin,
            q * cos + x_partner * sin,
        )
    q_out = tl.where(rot_valid, rotated, q)
    tl.store(q_ptrs, q_out, mask=d_valid & q_head_active)

    kv_head_active = head < num_kv_heads
    key_ptrs = (
        key_ptr
        + token * key_stride_t
        + head * key_stride_h
        + d * key_stride_d
    )
    value_ptrs = (
        value_ptr
        + token * value_stride_t
        + head * value_stride_h
        + d * value_stride_d
    )
    k = tl.load(key_ptrs, mask=d_valid & kv_head_active, other=0.0).to(tl.float32)
    v = tl.load(value_ptrs, mask=d_valid & kv_head_active, other=0.0)
    rot_valid = d < rotary_dim
    half = rotary_dim // 2
    if IS_NEOX:
        pair = d % half
        cos = tl.load(
            cos_sin_ptr + position * cos_sin_stride_t + pair,
            mask=rot_valid,
            other=1.0,
        ).to(tl.float32)
        sin = tl.load(
            cos_sin_ptr + position * cos_sin_stride_t + half + pair,
            mask=rot_valid,
            other=0.0,
        ).to(tl.float32)
        partner = tl.where(d < half, d + half, d - half)
        x_partner = tl.load(
            key_ptrs + (partner - d) * key_stride_d,
            mask=rot_valid & kv_head_active,
            other=0.0,
        ).to(tl.float32)
        k_rot = tl.where(
            d < half,
            k * cos - x_partner * sin,
            k * cos + x_partner * sin,
        ).to(tl.float32)
    else:
        pair = d // 2
        cos = tl.load(
            cos_sin_ptr + position * cos_sin_stride_t + pair,
            mask=rot_valid,
            other=1.0,
        ).to(tl.float32)
        sin = tl.load(
            cos_sin_ptr + position * cos_sin_stride_t + half + pair,
            mask=rot_valid,
            other=0.0,
        ).to(tl.float32)
        partner = tl.where((d % 2) == 0, d + 1, d - 1)
        x_partner = tl.load(
            key_ptrs + (partner - d) * key_stride_d,
            mask=rot_valid & kv_head_active,
            other=0.0,
        )
        k_rot = tl.where(
            (d % 2) == 0,
            k * cos - x_partner * sin,
            k * cos + x_partner * sin,
        )

    block = slot // block_size
    offset = slot % block_size
    cache_ptrs = (
        query_cache_ptr
        + block * cache_stride_b
        + offset * cache_stride_t
        + head * cache_stride_h
        + d * cache_stride_d
    )
    value_cache_ptrs = (
        value_cache_ptr
        + block * cache_stride_b
        + offset * cache_stride_t
        + head * cache_stride_h
        + d * cache_stride_d
    )
    tl.store(
        cache_ptrs,
        tl.where(rot_valid, k_rot, k),
        mask=d_valid & active & kv_head_active,
    )
    tl.store(value_cache_ptrs, v, mask=d_valid & active & kv_head_active)


def fused_rope_and_cache(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    rotary_dim: int | None = None,
    num_warps: int = 4,
) -> None:
    """Rotate Q/K and update an NHD paged KV cache in one Triton launch."""
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("query/key/value must have shape [tokens, heads, head_dim]")
    if key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError("only NHD paged KV cache is supported")
    if query.device != key.device or query.device != value.device:
        raise ValueError("query/key/value must share a device")
    if key_cache.device != query.device or value_cache.device != query.device:
        raise ValueError("cache tensors must share the query device")
    if positions.device != query.device or slot_mapping.device != query.device or cos_sin_cache.device != query.device:
        raise ValueError("positions, slots, and RoPE cache must share the query device")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError("query/key/value must share a dtype")
    if key_cache.shape != value_cache.shape:
        raise ValueError("key/value cache shapes must match")
    if key_cache.stride() != value_cache.stride():
        raise ValueError("key/value cache strides must match")
    if query.dtype not in (torch.bfloat16, torch.float16) or key_cache.dtype != query.dtype or value_cache.dtype != query.dtype:
        raise ValueError("only unquantized BF16/FP16 caches are supported")
    if positions.dtype not in (torch.int32, torch.int64) or slot_mapping.dtype not in (torch.int32, torch.int64):
        raise ValueError("positions and slot_mapping must contain integer indices")
    if positions.numel() != query.shape[0] or slot_mapping.numel() != query.shape[0]:
        raise ValueError("positions and slot_mapping must have one entry per token")
    if cos_sin_cache.ndim != 2 or cos_sin_cache.shape[1] % 2:
        raise ValueError("cos_sin_cache must be [max_position, rotary_dim]")

    tokens, query_heads, head_dim = query.shape
    kv_heads = key.shape[1]
    rotary_dim = rotary_dim or int(cos_sin_cache.shape[1])
    if (
        key.shape[0] != tokens
        or value.shape != key.shape
        or key_cache.shape[2] != kv_heads
        or key_cache.shape[3] != head_dim
        or rotary_dim != cos_sin_cache.shape[1]
        or rotary_dim <= 0
        or rotary_dim > head_dim
        or rotary_dim % 2
        or query_heads < kv_heads
        or head_dim not in (64, 128, 256)
        or key_cache.shape[1] % 16
    ):
        raise ValueError("unsupported RoPE/cache shape")
    tile_d = triton.next_power_of_2(head_dim)
    grid = (tokens, max(query_heads, kv_heads))
    _rope_and_cache_kernel[grid](
        query,
        key,
        value,
        cos_sin_cache,
        positions,
        slot_mapping,
        key_cache,
        value_cache,
        query.stride(0), query.stride(1), query.stride(2),
        key.stride(0), key.stride(1), key.stride(2),
        value.stride(0), value.stride(1), value.stride(2),
        cos_sin_cache.stride(0),
        key_cache.stride(0), key_cache.stride(1),
        key_cache.stride(2), key_cache.stride(3),
        num_query_heads=query_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        block_size=key_cache.shape[1],
        TILE_D=tile_d,
        IS_NEOX=is_neox,
        num_warps=num_warps,
    )


__all__ = ["fused_rope_and_cache"]
