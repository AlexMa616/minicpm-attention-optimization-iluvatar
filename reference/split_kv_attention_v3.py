"""
Split-KV Attention Triton Kernels for FlashAttention-3 Level Optimization
=============================================================================

Implements 3D tiling strategy that splits KV dimension across GPU blocks:
- Parallelism: num_heads × batch × K (instead of num_heads × batch)
- Online softmax reduction for numerical stability
- Support for GQA, Paged-KV cache, and causal masking

Author: Generated for MiniCPM5-2B Throughput Optimization
Date: 2026-09-28
"""

import torch
import triton
import triton.language as tl
from typing import Optional


# ============================================================================
# Stage 1: Split-KV Forward Kernel (3D Tiling)
# ============================================================================

@triton.jit
def _split_kv_forward_kernel(
    # Input tensors
    Q_ptr, K_cache_ptr, V_cache_ptr,
    BlockTable_ptr, ContextLens_ptr,
    # Output buffers (per-split intermediate results)
    O_splits_ptr, M_splits_ptr, L_splits_ptr,
    # Strides for Q [batch, num_q_heads, seq_len_q, head_dim]
    stride_q_batch, stride_q_head, stride_q_seq, stride_q_dim,
    # Strides for K/V cache [num_blocks, num_kv_heads, block_size, head_dim]
    stride_kc_block, stride_kc_head, stride_kc_seq, stride_kc_dim,
    stride_vc_block, stride_vc_head, stride_vc_seq, stride_vc_dim,
    # Strides for block_table [batch, max_num_blocks]
    stride_bt_batch, stride_bt_block,
    # Strides for outputs [num_splits, batch, num_q_heads, num_q_blocks, BLOCK_M, head_dim]
    stride_o_split, stride_o_batch, stride_o_head, stride_o_qblock, stride_o_qseq, stride_o_dim,
    stride_m_split, stride_m_batch, stride_m_head, stride_m_qblock, stride_m_qseq,
    stride_l_split, stride_l_batch, stride_l_head, stride_l_qblock, stride_l_qseq,
    # Dimensions
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    seq_len_q: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    # Tunable parameters
    num_splits: tl.constexpr,
    BLOCK_M: tl.constexpr,  # Q tiling size
    BLOCK_N: tl.constexpr,  # KV tiling size
    BLOCK_DMODEL: tl.constexpr,
    GQA_RATIO: tl.constexpr,
):
    """
    Split-KV forward kernel: compute partial attention for one KV split.

    Grid: (batch, num_q_heads, num_q_blocks, num_splits)

    Each block:
    1. Loads Q tile [BLOCK_M, head_dim]
    2. Iterates over assigned KV split range
    3. Computes partial QK^T, softmax, and output
    4. Stores (O_partial, m_partial, l_partial) for reduction

    Online softmax formula:
        m_new = max(m_old, m_local)
        l_new = exp(m_old - m_new) * l_old + exp(m_local - m_new) * l_local
        O_new = (exp(m_old - m_new) * l_old * O_old + exp(m_local - m_new) * l_local * O_local) / l_new
    """
    # Program IDs
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_q_block = tl.program_id(2)
    pid_split = tl.program_id(3)

    # GQA: map Q head to KV head
    kv_head_idx = pid_head // GQA_RATIO

    # Get context length for this sequence
    ctx_len = tl.load(ContextLens_ptr + pid_batch)
    num_kv_blocks = (ctx_len + block_size - 1) // block_size

    # Compute KV split range for this split
    kv_per_split = (ctx_len + num_splits - 1) // num_splits
    kv_start = pid_split * kv_per_split
    kv_end = tl.minimum(kv_start + kv_per_split, ctx_len)

    if kv_start >= kv_end:
        # This split has no work, write -inf to m and return
        offs_m = tl.arange(0, BLOCK_M)
        m_ptr = M_splits_ptr + (
            pid_split * stride_m_split +
            pid_batch * stride_m_batch +
            pid_head * stride_m_head +
            pid_q_block * stride_m_qblock +
            offs_m * stride_m_qseq
        )
        tl.store(m_ptr, -1e9, mask=offs_m < BLOCK_M)
        return

    # Load Q tile [BLOCK_M, head_dim]
    q_start = pid_q_block * BLOCK_M
    offs_q_seq = q_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = Q_ptr + (
        pid_batch * stride_q_batch +
        pid_head * stride_q_head +
        offs_q_seq[:, None] * stride_q_seq +
        offs_d[None, :] * stride_q_dim
    )
    q_mask = (offs_q_seq[:, None] < seq_len_q) & (offs_d[None, :] < head_dim)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Initialize accumulators
    acc_o = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    acc_m = tl.full([BLOCK_M], -1e9, dtype=tl.float32)
    acc_l = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Iterate over KV blocks in this split's range
    kv_block_start = kv_start // block_size
    kv_block_end = (kv_end + block_size - 1) // block_size

    for kv_block_idx in range(kv_block_start, kv_block_end):
        # Get physical block number from block_table
        block_table_ptr = BlockTable_ptr + pid_batch * stride_bt_batch + kv_block_idx * stride_bt_block
        physical_block_num = tl.load(block_table_ptr)

        # Compute KV tile range within this block
        kv_tile_start = tl.maximum(kv_start - kv_block_idx * block_size, 0)
        kv_tile_end = tl.minimum(kv_end - kv_block_idx * block_size, block_size)

        # Load K tile [BLOCK_N, head_dim]
        offs_kv_seq = kv_tile_start + tl.arange(0, BLOCK_N)
        k_ptrs = K_cache_ptr + (
            physical_block_num * stride_kc_block +
            kv_head_idx * stride_kc_head +
            offs_kv_seq[:, None] * stride_kc_seq +
            offs_d[None, :] * stride_kc_dim
        )
        k_mask = (offs_kv_seq[:, None] < kv_tile_end) & (offs_d[None, :] < head_dim)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # Compute QK^T [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, tl.trans(k)) * scale

        # Apply causal mask if needed
        offs_q_abs = q_start + tl.arange(0, BLOCK_M)
        offs_kv_abs = kv_block_idx * block_size + kv_tile_start + tl.arange(0, BLOCK_N)
        causal_mask = offs_q_abs[:, None] >= offs_kv_abs[None, :]
        qk = tl.where(causal_mask, qk, -1e9)

        # Online softmax update
        m_local = tl.max(qk, axis=1)  # [BLOCK_M]
        m_new = tl.maximum(acc_m, m_local)

        # Compute alpha (renormalization factors)
        alpha_old = tl.exp(acc_m - m_new)
        alpha_local = tl.exp(m_local - m_new)

        # Update l (denominator)
        p_local = tl.exp(qk - m_local[:, None])  # [BLOCK_M, BLOCK_N]
        l_local = tl.sum(p_local, axis=1)  # [BLOCK_M]
        l_new = alpha_old * acc_l + alpha_local * l_local

        # Load V tile [BLOCK_N, head_dim]
        v_ptrs = V_cache_ptr + (
            physical_block_num * stride_vc_block +
            kv_head_idx * stride_vc_head +
            offs_kv_seq[:, None] * stride_vc_seq +
            offs_d[None, :] * stride_vc_dim
        )
        v = tl.load(v_ptrs, mask=k_mask, other=0.0)

        # Compute local output
        o_local = tl.dot(p_local.to(v.dtype), v)  # [BLOCK_M, head_dim]

        # Update accumulated output
        acc_o = (alpha_old[:, None] * acc_l[:, None] * acc_o +
                 alpha_local[:, None] * l_local[:, None] * o_local) / l_new[:, None]

        # Update accumulators
        acc_m = m_new
        acc_l = l_new

    # Store results for this split
    # O_splits [num_splits, batch, num_q_heads, num_q_blocks, BLOCK_M, head_dim]
    o_ptrs = O_splits_ptr + (
        pid_split * stride_o_split +
        pid_batch * stride_o_batch +
        pid_head * stride_o_head +
        pid_q_block * stride_o_qblock +
        offs_q_seq[:, None] * stride_o_qseq +
        offs_d[None, :] * stride_o_dim
    )
    tl.store(o_ptrs, acc_o.to(o_ptrs.dtype.element_ty), mask=q_mask)

    # M_splits [num_splits, batch, num_q_heads, num_q_blocks, BLOCK_M]
    m_ptrs = M_splits_ptr + (
        pid_split * stride_m_split +
        pid_batch * stride_m_batch +
        pid_head * stride_m_head +
        pid_q_block * stride_m_qblock +
        offs_q_seq * stride_m_qseq
    )
    tl.store(m_ptrs, acc_m, mask=offs_q_seq < seq_len_q)

    # L_splits [num_splits, batch, num_q_heads, num_q_blocks, BLOCK_M]
    l_ptrs = L_splits_ptr + (
        pid_split * stride_l_split +
        pid_batch * stride_l_batch +
        pid_head * stride_l_head +
        pid_q_block * stride_l_qblock +
        offs_q_seq * stride_l_qseq
    )
    tl.store(l_ptrs, acc_l, mask=offs_q_seq < seq_len_q)


# ============================================================================
# Stage 2: Online Softmax Reduction Kernel
# ============================================================================

@triton.jit
def _online_softmax_reduce_kernel(
    # Input split results
    O_splits_ptr, M_splits_ptr, L_splits_ptr,
    # Output
    O_ptr,
    # Strides for split buffers [num_splits, batch, num_q_heads, num_q_blocks, BLOCK_M, head_dim]
    stride_o_split, stride_o_batch, stride_o_head, stride_o_qblock, stride_o_qseq, stride_o_dim,
    stride_m_split, stride_m_batch, stride_m_head, stride_m_qblock, stride_m_qseq,
    stride_l_split, stride_l_batch, stride_l_head, stride_l_qblock, stride_l_qseq,
    # Strides for output [batch, num_q_heads, seq_len_q, head_dim]
    stride_out_batch, stride_out_head, stride_out_seq, stride_out_dim,
    # Dimensions
    num_splits: tl.constexpr,
    seq_len_q: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """
    Reduce split results using online softmax formula.

    Grid: (batch, num_q_heads, num_q_blocks)

    For each Q block, reduce across all splits:
        m_global = max(m_0, m_1, ..., m_{K-1})
        l_global = sum(exp(m_i - m_global) * l_i)
        O_global = sum(exp(m_i - m_global) * l_i * O_i) / l_global
    """
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_q_block = tl.program_id(2)

    q_start = pid_q_block * BLOCK_M
    offs_q_seq = q_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Initialize global accumulators
    acc_o = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    acc_m = tl.full([BLOCK_M], -1e9, dtype=tl.float32)
    acc_l = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Reduce across splits
    for split_idx in range(num_splits):
        # Load split results
        m_ptr = M_splits_ptr + (
            split_idx * stride_m_split +
            pid_batch * stride_m_batch +
            pid_head * stride_m_head +
            pid_q_block * stride_m_qblock +
            offs_q_seq * stride_m_qseq
        )
        m_split = tl.load(m_ptr, mask=offs_q_seq < seq_len_q, other=-1e9)

        l_ptr = L_splits_ptr + (
            split_idx * stride_l_split +
            pid_batch * stride_l_batch +
            pid_head * stride_l_head +
            pid_q_block * stride_l_qblock +
            offs_q_seq * stride_l_qseq
        )
        l_split = tl.load(l_ptr, mask=offs_q_seq < seq_len_q, other=0.0)

        o_ptrs = O_splits_ptr + (
            split_idx * stride_o_split +
            pid_batch * stride_o_batch +
            pid_head * stride_o_head +
            pid_q_block * stride_o_qblock +
            offs_q_seq[:, None] * stride_o_qseq +
            offs_d[None, :] * stride_o_dim
        )
        o_mask = (offs_q_seq[:, None] < seq_len_q) & (offs_d[None, :] < head_dim)
        o_split = tl.load(o_ptrs, mask=o_mask, other=0.0)

        # Online reduction
        m_new = tl.maximum(acc_m, m_split)
        alpha_old = tl.exp(acc_m - m_new)
        alpha_split = tl.exp(m_split - m_new)

        l_new = alpha_old * acc_l + alpha_split * l_split

        # Avoid division by zero
        safe_l_new = tl.where(l_new > 0, l_new, 1.0)

        acc_o = (alpha_old[:, None] * acc_l[:, None] * acc_o +
                 alpha_split[:, None] * l_split[:, None] * o_split) / safe_l_new[:, None]

        acc_m = m_new
        acc_l = l_new

    # Store final output
    out_ptrs = O_ptr + (
        pid_batch * stride_out_batch +
        pid_head * stride_out_head +
        offs_q_seq[:, None] * stride_out_seq +
        offs_d[None, :] * stride_out_dim
    )
    out_mask = (offs_q_seq[:, None] < seq_len_q) & (offs_d[None, :] < head_dim)
    tl.store(out_ptrs, acc_o.to(out_ptrs.dtype.element_ty), mask=out_mask)


# ============================================================================
# Python Interface: Split-KV Attention Forward
# ============================================================================

def split_kv_attention_forward(
    query: torch.Tensor,  # [batch, num_q_heads, seq_len_q, head_dim]
    key_cache: torch.Tensor,  # [num_blocks, num_kv_heads, block_size, head_dim]
    value_cache: torch.Tensor,
    block_table: torch.Tensor,  # [batch, max_num_blocks]
    context_lens: torch.Tensor,  # [batch]
    scale: float,
    num_splits: int = 4,
    BLOCK_M: int = 16,
    BLOCK_N: int = 64,
) -> torch.Tensor:
    """
    Split-KV attention forward pass with online softmax reduction.

    Args:
        query: Query tensor [batch, num_q_heads, seq_len_q, head_dim]
        key_cache: Paged key cache [num_blocks, num_kv_heads, block_size, head_dim]
        value_cache: Paged value cache (same layout as key_cache)
        block_table: Block mapping [batch, max_num_blocks]
        context_lens: Actual context length per sequence [batch]
        scale: Attention scale factor (typically 1/sqrt(head_dim))
        num_splits: Number of KV splits (parallelism factor)
        BLOCK_M: Q tiling size
        BLOCK_N: KV tiling size

    Returns:
        Output tensor [batch, num_q_heads, seq_len_q, head_dim]
    """
    batch, num_q_heads, seq_len_q, head_dim = query.shape
    _, num_kv_heads, block_size, _ = key_cache.shape

    assert head_dim in [64, 128, 256], "head_dim must be 64, 128, or 256"
    assert num_q_heads % num_kv_heads == 0, "GQA: num_q_heads must be divisible by num_kv_heads"

    gqa_ratio = num_q_heads // num_kv_heads
    num_q_blocks = (seq_len_q + BLOCK_M - 1) // BLOCK_M

    # Allocate intermediate buffers
    o_splits = torch.zeros(
        (num_splits, batch, num_q_heads, num_q_blocks, BLOCK_M, head_dim),
        dtype=torch.float32, device=query.device
    )
    m_splits = torch.full(
        (num_splits, batch, num_q_heads, num_q_blocks, BLOCK_M),
        -1e9, dtype=torch.float32, device=query.device
    )
    l_splits = torch.zeros(
        (num_splits, batch, num_q_heads, num_q_blocks, BLOCK_M),
        dtype=torch.float32, device=query.device
    )

    # Output tensor
    output = torch.empty_like(query)

    # Launch split-KV forward kernel
    grid = (batch, num_q_heads, num_q_blocks, num_splits)

    _split_kv_forward_kernel[grid](
        query, key_cache, value_cache, block_table, context_lens,
        o_splits, m_splits, l_splits,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
        block_table.stride(0), block_table.stride(1),
        o_splits.stride(0), o_splits.stride(1), o_splits.stride(2), o_splits.stride(3), o_splits.stride(4), o_splits.stride(5),
        m_splits.stride(0), m_splits.stride(1), m_splits.stride(2), m_splits.stride(3), m_splits.stride(4),
        l_splits.stride(0), l_splits.stride(1), l_splits.stride(2), l_splits.stride(3), l_splits.stride(4),
        num_q_heads, num_kv_heads, seq_len_q, block_size, head_dim, scale,
        num_splits, BLOCK_M, BLOCK_N, head_dim, gqa_ratio,
    )

    # Launch reduction kernel
    grid_reduce = (batch, num_q_heads, num_q_blocks)

    _online_softmax_reduce_kernel[grid_reduce](
        o_splits, m_splits, l_splits, output,
        o_splits.stride(0), o_splits.stride(1), o_splits.stride(2), o_splits.stride(3), o_splits.stride(4), o_splits.stride(5),
        m_splits.stride(0), m_splits.stride(1), m_splits.stride(2), m_splits.stride(3), m_splits.stride(4),
        l_splits.stride(0), l_splits.stride(1), l_splits.stride(2), l_splits.stride(3), l_splits.stride(4),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        num_splits, seq_len_q, head_dim, BLOCK_M, head_dim,
    )

    return output
