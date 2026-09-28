"""
Split-KV Attention Kernels with Online Softmax Reduction

FlashAttention-3 inspired 3D tiling strategy:
- Splits KV dimension into chunks, enabling parallelism: num_heads × batch × K_splits
- Each split computes partial attention outputs with local max/sum statistics
- Final reduction uses numerically stable online softmax formula

Mathematical Proof of Correctness:
Given two partial softmax results:
  - Split A: exp(scores_A - m_A) with max=m_A, sum=l_A, output=O_A
  - Split B: exp(scores_B - m_B) with max=m_B, sum=l_B, output=O_B

Unified result:
  m_new = max(m_A, m_B)
  l_new = exp(m_A - m_new) * l_A + exp(m_B - m_new) * l_B
  O_new = [exp(m_A - m_new) * l_A * O_A + exp(m_B - m_new) * l_B * O_B] / l_new

This is exactly equivalent to computing softmax over the full concatenated [scores_A, scores_B].
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _split_kv_forward_kernel(
    # Input tensors
    Q,  # [batch, num_q_heads, seq_len_q, head_dim]
    K_cache,  # [num_blocks, num_kv_heads, block_size, head_dim]
    V_cache,  # [num_blocks, num_kv_heads, block_size, head_dim]
    block_tables,  # [batch, max_num_blocks]
    context_lens,  # [batch]
    # Output buffers (per-split statistics)
    Out_splits,  # [num_splits, batch, num_q_heads, num_q_blocks, BLOCK_M, head_dim]
    M_splits,  # [num_splits, batch, num_q_heads, num_q_blocks, BLOCK_M] (max values)
    L_splits,  # [num_splits, batch, num_q_heads, num_q_blocks, BLOCK_M] (sum exp values)
    # Strides
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob0, stride_ob1, stride_oh, stride_oqb, stride_om, stride_od,
    stride_mb0, stride_mb1, stride_mh, stride_mqb, stride_mm,
    stride_lb0, stride_lb1, stride_lh, stride_lqb, stride_lm,
    # Dimensions
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    seq_len_q: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    # Tiling config
    num_splits: tl.constexpr,
    BLOCK_M: tl.constexpr,  # Query block size (e.g., 16)
    BLOCK_N: tl.constexpr,  # KV block size (e.g., 64)
    BLOCK_DMODEL: tl.constexpr,  # head_dim must match
    GQA_RATIO: tl.constexpr,  # num_q_heads / num_kv_heads
):
    """
    Split-KV forward kernel - computes partial attention for one KV split.

    Grid: (batch, num_q_heads, num_q_blocks, num_splits)
    Each block processes:
      - BLOCK_M queries (along seq_len_q)
      - One split of the KV sequence (context_len / num_splits)
      - Outputs partial O, M, L statistics for later reduction

    Key optimizations:
      - Minimizes HBM access via block-level tiling
      - Accumulates in FP32 for numerical stability
      - Supports GQA with proper head indexing
      - Handles causal masking and variable context lengths
    """
    # Program ID
    batch_idx = tl.program_id(0)
    q_head_idx = tl.program_id(1)
    q_block_idx = tl.program_id(2)
    split_idx = tl.program_id(3)

    # Derive KV head index (GQA)
    kv_head_idx = q_head_idx // GQA_RATIO

    # Load context length for this sequence
    context_len = tl.load(context_lens + batch_idx)

    # Compute KV range for this split
    kv_chunk_size = (context_len + num_splits - 1) // num_splits
    kv_start = split_idx * kv_chunk_size
    kv_end = tl.minimum(kv_start + kv_chunk_size, context_len)

    # Early exit if this split is out of range
    if kv_start >= context_len:
        return

    # Query tile starting position
    q_start = q_block_idx * BLOCK_M
    q_offs = q_start + tl.arange(0, BLOCK_M)
    q_mask = q_offs < seq_len_q

    # Load Q tile: [BLOCK_M, head_dim]
    q_ptrs = Q + batch_idx * stride_qb + q_head_idx * stride_qh + \
             q_offs[:, None] * stride_qs + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)

    # Initialize accumulators (FP32 for numerical stability)
    acc_o = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    acc_m = tl.full([BLOCK_M], value=-1e9, dtype=tl.float32)  # max scores
    acc_l = tl.zeros([BLOCK_M], dtype=tl.float32)  # sum exp scores

    # Iterate over KV blocks in this split
    num_kv_blocks = (kv_end - kv_start + BLOCK_N - 1) // BLOCK_N
    for kv_block_iter in range(num_kv_blocks):
        kv_offset = kv_start + kv_block_iter * BLOCK_N
        kv_offs = kv_offset + tl.arange(0, BLOCK_N)
        kv_mask = kv_offs < kv_end

        # Map KV offset to paged block
        # block_idx = kv_offset // block_size
        # offset_in_block = kv_offset % block_size
        block_idx = kv_offs // block_size
        offset_in_block = kv_offs % block_size

        # Load block numbers from block_table
        block_table_ptrs = block_tables + batch_idx * tl.cdiv(context_len, block_size) + block_idx
        block_numbers = tl.load(block_table_ptrs, mask=kv_mask, other=0)

        # Load K tile: [BLOCK_N, head_dim]
        k_ptrs = K_cache + block_numbers[:, None] * stride_kb + \
                 kv_head_idx * stride_kh + \
                 offset_in_block[:, None] * stride_ks + \
                 tl.arange(0, BLOCK_DMODEL)[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0)

        # Load V tile: [BLOCK_N, head_dim]
        v_ptrs = V_cache + block_numbers[:, None] * stride_vb + \
                 kv_head_idx * stride_vh + \
                 offset_in_block[:, None] * stride_vs + \
                 tl.arange(0, BLOCK_DMODEL)[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0)

        # Compute attention scores: Q @ K^T
        qk = tl.dot(q, tl.trans(k)) * scale  # [BLOCK_M, BLOCK_N]

        # Apply causal mask (prefill) - only mask if q_pos < kv_pos
        # For decode (seq_len_q=1), no causal masking needed
        if seq_len_q > 1:
            causal_mask = q_offs[:, None] >= kv_offs[None, :]
            qk = tl.where(causal_mask, qk, float('-inf'))

        # Mask out-of-range KV positions
        qk = tl.where(kv_mask[None, :], qk, float('-inf'))

        # Online softmax update
        m_new = tl.maximum(acc_m, tl.max(qk, axis=1))  # [BLOCK_M]

        # Rescale previous accumulator
        alpha_old = tl.exp(acc_m - m_new)
        alpha_new = tl.exp(tl.max(qk, axis=1) - m_new)

        # Update sum
        p = tl.exp(qk - m_new[:, None])  # [BLOCK_M, BLOCK_N]
        l_new = alpha_old * acc_l + tl.sum(p, axis=1)

        # Update output
        acc_o = acc_o * alpha_old[:, None] + tl.dot(p.to(v.dtype), v)

        # Store updated statistics
        acc_m = m_new
        acc_l = l_new

    # Write partial results to global memory
    # Out_splits: [num_splits, batch, num_q_heads, num_q_blocks, BLOCK_M, head_dim]
    out_ptrs = Out_splits + split_idx * stride_ob0 + batch_idx * stride_ob1 + \
               q_head_idx * stride_oh + q_block_idx * stride_oqb + \
               tl.arange(0, BLOCK_M)[:, None] * stride_om + \
               tl.arange(0, BLOCK_DMODEL)[None, :] * stride_od
    tl.store(out_ptrs, acc_o, mask=q_mask[:, None])

    # M_splits: [num_splits, batch, num_q_heads, num_q_blocks, BLOCK_M]
    m_ptrs = M_splits + split_idx * stride_mb0 + batch_idx * stride_mb1 + \
             q_head_idx * stride_mh + q_block_idx * stride_mqb + \
             tl.arange(0, BLOCK_M) * stride_mm
    tl.store(m_ptrs, acc_m, mask=q_mask)

    # L_splits: [num_splits, batch, num_q_heads, num_q_blocks, BLOCK_M]
    l_ptrs = L_splits + split_idx * stride_lb0 + batch_idx * stride_lb1 + \
             q_head_idx * stride_lh + q_block_idx * stride_lqb + \
             tl.arange(0, BLOCK_M) * stride_lm
    tl.store(l_ptrs, acc_l, mask=q_mask)


@triton.jit
def _reduce_splits_kernel(
    # Input: per-split statistics
    Out_splits,  # [num_splits, batch, num_q_heads, num_q_blocks, BLOCK_M, head_dim]
    M_splits,  # [num_splits, batch, num_q_heads, num_q_blocks, BLOCK_M]
    L_splits,  # [num_splits, batch, num_q_heads, num_q_blocks, BLOCK_M]
    # Output: final attention result
    Out,  # [batch, num_q_heads, seq_len_q, head_dim]
    # Strides
    stride_ob0, stride_ob1, stride_oh, stride_oqb, stride_om, stride_od,
    stride_mb0, stride_mb1, stride_mh, stride_mqb, stride_mm,
    stride_lb0, stride_lb1, stride_lh, stride_lqb, stride_lm,
    stride_outb, stride_outh, stride_outs, stride_outd,
    # Dimensions
    num_splits: tl.constexpr,
    seq_len_q: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """
    Reduction kernel - combines partial attention outputs using online softmax.

    Grid: (batch, num_q_heads, num_q_blocks)
    Each block processes BLOCK_M queries and reduces across num_splits.

    Formula:
      m_global = max(m_0, m_1, ..., m_{K-1})
      alpha_i = exp(m_i - m_global)
      l_global = sum(alpha_i * l_i)
      O_global = sum(alpha_i * l_i * O_i) / l_global
    """
    batch_idx = tl.program_id(0)
    q_head_idx = tl.program_id(1)
    q_block_idx = tl.program_id(2)

    q_start = q_block_idx * BLOCK_M
    q_offs = q_start + tl.arange(0, BLOCK_M)
    q_mask = q_offs < seq_len_q

    # Initialize global accumulators
    global_o = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    global_m = tl.full([BLOCK_M], value=-1e9, dtype=tl.float32)
    global_l = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Reduce across splits
    for split_idx in range(num_splits):
        # Load partial O: [BLOCK_M, head_dim]
        o_ptrs = Out_splits + split_idx * stride_ob0 + batch_idx * stride_ob1 + \
                 q_head_idx * stride_oh + q_block_idx * stride_oqb + \
                 tl.arange(0, BLOCK_M)[:, None] * stride_om + \
                 tl.arange(0, BLOCK_DMODEL)[None, :] * stride_od
        o_split = tl.load(o_ptrs, mask=q_mask[:, None], other=0.0)

        # Load partial M: [BLOCK_M]
        m_ptrs = M_splits + split_idx * stride_mb0 + batch_idx * stride_mb1 + \
                 q_head_idx * stride_mh + q_block_idx * stride_mqb + \
                 tl.arange(0, BLOCK_M) * stride_mm
        m_split = tl.load(m_ptrs, mask=q_mask, other=-1e9)

        # Load partial L: [BLOCK_M]
        l_ptrs = L_splits + split_idx * stride_lb0 + batch_idx * stride_lb1 + \
                 q_head_idx * stride_lh + q_block_idx * stride_lqb + \
                 tl.arange(0, BLOCK_M) * stride_lm
        l_split = tl.load(l_ptrs, mask=q_mask, other=0.0)

        # Online softmax reduction
        m_new = tl.maximum(global_m, m_split)

        alpha_global = tl.exp(global_m - m_new)
        alpha_split = tl.exp(m_split - m_new)

        l_new = alpha_global * global_l + alpha_split * l_split

        # Update output
        global_o = global_o * (alpha_global * global_l / l_new)[:, None] + \
                   o_split * (alpha_split * l_split / l_new)[:, None]

        global_m = m_new
        global_l = l_new

    # Normalize and write final output
    final_o = global_o / global_l[:, None]

    out_ptrs = Out + batch_idx * stride_outb + q_head_idx * stride_outh + \
               q_offs[:, None] * stride_outs + \
               tl.arange(0, BLOCK_DMODEL)[None, :] * stride_outd
    tl.store(out_ptrs, final_o.to(Out.dtype.element_ty), mask=q_mask[:, None])


def split_kv_attention(
    query: torch.Tensor,  # [batch, num_q_heads, seq_len_q, head_dim]
    key_cache: torch.Tensor,  # [num_blocks, num_kv_heads, block_size, head_dim]
    value_cache: torch.Tensor,  # [num_blocks, num_kv_heads, block_size, head_dim]
    block_tables: torch.Tensor,  # [batch, max_num_blocks]
    context_lens: torch.Tensor,  # [batch]
    scale: float,
    num_splits: int = 4,
    BLOCK_M: int = 16,
    BLOCK_N: int = 64,
) -> torch.Tensor:
    """
    High-level API for split-KV attention.

    Args:
        query: Query tensor [batch, num_q_heads, seq_len_q, head_dim]
        key_cache: Paged key cache [num_blocks, num_kv_heads, block_size, head_dim]
        value_cache: Paged value cache [num_blocks, num_kv_heads, block_size, head_dim]
        block_tables: Block mapping [batch, max_num_blocks]
        context_lens: Context length per sequence [batch]
        scale: Attention scale factor (1 / sqrt(head_dim))
        num_splits: Number of KV splits (default: 4)
        BLOCK_M: Query block size (default: 16)
        BLOCK_N: KV block size (default: 64)

    Returns:
        Attention output [batch, num_q_heads, seq_len_q, head_dim]
    """
    batch, num_q_heads, seq_len_q, head_dim = query.shape
    num_kv_heads = key_cache.shape[1]
    block_size = key_cache.shape[2]

    assert num_q_heads % num_kv_heads == 0, "GQA: num_q_heads must be divisible by num_kv_heads"
    GQA_RATIO = num_q_heads // num_kv_heads

    # Allocate intermediate buffers
    num_q_blocks = (seq_len_q + BLOCK_M - 1) // BLOCK_M
    out_splits = torch.zeros(
        num_splits, batch, num_q_heads, num_q_blocks, BLOCK_M, head_dim,
        dtype=torch.float32, device=query.device
    )
    m_splits = torch.full(
        (num_splits, batch, num_q_heads, num_q_blocks, BLOCK_M),
        -1e9, dtype=torch.float32, device=query.device
    )
    l_splits = torch.zeros(
        num_splits, batch, num_q_heads, num_q_blocks, BLOCK_M,
        dtype=torch.float32, device=query.device
    )

    # Launch split kernel
    grid = (batch, num_q_heads, num_q_blocks, num_splits)
    _split_kv_forward_kernel[grid](
        query, key_cache, value_cache, block_tables, context_lens,
        out_splits, m_splits, l_splits,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
        out_splits.stride(0), out_splits.stride(1), out_splits.stride(2), out_splits.stride(3), out_splits.stride(4), out_splits.stride(5),
        m_splits.stride(0), m_splits.stride(1), m_splits.stride(2), m_splits.stride(3), m_splits.stride(4),
        l_splits.stride(0), l_splits.stride(1), l_splits.stride(2), l_splits.stride(3), l_splits.stride(4),
        num_q_heads, num_kv_heads, seq_len_q, block_size, head_dim, scale,
        num_splits, BLOCK_M, BLOCK_N, head_dim, GQA_RATIO
    )

    # Allocate output
    output = torch.empty_like(query)

    # Launch reduction kernel
    grid = (batch, num_q_heads, num_q_blocks)
    _reduce_splits_kernel[grid](
        out_splits, m_splits, l_splits, output,
        out_splits.stride(0), out_splits.stride(1), out_splits.stride(2), out_splits.stride(3), out_splits.stride(4), out_splits.stride(5),
        m_splits.stride(0), m_splits.stride(1), m_splits.stride(2), m_splits.stride(3), m_splits.stride(4),
        l_splits.stride(0), l_splits.stride(1), l_splits.stride(2), l_splits.stride(3), l_splits.stride(4),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        num_splits, seq_len_q, head_dim, BLOCK_M, head_dim
    )

    return output
