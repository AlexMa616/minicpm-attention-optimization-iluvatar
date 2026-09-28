"""
Iluvatar Attention Dispatcher

Main entry point for vLLM attention operations on Iluvatar BI-V150
Routes to optimized kernels based on feature flags and workload characteristics
"""

import os
from typing import Optional

import torch

# Import baseline vLLM unified attention
try:
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    VLLM_BASELINE_AVAILABLE = True
except ImportError:
    VLLM_BASELINE_AVAILABLE = False

# Import optimized kernels
from .triton_unified_attention_optimized import (
    unified_attention_optimized,
    ENABLE_SPLIT_KV,
    ENABLE_ROPE_FUSED,
    ENABLE_WARP_SPEC,
    ENABLE_PINGPONG,
    ENABLE_TMA,
)


# Global flag to enable/disable optimized path
USE_OPTIMIZED = os.environ.get("ILUVATAR_USE_OPTIMIZED", "1") == "1"


def iluvatar_attention(
    # Standard vLLM attention API
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    seqused_k: torch.Tensor,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool = True,
    window_size: tuple = (-1, -1),
    block_table: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    # Split-KV parameters
    seq_threshold_3D: Optional[int] = None,
    num_par_softmax_segments: Optional[int] = None,
    softmax_segm_output: Optional[torch.Tensor] = None,
    softmax_segm_max: Optional[torch.Tensor] = None,
    softmax_segm_expsum: Optional[torch.Tensor] = None,
    # Additional parameters
    alibi_slopes: Optional[torch.Tensor] = None,
    output_scale: Optional[torch.Tensor] = None,
    qq_bias: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    mm_prefix_range: Optional[tuple] = None,
    use_alibi_sqrt: bool = False,
    kv_quant_mode: int = 0,
    k_scale_cache: Optional[torch.Tensor] = None,
    v_scale_cache: Optional[torch.Tensor] = None,
    chunk_lookback: int = -1,
    use_td: bool = False,
    # Optimization overrides (from harness)
    block_m_override: Optional[int] = None,
    block_q_override: Optional[int] = None,
    prefill_tile_override: Optional[int] = None,
    launch_num_warps_override: Optional[int] = None,
    launch_num_stages_override: Optional[int] = None,
    # RoPE parameters
    cos_cache: Optional[torch.Tensor] = None,
    sin_cache: Optional[torch.Tensor] = None,
    rotary_dim: Optional[int] = None,
) -> torch.Tensor:
    """
    Iluvatar attention dispatcher

    Routes to optimized kernels when enabled, falls back to vLLM baseline otherwise

    Decision logic:
    1. Check USE_OPTIMIZED flag
    2. Check if any optimization feature is enabled
    3. Verify workload is supported (no quantization, no alibi, standard window)
    4. Route to unified_attention_optimized() or baseline

    Args:
        Same as vLLM unified_attention API
        Additional overrides for benchmarking/tuning

    Returns:
        out: Filled output tensor [total_tokens, num_q_heads, head_dim]
    """
    # Fast path checks for unsupported features
    use_baseline = (
        not USE_OPTIMIZED
        or not any([ENABLE_SPLIT_KV, ENABLE_TMA, ENABLE_PINGPONG, ENABLE_ROPE_FUSED])
        or softcap != 0.0
        or kv_quant_mode != 0
        or alibi_slopes is not None
        or window_size != (-1, -1)
        or q_descale is not None
        or sinks is not None
        or use_td
    )

    if use_baseline:
        if not VLLM_BASELINE_AVAILABLE:
            raise RuntimeError(
                "Optimized path disabled and vLLM baseline unavailable. "
                "Enable optimizations with ILUVATAR_USE_OPTIMIZED=1"
            )

        return unified_attention(
            q=q, k=k, v=v, out=out,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            block_table=block_table,
            softcap=softcap,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            seq_threshold_3D=seq_threshold_3D,
            num_par_softmax_segments=num_par_softmax_segments,
            softmax_segm_output=softmax_segm_output,
            softmax_segm_max=softmax_segm_max,
            softmax_segm_expsum=softmax_segm_expsum,
            alibi_slopes=alibi_slopes,
            output_scale=output_scale,
            qq_bias=qq_bias,
            sinks=sinks,
            mm_prefix_range=mm_prefix_range,
            use_alibi_sqrt=use_alibi_sqrt,
            kv_quant_mode=kv_quant_mode,
            k_scale_cache=k_scale_cache,
            v_scale_cache=v_scale_cache,
            chunk_lookback=chunk_lookback,
            use_td=use_td,
        )

    # Optimized path
    try:
        return unified_attention_optimized(
            q=q, k=k, v=v, out=out,
            block_table=block_table,
            seqused_k=seqused_k,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            block_m_override=block_m_override,
            block_q_override=block_q_override,
            prefill_tile_override=prefill_tile_override,
            launch_num_warps_override=launch_num_warps_override,
            launch_num_stages_override=launch_num_stages_override,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            rotary_dim=rotary_dim,
            num_splits_override=num_par_softmax_segments,
        )

    except Exception as e:
        # Fallback to baseline on any error
        if VLLM_BASELINE_AVAILABLE:
            import warnings
            warnings.warn(
                f"Optimized attention failed, falling back to baseline: {e}"
            )
            return unified_attention(
                q=q, k=k, v=v, out=out,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                seqused_k=seqused_k,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                block_table=block_table,
            )
        else:
            raise


# Export for vLLM dispatcher
__all__ = ['iluvatar_attention']
