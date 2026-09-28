# MiniCPM Attention Optimization for Iluvatar BI-V150

---

<a name="english"></a>


High-performance attention kernel optimizations for **MiniCPM5-2B** inference on **Iluvatar BI-V150 GPU**, achieving 1.5-2.5x throughput improvements through advanced tiling strategies and memory optimization techniques.

### 🎯 Performance Targets

| Workload | Baseline | Optimized | Speedup |
|----------|----------|-----------|---------|
| **Prefill (2K tokens)** | ~17 TFLOP/s | **25-30 TFLOP/s** | **1.5-1.8x** |
| **Mixed Batch (14K tokens)** | ~7.25 TFLOP/s | **12-15 TFLOP/s** | **1.7-2.0x** |
| **Long Context (8K+)** | Limited by SM utilization | Split-KV parallelism boost | **2.0-2.5x** |
| **Hopper GPU (TMA)** | N/A | Theoretical additional gain | **+10-15%** |

### 🔧 Technical Approach

#### Core Optimizations (Implementation)

1. **Split-KV 3D Tiling** 
   - Partition KV dimension into K chunks for independent processing
   - Increase parallelism: O(B×H) → O(B×H×K)
   - Online softmax reduction for numerical stability

2. **RoPE Fusion** 
   - Inline rotary position encoding during attention computation
   - Reduce HBM bandwidth by avoiding separate RoPE pass

3. **Warp Specialization** 
   - Producer-consumer pattern with async prefetch
   - Overlap memory loads with computation
   - Explicit warp-level barrier synchronization

4. **Pingpong Scheduling** 
   - Double-buffering with A/B staging areas
   - Hide memory latency behind compute

5. **TMA Integration** 
   - Hopper architecture acceleration (SM 9.0+)
   - Tensor Memory Accelerator for efficient global→shared transfers
   - Automatic fallback for non-Hopper GPUs

6. **Unified Routing + Autotuning** 
   - Intelligent kernel selection based on workload characteristics
   - Cached configuration for (seq_q, seq_k, head_dim) signatures
   - Feature flag control with priority ordering

### 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│  vLLM Service (Iluvatar Backend)                        │
├─────────────────────────────────────────────────────────┤
│  attention.py (Dispatcher)                              │
│  ├─ Feature detection (TMA, Split-KV, RoPE, Pingpong)  │
│  ├─ Workload analysis (seq_len, context_len, batch)    │
│  └─ Fallback to baseline if unsupported                │
├─────────────────────────────────────────────────────────┤
│  unified_attention_optimized() (Main Entry)             │
│  ├─ KernelSelector: choose optimal kernel              │
│  ├─ AutotuneConfig: cached tuning parameters           │
│  └─ Route to: TMA / Split-KV / RoPE / Pingpong         │
├─────────────────────────────────────────────────────────┤
│  Triton JIT Kernels                                     │
│  ├─ _split_kv_forward_kernel (3D tiling)               │
│  ├─ _split_kv_reduce_kernel (online softmax)           │
│  ├─ _rope_fused_attention_kernel                       │
│  ├─ _warp_specialized_attention_kernel                 │
│  ├─ _pingpong_attention_kernel (double-buffer)         │
│  └─ _tma_attention_kernel (Hopper)                     │
└─────────────────────────────────────────────────────────┘
```

### 🚀 Quick Start

#### 1. Installation

```bash
# Clone repository
git clone https://github.com/AlexMa616/minicpm-attention-optimization-iluvatar.git
cd minicpm-attention-optimization-iluvatar

# Install dependencies
pip install torch triton vllm

# Copy kernels to vLLM plugin directory
./scripts/install_to_vllm.sh
```

#### 2. Enable Optimizations

```bash
# Enable all optimizations
export ILUVATAR_USE_OPTIMIZED=1
export ILUVATAR_SPLIT_KV=1
export ILUVATAR_ROPE_FUSED=1
export ILUVATAR_PINGPONG=1
export ILUVATAR_TMA=1  # Hopper GPU only

# Run vLLM service
python -m vllm.entrypoints.api_server \
  --model openbmb/MiniCPM5-2B \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9
```

#### 3. Standalone Benchmarking

```bash
cd tests

# Test Split-KV optimization
VLLM_ILUVATAR_ATTN_SPLIT_KV_MIXED=1 \
python attention_harness.py \
  --config configs/attention_test.json \
  --device cuda:0 \
  --result ../experiments/results/split_kv.jsonl \
  --split-kv-mixed \
  --split-kv-segments 4

# Performance comparison
python attention_harness.py \
  --config configs/attention_test.json \
  --device cuda:0 \
  --result ../experiments/results/baseline_vs_optimized.jsonl \
  --shape mixed \
  --repeats 10
```

### 📊 Hardware Profile

**Iluvatar BI-V150 Specifications:**
- **Compute Units:** 80 Streaming Multiprocessors (SMs)
- **Shared Memory:** 164 KB per SM
- **HBM Bandwidth:** 2000 GB/s
- **FP16 Throughput:** ~160 TFLOPS (theoretical peak)
- **Block Size:** 16 tokens (paged KV cache)

### 📚 Documentation

- [Architecture Design](docs/architecture.md) - Detailed technical design
- [Performance Analysis](docs/performance.md) - Benchmark results and bottleneck analysis
- [Integration Guide](docs/integration-guide.md) - vLLM plugin installation
- [Hardware Profile](docs/hardware-profile.md) - BI-V150 characteristics

### 🔗 References

This project implements custom optimizations inspired by:
- **FlashAttention-2:** Dao et al., "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
- **FlashAttention-3:** Shah et al., "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision"
- **vLLM:** Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention"

### 📝 Citation

If you use this work in your research, please cite:

```bibtex
@software{minicpm_attention_optimization_2026,
  title = {MiniCPM Attention Optimization for Iluvatar BI-V150},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/YOUR_USERNAME/minicpm-attention-optimization-iluvatar}
}
```

### 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

<a name="chinese"></a>
- [硬件特性](docs/hardware-profile.md) - BI-V150参数

### 📄 许可证

MIT License - 详见[LICENSE](LICENSE)
