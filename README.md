# MiniCPM Attention Optimization for Iluvatar BI-V150

[English](#english) | [中文](#chinese)

---

<a name="english"></a>

## English

High-performance attention kernel optimizations for **MiniCPM5-2B** inference on **Iluvatar BI-V150 GPU**, achieving 1.5-2.5x throughput improvements through advanced tiling strategies and memory optimization techniques.

### 🎯 Performance Targets

| Workload | Baseline | Optimized | Speedup |
|----------|----------|-----------|---------|
| **Prefill (2K tokens)** | ~17 TFLOP/s | **25-30 TFLOP/s** | **1.5-1.8x** |
| **Mixed Batch (14K tokens)** | ~7.25 TFLOP/s | **12-15 TFLOP/s** | **1.7-2.0x** |
| **Long Context (8K+)** | Limited by SM utilization | Split-KV parallelism boost | **2.0-2.5x** |
| **Hopper GPU (TMA)** | N/A | Theoretical additional gain | **+10-15%** |

### 🔧 Technical Approach

#### Core Optimizations (7-Week Implementation)

1. **Split-KV 3D Tiling** (Week 1-2)
   - Partition KV dimension into K chunks for independent processing
   - Increase parallelism: O(B×H) → O(B×H×K)
   - Online softmax reduction for numerical stability

2. **RoPE Fusion** (Week 3)
   - Inline rotary position encoding during attention computation
   - Reduce HBM bandwidth by avoiding separate RoPE pass

3. **Warp Specialization** (Week 4)
   - Producer-consumer pattern with async prefetch
   - Overlap memory loads with computation
   - Explicit warp-level barrier synchronization

4. **Pingpong Scheduling** (Week 5)
   - Double-buffering with A/B staging areas
   - Hide memory latency behind compute

5. **TMA Integration** (Week 6)
   - Hopper architecture acceleration (SM 9.0+)
   - Tensor Memory Accelerator for efficient global→shared transfers
   - Automatic fallback for non-Hopper GPUs

6. **Unified Routing + Autotuning** (Week 7)
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
git clone https://github.com/YOUR_USERNAME/minicpm-attention-optimization-iluvatar.git
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

## 中文

针对**Iluvatar BI-V150 GPU**上**MiniCPM5-2B**推理的高性能Attention kernel优化，通过先进的分块策略和内存优化技术实现1.5-2.5倍吞吐量提升。

### 🎯 性能目标

| 工作负载 | 基线 | 优化后 | 加速比 |
|----------|------|--------|--------|
| **Prefill (2K tokens)** | ~17 TFLOP/s | **25-30 TFLOP/s** | **1.5-1.8x** |
| **混合批次 (14K tokens)** | ~7.25 TFLOP/s | **12-15 TFLOP/s** | **1.7-2.0x** |
| **长上下文 (8K+)** | 受SM利用率限制 | Split-KV并行度提升 | **2.0-2.5x** |
| **Hopper GPU (TMA)** | N/A | 理论额外增益 | **+10-15%** |

### 🔧 技术方案

#### 核心优化（7周实现计划）

1. **Split-KV 3D分块** (第1-2周)
   - 将KV维度切分为K个块独立处理
   - 并行度提升：O(B×H) → O(B×H×K)
   - 在线softmax归约保证数值稳定性

2. **RoPE融合** (第3周)
   - 在attention计算中内联旋转位置编码
   - 避免单独的RoPE pass，减少HBM带宽

3. **Warp专用化** (第4周)
   - 生产者-消费者模式的异步预取
   - 内存加载与计算重叠
   - 显式warp级barrier同步

4. **Pingpong调度** (第5周)
   - A/B双缓冲暂存区
   - 计算掩盖内存延迟

5. **TMA集成** (第6周)
   - Hopper架构加速（SM 9.0+）
   - 张量内存加速器优化全局→共享内存传输
   - 非Hopper GPU自动降级

6. **统一路由 + 自动调优** (第7周)
   - 基于工作负载特征的智能kernel选择
   - 缓存(seq_q, seq_k, head_dim)配置
   - Feature flag控制与优先级排序

### 🚀 快速开始

#### 1. 安装

```bash
# 克隆仓库
git clone https://github.com/YOUR_USERNAME/minicpm-attention-optimization-iluvatar.git
cd minicpm-attention-optimization-iluvatar

# 安装依赖
pip install torch triton vllm

# 将kernel复制到vLLM插件目录
./scripts/install_to_vllm.sh
```

#### 2. 启用优化

```bash
# 启用全部优化
export ILUVATAR_USE_OPTIMIZED=1
export ILUVATAR_SPLIT_KV=1
export ILUVATAR_ROPE_FUSED=1
export ILUVATAR_PINGPONG=1
export ILUVATAR_TMA=1  # 仅Hopper GPU

# 运行vLLM服务
python -m vllm.entrypoints.api_server \
  --model openbmb/MiniCPM5-2B \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9
```

#### 3. 独立性能测试

```bash
cd tests

# 测试Split-KV优化
VLLM_ILUVATAR_ATTN_SPLIT_KV_MIXED=1 \
python attention_harness.py \
  --config configs/attention_test.json \
  --device cuda:0 \
  --result ../experiments/results/split_kv.jsonl \
  --split-kv-mixed \
  --split-kv-segments 4
```

### 📚 文档

- [架构设计](docs/architecture.md) - 详细技术设计
- [性能分析](docs/performance.md) - Benchmark结果与瓶颈分析
- [集成指南](docs/integration-guide.md) - vLLM插件安装
- [硬件特性](docs/hardware-profile.md) - BI-V150参数

### 📄 许可证

MIT License - 详见[LICENSE](LICENSE)
