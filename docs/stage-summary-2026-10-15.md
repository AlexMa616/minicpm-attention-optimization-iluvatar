# JL2026 MiniCPM5-2B Iluvatar 最终阶段总结

日期：2026-10-15（Asia/Shanghai）。状态：**研究阶段收口，正式候选冻结**。

本文是实验仓库当前的主入口。完整日志、原始 JSON/CSV、trace 和失败方案仍保留在本仓库；生产算子和框架接入只保留在 [vllm-plugin-FL 的 `jl2026-native-2d` 分支](https://github.com/AlexMa616/vllm-plugin-FL/tree/jl2026-native-2d)。

## 最终采用方案

- Native 2D attention：paged-KV、causal、GQA、FP32 累积，不改变 KV-cache ABI。
- 候选配置：`BLOCK_M=128`、`TILE_SIZE=64`、`num_warps=8`、`num_stages=2`、`pipeline_stages=1`。
- 源码默认值仍为 TILE16；TILE64 通过候选服务配置显式启用，本阶段没有改生产默认值。
- 生产分支不包含 Decode-B8、FP8 KV cache、动态 shape 分派或 stages3 实验代码。

## 已取得的正式证据

### 准确率

TILE64 两轮 Level 3 合计 `204/210 = 97.14%`，达到项目约定的 `>=95%` 门槛。TILE16 仅作为工程对照记录，不作为 TILE64 的非劣效门槛。

BF16 新种子混合误差界和 RMSE 判据通过。全元素绝对误差 `1e-3` 不作为门槛，因为该阈值与部分 BF16 输出格式下界不兼容。

### 服务 A/B

同源码、同卡、同官方场景的 TILE16/TILE64 对照结果：

| 场景 | TILE64 相对 TILE16 的 total tok/s 增益 |
|---|---:|
| 4K / c64 | `+5.46%` |
| 16K / c64 | `+14.93%` |

早期 prefill-heavy 或不同窗口的绝对吞吐数据不作为官方比赛增益；官方 baseline、工程 TILE16 对照和候选绝对值必须分开记录。

### 纯 decode 取证

- profiling 已进入实际 `vllm_fl.worker.model_runner.ModelRunnerFL`。
- 4K、16K 两个上下文均完成纯 decode trace，客户端正常退出，trace 完整。
- 可识别 attention 约占设备 trace 的 `1.16%`；主要时间由其他 kernel 构成。
- 未观察到可作为正式证据的 Native Decode 专用 kernel 命中。

因此 Decode-B8 的 kernel-only 结果只能作为研究线索，不能转写为服务收益或正式候选收益。

### Shape / pipeline 扫描

共覆盖 54 个唯一配置，包含 BLOCK_M、TILE_SIZE、warps、stages 和 pipeline 组合。唯一看似更好的 stages3 配置在同一 GPU 配对复测中仅为：

- 4K：`+0.11%`
- 16K：`+0.04%`

该幅度视为测量噪声，不进入正式候选，也不做动态分派。

## 未采用方向

1. Decode-B8/Decode-3D：没有完成可复现的服务级稳定收益闭环。
2. FP8/INT8 KV：量化误差预算和服务收益没有形成可提交闭环。
3. 动态 shape 分派：没有测出足以抵消复杂度的稳定交叉点。
4. stages3、Split-KV、dual-launch、RoPE/TMA/warp specialization：未形成正式服务收益或硬件契约证据。

这些脚本、补丁、失败日志和原始证据均保留在实验仓库，仅供 review，不应自动应用到生产分支。

## 最终边界与提交状态

- 正式代码仓库：`AlexMa616/vllm-plugin-FL`，分支 `jl2026-native-2d`。
- 实验仓库：`AlexMa616/minicpm-attention-optimization-iluvatar`，保存本文、完整实验记录和证据归档。
- 两个仓库均通过普通提交同步；不使用 force push。
- 当前阶段的剩余工作只有：审查生产 diff、确认比赛提交格式、整理技术报告和 PR，不再新增未经验证的算子优化线。
