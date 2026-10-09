# JL2026 MiniCPM5-2B Iluvatar 优化阶段总结与交接

日期：2026-10-09（Asia/Shanghai）。状态：**阶段收口，旧环境不可用，等待新环境继续**。

本文供新同学review和接手。事实以保存的源码、CSV、路由事件及实验记录为准；
推断、历史结果和待验收事项分别标注。不是最终比赛成绩或完整上线验收报告。

## 1. 当前结论

1. 已实现Native 2D attention候选及框架opt-in接入，保持causal、paged-KV、GQA=8和FP32累积。
2. 历史prefill-heavy短测有较大改善，但不能作为官方decode主导负载的最终增益。
3. 历史BF16 candidate完整Level 3为 **102/105 = 97.14%**，满足本项目>=95%门槛，不覆盖后续实验补丁或新机器。
4. Decode-B8 kernel-only有明显收益，服务A/B尚未证实稳定增益。
5. 最新隔离取证确认vendor backend已加载、Native门控却回退。需先验证caller/guard一致性，再继续性能实验。
6. 旧机器访问返回资产权限403。最后一轮逐项guard结果未取回，最终恢复状态未确认；不推断服务损坏或实验已完成。

## 2. 仓库与代码边界

| 内容 | 归属 | 验收状态 |
|---|---|---|
| Native 2D算子、vendor接入、dispatch配置 | `AlexMa616/vllm-plugin-FL` 的 `jl2026-native-2d` | 正式候选基线，需新环境重验 |
| 官方benchmark、必要kernel harness、启动工具 | 同上 | 保留；不改变官方脚本 |
| A/B、profiler、调度扫描、路由诊断、失败原型 | 本实验仓库 | 留痕，不作为默认生产路径 |
| 完整记录、原始CSV/JSON、trace、阶段总结 | 本实验仓库 | 部分旧远端文件未下载 |
| 未提交Decode源码修改 | 本仓库 `experiments/patches/` | 默认不应用，无正式服务收益验收 |

收口前远程正式分支代码锚点：`02badf74e8a10aabc6f401407608c097ca30b163`。
本地开发分支：`fp8-kv-native2d-experiment`，HEAD
`edcb12819f6f87e0d6440559058eb8db9dae3fcc`，另外有两处未提交源码修改。
`edcb128` 相对 `02badf7` 只增加实验工具/说明，不是新的算子优化验收提交。

主要历史提交：

- `a175b28`：Split-KV原型。
- `cdc5fea`：Native 2D opt-in接入。
- `6ae58e1`：Native 2D候选收敛。
- `02badf7`：保持官方benchmark不变。
- `edcb128`：补充实验工具与归档说明，未同步到原正式分支。

本轮只做归档/文档收口，不把未验证源码补丁合入正式分支。
正式分支原先仍包含独立 `experiments/native_2d_tuning/` 和交叉A/B脚本，
不能把收口前的树称为“完全只有算子代码”。迁移不修改算子行为；
迁移后的commit使用上述代码锚点对比生产源码。

正式源码入口：

```text
vllm_fl/dispatch/backends/vendor/iluvatar/iluvatar.py
vllm_fl/dispatch/backends/vendor/iluvatar/impl/attention.py
vllm_fl/dispatch/backends/vendor/iluvatar/impl/ops/triton_unified_attention_native.py
vllm_fl/dispatch/config/iluvatar.yaml
tools/native_2d_harness.py
tools/start_native_2d_service.sh
```

## 3. 算子实际做了什么

采用tiled QK -> online softmax -> PV路径。按 `block_table` 访问paged-KV，
通过 `cu_seqlens_q` 处理变长query，保留causal mask、GQA=8映射及FP32累积，
不改KV-cache ABI。冻结配置：

```text
BLOCK_M=128
TILE_SIZE=16
num_warps=8
num_stages=2
pipeline_stages=1
scalar_block_lookup=False（生产）
mixed_dual_launch=False（生产）
```

这是测得的参数组合，不代表重写了softmax数学算法。
BLOCK_M影响每CTA的query行数、CTA数量及资源压力，不能简单解释为减少kernel launch次数。
Triton stages是编译hint，不能据此宣称实现了硬件异步、warp specialization或显式双缓冲。
kernel保留实验参数不意味着dispatcher默认开启或已经验收。

## 4. 从0到当前的实验主线

| 阶段 | 主要工作 | 结论 |
|---|---|---|
| 9月26日 | 固定版本、runtime、插件入口、模型/评测准备、基线 | 建立环境；解决runtime不匹配，不改变赛题源码基线 |
| 9月27-29日 | GEMM微基准、mixed profiler、Split-KV ABI/正确性/服务验证 | mixed attention占比高；Split-KV没有稳定服务收益 |
| 9月29-30日 | Native 2D参数化、路由修复、短A/B、并行扫描 | 冻结128/16/8/2/1，不固化scalar lookup |
| 9月30日 | dual-launch数值/设备/profile/服务验证 | 四组收益 -2.89/-4.88/+1.23/-3.95%，否决当前设计 |
| 9月30日-10月5日 | 官方candidate、调度扫描、异常清理、完整Level 3 | 保存candidate绝对结果；准确率97.14% |
| 10月5-8日 | FP8服务对照、FP8/INT8 kernel消融 | 未获稳定服务收益，停止当前量化线 |
| 10月8-9日 | Decode-B8精确grid、kernel-only、GPU4服务A/B | kernel快，但服务增益未证实 |
| 10月9日 | 排队诊断、真实route/Graph取证 | backend已加载，guard回退；最后逐项smoke未取回 |

早期RoPE+KV-cache standalone原型有数值测试，未合法接入服务、没有服务收益。
未做出MetaX官方C500最终优化成绩，不能混入非官方硬件观测。

## 5. 有效结果与不能宣称的内容

### 5.1 历史Native 2D与官方candidate

早期prefill-heavy短A/B观察约 **4K +90%、16K +209%~+211%**。
输入/输出/并发及当时路由来源均应回看完整记录，不能写成官方最终增益。

2026-09-30官方口径candidate两组各四轮完成，后三轮汇总total tok/s为
4K **2427.74**、16K **1841.89**。没有同窗口官方baseline，不能据此计算新增益。

2026-10-05干净candidate完整官方复核：

| 官方shape `[input,output,concurrency,prompts]` | 四轮output tok/s | 状态 |
|---|---|---|
| `[4096,1024,64,256]` | 372.37 / 484.71 / 478.27 / 503.18 | 全部成功，首轮按预热处理 |
| `[16384,1024,64,128]` | 106.88 / 108.42 / 108.29 / 108.37 | 全部成功；Mean TTFT后三轮约273秒 |

这是历史candidate绝对值，不是新的同窗口A/B。
不混淆output和total tok/s；不将约599秒TTFT的后续Decode实验与约273秒的
旧candidate当成相同环境结果。

2026-10-05完整Level 3：105/105完成，**102/105正确，97.14%**。
早期101/104部分结果不能代替完整准确率。外层exit文件曾有转义错误，
正式结果以EvalScope JSON/HTML报告为准。旧报告只有远端路径记录，
不应暗示本次已重新下载或在新机器复现。

### 5.2 Decode kernel-only

| c64 / median ms | upstream | Native-default | Decode-B8 |
|---|---:|---:|---:|
| 4K | 2.2373 | 1.4427 | 1.0482 |
| 16K | 8.8295 | 5.5816 | 4.3857 |

c8/c16/c32/c64、4K/16K的B8对照最大绝对误差均记录为0；
相对upstream kernel median降幅约49.6%-55.2%。这是所测输入上的kernel证据，
不是全模型正确性证明或服务收益。

原型使用 `BLOCK_M=8/BLOCK_Q=1`、每sequence一个CTA的精确grid，
复用 `MIXED_DECODE_ONLY` 分支，不是已经接入的Decode-3D。

### 5.3 GPU4/9034 Decode-off/on服务A/B

固定Native 2D prefill开关和其他服务参数，比较对象不是赛题原始baseline。
两种配置各8行raw，全部请求成功，每组四轮跳过第一轮。

| 场景 | off/on total tok/s | 差异 | off/on Median ITL ms | off/on P99 ITL ms |
|---|---|---:|---|---|
| 4K/c64 | 1966.85 / 2008.67 | +2.1262% | 92.18 / 91.88 | 758.24 / 754.89 |
| 16K/c64 | 915.56 / 915.36 | -0.0218% | 139.77 / 139.86 | 3307.09 / 3266.77 |

4K逐轮（非随机配对，仅描述性对照）：

| Run | off/on total tok/s | 差异 | off/on Median ITL ms | off/on P99 ITL ms |
|---|---|---:|---|---|
| 1，丢弃 | 1704.45 / 1622.29 | -4.82% | 92.02 / 91.76 | 833.04 / 856.64 |
| 2 | 1959.64 / 1979.44 | +1.01% | 92.25 / 92.19 | 773.58 / 762.45 |
| 3 | 1914.21 / 2017.62 | +5.40% | 91.96 / 91.56 | 752.23 / 747.34 |
| 4 | 2026.69 / 2028.95 | +0.11% | 92.33 / 91.90 | 748.91 / 754.87 |

4K稳态三轮CV：off 2.877%、on 1.291%。单次顺序A/B、仅三轮稳态且
无该轮实际路径的完整证明，不支持“稳定+2.13%”结论。
16K未观察到服务收益，不足以否定已经测得的kernel-only改善。

### 5.4 16K短输出排队诊断

每组输出128，prompts等于并发，单轮、不同seed；prefix caching开启，命中增量均0。

| c | 实际采样秒 | KV峰值 | running/waiting峰值 | 抢占增量 | 总tok/s |
|---|---:|---:|---|---:|---:|
| 8 | 145.7 | 25.22% | 8 / 7 | 0 | 1037.89 |
| 24 | 392.1 | 53.37% | 17 / 23 | 0 | 1063.89 |
| 32 | 509.3 | 53.37% | 17 / 31 | 0 | 1080.84 |
| 64 | 952.7 | 53.37% | 17 / 63 | 0 | 1140.39 |

| c | TTFT median/P95/P99秒 | ITL median/P95/P99秒 | 重建到末token median/P95/P99秒 |
|---|---|---|---|
| 8 | 71.136 / 116.478 / 121.019 | 0.0445 / 2.7443 / 3.8053 | 126.534 / 127.202 / 127.250 |
| 24 | 175.682 / 326.874 / 351.128 | 1.0588 / 3.1274 / 3.6066 | 368.464 / 372.240 / 372.411 |
| 32 | 232.582 / 437.025 / 465.067 | 1.3245 / 3.0740 / 3.6274 | 464.987 / 488.533 / 488.770 |
| 64 | 466.388 / 874.250 / 910.643 | 1.5066 / 3.3212 / 3.8209 | 692.408 / 925.913 / 926.437 |

“重建到末token”来自TTFT+sum(ITL)，不是原生完整E2E。
c8不足五分钟，其他窗口也包含入场/退场，不可称严格稳态。
不同prompts的单轮total tok/s不能作为正式吞吐比较。
短测不支持“满池/抢占重算主导”归因；旧输出1024负载有满KV日志，两者不矛盾。
不能仅凭KV或抢占阈值断言kernel优化无用。

## 6. 最新路由取证与未决问题

### 已确认事实

- 隔离副本实际backend/metadata为 `IluvatarAttentionBackend/IluvatarAttentionMetadata`。
- 实际 `max_num_seqs=256`、`max_num_batched_tokens=2048`、chunked prefill开启，没有配置硬上限17。
- Graph capture的35个不同batch记录 `decode_only=True`，但 `native_supported=False/native=False`。
- 真实prefill/mixed host路由也记录回退，不是“忘开decode开关”或“vendor未加载”。
- mixed trace连续8步：336次上游 `kernel_unified_attention`，13.659668/14.707141秒，Self CUDA占比92.88%；GEMM 6.11%。仅适用于该mixed窗口。
- 纯decode trace见8个 `cudaGraphLaunch`，graph内模型kernel未展开，不能据graph外采样占比判断模型attention/GEMM占比。
- 第一轮runner.exit=0，恢复服务smoke200；后续逐项guard smoke启动后失联，结果未取回。

### 源码疑点，尚非真实逐项guard验收

Native门控要求 `q_descale/k_descale/v_descale is None`，而非per-token-head的
auto/BF16调用分支无条件传 `layer._k_scale.expand(...)` 和 `layer._v_scale.expand(...)`。
本地合成guard可仅因k/v_descale复现回退。

隔离修复候选是仅在确实需要descale的量化路径传scale，BF16/auto传None；
实施前须确认上游scale语义、所有KV模式和调用者契约，不能直接放宽guard或
丢弃仍有意义的scale。**当前没有将这项修复合入正式分支。**

一次性 `_NATIVE_2D_ROUTE_LOGGED` 不能证明每次decode命中，也不能以缺少decode日志
证明未命中。需关联backend来源、guard、step metadata、capture/replay和kernel。
`max_query_len==1` 的专用路由不覆盖所有mixed step，真实执行机会比例仍需统计。

## 7. 未采用方案与保留理由

| 方向 | 已知结果 | 当前决定 |
|---|---|---|
| Split-KV | 做过ABI修复、正确性、服务验证，无稳定服务收益 | 历史留档，不恢复默认 |
| dual-launch | 已完成GPU编译、数值、profiler、服务A/B，三组回归 | 当前设计否决，不误称未实验 |
| scalar/tile/block/stages扫描 | 没有稳定的新服务收益 | 保持冻结配置 |
| exact q-block metadata | 少量CTA节省，metadata约1.27-1.58ms | 不接入生产 |
| 调度2048->4096 | 16K/c64短测吞吐+8.48%，Median/P99 ITL约+78% | 不作为正式配置；官方参数保持一致 |
| 普通FP8 cache | BI-V150通用路径硬件断言失败 | 不能直接当设备基线 |
| FP8 per-token-head | 可运行但服务无稳定收益；kernel约BF16的1.99倍时间 | 停止当前设计 |
| INT8 kernel | 14.6149ms vs BF16 13.7151ms，约慢6.56% | 不进行INT8服务A/B |
| FlagGems/GEMM | 历史小M比Torch慢；纯decode设备占比未知 | 先补decode归因，不开启新线 |
| Decode-3D | 未完成服务验收，当前B8不是3D | 路由问题优先，不继续堆参数 |

## 8. 旧环境与新环境恢复

历史环境仅作复现线索，**不是新机器已确认配置**：

```text
旧主机：ub39；容器：mllv；盘点时16张BI-V150
镜像：harbor.baai.ac.cn/plugin/iluvatar-corex4.5.0-flagtree0.6.0-triton3.6.0-cxnone-vllm_fl0.24.0:20260909
CoreX：4.5.0；Torch历史核验：2.10.0
vllm-plugin-FL赛题锚点：13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0
FlagGems记录：v5.3.5 / a7620cc191a0b42e040194622c5758b22a7a25dc
模型：/workspace/MiniCPM5-2B
数据集：/workspace/evalscope-datasets/math_500
插件入口：VLLM_PLUGINS=fl
```

新环境按门槛推进，不复用旧PID、socket、GPU号、marker或后台恢复任务：

1. 记录新资产、GPU/显存/空闲状态、runtime、镜像digest、依赖版本和模型/数据集来源。
2. 检查runtime兼容和插件加载，安装固定版本，不盲目升级或合并新框架。
3. 克隆正式分支，实验仓库只供资料和隔离工具，原型补丁不自动apply。
4. 建baseline/candidate独立工作区，记录commit和源码哈希，用官方参数启动单一隔离健康服务。
5. 运行真实逐项guard smoke，证实拒绝项后在隔离副本修复scale契约。
6. BF16 upstream/Native对拍：tail、GQA=8、causal、非连续paged-KV、pure prefill/decode/mixed；保存误差和容差依据。
7. 重启并重新capture Graph，确认capture路由和设备路径；旧graph不会因修改host源码自动更新。
8. profiler仅作诊断，分别获取mixed和纯decode证据，避免graph可见性缺口导致误归因。
9. 通过后做同配置/同负载多轮交叉A/B，保存raw、请求成功、TTFT/ITL和prefix命中，短测不过则停止。
10. 稳定后跑官方两组各4轮、跳首轮，baseline/candidate在可比窗口复测，不能跨机器直接算提升。
11. 单独完整Level 3，>=95%，保存报告与退出结果，无残留负载后才算正式性能。
12. 验收通过才将最小源码修复和必要测试提交正式分支，其余证据继续放本仓库。

官方负载 `[input_len, output_len, concurrency, num_prompts]`：
`[[4096,1024,64,256],[16384,1024,64,128]]`，最后一项不是输出长度。
现有benchmark无 `--runs/--skip-first`，四轮/跳首轮由常量控制。

旧环境遇到过CoreX不匹配、extension构建失败、残留客户端高load、PID1不回收zombie、
其他服务占同卡等问题。新机器先盘点和版本验证，不照搬旧绕过命令。
SSH仅有界检查；资产403要恢复新资产授权，不是调keepalive就能解决。

## 9. 资料完整性与交接清单

- 完整记录另存 `experiments/records/2026-10-09-experiment-log.md`，保留历史推断及后续更正。
- `evidence/` 保存能从本地找到的GPU4 A/B、queue诊断及第一轮route probe原件。
- 编译缓存、模型权重、凭据不上传；已下载trace保留压缩原件。
- 其他早期CSV、完整Level 3报告和最后guard smoke只有记录或旧远端路径，未取回就明确缺失。
- 原始证据SHA256列于 `evidence/SHA256SUMS`，不证明尚缺的远端数据。
- 未提交Decode补丁：`experiments/patches/2026-10-09-native-decode-unvalidated.patch`，不是正式生产修复。
- 本阶段不计算比赛总分，不宣称正式官方稳定增益，不将历史准确率套到新环境。

**接手后第一个可验收目标：取得真实guard证据，确认scale契约，完成隔离正确性与路由复核。**
在此之前不重启FP8、dual-launch、GEMM或Decode-3D优化线。
