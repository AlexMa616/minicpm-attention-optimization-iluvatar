# MiniCPM5-2B 推理吞吐量优化实验记录

> 目的：持续记录每一轮实验的目标、思路、证据、改动、结果和下一步，保证代码、技术报告和复现步骤保持一致。
>
> 规则：没有日志或可复现实验支撑的内容标记为 `HYPOTHESIS`，不要把推测写成结论。官方 C500 不可用时，C550 结果只作为兼容性和相对 A/B 参考，不能作为官方成绩。

## 1. 任务约束

- 模型：ModelBest MiniCPM5-2B
- 推理框架：FlagOS `vllm-plugin-FL`，分支 `flagos-2026-s2`
- 算子库：FlagGems，固定 tag `v5.3.5`
- 官方平台：Iluvatar CoreX BI-V150、MetaX C500 64 GB
- 当前可用平台：Iluvatar BI-V150；MetaX C550（非官方 C500，暂作兼容性和相对性能实验）
- 官方启动参数、benchmark 脚本和评估参数必须保持一致
- 禁止：通过 speculative decoding、修改超参数实现量化、修改 benchmark 脚本、移除核心算子选择逻辑、直接合并最新框架分支获取收益
- 精度门槛：`math_500 Level 3` 准确率 `>= 0.95`
- 性能指标：总 token/s；4k 和 16k 场景的 TTFT 不得超过对应基线 1%

## 2. 本地基线状态

记录时间：2026-09-26

### 2.1 代码版本

| 仓库 | 路径 | 分支/tag | commit | 工作区 |
|---|---|---|---|---|
| vllm-plugin-FL | `./vllm-plugin-FL` | `flagos-2026-s2` | `13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0` | clean |
| FlagGems | `./FlagGems` | `v5.3.5` | `a7620cc191a0b42e040194622c5758b22a7a25dc` | clean |

### 2.2 官方基线截图中的数据

| 平台 | 场景 | Total tok/s | Output tok/s | Mean TTFT |
|---|---:|---:|---:|---:|
| Iluvatar BI-V150 | 4k | 2028.010 | 405.600 | 11.573 s |
| Iluvatar BI-V150 | 16k | 915.150 | 53.830 | 599.262 s |
| MetaX C500 | 4k | 5089.645 | 1017.930 | 3.199 s |
| MetaX C500 | 16k | 7029.675 | 413.510 | 27.197 s |

准确率仍需在实际环境重跑确认：文档正文写 `0.962`，示例截图显示 `98.1%`。

## 3. 当前判断与待验证假设

- `FACT`：Iluvatar 16k 的 TTFT 和低 Output tok/s 明显异常，必须先确认是否为 prefill、编译、fallback、显存/KV cache 或调度问题。
- `FACT`：Iluvatar 启动命令包含 `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`，MetaX 命令不包含该参数。
- `INFERENCE`：Iluvatar 16k 可能没有进入正常的稳态 prefill/decode 路径，但当前没有远端日志或 profile，不能定论。
- `HYPOTHESIS H1`：长上下文 prefill 触发编译、graph break 或后端 fallback。
- `HYPOTHESIS H2`：BI-V150 的 16k attention/chunked-prefill kernel 配置不合适。
- `HYPOTHESIS H3`：64 并发长上下文下 KV cache 分配、碎片、swap 或 recompute 造成等待。
- `HYPOTHESIS H4`：continuous batching 在目标场景下出现调度空洞。
- `UNKNOWN`：C550 与官方 C500 的驱动、Torch、Triton、FlagGems backend 版本和性能关系。

最小区分实验：先跑官方原样命令和日志，再做精确 token 长度的单请求/预热后请求，与 64 并发 benchmark 对比。

## 4. 远端环境记录

- 登录目标：`bastion.aiops.baai.ac.cn:2224`
- 登录用户：用户提供的 SSH 登录标识（不在项目文件中持久化）
- 首次探针时间：2026-09-26 15:14:06 CST
- hostname：`ub39`
- 用户：`root`
- 当前目录：`/root`
- Docker：`24.0.7`
- 远端容器名：`mllv`
- 远端工作目录：`UNKNOWN`，必须由首条探针确认
- 赛题是否指定镜像：否；以下镜像是基于现有 BI-V150 环境的实验选择，不是赛题硬性要求
- 候选基础镜像：`harbor.baai.ac.cn/plugin/iluvatar-corex4.5.0-flagtree0.6.0-triton3.6.0-cxnone-vllm_fl0.24.0:20260909`
- 候选镜像已存在于远端：是，约 53.7 GB
- tmux session：`UNKNOWN`，如需 iTerm 实时观察，创建本任务专属 session 后记录名称

认证由用户在 iTerm 中完成；不在本记录中保存密码、MFA 或私钥内容。

### 4.1 Codex 直连尝试

- 时间：2026-09-26
- 目标：通过用户提供的 SSH 目标执行只读探针
- 结果：失败，远端返回 `Permission denied (password,publickey)`
- 远端修改：无
- 原因：Codex 进程未获得用户 iTerm 的密码、MFA、SSH agent 或 ControlMaster
- 下一步：由用户在 iTerm 完成认证并建立可复用 ControlMaster，随后由 Codex 通过 socket 执行显式远端命令

### 4.2 阶段性总结（2026-09-26）

- 已通过 ControlMaster 接入 `ub39`，确认主机为 16 张 Iluvatar BI-V150，Docker `24.0.7`。
- 初始候选镜像内 CoreX runtime 与主机不匹配，导致 Torch 错误 803；已保留旧容器 `mllv-runtime-debug` 作为诊断留档。
- 正式 `mllv` 使用同一 Python/Torch 镜像，并只读挂载主机 `/usr/local/corex-4.5.0`；当前 `torch.cuda.is_available()` 为 `True`，可见设备数为 16。
- 远端已拉取并校验固定版本：`vllm-plugin-FL` commit `13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0`，FlagGems tag `v5.3.5` / commit `a7620cc191a0b42e040194622c5758b22a7a25dc`。
- FlagGems 已 editable 安装；vllm-plugin-FL 的 native extension 构建因 CMake 4.4 与 CoreX nvcc 10.2 隐式链接解析失败，暂按文档最小要求使用 Python-only editable 安装。
- 当前分支 entry point 是 `fl`；`VLLM_PLUGINS=fl` 能识别 `PlatformFL(cuda/iluvatar)`，文档中的 `flvllm` 会回退到 `UnspecifiedPlatform`。这是待确认的环境兼容差异，不是源码修改结论。
- `/workspace` 尚无模型和 `math_500` 数据集；ModelScope、EvalScope 和磁盘空间已确认可用。
- 尚未启动模型服务、运行 benchmark 或评估；尚未修改比赛源码。

### 4.3 实验 2026-09-26-01：固定分支插件入口验证

- 时间：2026-09-26 17:07 CST
- 平台/设备：Iluvatar BI-V150，容器内可见 16 张设备
- 服务器 hostname：`ub39`
- 容器：`mllv`
- 目标：确认固定分支 `flagos-2026-s2` 的实际 `VLLM_PLUGINS` 入口，避免沿用文档中的失效名称启动基线。
- 代码：`vllm-plugin-FL` commit `13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0`；FlagGems `v5.3.5` commit `a7620cc191a0b42e040194622c5758b22a7a25dc`；无源码修改。
- 实际执行：在 `mllv` 中分别以 `VLLM_PLUGINS=fl` 和 `VLLM_PLUGINS=flvllm` 导入 `vllm.platforms.current_platform`，单次导入设置 20 秒硬超时。
- `FACT`：可用平台插件列表只有 `fl -> vllm_fl:register`。
- `FACT`：`VLLM_PLUGINS=fl` 输出 `Platform plugin fl is activated`，当前类为 `PlatformFL`，进程正常退出。
- `FACT`：`VLLM_PLUGINS=flvllm` 未加载平台插件，当前类为 `UnspecifiedPlatform`，进程正常退出。
- `FACT`：Torch `2.10.0`，`torch.cuda.is_available()` 为 `True`，设备数为 `16`。
- 说明：插件接口对 `platform_name` 打印了属性缺失 warning，并返回 `None`；本轮以插件激活状态、类名和 Torch 设备可见性作为判据，不据此误判平台不可用。
- 结论：当前固定分支应使用 `VLLM_PLUGINS=fl`。文档中的 Iluvatar 命令使用 `flvllm` 与当前分支不兼容，属于入口名称差异；启动参数其余部分保持官方原样。
- 追加执行：已启动容器内后台模型下载任务，命令为 `modelscope download --repo-type model --local-dir /workspace/MiniCPM5-2B OpenBMB/MiniCPM5-2B`，日志为 `/workspace/logs/model_download.log`。
- 当前结果：模型元数据已下载，`model-00000-of-00001.safetensors.incomplete` 正在断点下载；模型总权重约 5.03 GB。下载未完成，不能启动服务。
- 遇到的问题：直接前台下载受网络速度波动影响，约 1-3 MB/s，容易占用交互会话。
- 解决办法：中断前台任务并在持久容器内用 `nohup` 后台续传，保持 `/workspace/MiniCPM5-2B` 目标目录不变。
- 当前结果：模型已完成下载，完整权重 `model-00000-of-00001.safetensors` 大小为 `5033557096` 字节；数据集 `AI-ModelScope/MATH-500` 已下载到 `/workspace/evalscope-datasets/math_500`，包含 `test.jsonl`、`eval.yaml` 和元数据文件。
- 下一条最小命令：保持服务不变，等待官方两组 benchmark 完成并读取结果。

### 4.4 实验 2026-09-26-02：官方参数基线服务启动

- 时间：2026-09-26 17:47 CST
- 平台/设备：Iluvatar BI-V150，容器内可见 16 张设备
- 服务器 hostname：`ub39`
- 容器：`mllv`
- 目标：使用官方 Iluvatar 启动参数建立未修改源码的服务基线。
- 启动命令：`VLLM_PLUGINS=fl vllm serve /workspace/MiniCPM5-2B --port 9031 --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' --served-model-name minicpm --gpu-memory-utilization 0.85 --max-model-len 131072`
- 日志路径：`/workspace/logs/minicpm_baseline_server.log`
- `FACT`：服务加载 `PlatformFL`，使用 `/workspace/vllm-plugin-FL/vllm_fl/dispatch/config/iluvatar.yaml`，注册 Iluvatar vendor backends。
- `FACT`：权重加载完成，模型 checkpoint 约 4.69 GiB；KV cache size 为 523,056 tokens。
- `FACT`：首次启动完成 torch.compile、KV cache 初始化和 35 个 FULL decode CUDA graph 捕获，随后监听 `0.0.0.0:9031`。
- `FACT`：简单 chat 请求 HTTP 200，模型名 `minicpm`，请求总耗时约 29.1 秒，返回 221 个 completion tokens；该耗时包含首次请求预热，不作为吞吐结论。
- 注意：日志出现 `vllm._C` 未找到、Gloo hostname fallback 和 FlagGems operator override warning；服务仍成功启动并返回请求，先记录为环境现象，是否影响性能待 benchmark/profile 证据确认。
- 修改文件和位置：无比赛源码修改；仅新增远端日志和模型/数据文件。
- 下一步：等待官方两组 benchmark 完成，读取 `/workspace/benchmark_results/summary_*.csv`；随后停止或保持服务不变，单独执行 Level 3 accuracy。

### 4.5 实验 2026-09-26-03：官方吞吐 benchmark（进行中）

- 启动命令：`python3 /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py --model /workspace/MiniCPM5-2B --served-model-name minicpm --port 9031 --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'`
- 远端进程：PID `7705`
- 日志路径：`/workspace/logs/minicpm_baseline_benchmark.log`
- 结果：进行中，尚未产生 `summary_*.csv`；不能提前填写吞吐或 TTFT。
- 当前状态：截至 2026-09-26 17:51，进程 PID `7705` 仍存活，已运行约 4 分钟；日志文件仍为空，疑似 Python 输出缓冲，不能据此判断 benchmark 失败。
- 处理：不并发发送其他请求，不终止进程，避免污染或破坏官方基线；待进程退出后再一次性读取结果和服务日志。
- 阶段结果：`raw_runs_20260926_175332.csv` 已生成 4k 行：duration `746.1 s`，output `351.35 tok/s`，total `1756.76 tok/s`，mean TTFT `16486.62 ms`，256/256 请求成功。
- 对比官方截图基线：4k 参考为 total `2028.010 tok/s`、output `405.600 tok/s`、mean TTFT `11.573 s`；当前结果暂低于参考，且 TTFT 约 `16.487 s`。该差异只作为当前环境基线事实，尚未归因。
- 当前 16k 进度：服务日志显示 64 并发请求持续运行，生成吞吐约 `650-735 tok/s`，KV cache 使用约 `51%-61%`；由于 16k 输入阶段吞吐约 `1600-3700 tok/s` 且共有 128 个请求，测试仍在进行。
- 脚本事实：`benchmark_throughput_serve.py` 对每个 test case 执行 `RUNS=4` 次并跳过首轮（`SKIP_FIRST=1`）；因此 4k summary 已生成，16k 需要完成四次运行后才会进入最终 summary。
- 后续观察：截至 2026-09-26 18:24，第二批 16k 请求仍保持约 63 个活跃请求并持续返回 HTTP 200；高负载期间部分 `docker exec` 查询出现延迟，未发现服务错误日志，因此不终止 benchmark。
- 追加观察：截至 2026-09-26 19:21，16k benchmark 子进程仍在运行约 40 分钟，服务日志持续有 HTTP 200 和请求完成；等待队列由约 34 降至 6，说明请求在推进而非进程空转。该轮期间 GPU KV cache 使用率曾达到 100%，运行请求约 30-31 个，服务端 EngineCore CPU 占用约 95%。
- `FACT`：benchmark 脚本只在每个 `vllm bench serve` 子进程完整退出后追加一行 CSV；因此 16k 首轮未结束时，raw CSV 和 summary CSV 保持只有 4k 结果是预期行为。
- `FACT`：服务日志出现多次 FlagGems event timing fallback warning，但没有 traceback、请求失败或服务退出证据；当前只能记录为环境现象，不能直接归因于吞吐异常。
- `HYPOTHESIS`：16k 的极长耗时主要与 64 并发下 KV cache 接近满载、prefill/decode 交错调度和长序列请求排队有关；需要完整四轮结果或单独精确复现实验验证，当前不作为优化结论。
- 当前处理：不终止官方 benchmark、不发送额外请求、不改变服务参数，避免污染基线；继续等待首轮自然结束。
- 16k 首轮已完成：128/128 请求成功，耗时 `2562.43 s`，Output `51.15 tok/s`，Total `869.57 tok/s`，Mean TTFT `626659.42 ms`，Mean TPOT `516.60 ms`。该轮结果已追加到 `/workspace/benchmark_results/raw_runs_20260926_175332.csv`。
- 对比官方截图参考（16k Total `915.150 tok/s`、Mean TTFT `599.262 s`）：当前首轮 Total 约低 `4.98%`，Mean TTFT 约高 `4.57%`；这是单轮结果，最终基线仍须按脚本跳过首轮并对后 3 轮取平均，不能用本轮替代 summary。
- `FACT`：首轮总耗时约 42.7 分钟，期间没有失败请求或服务退出；说明长上下文基线可完成，但完整官方四轮可能需要较长时间。
- `FACT`：固定分支的 FlagGems dispatch 默认返回 vLLM `TRITON_ATTN`；仅设置 `VLLM_FL_USE_FLAGGEMS_ATTN=1` 才返回自定义 `AttentionFLBackend`。当前基线未设置该变量，不能将长上下文表现直接归因于 FlagGems 自定义 attention。该切换仅作为后续独立 A/B 候选，尚未验证性能或精度。
- 16k 第 2 轮已完成：128/128 请求成功，耗时 `2427.38 s`，Output `54.00 tok/s`，Total `917.96 tok/s`，Mean TTFT `598154.03 ms`，Mean TPOT `486.12 ms`。结果已追加到 `/workspace/benchmark_results/raw_runs_20260926_175332.csv`；官方脚本随后已启动第 3 轮。
- `FACT`：第 2 轮相对官方截图参考（Total `915.150 tok/s`、Mean TTFT `599.262 s`）在单轮上分别约高 `0.31%`、低 `0.19%`，但最终判定仍以跳过首轮后的 3 轮平均为准。
- 后处理：新增并启动 `run_accuracy_after_baseline.sh`，容器内 PID `16126`，日志 `/workspace/logs/baseline_to_accuracy.log`。脚本当前只等待 benchmark PID `7705` 退出；完成后会校验 raw CSV 的 8 次成功记录和 summary CSV 的两行，再执行固定 Level 3 eval。当前 eval 尚未启动。
- 编排问题：第一次试图在容器内用嵌套 `nohup sh -c` 启动串联 watcher，PID `16070` 随即成为僵尸且未创建日志；官方 benchmark PID `7705` 及第二轮子进程未受影响。原因 `UNKNOWN`（复杂 shell 引号或后台生命周期均未排除），不复用该启动命令。
- 改进方案：本地新增 `run_accuracy_after_baseline.sh`，在容器 detached 执行；它等 benchmark 退出，并用 CSV 结构化校验 4k/16k 各四次原始成功记录和两行 summary，验证通过才运行官方 Level 3 eval。若基线失败则退出并记日志，不自动发起会污染基线的请求。
- 16k 第 3 轮已完成：128/128 请求成功，耗时 `2427.44 s`，Output `54.00 tok/s`，Total `917.93 tok/s`，Mean TTFT `598208.80 ms`，Mean TPOT `485.99 ms`。结果已追加到 `/workspace/benchmark_results/raw_runs_20260926_175332.csv`。
- `FACT`：截至本次续作检查，16k 第 4 轮仍在运行，benchmark 父进程 `7705` 和准确率等待脚本 `16126` 均存活；raw CSV 共 8 行（含表头），说明当前已完成 4k 四轮和 16k 三轮，summary 暂只有 4k 行是脚本行为的正常结果。
- `FACT`：服务健康端点 `http://127.0.0.1:9031/health` 返回 HTTP 200；没有停止服务、benchmark 或修改启动参数。
- `FACT`：第 2、3 轮 16k 结果几乎一致（Total 分别 `917.96` 和 `917.93 tok/s`），当前未见长轮次性能继续恶化的证据；最终基线仍须等待第 4 轮并使用脚本生成的后 3 轮平均值。
- 连接诊断：一次远端状态检查中的 `awk` 过滤表达式发生本地/远端引号转义错误，但只影响诊断输出，不影响任何远端进程或结果文件；后续使用直接 PID 查询，避免复杂嵌套转义。
- 追加状态：第 4 轮 16k 子进程已启动（PID `16851`）；服务日志持续出现请求完成和 HTTP 200，运行请求约 30、等待请求约 34，KV cache 约 `96%`至`98%`。该现象与前 3 轮一致，当前没有卡死或服务退出证据。
- 基线闭环完成：16k 第 4 轮 `128/128 SUCCESS`，耗时 `2428.32 s`，Output `53.98 tok/s`，Total `917.60 tok/s`，Mean TTFT `598405.89 ms`。benchmark 父进程随后退出，自动后处理脚本完成 CSV 校验并启动官方 Level 3 EvalScope。
- 官方基线 summary：4k Total `1983.45 tok/s`、Output `396.69 tok/s`、Mean TTFT `11594.57 ms`；16k Total `917.83 tok/s`、Output `53.99 tok/s`、Mean TTFT `598256.24 ms`。4k/16k 均为后 3 轮平均，全部请求成功。
- Level 3 准确率：输出目录 `/workspace/evalscope-datasets/level3/20260926_212957`，`MATH-500 Level 3` 共 `105` 题，Accuracy `97.1%`，EvalScope exit code `0`；满足准确率门槛 `>=0.95`。
- `FACT`：截至 2026-09-26 22:05，benchmark 和评估进程均已退出，模型服务 PID `2943` 仍监听 `9031`，未停止服务。
- 阶段结论：未修改比赛源码的官方基线已经具备可比较的吞吐和准确率证据。下一阶段可在完全相同 benchmark/eval 条件下做独立 A/B，优先验证 `VLLM_FL_USE_FLAGGEMS_ATTN=1`；每个候选必须单独记录服务日志、吞吐 summary 和 Level 3 结果，只有吞吐提升且 TTFT 不劣化超过 1%、准确率仍不低于 `95%` 才保留。

### 4.7 实验 2026-09-26-05：P0 环境与 KV 容量只读核验

- 时间：2026-09-26 23:00 后
- 目标：在修改源码或启动参数前，核实 `vllm._C`、设备显存、KV cache 结构和抢占证据。
- 实际执行：通过现有 SSH ControlMaster 在 `mllv` 内检查 Python 包、服务 `/metrics`、Iluvatar `ixsmi`、固定安装代码和基线日志；没有重启服务、没有发送模型请求、没有修改比赛源码。
- `FACT`：当前服务 PID `2943` 仍运行在端口 `9031`；请求队列为空，未因本轮诊断新增负载。
- `FACT`：容器内 `vllm==0.24.0+empty`，`vllm._C` 的 module spec 不存在，包目录中没有 vLLM 自有 `.so` 文件；基线启动日志多次记录 `Failed to import from vllm._C`。这确认了环境缺口，但尚未证明它造成吞吐回退。
- `FACT`：容器内没有 `nvidia-smi`；宿主/挂载的 `/usr/local/corex-4.5.0/bin/ixsmi` 可用。BI-V150 共 16 张卡，每张显存 `32768 MiB`；当前服务只占用 GPU 0，查询时 GPU 0 使用约 `28770 MiB`，其余 GPU 基本空闲。当前单实例不会自动把其他卡的显存并入 GPU 0 的 KV 池。
- `FACT`：模型配置为 42 层、2 个 KV heads、head dimension 128、BF16。按 K/V 两份计算，理论 KV 约 `43,008 bytes/token`；当前 `523,056 tokens` KV 池约 `20.95 GiB`（不含块/元数据开销）。4k 场景 64 请求的理论工作集约 `13.13 GiB`，16k 场景约 `44.63 GiB`，后者约为 KV 池的 `2.13x`。
- `FACT`：服务 Prometheus 指标 `vllm:num_preemptions_total` 当前为 `16`。该计数器是服务启动后的累计值，混合了吞吐 benchmark 与 Level 3 评估，不能归因到 16k benchmark；它只能证明服务生命周期内确实发生过抢占。
- `FACT`：基线日志中可确认的 `fallback` 是 FlagGems replay benchmarker 回退到 event timing 的 runtime warning；目前没有具体算子 backend fallback 的证据。
- `FACT`：固定安装的 vLLM 代码中存在 `request.num_preemptions`、`vllm:num_preemptions_total` 和恢复路径，可用于下一轮隔离实验中的抢占计数；当前服务没有保留按 benchmark 分段的计数快照。
- `HYPOTHESIS`：16k 的 KV 容量压力是真实存在的，且与约 30 个请求驻留的数量级一致；但“绝大部分耗时来自抢占重算”尚未被当前累计计数证明。
- `UNKNOWN`：`vllm._C` 缺失是否影响 Iluvatar 路径；需要在不污染官方基线的独立服务实例中验证导入依赖和性能影响，不能直接构建或替换线上服务。
- 下一步最小动作：保存当前基线证据后，创建隔离的诊断服务/日志窗口，清零或记录 benchmark 前后的 `num_preemptions_total` 差值，短测精确 16k 并发场景；若抢占数量与吞吐/TTFT同步变化，再进入 KV 调度或 KV dtype 方案设计。
- 隔离诊断服务：在 GPU 1、端口 `9032` 启动同参数服务 `minicpm-diag`（PID `19936`），GPU 0 的官方服务 `9031` 保持运行且健康。诊断服务独立 metrics 初始 `num_preemptions_total=0`。
- 诊断探针：启动 `[[16384,64,64,32]]`（32 个请求、16k 输入、64 输出）的缩短 benchmark。运行中观察到 `num_requests_running` 约 `1`至`7`、`num_requests_waiting` 约 `25`至`31`、等待原因为 `capacity`，KV 使用率约 `8.6%`至`21.98%`，但 `num_preemptions_total` 始终为 `0`。
- 处理：该探针没有达到 KV 池满载，不能证明或否定正式 128 请求场景的抢占重算；为避免继续消耗时间，停止 benchmark 客户端并等待残留请求排空，未停止 9032 服务。
- `FACT`：截至 2026-09-27 07:xx，9032 诊断服务队列已归零，KV 使用率 `0`，累计抢占 `0`；9031 基线服务健康。该轮只读/诊断实验没有修改比赛源码或官方服务参数。
- 下一步最小动作：使用已就绪的 9032 隔离服务做专门容量探针，增加请求数量但将输出长度降到最小，记录达到 KV 高水位时的 `kv_cache_usage_perc`、capacity 队列和抢占计数；探针完成后再决定是否进入调度或 KV dtype 方案。
- 2026-09-27 隔离容量探针 A：在 GPU 1/9032 上运行 `[[16384,1,64,64]]`，即 64 个请求、16k 输入、输出 1 token。运行中最多约 1 个请求驻留、约 60 个等待，但 KV 使用率约 `2.3%`，抢占 `0`；原因是输出极短，请求迅速释放 KV，不能形成持续容量压力。
- 处理：停止该探针客户端并等待 9032 队列归零；没有影响 GPU 0/9031 官方服务，没有修改源码或正式 benchmark 参数。
- `FACT`：截至本轮结束，9032 诊断服务队列为 `0`，KV 使用率 `0`，`num_preemptions_total=0`。
- 结论：单纯增加请求数或把输出降到 1 不能有效测量长序列 KV 池边界；下一轮应保持足够长的 decode 工作集，同时减少总请求/轮次，或直接使用固定长度单请求/小批次的持驻实验，避免继续启动官方四轮 wrapper。
- 2026-09-27 持驻 KV 探针：复用 GPU 1/9032 隔离服务，固定输入 `16384`、输出 `1024`，分别运行 8、16、24 个请求；均为单次 `vllm bench serve`，不作为官方成绩。
- 8 请求：8/8 成功，耗时 `62.57 s`，Total `2225.82 tok/s`，Output `130.93 tok/s`，Mean TTFT `14.85 s`；观测到 8 个请求驻留、KV 约 `25.8%`，抢占 `0`。
- 16 请求：16/16 成功，耗时 `209.64 s`，Total `1328.62 tok/s`，Output `78.15 tok/s`，Mean TTFT `36.10 s`，P99 TTFT `120.87 s`；观测到 16 个请求驻留、KV 约 `51.7%`，抢占 `0`。
- 24 请求：24/24 成功，耗时 `247.62 s`，Total `1687.21 tok/s`，Output `99.25 tok/s`，Mean TTFT `29.23 s`，P99 TTFT `124.00 s`；观测到 24 个请求驻留、KV 峰值约 `79.3%`，抢占 `0`。该批次完成后服务队列归零。
- `FACT`：8/16/24 请求均未触发诊断实例的 `vllm:num_preemptions_total`；当前证据证明 KV 使用率和 TTFT/吞吐会随持驻工作集变化，但尚未证明抢占重算是正式 16k 基线的主导因素。
- `FACT`：24 请求探针的请求完成时间呈明显长尾（首个约 233.84 s，末批约 247.62 s），但这是单次诊断样本，不能直接与官方 128 请求 summary 比较。
- 下一步最小动作：在 9032 空闲状态下做 30/32 请求、16k/1024 的单次持驻探针，重点记录 KV 是否接近 `100%`、capacity 等待和抢占计数；若仍为 0，则停止“抢占重算主因”的判断，转向调度 admission/CPU 开销和 `vllm._C` 环境问题。
- 2026-09-27 持驻 KV 探针 30 请求：在 GPU 1/9032 上运行单次 `vllm bench serve`，固定输入 `16384`、输出 `1024`、并发/请求数 `30`。请求 `30/30` 成功，耗时 `277.81 s`，Total `1879.83 tok/s`，Output `110.58 tok/s`，Mean TTFT `19.14 s`，P99 TTFT `104.46 s`。
- `FACT`：运行中 `num_requests_running=30`，KV 使用率峰值约 `97.16%`；结束后队列归零，9032 的 `num_preemptions_total` 从 `0` 增至 `1`。这是本轮首次在隔离实例中观察到接近 KV 满载时发生抢占。
- `FACT`：本轮 Prometheus `request_prefill_kv_computed_tokens_count=83`、sum `357,712`；实际请求总输入是 `30*16,384=491,520`。该差异与调度/统计口径有关，不能简单将差额当作重算 token，但证明不能只用请求数推断一次 prefill。
- `FACT`：9031 官方基线服务仍 HTTP 200；9032 探针结束后运行/等待请求和 KV 使用率均为 `0`，没有残留负载。
- `INFERENCE`：KV 接近容量上限时确实会触发抢占，但目前只有一次独立探针事件；它支持“容量边界是重要因素”，尚不足以量化其对正式 16k summary 的主导程度。
- 下一步最小动作：先保留 9032 诊断服务和本轮日志，读取 vLLM metrics 的 preemption/finished request 统计定义；随后在独立实例上验证 `vllm._C` 缺失是否是预装包特性及是否影响当前 Iluvatar backend，避免在没有证据前直接进入 KV dtype 源码改造。

### 4.6 实验 2026-09-26-04：远端连接中断

- 时间：2026-09-26 18:45 CST 左右
- 现象：本地 SSH ControlMaster `/Users/mahanting/.ssh/codex-control/ub39` 消失，`ssh -S ... -O check ub39` 返回 socket 不存在；SSH 别名 `ub39` 也不在当前静态 `~/.ssh/config` 中。
- 最后已知远端状态：benchmark PID `7705` 仍存在；`raw_runs_20260926_175332.csv` 已增长到 903 字节，`summary_20260926_175332.csv` 已生成，但当时脚本进程仍在运行，16k 四轮结果尚未完成读取确认。
- 影响：无法确认 benchmark 是否已自然退出，也无法读取最终 CSV 或启动准确率评估。
- 处理：未停止容器、服务或 benchmark；等待用户在 iTerm 重新建立原 SSH ControlMaster 后继续。
- 下一条最小命令：恢复 socket 后读取进程、CSV 和 summary；若 benchmark 已退出，立即执行 Level 3 accuracy。
- 并发约束：benchmark 期间不运行 accuracy 或其他请求，避免影响基线。

## 5. 实验记录模板

### 实验 YYYY-MM-DD-NN：标题

- 时间：
- 平台/设备：
- 服务器 hostname：
- 容器：
- 目标：本轮要证明什么？
- 代码：仓库、分支、commit、工作区状态
- 环境：Python/Torch/vLLM/FlagGems/Triton/驱动/设备工具版本
- 启动命令：
- benchmark 命令：
- evalscope 命令：
- 日志路径：
- `FACT` 已知证据：
- `HYPOTHESIS`：
- 实际执行：
- 遇到的问题：
- 原因判断及证据：
- 解决办法：
- 修改文件和位置：无则写“无修改”
- 为什么这样做：
- 结果：吞吐、TTFT、准确率、稳定性
- 未覆盖范围：
- 下一条最小命令：

## 6. 远端首轮计划

1. 只读确认登录身份、hostname、时间、当前目录、Docker/Podman、已有 `mllv` 容器和可用镜像。
2. 使用已存在的 Iluvatar CoreX 镜像创建持久容器 `mllv`，挂载 `/workspace`；不覆盖同名已有容器。
3. 在容器内记录硬件、依赖、模型和仓库路径，确认 BI-V150 后再安装/更新本地提交版本。
4. 严格按文档启动服务，先做简单请求，再跑官方四次 benchmark 和 Level 3 accuracy。
5. 保存原始日志、summary CSV、评估结果和 commit 信息，再开始任何代码修改。

## 7. 阶段交接

- 服务器：
- 远端仓库：
- 目标：
- 已执行命令：
- 结果：
- 修改：
- 未覆盖：
- 下一条最小命令：

### 实验 2026-09-27-连接检查续记：远端探针未执行

- 目标：继续检查 `vllm._C` 安装来源、插件 custom-op schema 注册情况，并确认 9032 诊断服务空闲。
- 预定范围：只读查询 `mllv` 容器，不重启服务、不改包、不触碰 GPU 0/9031 官方服务。
- 实际执行：通过现有 SSH ControlMaster 发起 `docker exec` 元数据检查；首次命令的多层 shell 引号解析失败，未执行远端检查。随后使用简短命令检查 ControlMaster/容器状态，SSH 在配置的连接超时窗口内无输出，手动中断该客户端。
- 影响：没有发起模型请求，没有停止或重启远端服务，没有修改远端文件、容器或比赛源码。远端当前状态无法由本轮新查询确认。
- 结论：`vllm._C` 缺失对当前 Iluvatar 执行路径的影响仍是 `UNKNOWN`；不得据此归因为吞吐下降，也不得在 9031 上尝试重装或构建。
- 下一步最小命令：用户在 iTerm 确认/重建 ControlMaster 后，先单独执行 `ssh -S ~/.ssh/codex-control/ub39 -O check ...`；确认 socket 可用后，再执行只读的 Docker 容器状态与 9032 `/metrics` 检查。随后用容器内 Python 直接读取 `importlib.metadata`、尝试导入 `vllm._C`，以及检查 `vllm_fl` 注册的 op schemas。

### 实验 2026-09-27-分析续记：FlagGems GEMM、混合步与验证方案

- 输入：用户补充的 FlagGems 路由分析；结合固定分支源码和已有基线/隔离探针结果复核。
- `FACT`：`vllm_fl/dispatch/config/iluvatar.yaml` 的 `flagos_blacklist` 当前为空；该配置的默认优先级是 FlagGems，再尝试 Iluvatar vendor backend，再回退 reference。该事实说明平台层没有通过 blacklist 排除通用 FlagGems 算子，但不等于每个模型线性层都已被同一个 GEMM 实现执行，仍需运行时 oplist/dispatch 证据确认。
- `FACT`：FlagGems backend 的 `attention_backend()` 默认返回 vLLM `TRITON_ATTN`；只有设置 `VLLM_FL_USE_FLAGGEMS_ATTN=1` 才返回 `AttentionFLBackend`。因此当前官方基线不能归因于 FlagGems attention；把该环境变量作为提交方案仍存在“简单切换算子”和官方启动命令不会设置变量两项风险。
- `FACT`：固定分支已有 `FLAGGEMS_ENABLE_OPLIST_PATH` 机制，worker 只在 rank 0 记录实际启用的 FlagGems 算子；这是比推测路由更低侵入的证据采集方式。
- `FACT`：`model_runner.py` 已有 `total_num_scheduled_tokens`、每请求 `num_scheduled_tokens`、`max_num_scheduled_tokens` 和 cudagraph dispatcher；运行时会根据 uniform decode 与 token 数选择 FULL/PIECEWISE/NONE。当前没有证据证明所有长上下文 step 都命中 FULL decode graph。
- `INFERENCE`：用户分析中“prefill 与 decode 混合导致 TPOT 膨胀”的方向合理，但仅凭 benchmark 的 Mean TTFT/TPOT 不能拆分纯 decode 和 mixed step；30 请求探针与 128 请求正式场景的 TPOT 也不能直接比较，因为请求到达和 chunked-prefill 分布不同。
- `INFERENCE`：`vllm._C` 缺失目前更像该镜像的预期兼容路径，而非已证实的性能回退点。`vllm_fl/__init__.py` 在原生扩展不可用时会调用 `_C_ops_registry.register_op_schemas()`，并提供必要 fallback；不得在没有 A/B 证据时重装或构建原生扩展。
- 决策修正：不把 `VLLM_FL_USE_FLAGGEMS_ATTN=1` 作为主线提交；不先修改 KV dtype；不在远端链路不稳定时启动长 benchmark。
- 下一步实验顺序：
  1. 只读读取 9032/容器的实际 FlagGems oplist，以及服务日志中的 batch/cudagraph 选择信息；
  2. 在 9032 隔离实例做短时 GEMM 路由诊断 A/B（仅用于估算上限，使用 blacklist 视为诊断，不作为提交）；
  3. 若 GEMM 对照有明确收益，再定位 FlagGems Iluvatar matmul 的 shape/tune config，做源码级 kernel 优化；
  4. 若收益不明显，再做 attention step 类型和长上下文 kernel 的低侵入 profile；
  5. 所有候选最终必须用官方命令、官方参数和 Level 3 accuracy 复验。
- 当前状态：本轮没有修改比赛源码，没有发送模型请求；两路服务此前均健康，9032 队列为空、KV 为 0、累计抢占为 1。最近一次远端大范围读取因 SSH 会话无输出被中止，不能把未读到的 oplist 当作不存在。

### 实验 2026-09-27-06：GEMM 路由前置只读核查

- 目标：在启动 GEMM blacklist 对照前，确认基线分布、编译/cudagraph 配置、现有 oplist 和 autotune 缓存状态。
- 实际执行：读取 `/workspace/benchmark_results/raw_runs_20260926_175332.csv`、基线服务日志、容器内 `/tmp` 与 `/workspace/logs` 的 oplist 文件，并以 SQLite 只读模式扫描 `/root/.flaggems/config_cache` 中 `mm_kernel`/`linear_kernel` 表。
- `FACT`：官方 16k 后 3 轮的 Mean TPOT 为 `486.12/485.99/486.22 ms`，Median TPOT 为 `531.69/531.67/531.20 ms`，Mean/Median ITL 为 `486.12/139.88`、`485.99/139.85`、`486.22/140.08 ms`，P99 ITL 为 `3326.21/3294.88/3258.82 ms`。该指标显示明显长尾，但不能单独区分混合 prefill、抢占或 attention。
- `FACT`：基线服务日志确认 `cudagraph_mode=FULL_DECODE_ONLY`，启动后实际生成 `cudagraph_capture_sizes` 为 `1..512` 的预定义集合，`max_cudagraph_capture_size=512`；同时启用 `enable_chunked_prefill=True`、`enable_prefix_caching=True`。这证明纯 decode batch 有命中 FULL graph 的条件，但尚未证明每个正式 step 都命中。
- `FACT`：基线日志未出现 `Using FlagGems attention backend`；结合源码默认值 `VLLM_FL_USE_FLAGGEMS_ATTN=0`，当前基线仍使用 vLLM `TRITON_ATTN`。
- `FACT`：当前可见的 `/tmp/flaggems_enable_oplist.txt` 没有可归属到单一服务的 GEMM 行；由于 9031/9032 共用默认路径，且记录文件会被后启动进程截断，不能据此判定基线未使用 GEMM。后续必须为每个实例显式设置独立路径。
- `FACT`：autotune SQLite 中存在 Iluvatar `mm_kernel` 和 `linear_kernel` 表；缓存按 Triton/kernel hash 分表，至少包含 M 从 `1` 到 `2048`（mm）及 `1` 到 `65536`（linear）的历史形状。当前缓存不是当前进程实际命中形状的充分证据，后续只作为选型参考。
- `FACT`：9031/9032 健康端点均返回 `200`；两路当前运行/等待请求为 `0`、KV 使用率为 `0`。9031 生命周期累计抢占 `16`，9032 为 `1`；均不是本轮新产生。
- 判定调整：GEMM 对照 treatment 必须完整排除 `mm, mm_out, addmm, addmm_out, addmm_dtype, addmm_dtype_out, bmm, bmm_out, baddbmm, baddbmm_out, linear` 等精确函数名；treatment 独立 oplist 中只要出现任一 GEMM 行，本轮无效，不能据此判断收益。
- 下一步最小命令：只在 GPU 1/9032 隔离服务做 P/D/M 中的一组短探针，control/treatment 均使用新进程和独立 `FLAGGEMS_ENABLE_OPLIST_PATH`；首轮只筛查路由和量级，不启动官方四轮 benchmark。完成后按 `P>=5%`、`D>=5%`、`M>=3%` 和 `<2%`/ABA 规则判定。

### 实验 2026-09-27-07：短探针命令校验

- 目标：启动第一组 GEMM control D 探针，使用 9032 隔离服务验证短测试命令和独立 oplist。
- 实际执行：误将不属于 `benchmark_throughput_serve.py` 的 `--output-file` 参数传入脚本；脚本在参数解析阶段直接退出，未发送请求、未创建结果文件、未影响 9031/9032。
- 原因：该脚本固定执行 `RUNS=4`、`SKIP_FIRST=1`，输出目录和文件名由脚本内部生成，不支持外部指定单次输出文件。
- 解决办法：不再用该四轮 wrapper 做短筛查；改用容器内 `vllm bench serve` 单次命令或固定请求数的短客户端，将完整 stdout/stderr 保存到独立日志，再用同一字段正则提取指标。control/treatment 仍需分别新进程、同 GPU、同参数，并通过独立 oplist 验收。
- 额外检查：执行 `vllm bench serve --help` 只触发 Python 初始化和 Iluvatar patch 日志，没有发起推理请求。
- 当前状态：9031 官方服务未动；9032 仍空闲；比赛源码无修改。
- 下一条最小命令：读取 `vllm bench serve --help` 的完整相关参数或使用现有 benchmark 脚本中的底层命令构造单轮 D 请求，先跑 control，再跑完整 GEMM blacklist treatment。

### 实验 2026-09-27-08：GEMM control 短探针与 oplist 归属修正

- 目标：验证单轮 `vllm bench serve` 命令可用，并采集 control 路由结果。
- 实际执行：在 9032 上运行 `random-input-len=128`、`random-output-len=64`、`max-concurrency=8`、`num-prompts=8` 的单轮请求；benchmark 客户端日志保存为 `/workspace/logs/gemm_control_d.log`。
- 结果：`8/8` 成功，耗时 `14.12 s`，Total `108.81 tok/s`，Output `36.27 tok/s`，Mean TTFT `13199.29 ms`，Mean/Median ITL `14.46 ms`。该结果仅证明单轮命令链路可用，不作为 P/D/M 性能结论。
- 问题：本轮只在 benchmark 客户端设置了 `FLAGGEMS_ENABLE_OPLIST_PATH=/workspace/logs/oplist_gemm_on_d.txt`，而 oplist 是由 9032 服务进程在 worker 初始化时记录；服务进程没有继承该变量，因此目标文件未生成。性能请求本身没有受到影响。
- 原因证据：`vllm_fl/worker/worker.py` 在 `flag_gems.enable(... path=fl_envs.FLAGGEMS_ENABLE_OPLIST_PATH)` 处写入清单，环境变量必须在启动 9032 服务时设置；客户端环境变量不改变已运行 worker 的配置。
- 解决办法：下一轮先停止并仅在 GPU 1 重启诊断服务，启动命令中设置独立 oplist 路径；服务健康后再运行 control/treatment。9031 官方服务保持不动。
- 当前状态：9031/9032 请求已排空；无比赛源码修改；本轮未执行 blacklist treatment。
- 下一条最小命令：在 9032 服务启动环境中设置 `FLAGGEMS_ENABLE_OPLIST_PATH=/workspace/logs/oplist_gemm_on_d.txt`，确认服务日志监听成功后，重新跑短 control；确认 control oplist 含实际 GEMM 行后，再按完整 blacklist 启动 treatment。

### 实验 2026-09-27-09：独立 oplist 服务重启与 GEMM control 复核

- 目标：让 9032 服务进程继承独立 `FLAGGEMS_ENABLE_OPLIST_PATH`，完成 control 短探针，并验证 GEMM 路由证据是否可用。
- 初始状态：9031 官方服务 PID `2943`，端口健康；9032 原诊断服务 PID `19936`，端口健康但未设置独立 oplist。两路服务均空闲。
- 第一次重启问题：重启 9032 时未设置 `CUDA_VISIBLE_DEVICES=1`，新进程默认看到 GPU 0；由于官方服务占用 GPU 0 约 `28 GiB`，日志报 `Free memory on device (3.83/32.0 GiB) ... desired ... 27.2 GiB`，服务初始化失败。9031 未受影响。
- 修复：清理失败诊断进程后，以 `CUDA_VISIBLE_DEVICES=1` 重启 9032，环境变量设置为 `FLAGGEMS_ENABLE_OPLIST_PATH=/workspace/logs/oplist_gemm_on_d.txt`；服务最终 PID `32932`，EngineCore PID `33073`，端口 `/health` 返回 `200`，GPU 1 正常占用模型显存。
- control 探针：在 GPU 1/9032 执行单轮 `vllm bench serve`，参数为输入 `4096`、输出 `256`、并发 `8`、请求 `8`；日志 `/workspace/logs/gemm_control_d2.log`。
- control 结果：`8/8` 成功，耗时 `33.72 s`，Total `1032.60 tok/s`，Output `60.74 tok/s`，Mean TTFT `13760.99 ms`，Mean TPOT `77.19 ms`，Median TPOT `80.84 ms`，Mean ITL `77.19 ms`，Median ITL `22.06 ms`，P99 ITL `743.12 ms`。该结果是短探针，不与官方 4k/16k 成绩直接比较。
- 独立 oplist：文件 `/workspace/logs/oplist_gemm_on_d.txt` 共 `32` 行；记录了 `attention_backend: default.flagos` 和多项 FlagGems elementwise/index/sort/softmax 等算子，但没有 `mm/mm_out/addmm/bmm/baddbmm/linear` 行。
- 关键日志事实：9032 日志显示 `enable_chunked_prefill=True`、`cudagraph_mode=FULL_DECODE_ONLY`、模型主图从 `/root/.cache/vllm/torch_compile_cache/...` 直接加载；dispatch manager 记录 `Op 'attention_backend' using 'default.flagos'`。日志未直接打印编译图内部 GEMM 的最终实现。
- 证据边界：oplist 的 `once` 记录机制只记录 FlagGems Python dispatcher 首次调用点；Inductor 编译图从缓存加载后，主图内 `extern_kernels.mm`/`mm_out` 是否落到 FlagGems，不能由该 oplist 证明。因而“control oplist 无 GEMM 行”不能作为 treatment 验收标准，也不能据此判断模型未使用 FlagGems GEMM。
- 处理决定：本轮不启动 GEMM blacklist treatment。若直接启动，control 与 treatment 的路由差异无法由当前观测证据确认，结果容易产生不可解释的假阴性；同时 treatment 需要重启 9032，成本较高且可能污染当前诊断服务状态。
- 当前状态：9031 官方服务未重启、未发送请求；9032 服务健康且请求已排空；比赛源码无修改；本轮只产生诊断日志和独立 oplist。
- 下一步候选：
  1. 先做“编译缓存隔离”实验：为 control/treatment 指定不同的 `VLLM_CACHE_ROOT`/compile cache，并在服务日志中确认重新编译，避免旧 graph cache 掩盖 blacklist 差异；
  2. 或直接做 FlagGems/Inductor 的算子级微基准，使用 SQLite 已有的 Iluvatar `mm_kernel`/`linear_kernel` 形状，不经过 vLLM 服务，先回答 GEMM kernel 是否存在可测差距；
  3. 只有能证明 treatment 实际改变了 GEMM 路径后，才进行 P/D/M 服务级 A/B。


### 实验 2026-09-27-10：编译缓存、采样默认与 GEMM dispatch 校准

- 时间：2026-09-27 10:34-10:40 CST
- 平台/设备：Iluvatar BI-V150；GPU 2 做独立校准；未重启 9031/9032，未发送服务请求。
- 服务器/容器：`ub39` / `mllv`
- 目标：修正“编译缓存会使 blacklist 失效”的表述，确认缓存中的实际算子线索，校准 FlagGems GEMM 的可观测性，并确认当前 attention kernel 选择规则。
- 代码与环境：服务仍基于固定分支 `flagos-2026-s2`，commit `13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0`；FlagGems `v5.3.5`；模型配置为 Llama 架构、42 层、hidden size 2048、16 个 Q heads、2 个 KV heads、head dim 128、BF16、最大位置 131072。

#### 1. 编译缓存只读检查

- 检查目录：`/root/.cache/vllm/torch_compile_cache`
- `FACT`：缓存文件约 `40,744` 个。
- `FACT`：缓存文本中出现约 `480` 次 `extern_kernels.mm`。
- `FACT`：缓存中没有检出 `triton_tem_*`。
- `FACT`：缓存中大量出现 `flag_gems`，并出现 `torch.ops.aten.mm`、`torch.ops.vllm.unified_attention_with_output` 等图/算子符号。
- 判断：编译缓存本身不会自动令 blacklist 失效；缓存只是复用了已经生成的图。当前缓存同时包含 `extern_kernels.mm` 和 FlagGems 相关符号，但仅凭字符串不能断言 `extern_kernels.mm` 的最终实现来自哪条 backend 路径。
- 边界：若某个 GEMM 已被编译成 Inductor 自有模板，或 FlagGems 调用已被 trace 进图，服务级 oplist 都可能无法直接反映运行时最终 kernel。因此后续需要以 dispatch 注册表、编译输出或 profiler 证明路径变化。

#### 2. GPU 资源状态

- `ixsmi` 只读结果：GPU 0 约 `28770 MiB` 已用，GPU 1 约 `28688 MiB` 已用；GPU 2-15 每卡约 `68 MiB` 已用、约 `32700 MiB` 空闲。
- 说明：GPU 2 可用于独立校准和微基准；本轮没有触碰 9031 官方服务或 9032 诊断服务。

#### 3. GEMM dispatch/oplist 校准

- 执行：在 `CUDA_VISIBLE_DEVICES=2` 的独立 Python 进程中调用 `flag_gems.enable(record=True, once=True, path=...)`，分别执行 `torch.mm`、带 `out=` 的 `torch.mm` 和 `torch.nn.functional.linear`，再查询 ATen dispatch 表。
- `FACT`：`use_c_extension=False`。
- `FACT`：`aten_patch_list=[]`。
- `FACT`：`aten::mm`、`aten::linear`、`aten::addmm` 的 CUDA dispatch 行均指向 `/workspace/FlagGems/src/flag_gems/__init__.py:62 [kernel]`。
- `FACT`：校准 oplist 没有记录 `GEMS(_ILUVATAR)? MM/MM_OUT/LINEAR/ADDMM` 行。
- 解释：FlagGems 的 Python CUDA dispatch 注册确实接管了这些 ATen operator，但当前实现/记录机制没有把这次 GEMM 调用写入 oplist；因此“oplist 没有 GEMM”不能作为服务中 GEMM 未走 FlagGems 的证据。相反，dispatch 表证明至少在独立 eager 进程里，GEMM 的 CUDA operator 注册已由 FlagGems 覆盖。
- 未完成事项：校准进程没有做 profiler，也没有把服务编译图中的 `extern_kernels.mm` 解析到具体设备 kernel；不能据此宣布 blacklist treatment 有收益或无收益。

#### 4. benchmark 采样默认

- 读取 vLLM `bench serve` 实现：当命令未传 `--temperature` 时，参数字典不包含 temperature，并打印 warning；默认行为由服务端/模型 API 决定，不再自动强制 greedy。
- 官方 benchmark wrapper 当前未显式传 `--temperature=0`；因此后续严格复现实验应沿用官方 wrapper，不自行补参数。短探针的结果也不能与官方成绩混用。
- `random` 数据集在 OpenAI-compatible backend 下会自动设置 `ignore_eos=True`，因此输出长度可按 benchmark 设定完成。

#### 5. attention kernel 选择事实

- 当前基线 attention backend 为 vLLM `TRITON_ATTN`，不是 FlagGems 自定义 attention。
- 代码位置：`/usr/local/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_unified_attention.py:969-978`。
- `FACT`：只要 `max_seqlen_q > 1`（存在 prefill/mixed batch），unified attention 就不会走 3D kernel。
- `FACT`：只要请求数 `num_seqs > seq_threshold_3D`，也不会走 3D kernel。
- `FACT`：只有 3D 所需的 segment buffers 已分配、`max_seqlen_q == 1`、请求数未超过阈值且未启用 batch invariance 时，才走 3D；否则走 2D kernel。
- `FACT`：2D 使用 `TILE_SIZE_PREFILL`，3D 使用 `TILE_SIZE_DECODE`；代码注释明确 2D kernel 用于 prefill/mixed 场景，3D kernel 用于满足条件的 decode 场景。
- 代码位置：`/usr/local/lib/python3.12/site-packages/vllm/v1/attention/backends/triton_attn.py:135-168, 231-232, 634-662`；该 backend 计算 `seq_threshold_3D` 并传入 unified attention。
- 结论：此前 H7“mixed prefill/decode 可能触发 2D、长 decode 扫描 KV”已经得到代码级支持，但还没有请求级 profiler/运行时 kernel 计数，仍应标记为“路径事实 + 性能因果待测”，不直接当作最终瓶颈结论。

#### 6. 结果、问题与决策

- 本轮问题：第一条远端复合命令因嵌套 shell 引号失败；第二次检查因容器内无 `rg` 失败；均未改变远端状态。
- 解决：改用 stdin 传递临时只读脚本，并使用容器可用的 `grep/find/sed`；后续命令已成功完成。
- 修改文件和位置：无比赛源码修改；本地仅追加本实验记录；远端仅产生校准日志 `/workspace/logs/oplist_calib_gpu2.txt`。
- 为什么这样做：先验证观察工具是否真的能区分 control/treatment，再决定是否值得重启服务做 GEMM A/B，避免把不可观测的实验结果误判为优化效果。
- 当前结论：暂停 GEMM blacklist treatment 仍然正确，但理由应表述为“当前 oplist 未被校准、无法证明 treatment 改变了服务 GEMM 路径”，而不是“编译缓存会让 blacklist 失效”。缓存检查反而显示需要进一步解析 `extern_kernels.mm` 的最终实现。
- 当前状态：9031 官方服务、9032 诊断服务均未重启；GPU 2-15 空闲；没有比赛源码修改；没有新增服务级 benchmark。
- 未覆盖范围：未做 profiler；未对 `extern_kernels.mm` 做源码/二进制反解；未验证 2D/3D attention kernel 的实际运行占比；未做 GEMM treatment；未做 Level 3 复测。
- 下一条最小命令：在 GPU 2 做 attention 级微基准或运行时 kernel 观测，优先验证 2D/3D 路径的时延差异；若仍要研究 GEMM，先对 `extern_kernels.mm` 生成来源做定向解析，再决定是否建立隔离 cache 的 control/treatment。


### 实验 2026-09-27-11：4k 尾延画像与 GPU 2 GEMM 微基准

- 时间：2026-09-27 10:49-11:xx CST
- 目标：并行判断 4k 是否存在明显 mixed-step 长尾，并测量模型主要线性层形状下原生 Torch GEMM 与 FlagGems GEMM 的差异。
- 约束：只读官方 CSV；微基准运行在 GPU 2 独立进程；未重启 9031/9032，未修改比赛源码。

#### 1. 4k 官方 ITL/TPOT 画像

- 官方后 3 轮 4k summary：
  - Mean ITL `148.47 ms`
  - Median ITL `92.45 ms`
  - P99 ITL `773.93 ms`
  - Mean TPOT 与 Mean ITL 相同，为 `148.47 ms`
- 原始后 3 轮：Mean ITL `150.08/149.17/146.16 ms`；Median ITL `92.80/92.08/92.48 ms`；P99 ITL `775.40/782.60/763.78 ms`。
- 判断：4k 也存在明显长尾，Mean/Median 约 `1.61x`，P99/Median 约 `8.37x`。因此不能把 mixed-step/调度长尾限定为 16k 特有问题。
- 边界：ITL 统计是请求级时间分布，不能单独证明长尾来自 mixed prefill、attention 2D kernel、CPU 调度或其他事件；需要运行时 profile 才能拆分因果。
- 对比 16k：16k Mean/Median 约 `3.47x`，P99/Median 约 `23.54x`，说明长上下文把同类尾延问题显著放大。

#### 2. GPU 2 copy 参考

- 独立脚本在 GPU 2 对 256 MiB BF16 buffer 做 device-to-device copy，预热后 30 次。
- 结果：平均 `0.919 ms`，双向计量约 `584.18 GB/s`。
- 说明：这是当前运行时/拷贝实现的参考值，不等于芯片理论峰值，也不代表模型实际 KV 搬运带宽。

#### 3. GPU 2 GEMM control：原生 Torch

- dtype：BF16；计算：`torch.mm(x, w.t())`；预热 10 次、计时 50 次。
- 形状及结果：

| 形状 | Mean ms | TFLOP/s |
|---|---:|---:|
| M=30,K=2560,N=6912 | 0.07818 | 13.58 |
| M=64,K=2560,N=6912 | 0.08386 | 27.01 |
| M=2048,K=2560,N=6912 | 0.83222 | 87.09 |
| M=30,K=2560,N=2560 | 0.04183 | 9.40 |
| M=64,K=2560,N=2560 | 0.04799 | 17.48 |
| M=2048,K=2560,N=2560 | 0.35109 | 76.46 |

#### 4. GPU 2 GEMM treatment：FlagGems

- 独立新进程调用 `flag_gems.enable()` 后使用同样 shape、预热和计时。
- 结果：

| 形状 | Mean ms | TFLOP/s | 相对 Torch 延迟 |
|---|---:|---:|---:|
| M=30,K=2560,N=6912 | 0.17328 | 6.13 | `+121.7%` |
| M=64,K=2560,N=6912 | 0.17884 | 12.66 | `+113.3%` |
| M=2048,K=2560,N=6912 | 1.03054 | 70.33 | `+23.8%` |
| M=30,K=2560,N=2560 | 0.11591 | 3.39 | `+177.1%` |
| M=64,K=2560,N=2560 | 0.13588 | 6.17 | `+183.1%` |
| M=2048,K=2560,N=2560 | 0.46344 | 57.92 | `+32.0%` |

- 解释：在该独立 eager 微基准中，FlagGems 当前 GEMM 实现全面慢于原生 Torch，尤其是 decode 典型小 M（30/64）。这不是服务级结论，因为服务主图可能把 GEMM 编译为 `extern_kernels.mm` 或其他 Inductor 路径；但它明确说明“无条件把 GEMM 切到 FlagGems”没有优化依据。
- 重要边界：本实验比较的是独立 eager dispatch，不是官方服务的真实编译图；不能直接用这些百分比推算 4k/16k 吞吐收益。

#### 5. 本轮问题与决策

- 问题：初版对比脚本尝试调用不存在的 `flag_gems.disable()`，在 control 完成后退出，未产生有效 A/B；没有影响服务。
- 解决：改成两个完全独立进程，分别测 Torch 与 FlagGems，避免进程级 dispatch 状态污染。
- 修改文件：无比赛源码修改；仅追加实验记录；远端只运行 GPU 2 独立脚本。
- 决策：不启动 GEMM blacklist treatment。当前微基准显示 FlagGems GEMM 本身更慢，且服务级路由仍不可完全观测；继续做 blacklist A/B 的预期收益低、解释风险高。
- 下一步：保留 4k/16k 尾延结论，优先做 unified attention 的三类形状微基准或运行时 profile；重点比较 2D mixed/prefill 与 3D pure decode，而不是先改 GEMM。


### 结论更新（2026-09-27 11:xx）

- ✓ 已确认：官方 4k 后 3 轮 Mean/Median ITL=`1.61x`，P99/Median=`8.37x`。
- ✓ 已确认：官方 16k 后 3 轮 Mean/Median ITL=`3.47x`，P99/Median=`23.54x`。
- ✓ 已确认：GPU 2 独立 eager 微基准中，FlagGems GEMM 在全部测试形状下慢于原生 Torch，延迟增幅约 `24%~283%`。
- ✓ 已确认：当前服务主图从编译缓存加载；缓存出现 `extern_kernels.mm`，但其最终实现尚未由 profiler/二进制解析确认。
- ✓ 已确认：代码逻辑规定 mixed/prefill 场景走 unified attention 2D 路径；满足条件的纯 decode 才可能走 3D。
- ⚠ 待验证：服务编译图中 GEMM 的实际运行时路由，以及 blacklist treatment 是否会改变该路由。
- ⚠ 待验证：mixed step 在 4k/16k 墙钟时间中的真实占比；当前聚合日志没有逐 step `max_seqlen_q`、`total_num_scheduled_tokens`、cudagraph dispatch 或 step duration 记录。
- ⚠ 待验证：2D/3D attention 在 BI-V150 实际 shape 下的时延差异；GPU 2 首次直调微基准进入长时间编译/调优且无结果，已终止，不能把该轮当作失败性能结论。
- ❌ 已放弃作为当前主线：无条件 GEMM blacklist。理由不是“FlagGems GEMM 一定不是服务瓶颈”，而是路由尚不可观测且理论收益有限。

#### GEMM 收益边界修正

- `FACT`：小 M（30/64）FlagGems GEMM 比 Torch 慢约 `2.1x~2.8x`，大 M（2048）慢约 `24%~32%`。
- `INFERENCE`：若服务 decode GEMM 全部实际走 FlagGems，4k 可能存在可优化空间；但不能把独立 eager 延迟直接等同于服务吞吐损失。
- 原先按“42 层、3 次 GEMM/层、约 11 ms/token”给出的估算作废：模型真实线性层形状和每层 GEMM 数量未按 `config.json` 校准，而且小 M eager 计时混入 host dispatch 成本，不能直接折算服务 token 时间。
- 模型配置实际为 `hidden_size=2048`、`num_attention_heads=16`、`num_key_value_heads=2`、`head_dim=128`、`intermediate_size=6144`；典型形状应从 `K=2048` 的 QKV/O、gate/up 等层推导，不能使用此前记录中的 `K=2560` 或 `num_heads=40`。
- `INFERENCE`：decode 每步仍需读取大部分权重；如果把 eager 小 M 的有效带宽直接套到服务，单权重读取量级已经接近实测 decode step 周期，因此服务 decode 不太可能简单等同于独立 eager 小 M 路径。更可能是 CUDA graph/编译图改变了开销结构，或服务实际走了其他实现；需要 profiler 通过 kernel 名称确认。
- `DECISION`：GEMM 不再作为当前主线；profiler 只顺带确认其实际 kernel 路由，不再单独做 blacklist A/B。
- 16k mixed step 的主要额外工作随 KV/context 长度增长；M≈2048 时 GEMM 差异的相对占比更低，不能把 GEMM 作为 16k 第一优先级。

#### Step 统计核查

- 读取 `/workspace/logs/minicpm_baseline_server.log`，未发现逐 step 的 `total_num_scheduled_tokens`、`max_num_scheduled_tokens`、`max_seqlen_q`、cudagraph dispatch 或 step duration 日志。
- 9031、9032 健康端点均返回 `200`；9032 当前运行/等待请求为 `0`，KV 使用率为 `0`，累计抢占为 `0`（本轮未增加）。
- 代码中确实存在 `scheduler_output.total_num_scheduled_tokens`、每请求 `num_scheduled_tokens` 和 cudagraph dispatch 逻辑，但这些字段未被当前日志默认输出。
- 结论：不能仅凭现有官方日志统计 mixed/pure-decode 比例；下一次若需要该统计，应在隔离 9032 上开启详细 iteration logging，或加入仅诊断用的 step 采样日志后重启隔离服务。不得修改 9031 官方服务来采集。

#### Attention 微基准状态

- 第一次直调尝试使用扁平 K/V 张量，触发 `ZeroDivisionError`；原因是 unified attention 需要 paged KV cache 的合法布局和 block table，不能用 `(KV,Hkv,D)` 直接替代。
- 第二次按 vLLM 测试布局改为 `[num_blocks, block_size, num_kv_heads, head_size]`，并构造 `cu_seqlens_q`、`seq_lens`、`block_table` 和 3D segment buffers。
- 该轮在 GPU 2 首次编译/调优阶段长时间无输出，未得到任何时延值，随后主动终止；没有影响 9031/9032，也没有留下可用于结论的 attention 数据。
- 下一次不要直接用完整 16k/30 序列组合做首次编译；应先用 vLLM 测试中的小 shape 验证 kernel 调用和编译路径，再按相同进程逐步放大到目标 shape，并把编译时间和稳态时间分开记录。


### 实验 2026-09-27-12：Step 日志审计与编译缓存复用验证

- 时间：2026-09-27 11:xx CST
- 目标：评估三条后续路径 A/B/C 的可行性，先确认官方日志是否已有 step 级信息，再验证共享编译缓存能否直接支持 attention 独立微基准。
- 约束：只读检查官方日志；独立探针只使用 GPU 2；未重启 9031/9032，未修改比赛源码。

#### 1. 官方日志审计

- 文件：`/workspace/logs/minicpm_baseline_server.log`，约 `3708` 行、`580471` 字节。
- 未发现逐 step 的 `total_num_scheduled_tokens`、`num_prefill_tokens`、`num_decode_tokens`、`max_seqlen_q`、`Selected cudagraph capture size` 或 step duration 记录。
- 日志只包含服务启动配置，例如 `enable_logging_iteration_details=False`、`cudagraph_mode=FULL_DECODE_ONLY`、capture sizes 和 `enable_chunked_prefill=True`。
- `9031`、`9032` 健康端点均返回 HTTP `200`；9032 当前无运行/等待请求、KV 使用率为 `0`、累计抢占未增加。
- 结论：路径 B 的“直接从官方日志统计 mixed step”不可行；现有日志不能回答 mixed step 占比。

#### 2. 共享编译缓存审计

- `/root/.cache/vllm` 约 `40,746` 个文件、约 `2.07 GB`；`torch_compile_cache` 约 `40,744` 个文件、约 `2.07 GB`。
- 缓存中有近期生成的 Inductor/aotautograd/fxgraph 文件，但独立 attention 调用的参数布局和 vLLM 服务内部编译入口不完全相同。
- 采用路径 A：GPU 2 独立脚本设置 `VLLM_CACHE_ROOT=/root/.cache/vllm`，先测试小 shape（prefill 32/128、decode 8×256 的 2D/3D），预热后测稳态。
- 结果：首次 unified attention 调用持续处于编译/调优阶段，约一分钟内无任何测量输出；为避免不可控占用 GPU 2，主动终止。此前完整 16k shape 也有同样现象。
- 判定：共享 cache 路径不能保证独立 attention 微基准复用服务已编译 kernel；原因可能是调用入口、layout、Triton key 或编译缓存命名不一致。该路径当前不可作为“最快方案”。

#### 3. 路径调整

- 路径 A：暂停。不能继续扩大独立 attention shape，直到找到服务实际 kernel cache key 或改为从服务进程内采样。
- 路径 B：官方日志无 step 细节，不能直接完成；需要在 9032 隔离服务重启时打开 `enable_logging_iteration_details`/DEBUG，或增加仅诊断用的 runtime 计数。
- 路径 C：暂不直接注入源码级 profiler。Profiler 会改变 kernel 时序和调度，且需要处理长 trace；应先在 9032 上做短请求、短 trace 的服务内采样。
- 当前最小可行方案：只重启 9032 诊断服务，在保持 9031 官方服务不变的前提下，启用 iteration logging；使用 8 或 16 请求、16k 输入、256 输出的短探针，记录 step 级 token 数、attention metadata 和累计时间。若日志仍不含 `max_seqlen_q`，再在 9032 做 5-10 step 的轻量 profiler。

#### 4. 当前结论状态

- ✓ 已确认：官方日志没有 step 类型统计。
- ✓ 已确认：共享 `VLLM_CACHE_ROOT` 未能让独立 attention 调用快速复用已编译 kernel。
- ⚠ 待验证：服务内部 mixed/pure-decode step 比例。
- ⚠ 待验证：服务内 2D/3D attention 实际 kernel 与耗时。
- ❌ 暂停：继续在 GPU 2 直接调用完整 unified attention 做大 shape autotune。
- 未修改比赛源码；9031 官方服务未重启；9032 本轮未重启。


### 实验 2026-09-27-13：9032 iteration logging 结构化统计与 4k 对照

- 时间：2026-09-27 11:17-11:44 CST
- 目标：在不影响 9031 官方服务的前提下，验证 16k 探针中的 mixed step 是否真实出现，并用相同诊断服务做 4k 短对照；同时核查 `vllm._C` 缺失是否已有具体 fallback 证据。
- 约束：只使用 9032/GPU 1；不修改比赛源码；短探针结果不作为官方成绩。

#### 1. 16k iteration 日志统计

- 负载：8 请求，输入 16384，输出 256，并发 8；日志文件 `/workspace/logs/diag_iter.log`。
- 解析到有效 `Iteration(...)` 行 320 条，时间范围约 `11:17:17-11:19:54`。
- 分类规则：有 context、无 generation 为 `context-only`；两者同时存在为 `mixed`；只有 generation 为 `generation-only`。
- 结果：
  - `context-only`：8 条；context tokens 均为 2048。
  - `mixed`：57 条；多数 context tokens 为 2041-2047，少数收尾为 218；generation requests 为 1-7。
  - `generation-only`：255 条；generation requests 为 1-8。
- `iteration elapsed time` 分布异常：context-only 平均约 41.84 ms、median 0.03 ms、最大 303.84 ms；mixed 平均约 2.20 ms、median 0.03 ms、最大 80 ms。大量 0.02-0.04 ms 与实际长序列请求墙钟耗时不一致，因此该字段不能直接当作完整 GPU step duration，也不能据此计算 mixed 的 GPU 时间占比。
- `FACT`：16k 短探针确实存在 57 个 mixed iteration；mixed 不是仅由代码路径推断出来的。
- `UNKNOWN`：mixed iteration 中 attention、KV 搬运、调度等待和其他算子的分项 GPU 时间。

#### 2. 4k 短对照

- 负载：8 请求，输入 4096，输出 256，并发 8；使用 9032，官方 9031 未触碰。
- 结果：8/8 成功，耗时 `11.14 s`，Total token throughput `3126.12 tok/s`，Output tok/s `183.89`，Mean TTFT `5531.58 ms`，Median TTFT `5535.20 ms`，P99 TTFT `5537.79 ms`，Mean/Median/P99 ITL=`21.95/22.06/23.43 ms`。
- 对应服务 iteration 区间为 `Iteration(320)-Iteration(576)`，时间约 `11:39:03-11:39:09`，共 257 条：
  - `context-only`：1 条，context tokens=16；
  - `mixed`：1 条，context tokens=112，7 个 context requests 与 1 个 generation request 同时存在；
  - `generation-only`：255 条，主要为 8 个 generation requests，iteration elapsed 大多约 12-16.4 ms，最大 22.98 ms。
- `FACT`：在该短对照的调度轨迹中，4k 几乎立即进入纯 generation；16k 则有明显更长的 context/mixed 阶段。
- `INFERENCE`：该现象支持“16k 的长尾和 TTFT 放大与长 context 的 chunked prefill 与 generation 交错有关”，但尚不能把原因收敛到某一个 attention kernel。
- 重要更正：本轮 4k 探针不是冷启动对照。16k 探针结束后同一 9032 进程继续运行，4k 对应的两个 prefill iteration 只有 `16+112=128=8×16` 个 context tokens；而不是预期的 `8×4096`。9032 开启了 prefix caching，且 4k 使用相同默认 seed，导致 4k prompt 命中了此前 16k prompt 的前缀块。
- 证据：9032 metrics 在该轮结束后的 `prefix_cache_queries_total=163840`、`prefix_cache_hits_total=32640`，命中率约 `19.9%`；日志中的 4k prefill 只计算 128 个 token。该轮 `3126.12 tok/s`、`5.53 s TTFT` 只反映 prefix-cache 命中后的短残差请求，不能代表冷启动 4k 性能。
- 修正结论：撤回“4k 几乎立即进入纯 generation 是 4k 负载本身特征”的推断；也撤回用该轮 4k 与 16k 做 prefill 阶段对照的结论。以后对照必须更换 seed/请求前缀，或在测量前清空并重新启动 prefix cache，并记录 cache query/hit 增量。
- 注意：本轮 4k/16k 结果都不是官方成绩；4k 结果由于缓存污染降级为诊断记录，不用于性能排序。

#### 3. `vllm._C` 环境核查

- 运行时导入结果：
  - `vllm._C`：失败，`ModuleNotFoundError`；
  - `vllm._C_stable_libtorch`：失败，`ModuleNotFoundError`；
  - `vllm_fl._C`：失败，`ModuleNotFoundError`。
- 9032 日志多次出现 `Failed to import from vllm._C`，但没有发现带具体算子名的 `Op '<name>' fallback to ...` 运行时事件。
- 源码事实：`vllm_fl/platform.py` 会尝试 legacy 和 stable ABI；`vllm_fl/__init__.py` 在两者都不存在时注册 Python fallback schemas。因此“缺少 `vllm._C`”本身不能证明服务所有路径都退化到慢实现。
- `DECISION`：暂不直接修改安装包或补编译扩展；需要 profiler 或更细粒度 dispatch 观测确认具体受影响算子，再决定是否处理。

#### 4. 本轮结论状态

- ✓ 已确认：16k 诊断探针有大量 mixed iteration（57 条），4k 同规模短探针只有 1 条 mixed iteration。
- ⚠ 已确认但不可用于冷启动结论：4k 短探针因 prefix cache 命中只计算 128 个残差 context tokens；其纯 generation iteration 数据不能代表冷启动 4k 负载。
- ✓ 已确认：当前容器同时缺失 `vllm._C`、`vllm._C_stable_libtorch`、`vllm_fl._C`。
- ⚠ 待验证：mixed iteration 的真实 GPU 时间组成，尤其是长前缀 2D attention、KV 访问和调度等待占比。
- ⚠ 待验证：`vllm._C` 缺失是否对 MiniCPM 当前服务图造成可测吞吐损失。
- ❌ 未得出：不能仅凭 iteration elapsed 或 `_C` warning 宣称 attention kernel 或 `_C` 缺失是唯一瓶颈。

#### 5. 下一步建议

1. 保持 9031 不变，重启 9032 以清空 prefix cache，在 9032 做一次短服务内 profiler；使用新 seed，优先选择 8 请求、16k 输入、64-128 输出，避免 profiler 前 5-10 个 step 只落在低前缀阶段。
2. 使用两个窗口：W1 为 prefill/mixed 阶段连续 3-8 个有效 iteration，W2 为约 10 个 generation-only iteration；profiler 先确认能记录设备端 CUDA/CoreX kernel，再解释 trace。
3. profiler 前保存 9032 当前日志和 PID；只对诊断服务注入，关闭 shapes/stack/memory/flops 等高开销选项，避免把 profiler 数据混入官方成绩。
3. 若 profiler 显示 mixed 主要由长 KV attention 占用，再进入 attention 入口/内核优化评估；若显示 `_C` 相关 fallback 或 CPU 调度占主导，先解决环境/运行时路径问题。
4. 继续暂停独立 GPU 2 的 unified attention 大 shape autotune、无条件 GEMM blacklist 和 KV dtype 源码改造；目前证据还不足以安全进入这些改动。

### 实验 2026-09-27-13: W1 服务内 Torch profiler（9032）

- 时间：2026-09-27 13:37-13:38 CST。
- 目标：在 9032/GPU 1 的真实服务执行路径中，采集后段 mixed step 的设备端 kernel；9031/GPU 0 未重启、未修改。
- 负载：8 请求，随机输入 16384、输出 128，并发 8；使用新 seed `20260928`；9032 已关闭 prefix caching。
- 触发方式：远端 watcher 监视 iteration 日志，在累计出现 35 个 mixed iteration 后调用 `POST /start_profile`，避免手动调用错过 prefill/mixed 阶段。
- 原始文件：`/workspace/logs/profiler_w1/profiler_out_0.txt`、`/workspace/logs/profiler_w1/rank0.1790487403905808382.pt.trace.json`；trace 大小约 6.5 MB，事件数 27,431。

#### 观测结果

- `FACT`：Torch profiler 记录到了设备端 kernel，而非只有 CPU 事件。trace 类别包含 `kernel`、`gpu_user_annotation`、`gpu_memcpy`；可见 `kernel_unified_attention`、`mm_kernel`、`reshape_and_cache_kernel_flash` 等 kernel 名称。
- `FACT`：W1 汇总中 `vllm::unified_attention_with_output` / `kernel_unified_attention` 共 336 次，Self CUDA=`13.563 s`，占总 Self CUDA=`14.604 s` 的 `92.87%`，平均每次约 `40.365 ms`。
- `FACT`：`aten::mm` / `mm_kernel` 共 1,344 次，Self CUDA=`898.074 ms`，占 `6.15%`，平均每次约 `668.210 us`。
- `FACT`：KV cache 更新 `reshape_and_cache_kernel_flash` 共 336 次，Self CUDA=`9.862 ms`，占 `0.07%`。
- `FACT`：采样相关 CUDA 时间很小：`aten::_softmax` 约 `513 us`，`aten::argmax` 约 `399 us`；不能解释 W1 的主要设备时间。
- `FACT`：trace 中还可见大量 Triton fused/rms-norm kernel，但单项设备时间均显著低于 unified attention。

#### 判定

- `DECISION`：W1 已确认真实服务的 mixed 窗口由 unified attention 主导；GEMM 不是本窗口的第一瓶颈，暂停 GEMM blacklist A/B 的决定保持不变。
- `INFERENCE`：在当前长前缀 mixed 负载下，优化方向应优先放在 unified attention 的长 KV 访问、2D prefill/mixed 路径和其 tile/并行策略，而不是采样或 KV cache 写入。
- `UNKNOWN`：W1 汇总跨越 profiler 激活期间的多个 worker step，但当前汇总尚未按 iteration 拆分；不能仅凭 W1 断言“所有 16k step 都有 92.87% attention 占比”。
- `UNKNOWN`：尚未用同配置的纯 generation-only W2 测量 decode attention 占比与单 step 时延，因此 mixed 路径相对纯 decode 的额外成本仍待分离。

#### 失败/修正记录

- 首轮 profiler 配置只启动了 API，没有在 mixed 阶段调用 `POST /start_profile`，因此目录为空；第二轮手动调用时已进入 generation-only，也未得到有效目标窗口。
- 本轮改用日志 watcher 自动触发，成功在 mixed 阶段启动 profiler；该方法保留为后续 W2 的标准触发方式。

#### 当前结论状态（2026-09-27 13:xx 更新）

- ✓ 已确认：真实服务设备端存在 `kernel_unified_attention`。
- ✓ 已确认：后段 mixed 窗口中 attention CUDA 时间占绝对主导（92.87%）。
- ✓ 已确认：W1 中 GEMM CUDA 时间约 6.15%，暂不做 GEMM blacklist 主线。
- ✓ 已确认：KV cache 写入和采样不是 W1 的主要设备瓶颈。
- ⚠ 待验证：纯 generation-only 窗口的 attention 占比与 3D/解码路径成本。
- ⚠ 待验证：attention kernel 是否在不同前缀长度下随 KV 长度近似线性增长。
- ❌ 暂停：无服务缓存依据的独立大 shape unified-attention 微基准、无条件切换 attention backend、KV dtype 源码改造。

### 实验 2026-09-27-14：W2 纯 decode 窗口尝试

- 目标：在同一 9032 服务上抓取纯 generation-only 窗口，作为 W1 mixed attention 占比的对照。
- 方法：新 seed、`temperature=0`、8 请求、16k 输入、128 输出；远端 watcher 在日志检测到至少 12 个连续 generation-only step 后调用 `/start_profile`。
- 结果：本轮 benchmark 在本次记录结束前仍处于 context/mixed 阶段，未达到触发条件；未生成新的 profiler trace。为避免继续占用 GPU，已停止诊断 benchmark。9031/9032 健康检查均返回 HTTP 200。
- `FACT`：W2 未完成，不能把 W1 的 92.87% attention 占比外推到纯 decode。
- `DECISION`：W2 作为补充对照保留，不阻塞当前主线；已有 W1 证据足以进入 unified attention 路径源码和 kernel 参数审查，但在改源码前仍需一个纯 decode 对照或服务内 kernel 分类证据。

### 分析 2026-09-27-15：attention 接入路径与参数扫描边界

#### F1：接入路径已确认

- `FACT`：固定 commit 的 `vllm_fl/dispatch/config/iluvatar.yaml` 对 `attention_backend` 的顺序是 `flagos -> vendor:iluvatar -> reference`。
- `FACT`：当前 `vllm_fl/dispatch/backends/flaggems/flaggems.py` 在未设置 `VLLM_FL_USE_FLAGGEMS_ATTN` 时返回 vLLM `TRITON_ATTN`；当前 `vllm_fl/dispatch/backends/vendor/iluvatar/iluvatar.py` 也直接返回 vLLM `TRITON_ATTN`。
- `FACT`：`vllm_fl/platform.py` 通过 `call_op("attention_backend", ...)` 获取类路径，因此 vendor backend 可以提供自有 attention backend 类；该机制不要求修改 vLLM 本体。
- `DECISION`：推荐接入方式是 Iluvatar vendor 自有 backend + Iluvatar dispatch 优先级调整，并在 backend 内仅对明确支持的 BF16、无 sliding-window/alibi/sinks、GQA=8 情况选择新实现，其他情况回退原 `TRITON_ATTN`。不能只改 YAML；YAML 顺序必须与新 backend 和实际 kernel 一起提交。
- `RISK`：如果仅把现有 backend 顺序调成 vendor 优先而不增加底层实现，实质上只是 dispatch 表切换，合规证据不足。

#### F2：当前 kernel 参数事实

- `FACT`：仓库内 vendored unified attention 副本的参数逻辑为 `BLOCK_M=16`（当 GQA ratio<=16），`BLOCK_Q=BLOCK_M//num_queries_per_kv`；MiniCPM 的 GQA=8 时为 `BLOCK_Q=2`。
- `FACT`：2D 路径使用 `TILE_SIZE_PREFILL`，3D 路径使用 `TILE_SIZE_DECODE`；2D/3D 选择条件包含 `max_seqlen_q > 1` 或 `num_seqs > seq_threshold_3D`。
- `FACT`：W1 profiler 已记录设备端 `kernel_unified_attention`，Self CUDA `13.563 s`，占 W1 总 Self CUDA `14.604 s` 的 `92.87%`；`mm_kernel` 为 `898.074 ms / 6.15%`。
- `UNKNOWN`：远端 trace 的 kernel launch 参数/grid 尚未取回，尚不能仅凭 trace 证明实际安装版本与仓库副本完全一致；SSH 控制连接在本次取证时失效。

#### F3：参数扫描约束

- `INFERENCE`：直接增大 `BLOCK_Q` 会同时改变 mixed batch 中 decode 行的工作粒度；不能只用纯 prefill benchmark 选择参数。
- `DECISION`：任何 `BLOCK_M/BLOCK_Q/TILE_SIZE/warps/stages` 扫描必须覆盖真实 mixed 形状：一个约 2048-token prefill chunk 加约 30 个长 KV decode 请求，并单独保留纯 decode 回归检查。
- `DECISION`：阶段一只允许先改 2D prefill/mixed 路径，3D decode 路径保持原实现；候选参数先做数值对拍，再做服务级 A/B。

#### F4：当前工程结论

- 可以开始实现“实质性 Iluvatar vendor attention kernel 优化”，但接入代码和 kernel 代码必须作为一个完整变更审查。
- 报告必须同时给出：原 kernel 与新 kernel 的 CUDA 时间、attention 有效 TFLOP/s、mixed 场景吞吐、decode 回归、准确率结果。
- 在远端连接恢复前不修改 9031/9032，也不直接修改比赛源码；下一步先补纯 decode 对照和安装版本/grid 核验。

#### 阶段 0 取证结果（2026-09-27 15:xx CST）

- `FACT`：从远端 W1 trace 读取到 `kernel_unified_attention` 共 336 个事件，全部 launch 参数为 `grid=[1025, 2, 1]`、`block=[256, 1, 1]`。
- `FACT`：该 kernel 的寄存器数为 118/thread，共享内存为 20,736 bytes，trace 给出的估算 occupancy 为 19%；设备元数据为 Iluvatar BI-V150、32 GiB、16 SM、warp size 64。
- `FACT`：W1 中没有 `reduce_segments` kernel，因而该窗口没有走 3D split-KV reduce 路径；结合 mixed iteration 日志和 `max_seqlen_q>1` 条件，W1 可确认是 2D attention。
- `FACT`：attention kernel 时间最小约 5.277 ms、最大约 75.858 ms；按 trace 时间顺序每 42 个事件暂作一组，8 组 attention 总时间分别约为 `222.752、639.980、1061.518、1483.225、1905.159、2328.370、2749.056、3172.453 ms`。组内单次时间从约 5.3 ms 增长到约 75.9 ms，呈明显长 KV 前缀增长趋势。
- `FACT`：336 个 attention 事件总 CUDA 时间约 `13.563 s`；同一 trace 中 `mm_kernel` 总 CUDA 时间约 `898.074 ms`，KV cache 写入约 `9.862 ms`。
- `FACT`：trace 的 attention grid 与 `BLOCK_Q=2` 的服务配置相符：8 个 2048-token chunk、每 chunk 约 1024 个 query block、2 个 KV heads，得到约 `1025×2` 的 launch grid。
- `FACT`：容器中已安装 vLLM 为 `0.24.0`，其 `/usr/local/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_unified_attention.py` 的默认关键逻辑与仓库 vendored 副本一致：`BLOCK_M=16`、`BLOCK_Q=BLOCK_M//num_queries_per_kv`，prefill 使用 `_get_tile_size(..., is_prefill=True)`。
- `FACT`：FlagGems Iluvatar GEMM 实现定义为 `mm_kernel`，通用 FlagGems 实现定义为 `mm_kernel_general`；W1 trace 中服务 GEMM 名称为 `mm_kernel`，与 Iluvatar 路径一致。

#### 对阶段 1 的约束

- `DECISION`：当前证据支持实质优化 2D prefill attention；不支持把 GEMM 作为优先主线。
- `DECISION`：第一版只扫描 2D 的 `BLOCK_M/BLOCK_Q`、prefill tile 和 launch 参数；3D decode 保持原路径，避免 mixed batch 的 decode 行因统一增大 tile 而退化。
- `DECISION`：参数选择必须同时测真实 mixed 形状和纯 prefill 形状，并记录数值误差；仅纯 prefill 的最佳参数不能直接提交。

### 实验 2026-09-27-16：阶段 1 接入前检查

- 本地固定分支工作树保持干净，未修改比赛源码；新增的 `tools/attention_stage1_plan.md` 仅是诊断计划，不会被服务导入。
- 远端 9031/9032 均保持健康；本阶段没有重启或改变服务参数。
- `FACT`：完整 vendor attention 接入至少需要 `AttentionBackend`、`AttentionImpl` 和 metadata builder 三层 contract，不能只复制 Triton kernel 函数。
- `FACT`：已安装 vLLM 0.24.0 的 `TritonAttentionImpl.forward` 是服务调用边界；新 Iluvatar backend 必须在这一层保持 KV cache layout、slot mapping、metadata 和输出 buffer 语义一致。
- `DECISION`：先建立“原实现语义等价 + 2D 参数可替换 + 3D 原样回退”的诊断实现，再做 kernel 参数扫描；未完成数值对拍前不接入 9032。
- `UNKNOWN`：当前尚未完成完整 `AttentionImpl` 适配和远端编译验证；因此阶段 1 尚不能报告性能收益。

#### 新增兼容性核验

- `FACT`：vLLM 0.24.0 的 `TritonAttentionImpl.forward` 调用 `unified_attention` 时，除基础参数外还传入 `use_alibi_sqrt`、`kv_quant_mode`、`k_scale_cache`、`v_scale_cache`、`chunk_lookback`、`use_td` 等参数。
- `FACT`：仓库内 MetaX vendored `triton_unified_attention.py` 的旧版函数签名不包含这些参数，不能直接复制后接入 Iluvatar。
- `DECISION`：阶段 1 的实现基线必须来自容器中已安装的 vLLM 0.24.0 源码；MetaX 副本只能作为早期 kernel 结构参考。下一步应先导出/复制安装版本，再在 Iluvatar namespace 内做最小参数化修改。
- `RISK`：直接复用旧 MetaX 副本会在服务运行时因参数不匹配或 metadata/quantization 语义差异失败，已明确禁止该路径。

### 实验 2026-09-27-17：阶段 1 前的远端状态收口

#### 已完成的静态与远端核验

- `FACT`：SSH 控制连接恢复，远端 `mllv` 容器仍在运行；9031 官方服务 PID 2943，9032 诊断服务 PID 44068，均未被本轮重启或修改。
- `FACT`：容器版本为 vLLM `0.24.0`、Torch `2.10.0`；插件入口只列出 `fl -> vllm_fl:register`，与固定分支的 `VLLM_PLUGINS=fl` 结论一致。
- `FACT`：9032 日志仍反复报告 `ModuleNotFoundError: No module named 'vllm._C'`。该扩展缺失是环境事实，不能在本轮把它归因成 attention kernel 的性能原因；后续应在独立环境核查是否为镜像安装方式的预期降级。
- `FACT`：从 `/workspace/logs/prof_diag_server.log` 的 621 条 iteration 记录统计：纯 context 32 条、纯 generation 381 条、mixed 208 条；诊断运行的 generation 并发最高仅 8。
- `INFERENCE`：9032 确实存在纯 generation step，因此“纯 decode 路径完全不存在”的假设不成立；但最高并发为 8，不能外推到官方 4k/16k 的 30~64 并发 decode 成本，也不能替代 W2 目标实验。
- `FACT`：诊断日志中出现的较长 iteration elapsed time 包括约 `1314.05、265.16、243.64、231.19、102.11 ms` 等值，但该字段是 engine iteration 侧记录，不等价于单个 CUDA attention kernel 时间，不能直接用于计算 kernel 加速比。
- `FACT`：静态核对确认 vLLM 0.24.0 的 `unified_attention` 对 MiniCPM 的默认参数为 `BLOCK_M=16`、`BLOCK_Q=2`、prefill `TILE_SIZE=32`；W1 trace 的 `grid=[1025,2,1]` 与该配置一致。
- `DECISION`：阶段 1 的第一版应以已安装 vLLM 0.24.0 源码为语义基线，在 Iluvatar vendor namespace 中只改变 2D prefill/mixed 的参数或 kernel 实现；3D decode 保持原实现，unsupported shape 回退。

#### 本轮未完成及原因

- `UNKNOWN`：尚未完成新的 64 并发纯 decode profiler trace；9032 的现有短运行只覆盖到并发 8，不能填补该证据缺口。
- `UNKNOWN`：尚未开始比赛源码接入、数值对拍或真实 mixed shape 参数扫描；原因是尚未在远端完成可回滚的第一版 kernel 编译与验证，贸然接入会把接口风险和性能风险混在一起。
- `DECISION`：本轮不修改 `vllm-plugin-FL` 比赛源码，不改 YAML，不重启 9031；保留当前干净工作树，先把接入实现和对拍 harness 作为下一阶段独立变更。

#### 当前结论

- `✓ 已确认`：W1 中 2D unified attention 是主要设备瓶颈，336 次 CUDA 时间 `13.563 s`，占 Self CUDA `92.87%`。
- `✓ 已确认`：服务 GEMM 走 FlagGems Iluvatar `mm_kernel`，W1 占比约 `6.15%`，不作为当前主线。
- `✓ 已确认`：vLLM 0.24.0 调用 contract 与旧 MetaX vendored attention 不兼容，不能直接复制旧副本。
- `✓ 已确认`：9032 有纯 generation 与 mixed step，但本轮并发规模不足以完成 decode 回归判断。
- `⚠ 待验证`：Iluvatar 上 `BLOCK_M/BLOCK_Q` 候选对真实 mixed batch 的收益与 decode 副作用。
- `⚠ 待验证`：新实现的数值误差、服务吞吐、TTFT/ITL 回归和准确率。
- `❌ 不采用`：仅通过环境变量或 dispatch 顺序切换已有 attention backend 作为提交方案。

### 审查修订 2026-09-27-18：阶段一执行门槛调整

- `FACT`：阶段一接入路径需在报告中完整呈现：`vllm serve -> FLWorker.init_cache_engine -> dispatch loader -> iluvatar.yaml -> IluvatarBackend.attention_backend() -> IluvatarAttentionBackend -> optimized unified attention`。
- `DECISION`：最终提交的 YAML 必须直接体现 Iluvatar vendor 路径；不能先切换再提交 revert，也不能只提交 dispatch 开关。
- `DECISION`：harness 增加两类形状。纯 prefill 使用 `8 x 2048` query，在前缀 `0/2/4/6/8/10/12/14k` 扫描；mixed 使用 `1 x 2048 prefill + 约30 x 1 decode`，至少覆盖 `0/8/14k` 前缀。
- `DECISION`：参数候选扩大为 `BLOCK_Q={4,8,16,32,64,128}`、`BLOCK_M={32,64,128}`，但必须以编译资源和数值正确性为前提；不假设大 tile 一定更快。
- `DECISION`：harness 入服务门槛调整为有效算力 `>=15 TFLOP/s` 或相对基线 `>=4x`，且 mixed 中 decode 拖尾 `<20%`；这只是筛选门槛，不是性能承诺。
- `DECISION`：Level 3 官方参数双跑前置到参数调优开始前，记录每题结果和可获得的 top-3 logprob，先确定准确率噪声。
- `DECISION`：服务级 A/B 前在 9032 对比 `FULL_DECODE_ONLY` cudagraph 开关下的 decode 吞吐；9031 始终不动。
- `UNKNOWN`：BLOCK_Q=32/64/128 是否能在 BI-V150 的寄存器和共享内存限制下编译，以及 mixed decode 是否拖尾超过20%。
- `UNKNOWN`：Level 3 双跑的实际噪声范围。

### 工具准备 2026-09-27-19：双形状扫描与 9032 Level 3 命令

- 新增 `tools/stage1/scan_config.json`：固定 MiniCPM attention shape、纯 prefill 8 个前缀点、mixed 3 个前缀点、BLOCK_Q/BLOCK_M 候选和验收门槛。
- 新增 `tools/stage1/README.md`：规定结果字段、FLOP 估算和执行顺序，避免把纯 prefill 最优参数直接当作 mixed 最优。
- 新增 `tools/stage1/level3_9032_command.sh`：官方 Level 3 参数的 9032 诊断副本，明确不触碰 9031；本轮未启动长评测。
- `DECISION`：下一次远端执行先进行 Level 3 双跑，再进行 kernel 数值对拍和参数扫描；服务 A/B 最后进行。

### 实验 2026-09-27-20：阶段一优化实现的 9033 隔离服务验证

#### 实验边界

- `FACT`：本轮只在容器 `mllv` 内的隔离端口 `9033` 运行，GPU 2 使用；9031 官方服务和 9032 诊断服务均未重启、未修改。
- `FACT`：9033 使用 `BLOCK_M=128, BLOCK_Q=16, TILE=16, num_warps=8, num_stages=2`，关闭 prefix caching，`cudagraph_mode=FULL_DECODE_ONLY`，seed=`12345`。
- `FACT`：服务启动日志明确记录 `Op 'attention_backend' using 'vendor.iluvatar'`；本轮 `stage1_service_9033.log` 未发现 attention fallback、`Traceback` 或请求失败。
- `FACT`：本轮是并发 8、8 个请求、输出 256 的短基准，不是官方并发 64/128 的比赛成绩，不能直接与官方分数横比。

#### 9033 短基准结果

| 场景 | 成功请求 | 总吞吐 | 输出吞吐 | Mean TTFT | Median ITL | Mean TPOT |
|---|---:|---:|---:|---:|---:|---:|
| 4k 输入 / 256 输出 / 并发 8 | 8/8 | `3657.42 tok/s` | `60.74 tok/s` | `2251.6 ms` | `22.06 ms` | `77.19 ms` |
| 16k 输入 / 256 输出 / 并发 8 | 8/8 | `3046.81 tok/s` | `46.86 tok/s` | `18535.51 ms` | `44.47 ms` | `95.36 ms` |

- `FACT`：16k 四轮短测均成功；后续三轮稳定值约为 `3045.86~3048.91 tok/s`，最终汇总为 `3046.81 tok/s`，说明没有出现单轮异常抖动。
- `FACT`：服务 iteration 日志在请求完成后保持纯 generation step，典型 8 请求 iteration elapsed 约 `39 ms`；随着请求数下降，耗时逐步降至约 `18~36 ms`，与服务正常排空一致。
- `INFERENCE`：优化路径已经完成“能接入、能编译、能服务请求”的第一阶段闭环；但短基准只能证明隔离服务层面的可运行性和相对稳定性，不能证明官方负载下的收益。

#### 与微基准的联合判断

- `FACT`：纯 prefill 0k 的最佳配置有效算力约 `16.97 TFLOP/s`，达到阶段一 `15 TFLOP/s` 筛选门槛；长前缀 14k 约 `9.57 TFLOP/s`。
- `FACT`：mixed（1×2048 prefill + 30×1 decode）在 14k 前缀下约 `7.25 TFLOP/s`，decode proxy 约 `21.13%`，略超过预设 `<20%` 门槛。
- `FACT`：`BLOCK_M=64, BLOCK_Q=8` 的 14k mixed decode proxy 约 `14.3%`，但总 kernel 时间约 `28.40 ms`，明显慢于 `BLOCK_M=128, BLOCK_Q=16` 的约 `19.20 ms`。
- `DECISION`：暂保留 `128/16` 作为主候选，同时保留 `64/8` 作为 decode 回归对照；不能仅凭纯 prefill 峰值提交 `128/16`。

#### 当前未完成项

- `UNKNOWN`：尚未在 9032 上完成同参数短 A/B，因此暂不能把 9033 的吞吐数字表述为相对原实现的收益。
- `UNKNOWN`：Level 3 双跑尚未执行，准确率噪声和优化路径的最终精度安全边界未建立。
- `UNKNOWN`：尚未完成官方并发 64/128 的 16k 服务级复测；短基准不能回答 KV 容量、排队和 mixed 长尾在正式负载下是否改善。
- `RISK`：`BLOCK_Q=16` 的 mixed decode proxy 略超 20%，正式服务若出现 median ITL 或长尾回归，需要进一步拆分 prefill/decode launch，或转向 `64/8`。

#### 收口动作

- `FACT`：9033 服务已停止并释放运行请求；9031/9032 保持原状态。
- `DECISION`：下一步按顺序执行：9032 同形状短对照、Level 3 双跑、再决定是否进入正式 16k/4k A/B；在这些证据完成前不提交最终优化版本。

#### 9032 原实现对照结果（同日追加）

- `FACT`：已向未修改的 9032 诊断服务发起同参数短对照，4k 输入 / 256 输出 / 并发 8 已完成 `8/8`，总吞吐 `1979.38 tok/s`，Mean TTFT `6058.49 ms`，Median ITL `23.01 ms`。
- `FACT`：16k 输入 / 256 输出 / 并发 8 已完成 `8/8`，总吞吐 `1023.95 tok/s`，Mean TTFT `67010.42 ms`，Median ITL `46.75 ms`，Mean TPOT `243.55 ms`。
- `FACT`：9032 benchmark 已结束，仅留下正常退出后的僵尸父进程；9031/9032 服务进程仍在运行。
- `INFERENCE`：9033 的 16k `3046.81 tok/s` 相比 9032 的单轮 `1023.95 tok/s` 显示出强性能信号，但 9032 启用了 torch profiler 配置，且两边均未做多轮、同 GPU 状态和编译缓存完全对齐，因此暂不作为正式加速比。
- `DECISION`：该结果足以进入“准确率和正式 A/B 验收”，不足以直接提交比赛版本；后续正式 A/B 必须关闭 profiler 干扰并按官方并发与重复轮次执行。
- `PROBLEM`：本地 SSH 控制 socket 曾在轮询过程中消失，导致结果读取延迟；恢复后已补齐日志，未重复执行已完成的 4k 对照。

#### Level 3 准确率双跑（已完成）

- `FACT`：第 1 轮已完成，9032 原实现诊断服务得到 `102/105 = 97.1%`；EvalScope 报告目录为 `/workspace/evalscope-datasets/level3_diag_run1_20260927/20260927_173507`。
- `FACT`：第 2 轮已完成，得到 `102/105 = 97.1%`；报告目录为 `/workspace/evalscope-datasets/level3_diag_run2_20260927/20260927_180234`。
- `FACT`：第 2 轮平均延迟 `120.805 s`、平均吞吐 `27.33 tok/s`；第 1 轮平均延迟 `92.206 s`、平均吞吐 `33.28 tok/s`。两轮准确率相同，但评测耗时和吞吐有明显波动，说明性能指标不能用单轮 Level 3 结果替代正式 benchmark。
- `INFERENCE`：在当前采样配置和 9032 原实现上，Level 3 总分噪声至少未超过 `0` 题；两轮均保持 `102/105`，当前可将 `97.1%` 作为精度验收基线。
- `UNKNOWN`：尚未读取两轮逐题 JSONL 做题目级 diff；总分相同不等价于每一道题完全一致，后续恢复连接后可补充逐题差异统计。
- `DECISION`：准确率双跑已满足进入正式服务 A/B 的前置条件；优化版本必须仍达到 `>=95%`，并优先与这两个 `97.1%` 基线结果比较。

### 实验 2026-09-27-21：Split-KV harness 计时修订与 GPU 2 smoke

#### 范围与安全边界

- `FACT`：本轮执行在远端 `ub39` 的 `mllv` 容器、GPU 2；9031 `/health` 返回 HTTP 200，未重启或发请求。9032/9033 未运行。
- `FACT`：GPU 2 探针显示约 `31.9 GiB / 32 GiB` 可用；容器为 vLLM `0.24.0`、Torch `2.10.0`、16 张 Iluvatar BI-V150。
- `FACT`：远端主机时间显示 `2026-09-28 01:xx CST`，与任务侧 `2026-09-27` 不一致；本记录保留任务日期，远端日志时间以原始日志为准，时间差需后续确认。
- `FACT`：实际优化模块必须由 `PYTHONPATH=/workspace/stage1_patch/vllm-plugin-FL` 导入；`/workspace/stage1_patch/vllm_fl` 是另一份旧副本，不能用作本轮 kernel 来源。

#### harness 修改

- `FACT`：`tools/stage1/attention_harness.py` 将计时改为批量 CUDA event timing，并同时输出 wall time；重复 kernel launch 复用 output buffer，避免同步式单次 Python 调用主导有效 TFLOP/s。
- `FACT`：pure decode probe 现在提供 16-segment workspace、`seq_threshold_3D=64`，按 vendor backend 的 3D decode 路径运行，并在重复计时中复用 workspace。
- `FACT`：原 `decode_only_over_mixed_ratio` 命名被替换为 `decode_cost_proxy_ratio`；定义为独立 pure-decode 调用时间 / 完整 mixed 调用时间，明确不是可归因的 decoder tail fraction。
- `FACT`：结果新增 `baseline_median_ms`、`speedup_vs_vllm`、`timing_source`、候选/基线 wall median 等字段。
- `FACT`：本地 `py_compile` 通过；同步到容器后的脚本哈希为 `19ecddeec1514495655496002b5e5f0c4e271e62dacf8e4d4f256c4d60cbe12d`，README 哈希为 `502e495ed73f654c8a8590abe452a6c5ecfef5da244b381f1071dec328eb1f11`。

#### 0k mixed smoke（128/16，4 segments）

- `FACT`：1 个 2048-token prefill + 30 个 decode，BF16、paged KV；编译时间 `0.808 s`，warmup 2，event timing 为 3 组、每组 10 次。
- `FACT`：候选 median `2.221 ms`，原生 vLLM baseline median `5.159 ms`，同 shape kernel speedup `2.323x`；wall median 分别为 `2.424 ms` 与 `5.281 ms`。
- `FACT`：有效算力 `7.737 TFLOP/s`，低于阶段筛选目标 `15 TFLOP/s`，也未达到 `4x` 相对提升。
- `FACT`：相对 fp32 paged-SDPA reference 的 max/mean absolute error 为 `0.0078125 / 7.153e-05`，通过当前 harness 条件；相对原生 BF16 vLLM 输出的 max/mean diff 为 `1.8125 / 0.008634`，不可与 reference 误差混为一谈，需保留逐元素误差分布分析。
- `FACT`：pure decode 独立 median `0.03520 ms`，decode cost proxy `1.585%`；该值不是 mixed kernel 中 decoder 行的实测尾延迟。
- `FACT`：启动时 vLLM 插件自动发现打印 `module 'vllm_fl' has no attribute 'register'`，但 harness 继续执行且直接导入了 staging 内优化 kernel；这是诊断脚本的插件发现噪声，后续应通过关闭无关插件自动加载消除，避免和 kernel 错误混淆。

#### 判断与下一步

- `INFERENCE`：基于该单形状 smoke，Split-KV mixed 候选在本次状态下比原生实现快约 `2.32x`；单点数据不能宣称稳定服务收益。
- `UNKNOWN`：`BLOCK_M=128/BLOCK_Q=16` 在 8k/14k 前缀及 segment 数 `2/4/8/16` 下的吞吐、精度误差变化与编译资源。
- `DECISION`：先完成 8k/14k 的分段数微基准；仅在正确性持续通过且对照收益明确后，才回到隔离服务 A/B。9031 保持不动。
- `RISK`：该 harness timer 以 CUDA event 包围重复异步 launch，能显著降低单次 host 调用偏差，但仍不等价于 profiler 提取的单 kernel device duration；报告须同时列出 event 与 wall 指标。

### 方案审查与复测准备（2026-09-27）

#### 附件方案的适用边界

- `FACT`：附件中的 FlashAttention-3 说明、分阶段计划和吞吐/消融表属于设计材料及目标示例，不是 BI-V150 实测结果；不得直接作为技术报告成绩或已完成创新点。
- `DECISION`：本轮先沿用 vLLM 0.24.0 的 paged-KV ABI 和现存 Triton 3D attention 部分和归约逻辑，验证 mixed batch 的 Split-KV。附件的 dense `[B,H,N,D]` 代码片段不直接用于服务：它没有覆盖物理 block table、KV block 边界、chunked prefill、量化 cache、完整 metadata contract。
- `FACT`：Split-KV 改变 KV 轴上的工作分区和并行度，不会减少理论 KV 总读取字节数；L2 命中或更小 working set 的收益必须由硬件计数器/trace证实，不能仅由分块推导。
- `RISK`：Triton `num_stages` 不能等同 Hopper 的 warp specialization、TMA 或 ping-pong 调度。BI-V150 未证明支持这些硬件机制，报告不应称为“等价复现 FlashAttention-3”。
- `RISK`：附件 RoPE 融合假设 attention 输入是尚未旋转的 Q/K；vLLM 的 paged KV 路径会在 cache 写入路径处理 K。融合必须先追踪 MiniCPM 实际 Q/K、RoPE、reshape/cache 的调用与数据语义，不能直接把 RoPE 放进 attention kernel，否则可能对已旋转 K 重复旋转或破坏 cache。
- `RISK`：附件 dual-stream 示例未给出 vLLM 实际 metadata 的请求级拆分、输出索引映射、cache/table 子集和 stream event 依赖；也未证明同一批次能在设备上有益并行。暂不实现双 stream 调度。

#### 已下载的 segment 扫描数据复核

- `FACT`：取回 8 份 JSONL，全部 `status=ok`，覆盖 KV segment 数 `2/4/8/16` 与 mixed 前缀 `8192/14336`；每行都有 paged-SDPA 数值对照，`sdpa_max_abs_error` 为 `4.8828125e-4`。
- `FACT`：原结果文件名包含 segment 数，但 JSON 行没有记录该字段；本地汇总需从文件名解析，原始证据不覆盖、不改写。
- `FACT`：原 harness 的 `p95_ms` 是 3 个“每组 10 次调用的平均值”中的最大次序统计量，不是单次延迟分布的 P95。报告应弃用该字段，保留每组均值及 max，不能据此声称尾延迟。
- `FACT`：原 `effective_tflops` 估算式遗漏 causal 可见 token 对的精确计数，且只按每个 query-key 对 2 FLOP 计数；attention 同时含 QK 与 PV 两次乘加，标准 FLOP 计数需各按 2 FLOP/MAC。旧表中的有效 TFLOP/s 不与修订后的基线/门槛直接比较；同 shape 的 latency speedup 仍可作探索性对照。
- `DECISION`：复测结果 schema 显式记录 Split-KV 开关和 segment 数；采用 causal query-key pair 数 `q*prefix + q*(q+1)/2`，并计入 QK/PV 两部分；将重复批次均值原样输出，不再输出误导性的 P95。沿用同一 FLOP 定义重算基线与候选后，才讨论有效算力门槛。

#### 本轮本地改动与待补复测

- `FACT`：更新 `tools/stage1/attention_harness.py`、`README.md`、`scan_config.json` 和 `scan_split_segments.sh`；新增 repeat 参数（默认 5）、请求级 causal FLOP 估计、segment 配置元数据与重复样本记录，复测输出目录独立于旧结果。
- `FACT`：本地 `py_compile`、扫描脚本 `bash -n`、配置 JSON 解析均通过；更新版 harness 和 shell 脚本已同步到远端 staging 目录，容器内 `py_compile` / `bash -n` 通过。
- `FACT`：9031 `/health` 仍为 200；9032/9033 当前未运行。本轮只在 GPU 2 运行隔离微基准，不调用 9031、不改官方服务。
- `UNKNOWN`：5 组 repeat 的 segment 扫描是否完成、数值稳定性、长前缀最优 segment 数，以及更新口径下有效 TFLOP/s，等待本轮原始 JSONL 复核后填写。
- `DECISION`：微基准筛选通过后，服务级验证只在 GPU 1/9032 做隔离 A/B；9031 保持官方基线不动。微基准 latency 改善不得直接写成比赛吞吐收益。

#### 5-repeat GPU 2 复测结果（追加）

- `FACT`：复测产出 8/8 份 JSONL，全部 `status=ok`，每项含 5 个独立计时批次均值；所有 paged-SDPA `sdpa_max_abs_error=0.00048828125`。远端文件时间为 `2026-09-28 02:03-02:04`，较任务侧日期快一天，故以原始文件时间为准，不据此改写任务日期。
- `FACT`：以下候选 median 与 speedup 来自同一 GPU 2 harness，形状为 `1×2048 prefill + 30×1 decode`。baseline TFLOP/s 由相同 causal FLOP 估计除以实测 baseline median 推导；所有值均为 microbenchmark，不是服务吞吐。

| 前缀 | Split 数 | Candidate ms | Baseline ms | Speedup | Candidate TFLOP/s | Baseline TFLOP/s | 5 组计时跨度 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8k | 2 | 18.573 | 43.998 | 2.368x | 8.434 | 3.561 | 18.558–18.591 ms |
| 8k | 4 | 18.158 | 43.998 | 2.422x | 8.627 | 3.562 | 18.155–18.166 ms |
| 8k | 8 | 18.504 | 43.964 | 2.377x | 8.465 | 3.561 | 18.492–18.511 ms |
| 8k | 16 | 20.424 | 43.998 | 2.155x | 7.669 | 3.560 | 20.417–20.842 ms |
| 14k | 2 | 31.619 | 73.490 | 2.324x | 8.262 | 3.555 | 31.597–31.633 ms |
| 14k | 4 | 30.577 | 73.518 | 2.403x | 8.543 | 3.556 | 30.571–30.599 ms |
| 14k | 8 | 30.297 | 73.515 | 2.427x | 8.622 | 3.553 | 30.295–30.305 ms |
| 14k | 16 | 31.750 | 73.461 | 2.315x | 8.228 | 3.555 | 31.742–31.759 ms |

- `FACT`：本轮在 8k 最快为 4 splits，14k 最快为 8 splits；16 splits 在两个前缀都较慢。5 批次跨度很小，但这是同一进程中的重复均值，不是跨进程、跨时段统计置信区间。
- `FACT`：Split-KV candidate 在该 mixed microbenchmark 中相对原生 unified attention 为约 `2.15–2.43x`；最优项仍低于 `15 TFLOP/s` 和 `4x` 两个进阶段门槛。没有理由把它升级成最终服务/比赛成绩。
- `FACT`：记录中的 `decode_cost_proxy_ratio` 约 `7.2–8.3%`，其 decode-only 分子是单独的 16-segment pure-decode probe，不能解释成混合 kernel 中 decoder 行的尾延迟或真实 ITL。
- `DECISION`：将 `4 splits @ 8k`、`8 splits @ 14k` 保留为下一阶段候选；不根据不到约 1–2% 的微基准差异直接写入运行时自动策略。后续先做随机顺序/ABBA 重复，再以 9032 的真实 median ITL、TTFT 和服务吞吐判断。
- `FACT`：更新版 harness 增补 `baseline_effective_tflops` 记录；该纯输出字段变化不改变刚完成复测的 kernel 路径和计时。新版再次同步到 staging，容器内 `py_compile` 通过。
- `DECISION`：当前待做的实验不是整夜无限扫描，而是有停止条件的两项：先用随机顺序复测确认 4/8 split 的差异；再在 GPU 1/9032 进行相同参数的隔离服务 A/B。未达到 correctness、decode regression 和端到端收益门槛，不将实验开关固化为默认行为。

#### Decode segment 副作用探针（已完成）

- `FACT`：为了补足此前 mixed harness 只测了候选 mixed 时延、未测对应低 segment 数 3D decode 成本的缺口，新增独立 decode probe：按被测 mixed 候选 segments 运行 decode-only 3D kernel，并与 native 16-segment decode 对照；输出 candidate/baseline decode latency 和 speedup。
- `RISK`：该 decode-only probe 只能隔离 segment 数对 decode kernel 本体的影响，不能模拟 mixed prefill 同时占用设备时的尾延迟，也不代表服务 pure-decode 默认路径（服务仍使用 16 segments）。最终 ITL 回归仍只能由服务级 A/B 判定。
- `FACT`：更新后的 harness、本地 `py_compile`、shell `bash -n` 及容器内语法验证均通过；新的 8 组合复测写入远端独立目录 `/workspace/logs/split_kv_segment_scan_repeat5_decodeprobe`，不覆盖上一批数据。
- `FACT`：复测 8/8 行均完成、均为 `status=ok`、5 repeats；paged-SDPA max error 均为 `0.00048828125`。mixed latency 与前一轮基本一致。

| 前缀 | Split 数 | Mixed ms | Mixed speedup | Candidate decode ms | Native 16-seg decode ms | Decode speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 8k | 2 | 18.577 | 2.365x | 1.4558 | 1.4792 | 1.016x |
| 8k | 4 | 18.158 | 2.423x | 1.4501 | 1.4752 | 1.017x |
| 8k | 8 | 18.494 | 2.380x | 1.4516 | 1.4734 | 1.015x |
| 8k | 16 | 20.431 | 2.154x | 1.4750 | 1.4752 | 1.000x |
| 14k | 2 | 31.622 | 2.323x | 2.5257 | 2.5019 | 0.991x |
| 14k | 4 | 30.599 | 2.402x | 2.5086 | 2.5016 | 0.997x |
| 14k | 8 | 30.321 | 2.424x | 2.5027 | 2.4995 | 0.999x |
| 14k | 16 | 31.749 | 2.315x | 2.4958 | 2.4979 | 1.001x |

- `FACT`：低 segment 数对隔离 decode-only kernel 的变化在约 `-0.9%` 至 `+1.7%` 范围；该量级不足以证明服务 mixed ITL 不退化，但未显示明显的纯 decode kernel 惩罚。8k 以 4 segments 最快；14k 的 4/8/16 segments decode 延迟近似相同，8 segments mixed latency 略优。
- `INFERENCE`：当前最简候选可先统一使用 4 segments；若保留上下文长度自适应，14k 选择 8 segments 的 microbench 优势约 `0.9%`，必须经服务 A/B 证明后才值得增加策略复杂度。
- `DECISION`：该 decode probe 结果足以让我们把低 segments 保留为隔离服务 A/B 候选，但不替代 GPU 上 mixed 并发服务的 ITL/TTFT 测量。下一步应优先跑 9032/GPU 1 的短、同形状 A/B，9031/GPU 0 继续保持不动；candidate 与 baseline 都使用相同官方请求参数、prefix cache 关闭、无 profiler。

### 状态续记 2026-09-27：服务级 A/B 前的连接与本地验证

- 目标：恢复当前实验并确认是否可以开始 9032/GPU 1 的隔离服务 A/B；9031 官方服务保持不可触碰。
- 附件处理：`MiniCPM-FlashAttention3-Complete.md` 中的 kernel 片段、FlashAttention-3 硬件等价描述、RoPE 融合、双 CUDA stream 和性能目标属于方案/推导，不是 BI-V150 验证事实。本轮继续沿用已验证的 vLLM 0.24.0 paged-KV 接口；不直接复制附件 dense attention 代码，也不把 Triton `num_stages` 描述为 TMA/warp-specialization 等价实现。
- 已有实现状态：插件工作树包含 Iluvatar vendor attention 接入、可调 2D `BLOCK_M/BLOCK_Q`，以及 opt-in mixed Split-KV 路径；Split-KV 通过现有 3D partial-output/reduction 实现，并受环境变量控制。当前有未提交的用户/实验改动，本轮未覆盖或清理。
- 本地验证：优化 attention 两个 Python 文件 `py_compile` 通过；插件 `git diff --check` 通过；stage1 shell 脚本 `bash -n`、JSON 配置解析通过。
- 连接检查：本地 SSH ControlMaster `ssh -O check` 报告 master PID `89336` 运行，但随后的只读远端命令（容器进程/health 查询、Docker 容器状态、hostname）均在约 15-25 秒内无输出超时。故 socket 进程存在不等于远端命令通道可用；本轮未获得新的远端容器、服务、GPU 或 benchmark 状态。
- 影响与安全边界：本轮没有停止、重启、改写或向 9031/9032/9033 发请求；没有启动新服务 A/B；没有同步新代码或改变远端文件。
- 当前结论：已有 GPU 2 microbenchmark 中 mixed Split-KV 相对原生 unified attention 约 `2.15-2.43x`，最佳候选约 `8.6 TFLOP/s`；这不是端到端服务加速，且尚未达到原定 `15 TFLOP/s` 或 `4x` 微基准门槛。低-segment decode-only 探针约在原生 16-segment 的 `-0.9%` 至 `+1.7%` 范围，也不能替代服务 ITL 回归。
- 下一步：先由用户在 iTerm 核查或重建 SSH ControlMaster；恢复后第一步只读检查 `mllv`、9031/9032/9033 进程与健康状态及 GPU 占用。确认 9032 空闲且可安全重启后，才对 9032 做随机顺序的同参数短 A/B（原生 vs 4/8 segment candidate、prefix cache 关闭、profiler 关闭）；9031 始终不动。若 9032 状态不明确，则停止实验，不发送请求。

### 状态续记 2026-09-27：SSH 控制通道再次失去响应

- `FACT`：本地 `ssh -O check` 仍显示 ControlMaster PID `89336` 存活，但通过该 socket 执行最小远端命令及容器只读检查均在约 10-25 秒内无输出，不能据此确认远端状态。
- `FACT`：本轮未停止、重启或向 9031/9032/9033 发请求；未同步代码；未改变远端文件。
- `FACT`：本地 `attention.py`、`triton_unified_attention_optimized.py`、`attention_harness.py` 均通过 `py_compile`；`scan_split_segments.sh` 通过 `bash -n`；`scan_config.json` 可由 `json.tool` 解析；插件工作树 `git diff --check` 通过。
- `FACT`：已发送 `ssh -O exit` 关闭无响应的本地 ControlMaster，避免继续把“master 进程存活”误判为可用连接。
- `PROBLEM`：当前没有可验证的远端 mllv/9032/GPU 状态，因此不能安全启动 A/B 或发送 benchmark 请求。
- `DECISION`：等待用户在 iTerm 重建 SSH ControlMaster；恢复后严格按只读检查 → 确认 9032 空闲 → 原生/4-segment/8-segment 随机短 A/B 的顺序继续，9031 保持不动。

### 实验 2026-09-28：Split-KV ABI 审计与活动路径收敛

- `FACT`：独立仓库原活动文件 `kernels/triton_unified_attention_optimized.py` 已按原始提交哈希归档为 `reference/experimental_unvalidated_attention.py`；其中 RoPE fusion、warp specialization、ping-pong 和 TMA 没有 BI-V150 实测闭环，不能继续作为活动优化或性能声明。
- `FACT`：新增 `triton_split_kv_paged.py`，使用 vLLM flattened query、`cu_seqlens_q`、非连续 `block_table` 和真实 paged-KV strides；forward launch 将 `(query_block, split)` 打包到 Triton 的第三个 grid 轴，避免原实现四维 grid 与未读取 `program_id(3)` 的重复执行风险。
- `FACT`：Split-KV 仅允许 causal、未量化、无滑窗/ALiBi/sinks/softcap 的 prefill/mixed 路径；纯 decode 和不支持条件继续使用原生 vLLM attention。
- `FACT`：README、`docs/architecture.md`、`scripts/run_benchmark.sh` 和安装脚本已更新，不再启用或宣称未验证的四类优化；新增 `tests/test_paged_split_kv.py` 覆盖随机物理 block table、非均匀 query 长度、非对齐边界和 4/8 segments。
- `INFERENCE`：此前 GPU 2 的 `2.15–2.43x` Split-KV 微基准结果属于修复前实现/ABI 口径，不能直接作为修复后性能证据，必须重跑。
- `UNKNOWN`：修复后 Triton kernel 在 BI-V150 的编译、数值误差和服务级收益尚未验证。
- `DECISION`：先在 GPU 2 运行 `tests/test_paged_split_kv.py`；正确性通过后再运行 4/8 segments 微基准，最后才重启 9032 做短 A/B。9031 不停止、不重启、不发送请求。

### 实验 2026-09-28-02：修复版 correctness gate 与 launch mapping

- `FACT`：远端 `/workspace/split_kv_repair_20260928` 中修复版 paged Split-KV correctness gate 通过 6/6 组，覆盖随机物理 block table、非均匀 query 长度、非对齐边界和 4/8 segments；最大误差 `0.015625`，平均误差约 `3e-4`。
- `FACT`：首次 mixed 微基准发现按 `max_query_len` 为所有请求发射 query blocks 会让 `1x2048 + 30x1` 候选达到 `1110.29 ms`，native 为 `43.95 ms`；这是 launch mapping 缺陷，不是性能结论。
- `FACT`：forward/reduce 已改为按请求实际 query block 数压紧映射，并重新通过 6/6 correctness gate。
- `UNKNOWN`：压紧映射后的 GPU 2 latency、编译资源和服务级收益尚未完成。
- `DECISION`：继续只在 GPU 2 重测；9031/9032 不启动 candidate traffic，直到修复后 microbenchmark 有稳定结果。

### 实验 2026-09-28-03：活动实现收口与可复现集成补丁

- 时间：2026-09-28
- 目标：清除未完成的概念性算子，保证活动路径只包含能够在 BI-V150 上对拍和复测的实现，并同步服务接入修复。
- `FACT`：独立仓库活动实现继续只包含 paged Split-KV forward、online softmax partial-output/reduction、flattened variable-length query ABI、真实 `block_table` 索引和 GQA 映射。
- `FACT`：RoPE fusion、warp specialization、ping-pong scheduling、TMA 及旧版 FA3/Split-KV 草稿不再属于活动路径；它们不具备本环境下完整的调用链、硬件语义或实测闭环，保留的历史草稿只位于 `reference/experimental_unvalidated_attention.py`。
- `FACT`：删除 `vllm-plugin-FL` 工作树中未接入的 `attention_config.py`、`triton_fa3_kernels.py`、旧 Split-KV 变体和相关 ops 草稿；活动 vendor 目录只保留 `impl/attention.py`、`impl/ops/triton_split_kv_paged.py`、`impl/ops/triton_unified_attention_optimized.py` 及包初始化文件。
- `FACT`：修复服务门控：只有同时设置 `ILUVATAR_USE_OPTIMIZED=1` 和 `ILUVATAR_SPLIT_KV=1` 才进入 candidate；否则使用原生 vLLM attention。该修复避免 baseline 仅因误留 `ILUVATAR_SPLIT_KV=1` 而污染。
- `FACT`：移除没有实际 kernel 语义的独立 `BLOCK_Q`/prefill tile 参数；活动实现只接受 `BLOCK_M`、`BLOCK_N`、`num_splits`、`num_warps`、`num_stages`。
- `FACT`：新增 host-side ABI 校验，检查设备、形状、请求数、`cu_seqlens_q`/`seqused_k`、block size 和 tile 对齐关系；通过后才发射 Triton kernel。
- `FACT`：独立仓库新增 `integration/vllm-plugin-FL-split-kv.patch` 和 `integration/README.md`，包含 YAML 路由、Iluvatar backend 接入、双重门控和活动 kernel 文件，避免“本地修了但 GitHub 无法复现”。
- 验证：本地 `py_compile` 已通过；GitHub 仓库文档、harness 配置和安装脚本已同步更新；插件工作树 `git diff --check` 通过。
- 未完成：远端 SSH ControlMaster 当前可检查但执行命令无响应，故本轮没有重启 9032、没有发送 candidate 请求，也没有把修复后的性能写成结论；9031 未触碰。
- 下一步最小命令：恢复可用 SSH 通道后，只读确认 mllv/9031/9032 状态；在 GPU 2 运行修复版 correctness 及 4/8 segments 复测，完成后才做 9032 隔离 A/B。

### 实验 2026-09-28-04：修复后 Split-KV 回归与 RoPE Fusion 可行性

- 平台/设备：ub39、mllv、Iluvatar BI-V150 GPU 2；9031 和 9032 在只读检查时均 HTTP 200，9032 的队列与 KV 使用率为 0。没有重启服务或给 9031 发送 candidate 请求。
- 目标：读取已完成的修复版结果，确定 RoPE fusion 与 warp specialization 是否能在固定框架内成为真实而非概念性的服务优化。
- `FACT`：此前后台实验 `/workspace/logs/split_kv_repaired_gpu2_20260928/segments_4_prefix_8192_fixed.jsonl` 已完成。形状为 1×2048 prefill + 30×1 decode、前缀 8192、split=4；candidate CUDA-event median `75.886 ms`，native `44.022 ms`，speedup `0.580x`；对 fp32 paged-SDPA 参考的最大误差 `0.00048828125`。单次探针不代表跨时段置信区间，但已足以阻止直接部署该 candidate。此前 2.15–2.43x 数据属于修复前代码，不能转用于修复后版本。
- `FACT`：审计发现 `tests/attention_harness.py` 的 baseline 命名曾与真实路由不一致，pure-decode 探针也曾直调 Split-KV；已修正为 baseline 模式只调 native、candidate 只在 prefill/mixed 调修复版，pure decode 恒用 native；数值门槛由两种误差同时超标才失败修正为任意一种超标即失败。`scripts/run_benchmark.sh` 现在按 4/8 splits 指定参数，并使用单次对照中的同一输入。
- `FACT`：新建 `kernels/triton_rope_kv_cache.py`，在单次 GPU launch 内完成 Q 原位旋转、K 旋转后按 slot 写入 paged cache、V 写入 cache；覆盖 NeoX 与 interleaved、rotary_dim=64/128、负 slot。独立脚本 `tests/test_rope_kv_cache.py` 在 GPU 2 首轮 3/4 通过、第四组 max error 0.03125；旋转乘加显式提升至 fp32 后复测 4/4 通过，Q/K/V 与参考结果的最大绝对误差均为 0。
- `FACT`：容器安装的 vLLM 0.24.0 中 `RopeKVCacheFusionPass` 仅当 `pass_config.fuse_rope_kvcache` 为真才注册；`vllm/config/compilation.py:283-288` 对非 ROCm 平台强制将该 flag 改为 False。MiniCPM 模型先执行 `self.rotary_emb(positions,q,k)` 再调用 attention；当前 BI-V150 的 `rocm_aiter_ops.is_enabled()` 为 None。因此独立 kernel 的正确性不等于服务可达，更不等于吞吐收益。
- `DECISION`：撤销插件中尝试加入的 RoPE 导入/开关/路由，插件工作区回到先前 commit；独立 RoPE kernel 只以研究原型形式保留在 GitHub 仓库，尚未进入服务或比赛提交。不能在 attention 对已旋转 Q/K 再旋转，也不在固定 vLLM 本体里绕过平台门禁。
- `FACT`：旧版 Warp Specialization/Pingpong 只展示 `num_warps`、`num_stages`、普通 `tl.load` 与占位 barrier，没有经生成代码或 BI-V150 实测证实 producer/consumer warp 分工、异步访存或双缓冲。TMA 为 Hopper 特性，当前 BI-V150 不具备可验证的等价路径。
- `INFERENCE`：RoPE+cache 融合在技术上可完成且已数值通过，但该调用链的比赛合规集成点不在插件可控范围；即使可达，它优化的是 RoPE/cache 写入，不足以解释 profiler 中 16k attention 自身 92.87% 的 GPU 时间。Warp/Pingpong 可以转为具体的软件流水线或 tile 优化假设，不能先以硬件 FlashAttention-3 特性计分。
- 遇到的问题：SSH ControlMaster 可检查但远端命令间歇性超时；`scp` 经 bastion 无法写入映射目标，改用 `tar | ssh docker exec -i tar` 将文件传入 staging。没有覆盖 `/workspace/vllm-plugin-FL` 官方服务源目录。
- 未完成：修正后的 harness 尚未在 GPU 2 跑完整 4/8、8k/14k 交叉复测；RoPE 尚未做公平的内核延迟对照或服务路由验证。当前不能声称任何新的服务性能收益。
- 下一步：仅在 SSH 稳定时先对 staging 的修复版跑 4/8 split 同形状微基准；若仍慢于 native，停止该候选并回到 2D prefill 真实 tile/profile。RoPE 若未来获得规则许可修改框架编译 pass，再独立设计公平 A/B；此前不让它进入 9032。

### 实验 2026-09-28-05：修正 harness 交叉复测与 mixed decode 分离探针

- 环境：ub39 `mllv`，GPU 2；独立 staging `/workspace/split_kv_repair_20260928`，修正 harness SHA-256 `503600093463931b02d7127da57c45ba96d2044d532b157c9196776163edaab8`。9031 官方服务未操作；9032 未重启、未发送本轮请求。
- 命令：`PYTHONPATH=/workspace/split_kv_repair_20260928 VLLM_PLUGINS=fl python3 tools/repair_bench_20260928/tests/attention_harness.py --config tools/repair_bench_20260928/tests/configs/long_context.json --device cuda:2 --result <对应 jsonl> --shape mixed --prefix-token <8192|14336> --block-m 64 --block-n 64 --split-kv-mixed --split-kv-segments <4|8> --warmup 3 --iterations 10 --repeats 5`。结果在 `/workspace/logs/split_kv_repaired_gpu2_20260928/rerun*20260928.jsonl`，每个 `.exit` 为 `0`。

| 前缀 | Split | Candidate ms | Native ms | 比值 | Candidate TFLOP/s | SDPA max abs |
|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 4 | 76.198 | 43.965 | 0.577x | 2.056 | 0.000488 |
| 8192 | 8 | 74.530 | 44.011 | 0.591x | 2.102 | 0.000488 |
| 14336 | 4 | 125.789 | 73.490 | 0.584x | 2.077 | 0.000488 |
| 14336 | 8 | 119.756 | 73.464 | 0.613x | 2.181 | 0.000488 |

- `FACT`：候选四个形状均数值通过，且时延约为原生的 `1.63–1.73x`；这明确否定了修复前 `2.15–2.43x` 的加速声称。CUDA-event 与 wall 时延接近，不能简单归咎 Python 计时。
- `HYPOTHESIS`：统一 Split-KV 对 mixed 中 30 个 `q_len=1` decode 请求也按 `BLOCK_M=64` 发射 2D tile，浪费矩阵计算并导致拖尾；需在 dispatcher 层拆开 prefill 和 decode 来区分。`vllm-plugin-FL` 本地工作区增加了**未提交诊断原型**（`impl/attention.py:_run_unified_attention`）；独立仓库增加 `tests/dispatcher_mixed_probe.py`，并仅把原型复制到隔离 staging，不修改服务源目录。
- 探针命令：`PYTHONPATH=/workspace/split_kv_repair_20260928:/workspace/split_kv_repair_20260928/tools/repair_bench_20260928/tests VLLM_PLUGINS=fl python3 tools/repair_bench_20260928/tests/dispatcher_mixed_probe.py --device cuda:2 --prefix 8192 --splits 4 --warmup 3 --iterations 10 --repeats 5 --result /workspace/logs/split_kv_repaired_gpu2_20260928/dispatcher4_8k_20260928.txt`。
- `FACT`：dispatcher 探针候选 `43.580 ms`，native `44.000 ms`，`1.010x`；对 native max abs `0.001953`，对 fp32 SDPA max abs `0.000488`，5 repeats。解码行走 native 3D，prefill 行走 Split-KV。该微小差异尚在可能的环境噪声范围，不能称为服务加速。
- `RISK`：原型每次调用对 GPU `cu_seqlens_q` 做 `.cpu().tolist()`，强制 host 同步；还在每层构建新索引、工作区并执行 scatter。当前探针不覆盖 CUDA graph capture 和正式调度，不能直接作为可提交实现或启动 9032 A/B。若继续，必须从 scheduler/metadata 获得无需 GPU→CPU 同步的分组信息，且验证图捕获、各种请求排列和 workspace 语义。
- 问题：14k 和 8-split dispatcher 补测命令发出时，SSH 控制连接连续超时；没有对应 `.txt`/`.exit` 产物，不能假定执行或成功。已停止重复调用。候选默认不开启，本地插件工作区变脏；GitHub 的已推送集成补丁**不包含**这个原型，仍对应旧 `a175b28`。当前不推送原型、不宣称已同步正式修复。
- 决策：保留 8k 探针作为原因辨析，停止原 Split-KV 整批 candidate 的服务验证；先解决无同步分组与 2D prefill 的内核效率，再做 GPU 2 完整对拍，最后考虑 9032。不得触碰 9031。

#### 本地后续修正（尚未 GPU 复验）

- `FACT`：固定插件的 `model_runner.py` 创建 `CommonAttentionMetadata` 时已有 `query_start_loc_cpu=self.query_start_loc.cpu[...]`。诊断原型已改从该 CPU metadata 传入 vendor backend，缺失或非 CPU 时安全回退 native，不再对 GPU `cu_seqlens_q` 执行 `.cpu().tolist()`。
- `FACT`：原型先前按全局 token index 对 `softmax_segm_*` 做 `index_select`；真实 vendor backend 的 scratch 只分配 `seq_threshold_3D` 行（本场景约 64），可能在 2048-token prefill 中越界。现在 prefill 子调用不用 3D scratch，decode 子调用只取前 `num_decode_requests` 行。诊断探针同步改为真实 64 行 scratch，并增加 `--prefill-index` 检查非首行 prefill 和数值失败阈值。
- `UNKNOWN`：这些新改动仅经本地 `py_compile`/`git diff --check`，由于 SSH 命令连续超时，未完成 GPU 2 复验；之前 `1.010x` 只适用于旧诊断原型，不能移植到当前版本。每层索引构建、scatter 和 CUDA graph 仍须验证，当前不发布插件分流代码、不启用 9032。
- GitHub `main` 的 `438c0ec` 已同步本轮交叉复测数据和初版诊断探针；后续诊断脚本/记录可同步，但未经设备复验的插件修正不得加入正式集成补丁。已发布的 `integration/vllm-plugin-FL-split-kv.patch` 仍是旧的整批 Split-KV 候选，不包含分流原型。

### 实验 2026-09-28-06：9032 交叉顺序 A/B（8-split candidate vs native baseline）

- 环境：ub39 `mllv`，Iluvatar BI-V150 GPU 1，端口 9032；固定 `16384 input / 256 output / c16 / 16 prompts`。9031 官方服务未停止、未重启、未发送请求，实验结束后 9032 已释放。
- 目的：用 6 对交叉顺序、每次 benchmark 内置 4 轮且跳过首轮 warmup，判断 8-split candidate 是否有稳定端到端收益。
- 顺序：`baseline -> candidate -> candidate -> baseline -> baseline -> candidate -> candidate -> baseline -> baseline -> candidate -> candidate -> baseline`，共 12 次服务 benchmark，全部返回 `rc=0`。
- 原始日志：远端 `/workspace/logs/cross_ab_c16_20260928/`；编排日志 `/workspace/logs/cross_ab_c16_20260928.orchestrator.log`。
- `FACT`：排除每次 benchmark 首轮后，baseline 18 个 steady runs 的吞吐均值/中位数/CV 为 `9996.83 / 10305.75 tok/s / 13.0%`；candidate 为 `10320.11 / 10312.09 tok/s / 10.5%`。
- `FACT`：candidate 相对 baseline 的总体中位吞吐仅 `+0.06%`，未达到预设的 `+5%`；总体均值约 `+3.2%`，不能作为稳定收益结论。
- `FACT`：steady throughput 的经验 P90 为 baseline `10953.88`、candidate `11541.25 tok/s`。该指标只描述吞吐分布，不等价于 ITL 长尾；candidate 满足“P90 不超过 baseline×1.1”的宽松阈值，但不能据此证明尾延迟改善。
- `FACT`：P99 ITL 的跨轮中位数为 baseline `83.47 ms`、candidate `83.70 ms`，candidate 约高 `0.28%`；Median ITL 均约 `79.7 ms`。
- `INFERENCE`：candidate 的吞吐波动略小于 baseline，但中心位置基本相同；8-split 在 c16 下没有被本轮实验证明为有意义的端到端优化。
- `DECISION`：不把 8-split 固化为默认提交策略；不按“c16 自动 4、c64 自动 8”的假设直接实现自适应。后续若继续，应先在 9032 做同条件的固定 4/固定 8/auto 三组对照，并把判断依据扩展到 prefill 请求数、最大 KV 长度和实际 mixed 形状。
- `UNKNOWN`：本轮未覆盖官方 16k/128 请求/1024 输出、4k/256 请求、candidate Level 3、CUDA graph 回归及 kernel profiler；因此不能据此判断比赛正式负载收益。

### 诊断准备 2026-09-28：dispatcher / Split-KV profiler probe

- `FACT`：新增 `tests/dispatcher_profile_probe.py`，在同一 paged-KV mixed batch 上分别计时 native unified attention、raw Split-KV prefill 子批次和完整 dispatcher（Split-KV prefill + native decode + scatter）。
- `FACT`：probe 通过 scheduler 已有的 `query_start_loc_cpu` 分组，不对 GPU `cu_seqlens_q` 执行 `.cpu().tolist()`；同时记录 `_request_partition`、`_token_indices`、`_subset_attention_kwargs`、optimized/native launch 的 CPU wall time。
- `FACT`：probe 使用 `torch.profiler` 汇总 CUDA kernel、`aten::index_copy_`/拷贝事件、forward/reduce kernel 名称和 device time，并输出 JSON 与 TensorBoard trace 目录。
- `FACT`：本地 `py_compile` 与 `git diff --check` 通过；未启动服务、未访问 9031/9032。
- `PROBLEM`：当前本地 `ub39-fresh` ControlMaster 虽能返回 `ssh -O check` 的 “Master running”，但最小远端 `echo` 无回显，不能确认 mllv/GPU/9032 状态。
- `DECISION`：在远端命令通道恢复前不发送 profiler 或 benchmark，不修改 9031/9032；恢复后先做只读状态检查，再运行单次 GPU 2 probe。
- `UNKNOWN`：dispatcher CPU 分段、workspace/scatter、Split-KV forward/reduce、mixed kernel 的实际占比尚未取得设备证据。

### 实验 2026-09-28-07：GPU 2 dispatcher / Split-KV profiler

- 环境：ub39 `mllv`，GPU 2；9031 官方服务 PID `2943` 保持运行，9032 未启动。结果目录：`/workspace/logs/dispatcher_profile_20260928/`。
- 形状：`1 x 2048 prefill + 30 x 1 decode`，前缀分别为 `8192` 和 `14336`，`BLOCK_M=64`、`BLOCK_N=64`。所有路径使用同一输入和 paged-KV table；结果均通过正确性门槛，最大误差对 fp32 SDPA 为 `0.00048828125`。
- `FACT`（8k, 4 split）：native full mixed `43.99 ms`，完整 dispatcher `41.92 ms`，raw Split-KV prefill `38.81 ms`。独立 probe 中 dispatcher 相对 native 约 `+4.7%`，不等价于服务收益。
- `FACT`（8k, 8 split）：native `43.99 ms`，dispatcher `42.60 ms`，raw Split-KV `39.75 ms`；相对 4 split，forward 约慢 `3.0%`，reduction 约 `0.36 ms/call`，没有显示增加 split 的收益。
- `FACT`（14k, 4 split）：native `73.52 ms`，完整 dispatcher `63.07 ms`，raw Split-KV prefill `59.23 ms`；独立 probe 中 dispatcher 相对 native 约 `+16.6%`。
- `FACT`：profiler kernel 时间（每次调用折算）显示 8k/4 split 的 Split-KV forward 约 `37.95 ms`，native mixed kernel 约 `46.71 ms`，reduction 约 `0.30 ms`；14k/4 split forward 约 `60.01 ms`，native 约 `78.24 ms`，reduction 约 `0.30 ms`。reduction 没有随长 KV 成为主耗时。
- `FACT`：`index_copy_` CUDA 时间约 `0.09 ms/call`；`_request_partition` CPU 时间约 `0.02–0.03 ms/call`，`_token_indices` 每个子批次约 `0.2 ms`。这些都不是主要瓶颈。
- `FACT`：`_subset_attention_kwargs` 的 CPU wall 计时约 `15.5 ms/子批次`（8k）和 `23.8 ms/子批次`（14k），明显高于 metadata partition；它包含 device `index_select`、新建 `cu_seqlens`/request index、workspace 分配等，当前是主要 dispatcher 软件开销候选。
- `INFERENCE`：Split-KV forward 本体在独立 mixed shape 上比 native mixed kernel 快约 `19%`（8k）和 `23%`（14k），但 dispatcher 重组和 native decode 子调用吞掉部分收益；8 split 只增加开销，没有改善中心时延。
- `UNKNOWN`：`_subset_attention_kwargs` 内部具体是 q/index_select、KV metadata index_select、workspace 分配还是 allocator 同步占主导；需要下一轮逐操作 CUDA event/CPU 计时，不应把 profiler 的累计 `aten::copy_` CPU 时间直接等同于服务墙钟开销。
- `DECISION`：停止继续扫描 split 数；下一步优先做“无子批次复制/无每层 workspace 分配”的 dispatcher 原型测量，再评估是否保留 Split-KV kernel。暂不启动 9032，不做官方负载和 Level 3。
