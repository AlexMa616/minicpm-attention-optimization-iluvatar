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

### 实验 2026-09-28：修复版 paged Split-KV 正确性与首次性能反馈

- `FACT`：修复版代码同步到远端独立目录 `/workspace/split_kv_repair_20260928`，没有覆盖运行中的 `/workspace/vllm-plugin-FL`；9031/9032 均保持 HTTP 200，9033 未运行。
- `FACT`：容器内 `tests/test_paged_split_kv.py` 正确性闸门完成 6/6 组；覆盖随机物理 block table、非均匀 query 长度、非对齐边界以及 4/8 splits。最大绝对误差 `0.015625`，平均误差约 `3e-4`，全部通过 `max<=0.05 && mean<=0.001`。
- `FACT`：首次修复后 mixed 微基准暴露 launch mapping 缺陷：`1x2048 + 30x1, prefix=8192, split=4` 候选 `1110.29 ms`，native `43.95 ms`。原因是按 batch 的 `max_query_len` 发射了 decode 请求不需要的 query blocks；该结果不是硬件性能结论。
- `FACT`：已将 forward/reduce grid 改为按每个请求实际 query block 数压紧映射，逻辑 grid 为 `(request_q_block, query_head, kv_split)`；正确性闸门在该修改后再次 6/6 通过。
- `INFERENCE`：修复后的候选需要重新编译和重测；此前 `2.15–2.43x` 以及首次修复版 `0.04x` 都不能作为最终性能证据。
- `UNKNOWN`：压紧 grid 后 Triton 编译时间、GPU 2 mixed latency、显存工作区占用和服务级收益尚未取得结果；远端微基准当前仍在运行，读取窗口暂时无响应。
- `DECISION`：在 GPU 2 微基准结果收敛前不启动 9032；即使微基准通过，也先做无 profiler 的 9032 短 A/B，再决定官方 benchmark。

### 实验 2026-09-28：修复版性能判定和 RoPE/Warp 专项审计

- `FACT`：上述压紧 grid 的后台结果已落盘：`1x2048 prefill + 30x1 decode`、prefix 8192、split 4，candidate `75.886 ms`，原生 vLLM `44.022 ms`，仅 `0.580x`；fp32 paged-SDPA 最大误差 `0.00048828125`。该结果否定“直接上线修复版 Split-KV”的决定；修复前的 `2.15–2.43x` 不可移用。
- `FACT`：发现独立仓库的旧 harness 在 baseline 模式仍直调 candidate、pure decode 探针也直调 Split-KV；已修正开关与实际服务路由一致，且 `max` 或 `mean` 任一误差超阈即失败。已新增 2048-token long-context 配置；修正后的正式交叉复测仍待执行，旧结果只保留为诊断数据。
- `FACT`：新增独立 Triton RoPE+KV-cache 算子，融合 Q 原位旋转、K 旋转写入 NHD paged cache、V 写入 cache；GPU 2 的 NeoX/interleaved × rotary_dim 64/128 四组对拍均通过，Q/K/V 最大绝对误差为零。首轮第四组误差 `0.03125`，将旋转乘加提升至 fp32 后通过。
- `FACT`：固定 vLLM 0.24.0 的 `RopeKVCacheFusionPass` 仅在 `pass_config.fuse_rope_kvcache=True` 时注册；`vllm/config/compilation.py:283-288` 对非 ROCm 强制关闭该 pass。MiniCPM 的 RoPE 已先于 attention 执行，因此把旋转直接放进当前 attention 会重复旋转。试验性插件 RoPE 接入已撤销，原型只在独立仓库，**未集成服务，未取得性能收益**。
- `DECISION`：BI-V150 的 Warp Specialization/Pingpong/TMA 旧稿不作为完成项。`num_warps`/`num_stages` 只是 launch 参数，未证明设备级生产者/消费者 warp 异步流水化；TMA 不适用于 BI-V150。未来可另做软件流水线或 tile 优化，但需编译产物与 A/B 证据。
- 环境：9031、9032 只读健康检查均为 HTTP 200，9032 队列/KV 为空；未重启、未向 9031 发送候选流量。SSH 控制通道之后又发生间歇性超时，staging harness 更新尚未完成，不能宣称正式性能复测已结束。
- 完整命令、证据和后续判据同步在独立 GitHub 仓库的 `experiments/experiment-log.md`。下一步是 SSH 稳定后完成更正 harness 的 GPU 2 交叉复测；若仍慢于 native，则停止此候选，不消耗 9032 机时。

### 实验 2026-09-28：更正 harness 的交叉复测与 mixed 分流诊断

- 环境：ub39 `mllv`，GPU 2，隔离 staging `/workspace/split_kv_repair_20260928`；更正 harness 已通过 `tar | ssh docker exec -i tar` 放入 `tools/repair_bench_20260928`，SHA-256 与本地相符。9031/9032 未重启、未接收本轮候选请求。
- 运行命令和原始结果：独立仓库 `experiments/experiment-log.md` 的 `2026-09-28-05`；远端 `/workspace/logs/split_kv_repaired_gpu2_20260928/rerun*20260928.jsonl`。四组 `.exit=0`，相同输入 native 与 Split-KV，对 fp32 SDPA 最大误差均 `0.000488`。
- `FACT`：8k/4-split `76.198/43.965 ms`（0.577x），8k/8-split `74.530/44.011 ms`（0.591x），14k/4-split `125.789/73.490 ms`（0.584x），14k/8-split `119.756/73.464 ms`（0.613x）。修复版整批 Split-KV 全面慢于 native，否定先前旧 ABI 的加速数据；不进入 9032。
- `HYPOTHESIS`：`max_seqlen_q>1` 时整批 30 个 decode 行也被送入 `BLOCK_M=64` Split-KV，造成无效计算。只在隔离 staging 和本地插件工作树放置 mixed 分流原型，保留原生 decode 3D 路径，并添加独立 `dispatcher_mixed_probe.py`。
- `FACT`：GPU 2 的 8k/4-split dispatcher 探针 `43.580 ms` 对 native `44.000 ms`（1.010x），对 native max abs `0.001953`，对 fp32 SDPA max abs `0.000488`，5 repeats。说明分流消除上述明显回归，但 0.96% 差异没有足够统计或服务证据。
- `RISK`：该诊断原型用 `.cpu().tolist()` 每层同步设备 metadata，再重建张量并 scatter，尚未证明 CUDA graph 和服务调度兼容，不能进 9032 或提交为正式优化。后续 14k/8-split dispatcher 补测因 SSH 命令连续超时无产物，已停止重试；不推断运行成功。插件工作树有未提交的实验修改，已推送 GitHub 的集成补丁不包含该分流原型。
- `DECISION`：先找到无需 GPU→CPU 同步的调度层分组接口，或返回优化 2D prefill tile。完整数值/图捕获/服务链路未验收前不声称性能收益，9031 始终不动。

#### 本地后续修正（待设备验证）

- `FACT`：插件 `model_runner.py` 已向 `CommonAttentionMetadata` 提供 `query_start_loc_cpu`。本地 vendor 分流原型现从这份已有 CPU 元数据获取请求长度；缺失时回退原生 attention，不再对 GPU `cu_seqlens_q` 做每层 `.cpu().tolist()`。
- `FACT`：已修正原型的 decode 3D segment scratch 索引：真实服务 scratch 只按请求数预分配约 64 行，不能按 2048-token prefill 的全局 token 序号索引。诊断探针改为真实 64 行布局，并可把 prefill 放在 `--prefill-index` 指定的任意请求位置，数值误差超标时退出失败。
- `UNKNOWN`：新版本只过本地语法/差异检查，尚未重跑 GPU 2；原 `1.010x` 探针不能算作新版本收益。SSH 控制连接连续超时，未启动后续 14k/8-split dispatcher 实验。当前不更新 9032，不把本地原型加入 GitHub 的正式集成补丁。

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

### 诊断准备 2026-09-28：定位 dispatcher、workspace 与 mixed kernel 瓶颈

- `FACT`：独立仓库新增 `tests/dispatcher_profile_probe.py`，设计为同一输入上的三路对照：native full mixed、raw Split-KV prefill、完整 dispatcher 分流路径。
- `FACT`：诊断计时覆盖 `_request_partition`、`_token_indices`、`_subset_attention_kwargs`、optimized/native attention launch，以及 CUDA profiler 中的 Split-KV forward/reduce、native unified attention、`index_copy_`/拷贝事件。
- `FACT`：probe 通过已有 CPU metadata `query_start_loc_cpu` 分组，避免把 GPU metadata 同步误判为 dispatcher 固有成本；不启动服务、不向 9031/9032 发请求。
- `FACT`：本地 Python 编译和 diff 检查通过。
- `PROBLEM`：本地 `ub39-fresh` ControlMaster 仅能通过 `ssh -O check` 显示 master 存活，最小远端 `echo` 无回显；本轮没有获得远端容器、GPU 或端口状态。
- `DECISION`：等待可用 SSH 命令通道后，第一步只读核查 mllv、9031、9032 和 GPU；确认 GPU 2 空闲后才执行 profiler probe。9031 保持不动。
- `UNKNOWN`：当前没有 profiler 事实支持 dispatcher、scatter、reduction 或 mixed kernel 中任一项为主瓶颈。

### 实验 2026-09-28-07：GPU 2 dispatcher / Split-KV profiler

- 环境：ub39 `mllv`，GPU 2；9031 PID `2943` 保持运行，9032 未启动。远端结果位于 `/workspace/logs/dispatcher_profile_20260928/`。
- 形状：`1 x 2048 prefill + 30 x 1 decode`，前缀 `8192/14336`，`BLOCK_M=64`、`BLOCK_N=64`；所有路径复用同一输入和 paged-KV table，正确性最大误差对 fp32 SDPA 为 `0.00048828125`。
- `FACT`：8k/4 split 的 native、dispatcher、raw Split-KV 分别为 `43.99/41.92/38.81 ms`；8k/8 split 为 `43.99/42.60/39.75 ms`；14k/4 split 为 `73.52/63.07/59.23 ms`。
- `FACT`：Split-KV forward 折算约 `37.95 ms/call`（8k）和 `60.01 ms/call`（14k）；native mixed kernel 约 `46.71 ms/call` 和 `78.24 ms/call`。独立 kernel 侧存在约 `19%/23%` 的收益，但这不是服务吞吐结论。
- `FACT`：Split-KV reduction 约 `0.30 ms/call`，`index_copy_` 约 `0.09 ms/call`，`_request_partition` 约 `0.02–0.03 ms/call`；三者均不是主要瓶颈。
- `FACT`：`_subset_attention_kwargs` CPU wall 约 `15.5 ms/子批次`（8k）和 `23.8 ms/子批次`（14k），明显重于 request partition，包含子 Q/metadata index_select、cu-seqlens 构造和 workspace 分配。
- `INFERENCE`：当前端到端独立 probe 的剩余损失主要来自 dispatcher 子批次重组；继续增加 split 数不能解决，8 split 相对 4 split 反而略慢。
- `UNKNOWN`：subset helper 内部各个 device copy、workspace allocation 和 allocator 同步的精确占比尚未拆分。
- `DECISION`：停止 split 数扫描，下一步只做无复制/复用 workspace 的 dispatcher 诊断原型；不启动 9032，不跑官方负载和 Level 3。

### 实验 2026-09-28-08：dispatcher subset 原子操作归因

- `FACT`：GPU 2 同步测量显示 prefill/decode 两个 `full_subset_helper` 约 `0.25–0.33 ms/子批次`；Q、KV metadata index_select、workspace 分配和 selected cu-seqlens 构造均为亚毫秒级。
- `CORRECTION`：之前 wrapper 中的 `15–24 ms/子批次` 被前序异步 GPU 工作/allocator 等待污染，不代表 helper 固有开销，原“subset helper 是主要软件瓶颈”的判断撤回。
- `FACT`：`index_copy_` 约 `0.09 ms/call`，Split-KV reduction 约 `0.30 ms/call`；均不能解释主要时延。
- `INFERENCE`：当前剩余差额主要是 native decode 子调用、分流 launch/同步和少量 scatter；主瓶颈仍是 mixed prefill 的 Split-KV forward kernel。
- `DECISION`：停止 dispatcher 重写和 split 数扫描，转向 BI-V150 kernel 级带宽、占用和 tile 诊断；9031 不动，9032 暂不启动。

### 实验 2026-09-28-09：BLOCK_M 定点扫描与 9032 隔离服务短测

- 环境：ub39 `mllv`，GPU 2 微基准；服务 A/B 仅使用 GPU 1/端口 9032。9031 官方服务 PID `2943` 始终运行，健康检查保持 HTTP 200，未重启、未发送请求。
- 微基准形状：`1 x 2048 prefill + 30 x 1 decode`，prefix `8192/14336`，`BLOCK_N=64`、4 split；结果通过 fp32 paged-SDPA 对拍，最大误差 `0.00048828125`，对 native 最大误差 `0.001953125`（14k 为 `0.0029296875`）。
- `FACT`：prefix 8192、4 split 的 raw Split-KV CUDA event 时间：`BLOCK_M=32: 49.87 ms`，`BLOCK_M=64: 38.57 ms`，`BLOCK_M=128: 24.05 ms`；同批 native mixed 约 `44.01 ms`。`BLOCK_M=128` 比 `BLOCK_M=64` 的 raw forward 约快 `37.6%`。
- `FACT`：prefix 14336、4 split、`BLOCK_M=128` 的 raw Split-KV `36.27 ms`，native mixed `73.38 ms`，完整 dispatcher `40.26 ms`。
- `FACT`：9032 candidate 使用 `ILUVATAR_USE_OPTIMIZED=1`、`ILUVATAR_SPLIT_KV=1`、4 split、`BLOCK_M=128`、`BLOCK_N=64`；日志确认 `vendor.iluvatar` 路由，CUDA graph capture 完成，健康检查 HTTP 200。
- `FACT`：candidate 在 `16384 input / 256 output / c16 / 16 prompts` 四轮短测中，首轮总吞吐 `958.50 tok/s`；跳过首轮后的三轮汇总 `10569.22 tok/s`，平均 TTFT `4722.93 ms`。命令未显式关闭 prefix caching，日志显示后续轮次命中率最高约 `74.9%`。
- `FACT`：同条件 baseline 三个 steady run 为 `9817.69/11529.50/10901.00 tok/s`，均值 `10749.40`、中位数 `10901.00`、CV `6.58%`；candidate steady 均值 `10569.22`、中位数 `10256.21`、CV `6.43%`，相对均值 `-1.68%`、相对中位数 `-5.91%`。
- `DECISION`：当前 `BLOCK_M=128 + 4-split dispatcher` 暂不进入正式提交；GPU 2 的 kernel 级收益尚未转化为 9032 c16 端到端收益。下一步如继续，必须先显式关闭 prefix caching并增加 candidate prefill 分流的运行时计数/trace，再决定是否扩大负载。

### 实验 2026-09-28-10：修复后 candidate 与 baseline 同负载复测（16k/c8/64）

- 环境：ub39 `mllv`，Iluvatar BI-V150 GPU 1，端口 9032；9031 官方服务未停止、未重启、未发送请求。candidate 与 baseline 均使用 `16384 input / 64 output / c8 / 8 prompts`、prefix caching 关闭、相同 vLLM benchmark 命令；两边均跳过 benchmark 的首轮预热，比较后 3 轮。
- candidate 配置：`ILUVATAR_USE_OPTIMIZED=1`、`ILUVATAR_SPLIT_KV=1`、`ILUVATAR_NUM_SPLITS=4`、`BLOCK_M=64`、`BLOCK_N=64`；服务日志确认 `optimized_prefill` 与 `mixed_partitioned` 路由持续出现，`native_fallback=0`，因此本轮不是 fallback 假阳性。
- baseline 配置：`ILUVATAR_USE_OPTIMIZED=0`、`ILUVATAR_SPLIT_KV=0`；服务日志确认 `attention_backend` 使用 `default.flagos`。
- 原始日志：candidate `/workspace/logs/routefix_candidate_16k_c8_o64_v3.log`；baseline `/workspace/logs/routefix_baseline_16k_c8_o64_v1.log`。两组请求均 8/8 成功。

| 路径 | steady 总吞吐 tok/s（3 轮） | 均值 | 中位数 | Mean TTFT 均值 | Mean ITL 均值 |
|---|---:|---:|---:|---:|---:|
| candidate 4-split | 1065.02 / 1068.39 / 1075.43 | 1069.61 | 1068.39 | 67.60 s | 865.70 ms |
| baseline native | 1086.93 / 1086.80 / 1086.96 | 1086.90 | 1086.93 | 67.07 s | 842.90 ms |

- `FACT`：candidate 相对 baseline 的 steady 总吞吐均值为 `-1.59%`，中位数为 `-1.71%`；没有达到继续扩展 A/B 的 `+5%` 门槛。
- `FACT`：candidate Mean TTFT 均值约高 `0.8%`；Mean ITL 均值约高 `2.7%`。Median ITL 约 `48.15 ms` 对 `48.32 ms`，差异很小，但 P99 ITL/TTFT 没有形成改善。
- `FACT`：candidate 本轮路由已实际执行 Split-KV 分流，故“服务没有调用优化 kernel”的旧问题已排除；但 kernel 级收益尚未转化为端到端收益。
- `INFERENCE`：在真实服务的 `c8` 低并发形状下，dispatcher/分流和 mixed kernel 的综合开销抵消了 raw Split-KV 的局部收益；当前 4-split 不能作为默认策略。
- `DECISION`：停止 4-split 的多轮交叉 A/B 和官方负载扩展；只保留一次同负载 `8-split` 定点复测作为最后的低成本对照。若 8-split 仍不超过 baseline，停止当前 Split-KV 服务主线，转回 profiler/调度或撤回该候选，不再声称有端到端性能提升。

### 实验 2026-09-28-11：修复后 8-split candidate 最终定点复测（16k/c8/64；远端日志时间为 2026-09-29）

- 环境：ub39 `mllv`，Iluvatar BI-V150 GPU 1，端口 9032；9031 官方服务保持不动。candidate 与前一轮 baseline 使用同一负载：`16384 input / 64 output / c8 / 8 prompts`，prefix caching 关闭；benchmark 四轮，首轮作为启动/编译预热，比较后 3 轮。
- candidate 配置：`ILUVATAR_USE_OPTIMIZED=1`、`ILUVATAR_SPLIT_KV=1`、`ILUVATAR_NUM_SPLITS=8`、`BLOCK_M=64`、`BLOCK_N=64`。原始日志：`/workspace/logs/routefix_candidate_8split_16k_c8_o64_v1.log`。四轮均 `8/8` 成功。

| 路径 | steady 总吞吐 tok/s（3 轮） | 均值 | 中位数 | CV |
|---|---:|---:|---:|---:|
| candidate 8-split | 748.51 / 755.81 / 756.50 | 753.61 | 755.81 | 0.59% |
| native baseline | 1086.93 / 1086.80 / 1086.96 | 1086.90 | 1086.93 | 0.008% |

- `FACT`：8-split candidate steady 均值相对 native baseline 为 `-30.66%`，中位数为 `-30.46%`；稳定 CV 很低，排除“只是运行波动”的解释。
- `FACT`：8-split 后三轮 Mean TTFT 为 `97.40/96.66/96.52 s`，Mean TPOT 为 `1230.19/1215.05/1214.80 ms`，Median ITL 为 `48.15/48.21/48.17 ms`；吞吐回归主要来自 prefill/TTFT，而非 decode median ITL 的单独恶化。
- `INFERENCE`：增加 split 数没有解决真实服务的主要开销；多路 reduction、分流/输出合并与 prefill kernel 综合开销显著超过 raw kernel 局部收益。4-split 的 `-1.59%` 与 8-split 的 `-30.66%` 共同否定当前 Split-KV dispatcher 作为服务优化主线。
- `DECISION`：停止 4/8-split 的进一步 A/B、官方完整负载扩展和 Level 3 复测；当前候选不作为默认配置、不进入正式提交。后续若继续优化，应回到 profiler 结果，分别量化 dispatcher、scatter/index-copy、reduction 和 mixed-prefill kernel，不再以增加 split 数作为默认方向。
- `RESTORE`：8-split 服务已停止；9032 已在 `mllv` 内恢复 native baseline，进程 PID `171968`，健康检查 HTTP `200`。9031 全程未停止、未重启、未发送请求。

### 分析纠正 2026-09-28：c16/c64 差异的归因边界

- `FACT`：当前记录中 c16 的 8-split 交叉 A/B（`16384 input / 256 output / c16 / 16 prompts`）总体中位数仅比 baseline 高 `0.06%`，并未证明稳定收益；修复后 `c8` 的 4-split 与 8-split 分别为 `-1.59%` 和 `-30.66%`。因此不能把“c16 崩溃、c64 正常”作为已确认事实，必须先补齐同版本、同 GPU、同 prefix-cache 设置的 c64 对照。
- `FACT`：GPU 2 组件 profiler 已测得 `_request_partition` `0.02–0.03 ms`、`index_copy_` `约 0.09 ms`、Split-KV reduction `约 0.30 ms`；这些数量级远小于假设中的 `5–10 ms`、`2–5 ms`、`8–15 ms`，不能解释 30% 服务回归。
- `FACT`：同一 probe 中，8k/4-split 的 native/dispatcher/raw Split-KV 为 `43.99/41.92/38.81 ms`，8k/8-split 为 `43.99/42.60/39.75 ms`；14k/4-split 为 `73.52/63.07/59.23 ms`。raw kernel 有局部收益，但完整路径没有随 split 数增加而改善。
- `CORRECTION`：`total = prefill_compute / num_splits + 固定 dispatcher + scatter + reduction` 只能作为定性模型，不能用当前数字反推出 c16 的 32.4% 回归；真实路径还包含 native decode 子调用、子批次张量重建、launch/synchronization 和 mixed attention 的工作量变化，且这些项并不都按并发线性缩放。
- `HYPOTHESIS`：低并发下 candidate 的固定路径成本占比更高是合理假设，但目前更强的候选解释是 mixed-prefill/子批次 attention 的执行与同步成本，而非 Python partition、scatter 或 reduction 单项。
- `DECISION`：不做 C++ dispatcher 重写、不做 Fused Kernel 设计、不按“c16=4 split、c64=8 split”直接实现自适应。先完成最小矩阵：`c16/c64 × native/4-split/8-split`，每格同一服务参数、关闭 prefix cache、首轮预热后至少 3 个稳态轮；同时记录 Mean/Median TTFT、Mean/Median ITL、总吞吐和 candidate route counts。

### 实验 2026-09-29-02：c16/c64 最小服务验证矩阵（运行中）

- 环境：ub39 `mllv`，诊断端口 9032/GPU 1；9031 官方服务 PID `2943` 保持运行，健康检查未改变。
- 编排脚本：`vllm-plugin-FL/tools/cross_ab_matrix.py`，仅同步到远端 `/workspace/split_kv_repair_20260928/tools/`；每组服务使用 `--no-enable-prefix-caching`，benchmark 固定 `16384 input / 1024 output / 128 prompts`，并发分别为 c16/c64；每组 4 轮，跳过首轮。
- 计划补测：`c16_native`、`c16_4split`、`c64_native`、`c64_4split`、`c64_8split`。candidate 固定 `BLOCK_M=128`、`BLOCK_N=64`、4 warps、2 stages，以对齐已有 c16 交叉 A/B 的参数。
- `FACT`：脚本已启动，第一组 `c16_native` 服务 PID `172556` 于远端时间 `2026-09-29 08:13:25` 健康，benchmark 已进入第 1/4 轮；当前无结果可判定。
- `BOUNDARY`：该矩阵只补充并发度相关证据，不改变 9031，不把首轮启动/编译数据计入稳态比较；若某组服务启动失败，记录失败并停止后续无效重复。

#### 已完成的 c16 结果

| 配置 | 稳态总吞吐 tok/s（3 轮） | 均值 | 中位数 | Mean TTFT 均值 | Median ITL 均值 |
|---|---:|---:|---:|---:|---:|
| c16 native | 866.67 / 869.83 / 869.54 | 868.68 | 869.54 | 36.96 s | 81.56 ms |
| c16 4-split | 820.34 / 821.58 / 810.00 | 817.31 | 820.34 | 37.85 s | 81.58 ms |

- `FACT`：c16 4-split 相对 native 的均值/中位数吞吐分别为 `-5.91%/-5.66%`；两组请求均 `128/128` 成功。
- `FACT`：4-split 的 Median ITL 与 native 基本相同，但 Mean TTFT 约高 `2.4%`；当前回归主要表现为 prefill/总吞吐损失，不是 decode median ITL 恶化。
- `DECISION`：c16 不启用 4-split；该结果与已知 c8 4/8-split 回归一致，继续等待 c64 三组结果验证高并发是否存在收益。

#### 矩阵收缩决定

- `FACT`：在 c64 native 仅完成第 1 轮、尚未完成稳态结果时，因 c8/c16 已显示无收益或回归，且 c64 4/8-split 需要继续占用数小时，主动停止剩余矩阵；该动作不把未完成的 c64 数据当作结论。
- `FACT`：停止时已终止矩阵编排、benchmark 子进程并重新启动 9032 native baseline；最终检查 9031/9032 均 HTTP `200`，9031 PID `2943` 未改变。
- `DECISION`：Split-KV 服务路径冻结，不再投入长时间 c64 A/B、自动 split 或 reduction 扫描。已有证据足以判断当前实现没有达到可提交的收益门槛；此前“c64 约 +1%”即使复现，也低于 `+5%` 进入条件。
- `NEXT`：转向保持 native request batching/dispatch 的原生 2D attention 优化。优先验证 Iluvatar vendor 路径中的 tile/launch 参数是否能在真实 mixed 形状改善 TTFT；不复用当前 Python 子批次分流。`max_num_batched_tokens`/禁用 chunked prefill 只作为诊断实验，若改变 serve 参数则不直接视为合规提交方案。

### 实验 2026-09-29-03：Native 2D attention 参数化与 GPU 2 首轮扫描

- `FACT`：9031 PID `2943` 与 9032 PID `179069` 均保持运行且健康检查为 HTTP `200`；本轮只在容器 GPU 2 做独立 kernel harness，不向服务发送请求。
- `FACT`：远端实际安装的 vLLM unified attention 源码已核对。MiniCPM5-2B 配置为 `16 Q heads / 2 KV heads / head_dim 128 / block_size 16`；原生 2D 路径默认 `BLOCK_M=16`、`BLOCK_Q=2`、prefill tile `32`，未针对 Iluvatar 暴露可调 launch 参数。
- `CHANGE`：从远端安装版本复制 2D unified attention 到 `vllm_fl/dispatch/backends/vendor/iluvatar/impl/ops/triton_unified_attention_native.py`，仅增加 opt-in 的 `BLOCK_M`、prefill tile、warps、stages 覆盖；decode 3D、request batching、KV ABI 保持原路径。
- `CHANGE`：`vllm_fl/dispatch/backends/vendor/iluvatar/impl/attention.py` 增加 `ILUVATAR_NATIVE_2D_TUNE=1` 分支；默认值仍为关闭，未启用 Split-KV 或 Python mixed 分流。
- `CHANGE`：新增 `vllm-plugin-FL/tools/native_2d_harness.py`，同一 paged-KV 输入对比已安装 vLLM native kernel 与 vendor native 2D kernel，并用 fp32 SDPA 做数值闸门。
- `FACT`：smoke 形状为 `1 x 2048 prefill + 30 x 1 decode`、prefill prefix `8192`；配置 `BLOCK_M=32/TILE=32/warps=4/stages=2`。native `46.69 ms`，tuned `20.46 ms`，raw speedup `2.28x`；tuned 对 fp32 SDPA 最大绝对误差 `0.00342`，通过当前误差门槛。
- `FACT`：mixed 首轮扫描 `36` 个组合已完成。最佳已观测组合为 `BLOCK_M=128/TILE=64/warps=8/stages=2`：native `46.65 ms`，tuned `10.12 ms`，raw speedup `4.61x`，有效算力约 `8.69 TFLOP/s`，最大绝对误差 `0.00342`。
- `BOUNDARY`：上述是独立 kernel 结果，不等价于服务吞吐收益；当前候选仍未进入 9032。需要先完成 pure-prefill 交叉筛选，再以 9032 native batching 做短 A/B，确认 TTFT、吞吐、Median/P99 ITL 是否共同改善。
- `NEXT`：保留 mixed 扫描日志 `/workspace/logs/native_2d_20260929/mixed_scan.json`；继续读取 pure-prefill 扫描，随后只选择 mixed 与 pure 都不回退的前 1-2 组进入 9032。若服务收益低于 `+5%` 或 decode 长尾恶化，则冻结该参数化路径并转向调度诊断。

### 实验 2026-09-29-04：Native 2D 服务调用链修复与 candidate10 路由确认

- `FACT`：9031 官方服务保持运行，未停止、未重启、未发送请求；9032 始终作为 GPU 1 隔离服务。
- `PROBLEM`：远端 `vendor/iluvatar/iluvatar.py:IluvatarBackend.attention_backend()` 仍返回 vLLM 原生 `TRITON_ATTN` 类路径，导致已编写的 `impl/attention.py` 不在服务调用链中。之前看到的 native kernel cache 不能单独证明服务调用了该 kernel。
- `CHANGE`：同步本地已提交实现，使 Iluvatar backend 对非 MLA attention 返回 `vllm_fl.dispatch.backends.vendor.iluvatar.impl.attention.IluvatarAttentionBackend`；MLA 分支保持原路径。同步 native 2D attention 文件和可复用的 `tools/start_native_2d_service.sh`。
- `FACT`：candidate10 使用 `FULL_DECODE_ONLY`，启动日志显示 `Op 'attention_backend' using 'vendor.iluvatar'`；服务健康后，真实 9032 请求日志出现 `Iluvatar attention dispatch probe` 和 `Iluvatar native 2D attention enabled ... max_query_len=2048`，证明 vendor backend 和 native 2D kernel 已进入真实服务调用路径。
- `FACT`：candidate8 曾因默认 `FULL_AND_PIECEWISE` mixed CUDA graph capture 在 GPU 1 OOM，未作为性能结果；candidate10 改回 `FULL_DECODE_ONLY` 后完成 35 个 decode graph capture，9032/9031 均 HTTP `200`。
- `BOUNDARY`：candidate10 的真实服务短 benchmark 在 SSH 控制通道失效前已启动，但本轮尚未读取到吞吐结果；不能把 kernel harness 的 `4.61x` 当作服务收益。
- `NEXT`：恢复 SSH 后只读检查 candidate10 benchmark 是否仍在运行及日志；若已完成，提取 candidate 吞吐/TTFT/ITL，再停止 9032、移除 native marker、用同一 `FULL_DECODE_ONLY` 参数启动 native baseline，完成同负载对照。9031 继续保持隔离。

### 实验 2026-09-29-05：Native 2D candidate 真实服务反向复验（c16，4k/16k）

- `FACT`：9032 使用 GPU 1、`FULL_DECODE_ONLY`、关闭 prefix caching；baseline 与 candidate 使用相同模型、端口、服务参数和 `benchmark_throughput_serve.py` 负载：`c16`、16 prompts、`128` output tokens，分别测试 `4096` 和 `16384` input tokens。9031 官方服务 PID `2943` 全程未停止、未重启、未发送请求。
- `FACT`：candidate 启动日志明确出现 `Iluvatar native 2D attention enabled: BLOCK_M=128 TILE=16 warps=8 stages=2 max_query_len=2048`，证明本轮不是独立 harness 假收益，而是实际进入 vendor native 2D attention 调用路径。

| 输入 | 配置 | 稳态总吞吐 tok/s（跳过首轮） | Mean TTFT | Median ITL | P99 ITL |
|---|---|---:|---:|---:|---:|
| 4k | native baseline | 3027.03 | 10292.52 ms | 32.43 ms | 约 621--689 ms |
| 4k | native 2D candidate | 5765.28 | 4309.11 ms | 32.41 ms | 约 307--329 ms |
| 16k | native baseline | 1071.42 | 127004.78 ms | 81.31 ms | 约 3029--3290 ms |
| 16k | native 2D candidate | 3334.08 | 37596.41 ms | 81.11 ms | 约 824--933 ms |

- `FACT`：4k candidate 相对 baseline 为 `1.9046x`、`+90.46%` 吞吐；Mean TTFT 降低 `58.13%`。16k candidate 为 `3.1118x`、`+211.18%` 吞吐；Mean TTFT 降低 `70.40%`。
- `FACT`：candidate 的 Median ITL 与 baseline 基本持平（4k 约 `32.4 ms`，16k 约 `81.1--81.3 ms`），没有观察到 decode median 回退；P99 ITL 在两种输入长度均下降，但仍需更高并发和多轮交叉顺序复验。
- `BOUNDARY`：本轮是 `c16 / 16 prompts / 128 output` 的反向复验，不是官方完整 `c64/128` 或 `c64/256` 负载，也不是 Level 3 准确率验收；不能据此直接宣称比赛最终收益。
- `PROBLEM`：先前自动编排脚本把已完成的 baseline 僵尸 PID 误判为存活，且 `ps|awk` 查询会自匹配，导致 candidate 未自动启动；本轮改为直接控制已释放的 9032 服务并完成同口径测试。该编排问题不影响本次 benchmark 数据。
- `FACT`：测试完成后已停止 9032 candidate、删除 `/tmp/iluvatar_native_2d.enable`；9032 当前无服务，9031 健康检查为 HTTP `200`。下一轮启动 9032 baseline/candidate 前必须再次显式确认服务状态。
- `DECISION`：Native 2D attention 从“只在 harness 中快”升级为“真实服务 c16 负载下有强收益”的有效候选，保留并进入下一阶段；Split-KV 仍冻结，不重新打开。
- `NEXT`：在 9032 以同一 candidate 参数先完成 `c64` 4k/16k 短测，再做 baseline→candidate、candidate→baseline 的交叉复验；通过后再跑官方完整负载、准确率和更长尾统计。服务级验收仍要求吞吐提升、TTFT 不恶化、4k/16k Median ITL 不回退、Level 3 至少达到基线 `102/105`。

原始日志：远端 `/workspace/logs/native_2d_followup_20260929/baseline_c16_4k_16k.log`、`candidate_reverse_4k_16k.log`、`candidate_reverse_service.log`。

### 实验 2026-09-29-06：Native 2D c64 交叉对照与 staged-loop 原型

- `FACT`：9032 candidate 使用 GPU 1、`FULL_DECODE_ONLY`、关闭 prefix caching，负载为 `[[4096,128,64,64]]` 和 `[[16384,128,64,64]]`，每组四轮、丢弃启动后的首轮。9031 服务 PID `2943` 保持运行，未对其发请求。
- `FACT`：candidate c64/4k 三轮稳态总吞吐 `6257.51 / 6252.65 / 6256.42 tok/s`，汇总 `6255.53 tok/s`，Mean TTFT `17556.21 ms`；c64/16k 为 `3322.99 / 3323.49 / 3322.77 tok/s`，汇总 `3323.08 tok/s`，Mean TTFT `155851.39 ms`。
- `FACT`：candidate c64/16k 三轮 Median ITL 为 `564.18 / 559.23 / 562.80 ms`，P99 ITL 为 `1103.64 / 1064.50 / 1029.65 ms`。完整日志：`/workspace/logs/native_2d_followup_20260929/candidate_c64_4k_16k.log`。c64/4k 的首轮仅 `1736.27 tok/s`，故不能把首轮并入稳态平均。
- `FACT`：候选完成后停止 9032 candidate 并删除 `/tmp/iluvatar_native_2d.enable`；随后同一模型、端口、`FULL_DECODE_ONLY` 和关闭 prefix caching 的 9032 baseline 服务启动，日志显示 `native_env=0 split_env=0`。baseline c64/4k、c64/16k 正在执行；未完成前不计算 c64 相对收益。baseline 日志：`/workspace/logs/native_2d_followup_20260929/baseline_c64_4k_16k.log`。
- `FACT`：当前 Iluvatar native 2D kernel 已在一个 program 中使用 `kv_head_idx` 分组、`BLOCK_M/BLOCK_Q` 映射 8 个 Q heads，并针对每个 KV tile 只载入一份 K/V，且调用现有 `softmax_step` 在线归约。因此“GQA grouped”及 FP32 online softmax **不能**当作下一轮新增算法贡献。
- `CHANGE`：本地 `vllm-plugin-FL` 增加实验性 `tl.range(..., num_stages=PIPELINE_STAGES)` 循环深度，独立于 launch `num_stages`，并让 `tools/native_2d_harness.py` 扫描 `pipeline_stages=1/2/3`；已通过本地 Python 语法检查和 `git diff --check`。相关文件已传至远端工作树，但当前 baseline 服务在此之前启动且关闭了候选分支，**没有**将新循环用于 baseline 数据。
- `HYPOTHESIS`：显式 staged-loop 可改善 paged-KV block-table、K/V load 与 dot 的重叠；`UNKNOWN`：后端是否生成有效流水线、是否加速、以及数值表现。`tl.range` 编译器提示本身不证明异步 prefetch。原型必须先在 GPU 2 与 `pipeline_stages=1` 同形状对照，并过 fp32 SDPA 数值闸门，再考虑 9032 服务级 A/B。
- `NEXT`：完成 c64 baseline、记录准确差值与健康状态；用冻结的参数路径在 9032 做 Level 3；GPU 2 的 staged-loop 编译/消融独立于服务验收。官方完整负载及反序复验仍待执行。

#### GPU 2 prototype 消融与数值异常（2026-09-29 22:16 更新）

- `FACT`：GPU 2 使用同一 paged-KV/BF16/GQA=8 mixed 形状（`1 x 2048 prefill + 30 x 1 decode`）、固定 `BLOCK_M=128/TILE=16/warps=8/launch stages=2`，扫描 `(pipeline_stages, scalar_block_lookup)=(1,0),(1,1),(2,0),(2,1)`。四格对 fp32 参考的最大绝对误差均 `0.000488`，Triton 编译和执行均成功，脚本退出码 `0`。
- `FACT`：mixed 四格时延分别为 `13.714 / 13.788 / 13.983 / 13.762 ms`。同样的 pure prefill（8 个 2048 chunk，前缀 0/2/4/6/8/10/12/14k）四格分别为 `59.516 / 60.036 / 61.275 / 59.593 ms`，对 fp32 参考最大误差均 `0.015625`。新增流水线提示或标量索引均没有可分辨的加速；这两项仅保留实验代码，不进入服务 candidate 或创新收益声明。
- `BOUNDARY`：这些 GPU 2 探针与 9032 baseline c64/16k 同时运行，且只计时 5 次；不把微秒级差异视为稳定服务收益。该结论足以阻止当前配置直接升级到服务 A/B，不排除其他形状/更长预热出现差异。
- `FACT`：pure 形状下已安装 vLLM 原生 kernel 相对同一 fp32 SDPA 参考的最大绝对误差 `1.640625`，`39,752` 个输出元素误差 `>0.05`；最差索引 `[3,9,16]` 的原生值 `1.796875`，fp32 参考和 tuned 值均为 `3.4375`。tuned 路径最大绝对误差 `0.015625`，误差 `>0.05` 的元素数为 `0`。mixed 形状下原生最大误差仅 `0.003418`。
- `HYPOTHESIS`：原生 2D kernel 在 early-chunk、特定 TILE/BLOCK_Q 组合存在 mask/页表或归约边界问题；`UNKNOWN`：是 kernel 错误、harness 参考假设还是某个形状独有的路由。须构建最小 q-length/prefix 案例和独立参考交叉检查后再归因。
- 原始日志：`/workspace/logs/native_2d_followup_20260929/staged_scalar_smoke.log`、`staged_scalar_pure_smoke.log`、`native_2d_pure_error_audit.log`；三次命令完成时 `.exit=0`。

#### Phase 1 状态与新算法方向审查（2026-09-29 22:20）

- `FACT`：c64/4k baseline（9032、关闭 prefix cache、64 prompts/128 output）丢弃首轮后三轮吞吐 `3228.47 / 3227.88 / 3227.49 tok/s`，汇总 `3227.95 tok/s`；相同条件 candidate `6255.53 tok/s`，相对提升约 `93.8%`。baseline Mean TTFT 三轮约 `38.75 s`，candidate `17.56 s`。c64/16k baseline 已完成首轮 `1057.92 tok/s`，第 2/4 轮在跑；不以首轮推断稳态收益。9031 `/health` 为 HTTP 200，PID `2943` 未动。
- `FACT`：当前 2D kernel 的 grid 已是 `(query_tiles, num_kv_heads)`；`offs_m // num_queries_per_kv` 和 `% num_queries_per_kv` 将 GQA=8 的 Q 行合并在单个 program；每个循环只加载一次该 KV head 的 K/V tile，`acc` 和 `M/L` 用 fp32 在线累计。因此提议的“GQA grouped”与“+FP32 softmax”消融格与现有实现重合，不能作为新增贡献或做无意义的重实现。
- `FACT`：launch `num_stages=2` 不等同显式流水线。原型新增的循环 `tl.range(num_stages=2)` 和 scalar page-table lookup 在已测 mixed/pure 形状均无可区分的加速；不声称异步 prefetch 或新增服务收益。微基准曾与 c64 baseline 同时进行，需避免把其绝对时延当干净的隔离结果。
- `CHANGE`：harness 的 early-prefill 形状添加独立 `torch.nn.functional.scaled_dot_product_attention` fp32 对拍，防止将自写 einsum reference 与真实 kernel 的共同假设误判为正确性证据。待 c64 baseline 完成后运行 GPU 2 audit。
- `DECISION`：Phase 1 先以已提交的 `cdc5fea` 参数候选完成 c64/16k baseline 与 Level 3（既有 baseline 为两次 `102/105`）；准确率和独立数值问题未查明之前，不冻结为已验收 baseline-v2。Phase 2 不照伪代码再实现已有 GQA/softmax；若要引入新算法，须先有 profiler 指向的重复访存、浪费计算或 decode 拖尾，并建立与参数候选一致的消融对照。
- `UNKNOWN`：c64/16k 稳态对照、Level 3、pure-prefill 原生数值异常成因、官方完整负载及逆序 A/B；这些数据缺失时，`+90-211%` 只适用于已经测试的 c16 短负载。
- `CHANGE`：新增**默认关闭的原型** `mixed_dual_launch_override`，保持 paged-KV ABI 与同一输出缓冲区。首个 kernel 用既有 `BLOCK_M=128` 处理 prefill，跳过 `q_len=1`；第二个 kernel 按请求索引直接映射 decoder，用 `BLOCK_M=16` 处理 `q_len=1`，跳过 prefill；无需 Python 子批次、scatter 或 split-KV reduction。harness 在同一输入上并列测试既有 tuned 与 dual 路径，并添加 decode-first/interleaved 次序测试。相关代码完成 Python 语法、shell 语法与 diff 静态检查，已同步至远端，但**尚未在 GPU 上编译/验证，不作为有效优化结果**。
- `TEST`（待执行）：在 baseline c64 完成、避免并行干扰后运行 `bash /workspace/vllm-plugin-FL/tools/run_native_2d_smoke.sh /workspace/logs/native_2d_followup_20260929 early audit`、`... mixed dual` 和 `... mixed-order dual`；关注 fp32 参考误差、dual/tuned 稳态比值。只有 dual 数值正确且同形状重复稳定提升，才研究服务路由；如果两次 launch 不划算则撤回，不制造“算法创新”收益叙事。
- `ACTION`：后台进程 PID `244100` 运行 `tools/continue_native_2d_audits.sh`，仅等待 `/workspace/logs/native_2d_followup_20260929/baseline_c64.done` 后顺序启动上述 GPU 2 测试；每项结果与总退出码分别记录在同目录的 `.json/.log/.exit`。不改动 9031 或当前 9032 baseline 服务。Level 3 尚未启动，不从后台运行推断通过。
- `FACT`（中间值，非最终汇总）：c64/16k baseline 第 2/4 轮 `1072.71 tok/s`，第 3/4 轮刚开始；已另行将 `cdc5fea` 的原始 candidate `attention.py` 和 native kernel 归档到远端 `candidate_pinned/`，用于准确率测试前消除本轮未验收的实验改动。后台 GPU 2 审核会使用实验分支，而 Level 3 必须使用冻结提交，不应混用。
- `ACTION`：另一个后台进程 PID `244481` 运行 `tools/continue_native_2d_level3.sh`，等待 c64 完成及审核结果。脚本仅在 early-prefill 数值审核通过后校验预先指定的 9032 baseline PID `240995`，终止该服务、恢复已暂存的 `cdc5fea` 候选源码、启用原有 marker，以带 prefix caching 的官方式配置在 GPU 1/9032 启动服务并运行既有 `level3_9032_command.sh`。全过程不请求、不停止 9031。成功标志 `candidate_level3.done`，失败退出码 `native_2d_level3.exit`，服务及 evalscope 日志分别为 `candidate_level3_service.log`/`candidate_level3.log`；此处只记录**计划并已启动等待器**，不能写作 Level 3 已通过。

#### Phase 1 c64 收口与 Phase 2 GPU 2 初筛（2026-09-29 23:37）

- `COMMAND`：`tools/continue_native_2d_audits.sh` 在 `baseline_c64.done` 后依次调用 `tools/run_native_2d_smoke.sh` 的 `early audit`、`mixed dual`、`mixed-order dual`；总退出码 `native_2d_followup_audits.exit=0`。日志、JSON 均在远端 `/workspace/logs/native_2d_followup_20260929/`。
- `FACT`：c64/4k、64 prompts/128 output、关闭 prefix caching，首轮丢弃后的 baseline `3228.47/3227.88/3227.49 tok/s`，candidate `6257.51/6252.65/6256.42 tok/s`；三轮均值约 `3227.95` vs `6255.53 tok/s`，提升 `93.79%`。
- `FACT`：相同 c64/16k 口径，baseline 稳态 `1072.71/1072.64/1072.54 tok/s`，candidate `3322.99/3323.49/3322.77 tok/s`；均值约 `1072.63` vs `3323.08 tok/s`，提升 `209.81%`。baseline Mean TTFT 三轮约 `495.6 s`，candidate 约 `155.85 s`；baseline Median ITL 每轮 `1688.57/1616.72/1604.25 ms`，candidate `564.18/559.23/562.80 ms`。服务日志证据与数值原始文件为 `baseline_c64_4k_16k.log`、`candidate_c64_4k_16k.log`。
- `FACT`：GPU 2 early-prefill 三个 q/prefix 小形状中，自写 fp32 reference 与独立 `torch.nn.functional.scaled_dot_product_attention` 的最大误差不超过 `0.000488`。已安装的原生 kernel 对参考的最大误差分别为 `1.640625/0.660156/1.640625`，调优候选为 `0.015625/0.0078125/0.015625`；原生偏差在小形状复现，未能归因到具体上游代码行，也不代表比赛准确率已通过。
- `FACT`：同一 mixed 输入下，tuned 单次 2D `13.691 ms`，实验性 dual-launch `12.584 ms`，GPU 2 五次测量约 `+8.8%`；decode-first `12.845` vs `12.615 ms`（约 `+1.8%`），interleaved `13.395` vs `12.632 ms`（约 `+6.0%`）。三个形状 dual 输出对 fp32 reference 的最大误差均 `0.000488`；Triton 已完成 GPU 编译。
- `BOUNDARY`：dual-launch 只有单次、每格五次的 GPU 2 初筛，尚无独立重复、纯 decode 3D 对照或服务端到端结果；不能声称 `>5%` 稳定收益，不启用服务开关，也不提交为完成的算法优化。c64 是短负载且关闭 prefix caching，非官方完整负载。Level 3 此刻在 9032 已评至 `94/105`，尚未聚合，不写通过结论。9031/9032 `/health` 均 HTTP `200`。
- `NEXT`：读取 Level 3 的 105/105 最终报告与失败题，核实是否 `>=102/105`；然后先做 dual-launch component timing 和重复 mixed A/B，再决定是否值得设计真正的同条件 3D decode 路径与服务集成。

#### 9032 Level 3 后台续跑状态（2026-09-29 23:45）

- `COMMAND`：通过现有 SSH ControlMaster 只读执行 `docker exec mllv`，检查 `/workspace/logs/native_2d_followup_20260929/{baseline_c64.done,native_2d_followup_audits.exit,native_2d_level3.exit,candidate_level3.done}`、evalscope 与 9032 服务日志和进程。
- `FACT`：`baseline_c64.done` 已出现，`native_2d_followup_audits.exit=0`；`candidate_level3.done` 和 `native_2d_level3.exit` 均未出现。evalscope PID `245719` 与 9032 candidate PID `245174` 仍存活，进度 `103/105`；服务持续记录 `Running: 2 reqs`、生成吞吐和 KV cache 使用率增长，9032 `/health` 返回 `200`。9031 PID `2943` 存活且未操作。
- `INFERENCE`：最后两题还在生成，并非从单纯的进度栏停顿即可断定卡死。没有最终 105 题汇总或准确率结论。
- `PROBLEM`：尚无；没有重新启动评测或服务，也没有更改候选代码。下次先读最终报告及实际样本数，再判断准确率。

#### Phase 1 Level 3 准确率收口（2026-09-29 23:57）

- `COMMAND`：只读检查 `native_2d_level3.exit`、`candidate_level3.done`、`candidate_level3.log`、报告 `/workspace/evalscope-datasets/level3_diag_20260929_231222/20260929_231227/reports/minicpm-diag/math_500.json` 与逐题 reviews JSONL。
- `FACT`：后台脚本退出码 `0` 且 done 文件存在。报告 `Level 3 / 105 / 97.1%`，逐题文件共 105 条，得分 102 条，未得分 3 条索引为 `18, 26, 87`；这与此前两次 baseline `102/105` 一致。9031 服务仍为 PID `2943`，9032 candidate PID `245174` 保持运行。
- `INFERENCE`：固定 candidate 满足本轮 Level 3 与基线持平的准确率门槛；单轮通过不代替官方完整性能负载和更大规模随机性验证。
- `NEXT`：GPU 2 隔离重复 mixed dual-launch 与 tuned 单次路径的配对计时，验证初筛的 1.8%-8.8% 是否稳定；原型不启用 9032 服务路由。

#### GPU 2 dual-launch 隔离复测启动（2026-09-30 00:01）

- `COMMAND`：将当前远端工作树复制到 `/workspace/native2d_gpu2_prototype_20260929`，只向副本同步本地 `tools/native_2d_harness.py` 与 `triton_unified_attention_native.py`；SHA-256 分别为 `d33e8e...`、`5302ba...`，通过远端 `py_compile`。在副本 `PYTHONPATH` 上以 GPU 2 运行 `--shape mixed/mixed-order --block-m 128 --tile 16 --warps 8 --stages 2 --pipeline-stages 1 --scalar-block-lookup 0 --mixed-dual-launch --warmup 5 --iterations 20 --limit 1`，seed `20260929/20260930/20261001`，各次 240 秒上限，总命令 1600 秒上限，结果到 `/workspace/logs/native_2d_followup_20260929/dual_repeat_20260930/`。
- `FACT`：9032 Level 3 曾恢复工作树文件为已提交 candidate，当前主工作树 kernel SHA-256 `430f249...`，原型副本 kernel 为 `5302ba...`。没有修改运行中的 9032 或 9031。GPU 2 启动前显存占用约 68 MiB、利用率 0%。第一轮 mixed JSON 已产出，dual `12.579 ms`，相对 tuned 比值 `1.08697`，对参考最大误差 `0.000488`。
- `UNKNOWN`：其余五次测量、顺序变化下的稳定性及服务收益；此刻不能启用服务路由或推送原型。

#### GPU 2 dual-launch 三轮复测（2026-09-30 00:03）

- `FACT`：六份 JSON 全部生成且每个形状 `status=ok`。相对 tuned 单次 2D 的 `tuned_ms/dual_ms`：普通 mixed 三轮 `1.0870/1.0918/1.0950`；decode-first `1.0271/1.0223/1.0300`；interleaved `1.0589/1.0623/1.0664`。各形状中位数约 `+9.18%/+2.71%/+6.23%`。
- `BOUNDARY`：同一个 harness 依序测 baseline、tuned、dual，三个 seed 的重复尚未排除固定先后测量偏差；decode-first 也未达到预设 `>5%` 的独立收益门槛。以上只是 GPU 2 微基准，不是服务吞吐提升。
- `NEXT`：在隔离副本中添加反序计时选项，再用同一三 seed/形状重复；不改 9032 运行中的冻结 candidate。

#### GPU 2 dual-launch 反序计时收口（服务器时间 2026-09-30 00:05）

- `CHANGE`：本地 `tools/native_2d_harness.py` 添加 `--dual-first`，只调整 mixed 测量顺序，先测 dual 再测 tuned；Python 语法检查与 `git diff --check` 通过，仅同步至 `/workspace/native2d_gpu2_prototype_20260929`，没有覆盖远端 9032 服务使用的冻结 candidate 文件。
- `COMMAND`：相同 GPU 2、三 seed、`mixed/mixed-order`、5 次 warmup/20 次计时/固定 BLOCK_M=128、TILE=16、warps=8、stages=2、pipeline_stages=1，再加 `--dual-first`。六份结果在 `/workspace/logs/native_2d_followup_20260929/dual_repeat_20260930/*reverse.json`，全部 `status=ok`。
- `FACT`：反序后的 `tuned_ms/dual_ms`：普通 mixed `1.0952/1.0958/1.0868`，decode-first `1.0310/1.0256/1.0227`，interleaved `1.0636/1.0687/1.0668`。相比正序中位数 `1.0918/1.0271/1.0623`，没有出现顺序反转后的收益消失；常规 mixed 与 interleaved 的微基准收益均超过 5%，decode-first 没有。
- `INFERENCE`：dual-launch 对已测 mixed 排列的数值与计时具备复现性，但 decode-first 的增益不够强；仍需 profiler 分解两个 launch 和在 9032 做短服务 A/B，再判断是否值得集成。GPU 2 的 2%-9% 不应写成 4k/16k 端到端增益。
- `BOUNDARY`：没有官方完整负载、逆序服务 A/B 或 dual-launch 服务准确率证据；原型保持默认关闭，尚未提交或推送。

#### GPU 2 dual-launch 分项 profiler（远端记录时间 2026-09-30 00:19）

- `COMMAND`：在隔离副本运行 `tools/profile_native_2d_dual.py --device cuda:2 --seed 20260929`，对普通 mixed、decode-first、interleaved 各自先 warmup 5 次，再在 `torch.profiler` 中各调用 tuned/dual 3 次；Chrome traces 和精简事件记录位于 `/workspace/logs/native_2d_followup_20260929/dual_profile/`。
- `FACT`：普通 mixed 的 tuned 单 kernel 三次为 `14.558/14.658/14.561 ms`；dual 每次先执行 prefill `9.536/9.578/9.503 ms`，后执行 decode `3.910/3.925/3.916 ms`。decode-first 的 tuned 约 `13.69--13.74 ms`，dual prefill 约 `9.51--9.55 ms` 加 decode 约 `3.93 ms`；interleaved tuned 约 `14.19--14.27 ms`，dual prefill 约 `9.55 ms` 加 decode 约 `3.89 ms`。
- `INFERENCE`：额外 launch/reduction 不是主要开销；dual 在普通 mixed 的 prefill 减少工作量，但独立 decode 仍约 3.9 ms，解释 decode-first 净收益仅约 2%-3%。单次 profile 不能证明 SM 占用率或带宽饱和，也不能外推服务吞吐。
- `CHANGE`：在实验分支的 Iluvatar dispatch 增加默认关闭的 `ILUVATAR_NATIVE_2D_DUAL`/marker 入口；仅 BF16、GQA=8、无 alibi/sinks/window/量化等特性时向现有 native kernel 传 `mixed_dual_launch_override=True`。本轮先只同步到 GPU 2 隔离副本，不覆盖运行中的 9032 候选。

#### dual-launch 路由与 9032 A/B 启动（远端时钟 2026-09-30 00:27）

- `COMMAND`：GPU 2 隔离副本执行 `tools/test_native_2d_dual_route.py`，设置 `ILUVATAR_NATIVE_2D_DUAL=1` 并以 paged-KV/GQA=8/2048 prefill+30 decode 调用真正的 `IluvatarAttentionBackend` dispatch；再用 sinks 输入检查回退。原始日志 `/workspace/logs/native_2d_followup_20260929/dual_route_smoke.log`。
- `FACT`：日志明确显示 `dual_launch=True`；捕获的原生调用参数为 `[true, false]`，对应受支持与不支持输入；受支持输出相对 fp32 参考最大绝对误差 `0.00048828125`，脚本退出码 `0`。
- `ACTION`：远端隔离副本启动 `tools/native_dual_service_ab.py --expected-pid 245174 --log-dir /workspace/logs/native_2d_dual_service_ab_01`。脚本先验证 9031/9032 健康与精确 PID，仅停止 9032，然后同 `FULL_DECODE_ONLY`、关闭 prefix caching 运行 tuned 与 dual；每组 `[[4096,128,16,16],[16384,128,16,16],[4096,128,64,64],[16384,128,64,64]]`、各四轮且首轮丢弃，检查 raw CSV 成功请求数与 summary 案例数；最后移除 dual marker，恢复带 prefix caching 的 9032 冻结候选。主日志 `runner.log`、退出码 `runner.exit`、结果 `results.json`。9031 不停止、不发评测请求。
- `FACT`：当前 tuned 服务 PID `246751` 正在加载模型，9032 暂未就绪；9031 `/health=200`。没有声称 A/B 成绩，也未把 dual 原型上线为默认路径。
- `PROBLEM`：A/B 编排在 CSV 解析中把 `Successful Requests` 当作整数字符串；实际 raw CSV 写 `16.0`，因此首轮 tuned benchmark 完成后会在校验阶段抛 `ValueError`。测量任务本身继续运行，不中断、不重复启动；旧编排的 `finally` 将恢复 9032 冻结候选。
- `FIX`：本地将成功请求数改按 `float` 校验，抽出 `read_benchmark`，并加入 `--reuse-tuned-dir`。修复脚本已同步到远端隔离副本，**不会改变已在运行进程所加载的旧脚本**。待旧脚本结束且 9032 恢复后，先核对 tuned 的四组 raw/summary，再只补 dual 组，避免重跑有效的 tuned 数据。

#### dual-launch 首轮编排故障收口（2026-09-30）

- `FACT`：远端 `/workspace/logs/native_2d_dual_service_ab_01/runner.exit` 为 `1`；`runner.log` 显示 tuned 测量已完成并随后启动恢复服务，恢复 PID 为 `252000`。旧脚本的失败点仍是对 raw CSV 中 `Successful Requests=16.0/64.0` 使用 `int(...)`，未进入 dual 测量。
- `FACT`：故障检查时 9031 与 9032 均返回 HTTP `200`，未发现正在运行的 A/B benchmark 进程；9031 未停止、未重启、未发送请求。
- `PROBLEM`：现有 SSH ControlMaster 在后续只读核查中无响应，无法安全确认当前恢复 PID 的命令行和 dual marker 状态，也无法启动补测。
- `DECISION`：不在连接异常时重复启动或强制停止任何服务。待通道恢复后，先只读确认 9031/9032、恢复 PID、marker 和 tuned CSV 四组各四轮成功，再使用已修复脚本加 `--reuse-tuned-dir` 仅补 dual；若无法确认状态则不继续实验。

#### dual-launch 补测重启与运行时环境修复（2026-09-30）

- `FACT`：复用 tuned 数据目录 `/workspace/logs/native_2d_dual_service_ab_01/tuned`，确认四个 case 均有四轮成功记录，后三轮 `Run Status=SUCCESS`，没有重跑 tuned。
- `PROBLEM`：第一次只补 dual 时，脚本从 mllv 容器外部 namespace 启动子服务，子进程缺少 `LD_LIBRARY_PATH`，导致 `torch` 找不到 Iluvatar 的 `libcudart.so.10.2`/`libcublas.so`；dual 与恢复候选均在启动阶段退出。9031 保持健康，随后手动用完整容器运行时环境恢复 9032。
- `FIX`：`tools/native_dual_service_ab.py` 的 `start()` 与 benchmark 子进程显式继承 `/usr/local/corex/lib64:/usr/local/openmpi/lib:/usr/local/lib`；修复版已同步到 `/workspace/native2d_gpu2_prototype_20260929/tools/` 并通过远端 `py_compile`。此修复只影响实验编排，不改变 kernel 路由。
- `COMMAND`：在 `mllv` 容器内以当前 9032 容器 PID `253110` 为锚点重新执行 `--reuse-tuned-dir`，新日志目录 `/workspace/logs/native_2d_dual_service_ab_03`；当前 dual 服务 PID `253649`，benchmark PID `254188`。
- `FACT`：截至记录时 9031/9032 均 HTTP 200，dual marker `/tmp/iluvatar_native_2d_dual.enable` 存在，dual benchmark 正在进行；尚未生成 dual summary/raw，不提前判定收益。官方 9031 未停止、未发送请求。
- `NEXT`：等待 dual 四组 raw/summary 完成；完成后核对 `dual_launch=True`、吞吐、TTFT、Median/P99 ITL，确认脚本 finally 移除 dual marker 并恢复带 prefix caching 的冻结 9032，再决定是否进入官方完整负载。

#### dual-launch 服务级 A/B 收口（2026-09-30）

- `COMMAND`：复用 `/workspace/logs/native_2d_dual_service_ab_01/tuned` 的四组 tuned 汇总，仅在 9032/GPU 1 运行修复后的 `native_dual_service_ab.py --reuse-tuned-dir` dual 测量；dual 日志目录为 `/workspace/logs/native_2d_dual_service_ab_03`。负载为 `4096/16384 input × c16/c64 × 128 output`，每组 4 轮、首轮预热，关闭 prefix caching；9031 全程未停止、未重启、未发送请求。
- `FACT`：dual 原始 CSV 共 16 条成功记录，四个 case 各 4 轮，`Successful Requests` 分别为 `16.0/64.0`；tuned 复用目录也已确认四组各 4 轮且后三轮成功。dual 服务日志明确记录 `dual_launch=True`，证明本轮真实进入双 launch 路由，而非仅运行配置脚本。
- `FACT`：tuned 汇总吞吐为 `5758.72 / 3333.25 / 6247.52 / 3323.13 tok/s`（依次为 4K/c16、16K/c16、4K/c64、16K/c64）；dual 为 `5592.27 / 3170.61 / 6324.50 / 3192.08 tok/s`。相对 tuned 分别为 `-2.89% / -4.88% / +1.23% / -3.95%`，没有一个 case 达到服务收益门槛 `+5%`。
- `FACT`：dual Mean TTFT 为 `4598.67 / 40464.47 / 17675.33 / 162518.46 ms`，相对 tuned `4314.30 / 37609.16 / 17571.92 / 155860.90 ms` 分别为 `+6.59% / +7.06% / +0.59% / +4.28%`，不存在 TTFT 改善。Median ITL 在 4K/c64 从 `139.96` 增至 `175.36 ms`（约 `+25.3%`），其余 case 约持平或小幅变化；P99 ITL 没有恶化，但不能抵消吞吐与 TTFT 结果。
- `INFERENCE`：GPU 2 微基准中普通 mixed 的约 `+9%` dual 优势没有迁移到真实服务；服务调度、kernel launch 次序及请求混合状态会吞掉该收益。dual-launch 原型不应作为提交版本或默认路由。
- `DECISION`：停止 dual-launch 服务线的进一步大规模 A/B，不进入官方完整负载，不推送未验收原型。保留代码、日志和结果作为负面消融证据；当前冻结 candidate 仍是单次 native 2D tuned 路径。
- `FACT`：编排 finally 已移除 `/tmp/iluvatar_native_2d_dual.enable`，并恢复带 prefix caching 的冻结 9032 candidate（PID `259078`）；当前仅保留 `/tmp/iluvatar_native_2d.enable`。9031 PID `2943` 未改变，9031/9032 `/health` 均为 HTTP `200`。
- `BOUNDARY`：本轮 dual 与 tuned 使用同一服务负载和稳定轮汇总，但各配置仍是一次编排中的三轮稳态统计，不是跨天随机交叉 A/B；因此结论是“当前 dual 设计未显示可提交收益”，不是对所有可能的 kernel 分拆算法作理论上的否定。

#### 冻结 tuned candidate 官方完整负载验收启动（2026-09-30）

- `DECISION`：dual-launch 已证明没有稳定服务收益，停止其线路；当前只对冻结的单路 Native 2D tuned candidate 做官方口径完整负载，不向 9031 发送请求。
- `COMMAND`：9032/GPU 1 使用与 9031 相同模型和 `FULL_DECODE_ONLY` 配置，运行 `benchmark_throughput_serve.py --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'`。脚本按既有约定每个 case 执行 4 轮并跳过首轮；日志与结果保存到 `/workspace/logs/native_2d_official_candidate_20260930/`。
- `BOUNDARY`：本轮只确认冻结 candidate 在比赛官方负载下的吞吐、TTFT、ITL 和请求成功率；与 9031 的历史官方 baseline 比较时，必须同时注明运行时间、服务状态和 prefix-cache 配置，不能把不同口径的短测混为最终成绩。
- `FACT`：远端后台 benchmark 已启动，PID `259689`；初始日志确认脚本读取到 `RUNS=4`、`SKIP_FIRST=1`，第一组为 `4096/1024/c64/256`。启动后 9031/9032 均 HTTP `200`，没有启动第二个 benchmark。
- `FACT`：完整负载已完成，raw CSV 共 8 条运行记录加表头，两组各 4 轮且请求全部成功；benchmark 已退出。稳定汇总为 4K/c64/256：`2454.37 tok/s`、Mean TTFT `5843.0 ms`；16K/c64/128：`1790.51 tok/s`、Mean TTFT `282168.89 ms`。结果文件为 `/workspace/benchmark_results/raw_runs_20260930_084312.csv` 和 `/workspace/benchmark_results/summary_20260930_084312.csv`。
- `INFERENCE`：与实验记录中的官方基线 `1983.45 tok/s`（4K）和 `917.83 tok/s`（16K）相比，冻结 tuned candidate 分别约提升 `23.8%` 和 `95.1%`；Mean TTFT 分别约下降 `49.6%` 和 `52.8%`。该比较沿用历史官方基线，最终提交前仍应在同一环境窗口做一次交叉复验。
- `FACT`：benchmark 结束后 9031 PID `2943`、9032 PID `259078` 均保持运行，健康检查均为 HTTP `200`；9032 仍保留 `/tmp/iluvatar_native_2d.enable` 单路 tuned marker，未启用 dual marker。

#### 同窗口官方负载 baseline/candidate 交叉复验启动（2026-09-30）

- `CHANGE`：新增验收脚本 `vllm-plugin-FL/tools/native_official_ab.py`。它只控制 9032/GPU 1，baseline 阶段移除 native marker，candidate 阶段创建 marker，二者使用相同 `FULL_DECODE_ONLY`、模型、端口、prefix-cache 配置和官方两组负载；finally 始终恢复 candidate。脚本已通过本地及 mllv 容器内 `py_compile`。
- `COMMAND`：编排 PID `262945`，初始 9032 PID `259078` 已通过命令行和健康检查校验；日志目录 `/workspace/logs/native_official_ab_20260930/`。9031 PID `2943` 未停止、未重启、未发送 benchmark 请求。
- `FACT`：baseline 服务已完成启动和 CUDA graph 初始化，benchmark PID `263492` 已进入运行；9031/9032 当前均 HTTP `200`。本轮尚无 baseline/candidate 最终 CSV，不提前计算收益。

#### 同窗口官方负载 A/B 恢复后只读核查（2026-09-30）

- `COMMAND`：SSH ControlMaster `~/.ssh/codex-control/ub39-native-recover` 恢复后，仅检查编排 PID、benchmark PID、服务健康、日志和 marker；未停止、重启或重复启动任何进程。
- `FACT`：当前编排 PID `570321` 仍在运行，baseline benchmark PID `572488` 仍在运行；9031/9032 `/health` 均返回 HTTP `200`。
- `FACT`：4K/c64/256 baseline 已完成 4 轮；16K/c64/128 baseline 当前处于 `Run 1/4`，其请求子进程仍在运行。baseline benchmark 日志暂未生成最终 CSV；candidate 尚未开始。
- `FACT`：9032 当前仍为 baseline 测量服务，`/tmp/iluvatar_native_2d.enable` marker 在 baseline 期间已移除；编排器设计为 baseline 完成后再启动 candidate，并在 finally 恢复 candidate。
- `INFERENCE`：A/B 尚未产生可用于最终比较的同窗口数据，不能提前确认 tuned candidate 的官方收益；当前无故障证据，不干预长时负载。
- `NEXT`：等待 baseline 两个 case 完成后，再读取 baseline raw/summary；随后等待 candidate 两个 case 完成，最后核对 9032 恢复 candidate、9031 未变化，再计算吞吐、TTFT、Median/P99 ITL 和轮间波动。

#### 同窗口官方负载 A/B 超时与 9032 恢复故障（2026-09-30）

- `FACT`：A/B 编排 PID `570321` 最终因 benchmark 子进程超过 `10800s` 超时退出；`runner.log` 随后记录 stop 旧 9032 PID `262949` 未能 cleanly terminate。baseline 目录已有 4K/c64/256 四轮成功，以及 16K/c64/128 两轮成功；candidate 阶段未开始，没有 `results.json`。
- `FACT`：超时后残留的 9032 baseline 服务 PID `570374` 和 16K benchmark PID `660607` 已在 9032 范围内清理；9031 PID `2943` 未改变。清理操作未向 9031 发送请求。
- `FAILURE`：首次恢复尝试误在宿主 shell 使用容器内路径 `/usr/local/bin/python3.12`，立即失败；第二次进入 `mllv` 容器时多层 shell 破坏 `--compilation-config` JSON，引发 vLLM 参数校验失败。两次均未改变 9031。
- `COMMAND`：已改为容器内 base64 写入启动脚本，使用完整 `CUDA_VISIBLE_DEVICES=1`、`VLLM_PLUGINS=fl`、`PYTHONPATH`、`LD_LIBRARY_PATH` 和冻结 candidate marker 启动 9032；SSH 在等待后台命令时再次无响应，尚未取得健康回执。
- `BOUNDARY`：截至本记录追加时，不能声称 9032 已恢复，也不能继续任何 benchmark 或新实验；下一步必须先建立新 SSH 通道，只读确认 9031/9032、9032 服务 PID、candidate marker 和恢复日志，再决定是否需要一次恢复动作。
- `FACT`：新建 ControlMaster `ub39-native-recover-3` 后，认证可以建立，但包含容器查询的远端只读命令在 25 秒内未返回任何输出；未执行新的服务操作，也未触碰 9031。
- `NEXT`：改由已连接的远端交互终端执行最小健康检查，先取得 9031/9032 HTTP 状态和 `mllv` 容器内 9031/9032 服务 PID，再继续恢复或验证。

#### 9032 candidate 恢复确认（2026-09-30）

- `FACT`：通过用户已连接的远端终端确认 9031 `/health=200`、9032 `/health=200`。
- `FACT`：9031 服务 PID 仍为 `2943`；9032 已恢复为 PID `266232`，命令行为 `/usr/local/bin/vllm serve /workspace/MiniCPM5-2B --port 9032 --served-model-name minicpm-diag ... --compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY"}`。
- `DECISION`：9032 已恢复可用，继续保持 candidate 服务；不重跑原始 `10800s` 超时的整轮 A/B。下一步先确认 candidate marker/路由日志，再将 baseline 与 candidate 拆成可控的单 case、少量稳态轮次复验，避免再次把服务留在不可用状态。

#### 当前阶段完成度评估（2026-09-30）

- `FACT`：当前官方同窗口 A/B 验收轮只完成 baseline 侧的 4K 四轮和 16K 两轮；candidate 侧为零轮，且编排因 16K 长测超过 `10800s` 超时退出。因此这一个 A/B 轮按实验进度约为 `40%`，按“形成可比较结论”的交付物完成度为 `0%`。
- `FACT`：Native 2D tuned kernel 已实现并有历史完整负载结果（4K/c64 约 `+23.8%`、16K/c64 约 `+95.1%`），Level 3 为 `102/105=97.1%`；这些是已有证据，不属于本次失败 A/B 的新结论。
- `FACT`：dual-launch 已完成服务级消融并否决；当前 9032 已恢复单路 candidate，9031 未改动。
- `FACT`：scalar block lookup 的独立数值/计时消融、完整参数扫描、最终同窗口 A/B、代码清理归档、最终提交和 GitHub 推送均未完成。
- `INFERENCE`：整个优化项目按“核心实现与历史验证、负面方向收口、最终证据和交付”综合估算约 `65%`；剩余约 `35%` 主要是可复现验收、scalar/参数消融和工程收尾，不是重新设计 Native 2D 核心算法。
- `NEXT`：不再运行原始双 case 长编排；先将 4K 和 16K 拆成独立短测，限制每个 case 的稳态轮次和单轮超时，完成 baseline/candidate 同窗口对照后再进入 scalar lookup 消融。

#### 八卡并行实验编排准备（2026-09-30）

- `CHANGE`：本地新增 `vllm-plugin-FL/tools/run_native_2d_parallel_grid.sh`，默认只调度 GPU `2-7`，将 GPU `0/1` 作为 9031/9032 保留卡；六个 worker 分别覆盖 scalar lookup、TILE_SIZE、BLOCK_M、warps/stages 参数子空间。
- `CHANGE`：修正 `tools/native_2d_harness.py` 的组合过滤，使 `TILE_SIZE=8` 不再被无条件跳过；该值仅用于当前合法的 page-size=16 scalar/vector 消融。
- `FACT`：本地脚本已通过 `python3 -m py_compile` 和 `bash -n`；尝试经 SSH ControlMaster 同步到远端时认证通道返回 `Permission denied`，远端尚未确认包含这两个最新文件。
- `BOUNDARY`：不能据此声称八卡扫描已经启动。启动前仍需在用户已连接的 `mllv` 终端确认 GPU `2-7` 空闲；并行只适用于 GPU 2-7 的 kernel/正确性扫描，9032 服务级 A/B 仍必须串行。
- `NEXT`：先在 `mllv` 内启动六路扫描并保留每卡独立 JSON/log；扫描完成后只把通过数值门槛的前一到两个配置送入 9032 做拆分短 A/B，最后再运行官方完整负载和 Level 3（仅当最终 kernel/routing 有变化）。

#### 八卡并行扫描已启动（2026-09-30）

- `FACT`：用户已在 `mllv` 内成功启动六个独立 worker：GPU 2 PID `266811`、GPU 3 PID `266812`、GPU 4 PID `266813`、GPU 5 PID `266814`、GPU 6 PID `266815`、GPU 7 PID `266816`。
- `FACT`：输出目录为 `/workspace/logs/native_2d_grid_20260930_154719`，每卡分别写入 `gpu<N>.json`、`gpu<N>.log`、`gpu<N>.pid`。
- `FACT`：GPU 0/1 未被此次扫描使用，9031/9032 服务未重启；当前 9032 继续保持单路 Native 2D candidate。
- `DECISION`：不再启动第二批扫描。待六个 worker 结束后，按数值正确性、kernel event time 和 4K/16K 稳定性筛选前一到两个配置，再做 9032 短 A/B。

#### 八卡扫描首轮失败与远端代码同步修正（2026-09-30）

- `FACT`：GPU 2-7 首轮 worker 均已退出并生成 JSON，但每条记录都是 `TypeError: unified_attention() got an unexpected keyword argument 'pipeline_stages_override'`；该批没有产生任何有效 kernel 时间或正确性结果，不能计入参数扫描进度。
- `FACT`：`ixsmi` 显示实际节点有 16 张 BI-V150；GPU 0/1 分别由 9031/9032 占用，GPU 2-15 当时为约 `68MiB` 空闲状态。后续可用 GPU 2-15，而不是只用 2-7。
- `FACT`：远端 `/workspace/vllm-plugin-FL` 的 Native 2D kernel 版本落后本地，缺少 `pipeline_stages_override` 和 `scalar_block_lookup_override`；本地最新 `tools/native_2d_harness.py`、`tools/run_native_2d_parallel_grid.sh`、`triton_unified_attention_native.py` 已通过 tar 管道同步到 `mllv` 容器内。
- `FACT`：同步过程未重启或修改 9031/9032；两个服务此前均 HTTP `200`。同步后的导入/语法复核因 SSH 执行通道再次无响应，尚未取得远端回执。
- `DECISION`：不把首轮任务当成完成，不启动第二批直到拿到远端语法/签名校验结果；校验通过后再使用 GPU 2-15 分片并行扫描。

#### 16 卡节点第二批并行扫描启动（2026-09-30）

- `FACT`：新 SSH 通道恢复后，远端验证通过：9031/9032 均 HTTP `200`；同步后的 harness 语法检查通过；远端 Native 2D 函数包含 `pipeline_stages_override` 和 `scalar_block_lookup_override`。
- `FACT`：GPU 2-15 共 14 个 worker 已启动，PID 为 `267972-267985`（按 GPU 2 到 15 顺序），输出目录为 `/workspace/logs/native_2d_grid_20260930_161300`。每卡使用独立结果 JSON、日志和 Triton cache。
- `FACT`：本批覆盖 scalar/vector lookup、TILE_SIZE、BLOCK_M、warps/stages、pipeline hint、mixed-order、tail correctness 和重复稳定性；GPU 0/1 保留给 9031/9032。
- `DECISION`：不重复启动其他扫描。待结果生成后，先过滤 `status=ok`、数值误差和 kernel event time，再选择前一到两个配置进入 9032 单 case 短 A/B。


#### 16K/c8 拆分短 A/B 完成（2026-09-30）

- `COMMAND`：在 `mllv` 容器内、9032/GPU 1 运行独立单 case 编排，负载为 `16384 input / 64 output / c8 / 8 prompts`，baseline 与 candidate 各 4 轮、跳过首轮；服务启动显式关闭 prefix caching，日志目录为 `/workspace/logs/native_2d_service_short_16k_20260930/`。9031 全程未停止、未重启、未发送 benchmark 请求。
- `FACT`：baseline 四轮全部成功；后三轮汇总总吞吐 `1086.92 tok/s`，Mean TTFT `67045.43 ms`，Median ITL `48.59 ms`，P99 ITL `3466.68 ms`。candidate 四轮全部成功；后三轮汇总总吞吐 `3668.17 tok/s`，Mean TTFT `18900.37 ms`，Median ITL `48.24 ms`，P99 ITL `809.31 ms`。所有请求成功数均为 `8/8`。
- `FACT`：candidate 相对 baseline 总吞吐约 `+237.5%`，Mean TTFT 约下降 `71.8%`；Median ITL 约下降 `0.7%`，P99 ITL 约下降 `76.7%`。candidate 三个稳态轮次为 `3666.52/3668.66/3669.32 tok/s`，轮间 CV 约 `0.03%`，短测稳定性良好。
- `INFERENCE`：该结果确认 Native 2D tuned 在 16K/c8 的 prefill-heavy 短负载中有稳定服务收益，但它与官方 c64 负载不同，不能直接替代官方最终成绩。
- `FACT`：编排 finally 已恢复 9032 candidate，当前服务 PID `281563`，9031 PID `2943`；两端 `/health` 均为 `200`。本轮恢复服务仍带 `--no-enable-prefix-caching`，进入最终验收前需要重启为标准冻结 candidate 参数。
- `NEXT`：恢复标准 candidate 后运行 Level 3，并执行官方两组完整负载；最终报告区分短测与官方口径。

#### 最终 Level 3 与官方完整 candidate 负载（2026-09-30）

- `FACT`：Level 3 评测已完成，MATH-500 Level 3 共 `105` 题，正确 `101/105`，准确率 `96.19%`。评测输出位于 `/workspace/evalscope-datasets/level3_diag_20260930_171912/20260930_171918/`。
- `FACT`：官方完整 candidate 负载退出码为 `0`，两组共 `8` 个 raw runs、请求全部成功。原始结果为 `/workspace/benchmark_results/raw_runs_20260930_175541.csv`，汇总结果为 `/workspace/benchmark_results/summary_20260930_175541.csv`。
- `FACT`：4K/c64/256 candidate：总吞吐 `2427.74 tok/s`，Mean TTFT `5701.29 ms`，Median ITL `96.87 ms`，P99 ITL `442.10 ms`；后三轮总吞吐为 `2448.04/2321.47/2513.72 tok/s`。
- `FACT`：16K/c64/128 candidate：总吞吐 `1841.89 tok/s`，Mean TTFT `273400.12 ms`，Median ITL `147.66 ms`，P99 ITL `1011.41 ms`；后三轮总吞吐为 `1841.93/1841.84/1841.89 tok/s`。
- `BOUNDARY`：本轮只完成官方口径 candidate 负载，没有同一时间窗口、同一脚本参数的 baseline 对照，因此不能仅凭上述绝对值宣称新的官方增益；最终增益仍应引用已有同口径 A/B 数据或补跑 baseline。
- `FACT`：9031 PID 仍为 `2943`，9032 PID 为 `282180`；两端健康检查均为 HTTP `200`。本轮未修改 9031。
- `DECISION`：Native 2D 参数扫描、Level 3 和官方 candidate 完整负载已完成；scalar lookup 没有稳定收益，不固化。Split-KV、dual-launch 保持否决。
- `NEXT`：进入代码清理与最终交付审查：区分生产修改和实验脚本，补齐文档；若比赛要求严格同窗口官方增益，再单独安排可控的 baseline/candidate 对照，不与当前结果混报。

#### 提交收尾（2026-09-30）

- `CHANGE`：生产 dispatcher 已收敛为 Native 2D tuned 路径。对 BF16、无量化、causal、GQA=8 的 prefill 形状默认使用 `BLOCK_M=128`、`TILE_SIZE=16`、`num_warps=8`、`num_stages=2`、`pipeline_stages=1`；decode 和不支持的 attention 特性继续回退 vLLM 原生路径。
- `CHANGE`：Split-KV 文件已移至 `vllm-plugin-FL/experiments/native_2d_tuning/rejected_split_kv/`；dual-launch、profiler、A/B 和并行扫描脚本已移至 `vllm-plugin-FL/experiments/native_2d_tuning/service_and_profiler/`，不再被生产代码导入。
- `FACT`：生产代码、Native 2D harness 和实验脚本均通过 Python 语法检查；`git diff --check` 通过；实验 shell 脚本通过 `bash -n`。
- `CHANGE`：已生成比赛提交材料目录 `submission/FlagOS_S2_MiniCPM/`，包含 `readme.md`、`report.pdf` 和完整 `vllm-plugin-FL` 源码副本。报告明确区分同窗口短 A/B 相对收益与官方 candidate-only 绝对结果。
- `BOUNDARY`：未向比赛平台上传材料、未创建或推送外部 PR；提交包已准备好，等待人工最终审阅和正式上传。

#### Native 2D 后续调度优化准备（2026-10-01）

- `FACT`：本地仓库当前 HEAD 为 `02badf7 chore: keep official benchmark unchanged`，工作分支跟踪 `fork/jl2026-native-2d`；仓库下已有未跟踪嵌套目录 `vllm-plugin-FL/`，本轮未触碰或删除。
- `FACT`：远端只读健康检查成功：9031 与 9032 均返回 HTTP `200`；9031 PID `2943`，9032 PID `282180`。`/tmp/iluvatar_native_2d.enable` 存在，dual marker 不存在；未停止、重启或向 9031 发送请求。
- `FACT`：容器内 `SchedulerConfig.max_num_batched_tokens` 默认值为 `2048`，`enable_chunked_prefill` 默认开启；当前 9032 启动命令未显式覆盖 `--max-num-batched-tokens`。
- `CHANGE`：新增实验脚本 `vllm-plugin-FL/experiments/native_2d_tuning/service_and_profiler/scheduler_short_ab.py`。它只控制 9032/GPU 1，依次测试 `2048/4096/8192/16384`，负载为 `16384/64/c8/8`，每组 4 轮并跳过首轮，关闭 prefix caching，finally 恢复默认 scheduler 的冻结 candidate。
- `FACT`：脚本已通过本地 `py_compile` 与 `git diff --check`；生产 attention 路径没有修改。
- `FAILURE`：向远端同步脚本时，目标实验目录的 tar 解包返回 `No such file or directory`；随后针对容器路径的只读 `docker exec` 多次无回执。为避免在无法确认容器状态时误操作，未启动调度 A/B、未重启 9032、未修改任何远端生产文件。
- `BOUNDARY`：调度实验目前是“本地编排已准备、远端尚未执行”，不能声称任何 scheduler 收益。恢复可用远端终端后，第一步仍是确认容器工作树路径、9032 PID/marker 和无残留 benchmark，再同步脚本并只启动一次。

#### 剩余实验批处理启动状态（2026-10-01）

- `FACT`：远端容器路径已重新确认：`/workspace/vllm-plugin-FL` 存在，GPU 2-15 为空闲状态；9031/9032 健康，9031 PID `2943`，初始 9032 PID `282180`，tuned marker 存在且 dual marker 不存在。
- `COMMAND`：已同步并通过远端语法检查 `scheduler_short_ab.py`，随后以 9032 PID `282180` 启动唯一一轮调度 A/B，日志目录为 `/workspace/logs/native_2d_scheduler_20261001/`，runner PID `287053`。该编排依次测试 `2048/4096/8192/16384`，每组 4 轮并跳过首轮，finally 恢复默认 scheduler 的 tuned candidate。
- `FACT`：启动后首次检查确认 `scheduler_short_ab.py` 与其 9032 子服务 PID `287056` 正在运行，当前配置为 `--max-num-batched-tokens 2048`；未发现第二个 benchmark 或第二个调度编排。
- `CHANGE`：本地新增但尚未确认远端启动的 GPU 审计脚本：`run_adaptive_kernel_audit.sh`（GPU 2-5，形状自适应参数审计）和 `profile_q_block_mapping.py`（GPU 6，exact q-block metadata 成本测量）。二者均不修改生产 kernel。
- `BOUNDARY`：GPU 审计的后台启动命令随后没有返回回执；没有重试，不能声称这些 worker 已运行。不能把其结果写入性能结论。调度 A/B 仍保持单实例运行，不干预 9031/9032。

#### 调度短 A/B 与 GPU 审计收口（2026-10-04）

- `FACT`：调度 A/B 已完成，四组 `16384/64/c8/8` 均四轮成功，首轮作为预热；结果目录为 `/workspace/logs/native_2d_scheduler_20261001/`。9032 finally 已恢复默认 scheduler 的 tuned candidate，当前 9031/9032 均 HTTP `200`，tuned marker 存在，dual marker 不存在。
- `FACT`：稳态总吞吐中位数分别为 `mbt2048=3665.36`、`mbt4096=3958.93`、`mbt8192=4051.06`、`mbt16384=4094.06 tok/s`。相对默认 `2048`，`4096/8192/16384` 分别约 `+8.0%/+10.5%/+11.7%`；各配置稳态轮间波动很小。
- `FACT`：稳态 Mean TTFT 分别约为 `18905.77/17718.51/18180.98/19725.11 ms`；`4096` 最低，约比 `2048` 低 `6.3%`，`8192` 低 `3.8%`，`16384` 反而高 `4.3%`。P99 ITL 分别约为 `905.32/1435.91/2145.64/4120.82 ms`，更大 batch token 配置的尾延迟明显恶化。
- `INFERENCE`：在本次 16K/c8 短负载下，`max_num_batched_tokens=4096` 是延迟/吞吐更平衡的候选；`8192` 适合追求吞吐但 P99 ITL 代价明显；`16384` 不建议默认上线。该结论尚未经过官方 c64 完整负载，不能改写为比赛最终收益。
- `FACT`：GPU 2-5 自适应 kernel 审计全部完成且正确性通过；`BLOCK_M=128/TILE=16/warps=8` 在 pure、mixed、mixed-order 仍为稳定配置。`BLOCK_M=64` 在 pure prefill 明显慢，mixed 也未优于当前组合；`num_stages=1/2` 差异很小，不能形成稳定服务优化。
- `FACT`：GPU 6 exact q-block metadata 原型完成：典型 mixed 仅减少 `2/160` 个 CTA，chunked 16K 减少 `32/1056` 个 CTA，但 metadata 构造约 `1.27-1.58 ms`；当前不值得接入生产 kernel。
- `DECISION`：本轮不修改生产 Native 2D 参数、不启用 scalar lookup、不接入 exact q-block metadata；9032 保持已验证的 tuned candidate 默认调度配置。调度 `4096/8192` 仅作为后续官方 c64 短 A/B 候选，未验证前不提交或推送。

#### 16K/c64 调度候选 A/B 启动（2026-10-04）

- `COMMAND`：新增并同步 `scheduler_c64_ab.py`，只控制 9032/GPU 1；使用 `16384 input / 64 output / c64 / 64 prompts`，比较默认 `max_num_batched_tokens=2048` 与 `4096`，每组四轮并跳过首轮，prefix caching 关闭，finally 恢复常规 tuned candidate。
- `FACT`：启动前 9031/9032 均 HTTP `200`，9031 PID `2943`，9032 PID `311154`；tuned marker 存在、dual marker 不存在，无残留 benchmark。runner PID `311743`，当前 9032 已启动默认 `2048` 配置，9031 未停止、未重启、未发送请求。
- `BOUNDARY`：本轮尚未产生 c64 结果，不能提前宣称 `4096` 在比赛口径下有收益；完成后需核对两组 raw/summary、9032 恢复 PID、marker 和双服务健康状态。

#### 16K/c64 调度候选 A/B 收口（2026-10-04）

- `FACT`：同口径 16K/c64/64 A/B 已完成，默认 `max_num_batched_tokens=2048` 与候选 `4096` 各 4 轮、后三轮稳态、64/64 请求全部成功；结果目录为 `/workspace/logs/native_2d_scheduler_c64_20261004/`。
- `FACT`：默认 2048 汇总总吞吐 `3620.49 tok/s`，Mean TTFT `145443.36 ms`，Median ITL `542.11 ms`，P99 ITL `944.61 ms`；4096 汇总总吞吐 `3927.51 tok/s`，Mean TTFT `132780.26 ms`，Median ITL `965.86 ms`，P99 ITL `1685.78 ms`。
- `FACT`：4096 相对 2048 的吞吐提升约 `+8.48%`，Mean TTFT 改善约 `-8.71%`，但 Median ITL 恶化约 `+78.17%`，P99 ITL 恶化约 `+78.46%`；P99 TTFT 改善约 `-8.26%`，无法抵消 decode 交互延迟恶化。
- `DECISION`：`max_num_batched_tokens=4096` 不作为默认生产配置，也不进入比赛最终版本；它证明更大 prefill chunk 能提高吞吐，但会明显牺牲 decode 延迟。继续保持默认 2048，除非比赛明确只按吞吐评分且接受交互延迟代价。
- `FACT`：A/B 结束后曾发现恢复服务仍为无 prefix cache 的实验进程，已停止并重新启动标准 tuned candidate PID `316831`；当前 9031/9032 均 HTTP `200`，tuned marker 存在，dual marker 不存在，无残留 benchmark。

#### 最终官方 candidate 复核与 Level 3 复核（2026-10-04）

- `COMMAND`：在标准 9032 Native 2D tuned candidate 上启动官方负载 `[[4096,1024,64,256],[16384,1024,64,128]]`，每组 4 轮；9031 全程未停止、未重启、未发送请求。日志目录为 `/workspace/logs/native_2d_final_accept_20261004/`。
- `FACT`：4K/c64 四轮全部完成且请求成功，输出吞吐分别为 `414.71/482.93/467.65/502.98 tok/s`；对应 Mean TTFT 为 `7721.05/5792.82/5677.92/5653.23 ms`，Median ITL 为 `97.00/97.09/97.25/96.79 ms`，P99 ITL 为 `475.09/433.67/457.27/429.72 ms`。
- `FAILURE`：16K/c64 官方负载在混合官方脚本中第 1 轮出现极端异常：完成 `128/128` 请求但吞吐仅 `106.00 tok/s`、Mean TTFT `279802.68 ms`；随后第 2 轮长时间停在请求启动阶段。清理后以独立单 case 方式再次补跑 16K/c64，仍在首轮启动阶段超过 9 分钟且无统计，随后安全终止。该异常结果不纳入最终成绩。
- `FACT`：Level 3 运行了 105 题中的 104 题，预测/评测文件已生成；其中 `101/104` 正确，部分准确率 `97.1154%`。最后 1 题在服务端等待约 29 分钟无新日志，评测被安全终止，因此没有官方完整 `101/105` 汇总结果，不能将部分结果表述为完整通过。
- `FACT`：Level 3 原始文件位于 `/workspace/evalscope-datasets/level3_diag_20261004_142329/20261004_142335/`；部分结果为 `reviews/minicpm-diag/math_500_Level 3.jsonl.rerun-4fb2003092474f7d95435d81c8cb61e7`。官方 candidate 复核日志为 `/workspace/logs/native_2d_final_accept_20261004/official_candidate.log`，独立 16K 补测日志为 `/workspace/logs/native_2d_final_accept_20261004/final_16k_only/official_16k.log`。
- `FACT`：收尾时 9031 PID `2943`、9032 PID `316831` 均保持运行，健康检查均为 HTTP `200`；`/tmp/iluvatar_native_2d.enable` 存在，dual marker 不存在，无残留 benchmark 客户端。
- `BOUNDARY`：本轮没有得到可用于比赛宣称的完整官方 candidate 两组结果，也没有同窗口 baseline 对照；4K 结果可作为本轮有效观测，16K 与完整 Level 3 必须标记为异常/未完成，不能用于计算最终官方增益或正式准确率。
- `DECISION`：不修改生产参数、不提交异常调度配置；保留现场日志和部分 Level 3 结果，后续若仍需最终官方数字，应先重启/恢复干净的 9032 candidate，再单独诊断 16K 请求卡死原因后重测。

#### 干净 9032 官方验收收口与负载故障清理（2026-10-05）

- `FACT`：在干净重启的 9032 candidate（API PID `321635`，Engine PID `321777`，GPU 1）上重新运行官方负载 `[[4096,1024,64,256],[16384,1024,64,128]]`；9031 PID `2943` 全程未停止、未重启、未发送请求。
- `FACT`：两组均完成 4/4，所有请求成功。原始 CSV 为 `/workspace/benchmark_results/raw_runs_20261004_154106.csv`，汇总日志为 `/workspace/logs/native_2d_final_accept_20261004/official_clean_candidate/official_clean.log`。
- `FACT`：4K/c64/256 四轮输出吞吐 `372.37/484.71/478.27/503.18 tok/s`，四轮平均约 `459.63 tok/s`；Mean TTFT `6717.94/5746.66/5665.27/5650.48 ms`，后三轮吞吐稳定在 `478.27-503.18 tok/s`。
- `FACT`：16K/c64/128 四轮输出吞吐 `106.88/108.42/108.29/108.37 tok/s`，四轮平均约 `107.99 tok/s`；Mean TTFT `276724.98/273208.38/273518.13/273254.53 ms`，Median ITL 约 `147.38-147.84 ms`。第 1 轮虽明显变慢，但 `128/128` 请求完成，随后三轮正常收口，不能再标记为卡死。
- `FACT`：16K/c64 期间 GPU KV cache 使用率曾达到约 `99%`，请求排队和 chunked prefill 造成高等待时间；这解释了长 TTFT，但不构成服务进程崩溃。
- `FAILURE`：官方负载结束后，残留的评测/benchmark 控制进程使宿主机 load average 一度达到约 `24/33/36`，并导致 `docker exec` 通道阻塞；此时没有活动的 vLLM 服务故障。
- `CHANGE`：通过 `mllv` PID namespace 仅终止 `evalscope`、Level 3 和 `benchmark_throughput_serve` 残留进程，未停止 9031/9032。清理后 1 分钟 load 降至 `3.91`，9031/9032 健康检查均为 HTTP `200`。
- `FACT`：清理后 namespace 内仅保留 9031/9032 两个 vLLM 服务及其 EngineCore；存在约 189 个历史 zombie 子进程，均不占用 CPU，但容器 PID 1 为 `sleep infinity`、不会主动回收它们。当前不为清理 zombie 重启容器，以避免影响 9031。
- `BOUNDARY`：本轮官方 candidate 性能负载已完整完成；Level 3 在清理前未成功重新启动，因此不能把此前的 `101/104` 部分结果当作本轮完整 Level 3 成绩。后续如需正式准确率，只能在当前无残留负载的 9032 上单独启动一次 Level 3，并等待完整退出码。

#### 干净 9032 完整 Level 3 正式结果（2026-10-05）

- `COMMAND`：在干净 9032 candidate 上，仅针对 `minicpm-diag` 运行 `math_500 / Level 3`，共 105 题；9031 未停止、未重启、未发送请求。为修复 namespace 启动时缺失的运行库，使用 `LD_LIBRARY_PATH=/usr/local/corex-4.5.0/lib64:/usr/local/openmpi/lib`。
- `FACT`：105/105 题均完成，评测报告生成成功；Level 3 准确率 `97.14%`（102/105）。评测报告：`/workspace/evalscope-datasets/level3_diag_20261005_120601/20261005_120606/reports/minicpm-diag/math_500.json`；HTML 报告：同目录 `reports/report.html`。
- `FACT`：评测耗时约 `44分52秒`；平均单题延迟 `150.668 s`，平均输出吞吐 `24.47 tok/s`；请求覆盖 `105/105`。
- `FACT`：评测结束后无活动的 Level 3/evalscope/benchmark 进程，9031/9032 健康检查均为 HTTP `200`，tuned marker 存在，dual marker 不存在。
- `NOTE`：外层 `level3_clean.exit` 文件因后台 shell 的转义问题仅写入字符 `n`，不能作为可靠退出码；但 EvalScope 已完成 105/105、生成 JSON/HTML 正式报告且进程已退出，因此本轮结果以正式报告为准。

#### FP8 KV Cache 基线启动与 per-token-head 小负载（2026-10-05）

- `COMMAND`：复用 SSH 控制连接 `~/.ssh/codex-control/ub39-final`，仅启动/操作 9032/GPU 1；9031 未停止、未重启、未发送请求。实验日志目录为 `/workspace/logs/fp8_native2d_20261005/`。
- `FACT`：普通 `--kv-cache-dtype fp8` 启动失败于 vLLM Triton `triton_reshape_and_cache_flash` 的硬件断言：BI-V150 报 compute capability `(7, 1)`，该通用路径要求 native `fp8e4nv (SM89+)`。服务进程已退出；这证明普通 FP8 cache dtype 不能直接作为本设备基线。
- `FACT`：已有仓库实现的 `--kv-cache-dtype fp8_per_token_head` 可以完成模型加载、KV cache 分配和 CUDA Graph capture；服务 9032 健康，日志显示 GPU KV cache size `1,014,224` tokens。启动命令和服务日志位于 `/workspace/logs/fp8_native2d_20261005/per_token_head/service.log`。
- `FACT`：per-token-head 单请求 smoke test 成功返回非空文本；首次请求在 20 秒内超时，但服务保持健康，延长窗口后同一请求成功，说明该超时属于首次编译/预热延迟，不能视为功能失败。
- `FACT`：小负载 benchmark 使用仓库脚本 `benchmarks/benchmark_throughput_serve.py`，固定 `RUNS=4`、`SKIP_FIRST=1`，测试 `[[4096,64,1,8],[16384,64,8,8]]`。两组均 `4/4` 请求成功；原始 CSV 为 `/workspace/vllm-plugin-FL/benchmark_results/raw_runs_20261005_165748.csv`，汇总 CSV 为 `/workspace/vllm-plugin-FL/benchmark_results/summary_20261005_165748.csv`。
- `FACT`：FP8 per-token-head 汇总（后三轮）：4K/c1 总吞吐 `3746.21 tok/s`、Mean TTFT `165.21 ms`；16K/c8 总吞吐 `14951.24 tok/s`、Mean TTFT `5482.71 ms`。16K 首轮单独观测为 `103.27 s`、Mean TTFT `47120.92 ms`，后续三轮约 `6.23-12.01 s`；该首轮被脚本跳过，按预热/冷启动异常单独保留。
- `UNKNOWN`：当前远端 dirty 工作树的首请求日志出现 `Iluvatar native 2D attention enabled`，但普通 FP8/per-token-head dispatcher 的实际 kernel 路由尚未通过独立 marker 或 kernel 级证据确认；不能把上述数字称为“FP8 + Native 2D”最终收益。
- `DECISION`：不修改生产代码、不改变比赛 BF16 路径、不提交或推送；下一步必须在相同服务参数和相同负载下取得 BF16 baseline，再做 FP8 per-token-head 对照。只有对照稳定且确认实际路由后，才评估是否实现 per-token-head cache update 与 Native 2D 的正式兼容。

#### FP8 量化实验：无 prefix-cache 的 BF16 正式对照（2026-10-06）

- `COMMAND`：在 9032/GPU 1 上使用无 `--kv-cache-dtype` 的 BF16/auto 服务，显式设置 `--no-enable-prefix-caching`；9031 未停止、未重启、未发送请求。服务日志：`/workspace/logs/fp8_native2d_20261005/bf16_formal/service.log`。
- `COMMAND`：运行 `benchmarks/benchmark_throughput_serve.py`，测试 `[[4096,64,1,8],[16384,64,8,8]]`，固定 4 轮并跳过首轮。日志：`/workspace/logs/fp8_native2d_20261005/bf16_formal/benchmark.log`；raw：`/workspace/vllm-plugin-FL/benchmark_results/raw_runs_20261006_164723.csv`；summary：`/workspace/vllm-plugin-FL/benchmark_results/summary_20261006_164723.csv`。
- `FACT`：两组均 4/4 请求成功，后三轮无 prefix-cache 稳态结果为：4K/c1 总吞吐 `2869.74 tok/s`、Mean TTFT `513.60 ms`；16K/c8 总吞吐 `3669.52 tok/s`、Mean TTFT `18901.48 ms`、Median ITL `48.18 ms`、P99 ITL `796.30 ms`。
- `FACT`：该 BF16 结果是 FP8 per-token-head 的正式同口径对照；此前启用 prefix-cache 的 BF16/FP8 小负载结果不用于收益计算。
- `BOUNDARY`：FP8 per-token-head 尚未完成无 prefix-cache 同口径 A/B；当前不能宣称量化收益，也不能把此前启用 prefix-cache 的高吞吐数字写入比赛成绩。

#### FP8 量化实验：无 prefix-cache 同口径 A/B 收口（2026-10-06）

- `COMMAND`：停止 BF16 formal 服务后，在 9032/GPU 1 启动 `--kv-cache-dtype fp8_per_token_head --no-enable-prefix-caching`；9031 未停止、未重启、未发送请求。FP8 服务日志：`/workspace/logs/fp8_native2d_20261005/fp8_pth_formal/service.log`。
- `FACT`：服务健康，GPU KV cache size `1,012,064` tokens；日志确认 prefill `max_query_len=2048` 进入 `Iluvatar native 2D attention`，因此本轮是实际 Native 2D + FP8 per-token-head 路径，而非纯功能 smoke test。
- `COMMAND`：运行与 BF16 完全相同的 `[[4096,64,1,8],[16384,64,8,8]]`、4 轮、跳过首轮 benchmark。日志：`/workspace/logs/fp8_native2d_20261005/fp8_pth_formal/benchmark.log`；raw：`/workspace/vllm-plugin-FL/benchmark_results/raw_runs_20261006_170056.csv`；summary：`/workspace/vllm-plugin-FL/benchmark_results/summary_20261006_170056.csv`。
- `FACT`：两组均 4/4 请求成功。4K/c1 稳态总吞吐 `2432.08 tok/s`、Mean TTFT `759.89 ms`；16K/c8 稳态总吞吐 `1893.00 tok/s`、Mean TTFT `37437.92 ms`、Median ITL `54.06 ms`、P99 ITL `1870.83 ms`。
- `COMPARISON`：相对无 prefix-cache BF16 对照（4K `2869.74 tok/s`、TTFT `513.60 ms`；16K `3669.52 tok/s`、TTFT `18901.48 ms`），FP8 per-token-head：4K 吞吐 `-15.25%`、TTFT `+47.94%`；16K 吞吐 `-48.42%`、TTFT `+98.07%`；16K Median ITL `+12.20%`、P99 ITL `+135.00%`。
- `INFERENCE`：当前回归主要说明 per-token-head 的 scale 读取与 FP8 dequant 额外开销已经超过 KV 读带宽节省；这是性能归因假设，尚未用 kernel profiler 分离验证。不能把 FP8 cache 容量增加误写成吞吐收益。
- `DECISION`：现有 FP8 per-token-head 方案不进入生产路径，不修改 Native 2D dispatcher，不提交或推送；后续只有在实现融合 dequant/scale、减少 scale cache 访问并完成 kernel 对照后，才值得继续量化线。当前优先恢复 BF16 tuned candidate。
- `FACT`：实验收口后已停止 FP8 benchmark 和 FP8 服务，恢复标准 BF16 tuned candidate（9032，`minicpm-diag`，Native 2D marker 保留）；恢复后 9032 HTTP `200`，无残留 benchmark。9031 本轮未操作，当前机器上原本未运行。

#### FP8 后续实验边界：先做 kernel 级归因（2026-10-08）

- `FACT`：本地 Native 2D kernel 的 `FP8_PER_TOKEN_HEAD` 分支已经将 K scale 融入 `QK` 分数，将 V scale 融入 `P·V`，不是完全独立的 dequant pass；重复实现“表面 fusion”没有明确技术价值。
- `INFERENCE`：当前服务回归更可能来自每个 KV tile 的两组 scale-cache 向量读取、FP8 到 Q dtype 的转换，以及量化路径触发的编译/调度差异；这是待验证归因，不能直接当作硬件 profiler 结论。
- `PLAN`：下一轮只做 GPU 2 kernel 级消融：BF16 baseline、FP8 per-token-head 实际 scale、FP8 per-token-head 单位 scale/固定 scale，对齐相同 Q/K/V 形状和 warmup；分别记录 CUDA event 时间、数值误差和编译状态。只有确认 scale 读取或 cast 是主开销，才修改 kernel；否则停止当前 FP8 线。
- `BOUNDARY`：截至本记录写入时 SSH 控制 socket 已失效，未启动新的远端 kernel 实验，也未修改生产源码或推送提交。

#### FP8 kernel 首轮消融结果（2026-10-08）

- `COMMAND`：仅在 GPU 2 运行隔离脚本 `experiments/native_2d_tuning/service_and_profiler/kernel_fp8_ablation.py`；9032 未停止、未重启、未发送请求。结果：`/workspace/logs/fp8_native2d_20261008/kernel_fp8_ablation/result.json`。
- `FACT`：同一 Native 2D 配置（`BLOCK_M=128/TILE=16/warps=8/stages=2/pipeline=1/scalar lookup=1`）下，BF16 KV `13.8689 ms`，FP8 per-token-head 实际 scale `27.2030 ms`，FP8 单位 scale `27.2406 ms`。
- `FACT`：FP8 相对 BF16 kernel 约 `1.96x` 慢；实际 scale 与单位 scale 仅相差 `0.04 ms`，说明 scale 数值本身不是主要成本。该首轮尚未分离 BF16 数据上的 scale-cache/scale 运算与 FP8 数据转换成本。
- `BOUNDARY`：当前结果是 kernel 级归因证据，不是服务级收益；尚未修改生产 kernel，也未提交或推送。

#### FP8/INT8 kernel 第二轮消融结果（2026-10-08）

- `COMMAND`：将扩展后的隔离脚本同步至容器，仅在 GPU 2 运行 BF16、BF16 + per-token-head scale、FP8 + scale、INT8 + per-token-head scale、INT8 + 单位 scale 对照；9032 未停止、未重启、未发送请求。结果：`/workspace/logs/fp8_native2d_20261008/kernel_int8_v3/result.json`。
- `FACT`：同一 Native 2D 配置（`BLOCK_M=128/TILE=16/warps=8/stages=2/pipeline=1/scalar lookup=1`）下，BF16 KV `13.7151 ms`，BF16 + scale `14.3163 ms`，FP8 + 实际 scale `27.2558 ms`，FP8 + 单位 scale `27.2553 ms`，INT8 + 实际 scale `14.6149 ms`，INT8 + 单位 scale `14.6029 ms`。
- `COMPARISON`：INT8 相对 BF16 慢约 `6.56%`；BF16 加 scale 的额外开销约 `0.60 ms`，INT8 数据路径相对 BF16 的额外开销约 `0.90 ms`；INT8 实际 scale 与单位 scale 仅相差约 `0.012 ms`。FP8 仍约为 BF16 的 `1.99x` kernel 时间，且实际/单位 scale 几乎无差异。
- `INFERENCE`：当前量化回归主要来自低精度 cache 数据的读取、类型转换和路径布局，而不是 scale 数值计算。INT8 虽比 FP8 好很多，但仍未达到 BF16，不能期待服务级正收益。
- `DECISION`：停止当前 FP8/INT8 量化优化线，不启动 INT8 服务级 A/B，不修改生产 dispatcher、Native 2D kernel 或 KV-cache ABI；实验脚本保留为可复现实验，不提交或推送未经验证的量化实现。继续使用 BF16 tuned candidate 作为正式路径。

#### Decode 路径 GPU 2 基线与精确网格原型（2026-10-08）

- `COMMAND`：新增隔离脚本 `experiments/native_2d_tuning/service_and_profiler/profile_decode_baseline.py`，仅在 GPU 2-5 测试 q_len=1、GQA=8 的 BF16 paged-KV attention；9032 未发送请求时先完成 kernel-only 取证。结果目录：`/workspace/logs/native_decode_20261008/`。
- `FACT`：GPU 2、c64 kernel-only 对照中，4K 上游 `kernel_unified_attention` 中位数 `2.2318 ms`，现有 Native kernel `1.4132 ms`；16K 上游 `8.8344 ms`，Native kernel `5.6022 ms`。Native 相对上游分别下降约 `36.7%/36.6%`，两种形状最大绝对误差均为 `0.0`。
- `FACT`：为验证精确 decode grid，实验性地复用已有 `MIXED_DECODE_ONLY` kernel 分支，增加默认关闭的 `decode_only_override` 和 `decode_block_m_override` 参数，使 grid 使用 `num_seqs`、`BLOCK_M=8/BLOCK_Q=1`。GPU 2 c64 结果：4K `1.0532 ms`、16K `4.3827 ms`，相对上游分别下降约 `51.2%/51.2%`；`BLOCK_M=16` 对应约 `47.9%`/`47.9%`；所有最大绝对误差均为 `0.0`。
- `FACT`：GPU 3/4/5 扩展到 c8/c16/c64 后，`BLOCK_M=8` 相对上游 kernel 下降约 `50.6%-51.2%`，覆盖 4K/16K，最大绝对误差均为 `0.0`。收益不是 c64 特例。
- `CHANGE`：生产 dispatcher 增加环境变量门控 `ILUVATAR_NATIVE_DECODE_TUNE`，默认关闭；只有 BF16、GQA=8、q_len=1、causal 且无可选 attention 特性时才传递 decode 实验参数。未开启该变量时，原生产路由不变。
- `FACT`：9032 已用 `ILUVATAR_NATIVE_DECODE_TUNE=1`、`ILUVATAR_NATIVE_DECODE_BLOCK_M=8` 干净重启，HTTP `200`；小负载 smoke `4096/128/c8` 全部成功，total throughput `9357.4 tok/s`，Median ITL `22.01 ms`，P99 ITL `23.59 ms`。
- `BOUNDARY`：以上是 kernel 级和小负载 smoke 证据，尚未证明官方 4K/16K 服务级收益；正式 candidate A/B 正在 9032 上运行，需与 decode 开关关闭的同口径 control 对照后才能计算增量。

#### Decode-on 官方 candidate A/B 进行中（2026-10-08）

- `COMMAND`：在 9032/GPU 1、保持 `ILUVATAR_NATIVE_2D_TUNE=1`、`ILUVATAR_NATIVE_DECODE_TUNE=1`、`ILUVATAR_NATIVE_DECODE_BLOCK_M=8` 的服务上运行官方负载 `[[4096,1024,64,256],[16384,1024,64,128]]`；9031 未启动、未重启、未发送请求。
- `FACT`：benchmark 父进程 `354613` 与 9032 API 进程 `352731` 检查时仍在运行，9032 `/health` 返回 `200`；当时日志已完成第一组首轮 `256/256`，但整个 2 组 × 4 轮任务尚未结束。
- `FACT`：第一轮观测为 256/256 成功、0 失败、输出吞吐 `336.54 tok/s`、Mean TTFT `11649.80 ms`、Median TTFT `3614.79 ms`、Median ITL `92.40 ms`、P99 ITL `824.20 ms`；这是中间轮次，不作为最终 A/B 结论。
- `FAILURE`：一次后台自动衔接命令因 SSH/bash 引号解析错误提前退出，未停止或重启 9032、未启动 Decode-off；随后远端高负载导致只读 SSH 检查出现超时，未继续发送干预命令。
- `BOUNDARY`：当前只能确认 Decode-on 官方负载仍在自然执行；必须等两组各 4 轮全部成功后，再进行 Decode-off 同口径对照并恢复 Decode-on candidate，尚未形成正式服务级收益。
- `PROGRESS`：后续复核确认 4K/c64 的 4 轮已经全部完成；16K/c64 已进入第 1 轮，benchmark 子进程和 9032 EngineCore 仍在运行。未执行服务切换。

#### Decode 阶段 FlagGems 判断边界修正（2026-10-08）

- `CORRECTION`：W1 profiler 的 `kernel_unified_attention` 约 `92.87%`、FlagGems `mm_kernel` 约 `6.15%` 来自 mixed-prefill 窗口，不能作为官方 decode 主导负载中 GEMM 占比的结论；此前用该比例估算 FlagGems 端到端收益上限不适用 decode 阶段。
- `DECISION`：当前不启动 FlagGems 算子优化实验，先完成 Decode-on 官方 A/B、同口径 Decode-off 对照并恢复 candidate。
- `NEXT`：在独立 decode profiling 窗口采集 `kernel_unified_attention`、`mm_kernel` 的调用次数、Self CUDA 和绝对时间；只有确认 decode GEMM 占比足够高，才在 GPU 2 做 Torch/FlagGems/手写 Triton 小矩阵 GEMM 对照。FlagGems whitelist/blacklist 不作为首轮归因手段。

#### Decode-on 官方 candidate 负载收口（2026-10-08）

- `COMMAND`：在 9032 上完成官方负载 `[[4096,1024,64,256],[16384,1024,64,128]]`，4K/16K 各 4 轮，使用 `ILUVATAR_NATIVE_DECODE_TUNE=1`、`ILUVATAR_NATIVE_DECODE_BLOCK_M=8`。
- `FACT`：8 轮请求全部成功，0 失败。原始 CSV：`/workspace/vllm-plugin-FL/benchmark_results/raw_runs_20261008_124806.csv`；汇总 CSV：`/workspace/vllm-plugin-FL/benchmark_results/summary_20261008_124806.csv`。
- `FACT`：4K/c64 汇总总吞吐 `1963.40 tok/s`，Mean TTFT `11586.35 ms`；16K/c64 汇总总吞吐 `915.64 tok/s`，Mean TTFT `598824.65 ms`。16K 汇总 Median ITL `139.82 ms`、P99 ITL `3293.23 ms`。
- `BOUNDARY`：这是 Decode-on candidate 的绝对结果，不是相对 Decode-off 的正式收益；还不能据此计算 Decode 优化提升。
- `NEXT`：确认 9032 无残留 benchmark 后，使用完全相同负载关闭 `ILUVATAR_NATIVE_DECODE_TUNE` 做 Decode-off control，完成后恢复 Decode-on candidate。

#### Decode kernel-only 并发对照收口（2026-10-08）

- `COMMAND`：在空闲 GPU 2-5 分别运行 `profile_decode_baseline.py`，batch 为 c8/c16/c32/c64，context 为 4K/16K；每个配置 warmup 5 次、CUDA event 计时 20 次。9032 官方 benchmark 已结束；本轮仅使用 GPU 2-5，不更改服务。
- `FACT`：四个结果文件 `/workspace/logs/native_decode_20261008/kernel_ab_c{8,16,32,64}.json` 均生成；所有 8 个形状的 Decode-B8 相对 upstream 和 Decode-B16 的最大绝对误差均为 `0.0`。
- `FACT`：Decode-B8 相对 upstream 的 kernel median 降幅：c8 约 `50.9%/50.9%`（4K/16K），c16 `49.7%/49.6%`，c32 `54.9%/55.2%`，c64 `53.2%/50.3%`。Decode-B8 相对 Native 默认网格也在全部形状更快；幅度约 `14%-35%`，视 batch/context 而异。
- `FACT`：c64 的 upstream/Native-default/Decode-B8 median（ms）为 4K `2.2373/1.4427/1.0482`、16K `8.8295/5.5816/4.3857`；B8 对 Native-default 分别快约 `27.4%/21.4%`。
- `BOUNDARY`：以上只证明 GPU kernel-only 有收益且数值一致，不代表服务吞吐/ITL 收益，也不证明需要新的 Decode-3D kernel。
- `BLOCKER`：服务 A/B 准备前检查发现 9031 `/health` 返回 HTTP `000`，而 9032 为 HTTP `200`；此外节点存在其他 18080/18081 服务及活动 benchmark。本轮未操作这些服务、未启动 9032 A/B。为保证官方基线隔离，需先恢复 9031 或由用户确认允许在 9031 不可用时继续。
- `CHANGE`：本地新增实验编排 `service_and_profiler/native_decode_service_ab.py`，只在 9032 切换 `ILUVATAR_NATIVE_DECODE_TUNE=0/1`，固定 Native 2D prefill 和其他启动参数，完成后恢复 Decode-on；已通过本地 `py_compile`、`git diff --check`，并同步至远端且通过远端 `py_compile`。该脚本尚未执行服务切换。

#### Decode 服务 A/B 首次启动失败（2026-10-08）

- `COMMAND`：在 GPU 2-5 kernel-only 收口后，使用 `native_decode_service_ab.py` 对 9032 做 Decode-off/Decode-on 同口径 A/B；脚本先验证 9032 PID `352731`，停止旧服务，再依次启动 Decode-off 和 Decode-on，并在异常时尝试恢复 Decode-on。
- `FAILURE`：两个变体均未进入 API smoke 或 benchmark。9032 EngineCore 初始化报 `Free memory on device (15.45/32.0 GiB) ... desired ... 27.2 GiB`；Decode-off 和自动恢复 Decode-on 均启动失败。9032 当前无监听，不能把本轮写成 A/B 结果。
- `FACT`：GPU 1 的剩余显存被同一容器内另一个独立服务 `18080` 占用；该服务 PID `357893`，环境为 `CUDA_VISIBLE_DEVICES=1`，约占 `16.8 GiB`。另有 GPU 2/端口 `18081` 服务 PID `359873`，不属于本轮 9032 A/B。
- `BOUNDARY`：本轮没有杀死 `18080/18081`，也没有降低 `gpu-memory-utilization` 伪造对照；9031 本来就未运行。服务 A/B 需要先释放 GPU 1 上的 18080，再用完整显存配置恢复 9032，随后重新执行一次唯一编排。
- `NEXT`：释放/确认 GPU 1 空闲后，先恢复标准 Decode-on 9032 并验证健康，再重跑 Decode-off → Decode-on A/B；此前 kernel-only c8/c16/c32/c64 结果保持有效，不需重跑。

#### Decode 服务 A/B 转移至 GPU 4（2026-10-08）

- `CHANGE`：将 `service_and_profiler/native_decode_service_ab.py` 参数化为显式 `--gpu`、`--port`、`--served-model-name`；可选验证/接管指定 PID，或仅在空端口启动临时服务，并可选在实验后保留 Decode-on 服务。CSV 数值解析兼容 `16.0`。本地 `py_compile`、`--help`、`git diff --check` 通过；脚本同步至容器并通过远端 `py_compile`。
- `FACT`：SSH 恢复后检查容器内无 9032/9034 服务；既有 18080/18081 服务仍运行。本轮未操作这些服务、9031 或 9032。
- `FACT`：容器中设备工具为 `ixsmi`；本次调用超时，未取得实时显存表。因此启动前没有新的 GPU 4 显存读数。选择 GPU 4 是基于此前 GPU 2-15 可用的节点盘点及当前无已知 GPU 4 服务；引擎随后成功加载并初始化，侧面确认该卡足以启动配置，但不代表获得了精确显存占用数据。
- `COMMAND`：启动单一 A/B 编排器：`python3 -u /workspace/vllm-plugin-FL/experiments/native_2d_tuning/service_and_profiler/native_decode_service_ab.py --gpu 4 --port 9034 --served-model-name minicpm-decode-gpu4 --restore-decode-on --log-dir /workspace/logs/native_decode_service_ab_gpu4_20261008`。Runner PID `365039`，Decode-off vLLM PID `365041`。
- `FACT`：Decode-off 服务已完成 CUDA graph capture，9034 health HTTP `200`，8-token API smoke 成功；官方 benchmark 父进程 PID `365791` 已启动。Decode-off 官方负载仍在运行，CSV/summary 尚未收口，当前无 A/B 性能结论。
- `FAILURE`：随后一次远端只读状态查询因 SSH 超时结束；没有向远端发出停止、重启或重复启动命令。最近一次可确认状态仍是上述 Decode-off benchmark 运行中；需要下一次连接后先查既有 runner 和日志，再决定是否继续等待。
- `FAILURE`：2026-10-08 后续只读状态检查再次因 SSH 超时结束；没有远程副作用。远端当前 runner/benchmark 状态无法由本地确认，下一步应在 SSH 通道恢复后只读检查原日志目录和 PID，不得重复启动 A/B。
- `FACT`：用户在 `root@ub39` 终端于上述超时后回报：runner PID `365039` 和 GPU 4/9034 Decode-off 服务 PID `365041` 仍在，benchmark PID `365791` 已运行 `10:31`；9034 health 为 `200`。Decode-off benchmark 日志仅见 `4096_1024_c64 | Run 1/4`，尚无 Successful requests 行、CSV 或结果文件。该状态表示首个 4K 测量仍在运行或未输出完成标记，不能据此判断卡死；本地 Codex 再次只读 SSH 检查也超时，未对远端作任何修改。
- `BOUNDARY`：实验服务固定使用 `gpu-memory-utilization=0.85`、`max-model-len=131072`、`FULL_DECODE_ONLY`；不改变官方基线或其他服务。编排器完成 Decode-off 后会在同一 GPU/端口启动 Decode-on 并执行相同负载，最后保留 Decode-on 临时服务。
- `NEXT`：等待唯一编排器完成两种配置各 8 轮，检查结果 JSON、原始/汇总 CSV、每轮请求成功数及恢复服务 health，再计算服务级差异。不得重复启动第二个 runner。

#### GPU 4 Decode 服务 A/B 汇总取证（2026-10-09）

- `SOURCE`：用户提供远端 `runner.log` 末尾与 `results.json` 原文，本地附件为 `/Users/mahanting/.codex/attachments/5677bf4d-3638-4e8c-bc97-b2bf3977443d/已粘贴的文本.txt`。本轮通过 JSON 解析计算增量，不将用户回传误写为 Codex 直接远端核验。
- `FACT`：日志依次记录 Decode-off PID `365041`、Decode-on PID `371450`，两者 API smoke HTTP `200`，均输出 4K/16K 汇总；随后记录 `RESTORED decode_on port=9034 pid=374465`。编排已经进入正常恢复尾声，但恢复后的当前 health 和 runner 退出状态尚未独立复核。
- `FACT`：4K/c64 总吞吐 off/on 为 `1966.85/2008.67 tok/s`，增量 `+2.1262%`；输出吞吐 `393.37/401.73 tok/s`，增量 `+2.1252%`。Mean TTFT `11618.29/11605.56 ms`（`-0.1096%`），Median ITL `92.18/91.88 ms`（`-0.3255%`），P99 ITL `758.24/754.89 ms`（`-0.4418%`）。
- `FACT`：16K/c64 总吞吐 off/on 为 `915.56/915.36 tok/s`，增量 `-0.0218%`；输出吞吐 `53.86/53.84 tok/s`，增量 `-0.0371%`。Mean TTFT `598978.24/598996.73 ms`，Median ITL `139.77/139.86 ms`，P99 ITL `3307.09/3266.77 ms`（`-1.2192%`）。两组平均 TTFT 均约 599 秒，该长等待现象没有随 Decode 开关消失。
- `BOUNDARY`：比较对象是固定 Native 2D prefill 后的 Decode-off/on，不是比赛原始 baseline。日志出现 SUMMARY 与本地脚本“读取并校验 raw 后输出 SUMMARY”的流程一致，但本轮尚未直接取得 raw CSV、轮间波动、远端脚本哈希、实际 dispatch 命中记录和 Level 3；不能把以上写成新的正式比赛收益或准确率验收。
- `INFERENCE`：4K 观察到约 2.13% 的服务吞吐增量，稳定性仍需 raw 轮次支持；16K 未观察到服务吞吐收益。此前 kernel-only 约 50% 的耗时降幅不能外推为端到端收益。
- `DECISION`：保持 Decode 优化为实验性 opt-in，不默认启用、不据此推送新的正式性能声明。本轮没有停止、重启或新增远端负载。
- `NEXT`：优先收集 raw/summary、恢复服务 health、实际路由及 GPU 隔离证据；诊断 16K 长 TTFT（检查 KV 容量、preemption、调度和纯 decode profile，均为待验证原因），然后决定是否需要隔离复测和 Level 3。当前尚未完成全部最终验收。

#### GPU 4 A/B 独立复核与短时排队诊断（2026-10-09）

- `COMMAND`：通过现有 SSH multiplex 控制连接 `~/.ssh/codex-control/ub39-final-v2` 读取容器进程、9034 health、runner.log、服务环境和 `/metrics`；将完整 A/B 目录通过 tar 拉取至本地 `/tmp/native_decode_service_ab_gpu4_20261008/`，逐行解析两份 raw CSV。
- `FACT`：独立核验时原 A/B runner 与 benchmark 均已退出；恢复后的服务 PID 为 `374465`，GPU 为 `4`，9034 health 为 `200`。环境包含 `ILUVATAR_NATIVE_2D_TUNE=1`、`ILUVATAR_NATIVE_DECODE_TUNE=1`、`ILUVATAR_NATIVE_DECODE_BLOCK_M=8`。未操作 9031、9032 或 18080。
- `FACT`：Decode-off/on 各 8 行 raw，全部 `Run Status=SUCCESS`，Successful Requests 与 Num Prompts 一致。4K 四轮总吞吐 off 为 `[1704.45,1959.64,1914.21,2026.69]`，on 为 `[1622.29,1979.44,2017.62,2028.95]` tok/s；去首轮后三轮样本 CV 分别为 `2.877%/1.291%`。16K 四轮 off 为 `[910.20,915.43,915.78,915.46]`，on 为 `[908.06,915.25,915.51,915.31]` tok/s，后三轮 CV 为 `0.0212%/0.0149%`。
- `INFERENCE`：4K 约 `+2.13%` 的均值增量与轮间噪声处于相近量级，单次顺序 A/B 不足以证明稳定提升；16K 约 `-0.02%` 维持无服务收益判断。CV 比较不是显著性检验，不能据此给出置信区间或推广结论。
- `FACT`：Decode-on 日志中的 KV cache 容量为 `521,840 tokens`（修正早先约 `521,936` 的粗略值）。`Maximum concurrency ... 3.98x` 是以 `max-model-len=131072` 计算，不能解读为 16K 仅能容纳 4 个请求。16K 运行日志多次出现 `Running:31`、`Waiting:33`、KV usage `100.0%`。
- `INFERENCE`：仅按 token 容量估算，`521840/(16384+1024)≈29.98` 个完整长度序列；这是容量估算而非调度器 admission 规则。排队与容量压力已有直接日志证据，但抢占、重算和各阶段绝对耗时尚未获得直接量化，不能把长 TTFT 全部归因于 KV 饱和。
- `BOUNDARY`：A/B 服务日志未检索到明确的 `Iluvatar native 2D attention enabled` 命中行；进程环境与 dispatcher 源码开关存在不等于实际执行命中。仍需独立 profile/路由取证，不能将目前结果描述为已经完整验证 Decode kernel 接入收益。
- `CHANGE`：新增隔离诊断脚本 `experiments/native_2d_tuning/service_and_profiler/diagnose_decode_queue.py`。仅使用已有 GPU 4/9034 服务；核对显式 PID、GPU、端口与空闲指标，使用端口锁防止本脚本重复编排；不重启服务、不改调度配置、不改生产算子。每 2 秒保存 Prometheus 指标，记录 running/waiting、KV usage、preemptions、prefix hits 和原始 benchmark JSON；每个子任务限定 1800 秒，失败即停止后续任务，只终止自己创建的 benchmark 进程组。
- `VERIFICATION`：本地 `py_compile` 通过。远端通过 Prometheus 解析、model 标签过滤、缺失指标拒绝、PID/GPU/端口校验测试；启动前 running/waiting 均为 0，preemptions 与 prefix hits 均为 0。仅对新脚本做验证，未覆盖既有生产算子全部测试。
- `COMMAND`：以唯一后台任务启动 `python3 -u /workspace/vllm-plugin-FL/experiments/native_2d_tuning/service_and_profiler/diagnose_decode_queue.py --expected-pid 374465 --log-dir /workspace/logs/native_decode_queue_20261009_1022`；runner 日志为同路径加 `.runner.log`。
- `BOUNDARY`：本轮 c8/c24/c32/c64 均为输入 16384、输出 128、prompts=concurrency，使用不同随机 seed 并检查 prefix hit 增量。各 case 总工作量不同，且每组仅 1 轮，只用于排队/容量归因，不作为官方吞吐 A/B。保留当前 prefix caching 设置，不将其误写为已关闭；若有 prefix hits，必须标注混杂。
- `NEXT`：检查唯一诊断 runner 的退出标记、逐组结果和服务健康，再决定是否需要独立纯 decode profile 与路由命中核验；尚未启动新的 Level 3 或官方完整负载，不提交、不推送新的性能声明。
- `PROGRESS`：启动后独立检查确认唯一 runner PID `375252`，首组 c8 benchmark PID `375258` 已进入 main benchmark；9034 health `200`，尚无完整 c8 结果。其余 c24/c32/c64 在同一 runner 中串行排队，无第二个负载编排器。
- `FAILURE`：一次包含多个只读源码查询的 SSH 命令到 30 秒上限后退出 `142`，无服务变更；随后缩小为单个文件查询成功。未因超时重启服务或重复启动实验。此次本地 `git diff --check` 通过。
- `PROGRESS`：2026-10-09 10:26:50 CST 只读核验，c8 已完成、c24 正在运行，c32/c64 尚未开始；唯一 runner PID 仍为 `375252`，9034 health `200`。未插入新负载或改动服务。
- `ACCEPTANCE`：按用户要求，待四组全部完成后再给路由、容量/排队/抢占、时延分布与上轮逐轮 A/B 四类汇总。当前实际指标名为 `vllm:kv_cache_usage_perc`，不是旧版 `gpu_cache_usage_perc`；抢占报告每个 case 的 counter 增量，避免累计值混淆。短测不足 5 分钟时报告实际区间、全程峰值与时间分布，不伪称最后 5 分钟稳态。
- `BOUNDARY`：保存的 benchmark JSON 有逐请求 `ttfts`、`itls`，没有原生逐请求 `latencies`。可据这些数组计算 P95，但若报告 `TTFT+sum(ITL)` 必须注明为“到末 token 的重建时延”，不能冒充原生完整 E2E。KV 高占用或抢占次数阈值不能单独证明重算主导或 kernel 无效；当前 B8/BLOCK_Q=1 已实现，Decode-3D 未启用，不将它作为本轮尾延迟的归因对象。
- `FOLLOWUP`：按用户等待完成后汇总的要求，创建本对话单一自动跟进 `minicpm-decode`，每 15 分钟只读检查现有任务；运行中或无实质变化时不通知，不启动新实验。完成/故障时通知并暂停该跟进。未提交或推送代码。
- `HEARTBEAT`（2026-10-09 10:44:59 CST）：经现有 SSH socket 执行单次 20 秒限时只读检查，读取 runner.log、runner.exit 存在性、results.json 文件状态、PID `375252` 与 9034 health。c8/c24/c32 已完成，c64 正在运行，runner.exit 尚未生成，9034 HTTP `200`。日志报告三组峰值 KV usage 分别 `25.22%/53.37%/53.37%`、峰值 waiting `7/23/31`，各组 preemptions delta 均 `0`；这些是中间日志摘要，尚未核验完整采样、prefix hits 与时延分布，不外推到输出 1024 的官方负载。未重启、停止或新增负载，未改生产代码；任务仍正常运行且无需人工行动，本轮不通知。

#### 16K 短输出并发诊断最终汇总（2026-10-09 10:59 CST）

- `COMMAND`：经既有 SSH socket 单次 20 秒限时只读检查 runner.log、runner.exit、results.json、原 PID 和 health；通过 tar 下载已完成目录至 `/tmp/native_decode_queue_20261009_1022/`，使用 JSON/CSV 解析器汇总原始逐请求时延和 Prometheus 采样。另只读检查恢复服务日志中的 route marker。未运行新 benchmark，未操作 9031、9032、18080 或其他服务。
- `FACT`：四组全部完成，runner.exit=`0`，原 runner PID `375252` 已退出，9034 health=`200`。c8/c24/c32/c64 分别成功 `8/24/32/64` 请求，全部零失败，所有请求输出 128 token、每请求保存 127 个 ITL；各组指标读取错误为 0。远端目录 `/workspace/logs/native_decode_queue_20261009_1022/`。
- `ROUTING`：服务环境 decode 开关为 1，启动时保存的 attention.py 哈希可追溯，但恢复服务启动及本轮请求日志中的 `native 2D attention enabled` / `native decode attention enabled` marker 数为 0。当前实际 Native Decode 路由命中仍未证实；不能把缺少日志当成未命中证据，更不能据此宣称修改 metadata 能解决问题。

容量与排队（峰值为全程采样峰值，running/waiting 峰值不要求同时发生）：

| c | 采样区间秒 | 采样数 | 峰值 KV | 峰值 running | 峰值 waiting | 抢占增量 | prefix 命中增量 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8 | 145.7 | 72 | 25.22% | 8 | 7 | 0 | 0 |
| 24 | 392.1 | 192 | 53.37% | 17 | 23 | 0 | 0 |
| 32 | 509.3 | 249 | 53.37% | 17 | 31 | 0 | 0 |
| 64 | 952.7 | 465 | 53.37% | 17 | 63 | 0 | 0 |

- `FACT`：prefix queries 增量分别为 `131072/393216/524288/1048576`，prefix hits 增量均为 0，未观察到前缀复用污染。`enable_prefix_caching=True` 保持不变，不能写成关闭了 prefix cache。
- `FACT`：最后 300 秒窗口（c8 仅有 145.7 秒实际区间）样本均值 KV 为 `11.17%/42.07%/49.99%/50.86%`，running 为 `3.79/13.69/16.20/16.50`，waiting 为 `2.75/7.81/7.75/9.24`。这些窗口包含请求入场、退场和末尾空闲采样，不是严格稳态；全程没有 KV>=99% 的采样。

时延与吞吐（时延单位秒，ITL 汇总所有成功请求的 token 间隔；分位数采用排序线性插值，与 numpy percentile 默认方法一致）：

| c | TTFT median/P95/P99 | ITL median/P95/P99 | 重建到末 token 时延 median/P95/P99 | 总 tok/s |
|---|---|---|---|---:|
| 8 | 71.136/116.478/121.019 | 0.0445/2.7443/3.8053 | 126.534/127.202/127.250 | 1037.89 |
| 24 | 175.682/326.874/351.128 | 1.0588/3.1274/3.6066 | 368.464/372.240/372.411 | 1063.89 |
| 32 | 232.582/437.025/465.067 | 1.3245/3.0740/3.6274 | 464.987/488.533/488.770 | 1080.84 |
| 64 | 466.388/874.250/910.643 | 1.5066/3.3212/3.8209 | 692.408/925.913/926.437 | 1140.39 |

- `BOUNDARY`：原始 benchmark JSON 不提供逐请求原生完整 E2E，本表只能使用 `TTFT + sum(ITL)` 重建到最后 token 的时间，不包含可独立核验的后续协议完成开销。num_prompts 随并发变化、每组只有一轮、输出仅 128，不能将总 tok/s 差异写成官方性能改善；本诊断也未实现与 decode-off 的同 shape 对照。

上轮 GPU4/9034 的 4K/c64 Decode-off/on 逐轮对照（吞吐 tok/s，ITL ms；首轮为 warmup 丢弃，不将顺序轮次视为随机配对实验）：

| Run | off 吞吐 | on 吞吐 | 描述性差值 | off/on median ITL | off/on P99 ITL |
|---|---:|---:|---:|---|---|
| 1（丢弃） | 1704.45 | 1622.29 | -4.82% | 92.02/91.76 | 833.04/856.64 |
| 2 | 1959.64 | 1979.44 | +1.01% | 92.25/92.19 | 773.58/762.45 |
| 3 | 1914.21 | 2017.62 | +5.40% | 91.96/91.56 | 752.23/747.34 |
| 4 | 2026.69 | 2028.95 | +0.11% | 92.33/91.90 | 748.91/754.87 |

- `INFERENCE`：4K 稳态三轮的描述性吞吐差异全部为正但幅度为 `+1.01/+5.40/+0.11%`，不足以认定稳定 +2.13%；第4轮 P99 ITL 约恶化 0.8%，不是全部轮次稳定改善，也不能归因于未启用的 Decode-3D。顺序 A/B、无路由证据和仅3轮稳态仍是主要不确定性。
- `INFERENCE`：短输出 c24/c32/c64 即使 KV 峰值仅 53.37%、零抢占，TTFT 和等待仍显著增加，因此“当前短测由显存爆满/持续抢占重算主导”的解释不成立。旧官方输出1024负载曾满 KV，两者并不矛盾：较长输出允许更多同时存活序列累积 KV。当前数据不能定量解释旧负载重算成本，也不能证明 decode kernel 本身没有收益。
- `NEXT`：优先独立确认实际加载的 backend/dispatcher 与纯 decode kernel 命中，再在纯 decode 与 mixed 窗口区分每步耗时、调度/prefill 阻塞、attention 和 GEMM 时间；只有新证据支持时才设计后续优化。此处仅给后续建议，未启动 profiler、服务切换、Level 3、新算子或新负载。
- `FOLLOWUP`：本诊断任务完成，通知用户四类结果后停止本次自动轮询；不提交不推送，不清理或停止仍健康的服务。

#### 对后续 review 的证据校正（2026-10-09）

- `SOURCE`：用户附件 `/Users/mahanting/.codex/attachments/14bd60c9-b15a-4fb3-8d75-b1f63ffdd6a9/已粘贴的文本.txt`。本轮核对本地 dispatcher/kernel、历史实验记录与远端已安装 scheduler 源码，仅执行限时只读 SSH；未启动任何 profiler、新负载或服务重启。
- `FACT`：`_NATIVE_2D_ROUTE_LOGGED` 为共享的一次性日志闩锁，首次 native 调用后不会为 decode 再打印。若先记录 prefill，则后续没有 decode=True 不能用于否定 decode 命中。但首次调用“必然是实际4096/16384 prefill”不成立：启动 warmup/CUDA graph capture 也可能先调用 attention。此前恢复服务整个日志连 native prefill marker 也没有，不能仅凭闩锁解释全部缺失；日志缓存、日志范围、实际 backend 和执行路径仍未独立证实。
- `FACT`：本轮再次读取 9034 API PID `374465` 与 EngineCore PID `374606` 的实际环境，两者均为 GPU4、Native2D=1、NativeDecode=1、DecodeBlockM=8。因此“9034 忘开 decode 开关”已被本轮活进程证据排除，不因提交 README 默认关闭而盲目重跑 A/B。当前环境不替代上一轮所有退出进程的完整路由证据。
- `CORRECTION`：附件称 dual-launch “从未GPU编译验证”引用了历史准备状态（1178行），忽略后续记录。2026-09-29/30 已完成 GPU编译、fp32对拍、正反序三seed复测、分项profiler、真实dispatch与9032服务A/B。四组服务吞吐增量依次 `-2.89%/-4.88%/+1.23%/-3.95%`，当前设计已否决；只能有新机制/新证据时重新考虑，不能把既有负面结果当未实验。
- `FACT`：附件容量推算使用 `523056` tokens，但本轮服务实际为 `521840`；不同历史池容量不能混用。短输出四组确实零抢占、峰值KV<=53.37%，不支持本短测由抢占/满池主导。官方输出1024的旧服务日志则出现 running31和KV100%，因此不能推广“16K容量根本不是限制项”或“固定17的全局硬上限”。
- `INFERENCE`：running17 也可能是动态流水线平台，而非配置上限：粗略每2048 tokens预算完成一个16K prefill需至少约8步，已有decode占用预算会使准入更慢；输出128的请求在约128 decode步后退出，`128/8≈16` 的同时在场规模与观察17相近。此为待验证调度假说，不是实测step统计或根因结论。实际scheduler先排running、再在剩余token_budget内排waiting，并有max_num_seqs/KV分配等闸门；只有读取实际配置及逐步预算/准入/完成事件才能区分。
- `CORRECTION`：存在waiting不意味着每个step一定排到prefill。资源分配失败、序列数上限、token预算和已有running工作均可能阻止waiting准入。旧9034服务日志存在437个非零running/waiting且prompt吞吐显示0的10秒窗口；聚合日志并不能证明这些窗口每步均纯decode，但足以要求实际step统计，不能高置信认定c64纯decode机会几乎不存在。
- `BOUNDARY`：4K median ITL变化小不足以证明decode计算路径未变化，ITL包含整模型、调度/重叠/传输且分位数不保持阶段可加性；三稳态轮未提供可靠收益归因或统计显著性。应写“尚未证实稳定服务增益”，而不是FACT“全是噪声/路径未变化”。kernel名称可区分native/upstream，但native同一kernel名称也用于mixed，因此还需step类型与launch/constexpr关联，不能仅以名称证明B8 pure-decode命中。
- `NEXT`：维持最小诊断顺序：当前开关核验已完成；下一轮先确认实际加载的backend与代码来源、关联metadata/step类别及CUDA graph capture/replay，再在可识别的纯decode/mixed窗口取设备trace。暂不修改allowlist、调kernel参数、恢复旧dual方案或上新GEMM线。本轮未提交未推送。

#### 实际路由与 Graph 取证第一轮（2026-10-09）

- `CHANGE`：新增隔离实验工具 `service_and_profiler/route_probe.py` 与 `run_route_diagnostic.py`；只复制远端工作树到新的 `/workspace/native_route_probe_20261009_1145`，向副本注入 backend 返回路径、attention 路由与 model runner 每步分类/graph模式探针，不改主工作树生产文件。端口锁、新目录要求、显式PID/GPU/空闲指标保护；失败或成功均尝试恢复主工作树 GPU4/9034 Decode-on 服务。
- `VERIFICATION`：本地及远端隔离注入AST校验、重复注入拒绝测试通过；本地模拟step分类、去重与路由选择测试通过，`py_compile` 和 `git diff --check` 通过。首次重复注入测试暴露未拒绝二次注入，已加预先检测修复。一次本地文件读取因工作目录下重复前缀失败，后续改为正确相对路径；一次SSH命令漏写目标导致客户端退出255，修正目标后成功，均无远端服务副作用。
- `COMMAND`：唯一编排 `run_route_diagnostic.py --expected-pid 374465 --checkout /workspace/native_route_probe_20261009_1145 --log-dir /workspace/logs/native_route_probe_20261009_1145`，runner PID `376777`，诊断API PID `376785`，EngineCore PID `377061`。仅停止已确认空闲的9034实验服务；9031/9032/18080均不动。
- `FACT`：隔离服务backend选择与真实step metadata路径均为本副本 `IluvatarAttentionBackend/IluvatarAttentionMetadata`；decode开关仍为1。实际scheduler config为 `max_num_seqs=256`、`max_num_batched_tokens=2048`、`enable_chunked_prefill=True`、partial/long_partial均1，因此不存在 `max_num_seqs=17` 的配置硬上限。
- `FACT`：graph capture中35个不同batch的路由记录为 `decode_only=True`、`native_supported=False`、`native=False`。真实prefill/mixed host路由也全部 `supported=False/native=False`，说明本轮不是“未加载vendor backend”，而是在vendor内部门控回退。此处尚未确认哪个guard失败，不能提前修改生产allowlist。
- `FACT`：两组诊断 `16384 input /128 output /c8 /8prompts` 和 `256 input /512 output /c8 /8prompts` 均8/8成功、0失败。全程（含API smoke和benchmark单请求test）有639个q1/FULL、58个mixed/NONE和10个prefill/NONE step记录；这是host执行入口计数，不是每层attention次数或设备kernel次数，q1仅按本step调度token数分类。
- `FACT`：mixed触发profile后设备trace实际包含连续8个mixed/NONE标记、336次上游 `kernel_unified_attention`，耗时13.659668秒/总kernel14.707141秒=`92.88%`；`mm_kernel`1344次、0.898979秒=`6.11%`。本窗口确实执行上游attention，没有native命名kernel，比例仅适用于这个mixed窗口。
- `FACT`：纯decode触发条件为最新q1 step且8个请求均每次调度1token；trace连续8个q1/FULL标记及8个 `cudaGraphLaunch`，但graph内attention/GEMM kernel没有出现在当前设备kernel事件中。仅可见graph外采样等kernel，不得将所见采样占比58.9%或线性32.0%误作模型decode阶段占比。这是profiler可见性缺口，capture时host路由和设备replay需分开解释。
- `FACT`：第一轮runner.exit=0，两组完成后恢复主工作树Decode-on服务PID `383781`，API smoke200；第一轮目录已下载到 `/tmp/native_route_probe_20261009_1145/`。
- `INFERENCE`：生产 `forward` 的BF16/auto分支传入 `layer._k_scale.expand(...)` 与 `layer._v_scale.expand(...)`，而现有门控要求k/v_descale为None，存在明显caller/guard不一致。合成guard测试复现仅因这两项回退，但仍需真实请求逐项记录，不能仅凭合成测试作为修复验收。
- `COMMAND`：在第一轮已正常退出并恢复之后，启动单一 `--smoke-only` 逐项guard复核；新隔离目录 `/workspace/native_route_guard_20261009_1152`，日志 `/workspace/logs/native_route_guard_20261009_1152`，以恢复PID `383781` 为验证目标。探针逐项执行与原门控相同的纯host表达式，不读GPU张量数值、不改变门控；仅API smoke，无官方负载或新优化方向，结束后再恢复标准9034。当前尚待最终guard结果/恢复确认。
- `FAILURE`：第二轮启动后，汇总events的SSH只读查询和随后缩小为仅tail runner.log的查询均在20秒限时退出142，无回执。本轮停止进一步远端请求，未重复启动、停止或重启服务。第一轮恢复已证实，第二轮恢复尚未独立证实；下一次必须先核对既有guard runner.exit/失败文件/恢复PID与health，再读取guards，不得启动第三轮或把源码推断写成真实拒绝项验收。
- `2026-10-09 CONTINUE`：本地复核 `attention.py` 与 Native kernel 参数路径。确认 `kv_cache_dtype=auto` 的非 per-token-head 分支当前无条件传入 `layer._k_scale.expand(descale_shape)` / `layer._v_scale.expand(descale_shape)`；而 `_run_unified_attention` 的 Native 支持门控要求 `k_descale is None` 且 `v_descale is None`。这解释了合成 guard 中的拒绝项，但仍不替代远端 `native_route_guard_20260910_1152` 的真实逐项事件。
- `2026-10-09 FAILURE`：尝试通过既有 SSH control socket 进行单次 20 秒只读检查时，Codex 远端执行层返回 `403 permission_denied: 没有资产或资产未激活`，未执行远端命令；未启动、停止或重启任何服务，也未重复 guard runner。socket 本地状态仍显示 `Master running`，但当前任务无法读取远端 guard 结果。下一步应由已连接终端执行既有目录的只读核验，确认 `runner.exit`、恢复 PID 与 9034 health 后再做隔离修复。
- `2026-10-09 SSH诊断`：用户复现同一 `403`。本机 `ssh -G` 未发现 `ProxyCommand`，控制 master PID `52997` 为普通 OpenSSH `ssh -M` 进程且仍在运行；因此问题表现为当前控制会话/远程资产认证状态失效，不是 mllv 或 9034 服务故障。建议仅退出本地旧 master、用新 socket 重新认证，不触碰远端服务。

#### 阶段收口与双仓库交接（2026-10-09）

- `REQUEST`：用户确认旧机器已不能使用，下一阶段将更换环境；要求正式算法/算子放在 `AlexMa616/vllm-plugin-FL` 的 `jl2026-native-2d`，脚本、完整记录、证据和总结放在 `AlexMa616/minicpm-attention-optimization-iluvatar`。
- `COMMAND`：`git ls-remote --heads` 核对两仓库：正式分支为 `02badf74e8a10aabc6f401407608c097ca30b163`，实验仓库main为 `a21f22740d2e3a32f28c2be031e2b287723c84fe`。本地开发仍为 `fp8-kv-native2d-experiment` / `edcb128`，两处未提交源码修改与6个新实验脚本保持不动。
- `FACT`：远程正式分支仍有独立 `experiments/native_2d_tuning/` 和 `tools/cross_ab_c16_clean.py/cross_ab_matrix.py`；不能称旧树完全没有实验代码。实验仓库旧README仍以Split-KV为active，已加最新入口及历史状态警示，不追溯改写旧记录。
- `CHANGE`：克隆实验仓库到本地同名目录，归档开发工作区所有Native2D实验工具、外层legacy tools、两份交叉A/B工具，以及未提交Decode源码git diff。新建 `docs/stage-summary-2026-10-09.md`，覆盖算法、实验时间线、官方candidate绝对值、完整Level3 102/105=97.14%、Decode证据、未决路由问题和新环境验收顺序。
- `CHANGE`：把本地已下载的GPU4 A/B、queue诊断和第一轮route probe从 `/tmp` 备份到实验仓库 `evidence/`；不复制vllm_cache、模型权重或凭据。56个证据文件逐字节与源文件相等，JSON/JSONL和压缩trace解析通过，生成SHA256清单。20个Native2D源码/脚本与原工作区逐字节相等（README另外添加归档说明）。
- `COMMAND`：从远程正式分支新建隔离clone准备清理，仅删除已备份的独立实验目录和两份交叉A/B脚本，新增 `docs/jl2026-native-2d-status.md` 链接交接资料。`git diff --exit-code -- vllm_fl benchmarks tools/native_2d_harness.py tools/start_native_2d_service.sh` 通过，证明算子、官方benchmark、必要harness和启动工具不变；`git diff --check` 通过。
- `FAILURE`：首个父目录git检查因父目录不是git仓库失败，改用明确仓库路径；一次含清理临时目录的命令被执行策略拒绝，改用新建临时目录，未执行清理；首个文档patch同时delete/add同文件被拒绝，改用update后成功。无远端实验副作用。
- `BOUNDARY`：最后 `native_route_guard_20261009_1152` 的真实逐项guard结果及恢复状态未取回；k/v_descale仍是源码/合成测试支持的候选原因，不是已验收修复。其他历史CSV与Level3报告并非都已下载；只保留记录中的远端路径，不能宣称本轮取回原件。
- `DECISION`：本阶段只归档、文档和代码仓库职责整理，不修改生产算子、不应用未验证Decode或scale修复、不启动新的服务器实验。提交/push结果以本轮后续核验为准，不提前宣称上传成功。
- `NEXT`：新机器拿到后先建立固定版本环境与空闲资源盘点，然后做真实guard/scale契约隔离复核、正确性、Graph重capture、短交叉A/B、完整官方负载和Level3；稳定收益通过后才提交源码修复。
- `VERIFICATION`：56个原始证据文件SHA256复核通过，均已进入待提交集合，合计5,136,649字节；正式分支迁出的13个文件中，12个源码/脚本与实验仓库备份逐字节相同，README保留原说明并增加历史归档警示。归档Python语法与shell语法检查通过；正式树无引用已迁出工具的残留。
- `FAILURE/BOUNDARY`：实验仓库全量 `git diff --cached --check` 报告原始日志、profiler及patch上下文的历史空白字符；这些是证据原件，不修改。排除 `evidence/**` 与原始patch后的文档/脚本空白检查通过；没有以清理空白为由改写raw证据。
- `COMMAND`：`gh auth status` 确认AlexMa616已登录；再次有界执行两仓库 `git ls-remote --heads`，远端仍为上述锚点，没有发现并发更新。先上传实验归档，再上传不改变生产行为的正式树清理。
- `RESULT`：实验仓库首批归档提交 `d37ebb7`，普通 `git push origin main` 成功（`a21f227..d37ebb7`）；正式分支收口提交 `0d686d5`，普通 `git push origin jl2026-native-2d` 成功（`02badf7..0d686d5`）。没有force push，没有修改原开发工作区的两处源码和6个新脚本。
- `BOUNDARY`：本轮验收覆盖归档完整性、语法、文档和正式源码不变；没有可用GPU环境，未复测算子、服务性能或准确率。已上传的补丁仍标记未验证，历史Level3不等于新环境验收。
