# 已保存原始证据

截至2026-10-09。以下目录从旧机器下载到本地后复制保存，未重新运行实验。
JSON/CSV/log/trace保持原始字节，SHA256见 `SHA256SUMS`。

| 目录 | 内容 |
|---|---|
| `native_decode_service_ab_gpu4_20261008/` | Decode-off/on各8行raw、summary、服务/benchmark日志、results与恢复PID |
| `native_decode_queue_20261009_1022/` | c8/c24/c32/c64逐请求JSON、metrics.jsonl、命令、metadata、results与runner.exit |
| `native_route_probe_20261009_1145/` | backend/step/route事件、mixed和decode trace、服务日志、哈希与runner.exit |

PID/路径仅描述旧机器，不是新机器操作目标。`vllm_cache/`不归档，trace保留压缩原件。
未取得 `native_route_guard_20261009_1152` 最后逐项guard结果或恢复确认。
其他早期kernel、官方candidate、Level 3结论见完整记录，但对应原始文件并非
都包含在此目录，不能把远端路径当成已下载文件。

证据解释限制见 [阶段总结](../docs/stage-summary-2026-10-09.md)。
