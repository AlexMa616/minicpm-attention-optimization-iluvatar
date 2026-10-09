#!/usr/bin/env python3
"""Same-window 16K/c64 scheduler A/B for the frozen Native 2D candidate."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request


ROOT = Path("/workspace/vllm-plugin-FL")
MODEL = "/workspace/MiniCPM5-2B"
MARKER = Path("/tmp/iluvatar_native_2d.enable")
DUAL_MARKER = Path("/tmp/iluvatar_native_2d_dual.enable")
CASE = [16384, 64, 64, 64]
CONFIGS = (("default2048", None), ("mbt4096", 4096))


def healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as response:
            return response.status == 200
    except Exception:
        return False


def verify_pid(pid: int) -> None:
    command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    if "/vllm serve " not in command or "--port 9032" not in command:
        raise RuntimeError(f"PID {pid} is not the expected 9032 server: {command}")


def stop(pid: int) -> None:
    verify_pid(pid)
    os.kill(pid, signal.SIGTERM)
    for _ in range(180):
        stat = Path(f"/proc/{pid}/stat")
        if not stat.exists() or stat.read_text().split()[2] == "Z":
            return
        time.sleep(1)
    raise RuntimeError(f"9032 PID {pid} did not terminate cleanly")


def env() -> dict[str, str]:
    result = os.environ.copy()
    result.update({
        "CUDA_VISIBLE_DEVICES": "1",
        "VLLM_PLUGINS": "fl",
        "PYTHONPATH": f"{ROOT}:/usr/local/lib/python3.12/site-packages",
        "LD_LIBRARY_PATH": ":".join(filter(None, (
            "/usr/local/corex/lib64", "/usr/local/openmpi/lib", "/usr/local/lib",
            result.get("LD_LIBRARY_PATH", ""),
        ))),
    })
    return result


def start(
    log: Path,
    max_tokens: int | None,
    *,
    no_prefix_cache: bool,
) -> tuple[int, object]:
    MARKER.touch()
    DUAL_MARKER.unlink(missing_ok=True)
    command = [
        "/usr/local/bin/vllm", "serve", MODEL, "--port", "9032",
        "--served-model-name", "minicpm-diag", "--gpu-memory-utilization", "0.85",
        "--max-model-len", "131072",
        "--compilation-config", '{"cudagraph_mode":"FULL_DECODE_ONLY"}',
    ]
    if no_prefix_cache:
        command.append("--no-enable-prefix-caching")
    if max_tokens is not None:
        command.extend(["--max-num-batched-tokens", str(max_tokens)])
    handle = log.open("w")
    process = subprocess.Popen(command, cwd=ROOT, env=env(), stdout=handle,
                               stderr=subprocess.STDOUT, start_new_session=True)
    print(f"START pid={process.pid} max_num_batched_tokens={max_tokens}", flush=True)
    for _ in range(450):
        if process.poll() is not None:
            handle.close()
            raise RuntimeError(f"9032 exited during startup: {log}")
        if healthy(9032):
            return process.pid, handle
        time.sleep(2)
    handle.close()
    raise TimeoutError(f"9032 did not become healthy: {log}")


def benchmark(name: str, root: Path) -> list[dict[str, str]]:
    root.mkdir(parents=True, exist_ok=True)
    command = [
        "/usr/local/bin/python3.12", "-u", str(ROOT / "benchmarks/benchmark_throughput_serve.py"),
        "--served-model-name", "minicpm-diag", "--model", MODEL, "--port", "9032",
        "--test-cases", json.dumps([CASE]),
    ]
    with (root / "benchmark.log").open("w") as output:
        subprocess.run(command, cwd=root, env=env(), stdout=output,
                       stderr=subprocess.STDOUT, check=True, timeout=10800)
    summaries = list((root / "benchmark_results").glob("summary_*.csv"))
    raws = list((root / "benchmark_results").glob("raw_runs_*.csv"))
    if len(summaries) != 1 or len(raws) != 1:
        raise RuntimeError(f"{name}: missing summary/raw CSV")
    with summaries[0].open(newline="") as source:
        summary = list(csv.DictReader(source))
    with raws[0].open(newline="") as source:
        raw = list(csv.DictReader(source))
    if len(summary) != 1 or len(raw) != 4:
        raise RuntimeError(f"{name}: incomplete CSV {len(summary)}/{len(raw)}")
    if any(row["Run Status"] != "SUCCESS" or float(row["Successful Requests"]) != CASE[3]
           for row in raw[1:]):
        raise RuntimeError(f"{name}: failed steady-state request")
    print(f"SUMMARY {name}: {json.dumps(summary)}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-pid", type=int, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    verify_pid(args.expected_pid)
    if not healthy(9031) or not healthy(9032):
        raise RuntimeError("9031 and 9032 must be healthy before A/B")
    current = args.expected_pid
    handle = None
    results = {}
    try:
        for name, max_tokens in CONFIGS:
            stop(current)
            if handle is not None:
                handle.close()
            current, handle = start(
                args.log_dir / f"{name}_service.log",
                max_tokens,
                no_prefix_cache=True,
            )
            results[name] = benchmark(name, args.log_dir / name)
            (args.log_dir / "results.json").write_text(json.dumps(results, indent=2))
    finally:
        if current is not None:
            stop(current)
        if handle is not None:
            handle.close()
        DUAL_MARKER.unlink(missing_ok=True)
        MARKER.touch()
        restored, restored_handle = start(
            args.log_dir / "restored_candidate_service.log",
            None,
            no_prefix_cache=False,
        )
        (args.log_dir / "restored_pid").write_text(f"{restored}\n")
        restored_handle.close()
        print(f"RESTORED candidate pid={restored}", flush=True)
    if not healthy(9031):
        raise RuntimeError("9031 health check failed after A/B")


if __name__ == "__main__":
    main()
