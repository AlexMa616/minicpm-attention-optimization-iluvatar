#!/usr/bin/env python3
"""Matched service A/B for the opt-in exact decode launch."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request


ROOT = Path("/workspace/vllm-plugin-FL")
MODEL = "/workspace/MiniCPM5-2B"
CASES = [[4096, 1024, 64, 256], [16384, 1024, 64, 128]]


def healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def port_available(port: int) -> bool:
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def verify_server(pid: int, port: int) -> None:
    command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    if "/vllm serve " not in command or f"--port {port}" not in command:
        raise RuntimeError(f"PID {pid} is not the expected port-{port} server: {command}")


def stop_server(pid: int, port: int) -> None:
    verify_server(pid, port)
    os.kill(pid, signal.SIGTERM)
    for _ in range(180):
        stat_path = Path(f"/proc/{pid}/stat")
        if not stat_path.exists() or stat_path.read_text().split()[2] == "Z":
            return
        time.sleep(1)
    raise RuntimeError(f"port-{port} PID {pid} did not stop cleanly")


def runtime_env(gpu: int, decode_enabled: bool) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "VLLM_PLUGINS": "fl",
            "PYTHONPATH": f"{ROOT}:/usr/local/lib/python3.12/site-packages",
            "LD_LIBRARY_PATH": ":".join(
                filter(
                    None,
                    (
                        "/usr/local/corex/lib64",
                        "/usr/local/openmpi/lib",
                        "/usr/local/lib",
                        env.get("LD_LIBRARY_PATH", ""),
                    ),
                )
            ),
            "ILUVATAR_NATIVE_2D_TUNE": "1",
            "ILUVATAR_NATIVE_DECODE_TUNE": "1" if decode_enabled else "0",
            "ILUVATAR_NATIVE_DECODE_BLOCK_M": "8",
        }
    )
    return env


def start_server(
    gpu: int, port: int, served_model_name: str, decode_enabled: bool, log_path: Path
) -> tuple[int, object]:
    command = [
        "/usr/local/bin/vllm",
        "serve",
        MODEL,
        "--port",
        str(port),
        "--served-model-name",
        served_model_name,
        "--gpu-memory-utilization",
        "0.85",
        "--max-model-len",
        "131072",
        "--compilation-config",
        '{"cudagraph_mode":"FULL_DECODE_ONLY"}',
    ]
    output = log_path.open("w")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=runtime_env(gpu, decode_enabled),
        stdout=output,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print(
        f"START gpu={gpu} port={port} decode={int(decode_enabled)} pid={process.pid}",
        flush=True,
    )
    for _ in range(450):
        if process.poll() is not None:
            output.close()
            raise RuntimeError(
                f"port-{port} decode={int(decode_enabled)} exited during startup"
            )
        if healthy(port):
            return process.pid, output
        time.sleep(2)
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
    output.close()
    raise TimeoutError(f"port-{port} decode={int(decode_enabled)} did not become healthy")


def api_smoke(port: int, served_model_name: str, decode_enabled: bool) -> None:
    payload = json.dumps(
        {
            "model": served_model_name,
            "prompt": f"Decode A/B smoke {time.time_ns()}: reply briefly.",
            "max_tokens": 8,
            "temperature": 0.0,
            "stream": False,
        }
    ).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.loads(response.read())
    if response.status != 200 or not result.get("choices"):
        raise RuntimeError(f"port-{port} decode={int(decode_enabled)} API smoke failed")
    print(f"API_SMOKE decode={int(decode_enabled)} status=200", flush=True)


def run_benchmark(
    name: str, port: int, served_model_name: str, gpu: int, log_dir: Path
) -> list[dict[str, str]]:
    work = log_dir / name
    work.mkdir()
    command = [
        "/usr/local/bin/python3.12",
        "-u",
        str(ROOT / "benchmarks/benchmark_throughput_serve.py"),
        "--served-model-name",
        served_model_name,
        "--model",
        MODEL,
        "--port",
        str(port),
        "--test-cases",
        json.dumps(CASES),
    ]
    env = runtime_env(gpu, name == "decode_on")
    with (work / "benchmark.log").open("w") as output:
        subprocess.run(
            command,
            cwd=work,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=14400,
        )

    summaries = list((work / "benchmark_results").glob("summary_*.csv"))
    raw_runs = list((work / "benchmark_results").glob("raw_runs_*.csv"))
    if len(summaries) != 1 or len(raw_runs) != 1:
        raise RuntimeError(f"{name}: expected exactly one raw and summary CSV")
    with summaries[0].open(newline="") as source:
        summary = list(csv.DictReader(source))
    with raw_runs[0].open(newline="") as source:
        raw = list(csv.DictReader(source))
    if len(summary) != len(CASES) or len(raw) != 4 * len(CASES):
        raise RuntimeError(f"{name}: incomplete raw/summary rows {len(raw)}/{len(summary)}")
    for case in CASES:
        rows = [
            row
            for row in raw
            if int(float(row["Prefill"])) == case[0]
            and int(float(row["Conc"])) == case[2]
        ]
        if len(rows) != 4 or any(
            row["Run Status"] != "SUCCESS"
            or float(row["Successful Requests"]) != case[3]
            for row in rows
        ):
            raise RuntimeError(f"{name}: failed or incomplete case {case}")
    print(f"SUMMARY {name}: {json.dumps(summary)}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--served-model-name", default="minicpm-diag")
    parser.add_argument("--expected-pid", type=int)
    parser.add_argument("--restore-decode-on", action="store_true")
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    if args.expected_pid is not None:
        verify_server(args.expected_pid, args.port)
        if not healthy(args.port):
            raise RuntimeError(f"port-{args.port} must be healthy before the A/B")
    elif not port_available(args.port):
        raise RuntimeError(f"port-{args.port} is already bound; refusing to replace it")

    current_pid: int | None = args.expected_pid
    log_handle = None
    results: dict[str, list[dict[str, str]]] = {}
    try:
        for enabled, name in ((False, "decode_off"), (True, "decode_on")):
            if current_pid is not None:
                stop_server(current_pid, args.port)
                current_pid = None
            if log_handle is not None:
                log_handle.close()
            current_pid, log_handle = start_server(
                args.gpu,
                args.port,
                args.served_model_name,
                enabled,
                args.log_dir / f"{name}_service.log",
            )
            api_smoke(args.port, args.served_model_name, enabled)
            results[name] = run_benchmark(
                name, args.port, args.served_model_name, args.gpu, args.log_dir
            )
            (args.log_dir / "results.json").write_text(json.dumps(results, indent=2))
    finally:
        if current_pid is not None:
            stop_server(current_pid, args.port)
        if log_handle is not None:
            log_handle.close()
        if args.restore_decode_on:
            restored_pid, restored_log = start_server(
                args.gpu,
                args.port,
                args.served_model_name,
                True,
                args.log_dir / "restored_decode_on_service.log",
            )
            (args.log_dir / "restored_pid").write_text(f"{restored_pid}\n")
            restored_log.close()
            print(f"RESTORED decode_on port={args.port} pid={restored_pid}", flush=True)


if __name__ == "__main__":
    main()
