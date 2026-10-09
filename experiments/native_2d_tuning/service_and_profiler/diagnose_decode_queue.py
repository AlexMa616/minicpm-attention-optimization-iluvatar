#!/usr/bin/env python3
"""Sample an idle experimental service during short 16K concurrency probes.

Does not restart services or change scheduler/kernel settings. These single-run,
short-output probes are diagnostic data, not official performance measurements.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

from prometheus_client.parser import text_string_to_metric_families


ROOT = Path("/workspace/vllm-plugin-FL")
MODEL = "/workspace/MiniCPM5-2B"
METRICS = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:num_preemptions_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
)


def read_metrics(port: int, model: str) -> tuple[dict, str]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as r:
        text = r.read().decode()
    values = {name: 0.0 for name in METRICS}
    found = set()
    cache_config = []
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.labels.get("model_name", model) != model:
                continue
            if sample.name in values:
                values[sample.name] += sample.value
                found.add(sample.name)
            elif sample.name == "vllm:cache_config_info":
                cache_config.append(sample.labels)
    if set(METRICS) - found:
        raise RuntimeError(f"Missing metrics: {sorted(set(METRICS) - found)}")
    return {"time": time.time(), **values, "cache_config": cache_config}, text


def verify_service(pid: int, port: int) -> dict:
    command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    args = [part.decode() for part in command if part]
    if "serve" not in args or "--port" not in args:
        raise RuntimeError("Expected PID is not a vLLM server")
    if args[args.index("--port") + 1] != str(port):
        raise RuntimeError("Server port does not match expected PID")
    env = dict(
        item.split("=", 1)
        for item in Path(f"/proc/{pid}/environ").read_bytes().decode().split("\0")
        if "=" in item
    )
    if env.get("CUDA_VISIBLE_DEVICES") != "4" or port != 9034:
        raise RuntimeError("This diagnostic is restricted to GPU 4 / port 9034")
    return {
        "pid": pid,
        "cmdline": args,
        "environment": {
            key: value for key, value in env.items()
            if key.startswith(("ILUVATAR_", "CUDA_VISIBLE", "VLLM_", "PYTHONPATH"))
        },
    }


def wait_idle(port: int, model: str, limit: int = 120) -> dict:
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        values, _ = read_metrics(port, model)
        if not values[METRICS[0]] and not values[METRICS[1]]:
            return values
        time.sleep(2)
    raise RuntimeError("Service is not idle; refusing to overlap probes")


def run_case(args: argparse.Namespace, concurrency: int, index: int) -> dict:
    verify_service(args.expected_pid, args.port)
    before = wait_idle(args.port, args.served_model_name)
    work = args.log_dir / f"c{concurrency}"
    work.mkdir()
    result_file = work / "benchmark.json"
    command = [
        "/usr/local/bin/vllm", "bench", "serve", "--backend", "vllm",
        "--model", args.served_model_name, "--tokenizer", MODEL,
        "--endpoint", "/v1/completions", "--base-url", f"http://127.0.0.1:{args.port}",
        "--dataset-name", "random", "--ignore-eos", "--random-input-len", "16384",
        "--random-output-len", "128", "--max-concurrency", str(concurrency),
        "--num-prompts", str(concurrency), "--seed", str(2026100900 + index),
        "--save-result", "--save-detailed", "--result-dir", str(work),
        "--result-filename", result_file.name,
    ]
    (work / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    samples = [before]
    errors = 0
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "4"
    start = time.monotonic()
    print(f"START c{concurrency} input=16384 output=128 prompts={concurrency}", flush=True)
    with (work / "benchmark.log").open("w") as log, (work / "metrics.jsonl").open("w") as raw:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                if time.monotonic() - start > 1800:
                    raise TimeoutError(f"c{concurrency}: exceeded 1800 seconds")
                try:
                    sample, text = read_metrics(args.port, args.served_model_name)
                    samples.append(sample)
                    raw.write(json.dumps(sample) + "\n")
                    raw.flush()
                    (work / "last_metrics.prom").write_text(text)
                except Exception as exc:
                    errors += 1
                    raw.write(json.dumps({"time": time.time(), "error": str(exc)}) + "\n")
                    raw.flush()
                    if errors >= 3:
                        raise RuntimeError("Three failed metrics reads; stopping probe") from exc
                time.sleep(2)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
        if process.returncode:
            raise RuntimeError(f"c{concurrency}: benchmark exit {process.returncode}")
    after = wait_idle(args.port, args.served_model_name)
    samples.append(after)
    result = json.loads(result_file.read_text())
    if result.get("completed") != concurrency or result.get("failed", 0) != 0:
        raise RuntimeError(f"c{concurrency}: not all requests succeeded")
    summary = {
        "concurrency": concurrency, "num_prompts": concurrency,
        "elapsed_s": time.monotonic() - start, "samples": len(samples),
        "metric_read_errors": errors,
        "max_running": max(s[METRICS[0]] for s in samples),
        "max_waiting": max(s[METRICS[1]] for s in samples),
        "max_kv_usage": max(s[METRICS[2]] for s in samples),
        "samples_kv_ge_99pct": sum(s[METRICS[2]] >= 0.99 for s in samples),
        "preemptions_delta": after[METRICS[3]] - before[METRICS[3]],
        "prefix_queries_delta": after[METRICS[4]] - before[METRICS[4]],
        "prefix_hits_delta": after[METRICS[5]] - before[METRICS[5]],
        "benchmark": result,
    }
    print(f"DONE c{concurrency}: max_kv={summary['max_kv_usage']:.4f} "
          f"waiting={summary['max_waiting']} preemptions={summary['preemptions_delta']}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=9034)
    parser.add_argument("--expected-pid", type=int, required=True)
    parser.add_argument("--served-model-name", default="minicpm-decode-gpu4")
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=False)
    with Path(f"/tmp/jl2026_decode_queue_{args.port}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        metadata = verify_service(args.expected_pid, args.port)
        metadata["initial_metrics"] = wait_idle(args.port, args.served_model_name, 5)
        sources = [Path(__file__), ROOT / "vllm_fl/dispatch/backends/vendor/iluvatar/impl/attention.py"]
        metadata["sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        (args.log_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        results = []
        try:
            for index, concurrency in enumerate((8, 24, 32, 64)):
                results.append(run_case(args, concurrency, index))
                (args.log_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
            (args.log_dir / "runner.exit").write_text("0\n")
        except BaseException as exc:
            (args.log_dir / "failure.json").write_text(json.dumps({"error": repr(exc)}) + "\n")
            (args.log_dir / "runner.exit").write_text("1\n")
            raise


if __name__ == "__main__":
    main()
