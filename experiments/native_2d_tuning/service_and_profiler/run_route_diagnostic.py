#!/usr/bin/env python3
"""Isolated GPU4 route evidence, limited profiling, then restore main service."""

import argparse
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import urllib.request

import native_decode_service_ab as service


MAIN = Path("/workspace/vllm-plugin-FL")
NAME = "minicpm-decode-gpu4"
PORT = 9034


def inject(root, helper):
    if (root / "vllm_fl/route_probe.py").exists():
        raise RuntimeError("Probe is already installed in this checkout")
    shutil.copy2(helper, root / "vllm_fl/route_probe.py")
    changes = (
        ("vllm_fl/dispatch/backends/vendor/iluvatar/impl/attention.py",
         "    if native_2d_tuning and native_supported and (max_query_len > 1 or decode_only):",
         "    from vllm_fl.route_probe import probe_route\n"
         "    probe_route(kwargs, native_supported, decode_only, native_2d_tuning)\n"),
        ("vllm_fl/worker/model_runner.py",
         '            record_function_or_nullcontext("gpu_model_runner: forward"),',
         "            probe_step(self, scheduler_output, attn_metadata, cudagraph_mode),\n"),
    )
    for relative, anchor, addition in changes:
        path = root / relative
        text = path.read_text()
        if text.count(anchor) != 1:
            raise RuntimeError(f"Unique hook anchor not found in {path}")
        if "model_runner.py" in relative:
            text = "from vllm_fl.route_probe import probe_step\n" + text
        text = text.replace(anchor, addition + anchor, 1)
        ast.parse(text)
        path.write_text(text)
    path = root / "vllm_fl/dispatch/backends/vendor/iluvatar/iluvatar.py"
    text = path.read_text()
    anchor = '        return (\n            "vllm_fl.dispatch.backends.vendor.iluvatar.impl.attention."'
    if text.count(anchor) != 1:
        raise RuntimeError("Backend selection hook anchor not unique")
    text = text.replace(anchor,
        '        from vllm_fl.route_probe import probe_backend\n'
        '        probe_backend("vllm_fl.dispatch.backends.vendor.iluvatar.impl.attention.IluvatarAttentionBackend")\n' + anchor)
    ast.parse(text)
    path.write_text(text)


def start_probe(root, logs):
    service.ROOT = root
    env = service.runtime_env(4, True)
    env["JL2026_ROUTE_PROBE_DIR"] = str(logs / "events")
    env["VLLM_CACHE_ROOT"] = str(logs / "vllm_cache")
    config = {"profiler": "torch", "torch_profiler_dir": str(logs / "traces"),
              "torch_profiler_with_stack": False, "torch_profiler_record_shapes": False,
              "ignore_frontend": True, "max_iterations": 8}
    command = ["/usr/local/bin/vllm", "serve", service.MODEL,
               "--port", str(PORT), "--served-model-name", NAME,
               "--gpu-memory-utilization", "0.85", "--max-model-len", "131072",
               "--compilation-config", '{"cudagraph_mode":"FULL_DECODE_ONLY"}',
               "--profiler-config", json.dumps(config)]
    (logs / "start_command.json").write_text(json.dumps(command, indent=2) + "\n")
    output = (logs / "probe_service.log").open("w")
    process = subprocess.Popen(command, cwd=root, env=env, stdout=output,
                               stderr=subprocess.STDOUT, start_new_session=True)
    (logs / "probe_pid").write_text(str(process.pid) + "\n")
    for _ in range(600):
        if process.poll() is not None:
            output.close()
            raise RuntimeError("Probe service failed startup; inspect probe_service.log")
        if service.healthy(PORT):
            return process, output
        time.sleep(2)
    process.terminate()
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        # Only this newly launched process group is owned by the runner.
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
    output.close()
    raise TimeoutError("Diagnostic startup timed out")


def phase_event(events, since, kind):
    latest = None
    for path in events.glob("events_*.jsonl"):
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") == "step_begin" and row["time"] >= since:
                if latest is None or row["time"] > latest["time"]:
                    latest = row
    if latest and latest["kind"] == kind and time.time() - latest["time"] < 1:
        if kind != "q1" or latest["q1_requests"] >= 8:
            return latest
    return None


def profile_api(action):
    request = urllib.request.Request(f"http://127.0.0.1:{PORT}/{action}_profile",
                                     data=b"", method="POST")
    with urllib.request.urlopen(request, timeout=180) as response:
        return {"status": response.status, "body": response.read().decode()[:1000]}


def benchmark(root, logs, label, input_len, output_len, kind):
    work = logs / label
    work.mkdir()
    command = ["/usr/local/bin/vllm", "bench", "serve", "--backend", "vllm",
               "--model", NAME, "--tokenizer", service.MODEL, "--endpoint", "/v1/completions",
               "--base-url", f"http://127.0.0.1:{PORT}", "--dataset-name", "random",
               "--random-input-len", str(input_len), "--random-output-len", str(output_len),
               "--ignore-eos", "--max-concurrency", "8", "--num-prompts", "8",
               "--seed", str(input_len + 20261009), "--save-result", "--save-detailed",
               "--result-dir", str(work), "--result-filename", "benchmark.json"]
    (work / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    started = time.time()
    trigger = None
    with (work / "benchmark.log").open("w") as output:
        process = subprocess.Popen(command, cwd=root, env=service.runtime_env(4, True),
                                   stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            while process.poll() is None:
                if time.time() - started > 1800:
                    raise TimeoutError(f"{label} timed out")
                if trigger is None:
                    row = phase_event(logs / "events", started, kind)
                    if row:
                        trigger = {"event": row, "start": profile_api("start")}
                        (work / "profile_trigger.json").write_text(json.dumps(trigger, indent=2) + "\n")
                time.sleep(1)
            if process.returncode:
                raise RuntimeError(f"{label} benchmark failed: {process.returncode}")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            if trigger is not None:
                (work / "profile_stop.json").write_text(json.dumps(profile_api("stop")) + "\n")
    if trigger is None:
        raise RuntimeError(f"{label}: did not observe requested {kind} window")
    result = json.loads((work / "benchmark.json").read_text())
    if result.get("completed") != 8 or result.get("failed", 0):
        raise RuntimeError(f"{label}: incomplete requests")
    print(f"DONE {label} completed=8; diagnostic only", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-pid", type=int, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    lock = Path("/tmp/jl2026_route_diagnostic_9034.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.checkout.exists() or args.log_dir.exists():
        raise RuntimeError("Diagnostic paths must be new")
    service.verify_server(args.expected_pid, PORT)
    env = Path(f"/proc/{args.expected_pid}/environ").read_bytes().decode().split("\0")
    if "CUDA_VISIBLE_DEVICES=4" not in env:
        raise RuntimeError("Expected PID is not on GPU4")
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=5) as response:
        metrics = response.read().decode()
    for line in metrics.splitlines():
        if line.startswith(("vllm:num_requests_running{", "vllm:num_requests_waiting{")):
            if float(line.rsplit(" ", 1)[1]):
                raise RuntimeError("Refusing to stop a busy service")
    args.log_dir.mkdir(parents=True)
    shutil.copytree(MAIN, args.checkout, ignore=shutil.ignore_patterns(
        ".git", "__pycache__", "benchmark_results", "vllm-plugin-FL"))
    inject(args.checkout, Path(__file__).with_name("route_probe.py"))
    files = ["vllm_fl/route_probe.py", "vllm_fl/worker/model_runner.py",
             "vllm_fl/dispatch/backends/vendor/iluvatar/impl/attention.py",
             "vllm_fl/dispatch/backends/vendor/iluvatar/iluvatar.py"]
    (args.log_dir / "source_hashes.json").write_text(json.dumps({
        f: hashlib.sha256((args.checkout / f).read_bytes()).hexdigest() for f in files}, indent=2) + "\n")
    probe = None
    handle = None
    stopped = False
    outcome = 1
    try:
        service.stop_server(args.expected_pid, PORT)
        stopped = True
        print(f"STOPPED owned service {args.expected_pid}", flush=True)
        probe, handle = start_probe(args.checkout, args.log_dir)
        print(f"PROBE healthy pid={probe.pid}", flush=True)
        service.api_smoke(PORT, NAME, True)
        if not args.smoke_only:
            benchmark(args.checkout, args.log_dir, "mixed_16k_c8", 16384, 128, "mixed")
            benchmark(args.checkout, args.log_dir, "decode_256_c8", 256, 512, "q1")
        outcome = 0
    except BaseException as exc:
        (args.log_dir / "failure.json").write_text(json.dumps({"error": repr(exc)}) + "\n")
        raise
    finally:
        try:
            if probe is not None and probe.poll() is None:
                service.stop_server(probe.pid, PORT)
            if handle is not None:
                handle.close()
            if stopped:
                service.ROOT = MAIN
                pid, restored_handle = service.start_server(4, PORT, NAME, True,
                    args.log_dir / "restored_service.log")
                restored_handle.close()
                (args.log_dir / "restored_pid").write_text(str(pid) + "\n")
                service.api_smoke(PORT, NAME, True)
                print(f"RESTORED main checkout decode_on pid={pid}", flush=True)
        except BaseException as exc:
            outcome = 1
            (args.log_dir / "restore_failure.json").write_text(json.dumps({"error": repr(exc)}) + "\n")
            raise
        finally:
            (args.log_dir / "runner.exit").write_text(str(outcome) + "\n")


if __name__ == "__main__":
    main()
