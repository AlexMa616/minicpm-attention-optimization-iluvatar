"""Diagnostic hooks copied only into an isolated vllm_fl checkout."""

from contextlib import contextmanager
import inspect
import json
import os
from pathlib import Path
import time

import torch


_STEP = 0
_SEEN = set()
_OUTPUT = None


def emit(event, **fields):
    global _OUTPUT
    if _OUTPUT is None:
        directory = Path(os.environ["JL2026_ROUTE_PROBE_DIR"])
        directory.mkdir(parents=True, exist_ok=True)
        _OUTPUT = (directory / f"events_{os.getpid()}.jsonl").open("a", buffering=1)
    _OUTPUT.write(json.dumps({"event": event, "time": time.time(),
                              "pid": os.getpid(), **fields}) + "\n")


@contextmanager
def probe_step(runner, scheduler_output, metadata, graph_mode):
    global _STEP
    _STEP += 1
    _SEEN.clear()
    counts = list(scheduler_output.num_scheduled_tokens.values())
    one = sum(n == 1 for n in counts)
    many = sum(n > 1 for n in counts)
    kind = "q1" if not many else "mixed" if one else "prefill"
    bundles = metadata if isinstance(metadata, list) else [metadata]
    details = {}
    for bundle in bundles:
        if not isinstance(bundle, dict):
            continue
        for item in bundle.values():
            cls = type(item)
            key = f"{cls.__module__}.{cls.__qualname__}"
            details[key] = {"class": key, "source": inspect.getfile(cls),
                            "max_query_len": getattr(item, "max_query_len", None)}
    config = runner.scheduler_config
    fields = {name: getattr(config, name, None) for name in (
        "max_num_seqs", "max_num_batched_tokens", "max_num_scheduled_tokens",
        "enable_chunked_prefill", "max_num_partial_prefills",
        "max_long_partial_prefills", "long_prefill_token_threshold")}
    emit("step_begin", step=_STEP, kind=kind, q1_requests=one,
         q_gt1_requests=many, scheduled_tokens=sum(counts),
         graph_mode=str(graph_mode), metadata=list(details.values()),
         scheduler_config=fields)
    start = time.monotonic()
    with torch.profiler.record_function(f"JL2026_STEP_{_STEP}_{kind}_{graph_mode}"):
        try:
            yield
        finally:
            emit("step_end", step=_STEP, kind=kind,
                 host_elapsed_s=time.monotonic() - start)


def probe_route(kwargs, supported, decode_only, tune_enabled):
    capture = torch.cuda.is_current_stream_capturing()
    native = tune_enabled and supported and (kwargs["max_seqlen_q"] > 1 or decode_only)
    signature = (_STEP, capture, native, decode_only, kwargs["max_seqlen_q"],
                 tuple(kwargs["q"].shape))
    if signature in _SEEN:
        return
    _SEEN.add(signature)
    caller = inspect.currentframe().f_back
    # Evaluate exactly the guard from this isolated caller, without reading GPU data.
    import ast
    source = Path(caller.f_code.co_filename).read_text()
    function = next(n for n in ast.parse(source).body
                    if isinstance(n, ast.FunctionDef) and n.name == "_run_unified_attention")
    guard = next(n.value for n in function.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "native_supported" for t in n.targets))
    checks = []
    for predicate in guard.values:
        expression = ast.Expression(body=predicate)
        result = eval(compile(expression, caller.f_code.co_filename, "eval"),
                      caller.f_globals, caller.f_locals)
        checks.append({"predicate": ast.unparse(predicate), "passed": bool(result)})
    emit("route", step=_STEP, capturing=capture,
         max_query_len=kwargs["max_seqlen_q"], q_shape=list(kwargs["q"].shape),
         supported=supported, native=native, decode_only=decode_only,
         source=__file__, caller=caller.f_code.co_filename,
         guards=checks,
         block_m=(int(os.environ.get("ILUVATAR_NATIVE_DECODE_BLOCK_M", "8"))
                  if native and decode_only else None))


def probe_backend(path):
    emit("backend_selection", backend=path,
         caller=inspect.currentframe().f_back.f_code.co_filename)
