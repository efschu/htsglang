"""Who touches CUDA first? Stack-trace hook around torch.cuda initialization.

Answers "why does this process own a CUDA context at all" (task #237: the
GPU-passive launch_server parent held a ~634 MiB context). The hook wraps
``torch.cuda._lazy_init`` plus the query functions that trigger it
(``get_device_capability`` / ``get_device_properties`` / ``get_device_name``)
and appends one stack trace per call to ``cuda_init_trace.<pid>.log``. The
FIRST record in the parent's file is the answer.

Usage (GPU window, real server -- the parent process is the one of interest;
scheduler children are spawned without the hook and are not traced):

    python -m sglang.srt.utils.cuda_init_tracer --out /tmp/cuda_trace \\
        sglang.launch_server --model ... <usual args>

    -> read /tmp/cuda_trace/cuda_init_trace.<parent pid>.log

Zero-blind-spot variant (observes even ``import sglang`` itself, which the
``-m`` form performs before this module's main runs):

    python python/sglang/srt/utils/cuda_init_tracer.py --out /tmp/cuda_trace \\
        sglang.launch_server --model ... <usual args>

GPU-less falsification (CUDA_VISIBLE_DEVICES=99): pass ``--emulate`` to fake
a visible device (is_available -> True, capability (9, 0)); every code path
that WOULD have initialized CUDA on a GPU machine is then recorded without
any device present.

Limitations: a library that creates a context purely from C++ (bypassing
torch.cuda's Python layer) is invisible to this hook; nvidia-smi still shows
such a process. This module imports only stdlib + torch so the hook can be
installed before any sglang import.
"""

import argparse
import io
import os
import sys
import threading
import traceback
from typing import Optional

import torch

_MAX_RECORDS = 25

_lock = threading.Lock()
_num_records = 0
_log_path: Optional[str] = None


def _record(kind: str) -> None:
    global _num_records
    with _lock:
        _num_records += 1
        if _num_records > _MAX_RECORDS or _log_path is None:
            return
        buf = io.StringIO()
        buf.write(f"=== cuda touch #{_num_records} [{kind}] pid={os.getpid()} ===\n")
        # drop the two innermost frames (this function + the wrapper)
        traceback.print_stack(file=buf)
        lines = buf.getvalue().splitlines(keepends=True)
        with open(_log_path, "a") as f:
            f.writelines(lines)
            f.write("\n")


def install(out_dir: Optional[str] = None, emulate: bool = False) -> str:
    """Install the hooks. Returns the per-PID log path."""
    global _log_path
    out_dir = out_dir or "."
    os.makedirs(out_dir, exist_ok=True)
    _log_path = os.path.join(out_dir, f"cuda_init_trace.{os.getpid()}.log")

    orig_lazy_init = torch.cuda._lazy_init
    orig_capability = torch.cuda.get_device_capability
    orig_properties = torch.cuda.get_device_properties
    orig_name = torch.cuda.get_device_name

    def lazy_init(*args, **kwargs):
        _record("torch.cuda._lazy_init")
        if emulate:
            return None
        return orig_lazy_init(*args, **kwargs)

    def get_device_capability(device=None):
        _record("torch.cuda.get_device_capability")
        if emulate:
            return (9, 0)
        return orig_capability(device)

    def get_device_properties(device=None):
        _record("torch.cuda.get_device_properties")
        if emulate:
            return _FakeProperties()
        return orig_properties(device)

    def get_device_name(device=None):
        _record("torch.cuda.get_device_name")
        if emulate:
            return "EMULATED GPU"
        return orig_name(device)

    torch.cuda._lazy_init = lazy_init
    torch.cuda.get_device_capability = get_device_capability
    torch.cuda.get_device_properties = get_device_properties
    torch.cuda.get_device_name = get_device_name

    if emulate:
        torch.cuda.is_available = lambda: True
        torch.cuda.device_count = lambda: 1
        torch.cuda.current_device = lambda: 0

    return _log_path


class _FakeProperties:
    name = "EMULATED GPU"
    major = 9
    minor = 0
    total_memory = 32 << 30
    multi_processor_count = 100


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace the first torch.cuda initialization of a module run."
    )
    parser.add_argument(
        "--out", default="/tmp/cuda_trace", help="directory for per-PID trace logs"
    )
    parser.add_argument(
        "--emulate",
        action="store_true",
        help="fake a visible GPU so GPU-less runs record what WOULD init CUDA",
    )
    parser.add_argument("module", help="module to run as __main__ (like python -m)")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    ns = parser.parse_args()

    log_path = install(out_dir=ns.out, emulate=ns.emulate)
    print(f"[cuda_init_tracer] tracing pid {os.getpid()} -> {log_path}", flush=True)

    import runpy

    sys.argv = [ns.module] + ns.args
    runpy.run_module(ns.module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
