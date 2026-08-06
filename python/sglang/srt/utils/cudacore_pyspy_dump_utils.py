# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""CUDA core dump and py-spy dump utilities."""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from errno import ENXIO
from pathlib import Path
from typing import List, Optional

import psutil

logger = logging.getLogger(__name__)


def resolve_pyspy_binary() -> Optional[str]:
    """Absolute path to the py-spy executable, or None if it cannot be found.

    py-spy is a console script installed into the SAME bin directory as the
    interpreter running us. It is NOT necessarily on PATH: a server launched as
    ``/path/to/venv/bin/python -m sglang.launch_server`` never activates the
    venv, so a bare ``py-spy`` resolved through ``/bin/sh`` misses it and every
    automatic wedge dump silently produces nothing (observed 2026-08-06: the
    only dump of a live wedge failed with "/bin/sh: 1: py-spy: not found" while
    py-spy sat in the venv bin dir the whole time).

    Search sys.executable's directory WITHOUT resolving symlinks first. A venv's
    ``bin/python`` is normally a symlink to the base interpreter, so resolving it
    lands in ``/usr/bin`` -- where the venv's console scripts are not, which
    reproduces the very bug this function exists to prevent. The resolved
    directory is still tried afterwards, for the layouts where it is the right
    answer, and PATH last.
    """
    override = os.environ.get("SGLANG_PYSPY_BIN")
    if override:
        return override

    executable = Path(sys.executable)
    seen = set()
    for bindir in (executable.parent, executable.resolve().parent):
        if bindir in seen:
            continue
        seen.add(bindir)
        candidate = bindir / "py-spy"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)

    return shutil.which("py-spy")


def _resolve_cuda_coredump_pipe_path(proc: psutil.Process) -> Path:
    pipe_template = os.environ.get("CUDA_COREDUMP_PIPE")
    if pipe_template is None:
        pipe_path = f"corepipe.cuda.{platform.node()}.{proc.pid}"
    else:
        pipe_path = (
            pipe_template.replace("%h", platform.node())
            .replace("%p", str(proc.pid))
            .replace("%t", str(int(time.time())))
        )

    path = Path(pipe_path)
    if path.is_absolute():
        return path

    try:
        return Path(proc.cwd()) / path
    except (psutil.Error, OSError):
        return Path.cwd() / path


def _is_sglang_scheduler_process(proc: psutil.Process) -> bool:
    try:
        proc_title = " ".join(proc.cmdline())
    except (psutil.Error, OSError):
        return False
    return proc_title.startswith("sglang::scheduler")


def collect_scheduler_processes() -> List[psutil.Process]:
    current = psutil.Process()
    return [
        proc
        for proc in current.children(recursive=True)
        if _is_sglang_scheduler_process(proc)
    ]


def pyspy_dump_schedulers(scheduler_only=False):
    """py-spy dump on all scheduler in a local node."""
    if scheduler_only:
        procs = collect_scheduler_processes()
        if not procs:
            logger.error("No sglang scheduler processes found for py-spy dump.")
            return
        pids = [proc.pid for proc in procs]
    else:
        pids = [psutil.Process().pid]

    pyspy = resolve_pyspy_binary()
    if pyspy is None:
        logger.error(
            "py-spy executable not found next to %s nor on PATH; skipping the "
            "dump. Install it into this interpreter's environment, or point "
            "SGLANG_PYSPY_BIN at it.",
            sys.executable,
        )
        return

    for pid in pids:
        for attempt, native_flag in enumerate([["--native"], []]):
            argv = [pyspy, "dump", *native_flag, "--pid", str(pid)]
            try:
                result = subprocess.run(
                    argv, capture_output=True, text=True, check=True
                )
                logger.error(
                    f"Pyspy dump for PID {pid} ({' '.join(argv)}):\n{result.stdout}"
                )
                break
            except (subprocess.CalledProcessError, OSError) as e:
                stderr = getattr(e, "stderr", None) or str(e)
                logger.error(f"Pyspy failed ({' '.join(argv)}). Error: {stderr}")
                if attempt == 1:
                    logger.error(f"All pyspy dump attempts failed for PID {pid}.")


def trigger_cuda_user_coredump(scheduler_only=False):
    """Trigger CUDA user-induced GPU core dumps by writing to coredump pipes."""
    if os.environ.get("CUDA_ENABLE_USER_TRIGGERED_COREDUMP") != "1":
        logger.error(
            "CUDA user-triggered coredump is not enabled. Set "
            "CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1 before CUDA initialization."
        )

    if scheduler_only:
        procs = collect_scheduler_processes()
        if not procs:
            logger.error("No sglang scheduler processes found for CUDA coredump.")
            return
    else:
        procs = [psutil.Process()]

    for proc in procs:
        pipe_path = _resolve_cuda_coredump_pipe_path(proc)
        try:
            fd = os.open(pipe_path, os.O_WRONLY | os.O_NONBLOCK)
            try:
                os.write(fd, b"1")
            finally:
                os.close(fd)
            logger.error(
                "Triggered CUDA user coredump for PID %s via %s",
                proc.pid,
                pipe_path,
            )
        except FileNotFoundError:
            logger.error(
                "CUDA coredump pipe not found for PID %s: %s. Ensure "
                "CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1 was set before this "
                "process initialized CUDA.",
                proc.pid,
                pipe_path,
            )
        except OSError as e:
            if e.errno == ENXIO:
                logger.error(
                    "CUDA coredump pipe has no reader for PID %s: %s",
                    proc.pid,
                    pipe_path,
                )
            else:
                logger.exception(
                    "Failed to trigger CUDA user coredump for PID %s via %s",
                    proc.pid,
                    pipe_path,
                )
