# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Child-process and HTTP plumbing shared by the process-isolated adapters.

Process isolation is the load-bearing decision of §3.4: the process boundary
is the memory-saver tag scope, so parking a tenant is releasing its memory and,
at ``COLD``, exiting its process -- no allocator work required. That makes
"start a server, wait for it, ask it a question, stop it" the shared mechanic
of the Class-1 and Class-3 adapters, and it belongs in one place.

Only the standard library is used for HTTP. The registry has to be importable
and runnable in a planner process that installs nothing.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

#: How long a terminate is given before the process is killed. A model server
#: mid-CUDA-free that is killed too early leaks the device context, and the
#: next tenant then fails an admission that the ledger said would fit.
DEFAULT_STOP_GRACE_S = 30.0


class ProcessTenantError(RuntimeError):
    """The child process did not do what was asked."""


def http_json(
    url: str,
    *,
    method: str = "GET",
    payload: Mapping[str, Any] | None = None,
    timeout: float = 10.0,
) -> Any:
    """One request, standard library only. Raises on any non-2xx."""
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(  # noqa: S310 - fixed http scheme, local
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        raw = response.read()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw.decode(errors="replace")


def http_ok(url: str, *, timeout: float = 5.0) -> bool:
    try:
        http_json(url, timeout=timeout)
        return True
    except (urllib.error.URLError, OSError, ProcessTenantError):
        return False


@dataclass
class ChildProcess:
    """A tenant process the registry owns.

    ``env`` is built explicitly rather than inherited wholesale where it
    matters: ``CUDA_VISIBLE_DEVICES`` must name exactly the physical cards this
    tenant may touch, because that is the isolation strategy the whole design
    rests on -- inside such a process ``cuda:0`` is unambiguous and no
    logical-to-physical mapping table exists to get wrong.
    """

    argv: Sequence[str]
    env: Mapping[str, str]
    log_path: str | None = None
    _popen: subprocess.Popen | None = None
    _log_handle: Any = None

    @property
    def pid(self) -> int:
        return self._popen.pid if self._popen is not None else 0

    @property
    def running(self) -> bool:
        return self._popen is not None and self._popen.poll() is None

    @property
    def returncode(self) -> int | None:
        return None if self._popen is None else self._popen.poll()

    def start(self) -> None:
        if self.running:
            raise ProcessTenantError("process is already running")
        stdout: Any = subprocess.DEVNULL
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
            self._log_handle = open(self.log_path, "ab", buffering=0)
            stdout = self._log_handle
        env = {**os.environ, **self.env}
        logger.info("registry: starting tenant process: %s", " ".join(self.argv))
        self._popen = subprocess.Popen(  # noqa: S603 - argv is built, not shell
            list(self.argv),
            env=env,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def stop(self, *, grace_s: float = DEFAULT_STOP_GRACE_S) -> None:
        """SIGTERM the process group, then SIGKILL what is left.

        The group, not the pid: a model server is a tree (tokenizer, scheduler,
        detokenizer), and a pid-only terminate leaves the workers holding the
        device while the ledger has already given their bytes away.
        """
        popen = self._popen
        if popen is None:
            return
        if popen.poll() is None:
            try:
                os.killpg(os.getpgid(popen.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                popen.terminate()
            deadline = time.monotonic() + grace_s
            while time.monotonic() < deadline and popen.poll() is None:
                time.sleep(0.05)
            if popen.poll() is None:
                logger.warning(
                    "registry: tenant pid %d ignored SIGTERM for %.0fs; killing",
                    popen.pid,
                    grace_s,
                )
                try:
                    os.killpg(os.getpgid(popen.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    popen.kill()
                popen.wait(timeout=grace_s)
        self._popen = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def child_pids(self) -> tuple[int, ...]:
        """Every pid in the tenant's process group, best effort.

        NVML accounts device bytes per pid, and a server's device memory is
        held by its worker children, not by the launcher. Reading ``/proc`` is
        cheap and needs no extra dependency.
        """
        if not self.running:
            return ()
        root = self.pid
        try:
            pgid = os.getpgid(root)
        except (ProcessLookupError, PermissionError):
            return (root,)
        pids: list[int] = []
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            try:
                with open(f"/proc/{pid}/stat", "rb") as handle:
                    fields = handle.read().rsplit(b")", 1)[-1].split()
                # After the comm field: state, ppid, pgrp, ...
                if int(fields[2]) == pgid:
                    pids.append(pid)
            except (OSError, IndexError, ValueError):
                continue
        return tuple(sorted(pids)) or (root,)


def wait_for(
    predicate,
    *,
    timeout_s: float,
    poll_s: float = 0.5,
    on_dead=None,
    what: str = "condition",
) -> float:
    """Poll ``predicate`` until true; return elapsed milliseconds.

    ``on_dead`` is checked every iteration and aborts the wait immediately: a
    tenant whose process has exited will never become healthy, and waiting the
    full timeout for it turns a 3-second failure into a 5-minute one.
    """
    started = time.monotonic()
    deadline = started + timeout_s
    while True:
        if predicate():
            return (time.monotonic() - started) * 1000.0
        if on_dead is not None:
            dead = on_dead()
            if dead:
                raise ProcessTenantError(
                    f"tenant process exited while waiting for {what}: {dead}"
                )
        if time.monotonic() >= deadline:
            raise ProcessTenantError(
                f"timed out after {timeout_s:g}s waiting for {what}"
            )
        time.sleep(poll_s)


def cuda_runtime_library_path() -> str:
    """``LD_LIBRARY_PATH`` additions for the CUDA runtime shipped in the venv.

    The memory saver works by ``LD_PRELOAD``-ing a small shim, and the loader
    resolves that shim's dependencies *before* Python runs -- so the CUDA
    runtime has to be findable through the loader path, not through whatever
    torch imports later. On a rig whose system CUDA is older than the wheel's
    (here: ``libcudart.so.12`` installed, torch built against 13), the preload
    fails with ``cannot open shared object file`` and the scheduler exits 127
    before printing anything about memory saving. That is a five-minute boot
    that fails for a reason nothing in the log connects to the flag.

    The runtime that matches the wheel is right next to it, under
    ``site-packages/nvidia/*/lib``. Those directories go *first* on the child's
    loader path: a mismatched system runtime is exactly the failure being
    fixed, so it must not win.
    """
    import glob  # noqa: PLC0415
    import site  # noqa: PLC0415
    import sysconfig  # noqa: PLC0415

    roots: list[str] = []
    for base in (sysconfig.get_paths().get("purelib"), *site.getsitepackages()):
        if not base:
            continue
        roots.extend(sorted(glob.glob(os.path.join(base, "nvidia", "*", "lib"))))
    seen: dict[str, None] = {}
    for path in roots:
        if os.path.isdir(path):
            seen.setdefault(path, None)
    return os.pathsep.join(seen)
