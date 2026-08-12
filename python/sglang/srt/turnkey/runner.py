# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""The #604 watchdog daemon: probes in, ``systemctl restart`` out.

All policy lives in :mod:`~sglang.srt.turnkey.watchdog` as a pure function.
This module is the thin, impure shell around it: it reads the clock, opens
sockets, writes log lines and asks systemd to restart a unit. Keeping the
split sharp is what let the state machine be tested exhaustively without a
server.

**The restart action is `systemctl restart`, and that is a safety property,
not a stylistic choice.** #638: a serving process started by the watchdog
inherits the watchdog's cgroup (``setsid`` leaves the session, not the
cgroup), so stopping or restarting the watchdog killed live production with
it. Delegating the restart to systemd means the replacement process is
started by pid 1 into the serving unit's own cgroup, and the watchdog's own
lifecycle can no longer reach it.

**Orphans are handled by PID, never by pattern.** ``pkill -f sglang`` also
matches the router on :30099, whose liveness is a standing law on this rig.
Every signal this module sends goes to a pid that NVML named as holding
memory on a card we own, after that pid's cmdline was checked.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import signal
import subprocess
import time
from typing import Callable, Dict, List, Optional, Sequence

from sglang.srt.turnkey import probe as P
from sglang.srt.turnkey import watchdog as W

__all__ = ["RunnerDeps", "WatchdogRunner", "orphan_pids", "reap_orphans"]

logger = logging.getLogger("turnkey.watchdog")

#: Prefix on every line an operator or a log filter should be able to grep.
ALARM = "TURNKEY-ALARM"


@dataclasses.dataclass
class RunnerDeps:
    """Injected I/O. Defaults reach the real machine."""

    now: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    api_probe: Callable[[], P.ProbeResult] = None
    gen_probe: Callable[[], P.ProbeResult] = None
    port_probe: Callable[[], bool] = None
    restart_unit: Callable[[str], bool] = None
    log: Callable[[str, str], None] = None


def _systemctl_restart(unit: str) -> bool:
    """Ask systemd to restart the unit. Never spawns serving directly.

    ``reset-failed`` first, and it is not defensive noise: a unit that has
    tripped its own ``StartLimitBurst`` enters the ``failed`` state and
    systemd then REFUSES every subsequent ``restart`` until the counter is
    cleared. Without this the watchdog would keep issuing restarts that
    silently do nothing, and its own backoff ladder -- which is the layer
    that is supposed to be rate-limiting here -- would never get to act.
    Two rate limiters in series, only one of them visible, is how a lane
    stays down while the log says "restart requested".
    """
    try:
        subprocess.run(["systemctl", "reset-failed", unit],
                       capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        pass  # best effort; the restart below reports the real outcome
    try:
        r = subprocess.run(["systemctl", "restart", unit],
                           capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.error("%s systemctl restart %s failed: %s", ALARM, unit, e)
        return False
    if r.returncode != 0:
        logger.error("%s systemctl restart %s rc=%s stderr=%s", ALARM, unit,
                     r.returncode, (r.stderr or "").strip()[:400])
        return False
    return True


def _default_log(level: str, msg: str) -> None:
    getattr(logger, level, logger.info)(msg)


class WatchdogRunner:
    """One lane's supervision loop."""

    def __init__(self, unit: str, base_url: str, policy: W.Policy,
                 deps: Optional[RunnerDeps] = None,
                 generation_timeout_s: float = 60.0):
        self.unit = unit
        self.base_url = base_url.rstrip("/")
        self.policy = policy
        self.generation_timeout_s = generation_timeout_s
        d = deps or RunnerDeps()
        host, port = _split(self.base_url)
        d.api_probe = d.api_probe or (lambda: P.api_ok(self.base_url))
        d.gen_probe = d.gen_probe or (
            lambda: P.generation_ok(self.base_url,
                                    timeout=self.generation_timeout_s))
        d.port_probe = d.port_probe or (lambda: P.port_open(host, port))
        d.restart_unit = d.restart_unit or _systemctl_restart
        d.log = d.log or _default_log
        self.deps = d
        self.state = W.initial(d.now(), policy)

    # -- one iteration ----------------------------------------------------

    def tick(self) -> W.Decision:
        """Probe, decide, act. Returns the decision for tests and logging."""
        d = self.deps
        now = d.now()
        api = d.api_probe()
        obs = W.Observation(port_open=d.port_probe(), api_ok=api.ok,
                            generation=None)

        decision = W.step(self.state, obs, now, self.policy)

        if decision.action == W.ACT_PROBE_GENERATION:
            # Spend the expensive probe, then re-decide with the verdict in
            # hand. The state is advanced ONCE, by the second call: the first
            # decision only told us a probe was owed.
            g = d.gen_probe()
            d.log("info", f"generation probe: ok={g.ok} {g.detail}")
            decision = W.step(self.state,
                              W.Observation(obs.port_open, obs.api_ok, g.ok),
                              d.now(), self.policy)

        self._emit(decision)
        self.state = decision.state

        if decision.action == W.ACT_RESTART:
            ok = d.restart_unit(self.unit)
            d.log("error" if not ok else "warning",
                  f"{ALARM} restart of {self.unit} "
                  f"{'requested' if ok else 'FAILED'}")
        return decision

    def _emit(self, decision: W.Decision) -> None:
        level = "error" if decision.alarming else "info"
        prefix = f"{ALARM} " if decision.alarming else ""
        self.deps.log(level, f"{prefix}[{self.unit}] {decision.state.phase}: "
                             f"{decision.reason}")

    def run(self, max_ticks: Optional[int] = None) -> None:
        """Loop forever, or ``max_ticks`` times (tests, dry runs)."""
        n = 0
        while max_ticks is None or n < max_ticks:
            try:
                self.tick()
            except Exception:  # a watchdog that dies on a probe bug is worse
                logger.exception("%s watchdog tick raised; continuing", ALARM)
            n += 1
            if max_ticks is None or n < max_ticks:
                self.deps.sleep(self.policy.poll_s)


def _split(base_url: str):
    rest = base_url.split("://", 1)[-1]
    hostport = rest.split("/", 1)[0]
    if ":" in hostport:
        h, p = hostport.rsplit(":", 1)
        return h, int(p)
    return hostport, 80


# --- orphan handling ------------------------------------------------------


def _cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            return fh.read().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return ""


def orphan_pids(card_uuids: Sequence[str], *, protect: Sequence[int] = (),
                marker: str = "sglang.launch_server",
                procs_on=None, cmdline=_cmdline) -> Dict[int, str]:
    """Pids holding VRAM on our cards that look like a stale serving process.

    Three filters, all required, in this order:

    1. NVML says the pid holds memory on a card THIS stack owns. A pattern
       match alone would find processes on cards we were never given.
    2. Its cmdline contains the serving marker. The router also imports
       sglang, so ``sglang`` alone is not a discriminator -- and killing the
       router is prohibited.
    3. It is not in ``protect`` (our own pid, the router's, anything the
       caller names).

    Returns ``{pid: cmdline}`` so the caller logs WHAT it is about to signal
    before signalling it.
    """
    if procs_on is None:
        from sglang.srt.registry import nvml
        procs_on = nvml.process_bytes_on_uuid

    protected = set(protect) | {os.getpid(), os.getppid()}
    found: Dict[int, str] = {}
    for uuid in card_uuids:
        try:
            for pid in procs_on(uuid):
                if pid in protected or pid in found:
                    continue
                cl = cmdline(pid)
                if marker and marker not in cl:
                    continue
                found[pid] = cl
        except Exception as e:
            logger.warning("orphan scan on %s failed: %s", uuid, e)
    return found


def reap_orphans(pids: Dict[int, str], *, dry_run: bool = True,
                 grace_s: float = 15.0, kill=os.kill,
                 sleep=time.sleep) -> List[str]:
    """SIGTERM, wait, SIGKILL -- by explicit pid, one at a time.

    ``dry_run`` defaults to True: the destructive form must be asked for.
    """
    lines: List[str] = []
    if not pids:
        return ["no orphans"]
    for pid, cl in sorted(pids.items()):
        lines.append(f"orphan pid {pid}: {cl[:160]}")
    if dry_run:
        lines.append(f"dry-run: would SIGTERM {sorted(pids)}")
        return lines

    for pid in sorted(pids):
        try:
            kill(pid, signal.SIGTERM)
            lines.append(f"SIGTERM -> {pid}")
        except OSError as e:
            lines.append(f"SIGTERM -> {pid} failed: {e}")
    sleep(grace_s)
    for pid in sorted(pids):
        try:
            kill(pid, 0)
        except OSError:
            lines.append(f"pid {pid} gone")
            continue
        try:
            kill(pid, signal.SIGKILL)
            lines.append(f"SIGKILL -> {pid}")
        except OSError as e:
            lines.append(f"SIGKILL -> {pid} failed: {e}")
    return lines
