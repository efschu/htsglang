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
import re
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
    #: #799: read the scheduler's published admission-wedge verdict. Passive:
    #: it opens a file, never the server under suspicion.
    wedge_probe: Callable[[], "object"] = None
    #: #799: the operator stop marker, if any. Returns a reason or None.
    operator_stop: Callable[[], Optional[str]] = None
    #: #799: append one line per alarm to a durable ledger.
    append_alarm: Callable[[str], None] = None
    #: #799: veto a restart that would boot a DIFFERENT configuration than
    #: the one that was running. Returns a reason, or None to allow.
    restart_drift: Callable[[], Optional[str]] = None
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


#: #799: the operator's "production is stopped" marker. Already honoured by
#: ``/root/bin/start-serving-30030.sh`` (which exits 3 when it exists), so the
#: legacy shell watchdog inherits the guard for free through its start script.
#: The turnkey path restarts via ``systemctl restart``, which does NOT run that
#: script -- so without this check, arming this watchdog would drive boots into
#: GPU windows that an operator had explicitly closed. That is the collision
#: the marker was created for (two strands restored into each other's window
#: and OOM-killed both boots).
OPERATOR_STOP_MARKER = "/spinning/PRODUCTION_STOPPED"

#: Rate limit for the "supervision suspended" line. The marker can sit for a
#: whole GPU window; one line every five minutes says so without burying the
#: journal.
_STOP_LOG_EVERY_S = 300.0

#: #799 alarm sink. The honest position first: there is NO push notification
#: channel on this box -- no mail relay, no ntfy, no webhook, nothing in
#: /root/bin -- and #604 did not build one either; its "alarm" is a greppable
#: ``TURNKEY-ALARM`` line in a journal nobody was reading on 2026-08-22.
#:
#: This file is the smallest thing that is genuinely better: one short,
#: append-only, dated line per alarm, at a fixed path, so an operator or an
#: agent starting a session can answer "did serving fail while I was away?"
#: by reading a handful of lines instead of grepping a 4.8 MB boot log. It is
#: a durable ledger, NOT a notification: nothing wakes anyone up. Building a
#: real out-of-band channel needs a transport decision that is not mine to
#: make, and is filed rather than guessed.
ALARM_LEDGER = "/spinning/SERVING_ALARM.log"


def _append_alarm(line: str, path: str = ALARM_LEDGER) -> None:
    """Append one alarm line. Never raises: an unwritable ledger must not
    take down the watchdog that is trying to report an outage."""
    try:
        with open(path, "a") as fh:
            fh.write(line.rstrip("\n") + "\n")
    except OSError as e:
        logger.warning("%s alarm ledger %s unwritable: %s", ALARM, path, e)


def argv_model_path(argv: Sequence[str]) -> Optional[str]:
    """The ``--model-path`` a restart WOULD boot, out of the lane's argv."""
    argv = list(argv or ())
    for i, tok in enumerate(argv):
        if tok == "--model-path" and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith("--model-path="):
            return tok.split("=", 1)[1]
    return None


#: Bytes of boot log to scan for the running instance's identity. The
#: ``server_args=`` line is emitted within the first few seconds of a boot, so
#: the head of the file is where it lives; reading the whole file would mean
#: pulling multi-megabyte logs into a watchdog tick.
_BOOT_LOG_HEAD_BYTES = 256 * 1024
_MODEL_PATH_RE = re.compile(r"model_path='([^']+)'")


def boot_log_model_path(path: str) -> Optional[str]:
    """The ``--model-path`` the instance that wrote this log ACTUALLY booted."""
    try:
        with open(path, "r", errors="replace") as fh:
            head = fh.read(_BOOT_LOG_HEAD_BYTES)
    except OSError:
        return None
    m = _MODEL_PATH_RE.search(head)
    return m.group(1) if m else None


def restart_target_drift(configured: Optional[str],
                         booted: Optional[str]) -> Optional[str]:
    """#799: refuse to "restore" service as a DIFFERENT service.

    Measured on 2026-08-22, and this is why the check exists rather than a
    hypothetical:

    * ``/etc/htsglang/stack.toml`` (15 Aug) boots
      ``Qwen3.8-27B-INT8-yarn1.5`` with ``--pp-stage-ratio 14,10,8``;
    * ``/root/bin/start-serving-30030.sh`` (8 Aug), which the legacy shell
      watchdog runs on every restart, boots ``Qwen3.6-27B-INT8-W8A8`` at
      ``--tp-size 3`` -- a superseded model AND a different parallel layout;
    * the instance actually running that morning was
      ``Qwen3.8-27B-INT8-vocabint8-embed`` at ``--pp-stage-ratio 32,18,14``.

    All three differ. A watchdog that can finally SEE an outage and then
    "recovers" it by booting a two-week-old configuration has not restored
    anything -- it has silently replaced the service while reporting success,
    which is strictly worse than the blindness it just fixed. Drift is
    therefore a hard veto on the restart, never a warning that scrolls past.

    ``None`` on either side means the comparison could not be made (no argv,
    no boot log yet, a first boot). That is NOT drift: refusing a restart
    because a lane has never booted would make the watchdog useless exactly
    when a lane is down. The veto needs positive evidence of disagreement.
    """
    if not configured or not booted:
        return None
    if configured == booted:
        return None
    return (f"restart target has DRIFTED: the configuration would boot "
            f"{configured!r}, but the instance that last ran was {booted!r}. "
            f"Restarting would replace the service, not restore it")


def _read_wedge(directory: Optional[str]):
    """Read the published wedge verdict, or ``None`` if it cannot be read.

    ``None`` is a legitimate answer and NOT an error path to be hidden: it is
    the "no measurement" state the state machine already knows how to hold.
    """
    try:
        from sglang.srt.managers import wedge_status

        return wedge_status.read_wedge_signal(directory)
    except Exception as e:  # noqa: BLE001 - a watchdog must not die on a read
        logger.warning("%s wedge-status read failed: %s", ALARM, e)
        return None


def operator_stop_reason(path: str = OPERATOR_STOP_MARKER) -> Optional[str]:
    """The reason production is stopped, or ``None`` if it is not.

    An unreadable-but-present marker still stops the watchdog: the file's
    EXISTENCE is the order, and its contents are only the explanation.
    """
    try:
        with open(path, "r") as fh:
            return fh.read().strip() or "no reason given"
    except FileNotFoundError:
        return None
    except OSError as e:
        return f"marker present but unreadable ({e})"


class WatchdogRunner:
    """One lane's supervision loop."""

    def __init__(self, unit: str, base_url: str, policy: W.Policy,
                 deps: Optional[RunnerDeps] = None,
                 generation_timeout_s: float = 60.0,
                 wedge_status_dir: Optional[str] = None,
                 lane_argv: Optional[Sequence[str]] = None,
                 boot_log: Optional[str] = None):
        self.unit = unit
        self.base_url = base_url.rstrip("/")
        self.policy = policy
        self.generation_timeout_s = generation_timeout_s
        self.wedge_status_dir = wedge_status_dir
        self.lane_argv = list(lane_argv or ())
        self.boot_log = boot_log
        d = deps or RunnerDeps()
        host, port = _split(self.base_url)
        d.api_probe = d.api_probe or (lambda: P.api_ok(self.base_url))
        d.gen_probe = d.gen_probe or (
            lambda: P.generation_ok(self.base_url,
                                    timeout=self.generation_timeout_s))
        d.port_probe = d.port_probe or (lambda: P.port_open(host, port))
        d.wedge_probe = d.wedge_probe or (lambda: _read_wedge(
            self.wedge_status_dir))
        d.operator_stop = d.operator_stop or operator_stop_reason
        d.append_alarm = d.append_alarm or _append_alarm
        d.restart_drift = d.restart_drift or (lambda: restart_target_drift(
            argv_model_path(self.lane_argv),
            boot_log_model_path(self.boot_log) if self.boot_log else None))
        d.restart_unit = d.restart_unit or _systemctl_restart
        d.log = d.log or _default_log
        self.deps = d
        self.state = W.initial(d.now(), policy)
        self._stop_logged_at: Optional[float] = None

    # -- one iteration ----------------------------------------------------

    def tick(self) -> W.Decision:
        """Probe, decide, act. Returns the decision for tests and logging."""
        d = self.deps
        now = d.now()

        # #799: an operator stop outranks every probe. Serving is MEANT to be
        # down, so neither DEAD nor WEDGED is a fault, and a restart would
        # walk a boot into a GPU window somebody closed on purpose. The state
        # is re-armed to BOOTING on every stopped tick so that lifting the
        # marker gives the lane its full boot grace instead of a watchdog that
        # convicts it for having been down.
        stop = d.operator_stop()
        if stop is not None:
            self.state = W.initial(now, self.policy)
            if (self._stop_logged_at is None
                    or now - self._stop_logged_at >= _STOP_LOG_EVERY_S):
                self._stop_logged_at = now
                d.log("info", f"[{self.unit}] supervision suspended: "
                              f"{OPERATOR_STOP_MARKER} exists ({stop})")
            return W.Decision(state=self.state, action=W.ACT_NONE,
                              reason=f"operator stop in force: {stop}")
        self._stop_logged_at = None

        api = d.api_probe()
        # #799: the passive wedge verdict. Read on EVERY tick, unconditionally
        # -- gating the read on api.ok would blind the watchdog in exactly the
        # case where the scheduler is alive and publishing while the HTTP
        # layer has gone, which is the second half of boot 0822_0829.
        wedged = None
        if self.policy.wedge_signal_enabled:
            sig = d.wedge_probe()
            wedged = getattr(sig, "verdict", None)
            if wedged or getattr(sig, "stale", False):
                d.log("info", f"wedge signal: {getattr(sig, 'detail', sig)}")
        obs = W.Observation(port_open=d.port_probe(), api_ok=api.ok,
                            generation=None, wedged=wedged)

        before = self.state
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

        if decision.action == W.ACT_RESTART:
            drift = self.deps.restart_drift()
            if drift is not None:
                # The state is rolled back to BEFORE the decision on purpose:
                # a restart we refused must not spend the restart budget, or a
                # drifted lane would walk itself to GIVEN_UP without a single
                # restart ever having been attempted.
                decision = W.Decision(state=before, action=W.ACT_ALARM,
                                      reason=f"{decision.reason} -- REFUSED: "
                                             f"{drift}")

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
        if decision.alarming:
            self.deps.append_alarm(
                f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {self.unit} "
                f"{decision.state.phase}: {decision.reason}")

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
