from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from contextlib import contextmanager
from multiprocessing import Process
from typing import Callable, List, Optional, Tuple

import psutil

from flliper.srt.environ import envs
from flliper.srt.utils.cudacore_pyspy_dump_utils import pyspy_dump_schedulers

logger = logging.getLogger(__name__)


class Watchdog:
    @staticmethod
    def create(
        debug_name: str,
        watchdog_timeout: Optional[float],
        soft: bool = False,
        test_stuck_time: float = 0,
    ) -> Watchdog:
        if watchdog_timeout is None:
            assert (
                test_stuck_time == 0
            ), f"stuck tester can be enabled only if soft watchdog is enabled."
            return _WatchdogNoop()
        return _WatchdogReal(
            debug_name=debug_name,
            watchdog_timeout=watchdog_timeout,
            soft=soft,
            test_stuck_time=test_stuck_time,
        )

    def feed(self):
        pass

    @contextmanager
    def disable(self):
        yield


class _WatchdogReal(Watchdog):
    def __init__(
        self,
        debug_name: str,
        watchdog_timeout: float,
        soft: bool = False,
        test_stuck_time: float = 0,
    ):
        self._counter = 0
        self._active = True
        self._test_stuck_time = test_stuck_time
        self._test_stuck_triggered = False
        self._raw = WatchdogRaw(
            debug_name=debug_name,
            get_counter=lambda: self._counter,
            is_active=lambda: self._active,
            watchdog_timeout=watchdog_timeout,
            soft=soft,
        )
        logger.info(f"Watchdog {self._raw.debug_name} initialized.")
        if self._test_stuck_time > 0:
            logger.info(
                f"Watchdog {self._raw.debug_name} is configured to use {test_stuck_time=}."
            )

    def feed(self):
        # Only trigger the test stuck behavior once to avoid blocking server
        # startup health checks while still testing watchdog timeout detection
        if self._test_stuck_time > 0 and not self._test_stuck_triggered:
            self._test_stuck_triggered = True
            logger.info(
                f"Watchdog {self._raw.debug_name} start deliberately stuck for {self._test_stuck_time}s"
            )
            time.sleep(self._test_stuck_time)
            logger.info(
                f"Watchdog {self._raw.debug_name} end deliberately stuck for {self._test_stuck_time}s"
            )

        self._counter += 1

    @contextmanager
    def disable(self):
        assert self._active
        self._active = False
        try:
            yield
        finally:
            assert not self._active
            self._active = True


class _WatchdogNoop(Watchdog):
    pass


def compose_timeout_line(
    debug_name: str, watchdog_timeout: float, soft: bool, arm: str = ""
) -> str:
    """The line an operator reads first when a rank dies of a wedge.

    #824 W5(a). It used to carry only the name, the budget and the soft
    flag; on boot_827 that meant the one honest wedge report on the whole
    boot said nothing about WHICH wait had stopped the ring, and the answer
    had to be dug out of a py-spy dump taken minutes later. ``arm`` is
    appended only when the caller could actually name it, so a watchdog
    with no describer is worded exactly as before.

    Kept a plain function so the wording can be tested without arming a
    real watchdog thread and its SIGQUIT.
    """
    line = (
        f"{debug_name} watchdog timeout "
        f"(self.watchdog_timeout={watchdog_timeout!r}, self.soft={soft!r})"
    )
    return line + (f" tripped_by={arm}" if arm else "")


class WatchdogRaw:
    def __init__(
        self,
        debug_name: str,
        get_counter: Callable[[], int],
        is_active: Callable[[], bool],
        watchdog_timeout: float,
        soft: bool = False,
        dump_info: Optional[Callable[[], str]] = None,
        describe_arm: Optional[Callable[[], str]] = None,
        stall_age: Optional[Callable[[], float]] = None,
    ):
        self.debug_name = debug_name
        self.get_counter = get_counter
        self.is_active = is_active
        self.watchdog_timeout = watchdog_timeout
        self.soft = soft
        self.dump_info = dump_info
        #: #824 W5(a): name the arm that tripped, in the trigger line
        #: itself. On boot_827 the watchdog fired correctly and the line
        #: said only "Scheduler watchdog timeout (...)" -- which arm had
        #: gone overdue was left to be reconstructed from a py-spy dump
        #: taken minutes later. Returning "" keeps the old wording.
        self.describe_arm = describe_arm
        #: H86b: for an arm that switches is_active on LATE -- only after its
        #: stall has already lasted a while (#821 pp_receive_is_overdue arms
        #: once a PP receive has blocked longer than the timeout). Returns the
        #: seconds the stall had already lasted, as a DURATION: only a
        #: difference crosses this seam, never a timestamp, so the callee may
        #: measure in its own clock (time.monotonic) and this class subtracts
        #: it from its own (time.perf_counter) without mixing epochs.
        self.stall_age = stall_age

        self.parent_process = psutil.Process().parent()
        t = threading.Thread(target=self._watchdog_thread, daemon=True)
        t.start()

    def _watchdog_thread(self):
        try:
            while True:
                self._watchdog_once()
        except Exception as e:
            logger.error(
                f"{self.debug_name} watchdog thread crashed: {e}", exc_info=True
            )

    #: Class default so an instance built without __init__ (the H86 test does
    #: that) keeps the plain H86 behaviour.
    stall_age: Optional[Callable[[], float]] = None

    def _read_stall_age(self) -> float:
        """The late-arming arm's own stall age in seconds, 0.0 if unknown.

        A raising callback must never escape: _watchdog_thread would log
        "watchdog thread crashed" and RETURN, i.e. the process would lose its
        watchdog for good (#821's dump_info lesson).
        """
        try:
            age = self.stall_age()
        except Exception as e:  # noqa: BLE001 - the watchdog must survive
            logger.error(f"{self.debug_name} watchdog stall_age failed: {e}")
            return 0.0
        if age is None:
            return 0.0
        try:
            age = float(age)
        except (TypeError, ValueError):
            return 0.0
        return age if age > 0.0 else 0.0

    def _watchdog_once(self):
        watchdog_last_counter = 0
        watchdog_last_time = time.perf_counter()

        while True:
            current = time.perf_counter()
            if self.is_active():
                current_counter = self.get_counter()
                if watchdog_last_counter == current_counter:
                    if self.stall_age is not None:
                        # H86b: the H86 reset below forgot the stall's past
                        # while the arm was still dark, so #821 (dark for the
                        # first T of a blocked receive) fired only after 2T-2.5T.
                        # Credit the age the arm itself measured.
                        watchdog_last_time = min(
                            watchdog_last_time, current - self._read_stall_age()
                        )
                    if current > watchdog_last_time + self.watchdog_timeout:
                        break
                else:
                    watchdog_last_counter = current_counter
                    watchdog_last_time = current
            else:
                # H86: idle is not a stall. The clock only ran while is_active(), so a scheduler that slept past the
                # timeout carried its sleep into the first active comparison: fnNV4f4 P was dormant 293 s, PP1/PP2
                # set cur_batch on the wake and waited seconds for PP0's proxy tensors before forward_ct moved, and
                # the hard watchdog tripped 7 s into the needle (SIGQUIT, whole P group torn down). A real stall --
                # active with a frozen counter -- is still caught one timeout after it begins.
                watchdog_last_time = current
                if self.stall_age is not None:
                    # H86b: keep the counter known through the dark phase, so
                    # the first active sample of a late arm lands in the
                    # "counter stands" branch above and trips at once instead
                    # of spending one more half-period re-learning a counter
                    # that last moved before the stall. Gated: without
                    # stall_age this loop is exactly H86.
                    watchdog_last_counter = self.get_counter()
            time.sleep(self.watchdog_timeout / 2)

        if self.dump_info is not None and (info_msg := self.dump_info()):
            logger.error(f"{self.debug_name} debug info:\n{info_msg}")

        pyspy_dump_schedulers()
        arm = ""
        if self.describe_arm is not None:
            try:
                arm = self.describe_arm() or ""
            except Exception as e:  # noqa: BLE001 - diagnostics must not raise
                # This runs microseconds before SIGQUIT. A broken describer
                # must cost the arm name, never the death notice itself.
                arm = f"<arm describer failed: {e}>"
        logger.error(
            compose_timeout_line(
                self.debug_name, self.watchdog_timeout, self.soft, arm
            )
        )
        print(file=sys.stderr, flush=True)
        print(file=sys.stdout, flush=True)

        if not self.soft:
            # deskq 52: the hard stop has its own switch (default on = the
            # kill below, as ever). Off: the dump and the line above stay, the
            # parent is not signalled; the thread goes on watching, exactly
            # like the soft watchdog.
            if not envs.FLLIPER_ENABLE_SCHEDULER_WATCHDOG_KILL.get():
                logger.error(
                    f"{self.debug_name} watchdog: SIGQUIT to the parent process "
                    "suppressed (FLLIPER_ENABLE_SCHEDULER_WATCHDOG_KILL=0)"
                )
                return
            # Wait for some time so that the parent process can print the error.
            time.sleep(5)
            self.parent_process.send_signal(signal.SIGQUIT)


def _cgroup_memory_dir() -> Optional[str]:
    """The cgroup whose memory limit this process actually runs under.

    Inside an LXC container the namespaced root IS the container, so
    /sys/fs/cgroup/.lxc is where the serving processes are accounted even
    though the limit itself is set on the host side and reads as "max" here.
    """
    for path in ("/sys/fs/cgroup/.lxc", "/sys/fs/cgroup"):
        if os.path.exists(os.path.join(path, "memory.events")):
            return path
    return None


def _oom_kill_count() -> Optional[int]:
    """Cumulative cgroup OOM kills, or None where cgroup v2 is not readable.

    This counter is the ONLY in-container trace of a kernel OOM kill: the
    kernel's own report goes to the ring buffer, and a container has neither
    /dev/kmsg nor permission for dmesg. The counter carries no timestamp and
    no victim, so it is only evidence when a BEFORE value was taken -- which
    is why the watchdog samples one at construction.
    """
    path = _cgroup_memory_dir()
    if path is None:
        return None
    try:
        with open(os.path.join(path, "memory.events")) as fh:
            for line in fh:
                if line.startswith("oom_kill "):
                    return int(line.split()[1])
    except (OSError, ValueError):
        return None
    return None


def _host_memory_state() -> str:
    """One line of host-side memory state, for the report at a death."""
    parts = []
    path = _cgroup_memory_dir()
    if path is not None:
        try:
            with open(os.path.join(path, "memory.current")) as fh:
                parts.append(f"cgroup memory.current={int(fh.read()) // 1048576} MiB")
        except (OSError, ValueError):
            pass
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    parts.append(f"MemAvailable={int(line.split()[1]) // 1024} MiB")
                    break
    except (OSError, ValueError):
        pass
    return ", ".join(parts) if parts else "host memory state unavailable"


def _proc_state(pid: Optional[int]) -> Optional[str]:
    """The one-letter state of ``pid`` from /proc (``Z`` = zombie leader),
    None when unreadable or gone. Never raises."""
    if pid is None:
        return None
    try:
        with open(f"/proc/{int(pid)}/stat") as fh:
            return fh.read().rsplit(")", 1)[-1].split()[0]
    except (OSError, ValueError, IndexError):
        return None


def _dying(proc) -> bool:
    """Leader already a zombie, exit not yet reportable. Never raises."""
    try:
        return _proc_state(proc.pid) == "Z" and proc.is_alive()
    except Exception:  # noqa: BLE001 - a foreign-parent Process
        return False


def describe_exit(exitcode: Optional[int]) -> str:
    """Render a multiprocessing exitcode as something a human can act on.

    A negative exitcode is `-signum`, which in a log reads as an arbitrary
    small negative number. Naming the signal is the difference between "the
    rank died silently" and "the kernel killed the rank".
    """
    if exitcode is None:
        return "still running (exit code None)"
    if exitcode < 0:
        try:
            name = signal.Signals(-exitcode).name
        except ValueError:
            name = f"signal {-exitcode}"
        return f"killed by {name} (exit code {exitcode})"
    return f"exit code {exitcode}"


class SubprocessWatchdog:
    """Monitors subprocess liveness and triggers SIGQUIT when a crash is detected.

    When a subprocess crashes (e.g., NCCL timeout causing C++ std::terminate()),
    Python exception handlers never run, leaving the main process as a zombie
    service. This watchdog polls subprocess liveness in a daemon thread and
    sends SIGQUIT to trigger proper cleanup.

    See: https://github.com/sgl-project/sglang/issues/18421

    #485/C40 -- WHAT THE WATCHDOG OWES THE OPERATOR. This class is the only
    continuously-running observer of rank liveness in the launcher, so it is
    the one place that can answer "how did the child die". It previously
    answered with a bare integer, skipped `exitcode == 0` without a word, and
    returned at the first dead process. A rank that was SIGKILLed mid-decode
    was consequently written up as a SILENT DEATH for a full shift, while the
    line `crashed with exit code -9` sat in the log. The reporting below is
    therefore deliberately verbose; the SIGQUIT policy is deliberately
    unchanged.
    """

    def __init__(
        self,
        processes: List[Process],
        process_names: Optional[List[str]] = None,
        interval: float = 1.0,
    ):
        self._processes = processes
        self._names = process_names or [f"process_{i}" for i in range(len(processes))]
        self._interval = interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Sampled BEFORE anything can die, so a later reading is a difference
        # rather than an absolute nobody can interpret.
        self._oom_kills_at_start = _oom_kill_count()
        self._reported: set = set()
        # deskq 52: the suppressed-kill line is written once, not every poll.
        self._kill_suppressed_logged = False

    def processes_with_names(self) -> List[Tuple[Process, str]]:
        """Tracked (process, name) pairs, for external liveness checks."""
        return list(zip(self._processes, self._names))

    def start(self) -> None:
        if self._thread is not None or not self._processes:
            return
        self._thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="subprocess-watchdog"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval * 2)
            self._thread = None

    def _monitor_loop(self) -> None:
        try:
            while not self._stop_event.wait(self._interval):
                if self._check_processes():
                    return
        except Exception as e:
            logger.error(f"SubprocessWatchdog thread crashed: {e}", exc_info=True)

    def _oom_attribution(self) -> str:
        """Whether the kernel OOM-killed anything in this cgroup since start.

        A SIGKILL is never sent to a healthy rank by this codebase, so when one
        appears the question is only who sent it. On this rig the answer is
        almost always the kernel's cgroup OOM killer, and the report it writes
        is unreadable from inside a container -- so the counter delta is the
        evidence, and it is worth stating even when it is zero.
        """
        now = _oom_kill_count()
        before = self._oom_kills_at_start
        if now is None or before is None:
            return (
                "cgroup OOM counter unavailable, so an OOM kill can be neither "
                "confirmed nor excluded from inside this container"
            )
        if now > before:
            return (
                f"THE KERNEL OOM-KILLED A PROCESS IN THIS CGROUP: "
                f"{_cgroup_memory_dir()}/memory.events oom_kill went "
                f"{before} -> {now} since this server started. This is a HOST "
                f"memory event; GPU free memory is irrelevant to it"
            )
        return (
            f"no cgroup OOM kill recorded ({_cgroup_memory_dir()}/memory.events "
            f"oom_kill still {now}), so the SIGKILL came from outside the "
            f"kernel's OOM killer"
        )

    def _report_death(self, proc: Process, name: str) -> None:
        """Say how this child ended, once, in terms someone can act on."""
        if proc.pid in self._reported:
            return
        self._reported.add(proc.pid)

        detail = describe_exit(proc.exitcode)
        lines = [f"Subprocess {name} (pid={proc.pid}) is gone: {detail}."]
        if proc.exitcode is not None and proc.exitcode < 0:
            lines.append(f"  host memory at detection: {_host_memory_state()}")
            if proc.exitcode == -int(signal.SIGKILL):
                lines.append(f"  {self._oom_attribution()}")
        elif proc.exitcode == 0:
            lines.append(
                "  A long-running rank that returns cleanly has still stopped "
                "serving. Not signalling on it (unchanged policy), but it is "
                "recorded here rather than passed over in silence."
            )
        logger.error("\n".join(lines))

    #: report_dead_children: bounded wait for a zombie-leader child's exit
    DYING_WAIT_S = 2.0

    def report_dead_children(self, why: str) -> int:
        """EXIT WITNESS: report every child that is already gone, NOW.

        NF 20.10. 14:18:29Z (rc26, D TP0): the kernel's cgroup OOM killer
        SIGKILLed D rank 0 at 14:18:29.514; TP1/TP2 died of the peer-gone
        consequence ~0.5 s later and SIGQUIT the parent. This watchdog polls
        once per ``interval`` (1 s), so the peers' SIGQUIT arrived BEFORE its
        next poll -- and the SIGQUIT handler's first act was ``stop()``. The
        one observer that could have printed ``killed by SIGKILL ... oom_kill
        0 -> 1`` never polled again, and the rank's death read as SILENT for
        a whole investigation (the kernel's own report sits in the host
        journal, not in any boot log).

        So the handler calls THIS before it stops the watch: one synchronous
        sweep, every child, the same ``_report_death`` text the poll would
        have written (``is_alive`` reaps a zombie child, so the exit code of
        a rank killed a moment ago is there to be read). A child already
        reported by the poll is not reported twice. Never raises; returns the
        number of children found dead (0 is printed too, because "SIGQUIT
        with every child alive" is itself a finding).
        """
        dead = 0
        try:
            # A SIGKILLed process whose leader is already a zombie ("Z" in
            # /proc) can still have threads finishing their exit -- a rank
            # with ~250 GB of mappings tears down for a while -- and until the
            # last one is gone waitpid answers "still running". Measured in
            # this ticket's test: leader "Z", waitpid (0, 0), is_alive() True.
            # Give such DYING children a bounded moment (DYING_WAIT_S) so the
            # sweep can read their exit code, and name any still dying.
            deadline = time.monotonic() + self.DYING_WAIT_S
            while time.monotonic() < deadline:
                if not any(_dying(p) for p in self._processes):
                    break
                time.sleep(0.02)
            for proc, name in zip(self._processes, self._names):
                try:
                    alive = proc.is_alive()
                except Exception:  # noqa: BLE001 - a foreign-parent Process
                    continue
                if alive:
                    if _proc_state(proc.pid) == "Z":
                        dead += 1
                        logger.error(
                            "Subprocess %s (pid=%s) is DYING: its leader is a "
                            "zombie but the exit is not reportable yet (threads "
                            "still tearing down after %.1f s).\n  %s",
                            name, proc.pid, self.DYING_WAIT_S,
                            self._oom_attribution(),
                        )
                    continue
                dead += 1
                self._report_death(proc, name)
            if dead == 0:
                logger.error(
                    "EXIT-WITNESS at %s: all %d tracked children still alive "
                    "(the signal did not come from a child that already exited)",
                    why, len(self._processes),
                )
            else:
                logger.error(
                    "EXIT-WITNESS at %s: %d of %d tracked children gone (each "
                    "named above with its exit code; a negative code is the "
                    "signal that ended it)",
                    why, dead, len(self._processes),
                )
        except Exception:  # noqa: BLE001 - a witness may not mask the stop
            logger.error("EXIT-WITNESS at %s: sweep failed", why, exc_info=True)
        return dead

    def _check_processes(self) -> bool:
        # Report EVERY dead child before acting on any of them: when one rank
        # dies its peers follow within seconds, and a record naming only the
        # first invites the consequence to be mistaken for the cause.
        crashed = False
        for proc, name in zip(self._processes, self._names):
            if proc.is_alive():
                continue
            self._report_death(proc, name)
            if proc.exitcode != 0:
                crashed = True

        if crashed and not envs.FLLIPER_ENABLE_SUBPROCESS_WATCHDOG_KILL.get():
            # deskq 52: the stop has its own switch (default on). Off: every
            # death above is reported, nothing is signalled, the watch goes on.
            if not self._kill_suppressed_logged:
                self._kill_suppressed_logged = True
                logger.error(
                    "A subprocess crashed; SIGQUIT for cleanup suppressed "
                    "(FLLIPER_ENABLE_SUBPROCESS_WATCHDOG_KILL=0)"
                )
            return False

        if crashed:
            logger.error("Triggering SIGQUIT for cleanup...")
            os.kill(os.getpid(), signal.SIGQUIT)
            return True
        return False
