"""Host-side split of a host-bound prefill segment (fnFL2 H38).

PLE-GATHER-PREFILL names the host wall of the PP0 pread gather (262144 rows per
16384-token chunk, 32 threads, one ``preadv`` per row from Python) and nothing
else. When that wall doubles the line cannot say why: this rig runs the boot's
host threads next to agents' tests and builds, and the PLE shards sit on ZFS
behind a 5 GiB ARC, so a slow gather is either CPU contention (the worker
threads are runnable but get no core -- or wait for a GIL whose holder got
none), a cold read (ARC miss -> disk), or the stream sync in front of it.
x144 (24.09.): chunk 5 3107.9 ms against 1098-1289 ms for chunks 1-4 of the
same request, and the wall of the P request 38.53 s against 35.86 s (x143)
-- with no instrument that separates these terms.

This module takes two cheap snapshots around the segment (``sample``) and
reports the difference (``delta``):

* host CPU busy / this process / FOREIGN cores (``/proc/stat`` minus
  ``/proc/self/stat``) -- how many cores something else burned meanwhile;
* PSI ``some`` stall of CPU and IO (``/proc/pressure``);
* ZFS ARC demand-data hits and misses (``/proc/spl/kstat/zfs/arcstats``);

and a per-thread pair (``thread_sched``) from ``/proc/thread-self/schedstat``:
on-CPU ns and RUN-QUEUE ns (runnable, no core) -- the direct measure of CPU
contention of the gather's own workers. What a worker spends neither on a core
nor in the run queue is blocked: IO or the GIL.

Every reader degrades to -1 when its file does not exist (non-Linux, no ZFS,
no PSI), so the probe never raises inside a forward. Cost: ~6 small procfs
reads per snapshot, two per worker task -- well under 1 ms against a 1.1-s
gather, and only while the timing switch is on.
"""

from __future__ import annotations

import os
import time
from typing import Optional, Tuple

import msgspec

try:
    _CLK_TCK = float(os.sysconf("SC_CLK_TCK"))
except (ValueError, OSError, AttributeError):  # pragma: no cover - non-POSIX
    _CLK_TCK = 100.0


class ProcPaths(msgspec.Struct, frozen=True):
    """Where the snapshot reads; overridable so the parsers can be desk-tested."""

    stat: str = "/proc/stat"
    self_stat: str = "/proc/self/stat"
    psi_cpu: str = "/proc/pressure/cpu"
    psi_io: str = "/proc/pressure/io"
    arcstats: str = "/proc/spl/kstat/zfs/arcstats"
    thread_schedstat: str = "/proc/thread-self/schedstat"


DEFAULT_PATHS = ProcPaths()


class HostSample(msgspec.Struct, frozen=True):
    """One snapshot. Counters are cumulative; -1 = not available here."""

    t: float  # time.monotonic() seconds
    host_busy: int  # jiffies, all CPUs, non-idle
    host_total: int  # jiffies, all CPUs, idle + iowait included
    self_cpu: int  # jiffies, utime + stime of this process (all threads)
    psi_cpu_us: int  # /proc/pressure/cpu "some" total
    psi_io_us: int  # /proc/pressure/io "some" total
    arc_hits: int  # ZFS demand_data_hits
    arc_misses: int  # ZFS demand_data_misses


class HostDelta(msgspec.Struct, frozen=True):
    """What happened on the host between two snapshots."""

    wall_ms: float
    host_busy_cores: float  # -1.0 = unknown
    self_cores: float
    foreign_cores: float  # host busy minus this process, >= 0
    psi_cpu_ms: float  # -1.0 = unknown
    psi_io_ms: float
    arc_hits: int  # -1 = unknown (no ZFS)
    arc_misses: int


def _read(path: str) -> Optional[str]:
    try:
        with open(path, "r") as f:
            return f.read()
    except OSError:
        return None


def parse_proc_stat(text: Optional[str]) -> Tuple[int, int]:
    """``(busy, total)`` jiffies of the aggregate ``cpu`` line, (-1, -1) if absent.

    busy = user + nice + system + irq + softirq + steal (guest time is already
    inside user/nice); total = busy + idle + iowait."""
    if not text:
        return -1, -1
    for line in text.splitlines():
        if line.startswith("cpu "):
            v = [int(x) for x in line.split()[1:]]
            v += [0] * (8 - len(v))
            user, nice, system, idle, iowait, irq, softirq, steal = v[:8]
            busy = user + nice + system + irq + softirq + steal
            return busy, busy + idle + iowait
    return -1, -1


def parse_self_stat(text: Optional[str]) -> int:
    """utime + stime (jiffies) from ``/proc/<pid>/stat``, -1 if unreadable.

    The comm field may hold spaces and parentheses, so the fields are counted
    from the LAST ')': after it come state (field 3) ... utime (14), stime (15).
    """
    if not text or ")" not in text:
        return -1
    rest = text[text.rindex(")") + 1 :].split()
    try:
        return int(rest[11]) + int(rest[12])
    except (IndexError, ValueError):
        return -1


def parse_psi_some_total(text: Optional[str]) -> int:
    """The ``total=`` (us) of the ``some`` line of a PSI file, -1 if absent."""
    if not text:
        return -1
    for line in text.splitlines():
        if line.startswith("some "):
            for tok in line.split():
                if tok.startswith("total="):
                    try:
                        return int(tok[6:])
                    except ValueError:
                        return -1
    return -1


def parse_arcstats(text: Optional[str]) -> Tuple[int, int]:
    """``(demand_data_hits, demand_data_misses)``, (-1, -1) without ZFS."""
    if not text:
        return -1, -1
    hits = misses = -1
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        if parts[0] == "demand_data_hits":
            hits = int(parts[2])
        elif parts[0] == "demand_data_misses":
            misses = int(parts[2])
    return hits, misses


def parse_schedstat(text: Optional[str]) -> Tuple[int, int]:
    """``(on_cpu_ns, run_queue_ns)`` of ``/proc/thread-self/schedstat``, (-1, -1) if absent."""
    if not text:
        return -1, -1
    parts = text.split()
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return -1, -1


def sample(paths: ProcPaths = DEFAULT_PATHS) -> HostSample:
    busy, total = parse_proc_stat(_read(paths.stat))
    hits, misses = parse_arcstats(_read(paths.arcstats))
    return HostSample(
        t=time.monotonic(),
        host_busy=busy,
        host_total=total,
        self_cpu=parse_self_stat(_read(paths.self_stat)),
        psi_cpu_us=parse_psi_some_total(_read(paths.psi_cpu)),
        psi_io_us=parse_psi_some_total(_read(paths.psi_io)),
        arc_hits=hits,
        arc_misses=misses,
    )


def thread_sched(paths: ProcPaths = DEFAULT_PATHS) -> Tuple[int, int]:
    """On-CPU and run-queue ns of the CALLING thread so far."""
    return parse_schedstat(_read(paths.thread_schedstat))


def _d(a: int, b: int) -> int:
    return b - a if a >= 0 and b >= 0 else -1


def delta(a: HostSample, b: HostSample) -> HostDelta:
    dt = max(b.t - a.t, 1e-9)
    busy = _d(a.host_busy, b.host_busy)
    own = _d(a.self_cpu, b.self_cpu)
    host_cores = busy / _CLK_TCK / dt if busy >= 0 else -1.0
    self_cores = own / _CLK_TCK / dt if own >= 0 else -1.0
    if host_cores >= 0 and self_cores >= 0:
        foreign = max(0.0, host_cores - self_cores)
    else:
        foreign = -1.0
    psi_cpu = _d(a.psi_cpu_us, b.psi_cpu_us)
    psi_io = _d(a.psi_io_us, b.psi_io_us)
    return HostDelta(
        wall_ms=dt * 1000.0,
        host_busy_cores=host_cores,
        self_cores=self_cores,
        foreign_cores=foreign,
        psi_cpu_ms=psi_cpu / 1000.0 if psi_cpu >= 0 else -1.0,
        psi_io_ms=psi_io / 1000.0 if psi_io >= 0 else -1.0,
        arc_hits=_d(a.arc_hits, b.arc_hits),
        arc_misses=_d(a.arc_misses, b.arc_misses),
    )


class ThreadSplit(msgspec.Struct):
    """Summed over the worker tasks of one gather."""

    tasks: int = 0
    wall_ns: int = 0
    cpu_ns: int = 0
    runq_ns: int = 0
    unknown: int = 0  # tasks whose schedstat was unreadable

    def add(self, wall_ns: int, cpu_ns: int, runq_ns: int) -> None:
        self.tasks += 1
        self.wall_ns += wall_ns
        if cpu_ns < 0 or runq_ns < 0:
            self.unknown += 1
            return
        self.cpu_ns += cpu_ns
        self.runq_ns += runq_ns

    @property
    def blocked_ns(self) -> int:
        """Neither on a core nor runnable: IO or a lock (the GIL)."""
        if self.unknown:
            return -1
        return max(0, self.wall_ns - self.cpu_ns - self.runq_ns)

    def lead(self) -> str:
        """The largest of the three worker terms -- a name, not a verdict."""
        if self.unknown or not self.tasks:
            return "unknown"
        terms = {"cpu": self.cpu_ns, "runq": self.runq_ns, "blocked": self.blocked_ns}
        return max(terms, key=terms.get)


def timed_task(fn, *args, paths: ProcPaths = DEFAULT_PATHS) -> Tuple[int, int, int]:
    """Run ``fn(*args)`` on the calling (worker) thread; return its
    ``(wall_ns, on_cpu_ns, run_queue_ns)``."""
    c0, q0 = thread_sched(paths)
    t0 = time.monotonic_ns()
    fn(*args)
    t1 = time.monotonic_ns()
    c1, q1 = thread_sched(paths)
    return t1 - t0, _d(c0, c1), _d(q0, q1)
