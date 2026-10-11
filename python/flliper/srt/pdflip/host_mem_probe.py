"""Host memory-stall facts from /proc, for log lines only (D-HEALTH, 10.10.).

The D tokenizer took ~41 s for a 4096x4096 image whose CPU work measures
0.6 s; the suspicion is a memory stall (THP compaction, reclaim, swap). These
readers put the evidence beside the lines that report the stall. They never
raise: a missing /proc file (another kernel, a container) reads as -1.
"""

from __future__ import annotations

import resource
from typing import Dict, Iterable, Tuple

VMSTAT_KEYS: Tuple[str, ...] = ("compact_stall", "pswpin", "pswpout", "pgmajfault")


def read_vmstat(keys: Iterable[str] = VMSTAT_KEYS) -> Dict[str, int]:
    want = set(keys)
    out = {k: -1 for k in want}
    try:
        with open("/proc/vmstat") as fh:
            for line in fh:
                name, _, value = line.partition(" ")
                if name in want:
                    out[name] = int(value)
    except (OSError, ValueError):
        pass
    return out


def vmstat_delta(before: Dict[str, int], after: Dict[str, int]) -> Dict[str, int]:
    return {
        k: (after[k] - before[k]) if before.get(k, -1) >= 0 and after.get(k, -1) >= 0 else -1
        for k in after
    }


def read_psi_memory() -> Tuple[float, float]:
    """``(some avg10, full avg10)`` of /proc/pressure/memory, -1 when absent."""
    some = full = -1.0
    try:
        with open("/proc/pressure/memory") as fh:
            for line in fh:
                kind, _, rest = line.partition(" ")
                for field in rest.split():
                    if field.startswith("avg10="):
                        if kind == "some":
                            some = float(field[6:])
                        elif kind == "full":
                            full = float(field[6:])
    except (OSError, ValueError):
        pass
    return some, full


def read_swap_used_mib() -> int:
    total = free = -1
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("SwapTotal:"):
                    total = int(line.split()[1])
                elif line.startswith("SwapFree:"):
                    free = int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    if total < 0 or free < 0:
        return -1
    return (total - free) // 1024


def rusage_self() -> Tuple[float, float, int, int, int]:
    """``(utime_s, stime_s, minflt, majflt, nivcsw)`` of this process."""
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime, r.ru_stime, r.ru_minflt, r.ru_majflt, r.ru_nivcsw


class HostMemDeltas:
    """PSI memory avg10 and swap/compaction counters since the previous ``line()``."""

    def __init__(self):
        self._prev = read_vmstat()

    def line(self) -> str:
        cur = read_vmstat()
        d = vmstat_delta(self._prev, cur)
        self._prev = cur
        some, full = read_psi_memory()
        return (
            f"psi_mem_some_avg10={some:.2f} psi_mem_full_avg10={full:.2f} "
            f"swap_used_mib={read_swap_used_mib()} pswpin_d={d['pswpin']} pswpout_d={d['pswpout']} "
            f"compact_stall_d={d['compact_stall']} pgmajfault_d={d['pgmajfault']}"
        )
