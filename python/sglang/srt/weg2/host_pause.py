"""Host-pause-aware wall budgets for the flip's lane waits (W68 class).

NF death 14.10. 12:17:36Z (boot ..._2b27c61d48_1014_115938, flip 17 P->D):
the Proxmox host swap-thrashed (MemAvailable 2.2 GiB, load1 628, no scrape
on any exporter 12:15:42-12:17:37) and EVERY process -- P ranks, D ranks,
the tokenizer loops (TOKENIZER-LOOP-LAG 117 s on P and D) -- stood still
for ~117 s. The lane budgets are wall time, so the 117 s of freeze were
charged to them: P PP0's wait for 'free' of p0/weights_1 expired at
12:17:36, the very second D TP1 (alive, its collector) mapped that collect;
D TP2's c2 wait expired the same way. W68 -> W121 -> W29 -> gloo -> rank dead,
without any lane, credit or collector being stuck.

The same pattern as Hadoop's ``JvmPauseMonitor`` / HBase's pause detector
and Gil Tene's jHiccup: a loop that sleeps in short slices notices that one
slice took far longer than asked -- the process (host) was not scheduled.
That time is not the peer's fault, so the budget is extended by it. A real
hang (dead or stuck peer while this process runs) still expires after the
full budget of RUNNING time, and the credit per wait is capped.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

#: A gap between two consecutive checks of a wait loop longer than its slice
#: plus this is counted as a host/process pause (not the peer's time).
ENV_PAUSE_MIN_S = "SGLANG_WEG2_HOST_PAUSE_MIN_S"
#: Most pause time one wait may get back (a host that stands still longer is
#: not coming back in a state worth flipping in).
ENV_PAUSE_CREDIT_MAX_S = "SGLANG_WEG2_HOST_PAUSE_CREDIT_MAX_S"

_lock = threading.Lock()
_total_credit_s = 0.0
_n_pauses = 0


def _env_f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def pause_min_s() -> float:
    return max(0.5, _env_f(ENV_PAUSE_MIN_S, 2.0))


def pause_credit_max_s() -> float:
    return max(0.0, _env_f(ENV_PAUSE_CREDIT_MAX_S, 900.0))


def total_credit_s() -> float:
    return _total_credit_s


class PauseAwareDeadline:
    """``budget_s`` of RUNNING time from now. ``left()`` must be called once
    per slice of the wait loop (``slice_s`` = the longest the loop blocks
    between two calls); a gap beyond ``slice_s + pause_min_s()`` is credited
    back to the deadline and logged once as WEG2-HOST-PAUSE."""

    __slots__ = ("what", "slice_s", "deadline", "last", "credited", "_clock",
                 "_min", "_max")

    def __init__(self, budget_s: float, slice_s: float, what: str = "",
                 clock: Optional[Callable[[], float]] = None):
        self._clock = clock or time.monotonic
        now = self._clock()
        self.what = what
        self.slice_s = max(0.0, float(slice_s))
        self.deadline = now + float(budget_s)
        self.last = now
        self.credited = 0.0
        self._min = pause_min_s()
        self._max = pause_credit_max_s()

    def left(self) -> float:
        global _total_credit_s, _n_pauses
        now = self._clock()
        gap = now - self.last
        self.last = now
        over = gap - self.slice_s
        if over >= self._min and self.credited < self._max:
            c = min(over, self._max - self.credited)
            self.deadline += c
            self.credited += c
            with _lock:
                _total_credit_s += c
                _n_pauses += 1
                n = _n_pauses
            logger.warning(
                "WEG2-HOST-PAUSE wait=%s pause_s=%.1f credited_s=%.1f wait_credit_s=%.1f "
                "cap_s=%.0f n=%d -- this process was not scheduled for that long "
                "(host swap/freeze); the time is given back to the wait's budget "
                "instead of being charged to the peer (W68, NF 14.10. 12:17Z)",
                self.what or "?", gap, c, self.credited, self._max, n)
        return self.deadline - now

    def expired(self) -> bool:
        return self.left() <= 0
