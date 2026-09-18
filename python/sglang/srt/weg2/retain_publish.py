"""xsn338 (18.09.2026): P publishes a finished request's nodes AT ITS FINISH.

The store publish ran only in the PP loop's bubbles (weg2_bubble_publish) and
in the flush; a phase of back-to-back prefills has no bubble, so the pages of a
request that finished at 15:31:35 reached the arena at 15:32:12 -- with the
phase's last requests only at the sleep flush. D's dormant re-reads asked in
that gap and answered zero, and the wake paid the whole look-up burst.
"""
from __future__ import annotations

import os
from typing import Mapping, Optional


def publish_at_retain_on(env: Optional[Mapping[str, str]] = None) -> bool:
    env = os.environ if env is None else env
    if str(env.get("SGLANG_WEG2_PUBLISH_AT_RETAIN", "1")).strip().lower() in ("0", "false", "no", "off"):
        return False
    return str(env.get("SGLANG_WEG2_GROUP", "")).strip().upper() == "P"


def max_issue(env: Optional[Mapping[str, str]] = None) -> int:
    env = os.environ if env is None else env
    try:
        return max(1, int(env.get("SGLANG_WEG2_PUBLISH_AT_RETAIN_MAX", "64")))
    except ValueError:
        return 64


def budget_s(env: Optional[Mapping[str, str]] = None) -> float:
    """xsn342 (18.09.): the retain sweep runs in the scheduler thread; a
    refused chain of 43 nodes cost PP1 minutes (the arena clock-hand wrap,
    fixed in arena.c) and the PP ring stood. Wall budget per sweep in ms
    (SGLANG_WEG2_PUBLISH_AT_RETAIN_BUDGET_MS, default 800;
    xsn344: 24 nodes of a 100k request take ~700 ms, 400 left half unpublished); 0 = unbounded.
    Whatever is left is the bubble publisher's and the flush's, as before."""
    env = os.environ if env is None else env
    try:
        ms = float(env.get("SGLANG_WEG2_PUBLISH_AT_RETAIN_BUDGET_MS", "800"))
    except ValueError:
        ms = 800.0
    return max(0.0, ms) / 1000.0


class SweepClock:
    """One sweep's wall clock: `expired()` once the budget is spent (a budget
    of 0 never expires); `stop_reason` names why the sweep ended early."""

    def __init__(self, budget: float, now=None):
        import time as _t
        self._now = now or _t.perf_counter
        self.budget = float(budget)
        self.t0 = self._now()
        self.stop_reason = None

    def expired(self) -> bool:
        if self.budget <= 0.0:
            return False
        if self._now() - self.t0 >= self.budget:
            self.stop_reason = "budget"
            return True
        return False

    def elapsed_ms(self) -> float:
        return (self._now() - self.t0) * 1000.0


def mamba_full_stops_sweep(refused_why: Optional[str]) -> bool:
    """A node refused because the MAMBA arena has no free slot: every later
    node of this sweep needs an anchor slot too (states are per node), so the
    sweep stops instead of claiming+aborting 4096 KV pages per node (xsn342:
    16 evict rounds and 48 x 4096 aborts in one second, logs capped)."""
    return refused_why == "mamba_claim"


def dormant_standstill_holds(dormant: bool) -> bool:
    """xsn344 (18.09.): 2 of 4 100k requests were answered 503 (W88) by the
    SLEEPING D group after 64 passes without progress -- P had just finished
    them and publishes their pages at retain (budgeted), in bubbles and at its
    sleep flush, so the store WILL fill; a dormant standstill is a wait, not a
    dead read. The pass/wall bound applies from the wake on."""
    return bool(dormant)
