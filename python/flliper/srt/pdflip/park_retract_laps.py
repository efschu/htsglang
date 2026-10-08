"""FLIP-EDGE 2 (02.10.): the sub-laps of the park's ``retract`` lap.

N5d (boot ..._0c996cf05c_1002_124821, D log): the D->P warmup IS the
``/pdflip/park_running`` RPC, and its largest term is ``retract_ms`` -- 0.07 to
0.76 s, linear in the parked sequence length (~3.3-4 us per token at
page_size 1: e18 seq 210263 -> 667 ms with only 511 host slots written,
e24 seq 188367 -> 756 ms), NOT in the bytes the forced write-through took.
A CPU profile of the bare radix insert at 171k tokens costs 2 ms, so the
cost sits in one of the steps around it (key build, the component
prepare/cleanup, the write-through issue, the cap, the hand-off write, the
retain publish, the lock release, the request reset) -- which one the log
cannot say.

This module is the instrument for exactly that: ``park_running`` arms it
around ``retract_all(retain=True)``; ``release_req`` / ``release_kv_cache`` /
``UnifiedRadixCache.cache_finished_req`` call :func:`mark` after each step.
The marks are CONTIGUOUS laps (each closes the interval since the previous
mark), so their sum is the whole retract and no term is missing; a step
reached by several requests accumulates. :func:`acc` adds a NESTED term (the
write-through issue inside the insert) that is part of a lap, not beside it.
Unarmed, every call is one module-global test. Instrument only: nothing
branches on it. ``FLLIPER_PDFLIP_PARK_RETRACT_LAPS=0`` never arms it.
"""

from __future__ import annotations

import os
import time

_clock = time.perf_counter

_ARMED = False
_LAPS: dict = {}
_NESTED: dict = {}
_T = [0.0]


def enabled() -> bool:
    return os.environ.get("FLLIPER_PDFLIP_PARK_RETRACT_LAPS", "1") != "0"


def arm() -> None:
    """Start the contiguous clock (park_running, right before the retraction)."""
    global _ARMED
    _LAPS.clear()
    _NESTED.clear()
    if not enabled():
        _ARMED = False
        return
    _ARMED = True
    _T[0] = _clock()


def armed() -> bool:
    return _ARMED


def mark(name: str) -> None:
    """Close the lap ``name``: the time since the previous mark (or the arm)."""
    if not _ARMED:
        return
    t = _clock()
    _LAPS[name] = _LAPS.get(name, 0.0) + (t - _T[0])
    _T[0] = t


def acc(name: str, dt_s: float, n: int = 1) -> None:
    """A nested term (inside the current lap): seconds and a count."""
    if not _ARMED:
        return
    s, c = _NESTED.get(name, (0.0, 0))
    _NESTED[name] = (s + float(dt_s), c + int(n))


def disarm() -> tuple:
    """Stop; returns (laps, nested). The interval since the last mark is the
    lap ``rest`` (what lies between the last instrumented step and the end of
    the retraction), so ``sum(laps)`` is the retract lap of the park. No step
    reached = empty."""
    global _ARMED
    if not _ARMED:
        return {}, {}
    if not _LAPS:  # no step reached (nothing retracted): no split to print
        _ARMED = False
        _NESTED.clear()
        return {}, {}
    mark("rest")
    _ARMED = False
    laps, nested = dict(_LAPS), dict(_NESTED)
    _LAPS.clear()
    _NESTED.clear()
    return laps, nested


def describe(laps: dict, nested: dict) -> str:
    """``name_ms=..`` per lap in first-reached order, then the nested terms."""
    parts = ["%s_ms=%.1f" % (k, v * 1e3) for k, v in laps.items()]
    parts.append("sum_ms=%.1f" % (sum(laps.values()) * 1e3))
    for k, (s, c) in nested.items():
        parts.append("%s_ms=%.1f %s_n=%d" % (k, s * 1e3, k, c))
    return " ".join(parts)
