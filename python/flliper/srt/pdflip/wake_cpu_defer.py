# SPDX-License-Identifier: Apache-2.0
"""WAKEEXT-1023: deferrable CPU work waits until the wake's first forwards are launched.

Metal (27B AutoRound rc35i4ar, 2x3080, D = TP2 + DCP, boot 1023_140032 and 1023_144422): the first D extend after a
P>D wake (a hand-back: 1-6 new tokens over a 28k-344k prefix loaded from L2) costs TP0 284-1155 ms of "compute" while
TP1 computes the SAME forward in 57-224 ms and waits the rest in the collectives (``Prefill rank batch`` lines: 10 of
13 first hand-backs TP0-heavy, 2 TP1, 1 even); in steady state the slower rank of a 1-token extend alternates. The H2D is not it
(``PDFLIP-FIRST-FWD-TIMING load_wait_ms`` 0.1-28 ms; pdflip-21-121 loaded 159 tokens and still took TP0 487 ms vs
TP1 67 ms). TP0 is the group's L3 owner rank: only there do the write-behind thread and the wake's async store rescan
run, and only there the aux-thread read of the same pages takes 1.7-2.7x longer than on TP1 (``PDFLIP-READ-STAGES``
356 vs 130 ms, 617 vs 230 ms) -- the process is CPU-/GIL-contended in exactly that window. The rescan's journal apply
(``HiCacheFile eviction index updated from the journal at wake: ... 89-619 ms``, pure Python, growing with the
journal) runs on its thread right at the wake, beside the first eager extend's launch.

The rescan is bookkeeping (the eviction index; its W8b verdict is voted at the NEXT release fence anyway), so it
waits here until the wake's first forwards are launched:

* :func:`arm` at the wake, right before the rescan thread starts and AFTER that path joined the previous rescan
  (only when the wake holds parked or queued work -- ``_pdflip_wake_defer_work``, set at the wake);
* :func:`note_pass` after every launch (``run_batch``) of the overlap loop and of the PP loop -- released after
  ``N`` extend launches (``FLLIPER_PDFLIP_WAKE_DEFER_EXTENDS``, 2), or at the first non-extend launch with an empty
  waiting queue (nothing left to extend);
* :func:`release` by the rescan's join at the next release leg (never a wait on the flip's critical path);
* bounded by ``FLLIPER_PDFLIP_WAKE_DEFER_MAX_MS`` (4000) -- the deferral is never a gate.

WAKE-OUTLIERS-1024 (27B int8 boot ``..._1024_080355``, D): 5 of 14 D wakes released by the bound, two of them with
``extends=0`` (08:44:09, 08:48:21) -- the wake's first extend was STILL LAUNCHING when the 4 s ran out (its load took
3.3 s, its eager launch is host-paced), so the journal apply (690 ms at 08:48:21 on TP0, the L3 owner) ran inside
exactly the forward the deferral exists to protect: TP0 compute 2879 ms vs TP1 595 / TP2 264 ms for 244 tokens, the
peers waiting 3.3-3.6 s in the collectives (``PDFLIP-FIRST-FWD-TIMING`` gap_sum 3386 ms, load_wait 84 ms -- not the
H2D). Now a bound that expires while a launch is in flight (:func:`launch_begin` at the top of ``run_batch``,
:func:`launch_end` in its ``finally``) waits for that launch to return, at most ``FLLIPER_PDFLIP_WAKE_DEFER_LAUNCH_MAX_MS``
(8000) more -- still never a gate, and a join at the release leg still releases at once.

``FLLIPER_PDFLIP_WAKE_DEFER`` (default ON, "0" = the rescan starts at the wake as before).

The L3 write-behind is NOT held here: S36 SOLL 2 ENDGUELTIG (user 18.10.) -- the L2 -> L3 write never holds still.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

ENV = "FLLIPER_PDFLIP_WAKE_DEFER"
ENV_MAX_MS = "FLLIPER_PDFLIP_WAKE_DEFER_MAX_MS"
ENV_EXTENDS = "FLLIPER_PDFLIP_WAKE_DEFER_EXTENDS"
ENV_LAUNCH_MAX_MS = "FLLIPER_PDFLIP_WAKE_DEFER_LAUNCH_MAX_MS"
MAX_MS_DEFAULT = 4000.0
EXTENDS_DEFAULT = 2
LAUNCH_MAX_MS_DEFAULT = 8000.0

_clock = time.monotonic
_ev = threading.Event()
_ev.set()  # not armed = released
_st = {"armed_at": None, "extends": 0, "deadline": 0.0, "why": "init", "wake": 0, "in_launch": False,
       "launch_gen": 0, "launch_done_gen": 0}
_lock = threading.Lock()
_launch_idle = threading.Event()
_launch_idle.set()  # no launch in flight

#: scheduler-level ForwardMode names of a prefill launch (TARGET_VERIFY is the worker's decode form, not counted)
_EXTEND_MODES = ("EXTEND", "MIXED", "SPLIT_PREFILL", "DLLM_EXTEND")


def enabled(env=None) -> bool:
    env = os.environ if env is None else env
    return str(env.get(ENV, "1") or "1").strip() not in ("0", "false", "off")


def _max_s(env=None) -> float:
    env = os.environ if env is None else env
    try:
        v = float(env.get(ENV_MAX_MS, "") or MAX_MS_DEFAULT)
    except ValueError:
        v = MAX_MS_DEFAULT
    return max(0.0, v) / 1000.0


def _extends_n(env=None) -> int:
    env = os.environ if env is None else env
    try:
        return max(1, int(env.get(ENV_EXTENDS, "") or EXTENDS_DEFAULT))
    except ValueError:
        return EXTENDS_DEFAULT


def _launch_max_s(env=None) -> float:
    env = os.environ if env is None else env
    try:
        v = float(env.get(ENV_LAUNCH_MAX_MS, "") or LAUNCH_MAX_MS_DEFAULT)
    except ValueError:
        v = LAUNCH_MAX_MS_DEFAULT
    return max(0.0, v) / 1000.0


def armed() -> bool:
    return not _ev.is_set()


def hold_bound_s(env=None) -> float:
    """The longest a released-by-bound deferral can hold: the bound plus the launch cap (s)."""
    return _max_s(env) + _launch_max_s(env)


def launch_begin() -> None:
    """Top of ``run_batch``: a launch is in flight (one attribute read while not armed)."""
    if _ev.is_set():
        return
    # review 1024 SOLL 2: a GENERATION, so a waiter that saw launch g wait for g's end, not for the next launch's
    _st["launch_gen"] += 1
    _st["in_launch"] = True
    _launch_idle.clear()


def launch_end() -> None:
    """``run_batch``'s ``finally``: the launch returned (also when it raised)."""
    if not _st["in_launch"]:
        return
    _st["launch_done_gen"] = _st["launch_gen"]
    _st["in_launch"] = False
    _launch_idle.set()


def arm(has_work: bool = True) -> bool:
    """At the wake, before the rescan thread starts. Returns True when the deferral is armed."""
    with _lock:
        _st["wake"] += 1
        if not enabled() or not has_work or _max_s() <= 0:
            _st["armed_at"] = None
            _st["why"] = "off" if not enabled() else ("no_work" if not has_work else "max0")
            _ev.set()
            return False
        now = _clock()
        _st.update(armed_at=now, extends=0, deadline=now + _max_s(), why="armed")
        _ev.clear()
        return True


def release(why: str) -> None:
    with _lock:
        if _ev.is_set():
            return
        _st["why"] = why
        _ev.set()


def note_pass(mode_name: Optional[str], waiting: int) -> None:
    """After a launch (``run_batch`` returned). One attribute read while not armed."""
    if _ev.is_set():
        return
    m = str(mode_name or "").upper()
    if m in _EXTEND_MODES:
        with _lock:
            _st["extends"] += 1
            n = _st["extends"]
        if n >= _extends_n():
            release("extends=%d" % n)
        return
    if int(waiting or 0) <= 0:
        release("no_extend_left mode=%s" % (m or "?"))


def wait(who: str = "") -> float:
    """The deferred work's thread: wait until released or the bound. Returns the seconds waited (and names a
    non-trivial wait once)."""
    if _ev.is_set():
        return 0.0
    t0 = _clock()
    left = max(0.0, _st["deadline"] - t0)
    if not _ev.wait(left):
        if _st["in_launch"]:
            # WAKE-OUTLIERS-1024: the bound ran out inside a launch -- the deferred work starts when it returns
            # (or at a release meanwhile, e.g. the release leg's join), capped: never a gate
            gen = _st["launch_gen"]
            cap = _clock() + _launch_max_s()

            def _running() -> bool:
                return _st["launch_done_gen"] < gen

            while _running() and not _ev.is_set() and _clock() < cap:
                _launch_idle.wait(min(0.01, max(0.0, cap - _clock())))
            release("bound+launch_cap" if _running() else "bound+launch_end")
        else:
            release("bound")
    held = _clock() - t0
    if held >= 0.001:
        armed_at = _st.get("armed_at")
        logger.info("PDFLIP-WAKE-DEFER who=%s waited_ms=%.0f since_wake_ms=%.0f released_by=%s extends=%d "
                    "(WAKEEXT-1023: deferrable CPU work starts after the wake's first forwards are launched; "
                    "FLLIPER_PDFLIP_WAKE_DEFER=0 = at the wake)",
                    who, held * 1000.0, ((_clock() - armed_at) * 1000.0) if armed_at is not None else -1.0,
                    _st["why"], int(_st["extends"]))
    return held


def state() -> dict:
    return dict(_st, released=_ev.is_set())


def _reset_for_tests() -> None:
    with _lock:
        _st.update(armed_at=None, extends=0, deadline=0.0, why="init", wake=0, in_launch=False, launch_gen=0,
                   launch_done_gen=0)
        _ev.set()
        _launch_idle.set()
