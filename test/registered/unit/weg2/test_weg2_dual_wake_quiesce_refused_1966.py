# SPDX-License-Identifier: Apache-2.0
"""#1956 WAKE-SEES-LOAN, the quiesce-refused race (report 1966, b9p death 05.10. ~06:30Z):
the sleep/wake ladder must never wake a P that never slept.

Metal b9p (front.log 19442-44): PressureStages.tick sets p_state='sleeping' the moment it
ISSUES the sleep; the sleep leg starts with quiesce(P) (/flush_cache, 1548 ms there, answer
HTTP 400). In that window the next tick read 'sleeping' + a seat end and issued the wake --
P had never slept, PP0 alone raised Weg2DualPWakeShort.

The fix of 844404f4c0 closes it through ``sleep_leg_done`` (False from the issue of the sleep
until the release RPC answered 200); the five 1956 tests drive the ladder with that flag set
by hand. THIS file drives the real front coroutines (_dual_stage_tick -> _dual_p_sleep /
_dual_p_wake) with a quiesce that is refused (400) and counts the RPCs P receives:

* switch ON : no /resume_memory_occupation at any moment (in flight, after the refusal, later
  ticks), no /release_memory_occupation either (the sleep never started), p_state 'stopped',
  no STOP; a LATER sleep that succeeds wakes normally (the refusal leaves nothing sticky).
* switch OFF: the f9/b9p reading byte for byte -- the wake leg goes out while the quiesce is
  still in flight (the defect, pinned so a default change is seen).

Mutant that must turn the ON tests red: drop ``self.sleep_leg_done = False`` in
PressureStages.tick (dual_d_priority.py).
"""
from __future__ import annotations

import asyncio
import collections
import contextlib
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from sglang.srt.environ import envs
from sglang.srt.weg2 import dual_d_priority as DP
from sglang.srt.weg2 import front as FR
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

@contextlib.contextmanager
def _switch_on():
    """override(True) with a restore that also runs when an assertion fails (a failing test must not
    leave the switch on for the next one)."""
    cm = envs.SGLANG_WEG2_DUAL_WAKE_SEES_LOAN.override(True)
    cm.__enter__()
    try:
        yield
    finally:
        cm.__exit__(None, None, None)


ROOM = [(10 << 30, 1 << 30)] * 3        # every card has its loan free: only the leg state can hold P


def _front(switch: bool):
    """The real Front coroutines on a bare namespace; quiesce / leg_rpc are recorders."""
    if True:
        f = types.SimpleNamespace()
        f.counters = collections.Counter()
        f.groups = {"P": object(), "D": object()}
        f.boot_epoch = 7
        f.queue = []
        f._dual_inflight = 0
        f.state = "SERVING"
        f.rpcs = []
        f.stops = []
        f.gate = asyncio.Event()                       # the quiesce answers when this is set
        f.quiesce_ok = False                           # refused (HTTP 400) unless a test flips it
        f.seats = 0

        async def quiesce(group):
            await f.gate.wait()
            return (True, "") if f.quiesce_ok else (False, "HTTP 400 flush_cache: P not idle (1548 ms)")

        async def leg_rpc(group, path, body, timeout):
            f.rpcs.append(path)
            return 200, ""

        f.quiesce = quiesce
        f.leg_rpc = leg_rpc
        f.do_stop = lambda name, detail: f.stops.append(name)
        f._dual_p_stage_reading = lambda: dict(p_committed=0, free_min=10 << 30, p_grant_bytes=0,
                                               d_air_bytes=0, weights_bytes=0, p_lent=0,
                                               card_room=list(ROOM), d_short=0)
        f._dual_p_weights_tags = lambda: ["weights"]
        for name in ("_dual_stages", "_dual_stage_tick", "_dual_p_sleep_probe_tick", "_dual_p_sleep",
                     "_dual_p_wake", "_dual_p_lend"):
            setattr(f, name, types.MethodType(getattr(FR.Front, name), f))
        f.DUAL_P_SLEEP_PROBE_ENV = FR.Front.DUAL_P_SLEEP_PROBE_ENV
        f._dual_stages_obj = None
        st = f._dual_stages()                          # reads the switch exactly as the front does
        assert st.wake_needs_room is bool(switch)
        st.sleep_capable = True
        st.sleep_after = 1
        st.p_state = "lent"
        st._lend_seat_mark = 0
        f.st = st

        def seat_ends():
            f.seats += 1
            f.counters["d_seat_done"] = f.seats

        f.seat_ends = seat_ends
        return f


async def _settle(n=5):
    for _ in range(n):
        await asyncio.sleep(0)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv(FR.Front.DUAL_P_SLEEP_PROBE_ENV, raising=False)
    monkeypatch.setattr(FR, "_mem_available_bytes", lambda: 1 << 40)


def _issue_sleep(f):
    """Pressure with P lent and every card short: the ladder issues the sleep (task blocked in quiesce)."""
    f._dual_stage_tick(1)
    assert f.st.p_state == "sleeping"


def test_on_quiesce_refused_never_wakes_a_p_that_never_slept():
    async def main():
        with _switch_on():
            f = _front(True)
            _issue_sleep(f)
            await _settle()                            # the sleep task sits in quiesce()
            f.seat_ends()                              # b9p 19442: a D seat ends while the quiesce runs
            f._dual_stage_tick(0)
            await _settle()
            assert f.rpcs == [] and f.st.p_state == "sleeping"       # held: no wake into the quiesce window
            assert f.counters["dual_kv_pressure_wake"] == 0
            f.gate.set()                               # HTTP 400
            await _settle()
            assert f.st.p_state == "stopped"           # P stays stopped (stage 1), awake, never slept
            assert f.counters["dual_kv_pressure_sleep_not_idle"] == 1
            assert "/release_memory_occupation" not in f.rpcs
            for _ in range(4):                         # later seat ends / ticks: still no wake leg
                f.seat_ends()
                f._dual_stage_tick(0)
                await _settle()
            assert "/resume_memory_occupation" not in f.rpcs
            assert f.counters["dual_kv_pressure_wake"] == 0
            assert f.stops == [] and f.state == "SERVING"
    asyncio.run(main())


def test_on_refusal_leaves_nothing_sticky_a_later_sleep_wakes_normally():
    async def main():
        with _switch_on():
            f = _front(True)
            _issue_sleep(f)
            f.gate.set()
            await _settle()
            assert f.st.p_state == "stopped" and f.rpcs == []
            f.quiesce_ok = True                        # the next ask is answered
            f._dual_stage_tick(1)                      # stopped -> lent (P commits 0 B)
            await _settle()
            assert f.st.p_state == "lent"
            _issue_sleep(f)                            # lent -> sleeping, the leg is open again
            f.seat_ends()
            f._dual_stage_tick(0)
            await _settle()
            assert "/resume_memory_occupation" not in f.rpcs   # the new leg ran (answers in _settle) ...
            assert "/release_memory_occupation" in f.rpcs
            assert f.st.sleep_leg_done is True         # ... and answered 200: now the wake is allowed
            f.seat_ends()
            f._dual_stage_tick(0)
            await _settle()
            assert f.rpcs.count("/resume_memory_occupation") == 1
            assert f.st.p_state == "serving" and f.stops == []
    asyncio.run(main())


def test_off_is_the_f9_reading_wake_leg_goes_out_while_the_quiesce_is_in_flight():
    """Default OFF = the old path: the defect, pinned (the wake RPC is sent to a P that has not slept)."""
    async def main():
        assert envs.SGLANG_WEG2_DUAL_WAKE_SEES_LOAN.get() is False
        f = _front(False)
        _issue_sleep(f)
        assert f.st.sleep_leg_done is True             # the old ladder has no leg state
        await _settle()
        f.seat_ends()
        f._dual_stage_tick(0)
        await _settle()
        assert f.rpcs == ["/resume_memory_occupation"]  # b9p 19442-44: wake issued, P never slept
        f.gate.set()
        await _settle()
        assert f.counters["dual_kv_pressure_wake"] == 1
        assert "/release_memory_occupation" not in f.rpcs
    asyncio.run(main())
