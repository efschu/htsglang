# SPDX-License-Identifier: Apache-2.0
"""DECODE-COLLECT on the 27B front (user rule 02.10. ~19:07Z/19:12Z).

The 27B front parks D through Front._immediate_park_due (ARRIVAL-SEAT is off on
27B), so the window has to win there over PARK-NO-DWELL, PARK-COLLECT-WINDOW and
PARK-SEAT-FREE. Arrivals at 1 / 7 / 14 s hold, 15.5 s parks (P takes the set);
a set <= X at the 7.5 s D-check goes to D (no park); WINDOW_S=0 is the old path
(PARK-NO-DWELL parks at once). PARK-SEAT-FREE runs only when no window decided.
"""
from __future__ import annotations

import importlib.util
import time
import types
from pathlib import Path

import pytest

from sglang.srt.weg2 import front as F

_spec = importlib.util.spec_from_file_location(
    "_flipwait_harness_dc27b", Path(__file__).with_name("test_27b_flipwait_1002.py"))
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)

_LONG = 25215  # > X 3797 of the harness: P-bound


@pytest.fixture(autouse=True)
def _window(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_DECODE_COLLECT_WINDOW_S", "15")
    monkeypatch.delenv("SGLANG_WEG2_DECODE_COLLECT_D_CHECK_S", raising=False)  # default 7.5
    monkeypatch.setenv("SGLANG_WEG2_PARK_NO_DWELL", "1")
    monkeypatch.setenv("SGLANG_WEG2_ENABLE_PARK_COLLECT_WINDOW", "1")
    monkeypatch.setenv("SGLANG_WEG2_ENABLE_PARK_SEAT_FREE", "1")


def _front(waits_s, uncached=_LONG, running=2, **kw):
    fn, ns = H._front(running=running, waits_s=waits_s, awake_s=60.0, uncached=uncached, **kw)
    for q in ns.queue:
        q.fut = types.SimpleNamespace(done=lambda: False)
    ns.awake, ns.state = "D", "serving"
    ns.groups = {"D": types.SimpleNamespace(outstanding={})}
    for name in ("_decode_collect", "_dc_st", "_dc_window_s", "_dc_decoding"):
        setattr(ns, name, types.MethodType(getattr(F.Front, name), ns))
    return fn, ns


@pytest.mark.parametrize("waited_s", [1.0, 7.0, 14.0])
def test_a_p_bound_arrival_on_a_decoding_d_is_held(waited_s, monkeypatch):
    seen = []
    monkeypatch.setattr(F.Front, "_park_seat_free_due", lambda self, *a: seen.append(a) or True)
    fn, ns = _front([waited_s])
    assert fn(ns, None, time.time()) is None, "the window holds -- no park, D decodes on"
    assert ns.counters["decode_collect_hold"] == 1
    assert ns.counters.get("park_no_dwell", 0) == 0, "PARK-NO-DWELL is overruled"
    assert ns.counters["park_collect_holds"] == 0
    assert seen == [], "PARK-SEAT-FREE is overruled"
    assert ns.counters["decode_collect_dcheck_hold"] == (1 if waited_s >= 7.5 else 0)


def test_at_15_5_s_the_set_parks_and_p_takes_it():
    fn, ns = _front([15.5, 3.0])
    got = fn(ns, None, time.time())
    assert got is not None and got.rid == "weg2-16-39", "the oldest of the set fires the park"
    assert ns.counters["decode_collect_release_P"] == 1
    # the route stays P for the rest of this D phase: a later arrival rides the same flip
    ns.queue.append(types.SimpleNamespace(rid="late", est_uncached=500, t_arrive=time.time(),
                                          p_only=False, x_requeues=0, leg1_done=False,
                                          skip_leg1=False, x_deferred=False,
                                          fut=types.SimpleNamespace(done=lambda: False)))
    assert fn(ns, None, time.time()) is not None


def test_over_x_at_d_check_collects_on_no_second_d_check():
    fn, ns = _front([8.0])
    assert fn(ns, None, time.time()) is None
    assert ns.counters["decode_collect_dcheck_hold"] == 1
    assert ns.counters["decode_collect_release_D"] == 0


def test_a_set_at_most_x_at_d_check_goes_to_d():
    # two SHORT-sized arrivals, 1500 + 1500 <= X 3797, the first 7.6 s ago
    fn, ns = _front([7.6, 2.0], uncached=1500)
    assert fn(ns, None, time.time()) is None, "no park: D prefills the small set"
    assert ns.counters["decode_collect_release_D"] == 1
    assert ns.counters["decode_collect_release_P"] == 0
    assert ns._dc_state["route"] == "D"


def test_a_set_at_most_x_before_d_check_still_holds():
    fn, ns = _front([5.0], uncached=1500)
    assert fn(ns, None, time.time()) is None
    assert ns.counters["decode_collect_hold"] == 1
    assert ns.counters["decode_collect_release_D"] == 0


def test_zero_is_the_old_path(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_DECODE_COLLECT_WINDOW_S", "0")
    fn, ns = _front([1.0])
    got = fn(ns, None, time.time())
    assert got is not None, "PARK-NO-DWELL parks at once, as before"
    assert ns.counters["park_no_dwell"] == 1
    assert "_dc_state" not in ns.__dict__


def test_an_idle_d_is_not_held():
    fn, ns = _front([1.0], running=0)
    assert fn(ns, None, time.time()) is None  # the idle flip is the controller's own path
    assert "_dc_state" not in ns.__dict__
    assert ns.counters["decode_collect_hold"] == 0


def test_seat_free_runs_when_no_window_decided(monkeypatch):
    seen = []
    monkeypatch.setattr(F.Front, "_park_seat_free_due", lambda self, *a: seen.append(a) or True)
    monkeypatch.setenv("SGLANG_WEG2_DECODE_COLLECT_WINDOW_S", "0")
    monkeypatch.setenv("SGLANG_WEG2_PARK_NO_DWELL", "0")
    fn, ns = _front([1.0], running=2)
    ns.t_awake = time.time() - 60.0
    assert fn(ns, None, time.time()) is not None
    assert len(seen) == 1, "WINDOW_S=0: PARK-SEAT-FREE as before"


def test_release_p_parks_even_with_the_dwell_rules_back_on(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_PARK_NO_DWELL", "0")
    seen = []
    monkeypatch.setattr(F.Front, "_park_seat_free_due", lambda self, *a: seen.append(a) or None)
    fn, ns = _front([15.5])
    assert fn(ns, None, time.time()) is not None
    assert seen == [] and ns.counters["park_collect_holds"] == 0
