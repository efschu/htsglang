"""K10 SEAT-RELIEF trigger (c) on its own switch (h16q integration, 15.10.).

(c) = under AGE PLAN the seat gate's hold past SGLANG_WEG2_K10_HOLD_MAX_S (15 s after the DECODE-COLLECT release) or
the oldest waiter past the wait bound -> park D's running decodes + flip, P takes the oldest one. It parks decodes
that are running (D decoding on), which rubs against the user's "kein Massenparken" (14.10.); the user decides about
it. So (c) has its own switch, SGLANG_WEG2_K10_RELIEF_P_WAIT, default ON (new features on, user order 14.10.).
OFF = (c) never fires; (a) (a RESUME-VIA-P leg holds its seat) and (b) (no D token for SGLANG_WEG2_K10_STALL_S) stay
exactly as before. The (c) cases themselves are test_weg2_k10_seat_relief_1015.py (switch at its default).
Hermetic, no CUDA.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from test_weg2_k10_seat_relief_1015 import (  # noqa: E402
    MEASURED_LATENCY_S,
    Front,
    _full_gate,
    _k10,
    _rvp,
    _step,
)

from sglang.srt.environ import envs  # noqa: E402

SW = "SGLANG_WEG2_K10_RELIEF_P_WAIT"


def test_relief_c_switch_defaults_on(monkeypatch):
    monkeypatch.delenv(SW, raising=False)
    assert envs.SGLANG_WEG2_K10_RELIEF_P_WAIT.get() is True


@pytest.mark.parametrize("latency_s", MEASURED_LATENCY_S)
def test_relief_c_off_the_hold_bound_never_flips(monkeypatch, latency_s):
    """Default ON fires at 15.0-15.5 s (test_k10_repro_...); OFF: the gate holds as on 226aad7105."""
    _k10(monkeypatch)
    monkeypatch.setenv(SW, "0")
    f, now = _full_gate()
    t = 0.0
    while t < latency_s:
        f._ipc_out_book().token("r1", now + t)          # D decodes on: (b) never fires either
        assert _step(f, now + t) == (False, False, None)
        t += 0.5
    assert f.counters["k10_seat_relief_p_wait"] == 0


def test_relief_c_off_the_wait_bound_never_flips(monkeypatch):
    _k10(monkeypatch, hold="0")
    monkeypatch.setenv(SW, "0")
    f, now = _full_gate()
    for t in (0.0, 29.0, 31.0, 90.0):                     # the oldest waiter passes the 60 s bound at +30 s
        f._ipc_out_book().token("r1", now + t)
        assert _step(f, now + t) == (False, False, None)
    assert f.counters["k10_seat_relief_p_wait"] == 0


def test_relief_c_off_keeps_a(monkeypatch):
    _k10(monkeypatch)
    monkeypatch.setenv(SW, "0")
    f, now = _full_gate()
    leg = _rvp("weg2-10-51", now - 5.0)
    f.queue.append(leg)
    assert _step(f, now) == (True, True, leg)
    assert f.counters["k10_seat_relief_rvp_seat_held"] == 1 and Front._k10_rvp_only(f)


def test_relief_c_off_keeps_b(monkeypatch):
    """With (c) off the 15 s hold bound is no trigger any more: (b) alone decides, as with hold=0."""
    _k10(monkeypatch)                                     # hold at its default 15 s: only (c) reads it
    monkeypatch.setenv(SW, "0")
    f, now = _full_gate()
    f.d_wait_bound_s = 3600.0
    assert _step(f, now) == (False, False, None)           # the clock starts
    assert _step(f, now + 20.0) == (False, False, None)    # past the hold bound: (c) is off
    assert _step(f, now + 61.0) == (True, True, None)
    assert f.counters["k10_seat_relief_no_d_progress"] == 1 and f.counters["k10_seat_relief_p_wait"] == 0
