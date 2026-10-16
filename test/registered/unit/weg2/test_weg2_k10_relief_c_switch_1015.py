"""K10 SEAT-RELIEF trigger (c) on its own switch, default OFF (h16q review A1, user word, 15.10.).

(c) = under AGE PLAN the seat gate's hold past SGLANG_WEG2_K10_HOLD_MAX_S (15 s after the DECODE-COLLECT release) or
the oldest waiter past the wait bound -> park D's running decodes + flip, P takes the oldest one. It parks decodes D
is still running: a plaster against "kein Massenparken" (14.10.), not a fix -- the user's word turns it OFF in the
code (what the user switched off stays off), and the seat-gate cause fix (desk/nf-h16q-seatgate-1015) deletes it.
SGLANG_WEG2_K10_RELIEF_P_WAIT: 0 / unset = (c) never fires; (a) RESUME-VIA-P and (b) no-D-token stay exactly as
before. 1 = (c) as on 1c337d79a3 (its cases: test_weg2_k10_seat_relief_1015.py, which switches it on).
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


def _default(monkeypatch, **kw):
    _k10(monkeypatch, **kw)
    monkeypatch.delenv(SW, raising=False)        # the code default decides


def test_relief_c_switch_defaults_off(monkeypatch):
    monkeypatch.delenv(SW, raising=False)
    assert envs.SGLANG_WEG2_K10_RELIEF_P_WAIT.get() is False


@pytest.mark.parametrize("latency_s", MEASURED_LATENCY_S)
def test_relief_c_default_the_hold_bound_never_flips(monkeypatch, latency_s):
    """The default holds the gate as on 226aad7105 (no pauschal park of running decodes)."""
    _default(monkeypatch)
    f, now = _full_gate()
    t = 0.0
    while t < latency_s:
        f._ipc_out_book().token("r1", now + t)          # D decodes on: (b) never fires either
        assert _step(f, now + t) == (False, False, None)
        t += 0.5
    assert f.counters["k10_seat_relief_p_wait"] == 0


def test_relief_c_default_the_wait_bound_never_flips(monkeypatch):
    _default(monkeypatch, hold="0")
    f, now = _full_gate()
    for t in (0.0, 29.0, 31.0, 90.0):                     # the oldest waiter passes the 60 s bound at +30 s
        f._ipc_out_book().token("r1", now + t)
        assert _step(f, now + t) == (False, False, None)
    assert f.counters["k10_seat_relief_p_wait"] == 0


def test_relief_c_switched_on_fires_at_the_hold_bound(monkeypatch):
    _k10(monkeypatch)
    monkeypatch.setenv(SW, "1")
    f, now = _full_gate()
    fired = None
    t = 0.0
    while t < 30.0 and fired is None:
        f._ipc_out_book().token("r1", now + t)
        if _step(f, now + t)[0]:
            fired = t
        t += 0.5
    assert fired is not None and 15.0 <= fired < 15.5
    assert f.counters["k10_seat_relief_p_wait"] == 1


def test_relief_c_default_keeps_a(monkeypatch):
    _default(monkeypatch)
    f, now = _full_gate()
    leg = _rvp("weg2-10-51", now - 5.0)
    f.queue.append(leg)
    assert _step(f, now) == (True, True, leg)
    assert f.counters["k10_seat_relief_rvp_seat_held"] == 1 and Front._k10_rvp_only(f)


def test_relief_c_default_keeps_b(monkeypatch):
    """With (c) off the 15 s hold bound is no trigger: (b) alone decides, as with hold=0."""
    _default(monkeypatch)                                 # hold at its default 15 s: only (c) reads it
    f, now = _full_gate()
    f.d_wait_bound_s = 3600.0
    assert _step(f, now) == (False, False, None)           # the clock starts
    assert _step(f, now + 20.0) == (False, False, None)    # past the hold bound: (c) is off
    assert _step(f, now + 61.0) == (True, True, None)
    assert f.counters["k10_seat_relief_no_d_progress"] == 1 and f.counters["k10_seat_relief_p_wait"] == 0
