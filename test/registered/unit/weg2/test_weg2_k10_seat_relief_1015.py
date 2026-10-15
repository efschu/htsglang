"""K10 SEAT-GATE / D-WALL (NF h16p 15.10. 12:21:52-12:27:33Z, boot ..._226aad7105_1015_115748).

D (Form A) stood 6 min: running 0, queue 7, every pass ``FORM-A-CUT LOAD-BACK ROOM ... evicted=0``,
``ARENA-DROP freed=0`` x3366 (backup wall). The front's SEAT GATE held every D->P flip
(``seat-gate hold taken=11 n=6``, AGE PLAN ``plan=none``), the P-only legs of the two requests D
had handed back (W50 -> RESUME-VIA-P, weg2-10-51 / -21-67) waited at the gate although D held their
seats -- and only a flip clears D's tree (ARENA-REF RESET-RELEASE). PW-R had flapped on a rank-local
2 s memo since 12:12:57.

Pinned here:
  front  (a) a queued RESUME-VIA-P leg passes the full gate, P takes only such legs;
         (b) the gate held K10_STALL_S without a D token -> park + flip, P takes only such legs;
         (c) under AGE PLAN the hold past K10_HOLD_MAX_S (15) after the release, or the oldest waiter past
             the wait bound (d_wait_bound_s) -- D decoding on (h16p boots 124245 / 133855: release -> flip
             21.7 / 26.4 / 27.7 / 45.9 / 52.9 / 73.5 s) -> park + flip, P takes the oldest one;
         at most once per D epoch, the seat count never re-derived;
  (D's half, PW-F, is test_weg2_k10_pwf_nofund_1015.py.)
Hermetic, no CUDA.
"""
from __future__ import annotations

import asyncio
import inspect
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from test_weg2_arrival_seat_rule_0929 import _pending  # noqa: E402
from test_weg2_pbound_seat_hold_1013 import _on, _pendel  # noqa: E402

from sglang.srt.managers.scheduler_components import weight_updater as _wu  # noqa: E402,F401
from sglang.srt.weg2 import front as front_mod  # noqa: E402
from sglang.srt.weg2.front_state_ipc import OutstandingBook  # noqa: E402

Front = front_mod.Front


def _k10(monkeypatch, stall="60", hold=None):
    _on(monkeypatch)
    monkeypatch.delenv("SGLANG_WEG2_K10_SEAT_RELIEF", raising=False)  # default on
    monkeypatch.setenv("SGLANG_WEG2_K10_STALL_S", stall)
    if hold is None:
        monkeypatch.delenv("SGLANG_WEG2_K10_HOLD_MAX_S", raising=False)  # default 15 s
    else:
        monkeypatch.setenv("SGLANG_WEG2_K10_HOLD_MAX_S", hold)


def _full_gate(now_off=0.0):
    """The K10 shape: D full (6/6), a fresh route-P set collected past the window, the age plan
    finds nothing (``plan=none``): the gate's own no-seat path holds."""
    f, now, q = _pendel()
    f.__dict__.pop("_pbound_seat_held", None)
    f.t_awake = now - 30.0                     # the oldest waiter: 30 s into this D phase, below the bound
    f.queue.append(_pending("fresh", 87000, now - 16.0))
    f._arrival_seat_age_verdict = lambda *a, **k: None
    return f, now + now_off


def _rvp(rid, t):
    p = _pending(rid, 1921, t)
    p.resume_via_p = True
    p.p_only = True
    return p


def _step(f, now):
    return asyncio.run(f._arrival_seat_step(f.groups["D"], now))


# ------------------------------------------------------------------ (a) a seat-holding leg passes

def test_k10_without_relief_the_full_gate_holds_as_before(monkeypatch):
    _k10(monkeypatch)
    monkeypatch.setenv("SGLANG_WEG2_K10_SEAT_RELIEF", "0")
    f, now = _full_gate()
    f.queue.append(_rvp("weg2-10-51", now - 5.0))
    assert _step(f, now) == (False, False, None)
    assert f.counters["decode_collect_seat_hold"] == 1
    assert not Front._k10_rvp_only(f)


def test_k10_a_resume_via_p_leg_passes_the_full_gate_and_p_takes_only_it(monkeypatch):
    """RED on 226aad7105: (False, False, None) -- the leg waited at the gate (12:20:24 ->)."""
    _k10(monkeypatch)
    f, now = _full_gate()
    leg = _rvp("weg2-10-51", now - 5.0)
    f.queue.append(leg)
    taken0 = f._arrival_seat_taken()
    res = _step(f, now)
    assert res == (True, True, leg)
    assert f.counters["k10_seat_relief_rvp_seat_held"] == 1
    assert f._dc_p_cap == 1
    assert Front._k10_rvp_only(f)
    assert f._arrival_seat_taken() == taken0, "the seat count is never re-derived"


def test_k10_relief_fires_once_per_d_epoch(monkeypatch):
    """Mutant 'no epoch lock' dies here: the second tick of the same D epoch is the gate's own path."""
    _k10(monkeypatch)
    f, now = _full_gate()
    f.queue.append(_rvp("weg2-10-51", now - 5.0))
    assert _step(f, now)[0] is True
    assert _step(f, now + 0.2) == (False, False, None)
    assert f.counters["k10_seat_relief_rvp_seat_held"] == 1
    f.epoch += 2  # the next D epoch (one flip there, one back)
    assert _step(f, now + 30.0)[0] is True
    assert f.counters["k10_seat_relief_rvp_seat_held"] == 2


# ------------------------------------------------------------------ (b) the liveness latch

def test_k10_no_d_token_for_the_stall_bound_parks_and_flips(monkeypatch):
    """RED on 226aad7105: the gate held 12:20:24-12:27:33 with D serving nothing."""
    _k10(monkeypatch, hold="0")                # the hold bound out of the way
    f, now = _full_gate()
    f.d_wait_bound_s = 3600.0                  # (c) out of the way: (b) alone
    assert _step(f, now) == (False, False, None)           # the clock starts
    assert _step(f, now + 59.0) == (False, False, None)    # below the bound
    res = _step(f, now + 61.0)
    assert res == (True, True, None)
    assert f.counters["k10_seat_relief_no_d_progress"] == 1
    assert Front._k10_rvp_only(f), "P takes only seat-holding legs (none: an empty round trip)"
    assert _step(f, now + 200.0) == (False, False, None), "once per D epoch"


def test_k10_a_d_token_restarts_the_clock(monkeypatch):
    _k10(monkeypatch, hold="0")                # the hold bound out of the way
    f, now = _full_gate()
    f.d_wait_bound_s = 3600.0                  # (c) out of the way: (b) alone
    assert _step(f, now) == (False, False, None)
    f._ipc_out_book().token("r1", now + 40.0)                # D streamed a chunk
    assert _step(f, now + 61.0) == (False, False, None)      # the token is seen here: the clock restarts
    assert _step(f, now + 120.0) == (False, False, None)
    assert _step(f, now + 122.0) == (True, True, None)       # 61 s after the restart
    assert f.counters["k10_seat_relief_no_d_progress"] == 1


def test_k10_a_new_d_epoch_restarts_the_clock(monkeypatch):
    _k10(monkeypatch, hold="0")                # the hold bound out of the way
    f, now = _full_gate()
    f.d_wait_bound_s = 3600.0                  # (c) out of the way: (b) alone
    assert _step(f, now) == (False, False, None)
    f.epoch += 2
    assert _step(f, now + 61.0) == (False, False, None)
    assert f.counters["k10_seat_relief_no_d_progress"] == 0


def test_k10_stall_bound_zero_disarms_the_latch(monkeypatch):
    _k10(monkeypatch, stall="0", hold="0")
    f, now = _full_gate()
    f.d_wait_bound_s = 3600.0                  # (c) out of the way: (b) alone
    assert _step(f, now) == (False, False, None)
    assert _step(f, now + 1000.0) == (False, False, None)


def test_k10_the_wait_bound_flips_with_d_decoding_and_p_takes_the_oldest(monkeypatch):
    """RED on 226aad7105 (h16p boot 1242, epoch 30): the hold outlived the 60 s wait bound by 30 s
    while D decoded on -- the bound the hold line names was enforced by nothing on this path."""
    _k10(monkeypatch, hold="0")                # the hold bound out of the way
    f, now = _full_gate()
    f._ipc_out_book().token("r1", now)                     # D decodes: (b) never fires
    assert _step(f, now) == (False, False, None)
    f._ipc_out_book().token("r1", now + 29.0)
    assert _step(f, now + 29.0) == (False, False, None)    # oldest waiter 59 s
    res = _step(f, now + 31.0)                             # 61 s
    assert res == (True, True, None)
    assert f.counters["k10_seat_relief_p_wait"] == 1
    assert f._dc_p_cap == 1 and not Front._k10_rvp_only(f), "P takes the oldest one, not only seat holders"
    assert _step(f, now + 90.0) == (False, False, None), "once per D epoch"


def test_outstanding_book_last_any_is_monotonic():
    b = OutstandingBook()
    assert b.last_any == 0.0
    b.token("a", 10.0)
    b.token("b", 5.0)
    b.end("a")
    assert b.last_any == 10.0


# ------------------------------------------------------------------ the rvp-only flip

def test_k10_rvp_only_window_is_the_relief_epoch_and_the_next():
    f = types.SimpleNamespace(epoch=38)
    f.__dict__["_dc_p_rvp_only_epoch"] = 38
    assert Front._k10_rvp_only(f)
    f.epoch = 39
    assert Front._k10_rvp_only(f)
    f.epoch = 40
    assert not Front._k10_rvp_only(f)
    assert not Front._k10_rvp_only(types.SimpleNamespace(epoch=1))


def test_k10_rvp_pick_and_queued_and_early_legs():
    a, b, c = _pending("a", 9000, 1.0), _rvp("b", 2.0), _pending("c", 9000, 3.0)
    q = [a, b, c]
    assert front_mod.k10_rvp_pick(q) == 1
    assert front_mod.k10_rvp_pick(q, lambda p: p.rid != "b") is None
    assert front_mod.k10_rvp_queued(q) and not front_mod.k10_rvp_queued([a, c])
    for p in q:
        p.d_direct = p.client_gone = False
        p._leg1_early = None
    cands = front_mod.leg1_early_candidates(q, 6)
    assert [p.rid for p in cands] == ["a", "b", "c"]
    f = types.SimpleNamespace(epoch=5)
    assert Front._k10_early_filter(f, cands) == cands
    f.__dict__["_dc_p_rvp_only_epoch"] = 5
    assert [p.rid for p in Front._k10_early_filter(f, cands)] == ["b"]


def test_k10_wiring_the_p_phase_and_the_early_legs_read_the_rvp_only_flag():
    src = inspect.getsource(front_mod)
    assert "_k10_only = Front._k10_rvp_only(self)" in src
    assert "(not _k10_only or k10_rvp_queued(self.queue))" in src
    assert "k10_rvp_pick(q, pred)) if _k10_only" in src
    assert src.count("Front._k10_early_filter(self, leg1_early_candidates(self.queue, _lim))") == 2
    step = inspect.getsource(Front._dc_no_seat_step)
    assert step.index("_k10_seat_relief") < step.index("age_plan_enabled")


import pytest  # noqa: E402

#: h16p, release route=P -> WEG2-FLIP begin, the seat gate holding, D decoding on (NF-Sitz BELEG-K10-FRONT):
#: boot 124245 epochs 14 / 30 / 52 / -, boot 133855 epochs 8 / 10
MEASURED_LATENCY_S = [52.9, 73.5, 27.7, 26.4, 21.7, 45.9]


@pytest.mark.parametrize("latency_s", MEASURED_LATENCY_S)
def test_k10_repro_the_held_route_p_set_flips_before_the_measured_latency(monkeypatch, latency_s):
    """RED on 226aad7105: the gate holds until a seat frees (the measured latency); D decodes throughout,
    so (b) never fires and the wait bound (60 s from the D phase start) comes after the shorter ones.
    Green: the hold bound (15 s after the release) flips first -- once in the epoch, P takes the oldest."""
    _k10(monkeypatch)
    f, now = _full_gate()
    fired_at = None
    t = 0.0
    while t < latency_s:
        f._ipc_out_book().token("r1", now + t)          # D decodes on (gaps <= 15 s measured; here every tick)
        res = _step(f, now + t)
        if res[0]:
            assert fired_at is None, "once per D epoch: no second relief, no pendulum"
            fired_at = t
            assert res == (True, True, None) and f._dc_p_cap == 1 and not Front._k10_rvp_only(f)
        t += 0.5
    assert fired_at is not None and fired_at < latency_s
    assert 15.0 <= fired_at < 15.5
    assert f.counters["k10_seat_relief_p_wait"] == 1


def test_k10_hold_bound_zero_leaves_the_hold_to_the_wait_bound(monkeypatch):
    _k10(monkeypatch, hold="0")
    f, now = _full_gate()
    for t in (0.0, 16.0, 29.0):
        f._ipc_out_book().token("r1", now + t)
        assert _step(f, now + t) == (False, False, None)


@pytest.mark.parametrize("d_streams", [True, False])
def test_k10_a_route_d_seat_wait_is_real_queue_and_never_relieved(monkeypatch, d_streams):
    """The boundary (NF-Sitz 14:28:52Z, weg2-74-295): a collect set released route=D waits for a D seat
    106 s while all 6 seats decode (weg2-40-179 wall 1286 s) -- a real queue, D works. The relief lives
    only on the route-P seat-gate path (_dc_no_seat_step): a route-D wait never fires it, D streaming
    or not (a D that serves nothing is PW-F's, on D). (Other, older rules of the step -- PBOUND for
    over-X entries of the harness queue -- are not this test's subject.)"""
    _k10(monkeypatch)
    f, now = _full_gate()
    f._decode_collect = lambda now_, live_q=None: "D"
    handed = []
    f._dc_hand_to_d = lambda now_: handed.append(now_)
    for t in range(0, 121, 5):
        if d_streams:
            f._ipc_out_book().token("weg2-40-179", now + t)
        _step(f, now + t)
    assert handed, "route D: the set goes to D's admission line"
    assert sum(v for k, v in f.counters.items() if k.startswith("k10_seat_relief")) == 0
    assert not Front._k10_rvp_only(f)


def test_k10_route_p_with_d_decoding_is_relieved_route_d_is_not(monkeypatch):
    """Both cases side by side: the same full D, D streaming. Route P (work only P can prefill) -> the
    hold is bounded at 15 s; route D (work D itself takes when a seat frees) -> no relief."""
    _k10(monkeypatch)
    fp, now = _full_gate()
    fd, _ = _full_gate()
    fd._decode_collect = lambda now_, live_q=None: "D"
    fd._dc_hand_to_d = lambda now_: None
    got_p = None
    for t in [x * 0.5 for x in range(0, 61)]:
        for f in (fp, fd):
            f._ipc_out_book().token("r1", now + t)
        rp = _step(fp, now + t)
        _step(fd, now + t)
        got_p = t if (rp[0] and got_p is None) else got_p
    assert got_p is not None and 15.0 <= got_p < 15.5
    assert fd.counters["k10_seat_relief_p_wait"] == 0 and fd.counters["k10_seat_relief_no_d_progress"] == 0
