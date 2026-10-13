"""PENDEL-FLIP: PBOUND-FLIP-NOW must not undo the DECODE-COLLECT SEAT GATE (NF int24f3, 13.10.).

Boot ..._cfeb2896c1_1013_045811 (analysis nf-flip-halt-analyse-1013): in 45 D phases
D woke, ran 0 decode rounds and flipped back to P after 0.1-1.9 s (43 of them
"long halts" in flip.log; 45 round trips P>D>P, 290 s in 2.1 h, no decode step).
The chain: the seat gate caps a P phase at the free D seats (``_dc_p_cap``, mostly
1; D is 5-7/6 full) and leaves the rest queued ON PURPOSE; after the P->D flip the
DONE-KICK asks ``_arrival_seat_step``, whose PBOUND-FLIP-NOW ("no seat/KV test, no
dwell") treats exactly that rest as "P-bound work waits" and flips at once
(front.log 05:55:10.638, ARRIVAL-SEAT verdict=flip_now PBOUND-FLIP-NOW, taken=11 n=6).

Pinned: the rest the seat gate left behind (record ``_pbound_seat_held``, armed at the
end of the capped P phase) is HELD while D has no free seat -- D decodes -- and released
by the first free seat or by the existing wait bound. A P-bound request that was not in
the queue when the capped phase ended, and every phase without a gate cap, behave as
before. Hermetic, no CUDA.
"""
from __future__ import annotations

import asyncio
import collections
import inspect
import logging
import os
import time
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from test_weg2_arrival_seat_rule_0929 import X, _front, _pending  # noqa: E402

from sglang.srt.managers.scheduler_components import weight_updater as _wu  # noqa: E402,F401
from sglang.srt.weg2 import front as front_mod  # noqa: E402

Front = front_mod.Front
MARK = "PBOUND-HELD-BY-SEAT-GATE"


def _on(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_ENABLE_ARRIVAL_SEAT_RULE", "1")
    monkeypatch.delenv("SGLANG_WEG2_ENABLE_ARRIVAL_SEAT_AGE_PLAN", raising=False)  # default on
    monkeypatch.delenv("SGLANG_WEG2_PBOUND_FLIP_NOW", raising=False)              # default on
    monkeypatch.delenv("SGLANG_WEG2_DECODE_COLLECT_SEAT_GATE", raising=False)      # default on
    monkeypatch.delenv("SGLANG_WEG2_DECODE_COLLECT_WINDOW_S", raising=False)       # 15 s, only new arrivals


def _pendel(awake_s=1.0, n=6, running=6, bound=60.0):
    """The Pendel state: D just woke (``awake_s`` ago), 6/6 seats taken, four
    P-bound requests (LONG over X) left in the queue by a seat-gate-capped P phase
    (cap 1, ``end=leaving queue_left=4``). Returns (front, now, [queued, oldest first])."""
    now = time.time()
    f = _front(running=["r%d" % i for i in range(running)], n=n, bound=bound,
               kv={"available": 400000, "evictable": 0})
    f.t_awake = now - awake_s
    q = [_pending("held%d" % i, 87000, now - 60.0 + i) for i in range(4)]   # arrived before D woke
    f.queue = list(q)
    f.p_phase_max_requests = 6
    Front._pbound_hold_arm(f, 1, 1)                                         # the capped P phase ended
    f.__dict__["_pbound_seat_held"]["t"] = f.t_awake - 0.5                  # armed before D woke
    return f, now, q


def _step(f, now):
    return asyncio.run(f._arrival_seat_step(f.groups["D"], now))


# -------------------------------------------------------------------- the Pendel repro (red on b60520f01d)

def test_pendel_d_full_and_gate_held_rest_does_not_flip_back(monkeypatch, caplog):
    """RED on b60520f01d: (True, True, held0) = PBOUND-FLIP-NOW 1 s after the wake,
    D has run 0 rounds. Fixed: D keeps decoding, the marker names the hold."""
    _on(monkeypatch)
    f, now, q = _pendel()
    with caplog.at_level(logging.INFO, logger=front_mod.logger.name):
        res = _step(f, now)
    assert res == (False, False, None), "D decodes first: no D->P flip for the gate's rest"
    assert f.counters["arrival_seat_pbound_flip_now"] == 0
    assert f.counters["arrival_seat_flip_now"] == 0
    assert f.counters["pbound_held_by_seat_gate"] == 1
    lines = [r.getMessage() for r in caplog.records if MARK in r.getMessage()]
    assert len(lines) == 1 and "free_seats=0" in lines[0] and "epoch=7" in lines[0] and "rids=" in lines[0]
    assert "held0" in lines[0]


def test_the_marker_is_told_once_per_hold_not_once_per_tick(monkeypatch, caplog):
    _on(monkeypatch)
    f, now, _ = _pendel()
    with caplog.at_level(logging.INFO, logger=front_mod.logger.name):
        for k in range(5):
            assert _step(f, now + 0.2 * k) == (False, False, None)
    assert sum(MARK in r.getMessage() for r in caplog.records) == 1
    assert f.counters["pbound_held_by_seat_gate"] == 1


def test_pendel_with_the_real_two_flip_sequence_gives_d_a_decode_window(monkeypatch):
    """Before: tick 1 flips. After: every tick until a seat frees holds; the freed seat
    releases the hold and the OLDEST of the rest flips (arrival order kept)."""
    _on(monkeypatch)
    f, now, q = _pendel()
    D = f.groups["D"]
    for k in range(10):                      # D decodes: 10 ticks of 0.2 s, no seat frees
        assert _step(f, now + 0.2 * k) == (False, False, None)
    D.outstanding.pop("r5")                  # a decode finishes: seat 6 is free
    assert _step(f, now + 2.2) == (True, True, q[0]), "the oldest held request flips, as PBOUND always picked"
    assert f.counters["pbound_seat_hold_released_free_seat"] == 1
    assert "_pbound_seat_held" not in f.__dict__
    assert _step(f, now + 2.4) == (True, True, q[0]), "released for good: no second hold in this phase"


# -------------------------------------------------------------------- no starving

def test_the_wait_bound_releases_the_hold_when_no_seat_ever_frees(monkeypatch):
    _on(monkeypatch)
    f, now, q = _pendel(awake_s=30.0, bound=60.0)
    assert _step(f, now) == (False, False, None)            # 30 s awake: still held
    assert _step(f, now + 31.0) == (True, True, q[0])       # 61 s: the bound (d_wait_bound_s) fired
    assert f.counters["pbound_seat_hold_released_wait_bound"] == 1


def test_without_a_wait_bound_the_fairness_w_s_is_the_bound(monkeypatch):
    _on(monkeypatch)
    f, now, q = _pendel(awake_s=44.0, bound=0.0)
    f.w_s = 45.0
    assert _step(f, now) == (False, False, None)
    assert _step(f, now + 2.0) == (True, True, q[0])


def test_a_held_request_that_leaves_the_queue_ends_the_hold(monkeypatch):
    """A rid that left the queue (served, client gone) is pruned; the last one ends the record."""
    _on(monkeypatch)
    f, now, q = _pendel()
    f.queue = [q[1], q[2], q[3]]                                  # held0 left
    assert Front._pbound_seat_hold(f, f.queue, now)[1] == [q[1], q[2], q[3]]
    assert f.__dict__["_pbound_seat_held"]["rids"] == {"held1", "held2", "held3"}
    f.queue = []
    assert Front._pbound_seat_hold(f, [q[1]], now) == ([q[1]], [])
    assert "_pbound_seat_held" not in f.__dict__


# -------------------------------------------------------------------- what must stay as before

def test_a_new_p_bound_arrival_still_flips_now_and_the_rest_rides_along(monkeypatch):
    """A request that was not in the queue when the capped P phase ended is a real new
    arrival: PBOUND-FLIP-NOW unchanged (and it acts on the NEW one, not a held one)."""
    _on(monkeypatch)
    f, now, q = _pendel()
    new = _pending("new", 87000, f.t_awake - 0.3)    # queued while P->D flipped: after the capped phase ended
    f.queue.append(new)
    assert _step(f, now) == (True, True, new)
    assert f.counters["arrival_seat_pbound_flip_now"] == 1


def test_no_gate_cap_no_hold_pbound_is_unchanged_on_a_full_d(monkeypatch):
    """The ordinary full-D PBOUND case of test_weg2_pbound_flip_now_1002 (no record)."""
    _on(monkeypatch)
    f = _front(running=["r%d" % i for i in range(6)], n=6, kv={"available": 400000, "evictable": 0})
    p = _pending("big", 87000, time.time() - 30.0)
    f.queue = [p]
    assert "_pbound_seat_held" not in f.__dict__
    assert _step(f, time.time()) == (True, True, p)
    assert f.counters["pbound_held_by_seat_gate"] == 0


def test_switch_off_pbound_never_reads_the_record(monkeypatch):
    _on(monkeypatch)
    monkeypatch.setenv("SGLANG_WEG2_PBOUND_FLIP_NOW", "0")
    f, now, q = _pendel()
    _step(f, now)
    assert f.counters["pbound_held_by_seat_gate"] == 0


def test_a_free_seat_in_a_fresh_pendel_state_flips_at_once(monkeypatch):
    _on(monkeypatch)
    f, now, q = _pendel(running=5)
    assert _step(f, now) == (True, True, q[0])
    assert f.counters["pbound_seat_hold_released_free_seat"] == 1


def test_the_record_is_inert_in_a_phase_that_started_before_it(monkeypatch):
    """A stale record (``t_awake`` older than the arm) must not hold anything."""
    _on(monkeypatch)
    f, now, q = _pendel()
    f.__dict__["_pbound_seat_held"]["t"] = f.t_awake + 5.0
    assert _step(f, now) == (True, True, q[0])


# -------------------------------------------------------------------- D-direct yield (the SHORT path)

def test_d_direct_short_does_not_yield_to_a_flip_that_the_hold_suppresses(monkeypatch):
    """H5b: a SHORT yields to P's batch only if a D->P flip is foreseeable. The held rest
    flips nothing, so the SHORT would otherwise ride P, queue as a NEW P-bound request and
    trigger the very flip the hold suppresses."""
    _on(monkeypatch)
    f, now, q = _pendel()
    assert f._d_direct_yields("short") is False
    assert f.counters["d_direct_yield"] == 0


def test_d_direct_short_still_yields_to_a_new_p_bound_request(monkeypatch):
    _on(monkeypatch)
    f, now, q = _pendel()
    f.queue.append(_pending("new", 87000, f.t_awake - 0.3))
    assert f._d_direct_yields("short") is True


# -------------------------------------------------------------------- arming at the end of the P phase

def _armed(queue, dc_cap, dispatched, mx=6, stalled=False):
    f = types.SimpleNamespace(queue=queue, p_phase_max_requests=mx, epoch=9, counters=collections.Counter())
    Front._pbound_hold_arm(f, dc_cap, dispatched, stalled=stalled)
    return f.__dict__.get("_pbound_seat_held")


def test_arm_records_what_the_binding_gate_cap_left_in_the_queue():
    q = [_pending("a", 1, 1.0), _pending("b", 1, 2.0)]
    rec = _armed(q, 1, 1)
    assert rec is not None and rec["rids"] == {"a", "b"} and rec["epoch"] == 9


def test_arm_records_nothing_without_a_gate_cap():
    assert _armed([_pending("a", 1, 1.0)], 0, 6) is None            # PBOUND/fairness P phase: H91 cap only


def test_arm_records_nothing_when_the_cap_was_not_reached():
    assert _armed([_pending("a", 1, 1.0)], 3, 2) is None            # queue left for another reason


def test_arm_records_nothing_when_the_h91_cap_bound_first():
    assert _armed([_pending("a", 1, 1.0)], 8, 8, mx=6) is None      # gate cap above the phase cap


def test_arm_records_nothing_after_an_intake_stall_and_skips_stalled_entries():
    q = [_pending("a", 1, 1.0)]
    assert _armed(q, 1, 1, stalled=True) is None
    q2 = [_pending("a", 1, 1.0), _pending("b", 1, 2.0)]
    q2[0].intake_stalled = True                                       # it must go to P again anyway
    assert _armed(q2, 1, 1)["rids"] == {"b"}


def test_arm_equal_caps_still_count_as_the_gates_rest():
    assert _armed([_pending("a", 1, 1.0)], 6, 6, mx=6)["rids"] == {"a"}


def test_arm_empty_queue_leaves_no_record():
    assert _armed([], 1, 1) is None


def test_arm_overwrites_an_older_record():
    f = types.SimpleNamespace(queue=[], p_phase_max_requests=6, epoch=9)
    f.__dict__["_pbound_seat_held"] = {"rids": {"old"}, "t": 0.0, "epoch": 1, "told": True}
    Front._pbound_hold_arm(f, 0, 0)
    assert "_pbound_seat_held" not in f.__dict__


# -------------------------------------------------------------------- wiring / interactions

def test_wiring_the_p_phase_arms_and_clears_and_the_step_filters():
    src = inspect.getsource(front_mod)
    assert 'self.__dict__.pop("_pbound_seat_held", None)  # PENDEL-FLIP' in src
    assert "Front._pbound_hold_arm(self, _dc_cap" in src
    step = inspect.getsource(Front._arrival_seat_step)
    assert "Front._pbound_seat_hold(self, cands, now)" in step
    assert step.index("_pbound_seat_hold") < step.index("_asr_pbound_flip_now(cands")
    assert "_pbound_seat_hold" in inspect.getsource(Front._d_direct_yields)


def test_arming_sits_after_the_drain_and_before_the_flip_back():
    src = inspect.getsource(front_mod)
    i_drain = src.index("passes = await _p_drain_pool(")
    i_arm = src.index("Front._pbound_hold_arm(self, _dc_cap")
    i_log = src.index('"WEG2 P-PHASE epoch=%d dispatched=%d cap=%d prefilled=%d end=%s ')
    assert i_drain < i_arm < i_log


def test_collectp_a_fresh_route_p_set_with_a_free_seat_flips_under_the_gate_as_before(monkeypatch):
    """DECODE-COLLECT route P (a NEW arrival collected during this D phase) is not touched
    by the hold: with a free seat it flips, capped at the free seats (``_dc_p_cap``)."""
    _on(monkeypatch)
    f, now, q = _pendel(running=5)
    f.t_awake = now - 60.0
    f.__dict__["_pbound_seat_held"]["t"] = f.t_awake - 0.5
    fresh = _pending("fresh", 87000, now - 16.0)
    f.queue.append(fresh)
    res = _step(f, now)
    assert res[0] is True and res[1] is True
    assert f._dc_p_cap == 1 and f.counters["decode_collect_seat_flip"] == 1


def test_collectp_on_a_full_d_the_gate_keeps_its_own_no_seat_path(monkeypatch):
    """A route-P set on a full D goes to ``_dc_no_seat_step`` (seat-gate hold) -- the hold
    does not add a second path; nothing flips without a displacement plan."""
    _on(monkeypatch)
    f, now, q = _pendel()
    f.t_awake = now - 60.0
    f.__dict__["_pbound_seat_held"]["t"] = f.t_awake - 0.5
    f.queue.append(_pending("fresh", 87000, now - 16.0))
    f._arrival_seat_age_verdict = lambda *a, **k: None
    assert _step(f, now) == (False, False, None)
    assert f.counters["decode_collect_seat_hold"] == 1
    assert f.counters["pbound_held_by_seat_gate"] == 0               # the hold line is PBOUND's, not route P's


def test_earlycap_leg1_early_limit_still_reads_the_cap_and_never_the_hold(monkeypatch):
    """_leg1_early_limit reads ``_dc_p_cap`` (never pops it); the hold record does not change it."""
    f = types.SimpleNamespace(queue=[_pending("q%d" % i, 1, 1.0) for i in range(4)], p_concurrency=6, epoch=3)
    f.__dict__["_dc_p_cap"] = 2
    f.__dict__["_pbound_seat_held"] = {"rids": {"q0"}, "t": 0.0, "epoch": 2, "told": False}
    a = Front._leg1_early_limit(f)
    assert a == Front._leg1_early_limit(f) and f.__dict__["_dc_p_cap"] == 2
    f2 = types.SimpleNamespace(queue=f.queue, p_concurrency=6, epoch=3)
    assert Front._leg1_early_limit(f2) >= a
