"""SEAT-AT-HAND-OFF (user 15.10., Defekt 3 "Sitz-Sperre haelt P-Releases 20-74 s").

Design: P computes the prefill and hands it over; only the HAND-OFF to D needs a seat, and that wait is the
queue on D (``_ready_for_d`` / ``d_admitter``: priority = age, the head never skipped). The DECODE-COLLECT SEAT
GATE of 02.10. asked for the seat at the RELEASE instead, with a count (``_arrival_seat_taken``) that includes
D's own wait queue -- so a route-P set waited for a free D seat while P idled.

Measured (front logs, release route=P -> ``WEG2-FLIP begin``, ``seat-gate hold`` in between), boots
..._110016 (h16r), ..._124245 and ..._133855 (h16p): 49 holds, ``taken`` > n in 27, 12 of them 20..74 s
(V4 FLIP-LATENZ 52.9 / 73.5 / 27.7 / 26.4 / 21.7 / 45.9 / 20.5 / 29.7 s ...); 119 P phases capped at the free
seats, 95 of them left ``queue_left`` > 0 (one more flip each). The relief valves built on the gate
(PBOUND-SEAT-HOLD 13.10., K10 (a)/(b)/(c) 15.10.) are gone with it.

Pinned here: the release flips at the first tick whatever D's seats say (both call sites), the P phase carries
no seat cap, the hand-off still takes a seat oldest first, a route-D wait stays a queue, and the removed
gate / cap / valve names are not in the tree. Hermetic, no CUDA.
"""
from __future__ import annotations

import asyncio
import inspect
import os
import time

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from test_weg2_arrival_seat_rule_0929 import _front, _pending  # noqa: E402
from test_weg2_sched_fix3_0907 import _bare_pending, _front_with_reading, _until  # noqa: E402

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.scheduler_components import weight_updater as _wu  # noqa: E402,F401
from sglang.srt.weg2 import front as front_mod  # noqa: E402

Front = front_mod.Front

#: V4 FLIP-LATENZ of the h16p boots 124245 / 133855 (the user's list) -- the time a released route-P set sat at the gate
MEASURED_LATENCY_S = [52.9, 73.5, 27.7, 26.4, 21.7, 45.9, 20.5, 29.7]


def _on(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_ENABLE_ARRIVAL_SEAT_RULE", "1")
    monkeypatch.setenv("SGLANG_WEG2_DECODE_COLLECT_WINDOW_S", "15")
    monkeypatch.delenv("SGLANG_WEG2_DECODE_COLLECT_D_CHECK_S", raising=False)
    monkeypatch.delenv("SGLANG_WEG2_ENABLE_ARRIVAL_SEAT_AGE_PLAN", raising=False)  # default on
    monkeypatch.delenv("SGLANG_WEG2_PBOUND_FLIP_NOW", raising=False)              # default on


def _full_d_with_a_collected_set(running=6, ready=5, n=6, now=None):
    """The 13:05:10 shape (boot 124245, epoch 30): D full (6 running), five prefilled requests waiting in
    ``_ready_for_d`` for a seat -- ``taken=11 n=6`` -- and a P-bound set (two over X together) whose collect
    window ended: DECODE-COLLECT releases route P."""
    now = now or time.time()
    f = _front(running=["r%d" % i for i in range(running)], n=n, bound=600.0)
    f.t_awake = now - 60.0
    f._ready_for_d = [_pending("rdy%d" % i, 3000, now - 100.0 + i) for i in range(ready)]
    a, b = _pending("s1", 3000, now - 16.0), _pending("s2", 3000, now - 10.0)
    f.queue = [b, a]
    f._park_attempt_epoch = -1
    return f, now, a


def _step(f, now):
    return asyncio.run(f._arrival_seat_step(f.groups["D"], now))


def _free_a_seat(f):
    """What the gate waited for: D's wait queue drained and seats freed (taken < n)."""
    f._ready_for_d = []
    for r in list(f.groups["D"].outstanding)[:3]:
        f.groups["D"].outstanding.pop(r)


@pytest.mark.parametrize("latency_s", MEASURED_LATENCY_S)
def test_repro_a_released_route_p_set_flips_at_once_not_when_a_seat_frees(monkeypatch, latency_s):
    """RED on 3b1e746e0e: ``seat-gate hold`` -- the flip came at ``latency_s`` (the measured wait for a seat),
    K10 (c) only at 15 s. Green: the first tick after the release flips, D's seats are not asked."""
    _on(monkeypatch)
    f, now, head = _full_d_with_a_collected_set()
    assert f._arrival_seat_taken() == (11, 6), "the measured premise: D's wait queue counts as taken"
    fired_at, t = None, 0.0
    while t <= latency_s:
        if t >= latency_s - 0.5:
            _free_a_seat(f)  # the seat the gate waited for frees at the measured latency
        res = _step(f, now + t)
        if res[0] and fired_at is None:
            fired_at = t
            assert res == (True, True, head)
        t += 0.5
    assert fired_at == 0.0, "flipped at the release, not after %.1f s" % latency_s
    assert not f.__dict__.get("_dc_p_cap"), "no seat cap rides on the flip"


def test_the_immediate_park_path_flips_at_the_release_too(monkeypatch):
    """The second call site (``_immediate_park_due``): ``dc == "P"`` returns the head with D full."""
    _on(monkeypatch)
    f, now, head = _full_d_with_a_collected_set()
    got = f._immediate_park_due(f.groups["D"], now)
    assert got is head
    assert not f.__dict__.get("_dc_p_cap")


def test_the_flip_decision_never_reads_the_seat_count(monkeypatch):
    """Whatever ``taken`` is, the flip is the same: a counter that explodes would have held the gate."""
    _on(monkeypatch)
    f, now, head = _full_d_with_a_collected_set()
    calls = []
    real = f._arrival_seat_taken
    f._arrival_seat_taken = lambda: (calls.append(1), real())[1]
    assert _step(f, now) == (True, True, head)
    assert calls == [], "route P: the seat count is not consulted on the way to the flip"


def test_an_empty_queue_after_the_release_is_no_flip(monkeypatch):
    """Guard: the released SHORTs still on their way to the queue are not a flip (unchanged)."""
    _on(monkeypatch)
    f, now, _ = _full_d_with_a_collected_set()
    f.queue[:] = []
    f._decode_collect = lambda now_, live_q=None: "P"
    assert _step(f, now) == (False, False, None)


# ------------------------------------------------------------------ the hand-off is where the seat is taken

def test_the_hand_off_takes_a_seat_oldest_first_and_waits_for_one():
    """P prefilled four requests while D is full (2 seats): the first two take the seats, the others wait in the
    queue on D in arrival order and move up one by one as seats free."""
    async def body():
        f = _front_with_reading(None, d_bs=2)
        ps = [_bare_pending("r%d" % i) for i in range(4)]
        f._ready_for_d.extend(ps)
        f._sync_batch_gate()
        task = asyncio.create_task(f.d_admitter())
        for p in ps[:2]:
            assert await _until(lambda p=p: p.fut.done()), "%s not admitted" % p.rid
            p.posted_evt.set()
        await asyncio.sleep(0.3)
        assert not ps[2].fut.done() and not ps[3].fut.done(), "D is full: the hand-off waits, P is not blocked by it"
        assert list(f._ready_for_d) == [ps[2], ps[3]], "the queue on D keeps arrival order"
        ps[0].seat.release("leg2_finished")
        assert await _until(lambda: ps[2].fut.done()), "the freed seat goes to the oldest waiter"
        ps[2].posted_evt.set()
        await asyncio.sleep(0.2)
        assert not ps[3].fut.done()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(body())


# ------------------------------------------------------------------ the P phase carries no seat cap

def test_the_p_phase_is_capped_by_the_h91_cap_and_the_pool_only():
    src = inspect.getsource(front_mod)
    i = src.index("SEAT-AT-HAND-OFF (15.10.): no D seat cap on this phase")
    blk = src[i:i + 900]
    assert "_max_dispatch = self.p_phase_max_requests" in blk
    assert "min(_max_dispatch" not in blk and "_dc_cap" not in blk
    assert "_k10_only" not in src and "k10_rvp" not in src


def test_early_legs_are_posted_for_what_the_p_phase_takes():
    q = [_pending("a", 9000, 1.0), _pending("b", 9000, 2.0), _pending("c", 9000, 3.0)]
    for p in q:
        p.d_direct = p.client_gone = False
        p._leg1_early = None
    assert [p.rid for p in front_mod.leg1_early_candidates(q, 6)] == ["a", "b", "c"]
    src = inspect.getsource(front_mod)
    assert src.count("leg1_early_candidates(self.queue, self.p_concurrency)") == 2


# ------------------------------------------------------------------ route D stays a queue

def test_a_route_d_seat_wait_is_a_real_queue_and_nothing_flips_or_parks(monkeypatch):
    """Gegenprobe (NF-Sitz 14:28:52Z, weg2-74-295): a collect set released route=D waits 106 s for a D seat
    while 6 seats decode. A real queue: the REAL ``_decode_collect`` routes a small set (sum <= X, nothing P-bound,
    past the D-check) to D, it goes to D's admission line, nothing flips, nothing parks -- the seat wait of a
    request D itself takes is not touched by this change (and K10 (c), which never fired here, is gone)."""
    _on(monkeypatch)
    f, now, _ = _full_d_with_a_collected_set()
    f.queue[:] = [_pending("short0", 300, now - 10.0), _pending("short1", 300, now - 9.0)]
    handed = []
    f._dc_hand_to_d = lambda now_: handed.append(now_)
    for t in range(0, 121, 5):
        res = _step(f, now + t)
        assert res[0] is False, "route D never flips D->P"
    assert f.counters["decode_collect_release_D"] == 1 and f.counters["decode_collect_release_P"] == 0
    assert handed, "route D: the set goes to D's admission line (the queue)"
    assert not [k for k in f.counters if "park" in k or "relief" in k]
    assert f.counters["arrival_seat_flip_now"] == 0


# ------------------------------------------------------------------ unchanged neighbours (were in the PBOUND-SEAT-HOLD file)

def test_pbound_flip_now_on_a_full_d_is_unchanged(monkeypatch):
    """PBOUND-FLIP-NOW (02.10. "Request kommt = sofort Prefill") never needed a D seat -- the hand-off does."""
    _on(monkeypatch)
    f = _front(running=["r%d" % i for i in range(6)], n=6, kv={"available": 400000, "evictable": 0})
    p = _pending("big", 87000, time.time() - 30.0)
    f.queue = [p]
    assert _step(f, time.time()) == (True, True, p)


def test_a_d_direct_short_still_yields_to_a_p_bound_request(monkeypatch):
    _on(monkeypatch)
    f = _front(running=["r%d" % i for i in range(6)], n=6, kv={"available": 400000, "evictable": 0})
    f.queue = [_pending("new", 87000, time.time() - 1.0)]
    assert f._d_direct_yields("short") is True
    f.queue = []
    assert f._d_direct_yields("short") is False


# ------------------------------------------------------------------ the removed names stay removed

def test_the_gate_the_cap_and_the_valves_are_gone():
    src = inspect.getsource(front_mod)
    for name in ("_dc_p_seat_gate", "_dc_no_seat_step", "_k10_seat_relief", "_k10_early_filter", "_dc_p_cap",
                 "_pbound_hold_arm", "_pbound_seat_hold", "_pbound_seat_held", "K10-SEAT-RELIEF",
                 "seat-gate hold", "seat-gate flip", "seat-gate p-cap", "_leg1_early_limit"):
        assert name not in src, name
    for env in ("SGLANG_WEG2_DECODE_COLLECT_SEAT_GATE", "SGLANG_WEG2_K10_SEAT_RELIEF", "SGLANG_WEG2_K10_STALL_S",
                "SGLANG_WEG2_K10_HOLD_MAX_S", "SGLANG_WEG2_K10_RELIEF_P_WAIT"):
        assert not hasattr(envs, env), env
