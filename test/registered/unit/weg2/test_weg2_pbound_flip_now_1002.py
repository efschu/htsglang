"""PBOUND-FLIP-NOW (user law 02.10. "Request kommt = sofort Prefill").

y8a (d11000a830) 18:25:28Z: weg2-10-22, an image request (724 tokens, P-only by
the vision rule, routed LONG -> P) arrived on an awake D and got
'ARRIVAL-SEAT verdict=wait_seat why=kv taken=3 n=6 -- no flip: AGE PLAN'. It
waited 229.3 s (DP-WAIT hold_by=d-work, d_admitted_during_wait=17) until a D>P
flip at 18:29:17; 17 younger requests were admitted to D meanwhile. A request
that needs P never needs a D seat or D KV before P prefills it: it flips now,
and nothing younger is admitted to D past it.
"""
from __future__ import annotations

import asyncio
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from test_weg2_arrival_seat_rule_0929 import X, _front, _pending  # noqa: E402

from sglang.srt.managers.scheduler_components import weight_updater as _wu  # noqa: E402,F401


def _on(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_ENABLE_ARRIVAL_SEAT_RULE", "1")
    monkeypatch.delenv("SGLANG_WEG2_ENABLE_ARRIVAL_SEAT_AGE_PLAN", raising=False)  # default on
    monkeypatch.delenv("SGLANG_WEG2_PBOUND_FLIP_NOW", raising=False)              # default on


def _image(rid, t):
    p = _pending(rid, 724, t)
    p.p_only = True
    return p


def test_a_p_only_arrival_whose_kv_does_not_fit_flips_now(monkeypatch):
    """y8a weg2-10-22: free seats (3/6) but the KV test says no -> base:
    wait_seat why=kv, no flip. Fixed: flip_now, no seat/KV test."""
    _on(monkeypatch)
    f = _front(running=["a", "b", "c"], n=6, kv={"available": 100, "evictable": 0})
    p = _image("img", time.time())
    f.queue = [p]
    res = asyncio.run(f._arrival_seat_step(f.groups["D"], time.time()))
    assert res == (True, True, p), "a request that needs P flips D->P at once"
    assert f.counters["arrival_seat_wait_kv"] == 0
    assert f.counters["arrival_seat_pbound_flip_now"] == 1


def test_an_over_x_arrival_with_no_free_seat_flips_now(monkeypatch):
    """No free D seat: P prefills it first, the seat is the P->D re-admission's
    business -- base: wait_seat why=seat."""
    _on(monkeypatch)
    f = _front(running=["a"], n=1, kv={"available": 400000, "evictable": 0})
    p = _pending("big", 16448, time.time())
    f.queue = [p]
    assert asyncio.run(f._arrival_seat_step(f.groups["D"], time.time())) == (True, True, p)
    assert f.counters["arrival_seat_wait_seat"] == 0


def test_the_oldest_p_bound_request_acts_and_a_long_wait_is_loud(monkeypatch):
    _on(monkeypatch)
    f = _front(running=["a"], n=6, kv={"available": 100, "evictable": 0})
    now = time.time()
    f.t_awake = now - 300.0
    old, young = _image("old", now - 10.0), _pending("young", 16448, now - 1.0)
    f.queue = [young, old]
    assert asyncio.run(f._arrival_seat_step(f.groups["D"], now)) == (True, True, old)
    assert f.counters["arrival_seat_pbound_stall"] == 1


def test_no_younger_request_is_admitted_to_d_past_a_p_bound_elder(monkeypatch):
    """Base: the elder counted as 'blocked by its elders' (its KV did not fit,
    no displacement plan) and younger D-prefill arrivals took seats past it."""
    _on(monkeypatch)
    f = _front(running=["a"], n=6, kv={"available": 100, "evictable": 0})
    now = time.time()
    f.queue = [_image("img", now - 5.0)]
    blocked = asyncio.run(f._asr_older_blocked("short", now, True))
    assert blocked is False, "an older request that needs P keeps its right of way"


def test_switch_off_keeps_the_seat_verdict(monkeypatch):
    _on(monkeypatch)
    monkeypatch.setenv("SGLANG_WEG2_PBOUND_FLIP_NOW", "0")
    f = _front(running=["a"], n=1, kv={"available": 400000, "evictable": 0})
    f.queue = [_pending("big", 16448, time.time())]
    assert asyncio.run(f._arrival_seat_step(f.groups["D"], time.time())) == (False, False, None)
    assert f.counters["arrival_seat_wait_seat"] == 1
