"""DECODE-COLLECT SEAT GATE (user 02.10. ~20:25Z).

The rule that P does not prefill when D has no room must hold under the
collect window; filling prefills for free seats stay allowed. Base 05e99d7473:
the window's route P flipped at once (``_arrival_seat_step`` ``dc == "P"``),
before the ARRIVAL-SEAT seat count ran, and P then took the whole queue.
"""
from __future__ import annotations

import asyncio
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from test_pdflip_arrival_seat_rule_0929 import _front, _pending  # noqa: E402

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.managers.scheduler_components import weight_updater as _wu  # noqa: E402,F401


def _on(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_RULE", "1")
    monkeypatch.setenv("FLLIPER_PDFLIP_DECODE_COLLECT_WINDOW_S", "15")
    monkeypatch.delenv("FLLIPER_PDFLIP_DECODE_COLLECT_D_CHECK_S", raising=False)
    monkeypatch.delenv("FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_AGE_PLAN", raising=False)
    monkeypatch.delenv("FLLIPER_PDFLIP_PBOUND_FLIP_NOW", raising=False)
    monkeypatch.delenv("FLLIPER_PDFLIP_DECODE_COLLECT_SEAT_GATE", raising=False)


def _past_window(f, now):
    f.t_awake = now - 60.0
    a, b = _pending("s1", 3000, now - 16.0), _pending("s2", 3000, now - 10.0)
    f.queue = [b, a]
    return a


def test_the_seat_gate_is_on_by_default(monkeypatch):
    monkeypatch.delenv("FLLIPER_PDFLIP_DECODE_COLLECT_SEAT_GATE", raising=False)
    assert bool(envs.FLLIPER_PDFLIP_DECODE_COLLECT_SEAT_GATE.get()) is True


def test_no_free_seat_no_flip(monkeypatch):
    """Base: (True, True, s1) -- P prefilled while every D seat was taken."""
    _on(monkeypatch)
    f = _front(running=["a", "b", "c"], n=3, bound=600.0)
    now = time.time()
    _past_window(f, now)
    assert asyncio.run(f._arrival_seat_step(f.groups["D"], now)) == (False, False, None)
    assert f.counters["decode_collect_seat_hold"] == 1
    assert f.counters["arrival_seat_flip_now"] == 0
    assert not getattr(f, "_dc_p_cap", 0)


def test_a_free_seat_flips_and_caps_p_at_the_free_seats(monkeypatch):
    _on(monkeypatch)
    f = _front(running=["a", "b"], n=3, bound=600.0)
    now = time.time()
    head = _past_window(f, now)
    assert asyncio.run(f._arrival_seat_step(f.groups["D"], now)) == (True, True, head)
    assert f._dc_p_cap == 1
    assert f.counters["decode_collect_seat_flip"] == 1


def test_the_wait_bound_parks_the_youngest_when_no_seat_is_free(monkeypatch):
    _on(monkeypatch)
    monkeypatch.setenv("FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_AGE_PLAN", "0")
    f = _front(running=["a", "b", "c"], n=3, bound=5.0)
    now = time.time()
    _past_window(f, now)
    parked = []

    async def _park(D, wait_s, bound, now_):
        parked.append((wait_s, bound))
        return "c"
    f._arrival_seat_park_youngest = _park
    assert asyncio.run(f._arrival_seat_step(f.groups["D"], now)) == (False, False, None)
    assert len(parked) == 1 and parked[0][0] >= 5.0


def test_age_plan_without_room_keeps_waiting(monkeypatch):
    """AGE PLAN (default on): no seat and no displacement plan -> no flip."""
    _on(monkeypatch)
    f = _front(running=["a", "b", "c"], n=3, bound=600.0, kv={"available": 0, "evictable": 0})
    now = time.time()
    _past_window(f, now)
    f._arrival_seat_age_verdict = lambda *a, **k: None
    assert asyncio.run(f._arrival_seat_step(f.groups["D"], now)) == (False, False, None)
    assert f.counters["decode_collect_seat_hold"] == 1
    assert not getattr(f, "_dc_p_cap", 0)


def test_gate_off_is_the_old_route_p(monkeypatch):
    _on(monkeypatch)
    monkeypatch.setenv("FLLIPER_PDFLIP_DECODE_COLLECT_SEAT_GATE", "0")
    f = _front(running=["a", "b", "c"], n=3, bound=600.0)
    now = time.time()
    head = _past_window(f, now)
    assert asyncio.run(f._arrival_seat_step(f.groups["D"], now)) == (True, True, head)
    assert not getattr(f, "_dc_p_cap", 0)


def test_window_zero_never_reaches_the_gate(monkeypatch):
    _on(monkeypatch)
    monkeypatch.setenv("FLLIPER_PDFLIP_DECODE_COLLECT_WINDOW_S", "0")
    f = _front(running=["a", "b", "c"], n=3, bound=600.0)
    now = time.time()
    _past_window(f, now)
    asyncio.run(f._arrival_seat_step(f.groups["D"], now))
    assert f.counters.get("decode_collect_seat_hold", 0) == 0
    assert f.counters.get("decode_collect_seat_flip", 0) == 0
