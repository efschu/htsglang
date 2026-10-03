"""DECODE-COLLECT SEAT GATE (user 02.10. ~20:25Z), ported to the 27B front (item 340).

The rule that P does not prefill when D has no room must hold under the
collect window; filling prefills for free seats stay allowed. Base 05e99d7473:
the window's route P flipped at once (``_arrival_seat_step`` ``dc == "P"``),
before the ARRIVAL-SEAT seat count ran, and P then took the whole queue.
27B port (base 4d4eaee868, same gap): the gate lives in the ARRIVAL-SEAT branch
only; the classic ``_immediate_park_due`` path (27B default, no ARRIVAL-SEAT) is
untouched and proven unchanged by the last block.
"""
from __future__ import annotations

import asyncio
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from test_weg2_arrival_seat_rule_0929 import _front, _pending  # noqa: E402

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.scheduler_components import weight_updater as _wu  # noqa: E402,F401


def _on(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_ENABLE_ARRIVAL_SEAT_RULE", "1")
    monkeypatch.setenv("SGLANG_WEG2_DECODE_COLLECT_WINDOW_S", "15")
    monkeypatch.delenv("SGLANG_WEG2_DECODE_COLLECT_D_CHECK_S", raising=False)
    monkeypatch.delenv("SGLANG_WEG2_ENABLE_ARRIVAL_SEAT_AGE_PLAN", raising=False)
    monkeypatch.delenv("SGLANG_WEG2_PBOUND_FLIP_NOW", raising=False)
    monkeypatch.delenv("SGLANG_WEG2_DECODE_COLLECT_SEAT_GATE", raising=False)


def _past_window(f, now):
    f.t_awake = now - 60.0
    a, b = _pending("s1", 3000, now - 16.0), _pending("s2", 3000, now - 10.0)
    f.queue = [b, a]
    return a


def test_the_seat_gate_is_on_by_default(monkeypatch):
    monkeypatch.delenv("SGLANG_WEG2_DECODE_COLLECT_SEAT_GATE", raising=False)
    assert bool(envs.SGLANG_WEG2_DECODE_COLLECT_SEAT_GATE.get()) is True


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
    monkeypatch.setenv("SGLANG_WEG2_ENABLE_ARRIVAL_SEAT_AGE_PLAN", "0")
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
    monkeypatch.setenv("SGLANG_WEG2_DECODE_COLLECT_SEAT_GATE", "0")
    f = _front(running=["a", "b", "c"], n=3, bound=600.0)
    now = time.time()
    head = _past_window(f, now)
    assert asyncio.run(f._arrival_seat_step(f.groups["D"], now)) == (True, True, head)
    assert not getattr(f, "_dc_p_cap", 0)


def test_window_zero_never_reaches_the_gate(monkeypatch):
    _on(monkeypatch)
    monkeypatch.setenv("SGLANG_WEG2_DECODE_COLLECT_WINDOW_S", "0")
    f = _front(running=["a", "b", "c"], n=3, bound=600.0)
    now = time.time()
    _past_window(f, now)
    asyncio.run(f._arrival_seat_step(f.groups["D"], now))
    assert f.counters.get("decode_collect_seat_hold", 0) == 0
    assert f.counters.get("decode_collect_seat_flip", 0) == 0


# ---------------------------------------------------------------- the classic path stays as it was

def _classic_run(monkeypatch, gate):
    """The 27B default: ARRIVAL-SEAT off, ``_immediate_park_due`` with the window
    released to P (all D seats taken). Returns (is_head, counters, cap)."""
    from sglang.srt.weg2 import front as F
    monkeypatch.delenv("SGLANG_WEG2_ENABLE_ARRIVAL_SEAT_RULE", raising=False)
    monkeypatch.setenv("SGLANG_WEG2_DECODE_COLLECT_WINDOW_S", "15")
    monkeypatch.delenv("SGLANG_WEG2_DECODE_COLLECT_D_CHECK_S", raising=False)
    if gate is None:
        monkeypatch.delenv("SGLANG_WEG2_DECODE_COLLECT_SEAT_GATE", raising=False)
    else:
        monkeypatch.setenv("SGLANG_WEG2_DECODE_COLLECT_SEAT_GATE", gate)
    f = _front(running=["a", "b", "c"], n=3, bound=600.0)  # no free seat
    now = time.time()
    head = _past_window(f, now)
    f._park_attempt_epoch = -1
    got = F.Front._immediate_park_due(f, f.groups["D"], now)
    return got is head, dict(f.counters), getattr(f, "_dc_p_cap", 0)


def test_classic_path_is_byte_identical_whatever_the_gate_says(monkeypatch):
    """Without ARRIVAL-SEAT the route-P park fires on the oldest, no seat looked
    at, no cap armed: the same with the switch unset (default on), on and off."""
    runs = [_classic_run(monkeypatch, g) for g in (None, "1", "0")]
    for is_head, counters, cap in runs:
        assert is_head, "route P parks on the oldest queued request, as before the port"
        assert cap == 0
        assert counters.get("decode_collect_seat_hold", 0) == 0
        assert counters.get("decode_collect_seat_flip", 0) == 0
        assert counters.get("decode_collect_release_P") == 1
    assert runs[0][1] == runs[1][1] == runs[2][1]


def test_classic_source_is_untouched():
    """``_immediate_park_due`` carries no reference to the seat gate."""
    import inspect
    from sglang.srt.weg2 import front as F
    src = inspect.getsource(F.Front._immediate_park_due)
    assert "_dc_p_seat_gate" not in src and "SEAT_GATE" not in src and "_dc_p_cap" not in src
