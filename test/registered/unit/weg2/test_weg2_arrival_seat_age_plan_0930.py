"""ARRIVAL-SEAT AGE PLAN (Nutzer 30.09., the seat policy, wörtlich):

~14:40Z: „der große wartende füllt dann wenn eben andere große freigeben auf.
der ältere ist immer bevorzugt, außer es sind mehrere noch ältere große schon
drauf und ein jüngerer kleiner würde neben die ältesten großen passen. dann
wird der jüngere parallel bedient. und sobald keine ältesten mehr da sind die
den kv für den mittleren blockieren, wird der mittlere auf die karten gezogen
und verdrängt ggf. auch den jüngsten, der gerade dort decoded hat komplett,
dafür könnten dann andere noch jüngere nachziehen. usw.“
~14:45Z: „natürlich muss ein jüngerer nur verdrängt werden, wenn der ältere
nicht draufpasst. nicht pauschal den jüngeren verdrängen“

The front before (ARRIVAL-SEAT + #246, 1f749bdad0):
* the wait bound (c) parked the youngest ADMITTED decode once the oldest
  waiter passed 60 s -- whether or not that made it fit, whether or not the
  victim was younger than it, and even when it fit already (pauschal);
* the same bound ENDED the backfill: past 60 s a younger small that fits
  beside older runners waited, although nothing the head could use was free;
* an older waiter found every seat held by YOUNGER decodes -> nothing until
  the bound (the KV displacement acts only with a free seat);
* a D-prefill waiter's head was the oldest D-prefill waiter only: a younger
  small took a free seat past an OLDER queued request that fits (needs P).
"""
from __future__ import annotations

import asyncio
import json
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from test_weg2_arrival_seat_rule_0929 import X, _front, _pending  # noqa: E402

from sglang.srt.managers.scheduler_components import weight_updater as _wu  # noqa: E402,F401
from sglang.srt.weg2 import arrival_seat_rule as asr  # noqa: E402
from sglang.srt.weg2 import front as F  # noqa: E402


def _on(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_ENABLE_ARRIVAL_SEAT_RULE", "1")
    monkeypatch.setenv("SGLANG_WEG2_ENABLE_ARRIVAL_SEAT_AGE_PLAN", "1")


def _stamp(f, rid, t):
    f.__dict__.setdefault("_asr_arrive", {})[rid] = t


def _freeing_rpc(f, kv, tokens):
    """D parks the named decode; its KV comes back into D's free reading."""
    async def rpc(g, path, body, timeout):
        f.rpc_calls.append((path, dict(body)))
        v = body.get("youngest")
        if v:
            kv["available"] = int(kv.get("available", 0)) + int(tokens.get(v, 0))
            f.groups["D"].outstanding.pop(v, None)
        return 200, json.dumps({"parked": [v] if v else [], "held": []})
    f.rpc = rpc


# ---------------------------------------------------------------- pure

def test_the_plan_displaces_nobody_when_the_head_fits():
    assert asr.displace_plan(20.0, True, 0, ["y"], {"y": 30.0}, {"y": 5000}) == []


def test_the_plan_is_the_fewest_youngest_younger_decodes():
    arr = {"old": 10.0, "y1": 30.0, "y2": 40.0, "y3": 50.0}
    tok = {"old": 90000, "y1": 3000, "y2": 5000, "y3": 2000}
    run = ["old", "y1", "y2", "y3", "unstamped"]
    assert asr.displace_plan(20.0, True, 1500, run, arr, tok) == ["y3"]
    assert asr.displace_plan(20.0, True, 6000, run, arr, tok) == ["y3", "y2"]
    assert asr.displace_plan(20.0, True, 10000, run, arr, tok) == ["y3", "y2", "y1"]
    assert asr.displace_plan(20.0, True, 10001, run, arr, tok) is None     # never an elder
    assert asr.displace_plan(20.0, False, 0, run, arr, tok) == ["y3"]      # a seat: one
    assert asr.displace_plan(60.0, False, 0, run, arr, tok) is None        # only elders run
    assert asr.displace_plan(None, False, 0, run, arr, tok) is None


def test_the_switch_needs_the_rule():
    assert asr.age_plan_enabled({"SGLANG_WEG2_ENABLE_ARRIVAL_SEAT_AGE_PLAN": "1"}) is False
    assert asr.age_plan_enabled({"SGLANG_WEG2_ENABLE_ARRIVAL_SEAT_AGE_PLAN": "1",
                                 "SGLANG_WEG2_ENABLE_ARRIVAL_SEAT_RULE": "1"}) is True


# ---------------------------------------------------------------- nicht pauschal

def test_the_bound_parks_nobody_when_elders_hold_every_seat(monkeypatch):
    """The blanket (c): seats full with OLDER decodes (no arrival stamp = old),
    the big waited 120 s -> the base parks 'b', the youngest admitted. Parking
    an elder for a younger one is against the age rule: nobody parks."""
    _on(monkeypatch)
    f = _front(running=["a", "b", "c"], n=3, admit_t={"a": 1.0, "b": 3.0, "c": 2.0},
               kv={"available": 400000, "evictable": 0})
    now = time.time()
    f.t_awake = now - 200.0
    f.queue = [_pending("big", 16448, now - 120.0)]
    res = asyncio.run(f._arrival_seat_step(f.groups["D"], now))
    assert not f.rpc_calls, "nicht pauschal: nobody is displaced for a head its elders block"
    assert f.counters["arrival_seat_youngest_park"] == 0
    assert res == (False, False, None)


def test_a_head_that_fits_displaces_nobody_even_past_the_bound(monkeypatch):
    """Seat free, KV fits, the head waited 120 s: the base parks the youngest
    first and flips anyway. The plan flips and parks nobody."""
    _on(monkeypatch)
    f = _front(running=["a", "b"], n=6, admit_t={"a": 1.0, "b": 2.0},
               kv={"available": 400000, "evictable": 0})
    now = time.time()
    f.t_awake = now - 200.0
    _stamp(f, "a", now - 30.0)
    _stamp(f, "b", now - 20.0)
    p = _pending("big", 16448, now - 120.0)
    _stamp(f, "big", now - 120.0)
    f.queue = [p]
    res = asyncio.run(f._arrival_seat_step(f.groups["D"], now))
    assert res == (True, True, p)
    assert not f.rpc_calls, "the head fits: nobody is displaced"


def test_an_older_head_takes_a_seat_from_the_youngest_at_once(monkeypatch):
    """Every seat held, the one running decode arrived AFTER the head, the
    head's KV fits: the youngest parks now -- not after the 60 s bound."""
    _on(monkeypatch)
    now = time.time()
    kv = {"available": 400000, "evictable": 0, "decode_clip": 4096}
    f = _front(running=["young"], n=1, kv=kv, admit_t={"young": 1.0})
    _stamp(f, "young", now + 0.5)          # arrives after the head below
    F.Seat(f, "young", "short", tokens=3000)
    _freeing_rpc(f, kv, {"young": 3000})

    async def run():
        t = asyncio.ensure_future(f._acquire_short_seat("head", 2000, [], max_tokens=256))
        _stamp(f, "head", now)
        await asyncio.sleep(0.3)
        assert f.rpc_calls, "the older head displaced the younger at once"
        assert f.rpc_calls[0][1]["youngest"] == "young"
        assert f.rpc_calls[0][1]["reason"] == asr.REASON_AGE
        assert len(f.rpc_calls) == 1
        t.cancel()

    asyncio.run(run())


def test_the_kv_displacement_parks_only_as_many_as_needed(monkeypatch):
    """The control of minimality: two younger decodes, the youngest alone
    covers the deficit -> exactly one park, then the head is granted."""
    _on(monkeypatch)
    now = time.time()
    kv = {"available": 20000, "evictable": 0, "decode_clip": 4096}
    f = _front(running=["old", "y1", "y2"], n=5, kv=kv, admit_t={"old": 1.0, "y1": 2.0, "y2": 3.0})
    _stamp(f, "old", now - 100.0)
    _stamp(f, "head", now - 10.0)
    _stamp(f, "y1", now - 5.0)
    _stamp(f, "y2", now - 4.0)
    tok = {"old": 90000, "y1": 30000, "y2": 25000}
    for r, t in tok.items():
        F.Seat(f, r, "short", tokens=t)
    _freeing_rpc(f, kv, tok)

    async def run():
        seat = await asyncio.wait_for(f._acquire_short_seat("head", 40000, [], max_tokens=64000), 3.0)
        assert seat is not None and seat.rid == "head"
        assert [b["youngest"] for _p, b in f.rpc_calls] == ["y2"]

    asyncio.run(run())


# ---------------------------------------------------------------- backfill has no time limit

def test_a_younger_small_backfills_past_a_head_its_elders_block_even_past_the_bound(monkeypatch):
    """The head needs more KV than its elders leave and no younger decode can
    free it: blocked by elders. A younger small that fits runs in parallel --
    on the base only until the bound, then it waits with the room unused."""
    _on(monkeypatch)
    now = time.time()
    kv = {"available": 20000, "evictable": 0, "decode_clip": 4096}
    f = _front(running=["old"], n=5, kv=kv, bound=0.1)
    _stamp(f, "old", now - 100.0)
    F.Seat(f, "old", "short", tokens=90000)

    async def run():
        h = asyncio.ensure_future(f._acquire_short_seat("head", 40000, [], max_tokens=64000))
        _stamp(f, "head", now)
        await asyncio.sleep(0.3)                         # the head is past the 0.1 s bound
        seat = await asyncio.wait_for(f._acquire_short_seat("small", 1000, [], max_tokens=256), 1.0)
        assert seat is not None and seat.rid == "small"
        assert f.counters["arrival_seat_backfill"] == 1
        assert not h.done() and not f.rpc_calls
        h.cancel()

    asyncio.run(run())


# ---------------------------------------------------------------- one age order across the lines

def test_a_younger_small_does_not_take_the_seat_of_an_older_queued_request_that_fits(monkeypatch):
    """An older request that needs P fits a free seat (the step flips for it
    at its next tick); a younger small (<= X) arriving meanwhile must not take
    the seat first -- on the base its head is the oldest D-prefill waiter only."""
    _on(monkeypatch)
    now = time.time()
    f = _front(running=["a"], n=2, kv={"available": 400000, "evictable": 0, "decode_clip": 4096})
    big = _pending("big", X + 12000, now - 2.0, payload={"max_tokens": 256})
    f.queue = [big]

    async def run():
        t = asyncio.ensure_future(f._acquire_short_seat("small", 1000, [], max_tokens=256))
        await asyncio.sleep(0.3)
        assert not t.done(), "the older queued request keeps its right of way"
        res = await f._arrival_seat_step(f.groups["D"], time.time())
        assert res == (True, True, big)
        t.cancel()

    asyncio.run(run())
