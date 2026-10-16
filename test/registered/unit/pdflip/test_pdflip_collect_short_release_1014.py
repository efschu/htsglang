"""COLLECT-LANE 1014: a SHORT whose DECODE-COLLECT window another caller released.

NF h10 (b6e5685092) 13:31:52-13:32:08Z: pdflip-18-143 (lane 1) collected; LANE-PREEMPT parked D's six
decodes; the controller released the window route=D stage=d-idle (13:31:53.933) and booked 18-143 as D
prefill; 9 ms later pdflip-18-142 opened the next window (its `carried` = 18-143's row -> 13601 > X).
18-143's 50-ms poll then read "hold" and stayed in the NEXT window: no D-ADMIT, D idle 15 s, both to P.
Same at 13:33:14.341Z (pdflip-20-151 / pdflip-20-150).
"""
from __future__ import annotations

import asyncio
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from test_pdflip_arrival_seat_rule_0929 import _front  # noqa: E402

from flliper.srt.managers.scheduler_components import weight_updater as _wu  # noqa: E402,F401


def _on(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_RULE", "1")
    monkeypatch.setenv("FLLIPER_PDFLIP_DECODE_COLLECT_D_CHECK_S", "0")
    monkeypatch.delenv("FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_AGE_PLAN", raising=False)
    monkeypatch.delenv("FLLIPER_PDFLIP_PBOUND_FLIP_NOW", raising=False)
    monkeypatch.setenv("FLLIPER_PDFLIP_DECODE_COLLECT_WINDOW_S", "0.3")
    monkeypatch.delenv("FLLIPER_PDFLIP_ENABLE_DECODE_COLLECT_PREFILL_BUSY", raising=False)


def test_a_short_follows_the_release_of_its_window_not_the_next_one(monkeypatch):
    """NODECODE 1015: only a DECODE holds a window now, so a's window is held by one ("dec", no book row) -- a carried prefill
    alone (the old setup: "first" granted 3000) decides a's set at once. The decode ends (the lane park in h10), the controller
    releases a's window to D and books a; b's next window has no decode either, so it is decided AT ONCE (a's 2000 + 5000 > X ->
    P, sticky for the phase) -- before a's poll. a must still follow ITS release (D), not the phase's current P (``wid_route``)."""
    _on(monkeypatch)
    f = _front(running=["dec"], n=6, kv={"available": 400000, "evictable": 0})
    f.tp_prefill_max_tokens = 6008

    async def run():
        a = asyncio.ensure_future(f._decode_collect_short("a", 2000))
        await asyncio.sleep(0.01)  # a holds (a decode stands), its poll sleeps
        f.groups["D"].outstanding.pop("dec")  # D idle (the lane park in h10)
        assert f._decode_collect(time.time()) == "D"  # the controller releases a's window
        b = asyncio.ensure_future(f._decode_collect_short("b", 5000))  # the next window, before a's poll
        await asyncio.sleep(0)
        return await asyncio.gather(a, b)

    ra, rb = asyncio.run(run())
    assert ra == "D", "a's window went to D: a must take its D seat, not follow b's P"
    assert rb == "P", "b: a's 2000 now on D + 5000 > X"
    assert f._dc_st()["route"] == "P"
    assert f.counters["decode_collect_short_release_seen"] == 1
    assert f._d_pf_book().pending_tokens(now=time.time(), live={}) == 2000, "a booked once, b not at all"


def test_a_short_that_sees_its_own_release_is_unchanged(monkeypatch):
    _on(monkeypatch)
    f = _front(running=[], n=6, kv={"available": 400000, "evictable": 0})
    f.tp_prefill_max_tokens = 6008
    f._d_pf_book().grant(rid="first", tokens=3000, now=time.time())
    assert asyncio.run(f._decode_collect_short("a", 800)) == "D"
    assert f.counters["decode_collect_short_release_seen"] == 0
