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


def test_a_short_that_sees_its_own_release_is_unchanged(monkeypatch):
    _on(monkeypatch)
    f = _front(running=[], n=6, kv={"available": 400000, "evictable": 0})
    f.tp_prefill_max_tokens = 6008
    f._d_pf_book().grant(rid="first", tokens=3000, now=time.time())
    assert asyncio.run(f._decode_collect_short("a", 800)) == "D"
    assert f.counters["decode_collect_short_release_seen"] == 0
