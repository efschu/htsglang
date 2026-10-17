"""DECODE-COLLECT (user rule 02.10. ~19:07Z, 27B and NF).

"wenn prefill requests reinkommen und noch decoded wird, dann wird erstmal noch
15 sekunden weiterdecoded und prefill requests gesammelt, erst dann ... je nach
anzahl in P oder D" -- and "auf den alten weg ... indem man die zeit auf 0
stellt". Base d019aa8e1e: an arrival on a decoding D was served at once
(PBOUND-FLIP-NOW / ARRIVAL-SEAT flip_now, the SHORT took its D seat at once).
"""
from __future__ import annotations

import asyncio
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from test_pdflip_arrival_seat_rule_0929 import X, _front, _pending  # noqa: E402

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.managers.scheduler_components import weight_updater as _wu  # noqa: E402,F401


def _on(monkeypatch, window="15"):
    monkeypatch.setenv("FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_RULE", "1")
    monkeypatch.delenv("FLLIPER_PDFLIP_DECODE_COLLECT_D_CHECK_S", raising=False)  # default 7.5
    monkeypatch.delenv("FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_AGE_PLAN", raising=False)
    monkeypatch.delenv("FLLIPER_PDFLIP_PBOUND_FLIP_NOW", raising=False)
    monkeypatch.setenv("FLLIPER_PDFLIP_DECODE_COLLECT_WINDOW_S", window)


def test_an_idle_d_serves_at_once(monkeypatch):
    _on(monkeypatch)
    f = _front(running=[], n=6, kv={"available": 400000, "evictable": 0})
    p = _pending("big", 16448, time.time())
    f.queue = [p]
    assert asyncio.run(f._arrival_seat_step(f.groups["D"], time.time())) == (True, True, p)
    assert f.counters["decode_collect_hold"] == 0


def test_zero_is_the_old_path(monkeypatch):
    _on(monkeypatch, window="0")
    f = _front(running=["a"], n=6, kv={"available": 400000, "evictable": 0})
    p = _pending("big", 16448, time.time())
    f.queue = [p]
    assert asyncio.run(f._arrival_seat_step(f.groups["D"], time.time())) == (True, True, p)
    assert "_dc_state" not in f.__dict__


