"""ARRIVAL-SEAT KV READ BUDGET (NF D->P flip, 01.10.).

bfpgwv (front/D logs ...dauer10011823_a11cc7cbc6_1001_182321), D->P flip
pdflip-14-90 (epoch 15):

* 18:36:44.069 ``PDFLIP-ROUTE rid=pdflip-14-90 BATCH queued`` (LONG, uncached 94368)
* D TP0 was inside a D prefill pass (3507 tokens, 5.4 s wall); it ended at
  18:36:45.639 (``PDFLIP-VRAM-PEAK phase=chunk t_unix_ms=1790879805639``) and
  the next one (3520 tokens) ran to 18:36:50.857
* 18:36:46.070 ``ARRIVAL-SEAT rid=pdflip-14-90 verdict=flip_now ... kv=unread``
  -- the controller tick had awaited D's ``/server_info`` for its whole 2.0 s
  timeout: D answers it only at its pass boundary
* 18:36:46.071 park RPC issued, 18:36:51.210 ``PARK-RUNNING ... rpc_s=5.14``
  (D's own park dispatch: 316 ms) -- the park waited for the NEXT boundary
* 18:36:51.214 ``PDFLIP-FLIP begin``; DP-WAIT hold_s=7.1 of wait_s=9.0

All six bfpgwv D->P flips over 4 s (arrival -> first P chunk) carry
``kv=unread`` after exactly 2.0 s; NF park dispatch is 0.12-0.45 s (median
0.32 s), the park RPC's tail is the wait for D's pass boundary.

Pinned: an older reading is refreshed in the background; the tick waits at
most the budget, then decides on the last reading D gave; the refresh lands
for the next tick. No reading at all (boot, after a KV park) still waits for
D. A read D answered before a KV park never lands after it.
"""
from __future__ import annotations

import asyncio
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from test_pdflip_arrival_seat_rule_0929 import _front, _on, _pending  # noqa: E402

from flliper.srt.pdflip import arrival_seat_rule as asr  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _pbound_flip_now_off(monkeypatch):
    """These tests pin the seat/KV verdict and the dwell holds for requests that
    need P -- the rule PBOUND-FLIP-NOW (default on, 02.10.) replaces; they keep
    covering it as the switch-off path."""
    monkeypatch.setenv("FLLIPER_PDFLIP_PBOUND_FLIP_NOW", "0")


FITS = {"available": 400000, "evictable": 0}       # pdflip-14-90's ladder: free 436000
SHORT = {"available": 1000, "evictable": 0}


class _Resp:
    def __init__(self, body):
        self.status = 200
        self._body = body

    async def json(self):
        return self._body


class _Get:
    def __init__(self, sess):
        self.sess = sess

    async def __aenter__(self):
        self.sess.calls += 1
        body = {"internal_states": [{"pdflip_kv": dict(self.sess.reading)}]}
        await asyncio.sleep(self.sess.delay)        # D answers at its pass boundary
        return _Resp(body)

    async def __aexit__(self, *a):
        return False


class _Session:
    """D's /server_info: answers after ``delay`` (the rest of D's pass)."""

    def __init__(self, reading, delay=0.0):
        self.reading, self.delay, self.calls = reading, delay, 0

    def get(self, url, timeout=None):
        return _Get(self)


def _real_reader(f, sess):
    f.__dict__.pop("_arrival_seat_kv_reading", None)     # the harness stub off: the real read
    f.session = sess
    return f


def test_budget_default_and_env():
    assert asr.kv_read_budget_s({}) == 0.05
    assert asr.kv_read_budget_s({"FLLIPER_PDFLIP_ARRIVAL_KV_READ_BUDGET_S": "0"}) == 0.0
    assert asr.kv_read_budget_s({"FLLIPER_PDFLIP_ARRIVAL_KV_READ_BUDGET_S": "-1"}) == 0.0
    assert asr.kv_read_budget_s({"FLLIPER_PDFLIP_ARRIVAL_KV_READ_BUDGET_S": "x"}) == 0.05
    assert "arrival_seat_kv_stale" in asr.COUNTERS


def test_flip_now_does_not_wait_for_d_inside_a_pass(monkeypatch):
    """pdflip-14-90: D is inside a 1.2 s pass (the metal: 5.4 s, the read timed
    out at 2.0 s). The verdict must not wait for D's boundary -- the park RPC
    it starts waits for that boundary by itself."""
    _on(monkeypatch)
    f = _real_reader(_front(running=["a"], n=6), _Session(FITS, delay=0.0))

    async def run():
        assert await f._arrival_seat_kv_reading() == FITS          # boot: the first read waits
        st = f._asr_st()
        st["kv"] = (time.time() - 3.0, st["kv"][1])                 # 3 s old: a refresh is due
        f.session.delay = 1.2                                       # D is mid-pass now
        p = _pending("pdflip-14-90", 94368, time.time())
        f.queue = [p]
        t0 = time.monotonic()
        got = await f._arrival_seat_step(f.groups["D"], time.time())
        dt = time.monotonic() - t0
        assert got == (True, True, p)
        assert dt < 0.5, f"the flip_now verdict waited {dt:.2f} s for D's pass boundary"
        assert f.counters["arrival_seat_flip_now"] == 1
        assert f.counters["arrival_seat_kv_stale"] == 1
        # the refresh lands in the background for the next tick
        await asyncio.sleep(1.4)
        assert time.time() - st["kv"][0] < 2.0 and st["kv"][1] == FITS
        assert f.session.calls == 2

    asyncio.run(run())


def test_a_fresh_answer_within_the_budget_is_taken():
    f = _real_reader(_front(running=["a"], n=6), _Session(FITS, delay=0.0))

    async def run():
        await f._arrival_seat_kv_reading()
        st = f._asr_st()
        st["kv"] = (time.time() - 3.0, SHORT)                        # the old reading said short
        got = await f._arrival_seat_kv_reading()                     # D between passes: answers now
        assert got == FITS and f.counters["arrival_seat_kv_stale"] == 0

    asyncio.run(run())


def test_no_reading_still_waits_for_d_and_a_pre_park_read_never_lands():
    """After a KV park the next KV test reads D afresh and waits for it (the
    park takes effect at D's boundary); a read asked before the park that
    D answers later must not overwrite that."""
    f = _real_reader(_front(running=["a"], n=6), _Session(SHORT, delay=0.0))

    async def run():
        await f._arrival_seat_kv_reading()
        st = f._asr_st()
        st["kv"] = (time.time() - 3.0, SHORT)
        f.session.delay = 0.3
        assert await f._arrival_seat_kv_reading() == SHORT          # stale, refresh in flight
        old = st["kv_task"][1]
        f._asr_kv_invalidate()                                       # a KV park happened
        f.session.reading, f.session.delay = FITS, 0.05
        t0 = time.monotonic()
        assert await f._arrival_seat_kv_reading() == FITS           # waited for a NEW read
        assert time.monotonic() - t0 >= 0.04
        await old                                                    # the pre-park read ends ...
        assert st["kv"][1] == FITS                                   # ... and never lands

    asyncio.run(run())
