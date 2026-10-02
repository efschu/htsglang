# SPDX-License-Identifier: Apache-2.0
"""PRICE-BARRIER (02.10.): a SHORT priced beside a LONG candidate waits for its
verdict before taking a D seat; a LONG takes the SHORT along into P's batch.

N5x (..._5ddc067a81_1002_162100) 16:23:40.933: weg2-0-1 (25 tokens, 2 out) and
weg2-0-2 (74148 tokens) left the same BOOT-START HOLD together. The SHORT's
count finished first: 'D-ADMIT rid=weg2-0-1 ... source=short' at 41.147, the
LONG's verdict at 41.158, PARK-SEAT-FREE FIRE, PARK-RUNNING waited 236 ms for
the SHORT's first extend, and the SHORT sat parked through the whole P phase:
wall 19.1 s. Agent load 01./02.10.: ~1 such case per 1000 arrivals.

Switch SGLANG_WEG2_PRICE_BARRIER (default on). Marker 'WEG2 PRICE-BARRIER'.

PINNED: the N5x boot-start pair goes to P as a group and D does no extend for
the SHORT; a SHORT beside a candidate that ends SHORT loses at most that
count time and goes to D as before; nothing in flight costs nothing; a sibling
that never reaches its verdict (client gone) is not waited for; full seats /
P awake / switch off keep today's path. Hermetic, CPU, no HTTP.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.weg2 import front as F  # noqa: E402
from sglang.srt.weg2 import front_tokens as FT  # noqa: E402

X = 4096


class _Req:
    def __init__(self, payload, path="/v1/messages"):
        self._p = payload
        self.path = path

    async def json(self):
        return self._p


class _Tokens:
    """A front tokenizer: ``n`` tokens and ``delay`` seconds per payload tag,
    one worker thread (counts queue like the real executor). ``loading`` holds
    every route decision until ``release`` (the BOOT-START HOLD)."""

    def __init__(self, table, loading=False):
        from concurrent.futures import ThreadPoolExecutor

        self.table = table
        self.state, self.why = ("loading" if loading else "ready"), "fake"
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.ids_by_text = {}

    def count(self, path, payload):
        n, delay = self.table[payload["tag"]]
        time.sleep(delay)
        return FT.Count(n=n, ids=np.arange(n, dtype=np.int32) + 10, ms=delay * 1000.0,
                        reused=0, encoded=n)

    def remember(self, text, ids):
        self.ids_by_text[text] = ids

    def ids_for(self, text):
        return self.ids_by_text.get(text)


def _front(table, loading=False):
    with envs.SGLANG_WEG2_FRONT_EXACT_TOKENS.override(True):
        f = F.Front(prefill="http://p", decode="http://d", awake="D", tag="pbarrier",
                    store_dir="/tmp", prefill_sid=0, decode_sid=0, dc_reserve={}, w_s=45.0,
                    weight_chunks=2, tp_prefill_max_tokens=X, flip_min_work_tokens=X)
    f.state = "serving"
    f.ftok = _Tokens(table, loading=loading)
    f.d_legs = []

    async def seat(rid, est, refused=None, **kw):
        return 1

    async def leg2(request, rid, payload, text, stream, pending=None, **kw):
        f.d_legs.append((rid, time.monotonic()))  # a D extend for this rid
        return F.web.json_response({})

    async def solo(rid, rem):
        return True

    f._acquire_short_seat = seat
    f.leg2 = leg2
    f._x_solo_admits = solo
    f._kick_controller = lambda *a, **k: None
    return f


def _payload(tag, chars):
    return {"model": "m", "max_tokens": 2, "tag": tag,
            "messages": [{"role": "user", "content": "a" * chars}]}


SHORT = _payload("short", 75)            # chars/3 25: no candidate
LONG = _payload("long", 160000)          # chars/3 ~53k > X/2: candidate


async def _run(f, payloads, *, release_after=None, settle=0.6, cancel_tag=None, cancel_after=0.02):
    tasks = {}
    for p in payloads:
        tasks[p["tag"]] = asyncio.create_task(f.handle_generate(_Req(p)))
        await asyncio.sleep(0.005)
    if release_after is not None:
        await asyncio.sleep(release_after)
        f.ftok.state = "ready"
        f._x_exact_ready_event().set()
    if cancel_tag is not None:
        await asyncio.sleep(cancel_after)
        tasks[cancel_tag].cancel()
    await asyncio.sleep(settle)
    for t in tasks.values():
        if not t.done():
            t.cancel()
    await asyncio.gather(*tasks.values(), return_exceptions=True)


def _lines(caplog, mark="WEG2 PRICE-BARRIER"):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith(mark)]


def test_red_n5x_boot_start_pair_goes_to_p_as_a_group_and_d_does_no_extend(caplog):
    # the SHORT's count is the slower one but runs first (it waited longer in the
    # hold); the LONG's count queues behind it -- N5x: 127 ms, then 81 ms
    f = _front({"short": (25, 0.12), "long": (74148, 0.08)}, loading=True)
    with caplog.at_level(logging.INFO, logger="weg2.front"):
        asyncio.run(_run(f, [SHORT, LONG], release_after=0.05))
    assert f.d_legs == [], f"D extended {f.d_legs} -- the SHORT took a D seat beside the LONG"
    queued = {p.rid: p for p in f.queue}
    assert len(queued) == 2, [p.rid for p in f.queue]
    short_p = [p for p in f.queue if p.est_prompt == 25][0]
    assert short_p.d_eligible is False and short_p.skip_leg1 is False, "it rides P's batch"
    assert f.counters["price_barrier_to_p"] == 1
    lines = _lines(caplog)
    assert any("outcome=to_p" in l and "siblings=1" in l for l in lines), lines
    assert any("outcome=to_p sibling=" in l and "sibling_uncached=74148" in l for l in lines), lines
    assert sum(1 for r in caplog.records if r.getMessage().startswith("WEG2 X-EXACT-HOLD")) == 2


def test_a_sibling_that_ends_short_costs_at_most_its_count_time(caplog):
    # candidate by chars/3 (~2700 > X/2) but exactly 3000 <= X: SHORT
    sib = _payload("sib", 8100)
    f = _front({"short": (25, 0.01), "sib": (3000, 0.20)})
    with caplog.at_level(logging.INFO, logger="weg2.front"):
        # the SHORT arrives first and is counted first; the sibling's count runs
        # behind it, so the SHORT's verdict finds the sibling still being priced
        asyncio.run(_run(f, [SHORT, sib], settle=0.8))
    rids = [r for r, _ in f.d_legs]
    assert len(rids) == 2, f"both SHORTs go to D as before: {f.d_legs}"
    line = [l for l in _lines(caplog) if "outcome=to_d" in l]
    assert line, _lines(caplog)
    waited = float(line[0].split("waited_ms=")[1].split()[0])
    assert waited <= 200 + 80, f"waited {waited} ms -- more than the sibling's count time"
    assert f.counters["price_barrier_to_p"] == 0


def test_nothing_in_flight_costs_nothing(caplog):
    f = _front({"short": (25, 0.01)})
    with caplog.at_level(logging.INFO, logger="weg2.front"):
        asyncio.run(_run(f, [SHORT], settle=0.2))
    assert [r for r, _ in f.d_legs] and not _lines(caplog)
    assert f.counters["price_barrier_waits"] == 0


def test_a_sibling_whose_handler_ends_before_its_verdict_is_not_waited_for(caplog):
    f = _front({"short": (25, 0.05), "long": (74148, 0.6)})
    with caplog.at_level(logging.INFO, logger="weg2.front"):
        t0 = time.monotonic()
        # the SHORT (counted first) waits for the LONG; the LONG's client leaves
        # at 0.15 s, long before its 0.6 s count would end
        asyncio.run(_run(f, [SHORT, LONG], cancel_tag="long", cancel_after=0.14, settle=0.3))
    assert f.d_legs, "the SHORT goes to D"
    assert f.d_legs[0][1] - t0 < 0.4, "it waited for a pricing that could not come"
    assert any("outcome=to_d" in l for l in _lines(caplog)), _lines(caplog)
    assert not (f.__dict__.get("_pb_inflight") or {}), "the cancelled LONG left its entry"


def test_switch_off_is_todays_path_the_n5x_short_takes_its_d_seat(caplog):
    f = _front({"short": (25, 0.12), "long": (74148, 0.08)}, loading=True)
    with envs.SGLANG_WEG2_PRICE_BARRIER.override(False), \
            caplog.at_level(logging.INFO, logger="weg2.front"):
        asyncio.run(_run(f, [SHORT, LONG], release_after=0.05))
    assert len(f.d_legs) == 1 and not _lines(caplog)
    assert len(f.queue) == 1 and f.queue[0].est_prompt == 74148


def test_full_seats_and_an_awake_p_keep_their_paths(caplog):
    f = _front({"short": (25, 0.12), "long": (74148, 0.08)}, loading=True)
    f.seats_free = lambda: 0
    with caplog.at_level(logging.INFO, logger="weg2.front"):
        asyncio.run(_run(f, [SHORT, LONG], release_after=0.05))
    assert not _lines(caplog), "no free seat: the SHORT would not be admitted now -- no barrier"
    g = _front({"short": (25, 0.12), "long": (74148, 0.08)}, loading=True)
    g.awake = "P"
    with caplog.at_level(logging.INFO, logger="weg2.front"):
        asyncio.run(_run(g, [SHORT, LONG], release_after=0.05))
    assert g.d_legs == [] and g.counters["price_barrier_waits"] == 0
    assert g.counters["short_behind_p"] == 1


def test_the_candidate_rule():
    f = F.Front.__new__(F.Front)
    f.tp_prefill_max_tokens = X

    async def go():
        a = F.Front._pb_register(f, "a", X // 2)        # at the fraction: no candidate
        b = F.Front._pb_register(f, "b", X // 2 + 1)
        return a, b

    a, b = asyncio.run(go())
    assert a is None and b is not None
    assert envs.SGLANG_WEG2_PRICE_BARRIER.get() is True
