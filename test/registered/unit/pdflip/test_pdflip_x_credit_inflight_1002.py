# SPDX-License-Identifier: Apache-2.0
"""X-CREDIT-INFLIGHT-1002: the presence credit ignored prefixes realised on D
under a leg 2 that was still decoding.

NF boot dkrnfint4bar1dauer10020634 (5b46b8842e), front log
/spinning/docker-acceptance/nf/evidence/boot_weg2_dkrnfint4bar1dauer10020634_
5b46b8842e_1002_063419.front.log:

  06:39:51.9 'LEG2-FIRST-CONTENT rid=pdflip-12-20 ... via=after_p' (31566 tokens;
             D: '#988 LOADBACK ... anchor_depth=31552', later '#59b
             PARK-RESUMABLE pdflip-12-20=31552'); its leg 2 ends 06:40:44.
  06:39:53.9 'X-EXACT-PRICE rid=pdflip-14-23 pending=11755 tokens=31723
             credit=19968 src=d_leg2_cached ... reused=31566' -> LONG (X=3700),
             PARK-IMMEDIATE, flip pair, 'SERVED group=P ... cached_tokens=31552'
             -- P prefilled 171 tokens, TTFT 8.7 s.

16 of that boot's 30 P leg-1s are this shape. Fix: the first content of a D
leg 2 records the prompt's end anchor as a D presence of the in-flight leg
(``presence_src=d_inflight``), replaced by the finish reading, capped by the
#59b park depth, retracted when the leg ends without a reading and by
ANCHOR-LOST (switch FLLIPER_PDFLIP_ENABLE_D_INFLIGHT_PRESENCE, default on).
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
import types

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip.front_tokens import TokenSpans  # noqa: E402

X = 3700  # the boot's X at pdflip-14-23


def _base(n):
    return np.arange(n, dtype=np.int32)


class _Tok:
    def __init__(self):
        self.m = {}

    def ids_for(self, text):
        return self.m.get(text)


def _front():
    f = object.__new__(F.Front)
    f.counters = collections.Counter()
    f.x_exact = True
    f.epoch = 14
    f.tspans = TokenSpans(agent_span=False)
    f.ftok = _Tok()
    f._x_exact_rid = collections.OrderedDict()
    f._x_exact_reprice_queue = lambda why: 0
    return f


def _boot_state(f):
    """The finished reading the boot had (pdflip-10-18 -> 19968) and the texts."""
    f.ftok.m["pdflip-10-18"] = _base(20100)
    f.tspans.record_presence(_base(20100), 19968, prompt_tokens=20100, resumable_depth=19968)
    f.ftok.m["pdflip-12-20"] = _base(31566)
    return _base(31723)  # pdflip-14-23: reused=31566 of the predecessor


# ---- the metal case -----------------------------------------------------------

def test_pdflip_14_23_priced_short_on_the_in_flight_prefix(caplog):
    f = _front()
    cur = _boot_state(f)
    # base: priced against the finished reading only
    pending, credit, _k, src = f.tspans.pending(cur)
    assert (pending, credit, src) == (11755, 19968, "d_leg2_cached")
    with caplog.at_level(logging.INFO, logger=F.logger.name):
        anchor = f._d_inflight_presence("pdflip-12-20", "pdflip-12-20",
                                        types.SimpleNamespace(leg1_prompt_tokens=31566), "after_p")
    assert anchor == 31552, "P's END-ANCHOR = D's #988 anchor_depth / #59b park depth"
    pending, credit, known, src = f.tspans.pending(cur)
    assert known and credit == 31552 and src == "d_inflight"
    assert pending == 171 <= X, "SHORT: D reads 31552 back and prefills the 171 P prefilled"
    assert ("PDFLIP PRESENCE-INFLIGHT rid=pdflip-12-20 via=after_p anchor=31552 tokens_front=31566 "
            "p_prompt=31566 epoch=14 presence_src=d_inflight") in caplog.text
    assert f.counters["presence_d_inflight"] == 1
    assert f.counters["presence_d_inflight_tokens"] == 31552


def test_x_exact_err_names_the_d_inflight_witness(caplog):
    f = _front()
    cur = _boot_state(f)
    f._d_inflight_presence("pdflip-12-20", "pdflip-12-20",
                           types.SimpleNamespace(leg1_prompt_tokens=31566), "after_p")
    pending, _c, _k, src = f.tspans.pending(cur)
    f.ftok.m["pdflip-14-23"] = cur
    f._x_exact_rid["pdflip-14-23"] = (pending, int(cur.size), src)
    with caplog.at_level(logging.INFO, logger=F.logger.name):
        f._x_exact_record("pdflip-14-23", "pdflip-14-23", 31723, 31552, None, None,
                          resumable_depth=31680)
    assert ("X-EXACT-ERR rid=pdflip-14-23 via=d_direct pending_priced=171 d_uncached=171 err=+0 "
            "tokens_front=31723 tokens_d=31723 match=1 src=d_inflight") in caplog.text


def test_a_prompt_d_prefilled_itself_is_not_credited_in_flight():
    # D's own prefill anchors on its TRACK, not the page floor, in 8 of 54
    # in-flight RETAIN lines of the boot: 'pdflip-4-7 ... token_ids_len=19509
    # cache_len=17728 value=True' (floor would be 19456). Not a realised
    # in-flight reading -- the finish reading covers it.
    f = _front()
    f.ftok.m["pdflip-4-7"] = _base(19509)
    assert f._d_inflight_presence("pdflip-4-7", "pdflip-4-7", None, "d_direct") == 0
    assert f._d_inflight_presence("pdflip-4-7", "pdflip-4-7",
                                  types.SimpleNamespace(leg1_prompt_tokens=0), "after_p") == 0
    assert f.tspans.pending(_base(19900))[1] == 0
    assert f.counters["presence_d_inflight_no_p_anchor"] == 2


def test_never_credits_past_a_divergence_before_the_anchor():
    f = _front()
    f.ftok.m["a"] = _base(31566)
    f._d_inflight_presence("a", "a", types.SimpleNamespace(leg1_prompt_tokens=31566), "after_p")
    other = _base(32000)
    other[31000:] += 10 ** 6  # leaves the in-flight prompt at 31000 < 31552
    assert f.tspans.pending(other)[1] == 0, "PX: no state at the divergence point"


def test_a_deeper_existing_reading_stands():
    ts = TokenSpans(agent_span=False)
    ids = _base(31566)
    ts.record_presence(ids, 31552, prompt_tokens=31566, resumable_depth=31552)
    assert ts.record_d_inflight("r", ids, 31566) == 0
    assert ts.pending(_base(31723))[3] == "d_leg2_cached"


# ---- invalidation: mirrors the finished reading's ------------------------------

def test_finish_reading_replaces_the_in_flight_entry():
    f = _front()
    f.ftok.m["a"] = _base(31566)
    f._d_inflight_presence("a", "a", types.SimpleNamespace(leg1_prompt_tokens=31566), "after_p")
    f._x_exact_record("a", "a", 31566, 31564, types.SimpleNamespace(d_direct=False), None,
                      resumable_depth=31552)
    _p, credit, _k, src = f.tspans.pending(_base(31723))
    assert (credit, src) == (31552, "d_leg2_cached")
    f._d_inflight_end("a")  # the leg's finally: the measurement stays
    assert f.tspans.pending(_base(31723))[1] == 31552
    assert f.counters["presence_d_inflight_retracted"] == 0


def test_leg_end_without_a_finish_reading_retracts(caplog):
    f = _front()
    f.ftok.m["a"] = _base(31566)
    f._d_inflight_presence("a", "a", types.SimpleNamespace(leg1_prompt_tokens=31566), "after_p")
    with caplog.at_level(logging.INFO, logger=F.logger.name):
        f._d_inflight_end("a")
    assert f.tspans.pending(_base(31723))[1] == 0
    assert "PDFLIP PRESENCE-INFLIGHT-END rid=a retracted anchor=31552" in caplog.text
    assert not f.tspans.inflight_keys and not f.tspans.inflight_rid


def test_park_depth_caps_and_a_zero_retracts(caplog):
    f = _front()
    f.ftok.m["a"] = _base(31566)
    f.ftok.m["b"] = _base(24250) + 10 ** 6
    f._d_inflight_presence("a", "a", types.SimpleNamespace(leg1_prompt_tokens=31566), "after_p")
    f._d_inflight_presence("b", "b", types.SimpleNamespace(leg1_prompt_tokens=24250), "after_p")
    body = json.dumps({"parked": ["a", "b"], "held": [],
                       "pdflip_resumable_depth": {"a": 30976, "b": 0, "zz": 5}})
    with caplog.at_level(logging.INFO, logger=F.logger.name):
        f._d_inflight_park(F.park_resumable_depths(body))
    assert f.tspans.pending(_base(31723))[1] == 30976
    assert f.tspans.pending(np.concatenate([_base(24250) + 10 ** 6, _base(100)]))[1] == 0
    assert "PDFLIP PRESENCE-INFLIGHT-PARK rid=a credit 31552 -> 30976" in caplog.text
    # a park depth past the prompt (decoded tokens retained) never raises it
    f.ftok.m["c"] = _base(1000) + 5 * 10 ** 6
    f._d_inflight_presence("c", "c", types.SimpleNamespace(leg1_prompt_tokens=1000), "after_p")
    f._d_inflight_park({"c": 1300})
    assert f.tspans.pending(np.concatenate([_base(1000) + 5 * 10 ** 6, _base(50)]))[1] == 960


def test_park_body_without_depths_is_empty():
    assert F.park_resumable_depths('{"parked": []}') == {}
    assert F.park_resumable_depths("not json") == {}


def test_anchor_lost_retracts_the_in_flight_entry():
    f = _front()
    f.ftok.m["a"] = _base(31566)
    f._d_inflight_presence("a", "a", types.SimpleNamespace(leg1_prompt_tokens=31566), "after_p")
    gone = F.retract_lost_anchors(f.tspans, [31552])
    assert len(gone) == 1
    assert f.tspans.pending(_base(31723))[1] == 0
    assert not f.tspans.inflight_keys


def test_switch_default_on():
    from flliper.srt.environ import envs

    assert envs.FLLIPER_PDFLIP_ENABLE_D_INFLIGHT_PRESENCE.get() is True


# ---- through a real leg 2 ------------------------------------------------------

FIRST = "tools:" + "T" * 900 + "\nsystem:S\nuser:" + "u" * 3000
PT, CT = 1300, 1298
FIRST_IDS = np.arange(PT, dtype=np.int32)
TWIN_IDS = np.arange(PT + 195, dtype=np.int32)


def _asse(event, data) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def _fake_d(state: dict, finish: bool):
    from aiohttp import web

    async def messages(request):
        resp = web.StreamResponse(status=200)
        resp.content_type = "text/event-stream"
        await resp.prepare(request)
        await resp.write(_asse("message_start", {"type": "message_start", "message": {
            "usage": {"input_tokens": 0, "output_tokens": 0}}}))
        await asyncio.sleep(0.05)
        await resp.write(_asse("content_block_start", {"type": "content_block_start", "index": 0,
                                                        "content_block": {"type": "text", "text": ""}}))
        await resp.write(_asse("content_block_delta", {"type": "content_block_delta", "index": 0,
                                                        "delta": {"type": "text_delta", "text": "a"}}))
        state["decoding"].set()
        await state["release"].wait()
        if finish:
            await resp.write(_asse("message_delta", {"type": "message_delta",
                                                      "delta": {"stop_reason": "end_turn"},
                                                      "usage": {"input_tokens": PT - CT,
                                                                "cache_read_input_tokens": CT,
                                                                "output_tokens": 1}}))
            await resp.write(_asse("message_stop", {"type": "message_stop"}))
        await resp.write_eof()
        return resp

    async def info(request):
        return web.json_response({})

    app = web.Application()
    app.router.add_post("/v1/messages", messages)
    app.router.add_get("/get_server_info", info)
    return app


async def _mid_stream_price(finish: bool = True):
    import aiohttp
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    state = {"decoding": asyncio.Event(), "release": asyncio.Event()}
    d = TestServer(_fake_d(state, finish))
    await d.start_server()
    front = F.Front("http://p", str(d.make_url("")).rstrip("/"), "D", "t", "", 0, 0, {}, 45.0,
                    carrier_max_tokens=27466, tp_prefill_max_tokens=X)
    front.groups["D"].url = str(d.make_url("")).rstrip("/")
    front.session = aiohttp.ClientSession()
    front.epoch = 7
    front.x_exact = True
    front.ftok = _Tok()
    front.ftok.m[FIRST] = FIRST_IDS
    front.tspans = TokenSpans(agent_span=False)
    # an after_p leg 2: P served leg 1 with PT prompt tokens
    pending = F.Pending("r1", "/v1/messages", {}, FIRST, 0.0,
                        asyncio.get_running_loop().create_future(), est_prompt=PT,
                        est_uncached=PT)
    pending.leg1_prompt_tokens = PT

    async def handler(request):
        payload = await request.json()
        return await front.leg2(request, "r1", payload, FIRST, True, pending)

    app = web.Application()
    app.router.add_post("/v1/messages", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    host, port = runner.addresses[0][:2]
    try:
        async with aiohttp.ClientSession() as cs:
            post = asyncio.ensure_future(cs.post(f"http://{host}:{port}/v1/messages",
                                                 json={"messages": [], "stream": True}))
            await asyncio.wait_for(state["decoding"].wait(), 10)
            await asyncio.sleep(0.2)  # the front has read the content head
            mid = front.tspans.pending(TWIN_IDS)
            state["release"].set()
            resp = await post
            await resp.read()
        await asyncio.sleep(0.05)
        end = front.tspans.pending(TWIN_IDS)
    finally:
        await runner.cleanup()
        await front.session.close()
        await d.close()
    return front, mid, end


def test_leg2_credits_the_in_flight_prompt_mid_stream(caplog):
    with caplog.at_level(logging.INFO):
        front, mid, end = asyncio.run(_mid_stream_price(finish=True))
    # mid-stream: the twin is priced on the in-flight anchor (1300 -> 1280)
    assert mid == (PT + 195 - 1280, 1280, True, "d_inflight"), mid
    assert any("PDFLIP PRESENCE-INFLIGHT rid=r1 via=after_p anchor=1280" in r.getMessage()
               for r in caplog.records)
    # the finish's reading replaced it (no #59 depth on this wire: ct stands)
    assert end == (PT + 195 - CT, CT, True, "d_leg2_cached"), end
    assert front.counters["presence_d_inflight_retracted"] == 0


def test_leg2_ending_without_usage_retracts_the_in_flight_entry(caplog):
    with caplog.at_level(logging.INFO):
        front, mid, end = asyncio.run(_mid_stream_price(finish=False))
    assert mid[3] == "d_inflight"
    assert end[1] == 0 and end[3] == "none", end
    assert front.counters["presence_d_inflight_retracted"] == 1


def test_leg2_switch_off_records_nothing(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_ENABLE_D_INFLIGHT_PRESENCE", "0")
    front, mid, _end = asyncio.run(_mid_stream_price(finish=True))
    assert mid == (PT + 195, 0, False, "none"), mid
    assert front.counters["presence_d_inflight"] == 0
