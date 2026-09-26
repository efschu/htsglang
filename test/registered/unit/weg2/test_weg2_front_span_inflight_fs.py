"""#49 rest (FS 26.09.): the front-span gap that #49 left, and the unified-tree switch.

MEASURED on the three agent-load boots that ran WITH #49 (27B line):
``dkr27bbar1agent09252206`` (INT8), ``dkr27bnvfp4bar1agent09252328`` (NVFP4),
``dkr27bbar1final09260145`` (INT8), front logs under
/spinning/docker-acceptance/27b/evidence.

THE TWIN. Every Claude-Code turn carries a second request 1-10 s after the
first whose prompt is the first one's prompt plus ~155 tokens (the #49 replay
calls it "side"). When it arrives, the first is still DECODING on D -- D has
prefilled its whole prompt into the radix, and a concurrent request matches
it there (2206: weg2-12-11 matched 32972 of the still-running weg2-12-10's
33003 tokens) -- but the front learned the text only when that leg FINISHED.
So the twin was priced against an older entry and went over P:

* 2206 weg2-14-15: 157 real tokens behind the carried weg2-12-14 that D was
  serving, priced 5327 > X=4096, LONG, 55 s wait (LATE-BATCH).
* nvfp4 weg2-10-7: 138 real tokens behind weg2-8-6 on D, priced 6123, LONG,
  a P epoch of its own (flip pair) plus the rider weg2-11-8.
* nvfp4 weg2-12-13 / weg2-20-55: 156 / 159 real tokens behind the running
  twin, priced 5242 / 4259 > X_busy=4096 (X-SOLO verdict=p) -> P.

THE FIX (switch ``SGLANG_WEG2_FRONT_SPAN_INFLIGHT``, default off; UNIFY UN6:
the unified tree runs #49 itself unswitched since S7c 196f6a8f57, so FS's
``SGLANG_WEG2_FRONT_SPAN_49`` switch and its off-path tests are not ported): the moment a D leg-2 stream delivers its first CONTENT event, D
has prefilled the prompt; the front credits the text as held for that epoch
(``SpanLRU.record_inflight``). The leg's finish replaces it with the measured
entry. Model-neutral: it reads only the wire (first content event) and D's
radix behaviour, nothing about a template or a layer type.

DANGER DIRECTION = over-crediting. Bounded exactly like #49's held credit: only
in the epoch the content arrived in (D flushes its radix on sleep), and D's own
X gate (W31 before the first byte) re-queues any request D does not hold.
"""

import asyncio
import hashlib
import json

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from sglang.srt.weg2 import front as F
from sglang.srt.weg2.front import (
    CHARS_PER_TOKEN,
    Front,
    SpanLRU,
    price_remainder,
    stream_has_content,
)

X = 4096


def _on(monkeypatch, inflight=True):
    # the #49 rest builds on #49 (SGLANG_WEG2_ENABLE_AGENT_SPAN, on for the
    # qwen27b profile since the operator decision of 26.09.)
    monkeypatch.setenv("SGLANG_WEG2_ENABLE_AGENT_SPAN", "1")
    if inflight:
        monkeypatch.setenv("SGLANG_WEG2_FRONT_SPAN_INFLIGHT", "1")
    else:
        monkeypatch.delenv("SGLANG_WEG2_FRONT_SPAN_INFLIGHT", raising=False)


def _off(monkeypatch):
    monkeypatch.delenv("SGLANG_WEG2_FRONT_SPAN_INFLIGHT", raising=False)


# ---------------------------------------------------------------- switch OFF


def test_switch_default_off(monkeypatch):
    _off(monkeypatch)
    assert F.front_span_inflight() is False
    monkeypatch.setenv("SGLANG_WEG2_FRONT_SPAN_INFLIGHT", "1")
    assert F.front_span_inflight() is True
    # the #49 pricing it builds on is unswitched in the unified tree
    assert not hasattr(F, "front_span_49")


# ------------------------------------------------------------- the twin gap


def _twin_texts():
    old = "tools:" + "T" * 13302 + "\nsystem:S\n" + "u" * 50000   # D served earlier
    first = old + "t" * 13000    # the turn: ~4.3k est tokens of new tail
    twin = first + "s" * 585     # Claude Code's side request (+~155 tokens)
    return old, first, twin


def test_twin_behind_a_running_turn_prices_its_own_tail_only(monkeypatch):
    _on(monkeypatch)
    old, first, twin = _twin_texts()
    spans = SpanLRU()
    spans.record_presence(old, 21000, prompt_tokens=21002, held_epoch=10)
    # before: the running turn is unknown -> the twin prices first's tail too
    rem0, _, _ = price_remainder(twin, spans, epoch=10)
    assert rem0 > X, rem0
    # D delivered the running turn's first content in epoch 10
    spans.record_inflight(first, int(len(first) / CHARS_PER_TOKEN) + 1, held_epoch=10)
    rem, _, known = price_remainder(twin, spans, epoch=10)
    assert known and rem <= 585 / CHARS_PER_TOKEN + 1, rem
    assert spans.last_src == "d_served_epoch"


def test_inflight_credit_dies_with_its_epoch(monkeypatch):
    """Mutant guard: the credit across a flip would price a text D flushed."""
    _on(monkeypatch)
    _, first, twin = _twin_texts()
    spans = SpanLRU()
    spans.record_inflight(first, 30000, held_epoch=10)
    rem_next, est, _ = price_remainder(twin, spans, epoch=11)
    rem_none, _, _ = price_remainder(twin, spans, epoch=None)
    assert rem_next == est and rem_none == est


def test_inflight_keeps_a_measured_share_and_the_finish_replaces_it(monkeypatch):
    _on(monkeypatch)
    _, first, twin = _twin_texts()
    spans = SpanLRU()
    spans.record_presence(first, 25000, prompt_tokens=26000)  # an earlier measurement
    spans.record_inflight(first, 1, held_epoch=12)
    key = hashlib.sha1(first.encode()).hexdigest()
    assert spans.entries[key][1] == 25000 and spans.entries[key][2] == 26000
    # after the epoch the measured share still counts (a dead leg erases nothing)
    rem, est, _ = price_remainder(twin, spans, epoch=13)
    assert rem < est
    spans.record_presence(first, 25900, prompt_tokens=26000, held_epoch=12)
    assert spans.entries[key] == (first, 25900, 26000, 12)


def test_stream_has_content_on_both_wires():
    env = (b"event: message_start\ndata: {\"type\":\"message_start\"}\n\n"
           b"event: ping\ndata: {\"type\":\"ping\"}\n\n")
    assert not stream_has_content(env, "/v1/messages")
    assert stream_has_content(env + b"event: content_block_start\ndata: {}\n\n", "/v1/messages")
    assert stream_has_content(b"data: {\"choices\":[]}\n\n", "/v1/chat/completions")
    assert not stream_has_content(None, "/v1/messages")
    assert not stream_has_content(b"", "/v1/chat/completions")


# ------------------------------------------- the seam, through a real leg 2

FIRST = "tools:" + "T" * 900 + "\nsystem:S\nuser:" + "u" * 3000
TWIN = FIRST + "s" * 585
PT, CT = 1300, 1298


def _asse(event, data) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def _fake_d(state: dict) -> web.Application:
    async def messages(request: web.Request) -> web.StreamResponse:
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
        await resp.write(_asse("message_delta", {"type": "message_delta",
                                                  "delta": {"stop_reason": "end_turn"},
                                                  "usage": {"input_tokens": PT - CT,
                                                            "cache_read_input_tokens": CT,
                                                            "output_tokens": 1}}))
        await resp.write(_asse("message_stop", {"type": "message_stop"}))
        await resp.write_eof()
        return resp

    async def info(request: web.Request) -> web.Response:
        return web.json_response({})

    app = web.Application()
    app.router.add_post("/v1/messages", messages)
    app.router.add_get("/get_server_info", info)
    return app


async def _mid_stream_price():
    state = {"decoding": asyncio.Event(), "release": asyncio.Event()}
    d = TestServer(_fake_d(state))
    await d.start_server()
    front = Front("http://p", str(d.make_url("")).rstrip("/"), "D", "t", "", 0, 0, {}, 45.0,
                  carrier_max_tokens=27466, tp_prefill_max_tokens=X)
    front.groups["D"].url = str(d.make_url("")).rstrip("/")
    front.session = aiohttp.ClientSession()
    front.epoch = 7

    async def handler(request: web.Request) -> web.StreamResponse:
        payload = await request.json()
        return await front.leg2(request, "r1", payload, FIRST, True, None)

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
            mid = price_remainder(TWIN, front.spans, epoch=front.epoch)
            state["release"].set()
            resp = await post
            await resp.read()
        end = price_remainder(TWIN, front.spans, epoch=front.epoch)
    finally:
        await runner.cleanup()
        await front.session.close()
        await d.close()
    return front, mid, end


def test_leg2_credits_the_text_while_d_still_decodes(monkeypatch, caplog):
    _on(monkeypatch)
    import logging

    with caplog.at_level(logging.INFO):
        front, mid, end = asyncio.run(_mid_stream_price())
    # the instrument (routing-neutral, always on): one first-content line
    lines = [r.getMessage() for r in caplog.records if "LEG2-FIRST-CONTENT" in r.getMessage()]
    assert len(lines) == 1 and "rid=r1 epoch=7 via=d_direct" in lines[0], lines
    tail = int(585 / CHARS_PER_TOKEN) + 1
    assert mid[2] is True and mid[0] <= tail, mid
    assert front.counters["span_inflight_credited"] == 1
    # the finish books the measured entry for the same text
    key = hashlib.sha1(FIRST.encode()).hexdigest()
    assert front.spans.entries[key][1:] == (CT, PT, 7)
    assert end[0] <= tail


def test_leg2_without_the_switch_learns_only_at_the_finish(monkeypatch, caplog):
    _off(monkeypatch)
    import logging

    with caplog.at_level(logging.INFO):
        front, mid, end = asyncio.run(_mid_stream_price())
    assert mid[2] is False and mid[0] == mid[1], mid  # priced whole mid-stream
    # the instrument line does not depend on the switch
    assert any("LEG2-FIRST-CONTENT rid=r1" in r.getMessage() for r in caplog.records)
    assert front.counters["span_inflight_credited"] == 0
