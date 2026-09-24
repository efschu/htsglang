"""H61 (boot fnFL2x165, 2026-09-24): a client that hangs up right after the
finish chunk of a streamed leg 2 was SERVED, not failed.

MEASURED. rid weg2-2-6 had all 1024 tokens at the client (DECODE-PROBE code
19:38:31Z). The client closed on the finish chunk; the front's NEXT write, D's
usage chunk, raised ``ClientConnectionResetError: Cannot write to closing
transport`` and leg 2 was counted as a failure: no WEG2-SERVED, no presence
witness. The warm repeat of the same 12.7k prompt then priced
``presence_span=1``, routed LONG and paid a P prefill plus a flip pair --
TTFT 12.9 s where boot x163 served the identical probe from D's cache in 1.6 s.

Hermetic: a fake D and the front run as real in-process aiohttp servers on
loopback, the client really closes its socket. No CUDA, no model.
"""

import asyncio
import json

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer

from sglang.srt.weg2.front import Front, stream_finish_seen

PROMPT_TEXT = "the prompt text of a 12.7k code probe"
PT, CT = 12672, 12668


def _sse(obj) -> bytes:
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


def _fake_d(pause_between: float, pause_after_finish: float, seen: dict) -> web.Application:
    async def chat(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(status=200)
        resp.content_type = "text/event-stream"
        await resp.prepare(request)
        try:
            for i in range(3):
                await resp.write(_sse({"choices": [{"index": 0, "delta": {"content": f"t{i}"},
                                                    "finish_reason": None}]}))
                await asyncio.sleep(pause_between)
            await resp.write(_sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]}))
            await asyncio.sleep(pause_after_finish)
            await resp.write(_sse({"choices": [], "usage": {
                "prompt_tokens": PT, "completion_tokens": 3,
                "prompt_tokens_details": {"cached_tokens": CT}}}))
            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
            seen["d_completed"] = True
        except ConnectionResetError:
            # The front closed D's connection: D would abort the request.
            seen["d_aborted"] = True
        return resp

    async def info(request: web.Request) -> web.Response:
        return web.json_response({})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_get("/get_server_info", info)
    return app


def _front_app(front: Front) -> web.Application:
    async def handler(request: web.Request) -> web.StreamResponse:
        payload = await request.json()
        try:
            return await front.leg2(request, "r1", payload, PROMPT_TEXT, True, None)
        finally:
            front.counters["test_handler_done"] += 1

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    return app


async def _scenario(hang_up: str, pause_between: float = 0.0, pause_after_finish: float = 0.3):
    seen: dict = {}
    d = TestServer(_fake_d(pause_between, pause_after_finish, seen))
    await d.start_server()
    front = Front("http://p", str(d.make_url("")).rstrip("/"), "D", "t", "", 0, 0, {}, 45.0,
                  carrier_max_tokens=27466, tp_prefill_max_tokens=4096)
    front.groups["D"].url = str(d.make_url("")).rstrip("/")
    front.session = aiohttp.ClientSession()
    # The front runs as in production (``web.run_app``): handler_cancellation
    # stays OFF. aiohttp's TestServer turns it ON, which cancels the handler
    # on the hang-up and hides exactly the write error this file is about.
    runner = web.AppRunner(_front_app(front))
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    host, port = runner.addresses[0][:2]
    got = b""
    try:
        async with aiohttp.ClientSession() as cs:
            resp = await cs.post(f"http://{host}:{port}/v1/chat/completions",
                                 json={"messages": [{"role": "user", "content": PROMPT_TEXT}],
                                       "stream": True})
            async for chunk in resp.content.iter_any():
                got += chunk
                if hang_up == "after_finish" and stream_finish_seen(got):
                    break
                if hang_up == "mid_answer" and b"t0" in got:
                    break
            # "never" reads to the end; the others hang up here.
            resp.close()
        for _ in range(100):
            if front.counters["test_handler_done"]:
                break
            await asyncio.sleep(0.05)
    finally:
        await runner.cleanup()
        await front.session.close()
        await d.close()
    return front, seen, got


def test_hang_up_after_the_finish_chunk_is_served_and_books_presence():
    front, seen, got = asyncio.run(_scenario("after_finish"))
    assert stream_finish_seen(got), "the client must have seen the end of the answer"
    assert b'"usage"' not in got, "the hang-up must come before the usage chunk"
    assert front.counters["test_handler_done"] == 1
    assert front.counters["leg2_failures"] == 0
    assert front.counters["leg2_client_closed_after_finish"] == 1
    assert front.groups["D"].served == 1
    # The presence witness is D's own cached share, read from the usage chunk
    # the client never saw.
    assert front.spans.span_tokens(PROMPT_TEXT) == (CT, True)
    assert seen.get("d_completed") is True


def test_hang_up_mid_answer_still_aborts_and_books_nothing():
    front, seen, got = asyncio.run(_scenario("mid_answer", pause_between=0.3))
    assert not stream_finish_seen(got)
    assert front.counters["test_handler_done"] == 1
    assert front.counters["leg2_failures"] == 1
    assert front.counters["leg2_client_closed_after_finish"] == 0
    assert front.groups["D"].served == 0
    assert front.spans.span_tokens(PROMPT_TEXT) == (0, False)
    # Closing D's connection is what aborts decoding for a client that left.
    assert seen.get("d_completed") is not True


def test_a_client_that_reads_to_the_end_is_unchanged():
    front, seen, got = asyncio.run(_scenario("never"))
    assert b"[DONE]" in got and b'"usage"' in got
    assert front.counters["leg2_failures"] == 0
    assert front.counters["leg2_client_closed_after_finish"] == 0
    assert front.groups["D"].served == 1
    assert front.spans.span_tokens(PROMPT_TEXT) == (CT, True)


def test_finish_marker_on_both_wires():
    assert not stream_finish_seen(_sse({"choices": [{"delta": {"content": "a"}, "finish_reason": None}]}))
    assert stream_finish_seen(_sse({"choices": [{"delta": {}, "finish_reason": "length"}]}))
    assert stream_finish_seen(b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n')
    assert not stream_finish_seen(b'data: {"type":"message_delta","delta":{"stop_reason":null}}\n\n')
    assert stream_finish_seen(b'event: message_delta\ndata: {"type":"message_delta",'
                              b'"delta":{"stop_reason":"end_turn"}}\n\n')
    assert stream_finish_seen(b'event: message_stop\ndata: {"type":"message_stop"}\n\n')
