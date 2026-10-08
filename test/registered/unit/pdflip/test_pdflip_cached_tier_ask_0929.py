"""Feld 2, Operator-Entscheid 29.09.: the front ASKS its internal P/D hops for the
cached-token tier split (``return_cached_tokens_details``) and counts it into
``served_tokens.<row>.cached_tier``; the client gets D's answer BYTE-IDENTICAL to
what D sends without the ask, unless the client asked itself.

The fake D answers with D's OWN models and serializers -- ChatCompletionResponse /
CompletionResponse through FastAPI's ``jsonable_encoder`` + ``JSONResponse`` (what
a pydantic return becomes), stream chunks through ``model_dump_json()`` exactly as
serving_chat / serving_completions yield them, and the Anthropic adapter's own
``_sglext_of`` + ``exclude_none`` + ``_wrap_sse_event``. The ask decides only what
D decides: the ``sglext.cached_tokens_details`` member (serving_chat 1297/1414).

The REAL ``Front.leg2`` (stream and non-stream) behind a real aiohttp front app.
"""
from __future__ import annotations

import asyncio
import json
import time

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from fastapi.encoders import jsonable_encoder
from starlette.responses import JSONResponse

from flliper.srt.entrypoints.anthropic import serving as anth_serving
from flliper.srt.entrypoints.anthropic.protocol import (
    AnthropicMessageEndDelta,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicUsage,
    MessageDeltaEvent,
    MessageStopEvent,
    TextBlock,
)
from flliper.srt.entrypoints.openai.protocol import (
    CachedTokensDetails,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    DeltaMessage,
    PromptTokensDetails,
    SglExt,
    UsageInfo,
)
from flliper.srt.pdflip import front as front_mod
from flliper.srt.pdflip.front import Front, Pending
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="stage-a-pdflip-unit")

PT, CT = 1000, 800
DETAILS = {"device": 600, "host": 150, "storage": 50, "storage_backend": "file"}
TIER = {"device": 600, "host": 150, "storage": 50}
T0 = 1790680000
PATHS = ("/v1/chat/completions", "/v1/completions", "/v1/messages")


def _usage():
    return UsageInfo(prompt_tokens=PT, completion_tokens=3, total_tokens=PT + 3,
                     prompt_tokens_details=PromptTokensDetails(cached_tokens=CT))


def _sglext(asked: bool, depth, stream: bool):
    """D's rule (serving_chat 1297/1309 stream, 1414 non-stream)."""
    details = CachedTokensDetails(**DETAILS) if asked else None
    if stream:
        return SglExt(cached_tokens_details=details, pdflip_resumable_depth=depth) \
            if (details is not None or depth is not None) else None
    return SglExt(cached_tokens_details=details, pdflip_resumable_depth=depth) \
        if (details is not None or depth is not None) else None


def _openai_body(path: str, asked: bool, depth) -> bytes:
    ext = _sglext(asked, depth, False)
    if path == "/v1/completions":
        resp = CompletionResponse(id="cmpl-x", created=T0, model="m", usage=_usage(), sglext=ext,
                                  choices=[CompletionResponseChoice(index=0, text="ok", finish_reason="stop")])
    else:
        resp = ChatCompletionResponse(
            id="chat-x", created=T0, model="m", usage=_usage(), sglext=ext,
            choices=[ChatCompletionResponseChoice(index=0, message=ChatMessage(role="assistant", content="ok"),
                                                  finish_reason="stop")])
    return JSONResponse(content=jsonable_encoder(resp)).body


def _openai_stream(path: str, asked: bool, depth) -> bytes:
    chat = path == "/v1/chat/completions"
    cls = ChatCompletionStreamResponse if chat else CompletionStreamResponse

    def choice(text, fin):
        if chat:
            return ChatCompletionResponseStreamChoice(index=0, delta=DeltaMessage(content=text), finish_reason=fin)
        return CompletionResponseStreamChoice(index=0, text=text or "", finish_reason=fin)

    out = [f"data: {cls(id='s-x', created=T0, model='m', choices=[choice('ok', None)]).model_dump_json()}\n\n",
           f"data: {cls(id='s-x', created=T0, model='m', choices=[choice('', 'stop')]).model_dump_json()}\n\n"]
    ext = _sglext(asked, depth, True)
    if ext is not None:
        out.append(f"data: {cls(id='s-x', created=T0, model='m', choices=[], sglext=ext).model_dump_json()}\n\n")
    out.append(f"data: {cls(id='s-x', created=T0, model='m', choices=[], usage=_usage()).model_dump_json()}\n\n")
    out.append("data: [DONE]\n\n")
    return "".join(out).encode()


def _messages(asked: bool, depth, stream: bool) -> bytes:
    """The Anthropic adapter's own conversion of the OpenAI answer above."""
    openai_resp = ChatCompletionResponse(
        id="chat-x", created=T0, model="m", usage=_usage(), sglext=_sglext(asked, depth, False),
        choices=[ChatCompletionResponseChoice(index=0, message=ChatMessage(role="assistant", content="ok"),
                                              finish_reason="stop")])
    ext = anth_serving._sglext_of(openai_resp)
    usage = AnthropicUsage(input_tokens=PT - CT, output_tokens=3, cache_read_input_tokens=CT)
    if not stream:
        resp = AnthropicMessagesResponse(id="msg_x", content=[TextBlock(text="ok")], model="m",
                                         stop_reason="end_turn", usage=usage, sglext=ext)
        payload = resp.model_dump(exclude_none=True)
        payload.setdefault("stop_sequence", None)
        return JSONResponse(content=payload).body
    wrap = anth_serving._wrap_sse_event
    ev = [wrap(json.dumps({"type": "content_block_delta", "index": 0,
                           "delta": {"type": "text_delta", "text": "ok"}}), "content_block_delta"),
          wrap(MessageDeltaEvent(delta=AnthropicMessageEndDelta(stop_reason="end_turn"), usage=usage,
                                 sglext=ext).model_dump_json(exclude_none=True), "message_delta"),
          wrap(MessageStopEvent().model_dump_json(exclude_none=True), "message_stop")]
    return "".join(ev).encode()


def d_bytes(path: str, stream: bool, asked: bool, depth) -> bytes:
    if path == "/v1/messages":
        return _messages(asked, depth, stream)
    return _openai_stream(path, asked, depth) if stream else _openai_body(path, asked, depth)


def _no_usage_details(got: bytes) -> bytes:
    """The client bytes without the USAGE-DETAILS members (02.10.: an additive
    ``usage.total_tokens_details`` on the final usage, and on the OpenAI wire
    the tier split D reported as ``prompt_tokens_details.cached_device`` /
    ``cached_l2`` / ``cached_l3``); everything else must be D's own bytes."""
    from flliper.srt.pdflip import usage_true as UT

    for key in (b',"total_tokens_details":', b',"cached_device":', b',"cached_l2":',
                b',"cached_l3":'):
        while True:
            i = got.find(key)
            if i < 0:
                break
            got = got[:i] + got[UT._value_end(got, i + len(key)):]
    return got


def test_usage_details_ride_on_the_final_usage_only():
    """The members stripped above are there -- and only on the final usage."""
    for path in PATHS:
        got, _st, _ = asyncio.run(_leg(path, True, False, 7, after_p=True))
        assert got.count(b'"total_tokens_details":') == 1, path
        assert _no_usage_details(got) == d_bytes(path, True, False, 7)


def _fake_d(depth, seen: list, piece: int = 0) -> web.Application:
    async def handle(request: web.Request) -> web.StreamResponse:
        payload = await request.json()
        seen.append(payload)
        asked = bool(payload.get("return_cached_tokens_details"))
        data = d_bytes(request.path, bool(payload.get("stream")), asked, depth)
        if not payload.get("stream"):
            return web.Response(body=data, content_type="application/json")
        resp = web.StreamResponse(status=200)
        resp.content_type = "text/event-stream"
        await resp.prepare(request)
        if piece:                                     # events torn across TCP reads
            for i in range(0, len(data), piece):
                await resp.write(data[i:i + piece])
                await asyncio.sleep(0)
        else:
            for part in data.split(b"\n\n")[:-1]:    # one SSE event per write, as D yields
                await resp.write(part + b"\n\n")
        await resp.write_eof()
        return resp

    async def info(request: web.Request) -> web.Response:
        return web.json_response({"internal_states": [{}]})

    app = web.Application()
    for p in PATHS:
        app.router.add_post(p, handle)
    app.router.add_get("/get_server_info", info)
    return app


async def _leg(path: str, stream: bool, client_asks: bool, depth, after_p: bool, piece: int = 0):
    seen: list = []
    d = TestServer(_fake_d(depth, seen, piece))
    await d.start_server()
    front = Front("http://p", str(d.make_url("")).rstrip("/"), "D", "t", "", 0, 0, {}, 45.0,
                  carrier_max_tokens=262144, tp_prefill_max_tokens=4096)
    front.session = aiohttp.ClientSession()
    pend = Pending(rid="pdflip-0-1", path=path, payload={}, text="q", t_arrive=time.time(), fut=None)
    pend.leg1_ran = after_p

    async def handler(request: web.Request) -> web.StreamResponse:
        payload = await request.json()
        payload["rid"] = "pdflip-0-1"
        return await front.leg2(request, "pdflip-0-1", payload, "q", bool(payload.get("stream")), pend)

    app = web.Application()
    app.router.add_post(path, handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    host, port = runner.addresses[0][:2]
    body = {"model": "m", "max_tokens": 8, "stream": stream,
            "messages": [{"role": "user", "content": "q"}]}
    if client_asks:
        body["return_cached_tokens_details"] = True
    try:
        async with aiohttp.ClientSession() as cs:
            async with cs.post(f"http://{host}:{port}{path}", json=body) as resp:
                got = await resp.read()
    finally:
        await runner.cleanup()
        await front.session.close()
        await d.close()
    return got, front._ipc_front_fields()["served_tokens"], seen


CASES = [(p, s, depth) for p in PATHS for s in (False, True) for depth in (None, 7)]


@pytest.mark.parametrize("path,stream,depth", CASES)
def test_red_first_client_bytes_identical_and_cached_tier_counted(path, stream, depth):
    got, st, seen = asyncio.run(_leg(path, stream, False, depth, after_p=True))
    assert seen[0].get("return_cached_tokens_details") is True           # the internal hop asked
    assert _no_usage_details(got) == d_bytes(path, stream, False, depth)                    # the client: as if never asked
    assert b"cached_tokens_details" not in got
    assert st["D"]["cached_tier"] == TIER
    assert st["D_after_P"]["cached_tier"] == TIER


@pytest.mark.parametrize("path", PATHS)
def test_events_torn_across_reads_are_stripped_whole(path):
    """D's sglext / message_delta event split over several reads: the front
    strips it only once it is complete, the client bytes stay D's unasked ones."""
    got, st, _ = asyncio.run(_leg(path, True, False, 7, after_p=False, piece=13))
    assert _no_usage_details(got) == d_bytes(path, True, False, 7)
    assert st["D"]["cached_tier"] == TIER


@pytest.mark.parametrize("path,stream", [(p, s) for p in PATHS for s in (False, True)])
def test_a_client_that_asked_keeps_the_detail(path, stream):
    got, st, _ = asyncio.run(_leg(path, stream, True, 7, after_p=False))
    assert _no_usage_details(got) == d_bytes(path, stream, True, 7)
    assert b"cached_tokens_details" in got
    assert st["D"]["cached_tier"] == TIER


def test_generate_is_left_alone():
    assert front_mod.with_cached_tier_ask({"text": "q"}, "/generate") == {"text": "q"}


def test_the_client_body_is_never_mutated():
    body = {"model": "m"}
    out = front_mod.with_cached_tier_ask(body, "/v1/chat/completions")
    assert body == {"model": "m"} and out["return_cached_tokens_details"] is True


def test_strip_touches_nothing_but_the_member():
    """A model answer that TALKS about the key (escaped in a JSON string) is untouched."""
    raw = JSONResponse(content={"choices": [{"message": {"content": '"cached_tokens_details":{"device":1}'}}],
                                "sglext": None}).body
    assert front_mod.strip_cached_tier(raw, "/v1/chat/completions", False) == raw


@pytest.mark.parametrize("asked", [True, None])
def test_anthropic_adapter_carries_the_ask_into_the_chat_request(asked):
    """The Messages wire had no way to ask: the field was undeclared (dropped by
    extra="ignore") and the adapter's sglext knew only the resumable depth."""
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location(
        "_anth_serving_tests", pathlib.Path(__file__).parents[1] / "entrypoints" / "anthropic" / "test_serving.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    data = {"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "q"}]}
    if asked:
        data["return_cached_tokens_details"] = True
    req = AnthropicMessagesRequest.model_validate(data)
    chat = anth_serving.AnthropicServing(mod._FakeOpenAIServingChat())._convert_to_chat_completion_request(req)
    assert chat.return_cached_tokens_details is bool(asked)
    none = ChatCompletionResponse(id="c", created=T0, model="m", usage=_usage(), choices=[])
    assert anth_serving._sglext_of(none) is None
