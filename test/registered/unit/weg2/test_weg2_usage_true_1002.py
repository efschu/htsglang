# SPDX-License-Identifier: Apache-2.0
"""USAGE-TRUE (02.10.): the client sees the prefill that was REALLY computed.

THE REPORT. OpenWebUI shows ``usage.prompt_tokens_details.cached_tokens`` of a
flipped request as ~the whole prompt: the client got D's leg-2 usage verbatim,
and D read P's whole hand-off back from the store. The prefill P computed in
leg 1 was invisible.

THE LAW PINNED HERE.
* client cached = prompt - (sum of every P leg's (prompt - cached) + D's
  (prompt - cached)), clamped to [0, prompt]; prompt/completion/total stay D's;
* OpenAI non-stream body and the stream's usage chunk; Anthropic non-stream
  ``usage`` and the stream's ``message_delta`` (``cache_read_input_tokens`` =
  true cached, ``input_tokens`` = prompt - cached);
* a request without a P leg is relayed byte for byte;
* the internal presence witness keeps D's RAW cached count;
* one ``WEG2-USAGE-TRUE`` line per corrected request.

Hermetic: fake P and fake D (the D Anthropic stream is the adapter's REAL
generator output) and the front's real ``leg1`` / ``leg2`` run in-process on
loopback. No CUDA, no model.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede imports that may pull in sgl_kernel

import aiohttp  # noqa: E402
from aiohttp import web  # noqa: E402
from aiohttp.test_utils import TestServer  # noqa: E402

from sglang.srt.entrypoints.anthropic.protocol import (  # noqa: E402
    AnthropicMessagesRequest,
)
from sglang.srt.entrypoints.anthropic.serving import AnthropicServing  # noqa: E402
from sglang.srt.weg2 import front as F  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

TEXT = "a flipped agent turn"
RID = "weg2-1-1"
PROMPT = 20000
D_CACHED = 19990          # D read P's hand-off back: ~the whole prompt
P_CACHED = 4000           # P's own device hit on leg 1
OUT = 7
# true computed = (20000-4000) + (20000-19990) = 16010 -> cached 3990
TRUE_CACHED = PROMPT - ((PROMPT - P_CACHED) + (PROMPT - D_CACHED))
_ORIG_SR = web.StreamResponse


# ---------------------------------------------------------------- the wires --


def _oa_usage(prompt=PROMPT, cached=D_CACHED, out=OUT):
    return {"prompt_tokens": prompt, "completion_tokens": out, "total_tokens": prompt + out,
            "prompt_tokens_details": {"cached_tokens": cached}}


def _oa_body(prompt=PROMPT, cached=D_CACHED) -> bytes:
    return json.dumps({"id": "c", "object": "chat.completion", "created": 0, "model": "m",
                       "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi ü"},
                                    "finish_reason": "stop"}],
                       "usage": _oa_usage(prompt, cached)}).encode()


def _sse(obj) -> bytes:
    return b"data: " + json.dumps(obj, separators=(",", ":")).encode() + b"\n\n"


def _oa_stream() -> list:
    out = [_sse({"choices": [{"index": 0, "delta": {"content": f"t{i}"}, "finish_reason": None}]})
           for i in range(3)]
    out.append(_sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}))
    out.append(_sse({"choices": [], "usage": _oa_usage()}))
    out.append(b"data: [DONE]\n\n")
    return out


def _anth_body() -> bytes:
    return json.dumps({"id": "msg", "type": "message", "role": "assistant", "model": "m",
                       "content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn",
                       "stop_sequence": None,
                       "usage": {"input_tokens": PROMPT - D_CACHED,
                                 "cache_read_input_tokens": D_CACHED,
                                 "output_tokens": OUT}}).encode()


class _FakeChat:
    def __init__(self, lines):
        self.lines = lines
        self.tokenizer_manager = SimpleNamespace(tokenizer=SimpleNamespace(chat_template=None))

    def _generate_chat_stream(self, adapted_request, processed_request, raw_request):
        async def _gen():
            for line in self.lines:
                yield line
        return _gen()

    def apply_reasoning_enabled(self, *a, **kw):
        return None


def _anth_stream() -> list:
    """The adapter's REAL /v1/messages stream (totals in message_delta)."""
    usage = _oa_usage()
    lines = [f"data: {json.dumps({'id': 'c', 'object': 'chat.completion.chunk', 'created': 0, 'model': 'm', 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': f't{i}'}, 'finish_reason': None}], 'usage': usage})}\n\n"
             for i in range(3)]
    lines.append(f"data: {json.dumps({'id': 'c', 'object': 'chat.completion.chunk', 'created': 0, 'model': 'm', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}], 'usage': usage})}\n\n")
    lines.append(f"data: {json.dumps({'id': 'c', 'object': 'chat.completion.chunk', 'created': 0, 'model': 'm', 'choices': [], 'usage': usage})}\n\n")
    lines.append("data: [DONE]\n\n")
    serving = AnthropicServing(_FakeChat(lines))
    req = AnthropicMessagesRequest(model="m", max_tokens=64, stream=True,
                                   messages=[{"role": "user", "content": TEXT}])

    async def _collect():
        out = []
        async for sse in serving._generate_anthropic_stream(
            adapted_request=object(), processed_request=object(),
            anthropic_request=req, raw_request=object(),
        ):
            out.append(sse)
        return "".join(out).encode()

    loop = asyncio.new_event_loop()
    try:
        blob = loop.run_until_complete(_collect())
    finally:
        loop.close()
    return [part + b"\n\n" for part in blob.split(b"\n\n") if part.strip()]


# ---------------------------------------------------------------- the fakes --


def _fake_group(path: str, answer) -> web.Application:
    """``answer`` = bytes (a JSON body) or a list of SSE chunks (a stream)."""

    async def handler(request: web.Request) -> web.StreamResponse:
        await request.read()
        if isinstance(answer, bytes):
            return web.Response(body=answer, content_type="application/json")
        resp = _ORIG_SR(status=200)
        resp.content_type = "text/event-stream"
        await resp.prepare(request)
        for chunk in answer:
            await resp.write(chunk)
        await resp.write_eof()
        return resp

    async def info(request: web.Request) -> web.Response:
        return web.json_response({})

    app = web.Application()
    app.router.add_post(path, handler)
    app.router.add_get("/get_server_info", info)
    return app


def _p_body(path: str, prompt: int, cached: int) -> bytes:
    if path == "/v1/messages":
        return json.dumps({"id": "m", "type": "message", "role": "assistant", "model": "m",
                           "content": [{"type": "text", "text": "x"}], "stop_reason": "max_tokens",
                           "usage": {"input_tokens": prompt - cached, "cache_read_input_tokens": cached,
                                     "output_tokens": 1}}).encode()
    return _oa_body(prompt, cached)


async def _run(path: str, d_answer, stream: bool, p_legs=((PROMPT, P_CACHED),)):
    """leg 1 on a fake P once per ``p_legs`` entry (the same rid), then leg 2
    on a fake D; returns (front, client bytes)."""
    front = F.Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0,
                    carrier_max_tokens=27466, tp_prefill_max_tokens=4096)
    front.session = aiohttp.ClientSession()
    servers = []
    pending = None
    try:
        for prompt, cached in p_legs:
            p_srv = TestServer(_fake_group(path, _p_body(path, prompt, cached)))
            await p_srv.start_server()
            servers.append(p_srv)
            front.groups["P"].url = str(p_srv.make_url("")).rstrip("/")
            body = {"model": "m", "max_tokens": 64, "stream": stream,
                    "messages": [{"role": "user", "content": TEXT}]}
            pending = F.Pending(RID, path, body, TEXT, time.time(),
                                asyncio.get_event_loop().create_future(), est_prompt=PROMPT)
            await front.leg1(pending)
        d_srv = TestServer(_fake_group(path, d_answer))
        await d_srv.start_server()
        servers.append(d_srv)
        front.groups["D"].url = str(d_srv.make_url("")).rstrip("/")

        async def handler(request: web.Request) -> web.StreamResponse:
            payload = await request.json()
            return await front.leg2(request, RID, payload, TEXT, stream, pending)

        app = web.Application()
        app.router.add_post(path, handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        host, port = runner.addresses[0][:2]
        try:
            async with aiohttp.ClientSession() as cs:
                body = {"model": "m", "max_tokens": 64, "stream": stream,
                        "messages": [{"role": "user", "content": TEXT}]}
                async with cs.post(f"http://{host}:{port}{path}", json=body) as resp:
                    got = await resp.read()
        finally:
            await runner.cleanup()
    finally:
        await front.session.close()
        for s in servers:
            await s.close()
    return front, got


def _raw_presence(front):
    e = front.spans.entries.get(hashlib.sha1(TEXT.encode()).hexdigest())
    return None if e is None else int(e[1])


def _sse_objs(got: bytes):
    out = []
    for line in got.split(b"\n"):
        if line.startswith(b"data:") and line[5:].strip() not in (b"", b"[DONE]"):
            out.append(json.loads(line[5:]))
    return out


def _marker(caplog):
    return [r.getMessage() for r in caplog.records if "WEG2-USAGE-TRUE" in r.getMessage()]


# ---------------------------------------------------------------- the tests --


def test_openai_nonstream_flipped_reports_true_cached(caplog):
    caplog.set_level(logging.INFO, logger="weg2.front")
    front, got = asyncio.run(_run("/v1/chat/completions", _oa_body(), False))
    u = json.loads(got)["usage"]
    assert u["prompt_tokens_details"]["cached_tokens"] == TRUE_CACHED == 3990
    assert (u["prompt_tokens"], u["completion_tokens"], u["total_tokens"]) == (PROMPT, OUT, PROMPT + OUT)
    # nothing else changed: the body is D's with the one number swapped
    assert got == _oa_body().replace(b'"cached_tokens": 19990', b'"cached_tokens": 3990')
    assert _raw_presence(front) == D_CACHED  # internal bookkeeping on D's RAW numbers
    m = _marker(caplog)
    assert len(m) == 1
    assert (f"rid={RID} prompt={PROMPT} p_computed={PROMPT - P_CACHED} d_computed={PROMPT - D_CACHED} "
            f"cached_client={TRUE_CACHED} (was {D_CACHED})") in m[0]


def test_openai_stream_final_usage_chunk_reports_true_cached(caplog):
    caplog.set_level(logging.INFO, logger="weg2.front")
    front, got = asyncio.run(_run("/v1/chat/completions", _oa_stream(), True))
    objs = _sse_objs(got)
    usage = [o["usage"] for o in objs if o.get("usage")]
    assert len(usage) == 1
    assert usage[0]["prompt_tokens_details"]["cached_tokens"] == TRUE_CACHED
    assert usage[0]["prompt_tokens"] == PROMPT and usage[0]["total_tokens"] == PROMPT + OUT
    # every content chunk passes byte for byte
    raw = b"".join(_oa_stream())
    assert got == raw.replace(b'"cached_tokens":19990', b'"cached_tokens":3990')
    assert _raw_presence(front) == D_CACHED
    assert len(_marker(caplog)) == 1


def test_anthropic_nonstream_flipped_reports_true_cached(caplog):
    caplog.set_level(logging.INFO, logger="weg2.front")
    front, got = asyncio.run(_run("/v1/messages", _anth_body(), False))
    u = json.loads(got)["usage"]
    assert u["cache_read_input_tokens"] == TRUE_CACHED
    assert u["input_tokens"] == PROMPT - TRUE_CACHED
    assert u["output_tokens"] == OUT
    assert _raw_presence(front) == D_CACHED
    assert len(_marker(caplog)) == 1


def test_anthropic_stream_message_delta_reports_true_cached(caplog):
    caplog.set_level(logging.INFO, logger="weg2.front")
    frames = _anth_stream()
    front, got = asyncio.run(_run("/v1/messages", frames, True))
    deltas = [o for o in _sse_objs(got) if o.get("type") == "message_delta"]
    assert len(deltas) == 1
    u = deltas[0]["usage"]
    assert u["cache_read_input_tokens"] == TRUE_CACHED
    assert u["input_tokens"] == PROMPT - TRUE_CACHED
    # message_start (input 0, no cache field on this adapter) is untouched
    starts = [o for o in _sse_objs(got) if o.get("type") == "message_start"]
    assert starts and "cache_read_input_tokens" not in starts[0]["message"]["usage"]
    assert _raw_presence(front) == D_CACHED
    assert len(_marker(caplog)) == 1


def test_d_only_passthrough_is_byte_identical(caplog):
    caplog.set_level(logging.INFO, logger="weg2.front")
    _f, got = asyncio.run(_run("/v1/chat/completions", _oa_body(), False, p_legs=()))
    assert got == _oa_body()
    _f, got = asyncio.run(_run("/v1/chat/completions", _oa_stream(), True, p_legs=()))
    assert got == b"".join(_oa_stream())
    frames = _anth_stream()
    _f, got = asyncio.run(_run("/v1/messages", frames, True, p_legs=()))
    assert got == b"".join(frames)
    _f, got = asyncio.run(_run("/v1/messages", _anth_body(), False, p_legs=()))
    assert got == _anth_body()
    assert _marker(caplog) == []


def test_every_p_leg_of_the_rid_is_summed(caplog):
    """A re-route / RESUME-VIA-P runs a second P leg under the same rid: both
    legs' computed prefill count."""
    caplog.set_level(logging.INFO, logger="weg2.front")
    legs = ((PROMPT, P_CACHED), (PROMPT, 19000))
    front, got = asyncio.run(_run("/v1/chat/completions", _oa_body(), False, p_legs=legs))
    p_comp = (PROMPT - P_CACHED) + (PROMPT - 19000)
    want = PROMPT - (p_comp + (PROMPT - D_CACHED))
    assert want == 2990
    assert json.loads(got)["usage"]["prompt_tokens_details"]["cached_tokens"] == want
    m = _marker(caplog)
    assert len(m) == 1 and f"p_computed={p_comp} " in m[0] and "p_legs=2" in m[0]


def test_clamped_at_zero_when_p_recomputed_more_than_the_prompt():
    legs = ((PROMPT, 0), (PROMPT, 0))
    _f, got = asyncio.run(_run("/v1/chat/completions", _oa_body(), False, p_legs=legs))
    assert json.loads(got)["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
