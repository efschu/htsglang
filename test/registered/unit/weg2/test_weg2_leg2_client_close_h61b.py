# SPDX-License-Identifier: Apache-2.0
"""H61b (2026-09-26): the presence witness is booked once D's end and usage
are in hand, whether or not the client is still reading.

H61 (88dec66c7d) made a hang-up AFTER the finish chunk "served". Three
windows stayed open, and each one books no ``record_presence`` for a request
D fully served -- the next agent turn is then priced cold, goes LONG over P
and pays a flip pair (TTFT 12.9 s instead of 1.6 s on the 12.7k probe):

(a) The finish marker was scanned AFTER the write. On the Anthropic wire the
    adapter yields ``content_block_stop``, ``message_delta`` (stop_reason +
    usage) and ``message_stop`` back to back after [DONE]; a hang-up after the
    last ``content_block_stop`` makes the write of the chunk that CARRIES the
    end and the usage raise with ``finished`` still False -> leg-2 failure,
    and that chunk never reached the usage reader. Same on the OpenAI wire
    when the finish chunk and the usage chunk arrive in one read.
(b) Only ``ConnectionResetError`` was caught. A write parked on a paused
    transport gets aiohttp's plain ``ConnectionError("Connection lost")``
    (``base_protocol.connection_lost``), even after the end was written.
(c) A hang-up one chunk before D's end marker aborted D although the end was
    one scheduler step away.

Hermetic: a fake D (its /v1/messages stream is the adapter's REAL generator
output, cut into its SSE events) and the front run as in-process aiohttp
servers on loopback; the front runs with handler_cancellation OFF as under
``web.run_app``; the client really closes its socket. No CUDA, no model.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
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

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

PROMPT_TEXT = "the prompt text of a 12.7k agent turn"
PROMPT, CACHED = 12672, 12668
HANG_UP_PAUSE = 0.3  # long enough for the client's FIN to reach the front
# TOLERANT ON PURPOSE: the parent tree has no grace constant, and these tests
# must be red there for the BOOKING, not for a missing attribute.
GRACE_S = getattr(F, "H61B_END_GRACE_S", 0.25)

# The fake D writes with the ORIGINAL class even while a test swaps the
# front's ``web.StreamResponse``.
_ORIG_SR = web.StreamResponse


# ---------------------------------------------------------------- D's wires --


class _FakeChat:
    """The OpenAI chat handler as the adapter sees it: a fixed chunk stream."""

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


def _chat_chunk(choices=None, usage=None) -> str:
    data = {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": choices or []}
    if usage is not None:
        data["usage"] = usage
    return f"data: {json.dumps(data)}\n\n"


def _adapter_frames(n_text: int = 3) -> list:
    """The adapter's REAL /v1/messages stream for one engine stream, cut into
    its SSE events (each ``event: ...\\ndata: ...\\n\\n``)."""
    usage = {"prompt_tokens": PROMPT, "completion_tokens": n_text,
             "total_tokens": PROMPT + n_text,
             "prompt_tokens_details": {"cached_tokens": CACHED}}
    lines = [_chat_chunk([{"index": 0, "delta": {"role": "assistant", "content": "t0"},
                           "finish_reason": None}], usage=usage)]
    for i in range(1, n_text):
        lines.append(_chat_chunk([{"index": 0, "delta": {"content": f"t{i}"},
                                   "finish_reason": None}], usage=usage))
    lines += [_chat_chunk([{"index": 0, "delta": {}, "finish_reason": "stop"}], usage=usage),
              _chat_chunk([], usage=usage),
              "data: [DONE]\n\n"]
    serving = AnthropicServing(_FakeChat(lines))
    req = AnthropicMessagesRequest(model="m", max_tokens=64, stream=True,
                                   messages=[{"role": "user", "content": PROMPT_TEXT}])

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


def _kind(frame: bytes) -> str:
    m = re.search(rb"^event: (\w+)", frame, re.M)
    return m.group(1).decode() if m else "?"


def _split_tail(frames: list):
    """(head up to and including the last content_block_delta, cbs, md, ms)."""
    kinds = [_kind(f) for f in frames]
    assert kinds[-3:] == ["content_block_stop", "message_delta", "message_stop"], kinds
    last_delta = max(i for i, k in enumerate(kinds) if k == "content_block_delta")
    assert last_delta == len(frames) - 4, kinds
    return frames[:last_delta + 1], frames[-3], frames[-2], frames[-1]


def _fake_d(path: str, writes: list, seen: dict) -> web.Application:
    """D writes ``writes`` = [(bytes, pause_after_s), ...] in order."""

    async def handler(request: web.Request) -> web.StreamResponse:
        resp = _ORIG_SR(status=200)
        resp.content_type = "text/event-stream"
        await resp.prepare(request)
        try:
            for data, pause in writes:
                await resp.write(data)
                if pause:
                    await asyncio.sleep(pause)
            await resp.write_eof()
            seen["d_completed"] = True
        except ConnectionError:
            # The front closed D's connection: D would abort the request.
            seen["d_aborted"] = True
            seen["t_d_aborted"] = time.monotonic()
        except asyncio.CancelledError:
            # TestServer runs D with handler_cancellation ON: the front
            # closing D's connection cancels this handler instead.
            seen["d_aborted"] = True
            seen["t_d_aborted"] = time.monotonic()
            raise
        return resp

    async def info(request: web.Request) -> web.Response:
        return web.json_response({})

    app = web.Application()
    app.router.add_post(path, handler)
    app.router.add_get("/get_server_info", info)
    return app


# ------------------------------------------------------------- the harness --


def _front_app(front, path: str, seen: dict) -> web.Application:
    async def handler(request: web.Request) -> web.StreamResponse:
        payload = await request.json()
        try:
            return await front.leg2(request, "r1", payload, PROMPT_TEXT, True, None)
        finally:
            seen["t_handler_done"] = time.monotonic()
            front.counters["test_handler_done"] += 1

    app = web.Application()
    app.router.add_post(path, handler)
    return app


async def _scenario(path: str, writes: list, hang_up, front_sr=None):
    """``hang_up(got) -> bool`` closes the client socket once true; None reads
    to the end. ``front_sr`` swaps the class the FRONT streams with."""
    seen: dict = {}
    d = TestServer(_fake_d(path, writes, seen))
    await d.start_server()
    d_url = str(d.make_url("")).rstrip("/")
    front = F.Front("http://p", d_url, "D", "t", "", 0, 0, {}, 45.0,
                    carrier_max_tokens=27466, tp_prefill_max_tokens=4096)
    front.groups["D"].url = d_url
    front.session = aiohttp.ClientSession()
    # As in production (``web.run_app``): handler_cancellation OFF. aiohttp's
    # TestServer turns it ON and would hide the write errors under test.
    runner = web.AppRunner(_front_app(front, path, seen))
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    host, port = runner.addresses[0][:2]
    got = b""
    saved_sr = F.web.StreamResponse
    if front_sr is not None:
        F.web.StreamResponse = front_sr
    try:
        body = {"model": "m", "max_tokens": 64, "stream": True,
                "messages": [{"role": "user", "content": PROMPT_TEXT}]}
        async with aiohttp.ClientSession() as cs:
            resp = await cs.post(f"http://{host}:{port}{path}", json=body)

            async def _read():
                nonlocal got
                async for chunk in resp.content.iter_any():
                    got += chunk
                    if hang_up is not None and hang_up(got):
                        seen["t_hung_up"] = time.monotonic()
                        return

            try:
                await asyncio.wait_for(_read(), 15)
            except (aiohttp.ClientError, ConnectionError):
                pass
            resp.close()
        for _ in range(200):
            if front.counters["test_handler_done"]:
                break
            await asyncio.sleep(0.05)
        # Let D's handler observe the close of its connection, if any.
        for _ in range(60):
            if seen.get("d_completed") or seen.get("d_aborted"):
                break
            await asyncio.sleep(0.05)
    finally:
        F.web.StreamResponse = saved_sr
        await runner.cleanup()
        await front.session.close()
        await d.close()
    return front, seen, got


def _assert_served_and_booked(front, seen):
    assert front.counters["test_handler_done"] == 1
    assert front.counters["leg2_failures"] == 0, "a request D fully served was counted as a leg-2 failure"
    assert front.groups["D"].served == 1
    # The presence witness is D's own cached share, read from the usage the
    # client never saw.
    assert front.spans.span_tokens(PROMPT_TEXT) == (CACHED, True)
    assert seen.get("d_completed") is True


# ------------------------------------------------ (a) Anthropic, cbs | md+ms --


def test_messages_hang_up_after_last_content_block_stop_books_presence():
    """(a) The hang-up lands after the last content_block_stop; D's
    message_delta (end + usage) and message_stop arrive in one read after it.
    Before H61b the write of that chunk raised with ``finished`` False."""
    head, cbs, md, ms = _split_tail(_adapter_frames())
    writes = [(f, 0) for f in head] + [(cbs, HANG_UP_PAUSE), (md + ms, 0)]
    front, seen, got = asyncio.run(_scenario(
        "/v1/messages", writes, lambda g: b"event: content_block_stop" in g))
    assert b"event: content_block_stop" in got and b"message_delta" not in got
    _assert_served_and_booked(front, seen)
    assert front.counters["leg2_client_closed_after_finish"] == 1
    assert front.counters["leg2_client_closed_before_end_caught"] == 0


# ------------------------------------------- H61 + H100 pin, md | ms -------


def test_messages_hang_up_after_message_delta_books_the_delta_cache_count():
    """The window H61 already covered, on the Anthropic wire: the hang-up
    lands between message_delta and message_stop. Pinned because the witness
    must be the CORRECTED cache count of message_delta (H100), not the 0 that
    message_start ships."""
    head, cbs, md, ms = _split_tail(_adapter_frames())
    writes = [(f, 0) for f in head] + [(cbs, 0), (md, HANG_UP_PAUSE), (ms, 0)]
    front, seen, got = asyncio.run(_scenario(
        "/v1/messages", writes, lambda g: b"event: message_delta" in g))
    assert b"message_stop" not in got
    _assert_served_and_booked(front, seen)
    assert front.counters["leg2_client_closed_after_finish"] == 1


# ------------------------------------- (c) Anthropic, last delta | cbs | md --


def test_messages_hang_up_one_chunk_before_the_end_is_caught_within_the_grace(monkeypatch):
    """(c) The hang-up lands after the last text delta; D's content_block_stop
    comes alone (its write fails, it carries no end marker), the end follows
    one scheduler step later. D is read on for the grace and booked.

    The grace is widened here so a starved test process (cpu.idle cgroup
    next to a boot) cannot time it out; the subject is the mechanism, the
    production value is pinned by the mid-answer test below."""
    monkeypatch.setattr(F, "H61B_END_GRACE_S", 3.0, raising=False)
    head, cbs, md, ms = _split_tail(_adapter_frames())
    writes = [(f, 0) for f in head[:-1]] + [(head[-1], HANG_UP_PAUSE), (cbs, 0.02), (md + ms, 0)]
    front, seen, got = asyncio.run(_scenario(
        "/v1/messages", writes, lambda g: b'"t2"' in g))
    assert b"content_block_stop" not in got
    _assert_served_and_booked(front, seen)
    assert front.counters["leg2_client_closed_before_end_caught"] == 1
    assert front.counters["leg2_client_closed_after_finish"] == 0


# ---------------------------------------- mid-answer: the abort must stand --


def test_messages_hang_up_mid_answer_still_aborts_d_after_the_grace():
    """A client that leaves while D is still generating: D is aborted (its
    connection closed) once the grace runs out, nothing is booked."""
    frames = _adapter_frames(n_text=2)
    head, cbs, md, ms = _split_tail(frames)
    delta = head[-1]
    n_more = int((GRACE_S + 6.0) / 0.05)
    writes = ([(f, 0) for f in head[:-1]] + [(delta, 0.05)] * n_more
              + [(cbs, 0), (md + ms, 0)])
    front, seen, got = asyncio.run(_scenario(
        "/v1/messages", writes, lambda g: b"content_block_delta" in g))
    assert b"message_delta" not in got
    assert front.counters["test_handler_done"] == 1
    assert front.counters["leg2_failures"] == 1
    assert front.groups["D"].served == 0
    assert front.spans.span_tokens(PROMPT_TEXT) == (0, False)
    assert seen.get("d_completed") is not True
    assert seen.get("d_aborted") is True
    # Bounded: D stops within the grace (plus slack), not at the end of its answer.
    assert seen["t_handler_done"] - seen["t_hung_up"] < GRACE_S + 3.0


# ------------------------------- (b) drain waiter's plain ConnectionError --


class _DrainLostAtMessageStop(_ORIG_SR):
    """The front's response as a paused transport sees a lost peer: the write
    raises aiohttp's ``ConnectionError("Connection lost")`` -- NOT a
    ConnectionResetError -- here on the message_stop, i.e. AFTER the end."""

    async def write(self, data):
        if b"message_stop" in bytes(data):
            raise ConnectionError("Connection lost")
        return await super().write(data)


def test_messages_drain_connection_lost_after_the_end_books_presence():
    head, cbs, md, ms = _split_tail(_adapter_frames())
    writes = [(f, 0) for f in head] + [(cbs, 0), (md, 0.1), (ms, 0)]
    front, seen, got = asyncio.run(_scenario(
        "/v1/messages", writes, None, front_sr=_DrainLostAtMessageStop))
    assert b"message_delta" in got
    _assert_served_and_booked(front, seen)
    assert front.counters["leg2_client_closed_after_finish"] == 1


# ------------------------------------ (a) OpenAI wire, content | finish+usage --


def _oa(obj) -> bytes:
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


def test_chat_hang_up_before_the_finish_chunk_books_presence():
    """(a) on the OpenAI wire: the finish chunk and the usage chunk arrive in
    one read after the hang-up; that write raised before the scan."""
    writes = [(_oa({"choices": [{"index": 0, "delta": {"content": f"t{i}"},
                                 "finish_reason": None}]}), 0) for i in range(2)]
    writes.append((_oa({"choices": [{"index": 0, "delta": {"content": "t2"},
                                     "finish_reason": None}]}), HANG_UP_PAUSE))
    writes.append((_oa({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
                   + _oa({"choices": [], "usage": {"prompt_tokens": PROMPT, "completion_tokens": 3,
                                                   "prompt_tokens_details": {"cached_tokens": CACHED}}})
                   + b"data: [DONE]\n\n", 0))
    front, seen, got = asyncio.run(_scenario(
        "/v1/chat/completions", writes, lambda g: b'"t2"' in g))
    assert not F.stream_finish_seen(got)
    _assert_served_and_booked(front, seen)
    assert front.counters["leg2_client_closed_after_finish"] == 1
