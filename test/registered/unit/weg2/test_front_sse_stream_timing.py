"""SSE streamed through the weg2 front must arrive SPREAD IN TIME, never collected.

Befund 25.09. ~19:44Z (27B-INT8-Container, RC9 e914e89fde): a streamed
``/v1/messages`` looked "all at once at the end" -- 156 ``content_block_delta``
between 22.27 s and 22.39 s at the front, 266 between 35.08 s and 35.28 s via
the router. These tests pin the product side of that claim on the real wire
(real TCP sockets, real aiohttp on both hops, the real ``Front.leg2`` and the
real ``AnthropicServing._generate_anthropic_stream``):

* a mock D that sends SSE events with pauses; the front forwards them; a client
  timestamps each event on arrival. Every event must reach the client within a
  small bound of the moment D sent it, and the arrival spread must match the
  send spread.
* the three shapes that matter: Anthropic with D awake, Anthropic with D
  DORMANT at admission (``message_start`` plus keep-alive pings for the whole
  flip, content only after the wake -- the shape both measured requests had,
  front log rids weg2-63-48 / weg2-64-51), and the OpenAI wire.
* the instrument is shown able to go red: the same client against a proxy
  that buffers (``read()`` then write) fails the same assertions.

The instrument that produced the 25.09. measurement is NOT part of the
product: ``curl -sN ... | awk '/content_block_delta/{"date"|getline ...}'`` on
this rig is mawk 1.3.4, which block-buffers a pipe on input unless run with
``-W interactive``; on a local mock that sends 20 events 0.2 s apart it prints
``first=4.03s last=4.05s`` (collapsed), with ``-W interactive`` ``first=0.00s
last=3.82s``. It also counts every delta twice (the ``event:`` line and the
``data:`` line both contain the name).
"""

import asyncio
import collections
import json
import time
from types import SimpleNamespace

import pytest
from aiohttp import ClientSession, ClientTimeout, web

from sglang.srt.weg2 import front as F

GAP_S = 0.25  # D's pause between two events
N_DELTAS = 8
# How late an event may reach the client after D sent it. The three hops are
# local sockets in one event loop; a forwarded chunk arrives in milliseconds.
# A buffering hop holds the FIRST delta for (N_DELTAS - 1) * GAP_S = 1.75 s.
MAX_LAG_S = 0.5


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def _anthropic_events(dormant_pings: int):
    """(bytes, pause_after) pairs of an Anthropic stream as D emits it."""
    ev = [(_sse("message_start", {"type": "message_start", "message": {
        "id": "msg_t", "type": "message", "role": "assistant", "model": "m",
        "content": [], "usage": {"input_tokens": 27, "output_tokens": 0}}}), GAP_S)]
    # D dormant at admission: the adapter's keep-alive pings while the
    # scheduler waits for the wake (PING_INTERVAL_SECONDS on the metal).
    ev += [(_sse("ping", {"type": "ping"}), GAP_S) for _ in range(dormant_pings)]
    ev.append((_sse("content_block_start", {"type": "content_block_start", "index": 0,
                                            "content_block": {"type": "thinking", "thinking": ""}}), 0.0))
    for i in range(N_DELTAS):
        ev.append((_sse("content_block_delta", {"type": "content_block_delta", "index": 0,
                                                "delta": {"type": "thinking_delta",
                                                          "thinking": f"tok{i} "}}), GAP_S))
    ev.append((_sse("content_block_stop", {"type": "content_block_stop", "index": 0}), 0.0))
    ev.append((_sse("message_delta", {"type": "message_delta",
                                      "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                                      "usage": {"input_tokens": 27, "output_tokens": 40}}), 0.0))
    ev.append((_sse("message_stop", {"type": "message_stop"}), 0.0))
    return ev


def _openai_events():
    def chunk(delta, usage=None, finish=None):
        d = {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
             "choices": [] if usage else [{"index": 0, "delta": delta, "finish_reason": finish}]}
        if usage:
            d["usage"] = usage
        return f"data: {json.dumps(d)}\n\n".encode()

    ev = [(chunk({"role": "assistant", "content": ""}), 0.0)]
    ev += [(chunk({"content": f"tok{i} "}), GAP_S) for i in range(N_DELTAS)]
    ev.append((chunk({}, finish="stop"), 0.0))
    ev.append((chunk({}, usage={"prompt_tokens": 27, "completion_tokens": 40, "total_tokens": 67}), 0.0))
    ev.append((b"data: [DONE]\n\n", 0.0))
    return ev


def _is_delta(line: bytes) -> bool:
    # ONE line per delta: the `event:` line on Anthropic, the `data:` line
    # carrying content on OpenAI (never both -- the 25.09. awk counted both).
    return line.startswith(b"event: content_block_delta") or (
        line.startswith(b"data: {") and b'"content": "tok' in line)


def _stub_front(d_url: str, session: ClientSession):
    """A Front with exactly the state leg2's streamed branch reads."""
    fr = object.__new__(F.Front)
    fr.groups = {"D": F.Group("D", d_url)}
    fr._d_admissions = 0
    fr.counters = collections.Counter()
    fr.session = session
    fr.epoch = 1
    fr.spans = SimpleNamespace(record_presence=lambda *a, **k: None)
    fr._note_exact = lambda *a, **k: None
    fr._leg2_verdict = lambda *a, **k: "serve"
    fr.note_x_sample = lambda *a, **k: None
    fr._note_x_grant_realized = lambda *a, **k: None

    async def _draft_terms(*a, **k):
        return {"draft_pages": 0, "draft_miss": 0, "accept_len": 0.0, "accept_src": "none",
                "prefill_s": None, "prefill_src": "none"}

    fr._draft_terms = _draft_terms
    return fr


async def _serve(app: web.Application):
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


async def _through(path: str, events, buffering_proxy: bool = False):
    """Send `events` from a mock D through the front; return (sent, arrived)
    monotonic stamps of each delta, in order."""
    sent = []

    async def mock_d(request: web.Request) -> web.StreamResponse:
        await request.read()
        resp = web.StreamResponse(status=200)
        resp.content_type = "text/event-stream"
        await resp.prepare(request)
        for raw, pause in events:
            if _is_delta(raw.split(b"\n", 1)[0]) or _is_delta(raw):
                sent.append(time.monotonic())
            await resp.write(raw)
            if pause:
                await asyncio.sleep(pause)
        await resp.write_eof()
        return resp

    d_app = web.Application()
    d_app.router.add_post(path, mock_d)
    d_runner, d_url = await _serve(d_app)

    session = ClientSession(timeout=ClientTimeout(total=60))
    fr = _stub_front(d_url, session)

    async def front_handler(request: web.Request) -> web.StreamResponse:
        payload = await request.json()
        if buffering_proxy:
            # The mutant the test must catch: read D's whole answer, then write.
            async with session.post(f"{d_url}{request.path}", json=payload) as r:
                body = await r.read()
            resp = web.StreamResponse(status=200)
            resp.content_type = "text/event-stream"
            await resp.prepare(request)
            await resp.write(body)
            await resp.write_eof()
            return resp
        return await fr.leg2(request, "weg2-t-1", payload, "hello", True, None)

    f_app = web.Application()
    f_app.router.add_post(path, front_handler)
    f_runner, f_url = await _serve(f_app)

    arrived = []
    try:
        async with ClientSession(timeout=ClientTimeout(total=60)) as client:
            body = {"model": "m", "max_tokens": 64, "stream": True,
                    "messages": [{"role": "user", "content": "hello"}]}
            async with client.post(f"{f_url}{path}", json=body) as resp:
                assert resp.status == 200
                while True:
                    line = await resp.content.readline()
                    if not line:
                        break
                    if _is_delta(line):
                        arrived.append(time.monotonic())
    finally:
        await session.close()
        await f_runner.cleanup()
        await d_runner.cleanup()
    return sent, arrived, fr


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _assert_streamed(sent, arrived):
    assert len(sent) == N_DELTAS and len(arrived) == N_DELTAS, (len(sent), len(arrived))
    lags = [a - s for s, a in zip(sent, arrived)]
    assert max(lags) < MAX_LAG_S, (
        f"a delta reached the client {max(lags):.2f}s after D sent it -- a hop is buffering "
        f"(lags={['%.3f' % x for x in lags]})")
    spread_sent = sent[-1] - sent[0]
    spread_arrived = arrived[-1] - arrived[0]
    assert spread_arrived > 0.6 * spread_sent, (
        f"arrival spread {spread_arrived:.2f}s against send spread {spread_sent:.2f}s -- collected, not streamed")


@pytest.mark.parametrize("dormant_pings", [0, 3], ids=["d_awake", "d_dormant_at_admission"])
def test_anthropic_stream_is_forwarded_as_it_is_decoded(dormant_pings):
    sent, arrived, fr = _run(_through("/v1/messages", _anthropic_events(dormant_pings)))
    _assert_streamed(sent, arrived)
    assert fr.groups["D"].served == 1


def test_openai_stream_is_forwarded_as_it_is_decoded():
    sent, arrived, _ = _run(_through("/v1/chat/completions", _openai_events()))
    _assert_streamed(sent, arrived)


def test_the_instrument_goes_red_on_a_buffering_proxy():
    """Red-capability: the same client and D, a proxy that reads then writes."""
    sent, arrived, _ = _run(_through("/v1/messages", _anthropic_events(3), buffering_proxy=True))
    with pytest.raises(AssertionError, match="buffering|collected"):
        _assert_streamed(sent, arrived)


def test_lookahead_does_not_hold_content_behind_dormant_pings():
    """The Anthropic lookahead reads at most up to the first content event;
    with D dormant it sees message_start + pings, never a delta."""
    sent, arrived, _ = _run(_through("/v1/messages", _anthropic_events(F.ANTHROPIC_LOOKAHEAD_MAX_CHUNKS + 4)))
    _assert_streamed(sent, arrived)


# ---- D side: the Anthropic adapter yields per backend chunk ---------------

def test_anthropic_adapter_yields_each_backend_chunk_when_it_arrives(monkeypatch):
    from sglang.srt.entrypoints.anthropic import serving as S
    from sglang.srt.entrypoints.anthropic.protocol import AnthropicMessagesRequest

    monkeypatch.setattr(S, "PING_INTERVAL_SECONDS", 0.1)

    def chunk(delta, usage=None, finish=None):
        d = {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
             "choices": [] if usage else [{"index": 0, "delta": delta, "finish_reason": finish}]}
        if usage:
            d["usage"] = usage
        return f"data: {json.dumps(d)}\n\n"

    sent = []

    class _Chat:
        tokenizer_manager = SimpleNamespace(tokenizer=SimpleNamespace(chat_template=None))

        def _generate_chat_stream(self, *a):
            async def gen():
                await asyncio.sleep(0.35)  # D dormant: pings must flow meanwhile
                yield chunk({"role": "assistant", "content": ""})
                for i in range(N_DELTAS):
                    sent.append(time.monotonic())
                    yield chunk({"reasoning_content": f"tok{i} "})
                    await asyncio.sleep(GAP_S)
                yield chunk({}, finish="stop")
                yield chunk({}, usage={"prompt_tokens": 3, "completion_tokens": 8, "total_tokens": 11})
                yield "data: [DONE]\n\n"
            return gen()

    req = AnthropicMessagesRequest.model_validate(
        {"model": "m", "max_tokens": 16, "stream": True,
         "messages": [{"role": "user", "content": "hi"}]})
    serving = S.AnthropicServing(_Chat())

    async def collect():
        arrived, pings = [], 0
        async for frame in serving._generate_anthropic_stream(object(), object(), req, object()):
            if frame.startswith("event: ping"):
                pings += 1
            if frame.startswith("event: content_block_delta"):
                arrived.append(time.monotonic())
        return arrived, pings

    arrived, pings = _run(collect())
    assert pings >= 1, "no keep-alive while the backend was silent"
    _assert_streamed(sent, arrived)
