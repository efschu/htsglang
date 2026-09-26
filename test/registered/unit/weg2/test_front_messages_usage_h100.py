# SPDX-License-Identifier: Apache-2.0
"""H100 (#177): the front prices /v1/messages from the REAL prompt and cache counts.

THE DEFECT. htsglang's Anthropic adapter (entrypoints/anthropic/serving.py)
ships ``message_start`` BEFORE the backend reports usage -- ``input_tokens=0``,
no cache field -- and puts the real totals into the closing ``message_delta``:
``input_tokens`` = prompt - cached (``_anthropic_input_tokens``, the API's own
convention), ``cache_read_input_tokens`` = cached, ``output_tokens``. The
front's ``AnthropicStreamUsage`` read the cache count off ``message_start``
only and took ``input_tokens`` as the prompt, so every streamed leg booked
``record_presence(text, 0)`` and ``_note_exact(text, prompt - cached)``; the
non-streamed ``usage_of`` took ``input_tokens`` as the prompt too and priced
``uncached = prompt - 2*cached``.

Same root as the 27B line's RC7 Review V (4b) (71460c8130), which this tree
lacked; that fix is taken verbatim, plus ``cache_creation_input_tokens`` (the
third prompt term of the API; the adapter never emits it).

What this file adds over ``test_weg2_messages_wire_rc7_0925.py`` (ported from
the 27B line): the SEAM runs through the adapter's real stream GENERATOR and
its real non-stream converter, not a hand-built SSE body; and the wire shapes
the official streaming docs name are pinned, so no producer that follows the
docs is mis-read:

* real-API shape: totals in ``message_start``, ``message_delta`` has only
  ``output_tokens`` (docs' basic example);
* docs' web-search example: ``message_delta`` carries input, both cache fields
  and output (cumulative);
* ``cache_creation_input_tokens`` belongs to the prompt and to the uncached
  part, never to the cached one.

Hermetic, CPU.
"""
from __future__ import annotations

import asyncio
import json
import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede imports that may pull in sgl_kernel

from sglang.srt.entrypoints.anthropic.protocol import (  # noqa: E402
    AnthropicMessagesRequest,
)
from sglang.srt.entrypoints.anthropic.serving import AnthropicServing  # noqa: E402
from sglang.srt.entrypoints.openai.protocol import ChatCompletionResponse  # noqa: E402
from sglang.srt.weg2 import front as F  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

PROMPT = 18651
CACHED = 18649
OUT = 42


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


def _chunk(choices=None, usage=None) -> str:
    data = {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": choices or []}
    if usage is not None:
        data["usage"] = usage
    return f"data: {json.dumps(data)}\n\n"


def _engine_usage(prompt, cached, completion, report_cache=True) -> dict:
    u = {"prompt_tokens": prompt, "completion_tokens": completion,
         "total_tokens": prompt + completion}
    if report_cache:
        u["prompt_tokens_details"] = {"cached_tokens": cached}
    return u


def _adapter_sse(prompt, cached, completion, report_cache=True) -> bytes:
    """Run the adapter's REAL stream generator over one engine stream."""
    usage = _engine_usage(prompt, cached, completion, report_cache)
    lines = [
        _chunk([{"index": 0, "delta": {"role": "assistant", "content": "hi"},
                 "finish_reason": None}], usage=usage),
        _chunk([{"index": 0, "delta": {}, "finish_reason": "stop"}], usage=usage),
        _chunk([], usage=usage),
        "data: [DONE]\n\n",
    ]
    serving = AnthropicServing(_FakeChat(lines))
    req = AnthropicMessagesRequest(model="m", max_tokens=64, stream=True,
                                   messages=[{"role": "user", "content": "x"}])

    async def _collect():
        out = []
        async for sse in serving._generate_anthropic_stream(
            adapted_request=object(), processed_request=object(),
            anthropic_request=req, raw_request=object(),
        ):
            out.append(sse)
        return "".join(out).encode()

    return asyncio.new_event_loop().run_until_complete(_collect())


def _events(blob: bytes) -> dict:
    ev = {}
    for line in blob.decode().splitlines():
        if line.startswith("data: "):
            js = json.loads(line[6:])
            ev.setdefault(js["type"], js)
    return ev


def _adapter_body(prompt, cached, completion) -> dict:
    """Run the adapter's REAL non-stream converter; the JSON D sends back."""
    resp = ChatCompletionResponse.model_validate({
        "id": "c", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"},
                     "finish_reason": "stop"}],
        "usage": _engine_usage(prompt, cached, completion),
    })
    serving = AnthropicServing(_FakeChat([]))
    msg = serving._convert_response(resp)
    return json.loads(msg.model_dump_json(exclude_none=True))


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def _stream(start_usage: dict, delta_usage: dict) -> bytes:
    return b"".join([
        _sse("message_start", {"type": "message_start", "message": {
            "id": "msg_1", "type": "message", "role": "assistant", "model": "m",
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": start_usage}}),
        _sse("content_block_start", {"type": "content_block_start", "index": 0,
                                     "content_block": {"type": "text", "text": ""}}),
        _sse("content_block_delta", {"type": "content_block_delta", "index": 0,
                                     "delta": {"type": "text_delta", "text": "hi"}}),
        _sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
        _sse("message_delta", {"type": "message_delta",
                               "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                               "usage": delta_usage}),
        _sse("message_stop", {"type": "message_stop"}),
    ])


class TheAdapterWireIsWhatTheDefectAssumed(CustomTestCase):
    """The producer side, pinned: if the adapter ever moves its usage, the
    front tests below must be re-read, not silently keep passing."""

    def test_message_start_ships_zero_and_the_delta_carries_the_totals(self):
        ev = _events(_adapter_sse(PROMPT, CACHED, OUT))
        start = ev["message_start"]["message"]["usage"]
        delta = ev["message_delta"]["usage"]
        self.assertEqual(start.get("input_tokens"), 0)
        self.assertNotIn("cache_read_input_tokens", start)
        self.assertEqual(delta["input_tokens"], PROMPT - CACHED)
        self.assertEqual(delta["cache_read_input_tokens"], CACHED)
        self.assertEqual(delta["output_tokens"], OUT)

    def test_non_stream_body_reports_input_as_prompt_minus_cached(self):
        u = _adapter_body(PROMPT, CACHED, OUT)["usage"]
        self.assertEqual((u["input_tokens"], u["cache_read_input_tokens"]),
                         (PROMPT - CACHED, CACHED))


class TheFrontReadsTheRealAdapter(CustomTestCase):
    def test_stream_prompt_and_cache_through_the_real_generator(self):
        acc = F.AnthropicStreamUsage()
        acc.feed(_adapter_sse(PROMPT, CACHED, OUT))
        self.assertEqual(acc.result(), (PROMPT, CACHED, OUT, True),
                         "base read (2, 0, 42): presence 0, _note_exact prompt-cached")

    def test_stream_split_at_every_byte(self):
        blob = _adapter_sse(PROMPT, CACHED, OUT)
        acc = F.AnthropicStreamUsage()
        for i in range(len(blob)):
            acc.feed(blob[i:i + 1])
        self.assertEqual(acc.result(), (PROMPT, CACHED, OUT, True))

    def test_stream_without_a_cache_report(self):
        acc = F.AnthropicStreamUsage()
        acc.feed(_adapter_sse(9000, 0, OUT, report_cache=False))
        self.assertEqual(acc.result(), (9000, 0, OUT, True))

    def test_stream_fully_cached_is_priced(self):
        acc = F.AnthropicStreamUsage()
        acc.feed(_adapter_sse(5000, 5000, OUT))
        self.assertEqual(acc.result(), (5000, 5000, OUT, True), "base: unpriced (input 0)")

    def test_non_stream_uncached_is_what_d_prefilled(self):
        pt, ct, comp, priced = F.usage_of(_adapter_body(30000, 26000, OUT))
        self.assertTrue(priced)
        self.assertEqual((pt, ct, comp), (30000, 26000, OUT))
        self.assertEqual(pt - ct, 4000, "base: 4000 - 26000 -> clamped 0 uncached")

    def test_verdict_sees_the_true_uncached_extent(self):
        """The price feeds double_prefill_verdict: 4000 uncached against X=8192
        is 'serve' either way here, so pin the SIGN of the base error on a
        case where it flips: prompt 20000, cached 6000 -> 14000 uncached is a
        reroute at X=8192; the base read 14000 - 6000 = 8000 and served it."""
        pt, ct, _, _ = F.usage_of(_adapter_body(20000, 6000, OUT))
        self.assertEqual(F.double_prefill_verdict(pt, ct, 0, 8192), "reroute")


class TheDocumentedShapesStillRead(CustomTestCase):
    """Official streaming docs (platform.claude.com build-with-claude/streaming):
    message_delta usage counts are cumulative; message_start carries the
    input side on the real API."""

    def test_real_api_shape_totals_in_message_start(self):
        acc = F.AnthropicStreamUsage()
        acc.feed(_stream({"input_tokens": 25, "cache_read_input_tokens": 1000,
                          "cache_creation_input_tokens": 0, "output_tokens": 1},
                         {"output_tokens": 15}))
        self.assertEqual(acc.result(), (1025, 1000, 15, True))

    def test_docs_web_search_shape_all_fields_in_message_delta(self):
        acc = F.AnthropicStreamUsage()
        acc.feed(_stream({"input_tokens": 2679, "cache_creation_input_tokens": 0,
                          "cache_read_input_tokens": 0, "output_tokens": 3},
                         {"input_tokens": 10682, "cache_creation_input_tokens": 0,
                          "cache_read_input_tokens": 0, "output_tokens": 510,
                          "server_tool_use": {"web_search_requests": 1}}))
        self.assertEqual(acc.result(), (10682, 0, 510, True),
                         "cumulative: the delta's input wins over message_start's")

    def test_cache_creation_is_prompt_not_cached_stream(self):
        acc = F.AnthropicStreamUsage()
        acc.feed(_stream({"input_tokens": 0, "output_tokens": 0},
                         {"input_tokens": 10, "cache_creation_input_tokens": 3000,
                          "cache_read_input_tokens": 500, "output_tokens": 7}))
        pt, ct, _, priced = acc.result()
        self.assertTrue(priced)
        self.assertEqual((pt, ct), (3510, 500))

    def test_cache_creation_is_prompt_not_cached_body(self):
        body = {"type": "message", "usage": {"input_tokens": 10,
                                             "cache_creation_input_tokens": 3000,
                                             "cache_read_input_tokens": 500,
                                             "output_tokens": 7}}
        self.assertEqual(F.usage_of(body), (3510, 500, 7, True))

    def test_no_message_start_is_still_unpriced(self):
        acc = F.AnthropicStreamUsage()
        acc.feed(_sse("message_delta", {"type": "message_delta", "delta": {},
                                        "usage": {"input_tokens": 5, "output_tokens": 1}}))
        self.assertEqual(acc.result(), (0, 0, 0, False))


if __name__ == "__main__":
    unittest.main()
