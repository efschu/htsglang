# SPDX-License-Identifier: Apache-2.0
"""RC7 Review V (4): /v1/messages through the Weg-2 front, two wire defects.

(4a) THE RID. The front sets ``payload["rid"]`` so both legs of a request run
under ONE id (#1442 P->D hand-off, P's end-anchor trim contract, and
/abort_request all key on it). ``AnthropicMessagesRequest`` did not declare
``rid`` and pydantic's default ``extra="ignore"`` dropped it: on Messages
traffic D and P ran the request under ids of their own. Declared now and passed
into the chat request exactly as ``/v1/chat/completions`` does.

(4b) THE PROMPT COUNT. The adapter reports ``input_tokens = prompt - cached``
and the cached part as ``cache_read_input_tokens`` (anthropic/serving.py
``_anthropic_input_tokens``, like the Anthropic API). The front read
``input_tokens`` as the prompt: non-streamed legs priced uncached as
prompt - 2*cached; streamed legs (message_start goes out BEFORE usage is known,
with zeros) took the prompt minus cached from the closing message_delta and
never read its cache field at all -- every streamed Messages leg recorded
"D holds nothing of this text" into the span LRU and a wrong tokenisation into
``_note_exact``.

Both directions each; the adapter->front seam is exercised with the REAL
adapter function. Hermetic, CPU.
"""
from __future__ import annotations

import json
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from sglang.srt.entrypoints.anthropic.serving import (
    AnthropicServing,
    _anthropic_usage_from_openai,
)
from sglang.srt.weg2 import front as F
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _NoReasoning:
    def apply_reasoning_enabled(self, *a, **kw):
        return None


def _convert(**kw):
    base = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16}
    base.update(kw)
    serving = AnthropicServing.__new__(AnthropicServing)
    serving._merge_inline_system = False
    serving.openai_serving_chat = _NoReasoning()
    return AnthropicServing._convert_to_chat_completion_request(
        serving, AnthropicMessagesRequest(**base))


class RidReachesThePipeline(CustomTestCase):
    def test_the_fronts_rid_is_carried(self):
        self.assertEqual(_convert(rid="weg2-3-17").rid, "weg2-3-17")

    def test_no_rid_is_no_rid_as_before(self):
        self.assertIsNone(_convert().rid)

    def test_the_request_model_declares_it(self):
        req = AnthropicMessagesRequest(model="m", messages=[{"role": "user", "content": "x"}],
                                       max_tokens=1, rid="weg2-0-1")
        self.assertEqual(req.rid, "weg2-0-1", "undeclared, extra='ignore' dropped it")


class _Details:
    def __init__(self, cached):
        self.cached_tokens = cached


class _Usage:
    def __init__(self, prompt, cached, completion=42, report_cache=True):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.prompt_tokens_details = _Details(cached) if report_cache else None


def _wire(usage) -> dict:
    """What the adapter really puts on the wire for this engine usage."""
    return json.loads(_anthropic_usage_from_openai(
        usage, include_input=True, include_output=True).model_dump_json(exclude_none=True))


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def _adapter_stream(final_usage: dict) -> bytes:
    """The RC6 adapter's stream shape: message_start at once with zeros and no
    cache field (usage unknown yet), the real totals in the closing delta."""
    return b"".join([
        _sse("message_start", {"type": "message_start", "message": {
            "id": "msg_1", "type": "message", "role": "assistant", "model": "m",
            "content": [], "usage": {"input_tokens": 0, "output_tokens": 0}}}),
        _sse("content_block_delta", {"type": "content_block_delta", "index": 0,
                                     "delta": {"type": "text_delta", "text": "hi"}}),
        _sse("message_delta", {"type": "message_delta",
                               "delta": {"stop_reason": "end_turn"}, "usage": final_usage}),
        _sse("message_stop", {"type": "message_stop"}),
    ])


class ThePromptIsInputPlusCacheRead(CustomTestCase):
    """The leg-2 after P: prompt 18651, cached 18649 -> 2 uncached."""

    def test_non_streamed_body_through_the_real_adapter(self):
        u = _wire(_Usage(18651, 18649))
        self.assertEqual((u["input_tokens"], u["cache_read_input_tokens"]), (2, 18649),
                         "the adapter's own convention: input = prompt - cached")
        self.assertEqual(F.usage_of({"type": "message", "usage": u}), (18651, 18649, 42, True))

    def test_non_streamed_without_a_cache_report_is_the_prompt(self):
        u = _wire(_Usage(9000, 0, report_cache=False))
        self.assertNotIn("cache_read_input_tokens", u)
        self.assertEqual(F.usage_of({"type": "message", "usage": u}), (9000, 0, 42, True))

    def test_streamed_leg_reads_the_correcting_delta(self):
        acc = F.AnthropicStreamUsage()
        acc.feed(_adapter_stream(_wire(_Usage(18651, 18649))))
        self.assertEqual(acc.result(), (18651, 18649, 42, True),
                         "was (2, 0, 42): the prompt minus cached and NO cached count")

    def test_a_fully_cached_streamed_prompt_is_priced(self):
        acc = F.AnthropicStreamUsage()
        acc.feed(_adapter_stream(_wire(_Usage(5000, 5000))))
        self.assertEqual(acc.result(), (5000, 5000, 42, True), "was unpriced (input 0)")

    def test_streamed_without_a_cache_report(self):
        acc = F.AnthropicStreamUsage()
        acc.feed(_adapter_stream(_wire(_Usage(9000, 0, report_cache=False))))
        self.assertEqual(acc.result(), (9000, 0, 42, True))

    def test_uncached_is_what_d_computed(self):
        pt, ct, _, _ = F.usage_of({"usage": _wire(_Usage(30000, 26000))})
        self.assertEqual(pt - ct, 4000)


if __name__ == "__main__":
    unittest.main()
