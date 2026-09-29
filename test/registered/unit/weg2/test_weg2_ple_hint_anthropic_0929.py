"""H43 on /v1/messages: the front's PLE hint reaches P for an Anthropic leg 1.

NF z30k (29.09., boot ...dauer09290122_5527da6564): 71x ``BATCH queued
(awake=D ...)``, 0x ``WEG2 PLE-HINT`` -- the agent load comes over
``/v1/messages`` and ``ple_admit_hint`` knew only the OpenAI paths, so every
hint was dropped silently (``chunk=0 ... ready=no wait_ms~1000`` after each
D->P flip). The hint must tokenize an Anthropic body exactly as its leg-1 POST
does: AnthropicServing's conversion, then the chat serving's internal request.
"""

import asyncio
import json
import unittest
from types import SimpleNamespace

from sglang.srt.entrypoints.anthropic.serving import AnthropicServing
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.weg2 import ple_admit_hint as h


class _Chat:
    """The chat serving's ``_convert_to_internal_request`` stand-in: the ids are
    a pure function of what the chat template sees (messages, tools, kwargs),
    so two paths give equal ids iff they hand the same chat request over."""

    tokenizer_manager = None

    def __init__(self):
        self.seen = []

    def apply_reasoning_enabled(self, req, enabled):
        # the real one writes the reasoning toggle into chat_template_kwargs
        req.chat_template_kwargs = dict(req.chat_template_kwargs or {}, enable_thinking=enabled)

    def wrap_reasoning_history(self, text):
        return "<think>" + text + "</think>"

    def _convert_to_internal_request(self, req, raw_request=None):
        assert isinstance(req, ChatCompletionRequest)
        self.seen.append(req)
        view = json.dumps(
            {
                "messages": [m if isinstance(m, dict) else m.model_dump(exclude_none=True)
                             for m in req.messages],
                "tools": [t.model_dump(exclude_none=True) for t in (req.tools or [])],
                "kwargs": req.chat_template_kwargs,
            },
            sort_keys=True,
            default=str,
        )
        return SimpleNamespace(input_ids=[ord(c) % 997 for c in view], text=None), None


def _anthropic_body(rid="weg2-7-17"):
    return {
        "model": "nf",
        "max_tokens": 1,
        "system": "You are the agent. Tools follow.",
        "messages": [
            {"role": "user", "content": "Lies die Datei und fasse sie zusammen."},
            {"role": "assistant", "content": "Gern."},
            {"role": "user", "content": [{"type": "text", "text": "Weiter."}]},
        ],
        "rid": rid,
        "stream": True,
    }


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class TestPleHintAnthropic(unittest.TestCase):
    def setUp(self):
        self.chat = _Chat()
        self.anthropic = AnthropicServing(self.chat)

    def test_front_builds_a_body_for_messages(self):
        body = h.ple_hint_body("/v1/messages", _anthropic_body())
        self.assertIsNotNone(body, "z30k: /v1/messages hint dropped at the front")
        self.assertEqual(body["path"], "/v1/messages")
        self.assertNotIn("stream", body["payload"])
        self.assertIsNone(h.ple_hint_skip_reason("/v1/messages", _anthropic_body()))

    def test_messages_hint_tokens_equal_the_leg1_conversion(self):
        payload = _anthropic_body()
        hint = _run(h.build_ple_prefetch_hint(
            h.ple_hint_body("/v1/messages", payload),
            serving_chat=self.chat, serving_completion=None,
            encode=lambda s: [0], serving_anthropic=self.anthropic,
        ))
        self.assertIsNotNone(hint)
        self.assertEqual(hint.rid, "weg2-7-17")
        # the leg-1 POST: handle_messages -> the same conversion -> chat serving
        from sglang.srt.entrypoints.anthropic.protocol import AnthropicMessagesRequest

        leg1 = self.anthropic._convert_to_chat_completion_request(
            AnthropicMessagesRequest(**dict(payload, stream=False)))
        ids, _ = self.chat._convert_to_internal_request(leg1)
        self.assertEqual(hint.input_ids, ids.input_ids)
        self.assertEqual(leg1.rid, "weg2-7-17")

    def test_messages_without_anthropic_serving_is_no_hint(self):
        hint = _run(h.build_ple_prefetch_hint(
            h.ple_hint_body("/v1/messages", _anthropic_body()),
            serving_chat=self.chat, serving_completion=None, encode=lambda s: [0],
        ))
        self.assertIsNone(hint)

    def test_skip_reason_names_what_is_missing(self):
        self.assertEqual(h.ple_hint_skip_reason("/v1/embeddings", {"rid": "x"}), "path")
        self.assertEqual(h.ple_hint_skip_reason("/v1/messages", {"model": "m"}), "rid")
        self.assertEqual(h.ple_hint_skip_reason("/v1/messages", None), "payload")

    def test_openai_chat_path_unchanged(self):
        payload = {"model": "nf", "max_tokens": 1, "rid": "weg2-1-2",
                   "messages": [{"role": "user", "content": "hi"}]}
        hint = _run(h.build_ple_prefetch_hint(
            h.ple_hint_body("/v1/chat/completions", payload),
            serving_chat=self.chat, serving_completion=None, encode=lambda s: [0],
            serving_anthropic=self.anthropic,
        ))
        self.assertIsNotNone(hint)
        self.assertEqual(hint.rid, "weg2-1-2")


if __name__ == "__main__":
    unittest.main()
