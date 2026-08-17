"""#335 — the Ollama surface's observable shapes, pinned BEFORE the rewrite.

This is the NET. `ANALYSE_335_compat_surfaces.md` §6 says a compose-refactor of
the Ollama front "needs the contract tests extended to the happy path first, or
it is a rewrite without a net". This file is that, written against the
PARALLEL path and required to pass unchanged against the COMPOSED one.

Implemented against the Ollama API as its documentation describes `/api/chat`
and `/api/generate` (`docs/api.md` in ollama/ollama): NDJSON streaming, one
JSON object per line, `done` false until a final object that carries
`done_reason`; non-streaming returns a single object of the same shape. No
version is asserted -- Ollama versions the app, not the API, the same reason
the sdapi surface names a shape rather than a number.

WHAT IS GOLDEN HERE is the JSON a client sees: key sets, types, `done`
sequencing, the empty-prompt short circuit, and the refusal envelope. What is
deliberately NOT golden is anything internal -- which manager is called, how
sampling params are named -- because that is precisely what the rewrite
changes.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import asyncio
import json
import unittest
from typing import Any, Dict, List

from sglang.srt.entrypoints.ollama.protocol import (
    OllamaChatRequest,
    OllamaGenerateRequest,
    OllamaMessage,
)
from sglang.test.test_utils import CustomTestCase

#: Keys a stock client reads off a non-streaming /api/chat reply.
CHAT_KEYS = {
    "model",
    "created_at",
    "message",
    "done",
    "done_reason",
    "total_duration",
    "load_duration",
    "prompt_eval_count",
    "prompt_eval_duration",
    "eval_count",
    "eval_duration",
}

#: Keys on a streaming chunk. Deliberately a SUBSET of the above: Ollama's
#: stream objects are narrower than its final object, and a client that sees
#: timing fields mid-stream is being told the generation finished.
CHAT_STREAM_KEYS = {"model", "created_at", "message", "done", "done_reason"}


def _run(coro):
    return asyncio.run(coro)


class _FakeMeta(dict):
    pass


class _FakeTokenizerManager:
    """The parallel path's dependency, at its real call shape."""

    served_model_name = "test-model"

    def __init__(self, chunks: List[Dict[str, Any]]):
        self._chunks = chunks
        self.tokenizer = self
        self.last_obj = None

    # apply_chat_template is called by the parallel path; the composed path
    # must NOT call it (the OpenAI chat front owns templating).
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return [1, 2, 3]

    def generate_request(self, obj, raw_request):
        self.last_obj = obj

        async def _gen():
            for c in self._chunks:
                yield c

        return _gen()


def _one_shot(text="hello", prompt_tokens=3, completion_tokens=2):
    return [
        {
            "text": text,
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        }
    ]


def _serving(chunks=None):
    from sglang.srt.entrypoints.ollama.serving import OllamaServing

    return OllamaServing(_FakeTokenizerManager(chunks or _one_shot()))


def _body(response) -> Dict[str, Any]:
    """The JSON a client sees, whatever object the adapter returned."""
    if hasattr(response, "body"):
        return json.loads(response.body)
    return json.loads(response.model_dump_json())


async def _collect_ndjson(streaming_response) -> List[Dict[str, Any]]:
    out = []
    async for raw in streaming_response.body_iterator:
        line = raw.decode() if isinstance(raw, (bytes, bytearray)) else raw
        for part in line.splitlines():
            if part.strip():
                out.append(json.loads(part))
    return out


class TestChatNonStreaming(CustomTestCase):
    def test_the_reply_carries_exactly_the_protocol_keys(self):
        r = _run(
            _serving().handle_chat(
                OllamaChatRequest(
                    model="m",
                    messages=[OllamaMessage(role="user", content="hi")],
                    stream=False,
                ),
                None,
            )
        )
        self.assertEqual(set(_body(r)), CHAT_KEYS)

    def test_the_message_is_an_assistant_message_with_the_text(self):
        r = _run(
            _serving(_one_shot("hello there")).handle_chat(
                OllamaChatRequest(
                    model="m",
                    messages=[OllamaMessage(role="user", content="hi")],
                    stream=False,
                ),
                None,
            )
        )
        body = _body(r)
        self.assertEqual(body["message"]["role"], "assistant")
        self.assertEqual(body["message"]["content"], "hello there")

    def test_done_is_true_with_a_reason(self):
        r = _run(
            _serving().handle_chat(
                OllamaChatRequest(
                    model="m",
                    messages=[OllamaMessage(role="user", content="hi")],
                    stream=False,
                ),
                None,
            )
        )
        body = _body(r)
        self.assertIs(body["done"], True)
        self.assertEqual(body["done_reason"], "stop")

    def test_token_counts_are_reported(self):
        r = _run(
            _serving(_one_shot(prompt_tokens=7, completion_tokens=11)).handle_chat(
                OllamaChatRequest(
                    model="m",
                    messages=[OllamaMessage(role="user", content="hi")],
                    stream=False,
                ),
                None,
            )
        )
        body = _body(r)
        self.assertEqual(body["prompt_eval_count"], 7)
        self.assertEqual(body["eval_count"], 11)

    def test_the_model_name_is_the_served_model(self):
        r = _run(
            _serving().handle_chat(
                OllamaChatRequest(
                    model="whatever-the-client-said",
                    messages=[OllamaMessage(role="user", content="hi")],
                    stream=False,
                ),
                None,
            )
        )
        self.assertEqual(_body(r)["model"], "test-model")


class TestChatStreaming(CustomTestCase):
    """NDJSON: deltas, then a final object. A client concatenates the deltas,
    so emitting cumulative text instead would duplicate the whole reply."""

    def _chunks(self):
        return [
            {"text": "he", "meta_info": {}},
            {"text": "hello", "meta_info": {}},
            {
                "text": "hello",
                "meta_info": {"finish_reason": {"type": "stop"}},
            },
        ]

    def test_it_is_ndjson_one_object_per_line(self):
        r = _run(
            _serving(self._chunks()).handle_chat(
                OllamaChatRequest(
                    model="m",
                    messages=[OllamaMessage(role="user", content="hi")],
                    stream=True,
                ),
                None,
            )
        )
        self.assertEqual(r.media_type, "application/x-ndjson")
        lines = _run(_collect_ndjson(r))
        self.assertEqual(len(lines), 3)

    def test_content_is_the_DELTA_not_the_running_text(self):
        r = _run(
            _serving(self._chunks()).handle_chat(
                OllamaChatRequest(
                    model="m",
                    messages=[OllamaMessage(role="user", content="hi")],
                    stream=True,
                ),
                None,
            )
        )
        lines = _run(_collect_ndjson(r))
        self.assertEqual([l["message"]["content"] for l in lines], ["he", "llo", ""])

    def test_done_is_false_until_the_final_object(self):
        r = _run(
            _serving(self._chunks()).handle_chat(
                OllamaChatRequest(
                    model="m",
                    messages=[OllamaMessage(role="user", content="hi")],
                    stream=True,
                ),
                None,
            )
        )
        lines = _run(_collect_ndjson(r))
        self.assertEqual([l["done"] for l in lines], [False, False, True])

    def test_stream_objects_carry_the_narrow_key_set(self):
        r = _run(
            _serving(self._chunks()).handle_chat(
                OllamaChatRequest(
                    model="m",
                    messages=[OllamaMessage(role="user", content="hi")],
                    stream=True,
                ),
                None,
            )
        )
        for line in _run(_collect_ndjson(r)):
            self.assertEqual(set(line), CHAT_STREAM_KEYS)


class TestGenerate(CustomTestCase):
    def test_a_plain_prompt_answers_with_the_response_field(self):
        r = _run(
            _serving(_one_shot("world")).handle_generate(
                OllamaGenerateRequest(model="m", prompt="hello", stream=False), None
            )
        )
        self.assertEqual(_body(r)["response"], "world")

    def test_system_is_prepended_to_the_prompt(self):
        """Behavioural, not cosmetic: it is what the model actually sees."""
        tm = _FakeTokenizerManager(_one_shot())
        from sglang.srt.entrypoints.ollama.serving import OllamaServing

        _run(
            OllamaServing(tm).handle_generate(
                OllamaGenerateRequest(
                    model="m", prompt="hello", system="be terse", stream=False
                ),
                None,
            )
        )
        self.assertEqual(tm.last_obj.text, "be terse\n\nhello")

    def test_an_empty_prompt_short_circuits_without_generating(self):
        """The Ollama CLI sends an empty request on startup. Answering it with
        a real generation would burn a slot on every client launch."""
        tm = _FakeTokenizerManager(_one_shot())
        from sglang.srt.entrypoints.ollama.serving import OllamaServing

        r = _run(
            OllamaServing(tm).handle_generate(
                OllamaGenerateRequest(model="m", prompt="   ", stream=False), None
            )
        )
        body = _body(r)
        self.assertIs(body["done"], True)
        self.assertEqual(body["response"], "")
        self.assertIsNone(tm.last_obj, "an empty prompt must not reach the engine")

    def test_the_empty_prompt_short_circuit_also_holds_when_streaming(self):
        tm = _FakeTokenizerManager(_one_shot())
        from sglang.srt.entrypoints.ollama.serving import OllamaServing

        r = _run(
            OllamaServing(tm).handle_generate(
                OllamaGenerateRequest(model="m", prompt="", stream=True), None
            )
        )
        lines = _run(_collect_ndjson(r))
        self.assertEqual(len(lines), 1)
        self.assertIs(lines[0]["done"], True)
        self.assertIsNone(tm.last_obj)


class TestTheFourNamedRefusals(CustomTestCase):
    """9b5a72f826's work, which must survive the rewrite unchanged: a field the
    caller supplied that vanishes before the sampler is the #710 class."""

    def _chat(self, **kw):
        return _run(
            _serving().handle_chat(
                OllamaChatRequest(
                    model="m",
                    messages=[OllamaMessage(role="user", content="hi")],
                    stream=False,
                    **kw,
                ),
                None,
            )
        )

    def test_format_is_refused_today(self):
        r = self._chat(format="json")
        self.assertEqual(r.status_code, 400)
        self.assertIn("format", _body(r)["error"])

    def test_think_is_refused(self):
        r = self._chat(think=True)
        self.assertEqual(r.status_code, 400)
        self.assertIn("think", _body(r)["error"])

    def test_an_unmapped_option_is_refused_and_lists_the_mapped_set(self):
        r = self._chat(options={"mirostat": 2})
        self.assertEqual(r.status_code, 400)
        self.assertIn("mirostat", _body(r)["error"])

    def test_every_mapped_option_is_accepted(self):
        """The falsifier for the option gate: it must be able to pass."""
        r = self._chat(
            options={
                "temperature": 0.5,
                "top_p": 0.9,
                "top_k": 40,
                "num_predict": 64,
                "stop": ["x"],
                "presence_penalty": 0.1,
                "frequency_penalty": 0.2,
                "seed": 7,
            }
        )
        self.assertEqual(_body(r)["done"], True)

    def test_a_refused_request_never_reaches_the_engine(self):
        tm = _FakeTokenizerManager(_one_shot())
        from sglang.srt.entrypoints.ollama.serving import OllamaServing

        _run(
            OllamaServing(tm).handle_chat(
                OllamaChatRequest(
                    model="m",
                    messages=[OllamaMessage(role="user", content="hi")],
                    stream=False,
                    format="json",
                ),
                None,
            )
        )
        self.assertIsNone(tm.last_obj)


class TestSamplingDefaults(CustomTestCase):
    """The 2048 default is a deliberate adapter choice (Ollama clients expect
    longer replies than SGLang's default). Losing it in a rewrite would
    truncate every reply, which is the Kobold max_tokens=16 lesson."""

    def test_max_new_tokens_defaults_to_2048(self):
        tm = _FakeTokenizerManager(_one_shot())
        from sglang.srt.entrypoints.ollama.serving import OllamaServing

        _run(
            OllamaServing(tm).handle_generate(
                OllamaGenerateRequest(model="m", prompt="hi", stream=False), None
            )
        )
        self.assertEqual(tm.last_obj.sampling_params["max_new_tokens"], 2048)

    def test_num_predict_overrides_the_default(self):
        tm = _FakeTokenizerManager(_one_shot())
        from sglang.srt.entrypoints.ollama.serving import OllamaServing

        _run(
            OllamaServing(tm).handle_generate(
                OllamaGenerateRequest(
                    model="m", prompt="hi", stream=False, options={"num_predict": 17}
                ),
                None,
            )
        )
        self.assertEqual(tm.last_obj.sampling_params["max_new_tokens"], 17)


if __name__ == "__main__":
    unittest.main()
