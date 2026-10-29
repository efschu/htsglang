"""Lone UTF-16 surrogates must not turn a turn into a 500 (1029).

Mechanism: a JSON string ``"\\ud83d"`` (half an emoji, as an agent client that
cuts a JS string mid-emoji emits it) decodes to a ``str`` that cannot be
UTF-8 encoded. Input side: the HF tokenizer rejects it (``TypeError:
TextInputSequence must be str``) -> 500, and the text stays in the client's
history so every later turn fails too. Output side: parsed tool-call
arguments with a broken pair make orjson / Starlette's JSONResponse raise
-> 500. Each lone surrogate is now replaced by U+FFFD at the request / tool
call model boundary; text without one is returned as the same object.

No GPU, no model: run with CUDA_VISIBLE_DEVICES=99.
"""

from flliper.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import json
import unittest

from flliper.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ToolCall,
)
from flliper.srt.entrypoints.openai.text_sanitize import scrub_lone_surrogates
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_HALF = "\ud83d"  # lone high surrogate
_EMOJI = "\U0001f600"  # a complete emoji (one code point after json.loads)


def _every_string_encodes(value) -> bool:
    """True when no string anywhere in the structure is UTF-8-unencodable."""
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            return False
        return True
    if isinstance(value, list):
        return all(_every_string_encodes(v) for v in value)
    if isinstance(value, dict):
        return all(
            _every_string_encodes(k) and _every_string_encodes(v)
            for k, v in value.items()
        )
    return True


class TestScrubHelper(unittest.TestCase):
    def test_lone_surrogate_becomes_replacement_character(self):
        self.assertEqual(scrub_lone_surrogates("a" + _HALF + "b"), "a�b")
        self.assertEqual(
            scrub_lone_surrogates({"k" + _HALF: [_HALF, 1, None]}),
            {"k�": ["�", 1, None]},
        )

    def test_clean_text_is_returned_unchanged_and_identical(self):
        clean = {"a": ["hällo " + _EMOJI, {"b": "plain"}], "n": 3}
        self.assertIs(scrub_lone_surrogates(clean), clean)


class TestRequestIngress(unittest.TestCase):
    def test_surrogates_in_history_never_reach_the_chat_template(self):
        request = ChatCompletionRequest(
            model="x",
            messages=[
                {"role": "user", "content": "hi " + _HALF},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "Bash",
                                "arguments": json.dumps({"c": _HALF}),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [{"type": "text", "text": "out " + _HALF}],
                },
            ],
        )
        dumped = [m.model_dump() for m in request.messages]
        self.assertTrue(_every_string_encodes(dumped))

    def test_valid_requests_are_untouched(self):
        text = "hällo " + _EMOJI
        request = ChatCompletionRequest(
            model="x", messages=[{"role": "user", "content": text}]
        )
        self.assertEqual(request.messages[0].content, text)


class TestToolCallEgress(unittest.TestCase):
    def test_parsed_arguments_with_broken_pair_survive_json_encoding(self):
        call = ToolCall.model_validate(
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "Bash",
                    "arguments": json.dumps({"c": "x" + _HALF}, ensure_ascii=False),
                },
            }
        )
        # what the response encoders do with the model
        encoded = call.model_dump_json().encode("utf-8")
        self.assertIn("�", json.loads(encoded)["function"]["arguments"])

    def test_valid_arguments_are_untouched(self):
        arguments = json.dumps({"c": "hällo " + _EMOJI}, ensure_ascii=False)
        call = ToolCall.model_validate(
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "Bash", "arguments": arguments},
            }
        )
        self.assertEqual(call.function.arguments, arguments)


if __name__ == "__main__":
    unittest.main()
