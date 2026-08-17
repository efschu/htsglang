"""#335: the Ollama surface must not silently overrule the caller.

INVENTORY FINDING THIS PINS. The Ollama front is mounted and real
(`/api/chat`, `/api/generate`, `/api/tags`, `/api/show` in `http_server.py`),
but it declared four request fields it never read:

  * ``format`` -- structured output. A caller asking for JSON got free-form
    text. Their program then json.loads() it and breaks, with nothing in the
    response explaining why. This is the worst of the four because the
    failure surfaces far from its cause.
  * ``think`` -- never read.
  * ``keep_alive`` -- never read (harmless, but silence is still a claim).
  * every ``options`` key outside the eight it maps -- ``repeat_penalty``,
    ``num_ctx``, ``min_p``, ``mirostat``... all dropped. The model then
    samples differently than asked and says nothing.

That is the #710 tool-arg-loss family exactly: a value the caller supplied
that vanishes between request and sampler. The remedy is a NAMED refusal
before generation, never a plausible wrong answer.

WHY REFUSE RATHER THAN WIRE. ``format`` cannot simply be wired here: this
adapter drives the tokenizer manager DIRECTLY and never reaches the OpenAI
front's ``response_format`` machinery. Wiring it properly means making the
Ollama front compose the OpenAI serving path instead of paralleling it --
a structural change, recorded in the determination note, not smuggled into a
bug fix.

Hermetic: the refusal surface is a pure function of the request. No server,
no tokenizer, no CUDA.
"""

import unittest

from sglang.srt.entrypoints.ollama.protocol import (
    OllamaChatRequest,
    OllamaGenerateRequest,
)
from sglang.srt.entrypoints.ollama.serving import OllamaServing


def _serving():
    return OllamaServing.__new__(OllamaServing)


def _chat(**over):
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    body.update(over)
    return OllamaChatRequest(**body)


def _gen(**over):
    body = {"model": "m", "prompt": "hi"}
    body.update(over)
    return OllamaGenerateRequest(**body)


class TestAnHonourableRequestIsNotRefused(unittest.TestCase):
    """The refusal must be narrow, or it breaks every ordinary client."""

    def test_a_plain_chat_request_passes(self):
        self.assertEqual(_serving().unsupported_reasons(_chat()), [])

    def test_a_plain_generate_request_passes(self):
        self.assertEqual(_serving().unsupported_reasons(_gen()), [])

    def test_every_mapped_option_passes(self):
        options = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "num_predict": 128,
            "stop": ["\n"],
            "presence_penalty": 0.1,
            "frequency_penalty": 0.1,
            "seed": 42,
        }
        self.assertEqual(_serving().unsupported_reasons(_chat(options=options)), [])

    def test_keep_alive_is_inert_not_refused(self):
        """Accepted and ignored is legitimate HERE -- SGLang keeps the model
        resident, so the caller's intent is already satisfied. It is declared
        inert rather than left to look honoured by accident."""
        self.assertEqual(_serving().unsupported_reasons(_chat(keep_alive="5m")), [])
        self.assertIn("keep_alive", OllamaServing.INERT_FIELDS)


class TestUnhonouredFieldsAreRefusedByName(unittest.TestCase):
    def test_format_json_is_refused(self):
        reasons = _serving().unsupported_reasons(_chat(format="json"))
        self.assertEqual(len(reasons), 1)
        self.assertIn("format", reasons[0])

    def test_the_format_refusal_names_the_working_alternative(self):
        """A refusal that only blocks teaches nothing; this one routes."""
        reasons = _serving().unsupported_reasons(_chat(format="json"))
        self.assertIn("/v1/chat/completions", reasons[0])
        self.assertIn("response_format", reasons[0])

    def test_a_json_schema_format_is_refused_too(self):
        reasons = _serving().unsupported_reasons(
            _chat(format={"type": "object", "properties": {}})
        )
        self.assertTrue(reasons)

    def test_think_is_refused(self):
        reasons = _serving().unsupported_reasons(_chat(think=True))
        self.assertEqual(len(reasons), 1)
        self.assertIn("think", reasons[0])

    def test_generate_is_gated_the_same_way(self):
        """Both handlers, or a client just moves to the ungated one."""
        self.assertTrue(_serving().unsupported_reasons(_gen(format="json")))


class TestUnmappedOptionsAreRefusedNotDropped(unittest.TestCase):
    """The largest silent-drop surface: everything outside the mapped eight."""

    def test_repeat_penalty_is_refused(self):
        reasons = _serving().unsupported_reasons(_chat(options={"repeat_penalty": 1.2}))
        self.assertEqual(len(reasons), 1)
        self.assertIn("repeat_penalty", reasons[0])

    def test_num_ctx_is_refused(self):
        reasons = _serving().unsupported_reasons(_chat(options={"num_ctx": 8192}))
        self.assertIn("num_ctx", reasons[0])

    def test_the_refusal_lists_what_IS_supported(self):
        reasons = _serving().unsupported_reasons(_chat(options={"mirostat": 2}))
        self.assertIn("temperature", reasons[0])

    def test_it_says_why_dropping_would_be_worse(self):
        reasons = _serving().unsupported_reasons(_chat(options={"min_p": 0.05}))
        self.assertIn("sample differently than you asked", reasons[0])

    def test_several_unknown_options_are_named_together(self):
        reasons = _serving().unsupported_reasons(
            _chat(options={"mirostat": 2, "tfs_z": 1.0})
        )
        self.assertEqual(len(reasons), 1)
        self.assertIn("mirostat", reasons[0])
        self.assertIn("tfs_z", reasons[0])

    def test_a_mapped_and_an_unmapped_option_together_still_refuse(self):
        """Partial honouring is the silent-overrule shape."""
        reasons = _serving().unsupported_reasons(
            _chat(options={"temperature": 0.5, "repeat_penalty": 1.1})
        )
        self.assertTrue(reasons)
        self.assertNotIn("temperature", reasons[0].split(":")[0])


class TestTheMappedSetMatchesTheConverter(unittest.TestCase):
    """The declared support list and the actual mapping must not drift --
    a refusal that names a supported option, or admits an unmapped one, is
    worse than none."""

    def test_supported_options_equals_the_converters_mapping(self):
        import inspect

        src = inspect.getsource(OllamaServing._convert_options_to_sampling_params)
        for name in OllamaServing.SUPPORTED_OPTIONS:
            with self.subTest(option=name):
                self.assertIn(
                    f'"{name}"',
                    src,
                    "declared supported but not mapped by the converter",
                )


class TestTheGuardRunsBeforeGeneration(unittest.TestCase):
    """Source pin: refusing after the model has already sampled would defeat
    the point -- the caller would pay for a wrong answer and then be told."""

    def test_both_handlers_check_before_they_generate(self):
        import inspect

        for handler in (OllamaServing.handle_chat, OllamaServing.handle_generate):
            with self.subTest(handler=handler.__name__):
                src = inspect.getsource(handler)
                self.assertIn("unsupported_reasons", src)
                self.assertLess(
                    src.index("unsupported_reasons"),
                    src.index("sampling_params"),
                    "the refusal must precede sampling",
                )


if __name__ == "__main__":
    unittest.main()
