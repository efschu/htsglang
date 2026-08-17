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
    def test_format_is_MAPPED_now_not_refused(self):
        """This file's subject is fields that VANISH. ``format`` used to be
        refused because the parallel path could not reach structured output;
        composing made it reachable, so the honest pin is that it is mapped --
        the same intent, against the new truth."""
        self.assertEqual(_serving().unsupported_reasons(_chat(format="json")), [])

    def test_format_json_becomes_a_json_object_response_format(self):
        self.assertEqual(_serving()._response_format("json"), {"type": "json_object"})

    def test_a_json_schema_format_is_wrapped_as_a_json_schema(self):
        """Ollama passes the schema itself; OpenAI wants it wrapped and named.
        ``strict`` is set because a caller who supplied a schema wants it
        obeyed, not approximated."""
        schema = {"type": "object", "properties": {}}
        got = _serving()._response_format(schema)
        self.assertEqual(got["type"], "json_schema")
        self.assertEqual(got["json_schema"]["schema"], schema)
        self.assertIs(got["json_schema"]["strict"], True)

    def test_think_as_a_boolean_is_honoured_not_refused(self):
        """#557's mechanism reaches the chat front, so the toggle is wired."""
        self.assertEqual(_serving().unsupported_reasons(_chat(think=True)), [])
        self.assertEqual(_serving().unsupported_reasons(_chat(think=False)), [])

    def test_an_effort_level_is_refused_with_the_route_named(self):
        """It would become reasoning_effort, and the served checkpoint takes
        effort by OMISSION with explicit high/max observed to fail. Refusing
        beats turning the caller's request into an error they did not cause."""
        reasons = _serving().unsupported_reasons(_chat(think="high"))
        self.assertEqual(len(reasons), 1)
        self.assertIn("reasoning_effort", reasons[0])

    def test_generate_is_gated_the_same_way(self):
        """Both handlers, or a client just moves to the ungated one."""
        self.assertTrue(_serving().unsupported_reasons(_gen(think=True)))
        # /api/generate composes CompletionRequest, which carries no
        # chat_template_kwargs at all -- there is no template to toggle.


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

    def test_supported_options_equals_the_mapping_exactly(self):
        """Stronger than the source scan this replaces: after the compose
        rewrite the mapping IS a dict, so set equality can be asserted instead
        of looking for a quoted name in a function body. Drift in either
        direction now fails."""
        self.assertEqual(
            set(OllamaServing.SUPPORTED_OPTIONS),
            set(OllamaServing.OPTION_MAP),
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
                # The generating call is now handle_request on the OpenAI
                # front rather than a sampling_params construction; the pin's
                # intent -- refuse before anything is generated -- is
                # unchanged, only the marker for "generation starts here".
                self.assertLess(
                    src.index("unsupported_reasons"),
                    src.index("handle_request"),
                    "the refusal must precede generation",
                )


if __name__ == "__main__":
    unittest.main()
