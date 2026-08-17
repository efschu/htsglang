"""#335 — the Ollama surface COMPOSES; it must never serve again.

The parallel path is what made ``format`` unreachable
(`ANALYSE_335_compat_surfaces.md` §2): the adapter applied the chat template
itself and drove the tokenizer manager, so the request never met the machinery
that implements structured output. These pins make the composed shape a
PROPERTY rather than an intention, the same way the KoboldCpp and sdapi
surfaces do it.

The behavioural half -- that no client-visible shape changed -- lives in
``test_ollama_golden_shapes_335.py``, which was committed BEFORE the rewrite
precisely so it could not be edited to fit the outcome.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import asyncio
import inspect
import unittest

from sglang.srt.entrypoints.ollama.protocol import (
    OllamaChatRequest,
    OllamaGenerateRequest,
    OllamaMessage,
)
from sglang.srt.entrypoints.ollama.serving import DEFAULT_NUM_PREDICT, OllamaServing
from sglang.test.test_utils import CustomTestCase


def _code_only() -> str:
    """The serving module's source with comments and string literals removed.

    Docstrings ARE string literals, so this is what "the code, not the prose"
    means mechanically.
    """
    import io
    import tokenize

    from sglang.srt.entrypoints.ollama import serving as _mod

    src = inspect.getsource(_mod)
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        out.append(tok.string)
    return " ".join(out)


class _Front:
    status_code = 200

    def __init__(self, chat=True):
        self.chat = chat
        self.last_request = None

    async def handle_request(self, request, raw_request):
        self.last_request = request
        choice = type(
            "C", (), {"message": type("M", (), {"content": "x"})(), "text": "x"}
        )()
        return type(
            "R",
            (),
            {
                "status_code": 200,
                "choices": [choice],
                "usage": type("U", (), {"prompt_tokens": 1, "completion_tokens": 1})(),
            },
        )()


def _serving():
    chat, comp = _Front(True), _Front(False)
    return OllamaServing(chat, comp, model_name="test-model"), chat, comp


def _run(c):
    return asyncio.run(c)


def _chat(**kw):
    return OllamaChatRequest(
        model="m",
        messages=[OllamaMessage(role="user", content="hi")],
        stream=False,
        **kw,
    )


class TestItCannotServe(CustomTestCase):
    """The forbidden vocabulary. If any of this returns, the surface has grown
    a second sampling path and the whole family's discipline is gone."""

    def test_the_module_holds_no_engine_vocabulary(self):
        """CODE ONLY, deliberately.

        A plain substring scan over the source fired on this module's own
        docstring, which names ``tokenizer_manager`` and
        ``guard_generate_stream`` to explain what it no longer does and why
        the #344 watchdog sits upstream. Forbidding the EXPLANATION would
        punish the documentation for being specific.

        Stripping comments and strings makes the pin say what it means -- the
        module does not CALL these -- and makes it strictly stronger: it can
        no longer be satisfied by moving a call into a docstring, nor broken
        by describing the design accurately.
        """
        for forbidden in (
            "tokenizer_manager",
            "apply_chat_template",
            "sampling_params",
            "GenerateReqInput",
            "guard_generate_stream",
        ):
            with self.subTest(token=forbidden):
                self.assertNotIn(forbidden, _code_only())

    def test_it_calls_the_openai_fronts(self):
        """Whitespace-squeezed, because _code_only joins TOKENS: the call
        syntax survives, the spacing does not."""
        squeezed = "".join(_code_only().split())
        self.assertIn("self._chat.handle_request(", squeezed)
        self.assertIn("self._completion.handle_request(", squeezed)

    def test_chat_goes_to_the_chat_front_and_generate_to_the_completion_front(self):
        """Routing them the other way would apply a chat template to a raw
        /api/generate prompt, which is a different generation."""
        serving, chat, comp = _serving()
        _run(serving.handle_chat(_chat(), None))
        self.assertIsNotNone(chat.last_request)
        self.assertIsNone(comp.last_request)

        serving, chat, comp = _serving()
        _run(
            serving.handle_generate(
                OllamaGenerateRequest(model="m", prompt="hi", stream=False), None
            )
        )
        self.assertIsNotNone(comp.last_request)
        self.assertIsNone(chat.last_request)


class TestTheFormatWin(CustomTestCase):
    """What the rewrite was for: a refusal became a feature."""

    def test_format_json_reaches_the_front_as_response_format(self):
        serving, chat, _ = _serving()
        _run(serving.handle_chat(_chat(format="json"), None))
        self.assertEqual(
            chat.last_request.response_format.type,
            "json_object",
        )

    def test_a_schema_reaches_the_front_as_json_schema(self):
        serving, chat, _ = _serving()
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        _run(serving.handle_chat(_chat(format=schema), None))
        rf = chat.last_request.response_format
        self.assertEqual(rf.type, "json_schema")
        self.assertEqual(rf.json_schema.schema_, schema)

    def test_no_format_means_no_response_format(self):
        """The falsifier: the field must not appear when the caller did not
        ask for it, or every request would silently become structured."""
        serving, chat, _ = _serving()
        _run(serving.handle_chat(_chat(), None))
        self.assertIsNone(chat.last_request.response_format)

    def test_format_also_reaches_the_generate_path(self):
        serving, _, comp = _serving()
        _run(
            serving.handle_generate(
                OllamaGenerateRequest(
                    model="m", prompt="hi", stream=False, format="json"
                ),
                None,
            )
        )
        self.assertEqual(comp.last_request.response_format.type, "json_object")


class TestOptionsReachTheRequest(CustomTestCase):
    """Every mapped option must actually land on the OpenAI request -- the
    #710 lesson is that a mapping nobody checks is a drop with extra steps."""

    def test_each_mapped_option_lands(self):
        cases = {
            "temperature": ("temperature", 0.5),
            "top_p": ("top_p", 0.9),
            "top_k": ("top_k", 40),
            "num_predict": ("max_tokens", 64),
            "presence_penalty": ("presence_penalty", 0.1),
            "frequency_penalty": ("frequency_penalty", 0.2),
            "seed": ("seed", 7),
        }
        for ollama_key, (openai_key, value) in cases.items():
            with self.subTest(option=ollama_key):
                serving, chat, _ = _serving()
                _run(serving.handle_chat(_chat(options={ollama_key: value}), None))
                self.assertEqual(getattr(chat.last_request, openai_key), value)

    def test_the_2048_default_is_the_adapters_choice_not_the_openai_one(self):
        serving, chat, _ = _serving()
        _run(serving.handle_chat(_chat(), None))
        self.assertEqual(chat.last_request.max_tokens, DEFAULT_NUM_PREDICT)
        self.assertEqual(DEFAULT_NUM_PREDICT, 2048)


class TestErrorsArePassedThrough(CustomTestCase):
    def test_a_front_error_is_returned_unchanged(self):
        """Re-dressing it in an Ollama envelope would hide which layer
        refused; the OpenAI error already names its own cause."""

        class _Err:
            status_code = 503

        serving, chat, _ = _serving()
        chat.handle_request = lambda req, raw: _async(_Err())
        r = _run(serving.handle_chat(_chat(), None))
        self.assertEqual(r.status_code, 503)


async def _async(v):
    return v


if __name__ == "__main__":
    unittest.main()
