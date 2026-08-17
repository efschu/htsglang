"""#557: two user-flagged gaps on the Anthropic front.

Both are adapter-layer, both hermetic: a request model in, a converted request
or a usage block out. No server, no engine, no CUDA.

GAP 1 -- per-request ``chat_template_kwargs``. The Anthropic adapter converts
into an OpenAI ``ChatCompletionRequest`` and delegates
(``serving.py:_convert_to_chat_completion_request``, then
``self.openai_serving_chat``). The OpenAI front has carried
``chat_template_kwargs`` all along (``openai/protocol.py:844``), but
``AnthropicMessagesRequest`` declares no such field and sets no ``model_config``
-- so pydantic's default ``extra="ignore"`` applies and a client's
``chat_template_kwargs`` is SILENTLY DROPPED. Not rejected, which would at
least be visible: dropped.

The semantics are mirrored, not invented. ``apply_reasoning_enabled``
(``openai/serving_chat.py:1823-1825``) already MERGES into whatever the request
carries:

    chat_template_kwargs = dict(request.chat_template_kwargs or {})
    chat_template_kwargs[toggle_param] = enabled
    request.chat_template_kwargs = chat_template_kwargs

so a caller's other keys survive and only the reasoning toggle is overridden by
Anthropic's typed ``thinking`` semantics. That interaction is pinned below
rather than left to be rediscovered.

GAP 2 -- ``cache_read_input_tokens``. Already mapped, from the right source:
``_cached_prompt_tokens`` reads ``usage.prompt_tokens_details.cached_tokens``
(the real count, NOT the broken ``cache_hit_rate``). The defect is narrower and
is exactly the absent-vs-0 ambiguity: the field is written under

    if cached_tokens:

which is falsy at zero, so a zero-cache request OMITS the field instead of
reporting 0. A client then cannot tell "no cache hit" from "this server does
not report cache usage".
"""

import unittest

from sglang.srt.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from sglang.srt.entrypoints.anthropic.serving import (
    AnthropicServing,
    _anthropic_usage_from_openai,
)


class _PromptDetails:
    def __init__(self, cached_tokens):
        self.cached_tokens = cached_tokens


class _Usage:
    """A fake engine usage block. Only the attributes the adapter reads."""

    def __init__(self, prompt_tokens=100, completion_tokens=7, cached_tokens=0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.prompt_tokens_details = _PromptDetails(cached_tokens)


def _request(**kwargs):
    base = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 16,
    }
    base.update(kwargs)
    return AnthropicMessagesRequest(**base)


def _convert(request):
    """Run the real converter without constructing a whole serving stack."""
    serving = AnthropicServing.__new__(AnthropicServing)
    # The converter reads exactly one instance attribute set in __init__.
    # Supplying it beats constructing a whole serving stack, and False is the
    # inline-system-supporting default (detect_inline_system_support -> True).
    serving._merge_inline_system = False
    # The converter also calls apply_reasoning_enabled on the OpenAI handler
    # (Anthropic's thinking-off-by-default branch). A no-op stub is correct
    # HERE because these pins are about the kwargs reaching the converted
    # request; the real merge behaviour is pinned separately, and by SOURCE,
    # in TestTheReasoningInteractionIsMirroredNotInvented -- so the stub
    # cannot drift into standing for the thing under test (#624).
    serving.openai_serving_chat = _NoReasoning()
    return AnthropicServing._convert_to_chat_completion_request(serving, request)


class _NoReasoning:
    """Records the call and changes nothing."""

    def __init__(self):
        self.calls = []

    def apply_reasoning_enabled(self, request, enabled):
        self.calls.append(enabled)


class TestChatTemplateKwargsReachTheTemplateLayer(unittest.TestCase):
    def test_the_request_model_accepts_the_field(self):
        self.assertEqual(
            _request(chat_template_kwargs={"enable_thinking": False}).chat_template_kwargs,
            {"enable_thinking": False},
            "a declared field is the difference between passthrough and a "
            "silent drop: pydantic's default extra='ignore' discards unknowns",
        )

    def test_the_converted_request_carries_them(self):
        """THE GAP. What the client sent must reach the OpenAI request the
        adapter delegates with, or the template never sees it."""
        chat = _convert(_request(chat_template_kwargs={"custom_flag": True}))
        self.assertEqual((chat.chat_template_kwargs or {}).get("custom_flag"), True)

    def test_absent_stays_absent(self):
        """No invention: a request that did not ask must not acquire kwargs,
        or every plain request starts carrying a dict the caller never sent."""
        self.assertIsNone(_convert(_request()).chat_template_kwargs)

    def test_multiple_keys_survive_intact(self):
        chat = _convert(_request(chat_template_kwargs={"a": 1, "b": "two"}))
        self.assertEqual(chat.chat_template_kwargs, {"a": 1, "b": "two"})


class TestUsageCacheReadTokens(unittest.TestCase):
    """cached > 0 must surface; cached == 0 must report 0, never absent."""

    def test_cached_tokens_surface_as_cache_read_input_tokens(self):
        usage = _anthropic_usage_from_openai(
            _Usage(prompt_tokens=100, cached_tokens=40),
            include_input=True,
            include_output=True,
        )
        self.assertEqual(usage.cache_read_input_tokens, 40)

    def test_zero_cache_reports_zero_not_absent(self):
        """THE GAP. `if cached_tokens:` is falsy at 0, so the field was
        omitted and the client could not distinguish 'no cache hit' from
        'this server does not report cache usage'."""
        usage = _anthropic_usage_from_openai(
            _Usage(prompt_tokens=100, cached_tokens=0),
            include_input=True,
            include_output=True,
        )
        self.assertEqual(
            usage.cache_read_input_tokens,
            0,
            "zero must be reported as zero; absent-vs-0 is the ambiguity this "
            "pin exists to remove",
        )

    def test_input_tokens_still_exclude_the_cached_part(self):
        """The neighbouring arithmetic must not move: Anthropic bills
        input_tokens as the NON-cached remainder."""
        usage = _anthropic_usage_from_openai(
            _Usage(prompt_tokens=100, cached_tokens=40),
            include_input=True,
            include_output=True,
        )
        self.assertEqual(usage.input_tokens, 60)

    def test_output_only_blocks_carry_no_cache_field(self):
        """Streaming emits usage blocks that describe only the output half.
        Reporting a cache count there would attribute input accounting to a
        block that carries none."""
        usage = _anthropic_usage_from_openai(
            _Usage(cached_tokens=40),
            include_input=False,
            include_output=True,
        )
        self.assertIsNone(usage.cache_read_input_tokens)

    def test_a_backend_that_reported_nothing_stays_absent(self):
        """THE LINE THAT MATTERS, and it is not "always emit a number".

        A usage object with no prompt_tokens_details reported nothing about
        caching. Answering 0 would publish a measurement never taken and would
        silently turn "unknown" into "definitely no cache" -- the defaulted-
        measurement defect (#606). Absent is the honest answer here, and an
        existing pin (anthropic/test_serving.py:580) already required it.
        """

        class _NoDetails:
            prompt_tokens = 5
            completion_tokens = 0

        usage = _anthropic_usage_from_openai(
            _NoDetails(), include_input=True, include_output=True
        )
        self.assertEqual(usage.input_tokens, 5)
        self.assertIsNone(usage.cache_read_input_tokens)

    def test_a_missing_usage_object_is_still_answerable(self):
        usage = _anthropic_usage_from_openai(
            None, include_input=True, include_output=True
        )
        self.assertEqual(usage.input_tokens, 0)
        self.assertEqual(usage.output_tokens, 0)


class TestTheReasoningInteractionIsMirroredNotInvented(unittest.TestCase):
    """Pins the precedence so it is a decision rather than an accident.

    ``apply_reasoning_enabled`` merges into whatever the request carries, so a
    caller's unrelated keys survive while the reasoning toggle follows
    Anthropic's typed ``thinking`` semantics. Pinning the merge shape here
    means a future change to the OpenAI write side cannot silently start
    clobbering Anthropic callers' template kwargs.
    """

    def test_the_openai_write_side_still_merges(self):
        import inspect

        from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

        src = inspect.getsource(OpenAIServingChat.apply_reasoning_enabled)
        self.assertIn("dict(request.chat_template_kwargs or {})", src)
        self.assertIn("request.chat_template_kwargs = chat_template_kwargs", src)


if __name__ == "__main__":
    unittest.main()
