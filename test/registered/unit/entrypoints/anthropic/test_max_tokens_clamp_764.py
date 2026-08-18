"""#764: an Anthropic ``max_tokens`` ceiling above the context length must be
clamped, not rejected.

The failing shape in the field: claude-cli sends ``max_tokens: 32000`` on every
turn (a fixed per-model-family budget it cannot tune per server), the local
server runs with a smaller ``--context-length``, and the request came back as
``400 max_completion_tokens is too large``. ``max_tokens`` is a stop condition,
so the request is satisfiable — with a lower ceiling.

Three layers are exercised, because the reject was spread across two gates and
the opt-in across a third:

* the Anthropic front marks its requests as carrying a ceiling,
* ``OpenAIServingChat._validate_request`` stops rejecting those,
* ``TokenizerManager._validate_one_request`` does the actual shrink, where the
  real input length is known.

Every case also asserts the DEFAULT (no opt-in) path still rejects, so the
change cannot silently widen behaviour for callers that never asked.

No GPU, no model load: run with CUDA_VISIBLE_DEVICES=99.
"""

import unittest
from types import SimpleNamespace

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede imports that may pull in sgl_kernel

from sglang.srt.entrypoints.anthropic.protocol import (  # noqa: E402
    AnthropicMessage,
    AnthropicMessagesRequest,
)
from sglang.srt.entrypoints.anthropic.serving import AnthropicServing  # noqa: E402
from sglang.srt.entrypoints.openai.protocol import (  # noqa: E402
    ChatCompletionRequest,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat  # noqa: E402
from sglang.srt.managers.io_struct import GenerateReqInput  # noqa: E402
from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

CONTEXT_LEN = 8192


def _chat_handler(context_length=CONTEXT_LEN, allow_auto_truncate=False):
    """An OpenAIServingChat with only the surface ``_validate_request`` touches."""
    handler = OpenAIServingChat.__new__(OpenAIServingChat)
    handler.tokenizer_manager = SimpleNamespace(
        server_args=SimpleNamespace(
            context_length=context_length,
            allow_auto_truncate=allow_auto_truncate,
        )
    )
    return handler


def _chat_request(max_tokens, clamp):
    return ChatCompletionRequest(
        model="local-model",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=max_tokens,
        clamp_max_tokens=clamp,
    )


def _tokenizer_manager(context_len=CONTEXT_LEN, allow_auto_truncate=False):
    """A TokenizerManager with only the fields ``_validate_one_request`` reads."""
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.context_len = context_len
    tm.allow_auto_truncate = allow_auto_truncate
    tm.validate_total_tokens = True
    tm.num_reserved_tokens = 0
    tm.is_generation = True
    return tm


def _generate_req(max_new_tokens, clamp):
    return GenerateReqInput(
        text="hi",
        sampling_params={"max_new_tokens": max_new_tokens},
        clamp_max_new_tokens=clamp,
    )


class TestAnthropicFrontDeclaresCeiling(unittest.TestCase):
    """The front is what opts in; nothing else in the request may have to."""

    def _convert(self, max_tokens):
        front = AnthropicServing(
            SimpleNamespace(
                tokenizer_manager=SimpleNamespace(
                    tokenizer=SimpleNamespace(chat_template=None)
                ),
                apply_reasoning_enabled=lambda chat_request, enabled: None,
                wrap_reasoning_history=lambda text: text,
            )
        )
        request = AnthropicMessagesRequest(
            model="local-model",
            max_tokens=max_tokens,
            messages=[AnthropicMessage(role="user", content="hi")],
        )
        return front._convert_to_chat_completion_request(request)

    def test_converted_request_carries_the_ceiling_flag(self):
        chat_request = self._convert(32000)
        self.assertTrue(chat_request.clamp_max_tokens)
        # The requested value is forwarded verbatim -- the front does not
        # rewrite what the caller asked for, it only labels it a ceiling.
        self.assertEqual(chat_request.max_tokens, 32000)

    def test_flag_is_set_regardless_of_size(self):
        # Nothing here may depend on the current context length: the front does
        # not know it, and a value that fits today may not fit after a reboot
        # with a different --context-length.
        self.assertTrue(self._convert(64).clamp_max_tokens)


class TestValidateRequestGate(unittest.TestCase):
    """Gate 1: the 400 that produced the reported error string."""

    def test_default_path_still_rejects_an_oversized_value(self):
        error = _chat_handler()._validate_request(_chat_request(32000, clamp=False))
        self.assertIsNotNone(error)
        self.assertIn("max_completion_tokens is too large", error)

    def test_ceiling_request_is_accepted(self):
        self.assertIsNone(
            _chat_handler()._validate_request(_chat_request(32000, clamp=True))
        )

    def test_boundary_exactly_context_length_is_accepted_either_way(self):
        for clamp in (False, True):
            with self.subTest(clamp=clamp):
                self.assertIsNone(
                    _chat_handler()._validate_request(
                        _chat_request(CONTEXT_LEN, clamp=clamp)
                    )
                )

    def test_boundary_one_over_context_length(self):
        self.assertIsNotNone(
            _chat_handler()._validate_request(_chat_request(CONTEXT_LEN + 1, False))
        )
        self.assertIsNone(
            _chat_handler()._validate_request(_chat_request(CONTEXT_LEN + 1, True))
        )


class TestTokenizerManagerClamp(unittest.TestCase):
    """Gate 2: the shrink itself, where the input length is known."""

    def test_default_path_still_raises_on_overflow(self):
        tm = _tokenizer_manager()
        obj = _generate_req(32000, clamp=False)
        with self.assertRaises(ValueError) as caught:
            tm._validate_one_request(obj, [1] * 100)
        self.assertIn("maximum context length", str(caught.exception))

    def test_ceiling_request_is_shrunk_to_what_is_left(self):
        tm = _tokenizer_manager()
        obj = _generate_req(32000, clamp=True)
        tm._validate_one_request(obj, [1] * 100)
        self.assertEqual(obj.sampling_params["max_new_tokens"], CONTEXT_LEN - 100)

    def test_reserved_tokens_are_counted_against_the_budget(self):
        tm = _tokenizer_manager()
        tm.num_reserved_tokens = 7
        obj = _generate_req(32000, clamp=True)
        tm._validate_one_request(obj, [1] * 100)
        self.assertEqual(obj.sampling_params["max_new_tokens"], CONTEXT_LEN - 107)

    def test_a_value_that_fits_is_left_untouched(self):
        tm = _tokenizer_manager()
        obj = _generate_req(CONTEXT_LEN - 100, clamp=True)
        tm._validate_one_request(obj, [1] * 100)
        self.assertEqual(obj.sampling_params["max_new_tokens"], CONTEXT_LEN - 100)

    def test_input_is_never_truncated_by_the_ceiling_opt_in(self):
        # An input that alone exceeds the context is a real error and stays one:
        # clamping a ceiling must not become a licence to drop prompt tokens.
        tm = _tokenizer_manager()
        input_ids = [1] * (CONTEXT_LEN + 10)
        obj = _generate_req(32000, clamp=True)
        with self.assertRaises(ValueError) as caught:
            tm._validate_one_request(obj, input_ids)
        self.assertIn("longer than the", str(caught.exception))
        self.assertEqual(len(input_ids), CONTEXT_LEN + 10)


class TestPlumbing(unittest.TestCase):
    def test_generate_req_input_defaults_to_no_clamp(self):
        self.assertFalse(GenerateReqInput(text="hi").clamp_max_new_tokens)

    def test_batch_items_inherit_the_ceiling(self):
        # A batch is split into per-item requests before validation, and it is
        # the item that reaches the gate. A flag dropped here would be a
        # silently ignored opt-in for every batched caller.
        batch = GenerateReqInput(
            text=["a", "b"],
            sampling_params={"max_new_tokens": 32000},
            clamp_max_new_tokens=True,
        )
        batch.normalize_batch_and_arguments()
        self.assertTrue(batch[0].clamp_max_new_tokens)
        self.assertTrue(batch[1].clamp_max_new_tokens)

    def test_chat_completion_request_defaults_to_no_clamp(self):
        self.assertFalse(
            ChatCompletionRequest(
                model="m", messages=[{"role": "user", "content": "hi"}]
            ).clamp_max_tokens
        )


if __name__ == "__main__":
    unittest.main()
