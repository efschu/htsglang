"""OpenAI-front mapping of the per-request thinking budget (#540).

Covers the client-visible contract: ``extra_body={"thinking_budget": N}`` on
/v1/chat/completions ends up as an enforceable budget on the sampling params,
without the client serializing a logit processor.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

import unittest

from pydantic import ValidationError

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.sampling.thinking_budget import (
    THINKING_BUDGET_INTERNAL_KEY,
    THINKING_BUDGET_KEY,
    THINKING_BUDGET_TOKEN_IDS_KEY,
    ThinkingBudgetUnsupportedError,
    attach_thinking_budget,
    internal_thinking_budget_processor_str,
)
from sglang.test.test_utils import CustomTestCase

QWEN36_THINK_START_ID = 248068
QWEN36_THINK_END_ID = 248069
QWEN36_NEWLINE_ID = 198


class _FixtureTokenizer:
    name_or_path = "Qwen3.6-27B-fixture"

    def encode(self, text, add_special_tokens=False):
        table = {
            "<think>": [QWEN36_THINK_START_ID],
            "</think>": [QWEN36_THINK_END_ID],
            "\n": [QWEN36_NEWLINE_ID],
        }
        return list(table.get(text, [ord(c) for c in text] + [0]))


def _request(**overrides):
    data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
    }
    data.update(overrides)
    return ChatCompletionRequest(**data)


class TestChatRequestThinkingBudget(CustomTestCase):
    def test_default_is_absent(self):
        params = _request().to_sampling_params(stop=[], model_generation_config={})
        self.assertIsNone(params["thinking_budget"])

    def test_budget_reaches_sampling_params(self):
        params = _request(thinking_budget=256).to_sampling_params(
            stop=[], model_generation_config={}
        )
        self.assertEqual(params["thinking_budget"], 256)
        self.assertEqual(SamplingParams(**params).thinking_budget, 256)

    def test_minus_one_normalizes_to_no_budget(self):
        self.assertIsNone(_request(thinking_budget=-1).thinking_budget)

    def test_negative_budget_is_rejected_at_the_front(self):
        with self.assertRaises(ValidationError):
            _request(thinking_budget=-5)

    def test_float_budget_is_rejected_at_the_front(self):
        with self.assertRaises(ValidationError):
            _request(thinking_budget=1.5)

    def test_end_to_end_mapping_attaches_builtin_processor(self):
        """Request field -> sampling params -> attached processor.

        This is the whole client contract: no custom_logit_processor in the
        request body and no --enable-custom-logit-processor on the server.
        """
        request = _request(thinking_budget=256)
        sampling_params = SamplingParams(
            **request.to_sampling_params(stop=[], model_generation_config={})
        )
        processor = attach_thinking_budget(
            sampling_params,
            request.custom_logit_processor,
            _FixtureTokenizer(),
            "qwen3",
            "Qwen3.6-27B",
        )
        self.assertEqual(processor, internal_thinking_budget_processor_str())
        self.assertEqual(sampling_params.custom_params[THINKING_BUDGET_KEY], 256)
        self.assertEqual(
            sampling_params.custom_params[THINKING_BUDGET_TOKEN_IDS_KEY],
            [QWEN36_THINK_START_ID, QWEN36_THINK_END_ID, QWEN36_NEWLINE_ID],
        )
        self.assertIs(sampling_params.custom_params[THINKING_BUDGET_INTERNAL_KEY], True)

    def test_budget_without_reasoning_parser_is_refused(self):
        request = _request(thinking_budget=256)
        sampling_params = SamplingParams(
            **request.to_sampling_params(stop=[], model_generation_config={})
        )
        with self.assertRaises(ThinkingBudgetUnsupportedError):
            attach_thinking_budget(
                sampling_params,
                request.custom_logit_processor,
                _FixtureTokenizer(),
                None,
                "Qwen3.6-27B",
            )


if __name__ == "__main__":
    unittest.main()
