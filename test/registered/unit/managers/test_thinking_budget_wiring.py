"""Request-path wiring for the thinking budget (#540).

Executes the real tokenizer-manager step, the real Req construction and the
real batch-info gate, so the pieces are pinned where they actually bind:
a first-class ``thinking_budget`` must arrive at the scheduler as an attached
built-in processor, and must run without --enable-custom-logit-processor.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.sampling.sampling_batch_info import uses_custom_logit_processor
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.sampling.thinking_budget import (
    THINKING_BUDGET_INTERNAL_KEY,
    THINKING_BUDGET_KEY,
    THINKING_BUDGET_TOKEN_IDS_KEY,
    ThinkingBudgetUnsupportedError,
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


def _tokenize(sampling_params, reasoning_parser="qwen3", custom_logit_processor=None):
    """Run the real TokenizerManager._create_tokenized_object on a stub self."""
    obj = GenerateReqInput(
        text="hi",
        sampling_params=sampling_params,
        custom_logit_processor=custom_logit_processor,
    )
    obj.normalize_batch_and_arguments()
    stub = SimpleNamespace(
        preferred_sampling_params=None,
        sampling_params_class=SamplingParams,
        tokenizer=_FixtureTokenizer(),
        model_config=SimpleNamespace(vocab_size=250000),
        server_args=SimpleNamespace(
            reasoning_parser=reasoning_parser,
            model_path="Qwen3.6-27B",
            disaggregation_transfer_backend="none",
        ),
        fake_bootstrap_room_counter=0,
        rid_to_state={obj.rid: SimpleNamespace(time_stats=MagicMock())},
    )
    return TokenizerManager._create_tokenized_object(stub, obj, "hi", [1, 2, 3])


class TestThinkingBudgetRequestPath(CustomTestCase):
    def test_budget_request_arrives_with_builtin_processor(self):
        tokenized = _tokenize({"thinking_budget": 64})
        self.assertEqual(
            tokenized.custom_logit_processor, internal_thinking_budget_processor_str()
        )
        self.assertEqual(
            tokenized.sampling_params.custom_params,
            {
                THINKING_BUDGET_KEY: 64,
                THINKING_BUDGET_TOKEN_IDS_KEY: [
                    QWEN36_THINK_START_ID,
                    QWEN36_THINK_END_ID,
                    QWEN36_NEWLINE_ID,
                ],
                THINKING_BUDGET_INTERNAL_KEY: True,
            },
        )

    def test_request_without_budget_is_unchanged(self):
        tokenized = _tokenize({"max_new_tokens": 8})
        self.assertIsNone(tokenized.custom_logit_processor)
        self.assertIsNone(tokenized.sampling_params.custom_params)

    def test_budget_without_reasoning_parser_is_refused(self):
        with self.assertRaises(ThinkingBudgetUnsupportedError):
            _tokenize({"thinking_budget": 64}, reasoning_parser=None)

    def test_req_carries_the_internal_flag(self):
        tokenized = _tokenize({"thinking_budget": 64})
        req = Req(
            tokenized.rid,
            "hi",
            array("q", [1, 2, 3]),
            tokenized.sampling_params,
            custom_logit_processor=tokenized.custom_logit_processor,
        )
        self.assertTrue(req.custom_logit_processor_internal)
        # ... and the batch-info gate lets it run with the server flag off.
        self.assertTrue(uses_custom_logit_processor(req, False))

    def test_client_processor_stays_gated_by_the_server_flag(self):
        req = Req(
            "rid-1",
            "hi",
            array("q", [1, 2, 3]),
            SamplingParams(custom_params={THINKING_BUDGET_KEY: 64}),
            custom_logit_processor="client-processor-str",
        )
        self.assertFalse(req.custom_logit_processor_internal)
        self.assertFalse(uses_custom_logit_processor(req, False))
        self.assertTrue(uses_custom_logit_processor(req, True))

    def test_request_without_processor_never_passes_the_gate(self):
        req = Req("rid-2", "hi", array("q", [1, 2, 3]), SamplingParams())
        self.assertFalse(uses_custom_logit_processor(req, True))


if __name__ == "__main__":
    unittest.main()
