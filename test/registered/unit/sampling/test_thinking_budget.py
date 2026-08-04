"""Unit tests for the per-request thinking budget (#540).

No server, no model loading, no GPU. The tokenizer is a minimal fixture whose
ids are taken from the real Qwen3.6 checkpoint tokenizer, so the mismatch with
the legacy hardcoded ids (151667/151668) is reproduced exactly as reported in
sgl-project/sglang#25536.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from array import array
from unittest.mock import MagicMock

import torch

from sglang.srt.sampling.custom_logit_processor import (
    Qwen3ThinkingBudgetLogitProcessor,
    ThinkingBudgetLogitProcessor,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.sampling.thinking_budget import (
    THINKING_BUDGET_INTERNAL_KEY,
    THINKING_BUDGET_KEY,
    THINKING_BUDGET_TOKEN_IDS_KEY,
    ThinkingBudgetUnsupportedError,
    attach_thinking_budget,
    internal_thinking_budget_processor_str,
    resolve_thinking_budget_token_ids,
    validate_thinking_budget,
)
from sglang.test.test_utils import CustomTestCase

# Minimal tokenizer fixture. The ids are the ones the real Qwen3.6-27B
# tokenizer produces (verified against
# models-cache/Qwen3.6-27B-INT8-W8A8/tokenizer.json); the test does not read
# that path at runtime.
QWEN36_THINK_START_ID = 248068
QWEN36_THINK_END_ID = 248069
QWEN36_NEWLINE_ID = 198

# The ids Qwen3ThinkingBudgetLogitProcessor hardcodes. They belong to an older
# Qwen3 vocabulary and do not exist as thinking markers in Qwen3.6.
LEGACY_QWEN3_START_ID = 151667
LEGACY_QWEN3_END_ID = 151668


class _FixtureTokenizer:
    """Encodes only what the derivation needs; everything else is multi-token."""

    name_or_path = "Qwen3.6-27B-fixture"

    def __init__(self, table=None):
        self.table = (
            table
            if table is not None
            else {
                "<think>": [QWEN36_THINK_START_ID],
                "</think>": [QWEN36_THINK_END_ID],
                "\n": [QWEN36_NEWLINE_ID],
            }
        )

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        if text in self.table:
            return list(self.table[text])
        # Stand-in for "this string is not an atomic token in this
        # vocabulary": always more than one id, whatever its length.
        return [ord(c) for c in text] + [0]


def _make_req(origin_input_ids=None, output_ids=None):
    req = MagicMock()
    req.origin_input_ids = array("q", origin_input_ids or [])
    req.output_ids = array("q", output_ids or [])
    return req


def _derived_params(budget, req, ids=None):
    return {
        THINKING_BUDGET_KEY: budget,
        THINKING_BUDGET_TOKEN_IDS_KEY: ids
        or [QWEN36_THINK_START_ID, QWEN36_THINK_END_ID, QWEN36_NEWLINE_ID],
        "__req__": req,
    }


class TestValidateThinkingBudget(CustomTestCase):
    def test_none_and_minus_one_mean_no_budget(self):
        self.assertIsNone(validate_thinking_budget(None))
        self.assertIsNone(validate_thinking_budget(-1))

    def test_non_negative_int_passes_through(self):
        self.assertEqual(validate_thinking_budget(0), 0)
        self.assertEqual(validate_thinking_budget(1024), 1024)

    def test_bool_is_rejected(self):
        # bool is an int subclass; True must not become a 1-token budget.
        with self.assertRaises(ValueError):
            validate_thinking_budget(True)

    def test_float_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_thinking_budget(12.5)

    def test_negative_other_than_minus_one_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_thinking_budget(-7)

    def test_sampling_params_verify_rejects_malformed_budget(self):
        params = SamplingParams(thinking_budget=-7)
        with self.assertRaises(ValueError):
            params.verify(vocab_size=1000)

    def test_sampling_params_verify_accepts_budget(self):
        params = SamplingParams(thinking_budget=64)
        params.verify(vocab_size=1000)
        self.assertEqual(params.thinking_budget, 64)


class TestResolveThinkingBudgetTokenIds(CustomTestCase):
    def test_ids_come_from_the_tokenizer_not_from_class_constants(self):
        ids = resolve_thinking_budget_token_ids(
            _FixtureTokenizer(), "qwen3", "Qwen3.6-27B"
        )
        self.assertEqual(ids.start, QWEN36_THINK_START_ID)
        self.assertEqual(ids.end, QWEN36_THINK_END_ID)
        self.assertEqual(ids.newline, QWEN36_NEWLINE_ID)
        self.assertEqual(ids.start_str, "<think>")
        self.assertEqual(ids.end_str, "</think>")
        # The regression this feature exists for: the hardcoded constants are
        # NOT what this checkpoint uses.
        self.assertNotEqual(ids.start, LEGACY_QWEN3_START_ID)
        self.assertNotEqual(ids.end, LEGACY_QWEN3_END_ID)

    def test_missing_reasoning_parser_raises(self):
        with self.assertRaises(ThinkingBudgetUnsupportedError) as ctx:
            resolve_thinking_budget_token_ids(_FixtureTokenizer(), None, "some-model")
        self.assertIn("--reasoning-parser", str(ctx.exception))
        self.assertIn("some-model", str(ctx.exception))

    def test_unknown_reasoning_parser_raises(self):
        with self.assertRaises(ThinkingBudgetUnsupportedError):
            resolve_thinking_budget_token_ids(
                _FixtureTokenizer(), "no-such-parser", "some-model"
            )

    def test_missing_tokenizer_raises(self):
        with self.assertRaises(ThinkingBudgetUnsupportedError):
            resolve_thinking_budget_token_ids(None, "qwen3", "some-model")

    def test_multi_token_marker_raises_and_names_markers(self):
        """A checkpoint whose markers are not atomic tokens must be refused.

        This is the loud-failure proof: the old code path would have kept the
        hardcoded id, never matched it, and returned an unbudgeted answer.
        """
        tokenizer = _FixtureTokenizer(table={"\n": [QWEN36_NEWLINE_ID]})
        with self.assertRaises(ThinkingBudgetUnsupportedError) as ctx:
            resolve_thinking_budget_token_ids(tokenizer, "qwen3", "broken-model")
        message = str(ctx.exception)
        self.assertIn("broken-model", message)
        self.assertIn("<think>", message)
        self.assertIn("</think>", message)
        self.assertIn("not a single token", message)

    def test_tokenizer_without_single_token_newline_still_resolves(self):
        tokenizer = _FixtureTokenizer(
            table={
                "<think>": [QWEN36_THINK_START_ID],
                "</think>": [QWEN36_THINK_END_ID],
            }
        )
        ids = resolve_thinking_budget_token_ids(tokenizer, "qwen3", "no-newline-model")
        self.assertIsNone(ids.newline)


class TestThinkingBudgetProcessorWithDerivedIds(CustomTestCase):
    VOCAB = 250000

    def setUp(self):
        self.processor = ThinkingBudgetLogitProcessor()

    def _logits(self, batch_size=1):
        return torch.zeros(batch_size, self.VOCAB)

    def test_budget_exceeded_forces_newline_then_end(self):
        req = _make_req(origin_input_ids=[QWEN36_THINK_START_ID], output_ids=[7] * 5)
        out = self.processor(self._logits(), [_derived_params(5, req)])
        self.assertEqual(int(out[0].argmax()), QWEN36_NEWLINE_ID)

        req = _make_req(
            origin_input_ids=[QWEN36_THINK_START_ID],
            output_ids=[7] * 4 + [QWEN36_NEWLINE_ID],
        )
        out = self.processor(self._logits(), [_derived_params(5, req)])
        self.assertEqual(int(out[0].argmax()), QWEN36_THINK_END_ID)

    def test_budget_zero_closes_immediately(self):
        req = _make_req(origin_input_ids=[QWEN36_THINK_START_ID], output_ids=[])
        out = self.processor(self._logits(), [_derived_params(0, req)])
        self.assertEqual(int(out[0].argmax()), QWEN36_NEWLINE_ID)

    def test_within_budget_is_untouched(self):
        req = _make_req(origin_input_ids=[QWEN36_THINK_START_ID], output_ids=[7, 8])
        out = self.processor(self._logits(), [_derived_params(10, req)])
        self.assertTrue(torch.all(out == 0.0))

    def test_already_closed_thinking_is_untouched(self):
        req = _make_req(
            origin_input_ids=[QWEN36_THINK_START_ID],
            output_ids=[7] * 5 + [QWEN36_THINK_END_ID] + [9] * 20,
        )
        out = self.processor(self._logits(), [_derived_params(5, req)])
        self.assertTrue(torch.all(out == 0.0))

    def test_thinking_never_started_is_untouched(self):
        req = _make_req(origin_input_ids=[1, 2, 3], output_ids=[7] * 50)
        out = self.processor(self._logits(), [_derived_params(5, req)])
        self.assertTrue(torch.all(out == 0.0))

    def test_budget_none_is_untouched(self):
        req = _make_req(origin_input_ids=[QWEN36_THINK_START_ID], output_ids=[7] * 50)
        out = self.processor(self._logits(), [_derived_params(None, req)])
        self.assertTrue(torch.all(out == 0.0))

    def test_derived_ids_win_over_class_constants(self):
        """The Qwen3 subclass must follow the request's derived ids."""
        processor = Qwen3ThinkingBudgetLogitProcessor()
        req = _make_req(origin_input_ids=[QWEN36_THINK_START_ID], output_ids=[7] * 5)
        out = processor(torch.zeros(1, self.VOCAB), [_derived_params(5, req)])
        self.assertEqual(int(out[0].argmax()), QWEN36_NEWLINE_ID)

    def test_legacy_hardcoded_ids_would_silently_do_nothing(self):
        """Documents the defect: hardcoded ids against a Qwen3.6 sequence.

        Without derived ids the processor never sees its start marker and the
        budget is ignored. This is why the id must come from the tokenizer and
        why an unresolvable budget is refused at admission.
        """
        processor = Qwen3ThinkingBudgetLogitProcessor()
        req = _make_req(origin_input_ids=[QWEN36_THINK_START_ID], output_ids=[7] * 50)
        out = processor(
            torch.zeros(1, self.VOCAB),
            [{THINKING_BUDGET_KEY: 5, "__req__": req}],
        )
        self.assertTrue(torch.all(out == 0.0))

    def test_no_ids_available_raises_instead_of_skipping(self):
        req = _make_req(origin_input_ids=[QWEN36_THINK_START_ID], output_ids=[7] * 5)
        with self.assertRaises(ThinkingBudgetUnsupportedError):
            self.processor(self._logits(), [{THINKING_BUDGET_KEY: 5, "__req__": req}])

    def test_malformed_budget_raises_instead_of_skipping(self):
        req = _make_req(origin_input_ids=[QWEN36_THINK_START_ID], output_ids=[7] * 5)
        with self.assertRaises(ValueError):
            self.processor(self._logits(), [_derived_params(-5, req)])

    def test_per_row_isolation_in_a_batch(self):
        over = _make_req(origin_input_ids=[QWEN36_THINK_START_ID], output_ids=[7] * 5)
        under = _make_req(origin_input_ids=[QWEN36_THINK_START_ID], output_ids=[7])
        out = self.processor(
            self._logits(batch_size=2),
            [_derived_params(5, over), _derived_params(5, under)],
        )
        self.assertEqual(int(out[0].argmax()), QWEN36_NEWLINE_ID)
        self.assertTrue(torch.all(out[1] == 0.0))


class _FakeSamplingBatchInfo:
    """Just enough of SamplingBatchInfo for apply_custom_logit_processor."""

    def __init__(self, custom_params, processor):
        self.custom_params = custom_params
        mask = torch.ones(len(custom_params), dtype=torch.bool)
        self.custom_logit_processor = {0: (processor, mask)}

    def __len__(self):
        return len(self.custom_params)


class TestSpecDecodeRowAlignment(CustomTestCase):
    """Verify rows are request-major with draft_token_num rows per request.

    EAGLE/MTP verify applies the processors over that flattened layout, so the
    params must be repeated per row -- otherwise only the first row of each
    request would be capped and the budget would leak through the draft.
    """

    VOCAB = 250000
    DRAFT_TOKEN_NUM = 3

    def test_every_row_of_an_over_budget_request_is_forced(self):
        from sglang.srt.layers.sampler import apply_custom_logit_processor

        over = _make_req(origin_input_ids=[QWEN36_THINK_START_ID], output_ids=[7] * 5)
        under = _make_req(origin_input_ids=[QWEN36_THINK_START_ID], output_ids=[7])
        info = _FakeSamplingBatchInfo(
            [_derived_params(5, over), _derived_params(5, under)],
            ThinkingBudgetLogitProcessor(),
        )
        logits = torch.zeros(2 * self.DRAFT_TOKEN_NUM, self.VOCAB)
        apply_custom_logit_processor(
            logits, info, num_tokens_in_batch=self.DRAFT_TOKEN_NUM
        )
        for row in range(self.DRAFT_TOKEN_NUM):
            self.assertEqual(int(logits[row].argmax()), QWEN36_NEWLINE_ID)
        for row in range(self.DRAFT_TOKEN_NUM, 2 * self.DRAFT_TOKEN_NUM):
            self.assertTrue(torch.all(logits[row] == 0.0))


class TestAttachThinkingBudget(CustomTestCase):
    def setUp(self):
        self.tokenizer = _FixtureTokenizer()

    def test_first_class_budget_attaches_builtin_processor(self):
        params = SamplingParams(thinking_budget=128)
        processor = attach_thinking_budget(
            params, None, self.tokenizer, "qwen3", "Qwen3.6-27B"
        )
        self.assertEqual(processor, internal_thinking_budget_processor_str())
        self.assertEqual(params.custom_params[THINKING_BUDGET_KEY], 128)
        self.assertEqual(
            params.custom_params[THINKING_BUDGET_TOKEN_IDS_KEY],
            [QWEN36_THINK_START_ID, QWEN36_THINK_END_ID, QWEN36_NEWLINE_ID],
        )
        self.assertIs(params.custom_params[THINKING_BUDGET_INTERNAL_KEY], True)

    def test_no_budget_is_a_no_op(self):
        params = SamplingParams()
        processor = attach_thinking_budget(
            params, None, self.tokenizer, "qwen3", "Qwen3.6-27B"
        )
        self.assertIsNone(processor)
        self.assertIsNone(params.custom_params)

    def test_minus_one_disables_the_budget(self):
        params = SamplingParams(thinking_budget=-1)
        processor = attach_thinking_budget(
            params, None, self.tokenizer, "qwen3", "Qwen3.6-27B"
        )
        self.assertIsNone(processor)
        self.assertIsNone(params.thinking_budget)

    def test_legacy_custom_params_form_gets_derived_ids(self):
        """The documented custom_params form is repaired, not bypassed."""
        params = SamplingParams(custom_params={THINKING_BUDGET_KEY: 64})
        processor = attach_thinking_budget(
            params, "client-processor-str", self.tokenizer, "qwen3", "Qwen3.6-27B"
        )
        self.assertEqual(processor, "client-processor-str")
        self.assertEqual(
            params.custom_params[THINKING_BUDGET_TOKEN_IDS_KEY],
            [QWEN36_THINK_START_ID, QWEN36_THINK_END_ID, QWEN36_NEWLINE_ID],
        )
        self.assertNotIn(THINKING_BUDGET_INTERNAL_KEY, params.custom_params)

    def test_legacy_custom_params_without_processor_gets_builtin(self):
        params = SamplingParams(custom_params={THINKING_BUDGET_KEY: 64})
        processor = attach_thinking_budget(
            params, None, self.tokenizer, "qwen3", "Qwen3.6-27B"
        )
        self.assertEqual(processor, internal_thinking_budget_processor_str())
        self.assertIs(params.custom_params[THINKING_BUDGET_INTERNAL_KEY], True)

    def test_first_class_budget_with_client_processor_is_a_conflict(self):
        params = SamplingParams(thinking_budget=128)
        with self.assertRaises(ValueError) as ctx:
            attach_thinking_budget(
                params, "client-processor-str", self.tokenizer, "qwen3", "m"
            )
        self.assertIn("custom_logit_processor", str(ctx.exception))

    def test_unenforceable_budget_raises(self):
        params = SamplingParams(thinking_budget=128)
        with self.assertRaises(ThinkingBudgetUnsupportedError):
            attach_thinking_budget(params, None, self.tokenizer, None, "m")

    def test_client_supplied_server_owned_keys_are_stripped(self):
        params = SamplingParams(
            custom_params={
                THINKING_BUDGET_INTERNAL_KEY: True,
                THINKING_BUDGET_TOKEN_IDS_KEY: [1, 2, 3],
                "mine": 1,
            }
        )
        processor = attach_thinking_budget(
            params, None, self.tokenizer, "qwen3", "Qwen3.6-27B"
        )
        self.assertIsNone(processor)
        self.assertEqual(params.custom_params, {"mine": 1})

    def test_client_cannot_forge_the_internal_marker(self):
        params = SamplingParams(
            thinking_budget=32,
            custom_params={THINKING_BUDGET_TOKEN_IDS_KEY: [1, 2, 3]},
        )
        attach_thinking_budget(params, None, self.tokenizer, "qwen3", "Qwen3.6-27B")
        self.assertEqual(
            params.custom_params[THINKING_BUDGET_TOKEN_IDS_KEY],
            [QWEN36_THINK_START_ID, QWEN36_THINK_END_ID, QWEN36_NEWLINE_ID],
        )


if __name__ == "__main__":
    unittest.main()
