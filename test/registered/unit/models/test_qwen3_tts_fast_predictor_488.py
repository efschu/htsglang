# SPDX-License-Identifier: Apache-2.0
"""#488 graphs cut slice 1: the two things that can be proven without a GPU.

1. **The sampler is equivalent to transformers' own warpers.** Not "written to
   match" -- driven against the real ``TemperatureLogitsWarper``,
   ``TopKLogitsWarper`` and ``TopPLogitsWarper`` on fixed logits, including the
   three no-op cases (temperature 1.0, top_k 0, top_p 1.0) where a mismatch
   would silently make a DEFAULT config take a different path in the two
   implementations. The talker's own defaults are top_k=50, top_p=1.0,
   temperature=0.9 (`qwen3_tts_model.py:325-328`), i.e. exactly one of the
   no-op cases is live in production.

2. **The step schedule.** Prefill takes ``lm_head[0]`` and no codebook
   embedding; decode step g takes ``codec_embedding[g-1]`` and ``lm_head[g]``.
   An off-by-one picks a neighbouring codebook head and does not crash -- the
   audio just degrades in timbre, which is the hardest signal to debug.

End-to-end greedy code-token identity against the reference ``generate`` needs
the real checkpoint and the tenant's import shims; it runs in
``scripts/dev/488_talker_profile/validate_fast_predictor.py`` and is reported
with the GPU measurement. It is named here so the gap is visible from the test
file rather than only from the commit message.
"""

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "99")

import torch

from sglang.srt.models.qwen3_tts_fast_predictor import (
    apply_warpers,
    step_schedule,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _reference_warpers(logits, temperature, top_k, top_p):
    """transformers' own classes, in transformers' own order."""
    from transformers.generation.logits_process import (
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )

    out = logits.clone()
    ids = torch.zeros((logits.shape[0], 1), dtype=torch.long)
    if temperature is not None and temperature != 1.0:
        out = TemperatureLogitsWarper(temperature)(ids, out)
    if top_k:
        out = TopKLogitsWarper(top_k)(ids, out)
    if top_p is not None and top_p < 1.0:
        out = TopPLogitsWarper(top_p)(ids, out)
    return out


class TestSamplerEquivalence(CustomTestCase):
    """The sampler rewrite is the main correctness surface of slice 1."""

    def setUp(self):
        torch.manual_seed(20260804)
        # 2048 is the code predictor's real vocabulary.
        self.logits = torch.randn(2, 2048, dtype=torch.float32)

    def _assert_same(self, temperature, top_k, top_p):
        mine = apply_warpers(self.logits, temperature, top_k, top_p)
        theirs = _reference_warpers(self.logits, temperature, top_k, top_p)
        # -inf must land in exactly the same places, or a different token set
        # is reachable even when the surviving values agree.
        self.assertTrue(
            torch.equal(torch.isneginf(mine), torch.isneginf(theirs)),
            f"mask differs for temperature={temperature} top_k={top_k} "
            f"top_p={top_p}",
        )
        finite = ~torch.isneginf(theirs)
        torch.testing.assert_close(mine[finite], theirs[finite])

    def test_production_defaults(self):
        """top_k=50, top_p=1.0, temperature=0.9 -- what the talker ships."""
        self._assert_same(0.9, 50, 1.0)

    def test_temperature_only(self):
        self._assert_same(0.7, 0, 1.0)

    def test_top_p_only(self):
        self._assert_same(1.0, 0, 0.9)

    def test_top_k_and_top_p_together(self):
        self._assert_same(0.8, 50, 0.92)

    def test_aggressive_top_p(self):
        self._assert_same(1.0, 0, 0.1)

    def test_every_no_op_is_a_no_op(self):
        """A skip that transformers takes and we do not (or vice versa) is a
        silent divergence on a default config."""
        untouched = apply_warpers(self.logits, 1.0, 0, 1.0)
        torch.testing.assert_close(untouched, self.logits)

    def test_top_p_keeps_at_least_one_token(self):
        """CAN-FAIL-SHAPED: with a brutal top_p the mask must not empty a row,
        which would make multinomial raise on a distribution of all zeros."""
        out = apply_warpers(self.logits, 1.0, 0, 1e-6)
        survivors = (~torch.isneginf(out)).sum(dim=-1)
        self.assertTrue(torch.all(survivors >= 1), survivors)

    def test_temperature_and_top_k_genuinely_commute(self):
        """Established, not assumed: temperature is a monotonic scaling, so it
        cannot change WHICH tokens top-k keeps, and scaling the survivors
        after masking gives the same tensor. Worth pinning because it means
        an ordering bug between THESE two is unobservable -- which is why the
        detectability arm below uses top_p instead."""
        after = apply_warpers(self.logits, None, 50, None) / 0.9
        before = apply_warpers(self.logits, 0.9, 50, 1.0)
        torch.testing.assert_close(after, before)

    def test_a_wrong_order_against_top_p_is_detectable(self):
        """The instrument must be able to SEE an ordering bug, or the
        agreement tests above prove nothing. Temperature and top_p do NOT
        commute: top_p thresholds the softmax distribution, and temperature
        reshapes exactly that distribution."""
        correct = apply_warpers(self.logits, 0.7, 0, 0.9)
        # top_p first, then temperature -- the transposition a careless
        # rewrite makes.
        wrong = apply_warpers(self.logits, None, 0, 0.9) / 0.7
        self.assertFalse(
            torch.equal(torch.isneginf(wrong), torch.isneginf(correct)),
            "temperature/top_p transposition produced an identical mask; the "
            "comparator could not detect an ordering bug",
        )


class TestStepSchedule(CustomTestCase):
    def test_sixteen_groups_give_fifteen_steps(self):
        schedule = step_schedule(16)
        self.assertEqual(len(schedule), 15)

    def test_prefill_uses_head_zero_and_no_embedding(self):
        step, embedding, head = step_schedule(16)[0]
        self.assertEqual(step, 0)
        self.assertIsNone(embedding)
        self.assertEqual(head, 0)

    def test_decode_steps_lag_the_embedding_by_one(self):
        """`codec_embedding[g-1]` with `lm_head[g]` -- the off-by-one that
        picks a neighbouring codebook and only shows up as timbre."""
        for step, embedding, head in step_schedule(16)[1:]:
            self.assertEqual(head, step)
            self.assertEqual(embedding, step - 1)

    def test_indices_stay_inside_the_fifteen_slot_module_lists(self):
        """The checkpoint has 15 heads and 15 embeddings for 16 groups."""
        schedule = step_schedule(16)
        heads = [h for _, _, h in schedule]
        embeddings = [e for _, e, _ in schedule if e is not None]
        self.assertEqual(max(heads), 14)
        self.assertEqual(max(embeddings), 13)
        self.assertEqual(min(heads), 0)
        self.assertEqual(min(embeddings), 0)

    def test_every_head_is_used_exactly_once(self):
        heads = [h for _, _, h in step_schedule(16)]
        self.assertEqual(sorted(heads), list(range(15)))


if __name__ == "__main__":
    unittest.main()
