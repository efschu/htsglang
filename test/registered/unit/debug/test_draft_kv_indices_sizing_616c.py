"""Tests for draft_kv_indices_buffer_width and draft_kv_indices_used_len.

Pure-Python, hermetic unit tests -- no CUDA, no torch.
"""

import pytest

from sglang.srt.speculative.spec_utils import (
    draft_kv_indices_buffer_width,
    draft_kv_indices_used_len,
)


class TestDraftKvIndicesUsedLen:
    """draft_kv_indices_used_len(seq_lens_sum, topk, bs, num_steps)"""

    def test_hand_computed_case(self):
        """seq_lens_sum=100, topk=2, bs=8, num_steps=3 -> 224.

        Manual calculation: 100*2 + 8*3 = 200 + 24 = 224.
        """
        result = draft_kv_indices_used_len(100, 2, 8, 3)
        assert result == 224

    def test_grows_by_bs_per_extra_step(self):
        """Adding one more step should increase used_len by exactly bs."""
        seq_lens_sum = 500
        topk = 4
        bs = 12
        num_steps = 5

        val_base = draft_kv_indices_used_len(seq_lens_sum, topk, bs, num_steps)
        val_plus_one = draft_kv_indices_used_len(seq_lens_sum, topk, bs, num_steps + 1)
        assert val_plus_one - val_base == bs

    def test_zero_steps_equals_seq_lens_sum_times_topk(self):
        """At num_steps=0 there is no step contribution: result = seq_lens_sum*topk."""
        seq_lens_sum = 300
        topk = 3
        bs = 10  # irrelevant at num_steps=0
        result = draft_kv_indices_used_len(seq_lens_sum, topk, bs, 0)
        # 300*3 = 900
        assert result == 900

    def test_topk_one_chain_case(self):
        """topk=1 should match a simple per-step chain: one index per branch per step.

        With topk=1 there is a single branch per sequence, so the step
        contribution is just bs*num_steps.
        """
        seq_lens_sum = 75
        topk = 1
        bs = 6
        num_steps = 4
        # 75*1 + 6*4 = 75 + 24 = 99
        result = draft_kv_indices_used_len(seq_lens_sum, topk, bs, num_steps)
        assert result == 99


class TestDraftKvIndicesBufferWidth:
    """draft_kv_indices_buffer_width(num_seqs, topk, max_context_len)"""

    def test_hand_computed_case(self):
        """num_seqs=4, topk=3, max_context_len=256 -> 3072.

        Manual calculation: 4*3*256 = 3072.
        """
        result = draft_kv_indices_buffer_width(4, 3, 256)
        assert result == 3072

    def test_int32_overflow_fires(self):
        """Product exceeding 2**31 must raise AssertionError."""
        # 10_000 * 10_000 * 10_000 = 10^12 >> 2^31 (2147483648)
        with pytest.raises(AssertionError):
            draft_kv_indices_buffer_width(10_000, 10_000, 10_000)
