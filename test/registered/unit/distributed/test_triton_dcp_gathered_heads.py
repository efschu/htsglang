"""Decode-workspace head count under DCP: per-rank, not just the total.

Regression pin for a latent out-of-bounds write in the Triton backend.
``TritonAttnBackend.__init__`` used to size the decode attn_logits/attn_lse
workspaces as::

    _plan_aware_num_q_heads(model_config) * dcp_size

but ``_plan_aware_num_q_heads`` is ALREADY plan-aware and returns *this rank's*
(unequal) shard, so multiplying double-counts. With total_q=32, kv=8,
tp=dcp=3 and --rank-tp-ratio [12,6,6] the real split is [16,8,8]:

    rank 0: 16*3 = 48  vs true gathered 32   -> over-allocates (harmless)
    rank 1:  8*3 = 24  vs true gathered 32   -> EIGHT HEADS SHORT
    rank 2:  8*3 = 24  vs true gathered 32   -> EIGHT HEADS SHORT

WHY EVERY ASSERTION HERE IS PER-RANK: ``[12,6,6]`` and ``[16,8,8]`` both sum to
32, and so does a correct answer. A test that only checked the total would
happily pass a formula that distributes wrongly -- which is precisely the bug.
So each rank's allocation is asserted individually.

CPU only: this is pure integer partition arithmetic, no device involved.
"""

import unittest
from unittest import mock

from sglang.srt.distributed.utils import (
    get_tp_partition_ratios,
    set_tp_partition_ratios,
)
from sglang.srt.layers.attention.triton_backend import (
    _plan_aware_dcp_gathered_q_heads,
    _plan_aware_num_q_heads,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _HFTextConfig:
    swa_num_key_value_heads = None


class _ModelConfig:
    """Minimal stand-in: the helpers only read these three things."""

    def __init__(self, num_q, num_kv):
        self.num_attention_heads = num_q
        self._num_kv = num_kv
        self.hf_text_config = _HFTextConfig()

    def get_total_num_kv_heads(self):
        return self._num_kv


class _Parallel:
    def __init__(self, tp_size, tp_rank):
        self.attn_tp_size = tp_size
        self.attn_tp_rank = tp_rank


def _gathered_for_rank(cfg, tp_size, rank, dcp_size):
    with mock.patch(
        "sglang.srt.layers.attention.triton_backend.get_parallel",
        return_value=_Parallel(tp_size, rank),
    ):
        return _plan_aware_dcp_gathered_q_heads(cfg, dcp_size)


def _local_for_rank(cfg, tp_size, rank):
    with mock.patch(
        "sglang.srt.layers.attention.triton_backend.get_parallel",
        return_value=_Parallel(tp_size, rank),
    ):
        return _plan_aware_num_q_heads(cfg)


class TestTritonDcpGatheredHeads(CustomTestCase):
    def setUp(self):
        self._saved = get_tp_partition_ratios()

    def tearDown(self):
        set_tp_partition_ratios(self._saved)

    def test_uneven_plan_every_rank_holds_the_full_gathered_set(self):
        """THE regression pin. Each rank asserted separately."""
        cfg = _ModelConfig(num_q=32, num_kv=8)
        tp = dcp = 3
        for ratios in ([12, 6, 6], [2, 1, 1]):
            set_tp_partition_ratios(ratios)
            # The real shard split -- unequal, and that is intended.
            local = [_local_for_rank(cfg, tp, r) for r in range(tp)]
            self.assertEqual(local, [16, 8, 8], f"shard split changed for {ratios}")
            # ... but the GATHERED workspace must be the full set on EVERY rank.
            for r in range(tp):
                got = _gathered_for_rank(cfg, tp, r, dcp)
                self.assertEqual(
                    got,
                    32,
                    f"ratios={ratios} rank={r}: workspace sized {got}, "
                    f"need 32 (gathered)",
                )

    def test_old_formula_would_have_under_allocated(self):
        """Pin the BUG itself, so a revert cannot pass silently.

        Asserts that local*dcp_size is short on the small ranks -- if someone
        reinstates that expression, this test names exactly what breaks.
        """
        cfg = _ModelConfig(num_q=32, num_kv=8)
        tp = dcp = 3
        set_tp_partition_ratios([12, 6, 6])
        gathered = 32
        old = [_local_for_rank(cfg, tp, r) * dcp for r in range(tp)]
        self.assertEqual(old, [48, 24, 24])
        self.assertGreater(old[0], gathered)  # rank 0 over-allocated
        for r in (1, 2):
            self.assertLess(old[r], gathered)  # ranks 1/2 UNDER -> OOB write
            self.assertEqual(gathered - old[r], 8)

    def test_even_tp_is_byte_identical_to_the_old_expression(self):
        """No plan installed -> must reproduce (total_q // tp) * dcp exactly."""
        set_tp_partition_ratios(None)
        for num_q, tp, dcp in ((32, 4, 4), (32, 4, 2), (16, 2, 2), (64, 8, 8)):
            cfg = _ModelConfig(num_q=num_q, num_kv=tp)
            for r in range(tp):
                expected = (num_q // tp) * dcp
                self.assertEqual(_gathered_for_rank(cfg, tp, r, dcp), expected)

    def test_uneven_tp_without_dcp_is_unchanged(self):
        """dcp_size == 1 is a working, validated path: uneven TP and no gather.

        Each rank must keep its OWN shard count -- emphatically not the total,
        which would silently inflate every workspace on an existing feature.
        """
        cfg = _ModelConfig(num_q=32, num_kv=8)
        tp = 3
        set_tp_partition_ratios([12, 6, 6])
        for r in range(tp):
            self.assertEqual(
                _gathered_for_rank(cfg, tp, r, 1),
                _local_for_rank(cfg, tp, r),
            )
        # And they really are unequal, so this is a meaningful assertion.
        self.assertEqual([_gathered_for_rank(cfg, tp, r, 1) for r in range(tp)],
                         [16, 8, 8])

    def test_partial_dcp_group_never_under_allocates(self):
        """1 < dcp_size < tp_size: no exact per-group sum is available here, so
        the value must be an UPPER bound. Over-allocating is harmless;
        under-allocating writes out of bounds."""
        cfg = _ModelConfig(num_q=32, num_kv=8)
        tp, dcp = 4, 2
        set_tp_partition_ratios([4, 2, 1, 1])
        local = [_local_for_rank(cfg, tp, r) for r in range(tp)]
        worst_case_group = max(local) * dcp
        for r in range(tp):
            got = _gathered_for_rank(cfg, tp, r, dcp)
            self.assertGreaterEqual(
                got,
                worst_case_group,
                f"rank {r}: {got} is below the worst-case gathered "
                f"{worst_case_group}",
            )


if __name__ == "__main__":
    unittest.main()
