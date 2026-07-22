"""Unit tests for the draft-solo KV planning fixes -- CPU only, no GPU.

Two independent blockers of --speculative-draft-placement solo on the uneven
tp3 DCP topology are covered:

1. pool_configurator.solo_draft_kv_cell_factor: the per-rank KV/mamba planner
   must charge the solo draft's KV footprint on the HOSTING rank at the GLOBAL
   token scale (the solo draft pool is not token-sharded) and must NOT charge
   it at all on the SHADOW ranks (they allocate no draft pool). Split placement
   must stay byte-identical.

2. flashinfer_backend._DCP_VERIFY_SPEC_INPUT_TYPES: the uneven-DCP target-verify
   split (paged owned prefix + ragged draft->draft) must also cover
   DFLASH_VERIFY. Without it the ragged wrapper is never planned while
   _forward_extend_dcp dereferences it on every rank.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.distributed.utils import set_cp_token_ratios, set_tp_partition_ratios
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_mr(
    *,
    solo_active=True,
    solo_rank=0,
    tp_rank=0,
    dcp_size=3,
    is_draft_worker=False,
    num_kv_heads_local=2,
    num_kv_heads_total=8,
):
    mr = MagicMock()
    mr.tp_rank = tp_rank
    mr.dcp_size = dcp_size
    mr.is_draft_worker = is_draft_worker

    sa = SimpleNamespace()
    sa.speculative_draft_solo_active = lambda: solo_active
    sa.speculative_draft_solo_rank = lambda: solo_rank
    mr.server_args = sa

    mc = SimpleNamespace()
    mc.get_num_kv_heads = lambda tp_size: num_kv_heads_local
    mc.get_total_num_kv_heads = lambda: num_kv_heads_total
    mr.model_config = mc
    return mr


class TestSoloDraftKvCellFactor(unittest.TestCase):
    """solo_draft_kv_cell_factor: 1.0 (no-op) unless solo placement is on."""

    def setUp(self):
        set_cp_token_ratios(None)
        # The reference topology always carries a --rank-tp-ratio base plan,
        # which puts uneven-DCP KV replication in force (every rank stores the
        # FULL kv heads), so no head correction applies there.
        set_tp_partition_ratios([4, 1, 1])

    def tearDown(self):
        set_cp_token_ratios(None)
        set_tp_partition_ratios(None)

    def _factor(self, mr, *, dcp_size=3, tp_size=3):
        from sglang.srt.model_executor.pool_configurator import (
            solo_draft_kv_cell_factor,
        )

        with get_parallel().override(
            attn_tp_size=tp_size, attn_dcp_size=dcp_size, attn_dcp_rank=mr.tp_rank
        ):
            return solo_draft_kv_cell_factor(mr)

    def test_split_placement_is_a_noop(self):
        """Default placement: factor is exactly 1.0 on every rank."""
        set_cp_token_ratios([33, 13, 18])
        for rank in range(3):
            mr = _make_mr(solo_active=False, tp_rank=rank)
            self.assertEqual(self._factor(mr), 1.0)

    def test_server_args_without_the_flag_is_a_noop(self):
        """Old/mocked ServerArgs without the solo helpers must not raise."""
        mr = _make_mr()
        mr.server_args = SimpleNamespace()
        self.assertEqual(self._factor(mr), 1.0)

    def test_shadow_rank_is_not_charged(self):
        """Shadow ranks hold no draft KV pool -> factor 0.0."""
        set_cp_token_ratios([33, 13, 18])
        for rank in (1, 2):
            mr = _make_mr(solo_rank=0, tp_rank=rank)
            self.assertEqual(self._factor(mr), 0.0)

    def test_host_rank_scales_by_inverse_token_share(self):
        """The host's draft pool spans all S token slots, its target pool only
        ratio_host of them -> factor S / ratio_host."""
        set_cp_token_ratios([33, 13, 18])
        mr = _make_mr(solo_rank=0, tp_rank=0)
        self.assertAlmostEqual(self._factor(mr), 64 / 33)

        # A host with a SMALLER token share pays a proportionally larger
        # draft-KV charge per local token.
        set_cp_token_ratios([5, 13, 14])
        mr = _make_mr(solo_rank=0, tp_rank=0)
        self.assertAlmostEqual(self._factor(mr), 32 / 5)

    def test_host_on_non_zero_rank(self):
        set_cp_token_ratios([13, 33, 18])
        host = _make_mr(solo_rank=1, tp_rank=1)
        self.assertAlmostEqual(self._factor(host), 64 / 33)
        shadow = _make_mr(solo_rank=1, tp_rank=0)
        self.assertEqual(self._factor(shadow), 0.0)

    def test_even_dcp_scales_by_dcp_size(self):
        """No token vector installed: the even modulo owner rule gives every
        rank 1/dcp_size of the tokens."""
        mr = _make_mr(solo_rank=0, tp_rank=0, dcp_size=3)
        self.assertAlmostEqual(self._factor(mr), 3.0)

    def test_even_dcp_without_a_rank_plan_also_corrects_heads(self):
        """Stock (head-sharded) DCP: the target cell used this rank's kv-head
        SHARD, the solo draft keeps all heads -> head correction on top."""
        set_tp_partition_ratios(None)
        mr = _make_mr(solo_rank=0, tp_rank=0, dcp_size=3)
        self.assertAlmostEqual(self._factor(mr), 3.0 * (8 / 2))

    def test_no_dcp_head_correction_only(self):
        """dcp_size == 1: the solo draft keeps ALL kv heads while the target
        cell was computed on this rank's head shard."""
        set_tp_partition_ratios(None)
        mr = _make_mr(solo_rank=0, tp_rank=0, dcp_size=1)
        self.assertAlmostEqual(self._factor(mr, dcp_size=1), 8 / 2)

    def test_draft_worker_runner_untouched(self):
        mr = _make_mr(solo_rank=0, tp_rank=0, is_draft_worker=True)
        self.assertEqual(self._factor(mr), 1.0)


class TestApplySoloDraftKvCellFactor(unittest.TestCase):
    """apply_solo_draft_kv_cell_factor rescales ONLY the draft part."""

    def setUp(self):
        set_cp_token_ratios(None)
        # The reference topology always carries a --rank-tp-ratio base plan,
        # which puts uneven-DCP KV replication in force (every rank stores the
        # FULL kv heads), so no head correction applies there.
        set_tp_partition_ratios([4, 1, 1])

    def tearDown(self):
        set_cp_token_ratios(None)
        set_tp_partition_ratios(None)

    def _apply(self, mr, target_cell, cell_with_draft, *, dcp_size=3):
        from sglang.srt.model_executor.pool_configurator import (
            apply_solo_draft_kv_cell_factor,
        )

        with get_parallel().override(
            attn_tp_size=3, attn_dcp_size=dcp_size, attn_dcp_rank=mr.tp_rank
        ):
            return apply_solo_draft_kv_cell_factor(mr, target_cell, cell_with_draft)

    def test_split_is_byte_identical(self):
        set_cp_token_ratios([33, 13, 18])
        mr = _make_mr(solo_active=False)
        self.assertEqual(self._apply(mr, 1000, 1160), 1160)

    def test_no_draft_part_is_byte_identical(self):
        set_cp_token_ratios([33, 13, 18])
        mr = _make_mr(solo_rank=0, tp_rank=0)
        self.assertEqual(self._apply(mr, 1000, 1000), 1000)

    def test_shadow_drops_the_draft_term(self):
        set_cp_token_ratios([33, 13, 18])
        mr = _make_mr(solo_rank=0, tp_rank=1)
        self.assertEqual(self._apply(mr, 1000, 1160), 1000)

    def test_host_inflates_only_the_draft_term(self):
        set_cp_token_ratios([33, 13, 18])
        mr = _make_mr(solo_rank=0, tp_rank=0)
        got = self._apply(mr, 1000, 1160)
        self.assertEqual(got, 1000 + round(160 * 64 / 33))
        # The target term is untouched, and the cell can never shrink below it.
        self.assertGreater(got, 1000)

    def test_host_capacity_invariant(self):
        """The corrected cell makes the physical invariant hold:
        n_host * t_target + C * t_draft <= A_host, with n_host = C*ratio/S."""
        set_cp_token_ratios([33, 13, 18])
        mr = _make_mr(solo_rank=0, tp_rank=0)
        t_target, t_draft = 31800, 5100
        cell = self._apply(mr, t_target, t_target + t_draft)
        available = 2 * 1024**3
        p_host = available // cell  # == C * ratio_host / S
        context = p_host * 64 // 33
        used = p_host * t_target + context * t_draft
        self.assertLessEqual(used, available)

        # The OLD (unscaled) accounting over-commits the host: this is the
        # regression the fix removes.
        p_old = available // (t_target + t_draft)
        used_old = p_old * t_target + (p_old * 64 // 33) * t_draft
        self.assertGreater(used_old, available)


class TestDcpVerifySpecInputTypes(unittest.TestCase):
    """The uneven-DCP target-verify branch must cover DFLASH as well as EAGLE."""

    def test_both_verify_types_are_covered(self):
        from sglang.srt.layers.attention.flashinfer_backend import (
            _DCP_VERIFY_SPEC_INPUT_TYPES,
        )
        from sglang.srt.speculative.spec_info import SpecInputType

        self.assertIn(SpecInputType.EAGLE_VERIFY, _DCP_VERIFY_SPEC_INPUT_TYPES)
        self.assertIn(SpecInputType.DFLASH_VERIFY, _DCP_VERIFY_SPEC_INPUT_TYPES)

    def test_draft_only_types_are_not_covered(self):
        """Draft-side / non-verify spec inputs must keep the old branch."""
        from sglang.srt.layers.attention.flashinfer_backend import (
            _DCP_VERIFY_SPEC_INPUT_TYPES,
        )
        from sglang.srt.speculative.spec_info import SpecInputType

        self.assertNotIn(SpecInputType.EAGLE_DRAFT, _DCP_VERIFY_SPEC_INPUT_TYPES)

    def test_dflash_verify_is_a_linear_chain(self):
        """The DCP split plans the ragged draft->draft block as plain CAUSAL.
        That is only correct because DFLASH verify is non-tree (topk == 1)."""
        from sglang.srt.speculative.dflash_info import DFlashVerifyInput

        self.assertEqual(DFlashVerifyInput.topk, 1)

    def test_flashinfer_skips_the_dflash_verify_custom_mask(self):
        """...and because flashinfer gets no DFLASH custom mask, so the paged
        prefix read stays non-causal exactly as in the EAGLE branch."""
        from sglang.srt.speculative.dflash_utils import (
            resolve_dflash_verify_mask_policy,
        )

        class FlashInferAttnBackend:
            full_attn_backend = None

        _, needs_mask = resolve_dflash_verify_mask_policy(FlashInferAttnBackend())
        self.assertFalse(needs_mask)


if __name__ == "__main__":
    unittest.main()
