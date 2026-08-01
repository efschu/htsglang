"""The draft-EXTEND uneven-DCP metadata split (#108 slice 2).

WHAT THE SPLIT IS. Under a token-sharded DRAFT KV pool the draft-extend
forward is decomposed exactly like the target-verify forward:

    paged  -- this rank's OWNED slots of the COMMITTED prefix, full replicated
              kv-heads, NON-causal, cross-rank LSE-merged
    ragged -- the num_tokens_per_req tokens this step appends, LOCAL heads,
              causal, no collective

and the two are combined by the LSE merge. Same owner rule, same kernels, same
collectives as the target side -- if the two sides used different owner rules
the pools would disagree about who holds a token.

THE ONE THING THAT IS NOT SHARED WITH VERIFY, and the whole correctness content
of the branch: for verify, ``paged_kernel_lens`` IS the committed prefix (the
draft tokens are not in ``seq_lens``). For draft-extend, ``seq_lens`` ALREADY
counts the tokens this step appends -- they are written into the pool by the
owner-rule masked write at the top of the same forward. The paged read must
therefore cover ``seq_len - num_tokens_per_req``. Reading the full ``seq_len``
would let every query attend its OWN key through the non-causal paged stage as
well as through the causal ragged stage, i.e. count it twice in the LSE merge:
a wrong answer, not a crash.

Everything here is CPU-only and hermetic: the prefix rule and the owner rule
are pure integer functions, and the wiring is pinned by reading the source.
"""

import inspect
import unittest

import torch

from sglang.srt.layers.dcp.lockstep import (
    dcp_forces_prefix,
    draft_extend_prefix_lens,
    weightless_has_prefix,
)
from sglang.srt.layers.dcp.owner import (
    dcp_weighted_owner_bounds,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestDraftExtendPrefixLens(CustomTestCase):
    """The subtraction, which is the branch's whole correctness content."""

    def test_prefix_excludes_this_steps_appended_tokens(self):
        seq = torch.tensor([100, 64, 33], dtype=torch.int32)
        got = draft_extend_prefix_lens(seq.clone(), 4)
        self.assertEqual(got.tolist(), [96, 60, 29])

    def test_double_count_falsifier(self):
        """Stated as the defect it prevents: if the prefix were the full
        seq_len, the k tokens this step appends would be read by BOTH stages.
        The gap between the two candidate lengths is exactly k per request."""
        seq = torch.tensor([100, 64], dtype=torch.int32)
        k = 4
        prefix = draft_extend_prefix_lens(seq.clone(), k)
        overlap = (seq - prefix).tolist()
        self.assertEqual(overlap, [k, k])

    def test_a_request_shorter_than_the_step_clamps_to_zero(self):
        """No committed prefix at all; a negative length would index
        backwards into req_to_token."""
        seq = torch.tensor([3, 1, 0], dtype=torch.int32)
        self.assertEqual(draft_extend_prefix_lens(seq.clone(), 4).tolist(), [0, 0, 0])

    def test_zero_or_negative_k_is_identity(self):
        seq = torch.tensor([10, 20], dtype=torch.int32)
        self.assertEqual(draft_extend_prefix_lens(seq.clone(), 0).tolist(), [10, 20])

    def test_it_is_a_vector_op_over_a_constant_k(self):
        """num_tokens_per_req is constant across the batch by construction (the
        draft-extend qo layout is a fixed stride so it can be graph-captured),
        which is what makes one subtraction correct for every request."""
        seq = torch.tensor([7, 7, 7, 7], dtype=torch.int32)
        got = draft_extend_prefix_lens(seq.clone(), 2)
        self.assertEqual(len(set(got.tolist())), 1)


class TestForcesPrefixIsRankUniform(CustomTestCase):
    """#94 family: a rank-local condition must never decide a group collective.

    The prefix stage is where the Q all-gather and the LSE merge live. If
    'does this step have a prefix' were answered per rank, the owner of a short
    prefix would sit alone in an all-gather nobody joins.
    """

    def test_draft_extend_forces_the_prefix_stage(self):
        self.assertTrue(dcp_forces_prefix(False, True))

    def test_target_verify_still_forces(self):
        self.assertTrue(dcp_forces_prefix(True, False))

    def test_plain_extend_does_not_force(self):
        self.assertFalse(dcp_forces_prefix(False, False))

    def test_forcing_survives_an_empty_or_absent_length_vector(self):
        """THE falsifier for the hang: a draft-extend batch carries no
        extend_prefix_lens (ForwardBatch fills it from batch.prefix_lens, which
        the draft-extend batch never sets). A length-based answer would be
        False and the prefix stage -- with its two collectives -- would be
        skipped on some ranks and not others."""
        for lens in (None, [], [0, 0, 0]):
            with self.subTest(lens=lens):
                self.assertTrue(
                    weightless_has_prefix(dcp_forces_prefix(False, True), lens)
                )

    def test_without_forcing_the_same_vectors_answer_false(self):
        """Pins that the forcing is what carries it, not the vector."""
        for lens in (None, [], [0, 0, 0]):
            with self.subTest(lens=lens):
                self.assertFalse(
                    weightless_has_prefix(dcp_forces_prefix(False, False), lens)
                )


class TestOwnerRuleIsSharedWithTheTargetSide(CustomTestCase):
    """The draft side must shard by the SAME owner rule as the target side, on
    every geometry, or the two pools disagree about who holds a token."""

    GEOMETRIES = (
        # (name, token ratios)
        ("even_tp2", [1, 1]),
        ("even_tp3", [1, 1, 1]),
        ("uneven_rig", [30, 17, 17]),  # the reference rig's measured vector
        ("uneven_skew", [3, 1]),
        ("tp_gt_kv_heads", [2, 1, 1, 1]),  # TP=4 over 2 kv heads: the full-win case
    )

    def _owned(self, ratios, rank, n_slots):
        """Slots of [0, n_slots) this rank owns under the weighted rule."""
        S = sum(ratios)
        lo = sum(ratios[:rank])
        hi = lo + ratios[rank]
        return [L for L in range(n_slots) if lo <= (L % S) < hi]

    def test_every_prefix_slot_has_exactly_one_owner(self):
        for name, ratios in self.GEOMETRIES:
            with self.subTest(geometry=name):
                n = 257  # deliberately not a multiple of any S here
                seen = {}
                for rank in range(len(ratios)):
                    for slot in self._owned(ratios, rank, n):
                        self.assertNotIn(
                            slot, seen, f"slot {slot} owned by 2 ranks in {name}"
                        )
                        seen[slot] = rank
                self.assertEqual(len(seen), n, f"{name}: some slot has no owner")

    def test_owned_counts_are_ratio_proportional(self):
        """The property the byte-win finding rests on: a rank's share of the
        prefix is ratio_r / S, so the draft pool shrinks by that factor."""
        for name, ratios in self.GEOMETRIES:
            with self.subTest(geometry=name):
                S = sum(ratios)
                n = S * 40  # whole blocks, so the split is exact
                for rank, r in enumerate(ratios):
                    owned = len(self._owned(ratios, rank, n))
                    self.assertEqual(owned, n * r // S)

    def test_the_shared_helper_agrees_with_the_rule(self):
        """``dcp_weighted_owner_bounds`` is the function the draft branch's
        index builder resolves its (S, lo, hi) from -- the SAME one the target
        verify split uses. Drive it against an installed plan so the draft side
        cannot drift from the prose above.
        """
        from sglang.srt.distributed.utils import (
            get_cp_token_ratios,
            get_tp_partition_ratios,
            set_cp_token_ratios,
            set_tp_partition_ratios,
        )

        ratios = [30, 17, 17]
        saved_tp = get_tp_partition_ratios()
        saved_cp = get_cp_token_ratios()
        try:
            set_tp_partition_ratios(ratios)
            set_cp_token_ratios(ratios)
            for rank in range(len(ratios)):
                lo = sum(ratios[:rank])
                hi = lo + ratios[rank]
                bounds = dcp_weighted_owner_bounds(len(ratios), rank)
                self.assertIsNotNone(bounds)
                self.assertEqual(
                    (bounds[0], bounds[1], bounds[2]), (sum(ratios), lo, hi)
                )
                # and the read-slot helper reproduces the same ownership set
                owned = self._owned(ratios, rank, sum(ratios))
                self.assertEqual(len(owned), ratios[rank])
        finally:
            set_tp_partition_ratios(saved_tp)
            set_cp_token_ratios(saved_cp)


class TestSplitWiring(CustomTestCase):
    """The branch exists, is reached for the right spec type, and reuses the
    target machinery instead of a parallel scheme. Source-level so it is
    hermetic; the numeric behaviour is the GPU ticket's job."""

    def _backend_src(self):
        from sglang.srt.layers.attention import flashinfer_backend

        return inspect.getsource(flashinfer_backend)

    def test_the_draft_extend_branch_exists_and_is_its_own(self):
        src = self._backend_src()
        self.assertIn("SpecInputType.EAGLE_DRAFT_EXTEND", src)
        # NOT folded into the verify set: the prefix derivation differs.
        self.assertNotIn(
            "EAGLE_DRAFT_EXTEND}", src.split("_DCP_VERIFY_SPEC_INPUT_TYPES = ")[1][:200]
        )

    def test_the_branch_uses_the_shared_prefix_helper(self):
        self.assertIn("draft_extend_prefix_lens(", self._backend_src())

    def test_the_branch_uses_the_shared_owner_rule_builder(self):
        """Same index builder the verify split uses -- not a parallel one."""
        src = self._backend_src()
        self.assertGreaterEqual(src.count("_build_dcp_weighted_kv_indices("), 3)

    def test_force_prefix_goes_through_the_shared_rule(self):
        from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend

        body = inspect.getsource(FlashInferAttnBackend.forward_extend)
        self.assertIn("dcp_forces_prefix(", body)
        self.assertIn("is_draft_extend_v2()", body)

    def test_draft_extend_gets_its_own_graph_mode_ragged_wrapper(self):
        """The captured draft-extend graph runs the ragged stage inside the
        capture, so it needs fixed indptr buffers -- and its OWN per-bucket
        dict, because it shares bs with the verify graph but has a different
        qo stride."""
        from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend

        self.assertTrue(
            hasattr(FlashInferAttnBackend, "_get_draft_extend_ragged_cg_wrapper")
        )
        src = inspect.getsource(
            FlashInferAttnBackend._get_draft_extend_ragged_cg_wrapper
        )
        self.assertIn("use_cuda_graph=True", src)
        self.assertIn("qo_indptr_buf", src)
        self.assertIn("draft_extend_ragged_cg_wrappers", src)

    def test_the_wrapper_dicts_are_separate(self):
        src = self._backend_src()
        self.assertIn("self.verify_ragged_cg_wrappers: dict = {}", src)
        self.assertIn("self.draft_extend_ragged_cg_wrappers: dict = {}", src)

    def test_no_mask_buffer_on_the_draft_extend_wrapper(self):
        """A draft-extend chain is plain causal, and topk > 1 cannot reach this
        layout (boot-refused). A mask buffer would silently select flashinfer's
        CUSTOM mode on every replay."""
        from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend

        src = inspect.getsource(
            FlashInferAttnBackend._get_draft_extend_ragged_cg_wrapper
        )
        self.assertNotIn("custom_mask_buf", src.split('"""')[2])


class TestReplicatedDefaultIsUntouched(CustomTestCase):
    """Regression: none of this may move the default path.

    The layout predicate from slice 1 stays the single source of the decision;
    the split is reached only when the draft pool really is token-sharded, and
    a draft runner under the default layout never sets uneven_dcp at all.
    """

    def test_predicate_still_reads_the_default_as_replicated(self):
        from types import SimpleNamespace

        from sglang.srt.layers.dcp.owner import draft_pool_is_replicated

        for args in (
            None,
            SimpleNamespace(),
            SimpleNamespace(draft_kv_layout="replicated"),
        ):
            for is_draft in (False, True):
                with self.subTest(args=args, is_draft=is_draft):
                    self.assertEqual(draft_pool_is_replicated(is_draft, args), is_draft)

    def test_no_silent_auto_flip_to_dcp(self):
        """Nothing in the split may turn the layout on by itself -- the byte
        win is configuration-dependent (it only pays when ratio_r/S beats
        local_heads/total_heads), so the choice stays the user's."""
        from sglang.srt.layers.attention import flashinfer_backend

        src = inspect.getsource(flashinfer_backend)
        self.assertNotIn('draft_kv_layout = "dcp"', src)
        self.assertNotIn("draft_kv_layout='dcp'", src)

    def test_the_branch_is_gated_on_uneven_dcp(self):
        """A draft runner on the default layout has uneven_dcp False, so the
        branch is unreachable for it. Pinned because the gate is what keeps
        the default byte-identical."""
        from sglang.srt.layers.attention import flashinfer_backend

        src = inspect.getsource(flashinfer_backend)
        marker = "== SpecInputType.EAGLE_DRAFT_EXTEND"
        head = src[: src.index(marker)]
        tail = head[-400:]
        self.assertIn("self.attn_backend.uneven_dcp", tail)


if __name__ == "__main__":
    unittest.main()
