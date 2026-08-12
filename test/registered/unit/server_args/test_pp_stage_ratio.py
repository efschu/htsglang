"""Unit tests for --pp-stage-ratio: score-driven, full-attention-aware
derivation of the uneven pipeline layer split (#201 slice 3 item 2).

The planner contract under test:
* proportional in LAYER space for homogeneous models;
* for hybrids, boundaries snap so each stage also gets its
  score-proportional share of FULL-ATTENTION layers (a hybrid's KV mass
  follows the full-attention count -- the slice-2 finding);
* refusal, never a silent even split (the #202 lesson): unreadable depth,
  zero-full-attention stages, malformed scores.

CPU only, no model files: depth/kinds are patched onto ServerArgs.
"""

import os
import unittest
from unittest.mock import patch

from sglang.srt.distributed.utils import derive_pp_layer_split
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

PARTITION_ENV = "SGLANG_PP_LAYER_PARTITION"


def qwen_kinds(num_layers=64, interval=4):
    """Qwen3.5/3.6 GDN hybrid: every interval-th layer (1-based) is full
    attention."""
    return [(i + 1) % interval == 0 for i in range(num_layers)]


class TestDerivePPLayerSplit(CustomTestCase):
    def test_homogeneous_even_scores(self):
        self.assertEqual(
            derive_pp_layer_split([1, 1], num_hidden_layers=64), [32, 32]
        )

    def test_homogeneous_proportional(self):
        self.assertEqual(
            derive_pp_layer_split([3, 1], num_hidden_layers=64), [48, 16]
        )

    def test_hybrid_boundary_tracks_both_axes(self):
        # Qwen3.6 geometry: 64 layers, 16 full-attention. Scores 3:1 ->
        # 48/16 layers AND 12/4 full-attention layers.
        counts = derive_pp_layer_split([3, 1], is_full_attention=qwen_kinds())
        self.assertEqual(counts, [48, 16])
        kinds = qwen_kinds()
        stage0_full = sum(kinds[: counts[0]])
        self.assertEqual(stage0_full, 12)

    def test_hybrid_snap_window(self):
        # Scores 11:5 over 64 layers: the layer target is 44, and 44 is
        # inside the window that puts exactly 11 of 16 full-attention
        # layers on stage 0.
        counts = derive_pp_layer_split([11, 5], is_full_attention=qwen_kinds())
        self.assertEqual(counts, [44, 20])
        self.assertEqual(sum(qwen_kinds()[:44]), 11)

    def test_hybrid_even_scores_balance_full_attention(self):
        # The slice-2 2B smoke geometry: 24 layers, interval 4 (6 full).
        # Even scores must land 3 full-attention layers per stage.
        kinds = qwen_kinds(24, 4)
        counts = derive_pp_layer_split([1, 1], is_full_attention=kinds)
        self.assertEqual(sum(counts), 24)
        self.assertEqual(sum(kinds[: counts[0]]), 3)

    def test_hybrid_zero_full_attention_stage_refused(self):
        # 8 layers, full attention only at indices 3 and 7: a tiny stage 0
        # ends without any full-attention layer -> named refusal.
        kinds = qwen_kinds(8, 4)
        with self.assertRaisesRegex(ValueError, "zero of the model's"):
            derive_pp_layer_split([1, 1, 6], is_full_attention=kinds)

    def test_more_stages_than_layers_refused(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            derive_pp_layer_split([1, 1, 1], num_hidden_layers=2)

    def test_bad_scores_refused(self):
        with self.assertRaisesRegex(ValueError, "positive integers"):
            derive_pp_layer_split([1, 0], num_hidden_layers=8)

    def test_counts_always_sum_to_depth(self):
        for scores in ([1, 1], [5, 3], [7, 2, 2], [1, 6, 1]):
            for kinds in (qwen_kinds(64, 4), [True] * 64, qwen_kinds(48, 4)):
                counts = derive_pp_layer_split(scores, is_full_attention=kinds)
                self.assertEqual(sum(counts), len(kinds), (scores, counts))
                self.assertTrue(all(c >= 1 for c in counts), (scores, counts))

    def test_depth_mismatch_refused(self):
        with self.assertRaisesRegex(ValueError, "disagrees"):
            derive_pp_layer_split(
                [1, 1], is_full_attention=[True] * 8, num_hidden_layers=9
            )


def make_args(**kwargs):
    return ServerArgs(model_path="dummy", **kwargs)


class TestPPStageRatioHandler(CustomTestCase):
    def setUp(self):
        os.environ.pop(PARTITION_ENV, None)

    def tearDown(self):
        os.environ.pop(PARTITION_ENV, None)

    def test_requires_pipeline(self):
        args = make_args(pp_size=1, pp_stage_ratio=[3, 1])
        with self.assertRaisesRegex(ValueError, "no pipeline"):
            args._handle_pp_stage_ratio()

    def test_length_must_match_pp_size(self):
        args = make_args(pp_size=3, pp_stage_ratio=[3, 1])
        with self.assertRaisesRegex(ValueError, "one score per stage"):
            args._handle_pp_stage_ratio()

    def test_conflict_with_explicit_layer_ratio(self):
        args = make_args(pp_size=2, pp_stage_ratio=[3, 1], pp_layer_ratio=[44, 20])
        with self.assertRaisesRegex(ValueError, "not both"):
            args._handle_pp_stage_ratio()

    def test_unreadable_depth_refuses_not_defaults(self):
        args = make_args(pp_size=2, pp_stage_ratio=[3, 1])
        with self.assertRaisesRegex(ValueError, "Pass --pp-layer-ratio"):
            with patch.object(
                ServerArgs, "declared_num_hidden_layers", return_value=None
            ):
                args._handle_pp_stage_ratio()

    def test_derives_and_exports_partition(self):
        args = make_args(pp_size=2, pp_stage_ratio=[3, 1])
        with patch.object(
            ServerArgs, "declared_num_hidden_layers", return_value=64
        ), patch.object(
            ServerArgs, "declared_layer_kinds", return_value=qwen_kinds()
        ):
            args._handle_pp_stage_ratio()
            self.assertEqual(args.pp_layer_ratio, [48, 16])
            args._handle_pp_layer_ratio()
        self.assertEqual(os.environ.get(PARTITION_ENV), "48,16")

    def test_unset_touches_nothing(self):
        args = make_args(pp_size=2)
        args._handle_pp_stage_ratio()
        self.assertIsNone(args.pp_layer_ratio)
        self.assertNotIn(PARTITION_ENV, os.environ)


class TestPPLayerRatioFamilyGate(CustomTestCase):
    """#485: an EXPLICIT --pp-layer-ratio now meets the same hybrid check the
    derived path applies, instead of reaching get_pp_indices unexamined."""

    def setUp(self):
        os.environ.pop(PARTITION_ENV, None)

    def tearDown(self):
        os.environ.pop(PARTITION_ENV, None)

    def _handle(self, ratio, kinds=None, depth=64):
        args = make_args(pp_size=3, pp_layer_ratio=ratio)
        with patch.object(
            ServerArgs, "declared_num_hidden_layers", return_value=depth
        ), patch.object(
            ServerArgs,
            "declared_layer_kinds",
            return_value=qwen_kinds() if kinds is None else kinds,
        ):
            args._handle_pp_layer_ratio()
        return args

    def test_zero_full_attention_stage_is_refused(self):
        # Layers 0..2 carry no full-attention layer (the first is index 3).
        with self.assertRaisesRegex(ValueError, "zero of the model's 16"):
            self._handle([3, 45, 16])

    def test_valid_split_still_passes(self):
        self._handle([28, 20, 16])
        self.assertEqual(os.environ.get(PARTITION_ENV), "28,20,16")

    def test_decoupled_split_passes(self):
        self._handle([31, 17, 16])
        self.assertEqual(os.environ.get(PARTITION_ENV), "31,17,16")

    def test_non_hybrid_is_not_gated(self):
        """An all-attention stack has no family split to get wrong."""
        self._handle([3, 45, 16], kinds=[True] * 64)
        self.assertEqual(os.environ.get(PARTITION_ENV), "3,45,16")

    def test_gate_is_skipped_when_geometry_is_unknown(self):
        """No declared kinds -> no census, and certainly no invented one."""
        args = make_args(pp_size=3, pp_layer_ratio=[3, 45, 16])
        with patch.object(
            ServerArgs, "declared_num_hidden_layers", return_value=64
        ), patch.object(ServerArgs, "declared_layer_kinds", return_value=None):
            args._handle_pp_layer_ratio()
        self.assertEqual(os.environ.get(PARTITION_ENV), "3,45,16")


class TestPPAttnStageRatio(CustomTestCase):
    """--pp-attn-stage-ratio: the per-family decoupling on the CLI (#485)."""

    def setUp(self):
        os.environ.pop(PARTITION_ENV, None)

    def tearDown(self):
        os.environ.pop(PARTITION_ENV, None)

    def _derive(self, **kwargs):
        # Each derivation is its own launcher process in reality; clear the
        # export so the conflict guard is not tripped by the previous case.
        os.environ.pop(PARTITION_ENV, None)
        args = make_args(**kwargs)
        with patch.object(
            ServerArgs, "declared_num_hidden_layers", return_value=64
        ), patch.object(
            ServerArgs, "declared_layer_kinds", return_value=qwen_kinds()
        ):
            args._handle_pp_stage_ratio()
            args._handle_pp_layer_ratio()
        return args

    def test_requires_stage_ratio(self):
        args = make_args(pp_size=3, pp_attn_stage_ratio=[7, 5, 4])
        with self.assertRaisesRegex(ValueError, "cannot be used on its own"):
            args._handle_pp_stage_ratio()

    def test_length_must_match_pp_size(self):
        args = make_args(
            pp_size=3, pp_stage_ratio=[31, 17, 16], pp_attn_stage_ratio=[7, 5]
        )
        with patch.object(
            ServerArgs, "declared_num_hidden_layers", return_value=64
        ), patch.object(
            ServerArgs, "declared_layer_kinds", return_value=qwen_kinds()
        ):
            with self.assertRaisesRegex(ValueError, "one score per stage"):
                args._handle_pp_stage_ratio()

    def test_refuses_on_a_non_hybrid_model(self):
        args = make_args(
            pp_size=2, pp_stage_ratio=[3, 1], pp_attn_stage_ratio=[3, 1]
        )
        with patch.object(
            ServerArgs, "declared_num_hidden_layers", return_value=64
        ), patch.object(
            ServerArgs, "declared_layer_kinds", return_value=[True] * 64
        ):
            with self.assertRaisesRegex(ValueError, "not a hybrid"):
                args._handle_pp_stage_ratio()

    def test_reaches_the_split_the_coupled_vector_cannot(self):
        """31,17,16 with attention held at 7,5,4 -- three linear layers move
        off stage 1 and the KV split does not budge. The coupled derivation
        snaps the same request to 32,16,16 (attention 8,4,4)."""
        coupled = self._derive(pp_size=3, pp_stage_ratio=[31, 17, 16])
        self.assertEqual(coupled.pp_layer_ratio, [32, 16, 16])

        decoupled = self._derive(
            pp_size=3, pp_stage_ratio=[31, 17, 16], pp_attn_stage_ratio=[7, 5, 4]
        )
        self.assertEqual(decoupled.pp_layer_ratio, [31, 17, 16])
        self.assertEqual(os.environ.get(PARTITION_ENV), "31,17,16")

        kinds = qwen_kinds()
        for counts, expected in ((coupled.pp_layer_ratio, 8), ([31, 17, 16], 7)):
            self.assertEqual(sum(kinds[: counts[0]]), expected)

    def test_unset_is_byte_identical(self):
        with_flag = self._derive(
            pp_size=3, pp_stage_ratio=[14, 10, 8], pp_attn_stage_ratio=[7, 5, 4]
        )
        without = self._derive(pp_size=3, pp_stage_ratio=[14, 10, 8])
        # 14,10,8 already lands on 7,5,4, so asking for it explicitly must
        # change nothing at all.
        self.assertEqual(without.pp_layer_ratio, [28, 20, 16])
        self.assertEqual(with_flag.pp_layer_ratio, [28, 20, 16])


if __name__ == "__main__":
    unittest.main()
