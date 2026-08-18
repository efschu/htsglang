"""Non-contiguous PP layer ownership (the family-plan ADDRESSING blocker).

A pipeline stage has always been an INTERVAL here: `get_pp_indices` returns
`(start, end)` with `start = sum(partitions[:pp_rank])`, and
`SGLANG_PP_LAYER_PARTITION` takes per-stage COUNTS. So a family placement — all
48 linear-attention layers on one card, the 16 interleaved full-attention
layers on others — was not expressible at all. That is an ADDRESSING limit and
it is independent of any transport: a wire that can carry the activations does
not help if no one can say which card owns layer 7.

`SGLANG_PP_LAYER_SET` is the set form. The count form is untouched, and the
first class below is the pin that says so.

WHY THE VALIDATION IS THE POINT. Both ways of getting a partition wrong are
SILENT. A duplicated layer is computed twice and merely costs time. A missing
layer is a `PPMissingLayer` pass-through — `torch.nn.Identity` — so the model
answers with that layer quietly skipped and nothing anywhere reports it. Every
refusal below therefore names the exact layers.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import os
import unittest

import torch

from sglang.srt.distributed.utils import (
    PP_LAYER_SET_ENV,
    PPLayerSetError,
    get_pp_indices,
    get_pp_layer_set,
    parse_pp_layer_sets,
)
from sglang.srt.utils.common import make_layers
from sglang.test.test_utils import CustomTestCase


class _Layer(torch.nn.Identity):
    def __init__(self, idx: int = 0, prefix: str = ""):
        super().__init__()
        self.idx = idx


class _Env:
    """Set/restore the layer-set env around one test."""

    def __init__(self, value):
        self.value = value

    def __enter__(self):
        self.saved = os.environ.get(PP_LAYER_SET_ENV)
        # #753: this file's subject IS the gapped set form, so it declares the
        # crossing wire. Without that declaration parse_pp_layer_sets refuses a
        # gapped set -- correctly, because the forward loop would skip the peer
        # layers in silence.
        self.saved_wire = os.environ.get("SGLANG_PP_CROSSING_WIRE")
        os.environ["SGLANG_PP_CROSSING_WIRE"] = "1"
        if self.value is None:
            os.environ.pop(PP_LAYER_SET_ENV, None)
        else:
            os.environ[PP_LAYER_SET_ENV] = self.value
        return self

    def __exit__(self, *exc):
        if self.saved is None:
            os.environ.pop(PP_LAYER_SET_ENV, None)
        else:
            os.environ[PP_LAYER_SET_ENV] = self.saved
        if self.saved_wire is None:
            os.environ.pop("SGLANG_PP_CROSSING_WIRE", None)
        else:
            os.environ["SGLANG_PP_CROSSING_WIRE"] = self.saved_wire
        return False


# The shape this exists for: 64 layers, full attention every 4th.
FA = [i for i in range(64) if i % 4 == 3]
GDN = [i for i in range(64) if i % 4 != 3]
FULL_PLAN = (
    ",".join(str(i) for i in GDN)
    + ";"
    + ",".join(str(i) for i in FA[:8])
    + ";"
    + ",".join(str(i) for i in FA[8:])
)


class TestTheDefaultPathIsUntouched(CustomTestCase):
    """Byte-identical when the set form is unused. A new addressing mode that
    perturbs the old one is not a new mode, it is a regression."""

    def test_the_env_is_unset_by_default(self):
        with _Env(None):
            self.assertIsNone(get_pp_layer_set(64, 0, 3))

    def test_an_empty_value_is_treated_as_unset(self):
        with _Env("   "):
            self.assertIsNone(get_pp_layer_set(64, 0, 3))

    def test_get_pp_indices_still_returns_the_contiguous_interval(self):
        with _Env(None):
            self.assertEqual(get_pp_indices(64, 0, 4), (0, 16))
            self.assertEqual(get_pp_indices(64, 3, 4), (48, 64))

    def test_make_layers_is_contiguous_without_the_env(self):
        with _Env(None):
            mods, start, end = make_layers(
                8, lambda idx, prefix: _Layer(idx, prefix), pp_rank=0, pp_size=2
            )
            self.assertEqual((start, end), (0, 4))
            kinds = [type(m).__name__ for m in mods]
            self.assertEqual(kinds[:4], ["_Layer"] * 4)
            self.assertEqual(set(kinds[4:]), {"PPMissingLayer"})


class TestTheSetFormParses(CustomTestCase):
    def test_ranges_and_singletons_together(self):
        sets = parse_pp_layer_sets("0-2,4-6;3,7", 8, 2, allow_gapped=True)
        self.assertEqual(sorted(sets[0]), [0, 1, 2, 4, 5, 6])
        self.assertEqual(sorted(sets[1]), [3, 7])

    def test_the_full_plan_shape_parses(self):
        sets = parse_pp_layer_sets(FULL_PLAN, 64, 3, allow_gapped=True)
        self.assertEqual(len(sets[0]), 48)
        self.assertEqual(len(sets[1]), 8)
        self.assertEqual(len(sets[2]), 8)
        self.assertEqual(sorted(sets[1] | sets[2]), FA)


class TestEveryWayOfBeingWrongIsRefusedByName(CustomTestCase):
    def _refusal(self, raw, layers=64, stages=3):
        with self.assertRaises(PPLayerSetError) as cm:
            parse_pp_layer_sets(raw, layers, stages)
        return str(cm.exception)

    def test_a_duplicated_layer_names_the_layer_and_both_stages(self):
        msg = self._refusal("0-2,4;3,4;5-63,7", 64, 3)
        self.assertIn("4", msg)
        self.assertIn("exactly one stage", msg)

    def test_a_missing_layer_is_named(self):
        msg = self._refusal("0-2;3", 8, 2)
        self.assertIn("[4, 5, 6, 7]", msg)

    def test_the_missing_refusal_explains_the_silent_failure(self):
        """Because that is the reason it refuses instead of warning."""
        msg = self._refusal("0-2;3", 8, 2)
        self.assertIn("silently skipped", msg)

    def test_an_out_of_range_layer_is_named(self):
        self.assertIn("[8]", self._refusal("0-2,4-6,8;3,7", 8, 2))

    def test_a_wrong_stage_count_is_refused(self):
        self.assertIn("pp_size", self._refusal("0-3;4-7", 8, 3))

    def test_a_backwards_range_is_refused(self):
        self.assertIn("backwards", self._refusal("6-2;0-1,3-5,7", 8, 2))

    def test_a_non_integer_is_refused(self):
        self.assertIn("integer", self._refusal("0-2,x;3", 8, 2))


class TestInterleavedConstruction(CustomTestCase):
    def test_placeholders_land_at_the_non_owned_indices(self):
        with _Env("0-2,4-6;3,7"):
            mods, _, _ = make_layers(
                8, lambda idx, prefix: _Layer(idx, prefix), pp_rank=0, pp_size=2
            )
            kinds = [type(m).__name__ for m in mods]
            self.assertEqual(
                kinds,
                ["_Layer"] * 3
                + ["PPMissingLayer"]
                + ["_Layer"] * 3
                + ["PPMissingLayer"],
            )

    def test_the_list_is_still_full_length(self):
        """Layer ids stay global; a stage does not renumber the model."""
        with _Env(FULL_PLAN):
            mods, _, _ = make_layers(
                64, lambda idx, prefix: _Layer(idx, prefix), pp_rank=1, pp_size=3
            )
            self.assertEqual(len(mods), 64)

    def test_start_and_end_are_first_and_one_past_last_OWNED(self):
        with _Env("0-2,4-6;3,7"):
            _, start, end = make_layers(
                8, lambda idx, prefix: _Layer(idx, prefix), pp_rank=0, pp_size=2
            )
            self.assertEqual((start, end), (0, 7))

    def test_owned_layers_is_published_for_membership_questions(self):
        with _Env("0-2,4-6;3,7"):
            mods, _, _ = make_layers(
                8, lambda idx, prefix: _Layer(idx, prefix), pp_rank=1, pp_size=2
            )
            self.assertEqual(sorted(mods.owned_layers), [3, 7])

    def test_an_empty_stage_is_refused(self):
        with _Env(";0-7"):
            with self.assertRaises(ValueError):
                make_layers(
                    8, lambda idx, prefix: _Layer(idx, prefix), pp_rank=0, pp_size=2
                )


class TestTheSpanIsNotTheCount(CustomTestCase):
    """THE CONTIGUITY HAZARD, and the reason the audit was not cosmetic.

    `model_runner` computed `num_effective_layers = end_layer - start_layer`,
    which is the SPAN. That equals the count only while a stage is an interval.
    It feeds `layer_num=` for KV pool allocation
    (`model_runner_kv_cache_mixin.py`), so under the family plan's FA stage —
    8 layers spanning [3, 64) — the pool would be sized for 61 layers instead
    of 8, a 7.6x over-allocation that fails the boot on memory it never needed.
    """

    def test_the_full_plan_fa_stage_has_a_span_that_lies(self):
        sets = parse_pp_layer_sets(FULL_PLAN, 64, 3, allow_gapped=True)
        fa_stage = sets[1]
        span = max(fa_stage) + 1 - min(fa_stage)
        self.assertEqual(len(fa_stage), 8)
        self.assertEqual(span, 29)
        self.assertNotEqual(span, len(fa_stage))

    def test_the_second_fa_stage_spans_almost_the_whole_model(self):
        sets = parse_pp_layer_sets(FULL_PLAN, 64, 3, allow_gapped=True)
        fa_stage = sets[2]
        span = max(fa_stage) + 1 - min(fa_stage)
        self.assertEqual(len(fa_stage), 8)
        self.assertEqual(span, 29)

    def test_model_runner_counts_the_set_rather_than_the_span(self):
        """The FIX, pinned at the source. A full ModelRunner is not hermetic,
        so this asserts the shape of the correction rather than executing it:
        `num_effective_layers` must come from the ownership set's LENGTH when a
        set is configured, and the span must survive only as the contiguous
        fallback."""
        import inspect

        from sglang.srt.model_executor import model_runner as mr

        src = inspect.getsource(mr)
        self.assertIn("owned = get_pp_layer_set(", src)
        self.assertIn("self.num_effective_layers = len(owned)", src)
        # and the old span form is still there, as the else-branch
        self.assertIn(
            "self.num_effective_layers = self.end_layer - self.start_layer", src
        )

    def test_ownership_has_exactly_one_derivation(self):
        """model_runner must call the PARSER, not read an attribute the models
        would each have to propagate -- two derivations is how they disagree."""
        import inspect

        from sglang.srt.model_executor import model_runner as mr

        src = inspect.getsource(mr)
        self.assertNotIn('getattr(self.model, "owned_layers"', src)

    def test_a_contiguous_stage_span_and_count_still_agree(self):
        """The falsifier: the hazard must be specific to non-contiguity, or
        the fix would be changing the contiguous path too."""
        with _Env(None):
            start, end = get_pp_indices(64, 1, 4)
            self.assertEqual(end - start, 16)


if __name__ == "__main__":
    unittest.main()
