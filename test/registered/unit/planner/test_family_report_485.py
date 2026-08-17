# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#485 item 1: a BF16-resident family must be REPORTED as diverging.

THE DEFECT. `_per_family_formats` learned about per-family schemes from exactly
two places: ModelOpt's `quantized_layers`, and compressed-tensors'
`config_groups` **only when there is more than one group**. A checkpoint that
applies one scheme in one group and then EXCLUDES whole families through
`ignore` therefore reported no split at all, and
`checkpoint_compute_format_families` returned `{}`.

This fixes the reporting for configs whose groups name MODULE PATHS. It does
NOT make the serving checkpoint report, and that is worth stating here rather
than discovering later: `Qwen3.8-27B-INT8` declares its single group as
`targets: ["Linear"]` -- a module-CLASS selector, which `gemm_family_of_module`
maps to no family at all -- so the quantized side stays invisible, every
remaining signal comes from `ignore` and is bf16, and the caller's uniformity
check still collapses it to `{}`. Verified against the real checkpoint after
this fix. See NOTE_485 §7 item 1 for what closing it actually needs.

WHAT THIS FIX DOES NOT DO, deliberately. `GEMM_FAMILY_ATTN_GDN` spans BOTH
`self_attn` and `linear_attn`, and this checkpoint quantizes the first while
ignoring the second -- so that family is genuinely MIXED, not cleanly BF16.
Collapsing it with `_dominant` would pick by CONFIG-ENTRY count, which bears no
relation to how many layers are actually BF16 (48 linear vs 16 full attention
here). Inventing that number is the failure this module exists to avoid, so a
mixed family is reported as mixed in the description and left OUT of the dict
rather than given a fabricated key. Resolving it needs the per-layer census
(#371), and that is named as residue rather than guessed at here.
"""

from __future__ import annotations

import unittest

from sglang.srt.uneven_perf import (
    GEMM_FAMILY_ATTN_GDN,
    GEMM_FAMILY_MLP,
    GEMM_FAMILY_VOCAB,
    _per_family_formats,
)
from sglang.test.test_utils import CustomTestCase


def _ct(groups, ignore=()):
    """A compressed-tensors quantization_config in the shape the real one has."""
    return {
        "quant_method": "compressed-tensors",
        "format": "int-quantized",
        "config_groups": groups,
        "ignore": list(ignore),
    }


#: Shaped like the serving checkpoint's real group: W8A8, so `_ct_group_format`
#: yields "int8" (a real `_FORMAT_LANES` key) rather than the weight-only
#: "int8_a16", which has no lane table. A first version of this fixture omitted
#: `input_activations` and therefore asserted the contract against a key no
#: checkpoint here produces.
_INT8_GROUP = {
    "targets": ["Linear"],
    "weights": {"num_bits": 8, "type": "int", "strategy": "channel", "symmetric": True},
    "input_activations": {"num_bits": 8, "type": "int", "strategy": "token", "dynamic": True},
}


def _group_for(targets):
    group = dict(_INT8_GROUP)
    group["targets"] = list(targets)
    return group


class TestABf16ResidentFamilyIsReported(CustomTestCase):
    def test_an_entirely_ignored_family_is_reported_as_bf16(self):
        """The serving checkpoint's shape: one group, families excluded by ignore."""
        qc = _ct(
            {"group_0": _group_for(["re:.*mlp.*"])},
            ignore=["lm_head", "re:.*embed_tokens.*", "re:.*norm.*"],
        )
        families = _per_family_formats(qc)
        self.assertEqual(families.get(GEMM_FAMILY_VOCAB), "bf16")
        self.assertIn(GEMM_FAMILY_MLP, families)
        self.assertNotEqual(
            families[GEMM_FAMILY_MLP],
            "bf16",
            "the quantized family must keep its own key",
        )

    def test_the_divergence_survives_the_callers_uniformity_check(self):
        # checkpoint_compute_format_families drops the dict when every value is
        # the same, so a fix that reports one family only would still vanish.
        qc = _ct(
            {"group_0": _group_for(["re:.*mlp.*"])},
            ignore=["lm_head", "re:.*embed_tokens.*"],
        )
        families = _per_family_formats(qc)
        self.assertGreater(
            len(set(families.values())),
            1,
            "a single distinct value is discarded upstream as 'no split'",
        )

    def test_a_norm_only_ignore_reports_nothing(self):
        """REVERSE PIN: norms are not a GEMM family, so ignoring them is not a
        divergence and must not manufacture one."""
        qc = _ct(
            {"group_0": _group_for(["Linear"])},
            ignore=["re:.*norm.*", "re:.*conv1d.*"],
        )
        families = _per_family_formats(qc)
        self.assertNotIn("bf16", set(families.values()))


class TestUniformCheckpointsStayQuiet(CustomTestCase):
    """The migration promise: a uniform checkpoint keeps reading as it did."""

    def test_one_group_no_ignore_reports_no_split(self):
        qc = _ct({"group_0": _group_for(["Linear"])})
        families = _per_family_formats(qc)
        self.assertLessEqual(
            len(set(families.values())),
            1,
            "a checkpoint with one scheme everywhere has no split to report",
        )

    def test_an_empty_config_is_unchanged(self):
        self.assertEqual(_per_family_formats({}), {})


class TestAMixedFamilyIsNotInvented(CustomTestCase):
    """attn_gdn spans self_attn AND linear_attn; this checkpoint splits them."""

    def test_a_family_with_both_kinds_of_evidence_is_not_given_a_key(self):
        qc = _ct(
            {"group_0": _group_for(["re:.*self_attn.*", "re:.*mlp.*"])},
            ignore=["re:.*linear_attn.*"],
        )
        families = _per_family_formats(qc)
        self.assertNotIn(
            GEMM_FAMILY_ATTN_GDN,
            families,
            "a family that is part quantized and part bf16 has no single key; "
            "picking one by config-entry count would invent it",
        )
        # The unambiguous family is still reported.
        self.assertIn(GEMM_FAMILY_MLP, families)


class TestTheContractConsumersRelyOn(CustomTestCase):
    """#324 reads these values as _FORMAT_LANES keys."""

    def test_every_reported_value_is_a_known_format_key(self):
        from sglang.srt.uneven_perf import _FORMAT_LANES

        qc = _ct(
            {"group_0": _group_for(["re:.*mlp.*"])},
            ignore=["lm_head", "re:.*embed_tokens.*"],
        )
        for family, key in _per_family_formats(qc).items():
            with self.subTest(family=family):
                self.assertIn(key, _FORMAT_LANES, f"{family} -> {key!r}")


if __name__ == "__main__":
    unittest.main()


class TestClassSelectorIsAComplement(CustomTestCase):
    """#485 item 1a: `targets: ["Linear"]` names a module CLASS, not a path.

    `gemm_family_of_module` maps it to no family, so before this it contributed
    nothing and the quantized side of a single-group checkpoint was invisible.
    A class selector means "every GEMM family EXCEPT the ignored ones" -- a
    COMPLEMENT over the family enum, not an enumeration.
    """

    def test_a_class_selector_quantizes_every_unignored_family(self):
        qc = _ct(
            {"group_0": _group_for(["Linear"])},
            ignore=["lm_head", "re:.*embed_tokens.*"],
        )
        families = _per_family_formats(qc)
        self.assertEqual(families.get(GEMM_FAMILY_VOCAB), "bf16")
        self.assertEqual(families.get(GEMM_FAMILY_MLP), "int8")
        self.assertEqual(families.get(GEMM_FAMILY_ATTN_GDN), "int8")

    def test_a_regex_target_is_not_treated_as_a_class(self):
        # `re:.*router.*` maps to no family either, but it is a PATH pattern --
        # treating it as "all families" would quantize families it never named.
        qc = _ct({"group_0": _group_for(["re:.*router.*"])}, ignore=["lm_head"])
        families = _per_family_formats(qc)
        self.assertNotIn(GEMM_FAMILY_MLP, families)

    def test_a_dotted_path_is_not_treated_as_a_class(self):
        qc = _ct({"group_0": _group_for(["model.layers.0.foo"])}, ignore=["lm_head"])
        self.assertNotIn(GEMM_FAMILY_MLP, _per_family_formats(qc))


class TestMixedFamilyResolvedByLayers(CustomTestCase):
    """#485 item 1b: attn_gdn spans self_attn AND linear_attn.

    Family granularity has no single answer, but #371's per-layer counts do:
    on this checkpoint 48 layers are linear_attention (ignored -> bf16) and 16
    are full_attention (quantized). Resolving by LAYER is honest; resolving by
    config-entry count -- which is what `_dominant` does -- is not.
    """

    def test_the_layer_majority_decides(self):
        qc = _ct(
            {"group_0": _group_for(["re:.*self_attn.*", "re:.*mlp.*"])},
            ignore=["re:.*linear_attn.*"],
        )
        families = _per_family_formats(qc, layer_split={"gdn": 48, "full": 16})
        self.assertEqual(
            families.get(GEMM_FAMILY_ATTN_GDN),
            "bf16",
            "48 of 64 attention-family layers are bf16-resident GDN",
        )

    def test_the_other_majority_also_decides(self):
        # Reverse the split: the same config now resolves the other way.
        qc = _ct(
            {"group_0": _group_for(["re:.*self_attn.*", "re:.*mlp.*"])},
            ignore=["re:.*linear_attn.*"],
        )
        families = _per_family_formats(qc, layer_split={"gdn": 4, "full": 60})
        self.assertEqual(families.get(GEMM_FAMILY_ATTN_GDN), "int8")

    def test_without_a_layer_split_it_still_refuses_to_invent(self):
        qc = _ct(
            {"group_0": _group_for(["re:.*self_attn.*", "re:.*mlp.*"])},
            ignore=["re:.*linear_attn.*"],
        )
        self.assertNotIn(GEMM_FAMILY_ATTN_GDN, _per_family_formats(qc))


class TestKnownFormatsWithoutALane(CustomTestCase):
    """#485 item 1d: `int8_a16` is RECOGNISED, just unmeasured.

    `_FORMAT_LANES` deliberately omits lanes the serving path cannot take (its
    own comment says registering one 'would make the plan lie'), so weight-only
    int8 correctly has no row. What was missing is the #606 distinction: a
    reader of the fallback could not tell 'we know this format and have no
    measured lane' from 'we do not recognise this format at all'.
    """

    def test_int8_a16_is_declared_known_but_unmeasured(self):
        from sglang.srt.uneven_perf import FORMATS_WITHOUT_LANES

        self.assertIn("int8_a16", FORMATS_WITHOUT_LANES)

    def test_it_is_not_in_the_lane_table(self):
        from sglang.srt.uneven_perf import _FORMAT_LANES

        self.assertNotIn(
            "int8_a16",
            _FORMAT_LANES,
            "adding a lane the serving path cannot take would make the plan lie",
        )

    def test_the_warning_names_it_as_unmeasured_not_unknown(self):
        from sglang.srt.uneven_perf import rank_gemm_scores

        entries = [{"gemm_tflops": 100.0}]
        _scores, _labels, warnings = rank_gemm_scores(entries, "int8_a16")
        joined = " ".join(warnings)
        self.assertIn("int8_a16", joined)
        self.assertIn("weight-only", joined.lower())


class TestTheRealCheckpointNowReports(CustomTestCase):
    """#485 item 1 acceptance: `{}` was the proven RED here."""

    CKPT = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-yarn1.5"

    def setUp(self):
        import os

        if not os.path.isdir(self.CKPT):
            self.skipTest("serving checkpoint not on this box")

    def _report(self):
        from sglang.srt.uneven_perf import checkpoint_compute_format_families

        return checkpoint_compute_format_families(self.CKPT)

    def test_it_is_no_longer_empty(self):
        _fmt, _desc, families = self._report()
        self.assertTrue(families, "the divergence must now be visible")

    def test_the_bf16_resident_families_are_named(self):
        _fmt, _desc, families = self._report()
        self.assertEqual(families.get(GEMM_FAMILY_VOCAB), "bf16")
        self.assertEqual(families.get(GEMM_FAMILY_ATTN_GDN), "bf16")

    def test_the_quantized_families_keep_the_checkpoint_key(self):
        fmt, _desc, families = self._report()
        self.assertEqual(fmt, "int8")
        self.assertEqual(families.get(GEMM_FAMILY_MLP), "int8")

    def test_the_description_discloses_what_the_key_does_not_cover(self):
        # attn_gdn=bf16 is a MAJORITY statement: 48 GDN layers are bf16, but 16
        # full-attention layers are not, and a reader assuming uniformity would
        # mis-price them.
        _fmt, desc, _families = self._report()
        self.assertIn("48 linear_attention", desc)
        self.assertIn("16 full_attention", desc)
        self.assertIn("does NOT describe", desc)
