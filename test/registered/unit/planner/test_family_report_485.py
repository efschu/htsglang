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
