# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""#371: the family table must be read PER LAYER, not once for the model.

THE DEFECT, stated so the test can fail on it

``PerfCostModel`` builds its weight-family table as

    attn = full_layers * attn_layer
    mlp  = n_layers    * mlp_layer

with ONE uniform per-layer size each. A hybrid checkpoint corrects the
attention count through ``layer_types``. A HETEROGENEOUS-LAYER checkpoint --
Nemotron-NAS, the Puzzle architecture (``models/nemotron_nas.py``:
``config.block_configs[layer_idx]``, ``block_config.attention.no_op``,
``block_config.ffn.no_op``, ``ffn_mult``) -- declares no ``layer_types`` at
all, so it reaches the ``else`` branch that sets
``full_layers, gdn_layers = n_layers, 0``: **every no-attention layer is
counted as a full-attention layer and every no-FFN layer as an MLP layer.**

Why that is worse than a wrong byte total: the same table drives the uneven-TP
unit partition, #324's per-(rank, family) scores and the #348b cost library.
So the planner does not fail to load a model -- it hands ranks a share of a
family that is not there, and the plan looks reasonable.

The falsifier below is the #334 survey's claim turned into a test: it counts
what the OLD rule counts against what the blocks actually declare, shows the
size of the error, and shows the partition consequence. It is written to be
readable as evidence on its own, which is why it asserts on the concrete
numbers rather than only on the fixed behaviour.

CPU-only, hermetic: pure config arithmetic, no checkpoint (there is none on
this box -- see the boundary note at the end of the module).
"""

import unittest

from sglang.srt.uneven_perf import LayerFamilyCensus, layer_family_census
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _block(attn=True, ffn=True, ffn_mult=None):
    """One ``block_configs`` entry in the nemotron_nas.py shape."""
    return {
        "attention": {"no_op": not attn},
        "ffn": {"no_op": not ffn, **({"ffn_mult": ffn_mult} if ffn_mult else {})},
    }


def _puzzle_blocks():
    """A 12-layer Puzzle-shaped stack: 3 layers without attention, 2 without
    FFN, and a variable FFN width on the rest."""
    blocks = []
    for i in range(12):
        no_attn = i in (4, 7, 10)
        no_ffn = i in (5, 9)
        mult = 2.0 if i < 6 else 4.0  # the widths a NAS search produces
        blocks.append(_block(attn=not no_attn, ffn=not no_ffn, ffn_mult=mult))
    return blocks


class TestFalsifierOldRuleCountsWeightsThatDoNotExist(CustomTestCase):
    """What the pre-#371 rule got wrong, in numbers."""

    def setUp(self):
        self.blocks = _puzzle_blocks()
        self.n = len(self.blocks)
        self.census = layer_family_census({"block_configs": self.blocks}, self.n)

    def test_the_stack_really_is_heterogeneous(self):
        self.assertTrue(self.census.heterogeneous)
        self.assertEqual(self.census.n_layers, 12)

    def test_old_rule_over_counts_attention_layers(self):
        # OLD: no layer_types -> full_layers = n_layers = 12.
        old_attn_layers = self.n
        self.assertEqual(old_attn_layers, 12)
        # TRUTH: three layers declare attention.no_op.
        self.assertEqual(self.census.attn_layers, 9)
        # The error: three layers' worth of attention weights that do not
        # exist -- 33 % over-count on this stack.
        self.assertEqual(old_attn_layers - self.census.attn_layers, 3)

    def test_old_rule_over_counts_ffn_layers(self):
        old_ffn_layers = self.n
        self.assertEqual(self.census.ffn_layers, 10)
        self.assertEqual(old_ffn_layers - self.census.ffn_layers, 2)

    def test_old_rule_also_gets_the_ffn_WIDTH_wrong(self):
        # Even ignoring no_op, the old rule multiplies ONE width by every
        # layer. The real stack has 5 FFN layers at mult 2.0 (0-4; layer 5 is
        # no_op) and 5 at mult 4.0 (6,7,8,10,11; layer 9 is no_op) --
        # relative to the first real FFN (2.0) that is 5*1.0 + 5*2.0 = 15.0
        # layer-equivalents, not 12.
        self.assertAlmostEqual(self.census.ffn_width_factor, 15.0)
        self.assertNotAlmostEqual(self.census.ffn_width_factor, float(self.n))

    def test_the_two_errors_run_in_OPPOSITE_directions(self):
        # This is why the defect is hard to notice from a total: attention is
        # over-counted while the MLP mass is UNDER-counted (15.0 real vs 12
        # assumed layer-equivalents). A byte total can look plausible while
        # both families are wrong, and the per-family split -- which is what
        # the partitioner and #324's scores consume -- is wrong in both.
        attn_error = self.n - self.census.attn_layers          # +3 counted
        ffn_error = float(self.n) - self.census.ffn_width_factor  # -3 counted
        self.assertGreater(attn_error, 0)
        self.assertLess(ffn_error, 0)

    def test_the_partition_consequence(self):
        # The unit partition splits a family across ranks in proportion to
        # its mass. With attention over-counted by 3/12 and MLP under-counted
        # by 3/15, the ratio the partitioner sees between the two families is
        # off by a factor of 15/9 = 1.67 -- so a rank that should receive a
        # given share of attention units receives a materially different one,
        # on a plan that reports no error.
        assumed_ratio = self.n / float(self.n)                     # attn:mlp
        real_ratio = self.census.attn_layers / self.census.ffn_width_factor
        self.assertAlmostEqual(assumed_ratio, 1.0)
        self.assertAlmostEqual(real_ratio, 9.0 / 15.0)
        self.assertAlmostEqual(assumed_ratio / real_ratio, 15.0 / 9.0, places=6)


class TestCensusContract(CustomTestCase):
    def test_all_no_op_attention_is_zero_not_a_crash(self):
        # The aggregation must produce ZERO attention units, never a
        # divide-by-zero and never a silently reused global count.
        blocks = [_block(attn=False, ffn=True) for _ in range(4)]
        c = layer_family_census({"block_configs": blocks}, 4)
        self.assertEqual(c.attn_layers, 0)
        self.assertEqual(c.ffn_layers, 4)
        self.assertTrue(c.heterogeneous)

    def test_all_no_op_ffn_is_zero_width(self):
        blocks = [_block(attn=True, ffn=False) for _ in range(4)]
        c = layer_family_census({"block_configs": blocks}, 4)
        self.assertEqual(c.ffn_layers, 0)
        self.assertEqual(c.ffn_width_factor, 0.0)

    def test_uniform_ffn_mult_still_yields_the_layer_count(self):
        # Widths are RELATIVE: a stack uniform in ffn_mult must give exactly
        # its layer count, or every existing model's byte model would move.
        for mult in (1.0, 2.5, 8.0):
            blocks = [_block(ffn_mult=mult) for _ in range(7)]
            c = layer_family_census({"block_configs": blocks}, 7)
            self.assertAlmostEqual(c.ffn_width_factor, 7.0)
            self.assertFalse(c.heterogeneous)

    def test_objects_are_read_like_dicts(self):
        # A parsed HF config gives objects, raw JSON gives dicts.
        class _Sub:
            def __init__(self, **kw):
                self.__dict__.update(kw)

        class _Blk:
            def __init__(self, a, f):
                self.attention, self.ffn = a, f

        blocks = [
            _Blk(_Sub(no_op=True), _Sub(no_op=False, ffn_mult=2.0)),
            _Blk(_Sub(no_op=False), _Sub(no_op=False, ffn_mult=2.0)),
        ]
        c = layer_family_census({"block_configs": blocks}, 2)
        self.assertEqual(c.attn_layers, 1)
        self.assertEqual(c.ffn_layers, 2)

    def test_an_unreadable_block_degrades_toward_PRESENT(self):
        # Under-counting a block that IS there would size a pool too small
        # and fail a boot; over-counting is the error we are removing. When
        # the shape is unreadable the census must degrade toward today's
        # behaviour, not toward the smaller number.
        blocks = [{}, {"attention": None, "ffn": None}]
        c = layer_family_census({"block_configs": blocks}, 2)
        self.assertEqual(c.attn_layers, 2)
        self.assertEqual(c.ffn_layers, 2)


class TestUniformModelsAreByteIdentical(CustomTestCase):
    """Everything this fork serves today must not move by one parameter."""

    def test_no_block_configs_is_the_uniform_census(self):
        for n in (1, 12, 48, 64):
            c = layer_family_census({}, n)
            self.assertFalse(c.heterogeneous)
            self.assertTrue(c.uniform)
            self.assertEqual(c.attn_layers, n)
            self.assertEqual(c.ffn_layers, n)
            self.assertEqual(c.ffn_width_factor, float(n))

    def test_empty_block_configs_is_also_uniform(self):
        c = layer_family_census({"block_configs": []}, 32)
        self.assertFalse(c.heterogeneous)
        self.assertEqual(c.ffn_width_factor, 32.0)

    def test_the_mlp_factor_is_exactly_n_layers_when_uniform(self):
        # The expression the family table multiplies by. For a uniform stack
        # it must be the literal layer count, so `factor * mlp_layer` is the
        # same float as the old `n_layers * mlp_layer`.
        class _Stub:
            pass

        from sglang.srt.uneven_perf import PerfCostModel

        for n in (12, 48, 64):
            stub = _Stub()
            stub.n_layers = n
            stub.layer_census = layer_family_census({}, n)
            self.assertEqual(PerfCostModel._mlp_layer_factor(stub), float(n))
            self.assertIsInstance(PerfCostModel._mlp_layer_factor(stub), float)

    def test_a_model_without_a_census_attribute_keeps_the_old_number(self):
        # Defensive: older stubs and partially-constructed models must not
        # start returning something new.
        class _Stub:
            n_layers = 40

        from sglang.srt.uneven_perf import PerfCostModel

        self.assertEqual(PerfCostModel._mlp_layer_factor(_Stub()), 40.0)


class TestCensusIsADataclassContract(CustomTestCase):
    def test_fields_and_uniform_property(self):
        c = LayerFamilyCensus(
            n_layers=4, attn_layers=3, ffn_layers=4,
            ffn_width_factor=4.0, heterogeneous=True,
        )
        self.assertTrue(c.heterogeneous)
        self.assertFalse(c.uniform)


# BOUNDARY, stated rather than papered over: there is no Nemotron-Puzzle
# checkpoint on this box (the #334 inventory), so nothing here proves a BOOT.
# What is proven is the census arithmetic and that uniform models do not move.
# The follow-up is a download + boot ticket: fetch a Nemotron-NAS checkpoint,
# confirm the census matches its real block_configs, and compare the planned
# vector against a measured one. Until then this fix removes a known wrong
# number; it does not claim Puzzle coverage.


if __name__ == "__main__":
    unittest.main()
