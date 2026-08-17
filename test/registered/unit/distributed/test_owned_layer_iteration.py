"""Iterating OWNED layers, rather than the span between the first and last.

The translation half of this work (`KVCache.local_slot`) fixed global -> local
INDEXING. This is the other half: ITERATION. `for i in range(self.start_layer,
self.end_layer)` is correct only while ownership is an interval. Under
`SGLANG_PP_LAYER_SET` the span is wider than the ownership -- a stage owning
[35, 39, ..., 63] has start 35 and end 64, so the range yields 29 ids of which
21 are not owned.

Why that is not merely wasteful: the unowned slots hold `PPMissingLayer`, whose
forward returns its FIRST argument. Model loops call layers as
`layer(positions, hidden_states, ...)` (or the keyword equivalent) and unpack
two values, so an invoked placeholder returns `positions` -- the wrong tensor,
or a failed unpack. Placeholders were never invoked before, because the loop
bounds and the ownership were the same thing; a gapped set is the first case
where the loop can reach one. That is corrected here at the source: the loop
iterates ownership, so a placeholder is still never invoked.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.layers.utils.common import PPMissingLayer
from sglang.srt.utils.common import owned_layer_ids
from sglang.test.test_utils import CustomTestCase

#: The family plan's second full-attention stage: 8 owned layers spanning 29.
FA_STAGE = [35, 39, 43, 47, 51, 55, 59, 63]


class _Layers(list):
    """Stands in for the ModuleList `make_layers` returns; only the
    `owned_layers` attribute is read."""

    owned_layers = None


def _layers(owned=None):
    lst = _Layers()
    if owned is not None:
        lst.owned_layers = frozenset(owned)
    return lst


class TestContiguousOwnershipIsUnchanged(CustomTestCase):
    """The default path must not merely agree -- it must be the same object."""

    def test_it_returns_the_identical_range(self):
        got = owned_layer_ids(_layers(), 12, 20)
        self.assertIsInstance(got, range)
        self.assertEqual(got, range(12, 20))

    def test_it_agrees_with_the_old_expression_everywhere(self):
        for start in range(0, 8):
            for end in range(start, start + 12):
                with self.subTest(start=start, end=end):
                    self.assertEqual(
                        list(owned_layer_ids(_layers(), start, end)),
                        list(range(start, end)),
                    )

    def test_an_empty_stage_stays_empty(self):
        self.assertEqual(list(owned_layer_ids(_layers(), 5, 5)), [])


class TestSetOwnershipIteratesOwnership(CustomTestCase):
    def test_it_yields_only_owned_layers(self):
        self.assertEqual(list(owned_layer_ids(_layers(FA_STAGE), 35, 64)), FA_STAGE)

    def test_it_diverges_from_the_span(self):
        got = list(owned_layer_ids(_layers(FA_STAGE), 35, 64))
        span = list(range(35, 64))
        self.assertEqual(len(got), 8)
        self.assertEqual(len(span), 29)
        self.assertEqual(sorted(set(span) - set(got)), [i for i in span if i % 4 != 3])

    def test_it_is_ascending_so_layer_order_is_preserved(self):
        """Execution order is semantic: layer 39 must run after 35. A set has
        no order, so the helper -- not the caller -- guarantees it."""
        got = list(owned_layer_ids(_layers(reversed(FA_STAGE)), 35, 64))
        self.assertEqual(got, sorted(got))
        self.assertEqual(got, FA_STAGE)

    def test_a_gap_before_the_tested_layer_is_where_span_and_set_diverge(self):
        owned = [3, 7, 11]
        self.assertEqual(list(owned_layer_ids(_layers(owned), 3, 12)), owned)
        self.assertNotEqual(
            list(owned_layer_ids(_layers(owned), 3, 12)), list(range(3, 12))
        )


class TestWhyTheSpanIsUnsafeNotJustSlow(CustomTestCase):
    """Characterises the failure the conversion prevents, so the cost of
    getting this wrong stays visible."""

    def test_a_placeholder_returns_its_first_argument_not_the_hidden_states(self):
        positions = torch.tensor([0, 1, 2])
        hidden = torch.ones(3, 4)
        got = PPMissingLayer()(positions, hidden)
        self.assertIs(got, positions)
        self.assertIsNot(got, hidden)

    def test_the_keyword_call_shape_fails_the_same_way(self):
        """qwen3_5 -- the family plan's own model -- calls with keywords."""
        positions = torch.tensor([0, 1, 2])
        hidden = torch.ones(3, 4)
        got = PPMissingLayer()(positions=positions, hidden_states=hidden)
        self.assertIs(got, positions)

    def test_unpacking_an_invoked_placeholder_raises(self):
        layer = PPMissingLayer(return_tuple=True)
        with self.assertRaises(ValueError):
            _a, _b = layer(torch.tensor([0, 1, 2]), torch.ones(3, 4))

    def test_the_placeholders_a_span_loop_would_invoke(self):
        span = set(range(35, 64))
        self.assertEqual(len(span - set(FA_STAGE)), 21)


class TestTheFamilyModelLoopIteratesOwnership(CustomTestCase):
    def test_qwen3_5_forward_does_not_iterate_the_raw_span(self):
        import inspect

        from sglang.srt.models import qwen3_5

        src = inspect.getsource(qwen3_5)
        self.assertNotIn("range(self.start_layer, self.end_layer)", src)
        self.assertIn("owned_layer_ids(", src)


if __name__ == "__main__":
    unittest.main()
