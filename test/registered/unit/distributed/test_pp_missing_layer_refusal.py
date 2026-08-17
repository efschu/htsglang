"""A placeholder that is INVOKED means the caller iterated the span.

`PPMissingLayer` stands in for a layer this stage does not own. Its forward
returns its first argument, which was harmless only because it was never
called: loop bounds and ownership were the same thing, so placeholders sat
outside `[start_layer, end_layer)`.

A gapped owned set puts placeholders INSIDE that interval. A loop over the span
then invokes one, and:

* `hidden_states, residual = layer(positions, hidden_states, ...)` raises a
  confusing unpack ValueError; but
* `hidden_states = layer(positions, hidden_states, forward_batch)` -- the shape
  in orion.py, persimmon.py, phi3_small.py -- SILENTLY replaces the hidden
  states with the positions tensor.

The silent case is why this is a refusal and not a comment. `owned_layer_ids`
fixes the loops; this makes any loop that was missed fail loudly and by name
instead of corrupting a forward pass. It is armed per placeholder at
construction, only under non-contiguous ownership, so the contiguous default
path keeps the pass-through behaviour byte-for-byte.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import os
import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.utils.common import PPMissingLayer
from sglang.srt.utils.common import make_layers
from sglang.test.test_utils import CustomTestCase


class _Layer(torch.nn.Module):
    def __init__(self, idx, prefix=""):
        super().__init__()
        self.idx = idx


class TestTheDefaultPassThroughIsUnchanged(CustomTestCase):
    """Contiguous ownership must behave exactly as before."""

    def test_positional_call_returns_the_first_argument(self):
        positions = torch.tensor([0, 1, 2])
        hidden = torch.ones(3, 4)
        self.assertIs(PPMissingLayer()(positions, hidden), positions)

    def test_keyword_call_returns_the_first_value(self):
        positions = torch.tensor([0, 1, 2])
        got = PPMissingLayer()(positions=positions, hidden_states=torch.ones(3, 4))
        self.assertIs(got, positions)

    def test_return_tuple_still_wraps(self):
        positions = torch.tensor([0, 1, 2])
        got = PPMissingLayer(return_tuple=True)(positions)
        self.assertEqual(len(got), 1)
        self.assertIs(got[0], positions)

    def test_it_is_still_an_identity_module(self):
        self.assertIsInstance(PPMissingLayer(), torch.nn.Identity)


class TestAnUnownedPlaceholderRefuses(CustomTestCase):
    def test_invoking_it_raises(self):
        layer = PPMissingLayer(unowned_layer_id=39)
        with self.assertRaises(RuntimeError):
            layer(torch.tensor([0, 1, 2]), torch.ones(3, 4))

    def test_the_error_names_the_layer_and_the_fix(self):
        layer = PPMissingLayer(unowned_layer_id=39)
        with self.assertRaises(RuntimeError) as cm:
            layer(torch.tensor([0, 1, 2]))
        msg = str(cm.exception)
        self.assertIn("39", msg)
        self.assertIn("owned_layer_ids", msg)

    def test_it_refuses_the_silent_shape_too(self):
        """The single-assignment shape (orion/persimmon/phi3_small) is the one
        that would NOT have crashed. It must raise now."""
        positions = torch.tensor([0, 1, 2])
        layer = PPMissingLayer(unowned_layer_id=7)
        with self.assertRaises(RuntimeError):
            _hidden = layer(positions, torch.ones(3, 4), None)

    def test_it_refuses_the_keyword_shape_too(self):
        layer = PPMissingLayer(unowned_layer_id=7)
        with self.assertRaises(RuntimeError):
            layer(positions=torch.tensor([0, 1, 2]), hidden_states=torch.ones(3, 4))


class TestMakeLayersArmsOnlyTheInteriorMode(CustomTestCase):
    def test_contiguous_placeholders_still_pass_through(self):
        mods, start, end = make_layers(
            8, lambda idx, prefix: _Layer(idx, prefix), pp_rank=0, pp_size=2
        )
        self.assertEqual((start, end), (0, 4))
        tail = [m for m in list(mods)[4:] if isinstance(m, PPMissingLayer)]
        self.assertEqual(len(tail), 4)
        positions = torch.tensor([0, 1, 2])
        for m in tail:
            self.assertIs(m(positions), positions)

    def test_set_placeholders_are_armed(self):
        with patch.dict(os.environ, {"SGLANG_PP_LAYER_SET": "0,4;1,2,3,5,6,7"}):
            mods, start, end = make_layers(
                8, lambda idx, prefix: _Layer(idx, prefix), pp_rank=0, pp_size=2
            )
        self.assertEqual((start, end), (0, 5))
        placeholders = {
            i: m for i, m in enumerate(mods) if isinstance(m, PPMissingLayer)
        }
        # The interior placeholders are the ones a span loop would reach.
        for i in (1, 2, 3):
            with self.subTest(layer=i):
                self.assertIsNotNone(placeholders[i].unowned_layer_id)
                with self.assertRaises(RuntimeError):
                    placeholders[i](torch.tensor([0, 1, 2]))

    def test_the_armed_error_names_its_own_layer(self):
        with patch.dict(os.environ, {"SGLANG_PP_LAYER_SET": "0,4;1,2,3,5,6,7"}):
            mods, _, _ = make_layers(
                8, lambda idx, prefix: _Layer(idx, prefix), pp_rank=0, pp_size=2
            )
        with self.assertRaises(RuntimeError) as cm:
            list(mods)[2](torch.tensor([0, 1, 2]))
        self.assertIn("2", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
