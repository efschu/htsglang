"""#715: the RADIX SHAPE reporter died on the crash it exists to explain.

Measured 2026-08-17 02:18:09, all three ranks, inside the OOM this walk was
built (#695) to diagnose:

    RADIX SHAPE: walk failed after 3 nodes (RuntimeError('Boolean value of
    Tensor with more than one value is ambiguous')). Partial: tokens=1,
    locked_nodes=1.

Cause: ``len(getattr(node, "value", ()) or ())``. ``node.value`` is a TENSOR,
and ``tensor or ()`` evaluates ``bool(tensor)``, which raises for any tensor
with more than one element. The walk therefore survived only nodes whose value
was empty or single-element -- i.e. an empty tree -- and fell over on the first
real one.

A diagnostic that only works when there is nothing to diagnose is not a
diagnostic. This pins the shape of the input that broke it.
"""

import unittest

import torch

from sglang.test.test_utils import CustomTestCase


class _Node:
    """Minimal stand-in with the two attributes the walk reads off a node."""

    def __init__(self, value, children=None, mamba_value=None):
        self.value = value
        self.children = children or {}
        self.mamba_value = mamba_value
        self.full_lock_ref = 0
        self.mamba_lock_ref = 0
        self.key = None


class TestRadixShapeSurvivesTensorValues715(CustomTestCase):
    def _summary(self, root):
        from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache

        cache = MambaRadixCache.__new__(MambaRadixCache)
        cache.root_node = root
        cache.page_size = 1
        return cache._shape_summary()

    def test_multi_element_tensor_value_does_not_break_the_walk(self):
        """THE SPECIMEN SHAPE. A node whose value is a multi-element tensor is
        the ordinary case in a live tree, and it is what raised."""
        child = _Node(torch.arange(512, dtype=torch.int64))
        root = _Node(torch.tensor([], dtype=torch.int64), {0: child})
        text = self._summary(root)
        self.assertNotIn("walk failed", text, text)
        self.assertIn("512", text, f"the 512 tokens must be counted: {text}")

    def test_counts_tokens_across_a_deeper_tree(self):
        leaf_a = _Node(torch.arange(100, dtype=torch.int64))
        leaf_b = _Node(torch.arange(30, dtype=torch.int64))
        mid = _Node(torch.arange(7, dtype=torch.int64), {0: leaf_a, 1: leaf_b})
        root = _Node(torch.tensor([], dtype=torch.int64), {0: mid})
        text = self._summary(root)
        self.assertNotIn("walk failed", text, text)
        self.assertIn("137", text, f"100+30+7 tokens expected: {text}")

    def test_none_and_empty_values_still_count_zero(self):
        """CAN-FAIL BOUNDARY: the fix must not turn a missing value into a
        crash of its own, nor start counting None as a token."""
        for val in (None, torch.tensor([], dtype=torch.int64)):
            with self.subTest(value=type(val).__name__):
                root = _Node(val, {0: _Node(torch.arange(5, dtype=torch.int64))})
                text = self._summary(root)
                self.assertNotIn("walk failed", text, text)

    def test_a_locked_multi_element_node_is_still_counted_as_locked(self):
        """The locked accounting is what the crash needed most: it is how an
        operator sees whether the evictable rows are reachable."""
        child = _Node(torch.arange(64, dtype=torch.int64))
        child.full_lock_ref = 1
        root = _Node(torch.tensor([], dtype=torch.int64), {0: child})
        text = self._summary(root)
        self.assertNotIn("walk failed", text, text)
        self.assertIn("64", text, text)


if __name__ == "__main__":
    unittest.main()
