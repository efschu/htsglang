"""#872 SIBLING: the watermark rung prices a probe miss as a measurement.

Same defect class as the flip writeback fence, different site. ``_node_rows``
(``kv_radix_watermark.py``) reads the row tensor duck-typed::

    value = getattr(node, "value", None)
    if value is None:
        return None

and ``None`` is also exactly what a genuinely empty node returns. So a tree
whose node type does not carry ``.value`` prices out at "0 rows evictable",
which is indistinguishable from "this tree has nothing to give up" -- and the
consumer in ``kv_backing_relief`` narrates that zero as *"Healthy if those rows
are genuinely live; a DEFECT if they are unaccounted"*, never able to resolve
which, because the probe miss that produced it is invisible from there.

IT IS LIVE, NOT HYPOTHETICAL. ``UnifiedTreeNode`` -- the node type of
``UnifiedRadixCache``, which is what this build actually binds -- has no
``.value`` attribute at all, at class or instance level; its payload sits at
``component_data[BASE_COMPONENT_TYPE].value``. So the #662 "evict recomputable
prefix before capping the pool" rung has always freed exactly zero rows on the
live cache, on every call, without ever saying so. ``SGLANG_KV_RADIX_EVICT_RELIEF``
defaults to on, so this is the default path, not an experimental one.

WHAT IS AND IS NOT FIXED HERE. The alarm is the fix; the rung stays a no-op.
Teaching ``_node_rows`` to read ``component_data`` would be a HARM, not a
half-measure: the primitives the module then reaches for (``_evict_leaf_node``,
``_delete_leaf``) are equally absent from ``UnifiedRadixCache``, whose eviction
is the multi-component ``evict(EvictParams)`` /
``_evict_component_and_detach_lru`` / ``_remove_leaf_from_parent`` shape. A
widened read with no matching actuator frees rows and never unlinks the node --
a dangling tree node over freed KV rows, which is the #718 silent-corruption
family and strictly worse than freeing nothing. Building that actuator needs a
boot to validate and is not attempted here.

Hermetic: no CUDA, no pool, no server.
"""

import unittest

from sglang.srt.managers import kv_radix_watermark
from sglang.srt.managers.kv_radix_watermark import (
    evictable_rows_above,
    tree_ceiling,
)
from sglang.test.test_utils import CustomTestCase


class _Rows:
    """A row tensor stand-in with the ``numel`` the module asks for."""

    def __init__(self, ids):
        self._ids = list(ids)

    def numel(self):
        return len(self._ids)

    def tolist(self):
        return list(self._ids)

    def __len__(self):
        return len(self._ids)

    def __iter__(self):
        return iter(self._ids)

    def max(self):
        return max(self._ids)


class _ReadableNode:
    """A node shaped like ``RadixCache``'s ``TreeNode``: rows at ``.value``."""

    def __init__(self, ids):
        self.value = _Rows(ids)
        self.children = {}
        self.lock_ref = 0


class _UnreadableNode:
    """A node shaped like ``UnifiedTreeNode``: rows NOT at ``.value``.

    Deliberately carries a real payload behind a different name, because the
    condition under test is "the rows exist and this module cannot see them",
    not "the node is empty".
    """

    def __init__(self, ids):
        self.component_data = {"base": _Rows(ids)}
        self.children = {}
        self.lock_ref = 0


class _Tree:
    def __init__(self, nodes):
        self.root_node = type("_Root", (), {"children": {}})()
        for i, node in enumerate(nodes):
            self.root_node.children[i] = node


class TestWatermarkUnreadableNodes(CustomTestCase):
    def setUp(self):
        # The alarm is once-per-class-pair per process; each test needs a
        # clean slate or the second one silently observes the first's mute.
        kv_radix_watermark._UNREADABLE_WARNED.clear()
        self.addCleanup(kv_radix_watermark._UNREADABLE_WARNED.clear)

    def test_unreadable_nodes_are_reported_not_priced_as_zero(self):
        """RED before the fix: silently ``(0, 0)`` and not one log record."""
        tree = _Tree([_UnreadableNode([10, 11]), _UnreadableNode([12])])
        with self.assertLogs(kv_radix_watermark.__name__, level="ERROR") as cm:
            rows, nodes = evictable_rows_above(tree, 0)
        joined = "\n".join(cm.output)
        self.assertIn("#872", joined)
        self.assertIn("UNREADABLE NODES", joined)
        self.assertIn("_UnreadableNode", joined)
        # The rung still frees nothing -- that half is deliberate, see the
        # module docstring. What changed is that it no longer does so silently.
        self.assertEqual((rows, nodes), (0, 0))

    def test_a_readable_tree_is_not_alarmed(self):
        """No crying wolf: the alarm must stay silent where rows ARE visible."""
        tree = _Tree([_ReadableNode([10, 11]), _ReadableNode([12])])
        with self.assertNoLogs(kv_radix_watermark.__name__, level="ERROR"):
            rows, nodes = evictable_rows_above(tree, 0)
        self.assertGreater(rows, 0)
        self.assertGreater(nodes, 0)

    def test_an_empty_tree_is_not_alarmed(self):
        """An empty tree pricing at zero is CORRECT, and must not alarm.

        This is the distinction the whole ticket turns on: the alarm fires on
        "there are nodes and none is readable", never on "there are no nodes".
        A gate that cannot tell those apart is the crying-wolf instrument that
        got the original condition ignored.
        """
        with self.assertNoLogs(kv_radix_watermark.__name__, level="ERROR"):
            rows, nodes = evictable_rows_above(_Tree([]), 0)
        self.assertEqual((rows, nodes), (0, 0))

    def test_a_partially_readable_tree_is_not_alarmed(self):
        """One readable node is enough: the rung can still do work."""
        tree = _Tree([_UnreadableNode([10]), _ReadableNode([11])])
        with self.assertNoLogs(kv_radix_watermark.__name__, level="ERROR"):
            evictable_rows_above(tree, 0)

    def test_the_alarm_is_once_per_pair_not_once_per_call(self):
        """A line per pricing call would bury the log it exists to inform."""
        tree = _Tree([_UnreadableNode([10])])
        with self.assertLogs(kv_radix_watermark.__name__, level="ERROR") as cm:
            for _ in range(5):
                evictable_rows_above(tree, 0)
        self.assertEqual(len(cm.output), 1, cm.output)

    def test_tree_ceiling_reports_it_too(self):
        """Both entry points into the walk, not just the pricing one."""
        tree = _Tree([_UnreadableNode([10])])
        with self.assertLogs(kv_radix_watermark.__name__, level="ERROR") as cm:
            self.assertEqual(tree_ceiling(tree), -1)
        self.assertIn("#872", "\n".join(cm.output))

    def test_the_live_node_type_is_one_of_the_unreadable_ones(self):
        """The specimen, pinned so this stops being an argument about a fake.

        If ``UnifiedTreeNode`` ever grows a ``.value``, or the module learns to
        read ``component_data``, this assertion is what says so out loud
        instead of leaving the alarm above testing only its own stand-in.
        """
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedTreeNode

        self.assertFalse(
            hasattr(UnifiedTreeNode, "value"),
            "UnifiedTreeNode now carries `.value`; kv_radix_watermark._node_rows "
            "may finally be able to read the live tree -- re-check whether the "
            "eviction primitives it reaches for exist too before removing the "
            "#872 alarm, or a freed row will be left under a linked node.",
        )


if __name__ == "__main__":
    unittest.main()
