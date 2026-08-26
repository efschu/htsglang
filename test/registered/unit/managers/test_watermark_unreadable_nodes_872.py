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

THE MODULE MAKES THREE SHAPE ASSUMPTIONS, AND ALL THREE ARE WRONG HERE.
``UnifiedTreeNode`` carries none of ``value``, ``lock_ref``, ``full_lock_ref``,
``mamba_lock_ref`` -- every one of them lives per component, in
``component_data[ct]``:

1. rows at ``node.value``       -> miss returns None  -> "nothing evictable"
2. refs at ``node.*lock_ref``   -> miss returns 0     -> "UNLOCKED"
3. eviction via ``_evict_leaf_node`` / ``_delete_leaf`` -> absent from
   ``UnifiedRadixCache``, whose eviction is the multi-component
   ``evict(EvictParams)`` / ``_evict_component_and_detach_lru`` /
   ``_remove_leaf_from_parent`` shape.

(1) and (3) miss toward SAFE -- nothing is offered, nothing is freed. (2)
misses toward UNSAFE: 0 means unlocked, so a node pinned by a running request
reads as evictable. Today (2) is MASKED by (1): no node survives the row probe,
so none reaches the lock check. They are one ``_node_rows`` fix apart from
evicting rows a request is still using.

WHAT IS AND IS NOT FIXED HERE. (1) is alarmed, not repaired -- the rung stays a
no-op. (2) is repaired, by failing safe: an unreadable lock state now reads as
LOCKED. That costs nothing today (the rung frees nothing anyway) and removes
the trap from under whoever fixes (1) next. Teaching ``_node_rows`` to read
``component_data`` on its own would be a HARM, not a half-measure: it would
arm (2) and, with (3) still absent, free rows and never unlink the node -- a
dangling tree node over freed KV rows, the #718 silent-corruption family.
Building the actuator needs a boot to validate and is not attempted here.

Hermetic: no CUDA, no pool, no server.
"""

import inspect
import re
import unittest

from sglang.srt.managers import kv_radix_watermark
from sglang.srt.managers.kv_radix_watermark import (
    evictable_rows_above,
    tree_ceiling,
)
from sglang.test.test_utils import CustomTestCase


def _safe_source(cls) -> bool:
    try:
        inspect.getsource(cls)
        return True
    except (OSError, TypeError):  # pragma: no cover - C or builtin base
        return False


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
        # No `value` AND no `lock_ref`: on `UnifiedTreeNode` both live per
        # component, at `component_data[ct].value` / `.lock_ref`. An earlier
        # version of this fake carried a bare `lock_ref = 0`, which made it
        # look readable to `_is_unlocked` and hid the very miss the suite is
        # about -- a fake built to the caller's assumption instead of to the
        # class, which is the mistake #872 was filed over.
        self.component_data = {"base": {"value": _Rows(ids), "lock_ref": 0}}
        self.children = {}


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

    def test_a_readable_tree_whose_nodes_are_all_EMPTY_is_not_alarmed(self):
        """#872b: the false positive the first form of this alarm carried.

        The original test was "did ANY node yield rows", so a tree of readable
        nodes that all happen to hold zero rows answered no and drew the
        UNREADABLE alarm -- a tree this module can read perfectly well,
        reported as a defect. An alarm that fires on a legitimate state is the
        crying-wolf instrument that got the silent condition ignored in the
        first place, which makes this the same mistake one level up.

        The type test cannot make this error: these nodes expose ``.value``,
        so they are READABLE and merely empty.
        """
        tree = _Tree([_ReadableNode([]), _ReadableNode([])])
        with self.assertNoLogs(kv_radix_watermark.__name__, level="ERROR"):
            rows, nodes = evictable_rows_above(tree, 0)
        self.assertEqual((rows, nodes), (0, 0))

    def test_an_evicted_node_is_readable_not_unreadable(self):
        """``value = None`` is an evicted node, a normal state, not a defect."""
        node = _ReadableNode([])
        node.value = None
        with self.assertNoLogs(kv_radix_watermark.__name__, level="ERROR"):
            evictable_rows_above(_Tree([node]), 0)
        self.assertTrue(kv_radix_watermark._node_rows_readable(node))

    def test_readability_is_decided_by_type_not_by_content(self):
        """The predicate itself, pinned directly.

        Readability must depend only on whether the row slot EXISTS. If this
        ever starts consulting the payload, the two failure modes above come
        back together.
        """
        self.assertTrue(kv_radix_watermark._node_rows_readable(_ReadableNode([])))
        self.assertTrue(kv_radix_watermark._node_rows_readable(_ReadableNode([1, 2])))
        self.assertFalse(
            kv_radix_watermark._node_rows_readable(_UnreadableNode([1, 2]))
        )

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

    def test_an_unreadable_LOCK_state_fails_safe_as_locked(self):
        """#872b: the miss whose neutral value points the UNSAFE way.

        ``_is_unlocked`` probes ``full_lock_ref`` / ``mamba_lock_ref`` /
        ``lock_ref`` with a default of ``0``, and 0 means UNLOCKED -- the
        PERMISSIVE answer. ``UnifiedTreeNode`` carries none of those names
        (its refs are per-component, at ``component_data[ct].lock_ref`` and
        ``.host_lock_ref``), so on the live cache every node, INCLUDING one
        pinned by a running request, reads as evictable.

        That is not the same as the row miss. A node whose rows cannot be read
        prices as zero and nothing happens -- silent, but safe. A node whose
        LOCKS cannot be read is offered up for eviction. Today it is masked:
        ``_node_rows`` misses first, so no node ever reaches the actuator. The
        two misses are one ``_node_rows`` fix apart from becoming an eviction
        of pinned rows, which is the #718 family.

        So the unreadable lock state must resolve to LOCKED. It can only make
        this rung more conservative, and this rung currently frees nothing at
        all, so the cost today is exactly zero.
        """
        self.assertFalse(
            kv_radix_watermark._is_unlocked(_UnreadableNode([10])),
            "a node whose lock state this module cannot read was reported "
            "UNLOCKED, i.e. evictable. The default on that getattr is 0, and "
            "0 means unlocked -- the permissive direction. Fail safe instead.",
        )

    def test_a_readable_lock_state_still_decides_normally(self):
        """Fail-safe must not become fail-always: real refs still rule."""
        free = _ReadableNode([10])
        free.lock_ref = 0
        self.assertTrue(kv_radix_watermark._is_unlocked(free))
        held = _ReadableNode([10])
        held.lock_ref = 1
        self.assertFalse(kv_radix_watermark._is_unlocked(held))

    def test_every_live_node_class_has_an_EXPLICIT_readability_verdict(self):
        """THE CHECK FOR THE SHAPE CLASS, which the name check cannot cover.

        ``test_every_probed_name_exists_on_every_bindable_cache`` (#872) asks
        "does this class carry the name the caller probes for". That question
        is the right one for a METHOD miss and useless here: the watermark's
        problem is not a name that could be aliased into place, it is that the
        node's payload has a DIFFERENT SHAPE -- one row slot assumed, N
        per-component slots present. There is no name to add.

        So this asks the only question that generalises: for every concrete
        tree-node class this build can construct, is its readability KNOWN? A
        class that is readable is fine. A class that is unreadable is fine
        PROVIDED it is listed below, because that listing is what proves
        someone looked. A NEW node class is neither, and lands here as a
        failure rather than as another silent zero in a boot log.
        """
        # The verdict table. Unreadable entries are not permission to ignore
        # the condition -- they are the record that the condition is known,
        # alarmed at runtime, and costed in the module docstring.
        expected_unreadable = {"UnifiedTreeNode"}

        found = {}
        for mod_name, cls_name in (
            ("sglang.srt.mem_cache.unified_radix_cache", "UnifiedTreeNode"),
            ("sglang.srt.mem_cache.radix_cache", "TreeNode"),
            ("sglang.srt.mem_cache.mamba_radix_cache", "MambaRadixTreeNode"),
        ):
            try:
                mod = __import__(mod_name, fromlist=[cls_name])
            except Exception:  # pragma: no cover - absent in this build
                continue
            cls = getattr(mod, cls_name, None)
            if cls is None:
                continue
            # Readability is a TYPE property, so the class body (plus bases)
            # answers it without constructing anything -- these constructors
            # want pools and components.
            src = "".join(
                inspect.getsource(base)
                for base in inspect.getmro(cls)
                if base is not object and _safe_source(base)
            )
            found[cls_name] = bool(
                re.search(r"self\.value\s*(?::[^=\n]+)?=", src) or hasattr(cls, "value")
            )

        self.assertTrue(found, "no tree-node class could be imported")
        unreadable = {name for name, ok in found.items() if not ok}
        self.assertEqual(
            unreadable,
            expected_unreadable & set(found),
            "the set of node classes kv_radix_watermark cannot read has "
            f"CHANGED (found unreadable: {sorted(unreadable)}, recorded: "
            f"{sorted(expected_unreadable & set(found))}). Either a new node "
            "class arrived without anyone deciding whether this rung can read "
            "it -- which is how the silent zero got here -- or one was fixed "
            "and this table should shrink.",
        )

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
