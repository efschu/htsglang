# SPDX-License-Identifier: Apache-2.0
"""#910: the WALKDOWN half of the LRU refresh has no other test.

THE GAP, AND HOW IT WAS FOUND. #862's faithfulness proof for the unified
radix suite planted four mutants against the 626 newly-lit tests. Three were
caught. The fourth was not:

    D  `_touch_node` WALKDOWN refresh off        0 red  -- a GENUINE gap

A mutant that survives is a statement about the suite, not about the mutant:
every node the tree walks past on an insert or a match could stop being
refreshed to MRU and 2488 passing tests would still pass. The consequence is
not cosmetic -- the LRU end is exactly what `evict` takes, so with the
refresh off the evictor picks the branch the walk just went through, which is
the branch most likely to be needed next.

WHAT IS PINNED HERE, all five parts of one contract
(`UnifiedRadixCache._touch_node` + `TreeComponent.refresh_lru`'s
`LRURefreshPhase.WALKDOWN` branch):

  1. a touched node moves to the MRU end of its component's LRU list, so the
     LRU end -- what the evictor reads -- is some OTHER node.  THIS IS THE
     ASSERTION THE #862 MUTANT SURVIVED.
  2. `last_access_time` advances.  This is the half of `_touch_node` that
     the mutant left intact, which is precisely why it must be pinned
     SEPARATELY: a test that only checked the timestamp would look like
     coverage of this function and catch nothing.
  3. the BASE (Full) component is skipped.  Full's own device LRU is driven
     by the match/insert end and by explicit `_for_each_component_lru`
     calls; refreshing it on every walked-past node here would be a second,
     unowned writer into it.
  4. the root node is never refreshed -- it is not in any LRU list, and
     `reset_node_mru` asserts membership, so this guard is load-bearing
     rather than tidy.
  5. a node whose component value is `None` is left where it is.  Evicted
     and tombstone nodes must stay at the LRU end; being walked past is not
     a reason to promote a node that holds nothing.

Plus the deliberate exception, pinned so it reads as a decision rather than
an omission: `SWAComponent.refresh_lru` makes WALKDOWN a no-op on purpose
(most walked ancestors are outside the sliding window and must stay
evictable; SWA refreshes window-bounded at MATCH_END/INSERT_END instead).

WHY THE SHIPPED METHOD IS BOUND TO A MINIMAL HOLDER rather than a fully
constructed `UnifiedRadixCache`. `_touch_node` reads exactly three things off
its holder -- `root_node`, `_components_tuple` and (through the component)
`lru_lists` -- and none of them require a KV pool, a device or a model. The
LRU lists, tree nodes and the `refresh_lru` implementation under test are the
REAL shipped classes; only the pool-owning shell around them is stood in for.
Same technique as the PP-mixin tests
(`test_pp_admission_wraparound_never_blocks.py::_make_holder`). This keeps
the module hermetic, so it runs at the desk on every gate rather than only in
a GPU window -- which is how the gap stayed open in the first place.
"""

import types
import unittest

import torch

from sglang.srt.mem_cache.unified_cache_components.swa_component import SWAComponent
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    BASE_COMPONENT_TYPE,
    ComponentType,
    LRURefreshPhase,
    TreeComponent,
)
from sglang.srt.mem_cache.unified_radix_cache import (
    UnifiedLRUList,
    UnifiedRadixCache,
    UnifiedTreeNode,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

#: The two components a node in these tests carries. MAMBA stands for "any
#: non-base component that does not override refresh_lru" -- it inherits
#: `TreeComponent.refresh_lru` verbatim, which is the code under test.
TREE_COMPONENTS = (ComponentType.FULL, ComponentType.MAMBA)


class _AuxComponent(TreeComponent):
    """A non-base component that uses the SHIPPED `TreeComponent.refresh_lru`.

    Subclassed rather than instantiating `MambaComponent` so this module owns
    no dependency on that component's constructor (pools, device, mamba
    geometry) and so a change there cannot make this contract test go green
    or red for an unrelated reason. `refresh_lru` is deliberately NOT
    overridden: the base implementation is the subject.

    The abstract surface is filled in with refusals -- `_touch_node` calls
    none of it, and a refusal makes that a checked fact instead of an
    assumption.
    """

    component_type = ComponentType.MAMBA

    def __init__(self, cache):
        self.cache = cache

    def _never(self, *args, **kwargs):
        raise AssertionError("_touch_node must not reach this component method")

    create_match_validator = _never
    redistribute_on_node_split = _never
    evict_component = _never
    drive_eviction = _never
    acquire_component_lock = _never
    release_component_lock = _never


class _SpyBaseComponent(TreeComponent):
    """Stands in for the Full component and RECORDS whether it was asked to
    refresh. `_touch_node` must skip it by `component_type`, so any call here
    is the contract being broken."""

    component_type = BASE_COMPONENT_TYPE

    def __init__(self, cache):
        self.cache = cache
        self.refresh_calls = []

    def refresh_lru(self, phase, node, root_node):
        self.refresh_calls.append((phase, node))

    def _never(self, *args, **kwargs):
        raise AssertionError("_touch_node must not reach this component method")

    create_match_validator = _never
    redistribute_on_node_split = _never
    evict_component = _never
    drive_eviction = _never
    acquire_component_lock = _never
    release_component_lock = _never


class _Holder:
    """The three attributes `_touch_node` actually reads, and nothing else."""

    def __init__(self):
        self.root_node = UnifiedTreeNode(TREE_COMPONENTS)
        self.lru_lists = {
            ComponentType.MAMBA: UnifiedLRUList(ComponentType.MAMBA, TREE_COMPONENTS),
            BASE_COMPONENT_TYPE: UnifiedLRUList(BASE_COMPONENT_TYPE, TREE_COMPONENTS),
        }
        self.base = _SpyBaseComponent(self)
        self.aux = _AuxComponent(self)
        self._components_tuple = (self.base, self.aux)
        self.touch = types.MethodType(UnifiedRadixCache._touch_node, self)

    def new_node(self, *, has_value: bool = True) -> UnifiedTreeNode:
        """A node in the aux LRU list, MRU-inserted the way `_add_new_node`
        and `_split_node` put nodes there."""
        node = UnifiedTreeNode(TREE_COMPONENTS)
        node.parent = self.root_node
        if has_value:
            node.component_data[ComponentType.MAMBA].value = torch.zeros(1)
        node.component_data[BASE_COMPONENT_TYPE].value = torch.zeros(1)
        self.lru_lists[ComponentType.MAMBA].insert_mru(node)
        return node

    def lru_order(self) -> list[int]:
        """Node ids from the LRU end towards MRU -- i.e. eviction order."""
        lru = self.lru_lists[ComponentType.MAMBA]
        out, node = [], lru.tail.lru_prev[lru._pt]
        while node is not lru.head:
            out.append(node.id)
            node = node.lru_prev[lru._pt]
        return out


class WalkdownRefreshesTheLRU(unittest.TestCase):
    def test_a_touched_node_leaves_the_lru_end(self):
        """THE #862 MUTANT D KILLER.

        Three nodes are inserted oldest-first, so `oldest` sits at the LRU
        end -- it is what `evict` would take next. Walking down through it
        must move it to MRU and leave a DIFFERENT node at the LRU end. With
        the WALKDOWN refresh removed (either by dropping the call in
        `_touch_node` or by making the WALKDOWN branch of
        `TreeComponent.refresh_lru` return early) nothing moves and the
        evictor still takes the node the walk just went through.
        """
        h = _Holder()
        oldest = h.new_node()
        middle = h.new_node()
        newest = h.new_node()
        lru = h.lru_lists[ComponentType.MAMBA]

        self.assertEqual(h.lru_order(), [oldest.id, middle.id, newest.id])
        self.assertIs(lru.get_lru_no_lock(), oldest)

        h.touch(oldest)

        self.assertEqual(
            h.lru_order(),
            [middle.id, newest.id, oldest.id],
            "WALKDOWN must move the touched node to the MRU end",
        )
        self.assertIs(
            lru.get_lru_no_lock(),
            middle,
            "after walking through it, the touched node must not still be "
            "the one the evictor takes",
        )

    def test_the_timestamp_is_bumped_as_well(self):
        """The half of `_touch_node` the surviving mutant left intact.

        Pinned separately and named as such: this assertion passes under
        mutant D. A module that checked only this would look like coverage
        of `_touch_node` and would have caught nothing.
        """
        h = _Holder()
        node = h.new_node()
        before = node.last_access_time
        h.touch(node)
        self.assertGreater(node.last_access_time, before)

    def test_the_base_component_is_not_refreshed_on_walkdown(self):
        """Full's device LRU has other owners; walkdown is not one of them."""
        h = _Holder()
        node = h.new_node()
        h.touch(node)
        self.assertEqual(
            h.base.refresh_calls,
            [],
            "the base component must be skipped by component_type, not by "
            f"luck: {h.base.refresh_calls}",
        )

    def test_the_root_node_is_never_refreshed(self):
        """The root is in no LRU list, and `reset_node_mru` asserts
        membership -- so this guard is load-bearing, not cosmetic. Touching
        the root must be inert rather than an AssertionError."""
        h = _Holder()
        node = h.new_node()
        h.touch(h.root_node)
        self.assertEqual(h.lru_order(), [node.id])
        self.assertEqual(h.base.refresh_calls, [])

    def test_a_valueless_node_keeps_its_place_at_the_lru_end(self):
        """Evicted / tombstone nodes must not be promoted by being walked
        past: they hold nothing, so the evictor should still reach them
        first."""
        h = _Holder()
        empty = h.new_node(has_value=False)
        held = h.new_node()
        lru = h.lru_lists[ComponentType.MAMBA]

        self.assertIs(lru.get_lru_no_lock(), empty)
        h.touch(empty)
        self.assertIs(
            lru.get_lru_no_lock(),
            empty,
            "a node with no component value must not be promoted to MRU",
        )
        self.assertEqual(h.lru_order(), [empty.id, held.id])


class SWAOptsOutOfWalkdownDeliberately(unittest.TestCase):
    def test_swa_walkdown_is_a_documented_no_op(self):
        """The exception that proves the rule, pinned so a future reader does
        not "fix" it into a refresh. Most ancestors a walk passes are outside
        the sliding window and must stay evictable; SWA does its
        window-bounded refresh at MATCH_END/INSERT_END instead. The check is
        that WALKDOWN returns without touching the cache at all -- a holder
        with NO swa LRU list would raise if it did."""
        h = _Holder()
        node = h.new_node()
        swa = object.__new__(SWAComponent)
        swa.cache = h
        self.assertNotIn(ComponentType.SWA, h.lru_lists)
        swa.refresh_lru(LRURefreshPhase.WALKDOWN, node, h.root_node)
        self.assertEqual(h.lru_order(), [node.id])


if __name__ == "__main__":
    unittest.main()
