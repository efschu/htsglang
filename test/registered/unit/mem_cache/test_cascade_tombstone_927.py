# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#927: no live tree node may name a row the allocator has handed back.

WHAT THIS FILE IS, STATED PRECISELY, because its first two versions were both
mislabelled.

It is an INVARIANT PIN, not a red-first proof. It passes on the unmodified
tree and it is EXPECTED to: the defect it was written to catch does not exist.

The story, kept because the lesson outlived the bug:

1. The first version asserted on ``inspect.getsource`` strings and was checked
   in as "red-first proven". That proof was an artefact -- the mutant I ran had
   restored the original's exact multi-line formatting, which is what
   ``assertNotIn`` matched. A one-line mutant leaves all five green. A
   source-string test observes the source, not the system, and structurally
   cannot go red for a behavioural change.

2. The rewrite below builds a real cache on CPU and asserts on TREE STATE --
   and a one-line mutant back to the old condition STILL leaves it green. That
   is not a second test weakness. The two conditions are EQUIVALENT:
   ``_cascade_evict`` is reached on the DEVICE target with a non-BASE trigger
   only from ``mamba_component.py:529`` and ``swa_component.py:441``, both on
   INTERNAL nodes, where the priorities are "full=2 > swa=1 > mamba=0"
   (``tree_component.py:292``); the cascade admits a component only at
   ``eviction_priority <= trigger_priority``, so Full at 2 is unreachable from
   a trigger at 0 or 1. The leaf path never calls the function
   (``_evict_device_leaf`` loops components directly) and ``_evict_to_host``,
   the only path leaving a node in the tree, passes BASE as the trigger
   explicitly.

So the #927 fix commit was a no-op and has been reverted. What survives is the
INVARIANT these tests state, which is worth pinning on its own account and is
the property the on-idle ledger's ``double_owned`` term measures: after an
eviction, the intersection of the allocator's free list with the rows the tree
still names must be EMPTY. If a future change breaks it, this goes red -- which
is the only claim made for it.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(__file__)

import unittest

import torch

from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from sglang.test.test_utils import CustomTestCase

from test_unified_radix_cache_unittest import CacheConfig, build_fixture

BASE = ComponentType.FULL


def _leaf_of(cache):
    """The single leaf of a one-branch tree."""
    node = cache.root_node
    while node.children:
        node = next(iter(node.children.values()))
    return node


class TestACascadeThatFreesFullTombstonesIt(CustomTestCase):
    """The invariant, stated without reference to the ledger that noticed it:
    after an eviction, no live node may name a row that has been freed."""

    def _fixture(self):
        cfg = CacheConfig(
            page_size=1,
            components=(ComponentType.FULL, ComponentType.MAMBA),
            mamba_cache_size=8,
        )
        return build_fixture(cfg)

    def _insert(self, cache, allocator, tokens):
        value = allocator.alloc(len(tokens))
        self.assertIsNotNone(value, "fixture pool too small")
        cache.insert(
            InsertParams(
                key=RadixKey(list(tokens), None),
                value=value.to(dtype=torch.int64),
            )
        )
        return value

    def test_evicting_the_tree_leaves_no_node_naming_a_freed_row(self):
        """RED ON THE OLD CONDITION. Evict everything, then intersect the
        allocator's free list with the rows the tree still names. A node that
        kept a tombstone-less value puts its freed ids in both sets.

        This is exactly the population `_live_double_claimed_rows` counts as
        `double_owned`, computed here from the same two sources."""
        cache, allocator, _ = self._fixture()
        self._insert(cache, allocator, range(1, 33))

        cache.evict(EvictParams(num_tokens=10**6))

        cached = {int(v) for v in cache.all_values_flatten().tolist()}
        free = {int(v) for v in allocator.free_pages.tolist()}
        doubly_claimed = cached & free
        self.assertEqual(
            doubly_claimed,
            set(),
            "the tree still names rows the allocator has handed back; those "
            "ids stay matchable and a later hit would serve reissued KV",
        )

    def test_the_evicted_node_carries_no_full_value(self):
        """The same property read directly off the node, so a failure says
        WHICH half broke rather than only that the sets intersect."""
        cache, allocator, _ = self._fixture()
        self._insert(cache, allocator, range(1, 33))
        leaf = _leaf_of(cache)
        self.assertIsNotNone(leaf.component_data[BASE].value, "nothing to evict")

        cache.evict(EvictParams(num_tokens=10**6))

        node = cache.root_node
        while node.children:
            node = next(iter(node.children.values()))
            self.assertIsNone(
                node.component_data[BASE].value,
                "an evicted node kept its Full value; its rows are free and "
                "still named",
            )


class TestTheOppositeDirectionMustNotMove(CustomTestCase):
    """Clearing `value` on a node whose Full rows were NOT freed would strand
    live KV -- a deficit, the worse sign. Pinned so the fix cannot be widened
    into that."""

    def test_a_live_node_keeps_its_value(self):
        cfg = CacheConfig(page_size=1, components=(ComponentType.FULL,))
        cache, allocator, _ = build_fixture(cfg)
        value = allocator.alloc(16)
        cache.insert(
            InsertParams(
                key=RadixKey(list(range(1, 17)), None),
                value=value.to(dtype=torch.int64),
            )
        )
        leaf = _leaf_of(cache)
        cache.inc_lock_ref(leaf)

        cache.evict(EvictParams(num_tokens=10**6))

        self.assertIsNotNone(
            leaf.component_data[BASE].value,
            "a locked node was tombstoned; its KV is live and now unreachable",
        )
        cached = {int(v) for v in cache.all_values_flatten().tolist()}
        free = {int(v) for v in allocator.free_pages.tolist()}
        self.assertEqual(
            cached & free,
            set(),
            "a locked node's rows were freed while it still names them",
        )


if __name__ == "__main__":
    unittest.main()
