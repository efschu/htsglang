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
"""#936: does the pool census enumerate a HOST-BACKED node's rows?

THE RECONCILIATION THIS DECIDES. Two measured statements looked contradictory:

  (A) On a cold miss `cache_protected_len` is LEGITIMATELY the full prompt
      length -- `cache_unfinished_req` inserts each chunk and re-matches, so
      the tree really owns [0, cpl).
  (B) The 2i boot counted exactly those rows as "belong to NO ENUMERATED
      OWNER" (183x NO-SECOND-POOL), the deficit grew per request to 36824 and
      killed the boot at on_idle.

Both can hold only if the tree HOLDS the rows while the census does not
ENUMERATE its holding.

WHERE THAT WOULD COME FROM, and there is a precedent in the same family.
The census builds its owner set as::

    cached = set(tree.all_values_flatten().tolist())   # phase_flip_runtime:6131

and `all_values_flatten` (`unified_radix_cache.py:4115-4128`) appends
`component_data[BASE].value` ONLY when it is not None -- no `host_value`
fallback, no `evicted` handling. A node whose KV lives on the host tier
(`host_value` set, `value=None`) therefore contributes NOTHING. Its rows then
fall through the census's deficit arithmetic::

    leaked = set(range(1, size + 1)) - free - cached - withheld

into `leaked`. That is the same blind spot #927 resolved from the other side:
`_insert_helper_host` creates exactly such nodes (`:1788`), which is why
`double_owned` read 0 until `load_back` populated `value`. Invisible nodes
suppress `double_owned` AND inflate `leaked` -- one cause, opposite signs.

VERDICT: NEGATIVE -- see `WhatThisRulesOut936`. The blind spot is real, but
reaching it requires an eviction and the eviction FREES the device rows on the
way in, so no row is left unaccounted. The 36824 deficit is NOT this.

WHAT THIS FILE IS. A decider, not a fix. It states the enumeration's behaviour
on a host-backed node so the verdict rests on a run rather than on a reading --
the discipline #927's tombstone cost me. The hypothesis it was built to test
(deficit == census artefact, remedy == a #902-form declaration) is REFUTED
here, and the file is kept because a refuted hypothesis with a run behind it
is what stops the next reader spending a day on it again.

Hermetic: real `UnifiedRadixCache` on CPU, real allocator. No CUDA.
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
ROWS = 24


def _leaf_of(cache):
    node = cache.root_node
    while node.children:
        node = next(iter(node.children.values()))
    return node


def _census_cached(cache):
    """Exactly what phase_flip_runtime.py:6131 computes."""
    return set(cache.all_values_flatten().tolist())


class CensusSeesHostOnlyNodes936(CustomTestCase):
    def _tree_with_host_backed_leaf(self):
        """A node holding its KV on the host tier: `host_value` set, `value`
        None. That is the state `_evict_to_host` leaves in the tree (the node
        stays, its device rows are freed and tombstoned) and the state
        `_insert_helper_host` creates directly."""
        cfg = CacheConfig(page_size=1, components=(BASE,))
        cache, allocator, _ = build_fixture(cfg)
        value = allocator.alloc(ROWS)
        self.assertIsNotNone(value, "fixture pool too small")
        cache.insert(
            InsertParams(
                key=RadixKey(list(range(1, ROWS + 1)), None),
                value=value.to(dtype=torch.int64),
            )
        )
        leaf = _leaf_of(cache)
        rows = {int(v) for v in leaf.component_data[BASE].value.tolist()}

        # Demote to host: the tree keeps the node, the KV is named by
        # host_value, the device slot is no longer named by `value`.
        leaf.component_data[BASE].host_value = leaf.component_data[BASE].value
        leaf.component_data[BASE].value = None
        return cache, allocator, rows

    def test_a_device_backed_node_is_enumerated(self):
        """The control: while `value` is set the census sees the rows."""
        cfg = CacheConfig(page_size=1, components=(BASE,))
        cache, allocator, _ = build_fixture(cfg)
        value = allocator.alloc(ROWS)
        cache.insert(
            InsertParams(
                key=RadixKey(list(range(1, ROWS + 1)), None),
                value=value.to(dtype=torch.int64),
            )
        )
        rows = {int(v) for v in value.tolist()}
        self.assertTrue(
            rows <= _census_cached(cache),
            "the census cannot see a plain device-backed node either -- the "
            "premise of this whole file is wrong",
        )

    def test_the_verdict_a_host_backed_node_is_invisible_to_the_census(self):
        """THE DECIDER. If these rows are absent from `cached`, the census's
        deficit arithmetic charges them to `leaked` while the tree holds them
        -- and the 36824-row deficit is an enumeration artefact, not a leak."""
        cache, _, rows = self._tree_with_host_backed_leaf()

        cached = _census_cached(cache)
        invisible = rows - cached

        self.assertEqual(
            invisible,
            rows,
            "expected EVERY host-backed row to be missing from the census's "
            "owner set; got a partial result, which means the enumeration is "
            "doing something this file does not model",
        )

    def test_but_the_real_demotion_frees_them_first_so_no_deficit_arises(self):
        """THE VERDICT, AND IT IS NEGATIVE. The invisibility above is real and
        is NOT enough to orphan a row, because nothing ever tombstones a BASE
        value without freeing it first.

        The only BASE tombstone in the tree is `_cascade_evict`'s, and it runs
        only after `full_component.evict_component` has already called
        `_free_full(cd.value)` (`full_component.py:115-121`; the clear is
        deferred precisely so `free_swa` can read the value first). So a node
        that reaches `value=None` has ALREADY returned its device rows: they
        are in `free`, and `leaked = range - free - cached - withheld` does not
        claim them.

        Driven through the REAL eviction rather than by hand -- the
        hand-assembled state above is a mechanism demonstration, not a state
        this system reaches."""
        cfg = CacheConfig(page_size=1, components=(BASE,))
        cache, allocator, _ = build_fixture(cfg)
        value = allocator.alloc(ROWS)
        cache.insert(
            InsertParams(
                key=RadixKey(list(range(1, ROWS + 1)), None),
                value=value.to(dtype=torch.int64),
            )
        )
        rows = {int(v) for v in value.tolist()}

        cache.evict(EvictParams(num_tokens=10**6))

        free = {int(v) for v in allocator.free_pages.tolist()}
        cached = _census_cached(cache)
        leaked = set(range(1, allocator.size + 1)) - free - cached - set()

        self.assertEqual(
            rows & leaked,
            set(),
            "a really-evicted node's rows landed in the census deficit; then "
            "the host-only blind spot WOULD explain #935 and this verdict is "
            "wrong",
        )
        self.assertTrue(
            rows <= free,
            "the eviction did not return the rows to the allocator",
        )


class WhatThisRulesOut936(CustomTestCase):
    """Recorded so the next reader does not re-run this hypothesis.

    The census's blind spot is real: a host-backed node contributes nothing to
    `cached`. It is still NOT the source of #935's 36824-row deficit, because
    reaching that invisible state requires an eviction, and the eviction frees
    the device rows on the way in. Invisible node, but no unaccounted row.

    So the deficit is NOT a third census/checker false positive, and the
    remedy is NOT a census declaration. The hunt goes back to a path that
    loses rows for real."""

    def test_the_only_base_tombstone_is_preceded_by_a_free(self):
        import inspect

        from sglang.srt.mem_cache.unified_cache_components.full_component import (
            FullComponent,
        )

        src = inspect.getsource(FullComponent.evict_component)
        free_at = src.index("_free_full(cd.value)")
        self.assertIn("cd.value = None is deferred", src[free_at:])


if __name__ == "__main__":
    unittest.main()
