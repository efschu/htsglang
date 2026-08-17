# Copyright 2026 SGLang Team
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
"""#698: the admissibility probe must ask the cache the question it answers.

THE WEDGE. 2026-08-16 16:23:10, serving stopped for 54 minutes with health
returning 200 the whole time. 325 consecutive lines of::

    PHASE-POLICY holding in pp: BOTH BLOCKED: nothing can run in the pp layout
    and the target cannot admit either (0 req resident, 10495392 tok pending)

Last real batch at 16:23:11: ``full token usage: 1.00``, ``#running-req: 0``,
``mamba usage: 0.17``. The entire KV pool was radix cache with ZERO resident
requests -- every row of it unlocked and evictable -- while ten million tokens
of prefill queued behind it and three GPUs sat at 0%.

THE CAUSE, and it is one swallowed exception. ``Scheduler._post_evict_rows``
asks ``tree_cache.evictable_size()``. On ``MambaRadixCache`` that method does
not return a number::

    def evictable_size(self) -> Tuple[int, int]:
        # Note: use full_evictable_size() and mamba_evictable_size() instead.
        raise NotImplementedError

The probe caught the exception and used ``evictable = 0``, so it returned
``available`` alone. At usage 1.00 that is ~0, which made every admissibility
question answer "no":

    pp cannot admit (rows < one chunk)   -> nothing_can_run = True
    tp cannot decode (0 resident)        -> target_admissible = False
    => BOTH BLOCKED, which declines the flip and returns

and BOTH BLOCKED refuses BEFORE ``alloc_token_slots``, so the allocator was
never reached, so eviction never ran, so the unlocked cache was never freed.
The receipt named it "an evict trigger" while no evict could occur. Confirmed
by absence in the specimen: zero ``RADIX SHAPE``, zero ``Out of memory``, zero
``EVICTION UNDER-DELIVERED`` -- the allocation path was never entered at all.

THE SAME TRAP IS ALREADY DOCUMENTED IN THIS TREE, at
``mem_cache/common.py:411-425``::

    `evictable_size()` RAISES NotImplementedError on MambaRadixCache and
    SWARadixCache -- both split the count in two and say so ... Ask for the
    full-attention count first and fall back only for the flat classes.

That comment was read on the same day, while diagnosing #694, and not applied
to this function. ``_post_evict_rows``' own docstring even states the failure
it then committed: "An admissibility answer computed from ``available`` alone
would call that layout unusable and be wrong by 151040 rows."

WHY IT SURFACED WHEN IT DID. The bug shipped with #688's admissibility
simulation and needed ``usage == 1.00`` to bite. #696's floor repair shrank the
pool by 39,504 tokens, so full occupancy arrived sooner -- the wedge began 12
minutes after that boot. #696 EXPOSED this; it did not cause it.

A SWALLOWED EXCEPTION THAT RETURNS A PLAUSIBLE NUMBER IS THE DANGEROUS SHAPE:
zero is a legal row count, so nothing downstream could tell "the cache has
nothing" from "the cache was never asked".
"""

import unittest

from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5)


class _MambaLikeCache:
    """Splits the count in two and refuses the flat accessor, as the real one does."""

    def __init__(self, full=150_000, mamba=6):
        self._full = full
        self._mamba = mamba

    def evictable_size(self):
        raise NotImplementedError

    def full_evictable_size(self):
        return self._full

    def mamba_evictable_size(self):
        return self._mamba


class _FlatCache:
    def __init__(self, n=1234):
        self._n = n

    def evictable_size(self):
        return self._n


class _Alloc:
    def __init__(self, avail):
        self._avail = avail

    def available_size(self):
        return self._avail


class _StandIn:
    def __init__(self, avail, tree):
        self.token_to_kv_pool_allocator = _Alloc(avail)
        self.tree_cache = tree


def _rows(avail, tree):
    return Scheduler._post_evict_rows(_StandIn(avail, tree))


class TheProbeMustReachASplitCountCache(unittest.TestCase):
    def test_the_wedge_specimen(self):
        """usage 1.00, 0 resident: the pool is ALL evictable cache."""
        rows = _rows(0, _MambaLikeCache(full=150_000))
        self.assertGreaterEqual(
            rows,
            150_000,
            "the probe reported ~0 admissible rows while 150000 unlocked cache "
            "rows sat in the tree. That is the 2026-08-16 16:23 wedge: every "
            "admissibility question answered 'no', BOTH BLOCKED refused before "
            "alloc_token_slots, eviction never ran, and serving stopped for 54 "
            "minutes with health 200.",
        )

    def test_available_is_still_added(self):
        self.assertEqual(_rows(500, _MambaLikeCache(full=1000)), 1500)

    def test_a_flat_cache_still_works(self):
        """SWA/Mamba split the count; the plain classes do not. Both must work."""
        self.assertEqual(_rows(10, _FlatCache(90)), 100)


class ItNeverReportsAPlausibleFiction(unittest.TestCase):
    """Zero is a legal row count, which is what made this silent."""

    def test_a_cache_with_no_accessor_at_all_reports_available_only(self):
        class _Opaque:
            pass

        self.assertEqual(_rows(77, _Opaque()), 77)

    def test_no_tree_reports_available_only(self):
        self.assertEqual(_rows(42, None), 42)


if __name__ == "__main__":
    unittest.main()
