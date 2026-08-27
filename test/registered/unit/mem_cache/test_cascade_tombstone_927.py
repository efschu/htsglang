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
"""#927: a cascade that frees Full's rows must tombstone them, whoever triggered it.

THE DEFECT, and it is an EXCLUSIVITY defect before it is an accounting one.
``full_component.evict_component`` frees the device rows on any cascade that
reaches it and deliberately leaves ``cd.value`` set, because ``free_swa`` has
to read it first; the tombstone is deferred to ``_cascade_evict``. That
tombstone asked whether the Full component was the TRIGGER -- a different
question from whether its rows were freed. A cascade triggered by MAMBA or SWA
(``mamba_component.py:529``, ``swa_component.py:441``) that reaches Full freed
its rows and never cleared them, so a LIVE TREE NODE went on naming ids the
allocator had already handed back. Those ids stay matchable, so a later prefix
hit can serve KV out of rows that have been reissued -- the #767 direction.

HOW IT SURFACED. The on-idle ledger looks for exactly this: it intersects the
free list with ``all_values_flatten()``, which reads this ``value`` and does
not test ``evicted``. The intersection can only be as large as the tree, and
until the mamba checkpoint grid stopped vetoing every anchor no prefix match
ever succeeded -- the tree stayed at ``evictable=1`` and the term measured
about one row, which is the 2c precedent that motivated it. The first real
8538-row cached prefix made it 8129 against a 120-row surplus, an 8009-row
DEFICIT, and PP0 raised 14 s after the first hit
(SPECIMEN-2026-08-27T0643Z-HIT-DOUBLE-OWNED-CRASH.txt). Old defect, newly
reachable.

WHAT IS PINNED HERE is the invariant itself, stated without reference to the
ledger that happened to notice it: after an eviction, no node may name a row
that has been freed. Both trigger directions, because fixing only the observed
one would leave the sibling live.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(__file__)

import inspect
import unittest

from sglang.test.test_utils import CustomTestCase


class TestTheTombstoneAsksAboutFreeingNotTriggering(CustomTestCase):
    """Source-level, because the condition IS the defect and it is one line."""

    def test_the_tombstone_is_not_gated_on_who_triggered(self):
        """RED BEFORE THE FIX: the guard read
        `trigger.component_type == BASE_COMPONENT_TYPE`, so a mamba- or
        SWA-triggered cascade skipped it."""
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        src = inspect.getsource(UnifiedRadixCache._cascade_evict)
        tomb = src.index("value = None")
        guard = src[:tomb]
        self.assertNotIn(
            "trigger.component_type == BASE_COMPONENT_TYPE\n        ):",
            guard,
            "the tombstone is still gated on the trigger's identity",
        )
        self.assertIn("base_rows_freed", guard)

    def test_the_flag_is_set_when_full_is_cascaded_into(self):
        """The other half: being cascaded into must count, or the fix only
        renames the old condition."""
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        src = inspect.getsource(UnifiedRadixCache._cascade_evict)
        self.assertIn("base_rows_freed = True", src)
        setter = src.index("base_rows_freed = True")
        evict_call = src.index("_evict_component_and_detach_lru")
        self.assertLess(
            evict_call,
            setter,
            "the flag must be set after the eviction that frees the rows",
        )

    def test_a_cascade_that_never_reaches_full_must_not_tombstone(self):
        """The direction that must NOT move. Clearing `value` on a node whose
        Full rows were never freed would strand live KV -- a deficit, the
        opposite and worse defect."""
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        src = inspect.getsource(UnifiedRadixCache._cascade_evict)
        init = src.index("base_rows_freed =")
        self.assertIn(
            "trigger.component_type == BASE_COMPONENT_TYPE", src[init : init + 120]
        )


class TestTheInstrumentReadsWhatTheTombstoneClears(CustomTestCase):
    """Why the ledger is the thing that noticed, stated so the two cannot be
    fixed apart by accident."""

    def test_all_values_flatten_does_not_filter_evicted(self):
        """`all_values_flatten` reads `value` with no `evicted` test, which is
        correct ONLY while an evicted node's value is tombstoned. If this ever
        starts filtering, the tombstone stops being load-bearing and this
        file's premise needs re-reading."""
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        src = inspect.getsource(UnifiedRadixCache.all_values_flatten)
        self.assertNotIn("evicted", src)
        self.assertIn("value", src)

    def test_the_live_reading_is_an_enumerated_intersection(self):
        """It returns known ids, not an estimate -- which is why 8129 was a
        real claim about real rows and not a tolerance to widen."""
        from sglang.srt.managers.scheduler_components.invariant_checker import (
            SchedulerInvariantChecker,
        )

        src = inspect.getsource(SchedulerInvariantChecker._live_double_claimed_rows)
        self.assertIn("free_rows & cached_rows", src)


if __name__ == "__main__":
    unittest.main()
