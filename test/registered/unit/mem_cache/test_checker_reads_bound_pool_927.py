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
"""#927: the on-idle watchdog must audit the pool that is BOUND, not the one it
was built with.

THE DEFECT, and it is the watchdog itself. ``SchedulerInvariantChecker`` stored
``token_to_kv_pool_allocator`` as a dataclass field taken once at construction.
The phase flip REBINDS that allocator (``hicache_phase_binding._stamp`` sets
``token_to_kv_pool_allocator = incoming.allocator``, and ``phase_pools_for``
takes it from the incoming phase's OWN worker stack, so it is a different
object per phase) -- but ``readers_of`` names exactly three readers, scheduler,
tree_cache and cache_controller, and the checker is not one of them. Its own
docstring names the consequence: "a reader this function forgets is a reader
the rebind silently leaves behind."

WHAT THAT MANUFACTURES. After the first cutover, ``cache_controller.load``
allocates load-back rows from the INCOMING allocator while the ledger reads the
BOOT one. Both address the same id space, so those rows read as FREE to the
checker while the tree legitimately names them, and
``_live_double_claimed_rows`` reports the overlap as ``double_owned src=live``
-- in the magnitude of the loaded-back prefix, at the moment ``load_back`` fills
the nodes' ``value``.

SO #927 IS A FALSE POSITIVE. The rows were never doubly owned; the guard read
the wrong object. That is the indicator law in its purest form, and it is why
the fix touches no pool: the checker resolves the allocator PER ACCESS.

Hermetic: two real allocators over one id space, a real cache, real
``read_free_rows``/``_live_double_claimed_rows``. No CUDA.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(__file__)

import unittest

import torch

from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.kv_row_ownership import read_free_rows
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker,
)
from sglang.test.test_utils import CustomTestCase

from test_unified_radix_cache_unittest import CacheConfig, build_fixture

ROWS = 24


def _checker(boot_allocator, live_getter=None):
    """Only the two fields `_allocator` reads. The REAL method is bound off the
    class, so a change to it is caught here."""
    import types

    h = types.SimpleNamespace(
        token_to_kv_pool_allocator=boot_allocator,
        get_token_to_kv_pool_allocator=live_getter,
    )
    h._allocator = types.MethodType(SchedulerInvariantChecker._allocator, h)
    return h


class TheCheckerFollowsTheRebind927(CustomTestCase):
    def _two_phases(self):
        """Two independent allocators over the SAME id space -- what a phase
        cutover leaves behind. `build_fixture` builds a fresh allocator each
        call, which is exactly the per-phase worker stack shape."""
        cfg = CacheConfig(page_size=1, components=(ComponentType.FULL,))
        cache, incoming_alloc, _ = build_fixture(cfg)
        _, boot_alloc, _ = build_fixture(cfg)
        return cache, boot_alloc, incoming_alloc

    def test_the_checker_resolves_the_live_allocator(self):
        """RED BEFORE THE FIX: with no live getter the checker returns the
        object it was constructed with, whatever the flip did afterwards."""
        _, boot_alloc, incoming_alloc = self._two_phases()
        h = _checker(boot_alloc, live_getter=lambda: incoming_alloc)
        self.assertIs(
            h._allocator(),
            incoming_alloc,
            "the checker is still auditing the pool it was built with",
        )

    def test_without_a_getter_it_falls_back_to_the_field(self):
        """The compatibility direction: a construction outside the phase-flip
        boot keeps the pre-#927 behaviour rather than crashing."""
        _, boot_alloc, _ = self._two_phases()
        h = _checker(boot_alloc, live_getter=None)
        self.assertIs(h._allocator(), boot_alloc)

    def test_reading_the_stale_pool_manufactures_a_false_double_claim(self):
        """CHARACTERISATION -- the crash, reproduced as arithmetic.

        Rows allocated from the INCOMING allocator and held by the tree read as
        FREE against the BOOT allocator, so the overlap the ledger calls
        `double_owned` appears without a single row being doubly owned."""
        cache, boot_alloc, incoming_alloc = self._two_phases()

        value = incoming_alloc.alloc(ROWS)
        self.assertIsNotNone(value, "fixture pool too small")
        cache.insert(
            InsertParams(
                key=RadixKey(list(range(1, ROWS + 1)), None),
                value=value.to(dtype=torch.int64),
            )
        )

        stale = SchedulerInvariantChecker._live_double_claimed_rows(
            read_free_rows(boot_alloc), cache
        )
        live = SchedulerInvariantChecker._live_double_claimed_rows(
            read_free_rows(incoming_alloc), cache
        )

        self.assertEqual(
            stale,
            ROWS,
            "the stale reading did not manufacture the overlap; the "
            "characterisation no longer models the crash",
        )
        self.assertEqual(
            live,
            0,
            "against the BOUND allocator the same rows are not doubly claimed "
            "-- which is what makes the on-idle raise a false positive",
        )

    def test_the_scheduler_hands_the_checker_a_live_getter(self):
        """Wiring: the fix is inert unless the construction passes it."""
        import inspect

        from sglang.srt.managers.scheduler import Scheduler

        src = inspect.getsource(Scheduler.init_invariant_checker)
        self.assertIn("get_token_to_kv_pool_allocator=", src)


if __name__ == "__main__":
    unittest.main()
