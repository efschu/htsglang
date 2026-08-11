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
"""#656 spec item 12: KV is a SPILL CLASS, not a fixed maximum.

    "ES GIBT KEIN FESTES MAX KV: KV selbst ist Spill-Klasse in den
     System-RAM. Im VRAM liegt zu jedem Zeitpunkt GENAU das, was gerade dort
     liegen muss, der Rest im System-RAM."

The KV pool already sits on a VA reservation (``swappable_backing=True``), so
the addresses never move and CUDA graphs survive a residency change (item 13).
What did not exist is the thing that makes shrinking it SAFE:

**THE ALLOCATOR CAP.** ``runtime_set_backing_rows`` unmaps physical pages
under a VA range that the allocator still believes it may hand out. The next
allocation above the watermark then touches unbacked memory, which is a FAULT
and not an error -- it takes every rank down. So the cap is not a refinement
of this rung; it is the difference between a build and a fake, and these tests
pin it first.

The cap is non-destructive on purpose. The #330 dial shrinks by DESTROYING the
live set (``tree_cache.reset()`` + ``allocator.resize()``), which cannot be
done under load. This withholds high ids from the FREE list instead: live
allocations are never touched, and ``available_size()`` falls out correct
because it is derived from the free list.

Hermetic: tensor-backed fakes for the allocator and the pool, no CUDA.
"""

from __future__ import annotations

import unittest

import torch

from sglang.srt.managers import kv_backing_relief as kbr

MIB = 1024 * 1024


class _FakeAllocator:
    """The free-list mechanics that matter, and no more.

    Mirrors ``TokenToKVPoolAllocator``: ids are ``arange(1, size+1)``, ``alloc``
    takes from the FRONT, ``free`` appends, and listeners are notified after
    the append.
    """

    def __init__(self, size: int):
        self.size = size
        self.page_size = 1
        self._free_listeners = []
        self.clear()

    def clear(self):
        self.free_pages = torch.arange(1, self.size + 1, dtype=torch.int64)
        self.release_pages = torch.empty((0,), dtype=torch.int64)
        for _on_free, on_clear in self._free_listeners:
            if on_clear is not None:
                on_clear()

    def register_free_listener(self, on_free, on_clear=None):
        self._free_listeners.append((on_free, on_clear))

    def available_size(self):
        return len(self.free_pages) + len(self.release_pages)

    def alloc(self, need_size: int):
        if need_size > len(self.free_pages):
            return None
        out = self.free_pages[:need_size]
        self.free_pages = self.free_pages[need_size:]
        return out

    def free(self, idx: torch.Tensor):
        self.free_pages = torch.cat((self.free_pages, idx))
        for on_free, _on_clear in self._free_listeners:
            on_free(idx)


class _FakePool:
    """A VA-reserved pool whose backing follows ``runtime_set_backing_rows``."""

    def __init__(self, rows: int, bytes_per_row: int = 4096, *, card=None):
        self.size = rows
        self.page_size = 1
        self._bytes_per_row = bytes_per_row
        self._card = card
        self.supports_backing_spans = True
        self.calls = []

    def runtime_set_backing_rows(self, rows: int) -> int:
        rows = int(rows)
        self.calls.append(rows)
        delta = self.size - rows
        self.size = rows
        released = max(0, delta) * self._bytes_per_row
        if self._card is not None:
            self._card.free += released
        return released


class _Card:
    def __init__(self, free_mib):
        self.free = free_mib * MIB

    def probe(self):
        return self.free


def _relief(pool, alloc, live=(), card=None, **kw):
    return kbr.KvBackingRelief(
        pool,
        alloc,
        live_slots_fn=lambda: torch.tensor(list(live), dtype=torch.int64),
        bytes_per_row=pool._bytes_per_row,
        probe=(card.probe if card is not None else None),
        **kw,
    )


class AllocatorCapTest(unittest.TestCase):
    """Step 4 of the safe-shrink sequence, and the one that makes it real."""

    def test_capping_withholds_high_ids_from_the_free_list(self):
        a = _FakeAllocator(1000)
        cap = kbr.KvRowCap(a)
        cap.engage(400)
        self.assertEqual(int(a.free_pages.max()), 400)
        self.assertEqual(a.available_size(), 400)

    def test_capping_does_not_touch_live_allocations(self):
        # The #330 dial resizes by clearing the tree and the allocator, which
        # cannot be done under load. Live rows must survive a cap untouched.
        a = _FakeAllocator(1000)
        live = a.alloc(300)
        kbr.KvRowCap(a).engage(400)
        self.assertEqual(int(live.min()), 1)
        self.assertEqual(int(live.max()), 300)

    def test_an_id_freed_above_the_cap_does_not_re_enter_the_free_list(self):
        # THE HOLE THIS CLOSES. Eviction does not compact: a freed high id
        # goes back on the free list at its original value, so a cap applied
        # once and never re-applied leaks unbacked ids straight back into
        # circulation and the next allocation faults.
        a = _FakeAllocator(1000)
        # Everything is allocated, so the cap has nothing to withhold at
        # engage time -- exactly the state in which the leak is invisible
        # until the high rows come back.
        high = a.alloc(1000)[900:]
        cap = kbr.KvRowCap(a)
        cap.engage(400)
        self.assertEqual(cap.withheld, 0)
        a.free(high)
        self.assertEqual(a.free_pages.numel(), 0)
        self.assertEqual(cap.withheld, 100)

    def test_a_clear_re_applies_the_cap_rather_than_dropping_it(self):
        # clear() rebuilds arange(1, size+1), which silently re-admits every
        # id above the watermark while the backing is still unmapped.
        a = _FakeAllocator(1000)
        cap = kbr.KvRowCap(a)
        cap.engage(400)
        a.clear()
        self.assertEqual(int(a.free_pages.max()), 400)

    def test_releasing_the_cap_returns_every_withheld_id(self):
        a = _FakeAllocator(1000)
        cap = kbr.KvRowCap(a)
        cap.engage(400)
        cap.release()
        self.assertEqual(a.available_size(), 1000)
        self.assertEqual(int(a.free_pages.max()), 1000)
        self.assertFalse(cap.engaged)

    def test_the_free_list_stays_sorted_so_low_ids_are_reused_first(self):
        # Allocation order decides whether the high-water mark tracks
        # occupancy. If freed ids landed at the back, the watermark would
        # ratchet up and this whole rung would stop paying.
        a = _FakeAllocator(1000)
        taken = a.alloc(100)
        cap = kbr.KvRowCap(a)
        cap.engage(500)
        a.free(taken)
        cap.release()
        nxt = a.alloc(1)
        self.assertEqual(int(nxt[0]), 1)


class WatermarkTest(unittest.TestCase):
    def test_never_shrinks_below_the_highest_live_row(self):
        # The precondition of shrink() is "rows above the new span must be
        # dead". Violating it is a fault, not an error.
        card = _Card(1100)
        pool = _FakePool(1000, card=card)
        a = _FakeAllocator(1000)
        r = _relief(pool, a, live=[10, 20, 880], card=card)
        r.free_up_to(4000 * MIB)
        self.assertGreater(pool.size, 880)

    def test_a_full_pool_yields_nothing_rather_than_faulting(self):
        card = _Card(1100)
        pool = _FakePool(1000, card=card)
        a = _FakeAllocator(1000)
        r = _relief(pool, a, live=[999], card=card)
        self.assertEqual(r.free_up_to(4000 * MIB), 0)
        self.assertEqual(pool.calls, [])

    def test_frees_only_as_many_rows_as_the_ask_needs(self):
        # A provider that dumped the whole pool on a small ask would pay a
        # full restore for a few MiB.
        card = _Card(1100)
        pool = _FakePool(100000, bytes_per_row=MIB // 100, card=card)
        a = _FakeAllocator(100000)
        r = _relief(pool, a, live=[5], card=card)
        r.free_up_to(100 * MIB)
        # 100 MiB / (MIB/100) = 10000 rows, not all 100000.
        self.assertGreater(pool.size, 80000)


class HonestAccountingTest(unittest.TestCase):
    def test_reports_the_measured_driver_delta_not_the_pools_claim(self):
        # With SGLANG_FLIP_SEAM_RETAIN_HANDLES the arena UNMAPS without
        # releasing, so the pool's returned byte count is address space and
        # NVML never moves. The ledger law says price from what the driver
        # gave back, so the provider measures instead of believing.
        card = _Card(1100)
        pool = _FakePool(100000, bytes_per_row=MIB // 10, card=None)  # card unmoved
        a = _FakeAllocator(100000)
        r = _relief(pool, a, live=[5], card=card)
        self.assertEqual(r.free_up_to(500 * MIB), 0)

    def test_a_cap_that_bought_nothing_is_released_again(self):
        # Holding a cap that yielded no driver bytes would cost capacity for
        # nothing -- the worst of both directions.
        card = _Card(1100)
        pool = _FakePool(100000, bytes_per_row=MIB // 10, card=None)
        a = _FakeAllocator(100000)
        r = _relief(pool, a, live=[5], card=card)
        r.free_up_to(500 * MIB)
        self.assertEqual(a.available_size(), 100000)

    def test_recover_restores_both_the_backing_and_the_cap(self):
        card = _Card(1100)
        pool = _FakePool(100000, bytes_per_row=MIB // 10, card=card)
        a = _FakeAllocator(100000)
        r = _relief(pool, a, live=[5], card=card)
        r.free_up_to(500 * MIB)
        self.assertLess(pool.size, 100000)
        r.recover()
        self.assertEqual(pool.size, 100000)
        self.assertEqual(a.available_size(), 100000)

    def test_recover_without_a_prior_shrink_is_a_no_op(self):
        card = _Card(1100)
        pool = _FakePool(1000, card=card)
        a = _FakeAllocator(1000)
        r = _relief(pool, a, live=[5], card=card)
        self.assertEqual(r.recover(), 0)
        self.assertEqual(pool.calls, [])

    def test_an_unsupported_pool_is_inert_rather_than_raising(self):
        class _Plain:
            size = 1000
            page_size = 1
            _bytes_per_row = 4096

        card = _Card(1100)
        r = _relief(_Plain(), _FakeAllocator(1000), live=[5], card=card)
        self.assertEqual(r.free_up_to(500 * MIB), 0)


class FlushMustNotTouchUnbackedRowsTest(unittest.TestCase):
    """The fault this rung would otherwise have made reachable.

    A KV buffer is VA-sized; its backing is not. ``zero_kv_data_buffers``
    zeroed the WHOLE tensor, so once residency follows the load, a
    ``/flush_cache`` writes into unmapped address space --
    ``cudaErrorIllegalAddress``, which kills every rank rather than raising.
    It was unreachable while the only shrink path destroyed the live set and
    never ran under load; it stops being unreachable here. And the corridor
    procedure itself calls ``/flush_cache`` before every idle reading, so the
    two would have met on the acceptance run.
    """

    class _Pool:
        def __init__(self, rows, limit):
            self.k_buffer = [torch.ones(rows, 4)]
            self.v_buffer = [torch.ones(rows, 4)]
            self.safe_zero_rows = limit

    def test_only_the_backed_rows_are_zeroed(self):
        from sglang.srt.mem_cache.memory_pool import zero_kv_data_buffers

        pool = self._Pool(1000, 400)
        zero_kv_data_buffers(pool)
        self.assertEqual(float(pool.k_buffer[0][:400].sum()), 0.0)
        # Beyond the watermark the pages are unmapped; the test's proxy for
        # "was not written" is that the ones survive.
        self.assertEqual(float(pool.k_buffer[0][400:].sum()), 2400.0)

    def test_a_fully_backed_pool_is_still_zeroed_whole(self):
        from sglang.srt.mem_cache.memory_pool import zero_kv_data_buffers

        pool = self._Pool(1000, None)
        zero_kv_data_buffers(pool)
        self.assertEqual(float(pool.k_buffer[0].sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
