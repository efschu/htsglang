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

import os
import unittest
import unittest.mock

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
        # THE HALF THIS TEST USED TO MISS (#485). Re-applying the cap was
        # already correct; BOOKING the re-apply was not. The clear hook was
        # the accumulating _apply, so the 600 ids above the cap were taken a
        # second time and concatenated onto the 600 already held.
        self.assertEqual(cap.withheld, 600)
        self.assertEqual(a.residency_withheld_slots, 600)

    def test_a_clear_does_not_double_book_the_idle_pool_invariant(self):
        # The measured crash, in one assertion. on_idle checks
        # available + ... + withheld == total; a doubled withheld reports a
        # pool memory leak on a pool that is perfectly intact. Metal
        # 2026-08-12: total=280000 available=267217 withheld=25566, i.e.
        # withheld exactly 2x the true 12783, raised inside on_idle right
        # after a /flush_cache.
        a = _FakeAllocator(1000)
        cap = kbr.KvRowCap(a)
        cap.engage(400)
        for _ in range(3):
            a.clear()
            self.assertEqual(
                a.available_size() + cap.withheld,
                1000,
                "available + withheld must equal the pool, on every clear",
            )

    def test_a_release_after_a_clear_leaves_no_duplicate_ids(self):
        # The worse half of the same bug: release() cats the withheld tensor
        # straight back into free_pages, so a double-booked id is handed out
        # TWICE -- two requests writing the same KV row, with no error.
        import torch

        a = _FakeAllocator(1000)
        cap = kbr.KvRowCap(a)
        cap.engage(400)
        a.clear()
        cap.release()
        self.assertEqual(a.free_pages.numel(), 1000)
        self.assertEqual(int(torch.unique(a.free_pages).numel()), 1000)

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

    def test_a_shrink_that_freed_nothing_is_never_undone_in_the_gate(self):
        # THE 2.5 GiB METAL BUG, 2026-08-11.
        #
        # Undoing a failed shrink means GROWING, and growing calls finalize ->
        # cuMemCreate. So the "cleanup" ALLOCATES, inside a gate that armed
        # because memory was short. Measured: free 3040 -> 460 MiB across one
        # refusal that reported "reclaimed 428 MiB", ending in
        # cuMemCreate CUDA_ERROR_OUT_OF_MEMORY. The relief provider was the
        # biggest consumer on the card.
        card = _Card(1100)
        pool = _FakePool(100000, bytes_per_row=MIB // 10, card=None)
        a = _FakeAllocator(100000)
        r = _relief(pool, a, live=[5], card=card)
        r.free_up_to(500 * MIB)
        self.assertEqual(len(pool.calls), 1, "the gate must not grow the pool")
        self.assertLess(pool.calls[0], 100000)

    def test_the_cap_stays_on_after_a_zero_byte_shrink(self):
        # The cap is the invariant "nothing above the watermark is handed
        # out". Lifting it while the pool is still shrunk is the fault the
        # cap exists to prevent -- and lifting it costs nothing to keep.
        card = _Card(1100)
        pool = _FakePool(100000, bytes_per_row=MIB // 10, card=None)
        a = _FakeAllocator(100000)
        r = _relief(pool, a, live=[5], card=card)
        r.free_up_to(500 * MIB)
        self.assertLess(a.available_size(), 100000)

    def test_a_pool_that_paid_nothing_is_not_asked_twice(self):
        # One failed shrink is evidence about the ARENA, not about this
        # moment. Retrying can only cost time and risk; on metal it cost
        # 2.5 GiB per gate arm because every arm repeated the attempt.
        card = _Card(1100)
        pool = _FakePool(100000, bytes_per_row=MIB // 10, card=None)
        a = _FakeAllocator(100000)
        r = _relief(pool, a, live=[5], card=card)
        r.free_up_to(500 * MIB)
        r.free_up_to(500 * MIB)
        r.free_up_to(500 * MIB)
        self.assertEqual(len(pool.calls), 1)

    def test_a_failed_recovery_keeps_the_cap_engaged(self):
        # Growing can fail for want of memory. When it does, the watermark
        # has not moved, so the cap must not move either: a capacity loss is
        # survivable, handing out unbacked ids is not.
        class _StuckPool(_FakePool):
            def runtime_set_backing_rows(self, rows):
                if rows > self.size:
                    raise RuntimeError("cuMemCreate failed: OUT_OF_MEMORY")
                return super().runtime_set_backing_rows(rows)

        card = _Card(1100)
        pool = _StuckPool(100000, bytes_per_row=MIB // 10, card=card)
        a = _FakeAllocator(100000)
        r = _relief(pool, a, live=[5], card=card)
        r.free_up_to(500 * MIB)
        self.assertEqual(r.recover(), 0)
        self.assertLess(a.available_size(), 100000)

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


class ReleaseGranularityTest(unittest.TestCase):
    """Release is extent-granular PER BUFFER, and that is coarse.

    The arena holds each of the 2*layer_num buffers at its own offset, and
    ``decommit_range`` frees only extents lying wholly above the keep point.
    A shrink is split across every buffer, so an ask for N bytes moves only
    N/n_buffers in each -- and below one commit chunk, nothing is released
    anywhere.

    Measured 2026-08-11 with a 256 MiB chunk: a 78262-row shrink asked about
    40 MiB of each of ~28 buffers, cleared no extent in any of them, and
    returned 0 while the log read like a working rung.
    """

    def test_a_small_ask_is_rounded_up_to_one_chunk_per_buffer(self):
        card = _Card(1100)
        pool = _FakePool(500000, bytes_per_row=15 * 1024, card=card)
        pool.backing_commit_chunk_bytes = 256 * MIB
        a = _FakeAllocator(500000)
        r = kbr.KvBackingRelief(
            pool,
            a,
            live_slots_fn=lambda: torch.tensor([5], dtype=torch.int64),
            bytes_per_row=15 * 1024,
            probe=card.probe,
            buffers=28,
        )
        r.free_up_to(489 * MIB)
        # 256 MiB x 28 buffers / 15 KiB per row = 489132 rows, so the ask must
        # grow far beyond the 33000 rows 489 MiB alone would suggest.
        self.assertLess(pool.calls[0], 500000 - 400000)

    def test_no_chunk_reported_means_no_rounding(self):
        card = _Card(1100)
        pool = _FakePool(500000, bytes_per_row=15 * 1024, card=card)
        a = _FakeAllocator(500000)
        r = kbr.KvBackingRelief(
            pool,
            a,
            live_slots_fn=lambda: torch.tensor([5], dtype=torch.int64),
            bytes_per_row=15 * 1024,
            probe=card.probe,
            buffers=28,
        )
        r.free_up_to(489 * MIB)
        self.assertGreater(pool.calls[0], 400000)


class WithheldCapacityIsANamedPostenTest(unittest.TestCase):
    """The second metal kill: the cap read as a pool leak.

    The scheduler's idle invariant is ``available + evictable + protected +
    session_held + uncached == total``. Withheld capacity is in none of those
    buckets, so the first boot that exercised the cap died at its first idle
    check with ``pool memory leak detected! [full] total=500000,
    available=419745`` -- 80255 slots that were exactly the cap.

    It is a NAMED term for the #486 reason: anything that durably removes pool
    slots must be named in this ledger, or the next unexplained delta gets
    attributed to the wrong holder.
    """

    def test_the_cap_publishes_its_size_to_the_allocator(self):
        a = _FakeAllocator(1000)
        cap = kbr.KvRowCap(a)
        cap.engage(400)
        self.assertEqual(a.residency_withheld_slots, 600)
        cap.release()
        self.assertEqual(a.residency_withheld_slots, 0)

    def test_it_is_published_in_tokens_not_ids_on_a_paged_allocator(self):
        # available_size() multiplies the free list by page_size, so a raw id
        # count would be wrong by exactly that factor on every paged lane.
        a = _FakeAllocator(1000)
        a.page_size = 16
        cap = kbr.KvRowCap(a)
        cap.engage(400)
        self.assertEqual(a.residency_withheld_slots, 600 * 16)

    def test_the_invariant_accepts_withheld_capacity(self):
        from sglang.srt.managers.scheduler_components.invariant_checker import (
            SchedulerInvariantChecker,
        )

        leak, msg = SchedulerInvariantChecker._check_pool_invariant(
            "full", 419745, 90, 0, 0, 500000, 0, 80165
        )
        self.assertFalse(leak, msg)

    def test_the_invariant_still_catches_a_real_leak(self):
        # The term must not become a licence: an unexplained shortfall with no
        # cap engaged is still a leak.
        from sglang.srt.managers.scheduler_components.invariant_checker import (
            SchedulerInvariantChecker,
        )

        leak, _ = SchedulerInvariantChecker._check_pool_invariant(
            "full", 419745, 90, 0, 0, 500000, 0, 0
        )
        self.assertTrue(leak)


class ChunklessArenaIsDisqualifiedTest(unittest.TestCase):
    """The root cause of the metal incident, pinned at the registration site.

    Without a commit chunk the arena holds ONE extent per buffer, and
    ``decommit_range`` releases only extents lying wholly above the keep
    point -- so a shrink to any watermark inside that extent releases exactly
    zero while still lowering ``pool.size``. Registering against such a pool
    produced a provider that consumed memory instead of freeing it.
    """

    class _Sched:
        def __init__(self, supports):
            pool = _FakePool(1000)
            pool.supports_backing_spans = supports
            self._pool = pool
            self.token_to_kv_pool_allocator = _FakeAllocator(1000)
            self.token_to_kv_pool_allocator.get_kvcache = lambda: pool

    def test_a_chunkless_pool_does_not_register_a_provider(self):
        # ENABLED explicitly: the rung is opt-in until its shrink target is a
        # collective minimum, and without this the assertion would pass on the
        # opt-in gate rather than on the chunk check it names.
        with unittest.mock.patch.dict(os.environ, {"SGLANG_KV_BACKING_RELIEF": "1"}):
            self.assertIsNone(
                kbr.kv_backing_provider(self._Sched(False), device_index=0)
            )

    def test_the_rung_is_on_by_default_now_that_the_target_is_collective(self):
        # It was opt-in for exactly one shift, while the shrink target was
        # still rank-local and ranks that disagreed about admission desynced
        # the PP group. The target is agreed by a MIN all-reduce now, and the
        # uniformity was measured on metal (347161 rows on all three ranks),
        # so the default faces the other way.
        # Asserted on the GATE, not on the returned object: these fakes have
        # no arena geometry, so the builder declines further down for an
        # unrelated and equally correct reason. The gate announces itself in
        # the log when it declines, so its silence is the evidence.
        with unittest.mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_KV_BACKING_RELIEF", None)
            with self.assertLogs(kbr.logger, level="DEBUG") as caught:
                kbr.logger.debug("marker")
                kbr.kv_backing_provider(self._Sched(True), device_index=0)
        self.assertFalse(
            [line for line in caught.output if "DISABLED" in line],
            "the env gate must not be what stops it any more",
        )

    def test_the_escape_hatch_still_turns_it_off(self):
        with unittest.mock.patch.dict(
            os.environ, {"SGLANG_KV_BACKING_RELIEF": "0"}, clear=False
        ):
            with self.assertLogs(kbr.logger, level="WARNING") as caught:
                self.assertIsNone(
                    kbr.kv_backing_provider(self._Sched(True), device_index=0)
                )
        self.assertTrue([line for line in caught.output if "DISABLED" in line])


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


class ProposalTraceReasonTest(unittest.TestCase):
    """#656 D5: the trace must not state a cause it did not evaluate.

    The first version of this diagnostic printed "the cheaper tier covers the
    whole gap" on EVERY non-shrinking path -- including the two where no gap
    is ever computed. A diagnostic that states a FALSE cause is worse than one
    that states none, because the next reader stops looking.
    """

    def _propose_and_capture(self, relief):
        with self.assertLogs(kbr.__name__, level="INFO") as cm:
            relief.propose(
                want_bytes=100 * MIB,
                floor_bytes=1024 * MIB,
                delta_bytes=256 * MIB,
                cheap_relief_bytes=0,
            )
        return "\n".join(cm.output)

    def test_an_exhausted_arena_says_so_instead_of_blaming_the_cheap_tier(self):
        """RESTATED 2026-08-15: the trace still names the ARENA rather than
        blaming the cheap tier, but it now names the FAILED TARGET too.

        Keying exhaustion to the level alone was self-locking -- a shrink that
        releases nothing leaves the level unchanged, so the marker marked the
        level the rung was stuck at and nothing could ever move it. Measured:
        47 s of declining with 72981 rows of slack in front of it. The refusal
        now has to say which ask it is refusing, so a deeper one can be
        recognised as a different question."""
        # A TIGHT card, so a shrink is actually wanted. With plenty free the
        # deficit is negative and "the cheaper tier covers the gap" is the
        # honest answer whatever the arena has done -- exhaustion only has
        # anything to say when the rung is being asked to pay.
        #
        # AND a small live-set gap, because SLACK now overrides the marker:
        # when the rows in front of the rung dwarf the ask, one earlier
        # failure is not decisive and it tries anyway. Here the pool is nearly
        # all live, so there is nothing to try with and the marker stands.
        pool = _FakePool(rows=1000)  # chunkless: no extent can clear, so the
        r = _relief(  # marker stands and slack is not evidence
            pool, _FakeAllocator(1000), live=(5,), card=_Card(100)
        )
        r._mark_exhausted(target=1)
        out = self._propose_and_capture(r)
        self.assertIn("returned no driver bytes at a shrink to", out)
        self.assertNotIn("the cheaper tier covers the whole gap", out)

    def test_no_slack_above_the_live_set_says_so(self):
        """floor_rows >= current: there is nothing this rung MAY give up."""
        pool = _FakePool(rows=10)
        r = _relief(pool, _FakeAllocator(10), live=(9,), card=_Card(2000))
        out = self._propose_and_capture(r)
        self.assertIn("no slack above the live set", out)
        self.assertNotIn("the cheaper tier covers the whole gap", out)

    def test_a_real_decline_still_blames_the_cheap_tier_correctly(self):
        pool = _FakePool(rows=100000)
        r = _relief(pool, _FakeAllocator(100000), live=(5,), card=_Card(64000))
        out = self._propose_and_capture(r)
        self.assertIn("the cheaper tier covers the whole gap", out)


class AbstainIsNeverSilentTest(unittest.TestCase):
    """HANDOFF_678 §4.0: an ABSTAIN was still silent, and it is the worst case.

    ``propose`` has four early ``ABSTAIN`` returns above the trace, so a rank
    that cannot take part at all logged NOTHING while doing it. That is not a
    decline -- :func:`collective_kv_target` cancels the WHOLE group's decision
    when any single ``current`` is not positive, so ONE silent abstention turns
    spec item 12 off node-wide and looks exactly like a rung that declined on
    the arithmetic.

    D5 fixed the two skip paths BELOW the early returns. These are the returns
    themselves, and each one must name which precondition it failed, loudly
    enough that a reader hunting a dead rung finds it instead of the four terms
    of a deficit that was never computed.
    """

    ASK = dict(
        want_bytes=100 * MIB,
        floor_bytes=1024 * MIB,
        delta_bytes=256 * MIB,
        cheap_relief_bytes=0,
    )

    def _capture(self, relief, level="WARNING"):
        with self.assertLogs(kbr.__name__, level=level) as cm:
            out = relief.propose(**self.ASK)
        return out, "\n".join(cm.output)

    def test_an_unsupported_pool_names_the_missing_entry_point(self):
        pool = _FakePool(rows=1000)
        pool.runtime_set_backing_rows = None  # a pool without the entry point
        r = _relief(pool, _FakeAllocator(1000), live=(5,), card=_Card(2000))
        proposal, out = self._capture(r)
        self.assertEqual(proposal, kbr.ABSTAIN)
        self.assertIn("ABSTAIN", out)
        self.assertIn("runtime_set_backing_rows", out)

    def test_a_zero_bytes_per_row_says_so_rather_than_dividing_by_it(self):
        pool = _FakePool(rows=1000, bytes_per_row=0)
        r = _relief(pool, _FakeAllocator(1000), live=(5,), card=_Card(2000))
        proposal, out = self._capture(r)
        self.assertEqual(proposal, kbr.ABSTAIN)
        self.assertIn("ABSTAIN", out)
        self.assertIn("bytes_per_row", out)

    def test_an_empty_pool_names_the_row_count_it_read(self):
        pool = _FakePool(rows=0)
        r = _relief(pool, _FakeAllocator(1), live=(), card=_Card(2000))
        proposal, out = self._capture(r)
        self.assertEqual(proposal, kbr.ABSTAIN)
        self.assertIn("ABSTAIN", out)
        self.assertIn("backed rows", out)

    def test_an_unreadable_live_set_is_reported_as_the_abstain_cause(self):
        """The probe already warns; the ABSTAIN it CAUSES must be its own line.

        The probe's warning says "live-set probe failed". It does not say that
        the group's decision was cancelled as a result, and those are different
        facts for the reader.
        """

        def _boom():
            raise RuntimeError("no live set here")

        pool = _FakePool(rows=1000)
        r = kbr.KvBackingRelief(
            pool,
            _FakeAllocator(1000),
            live_slots_fn=_boom,
            bytes_per_row=pool._bytes_per_row,
            probe=_Card(2000).probe,
        )
        proposal, out = self._capture(r)
        self.assertEqual(proposal, kbr.ABSTAIN)
        self.assertIn("ABSTAIN", out)
        self.assertIn("live set", out)

    def test_the_abstain_line_says_the_group_decision_is_cancelled(self):
        """A reader must not have to know the reduction to read the damage."""
        pool = _FakePool(rows=0)
        r = _relief(pool, _FakeAllocator(1), live=(), card=_Card(2000))
        _, out = self._capture(r)
        self.assertIn("whole group", out.lower())

    def test_a_persistent_abstain_does_not_flood_but_keeps_a_count(self):
        """Edge-triggered like the deficit trace, and it says how many.

        ``propose`` runs a few times a minute, so the cost of repeating is low
        -- but a per-call line would bury the FIRST one, which is the one that
        dates the failure. The count is what keeps "still abstaining" legible.
        """
        pool = _FakePool(rows=0)
        r = _relief(pool, _FakeAllocator(1), live=(), card=_Card(2000))
        with self.assertLogs(kbr.__name__, level="WARNING") as cm:
            for _ in range(5):
                r.propose(**self.ASK)
        self.assertEqual(len(cm.output), 1, cm.output)
        # ... and a changed cause re-arms the edge, because a DIFFERENT
        # precondition failing is new information.
        pool.size = 1000
        pool._bytes_per_row = 0
        r._bytes_per_row = 0
        with self.assertLogs(kbr.__name__, level="WARNING") as cm2:
            r.propose(**self.ASK)
        self.assertIn("bytes_per_row", "\n".join(cm2.output))

    def test_recovering_from_an_abstain_is_announced_too(self):
        """The rank came back. A reader watching for the rung needs that fact.

        Without it, an abstain that healed leaves a WARNING as the last word on
        this rung for the rest of the run.
        """
        pool = _FakePool(rows=0)
        r = _relief(pool, _FakeAllocator(1000), live=(5,), card=_Card(64000))
        with self.assertLogs(kbr.__name__, level="WARNING"):
            r.propose(**self.ASK)
        pool.size = 100000
        with self.assertLogs(kbr.__name__, level="INFO") as cm:
            r.propose(**self.ASK)
        self.assertIn("no longer abstaining", "\n".join(cm.output).lower())
