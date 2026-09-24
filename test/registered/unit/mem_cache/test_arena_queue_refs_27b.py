# SPDX-License-Identifier: Apache-2.0
"""27B (24.09.): arena reader references out of the host release queue.

THE DEFECT. A store prefetch resolves its rows in the shared arena
(`HiCacheController._arena_page_get`): one reader reference (+1) per page, the
row id = staging_rows + slot. Rows the tree does not adopt -- the unclaimed head
(already on device), an unsynced tail, an aborted or retired span -- go back
through `append_host_mem_release` -> the scheduler drain -> `HostPoolGroup.free`.
That free bounded ids by `pool.size`, the STAGING range of an arena host pool, so
every arena id was dropped as a #718 stray and its reference was never returned
(xsn429 P: "4314 index(es) outside [0, 2442)"; D: 51942 such rows in 12 min). A
referenced slot is never evicted: a long boot pins the arena until claims are
refused.

SGLANG_HICACHE_ARENA_QUEUE_REFS=1 (default off): the group hands arena ids back
(one reference per row, duplicates collapse, pending writes skipped), through a
per-process RefLedger that refuses any release beyond this process's own
references. Real objects throughout: the C arena on a temp file, a bound
ArenaMHAHostPool, HostPoolGroup, HybridCacheController and the scheduler drain.
A second ShmArena with its own ledger on the same file plays another rank.

The MUTANT classes run the danger scenarios against deliberately broken
variants of the fix (double release, release of a slot still read / still being
written) and assert that the scenario catches each one.
"""

import os
import tempfile
import types
import unittest
from queue import Queue
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache import unified_radix_cache as u
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import HybridCacheController
from sglang.srt.mem_cache.memory_pool_host import HostPoolGroup, PoolEntry
from sglang.srt.mem_cache.pool_host.arena_pool import ArenaMHAHostPool
from sglang.srt.mem_cache.storage.file import hicache_arena as ha
from sglang.test.test_utils import CustomTestCase

ENV = "SGLANG_HICACHE_ARENA_QUEUE_REFS"  # literal: the parent tree has no constant
PAGE = 64  # 1 layer x 1 head x 16 dims x bf16, K and V
S = 16     # staging rows


class _Backend:
    def _get_suffixed_key(self, h):
        return h

    def _suffix_for_key(self, k):
        return ("", None)

    def _arena_evict_to_disk(self, arena, want):
        # NF line (H62): a claim that finds no free slot evicts through the
        # backend (the 27B frees in-pool via _evict_for_claim, 479f6eccb0, not
        # on this line). Same candidates -- unreferenced COMPLETE slots only --
        # freed without disk I/O, as the 27B claim does.
        cands = [c[0] for c in arena.evict_candidates(int(want))]
        if cands:
            arena.free_slots(cands)
        return len(cands)


class _Op:
    def __init__(self):
        self.completed = 0

    def increment(self, n):
        self.completed += n
        return True


def _hdr(arena):
    """(state, refcount) arrays read from the slot headers."""
    out = (np.ctypeslib.ctypes.c_int64 * 6)()
    arena._lib.arena_layout(arena.slots, arena.slot_bytes, out)
    hb, hoff = int(out[0]), int(out[3])
    u32 = np.frombuffer(arena._mm, dtype=np.uint32, count=arena.slots * hb // 4, offset=hoff)
    h = u32.reshape(arena.slots, hb // 4)
    return h[:, 0].copy(), h[:, 1].copy()


class _Rank:
    """One rank process: its arena view, host pool, group, controller, cache."""

    def __init__(self, path, slots, own_ledger=False):
        self.arena = ha.ShmArena(path, PAGE, slots)
        if own_ledger and getattr(self.arena, "_ledger", None) is not None:
            self.arena._ledger = ha.RefLedger(slots)  # another PROCESS: its own ledger
        pool = ArenaMHAHostPool.__new__(ArenaMHAHostPool)
        pool.size, pool.page_size, pool.layout, pool.device = S, 1, "layer_first", "cpu"
        pool.layer_num, pool.dtype, pool.head_num, pool.head_dim = 1, torch.bfloat16, 1, 16
        pool.pin_memory = False
        pool._arena_init_fields()
        pool.bind(self.arena, types.SimpleNamespace(extents=[(0, 32), (32, 32)], total_bytes=PAGE),
                  role="kv", pin=False)
        pool._backend = _Backend()
        self.pool = pool
        self.group = HostPoolGroup([PoolEntry(name=PoolName.KV, host_pool=pool, device_pool=None,
                                              layer_mapping={}, is_primary_index_anchor=True)])
        cc = HybridCacheController.__new__(HybridCacheController)
        cc.mem_pool_host, cc.storage_backend, cc.page_size = self.group, _Backend(), 1
        cc.host_mem_release_queue, cc.extra_host_mem_release_queues = Queue(), {}
        cc.prefetch_revoke_queue, cc.ack_backup_queue = Queue(), Queue()
        self.cc = cc
        c = u.UnifiedRadixCache.__new__(u.UnifiedRadixCache)
        c.cache_controller = cc
        c.attn_cp_group = c.attn_tp_group = None
        c.tp_world_size = 1
        c.enable_storage, c.enable_storage_metrics, c.storage_metrics_collector = True, False, None
        c.ongoing_prefetch, c.ongoing_backup, c.staging_write_ring = {}, {}, None
        c._pin_trace_every = 0
        c._drain_async_work = lambda: None
        c.writing_check = lambda *a, **k: None
        c.loading_check = lambda: None
        self.cache = c

    def publish(self, hashes):
        """P: claim, the copy lands (COMPLETE, node ref), the node is evicted."""
        rows = self.pool.alloc_write(hashes)
        if rows is None:
            return None
        self.pool.complete_write(rows)
        self.pool.free(rows)
        return rows - S

    def prefetch(self, hashes):
        """D: registration placeholders, resolved in the arena (+1 per page)."""
        hi = self.pool.alloc_read(len(hashes))
        got = HiCacheController._arena_page_get(self.cc, _Op(), hashes, hi)
        return hi, got

    def release_via_queue(self, rows):
        self.cc.append_host_mem_release(host_indices=rows)
        self.cache.check_hicache_events()


class _ArenaCase(CustomTestCase):
    SLOTS = 64

    def setUp(self):
        super().setUp()
        self._saved = os.environ.pop(ENV, None)
        fd, self.path = tempfile.mkstemp(prefix="arena-queue-refs-", suffix=".bin")
        os.close(fd)
        os.unlink(self.path)

    def tearDown(self):
        if self._saved is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = self._saved
        try:
            os.unlink(self.path)
        except OSError:
            pass
        super().tearDown()

    def on(self):
        os.environ[ENV] = "1"


class TheDefaultIsUnchanged(_ArenaCase):
    def test_default_drops_the_queued_arena_rows_and_keeps_their_references(self):
        """The defect, as it stands on the unchanged path."""
        a = _Rank(self.path, self.SLOTS)
        self.assertIsNone(getattr(a.arena, "_ledger", None))
        slots = a.publish([f"p{j}" for j in range(8)])
        hi, got = a.prefetch([f"p{j}" for j in range(8)])
        self.assertEqual(got, 8)
        a.release_via_queue(hi[:5])  # the unclaimed head
        _st, ref = _hdr(a.arena)
        self.assertEqual(ref[slots[:5].numpy()].tolist(), [1] * 5)  # never returned


class TheQueueReturnsItsReferences(_ArenaCase):
    def test_the_unclaimed_head_returns_its_reader_references(self):
        self.on()
        a = _Rank(self.path, self.SLOTS)
        slots = a.publish([f"p{j}" for j in range(8)])
        hi, got = a.prefetch([f"p{j}" for j in range(8)])
        a.release_via_queue(hi[:5])
        _st, ref = _hdr(a.arena)
        self.assertEqual(ref[slots[:5].numpy()].tolist(), [0] * 5)
        self.assertEqual(ref[slots[5:].numpy()].tolist(), [1] * 3)  # still the reader's (adopted)
        self.assertEqual(int(a.arena._ledger.held.sum()), 3)

    def test_many_prefetch_cycles_never_pin_the_arena(self):
        """The desk run: every cycle's unclaimed head through the queue. The
        unchanged path pins 5 slots per cycle and refuses a claim once the
        arena is pinned full; the fix keeps the arena evictable."""
        self.on()
        a = _Rank(self.path, 200)
        for i in range(60):
            hashes = [f"c{i}-p{j}" for j in range(20)]
            self.assertIsNotNone(a.publish(hashes), f"claim refused at cycle {i}")
            hi, got = a.prefetch(hashes)
            self.assertEqual(got, 20)
            a.release_via_queue(hi[:5])
            a.pool.free(hi[5:got])  # the adopted rest, evicted by the tree later
        pinned, refs, _complete = a.arena.ref_census()
        self.assertEqual((pinned, refs), (0, 0))

    def test_placeholders_are_dropped_without_a_reference_or_a_stray_line(self):
        self.on()
        a = _Rank(self.path, self.SLOTS)
        hi = a.pool.alloc_read(6)  # never resolved (a revoke's span)
        with mock.patch("sglang.srt.mem_cache.memory_pool_host.logger") as lg:
            a.group.free(hi)
        self.assertFalse(any("HICACHE-INDEX REFUSED" in str(c) for c in lg.error.call_args_list))
        self.assertEqual(a.group._queue_refs["placeholders"], 6)

    def test_staging_rows_still_take_the_unchanged_path(self):
        self.on()
        a = _Rank(self.path, self.SLOTS)
        seen = []
        a.pool.free = lambda idx: seen.append(idx.clone()) or int(idx.numel())
        a.group.free(torch.tensor([3, 5, S + 1, S + self.SLOTS + 2], dtype=torch.int64))
        self.assertEqual(torch.cat(seen).tolist(), [3, 5])


class TheLedgerOnlyReleasesOwnReferences(_ArenaCase):
    def test_a_release_this_process_never_took_is_refused(self):
        """Another rank reads the slot; a stray arena id for it reaches this
        rank's queue. This rank holds nothing there: refused, counted."""
        self.on()
        a = _Rank(self.path, self.SLOTS)
        b = _Rank(self.path, self.SLOTS, own_ledger=True)
        slots = a.publish(["x0", "x1"])
        b.arena.ref_slots(slots.tolist(), +1)  # B reads
        a.release_via_queue(slots + S)
        _st, ref = _hdr(a.arena)
        self.assertEqual(ref[slots.numpy()].tolist(), [1, 1])
        self.assertEqual(a.arena._ledger.refused, 2)
        self.assertEqual(a.group._queue_refs["refused"], 2)

    def test_a_partial_reference_batch_is_not_recorded(self):
        """+1 on [COMPLETE, FREE]: C takes one, the ledger records none -- the
        reference it cannot name is leaked, never released twice."""
        self.on()
        a = _Rank(self.path, self.SLOTS)
        s = int(a.publish(["y0"])[0])
        free_slot = next(i for i in range(self.SLOTS) if i != s)
        self.assertEqual(a.arena.ref_slots([s, free_slot], +1), 1)
        self.assertEqual(int(a.arena._ledger.held[s]), 0)
        self.assertEqual(a.arena.ref_slots([s], -1), 0)  # refused
        _st, ref = _hdr(a.arena)
        self.assertEqual(int(ref[s]), 1)


class TheCensusCountsWhatIsPinned(_ArenaCase):
    def test_census_names_pinned_slots_and_references(self):
        a = _Rank(self.path, self.SLOTS)
        slots = a.publish(["k0", "k1", "k2"])
        a.arena.ref_slots(slots[:2].tolist(), +1)
        a.arena.ref_slots(slots[:1].tolist(), +1)
        self.assertEqual(a.arena.ref_census(), (2, 3, 3))

    def test_the_census_thread_logs_and_stops(self):
        import time as _t

        os.environ[ha.ENV_REF_CENSUS_S] = "0.02"
        try:
            a = _Rank(self.path, self.SLOTS)
            key = os.path.realpath(a.arena.path)
            with self.assertLogs(ha.logger, level="INFO") as cm:
                for _ in range(200):
                    if any("ARENA-REF-CENSUS" in m for m in cm.output):
                        break
                    _t.sleep(0.01)
                    ha.logger.info("tick")
            self.assertTrue(any("ARENA-REF-CENSUS" in m and "pinned=0" in m for m in cm.output))
        finally:
            os.environ.pop(ha.ENV_REF_CENSUS_S, None)
            stop = ha._CENSUS_THREADS.pop(key, None)
            if stop is not None:
                stop.set()

    def test_two_views_of_one_file_in_one_process_share_the_ledger(self):
        self.on()
        a = _Rank(self.path, self.SLOTS)
        again = ha.ShmArena(self.path, PAGE, self.SLOTS)
        self.assertIs(again._ledger, a.arena._ledger)


# -- the danger scenarios, each returning what it observed ------------------------


def _scenario_double_queued_span(case):
    """A span queued TWICE (#989 shape), drained in two rounds. B also reads
    the pages. Returns B's reference count left on the slots."""
    case.on()
    a = _Rank(case.path, case.SLOTS)
    b = _Rank(case.path, case.SLOTS, own_ledger=True)
    hashes = [f"d{j}" for j in range(4)]
    slots = a.publish(hashes)
    b.arena.ref_slots(slots.tolist(), +1)  # B reads
    hi, _ = a.prefetch(hashes)             # A resolves: +1 each
    a.release_via_queue(hi)
    a.release_via_queue(hi.clone())        # the duplicate, next round
    _st, ref = _hdr(a.arena)
    return ref[slots.numpy()].tolist()


def _scenario_row_twice_in_one_call(case):
    """The rebind settle shape (no drain dedup): one free call names the same
    row twice. A holds two references there (two resolutions). Returns the
    slot's reference count (2 -> 1 expected)."""
    case.on()
    a = _Rank(case.path, case.SLOTS)
    s = a.publish(["e0"])
    a.prefetch(["e0"])
    hi, _ = a.prefetch(["e0"])
    a.group.free(torch.cat([hi, hi]))
    _st, ref = _hdr(a.arena)
    return int(ref[int(s[0])])


def _scenario_other_rank_reading(case):
    """B reads the slot while A returns its unclaimed row. Returns the slot's
    (state, refcount) and whether the evictor would take it."""
    case.on()
    a = _Rank(case.path, case.SLOTS)
    b = _Rank(case.path, case.SLOTS, own_ledger=True)
    s = int(a.publish(["r0"])[0])
    b.arena.ref_slots([s], +1)
    hi, _ = a.prefetch(["r0"])
    a.release_via_queue(hi)
    st, ref = _hdr(a.arena)
    cands = [c[0] for c in a.arena.evict_candidates(case.SLOTS)]
    return int(st[s]), int(ref[s]), s in cands


def _scenario_pending_write(case):
    """A's writer has claimed the page (pending, DMA in flight) and the same
    row reaches the queue. Returns the slot state (1 = CLAIMED expected)."""
    case.on()
    a = _Rank(case.path, case.SLOTS)
    rows = a.pool.alloc_write(["w0"])      # claimed, not completed
    a.release_via_queue(rows)
    st, _ref = _hdr(a.arena)
    return int(st[int(rows[0]) - S])


class TheDangerScenariosHold(_ArenaCase):
    def test_a_double_queued_span_takes_no_other_reference(self):
        self.assertEqual(_scenario_double_queued_span(self), [1, 1, 1, 1])

    def test_a_row_named_twice_in_one_call_drops_one_reference(self):
        self.assertEqual(_scenario_row_twice_in_one_call(self), 1)

    def test_a_slot_another_rank_reads_stays_complete_and_unevictable(self):
        self.assertEqual(_scenario_other_rank_reading(self), (2, 1, False))

    def test_a_pending_write_is_not_released(self):
        self.assertEqual(_scenario_pending_write(self), 1)


class TheMutantsAreCaught(_ArenaCase):
    """Each mutant breaks the fix in a danger direction; its scenario must show it."""

    def test_mutant_no_ledger_gate_double_release(self):
        with mock.patch.object(ha.RefLedger, "allow_release",
                               lambda self, slots: torch.as_tensor(slots, dtype=torch.int64).reshape(-1)):
            got = _scenario_double_queued_span(self)
        self.assertNotEqual(got, [1, 1, 1, 1], "mutant survived: B's reference was not taken")
        self.assertEqual(got, [0, 0, 0, 0])

    def test_mutant_no_dedup_double_release_in_one_call(self):
        # NF line (H62): no release_tree_rows here (27B 479f6eccb0) -- the
        # mutant is the same "one reference per row, no dedup" in place.
        def no_dedup(self, rows):
            slots = rows.reshape(-1).cpu().to(torch.int64) - self.staging_rows
            return int(self.arena.ref_slots_np(slots.numpy(), -1))

        with mock.patch.object(ArenaMHAHostPool, "release_queued_rows", no_dedup):
            self.assertEqual(_scenario_row_twice_in_one_call(self), 0)

    def test_mutant_frees_the_slot_instead_of_dropping_a_reference(self):
        def free_slots(self, rows):
            slots = (rows.reshape(-1).cpu().to(torch.int64) - self.staging_rows).tolist()
            self.arena.free_slots(slots)
            return len(slots)

        with mock.patch.object(ArenaMHAHostPool, "release_queued_rows", free_slots):
            st, _ref, _ev = _scenario_other_rank_reading(self)
        self.assertEqual(st, 0, "mutant survived: the slot another rank reads was not freed")

    def test_mutant_drops_every_reference_on_the_slot(self):
        def drop_all(self, rows):
            slots = (rows.reshape(-1).cpu().to(torch.int64) - self.staging_rows).tolist()
            _st, ref = _hdr(self.arena)
            n = 0
            for s in slots:
                for _ in range(int(ref[s])):
                    n += self.arena._lib.arena_ref_slots(self.arena._base, 1,
                                                         (np.ctypeslib.ctypes.c_int64 * 1)(s), -1)
            return n

        with mock.patch.object(ArenaMHAHostPool, "release_queued_rows", drop_all):
            st, ref, evictable = _scenario_other_rank_reading(self)
        self.assertEqual((ref, evictable), (0, True))

    def test_mutant_ignores_the_pending_write(self):
        with mock.patch.object(ArenaMHAHostPool, "release_queued_rows",
                               lambda self, rows: ArenaMHAHostPool.free(self, rows)):
            self.assertEqual(_scenario_pending_write(self), 0)  # the writer's slot freed


if __name__ == "__main__":
    unittest.main()
