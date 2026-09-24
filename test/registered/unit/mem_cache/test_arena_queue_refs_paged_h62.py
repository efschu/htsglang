# SPDX-License-Identifier: Apache-2.0
"""H62 (NF line): the 27B arena-queue-refs fix (7184ec6f71) on the PAGED arena.

Next Flash binds its KV arena PAGED (x59): one arena slot per page of P tokens,
host ids ``S + slot * P + t``, placeholders from ``S + A * P``. The 27B commit
bounds arena ids by ``S + A`` (its page is one token) and hands them to
``release_tree_rows`` (27B 479f6eccb0, not on this line, unpaged). On NF both
would be wrong: every token id of slots ``>= A / P`` would read as a
placeholder and be dropped WITHOUT returning its reference -- the leak, with a
line claiming it was a placeholder. The NF form classifies by the paged range
(``_arena_mask`` / ``_atok``) and releases ONE reference per page.

MEASURED on NF (fnFL2x165): the #718 drop is live on both groups -- D TP0
19:37:08 "free ... was handed 640 index(es) outside [0, 4096) -- highest
354239" (10 pages), P PP0 19:37:29 "97792 index(es) outside [0, 6976) --
highest 454271" (1528 pages); the line is printed once per pool, the rest is
counted silently. `_arena_page_get` takes one reader reference per page there.

Real objects: the C arena on a temp file, a bound paged ArenaMHAHostPool
(x59 geometry, P = 4), HostPoolGroup, HybridCacheController._arena_page_get and
the scheduler drain (check_hicache_events). Needs gcc (the arena library).
"""

import os
import shutil
import tempfile
import threading
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
from sglang.srt.mem_cache.pool_host.arena_pool import PLACEHOLDERS, ArenaMHAHostPool
from sglang.srt.mem_cache.storage.file import hicache_arena as ha
from sglang.test.test_utils import CustomTestCase

ENV = "SGLANG_HICACHE_ARENA_QUEUE_REFS"
P, L, LT, H, D = 4, 3, 3, 2, 4   # x59 geometry (page tokens, layers, model layers, heads, head_dim)
CELL = H * D
BLOCK = P * CELL
PAGE = 2 * LT * BLOCK
S = 8                            # staging rows (two pages)
A = 6                            # arena slots -> arena ids [S, S + A * P) = [8, 32)


class _Win:
    """This rank holds every layer of the page, so its ack completes the page
    (a partial window would leave it for the other ranks' extents)."""
    total_bytes = PAGE
    extents = ((0, L * BLOCK), (PAGE // 2, L * BLOCK))


class _Backend:
    def _get_suffixed_key(self, h):
        return h

    def _suffix_for_key(self, k):
        return ("", None)

    def _arena_evict_to_disk(self, arena, want):
        """The claim's eviction round without disk I/O: only UNREFERENCED
        complete slots are candidates -- a leaked reference pins its slot."""
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
    out = (np.ctypeslib.ctypes.c_int64 * 6)()
    arena._lib.arena_layout(arena.slots, arena.slot_bytes, out)
    hb, hoff = int(out[0]), int(out[3])
    u32 = np.frombuffer(arena._mm, dtype=np.uint32, count=arena.slots * hb // 4, offset=hoff)
    h = u32.reshape(arena.slots, hb // 4)
    return h[:, 0].copy(), h[:, 1].copy()


class _Rank:
    """One NF rank: a PAGED arena pool (P = 4), its group, controller, cache."""

    def __init__(self, path, own_ledger=False):
        self.arena = ha.ShmArena(path, PAGE, A)
        if own_ledger and getattr(self.arena, "_ledger", None) is not None:
            self.arena._ledger = ha.RefLedger(A)  # another PROCESS: its own ledger
        p = object.__new__(ArenaMHAHostPool)
        p.layout = "layer_first"; p.page_size = P; p.layer_num = L; p.head_num = H; p.head_dim = D
        p.dtype = torch.uint8; p.device = "cpu"; p.pin_memory = False; p.size = S
        p.element_dim = H * D; p.can_use_jit = False; p.token_stride_size = CELL
        p.lock = threading.Lock()
        p.free_slots = torch.arange(S, dtype=torch.int64); p.slot_used = torch.zeros(S, dtype=torch.bool)
        p.kv_buffer = torch.zeros(2, L, S, H, D, dtype=torch.uint8)
        p._arena_init_fields()
        p.bind(self.arena, _Win(), role="kv", pin=False)
        p._pinned[:] = True
        p._backend = _Backend()
        self.pool = p
        self.group = HostPoolGroup([PoolEntry(name=PoolName.KV, host_pool=p, device_pool=None,
                                              layer_mapping={}, is_primary_index_anchor=True)])
        cc = HybridCacheController.__new__(HybridCacheController)
        cc.mem_pool_host, cc.storage_backend, cc.page_size = self.group, _Backend(), P
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
        """P: claim (P ids per page), the copy lands (COMPLETE, node ref), the
        node is evicted. Returns the slots."""
        rows = self.pool.alloc_write(hashes)
        assert rows is not None, "claim refused"
        self.pool.complete_write(rows)
        self.pool.free(rows)
        return torch.unique_consecutive((rows - S) // P)

    def prefetch(self, hashes):
        """D: len(hashes) * P placeholders, resolved in the arena (+1 per page)."""
        hi = self.pool.alloc_read(len(hashes) * P)
        got = HiCacheController._arena_page_get(self.cc, _Op(), hashes, hi)
        return hi, got

    def release_via_queue(self, rows):
        self.cc.append_host_mem_release(host_indices=rows)
        self.cache.check_hicache_events()


@unittest.skipIf(shutil.which("gcc") is None, "needs gcc (the arena library)")
class _Case(CustomTestCase):
    def setUp(self):
        super().setUp()
        self._saved = os.environ.pop(ENV, None)
        fd, self.path = tempfile.mkstemp(prefix="arena-queue-refs-paged-", suffix=".bin")
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


class TheGeometryIsPaged(_Case):
    def test_arena_ids_count_tokens(self):
        a = _Rank(self.path)
        self.assertEqual(a.pool.arena_tokens, A * P)
        self.assertEqual(a.pool.id_space, S + A * P + PLACEHOLDERS)


class TheDefaultStillLeaks(_Case):
    def test_default_drops_the_queued_pages_and_keeps_their_references(self):
        """The defect as it stands on NF (the x165 #718 line)."""
        a = _Rank(self.path)
        slots = a.publish([f"p{j}" for j in range(4)])
        hi, got = a.prefetch([f"p{j}" for j in range(4)])
        self.assertEqual(got, 4)
        a.release_via_queue(hi[: 2 * P])  # the unclaimed head: two pages
        _st, ref = _hdr(a.arena)
        self.assertEqual(ref[slots.numpy()].tolist(), [1, 1, 1, 1])  # never returned

    def test_default_pins_the_arena_until_a_claim_is_refused(self):
        """One unclaimed page per cycle through the queue: its slot stays
        referenced, eviction cannot take it, and the writer's claim is refused
        within a few cycles of a 6-slot arena."""
        a = _Rank(self.path)
        refused_at = None
        for i in range(12):
            hashes = [f"c{i}-p{j}" for j in range(4)]
            rows = a.pool.alloc_write(hashes)
            if rows is None:
                refused_at = i
                break
            a.pool.complete_write(rows)
            a.pool.free(rows)
            hi, got = a.prefetch(hashes)
            a.release_via_queue(hi[:P])
            a.pool.free(hi[P: got * P])
        self.assertIsNotNone(refused_at, "the unchanged path did not pin the arena")


class ThePagedQueueReturnsOneReferencePerPage(_Case):
    def test_the_unclaimed_head_returns_one_reference_per_page(self):
        self.on()
        a = _Rank(self.path)
        slots = a.publish([f"p{j}" for j in range(4)])
        hi, _ = a.prefetch([f"p{j}" for j in range(4)])
        a.release_via_queue(hi[: 2 * P])
        _st, ref = _hdr(a.arena)
        self.assertEqual(ref[slots[:2].numpy()].tolist(), [0, 0])
        self.assertEqual(ref[slots[2:].numpy()].tolist(), [1, 1])  # the adopted pages
        self.assertEqual(int(a.arena._ledger.held.sum()), 2)
        self.assertEqual(a.group._queue_refs["returned"], 2)
        self.assertEqual(a.group._queue_refs["rows"], 2 * P)

    def test_high_slots_are_arena_rows_not_placeholders(self):
        """The 27B bound (S + A) on NF: slot 5's ids are S + 20 .. S + 23, far
        above S + A = 14 -- they must return their reference, not be counted
        as placeholders."""
        self.on()
        a = _Rank(self.path)
        slots = a.publish([f"h{j}" for j in range(A)])  # every slot, 5 included
        self.assertEqual(sorted(slots.tolist()), list(range(A)))
        hi, got = a.prefetch([f"h{j}" for j in range(A)])
        self.assertEqual(got, A)
        a.release_via_queue(hi)
        _st, ref = _hdr(a.arena)
        self.assertEqual(ref.tolist(), [0] * A)
        self.assertEqual(a.group._queue_refs["placeholders"], 0)

    def test_many_cycles_never_pin_the_arena(self):
        self.on()
        a = _Rank(self.path)
        for i in range(30):
            hashes = [f"c{i}-p{j}" for j in range(4)]
            a.publish(hashes)
            hi, got = a.prefetch(hashes)
            self.assertEqual(got, 4, f"cycle {i}: the arena is pinned")
            a.release_via_queue(hi[:P])       # one unclaimed page per cycle
            a.pool.free(hi[P: got * P])       # the adopted rest, evicted by the tree
        pinned, refs, _complete = a.arena.ref_census()
        self.assertEqual((pinned, refs), (0, 0))

    def test_placeholders_are_dropped_without_a_reference_or_a_stray_line(self):
        self.on()
        a = _Rank(self.path)
        hi = a.pool.alloc_read(2 * P)  # never resolved (a revoke's span)
        with mock.patch("sglang.srt.mem_cache.memory_pool_host.logger") as lg:
            a.group.free(hi)
        self.assertFalse(any("HICACHE-INDEX REFUSED" in str(c) for c in lg.error.call_args_list))
        self.assertEqual(a.group._queue_refs["placeholders"], 2 * P)

    def test_a_page_named_twice_in_one_call_releases_once(self):
        """Two resolutions of one page (two references), the rows named twice
        in one free: one reference dropped."""
        self.on()
        a = _Rank(self.path)
        s = int(a.publish(["e0"])[0])
        a.prefetch(["e0"])
        hi, _ = a.prefetch(["e0"])
        a.group.free(torch.cat([hi, hi]))
        _st, ref = _hdr(a.arena)
        self.assertEqual(int(ref[s]), 1)

    def test_a_partial_page_releases_its_page_once(self):
        """Rows of one page split across the head/tail boundary still name ONE
        page: two tokens of it -> one reference."""
        self.on()
        a = _Rank(self.path)
        s = int(a.publish(["q0"])[0])
        hi, _ = a.prefetch(["q0"])
        a.group.free(hi[:2])
        _st, ref = _hdr(a.arena)
        self.assertEqual(int(ref[s]), 0)

    def test_staging_rows_still_take_the_unchanged_path(self):
        self.on()
        a = _Rank(self.path)
        seen = []
        a.pool.free = lambda idx: seen.append(idx.clone()) or int(idx.numel())
        a.group.free(torch.tensor([3, 5, S + 1, S + A * P + 2], dtype=torch.int64))
        self.assertEqual(torch.cat(seen).tolist(), [3, 5])

    def test_a_pending_write_is_not_released(self):
        self.on()
        a = _Rank(self.path)
        rows = a.pool.alloc_write(["w0"])  # claimed, copy not acked
        a.release_via_queue(rows)
        st, _ref = _hdr(a.arena)
        self.assertEqual(int(st[int(rows[0] - S) // P]), 1)  # still CLAIMED


class TheLedgerOnlyReleasesOwnReferences(_Case):
    def test_a_page_another_rank_reads_keeps_that_reference(self):
        self.on()
        a = _Rank(self.path)
        b = _Rank(self.path, own_ledger=True)
        slots = a.publish(["x0", "x1"])
        b.arena.ref_slots(slots.tolist(), +1)  # B reads
        a.release_via_queue(a.pool.arena_ids(slots))
        _st, ref = _hdr(a.arena)
        self.assertEqual(ref[slots.numpy()].tolist(), [1, 1])
        self.assertEqual(a.arena._ledger.refused, 2)

    def test_mutant_unpaged_bound_leaks_the_high_slots(self):
        """The 27B bound (S + A) put back: slot >= (A / P) pages read as
        placeholders and keep their reference -- the test above must catch it."""
        self.on()
        a = _Rank(self.path)
        with mock.patch("sglang.srt.mem_cache.memory_pool_host._arena_id_tokens",
                        lambda pool: int(pool.arena_slots)):
            slots = a.publish([f"m{j}" for j in range(A)])
            hi, _ = a.prefetch([f"m{j}" for j in range(A)])
            a.release_via_queue(hi)
        _st, ref = _hdr(a.arena)
        self.assertGreater(int(ref[slots.numpy()].sum()), 0, "mutant survived")


if __name__ == "__main__":
    unittest.main()
