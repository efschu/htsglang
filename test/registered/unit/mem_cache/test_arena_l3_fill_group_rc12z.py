# SPDX-License-Identifier: Apache-2.0
"""rc12z (786d2f615d, 28.09. 01:33Z): the L3 -> L2 return path (#1433) on a
HostPoolGroup.

D TP0 (Form-A host) and every P stage hold their arena host pool inside a
``HostPoolGroup``. ``HiCacheController._arena_page_get`` asks the L3 for a page
that is not in the arena with ``pool._page_bytes`` -- on the group that fell to
``__getattr__`` and raised ``AttributeError: _page_bytes``; #1033d refused the
whole read (``PREFETCH IO REFUSED``), TP0 delivered 0 of 69888 tokens while the
workers (bare arena pools) delivered 69568, and pdflip-24-93 was re-prefilled in
full on P, where PP0 failed the same way.

Real objects: the C arena on a temp file, a bound ArenaMHAHostPool inside a
HostPoolGroup, the controller's ``_arena_page_get``. The pages are published,
then evicted from the arena (freed), and the backend's ``arena_fill_from_disk``
reads them back into fresh slots.
"""

import os
import tempfile
import types
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from flliper.srt.managers.cache_controller import HiCacheController
from flliper.srt.mem_cache.hicache_storage import PoolName
from flliper.srt.mem_cache.memory_pool_host import HostPoolGroup, PoolEntry
from flliper.srt.mem_cache.pool_host.arena_pool import ArenaMHAHostPool
from flliper.srt.mem_cache.storage.file import hicache_arena as ha
from flliper.test.test_utils import CustomTestCase

PAGE = 64  # 1 layer x 1 head x 16 dims x bf16, K and V
S = 16     # staging rows
SLOTS = 64


class _Backend:
    """The L3: every stem it was asked about is 'on disk'; a fill claims a
    fresh slot with the requested page size and completes it."""

    def __init__(self):
        self.fill_calls = []

    def _get_suffixed_key(self, h):
        return h

    def _suffix_for_key(self, k):
        return ("", None)

    def _arena_evict_to_disk(self, arena, want):
        cands = [c[0] for c in arena.evict_candidates(int(want))]
        if cands:
            arena.free_slots(cands)
        return len(cands)

    def arena_fill_from_disk(self, arena, stems, total_bytes):
        self.fill_calls.append(int(total_bytes))
        out = []
        for st in stems:
            (slot, status, gen), = arena.claim_slots([st], [int(total_bytes)])
            if status == 2:
                out.append(slot)
                continue
            if status != 0:
                out.append(None)
                continue
            cs = arena.complete_slots([slot], [gen], [(0, int(total_bytes))])
            out.append(slot if cs and cs[0] in (1, 2) else None)
        return out


class _Op:
    def __init__(self):
        self.completed = 0

    def increment(self, n):
        self.completed += n
        return True


def _rank(path, backend):
    arena = ha.ShmArena(path, PAGE, SLOTS)
    pool = ArenaMHAHostPool.__new__(ArenaMHAHostPool)
    pool.size, pool.page_size, pool.layout, pool.device = S, 1, "layer_first", "cpu"
    pool.layer_num, pool.dtype, pool.head_num, pool.head_dim = 1, torch.bfloat16, 1, 16
    pool.pin_memory = False
    pool._arena_init_fields()
    pool.bind(arena, types.SimpleNamespace(extents=[(0, 32), (32, 32)], total_bytes=PAGE),
              role="kv", pin=False)
    pool._backend = backend
    group = HostPoolGroup([PoolEntry(name=PoolName.KV, host_pool=pool, device_pool=None,
                                     layer_mapping={}, is_primary_index_anchor=True)])
    cc = types.SimpleNamespace(mem_pool_host=group, storage_backend=backend, page_size=1)
    return arena, pool, group, cc


class L3FillThroughTheGroup(CustomTestCase):
    def setUp(self):
        super().setUp()
        fd, self.path = tempfile.mkstemp(prefix="arena-l3-fill-group-", suffix=".bin")
        os.close(fd)
        os.unlink(self.path)

    def tearDown(self):
        try:
            os.unlink(self.path)
        except OSError:
            pass
        super().tearDown()

    def _evicted_pages(self, n=8):
        backend = _Backend()
        arena, pool, group, cc = _rank(self.path, backend)
        hashes = [f"p{j}" for j in range(n)]
        rows = pool.alloc_write(hashes)
        self.assertIsNotNone(rows)
        pool.complete_write(rows)
        pool.free(rows)
        slots = [s for s, _st in arena.find_slots(hashes)]
        self.assertTrue(all(s >= 0 for s in slots))
        arena.free_slots(slots)  # ARENA-EVICT: the pages now live only in the L3
        self.assertTrue(all(s < 0 or st != 2 for s, st in arena.find_slots(hashes)))
        return backend, arena, pool, group, cc, hashes

    def test_the_group_names_the_arena_page_size(self):
        _b, _a, pool, group, _cc, _h = self._evicted_pages()
        self.assertEqual(group._page_bytes, PAGE)
        self.assertEqual(group._page_bytes, pool._page_bytes)
        self.assertIs(group.arena, pool.arena)

    def test_a_page_evicted_to_disk_comes_back_through_the_group(self):
        """rc12z pdflip-24-93: before the fix AttributeError: _page_bytes (the
        caller's #1033d refused the read, 0 delivered)."""
        backend, arena, pool, group, cc, hashes = self._evicted_pages()
        hi = pool.alloc_read(len(hashes))
        op = _Op()
        got = HiCacheController._arena_page_get(cc, op, hashes, hi)
        self.assertEqual(got, len(hashes))
        self.assertEqual(op.completed, len(hashes))
        self.assertEqual(backend.fill_calls, [PAGE] * len(hashes))
        self.assertTrue(all(st == 2 for _s, st in arena.find_slots(hashes)))


if __name__ == "__main__":
    unittest.main()
