"""#1424f (rc12p fced72cf5b, P-PP0 15:15:51 node 999 and 15:21:05 node 1109,
P-Log Z. 150760 / 164056): "WEG2 PUBLISH-SWEEP write_backup raised ...
RuntimeError: #1424 paged arena load: a page's token rows are not consecutive
from its first id" -- on the WRITE side, caught by the sweep.

The claim of a node's pages (``ArenaMHAHostPool._claim_np``) answers, per
page, fresh / join / already COMPLETE / no free slot. The COMPLETE pages took
their reader reference only at the very end -- AFTER the claim had made room
for the "no free slot" pages (``_evict_for_claim``: the oldest UNREFERENCED
complete slots). A page this very claim had just found COMPLETE was such a
slot: the room-making freed it and the redo handed the same slot to another
page of the node. One node, one slot twice. The write's host ids are sorted
by the controller (``move_indices``, layer_first/direct), the two copies of
the page interleave, and the page check refuses the write -- the lucky
outcome: unsorted, the second page would have overwritten the first, and the
first page's key no longer names that slot at all (its index cell was
unlinked by the eviction). Both P incidents sit next to claim-time room
making under a full arena (#1427 ARENA-DROP 15:14:42 n=256, 15:21:05 n=512
freed=0; ARENA-CLAIM REFUSED from 15:18:19).

Fix: the found-COMPLETE pages are referenced BEFORE any room is made (the
evictor skips referenced slots), released again when the claim is refused;
a slot named twice by one claim is refused by name (#1424f ARENA-CLAIM
DUPLICATE), never written.

Hermetic: the real C arena on a temp file, the real ``alloc_write`` /
``_claim_np`` / ``_evict_for_claim`` of the KV pool."""
from __future__ import annotations

import ctypes
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from sglang.srt.mem_cache.pool_host import arena_pool as ap  # noqa: E402
from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena  # noqa: E402

P = 4
SLOTS = 4
SLOT_BYTES = 256
S = 5


@pytest.fixture
def arena(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_HICACHE_ARENA_QUEUE_REFS", "1")
    a = ShmArena(str(tmp_path / "arena-kv-race.bin"), SLOT_BYTES, SLOTS)
    yield a
    a.close()


def _pool(arena):
    pool = object.__new__(ap.ArenaMHAHostPool)
    pool._arena_page_tokens = P
    pool.staging_rows = S
    pool.arena = arena
    pool.arena_tokens = SLOTS * P
    pool.arena_slots = SLOTS
    pool.row_slot = None
    pool._page_bytes = SLOT_BYTES
    pool._backend = types.SimpleNamespace(_get_suffixed_key=lambda k: k)
    pool._pending = {}
    pool._pending_mask = torch.zeros(SLOTS, dtype=torch.bool)
    pool._pending_gen = torch.zeros(SLOTS, dtype=torch.int64)
    pool._pending_fresh = torch.zeros(SLOTS, dtype=torch.bool)
    return pool


def _publish(arena, stems):
    out = {}
    for stem in stems:
        (s, st, g), = arena.claim_slots([stem], [SLOT_BYTES])
        assert st == 0
        assert arena.complete_slots([s], [g], [(0, SLOT_BYTES)]) == [1]
        out[stem] = s
    return out


def _foreign_ref(arena, slot):
    """Another process's reader reference (bypasses this process's ledger)."""
    c = (ctypes.c_int64 * 1)(slot)
    assert arena._lib.arena_ref_slots(arena._base, 1, c, 1) == 1


def _refs(arena):
    out = (ctypes.c_int64 * 6)()
    arena._lib.arena_layout(arena.slots, arena.slot_bytes, out)
    hb, hoff = int(out[0]), int(out[3])
    u32 = np.frombuffer(arena._mm, dtype=np.uint32, count=arena.slots * hb // 4, offset=hoff)
    return u32.reshape(arena.slots, hb // 4)[:, 1].astype(np.int64).tolist()


def _sorted_write_rows(ids):
    """What the controller hands the backup: host ids sorted (move_indices,
    layer_first/direct), then the pool's page check."""
    rows = torch.as_tensor(ids).sort().values - S
    return ap._page_slots(types.SimpleNamespace(_arena_page_tokens=P), rows)


def test_metal_shape_a_found_page_is_never_evicted_under_its_own_claim(arena):
    """RED on 11554824aa (+#1424d/e). The arena is full; the node's first page
    is COMPLETE (found, still unreferenced), its second page needs a slot, and
    the only unreferenced complete slot IS the first page's. The shipped claim
    freed it for the second page: the node named one slot twice and the
    sorted write refused ("not consecutive"). Now the found page is referenced
    before the room is made -- nothing evictable remains, the claim is
    refused by name, the first page's slot still holds its key, every
    reference taken is given back."""
    slot_of = _publish(arena, ["page-a", "x1", "x2", "x3"])
    for k in ("x1", "x2", "x3"):
        _foreign_ref(arena, slot_of[k])          # other holders keep these
    base = _refs(arena)
    pool = _pool(arena)
    ids = pool.alloc_write(["page-a", "page-b"])
    if ids is not None:
        slots = ((torch.as_tensor(ids) - S).view(-1, P)[:, 0] // P).tolist()
        assert len(set(slots)) == len(slots), f"one node named slot {slots} twice"
        _sorted_write_rows(ids)                  # the write the controller would issue
    assert arena.find_slots(["page-a"]) == [(slot_of["page-a"], 2)], \
        "the found page keeps its slot and its key"
    assert _refs(arena) == base, "a refused claim gives back what it took"


def test_room_from_another_slot_keeps_both_pages_distinct(arena):
    """With another unreferenced slot to give, the claim makes room from it:
    two distinct slots, the found page carries this node's reference, the new
    page is pending; the sorted write is whole pages."""
    slot_of = _publish(arena, ["page-a", "old", "x2", "x3"])
    for k in ("x2", "x3"):
        _foreign_ref(arena, slot_of[k])
    pool = _pool(arena)
    ids = pool.alloc_write(["page-a", "page-b"])
    assert ids is not None
    slots = ((torch.as_tensor(ids) - S).view(-1, P)[:, 0] // P).tolist()
    assert slots[0] == slot_of["page-a"] and slots[1] != slots[0]
    assert slots[1] == slot_of["old"], "room was made from the other unreferenced slot"
    assert _sorted_write_rows(ids).tolist() == sorted(slots)
    assert _refs(arena)[slot_of["page-a"]] == 1, "the node's reader reference on the found page"
    assert bool(pool._pending_mask[slots[1]]) and not bool(pool._pending_mask[slots[0]])


def test_duplicate_mask_names_the_repeat_only():
    """The belt (#1424f ARENA-CLAIM DUPLICATE): the second naming of a slot
    in one claim is marked, the first and the misses are not."""
    m = ap._claim_duplicates(np.array([3, 1, 3, -1, -1, 2]), np.array([2, 0, 0, 4, 4, 1]))
    assert m.tolist() == [False, False, True, False, False, False]
    assert not ap._claim_duplicates(np.array([0, 1, 2]), np.array([0, 0, 2])).any()


def test_premise_one_slot_twice_is_the_metal_refusal_after_the_sort():
    """The metal signature from the shape the shipped claim produced: a node
    naming slot 0 twice, its host ids sorted by the controller -> the two
    copies interleave -> "a page's token rows are not consecutive"."""
    pool = _pool(types.SimpleNamespace())
    ids = pool.arena_ids([0, 0])
    with pytest.raises(RuntimeError, match="not consecutive from its first id"):
        _sorted_write_rows(ids)


def test_a_write_that_raises_gives_its_claim_back(arena):
    """RED on 11554824aa (+#1424d/e): the write raised after the direct claim
    and the claimed pages stayed PENDING for good (no ack, no abort; the KV
    arena has no orphan reap). Now the claim goes back (fresh slots freed,
    found references released) and the error still propagates to the sweep."""
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    slot_of = _publish(arena, ["page-a"])
    base = _refs(arena)
    pool = _pool(arena)
    pre = pool.alloc_write(["page-a", "page-b"])
    assert pre is not None and int(pool._pending_mask.sum()) == 1

    def _raise(*a, **k):
        raise RuntimeError("#1424 paged arena load: a page's token rows are not consecutive from its first id")

    t = object.__new__(UnifiedRadixCache)
    t.cache_controller = types.SimpleNamespace(write=_raise, mem_pool_host=pool, mem_pool_host_draft=None)
    node = types.SimpleNamespace(id=999)
    helper = getattr(t, "_weg2_write_or_abort", None)
    assert helper is not None, "the controller write has no abort path"
    with pytest.raises(RuntimeError, match="not consecutive"):
        helper(node, torch.arange(2 * P), [], pre, None)
    assert int(pool._pending_mask.sum()) == 0, "no page of the claim stays pending"
    assert arena.find_slots(["page-b"])[0][1] != 1, "page-b's fresh claim is freed"
    assert _refs(arena) == base
    assert arena.find_slots(["page-a"]) == [(slot_of["page-a"], 2)]
