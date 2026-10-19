"""S83 (rc18 boot ``..._1019_200330``, D group TP=3, Form A): the KV arena full at the
sleep flush, no victim, un-backed nodes dropped by the reset.

Metal (D.log): 20:21:22 ``ARENA-REF-HOLDERS`` TP0 ``tree=3931 queue=0``, TP1/TP2
``tree=3931 queue=4160``, ``arena_pinned=5180`` (P ``own_held=0``, every other holder
class 0) -- 1249 of 6485 slots pinned by the WORKERS' queued release entries alone;
20:22:10 ``#1427 ARENA-DROP freed=0 stages=i:0,ii:0,iii:0`` on all ranks, 20:22:12
``#1470 FLUSH-PUBLISH ARENA-FULL-STOP unbacked=16``, 20:22:13 ``#1470 FLUSH-UNBACKED
nodes=16 tokens=8512``.

The cause: ``drain_storage_control_queues`` releases ``MIN(qsize)`` entries over the TP
group; TP0 queues none (it released at once), so the MIN is 0 and the workers' arena
references wait for the reset. The fix (``_s83_release_queue_surplus_refs``): after the
group drain, and before each #1470 round, each rank gives back the ARENA references
still queued on it -- through the drain's own rule (``HostPoolGroup.free`` ->
``_free_arena_rows`` -> ``release_queued_rows``, the per-process RefLedger); non-arena
rows stay queued in place for the MIN.

Hermetic: the real C arena (``ShmArena`` with the per-process ledger), the real
``ArenaMHAHostPool`` release rules, the real ``HostPoolGroup.split_arena_rows`` /
``free``; the balance is read from the slot headers and the ledger. Red with
``FLLIPER_PDFLIP_RELEASE_QUEUE_LOCAL_REFS=0`` (the MIN-only release), green with the
default."""
from __future__ import annotations

import ctypes
import logging
import os
import queue as _queue
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.mem_cache.memory_pool_host import HostPoolGroup  # noqa: E402
from flliper.srt.mem_cache.pool_host import arena_pool as ap  # noqa: E402
from flliper.srt.mem_cache.storage.file.hicache_arena import ShmArena  # noqa: E402

P = 4
SLOTS = 8
SLOT_BYTES = 256
S = 5  # staging rows before the arena ids

TREE = ["t0", "t1", "t2"]       # a device node the tree still backs on the host (1 ref per page)
GONE = ["g0", "g1", "g2", "g3", "g4"]   # pages of nodes the tree let go; only the queue still names them


@pytest.fixture
def arena(tmp_path, monkeypatch):
    monkeypatch.setenv("FLLIPER_HICACHE_ARENA_QUEUE_REFS", "1")
    a = ShmArena(str(tmp_path / "arena-kv-s83.bin"), SLOT_BYTES, SLOTS)
    yield a
    a.close()


def _publish(arena, stems):
    out = {}
    for stem in stems:
        (s, st, g), = arena.claim_slots([stem], [SLOT_BYTES])
        assert st == 0
        assert arena.complete_slots([s], [g], [(0, SLOT_BYTES)]) == [1]
        out[stem] = s
    return out


def _refs(arena):
    out = (ctypes.c_int64 * 6)()
    arena._lib.arena_layout(arena.slots, arena.slot_bytes, out)
    hb, hope = int(out[0]), int(out[3])
    u32 = np.frombuffer(arena._mm, dtype=np.uint32, count=arena.slots * hb // 4, offset=hope)
    return u32.reshape(arena.slots, hb // 4)[:, 1].astype(np.int64).tolist()


def _own(arena):
    return int(arena._ledger.held.sum())


def _pool(arena):
    pool = object.__new__(ap.ArenaMHAHostPool)
    pool._arena_page_tokens = P
    pool.staging_rows = S
    pool.arena = arena
    pool.arena_tokens = SLOTS * P
    pool.row_slot = None
    pool._pending_mask = None
    pool._pending = {}
    return pool


def _group(pool):
    g = object.__new__(HostPoolGroup)
    g.anchor_entry = types.SimpleNamespace(host_pool=pool)
    return g


def _rows(slots):
    return torch.cat([torch.arange(s * P, s * P + P) for s in slots]) + S


def _resolve(arena, slot_of, keys):
    slots = [slot_of[k] for k in keys]
    assert arena.ref_slots_np(np.asarray(slots, dtype=np.int64), +1) == len(slots)
    return _rows(slots)


def _tree(arena, pool):
    from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType
    from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    FULL = ComponentType.FULL
    t = object.__new__(UnifiedRadixCache)
    t.page_size = P
    t.cache_controller = types.SimpleNamespace(
        enable_storage=True,
        host_mem_release_queue=_queue.Queue(),
        extra_host_mem_release_queues={},
        prefetch_revoke_queue=_queue.Queue(),
        ack_backup_queue=_queue.Queue(),
        mem_pool_host=_group(pool),
    )
    t.root_node = types.SimpleNamespace(children={})
    return t, FULL


def _node(FULL, nid, rows, device=True):
    cd = types.SimpleNamespace(host_value=rows, value=(torch.zeros(1) if device else None), host_lock_ref=0)
    return types.SimpleNamespace(id=nid, children={}, component_data={FULL: cd})


def _worker_after_tp0_released(arena):
    """The rc18 shape on ONE worker process: the tree backs TREE (1 ref per page); a
    store prefetch resolved TREE again and queued its unclaimed head (a 2nd ref per
    page, #989b); the nodes of GONE left the tree, their rows' references queued as one
    entry each. TP0 (another process) has given its own references back already."""
    slot_of = _publish(arena, TREE + GONE)
    pool = _pool(arena)
    t, FULL = _tree(arena, pool)
    t.root_node.children = {1: _node(FULL, 1, _resolve(arena, slot_of, TREE))}
    q = t.cache_controller.host_mem_release_queue
    q.put(_resolve(arena, slot_of, TREE))            # the unclaimed head: 2nd ref on the tree's pages
    for k in GONE:
        q.put(_resolve(arena, slot_of, [k]))          # a page only the queue still references
    return t, pool, slot_of


def test_red_min_only_release_keeps_the_queue_only_slots_pinned(arena, monkeypatch):
    """Switch off (the MIN-only form): nothing is given back, the GONE slots stay
    referenced and the claim clock finds no victim -- the rc18 ARENA-DROP freed=0."""
    monkeypatch.setenv("FLLIPER_PDFLIP_RELEASE_QUEUE_LOCAL_REFS", "0")
    t, _pool_, slot_of = _worker_after_tp0_released(arena)
    assert t._s83_release_queue_surplus_refs(group_min=0) == 0
    refs = _refs(arena)
    assert [refs[slot_of[k]] for k in GONE] == [1] * len(GONE)
    assert arena.evict_candidates(SLOTS) == []          # every COMPLETE slot referenced
    assert t.cache_controller.host_mem_release_queue.qsize() == 1 + len(GONE)


def test_green_queued_arena_refs_go_back_and_the_clock_finds_victims(arena):
    """Default: the queued references go back (one per row of P ids); the GONE slots are
    unreferenced -> the clock takes them; the tree's pages keep exactly the tree's
    reference; the ledger equals the tree; the queue is empty."""
    t, _pool_, slot_of = _worker_after_tp0_released(arena)
    rows = t._s83_release_queue_surplus_refs(group_min=0)
    assert rows == (len(TREE) + len(GONE)) * P
    refs = _refs(arena)
    assert [refs[slot_of[k]] for k in GONE] == [0] * len(GONE)
    assert [refs[slot_of[k]] for k in TREE] == [1] * len(TREE)
    assert _own(arena) == len(TREE)
    assert arena._ledger.refused == 0
    assert t.cache_controller.host_mem_release_queue.qsize() == 0
    got = sorted(c[0] for c in arena.evict_candidates(SLOTS))
    assert got == sorted(slot_of[k] for k in GONE)


def test_non_arena_rows_stay_queued_in_place_for_the_min(arena):
    """A mixed entry keeps its place with only its staging rows; a pure staging entry is
    untouched; their order is unchanged (the MIN drain still frees them as before)."""
    t, _pool_, slot_of = _worker_after_tp0_released(arena)
    q = t.cache_controller.host_mem_release_queue
    q.queue.clear()
    staging_a = torch.tensor([0, 1], dtype=torch.int64)
    mixed = torch.cat([torch.tensor([2, 3], dtype=torch.int64), _resolve(arena, slot_of, ["g0"])])
    staging_b = torch.tensor([4], dtype=torch.int64)
    q.put(staging_a)
    q.put(mixed)
    q.put(staging_b)
    before = _refs(arena)[slot_of["g0"]]       # the cleared fixture entry's reference stays (not ours to drop)
    rows = t._s83_release_queue_surplus_refs(group_min=0)
    assert rows == P
    left = list(q.queue)
    assert len(left) == 3
    assert left[0] is staging_a and left[2] is staging_b
    assert left[1].tolist() == [2, 3]
    assert _refs(arena)[slot_of["g0"]] == before - 1


def test_another_process_reference_is_never_taken(arena):
    """Rank unity / ledger bound: a reference another rank process holds on a queued
    page stays (the header keeps it); only this process's own reference goes."""
    t, _pool_, slot_of = _worker_after_tp0_released(arena)
    c = (ctypes.c_int64 * 1)(slot_of["g1"])
    assert arena._lib.arena_ref_slots(arena._base, 1, c, 1) == 1    # a foreign +1 (header only)
    t._s83_release_queue_surplus_refs(group_min=0)
    assert _refs(arena)[slot_of["g1"]] == 1
    assert int(arena._ledger.held[slot_of["g1"]]) == 0


def test_a_second_pass_releases_nothing_twice(arena):
    t, _pool_, slot_of = _worker_after_tp0_released(arena)
    t._s83_release_queue_surplus_refs(group_min=0)
    assert t._s83_release_queue_surplus_refs(group_min=0) == 0
    assert arena._ledger.refused == 0
    assert _own(arena) == len(TREE)


def test_pass_bound_leaves_the_rest_for_the_next_pass(arena, monkeypatch):
    from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    t, _pool_, slot_of = _worker_after_tp0_released(arena)
    monkeypatch.setattr(UnifiedRadixCache, "S83_QUEUE_REF_PASS_MAX", 2)
    assert t._s83_release_queue_surplus_refs() == (len(TREE) + 1) * P    # head entry (3 pages) + g0
    assert t.cache_controller.host_mem_release_queue.qsize() == len(GONE) - 1
    while t._s83_release_queue_surplus_refs():
        pass
    assert [_refs(arena)[slot_of[k]] for k in GONE] == [0] * len(GONE)


def test_drain_releases_the_workers_refs_although_the_group_min_is_zero(arena, monkeypatch):
    """Wiring: ``drain_storage_control_queues`` with TP0's queue empty (MIN = 0 for the
    release queue): the group impl drains nothing, the S83 pass gives the arena
    references back on this rank anyway."""
    from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    t, _pool_, slot_of = _worker_after_tp0_released(arena)
    seen = {}

    def _ar(self, tensor, op, label=""):
        tensor.zero_()           # TP0 has queued nothing: every MIN is 0

    def _impl(self, **kw):
        seen.update(kw)

    monkeypatch.setattr(UnifiedRadixCache, "_all_reduce_attn_groups", _ar, raising=False)
    monkeypatch.setattr(UnifiedRadixCache, "_drain_storage_control_queues_impl", _impl)
    t._hc_parts = None
    t.drain_storage_control_queues()
    assert seen.get("n_release") == 0
    assert [_refs(arena)[slot_of[k]] for k in GONE] == [0] * len(GONE)
    assert t.cache_controller.host_mem_release_queue.qsize() == 0


def test_victim_census_names_queue_only_slots_and_tree_classes(arena, caplog):
    """The instrument line: queue_only_slots = slots only this rank's queue references;
    tree_device / tree_host_only split; unpinned_complete = the clock's candidates."""
    t, pool, slot_of = _worker_after_tp0_released(arena)
    from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType

    t.root_node.children[2] = _node(ComponentType.FULL, 2, _resolve(arena, slot_of, ["g4"]), device=False)
    t._s83_spill_why = {"aux_host_only": [2, 7]}
    with caplog.at_level(logging.INFO):
        line = t.s83_victim_census("flush_stop", pool=pool)
    assert "site=flush_stop" in line
    assert "queue_only_slots=4 " in line, line            # g0..g3 (g4 is a host-only tree node now)
    assert "tree_device=3 " in line and "tree_host_only=1 " in line, line
    assert "unpinned_complete=0 " in line, line
    assert "spill_why=aux_host_only:2/7" in line, line
    assert any("S83 VICTIM-CENSUS" in r.getMessage() for r in caplog.records)
    t._s83_release_queue_surplus_refs()
    line = t.s83_victim_census("flush_end", pool=pool)
    assert "queue_only_slots=0 " in line and "unpinned_complete=4 " in line, line


def test_victim_census_rate_limit_spares_the_flush_sites(arena):
    t, pool, _slot_of = _worker_after_tp0_released(arena)
    assert t.s83_victim_census("spill", pool=pool) is not None
    assert t.s83_victim_census("spill", pool=pool) is None          # within 5 s
    assert t.s83_victim_census("flush_stop", pool=pool) is not None
    assert t.s83_victim_census("flush_stop", pool=pool) is not None


# ---- review A1 (nf-review-s83-1019): the three danger directions the first test set left open ----


def test_review_a1_order_beyond_the_pass_bound_is_kept(arena, monkeypatch):
    """Mc: with more entries than one pass walks, the kept (non-arena) parts stay AHEAD of
    the entries the pass did not reach -- else ranks with different queue lengths would
    MIN-drain different entries (the mirrored-allocator break the MIN exists for)."""
    from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    t, _pool_, slot_of = _worker_after_tp0_released(arena)
    q = t.cache_controller.host_mem_release_queue
    q.queue.clear()
    a = torch.tensor([0, 1], dtype=torch.int64)
    mixed = torch.cat([torch.tensor([2, 3], dtype=torch.int64), _resolve(arena, slot_of, ["g0"])])
    b = torch.tensor([4], dtype=torch.int64)
    for e in (a, mixed, b):
        q.put(e)
    monkeypatch.setattr(UnifiedRadixCache, "S83_QUEUE_REF_PASS_MAX", 2)
    t._s83_release_queue_surplus_refs(group_min=0)
    left = list(q.queue)
    assert left[0] is a and left[1].tolist() == [2, 3] and left[2] is b


def test_review_a1_two_entries_on_the_same_pages_return_two_references(arena):
    """Me / #989b: two store prefetches of one prefix each queue their own reference on the
    same pages -- both go back (never merged and de-duplicated into one)."""
    t, _pool_, slot_of = _worker_after_tp0_released(arena)
    q = t.cache_controller.host_mem_release_queue
    q.queue.clear()
    s = slot_of["g2"]
    before = _refs(arena)[s]                       # the cleared fixture entry's reference stays held
    q.put(_resolve(arena, slot_of, ["g2"]))        # prefetch 1: +1
    q.put(_resolve(arena, slot_of, ["g2"]))        # prefetch 2: +1, the same page
    assert _refs(arena)[s] == before + 2
    t._s83_release_queue_surplus_refs(group_min=0)
    assert _refs(arena)[s] == before


def test_review_a1_a_pending_write_keeps_its_reference(arena):
    """Mb: rows of a pending write belong to their writer -- the early rank-local release
    skips them (release_queued_rows, _pending)."""
    t, pool, slot_of = _worker_after_tp0_released(arena)
    q = t.cache_controller.host_mem_release_queue
    q.queue.clear()
    s = slot_of["g3"]
    q.put(_resolve(arena, slot_of, ["g3"]))
    held = _refs(arena)[s]
    pool._pending = {s: object()}                  # a write in flight on that slot
    t._s83_release_queue_surplus_refs(group_min=0)
    assert _refs(arena)[s] == held                 # not taken
