"""#1424g (rc12q D-TP0, ``ARENA-REF-HOLDERS at=reset tree=0 sum=0
own_held=1016 gap=1016`` at the resets 15:56:23 and 16:00:05 -- 18.6 % of the
KV arena pinned by no holder, then ``ARENA-CLAIM REFUSED (4 = no free slot)``
and ``BACKUP-REFUSED why=arena_claim``): a reset must leave this process
holding exactly what a surviving holder names.

The metal shape: the flip park retained six requests, two of them with the
same 59648-token prefix (weg2-8-28 / weg2-8-29). A second read of a prefix the
tree already holds takes its own reader reference per page (the resolve,
``_arena_page_get``) and queues the unclaimed head for release
(``append_host_mem_release``); the release queue drains by the group's MIN of
the queue sizes, and the controller reset CLEARS the queues -- the head's
references lost their release for good. Every other reference the reset
leaves without a holder (a node the reset skipped, a slot two nodes named, a
reference taken outside every class) ends the same way.

The reset now gives the queued rows back before the controller drops them
(``_weg2_release_queued_refs_before_reset``) and, once the controller has
stopped, every reference of this process no surviving holder names
(``_weg2_release_orphan_refs``, cumulative ``reset_orphans`` in
ARENA-REF-HOLDERS). The census counts a prefetch's pages only once resolved
(27B rc12q b1: gap -132321 = -prefetch under load).

Hermetic: the real C arena (``ShmArena`` with the per-process ledger), the
real ``UnifiedRadixCache._reset_full`` on a tree shell, the real
``ArenaMHAHostPool`` release rules; the balance is read from the slot headers
and the ledger."""
from __future__ import annotations

import ctypes
import os
import queue as _queue
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from sglang.srt.mem_cache.pool_host import arena_pool as ap  # noqa: E402
from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena  # noqa: E402

P = 4
SLOTS = 16
SLOT_BYTES = 256
S = 5  # staging rows before the arena ids

PREFIX = ["a0", "a1", "a2"]          # the shared prefix of the two parked rids
TAIL_28 = ["b3"]                      # weg2-8-28's own tail page
TAIL_29 = ["c3"]                      # weg2-8-29's own tail page


@pytest.fixture
def arena(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_HICACHE_ARENA_QUEUE_REFS", "1")
    a = ShmArena(str(tmp_path / "arena-kv-1424g.bin"), SLOT_BYTES, SLOTS)
    yield a
    a.close()


def _publish(arena, stems):
    """COMPLETE pages, no reference (P's writer let them go)."""
    out = {}
    for stem in stems:
        (s, st, g), = arena.claim_slots([stem], [SLOT_BYTES])
        assert st == 0
        assert arena.complete_slots([s], [g], [(0, SLOT_BYTES)]) == [1]
        out[stem] = s
    return out


def _refs(arena):
    """Every slot's reader references, read from the headers (all processes)."""
    out = (ctypes.c_int64 * 6)()
    arena._lib.arena_layout(arena.slots, arena.slot_bytes, out)
    hb, hoff = int(out[0]), int(out[3])
    u32 = np.frombuffer(arena._mm, dtype=np.uint32, count=arena.slots * hb // 4, offset=hoff)
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


def _rows(slots):
    return torch.cat([torch.arange(s * P, s * P + P) for s in slots]) + S


def _resolve(arena, slot_of, keys):
    """The read resolve: one reader reference per page key, rows = its slots."""
    slots = [slot_of[k] for k in keys]
    assert arena.ref_slots_np(np.asarray(slots, dtype=np.int64), +1) == len(slots)
    return _rows(slots)


class _Controller:
    """The controller surface the reset touches: the release queues, which
    ``reset()`` clears under storage (HybridCacheController.reset)."""

    def __init__(self, pool):
        self.enable_storage = True
        self.host_mem_release_queue = _queue.Queue()
        self.extra_host_mem_release_queues = {}
        self.mem_pool_host = types.SimpleNamespace(arena_read=True, clear=lambda: None)
        self.resets = 0

    def entry_for_extra_release(self, name):
        return None

    def reset(self):
        self.resets += 1
        self.host_mem_release_queue.queue.clear()
        for q in self.extra_host_mem_release_queues.values():
            q.queue.clear()


def _tree(arena, pool):
    """A D tree shell `_reset_full` runs on (FULL component only)."""
    from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    FULL = ComponentType.FULL
    t = object.__new__(UnifiedRadixCache)
    t._components_tuple = (types.SimpleNamespace(component_type=FULL, _full_kv_pool_host=pool),)
    t.tree_components = (FULL,)
    t.cache_controller = _Controller(pool)
    t.ongoing_prefetch = {}
    t._retired_prefetch = []
    t._weg2_carrier_rows = {}
    t.session = types.SimpleNamespace(slots={})
    t.device = "cpu"
    t._drop_staging_write_ring = lambda: None
    t._init_pin_trace = lambda: None
    t._record_all_cleared_event = lambda: None
    t._weg2_orphan_sweep_armed = True   # init_hicache arms the real tree
    return t, FULL


def _node(FULL, nid, rows, children=None, lock=0):
    cd = types.SimpleNamespace(host_value=rows, host_lock_ref=lock)
    return types.SimpleNamespace(id=nid, children=children or {}, write_through_pending_id=None,
                                 component_data={FULL: cd})


def _parked_two_rids_same_prefix(arena):
    """The metal shape after the flip park: the tree holds the shared prefix
    once (node 1) and each rid's own tail (nodes 2, 3), one reference per
    page; the second rid's hold read resolved the prefix AGAIN (its own
    reference per page) and queued the unclaimed head -- still in the queue
    when the sleep's reset comes (the MIN-gated drain had not run)."""
    slot_of = _publish(arena, PREFIX + TAIL_28 + TAIL_29)
    pool = _pool(arena)
    t, FULL = _tree(arena, pool)
    tail28 = _node(FULL, 2, _resolve(arena, slot_of, TAIL_28))
    tail29 = _node(FULL, 3, _resolve(arena, slot_of, TAIL_29))
    prefix = _node(FULL, 1, _resolve(arena, slot_of, PREFIX), children={1: tail28, 2: tail29})
    t.root_node = types.SimpleNamespace(children={1: prefix})
    head = _resolve(arena, slot_of, PREFIX)            # weg2-8-29's second read of the prefix
    t.cache_controller.host_mem_release_queue.put(head)
    return t, pool, slot_of


def test_two_rids_same_prefix_park_then_reset_leaves_no_reference(arena):
    """Before the reset this process holds 2 per prefix page (tree + queued
    head) and 1 per tail page; after it, 0 everywhere -- headers and ledger.
    Without #1424g the controller reset dropped the queued head and 3 references
    (one per prefix page) stayed with no holder: the rc12q gap."""
    t, _pool_, slot_of = _parked_two_rids_same_prefix(arena)
    refs = _refs(arena)
    assert [refs[slot_of[k]] for k in PREFIX] == [2, 2, 2]
    assert _own(arena) == 8
    line = t.weg2_arena_holder_census(arena)
    assert "tree=5 " in line and "queue=3 " in line and "gap=0" in line, line
    t._reset_full()
    assert t.cache_controller.resets == 1
    assert _refs(arena) == [0] * SLOTS
    assert _own(arena) == 0
    assert arena._ledger.refused == 0
    line = t.weg2_arena_holder_census(arena)
    assert "own_held=0 gap=0" in line, line


def test_a_reference_no_holder_names_is_given_back_by_the_reset(arena, caplog):
    """A reference this process took outside every holder class (the rc12q
    gap: no node, no record, no queue entry) is given back by the reset,
    named (RESET-ORPHANS) and counted (reset_orphans=)."""
    import logging

    slot_of = _publish(arena, PREFIX)
    pool = _pool(arena)
    t, FULL = _tree(arena, pool)
    t.root_node = types.SimpleNamespace(children={1: _node(FULL, 1, _resolve(arena, slot_of, PREFIX))})
    assert arena.ref_slots([slot_of["a1"]], +1) == 1          # nobody's class
    assert "gap=1" in t.weg2_arena_holder_census(arena)
    with caplog.at_level(logging.INFO):
        t._reset_full()
    assert _refs(arena) == [0] * SLOTS and _own(arena) == 0
    msgs = [r.getMessage() for r in caplog.records if "RESET-ORPHANS" in r.getMessage()]
    assert msgs and "released=1 slots=1" in msgs[0], msgs
    assert "reset_orphans=1" in t.weg2_arena_holder_census(arena)


def test_a_node_the_release_skips_is_given_back_after_the_controller_stopped(arena):
    """#1424e named it (RESET-SKIPPED: a host-locked node's reference, its
    in-flight op dropped with the tree); the reset's orphan pass returns it."""
    slot_of = _publish(arena, PREFIX)
    pool = _pool(arena)
    t, FULL = _tree(arena, pool)
    locked = _node(FULL, 1, _resolve(arena, slot_of, PREFIX), lock=1)
    t.root_node = types.SimpleNamespace(children={1: locked})
    t._reset_full()
    assert _refs(arena) == [0] * SLOTS and _own(arena) == 0


def test_the_carrier_hold_survives_the_reset(arena):
    """Group P: the END anchors held across D's phase are a named holder --
    the orphan pass keeps them (released at P's next wake)."""
    slot_of = _publish(arena, PREFIX)
    pool = _pool(arena)
    t, FULL = _tree(arena, pool)
    held = _resolve(arena, slot_of, ["a2"])
    t._weg2_carrier_rows = {id(pool): (pool, [held])}
    t.root_node = types.SimpleNamespace(children={})
    t._reset_full()
    refs = _refs(arena)
    assert refs[slot_of["a2"]] == 1 and _own(arena) == 1


def test_another_process_reference_is_never_taken(arena):
    """A reference another rank process holds on the same slot (header only,
    not in this ledger) stays: the orphan pass releases through the ledger."""
    slot_of = _publish(arena, PREFIX)
    pool = _pool(arena)
    t, FULL = _tree(arena, pool)
    c = (ctypes.c_int64 * 1)(slot_of["a0"])
    assert arena._lib.arena_ref_slots(arena._base, 1, c, 1) == 1   # a foreign +1
    t.root_node = types.SimpleNamespace(children={1: _node(FULL, 1, _resolve(arena, slot_of, ["a0"]))})
    t._reset_full()
    assert _refs(arena)[slot_of["a0"]] == 1 and _own(arena) == 0


def test_census_counts_a_prefetch_only_as_far_as_it_resolved(arena):
    """27B rc12q b1 (gap -132321 = -prefetch): the record's rows beyond
    ``operation.completed_tokens`` hold no reference yet -- not counted."""
    slot_of = _publish(arena, PREFIX)
    pool = _pool(arena)
    t, FULL = _tree(arena, pool)
    t.root_node = types.SimpleNamespace(children={})
    rows = torch.cat([_resolve(arena, slot_of, ["a0"]), _rows([slot_of["a1"], slot_of["a2"]])])
    op = types.SimpleNamespace(completed_tokens=P)
    t.ongoing_prefetch = {"r": types.SimpleNamespace(host_indices=rows, operation=op, comp_xfers={})}
    line = t.weg2_arena_holder_census(arena)
    assert "prefetch=1 " in line and "own_held=1 gap=0" in line, line


def test_another_armed_tree_of_this_process_keeps_its_references(arena):
    """Two trees of one process on one arena share its ledger: the reset of
    one names the other's holders (armed trees of the process), so the other
    tree's node keeps its reference; an unarmed shell sweeps nothing."""
    from sglang.srt.mem_cache import unified_radix_cache as u

    slot_of = _publish(arena, PREFIX)
    a, FULL = _tree(arena, _pool(arena))
    b, _ = _tree(arena, _pool(arena))
    a.root_node = types.SimpleNamespace(children={1: _node(FULL, 1, _resolve(arena, slot_of, ["a0"]))})
    b.root_node = types.SimpleNamespace(children={1: _node(FULL, 2, _resolve(arena, slot_of, ["a1"]))})
    u._WEG2_ARMED_TREES.add(b)
    try:
        a._reset_full()
        refs = _refs(arena)
        assert refs[slot_of["a0"]] == 0 and refs[slot_of["a1"]] == 1 and _own(arena) == 1
    finally:
        u._WEG2_ARMED_TREES.discard(b)
    c, _ = _tree(arena, _pool(arena))
    c._weg2_orphan_sweep_armed = False
    c.root_node = types.SimpleNamespace(children={})
    c._reset_full()
    assert _own(arena) == 1, "an unarmed tree gives nothing back"
