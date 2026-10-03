"""#1424e (rc12p fced72cf5b, D-TP0 ARENA-REF-CENSUS of arena-786432.bin three
minutes after OBEN: complete 4669 of 5461, pinned 4315): a re-pointed page of
a load-back chain kept the reader reference on its OLD slot for good.

A page of a load-back chain holds one reference on the slot its rows address
-- the read resolve (``_arena_page_get``) takes +1 per page key, a publish
(``complete_write``) +1 -- and the node's release (``free`` /
``release_tree_rows``) gives back the slots its rows address at THAT time.
The #1424c/d proof re-points a page (re-keyed to its tokens' key, or rows
that were not its key's slot) and referenced the new slot, but never gave the
old one back: after the re-point the release hands back the new slot, the old
one stays pinned -- one leaked reference per REPAIRED/REKEYED page, never
evictable again.

The re-point now gives the old reference back (``_repoint_unrefs``), and only
one that is really held: no whole arena page (staging rows), a pending write
of this pool, or a slot this process holds no reference on gives nothing
back -- a counter never goes below what exists and never takes another
holder's reference.

Hermetic: the real C arena (``ShmArena`` on a temp file, with and without the
per-process reference ledger), the real ``verify_load_chain`` and
``ArenaMHAHostPool.release_tree_rows``; the balance is read from the slot
headers: before the load, after the re-point, after the tree's release."""
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
SLOTS = 16
SLOT_BYTES = 256
S = 5  # staging rows before the arena ids

# P's hand-off chain of the conversation, indexed from token 0 (page j -> key)
CHAIN = ["a0", "a1", "a2", "a3", "a4", "a5"]


@pytest.fixture(params=["ledger", "header"])
def arena(tmp_path, monkeypatch, request):
    """``ledger``: SGLANG_HICACHE_ARENA_QUEUE_REFS on (the rig's form, the
    per-process reference ledger); ``header``: off (the C count alone)."""
    monkeypatch.setenv("SGLANG_HICACHE_ARENA_QUEUE_REFS", "1" if request.param == "ledger" else "0")
    a = ShmArena(str(tmp_path / f"arena-kv-{request.param}.bin"), SLOT_BYTES, SLOTS)
    yield a
    a.close()


def _publish(arena, stems):
    """COMPLETE pages, no reference (a writer's node lets them go)."""
    out = {}
    for stem in stems:
        (s, st, g), = arena.claim_slots([stem], [SLOT_BYTES])
        assert st == 0
        assert arena.complete_slots([s], [g], [(0, SLOT_BYTES)]) == [1]
        out[stem] = s
    return out


def _refs(arena):
    """Every slot's reader references, read from the headers (all ranks)."""
    out = (ctypes.c_int64 * 6)()
    arena._lib.arena_layout(arena.slots, arena.slot_bytes, out)
    hb, hoff = int(out[0]), int(out[3])
    u32 = np.frombuffer(arena._mm, dtype=np.uint32, count=arena.slots * hb // 4, offset=hoff)
    return u32.reshape(arena.slots, hb // 4)[:, 1].astype(np.int64).tolist()


def _refused(arena):
    return arena._ledger.refused if arena._ledger is not None else 0


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


def _resolved(arena, slot_of, nid, keys):
    """A node as the read resolve leaves it (``_arena_page_get``): its rows are
    the slots its stored keys name, one reference per page."""
    slots = [slot_of[k] for k in keys]
    assert arena.ref_slots_np(np.asarray(slots, dtype=np.int64), +1) == len(slots)
    return types.SimpleNamespace(id=nid, hash_value=list(keys), host_value=_rows(slots))


def _verify(pool, nodes, **kw):
    host = torch.cat([n.host_value for n in nodes])
    return ap.verify_load_chain(pool, nodes, host, rows_of=lambda n: n.host_value,
                                stems_of=lambda h: list(h), p_chain=CHAIN, page0=1,
                                prior0="a0", **kw)


def _shifted(arena, slot_of):
    """Depth 4 carrying a5 (P's key of depth 5): the chain is proven page by
    page, so the odd page before it is examined too."""
    return _resolved(arena, slot_of, 14, ["a5"])


def _release(pool, nodes):
    """The tree reset (flip / sleep): every node's host rows given back."""
    return pool.release_tree_rows(torch.cat([n.host_value for n in nodes]))


def test_twin_the_metal_shape_balances_before_repoint_and_after_release(arena):
    """RED on fced72cf5b. rc12n's shape (#1424c): the second piece carries the
    keys of one page earlier (a3,a4 instead of a4,a5), so a3's slot is in the
    chain twice. The re-key re-points both pages: a3's slot and a4's slot each
    had one reference too many afterwards, and kept it past the tree's
    release."""
    slot_of = _publish(arena, CHAIN)
    base = _refs(arena)
    assert sum(base) == 0
    pool = _pool(arena)
    nodes = [_resolved(arena, slot_of, 12, ["a1", "a2"]),
             _resolved(arena, slot_of, 13, ["a3"]),
             _resolved(arena, slot_of, 14, ["a3", "a4"])]      # shifted: should be a4, a5
    loaded = _refs(arena)
    assert loaded[slot_of["a3"]] == 2, "the twin page took its own reference on a3's slot"
    host = _verify(pool, nodes)
    assert nodes[2].hash_value == ["a4", "a5"]
    assert ((host - S).view(-1, P)[:, 0] // P).tolist() == [slot_of[k] for k in CHAIN[1:6]]
    after = _refs(arena)
    for k in CHAIN[1:6]:
        assert after[slot_of[k]] == 1, f"one reference per page of the proven chain ({k})"
    assert after[slot_of["a0"]] == 0
    assert _release(pool, nodes) == 5
    assert _refs(arena) == base, "the tree's release gives back everything the load took"
    assert _refused(arena) == 0


def test_single_shifted_key_without_twin_balances(arena):
    """RED on fced72cf5b. A node at depth 3 carries a4 (P's key of depth 4)
    and a4's rows -- a clean-looking chain, no twin. Re-keyed to a3: a4's
    slot had kept the page's reference after the release."""
    slot_of = _publish(arena, CHAIN)
    base = _refs(arena)
    pool = _pool(arena)
    nodes = [_resolved(arena, slot_of, 12, ["a1", "a2"]),
             _resolved(arena, slot_of, 13, ["a4"])]             # depth 3: should be a3
    _verify(pool, nodes)
    assert nodes[1].hash_value == ["a3"]
    after = _refs(arena)
    assert after[slot_of["a4"]] == 0, "the old slot's reference is given back with the re-point"
    assert [after[slot_of[k]] for k in ("a1", "a2", "a3")] == [1, 1, 1]
    assert _release(pool, nodes) == 3
    assert _refs(arena) == base
    assert _refused(arena) == 0


def test_staging_rows_held_no_slot_reference_and_give_none_back(arena):
    """D's own page whose rows are still staging rows (no arena page, no
    ``_arena_page_get``): re-pointed to its key's slot, nothing to give back,
    no counter touched beyond the new reference."""
    slot_of = _publish(arena, CHAIN)
    base = _refs(arena)
    pool = _pool(arena)
    n12 = _resolved(arena, slot_of, 12, ["a1", "a2"])
    n13 = types.SimpleNamespace(id=13, hash_value=["a3"], host_value=torch.arange(0, P))
    n14 = _shifted(arena, slot_of)
    before = _refs(arena)
    _verify(pool, [n12, n13, n14])
    after = _refs(arena)
    assert after[slot_of["a3"]] == 1 and after[slot_of["a4"]] == 1
    drops = [i for i, (a, b) in enumerate(zip(after, before)) if a < b]
    assert drops == [slot_of["a5"]], "the only -1 is the shifted page's own old slot"
    _release(pool, [n12, n13, n14])
    assert _refs(arena) == base
    assert _refused(arena) == 0


def test_a_pending_write_of_this_pool_keeps_its_writer_reference(arena):
    """The old rows address a slot of THIS pool's write still in flight (a
    claim not acked): its reference is its writer's, it is never given back by
    a re-point."""
    slot_of = _publish(arena, CHAIN)
    pool = _pool(arena)
    (w, st, g), = arena.claim_slots(["d-own-in-flight"], [SLOT_BYTES])
    assert st == 0
    pool._pending[w] = (g, True)
    assert arena.ref_slots([w], +1) == 1                      # the writer node's reference
    n12 = _resolved(arena, slot_of, 12, ["a1", "a2"])
    n13 = types.SimpleNamespace(id=13, hash_value=["a3"], host_value=_rows([w]))
    _verify(pool, [n12, n13, _shifted(arena, slot_of)])
    assert _refs(arena)[w] == 1, "the writer's reference stays"
    assert w in pool._pending
    assert _refs(arena)[slot_of["a3"]] == 1


def test_a_slot_this_process_holds_no_reference_on_is_never_released(tmp_path, monkeypatch):
    """Ledger form (the rig's): the old rows address a COMPLETE slot another
    process references (another rank's reader), this process holds none there
    -- the re-point gives nothing back, the other holder keeps its
    reference, the ledger refuses nothing."""
    monkeypatch.setenv("SGLANG_HICACHE_ARENA_QUEUE_REFS", "1")
    arena = ShmArena(str(tmp_path / "arena-kv-foreign.bin"), SLOT_BYTES, SLOTS)
    try:
        slot_of = _publish(arena, CHAIN + ["zz"])
        x = slot_of["zz"]
        c = (ctypes.c_int64 * 1)(x)
        assert arena._lib.arena_ref_slots(arena._base, 1, c, 1) == 1   # the other process's +1
        pool = _pool(arena)
        n12 = _resolved(arena, slot_of, 12, ["a1", "a2"])
        n13 = types.SimpleNamespace(id=13, hash_value=["a3"], host_value=_rows([x]))
        _verify(pool, [n12, n13, _shifted(arena, slot_of)])
        assert _refs(arena)[x] == 1, "the other holder's reference is untouched"
        assert _refused(arena) == 0
        assert _refs(arena)[slot_of["a3"]] == 1
    finally:
        arena.close()


# -- the holder census (ARENA-REF-HOLDERS) and the reset's named skip --------

def _tree(arena, slot_of):
    """A D tree as the census reads it: a free host node (2 pages), a host
    node an in-flight op has locked (1 page), an open store prefetch (1 page)
    and a release-queue entry (1 page) -- each with the reference its path
    takes (the resolve's +1 per page)."""
    import queue as _queue

    from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    FULL = ComponentType.FULL
    pool = _pool(arena)

    def _node(nid, keys, lock=0):
        rows = _resolved(arena, slot_of, nid, keys).host_value
        cd = types.SimpleNamespace(host_value=rows, host_lock_ref=lock)
        return types.SimpleNamespace(id=nid, children={}, write_through_pending_id=None,
                                     component_data={FULL: cd})

    free = _node(21, ["a1", "a2"])
    locked = _node(22, ["a3"], lock=1)
    root = types.SimpleNamespace(children={1: free, 2: locked})
    rec = types.SimpleNamespace(host_indices=_resolved(arena, slot_of, 0, ["a4"]).host_value,
                                comp_xfers={})
    q = _queue.Queue()
    q.put(_resolved(arena, slot_of, 0, ["a5"]).host_value)
    t = object.__new__(UnifiedRadixCache)
    t._components_tuple = (types.SimpleNamespace(component_type=FULL, _full_kv_pool_host=pool),)
    t.root_node = root
    t.ongoing_prefetch = {"r1": rec}
    t._retired_prefetch = []
    t._weg2_carrier_rows = {}
    t.cache_controller = types.SimpleNamespace(
        host_mem_release_queue=q, mem_pool_host=types.SimpleNamespace(arena_read=True))
    return t, pool


def test_holder_census_names_every_class_and_the_gap(tmp_path, monkeypatch):
    """The census sums the holder classes against this process's ledger:
    tree 2 + in use 1 + prefetch 1 + queue 1 = own_held 5, gap 0; a reference
    no holder names (taken outside every class) shows as gap 1; another arena
    gets no line."""
    monkeypatch.setenv("SGLANG_HICACHE_ARENA_QUEUE_REFS", "1")
    arena = ShmArena(str(tmp_path / "arena-kv-census.bin"), SLOT_BYTES, SLOTS)
    other = ShmArena(str(tmp_path / "arena-other.bin"), SLOT_BYTES, SLOTS)
    try:
        slot_of = _publish(arena, CHAIN)
        t, _p = _tree(arena, slot_of)
        line = t.weg2_arena_holder_census(arena)
        assert ("pool=FULL tree=2 tree_in_use=1 prefetch=1 retired=0 queue=1 carrier=0 dormant_hold=0 "
                "sum=5 own_held=5 gap=0") in line, line
        assert arena.ref_slots([slot_of["a0"]], +1) == 1          # nobody's class
        assert "gap=1" in t.weg2_arena_holder_census(arena)
        assert t.weg2_arena_holder_census(other) is None
    finally:
        other.close()
        arena.close()


def test_census_thread_logs_the_holder_line(tmp_path, monkeypatch, caplog):
    """The 60-s census thread calls the registered provider: one
    ARENA-REF-HOLDERS line next to ARENA-REF-CENSUS, no round path."""
    import logging

    from sglang.srt.mem_cache.storage.file import hicache_arena as ha

    monkeypatch.setenv("SGLANG_HICACHE_ARENA_QUEUE_REFS", "1")
    arena = ShmArena(str(tmp_path / "arena-kv-thread.bin"), SLOT_BYTES, SLOTS)
    try:
        slot_of = _publish(arena, CHAIN)
        t, _p = _tree(arena, slot_of)
        ha.register_holder_census(t.weg2_arena_holder_census)
        with caplog.at_level(logging.INFO):
            lines = ha._holder_lines(arena)
        assert any("sum=5 own_held=5 gap=0" in x for x in lines), lines
    finally:
        arena.close()


def test_reset_names_the_skipped_in_use_node_and_releases_the_rest(tmp_path, monkeypatch, caplog):
    """The reset gives the free node's references back and SKIPS the locked
    one -- named now (#1424e RESET-SKIPPED, node, component, lock, rows); its
    reference stays (the leak class the census shows as gap after the
    reset)."""
    import logging

    monkeypatch.setenv("SGLANG_HICACHE_ARENA_QUEUE_REFS", "1")
    arena = ShmArena(str(tmp_path / "arena-kv-reset.bin"), SLOT_BYTES, SLOTS)
    try:
        slot_of = _publish(arena, CHAIN)
        t, _p = _tree(arena, slot_of)
        with caplog.at_level(logging.WARNING):
            released = t._release_host_values_before_reset()
        assert released == 2
        refs = _refs(arena)
        assert refs[slot_of["a1"]] == 0 and refs[slot_of["a2"]] == 0
        assert refs[slot_of["a3"]] == 1, "the locked node's reference is not given back"
        msgs = [r.getMessage() for r in caplog.records if "RESET-SKIPPED" in r.getMessage()]
        assert msgs and "22:FULL-lock1/rows4" in msgs[0], msgs
    finally:
        arena.close()
