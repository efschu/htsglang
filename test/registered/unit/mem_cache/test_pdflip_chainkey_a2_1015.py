"""A2 CHAIN-KEY 1015 (nf-h54-review-1015 B1 HIGH: latent silent corruption).

The class: a host-write writer whose ack comes AFTER its claim lost the key (an
H54 take-over, ``arena_quarantine_host_write``, or the JOIN-IDLE reap,
``arena_quarantine_stale``) gets ``arena_complete`` 6. ``complete_write``
takes no reference for it, but the node's host rows stayed on the old slot.
Once the slot is recycled for another key:

* the load-back proof passed the rows UNPROVEN -- a clean chain (whole distinct
  pages) took the fast path without looking at a key -- and loaded the other
  key's KV under these tokens (``verify_load_chain`` / ``provable_pages``);
* the node's release took, through the process-wide reference ledger (slot-
  exact, not generation-exact), a reference another node of the same process
  holds on the NEW content.

A2: (1) the fast path proves key -> slot -> COMPLETE for every page (one
``find_slots_np`` per chain), else the full proof (re-point or named stop on
the group-legal path); (2) ``complete_write`` 3..6 settles the node's rows at
the ack (re-point to the key's COMPLETE slot + reference, else orphan ids no
release and no proof passes); (3) the ledger stamps every reference with the
slot generation -- a release in a moved generation is refused; a re-point
gives the old slot's reference back only while the slot is still the page's
(its key, or QUARANTINE). B5: a writer giving up a QUARANTINE claim is
resolved (``release_claims`` / ``unclaim``) and the last reference of a
QUARANTINE slot without open writers frees it (``ref_slots`` -1).
"""

import os
import shutil
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["FLLIPER_HICACHE_ARENA_QUEUE_REFS"] = "1"

import numpy as np
import pytest
import torch

from flliper.srt.mem_cache.pool_host import arena_pool as ap
from flliper.srt.mem_cache.storage.file import hicache_arena as ha
from flliper.srt.mem_cache.storage.file.hicache_arena import ShmArena

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

P = 4
S = 16
TOTAL = 4096
HALF = TOTAL // 2


def _arena(tmp_path, slots, name="kv.bin", own_ledger=True):
    """One PROCESS's view of the shared arena file (its own reference ledger)."""
    a = ShmArena(str(tmp_path / name), TOTAL, slots)
    if own_ledger:
        a._ledger = ha.RefLedger(slots)
    return a


def _pool(arena, extents=((0, TOTAL),)):
    pool = object.__new__(ap.ArenaMHAHostPool)
    pool._arena_page_tokens = P
    pool.staging_rows = S
    pool.arena = arena
    pool.arena_slots = arena.slots
    pool.arena_tokens = arena.slots * P
    pool.row_slot = None
    pool._pending = {}
    pool._pending_mask = torch.zeros(arena.slots, dtype=torch.bool)
    pool._pending_gen = torch.zeros(arena.slots, dtype=torch.int64)
    pool._pending_fresh = torch.zeros(arena.slots, dtype=torch.bool)
    pool._own_extents = list(extents)
    pool._page_bytes = TOTAL
    pool._backend = types.SimpleNamespace(_get_suffixed_key=lambda h: h)
    pool.id_space = S + arena.slots * P + ap.PLACEHOLDERS
    return pool


def _rows(slots):
    return torch.cat([torch.arange(s * P, s * P + P) for s in slots]) + S


def _node(nid, keys, rows):
    return types.SimpleNamespace(id=nid, hash_value=list(keys), host_value=rows)


def _verify(pool, nodes):
    host = torch.cat([n.host_value for n in nodes])
    return ap.verify_load_chain(pool, nodes, host, rows_of=lambda n: n.host_value,
                                stems_of=lambda h: list(h))


def _provable(pool, nodes):
    return ap.provable_pages(pool, nodes, rows_of=lambda n: n.host_value, stems_of=lambda h: list(h))


def _publish(arena, stems):
    out = {}
    for stem in stems:
        (s, st, g), = arena.claim_slots([stem], [TOTAL])
        assert st == 0
        assert arena.complete_slots([s], [g], [(0, TOTAL)]) == [1]
        out[stem] = s
    return out


def _late_writer(tmp_path, slots=4):
    """Writer A (process A) claims key X host-write, merges its half and references its partial slot;
    writer B (process B, through its POOL) joins; the reader takes the claim over (H54); B's ack comes
    late -> 6. Returns (arena_a, arena_b, pool_b, slot, rows_b)."""
    arena_a = _arena(tmp_path, slots)
    arena_b = _arena(tmp_path, slots)
    (slot, st, gen), = arena_a.claim_slots(["X"], [TOTAL], role=ha.ROLE_HOST_WRITE)
    assert st == 0
    pool_b = _pool(arena_b, extents=((HALF, HALF),))
    assert pool_b._claim(["X"]) == [slot]                       # B joins (pending)
    assert arena_a.complete_slots([slot], [gen], [(0, HALF)]) == [0]
    assert arena_a.ref_slots([slot], +1) == 1                    # A's node reference
    assert arena_a.quarantine_host_write([slot], [gen]) == [1]   # H54 take-over
    rows_b = _rows([slot])
    assert pool_b.complete_write(rows_b) == 0                    # B's late ack: 6, no reference
    return arena_a, arena_b, pool_b, slot, rows_b, gen


# ------------------------------------------------------------------ (1) the fast path proves keys ------


def test_clean_chain_with_its_keys_takes_the_fast_path_unchanged(tmp_path, monkeypatch):
    arena = _arena(tmp_path, 8)
    slot_of = _publish(arena, ["a", "b", "c"])
    pool = _pool(arena)
    node = _node(1, ["a", "b", "c"], _rows([slot_of[k] for k in "abc"]))
    monkeypatch.setattr(ap, "_chain_plan", lambda *a, **k: pytest.fail("the full proof ran on a proven chain"))
    host = torch.cat([node.host_value])
    assert ap.verify_load_chain(pool, [node], host, rows_of=lambda n: n.host_value,
                                stems_of=lambda h: list(h)) is host
    assert _provable(pool, [node]) == (3, 3)


def test_clean_chain_on_a_recycled_slot_is_never_loaded(tmp_path):
    """The reviewer's probe on the pool's own ack path: B's rows would have loaded Y's KV."""
    arena_a, arena_b, pool_b, slot, rows_b, _g = _late_writer(tmp_path, slots=1)
    arena_a.ref_slots([slot], -1)                                 # A's node goes
    arena_a.quarantine_sweep(0)
    assert arena_a.claim_census([slot])[0]["state"] == 0
    (sy, st, gy), = arena_a.claim_slots(["Y"], [TOTAL])           # recycled for ANOTHER key
    assert sy == slot and st == 0 and arena_a.complete_slots([sy], [gy], [(0, TOTAL)]) == [1]
    stale = _node(7, ["X"], _rows([slot]))                        # the pre-A2 node: rows left on the slot
    assert _provable(pool_b, [stale]) == (0, 1), "a page whose key names no COMPLETE slot is not proven"
    with pytest.raises(ap.ArenaChainMismatch, match="key slot -1"):
        _verify(pool_b, [stale])


def test_fast_path_mismatch_repoints_to_the_keys_complete_slot(tmp_path):
    """The take-over's reader filled X from L3 into a new slot: the rows move there (+1), and the old
    QUARANTINE slot's reference (this page's, same generation) is given back -- it frees."""
    arena = _arena(tmp_path, 4)
    (old, st, g), = arena.claim_slots(["X"], [TOTAL], role=ha.ROLE_HOST_WRITE)
    assert arena.complete_slots([old], [g], [(0, HALF)]) == [0]
    assert arena.ref_slots([old], +1) == 1                        # the writer node's own reference
    assert arena.quarantine_host_write([old], [g]) == [1]
    new = _publish(arena, ["X"])["X"]                             # the reader's L3 fill
    assert new != old
    pool = _pool(arena)
    node = _node(3, ["X"], _rows([old]))
    out = _verify(pool, [node])
    assert out.tolist() == _rows([new]).tolist()
    assert arena.slot_refs([new]) == [1]
    assert arena.claim_census([old])[0]["state"] == 0, "the old slot's last reference went, it is free"


def test_repoint_never_gives_back_a_recycled_slots_reference(tmp_path):
    """Second arm on the re-point path: the old slot carries ANOTHER key now and THIS process holds a
    reference there for another node -- the re-point must not take it."""
    arena_a, arena_b, pool_b, slot, rows_b, _g = _late_writer(tmp_path, slots=4)
    dummies = _publish(arena_a, ["d1", "d2", "d3"])               # the arena is full but for X's old slot
    arena_a.ref_slots([slot], -1)                                 # A's node goes: the old slot frees
    assert arena_a.claim_census([slot])[0]["state"] == 0
    (sy, st, gy), = arena_a.claim_slots(["Y"], [TOTAL])
    assert sy == slot and st == 0                                 # recycled for ANOTHER key
    assert arena_a.complete_slots([sy], [gy], [(0, TOTAL)]) == [1]
    assert arena_b.ref_slots([sy], +1) == 1                       # B's OTHER node reads Y
    arena_a.free_slots([dummies["d1"]])
    zx = _publish(arena_a, ["X"])["X"]                            # X refilled elsewhere
    stale = _node(7, ["X"], _rows([slot]))
    out = _verify(pool_b, [stale])
    assert out.tolist() == _rows([zx]).tolist()
    assert arena_b.slot_refs([sy]) == [1], "Y's reader keeps its reference"


# ------------------------------------------------------------------ (2) settled at the ack -------------


def test_late_ack_orphans_the_rows_when_the_key_has_no_slot(tmp_path):
    arena_a, arena_b, pool_b, slot, rows_b, _g = _late_writer(tmp_path, slots=1)
    res = pool_b.settle_lost_rows(rows_b, ["X"])
    assert res["lost"] == 1 and res["orphaned"] == 1 and res["repointed"] == 0
    assert int(rows_b.min()) >= S + pool_b.arena_tokens, "no row of the node is an arena row any more"
    # the slot is recycled; B's process reads the new key
    arena_a.ref_slots([slot], -1)
    arena_a.quarantine_sweep(0)
    (sy, _st, gy), = arena_a.claim_slots(["Y"], [TOTAL])
    assert sy == slot and arena_a.complete_slots([sy], [gy], [(0, TOTAL)]) == [1]
    assert arena_b.ref_slots([sy], +1) == 1
    node = _node(7, ["X"], rows_b)
    # the release of the node takes nothing (second arm closed at the source)
    assert ap.ArenaMHAHostPool.release_queued_rows(pool_b, rows_b) == 0
    assert pool_b.free(rows_b.clone()) == 0
    assert arena_b.slot_refs([sy]) == [1]
    # the load proof stops by name instead of loading Y's KV
    assert _provable(pool_b, [node]) == (0, 1)
    with pytest.raises(ap.ArenaChainMismatch):
        _verify(pool_b, [node])


def test_late_ack_repoints_to_the_refilled_key_and_references_it(tmp_path):
    arena_a, arena_b, pool_b, slot, rows_b, _g = _late_writer(tmp_path, slots=4)
    z = _publish(arena_a, ["X"])["X"]                             # the reader filled X from L3
    res = pool_b.settle_lost_rows(rows_b, ["X"])
    assert res["repointed"] == 1 and res["orphaned"] == 0
    assert rows_b.tolist() == _rows([z]).tolist()
    assert arena_b.slot_refs([z]) == [1] and int(arena_b._ledger.held[z]) == 1
    node = _node(7, ["X"], rows_b)
    assert _verify(pool_b, [node]) is not None and _provable(pool_b, [node]) == (1, 1)
    assert pool_b.free(rows_b.clone()) == P                       # the node's release balances it
    assert arena_b.slot_refs([z]) == [0]


def test_settle_leaves_kept_pages_alone(tmp_path):
    arena = _arena(tmp_path, 4)
    pool = _pool(arena)
    assert len(pool._claim(["k0", "k1"])) == 2
    rows = _rows(pool._pending_mask.nonzero()[:, 0].tolist())
    pool.complete_write(rows)
    assert pool.take_lost_slots() == []
    before = rows.clone()
    assert pool.settle_lost_rows(rows, ["k0", "k1"])["lost"] == 0
    assert torch.equal(rows, before)


def test_ack_path_settles_the_nodes_host_value(tmp_path):
    """``_pdflip_direct_complete`` (the write-through ack) runs the settle on the node's own host_value."""
    from flliper.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE, UnifiedRadixCache

    arena_a, arena_b, pool_b, slot, _rows_b, gen = _late_writer(tmp_path, slots=1)
    # rebuild the scene up to the ack: _late_writer already acked -- re-run the ack through the tree
    hv = _rows([slot])
    pool_b._pending_mask[slot] = True
    pool_b._pending_gen[slot] = gen
    pool_b._pending_fresh[slot] = False
    t = object.__new__(UnifiedRadixCache)
    t.cache_controller = types.SimpleNamespace(mem_pool_host=pool_b, mem_pool_host_draft=None)
    t._pdflip_mamba_pool = lambda: None
    cd = {BASE_COMPONENT_TYPE: types.SimpleNamespace(host_value=hv)}
    node = types.SimpleNamespace(id=9, hash_value=["X"], component_data=cd, l3_present=False)
    assert t._pdflip_direct_complete(node) is True
    assert int(hv.min()) >= S + pool_b.arena_tokens, "the ack left the node's rows on the lost slot"


def test_lost_mamba_anchor_is_tombstoned(tmp_path):
    from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType

    arena = _arena(tmp_path, 4)
    mp = _pool(arena)
    mp._arena_page_tokens = 1
    mp.arena_tokens = arena.slots
    (slot, st, gen), = arena.claim_slots(["m"], [TOTAL], role=ha.ROLE_HOST_WRITE)
    mp._pending_mask[slot] = True
    mp._pending_gen[slot] = gen
    other = _arena(tmp_path, 4)
    assert other.quarantine_host_write([slot], [gen]) == [1]
    # an open writer is still on it: the slot stays QUARANTINE
    mhv = torch.tensor([S + slot])
    mp.complete_write(mhv)
    removed = []
    lru = types.SimpleNamespace(in_list=lambda n: True, remove_node=lambda n: removed.append(n))
    t = object.__new__(UnifiedRadixCache)
    t.host_lru_lists = {ComponentType.MAMBA: lru}
    cd = types.SimpleNamespace(host_value=mhv)
    node = types.SimpleNamespace(id=4, component_data={ComponentType.MAMBA: cd})
    assert t._a2_mamba_lost(node, mp, mhv) is True
    assert cd.host_value is None and removed == [node]


# ------------------------------------------------------------------ (3) generation-exact ledger ---------


def test_ledger_never_releases_into_a_moved_generation(tmp_path):
    """This process held a reference; the slot was freed hard under it (the claim resets the C refcount)
    and recycled -- another process references the new content. The stale release must take nothing."""
    mine = _arena(tmp_path, 1)
    other = _arena(tmp_path, 1)
    s = _publish(mine, ["X"])["X"]
    assert mine.ref_slots([s], +1) == 1
    mine.free_slots([s])                                          # hard free under the reference
    (s2, st, g2), = other.claim_slots(["Y"], [TOTAL])
    assert s2 == s and other.complete_slots([s2], [g2], [(0, TOTAL)]) == [1]
    assert other.ref_slots([s2], +1) == 1
    assert mine.ref_slots([s], -1) == 0
    assert other.slot_refs([s]) == [1], "the other process's reference on the new content survives"
    assert mine._ledger.refused_gen == 1 and int(mine._ledger.held[s]) == 0


def test_ledger_same_generation_releases_as_before(tmp_path):
    arena = _arena(tmp_path, 2)
    s = _publish(arena, ["X"])["X"]
    assert arena.ref_slots([s], +1) == 1 and arena.ref_slots([s], +1) == 1
    assert arena.ref_slots([s, s, s], -1) == 2
    assert arena.slot_refs([s]) == [0] and arena._ledger.refused == 1 and arena._ledger.refused_gen == 0


# ------------------------------------------------------------------ B5 ---------------------------------


def _quarantined_two_writers(arena):
    (s, st, g), = arena.claim_slots(["Q"], [TOTAL], role=ha.ROLE_HOST_WRITE)
    (s1, st1, _g1), = arena.claim_slots(["Q"], [TOTAL], role=ha.ROLE_HOST_WRITE)
    assert s1 == s and st1 == 1
    assert arena.quarantine_host_write([s], [g]) == [1]
    assert arena.claim_census([s])[0]["state"] == 4
    return s, g


def test_release_claims_on_quarantine_resolves_the_writer(tmp_path):
    arena = _arena(tmp_path, 2)
    s, g = _quarantined_two_writers(arena)
    assert arena.release_claims([s], [g]) == [1]
    assert arena.claim_census([s])[0]["state"] == 4, "one writer still open"
    assert arena.release_claims([s], [g]) == [1]
    assert arena.claim_census([s])[0]["state"] == 0, "the last writer out frees the quarantined slot"


def test_unclaim_on_quarantine_frees_with_the_last_writer(tmp_path):
    arena = _arena(tmp_path, 2)
    s, g = _quarantined_two_writers(arena)
    arena.unclaim([s], [g])
    assert arena.claim_census([s])[0]["state"] == 4
    arena.unclaim([s], [g])
    assert arena.claim_census([s])[0]["state"] == 0


def test_last_reference_release_frees_a_quarantined_slot(tmp_path):
    """B5: refs gone, no writer open -> free at once (before: QUARANTINE in n_claimed until a join reap)."""
    arena = _arena(tmp_path, 2)
    (s, st, g), = arena.claim_slots(["Q"], [TOTAL], role=ha.ROLE_HOST_WRITE)
    assert arena.complete_slots([s], [g], [(0, HALF)]) == [0]   # the only writer merged: none open
    assert arena.ref_slots([s], +1) == 1
    assert arena.quarantine_host_write([s], [g]) == [1]
    assert arena.claim_census([s])[0]["state"] == 4, "referenced: kept"
    assert arena.ref_slots([s], -1) == 1
    assert arena.claim_census([s])[0]["state"] == 0
    assert arena.stats()["claimed"] == 0


def _draft_scene(tmp_path, refill):
    from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    arena_a, arena_b, pool_b, slot, rows_b, gen = _late_writer(tmp_path, slots=4)
    z = _publish(arena_a, ["X"])["X"] if refill else None
    darena = _arena(tmp_path, 4, name="draft.bin")
    dpool = _pool(darena)
    dpool.row_slot = {}
    hv = _rows([slot])
    ds, = dpool._claim(["X.draft"])
    dpool.row_slot[int(hv[0]) - S] = ds
    dpool.complete_write(hv)                                      # the draft page completes (+1)
    assert darena.slot_refs([ds]) == [1]
    # B's KV page lost again at the ack: re-arm the lost record the way complete_write leaves it
    pool_b._a2_lost = [slot]
    t = object.__new__(UnifiedRadixCache)
    node = types.SimpleNamespace(id=5, hash_value=["X"])
    t._a2_settle_lost(node, pool_b, dpool, hv)
    return hv, dpool, darena, ds, z


def test_draft_row_map_follows_a_repointed_kv_page(tmp_path):
    hv, dpool, darena, ds, z = _draft_scene(tmp_path, refill=True)
    assert hv.tolist() == _rows([z]).tolist()
    assert dpool.row_slot == {z * P: ds}, "the draft row follows its KV row"
    assert darena.slot_refs([ds]) == [1]


def test_draft_page_of_an_orphaned_kv_page_gives_its_reference_back(tmp_path):
    hv, dpool, darena, ds, _z = _draft_scene(tmp_path, refill=False)
    assert int(hv.min()) >= S + dpool.arena_tokens
    assert dpool.row_slot == {}
    assert darena.slot_refs([ds]) == [0]
