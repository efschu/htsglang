"""K15-TP (W65 22.10., 27B INT8 round 3, boot 1022_003738 on a4f4b8fec1, D TP3, not Form A).

The mamba arena slot is ONE canonical blob shared by the TP ranks (each rank owns its layers' byte ranges and holds a
reference, ``arena_mamba_pool``). The K15 eviction spill releases this rank's reference on a victim and retries the
save; the slot frees only at the LAST releaser. Metal (D log 00:45:58, node 307 depth 51712, victim 282):

  TP2 ``FLUSH-SPILL ... dropped=0`` (released, slot still referenced by another process)
  TP1 ``FLUSH-SPILL ... dropped=0`` + ``PDFLIP-ANCHOR-LOST at=evict n=23 node=307 depth=51712``
  TP0 ``FLUSH-SPILL ... dropped=1`` (its slot freed for the un-backed node) -> TP0 keeps the anchor

so the ranks' anchors split per node (pdflip-4-73: 51712 / 52416 / 53120) and the intake prefetch died on W65 at
00:47:28. The PP rule (K15 review A2/B2, same file) already skipped the spill where stages share the slot; the same
holds for a pure TP group off Form A. Form A keeps the slots on TP0 alone (workers hold none): the spill stays there.

RED on a4f4b8fec1: the three ranks of the simulated shared slot end divergent (lost, lost, saved).
GREEN: no rank spills, all three take the same refusal, every victim keeps its anchor on every rank.
"""
from __future__ import annotations

import logging
import os
import sys
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest  # noqa: E402

import test_pdflip_k15_evict_anchor_save_1016 as k15  # noqa: E402

from flliper.srt.mem_cache import form_a_host_shadow as _r12  # noqa: E402
from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache  # noqa: E402
from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType  # noqa: E402

NODE_307, DEPTH_307 = 307, 51712


class _SharedSlotArena:
    """One full arena; the spill victim's slot is referenced by every rank of the group."""

    def __init__(self, ranks=3):
        self.victim_refs = ranks
        self.free = 0
        self.victim_anchor_on = {r: True for r in range(ranks)}


def _rank_tree(arena, rank, tp=3):
    t = k15._tree([])
    t.tp_world_size, t.pp_size, t.pp_rank = tp, 1, 0

    def _wb(node, write_back=False):
        t.calls.append("backup")
        if arena.free > 0:                    # the claim takes the freed slot (others would join its stem)
            arena.free -= 1
            node._landed = True
            return 1
        t._k15_last_refusal = "mamba_claim"   # #1421 BACKUP-REFUSED why=mamba_claim
        return 0

    def _spill(node, mp):
        t.calls.append("spill")
        arena.victim_anchor_on[rank] = False  # this rank's host value of the victim goes (release)
        arena.victim_refs -= 1
        dropped = arena.victim_refs == 0      # drop_unreferenced: only the last releaser frees it
        if dropped:
            arena.free += 1
        return dropped

    t.write_backup = _wb
    t.write_backup_anchor_only = lambda node, kv_on_device=False: _wb(node)
    t._pdflip_flush_spill_room = _spill
    return t


def _evict_on_every_rank(tp=3, role=None, monkeypatch=None):
    monkeypatch.setattr(_r12, "role", lambda: role)
    arena = _SharedSlotArena(ranks=tp)
    saved, trees = {}, {}
    for r in range(tp):                       # the metal order: one rank after the other
        t, n = _rank_tree(arena, r, tp=tp), k15._node(NODE_307, DEPTH_307)
        trees[r] = t
        t.nodes.append(n)
        saved[r] = UnifiedRadixCache.pdflip_save_anchor_before_tombstone(t, n)
        saved[r] = bool(saved[r]) and n.component_data[ComponentType.MAMBA].host_value is not None
    return arena, saved, trees


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("FLLIPER_PDFLIP_EVICT_ANCHOR_SAVE", raising=False)
    monkeypatch.setattr(UnifiedRadixCache, "_k15_saved_n", 0)
    monkeypatch.setattr(UnifiedRadixCache, "_k15_tp_skip_n", 0, raising=False)


def test_tp3_shared_slot_ranks_end_alike(monkeypatch, caplog):
    with caplog.at_level(logging.INFO):
        arena, saved, trees = _evict_on_every_rank(tp=3, role=None, monkeypatch=monkeypatch)
    # RED shape on a4f4b8fec1: {0: False, 1: False, 2: True} -- the last releaser alone keeps the anchor
    assert len(set(saved.values())) == 1, f"the ranks' anchor of node 307 diverged: {saved}"
    assert saved == {0: False, 1: False, 2: False}
    # no rank released the victim: its anchor stays on every rank
    assert arena.victim_anchor_on == {0: True, 1: True, 2: True}
    assert all("spill" not in t.calls for t in trees.values())
    assert any("K15 EVICT-SPILL SKIPPED tp_size=3" in m for m in caplog.messages)


def test_form_a_host_keeps_the_spill(monkeypatch):
    # Form A: TP0 holds the slots alone (one reference) -> its spill frees the slot, the save works
    monkeypatch.setattr(_r12, "role", lambda: "host")
    arena = _SharedSlotArena(ranks=1)
    t, n = _rank_tree(arena, 0, tp=3), k15._node(NODE_307, DEPTH_307)
    t.nodes.append(n)
    assert UnifiedRadixCache.pdflip_save_anchor_before_tombstone(t, n) is True
    assert t.calls == ["backup", "spill", "backup", "join"]


def test_single_rank_keeps_the_spill(monkeypatch):
    arena, saved, trees = _evict_on_every_rank(tp=1, role=None, monkeypatch=monkeypatch)
    assert saved == {0: True}
    assert trees[0].calls == ["backup", "spill", "backup", "join"]


def test_predicate_is_group_uniform_inputs_only(monkeypatch):
    t = types.SimpleNamespace(tp_world_size=3)
    monkeypatch.setattr(_r12, "role", lambda: None)
    assert UnifiedRadixCache._k15_tp_slot_shared(t) is True
    monkeypatch.setattr(_r12, "role", lambda: "worker")
    assert UnifiedRadixCache._k15_tp_slot_shared(t) is False
    monkeypatch.setattr(_r12, "role", lambda: None)
    assert UnifiedRadixCache._k15_tp_slot_shared(types.SimpleNamespace(tp_world_size=1)) is False
    assert UnifiedRadixCache._k15_tp_slot_shared(types.SimpleNamespace()) is False


# --- SPILL-TP: the sleep flush and the park spill (round 3b, 1022_010424, 01:30:09) --------------------------------
# TP0 "FLUSH-SPILL node=1336 ... victim_node=1214 dropped=1", TP1/TP2 "dropped=0" -> node 1342 parent_backuped True on
# TP0 only -> W65 REGROUP pdflip-10-212 (TP0 140480, TP1/TP2 140288) at 01:30:47. The real spill functions are called
# on a stub tree that has NO root_node: any spill work past the guard raises AttributeError (red on a4f4b8fec1).

class _NoTree:
    def __init__(self, tp):
        self.tp_world_size = tp
        self.calls = []


@pytest.mark.parametrize("fn,site", [("_pdflip_flush_spill_room", "flush_spill"),
                                     ("_pdflip_park_spill_room", "park_first")])
def test_shared_tp_group_never_spills_rank_locally(monkeypatch, caplog, fn, site):
    monkeypatch.setattr(_r12, "role", lambda: None)
    monkeypatch.setattr(UnifiedRadixCache, "_spill_tp_skip_n", 0, raising=False)
    t = _NoTree(3)
    t._shared_slot_spill_skipped = types.MethodType(UnifiedRadixCache._shared_slot_spill_skipped, t)
    node = types.SimpleNamespace(id=1336, pdflip_park_end=False)
    mp = types.SimpleNamespace(arena=object())
    with caplog.at_level(logging.INFO):
        assert getattr(UnifiedRadixCache, fn)(t, node, mp) is False
    assert any(f"MAMBA-ARENA SPILL SKIPPED site={site} tp_size=3 node=1336" in m for m in caplog.messages)


@pytest.mark.parametrize("fn", ["_pdflip_flush_spill_room", "_pdflip_park_spill_room"])
@pytest.mark.parametrize("tp,role", [(1, None), (3, "host")])
def test_single_rank_and_form_a_still_reach_the_spill(monkeypatch, fn, tp, role):
    monkeypatch.setattr(_r12, "role", lambda: role)
    t = _NoTree(tp)
    t._shared_slot_spill_skipped = types.MethodType(UnifiedRadixCache._shared_slot_spill_skipped, t)
    node = types.SimpleNamespace(id=1336, pdflip_park_end=False)
    mp = types.SimpleNamespace(arena=object())
    with pytest.raises(AttributeError):     # past the guard: the spill walks the (absent) tree
        getattr(UnifiedRadixCache, fn)(t, node, mp)
