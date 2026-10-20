"""K15c (h16s 1f4b3e73d4, boot 1016_103428, D TP0 10:41:55): K15's spill released victims whose
mamba-arena slot group P still referenced (``MAMBA-ARENA DISPLACE ... dropped=0``; ``ARENA-REF-CENSUS
arena-58834944 refs=47 own_held=20``), the retry was refused again and the anchor was lost
(``PDFLIP-ANCHOR-LOST at=evict node=97 depth=123904`` / ``node=102 depth=118208``, why=mamba_claim).
Pinned: the spill (real ``UnifiedRadixCache._pdflip_flush_spill_room`` and ``_pdflip_last_resort_victim``)
takes only a victim whose every slot has exactly one reader reference -- its release frees the slot;
when none exists it releases nothing and names it (``K15C SPILL-NO-FREE-VICTIM``).
RED on 1f4b3e73d4: the shared shallow victim (depth 2688) is released."""
from __future__ import annotations

import logging
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.mem_cache import unified_radix_cache as U  # noqa: E402
from flliper.srt.pdflip.mamba_arena_displace import OwnedAnchor  # noqa: E402


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("FLLIPER_PDFLIP_SPILL_REF_FREE", raising=False)
    monkeypatch.setattr(U._r12, "role", lambda: "host")      # D TP0 (Form A host)


class _Arena:
    slot_bytes = 58834944

    def __init__(self, refs):
        self.refs = refs

    def slot_refs(self, slots):
        return [self.refs.get(int(s), 0) for s in slots]


def _tree(anchors, refs, monkeypatch):
    """anchors: list of (node_id, depth, slot); the metal: node 53 @2688 slot 3 shared with P (refs 4),
    node 58 @52992 slot 7 shared (refs 4), node 83 @123392 slot 11 D-only (refs 1, a K15-saved anchor)."""
    arena = _Arena(refs)
    secured, released = [], []
    be = types.SimpleNamespace(arena_secure_to_disk=lambda a, c, writer=None: (secured.append([x[0] for x in c]),
                                                                              {"on_disk": 1, "written": 0, "lost": 0})[1])
    mp = types.SimpleNamespace(arena=arena, _backend=be)
    owned = [OwnedAnchor(node=types.SimpleNamespace(id=i), depth=d, rid="", slots=(s,)) for i, d, s in anchors]
    monkeypatch.setattr(U._mad, "tree_anchors", lambda **k: list(owned))
    monkeypatch.setattr(U._mad, "lane_filter", lambda t, lane: list(t))
    # untagged D anchors: pick_foreign_victim finds none (as on the metal), the last resort picks the shallowest
    monkeypatch.setattr(U._mad, "pick_foreign_victim", lambda t, rid: None)
    monkeypatch.setattr(U._mad, "pick_last_resort_victim",
                        lambda c, chain_ids, claimer_lane: min(c, key=lambda a: a.depth) if c else None)
    monkeypatch.setattr(U.envs.FLLIPER_PDFLIP_MAMBA_SPILL_LAST_RESORT, "get", lambda: True)
    monkeypatch.setattr(U._mad, "ancestor_path", lambda target, root: ([], 0))
    t = types.SimpleNamespace(root_node=object(), _pdflip_anchor_ledger=types.SimpleNamespace(of=lambda rid: None))
    t._pdflip_anchor_of = lambda n, mp, allow_end=False: (True, None)
    t._pdflip_claim_lane = lambda node, park=False: None
    t._pdflip_park_chain = lambda: set()
    t.PDFLIP_PARK_SPILL_TRIES = 3

    def _release(victim, mp, st, why, for_rid):
        # K15c review A2: the release returns the slots it dropped (0 while another process references one)
        released.append(victim.node.id)
        return int(all(arena.refs.get(int(x), 0) <= 1 for x in victim.slots))
    t._pdflip_release_anchor = _release
    t._pdflip_last_resort_victim = types.MethodType(U.UnifiedRadixCache._pdflip_last_resort_victim, t)
    # S22 (R12 anchor victim choice): the real filter is bound like the victim helper; with no worker nd set it keeps all
    t._r12_anchor_deletable = types.MethodType(U.UnifiedRadixCache._r12_anchor_deletable, t)
    return t, mp, secured, released


METAL = [(53, 2688, 3), (58, 52992, 7), (83, 123392, 11)]


def test_pdflip_56_shared_shallow_victim_is_skipped_for_a_d_only_one(monkeypatch):
    t, mp, secured, released = _tree(METAL, {3: 4, 7: 4, 11: 1}, monkeypatch)
    claimer = types.SimpleNamespace(id=97)
    assert U.UnifiedRadixCache._pdflip_flush_spill_room(t, claimer, mp) is True
    assert released == [83] and secured == [[11]], "a victim whose slot P still references frees nothing"


def test_no_ref_free_victim_releases_nothing_and_names_it(monkeypatch, caplog):
    t, mp, secured, released = _tree(METAL, {3: 4, 7: 4, 11: 2}, monkeypatch)
    with caplog.at_level(logging.WARNING):
        assert U.UnifiedRadixCache._pdflip_flush_spill_room(t, types.SimpleNamespace(id=102), mp) is False
    assert released == [] and secured == []
    assert any("K15C SPILL-NO-FREE-VICTIM" in m and "shared=3" in m for m in caplog.messages)


def test_switch_off_is_the_old_choice(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_SPILL_REF_FREE", "0")
    t, mp, secured, released = _tree(METAL, {3: 4, 7: 4, 11: 1}, monkeypatch)
    # K15c review A2: the shared victim is released but frees no slot -- no success, named honestly
    assert U.UnifiedRadixCache._pdflip_flush_spill_room(t, types.SimpleNamespace(id=97), mp) is False
    assert released == [53]


def test_unreadable_refs_are_not_free():
    class _Bad:
        def slot_refs(self, slots):
            raise RuntimeError("header")
    f = U.UnifiedRadixCache._k15c_ref_free_filter(_Bad())
    assert f(OwnedAnchor(node=None, depth=1, rid="", slots=(1,))) is False


def test_off_form_a_host_the_old_choice_stands(monkeypatch):
    """P's stages each hold one reference on the shared slot: there 'refs <= 1' is not the group's share."""
    monkeypatch.setattr(U._r12, "role", lambda: None)
    t, mp, secured, released = _tree(METAL, {3: 4, 7: 4, 11: 1}, monkeypatch)
    U.UnifiedRadixCache._pdflip_flush_spill_room(t, types.SimpleNamespace(id=97), mp)
    assert released == [53]


def _park_tree(monkeypatch, refs):
    t, mp, secured, released = _tree(METAL, refs, monkeypatch)
    monkeypatch.setattr(U._mad, "pick_park_victim",
                        lambda c, chain_ids, claimer_is_end, claimer_lane: min(c, key=lambda a: a.depth) if c else None)
    node = types.SimpleNamespace(id=97, pdflip_park_end=True)
    node.__dict__["_pdflip_park_spill_tries"] = 0
    return t, mp, secured, released, node


def test_review_a1_park_spill_on_form_a_host_takes_the_d_only_victim(monkeypatch):
    """K15c review A1 (boot 1016_103428 10:42:16 TP0 ``DISPLACE why=park_first node=59 dropped=0``):
    the PARK-END-ANCHOR-FIRST spill took the shared shallow victim (53) -- now the D-only one (83).
    RED on d5ef021983 (released == [53])."""
    t, mp, secured, released, node = _park_tree(monkeypatch, {3: 4, 7: 4, 11: 1})
    assert U.UnifiedRadixCache._pdflip_park_spill_room(t, node, mp) is True
    assert released == [83] and secured == [[11]], released


def test_review_a1_park_spill_without_a_free_victim_names_it(monkeypatch, caplog):
    t, mp, secured, released, node = _park_tree(monkeypatch, {3: 4, 7: 4, 11: 2})
    with caplog.at_level(logging.WARNING):
        assert U.UnifiedRadixCache._pdflip_park_spill_room(t, node, mp) is False
    assert released == [] and secured == []
    assert any("K15C SPILL-NO-FREE-VICTIM" in m for m in caplog.messages)


def test_review_a3_foreign_branch_filtered(monkeypatch):
    """K15c review A3 (probe test_foreign_branch_filtered): pick_foreign_victim offers a candidate --
    the filter there must hold (kills the surviving mutant 'tree=[a for a in tree if _rf(a)]' removed)."""
    t, mp, secured, released = _tree(METAL, {3: 4, 7: 4, 11: 1}, monkeypatch)
    monkeypatch.setattr(U._mad, "pick_foreign_victim",
                        lambda tr, rid: min(tr, key=lambda a: a.depth) if tr else None)
    assert U.UnifiedRadixCache._pdflip_flush_spill_room(t, types.SimpleNamespace(id=97), mp) is True
    assert released == [83]


def test_review_a2_flush_spill_without_a_dropped_slot_is_no_success(monkeypatch, caplog):
    """K15c review A2 (boot 1016_103428 10:59:57 ``DISPLACE n=29 dropped=0`` then ``FLUSH-SPILL ... its
    slot freed``): a reader takes a reference between the filter and drop_unreferenced -- the release
    drops nothing; the line says so and the spill returns False."""
    t, mp, secured, released = _tree(METAL, {3: 4, 7: 4, 11: 1}, monkeypatch)
    t._pdflip_release_anchor = lambda victim, mp, st, why, for_rid: (released.append(victim.node.id), 0)[1]
    with caplog.at_level(logging.INFO):
        assert U.UnifiedRadixCache._pdflip_flush_spill_room(t, types.SimpleNamespace(id=97), mp) is False
    assert released == [83]
    assert any("slot still referenced" in m and "dropped=0" in m for m in caplog.messages)
    assert not any("its slot freed" in m for m in caplog.messages)


def test_review_a2_release_anchor_returns_dropped(monkeypatch):
    from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType

    monkeypatch.setattr(U._r12, "role", lambda: None)
    t = types.SimpleNamespace(components={ComponentType.MAMBA: object()},
                              _evict_component_and_detach_lru=lambda *a, **k: (0, 0),
                              _update_evictable_leaf_sets=lambda n: None)
    st = types.SimpleNamespace(displaced_share=0, displaced_full=0, dropped=0)
    victim = OwnedAnchor(node=types.SimpleNamespace(id=5), depth=10, rid="r", slots=(1, 2))
    for dropped in (0, 2):
        mp = types.SimpleNamespace(drop_unreferenced=lambda slots, d=dropped: d)
        assert U.UnifiedRadixCache._pdflip_release_anchor(t, victim, mp, st, why="flush_spill", for_rid="flush") == dropped
