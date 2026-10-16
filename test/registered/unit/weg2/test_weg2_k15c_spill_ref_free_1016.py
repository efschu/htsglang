"""K15c (h16s 1f4b3e73d4, boot 1016_103428, D TP0 10:41:55): K15's spill released victims whose
mamba-arena slot group P still referenced (``MAMBA-ARENA DISPLACE ... dropped=0``; ``ARENA-REF-CENSUS
arena-58834944 refs=47 own_held=20``), the retry was refused again and the anchor was lost
(``WEG2-ANCHOR-LOST at=evict node=97 depth=123904`` / ``node=102 depth=118208``, why=mamba_claim).
Pinned: the spill (real ``UnifiedRadixCache._weg2_flush_spill_room`` and ``_weg2_last_resort_victim``)
takes only a victim whose every slot has exactly one reader reference -- its release frees the slot;
when none exists it releases nothing and names it (``K15C SPILL-NO-FREE-VICTIM``).
RED on 1f4b3e73d4: the shared shallow victim (depth 2688) is released."""
from __future__ import annotations

import logging
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.mem_cache import unified_radix_cache as U  # noqa: E402
from sglang.srt.weg2.mamba_arena_displace import OwnedAnchor  # noqa: E402


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("SGLANG_WEG2_SPILL_REF_FREE", raising=False)
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
    monkeypatch.setattr(U.envs.SGLANG_WEG2_MAMBA_SPILL_LAST_RESORT, "get", lambda: True)
    monkeypatch.setattr(U._mad, "ancestor_path", lambda target, root: ([], 0))
    t = types.SimpleNamespace(root_node=object(), _weg2_anchor_ledger=types.SimpleNamespace(of=lambda rid: None))
    t._weg2_anchor_of = lambda n, mp, allow_end=False: (True, None)
    t._weg2_claim_lane = lambda node, park=False: None
    t._weg2_park_chain = lambda: set()
    t._weg2_release_anchor = lambda victim, mp, st, why, for_rid: released.append(victim.node.id)
    t._weg2_last_resort_victim = types.MethodType(U.UnifiedRadixCache._weg2_last_resort_victim, t)
    return t, mp, secured, released


METAL = [(53, 2688, 3), (58, 52992, 7), (83, 123392, 11)]


def test_weg2_56_shared_shallow_victim_is_skipped_for_a_d_only_one(monkeypatch):
    t, mp, secured, released = _tree(METAL, {3: 4, 7: 4, 11: 1}, monkeypatch)
    claimer = types.SimpleNamespace(id=97)
    assert U.UnifiedRadixCache._weg2_flush_spill_room(t, claimer, mp) is True
    assert released == [83] and secured == [[11]], "a victim whose slot P still references frees nothing"


def test_no_ref_free_victim_releases_nothing_and_names_it(monkeypatch, caplog):
    t, mp, secured, released = _tree(METAL, {3: 4, 7: 4, 11: 2}, monkeypatch)
    with caplog.at_level(logging.WARNING):
        assert U.UnifiedRadixCache._weg2_flush_spill_room(t, types.SimpleNamespace(id=102), mp) is False
    assert released == [] and secured == []
    assert any("K15C SPILL-NO-FREE-VICTIM" in m and "shared=3" in m for m in caplog.messages)


def test_switch_off_is_the_old_choice(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_SPILL_REF_FREE", "0")
    t, mp, secured, released = _tree(METAL, {3: 4, 7: 4, 11: 1}, monkeypatch)
    assert U.UnifiedRadixCache._weg2_flush_spill_room(t, types.SimpleNamespace(id=97), mp) is True
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
    assert U.UnifiedRadixCache._weg2_flush_spill_room(t, types.SimpleNamespace(id=97), mp) is True
    assert released == [53]
