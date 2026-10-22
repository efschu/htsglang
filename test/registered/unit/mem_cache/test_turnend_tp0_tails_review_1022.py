"""Review TURNEND-1022 (de700355e9 on rc34 170e3c84b9): the turn end's mamba spill through the REAL victim choice.

The author's module stubs ``tree_anchors`` (one list for both walks) and ``pick_foreign_victim``; the read guards of
``_pdflip_anchor_of`` (load-back host lock, write-through pending, rows claimed in this sweep), the park chain, the
S22 worker-delete filter and the untagged D tree never run there. Here the real ``_mad.tree_anchors`` walks a real
tree shape with the real ``_pdflip_anchor_of``, ``_pdflip_flush_spill_room``, ``_pdflip_last_resort_victim``,
``_r12_anchor_deletable`` and ``_pdflip_release_anchor``; only the pool, the L3 record and R12's transport are stubs.

* R1-R5 danger direction "token loss / wrong KV": an anchor a reader is loading, an anchor still being written, an
  anchor claimed in this very sweep, a park chain anchor and an L3 copy that vanished after the choice never leave.
* R6 danger direction "ranks disagree": S22 -- a victim whose release would DELETE a node a worker cannot delete is
  not taken; B2 -- no spill on a TP group without the Form A host (no R12 verdict, rank-local choice).
* R7 (review B1): group D's anchors are untagged (rc33 1022_060647 D: 1624/1624 K15 DISPLACE ``victim_rid=``), so
  the tagged census read 0 and one PENDING shallowest anchor blocked the spill while a secured one existed.
CPU only.
"""
from __future__ import annotations

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.mem_cache import l3_secured as L3S  # noqa: E402
from flliper.srt.mem_cache import unified_radix_cache as urc  # noqa: E402

U = urc.UnifiedRadixCache
M = urc.ComponentType.MAMBA
B = urc.BASE_COMPONENT_TYPE
L3_ASYNC_REAL = True   # this module sets the L3 record itself


class _Arena:
    slot_bytes = 58834944

    def __init__(self, refs):
        self.refs = refs

    def slot_refs(self, slots):
        return [self.refs.get(int(s), 0) for s in slots]


class _Pool:
    """Full mamba arena: no FREE slot, stage (i) frees nothing; ``drop_unreferenced`` frees a slot only our
    reference held; host ids == slots."""

    def __init__(self, refs, l3_index):
        self.arena = _Arena(refs)
        self._backend = types.SimpleNamespace(arena_secure_to_disk=lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("a turn end never writes to L3")))
        self.free, self.l3_index, self.dropped, self.claimed = 0, set(l3_index), [], []

    def claim_needs_room(self, hashes):
        return len(hashes) > self.free

    def make_secured_room(self, need):
        return 0

    def alloc_write(self, hashes):
        assert len(hashes) <= self.free
        self.free -= len(hashes)
        self.claimed.append(list(hashes))
        return torch.arange(len(hashes))

    def drop_unreferenced(self, slots):
        n = 0
        for s in slots:
            if self.arena.refs.get(int(s), 0) <= 1:
                self.arena.refs[int(s)] = 0
                self.dropped.append(int(s))
                n += 1
        self.free += n
        return n

    def is_arena_id(self, i):
        return True

    def settled_anchor_slots(self, hv):
        return [int(x) for x in hv.tolist()]

    def anchor_held(self, last_hash):
        return last_hash in self.l3_index


def _node(nid, parent, tokens, slot, kv_host=True, device=True):
    cds = [None] * (max(int(M), int(B)) + 1)
    cds[int(B)] = types.SimpleNamespace(value=torch.arange(4) if device else None,
                                        host_value=torch.arange(4) if kv_host else None)
    cds[int(M)] = types.SimpleNamespace(value=torch.arange(1) if device else None,
                                        host_value=None if slot is None else torch.tensor([slot]), host_lock_ref=0)
    n = types.SimpleNamespace(id=nid, parent=parent, key=[0] * tokens, children={}, component_data=cds,
                              pdflip_anchor_rid=None, _pdflip_end_anchor=False, write_through_pending_id=None,
                              evicted=not device, hash_value=["h%d" % nid], pdflip_anchor_secured=False, pdflip_lane=0)
    parent.children[nid] = n
    return n


@pytest.fixture
def d_tp0(monkeypatch):
    records, nodelete, l3 = [], set(), {"secured": set(), "check": None}
    monkeypatch.setattr(urc._r12, "role", lambda: "host")
    monkeypatch.setattr(urc._r12, "record_state", lambda tree, node, why="": records.append((node.id, why)))
    monkeypatch.setattr(urc._r12, "worker_nodelete_keys", lambda tree: set(nodelete))
    monkeypatch.setattr(urc._r12, "node_key", lambda n: n.id)
    monkeypatch.setattr(urc._r12, "kept_note", lambda key, n=1: None)
    monkeypatch.setattr(urc.envs.FLLIPER_PDFLIP_MAMBA_SPILL_LAST_RESORT, "get", lambda: True)
    monkeypatch.setattr(urc._mad, "ancestor_path", lambda target, root: ([], 0))

    def _sec(arena, slots, backend=None, evicting_ok=False):
        return np.array([int(s) in l3["secured"] for s in slots], dtype=bool)

    def _chk(arena, slots, backend=None, writer=""):
        ok = l3["secured"] if l3["check"] is None else l3["check"]
        return np.array([int(s) in ok for s in slots], dtype=bool)
    monkeypatch.setattr(L3S, "secured", _sec)
    monkeypatch.setattr(L3S, "check", _chk)

    def build(secured, refs=None, budget=4, tp=3):
        root = types.SimpleNamespace(id=0, children={}, key=[], parent=None)
        l3["secured"] = set(secured)
        released = []
        t = types.SimpleNamespace(root_node=root, _pdflip_direct_mamba_rows={}, _s108_turn_end_active=True,
                                  _s108_room_tried=set(), _s108_room_need={"mamba": budget},
                                  _s108_spill_left={"mamba": budget}, _pdflip_flush_spill=False,
                                  components={M: object()}, tp_world_size=tp, park=set(),
                                  _pdflip_anchor_ledger=types.SimpleNamespace(of=lambda rid: types.SimpleNamespace(
                                      displaced_share=0, displaced_full=0, dropped=0)))
        mp = _Pool(refs or {}, set())
        t._pdflip_mamba_pool = lambda: mp
        t._pdflip_claim_lane = lambda node, park=False: None
        t._pdflip_park_chain = lambda: set(t.park)

        def _evict(node, comp, target=None, tracker=None):
            released.append(node.id)
            node.component_data[int(M)].host_value = None
        t._evict_component_and_detach_lru = _evict
        t._update_evictable_leaf_sets = lambda node: None
        for name in ("_pdflip_anchor_of", "_pdflip_last_resort_victim", "_r12_anchor_deletable",
                     "_pdflip_release_anchor", "_s108_mamba_free_claim"):
            setattr(t, name, types.MethodType(getattr(U, name), t))
        return t, mp, released, root

    return build, records, nodelete, l3


def _claim(t, mp, nid=900):
    return t._s108_mamba_free_claim(mp, "tail%d" % nid, node=types.SimpleNamespace(id=nid))


def _two(build, **kw):
    """shallow (slot 3, 2688 tokens) and deep (slot 7, 2688+80000) anchors, both L3-secured, D-only references."""
    t, mp, released, root = build(secured={3, 7}, refs={3: 1, 7: 1}, **kw)
    a = _node(50, root, 2688, 3)
    b = _node(60, a, 80000, 7)
    mp.l3_index |= {"h50", "h60"}
    return t, mp, released, a, b


# ---------------------------------------------------------------------------------- token loss / wrong KV
@pytest.mark.parametrize("guard", ["host_lock", "write_pending", "claimed_this_sweep", "park_chain"])
def test_r1_r4_an_anchor_in_use_never_leaves_the_next_secured_one_does(d_tp0, guard):
    """The shallowest secured anchor is being loaded (host lock), written (write-through pending), was claimed in
    this very sweep (direct rows), or sits on a park chain: it keeps its slot; the next secured anchor goes."""
    build, records, _nd, _l3 = d_tp0
    t, mp, released, a, b = _two(build)
    if guard == "host_lock":
        a.component_data[int(M)].host_lock_ref = 1
    elif guard == "write_pending":
        a.write_through_pending_id = 77
    elif guard == "claimed_this_sweep":
        t._pdflip_direct_mamba_rows[a.id] = torch.tensor([3])
    else:
        t.park = {id(a)}
    assert _claim(t, mp) is not None
    assert released == [60] and mp.dropped == [7] and records == [(60, "h19_turn_end_spill")]
    assert a.component_data[int(M)].host_value is not None


@pytest.mark.parametrize("guard", ["host_lock", "write_pending", "claimed_this_sweep", "park_chain"])
def test_r1_r4_with_only_anchors_in_use_nothing_leaves(d_tp0, guard):
    build, records, _nd, _l3 = d_tp0
    t, mp, released, a, b = _two(build)
    for n in (a, b):
        if guard == "host_lock":
            n.component_data[int(M)].host_lock_ref = 1
        elif guard == "write_pending":
            n.write_through_pending_id = 77
        elif guard == "claimed_this_sweep":
            t._pdflip_direct_mamba_rows[n.id] = torch.tensor([0])
        else:
            t.park.add(id(n))
    assert _claim(t, mp) is None
    assert released == [] and mp.dropped == [] and records == [] and mp.claimed == []


def test_r5_an_l3_copy_gone_after_the_choice_releases_nothing(d_tp0):
    """The record listed both copies when the victim was chosen; the L3 evictor took them before the release
    (``l3_secured.check`` fails): nothing leaves L2, the tail is deferred by name."""
    build, records, _nd, l3 = d_tp0
    t, mp, released, a, b = _two(build)
    l3["check"] = set()
    assert _claim(t, mp) is None
    assert released == [] and mp.dropped == [] and records == []


# ---------------------------------------------------------------------------------- ranks disagree
def test_r6_s22_a_release_a_worker_cannot_follow_is_not_chosen(d_tp0):
    """S22: releasing the anchor of a device-evicted, childless node without KV host DELETES it on TP0; a worker
    holding it device-resident cannot follow. Such an anchor is no victim; the next secured one goes."""
    build, records, nodelete, _l3 = d_tp0
    t, mp, released, root = build(secured={3, 7}, refs={3: 1, 7: 1})
    a = _node(50, root, 2688, 3, kv_host=False, device=False)     # evicted, childless, no KV host
    b = _node(60, root, 80000, 7)
    nodelete.add(a.id)
    assert _claim(t, mp) is not None
    assert released == [60] and records == [(60, "h19_turn_end_spill")]
    assert a.component_data[int(M)].host_value is not None


@pytest.mark.parametrize("role,tp,spills", [("host", 3, True), (None, 3, False), ("worker", 3, False),
                                            (None, 1, True)])
def test_r6_b2_the_spill_runs_only_where_tp0_is_the_form_a_host(d_tp0, monkeypatch, role, tp, spills):
    """Review B2 (SOLL R11/R12): the verdict is TP0's and R12 carries it -- a TP group without the Form A plan
    would pick its victims rank-locally. A single rank (TP1) has no peer to disagree with."""
    build, records, _nd, _l3 = d_tp0
    monkeypatch.setattr(urc._r12, "role", lambda: role)
    t, mp, released, a, b = _two(build, tp=tp)
    got = _claim(t, mp)
    assert (got is not None) is spills
    assert bool(released) is spills


# ---------------------------------------------------------------------------------- review B1: untagged D anchors
def test_r7_b1_a_pending_shallowest_anchor_does_not_block_a_secured_one(d_tp0):
    """D's anchors are untagged: the tagged walk is empty, the last resort chooses. Its shallowest candidate's L3
    copy is still queued (PENDING); a deeper one is secured. RED before review B1: the last resort took the
    shallowest, ``l3_secured.check`` refused it, nothing was freed (rc33: 8x FLUSH-SPILL PENDING)."""
    build, records, _nd, _l3 = d_tp0
    t, mp, released, root = build(secured={7}, refs={3: 1, 7: 1})
    a = _node(50, root, 2688, 3)
    b = _node(60, root, 80000, 7)
    mp.l3_index |= {"h60"}
    info: dict = {}
    t._s108_spill_left = {"mamba": 1}
    assert U._pdflip_flush_spill_room(t, types.SimpleNamespace(id=900), mp, why="turn_end_spill", info=info)
    assert released == [60] and a.component_data[int(M)].host_value is not None
    assert (info["tree"], info["secured"], info["pending"], info["pick"]) == (2, 1, 1, "last_resort"), info


def test_r7_b1_the_flush_and_k15_keep_their_old_last_resort(d_tp0):
    """Without ``info`` (K15 eviction save, sleep flush) the last resort is the old call: the shallowest, checked,
    PENDING -> nothing freed (byte for byte the pre-review behaviour)."""
    build, records, _nd, _l3 = d_tp0
    t, mp, released, root = build(secured={7}, refs={3: 1, 7: 1})
    _node(50, root, 2688, 3)
    _node(60, root, 80000, 7)
    assert not U._pdflip_flush_spill_room(t, types.SimpleNamespace(id=900), mp)
    assert released == []
