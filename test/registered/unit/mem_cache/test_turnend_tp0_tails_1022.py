"""TURNEND-1022 (NF rc33 fbb680d6d3, boot ..._1022_060647 D): TP0's turn-end tails stay un-backed under load.

THE METAL (D log, 06:12-07:24, all from TP0/TP1/TP2 lines):

* KV: 0 ``S108 TURN-END-ROOM kind=kv`` lines on ANY rank, 753/712/712 ``TURN-END-DEFER`` (arena_room) on
  TP0/TP1/TP2. The hybrid's KV host pool is a ``HostPoolGroup``; its ``__getattr__`` forwarded ``claim_needs_room``
  but not ``make_secured_room``, so ``_s108_make_room`` found no room maker and returned False before its count.
  KVLEITER F1 never ran for KV, on no rank.
* Mamba (TP0 only, Form A): the 84-slot anchor arena was 84 COMPLETE / 84 referenced from 06:17 to the flip at 06:35,
  77 references by TP0's own tree (``ARENA-REF-HOLDERS pool=MAMBA tree=77 handoff_kept=0/0 park_kept=0/0``). Stage (i)
  takes only unreferenced slots: ``TURN-END-ROOM kind=mamba freed=0`` from n=12 on, 307 of 979 by n=256. In the same
  minutes the K15 eviction save spilled 1219 tree anchors through ``_pdflip_flush_spill_room``, every one
  ``on_disk=1 written=0`` -- the L3-secured room existed; the turn end was not allowed to take it.

THE RULE. The turn end takes that room the same way (an L3-secured, ref-free tree anchor leaves L2; TP0 decides, R12
makes the workers follow), at most once per anchor this turn end demands. Nothing without an L3 copy leaves, nothing
is written, nothing waits. CPU only.
"""
from __future__ import annotations

import logging
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.mem_cache import l3_secured as L3S  # noqa: E402
from flliper.srt.mem_cache import unified_radix_cache as urc  # noqa: E402
from flliper.srt.pdflip.mamba_arena_displace import OwnedAnchor  # noqa: E402

U = urc.UnifiedRadixCache
M = urc.ComponentType.MAMBA
B = urc.BASE_COMPONENT_TYPE
PAGE = 64
L3_ASYNC_REAL = True   # this module sets the L3 record itself (secured / pending per slot)


# ============================================================================ KV: the HostPoolGroup forwards the room
class _KvArena:
    """The KV arena pool behind the group: free slots, a stage-(i) room maker, a claim."""

    layout = "layer_first"
    page_size = PAGE
    device = "cpu"
    size = 1

    def __init__(self, free=0, room=0):
        self.free, self.room, self.made, self.claimed = free, room, [], []

    def claim_needs_room(self, hashes):
        return len(hashes) > self.free

    def make_secured_room(self, need):
        self.made.append(int(need))
        got = min(int(need), self.room)
        self.room -= got
        self.free += got
        return got

    def alloc_write(self, hashes):
        assert len(hashes) <= self.free, "claimed beyond the free room"
        self.free -= len(hashes)
        self.claimed.append(list(hashes))
        return torch.arange(len(hashes))


def _group(kv):
    from flliper.srt.mem_cache.memory_pool_host import HostPoolGroup, PoolEntry

    mamba = types.SimpleNamespace(layout="layer_first", page_size=PAGE, device="cpu", size=1)
    return HostPoolGroup([PoolEntry(name="kv", host_pool=kv, device_pool=None, layer_mapping={},
                                    is_primary_index_anchor=True),
                          PoolEntry(name="mamba", host_pool=mamba, device_pool=None, layer_mapping={},
                                    is_primary_index_anchor=False)])


def _kv_node(nid, pages):
    return types.SimpleNamespace(id=nid, hash_value=["h%d_%d" % (nid, i) for i in range(pages)],
                                 component_data={B: types.SimpleNamespace(value=torch.arange(pages * PAGE))})


def test_the_host_pool_group_forwards_make_secured_room():
    """RED on fbb680d6d3: getattr(group, 'make_secured_room', None) is None -- the turn end found no room maker."""
    kv = _KvArena(free=0, room=7)
    g = _group(kv)
    mk = getattr(g, "make_secured_room", None)
    assert callable(mk)
    assert mk(5) == 5 and kv.made == [5]


def test_rc33_the_turn_end_makes_kv_room_through_the_group_on_every_rank():
    """rc33 06:21:06 TP1 ``TURN-END-DEFER why=arena_room:1`` (and 753/712/712 like it): the claim through the
    group -- the pool every NF rank has -- makes its stage-(i) room once and claims. RED on fbb680d6d3."""
    kv = _KvArena(free=0, room=100)
    why = []
    stub = types.SimpleNamespace(_s108_turn_end_active=True, page_size=PAGE, cache_controller=None,
                                 _pdflip_direct_pool=lambda: _group(kv), _1421_refused=lambda w, n: why.append(w),
                                 _s108_room_tried=set(), _s108_room_need={"kv": 5, "mamba": 0})
    rows = U._pdflip_direct_claim(stub, _kv_node(1, 3))
    assert rows is not False and rows is not None, why
    assert kv.made == [5] and why == [] and kv.claimed == [["h1_0", "h1_1", "h1_2"]]


def test_a_pool_without_a_room_maker_is_named_once(caplog):
    """The silent return that hid the KV defect for a whole boot names itself (once per process and kind)."""
    U.__dict__.get("_s108_room_unavailable_kinds", set()).discard("kv")
    pool = types.SimpleNamespace()
    stub = types.SimpleNamespace(_s108_room_tried=set())
    with caplog.at_level(logging.WARNING):
        assert U._s108_make_room(stub, pool, 3, "kv") is False
        stub._s108_room_tried = set()
        assert U._s108_make_room(stub, pool, 3, "kv") is False
    assert sum("TURN-END-ROOM UNAVAILABLE kind=kv" in m for m in caplog.messages) == 1


# ============================================================================ mamba: the turn end's anchor spill
class _Arena:
    slot_bytes = 58834944

    def __init__(self, refs):
        self.refs = refs

    def slot_refs(self, slots):
        return [self.refs.get(int(s), 0) for s in slots]


class _MambaPool:
    """84 slots on the metal; here: ``free`` FREE slots, stage (i) frees nothing (every slot referenced),
    ``drop_unreferenced`` frees a slot only our reference held, ``anchor_held`` answers from the L3 index."""

    def __init__(self, refs, l3_index):
        self.arena = _Arena(refs)
        self._backend = types.SimpleNamespace(arena_secure_to_disk=lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("a turn end never writes to L3")))
        self.free, self.l3_index, self.made, self.claimed, self.dropped = 0, set(l3_index), [], [], []

    def claim_needs_room(self, hashes):
        return len(hashes) > self.free

    def make_secured_room(self, need):
        self.made.append(int(need))
        return 0   # 84/84 referenced: stage (i) has nothing (rc33 TURN-END-ROOM freed=0 from n=12)

    def alloc_write(self, hashes):
        assert len(hashes) <= self.free, "claimed beyond the free room"
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

    def anchor_held(self, last_hash):
        return last_hash in self.l3_index


def _anchor_node(nid, device_anchor=True):
    cds = {B: types.SimpleNamespace(value=torch.arange(4), host_value=torch.arange(4)),
           M: types.SimpleNamespace(value=torch.arange(1) if device_anchor else None, host_value=torch.arange(1),
                                    host_lock_ref=0)}
    return types.SimpleNamespace(id=nid, evicted=False, children={}, component_data=cds, hash_value=["a%d" % nid],
                                 pdflip_anchor_secured=False, pdflip_anchor_rid=None)


@pytest.fixture
def tp0(monkeypatch):
    """D TP0 of a Form A group at a turn end: the real ``_s108_mamba_free_claim`` -> ``_s108_make_room`` ->
    ``_s108_anchor_spill`` -> ``_pdflip_flush_spill_room`` -> ``_pdflip_last_resort_victim`` ->
    ``_pdflip_release_anchor`` (R12 record) -> ``_s25_mark_released_anchor_secured``; the tree walk and the
    victim order are the metal's (untagged D anchors: the last resort takes the shallowest)."""
    records, secured_slots = [], set()
    monkeypatch.setattr(urc._r12, "role", lambda: "host")
    monkeypatch.setattr(urc._r12, "record_state", lambda tree, node, why="": records.append((node.id, why)))
    monkeypatch.setattr(urc._r12, "worker_nodelete_keys", lambda tree: set())
    monkeypatch.setattr(urc.envs.FLLIPER_PDFLIP_MAMBA_SPILL_LAST_RESORT, "get", lambda: True)
    monkeypatch.setattr(urc._mad, "lane_filter", lambda t, lane: list(t))
    monkeypatch.setattr(urc._mad, "pick_foreign_victim", lambda t, rid: None)
    monkeypatch.setattr(urc._mad, "ancestor_path", lambda target, root: ([], 0))
    monkeypatch.setattr(L3S, "secured", lambda arena, slots, backend=None, evicting_ok=False: __import__(
        "numpy").array([int(s) in secured_slots for s in slots], dtype=bool))
    monkeypatch.setattr(L3S, "check", lambda arena, slots, backend=None, writer="": __import__(
        "numpy").array([int(s) in secured_slots for s in slots], dtype=bool))
    state = {"anchors": []}
    monkeypatch.setattr(urc._mad, "tree_anchors", lambda **k: [
        OwnedAnchor(node=n, depth=d, rid="", slots=(s,)) for n, d, s in state["anchors"]
        if n.component_data[M].host_value is not None])

    def build(anchors, refs, l3, spill_left=2, l3_index=None):
        """anchors: (node, depth, slot); refs: slot -> readers; l3: slots with their L3 copy (record + index)."""
        state["anchors"] = anchors
        secured_slots.clear()
        secured_slots.update(l3)
        mp = _MambaPool(refs, l3_index if l3_index is not None else {n.hash_value[-1] for n, _d, s in anchors
                                                                       if s in l3})
        evicted = []
        t = types.SimpleNamespace(root_node=object(), _s108_turn_end_active=True, _s108_room_tried=set(),
                                  _s108_room_need={"mamba": spill_left}, _s108_spill_left={"mamba": spill_left},
                                  _pdflip_flush_spill=False, components={M: object()},
                                  _pdflip_anchor_ledger=types.SimpleNamespace(of=lambda rid: types.SimpleNamespace(
                                      displaced_share=0, displaced_full=0, dropped=0)))
        t._pdflip_mamba_pool = lambda: mp
        t._pdflip_anchor_of = lambda n, mp_, allow_end=False: (True, None)   # unused: tree_anchors is the metal's list
        t._pdflip_claim_lane = lambda node, park=False: None
        t._pdflip_park_chain = lambda: set()

        def _evict(node, comp, target=None, tracker=None):
            evicted.append(node.id)
            node.component_data[M].host_value = None
        t._evict_component_and_detach_lru = _evict
        t._update_evictable_leaf_sets = lambda node: None
        for name in ("_pdflip_last_resort_victim", "_r12_anchor_deletable", "_pdflip_release_anchor",
                     "_s108_mamba_free_claim"):
            setattr(t, name, types.MethodType(getattr(U, name), t))
        return t, mp, evicted

    return build, records


def _claimer(nid=500):
    return types.SimpleNamespace(id=nid)


def test_rc33_a_full_mamba_arena_gives_the_turn_end_an_l3_secured_slot(tp0):
    """rc33 06:20:02 TP0 ``S108 TURN-END-L2 rid=pdflip-2-87 ... anchor_deferred=84``: 84/84 referenced, stage (i)
    freed 0. Now the shallowest L3-secured, ref-free tree anchor leaves L2, TP0 records it for R12 (the workers drop
    it too), the victim stays backed (its anchor is in L3), and the tail anchor gets the slot. RED on fbb680d6d3."""
    build, records = tp0
    v1, v2 = _anchor_node(50), _anchor_node(60)
    t, mp, evicted = build([(v1, 80640, 3), (v2, 84480, 7)], refs={3: 1, 7: 1}, l3={3, 7})
    rows = t._s108_mamba_free_claim(mp, "tail", node=_claimer())
    assert rows is not None and mp.claimed == [["tail"]]
    assert mp.made == [2], "stage (i) first, once"
    assert evicted == [50] and mp.dropped == [3]
    assert records == [(50, "h19_turn_end_spill")], "TP0's release is recorded for the workers (R12)"
    assert v1.pdflip_anchor_secured is True, "the victim's anchor is in L3: it stays backed"
    assert t._s108_spill_left == {"mamba": 1}


def test_guard_token_loss_an_anchor_without_its_l3_copy_never_leaves(tp0):
    """Danger direction 'token loss': a tree anchor whose L3 copy the write-behind has not made yet (pending) is
    never released for a turn end -- the tail is deferred by name instead."""
    build, records = tp0
    v1, v2 = _anchor_node(50), _anchor_node(60)
    t, mp, evicted = build([(v1, 80640, 3), (v2, 84480, 7)], refs={3: 1, 7: 1}, l3=set())
    assert t._s108_mamba_free_claim(mp, "tail", node=_claimer()) is None
    assert evicted == [] and records == [] and mp.dropped == [] and mp.claimed == []
    assert v1.component_data[M].host_value is not None and v2.component_data[M].host_value is not None


def test_guard_token_loss_the_secured_mark_needs_the_bytes(tp0):
    """The victim is marked ``pdflip_anchor_secured`` only when ``anchor_held`` finds its bytes; an index that
    does not list it leaves the mark off (the node then reads un-backed, honestly)."""
    build, records = tp0
    v1 = _anchor_node(50)
    t, mp, evicted = build([(v1, 80640, 3)], refs={3: 1}, l3={3}, l3_index=set())
    assert t._s108_mamba_free_claim(mp, "tail", node=_claimer()) is not None
    assert evicted == [50] and v1.pdflip_anchor_secured is False


def test_a_slot_another_process_references_is_not_taken(tp0):
    """K15c: P's stages hold their own references (rc33 refs - own_held = 21); releasing ours there frees no slot.
    The D-only anchor is taken."""
    build, records = tp0
    shared, own = _anchor_node(50), _anchor_node(60)
    t, mp, evicted = build([(shared, 2688, 3), (own, 84480, 7)], refs={3: 4, 7: 1}, l3={3, 7})
    assert t._s108_mamba_free_claim(mp, "tail", node=_claimer()) is not None
    assert evicted == [60] and records == [(60, "h19_turn_end_spill")]


def test_the_spill_is_bounded_by_the_turn_ends_own_demand(tp0):
    """No reserve: at most one spill per anchor this turn end demands; a further claim is deferred by name."""
    build, records = tp0
    a = [(_anchor_node(50 + i), 1000 * (i + 1), i) for i in range(4)]
    t, mp, evicted = build(a, refs={i: 1 for i in range(4)}, l3=set(range(4)), spill_left=2)
    got = [t._s108_mamba_free_claim(mp, "tail%d" % i, node=_claimer(500 + i)) for i in range(3)]
    assert [g is not None for g in got] == [True, True, False]
    assert evicted == [50, 51] and len(records) == 2 and mp.made == [2]


def test_no_spill_outside_a_turn_end_sweep(tp0):
    """Fail closed: without the turn end's budget (``_s108_spill_left``) the claim stays FREE-room only."""
    build, records = tp0
    t, mp, evicted = build([(_anchor_node(50), 80640, 3)], refs={3: 1}, l3={3})
    t._s108_spill_left = None
    assert t._s108_mamba_free_claim(mp, "tail", node=_claimer()) is None
    assert evicted == [] and records == []


def test_guard_ranks_diverge_no_stage_local_spill_on_p(tp0):
    """Danger direction 'ranks disagree': on P (PP > 1) the mamba slot is shared by the stages -- a stage-local
    release frees no slot and leaves the stages' anchors different (K15 review A2/B2). The turn end never spills
    there."""
    build, records = tp0
    t, mp, evicted = build([(_anchor_node(50), 80640, 3)], refs={3: 1}, l3={3})
    t.pp_size = 3
    assert t._s108_mamba_free_claim(mp, "tail", node=_claimer()) is None
    assert evicted == [] and records == []


def test_the_claim_paths_pass_the_node_to_the_spill():
    """Both turn-end anchor claims (direct claim with KV, anchor-only backup) hand the claimer to the spill: the
    spill must never pick the claimer itself, and needs it for the lane rule."""
    import inspect

    assert "_s108_mamba_free_claim(mp, last_hash, node=node)" in inspect.getsource(U._pdflip_mamba_claim)
    assert "_s108_mamba_free_claim(mp, hashes[-1], node=node)" in inspect.getsource(U._pdflip_direct_claim)


# ============================================================================ the turn end: demand and deferred anchors
def _tree_node(nid, parent, backed=True, anchor_dev=True, anchor_host=False):
    cds = [None] * (max(int(M), int(B)) + 1)
    cds[int(B)] = types.SimpleNamespace(value=torch.arange(4), host_value=torch.arange(4) if backed else None)
    cds[int(M)] = types.SimpleNamespace(value=torch.arange(1) if anchor_dev else None,
                                        host_value=torch.arange(1) if anchor_host else None)
    n = types.SimpleNamespace(id=nid, parent=parent, children={}, component_data=cds, backuped=backed,
                              l3_present=False, evicted=False, hash_value=["k%d" % nid], key=[0] * 64,
                              write_through_pending_id=None)
    if parent is not None:
        parent.children[nid] = n
    return n


def _ao(n):
    cd = n.component_data[int(M)]
    return (n.backuped or n.l3_present) and cd.value is not None and cd.host_value is None


def test_the_demand_counts_the_anchor_of_a_kv_backed_node():
    """RED on fbb680d6d3: a node whose KV is backed and whose anchor is on the device only (the anchor-only backup
    the sweep issues for it) counted 0 -- the turn end had no anchor demand to spill for."""
    root = types.SimpleNamespace(children={})
    a = _tree_node(1, root, backed=True)
    b = _tree_node(2, a, backed=False)
    stub = types.SimpleNamespace(tree_components=(B, M), _pdflip_anchor_only_candidate=_ao)
    assert U._s108_room_demand(stub, [a, b]) == {"kv": 1, "mamba": 2}


def test_a_deferred_anchor_comes_back_at_the_next_turn_end_and_leaves_once_secured(monkeypatch):
    """``TURN-END-DEFER ... (kept unbacked for the next turn end ...)``: on fbb680d6d3 nobody read
    ``_s108_anchor_deferred`` again (rc33 anchor_deferred 2 -> 1096, never smaller). The next turn end now offers
    the deferred anchor to its sweep with the spill budget for it, and drops it from the list once secured.
    RED on fbb680d6d3."""
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    root = types.SimpleNamespace(children={}, parent=None)
    old = _tree_node(10, root, backed=True)           # an earlier turn end's tail: KV backed, anchor deferred
    chain = [_tree_node(20, root, backed=True, anchor_dev=False)]
    seen = {}

    def _sweep(max_issue, clock, first, chain_only):
        seen["first"] = [n.id for n in first]
        seen["left"] = dict(t._s108_spill_left)
        old.component_data[int(M)].host_value = torch.arange(1)   # the sweep secured it
        return {"unbacked": 1, "issued": 1, "refused": 0}

    cc = types.SimpleNamespace(_s108_submit=lambda *a: None)
    t = types.SimpleNamespace(enable_storage=True, cache_controller=cc, root_node=root,
                              tree_components=(B, M), _pdflip_anchor_only_candidate=_ao,
                              _s108_anchor_deferred={old.id: old}, _s108_deferred={})
    t._pdflip_direct_pool = lambda: object()
    t._pdflip_chain_nodes = lambda key: list(chain)
    t.publish_unbacked_sweep = _sweep
    U._s108_turn_end_l2(t, types.SimpleNamespace(rid="pdflip-2-88"), radix_key=object())
    assert seen.get("first") == [20, 10], seen
    assert seen["left"] == {"mamba": 1}
    assert t._s108_spill_left is None and t._s108_room_tried is None
    assert old.id not in t._s108_anchor_deferred


def test_an_anchor_deferred_while_its_kv_write_is_in_flight_stays_for_the_next_turn_end(monkeypatch):
    """The S108 path itself (rc33 06:20:02 ``deferred=4``): the KV goes out (write in flight), the anchor finds no
    slot and is deferred by name. That node is no anchor-only candidate at this moment (write pending) -- it must
    stay in the list, or the next turn end never offers it again."""
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    root = types.SimpleNamespace(children={}, parent=None)
    tail = _tree_node(30, root, backed=False)

    def _sweep(max_issue, clock, first, chain_only):
        tail.write_through_pending_id = 30                        # KV issued through the copier
        t._s108_anchor_deferred = {tail.id: tail}                 # the anchor found no slot
        return {"unbacked": 1, "issued": 1, "refused": 0}

    def _cand(n):
        return n.write_through_pending_id is None and _ao(n)

    t = types.SimpleNamespace(enable_storage=True, cache_controller=types.SimpleNamespace(_s108_submit=None),
                              root_node=root, tree_components=(B, M), _pdflip_anchor_only_candidate=_cand,
                              _s108_deferred={})
    t._pdflip_direct_pool = lambda: object()
    t._pdflip_chain_nodes = lambda key: [tail]
    t.publish_unbacked_sweep = _sweep
    U._s108_turn_end_l2(t, types.SimpleNamespace(rid="pdflip-2-87"), radix_key=object())
    assert set(t._s108_anchor_deferred) == {30}
