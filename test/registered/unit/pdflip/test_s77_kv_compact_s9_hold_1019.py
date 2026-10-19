"""S77-A: a decided KV-stage shrink is not held by the tree's idle cached prefix.

THE METAL (NF rc17 7d93470b5e, boot 1019_171636, D.log from line 32274). D decoded
15 min without a flip; every end event read ``end event: live pages up to 405056
hold S9 (wanted S8)`` (17:28:28Z .. 17:33:52Z, up to 418176; rc16: 263 times)
while ``used`` (278k-383k) fit S8 = 393216. The pages above the boundary were the
finished turns' prefixes the tree RETAINS (``#1469 RETAIN`` / ``#59 RESUMABLE``,
86-98k tokens each), idle between turns and backed in L2 -- ``max_live_page``
counted them like a running request's. S9 instead of S8 cost the 3080 ranks
6-10 expert seat rows (misses per forward x2.2-2.4; bs4 +14 %, bs3 +10 % gpu-ms).

WHAT MUST HOLD.
(1) ``KvPageCopier`` moves a whole page in every born KV tensor with the geometry
    the stage form unmaps it by (K/V: a slot per token; QSA keys: page//ratio
    slots; a DCP owner share: (P//S)*ratio rows, the owner class kept).
(2) Idle tree nodes above the boundary move into free pages below it; the node's
    token ids are re-pointed (offsets kept), the sources go back to the allocator.
(3) Un-backed nodes take the room first; a BACKED node without room is evicted
    (host copy stays), never an un-backed one, never one with a device child that
    stays. A page above the boundary held outside the tree (a running request) or
    a pinned node stops the plan: nothing moves.
(4) MemSched: a floor that blocks a decided shrink is handed to the compactor; the
    shrink then goes as far as the new floor allows.
(5) The runtime tick: the group verdict (MIN) over the candidate stages, moves
    before evictions, a refusal not repeated until the next end event.
(6) The metal sequence: S9, the turn ends, its retained prefix sits above S8 ->
    RED on 7d93470b5e ("hold S9 (wanted S8)"), GREEN: S8, the prefix still on
    the device below the boundary, bytes intact.
"""
from __future__ import annotations

import logging
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator  # noqa: E402
from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType  # noqa: E402
try:  # the RED run on 7d93470b5e has no such module: only the metal test runs there
    from flliper.srt.pdflip import d_kv_compact as KC  # noqa: E402
except ImportError:  # pragma: no cover - base revision
    KC = None
from flliper.srt.pdflip import d_seat_vram as dsv  # noqa: E402
from flliper.srt.pdflip.d_mem_sched import MemSched  # noqa: E402

FULL, MAMBA = int(ComponentType.FULL), int(ComponentType.MAMBA)
_REAL_MAX_LIVE_PAGE = dsv.max_live_page   # the tick fixture patches it; the metal test reads pages


# ---- a tree the size of a desk --------------------------------------------------

class CD:
    def __init__(self, value=None, host=None, lock=0):
        self.value, self.host_value, self.lock_ref = value, host, lock


class Node:
    _n = 0

    def __init__(self, parent=None, last=0):
        Node._n += 1
        self.id = Node._n
        self.parent = parent
        self.children = {}
        self.component_data = [CD(), CD(), CD()]
        self.last_access_time = last
        self.write_through_pending_id = None
        if parent is not None:
            parent.children[self.id] = self


class Tree:
    """The unified tree as d_kv_compact reads it."""

    def __init__(self, alloc):
        self.alloc = alloc
        self.root_node = Node()
        self.components = {ComponentType.FULL: object(), ComponentType.MAMBA: object()}
        self.tree_components = (ComponentType.FULL, ComponentType.MAMBA)
        self.ongoing_write_through = {}
        self.ongoing_load_back = {}
        self.evicted = []

    def _collect_all_nodes(self):
        out, st = [], [self.root_node]
        while st:
            n = st.pop()
            out.append(n)
            st.extend(n.children.values())
        return out

    def _is_device_leaf(self, node):
        if node is self.root_node or node.component_data[FULL].value is None:
            return False
        if any(cd.lock_ref > 0 for cd in node.component_data):
            return False
        return not any(c.component_data[FULL].value is not None for c in node.children.values())

    def _evict_to_host(self, node, tracker):
        assert node.component_data[FULL].host_value is not None
        v = node.component_data[FULL].value
        self.alloc.free(v)
        tracker[ComponentType.FULL] += int(v.numel())
        node.component_data[FULL].value = None
        self.evicted.append(node.id)


def _alloc(num_pages, P):
    a = PagedTokenToKVPoolAllocator(num_pages * P, page_size=P, dtype=torch.int64, device="cpu",
                                    kvcache=types.SimpleNamespace(), need_sort=False)
    return a


def _take(alloc, pages):
    """Take exactly these pages out of the free list (the allocator hands them out)."""
    want = torch.tensor(sorted(set(int(p) for p in pages)), dtype=torch.int64)
    if not want.numel():
        return
    hit = torch.isin(alloc.free_pages, want)
    assert int(hit.sum()) == want.numel(), "pages not free: %s" % pages
    alloc.free_pages = alloc.free_pages[~hit]


def _tokens(pages, P):
    return torch.tensor([p * P + o for p in pages for o in range(P)], dtype=torch.int64)


def _node(tree, alloc, pages, P, *, parent=None, backed=True, lock=0, last=0, mamba=None):
    _take(alloc, pages)
    n = Node(parent or tree.root_node, last=last)
    n.component_data[FULL] = CD(_tokens(pages, P), host=(object() if backed else None), lock=lock)
    if mamba is not None:
        n.component_data[MAMBA] = CD(torch.tensor([1]), host=mamba)
    return n


class _Copier:
    """A KvPageCopier over one K buffer [ (pages+1)*P rows, 4 bytes ] whose rows hold
    their own token id -- a moved page is visible byte for byte."""

    def __init__(self, num_pages, P):
        self.page_size = P
        self.t = torch.arange((num_pages + 1) * P, dtype=torch.int32)
        g = dsv.KvTensorGeom(dsv.SlotTensorGeom("k0", 1, (num_pages + 1) * P, 4,
                                                (num_pages + 1) * P * 4), 1, P)
        self.real = dsv.KvPageCopier([(self.t.data_ptr(), g)], {self.t.data_ptr(): self.t}, P)
        self.copies = []

    def copy(self, src, dst):
        self.copies.append((list(src), list(dst)))
        return self.real.copy(src, dst)

    def ptrs(self):
        return self.real.ptrs()


# ---- (1) the copier ---------------------------------------------------------------

def _geom(name, layers, slots, slot_bytes, ratio=1, pad=0, owner=(0, 0)):
    return dsv.KvTensorGeom(dsv.SlotTensorGeom(name, layers, slots, slot_bytes, layers * slots * slot_bytes),
                            ratio, pad, owner)


def test_copier_moves_a_page_in_kv_qsa_and_an_owner_share_alike():
    P = 8
    kv = torch.arange(2 * 40 * 2, dtype=torch.int16).reshape(2, 40, 2)          # [L, slot, 4 B]
    qsa = torch.arange(3 * 10, dtype=torch.int32).reshape(3, 10)                 # ratio 4: 2 slots/page
    own = torch.arange(20 * 3, dtype=torch.uint8).reshape(20, 3)                 # S=2, ratio 1: 4 rows/page
    born = [(kv.data_ptr(), _geom("kv", 2, 40, 4)),
            (qsa.data_ptr(), _geom("qsa", 3, 10, 4, ratio=4)),
            (own.data_ptr(), _geom("own", 1, 20, 3, owner=(2, 1)))]
    tensors = {kv.data_ptr(): kv, qsa.data_ptr(): qsa, own.data_ptr(): own}
    c = dsv.KvPageCopier(born, tensors, P)
    kv0, qsa0, own0 = kv.clone(), qsa.clone(), own.clone()
    c.copy([3], [1])
    assert torch.equal(kv[:, 8:16], kv0[:, 24:32]) and torch.equal(kv[:, 24:32], kv0[:, 24:32])
    assert torch.equal(kv[:, :8], kv0[:, :8]) and torch.equal(kv[:, 16:24], kv0[:, 16:24])
    assert torch.equal(qsa[:, 2:4], qsa0[:, 6:8]) and torch.equal(qsa[:, 0:2], qsa0[:, 0:2])
    assert torch.equal(own[4:8], own0[12:16]) and torch.equal(own[0:4], own0[0:4])


def test_copier_refuses_a_page_the_geometry_cannot_cut():
    t = torch.zeros(10, 4, dtype=torch.uint8)
    with pytest.raises(dsv.KvCopyRefused):
        dsv.KvPageCopier([(t.data_ptr(), _geom("q", 1, 10, 4, ratio=3))], {t.data_ptr(): t}, 8)
    with pytest.raises(dsv.KvCopyRefused):
        dsv.KvPageCopier([(t.data_ptr(), _geom("q", 1, 10, 4))], {}, 8)  # no tensor reference


# ---- (2)/(3) survey, plan, execute ---------------------------------------------------

P, NP, B = 4, 32, 16   # 32 pages of 4 tokens; the boundary at page 16


def _world():
    alloc = _alloc(NP, P)
    return alloc, Tree(alloc), _Copier(NP, P)


def test_idle_nodes_above_the_boundary_move_down_and_keep_their_bytes():
    alloc, tree, cop = _world()
    _take(alloc, list(range(1, 9)))                        # running requests' pages below
    hot = _node(tree, alloc, [20, 21, 22], P, last=9)       # a retained turn above
    cold = _node(tree, alloc, [9, 30], P, backed=False, last=1)   # un-backed, straddles
    before = {n.id: cop.t[n.component_data[FULL].value].clone() for n in (hot, cold)}
    sv = KC.survey(tree, alloc, P, B)
    plan, why = KC.plan_for(sv, B)
    assert plan is not None and not plan.evicts, why
    res = KC.execute(tree, alloc, plan, cop)
    assert res.moved_nodes == 2 and res.moved_pages == 4 and res.evicted_nodes == 0
    for n in (hot, cold):
        v = n.component_data[FULL].value
        assert int((v // P).max()) <= B
        assert torch.equal(cop.t[v], before[n.id]), "a moved page lost its bytes"
        assert torch.equal(v % P, torch.arange(v.numel()) % P)
    sv2 = KC.survey(tree, alloc, P, B)
    assert sv2.live_above == [] and 20 in sv2.free and 30 in sv2.free
    # the un-backed node took the room first (page 10), the hot one the next three
    assert cop.copies[0][1] == [10, 11, 12, 13]


def test_a_running_requests_page_above_the_boundary_stops_the_plan():
    alloc, tree, cop = _world()
    _node(tree, alloc, [20], P)
    _take(alloc, [25])                                     # no tree node holds it
    plan, why = KC.plan_for(KC.survey(tree, alloc, P, B), B)
    assert plan is None and "outside the tree" in why


@pytest.mark.parametrize("pin", ["lock", "write_through", "load_back", "shared"])
def test_a_pinned_node_stops_the_plan(pin):
    alloc, tree, cop = _world()
    n = _node(tree, alloc, [20], P, lock=1 if pin == "lock" else 0)
    if pin == "write_through":
        n.write_through_pending_id = n.id
    if pin == "load_back":
        tree.ongoing_load_back[n.id] = object()
    if pin == "shared":
        m = Node(tree.root_node)
        m.component_data[FULL] = CD(_tokens([20], P), host=object())
    plan, why = KC.plan_for(KC.survey(tree, alloc, P, B), B)
    assert plan is None and "pinned" in why


def test_no_room_below_a_backed_leaf_is_evicted_an_unbacked_one_never():
    alloc, tree, cop = _world()
    _take(alloc, list(range(1, 17)))                       # nothing free below the boundary
    leaf = _node(tree, alloc, [20], P, backed=True)
    plan, why = KC.plan_for(KC.survey(tree, alloc, P, B), B)
    assert plan is not None and [m.node for m in plan.evicts] == [leaf]
    KC.execute(tree, alloc, plan, cop)
    assert tree.evicted == [leaf.id] and leaf.component_data[FULL].value is None
    _node(tree, alloc, [21], P, backed=False)
    plan, why = KC.plan_for(KC.survey(tree, alloc, P, B), B)
    assert plan is None and "un-backed" in why


def test_an_anchor_without_its_host_copy_is_not_backed():
    alloc, tree, cop = _world()
    _take(alloc, list(range(1, 17)))
    _node(tree, alloc, [20], P, backed=True, mamba=None)
    n = tree.root_node.children[max(tree.root_node.children)]
    n.component_data[MAMBA] = CD(torch.tensor([3]), host=None)   # the GDN anchor only on device
    plan, why = KC.plan_for(KC.survey(tree, alloc, P, B), B)
    assert plan is None and "un-backed" in why


def test_a_backed_node_with_a_device_child_that_stays_is_not_evicted():
    alloc, tree, cop = _world()
    _take(alloc, list(range(1, 16)))                       # one free page below: 16
    parent = _node(tree, alloc, [20, 21], P, backed=True, last=1)
    child = Node(parent)                                   # holds page 1 (below): stays on device
    child.component_data[FULL] = CD(_tokens([1], P), host=object())
    plan, why = KC.plan_for(KC.survey(tree, alloc, P, B), B)
    assert plan is None and "device child" in why


def test_mru_first_gets_the_room_the_cold_backed_one_is_evicted():
    alloc, tree, cop = _world()
    _take(alloc, list(range(1, 16)))                       # one free page: 16
    hot = _node(tree, alloc, [20], P, last=9)
    cold = _node(tree, alloc, [21], P, last=1)
    plan, _ = KC.plan_for(KC.survey(tree, alloc, P, B), B)
    assert [m.node for m, _s, _d in plan.moves] == [hot] and [m.node for m in plan.evicts] == [cold]


def test_a_failed_copy_moves_no_node_and_frees_the_destinations():
    alloc, tree, cop = _world()
    n = _node(tree, alloc, [20], P)
    v0 = n.component_data[FULL].value
    plan, _ = KC.plan_for(KC.survey(tree, alloc, P, B), B)
    cop.copy = lambda s, d: (_ for _ in ()).throw(RuntimeError("boom"))
    free0 = len(KC.survey(tree, alloc, P, B).free)
    with pytest.raises(KC.CompactRefused):
        KC.execute(tree, alloc, plan, cop)
    assert n.component_data[FULL].value is v0
    assert len(KC.survey(tree, alloc, P, B).free) == free0


# ---- (4) MemSched ------------------------------------------------------------------

LADDER = (32768, 65536, 98304, 131072, 163840, 196608, 229376, 262144, 393216, 524288)


def _at_s9():
    ms = MemSched(stage_tokens=LADDER, air_tokens=0)
    ms.step(450_000)
    assert ms.stage == 9
    return ms


def test_memsched_hands_a_blocking_floor_to_the_compactor():
    ms = _at_s9()
    calls = []
    ms.compactor = lambda want, end, cur: calls.append((want, end, cur)) or 300_000
    st = ms.step(335_068, ended=True, floor_tokens=405_056)
    assert calls == [(8, True, 405_056)]
    assert st.changed and ms.stage == 8 and ms.pending is None
    assert ms.counters["stage_down_compacted"] == 1


def test_memsched_without_a_lower_floor_holds_as_before():
    ms = _at_s9()
    ms.compactor = lambda want, end, cur: None
    st = ms.step(335_068, ended=True, floor_tokens=405_056)
    assert not st.changed and ms.stage == 9 and ms.pending == 8
    assert "hold S9 (wanted S8)" in st.reason


def test_memsched_does_not_call_the_compactor_on_a_free_shrink():
    ms = _at_s9()
    ms.compactor = lambda *a: pytest.fail("called without a blocking floor")
    st = ms.step(335_068, ended=True, floor_tokens=300_000)
    assert st.changed and ms.stage == 8


# ---- (5)/(6) the runtime tick -------------------------------------------------------

from test_pdflip_d_mem_sched_0929 import FLOOR_LADDER, _req, floor_env, tick_env  # noqa: E402,F401

PG = 64
S8_PAGE = FLOOR_LADDER[8] // PG     # 6144


class _Born:
    """One born K buffer of the whole top stage (1 byte a token: its token id % 251)."""

    def __init__(self):
        rows = FLOOR_LADDER[-1] + PG
        self.t = (torch.arange(rows, dtype=torch.int64) % 251).to(torch.uint8)
        self.geom = dsv.KvTensorGeom(dsv.SlotTensorGeom("k0", 1, rows, 1, rows), 1, PG)


def _metal_env(floor_env, monkeypatch, *, real_floor=True):
    dsv_, sched, caps, floor = floor_env
    alloc = _alloc(FLOOR_LADDER[-1] // PG, PG)
    born = _Born()
    monkeypatch.setattr(dsv_, "_KV_BORN", [(born.t.data_ptr(), born.geom)])
    monkeypatch.setattr(dsv_, "_KV_BORN_T", {born.t.data_ptr(): born.t}, raising=False)
    alloc._kvcache = types.SimpleNamespace(k_buffer=[born.t], v_buffer=[])
    sched.token_to_kv_pool_allocator = alloc
    tree = Tree(alloc)
    sched.tree_cache = tree
    if real_floor:
        monkeypatch.setattr(dsv_, "max_live_page", _REAL_MAX_LIVE_PAGE)
    return dsv_, sched, caps, alloc, tree, born


def _to_s9(dsv_, sched):
    sched.running_batch.reqs = [_req("pdflip-2-66", 98_221, 61), _req("pdflip-2-68", 86_604, 51),
                                _req("pdflip-2-70", 150_000), _req("pdflip-2-71", 120_000)]
    dsv_.runtime_tick(sched)
    ms = getattr(sched, dsv_.MEM_SCHED_ATTR)
    assert ms.stage == 9
    return ms


def test_rc17_the_retained_turn_above_s8_no_longer_holds_s9(floor_env, monkeypatch, caplog):
    """RED on 7d93470b5e: ``end event: live pages up to 405056 hold S9 (wanted S8)``.
    GREEN: the retained turn's pages above 393216 move below it, S8, its bytes the same."""
    caplog.set_level(logging.INFO)
    dsv_, sched, caps, alloc, tree, born = _metal_env(floor_env, monkeypatch)
    ms = _to_s9(dsv_, sched)
    _take(alloc, list(range(1, 5000)))                               # the running decodes, below
    turn = _node(tree, alloc, list(range(5800, 6000)) + list(range(6145, 6330)), PG, last=7)
    before = born.t[turn.component_data[FULL].value].clone()
    # pdflip-2-66 / 2-68 finish (their prefix is the retained turn): the end event
    sched.running_batch.reqs = [_req("pdflip-2-70", 150_000), _req("pdflip-2-71", 120_000)]
    st = dsv_.runtime_tick(sched)
    assert st.changed and ms.stage == 8, st.reason
    assert caps[-1] == FLOOR_LADDER[8]
    v = turn.component_data[FULL].value
    assert int((v // PG).max()) <= S8_PAGE, "a page of the turn stayed above S8"
    assert torch.equal(born.t[v], before), "the moved prefix lost its bytes"
    assert tree.evicted == [], "a turn that fits below was evicted"
    assert any(dsv_.KV_COMPACT_MARK in m and "verdict=moved" in m for m in caplog.messages)


def test_a_peer_that_cannot_clear_keeps_every_rank_on_s9(floor_env, monkeypatch):
    dsv_, sched, caps, alloc, tree, born = _metal_env(floor_env, monkeypatch)
    ms = _to_s9(dsv_, sched)
    _take(alloc, list(range(1, 5000)))
    turn = _node(tree, alloc, list(range(6145, 6200)), PG)
    v0 = turn.component_data[FULL].value
    surveys = []
    real = dsv_._kv_compact_rank_plans

    def counted(sched_, boundaries):
        surveys.append(list(boundaries))
        plans, why, copier = real(sched_, boundaries)
        return plans, why, copier
    monkeypatch.setattr(dsv_, "_kv_compact_rank_plans", counted)

    def gmin(vals):
        # the peer clears no stage: every KV-COMPACT flag is 0 (2 per candidate)
        return [0] * len(vals) if len(vals) == 2 * len(surveys[-1] if surveys else ()) else list(vals)
    sched._pdflip_group_min_ints = gmin
    sched.running_batch.reqs = [_req("pdflip-2-70", 150_000)]
    st = dsv_.runtime_tick(sched)
    assert not st.changed and ms.stage == 9 and ms.pending is not None
    assert turn.component_data[FULL].value is v0 and len(surveys) == 1
    # the refusal is not surveyed again on a pending re-read, only at the next end event
    sched.running_batch.reqs = [_req("pdflip-2-70", 150_000, 5)]
    dsv_.runtime_tick(sched)
    assert len(surveys) == 1
    sched.running_batch.reqs = []                                     # an end event: look again
    dsv_.runtime_tick(sched)
    assert len(surveys) == 2


def test_a_running_page_above_the_boundary_waits_for_its_end(floor_env, monkeypatch):
    dsv_, sched, caps, alloc, tree, born = _metal_env(floor_env, monkeypatch)
    ms = _to_s9(dsv_, sched)
    _take(alloc, list(range(1, 5000)) + [6200])                       # 6200: a running request's
    _node(tree, alloc, list(range(6145, 6150)), PG)
    sched.running_batch.reqs = [_req("pdflip-2-70", 150_000)]
    st = dsv_.runtime_tick(sched)
    assert not st.changed and ms.stage == 9 and "hold S9" in st.reason
