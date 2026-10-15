# SPDX-License-Identifier: Apache-2.0
"""READ-ROOM-D 1015: a P read on the shared arena gets the room group D's tree holds -- TP0 decides for group D,
the workers follow its R12 verdicts, anchors stay, and a parked span it gave back is read again in full from L3.

MEASURED (boot ..._226aad7105_1015_124245, weg2-70-293, 13:35:34-39): ``WEG2-READ-DEPTH landed=47 of=1832
cause=arena_full`` on all three P stages; the KV arena (arena-786432.bin) ``pinned=6485 of 6485 refs=21495``.
P held ``own_held=2258`` per stage (running requests, 3 x 2258 = 6774); group D held the rest:
``ARENA-REF-HOLDERS pool=FULL tree=3215 tree_in_use=1692 own_held=4907`` on D TP0/TP1/TP2 (13:35:42) -- D's live
tree and the spans of the five requests D had just parked (``WEG2-D-PARK park_running ... span retained``,
``#248 PARK-DEMOTE ... l3_pages=996 absent=0``: on the L3 disk). D answered no room request: READ-ROOM was armed
on the local-PP floor only, and ``_w3_arena_spill`` returns 0 under R12.

Driven through the REAL C arena and the REAL ``HiCacheFile.arena_fill_from_disk`` (P's read and D's resume read),
REAL ``UnifiedRadixCache`` trees for D TP0 and its worker (unit fixture; their host rows name arena slots), the
REAL R12 wire (``form_a_host_shadow.attach`` / ``consume`` -> ``_reconcile``), and ``read_room.serve``.
RED on f5278f4ff2 (D's serve reaches ``_w3_arena_spill``, which stops under R12: nothing is released, the read
ends short), GREEN with READ-ROOM-D.
"""

import os
import shutil
import threading
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch
from unittest import mock

from sglang.srt.mem_cache import form_a_host_shadow as R12
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import EvictLayer, UnifiedRadixCache
from sglang.srt.weg2 import pp_slot_fidelity as SF

from test_unified_radix_cache_unittest import CacheConfig, build_fixture
from test_weg2_read_room_1015 import (
    TOTAL,
    _fill_arena_with_tree_pages,
    _ov,
    _read,
    _rr,
    _Scheduler,
    _setup,
    _TreeSim,
)
from sglang.srt.environ import envs

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

FULL = ComponentType.FULL
MAMBA = ComponentType.MAMBA


@pytest.fixture(autouse=True)
def _clean():
    _rr().reset_for_tests()
    R12._reset_state_for_tests()
    with envs.SGLANG_WEG2_L3FILL_ROOM_WAIT_MS.override(300), \
            _ov("SGLANG_WEG2_READ_ROOM_WAIT_MS", 4000), \
            envs.SGLANG_WEG2_L3FILL_SELF_JOIN_TAKEOVER.override(False):
        yield
    _rr().reset_for_tests()
    R12._reset_state_for_tests()


class _Roles:
    """The R12 role per thread: one process plays group P (no role), D TP0 (``host``) and its worker (``worker``)
    in turn, like the ranks of the two groups do in their own processes."""

    def __init__(self):
        self.now = {}

    def __call__(self):
        return self.now.get(threading.current_thread().name)


@pytest.fixture
def roles(monkeypatch):
    r = _Roles()
    monkeypatch.setattr(R12, "role", r)
    monkeypatch.setattr(_rr(), "form_a_role", r, raising=False)
    return r


# --- group D's tree: a real UnifiedRadixCache whose host rows name arena slots -----------------------------

_REAL_EVICT = UnifiedRadixCache._evict_component_and_detach_lru


def _evict_host(self_, node, comp, target=None, tracker=None):
    """The host-layer release of a D tree: the KV host rows give their arena references back (one per rank, the
    rows name the slots at page_size 1); an aux host row just goes."""
    ct = getattr(comp, "component_type", comp)
    cd = node.component_data[ct]
    hv = cd.host_value
    if target is None or EvictLayer.HOST not in target or hv is None:
        return 0, 0
    cd.host_value = None
    if ct == FULL:
        arena = getattr(self_, "_t_arena", None)
        if arena is not None:
            arena.ref_slots(hv.tolist(), -1)
        first = node.key.token_ids[0] if hasattr(node.key, "token_ids") else node.id
        self_._t_released.append(first)
        calls = getattr(self_, "_t_calls", None)
        if calls is not None:
            calls.append(("evict", first))
        return 0, int(hv.numel())
    return 0, 0


@pytest.fixture
def evict_mock():
    with mock.patch.object(UnifiedRadixCache, "_evict_component_and_detach_lru", _evict_host):
        yield


def _d_tree(spans, arena=None, tag="tp0"):
    """``spans``: [(first_token, slots)] -- one node per span (disjoint first tokens), its device value from the
    fixture allocator, its host rows = the arena slots (page_size 1, staging 0). The same spans on a second call
    build the worker's replica (same content keys: R12 matches by last page hash + depth)."""
    cache, allocator, _ = build_fixture(CacheConfig(page_size=1, components=(FULL,)))
    for first, slots in spans:
        n = len(slots)
        cache.insert(InsertParams(key=RadixKey(list(range(first, first + n)), None),
                                  value=allocator.alloc(n).to(dtype=torch.int64)))
    by_first = {}
    for node in cache._collect_all_nodes():
        if node is cache.root_node:
            continue
        first = list(node.key.token_ids)[0] if hasattr(node.key, "token_ids") else list(node.key)[0]
        by_first[first] = node
    for first, slots in spans:
        node = by_first[first]
        node.component_data[FULL].host_value = torch.tensor(list(slots), dtype=torch.int64)
        node.hash_value = [f"h{first}"]
    cache._t_arena = arena
    cache._t_released = []
    cache._t_tag = tag
    cache._t_nodes = by_first
    return cache


class _Pool:
    """D TP0's arena KV host pool, as far as the spill reads it: ``secure_rows_to_l3`` gives every page an L3
    copy and reports ``lost`` for a page whose stem has no file (the #257 (d) write cannot happen); the rows are
    the slots (staging 0, one row per page)."""

    staging_rows = 0
    page_size = 1

    def __init__(self, arena, stem_of_slot, store_root, calls=None):
        self.arena = arena
        self.stem_of_slot = stem_of_slot
        self.root = store_root
        self.calls = calls if calls is not None else []

    def secure_rows_to_l3(self, hv):
        slots = [int(s) for s in hv.tolist()]
        lost = sum(1 for s in slots if not os.path.exists(os.path.join(self.root, self.stem_of_slot[s] + ".bin")))
        self.calls.append(("secure", slots[0]))
        return {"pages": len(slots), "on_disk": len(slots) - lost, "written": 0, "lost": lost}


def _d_group(rr, roles, tp0, worker, pool, arena, stop, served):
    """Group D's scheduler pass, one thread: TP0 reads the room request and serves it, TP0 attaches the pass's
    R12 verdict to the broadcast, every worker consumes it before the pass's requests."""
    me = threading.current_thread().name
    seen = {}
    while not stop.is_set():
        roles.now[me] = "host"
        n = rr._unseen(arena, seen)
        if n > 0:
            served.append(rr.serve(tp0, pool, arena, n))
        verdicts = R12.attach([])
        roles.now[me] = "worker"
        R12.consume(verdicts, worker)
        roles.now[me] = "host"
        time.sleep(0.02)


def _store_files(tmp_path, stems):
    root = tmp_path / "store"
    for s in stems:
        (root / (s + ".bin")).write_bytes(b"\x33" * TOTAL)
    return str(root)


def _run_group_d(rr, roles, tp0, worker, pool, arena):
    stop = threading.Event()
    served: list = []
    th = threading.Thread(target=_d_group, args=(rr, roles, tp0, worker, pool, arena, stop, served), name="d-group")
    th.start()
    return stop, served, th


# --- the cases ---------------------------------------------------------------------------------------------

def test_weg2_70_293_group_d_gives_back_its_l3_backed_copies_for_p_s_read(tmp_path, roles, evict_mock):
    """The 13:35:34 shape at 1:100: 64 slots, P's running requests hold 22 (nothing to spill), group D's tree holds
    42 on TP0 AND the worker (two references per slot; one span without an L3 copy). P's read of 16 pages: RED
    before (D answers nothing, the read ends short), GREEN: TP0 releases the L3-backed spans, the R12 verdict drops
    the worker's references, the clock takes the slots, every page lands. The span without an L3 copy is kept."""
    stems = [f"r{i}" for i in range(16)]
    be, arena = _setup(tmp_path, stems, slots=64)
    _fill_arena_with_tree_pages(arena, 22, tag="pin")                  # P: running requests, not spillable
    d_slots = _fill_arena_with_tree_pages(arena, 42, tag="dpage")      # D TP0's reference
    arena.ref_slots(d_slots, +1)                                       # ... and the worker's
    stem_of = {s: f"dpage{i}" for i, s in enumerate(d_slots)}
    root = _store_files(tmp_path, [f"dpage{i}" for i in range(42) if not 34 <= i < 42])   # the last span: no L3
    spans = [(1, d_slots[0:10]), (2001, d_slots[10:20]), (4001, d_slots[20:34]), (6001, d_slots[34:42])]
    tp0 = _d_tree(spans, arena, "tp0")
    worker = _d_tree(spans, arena, "worker")
    pool = _Pool(arena, stem_of, root)
    p_tree = _TreeSim(arena, [])
    rr = _rr()
    stop, served, th = _run_group_d(rr, roles, tp0, worker, pool, arena)
    try:
        with _Scheduler(p_tree):
            out = _read(be, arena, stems)
    finally:
        stop.set()
        th.join(5)
    landed = sum(1 for s in out if s is not None)
    assert landed == len(stems), (landed, served, tp0._t_released, worker._t_released)
    assert sorted(tp0._t_released) == sorted(worker._t_released), (tp0._t_released, worker._t_released)
    assert 6001 not in tp0._t_released                                # no L3 copy: kept on every rank
    assert [int(r) for r in arena.slot_refs(d_slots[34:42])] == [2] * 8


def test_a_parked_span_given_back_is_read_again_in_full_from_l3_and_its_anchor_stays(tmp_path, roles, evict_mock):
    """Constraint D-NORECOMPUTE: the parked span (20 pages, ``#248 PARK-DEMOTE ... l3_pages=all``) and a finished
    node give their arena copies back for P's read; the park's anchor node (a Mamba state on the host) is NOT
    touched. P's read lands; P's prefill ends (its read references go). D's resume reads the parked span: all 20
    pages land -- the ones the clock took come back from their L3 files -- and the anchor's pages never left L2."""
    stems = [f"r{i}" for i in range(24)]
    be, arena = _setup(tmp_path, stems, slots=48)
    _fill_arena_with_tree_pages(arena, 16, tag="pin")
    park = _fill_arena_with_tree_pages(arena, 20, tag="park")
    anchor = _fill_arena_with_tree_pages(arena, 4, tag="anc")
    fin = _fill_arena_with_tree_pages(arena, 8, tag="fin")
    d_slots = park + anchor + fin
    arena.ref_slots(d_slots, +1)
    stem_of = {**{s: f"park{i}" for i, s in enumerate(park)}, **{s: f"anc{i}" for i, s in enumerate(anchor)},
               **{s: f"fin{i}" for i, s in enumerate(fin)}}
    root = _store_files(tmp_path, list(stem_of.values()))
    spans = [(1, park), (3001, anchor), (5001, fin)]
    tp0 = _d_tree(spans, arena, "tp0")
    worker = _d_tree(spans, arena, "worker")
    for t in (tp0, worker):
        t._t_nodes[3001].component_data[MAMBA].host_value = torch.tensor([7], dtype=torch.int64)
    pool = _Pool(arena, stem_of, root)
    rr = _rr()
    stop, served, th = _run_group_d(rr, roles, tp0, worker, pool, arena)
    try:
        with _Scheduler(_TreeSim(arena, [])):
            out = _read(be, arena, stems)
    finally:
        stop.set()
        th.join(5)
    assert all(s is not None for s in out), [i for i, s in enumerate(out) if s is None]
    assert 3001 not in tp0._t_released and 3001 not in worker._t_released
    assert 1 in tp0._t_released and 1 in worker._t_released
    arena.ref_slots([s for s in out if s is not None], -1)            # P's prefill is done with its read
    _slots, states = arena.find_slots_np([f"park{i}" for i in range(20)])
    gone = [i for i in range(20) if int(_slots[i]) < 0 or int(states[i]) != 2]
    assert gone, "the clock took parked slots for P's read -- the resume must come from L3"
    resume = _read(be, arena, [f"park{i}" for i in range(20)])        # D's wake: the parked span, by stem
    assert all(s is not None for s in resume), [i for i, s in enumerate(resume) if s is None]
    _slots, states = arena.find_slots_np([f"park{i}" for i in gone])
    assert all(int(x) >= 0 and int(st) == 2 for x, st in zip(_slots, states))   # back in L2, read from L3
    assert [int(r) for r in arena.slot_refs(anchor)] == [2] * 4       # the anchor's pages: untouched, both ranks


def test_tp0_releases_only_l3_backed_device_copies_and_sends_the_verdict(tmp_path, roles, evict_mock):
    """``form_a_spill`` on a real tree: A goes (L3 copy first, then the host rows, ``l3_present``, STATE kv=0);
    B (Mamba anchor on the host), C (host lock: a load in flight), D (pending write-through), E (host-only), G (no
    L3 copy) all stay."""
    rr = _rr()
    roles.now[threading.current_thread().name] = "host"
    root = tmp_path / "store"
    root.mkdir()
    names = {1: "A", 101: "B", 201: "C", 301: "D", 401: "E", 501: "G"}
    spans = [(f, list(range(i * 8, i * 8 + 8))) for i, f in enumerate(names)]
    stem_of = {s: f"s{s}" for _f, sl in spans for s in sl}
    for f, sl in spans:
        if names[f] != "G":
            for s in sl:
                (root / f"s{s}.bin").write_bytes(b"\x01")
    tree = _d_tree(spans, None, "tp0")
    nd = tree._t_nodes
    nd[101].component_data[MAMBA].host_value = torch.tensor([3], dtype=torch.int64)
    nd[201].component_data[FULL].host_lock_ref = 1
    nd[301].write_through_pending_id = 7
    nd[401].component_data[FULL].value = None
    calls = []
    tree._t_calls = calls
    pool = _Pool(None, stem_of, str(root), calls)
    got = rr.form_a_spill(tree, pool, 1)
    assert got == 8 and tree._t_released == [1], (got, tree._t_released)
    assert nd[1].l3_present
    assert calls.index(("secure", 0)) < calls.index(("evict", 1)), calls     # the L3 copy before the release
    v = R12.attach([])
    states = [e for e in v[0].events if e[0] == R12.STATE]
    assert [(e[1], e[4]) for e in states] == [("h1", 0)], states


def test_a_worker_never_releases_on_its_own(tmp_path, roles, evict_mock):
    rr = _rr()
    roles.now[threading.current_thread().name] = "worker"
    tree = _d_tree([(1, list(range(8)))], None, "worker")
    assert rr.form_a_spill(tree, _Pool(None, {}, str(tmp_path)), 64) == 0
    assert tree._t_released == []
    # and its round never reads the request file: TP0 answers for the group
    seen = []
    with mock.patch.object(rr, "_unseen", lambda a, s=None: (seen.append(1), 0)[1]):
        tree._weg2_direct_pool = lambda: types_ns(arena=object())
        for _ in range(5):
            rr.tick(tree)
            time.sleep(0.03)
    assert seen == []
    assert rr.armed(tree) and not rr.serves(tree)


def types_ns(**kw):
    import types

    return types.SimpleNamespace(**kw)


def test_the_r12_verdict_drops_the_same_rows_on_the_worker_and_keeps_the_anchor(tmp_path, roles, evict_mock):
    """TP0's verdict on the real wire: the worker's replica drops the KV host rows of the released node (its own
    ``_reconcile``, the same code every Form A verdict runs) and keeps the anchor node whole."""
    rr = _rr()
    me = threading.current_thread().name
    root = tmp_path / "store"
    root.mkdir()
    spans = [(1, list(range(0, 8))), (101, list(range(8, 16)))]
    for _f, sl in spans:
        for s in sl:
            (root / f"s{s}.bin").write_bytes(b"\x01")
    tp0 = _d_tree(spans, None, "tp0")
    worker = _d_tree(spans, None, "worker")
    for t in (tp0, worker):
        t._t_nodes[101].component_data[MAMBA].host_value = torch.tensor([3], dtype=torch.int64)
    roles.now[me] = "host"
    rr.form_a_spill(tp0, _Pool(None, {s: f"s{s}" for _f, sl in spans for s in sl}, str(root)), 1)
    v = R12.attach([])
    roles.now[me] = "worker"
    R12.consume(v, worker)
    assert worker._t_released == [1], worker._t_released
    assert worker._t_nodes[1].component_data[FULL].host_value is None
    assert worker._t_nodes[101].component_data[FULL].host_value is not None


def test_arming_local_pp_and_form_a_tp0_serve_a_worker_waits_a_plain_tp_group_neither(tmp_path, roles):
    rr = _rr()
    me = threading.current_thread().name
    p = _TreeSim(None, [])
    d = _TreeSim(None, [])
    setattr(d, SF.FLOOR_LOCAL_PP_ATTR, False)
    assert rr.armed(p) and rr.serves(p)
    roles.now[me] = "host"
    assert rr.armed(d) and rr.serves(d)
    roles.now[me] = "worker"
    assert rr.armed(d) and not rr.serves(d)
    roles.now[me] = None
    assert not rr.armed(d) and not rr.serves(d)
    with envs.SGLANG_WEG2_READ_ROOM.override(False):
        roles.now[me] = "host"
        assert not rr.armed(d) and not rr.serves(d)


def test_tp0_spills_through_the_anchor_host_pool_of_a_hybrid_group(tmp_path, roles, evict_mock):
    """NF D's host pool is a hybrid ``HostPoolGroup`` (KV + anchors) that forwards the claim calls only: the spill
    finds ``secure_rows_to_l3`` on its anchor host pool, as the P stages' W3 spill does -- with the P-only profile
    switch ``SGLANG_WEG2_ENABLE_W3_SPILL_ANCHOR_POOL`` OFF, as it is on D (``NF_ENV_P`` only)."""
    import types

    rr = _rr()
    roles.now[threading.current_thread().name] = "host"
    root = tmp_path / "store"
    root.mkdir()
    spans = [(1, list(range(0, 8)))]
    for s in range(8):
        (root / f"s{s}.bin").write_bytes(b"\x01")
    tree = _d_tree(spans, None, "tp0")
    inner = _Pool(None, {s: f"s{s}" for s in range(8)}, str(root))
    group = types.SimpleNamespace(staging_rows=0, anchor_entry=types.SimpleNamespace(host_pool=inner))
    with envs.SGLANG_WEG2_ENABLE_W3_SPILL_ANCHOR_POOL.override(False):
        assert rr.form_a_spill(tree, group, 1) == 8
    assert inner.calls == [("secure", 0)]


def test_tp0_stops_at_the_want(tmp_path, roles, evict_mock):
    """want = max(need, 64): a need of 70 takes two 40-page nodes, not all four."""
    rr = _rr()
    roles.now[threading.current_thread().name] = "host"
    root = tmp_path / "store"
    root.mkdir()
    spans = [(1 + 1000 * i, list(range(40 * i, 40 * i + 40))) for i in range(4)]
    for _f, sl in spans:
        for s in sl:
            (root / f"s{s}.bin").write_bytes(b"\x01")
    tree = _d_tree(spans, None, "tp0")
    got = rr.form_a_spill(tree, _Pool(None, {s: f"s{s}" for _f, sl in spans for s in sl}, str(root)), 70)
    assert got == 80 and len(tree._t_released) == 2, (got, tree._t_released)
