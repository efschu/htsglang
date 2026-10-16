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
        node.hash_value = [f"h{first}_{i}" for i in range(len(slots))]     # one page hash per page (page_size 1)
    cache._t_arena = arena
    cache._t_alloc = allocator
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
        self.arena_slots = int(arena.stats()["slots"]) if arena is not None else 0
        self.stem_of_slot = stem_of_slot
        self.root = store_root
        self.calls = calls if calls is not None else []

    def secure_rows_to_l3(self, hv):
        slots = [int(s) for s in hv.tolist()]
        lost = sum(1 for s in slots if not os.path.exists(os.path.join(self.root, self.stem_of_slot[s] + ".bin")))
        self.calls.append(("secure", slots[0]))
        return {"pages": len(slots), "on_disk": len(slots) - lost, "written": 0, "lost": lost, "unpaired": 0}


# --- review readroomd (wf_a0c826e9-e01): the REAL release funnel, service at the broadcast ----------------------
# The tree's own ``_evict_component_and_detach_lru`` (y5h aux net, LRU detach, the FULL component's evict_component)
# runs unstubbed; only the FULL component's host pool is a stand-in whose ``free`` gives the rows' arena references
# back (one per rank, the rows name the slots at page_size 1). An SWA stand-in carries an aux state with a host value.

SWA = ComponentType.SWA


class _HostPool:
    def __init__(self, arena):
        self.arena = arena

    def free(self, hv):
        if self.arena is not None:
            self.arena.ref_slots([int(x) for x in hv.tolist()], -1)


class _Lru:
    def in_list(self, n):
        return False

    def remove_node(self, n):
        pass


class _AuxSWA:
    """An aux component with a device and a host value (the order "aux host before KV host" is what MA breaks)."""

    component_type = SWA

    def evict_component(self, node, target=None):
        cd = node.component_data[SWA]
        hf = 0
        if EvictLayer.HOST in target and cd.host_value is not None:
            hf = int(cd.host_value.numel())
            cd.host_value = None
        return 0, hf


def _real_d_tree(spans, arena=None, tag="tp0", aux=False):
    t = _d_tree(spans, arena, tag)
    for comp in t._components_tuple:
        if comp.component_type == FULL:
            comp._full_kv_pool_host = _HostPool(arena)
    if aux:
        t._components_tuple = tuple(t._components_tuple) + (_AuxSWA(),)
        t.lru_lists[SWA] = _Lru()
        t.host_lru_lists[SWA] = _Lru()
    return t


def _released(tree):
    return sorted(f for f, n in tree._t_nodes.items() if n.component_data[FULL].host_value is None)


def _broadcast(roles, tp0, worker):
    """One pass of group D: TP0 attaches (serves the room requests, releases what it decided, the verdicts ride the
    broadcast), the worker consumes before the pass's requests."""
    me = threading.current_thread().name
    roles.now[me] = "host"
    R12.register_tree(tp0)
    v = R12.attach([])
    roles.now[me] = "worker"
    R12.consume(v, worker)
    roles.now[me] = "host"
    return v


def _store_files(tmp_path, stems):
    root = tmp_path / "store"
    root.mkdir(exist_ok=True)
    for s in stems:
        (root / (s + ".bin")).write_bytes(b"\x33" * TOTAL)
    return str(root)


def _group_d_loop(roles, tp0, worker, stop):
    while not stop.is_set():
        _broadcast(roles, tp0, worker)
        time.sleep(0.02)


def test_weg2_70_293_group_d_gives_back_its_l3_backed_copies_at_its_broadcast(tmp_path, roles):
    """The 13:35:34 shape at 1:100 with the REAL funnel: 64 slots, P's running requests hold 22, D's tree 42 on TP0
    AND the worker (two references per slot; one span without an L3 copy). P's read of 16 pages: TP0 serves the
    request at its broadcast (attach), releases the L3-backed spans, the STATE + L3 verdicts make the worker do the
    same before the pass's admission, the clock takes the slots, every page lands. The span without an L3 copy is
    kept on every rank; l3_present is equal on both ranks."""
    stems = [f"r{i}" for i in range(16)]
    be, arena = _setup(tmp_path, stems, slots=64)
    _fill_arena_with_tree_pages(arena, 22, tag="pin")
    d_slots = _fill_arena_with_tree_pages(arena, 42, tag="dpage")
    arena.ref_slots(d_slots, +1)
    stem_of = {s: f"dpage{i}" for i, s in enumerate(d_slots)}
    root = _store_files(tmp_path, [f"dpage{i}" for i in range(34)])
    spans = [(1, d_slots[0:10]), (2001, d_slots[10:20]), (4001, d_slots[20:34]), (6001, d_slots[34:42])]
    tp0 = _real_d_tree(spans, arena, "tp0")
    worker = _real_d_tree(spans, arena, "worker")
    pool = _Pool(arena, stem_of, root)
    tp0._weg2_direct_pool = lambda: pool
    _rr()._ST.arena = arena
    stop = threading.Event()
    th = threading.Thread(target=_group_d_loop, args=(roles, tp0, worker, stop), name="d-group")
    th.start()
    try:
        time.sleep(0.1)                                    # D's first broadcast arms READ-ROOM on this arena
        out = _read(be, arena, stems)
    finally:
        stop.set()
        th.join(5)
    assert sum(1 for s in out if s is not None) == len(stems)
    assert _released(tp0) == _released(worker) and 6001 not in _released(tp0), (_released(tp0), _released(worker))
    for f in _released(tp0):
        assert tp0._t_nodes[f].l3_present and worker._t_nodes[f].l3_present, f
    assert not worker._t_nodes[6001].l3_present and not tp0._t_nodes[6001].l3_present
    assert [int(r) for r in arena.slot_refs(d_slots[34:42])] == [2] * 8


def test_tp0_never_serves_inside_the_round_and_not_without_a_broadcast(tmp_path, roles):
    """B2/B5: a room request posted while D sleeps (sleep drain: rounds, no broadcast) is not served; TP0's tree
    and the worker's stay equal inside the pass. The next broadcast serves it."""
    rr = _rr()
    me = threading.current_thread().name
    spans = [(1, list(range(0, 8)))]
    be, arena = _setup(tmp_path, ["r0"], slots=16)
    _fill_arena_with_tree_pages(arena, 16, tag="live")              # full: the request has a deficit
    root = _store_files(tmp_path, [])
    for i in range(8):
        (tmp_path / "store" / f"s{i}.bin").write_bytes(b"\x01")
    tp0 = _real_d_tree(spans, None, "tp0")
    worker = _real_d_tree(spans, None, "worker")
    pool = _Pool(arena, {s: f"s{s}" for s in range(8)}, root)
    tp0._weg2_direct_pool = lambda: pool
    roles.now[me] = "host"
    rr._ST.armed, rr._ST.arena = True, arena
    assert rr.post(arena, 4, read_id=7)
    for _ in range(5):
        rr._ST.poll_t = 0.0
        rr.tick(tp0)
    assert _released(tp0) == [] and tp0._t_nodes[1].backuped == worker._t_nodes[1].backuped
    assert pool.calls == [] and rr._ST.d_pending == [], "no L3 write, no decision inside the round"
    _broadcast(roles, tp0, worker)
    assert _released(tp0) == _released(worker) == [1]


def test_tp0_releases_only_l3_backed_childless_device_copies_through_the_real_funnel(tmp_path, roles):
    """``form_a_spill`` decides, the broadcast releases (real funnel): A goes (L3 copy first, then the host rows,
    l3_present on both ranks, STATE kv=0 + L3). Kept: B (Mamba anchor on the host), B2 (Mamba state on the device),
    C (host lock), D (pending write-through), W (ongoing_write_through), E (host-only), G (no L3 copy), H (staging
    rows), K (has a child -- its drop would need a claim)."""
    rr = _rr()
    me = threading.current_thread().name
    roles.now[me] = "host"
    root = tmp_path / "store"
    root.mkdir()
    firsts = [1, 101, 151, 201, 301, 351, 401, 501, 601, 701]
    spans = [(f, list(range(100 + 8 * i, 108 + 8 * i))) for i, f in enumerate(firsts)]
    spans[firsts.index(601)] = (601, list(range(0, 8)))      # H: staging rows
    for f, sl in spans:
        if f != 501:
            for s in sl:
                (root / f"s{s}.bin").write_bytes(b"\x01")
    tp0 = _real_d_tree(spans, None, "tp0")
    worker = _real_d_tree(spans, None, "worker")
    # K: a node with a child (insert a longer key below 701)
    for t in (tp0, worker):
        nd = t._t_nodes
        nd[101].component_data[MAMBA].host_value = torch.tensor([3], dtype=torch.int64)
        nd[151].component_data[MAMBA].value = torch.tensor([4], dtype=torch.int64)
        nd[201].component_data[FULL].host_lock_ref = 1
        nd[301].write_through_pending_id = 7
        t.ongoing_write_through[nd[351].id] = object()
        nd[401].component_data[FULL].value = None
        t.insert(InsertParams(key=RadixKey(list(range(701, 713)), None),
                              value=t._t_alloc.alloc(12).to(dtype=torch.int64)))
        assert nd[701].children
    pool = _Pool(None, {s: f"s{s}" for _f, sl in spans for s in sl}, str(root))
    pool.staging_rows = 4
    assert rr.form_a_spill(tp0, pool, 1) == 8
    assert _released(tp0) == [], "decided, not released inside the pass"
    v = _broadcast(roles, tp0, worker)
    assert _released(tp0) == [1] and _released(worker) == [1], (_released(tp0), _released(worker))
    assert tp0._t_nodes[1].l3_present and worker._t_nodes[1].l3_present
    kinds = [(e[0], e[1]) for e in v[0].events]
    assert (R12.RRSTATE, "h1_7") in kinds and (R12.L3, "h1_7") in kinds, kinds


def types_ns(**kw):
    import types

    return types.SimpleNamespace(**kw)


def test_aux_host_rows_leave_before_the_kv_rows_and_the_host_leaf_entry_goes(tmp_path, roles):
    """The real funnel, an aux component with a host value: the aux host state is released BEFORE the KV host rows
    (the tree law "aux host requires Full host"), and a stale host-leaf entry of the node is dropped."""
    rr = _rr()
    roles.now[threading.current_thread().name] = "host"
    root = tmp_path / "store"
    root.mkdir()
    for s in range(8):
        (root / f"s{s}.bin").write_bytes(b"\x01")
    tp0 = _real_d_tree([(1, list(range(8)))], None, "tp0", aux=True)
    node = tp0._t_nodes[1]
    node.component_data[SWA].value = torch.tensor([1], dtype=torch.int64)
    node.component_data[SWA].host_value = torch.tensor([9], dtype=torch.int64)
    tp0.evictable_host_leaves.add(node)
    order = []
    real = UnifiedRadixCache._evict_component_and_detach_lru

    def spy(self_, n, comp, target=EvictLayer.DEVICE, tracker=None):
        order.append(getattr(comp, "component_type", comp))
        return real(self_, n, comp, target=target, tracker=tracker)

    with mock.patch.object(UnifiedRadixCache, "_evict_component_and_detach_lru", spy):
        rr.form_a_spill(tp0, _Pool(None, {s: f"s{s}" for s in range(8)}, str(root)), 1)
        rr.flush_forma_pending()
    assert order and order[0] == SWA and FULL in order, order
    assert node.component_data[SWA].host_value is None and node.component_data[FULL].host_value is None
    assert node not in tp0.evictable_host_leaves


def test_a_page_without_a_complete_l3_copy_is_kept(tmp_path, roles):
    """B3: unpaired pages, a page that dropped out of ``pages`` (not COMPLETE) and a copy neither on disk nor written
    keep the node's host copy and leave l3_present False."""
    rr = _rr()
    roles.now[threading.current_thread().name] = "host"
    tp0 = _real_d_tree([(1, list(range(8)))], None, "tp0")

    class P:
        staging_rows = 0

        def __init__(self, answer):
            self.answer = answer

        def secure_rows_to_l3(self, hv):
            return dict(self.answer)

    for ans in ({"pages": 8, "on_disk": 0, "written": 0, "lost": 0, "unpaired": 8},
                {"pages": 8, "on_disk": 8, "written": 0, "lost": 0, "unpaired": 2},     # KV on disk, QSA twin not
                {"pages": 7, "on_disk": 7, "written": 0, "lost": 0, "unpaired": 0},
                {"pages": 8, "on_disk": 4, "written": 0, "lost": 0, "unpaired": 0}):
        assert rr.form_a_spill(tp0, P(ans), 1) == 0, ans
        rr.flush_forma_pending()
        assert _released(tp0) == [] and not tp0._t_nodes[1].l3_present, ans
    assert rr.l3_complete(tp0._t_nodes[1], {"pages": 8, "on_disk": 5, "written": 3, "lost": 0, "unpaired": 0})


def test_a_worker_never_releases_on_its_own(tmp_path, roles):
    rr = _rr()
    roles.now[threading.current_thread().name] = "worker"
    tree = _real_d_tree([(1, list(range(8)))], None, "worker")
    assert rr.form_a_spill(tree, _Pool(None, {}, str(tmp_path)), 64) == 0
    rr.flush_forma_pending()
    assert _released(tree) == []
    assert rr.armed(tree) and not rr.serves(tree)


def test_the_r12_verdict_drops_the_same_rows_and_sets_l3_present_on_the_worker_anchor_kept(tmp_path, roles):
    rr = _rr()
    root = tmp_path / "store"
    root.mkdir()
    spans = [(1, list(range(0, 8))), (101, list(range(8, 16)))]
    for _f, sl in spans:
        for s in sl:
            (root / f"s{s}.bin").write_bytes(b"\x01")
    tp0 = _real_d_tree(spans, None, "tp0")
    worker = _real_d_tree(spans, None, "worker")
    for t in (tp0, worker):
        t._t_nodes[101].component_data[MAMBA].host_value = torch.tensor([3], dtype=torch.int64)
    roles.now[threading.current_thread().name] = "host"
    rr.form_a_spill(tp0, _Pool(None, {s: f"s{s}" for _f, sl in spans for s in sl}, str(root)), 1)
    _broadcast(roles, tp0, worker)
    assert _released(worker) == [1] and worker._t_nodes[1].l3_present
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


def test_tp0_spills_through_the_anchor_host_pool_of_a_hybrid_group(tmp_path, roles):
    import types

    rr = _rr()
    roles.now[threading.current_thread().name] = "host"
    root = tmp_path / "store"
    root.mkdir()
    for s in range(8):
        (root / f"s{s}.bin").write_bytes(b"\x01")
    tree = _real_d_tree([(1, list(range(0, 8)))], None, "tp0")
    inner = _Pool(None, {s: f"s{s}" for s in range(8)}, str(root))
    group = types.SimpleNamespace(staging_rows=0, anchor_entry=types.SimpleNamespace(host_pool=inner))
    with envs.SGLANG_WEG2_ENABLE_W3_SPILL_ANCHOR_POOL.override(False):
        assert rr.form_a_spill(tree, group, 1) == 8
    assert inner.calls == [("secure", 0)]


def test_tp0_stops_at_the_want(tmp_path, roles):
    rr = _rr()
    roles.now[threading.current_thread().name] = "host"
    root = tmp_path / "store"
    root.mkdir()
    spans = [(1 + 1000 * i, list(range(40 * i, 40 * i + 40))) for i in range(4)]
    for _f, sl in spans:
        for s in sl:
            (root / f"s{s}.bin").write_bytes(b"\x01")
    tree = _real_d_tree(spans, None, "tp0")
    got = rr.form_a_spill(tree, _Pool(None, {s: f"s{s}" for _f, sl in spans for s in sl}, str(root)), 70)
    rr.flush_forma_pending()
    assert got == 80 and len(_released(tree)) == 2, (got, _released(tree))


class _WbController:
    write_policy = "write_back"

    def __init__(self):
        self.claims = 0


def test_a_spilled_d_leaf_leaves_the_device_without_a_claim_alike_on_tp0_and_the_worker(tmp_path, roles):
    """B4 with the REAL ``_evict_device_leaf`` on the TP0 and the worker tree at a FULL arena: after the spill and its
    broadcast the node is store-backed on both ranks (l3_present, no host copy). Evicting it from the device takes
    the same path on both: dropped without a new arena claim (``write_backup`` is never asked), the same tokens freed.
    The old path asked write_backup for a claim the full arena refuses -- the device stayed occupied."""
    rr = _rr()
    me = threading.current_thread().name
    root = tmp_path / "store"
    root.mkdir()
    for s in range(8):
        (root / f"s{s}.bin").write_bytes(b"\x01")
    spans = [(1, list(range(8)))]
    tp0 = _real_d_tree(spans, None, "tp0")
    worker = _real_d_tree(spans, None, "worker")
    roles.now[me] = "host"
    rr.form_a_spill(tp0, _Pool(None, {s: f"s{s}" for s in range(8)}, str(root)), 1)
    _broadcast(roles, tp0, worker)
    freed = {}
    for t, role in ((tp0, "host"), (worker, "worker")):
        roles.now[me] = role
        t.cache_controller = _WbController()
        node = t._t_nodes[1]
        assert node.l3_present and not node.backuped and t._is_device_leaf(node)

        def no_claim(*a, **k):
            raise AssertionError("a store-backed D leaf asked write_backup for a new arena claim")

        t.write_backup = no_claim
        tracker = {ct: 0 for ct in t.tree_components}
        t._evict_device_leaf(node, tracker)
        freed[role] = tracker[FULL]
        assert not any(c is node for c in t.root_node.children.values())
    assert freed["host"] == freed["worker"] == 8, freed


def test_over_spill_is_bounded_by_the_broadcast(tmp_path, roles):
    """B5: two broadcasts while the worker still holds (its consume not yet run): TP0 releases once for the read."""
    rr = _rr()
    me = threading.current_thread().name
    be, arena = _setup(tmp_path, ["r0"], slots=32)
    d_slots = _fill_arena_with_tree_pages(arena, 32, tag="dpage")
    arena.ref_slots(d_slots, +1)
    spans = [(1 + 1000 * i, d_slots[8 * i:8 * i + 8]) for i in range(4)]
    root = _store_files(tmp_path, [f"dpage{i}" for i in range(32)])
    stem_of = {s: f"dpage{i}" for i, s in enumerate(d_slots)}
    tp0 = _real_d_tree(spans, arena, "tp0")
    pool = _Pool(arena, stem_of, root)
    tp0._weg2_direct_pool = lambda: pool
    tp0.W3_SPILL_MIN_PAGES = 8
    roles.now[me] = "host"
    rr._ST.armed, rr._ST.arena = True, arena
    R12.register_tree(tp0)
    rid = rr.new_read_id(["x", "last"])
    for k in range(3):
        rr.post(arena, 8, read_id=rid) if k == 0 else None
        m = rr._map(arena, create=True)
        m[0:rr._REC.size] = rr._REC.pack(rid, 8, rr._now_ms() + k, 1234)     # a re-post every pass
        R12.attach([])
    assert len(_released(tp0)) == 1, _released(tp0)


def test_the_shared_w3_spill_releases_only_a_node_whose_every_page_has_its_l3_copy(tmp_path):
    """B3 in ``_w3_arena_spill`` too (every role): a page neither on disk nor written now keeps the node."""
    tree = _real_d_tree([(1, list(range(100, 108)))], None, "p")

    class P:
        staging_rows = 0
        arena = None

        def __init__(self, on_disk):
            self.on_disk = on_disk

        def secure_rows_to_l3(self, hv):
            return {"pages": 8, "on_disk": self.on_disk, "written": 0, "lost": 0, "unpaired": 0}

    tree.arena = None
    p = P(4)
    p.arena = object()
    assert tree._w3_arena_spill(p, 8) == 0 and _released(tree) == []
    tree._w3_spill_quiet_until = 0.0
    p.on_disk = 8
    assert tree._w3_arena_spill(p, 8) == 8 and _released(tree) == [1]


def test_at_the_broadcast_only_childless_anchorless_nodes_go(tmp_path, roles):
    """Through ``attach`` (the service path): a plain node goes; a node with a child (its drop would need a claim)
    and a node with a Mamba state on the device stay -- on TP0 and on the worker."""
    rr = _rr()
    me = threading.current_thread().name
    be, arena = _setup(tmp_path, ["r0"], slots=8)          # creates <tmp>/store
    root = tmp_path / "store"
    spans = [(1, list(range(0, 8))), (101, list(range(8, 16))), (201, list(range(16, 24)))]
    for s_ in range(24):
        (root / f"s{s_}.bin").write_bytes(b"\x01")
    _fill_arena_with_tree_pages(arena, 8, tag="live")
    tp0 = _real_d_tree(spans, None, "tp0")
    worker = _real_d_tree(spans, None, "worker")
    for t in (tp0, worker):
        t.insert(InsertParams(key=RadixKey(list(range(101, 113)), None),
                              value=t._t_alloc.alloc(12).to(dtype=torch.int64)))
        t._t_nodes[201].component_data[MAMBA].value = torch.tensor([4], dtype=torch.int64)
    pool = _Pool(arena, {s_: f"s{s_}" for s_ in range(24)}, str(root))
    tp0._weg2_direct_pool = lambda: pool
    roles.now[me] = "host"
    rr._ST.armed, rr._ST.arena = True, arena
    assert rr.post(arena, 24, read_id=9)
    _broadcast(roles, tp0, worker)
    assert _released(tp0) == _released(worker) == [1], (_released(tp0), _released(worker))


# --- review readroomd2 (wf_06e24d8f-8f0): the reviewer's probes R2_1-R2_4, R2_5b, R2_6, R2_7, R2_8, taken over
# unchanged (R2_5, the open pending window, stays out: it is named, not stopped -- review A4 -- and R2_5b pins that
# the window closes). Source: /spinning/wt-rev2-readroomd-1016/.../test_rev2_arenaread_d_1016.py

def _store(tmp_path, n):
    root = tmp_path / "store"
    root.mkdir(exist_ok=True)
    for s in range(n):
        (root / f"s{s}.bin").write_bytes(b"\x01")
    return str(root), {s: f"s{s}" for s in range(n)}


def _attached(t, node):
    return R12._attached(node) and any(c is node for c in node.parent.children.values())


def _store_backed_leaf_with(tmp_path, roles, extra=None):
    """TP0 tree with node 1 (8 tokens) l3_present, no host copy (the state after a READ-ROOM-D spill), write_back."""
    me = threading.current_thread().name
    roles.now[me] = "host"
    t = _real_d_tree([(1, list(range(8)))], None, "tp0")
    t.cache_controller = _WbController()
    n = t._t_nodes[1]
    n.component_data[FULL].host_value = None
    n.l3_present = True
    if extra is not None:
        extra(t, n)

    def no_claim(*a, **k):
        return 0

    t.write_backup = no_claim
    return t, n


def test_R2_1_forma_drop_never_orphans_a_host_only_child(tmp_path, roles):
    """A device leaf may carry host-only children (_is_device_leaf looks at device children only). The Form A
    L3-DROP must not delete such a node -- its children would be cut off the tree (#841)."""
    def add_host_child(t, n):
        t.insert(InsertParams(key=RadixKey(list(range(1, 13)), None),
                              value=t._t_alloc.alloc(12).to(dtype=torch.int64)))
        child = next(iter(n.children.values()))
        child.component_data[FULL].host_value = torch.tensor([90, 91, 92, 93], dtype=torch.int64)
        child.component_data[FULL].value = None
    t, n = _store_backed_leaf_with(tmp_path, roles, add_host_child)
    assert n.children
    tracker = {ct: 0 for ct in t.tree_components}
    dropped = UnifiedRadixCache._weg2_l3_drop(t, n, tracker)
    assert not dropped and _attached(t, n), (dropped, _attached(t, n))


def test_R2_2_forma_drop_never_takes_a_mamba_anchor(tmp_path, roles):
    """D-NORECOMPUTE: a node carrying a Mamba state on the device leaves the device only with a backup."""
    def add_anchor(t, n):
        n.component_data[MAMBA].value = torch.tensor([4], dtype=torch.int64)
    t, n = _store_backed_leaf_with(tmp_path, roles, add_anchor)
    tracker = {ct: 0 for ct in t.tree_components}
    dropped = UnifiedRadixCache._weg2_l3_drop(t, n, tracker)
    assert not dropped and _attached(t, n)


def test_R2_3_forma_drop_waits_for_a_write_in_flight(tmp_path, roles):
    def busy(t, n):
        t.ongoing_write_through[4711] = (n, None, [n])
        n.write_through_pending_id = 4711
    t, n = _store_backed_leaf_with(tmp_path, roles, busy)
    tracker = {ct: 0 for ct in t.tree_components}
    dropped = UnifiedRadixCache._weg2_l3_drop(t, n, tracker)
    t.ongoing_write_through.clear()
    assert not dropped and _attached(t, n)


def test_R2_4_forma_drop_never_takes_a_node_that_is_not_in_the_store(tmp_path, roles):
    """D-NORECOMPUTE: an un-backed device leaf WITHOUT l3_present has its KV nowhere else -- never dropped."""
    def not_in_l3(t, n):
        n.l3_present = False
    t, n = _store_backed_leaf_with(tmp_path, roles, not_in_l3)
    tracker = {ct: 0 for ct in t.tree_components}
    dropped = UnifiedRadixCache._weg2_l3_drop(t, n, tracker)
    assert not dropped and _attached(t, n)


def test_R2_6_an_unkeyed_node_is_never_released_without_a_verdict(tmp_path, roles):
    """record_state / record_l3 drop an unkeyed node silently (no hash_value: 'unkeyed state' warning, no event).
    l3_complete takes need = pages when hash_value is empty -- so TP0 would drop the host rows and set l3_present
    while no verdict reaches the worker."""
    rr = _rr()
    me = threading.current_thread().name
    root, stem_of = _store(tmp_path, 8)
    spans = [(1, list(range(8)))]
    tp0 = _real_d_tree(spans, None, "tp0")
    worker = _real_d_tree(spans, None, "worker")
    for t in (tp0, worker):
        t._t_nodes[1].hash_value = None
    roles.now[me] = "host"
    rr.form_a_spill(tp0, _Pool(None, stem_of, root), 1)
    _broadcast(roles, tp0, worker)
    a, b = tp0._t_nodes[1], worker._t_nodes[1]
    assert (a.backuped, a.l3_present) == (b.backuped, b.l3_present), ((a.backuped, a.l3_present),
                                                                      (b.backuped, b.l3_present))


class _PartialPool(_Pool):
    """secure_rows_to_l3 as for a node with 2 of 8 pages not COMPLETE (an owner's write unfinished): they drop out
    of ``pages``; lost 0, unpaired 0."""

    def secure_rows_to_l3(self, hv):
        return {"pages": 6, "on_disk": 6, "written": 0, "lost": 0, "unpaired": 0}


def test_R2_7_strict_off_never_releases_a_node_with_unfinished_pages(tmp_path, roles):
    """SGLANG_WEG2_L3_PRESENT_STRICT=0 is H16P A2's switch for UNPAIRED pages. With it off, l3_complete returns
    ``pages > 0`` -- the page-count check (a page some owner has not finished) goes too, in form_a_spill only
    (_w3_arena_spill keeps its Form A ``pages < len(hash_value)`` check, spill_host_only its ``pages != len``)."""
    rr = _rr()
    me = threading.current_thread().name
    root, stem_of = _store(tmp_path, 8)
    tp0 = _real_d_tree([(1, list(range(8)))], None, "tp0")
    roles.now[me] = "host"
    with envs.SGLANG_WEG2_L3_PRESENT_STRICT.override(False):
        got = rr.form_a_spill(tp0, _PartialPool(None, stem_of, root), 1)
        rr.flush_forma_pending()
    assert got == 0 and not tp0._t_nodes[1].l3_present, (got, tp0._t_nodes[1].l3_present)


def test_R2_5b_the_window_closes_when_the_pending_state_applies(tmp_path, roles):
    """Companion of R2_5 (expected GREEN): the next broadcast after the worker's lock is gone applies the pending
    STATE; the worker's host leaf is deleted -> both trees agree again. The difference lives exactly one pending
    window long."""
    rr = _rr()
    me = threading.current_thread().name
    root, stem_of = _store(tmp_path, 8)
    spans = [(1, list(range(8)))]
    tp0 = _real_d_tree(spans, None, "tp0")
    worker = _real_d_tree(spans, None, "worker")
    roles.now[me] = "host"
    rr.form_a_spill(tp0, _Pool(None, stem_of, root), 1)
    wn = worker._t_nodes[1]
    wn.component_data[FULL].host_lock_ref = 1
    _broadcast(roles, tp0, worker)
    wn.component_data[FULL].host_lock_ref = 0
    for t, role in ((tp0, "host"), (worker, "worker")):
        roles.now[me] = role
        t.cache_controller = _WbController()
        t.write_backup = lambda *a, **k: 0
        t._evict_device_leaf(t._t_nodes[1], {ct: 0 for ct in t.tree_components})
    _broadcast(roles, tp0, worker)          # empty verdict list: the worker re-applies its pending STATE
    assert _attached(tp0, tp0._t_nodes[1]) == _attached(worker, wn) is False


class _DirectPool:
    """The arena KV host pool as ``_weg2_direct_complete`` reads it at a worker's own direct-write ack: the page is
    NOT completed for all owners yet (metal h16q2 D: ``#1427 DIRECT-WRITE ACK node=32 ... completed_for_all=0`` on
    TP1)."""
    staging_rows = 0
    arena = object()

    def complete_write(self, hv):
        return 0

    def settle_lost_rows(self, hv, hashes):
        return {}


class _DirectCC:
    write_policy = "write_back"

    def __init__(self):
        self.mem_pool_host = _DirectPool()


def test_R2_8_a_worker_never_drops_a_device_leaf_tp0_keeps(tmp_path, roles):
    """THE #791b SHAPE, new with _weg2_l3_drop_forma. Per-rank arena claims (K10: 'TP1's claims needed 1409 pages,
    TP0's only 12'): TP0's backup of node 1 was refused (#1421 -> record_state, 'a worker may back it up'), the
    worker's own direct write landed. At the worker's OWN ack the real ``_weg2_direct_complete`` sets
    ``l3_present = True`` unconditionally (line 6386; completed_for_all=0) -- the writer K12b H1 did not route over
    R12. TP0's STATE (kv=0) then drops the worker's host rows. Both nodes: device-resident, un-backed; l3_present
    TP0 False, worker True. Device eviction at a full arena: TP0 asks write_backup (refused) and KEEPS the device
    rows, the worker takes the Form A L3-DROP and FREES them. Before 844e0aea3e both kept them."""
    me = threading.current_thread().name
    spans = [(1, list(range(8)))]
    tp0 = _real_d_tree(spans, None, "tp0")
    worker = _real_d_tree(spans, None, "worker")
    a, b = tp0._t_nodes[1], worker._t_nodes[1]
    a.component_data[FULL].host_value = None            # TP0's claim was refused: no host copy on TP0
    roles.now[me] = "worker"
    worker.cache_controller = _DirectCC()
    UnifiedRadixCache._weg2_direct_complete(worker, b)   # the worker's own direct-write ack
    roles.now[me] = "host"
    tp0.cache_controller = _WbController()
    UnifiedRadixCache._1421_refused(tp0, "arena_claim", a)   # R12: TP0 sends its state (no KV host)
    _broadcast(roles, tp0, worker)
    state = ((a.backuped, a.l3_present), (b.backuped, b.l3_present))
    freed = {}
    for t, role in ((tp0, "host"), (worker, "worker")):
        roles.now[me] = role
        t.cache_controller = _WbController()
        t.write_backup = lambda *a_, **k: 0             # the full arena refuses every new claim
        tracker = {ct: 0 for ct in t.tree_components}
        t._evict_device_leaf(t._t_nodes[1], tracker)
        freed[role] = tracker[FULL]
    assert freed["host"] == freed["worker"], (state, freed)


def test_a_failing_release_still_sends_the_verdict_of_what_was_done(tmp_path, roles):
    """Review readroomd2 A7: an exception inside the release (after the aux host state went) still sends the
    READ-ROOM STATE, which reads the node's ACTUAL flags -- TP0 never changes alone."""
    import pytest

    rr = _rr()
    roles.now[threading.current_thread().name] = "host"
    root = tmp_path / "store"
    root.mkdir()
    for s_ in range(8):
        (root / f"s{s_}.bin").write_bytes(b"\x01")
    tp0 = _real_d_tree([(1, list(range(8)))], None, "tp0", aux=True)
    node = tp0._t_nodes[1]
    node.component_data[SWA].value = torch.tensor([1], dtype=torch.int64)
    node.component_data[SWA].host_value = torch.tensor([9], dtype=torch.int64)
    rr.form_a_spill(tp0, _Pool(None, {s_: f"s{s_}" for s_ in range(8)}, str(root)), 1)
    real = UnifiedRadixCache._evict_component_and_detach_lru

    def boom(self_, n, comp, target=EvictLayer.DEVICE, tracker=None):
        if getattr(comp, "component_type", comp) == FULL:
            raise RuntimeError("release failed")
        return real(self_, n, comp, target=target, tracker=tracker)

    with mock.patch.object(UnifiedRadixCache, "_evict_component_and_detach_lru", boom), \
            pytest.raises(RuntimeError):
        rr.flush_forma_pending()
    events = [e for e in R12._S.ledger if e[0] == R12.RRSTATE]
    assert events and events[-1][4] == 1, (events, "the verdict says what TP0 holds: KV host rows still there")
    assert not node.l3_present and not any(e[0] == R12.L3 for e in R12._S.ledger)


def test_a_read_room_state_that_stays_pending_on_a_worker_is_named(tmp_path, roles, caplog):
    """Review readroomd2 A4: the worker's node is busy at consume (its own store write) -- the READ-ROOM STATE stays
    pending and is NAMED (READ-ROOM-STATE-DIVERGED out=pending); the next broadcast applies it."""
    import logging

    rr = _rr()
    me = threading.current_thread().name
    root = tmp_path / "store"
    root.mkdir()
    for s_ in range(8):
        (root / f"s{s_}.bin").write_bytes(b"\x01")
    spans = [(1, list(range(8)))]
    tp0 = _real_d_tree(spans, None, "tp0")
    worker = _real_d_tree(spans, None, "worker")
    roles.now[me] = "host"
    rr.form_a_spill(tp0, _Pool(None, {s_: f"s{s_}" for s_ in range(8)}, str(root)), 1)
    worker._t_nodes[1].component_data[FULL].host_lock_ref = 1
    with caplog.at_level(logging.WARNING):
        _broadcast(roles, tp0, worker)
    msgs = [r.getMessage() for r in caplog.records if "READ-ROOM-STATE-DIVERGED" in r.getMessage()]
    assert msgs and "out=pending" in msgs[-1], msgs
    worker._t_nodes[1].component_data[FULL].host_lock_ref = 0
    _broadcast(roles, tp0, worker)
    assert _released(worker) == [1]
