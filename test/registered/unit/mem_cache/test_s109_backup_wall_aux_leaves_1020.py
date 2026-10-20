"""S109 (NF 20.10.): the D backup wall -- host-only leaves whose END mamba anchor lives on the host only hold the KV
arena, and the host-leaf spill refused every one of them (``aux_host_only``).

METAL (D TP0, boots of 20.10.): ``PDFLIP D-WALL-HEAD ... reported_evictable=119616 payable_evictable=0`` (rc19
_074525 08:04:07Z) after ``EVICT-FRONTIER-CENSUS ... delivered_after_repair=0``, ``#1421 BACKUP-REFUSED
why=arena_claim`` and ``#1427 ARENA-DROP need=8 freed=0 stages=i:0,ii:0,iii:0``; the holders:
``S83 VICTIM-CENSUS ... pinned=6485 ... tree_host_only=6130 spill_why=host_only:4/6130`` (rc20 _081757 08:23:03Z)
beside ``W3-HOST-LEAF SPILL candidates=0 refused=device_resident:7,skip:6,aux_host_only:3,has_children:1``.

THE RULE (port of the 27B Q-1190b ``allow_aux`` stage, without its timer): after every plain leaf, a host-only leaf
whose only host-only aux state is its MAMBA anchor is a candidate; its KV pages and its anchor get their L3 copy first
(the anchor by the S39 helper, bounded); an anchor not settled / not secured keeps the leaf, named. Form A: TP0
decides alone, a key a worker cannot delete stays (K23/S22), workers never spill (R12).
"""
import heapq  # noqa: F401  (the module under test orders by heap)
import os
import types
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from flliper.srt.mem_cache import form_a_host_shadow as R12  # noqa: E402
from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType  # noqa: E402
from flliper.srt.pdflip import host_only_spill as HOS  # noqa: E402

L3_ASYNC_REAL = True   # S109b: this module tests the asynchronous forms themselves
FULL, MAMBA, SWA = ComponentType.FULL, ComponentType.MAMBA, getattr(ComponentType, "SWA", None)
P = 1


class _CD:
    def __init__(self, value=None, host_value=None):
        self.value, self.host_value = value, host_value
        self.lock_ref = self.host_lock_ref = 0


class _Node:
    _ids = iter(range(1000, 10 ** 6))

    def __init__(self, parent, pages, anchor_slot=None, aux_ct=MAMBA, start=0):
        self.id = next(_Node._ids)
        self.parent = parent
        self.children = {}
        self.evicted = True
        self.backuped = True
        self.hash_value = ["h%d-%d" % (self.id, j) for j in range(pages)]
        self.component_data = {FULL: _CD(None, torch.arange(start, start + pages, dtype=torch.int64)),
                               MAMBA: _CD(), SWA: _CD()} if SWA is not None else {
            FULL: _CD(None, torch.arange(start, start + pages, dtype=torch.int64)), MAMBA: _CD()}
        if anchor_slot is not None:
            self.component_data[aux_ct].host_value = torch.tensor([anchor_slot], dtype=torch.int64)
        if parent is not None:
            parent.children[self.id] = self


class _Keys(dict):
    def incomplete(self, node):
        return False

    def slots_of(self, node):
        return [int(x) for x in node.component_data[FULL].host_value.tolist()]


class _Backend:
    """The store side: a backend whose synchronous writer must NEVER be called from a spill (S109b)."""

    def _stat_stems(self, stems):
        raise AssertionError("S109b: no stat in the scheduler path")

    def arena_secure_to_disk(self, arena, cands, writer="claim_room"):
        raise AssertionError("S109b: no L3 write in the scheduler path")


class _Arena:
    """The mamba anchor arena: header generations/states and the write-behind's shared record (``_l3sec_pub``)."""

    def __init__(self, n=64, secured=()):
        self.slots = n
        self.path = None
        self.slot_bytes = 59 * 1024 * 1024
        self.gens = np.arange(n, dtype=np.int64) + 1
        self._l3sec_pub = (np.full(n, -1, dtype=np.int64), np.zeros(n, dtype=np.uint64))
        for s in secured:
            self.write_behind(s)

    def write_behind(self, slot):
        """What the write-behind thread records once the slot's L3 copy exists."""
        self._l3sec_pub[0][int(slot)] = self.gens[int(slot)]

    def slot_gens_np(self, slots):
        return self.gens[np.asarray(slots, dtype=np.int64)]

    def slot_states(self, slots):
        return [2] * len(slots)

    def slot_stem(self, s):
        return "a%d" % int(s)


class _MambaPool:
    def __init__(self, backend=None, settled=True, secured=(7,)):
        self._backend = backend or _Backend()
        self.arena = _Arena(secured=secured)
        self._settled = settled

    def settled_anchor_slots(self, hv):
        return [int(x) for x in hv.tolist()] if self._settled else []


class _KvPool:
    staging_rows = 0

    def __init__(self, lose=()):
        self.arena = object()
        self.lose = set(lose)
        self.secured = []

    def secure_rows_to_l3(self, hv):
        n = int(hv.numel())
        if int(hv.min()) in self.lose:
            return {"pages": n, "on_disk": 0, "written": 0, "lost": n}
        self.secured.append(n)
        return {"pages": n, "on_disk": n, "written": 0, "lost": 0}


class _Tree:
    """The parts of UnifiedRadixCache the host-leaf spill reads (FULL + MAMBA [+ SWA])."""

    def __init__(self, mamba_pool):
        self.root_node = _Node(None, 0)
        self.root_node.evicted = False
        self.ongoing_write_through = {}
        self.tree_components = (FULL, MAMBA)
        self._components_tuple = tuple(types.SimpleNamespace(component_type=c)
                                       for c in ((FULL, MAMBA, SWA) if SWA is not None else (FULL, MAMBA)))
        self._mp = mamba_pool
        self.evicted_order = []

    def _pdflip_mamba_pool(self):
        return self._mp

    def _collect_all_nodes(self):
        out, st = [], [self.root_node]
        while st:
            n = st.pop()
            out.append(n)
            st.extend(n.children.values())
        return out

    def _is_host_leaf(self, n):
        return (n is not self.root_node and n.evicted and n.backuped and not n.children
                and not any(cd.host_lock_ref for cd in n.component_data.values()))

    def _w3_spill_keys(self, pool, nodes):
        return _Keys({id(n): (0, n.id) for n in nodes})

    def _evict_host_leaf(self, node, tracker):
        assert self._is_host_leaf(node)
        tracker[FULL] += int(node.component_data[FULL].host_value.numel())
        for cd in node.component_data.values():
            cd.host_value = None
        node.parent.children.pop(node.id)
        self.evicted_order.append(node.id)


def _shape(secured=(7,), settled=True, aux_ct=MAMBA):
    """A plain host-only leaf (8 pages) and an aux leaf (16 pages, host-only mamba anchor in slot 7)."""
    t = _Tree(_MambaPool(settled=settled, secured=secured))
    plain = _Node(t.root_node, 8, start=100)
    aux = _Node(t.root_node, 16, anchor_slot=7, aux_ct=aux_ct, start=200)
    return t, plain, aux


@pytest.fixture(autouse=True)
def _no_role():
    with mock.patch.object(R12, "role", return_value=None):
        yield


def test_red_metal_shape_the_aux_leaf_was_refused_and_the_arena_stayed_full():
    """Before S109: ``refused=aux_host_only`` and released 0 for a want only the aux leaf could pay."""
    t, plain, aux = _shape()
    plain.parent.children.pop(plain.id)            # 081757: every host-only leaf carried its anchor
    pool = _KvPool()
    res = HOS.spill_host_only(t, pool, 16, P, set())
    assert res["released"] == 16 and res["aux_leaves"] == 1
    assert pool.secured == [16], "the KV pages got their L3 copy first"
    assert aux.id in t.evicted_order


def test_plain_leaves_go_first_the_aux_leaf_only_for_what_they_miss():
    t, plain, aux = _shape()
    pool = _KvPool()
    res = HOS.spill_host_only(t, pool, 8, P, set())
    assert res["released"] == 8 and t.evicted_order == [plain.id] and res["aux_leaves"] == 0
    res = HOS.spill_host_only(t, pool, 8, P, set())
    assert t.evicted_order == [plain.id, aux.id] and res["aux_leaves"] == 1


def test_an_anchor_the_write_behind_secured_goes_without_any_io():
    """The backend's stat and write raise: the spill reads only the write-behind's shared record."""
    t, plain, aux = _shape(secured=(7,))
    HOS.spill_host_only(t, _KvPool(), 24, P, set())
    assert aux.id in t.evicted_order


def test_an_anchor_without_l3_copy_is_queued_and_the_leaf_goes_only_after_the_write_behind(caplog):
    t, plain, aux = _shape(secured=())
    with caplog.at_level("INFO"):
        res = HOS.spill_host_only(t, _KvPool(), 24, P, set())
    assert aux.id not in t.evicted_order and res["released"] == 8
    assert aux.component_data[MAMBA].host_value is not None and aux.component_data[FULL].host_value is not None
    assert getattr(t._mp.arena, "_l3wb_urgent", set()) == {7}, "the scheduler only queues for the write-behind"
    assert any("aux_kept=child_anchor_pending:1" in m for m in caplog.messages)
    t._mp.arena.write_behind(7)                    # the write-behind thread's pass
    res = HOS.spill_host_only(t, _KvPool(), 24, P, set())
    assert aux.id in t.evicted_order and res["aux_leaves"] == 1


def test_an_anchor_in_flight_keeps_the_leaf():
    t, plain, aux = _shape(settled=False)
    res = HOS.spill_host_only(t, _KvPool(), 24, P, set())
    assert aux.id not in t.evicted_order and res["released"] == 8


def test_a_kv_page_without_l3_copy_keeps_the_aux_leaf():
    t, plain, aux = _shape()
    res = HOS.spill_host_only(t, _KvPool(lose={200}), 24, P, set())
    assert aux.id not in t.evicted_order and res["released"] == 8


@pytest.mark.skipif(SWA is None, reason="no second aux component type on this tree")
def test_another_host_only_aux_component_still_refuses():
    t, plain, aux = _shape(aux_ct=SWA)
    res = HOS.spill_host_only(t, _KvPool(), 24, P, set())
    assert aux.id not in t.evicted_order and res["released"] == 8


def test_an_aux_leaf_whose_parent_becomes_a_leaf_cascades():
    t, _p, _a = _shape()
    t.root_node.children.clear()
    mid = _Node(t.root_node, 8, anchor_slot=5, start=400)
    leaf = _Node(mid, 8, anchor_slot=6, start=500)
    t._mp.arena.write_behind(5)
    t._mp.arena.write_behind(6)
    res = HOS.spill_host_only(t, _KvPool(), 16, P, set())
    assert t.evicted_order == [leaf.id, mid.id] and res["aux_leaves"] == 2


# ---------------------------------------------------------------------------------------------------- Form A, three ranks
def _key(n):
    return ("k%d" % n.id, 0)


def test_three_ranks_tp0_decides_alone_and_a_key_a_worker_cannot_delete_stays():
    """Three differently cut trees: TP0 holds two aux leaves; worker 1 holds both as host-only leaves (deletable),
    worker 2 holds the first one WITH a child (cannot delete it). TP0 drops only the second; a worker never spills."""
    t0, _p, _a = _shape()
    t0.root_node.children.clear()
    a1 = _Node(t0.root_node, 8, anchor_slot=1, start=600)
    a2 = _Node(t0.root_node, 8, anchor_slot=2, start=700)
    t0._mp.arena.write_behind(1)
    t0._mp.arena.write_behind(2)
    nodelete = {_key(a1)}                       # worker 2's cut: a1 has a child there
    with mock.patch.object(R12, "role", return_value="host"), \
            mock.patch.object(R12, "worker_nodelete_keys", return_value=nodelete), \
            mock.patch.object(R12, "node_key", side_effect=_key):
        res = HOS.spill_host_only(t0, _KvPool(), 10 ** 6, P, set())
    assert t0.evicted_order == [a2.id] and res["aux_leaves"] == 1
    # a worker's spill never runs on its own (R12: _w3_arena_spill returns before the host-leaf round)
    import inspect

    from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    src = inspect.getsource(UnifiedRadixCache._w3_arena_spill)
    i_worker = src.index('if _role == "worker":')
    i_req = src.index("_r12.request_spill(", i_worker)
    i_hos = src.index("_hos.spill_host_only(")
    assert i_worker < i_req < i_hos


# ---------------------------------------------------------------------------------------------------- S109b: no file I/O
def test_no_scheduler_path_calls_a_stat_or_an_l3_write():
    """User 20.10.: "UND WEHE DAS SCHREIBEN IN DEN CACHE BLOCKIERT DURCH IRGENDEINE SYNCHRON ODER WARTESCHEISSE".
    The paths the spills run in the scheduler thread read the write-behind's record and queue -- none of them calls the
    synchronous writer or a stat any more."""
    import inspect

    from flliper.srt.mem_cache import l3_secured as L3S
    from flliper.srt.mem_cache.pool_host.arena_pool import ArenaMHAHostPool
    from flliper.srt.pdflip import forma_l3_subtree as FLS

    from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache as URC

    for fn in (ArenaMHAHostPool.secure_rows_to_l3, FLS._secure_child_anchors, HOS.spill_host_only, HOS._secure_anchor,
               L3S.secured, L3S.request, L3S.check, URC._pdflip_flush_spill_room, URC._pdflip_park_spill_room,
               URC._s109c_aux_device_round, URC._pdflip_mamba_claim_h19):
        src = inspect.getsource(fn)
        code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines()
                         if not ln.strip().startswith(('"', "'")))
        for bad in ("secure(arena", "arena_secure_to_disk(", "_stat_stems(", "write_pages(", "os.stat(", "fsync"):
            assert bad not in code, (fn.__qualname__, bad)


def test_secured_reads_the_shared_record_and_request_only_queues():
    from flliper.srt.mem_cache import l3_secured as L3S

    a = _Arena(n=8, secured=(1, 3))
    assert L3S.secured(a, [1, 2, 3]).tolist() == [True, False, True]
    a.gens[3] += 1                                   # the slot was reused: the old copy is not this page's
    assert L3S.secured(a, [3]).tolist() == [False]
    assert L3S.request(a, [2, 3]) == 2 and a._l3wb_urgent == {2, 3}
    assert sorted(L3S.take_urgent(a).tolist()) == [2, 3] and a._l3wb_urgent == set()


def test_no_record_yet_means_nothing_is_secured():
    from flliper.srt.mem_cache import l3_secured as L3S

    a = types.SimpleNamespace(slots=4, path=None, slot_gens_np=lambda s: np.ones(len(s), dtype=np.int64),
                              slot_states=lambda s: [2] * len(s))
    assert L3S.secured(a, [0, 1]).tolist() == [False, False]


def test_the_shared_record_is_one_mapping_for_every_process(tmp_path):
    """The write-behind (create) and a reader of another process (attach) see the same bytes."""
    from flliper.srt.mem_cache import l3_secured as L3S

    owner = types.SimpleNamespace(slots=8, path=str(tmp_path / "arena.bin"))
    reader = types.SimpleNamespace(slots=8, path=str(tmp_path / "arena.bin"),
                                   slot_gens_np=lambda s: np.full(len(s), 5, dtype=np.int64),
                                   slot_states=lambda s: [2] * len(s))
    assert L3S.arrays(reader, create=False) is None, "a reader never creates the record"
    g, _lo = L3S.arrays(owner, create=True)
    g[4] = 5
    g.flush()
    assert L3S.secured(reader, [3, 4]).tolist() == [False, False], "the reader re-tries attaching at most once a second"
    reader._l3sec_retry_at = 0.0                             # a second later
    assert L3S.secured(reader, [3, 4]).tolist() == [False, True]


# ---------------------------------------------------------------------------------------------------- S109c device round
class _DevNode:
    _ids = iter(range(5000, 10 ** 6))

    def __init__(self, pages, start):
        self.id = next(_DevNode._ids)
        self.hash_value = ["d%d-%d" % (self.id, j) for j in range(pages)]
        self.l3_present = False
        self.component_data = {FULL: _CD(torch.arange(pages), torch.arange(start, start + pages, dtype=torch.int64)),
                               MAMBA: _CD(None, torch.tensor([start], dtype=torch.int64))}


def _dev_tree(anchor_ok=True):
    from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    t = types.SimpleNamespace(components={FULL: object(), MAMBA: object()}, evictable_host_leaves=set(), released=[])
    t._w3_spill_keys = lambda pool, nodes: _Keys({id(n): (0, n.id) for n in nodes})

    def _evict(node, comp, target=None, tracker=None):
        assert comp is t.components[FULL], "the anchor leaves by the tree law with the Full host copy"
        n = int(node.component_data[FULL].host_value.numel())
        node.component_data[MAMBA].host_value = None       # the law: aux host first
        node.component_data[FULL].host_value = None
        t.released.append(node.id)
        return 0, n

    t._evict_component_and_detach_lru = _evict
    t.round = lambda pool, cands, want, role: UnifiedRadixCache._s109c_aux_device_round(t, pool, cands, want, role, 1)
    return t


def test_s109c_a_device_node_with_a_host_only_anchor_gives_kv_and_anchor_up_once_both_are_on_l3():
    from flliper.srt.pdflip import forma_l3_subtree as FLS

    t = _dev_tree()
    n = _DevNode(16, 900)
    states = []
    with mock.patch.object(FLS, "_secure_child_anchors", lambda tree, order, budget: None), \
            mock.patch.object(R12, "record_state", lambda tree, node, why=None: states.append((node.id, why))):
        out = t.round(_KvPool(), [n], 16, "host")
    assert out["released"] == 16 and out["spilled"] == 1 and n.l3_present
    assert n.component_data[FULL].value is not None, "the node stays on the device"
    assert states == [(n.id, "s109c_aux_device_spill")], "TP0's STATE goes to the workers"


def test_s109c_a_pending_anchor_keeps_the_node():
    from flliper.srt.pdflip import forma_l3_subtree as FLS

    t = _dev_tree()
    n = _DevNode(16, 900)
    with mock.patch.object(FLS, "_secure_child_anchors", lambda tree, order, budget: ("child_anchor_pending", n.id)):
        out = t.round(_KvPool(), [n], 16, "host")
    assert out["released"] == 0 and out["pending"] == 1 and t.released == []


def test_s109c_pending_kv_pages_keep_the_node():
    t = _dev_tree()
    n = _DevNode(16, 900)
    out = t.round(_KvPool(lose={900}), [n], 16, "host")
    assert out["released"] == 0 and out["pending"] == 1 and t.released == []


def test_s109c_a_worker_never_decides():
    t = _dev_tree()
    out = t.round(_KvPool(), [_DevNode(16, 900)], 16, "worker")
    assert out["released"] == 0 and t.released == []


def test_s109c_plain_device_nodes_first_then_the_aux_stage_then_the_host_leaf_round():
    import inspect

    from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    src = inspect.getsource(UnifiedRadixCache._w3_arena_spill)
    i_plain = src.index("for node in cands:")
    i_aux = src.index("_s109c_aux_device_round(")
    i_hos = src.index("_hos.spill_host_only(")
    assert i_plain < i_aux < i_hos
    assert "if released < want and aux_cands:" in src


def test_the_write_behind_keeps_its_record_in_the_shared_mapping_and_serves_requests_first():
    """The one writer: the write-behind pass (its own thread). Its secured record is the shared one every rank reads,
    and the slots a spill asked for lead its list."""
    import inspect

    from flliper.srt.mem_cache.hicache_storage import HiCacheFile

    src = inspect.getsource(HiCacheFile.l3_write_behind_pass)
    # S109-A5-N1: every publish goes through _l3wb_publish (publish, then the #1459 index re-read)
    assert src.count("self._l3wb_publish(arena,") == 3, "on disk, written, and written by the sibling group: all published"
    assert "_l3s.publish(" not in src, "a bare publish skips the index re-read (A5-N1)"
    hsrc = inspect.getsource(HiCacheFile._l3wb_publish)
    assert hsrc.index("_l3s.publish(arena, slots, gens)") < hsrc.index("idx.has(") < hsrc.index("base_gen[_b] = -1")
    i_take = src.index("_l3s.take_urgent(arena)")
    i_cur = src.index("new = new[int(np.searchsorted(slots[new], cur)):]")
    i_cat = src.index("new = np.concatenate([_urg_new, new])")
    assert i_take < i_cur < i_cat


# ---------------------------------------------------------------------------------------------------- S109d claim / clock / anchor
_needs_gcc = pytest.mark.skipif(__import__("shutil").which("gcc") is None, reason="needs gcc")


class _Idx:
    """The #1459 L3 stem index (shm): the evictor lists a file on commit and unlists it on its LRU unlink."""

    def __init__(self):
        self.files = set()

    def has(self, stems):
        return [st in self.files for st in stems]


def _write_behind(arena, idx, slots):
    """What one write-behind pass leaves behind for ``slots``: the file in the store (index) AND the record."""
    from flliper.srt.mem_cache import l3_secured as L3S

    for s_ in slots:
        idx.files.add(arena.slot_stem(int(s_)))
    L3S.mark(arena, slots)


def _unreferenced_full_arena(tmp_path, n=6):
    """A real C arena: ``n`` COMPLETE pages nobody references (the clock's candidates), a backend whose WRITER raises."""
    from test_w3_arena_spill_0929 import _finished, _pool

    p, arena, root = _pool(tmp_path, n)
    rows = [_finished(p, arena, [f"u{i}"], 0x50 + i) for i in range(n)]
    slots = []
    for r in rows:
        s_ = [int(x) for x in p.arena_ids(r).tolist()] if hasattr(p, "arena_ids") else []
        slots.extend(s_)
    stems = p._stems([f"u{i}" for i in range(n)])
    slots = [int(s) for s, _st in arena.find_slots(stems)]
    assert arena.ref_slots(slots, -1) == len(slots)          # the tree let go: unreferenced, COMPLETE

    def _no_write(*a, **k):
        raise AssertionError("S109d: no L3 write on the claim / clock path")

    p._backend.arena_secure_to_disk = _no_write
    p._backend._stat_stems = _no_write
    p._backend._l3idx = _Idx()
    return p, arena, slots


@_needs_gcc
def test_s109d_a_claim_frees_only_pages_with_a_published_copy_and_queues_the_rest(tmp_path):
    from flliper.srt.mem_cache import l3_secured as L3S

    p, arena, slots = _unreferenced_full_arena(tmp_path)
    assert p._evict_for_claim(arena, 2) == 0, "no page has an L3 copy yet: nothing is freed, nothing written"
    assert set(getattr(arena, "_l3wb_urgent", set())) == set(slots), "all queued for the write-behind"
    L3S.take_urgent(arena)
    _write_behind(arena, p._backend._l3idx, slots[:2])      # the write-behind's pass
    assert p._evict_for_claim(arena, 2) == 2
    assert arena.slot_states(slots[:2]) != [2, 2]
    assert arena.slot_states(slots[2:]) == [2] * (len(slots) - 2), "uncopied pages stay in L2"


@_needs_gcc
def test_s109d_the_clock_evict_frees_only_copied_pages(tmp_path):
    from flliper.srt.mem_cache import l3_secured as L3S

    p, arena, slots = _unreferenced_full_arena(tmp_path)
    from flliper.srt.mem_cache.hicache_storage import HiCacheFile

    be = p._backend
    be.pins = None
    assert HiCacheFile._arena_evict_to_disk(be, arena, 4) == 0
    assert arena.slot_states(slots) == [2] * len(slots), "no page has a copy: the clock frees nothing"
    _write_behind(arena, be._l3idx, slots[:3])
    # the clock takes at most an eighth of the arena per call (EVICT-KEEP): one page here; it WRITES nothing (returns 0)
    assert [HiCacheFile._arena_evict_to_disk(be, arena, 4) for _ in range(4)] == [0, 0, 0, 0]
    assert all(st != 2 for st in arena.slot_states(slots[:3])), "the copied pages left L2"
    assert arena.slot_states(slots[3:]) == [2] * (len(slots) - 3), "uncopied pages stay in L2"


@_needs_gcc
def test_s109_a1_record_and_index_both_or_nothing(tmp_path):
    """Review S109-1 (R4): a page the write-behind copied whose L3 file the store's LRU then unlinked is NOT secured;
    a file in the index without the record of THIS generation is not either."""
    from flliper.srt.mem_cache import l3_secured as L3S

    p, arena, slots = _unreferenced_full_arena(tmp_path)
    be = p._backend
    _write_behind(arena, be._l3idx, slots[:3])
    assert L3S.secured(arena, slots[:3], be).tolist() == [True, True, True]
    be._l3idx.files.discard(arena.slot_stem(slots[0]))       # the L3 LRU took the file of slot 0
    L3S.invalidate_stems(arena, [arena.slot_stem(slots[1])])  # ... and of slot 1, with the evict hook
    assert L3S.secured(arena, slots[:3], be).tolist() == [False, False, True]
    be._l3idx.files.add(arena.slot_stem(slots[4]))           # a file in the store, no record of this generation
    assert L3S.secured(arena, [slots[4]], be).tolist() == [False]
    assert p._evict_for_claim(arena, 3) == 1, "only the page both sources hold leaves L2"

def test_s109d_anchor_held_asks_the_index_never_a_stat():
    from flliper.srt.mem_cache.pool_host.arena_mamba_pool import ArenaMambaPoolHost

    def _stat(stems):
        raise AssertionError("S109d: no stat")

    def _pool(idx, listed):
        return types.SimpleNamespace(arena=types.SimpleNamespace(find_slots=lambda st: [(-1, 0)]),
                                     _stems=lambda hs: ["anc"],
                                     _backend=types.SimpleNamespace(_l3idx=idx, l3_listed=listed, _stat_stems=_stat))

    assert ArenaMambaPoolHost.anchor_held(_pool(object(), lambda st: [True]), "h") is True
    assert ArenaMambaPoolHost.anchor_held(_pool(object(), lambda st: [False]), "h") is False
    assert ArenaMambaPoolHost.anchor_held(_pool(None, lambda st: [True]), "h") is None, "no opened index: cannot tell"


def test_s109d_zero_file_io_on_every_scheduler_form():
    """The source check over ALL scheduler forms (S109b + S109d)."""
    import inspect

    from flliper.srt.mem_cache import l3_secured as L3S
    from flliper.srt.mem_cache.hicache_storage import HiCacheFile
    from flliper.srt.mem_cache.pool_host.arena_mamba_pool import ArenaMambaPoolHost
    from flliper.srt.mem_cache.pool_host.arena_pool import ArenaMHAHostPool
    from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache as URC
    from flliper.srt.pdflip import forma_l3_subtree as FLS

    fns = (ArenaMHAHostPool.secure_rows_to_l3, ArenaMHAHostPool._evict_for_claim, HiCacheFile._arena_evict_to_disk,
           HiCacheFile._arena_evict_candidates, HiCacheFile.l3_listed, ArenaMambaPoolHost.anchor_held,
           FLS._secure_child_anchors, HOS.spill_host_only, HOS._secure_anchor, L3S.secured, L3S.request, L3S.check,
           L3S.unsecured_keys_lo, URC._pdflip_flush_spill_room, URC._pdflip_park_spill_room,
           URC._s109c_aux_device_round, URC._pdflip_mamba_claim_h19)
    bad_tokens = ("secure(arena", "arena_secure_to_disk(", "_stat_stems(", "stat(", "write_pages(", "stat_sizes(",
                  "fsync", "open(")
    for fn in fns:
        src = inspect.getsource(fn)
        lines, in_doc = [], False
        for ln in src.splitlines():
            st = ln.strip()
            if st.count('"""') == 1:
                in_doc = not in_doc
                continue
            if in_doc or st.startswith(('"""', '"', "'", "#")):
                continue
            lines.append(ln.split("#", 1)[0])
        code = "\n".join(lines)
        for bad in bad_tokens:
            assert bad not in code, (fn.__qualname__, bad)


@_needs_gcc
def test_s109b_secure_rows_to_l3_reads_and_queues_never_writes(tmp_path):
    """The real pool's scheduler form: an uncopied page is pending (kept by every caller), queued, never written."""
    from test_w3_arena_spill_0929 import _finished, _pool

    from flliper.srt.mem_cache import l3_secured as L3S

    p, arena, root = _pool(tmp_path, 4)

    def _no_write(*a, **k):
        raise AssertionError("S109b: no L3 write / stat in secure_rows_to_l3")

    p._backend.arena_secure_to_disk = _no_write
    p._backend._stat_stems = _no_write
    p._backend._l3idx = _Idx()
    rows = _finished(p, arena, ["r0", "r1"], 0x61)
    out = p.secure_rows_to_l3(rows)
    assert (out["pages"], out["on_disk"], out["written"], out["pending_l3"]) == (2, 0, 0, 2)
    assert out["lost"] == 2 and out["incomplete"] == 2, "every caller keeps the node"
    slots = sorted(getattr(arena, "_l3wb_urgent", set()))
    assert len(slots) == 2
    _write_behind(arena, p._backend._l3idx, L3S.take_urgent(arena))
    out = p.secure_rows_to_l3(rows)
    assert (out["on_disk"], out["pending_l3"], out["lost"]) == (2, 0, 0)


# ---------------------------------------------------------------------------------------------------- S109-A2 transitive guard
import builtins as _builtins  # noqa: E402
import contextlib as _contextlib  # noqa: E402
import io as _io  # noqa: E402
import threading as _threading  # noqa: E402


@_contextlib.contextmanager
def _no_file_io_in_this_thread():
    """S109-A2 (review 20.10.): the TRANSITIVE check -- every call the code under test makes on THIS thread (the
    scheduler's) to the file system or the store's writer is counted, whatever function it went through. Other threads
    (the write-behind, the hand-over writer) are not counted."""
    from flliper.srt.mem_cache.hicache_storage import HiCacheFile
    from flliper.srt.pdflip import w3_room_set  # noqa: F401  (imported at boot: reads /proc once)

    me = _threading.get_ident()
    hits = []
    busy = []
    orig_open = _builtins.open

    def _wrap(owner, name):
        orig = getattr(owner, name)

        def w(*a, **k):
            if _threading.get_ident() == me and not busy:
                hits.append(name)
                if os.environ.get("S109_IO_TRACE"):
                    import traceback as _tb
                    busy.append(1)
                    try:
                        with orig_open("/tmp/s105/io_trace.txt", "a") as fh:
                            fh.write("IO %s %r\n%s\n" % (name, a[:1], "".join(_tb.format_stack(limit=9)[:-1])))
                    finally:
                        busy.pop()
            return orig(*a, **k)
        return orig, w

    targets = [(os, "stat"), (os, "lstat"), (os, "open"), (os, "fsync"), (_builtins, "open"), (_io, "open"),
               (HiCacheFile, "arena_secure_to_disk"), (HiCacheFile, "_stat_stems")]
    saved = []
    try:
        for owner, name in targets:
            orig, w = _wrap(owner, name)
            saved.append((owner, name, orig))
            setattr(owner, name, w)
        yield hits
    finally:
        for owner, name, orig in saved:
            setattr(owner, name, orig)


@_needs_gcc
def test_s109_a2_claim_clock_and_secure_rows_make_no_file_io_transitively(tmp_path):
    from test_w3_arena_spill_0929 import _finished
    from flliper.srt.mem_cache import l3_secured as L3S
    from flliper.srt.mem_cache.hicache_storage import HiCacheFile

    p, arena, slots = _unreferenced_full_arena(tmp_path)
    be = p._backend
    be.pins = None
    del be.arena_secure_to_disk, be._stat_stems            # the class methods, counted by the guard
    L3S.arrays(arena, create=True)                           # attached at the arena's open (HiCacheFile._open_arena)
    _write_behind(arena, be._l3idx, slots[:2])
    with _no_file_io_in_this_thread() as hits:
        p._evict_for_claim(arena, 2)
        HiCacheFile._arena_evict_to_disk(be, arena, 4)
    assert hits == [], "file I/O on the scheduler thread: %s" % hits
    rows = _finished(p, arena, ["g0"], 0x71)               # a slot the claim freed
    with _no_file_io_in_this_thread() as hits:
        p.secure_rows_to_l3(rows)
    assert hits == [], "file I/O on the scheduler thread: %s" % hits


@_needs_gcc
def test_s109_a2_h19_displacement_drops_no_anchor_without_a_copy_and_makes_no_file_io(tmp_path):
    """S108-2/S109-2: H19's SHARE displacement -> _pdflip_release_anchor -> drop_unreferenced -> _secure_before_drop
    made a stat and a 56-MiB write with fsync on the scheduler thread. Now: read the record, keep and queue."""
    import importlib.util

    from flliper.srt.mem_cache import l3_secured as L3S

    spec = importlib.util.spec_from_file_location(
        "_h19_s109", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pdflip",
                                  "test_pdflip_mamba_arena_displace_h19.py"))
    h19 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h19)
    arena = h19._arena(tmp_path, 6)
    L3S.arrays(arena, create=True)
    ranks = [h19._Rank(arena, r, cfg=2) for r in range(h19.RANKS)]
    with _no_file_io_in_this_thread() as hits:
        hs = h19._prefill(ranks, "pdflip-9-1", 5)
    assert hits == [], "file I/O on the scheduler thread: %s" % hits
    # no displaced anchor left L2 without a copy: every anchor is still in its slot (nothing was secured)
    assert all(h19._state(arena, h) == 2 for h in hs), [h19._state(arena, h) for h in hs]
    assert getattr(arena, "_l3wb_urgent", set()), "the anchors to give up were queued for the write-behind"


# ---------------------------------------------------------------------------------------------------- S86 before the sleep
def _s86_group(tmp_path, copied):
    """The rc19 shape (test_s86_sleep_l3_release_1020): TP0 and a worker on ONE arena, every slot COMPLETE and held by
    both trees, a park-end chain of two nodes below. ``copied`` = the pages the write-behind has copied (record AND
    index) by the time the sleep comes -- the S109 async form never writes them itself."""
    import test_s86_sleep_l3_release_1020 as S86T

    from flliper.srt.mem_cache import form_a_host_shadow as Rr
    from flliper.srt.mem_cache import l3_secured as L3S

    Rr._reset_state_for_tests()
    g = S86T._group(tmp_path)
    idx = _Idx()
    for be in (g.p0._backend, g.p1._backend):
        be._l3idx = idx
    L3S.arrays(g.a0, create=True)
    pinned = sorted({int(x) for n in g.n0 for x in g.p0.arena_ids(n.component_data[FULL].host_value).tolist()}) \
        if hasattr(g.p0, "arena_ids") else []
    stems = [g.p0._stems([f"a{i}"])[0] for i in range(S86T.SLOTS)]
    slots = [int(s) for s, _st in g.a0.find_slots(stems)]
    _write_behind(g.a0, idx, slots[:copied])
    g.c0 = S86T._chain(g.t0, g.n0[-1], ["c0", "c1"], "pdflip-112-2708")
    g.c1 = S86T._chain(g.t1, g.n1[-1], ["c0", "c1"], "pdflip-112-2708")
    return S86T, g, slots


@_needs_gcc
def test_s109_a4_rc19_the_sleep_release_still_lets_the_park_end_chain_in(tmp_path):
    """Review S109-4: S86 now refuses uncopied pages (``not_on_l3``) instead of writing them. rc19's pinned pages had
    their L3 copies (the boot freed only ``on_disk`` slots); the write-behind never pauses (S109d: 0 gate changes,
    passes during sleep). With the copies published, the chain is backed up: lost_tokens 0 -- and no file I/O."""
    S86T, g, slots = _s86_group(tmp_path, copied=8)
    with _no_file_io_in_this_thread() as hits:
        S86T._sleep_release(g)
        S86T._flush_round(g)
    assert hits == [], hits
    assert S86T._lost(g.t0) == 0


@_needs_gcc
def test_s109_a4_a_write_behind_backlog_keeps_the_uncopied_pages_and_still_lets_the_chain_in(tmp_path):
    """Half the pinned pages not yet copied: S86 keeps them (no loss) and releases the copied half -- room enough for
    the two-node chain; the uncopied ones are queued for the write-behind (woken)."""
    from flliper.srt.mem_cache import l3_secured as L3S

    S86T, g, slots = _s86_group(tmp_path, copied=4)
    L3S.WAKE.clear()
    S86T._sleep_release(g)
    S86T._flush_round(g)
    assert S86T._lost(g.t0) == 0
    kept = [n for n in g.n0 if n.component_data[FULL].host_value is not None]
    assert len(kept) >= 4, "the uncopied pages kept their host copies"
    assert L3S.WAKE.is_set(), "the request woke the write-behind (S109-A4)"


# ---------------------------------------------------------------------------------------------------- S109-A3 / A4
@_needs_gcc
def test_s109_a3_a_race_candidate_is_handed_over_not_freed(tmp_path):
    """A clock candidate that re-completed between the census and the clock (no copy): never freed on the scheduler
    thread; the hand-over writer copies it on its own thread, then frees it."""
    import time as _t

    from flliper.srt.mem_cache import l3_secured as L3S
    from flliper.srt.mem_cache.hicache_storage import HiCacheFile

    p, arena, slots = _unreferenced_full_arena(tmp_path)
    be = p._backend
    be.pins = None
    copied = []

    go = _threading.Event()

    def _writer(ar, cands, writer="claim_room"):
        go.wait(5.0)                                         # the hand-over thread waits until the test looked
        copied.append((writer, [int(c[0]) for c in cands], _threading.current_thread().name))
        return {"on_disk": 0, "written": len(cands), "lost": 0}

    be.arena_secure_to_disk = _writer
    # the race: the census saw no candidate, the clock took one anyway (an empty keep list stands in for it)
    import unittest.mock as _m

    with _m.patch.object(L3S, "unsecured_keys_lo", lambda a, b=None: np.zeros(0, dtype=np.uint64)):
        assert HiCacheFile._arena_evict_to_disk(be, arena, 4) == 0
    took = [s_ for s_ in slots if arena.slot_states([s_])[0] == 3]
    go.set()
    assert len(took) == 1, "the candidate is EVICTING, not freed"
    for _ in range(200):
        if copied and arena.slot_states(took)[0] != 3:
            break
        _t.sleep(0.01)
    assert copied and copied[0][0] == "evict_handover" and copied[0][2] == "pdflip-l3-evict-handover"
    assert arena.slot_states(took)[0] not in (2, 3), "copied, then freed by the hand-over writer"


def test_s109_a4_request_wakes_the_write_behind():
    from flliper.srt.mem_cache import l3_secured as L3S

    a = _Arena(n=4)
    stop = _threading.Event()
    L3S.WAKE.clear()
    t0 = __import__("time").monotonic()
    th = _threading.Thread(target=lambda: L3S.wait_or_stop(stop, 5.0))
    th.start()
    L3S.request(a, [1])
    th.join(2.0)
    assert not th.is_alive() and __import__("time").monotonic() - t0 < 2.0, "the pass did not wake at the request"


@_needs_gcc
def test_s109_a1_the_l3_lru_evict_voids_the_published_record(tmp_path):
    """S109-A1 preferred form: the store's LRU on_evict (``HiCacheFile._l3sec_on_evict``, wired into LRUFileEvictor)
    voids the record of every slot holding the evicted stem -- independent of the index AND."""
    import inspect

    from flliper.srt.mem_cache import l3_secured as L3S
    from flliper.srt.mem_cache.hicache_storage import HiCacheFile

    p, arena, slots = _unreferenced_full_arena(tmp_path)
    _write_behind(arena, p._backend._l3idx, slots[:2])
    yes = types.SimpleNamespace(l3_listed=lambda stems: [True] * len(stems))  # the index alone would still say yes
    assert L3S.secured(arena, slots[:2], yes).tolist() == [True, True]
    HiCacheFile._l3sec_on_evict(types.SimpleNamespace(metadata_cache=None, _arenas={"kv": arena}),
                                arena.slot_stem(slots[0]))
    assert L3S.secured(arena, slots[:2], yes).tolist() == [False, True]
    src = inspect.getsource(HiCacheFile)
    assert "self._l3sec_on_evict" in src.split("LRUFileEvictor(", 1)[1].split("pins=", 1)[0], "hook not wired"


@_needs_gcc
def test_s109_a3_a_claim_race_candidate_is_handed_over_not_freed(tmp_path):
    import time as _t
    import unittest.mock as _m

    from flliper.srt.mem_cache import l3_secured as L3S

    p, arena, slots = _unreferenced_full_arena(tmp_path)
    be = p._backend
    be.pins = None
    copied = []

    go = _threading.Event()

    def _writer(ar, cands, writer="claim_room"):
        go.wait(5.0)                                         # the hand-over thread waits until the test looked
        copied.append((writer, len(cands), _threading.current_thread().name))
        return {"on_disk": 0, "written": len(cands), "lost": 0}

    be.arena_secure_to_disk = _writer
    with _m.patch.object(L3S, "unsecured_keys_lo", lambda a, b=None: np.zeros(0, dtype=np.uint64)):
        p._evict_for_claim(arena, 2)
    st = arena.slot_states(slots)
    go.set()
    assert sum(1 for x in st if x == 3) == 2 and sum(1 for x in st if x == 0) == 0, st
    for _ in range(200):
        if copied and sum(1 for x in arena.slot_states(slots) if x == 0) == 2:
            break
        _t.sleep(0.01)
    assert copied and copied[0][0] == "evict_handover" and copied[0][2] == "pdflip-l3-evict-handover"
    assert sum(1 for x in arena.slot_states(slots) if x == 0) == 2


@_needs_gcc
def test_s109_a3_a_handover_whose_copy_fails_keeps_the_page(tmp_path):
    from flliper.srt.mem_cache import l3_secured as L3S

    p, arena, slots = _unreferenced_full_arena(tmp_path)
    be = p._backend
    seen = []

    def _writer(ar, cands, writer="x"):
        seen.append(ar.slot_states([cands[0][0]])[0])          # still EVICTING while it copies
        return {"on_disk": 0, "written": 0, "lost": 1}

    be.arena_secure_to_disk = _writer
    cands = arena.evict_candidates(1)
    with L3S._HAND_LOCK:
        L3S._HAND.extend((arena, be, c, "test") for c in cands)
    L3S._hand_drain()                                         # (or the live hand-over thread drains it first)
    for _ in range(300):
        if seen:
            break
        __import__("time").sleep(0.01)
    assert seen == [3]
    assert arena.slot_states([cands[0][0]])[0] == 3, "no copy: never freed"


@_needs_gcc
def test_s109_a1_the_census_keeps_a_page_whose_l3_file_is_gone(tmp_path):
    """The census of the uncopied (the clock's/claim's keep list) asks record AND index too: a page whose L3 file
    left the index is kept in L2 -- not even made a candidate (no hand-over, no EVICTING)."""
    from flliper.srt.mem_cache import l3_secured as L3S

    p, arena, slots = _unreferenced_full_arena(tmp_path)
    be = p._backend
    be.pins = None
    _write_behind(arena, be._l3idx, slots[:3])
    be._l3idx.files.discard(arena.slot_stem(slots[0]))       # the file is gone; the record was not voided
    q0 = L3S.HAND_STATS["queued"]
    assert p._evict_for_claim(arena, 3) == 2
    assert arena.slot_states([slots[0]])[0] == 2, "kept COMPLETE in L2, never a candidate"
    assert L3S.HAND_STATS["queued"] == q0


# ---------------------------------------------------------------------------------------------------- Re-review A5..A8
def _reuse_boot(tmp_path, monkeypatch):
    import test_l3_reuse_after_restart_0928 as RR
    from flliper.srt.mem_cache import l3_write_behind as gate

    gate._reset_for_tests()
    be = RR._boot(tmp_path, monkeypatch, "boot1")
    hs = RR._hashes("s")
    RR._prefill(be, hs)
    assert be.l3_write_behind_pass(slice_s=0)["written"] == 2 * RR.PAGES
    return RR, be, hs


@_needs_gcc
def test_s109_a5_the_l3_lru_unlink_makes_the_real_pass_copy_the_page_again(tmp_path, monkeypatch):
    """Re-review R11, the real pass: the LRU unlinks a copied page's file (its on_evict hook voids the record); the
    page is still COMPLETE in L2 -> the next pass writes it again, and only it."""
    from flliper.srt.mem_cache import l3_secured as L3S

    RR, be, hs = _reuse_boot(tmp_path, monkeypatch)
    kv = be._arenas[RR.KV_TOTAL]
    victim = be._get_suffixed_key(hs[0])
    (slot, _st), = kv.find_slots([victim])
    assert L3S.secured(kv, [slot], be).tolist() == [True]
    os.remove(be._existing_path(victim))                       # the LRU's unlink ...
    be._l3idx.remove([victim])
    be._l3sec_on_evict(victim)                                 # ... and its on_evict hook
    assert L3S.secured(kv, [slot], be).tolist() == [False]
    r = be.l3_write_behind_pass(slice_s=0)
    assert r["written"] == 1 and os.path.exists(be._existing_path(victim)), r
    assert L3S.secured(kv, [slot], be).tolist() == [True], "re-copied, published again"
    assert be.l3_write_behind_pass(slice_s=0)["written"] == 0


@_needs_gcc
def test_s109_a5_the_sister_writers_unlink_is_seen_through_the_shared_record(tmp_path, monkeypatch):
    """P and D share the arena, each writer has its own baseline: the OTHER group's evictor voids only the shared
    record. This process's baseline still says copied -- the pass must see the record and copy again."""
    from flliper.srt.mem_cache import l3_secured as L3S

    RR, be, hs = _reuse_boot(tmp_path, monkeypatch)
    kv = be._arenas[RR.KV_TOTAL]
    victim = be._get_suffixed_key(hs[1])
    (slot, _st), = kv.find_slots([victim])
    os.remove(be._existing_path(victim))
    be._l3idx.remove([victim])
    L3S.arrays(kv)[0][slot] = -1                                # the sister process's invalidate_stems
    assert kv._l3wb_sec[0][slot] >= 0, "this process's baseline is untouched"
    assert be.l3_write_behind_pass(slice_s=0)["written"] == 1


class _PromoteBe:
    def __init__(self, enabled=True):
        self._evictor = types.SimpleNamespace(enabled=enabled)
        self.calls = []
        self.ev = _threading.Event()

    def _l3_refresh_kept(self, stems, mtime=True):
        self.calls.append((list(stems), mtime, _threading.current_thread().name))
        self.ev.set()
        return len(stems)

    def l3_listed(self, stems):
        return [True] * len(stems)

    def arena_secure_to_disk(self, *a, **k):          # present (the backend's shape), never called by the async forms
        raise AssertionError("S109: no synchronous write")


@_needs_gcc
def test_s109_a6_a_release_on_an_l3_copy_promotes_it_in_the_lru_off_the_scheduler_thread(tmp_path):
    """Re-review R8 (ZERO-ANCHOR 1015): every scheduler form that gives L2 up on an L3 copy queues that copy's LRU
    promotion; the hand-over thread runs ``_l3_refresh_kept(..., mtime=False)`` -- never the calling thread."""
    from flliper.srt.mem_cache import l3_secured as L3S

    from test_w3_arena_spill_0929 import _finished

    p, arena, slots = _unreferenced_full_arena(tmp_path)
    p._backend.pins = None
    _write_behind(arena, p._backend._l3idx, slots[:4])
    assert p._evict_for_claim(arena, 2) == 2                 # room for one more page (the secure_rows form)
    rows = _finished(p, arena, ["g0"], 0x71)
    (g0, _st), = arena.find_slots(p._stems(["g0"]))
    L3S.mark(arena, [g0])
    live = [s_ for s_ in slots if arena.slot_states([s_])[0] == 2][:2]
    for form, want in (("check", live), ("verify", live), ("secure_rows", [int(g0)])):
        be = _PromoteBe()
        with _no_file_io_in_this_thread() as hits:
            if form == "check":
                L3S.check(arena, live, be)
            elif form == "verify":
                L3S.verify_candidates(arena, [(s_, 0, 0, 1) for s_ in live], be)
            else:
                p._backend = be
                _r = p.secure_rows_to_l3(rows)
                assert _r["lost"] == 0, (_r, arena.slot_states([int(g0)]), L3S.secured(arena, [int(g0)], be))
        assert hits == []
        assert be.ev.wait(3.0), form
        stems, mtime, th = be.calls[0]
        assert sorted(stems) == sorted(arena.slot_stem(s_) for s_ in want) and mtime is False, (form, be.calls)
        assert th == "pdflip-l3-evict-handover", th
    # not the LRU owner: a no-op, as before (review anchor0 B1)
    off = _PromoteBe(enabled=False)
    assert L3S.promote(arena, slots[:2], off) == 0


def test_s109_a6_the_old_writer_promoted_too():
    """The pin the async forms replace: arena_secure_to_disk promoted what it relied on (in-memory form)."""
    import inspect

    from flliper.srt.mem_cache.hicache_storage import HiCacheFile

    assert "_l3_refresh_kept(_relied, mtime=False)" in inspect.getsource(HiCacheFile.arena_secure_to_disk)


@_needs_gcc
def test_s109_a7_the_quiesce_entrance_queues_every_unsecured_page_and_wakes_the_write_behind(tmp_path):
    from flliper.srt.managers import pdflip_bubble_publish as BP
    from flliper.srt.mem_cache import l3_secured as L3S

    p, arena, slots = _unreferenced_full_arena(tmp_path)
    be = p._backend
    L3S.mark(arena, slots[:2])
    _write_behind(arena, be._l3idx, slots[:2])
    saved = list(L3S._REG)
    try:
        L3S._REG[:] = []
        be._l3wb_thread = None
        L3S.register_backend(arena, be)
        sched = types.SimpleNamespace()
        L3S.WAKE.clear()
        BP.quiesce_begin(sched)
        assert not getattr(arena, "_l3wb_urgent", set()), "no write-behind on this rank: nothing queued"
        BP.quiesce_end(sched, "t")
        be._l3wb_thread = object()                               # the store's LRU owner (D TP0 / P PP0)
        with _no_file_io_in_this_thread() as hits:
            BP.quiesce_begin(sched)
        assert hits == []
        assert set(arena._l3wb_urgent) == set(slots[2:]) and L3S.WAKE.is_set()
        arena._l3wb_urgent.clear()
        BP.quiesce_begin(sched)                                  # still open: once per quiesce
        assert not arena._l3wb_urgent
    finally:
        L3S._REG[:] = saved


@_needs_gcc
def test_s109_a8_a_reaped_and_reclaimed_slot_is_never_copied_and_counts_as_dropped(tmp_path):
    from flliper.srt.mem_cache import l3_secured as L3S
    from flliper.srt.mem_cache.pool_host.arena_pool import ArenaMHAHostPool

    p, arena, slots = _unreferenced_full_arena(tmp_path)
    be = p._backend
    copied = []
    be.arena_secure_to_disk = lambda ar, cands, writer="x": copied.append(cands) or {"written": 1, "lost": 0}
    cands = arena.evict_candidates(1)
    s0 = cands[0][0]
    d0 = getattr(ArenaMHAHostPool, "_257_dropped_without_l3", 0)
    h0 = L3S.HAND_STATS["dropped_without_l3"]
    # the hand-over is late: the aged reaper freed the slot and a new claim filled it (new generation)
    import flliper.srt.mem_cache.storage.file.hicache_arena as HA

    gen = HA._evict_gens_for(arena.path).get(s0)
    arena.free_slots([s0])
    with L3S._HAND_LOCK:
        L3S._HAND.append((arena, be, tuple(cands[0]), "test", gen))
    L3S._hand_drain()
    assert copied == [], "no bytes copied under the old stem"
    assert L3S.HAND_STATS["dropped_without_l3"] == h0 + 1
    assert ArenaMHAHostPool._257_dropped_without_l3 == d0 + 1


@_needs_gcc
def test_s109_a8_a_failed_hand_over_copy_counts_as_dropped(tmp_path):
    from flliper.srt.mem_cache import l3_secured as L3S

    p, arena, slots = _unreferenced_full_arena(tmp_path)
    be = p._backend
    be.arena_secure_to_disk = lambda ar, cands, writer="x": {"written": 0, "lost": 1}
    cands = arena.evict_candidates(1)
    import flliper.srt.mem_cache.storage.file.hicache_arena as HA

    gen = HA._evict_gens_for(arena.path).get(cands[0][0])
    h0 = L3S.HAND_STATS["dropped_without_l3"]
    with L3S._HAND_LOCK:
        L3S._HAND.append((arena, be, tuple(cands[0]), "test", gen))
    L3S._hand_drain()
    assert L3S.HAND_STATS["dropped_without_l3"] == h0 + 1
    assert arena.slot_states([cands[0][0]])[0] == 3


# ---------------------------------------------------------------------------------------------------- re-review 2: N1/N2/M
@_needs_gcc
def test_s109_a8n1_the_real_aged_reaper_cannot_take_the_slot_while_the_hand_over_copies(tmp_path):
    """The hand-over stamps its candidate (arena_touch_evicting) before the check: the real aged reaper, firing while
    the copy runs, backs off -- the copy is of the candidate, the hand-over frees it. (Without the stamp the slot,
    EVICTING longer than the reaper's age, is reaped mid-copy.)"""
    import time as _t

    from flliper.srt.mem_cache import l3_secured as L3S
    import flliper.srt.mem_cache.storage.file.hicache_arena as HA

    p, arena, slots = _unreferenced_full_arena(tmp_path)
    be = p._backend
    cands = arena.evict_candidates(1)
    s0 = cands[0][0]
    gen = HA._evict_gens_for(arena.path).get(s0)
    _t.sleep(0.25)                                           # EVICTING older than the reaper's age below
    seen, reaped = [], []

    def _copy(ar, cs, writer="x"):
        reaped.append(int(ar._lib.arena_reap_stale_aged(ar._base, 200)))   # the reaper fires mid-copy
        seen.append((int(ar.slot_states([s0])[0]), int(np.asarray(ar.slot_gens_np([s0]))[0])))
        return {"written": 1, "lost": 0}

    be.arena_secure_to_disk = _copy
    f0 = L3S.HAND_STATS["freed"]
    with L3S._HAND_LOCK:
        L3S._HAND.append((arena, be, tuple(cands[0]), "test", gen))
    L3S._hand_drain()
    assert reaped == [0] and seen == [(3, gen)], (reaped, seen)
    assert arena.slot_states([s0])[0] == 0 and L3S.HAND_STATS["freed"] == f0 + 1


@_needs_gcc
def test_s109_a8n2_freed_counts_only_a_free_that_happened(tmp_path):
    from flliper.srt.mem_cache import l3_secured as L3S
    import flliper.srt.mem_cache.storage.file.hicache_arena as HA

    p, arena, slots = _unreferenced_full_arena(tmp_path)
    be = p._backend
    be.arena_secure_to_disk = lambda ar, cs, writer="x": {"written": 1, "lost": 0}
    cands = arena.evict_candidates(1)
    gen = HA._evict_gens_for(arena.path).get(cands[0][0])
    arena.free_evicted = lambda sl, reason="": 0           # the generation-bound free refuses
    f0, r0 = L3S.HAND_STATS["freed"], L3S.HAND_STATS["free_refused"]
    with L3S._HAND_LOCK:
        L3S._HAND.append((arena, be, tuple(cands[0]), "test", gen))
    L3S._hand_drain()
    assert L3S.HAND_STATS["freed"] == f0 and L3S.HAND_STATS["free_refused"] == r0 + 1


@_needs_gcc
def test_s109_a5m_the_lru_taking_an_l2_page_is_counted(tmp_path):
    from flliper.srt.mem_cache import l3_secured as L3S
    from flliper.srt.mem_cache.hicache_storage import HiCacheFile

    p, arena, slots = _unreferenced_full_arena(tmp_path)
    _write_behind(arena, p._backend._l3idx, slots[:2])
    n0, b0 = L3S.LRU_L2_STATS["n"], L3S.LRU_L2_STATS["bytes"]
    fake = types.SimpleNamespace(metadata_cache=None, _arenas={"kv": arena})
    HiCacheFile._l3sec_on_evict(fake, arena.slot_stem(slots[0]))
    HiCacheFile._l3sec_on_evict(fake, "not-in-l2")
    assert L3S.LRU_L2_STATS["n"] == n0 + 1 and L3S.LRU_L2_STATS["bytes"] == b0 + int(arena.slot_bytes)


@_needs_gcc
def test_s109_a6m_promote_stats_are_logged_with_missed(tmp_path):
    from flliper.srt.mem_cache import l3_secured as L3S

    p, arena, slots = _unreferenced_full_arena(tmp_path)
    be = _PromoteBe()
    be._evictor._lru = {arena.slot_stem(slots[0]): 1}       # slot 1's file left the LRU before the drain
    m0 = L3S.PROMOTE_STATS["missed"]
    with L3S._HAND_LOCK:
        L3S._PROMOTE[id(be)] = (be, {arena.slot_stem(slots[0]): None, arena.slot_stem(slots[1]): None})
    L3S._promote_drain()
    assert L3S.PROMOTE_STATS["missed"] == m0 + 1
    line = L3S.stats_line(force=True)
    assert line.startswith("S109 L3-STATS promote queued=") and " missed=%d " % L3S.PROMOTE_STATS["missed"] in line
    assert L3S.stats_line() is None, "throttled: nothing moved, under STATS_LOG_S"
