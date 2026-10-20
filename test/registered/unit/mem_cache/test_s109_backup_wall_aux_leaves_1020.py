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
    assert src.count("_l3s.publish(arena,") == 3, "on disk, written, and written by the sibling group: all published"
    i_take = src.index("_l3s.take_urgent(arena)")
    i_cur = src.index("new = new[int(np.searchsorted(slots[new], cur)):]")
    i_cat = src.index("new = np.concatenate([_urg_new, new])")
    assert i_take < i_cur < i_cat
