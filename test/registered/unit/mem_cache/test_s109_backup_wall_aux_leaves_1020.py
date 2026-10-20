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
import torch  # noqa: E402

from flliper.srt.mem_cache import form_a_host_shadow as R12  # noqa: E402
from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType  # noqa: E402
from flliper.srt.pdflip import host_only_spill as HOS  # noqa: E402

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
    """The store: ``disk`` = anchor stems already in L3; ``secure`` writes (or loses) the rest."""

    def __init__(self, disk=(), lose=()):
        self.disk, self.lose, self.written = set(disk), set(lose), []

    def _stat_stems(self, stems):
        return [s for s in stems if s in self.disk]

    def arena_secure_to_disk(self, arena, cands, writer="claim_room"):
        out = {"on_disk": 0, "written": 0, "lost": 0}
        for slot, *_ in cands:
            if int(slot) in self.lose:
                out["lost"] += 1
            else:
                self.written.append(int(slot))
                self.disk.add("a%d" % int(slot))
                out["written"] += 1
        return out


class _MambaPool:
    def __init__(self, backend, settled=True):
        self._backend = backend
        self.arena = types.SimpleNamespace(slot_bytes=59 * 1024 * 1024, slot_stem=lambda s: "a%d" % int(s))
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


def _shape(backend=None, settled=True, aux_ct=MAMBA):
    """A plain host-only leaf (8 pages) and an aux leaf (16 pages, host-only mamba anchor in slot 7)."""
    t = _Tree(_MambaPool(backend or _Backend(disk={"a7"}), settled=settled))
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


def test_an_anchor_already_in_l3_costs_a_stat_no_write():
    b = _Backend(disk={"a7"})
    t, plain, aux = _shape(b)
    HOS.spill_host_only(t, _KvPool(), 24, P, set())
    assert b.written == [] and aux.id in t.evicted_order


def test_an_anchor_not_in_l3_is_written_first_then_the_leaf_goes():
    b = _Backend(disk=())
    t, plain, aux = _shape(b)
    HOS.spill_host_only(t, _KvPool(), 24, P, set())
    assert b.written == [7] and aux.id in t.evicted_order


def test_an_anchor_that_cannot_be_secured_keeps_the_leaf_named(caplog):
    b = _Backend(disk=(), lose={7})
    t, plain, aux = _shape(b)
    with caplog.at_level("INFO"):
        res = HOS.spill_host_only(t, _KvPool(), 24, P, set())
    assert aux.id not in t.evicted_order and res["released"] == 8
    assert aux.component_data[MAMBA].host_value is not None and aux.component_data[FULL].host_value is not None
    assert any("aux_kept=child_anchor_lost:1" in m for m in caplog.messages)


def test_an_anchor_in_flight_keeps_the_leaf():
    t, plain, aux = _shape(settled=False)
    res = HOS.spill_host_only(t, _KvPool(), 24, P, set())
    assert aux.id not in t.evicted_order and res["released"] == 8


def test_a_kv_page_without_l3_copy_keeps_the_aux_leaf():
    t, plain, aux = _shape()
    res = HOS.spill_host_only(t, _KvPool(lose={200}), 24, P, set())
    assert aux.id not in t.evicted_order and res["released"] == 8


def test_the_anchor_writes_per_call_are_bounded():
    b = _Backend(disk=())
    t = _Tree(_MambaPool(b))
    leaves = [_Node(t.root_node, 4, anchor_slot=10 + i, start=300 + 10 * i) for i in range(HOS.ANCHOR_SECURE_PER_CALL + 2)]
    res = HOS.spill_host_only(t, _KvPool(), 10 ** 6, P, set())
    assert len(b.written) == HOS.ANCHOR_SECURE_PER_CALL and res["aux_leaves"] == HOS.ANCHOR_SECURE_PER_CALL
    assert sum(1 for n in leaves if n.id in t.evicted_order) == HOS.ANCHOR_SECURE_PER_CALL


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
    t._mp._backend.disk |= {"a5", "a6"}
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
    t0._mp._backend.disk |= {"a1", "a2"}
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
