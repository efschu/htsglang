"""M1s rc12z30j (bb82fbcb68, D-only, #239 token cut [0,32,32]/64): D died at
01:06:34 on TP0 with ``Bar1CollectiveAborted`` in an eager extend's MoE
all-reduce. All three ranks entered that extend at 01:04:50 (``#969 EXTENT
n=62``); TP0 evicted by write-back and went on to the forward, TP1/TP2 never
arrived: from 01:04:50 on they printed only ``#969H BACKUP`` / ``#1421
BACKUP-REFUSED why=mamba_pool_unbound`` / ``R12 SHADOW-SHORT`` (16189 lines,
up to 5700/min) over the same node ids.

Two faults, one per test group:

1. Under the cut a Form A worker's KV host pool is an ``ArenaMHAHostPool``
   (its rows are the page's bytes) while its mamba host pool stays the plain
   byteless R12 shadow pool (``mambahost=MambaPoolHost``, ``R12
   SHADOW-CAPACITY mamba anchor rows +64 ... 0 B/row``). The direct write
   path demanded a mamba ARENA slot for the anchor, found none and refused
   the whole node -- every backup of a worker, the eviction's included.
2. ``MambaComponent.drive_eviction`` restarts at the LRU end whenever the
   walk's successor is not in the list -- which is also the case at the list
   head (``get_prev_no_lock`` -> None). With every leaf refused nothing ever
   left the list: an endless walk inside the scheduler step.

Hermetic: real ``UnifiedTreeNode``/``UnifiedLRUList`` and the real
``_evict_device_leaf`` -> ``write_backup`` -> ``_weg2_direct_claim`` chain;
pools, controller and components are recording fakes."""
from __future__ import annotations

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from sglang.srt import rank_role  # noqa: E402
from sglang.srt.mem_cache.base_prefix_cache import EvictParams  # noqa: E402
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer  # noqa: E402
from sglang.srt.mem_cache.unified_cache_components.mamba_component import MambaComponent  # noqa: E402
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType  # noqa: E402
from sglang.srt.mem_cache.unified_radix_cache import (  # noqa: E402
    UnifiedLRUList,
    UnifiedRadixCache,
    UnifiedTreeNode,
)

TC = (ComponentType.FULL, ComponentType.MAMBA)
F, M = ComponentType.FULL, ComponentType.MAMBA
PAGE = 64
PAGES = 4


class _KvArena:
    arena_read = True

    def __init__(self):
        self.aborted = []

    def alloc_write(self, hashes):
        return torch.arange(1000, 1000 + len(hashes) * PAGE)

    def abort_write(self, rows):
        self.aborted.append(rows)


class _ShadowMambaHost:
    """The worker's plain byteless mamba host pool: no ``arena_resolve_reads``."""

    def __init__(self, free):
        self.free_rows = free

    def available_size(self):
        return self.free_rows


class _HostGroup:
    arena_read = True

    def __init__(self, mamba):
        self.entry_map = {PoolName.KV: None, PoolName.MAMBA: None}
        self._mamba = mamba

    def get_pool(self, name):
        return self._mamba if name == PoolName.MAMBA else None


class _Comp:
    def __init__(self, ct):
        self.component_type = ct
        self.commits = []

    def build_hicache_transfers(self, node, phase):
        cd = node.component_data[self.component_type]
        if self.component_type == M and cd.value is not None:
            return [PoolTransfer(name=PoolName.MAMBA, device_indices=cd.value)]
        return None

    def commit_hicache_transfer(self, node, phase, transfers=()):
        self.commits.append(list(transfers))
        if transfers and transfers[0].host_indices is not None:
            node.component_data[self.component_type].host_value = transfers[0].host_indices.clone()


def _tree(shadow_free=64):
    t = object.__new__(UnifiedRadixCache)
    t.root_node = UnifiedTreeNode(TC)
    t.page_size = PAGE
    t.sidecar_pool_specs = []
    t.ongoing_write_through = {}
    t._weg2_rid_anchor_cfg = 8
    t.components = {F: _Comp(F), M: _Comp(M)}
    t._components_tuple = (t.components[F], t.components[M])
    t.kv = _KvArena()
    t._weg2_direct_pool = lambda: t.kv
    t._weg2_mamba_pool = lambda: None  # the worker has no mamba arena
    t.writes = []

    def _write(device_value, node_id=None, extra_pools=None, host_indices=None):
        t.writes.append([(x.name, x.host_indices) for x in (extra_pools or [])])
        return host_indices

    t.cache_controller = types.SimpleNamespace(
        write_policy="write_back", write=_write,
        mem_pool_host=_HostGroup(_ShadowMambaHost(shadow_free)), mem_pool_host_draft=None,
    )
    t.demoted = []
    t.writing_check = lambda write_back=False: None

    def _to_host(node, tracker):
        t.demoted.append(node)
        tracker[F] = tracker.get(F, 0) + len(node.key)

    t._evict_to_host = _to_host
    return t


def _leaf(t, name="tail"):
    n = UnifiedTreeNode(TC)
    n.key = [0] * (PAGE * PAGES)
    n.parent = t.root_node
    t.root_node.children[name] = n
    n.hash_value = [f"{name}{i}" for i in range(PAGES)]
    n.component_data[F].value = torch.arange(PAGE * PAGES)
    n.component_data[M].value = torch.tensor([7])
    n.weg2_anchor_rid = None
    n._weg2_end_anchor = False
    return n


@pytest.fixture
def cut_worker(monkeypatch):
    """TP1/TP2 of the M1s form: a KV-only rank (#239 cut, share > 0)."""
    monkeypatch.setattr(rank_role, "kv_only_rank", lambda: True)


# ---------------------------------------------------------------- fault 1


def test_m1s_shape_the_cut_worker_backs_its_node_up(cut_worker):
    """RED on bb82fbcb68: '#1421 BACKUP-REFUSED why=mamba_pool_unbound', 0."""
    t = _tree()
    n = _leaf(t)
    assert t.write_backup(n, write_back=True) == PAGE * PAGES
    assert n.backuped
    assert t.kv.aborted == [], "the granted KV claim stands"


def test_the_anchor_takes_a_shadow_row_not_an_arena_slot(cut_worker):
    """The mamba transfer reaches the controller WITHOUT host rows -- the
    controller allocates them from the plain shadow pool (staging path)."""
    t = _tree()
    t.write_backup(_leaf(t), write_back=True)
    assert t.writes == [[(PoolName.MAMBA, None)]]


def test_m1s_shape_the_eviction_frees_the_worker_leaf(cut_worker):
    """RED on bb82fbcb68: the refusal left the leaf on the device, freed 0."""
    t = _tree()
    n = _leaf(t)
    tracker = {F: 0, M: 0}
    t._evict_device_leaf(n, tracker)
    assert t.demoted == [n]
    assert tracker[F] == PAGE * PAGES


def test_a_full_shadow_pool_under_eviction_goes_kv_only(cut_worker):
    """The worker's anchor is bookkeeping; with no shadow row left the
    eviction still frees the KV (P-FUND form), it never spins."""
    t = _tree(shadow_free=0)
    n = _leaf(t)
    tracker = {F: 0, M: 0}
    t._evict_device_leaf(n, tracker)
    assert t.demoted == [n]
    assert t.writes == [[]], "no mamba transfer on a KV-only backup"


def test_the_weight_rank_without_a_mamba_arena_still_refuses(monkeypatch):
    """Not a KV-only rank (TP0 / a classic rank): an unbound mamba arena is
    still a named refusal, as before."""
    monkeypatch.setattr(rank_role, "kv_only_rank", lambda: False)
    t = _tree()
    n = _leaf(t)
    assert t.write_backup(n, write_back=True) == 0
    assert t._weg2_sweep_last_refusal == "mamba_pool_unbound"
    assert t.kv.aborted, "the KV claim goes back with the refused node"


# ---------------------------------------------------------------- fault 2


class _LeafCache:
    """Every device leaf refuses its eviction (the M1s workers)."""

    def __init__(self, nodes, frees=()):
        self.lru_lists = {M: UnifiedLRUList(M, TC)}
        for n in nodes:
            self.lru_lists[M].insert_mru(n)
        self.evictable_device_leaves = set(nodes)
        self.frees = set(frees)
        self.calls = 0

    def _evict_device_leaf(self, x, tracker):
        self.calls += 1
        if self.calls > 1000:
            raise RuntimeError("eviction walk livelocked (M1s rc12z30j)")
        if x in self.frees:
            self.frees.discard(x)
            self.lru_lists[M].remove_node(x)
            self.evictable_device_leaves.discard(x)
            tracker[M] += 1


def _nodes(k):
    out = []
    for _ in range(k):
        n = UnifiedTreeNode(TC)
        n.component_data[M].value = torch.tensor([1])
        out.append(n)
    return out


def _mamba(cache):
    comp = object.__new__(MambaComponent)
    comp.cache = cache
    return comp


def test_m1s_shape_a_round_of_refusals_ends_the_walk():
    """RED on bb82fbcb68: RuntimeError from the 1000-call guard (endless)."""
    nodes = _nodes(3)
    cache = _LeafCache(nodes)
    tracker = {F: 0, M: 0}
    _mamba(cache).drive_eviction(EvictParams(mamba_num=2), tracker)
    assert tracker[M] == 0
    assert cache.calls == 3, "one round over the LRU, then stop"


def test_a_round_that_freed_something_gets_another_round():
    """A freeing round still restarts at the LRU end (unchanged behaviour)."""
    nodes = _nodes(3)
    cache = _LeafCache(nodes, frees=[nodes[0]])
    tracker = {F: 0, M: 0}
    _mamba(cache).drive_eviction(EvictParams(mamba_num=2), tracker)
    assert tracker[M] == 1
    assert cache.calls == 3 + 2, "round 1 frees one, round 2 frees none, stop"
