"""y6h-kvh (01.10. 15:27:56Z, f20011a9fd, D token vector 64,0,0 = Form A
dcp 1, the whole D KV on TP0): TP1/TP2 died in ``alloc_req_slots`` with
``mamba_available=2, mamba_total=38`` while TP0 sat at mamba usage 0.57.

On the workers the eviction chain was ``MAMBA-EVICT NO-PROGRESS`` (every
device leaf refused) behind ``#1421 BACKUP-REFUSED why=write_none:?`` and
``R12 SHADOW-OWN-EVICT-REFUSED comp=MAMBA``. A worker that owns no KV rows
has no KV arena (``store=FormAWorkerNullStorage``), so ``_pdflip_direct_claim``
returns None and the backup takes the staging path; the controller needs a
row of the byteless R12 shadow mamba pool for the anchor, the pool is full,
R12 refuses the worker's own host eviction, ``write`` returns None and the
leaf stays on the device. The M1s rule (a full shadow pool under eviction
goes KV-only) existed only on the direct path, i.e. only for a worker that
owns KV rows (#239 cut) -- never for dcp 1.

Hermetic: real ``UnifiedTreeNode`` and the real ``_evict_device_leaf`` ->
``write_backup`` chain; controller, pools and components are fakes."""
from __future__ import annotations

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.mem_cache import form_a_host_shadow as r12  # noqa: E402
from flliper.srt.mem_cache import unified_radix_cache as urc  # noqa: E402
from flliper.srt.mem_cache.hicache_storage import PoolName, PoolTransfer  # noqa: E402
from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType  # noqa: E402
from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache, UnifiedTreeNode  # noqa: E402

TC = (ComponentType.FULL, ComponentType.MAMBA)
F, M = ComponentType.FULL, ComponentType.MAMBA
PAGE = 64
PAGES = 4


class _ShadowMambaHost:
    """The worker's plain byteless mamba host pool (no ``arena_resolve_reads``)."""

    def __init__(self, free):
        self.free_rows = free

    def available_size(self):
        return self.free_rows


class _HostGroup:
    def __init__(self, mamba):
        self.entry_map = {PoolName.KV: None, PoolName.MAMBA: None}
        self._mamba = mamba

    def get_pool(self, name):
        return self._mamba if name == PoolName.MAMBA else None


class _Comp:
    def __init__(self, ct):
        self.component_type = ct

    def build_hicache_transfers(self, node, phase):
        cd = node.component_data[self.component_type]
        if self.component_type == M and cd.value is not None:
            return [PoolTransfer(name=PoolName.MAMBA, device_indices=cd.value)]
        return None

    def commit_hicache_transfer(self, node, phase, transfers=()):
        if transfers and transfers[0].host_indices is not None:
            node.component_data[self.component_type].host_value = transfers[0].host_indices.clone()


def _tree(shadow_free):
    t = object.__new__(UnifiedRadixCache)
    t.root_node = UnifiedTreeNode(TC)
    t.page_size = PAGE
    t.sidecar_pool_specs = []
    t.ongoing_write_through = {}
    t.staging_write_ring = None
    t._pdflip_rid_anchor_cfg = 8
    t.components = {F: _Comp(F), M: _Comp(M)}
    t._components_tuple = (t.components[F], t.components[M])
    t._pdflip_direct_pool = lambda: None  # FormAWorkerNullStorage: no KV arena
    shadow = _ShadowMambaHost(shadow_free)
    t.writes = []

    def _write(device_value, node_id=None, extra_pools=None, host_indices=None):
        # HybridCacheController.write: the anchor's shadow row is allocated
        # here; a full shadow pool (host eviction refused by R12) -> None.
        t.writes.append([x.name for x in (extra_pools or [])])
        for x in extra_pools or []:
            if x.name == PoolName.MAMBA and x.host_indices is None:
                if shadow.available_size() < len(x.device_indices):
                    return None
                x.host_indices = torch.tensor([3])
        return torch.arange(len(device_value))

    t.cache_controller = types.SimpleNamespace(
        write_policy="write_back", write=_write,
        mem_pool_host=_HostGroup(shadow), mem_pool_host_draft=None,
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
    n.pdflip_anchor_rid = None
    n._pdflip_end_anchor = False
    return n


@pytest.fixture
def byteless_worker(monkeypatch):
    """TP1/TP2 under --d-kv-token-cut 64,0,0: an R12 worker with no KV rows."""
    monkeypatch.setattr(r12, "role", lambda: "worker")
    monkeypatch.setattr(urc, "uniform_host_avail_for_backup", lambda tree, pool: 1 << 30)
    monkeypatch.setattr(urc, "uniform_host_floor_active", lambda tree: False)
    monkeypatch.setattr(urc, "note_uniform_host_admitted", lambda tree, n: None)
    monkeypatch.setattr(UnifiedRadixCache, "_mamba_write_through_pin_admissible",
                        lambda self, node, write_back=False: True)


def test_kvh_shape_full_shadow_pool_eviction_frees_the_leaf(byteless_worker):
    """RED on f20011a9fd: write_backup -> 0 ('write_none:?'), leaf stays,
    freed 0 -- the 'MAMBA-EVICT NO-PROGRESS' of TP1/TP2 at 15:27:54-56Z."""
    t = _tree(shadow_free=0)
    n = _leaf(t)
    tracker = {F: 0, M: 0}
    t._evict_device_leaf(n, tracker)
    assert t.demoted == [n]
    assert tracker[F] == PAGE * PAGES
    assert t.writes == [[]], "the backup went KV-only: no mamba transfer"


def test_with_shadow_room_the_anchor_is_still_backed_up(byteless_worker):
    """Room in the shadow pool: unchanged, the anchor takes a shadow row."""
    t = _tree(shadow_free=8)
    n = _leaf(t)
    tracker = {F: 0, M: 0}
    t._evict_device_leaf(n, tracker)
    assert t.demoted == [n]
    assert t.writes == [[PoolName.MAMBA]]
    assert n.component_data[M].host_value is not None


def test_a_plain_backup_outside_eviction_still_refuses(byteless_worker):
    """Only the eviction (kv_only_if_mamba_refused) drops the anchor; an
    ordinary write-back backup with a full shadow pool stays a refusal."""
    t = _tree(shadow_free=0)
    assert t.write_backup(_leaf(t), write_back=True) == 0


def test_tp0_and_classic_ranks_are_untouched(byteless_worker, monkeypatch):
    """Not an R12 worker: the staging path behaves exactly as before."""
    monkeypatch.setattr(r12, "role", lambda: "host")
    t = _tree(shadow_free=0)
    n = _leaf(t)
    tracker = {F: 0, M: 0}
    t._evict_device_leaf(n, tracker)
    assert t.demoted == []
    assert tracker[F] == 0
