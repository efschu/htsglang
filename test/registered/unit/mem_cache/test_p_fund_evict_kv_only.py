"""P-FUND (rc12k 10:51:00, PP0, pdflip-10-53 16384 @32768): what the admission
counts as fundable, the eviction must be able to pay.

The chunked gate (schedule_policy -> common.fundable_extend_tokens ->
chunk_tokens_the_pool_can_fund) granted 16384 on ``available 3328 +
evictable 226048``. The eviction in front of the allocation freed nothing:
under P's ``write_back`` policy every un-backed device leaf has to be written
back first, and with the mamba host arena full (32 slots, ``MAMBA-ARENA
end_anchor=refused`` for pdflip-8-43/8-44/9-46/9-47 right before the raise) the
arena claim for each leaf's mamba ANCHOR was refused -- which refused the
whole node, KV rows included -- ``_evict_device_leaf`` returned without
freeing, and ``Prefill out of memory ... 16384 tokens`` killed the scheduler.

Fix: under eviction a refused mamba claim drops only the anchor; the node's
KV goes to its (granted) arena rows and the device rows are freed.

Hermetic: the real ``_evict_device_leaf`` -> ``write_backup`` ->
``_pdflip_direct_claim`` -> ``_pdflip_mamba_claim`` chain on a real
``UnifiedTreeNode``; pools, controller and components are recording fakes."""
from __future__ import annotations

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from flliper.srt.mem_cache import common  # noqa: E402
from flliper.srt.mem_cache.hicache_storage import PoolName, PoolTransfer  # noqa: E402
from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType  # noqa: E402
from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache, UnifiedTreeNode  # noqa: E402

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


class _MambaArena:
    """Arena with every slot taken: each claim is refused."""
    arena_slots = 32

    def __init__(self):
        self.claims = 0

    def alloc_write(self, hashes):
        self.claims += 1
        return None

    def abort_write(self, rows):
        pass


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


def _tree():
    t = object.__new__(UnifiedRadixCache)
    t.root_node = UnifiedTreeNode(TC)
    t.page_size = PAGE
    t.sidecar_pool_specs = []
    t.ongoing_write_through = {}
    t._pdflip_rid_anchor_cfg = 8
    t.components = {F: _Comp(F), M: _Comp(M)}
    t._components_tuple = (t.components[F], t.components[M])
    t.kv, t.mamba = _KvArena(), _MambaArena()
    t._pdflip_direct_pool = lambda: t.kv
    t._pdflip_mamba_pool = lambda: t.mamba
    t.writes = []

    def _write(device_value, node_id=None, extra_pools=None, host_indices=None):
        t.writes.append([x.name for x in (extra_pools or [])])
        return host_indices

    t.cache_controller = types.SimpleNamespace(
        write_policy="write_back", write=_write,
        mem_pool_host=types.SimpleNamespace(arena_read=True), mem_pool_host_draft=None,
    )
    t.demoted = []
    t.writing_check = lambda write_back=False: None

    def _to_host(node, tracker):
        t.demoted.append(node)
        tracker[F] = tracker.get(F, 0) + len(node.key)

    t._evict_to_host = _to_host
    return t


def _leaf(t, rid=None, end_anchor=False):
    """A finished request's tail node: KV and a mamba anchor on the device,
    nothing on the host (P never got the anchor into the arena)."""
    n = UnifiedTreeNode(TC)
    n.key = [0] * (PAGE * PAGES)
    n.parent = t.root_node
    t.root_node.children["tail"] = n
    n.hash_value = [f"h{i}" for i in range(PAGES)]
    n.component_data[F].value = torch.arange(PAGE * PAGES)
    n.component_data[M].value = torch.tensor([7])
    n.pdflip_anchor_rid = rid
    n._pdflip_end_anchor = end_anchor
    return n


def test_rc12k_shape_the_eviction_frees_the_leaf_whose_anchor_has_no_slot():
    """RED on dfce479f08: the refused anchor refused the node, tracker 0."""
    t = _tree()
    n = _leaf(t)
    tracker = {F: 0, M: 0}
    t._evict_device_leaf(n, tracker)
    assert t.mamba.claims >= 1, "the mamba arena was asked (and refused)"
    assert t.demoted == [n], "the leaf must go down to the host tier"
    assert tracker[F] == PAGE * PAGES
    assert n.backuped, "its KV has the arena rows the KV claim granted"
    assert t.kv.aborted == [], "the granted KV claim must not be aborted"


def test_the_kv_only_backup_carries_no_mamba_transfer():
    t = _tree()
    n = _leaf(t)
    t._evict_device_leaf(n, {F: 0, M: 0})
    assert t.writes == [[]], "the controller write must not carry the refused anchor"
    assert t.components[M].commits == [], "no mamba host value is committed"
    assert n.component_data[M].host_value is None
    assert not getattr(t, "_pdflip_direct_mamba_rows", {}), "no pending mamba rows"


def test_the_publish_sweep_still_refuses_the_whole_node():
    """Only the eviction trades the anchor for the KV: the write-through
    sweeps keep refusing (and stop on a full arena) as before."""
    t = _tree()
    n = _leaf(t)
    assert t.write_backup(n, write_back=True) == 0
    assert t.kv.aborted, "the KV claim is returned when the node is refused"
    assert t._pdflip_sweep_last_refusal == "mamba_claim"
    assert not n.backuped


def test_a_granted_anchor_still_travels_with_the_node():
    t = _tree()
    t.mamba.alloc_write = lambda hashes: torch.tensor([3])
    n = _leaf(t)
    t._evict_device_leaf(n, {F: 0, M: 0})
    assert t.writes == [[PoolName.MAMBA]]
    assert int(n.component_data[M].host_value[0]) == 3
    assert t.demoted == [n]


def test_the_extend_oom_names_the_undelivered_eviction():
    """Instrument: rc12k's message said 'evictable 226048' beside 'failed to
    allocate 16384' and nothing about the eviction in between."""
    alloc = types.SimpleNamespace(
        page_size=PAGE,
        available_size=lambda: 3328,
        alloc_extend=lambda *a, **k: None,
        free_group=None,
    )
    tree = types.SimpleNamespace(
        token_to_kv_pool_allocator=alloc,
        is_chunk_cache=lambda: False,
        uniform_avail_floor=None,
        evict=lambda params: types.SimpleNamespace(num_tokens_evicted=0),
        full_evictable_size=lambda: 226048,
        evictable_size=lambda: 226048,
        pretty_print=lambda: None,
        available_and_evictable_str=lambda: "Available full tokens: 229376\n",
    )
    seq = torch.tensor([49152])
    try:
        common.alloc_paged_token_slots_extend(
            tree, torch.tensor([32768]), [32768], seq, [49152],
            torch.tensor([0]), 16384,
        )
    except RuntimeError as e:
        msg = str(e)
    else:  # pragma: no cover
        raise AssertionError("allocation must fail")
    assert "EVICTION UNDER-DELIVERED" in msg
    assert "the pool received 0" in msg
