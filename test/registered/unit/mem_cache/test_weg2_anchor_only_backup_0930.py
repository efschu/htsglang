"""ANCHOR-ONLY BACKUP (NF y5a 30.09.): 16x ``WEG2-ANCHOR-LOST at=flush`` on D.

``publish_unbacked_sweep`` skipped every node whose KV was already backed
(``backuped`` or ``l3_present``) -- and a node whose Mamba anchor came LATER
(an anchor given to a node that held only a host copy,
``commit_insert_component_data``) had its anchor on the device only. The flush's
``weg2_unbacked_anchors`` probe named it (e.g. 17:47:42 ``depths=[42880]``,
next to ``#1470 FLUSH-PUBLISH ... unbacked_left=0``) and the reset dropped it.

Now the sweep copies ONLY the anchor D->H into the Mamba arena (the KV write is
empty -- the KV is already backed): claim, controller write, commit the host
value, pin and track like a write-through; the ack completes the anchor's arena
slot only (no second KV completion, no KV store write). After the sweep the
probe is empty. Switch SGLANG_WEG2_ANCHOR_ONLY_BACKUP (default on).
Real UnifiedRadixCache over a real HybridReqToTokenPool; the host tier
(controller, Mamba arena) is a recording double.
"""
from __future__ import annotations

import importlib.util
import os
from array import array
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.mem_cache.base_prefix_cache import InsertParams  # noqa: E402
from sglang.srt.mem_cache.radix_cache import RadixKey  # noqa: E402
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType  # noqa: E402
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_t_compact_ao", os.path.join(os.path.dirname(__file__), "..", "weg2", "test_weg2_d_seat_compact_0930.py"))
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)
MC = ComponentType.MAMBA
KV = ComponentType.FULL


class _Arena:
    """The Mamba arena pool: claim, abort, complete (recorded)."""

    def __init__(self, room=True):
        self.room, self.claimed, self.aborted, self.completed = room, [], [], []
        self.arena_slots = 32

    def alloc_write(self, hashes):
        if not self.room:
            return None
        rows = torch.tensor([1000 + len(self.claimed)], dtype=torch.int64)
        self.claimed.append(list(hashes))
        return rows

    def abort_write(self, rows):
        self.aborted.append(rows)

    def complete_write(self, rows):
        self.completed.append([int(x) for x in rows.tolist()])

    def is_arena_id(self, i):
        return i >= 1000


class _Ctl:
    write_policy = "write_back"

    def __init__(self, fail=False):
        self.writes, self.fail = [], fail

    def write(self, device_indices, node_id=-1, extra_pools=None, host_indices=None):
        if self.fail:
            return None
        self.writes.append((int(device_indices.numel()), node_id, extra_pools, host_indices))
        return host_indices


def _tree_with_backed_kv_and_device_anchor(l3=False):
    """A node whose KV is backed (a host KV copy, or l3_present) and whose Mamba
    anchor lives on the device only -- the y5a ANCHOR-LOST shape."""
    cache, pool, kalloc = T._build()
    toks = list(range(2000, 2128))
    slot = pool.mamba_allocator.alloc(1)
    cache.insert(InsertParams(key=RadixKey(array("q", toks)), value=kalloc.alloc(128),
                              mamba_value=slot.reshape(1)))
    node = [n for n in cache._collect_all_nodes()
            if n is not cache.root_node and n.component_data[MC].value is not None][0]
    if l3:
        node.l3_present = True
    else:
        node.component_data[KV].host_value = torch.arange(500, 628, dtype=torch.int64)
    node.hash_value = ["hk%d" % i for i in range(128)]
    cache.cache_controller = _Ctl()
    return cache, node


def _sweep(cache, arena):
    with mock.patch.object(UnifiedRadixCache, "_weg2_mamba_pool", lambda self: arena), \
            mock.patch.object(UnifiedRadixCache, "_mamba_write_through_pin_admissible",
                              lambda self, node, write_back=False: True):
        return cache.publish_unbacked_sweep(max_issue=64)


def test_y5a_the_flush_sweep_backs_the_anchor_alone_and_nothing_is_lost():
    cache, node = _tree_with_backed_kv_and_device_anchor()
    depth = cache.weg2_node_depth(node)
    assert cache.weg2_unbacked_anchors() == [(depth, None)], "the y5a ANCHOR-LOST shape"
    arena = _Arena()
    stats = _sweep(cache, arena)
    assert stats["issued"] == 1 and stats.get("anchor_only") == 1
    (n_kv, nid, pools, host), = cache.cache_controller.writes
    assert n_kv == 0 and host.numel() == 0, "no KV row is copied"
    assert nid == node.id and len(pools) == 1 and int(pools[0].host_indices[0]) == 1000
    assert torch.equal(pools[0].device_indices, node.component_data[MC].value)
    assert arena.claimed == [["hk127"]], "the anchor's arena key is the node's last page hash"
    assert int(node.component_data[MC].host_value[0]) == 1000
    assert cache.weg2_unbacked_anchors() == [], "the flush probe finds nothing to drop"
    assert node.write_through_pending_id == node.id and node.id in cache.ongoing_write_through
    assert node.component_data[MC].lock_ref >= 1, "pinned until the ack"


def test_the_ack_completes_the_anchor_only():
    cache, node = _tree_with_backed_kv_and_device_anchor()
    arena = _Arena()
    _sweep(cache, arena)
    with mock.patch.object(UnifiedRadixCache, "_weg2_mamba_pool", lambda self: arena), \
            mock.patch.object(UnifiedRadixCache, "_weg2_release_inner_anchor", lambda self, n, mp: None):
        assert cache._weg2_direct_complete(node) is True
    assert arena.completed == [[1000]], "only the anchor's slot completes -- no second KV completion"
    assert node.id not in cache._weg2_anchor_only_ids


def test_a_store_backed_node_too():
    cache, node = _tree_with_backed_kv_and_device_anchor(l3=True)
    stats = _sweep(cache, _Arena())
    assert stats["issued"] == 1 and cache.weg2_unbacked_anchors() == []


def test_a_full_arena_or_a_refused_write_is_named_and_leaves_the_anchor():
    cache, node = _tree_with_backed_kv_and_device_anchor()
    stats = _sweep(cache, _Arena(room=False))
    assert stats["issued"] == 0 and stats["refused"] == 1 and stats["unbacked"] == 1
    assert node.component_data[MC].host_value is None and node.component_data[MC].value is not None
    cache2, node2 = _tree_with_backed_kv_and_device_anchor()
    cache2.cache_controller = _Ctl(fail=True)
    arena = _Arena()
    stats2 = _sweep(cache2, arena)
    assert stats2["refused"] == 1 and len(arena.aborted) == 1, "the claim goes back"


def test_switch_off_is_the_old_sweep():
    with envs.SGLANG_WEG2_ANCHOR_ONLY_BACKUP.override(False):
        cache, node = _tree_with_backed_kv_and_device_anchor()
        stats = _sweep(cache, _Arena())
        assert stats["issued"] == 0 and stats["unbacked"] == 0
        assert cache.weg2_unbacked_anchors() != []


def test_a_node_whose_anchor_is_on_the_host_is_not_touched():
    cache, node = _tree_with_backed_kv_and_device_anchor()
    node.component_data[MC].host_value = torch.tensor([7], dtype=torch.int64)
    stats = _sweep(cache, _Arena())
    assert stats["issued"] == 0 and cache.cache_controller.writes == []
