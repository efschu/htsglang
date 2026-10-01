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


# ------------------------------------------------------------------ y5d
# NF y5d (image rc12z30y5f, 60e26b59d1) D TP0 19:52:54Z, rank death in on_idle ->
# sanity_check: "node 147 mamba host present but Full.host_value=None; node 146
# dito". Tree [137] root-child 0 -> [147] 41024 full+mamba -> [149] 192;
# [146] 41024 -> [148] 192. 146/147 came out of L3 (l3_present, KV on the
# device, NO Full host copy); WEG2 ANCHOR-ONLY-BACKUP n=5/6 kv=store committed
# the anchor as the tree's Mamba host value. The tree law "aux host requires
# Full host" stands (every host-tier reader keys on Full.host_value): with the
# KV only in the store, the anchor lives in its arena slot, not in the tree.

class _ArenaRef(_Arena):
    def __init__(self, room=True):
        super().__init__(room)
        self.freed = []

    def free(self, rows):
        self.freed.append([int(x) for x in rows.tolist()])
        return int(rows.numel())


def _y5d_tree():
    """Two 64-token siblings under the root, each with a 3-token child; the
    siblings' KV came from L3 (l3_present, no host copy), anchors on device."""
    cache, pool, kalloc = T._build()
    heads = []
    for base in (3000, 4000):
        head = list(range(base, base + 64))
        cache.insert(InsertParams(key=RadixKey(array("q", head)), value=kalloc.alloc(64),
                                  mamba_value=pool.mamba_allocator.alloc(1).reshape(1)))
        full = head + [base + 900, base + 901, base + 902]
        cache.insert(InsertParams(key=RadixKey(array("q", full)), value=kalloc.alloc(67),
                                  mamba_value=pool.mamba_allocator.alloc(1).reshape(1)))
    for n in cache._collect_all_nodes():
        if n is cache.root_node:
            continue
        n.hash_value = ["h%d_%d" % (n.id, i) for i in range(len(n.key))]
        if len(n.key) == 64:
            n.l3_present = True          # KV in the store only (the L3 read's span)
            heads.append(n)
    cache.cache_controller = _Ctl()
    assert len(heads) == 2 and all(h.component_data[KV].host_value is None for h in heads)
    return cache, heads


def test_y5d_store_anchor_keeps_the_tree_law_and_the_arena_carries_it():
    cache, heads = _y5d_tree()
    cache.sanity_check()                                     # the state before: legal
    arena = _ArenaRef()
    stats = _sweep(cache, arena)
    assert stats.get("anchor_only") == 2 and len(cache.cache_controller.writes) == 2
    cache.sanity_check()                                     # y5d: AssertionError on 60e26b59d1
    for h in heads:
        assert h.component_data[MC].host_value is None, "no Mamba host value without a Full host value"
    assert 64 not in [d for d, _ in cache.weg2_unbacked_anchors()], "in flight: not counted lost"
    with mock.patch.object(UnifiedRadixCache, "_weg2_mamba_pool", lambda self: arena), \
            mock.patch.object(UnifiedRadixCache, "_weg2_release_inner_anchor", lambda self, n, mp: None):
        for h in heads:
            assert cache._weg2_direct_complete(h) is True
    assert sorted(arena.completed) == [[1000], [1001]], "each anchor's arena slot completes"
    assert sorted(arena.freed) == [[1000], [1001]], "and this write's reader reference goes back"
    assert all(h.weg2_anchor_secured for h in heads)
    assert 64 not in [d for d, _ in cache.weg2_unbacked_anchors()], "secured: the flush drops nothing"
    cache.sanity_check()
    n_before = len(cache.cache_controller.writes)
    _sweep(cache, arena)
    assert len(cache.cache_controller.writes) == n_before, "a secured anchor is not rewritten"


def test_a_host_backed_kv_still_commits_the_anchor_into_the_tree():
    cache, node = _tree_with_backed_kv_and_device_anchor()
    _sweep(cache, _ArenaRef())
    assert int(node.component_data[MC].host_value[0]) == 1000
    assert not getattr(node, "weg2_anchor_secured", False)


# ------------------------------------------------------------------ y6b
# Review a54a22e54d F1: the ack hands the write's reference back (the tree holds
# none), so the anchor's slot is a clock candidate. The clock writes it to L3
# before it frees it (#257 (d)) -- but `weg2_anchor_secured` was set once and
# never cleared: a slot that left L2 with NO L3 copy (dropped_without_l3) stayed
# "secured", the sweep never rewrote the still-present device anchor and the
# ANCHOR-LOST probe never counted it. The mark now holds only while the bytes do
# (ArenaMambaPoolHost.anchor_held: a COMPLETE slot or an L3 copy).

class _ArenaHeld(_ArenaRef):
    def __init__(self, room=True):
        super().__init__(room)
        self.gone = set()        # last-page hashes whose slot left L2 with no L3 copy
        self.asked = []

    def anchor_held(self, last_hash):
        self.asked.append(last_hash)
        return last_hash not in self.gone


def _secured_heads():
    cache, heads = _y5d_tree()
    arena = _ArenaHeld()
    _sweep(cache, arena)
    with mock.patch.object(UnifiedRadixCache, "_weg2_mamba_pool", lambda self: arena), \
            mock.patch.object(UnifiedRadixCache, "_weg2_release_inner_anchor", lambda self, n, mp: None):
        for h in heads:
            assert cache._weg2_direct_complete(h) is True
            ow = cache.ongoing_write_through.pop(h.id)    # the rest of the ack (writing_check)
            if ow.lock_params is not None:
                cache.dec_lock_ref(h, ow.lock_params)
            h.write_through_pending_id = None
    assert all(h.weg2_anchor_secured for h in heads)
    return cache, heads, arena


def _probe(cache, arena):
    with mock.patch.object(UnifiedRadixCache, "_weg2_mamba_pool", lambda self: arena):
        return [d for d, _ in cache.weg2_unbacked_anchors()]


def test_y6b_a_secured_anchor_whose_bytes_are_gone_is_counted_and_written_again():
    cache, heads, arena = _secured_heads()
    assert 64 not in _probe(cache, arena), "bytes held: secured, nothing lost"
    lost = heads[0]
    arena.gone.add(lost.hash_value[-1])        # the clock freed it, the L3 write was lost
    assert _probe(cache, arena).count(64) == 1, "ANCHOR-LOST names the anchor (stuck mark on cb3aa0c2da)"
    assert lost.weg2_anchor_secured is False and heads[1].weg2_anchor_secured is True
    n_before = len(cache.cache_controller.writes)
    stats = _sweep(cache, arena)
    assert stats.get("anchor_only") == 1 and len(cache.cache_controller.writes) == n_before + 1, \
        "the sweep writes the device anchor again"
    assert cache.cache_controller.writes[-1][1] == lost.id
    cache.sanity_check()


def test_y6b_an_anchor_with_an_l3_copy_stays_secured_and_is_not_rewritten():
    cache, heads, arena = _secured_heads()   # anchor_held answers True: arena slot or L3 copy
    n_before = len(cache.cache_controller.writes)
    _sweep(cache, arena)
    assert len(cache.cache_controller.writes) == n_before and all(h.weg2_anchor_secured for h in heads)
    assert {h.hash_value[-1] for h in heads} <= set(arena.asked), "the mark is checked, not trusted"


def test_y6b_a_pool_that_cannot_tell_keeps_the_mark():
    cache, heads = _y5d_tree()
    arena = _ArenaRef()                       # no anchor_held: the old behaviour
    _sweep(cache, arena)
    with mock.patch.object(UnifiedRadixCache, "_weg2_mamba_pool", lambda self: arena), \
            mock.patch.object(UnifiedRadixCache, "_weg2_release_inner_anchor", lambda self, n, mp: None):
        for h in heads:
            cache._weg2_direct_complete(h)
    assert 64 not in _probe(cache, arena) and all(h.weg2_anchor_secured for h in heads)


def test_y6b_anchor_held_reads_the_arena_then_the_l3_index():
    from types import SimpleNamespace

    from sglang.srt.mem_cache.pool_host.arena_mamba_pool import ArenaMambaPoolHost

    states = {"s:a": (3, 2), "s:b": (4, 1)}        # a COMPLETE, b CLAIMED (not readable)
    disk = {"s:c"}
    arena = SimpleNamespace(find_slots=lambda stems: [states.get(s, (-1, 0)) for s in stems])
    backend = SimpleNamespace(_stat_stems=lambda stems: {s: 1 for s in stems if s in disk})
    pool = SimpleNamespace(arena=arena, _backend=backend, _stems=lambda hs: ["s:" + h for h in hs])
    held = lambda h: ArenaMambaPoolHost.anchor_held(pool, h)  # noqa: E731
    assert held("a") is True, "COMPLETE slot"
    assert held("b") is False, "a CLAIMED slot is not the anchor's bytes"
    assert held("c") is True, "freed by the clock WITH an L3 copy (#257 (d))"
    assert held("d") is False, "freed with no copy: dropped_without_l3"
    pool.arena = None
    assert held("a") is None, "unbound: cannot tell"
