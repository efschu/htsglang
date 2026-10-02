# SPDX-License-Identifier: Apache-2.0
"""L15-FIX-HOSTGROUP (N3j 02.10. 00:24Z): the live host pools are reached
THROUGH the HostPoolGroup.

On the 27B hybrid the cache controller's ``mem_pool_host`` is a
``HostPoolGroup`` (hybrid_pool_assembler builds HybridCacheController with
the group), not the ArenaMHAHostPool. The group has no ``slot_gens`` /
``staging_rows``, so build_retain_kwargs raised ``AttributeError: slot_gens``
on every rank in every retain round that held a candidate (N3j D.log,
"L15-RETAIN failed before the move"). The controller also has no
``mamba_pool_host``; the anchor pool is the group's MAMBA entry.

Pinned here, with the REAL HostPoolGroup/PoolEntry classes:
(a) the KV pool is the group's KV entry: l2_of maps rows through the
    entry's staging_rows and asks the entry's slot_gens;
(b) the anchor pool is the group's MAMBA entry: anchor_l2_of reads it;
(c) a resolved pool WITHOUT slot_gens (plain MambaPoolHost on a form-A
    worker) degrades to gen -1 for those slots instead of failing the round.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.memory_pool_host import HostPoolGroup, PoolEntry
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    ComponentType,
)
from sglang.srt.weg2.l15_bind import build_retain_kwargs


class _KVPool:
    """ArenaMHAHostPool duck (P == 1, 27B form)."""

    layout = "layer_first"
    page_size = 1
    device = "cpu"
    size = 64

    def __init__(self, staging_rows, gens):
        self.staging_rows = staging_rows
        self._arena_page_tokens = 1
        self.row_slot = None
        self._gens = gens

    def slot_gens(self, slots):
        return [int(self._gens.get(int(s), -1)) for s in slots]


class _MambaPool:
    """ArenaMambaPoolHost duck: one state per slot."""

    layout = "layer_first"
    page_size = 1
    device = "cpu"
    size = 8

    def __init__(self, staging_rows, gens):
        self.staging_rows = staging_rows
        self._gens = gens

    def slot_gens(self, slots):
        return [int(self._gens.get(int(s), -1)) for s in slots]


class _PlainMambaPool:
    """Plain MambaPoolHost duck: no arena census, no slot_gens."""

    layout = "layer_first"
    page_size = 1
    device = "cpu"
    size = 8
    staging_rows = 2


def _group(kv, mamba):
    return HostPoolGroup([
        PoolEntry(name=PoolName.KV, host_pool=kv, device_pool=None,
                  layer_mapping={}, is_primary_index_anchor=True),
        PoolEntry(name=PoolName.MAMBA, host_pool=mamba, device_pool=None,
                  layer_mapping={}),
    ])


def _cd(value, host_value):
    def _t(v):
        if v is None:
            return None
        return torch.tensor(v if isinstance(v, list) else [v],
                            dtype=torch.int64)

    return SimpleNamespace(value=_t(value), host_value=_t(host_value),
                           host_lock_ref=0)


def _node(kv_host_rows, anchor, anchor_host_row):
    return SimpleNamespace(
        parent=None,
        key=(0,),
        tree_components=(ComponentType.FULL, ComponentType.MAMBA),
        component_data={
            ComponentType.FULL: _cd(None, kv_host_rows),
            ComponentType.MAMBA: _cd(anchor, anchor_host_row),
        },
    )


def _req(node):
    return SimpleNamespace(
        rid="r1",
        req_pool_idx=0,
        origin_input_ids=list(range(4)),
        output_ids=[],
        mamba_pool_idx=4,
        last_node=node,
        l15_kind="served",
        l15_last_active=1.0,
    )


def _bound_reset_keep(group):
    tree_cache = SimpleNamespace(
        cache_controller=SimpleNamespace(mem_pool_host=group),
        reset_keep=lambda _ns: None,
    )

    def reset_keep(_ns):  # bound-method stand-in with __self__
        return tree_cache.reset_keep(_ns)

    reset_keep.__self__ = tree_cache
    return reset_keep


def _build(tmp_path, group):
    rtt = torch.zeros(2, 8, dtype=torch.int64)
    rtt[0, :4] = torch.tensor([1, 2, 3, 5], dtype=torch.int64)
    # span 3: KV host rows 2,3,4 on staging_rows=2 -> slots 0,1,2;
    # anchor host row 9 on mamba staging_rows=2 -> slot 7.
    req = _req(_node([2, 3, 4], 4, 9))
    return build_retain_kwargs(
        [req],
        rtt,
        caps_rows_by_rank=(100,),
        cap_anchor_slots=4,
        prefix=(0, 100),
        rank=0,
        epoch=3,
        pid=7,
        kv_buffers=[],
        mamba_buffers=[],
        allocator=None,
        reset_keep=_bound_reset_keep(group),
        set_keep=lambda _b, _s: None,
        manifest_path=str(tmp_path / "m.json"),
        log=lambda _s: None,
    )


def test_kv_pool_is_the_groups_kv_entry(tmp_path):
    group = _group(_KVPool(2, {0: 5, 1: 6, 2: 7}), _MambaPool(2, {7: 42}))
    assert not hasattr(group, "slot_gens")  # the N3j shape
    kw = _build(tmp_path, group)
    assert kw["l2_of"]("r1") == ((0, 1, 2), (5, 6, 7))


def test_anchor_pool_is_the_groups_mamba_entry(tmp_path):
    group = _group(_KVPool(2, {0: 5, 1: 6, 2: 7}), _MambaPool(2, {7: 42}))
    kw = _build(tmp_path, group)
    assert kw["anchor_l2_of"]("r1") == (7, 42)


def test_pool_without_slot_gens_degrades_to_minus_one(tmp_path):
    group = _group(_KVPool(2, {0: 5, 1: 6, 2: 7}), _PlainMambaPool())
    kw = _build(tmp_path, group)
    # the KV identity is intact; the anchor has a slot but no generation
    assert kw["l2_of"]("r1") == ((0, 1, 2), (5, 6, 7))
    assert kw["anchor_l2_of"]("r1") == (7, -1)


def test_live_host_pools_unwraps_the_group_for_the_refill():
    # the wake refill (_l15_do_refill) resolves its pools through the same
    # helper; a group gives its KV / MAMBA entries
    from sglang.srt.weg2.l15_bind import live_host_pools

    kv, mamba = _KVPool(2, {}), _MambaPool(2, {})
    tree = SimpleNamespace(
        cache_controller=SimpleNamespace(mem_pool_host=_group(kv, mamba)))
    assert live_host_pools(tree) == (kv, mamba)


def test_live_host_pools_plain_pool_falls_back_to_tree_mamba_attr():
    # older hybrid mamba cache: plain KV pool on the controller, the mamba
    # host pool on the tree itself
    from sglang.srt.weg2.l15_bind import live_host_pools

    kv, mamba = _KVPool(2, {}), _MambaPool(2, {})
    tree = SimpleNamespace(cache_controller=SimpleNamespace(mem_pool_host=kv),
                           mamba_pool_host=mamba)
    assert live_host_pools(tree) == (kv, mamba)
    assert live_host_pools(None) == (None, None)
