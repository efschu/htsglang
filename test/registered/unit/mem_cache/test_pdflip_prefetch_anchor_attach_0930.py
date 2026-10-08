"""PREFETCH ANCHOR ATTACH (NF y5a 30.09., pdflip-19-28, D-Log 0110f8e132_0930_174039).

The chain, from the D log (TP0):
* 17:48:10 pdflip-19-28's hold read, OWN keys (``#1442 HANDOFF-KEYS NONE``), finds
  the session's prefix in the store: ``#1028B FETCH CAP kv=675 claimed=665 ...
  anchors_in_range mamba (9, 664)`` -- 665 pages = 42560 tokens WITH the Mamba
  anchor at page 664 (the key identity is NOT the gap: 18-24 read the same
  prefix with own keys too, store_hit=42240).
* 17:48:12 the sibling pdflip-18-24 is loaded back first (``#988 LOADBACK ...
  prefix moved to 43520 ... anchor_depth=43520``): its KV lands on the DEVICE
  with the only state at 43520.
* 19-28's read completes (``PREFETCH-COMPLETE completed=42560``): the host
  insert walks a span that is ALREADY in the tree, inserts no host node
  (``inserted_host_node`` None, the node carries no host KV copy) -- and
  ``MambaComponent.commit_hicache_transfer(PREFETCH)`` RELEASED the anchor.
* ``[#904 match-census] reached=43200 accepted=0 refusers=MambaComponent:
  absent=43200`` -> ``[#928 anchor] REFUSING resume ... best_value_len=0
  NONE-ON-THIS-PATH`` -> W31 -> W50 x_refusal_midstream, 43396 tokens to P.

Now the host insert names the node its key ENDED at (``matched_end_node``, the
walk split it at the read's end = the anchor's depth), and a node with no
state takes the anchor. A node with a state keeps its own (the read's copy is
released as before). Switch FLLIPER_PDFLIP_PREFETCH_ANCHOR_ATTACH (default on).
Real UnifiedRadixCache over a real HybridReqToTokenPool.
"""
from __future__ import annotations

import importlib.util
import os
import types
from array import array

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.mem_cache.base_prefix_cache import InsertParams, InsertResult  # noqa: E402
from flliper.srt.mem_cache.radix_cache import RadixKey  # noqa: E402
from flliper.srt.mem_cache.unified_cache_components.tree_component import (  # noqa: E402
    CacheTransferPhase,
    ComponentType,
)

_spec = importlib.util.spec_from_file_location(
    "_t_compact_aa", os.path.join(os.path.dirname(__file__), "..", "pdflip", "test_pdflip_d_seat_compact_0930.py"))
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)
MC = ComponentType.MAMBA


class _Ctl:
    write_policy = "write_back"   # group D's policy (y5a hicache_write_policy='write_back')

    def __init__(self):
        self.released = []

    def append_host_mem_release(self, extra_pools=None):
        self.released.append(extra_pools)


def _device_path(n_tokens=128, anchor=None):
    """A device path of ``n_tokens`` (the sibling's load), state only where given."""
    cache, pool, kalloc = T._build()
    toks = list(range(1000, 1000 + n_tokens))
    mv = None
    if anchor is not None:
        slot = pool.mamba_allocator.alloc(1)
        mv = slot.reshape(1)
    cache.insert(InsertParams(key=RadixKey(array("q", toks)), value=kalloc.alloc(n_tokens), mamba_value=mv))
    cache.cache_controller = _Ctl()   # the read side (the device path was built without a host tier)
    return cache, pool, toks


def _read_span(cache, toks, n):
    """The host insert of a store read covering ``toks[:n]`` (as check_prefetch_progress does)."""
    hv = torch.arange(10000, 10000 + n, dtype=torch.int64)
    return cache._insert_helper_host(cache.root_node, RadixKey(array("q", toks[:n])), hv,
                                     ["h%d" % i for i in range(n)])


def _commit(cache, res, loaded=True):
    comp = cache.components[MC]
    xfer = types.SimpleNamespace(host_indices=torch.tensor([7], dtype=torch.int64))
    psr = types.SimpleNamespace(extra_pool_hit_pages={__import__(
        "flliper.srt.mem_cache.hicache_storage", fromlist=["PoolName"]).PoolName.MAMBA: 1 if loaded else 0})
    comp.commit_hicache_transfer(None, CacheTransferPhase.PREFETCH, [xfer],
                                 insert_result=res, pool_storage_result=psr)


def _node_at(cache, depth):
    for n in cache._collect_all_nodes():
        if n is not cache.root_node and cache.pdflip_node_depth(n) == depth:
            return n
    return None


def test_y5a_the_read_anchor_stays_when_the_span_is_already_on_the_device():
    """19-28's read ends at 96 inside the sibling's device path (state nowhere
    on it): the walk splits the node at 96 and the anchor stays there."""
    cache, pool, toks = _device_path(128, anchor=None)
    res = _read_span(cache, toks, 96)
    assert res.inserted_host_node is None, "the span was in the tree: no host node"
    assert res.matched_end_node is not None
    _commit(cache, res)
    node = _node_at(cache, 96)
    assert node is res.matched_end_node
    assert node.component_data[MC].host_value is not None and int(node.component_data[MC].host_value[0]) == 7
    assert cache.cache_controller.released == [], "the anchor is not released"
    assert cache.host_lru_lists[MC].in_list(node)


def test_a_node_with_a_state_keeps_its_own_and_the_copy_is_released():
    cache, pool, toks = _device_path(128, anchor=True)
    res = _read_span(cache, toks, 128)
    node = res.matched_end_node
    assert node is not None and node.component_data[MC].value is not None
    _commit(cache, res)
    assert node.component_data[MC].host_value is None
    assert len(cache.cache_controller.released) == 1


def test_switch_off_releases_as_before():
    with envs.FLLIPER_PDFLIP_PREFETCH_ANCHOR_ATTACH.override(False):
        cache, pool, toks = _device_path(128, anchor=None)
        res = _read_span(cache, toks, 96)
        _commit(cache, res)
        assert _node_at(cache, 96).component_data[MC].host_value is None
        assert len(cache.cache_controller.released) == 1


def test_no_mamba_blob_loaded_attaches_nothing():
    cache, pool, toks = _device_path(128, anchor=None)
    res = _read_span(cache, toks, 96)
    _commit(cache, res, loaded=False)
    assert _node_at(cache, 96).component_data[MC].host_value is None
    assert len(cache.cache_controller.released) == 1


def test_a_new_host_node_keeps_the_old_path():
    """The ordinary case (the read's tail is NEW to the tree): unchanged."""
    cache, pool, toks = _device_path(64, anchor=True)
    cache.cache_controller = _Ctl()
    toks2 = toks + list(range(5000, 5064))
    hv = torch.arange(20000, 20128, dtype=torch.int64)
    res = cache._insert_helper_host(cache.root_node, RadixKey(array("q", toks2)), hv,
                                    ["g%d" % i for i in range(128)])
    assert res.inserted_host_node is not None and res.matched_end_node is None
    _commit(cache, res)
    assert res.inserted_host_node.component_data[MC].host_value is not None


def test_the_next_match_resumes_at_the_attached_anchor():
    """What 19-28's admission match reads afterwards: the anchor's node is the
    best match (state depth 96, one Mamba host hit -> init_load_back restores
    the state into the request's slot and takes the device KV above it).
    Without the attach: no state on the path (the y5a refusal)."""
    from flliper.srt.mem_cache.base_prefix_cache import MatchPrefixParams

    for attach, want in ((True, 96), (False, 0)):
        with envs.FLLIPER_PDFLIP_PREFETCH_ANCHOR_ATTACH.override(attach):
            cache, pool, toks = _device_path(128, anchor=None)
            res = _read_span(cache, toks, 96)
            _commit(cache, res)
            m = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", toks[:120]))))
        assert m.state_anchor_depth == want, attach
        assert m.mamba_host_hit_length == (1 if attach else 0)
        if attach:
            assert m.best_match_node is res.matched_end_node
