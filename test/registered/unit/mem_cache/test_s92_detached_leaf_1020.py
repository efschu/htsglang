"""S92: ``AssertionError: assert v == node`` -- a removed node came back into the device leaf set.

THE METAL (NF rc20 b40f47c91c, boot 1020_081757, D.log line 172855, TP0 08:50:12Z): pdflip-20-125's admission
(``_h105d_cut_load_back_room`` -> ``evict``) peeled ``L3-DROP FORMA n=3 node=514``, then the next leaf ended in
``_evict_device_leaf -> _pdflip_l3_drop_forma -> _ud_drop_unbacked_leaf -> _remove_leaf_from_parent``:
``assert v == node`` -- the leaf the peel took was no child of its parent any more. TP1/TP2 then died on the
closed gloo pair.

THE ROOT. ``_ud_drop_unbacked_leaf`` (and the write_through delete in ``_evict_device_leaf``) free a node's rows
and unlink it, but never clear its device values (``evict_component`` defers Full's ``value = None`` to
``_cascade_evict`` for SWA's sake; these paths never reach it). The removed node keeps ``evicted == False`` and
``_is_device_leaf == True``; one ``_update_evictable_leaf_sets`` on a stale handle (``dec_lock_ref`` ends with
it) puts it back into ``evictable_device_leaves``; the next peel frees its Full rows AGAIN -- rows that may be
another request's -- and only the parent's ``children.pop`` stopped the rank. Bestand: the same code on rc19.

WHAT MUST HOLD.
(1) A dropped node is neither a D-leaf nor an H-leaf for any caller; a stale set update cannot bring it back;
    the next eviction neither asserts nor frees its rows a second time (RED on 50a5943916: the metal assertion).
(2) Both removal paths clear the device values (UD drop and the write_through delete).
(3) A set update on a removed node is named (``LEAF-SET DETACHED``) and leaves both sets without it.
(4) No false detachment: a split keeps its two nodes in their sets, ordinary eviction is unchanged.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import pathlib
import sys
from array import array

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

LOGGER = "flliper.srt.mem_cache.unified_radix_cache"


def _fx():
    name = "_urc_unittest_fixture_s92"
    if name not in sys.modules:
        path = pathlib.Path(__file__).resolve().parent / "test_unified_radix_cache_unittest.py"
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    mod = sys.modules[name]
    mod.get_device = lambda: "cpu"
    return mod


@pytest.fixture
def world():
    fx = _fx()
    from flliper.srt.mem_cache.unified_radix_cache import ComponentType

    cfg = fx.CacheConfig(page_size=2, components=(ComponentType.FULL,), kv_size=64, max_context_len=64)
    cache, alloc, _ = fx.build_fixture(cfg)
    from flliper.srt.mem_cache.base_prefix_cache import InsertParams
    from flliper.srt.mem_cache.radix_cache import RadixKey

    def ins(t):
        v = alloc.alloc(len(t))
        cache.insert(InsertParams(key=RadixKey(array("q", t)), value=v[: len(t)]))

    ins([1, 2, 3, 4, 5, 6, 7, 8])
    ins([1, 2, 3, 4, 9, 10, 11, 12])
    a = list(cache.root_node.children.values())[0]
    b, c = list(a.children.values())
    return cache, alloc, a, b, c, ComponentType.FULL


def _tracker(cache):
    return {ct: 0 for ct in cache.tree_components}


def _free_ids(alloc):
    ids = [alloc.free_pages] + ([alloc.release_pages] if getattr(alloc, "release_pages", None) is not None else [])
    return torch.cat([t.reshape(-1) for t in ids if t is not None and t.numel()]).tolist()


# ---- (1) the metal ---------------------------------------------------------------------------------------------

def test_rc20_tp0_a_dropped_leaf_handed_back_by_a_stale_handle_is_never_peeled_again(world):
    """RED on 50a5943916: the next eviction asserts ``v == node`` after freeing the node's rows twice."""
    from flliper.srt.mem_cache.unified_radix_cache import EvictParams

    cache, alloc, a, b, c, F = world
    rows = b.component_data[F].value.clone()
    cache._ud_drop_unbacked_leaf(b, _tracker(cache))
    cache._update_evictable_leaf_sets(b)                 # a stale handle (any later set update)
    cache.evict(EvictParams(num_tokens=64))              # the metal: no ``assert v == node``
    assert b.evicted and not cache._is_device_leaf(b) and not cache._is_host_leaf(b)
    assert b not in cache.evictable_device_leaves and b not in cache.evictable_host_leaves
    free = _free_ids(alloc)
    assert len(free) == len(set(free)), "a row is in the free list twice: the dropped leaf was freed again"
    for r in rows.tolist():
        assert free.count(r) <= 1


def test_a_host_child_left_below_a_dropped_leaf_cannot_revive_it(world, caplog):
    """The stale handle through tree code: a dropped leaf B with a host-only child H left below it. Evicting H
    walks the tombstones up to B -- RED on 50a5943916: B (stale Full value) re-enters the device set and the
    next eviction asserts ``v == node``. GREEN: the walk stops at the detached B, named; nothing re-enters."""
    from flliper.srt.mem_cache.base_prefix_cache import InsertParams
    from flliper.srt.mem_cache.radix_cache import RadixKey
    from flliper.srt.mem_cache.unified_radix_cache import EvictParams

    caplog.set_level(logging.WARNING, LOGGER)
    cache, alloc, a, b, c, F = world
    v = alloc.alloc(10)
    cache.insert(InsertParams(key=RadixKey(array("q", [1, 2, 3, 4, 5, 6, 7, 8, 13, 14])), value=v[:10]))
    h = list(b.children.values())[0]
    h.component_data[F].host_value = torch.arange(2)
    cache._evict_to_host(h, _tracker(cache))             # H: host-only child of B; B is a D-leaf again
    assert cache._is_device_leaf(b) and cache._is_host_leaf(h)
    cache._ud_drop_unbacked_leaf(b, _tracker(cache))     # B leaves with H still below it
    cache._evict_host_leaf(h, _tracker(cache))           # the tombstone walk reaches the detached B
    assert b not in cache.evictable_device_leaves and b not in cache.evictable_host_leaves
    assert any("LEAF-SET DETACHED" in m for m in caplog.messages), caplog.messages
    cache.evict(EvictParams(num_tokens=64))              # no AssertionError
    free = _free_ids(alloc)
    assert len(free) == len(set(free))


# ---- (2) both removal paths ----------------------------------------------------------------------------------

def test_the_ud_drop_clears_every_device_value(world):
    cache, alloc, a, b, c, F = world
    cache._ud_drop_unbacked_leaf(b, _tracker(cache))
    assert all(cd.value is None for cd in b.component_data)


def test_the_write_through_delete_clears_every_device_value(world):
    cache, alloc, a, b, c, F = world
    assert not b.backuped and cache.cache_controller is None     # the write_through delete branch
    cache._evict_device_leaf(b, _tracker(cache))
    assert b not in a.children.values()
    assert all(cd.value is None for cd in b.component_data)
    cache._update_evictable_leaf_sets(b)
    assert b not in cache.evictable_device_leaves


# ---- (4) no false detachment ------------------------------------------------------------------------------------

def test_a_split_keeps_its_nodes_in_their_sets_and_nothing_is_named_detached(world, caplog):
    caplog.set_level(logging.WARNING, LOGGER)
    cache, alloc, a, b, c, F = world
    from flliper.srt.mem_cache.base_prefix_cache import InsertParams
    from flliper.srt.mem_cache.radix_cache import RadixKey

    v = alloc.alloc(6)
    cache.insert(InsertParams(key=RadixKey(array("q", [1, 2, 3, 4, 5, 6, 13, 14])), value=torch.cat(
        [b.component_data[F].value[:0], alloc.alloc(8)])[:8]))          # splits B after 5,6
    del v
    leaves = cache.evictable_device_leaves
    for n in list(leaves):
        assert not cache._leaf_set_detached(n)
    assert not [m for m in caplog.messages if "LEAF-SET DETACHED" in m]
    assert len(leaves) == 3, [n.id for n in leaves]


def test_ordinary_eviction_still_empties_the_tree(world):
    from flliper.srt.mem_cache.unified_radix_cache import EvictParams

    cache, alloc, a, b, c, F = world
    n_free = len(_free_ids(alloc))
    cache.evict(EvictParams(num_tokens=64))
    assert not cache.root_node.children
    free = _free_ids(alloc)
    assert len(free) == len(set(free)) and len(free) == n_free + 12


def test_a_removal_path_that_forgets_to_clear_still_cannot_reenter_the_sets(world, caplog):
    """Defence in depth for the next removal path: a detached node that kept a device value (a path that forgot
    ``_clear_dropped_device_values``) is still refused by the leaf sets, by name."""
    caplog.set_level(logging.WARNING, LOGGER)
    cache, alloc, a, b, c, F = world
    rows = b.component_data[F].value
    cache._ud_drop_unbacked_leaf(b, _tracker(cache))
    b.component_data[F].value = rows                     # the forgetful path
    assert cache._is_device_leaf(b)
    cache._update_evictable_leaf_sets(b)
    assert b not in cache.evictable_device_leaves
    assert any("LEAF-SET DETACHED" in m for m in caplog.messages)
