"""ANCHOR-PIN (z30y15, P PP0 04:21:39Z, rid weg2-12-56): the recurrent anchor a
storage prefetch brings back must stay pinned until the admission, like the
KV chain #1417 pins.

MEASURED (P log boot_weg2_dkr27browauthoritybar1fs10010415_86d80796e0):

    #1028B FETCH CAP ... caps={mamba: 16383} anchors_in_range=(4, 16382)
    #1423 INSERT-PLACED req=weg2-12- ... matched=16383 inserted=0 deepest=107
    #1427 ARENA-DROP n=3 need=1 freed=1 slot_bytes=78446592   (the 112-slot mamba arena)
    #TF TOLD-FIDELITY rid=weg2-12-56 told=16383 pp0_admissible=0   (04:21:40)
    PP1/PP2 [#928 anchor] REFUSING resume ... best_value_len=0 NONE-ON-THIS-PATH

The read brought the anchor at 16383, the insert named node 107 (Full host
copy) and the mamba PREFETCH commit attached the anchor there -- but
``check_prefetch_progress`` takes the #1417 pin BEFORE the components'
commit: at pin time node 107 carries no mamba host value, so its mamba lock
is skipped (``skip_lock_node_ids``), and the commit then leaves the fresh
anchor UNLOCKED on the mamba host LRU. One mamba-arena eviction later the
prefix had no state, PP0's told-fidelity sent told=0 and P re-prefilled 16k.
(test_prefetch_pin_host_lru_sanity_1417b.py describes the same order.)

Driven through the real ``check_prefetch_progress`` (#1157 reap harness) on
a real FULL+MAMBA tree."""

import importlib.util
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from sglang.srt.mem_cache.hicache_phase_binding import binding_state  # noqa: E402
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer  # noqa: E402
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType  # noqa: E402

from test_unified_radix_cache_unittest import CacheConfig, build_fixture  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_t_1157_anchor", os.path.join(os.path.dirname(__file__), "test_1157_reaper_prices_requested_span.py"))
h1157 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h1157)

FULL, MAMBA = ComponentType.FULL, ComponentType.MAMBA
REQ = h1157.REAP_REQ


def _mamba_tree():
    cache = build_fixture(CacheConfig(page_size=h1157.PAGE_SIZE, components=(FULL, MAMBA)))[0]
    # the harness builds its host pool over the FULL KV pool of the hybrid
    alloc = cache.token_to_kv_pool_allocator
    full = alloc.get_kvcache().full_kv_pool
    alloc.get_kvcache = lambda: full
    return cache


def _completed_with_anchor(monkeypatch):
    monkeypatch.setattr(h1157, "_build_cache", _mamba_tree)
    cache, op = h1157._reap_scenario(probed=True)
    e = cache.ongoing_prefetch[REQ]
    cache.ongoing_prefetch[REQ] = type(e)(
        e[0], e[1], e[2], e[3], e[4],
        {MAMBA: [PoolTransfer(name=PoolName.MAMBA, host_indices=torch.tensor([7]))]})
    op.pool_storage_result.extra_pool_hit_pages[PoolName.MAMBA] = 1
    assert cache.check_prefetch_progress(REQ)
    node = cache.root_node
    while node.children:
        node = next(iter(node.children.values()))
    return cache, node


def test_the_prefetched_anchor_is_pinned_until_admission(monkeypatch):
    """RED on 38c1d66516: host_lock_ref 0 and the anchor on the mamba host LRU
    (an eviction candidate) between the completion and the admission."""
    try:
        cache, node = _completed_with_anchor(monkeypatch)
        cd = node.component_data[MAMBA]
        assert cd.host_value is not None and int(cd.host_value[0]) == 7  # the anchor was attached
        assert cd.host_lock_ref == 1, cd.host_lock_ref
        assert not cache.host_lru_lists[MAMBA].in_list(node)
    finally:
        binding_state().reset()


def test_the_mamba_host_eviction_cannot_take_it(monkeypatch):
    """The z30y15 failure itself: a mamba host eviction between completion and
    admission must not strip the anchor of a pinned prefetch."""
    try:
        cache, node = _completed_with_anchor(monkeypatch)
        cache.components[MAMBA].drive_host_eviction(1, {MAMBA: 0, FULL: 0})
        assert node.component_data[MAMBA].host_value is not None
    finally:
        binding_state().reset()


def test_the_admission_releases_the_pin(monkeypatch):
    """pop_prefetch_loaded_tokens (the admission) gives the anchor back to the
    host LRU: pinned for exactly the window, never leaked."""
    try:
        cache, node = _completed_with_anchor(monkeypatch)
        cache.pop_prefetch_loaded_tokens(REQ)
        cd = node.component_data[MAMBA]
        assert cd.host_lock_ref == 0
        assert cache.host_lru_lists[MAMBA].in_list(node)
        assert not getattr(cache, "_prefetch_span_pins", {})
    finally:
        binding_state().reset()
