"""TOLD-PIN + ACK-RESUMABLE (N1 dkr27browauthoritybar1fs10010740, image
32fc2683b1 = 341d089831 + L1,5 shadow, PP1 death 07:46:29Z).

MEASURED (P log, rid weg2-10-16):

    PP0  PF TOLD-ACKED told=17406 acks={1: 17406, 2: 17406}
    PP0  #988 LOADBACK prefix moved to 17406 ... mamba_restored=5
    PP1  #1423 INSERT-PLACED ... matched=17406 inserted=0 deepest=122
    PP1  H91 STORE-TOLD KEPT told=17406 credit=0      (not seated in the told visit)
    PP1  [#928 anchor] REFUSING resume ... match_tokens=17406 best_value_len=0
    PP1  #1175 PREFIX-EXEC UNDER-COVERAGE local=0 scheduled=17406 -> 19 s -> #968 STOP

Two gaps, both closed here:
  (a) the told admission popped the loaded count with
      ``pop_prefetch_loaded_tokens``, which also released the #1417 span pin
      (and the anchor ANCHOR-PIN keeps under it) -- while the request waited
      for a later visit (H91 KEPT) the anchor was evictable. The credit is
      popped, the pin stays until the request leaves the queue
      (``p_intake.settle_told``).
  (b) the follower's PF ack named its KV depth, not what it can RESUME from:
      KV without a recurrent state acked told, PP0 admitted, the follower
      could not materialise it. The ack now asks the told-fidelity probe; an
      unresumable span acks less than told and PP0 sends told=0 to every rank.

Real ``check_prefetch_progress`` (#1157 harness) on a real FULL+MAMBA tree."""

import importlib.util
import os
from array import array
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from sglang.srt.managers import weg2_store_told as st  # noqa: E402
from sglang.srt.managers import weg2_told_fallback as fb  # noqa: E402
from sglang.srt.mem_cache.hicache_phase_binding import binding_state  # noqa: E402
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer  # noqa: E402
from sglang.srt.mem_cache.radix_cache import RadixKey  # noqa: E402
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType  # noqa: E402
from sglang.srt.weg2 import p_intake  # noqa: E402

from test_unified_radix_cache_unittest import CacheConfig, build_fixture  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_t_1157_told_pin", os.path.join(os.path.dirname(__file__), "test_1157_reaper_prices_requested_span.py"))
h1157 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h1157)

FULL, MAMBA = ComponentType.FULL, ComponentType.MAMBA
REQ = h1157.REAP_REQ
TOLD = h1157.REAP_TOKENS


def _mamba_tree():
    cache = build_fixture(CacheConfig(page_size=h1157.PAGE_SIZE, components=(FULL, MAMBA)))[0]
    alloc = cache.token_to_kv_pool_allocator
    full = alloc.get_kvcache().full_kv_pool
    alloc.get_kvcache = lambda: full
    return cache


def _completed(monkeypatch, with_anchor: bool):
    monkeypatch.setattr(h1157, "_build_cache", _mamba_tree)
    cache, op = h1157._reap_scenario(probed=True)
    e = cache.ongoing_prefetch[REQ]
    # the server's key form: array('q') ids (the told probe hashes the same form)
    key = RadixKey(array("q", list(e[1].token_ids)))
    xfers = ({MAMBA: [PoolTransfer(name=PoolName.MAMBA, host_indices=torch.tensor([7]))]}
             if with_anchor else {})
    cache.ongoing_prefetch[REQ] = type(e)(e[0], key, e[2], e[3], e[4], xfers)
    if with_anchor:
        op.pool_storage_result.extra_pool_hit_pages[PoolName.MAMBA] = 1
    assert cache.check_prefetch_progress(REQ)
    node = cache.root_node
    while node.children:
        node = next(iter(node.children.values()))
    return cache, node


def _follower(cache):
    return SimpleNamespace(tree_cache=cache, _weg2_store_told={REQ: TOLD},
                           ps=SimpleNamespace(pp_rank=1, pp_size=3, tp_size=1))


def _req():
    return SimpleNamespace(rid=REQ, origin_input_ids=list(range(1, TOLD + 1)), extra_key=None,
                           full_untruncated_fill_ids=None)


def test_the_told_admission_keeps_the_anchor_pinned_until_the_request_leaves_the_queue(monkeypatch):
    """RED on 341d089831: the told admission (credit popped, request not yet
    seated -- H91 KEPT) leaves the anchor unlocked on the mamba host LRU."""
    try:
        cache, node = _completed(monkeypatch, with_anchor=True)
        sched, req = _follower(cache), _req()
        credit = p_intake.told_admission(sched, req, lambda *a: None, st.admission)
        assert credit == TOLD
        cd = node.component_data[MAMBA]
        assert cd.host_value is not None
        assert cd.host_lock_ref == 1, cd.host_lock_ref          # still pinned while KEPT
        cache.components[MAMBA].drive_host_eviction(1, {MAMBA: 0, FULL: 0})
        assert cd.host_value is not None                         # eviction cannot take it
        # the request is seated (no longer queued): the pin goes with the verdict
        assert p_intake.settle_told(sched, waiting_queue=[]) == 1
        assert cd.host_lock_ref == 0
        assert cache.host_lru_lists[MAMBA].in_list(node)
        assert not getattr(cache, "_prefetch_span_pins", {})
    finally:
        binding_state().reset()


def test_a_follower_without_an_anchor_acks_less_than_told(monkeypatch):
    """RED on 341d089831: KV complete, no recurrent state at its end -> the
    ack said told (17406 on the metal) and the group died."""
    try:
        cache, _ = _completed(monkeypatch, with_anchor=False)
        own = fb.own_prefix(_follower(cache), _req(), REQ, TOLD)
        assert own < TOLD, own
    finally:
        binding_state().reset()


def test_a_follower_with_its_anchor_acks_told(monkeypatch):
    """No false fallback: the anchored span acks exactly told."""
    try:
        cache, _ = _completed(monkeypatch, with_anchor=True)
        assert fb.own_prefix(_follower(cache), _req(), REQ, TOLD) == TOLD
    finally:
        binding_state().reset()
