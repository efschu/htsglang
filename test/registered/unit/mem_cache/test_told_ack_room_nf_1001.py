"""ACK-ROOM for the NF line (port of the ACK-ROOM half of 27B 1cd3c5ac00,
NVFP4 dual1k dkr27bnvfp4dual1kbar1fs10010950, PP1 death 09:55:20Z, rid
pdflip-0-10).

MEASURED on 27B (P log):

    PP0  PF TOLD-ACKED told=16383 acks={1: 16383, 2: 16383}
    PP0/PP2  resident=16383 (on the device);  PP1 resident=0 host_hit=16383
    PP1  SF LOADBACK-ROOM PP-RESIDUAL kv_tokens=12288 avail=1717 evictable=0
    PP1  #968 PREFIX MATERIALISATION SHORTFALL prefix_len=16383, holds 0 after 0.00 s

The PF-told ack path (pdflip_told_fallback.own_prefix -> _resumable_own) is the
general one, not dual-only: NF runs it since the TOLD-PIN pick (2fd7585584).
A follower whose told span is HOST-only and does not fit its pool even with
every evictable row freed now acks 0 ("PF TOLD-ACK NO-ROOM") -> PP0 sends
told=0 to every rank instead of the #968 group death after admission.

GRANT-SUM (the other half of 1cd3c5ac00) is NOT ported: it lives in
pdflip/dual_p_kv_stage.py (dual TP3PP3 P-KV grant), which the NF line does not
carry and does not execute.

NF geometry (NF_PROFILE argv): --tp-size 1 --pp-size 3 --page-size 1, a
FULL+MAMBA hybrid tree; both followers (pp 1 and pp 2) are exercised. Real
``check_prefetch_progress`` (#1157 harness) on a real FULL+MAMBA tree."""

import importlib.util
import os
from array import array
from types import SimpleNamespace

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from flliper.srt.managers import pdflip_told_fallback as fb  # noqa: E402
from flliper.srt.mem_cache.hicache_phase_binding import binding_state  # noqa: E402
from flliper.srt.mem_cache.hicache_storage import PoolName, PoolTransfer  # noqa: E402
from flliper.srt.mem_cache.radix_cache import RadixKey  # noqa: E402
from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType  # noqa: E402

from test_unified_radix_cache_unittest import CacheConfig, build_fixture  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_t_1157_nf_room", os.path.join(os.path.dirname(__file__), "test_1157_reaper_prices_requested_span.py"))
h1157 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h1157)

FULL, MAMBA = ComponentType.FULL, ComponentType.MAMBA
REQ = h1157.REAP_REQ
TOLD = h1157.REAP_TOKENS

# NF geometry (NF_PROFILE.md Anhang A argv): TP1 PP3, page 1.
NF_TP, NF_PP, NF_PAGE = 1, 3, 1


def _mamba_tree():
    assert h1157.PAGE_SIZE == NF_PAGE
    cache = build_fixture(CacheConfig(page_size=NF_PAGE, components=(FULL, MAMBA)))[0]
    alloc = cache.token_to_kv_pool_allocator
    full = alloc.get_kvcache().full_kv_pool
    alloc.get_kvcache = lambda: full
    return cache


def _host_only_span_with_anchor(monkeypatch):
    """The told span sits on this rank's HOST only (KV + mamba anchor), so the
    told-fidelity probe says resumable -- the pre-port ack is told."""
    monkeypatch.setattr(h1157, "_build_cache", _mamba_tree)
    cache, op = h1157._reap_scenario(probed=True)
    e = cache.ongoing_prefetch[REQ]
    cache.ongoing_prefetch[REQ] = type(e)(
        e[0], RadixKey(array("q", list(e[1].token_ids))), e[2], e[3], e[4],
        {MAMBA: [PoolTransfer(name=PoolName.MAMBA, host_indices=torch.tensor([7]))]})
    op.pool_storage_result.extra_pool_hit_pages[PoolName.MAMBA] = 1
    assert cache.check_prefetch_progress(REQ)
    return cache


def _follower(cache, pp_rank):
    return SimpleNamespace(tree_cache=cache, _pdflip_store_told={REQ: TOLD},
                           ps=SimpleNamespace(pp_rank=pp_rank, pp_size=NF_PP, tp_size=NF_TP))


def _ack_req():
    return SimpleNamespace(rid=REQ, origin_input_ids=list(range(1, TOLD + 1)), extra_key=None,
                           full_untruncated_fill_ids=None)


@pytest.mark.parametrize("pp_rank", [1, 2])
def test_nf_follower_that_cannot_hold_the_load_back_acks_zero(monkeypatch, caplog, pp_rank):
    """RED on e1f7a22488: the follower acked told (host KV + anchor resume)
    although its pool cannot take the load-back -> PP0 admits at told, the
    follower dies SF LOADBACK-ROOM PP-RESIDUAL -> #968."""
    try:
        cache = _host_only_span_with_anchor(monkeypatch)
        alloc = cache.token_to_kv_pool_allocator
        hog = alloc.alloc(int(alloc.available_size()) - (TOLD // 2))   # a concurrent prefill's rows
        assert hog is not None and int(alloc.available_size()) < TOLD
        assert int(cache.evictable_size()) == 0
        with caplog.at_level("WARNING"):
            assert fb.own_prefix(_follower(cache, pp_rank), _ack_req(), REQ, TOLD) == 0
        assert any("PF TOLD-ACK NO-ROOM" in r.getMessage() for r in caplog.records)
    finally:
        binding_state().reset()


@pytest.mark.parametrize("pp_rank", [1, 2])
def test_nf_follower_with_room_acks_told(monkeypatch, caplog, pp_rank):
    """No false fallback: room for the load-back -> the ack is exactly told."""
    try:
        cache = _host_only_span_with_anchor(monkeypatch)
        assert int(cache.token_to_kv_pool_allocator.available_size()) >= TOLD
        with caplog.at_level("WARNING"):
            assert fb.own_prefix(_follower(cache, pp_rank), _ack_req(), REQ, TOLD) == TOLD
        assert not any("PF TOLD-ACK NO-ROOM" in r.getMessage() for r in caplog.records)
    finally:
        binding_state().reset()
