"""GRANT-SUM + ACK-ROOM (NVFP4 dual1k dkr27bnvfp4dual1kbar1fs10010950, image
y6e = freeze-final 90945aeba9, PP1 death 09:55:20Z, rid weg2-0-10).

MEASURED (P log):

    PP0  DUAL-TP3PP3 P-KV PP0 GRANT rid=weg2-0-9  tokens=61440 on all 3 cards
    PP0  DUAL-TP3PP3 P-KV PP0 GRANT rid=weg2-0-10 tokens=20480 on all 3 cards
    PP1  MAPPED-BY-GRANT tokens=20480 ... returned=83886080 B   (level stays 61440)
    PP0  #TW TWIN-TOLD rid=weg2-0-10 head=16383 told=16383; PF TOLD-ACKED acks={1: 16383, 2: 16383}
    PP0/PP2  resident=16383 (the twin head on the device);  PP1 resident=0 host_hit=16383
    PP1  SF LOADBACK-ROOM PP-RESIDUAL kv_tokens=12288 avail=1717 evictable=0
    PP1  #968 PREFIX MATERIALISATION SHORTFALL prefix_len=16383, holds 0 after 0.00 s

Root: the dual P KV mapping is ONE high-water level shared by every request
P holds, and each grant was sized for its own prompt only -- weg2-0-10's 20480
fit "under" weg2-0-9's 61440 level although weg2-0-9's prefill was filling it.
PP1, the one stage that held the twin head on its host only, had to load it
back and found no room. Not an anchor depth off-by-one (16383 on every rank).

  (a) GRANT-SUM: PP0's grant covers this prompt plus every other request that
      holds a grant (dual_p_kv_stage.live_grant_tokens).
  (b) ACK-ROOM: a follower whose told span is host-only and does not fit its
      pool even with every evictable row freed acks 0 -> PP0 sends told=0 to
      every rank (PF), the twin path included (its acks are PF acks).
"""

import importlib.util
import json
import os
import tempfile
from array import array
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from sglang.srt.managers import weg2_told_fallback as fb  # noqa: E402
from sglang.srt.mem_cache.hicache_phase_binding import binding_state  # noqa: E402
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer  # noqa: E402
from sglang.srt.mem_cache.radix_cache import RadixKey  # noqa: E402
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType  # noqa: E402
from sglang.srt.weg2 import dual_p_kv_stage as S  # noqa: E402

from test_unified_radix_cache_unittest import CacheConfig, build_fixture  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_t_1157_dual_room", os.path.join(os.path.dirname(__file__), "test_1157_reaper_prices_requested_span.py"))
h1157 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h1157)

FULL, MAMBA = ComponentType.FULL, ComponentType.MAMBA
REQ = h1157.REAP_REQ
TOLD = h1157.REAP_TOKENS


def _req(rid, n, granted=0):
    r = SimpleNamespace(rid=rid, origin_input_ids=list(range(n)))
    if granted:
        r._dual_kv_tokens = granted
    return r


# --- (a) GRANT-SUM ------------------------------------------------------------------------


def test_live_grant_tokens_counts_every_other_granted_request_once():
    r9 = _req("weg2-0-9", 61440, granted=61440)
    r10 = _req("weg2-0-10", 19681)
    plain = _req("weg2-0-8", 500)                      # no grant: not counted
    sched = SimpleNamespace(running_batch=SimpleNamespace(reqs=[r9]), running_mbs=[SimpleNamespace(reqs=[r9])],
                            chunked_req=None, waiting_queue=[plain, r10], _weg2_store_held={"weg2-0-10": r10})
    assert S.live_grant_tokens(sched, r10, 64) == 61440 + 64


def test_pp0_grant_sizes_the_level_for_all_held_requests(monkeypatch):
    """RED on 90945aeba9: the grant asked 19681+64 tokens for weg2-0-10 while
    weg2-0-9 (61440) held the same high-water mapping."""
    d = tempfile.mkdtemp(prefix="wkvs")
    stages = [{"ledger": os.path.join(d, f"c{i}"), "step": 4096, "top": 196608, "bytes": [0] * 64}
              for i in range(3)]
    for i, st in enumerate(stages):
        with open(os.path.join(d, f"stage{i}"), "w") as f:
            json.dump(st, f)
    monkeypatch.setattr(S, "stage_file", lambda tag, r, root="/dev/shm": os.path.join(d, f"stage{r}"))
    mapped = []
    actor = SimpleNamespace(page=64, _committed=0, map_granted=lambda lvl, charged=None: mapped.append(lvl))
    monkeypatch.setattr(S, "_actor", lambda sched: actor)
    asked = []

    def _grant(stages_, tokens, open_ledger, covered=None):
        asked.append(tokens)
        return S.round_up(tokens, 4096)

    monkeypatch.setattr(S, "group_grant", _grant)
    r9 = _req("weg2-0-9", 61440, granted=61440)
    r10 = _req("weg2-0-10", 19681)
    sched = SimpleNamespace(ps=SimpleNamespace(pp_rank=0, pp_size=3),
                            running_batch=SimpleNamespace(reqs=[r9]), waiting_queue=[r10])
    lvl = S.pp0_grant(sched, r10)
    assert asked == [19681 + 64 + 61440 + 64]
    assert lvl == S.round_up(19681 + 64 + 61440 + 64, 4096) and r10._dual_kv_tokens == lvl


# --- (b) ACK-ROOM -------------------------------------------------------------------------


def _mamba_tree():
    cache = build_fixture(CacheConfig(page_size=h1157.PAGE_SIZE, components=(FULL, MAMBA)))[0]
    alloc = cache.token_to_kv_pool_allocator
    full = alloc.get_kvcache().full_kv_pool
    alloc.get_kvcache = lambda: full
    return cache


def _host_only_span_with_anchor(monkeypatch):
    monkeypatch.setattr(h1157, "_build_cache", _mamba_tree)
    cache, op = h1157._reap_scenario(probed=True)
    e = cache.ongoing_prefetch[REQ]
    cache.ongoing_prefetch[REQ] = type(e)(
        e[0], RadixKey(array("q", list(e[1].token_ids))), e[2], e[3], e[4],
        {MAMBA: [PoolTransfer(name=PoolName.MAMBA, host_indices=torch.tensor([7]))]})
    op.pool_storage_result.extra_pool_hit_pages[PoolName.MAMBA] = 1
    assert cache.check_prefetch_progress(REQ)
    return cache


def _follower(cache):
    return SimpleNamespace(tree_cache=cache, _weg2_store_told={REQ: TOLD},
                           ps=SimpleNamespace(pp_rank=1, pp_size=3, tp_size=1))


def _ack_req():
    return SimpleNamespace(rid=REQ, origin_input_ids=list(range(1, TOLD + 1)), extra_key=None,
                           full_untruncated_fill_ids=None)


def test_a_follower_that_cannot_hold_the_load_back_acks_zero(monkeypatch):
    """RED on 90945aeba9: the follower acked told (host KV + anchor resume) and
    then could not load it back (SF LOADBACK-ROOM PP-RESIDUAL) -> #968."""
    try:
        cache = _host_only_span_with_anchor(monkeypatch)
        alloc = cache.token_to_kv_pool_allocator
        hog = alloc.alloc(int(alloc.available_size()) - (TOLD // 2))   # a concurrent prefill's rows
        assert hog is not None and int(alloc.available_size()) < TOLD
        assert int(cache.evictable_size()) == 0
        assert fb.own_prefix(_follower(cache), _ack_req(), REQ, TOLD) == 0
    finally:
        binding_state().reset()


def test_a_follower_with_room_acks_told(monkeypatch):
    try:
        cache = _host_only_span_with_anchor(monkeypatch)
        assert int(cache.token_to_kv_pool_allocator.available_size()) >= TOLD
        assert fb.own_prefix(_follower(cache), _ack_req(), REQ, TOLD) == TOLD
    finally:
        binding_state().reset()
