"""S56r (NF review 19.10.): the D-MATCH-PROBE reads D's tree WITHOUT TOUCHING IT, and its D-side calls coalesce.

User 19.10.: "warum wird irgendwas geschätzt? alle zahlen liegen doch schwarz auf weiß vor zu jedem zeitpunkt?" --
review question (2): the probe must have no side effect on D (no allocation, no LRU move or touch, no refcount, no
mamba slot, no hold), proven in code. S56 called ``UnifiedRadixCache.match_prefix``, which at a partial match SPLITS
a node (new node, LRU remove/insert), moves the matched chain to MRU in every component LRU and re-stamps
``last_access_time`` up to the root -- a probe for a request that then goes to P re-ordered D's eviction.

These tests run on a REAL hybrid (FULL + MAMBA) ``UnifiedRadixCache`` (CPU pools, the fixture of
test_pdflip_fork_anchor_0926): the read-only ``peek_admission_match`` must give the admission's numbers (the real
walk with the #928 refusal applied) and leave the tree byte-identical; the real walk demonstrably does not.
Review question (3): k probes arriving together cost at most two scheduler rounds, not k.
"""
from __future__ import annotations

import asyncio
import os
import types
from array import array
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape  # noqa: E402
from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE  # noqa: E402
from flliper.srt.managers.schedule_batch import Req  # noqa: E402
from flliper.srt.managers.tp_match_floor import anchor_unusable  # noqa: E402
from flliper.srt.mem_cache.allocator import TokenToKVPoolAllocator  # noqa: E402
from flliper.srt.mem_cache.base_prefix_cache import MatchPrefixParams  # noqa: E402
from flliper.srt.mem_cache.cache_init_params import CacheInitParams  # noqa: E402
from flliper.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool  # noqa: E402
from flliper.srt.mem_cache.radix_cache import RadixKey  # noqa: E402
from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType  # noqa: E402
from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache  # noqa: E402
from flliper.srt.pdflip import d_match_probe as dmp  # noqa: E402
from flliper.srt.sampling.sampling_params import SamplingParams  # noqa: E402
from flliper.srt.server_args import ServerArgs, set_global_server_args_for_scheduler  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=10, suite="stage-a-pdflip-unit")

KV_SIZE = 512
NUM_LAYERS = 8
FULL_LAYER_IDS = (3, 7)
MAMBA_LAYER_IDS = [i for i in range(NUM_LAYERS) if i not in FULL_LAYER_IDS]
MAMBA_SLOTS = 20
CHUNK = 4


def _fixture():
    server_args = ServerArgs(model_path="dummy", page_size=1)
    server_args._mamba_cache_chunk_size = FLA_CHUNK_SIZE
    server_args.chunked_prefill_size = CHUNK
    set_global_server_args_for_scheduler(server_args)
    with envs.FLLIPER_MAMBA_SSM_DTYPE.override("bfloat16"):
        shape = Mamba2StateShape.create(tp_world_size=1, intermediate_size=256, n_groups=1, num_heads=2,
                                        head_dim=16, state_size=16, conv_kernel=4)
        cache_params = Mamba2CacheParams(shape=shape, layers=MAMBA_LAYER_IDS)
    pool = HybridReqToTokenPool(size=10, mamba_size=MAMBA_SLOTS, mamba_spec_state_size=10, max_context_len=256,
                                device="cpu", enable_memory_saver=False, cache_params=cache_params,
                                mamba_layer_ids=MAMBA_LAYER_IDS, enable_mamba_extra_buffer=False,
                                speculative_num_draft_tokens=3)
    kv_pool = HybridLinearKVPool(size=KV_SIZE, dtype=torch.bfloat16, page_size=1, head_num=2, head_dim=64,
                                 full_attention_layer_ids=list(FULL_LAYER_IDS), device="cpu",
                                 enable_memory_saver=False, mamba_pool=pool.mamba_pool)
    allocator = TokenToKVPoolAllocator(size=KV_SIZE, dtype=torch.bfloat16, device="cpu", kvcache=kv_pool,
                                       need_sort=False)
    params = CacheInitParams(req_to_token_pool=pool, token_to_kv_pool_allocator=allocator, page_size=1,
                             disable=False, sliding_window_size=None,
                             tree_components=(ComponentType.FULL, ComponentType.MAMBA),
                             enable_mamba_extra_buffer=False, enable_kv_cache_events=False,
                             eviction_policy="lru", is_eagle=False)
    cache = UnifiedRadixCache(params=params)
    cache.cache_init_params = params
    return SimpleNamespace(cache=cache, allocator=allocator, pool=pool)


def _prefill(fx, ids, rid):
    """A request's admission + chunked prefill + finish (the real insert path): anchors every CHUNK tokens."""
    req = Req(rid=rid, origin_input_text="", origin_input_ids=array("q", ids),
              sampling_params=SamplingParams(temperature=0, max_new_tokens=1))
    fx.pool.alloc([req])
    req.output_ids = array("q")
    req.full_untruncated_fill_ids = array("q", ids)
    req.swa_uuid_for_lock = None
    req.extra_key = None
    mr = fx.cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", ids))))
    req.prefix_indices = mr.device_indices.to(torch.int64)
    req.cache_protected_len = len(req.prefix_indices)
    req.key_match_depth = mr.key_match_depth
    req.last_node = mr.last_device_node
    fx.cache.inc_lock_ref(req.last_node)
    for end in range(len(req.prefix_indices) + CHUNK, len(ids), CHUNK):
        start = len(req.prefix_indices)
        fx.pool.write((req.req_pool_idx, slice(start, end)), fx.allocator.alloc(end - start))
        req.set_extend_range(start, end)
        fx.cache.cache_unfinished_req(req, chunked=True)
    n, start = len(ids), len(req.prefix_indices)
    if start < n:
        fx.pool.write((req.req_pool_idx, slice(start, n)), fx.allocator.alloc(n - start))
    req.set_extend_range(start, n)
    req.kv_committed_len = n
    req.kv_allocated_len = n
    fx.cache.cache_finished_req(req, is_insert=True)
    return req


def _nodes(cache):
    out, todo = [], [cache.root_node]
    while todo:
        n = todo.pop()
        out.append(n)
        todo.extend(n.children.values())
    return out


def _lru_order(cache, ct):
    lst = cache.lru_lists[ct]
    pt = lst._pt
    out, n = [], lst.head.lru_next[pt]
    while n is not lst.tail:
        out.append(n.id)
        n = n.lru_next[pt]
    return out


def _snapshot(cache):
    """Everything a walk could write: the node set, keys, access stamps, both LRU orders, lock refs, values."""
    rows = []
    for n in _nodes(cache):
        cds = n.component_data
        rows.append((n.id, len(n.key) if n.key is not None else -1, n.last_access_time, tuple(sorted(
            str(k) for k in n.children)),
            tuple((getattr(cd, "lock_ref", None), None if cd.value is None else len(cd.value),
                   None if getattr(cd, "host_value", None) is None else len(cd.host_value)) for cd in cds)))
    return (sorted(rows, key=lambda r: r[0]), tuple(_lru_order(cache, ct) for ct in (ComponentType.FULL,
                                                                                      ComponentType.MAMBA)),
            dict(cache.component_evictable_size_))


def _admission_numbers(cache, ids):
    """What D's admission uses: the real walk (key limit n-1) + the #928 zeroing of MambaComponent's finalize."""
    mr = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", ids), limit=len(ids) - 1)))
    if mr.best_match_node is not cache.root_node and anchor_unusable(cache, mr.best_match_node):
        return 0, 0
    return len(mr.device_indices), int(mr.host_hit_length or 0)


A = list(range(1000, 1041))                    # 41 tokens
B = A[:22] + list(range(5000, 5019))           # shares 22 with A, then diverges


@pytest.fixture
def fx():
    f = _fixture()
    _prefill(f, A, "a")
    _prefill(f, B, "b")
    return f


QUERIES = {
    "a_plus_tail": A + [7, 8, 9, 10, 11],
    "a_exact": A,
    "partial_inside_a_node": A[:30] + [9, 9, 9],      # the real walk SPLITS here
    "b_prefix": B[:35] + [1],
    "cold": list(range(9000, 9020)),
    "short": A[:3],
}


@pytest.mark.parametrize("name", list(QUERIES))
def test_peek_gives_the_admission_numbers_and_writes_nothing(fx, name):
    ids = QUERIES[name]
    before = _snapshot(fx.cache)
    dev, host, depth = dmp.peek_admission_match(fx.cache, array("q", ids), None, len(ids) - 1)
    assert _snapshot(fx.cache) == before, "the probe touched D's tree (split / stamp / LRU / lock)"
    assert (dev, host) == _admission_numbers(fx.cache, ids), name
    assert depth >= dev


def test_the_real_walk_is_not_read_only_which_is_why_the_probe_must_not_call_it(fx):
    """The defect S56 had: ``match_prefix`` at a partial match adds a node and re-stamps the chain."""
    ids = QUERIES["partial_inside_a_node"]
    before = _snapshot(fx.cache)
    fx.cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", ids), limit=len(ids) - 1)))
    after = _snapshot(fx.cache)
    assert len(after[0]) == len(before[0]) + 1, "the split node"
    assert after != before


def test_peek_on_a_host_backed_path_counts_the_host_tier_like_the_admission(fx, monkeypatch):
    """HiCache shape: the deepest node of A is evicted to the host (FULL + MAMBA host values); with a cache
    controller the walk accepts host-backed anchors and the admission's host hit counts them."""
    cache = fx.cache
    leaf = None
    node = cache.root_node
    key = RadixKey(array("q", A))
    while key and key.child_key(1) in node.children:
        node = node.children[key.child_key(1)]
        key = key[len(node.key):]
        leaf = node
    full = leaf.component_data[ComponentType.FULL]
    mam = leaf.component_data[ComponentType.MAMBA]
    assert full.value is not None and mam.value is not None
    full.host_value, full.value = full.value.clone(), None
    mam.host_value, mam.value = mam.value.clone(), None
    monkeypatch.setattr(cache, "cache_controller", SimpleNamespace(write_policy="write_through"), raising=False)
    for ids in (A + [3, 3], A[:30] + [4]):
        before = _snapshot(cache)
        dev, host, _ = dmp.peek_admission_match(cache, array("q", ids), None, len(ids) - 1)
        assert _snapshot(cache) == before
        assert (dev, host) == _admission_numbers(cache, ids), ids
    dev, host, _ = dmp.peek_admission_match(cache, array("q", A + [3, 3]), None, len(A) + 1)
    assert host > 0, "the host-backed anchor counts as host tier"


def test_peek_zeroes_a_foreign_pool_anchor_exactly_like_the_admission_finalize(fx, monkeypatch):
    """#928 (b): the anchor's state bytes belong to the other phase's pool -- the admission's finalize (cow_mamba,
    the REAL MambaComponent path with a slot-holding stand-in) zeroes the match; the peek must say 0 too."""
    from flliper.srt.mem_cache import mamba_state_pool as msp
    from flliper.srt.mem_cache.unified_cache_components import mamba_component as mc

    ids = QUERIES["a_plus_tail"]
    assert dmp.peek_admission_match(fx.cache, array("q", ids), None, len(ids) - 1)[0] > 0
    monkeypatch.setattr(msp, "anchor_bytes_reachable", lambda tc, v: False)
    monkeypatch.setattr(mc, "anchor_bytes_reachable", lambda tc, v: False)
    stand_in = SimpleNamespace(rid="x~ref", mamba_pool_idx=-1, mamba_cow_src_index=None, mamba_needs_clear=True,
                               mamba_slot_acquired_this_admission=False, session=None, last_node=None,
                               extra_key=None)
    mr = fx.cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", ids), limit=len(ids) - 1),
                                                 req=stand_in, cow_mamba=True))
    assert len(mr.device_indices) == 0, "the admission's finalize refuses the foreign-pool anchor (#928 b)"
    assert dmp.peek_admission_match(fx.cache, array("q", ids), None, len(ids) - 1)[:2] == (0, 0)


def test_peek_refuses_a_component_it_has_no_read_only_rule_for():
    cache = SimpleNamespace(_components_tuple=(SimpleNamespace(component_type=ComponentType.FULL),
                                               SimpleNamespace(component_type=ComponentType.SWA)))
    with pytest.raises(dmp.PeekUnsupported):
        dmp.peek_admission_match(cache, array("q", [1, 2, 3]), None, 2)


# --------------------------------------------------------------------------- (3) the D-side calls coalesce (R2)

def test_a_burst_of_probes_costs_two_scheduler_rounds_not_k():
    from flliper.srt.managers.io_struct import (PdFlipMatchProbeReqInput, PdFlipMatchProbeReqOutput,
                                                unwrap_from_pickle, wrap_as_pickle)
    from flliper.srt.managers.tokenizer_control_mixin import TokenizerControlMixin as M

    rounds = []

    async def communicator(obj):
        batch = unwrap_from_pickle(obj.batch) if obj.batch is not None else []
        rounds.append([obj.rid] + [d["rid"] for d in batch])
        await asyncio.sleep(0.05)                       # one scheduler intake round
        out = PdFlipMatchProbeReqOutput(rid=obj.rid, success=True, device_tokens=len(obj.rid))
        out.batch = wrap_as_pickle([{"rid": d["rid"], "success": True, "device_tokens": len(d["rid"])}
                                    for d in batch]) if batch else None
        return [out]

    fake = types.SimpleNamespace(auto_create_handle_loop=lambda: None, pdflip_match_probe_communicator=communicator)
    fake._pdflip_probe_drain_loop = lambda: M._pdflip_probe_drain_loop(fake)

    async def go():
        rids = ["r1", "r22", "r333", "r4444", "r55555", "r666666"]
        outs = await asyncio.gather(*[M.pdflip_match_probe(fake, PdFlipMatchProbeReqInput(rid=r)) for r in rids])
        return rids, outs

    rids, outs = asyncio.run(go())
    assert [o.rid for o in outs] == rids and [o.device_tokens for o in outs] == [len(r) for r in rids]
    assert len(rounds) <= 2, rounds


def test_a_failed_round_fails_every_waiter_and_none_hangs():
    from flliper.srt.managers.io_struct import PdFlipMatchProbeReqInput
    from flliper.srt.managers.tokenizer_control_mixin import TokenizerControlMixin as M

    async def communicator(obj):
        await asyncio.sleep(0.01)
        raise ConnectionError("zmq gone")

    fake = types.SimpleNamespace(auto_create_handle_loop=lambda: None, pdflip_match_probe_communicator=communicator)
    fake._pdflip_probe_drain_loop = lambda: M._pdflip_probe_drain_loop(fake)

    async def go():
        return await asyncio.wait_for(asyncio.gather(
            *[M.pdflip_match_probe(fake, PdFlipMatchProbeReqInput(rid=str(i))) for i in range(4)],
            return_exceptions=True), timeout=2.0)

    res = asyncio.run(go())
    assert all(isinstance(r, ConnectionError) for r in res)


# ----------------------------------------------------------------------- A1: unpriced never passes as exact

def test_an_unpriced_media_request_is_not_context_refused_as_exact(caplog):
    """A1: the size of an unpriced media request is no exact count -- the context gate must not call it ``exact``
    (it takes the labelled chars3 path with its margin, as without a count)."""
    import logging

    from test_pdflip_front_x_exact import _front

    f = _front(True)
    f._context_len = lambda: 1000
    xx = f._x_unpriced("r", None, 5000, -1, True)            # media, no D tokenization: n is a size figure only
    # EZ-S1: the size of an unpriced media request is named n_est (an estimate, display only); with
    # FLLIPER_PDFLIP_EXACT_CLASS off it keeps the rc17 name "size_only"
    assert xx.unpriced and xx.pending == 0 and xx.n_src == "n_est" and xx.src == "d_probe_unpriced"
    with caplog.at_level(logging.WARNING, logger="pdflip.front"):
        f._context_gate("r", "x" * 30, xx, 5000)
    assert not any("by=exact" in r.getMessage() for r in caplog.records)
    text = f._x_unpriced("r2", SimpleNamespace(n=1234), 9999, 1000, False)
    assert text.n == 1234 and text.n_src == "front_ids" and text.uncached_p == 1000
