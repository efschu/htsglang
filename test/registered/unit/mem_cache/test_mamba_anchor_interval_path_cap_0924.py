"""27B line, 24.09. -- the rest of the user's anchor decision ("1 ja 2 ja 3 ja
4 spaeter"; 1 = Agent B's c255e10ddb, 4 = int8 pool, deferred):

  (B) point 3: group P's anchors "simply spread wider, not after every chunk"
      -- one every SGLANG_WEG2_MAMBA_ANCHOR_INTERVAL tokens (4096) whatever the
      chunk size, plus at the request end (#1481 N-1 stays), NO rigid grid: a
      WRITE-side thinning, every anchor stays matchable; forks keep theirs
      under the cap;
  (A) point 2: an upper bound per root-to-tail path
      (SGLANG_WEG2_MAMBA_MAX_STATES_PER_PATH = 4, upstream 1417345f5f
      `_evict_excess_path_states`): the shallowest anchors beyond it are taken,
      tail / forks / the end anchor stay, the KV always stays.

Hermetic, CPU, a REAL UnifiedRadixCache with the REAL MambaComponent, allocator
and pools (the #924/#773 harness shape); the prefill is driven chunk by chunk
through the shipped `cache_unfinished_req` / `cache_finished_req`.
"""

from array import array
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.environ import envs
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.mamba_ckpt_utils import (
    ANCHOR_INTERVAL_ENV,
    ANCHOR_STEP_END,
    ANCHOR_STEP_INTERVAL,
    MAX_STATES_PER_PATH_ENV,
    RESUME_REFUSAL_PATH_CAP,
    weg2_anchor_interval,
    weg2_anchor_step,
    weg2_max_states_per_path,
)
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    ComponentType,
    EvictLayer,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

MAMBA_SLOTS = 20
KV_SIZE = 512
NUM_LAYERS = 8
FULL_LAYER_IDS = (3, 7)
MAMBA_LAYER_IDS = [i for i in range(NUM_LAYERS) if i not in FULL_LAYER_IDS]

N = 21            # prompt length
CHUNK = 2         # chunked_prefill_size of the test ("512" scaled down)
INTERVAL = 8      # anchor spacing of the test ("4096" scaled down: every 4th chunk)
UNFINISHED = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]   # 20 = N-1, the #1481 split
PROMPT = list(range(1000, 1000 + N))


# -- switches -----------------------------------------------------------------


@pytest.fixture
def group_p(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
    monkeypatch.delenv(ANCHOR_INTERVAL_ENV, raising=False)
    monkeypatch.delenv(MAX_STATES_PER_PATH_ENV, raising=False)
    return monkeypatch


def test_switches_default_off_and_group_p_only():
    assert weg2_anchor_interval({}) == 0
    assert weg2_max_states_per_path({}) == -1
    p = {"SGLANG_WEG2_GROUP": "P"}
    assert weg2_anchor_interval(p) == 0, "default off"
    assert weg2_max_states_per_path(p) == -1, "default off"
    for bad in ("", "0", "-4", "abc", "4.5"):
        assert weg2_anchor_interval({**p, ANCHOR_INTERVAL_ENV: bad}) == 0
        assert weg2_max_states_per_path({**p, MAX_STATES_PER_PATH_ENV: bad}) == -1
    assert weg2_anchor_interval({**p, ANCHOR_INTERVAL_ENV: "4096"}) == 4096
    assert weg2_max_states_per_path({**p, MAX_STATES_PER_PATH_ENV: "4"}) == 4
    d = {"SGLANG_WEG2_GROUP": "D", ANCHOR_INTERVAL_ENV: "4096", MAX_STATES_PER_PATH_ENV: "4"}
    assert weg2_anchor_interval(d) == 0 and weg2_max_states_per_path(d) == -1, "group P only"


# -- the spacing, as arithmetic (the expectation the boot is read against) -----


def _anchored(prompt_len, chunk, interval, start=0):
    """Chunk boundaries of an unfinished prefill that keep an anchor, with the
    #1481 split at N-1 (the distance rule reads the LAST anchor, as
    `cache_protected_len` does in the tree)."""
    bounds = list(range(start + chunk, prompt_len - 1, chunk)) + [prompt_len - 1]
    last, out = start, []
    for pos in bounds:
        why = weg2_anchor_step(pos, prompt_len, last, interval)
        if why is not None:
            out.append((pos, why))
            last = pos
    return out


@pytest.mark.parametrize("prompt_len,per_prompt", [(98_304, 25), (98_000, 25), (262_144, 65)])
def test_anchor_count_per_prompt_does_not_depend_on_the_chunk_size(prompt_len, per_prompt):
    at_4096 = _anchored(prompt_len, 4096, 4096)
    at_512 = _anchored(prompt_len, 512, 4096)
    # + the finished insert's anchor at N (never thinned)
    assert len(at_4096) + 1 == per_prompt
    assert len(at_512) + 1 == per_prompt
    assert [p for p, _ in at_512 if p % 4096 == 0] == [4096 * k for k in range(1, per_prompt - 1)]
    assert at_512[-1] == (prompt_len - 1, ANCHOR_STEP_END), "the #1481 hand-back anchor N-1"
    # the per-node law at 512-token chunks: every boundary
    per_node_512 = len(range(512, prompt_len - 1, 512)) + 1 + 1
    assert per_node_512 == {98_304: 193, 98_000: 193, 262_144: 513}[prompt_len]


def test_after_a_prefix_hit_the_spacing_counts_from_the_resume_anchor():
    got = _anchored(20_000, 512, 4096, start=1234)
    assert got[0] == (1234 + 4096, ANCHOR_STEP_INTERVAL), "every 8th 512-chunk after 1234"
    assert all(b - a >= 4096 for (a, _), (b, _) in zip(got, got[1:-1]))


def test_the_rule_reads_only_rank_uniform_inputs():
    """Forwarded extent, tree-owned prefix after PP0's geometry, prompt length,
    switch -- no rank-local match depth (on P only PP0 reads the store)."""
    import inspect

    assert list(inspect.signature(weg2_anchor_step).parameters) == [
        "pos", "prompt_len", "last_anchor", "interval"]
    assert weg2_anchor_step(8999, 9000, 8000, 4096) == ANCHOR_STEP_END
    assert weg2_anchor_step(8998, 9000, 8000, 4096) is None


# -- the live tree ------------------------------------------------------------


def _fixture(chunk=CHUNK):
    server_args = ServerArgs(model_path="dummy", page_size=1)
    server_args._mamba_cache_chunk_size = FLA_CHUNK_SIZE
    server_args.chunked_prefill_size = chunk
    set_global_server_args_for_scheduler(server_args)
    with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
        shape = Mamba2StateShape.create(
            tp_world_size=1, intermediate_size=256, n_groups=1, num_heads=2,
            head_dim=16, state_size=16, conv_kernel=4,
        )
        cache_params = Mamba2CacheParams(shape=shape, layers=MAMBA_LAYER_IDS)
    pool = HybridReqToTokenPool(
        size=10, mamba_size=MAMBA_SLOTS, mamba_spec_state_size=10, max_context_len=256,
        device="cpu", enable_memory_saver=False, cache_params=cache_params,
        mamba_layer_ids=MAMBA_LAYER_IDS, enable_mamba_extra_buffer=False,
        speculative_num_draft_tokens=3,
    )
    kv_pool = HybridLinearKVPool(
        size=KV_SIZE, dtype=torch.bfloat16, page_size=1, head_num=2, head_dim=64,
        full_attention_layer_ids=list(FULL_LAYER_IDS), device="cpu",
        enable_memory_saver=False, mamba_pool=pool.mamba_pool,
    )
    allocator = TokenToKVPoolAllocator(
        size=KV_SIZE, dtype=torch.bfloat16, device="cpu", kvcache=kv_pool, need_sort=False,
    )
    params = CacheInitParams(
        req_to_token_pool=pool, token_to_kv_pool_allocator=allocator, page_size=1,
        disable=False, sliding_window_size=None,
        tree_components=(ComponentType.FULL, ComponentType.MAMBA),
        enable_mamba_extra_buffer=False, enable_kv_cache_events=False,
        eviction_policy="lru", is_eagle=False,
    )
    cache = UnifiedRadixCache(params=params)
    cache.cache_init_params = params
    return SimpleNamespace(
        cache=cache, allocator=allocator, pool=pool,
        kv0=allocator.available_size(), m0=pool.mamba_allocator.available_size(),
    )


def _req(fx, rid, prompt):
    req = Req(rid=rid, origin_input_text="", origin_input_ids=array("q", prompt),
              sampling_params=SamplingParams(temperature=0, max_new_tokens=1))
    fx.pool.alloc([req])
    req.output_ids = array("q")
    req.full_untruncated_fill_ids = array("q", prompt)
    req.swa_uuid_for_lock = None
    req.extra_key = None
    # the admission: resume at the deepest anchor (PP0's geometry on a PP group)
    mr = fx.cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", prompt))))
    req.prefix_indices = mr.device_indices.to(torch.int64)
    req.cache_protected_len = len(req.prefix_indices)
    req.key_match_depth = mr.key_match_depth
    req.last_node = mr.last_device_node
    fx.cache.inc_lock_ref(req.last_node)
    if len(req.prefix_indices):
        fx.pool.write((req.req_pool_idx, slice(0, len(req.prefix_indices))), req.prefix_indices)
    return req


def _chunk(fx, req, end):
    start = len(req.prefix_indices)
    fx.pool.write((req.req_pool_idx, slice(start, end)), fx.allocator.alloc(end - start))
    req.set_extend_range(start, end)
    fx.cache.cache_unfinished_req(req, chunked=True)


def _finish(fx, req):
    n, start = len(req.origin_input_ids), len(req.prefix_indices)
    if start < n:
        fx.pool.write((req.req_pool_idx, slice(start, n)), fx.allocator.alloc(n - start))
    req.set_extend_range(start, n)
    req.kv_committed_len = n
    req.kv_allocated_len = n
    fx.cache.cache_finished_req(req, is_insert=True)


def _prefill(fx, rid, prompt, bounds):
    req = _req(fx, rid, prompt)
    for end in bounds:
        if end > len(req.prefix_indices):
            _chunk(fx, req, end)
    _finish(fx, req)
    return req


def _nodes(cache):
    out, stack = [], [(c, len(c.key)) for c in cache.root_node.children.values()]
    while stack:
        node, depth = stack.pop()
        out.append((depth, node))
        stack.extend((c, depth + len(c.key)) for c in node.children.values())
    return sorted(out, key=lambda x: x[0])


def _anchor_depths(cache):
    return [d for d, n in _nodes(cache) if n.component_data[ComponentType.MAMBA].value is not None]


def _node_at(cache, depth, prompt=PROMPT):
    """The node of `prompt`'s path that ends at `depth`."""
    hits = []
    for d, n in _nodes(cache):
        if d != depth:
            continue
        toks, x = [], n
        while x is not None and x is not cache.root_node:
            toks[:0] = list(x.key.raw_token_ids())
            x = x.parent
        if toks == list(prompt[:depth]):
            hits.append(n)
    assert len(hits) == 1, (depth, [d for d, _ in _nodes(cache)])
    return hits[0]


def _resume(cache, prompt, k):
    return len(cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", prompt[:k])))).device_indices)


def _assert_owned_once(fx):
    """Every KV row and mamba slot is on the free list or in the tree, never both."""
    rows = [n.component_data[ComponentType.FULL].value for _, n in _nodes(fx.cache)]
    rows = torch.cat([r for r in rows if r is not None]) if rows else torch.empty(0)
    assert len(set(rows.tolist())) == len(rows)
    assert fx.allocator.available_size() + len(rows) == fx.kv0
    slots = [n.component_data[ComponentType.MAMBA].value for _, n in _nodes(fx.cache)]
    held = sum(len(s) for s in slots if s is not None)
    assert fx.pool.mamba_allocator.available_size() + held == fx.m0


def test_the_spacing_anchors_every_interval_and_the_end_and_keeps_the_kv(group_p):
    group_p.setenv(ANCHOR_INTERVAL_ENV, str(INTERVAL))
    fx = _fixture()
    req = _req(fx, "iv", PROMPT)
    seen = []
    for end in UNFINISHED:
        before = req.cache_protected_len
        _chunk(fx, req, end)
        anchored = req.cache_protected_len != before
        seen.append((end, anchored))
        # a declined boundary inserts nothing and the request keeps its rows
        assert len(req.prefix_indices) == end
        if not anchored:
            assert end not in _anchor_depths(fx.cache)
    assert [e for e, a in seen if a] == [8, 16, 20], seen
    _finish(fx, req)
    assert _anchor_depths(fx.cache) == [8, 16, 20, 21]
    assert [_resume(fx.cache, PROMPT, k) for k in (7, 12, 19, 20, 21)] == [0, 8, 16, 20, 21]
    _assert_owned_once(fx)


def test_the_switch_off_is_the_per_node_default(group_p):
    fx = _fixture()
    _prefill(fx, "off", PROMPT, UNFINISHED)
    assert _anchor_depths(fx.cache) == UNFINISHED + [21]
    _assert_owned_once(fx)


def test_group_d_is_unchanged(group_p):
    group_p.setenv("SGLANG_WEG2_GROUP", "D")
    group_p.setenv(ANCHOR_INTERVAL_ENV, str(INTERVAL))
    group_p.setenv(MAX_STATES_PER_PATH_ENV, "2")
    fx = _fixture()
    _prefill(fx, "d", PROMPT, UNFINISHED)
    assert _anchor_depths(fx.cache) == UNFINISHED + [21]
    assert not any(getattr(n, "_weg2_capped", False) for _, n in _nodes(fx.cache))


def test_the_cap_takes_the_shallowest_anchors_the_kv_stays(group_p, monkeypatch):
    from sglang.srt.mem_cache import unified_radix_cache as urc

    monkeypatch.setattr(urc, "_WEG2_END_ANCHOR", True)   # the launcher sets it on P
    group_p.setenv(ANCHOR_INTERVAL_ENV, str(INTERVAL))
    group_p.setenv(MAX_STATES_PER_PATH_ENV, "2")
    fx = _fixture()
    _prefill(fx, "cap", PROMPT, UNFINISHED)
    n8, n16, n20, n21 = (_node_at(fx.cache, d) for d in (8, 16, 20, 21))
    assert n20._weg2_end_anchor, "#1481 mark"
    assert (n8._weg2_capped, n16._weg2_capped) == (True, True)
    assert not getattr(n20, "_weg2_capped", False) and not getattr(n21, "_weg2_capped", False)
    # their device slots went back, the KV did not
    assert _anchor_depths(fx.cache) == [20, 21]
    assert all(n.component_data[ComponentType.FULL].value is not None for n in (n8, n16))
    # no resume point below the end anchor any more; N-1 and N still serve
    assert [_resume(fx.cache, PROMPT, k) for k in (12, 19, 20, 21)] == [0, 0, 20, 21]
    mc = fx.cache.components[ComponentType.MAMBA]
    assert mc.explain_match_refusal(n8, 8) == RESUME_REFUSAL_PATH_CAP
    _assert_owned_once(fx)


def test_forks_and_the_end_anchor_stay(group_p, monkeypatch):
    from sglang.srt.mem_cache import unified_radix_cache as urc

    monkeypatch.setattr(urc, "_WEG2_END_ANCHOR", True)
    group_p.setenv(ANCHOR_INTERVAL_ENV, str(INTERVAL))
    group_p.setenv(MAX_STATES_PER_PATH_ENV, "10")    # armed from the start, nothing over it yet
    fx = _fixture()
    _prefill(fx, "a", PROMPT, UNFINISHED)
    # a second prompt continuing from the anchor at 8 makes node 8 a FORK
    b_prompt = PROMPT[:8] + list(range(3000, 3000 + N - 8))
    _prefill(fx, "b", b_prompt, [10, 12, 14, 16, 18, 20])
    n8 = _node_at(fx.cache, 8)
    assert len(n8.children) == 2
    assert not any(getattr(n, "_weg2_capped", False) for _, n in _nodes(fx.cache))
    group_p.setenv(MAX_STATES_PER_PATH_ENV, "1")
    # a third prompt on a's path pushes it over the bound
    c_prompt = PROMPT + list(range(5000, 5010))
    _prefill(fx, "c", c_prompt, [22, 24, 26, 28, 30])
    taken = sorted(d for d, n in _nodes(fx.cache) if getattr(n, "_weg2_capped", False))
    assert taken == [16, 21], taken      # a's inner anchor and a's old N; c's N-1 is its end anchor
    assert not getattr(n8, "_weg2_capped", False), "a fork keeps its anchor"
    assert _resume(fx.cache, b_prompt, 12) == 8
    assert not getattr(_node_at(fx.cache, 20), "_weg2_capped", False), "the end anchor stays"
    _assert_owned_once(fx)


def test_a_child_only_pp0_has_does_not_make_a_fork(group_p):
    """Only PP0 reads the store on group P: a host-prefetched child can hang
    below a node on PP0 alone. The fork the cap keeps is the one a DEVICE
    insert made (every rank, same step) -- `len(children)` would keep node 8 on
    PP0 and take it on its peers."""
    group_p.setenv(ANCHOR_INTERVAL_ENV, str(INTERVAL))
    group_p.setenv(MAX_STATES_PER_PATH_ENV, "2")
    ranks = [_fixture(), _fixture()]
    reqs = [_req(fx, "h", PROMPT) for fx in ranks]
    for end in [2, 4, 6, 8, 10, 12, 14, 16]:
        for fx, r in zip(ranks, reqs):
            _chunk(fx, r, end)
    pp0 = ranks[0].cache
    n8 = _node_at(pp0, 8)
    stray = SimpleNamespace(parent=n8, children={}, key=None)   # PP0's host-prefetched child
    n8.children["pp0-host-only"] = stray
    for end in [18, 20]:
        for fx, r in zip(ranks, reqs):
            _chunk(fx, r, end)
    del n8.children["pp0-host-only"]
    taken = [[d for d, n in _nodes(fx.cache) if getattr(n, "_weg2_capped", False)] for fx in ranks]
    assert taken[0] == taken[1] == [8]


def test_a_locked_anchor_is_no_resume_point_at_once_its_slot_goes_when_free(group_p):
    group_p.setenv(ANCHOR_INTERVAL_ENV, str(INTERVAL))
    group_p.setenv(MAX_STATES_PER_PATH_ENV, "2")
    fx = _fixture()
    req = _req(fx, "lock", PROMPT)
    for end in [2, 4, 6, 8, 10, 12, 14, 16]:
        _chunk(fx, req, end)
    n8 = _node_at(fx.cache, 8)
    fx.cache.inc_lock_ref(n8)          # a reader (or a write-through) holds it
    for end in [18, 20]:
        _chunk(fx, req, end)           # the anchor at 20 pushes 8 over the bound
    assert n8._weg2_capped
    assert n8.component_data[ComponentType.MAMBA].value is not None, "the copy waits for the lock"
    assert n8.id in fx.cache._weg2_cap_deferred
    assert _resume(fx.cache, PROMPT, 12) == 0, "but it is no resume point any more"
    fx.cache.dec_lock_ref(n8)
    assert fx.cache._weg2_cap_drain() == 1
    assert n8.component_data[ComponentType.MAMBA].value is None
    _finish(fx, req)
    _assert_owned_once(fx)


def test_the_cap_decides_the_same_on_every_rank_whatever_a_rank_local_lock_says(group_p):
    """raenge-nie-uneins: a write-through acks at a different time on every rank
    (#737, rank-local drain). A cap that skipped the locked node would take a
    DIFFERENT anchor on the rank whose write is still in flight -- and the
    forwarded admission of the next request would meet local < told."""
    group_p.setenv(ANCHOR_INTERVAL_ENV, str(INTERVAL))
    group_p.setenv(MAX_STATES_PER_PATH_ENV, "2")
    ranks = [_fixture(), _fixture()]
    reqs = [_req(fx, "u", PROMPT) for fx in ranks]
    for end in [2, 4, 6, 8, 10, 12, 14, 16]:
        for fx, r in zip(ranks, reqs):
            _chunk(fx, r, end)
    slow = ranks[1]
    slow.cache.inc_lock_ref(_node_at(slow.cache, 8))     # this rank's write still in flight
    for end in [18, 20]:
        for fx, r in zip(ranks, reqs):
            _chunk(fx, r, end)
    probes = [PROMPT[:k] for k in range(1, N + 1)] + [PROMPT[:13] + [1, 2, 3]]
    for p in probes:
        assert _resume(ranks[0].cache, p, len(p)) == _resume(slow.cache, p, len(p)), p
    taken = [[d for d, n in _nodes(fx.cache) if getattr(n, "_weg2_capped", False)] for fx in ranks]
    assert taken[0] == taken[1] == [8]


def test_an_insert_at_a_taken_anchor_makes_it_an_anchor_again(group_p):
    group_p.setenv(ANCHOR_INTERVAL_ENV, str(INTERVAL))
    group_p.setenv(MAX_STATES_PER_PATH_ENV, "2")
    fx = _fixture()
    _prefill(fx, "x", PROMPT, UNFINISHED)
    n8 = _node_at(fx.cache, 8)
    assert n8._weg2_capped and _resume(fx.cache, PROMPT, 12) == 0
    slot = fx.pool.mamba_allocator.alloc(1)
    res = fx.cache.insert(InsertParams(key=RadixKey(array("q", PROMPT[:8])),
                                       value=fx.allocator.alloc(8), mamba_value=slot))
    if res.mamba_exist:
        fx.pool.mamba_allocator.free(slot)
    assert not n8._weg2_capped and n8._weg2_anchored
    assert _resume(fx.cache, PROMPT, 12) == 8
    _assert_owned_once(fx)


def test_the_reset_drops_waiting_releases(group_p):
    group_p.setenv(ANCHOR_INTERVAL_ENV, str(INTERVAL))
    group_p.setenv(MAX_STATES_PER_PATH_ENV, "1")
    fx = _fixture()
    req = _req(fx, "r", PROMPT)
    for end in [2, 4, 6, 8]:
        _chunk(fx, req, end)
    fx.cache.inc_lock_ref(_node_at(fx.cache, 8))
    for end in [10, 12, 14, 16]:
        _chunk(fx, req, end)
    assert fx.cache._weg2_cap_deferred
    fx.cache.reset()
    assert fx.cache._weg2_cap_deferred == {} and fx.cache._weg2_cap_tail is None


def test_the_ack_drains_the_waiting_release():
    import inspect

    src = inspect.getsource(UnifiedRadixCache._finish_write_through_ack)
    assert src.index("self.dec_lock_ref(lock_node, lock_params)") < src.index("self._weg2_cap_drain()")


def test_an_anchor_awaiting_its_publish_keeps_its_device_copy(group_p):
    """With a host tier, a taken anchor not yet in the arena keeps its device
    slot until the publish took it (the arena copy is the prefix cache the
    cap leaves behind)."""
    group_p.setenv(MAX_STATES_PER_PATH_ENV, "2")
    fx = _fixture()
    mc = fx.cache.components[ComponentType.MAMBA]
    _prefill(fx, "pub", PROMPT, [8, 16, 20])
    n8 = _node_at(fx.cache, 8)
    assert n8._weg2_capped
    # re-create the situation with a controller present and n8 unpublished
    slot = fx.pool.mamba_allocator.alloc(1)
    n8.component_data[ComponentType.MAMBA].value = slot
    fx.cache.component_evictable_size_[ComponentType.MAMBA] += 1
    fx.cache.cache_controller = SimpleNamespace()
    try:
        assert fx.cache._weg2_cap_release(n8, mc) is False
        n8.component_data[ComponentType.FULL].host_value = torch.tensor([0])  # published
        assert fx.cache._weg2_cap_release(n8, mc) is True
        assert n8.component_data[ComponentType.MAMBA].value is None
    finally:
        fx.cache.cache_controller = None
        n8.component_data[ComponentType.FULL].host_value = None
    assert EvictLayer.ALL == EvictLayer.DEVICE | EvictLayer.HOST


def test_a_copy_lost_on_one_rank_does_not_change_what_the_cap_takes(group_p):
    """The device LRU's victim follows the write-through locks and Agent B's
    release follows the ack -- both rank-local. Counting holders by the copies
    would give a rank that lost node 8's copy a different excess than its
    peers; the insert-set flag gives both the same."""
    group_p.setenv(ANCHOR_INTERVAL_ENV, str(INTERVAL))
    group_p.setenv(MAX_STATES_PER_PATH_ENV, "2")
    ranks = [_fixture(), _fixture()]
    reqs = [_req(fx, "l", PROMPT) for fx in ranks]
    for end in [2, 4, 6, 8, 10, 12, 14, 16]:
        for fx, r in zip(ranks, reqs):
            _chunk(fx, r, end)
    lossy = ranks[1].cache
    n8 = _node_at(lossy, 8)
    lossy._evict_component_and_detach_lru(
        n8, lossy.components[ComponentType.MAMBA], target=EvictLayer.DEVICE, tracker=None)
    assert n8.component_data[ComponentType.MAMBA].value is None
    for end in [18, 20]:
        for fx, r in zip(ranks, reqs):
            _chunk(fx, r, end)
    taken = [[d for d, n in _nodes(fx.cache) if getattr(n, "_weg2_capped", False)] for fx in ranks]
    assert taken[0] == taken[1] == [8]
