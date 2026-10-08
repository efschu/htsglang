"""AP L15-06: ``UnifiedRadixCache.reset_keep`` -- partial tree reset.

The tree half of the L1.5 hold: the caller hands the cache the last node of
every request whose chain must survive; everything else in the tree is
dropped exactly as ``reset()`` would drop it. Pure tree bookkeeping -- no
pool memory is freed here (the kept slots stay allocated in the allocator;
wired later in L15-11).

CPU-only fixture (FULL + Mamba hybrid tree) copied from
test_pdflip_end_anchor_exact_probe_0928.py. Plain pytest functions: the
CustomTestCase wrapper in srt/utils/common.py retries test bodies, which
would mask a one-shot bookkeeping inconsistency.
"""

from array import array
from types import SimpleNamespace

import torch

from flliper.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from flliper.srt.environ import envs
from flliper.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from flliper.srt.managers.schedule_batch import Req
from flliper.srt.mem_cache.allocator import TokenToKVPoolAllocator
from flliper.srt.mem_cache.base_prefix_cache import EvictParams, MatchPrefixParams
from flliper.srt.mem_cache.cache_init_params import CacheInitParams
from flliper.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from flliper.srt.mem_cache.radix_cache import RadixKey
from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from flliper.srt.sampling.sampling_params import SamplingParams
from flliper.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

PAGE = 1
KV_SIZE = 512
NUM_LAYERS = 4
FULL_LAYER_IDS = (1, 3)
MAMBA_LAYER_IDS = [i for i in range(NUM_LAYERS) if i not in FULL_LAYER_IDS]
S_END = 1.25


def _fixture(exact=True, bigram=True):
    server_args = ServerArgs(model_path="dummy", page_size=PAGE)
    server_args._mamba_cache_chunk_size = FLA_CHUNK_SIZE
    set_global_server_args_for_scheduler(server_args)
    with envs.FLLIPER_MAMBA_SSM_DTYPE.override("bfloat16"):
        shape = Mamba2StateShape.create(
            tp_world_size=1, intermediate_size=64, n_groups=1, num_heads=1,
            head_dim=16, state_size=8, conv_kernel=4,
        )
        cache_params = Mamba2CacheParams(shape=shape, layers=MAMBA_LAYER_IDS)
    pool = HybridReqToTokenPool(
        size=4, mamba_size=8, mamba_spec_state_size=4, max_context_len=KV_SIZE,
        device="cpu", enable_memory_saver=False, cache_params=cache_params,
        mamba_layer_ids=MAMBA_LAYER_IDS, enable_mamba_extra_buffer=False,
        speculative_num_draft_tokens=3,
    )
    kv_pool = HybridLinearKVPool(
        size=KV_SIZE, dtype=torch.bfloat16, page_size=1, head_num=1, head_dim=8,
        full_attention_layer_ids=list(FULL_LAYER_IDS), device="cpu",
        enable_memory_saver=False, mamba_pool=pool.mamba_pool,
    )
    allocator = TokenToKVPoolAllocator(
        size=KV_SIZE, dtype=torch.bfloat16, device="cpu", kvcache=kv_pool,
        need_sort=False,
    )
    params = CacheInitParams(
        req_to_token_pool=pool, token_to_kv_pool_allocator=allocator,
        page_size=PAGE, disable=False, sliding_window_size=None,
        tree_components=(ComponentType.FULL, ComponentType.MAMBA),
        enable_mamba_extra_buffer=False, enable_kv_cache_events=False,
        eviction_policy="lru", is_eagle=bigram,
    )
    cache = UnifiedRadixCache(params=params)
    cache.cache_init_params = params
    cache.__dict__["_pdflip_bigram_anchor_exact"] = bool(exact and bigram)
    return SimpleNamespace(cache=cache, allocator=allocator, pool=pool)


def _req(fx, ids, rid):
    req = Req(rid=rid, origin_input_text="", origin_input_ids=array("q", ids),
              sampling_params=SamplingParams(temperature=0, max_new_tokens=1))
    fx.pool.alloc([req])
    req.output_ids = array("q")
    req.full_untruncated_fill_ids = array("q", ids)
    req.swa_uuid_for_lock = None
    req.extra_key = None
    req.prefix_indices = torch.empty(0, dtype=torch.int64)
    req.cache_protected_len = 0
    req.last_node = fx.cache.root_node
    fx.cache.inc_lock_ref(req.last_node)
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
    fx.pool.mamba_pool.mamba_cache.temporal[:, req.mamba_pool_idx] = S_END
    fx.cache.cache_finished_req(req, is_insert=True)


def _match(fx, ids):
    return fx.cache.match_prefix(MatchPrefixParams(
        key=RadixKey(array("q", ids), is_bigram=fx.cache.is_eagle).page_aligned(PAGE)))


def _insert(fx, rid, ids, chunk_end=None):
    """One finished request; returns the tree node its key ends in."""
    req = _req(fx, ids, rid)
    if chunk_end is not None:
        _chunk(fx, req, chunk_end)
    _finish(fx, req)
    return _match(fx, ids).last_device_node


def _chain(node, root):
    """Nodes on the chain root -> node, root excluded, deepest first."""
    out = []
    cur = node
    while cur is not None and cur is not root:
        out.append(cur)
        cur = cur.parent
    return out


A_IDS = list(range(1100, 1108))
B_IDS = list(range(2100, 2108))
C_IDS = list(range(3100, 3106))


def _build_three(fx):
    a_node = _insert(fx, "A", A_IDS, chunk_end=4)
    b_node = _insert(fx, "B", B_IDS)
    c_node = _insert(fx, "C", C_IDS)
    return a_node, b_node, c_node


def test_reset_keep_keeps_only_the_chains():
    fx = _fixture()
    a_node, _b_node, c_node = _build_three(fx)
    a_chain = _chain(a_node, fx.cache.root_node)
    assert len(a_chain) == 2  # chunked insert: A really is a two-node chain
    # A is still admitted at the reset point: its leaf carries a lock.
    fx.cache.inc_lock_ref(a_node)
    # A stale host pointer the reset must drop: the fixture has no host
    # arena, so the release inside _reset_full is a no-op and without the
    # fix this pointer would survive into the new tree.
    a_node.component_data[ComponentType.FULL].host_value = torch.arange(3)
    a_node.write_through_pending_id = 7
    # Baseline device hit per kept request exactly as the insert stored it
    # (the bigram finish files N-1 units -- see the probe test's _n1_node).
    a_before = len(_match(fx, A_IDS).device_indices)
    c_before = len(_match(fx, C_IDS).device_indices)
    assert a_before == len(A_IDS) - 1 and c_before == len(C_IDS) - 1

    fx.cache.reset_keep([a_node, c_node])

    # (a) kept chains keep their full device hit
    assert len(_match(fx, A_IDS).device_indices) == a_before
    assert len(_match(fx, C_IDS).device_indices) == c_before
    # (b) the dropped request matches nothing
    mr_b = _match(fx, B_IDS)
    assert len(mr_b.device_indices) == 0
    assert mr_b.last_device_node is fx.cache.root_node
    # (c) every kept node is unlocked for every component and device-only
    for node in set(a_chain) | {c_node}:
        for cd in node.component_data:
            assert cd.lock_ref == 0, (node.id, cd.lock_ref)
            assert cd.host_value is None, (node.id, "stale host_value")
        assert node.backuped is False
        assert node.write_through_pending_id is None
    # (d) sizes, leaf sets and LRU lists hold exactly the kept chains
    kept = set(a_chain) | {c_node}
    exp_full = sum(
        len(n.component_data[ComponentType.FULL].value)
        for n in kept if n.component_data[ComponentType.FULL].value is not None
    )
    assert fx.cache.component_evictable_size_[ComponentType.FULL] == exp_full
    assert exp_full == a_before + c_before
    assert fx.cache.component_protected_size_[ComponentType.FULL] == 0
    assert fx.cache.evictable_device_leaves == {a_node, c_node}
    assert fx.cache.evictable_host_leaves == set()
    mamba_kept = {
        n for n in kept if n.component_data[ComponentType.MAMBA].value is not None
    }
    mamba_lru = fx.cache.lru_lists[ComponentType.MAMBA]
    assert set(mamba_lru.cache.values()) == mamba_kept
    exp_mamba = sum(len(n.component_data[ComponentType.MAMBA].value) for n in mamba_kept)
    assert fx.cache.component_evictable_size_[ComponentType.MAMBA] == exp_mamba
    for lru in fx.cache.host_lru_lists.values():
        assert len(lru.cache) == 0
    # (e) eviction runs and can evict a kept chain
    before = fx.cache.component_evictable_size_[ComponentType.FULL]
    res = fx.cache.evict(EvictParams(num_tokens=6, mamba_num=1))
    assert res.num_tokens_evicted >= 6
    after = fx.cache.component_evictable_size_[ComponentType.FULL]
    assert after < before


def test_reset_keep_empty_matches_reset():
    fx1, fx2 = _fixture(), _fixture()
    _build_three(fx1)
    _build_three(fx2)

    fx1.cache.reset_keep([])
    fx2.cache.reset()

    assert fx1.cache.root_node.children == {} == fx2.cache.root_node.children
    for ct in (ComponentType.FULL, ComponentType.MAMBA):
        assert fx1.cache.component_evictable_size_[ct] == 0
        assert fx2.cache.component_evictable_size_[ct] == 0
        assert fx1.cache.component_protected_size_[ct] == 0
    assert fx1.cache.evictable_device_leaves == set()
    assert fx1.cache.evictable_host_leaves == set()
    assert fx2.cache.evictable_device_leaves == set()
    assert fx2.cache.evictable_host_leaves == set()
    for lru_dict in (fx1.cache.lru_lists, fx1.cache.host_lru_lists,
                     fx2.cache.lru_lists, fx2.cache.host_lru_lists):
        for lru in lru_dict.values():
            assert len(lru.cache) == 0
