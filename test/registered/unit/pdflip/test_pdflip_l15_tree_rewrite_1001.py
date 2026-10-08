"""AP L15-11d: retain must rewrite the REAL radix tree.

Red without the fix: the old step (4) of l15_retain.retain_at_sleep wrote
``node.kv_slots`` / ``node.anchor_slot`` -- attributes that exist only on
the unit-test fakes. The real UnifiedTreeNode has neither field, so the
write was silently accepted and never read: after a retain round the KV
rows and the anchors moved to their new slots while the tree still pointed
at the OLD ones, and the next prefix hit would read foreign KV.

Green with the fix: l15_bind.rewrite_tree_chain remaps the component
values of every node on root -> last_node (the FULL value element-wise,
the mamba value on the anchor node) and remaps each node exactly once
(visited set shared across the held requests of one retain call).

CPU-only fixture (FULL + Mamba hybrid tree) built exactly like
test_pdflip_l15_reset_keep_0930.py. Plain pytest functions: the
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
from flliper.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from flliper.srt.mem_cache.cache_init_params import CacheInitParams
from flliper.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from flliper.srt.mem_cache.radix_cache import RadixKey
from flliper.srt.mem_cache.unified_cache_components.tree_component import (
    ComponentType,
)
from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from flliper.srt.sampling.sampling_params import SamplingParams
from flliper.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from flliper.srt.pdflip.l15_bind import rewrite_tree_chain

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


def _chain_slots(fx, ids):
    """The matched chain's FULL slot indices in token order."""
    last = _match(fx, ids).last_device_node
    return [
        int(s)
        for n in reversed(_chain(last, fx.cache.root_node))
        for s in n.component_data[ComponentType.FULL].value.tolist()
    ]


IDS = list(range(100, 112))


def test_rewrite_tree_chain_remapped_match_returns_new_slots():
    fx = _fixture()
    last = _insert(fx, "R1", IDS)
    assert last is not fx.cache.root_node
    old_flat = _chain_slots(fx, IDS)
    assert old_flat == _match(fx, IDS).device_indices.tolist()
    anchor_old = int(last.component_data[ComponentType.MAMBA].value.flatten()[0])

    # move two of the chain's slots (first and last token) plus the anchor
    kv_map = {old_flat[0]: 400, old_flat[-1]: 401}
    anchor_map = {anchor_old: 402}

    rewrite_tree_chain(last, kv_map, anchor_map)

    want = [kv_map.get(s, s) for s in old_flat]
    got = _match(fx, IDS).device_indices.tolist()
    assert got == want, (got, want)
    # the mamba value on the anchor node is the new anchor
    assert int(last.component_data[ComponentType.MAMBA].value.flatten()[0]) == 402


def test_rewrite_tree_chain_remaps_each_node_once_across_chains():
    fx = _fixture()
    a_ids = list(range(200, 208))
    b_ids = a_ids[:4] + [210, 211, 212, 213]
    _insert(fx, "RA", a_ids)
    _insert(fx, "RB", b_ids)

    # The shared prefix node is b's last node's parent (bigram keys: 4
    # shared tokens are 3 UNITS, so the shared length is 3, not 4).
    b_last = _match(fx, b_ids).last_device_node
    shared = b_last.parent
    assert shared is not None and shared is not fx.cache.root_node
    shared_len = len(shared.component_data[ComponentType.FULL].value)

    old_a, old_b = _chain_slots(fx, a_ids), _chain_slots(fx, b_ids)
    assert old_a[:shared_len] == old_b[:shared_len], "chains share a prefix"
    assert shared_len >= 2, "two moved slots must fit on the shared node"

    # The map is NOT idempotent: old_a[0] -> old_a[1] -> 402. A double remap
    # of the shared node would put 402 where old_a[1] belongs.
    kv_map = {old_a[0]: old_a[1], old_a[1]: 402}
    # both requests' anchors are HELD (identity: they do not move) -- since
    # L15-FIX-MAMBA-ALIAS the rewrite drops any chain mamba value outside
    # the held-anchor map, so an empty map would strip the anchors too
    held = {}
    for ids in (a_ids, b_ids):
        mv = _match(fx, ids).last_device_node.component_data[
            ComponentType.MAMBA].value
        held.update({int(x): int(x) for x in mv.flatten().tolist()})
    visited = set()
    rewrite_tree_chain(_match(fx, a_ids).last_device_node, kv_map, held, visited)
    rewrite_tree_chain(_match(fx, b_ids).last_device_node, kv_map, held, visited)

    got_a = _match(fx, a_ids).device_indices.tolist()
    got_b = _match(fx, b_ids).device_indices.tolist()
    assert got_a == [kv_map.get(s, s) for s in old_a]
    assert got_b == [kv_map.get(s, s) for s in old_b]
    # exactly-once on the shared prefix (a double pass would yield 402)
    assert got_a[0] == old_a[1]
