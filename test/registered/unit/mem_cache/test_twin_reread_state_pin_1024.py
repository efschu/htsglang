"""TWIN-REREAD review (review-twin-reread-1024.md) MUSS 1 + SOLL 3, on the REAL UnifiedRadixCache.

The twin path, scaled from the live 27B rc39a _1024_080355 PP2 shape (pdflip-0-2: source pdflip-0-1 prefilled the
shared prefix with states only at its chunk ends; the twin shares 18064, PP0 told 16384, the follower's tree holds
the KV past 16384 in ONE node and no state at 16384):

  * source: PROMPT (21 tokens) prefilled with a chunk end at 20 -> nodes 0..20 (state at 20) and 20..21;
  * twin: shares PROMPT[:18] (= 18064), own = head 8 + span 8 = 16 (= 16384, absolute);
  * the capped probe (``match_prefix`` of ids[:own]) SPLITS 0..20 at 16 -- the node at own exists, KV on the
    device, no recurrent state: resumable 0 (the live ``TOLD-ACK UNRESUMABLE kv=16384 resumable=0``);
  * S90b reads the anchor keyed by THAT node's last page hash (the hash chain of the 16 tokens, not the 20's),
    attaches it into a device slot via the LOAD_BACK commit -- and (MUSS 1) locks it under the rid: an eviction
    between ack and seat leaves it standing, the probe at the seat still answers own. Released at the #1417 exits.
"""

from __future__ import annotations

import logging
from array import array
from types import SimpleNamespace

import pytest
import torch

from flliper.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from flliper.srt.environ import envs
from flliper.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from flliper.srt.managers import cache_controller as ccmod
from flliper.srt.managers.schedule_batch import Req
from flliper.srt.mem_cache.allocator import TokenToKVPoolAllocator
from flliper.srt.mem_cache.base_prefix_cache import EvictParams, MatchPrefixParams
from flliper.srt.mem_cache.cache_init_params import CacheInitParams
from flliper.srt.mem_cache.hicache_storage import PoolName
from flliper.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from flliper.srt.mem_cache.radix_cache import RadixKey
from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from flliper.srt.mem_cache.utils import compute_node_hash_values, get_hash_str
from flliper.srt.sampling.sampling_params import SamplingParams
from flliper.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

MAMBA_SLOTS = 20
KV_SIZE = 512
NUM_LAYERS = 8
FULL_LAYER_IDS = (3, 7)
MAMBA_LAYER_IDS = [i for i in range(NUM_LAYERS) if i not in FULL_LAYER_IDS]

N = 21
PROMPT = list(range(1000, 1000 + N))
SHARED = 18                     # 18064 scaled
OWN = 16                        # told 16384 scaled: head 8 + span 8
HEAD = 8
TWIN = PROMPT[:SHARED] + list(range(5000, 5000 + 5))
RID = "pdflip-0-2"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "P")
    monkeypatch.delenv("FLLIPER_PDFLIP_DUAL_LAYOUT", raising=False)
    monkeypatch.delenv("FLLIPER_PDFLIP_FOLLOWER_STATE_AT_TOLD", raising=False)
    monkeypatch.setattr(ccmod, "rank_holds_gdn_state", lambda cc: True)
    monkeypatch.setattr(UnifiedRadixCache, "_s90b_follower_n", 0, raising=False)


def _fixture():
    server_args = ServerArgs(model_path="dummy", page_size=1)
    server_args._mamba_cache_chunk_size = FLA_CHUNK_SIZE
    server_args.chunked_prefill_size = 4
    set_global_server_args_for_scheduler(server_args)
    with envs.FLLIPER_MAMBA_SSM_DTYPE.override("bfloat16"):
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
    return SimpleNamespace(cache=cache, allocator=allocator, pool=pool)


def _prefill_source(fx):
    """pdflip-0-1: chunk end at 20 (state there), finish at 21 -- no state at 16."""
    req = Req(rid="pdflip-0-1", origin_input_text="", origin_input_ids=array("q", PROMPT),
              sampling_params=SamplingParams(temperature=0, max_new_tokens=1))
    fx.pool.alloc([req])
    req.output_ids = array("q")
    req.full_untruncated_fill_ids = array("q", PROMPT)
    req.swa_uuid_for_lock = None
    req.extra_key = None
    mr = fx.cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", PROMPT))))
    req.prefix_indices = mr.device_indices.to(torch.int64)
    req.cache_protected_len = 0
    req.last_node = mr.last_device_node
    fx.cache.inc_lock_ref(req.last_node)
    fx.pool.write((req.req_pool_idx, slice(0, 20)), fx.allocator.alloc(20))
    req.set_extend_range(0, 20)
    fx.cache.cache_unfinished_req(req, chunked=True)
    fx.pool.write((req.req_pool_idx, slice(20, N)), fx.allocator.alloc(N - 20))
    req.set_extend_range(20, N)
    req.kv_committed_len = N
    req.kv_allocated_len = N
    fx.cache.cache_finished_req(req, is_insert=True)


def _hash_tree(cache):
    """The storage tier's page hashes (``compute_node_hash_values``, as an enable_storage insert sets them)."""
    stack = list(cache.root_node.children.values())
    while stack:
        n = stack.pop(0)
        n.hash_value = compute_node_hash_values(n, cache.page_size)
        stack.extend(n.children.values())


def _resume(cache, ids, k):
    return len(cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", ids[:k])))).device_indices)


class _HostPool:
    def __init__(self):
        self.freed, self.loads = [], []

    def alloc_read(self, n):
        return torch.tensor([10_000], dtype=torch.int64)

    def is_placeholder(self, i):
        return int(i) >= 10_000

    def free(self, idx):
        self.freed.append(int(idx[0]))

    def load_to_device_per_layer(self, device_pool, host_indices, device_indices, layer_id, io_backend="direct"):
        self.loads.append(layer_id)


class _Backend:
    def __init__(self, ok=True):
        self.ok, self.calls = ok, []

    def batch_get_v2(self, transfers):
        t = transfers[0]
        self.calls.append((t.name, list(t.keys)))
        if self.ok:
            t.host_indices[0] = 5
        return {PoolName.MAMBA: [self.ok]}


def _follower(fx, ok=True):
    """The follower's store tier (the arena read of S90b); the tree itself is the real one."""
    be, hp = _Backend(ok), _HostPool()
    comp = fx.cache.components[ComponentType.MAMBA]
    comp._mamba_pool_host = hp
    entry = SimpleNamespace(host_pool=hp, device_pool=SimpleNamespace(),
                            layer_mapping={l: i for i, l in enumerate(MAMBA_LAYER_IDS)})
    fx.cache.cache_controller = SimpleNamespace(
        storage_backend=be, io_backend="direct",
        mem_pool_host=SimpleNamespace(entry_map={PoolName.MAMBA: entry}))
    return be, hp


def _twin():
    return SimpleNamespace(rid=RID, full_untruncated_fill_ids=array("q", TWIN), extra_key=None)


def _setup():
    fx = _fixture()
    _prefill_source(fx)
    _hash_tree(fx.cache)
    # the capped probe at own (rank_resumable's match_prefix of ids[:own]) splits 0..20 at 16
    assert _resume(fx.cache, TWIN, OWN) == 0          # live: resumable=0
    node = fx.cache._pdflip_kv_chain_end(array("q", TWIN), None, OWN)
    assert node is not None, "the probe split the node at own"
    cd = node.component_data[ComponentType.MAMBA]
    assert cd.value is None and node.component_data[ComponentType.FULL].value is not None
    return fx, node


def _evict_all(fx):
    # the store double served the S90b read only; the eviction runs on the plain device tree (no HiCache tier)
    fx.cache.cache_controller = None
    fx.cache.evict(EvictParams(num_tokens=10 ** 6, mamba_num=10 ** 6))


def test_twin_attach_on_the_real_tree_reads_the_key_of_the_node_at_own(caplog):
    """SOLL 3: absolute depth head+span, the split node at own, its key -- not the deeper node's."""
    fx, node = _setup()
    be, hp = _follower(fx)
    req = _twin()
    with caplog.at_level(logging.INFO):
        assert fx.cache.pdflip_follower_state_at_told(req, HEAD + 8) == OWN
    want = get_hash_str(RadixKey(array("q", PROMPT[:OWN])), None, page_size=1)[-1]
    assert be.calls == [(PoolName.MAMBA, [want])]
    deeper = fx.cache._pdflip_kv_chain_end(array("q", PROMPT), None, 20)
    assert deeper is not None and want != deeper.get_last_hash_value()
    assert node.component_data[ComponentType.MAMBA].value is not None
    assert _resume(fx.cache, TWIN, OWN) == OWN
    assert any("how=device_attach" in m for m in caplog.messages)
    assert getattr(req, "_pdflip_s90b_attach_ms", None) is not None


def test_the_attached_state_survives_an_eviction_until_the_seat():
    """MUSS 1: pinned under the rid -- an eviction between ack and seat leaves it; the seat's probe still says own."""
    fx, node = _setup()
    _follower(fx)
    assert fx.cache.pdflip_follower_state_at_told(_twin(), OWN) == OWN
    assert RID in fx.cache.__dict__.get("_pdflip_s90b_pins", {})
    _evict_all(fx)
    assert node.component_data[ComponentType.MAMBA].value is not None
    assert _resume(fx.cache, TWIN, OWN) == OWN


@pytest.mark.parametrize("exit_", ["settle", "abort", "pop"])
def test_the_pin_goes_at_the_1417_exits(exit_):
    fx, node = _setup()
    _follower(fx)
    assert fx.cache.pdflip_follower_state_at_told(_twin(), OWN) == OWN
    if exit_ == "settle":
        fx.cache._unpin_prefetched_span(RID)              # p_intake.settle_told
    elif exit_ == "abort":
        fx.cache.release_aborted_request(RID)             # follower_release / retract
    else:
        fx.cache.pop_prefetch_loaded_tokens(RID)          # the admission pop
    assert RID not in (fx.cache.__dict__.get("_pdflip_s90b_pins") or {})
    _evict_all(fx)
    assert node.component_data[ComponentType.MAMBA].value is None   # evictable again: no leaked lock
    assert _resume(fx.cache, TWIN, OWN) == 0


def test_a_floor_below_own_releases_it_a_floor_at_own_keeps_it():
    fx, node = _setup()
    _follower(fx)
    assert fx.cache.pdflip_follower_state_at_told(_twin(), OWN) == OWN
    assert fx.cache._pdflip_s90b_unpin(RID, above=OWN) == 0
    assert RID in fx.cache.__dict__["_pdflip_s90b_pins"]
    assert fx.cache._pdflip_s90b_unpin(RID, above=8) == 1
    _evict_all(fx)
    assert node.component_data[ComponentType.MAMBA].value is None


def test_a_later_span_pin_of_the_rid_does_not_release_the_state_pin():
    fx, node = _setup()
    _follower(fx)
    assert fx.cache.pdflip_follower_state_at_told(_twin(), OWN) == OWN
    fx.cache._pin_prefetched_span(RID, node, fx.cache.root_node)   # a W27 re-read's completion
    assert RID in fx.cache.__dict__["_pdflip_s90b_pins"]
    fx.cache._unpin_prefetched_span(RID)
    assert not fx.cache.__dict__["_pdflip_s90b_pins"]


def test_store_miss_pins_nothing():
    fx, node = _setup()
    _follower(fx, ok=False)
    assert fx.cache.pdflip_follower_state_at_told(_twin(), OWN) == 0
    assert not (fx.cache.__dict__.get("_pdflip_s90b_pins") or {})
    assert _resume(fx.cache, TWIN, OWN) == 0
