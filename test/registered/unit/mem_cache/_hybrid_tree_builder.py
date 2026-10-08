"""A real UnifiedRadixCache over a real HybridReqToTokenPool on CPU -- the builder of NF's
test_pdflip_d_seat_compact_0930.py (``_build``), without that test's D-seat modules, for the lines that do not
carry them (27B). test_pdflip_prefetch_anchor_attach_0930.py loads it when the NF file is absent."""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from flliper.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape  # noqa: E402
from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE  # noqa: E402
from flliper.srt.mem_cache.allocator import TokenToKVPoolAllocator  # noqa: E402
from flliper.srt.mem_cache.cache_init_params import CacheInitParams  # noqa: E402
from flliper.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool  # noqa: E402
from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType  # noqa: E402
from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache  # noqa: E402
from flliper.srt.server_args import ServerArgs, set_global_server_args_for_scheduler  # noqa: E402

NUM_LAYERS = 8
FULL = (3, 7)
LIN = [i for i in range(NUM_LAYERS) if i not in FULL]
SIZE = 38


def _build():
    sa = ServerArgs(model_path="dummy", page_size=1)
    sa._mamba_cache_chunk_size = FLA_CHUNK_SIZE
    sa.max_running_requests = 6
    sa.enable_hierarchical_cache = False
    sa.disable_radix_cache = False
    sa.disable_overlap_schedule = True
    set_global_server_args_for_scheduler(sa)
    with envs.FLLIPER_MAMBA_SSM_DTYPE.override("bfloat16"):
        shape = Mamba2StateShape.create(tp_world_size=1, intermediate_size=256, n_groups=1, num_heads=2,
                                        head_dim=16, state_size=16, conv_kernel=4)
        cp = Mamba2CacheParams(shape=shape, layers=LIN)
    pool = HybridReqToTokenPool(size=10, mamba_size=SIZE, mamba_spec_state_size=10, max_context_len=256,
                                device="cpu", enable_memory_saver=False, cache_params=cp,
                                mamba_layer_ids=LIN, enable_mamba_extra_buffer=False,
                                speculative_num_draft_tokens=3)
    kv = HybridLinearKVPool(size=512, dtype=torch.bfloat16, page_size=1, head_num=2, head_dim=64,
                            full_attention_layer_ids=list(FULL), device="cpu", enable_memory_saver=False,
                            mamba_pool=pool.mamba_pool)
    alloc = TokenToKVPoolAllocator(size=512, dtype=torch.bfloat16, device="cpu", kvcache=kv, need_sort=False)
    params = CacheInitParams(req_to_token_pool=pool, token_to_kv_pool_allocator=alloc, page_size=1,
                             disable=False, sliding_window_size=None,
                             tree_components=(ComponentType.FULL, ComponentType.MAMBA),
                             enable_mamba_extra_buffer=False, enable_kv_cache_events=False,
                             eviction_policy="lru", is_eagle=False)
    return UnifiedRadixCache(params=params), pool, alloc
