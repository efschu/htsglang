"""L15-11b (AP 1001): the held mamba rows must survive the flush reset.

The D-rank sleep flush (``HybridReqToTokenPool.clear`` ->
``mamba_pool.reset_state``) zeroed EVERY mamba slot, which would wipe the
compacted GDN anchors that ``weg2/l15_retain.py`` keeps in slots
``[1, A_H)``. ``reset_state(keep_rows=N)`` and ``clear(keep_mamba_rows=N)``
must spare rows ``[0, N)`` of every slot-axis buffer (conv, temporal, the
per-slot ReplaySSM rings and the per-slot write cursors); request/spec-row
scratch has no slot axis and stays fully zeroed. Default ``N=0`` is
today's byte-identical full-zero path.

CPU-only: the pool is built exactly like ``test_mamba_pool_floor.py``
(same scaffold, no accelerator).
"""

import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.environ import envs
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10)

NUM_LAYERS = 8
GLOBAL_INTERVAL = 4
MAMBA_SIZE = 6
MAX_CONTEXT_LEN = 128
KEPT = 3


def _server_args() -> ServerArgs:
    server_args = ServerArgs(model_path="dummy", page_size=1)
    # The property would otherwise load the HF config of the dummy model.
    server_args._mamba_cache_chunk_size = FLA_CHUNK_SIZE
    return server_args


def _build_pool() -> HybridReqToTokenPool:
    """CPU HybridReqToTokenPool (scaffold copied from test_mamba_pool_floor.py)."""
    set_global_server_args_for_scheduler(_server_args())
    full_attention_layer_ids = [
        i for i in range(GLOBAL_INTERVAL - 1, NUM_LAYERS, GLOBAL_INTERVAL)
    ]
    mamba_layers = [i for i in range(NUM_LAYERS) if i not in full_attention_layer_ids]
    with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
        shape = Mamba2StateShape.create(
            tp_world_size=1,
            intermediate_size=512,
            n_groups=4,
            num_heads=8,
            head_dim=64,
            state_size=32,
            conv_kernel=4,
        )
        cache_params = Mamba2CacheParams(shape=shape, layers=mamba_layers)
    return HybridReqToTokenPool(
        size=10,
        mamba_size=MAMBA_SIZE,
        mamba_spec_state_size=10,
        max_context_len=MAX_CONTEXT_LEN,
        device="cpu",
        enable_memory_saver=False,
        cache_params=cache_params,
        mamba_layer_ids=mamba_layers,
        enable_mamba_extra_buffer=False,
        enable_linear_replayssm=False,
    )


def _fill_ones(mamba_pool) -> None:
    for conv in mamba_pool.mamba_cache.conv:
        conv.fill_(1.0)
    mamba_pool.mamba_cache.temporal.fill_(1.0)


def _assert_rows(mamba_pool, kept: int, zero_tail: bool) -> None:
    for conv in mamba_pool.mamba_cache.conv:
        assert torch.all(conv[:, :kept] == 1.0), (
            f"held rows [0, {kept}) were wiped: {conv.shape}"
        )
        if zero_tail:
            assert torch.all(conv[:, kept:] == 0.0), (
                f"tail rows [{kept}:] were not zeroed: {conv.shape}"
            )
    temporal = mamba_pool.mamba_cache.temporal
    assert torch.all(temporal[:, :kept] == 1.0), "held temporal rows were wiped"
    if zero_tail:
        assert torch.all(temporal[:, kept:] == 0.0), "tail temporal rows not zeroed"


def _assert_all_zero(mamba_pool) -> None:
    for conv in mamba_pool.mamba_cache.conv:
        assert torch.all(conv == 0.0), "conv not fully zeroed"
    assert torch.all(mamba_pool.mamba_cache.temporal == 0.0), (
        "temporal not fully zeroed"
    )


def test_reset_state_keep_rows():
    """keep_rows=3: rows [0,3) keep their bytes, rows [3:) are zeroed."""
    mamba_pool = _build_pool().mamba_pool
    _fill_ones(mamba_pool)
    mamba_pool.reset_state(keep_rows=KEPT)
    _assert_rows(mamba_pool, KEPT, zero_tail=True)


def test_reset_state_default_zeroes_all():
    """keep_rows=0 (default): byte-identical to today -- everything zeroed."""
    mamba_pool = _build_pool().mamba_pool
    _fill_ones(mamba_pool)
    mamba_pool.reset_state()
    _assert_all_zero(mamba_pool)


def test_clear_keep_mamba_rows_passthrough():
    """HybridReqToTokenPool.clear(keep_mamba_rows=3) spares the held rows."""
    hrt = _build_pool()
    mamba_pool = hrt.mamba_pool
    _fill_ones(mamba_pool)
    hrt.clear(keep_mamba_rows=KEPT)
    _assert_rows(mamba_pool, KEPT, zero_tail=True)
