"""REPLICATED-KV rank-local attention (fn5i/fn5m 19.09.): each rank's q heads
attend to exactly ONE kv head under the model's global GQA grouping; the
NEXTN draft's pool keeps only that head; the prefix-free QSA extend feeds the
kernel only that head."""

import types

import pytest
import torch

from sglang.srt.distributed.utils import (
    attn_replicated_kv_local_head,
    draft_rank_local_single_kv_head,
    set_tp_partition_ratios,
)


@pytest.fixture(autouse=True)
def _clean_plan():
    set_tp_partition_ratios(None)
    yield
    set_tp_partition_ratios(None)


def test_local_kv_head_next_flash_plan():
    set_tp_partition_ratios([39, 13, 12])  # q heads 12 / 6 / 6 of 24, kv 2
    assert [attn_replicated_kv_local_head(24, 2, 3, r) for r in range(3)] == [0, 1, 1]


def test_local_kv_head_none_without_geometry():
    assert attn_replicated_kv_local_head(24, 2, 3, 0) is None  # no plan
    set_tp_partition_ratios([39, 13, 12])
    assert attn_replicated_kv_local_head(24, 8, 3, 0) is None  # kv >= tp: not replicated


def test_draft_single_kv_head_switch():
    assert not draft_rank_local_single_kv_head(True, 2, 3)  # no plan
    set_tp_partition_ratios([39, 13, 12])
    assert draft_rank_local_single_kv_head(True, 2, 3)
    assert not draft_rank_local_single_kv_head(False, 2, 3)  # target keeps all heads
    assert not draft_rank_local_single_kv_head(True, 8, 3)  # DFlash2/27B-shaped drafts


def test_select_rank_local_kv_head():
    from sglang.srt.models.qwen3_5 import select_rank_local_kv_head

    T, nkv, d = 5, 2, 4
    k = torch.arange(T * nkv * d, dtype=torch.float32).reshape(T, nkv * d)
    v = -k
    k1, v1 = select_rank_local_kv_head(k, v, nkv, d, 1)
    assert k1.shape == (T, d) and torch.equal(k1, k[:, d:])
    assert torch.equal(v1, v[:, d:])
    k0, _ = select_rank_local_kv_head(k, v, nkv, d, 0)
    assert torch.equal(k0, k[:, :d])


def test_backend_rank_local_kv_inputs(monkeypatch):
    from sglang.srt.layers.attention import qwen_sparse_attn_backend as qsb
    import sglang.srt.runtime_context as rc

    set_tp_partition_ratios([39, 13, 12])
    monkeypatch.setattr(
        rc, "get_parallel", lambda: types.SimpleNamespace(attn_tp_size=3, attn_tp_rank=1)
    )
    backend = qsb.QwenSparseAttnBackend.__new__(qsb.QwenSparseAttnBackend)
    backend.dcp_model_config = types.SimpleNamespace(
        get_total_num_kv_heads=lambda: 2,
        hf_text_config=types.SimpleNamespace(num_attention_heads=24),
    )
    layer = types.SimpleNamespace(tp_k_head_num=2, tp_q_head_num=6)
    k = torch.randn(7, 2, 8)
    v = torch.randn(7, 2, 8)
    k1, v1 = backend._rank_local_kv_inputs(layer, k, v)
    assert k1.shape == (7, 1, 8) and torch.equal(k1[:, 0], k[:, 1])
    assert torch.equal(v1[:, 0], v[:, 1])
    # a single-head layer (the draft's pool after the fix) is left alone
    single = types.SimpleNamespace(tp_k_head_num=1, tp_q_head_num=6)
    k2, _ = backend._rank_local_kv_inputs(single, k[:, :1], v[:, :1])
    assert k2.shape == (7, 1, 8)
