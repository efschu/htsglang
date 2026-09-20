"""QSA-DENSE-CHECK (fn5i): the dense causal reference matches torch SDPA with
GQA, the env switch parses, eligibility is single-request/prefix-free only,
and the forward_extend wrapper substitutes the reference only in subst mode."""

import types

import pytest
import torch

from sglang.srt.layers.attention import qwen_sparse_attn_backend as qsb


def _sdpa_reference(q, k, v, scaling):
    n, hq, d = q.shape
    g = hq // k.shape[1]
    qf = q.float().transpose(0, 1)
    kf = k.float().transpose(0, 1).repeat_interleave(g, 0)
    vf = v.float().transpose(0, 1).repeat_interleave(g, 0)
    out = torch.nn.functional.scaled_dot_product_attention(
        qf, kf, vf, is_causal=True, scale=scaling
    )
    return out.transpose(0, 1).reshape(n, hq * d)


def test_dense_reference_matches_sdpa_gqa():
    torch.manual_seed(0)
    n, hq, hkv, d = 37, 6, 2, 16
    q = torch.randn(n, hq, d, dtype=torch.bfloat16)
    k = torch.randn(n + 3, hkv, d, dtype=torch.bfloat16)
    v = torch.randn(n + 3, hkv, d, dtype=torch.bfloat16)
    ref = qsb._qsa_dense_reference(q, k, v, d**-0.5)
    assert ref.shape == (n, hq * d) and ref.dtype == torch.bfloat16
    expect = _sdpa_reference(q, k[:n], v[:n], d**-0.5)
    assert torch.allclose(ref.float(), expect, atol=2e-2, rtol=2e-2)


def test_dense_reference_is_causal():
    torch.manual_seed(1)
    n, hq, hkv, d = 9, 2, 1, 8
    q = torch.randn(n, hq, d)
    k = torch.randn(n, hkv, d)
    v = torch.randn(n, hkv, d)
    ref = qsb._qsa_dense_reference(q, k, v, 1.0)
    v2 = v.clone()
    v2[5:] += 100.0  # future rows must not influence rows < 5
    ref2 = qsb._qsa_dense_reference(q, k, v2, 1.0)
    assert torch.allclose(ref[:5], ref2[:5])
    assert not torch.allclose(ref[5:], ref2[5:])


def test_mode_switch(monkeypatch):
    monkeypatch.delenv("SGLANG_QSA_DENSE_CHECK", raising=False)
    assert qsb._qsa_dense_check_mode() == ""
    monkeypatch.setenv("SGLANG_QSA_DENSE_CHECK", "1")
    assert qsb._qsa_dense_check_mode() == "check"
    monkeypatch.setenv("SGLANG_QSA_DENSE_CHECK", "subst")
    assert qsb._qsa_dense_check_mode() == "subst"
    monkeypatch.setenv("SGLANG_QSA_DENSE_CHECK", "0")
    assert qsb._qsa_dense_check_mode() == ""


def test_eligible_single_request_prefix_free_only():
    k = torch.zeros(10, 2, 4)
    fb = types.SimpleNamespace(seq_lens_cpu=[10], extend_seq_lens_cpu=[10])
    assert qsb._qsa_dense_check_eligible(fb, 10, k)
    fb = types.SimpleNamespace(seq_lens_cpu=[12], extend_seq_lens_cpu=[10])
    assert not qsb._qsa_dense_check_eligible(fb, 10, k)  # prefix
    fb = types.SimpleNamespace(seq_lens_cpu=[5, 5], extend_seq_lens_cpu=[5, 5])
    assert not qsb._qsa_dense_check_eligible(fb, 10, k)  # two requests
    fb = types.SimpleNamespace(seq_lens_cpu=[1], extend_seq_lens_cpu=[1])
    assert not qsb._qsa_dense_check_eligible(fb, 1, k[:1])


def _fake_backend(impl_output):
    backend = qsb.QwenSparseAttnBackend.__new__(qsb.QwenSparseAttnBackend)
    backend.is_draft_worker = True
    backend.dcp_size = 1
    backend.token_to_kv_pool = types.SimpleNamespace(size=64)
    backend._forward_extend_impl = lambda *a, **kw: impl_output.clone()
    return backend


@pytest.mark.parametrize("mode,expect_ref", [("check", False), ("subst", True)])
def test_wrapper_substitutes_only_in_subst_mode(monkeypatch, mode, expect_ref):
    monkeypatch.setenv("SGLANG_QSA_DENSE_CHECK", mode)
    torch.manual_seed(2)
    n, hq, hkv, d = 6, 4, 2, 8
    q = torch.randn(n, hq * d)
    k = torch.randn(n, hkv, d)
    v = torch.randn(n, hkv, d)
    layer = types.SimpleNamespace(tp_q_head_num=hq, tp_k_head_num=hkv, head_dim=d, scaling=1.0, layer_id=0)
    fb = types.SimpleNamespace(
        seq_lens_cpu=[n], extend_seq_lens_cpu=[n], out_cache_loc=torch.arange(n),
        forward_mode=types.SimpleNamespace(name="DRAFT_EXTEND_V2"),
    )
    wrong = torch.full((n, hq * d), 7.0)
    backend = _fake_backend(wrong)
    out = qsb.QwenSparseAttnBackend.forward_extend(
        backend, q, k, v, layer, fb, topk_indices=torch.zeros(n, 3, dtype=torch.int32)
    )
    ref = qsb._qsa_dense_reference(q.reshape(n, hq, d), k, v, 1.0)
    assert torch.allclose(out, ref) == expect_ref
    assert torch.allclose(out, wrong) == (not expect_ref)


def test_wrapper_off_passes_through(monkeypatch):
    monkeypatch.delenv("SGLANG_QSA_DENSE_CHECK", raising=False)
    wrong = torch.full((3, 8), 7.0)
    backend = _fake_backend(wrong)
    layer = types.SimpleNamespace(tp_q_head_num=2, tp_k_head_num=1, head_dim=4, scaling=1.0, layer_id=0)
    out = qsb.QwenSparseAttnBackend.forward_extend(
        backend, torch.zeros(3, 8), torch.zeros(3, 1, 4), torch.zeros(3, 1, 4), layer,
        types.SimpleNamespace(seq_lens_cpu=[3], extend_seq_lens_cpu=[3]),
        topk_indices=torch.zeros(3, 2, dtype=torch.int32),
    )
    assert torch.equal(out, wrong)


def test_global_kv_map_replicated_kv_plan():
    """Plan [12, 6, 6] q heads over 24 heads / 2 kv heads: rank 0 -> kv 0 for
    all 12 heads, ranks 1 and 2 -> kv 1 for all their heads (the kernels'
    local grouping would pair rank 0's heads 6..11 with kv 1)."""
    assert qsb._qsa_global_kv_map(0, 12, 24, 2).tolist() == [0] * 12
    assert qsb._qsa_global_kv_map(12, 6, 24, 2).tolist() == [1] * 6
    assert qsb._qsa_global_kv_map(18, 6, 24, 2).tolist() == [1] * 6
    local = (torch.arange(12) // 6).tolist()
    assert local != [0] * 12  # the local grouping differs on rank 0


def test_dense_reference_with_kv_map_uses_given_heads():
    torch.manual_seed(3)
    n, hq, hkv, d = 7, 4, 2, 8
    q = torch.randn(n, hq, d)
    k = torch.randn(n, hkv, d)
    v = torch.randn(n, hkv, d)
    all_kv1 = torch.tensor([1, 1, 1, 1])
    ref = qsb._qsa_dense_reference(q, k, v, 1.0, kv_map=all_kv1)
    expect = _sdpa_reference(q, k[:, 1:2], v[:, 1:2], 1.0)
    assert torch.allclose(ref.float(), expect, atol=1e-5)
    # identity map == local grouping
    local_map = torch.arange(hq) // (hq // hkv)
    assert torch.allclose(
        qsb._qsa_dense_reference(q, k, v, 1.0, kv_map=local_map),
        qsb._qsa_dense_reference(q, k, v, 1.0),
    )


def test_mode_subst_global_parses(monkeypatch):
    monkeypatch.setenv("SGLANG_QSA_DENSE_CHECK", "subst_global")
    assert qsb._qsa_dense_check_mode() == "subst_global"
