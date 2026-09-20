"""Task #42: owned-row compaction of the QSA rows kernel under DCP."""

import pytest
import torch

from sglang.srt.layers.attention.qsa import sparse_attn as sa


def test_compact_owned_rows_sorts_owned_first_and_counts():
    rows = torch.tensor([[-1, 5, -1, 2], [-1, -1, -1, -1], [7, 1, 3, 0]], dtype=torch.int32)
    sorted_rows, counts = sa.compact_owned_rows(rows)
    assert counts.tolist() == [2, 0, 4]
    assert sorted_rows[0, :2].tolist() == [5, 2] and sorted_rows[0, 2:].tolist() == [-1, -1]
    assert sorted_rows[1].tolist() == [-1, -1, -1, -1]
    assert sorted(sorted_rows[2].tolist()) == [0, 1, 3, 7]


def test_the_kernel_bounds_the_loop_by_counts():
    import inspect

    # the kernel is a triton JITFunction: read its module source
    src = open(inspect.getsourcefile(sa)).read()
    kernel = src[src.index("def _sparse_attn_rows_fwd("): src.index("def compact_owned_rows(")]
    assert "if USE_COUNTS:" in kernel and "limit = tl.load(counts + query)" in kernel
    assert "for start in range(0, limit, BLOCK_N):" in kernel
    assert "USE_COUNTS=use_counts" in inspect.getsource(sa.sparse_attn_rows_triton)


def test_the_backend_switch_defaults_on(monkeypatch):
    from sglang.srt.layers.attention import qwen_sparse_attn_backend as qb

    for raw, want in (("", True), ("0", False), ("1", True)):
        qb._QSA_ROWS_COMPACT["on"] = None
        if raw == "":
            monkeypatch.delenv("SGLANG_QSA_ROWS_COMPACT", raising=False)
        else:
            monkeypatch.setenv("SGLANG_QSA_ROWS_COMPACT", raw)
        assert qb._qsa_rows_compact_on() is want, raw
    qb._QSA_ROWS_COMPACT["on"] = None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a card")
@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_compacted_rows_match_the_masked_loop_on_gpu(kv_dtype):
    torch.manual_seed(0)
    Tq, Hq, Hkv, D, pool, topk = 64, 24, 2, 128, 4096, 512
    q = torch.randn(Tq, Hq, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(pool, Hkv, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(pool, Hkv, D, device="cuda", dtype=torch.bfloat16)
    if kv_dtype == torch.float8_e4m3fn:
        k, v = k.to(kv_dtype), v.to(kv_dtype)
    rows = torch.randint(0, pool, (Tq, topk), device="cuda", dtype=torch.int32)
    owned = torch.rand(Tq, topk, device="cuda") < 0.3  # ~30 % owned, like a 3080 shard
    rows = torch.where(owned, rows, torch.full_like(rows, -1))
    rows[3] = -1  # a query with no owned row: lse must stay -inf, out 0
    out_ref, lse_ref = sa.sparse_attn_rows_triton(q, k, v, rows, 0.1)
    rows_c, counts = sa.compact_owned_rows(rows)
    out, lse = sa.sparse_attn_rows_triton(q, k, v, rows_c, 0.1, row_counts=counts)
    assert torch.allclose(out.float(), out_ref.float(), atol=2e-2, rtol=2e-2)
    finite = torch.isfinite(lse_ref)
    assert torch.equal(finite, torch.isfinite(lse))
    assert torch.allclose(lse[finite], lse_ref[finite], atol=1e-3, rtol=1e-3)
    assert not torch.isfinite(lse[3]).any() and out[3].abs().max().item() == 0
