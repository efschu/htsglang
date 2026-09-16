"""WP3b: QSA sparse attention under this line's uneven DCP.

Upstream's QSA backend reads and writes the KV pool with GLOBAL req_to_token
slots; under --rank-tp-ratio every rank stores only the rows it owns
(layers/dcp/owner.py). The rows kernel attends this rank's OWNED subset of a
query's top-k on the gathered heads and emits the natural-log LSE; the group
merge (the math of cp_lse_ag_out_ar_mha_uneven) recombines the partials.
Pinned here on CPU against the reference; the Triton kernel itself is pinned
against the same reference on the metal (test marked cuda)."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.qsa.kernel import qsa_sparse_attention_reference
from sglang.srt.layers.attention.qsa.sparse_attn import (
    merge_partial_attention,
    sparse_attn_rows_reference,
    sparse_attn_rows_triton,
)
from sglang.srt.layers.dcp.owner import dcp_weighted_read_slots

TQ, HQ, HKV, D, N, TOPK = 5, 6, 2, 32, 64, 8


def _case(seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(TQ, HQ, D, generator=g)
    k = torch.randn(N, HKV, D, generator=g)
    v = torch.randn(N, HKV, D, generator=g)
    rows = torch.randint(0, N, (TQ, TOPK), generator=g)
    rows[0, 5:] = -1  # a short selection
    rows[3, :] = -1  # a query that attends nothing at all
    return q, k, v, rows.to(torch.int32)


def test_rows_reference_matches_the_slot_reference():
    q, k, v, rows = _case()
    out, lse = sparse_attn_rows_reference(q, k, v, rows, 0.3)
    ref = qsa_sparse_attention_reference(q, k, v, rows, 0.3)
    live = rows[:, 0] >= 0
    assert torch.allclose(out[live], ref[live], atol=1e-5)
    assert torch.isfinite(lse[live]).all()
    assert bool((out[3] == 0).all()) and bool(torch.isinf(lse[3]).all())


def _owner_split(rows, dcp_size, ratios=(3, 1)):
    """Each rank's rows under the weighted owner rule (cp_S = sum(ratios))."""
    cp_S = sum(ratios)
    parts = []
    lo = 0
    for r in ratios:
        compact, owned = dcp_weighted_read_slots(rows.clamp(min=0), cp_S, lo, lo + r, r)
        parts.append(torch.where((rows >= 0) & owned, compact, torch.full_like(compact, -1)))
        lo += r
    return parts


def test_owned_partials_merge_to_the_full_attention():
    q, k, v, rows = _case(1)
    ratios = (3, 1)
    cp_S = sum(ratios)
    # Every rank's pool holds its compact rows: build them from the global k/v.
    parts = _owner_split(rows, 2, ratios)
    outs, lses = [], []
    lo = 0
    for r, local_rows in zip(ratios, parts):
        # compact row c of this rank = global slot (c // r) * cp_S + lo + c % r
        n_local = (N // cp_S) * r
        c = torch.arange(n_local)
        gslot = (c // r) * cp_S + lo + (c % r)
        k_local, v_local = k[gslot], v[gslot]
        out_r, lse_r = sparse_attn_rows_reference(q, k_local, v_local, local_rows, 0.3)
        outs.append(out_r)
        lses.append(lse_r)
        lo += r
    merged, _ = merge_partial_attention(outs, lses)
    full, _ = sparse_attn_rows_reference(q, k, v, rows, 0.3)
    live = rows[:, 0] >= 0
    assert torch.allclose(merged[live], full[live].float(), atol=1e-4)
    # the empty query stays zero after the merge (no nan from exp(-inf - -inf))
    assert bool((merged[3] == 0).all())
    # every attended row is owned by exactly one rank
    owned = torch.stack([p >= 0 for p in parts]).int().sum(0)
    assert torch.equal(owned, (rows >= 0).int())


def _backend(dcp_size=1, weighted=False, ratios=(3, 1), rank=0):
    from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend

    b = QwenSparseAttnBackend(runner=None)
    assert b.dcp_size == 1  # no runner: DCP off, identity mapping
    if dcp_size > 1:
        b.dcp_size, b.dcp_rank = dcp_size, rank
        b.uneven_dcp = b.uneven_dcp_weighted = weighted
        if weighted:
            lo = sum(ratios[:rank])
            b.cp_S, b.cp_lo, b.cp_hi, b.cp_ratio = sum(ratios), lo, lo + ratios[rank], ratios[rank]
    return b


def test_backend_local_rows_partition_the_slots_across_ranks():
    slots = torch.tensor([[0, 1, 2, 3, 4, 7, -1, -1], [5, 6, -1, -1, -1, -1, -1, -1]])
    assert torch.equal(_backend()._local_rows(slots), slots.to(torch.int32))
    r0 = _backend(2, True, (3, 1), rank=0)._local_rows(slots)
    r1 = _backend(2, True, (3, 1), rank=1)._local_rows(slots)
    assert r0.tolist()[0] == [0, 1, 2, -1, 3, -1, -1, -1]  # slots 0,1,2 -> rows 0,1,2; 4 -> row 3
    assert r1.tolist()[0] == [-1, -1, -1, 0, -1, 1, -1, -1]  # slots 3 and 7 -> rows 0 and 1
    assert torch.equal((r0 >= 0).int() + (r1 >= 0).int(), (slots >= 0).int())
    e0 = _backend(2, False, rank=0)._local_rows(slots)
    assert e0.tolist()[1] == [-1, 3, -1, -1, -1, -1, -1, -1]  # even rule: slot 6 -> row 3


def test_backend_write_is_masked_to_the_owned_rows_under_dcp():
    calls = []
    pool = SimpleNamespace(set_kv_buffer=lambda layer, loc, k, v, **kw: calls.append((loc.tolist(), kw)))
    fb = SimpleNamespace(out_cache_loc=torch.tensor([0, 1, 2, 3, 4, 7]), positions=None)
    b = _backend()
    b.token_to_kv_pool = pool
    b._set_kv_buffer(fb, "layer", None, None)
    assert calls[-1] == ([0, 1, 2, 3, 4, 7], {})
    b = _backend(2, True, (3, 1), rank=1)
    b.token_to_kv_pool = pool
    b._set_kv_buffer(fb, "layer", None, None)
    loc, kw = calls[-1]
    assert kw["dcp_kv_mask"].tolist() == [False, False, False, True, False, True]
    assert loc[3] == 0 and loc[5] == 1


def test_exports():
    from sglang.srt.layers.attention.qsa import sparse_attn as m

    assert {"sparse_attn_rows_triton", "sparse_attn_rows_reference", "merge_partial_attention"} <= set(m.__all__)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernel")
def test_rows_kernel_matches_the_reference_on_the_metal():
    q, k, v, rows = _case(2)
    q, k, v, rows = (t.cuda() for t in (q, k, v, rows))
    q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
    out, lse = sparse_attn_rows_triton(q, k, v, rows, 0.3)
    ref, ref_lse = sparse_attn_rows_reference(q, k, v, rows, 0.3)
    live = rows[:, 0] >= 0
    assert torch.allclose(out[live].float(), ref[live].float(), atol=3e-2, rtol=3e-2)
    assert torch.allclose(lse[live], ref_lse[live], atol=1e-2)
    assert bool((out[3] == 0).all()) and bool(torch.isinf(lse[3]).all())


def test_backend_is_a_valid_owner_bounds_consumer(monkeypatch):
    """The owner-bounds registry (layers/dcp/owner.py) refuses a consumer
    without refresh_dcp_owner_bounds(); fn1v (2026-09-16) died at backend
    init on exactly that. The refresh re-derives the bounds."""
    from sglang.srt.layers.dcp import owner
    from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend

    b = _backend(2, True, (3, 1), rank=1)
    assert callable(getattr(b, "refresh_dcp_owner_bounds", None))
    owner.register_owner_bounds_consumer(b)  # must not raise
    monkeypatch.setattr(owner, "dcp_weighted_owner_bounds", lambda size, rank: (8, 6, 8, 2))
    b.refresh_dcp_owner_bounds()
    assert (b.cp_S, b.cp_lo, b.cp_hi, b.cp_ratio) == (8, 6, 8, 2)
    plain = _backend()
    plain.refresh_dcp_owner_bounds()  # no-op off the weighted lane


def test_init_dcp_runs_end_to_end_the_way_the_server_builds_it(monkeypatch):
    """The whole _init_dcp path with a fake parallel state (DCP 3, weighted,
    replicated kv), including the owner-bounds registry -- the site fn1v
    (2026-09-16) died at."""
    import sglang.srt.runtime_context as rc
    from sglang.srt.layers.attention import qwen_sparse_attn_backend as qb
    from sglang.srt.layers.dcp import owner

    monkeypatch.setattr(rc, "get_parallel", lambda: SimpleNamespace(attn_dcp_size=3, attn_dcp_rank=1, attn_tp_size=3))
    monkeypatch.setattr("sglang.srt.distributed.utils.uneven_dcp_kv_replicated", lambda n: True)
    monkeypatch.setattr("sglang.srt.distributed.utils.uneven_dcp_active", lambda n: True)
    monkeypatch.setattr("sglang.srt.distributed.utils.attn_kv_replicated", lambda tp, kv: True)
    monkeypatch.setattr(owner, "dcp_weighted_owner_bounds", lambda size, rank: (64, 39, 52, 13))
    cfg = SimpleNamespace(get_total_num_kv_heads=lambda: 2, hf_text_config=None, hf_config=None, context_len=32768)
    b = qb.QwenSparseAttnBackend(runner=SimpleNamespace(model_config=cfg, is_draft_worker=False))
    assert (b.dcp_size, b.dcp_rank, b.uneven_dcp_weighted) == (3, 1, True)
    assert (b.cp_S, b.cp_lo, b.cp_hi, b.cp_ratio) == (64, 39, 52, 13)
    # the draft worker's backend keeps DCP off (its pool holds the full context)
    d = qb.QwenSparseAttnBackend(runner=SimpleNamespace(model_config=cfg, is_draft_worker=True))
    assert d.dcp_size == 1


def test_fp8_byte_decode_matches_torch_for_every_code():
    """The rows kernel loads an fp8 pool as bytes and decodes them itself
    (Triton has no fp8e4nv on sm86 -- the 3080 ranks died at compile in
    fn1w). Same arithmetic, pinned against torch's conversion for all 256."""
    from sglang.srt.layers.attention.qsa.sparse_attn import fp8_e4m3_bytes_to_f32_reference

    codes = torch.arange(256, dtype=torch.uint8)
    ref = codes.view(torch.float8_e4m3fn).float()
    got = fp8_e4m3_bytes_to_f32_reference(codes)
    nan = torch.isnan(ref)
    assert torch.equal(nan, torch.isnan(got)) and nan.sum() == 2
    assert torch.equal(got[~nan], ref[~nan])
