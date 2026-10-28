# SPDX-License-Identifier: Apache-2.0
"""formhp-r2tbound-1028: a decode allocation at the context end must never write past its req_to_token row.

``assign_req_to_token_pool`` writes row ``r`` as ``req_to_token + r * row_width + col``: a column >= row_width IS row ``r + 1``.
The DFLASH ``prepare_for_decode`` writes ``[cur, ceil_page(committed + reserve))``; the old row headroom ``4 + max_draft_tokens``
never counted the page rounding. NF (context 262144, page 64, reserve 16 / 24 with FORMHP-DRAFT-CHAIN): a request at committed
262140 (the largest a still-running request reaches, ``context_len - 4``) gets the watermark 262208 in a row 262156 / 262164 wide
-> 52 / 44 slot ids written into the NEXT request's row (its prompt positions 0..43). 27B (page 1, reserve 16): 262140 + 16 =
262156 = the old width -- fits, with zero margin.

CPU only: the real ``DFlashDraftInputV2.prepare_for_decode`` and the real Triton kernel under ``TRITON_INTERPRET=1``; the kernel's
sm_120 / sm_86 compile (triton_compile_check) in a subprocess without the interpreter.

Red on dcba244df: ``test_nf_context_end_prepare_keeps_the_neighbor_row`` fails on the overwritten neighbor row itself (the module
imports nothing new at top level, so that red is the defect, not an ImportError).
"""

import ast
import contextlib
import os
import subprocess
import sys
import textwrap
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest
import torch

from flliper.kernels.ops.speculative import cache_locs as cl
from flliper.srt.mem_cache import common as mc
from flliper.srt.speculative import dflash_info_v2 as di

CTX = 262144
SENTINEL = -7
SLOT0 = 5_000_000


def _sa(*, algo="DFLASH", page=64, k=8, overlap=True):
    return types.SimpleNamespace(
        speculative_algorithm=algo,
        disable_overlap_schedule=not overlap,
        max_speculative_num_draft_tokens=k,
        speculative_num_draft_tokens=k,
        speculative_num_steps=1,
        speculative_eagle_topk=None,
        page_size=page,
    )


def _nf(monkeypatch):
    # 27b-r2tbound-1028 port: this tree has no FORMHP-DRAFT-CHAIN (formhp-devdraft-b3-1028 not ported), so the NF reserve is 16, not 24
    monkeypatch.delenv("FLLIPER_FORMHP", raising=False)
    return _sa(page=64)


def _b27(monkeypatch):
    monkeypatch.delenv("FLLIPER_FORMHP", raising=False)
    return _sa(page=1)


class _Req:
    def __init__(self, committed, allocated):
        self.kv_committed_len = committed
        self.kv_allocated_len = allocated
        self.sampling_params = types.SimpleNamespace(top_k=1)


def _run_prepare(monkeypatch, sa, width, committed, allocated):
    """The real prepare_for_decode of one decode batch [request at the context end in row 1, a neighbor request in row 2]."""
    monkeypatch.setattr(di, "get_server_args", lambda: sa)
    monkeypatch.setattr(mc, "get_server_args", lambda: sa)
    monkeypatch.setattr(di, "_get_overlap_plan_stream", lambda device: (None, contextlib.nullcontext()))
    calls = []

    def _alloc_paged(tree_cache, cur_gpu, cur_cpu, nxt_gpu, nxt_cpu, last_loc, n, *a, **k):
        calls.append(int(n))
        return torch.arange(SLOT0, SLOT0 + int(n), dtype=torch.int64)

    def _alloc(tree_cache, n, *a, **k):
        calls.append(int(n))
        return torch.arange(SLOT0, SLOT0 + int(n), dtype=torch.int64)

    monkeypatch.setattr(di, "alloc_paged_token_slots_extend", _alloc_paged)
    monkeypatch.setattr(di, "alloc_token_slots", _alloc)
    monkeypatch.setattr(di, "get_last_loc", lambda r2t, idx, cur: torch.zeros_like(cur))
    r2t = torch.full((3, width), SENTINEL, dtype=torch.int32)
    neighbor_len = 1000
    r2t[2, :neighbor_len] = torch.arange(neighbor_len, dtype=torch.int32) + 100  # the neighbor's prompt slots
    reqs = [_Req(committed, allocated), _Req(neighbor_len, neighbor_len + 64 + (64 - neighbor_len % 64) % 64)]
    batch = types.SimpleNamespace(
        reqs=reqs,
        batch_size=lambda: 2,
        device=torch.device("cpu"),
        token_to_kv_pool_allocator=types.SimpleNamespace(page_size=int(sa.page_size)),
        req_to_token_pool=types.SimpleNamespace(req_to_token=r2t),
        req_pool_indices=torch.tensor([1, 2], dtype=torch.int64),
        tree_cache=None,
        seq_lens_cpu=None,
        seq_lens_sum=None,
    )
    spec = di.DFlashDraftInputV2(
        topk_p=torch.zeros(2, 1),
        topk_index=torch.zeros(2, 1, dtype=torch.int64),
        bonus_tokens=torch.zeros(2, dtype=torch.int64),
        new_seq_lens=torch.zeros(2, dtype=torch.int64),
        hidden_states=torch.zeros(2, 4),
    )
    return spec, batch, r2t, reqs, calls, neighbor_len


def _ceil(x, p):
    return (x + p - 1) // p * p


# ---------------------------------------------------------------------------------------------------------------------------------
# the repro (base-compatible imports only)
# ---------------------------------------------------------------------------------------------------------------------------------
def test_nf_context_end_prepare_keeps_the_neighbor_row(monkeypatch):
    sa = _nf(monkeypatch)
    width = CTX + mc.get_req_to_token_extra_context_len(sa)  # what the model runner allocates
    committed = CTX - 4  # the largest committed length of a request the host still prepares
    allocated = _ceil(committed - 4, 64)  # the watermark the round before left (262144)
    spec, batch, r2t, reqs, calls, nlen = _run_prepare(monkeypatch, sa, width, committed, allocated)
    neighbor_before = r2t[2].clone()
    spec.prepare_for_decode(batch)
    reserve = mc.get_alloc_reserve_per_decode(sa, alloc_len=8)
    nxt = _ceil(committed + reserve, 64)
    assert nxt == CTX + 64  # 262208: the page rounding of the watermark
    assert calls == [nxt - allocated] and reqs[0].kv_allocated_len == nxt
    assert torch.equal(r2t[2], neighbor_before), (
        f"the neighbor row was overwritten: width {width}, watermark {nxt}, columns 0..{int((r2t[2] != neighbor_before).sum()) - 1}"
    )
    assert torch.equal(r2t[1, allocated:nxt].to(torch.int64), torch.arange(SLOT0, SLOT0 + nxt - allocated))


def test_27b_context_end_fits(monkeypatch):
    """27B (DFLASH, page 1, no Form B): watermark committed + 16 -- 262156 at most, the old width exactly; still in its row."""
    sa = _b27(monkeypatch)
    width = CTX + mc.get_req_to_token_extra_context_len(sa)
    committed = CTX - 4
    spec, batch, r2t, reqs, calls, _ = _run_prepare(monkeypatch, sa, width, committed, committed)
    before = r2t[2].clone()
    spec.prepare_for_decode(batch)
    assert reqs[0].kv_allocated_len == CTX + 12 and calls == [16]
    assert torch.equal(r2t[2], before)


# ---------------------------------------------------------------------------------------------------------------------------------
# the bound
# ---------------------------------------------------------------------------------------------------------------------------------
def test_headroom_numbers(monkeypatch):
    sa = _nf(monkeypatch)
    assert mc.get_req_to_token_row_headroom(sa) == 16 + 63 == mc.get_req_to_token_extra_context_len(sa)
    sa = _b27(monkeypatch)
    assert mc.get_req_to_token_row_headroom(sa) == 16
    assert mc.get_req_to_token_row_headroom(_sa(algo=None, k=None, page=1)) >= 4


@pytest.mark.parametrize("page", [1, 16, 64])
@pytest.mark.parametrize("ctx", [CTX, CTX - 37, 8192])
def test_every_reachable_watermark_fits_the_row(monkeypatch, page, ctx):
    sa = _sa(page=page)
    headroom = mc.get_req_to_token_extra_context_len(sa)
    for reserve in (mc.get_alloc_reserve_per_decode(sa), mc.get_alloc_reserve_per_decode(sa, alloc_len=8)):
        for c in range(ctx - 300, ctx - 3):  # every committed length a running request reaches
            assert _ceil(c + reserve, page) <= ctx + headroom, (c, reserve, page)


def test_old_headroom_did_not_count_the_page():
    """The arithmetic the old width missed: 4 + 8 columns against a watermark 64 past the context."""
    assert _ceil(CTX - 4 + 16, 64) - (CTX + 4 + 8) == 52
    assert (CTX - 4 + 16) - (CTX + 4 + 8) == 0  # 27B: page 1, zero margin


def test_every_row_allocation_site_uses_the_headroom():
    import inspect

    from flliper.srt.model_executor import model_runner_kv_cache_mixin as mix

    src = inspect.getsource(mix)
    assert src.count("extra_max_context_len = get_req_to_token_row_headroom(self.server_args, page_size=self.page_size)") == 2
    assert "extra_max_context_len = 4\n" not in src
    assert "extra_max_context_len = get_req_to_token_extra_context_len(self.server_args)" in src  # the classic path
    assert "extra = get_req_to_token_row_headroom(server_args)" in inspect.getsource(mc.get_req_to_token_extra_context_len)


# ---------------------------------------------------------------------------------------------------------------------------------
# the host check: a Python error before anything is allocated or written
# ---------------------------------------------------------------------------------------------------------------------------------
def test_dflash_prepare_raises_before_writing(monkeypatch):
    sa = _nf(monkeypatch)
    old_width = CTX + 4 + 8
    committed = CTX - 4
    spec, batch, r2t, reqs, calls, _ = _run_prepare(monkeypatch, sa, old_width, committed, CTX)
    before = r2t.clone()
    with pytest.raises(mc.ReqToTokenRowOverflow) as ei:
        spec.prepare_for_decode(batch)
    msg = str(ei.value)
    assert "DFLASH prepare_for_decode" in msg and "262208" in msg and f"{old_width}" in msg and "by 52" in msg
    assert calls == [] and reqs[0].kv_allocated_len == CTX and torch.equal(r2t, before)


def test_eagle_prepare_raises_before_writing(monkeypatch):
    from flliper.srt.speculative import eagle_utils as eu

    sa = _sa(algo="EAGLE", page=1, k=4)
    monkeypatch.setattr(mc, "get_server_args", lambda: sa)
    r2t = torch.full((3, 100), SENTINEL, dtype=torch.int32)
    reqs = [types.SimpleNamespace(kv_allocated_len=95, kv_committed_len=95, decode_batch_idx=0)]
    batch = types.SimpleNamespace(
        maybe_evict_swa=lambda: None,
        batch_size=lambda: 1,
        sampling_info=types.SimpleNamespace(penalizer_orchestrator=types.SimpleNamespace(is_required=False)),
        token_to_kv_pool_allocator=types.SimpleNamespace(page_size=1),
        reqs=reqs,
        req_to_token_pool=types.SimpleNamespace(req_to_token=r2t),
    )
    with pytest.raises(mc.ReqToTokenRowOverflow) as ei:
        eu.eagle_prepare_for_decode(batch)
    assert "EAGLE prepare_for_decode" in str(ei.value) and "103" in str(ei.value)
    assert torch.equal(r2t, torch.full((3, 100), SENTINEL, dtype=torch.int32))


def test_check_helper():
    r2t = torch.zeros(2, 10, dtype=torch.int32)
    mc.check_req_to_token_row(10, r2t, where="x")
    with pytest.raises(mc.ReqToTokenRowOverflow, match="x: req_to_token row write up to column 11 exceeds the row width 10 by 1"):
        mc.check_req_to_token_row(11, r2t, where="x")
    assert issubclass(mc.ReqToTokenRowOverflow, RuntimeError)


# ---------------------------------------------------------------------------------------------------------------------------------
# the kernel: never past the row end, whatever a caller asks
# ---------------------------------------------------------------------------------------------------------------------------------
def test_kernel_stays_in_its_row():
    width = 100
    r2t = torch.full((3, width), SENTINEL, dtype=torch.int32)
    start = torch.tensor([60, 10], dtype=torch.int64)
    end = torch.tensor([150, 20], dtype=torch.int64)  # row 1 asks for 90 columns past 60 -> 50 past its end
    out = torch.arange(1000, 1000 + 90 + 10, dtype=torch.int64)
    cl.assign_req_to_token_pool_func(torch.tensor([1, 0]), r2t, start, end, out, 2)
    assert torch.equal(r2t[1, 60:100].to(torch.int64), torch.arange(1000, 1040))  # its own columns written
    assert torch.equal(r2t[2], torch.full((width,), SENTINEL, dtype=torch.int32))  # nothing in the next row
    assert torch.equal(r2t[0, 10:20].to(torch.int64), torch.arange(1090, 1100))  # the next launch row reads its own run
    assert int((r2t[0] != SENTINEL).sum()) == 10


# ---------------------------------------------------------------------------------------------------------------------------------
# sm_120 / sm_86 compile of the changed kernel (no interpreter)
# ---------------------------------------------------------------------------------------------------------------------------------
_PY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "python"))
_CODE = textwrap.dedent(
    """
    import torch
    from flliper.kernels.ops.speculative import cache_locs as cl
    from flliper.test import triton_compile_check as tcc
    with tcc.capture(cl, "assign_req_to_token_pool") as rec:
        # the card's forms: NF row 262144 + 87 (page 64, chain) / + 79, 27B row 262144 + 16; bs 1..6 (NF seats) and 8; the
        # length tensors as prepare_for_decode passes them (int64 device buffers) and int32; the draft pool's narrow rows
        for width in (262231, 262223, 262160, 2056):
            for bs in (1, 2, 3, 6, 8):
                for dt in (torch.int64, torch.int32):
                    idx = torch.arange(bs, dtype=torch.int64)
                    r2t = torch.zeros(bs + 1, width, dtype=torch.int32)
                    z = torch.zeros(bs, dtype=dt)
                    cl.assign_req_to_token_pool_func(idx, r2t, z, z + 8, torch.zeros(8 * bs, dtype=torch.int64), bs)
    res = tcc.compile_all(rec, cc=120) + tcc.compile_all(rec, cc=86)
    bad = [(k, e) for k, e in res if e is not None]
    print("RES", [len(rec), len(res), [k + " -> " + e for k, e in bad][:4]])
    """
)


def test_assign_kernel_compiles_for_sm120_and_sm86():
    env = {k: v for k, v in os.environ.items() if k != "TRITON_INTERPRET"}
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONPATH=_PY)
    p = subprocess.run([sys.executable, "-c", _CODE], env=env, capture_output=True, text=True, timeout=1800)
    out = [line for line in p.stdout.splitlines() if line.startswith("RES ")]
    assert out, p.stdout[-2000:] + p.stderr[-3000:]
    n_launch, n_variants, bad = ast.literal_eval(out[0][4:])
    assert n_launch == 4 * 5 * 2 and n_variants >= 8, (n_launch, n_variants)
    assert bad == [], bad
