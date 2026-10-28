# SPDX-License-Identifier: Apache-2.0
"""27b-r2tbound-1028: what the formhp-r2tbound-1028 port (fLLiper 6ec9c5b67) does to the 27B forms.

Row width of the req_to_token table = context_len + get_req_to_token_extra_context_len. BEFORE = the real function on f1d9f562e6
(desk/refactor-pdflip-1027, ``4 + max_draft``), run with the same arguments as here; AFTER = this tree
(``max(4 + max_draft, W + L + page - 1)``). All forms run page 1. Result:

* unchanged: DFLASH2 D=8 without overlap, NEXTN steps 2 / draft 3 (flip forms), NEXTN steps 3 / draft 4 (MTP), no spec (P);
* DFLASH2 D=8 WITH overlap: +4 columns (262156 -> 262160, 300012 -> 300016). The old width had zero margin there: committed
  C - 4 (the largest a running request reaches) + reserve 16 = C + 12 = the old width; committed C - 3 overflowed by 1;
* DFLASH D=16 with overlap: +12 -- the old width C + 20 was 8 columns short of C - 4 + 32 (a real neighbor-row write).

The host check (check_req_to_token_row) never fires on these forms for any committed <= C -- shown on the real
DFlashDraftInputV2.prepare_for_decode and eagle_prepare_for_decode at committed C - 4, C - 3 and C.
"""

import contextlib
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest
import torch

from flliper.srt.mem_cache import common as mc
from flliper.srt.speculative import dflash_info_v2 as di

SENTINEL = -7
SLOT0 = 5_000_000
CTXS = (262144, 300000)  # 27B native and the YaRN context

# name: (algorithm as resolved by handle_speculative_decoding, max draft tokens, num_steps, topk)
FORMS = {
    "dflash2_d8": ("DFLASH", 8, 1, 1),
    "dflash_d16": ("DFLASH", 16, 1, 1),
    "nextn_flip_s2_d3": ("EAGLE", 3, 2, 1),
    "nextn_mtp_s3_d4": ("EAGLE", 4, 3, 1),
    "no_spec": (None, None, None, None),
}
# (form, overlap) -> (extra BEFORE on f1d9f562e6, extra AFTER), both from the real get_req_to_token_extra_context_len
EXTRA = {
    ("dflash2_d8", True): (12, 16),
    ("dflash2_d8", False): (12, 12),
    ("dflash_d16", True): (20, 32),
    ("dflash_d16", False): (20, 20),
    ("nextn_flip_s2_d3", True): (7, 7),
    ("nextn_flip_s2_d3", False): (7, 7),
    ("nextn_mtp_s3_d4", True): (8, 8),
    ("nextn_mtp_s3_d4", False): (8, 8),
    ("no_spec", True): (4, 4),
    ("no_spec", False): (4, 4),
}
CHANGED = {("dflash2_d8", True): 4, ("dflash_d16", True): 12}


def _sa(form, *, overlap, algo=None):
    a, d, steps, topk = FORMS[form]
    return types.SimpleNamespace(
        speculative_algorithm=algo or a,
        disable_overlap_schedule=not overlap,
        max_speculative_num_draft_tokens=d,
        speculative_num_draft_tokens=d,
        speculative_num_steps=steps,
        speculative_eagle_topk=topk,
        page_size=1,
    )


# ---------------------------------------------------------------------------------------------------------------------------------
# the width, before == after except the two named forms
# ---------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("ctx", CTXS)
@pytest.mark.parametrize("key", sorted(EXTRA))
def test_row_width_before_after(key, ctx):
    form, overlap = key
    before, after = EXTRA[key]
    sa = _sa(form, overlap=overlap)
    d = FORMS[form][1] or 0
    assert before == 4 + d  # the old formula, literally
    assert mc.get_req_to_token_extra_context_len(sa) == after
    assert mc.get_req_to_token_row_headroom(sa, page_size=1) == after  # the two model_runner_kv_cache_mixin sites
    assert (ctx + after) - (ctx + before) == CHANGED.get(key, 0)


def test_concrete_widths():
    """The numbers of the report, spelled out."""
    for ctx, old, new in ((262144, 262156, 262160), (300000, 300012, 300016)):
        assert ctx + mc.get_req_to_token_extra_context_len(_sa("dflash2_d8", overlap=True)) == new
        assert ctx + mc.get_req_to_token_extra_context_len(_sa("dflash2_d8", overlap=False)) == old
        assert ctx + mc.get_req_to_token_extra_context_len(_sa("nextn_flip_s2_d3", overlap=True)) == ctx + 7


@pytest.mark.parametrize("overlap", [True, False])
def test_unresolved_nextn_alias_does_not_raise(overlap):
    """ServerArgs._pp_cut_hybrid_pool_cap runs before handle_speculative_decoding turns NEXTN into EAGLE; at page 1 the width must not
    parse the algorithm (from_string('NEXTN') raises)."""
    for form in ("nextn_flip_s2_d3", "nextn_mtp_s3_d4"):
        raw = mc.get_req_to_token_extra_context_len(_sa(form, overlap=overlap, algo="NEXTN"))
        assert raw == mc.get_req_to_token_extra_context_len(_sa(form, overlap=overlap))


# ---------------------------------------------------------------------------------------------------------------------------------
# the host check at the 27B context end: DFLASH (real prepare_for_decode)
# ---------------------------------------------------------------------------------------------------------------------------------
class _Req:
    def __init__(self, committed, allocated):
        self.kv_committed_len = committed
        self.kv_allocated_len = allocated
        self.sampling_params = types.SimpleNamespace(top_k=1)


def _dflash_batch(monkeypatch, sa, width, committed):
    monkeypatch.setattr(di, "get_server_args", lambda: sa)
    monkeypatch.setattr(mc, "get_server_args", lambda: sa)
    monkeypatch.setattr(di, "_get_overlap_plan_stream", lambda device: (None, contextlib.nullcontext()))
    calls = []

    def _alloc(tree_cache, n, *a, **k):
        calls.append(int(n))
        return torch.arange(SLOT0, SLOT0 + int(n), dtype=torch.int64)

    monkeypatch.setattr(di, "alloc_token_slots", _alloc)
    r2t = torch.full((3, width), SENTINEL, dtype=torch.int32)
    r2t[2, :1000] = torch.arange(1000, dtype=torch.int32) + 100  # the neighbor's prompt slots
    reqs = [_Req(committed, committed), _Req(1000, 1000)]
    batch = types.SimpleNamespace(
        reqs=reqs,
        batch_size=lambda: 2,
        device=torch.device("cpu"),
        token_to_kv_pool_allocator=types.SimpleNamespace(page_size=1),
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
    return spec, batch, r2t, reqs, calls


@pytest.mark.parametrize("ctx", CTXS)
@pytest.mark.parametrize("back", [4, 3, 0])  # committed = C - 4 (reachable max), C - 3 (off by one), C
@pytest.mark.parametrize("form", ["dflash2_d8", "dflash_d16"])
@pytest.mark.parametrize("overlap", [True, False])
def test_dflash_context_end_never_raises(monkeypatch, overlap, form, back, ctx):
    sa = _sa(form, overlap=overlap)
    width = ctx + mc.get_req_to_token_extra_context_len(sa)  # what the model runner allocates
    committed = ctx - back
    spec, batch, r2t, reqs, calls = _dflash_batch(monkeypatch, sa, width, committed)
    neighbor = r2t[2].clone()
    spec.prepare_for_decode(batch)  # no ReqToTokenRowOverflow
    reserve = mc.get_alloc_reserve_per_decode(sa, alloc_len=FORMS[form][1])
    assert reserve == (2 if overlap else 1) * FORMS[form][1]  # W + L
    assert reqs[0].kv_allocated_len == committed + reserve <= width
    assert calls == [2 * reserve]  # one allocation for both rows: the context-end row and the neighbor (committed 1000, allocated 1000)
    assert torch.equal(r2t[2, :1000], neighbor[:1000])  # the neighbor's prompt slots, untouched
    assert torch.equal(r2t[2, 1000 + reserve :], neighbor[1000 + reserve :])
    assert torch.equal(r2t[1, :committed], torch.full((committed,), SENTINEL, dtype=torch.int32))
    assert torch.equal(r2t[1, committed : committed + reserve].to(torch.int64), torch.arange(SLOT0, SLOT0 + reserve))


@pytest.mark.parametrize(
    "form,back,excess",
    [("dflash2_d8", 4, None), ("dflash2_d8", 3, 1), ("dflash_d16", 4, 8)],
)
def test_old_width_margins(monkeypatch, form, back, excess):
    """Why the +4 / +12: on the OLD width (C + 4 + D) the D=8 overlap form fits C - 4 with zero margin and overflows at C - 3 by one
    column; D=16 overflows by 8 at the reachable C - 4. The check stops it before anything is allocated or written."""
    ctx = 262144
    sa = _sa(form, overlap=True)
    old_width = ctx + 4 + FORMS[form][1]
    spec, batch, r2t, reqs, calls = _dflash_batch(monkeypatch, sa, old_width, ctx - back)
    before = r2t.clone()
    if excess is None:
        spec.prepare_for_decode(batch)
        assert reqs[0].kv_allocated_len == old_width
        return
    with pytest.raises(mc.ReqToTokenRowOverflow, match=f"exceeds the row width {old_width} by {excess} "):
        spec.prepare_for_decode(batch)
    assert calls == [] and torch.equal(r2t, before)


# ---------------------------------------------------------------------------------------------------------------------------------
# the host check at the 27B context end: EAGLE / NEXTN (real eagle_prepare_for_decode up to its H2D copy)
# ---------------------------------------------------------------------------------------------------------------------------------
class _PastCheck(Exception):
    pass


class _EagleBatch:
    def __init__(self, r2t, reqs):
        self.reqs = reqs
        self.req_to_token_pool = types.SimpleNamespace(req_to_token=r2t)
        self.token_to_kv_pool_allocator = types.SimpleNamespace(page_size=1)
        self.sampling_info = types.SimpleNamespace(penalizer_orchestrator=types.SimpleNamespace(is_required=False))

    def maybe_evict_swa(self):
        return None

    def batch_size(self):
        return len(self.reqs)

    @property
    def device(self):  # the first use after the host check: reaching it means the check passed
        raise _PastCheck()


@pytest.mark.parametrize("ctx", CTXS)
@pytest.mark.parametrize("back", [4, 3, 0])
@pytest.mark.parametrize("form", ["nextn_flip_s2_d3", "nextn_mtp_s3_d4"])
@pytest.mark.parametrize("overlap", [True, False])
def test_eagle_context_end_never_raises(monkeypatch, overlap, form, back, ctx):
    from flliper.srt.speculative import eagle_utils as eu

    sa = _sa(form, overlap=overlap)
    monkeypatch.setattr(mc, "get_server_args", lambda: sa)
    width = ctx + mc.get_req_to_token_extra_context_len(sa)
    committed = ctx - back
    reqs = [types.SimpleNamespace(kv_allocated_len=committed, kv_committed_len=committed, decode_batch_idx=0)]
    r2t = torch.zeros((2, width), dtype=torch.int32)
    with pytest.raises(_PastCheck):
        eu.eagle_prepare_for_decode(_EagleBatch(r2t, reqs))
    assert reqs[0].kv_allocated_len == committed + mc.get_alloc_reserve_per_decode(sa) <= width


# ---------------------------------------------------------------------------------------------------------------------------------
# sm_120 / sm_86 compile of the changed assign kernel at every 27B row width (pool_len and bs_upper are constexpr: one variant each)
# ---------------------------------------------------------------------------------------------------------------------------------
_PY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "python"))
_CODE = """
import torch
from flliper.kernels.ops.speculative import cache_locs as cl
from flliper.test import triton_compile_check as tcc
widths = sorted({ctx + e for ctx in WIDTH_CTXS for e in WIDTH_EXTRAS} | {2056})
def launch(width, bs, idx_dt, len_dt, out_dt):
    idx = torch.arange(bs, dtype=idx_dt)
    r2t = torch.zeros(bs + 1, width, dtype=torch.int32)
    z = torch.zeros(bs, dtype=len_dt)
    cl.assign_req_to_token_pool_func(idx, r2t, z, z + 8, torch.zeros(8 * bs, dtype=out_dt), bs)
with tcc.capture(cl, "assign_req_to_token_pool") as rec:
    for width in widths:  # every 27B width (old and new), the DFLASH draft window row; bs 1 and 6; int32 (DFLASH/EAGLE) and int64 lens
        for bs in (1, 6):
            for len_dt in (torch.int32, torch.int64):
                launch(width, bs, torch.int64, len_dt, torch.int64)
    for bs in (2, 3, 4, 5, 8, 16, 32, 48, 64):  # every bs_upper the wrapper derives up to 64 rows
        launch(262160, bs, torch.int64, torch.int32, torch.int64)
    for idx_dt in (torch.int32, torch.int64):  # every index / slot dtype the wrapper accepts
        for out_dt in (torch.int32, torch.int64):
            launch(262160, 8, idx_dt, torch.int32, out_dt)
res = tcc.compile_all(rec, cc=120) + tcc.compile_all(rec, cc=86)
bad = [(k, e) for k, e in res if e is not None]
print("RES", [len(rec), len(res), len(widths), [k + " -> " + e for k, e in bad][:4]])
"""


def test_assign_kernel_compiles_for_sm120_and_sm86_at_27b_widths():
    import ast
    import subprocess
    import sys

    extras = sorted({e for pair in EXTRA.values() for e in pair})
    code = _CODE.replace("WIDTH_CTXS", repr(CTXS)).replace("WIDTH_EXTRAS", repr(extras))
    env = {k: v for k, v in os.environ.items() if k != "TRITON_INTERPRET"}
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONPATH=_PY)
    p = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=3000)
    out = [line for line in p.stdout.splitlines() if line.startswith("RES ")]
    assert out, p.stdout[-2000:] + p.stderr[-3000:]
    n_launch, n_variants, n_widths, bad = ast.literal_eval(out[0][4:])
    assert n_widths == 2 * len(extras) + 1 == 15, n_widths
    assert n_launch == n_widths * 2 * 2 + 9 + 4, n_launch  # recorded wrapper starts: 0 would be no test at all
    assert n_variants >= 2 * n_widths * 2, n_variants  # at least one variant per (width, bs_upper) and card
    assert bad == [], bad
