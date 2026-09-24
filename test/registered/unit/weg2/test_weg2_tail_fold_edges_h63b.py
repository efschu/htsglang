"""fnFL2 H63b, Pruefpflicht der Faltung: D sees every position [0, N) at the 4096/page edges.

Hermetic (no CUDA). The 27B line's trim of its 1-token anchor forward (xsn437)
ended two leg-2 requests with W28 because D's store read came up short by 1
unit (bigram key) and by 219 (a 4096 window). The NF fold (H63) must not have
either edge. Every case below walks one prompt of N tokens through P's REAL
arithmetic twice -- with the fold and with the H24 cut:

* the chunk loop at the 16384 budget and the END-ANCHOR decision of the chunk
  that reaches N (``PrefillAdder._weg2_end_anchor_split``);
* the GDN anchor the extra_buffer track of each chunk files
  (``ScheduleBatch._mamba_radix_cache_v2_req_prepare_for_extend``) -- the
  finish inserts the tree (and D's #1442 page keys) up to it;
* the retention key of that insert under BIGRAM keys, exact form
  (``bigram_anchor_key``: the key takes ONE token past the anchor);
* the tail hand-off geometry (``tail_handoff.spec_for`` / ``end_geometry``).

and asserts: the anchor is the same with and without the fold; its key needs
no token past the prompt and covers exactly the anchor's pages; D's resume --
store pages [0, anchor) plus the END rows [anchor, N) when P hands a part over,
else D's own extend [anchor, N) -- covers [0, N) with nothing missing and
nothing twice. N = 4096k+1, 4096k, 64k+1, 64k (and neighbours).
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.weg2 import tail_handoff as th

PAGE, RATIO, CHUNK = 64, 4, 16384


def _qsa_allocator():
    from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

    kv = object.__new__(QSATokenToKVPool)
    kv.qsa_compress_ratio = RATIO
    return SimpleNamespace(get_kvcache=lambda: kv)


@pytest.fixture
def p_group(monkeypatch):
    import sglang.srt.managers.schedule_batch as sb
    import sglang.srt.managers.schedule_policy as sp

    monkeypatch.setattr(sp, "_WEG2_END_ANCHOR", True)  # group P
    monkeypatch.setattr(sb, "get_server_args", lambda: SimpleNamespace(
        mamba_cache_chunk_size=PAGE, mamba_checkpoint_interval=None,  # the NF P form (interval None)
        enable_mamba_extra_buffer_lazy=lambda: False))
    with envs.SGLANG_WEG2_TAIL_HANDOFF.override(True), envs.SGLANG_WEG2_TAIL_ADOPT.override(True), \
            envs.SGLANG_WEG2_TAIL_SKIP_EXTEND.override(True):
        yield sp, sb


def _walk(p_group, n: int, fold: bool):
    """P's chunk loop for one N-token prompt: (forwards, anchor, extents)."""
    sp, sb = p_group
    adder = SimpleNamespace(rem_chunk_tokens=CHUNK, page_size=PAGE, token_to_kv_pool_allocator=_qsa_allocator())
    batch = SimpleNamespace(req_to_token_pool=SimpleNamespace(get_mamba_ping_pong_other_idx=lambda i: 1 - i))
    req = SimpleNamespace(rid="weg2-0-4", full_untruncated_fill_ids=list(range(n)), prefix_indices=[],
                          mamba_ping_pong_track_buffer=torch.tensor([0, 1]), mamba_next_track_idx=0,
                          mamba_branching_seqlen=None, mamba_last_track_seqlen=None)
    extents = []
    start = 0
    with envs.SGLANG_WEG2_ENABLE_P_TAIL_FOLD.override(fold):
        while start < n:
            length = min(n - start, CHUNK)
            length, _forced = sp.PrefillAdder._weg2_end_anchor_split(adder, req, start, length)
            req.prefix_indices = list(range(start))
            req.extend_range = SimpleNamespace(start=start, end=start + length, length=length)
            sb.ScheduleBatch._mamba_radix_cache_v2_req_prepare_for_extend(batch, req)
            extents.append((start, start + length))
            start += length
    return len(extents), req.mamba_last_track_seqlen, extents


def _d_resume(n: int, anchor: int):
    """What D resumes with: the tree/store covers [0, anchor) (the finish's
    retention key, exact bigram form, and the #1442 page keys), the tail part
    (when P publishes one) the rows [page_prefix, N) + the state after N."""
    from sglang.srt.mem_cache.unified_radix_cache import bigram_anchor_key

    ids = list(range(n))  # at the finish kv_committed_len = N: token_ids_full holds the prompt
    key = bigram_anchor_key(ids, anchor, None, is_bigram=True, exact=True, page_size=PAGE)
    spec = th.spec_for("weg2-0-4", ids, None, PAGE, RATIO)
    return key, spec


EDGES = [
    98305,  # 4096k+1: N-1 on a 4096 window edge
    98304,  # 4096k
    97793,  # 64k+1: N-1 on a page edge
    97792,  # 64k
    97794, 97796, 97797,  # just above a page edge (64k+2, 64k+4, 64k+5)
    97841,  # the 97k probe
    4097, 4096, 4101,  # single-chunk prompts at the first 4096 edge
    16385, 16384, 16389,  # one chunk + one token / exactly one chunk / one chunk + 5
]


@pytest.mark.parametrize("n", EDGES)
def test_fold_and_cut_leave_d_no_gap(p_group, n):
    f_fwd, f_anchor, f_ext = _walk(p_group, n, fold=True)
    c_fwd, c_anchor, c_ext = _walk(p_group, n, fold=False)
    page_floor_n1 = (n - 1) // PAGE * PAGE
    # the tree anchor (what the finish inserts and hands D) is the same both ways,
    # and it is floor_page(N-1) -- the deepest page a reader may claim
    assert f_anchor == c_anchor == page_floor_n1, (f_ext, c_ext)
    # the extents tile [0, N) in both forms; the fold saves exactly the cut's forward
    for ext in (f_ext, c_ext):
        assert ext[0][0] == 0 and ext[-1][1] == n and all(a[1] == b[0] for a, b in zip(ext, ext[1:]))
    if n % PAGE:
        assert f_fwd == c_fwd - (1 if c_ext[-1][1] - c_ext[-1][0] < f_ext[-1][1] - f_ext[-1][0] else 0)
    else:
        assert f_ext == c_ext  # N % page == 0: the cut stays under the fold
    key, spec = _d_resume(n, f_anchor)
    # the bigram key of the anchor takes ONE token past it -- inside the prompt --
    # and covers exactly the anchor's pages (no unit short, none beyond)
    assert f_anchor < n and len(key) == f_anchor
    if spec is None:
        # no partial page below the cut (N-1 within 4 of a page edge): no part,
        # D extends [anchor, N) itself -- at most 4 tokens, nothing missing
        assert 1 <= n - f_anchor <= RATIO
        return
    rows, groups, ring = th.end_geometry(spec, RATIO)
    assert spec.page_prefix == f_anchor  # the END rows start where the store's pages end
    assert spec.page_prefix + rows == n and rows <= PAGE  # ... and end at N
    assert spec.page_prefix + groups * RATIO + ring == n  # every row once: complete groups + the open one
