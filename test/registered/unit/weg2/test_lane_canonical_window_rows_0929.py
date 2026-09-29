"""Rank form (29.09., NF review of the kv_only_rank union): under the weightless-KV
lane with ``--uneven-token-vector 1,2,2`` every rank's PUBLISHED canonical KV window
(``_generate_storage_config`` -> ``_dcp_owner_ctx`` -> ``canonical_kv_owner_rows_for``)
must cut to exactly the token rows that rank's attention backend WRITES
(``cp_token_prefix`` over the one installed vector, flashinfer ``cp_lo``/``cp_hi``).
A window read from a different map (the F14 ``--d-kv-token-cut`` owners) would
publish foreign rows -- the task #60 class. There is ONE installed vector per
process; this pins that the window and the backend read it."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import pytest

from sglang.srt.distributed import utils as du
from sglang.srt.managers.cache_controller import canonical_kv_owner_rows_for

PAGE = 64
VEC = [1, 2, 2]


@contextmanager
def _lane(rank, vector=VEC, head=0):
    saved = du.get_cp_token_ratios()
    du.set_weightless_kv_head_rank(head)
    du.set_cp_token_ratios(list(vector))
    par = SimpleNamespace(tp_rank=rank, attn_dcp_rank=rank, attn_dcp_size=len(vector))
    try:
        with mock.patch("sglang.srt.runtime_context.get_parallel", lambda: par):
            yield
    finally:
        du.set_cp_token_ratios(saved)
        du.set_weightless_kv_head_rank(None)


def _backend_rows(rank, dcp):
    """What the flashinfer lane writes on this rank (flashinfer_backend ~979)."""
    p = du.cp_token_prefix(dcp)
    return p[-1], p[rank], p[rank + 1]


def _published(rank):
    ctx = du.uneven_dcp_owner_bounds()
    assert ctx is not None, "lane with DCP 3 must be an owner-rule boot"
    return canonical_kv_owner_rows_for(ctx, PAGE, canonical_kv_page=object())


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_lane_window_is_the_backend_rows(rank):
    with _lane(rank):
        page, S, lo, hi = _published(rank)
        assert (S, lo, hi) == _backend_rows(rank, 3)
        assert page == PAGE and S == sum(VEC)
        assert (lo, hi) == [(0, 1), (1, 3), (3, 5)][rank]


def test_lane_windows_partition_the_page_without_foreign_rows():
    seen = []
    for r in range(3):
        with _lane(r):
            _, S, lo, hi = _published(r)
            seen.append(set(range(lo, hi)))
    for a in range(3):
        for b in range(a + 1, 3):
            assert not (seen[a] & seen[b]), "two ranks publish the same rows"
    assert set().union(*seen) == set(range(sum(VEC)))


def test_the_window_follows_the_installed_vector_not_a_second_map():
    """Mutant check: a window derived from another vector (an F14 cut such as
    0,32,32 reduced to 0,1,1) disagrees with the lane backend -- so the window
    MUST come from the installed vector, which is what it reads."""
    with _lane(1):
        _, S, lo, hi = _published(1)
        f14_prefix = [0, 0, 1, 2]
        assert (S, lo, hi) != (f14_prefix[-1], f14_prefix[1], f14_prefix[2])
    with _lane(1, vector=[2, 3, 3]):
        _, S, lo, hi = _published(1)
        assert (S, lo, hi) == (8, 2, 5)


def test_page_one_lane_keeps_the_page_one_owner_form():
    with _lane(2):
        ctx = du.uneven_dcp_owner_bounds()
        assert canonical_kv_owner_rows_for(ctx, 1, canonical_kv_page=object()) is None
