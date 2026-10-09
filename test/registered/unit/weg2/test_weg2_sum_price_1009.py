"""X-SUM-PRICE, the one rule (user 05.10. / 09.10.): X bounds what D prefills IN
TOTAL. Five call sites used to carry their own comparison; they all go through
`sum_price.fits` now, with ONE `carried` (`Front._d_carried_tokens`).
"""
from __future__ import annotations

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import sum_price as sp  # noqa: E402


def test_fits_is_inclusive_at_the_limit():
    assert sp.fits(carried=6000, tokens=6288, limit=12288)
    assert not sp.fits(carried=6000, tokens=6289, limit=12288)


def _p(rid, t, u):
    return types.SimpleNamespace(rid=rid, t_arrive=t, est_uncached=u)


def test_take_oldest_first_keeps_back_the_one_that_breaks_the_sum_and_still_takes_a_later_small_one():
    """6239 + 6239 = 12478 > 12288: the second SHORT stays queued for P; a small
    third one still fits behind the first (the take is greedy in arrival order,
    it does not stop at the first refusal)."""
    es = [_p("c", 3.0, 800), _p("a", 1.0, 6239), _p("b", 2.0, 6239)]
    taken, kept = sp.take_oldest_first(es, tokens_of=lambda e: e.est_uncached,
                                       arrived_of=lambda e: e.t_arrive, carried=0, limit=12288)
    assert [e.rid for e in taken] == ["a", "c"]
    assert [e.rid for e in kept] == ["b"]


def test_what_d_already_carries_counts_against_the_take():
    es = [_p("a", 1.0, 6239)]
    taken, kept = sp.take_oldest_first(es, tokens_of=lambda e: e.est_uncached,
                                       arrived_of=lambda e: e.t_arrive, carried=6239, limit=12288)
    assert taken == [] and [e.rid for e in kept] == ["a"]
