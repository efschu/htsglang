"""Tests for L15-02 pure hold-set selection policy (flliper.srt.pdflip.l15_policy).

Hermetic: no CUDA, no model, no network.

Run:
  cd /spinning/wt-l15-0930 && CUDA_VISIBLE_DEVICES="" \
    PYTHONPATH=/spinning/wt-l15-0930/python \
    /spinning/htsglang-gpu/.venv/bin/python -m pytest \
    test/registered/unit/pdflip/test_pdflip_l15_policy_0930.py -q -p no:cacheprovider

Plain pytest functions only -- CustomTestCase (srt/utils/common.py) wraps tests
in retry(), which is inappropriate for deterministic pure-function tests.
"""

import random

from flliper.srt.pdflip.l15_policy import (
    Candidate,
    HoldSet,
    select_hold,
    shadow_line,
)


def cand(rid, kind, last_active, rows, anchor_depth, kv_depth):
    return Candidate(
        rid=rid,
        kind=kind,
        last_active=last_active,
        rows_by_rank=tuple(rows),
        anchor_depth=anchor_depth,
        kv_depth=kv_depth,
    )


def test_ordering_kind_younger_then_rid():
    # seat > parked > served; within a kind, younger (larger last_active)
    # first; exact last_active ties broken by rid ascending. Caps non-binding.
    cands = [
        cand("sv_young", "served", 30.0, [1], 1, 1),
        cand("pk_old",   "parked", 10.0, [1], 1, 1),
        cand("se_old",   "seat",     5.0, [1], 1, 1),
        cand("se_young", "seat",     9.0, [1], 1, 1),
        cand("pk_young", "parked",  20.0, [1], 1, 1),
        cand("tie_b",    "seat",     9.0, [1], 1, 1),
        cand("tie_a",    "seat",     9.0, [1], 1, 1),
    ]
    hs = select_hold(cands, [1000], 100)
    assert hs.rids == (
        "se_young", "tie_a", "tie_b", "se_old", "pk_young", "pk_old", "sv_young"
    ), hs.rids
    assert hs.excluded == ()
    assert hs.rows_by_rank == (7,)
    assert hs.anchors == 7


def test_per_rank_cap_no_room_then_smaller_admitted():
    # Cap of 100 rows on the single rank: "big" (120 rows) does not fit and is
    # skipped with reason "no_room"; the later, smaller "small" (40 rows)
    # still fits and is admitted.
    cands = [
        cand("big",   "seat", 90.0, [120], 1, 1),
        cand("small", "seat", 80.0, [ 40], 1, 1),
    ]
    hs = select_hold(cands, [100], 10)
    assert hs.rids == ("small",), hs.rids
    assert ("big", "no_room") in hs.excluded
    assert ("small", "no_room") not in hs.excluded
    assert hs.rows_by_rank == (40,)
    assert hs.anchors == 1


def test_cap_zero_rank_never_blocks():
    # Rank 0 has cap 0 (the 5090 / TP0 rank, refilled from L2 at the wake):
    # its rows are ignored for the fit check, so candidates that would
    # "exceed" rank 0 are still admitted; only cap > 0 ranks are checked.
    cands = [
        cand("a", "seat", 90.0, [50, 10], 1, 1),
        cand("b", "seat", 80.0, [30, 20], 1, 1),
    ]
    hs = select_hold(cands, [0, 100], 10)
    assert hs.rids == ("a", "b"), (hs.rids, hs.excluded)
    # Cap-0 rank rows are still included in the summed rows_by_rank.
    assert hs.rows_by_rank == (80, 30), hs.rows_by_rank
    assert hs.anchors == 2
    assert hs.excluded == ()


def test_full_capped_rank_blocks_further_candidates():
    # Regression (lead review 2026-09-30): the "not held here" decision must
    # use the ORIGINAL cap (cap 0 = 5090 / TP0), never the remaining capacity:
    # a capped rank that is exactly full (rem == 0) still blocks.
    cands = [
        cand("a", "seat", 2.0, [5, 10, 2], 100, 100),
        cand("b", "seat", 1.0, [5, 3, 2], 100, 100),
    ]
    caps = [0, 10, 10]
    hs = select_hold(cands, caps, 10)
    assert hs.rids == ("a",), hs.rids
    assert ("b", "no_room") in hs.excluded
    # Rank 1 is exactly at its cap, and no capped rank exceeds its cap.
    assert hs.rows_by_rank[1] == 10, hs.rows_by_rank
    for r, cap in enumerate(caps):
        if cap > 0:
            assert hs.rows_by_rank[r] <= cap, (r, hs.rows_by_rank, caps)


def test_anchorless_excluded():
    # anchor_depth != kv_depth -> excluded up front with reason "anchorless"
    # (not "no_room"), even though it would have fit.
    cands = [
        cand("ok",  "seat", 90.0, [5], 3, 3),
        cand("bad", "seat", 95.0, [5], 2, 7),
    ]
    hs = select_hold(cands, [100], 10)
    assert hs.rids == ("ok",), hs.rids
    assert ("bad", "anchorless") in hs.excluded
    assert ("bad", "no_room") not in hs.excluded
    assert hs.rows_by_rank == (5,)
    assert hs.anchors == 1


def test_anchor_cap_full_excluded_as_anchor_full():
    # Row caps have room but cap_anchor_slots is 1: the first candidate is
    # admitted, the second FITS in rows yet is excluded with "anchor_full"
    # (audit item 10), not "no_room". Row-cap exclusions keep "no_room".
    cands = [
        cand("a", "seat", 90.0, [10, 10], 1, 1),
        cand("b", "seat", 80.0, [10, 10], 1, 1),
        cand("c", "seat", 70.0, [200, 0], 1, 1),
    ]
    hs = select_hold(cands, [100, 100], 1)
    assert hs.rids == ("a",), hs.rids
    assert ("b", "anchor_full") in hs.excluded
    assert ("b", "no_room") not in hs.excluded
    assert ("c", "no_room") in hs.excluded
    assert hs.rows_by_rank == (10, 10)
    assert hs.anchors == 1


def test_determinism_under_permutation():
    # 20 permutations of the same candidate list from one fixed seed must
    # all produce the identical HoldSet. A single Random(42) instance is used
    # so that each shuffle() advances the RNG state (20 distinct permutations)
    # while the sequence itself is reproducible across runs.
    base = [
        cand(
            f"r{i:02d}",
            ("seat", "parked", "served")[i % 3],
            100.0 - i * 7.0,
            [i % 3 + 1, (i * 5) % 4 + 1],
            i % 4 + 1,
            i % 4 + 1,
        )
        for i in range(20)
    ]
    rng = random.Random(42)
    first = None
    for _ in range(20):
        shuffled = base[:]
        rng.shuffle(shuffled)
        caps = [100, 100]
        hs = select_hold(shuffled, caps, 25)
        # Invariant: no capped rank ever exceeds its cap.
        for r, cap in enumerate(caps):
            if cap > 0:
                assert hs.rows_by_rank[r] <= cap, (r, hs.rows_by_rank, caps)
        if first is None:
            first = hs
        else:
            assert hs == first, (hs, first)
    assert first is not None


def test_shadow_line_format():
    hs = HoldSet(
        rids=("a", "b", "c"),
        rows_by_rank=(10, 20, 0),
        anchors=3,
        excluded=(("x", "no_room"), ("y", "anchorless")),
    )
    assert shadow_line(7, hs, (50, 60, 0)) == (
        "L15-SHADOW at=sleep epoch=7 n=3 rows_by_rank=10,20,0 "
        "anchors=3 cap=50,60,0 excluded=2"
    )
