# SPDX-License-Identifier: Apache-2.0
"""L15-05: compaction plan for the L1.5 hold (pure).

The D sleep keeps a PREFIX [0, L_H) of D's global slots mapped for the held
requests (after the partial tree reset everything else is free). This suite
pins ``sglang.srt.weg2.l15_compact``:

* owner rule (``layers/dcp/owner.py``): rank ``r`` owns global slot ``L`` iff
  ``prefix[r] <= L % S < prefix[r+1]`` with ``S = prefix[-1]``; a token must
  never change owner rank, so every compacted slot stays in its class;
* ``hold_prefix``: the smallest ``L_H`` that is a multiple of ``S`` and holds
  each rank's need inside ``[0, L_H)``;
* ``compact_plan``: held slots already below ``L_H`` stay, the rest move to
  FREE slots of their own class inside ``[0, L_H)``, deterministically
  (sources ascending, targets ascending per class, moves sorted by old slot);
* ``anchor_plan``: GDN state slots of the held requests squeezed into
  ``[0, A_H)`` with ``A_H = len(anchor_slots)``.

Prefixes under test are the real ones: 27B ``[0, 7, 11, 16]`` and NF
``[0, 0, 9, 16]`` (rank 0 has ratio 0 there, so it can own no slot).
Plain pytest functions on purpose: ``CustomTestCase`` wraps tests in
``srt/utils/common.py`` ``retry()`` and this suite needs no re-runs.
"""

from __future__ import annotations

import random

import pytest

from sglang.srt.weg2.l15_compact import (
    anchor_plan,
    compact_plan,
    hold_prefix,
    owner_of,
)

# Cumulative owner vectors (layers/dcp/owner.py convention), S = prefix[-1].
P27 = (0, 7, 11, 16)  # ratios 7, 4, 5
PNF = (0, 0, 9, 16)  # ratios 0, 9, 7 -- rank 0 owns nothing

# Hand-computed deterministic scenario on P27 (S = 16).
# Classes: c0 = residue < 7, c1 = 7..10, c2 = 11..15.
# need = [2, 9, 13] -> blocks = max(ceil(2/7), ceil(9/4), ceil(13/5)) = 3
# -> L_H = 48, rows = (21, 12, 15).
HELD_27B = {
    "a": [3, 19, 27, 43, 59],  # c0: 3, 19; c2: 27, 43, 59
    "b": [7, 8, 23, 24, 39, 40, 55, 56, 71],  # all c1
    "c": [11, 75, 91, 107, 123, 139, 155, 171, 187, 203],  # all c2
}
EXPECTED_MOVES_27B = (
    (55, 9),
    (56, 10),
    (59, 12),
    (71, 25),
    (75, 13),
    (91, 14),
    (107, 15),
    (123, 28),
    (139, 29),
    (155, 30),
    (171, 31),
    (187, 44),
    (203, 45),
)
EXPECTED_NEW_27B = {
    "a": (3, 19, 27, 43, 12),
    "b": (7, 8, 23, 24, 39, 40, 9, 10, 25),
    "c": (11, 13, 14, 15, 28, 29, 30, 31, 44, 45),
}


def _ratio(prefix, r):
    return prefix[r + 1] - prefix[r]


def _capacity(l_h, prefix, r):
    """Slots of class r inside [0, l_h): (l_h // S) * ratio_r."""
    return (l_h // prefix[-1]) * _ratio(prefix, r)


def _row(slot, prefix, r):
    """Compact pool row of global slot `slot` for its owner rank r."""
    s = prefix[-1]
    return (slot // s) * _ratio(prefix, r) + (slot % s - prefix[r])


def _need_of(held, prefix):
    return [
        sum(1 for slots in held.values() for sl in slots if owner_of(sl, prefix) == r)
        for r in range(len(prefix) - 1)
    ]


def _assert_core(plan, held, prefix):
    """Checks (a)-(d) of the plan contract, reusable for scenario and property."""
    s = prefix[-1]
    n_ranks = len(prefix) - 1

    # (a) owner preserved for every move; all new slots < L_H; injective.
    for old, new in plan.moves:
        assert owner_of(new, prefix) == owner_of(old, prefix), (old, new)
    final = [x for rid in held for x in plan.new_slots[rid]]
    assert set(plan.new_slots) == set(held)
    assert all(len(plan.new_slots[rid]) == len(held[rid]) for rid in held)
    assert all(x < plan.l_h for x in final)
    assert len(set(final)) == len(final)
    # moves sorted by old slot (old slots are distinct, so plain sort works).
    assert plan.moves == tuple(sorted(plan.moves))

    # (b) L_H is a multiple of S and minimal: L_H - S misses some rank's need.
    assert plan.l_h % s == 0
    need = _need_of(held, prefix)
    if plan.l_h > 0:
        assert any(
            _capacity(plan.l_h - s, prefix, r) < need[r] for r in range(n_ranks)
        )

    # (c) slots already inside the prefix do not move.
    sources = {old for old, _ in plan.moves}
    for rid, slots in held.items():
        for i, sl in enumerate(slots):
            if sl < plan.l_h:
                assert sl not in sources
                assert plan.new_slots[rid][i] == sl

    # (d) rows_by_rank == (L_H // S) * ratio, every new row inside it.
    assert plan.rows_by_rank == tuple(
        (plan.l_h // s) * _ratio(prefix, r) for r in range(n_ranks)
    )
    for x in final:
        r = owner_of(x, prefix)
        assert 0 <= _row(x, prefix, r) < plan.rows_by_rank[r]


# --------------------------------------------------------------------------
# owner_of
# --------------------------------------------------------------------------


def test_owner_of_matches_interval_rule():
    for prefix in (P27, PNF):
        s = prefix[-1]
        for slot in range(2 * s):
            r = owner_of(slot, prefix)
            assert prefix[r] <= slot % s < prefix[r + 1]
    # NF rank 0 has ratio 0: no slot of any residue belongs to it.
    for slot in range(10 * PNF[-1]):
        assert owner_of(slot, PNF) != 0


def test_owner_of_rejects_bad_prefix():
    with pytest.raises(ValueError):
        owner_of(3, (0, 11, 7, 16))  # not non-decreasing
    with pytest.raises(ValueError):
        owner_of(3, (0,))  # no rank at all


# --------------------------------------------------------------------------
# hold_prefix
# --------------------------------------------------------------------------


def test_hold_prefix_sizes_and_zero_need():
    assert hold_prefix([2, 9, 13], P27) == 48
    assert hold_prefix([0, 0, 0], P27) == 0
    assert hold_prefix([1, 0, 0], P27) == 16  # one block covers 7 of rank 0
    assert hold_prefix([8, 0, 0], P27) == 32  # 7 < 8 needs two blocks
    assert hold_prefix([0, 10, 0], PNF) == 32  # ceil(10/9) = 2 blocks
    assert hold_prefix([0, 0, 1], PNF) == 16


def test_hold_prefix_zero_ratio_rank_names_the_rank():
    # (e) NF: rank 0 has ratio 0, so any need on it is impossible.
    with pytest.raises(ValueError, match="rank 0"):
        hold_prefix([1, 0, 0], PNF)
    with pytest.raises(ValueError, match="rank 0"):
        hold_prefix([3, 5, 5], PNF)


def test_hold_prefix_rejects_bad_input():
    with pytest.raises(ValueError):
        hold_prefix([1, 2], P27)  # length != number of ranks
    with pytest.raises(ValueError):
        hold_prefix([-1, 0, 0], P27)  # negative need


# --------------------------------------------------------------------------
# compact_plan -- hand-computed deterministic scenario on the 27B prefix
# --------------------------------------------------------------------------


def test_compact_plan_exact_scenario_27b():
    plan = compact_plan(HELD_27B, P27)
    assert plan.l_h == 48
    assert plan.rows_by_rank == (21, 12, 15)
    assert plan.moves == EXPECTED_MOVES_27B
    assert plan.new_slots == EXPECTED_NEW_27B
    _assert_core(plan, HELD_27B, P27)


def test_compact_plan_owner_and_injectivity():
    # (a)
    plan = compact_plan(HELD_27B, P27)
    _assert_core(plan, HELD_27B, P27)


def test_compact_plan_l_h_minimal():
    # (b) literally: L_H - S does not satisfy the need of rank 1 (needs 9, fits 8).
    plan = compact_plan(HELD_27B, P27)
    assert plan.l_h % P27[-1] == 0
    assert _capacity(plan.l_h - P27[-1], P27, 1) < 9
    assert _capacity(plan.l_h, P27, 1) >= 9


def test_compact_plan_in_prefix_slots_stay():
    # (c)
    plan = compact_plan(HELD_27B, P27)
    sources = {old for old, _ in plan.moves}
    inside = [sl for slots in HELD_27B.values() for sl in slots if sl < plan.l_h]
    assert set(inside) & sources == set()
    assert plan.new_slots["a"][:4] == (3, 19, 27, 43)


def test_compact_plan_rows_bounds():
    # (d)
    plan = compact_plan(HELD_27B, P27)
    assert plan.rows_by_rank == (48 // 16 * 7, 48 // 16 * 4, 48 // 16 * 5)
    for rid, slots in plan.new_slots.items():
        for x in slots:
            r = owner_of(x, P27)
            assert _row(x, P27, r) < plan.rows_by_rank[r]


def test_compact_plan_rejects_duplicate_slots():
    with pytest.raises(ValueError, match="19"):
        compact_plan({"a": [3, 19], "b": [19, 20]}, P27)


def test_compact_plan_nf_prefix_zero_ratio_rank_unreachable():
    # (e) via the plan: under NF no held slot can be of class 0, so the plan
    # always succeeds for the two live ranks; the ValueError path of
    # hold_prefix is what a class-0 need would hit (tested above).
    held = {"a": [7, 8, 9, 10, 300], "b": [11, 12, 250]}
    plan = compact_plan(held, PNF)
    _assert_core(plan, held, PNF)
    assert plan.rows_by_rank[0] == 0


# --------------------------------------------------------------------------
# anchor_plan
# --------------------------------------------------------------------------


def test_anchor_plan_basic():
    anchors = {"a": 0, "b": 40, "c": 17, "d": 3, "e": 9}
    a_h, moves = anchor_plan(anchors)
    assert a_h == 5
    # staying: 0, 3; free targets ascending: 1, 2, 4; sources ascending: 9, 17, 40
    assert moves == ((9, 1), (17, 2), (40, 4))
    stays = {v for v in anchors.values() if v < a_h}
    assert stays | {n for _, n in moves} == set(range(a_h))
    assert moves == tuple(sorted(moves))


def test_anchor_plan_rejects_duplicates():
    with pytest.raises(ValueError):
        anchor_plan({"a": 4, "b": 4})


# --------------------------------------------------------------------------
# (f) property test: 200 random states, fixed seed
# --------------------------------------------------------------------------


def test_property_200_random_compact_states():
    rng = random.Random(0x1505)
    for _ in range(200):
        prefix = rng.choice([P27, PNF])
        n_req = rng.randint(1, 8)
        counts = [rng.randint(1, 300) for _ in range(n_req)]
        slots = rng.sample(range(20000), sum(counts))
        held = {}
        i = 0
        for k, n in enumerate(counts):
            held[f"r{k}"] = slots[i : i + n]
            i += n
        plan = compact_plan(held, prefix)
        _assert_core(plan, held, prefix)


def test_property_200_random_anchor_states():
    rng = random.Random(0x0515)
    for _ in range(200):
        n = rng.randint(1, 8)
        values = rng.sample(range(20000), n)
        anchors = {f"a{j}": v for j, v in enumerate(values)}
        a_h, moves = anchor_plan(anchors)
        assert a_h == n
        stays = {v for v in values if v < a_h}
        targets = {new for _, new in moves}
        sources = {old for old, _ in moves}
        # targets (with the untouched ones) are exactly range(A_H)
        assert stays | targets == set(range(a_h))
        # injective moves, untouched ones stay, moves sorted by old slot
        assert len(targets) == len(sources) == len(moves)
        assert stays & sources == set()
        assert sources == {v for v in values if v >= a_h}
        assert all(t < a_h for t in targets)
        assert moves == tuple(sorted(moves))
