# SPDX-License-Identifier: Apache-2.0
"""#1494 / Task #51: the leftover register and the legs it removes.

Design and the derivation of every number:
/spinning/gpu-arb/weg2/DESIGN_LEFTOVER_27B_0920.md

The bands below are boot weg2xsn408's own `WEG2-BAR1 lane-time phase=collect`
lines at seq=9 (the last P->D flip) with the tag sizes from that rank's
`WEG2-DC-BREAKDOWN`. TP1 is the flip's pacemaker: leg_collects 1645 ms against
TP0's 1539 and TP2's 1462.

ONE SIMPLIFICATION, stated because it changes a number below: TP1's log
carries TWO `weights` collect legs at seq=9 (p0 89 ms, p5 224 ms). The
register is keyed by TAG, so the fixture carries the p0 one only. That lowers
p5 from the logged 1190 ms to 966 ms and the all-lane sum from 2472 to 2248;
it does not touch the busiest lane (p0, 1282 ms), which is the only figure the
register decides on.
"""

from __future__ import annotations

import pytest

from sglang.srt.weg2.leftover_register import (
    KV_TAG,
    Band,
    Card,
    critical_path_ms,
    eligible_bands,
    format_plan,
    legs_to_skip,
    plan_card,
    plan_leftovers,
)

# D TP1 on GPU-5c648f96 (3080), seq=9. (tag, resident MiB, leg ms, lane)
TP1 = [
    Band("weights_0", 924, 248, "p0"),
    Band("weights_1", 852, 221, "p0"),
    Band("weights_2", 852, 250, "p0"),
    Band("weights_3", 852, 193, "p0"),
    Band("weights_4", 852, 237, "p0"),
    Band("weights_5", 852, 44, "p0"),
    Band("weights", 1218, 89, "p0"),
    Band("weights_6", 852, 215, "p5"),
    Band("weights_7", 852, 495, "p5"),
    Band("weights_draft", 940, 256, "p5"),
    Band(KV_TAG, 6904, 0, ""),
    Band("cuda_graph", 488, 0, ""),
]

# D TP2 on GPU-62dbbae1 (3080), the tight 3080.
TP2 = [
    Band("weights_0", 894, 243, "p1"),
    Band("weights_1", 816, 218, "p1"),
    Band("weights_2", 816, 230, "p1"),
    Band("weights_3", 816, 187, "p1"),
    Band("weights_4", 816, 209, "p1"),
    Band("weights_5", 816, 60, "p1"),
    Band("weights", 1218, 96, "p1"),
    Band("weights_6", 816, 100, "p3"),
    Band(KV_TAG, 7104, 0, ""),
]


# --- rule 2: D's KV is never a candidate ------------------------------------


def test_the_kv_tag_is_never_eligible():
    """"D-KV nie opfern". It is also the biggest tag on every rank, so a
    register that ranked by size would reach for it first."""
    tags = {b.tag for b in eligible_bands(TP1)}
    assert KV_TAG not in tags


def test_the_graph_tag_is_never_eligible():
    assert "cuda_graph" not in {b.tag for b in eligible_bands(TP1)}


def test_the_kv_tag_is_not_chosen_even_on_an_empty_card():
    plan = plan_card(Card("c", free_while_p_mib=999999), TP1)
    assert KV_TAG not in plan.resident
    assert "cuda_graph" not in plan.resident


def test_a_zero_sized_band_is_not_eligible():
    assert eligible_bands([Band("weights_0", 0, 100, "p0")]) == []


# --- the lane arithmetic ----------------------------------------------------


def test_the_critical_path_is_the_busiest_lane_not_the_sum():
    """p0 carries 1282 ms and p5 carries 966 ms in this fixture (1190 in the
    log, see the module docstring); the all-lane sum would promise nearly
    twice the savings that exist."""
    assert critical_path_ms(TP1) == 1282
    assert sum(b.leg_ms for b in TP1) == 2248


def test_an_empty_band_list_has_no_critical_path():
    assert critical_path_ms([]) == 0


def test_removing_a_band_from_the_quiet_lane_saves_nothing():
    """weights_6 is on p5 (1190 ms), which is not the busiest lane."""
    before = critical_path_ms(TP1)
    after = critical_path_ms([b for b in TP1 if b.tag != "weights_6"])
    assert before == after == 1282


# --- rule 3/4: the planner term and the needle refusal ----------------------


def test_the_measured_tp1_card_keeps_nine_bands_and_saves_over_a_second():
    plan = plan_card(Card("GPU-5c648f96", free_while_p_mib=8775), TP1)
    assert plan.resident_mib <= 8775
    assert plan.saved_ms >= 1000
    assert set(plan.resident) | set(plan.travelling) == {b.tag for b in TP1}


def test_the_tight_3080_keeps_only_what_fits():
    plan = plan_card(Card("GPU-62dbbae1", free_while_p_mib=2945), TP2)
    assert plan.resident_mib <= 2945
    assert 0 < len(plan.resident) < len(eligible_bands(TP2))
    assert plan.saved_ms > 0


def test_the_5090_shaped_card_keeps_almost_nothing():
    """~2285 MiB free against 1352 MiB bands: the card P fills is not the
    place for leftovers, and it falls out of the arithmetic, not a model name."""
    big = [Band(f"weights_{i}", 1352, 200, "p4") for i in range(8)]
    plan = plan_card(Card("GPU-31d7ef41", free_while_p_mib=2285), big)
    assert plan.resident_mib <= 2285
    assert len(plan.resident) == 1


def test_value_order_is_ms_per_mib_not_size():
    """Two equal-sized bands, very different legs: the expensive leg wins.
    Measured: weights_7 495 ms vs weights_0 248 ms, both 825 MiB."""
    bands = [Band("cheap", 800, 100, "p0"), Band("dear", 800, 495, "p0")]
    plan = plan_card(Card("c", free_while_p_mib=800), bands)
    assert plan.resident == ("dear",)
    assert plan.saved_ms == 495


def test_the_planner_term_is_subtracted_from_ps_budget():
    plan = plan_card(Card("c", free_while_p_mib=2000, p_budget_mib=9000), TP1)
    assert plan.p_budget_after_mib == 9000 - plan.resident_mib


def test_a_leftover_that_would_starve_the_needle_is_refused_by_name():
    plan = plan_card(
        Card("c", free_while_p_mib=9000, p_budget_mib=9000, p_needle_floor_mib=8500),
        TP1,
    )
    assert plan.refusals, "the needle floor must bite"
    why = plan.refusals[0]
    assert "W116 Weg2LeftoverRefused" in why
    assert "under the 8500 MiB the needle needs" in why
    assert "PLANNER TERM, not a reserve" in why
    assert plan.p_budget_after_mib >= 8500


def test_a_refused_band_travels_instead_of_vanishing():
    plan = plan_card(
        Card("c", free_while_p_mib=9000, p_budget_mib=9000, p_needle_floor_mib=8500),
        TP1,
    )
    assert set(plan.resident).isdisjoint(plan.travelling)
    assert set(plan.resident) | set(plan.travelling) == {b.tag for b in TP1}


def test_no_needle_floor_means_no_refusal():
    """An unknown requirement is not a refusal."""
    plan = plan_card(Card("c", free_while_p_mib=9000, p_budget_mib=9000), TP1)
    assert plan.refusals == ()


def test_a_card_with_no_room_keeps_nothing_and_refuses_nothing():
    plan = plan_card(Card("c", free_while_p_mib=0), TP1)
    assert plan.resident == () and plan.refusals == ()
    assert plan.saved_ms == 0
    assert len(plan.travelling) == len(TP1)


# --- the legs the flip must not run -----------------------------------------


def test_legs_to_skip_is_exactly_the_resident_set():
    plan = plan_card(Card("c", free_while_p_mib=2000), TP1)
    assert legs_to_skip(plan) == frozenset(plan.resident)


def test_no_plan_skips_no_leg():
    """Skipping a leg for a band that is not resident costs the weights;
    running one for a band that is costs time. Absence takes the safe side."""
    assert legs_to_skip(None) == frozenset()


def test_a_skipped_leg_is_never_the_kv_leg():
    plan = plan_card(Card("c", free_while_p_mib=999999), TP1)
    assert KV_TAG not in legs_to_skip(plan)


# --- the multi-card plan ----------------------------------------------------


def test_cards_are_planned_independently():
    plans = plan_leftovers(
        [Card("GPU-5c648f96", 8775), Card("GPU-62dbbae1", 2945)],
        {"GPU-5c648f96": TP1, "GPU-62dbbae1": TP2},
    )
    assert set(plans) == {"GPU-5c648f96", "GPU-62dbbae1"}
    assert plans["GPU-5c648f96"].resident_mib > plans["GPU-62dbbae1"].resident_mib


def test_a_card_with_no_bands_gets_an_empty_plan():
    plans = plan_leftovers([Card("GPU-x", 9999)], {})
    assert plans["GPU-x"].resident == () and plans["GPU-x"].saved_ms == 0


def test_the_line_names_its_instrument_and_never_sums_lanes():
    line = format_plan(plan_card(Card("GPU-5c648f96", 8775), TP1))
    assert "WEG2-LEFTOVER card=GPU-5c648f96" in line
    assert "never summed across lanes" in line
    assert "P planned with that much less" in line
