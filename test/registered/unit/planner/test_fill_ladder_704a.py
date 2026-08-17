# SPDX-License-Identifier: Apache-2.0
"""#704a: the layout ladder indexed by FILL, and what a rung change really costs.

The finding these tests exist to hold: a rung change is NOT a delta move. The
actuator is a whole-arena host->device refill, so the bytes on the wire are the
same whether one layer moves or six. Pricing a step by its moved layers makes
the ladder look like a cheap dial when it is a flip-scale event.

No pool VALUE below is bootable. The free-bytes vector is the same STRUCTURAL
fixture the #704 suite uses -- fitted against the incumbent, and therefore not
a calibrated predictor. What is asserted here is structure and arithmetic:
concurrency of the refill, the size of the mispricing, the payback identity,
and the direction the bands move once the real cost is used.
"""

import math

import pytest

from sglang.srt.planner.layout_ladder import (
    LadderInputs,
    arena_refill_ms,
    solve_fill_ladder,
    solve_layout_ladder,
    switch_payback_s,
)
from sglang.srt.planner.pp_cut import (
    PhasePoolModel,
    kv_mib_per_token_per_attn_layer_from_config,
    layer_families_from_config,
)

FAMILIES = layer_families_from_config(
    {"num_hidden_layers": 64, "full_attention_interval": 4}
)
KV_CELL = kv_mib_per_token_per_attn_layer_from_config(
    {"num_key_value_heads": 4, "head_dim": 256}, "fp8_e4m3"
)

# AUTHORITATIVE card mapping (corrected earlier in #704b): the 5090 and one
# 3080 sit on x8, the other 3080 on x4. Measured 13 / 13 / 6.4 GB/s.
# MiB/s, since the arena is quoted in MiB.
GB_S_TO_MIB_S = 1e9 / (1024.0 * 1024.0)
LINKS = (13.0 * GB_S_TO_MIB_S, 13.0 * GB_S_TO_MIB_S, 6.4 * GB_S_TO_MIB_S)

# #690-rev2 refill, per rank.
ARENA_MIB = (9614.9, 9614.9, 9614.9)


def _flat_floor(counts):
    return (1728.0, 1825.0, 2467.0)


def _inputs(**over) -> LadderInputs:
    base = dict(
        pool=PhasePoolModel(
            free_mib=(26721.2, 16051.3, 13984.3),
            weight_mib_per_layer=450.7,
            kv_mib_per_token_per_attn_layer=KV_CELL,
            arming_floor_mib=(1728.0, 1825.0, 2467.0),
            mamba_mib_per_linear_layer_per_slot=50.85,
            mamba_slots=1,
        ),
        layer_families=FAMILIES,
        arming_floor_for=_flat_floor,
        ms_per_layer=(1.7571, 7.740, 7.275),
        fixed_ms=(0.0, 0.0, 0.0),
        link_mib_per_s=LINKS,
        min_pool_tokens=200_000.0,
        prefill_tokens_per_s=20_000.0,
    )
    base.update(over)
    return LadderInputs(**base)


# --- the actuator's cost ----------------------------------------------------


def test_the_refill_is_CONCURRENT_so_the_slowest_card_sets_the_step():
    """Nothing crosses a rank boundary (no dist.* in PhaseFlipStacks.refill),
    so the refills overlap and the cost is the max, never the sum."""
    ms = arena_refill_ms(ARENA_MIB, LINKS)
    per_rank = [m / l * 1000.0 for m, l in zip(ARENA_MIB, LINKS)]
    assert ms == pytest.approx(max(per_rank))
    assert ms < sum(per_rank), "summing would triple-charge a concurrent refill"
    # The x4 card gates it, however few layers it gained.
    assert ms == pytest.approx(per_rank[2])
    assert per_rank[2] / per_rank[0] == pytest.approx(13.0 / 6.4, rel=1e-6)


def test_the_step_is_flip_scale_not_a_cheap_dial():
    """~1.58 s against the #690 fixed flip cost of 2.0-4.2 s. A rung change is
    a large fraction of a whole phase flip, which is the entire reason the
    ladder needs an economics gate rather than a threshold."""
    ms = arena_refill_ms(ARENA_MIB, LINKS)
    assert 1500.0 < ms < 1650.0
    assert 0.35 < (ms / 1000.0) / 4.2 and (ms / 1000.0) / 2.0 < 0.85


def test_a_dead_link_is_refused_rather_than_priced_as_instant():
    with pytest.raises(ValueError, match="never complete"):
        arena_refill_ms(ARENA_MIB, (LINKS[0], LINKS[1], 0.0))


def test_a_rank_count_mismatch_is_refused():
    with pytest.raises(ValueError, match="cover"):
        arena_refill_ms(ARENA_MIB, LINKS[:2])


# --- the mispricing this slice corrects -------------------------------------


def test_THE_MOVED_LAYER_ESTIMATE_UNDERSTATES_THE_STEP(_=None):
    """CAN-FAIL and the headline. The delta a cross-rank mover WOULD have sent
    is a small multiple of a layer; the actuator sends the whole arena."""
    ladder = solve_layout_ladder(_inputs(arena_refill_mib=ARENA_MIB))
    assert ladder.transitions, "fixture must produce at least one step"
    for t in ladder.transitions:
        assert t.switch_is_arena_refill
        assert t.switch_ms > t.weight_move_ms, (
            "the arena refill must cost more than the moved-layer delta, or "
            "the correction is pointless"
        )
        assert t.switch_ms / t.weight_move_ms > 1.5


def test_without_an_arena_size_the_step_is_marked_unpriced_not_free():
    ladder = solve_layout_ladder(_inputs())
    for t in ladder.transitions:
        assert t.switch_ms == 0.0
        assert t.switch_is_arena_refill is False


def test_the_real_cost_makes_the_BANDS_STRICTLY_MORE_CONSERVATIVE():
    """The consequence of the correction, in the direction safety requires:
    a longer move must be started earlier, so the ascend trigger drops."""
    cheap = {t.deeper: t for t in solve_layout_ladder(_inputs()).transitions}
    real = {
        t.deeper: t
        for t in solve_layout_ladder(_inputs(arena_refill_mib=ARENA_MIB)).transitions
    }
    shared = set(cheap) & set(real)
    assert shared, "no comparable step between the two pricings"
    for k in shared:
        assert real[k].ascend_above_tokens < cheap[k].ascend_above_tokens


# --- the payback identity ---------------------------------------------------


def test_payback_is_infinite_when_the_target_is_not_faster():
    """A switch that buys no speed never repays a stall. Stating that beats
    dividing by zero."""
    assert switch_payback_s(1.11, 1.11, 1575.0) == math.inf
    assert switch_payback_s(1.11, 1.05, 1575.0) == math.inf


def test_payback_matches_the_closed_form():
    """1.00 -> 1.11 saves 1 - 1/1.11 of every prefill second afterwards, so a
    1.575 s stall needs ~15.9 s of prefill work at the faster rung."""
    got = switch_payback_s(1.0, 1.11, 1575.0)
    assert got == pytest.approx(1.575 / (1.0 - 1.0 / 1.11), rel=1e-9)
    assert 15.0 < got < 17.0


def test_a_bigger_gain_repays_sooner():
    assert switch_payback_s(1.0, 1.30, 1575.0) < switch_payback_s(1.0, 1.11, 1575.0)


# --- the fill-indexed ladder ------------------------------------------------


def test_the_fill_ladder_picks_the_FASTEST_rung_that_still_holds_the_set():
    inp = _inputs(arena_refill_mib=ARENA_MIB)
    ladder = solve_layout_ladder(inp)
    levels = [0.0, 100_000.0, 200_000.0, 300_000.0]
    levels = [f for f in levels if f <= max(r.admit_up_to_tokens for r in ladder.rungs)]
    table = solve_fill_ladder(ladder, inp, levels)
    assert len(table) == len(levels)
    for row in table:
        admissible = [r for r in ladder.rungs if r.admit_up_to_tokens >= row.fill_tokens]
        best = max(r.pipelined_speedup for r in admissible)
        assert row.pipelined_speedup == pytest.approx(best)


def test_speed_is_free_while_the_pool_is_slack_and_is_given_back_as_it_fills():
    """The ladder's founding argument, asserted as monotonicity."""
    inp = _inputs(arena_refill_mib=ARENA_MIB)
    ladder = solve_layout_ladder(inp)
    top = max(r.admit_up_to_tokens for r in ladder.rungs)
    levels = [0.0, top * 0.25, top * 0.5, top * 0.75, top]
    table = solve_fill_ladder(ladder, inp, levels)
    speeds = [r.pipelined_speedup for r in table]
    pools = [r.pool_tokens for r in table]
    assert speeds == sorted(speeds, reverse=True), "speed must not rise with fill"
    assert pools == sorted(pools), "pool must not fall as fill rises"


def test_an_unchanged_rung_costs_NOTHING_which_is_the_point_of_a_ladder():
    inp = _inputs(arena_refill_mib=ARENA_MIB)
    ladder = solve_layout_ladder(inp)
    table = solve_fill_ladder(ladder, inp, [0.0, 1.0, 2.0])
    assert all(not r.changed for r in table[1:])
    assert all(r.switch_ms_from_previous == 0.0 for r in table)
    assert all(r.payback_prefill_s == 0.0 for r in table)


def test_a_changed_rung_carries_the_full_arena_cost():
    inp = _inputs(arena_refill_mib=ARENA_MIB)
    ladder = solve_layout_ladder(inp)
    top = max(r.admit_up_to_tokens for r in ladder.rungs)
    table = solve_fill_ladder(ladder, inp, [0.0, top])
    changed = [r for r in table if r.changed]
    if changed:  # the fixture may collapse to one rung; then there is no step
        for r in changed:
            assert r.switch_ms_from_previous == pytest.approx(
                arena_refill_ms(ARENA_MIB, LINKS)
            )


def test_the_first_row_is_never_a_switch():
    """There is nothing to switch FROM at the first observation."""
    inp = _inputs(arena_refill_mib=ARENA_MIB)
    table = solve_fill_ladder(solve_layout_ladder(inp), inp, [0.0, 50_000.0])
    assert table[0].changed is False and table[0].switch_ms_from_previous == 0.0


def test_non_monotone_fill_levels_are_refused():
    inp = _inputs(arena_refill_mib=ARENA_MIB)
    ladder = solve_layout_ladder(inp)
    with pytest.raises(ValueError, match="MONOTONE"):
        solve_fill_ladder(ladder, inp, [0.0, 200_000.0, 100_000.0])


def test_a_DRAINING_trajectory_is_admissible_and_is_where_economics_live():
    """CAN-FAIL, and it caught a real defect in this API.

    An earlier cut required ASCENDING fill levels. Rising fill only ever moves
    to roomier, slower rungs, so every step it could express was a mandatory
    retreat with an infinite payback -- the discretionary step was
    unreachable through the interface that exists to price it.
    """
    inp = _inputs(arena_refill_mib=ARENA_MIB)
    ladder = solve_layout_ladder(inp)
    top = max(r.admit_up_to_tokens for r in ladder.rungs)
    draining = solve_fill_ladder(ladder, inp, [top, top * 0.5, 0.0])
    steps = [r for r in draining if r.changed]
    assert steps, "a full drain must cross at least one rung"
    assert all(not r.mandatory for r in steps), "draining steps buy speed"
    assert all(0.0 < r.payback_prefill_s < math.inf for r in steps), (
        "a discretionary step must carry a FINITE payback, or the economics "
        "controller has nothing to decide with"
    )


def test_a_FILLING_trajectory_is_all_mandatory_retreats():
    """The mirror image, stated so the asymmetry is pinned rather than
    implied: on the way up there is no economic choice to make."""
    inp = _inputs(arena_refill_mib=ARENA_MIB)
    ladder = solve_layout_ladder(inp)
    top = max(r.admit_up_to_tokens for r in ladder.rungs)
    filling = solve_fill_ladder(ladder, inp, [0.0, top * 0.5, top])
    steps = [r for r in filling if r.changed]
    assert steps, "fixture must cross a rung while filling"
    assert all(r.mandatory for r in steps)
    assert all(r.payback_prefill_s == math.inf for r in steps), (
        "a forced retreat never repays itself in throughput; reporting a "
        "finite payback would invite a controller to decline it"
    )


def test_a_negative_occupancy_is_refused():
    inp = _inputs(arena_refill_mib=ARENA_MIB)
    with pytest.raises(ValueError, match="negative occupancy"):
        solve_fill_ladder(solve_layout_ladder(inp), inp, [-1.0])


def test_an_occupancy_no_layout_can_serve_is_a_CAPACITY_answer():
    """Refusing with 'no layout serves this' is the honest reply; silently
    returning the roomiest rung would report a layout fix for a capacity
    problem."""
    inp = _inputs(arena_refill_mib=ARENA_MIB)
    ladder = solve_layout_ladder(inp)
    too_big = max(r.admit_up_to_tokens for r in ladder.rungs) * 2.0
    with pytest.raises(ValueError, match="capacity answer"):
        solve_fill_ladder(ladder, inp, [too_big])
