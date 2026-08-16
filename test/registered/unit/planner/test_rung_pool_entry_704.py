"""#704: ONE pool entry point for the ladder, with refusal semantics inherited.

Slot-3's ``LadderInputs`` already consumes my ``PhasePoolModel`` and requires an
``arming_floor_for`` provider, but it builds the model itself in
``pool_model_for`` -- which is the drift surface: two solvers that must agree
and are computed twice. ``solve_rung_pool`` is the single surface his rung
table and boundary actuator call instead.

What it inherits, and why each matters:

* **no default reserve.** The per-rank reserve tracks CUDA-graph capture and
  does NOT transfer between layouts -- 3,818 / 5,164 / 8,848 MiB across three
  stages of one boot. An unbooted rung with no reserve source RAISES rather
  than returning a number that looks measured.
* **provenance is a field.** "measured" only when instruments for that exact
  layout are supplied; anything else is "extrapolated".
* **calibration coverage is always returned**, so a rung whose reserve came
  from a pair that cannot identify a rank says so. Every candidate cut so far,
  [33,15,16] included, keeps rank2 at 16 layers -- rank2 is unidentified.

Hermetic: pure arithmetic, no CUDA.
"""

import pytest

from sglang.srt.planner.rung_pool import (
    RungPoolSolution,
    solve_rung_pool,
)

INCUMBENT = (28, 20, 16)
INCUMBENT_ATTN = (7, 5, 4)
DEEP = (33, 15, 16)
DEEP_ATTN = (8, 4, 4)

# Measured per-stage reserves and rests from the live [28,20,16] boot.
MEASURED_RESERVE = {INCUMBENT: (8847.9, 3818.2, 5163.8)}
MEASURED_REST_MIB = {INCUMBENT: (14.472 * 1024, 7.894 * 1024, 8.375 * 1024)}

FLOORS = {INCUMBENT: (1728.0, 1825.0, 2467.0), DEEP: (2255.0, 1728.0, 2467.0)}


def _floor_for(counts):
    return FLOORS[tuple(counts)]


def _reserve_for(counts):
    return MEASURED_RESERVE.get(tuple(counts))


def _rest_for(counts):
    return MEASURED_REST_MIB.get(tuple(counts))


def test_a_booted_rung_reproduces_the_boot_and_is_marked_measured():
    sol = solve_rung_pool(
        INCUMBENT,
        INCUMBENT_ATTN,
        arming_floor_for=_floor_for,
        reserve_for=_reserve_for,
        rest_for=_rest_for,
        measured=True,
    )
    assert isinstance(sol, RungPoolSolution)
    assert sol.pool_tokens == pytest.approx(436766, rel=5e-3)
    assert sol.provenance == "measured"
    assert sol.binding_stage in (0, 1, 2)


def test_an_unbooted_rung_without_a_reserve_is_refused_not_guessed():
    """The whole reason there is one surface."""
    # rest IS available for the deep rung; only the reserve is missing, so the
    # refusal must name the reserve rather than tripping on an earlier term.
    with pytest.raises(ValueError, match="reserve"):
        solve_rung_pool(
            DEEP,
            DEEP_ATTN,
            arming_floor_for=_floor_for,
            reserve_for=lambda c: None,
            rest_for=lambda c: (14000.0, 8000.0, 8500.0),
        )


def test_the_arming_floor_provider_is_still_required():
    """Slot-3's existing contract, unchanged."""
    with pytest.raises(ValueError, match="arming_floor_for"):
        solve_rung_pool(
            INCUMBENT,
            INCUMBENT_ATTN,
            arming_floor_for=None,
            reserve_for=_reserve_for,
            rest_for=_rest_for,
        )


def test_an_extrapolated_rung_is_labelled_and_carries_a_caveat():
    sol = solve_rung_pool(
        INCUMBENT,
        INCUMBENT_ATTN,
        arming_floor_for=_floor_for,
        reserve_for=_reserve_for,
        rest_for=_rest_for,
        measured=False,
    )
    assert sol.provenance == "extrapolated"
    assert any("extrapolat" in c.lower() for c in sol.caveats)


def test_coverage_is_always_reported_and_flags_rank2():
    sol = solve_rung_pool(
        INCUMBENT,
        INCUMBENT_ATTN,
        arming_floor_for=_floor_for,
        reserve_for=_reserve_for,
        rest_for=_rest_for,
        reference_cut=DEEP,
    )
    assert len(sol.coverage) == 3
    assert not sol.coverage[2].identified
    assert any("rank2" in c or "rank 2" in c for c in sol.caveats)


def test_the_pool_is_the_min_over_stages_and_names_the_binder():
    sol = solve_rung_pool(
        INCUMBENT,
        INCUMBENT_ATTN,
        arming_floor_for=_floor_for,
        reserve_for=_reserve_for,
        rest_for=_rest_for,
        measured=True,
    )
    assert sol.pool_tokens == min(sol.per_stage_tokens)
    assert sol.per_stage_tokens[sol.binding_stage] == sol.pool_tokens


def test_a_stage_with_no_attention_layer_is_refused():
    with pytest.raises(ValueError, match="attention"):
        solve_rung_pool(
            (3, 45, 16),
            (0, 12, 4),
            arming_floor_for=lambda c: (0.0, 0.0, 0.0),
            reserve_for=lambda c: (0.0, 0.0, 0.0),
            rest_for=lambda c: (1000.0, 1000.0, 1000.0),
        )


def test_reserve_may_be_a_mapping_instead_of_a_callable():
    """Adaptability for Slot-3's call sites without a second surface."""
    sol = solve_rung_pool(
        INCUMBENT,
        INCUMBENT_ATTN,
        arming_floor_for=FLOORS,
        reserve_for=MEASURED_RESERVE,
        rest_for=MEASURED_REST_MIB,
        measured=True,
    )
    assert sol.pool_tokens == pytest.approx(436766, rel=5e-3)


def test_stage_count_mismatch_is_refused_by_name():
    with pytest.raises(ValueError, match="stage"):
        solve_rung_pool(
            INCUMBENT,
            (7, 5),
            arming_floor_for=_floor_for,
            reserve_for=_reserve_for,
            rest_for=_rest_for,
        )
