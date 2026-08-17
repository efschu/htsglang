"""#677: the phase window solved from flip cost and load, not set as a constant.

Calibration data for this rig lives here, not in the solver: the measured flip
fixed cost is 2.0-4.2 s (#690, with an unexplained residual), and batch
formation collapses toward size 1 below a queue of ~4 (#689).

Hermetic: pure arithmetic, no CUDA.
"""

import pytest
from sglang.srt.planner.phase_window import (
    PhaseWindowError,
    WindowInputs,
    batch_floor_s,
    latency_ceiling_s,
    solve_window,
    stability_floor_s,
)

FLIP_MIN, FLIP_MAX = 2.0, 4.2  # #690 measured band
DECODE_QUEUE = 4  # #689 batch mode


def _inputs(**over):
    kw = {
        "flip_cost_s": 3.0,
        "prefill_utilisation": 0.25,
        "decode_utilisation": 0.25,
        "ttft_budget_s": 10.0,
        "arrival_rate_per_s": 2.0,
        "min_decode_queue": DECODE_QUEUE,
    }
    kw.update(over)
    return WindowInputs(**kw)


def test_the_window_is_a_function_of_flip_cost_not_a_constant():
    """The whole point: double the flip cost, double the stability floor."""
    a = stability_floor_s(_inputs(flip_cost_s=2.0))
    b = stability_floor_s(_inputs(flip_cost_s=4.0))
    assert b == pytest.approx(2.0 * a)


def test_the_floor_diverges_as_load_approaches_saturation():
    light = stability_floor_s(_inputs(prefill_utilisation=0.1, decode_utilisation=0.1))
    heavy = stability_floor_s(
        _inputs(prefill_utilisation=0.45, decode_utilisation=0.45)
    )
    assert heavy > 4.0 * light
    with pytest.raises(PhaseWindowError, match="capacity problem"):
        stability_floor_s(_inputs(prefill_utilisation=0.6, decode_utilisation=0.6))


def test_flip_cost_closes_the_feasible_band_from_BOTH_ends():
    """The sharpest economic statement, and it is why #690 matters.

    The stability floor rises as flips*F while the latency ceiling falls as
    -F/rho_d. A dearer flip does not merely add an overhead line: it squeezes
    the admissible band from both sides at once.
    """
    cheap = _inputs(flip_cost_s=1.0)
    dear = _inputs(flip_cost_s=4.0)
    assert stability_floor_s(dear) > stability_floor_s(cheap)
    assert latency_ceiling_s(dear) < latency_ceiling_s(cheap)


def test_a_dear_flip_under_load_admits_no_window_at_all():
    """Not "a worse window" -- no window. The refusal must say so."""
    p = solve_window(
        _inputs(flip_cost_s=4.2, prefill_utilisation=0.4, decode_utilisation=0.4)
    )
    assert not p.feasible
    assert "band shuts from both ends" in p.refusal
    assert p.stability_floor_s > p.latency_ceiling_s


def test_the_economic_window_is_the_LARGEST_admissible_cycle():
    """Overhead falls as the cycle grows, so throughput wants the ceiling."""
    p = solve_window(_inputs())
    assert p.feasible
    assert p.cycle_s == pytest.approx(latency_ceiling_s(_inputs()))
    assert p.cycle_s >= p.stability_floor_s
    # And the windows split by utilisation.
    assert p.prefill_window_s == pytest.approx(0.25 * p.cycle_s)
    assert p.decode_window_s == pytest.approx(0.25 * p.cycle_s)


def test_overhead_falls_as_the_cycle_lengthens():
    tight = solve_window(_inputs(ttft_budget_s=8.0))
    loose = solve_window(_inputs(ttft_budget_s=30.0))
    assert loose.cycle_s > tight.cycle_s
    assert loose.flip_overhead_fraction < tight.flip_overhead_fraction


def test_halving_the_flip_cost_is_worth_more_than_an_overhead_line():
    """Prices #690 directly, at a fixed budget and load."""
    dear = solve_window(_inputs(flip_cost_s=4.2))
    cheap = solve_window(_inputs(flip_cost_s=2.0))
    assert dear.feasible and cheap.feasible
    assert cheap.flip_overhead_fraction < 0.5 * dear.flip_overhead_fraction


def test_the_batch_floor_keeps_the_decode_window_worth_entering():
    """#689: entering decode below the queue threshold buys batch-size 1.

    So the prefill window must last long enough to accumulate a batch, which is
    a floor on the CYCLE, not a preference.
    """
    f = batch_floor_s(_inputs(min_decode_queue=4, arrival_rate_per_s=2.0))
    assert f == pytest.approx(4 / 2.0 / 0.25)
    assert batch_floor_s(_inputs(min_decode_queue=0)) == 0.0
    # A slow arrival stream needs a longer prefill window to form the same batch.
    assert batch_floor_s(_inputs(arrival_rate_per_s=0.5)) > f


def test_the_batch_floor_can_bind_instead_of_stability():
    """At light load the batch floor is what sets the window, not stability."""
    p = solve_window(
        _inputs(
            flip_cost_s=0.5,
            prefill_utilisation=0.05,
            decode_utilisation=0.05,
            arrival_rate_per_s=0.2,
            ttft_budget_s=600.0,
        )
    )
    assert p.batch_floor_s > p.stability_floor_s
    assert p.binding_constraint in ("batch-formation", "latency")


def test_a_seam_that_cannot_arm_refuses_before_any_arithmetic():
    """Composes the #707 bound: no flip exists, so no window is meaningful."""
    p = solve_window(_inputs(seam_can_arm=False))
    assert not p.feasible
    assert "cannot arm" in p.refusal
    assert p.binding_constraint == "seam"


def test_a_pure_prefill_regime_has_no_latency_ceiling():
    """With no decode window nothing waits behind one, so only stability binds."""
    p = solve_window(_inputs(decode_utilisation=0.0, min_decode_queue=0))
    assert latency_ceiling_s(_inputs(decode_utilisation=0.0)) == float("inf")
    assert p.feasible
    assert p.cycle_s == pytest.approx(p.stability_floor_s)


def test_foreign_profiles_are_solved_the_same_way():
    """No rig constants: a different flip cost, load and budget entirely."""
    p = solve_window(
        WindowInputs(
            flip_cost_s=0.05,
            prefill_utilisation=0.6,
            decode_utilisation=0.2,
            ttft_budget_s=2.0,
            arrival_rate_per_s=50.0,
            min_decode_queue=8,
            flips_per_cycle=2,
        )
    )
    assert p.feasible
    assert p.flip_overhead_fraction < 0.02
    assert p.prefill_window_s > p.decode_window_s


def test_malformed_inputs_are_refused():
    with pytest.raises(PhaseWindowError, match="flip cost"):
        _inputs(flip_cost_s=-1.0)
    with pytest.raises(PhaseWindowError, match="at least one flip"):
        _inputs(flips_per_cycle=0)
    with pytest.raises(PhaseWindowError, match="arrival rate"):
        _inputs(arrival_rate_per_s=-1.0)
