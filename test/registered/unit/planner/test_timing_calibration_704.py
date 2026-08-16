"""#704 slice 1a-i: pinning the prefill timing intercept from two measured cuts.

`PrefillTiming` models a stage as ``fixed_ms[r] + ms_per_layer[r] * n_r``, and
its own docstring records the limitation: ONE measured cut gives one
(layer-count, time) point per rank, which cannot separate the per-layer slope
from a fixed per-stage cost -- both fit the point exactly.
`prefill_timing_from_measurement` therefore defaults `fixed_ms` to zero, the
OPTIMISTIC end of the family, which is why every speedup in DESIGN_704 is an
upper bound rather than a prediction.

Two cuts fix that, per rank, by simple elimination. This file pins the solver
and, more importantly, pins the two ways it can mislead:

* **A rank whose layer count did not change is not calibrated by the pair.**
  The slice-1a pair `[28,20,16] -> [29,19,16]` leaves rank2 at 16 layers, so
  its intercept remains unsolvable. Returning a confident-looking number for it
  would be the worst outcome, so it is refused unless explicitly allowed.
* **A one-layer delta amplifies measurement noise into the slope.** The slope
  is a difference of two means divided by the layer delta, so its standard
  error is ``sqrt(2) * sigma / |dn|``. With `dn = 1` the whole per-stage noise
  lands on the slope. That matters unevenly: rank1's slope is ~7.74 ms/layer
  and tolerates it, rank0's is ~1.76 and does not.

Hermetic: pure arithmetic, no CUDA.
"""

import pytest
from sglang.srt.planner.timing_calibration import (
    TimingCalibrationError,
    TimingPoint,
    solve_timing_from_two_cuts,
)

INCUMBENT = (28, 20, 16)
SLICE_1A = (29, 19, 16)
# Measured at the incumbent (DESIGN_704 sec 3.2).
INCUMBENT_MS = (49.2, 154.8, 116.4)


def _synthetic(counts, slope, fixed):
    return tuple(f + s * n for n, s, f in zip(counts, slope, fixed))


def test_two_cuts_recover_a_known_slope_and_intercept_exactly():
    """Round trip on synthetic data: the arithmetic must be exact."""
    slope = (1.7571, 7.740, 7.275)
    fixed = (11.0, 23.0, 5.0)
    a = TimingPoint(
        counts=(28, 20, 16), stage_ms=_synthetic((28, 20, 16), slope, fixed)
    )
    b = TimingPoint(
        counts=(34, 15, 15), stage_ms=_synthetic((34, 15, 15), slope, fixed)
    )
    got = solve_timing_from_two_cuts(a, b)
    for r in range(3):
        assert got.timing.ms_per_layer[r] == pytest.approx(slope[r], rel=1e-9)
        assert got.timing.fixed_ms[r] == pytest.approx(fixed[r], rel=1e-9)
        assert got.determined[r] is True


def test_a_rank_with_an_unchanged_layer_count_is_refused():
    """THE trap in the slice-1a pair.

    rank2 holds 16 layers in both cuts, so the pair carries no information
    about it at all. A solver that quietly returned fixed_ms=0 for that rank
    would look calibrated and be exactly as wrong as before.
    """
    a = TimingPoint(counts=INCUMBENT, stage_ms=INCUMBENT_MS)
    b = TimingPoint(counts=SLICE_1A, stage_ms=(50.96, 147.06, 116.4))
    with pytest.raises(TimingCalibrationError, match="rank2"):
        solve_timing_from_two_cuts(a, b)


def test_an_undetermined_rank_can_be_allowed_but_is_flagged_and_stays_optimistic():
    a = TimingPoint(counts=INCUMBENT, stage_ms=INCUMBENT_MS)
    # rank0 50.5 keeps its intercept comfortably positive; this test is about
    # the UNDETERMINED flag, not about intercept sensitivity.
    b = TimingPoint(counts=SLICE_1A, stage_ms=(50.5, 147.06, 116.4))
    got = solve_timing_from_two_cuts(a, b, allow_undetermined=True)
    assert got.determined == (True, True, False)
    # The undetermined rank keeps the one-cut optimistic form: all time is
    # attributed to layers, which is the upper bound on any reallocation gain.
    assert got.timing.fixed_ms[2] == 0.0
    assert got.timing.ms_per_layer[2] == pytest.approx(116.4 / 16)
    # And its uncertainty is reported as unknown, not as zero.
    assert got.slope_stderr[2] is None


def test_slope_uncertainty_scales_as_one_over_the_layer_delta():
    """Why a one-layer probe is a weak calibrator.

    The slope is a difference of two means over the layer delta, so its
    standard error is sqrt(2)*sigma/|dn|. Doubling the delta must halve it.
    """
    sigma = 0.5
    slope = (1.7571, 7.740, 7.275)
    fixed = (0.0, 0.0, 0.0)

    def _solve(second_counts):
        a = TimingPoint(
            counts=INCUMBENT,
            stage_ms=_synthetic(INCUMBENT, slope, fixed),
            stage_ms_stderr=(sigma,) * 3,
        )
        b = TimingPoint(
            counts=second_counts,
            stage_ms=_synthetic(second_counts, slope, fixed),
            stage_ms_stderr=(sigma,) * 3,
        )
        return solve_timing_from_two_cuts(a, b)

    near = _solve((29, 19, 15))  # every rank moves by 1
    far = _solve((30, 18, 14))  # every rank moves by 2
    for r in range(3):
        assert far.slope_stderr[r] == pytest.approx(near.slope_stderr[r] / 2.0)
        assert near.slope_stderr[r] == pytest.approx(sigma * (2**0.5) / 1.0)


def test_relative_precision_gate_rejects_a_slope_the_noise_swamps():
    """A calibration that cannot beat its own noise must not be published.

    With dn=1, rank0's slope (~1.76 ms/layer) needs a per-stage standard error
    below ~0.12 ms to reach 10% relative precision. rank1's (~7.74) tolerates
    four times as much. The gate must catch the first and pass the second.
    """
    slope = (1.7571, 7.740, 7.275)
    fixed = (0.0, 0.0, 0.0)
    a = TimingPoint(
        counts=INCUMBENT,
        stage_ms=_synthetic(INCUMBENT, slope, fixed),
        stage_ms_stderr=(0.30, 0.30, 0.30),
    )
    b = TimingPoint(
        counts=(29, 19, 15),
        stage_ms=_synthetic((29, 19, 15), slope, fixed),
        stage_ms_stderr=(0.30, 0.30, 0.30),
    )
    with pytest.raises(TimingCalibrationError, match="rank0"):
        solve_timing_from_two_cuts(a, b, require_relative_precision=0.10)
    # rank1 alone would have passed the same gate.
    got = solve_timing_from_two_cuts(a, b)
    assert got.slope_relative_error[1] < 0.10
    assert got.slope_relative_error[0] > 0.10


def test_required_sample_count_is_reported_so_the_window_can_be_planned():
    """Turn the precision demand into an instruction for the boot.

    The standard error of a mean falls as 1/sqrt(N), so the harness can say how
    many chunk samples a target precision needs -- before the window, not after.
    """
    from sglang.srt.planner.timing_calibration import samples_needed

    # per-chunk spread 3 ms, dn=1, want rank0's 1.757 slope to 10%.
    n = samples_needed(
        per_sample_sd_ms=3.0, layer_delta=1, slope_ms=1.7571, relative_target=0.10
    )
    assert n > 1
    # Achieving it must actually satisfy the gate.
    sigma = 3.0 / (n**0.5)
    assert (2**0.5) * sigma / 1.7571 <= 0.10 + 1e-9
    # A bigger layer delta must need fewer samples.
    assert (
        samples_needed(
            per_sample_sd_ms=3.0, layer_delta=4, slope_ms=1.7571, relative_target=0.10
        )
        < n
    )


def test_identical_cuts_are_refused_outright():
    a = TimingPoint(counts=INCUMBENT, stage_ms=INCUMBENT_MS)
    with pytest.raises(TimingCalibrationError, match="identical"):
        solve_timing_from_two_cuts(a, a, allow_undetermined=True)


def test_a_negative_intercept_is_reported_not_silently_clamped():
    """A negative fixed cost is physically impossible and means the model is
    wrong -- most likely the per-layer cost is not linear over that range. It
    must surface rather than be clipped to zero and hidden."""
    a = TimingPoint(counts=(28, 20, 16), stage_ms=(49.2, 154.8, 116.4))
    b = TimingPoint(counts=(30, 18, 16), stage_ms=(60.0, 130.0, 116.4))
    with pytest.raises(TimingCalibrationError, match="negative"):
        solve_timing_from_two_cuts(a, b, allow_undetermined=True)


def test_mismatched_rank_counts_are_refused():
    a = TimingPoint(counts=(28, 20, 16), stage_ms=(49.2, 154.8, 116.4))
    b = TimingPoint(counts=(28, 20), stage_ms=(49.2, 154.8))
    with pytest.raises(TimingCalibrationError, match="ranks"):
        solve_timing_from_two_cuts(a, b, allow_undetermined=True)
