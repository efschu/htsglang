"""Which ranks does a boot pair actually calibrate? Make it a solver OUTPUT.

Two independent findings converged on the same blind spot:

* pool side (my #702 rev5): two metal points give two equations for three
  ranks. Rank1 binds at [28,20,16] and rank0 at [32,16,16]; rank2 binds at
  neither, so its free constant is BOUNDED, not identified.
* timing side (Slot-3, ``planner/timing_calibration.py``): the slice-1a pair
  [28,20,16] <-> [29,19,16] is three different calibrators -- rank0 dn=+1
  (weak: ~584 chunks at SD=3 ms for a 10 % estimate), rank1 dn=-1 (strong:
  ~31), rank2 UNCHANGED, i.e. zero intercept information.

Same structural cause: every candidate cut so far keeps rank2 at 16 layers.
The [33,15,16] discriminator does too. So a boot spent on it cannot identify
rank2 on either axis, and that must be visible BEFORE the window is spent
rather than discovered afterwards.

Two rules pinned here:

1. ``calibration_coverage`` reports, per rank, whether a pair identifies it and
   how strong the lever is. A rank whose layer count does not move is
   UNIDENTIFIED, full stop -- no amount of noise averaging recovers it.
2. An intercept without a standard error must not reach the solver.
   ``publishable_intercept`` refuses it. Slot-3's solver refuses rather than
   emitting ``fixed_ms=0``; this is the same refusal expressed as a gate, so
   the two solvers cannot disagree about what counts as measured.

Hermetic: pure arithmetic, no CUDA.
"""

import pytest

from sglang.srt.planner.calibration_coverage import (
    calibration_coverage,
    publishable_intercept,
    suggest_rank_moving_arm,
)

INCUMBENT = (28, 20, 16)
SLICE_1A = (29, 19, 16)
DISCRIMINATOR = (33, 15, 16)


def test_the_slice_1a_pair_leaves_rank2_unidentified():
    cov = calibration_coverage(INCUMBENT, SLICE_1A)
    assert cov[0].identified and cov[0].delta_layers == 1
    assert cov[1].identified and cov[1].delta_layers == -1
    assert not cov[2].identified
    assert cov[2].delta_layers == 0


def test_the_discriminator_pair_also_leaves_rank2_unidentified():
    """The finding that matters for the next window."""
    cov = calibration_coverage(INCUMBENT, DISCRIMINATOR)
    assert cov[0].identified and cov[1].identified
    assert not cov[2].identified


def test_an_unidentified_rank_reports_infinite_chunk_demand():
    """Zero lever is not 'a lot of samples'; it is unrecoverable."""
    cov = calibration_coverage(INCUMBENT, SLICE_1A)
    assert cov[2].chunks_for_target(sd_ms=3.0, target_frac=0.10) == float("inf")


def test_lever_strength_orders_the_ranks_like_the_timing_analysis():
    """rank1 (dn=-1 off a 7.74 ms/layer stage) is the strong calibrator."""
    cov = calibration_coverage(INCUMBENT, SLICE_1A, ms_per_layer=(1.757, 7.740, 7.275))
    n0 = cov[0].chunks_for_target(sd_ms=3.0, target_frac=0.10)
    n1 = cov[1].chunks_for_target(sd_ms=3.0, target_frac=0.10)
    assert n1 < n0
    # Slot-3's numbers: ~584 for rank0, ~31 for rank1.
    assert n0 == pytest.approx(584, rel=0.15)
    assert n1 == pytest.approx(31, rel=0.15)


def test_a_rank2_moving_arm_is_suggested_when_rank2_is_load_bearing():
    """Schedule the gap deliberately instead of discovering it after a boot."""
    arms = suggest_rank_moving_arm(INCUMBENT, rank=2)
    assert arms, "no arm proposed"
    for a in arms:
        assert sum(a) == sum(INCUMBENT)
        assert a[2] != INCUMBENT[2]
    assert any(a[2] == 15 for a in arms)
    assert any(a[2] == 17 for a in arms)


def test_an_intercept_without_a_standard_error_is_refused():
    with pytest.raises(ValueError, match="standard error"):
        publishable_intercept(fixed_ms=12.0, standard_error_ms=None, n=100)


def test_an_intercept_with_no_samples_is_refused():
    with pytest.raises(ValueError, match="samples"):
        publishable_intercept(fixed_ms=12.0, standard_error_ms=1.0, n=0)


def test_an_intercept_indistinguishable_from_zero_is_refused_not_zeroed():
    """Refusing beats emitting fixed_ms=0, which reads as a measurement."""
    with pytest.raises(ValueError, match="indistinguishable"):
        publishable_intercept(fixed_ms=0.4, standard_error_ms=3.0, n=50)


def test_a_well_separated_intercept_is_published():
    v = publishable_intercept(fixed_ms=12.0, standard_error_ms=1.0, n=50)
    assert v == 12.0
