"""Which ranks does a boot pair calibrate? An explicit solver output.

Two independent calibrations found the same blind spot from the same cause:

* pool side (#702 rev5): two metal points give two equations for three ranks.
  Rank1 binds at [28,20,16] and rank0 at [32,16,16]; rank2 binds at neither, so
  its free constant is bounded, not identified.
* timing side (``planner/timing_calibration.py``): the slice-1a pair
  [28,20,16] <-> [29,19,16] is three different calibrators -- rank0 a weak one,
  rank1 a strong one, rank2 carrying ZERO information because its layer count
  does not move.

Every candidate cut so far keeps rank2 at 16 layers, and the [33,15,16]
discriminator does too. A window spent on it therefore cannot identify rank2 on
either axis. This module makes that visible before the window is spent.

Nothing here is rig-specific: the coverage of a pair is a property of the two
layer vectors, and the sample demand is a property of the lever and the noise.
"""

from __future__ import annotations

import dataclasses
import math
from typing import List, Optional, Sequence, Tuple


@dataclasses.dataclass(frozen=True)
class RankCoverage:
    """What one boot pair tells you about one rank."""

    rank: int
    delta_layers: int
    ms_per_layer: Optional[float] = None

    @property
    def identified(self) -> bool:
        """A rank whose layer count does not move carries NO information.

        This is not a weak lever that more samples would rescue -- the
        regressor column is identically zero, so the intercept is unidentified
        at any sample size.
        """
        return int(self.delta_layers) != 0

    def chunks_for_target(self, sd_ms: float, target_frac: float) -> float:
        """Chunks needed to estimate this rank's term to ``target_frac``.

        Two-sample standard error on the difference of two boot means:
        ``SE = SD * sqrt(2/N)``, and the signal is ``|delta_layers| *
        ms_per_layer``. Requiring ``SE <= target_frac * signal`` gives

            N = ( sqrt(2) * SD / (target_frac * signal) )^2

        Infinite when the rank is unidentified, which is the honest answer
        rather than a large finite one.
        """
        if not self.identified or not self.ms_per_layer:
            return float("inf")
        signal = abs(int(self.delta_layers)) * float(self.ms_per_layer)
        if signal <= 0.0 or target_frac <= 0.0:
            return float("inf")
        return (math.sqrt(2.0) * float(sd_ms) / (float(target_frac) * signal)) ** 2


def calibration_coverage(
    cut_a: Sequence[int],
    cut_b: Sequence[int],
    ms_per_layer: Optional[Sequence[float]] = None,
) -> Tuple[RankCoverage, ...]:
    """Per-rank coverage of the pair ``(cut_a, cut_b)``."""
    if len(cut_a) != len(cut_b):
        raise ValueError(
            f"cuts have different stage counts: {tuple(cut_a)} vs {tuple(cut_b)}."
        )
    out: List[RankCoverage] = []
    for r, (a, b) in enumerate(zip(cut_a, cut_b)):
        out.append(
            RankCoverage(
                rank=r,
                delta_layers=int(b) - int(a),
                ms_per_layer=None if ms_per_layer is None else float(ms_per_layer[r]),
            )
        )
    return tuple(out)


def suggest_rank_moving_arm(
    base: Sequence[int], rank: int, donor: Optional[int] = None
) -> List[Tuple[int, ...]]:
    """Arms that deliberately MOVE ``rank``, for when its term is load-bearing.

    Returned in both directions, because a single direction confounds the term
    with any monotone drift. The donor defaults to the largest other stage, so
    the arm stays as close to the base as possible.
    """
    base = tuple(int(n) for n in base)
    if not 0 <= int(rank) < len(base):
        raise ValueError(f"rank {rank} out of range for cut {base}.")
    if donor is None:
        donor = max((i for i in range(len(base)) if i != rank), key=lambda i: base[i])
    arms: List[Tuple[int, ...]] = []
    for step in (-1, +1):
        cand = list(base)
        cand[rank] += step
        cand[donor] -= step
        if min(cand) >= 1:
            arms.append(tuple(cand))
    return arms


def publishable_intercept(
    fixed_ms: float, standard_error_ms: Optional[float], n: int
) -> float:
    """Gate an intercept before it reaches the solver.

    Refusing beats emitting ``fixed_ms=0``: a zero reads downstream as a
    MEASUREMENT that the fixed cost is absent, when what actually happened is
    that the pair could not see it. Per-stage mean, per-chunk SD, N and the
    resulting standard error are mandatory boot outputs for exactly this
    reason.
    """
    if standard_error_ms is None:
        raise ValueError(
            f"intercept {fixed_ms} ms carries no standard error. An intercept "
            "without an SE cannot be distinguished from zero and must not be "
            "published into the solver."
        )
    if int(n) <= 0:
        raise ValueError(
            f"intercept {fixed_ms} ms rests on {n} samples; N must be positive."
        )
    if abs(float(fixed_ms)) < 2.0 * float(standard_error_ms):
        raise ValueError(
            f"intercept {fixed_ms} ms is indistinguishable from zero at "
            f"SE={standard_error_ms} ms (|value| < 2 SE, N={n}). Refusing rather "
            "than publishing a zero that would read as a measurement."
        )
    return float(fixed_ms)
