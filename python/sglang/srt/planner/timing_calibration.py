"""#704 slice 1a-i: pin the prefill timing intercept from two measured cuts.

``PrefillTiming`` models a stage as ``fixed_ms[r] + ms_per_layer[r] * n_r`` and
its own docstring states the limitation plainly: ONE measured cut gives one
(layer-count, time) point per rank, which cannot separate the per-layer slope
from a fixed per-stage cost -- both fit that point exactly.
``prefill_timing_from_measurement`` therefore defaults ``fixed_ms`` to zero,
the OPTIMISTIC end of the family, and that is precisely why every speedup in
DESIGN_704 is an upper bound rather than a prediction.

Two cuts resolve it per rank by elimination:

    s_r = (t_b,r - t_a,r) / (n_b,r - n_a,r)
    f_r = t_a,r - s_r * n_a,r

The arithmetic is trivial. What is not trivial, and is the reason this module
exists rather than four lines at a call site, is that the result can be
confidently wrong in two specific ways:

**1. A rank whose layer count did not change is not calibrated at all.** The
slice-1a pair ``[28,20,16] -> [29,19,16]`` leaves rank2 at 16 layers in both.
The pair carries literally no information about rank2's intercept, and a solver
that quietly returned ``fixed_ms=0`` for it would look calibrated while being
exactly as optimistic as before. Refused by default.

**2. A small layer delta amplifies measurement noise straight into the slope.**
The slope is a difference of two means divided by the delta, so

    stderr(s_r) = sqrt(sigma_a^2 + sigma_b^2) / |dn_r|

With ``dn = 1`` the entire per-stage noise lands on the slope. That bites
unevenly: on this rig rank1's slope is ~7.74 ms/layer and tolerates it, while
rank0's is ~1.76 and does not. So the slice-1a pair is a GOOD calibrator for
rank1, a WEAK one for rank0, and NO calibrator for rank2 -- and the boot plan
has to say so in advance rather than discover it in the analysis.

:func:`samples_needed` turns a target precision into a chunk count before the
window, so the measurement can be planned rather than post-rationalised.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence

from sglang.srt.planner.pp_cut import PrefillTiming


class TimingCalibrationError(ValueError):
    """A calibration that cannot be trusted. Never downgraded to a warning."""


@dataclasses.dataclass(frozen=True)
class TimingPoint:
    """One measured cut: layer counts and per-stage prefill times.

    ``stage_ms_stderr`` is the STANDARD ERROR OF THE MEAN of each stage time,
    not the per-chunk spread. Supply it whenever the precision of the result
    matters, which is whenever the result is used for anything.
    """

    counts: tuple[int, ...]
    stage_ms: tuple[float, ...]
    stage_ms_stderr: tuple[float, ...] | None = None


@dataclasses.dataclass(frozen=True)
class CalibratedTiming:
    timing: PrefillTiming
    #: Per rank: was this rank's intercept actually determined by the pair?
    determined: tuple[bool, ...]
    #: ``None`` where undetermined or where no stderr was supplied -- never 0.0,
    #: because "unknown" and "exact" must not be confusable.
    slope_stderr: tuple[float | None, ...]
    intercept_stderr: tuple[float | None, ...]
    slope_relative_error: tuple[float | None, ...]


def _check_shapes(a: TimingPoint, b: TimingPoint) -> int:
    n_ranks = len(a.counts)
    for p in (a, b):
        if len(p.counts) != len(p.stage_ms):
            raise TimingCalibrationError(
                f"a timing point has {len(p.counts)} layer counts but "
                f"{len(p.stage_ms)} stage times."
            )
        if p.stage_ms_stderr is not None and len(p.stage_ms_stderr) != len(p.counts):
            raise TimingCalibrationError(
                "stage_ms_stderr must cover every rank or be omitted entirely."
            )
    if len(b.counts) != n_ranks:
        raise TimingCalibrationError(
            f"the two cuts describe different numbers of ranks: {n_ranks} vs "
            f"{len(b.counts)}."
        )
    if tuple(a.counts) == tuple(b.counts):
        raise TimingCalibrationError(
            f"both points describe the identical cut {tuple(a.counts)}; two "
            "measurements of the same layout cannot separate a per-layer slope "
            "from a fixed per-stage cost, which is the entire purpose here."
        )
    return n_ranks


def solve_timing_from_two_cuts(
    a: TimingPoint,
    b: TimingPoint,
    allow_undetermined: bool = False,
    require_relative_precision: float | None = None,
) -> CalibratedTiming:
    """Solve slope and intercept per rank from two measured cuts.

    ``allow_undetermined`` lets a rank whose layer count did not change fall
    back to the one-cut OPTIMISTIC form (all time attributed to layers,
    ``fixed_ms=0``). It is off by default because that fallback is
    indistinguishable, at the call site, from a real calibration.

    ``require_relative_precision`` gates on ``stderr(s)/|s|`` and refuses a
    result the measurement noise swamps.
    """
    n_ranks = _check_shapes(a, b)

    # STRUCTURAL refusals first, before any per-rank arithmetic. Whether a pair
    # can calibrate a rank at all is a property of the two LAYOUTS; whether the
    # numbers come out sane is a property of the MEASUREMENT. Reporting the
    # second while the first is outstanding sends the reader to inspect timings
    # when the pair was never capable of the answer.
    if not allow_undetermined:
        blind = [r for r in range(n_ranks) if int(b.counts[r]) == int(a.counts[r])]
        if blind:
            names = ", ".join(f"rank{r}" for r in blind)
            raise TimingCalibrationError(
                f"{names} hold the same layer count in BOTH cuts, so this pair "
                "carries no information about their fixed per-stage cost. "
                "Refusing to report an intercept for them: the one-cut fallback "
                "(fixed_ms=0) is the optimistic end of the family and would be "
                "indistinguishable from a measurement. Use a pair that changes "
                "these ranks' layer counts, or pass allow_undetermined=True and "
                "read `determined`."
            )

    slopes: list[float] = []
    fixed: list[float] = []
    determined: list[bool] = []
    s_err: list[float | None] = []
    f_err: list[float | None] = []
    rel: list[float | None] = []

    for r in range(n_ranks):
        na, nb = int(a.counts[r]), int(b.counts[r])
        ta, tb = float(a.stage_ms[r]), float(b.stage_ms[r])
        dn = nb - na
        if dn == 0:
            # Refused above unless allow_undetermined; here we take the
            # explicitly-requested optimistic fallback.
            if na <= 0:
                raise TimingCalibrationError(
                    f"rank{r} holds {na} layers, so its time cannot be turned "
                    "into a per-layer cost."
                )
            slopes.append(ta / na)
            fixed.append(0.0)
            determined.append(False)
            s_err.append(None)
            f_err.append(None)
            rel.append(None)
            continue

        s = (tb - ta) / dn
        f = ta - s * na
        # Clamp only FLOATING-POINT dust to zero; a materially negative
        # intercept means the linear model is wrong over this range (most
        # likely the per-layer cost is not constant), and hiding it by clipping
        # would leave a wrong model looking calibrated.
        tol = max(1e-9, 1e-6 * max(abs(ta), abs(tb)))
        if f < -tol:
            raise TimingCalibrationError(
                f"rank{r} solves to a negative fixed cost of {f:,.3f} ms "
                f"(slope {s:,.4f} ms/layer). A stage cannot cost less than "
                "nothing before its first layer, so the linear model does not "
                "hold across these two cuts -- most likely the per-layer cost "
                "is not constant over that range. Reporting rather than "
                "clamping, because a clamped fit looks calibrated and is not."
            )
        f = max(f, 0.0)

        slopes.append(s)
        fixed.append(f)
        determined.append(True)

        if a.stage_ms_stderr is None or b.stage_ms_stderr is None:
            s_err.append(None)
            f_err.append(None)
            rel.append(None)
            continue
        sa = float(a.stage_ms_stderr[r])
        sb = float(b.stage_ms_stderr[r])
        se_s = math.sqrt(sa * sa + sb * sb) / abs(dn)
        # f = ta - s*na, with s itself a function of ta and tb:
        #   df/dta = 1 + na/dn ,  df/dtb = -na/dn
        c_a = 1.0 + na / dn
        c_b = -na / dn
        se_f = math.sqrt((c_a * sa) ** 2 + (c_b * sb) ** 2)
        s_err.append(se_s)
        f_err.append(se_f)
        rel.append(se_s / abs(s) if s != 0.0 else None)

    if require_relative_precision is not None:
        target = float(require_relative_precision)
        for r, value in enumerate(rel):
            if value is None:
                continue
            if value > target:
                raise TimingCalibrationError(
                    f"rank{r}'s slope is {slopes[r]:,.4f} ms/layer with a "
                    f"standard error of {s_err[r]:,.4f}, i.e. {value:.1%} "
                    f"relative against a {target:.1%} requirement. The layer "
                    "delta is too small for this rank's slope to survive the "
                    "measurement noise; take more samples (see samples_needed) "
                    "or use a pair with a larger delta on this rank."
                )

    return CalibratedTiming(
        timing=PrefillTiming(ms_per_layer=tuple(slopes), fixed_ms=tuple(fixed)),
        determined=tuple(determined),
        slope_stderr=tuple(s_err),
        intercept_stderr=tuple(f_err),
        slope_relative_error=tuple(rel),
    )


def samples_needed(
    per_sample_sd_ms: float,
    layer_delta: int,
    slope_ms: float,
    relative_target: float,
) -> int:
    """Chunk samples per stage needed to reach ``relative_target`` on the slope.

    The standard error of a mean falls as ``1/sqrt(N)``, so

        sqrt(2) * sd / (sqrt(N) * |dn| * s)  <=  target

    Answered BEFORE the window so the measurement can be planned; discovering
    afterwards that the samples could never have supported the claim is the
    expensive way to learn it.
    """
    if layer_delta == 0:
        raise TimingCalibrationError(
            "a layer delta of zero can never determine the slope, at any sample count."
        )
    if slope_ms <= 0.0 or relative_target <= 0.0 or per_sample_sd_ms < 0.0:
        raise TimingCalibrationError(
            "slope, target and spread must be positive to size a sample."
        )
    need = (
        2.0
        * per_sample_sd_ms**2
        / (float(layer_delta) ** 2 * slope_ms**2 * relative_target**2)
    )
    return max(1, math.ceil(need))


def describe_pair(
    a: TimingPoint, b: TimingPoint, slopes_hint: Sequence[float] | None = None
) -> str:
    """Human-readable statement of what a candidate pair can and cannot pin."""
    lines = []
    for r in range(len(a.counts)):
        dn = int(b.counts[r]) - int(a.counts[r])
        if dn == 0:
            lines.append(f"rank{r}: UNCHANGED ({a.counts[r]} layers) -- not calibrated")
        else:
            hint = ""
            if slopes_hint is not None and r < len(slopes_hint) and slopes_hint[r]:
                hint = f", expected dt ~ {abs(dn) * float(slopes_hint[r]):,.2f} ms"
            lines.append(f"rank{r}: {a.counts[r]} -> {b.counts[r]} (dn={dn:+d}){hint}")
    return "\n".join(lines)
