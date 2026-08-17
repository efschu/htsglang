# SPDX-License-Identifier: Apache-2.0
"""#709 -- the A/B's decision rules, separated from the measurement.

Pure Python: no torch, no CUDA, no server. The rules decide whether uneven-TP
proportional sharding gets switched on in the decode layout, so they are the
part that must be falsifiable at a desk.

THE MEASUREMENT PROBLEM THIS FILE EXISTS TO SOLVE, and it is the whole reason
the obvious acceptance rule is wrong.

#705 prices uneven TP at **+0.780 ms/round**, derived as the difference between
an EQUAL 1/3 shard (2.506 ms) and a bandwidth-PROPORTIONAL shard (1.726 ms) for
the GDN attention family. But a bs=1 decode round is **~30 ms**. So:

    end-to-end round delta = 0.780 / 30   = 2.6 %
    rig A-vs-A noise floor                = 14.1 %

The predicted win is **5.4x SMALLER than the floor it would have to clear**. An
acceptance rule of the form "CONFIRM if the decode round improves beyond the
A-vs-A floor" would therefore DECLINE a fully real +0.780 ms/round win, every
time, on a correct implementation. That is not a conservative rule; it is a
rule that cannot return the right answer.

WHAT IS RESOLVABLE. On the family slice the same delta is 0.780 / 2.506 =
**31.1 %**, comfortably above the floor. And ``utils/collective_clock.py`` was
built for precisely this question -- its own docstring says *"the spread of
``wait`` across ranks is the shard-imbalance signal"*. Under an equal shard the
fast card finishes early and WAITS; under a proportional shard that wait should
collapse. So:

    PRIMARY   discriminator: the per-rank WAIT SPREAD (max-min across ranks).
    SECONDARY reported only: the end-to-end round, stated as UNRESOLVABLE so
              nobody reads its silence as a refutation.
    GATE      coherence: outputs must be identical. This lever is lossless;
              a changed answer voids the run regardless of speed.

CROSS-BOOT, AND WHY THE FLOOR IS MEASURED TWICE. ``--rank-tp-ratio`` is a boot
flag, so arm A and arm B are necessarily different boots -- the same-boot floor
rule cannot make this comparison same-boot. What it CAN do is forbid a floor
imported from another boot: each arm measures its OWN A-vs-A floor, and the
cross-boot delta must clear the LARGER of the two.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence

#: Rig prior. Never a substitute for the per-boot measurement.
RIG_NOISE_FLOOR = 0.141

#: #705's desk numbers, kept so the report can say what it expected.
EQUAL_SHARD_MS = 2.506
PROPORTIONAL_SHARD_MS = 1.726
PREDICTED_GAIN_MS = 0.780
TYPICAL_BS1_ROUND_MS = 30.0

#: Measured card bandwidths behind #705's proportional split, TB/s.
BANDWIDTH_TBS = (1.79, 0.76, 0.76)


class AcceptanceError(ValueError):
    """A run that cannot be judged. Never downgraded to a warning."""


@dataclasses.dataclass(frozen=True)
class RankRound:
    """One rank's decode round in one arm."""

    arm: str  # "A_equal" | "B_proportional"
    rank: int
    card: str
    batch: int
    ms_round: float
    ms_compute: float
    ms_wait: float
    seconds_measured: float


@dataclasses.dataclass(frozen=True)
class ArmRun:
    arm: str
    ratio_flag: str  # what --rank-tp-ratio actually was, verbatim
    boot_id: str
    own_noise_floor: float  # A-vs-A measured IN THIS BOOT
    rounds: List[RankRound]
    answer_digest: str  # determined-answer coherence probe


def admissible_ratios(sharded_dims: Sequence[int], n_ranks: int = 3,
                      max_sum: int = 16) -> List[tuple]:
    """Integer weight vectors this model can actually take.

    ``--rank-tp-ratio`` requires ``sum(weights)`` to divide EVERY sharded
    dimension. That constraint is why "just enable proportional" is not a
    thing: the ideal bandwidth ratio is 2.36:1:1, and whether any admissible
    vector lands near it depends on the checkpoint's dimensions, not on taste.
    """
    if not sharded_dims:
        raise AcceptanceError("no sharded dimensions given; cannot judge divisibility")
    out = []
    for total in range(n_ranks, max_sum + 1):
        if any(d % total for d in sharded_dims):
            continue
        for a in range(1, total - n_ranks + 2):
            rest = total - a
            if rest < n_ranks - 1:
                continue
            b = rest // (n_ranks - 1)
            if b < 1 or a + b * (n_ranks - 1) != total:
                continue
            out.append((a,) + (b,) * (n_ranks - 1))
    return sorted(set(out))


def ideal_bandwidth_ratio() -> float:
    return BANDWIDTH_TBS[0] / BANDWIDTH_TBS[1]


def ratio_shortfall(weights: Sequence[int]) -> float:
    """How far an admissible vector falls short of the bandwidth ideal.

    Reported rather than hidden: a 2:1:1 vector is ratio 2.0 against an ideal
    2.36, so it cannot deliver the full +0.780 and the run must not be judged
    as though it should.
    """
    got = weights[0] / weights[1]
    return (ideal_bandwidth_ratio() - got) / ideal_bandwidth_ratio()


def wait_spread(rounds: Sequence[RankRound]) -> float:
    """max-min of WAIT across ranks. The shard-imbalance signal."""
    waits = [r.ms_wait for r in rounds]
    if not waits:
        raise AcceptanceError("no rounds to measure wait spread over")
    return max(waits) - min(waits)


def validate(arm: ArmRun, *, min_seconds: float = 10.0) -> None:
    if not arm.rounds:
        raise AcceptanceError(f"arm {arm.arm} has no rounds")
    short = [r for r in arm.rounds if r.seconds_measured < min_seconds]
    if short:
        raise AcceptanceError(
            f"arm {arm.arm}: {len(short)} measurement(s) under {min_seconds}s. "
            "The ms-per-round canon bounds runs by TIME; a shorter run prices "
            "the clock, not the shard."
        )
    if arm.own_noise_floor <= 0.0:
        raise AcceptanceError(
            f"arm {arm.arm} carries no A-vs-A floor of its own. The floor is "
            "measured per boot; importing one from another boot is exactly "
            "the cross-boot delta this rule forbids."
        )
    if not arm.ratio_flag:
        raise AcceptanceError(
            f"arm {arm.arm} did not record the --rank-tp-ratio it actually ran "
            "with. The flag has three modes with different meanings ('auto' is "
            "CAPACITY-first, not speed), so an unrecorded arm is unjudgeable."
        )


def evaluate(arm_a: ArmRun, arm_b: ArmRun, *, batch: Optional[int] = None) -> Dict:
    """CONFIRM or DECLINE, on the resolvable metric."""
    validate(arm_a)
    validate(arm_b)
    if arm_a.boot_id == arm_b.boot_id:
        raise AcceptanceError(
            "both arms report the same boot_id. --rank-tp-ratio is a boot flag; "
            "if one boot produced both arms, one of them is mislabelled."
        )

    def _sel(arm):
        return [r for r in arm.rounds if batch is None or r.batch == batch]

    a_rounds, b_rounds = _sel(arm_a), _sel(arm_b)
    floor = max(arm_a.own_noise_floor, arm_b.own_noise_floor)

    a_spread, b_spread = wait_spread(a_rounds), wait_spread(b_rounds)
    spread_gain = (a_spread - b_spread) / a_spread if a_spread > 0 else 0.0

    a_round = max(r.ms_round for r in a_rounds)
    b_round = max(r.ms_round for r in b_rounds)
    round_gain = (a_round - b_round) / a_round if a_round > 0 else 0.0

    coherent = arm_a.answer_digest == arm_b.answer_digest
    resolvable = (PREDICTED_GAIN_MS / TYPICAL_BS1_ROUND_MS) > floor

    confirm = spread_gain > floor and coherent
    if not coherent:
        verdict = "DECLINE (coherence broken: the lever is lossless, so a changed answer voids it)"
    elif confirm:
        verdict = "CONFIRM"
    else:
        verdict = (
            f"DECLINE (wait-spread gain {spread_gain:+.1%} did not clear the "
            f"per-boot floor {floor:.1%})"
        )

    return {
        "verdict": verdict,
        "confirm": confirm,
        "coherent": coherent,
        "floor_used": floor,
        "floor_per_boot": {arm_a.boot_id: arm_a.own_noise_floor,
                           arm_b.boot_id: arm_b.own_noise_floor},
        "primary": {
            "metric": "wait_spread_ms (max-min across ranks)",
            "arm_a": round(a_spread, 4),
            "arm_b": round(b_spread, 4),
            "gain": round(spread_gain, 4),
            "cleared_floor": spread_gain > floor,
        },
        "secondary_not_the_discriminator": {
            "metric": "slowest-rank round ms",
            "arm_a": round(a_round, 4),
            "arm_b": round(b_round, 4),
            "gain": round(round_gain, 4),
            "end_to_end_is_resolvable": resolvable,
            "note": (
                f"#705 predicts {PREDICTED_GAIN_MS} ms on a ~"
                f"{TYPICAL_BS1_ROUND_MS:.0f} ms round = "
                f"{PREDICTED_GAIN_MS / TYPICAL_BS1_ROUND_MS:.1%}, against a "
                f"{floor:.1%} floor. A null here is EXPECTED and is not "
                "evidence against the lever."
            ),
        },
        "ratio_flags": {"arm_a": arm_a.ratio_flag, "arm_b": arm_b.ratio_flag},
    }


def render(report: Dict) -> str:
    p, s = report["primary"], report["secondary_not_the_discriminator"]
    out = ["", "=" * 72, "#709 uneven-TP proportional A/B", "=" * 72,
           f"arm A ratio: {report['ratio_flags']['arm_a']}",
           f"arm B ratio: {report['ratio_flags']['arm_b']}",
           "",
           f"PRIMARY   {p['metric']}",
           f"          A {p['arm_a']:.4f}  ->  B {p['arm_b']:.4f}   "
           f"gain {p['gain']:+.1%}   cleared floor: {p['cleared_floor']}",
           "",
           f"SECONDARY {s['metric']} (NOT the discriminator)",
           f"          A {s['arm_a']:.4f}  ->  B {s['arm_b']:.4f}   "
           f"gain {s['gain']:+.1%}",
           f"          {s['note']}",
           "",
           f"per-boot floors: {report['floor_per_boot']}  -> using "
           f"{report['floor_used']:.1%}",
           f"coherence clean: {report['coherent']}",
           "",
           f"VERDICT: {report['verdict']}", "=" * 72]
    return "\n".join(out)
