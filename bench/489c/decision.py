# SPDX-License-Identifier: Apache-2.0
"""#489 (c) / #726 -- the decision rules, separated from the measurement.

Pure Python: no torch, no CUDA, no device. The rules are the part that decides
whether a kernel family gets built, so they are the part that must be
falsifiable at a desk. The runner measures; this file judges; the tests drive
this file with synthetic results, including results that must produce a
DECLINE.

TWO RULES, and they are NOT the same rule. Both are evaluated and both are
printed, because they can disagree and the disagreement is informative:

* the SPEC's kill condition (#489 (c), written before any of this):
  "if the 58K point reproduces the published inversion on sm_86, the ticket
  closes." That is a stop rule about one depth on one card family.
* the BUILD rule (window brief): build if IMMA wins at ALL depths on at least
  two of the three cards; otherwise DECLINE-AGAIN, with numbers.

A run where the kill fires but the build rule would pass is not a contradiction
to be smoothed over -- it means the inversion reproduced on sm_86 while sm_120
carried the average, which is exactly the outcome #489 forbids reporting as a
win.

NEVER AVERAGE ACROSS CARDS. This rig is heterogeneous in SM version -- two
sm_86 3080s and one sm_120 5090 -- and #489 (c) says so in its first sentence.
:func:`evaluate` refuses a result set that has already been aggregated.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional

#: The rig's measured A-vs-A spread. A gain under this cannot be told from
#: noise by any measurement on this hardware, so it is not a win.
RIG_NOISE_FLOOR = 0.141

#: Depths the spec requires, in tokens. 58K is the published inversion point
#: and is never dropped from the sweep.
SPEC_DEPTHS = (1_024, 8_192, 32_768, 58_000, 131_072, 327_680)

#: Batch sizes from the written spec. The brief said 1-8; the spec says 1 and
#: 4, and the spec is what the kill condition was written against.
SPEC_BATCHES = (1, 4)

INVERSION_DEPTH = 58_000

#: Quantisation error bounds measured by #726's codec oracle, as relative RMS.
#: Fixed BEFORE the run, per the #489 (e) quality gate.
CODEC_RMS_NORMAL = 0.0059
CODEC_RMS_HEAVY_TAIL = 0.0124


class BenchError(ValueError):
    """A result set that cannot be judged. Never downgraded to a warning."""


@dataclasses.dataclass(frozen=True)
class ArmResult:
    """One arm at one (card, depth, batch) point."""

    card: str
    sm: str
    depth: int
    batch: int
    arm: str  # "int8_imma" | "fp8_deployed" | "bf16_reference"
    ms_per_round: float
    seconds_measured: float
    #: What the arm ACTUALLY ran, read back from the kernel, not inferred from
    #: the arch string (#489 (c): "log which backend each arm actually
    #: selected rather than inferring it").
    selected_shape: str
    #: Relative RMS against the codec oracle, or None for the reference arm.
    rel_rms_vs_oracle: Optional[float] = None


def _key(r: ArmResult):
    return (r.card, r.depth, r.batch)


def speedup(int8_ms: float, fp8_ms: float) -> float:
    """fp8 time over int8 time: >1 means int8 is faster."""
    if int8_ms <= 0.0:
        raise BenchError(f"non-positive int8 time {int8_ms}")
    return fp8_ms / int8_ms


def is_win(int8_ms: float, fp8_ms: float, noise_floor: float) -> bool:
    """A win must clear the noise floor, not merely be a smaller number."""
    return (speedup(int8_ms, fp8_ms) - 1.0) > noise_floor


def validate(results: List[ArmResult], *, min_seconds: float = 10.0) -> None:
    """Refuse a result set that cannot answer the question."""
    if not results:
        raise BenchError("no results")
    short = [r for r in results if r.seconds_measured < min_seconds]
    if short:
        raise BenchError(
            f"{len(short)} arm(s) measured under {min_seconds}s "
            f"(shortest {min(r.seconds_measured for r in short):.2f}s). The "
            "ms-per-round canon requires runs of at least ten seconds; a "
            "shorter run prices the clock, not the kernel."
        )
    if any(r.card.lower() in ("mean", "average", "all", "aggregate") for r in results):
        raise BenchError(
            "an aggregated card entry is present. #489 (c) requires per-card "
            "reporting on this heterogeneous rig and forbids averaging sm_86 "
            "with sm_120 -- the two families do not even run the same arm B."
        )
    for r in results:
        if not r.selected_shape:
            raise BenchError(
                f"{r.arm} at {r.card}/{r.depth} did not report which shape it "
                "selected; the spec requires that to be logged, not inferred."
            )


def accuracy_ok(results: List[ArmResult], *, heavy_tail: bool) -> bool:
    """int8 arm must stay inside the codec oracle's measured error."""
    bound = CODEC_RMS_HEAVY_TAIL if heavy_tail else CODEC_RMS_NORMAL
    for r in results:
        if r.arm != "int8_imma" or r.rel_rms_vs_oracle is None:
            continue
        if r.rel_rms_vs_oracle > bound:
            return False
    return True


def per_card(results: List[ArmResult], noise_floor: float = RIG_NOISE_FLOOR) -> Dict:
    """Per-card verdicts. The only aggregation this module performs."""
    by_point: Dict = {}
    for r in results:
        by_point.setdefault(_key(r), {})[r.arm] = r
    cards: Dict = {}
    for (card, depth, batch), arms in sorted(by_point.items()):
        if "int8_imma" not in arms or "fp8_deployed" not in arms:
            continue
        i, f = arms["int8_imma"], arms["fp8_deployed"]
        entry = cards.setdefault(
            card, {"sm": i.sm, "points": [], "wins": 0, "losses": 0}
        )
        won = is_win(i.ms_per_round, f.ms_per_round, noise_floor)
        entry["points"].append(
            {
                "depth": depth,
                "batch": batch,
                "speedup": round(speedup(i.ms_per_round, f.ms_per_round), 4),
                "win": won,
                "int8_shape": i.selected_shape,
                "fp8_shape": f.selected_shape,
            }
        )
        entry["wins" if won else "losses"] += 1
    for entry in cards.values():
        entry["wins_at_all_depths"] = entry["losses"] == 0 and entry["wins"] > 0
    return cards


def kill_condition_fired(results: List[ArmResult], noise_floor=RIG_NOISE_FLOOR) -> bool:
    """#489 (c): the 58K point reproducing the inversion on sm_86.

    "Reproduces the inversion" means int8 is SLOWER than fp8 there by more than
    the noise floor -- the published result was a large negative, and a wash is
    not a reproduction.
    """
    for card, entry in per_card(results, noise_floor).items():
        if not entry["sm"].startswith("sm_8"):
            continue
        for p in entry["points"]:
            if p["depth"] == INVERSION_DEPTH and (1.0 / p["speedup"] - 1.0) > noise_floor:
                return True
    return False


def evaluate(results: List[ArmResult], *, heavy_tail: bool = True,
             noise_floor: float = RIG_NOISE_FLOOR) -> Dict:
    """The full verdict. Both rules, stated separately."""
    validate(results)
    cards = per_card(results, noise_floor)
    killed = kill_condition_fired(results, noise_floor)
    winning_cards = [c for c, e in cards.items() if e["wins_at_all_depths"]]
    accurate = accuracy_ok(results, heavy_tail=heavy_tail)

    build = len(winning_cards) >= 2 and accurate and not killed
    if build:
        verdict = "BUILD"
    elif killed:
        verdict = "DECLINE-AGAIN (kill condition: 58K inversion reproduced on sm_86)"
    elif not accurate:
        verdict = "DECLINE-AGAIN (accuracy outside the codec oracle's measured bound)"
    else:
        verdict = (
            f"DECLINE-AGAIN (IMMA wins at all depths on {len(winning_cards)} "
            "card(s), fewer than the two required)"
        )
    return {
        "verdict": verdict,
        "build": build,
        "kill_condition_fired": killed,
        "accuracy_within_bound": accurate,
        "cards_winning_at_all_depths": sorted(winning_cards),
        "noise_floor": noise_floor,
        "per_card": cards,
    }


def render(report: Dict) -> str:
    """The end-of-run print. Per card, never averaged."""
    out = ["", "=" * 72, "#489 (c) / #726 IMMA-QK verdict", "=" * 72]
    for card, e in sorted(report["per_card"].items()):
        out.append(f"\n{card} ({e['sm']}) -- wins {e['wins']}, losses {e['losses']}")
        for p in e["points"]:
            flag = "WIN " if p["win"] else "loss"
            out.append(
                f"   d={p['depth']:>7} bs={p['batch']}  x{p['speedup']:.3f}  {flag}"
                f"   int8={p['int8_shape']}  fp8={p['fp8_shape']}"
            )
    out += [
        "",
        f"noise floor ............ {report['noise_floor']:.3f}",
        f"kill condition fired ... {report['kill_condition_fired']}",
        f"accuracy within bound .. {report['accuracy_within_bound']}",
        f"cards winning all depths {report['cards_winning_at_all_depths']}",
        "",
        f"VERDICT: {report['verdict']}",
        "=" * 72,
    ]
    return "\n".join(out)
