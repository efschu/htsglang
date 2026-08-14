# SPDX-License-Identifier: Apache-2.0
"""The per-stage measurement pass: turn controller traces into a canon record.

WHAT IT MEASURES, AND FROM WHAT. Nothing new is instrumented here. The #363
controller already writes, per consensus boundary, the group's ms/round split
compute vs wait (``regime_runtime`` record field ``ms_decision``, produced by
``regime_ms_clock.MsStageDecider`` from a MIN-reduced group sample). The #297
reshard already reports its own ``total_ms`` per move, and the #631 seam census
already marks a cutover stage by stage. This pass READS those three and
projects them onto the three numbers the stage table refuses a candidate for
lacking -- exactly the shape ``card_rate_pass`` has to the card probe.

    gain_pct   <- mean ms/round of the reference arm vs the stage's arm
    band_pct   <- max(same-boot A-vs-A floor, within-run drift)
    flip_cost_s <- the MAXIMUM instrumented flip duration observed

THE THREE RULES THIS PASS IS BUILT AROUND, each one bought by an incident.

1. **A-vs-A FIRST, ALWAYS.** The floor comes from a pair of IDENTICAL runs
   (``--floor-a`` / ``--floor-b``), and no gain is computed without one. The
   house rule is "same-boot A-vs-A floor before any delta is read"; a delta
   read against no floor is a number with a sign.

2. **DRIFT IS NOT NOISE, AND IT WIDENS THE BAND.** #459 measured three draws
   48-51 s apart reporting a monotone 13.0 % as a noise floor where a
   back-to-back pair reported 3.0 %. So the band is the LARGER of the A-vs-A
   floor and the run's own first-half/second-half drift, and both terms are
   written to the record so a reader can see which one bound.

3. **AN UNPRICED TERM MUST NOT READ AS A FREE ONE.** No flip sample, no
   record: ``flip_cost_s`` is refused rather than defaulted to zero. The
   transient census learned this the expensive way, and the stage table's own
   #578 refusal exists because the solver's placeholder zeros were readable as
   a measured zero.

WHAT THIS PASS REFUSES TO DO. It does not compare two boots that were not the
same workload, it does not read a trace with no summary line as a complete run
(a killed server's trace says "zero so far", not "zero"), and it does not
accept an arm shorter than the 10 s measurement-run floor. Each refusal names
the arm, the number and the remedy.

CLI
---
    python -m sglang.srt.planner.stage_measure_pass \
        --stage solved-enc --regime prefill_heavy --reference booted \
        --model-key /models/Qwen3.6-27B-FP8 \
        --reference-trace $B3/trace.rank0.jsonl --reference-to-round 4200 \
        --stage-trace     $B3/trace.rank0.jsonl --stage-from-round  4400 \
        --floor-a $B1/trace.rank0.jsonl --floor-b $B2/trace.rank0.jsonl \
        --flip-log $B3/boot.log \
        --source "363-act window, boots B1/B2/B3" --write

(``--regime-trace PATH`` writes ``PATH.rank<N>.jsonl``, one file per rank; the
reference and stage arms above are two SEGMENTS of one boot, because the KV
token vector is reached with ``/kv_reshard`` rather than with a boot flag.)

    python -m sglang.srt.planner.stage_measure_pass --show
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import List, Optional, Sequence

from sglang.srt.planner.stage_measure_store import (
    StageMeasurement,
    StageMeasurementError,
    StageMeasurementLibrary,
    rig_key,
    stage_measure_path,
)

__all__ = [
    "ArmSeries",
    "StageMeasurePassError",
    "avs_a_floor_pct",
    "band_from_floor_and_drift",
    "build_measurement",
    "drift_pct",
    "flip_seconds_from_log",
    "gain_pct",
    "merge_rank_arms",
    "read_arm",
]

#: `KV-RESHARD DONE 2,11,10 -> 3,10,10 (epoch 4) in 812.4 ms: ...`
#: The #297 actuator's own instrumented duration -- the only flip cost this
#: pass accepts from a log, because it is the only one measured by the mover.
_RESHARD_MS_RE = re.compile(
    r"KV-RESHARD\s+DONE\b.*?\bin\s+([0-9]+(?:\.[0-9]+)?)\s*ms", re.IGNORECASE
)

#: `[#631 seam-census] pp->tp rank 0 ... elapsed_ms=1234.5`
_SEAM_MS_RE = re.compile(r"seam-census\].*?\belapsed_ms=([0-9]+(?:\.[0-9]+)?)")

#: Two ranks' round-time means may differ by at most this (percent) before the
#: pass refuses to merge them. The series is derived from a MIN-REDUCED group
#: sample, so the ranks should agree to the quantum; a real disagreement means
#: the traces are from different boots or different windows, and merging them
#: would average two runs into one that never happened.
_RANK_AGREEMENT_PCT = 1.0

#: Below this many boundaries an arm is refused outright, before the seconds
#: floor is even consulted. Mirrors `MsRoundWindow`'s own readiness floor.
MIN_BOUNDARIES = 8


class StageMeasurePassError(RuntimeError):
    """A refusal with a named cause. Never a fallback."""


class ArmSeries:
    """One boot's ms/round series, as read from one rank's verdict trace."""

    def __init__(
        self,
        path: str,
        round_ms: Sequence[float],
        *,
        interval: int,
        rank: Optional[int],
        has_summary: bool,
        wait_shares: Sequence[Optional[float]] = (),
        regimes: Sequence[str] = (),
    ):
        self.path = path
        self.round_ms = [float(x) for x in round_ms]
        self.interval = max(1, int(interval))
        self.rank = rank
        self.has_summary = bool(has_summary)
        self.wait_shares = list(wait_shares)
        self.regimes = list(regimes)

    def __len__(self) -> int:
        return len(self.round_ms)

    @property
    def mean_ms(self) -> float:
        if not self.round_ms:
            raise StageMeasurePassError(f"{self.path}: no ms/round samples")
        return sum(self.round_ms) / len(self.round_ms)

    @property
    def covered_s(self) -> float:
        """Device time this arm covers, seconds.

        THE TRACE HAS NO WALL CLOCK (#363 window finding F3: records carry
        ``round`` and ``epoch`` and no timestamp), so the covered time is
        reconstructed from what the trace does carry: each boundary's mean
        round time times the rounds that boundary spans. That is device time
        under load, which is the quantity the 10 s canon is about -- an idle
        server accumulating wall-clock seconds is not a measurement run.
        """
        return sum(self.round_ms) * self.interval / 1000.0

    @property
    def mean_wait_share(self) -> Optional[float]:
        vals = [w for w in self.wait_shares if w is not None]
        if not vals:
            return None
        return sum(vals) / len(vals)


def read_arm(
    path: str,
    *,
    regime: Optional[str] = None,
    warmup: int = 0,
    require_summary: bool = True,
    from_round: Optional[int] = None,
    to_round: Optional[int] = None,
) -> ArmSeries:
    """Read one rank's verdict trace into a ms/round series.

    ``regime`` selects boundaries the controller itself labelled with that
    regime -- the #363 window's F3 remedy for a trace with no timestamps.
    ``warmup`` drops the first N boundaries AFTER that selection, because the
    window's first boundaries are the ms window filling up.

    ``from_round`` / ``to_round`` bound the segment by the controller's own
    round index. THE REASON THIS EXISTS RATHER THAN A TIMESTAMP: a stage's arm
    is often the SAME BOOT after a reshard moved it onto the candidate layout
    (the KV token vector is not a boot flag; ``/kv_reshard`` is how a server
    reaches it), so the arm is a segment of one trace and not a whole file.
    The round is the only monotone index the trace carries -- there is no wall
    clock in it -- and it is replicated across ranks, so the same bounds select
    the same window on every rank's file.
    """
    if not os.path.exists(path):
        raise StageMeasurePassError(f"trace {path!r} does not exist")
    interval = 1
    rank: Optional[int] = None
    has_summary = False
    ms: List[float] = []
    waits: List[Optional[float]] = []
    regimes: List[str] = []
    with open(path) as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise StageMeasurePassError(
                    f"{path}:{line_no} is not JSON ({exc}). A trace that "
                    f"cannot be parsed is not a short trace."
                ) from exc
            kind = row.get("kind")
            if kind == "header":
                interval = int(row.get("interval") or 1)
                rank = row.get("rank")
                continue
            if kind == "summary":
                has_summary = True
                continue
            if kind != "verdict":
                continue
            decision = row.get("ms_decision")
            if not decision:
                # A boot without --regime-stage-clock carries no split at all
                # (#363 window F4). Such an arm cannot be measured here, and
                # the refusal below says so by name rather than by an empty
                # series.
                continue
            if regime is not None and row.get("regime") != regime:
                continue
            round_index = row.get("round")
            if from_round is not None and (
                round_index is None or int(round_index) < int(from_round)
            ):
                continue
            if to_round is not None and (
                round_index is None or int(round_index) > int(to_round)
            ):
                continue
            total = decision.get("mean_total_ms")
            if total is None or float(total) <= 0.0:
                continue
            ms.append(float(total))
            waits.append(decision.get("mean_wait_share"))
            regimes.append(str(row.get("regime")))
    if require_summary and not has_summary:
        raise StageMeasurePassError(
            f"{path}: no summary line. The summary is written last by "
            f"construction, so its absence means the server was killed and "
            f"the trace supports 'so far', not a measurement."
        )
    if warmup:
        ms = ms[warmup:]
        waits = waits[warmup:]
        regimes = regimes[warmup:]
    if not ms:
        raise StageMeasurePassError(
            f"{path}: no boundary carries an ms/round split"
            + (f" for regime {regime!r}" if regime else "")
            + (
                f" in rounds [{from_round}, {to_round}]"
                if (from_round is not None or to_round is not None)
                else ""
            )
            + ". Boot with --regime-stage-clock (a boot without it writes "
            "ms_decision=None on every record), and check the regime label "
            "the controller actually assigned."
        )
    return ArmSeries(
        path,
        ms,
        interval=interval,
        rank=rank,
        has_summary=has_summary,
        wait_shares=waits,
        regimes=regimes,
    )


def merge_rank_arms(arms: Sequence[ArmSeries]) -> ArmSeries:
    """One arm from several ranks' traces of the SAME boot.

    The series is derived from a MIN-reduced GROUP sample, so every rank
    should carry the same numbers. Disagreement beyond
    ``_RANK_AGREEMENT_PCT`` is refused rather than averaged: it means the
    traces are not from one boot, and the average of two boots is a boot that
    never ran.
    """
    if not arms:
        raise StageMeasurePassError("no trace given for this arm")
    if len(arms) == 1:
        return arms[0]
    means = [a.mean_ms for a in arms]
    lo, hi = min(means), max(means)
    if lo <= 0.0:
        raise StageMeasurePassError("a rank reported a non-positive mean round")
    spread = 100.0 * (hi - lo) / hi
    if spread > _RANK_AGREEMENT_PCT:
        detail = ", ".join(f"{a.path}={m:.2f} ms" for a, m in zip(arms, means))
        raise StageMeasurePassError(
            f"the ranks of this arm disagree by {spread:.2f} % on the mean "
            f"round ({detail}), above the {_RANK_AGREEMENT_PCT:.1f} % the "
            f"group reduction permits. These traces are not one boot; merging "
            f"them would average two runs into one that never happened."
        )
    longest = max(arms, key=len)
    return longest


def gain_pct(reference: ArmSeries, stage: ArmSeries) -> float:
    """Percent of the reference round the stage gives back. Positive = faster."""
    ref = reference.mean_ms
    if ref <= 0.0:
        raise StageMeasurePassError("the reference arm has a non-positive mean round")
    return 100.0 * (ref - stage.mean_ms) / ref


def avs_a_floor_pct(floor_a: ArmSeries, floor_b: ArmSeries) -> float:
    """The same-boot A-vs-A floor, percent of the pair's mean.

    Two IDENTICAL configurations. Whatever difference they show is the floor a
    cross-arm delta has to beat; the arithmetic is deliberately the same shape
    as the gain above so the two numbers are comparable without conversion.
    """
    a, b = floor_a.mean_ms, floor_b.mean_ms
    base = (a + b) / 2.0
    if base <= 0.0:
        raise StageMeasurePassError("the A-vs-A pair has a non-positive mean round")
    return 100.0 * abs(a - b) / base


def drift_pct(arm: ArmSeries) -> float:
    """First half against second half of one run, percent of its own mean.

    #459's lesson in one number: a run that walks is not a run that is quiet,
    and a spread taken across a walking run reports the walk as noise. A
    single-sample arm has no halves and returns 0.0 -- it is refused earlier,
    by the boundary floor, and returning 0.0 here keeps that refusal in ONE
    place instead of two.
    """
    n = len(arm.round_ms)
    if n < 2:
        return 0.0
    half = n // 2
    first = arm.round_ms[:half]
    second = arm.round_ms[half:]
    mean_first = sum(first) / len(first)
    mean_second = sum(second) / len(second)
    base = arm.mean_ms
    if base <= 0.0:
        return 0.0
    return 100.0 * abs(mean_first - mean_second) / base


def band_from_floor_and_drift(floor: float, *drifts: float) -> float:
    """The band a gain must clear: the LARGEST of the floor and the drifts.

    Not a quadrature sum. Quadrature is right for combining two INDEPENDENT
    noise terms (`regime_ms_clock.combined_band_pct` does exactly that for two
    stages' bands); floor and drift are two ESTIMATES OF THE SAME QUANTITY --
    how much this rig's ms/round moves without anything being changed -- and
    the honest combination of two estimates of one quantity is the larger.
    """
    return max([float(floor)] + [abs(float(d)) for d in drifts])


def flip_seconds_from_log(text: str) -> List[float]:
    """Instrumented flip durations, seconds, from a boot log.

    Reads the ACTUATOR's own report and nothing else: the #297 reshard's
    ``total_ms`` (the mover times itself around the whole read/exchange/write
    /cutover walk) and the #631 seam census's ``elapsed_ms``. A duration
    computed by the reader from two log timestamps is not accepted -- it would
    include whatever else the scheduler did between the two lines.
    """
    out = [float(m.group(1)) / 1000.0 for m in _RESHARD_MS_RE.finditer(text)]
    out += [float(m.group(1)) / 1000.0 for m in _SEAM_MS_RE.finditer(text)]
    return out


def build_measurement(
    *,
    stage: str,
    regime: str,
    reference_arm: ArmSeries,
    stage_arm: ArmSeries,
    floor_a: ArmSeries,
    floor_b: ArmSeries,
    flip_samples_s: Sequence[float],
    reference_name: str,
    model_key: str,
    rig: str,
    source: str,
) -> StageMeasurement:
    """Assemble one record. Refuses; never fills a term it did not measure."""
    for label, arm in (
        ("reference", reference_arm),
        ("stage", stage_arm),
        ("floor-a", floor_a),
        ("floor-b", floor_b),
    ):
        if len(arm) < MIN_BOUNDARIES:
            raise StageMeasurePassError(
                f"the {label} arm ({arm.path}) carries {len(arm)} boundaries, "
                f"below the {MIN_BOUNDARIES} this pass requires. A mean over "
                f"fewer is decided by those samples."
            )
    if not flip_samples_s:
        raise StageMeasurePassError(
            "no instrumented flip cost. flip_cost_s is the term a controller "
            "must not assume: pass --flip-log naming a boot log that contains "
            "the actuator's own DONE line, or --flip-cost-ms with a measured "
            "value. It is not defaulted to zero, because an unpriced term "
            "that reads as free is how a flip gets taken for nothing."
        )
    flips = [float(x) for x in flip_samples_s if float(x) >= 0.0]
    if not flips:
        raise StageMeasurePassError("every flip sample was negative or unusable")
    floor = avs_a_floor_pct(floor_a, floor_b)
    band = band_from_floor_and_drift(
        floor, drift_pct(reference_arm), drift_pct(stage_arm)
    )
    return StageMeasurement(
        stage=stage,
        regime=regime,
        reference=reference_name,
        rig_key=rig,
        model_key=model_key,
        gain_pct=gain_pct(reference_arm, stage_arm),
        band_pct=band,
        # THE MAXIMUM, not the mean. The controller pays the cost of the flip
        # it is about to make, not the average of the flips somebody measured;
        # pricing by the mean under-charges exactly the expensive one.
        flip_cost_s=max(flips),
        avs_a_floor_pct=floor,
        drift_pct=max(drift_pct(reference_arm), drift_pct(stage_arm)),
        covered_s_reference=reference_arm.covered_s,
        covered_s_stage=stage_arm.covered_s,
        boundaries_reference=len(reference_arm),
        boundaries_stage=len(stage_arm),
        flip_samples=len(flips),
        flip_cost_mean_s=sum(flips) / len(flips),
        source=source,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m sglang.srt.planner.stage_measure_pass",
        description="Measure a stage's gain, band and flip cost from #363 traces.",
    )
    p.add_argument("--show", action="store_true", help="list the store and exit")
    p.add_argument("--path", default=None, help="store path (default: canonical)")
    p.add_argument("--stage", default=None)
    p.add_argument("--regime", default=None)
    p.add_argument("--reference", default="booted")
    p.add_argument("--model-key", default=None)
    p.add_argument(
        "--rig-uuid",
        action="append",
        default=[],
        help="card UUID (repeatable). Omit to resolve the rig from NVML.",
    )
    p.add_argument("--reference-trace", action="append", default=[])
    p.add_argument("--stage-trace", action="append", default=[])
    p.add_argument("--floor-a", default=None)
    p.add_argument("--floor-b", default=None)
    p.add_argument("--flip-log", action="append", default=[])
    p.add_argument("--flip-cost-ms", action="append", type=float, default=[])
    p.add_argument("--phase-regime", default=None, help="select boundaries by label")
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument(
        "--stage-from-round",
        type=int,
        default=None,
        help="first round of the STAGE arm's segment (use when the stage arm "
        "is the same boot after a /kv_reshard moved it onto the layout)",
    )
    p.add_argument("--stage-to-round", type=int, default=None)
    p.add_argument("--reference-from-round", type=int, default=None)
    p.add_argument("--reference-to-round", type=int, default=None)
    p.add_argument("--source", default="")
    p.add_argument("--allow-no-summary", action="store_true")
    p.add_argument("--write", action="store_true", help="persist the record")
    return p


def _show(path: Optional[str]) -> int:
    lib = StageMeasurementLibrary.load(path)
    target = stage_measure_path(path)
    print(f"stage measurement store: {target} ({len(lib)} record(s))")
    for rec in lib.records:
        print(f"  {rec.describe()}")
        for why in rec.refusals:
            print(f"      REFUSED: {why}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.show:
        return _show(args.path)
    missing = [
        name
        for name, value in (
            ("--stage", args.stage),
            ("--regime", args.regime),
            ("--model-key", args.model_key),
            ("--floor-a", args.floor_a),
            ("--floor-b", args.floor_b),
        )
        if not value
    ]
    if not args.reference_trace:
        missing.append("--reference-trace")
    if not args.stage_trace:
        missing.append("--stage-trace")
    if missing:
        print(f"refused: missing {', '.join(missing)}", file=sys.stderr)
        return 2

    def load(
        paths: Sequence[str],
        *,
        from_round: Optional[int] = None,
        to_round: Optional[int] = None,
    ) -> ArmSeries:
        return merge_rank_arms(
            [
                read_arm(
                    p,
                    regime=args.phase_regime,
                    warmup=args.warmup,
                    require_summary=not args.allow_no_summary,
                    from_round=from_round,
                    to_round=to_round,
                )
                for p in paths
            ]
        )

    try:
        reference_arm = load(
            args.reference_trace,
            from_round=args.reference_from_round,
            to_round=args.reference_to_round,
        )
        stage_arm = load(
            args.stage_trace,
            from_round=args.stage_from_round,
            to_round=args.stage_to_round,
        )
        floor_a = load([args.floor_a])
        floor_b = load([args.floor_b])
        flips = [float(x) / 1000.0 for x in args.flip_cost_ms]
        for log in args.flip_log:
            with open(log) as fh:
                flips += flip_seconds_from_log(fh.read())
        record = build_measurement(
            stage=args.stage,
            regime=args.regime,
            reference_arm=reference_arm,
            stage_arm=stage_arm,
            floor_a=floor_a,
            floor_b=floor_b,
            flip_samples_s=flips,
            reference_name=args.reference,
            model_key=args.model_key,
            rig=rig_key(args.rig_uuid or None),
            source=args.source,
        )
    except (StageMeasurePassError, StageMeasurementError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1

    print(record.describe())
    for why in record.refusals:
        print(f"  REFUSED: {why}")
    if not args.write:
        print("(not written -- pass --write)")
        return 0 if record.usable else 1
    lib = StageMeasurementLibrary.load(args.path)
    lib.add(record)
    target = lib.save(args.path)
    print(f"written to {target}")
    return 0 if record.usable else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
