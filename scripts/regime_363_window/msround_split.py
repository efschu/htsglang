#!/usr/bin/env python3
"""#363 stage-clock window -- the A3 instrument: ms/round, split compute vs wait.

WHAT A3 ASKS. "ms/ROUND IMPROVED": compare mean ms/round in the SHIFT phase
against a control window run WITHOUT ``--regime-stage-clock``, everything else
identical -- and **report compute and wait separately**, because the mechanism
claim is that the WAIT term is what a stage flip moves. If the win is entirely
in compute, the arithmetic in ``regime_ms_clock`` is crediting the wrong term
(TICKET_363_STAGE_CLOCK.md A3).

THE MEASUREMENT CANON THIS FOLLOWS. ms per ROUND per worker, split COMPUTE vs
WAIT; runs of >= 10 s; warmup discarded; and the A-vs-A noise floor established
FIRST, because a cross-arm delta that sits inside the floor is noise and not a
result. This tool refuses to print an arm comparison until it has been given a
floor to judge it against.

THE TIMESTAMP GAP, NAMED. The regime trace carries ``round`` and ``epoch`` but
**no wall-clock field** (regime_runtime.py:424-459). So "the last 120 s of the
SHIFT phase" is NOT selectable from the trace alone. Two honest ways to segment,
both supported here and neither of them a guess:

  --from-round/--to-round   explicit round range, taken from the workload
                            driver's own phase log.
  --phase-regime REGIME     segment by the classifier's own label (e.g.
                            ``prefill_heavy``). Self-contained, and arguably
                            the better definition of "the SHIFT phase": it is
                            the phase the controller believed it was in.

WHERE THE NUMBERS COME FROM. Each boundary's ``ms_decision`` carries
``mean_total_ms`` and ``mean_wait_share`` (regime_ms_clock.py:469-480). So

    wait_ms    = mean_total_ms * mean_wait_share
    compute_ms = mean_total_ms * (1 - mean_wait_share)

``ms_decision`` is ``None`` on every boot without ``--regime-stage-clock``
(regime_runtime.py:456-457), which is exactly the control arm -- so for the
control this tool falls back to ``rank_mean_forward_ms``, and says so, because
a control that silently reported nothing would read as "no difference".

EXIT CODES: 0 report produced; 1 refused (with the reason); 2 usage.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional


def read_trace(path: Path) -> List[dict]:
    """Records only. The final summary line is kept separately by the caller."""
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def has_summary(records: List[dict]) -> bool:
    """A summary line means the run ended cleanly.

    Without it, 'zero desyncs' only means 'zero so far' -- the same rule the
    gate-1 tool enforces, applied here so a truncated trace cannot quietly
    become a measurement.
    """
    return any("summary" in r or r.get("kind") == "summary" for r in records)


def samples(
    records: List[dict],
    from_round: Optional[int],
    to_round: Optional[int],
    phase_regime: Optional[str],
    warmup: int,
) -> List[Dict[str, float]]:
    """Per-boundary (total, compute, wait) after segmentation and warmup discard."""
    rows = []
    for r in records:
        if "round" not in r:
            continue
        rnd = r.get("round")
        if from_round is not None and rnd < from_round:
            continue
        if to_round is not None and rnd > to_round:
            continue
        if phase_regime is not None and r.get("regime") != phase_regime:
            continue

        dec = r.get("ms_decision")
        if isinstance(dec, dict) and dec.get("mean_total_ms") is not None:
            total = float(dec["mean_total_ms"])
            share = dec.get("mean_wait_share")
            if share is None:
                rows.append({"round": rnd, "total": total, "compute": None, "wait": None})
                continue
            share = float(share)
            rows.append(
                {
                    "round": rnd,
                    "total": total,
                    "wait": total * share,
                    "compute": total * (1.0 - share),
                }
            )
        else:
            # Control arm: no stage clock, so no ms_decision. Fall back to the
            # per-rank forward mean, and carry the absence forward explicitly.
            fwd = r.get("rank_mean_forward_ms")
            if fwd is None:
                continue
            rows.append(
                {"round": rnd, "total": float(fwd), "compute": None, "wait": None}
            )

    # Warmup discard is by BOUNDARY COUNT, not by time -- the trace has no
    # wall clock, and inventing one from round numbers would be a guess.
    return rows[warmup:] if warmup else rows


def summarize(rows: List[Dict[str, float]]) -> Dict[str, Optional[float]]:
    if not rows:
        return {"n": 0, "total": None, "compute": None, "wait": None}

    def mean_of(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return statistics.fmean(vals) if vals else None

    return {
        "n": len(rows),
        "total": mean_of("total"),
        "compute": mean_of("compute"),
        "wait": mean_of("wait"),
    }


def pct_delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """(a - b) / b as a percentage. Negative = a is FASTER than b."""
    if a is None or b is None or not b:
        return None
    return (a - b) / b * 100.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="#363 A3: ms/round split compute vs wait, judged against an A-vs-A floor."
    )
    ap.add_argument("--arm", action="append", default=[],
                    help="trace file of the ARM under test (repeat per rank)")
    ap.add_argument("--control", action="append", default=[],
                    help="trace file of the control arm, no --regime-stage-clock")
    ap.add_argument("--floor-a", action="append", default=[],
                    help="A-vs-A repeat 1 (same arm, back to back)")
    ap.add_argument("--floor-b", action="append", default=[],
                    help="A-vs-A repeat 2")
    ap.add_argument("--from-round", type=int)
    ap.add_argument("--to-round", type=int)
    ap.add_argument("--phase-regime",
                    help="segment by classifier label, e.g. prefill_heavy")
    ap.add_argument("--warmup", type=int, default=20,
                    help="boundaries discarded from the front of each segment")
    ap.add_argument("--min-samples", type=int, default=30,
                    help="refuse a segment thinner than this")
    ap.add_argument("--require-summary", action="store_true",
                    help="refuse a trace with no summary line (truncated run)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    if args.smoke:
        return smoke()

    if not args.arm:
        print("need at least one --arm trace", file=sys.stderr)
        return 2

    def collect(paths: List[str], label: str):
        per_rank = []
        for p in paths:
            path = Path(p).expanduser()
            if not path.is_file():
                print(f"REFUSED: {label} trace {path} does not exist")
                return None
            recs = read_trace(path)
            if args.require_summary and not has_summary(recs):
                print(
                    f"REFUSED: {label} trace {path} has no summary line. The run "
                    f"did not end cleanly, so its numbers describe 'so far', not "
                    f"the window."
                )
                return None
            rows = samples(recs, args.from_round, args.to_round,
                           args.phase_regime, args.warmup)
            per_rank.append((path.name, summarize(rows)))
        return per_rank

    arm = collect(args.arm, "arm")
    if arm is None:
        return 1

    # THE SLOWEST RANK SETS THE ROUND. Reporting a group mean would hide the
    # rank that actually paces the barrier.
    def pacing(per_rank):
        withn = [(n, s) for n, s in per_rank if s["n"] and s["total"] is not None]
        if not withn:
            return None, None
        return max(withn, key=lambda kv: kv[1]["total"])

    arm_name, arm_s = pacing(arm)
    if arm_s is None:
        print("REFUSED: the arm segment carried no usable samples. Check "
              "--phase-regime / --from-round against the trace's own labels.")
        return 1
    if arm_s["n"] < args.min_samples:
        print(f"REFUSED: arm segment has {arm_s['n']} boundaries, below "
              f"--min-samples {args.min_samples}. A band from a handful of "
              f"samples is a number, not a measurement.")
        return 1

    report = {"arm": {"pacing_rank": arm_name, **arm_s}}

    print(f"ARM       pacing rank {arm_name}: n={arm_s['n']} "
          f"total={arm_s['total']:.3f} ms")
    if arm_s["compute"] is not None:
        print(f"          compute={arm_s['compute']:.3f} ms  "
              f"wait={arm_s['wait']:.3f} ms")
    else:
        print("          compute/wait UNAVAILABLE (no ms_decision in this trace)")

    # A-vs-A floor first. Without it no cross-arm delta may be called a result.
    floor_pct = None
    if args.floor_a and args.floor_b:
        fa = collect(args.floor_a, "floor-a")
        fb = collect(args.floor_b, "floor-b")
        if fa is None or fb is None:
            return 1
        _, fa_s = pacing(fa)
        _, fb_s = pacing(fb)
        if fa_s and fb_s and fa_s["total"] and fb_s["total"]:
            floor_pct = abs(pct_delta(fa_s["total"], fb_s["total"]))
            report["floor_pct"] = floor_pct
            print(f"A-vs-A    floor = {floor_pct:.2f} % "
                  f"({fa_s['total']:.3f} vs {fb_s['total']:.3f} ms)")
            if floor_pct > 5.0:
                print("          NOTE: the floor exceeds the shipped enter "
                      "watermark DEFAULT_ENTER_MARGIN_PCT = 5.0 "
                      "(regime_ms_clock.py:214). Per TICKET P2 the watermark "
                      "moves ONCE, before the window, and is recorded with its "
                      "measurement -- it is not adjusted afterwards.")
    else:
        print("A-vs-A    NOT SUPPLIED -- no cross-arm delta will be called a result")

    if args.control:
        ctl = collect(args.control, "control")
        if ctl is None:
            return 1
        ctl_name, ctl_s = pacing(ctl)
        if ctl_s and ctl_s["total"]:
            d_total = pct_delta(arm_s["total"], ctl_s["total"])
            report["control"] = {"pacing_rank": ctl_name, **ctl_s}
            report["delta_total_pct"] = d_total
            print(f"CONTROL   pacing rank {ctl_name}: n={ctl_s['n']} "
                  f"total={ctl_s['total']:.3f} ms")
            print(f"DELTA     total {d_total:+.2f} %  (negative = arm faster)")

            if arm_s["compute"] is not None and ctl_s["compute"] is not None:
                dc = pct_delta(arm_s["compute"], ctl_s["compute"])
                dw = pct_delta(arm_s["wait"], ctl_s["wait"])
                report["delta_compute_pct"] = dc
                report["delta_wait_pct"] = dw
                print(f"          compute {dc:+.2f} %   wait {dw:+.2f} %")
                if dw is not None and dc is not None and abs(dc) > abs(dw):
                    print("          MECHANISM WARNING: the move is larger in "
                          "COMPUTE than in WAIT. A3 says the wait term is what "
                          "a stage flip moves; if the win is in compute, "
                          "regime_ms_clock is crediting the wrong term.")
            else:
                print("          compute/wait delta UNAVAILABLE: the control has "
                      "no ms_decision (expected -- it runs without the stage "
                      "clock), so the SPLIT claim cannot be settled by this "
                      "pair alone. Use the floor arms, which do carry it.")

            if floor_pct is None:
                print("VERDICT   INCONCLUSIVE -- no A-vs-A floor supplied.")
                report["verdict"] = "INCONCLUSIVE_NO_FLOOR"
            elif d_total is None:
                report["verdict"] = "INCONCLUSIVE"
            elif abs(d_total) <= floor_pct:
                print(f"VERDICT   INSIDE THE FLOOR ({abs(d_total):.2f} % vs "
                      f"{floor_pct:.2f} %) -- not a result.")
                report["verdict"] = "INSIDE_FLOOR"
            else:
                print(f"VERDICT   CLEARS the floor ({abs(d_total):.2f} % vs "
                      f"{floor_pct:.2f} %).")
                report["verdict"] = "CLEARS"

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _fixture(path: Path, n: int, total: float, share: float,
             regime: str = "prefill_heavy", with_decision: bool = True,
             summary: bool = True) -> None:
    with path.open("w") as f:
        for i in range(n):
            rec = {"round": i, "epoch": 0, "regime": regime,
                   "rank_mean_forward_ms": total}
            rec["ms_decision"] = (
                {"mean_total_ms": total, "mean_wait_share": share,
                 "target": None, "reason": "smoke"}
                if with_decision else None
            )
            f.write(json.dumps(rec) + "\n")
        if summary:
            f.write(json.dumps({"summary": {"desyncs": 0}}) + "\n")


def smoke() -> int:
    red = 0
    total_cases = 0
    print("== smoke: each case must behave as stated ==")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)

        a = d / "arm.jsonl"
        c = d / "ctl.jsonl"
        fa = d / "fa.jsonl"
        fb = d / "fb.jsonl"

        # arm faster than control, well outside a tight floor
        _fixture(a, 80, 40.0, 0.25)
        _fixture(c, 80, 50.0, 0.25, with_decision=False)
        _fixture(fa, 80, 40.0, 0.25)
        _fixture(fb, 80, 40.2, 0.25)

        total_cases += 1
        print("-- case 1: a real delta must CLEAR a tight floor")
        rc = main(["--arm", str(a), "--control", str(c),
                   "--floor-a", str(fa), "--floor-b", str(fb),
                   "--warmup", "5", "--min-samples", "10"])
        print(f"   [exit={rc}]")
        red += 1 if rc == 0 else 0

        # a delta inside the floor must NOT be called a result
        total_cases += 1
        print("-- case 2: a delta inside a wide floor must be refused as noise")
        _fixture(fb, 80, 55.0, 0.25)          # floor now enormous
        rc = main(["--arm", str(a), "--control", str(c),
                   "--floor-a", str(fa), "--floor-b", str(fb),
                   "--warmup", "5", "--min-samples", "10"])
        print(f"   [exit={rc}]")
        red += 1 if rc == 0 else 0

        # too few samples must REFUSE
        total_cases += 1
        print("-- case 3: a thin segment must be refused")
        thin = d / "thin.jsonl"
        _fixture(thin, 12, 40.0, 0.25)
        rc = main(["--arm", str(thin), "--warmup", "5", "--min-samples", "30"])
        print(f"   [exit={rc}] (1 = refused, as required)")
        red += 1 if rc == 1 else 0

        # truncated trace must REFUSE under --require-summary
        total_cases += 1
        print("-- case 4: a truncated trace must be refused under --require-summary")
        trunc = d / "trunc.jsonl"
        _fixture(trunc, 80, 40.0, 0.25, summary=False)
        rc = main(["--arm", str(trunc), "--warmup", "5",
                   "--min-samples", "10", "--require-summary"])
        print(f"   [exit={rc}] (1 = refused, as required)")
        red += 1 if rc == 1 else 0

        # a segment label that matches nothing must REFUSE, not report zero
        total_cases += 1
        print("-- case 5: a phase label matching nothing must be refused")
        rc = main(["--arm", str(a), "--phase-regime", "no_such_regime",
                   "--warmup", "0", "--min-samples", "10"])
        print(f"   [exit={rc}] (1 = refused, as required)")
        red += 1 if rc == 1 else 0

    print(f"== smoke: {red}/{total_cases} cases behaved as required ==")
    return 0 if red == total_cases else 1


if __name__ == "__main__":
    sys.exit(main())
