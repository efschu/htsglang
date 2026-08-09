#!/usr/bin/env python3
"""#631 A-vs-A gate verdict: cross-commit delta, judged against the
same-boot noise floor.

THE VERDICT RULE, and it is deliberately conservative. For each rung the
floor is the WIDEST same-configuration spread either boot produced --
max over both boots of (|A1 vs A2| block delta, full min-max spread). A
cross-commit delta inside that floor is NOT a regression, because a rerun of
the same build produced at least that much. A delta outside it is a finding
and must be named as one, not filed as a footnote.

Direction matters: for tok/s, higher is better, so a NEGATIVE delta
(flip build slower than baseline) is the regression direction.
"""

from __future__ import annotations

import argparse
import json


# A pass whose own two blocks disagree by more than this is not a
# steady-state pass and may not carry a cross-commit number. MEASURED, not
# picked: on this rig a boot taken from cold decays monotonically through
# its first block as the 3080s reach their thermal ceiling (4106 -> 3809
# tok/s across six reps, A1 vs A2 -4.6 %), while the same boot measured
# after a soak repeats to within 0.06 %. The rule is what makes the drift
# fail loudly instead of masquerading as a code delta -- in either
# direction, since whichever build happens to be measured cold looks worse.
STEADY_STATE_TOLERANCE_PCT = 1.0


def floor_pct(a: dict, b: dict) -> float:
    cands = []
    for s in (a, b):
        cands.append(abs(s.get("AvsA_pct") or 0.0))
        cands.append(abs(s.get("spread_pct") or 0.0))
    return max(cands)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()

    base = json.load(open(args.baseline))
    cand = json.load(open(args.candidate))
    report: dict = {
        "baseline_label": base["label"],
        "candidate_label": cand["label"],
        "rungs": {},
        "verdict": "PASS",
    }

    for rung, key in (("prefill", "tok_s"), ("decode", "tok_s")):
        b = base["rungs"][rung][key]
        c = cand["rungs"][rung][key]
        delta_pct = 100.0 * (c["mean"] - b["mean"]) / b["mean"]
        fl = floor_pct(b, c)
        entry = {
            "metric": key,
            "baseline_mean": b["mean"],
            "candidate_mean": c["mean"],
            "delta_pct": delta_pct,
            "noise_floor_pct": fl,
            "baseline_AvsA_pct": b.get("AvsA_pct"),
            "candidate_AvsA_pct": c.get("AvsA_pct"),
            "baseline_spread_pct": b.get("spread_pct"),
            "candidate_spread_pct": c.get("spread_pct"),
            "inside_floor": abs(delta_pct) <= fl,
        }
        unsteady = [
            name
            for name, s in (("baseline", b), ("candidate", c))
            if abs(s.get("AvsA_pct") or 0.0) > STEADY_STATE_TOLERANCE_PCT
        ]
        if unsteady:
            entry["admissible"] = False
            entry["finding"] = (
                f"NOT STEADY STATE ({', '.join(unsteady)}): the pass's own two "
                f"blocks disagree by more than {STEADY_STATE_TOLERANCE_PCT}%. "
                "No cross-commit number may be quoted from it."
            )
            report["verdict"] = "INADMISSIBLE"
            report["rungs"][rung] = entry
            continue
        entry["admissible"] = True

        if delta_pct < 0 and not entry["inside_floor"]:
            entry["finding"] = "REGRESSION: outside the same-boot floor"
            report["verdict"] = "FAIL"
        elif not entry["inside_floor"]:
            entry["finding"] = "faster than baseline, outside the floor"
        report["rungs"][rung] = entry

    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
