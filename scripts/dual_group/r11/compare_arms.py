#!/usr/bin/env python3
"""Task #340 posten 2 readout: what is the deviating ingredient?

The established fact this arm set is built on (#336 ARM B): a TP=2 serving
group that was BOTH 3:1-uneven AND carrying a dual-group lane diverges from the
TP=1 reference at index 1 on all three prompts, while the TP=1 reference is
identical on the 5090 and on a 3080. That run cannot say which of its two
ingredients did it, because it changed both at once.

This readout takes the same reference (re-measured on THIS commit, not carried
over) and the same three prompts, and asks the two ingredients separately:

  even_tp2_nolane        TP=2, even 1:1 split, no lane
  uneven31_tp2_nolane    TP=2, 3:1 split, no lane
  even_tp2_stock         TP=2, even, no placement flags at all (control:
                         separates "TP=2" from "the fork's placement path")

Reading rules, in force before any verdict is spoken:
  * A prompt whose REFERENCE floor is red is VOID. The instrument saying it
    cannot see is not evidence about the arms.
  * An arm whose own floor is red is VOID for that prompt, not "deviating".
  * An arm that never ran is SKIPPED, which is not a result either.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

REF_LABEL = "tp1_ref_recheck"
ARM_LABELS = [
    "uneven31_tp2_lane",
    "uneven31_tp2_nolane",
    "even_tp2_nolane",
    "even_tp2_stock",
]
PROMPTS = ["alphabet", "squares", "code"]

# Numbers from the #336 GPU window (a DIFFERENT commit). They are printed as
# context and used for a drift check on the reference only -- never as a
# substitute for a reference measured here.
# armB/ref_tp1_5090.json and armB/ref_tp1_3080.json, which were identical.
PRIOR_REF_HEAD = {
    "alphabet": [86, 198, 87, 198],
    "squares": [717, 220, 8929, 198],
    "code": [286, 1853, 284, 659],
}
PRIOR_LANE_GROUP_INDEX1 = {"alphabet": 326, "squares": 62, "code": 311}
PRIOR_LANE_GROUP_HEAD = {"alphabet": [86, 326, 326, 320]}


def load(out_dir: str, label: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(out_dir, f"{label}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            report = json.load(f)
    except (OSError, ValueError):
        return None
    report["_rows"] = {row["prompt"]: row for row in report.get("prompts", [])}
    return report


def first_divergent(a: Optional[List[int]], b: Optional[List[int]]) -> Optional[int]:
    if not a or not b:
        return None
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def arm_state(report, ref_ids, prompt):
    """(state, head, first divergent index) for one arm on one prompt."""
    if report is None:
        return "SKIPPED", None, None
    row = report["_rows"].get(prompt)
    if row is None:
        return "SKIPPED", None, None
    if row.get("status") == "SKIPPED":
        return "SKIPPED", None, None
    ids = row.get("ids")
    if not row.get("floor_identical") or not ids:
        return "VOID", (ids or [])[:4] or None, None
    idx = first_divergent(ref_ids, ids)
    return ("MATCHES" if idx is None else "DEVIATES"), ids[:4], idx


def verdict_for(states: Dict[str, str]) -> str:
    even = states.get("even_tp2_nolane", "SKIPPED")
    uneven = states.get("uneven31_tp2_nolane", "SKIPPED")
    stock = states.get("even_tp2_stock", "SKIPPED")

    if even in ("VOID", "SKIPPED") or uneven in ("VOID", "SKIPPED"):
        return (
            f"VOID: even={even}, uneven31={uneven} -- both arms are needed to "
            "separate the ingredients"
        )
    if even == "DEVIATES":
        text = "even TP=2 also deviates -> generic TP=2 path"
        if stock == "DEVIATES":
            text += " (and with NO placement flags at all: plain TP=2)"
        elif stock == "MATCHES":
            text += " (but plain TP=2 matches -> the placement path, not TP=2 itself)"
        return text
    if uneven == "DEVIATES":
        return "only 3:1 deviates -> uneven-TP specific"
    # Both ingredient arms match. The remaining explanation is the lane, but it
    # is only an explanation if the lane arm actually reproduces the deviation
    # HERE -- otherwise the #336 result did not survive the commit and the
    # honest readout says so instead of naming a culprit.
    lane = states.get("uneven31_tp2_lane", "SKIPPED")
    if lane == "DEVIATES":
        return (
            "neither ingredient deviates without the lane, and the lane arm "
            "reproduces the deviation -> the lane's presence perturbs the "
            "serving group"
        )
    if lane == "MATCHES":
        return (
            "nothing deviates here, including the lane arm -> the #336 "
            "deviation is NOT reproducible on this commit"
        )
    return (
        f"neither ingredient deviates without the lane, lane arm={lane} -- "
        "the lane arm is needed to close this"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="/spinning/gpu-battery-results/2026-07-31_340_gpu/posten2",
        help="directory holding the per-arm JSON files",
    )
    ap.add_argument("--write", default="", help="also write the table to this file")
    args = ap.parse_args()

    lines: List[str] = []

    def emit(text: str = "") -> None:
        lines.append(text)
        print(text, flush=True)

    ref = load(args.out, REF_LABEL)
    arms = {label: load(args.out, label) for label in ARM_LABELS}

    emit("=" * 78)
    emit(
        "#340 posten 2 -- which ingredient deviates: uneven TP, TP itself, or the lane?"
    )
    emit("=" * 78)
    if ref is None:
        emit(f"NO REFERENCE: {REF_LABEL}.json missing in {args.out}.")
        emit("Every prompt is VOID -- the reference is measured, never assumed.")
        return 1

    # Drift check: does this commit's TP=1 reference still say what #336 said?
    emit("")
    emit("-- reference drift check against the #336 window (different commit) --")
    for prompt in PROMPTS:
        row = ref["_rows"].get(prompt, {})
        got = (row.get("ids") or [])[:4]
        want = PRIOR_REF_HEAD.get(prompt)
        mark = "same" if got == want else "DRIFTED"
        emit(
            f"  {prompt:10s} floor={bool(row.get('floor_identical')):<5} "
            f"head now={str(got):26s} #336={str(want):26s} [{mark}]"
        )
    emit(
        f"  #336 uneven31+LANE serving group alphabet "
        f"{PRIOR_LANE_GROUP_HEAD['alphabet']}, index1 {PRIOR_LANE_GROUP_INDEX1}"
    )

    emit("")
    emit("-- per-arm trajectories against this commit's TP=1 reference --")
    header = f"{'prompt':10s} {'arm':22s} {'head[0:4]':26s} {'firstdiv':9s} state"
    emit(header)
    emit("-" * len(header))

    verdicts: Dict[str, str] = {}
    for prompt in PROMPTS:
        ref_row = ref["_rows"].get(prompt, {})
        ref_ids = ref_row.get("ids")
        ref_ok = bool(ref_row.get("floor_identical")) and bool(ref_ids)
        emit(
            f"{prompt:10s} {'REFERENCE (tp1)':22s} {str((ref_ids or [])[:4]):26s} "
            f"{'-':9s} {'GREEN' if ref_ok else 'FLOOR RED'}"
        )
        if not ref_ok:
            verdicts[prompt] = "VOID: reference floor is red on this prompt"
            emit(f"{'':10s} -> {verdicts[prompt]}")
            emit("")
            continue

        states: Dict[str, str] = {}
        for label in ARM_LABELS:
            state, head, idx = arm_state(arms[label], ref_ids, prompt)
            states[label] = state
            emit(
                f"{'':10s} {label:22s} {str(head) if head else '-':26s} "
                f"{str(idx) if idx is not None else '-':9s} {state}"
            )
        verdicts[prompt] = verdict_for(states)
        emit(f"{'':10s} -> {verdicts[prompt]}")
        emit("")

    emit("-- summary --")
    for prompt in PROMPTS:
        emit(f"  {prompt:10s} {verdicts.get(prompt, 'VOID: not measured')}")

    decided = [v for v in verdicts.values() if not v.startswith("VOID")]
    emit("")
    if not decided:
        emit("OVERALL: VOID -- no prompt carries a verdict.")
    elif len(set(decided)) == 1:
        emit(f"OVERALL: {decided[0]} (unanimous on {len(decided)} prompt(s))")
    else:
        emit("OVERALL: prompts disagree -- see the per-prompt lines above.")

    if args.write:
        with open(args.write, "w") as f:
            f.write("\n".join(lines) + "\n")
    return 0 if decided else 1


if __name__ == "__main__":
    raise SystemExit(main())
