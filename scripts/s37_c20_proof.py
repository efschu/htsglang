#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#656 register C20, the proof line: the IN-CUTOVER minimum, per card.

WHY THIS IS ITS OWN LINE AND NOT A FOOTNOTE. Every acceptance extract in this
chain reports a per-card minimum over the whole window, and for four shifts
that number was read as "the corridor held". It did hold -- but successor 36
showed 20/20 of the deepest samples sit INSIDE a flip cutover, and successor
37 measured that s34's binding margin at that instant was +19 MiB while
s36's identical trough was -23. A window minimum that is made in one place
should be reported for that place, or the next shift inherits the same luck
and calls it headroom.

Prints, per card:
  * the minimum over samples INSIDE a cutover (the trough the law is
    actually tested at),
  * the minimum over samples OUTSIDE one (the resting level, which no
    mechanism in this corpus has ever had trouble with),
  * both against the 1024 MiB law and against s34's own numbers.

Then the seam-entry gate's own counters, because a mechanism that cannot say
it did nothing is indistinguishable from one that was never wired
(HANDOFF_680 1b).

Usage: s37_c20_proof.py <corridor.csv> <serving.log>
"""

from __future__ import annotations

import csv
import datetime
import re
import subprocess
import sys

LAW = 1024
# s34's accept2, the standing green baseline this run is judged against.
S34_WINDOW_MIN = [1043, 1922, 1541]
# The cutover is a ~1 s plateau (HANDOFF_680 1d), so a sample is "in-cutover"
# if it lands in [T-0.2 s, T+1.5 s] of a logged cutover second.
IN_LO, IN_HI = -0.2, 1.5


def main() -> int:
    csv_path, log_path = sys.argv[1], sys.argv[2]
    rows = []
    with open(csv_path) as f:
        for d in csv.DictReader(f):
            try:
                rows.append(
                    (
                        int(d["ts_ms"]) / 1000.0,
                        int(d["gpu0_free"]),
                        int(d["gpu1_free"]),
                        int(d["gpu2_free"]),
                    )
                )
            except Exception:
                pass
    if not rows:
        print("   NO SAMPLES -- corridor.csv unreadable")
        return 1
    rows.sort()
    day = datetime.datetime.fromtimestamp(rows[0][0])

    out = subprocess.run(
        ["grep", "-oE", r"^\[[0-9-]+ [0-9:]+ PP0\].*cutover", log_path],
        capture_output=True,
        text=True,
    ).stdout
    secs = sorted(
        {
            m.group(1)
            for m in (
                re.match(r"^\[[0-9-]+ ([0-9:]+) PP0\]", ln) for ln in out.splitlines()
            )
            if m
        }
    )
    marks = []
    for hh in secs:
        t = datetime.datetime.strptime(hh, "%H:%M:%S").replace(
            year=day.year, month=day.month, day=day.day
        )
        marks.append(t.timestamp())
    marks.sort()

    def in_cutover(ts: float) -> bool:
        # marks is sorted; a linear scan over a bounded neighbourhood is
        # plenty for a 30k-sample window and keeps this dependency-free.
        lo, hi = 0, len(marks) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if marks[mid] < ts + IN_LO:
                lo = mid + 1
            else:
                hi = mid - 1
        for j in range(max(0, lo - 2), min(len(marks), lo + 2)):
            if IN_LO <= ts - marks[j] <= IN_HI:
                return True
        return False

    inside, outside = [], []
    for r in rows:
        (inside if in_cutover(r[0]) else outside).append(r)

    print(
        f"-- C20: THE IN-CUTOVER MINIMUM, PER CARD  "
        f"({len(inside)} in-cutover of {len(rows)} samples, "
        f"{len(marks)} cutover seconds)"
    )
    if not inside:
        print("   NO IN-CUTOVER SAMPLES -- either no flips ran or the log")
        print("   wording moved; this line is unproven, do not read it as a pass.")
        return 1
    verdict_ok = True
    for g in (0, 1, 2):
        mi = min(r[1 + g] for r in inside)
        mo = min(r[1 + g] for r in outside) if outside else -1
        s34 = S34_WINDOW_MIN[g]
        flag = "" if mi >= LAW else "   *** BREACH ***"
        if mi < LAW:
            verdict_ok = False
        print(
            f"   gpu{g}  IN-CUTOVER MIN {mi:5d} MiB  (margin over the law "
            f"{mi - LAW:+5d})   non-seam MIN {mo:5d}   "
            f"s34 window MIN {s34} ({s34 - LAW:+d}){flag}"
        )
    print(
        "   READ: the left column is where this corpus's depth is made. The\n"
        "   s34 column is a WINDOW minimum and s34's own deepest samples were\n"
        "   in-cutover too, so the comparison is like for like."
    )

    def count(pattern: str) -> int:
        # grep exits 1 with no output when nothing matches, which is a count
        # of zero rather than an error.
        res = subprocess.run(
            ["grep", "-oF", pattern, log_path], capture_output=True, text=True
        )
        return len(res.stdout.splitlines())

    delays = count("seam entry DELAYED")
    yields_ = count("seam entry margin YIELDED")
    refused = count("corridor gate refused the seam staging")
    asks = count("C20 entry margin")
    print()
    print("-- the seam-entry gate's own account (a gate that cannot say it did")
    print("   nothing is indistinguishable from one that was never wired)")
    print(f"   asks carrying the C20 margin:            {asks}")
    print(f"   seams DELAYED for the margin:            {delays}")
    print(f"   seams entered on the law (budget spent): {yields_}")
    print(f"   seams REFUSED (below the law):           {refused}")
    if asks == 0:
        print("   *** THE MARGIN NEVER REACHED THE GUARD -- the term is inert ***")
        verdict_ok = False
    if delays == 0 and yields_ == 0:
        print(
            "   NOTE: no seam was ever short of the margin. That is a PASS only\n"
            "   together with the in-cutover minima above -- it means the ask\n"
            "   was funded every time, not that the gate was asleep (the ask\n"
            "   count above proves it ran)."
        )
    print()
    print(f"   C20 VERDICT: {'HELD' if verdict_ok else 'FAILED'}")
    return 0 if verdict_ok else 1


if __name__ == "__main__":
    sys.exit(main())
