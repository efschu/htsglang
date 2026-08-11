#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fill HANDOFF_678's acceptance table FROM the extract, never by hand.

Every number in the handoff's axis table also appears in EXTRACT.txt, and a
handoff that disagrees with its own extract is the exact failure the
contradictions register exists to track. So the table's cells are tokens that
this script substitutes from the extract's text -- copying by eye is how a
retracted number survives into the next shift.

Usage: s34_fill_handoff.py <EXTRACT.txt> <HANDOFF.md>
"""

from __future__ import annotations

import re
import sys


def main() -> int:
    extract = open(sys.argv[1]).read()
    path = sys.argv[2]
    doc = open(path).read()

    def find(pattern, default="?"):
        m = re.search(pattern, extract)
        return m.group(1).strip() if m else default

    minima = find(r"per-card minima: ([0-9]+ / [0-9]+ / [0-9]+) MiB")
    parts = [int(x) for x in re.findall(r"\d+", minima)] if minima != "?" else []
    spread = str(max(parts) - min(parts)) if len(parts) == 3 else "?"

    subs = {
        "MIN0 / MIN1 / MIN2": minima,
        "SPREAD": spread,
        "FLIPS": "{} pp_to_tp + {} tp_to_pp".format(
            find(r"pp_to_tp: (\d+)"), find(r"tp_to_pp: (\d+)")
        ),
        "PREFILLB": find(r"prefill batches: (\d+)"),
        "DGRAPH": find(r"decode graph share: ([0-9.]+%)"),
        "ACCEPT": find(r"accept length: mean=([0-9.]+)"),
        "OCC": find(r"live slots: max=(\d+\s+= [0-9.]+% of pool)").replace("  ", " "),
        "SHRINKS": find(r"driver-measured\):\s+(\d+)"),
        "PGATE": find(r"prefill-gate arms \(spill BEFORE the chunk\):\s+(\d+)"),
        "PSHORT": find(r"prefill-gate short \(ladder exhausted\):\s+(\d+)"),
        "GATECLR": find(r"gate: (\d+) cleared"),
    }
    for token, value in subs.items():
        doc = doc.replace("`" + token + "`", "**" + value + "**")
    open(path, "w").write(doc)
    for k, v in subs.items():
        print(f"  {k:12s} -> {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
