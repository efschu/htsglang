"""#190: segment a gdnhooks record file into forwards and diff them.

The in-hook forward counter is unreliable when the first-registered module is
not called on every forward, so segment the append-ordered record stream
instead: a module name that repeats starts a new forward.

Usage: python hook_diff.py <hookfile> [n_last]
"""

import json
import sys
from collections import Counter

path = sys.argv[1]
NLAST = int(sys.argv[2]) if len(sys.argv) > 2 else 5

recs = [json.loads(l) for l in open(path)]
segments = []
cur, seen = [], set()
for r in recs:
    if r["m"] in seen:
        segments.append(cur)
        cur, seen = [], set()
    seen.add(r["m"])
    cur.append(r)
if cur:
    segments.append(cur)

sizes = Counter(len(s) for s in segments)
full = max(sizes)
sel = [s for s in segments if len(s) == full][-NLAST:]
print(f"{path}: {len(segments)} forwards, sizes={dict(sizes)}, comparing last {len(sel)} of size {full}")
if len(sel) < 2:
    sys.exit(0)

order = [r["m"] for r in sel[0]]
maps = [dict((r["m"], r["h"]) for r in s) for s in sel]

first = None
ndiff = 0
diverged = []
for m in order:
    hs = [mp.get(m, "MISSING") for mp in maps]
    if len(set(hs)) > 1:
        ndiff += 1
        diverged.append(m)
        if first is None:
            first = m
            print(f"\nFIRST DIVERGENT MODULE: {m}")
            for i, h in enumerate(hs):
                print(f"  run{i}: {h}")
print(f"\ndivergent: {ndiff} / {len(order)}")
if diverged:
    print("first 25 divergent modules:")
    for m in diverged[:25]:
        print("   ", m)
else:
    print("ALL MODULE OUTPUTS BIT-IDENTICAL ACROSS RUNS")
