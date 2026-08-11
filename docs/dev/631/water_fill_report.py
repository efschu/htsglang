#!/usr/bin/env python3
"""Item 16 over a measured free-column series: the levelling NOT performed.

Applies corridor_guard.water_fill_transfers to every sample of the 100 ms
spread series and reports, in one place, how much payload the water-fill
objective wants moved and which card it wants it moved onto. No actuator
exists for that move; this is what the missing tier would have been worth.
"""
import csv
import sys

sys.path.insert(0, "/spinning/wt-631-routea/python")
from sglang.srt.managers.corridor_guard import (  # noqa: E402
    DEFAULT_FLOOR_MIB,
    free_spread_mib,
    water_fill_transfers,
)

MIB = 1024 * 1024
rows = list(csv.DictReader(open(sys.argv[1])))
if not rows:
    sys.exit("empty series")

shed_mib, spreads = [], []
binding = None
for r in rows:
    col = [int(r[f"free{i}_mib"]) * MIB for i in range(3)]
    t = water_fill_transfers(col)
    shed = max(t) / MIB
    shed_mib.append(shed)
    spreads.append(free_spread_mib(col))
    mn = min(col) // MIB
    if binding is None or mn < binding[0]:
        binding = (mn, col, t, free_spread_mib(col))


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(len(v) * p / 100))]


n = len(rows)
print(f"samples                {n} at 100 ms  ({n / 10:.0f} s of wall clock)")
print(f"corridor law           {DEFAULT_FLOOR_MIB} MiB free per card")
print()
print("SPREAD OF THE FREE COLUMN (item 16's levelness metric)")
print(
    f"  min {min(spreads)}  p50 {pct(spreads, 50)}  p90 {pct(spreads, 90)}  "
    f"max {max(spreads)} MiB"
)
print()
print("WATER-FILL: PAYLOAD THE OBJECTIVE WANTS MOVED OFF THE FULLEST CARD")
print(
    f"  min {min(shed_mib):.0f}  p50 {pct(shed_mib, 50):.0f}  "
    f"p90 {pct(shed_mib, 90):.0f}  max {max(shed_mib):.0f} MiB"
)
print()
mn, col, t, spr = binding
print("AT THE BINDING INSTANT (the lowest free any card reached)")
print(f"  free column   {[c // MIB for c in col]} MiB")
print(f"  min free      {mn} MiB   -> margin {mn - DEFAULT_FLOOR_MIB} MiB over the law")
print(f"  spread        {spr} MiB")
print(f"  water-fill    {[round(x / MIB) for x in t]} MiB  (positive = SHED)")
print(
    f"  verdict       card {t.index(max(t))} should shed {max(t) / MIB:.0f} MiB "
    f"onto card {t.index(min(t))}"
)
