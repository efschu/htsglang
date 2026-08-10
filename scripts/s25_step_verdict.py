#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#631/#656 capacity ladder verdict, successor 25.

Reads ONE step's corridor csv plus the serving log and answers the only
three questions a capacity step can settle:

1. did the corridor hold (never below 1024 MiB free on any card), and how
   WELL FILLED was it (the law has two halves -- free near 1024, not
   multiple GiB above);
2. how large did the LIVE SET actually get -- a step whose occupancy
   stayed low has not tested the staging term and must not be read as
   evidence about the ceiling;
3. what the seam actually reserved (``staging reserved`` on the flip DONE
   lines) against what it was predicted to.

EVERY COUNT IS TAKEN AFTER THE LAST ``PHASE-FLIP armed at boot``. The log
holds several boots and mixing them silently blends configurations --
successor 24 recorded this trap and it is enforced here rather than
remembered.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys

LOG = "/spinning/serving-30030.boot.log"
FLOOR_MIB = 1024

DONE = re.compile(
    r"PHASE-FLIP DONE (\w+) \(epoch (\d+)\) in ([\d.]+) ms over (\d+) seam wave\(s\): "
    r"(\d+) live slots,.*?staging reserved ([\d.]+) MiB"
)
ARMED = re.compile(r"PHASE-FLIP armed at boot")
ABANDON = re.compile(r"FLIP ABANDONED")
GRAPH = re.compile(r"cuda graph: (True|False)")


def tail_since_last_boot(path: str, max_bytes: int = 400 * 1024 * 1024):
    """Lines after the LAST 'armed at boot', without holding the whole log."""
    size = os.path.getsize(path)
    start = max(0, size - max_bytes)
    with open(path, "r", errors="replace") as fh:
        fh.seek(start)
        lines = fh.readlines()
    last = 0
    for i, ln in enumerate(lines):
        if ARMED.search(ln):
            last = i
    return lines[last:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--log", default=LOG)
    args = ap.parse_args()

    pool_path = os.path.join(args.outdir, "pool")
    pool = int(open(pool_path).read().strip()) if os.path.exists(pool_path) else 0

    # -- corridor -----------------------------------------------------------
    csv_path = os.path.join(args.outdir, "corridor.csv")
    rows = list(csv.DictReader(open(csv_path)))
    cols = [c for c in rows[0] if c.endswith("_free")] if rows else []
    print(f"== capacity step verdict: pool={pool} outdir={args.outdir}")
    print(f"corridor samples: {len(rows)}")
    breaches = 0
    for c in cols:
        v = sorted(int(r[c]) for r in rows if r.get(c))
        if not v:
            continue
        n = len(v)
        b = sum(1 for x in v if x < FLOOR_MIB)
        breaches += b
        print(
            f"  {c}: min={v[0]} p1={v[n//100]} p50={v[n//2]} max={v[-1]} "
            f"margin_over_floor={v[0]-FLOOR_MIB} breaches={b}"
        )

    # -- the log, after the last boot ---------------------------------------
    lines = tail_since_last_boot(args.log)
    dones, abandons = [], 0
    live_max = 0
    graph_true = graph_false = 0
    for ln in lines:
        m = DONE.search(ln)
        if m:
            dones.append(
                dict(
                    direction=m.group(1),
                    waves=int(m.group(4)),
                    live=int(m.group(5)),
                    staging=float(m.group(6)),
                )
            )
            live_max = max(live_max, int(m.group(5)))
        if ABANDON.search(ln):
            abandons += 1
        g = GRAPH.search(ln)
        if g:
            if g.group(1) == "True":
                graph_true += 1
            else:
                graph_false += 1

    print(f"lines since last boot: {len(lines)}")
    print(f"flip DONE lines: {len(dones)}  FLIP ABANDONED lines: {abandons}")
    if dones:
        st = [d["staging"] for d in dones]
        print(
            f"staging reserved MiB: min={min(st):.1f} max={max(st):.1f} "
            f"mean={sum(st)/len(st):.1f}   seam waves={dones[-1]['waves']}"
        )
        print(f"max live slots seen: {live_max}")
        if pool:
            occ = live_max / pool
            print(f"peak occupancy: {occ:.1%} of the pool")
            if occ < 0.5:
                print(
                    "  WARNING: occupancy below 50%. The staging term scales "
                    "with the LIVE SET, so this step has NOT tested the term "
                    "that decides the ceiling. Do not read it as evidence "
                    "about capacity -- only as a pass for this load."
                )
    print(f"batches with a CUDA graph: {graph_true} / without: {graph_false}")

    # -- the anti-wedge extrapolation ---------------------------------------
    #
    # A step that holds the corridor at LOW occupancy says nothing about the
    # ceiling, because the ceiling is the ANTI-WEDGE condition: a flip must
    # stay affordable when the live set FILLS the pool. Rather than leave
    # that arithmetic to be re-derived (and mis-derived) every time, do it
    # here from THIS step's own numbers.
    if dones and rows and pool:
        gpu0 = sorted(int(r["gpu0_free"]) for r in rows if r.get("gpu0_free"))
        if gpu0:
            peak_stag = max(d["staging"] for d in dones)
            # Free on the binding card excluding staging, at THIS pool.
            base = gpu0[0] + peak_stag
            resident = 5 * 2048 / 1048576 * 1000  # MiB/1000 tok, rank1 = 5 layers
            slope, const = 4.517, 357             # corroborated staging model
            lo, hi = 50_000, 1_500_000
            for _ in range(80):
                mid = (lo + hi) / 2
                b = base + (pool - mid) * resident / 1000
                s = slope * mid / 1000 + const
                if b - s >= FLOOR_MIB:
                    lo = mid
                else:
                    hi = mid
            print(
                f"\nanti-wedge extrapolation from THIS step:\n"
                f"  binding-card baseline excluding staging = "
                f"{gpu0[0]} + {peak_stag:.0f} = {base:.0f} MiB at pool {pool}\n"
                f"  largest pool whose FULL-occupancy flip still clears "
                f"{FLOOR_MIB} MiB: {lo:,.0f}"
            )
            if pool > lo:
                print(
                    f"  WARNING: the running pool {pool:,} EXCEEDS that. It "
                    f"holds this load only because occupancy stayed low; a "
                    f"single long request can reach the unsafe region, and "
                    f"under strict purity an unaffordable flip WEDGES."
                )

    ok = breaches == 0 and abandons == 0
    print(f"VERDICT: {'PASS' if ok else 'FAIL'} "
          f"(breaches={breaches}, abandons={abandons})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
