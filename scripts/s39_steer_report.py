#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#657: judge allocation steering on PLACEMENT, and the column as a consequence.

WHY PLACEMENT IS THE HEADLINE AXIS. The lender this tier last tried was
falsified because its action was to free memory and its metric was free
memory. Steering frees nothing, so the free column is not evidence that it
acted -- it is only evidence of what followed. What proves the action is
WHERE the decisions pointed and whether every rank pointed there together.

Usage: s39_steer_report.py <window-dir> [baseline-window-dir]
"""

import csv
import re
import statistics as st
import sys
from pathlib import Path

_STEER = re.compile(r"CORRIDOR-STEER (.*)")
_TOWARD = re.compile(r"steering NEW KV allocations toward rank (\d+)")
_PROMOTED = re.compile(r": (\d+) free slots promoted")
_PERM = re.compile(r"permutation is \[([0-9, ]+)\]")
_PINNED = re.compile(r"ceiling pinned by ([a-z ]+) \(tree_max=(-?\d+).*?req_max=(-?\d+)")
_MAXLIVE = re.compile(r"max_live=(\d+)")
_USAGE = re.compile(r"full token usage: ([0-9.]+)")


def spread_stats(path: Path):
    rows = []
    with open(path) as f:
        r = csv.reader(f)
        next(r, None)
        for x in r:
            try:
                rows.append([int(v) for v in x[1:4]])
            except (ValueError, IndexError):
                continue
    if not rows:
        return None
    spread = sorted(max(v) - min(v) for v in rows)

    def p(q):
        return spread[int(q * (len(spread) - 1))]

    per_card_min = [min(r[i] for r in rows) for i in range(3)]
    per_card_p50 = [
        sorted(r[i] for r in rows)[len(rows) // 2] for i in range(3)
    ]
    # The tight window: what the column looks like when the binding card is
    # at its worst. A mechanism that only levels an idle rig is not levelling.
    g0 = sorted(r[0] for r in rows)
    thr = g0[int(0.1 * (len(g0) - 1))]
    tight = [r for r in rows if r[0] <= thr]
    tight_spread = sorted(max(r) - min(r) for r in tight) if tight else [0]
    return {
        "samples": len(rows),
        "spread_p10": p(0.10),
        "spread_p50": p(0.50),
        "spread_p90": p(0.90),
        "spread_max": spread[-1],
        "spread_mean": round(st.mean(spread)),
        "per_card_min": per_card_min,
        "per_card_p50": per_card_p50,
        "breaches": sum(1 for r in rows if min(r) < 1024),
        "tight_thr": thr,
        "tight_n": len(tight),
        "tight_spread_p50": tight_spread[len(tight_spread) // 2],
    }


def steer_stats(log: Path):
    out = {
        "armed": 0,
        "not_applicable": 0,
        "disarmed": [],
        "toward": {},
        "promoted": [],
        "perm": None,
        "lines": 0,
        "pinned_by_tree": 0,
        "pinned_by_reqs": 0,
        "pin_gap": [],
        "max_live": [],
        "usage_at_proposal": [],
    }
    usage = 0.0
    with open(log, errors="replace") as f:
        for line in f:
            m = _USAGE.search(line)
            if m:
                usage = float(m.group(1))
            if "KV-BACKING" in line or "proposal on device" in line:
                m = _MAXLIVE.search(line)
                if m:
                    out["max_live"].append(int(m.group(1)))
                    out["usage_at_proposal"].append(usage)
                m = _PINNED.search(line)
                if m:
                    who, tree_max, req_max = m.group(1), int(m.group(2)), int(m.group(3))
                    if "radix" in who:
                        out["pinned_by_tree"] += 1
                    else:
                        out["pinned_by_reqs"] += 1
                    out["pin_gap"].append(max(0, tree_max - req_max))
            if "CORRIDOR-STEER" not in line:
                continue
            out["lines"] += 1
            if "armed on rank" in line:
                out["armed"] += 1
            if "not applicable" in line:
                out["not_applicable"] += 1
            if "DISARMED" in line:
                out["disarmed"].append(line.strip()[-160:])
            m = _TOWARD.search(line)
            if m:
                r = int(m.group(1))
                out["toward"][r] = out["toward"].get(r, 0) + 1
            m = _PROMOTED.search(line)
            if m:
                out["promoted"].append(int(m.group(1)))
            m = _PERM.search(line)
            if m and out["perm"] is None:
                out["perm"] = [int(v) for v in m.group(1).split(",")]
    return out


def decisions_vs_column(win: Path, perm):
    """Did each steer name the card that actually had the most free memory?

    The decision is taken from a rank-local NVML read and then reduced with
    MIN, so it is not required to equal any single rank's view -- but if it
    routinely named a card that was NOT the fullest, the steer would be
    chasing noise, and that is worth knowing separately from whether it was
    UNIFORM. Joined on the log's own second-resolution timestamp against the
    corridor sampler, which is the only clock both instruments share.
    """
    if not perm:
        return None
    col = {}
    with open(win / "corridor.csv") as f:
        r = csv.reader(f)
        next(r, None)
        for x in r:
            try:
                col[int(x[0]) // 1000] = [int(v) for v in x[1:4]]
            except (ValueError, IndexError):
                continue
    if not col:
        return None
    import datetime as dt

    hits = miss = unknown = 0
    for line in open(win / "serving.log", errors="replace"):
        if "CORRIDOR-STEER steering NEW KV" not in line:
            continue
        m = _TOWARD.search(line)
        ts = re.match(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
        if not (m and ts):
            continue
        epoch = int(
            dt.datetime.strptime(ts.group(1), "%Y-%m-%d %H:%M:%S")
            .replace(tzinfo=dt.timezone.utc)
            .timestamp()
        )
        sample = None
        for d in range(0, 3):
            sample = col.get(epoch - d) or col.get(epoch + d)
            if sample:
                break
        if not sample:
            unknown += 1
            continue
        want_card = perm[int(m.group(1))]
        if want_card == max(range(3), key=lambda i: sample[i]):
            hits += 1
        else:
            miss += 1
    return hits, miss, unknown


def main():
    win = Path(sys.argv[1])
    base = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    print(f"=== #657 STEERING REPORT: {win}")

    s = steer_stats(win / "serving.log")
    print("\n-- PLACEMENT (the axis the mechanism is judged on)")
    print(f"   ranks armed                : {s['armed']} (expect 3)")
    print(f"   'not applicable'           : {s['not_applicable']} (expect 0)")
    print(f"   rank -> NVML permutation   : {s['perm']}")
    print(f"   decisions by absorbing rank: {dict(sorted(s['toward'].items()))}")
    if s["promoted"]:
        counts = {}
        for v in s["promoted"]:
            counts[v] = counts.get(v, 0) + 1
        # THE REPLICATION CHECK, IN THE OPEN. A steer decision is taken by
        # all three ranks at the same seam, and each logs how many slots of
        # the class its OWN free list held. If the list is replicated the
        # three numbers are the same, so every distinct value should appear
        # in a multiple of three. A value appearing once or twice means the
        # ranks were looking at different lists.
        triples = sum(1 for v, c in counts.items() if c % 3 == 0)
        print(f"   slots promoted, distinct   : {len(counts)} value(s), "
              f"min {min(counts)} max {max(counts)}")
        print(f"   values seen a multiple of 3 times: {triples}/{len(counts)} "
              f"({'ALL -- the three ranks agree slot-for-slot' if triples == len(counts) else 'MISMATCH: see below'})")
        if triples != len(counts):
            odd = {v: c for v, c in counts.items() if c % 3}
            print(f"     ! not a multiple of three: {odd}")
    dv = decisions_vs_column(win, s["perm"])
    if dv:
        hits, miss, unknown = dv
        tot = hits + miss
        print(f"   decisions naming the ACTUALLY fullest card: {hits}/{tot}"
              + (f" ({100*hits//tot}%)" if tot else "")
              + (f", {unknown} unjoinable" if unknown else ""))
    print(f"   DISARMS                    : {len(s['disarmed'])}")
    for d in s["disarmed"][:3]:
        print(f"     ! {d}")

    print("\n-- WHAT PINS THE KV RUNG'S CEILING (the instrument, #657)")
    tot = s["pinned_by_tree"] + s["pinned_by_reqs"]
    if tot:
        print(f"   proposals             : {tot}")
        print(f"   pinned by radix tree  : {s['pinned_by_tree']} "
              f"({100*s['pinned_by_tree']//max(1,tot)}%)")
        print(f"   pinned by resident req: {s['pinned_by_reqs']}")
        if s["pin_gap"]:
            g = sorted(s["pin_gap"])
            print(f"   evictable head-room   : p50 {g[len(g)//2]} rows, "
                  f"max {g[-1]} rows "
                  f"(= {g[len(g)//2]*8.5/1024:.0f} / {g[-1]*8.5/1024:.0f} MiB "
                  f"on a 3080, x1.76 on the 5090)")
    else:
        print("   no proposal carried the split clause")
    if s["max_live"]:
        # THE CEILING ITSELF, which is what the rung's floor is made of. A
        # mechanism that raises it makes the rung able to return LESS, so
        # this is the axis on which a steer can do harm, and comparing it
        # between windows only means something at matched load -- the
        # occupancy leg is identical in both, so the usage column is what
        # makes the rows comparable.
        ml = sorted(m for m in s["max_live"] if m > 1000)
        if ml:
            print(f"   max_live at proposals  : p50 {ml[len(ml)//2]}, "
                  f"max {ml[-1]} of a 512552-row pool "
                  f"({100*ml[-1]//512552}% of the id space)")
            pairs = [
                (u, m)
                for u, m in zip(s["usage_at_proposal"], s["max_live"])
                if m > 1000
            ]
            hi = [m for u, m in pairs if u >= 0.20]
            if hi:
                print(f"   ... of those, at usage >= 0.20: p50 "
                      f"{sorted(hi)[len(hi)//2]}, max {max(hi)}")

    print("\n-- THE FREE COLUMN (a consequence, not the proof)")
    cur = spread_stats(win / "corridor.csv")
    ref = spread_stats(base / "corridor.csv") if base else None
    if cur:
        rows = [("samples", "samples"), ("breaches", "breaches"),
                ("spread_p50", "spread p50 MiB"), ("spread_p90", "spread p90 MiB"),
                ("spread_mean", "spread mean MiB"), ("spread_max", "spread max MiB"),
                ("tight_spread_p50", "spread p50 when gpu0 is in its worst decile")]
        width = max(len(lbl) for _, lbl in rows)
        head = f"   {'':<{width}}   this"
        if ref:
            head += "      baseline"
        print(head)
        for key, lbl in rows:
            line = f"   {lbl:<{width}} : {cur[key]:>7}"
            if ref:
                line += f"   {ref[key]:>7}"
            print(line)
        print(f"   per-card min MiB       : {cur['per_card_min']}"
              + (f"   baseline {ref['per_card_min']}" if ref else ""))
        print(f"   per-card p50 MiB       : {cur['per_card_p50']}"
              + (f"   baseline {ref['per_card_p50']}" if ref else ""))


if __name__ == "__main__":
    main()
