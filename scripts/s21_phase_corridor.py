#!/usr/bin/env python3
"""#631/#656 successor 21: attribute the VRAM corridor series to PP and TP windows.

WHY THIS EXISTS. Every capacity verdict in this chain has been argued from a
single corridor MINIMUM, which is a worst-TIME quantity over BOTH phases at
once. That number cannot answer the only question that decides whether a
spill can buy anything: *which phase is binding on each card*. An asset that
is cold in the non-binding phase is worth exactly 0 MiB to a continuous floor,
and an asset that is cold in the BINDING phase is worth its full size. The two
cases are indistinguishable in the aggregate minimum, and the difference is
the whole feature.

Method: the serving log's `event loop re-dispatch after <dir>` lines give the
exact wall time at which each rank's active stack changed. Those instants cut
the NVML free-memory series into alternating PP and TP windows. Per card we
then report the minimum, the median and the sample count SEPARATELY per phase.

The seam itself is excluded by a settle margin: for `margin` seconds after each
transition the samples belong to neither phase, because the cutover's own
transient (KV backing release/restore, weights refill) is a third regime and
folding it into either phase misattributes it.

Card indices are NVML/nvidia-smi indices, which are NOT rank indices. The
mapping is printed from the log's own boot lines when available.

Usage:
  s21_phase_corridor.py --corridor <csv> --log <serving log> [--margin 1.5]
                        [--since <unix ts>] [--until <unix ts>]
"""

from __future__ import annotations

import argparse
import bisect
import os
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from typing import List, Optional, Sequence, Tuple

# `event loop re-dispatch after pp_to_tp (active stack now tp)`
RE_DISPATCH = re.compile(
    r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})[^\]]*\].*"
    r"event loop re-dispatch after (pp_to_tp|tp_to_pp)"
)


def _parse_log_ts(stamp: str) -> float:
    """The serving log stamps LOCAL time without a zone; the box runs UTC."""
    dt = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return dt.timestamp()


def read_transitions(path: str) -> List[Tuple[float, str]]:
    """Wall time -> phase entered, deduplicated across the three ranks.

    All three ranks log their own re-dispatch within the same second, so a
    naive read triples every transition. We keep the FIRST occurrence per
    (second, direction): the phase boundary is one event, not three.
    """
    seen = set()
    out: List[Tuple[float, str]] = []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            m = RE_DISPATCH.match(line)
            if not m:
                continue
            key = (m.group(1), m.group(2))
            if key in seen:
                continue
            seen.add(key)
            out.append((_parse_log_ts(m.group(1)), "tp" if m.group(2) == "pp_to_tp" else "pp"))
    out.sort()
    return out


def read_corridor(path: str) -> Tuple[List[float], List[List[int]]]:
    ts: List[float] = []
    cards: List[List[int]] = []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("ts"):
                continue
            parts = [p for p in line.split(",") if p != ""]
            if len(parts) < 2:
                continue
            try:
                t = float(parts[0])
                vals = [int(float(v)) for v in parts[1:]]
            except ValueError:
                continue
            ts.append(t)
            cards.append(vals)
    return ts, cards


def phase_at(transitions: Sequence[Tuple[float, str]], t: float, margin: float) -> Optional[str]:
    """The phase in force at t, or None inside a seam / before the first mark."""
    keys = [x[0] for x in transitions]
    i = bisect.bisect_right(keys, t) - 1
    if i < 0:
        return None
    if t - transitions[i][0] < margin:
        return None  # seam: the cutover's own transient, not a phase
    # also exclude the run-up to the NEXT transition, where the flip is arming
    if i + 1 < len(transitions) and transitions[i + 1][0] - t < margin:
        return None
    return transitions[i][1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corridor", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--margin", type=float, default=1.5)
    ap.add_argument("--since", type=float, default=0.0)
    ap.add_argument("--until", type=float, default=float("inf"))
    ap.add_argument("--floor", type=int, default=1024)
    args = ap.parse_args()

    for p in (args.corridor, args.log):
        if not os.path.exists(p):
            print(f"REFUSE: missing {p}", file=sys.stderr)
            return 2

    transitions = read_transitions(args.log)
    ts, cards = read_corridor(args.corridor)
    if not transitions:
        print("REFUSE: no phase transitions in the log; nothing to attribute to", file=sys.stderr)
        return 2
    if not ts:
        print("REFUSE: empty corridor series", file=sys.stderr)
        return 2

    ncards = min(len(c) for c in cards)
    buckets = {ph: [[] for _ in range(ncards)] for ph in ("pp", "tp")}
    seam = [[] for _ in range(ncards)]
    used = 0
    for t, vals in zip(ts, cards):
        if t < args.since or t > args.until:
            continue
        ph = phase_at(transitions, t, args.margin)
        used += 1
        for c in range(ncards):
            if ph is None:
                seam[c].append(vals[c])
            else:
                buckets[ph][c].append(vals[c])

    span = (max(ts) - min(ts)) if len(ts) > 1 else 0.0
    print(f"corridor samples {len(ts)} over {span/60.0:.1f} min, "
          f"{used} in window; {len(transitions)} phase transitions; "
          f"seam margin {args.margin}s")
    print(f"phase transitions span "
          f"{time.strftime('%H:%M:%SZ', time.gmtime(transitions[0][0]))} .. "
          f"{time.strftime('%H:%M:%SZ', time.gmtime(transitions[-1][0]))}")
    print()
    print("NVML FREE MiB, per card, split by the phase in force")
    print(f"{'card':>4} {'phase':>5} {'n':>7} {'min':>7} {'p05':>7} {'median':>7} {'max':>7}")
    holds = []
    for c in range(ncards):
        row = {}
        for ph in ("pp", "tp"):
            v = sorted(buckets[ph][c])
            if not v:
                print(f"{c:>4} {ph:>5} {0:>7}       -       -       -       -")
                continue
            p05 = v[max(0, int(0.05 * len(v)) - 1)]
            row[ph] = v[0]
            print(f"{c:>4} {ph:>5} {len(v):>7} {v[0]:>7} {p05:>7} "
                  f"{int(statistics.median(v)):>7} {v[-1]:>7}")
        v = sorted(seam[c])
        if v:
            seam_p05 = v[max(0, int(0.05 * len(v)) - 1)]
            print(f"{c:>4} {'seam':>5} {len(v):>7} {v[0]:>7} {seam_p05:>7} "
                  f"{int(statistics.median(v)):>7} {v[-1]:>7}")
        if "pp" in row and "tp" in row:
            holds.append((c, row["pp"], row["tp"]))
        print()

    print("BINDING PHASE per card -- the phase whose MINIMUM is lower.")
    print("An asset that is cold in the binding phase is worth its full size to")
    print("the corridor; one cold only in the other phase is worth 0 MiB.")
    print(f"{'card':>4} {'pp_min':>8} {'tp_min':>8} {'binding':>8} {'margin':>8} {'vs floor':>9}")
    for c, ppmin, tpmin in holds:
        binding = "pp" if ppmin <= tpmin else "tp"
        print(f"{c:>4} {ppmin:>8} {tpmin:>8} {binding:>8} "
              f"{abs(ppmin - tpmin):>8} {min(ppmin, tpmin) - args.floor:>9}")
    if not holds:
        print("  (no card had samples in both phases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
