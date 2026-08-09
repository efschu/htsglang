#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#631: VRAM corridor sampler -- per-card FREE, 100 ms, time-series MINIMUM.

THE RULE THIS ENFORCES, verbatim from the operator and not negotiable:
EXACTLY ~1024 MiB free PER CARD, CONTINUOUSLY under load. Three parts of
that wording are load-bearing and each has been got wrong before:

  * FREE means the NVML FREE field. NEVER total - used: a carve-out of
    roughly 424-518 MiB is invisible to that subtraction, so total-used
    reads high and hides a breach.
  * CONTINUOUSLY means the TIME-SERIES MINIMUM under load, not a snapshot
    after boot. The binding moment is a transient -- a flip's staging
    buffer, a graph capture -- which a one-shot reading cannot see.
  * "as full as possible" means free should sit NEAR 1024, not far above
    it. Leaving 4 GiB free per card passes the floor and wastes the rig,
    so this reports the headroom above the floor as well as the breach.

Sampling is via NVML directly (pynvml), because shelling out to
nvidia-smi cannot hold a 100 ms cadence.

    # in the background, for the length of a load run
    python3 scripts/route_a_631_corridor.py --out /tmp/corridor.json &
    ...run the load...
    kill -INT %1        # prints the report

    # or bounded
    python3 scripts/route_a_631_corridor.py --seconds 300
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from typing import Dict, List

MIB = 1024 * 1024
FLOOR_MIB = 1024.0

_STOP = False


def _stop(signum, frame):  # noqa: ARG001
    global _STOP
    _STOP = True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval-ms", type=float, default=100.0)
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="0 = run until SIGINT/SIGTERM")
    ap.add_argument("--floor", type=float, default=FLOOR_MIB)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    try:
        import pynvml
    except ImportError:
        print("FAILED: pynvml not importable; cannot read the NVML FREE field")
        return 2

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    pynvml.nvmlInit()
    n = pynvml.nvmlDeviceGetCount()
    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]
    names = [
        pynvml.nvmlDeviceGetName(h).decode()
        if isinstance(pynvml.nvmlDeviceGetName(h), bytes)
        else pynvml.nvmlDeviceGetName(h)
        for h in handles
    ]
    totals = [
        pynvml.nvmlDeviceGetMemoryInfo(h).total / MIB for h in handles
    ]

    mins: List[float] = [float("inf")] * n
    maxs: List[float] = [0.0] * n
    sums: List[float] = [0.0] * n
    breaches: List[int] = [0] * n
    min_at: List[float] = [0.0] * n
    samples = 0
    t0 = time.time()
    interval = args.interval_ms / 1000.0

    while not _STOP:
        if args.seconds and (time.time() - t0) >= args.seconds:
            break
        now = time.time()
        for i, h in enumerate(handles):
            free = pynvml.nvmlDeviceGetMemoryInfo(h).free / MIB
            if free < mins[i]:
                mins[i] = free
                min_at[i] = now - t0
            maxs[i] = max(maxs[i], free)
            sums[i] += free
            if free < args.floor:
                breaches[i] += 1
        samples += 1
        # Sleep the remainder of the cadence, never a fixed sleep: the NVML
        # reads themselves cost time and would stretch the interval.
        slept = interval - (time.time() - now)
        if slept > 0:
            time.sleep(slept)

    elapsed = time.time() - t0
    print()
    print("=" * 74)
    print(f"#631 VRAM CORRIDOR -- {samples} samples over {elapsed:.1f}s "
          f"at {args.interval_ms:.0f} ms, floor {args.floor:.0f} MiB")
    print(f"{'gpu':<4}{'name':<26}{'total':>9}{'MIN free':>10}"
          f"{'mean':>9}{'breaches':>10}")
    print("-" * 74)
    rows: List[Dict] = []
    ok = True
    for i in range(n):
        mn = mins[i] if mins[i] != float("inf") else 0.0
        mean = sums[i] / samples if samples else 0.0
        if breaches[i]:
            ok = False
        print(f"{i:<4}{names[i][:25]:<26}{totals[i]:>9.0f}{mn:>10.1f}"
              f"{mean:>9.1f}{breaches[i]:>10}")
        rows.append({
            "gpu": i, "name": names[i], "total_mib": totals[i],
            "min_free_mib": mn, "mean_free_mib": mean,
            "max_free_mib": maxs[i], "breaches": breaches[i],
            "min_at_s": min_at[i],
        })
    print("-" * 74)
    worst = min((r["min_free_mib"] for r in rows), default=0.0)
    per_card = ", ".join(f"{r['min_free_mib']:.0f}" for r in rows)
    print(f"per-card MINIMUM free: {per_card} MiB   "
          f"(worst {worst:.0f}, floor {args.floor:.0f})")
    # Headroom: the rule wants free NEAR the floor, not far above it.
    over = [r["min_free_mib"] - args.floor for r in rows]
    print(f"headroom above the floor: "
          f"{', '.join(f'{o:+.0f}' for o in over)} MiB "
          f"(large positives mean the budget is leaving VRAM unused)")
    print(f"CORRIDOR HELD: {ok}")
    print("=" * 74)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(
                {"samples": samples, "elapsed_s": elapsed,
                 "floor_mib": args.floor, "held": ok, "cards": rows},
                fh, indent=2,
            )
        print(f"written: {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
