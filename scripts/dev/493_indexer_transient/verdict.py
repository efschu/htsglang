#!/usr/bin/env python3
"""Read the two arms of the #493 A/B and say whether the attribution holds.

Consumes what the run already writes -- no new instrument:

* ``SGLANG_FORWARD_PEAK_PATH`` per-rank JSON (``model_executor/forward_peak.py``):
  ``peak_bytes_max`` is torch's own per-forward peak and
  ``nvml_free_bytes_min`` is the driver's, both per phase and token bucket.
* the corridor CSV written by ``sample_corridor.sh``.

Usage:
    python3 verdict.py --off  RUN/peak_off  --on RUN/peak_on \\
                       --off-corridor RUN/corridor_off.csv \\
                       --on-corridor  RUN/corridor_on.csv \\
                       --predicted-delta-mib 326

The gate has two halves and both must pass:
  1. ATTRIBUTION -- peak_bytes_max falls between the arms by roughly the
     predicted delta. If it does not, the breach is some other allocation and
     the #493 root cause is refuted.
  2. CORRIDOR -- the ON arm holds >= 400 MiB free on every card at peak.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Dict

MIB = 1024 * 1024


def _peaks(prefix: str) -> Dict[str, dict]:
    """{rank_tag: row with the largest peak} over every per-rank JSON."""
    out: Dict[str, dict] = {}
    for path in sorted(glob.glob(f"{prefix}*.json")):
        with open(path) as fh:
            payload = json.load(fh)
        rows = payload.get("rows") or []
        if not rows:
            continue
        top = max(rows, key=lambda r: r.get("peak_bytes_max", 0))
        out[str(payload.get("rank_tag", os.path.basename(path)))] = top
    return out


def _corridor_min(path: str) -> Dict[str, int]:
    """{gpu index: minimum free MiB} from the sampler CSV."""
    mins: Dict[str, int] = {}
    if not path or not os.path.exists(path):
        return mins
    with open(path) as fh:
        for row in csv.DictReader(fh):
            try:
                idx = str(row["index"]).strip()
                free = int(str(row["memory_free_mib"]).strip())
            except (KeyError, ValueError):
                continue
            mins[idx] = min(mins.get(idx, free), free)
    return mins


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--off", required=True, help="forward-peak path prefix, budget OFF")
    ap.add_argument("--on", required=True, help="forward-peak path prefix, budget ON")
    ap.add_argument("--off-corridor", default="")
    ap.add_argument("--on-corridor", default="")
    ap.add_argument("--predicted-delta-mib", type=float, required=True)
    ap.add_argument(
        "--tolerance",
        type=float,
        default=0.30,
        help="fractional tolerance on the predicted delta",
    )
    ap.add_argument("--corridor-floor-mib", type=int, default=400)
    args = ap.parse_args()

    off, on = _peaks(args.off), _peaks(args.on)
    if not off or not on:
        print(
            "NO DATA -- forward_peak wrote nothing. Was SGLANG_FORWARD_PEAK_PATH set "
            "in BOTH arms, and did the ranks exit cleanly (the dump is at exit)?"
        )
        return 2

    print(
        f"{'rank':>10} {'peak off MiB':>13} {'peak on MiB':>12} {'delta':>9} "
        f"{'nvml free min on':>17}"
    )
    deltas = []
    for tag in sorted(set(off) & set(on)):
        a = off[tag]["peak_bytes_max"] / MIB
        b = on[tag]["peak_bytes_max"] / MIB
        free_on = on[tag].get("nvml_free_bytes_min")
        free_txt = "-" if not free_on else f"{free_on / MIB:.0f}"
        deltas.append(a - b)
        print(f"{tag:>10} {a:>13.1f} {b:>12.1f} {a - b:>9.1f} {free_txt:>17}")

    if not deltas:
        print("NO OVERLAP -- the two arms name different ranks.")
        return 2

    worst = min(deltas)
    lo = args.predicted_delta_mib * (1 - args.tolerance)
    print()
    print(
        f"predicted delta {args.predicted_delta_mib:.1f} MiB, "
        f"smallest measured delta {worst:.1f} MiB"
    )
    attribution = worst >= lo
    print(
        f"1. ATTRIBUTION: {'PASS' if attribution else 'FAIL'} "
        f"(needs >= {lo:.1f} MiB on every rank)"
    )
    if not attribution:
        print(
            "   -> the corridor breach is NOT (only) the indexer transient. "
            "Do not report #493 as fixed; report this number."
        )

    corridor = True
    for label, path in (("off", args.off_corridor), ("on", args.on_corridor)):
        mins = _corridor_min(path)
        if not mins:
            continue
        worst_card = min(mins.items(), key=lambda kv: kv[1])
        ok = worst_card[1] >= args.corridor_floor_mib
        if label == "on":
            corridor = ok
        print(
            f"   corridor {label}: min free {worst_card[1]} MiB on gpu{worst_card[0]} "
            f"({'OK' if ok else 'BREACH'}); all cards: "
            + ", ".join(f"gpu{k}={v}" for k, v in sorted(mins.items()))
        )
    print(
        f"2. CORRIDOR (on arm): {'PASS' if corridor else 'FAIL'} "
        f"(floor {args.corridor_floor_mib} MiB)"
    )
    print()
    print(
        "REMINDER: corridor repairs apply to EVERY violating card, not only the "
        "one a briefing happened to name (runbook section 4.5.4)."
    )
    return 0 if (attribution and corridor) else 1


if __name__ == "__main__":
    raise SystemExit(main())
