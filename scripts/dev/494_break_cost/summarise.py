#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Turn #494 break-cost JSONL into the F2 table, per rank.

Input: one or more ``break_cost.<rank_tag>.jsonl`` files written by
``srt/utils/break_cost_clock.py`` (``SGLANG_BREAK_COST_PROBE=1``).
Output: per rank and per break point, the mean cost of ONE crossing and the
per-step sum over all crossings of that step -- which is the left-hand side of
``TICKET_462_f2_and_replay.md`` §3's verdict.

Reports means over rounds, and the median as the outlier-insensitive twin. No
interpretation, no verdict: the ratio against the graph's saving is quoted by
the ticket, not by this script.

Usage:
  python3 scripts/dev/494_break_cost/summarise.py RUN_DIR/break_cost.*.jsonl
  python3 scripts/dev/494_break_cost/summarise.py --drop-rounds 20 FILE...
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from typing import Any, Dict, List


def load(paths: List[str], drop_rounds: int) -> Dict[str, List[Dict[str, Any]]]:
    by_rank: Dict[str, List[Dict[str, Any]]] = {}
    for path in paths:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("round", 0) < drop_rounds:
                    continue
                by_rank.setdefault(rec.get("rank_tag", "rank?"), []).append(rec)
    return by_rank


def _stat(values: List[float]) -> str:
    if not values:
        return "        -"
    return f"{statistics.fmean(values):9.3f}"


def _median(values: List[float]) -> str:
    if not values:
        return "        -"
    return f"{statistics.median(values):9.3f}"


def report(rank_tag: str, records: List[Dict[str, Any]]) -> None:
    print(f"\n=== {rank_tag}: {len(records)} rounds ===")
    if not records:
        return

    span = [r["span_ms"] for r in records]
    compute = [r["compute_ms"] for r in records]
    wait = [r["wait_ms"] for r in records]
    host = [r["host_ms"] for r in records]
    residual = [abs(r["residual_ms"]) for r in records]
    crossings = [r["crossings"] for r in records]

    print(f"  crossings/round     : {statistics.fmean(crossings):.1f}")
    print(f"  span_ms      mean {_stat(span)}  median {_median(span)}")
    print(f"  compute_ms   mean {_stat(compute)}  median {_median(compute)}")
    print(f"  wait_ms      mean {_stat(wait)}  median {_median(wait)}")
    print(f"  host_ms      mean {_stat(host)}  median {_median(host)}")
    print(f"  |residual|   max  {max(residual):9.3f}   (coherence check)")

    names = sorted({name for r in records for name in r.get("by_name", {})})
    for name in names:
        rows = [r["by_name"][name] for r in records if name in r.get("by_name", {})]
        counts = [row["count"] for row in rows]
        n = statistics.fmean(counts) if counts else 0.0
        print(f"\n  break point '{name}': {n:.1f} crossings/round")
        print("    term          per-step-sum      per-crossing")
        for term in ("gap_in_ms", "slot_ms", "gap_out_ms", "host_ms"):
            total = [row[term] for row in rows]
            per = [row[term] / row["count"] for row in rows if row["count"]]
            print(f"    {term:<12}{_stat(total)}         {_stat(per)}")
        phases = sorted({p for row in rows for p in row.get("phases", {})})
        for phase in phases:
            total = [row["phases"].get(phase, 0.0) for row in rows]
            per = [
                row["phases"].get(phase, 0.0) / row["count"] for row in rows if row["count"]
            ]
            print(f"    {('host:' + phase):<12}{_stat(total)}         {_stat(per)}")
        break_cost = [
            row["gap_in_ms"] + row["slot_ms"] + row["gap_out_ms"] for row in rows
        ]
        print(f"    BREAK COST/step (gap_in+slot+gap_out) mean {_stat(break_cost)}")


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+")
    ap.add_argument(
        "--drop-rounds",
        type=int,
        default=0,
        help="skip rounds with index < N (warmup / capture rounds)",
    )
    args = ap.parse_args(argv)

    by_rank = load(args.files, args.drop_rounds)
    if not by_rank:
        print("no records (wrong path, or the probe was never armed)")
        return 1
    for rank_tag in sorted(by_rank):
        report(rank_tag, by_rank[rank_tag])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
