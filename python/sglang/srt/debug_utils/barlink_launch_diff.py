"""Automated comparison of BAR1-collective launch logs across ranks.

When multiple GPU ranks hang, each rank writes a ~1 Hz record of its last
BAR1 collective to its own log file.  This module parses those logs and
compares them rank-by-rank per shared timestamp, reporting the first
timestamp where the ranks disagree.

Usage:
    python -m sglang.srt.debug_utils.barlink_launch_diff rank0.log rank1.log rank2.log
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_FIELD_RE = re.compile(r"(\w+)=([\S\(]+(?:\)[\S]*)?)")
_TS_RE = re.compile(r"^(\d{2}:\d{2}:\d{2})\s")
_RANK_RE = re.compile(r"rank=(\d+)/")


def _parse_bool(value: str) -> bool:
    return value == "True"


def parse_line(line: str) -> Optional[Dict]:
    """Parse a single log line into a dict, or return *None* on bad input."""
    line = line.strip()
    if not line:
        return None

    ts_match = _TS_RE.match(line)
    if ts_match is None:
        return None

    rank_match = _RANK_RE.search(line)
    if rank_match is None:
        return None

    fields = _FIELD_RE.findall(line)
    field_map: Dict[str, str] = {k: v for k, v in fields}

    try:
        captured_launches_str = field_map.get("captured_launches", "False")
        last_op_captured_str = field_map.get("last_op_captured", "False")

        return {
            "ts": ts_match.group(1),
            "rank": int(rank_match.group(1)),
            "last_op": field_map.get("last_op", ""),
            "last_nbytes": int(field_map.get("last_nbytes", "0")),
            "captured_launches": _parse_bool(captured_launches_str),
            "last_op_captured": _parse_bool(last_op_captured_str),
        }
    except (ValueError, TypeError):
        return None


def parse_file(path: str) -> List[Dict]:
    """Parse every line in *path*, returning only successfully parsed records."""
    records: List[Dict] = []
    with open(path, "r") as fh:
        for line in fh:
            rec = parse_line(line)
            if rec is not None:
                records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Diff logic
# ---------------------------------------------------------------------------

def diff_ranks(records_by_rank: Dict[int, List[Dict]]) -> str:
    """Compare (last_op, last_nbytes) multisets across all ranks per timestamp.

    Only timestamps present in **every** rank are compared.  On disagreement
    the report names the **first** offending timestamp and shows what each
    rank had.  On agreement it states how many timestamps were compared.
    """
    if not records_by_rank:
        return "No ranks provided."

    # Build per-rank per-timestamp multisets of (last_op, last_nbytes).
    # Value:  {rank: {ts: Counter[(last_op, last_nbytes)]}}
    rank_ts_counters: Dict[int, Dict[str, Counter]] = {}
    for rank, records in records_by_rank.items():
        counters: Dict[str, Counter] = {}
        for rec in records:
            ts = rec["ts"]
            counters.setdefault(ts, Counter())
            counters[ts][(rec["last_op"], rec["last_nbytes"])] += 1
        rank_ts_counters[rank] = counters

    # Find timestamps common to all ranks.
    ranks = list(records_by_rank.keys())
    if len(ranks) < 2:
        return "Need at least 2 ranks to compare."

    ts_sets = [set(rank_ts_counters[r]) for r in ranks]
    common_ts = sorted(set.intersection(*ts_sets))

    if not common_ts:
        return "No common timestamps found across all ranks."

    # Compare each common timestamp in chronological order.
    for ts in common_ts:
        multisets = [(r, rank_ts_counters[r][ts]) for r in sorted(ranks)]
        # Check equality of all multisets.
        first_counter = multisets[0][1]
        if any(c != first_counter for _, c in multisets[1:]):
            # Disagreement -- build report.
            lines = [
                f"DISAGREEMENT at timestamp {ts}",
                "",
            ]
            for r, cnt in multisets:
                parts = ", ".join(
                    f"({op!r}, {nbytes}) x{count}"
                    for (op, nbytes), count in sorted(cnt.items())
                )
                lines.append(f"  rank {r}: {parts}")
            return "\n".join(lines)

    return (
        f"No disagreement found across all ranks. "
        f"Compared {len(common_ts)} common timestamps."
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) < 2:
        print("Usage: barlink_launch_diff rank0.log rank1.log [rank2.log ...]")
        sys.exit(1)

    records_by_rank: Dict[int, List[Dict]] = {}
    for path in argv:
        for rec in parse_file(path):
            records_by_rank.setdefault(rec["rank"], []).append(rec)

    print(diff_ranks(records_by_rank))


if __name__ == "__main__":
    main()
