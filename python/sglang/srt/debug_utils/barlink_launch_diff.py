"""Compare BAR1 collective-launch log files across ranks.

Usage:
    python -m sglang.srt.debug_utils.barlink_launch_diff rank0.log rank1.log rank2.log

Each log file contains one 1-Hz sampler record per line, e.g.:
    17:47:59 group=? rank=0/? last_op=all_gather last_nbytes=192512 ...
"""

from __future__ import annotations

import re
import sys
from typing import Optional


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_LINE_RE = re.compile(
    r"^(\d{2}:\d{2}:\d{2})\s+"
    r"group=\S+\s+"
    r"rank=(\d+)/\S+\s+"
    r"last_op=(\S+)\s+"
    r"last_nbytes=(\d+)\s+"
    r".*?"
    r"captured_launches=(\S+)\s+"
    r"last_op_captured=(\S+)"
)


def _parse_bool(val: str) -> bool:
    return val.lower() in ("true", "yes", "1")


def parse_line(line: str) -> Optional[dict]:
    """Parse a single log line into a dict, or return None on failure.

    Never raises.
    """
    line = line.strip()
    if not line:
        return None
    m = _LINE_RE.search(line)
    if m is None:
        return None
    return {
        "ts": m.group(1),
        "rank": int(m.group(2)),
        "last_op": m.group(3),
        "last_nbytes": int(m.group(4)),
        "captured_launches": _parse_bool(m.group(5)),
        "last_op_captured": _parse_bool(m.group(6)),
    }


def parse_file(path: str) -> list[dict]:
    """Parse an entire log file into a list of record dicts."""
    records: list[dict] = []
    with open(path, "r") as f:
        for raw in f:
            rec = parse_line(raw)
            if rec is not None:
                records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Diff logic
# ---------------------------------------------------------------------------


def diff_ranks(records_by_rank: dict[int, list[dict]]) -> str:
    """Compare the multiset of (last_op, last_nbytes) across ranks per timestamp.

    Only timestamps present in **all** ranks are compared.
    Returns the first disagreement (or a "no disagreement" message).
    """
    if not records_by_rank:
        return "No records provided."

    all_ranks = sorted(records_by_rank.keys())

    # Build per-rank timestamp -> list of (last_op, last_nbytes)
    ts_by_rank: dict[int, dict[str, list[tuple[str, int]]]] = {}
    for rank, recs in records_by_rank.items():
        ts_map: dict[str, list[tuple[str, int]]] = {}
        for r in recs:
            ts_map.setdefault(r["ts"], []).append((r["last_op"], r["last_nbytes"]))
        ts_by_rank[rank] = ts_map

    # Intersection of timestamps across all ranks (preserving file order of
    # the first rank)
    first_rank = all_ranks[0]
    seen: set[str] = set()
    ordered_ts: list[str] = []
    for ts in ts_by_rank[first_rank]:
        if ts not in seen:
            seen.add(ts)
            ordered_ts.append(ts)

    common_ts = [ts for ts in ordered_ts if all(ts in ts_by_rank[r] for r in all_ranks)]

    # Compare each common timestamp
    for ts in common_ts:
        rank_tuples = {r: tuple(sorted(ts_by_rank[r][ts])) for r in all_ranks}
        values = list(rank_tuples.values())
        if not all(v == values[0] for v in values):
            lines = [f"Disagreement at timestamp {ts}:"]
            for r in all_ranks:
                lines.append(f"  rank {r}: {ts_by_rank[r][ts]}")
            return "\n".join(lines)

    return (
        f"No disagreements found. Compared {len(common_ts)} common timestamps "
        f"across ranks {all_ranks}."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> None:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) < 2:
        print(
            "Usage: barlink_launch_diff <log0> <log1> [log2 ...]",
            file=sys.stderr,
        )
        sys.exit(1)

    # Parse each file and group by rank (inferred from content, not filename)
    records_by_rank: dict[int, list[dict]] = {}
    for path in args:
        recs = parse_file(path)
        for r in recs:
            records_by_rank.setdefault(r["rank"], []).append(r)

    print(diff_ranks(records_by_rank))


if __name__ == "__main__":
    main()
