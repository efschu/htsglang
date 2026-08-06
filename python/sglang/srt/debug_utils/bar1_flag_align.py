"""bar1_flag_align -- compare BAR1 abort-flag snapshots rank-by-rank, cell-by-cell.

Comparing per-rank maxima is INVALID.  Each 256-byte line in the BAR1 region
corresponds to one (topology, step, sender) triple.  With world-size R, line
index = block_index * R + sender_index.  A rank's OWN sender slot is normally 0
(because each rank's region holds flags written by its PEERS).  Only the SAME
(block, sender) cell across ranks is comparable.

This module parses the snapshot lines emitted by the server, aligns them by
(block, sender), and reports disagreements.
"""

from __future__ import annotations

import re
import sys
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_LINE_RE = re.compile(
    r"barlink-BAR1\s+abort\s+flag\s+snapshot\s+"
    r"rank\s+(\d+)/(\d+)\s+"
    r"group\s+(\S+):\s+"
    r"\d+\s+lines\s+of\s+\d+\s+bytes,\s+"
    r"first\s+dword\s+per\s+line\s+--\s+"
    r"(.+?)\.\s+Compare"
)

_PAIR_RE = re.compile(r"(\d+):(\d+)")


def parse_snapshot(line: str) -> Optional[Dict]:
    """Parse a single snapshot log line.

    Returns ``None`` on unparseable input (never raises).
    On success returns:
        {"rank": int, "world": int, "group": str,
         "values": {line_index: dword_value, ...}}
    """
    try:
        m = _LINE_RE.search(line)
        if m is None:
            return None
        rank = int(m.group(1))
        world = int(m.group(2))
        group = m.group(3)
        raw = m.group(4)
        values: Dict[int, int] = {}
        for pm in _PAIR_RE.finditer(raw):
            values[int(pm.group(1))] = int(pm.group(2))
        return {"rank": rank, "world": world, "group": group, "values": values}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


def align(snapshots: List[Dict], world: int) -> List[Dict]:
    """Align parsed snapshots by (block, sender).

    For every (block, sender) that appears in at least one snapshot, produce:
        {"block": int, "sender": int,
         "by_rank": {rank: value_or_None, ...}}
    Line index = block * world + sender.
    """
    # Collect all (block, sender) keys and all ranks
    keys: Dict[tuple, set] = {}
    ranks: set = set()
    for snap in snapshots:
        ranks.add(snap["rank"])
        for line_idx, val in snap["values"].items():
            block = line_idx // world
            sender = line_idx % world
            keys.setdefault((block, sender), set())
            keys[(block, sender)].add(snap["rank"])

    result: List[Dict] = []
    for block, sender in sorted(keys):
        line_idx = block * world + sender
        by_rank: Dict[int, Optional[int]] = {}
        for snap in snapshots:
            r = snap["rank"]
            by_rank[r] = snap["values"].get(line_idx)
        result.append({"block": block, "sender": sender, "by_rank": by_rank})
    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _header() -> str:
    return (
        "\n"
        "====================================================================\n"
        "  BAR1 Abort-Flag Alignment Report\n"
        "====================================================================\n"
        "  IMPORTANT: Comparing per-rank maxima is INVALID.\n"
        "  Each line in the snapshot corresponds to a specific\n"
        "  (block, sender) cell.  Only values in the SAME cell are\n"
        "  comparable across ranks.  A rank's own sender slot is\n"
        "  normally 0 because each rank's region holds flags written\n"
        "  by its PEERS.\n"
        "====================================================================\n"
    )


def report(snapshots: List[Dict], world: int) -> str:
    """Produce a human-readable text report.

    Sections:
      1. Per-(block, sender) table grouped by block.
      2. CONCLUSION listing every cell where ranks disagree, with spread.
    """
    aligned = align(snapshots, world)
    ranks = sorted({snap["rank"] for snap in snapshots})

    lines = [_header()]

    # --- Table grouped by block ---
    lines.append("")
    lines.append("Flag values by (block, sender):\n")

    cur_block = -1
    for entry in aligned:
        block = entry["block"]
        sender = entry["sender"]
        by_rank = entry["by_rank"]

        if block != cur_block:
            if cur_block >= 0:
                lines.append("")
            lines.append(f"  Block {block}:")
            cur_block = block

        cells = []
        for r in ranks:
            v = by_rank.get(r)
            cells.append(f"r{r}={v}")
        lines.append(f"    sender {sender}: {'  '.join(cells)}")

    # --- Conclusion ---
    lines.append("")
    lines.append("--------------------------------------------------------------------")
    lines.append("CONCLUSION")
    lines.append("--------------------------------------------------------------------")

    disagreements = []
    for entry in aligned:
        vals = [v for v in entry["by_rank"].values() if v is not None and v != 0]
        if len(vals) < 2:
            continue  # only one rank has a nonzero value -- not a disagreement
        unique = set(vals)
        if len(unique) > 1:
            spread = max(vals) - min(vals)
            disagreements.append((entry, spread, vals))

    if disagreements:
        lines.append("")
        lines.append("DISAGREEMENTS found (ranks hold different nonzero values")
        lines.append("in the SAME (block, sender) cell):\n")
        for entry, spread, vals in disagreements:
            block = entry["block"]
            sender = entry["sender"]
            detail = ", ".join(
                f"rank {r} = {v}"
                for r, v in sorted(entry["by_rank"].items())
                if v is not None and v != 0
            )
            lines.append(
                f"  (block={block}, sender={sender}): spread={spread} -- {detail}"
            )
    else:
        lines.append("")
        lines.append("No disagreements found.")
        lines.append("All ranks that hold nonzero values in the same")
        lines.append("(block, sender) cell agree on the value.")

    lines.append("")
    lines.append("--------------------------------------------------------------------")
    lines.append("Note on per-rank maxima: comparing each rank's maximum flag")
    lines.append("value is a MEANINGLESS operation.  The correct unit of")
    lines.append("comparison is the (block, sender) cell.")
    lines.append(
        "--------------------------------------------------------------------\n"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Read one or more log files from the command line and print the report."""
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        print("Usage: bar1_flag_align <log-file> [log-file ...]", file=sys.stderr)
        sys.exit(1)

    snapshots: list[dict] = []
    for path in argv:
        with open(path) as fh:
            for line in fh:
                parsed = parse_snapshot(line)
                if parsed is not None:
                    snapshots.append(parsed)

    if not snapshots:
        print("No snapshot lines found in the given files.", file=sys.stderr)
        sys.exit(1)

    world = snapshots[0]["world"]
    print(report(snapshots, world))


if __name__ == "__main__":
    main()
