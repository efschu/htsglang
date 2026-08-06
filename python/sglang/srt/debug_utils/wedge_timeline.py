"""Quickly locate the moment before a server wedge from a log file.

Usage as CLI:
    python -m sglang.srt.debug_utils.wedge_timeline <log_file> [--window 120]

Usage as library:
    from sglang.srt.debug_utils.wedge_timeline import (
        find_freeze_point, timeline, summarize,
    )
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timedelta
from typing import List, Optional

# Timestamp pattern: [YYYY-MM-DD HH:MM:SS ...]
_TS_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s.*?\]")

PROGRESS_MARKERS = [
    "POST /v1/chat/completions",
    "Decode batch",
    "Prefill batch",
]

TROUBLE_MARKERS = [
    "Health check failed",
    "Bar1CollectiveAborted",
    "index out of bounds",
]


def _parse_ts(line: str) -> Optional[datetime]:
    """Return the datetime from the line's leading timestamp, or None."""
    m = _TS_RE.match(line)
    if m is None:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _is_progress(line: str) -> bool:
    return any(marker in line for marker in PROGRESS_MARKERS)


def _is_trouble(line: str) -> bool:
    return any(marker in line for marker in TROUBLE_MARKERS)


def find_freeze_point(lines: List[str]) -> Optional[str]:
    """Return the timestamp of the last normal progress line before the
    first trouble line, or None if no trouble is found."""
    first_trouble_idx: Optional[int] = None
    last_progress_ts: Optional[str] = None

    for line in lines:
        if _is_trouble(line) and first_trouble_idx is None:
            first_trouble_idx = len(lines)  # sentinel to stop scanning
        if first_trouble_idx is not None:
            break

        if _is_progress(line):
            ts = _parse_ts(line)
            if ts is not None:
                last_progress_ts = ts.strftime("%Y-%m-%d %H:%M:%S")

    if first_trouble_idx is None:
        return None

    return last_progress_ts


def timeline(lines: List[str], window_s: int = 60) -> List[str]:
    """Return lines whose timestamp falls within window_s before the freeze
    point, plus all trouble lines that appear after the freeze point."""
    freeze_point = find_freeze_point(lines)
    if freeze_point is None:
        return []

    freeze_dt = datetime.strptime(freeze_point, "%Y-%m-%d %H:%M:%S")
    cutoff_dt = freeze_dt - timedelta(seconds=window_s)

    # Locate the first trouble line index to separate pre / post sections.
    first_trouble_idx: Optional[int] = None
    for i, line in enumerate(lines):
        if _is_trouble(line):
            first_trouble_idx = i
            break

    result: List[str] = []
    in_trouble = False

    for i, line in enumerate(lines):
        if first_trouble_idx is not None and i >= first_trouble_idx:
            in_trouble = True

        if in_trouble:
            # All trouble lines and everything after first trouble.
            result.append(line)
        else:
            ts = _parse_ts(line)
            if ts is not None and cutoff_dt <= ts <= freeze_dt:
                result.append(line)

    return result


def summarize(lines: List[str]) -> str:
    """Produce a human-readable diagnostic report.

    Sections:
      - Freeze timestamp & gap to first trouble line
      - Trouble marker counts
      - Last 20 progress lines before the freeze
    """
    freeze_point = find_freeze_point(lines)

    parts: list[str] = []
    parts.append("=" * 60)
    parts.append("  WEDGE TIMELINE REPORT")
    parts.append("=" * 60)

    if freeze_point is None:
        parts.append("No trouble lines found - no freeze detected.")
        parts.append("")
        return "\n".join(parts)

    parts.append(f"  Freeze point  : {freeze_point}")
    freeze_dt = datetime.strptime(freeze_point, "%Y-%m-%d %H:%M:%S")

    # --- Gap to first trouble ---
    first_trouble_ts: Optional[str] = None
    first_trouble_idx: Optional[int] = None
    for i, line in enumerate(lines):
        if _is_trouble(line):
            first_trouble_ts_raw = _parse_ts(line)
            if first_trouble_ts_raw is not None:
                first_trouble_ts = first_trouble_ts_raw.strftime("%Y-%m-%d %H:%M:%S")
            first_trouble_idx = i
            break

    if first_trouble_ts is not None and first_trouble_idx is not None:
        trouble_dt = datetime.strptime(first_trouble_ts, "%Y-%m-%d %H:%M:%S")
        gap = (trouble_dt - freeze_dt).total_seconds()
        parts.append(
            f"  First trouble : {first_trouble_ts}  (line {first_trouble_idx + 1})"
        )
        parts.append(f"  Gap           : {gap:.1f} s after freeze point")

    parts.append("")

    # --- Trouble marker counts ---
    trouble_counts: dict[str, int] = {}
    for line in lines:
        for marker in TROUBLE_MARKERS:
            if marker in line:
                trouble_counts[marker] = trouble_counts.get(marker, 0) + 1

    if trouble_counts:
        parts.append("  Trouble markers:")
        for marker, count in sorted(trouble_counts.items()):
            parts.append(f"    {marker:<40s} {count}")
        parts.append("")

    # --- Last 20 progress lines before freeze ---
    progress_lines: list[str] = []
    in_trouble_section = False
    for line in lines:
        if in_trouble_section:
            break
        if _is_trouble(line):
            in_trouble_section = True
            break
        if _is_progress(line):
            progress_lines.append(line)

    last_20 = progress_lines[-20:] if len(progress_lines) > 20 else progress_lines
    parts.append(f"  Last {len(last_20)} progress line(s) before freeze:")
    for pl in last_20:
        parts.append(f"  {pl}")

    parts.append("")
    parts.append("=" * 60)
    return "\n".join(parts)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Quickly locate the moment before a server wedge."
    )
    parser.add_argument("log_path", help="Path to the server log file")
    parser.add_argument(
        "--window",
        type=int,
        default=60,
        help="Seconds to look back before the freeze point (default: 60)",
    )
    args = parser.parse_args(argv)

    with open(args.log_path, "r", errors="replace") as f:
        lines = f.read().splitlines()

    print(summarize(lines))


if __name__ == "__main__":
    main()
