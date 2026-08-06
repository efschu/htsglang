"""Parse per-rank timing lines from server logs and produce a short imbalance report.

Target lines match the ``Prefill rank batch`` / ``Decode rank batch`` patterns that
carry ``gpu-ms`` / ``compute`` / ``wait`` / ``wait by family`` fields, e.g.:

    [2026-08-06 19:15:53 TP0] Prefill rank batch,
        #new-token: 53, #cached-token: 0, #chunks: 1,
        gpu-ms: 111.0 (compute 32.8, wait 78.1)
        (wait by family: tp.all_reduce 62.5/129x, dcp.all_gather 5.3/16x, ...)
"""

from __future__ import annotations

import re
import sys
from typing import Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Regex
# ---------------------------------------------------------------------------

_LINE_RE = re.compile(
    r"\[(?P<ts_and_rank>[^\]]+)\]\s+\w+\s+rank batch"
    r".*?gpu-ms:\s*(?P<gpu_ms>[\d.]+)"
    r"\s*\(compute\s+(?P<compute_ms>[\d.]+),\s*wait\s+(?P<wait_ms>[\d.]+)\)"
    r"(?:\s*\(wait by family:\s*(?P<families>.*?)\))?"
)

# Extract the trailing TP\d+ from the bracket content (e.g. "2026-08-06 19:15:53 TP0")
_RANK_RE = re.compile(r"\s+(?P<rank>TP\d+)\s*$")

_FAMILY_ENTRY_RE = re.compile(
    r"(?P<name>[A-Za-z0-9_.]+)\s+(?P<ms>[\d.]+)/(?P<count>\d+)x"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_rank_batch_line(line: str) -> Optional[dict]:
    """Return parsed timing dict or ``None`` on unparseable input. Never raises."""
    try:
        m = _LINE_RE.search(line)
        if m is None:
            return None

        ts_and_rank = m.group("ts_and_rank")
        rm = _RANK_RE.search(ts_and_rank)
        if rm is None:
            return None
        rank = rm.group("rank")
        ts = ts_and_rank[: rm.start("rank")].rstrip()

        families_raw = m.group("families")
        family_dict: Dict[str, Tuple[float, int]] = {}
        if families_raw:
            for fm in _FAMILY_ENTRY_RE.finditer(families_raw):
                family_dict[fm.group("name")] = (
                    float(fm.group("ms")),
                    int(fm.group("count")),
                )

        return {
            "ts": ts,
            "rank": rank,
            "gpu_ms": float(m.group("gpu_ms")),
            "compute_ms": float(m.group("compute_ms")),
            "wait_ms": float(m.group("wait_ms")),
            "wait_by_family": family_dict,
        }
    except Exception:
        return None


def summarize(lines: Iterable[str]) -> dict:
    """Aggregate parsed lines per rank.

    Returns a dict keyed by rank string, each value containing:
        count, mean_gpu_ms, max_gpu_ms, mean_compute_ms, mean_wait_ms,
        total_wait_by_family  (family -> (total_ms, total_count))
    """
    accum: Dict[str, dict] = {}

    for line in lines:
        parsed = parse_rank_batch_line(line)
        if parsed is None:
            continue

        rank = parsed["rank"]
        if rank not in accum:
            accum[rank] = {
                "count": 0,
                "gpu_ms_values": [],
                "compute_ms_sum": 0.0,
                "wait_ms_sum": 0.0,
                "total_wait_by_family": {},
            }

        a = accum[rank]
        a["count"] += 1
        a["gpu_ms_values"].append(parsed["gpu_ms"])
        a["compute_ms_sum"] += parsed["compute_ms"]
        a["wait_ms_sum"] += parsed["wait_ms"]

        for fname, (fms, fcount) in parsed.get("wait_by_family", {}).items():
            prev = a["total_wait_by_family"].get(fname, (0.0, 0))
            a["total_wait_by_family"][fname] = (prev[0] + fms, prev[1] + fcount)

    result: Dict[str, dict] = {}
    for rank, a in accum.items():
        n = a["count"]
        gpu_vals = a["gpu_ms_values"]
        result[rank] = {
            "count": n,
            "mean_gpu_ms": sum(gpu_vals) / n,
            "max_gpu_ms": max(gpu_vals),
            "mean_compute_ms": a["compute_ms_sum"] / n,
            "mean_wait_ms": a["wait_ms_sum"] / n,
            "total_wait_by_family": a["total_wait_by_family"],
        }
    return result


def report(summary: dict) -> str:
    """Return a human-readable table and conclusion."""
    if not summary:
        return "No timing data found in the log.\n"

    ranks = sorted(summary.keys())

    width = 60
    lines: List[str] = []
    lines.append("=" * width)
    lines.append("Per-Rank Phase Timing Summary")
    lines.append("=" * width)

    for rank in ranks:
        s = summary[rank]
        lines.append(f"\n{rank}:  {s['count']} samples")
        lines.append(
            f"  mean gpu-ms:   {s['mean_gpu_ms']:>10.1f}   "
            f"max gpu-ms:  {s['max_gpu_ms']:.1f}"
        )
        lines.append(
            f"  mean compute-ms: {s['mean_compute_ms']:>10.1f}   "
            f"mean wait-ms: {s['mean_wait_ms']:>10.1f}"
        )
        families = s["total_wait_by_family"]
        if families:
            lines.append("  Total wait by family (across all samples):")
            for fname, (tms, tcnt) in sorted(families.items(), key=lambda x: -x[1][0]):
                lines.append(f"    {fname:>20s}  {tms:>10.1f} ms  ({tcnt} calls)")

    lines.append("\n" + "-" * width)
    lines.append("CONCLUSION")
    lines.append("-" * width)

    slowest_rank = max(summary, key=lambda r: summary[r]["mean_wait_ms"])
    slowest_wait = summary[slowest_rank]["mean_wait_ms"]
    lines.append(
        f"Rank with highest mean wait: {slowest_rank}  "
        f"({slowest_wait:.1f} ms per sample)"
    )

    global_families: Dict[str, float] = {}
    for s in summary.values():
        for fname, (tms, _) in s["total_wait_by_family"].items():
            global_families[fname] = global_families.get(fname, 0.0) + tms

    if global_families:
        dominant = max(global_families, key=global_families.get)
        lines.append(
            f"Family with largest total wait: {dominant}  "
            f"({global_families[dominant]:.1f} ms across all ranks)"
        )

    lines.append(
        "\nThe slowest rank sets the pace at each barrier; "
        "reduce its wait time to improve overall throughput."
    )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        print("Usage: rank_phase_summary.py <log-file>", file=sys.stderr)
        sys.exit(1)

    with open(argv[0]) as f:
        text = f.read()

    print(report(summarize(text.splitlines())))


if __name__ == "__main__":
    main()
