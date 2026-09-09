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
    r"\[(?P<ts_and_rank>[^\]]+)\]\s+(?P<phase>\w+)\s+rank batch"
    # #1241: the decode half of the line names its rank as a FIELD. Optional,
    # so the prefill line (which has always taken its rank from the log
    # prefix) parses byte-identically.
    r"(?:,\s*rank:\s*(?P<rank_field>\d+))?"
    r"(?:.*?#round:\s*(?P<round>\d+))?"
    # #1241 the LADDER JOIN AXIS: UNIX epoch at which the round was opened.
    # Optional, so the prefill line (which has none) parses unchanged.
    r"(?:.*?\bt:\s*(?P<wall>\d+\.\d+))?"
    r"(?:.*?bs:\s*(?P<bs>\d+))?"
    # Rows SUBMITTED this round. Named `#rows` and not `#tokens` on purpose:
    # the ladder counts tokens ACCEPTED, and under MTP the two differ by the
    # acceptance rate. A reader who equates them mis-scales the join.
    r"(?:.*?#rows:\s*(?P<rows>\d+))?"
    r".*?gpu-ms:\s*(?P<gpu_ms>[\d.]+)"
    r"\s*\(compute\s+(?P<compute_ms>[\d.]+),\s*wait\s+(?P<wait_ms>[\d.]+)\)"
    r"(?:\s*\(wait by family:\s*(?P<families>.*?)\))?"
)

#: #1241. A round whose split was WITHHELD. Parsed on purpose rather than
#: dropped: a summary that silently skips graph-replayed rounds reports a
#: compute/wait mean over the eager minority and calls it the boot's.
_NO_SPLIT_RE = re.compile(
    r"\[(?P<ts_and_rank>[^\]]+)\]\s+(?P<phase>\w+)\s+rank batch"
    r"(?:,\s*rank:\s*(?P<rank_field>\d+))?"
    r"(?:.*?#round:\s*(?P<round>\d+))?"
    r"(?:.*?\bt:\s*(?P<wall>\d+\.\d+))?"
    r"(?:.*?bs:\s*(?P<bs>\d+))?"
    r".*?gpu-ms:\s*(?P<gpu_ms>[\d.]+)"
    r"\s*\(split unavailable:\s*(?P<reason>[^,]+),"
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
        rank_field = m.groupdict().get("rank_field")
        if rm is None and rank_field is None:
            return None
        if rm is not None:
            rank = rm.group("rank")
            ts = ts_and_rank[: rm.start("rank")].rstrip()
        else:
            # The explicit field WINS when both are present: the two groups of
            # a Weg-2 boot write different prefixes, and a cross-rank join must
            # not depend on the formatter.
            rank = "TP%s" % rank_field
            ts = ts_and_rank.rstrip()
        if rank_field is not None:
            rank = "TP%s" % rank_field

        families_raw = m.group("families")
        family_dict: Dict[str, Tuple[float, int]] = {}
        if families_raw:
            for fm in _FAMILY_ENTRY_RE.finditer(families_raw):
                family_dict[fm.group("name")] = (
                    float(fm.group("ms")),
                    int(fm.group("count")),
                )

        out = {
            "ts": ts,
            "rank": rank,
            "phase": m.group("phase"),
            "gpu_ms": float(m.group("gpu_ms")),
            "compute_ms": float(m.group("compute_ms")),
            "wait_ms": float(m.group("wait_ms")),
            "wait_by_family": family_dict,
            "split_known": True,
        }
        if m.group("round") is not None:
            out["round"] = int(m.group("round"))
        if m.group("bs") is not None:
            out["bs"] = int(m.group("bs"))
        if m.group("rows") is not None:
            out["rows"] = int(m.group("rows"))
        if m.group("wall") is not None:
            out["wall"] = float(m.group("wall"))
        return out
    except Exception:
        return None


def parse_unsplit_line(line: str) -> Optional[dict]:
    """#1241. A ``Decode rank batch ... (split unavailable: R, ...)`` line.

    Returns the rank, the honest ``gpu_ms`` and the REASON. Never invents a
    compute/wait pair for it: the absent split is reported as absent, which is
    the only reading that keeps a mean over the readable rounds from being
    quoted as a mean over the boot.
    """
    try:
        m = _NO_SPLIT_RE.search(line)
        if m is None:
            return None
        rank_field = m.group("rank_field")
        if rank_field is not None:
            rank = "TP%s" % rank_field
        else:
            rm = _RANK_RE.search(m.group("ts_and_rank"))
            if rm is None:
                return None
            rank = rm.group("rank")
        out = {
            "rank": rank,
            "phase": m.group("phase"),
            "gpu_ms": float(m.group("gpu_ms")),
            "split_known": False,
            "reason": m.group("reason").strip(),
        }
        if m.group("round") is not None:
            out["round"] = int(m.group("round"))
        if m.group("bs") is not None:
            out["bs"] = int(m.group("bs"))
        if m.group("wall") is not None:
            out["wall"] = float(m.group("wall"))
        return out
    except Exception:
        return None


def summarize(lines: Iterable[str]) -> dict:
    """Aggregate parsed lines per rank.

    Returns a dict keyed by rank string, each value containing:
        count, mean_gpu_ms, max_gpu_ms, mean_compute_ms, mean_wait_ms,
        total_wait_by_family  (family -> (total_ms, total_count)),
        withheld, withheld_gpu_ms_values, withheld_reasons

    THE DENOMINATOR IS PART OF THE ANSWER (#1241). A ``Decode rank batch``
    line whose split was WITHHELD (graph replay, or a slot contended with the
    prefill half) carries an honest ``gpu-ms`` and no compute/wait pair. The
    first version of this function dropped those lines silently, so the mean
    it printed was a mean over the readable MINORITY of a graph-covered boot
    while being labelled the boot's. They are counted here instead, by rank
    and by reason, and ``report`` prints the count next to the mean it does
    not contain.
    """
    accum: Dict[str, dict] = {}

    def _slot(rank: str) -> dict:
        if rank not in accum:
            accum[rank] = {
                "count": 0,
                "gpu_ms_values": [],
                "compute_ms_sum": 0.0,
                "wait_ms_sum": 0.0,
                "total_wait_by_family": {},
                "withheld": 0,
                "withheld_gpu_ms_values": [],
                "withheld_reasons": {},
            }
        return accum[rank]

    for line in lines:
        parsed = parse_rank_batch_line(line)
        if parsed is None:
            unsplit = parse_unsplit_line(line)
            if unsplit is None:
                continue
            a = _slot(unsplit["rank"])
            a["withheld"] += 1
            a["withheld_gpu_ms_values"].append(unsplit["gpu_ms"])
            reason = unsplit["reason"]
            a["withheld_reasons"][reason] = a["withheld_reasons"].get(reason, 0) + 1
            continue

        a = _slot(parsed["rank"])
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
        row = {
            "count": n,
            # A rank with ONLY withheld rounds has no split to average. None,
            # never 0.0: the whole point of counting the withheld rounds is
            # that an absent split must not be readable as a small one.
            "mean_gpu_ms": (sum(gpu_vals) / n) if n else None,
            "max_gpu_ms": max(gpu_vals) if n else None,
            "mean_compute_ms": (a["compute_ms_sum"] / n) if n else None,
            "mean_wait_ms": (a["wait_ms_sum"] / n) if n else None,
            "total_wait_by_family": a["total_wait_by_family"],
            "withheld": a["withheld"],
            "withheld_reasons": dict(a["withheld_reasons"]),
        }
        wv = a["withheld_gpu_ms_values"]
        row["withheld_mean_gpu_ms"] = (sum(wv) / len(wv)) if wv else None
        result[rank] = row
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
        withheld = int(s.get("withheld", 0) or 0)
        total = s["count"] + withheld
        lines.append(
            f"\n{rank}:  {s['count']} split samples of {total} rounds"
            f"  ({withheld} withheld)"
        )
        if s["count"]:
            lines.append(
                f"  mean gpu-ms:   {s['mean_gpu_ms']:>10.1f}   "
                f"max gpu-ms:  {s['max_gpu_ms']:.1f}"
            )
            lines.append(
                f"  mean compute-ms: {s['mean_compute_ms']:>10.1f}   "
                f"mean wait-ms: {s['mean_wait_ms']:>10.1f}"
            )
        else:
            lines.append(
                "  NO SPLIT ON THIS RANK: every round withheld its "
                "compute/wait pair. There is no mean to print."
            )
        if withheld:
            reasons = ", ".join(
                f"{r} x{c}"
                for r, c in sorted(
                    s.get("withheld_reasons", {}).items(), key=lambda kv: -kv[1]
                )
            )
            wm = s.get("withheld_mean_gpu_ms")
            wm_txt = f"{wm:.1f}" if wm is not None else "n/a"
            lines.append(
                f"  WITHHELD {withheld}/{total} rounds (mean gpu-ms {wm_txt}); "
                f"the means above are over the other {s['count']}. "
                f"Reasons: {reasons}"
            )
        families = s["total_wait_by_family"]
        if families:
            lines.append("  Total wait by family (across all samples):")
            for fname, (tms, tcnt) in sorted(families.items(), key=lambda x: -x[1][0]):
                lines.append(f"    {fname:>20s}  {tms:>10.1f} ms  ({tcnt} calls)")

    lines.append("\n" + "-" * width)
    lines.append("CONCLUSION")
    lines.append("-" * width)

    split_ranks = [r for r in summary if summary[r]["count"]]
    if split_ranks:
        slowest_rank = max(split_ranks, key=lambda r: summary[r]["mean_wait_ms"])
        slowest_wait = summary[slowest_rank]["mean_wait_ms"]
        lines.append(
            f"Rank with highest mean wait: {slowest_rank}  "
            f"({slowest_wait:.1f} ms per sample)"
        )
    else:
        lines.append(
            "No rank reported a compute/wait split -- every round withheld "
            "it. Re-run the window with the decode graph off "
            "(--disable-cuda-graph) before reading a pacemaker out of this."
        )
    total_withheld = sum(int(s.get("withheld", 0) or 0) for s in summary.values())
    total_split = sum(s["count"] for s in summary.values())
    if total_withheld > total_split:
        lines.append(
            f"WINDOW NOT EVIDENCE: {total_withheld} withheld rounds against "
            f"{total_split} split ones. The majority of this window carries no "
            f"split, so any compute/wait conclusion drawn from it describes "
            f"the minority that fell out of the graph, not the boot."
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
