#!/usr/bin/env python3
"""#404: read the lane's per-round pool checksums and LOCALIZE a difference.

The probe (``SGLANG_LANE_POOL_CHECKSUM=1``, ``dual_group_lane.py``) emits one
record per committed round with a digest of each surface a rejected speculative
candidate could leave residue in. This module turns those records into the two
statements the window needs, and it is deliberately importable as well as
runnable -- ``bracket_arms.py`` calls it per arm so the verdict is in the arm's
json rather than in a second pass someone has to remember to run.

READING 1 -- APPEND-ONLY, within one job, no reference needed.
    Round R's ``kv_stable`` / ``map_stable`` digest covers exactly the prefix
    that was already committed before round R. It must equal round R-1's full
    ``kv`` / ``map``. When it does not, something round R did -- the verify's
    candidate writes, the free of the rejected slots, the state commit --
    reached back into a position the lane had already committed. That is the
    leak #399 proved by elimination, and this names the ROUND and the SURFACE
    it happened on.

READING 2 -- CROSS-JOB, a speculative arm against its no-spec reference.
    ``kv`` / ``conv`` / ``ssm`` join on ``committed_len``, which the plain and
    the speculative path maintain identically (``_pool_committed_len``). The
    first committed position where the two disagree, and on which surface, is
    the localisation. ``map`` is NOT joined across jobs: it hashes physical
    slot ids, which two correct jobs draw differently.

Both readings report the FIRST offence and the full list, because "how many"
separates a one-off from a leak that grows with the residue.

Usage::

    pool_checksum_diff.py --spec spec.jsonl [--ref ref.jsonl] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence

#: Surfaces that are comparable BETWEEN two jobs. ``map`` is absent on purpose.
CROSS_JOB_SURFACES = ("kv", "conv", "ssm")

#: (stable digest, the previous round's full digest it must equal).
APPEND_ONLY_PAIRS = (("kv_stable", "kv"), ("map_stable", "map"))


def load_records(source: Any) -> List[Dict[str, Any]]:
    """Records from a jsonl path, a lane result row, or a list of records."""
    if isinstance(source, list):
        return [dict(r) for r in source]
    if isinstance(source, dict):
        return [dict(r) for r in (source.get("pool_checksums") or [])]
    out: List[Dict[str, Any]] = []
    with open(source) as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def append_only_violations(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rounds where an already-committed prefix moved."""
    out: List[Dict[str, Any]] = []
    by_round = {int(r["round"]): r for r in records}
    for rec in records:
        prev = by_round.get(int(rec["round"]) - 1)
        if prev is None:
            continue
        for stable_key, full_key in APPEND_ONLY_PAIRS:
            stable = rec.get(stable_key)
            if stable is None:
                continue
            # The comparison is only meaningful when the two records describe
            # the same prefix. A round that committed nothing (or a record from
            # the freed-tail can-fail arm) is skipped rather than counted as
            # clean -- silence and a pass are not the same answer.
            if int(rec.get("prev_committed_len") or 0) != int(
                prev.get("committed_len") or -1
            ):
                continue
            if stable != prev.get(full_key):
                out.append(
                    {
                        "round": int(rec["round"]),
                        "surface": full_key,
                        "committed_len": prev.get("committed_len"),
                        "path": rec.get("path"),
                        "n_accept": rec.get("n_accept"),
                        "rung": rec.get("rung"),
                        "expected": prev.get(full_key),
                        "got": stable,
                    }
                )
    return out


def cross_job_differences(
    ref: Sequence[Dict[str, Any]],
    spec: Sequence[Dict[str, Any]],
    surfaces: Iterable[str] = CROSS_JOB_SURFACES,
) -> Dict[str, Any]:
    """Every committed position where the two jobs' surfaces disagree."""
    by_len: Dict[int, Dict[str, Any]] = {}
    for rec in ref:
        by_len.setdefault(int(rec["committed_len"]), rec)
    diffs: List[Dict[str, Any]] = []
    compared = 0
    for rec in spec:
        other = by_len.get(int(rec["committed_len"]))
        if other is None:
            continue
        compared += 1
        for surface in surfaces:
            mine, theirs = rec.get(surface), other.get(surface)
            if mine is None or theirs is None:
                continue
            if mine != theirs:
                entry = {
                    "committed_len": int(rec["committed_len"]),
                    "surface": surface,
                    "spec_round": int(rec["round"]),
                    "ref_round": int(other["round"]),
                    "spec_path": rec.get("path"),
                    "rung": rec.get("rung"),
                }
                # Per-position digests narrow a KV difference from "the prefix"
                # to "this token", which is the difference between a surface
                # and an address. Present only under ..._PER_POS=1.
                if surface == "kv" and rec.get("kv_pos") and other.get("kv_pos"):
                    moved = [
                        i
                        for i, (a, b) in enumerate(zip(rec["kv_pos"], other["kv_pos"]))
                        if a != b
                    ]
                    entry["positions"] = moved[:16]
                    entry["n_positions"] = len(moved)
                diffs.append(entry)
    return {"compared_positions": compared, "differences": diffs}


def analyse(
    spec: Sequence[Dict[str, Any]], ref: Optional[Sequence[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Both readings, plus a one-line verdict."""
    violations = append_only_violations(spec)
    report: Dict[str, Any] = {
        "spec_rounds": len(spec),
        "append_only_violations": violations,
        "first_append_only": violations[0] if violations else None,
        "surfaces_seen": sorted(
            {
                key
                for rec in spec
                for key in ("map", "kv", "conv", "ssm")
                if rec.get(key) is not None
            }
        ),
        "region": sorted({str(r.get("region")) for r in spec}) if spec else [],
    }
    if ref is not None:
        report["ref_rounds"] = len(ref)
        report["cross_job"] = cross_job_differences(ref, spec)
        first = (report["cross_job"]["differences"] or [None])[0]
        report["first_cross_job"] = first
    bad = bool(violations) or bool(report.get("cross_job", {}).get("differences"))
    report["clean"] = not bad
    if bad:
        parts = []
        if violations:
            first = violations[0]
            parts.append(
                f"append-only broken on {first['surface']} at round "
                f"{first['round']} (committed_len {first['committed_len']})"
            )
        first_cross = report.get("first_cross_job")
        if first_cross:
            parts.append(
                f"{first_cross['surface']} differs from the reference at "
                f"committed_len {first_cross['committed_len']} "
                f"(spec round {first_cross['spec_round']})"
            )
        report["verdict"] = "; ".join(parts)
    else:
        report["verdict"] = "clean on every surface that was recorded"
    return report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="jsonl of the speculative job")
    ap.add_argument("--ref", help="jsonl of the no-spec reference job")
    ap.add_argument("--json", help="write the full report here")
    args = ap.parse_args(argv)

    spec = load_records(args.spec)
    if not spec:
        print("no records in --spec; was SGLANG_LANE_POOL_CHECKSUM=1 set?")
        return 2
    ref = load_records(args.ref) if args.ref else None
    report = analyse(spec, ref)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(report, handle, indent=2)
    print(f"rounds spec={report['spec_rounds']} ref={report.get('ref_rounds')}")
    print(f"surfaces: {', '.join(report['surfaces_seen']) or 'none'}")
    print(f"region:   {', '.join(report['region']) or 'none'}")
    print(f"verdict:  {report['verdict']}")
    return 0 if report["clean"] else 1


if __name__ == "__main__":
    sys.exit(main())
