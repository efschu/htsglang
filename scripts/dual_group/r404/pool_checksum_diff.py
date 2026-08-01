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

    AND IT HAS A NOISE FLOOR, which round 1 of this window did not measure and
    was wrong about. The 2026-08-02 STEPS=3 boot returned "dirty" on 15 of 15
    arms, including ``spec_steps_0`` -- an arm whose speculative side takes
    zero draft steps -- at round 0, the PREFILL, on 100 % of positions and on
    all three surfaces at once. Two no-spec REFERENCE jobs on the same prompt
    read the same way against each other, with disjoint per-position digest
    sets and identical emitted tokens. A digest has no tolerance: on a stack
    whose forwards are not bitwise reproducible run to run, byte equality
    across two jobs is not a property of a correct run, so the byte reading has
    NO cross-job resolution there and every "dirty" it printed was its own
    floor.

    The reading that survives it is numeric and control-subtracted. The probe
    records ``kv_num`` / ``conv_num`` / ``ssm_num`` -- per logical position, a
    float32 sum and absmax of the same bytes -- and the floor is MEASURED from
    two or more reference draws (A-vs-A first, then A-vs-B). A position counts
    as dirty only when its deviation exceeds that measured floor by a factor,
    and when no floor was measured the reading says so instead of deciding.

Both readings report the FIRST offence and the full list, because "how many"
separates a one-off from a leak that grows with the residue.

Usage::

    pool_checksum_diff.py --spec spec.jsonl [--ref ref.jsonl]
                          [--control ref2.jsonl ...] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence

#: Surfaces that are comparable BETWEEN two jobs. ``map`` is absent on purpose.
CROSS_JOB_SURFACES = ("kv", "conv", "ssm")

#: Their numeric counterparts, and the ones the cross-job VERDICT rests on.
#: ``kv_num`` is a list of ``[sum, absmax]`` per logical position; the two state
#: fields are one such pair each.
NUMERIC_SURFACES = ("kv_num", "conv_num", "ssm_num")

#: How far above the MEASURED control floor a deviation has to sit to count.
#: Not a tolerance in its own right -- with a floor of zero this is zero, and
#: an exact stack keeps an exact reading.
DEFAULT_TOL_FACTOR = 4.0

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


def _pairs(value: Any) -> Optional[List[List[float]]]:
    """A numeric field as a list of ``[sum, absmax]`` pairs, or None.

    One record's ``kv_num`` is already such a list; ``conv_num`` / ``ssm_num``
    are a single pair, which is the same thing with one position in it.
    """
    if not value:
        return None
    if isinstance(value[0], (list, tuple)):
        return [[float(a) for a in pair] for pair in value]
    return [[float(a) for a in value]]


def _deviation(a: Sequence[float], b: Sequence[float]) -> float:
    """Relative distance between two ``[sum, absmax]`` fingerprints.

    Scaled by the larger of the magnitudes involved, so the number means the
    same thing at position 3 of a prompt and at position 700 of a generation.
    Both components are compared and the larger wins: a last-bit difference
    moves the sum and leaves the absmax alone, a leaked row moves both.
    """
    scale = max(abs(a[0]), abs(b[0]), abs(a[1]), abs(b[1]), 1e-9)
    return max(abs(x - y) for x, y in zip(a, b)) / scale


def numeric_differences(
    ref: Sequence[Dict[str, Any]],
    spec: Sequence[Dict[str, Any]],
    tol: Optional[float] = None,
) -> Dict[str, Any]:
    """The cross-job reading that has a tolerance.

    ``tol`` is the measured floor times the factor; ``None`` means "report the
    deviations, decide nothing" -- which is the honest state before a control
    has been run.
    """
    by_len: Dict[int, Dict[str, Any]] = {}
    for rec in ref:
        by_len.setdefault(int(rec["committed_len"]), rec)
    diffs: List[Dict[str, Any]] = []
    compared = 0
    worst = 0.0
    worst_at: Optional[Dict[str, Any]] = None
    for rec in spec:
        other = by_len.get(int(rec["committed_len"]))
        if other is None:
            continue
        joined = False
        for surface in NUMERIC_SURFACES:
            mine, theirs = _pairs(rec.get(surface)), _pairs(other.get(surface))
            if mine is None or theirs is None:
                continue
            joined = True
            devs = [_deviation(a, b) for a, b in zip(mine, theirs)]
            if not devs:
                continue
            local = max(devs)
            if local > worst:
                worst = local
                worst_at = {
                    "committed_len": int(rec["committed_len"]),
                    "surface": surface,
                    "spec_round": int(rec["round"]),
                    "deviation": local,
                }
            if tol is None:
                continue
            over = [i for i, d in enumerate(devs) if d > tol]
            if over:
                diffs.append(
                    {
                        "committed_len": int(rec["committed_len"]),
                        "surface": surface,
                        "spec_round": int(rec["round"]),
                        "ref_round": int(other["round"]),
                        "spec_path": rec.get("path"),
                        "rung": rec.get("rung"),
                        "deviation": local,
                        "positions": over[:16],
                        "n_positions": len(over),
                    }
                )
        compared += int(joined)
    return {
        "compared_positions": compared,
        "tolerance": tol,
        # None, not 0.0, when nothing numeric was joined: "the two runs agree
        # exactly" and "the two runs were never compared" are different answers
        # and a floor built from the second would be a fiction.
        "max_deviation": worst if compared else None,
        "max_deviation_at": worst_at,
        "differences": diffs,
    }


def noise_floor(draws: Sequence[Sequence[Dict[str, Any]]]) -> Dict[str, Any]:
    """What two CORRECT jobs disagree by -- measured, not assumed.

    Every draw after the first is read against the first exactly as a
    speculative arm would be. The result is the floor of both readings at once:
    how many byte differences a clean pair produces (if that is not zero, the
    byte reading has no cross-job resolution on this stack) and how far apart
    two clean numeric fingerprints get.
    """
    if len(draws) < 2:
        return {
            "draws": len(draws),
            "measured": False,
            "byte_differences": None,
            "numeric_max_deviation": None,
        }
    byte_diffs = 0
    compared = 0
    deviations: List[float] = []
    for other in draws[1:]:
        byte = cross_job_differences(draws[0], other)
        byte_diffs += len(byte["differences"])
        compared += byte["compared_positions"]
        dev = numeric_differences(draws[0], other)["max_deviation"]
        if dev is not None:
            deviations.append(dev)
    return {
        "draws": len(draws),
        "measured": True,
        "compared_positions": compared,
        "byte_differences": byte_diffs,
        "byte_reading_usable": byte_diffs == 0,
        "numeric_max_deviation": max(deviations) if deviations else None,
    }


def analyse(
    spec: Sequence[Dict[str, Any]],
    ref: Optional[Sequence[Dict[str, Any]]] = None,
    controls: Optional[Sequence[Sequence[Dict[str, Any]]]] = None,
    tol_factor: float = DEFAULT_TOL_FACTOR,
) -> Dict[str, Any]:
    """Every reading, the floor each one was judged against, and a verdict.

    ``controls`` are FURTHER no-spec draws of the same prompt. They are what
    turns "the two jobs differ" into "the two jobs differ by more than two
    reference jobs do", and without them the cross-job verdict is withheld
    rather than guessed: an unmeasured floor is reported as ``resolution:
    unmeasured``, not silently taken to be zero.
    """
    violations = append_only_violations(spec)
    report: Dict[str, Any] = {
        "spec_rounds": len(spec),
        "append_only_violations": violations,
        "first_append_only": violations[0] if violations else None,
        "surfaces_seen": sorted(
            {
                key
                for rec in spec
                for key in ("map", "kv", "conv", "ssm") + NUMERIC_SURFACES
                if rec.get(key) is not None
            }
        ),
        "region": sorted({str(r.get("region")) for r in spec}) if spec else [],
    }
    cross_bad = False
    if ref is not None:
        draws = [list(ref)] + [list(c) for c in (controls or [])]
        floor = noise_floor(draws)
        report["ref_rounds"] = len(ref)
        report["noise_floor"] = floor
        report["cross_job"] = cross_job_differences(ref, spec)
        first = (report["cross_job"]["differences"] or [None])[0]
        report["first_cross_job"] = first
        tol = None
        if floor["measured"] and floor["numeric_max_deviation"] is not None:
            tol = floor["numeric_max_deviation"] * tol_factor
        report["cross_job_numeric"] = numeric_differences(ref, spec, tol)
        num = report["cross_job_numeric"]
        report["first_cross_job_numeric"] = (num["differences"] or [None])[0]
        # WHICH reading is allowed to decide, and why. The byte reading is
        # trusted only where a control pair proved it can read a correct pair
        # as clean; the numeric one only where a floor was measured at all.
        if tol is not None and tol >= 1.0:
            # The floor swamps the signal: a relative deviation of 1.0 is a
            # different value, not a different last bit, so nothing short of a
            # sign flip could clear the bar. Two reference draws that far apart
            # are not a reference -- most often the prompt is bimodal -- and
            # the reading says so instead of returning a green it cannot back.
            report["resolution"] = "none"
        elif num["compared_positions"] and tol is not None:
            report["resolution"] = "numeric"
            cross_bad = bool(num["differences"])
        elif floor.get("byte_reading_usable"):
            report["resolution"] = "byte"
            cross_bad = bool(report["cross_job"]["differences"])
        elif not floor["measured"]:
            report["resolution"] = "unmeasured"
            cross_bad = bool(report["cross_job"]["differences"])
        else:
            report["resolution"] = "none"
    bad = bool(violations) or cross_bad
    report["clean"] = not bad
    parts = []
    if violations:
        first = violations[0]
        parts.append(
            f"append-only broken on {first['surface']} at round "
            f"{first['round']} (committed_len {first['committed_len']})"
        )
    if cross_bad:
        first_cross = report.get("first_cross_job_numeric") or report.get(
            "first_cross_job"
        )
        if first_cross:
            parts.append(
                f"{first_cross['surface']} differs from the reference at "
                f"committed_len {first_cross['committed_len']} "
                f"(spec round {first_cross['spec_round']})"
            )
    if report.get("resolution") == "unmeasured" and cross_bad:
        parts.append(
            "no control draw was given, so the byte reading decided against an "
            "UNMEASURED floor -- draw the reference twice before believing it"
        )
    if report.get("resolution") == "none":
        floor = report["noise_floor"]
        if floor["numeric_max_deviation"] is None:
            parts.append(
                "cross-job reading has NO resolution here: the control pair is "
                f"{floor['byte_differences']} byte differences apart and the "
                "records carry no numeric fingerprints to fall back on"
            )
        else:
            parts.append(
                "cross-job reading has NO resolution here: the control pair is "
                f"{floor['byte_differences']} byte differences and "
                f"{floor['numeric_max_deviation']} relative deviation apart"
            )
    report["verdict"] = "; ".join(parts) or "clean on every surface that was recorded"
    return report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="jsonl of the speculative job")
    ap.add_argument("--ref", help="jsonl of the no-spec reference job")
    ap.add_argument(
        "--control",
        action="append",
        default=[],
        help="a FURTHER no-spec draw of the same prompt; repeatable. Without "
        "at least one the cross-job floor is unmeasured and the verdict is "
        "withheld rather than guessed.",
    )
    ap.add_argument("--json", help="write the full report here")
    args = ap.parse_args(argv)

    spec = load_records(args.spec)
    if not spec:
        print("no records in --spec; was SGLANG_LANE_POOL_CHECKSUM=1 set?")
        return 2
    ref = load_records(args.ref) if args.ref else None
    controls = [load_records(path) for path in args.control]
    report = analyse(spec, ref, controls or None)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(report, handle, indent=2)
    print(f"rounds spec={report['spec_rounds']} ref={report.get('ref_rounds')}")
    print(f"surfaces: {', '.join(report['surfaces_seen']) or 'none'}")
    print(f"region:   {', '.join(report['region']) or 'none'}")
    floor = report.get("noise_floor")
    if floor:
        print(
            f"floor:    draws={floor['draws']} byte_diffs="
            f"{floor['byte_differences']} numeric_max="
            f"{floor['numeric_max_deviation']} -> resolution "
            f"{report.get('resolution')}"
        )
    print(f"verdict:  {report['verdict']}")
    return 0 if report["clean"] else 1


if __name__ == "__main__":
    sys.exit(main())
