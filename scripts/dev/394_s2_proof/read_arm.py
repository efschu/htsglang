#!/usr/bin/env python3
"""Read one #394 slice-2 arm out of its #390 expert-stats dumps.

Prints the three things an arm has to be able to prove about itself, and
nothing else -- a server log is never read into anybody's context.

  1. WHICH ARM produced this file. ``host_shard.policy`` (equal vs
     link-proportional) and ``host_shard.reachability`` (local-only, refused,
     shared-cold-tier). A proportional arm whose reachability says anything
     other than ``shared-cold-tier`` silently ran the baseline, which is the
     way an A/B most often reports a null that was never tested.
     For a slice-3 arm the same question is ``moe_compute_policy``: an arm that
     reads ``base-plan`` served the baseline.
  2. PER-RANK H2D, the primary readout. Slice 2 moves byte ownership, not
     compute, so the prediction is a null delta here; see boot_ab.sh. The
     remote share says how much of it came out of a peer's segment. For the
     slice-3 arms the sign is inverted: a null delta FALSIFIES them, and the
     predicted per-rank figures are tabulated in ARM3_COMPUTE.md.
  3. The x4 rank's implied transfer time, which is the clock the whole feature
     is aimed at.
  4. The per-rank hit rate, which fills the row ANALYSE_389 left open and is
     the input the calibrated slice-3 sub-arm needs.
  5. The WORK POINT the dump was written at (tokens / forwards / activations),
     without which none of the above may be divided by another arm's.

WHICH REVISION TO READ (rule of 2026-08-03, replaces the earlier one).
Read the FINAL, WORK-MATCHED revision: run this tool AFTER teardown. The
previous rule -- "quote the pre-teardown numbers, never the post-SIGTERM
revision" -- is WITHDRAWN, because it compares two arms at different points of
their runs. Each rank writes its dump on its own 45 s timer, so a pre-teardown
read catches whatever fraction of the run the last tick landed on; in the
#439 green-corridor window that was 96.8 % for one arm against 91.9 % for the
other, and the treatment arm's accumulating H2D counter was read ~5 % early.
The resulting transfer-term ratio was inflated by about that much (1.5028x
pre-teardown against 1.4307x work-matched, prediction 1.427x). The final
revision is a common, well-defined endpoint and is the only valid basis for a
ratio. Evidence: 2026-08-03_439_green/RESULTS.md, "Which revision to read".

This tool therefore prints each rank's work counters and refuses to be quiet
when they disagree: within one final revision all ranks of an arm carry
identical tokens/forwards/activations, so a spread is how a non-final revision
announces itself.

THE COMPARISON IS THE TOOL'S JOB, NOT THE READER'S (#523). Until #523 the two
sentences above were the whole enforcement: the single-arm readout printed a
``work=`` line and left it to a human to hold the two lines side by side before
dividing one arm's counters by another's. That is exactly the check the #439
windows skipped twice, and it is why 1.5028x and 1.496x were published. There
is therefore no way to obtain a cross-arm number out of this tool except
through ``--against``, and that path REFUSES -- loudly, by name, with a
non-zero exit and no number printed at all -- when the two arms did not do the
same work. A silent comparison of unequal work is not reachable from here.

Usage:
  python3 read_arm.py <run_dir> <arm>
  python3 read_arm.py <run_dir> <arm_a> --against <arm_b>
        [--links GB/s,GB/s,...] [--work-tolerance-pct P]
"""

import argparse
import json
import pathlib
import sys

#: Work counters, per rank. A ratio between two arms is only defined at a
#: common work point -- see "WHICH REVISION TO READ" above.
WORK_KEYS = ("tokens", "forwards", "activations")

#: Arms whose treatment is the COMPUTE assignment (#394 slice 3). Their
#: prediction has the opposite sign to slice 2's, so they are named rather
#: than inferred: an arm nobody listed here is read as a slice-2 arm.
COMPUTE_ARMS = ("compute", "compute-cal")

#: How far two arms' work counters may differ and still be divided by each
#: other. NOT a taste value, and it BINDS at the geometry this tool is used at
#: (CLAUDE.md, "reach includes parameters"):
#:
#:   * the #439 green-corridor window's two FINAL revisions differ by 0.053 %
#:     (163486 vs 163572 tokens) and must pass -- that pair is the published
#:     1.4307x point;
#:   * the same window's two PRE-TEARDOWN revisions differ by ~5 % (96.8 % of
#:     one run against 91.9 % of the other) and must be refused -- that pair is
#:     the published-and-withdrawn 1.5028x.
#:
#: A work mismatch propagates into an accumulating-counter ratio roughly 1:1,
#: so the tolerance has to sit below the window's own A-vs-A floor to be worth
#: anything; the green window's floor was 0.424 % spread. 0.5 % is that floor,
#: rounded once: it admits the real work-matched pair with a 10x margin and
#: refuses the real mismatched one by a factor of 10. Both directions are pinned
#: in test_work_matched_counters_523.py.
DEFAULT_WORK_TOLERANCE_PCT = 0.5

#: Exit code of a refused comparison. Distinct from 1 (usage / no dumps) so a
#: driver script can tell "this window may not be divided" from "this tool was
#: called wrong".
EXIT_REFUSED = 3


class WorkMatchRefused(Exception):
    """A cross-arm number was requested for work that is not comparable.

    Carries a machine-readable ``reason`` so a caller (and a test) can assert
    WHICH rule refused, not merely that something did.
    """

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason
        self.message = message


def load(run_dir: pathlib.Path, arm: str):
    files = sorted(run_dir.glob(f"expert_stats_{arm}.tp*ep*.json"))
    if not files:
        raise SystemExit(f"no expert_stats_{arm}.tp*ep*.json in {run_dir}")
    return [(f.name, json.loads(f.read_text())) for f in files]


def work_counters(entries) -> dict:
    """``{key: {value, ...}}`` over the ranks of one arm.

    A set per key rather than a number: within one FINAL revision all ranks of
    an arm carry identical counters, so a set of size > 1 is how a non-final
    revision announces itself.
    """
    seen: dict = {key: set() for key in WORK_KEYS}
    for _name, payload in entries:
        totals = payload.get("totals", {})
        for key in WORK_KEYS:
            seen[key].add(totals.get(key, "?"))
    return seen


def format_work_point(seen: dict) -> str:
    return " ".join(
        f"{key}={sorted(seen[key])[0]}" if len(seen[key]) == 1 else f"{key}=*"
        for key in WORK_KEYS
    )


def final_work_point(arm: str, entries) -> dict:
    """``{key: int}`` for one arm, or refuse with the reason it is not final.

    Two distinct refusals, because they are two distinct defects:
    ``missing-counter`` is a dump revision that predates the work counters (a
    ratio taken from it is unfalsifiable, not merely wrong), ``non-final-
    revision`` is the 45 s-timer skew this whole rule exists for.
    """
    seen = work_counters(entries)
    missing = [key for key in WORK_KEYS if any(v == "?" for v in seen[key])]
    if missing:
        raise WorkMatchRefused(
            "missing-counter",
            f"arm {arm!r} carries no {'/'.join(missing)} counter in its #390 "
            "dump, so its work point is unknown and nothing may be divided by "
            "it. This is a dump written before the work counters existed; "
            "re-run the window, do not estimate the work point.",
        )
    spread = {key: sorted(seen[key]) for key in WORK_KEYS if len(seen[key]) > 1}
    if spread:
        detail = "; ".join(f"{key}={values}" for key, values in spread.items())
        raise WorkMatchRefused(
            "non-final-revision",
            f"the ranks of arm {arm!r} disagree on their work counters "
            f"({detail}), so this is NOT the final revision -- each rank was "
            "caught at its own point of the run by its own 45 s timer. Read "
            "the dumps AFTER teardown; see ARM3_COMPUTE.md, 'Which revision "
            "to read'.",
        )
    return {key: sorted(seen[key])[0] for key in WORK_KEYS}


def work_mismatch_pct(work_a: dict, work_b: dict) -> dict:
    """Per-counter relative difference in percent, symmetric in the two arms.

    Against the MEAN of the two, so the verdict does not depend on which arm
    was named first -- an asymmetric base would make ``a --against b`` and
    ``b --against a`` two different gates.
    """
    out = {}
    for key in WORK_KEYS:
        a = float(work_a[key])
        b = float(work_b[key])
        mean = (a + b) / 2.0
        out[key] = 0.0 if mean == 0 else abs(a - b) / mean * 100.0
    return out


def require_work_matched(
    arm_a: str,
    work_a: dict,
    arm_b: str,
    work_b: dict,
    tolerance_pct: float = DEFAULT_WORK_TOLERANCE_PCT,
) -> dict:
    """The rule of #482, as a gate: same work, or no number."""
    mismatch = work_mismatch_pct(work_a, work_b)
    over = {k: v for k, v in mismatch.items() if v > tolerance_pct}
    if over:
        detail = "; ".join(
            f"{key}: {arm_a}={work_a[key]} {arm_b}={work_b[key]} "
            f"({mismatch[key]:.3f} %)"
            for key in WORK_KEYS
            if key in over
        )
        raise WorkMatchRefused(
            "work-mismatch",
            f"arms {arm_a!r} and {arm_b!r} did NOT do the same work "
            f"({detail}), tolerance {tolerance_pct:.3f} %. Their accumulating "
            "counters (h2d_bytes above all) are sampled at different fractions "
            "of their runs, so a ratio between them measures the sampling "
            "difference as much as the treatment: in the #439 green-corridor "
            "window a ~5 % work gap inflated the transfer term from 1.4307x to "
            "1.5028x. Read both arms' FINAL dump revisions; see "
            "ARM3_COMPUTE.md, 'Which revision to read'.",
        )
    return mismatch


def h2d_per_rank(entries) -> list:
    return [
        int(payload.get("totals", {}).get("h2d_bytes", 0)) for _n, payload in entries
    ]


def transfer_seconds(h2d_bytes: list, links_gb_s: list) -> list:
    """``h2d_bytes / link`` per rank -- the transfer term, in seconds.

    Links are decimal GB/s (the unit the #394 card probe reports); the bytes
    are bytes. The clock is the slowest rank, never the group mean.
    """
    if len(h2d_bytes) != len(links_gb_s):
        raise WorkMatchRefused(
            "link-count-mismatch",
            f"{len(links_gb_s)} link speeds given for {len(h2d_bytes)} ranks. "
            "The transfer term is per rank and the clock is the slowest one, "
            "so a link vector that does not cover every rank cannot produce a "
            "clock.",
        )
    return [b / (float(link) * 1e9) for b, link in zip(h2d_bytes, links_gb_s)]


def clock_rank(seconds: list) -> tuple:
    """(rank, seconds) of the slowest rank -- the group's clock."""
    rank = max(range(len(seconds)), key=lambda r: seconds[r])
    return rank, seconds[rank]


def read_single(run_dir: pathlib.Path, arm: str) -> list:
    """The single-arm readout. Output format unchanged since #439."""
    entries = load(run_dir, arm)
    print(f"== arm {arm!r} in {run_dir} ==")
    total_h2d = 0
    total_remote = 0
    seen = work_counters(entries)
    for name, payload in entries:
        totals = payload.get("totals", {})
        policy = totals.get("host_shard_policy", "?")
        reach = totals.get("host_shard_reachability", "?")
        compute_policy = totals.get("moe_compute_policy", "?")
        vector = totals.get("moe_compute_vector", "?")
        h2d = int(totals.get("h2d_bytes", 0))
        remote = int(totals.get("remote_h2d_bytes", 0))
        total_h2d += h2d
        total_remote += remote
        share = (100.0 * remote / h2d) if h2d else 0.0
        work = {key: totals.get(key, "?") for key in WORK_KEYS}
        print(
            f"  {name}: policy={policy} reachability={reach} "
            f"compute={compute_policy} vector={vector} "
            f"h2d={h2d / 2**30:.1f} GiB remote={remote / 2**30:.1f} GiB "
            f"({share:.1f} %) hit_rate={totals.get('hit_rate', '?')} "
            f"unique_hit_rate={totals.get('unique_hit_rate', '?')} "
            "work=" + "/".join(f"{key}={work[key]}" for key in WORK_KEYS)
        )
        if arm == "proportional" and reach != "shared-cold-tier":
            print(
                "    WARNING: this arm did NOT attach the shared cold tier. "
                "Any delta reported against it is a delta between two "
                "baselines."
            )
        if arm in COMPUTE_ARMS and not str(compute_policy).startswith("link-"):
            print(
                "    WARNING: this arm did NOT move the compute assignment "
                f"(moe_compute_policy={compute_policy!r}). Any delta reported "
                "against it is a delta between two baselines."
            )
        if arm == "compute-cal" and compute_policy != "link-proportional-calibrated":
            print(
                "    WARNING: the calibrated sub-arm ran UNCALIBRATED "
                "(SGLANG_MOE_COLD_TRAFFIC_COEFFICIENTS did not reach the "
                "launcher); it is a duplicate of the 'compute' arm."
            )

    print(
        f"  group h2d={total_h2d / 2**30:.1f} GiB remote={total_remote / 2**30:.1f} GiB"
    )
    print(f"  work point of this arm: {format_work_point(seen)}")
    if any(len(values) > 1 for values in seen.values()):
        print(
            "    WARNING: the ranks of this arm disagree on their work "
            "counters, so this is NOT the final revision -- each rank was "
            "caught at its own point of the run. Read the dumps AFTER "
            "teardown. Dividing this arm's counters by another arm's is "
            "undefined; see ARM3_COMPUTE.md, 'Which revision to read'."
        )
    else:
        print(
            "    Quotable only against another arm at the SAME work point: "
            "run this tool with --against <other arm>, which is the only path "
            "that produces a cross-arm number and which refuses when the work "
            "does not match."
        )
    if arm in COMPUTE_ARMS:
        print(
            "  Reminder: for a slice-3 arm a NON-null per-rank H2D delta is "
            "the predicted result, and a null one falsifies it. Compare "
            "per-rank against the table in ARM3_COMPUTE.md, never the group "
            "total -- the group total is predicted ~unchanged and the whole "
            "effect is in which rank carries which part of it."
        )
    else:
        print(
            "  Reminder: a null per-rank H2D delta is the PREDICTED result for "
            "slice 2 (byte ownership moved, compute did not). A non-null one "
            "falsifies the analysis in boot_ab.sh and is the finding."
        )
    return entries


def compare(
    run_dir: pathlib.Path,
    arm_a: str,
    arm_b: str,
    links_gb_s=None,
    tolerance_pct: float = DEFAULT_WORK_TOLERANCE_PCT,
) -> dict:
    """Cross-arm readout, gated on the work point. Refuses or returns numbers.

    Every quantity this returns is an ACCUMULATING counter or is derived from
    one, which is precisely why the gate runs first and why nothing is printed
    or returned when it refuses. There is no partial result: a caller that
    catches the refusal and prints "the delta was roughly X" has reintroduced
    the defect.
    """
    entries_a = load(run_dir, arm_a)
    entries_b = load(run_dir, arm_b)
    if len(entries_a) != len(entries_b):
        raise WorkMatchRefused(
            "rank-count-mismatch",
            f"arm {arm_a!r} has {len(entries_a)} rank dumps and arm {arm_b!r} "
            f"has {len(entries_b)}. Two arms of one window are the same group; "
            "a different rank count means a different geometry, and per-rank "
            "numbers are not aligned.",
        )
    work_a = final_work_point(arm_a, entries_a)
    work_b = final_work_point(arm_b, entries_b)
    mismatch = require_work_matched(arm_a, work_a, arm_b, work_b, tolerance_pct)

    h2d_a = h2d_per_rank(entries_a)
    h2d_b = h2d_per_rank(entries_b)
    result = {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "work_a": work_a,
        "work_b": work_b,
        "work_mismatch_pct": mismatch,
        "tolerance_pct": tolerance_pct,
        "h2d_a": h2d_a,
        "h2d_b": h2d_b,
        "h2d_delta_pct": [
            (100.0 * (b - a) / a) if a else 0.0 for a, b in zip(h2d_a, h2d_b)
        ],
        "group_h2d_delta_pct": (
            100.0 * (sum(h2d_b) - sum(h2d_a)) / sum(h2d_a) if sum(h2d_a) else 0.0
        ),
    }
    if links_gb_s:
        seconds_a = transfer_seconds(h2d_a, links_gb_s)
        seconds_b = transfer_seconds(h2d_b, links_gb_s)
        rank_a, clock_a = clock_rank(seconds_a)
        rank_b, clock_b = clock_rank(seconds_b)
        result.update(
            {
                "links_gb_s": list(links_gb_s),
                "transfer_s_a": seconds_a,
                "transfer_s_b": seconds_b,
                "clock_rank_a": rank_a,
                "clock_rank_b": rank_b,
                "clock_s_a": clock_a,
                "clock_s_b": clock_b,
                "speedup": (clock_a / clock_b) if clock_b else 0.0,
            }
        )
    return result


def format_work_point_from(work: dict) -> str:
    return " ".join(f"{key}={work[key]}" for key in WORK_KEYS)


def print_comparison(run_dir: pathlib.Path, result: dict) -> None:
    arm_a = result["arm_a"]
    arm_b = result["arm_b"]
    print(f"== work-matched comparison {arm_a!r} -> {arm_b!r} in {run_dir} ==")
    print(f"  work point {arm_a}: {format_work_point_from(result['work_a'])}")
    print(f"  work point {arm_b}: {format_work_point_from(result['work_b'])}")
    print(
        "  work match: "
        + " ".join(
            f"{key}={result['work_mismatch_pct'][key]:.4f} %" for key in WORK_KEYS
        )
        + f" (tolerance {result['tolerance_pct']:.3f} %) -> MATCHED"
    )
    for rank, (a, b, delta) in enumerate(
        zip(result["h2d_a"], result["h2d_b"], result["h2d_delta_pct"])
    ):
        print(
            f"  tp{rank} h2d {a / 2**30:.1f} GiB -> {b / 2**30:.1f} GiB "
            f"({delta:+.1f} %)"
        )
    print(f"  group h2d {result['group_h2d_delta_pct']:+.1f} %")
    if "speedup" in result:
        links = "/".join(f"{link:g}" for link in result["links_gb_s"])
        print(f"  transfer term at links {links} GB/s:")
        print(
            f"    {arm_a}: clock tp{result['clock_rank_a']} "
            f"{result['clock_s_a']:.2f} s"
        )
        print(
            f"    {arm_b}: clock tp{result['clock_rank_b']} "
            f"{result['clock_s_b']:.2f} s"
        )
        print(f"    speedup {result['speedup']:.4f}x")
    else:
        print(
            "  no --links given, so no transfer term: the clock is "
            "h2d_bytes / link_GB_s per rank and the link vector is a "
            "measurement of the rig, never a default."
        )


def parse_links(raw: str) -> list:
    if not raw:
        return []
    try:
        return [float(part) for part in raw.split(",")]
    except ValueError:
        raise SystemExit(f"--links wants decimal GB/s per rank, got {raw!r}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="read_arm.py",
        description="Read one #394 arm, or compare two at a common work point.",
    )
    parser.add_argument("run_dir")
    parser.add_argument("arm")
    parser.add_argument(
        "--against",
        default="",
        metavar="ARM",
        help="the other arm of the window. THE ONLY path to a cross-arm "
        "number, and it refuses when the two arms did not do the same work.",
    )
    parser.add_argument(
        "--links",
        default="",
        metavar="GB/s,GB/s,...",
        help="measured per-rank H2D link speeds, for the transfer term. Read "
        "them off preflight's table; there is no default.",
    )
    parser.add_argument(
        "--work-tolerance-pct",
        type=float,
        default=DEFAULT_WORK_TOLERANCE_PCT,
        help=f"how far the arms' work counters may differ "
        f"(default {DEFAULT_WORK_TOLERANCE_PCT} %%, below the window's own "
        "A-vs-A floor by construction).",
    )
    args = parser.parse_args(argv)

    run_dir = pathlib.Path(args.run_dir)
    if not args.against:
        read_single(run_dir, args.arm)
        return 0

    try:
        result = compare(
            run_dir,
            args.arm,
            args.against,
            parse_links(args.links),
            args.work_tolerance_pct,
        )
    except WorkMatchRefused as refusal:
        print(
            f"REFUSED ({refusal.reason}): {refusal.message}",
            file=sys.stderr,
        )
        return EXIT_REFUSED
    print_comparison(run_dir, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
