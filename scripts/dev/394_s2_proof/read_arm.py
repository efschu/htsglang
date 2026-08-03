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
announces itself. Comparing two arms is still the reader's job -- their work
counters have to agree too, and the printed ``work=`` line is what that check
reads.

Usage:
  python3 read_arm.py <run_dir> <arm>
"""

import json
import pathlib
import sys


def load(run_dir: pathlib.Path, arm: str):
    files = sorted(run_dir.glob(f"expert_stats_{arm}.tp*ep*.json"))
    if not files:
        raise SystemExit(f"no expert_stats_{arm}.tp*ep*.json in {run_dir}")
    return [(f.name, json.loads(f.read_text())) for f in files]


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    run_dir = pathlib.Path(sys.argv[1])
    arm = sys.argv[2]

    #: Arms whose treatment is the COMPUTE assignment (#394 slice 3). Their
    #: prediction has the opposite sign to slice 2's, so they are named rather
    #: than inferred: an arm nobody listed here is read as a slice-2 arm.
    compute_arms = ("compute", "compute-cal")

    print(f"== arm {arm!r} in {run_dir} ==")
    total_h2d = 0
    total_remote = 0
    #: Work counters, per rank. A ratio between two arms is only defined at a
    #: common work point -- see "WHICH REVISION TO READ" above.
    work_keys = ("tokens", "forwards", "activations")
    work_seen: dict[str, set] = {key: set() for key in work_keys}
    for name, payload in load(run_dir, arm):
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
        work = {key: totals.get(key, "?") for key in work_keys}
        for key, value in work.items():
            work_seen[key].add(value)
        print(
            f"  {name}: policy={policy} reachability={reach} "
            f"compute={compute_policy} vector={vector} "
            f"h2d={h2d / 2**30:.1f} GiB remote={remote / 2**30:.1f} GiB "
            f"({share:.1f} %) hit_rate={totals.get('hit_rate', '?')} "
            f"unique_hit_rate={totals.get('unique_hit_rate', '?')} "
            "work=" + "/".join(f"{key}={work[key]}" for key in work_keys)
        )
        if arm == "proportional" and reach != "shared-cold-tier":
            print(
                "    WARNING: this arm did NOT attach the shared cold tier. "
                "Any delta reported against it is a delta between two "
                "baselines."
            )
        if arm in compute_arms and not str(compute_policy).startswith("link-"):
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
    work_point = " ".join(
        f"{key}={sorted(work_seen[key])[0]}" if len(work_seen[key]) == 1 else f"{key}=*"
        for key in work_keys
    )
    print(f"  work point of this arm: {work_point}")
    if any(len(values) > 1 for values in work_seen.values()):
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
            "compare this line with the other arm's before dividing any "
            "counter by any other."
        )
    if arm in compute_arms:
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
