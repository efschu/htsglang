#!/usr/bin/env python3
"""Read one #394 slice-2 arm out of its #390 expert-stats dumps.

Prints the three things an arm has to be able to prove about itself, and
nothing else -- a server log is never read into anybody's context.

  1. WHICH ARM produced this file. ``host_shard.policy`` (equal vs
     link-proportional) and ``host_shard.reachability`` (local-only, refused,
     shared-cold-tier). A proportional arm whose reachability says anything
     other than ``shared-cold-tier`` silently ran the baseline, which is the
     way an A/B most often reports a null that was never tested.
  2. PER-RANK H2D, the primary readout. Slice 2 moves byte ownership, not
     compute, so the prediction is a null delta here; see boot_ab.sh. The
     remote share says how much of it came out of a peer's segment.
  3. The x4 rank's implied transfer time, which is the clock the whole feature
     is aimed at.

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

    print(f"== arm {arm!r} in {run_dir} ==")
    total_h2d = 0
    total_remote = 0
    for name, payload in load(run_dir, arm):
        totals = payload.get("totals", {})
        policy = totals.get("host_shard_policy", "?")
        reach = totals.get("host_shard_reachability", "?")
        h2d = int(totals.get("h2d_bytes", 0))
        remote = int(totals.get("remote_h2d_bytes", 0))
        total_h2d += h2d
        total_remote += remote
        share = (100.0 * remote / h2d) if h2d else 0.0
        print(
            f"  {name}: policy={policy} reachability={reach} "
            f"h2d={h2d / 2**30:.1f} GiB remote={remote / 2**30:.1f} GiB "
            f"({share:.1f} %) hit_rate={totals.get('hit_rate', '?')}"
        )
        if arm == "proportional" and reach != "shared-cold-tier":
            print(
                "    WARNING: this arm did NOT attach the shared cold tier. "
                "Any delta reported against it is a delta between two "
                "baselines."
            )

    print(
        f"  group h2d={total_h2d / 2**30:.1f} GiB remote={total_remote / 2**30:.1f} GiB"
    )
    print(
        "  Reminder: a null per-rank H2D delta is the PREDICTED result for "
        "slice 2 (byte ownership moved, compute did not). A non-null one "
        "falsifies the analysis in boot_ab.sh and is the finding."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
