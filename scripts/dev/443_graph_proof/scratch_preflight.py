#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#443 preflight: does the captured decode graph's scratch region fit?

The capturable offload step assigns one scratch slot per DISTINCT routed spill
expert, so a bucket is capture-eligible iff

    worst_case_unique_spill(bs * top_k, E_local, R_local) <= SCRATCH_SLOTS

Getting this wrong costs a full model load: the check fires in ``layer.py``
during warmup capture, several minutes in. This script answers it in a second,
and it calls the SAME function the runtime calls, so the two cannot drift.

Inputs are per RANK, because ``--rank-tp-ratio`` and
``--rank-moe-resident-fraction`` make E_local and R_local rank-dependent -- the
binding rank is whichever one has the largest cold set, not rank 0.

Usage:
  scratch_preflight.py --experts-per-rank 32,24,24 --resident-fraction 0.485,0.42,0.42 \
                       --top-k 8 --max-graph-bs 4 --scratch-slots 6
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "python"
    ),
)

from sglang.srt.layers.moe.expert_offload import worst_case_unique_spill  # noqa: E402


def _csv_ints(text: str):
    return [int(x) for x in text.split(",") if x.strip()]


def _csv_floats(text: str):
    return [float(x) for x in text.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experts-per-rank", required=True, type=_csv_ints)
    ap.add_argument("--resident-fraction", required=True, type=_csv_floats)
    ap.add_argument("--top-k", required=True, type=int)
    ap.add_argument("--max-graph-bs", required=True, type=int)
    ap.add_argument("--scratch-slots", required=True, type=int)
    args = ap.parse_args()

    experts = args.experts_per_rank
    fractions = args.resident_fraction
    if len(experts) != len(fractions):
        print(
            f"FAIL --experts-per-rank has {len(experts)} entries and "
            f"--resident-fraction has {len(fractions)}; one per rank, in rank order",
            file=sys.stderr,
        )
        return 2

    routed_slots = args.max_graph_bs * args.top_k
    worst = 0
    print(
        f"routed slots per captured step: bs {args.max_graph_bs} x top_k {args.top_k} = {routed_slots}"
    )
    print("rank  E_local  R_local  cold  needed_slots")
    for rank, (e, frac) in enumerate(zip(experts, fractions)):
        # Same rounding as resident_fraction_for_rank -> plan_load_time_staging.
        r = int(math.ceil(frac * e))
        need = worst_case_unique_spill(routed_slots, e, r)
        worst = max(worst, need)
        print(f"{rank:>4}  {e:>7}  {r:>7}  {e - r:>4}  {need:>12}")

    print(f"\nbinding requirement: SGLANG_MOE_SCRATCH_SLOTS >= {worst}")
    if args.scratch_slots < worst:
        print(
            f"FAIL SGLANG_MOE_SCRATCH_SLOTS={args.scratch_slots} < {worst}. Either "
            f"raise it (costs {worst - args.scratch_slots} more resident slots of "
            f"VRAM per layer) or lower SGLANG_MOE_OFFLOAD_MAX_GRAPH_BS. Note the "
            f"corridor rule before raising it: the extra slots come out of the KV "
            f"pool, and free VRAM must stay >= 400 MiB per card.",
            file=sys.stderr,
        )
        return 1
    print(f"PREFLIGHT OK scratch_slots={args.scratch_slots} >= {worst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
