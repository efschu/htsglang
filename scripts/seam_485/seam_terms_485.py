#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#485 -- decompose the measured seam transient into named terms, per flip.

The seam census reports one number per flip: transient = entry_free - trough.
The planner's census reports a different one: at_rest_baseline - trough. The
difference between them is a term in its own right (the rank is already below
its best at-rest level when the flip begins) and it is the term every
mechanism proposal so far has silently assumed away.

Terms, all measured, all at the instant of the trough:

  entry_deficit  at_rest_baseline - entry_free
  wave           net of backing_restore_span / backing_release_span held
  kv_stage       net of kv_pack / kv_local_read / kv_write held
  arena          net of weights_refill held (the |PP-TP| layout delta + checksum)
  gdn            net of gdn_state held
  other          plan / allocator_cache_release / cutover / backing_restore held

By construction: census_seam = entry_deficit + (-sum of the held terms).
"""
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seam_decompose_485 import parse  # noqa: E402

GROUP = {
    "backing_restore_span": "wave",
    "backing_release_span": "wave",
    "backing_restore": "wave",
    "kv_pack": "kv_stage",
    "kv_local_read": "kv_stage",
    "kv_write": "kv_stage",
    "weights_refill": "arena",
    "gdn_state": "gdn",
    "plan": "other",
    "allocator_cache_release": "other",
    "cutover": "other",
    "done": "other",
}
ORDER = ["wave", "kv_stage", "arena", "gdn", "other"]


def held_at_trough(flip):
    """Cumulative step by group up to and including the trough mark."""
    ti = flip.trough_index()
    out = defaultdict(float)
    for m in flip.marks[: ti + 1]:
        out[GROUP.get(m["stage"], "other")] += m["step"]
    return out


def main():
    census_dir, log = sys.argv[1], sys.argv[2]
    at_rest, seam_state = {}, {}
    for p in sorted(glob.glob(os.path.join(census_dir, "transient_pp*.json"))):
        d = json.load(open(p))
        r = int(d["pp_rank"])
        at_rest[r] = d["baseline_free_mib"]
        for k, v in (d.get("transient_mib_by_load_state") or {}).items():
            if k.startswith("SEAM_"):
                seam_state[(r, k)] = v

    flips = parse(log)
    by = defaultdict(list)
    for f in flips:
        by[(f.rank, f.dir)].append(f)

    print(f"census {census_dir}")
    print(f"log    {log}\n")
    hdr = (f"{'rank/leg':<22}{'n':>4}{'census':>9}{'at_rest':>9}"
           f"{'entry_def':>11}{'wave':>10}{'kv_stg':>9}{'arena':>9}"
           f"{'gdn':>7}{'other':>8}{'d_alloc':>9}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for (rank, direction), fs in sorted(by.items()):
        state = "SEAM_" + direction.upper()
        census = seam_state.get((rank, state))
        rest = at_rest[rank]
        # the flip that SETS the census number is the worst-trough flip
        tgt = min(fs, key=lambda f: f.trough)
        held = held_at_trough(tgt)
        ed = rest - tgt.baseline
        ti = tgt.trough_index()
        a0 = next((m["alloc"] for m in tgt.marks if m["alloc"] is not None), None)
        a1 = tgt.marks[ti]["alloc"]
        dalloc = (a1 - a0) if (a0 is not None and a1 is not None) else float("nan")
        print(f"{f'{rank} {direction}':<22}{len(fs):>4}"
              f"{(census if census is not None else float('nan')):>9.0f}{rest:>9.0f}"
              f"{ed:>11.0f}"
              + "".join(f"{held.get(g, 0.0):>{w}.0f}"
                        for g, w in zip(ORDER, (10, 9, 9, 7, 8)))
              + f"{dalloc:>9.0f}")
        rows.append((rank, direction, census, rest, ed, held, tgt))

    # flip-to-flip spread of each term, per rank/leg
    print("\n\nFLIP-TO-FLIP SPREAD (min / p50 / max over every flip)\n")
    for (rank, direction), fs in sorted(by.items()):
        agg = defaultdict(list)
        for f in fs:
            h = held_at_trough(f)
            for g in ORDER:
                agg[g].append(h.get(g, 0.0))
            agg["entry_deficit"].append(at_rest[rank] - f.baseline)
            agg["draw"].append(f.transient)
        print(f"  rank {rank} {direction}  (n={len(fs)})")
        for g in ["draw", "entry_deficit"] + ORDER:
            v = sorted(agg[g])
            print(f"     {g:<16}{v[0]:>9.0f}{v[len(v)//2]:>9.0f}{v[-1]:>9.0f}")
        print()


if __name__ == "__main__":
    main()
