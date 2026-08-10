#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#631: report the phase-flip mover's LIVE SET against its irreducible floor.

Runs the hermetic three-thread flip from
``test_phase_flip_mover_streaming_631`` and prints the high-water staging
bytes per direction next to the bytes the plan says the move owes. Point it
at a tree with the streaming packer and at one without to get the pair; the
probe excludes the persistent KV pools, so the two readings differ only by
copies the mover chose to keep.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "test/registered/unit/managers"))
import test_phase_flip_mover_streaming_631 as T  # noqa: E402

MIB = 1024 * 1024


def measure(direction):
    ref, live, pp_pools, pp_views, tp_pools, tp_views = T._make_layout_pools(
        T.MAP_625, T.VEC, T.NUM_SLOTS
    )
    runtimes = T._build_runtimes(pp_views, tp_views, live)
    persistent = T._pool_tensors(pp_pools, tp_pools)
    if direction == T.TP_TO_PP:
        errs = T._run_ranks_probing(
            runtimes, [T.PP_TO_TP] * 3, 0, T.LiveStorageProbe(exclude=persistent)
        )
        assert not [e for e in errs if e], errs
    src = pp_views[0] if direction == T.PP_TO_TP else tp_views[0]
    dst = tp_views[0] if direction == T.PP_TO_TP else pp_views[0]
    tr, out, inc, loc = T._plan_legs(live, 0, direction, src, dst)
    probe = T.LiveStorageProbe(exclude=persistent)
    errs = T._run_ranks_probing(runtimes, [direction] * 3, 0, probe)
    assert not [e for e in errs if e], errs
    predicted = runtimes[0]._staging_bytes(tr, direction, src, dst)
    return probe.peak, out, inc, loc, predicted


print(
    f"{'direction':<10} {'peak':>9} {'out':>8} {'in':>8} {'local':>8} "
    f"{'floor':>8} {'peak/floor':>11} {'_staging_bytes':>15}"
)
for d in (T.PP_TO_TP, T.TP_TO_PP):
    peak, out, inc, loc, predicted = measure(d)
    floor = max(out + inc, inc + loc)
    print(
        f"{d:<10} {peak / MIB:>8.1f}M {out / MIB:>7.1f}M {inc / MIB:>7.1f}M "
        f"{loc / MIB:>7.1f}M {floor / MIB:>7.1f}M {peak / floor:>10.2f}x "
        f"{predicted / MIB:>14.1f}M"
    )
