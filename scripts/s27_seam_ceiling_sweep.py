"""#631 section 2.1: price the ROW-BLOCKED seam against the rig geometry.

Exact arithmetic, no boot. ``staging reserved`` on a flip DONE line IS
``_staging_bytes()``, a pure function of the plan, so a sweep over the
block count run through the REAL accounting cannot drift from what the
gate will decide -- which is the point of importing the runtime here
instead of re-deriving the formula.

What it answers: for each block count B, how large a pool can the binding
rank still flip at FULL OCCUPANCY without eating the 1024 MiB corridor
reserve. HANDOFF_669 measured the 2.1b answer at ~501,000 and located the
blocker in the backing transient (1821 of 2508 MiB at 600,000).

CALIBRATION. One free parameter, the per-row-per-layer byte width at full
head count, taken from the same measured point HANDOFF_669 used
(1132.0 MiB of staging at 163626 live slots) or overridden with --row-bytes.
The rank shares come from the TP vector, so a TP layer's span is the full
row width scaled by that rank's head share -- which is why the release and
commit legs balance exactly per rank and the whole slack term is
TRANSIENT rather than net growth.
"""

from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace as NS

from sglang.srt.managers.phase_flip_runtime import PP_TO_TP, TP_TO_PP, PhaseFlipRuntime

MIB = 1024 * 1024

# The rig: 16 full-attention ordinals over three PP stages, TP head vector.
RIG_MAP = ((0, 1, 2, 3, 4, 5, 6), (7, 8, 9, 10, 11), (12, 13, 14, 15))
RIG_VEC = (14, 10, 8)


def _runtime(rank, blocks, chunk_bytes, restore_first=True):
    rt = PhaseFlipRuntime.__new__(PhaseFlipRuntime)
    rt._map = RIG_MAP
    rt._vec = RIG_VEC
    rt._rank = rank
    rt._seam_row_blocks = blocks
    rt._seam_restore_first = restore_first
    rt._n_waves = None
    rt._pools_alias = lambda: False
    rt._seam_backing_is_swappable = lambda: True
    rt._seam_swap = lambda: NS(
        is_swappable=True,
        is_span_swappable=lambda d: True,
        commit_chunk_bytes=lambda d: chunk_bytes,
    )
    return rt


def _views(rank, pool_rows, row_bytes):
    """PP and TP pool views for this rank.

    PP: this rank's stage layers, every head, every pool row.
    TP: every layer, this rank's head share, every pool row.
    """
    share = RIG_VEC[rank] / sum(RIG_VEC)
    pp = NS(
        num_layers=len(RIG_MAP[rank]),
        num_rows=pool_rows,
        row_nbytes=lambda i: row_bytes,
    )
    tp = NS(
        num_layers=16,
        num_rows=pool_rows,
        row_nbytes=lambda i: int(row_bytes * share),
    )
    return pp, tp


def slack_mib(rank, pool_rows, row_bytes, blocks, chunk_bytes, direction):
    rt = _runtime(rank, blocks, chunk_bytes)
    pp, tp = _views(rank, pool_rows, row_bytes)
    src, dst = (pp, tp) if direction == PP_TO_TP else (tp, pp)
    waves = rt._flip_waves(direction)
    return rt._backing_slack_bytes(direction, src, dst, waves) / MIB


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--row-bytes", type=int, default=2048)
    ap.add_argument("--budget-mib", type=float, default=753.0,
                    help="staging room the binding card has at --at-pool "
                         "(HANDOFF_669 measured 753 MiB at 600000)")
    ap.add_argument("--at-pool", type=int, default=600000)
    ap.add_argument("--payload-mib", type=float, default=687.0,
                    help="the non-backing legs at --at-pool (HANDOFF_669)")
    ap.add_argument("--chunk-mib", type=int, nargs="*", default=[0, 2, 8, 32])
    ap.add_argument("--blocks", type=int, nargs="*",
                    default=[1, 2, 4, 8, 16, 32, 64, 128])
    a = ap.parse_args()

    print(f"rig map={RIG_MAP} vec={RIG_VEC} row_bytes={a.row_bytes}")
    print(f"pool={a.at_pool} budget={a.budget_mib} MiB payload={a.payload_mib} MiB\n")

    # REPORT PER RANK, NEVER AS ONE WORST. ``ordered_layer_waves`` assigns
    # the transient ON PURPOSE to the largest-share rank -- the card the
    # operator sized largest, which on this rig is the 5090 and is the one
    # card with GiB of corridor slack. Collapsing the three into a single
    # max reads that deliberate absorption as the binding constraint and
    # makes the seam look far worse than it is; it cost this analysis one
    # wrong conclusion before the split was put back in.
    absorber = max(range(3), key=lambda r: RIG_VEC[r])
    print(f"absorber rank (largest TP share) = {absorber}; "
          f"the others are the BINDING cards\n")

    for chunk_mib in a.chunk_mib:
        chunk = chunk_mib * MIB
        print(f"--- commit chunk {chunk_mib} MiB "
              f"({'no floor' if chunk_mib == 0 else 'floor active'}) ---")
        head = " ".join(
            f"{'r%d%s' % (r, '*' if r == absorber else ''):>9}" for r in range(3)
        )
        print(f"{'B':>5} {head}   {'binding worst':>13} {'+payload':>10} "
              f"{'fits?':>7}")
        for b in a.blocks:
            per_rank = [
                max(
                    slack_mib(r, a.at_pool, a.row_bytes, b, chunk, d)
                    for d in (PP_TO_TP, TP_TO_PP)
                )
                for r in range(3)
            ]
            binding = max(v for r, v in enumerate(per_rank) if r != absorber)
            total = binding + a.payload_mib
            cells = " ".join(f"{v:>9.1f}" for v in per_rank)
            print(f"{b:>5} {cells}   {binding:>13.1f} {total:>10.1f} "
                  f"{'YES' if total <= a.budget_mib else 'no':>7}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
