"""L15-16: the cap-0 rank parks its held KV rows on a capped rank's card.

User law 02.10.: with L1.5 on, the flip must be faster in BOTH directions and
move fewer host bytes. A rank with cap 0 (its card has no room in the P
layout) keeps nothing through the P phase; today it refills its owned held
rows from L2 at the wake (H2D, ``L15-HOSTBYTES h2d_refill``). Here it parks
them instead -- card to card over the BAR1 lanes -- into the FREE part of a
capped rank's hold region (compact rows ``[keep_rows_r, cap_r)``, kept
physically by the split hold extents), and takes them back at the wake.

This module is the pure plan (no torch, no transport):

* :func:`park_plan` -- which cap-0 rank's compact rows ``[0, keep_rows)`` go
  to which capped rank at which row, largest free region first, split over
  several capped ranks when one does not suffice; refused by name when the
  capped ranks' free rows cannot take them all (the L2 refill serves then).
* keyed by cap, never by card name or ordinal (user order 07:05Z).

The anchors (GDN head shares, a few MiB each) stay on the L2 refill; the
deposit region (L15-14) shares the same free rows and is off while a park
is planned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ParkPiece:
    src: int        # cap-0 rank whose compact rows are parked
    dst: int        # capped rank holding them through the P phase
    src_row: int    # first compact row on src
    dst_row: int    # first row on dst (inside its hold region, after its own)
    rows: int


def park_plan(keep_rows: Sequence[int], caps: Sequence[int]
              ) -> Tuple[List[ParkPiece], Optional[str]]:
    """``(pieces, None)`` or ``([], reason)``.

    ``keep_rows[r]``: rank r's compacted keep rows of this hold (the
    manifest's rows_by_rank); ``caps[r]``: its hold region in rows (0 = not
    held here). Every cap-0 rank with rows parks ALL of them, or the plan is
    refused (a partial park would need the L2 refill anyway)."""
    R = len(keep_rows)
    if len(caps) != R:
        return [], "keep_rows for %d ranks, caps for %d" % (R, len(caps))
    free = {r: int(caps[r]) - int(keep_rows[r]) for r in range(R) if int(caps[r]) > 0}
    if any(v < 0 for v in free.values()):
        bad = [r for r, v in free.items() if v < 0]
        return [], "capped rank(s) %s keep more rows than their cap" % bad
    cursor = {r: int(keep_rows[r]) for r in free}
    pieces: List[ParkPiece] = []
    for src in range(R):
        if int(caps[src]) > 0 or int(keep_rows[src]) <= 0:
            continue
        need, row = int(keep_rows[src]), 0
        while need > 0:
            # largest free region first; ties to the lower rank (deterministic)
            cands = sorted((r for r in free if free[r] > 0),
                           key=lambda r: (-free[r], r))
            if not cands:
                have = sum(max(0, v) for v in free.values())
                return [], ("cap-0 rank %d needs %d more rows to park, capped "
                            "ranks have %d free" % (src, need, have))
            dst = cands[0]
            n = min(need, free[dst])
            pieces.append(ParkPiece(src, dst, row, cursor[dst], n))
            cursor[dst] += n
            free[dst] -= n
            need -= n
            row += n
    return pieces, None


def parked_rows_on(pieces: Sequence[ParkPiece], dst: int) -> int:
    return sum(p.rows for p in pieces if p.dst == dst)


def park_bytes(pieces: Sequence[ParkPiece], row_bytes_all_layers: int) -> int:
    """Bytes the park moves per direction (all layers, K and V)."""
    return sum(p.rows for p in pieces) * int(row_bytes_all_layers)
