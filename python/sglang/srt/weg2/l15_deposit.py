"""L15-14: the phase-2 deposit -- a running P prefill writes its finished
tokens straight into D's held deposit region (plan 2.5), so the P->D flip
needs no L2 re-read for them.

The pure part (this module):

* :func:`deposit_region` -- D side at publish time: the free part of the hold
  region after the held spans. Global D slots ``[e0, e1)`` with ``e0`` the
  held high-water ``L_H`` (blocks of ``S`` slots, from the manifest's per-rank
  keep rows) and ``e1`` the last full block every CAPPED rank can still keep
  (cap rows // ratio); ranks with cap 0 are listed as ``skip_ranks`` -- their
  rows are never deposited, they come back from L2 at the wake. Anchor rows
  ``[a0, a1)``: after the held anchors up to the split's anchor cap.
* :class:`DepositBook` -- front side: hands each P request a contiguous slot
  range ``[e_start, e_start + n)`` and one anchor row inside the region, in
  arrival order, until it is full (then: no deposit, today's path). The front
  is the one place that sees every P request's exact prompt length before any
  stage admits it, so every P stage reads the SAME assignment (rid-keyed
  hint file), no cross-stage agreement needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class DepositRegion:
    e0: int
    e1: int
    a0: int
    a1: int
    skip_ranks: Tuple[int, ...]


def deposit_region(rows_by_rank: Sequence[int], prefix: Sequence[int],
                   caps: Sequence[int], anchor_slots: int,
                   anchor_cap: int) -> Optional[DepositRegion]:
    """See the module docstring; None when nothing is left to deposit into."""
    S = int(prefix[-1])
    R = len(prefix) - 1
    ratios = [int(prefix[r + 1]) - int(prefix[r]) for r in range(R)]
    blocks_held = max((int(rows_by_rank[r]) // ratios[r]
                       for r in range(R) if ratios[r] > 0), default=0)
    capped = [r for r in range(R) if int(caps[r]) > 0 and ratios[r] > 0]
    if not capped:
        return None
    blocks_cap = min(int(caps[r]) // ratios[r] for r in capped)
    e0, e1 = blocks_held * S, blocks_cap * S
    if e1 <= e0:
        return None
    a0, a1 = int(anchor_slots), int(anchor_cap) + 1
    skip = tuple(r for r in range(R) if int(caps[r]) <= 0)
    return DepositRegion(e0, e1, a0, max(a0, a1), skip)


class DepositBook:
    """Front-side assignment of deposit slots and anchor rows."""

    def __init__(self, region: DepositRegion):
        self.region = region
        self._cursor = region.e0
        self._anchor = region.a0
        self.assigned: Dict[str, Tuple[int, int, int]] = {}

    def assign(self, rid: str, n_tokens: int) -> Optional[Tuple[int, int, int]]:
        """(e_start, n, anchor_row) for ``rid``, or None when the region or
        the anchor rows are exhausted (the request goes today's way)."""
        if rid in self.assigned:
            return self.assigned[rid]
        n = int(n_tokens)
        if n <= 0 or self._cursor + n > self.region.e1:
            return None
        if self._anchor >= self.region.a1:
            return None
        got = (self._cursor, n, self._anchor)
        self._cursor += n
        self._anchor += 1
        self.assigned[rid] = got
        return got
