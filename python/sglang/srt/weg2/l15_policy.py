"""L1.5 (L15-02) hold-set selection policy, pure and stdlib-only.

The scheduler owns the sleep transition; this module only decides *which*
requests a D rank keeps resident in KV/weights when it sleeps, and formats
the one-line SHADOW log used during the shadow-instrumentation phase.

Policy (see docs/L15-PLAN-0930.md rule 2.2):
- Anchorless requests (anchor_depth != kv_depth) are excluded up front with
  reason "anchorless"; they never enter the candidate ordering.
- Candidates are ordered kind seat > parked > served, then last_active
  descending (larger = younger), then rid ascending (deterministic tie-break).
  The ordering is computed internally, so the result is independent of the
  input ordering.
- Admit greedily in that order while every rank r keeps sum(rows) <=
  cap_rows_by_rank[r] and anchors <= cap_anchor_slots (one anchor slot per
  admitted request). A candidate that does not fit is skipped with reason
  "no_room"; a candidate that fits in rows but is turned away because the
  anchor cap is full is skipped with reason "anchor_full"; later smaller
  candidates may still fit.
- A rank whose cap is 0 (the 5090 / TP0 rank) does NOT block: its rows are
  filled from L2 at the wake, so cap 0 means "not held here", not "no room"
  -- only ranks with cap > 0 are checked. HoldSet.rows_by_rank still sums the
  admitted rows for cap-0 ranks (they are part of the sum).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

_KIND_ORDER = {"seat": 0, "parked": 1, "served": 2}


@dataclass(frozen=True)
class Candidate:
    """One request a D rank could keep resident when it sleeps."""

    rid: str
    kind: str  # "seat" | "parked" | "served"
    last_active: float  # larger = younger
    rows_by_rank: Tuple[int, ...]  # compact KV rows this request holds per D rank
    anchor_depth: int
    kv_depth: int


@dataclass(frozen=True)
class HoldSet:
    """Result of one hold-set selection."""

    rids: Tuple[str, ...]
    rows_by_rank: Tuple[int, ...]
    anchors: int
    excluded: Tuple[Tuple[str, str], ...]  # (rid, reason)


def _order_key(candidate: Candidate):
    # kind seat > parked > served, then last_active descending (negate), then rid ascending.
    return (_KIND_ORDER[candidate.kind], -candidate.last_active, candidate.rid)


def select_hold(
    candidates: Sequence[Candidate],
    cap_rows_by_rank: Sequence[int],
    cap_anchor_slots: int,
) -> HoldSet:
    """Select the resident hold set under per-rank row caps and an anchor cap.

    The MiB budget is already converted to per-rank row caps by the caller, so
    this function only compares integer rows; it applies no additional ceiling.
    """
    candidates = list(candidates)
    excluded: Dict[str, Tuple[str, str]] = {
        c.rid: (c.rid, "anchorless")
        for c in candidates
        if c.anchor_depth != c.kv_depth
    }
    ordered = sorted(
        (c for c in candidates if c.anchor_depth == c.kv_depth), key=_order_key
    )

    rem = list(cap_rows_by_rank)
    admitted = []
    for c in ordered:
        fits = True
        for r, rows in enumerate(c.rows_by_rank):
            # "Not held here" is decided by the ORIGINAL cap, never by the
            # remaining capacity: cap 0 (the 5090 / TP0 rank, refilled from
            # L2 at the wake) does not block, but a capped rank that is
            # exactly full (rem == 0) DOES block -- skipping on rem == 0
            # would over-admit past the cap and OOM on metal.
            if cap_rows_by_rank[r] == 0:
                continue
            if rows > rem[r]:
                fits = False
                break
        if fits and len(admitted) < cap_anchor_slots:
            admitted.append(c)
            for r, rows in enumerate(c.rows_by_rank):
                rem[r] -= rows
        elif fits:
            # Fits in rows but the anchor cap is full: a distinct reason
            # (audit item 10) keeps the log unambiguous vs row-cap "no_room".
            excluded[c.rid] = (c.rid, "anchor_full")
        else:
            excluded[c.rid] = (c.rid, "no_room")

    rows_by_rank = tuple(sum(c.rows_by_rank[r] for c in admitted) for r in range(len(rem)))
    return HoldSet(
        rids=tuple(c.rid for c in admitted),
        rows_by_rank=rows_by_rank,
        anchors=len(admitted),
        excluded=tuple(excluded.values()),
    )


def shadow_line(epoch: int, hs: HoldSet, cap_rows_by_rank: Sequence[int]) -> str:
    """One-line SHADOW log of a hold-set selection (shadow phase only)."""
    fmt = lambda values: ",".join(str(v) for v in values)
    return (
        f"L15-SHADOW at=sleep epoch={epoch} n={len(hs.rids)} "
        f"rows_by_rank={fmt(hs.rows_by_rank)} anchors={hs.anchors} "
        f"cap={fmt(cap_rows_by_rank)} excluded={len(hs.excluded)}"
    )
