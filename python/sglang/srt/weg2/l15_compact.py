# SPDX-License-Identifier: Apache-2.0
"""L1.5 compaction plan for the D hold (L15-05, pure).

At the D sleep, after the partial tree reset, ONLY the held requests' slots
are still allocated; every other slot is free. The hold keeps a PREFIX
``[0, L_H)`` of D's global slots mapped (``L15-PLAN-0930``). This module
computes, purely and deterministically (stdlib only, no torch, no allocator
calls), how the held slots compact into that prefix:

* owner rule (``layers/dcp/owner.py``, used not restated): rank ``r`` owns
  global slot ``L`` iff ``prefix[r] <= L % S < prefix[r+1]``, with ``prefix``
  the cumulative owner vector and ``S = prefix[-1]``. A token must never
  change its owner rank (that would cross cards), so every compacted slot
  stays in the SAME class as its source;
* :func:`hold_prefix` sizes the prefix: the smallest ``L_H`` that is a
  multiple of ``S`` and holds each rank's need inside ``[0, L_H)`` (each
  block of ``S`` slots holds ``ratio_r`` slots of class ``r``);
* :func:`compact_plan` produces the moves: a held slot already below ``L_H``
  stays; every held slot at or above ``L_H`` moves to a FREE slot (not held
  by anyone) of its own class inside ``[0, L_H)``, taking the free targets of
  each class in ascending order and the sources in ascending old-slot order,
  with ``moves`` sorted by old slot;
* :func:`anchor_plan` does the same squeeze for the GDN state slots of the
  held requests: they compact to ``[0, A_H)`` with ``A_H = len(anchor_slots)``.

The compact pool row of a global slot (what the per-rank KV buffer addresses)
is ``(L // S) * ratio_r + (L % S - prefix[r])``; ``CompactPlan.rows_by_rank``
carries ``(L_H // S) * ratio_r``, the number of rows rank ``r`` keeps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

__all__ = [
    "CompactPlan",
    "anchor_plan",
    "compact_plan",
    "hold_prefix",
    "owner_of",
]


def _check_prefix(prefix: Sequence[int]) -> int:
    """Validate a cumulative owner vector, return S = prefix[-1]."""
    if len(prefix) < 2:
        raise ValueError(
            f"prefix must hold at least two cumulative bounds, got {list(prefix)!r}"
        )
    for bound in prefix:
        if int(bound) < 0:
            raise ValueError(f"prefix bounds must be non-negative, got {list(prefix)!r}")
    for r in range(len(prefix) - 1):
        if int(prefix[r]) > int(prefix[r + 1]):
            raise ValueError(
                f"prefix must be non-decreasing, got {list(prefix)!r}"
            )
    s = int(prefix[-1])
    if s <= 0:
        raise ValueError(f"prefix[-1] (S) must be positive, got {s}")
    return s


def owner_of(slot: int, prefix: Sequence[int]) -> int:
    """Rank owning global ``slot``: ``prefix[r] <= slot % S < prefix[r+1]``."""
    s = _check_prefix(prefix)
    residue = int(slot) % s
    for r in range(len(prefix) - 1):
        if int(prefix[r]) <= residue < int(prefix[r + 1]):
            return r
    # Only reachable when the residues [0, S) do not cover the vector start
    # (prefix[0] > 0): a residue below prefix[0] has no owner.
    raise ValueError(
        f"slot {slot} (residue {residue}) has no owning rank under prefix "
        f"{list(prefix)!r}"
    )


def hold_prefix(need_by_rank: Sequence[int], prefix: Sequence[int]) -> int:
    """Smallest multiple of S holding ``need_by_rank[r]`` class-r slots in [0, L_H).

    Each full block of ``S`` global slots contributes ``ratio_r =
    prefix[r+1] - prefix[r]`` slots of class ``r``, so ``blocks = ceil(need /
    ratio)`` per rank and ``L_H = max(blocks) * S``. A rank with ``ratio == 0``
    (e.g. rank 0 of the NF prefix ``[0, 0, 9, 16]``) can never hold a slot --
    a positive need on it is impossible and raises ValueError naming the rank.
    """
    s = _check_prefix(prefix)
    n_ranks = len(prefix) - 1
    if len(need_by_rank) != n_ranks:
        raise ValueError(
            f"need_by_rank has {len(need_by_rank)} entries but prefix "
            f"{list(prefix)!r} describes {n_ranks} ranks"
        )
    blocks = 0
    for r in range(n_ranks):
        need = int(need_by_rank[r])
        if need < 0:
            raise ValueError(f"need_by_rank[{r}] is negative: {need}")
        ratio = int(prefix[r + 1]) - int(prefix[r])
        if ratio == 0:
            if need > 0:
                raise ValueError(
                    f"rank {r} has ratio 0 in prefix {list(prefix)!r} (owns no "
                    f"slot) but need_by_rank[{r}] = {need}"
                )
            continue
        rank_blocks = -(-need // ratio)  # ceil division
        if rank_blocks > blocks:
            blocks = rank_blocks
    return blocks * s


@dataclass(frozen=True)
class CompactPlan:
    """The deterministic compaction of held slots into the L1.5 hold prefix.

    ``l_h``: prefix length, a multiple of S. ``moves``: (old_slot, new_slot)
    sorted by old slot; only slots at or above ``l_h`` move, each to a free
    slot of its own class. ``new_slots``: rid -> final global slots in token
    order. ``rows_by_rank``: ``(l_h // S) * ratio_r`` rows kept per rank.
    """

    l_h: int
    moves: Tuple[Tuple[int, int], ...]
    new_slots: Dict[str, Tuple[int, ...]]
    rows_by_rank: Tuple[int, ...]


def compact_plan(
    held: Mapping[str, Sequence[int]],
    prefix: Sequence[int],
    reserved: Sequence[int] = (),
) -> CompactPlan:
    """Plan the compaction of ``held`` (rid -> global slots, token order).

    All slots across all requests must be distinct. ``need_by_rank`` is the
    per-class count of held slots, ``L_H = hold_prefix(need)``; see the module
    docstring for the move rule. ``reserved`` (e.g. padding slot 0, the dummy
    write target for padded tokens) counts toward its owner's need, is never
    a compaction target and is not part of ``moves``/``new_slots``; the
    default ``()`` reproduces the pre-L15-11b behaviour exactly.
    """
    s = _check_prefix(prefix)
    n_ranks = len(prefix) - 1
    reserved_set = {int(x) for x in reserved}

    need = [0] * n_ranks
    held_all: set = set()
    for rid, slots in held.items():
        for slot in slots:
            slot = int(slot)
            if slot in held_all:
                raise ValueError(
                    f"slot {slot} is held twice (request {rid!r}); held slots "
                    "must be all distinct"
                )
            held_all.add(slot)
            need[owner_of(slot, prefix)] += 1
    for slot in reserved_set:
        need[owner_of(slot, prefix)] += 1

    l_h = hold_prefix(need, prefix)
    blocks = l_h // s
    rows_by_rank = tuple(
        blocks * (int(prefix[r + 1]) - int(prefix[r])) for r in range(n_ranks)
    )

    # Free slots of each class inside [0, l_h), ascending (scanning ascending).
    free_by_rank: list = [[] for _ in range(n_ranks)]
    for slot in range(l_h):
        if slot not in held_all and slot not in reserved_set:
            free_by_rank[owner_of(slot, prefix)].append(slot)

    # Sources at or above l_h, per class; paired ascending with the targets.
    src_by_rank: list = [[] for _ in range(n_ranks)]
    for slot in held_all:
        if slot >= l_h:
            src_by_rank[owner_of(slot, prefix)].append(slot)

    mapping: Dict[int, int] = {}
    for r in range(n_ranks):
        srcs = sorted(src_by_rank[r])
        targets = free_by_rank[r]
        if len(srcs) > len(targets):
            # Unreachable when hold_prefix is the sole sizer; defensive.
            raise ValueError(
                f"rank {r}: {len(srcs)} slots to move but only {len(targets)} "
                f"free class-{r} slots inside [0, {l_h})"
            )
        for src, tgt in zip(srcs, targets):
            mapping[src] = tgt

    moves = tuple(sorted((src, tgt) for src, tgt in mapping.items()))
    new_slots = {
        rid: tuple(
            (int(t) if int(t) < l_h else mapping[int(t)]) for t in slots
        )
        for rid, slots in held.items()
    }
    return CompactPlan(
        l_h=l_h, moves=moves, new_slots=new_slots, rows_by_rank=rows_by_rank
    )


def anchor_plan(
    anchor_slots: Mapping[str, int],
    reserved: Sequence[int] = (),
) -> Tuple[int, Tuple[Tuple[int, int], ...]]:
    """Squeeze the GDN state slots of the held requests to ``[0, A_H)``.

    ``A_H = len(anchor_slots) + len(reserved)``; slots already below ``A_H``
    stay, the others take the free targets of ``[0, A_H)`` in ascending order
    with the sources taken in ascending order (moves sorted by old slot).
    Returns ``(A_H, moves)``; the staying slots plus the targets are exactly
    ``range(A_H)`` minus ``reserved``. ``reserved`` (padding slot 0, the dummy
    write target for padded tokens) stays where it is and never becomes a
    target; an anchor already sitting on a reserved slot is a ValueError --
    the hold must never land there. The default ``()`` reproduces the
    pre-L15-11b behaviour exactly.
    """
    seen: Dict[int, str] = {}
    for rid, slot in anchor_slots.items():
        slot = int(slot)
        if slot < 0:
            raise ValueError(f"anchor slot of {rid!r} is negative: {slot}")
        if slot in seen:
            raise ValueError(
                f"anchor slot {slot} is held by both {seen[slot]!r} and {rid!r}"
            )
        seen[slot] = rid

    reserved_set = {int(x) for x in reserved}
    hit = sorted(reserved_set & set(seen))
    if hit:
        raise ValueError(
            f"anchor slot(s) {hit} sit on a reserved slot (padding slot 0 is "
            "the dummy write target for padded tokens); a held anchor must "
            "never land there"
        )
    a_h = len(anchor_slots) + len(reserved_set)
    staying = {slot for slot in seen if slot < a_h}
    free = [x for x in range(a_h) if x not in staying and x not in reserved_set]
    srcs = sorted(slot for slot in seen if slot >= a_h)
    # len(srcs) == a_h - len(staying) == len(free) holds by construction.
    moves = tuple(sorted(zip(srcs, free)))
    return a_h, moves
