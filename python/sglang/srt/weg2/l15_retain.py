# SPDX-License-Identifier: Apache-2.0
"""L1.5 retain orchestrator at the D sleep (AP L15-11a, pure + injectable I/O).

When a D rank sleeps it keeps a hold set resident instead of dropping the
tree: the held KV compacts into the prefix ``[0, L_H)`` (``l15_compact``),
the anchors into ``[0, A_H)``, the tree nodes are rewritten to the new slots,
the allocator is told which slots stay taken, and the TMS keep windows plus a
manifest publish the result. The scheduler owns WHEN this runs; the real
device bindings (node attributes, ``reset_keep``, ``set_keep``) are injected
callables -- the one-line scheduler hook is AP L15-11b.
OPEN for L15-11b (not built here): HybridReqToTokenPool.clear()
(mem_cache/memory_pool.py ~2871) calls mamba_pool.reset_state(), which zeroes
EVERY mamba slot including the held anchors -- the real sleep path must skip
reset_state for ``[0, A_H)`` when a hold exists.
OPEN for L15-11b (pool-side keep_rows is in): the caller of
retain_at_sleep must flush via HybridReqToTokenPool.clear(keep_mamba_rows=A_H) so the held rows [0, A_H) survive the mamba reset (the scheduler hook making that call is still open).
OPEN for L15-11b (not here): the scheduler hook with the actual
node_of/reset_keep/set_keep bindings.

The DANGER DIRECTION is a wrong step order: moving buffers after the
allocator/tracing was re-armed, or resetting the keep set before the rows
physically landed, silently corrupts the held KV. The order is therefore
fixed here and pinned by the test:

    (1) select_hold          -- who stays (l15_policy)
    (2) compact_plan + anchor_plan
    (3) apply_moves          -- rows land in the compact buffer, this rank
    (4) rewrite kept nodes   -- device values := plan.new_slots
    (5) reset_keep(nodes)    -- partial tree reset over the kept nodes
    (6) allocator.clear(); reserve_slots(new slots)
    (7) set_keep per buffer  -- kv rows [0, rows_by_rank[rank]), mamba [0, A_H)
    (8) write the manifest
    (9) one "L15-RETAIN" log line, return the RetainResult

Steps 1-2 failing is a benign skip: the caller runs today's full reset
(returns None, logs "L15-RETAIN skipped reason=..."). From step 3 on the
buffers are already half-moved, so an exception MUST re-raise -- a half
state hidden behind a None would let the group agree on memory that is not
there (memory rules: raenge-nie-uneins, no silent half states).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from sglang.srt.weg2.l15_compact import CompactPlan, anchor_plan, compact_plan
from sglang.srt.weg2.l15_manifest import (
    HoldSpan,
    Manifest,
    fingerprint,
)
from sglang.srt.weg2.l15_manifest import write as manifest_write
from sglang.srt.weg2.l15_policy import HoldSet, select_hold

__all__ = [
    "PAD_SLOTS",
    "RetainResult",
    "apply_moves",
    "reserve_mamba_slots",
    "reserve_slots",
    "retain_at_sleep",
]

# Slot 0 (KV and mamba alike) is the dummy write target for padded tokens:
# both clear() sites hand out arange(1, size+1) -- mamba.py
# MambaSlotAllocator.clear() and token.py TokenToKVPoolAllocator.clear().
# A real hold must never land on it; it stays inside the keep window harmlessly.
PAD_SLOTS = (0,)


def reserve_slots(allocator, slots: Sequence[int]) -> int:
    """Carve ``slots`` out of a freshly cleared allocator's free_pages.

    ``allocator.clear()`` (see ``mem_cache/allocator/token.py``) resets
    ``free_pages`` to ``arange(1, size + 1)`` and empties ``release_pages``;
    this then removes exactly the requested slots, keeping the order of the
    remaining entries, and returns how many were removed. A slot that is not
    free (busy, or the padded slot 0, or requested twice) is a ValueError --
    reserving around it would double-book memory against the hold.
    """
    wanted = [int(s) for s in slots]
    if not wanted:
        return 0
    # Counter keeps the dupe scan linear: a real hold is tens of thousands of
    # slots and this runs inside D's sleep, where wanted.count(s) per distinct
    # slot would be O(n^2) (~4e9 ops for a single 64k-token request).
    counts = Counter(wanted)
    dupes = sorted(s for s, c in counts.items() if c > 1)
    if dupes:
        raise ValueError(
            f"reserve_slots: slot(s) {dupes} requested twice; a slot can be "
            "reserved once"
        )
    distinct = sorted(counts)
    free = allocator.free_pages
    free_set = set(int(x) for x in free.tolist())
    missing = [s for s in distinct if s not in free_set]
    if missing:
        raise ValueError(
            f"reserve_slots: slot(s) {missing} are not free "
            f"(free_pages holds {len(free_set)} entries, {len(distinct)} "
            "requested)"
        )
    take = torch.tensor(distinct, dtype=free.dtype, device=free.device)
    allocator.free_pages = free[~torch.isin(free, take)]
    return len(distinct)


def reserve_mamba_slots(mamba_allocator, slots: Sequence[int]) -> int:
    """Carve ``slots`` out of a freshly cleared mamba allocator's free_slots.

    The caller runs ``mamba_allocator.clear()`` first (free_slots becomes
    ``arange(1, size + 1)``, int64, padding slot 0 never in it); this then
    removes exactly the requested slots -- the held anchors' compacted slots
    -- keeping the order of the remaining entries, and returns how many were
    removed. A slot that is not free (already taken, or padding slot 0, or
    requested twice) is a ValueError: handing it out again would overwrite
    the held GDN state.
    """
    wanted = [int(s) for s in slots]
    if not wanted:
        return 0
    counts = Counter(wanted)
    dupes = sorted(s for s, c in counts.items() if c > 1)
    if dupes:
        raise ValueError(
            f"reserve_mamba_slots: slot(s) {dupes} requested twice; a slot "
            "can be reserved once"
        )
    distinct = sorted(counts)
    free = mamba_allocator.free_slots
    free_set = set(int(x) for x in free.tolist())
    missing = [s for s in distinct if s not in free_set]
    if missing:
        raise ValueError(
            f"reserve_mamba_slots: slot(s) {missing} are not free "
            f"(free_slots holds {len(free_set)} entries, {len(distinct)} "
            "requested)"
        )
    take = torch.tensor(distinct, dtype=torch.int64, device=free.device)
    mamba_allocator.free_slots = free[~torch.isin(free, take)]
    return len(distinct)


def apply_moves(buffers, moves: Sequence[Tuple[int, int]], owner_rows) -> None:
    """Copy moved rows ``old -> new`` in every per-layer buffer, this rank only.

    ``moves`` carry GLOBAL slots (``l15_compact``); ``owner_rows(slot)`` maps
    a global slot to the compact row on THIS rank, or None when another rank
    owns it (both ends of one move always share an owner: compaction keeps
    every slot in its class). Sources and targets are disjoint by
    construction, so a gather-then-scatter is safe; the clone guards against
    any index overlap a caller-side bug might introduce.
    """
    pairs: List[Tuple[int, int]] = []
    for old_slot, new_slot in moves:
        src = owner_rows(int(old_slot))
        dst = owner_rows(int(new_slot))
        if src is None or dst is None:
            continue
        pairs.append((src, dst))
    if not pairs:
        return
    old_idx = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    new_idx = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    for buf in buffers:
        buf[new_idx.to(buf.device)] = buf[old_idx.to(buf.device)].clone()


@dataclass
class RetainResult:
    """Everything one retain round decided, for the caller and the tests."""

    hold: HoldSet
    plan: CompactPlan
    a_h: int
    manifest: Manifest
    keep_nodes: list


def _owner_rows(prefix: Sequence[int], rank: int) -> Callable[[int], Optional[int]]:
    """Global slot -> compact row on ``rank``; None when another rank owns it.

    Mirrors the l15_compact owner rule: rank ``r`` owns slot ``L`` iff
    ``prefix[r] <= L % S < prefix[r+1]`` (S = prefix[-1]), and the compact
    row inside [0, L_H) is ``(L // S) * ratio_r + (L % S - prefix[r])``.
    """
    s = int(prefix[-1])
    lo = int(prefix[rank])
    hi = int(prefix[rank + 1])
    ratio = hi - lo

    def owner_rows(slot: int) -> Optional[int]:
        residue = int(slot) % s
        if not (lo <= residue < hi):
            return None
        return (int(slot) // s) * ratio + (residue - lo)

    return owner_rows


def retain_at_sleep(
    *,
    candidates: Iterable,
    node_of: Callable[[str], object],
    slots_of: Callable[[str], Sequence[int]],
    anchor_slot_of: Callable[[str], int],
    l2_of: Callable[[str], Tuple[Sequence[int], Sequence[int]]],
    caps_rows_by_rank: Sequence[int],
    cap_anchor_slots: int,
    prefix: Sequence[int],
    rank: int,
    epoch: int,
    pid: int,
    kv_buffers,
    mamba_buffers,
    allocator,
    mamba_allocator=None,
    reset_keep: Callable[[list], None],
    set_keep: Callable[[object, Tuple[Tuple[int, int], ...]], None],
    manifest_path: str,
    log: Callable[[str], None],
) -> Optional[RetainResult]:
    """Run the whole L1.5 retain round at D's sleep; None means "skip it".

    None is returned ONLY before anything was touched: an empty hold set
    (the caller runs today's reset) or a step 1-2 planning failure (logged as
    "L15-RETAIN skipped reason=..."). From step 3 on, exceptions re-raise --
    the buffers are half-moved by then and the group must not be told this
    rank holds memory it does not.

    ``node_of``/``reset_keep``/``set_keep`` are the injectable bindings to
    the tree cache and the TMS (L15-03/L15-06); the real binding is the
    L15-11b scheduler hook. ``l2_of(rid)`` yields the suffix's
    ``(l2_slots, l2_gens)`` for the manifest.
    """
    # Materialise once: select_hold (step 1) and the cand_depth map (step 8)
    # both walk ``candidates``. A generator argument would be exhausted after
    # step 1, and step 8 would then KeyError AFTER the buffers already moved --
    # a half state that must re-raise, not masquerade as a benign skip.
    candidates = list(candidates)

    # (1) who stays
    try:
        hs = select_hold(candidates, caps_rows_by_rank, cap_anchor_slots)
        if not hs.rids:
            log(f"L15-RETAIN epoch={epoch} n=0 nothing_to_hold")
            return None
        # (2) where the survivors land; PAD_SLOTS (padding slot 0, the dummy
        # write target for padded tokens) is never a compaction target.
        plan = compact_plan(
            {rid: slots_of(rid) for rid in hs.rids},
            prefix,
            reserved=PAD_SLOTS,
        )
        a_h, anchor_moves = anchor_plan(
            {rid: anchor_slot_of(rid) for rid in hs.rids},
            reserved=PAD_SLOTS,
        )
    except Exception as exc:  # noqa: BLE001 - benign skip, nothing touched yet
        log(f"L15-RETAIN skipped reason={type(exc).__name__}: {exc}")
        return None

    anchor_map = {int(old): int(new) for old, new in anchor_moves}
    new_anchors = {
        rid: anchor_map.get(int(anchor_slot_of(rid)), int(anchor_slot_of(rid)))
        for rid in hs.rids
    }

    # (3) rows land in the compact buffers: kv rows of THIS rank, every rank
    # applies the anchor moves (mamba slots are not owner-sharded)
    apply_moves(kv_buffers, plan.moves, _owner_rows(prefix, rank))
    apply_moves(mamba_buffers, anchor_moves, lambda slot: int(slot))

    # (4) the tree nodes follow the plan
    nodes = []
    for rid in hs.rids:
        node = node_of(rid)
        node.kv_slots = tuple(plan.new_slots[rid])
        node.anchor_slot = new_anchors[rid]
        nodes.append(node)

    # (5) partial tree reset over exactly the kept nodes
    reset_keep(nodes)

    # (6) re-arm the allocator, then take back the held prefix slots; the
    # padding slot is never free after clear() and is not a held slot, so it
    # never goes to reserve_slots. A given mamba_allocator is re-armed the
    # same way for the held anchors' compacted slots.
    allocator.clear()
    new_all = sorted(
        {int(s) for rid in hs.rids for s in plan.new_slots[rid]}
        - set(PAD_SLOTS)
    )
    reserve_slots(allocator, new_all)
    if mamba_allocator is not None:
        mamba_allocator.clear()
        reserve_mamba_slots(
            mamba_allocator, sorted({int(a) for a in new_anchors.values()})
        )

    # (7) keep windows: kv rows [0, rows_by_rank[rank]), mamba rows [0, A_H)
    kv_range = ((0, int(plan.rows_by_rank[rank])),)
    for buf in kv_buffers:
        set_keep(buf, kv_range)
    mamba_range = ((0, int(a_h)),)
    for buf in mamba_buffers:
        set_keep(buf, mamba_range)

    # (8) publish the hold for the group agreement
    cand_depth = {c.rid: int(c.kv_depth) for c in candidates}
    spans = tuple(
        HoldSpan(
            rid=rid,
            depth=cand_depth[rid],
            slots=tuple(int(s) for s in plan.new_slots[rid]),
            anchor_slot=new_anchors[rid],
            l2_slots=tuple(int(x) for x in l2_of(rid)[0]),
            l2_gens=tuple(int(x) for x in l2_of(rid)[1]),
        )
        for rid in hs.rids
    )
    manifest = Manifest(
        epoch=int(epoch),
        pid=int(pid),
        spans=spans,
        rows_by_rank=tuple(int(x) for x in plan.rows_by_rank),
        anchor_slots=int(a_h),
    )
    manifest_write(manifest_path, manifest)

    # (9) the one line
    fp = fingerprint(manifest)
    rows_fmt = ",".join(str(x) for x in manifest.rows_by_rank)
    log(
        f"L15-RETAIN epoch={epoch} n={len(hs.rids)} rows_by_rank={rows_fmt} "
        f"l_h={plan.l_h} anchors={a_h} fp={fp}"
    )
    return RetainResult(
        hold=hs, plan=plan, a_h=a_h, manifest=manifest, keep_nodes=nodes
    )
