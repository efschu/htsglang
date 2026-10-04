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

import dataclasses

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
from sglang.srt.weg2 import l15_pool
from sglang.srt.weg2.l15_park import park_plan

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
    # L15-FLIPCOST (N4a: retain step "alloc" 284-319 ms): vectorised -- the
    # old form built a Python set over every free page (~1M) per sleep.
    free = allocator.free_pages
    want = (slots if torch.is_tensor(slots)
            else torch.as_tensor(list(slots), dtype=torch.int64)).to(free.dtype).reshape(-1)
    if want.numel() == 0:
        return 0
    uniq, counts = torch.unique(want, return_counts=True)
    if bool((counts > 1).any()):
        dupes = sorted(int(x) for x in uniq[counts > 1].tolist())
        raise ValueError(
            f"reserve_slots: slot(s) {dupes} requested twice; a slot can be "
            "reserved once"
        )
    uniq_d = uniq.to(free.device)
    present = torch.isin(uniq_d, free)
    if not bool(present.all()):
        missing = sorted(int(x) for x in uniq_d[~present].tolist())
        raise ValueError(
            f"reserve_slots: slot(s) {missing} are not free "
            f"(free_pages holds {int(free.numel())} entries, {int(uniq.numel())} "
            "requested)"
        )
    allocator.free_pages = free[~torch.isin(free, uniq_d)]
    return int(uniq.numel())


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
    # L15-MAMBA-LEDGER: the #924 ownership ledger must see the reserved
    # slots as USED. Carving them out of free_slots alone left slot_used
    # False, so the first legitimate release of a held/adopted anchor (node
    # eviction, request finish) was refused as a double free
    # (MambaSlotDoubleFree raises -> rank death).
    used = getattr(mamba_allocator, "slot_used", None)
    if used is not None:
        used[take.to(used.device)] = True
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


def _no_anchor_l2(rid: str) -> Tuple[int, int]:
    """Default anchor_l2_of: no anchor L2 identity recorded (-1, -1) --
    the pre-E2a behaviour for callers that do not supply the callable."""
    return (-1, -1)


def _no_l2_lanes(rid: str) -> Tuple[int, ...]:
    """Default l2_lanes_of: no lanes recorded (()) -- the pre-P1
    behaviour for callers that do not supply the callable."""
    return ()


@dataclass
class RoundPlan:
    """Steps (1)-(2) of one retain round: pure planning, nothing touched.

    L15-SLEEP-DECIDE-FIRST: the scheduler computes this BEFORE the group's
    cap-0 vote, so a round the group refuses is never paid (the moves, tree
    rewrite, host pins, reset and arm). ``manifest`` is filled lazily by
    :func:`manifest_of_plan` (identical to what step (8) publishes)."""

    hs: HoldSet
    plan: CompactPlan
    a_h: int
    anchor_moves: list
    new_anchors: Dict
    manifest: Optional[Manifest] = None
    # L15-POOL S2 (None = the per-card path): the guest pieces the cap-0
    # ranks' compacted rows park in (``l15_park.ParkPiece``) and the digest
    # of the whole pool decision the group compares (``agree_pool``)
    guest_pieces: Optional[tuple] = None
    pool_fp: Optional[str] = None
    # L15-POOL S3: the plan was made with the overflow of EVERY rank; the
    # planned caps ride the manifest v2 (None = S2 / per-card path)
    s3: bool = False
    caps: Optional[tuple] = None


def plan_round(
    *,
    candidates: Iterable,
    slots_of: Callable[[str], Sequence[int]],
    anchor_slot_of: Callable[[str], int],
    caps_rows_by_rank: Sequence[int],
    cap_anchor_slots: int,
    prefix: Sequence[int],
    epoch: int,
    log: Callable[[str], None],
    pool: Optional[bool] = None,
    s3: Optional[bool] = None,
    rates: Optional[dict] = None,
    rates_src: Optional[str] = None,
) -> Optional["RoundPlan"]:
    """Steps (1)-(2) of :func:`retain_at_sleep` (select_hold, compact_plan,
    anchor_plan). None = benign skip, nothing touched (logged as before).

    ``pool`` (None = ``SGLANG_WEG2_L15_POOL`` from the environment): the S2
    pooled hold -- the cap-0 ranks' shards need guest room in the capped
    ranks' free hold rows. The room is checked HERE on the COMPACTED rows
    (``park_plan``) before anything moves: a request without room is dropped,
    lowest priority first (reason ``pool_full``), never held half and never
    paid for in a retain the park would refuse afterwards. False = today's
    path, byte for byte.

    ``s3`` (None = ``SGLANG_WEG2_L15_POOL_S3`` with the pool on): EVERY rank
    may overflow -- the admission runs against the SUM of the segments
    (``select_hold_pool_s3``), a capped rank's rows beyond its hold region are
    guest rows too, the placement is ``l15_pool.pool_park_plan`` (Q3 rule with
    the measured pair ``rates``; ``rates=None`` = resolved from the env / the
    barlink matrix, ``rates_src`` names where they came from) and
    the digest every rank compares carries the rates. A capped rank never
    gets ``keep_over_cap``-trimmed in this mode: its overflow is a guest."""
    import os as _os

    candidates = list(candidates)
    if pool is None:
        pool = l15_pool.pool_on(_os.environ)
    if s3 is None:
        s3 = l15_pool.pool_s3_on(_os.environ)
    s3 = bool(s3) and bool(pool)
    if s3 and rates is None:
        rates, rates_src = l15_pool.resolve_rates(_os.environ, len(caps_rows_by_rank))
    rates = l15_pool.quantize_rates(rates) if s3 else None
    rates_src = (rates_src or "none") if s3 else None
    guest_pieces = None
    pool_fp = None
    # (1) who stays
    try:
        hs = (l15_pool.select_hold_pool_s3 if s3 else
              l15_pool.select_hold_pool if pool else select_hold)(
            candidates, caps_rows_by_rank, cap_anchor_slots)
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
        # L15-FIX-KEEP-OVER-CAP (N3y 08:41:52Z TP2: keep-arm FAILED
        # not-split-or-outside-hold with keep_rows 68820 > cap 57344): the
        # compacted keep capacity is a WHOLE number of blocks on every rank
        # (rows = blocks * ratio_r, L_H driven by the rank with the largest
        # need/ratio), so a set select_hold admitted by summed rows can still
        # overrun a capped rank's hold region. Drop the lowest-priority held
        # request until every capped rank's keep fits its cap.
        trimmed = []
        _rows0 = tuple(int(x) for x in plan.rows_by_rank)
        def _home_over(pl):
            if s3:
                return False  # S3: a capped rank's overflow is a guest, not a trim
            return any(
                int(caps_rows_by_rank[r]) > 0
                and int(pl.rows_by_rank[r]) > int(caps_rows_by_rank[r])
                for r in range(len(pl.rows_by_rank)))

        def _guest_refusal(pl):
            # S2: every cap-0 rank's compacted rows must find guest room in
            # the capped ranks' free hold rows (the SAME function the park
            # runs at the release, so plan and park cannot disagree)
            if not pool:
                return None
            if s3:
                return l15_pool.pool_park_plan(
                    list(pl.rows_by_rank), [int(c) for c in caps_rows_by_rank],
                    rates)[1]
            return park_plan(list(pl.rows_by_rank),
                             [int(c) for c in caps_rows_by_rank])[1]

        _reasons = []
        while hs.rids and (_home_over(plan) or _guest_refusal(plan) is not None):
            _why = "keep_over_cap" if _home_over(plan) else l15_pool.REASON_POOL_FULL
            _reasons.append(_why)
            trimmed.append(hs.rids[-1])
            keep_rids = hs.rids[:-1]
            _rows = {c.rid: c.rows_by_rank for c in candidates}
            hs = dataclasses.replace(
                hs, rids=keep_rids, anchors=len(keep_rids),
                rows_by_rank=tuple(
                    sum(int(_rows[x][r]) for x in keep_rids)
                    for r in range(len(hs.rows_by_rank))),
                excluded=tuple(hs.excluded) + ((trimmed[-1], _why),))
            if not keep_rids:
                break
            plan = compact_plan({rid: slots_of(rid) for rid in hs.rids},
                                prefix, reserved=PAD_SLOTS)
        if trimmed:
            log(f"L15-RETAIN trimmed n={len(trimmed)} rids={trimmed} "
                f"keep_rows={_rows0} caps={tuple(int(c) for c in caps_rows_by_rank)} "
                f"l_h={plan.l_h} "
                + ("(compacted keep rows over a capped rank's hold region)"
                   if not pool else
                   "(compacted keep rows over a capped rank's hold region or "
                   f"without guest room for the cap-0 rows; reasons={_reasons})"))
        if not hs.rids:
            log(f"L15-RETAIN epoch={epoch} n=0 nothing_to_hold (all trimmed)")
            return None
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
    if pool:
        # the final plan's guest placement: the loop above ended only when
        # park_plan accepted it, so a refusal here is impossible -- if it
        # happens anyway the round is not held (never "held without room")
        if s3:
            guest_pieces, _gwhy = l15_pool.pool_park_plan(
                list(plan.rows_by_rank), [int(c) for c in caps_rows_by_rank], rates)
        else:
            guest_pieces, _gwhy = park_plan(list(plan.rows_by_rank),
                                            [int(c) for c in caps_rows_by_rank])
        if _gwhy is not None:
            log(f"L15-RETAIN skipped reason=pool-no-guest-room: {_gwhy}")
            return None
        guest_pieces = tuple(guest_pieces)
        pool_fp = l15_pool.plan_fingerprint(
            hs.rids, plan.rows_by_rank, caps_rows_by_rank, guest_pieces)
        if s3:
            log(l15_pool.plan_line_s3(
                epoch, hs, caps_rows_by_rank, plan.rows_by_rank, guest_pieces,
                pool_fp, len(candidates), rates_src, l15_pool.rates_digest(rates)))
        else:
            log(l15_pool.plan_line(epoch, hs, caps_rows_by_rank, plan.rows_by_rank,
                                   guest_pieces, pool_fp, len(candidates)))
    return RoundPlan(hs=hs, plan=plan, a_h=a_h, anchor_moves=anchor_moves,
                     new_anchors=new_anchors, guest_pieces=guest_pieces,
                     pool_fp=pool_fp, s3=s3,
                     caps=(tuple(int(c) for c in caps_rows_by_rank) if s3 else None))


def manifest_of_plan(
    rp: "RoundPlan",
    *,
    candidates: Iterable,
    l2_of: Callable[[str], Tuple[Sequence[int], Sequence[int]]],
    anchor_l2_of: Callable[[str], Tuple[int, int]],
    l2_lanes_of: Callable[[str], Tuple[int, ...]],
    epoch: int,
    pid: int,
) -> Manifest:
    """The manifest step (8) publishes, built from the plan alone (all of it
    is known after steps 1-2: the new slots, the bind-time L2 snapshot)."""
    cand_depth = {c.rid: int(c.kv_depth) for c in candidates}
    spans = tuple(
        HoldSpan(
            rid=rid,
            depth=cand_depth[rid],
            slots=tuple(int(s) for s in rp.plan.new_slots[rid]),
            anchor_slot=rp.new_anchors[rid],
            l2_slots=tuple(int(x) for x in l2_of(rid)[0]),
            l2_gens=tuple(int(x) for x in l2_of(rid)[1]),
            # L15-12c-E2a: the anchor's L2 identity for the cap-0 wake
            anchor_l2_slot=int(anchor_l2_of(rid)[0]),
            anchor_l2_gen=int(anchor_l2_of(rid)[1]),
            # L15-12c-P1: the lane each held token owns in its L2 page
            l2_lanes=tuple(int(x) for x in l2_lanes_of(rid)),
        )
        for rid in rp.hs.rids
    )
    return Manifest(
        epoch=int(epoch),
        pid=int(pid),
        spans=spans,
        rows_by_rank=tuple(int(x) for x in rp.plan.rows_by_rank),
        anchor_slots=int(rp.a_h),
        # MANIFEST v2 (S3 only): the guest placement and the planned caps ride
        # the record and its fingerprint; a v1 round writes the old record
        guests=(l15_pool.guest_tuples(rp.guest_pieces or ()) if rp.s3 else None),
        caps=(tuple(rp.caps or ()) if rp.s3 else None),
    )


def retain_at_sleep(
    *,
    candidates: Iterable,
    node_of: Callable[[str], object],
    slots_of: Callable[[str], Sequence[int]],
    anchor_slot_of: Callable[[str], int],
    l2_of: Callable[[str], Tuple[Sequence[int], Sequence[int]]],
    anchor_l2_of: Callable[[str], Tuple[int, int]] = _no_anchor_l2,
    l2_lanes_of: Callable[[str], Tuple[int, ...]] = _no_l2_lanes,
    rewrite_tree: Callable[
        [object, Dict[int, int], Dict[int, int], set], None
    ],
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
    hold_l2_refs: Optional[
        Callable[[Sequence[int], Sequence[int]], None]] = None,
    reset_keep: Callable[[list], None],
    set_keep: Callable[[object, Tuple[Tuple[int, int], ...]], None],
    manifest_path: str,
    log: Callable[[str], None],
    planned: Optional["RoundPlan"] = None,
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

    ``rewrite_tree(node, kv_map, anchor_map, visited)`` is the injected
    rewrite of the REAL tree (L15-11d): the old step (4) wrote
    kv_slots/anchor_slot attributes that only the unit-test fakes have;
    the real UnifiedTreeNode carries the KV indices in
    component_data[FULL].value of every chain node and the anchor slot in
    the mamba value, so the remap has to touch those. ``visited`` is a
    set shared across the held requests of this call so a shared prefix
    chain is remapped exactly once (id(node) -- UnifiedTreeNode.id is an
    int counter, not the identity).
    """
    # Materialise once: select_hold (step 1) and the cand_depth map (step 8)
    # both walk ``candidates``. A generator argument would be exhausted after
    # step 1, and step 8 would then KeyError AFTER the buffers already moved --
    # a half state that must re-raise, not masquerade as a benign skip.
    candidates = list(candidates)
    # L15-FLIPCOST: per-step wall clock, printed on the L15-RETAIN line
    import time as _time
    _t = [_time.perf_counter()]
    _ms: list = []

    def _lap(name: str) -> None:
        now = _time.perf_counter()
        _ms.append("%s:%.0f" % (name, (now - _t[0]) * 1000.0))
        _t[0] = now

    _t0 = _t[0]

    # (1)-(2) who stays, where the survivors land (pure; nothing touched)
    rp = planned if planned is not None else plan_round(
        candidates=candidates, slots_of=slots_of, anchor_slot_of=anchor_slot_of,
        caps_rows_by_rank=caps_rows_by_rank, cap_anchor_slots=cap_anchor_slots,
        prefix=prefix, epoch=epoch, log=log)
    if rp is None:
        return None
    hs, plan, a_h = rp.hs, rp.plan, rp.a_h
    anchor_moves, new_anchors = rp.anchor_moves, rp.new_anchors

    _lap("plan")
    # (3) rows land in the compact buffers: kv rows of THIS rank, every rank
    # applies the anchor moves (mamba slots are not owner-sharded)
    apply_moves(kv_buffers, plan.moves, _owner_rows(prefix, rank))
    apply_moves(mamba_buffers, anchor_moves, lambda slot: int(slot))

    _lap("moves")
    # (4) the tree nodes follow the plan. The injected rewrite_tree remaps
    # the REAL UnifiedTreeNode component values (L15-11d): the old code
    # wrote kv_slots/anchor_slot attributes that only the unit-test fakes
    # have -- on the real tree the write was silently accepted and never
    # read, and the next prefix hit would have picked up foreign KV.
    # kv_map/anchor_map carry GLOBAL old->new slots; ``visited`` is shared
    # across the held requests so a shared prefix chain is remapped once.
    kv_map = {int(old): int(new) for old, new in plan.moves}
    try:  # L15-CHECK-SNAP: the wake diag names a bad row's pre-move slot
        from sglang.srt.weg2 import l15_check_snap as _l15_cs
        _l15_cs.note_moves(plan.moves)
    except Exception:  # noqa: BLE001 -- diagnostics only
        pass
    # every HELD anchor old->new, identity for one that did not move: the
    # rewrite drops any chain mamba value outside this map (only these
    # slots survive the mamba allocator re-arm in step 6)
    hold_anchor_map = {int(anchor_slot_of(rid)): int(new_anchors[rid])
                       for rid in hs.rids}
    visited: set = set()
    seen_last: set = set()
    nodes = []
    for rid in hs.rids:
        node = node_of(rid)
        nodes.append(node)
        if id(node) not in seen_last:
            seen_last.add(id(node))
            rewrite_tree(node, kv_map, hold_anchor_map, visited)

    # L15-HOSTLOCK (LCHOST defect 2): BEFORE reset_keep (whose _reset_full
    # hands every kept chain's arena references back -- kept chains never
    # carry host_lock_ref) pin this rank's held L2 slots once, in this
    # rank's own host pools: a rank's l2_slots are its OWN shard's arena
    # slots, so cap-0 ranks take refs too -- the rank(s) whose cap is 0 are
    # exactly the ones that refill from L2 at the wake. The wake act gives them back (hold: after
    # the refill copied; fallback: in the drop). Master off: the scheduler
    # hook passes no callable, no reference is taken (byte-identical). The
    # slot list is the bind-time l2_of/anchor_l2_of snapshot -- the same
    # values step (8) publishes into the manifest.
    _lap("rewrite")
    if hold_l2_refs is not None:
        kv_l2: list = []
        anchor_l2: list = []
        for rid in hs.rids:
            kv_l2.extend(int(x) for x in l2_of(rid)[0])
            anchor_l2.append(int(anchor_l2_of(rid)[0]))
        # L15-HOSTLOCK-COVER (N5n: slots=2304 pinned for 133k held tokens --
        # the chain was barely written back to L2): held tokens against the
        # tokens with an L2 source, per rank, beside the pin line.
        _backed = sum(1 for x in kv_l2 if int(x) >= 0)
        log(f"L15-HOSTLOCK-COVER epoch={epoch} tokens={len(kv_l2)} "
            f"backed={_backed} unbacked={len(kv_l2) - _backed} "
            f"distinct_slots={len({int(x) for x in kv_l2 if int(x) >= 0})} "
            f"rids={len(hs.rids)}")
        hold_l2_refs(kv_l2, anchor_l2)

    _lap("hostlock")
    # (5) partial tree reset over exactly the kept nodes
    reset_keep(nodes)
    _lap("reset")

    # (6) re-arm the allocator, then take back the held prefix slots; the
    # padding slot is never free after clear() and is not a held slot, so it
    # never goes to reserve_slots. A given mamba_allocator is re-armed the
    # same way for the held anchors' compacted slots.
    allocator.clear()
    import numpy as _np

    _cat = _np.concatenate([_np.asarray(plan.new_slots[rid], dtype=_np.int64)
                            for rid in hs.rids]) if hs.rids else _np.zeros(0, _np.int64)
    _u = _np.unique(_cat)
    new_all = _u[~_np.isin(_u, _np.asarray(sorted(PAD_SLOTS), dtype=_np.int64))].tolist()
    reserve_slots(allocator, new_all)
    if mamba_allocator is not None:
        mamba_allocator.clear()
        reserve_mamba_slots(
            mamba_allocator, sorted({int(a) for a in new_anchors.values()})
        )

    _lap("alloc")
    # (7) keep windows: kv rows [0, rows_by_rank[rank]), mamba rows [0, A_H).
    # L15-FIX-CAP0-KEEP (N3l 02:29:56Z: the cap-0 rank kept 555 MB): a rank
    # whose cap is 0 (any of them, if several) is "not held here" -- select_hold does
    # not charge it, its rows come back from L2 at the wake (refill), so it
    # must NOT pin VRAM through the P phase, which never budgeted it. Its
    # keep windows are EMPTY; the rows are still compacted and the manifest
    # still published (the wake refill and the group vote need both).
    cap0 = int(caps_rows_by_rank[rank]) == 0
    _keep_hi = int(plan.rows_by_rank[rank])
    if rp.s3 and not cap0:
        # S3: a capped rank's overflow is no home row (its keep window is its
        # hold region at most); a rank that HOSTS guests keeps its whole hold
        # region (the guests lie in the rows after its own)
        _keep_hi = (int(caps_rows_by_rank[rank])
                    if any(int(p.dst) == int(rank) for p in (rp.guest_pieces or ()))
                    else min(_keep_hi, int(caps_rows_by_rank[rank])))
    kv_range = () if cap0 else ((0, _keep_hi),)
    for buf in kv_buffers:
        set_keep(buf, kv_range)
    mamba_range = () if cap0 else ((0, int(a_h)),)
    for buf in mamba_buffers:
        set_keep(buf, mamba_range)

    _lap("keep")
    # (8) publish the hold for the group agreement
    manifest = rp.manifest if rp.manifest is not None else manifest_of_plan(
        rp, candidates=candidates, l2_of=l2_of, anchor_l2_of=anchor_l2_of,
        l2_lanes_of=l2_lanes_of, epoch=epoch, pid=pid)
    manifest_write(manifest_path, manifest)

    # (9) the one line
    fp = fingerprint(manifest)
    _lap("manifest")
    rows_fmt = ",".join(str(x) for x in manifest.rows_by_rank)
    # keep_rows_by_rank: this is the manifest's per-rank KEEP capacity
    # (plan.rows_by_rank = blocks*ratio_r), not the admitted rows of the
    # HoldSet -- printed under its own key so the two never clash in the log.
    log(
        f"L15-RETAIN epoch={epoch} n={len(hs.rids)} "
        f"keep_rows_by_rank={rows_fmt} "
        f"l_h={plan.l_h} anchors={a_h} fp={fp} "
        f"ms={(_time.perf_counter() - _t0) * 1000.0:.0f} steps={','.join(_ms)}"
    )
    return RetainResult(
        hold=hs, plan=plan, a_h=a_h, manifest=manifest, keep_nodes=nodes
    )
