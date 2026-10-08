"""L15-FIX-KEEP-SPLIT (N3t 02.10. 07:02:35Z): the L1.5 hold only survives the
kv pause on SPAN-MAPPED allocations.

N3t: every sampled held row read back as zeros after the wake
(L15-CHECK-DIAG first=(... live_absmax=0 l2_absmax=218)), and the kv pause of
a hold sleep freed the whole pool (TP1 nvml_proc 8714 -> 8856 MiB with 2.46 GB
"kept"). Root: the native pause (tms_csrc/core.cpp pass 3) applies the keep
set ONLY to allocations with span extents (``metadata.pdflip_extents``); a stock
allocation is unmapped and released whole, keep set or not -- and the 27B KV
and mamba pools are stock (PDFLIP-PAUSE-SUB extents=0). set_keep_byte_spans
returned 0 (the set is stored) while nothing was kept.

The fix splits each hold base, ONCE and while its content does not matter
(the plain flush, pools already reset), into extents: per view the HOLD
REGION ``[view_off, view_off + hold_rows * unit)`` rounded up to the granule
becomes its own extent(s), the rest of the allocation others
(tms_set_spans, now=True -- it maps fresh pages, the content is lost, which is
why it only runs when the pool is empty). The keep arm then keeps WHOLE hold
extents (an extent stays only if wholly inside a keep range), and refuses
when a base is not split or a requested range leaves the hold region -- a
refusal is the clean fallback, never the silent loss N3t showed.

Rank-agnostic (user order 07:05Z): the hold rows come from the rank's own
cap (cap 0 -> nothing to split), never from a card name or ordinal.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

#: base data_ptr -> sorted extents (lo, hi) of the split plan (bytes)
_PLANS: Dict[int, Tuple[Tuple[int, int], ...]] = {}
#: base data_ptr -> the hold extents (subset of the plan)
_HOLD: Dict[int, Tuple[Tuple[int, int], ...]] = {}


def _round_up(x: int, g: int) -> int:
    return -(-int(x) // int(g)) * int(g)


def _round_down(x: int, g: int) -> int:
    return (int(x) // int(g)) * int(g)


def hold_regions(views: Sequence[Tuple[int, int]], hold_rows: int,
                 granule: int, size: int) -> List[Tuple[int, int]]:
    """Granule-aligned, merged hold regions of one base: per view
    ``(view_off_bytes, unit_bytes)`` the bytes ``[off, off + rows*unit)``,
    widened outward to the granule and capped at ``size``."""
    out: List[Tuple[int, int]] = []
    for off, unit in sorted(views):
        if hold_rows <= 0 or unit <= 0:
            continue
        a = _round_down(int(off), granule)
        b = min(_round_up(int(off) + int(hold_rows) * int(unit), granule),
                _round_up(int(size), granule))
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def split_plan(holds: Sequence[Tuple[int, int]], size: int,
               granule: int) -> List[Tuple[int, int]]:
    """Cover ``[0, size_rounded)`` with the hold regions as their own extents
    and the gaps between them as the others; sorted, disjoint."""
    top = _round_up(int(size), granule)
    plan: List[Tuple[int, int]] = []
    cur = 0
    for a, b in holds:
        if a > cur:
            plan.append((cur, a))
        plan.append((a, b))
        cur = b
    if cur < top:
        plan.append((cur, top))
    return plan


def ensure_split(bases: Dict[int, Tuple[object, List[Tuple[int, int]]]],
                 hold_rows_of, granule_of, spans, sync, log) -> int:
    """Split every base not yet split. ``bases``: base_ptr -> (base_tensor,
    [(view_off, unit), ...]); ``hold_rows_of(base_ptr)`` the rows to hold;
    ``granule_of(base)``; ``spans`` a d_seat_vram.TmsSpans; ``sync(t)`` waits
    for kernels on t. Only call when the bases' content does not matter.
    Returns the number of bases split now."""
    if spans is None or not getattr(spans, "available", False):
        return 0
    n = 0
    for ptr, (base, views) in bases.items():
        if ptr in _PLANS:
            continue
        info = spans.info(ptr)
        if info is None or not info.active:
            continue
        g = int(granule_of(base))
        holds = hold_regions(views, int(hold_rows_of(ptr)), g, info.size)
        if not holds:
            continue
        plan = split_plan(holds, info.size, g)
        if len(plan) == 1 and int(plan[0][1]) - int(plan[0][0]) >= 2 * g:
            # L15-FIX-WHOLE-HOLD (N4f 11:09:36 TP1/TP2 keep-arm FAILED, N3y
            # 'export of extent @0 refused rc=-2'): when the hold regions
            # merge to the WHOLE allocation (e.g. the mamba conv base: every
            # layer's anchor rows widened to the granule), the one-range plan
            # IS the saver's stock mapping -- no span extent exists, the
            # pause unmaps the base whole whatever the keep set says, and
            # L15-EXTENTS (rightly) refuses the arm. Cut the region into two
            # span extents at a granule: both lie inside the hold region and
            # are kept.
            a0, b0 = int(plan[0][0]), int(plan[0][1])
            plan = [(a0, a0 + g), (a0 + g, b0)]
        if len(plan) <= 1:
            _PLANS[ptr] = tuple(plan)
            _HOLD[ptr] = tuple(holds)
            continue
        sync(base)
        rc = spans.set_spans(ptr, plan, now=True)
        if rc != 0:
            log("L15-KEEP-SPLIT base=%#x failed rc=%d (no hold on this base)"
                % (ptr, rc))
            continue
        _PLANS[ptr] = tuple(plan)
        _HOLD[ptr] = tuple(holds)
        n += 1
    if n:
        log("L15-KEEP-SPLIT split=%d bases (hold extents now survive the kv "
            "pause)" % n)
    return n


def keep_extents(base_ptr: int, ranges: Sequence[Tuple[int, int]],
                 native: Optional[Sequence[Tuple[int, int]]] = None
                 ) -> Optional[List[Tuple[int, int]]]:
    """The WHOLE hold extents covering ``ranges`` on a split base, or None
    when the base is not split or a range leaves the hold region (the arm
    must then refuse: a partial extent would be dropped by the pause).

    L15-FIX-CAP0-SPLIT (N3y 08:41:17Z): nothing to keep is always keepable --
    the cap-0 rank arms EMPTY windows (L15-FIX-CAP0-KEEP) on bases it never
    split; refusing them discarded its manifest, so it voted None and EVERY
    wake fell back (verdict=fallback on the whole group)."""
    if not any(int(hi) > int(lo) for lo, hi in ranges):
        return []
    holds = _HOLD.get(int(base_ptr))
    if not holds:
        return None
    out: List[Tuple[int, int]] = []
    for lo, hi in ranges:
        if hi <= lo:
            continue
        hit = next(((a, b) for a, b in holds if a <= lo and hi <= b), None)
        if hit is None:
            return None
        if hit not in out:
            out.append(hit)
    # L15-EXTENTS (N3y share rc=-2): trust the saver, not this module's
    # memory -- a hold region the saver no longer covers with span extents
    # (re-planned / re-allocated base) would be unmapped whole by the pause
    # while the manifest claims a hold. Refuse (clean fallback) instead.
    if native is None:
        from flliper.srt.pdflip.l15_hold_share import list_extents

        native = list_extents(int(base_ptr))
    if native is not None:
        from flliper.srt.pdflip.l15_hold_share import native_cover

        if native_cover(native, out) is None:
            return None
    return sorted(out)


def forget_all() -> None:
    """Tests only."""
    _PLANS.clear()
    _HOLD.clear()


def anchor_cap(env) -> int:
    """Max anchors one D sleep may hold (FLLIPER_PDFLIP_L15_ANCHOR_CAP, default
    8): bounds the mamba hold region the split reserves; select_hold admits
    at most this many anchors."""
    try:
        return max(0, int(str(env.get("FLLIPER_PDFLIP_L15_ANCHOR_CAP", "8")).strip()))
    except ValueError:
        return 8


def hold_bases(kv_buffers, mamba_buffers, kv_rows: int, mamba_rows: int):
    """(bases, rows) for ensure_split from the SAME buffer views the retain
    hook arms: KV views hold ``kv_rows``, mamba views ``mamba_rows``.
    Returns ({base_ptr: (base, [(view_off, unit), ...])}, {base_ptr: rows})."""
    bases: Dict[int, Tuple[object, List[Tuple[int, int]]]] = {}
    rows: Dict[int, int] = {}
    for bufs, n in ((kv_buffers, kv_rows), (mamba_buffers, mamba_rows)):
        for buf in bufs or ():
            b = buf._base if getattr(buf, "_base", None) is not None else buf
            key = int(b.data_ptr())
            off = int(buf.data_ptr()) - key
            unit = int(buf.stride(0)) * int(buf.element_size())
            bases.setdefault(key, (b, []))[1].append((off, unit))
            rows[key] = max(rows.get(key, 0), int(n))
    return bases, rows


def ensure_split_for_sched(sched, env, log) -> int:
    """Scheduler entry (the plain-flush branch: pools already reset, content
    irrelevant): split this rank's KV and mamba hold bases when the rank has a
    hold cap. Same buffer walk and cap source as the retain hook."""
    import torch

    from flliper.srt.pdflip import l15_shadow
    from flliper.srt.pdflip.d_seat_vram import _sync_before_unmap, granule_for, tms

    mr = getattr(getattr(sched, "tp_worker", None), "model_runner", None)
    pool = l15_shadow.kv_pool_of(getattr(mr, "token_to_kv_pool", None))
    kv = [t for t in l15_shadow.kv_buffers_of(pool) if isinstance(t, torch.Tensor)]
    if not kv:
        return 0
    mc = getattr(getattr(getattr(sched, "req_to_token_pool", None),
                         "mamba_pool", None), "mamba_cache", None)
    mb = []
    for c in getattr(mc, "conv", None) or []:
        mb.extend(c[i] for i in range(int(c.shape[0])))
    temp = getattr(mc, "temporal", None)
    if temp is not None:
        mb.extend(temp[i] for i in range(int(temp.shape[0])))
    tp = int(getattr(sched, "tp_size", 0)
             or getattr(getattr(sched, "server_args", None), "tp_size", 1) or 1)
    rank = int(getattr(getattr(sched, "ps", None), "tp_rank", 0) or 0)
    rgid = getattr(getattr(sched, "server_args", None), "rank_gpu_id", None)
    cards = (list(rgid) if isinstance(rgid, (list, tuple)) and len(rgid) == tp
             else list(range(tp)))
    caps = l15_shadow.caps_from_env(env, tp, [l15_shadow.cell_bytes_from(pool)] * tp,
                                    cards)
    kv_rows = int(caps[rank]) if rank < len(caps) else 0
    if kv_rows <= 0:
        return 0  # cap 0: this rank keeps nothing through the sleep
    bases, rows = hold_bases(kv, mb, kv_rows, anchor_cap(env) + 1)
    return ensure_split(bases, lambda p: rows.get(p, 0),
                        lambda b: granule_for(getattr(b, "device", "cuda")),
                        tms(), _sync_before_unmap, log)
