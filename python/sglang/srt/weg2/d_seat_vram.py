"""H95c (Nutzer 26.09.): D's per-seat posts are backed only for the seats the
phase occupies; the same VRAM backs extra expert-LRU rows otherwise.

  "soll ja dynamisches bs geben, 1,6gb experten cache kostet es nur bei
  tatsaechlich 6 sitzen. bei weniger sitzen kostet es weniger experten cache"

H95 B decides the phase's seat count n per P->D flip (``d_seats.phase_seats``
of the wake's ``handoff_n``/``parked_n``), but the posts stayed allocated for
the --d-bs cap: 38 GDN slots x 56.2 MiB on the attention host (Form A TP0) and
the expert bank booked at the cap's row count. This module turns n into VRAM.

THE PHYSICS (why it is a span map and not a re-allocation)
----------------------------------------------------------
Every captured decode graph holds the VIRTUAL addresses of the Mamba/GDN state
pool and of each expert tensor. So both keep their full virtual range at the
cap; what changes per phase is which PAGES are behind it:

* the GDN temporal state ``[L, S+1, HV, V, K]`` (TP0, kv_cache tag): per layer
  the slots ``[0, L(n)]`` stay mapped, the tail of every layer's slot block is
  unmapped (``slot_spans``; one layer's block is 39 x 1.5 MiB, the granule is
  2 MiB, so the tail rounds INWARD per layer -- a partly live granule is never
  released);
* each expert tensor of the bank ``[R+C+X, ...]`` (TP0, its weights chunk tag):
  the prefix ``[0, R+C+k(n))`` is mapped (``row_spans``); with k = 0 that is
  exactly the rows H95 B mapped.

Both are torch_memory_saver allocations, and D's memory is released at EVERY
D sleep and re-mapped at every D wake anyway (the flip). The saver's span map
(tms_csrc patch 3, ``tms_set_spans``) makes that re-map follow the phase: the
Mamba pool's plan is set BEFORE its tag resumes; the expert tensors resume in
the cap form and GROW in place to k(n) rows once n is known (``now``: the
mapped extents keep their pages and bytes). The exchange of VRAM between the
two posts is the driver's free pool at the wake -- no page changes owner
inside a phase, no alias, no copy.

k(n) is exact granule arithmetic over the real tensors (``seat_vram_rows``):
the largest k whose extra expert pages fit into the Mamba pages the phase does
not map -- so the total mapped at every n is at most the total at the cap, and
the cap is what the boot riegel (the D-FRACTION-SOLVE at --d-bs) prices.

WHAT ENFORCES n (the graphs for bs > n exist and are never replayed)
-------------------------------------------------------------------
* the slot allocator hands out only slots ``1..L(n)`` (``MambaSlotAllocator.
  set_phase_limit``) -- every slot a kernel can index is mapped;
* D admits at most n running requests (``admission_cap`` in
  ``Scheduler.get_num_allocatable_reqs``), so no decode batch is wider than n;
* ``guard`` refuses a forward batch wider than n by name (W-SEAT) before it
  reaches a graph.
An unmapped expert row is a device value in the pool tables (``SEAT_OFF_KEY``,
``expert_pool_device``): never a victim, never routed, never read.

RANKS NEVER DISAGREE: n, the cap, the switch and the slot allocator's state are
replicated, so every rank sets the same slot limit and the same admission cap
without a collective; the physical half (spans, rows) is rank-local and only
TP0 has any (the 3080 workers hold no Mamba state and X = 0 there).

Switch: ``SGLANG_OPT_WEG2_D_SEAT_VRAM`` (environ.py; the Next-Flash launcher
profile writes it into --env-d). Off: nothing here runs, byte-identical to H95 B.
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: fallback when the driver cannot be asked (desk); the 3080/5090 answer 2 MiB
GRANULE_DEFAULT = 2 << 20
GROUP_ENV = "SGLANG_WEG2_GROUP"
POOL_GRAPH_MODE_ENV = "SGLANG_MOE_OFFLOAD_GRAPH_MODE"
LINE_MARK = "WEG2 D-SEAT-VRAM (H95c)"
REFUSAL_CODE = "W-SEAT Weg2DSeatOverrun"
_MIB = float(1 << 20)


class Weg2DSeatOverrun(RuntimeError):
    """A D forward batch wider than the phase's seat count n."""


class Weg2DSeatVramRefused(RuntimeError):
    """The seat-form expert bank could not be trimmed to its cap form."""


# ---------------------------------------------------------------------------
# switch
# ---------------------------------------------------------------------------


def switch_on() -> bool:
    from sglang.srt.environ import envs

    return bool(envs.SGLANG_OPT_WEG2_D_SEAT_VRAM.get())


def armed(env: Optional[Dict[str, str]] = None) -> bool:
    """The switch AND group D (the posts only move on D)."""
    env = os.environ if env is None else env
    if str(env.get(GROUP_ENV, "")).strip().upper() != "D":
        return False
    return switch_on()


def extra_rows_for_rank(rank: int, raw: Optional[str] = None) -> int:
    """``SGLANG_WEG2_D_SEAT_EXPERT_ROWS`` ("16,0,0") for one MoE TP rank; a
    single value applies to every rank; unparsable/absent = 0."""
    if raw is None:
        from sglang.srt.environ import envs

        raw = envs.SGLANG_WEG2_D_SEAT_EXPERT_ROWS.get() or ""
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    if not parts:
        return 0
    try:
        vals = [max(0, int(p)) for p in parts]
    except ValueError:
        return 0
    if len(vals) == 1:
        return vals[0]
    return vals[rank] if 0 <= int(rank) < len(vals) else 0


# ---------------------------------------------------------------------------
# the pure arithmetic (shared by the runtime and the planner's seat table)
# ---------------------------------------------------------------------------


def align_up(x: int, g: int) -> int:
    return -(-int(x) // int(g)) * int(g)


def align_down(x: int, g: int) -> int:
    return int(x) // int(g) * int(g)


def _coalesce(ranges: Sequence[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    out: List[List[int]] = []
    for lo, hi in sorted((int(a), int(b)) for a, b in ranges if int(b) > int(a)):
        if out and lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return tuple((a, b) for a, b in out)


def span_bytes(spans: Sequence[Tuple[int, int]]) -> int:
    return int(sum(int(b) - int(a) for a, b in spans))


@dataclass(frozen=True)
class SlotTensorGeom:
    """A slot-indexed state tensor ``[layers, slots, ...]`` (GDN temporal
    state): layer ``l``'s slot ``s`` is ``slot_bytes`` at
    ``(l * slots + s) * slot_bytes`` of its allocation of ``alloc_bytes``."""

    name: str
    layers: int
    slots: int
    slot_bytes: int
    alloc_bytes: int

    @property
    def tensor_bytes(self) -> int:
        return int(self.layers) * int(self.slots) * int(self.slot_bytes)


def slot_spans(g: SlotTensorGeom, keep: int, granule: int) -> Tuple[Tuple[int, int], ...]:
    """The mapped ranges that keep slots ``[0, keep)`` of every layer:
    OUTWARD-rounded per layer (a granule holding one live byte stays), plus
    the allocation's slack behind the tensor (the caching allocator may place
    a small block there). ``keep >= slots``: the whole allocation."""
    alloc = int(g.alloc_bytes)
    if int(keep) >= int(g.slots):
        return ((0, alloc),)
    sb, S = int(g.slot_bytes), int(g.slots)
    ranges = []
    for layer in range(int(g.layers)):
        lo = layer * S * sb
        ranges.append((align_down(lo, granule), min(alloc, align_up(lo + max(0, int(keep)) * sb, granule))))
    if alloc > g.tensor_bytes:
        ranges.append((align_down(g.tensor_bytes, granule), alloc))
    return _coalesce(ranges)


@dataclass(frozen=True)
class RowTensorGeom:
    """An expert-major bank tensor ``[rows_max, ...]``: ``rows_boot`` = R + C
    rows mapped in the cap form, ``rows_max`` = R + C + X reserved."""

    name: str
    rows_boot: int
    rows_max: int
    row_bytes: int
    alloc_bytes: int


def row_spans(g: RowTensorGeom, rows_on: int, granule: int) -> Tuple[Tuple[int, int], ...]:
    """The mapped prefix for ``rows_on`` rows (outward rounded); every row a
    kernel may touch lies in it. ``rows_on >= rows_max``: the allocation."""
    alloc = int(g.alloc_bytes)
    if int(rows_on) >= int(g.rows_max):
        return ((0, alloc),)
    return ((0, min(alloc, align_up(int(rows_on) * int(g.row_bytes), granule))),)


def phase_slot_limit(size: int, n: int, cap: int) -> int:
    """The slots (1..L) the allocator hands out in a phase of ``n`` seats --
    exactly what a boot for ``n`` seats would size its pool to
    (``d_seats.mamba_slots_for_seats``, the runtime's own
    ``_auto_mamba_demand_size`` shape) when the pool has the cap's size; a pool
    sized otherwise (another ratio, a ceiling fit) is cut proportionally,
    rounded UP. The whole pool at the cap. NF form (38 slots at 6 seats):
    7/13/19/25/32/38."""
    from sglang.srt.weg2.d_seats import mamba_slots_for_seats

    size, n, cap = int(size), max(1, int(n)), max(1, int(cap))
    if n >= cap:
        return size
    if size >= mamba_slots_for_seats(cap):
        return max(1, min(size, mamba_slots_for_seats(n)))
    return max(1, min(size, -(-size * n // cap)))


@dataclass(frozen=True)
class SeatVramRow:
    n: int
    slot_limit: int
    extra_rows: int
    mamba_mapped: int
    expert_mapped: int
    mamba_cap: int
    expert_cap: int

    @property
    def mapped(self) -> int:
        return int(self.mamba_mapped) + int(self.expert_mapped)

    @property
    def cap_mapped(self) -> int:
        return int(self.mamba_cap) + int(self.expert_cap)


def seat_vram_rows(
    slot_tensors: Sequence[SlotTensorGeom],
    row_tensors: Sequence[RowTensorGeom],
    *,
    cap: int,
    pool_size: int,
    extra_max: int,
    granule: int = GRANULE_DEFAULT,
) -> Tuple[SeatVramRow, ...]:
    """n = 1..cap: the slot limit, the extra expert rows k(n) and the mapped
    bytes. k(n) = the largest k <= ``extra_max`` whose expert pages beyond the
    cap form fit into the Mamba pages the phase does not map -- the total
    mapped never exceeds the cap's."""
    cap = max(1, int(cap))

    def mamba(keep: int) -> int:
        return sum(span_bytes(slot_spans(g, keep, granule)) for g in slot_tensors)

    def expert(k: int) -> int:
        return sum(span_bytes(row_spans(g, g.rows_boot + k, granule)) for g in row_tensors)

    m_cap = mamba(int(pool_size) + 1)
    e_cap = expert(0)
    x_max = max(0, int(extra_max))
    if row_tensors:
        x_max = min(x_max, min(int(g.rows_max) - int(g.rows_boot) for g in row_tensors))
    else:
        x_max = 0
    out = []
    for n in range(1, cap + 1):
        lim = phase_slot_limit(pool_size, n, cap)
        m_n = mamba(lim + 1)
        freed = m_cap - m_n
        k = 0
        while k < x_max and expert(k + 1) - e_cap <= freed:
            k += 1
        out.append(SeatVramRow(
            n=n, slot_limit=lim, extra_rows=k, mamba_mapped=m_n,
            expert_mapped=expert(k), mamba_cap=m_cap, expert_cap=e_cap))
    return tuple(out)


# ---------------------------------------------------------------------------
# the saver's span map (tms_csrc patch 3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AllocInfo:
    size: int
    mapped: int
    planned: int
    active: bool


class TmsSpans:
    """``tms_set_spans`` / ``tms_alloc_info`` of the preloaded hook; ``None``
    symbols (a stock wheel without patch 3) make :attr:`available` False."""

    def __init__(self, symbol: Optional[Callable[[str], object]] = None):
        if symbol is None:
            from sglang.srt.utils.torch_memory_saver_adapter import _weg2_ring_symbol

            symbol = _weg2_ring_symbol
        self._set = symbol("tms_set_spans")
        self._info = symbol("tms_alloc_info")
        if self._set is not None and self._info is not None:
            import ctypes

            self._set.restype = ctypes.c_int
            self._set.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                  ctypes.POINTER(ctypes.c_uint64),
                                  ctypes.POINTER(ctypes.c_uint64), ctypes.c_int]
            self._info.restype = ctypes.c_int
            self._info.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_uint64)] * 3 + [
                ctypes.POINTER(ctypes.c_int)]

    @property
    def available(self) -> bool:
        return self._set is not None and self._info is not None

    def info(self, ptr: int) -> Optional[AllocInfo]:
        if not self.available:
            return None
        import ctypes

        size, mapped, planned = ctypes.c_uint64(0), ctypes.c_uint64(0), ctypes.c_uint64(0)
        active = ctypes.c_int(0)
        rc = self._info(ctypes.c_void_p(int(ptr)), ctypes.byref(size), ctypes.byref(mapped),
                        ctypes.byref(planned), ctypes.byref(active))
        if int(rc) != 0:
            return None
        return AllocInfo(int(size.value), int(mapped.value), int(planned.value), bool(active.value))

    def set_spans(self, ptr: int, spans: Sequence[Tuple[int, int]], *, now: bool) -> int:
        if not self.available:
            return -9
        import ctypes

        n = len(spans)
        lo = (ctypes.c_uint64 * max(1, n))(*[int(a) for a, _ in spans])
        hi = (ctypes.c_uint64 * max(1, n))(*[int(b) for _, b in spans])
        return int(self._set(ctypes.c_void_p(int(ptr)), ctypes.c_size_t(n), lo, hi,
                             ctypes.c_int(1 if now else 0)))


_TMS: Optional[TmsSpans] = None


def tms() -> TmsSpans:
    global _TMS
    if _TMS is None:
        _TMS = TmsSpans()
    return _TMS


def granule_for(device) -> int:
    """The driver's VMM granularity for ``device`` (2 MiB on this rig)."""
    try:
        import torch

        from sglang.srt.mem_cache.kv_vmm_backing import query_granularity

        idx = torch.device(device).index
        return int(query_granularity(torch.cuda.current_device() if idx is None else idx))
    except Exception:  # noqa: BLE001 -- the desk has no driver
        return GRANULE_DEFAULT


# ---------------------------------------------------------------------------
# load time: the seat-form expert bank (expert_offload's presplit)
# ---------------------------------------------------------------------------


def presplit_seat_rows(layer) -> int:
    """X for this layer's bank, or 0 (everything below stays byte-identical):
    armed on group D, the device-planned pool, the saver's span map present,
    and the launcher reserved rows for this MoE rank."""
    if not armed():
        return 0
    if str(os.environ.get(POOL_GRAPH_MODE_ENV, "")).strip().lower() != "pool":
        return 0
    x = extra_rows_for_rank(int(getattr(layer, "moe_tp_rank", 0) or 0))
    if x <= 0:
        return 0
    if not tms().available:
        logger.warning(
            "%s: SGLANG_WEG2_D_SEAT_EXPERT_ROWS asks for %d seat rows but the running "
            "saver has no span map (tms_set_spans, tms_csrc patch 3) -- the bank stays "
            "at its cap form, no seat rows", LINE_MARK, x)
        return 0
    return int(x)


def seat_expert_buffer(*, rows: int, extra: int, tail: Tuple[int, ...], dtype, device,
                       in_tag_pool: bool = True, name: str = "", spans: Optional[TmsSpans] = None,
                       granule: Optional[int] = None):
    """The ``[rows + extra, *tail]`` bank tensor of one expert attribute, its
    pages trimmed to the cap form ``[0, rows)`` before anything is written.

    The storage is padded to whole granules (so the caching allocator can put
    nothing behind it: the tail granule is a seat page) and must be ONE saver
    allocation starting at the tensor. A small tensor (its X rows are less than
    two granules: the scales) is kept whole -- its extra rows stay mapped and
    are counted by the runtime line as fixed. A big one that cannot be trimmed
    is refused by name: it would cost its X rows on the card for the whole
    boot, outside everything the planner priced."""
    import torch

    spans = tms() if spans is None else spans
    g = int(granule or granule_for(device))
    elem = torch.empty((), dtype=dtype).element_size()
    row_elems = int(math.prod(tail)) if tail else 1
    row_bytes = row_elems * elem
    rows_max = int(rows) + int(extra)
    need = rows_max * row_bytes
    padded = align_up(need, g)
    flat = torch.empty(padded // elem, dtype=dtype, device=device)
    buf = flat[: rows_max * row_elems].view((rows_max,) + tuple(tail))
    if int(extra) * row_bytes < 2 * g:
        return buf
    info = spans.info(flat.data_ptr())
    if info is None or int(info.size) != int(padded):
        raise Weg2DSeatVramRefused(
            "%s: the seat-form expert bank %s (%d + %d rows x %d B) is not one saver "
            "allocation (%s, padded %d B, in tag pool %s) -- its %d seat rows (%.1f MiB) "
            "could not be unmapped and would sit on the card unpriced. Turn "
            "SGLANG_OPT_WEG2_D_SEAT_VRAM off for this boot." % (
                LINE_MARK, name, int(rows), int(extra), row_bytes,
                "no allocation at this address" if info is None
                else "allocation of %d B" % info.size, padded, in_tag_pool,
                int(extra), int(extra) * row_bytes / _MIB))
    rc = spans.set_spans(flat.data_ptr(), row_spans(
        RowTensorGeom(name, int(rows), rows_max, row_bytes, padded), int(rows), g), now=True)
    if rc != 0:
        raise Weg2DSeatVramRefused(
            "%s: trimming the seat-form expert bank %s to its cap form failed "
            "(tms_set_spans rc=%d)" % (LINE_MARK, name, rc))
    return buf




# ---------------------------------------------------------------------------
# the runtime (one per D rank): a REPLICATED phase state and a RANK-LOCAL
# physical controller
# ---------------------------------------------------------------------------


@dataclass
class _Managed:
    ptr: int
    geom: object


@dataclass
class PhaseState:
    """REPLICATED: what every rank of D derives from the same wake requests --
    the phase's n, the slot limit it set, the epoch it belongs to."""

    epoch: Optional[str] = None
    n: Optional[int] = None
    cap: int = 1
    has_n: bool = False
    done: bool = False
    slot_limit: Optional[int] = None
    note: str = ""


@dataclass
class PhaseApplied:
    """RANK-LOCAL: the pages this rank mapped for the phase."""

    n: int
    extra_rows: int
    mamba_mapped: int
    expert_mapped: int
    cap_mapped: int
    note: str = ""


@dataclass
class SeatVram:
    """The seat posts of ONE rank's pages. Built once (lazily, at the first D
    wake -- after the capture) from the rank's live tensors."""

    cap: int
    pool_size: int
    slot_tensors: List[_Managed]
    row_tensors: List[_Managed]
    caches: List[object]
    fixed_bytes: int
    spans: TmsSpans
    granule: int
    rows: Tuple[SeatVramRow, ...] = ()
    applied: Optional[PhaseApplied] = None
    #: the MambaPool whose ``reset_state`` must zero only mapped slots
    mamba_pool: object = None

    def __post_init__(self):
        x_max = min((int(getattr(c, "seat_rows", 0)) for c in self.caches), default=0)
        self.rows = seat_vram_rows(
            [m.geom for m in self.slot_tensors], [m.geom for m in self.row_tensors],
            cap=self.cap, pool_size=self.pool_size, extra_max=x_max, granule=self.granule)

    @classmethod
    def from_runtime(cls, *, cap: int, req_to_token_pool, model, spans: Optional[TmsSpans] = None,
                     granule: Optional[int] = None) -> "SeatVram":
        spans = tms() if spans is None else spans
        pool = getattr(req_to_token_pool, "mamba_pool", None)
        pool_size = int(getattr(pool, "size", 0) or 0)
        g = int(granule or GRANULE_DEFAULT)
        slot_tensors: List[_Managed] = []
        temporal = getattr(getattr(pool, "mamba_cache", None), "temporal", None)
        if (temporal is not None and spans.available and temporal.dim() >= 2
                and temporal.numel() > 0 and temporal.is_contiguous()):
            if granule is None:
                g = granule_for(temporal.device)
            info = spans.info(temporal.data_ptr())
            if info is not None:
                slot_bytes = int(temporal[0, 0].numel()) * int(temporal.element_size())
                slot_tensors.append(_Managed(temporal.data_ptr(), SlotTensorGeom(
                    "gdn_temporal", int(temporal.shape[0]), int(temporal.shape[1]),
                    slot_bytes, int(info.size))))
        caches, row_tensors, fixed = [], [], 0
        for module in (model.modules() if model is not None else ()):
            cache = getattr(module, "_expert_offload", None)
            x = int(getattr(cache, "seat_rows", 0) or 0) if cache is not None else 0
            if x <= 0:
                continue
            caches.append(cache)
            rows_boot = int(cache.planner.buffer_size)
            for attr, buf in cache._resident.items():
                row_bytes = int(buf[0].numel()) * int(buf.element_size()) if buf.shape[0] else 0
                info = spans.info(buf.data_ptr()) if spans.available else None
                if info is None or x * row_bytes < 2 * g:
                    fixed += x * row_bytes
                    continue
                row_tensors.append(_Managed(buf.data_ptr(), RowTensorGeom(
                    "%s.%s" % (getattr(cache.layer, "layer_id", "?"), attr), rows_boot,
                    rows_boot + x, row_bytes, int(info.size))))
        return cls(cap=int(cap), pool_size=pool_size, slot_tensors=slot_tensors,
                   row_tensors=row_tensors, caches=caches, fixed_bytes=int(fixed),
                   spans=spans, granule=g, mamba_pool=pool)

    def row_for(self, n: int) -> SeatVramRow:
        return self.rows[max(1, min(int(n), self.cap)) - 1]

    def apply(self, n: int) -> PhaseApplied:
        """The pages of a phase of ``n`` seats (``n`` = cap: the cap form).
        Called BEFORE the saver resumes the tag of a paused allocation (its
        plan is set) and at any time for a mapped one (the expert bank grows
        or shrinks in place)."""
        n = max(1, min(int(n), self.cap))
        notes = []
        k = self.row_for(n).extra_rows if (self.slot_tensors and self.row_tensors) else 0
        if n < self.cap and k == 0:
            notes.append("no expert row fits the freed Mamba pages: the Mamba pool keeps "
                         "the cap form (nothing to hand over)")
        keep = (self.row_for(n).slot_limit + 1) if k > 0 else self.pool_size + 1
        infos = [self.spans.info(m.ptr) for m in self.slot_tensors]
        if keep <= self.pool_size and any(i is not None and i.active for i in infos):
            notes.append("the Mamba pool is mapped (live) at this request: it cannot "
                         "shrink, no seat rows this phase")
            k, keep = 0, self.pool_size + 1
        for m, info in zip(self.slot_tensors, infos):
            if info is not None and not info.active:
                rc = self.spans.set_spans(m.ptr, slot_spans(m.geom, keep, self.granule), now=False)
                if rc != 0:
                    raise Weg2DSeatVramRefused("%s: tms_set_spans(%s) rc=%d"
                                               % (LINE_MARK, m.geom.name, rc))
        if self.mamba_pool is not None and self.slot_tensors:
            # MambaPool.reset_state (the flush at the wake and before the next
            # sleep) zeroes only the slots that have pages
            self.mamba_pool._weg2_seat_keep = None if keep > self.pool_size else int(keep)
        experts_live = False
        for m in self.row_tensors:
            info = self.spans.info(m.ptr)
            live = info is not None and info.active
            experts_live = experts_live or live
            rc = self.spans.set_spans(
                m.ptr, row_spans(m.geom, m.geom.rows_boot + k, self.granule), now=live)
            if rc != 0:
                raise Weg2DSeatVramRefused("%s: tms_set_spans(%s, now=%s) rc=%d"
                                           % (LINE_MARK, m.geom.name, live, rc))
        # the tables: written now when the bank is mapped (the wake's rearm
        # already ran), else recorded for the rearm's reinit
        for cache in self.caches:
            cache.set_seat_rows_on(k, device_write=experts_live)
        applied = PhaseApplied(
            n=n, extra_rows=int(k),
            mamba_mapped=sum(span_bytes(slot_spans(m.geom, keep, self.granule))
                             for m in self.slot_tensors),
            expert_mapped=sum(span_bytes(row_spans(m.geom, m.geom.rows_boot + k, self.granule))
                              for m in self.row_tensors),
            cap_mapped=int(self.row_for(self.cap).cap_mapped), note="; ".join(notes))
        self.applied = applied
        return applied

    def table_lines(self) -> List[str]:
        return [
            "%s table n=%d: slots 1..%d, expert rows +%d, mapped %.1f MiB (mamba %.1f + "
            "experts %.1f) of cap form %.1f MiB -- GERECHNET over %d GDN / %d expert "
            "tensor(s), granule %d KiB, fixed %.1f MiB (seat rows of tensors too small "
            "to unmap)"
            % (LINE_MARK, r.n, r.slot_limit, r.extra_rows, r.mapped / _MIB,
               r.mamba_mapped / _MIB, r.expert_mapped / _MIB, r.cap_mapped / _MIB,
               len(self.slot_tensors), len(self.row_tensors), self.granule // 1024,
               self.fixed_bytes / _MIB)
            for r in self.rows
        ]


# ---------------------------------------------------------------------------
# the scheduler's entry points (d_park_runtime delegates here)
# ---------------------------------------------------------------------------

PHASE_ATTR = "_weg2_d_seat_phase"
CTL_ATTR = "_weg2_d_seat_vram"


def _cap_of(sched) -> int:
    return int(getattr(getattr(sched, "server_args", None), "max_running_requests", 0) or 1)


def _req_pool(sched):
    runner = getattr(getattr(sched, "tp_worker", None), "model_runner", None)
    return getattr(runner, "req_to_token_pool", None) or getattr(sched, "req_to_token_pool", None)


def controller(sched) -> Optional[SeatVram]:
    """This rank's page controller, built on first use; None when nothing on
    this rank is seat-proportional (the 3080 workers) or a piece is missing."""
    ctl = getattr(sched, CTL_ATTR, None)
    if ctl is not None:
        return ctl if ctl is not False else None
    runner = getattr(getattr(sched, "tp_worker", None), "model_runner", None)
    try:
        ctl = SeatVram.from_runtime(cap=_cap_of(sched), req_to_token_pool=_req_pool(sched),
                                    model=getattr(runner, "model", None))
        if not ctl.slot_tensors and not ctl.row_tensors:
            ctl = None
    except Exception as exc:  # noqa: BLE001 -- a missing piece names itself, never guesses
        logger.warning("%s: no page controller on this rank: %s: %s", LINE_MARK,
                       type(exc).__name__, exc)
        ctl = None
    setattr(sched, CTL_ATTR, ctl if ctl is not None else False)
    if ctl is not None:
        for ln in ctl.table_lines():
            logger.info("%s", ln)
    return ctl


def on_wake(sched, recv_req, seats) -> Optional[PhaseState]:
    """Every D resume request (weight_updater, before the saver resumes a
    tag). ``seats`` = ``d_seats.PhaseSeats`` of this request, or None when it
    carries no count.

    The FIRST request of a wake without a count resets the phase to the cap
    form (the front sends the weights leg before -- or beside -- the kv_cache
    leg that carries handoff_n); the request that carries n applies n; any
    later request of the same wake changes nothing. None when not armed."""
    if not armed():
        return None
    epoch = getattr(recv_req, "epoch", None)
    n = None if seats is None else int(seats.n)
    st = getattr(sched, PHASE_ATTR, None)
    if st is None:
        st = PhaseState(cap=_cap_of(sched))
        setattr(sched, PHASE_ATTR, st)
    if epoch != st.epoch:
        st.epoch, st.has_n, st.done, st.note = epoch, False, False, ""
    elif st.has_n or (n is None and st.done):
        return None
    st.cap = _cap_of(sched)
    target = st.cap if n is None else max(1, min(n, st.cap))
    # 1. REPLICATED: the slot limit -- every rank, same inputs, no collective
    pool = _req_pool(sched)
    allocator = getattr(pool, "mamba_allocator", None)
    limit = None
    n_phys = target
    if allocator is not None and hasattr(allocator, "set_phase_limit"):
        size = int(getattr(allocator, "size", 0) or 0)
        want = None if target >= st.cap else phase_slot_limit(size, target, st.cap)
        if allocator.set_phase_limit(want):
            limit = want
        else:
            st.note = ("slots above %s still in use at the wake: slot limit NOT applied, "
                       "the phase keeps the cap form" % want)
            n_phys = st.cap
            allocator.set_phase_limit(None)
    st.n, st.slot_limit, st.done = target, limit, True
    st.has_n = n is not None
    # 2. RANK-LOCAL: the pages
    ctl = controller(sched)
    applied = ctl.apply(n_phys) if ctl is not None else None
    logger.info("%s", phase_line(st, applied, ctl))
    return st


def phase_line(st: PhaseState, applied: Optional[PhaseApplied], ctl: Optional[SeatVram]) -> str:
    phys = (
        "expert_rows=+%d (vs n=%d) mapped_mib=%.1f (mamba %.1f + experts %.1f; cap form "
        "%.1f) fixed_mib=%.1f"
        % (applied.extra_rows, st.cap, (applied.mamba_mapped + applied.expert_mapped) / _MIB,
           applied.mamba_mapped / _MIB, applied.expert_mapped / _MIB,
           applied.cap_mapped / _MIB, ctl.fixed_bytes / _MIB)
        if applied is not None else "no seat-proportional pages on this rank"
    )
    notes = [x for x in (st.note, applied.note if applied is not None else "") if x]
    return "%s n=%d of cap %d epoch=%s mamba_slots=%s %s%s" % (
        LINE_MARK, st.n, st.cap, st.epoch,
        "all" if st.slot_limit is None else "1..%d" % st.slot_limit, phys,
        (" -- " + "; ".join(notes)) if notes else "")


def admission_cap(sched) -> Optional[int]:
    """At most n running requests in a D phase of n < cap seats (None = no
    cap). REPLICATED: a function of the wake requests only."""
    st = getattr(sched, PHASE_ATTR, None)
    if st is None or st.n is None or st.n >= st.cap or not armed():
        return None
    return int(st.n)


def guard(sched, batch) -> None:
    """W-SEAT: a forward batch wider than the phase's n never reaches a graph."""
    cap = admission_cap(sched)
    if cap is None or batch is None:
        return
    bs = len(getattr(batch, "reqs", None) or ())
    if bs > cap:
        st = getattr(sched, PHASE_ATTR)
        raise Weg2DSeatOverrun(
            "%s: a forward batch of %d requests in a D phase of n=%d seats (epoch %s) -- "
            "the posts of seats above n are not mapped (slots 1..%s); the admission cap "
            "should have held it" % (REFUSAL_CODE, bs, cap, st.epoch, st.slot_limit))
