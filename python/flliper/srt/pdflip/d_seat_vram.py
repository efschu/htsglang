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

Switch: ``FLLIPER_OPT_PDFLIP_D_SEAT_VRAM`` (environ.py; the Next-Flash launcher
profile writes it into --env-d). Off: nothing here runs, byte-identical to H95 B.
"""
from __future__ import annotations

import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: fallback when the driver cannot be asked (desk); the 3080/5090 answer 2 MiB
GRANULE_DEFAULT = 2 << 20
GROUP_ENV = "FLLIPER_PDFLIP_GROUP"
POOL_GRAPH_MODE_ENV = "FLLIPER_MOE_OFFLOAD_GRAPH_MODE"
LINE_MARK = "PDFLIP D-SEAT-VRAM (H95c)"
REFUSAL_CODE = "W-SEAT PdFlipDSeatOverrun"
#: S1-Wisch (rc12z17 28.09.): a live apply that would release mapped bytes
WIPE_CODE = "W-SEAT-WIPE"
#: one line per live apply: extents kept, cells freed / mapped, wiped=0
LIVE_MARK = "PDFLIP D-SEAT-VRAM LIVE-SPANS"
#: S106: the lend's own marker (d_transient_lend.MARK) for the stage-keep line
LEND_MARK = "PDFLIP D-TRANSIENT-LEND"
_MIB = float(1 << 20)


class PdFlipDSeatOverrun(RuntimeError):
    """A D forward batch wider than the phase's seat count n."""


class PdFlipDSeatVramRefused(RuntimeError):
    """The seat-form expert bank could not be trimmed to its cap form."""


class PdFlipDSeatVramNoMemory(PdFlipDSeatVramRefused):
    """A live ``tms_set_spans`` answered rc=2 (cuMemCreate out of memory)."""


#: torch_memory_saver's rc for a map the card cannot back (cuMemCreate OOM)
TMS_RC_OUT_OF_MEMORY = 2


#: S77-B: one line per rank whenever a live KV grow found the card short
GROW_ROOM_MARK = "PDFLIP D-SEAT-VRAM GROW-ROOM"


def _fresh_bytes(prev, spans, mapped: int) -> int:
    """Bytes ``set_spans(spans, now=True)`` maps fresh over the extents ``prev``: the
    new plan's ranges minus the extents it keeps. ``prev`` unknown: the mapped bytes
    count as kept (never below zero)."""
    want = span_bytes(spans)
    if prev is None:
        return max(0, want - int(mapped))
    kept = [e for e in prev if any(lo <= e[0] and e[1] <= hi for lo, hi in spans)]
    return max(0, want - span_bytes(kept))


def _released_bytes(prev, spans, mapped: int) -> int:
    """Bytes a live ``set_spans(spans)`` gives back: the extents of ``prev`` that lie
    wholly inside no range of the new plan."""
    if prev is None:
        return max(0, int(mapped) - span_bytes(spans))
    return span_bytes([e for e in prev if not any(lo <= e[0] and e[1] <= hi for lo, hi in spans)])


@dataclass
class GrowRoom:
    """S77-B: one rank's measured answer for a live KV grow (bytes)."""

    ok: bool
    need: int
    free_before: int
    free_after: int
    trimmed: int
    rows_from: int
    rows_cell: int
    rows_to: Optional[int]
    release_all: int

    def line(self, *, n: int, stage: int) -> str:
        return ("%s n=%d stage=S%d need_mib=%.1f free_mib=%.1f trimmed_mib=%.1f free_after_mib=%.1f "
                "rows_on=%d cell_rows=%d rows_to=%s release_all_mib=%.1f verdict=%s (the KV the cell "
                "books is mapped into MEASURED bytes: the cache trimmed first, then this rank's own "
                "expert rows OFF; short = no rank moves, the admission waits)"
                % (GROW_ROOM_MARK, n, stage, self.need / _MIB, self.free_before / _MIB,
                   self.trimmed / _MIB, self.free_after / _MIB, self.rows_from, self.rows_cell,
                   "-" if self.rows_to is None else self.rows_to, self.release_all / _MIB,
                   "ok" if self.ok else "short"))


class CardRoom:
    """S77-B: the card as the CUDA driver reports it (``cudaMemGetInfo``) and the
    caching allocator's trim (synchronize + empty_cache, as PDFLIP-EXTEND-CACHE-TRIM).
    ``free`` is None without an initialised CUDA context (CPU tests, P): nothing is
    measured then and the grow keeps its plan."""

    def free(self) -> Optional[int]:
        try:
            import torch

            if not torch.cuda.is_available() or not torch.cuda.is_initialized():
                return None
            return int(torch.cuda.mem_get_info()[0])
        except Exception:  # noqa: BLE001 -- an unreadable card is "not measured"
            return None

    def trim(self) -> int:
        import torch

        try:
            if torch.cuda.is_current_stream_capturing():
                return 0
        except Exception:  # noqa: BLE001
            pass
        r0 = int(torch.cuda.memory_reserved())
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return max(0, r0 - int(torch.cuda.memory_reserved()))


_CARD = CardRoom()


class PdFlipDSeatVramWipe(PdFlipDSeatVramRefused):
    """W-SEAT-WIPE: a live span plan would release an extent holding bytes it
    keeps. Never swallowed into "no controller on this rank" (``controller``):
    a rank whose pages would be remapped fresh stops by name."""


# ---------------------------------------------------------------------------
# switch
# ---------------------------------------------------------------------------


def switch_on() -> bool:
    from flliper.srt.environ import envs

    return bool(envs.FLLIPER_OPT_PDFLIP_D_SEAT_VRAM.get())


def _rewake_on() -> bool:
    """D-SEAT-REWAKE (pdflip/d_seat_rewake.py): the live seat re-plan."""
    try:
        from flliper.srt.environ import envs

        return bool(envs.FLLIPER_PDFLIP_D_SEAT_REWAKE.get())
    except Exception:  # noqa: BLE001 -- an unreadable switch is off
        return False


def armed(env: Optional[Dict[str, str]] = None) -> bool:
    """The switch AND group D (the posts only move on D)."""
    env = os.environ if env is None else env
    if str(env.get(GROUP_ENV, "")).strip().upper() != "D":
        return False
    return switch_on()


def extra_rows_for_rank(rank: int, raw: Optional[str] = None) -> int:
    """``FLLIPER_PDFLIP_D_SEAT_EXPERT_ROWS`` ("16,0,0") for one MoE TP rank; a
    single value applies to every rank; unparsable/absent = 0."""
    if raw is None:
        from flliper.srt.environ import envs

        raw = envs.FLLIPER_PDFLIP_D_SEAT_EXPERT_ROWS.get() or ""
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


def lend_on() -> bool:
    """D-TRANSIENT-LEND (pdflip/d_transient_lend.py): the switch AND the elastic
    rows (its diagnostic stop stops the lend too)."""
    from flliper.srt.environ import envs

    return bool(envs.FLLIPER_PDFLIP_D_TRANSIENT_LEND.get()) and elastic_on()


def lend_armed() -> bool:
    """The lend can run: switched on AND the launcher wrote the floor (the
    near-OOM edge it lends down to). Without the floor the bank keeps its
    lattice and nothing is lent -- byte-identical to before."""
    if not lend_on():
        return False
    from flliper.srt.environ import envs

    return bool(str(envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.get() or "").strip())


def lend_head_rows_for_rank(rank: int, raw: Optional[str] = None) -> int:
    """``FLLIPER_PDFLIP_D_LEND_HEAD_ROWS`` for one MoE TP rank (one value = every
    rank); 0 unless the lend is armed (``lend_armed``)."""
    if raw is None:
        if not lend_armed():
            return 0
        from flliper.srt.environ import envs

        raw = envs.FLLIPER_PDFLIP_D_LEND_HEAD_ROWS.get() or ""
    return extra_rows_for_rank(rank, raw)


def lend_floor_mib_for_rank(rank: int, raw: Optional[str] = None) -> Optional[float]:
    """The near-OOM edge of this D rank (``FLLIPER_PDFLIP_D_LEND_FLOOR_MIB``),
    None when the launcher wrote none -- no lend then, nothing is guessed."""
    if raw is None:
        from flliper.srt.environ import envs

        raw = envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.get()
    parts = [p.strip() for p in str(raw or "").split(",") if p.strip()]
    if not parts:
        return None
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        return None
    if len(vals) == 1:
        return vals[0]
    return vals[rank] if 0 <= int(rank) < len(vals) else None


def lend_lattice(x_max: int, step: int) -> Tuple[int, ...]:
    """The row counts a lend may end at: every ``step`` rows and the top."""
    x_max, step = int(x_max), int(step)
    if x_max <= 0 or step <= 0:
        return ()
    return tuple(sorted(set(range(step, x_max + 1, step)) | {x_max}))


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


def split_at(ranges: Sequence[Tuple[int, int]], points: Iterable[int]) -> Tuple[Tuple[int, int], ...]:
    """``ranges`` cut at every point strictly inside one of them."""
    pts = sorted({int(p) for p in points})
    out: List[Tuple[int, int]] = []
    for lo, hi in ranges:
        cur = int(lo)
        for p in pts:
            if cur < p < int(hi):
                out.append((cur, p))
                cur = p
        out.append((cur, int(hi)))
    return tuple(out)


def slot_spans(g: SlotTensorGeom, keep: int, granule: int,
               cuts: Sequence[int] = ()) -> Tuple[Tuple[int, int], ...]:
    """The mapped ranges that keep slots ``[0, keep)`` of every layer:
    OUTWARD-rounded per layer (a granule holding one live byte stays), plus
    the allocation's slack behind the tensor (the caching allocator may place
    a small block there). ``keep >= slots``: the whole allocation.

    ``cuts`` (S1-Wisch, rc12z17 28.09.): the keep values a later phase may
    take. The saver maps ONE extent per range and a live ``tms_set_spans``
    keeps only extents wholly inside one range of the new plan -- every other
    extent is released and its range mapped FRESH, i.e. its bytes are gone.
    Cut at every layer start and at every such keep, each extent is one cell
    of a lattice all these plans share: a live shrink frees whole cells above
    the new end and never drops a cell it keeps."""
    alloc = int(g.alloc_bytes)
    sb, S = int(g.slot_bytes), int(g.slots)
    if int(keep) >= S:
        ranges: Tuple[Tuple[int, int], ...] = ((0, alloc),)
    else:
        rl = []
        for layer in range(int(g.layers)):
            lo = layer * S * sb
            rl.append((align_down(lo, granule), min(alloc, align_up(lo + max(0, int(keep)) * sb, granule))))
        if alloc > g.tensor_bytes:
            rl.append((align_down(g.tensor_bytes, granule), alloc))
        ranges = _coalesce(rl)
    if not cuts:
        return ranges
    points = []
    for layer in range(int(g.layers)):
        lo = layer * S * sb
        points.append(align_down(lo, granule))
        points.extend(min(alloc, align_up(lo + int(c) * sb, granule)) for c in cuts if 0 < int(c) < S)
    if alloc > g.tensor_bytes:
        points.append(align_down(g.tensor_bytes, granule))
    return split_at(ranges, points)


@dataclass(frozen=True)
class RowTensorGeom:
    """An expert-major bank tensor ``[rows_max, ...]``: ``rows_boot`` = R + C
    rows mapped in the cap form, ``rows_max`` = R + C + X reserved."""

    name: str
    rows_boot: int
    rows_max: int
    row_bytes: int
    alloc_bytes: int


def row_spans(g: RowTensorGeom, rows_on: int, granule: int,
              cuts: Sequence[int] = ()) -> Tuple[Tuple[int, int], ...]:
    """The mapped prefix for ``rows_on`` rows (outward rounded); every row a
    kernel may touch lies in it. ``rows_on >= rows_max``: the allocation.

    ``cuts`` (S1-Wisch, rc12z17 28.09.): the row counts a later phase may map.
    rc12z17 10:51:24 had the LIVE bank mapped as ONE extent ``[0, R+C+33)``
    and shrank it to ``R+C+23`` -- the saver released that extent (it is not
    inside the smaller plan) and mapped the prefix fresh: residents, staging
    and LRU rows of all 48 layers were gone, D decoded with uninitialised
    experts (accept 2.3 -> 1.05) and its next sleep handed them to the store.
    Cut at every such row count, a shrink releases only whole cells above the
    new end."""
    alloc = int(g.alloc_bytes)
    rb = int(g.row_bytes)
    end = alloc if int(rows_on) >= int(g.rows_max) else min(alloc, align_up(int(rows_on) * rb, granule))
    if not cuts:
        return ((0, end),)
    return split_at(((0, end),), (min(alloc, align_up(int(c) * rb, granule)) for c in cuts if int(c) > 0))


def straddling(prev: Sequence[Tuple[int, int]],
               new: Sequence[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    """The first mapped extent of ``prev`` a live apply of ``new`` would
    release WITH bytes the new plan keeps -- neither wholly inside one range
    of ``new`` (kept, core.cpp ``set_spans``) nor disjoint from all of them
    (freed whole). None: the apply keeps every byte it keeps."""
    for a, b in prev:
        if any(lo <= a and b <= hi for lo, hi in new):
            continue
        if all(b <= lo or a >= hi for lo, hi in new):
            continue
        return (int(a), int(b))
    return None


def phase_slot_limit(size: int, n: int, cap: int) -> int:
    """The slots (1..L) the allocator hands out in a phase of ``n`` seats --
    exactly what a boot for ``n`` seats would size its pool to
    (``d_seats.mamba_slots_for_seats``, the runtime's own
    ``_auto_mamba_demand_size`` shape) when the pool has the cap's size; a pool
    sized otherwise (another ratio, a ceiling fit) is cut proportionally,
    rounded UP. The whole pool at the cap. NF form (38 slots at 6 seats):
    7/13/19/25/32/38."""
    from flliper.srt.pdflip.d_seats import mamba_slots_for_seats

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
# #251c: the attention host's KV pool as the THIRD post (KV stages)
# ---------------------------------------------------------------------------
#
# Form A keeps the whole KV on TP0. A KV stage S_j maps the token prefix
# [0, T_j) of every KV tensor; the pages a stage above S0 maps come out of
# the expert bank's tail rows, exactly the way H95c hands the pages of empty
# seats to extra expert rows. The virtual ranges stay at the top stage (the
# graphs), the boot form is S0 with ``boot_rows_on`` stage rows ON -- the form
# the planner prices -- and every (n, j) cell maps at most what it maps.


@dataclass(frozen=True)
class KvTensorGeom:
    """A token-indexed KV tensor (``SlotTensorGeom`` at the TOP stage) and how
    tokens become its slots: a paged K/V buffer holds ``tokens + page`` rows
    (``token_pad`` = page, ``token_ratio`` = 1); the QSA compressed keys hold
    one slot per ``ratio`` tokens of the same padded space."""

    geom: SlotTensorGeom
    token_ratio: int = 1
    token_pad: int = 0
    #: #239 S3g: (cp_S, cp_ratio) of a rank holding its owner share of a
    #: token-cut FA pool -- global slot L sits at row (L // S) * ratio + (L %
    #: S - lo), so the first T global tokens live in the first
    #: ``dcp_compact_pool_rows(T)`` = (T // S + 1) * ratio rows. (0, 0) = the
    #: whole context on this rank (every other pool), byte-identical.
    owner_block: Tuple[int, int] = (0, 0)

    def slots_for(self, tokens: int) -> int:
        S, ratio = (int(x) for x in self.owner_block)
        if S > 0 and ratio > 0:
            need = (int(tokens) // S + 1) * ratio + int(self.token_pad)
            return max(0, min(int(self.geom.slots), need))
        need = -(-(int(tokens) + int(self.token_pad)) // max(1, int(self.token_ratio)))
        return max(0, min(int(self.geom.slots), need))


def kv_mapped_bytes(kv_tensors: Sequence[KvTensorGeom], tokens: int, granule: int) -> int:
    return sum(span_bytes(slot_spans(k.geom, k.slots_for(tokens), granule)) for k in kv_tensors)


@dataclass(frozen=True)
class StageCell:
    """One (n seats, KV stage j) phase form. ``extra_rows`` = the expert rows
    ON (k), ``mamba_keep`` = the slots per layer the GDN pool maps
    (``pool_size + 1`` = its cap form); ``feasible`` = some k >= 0 keeps the
    mapped total within the boot form's."""

    n: int
    stage: int
    tokens: int
    slot_limit: int
    mamba_keep: int
    extra_rows: int
    mamba_mapped: int
    expert_mapped: int
    kv_mapped: int
    cap_mapped: int
    feasible: bool
    #: k with the GDN pool in its cap form (it cannot shrink while it is
    #: mapped and live -- H95c's rule); -1 = the stage needs the GDN pages
    rows_full: int = -1
    #: D-SEAT-REWAKE: k with the GDN pool mapped for n seats (``slot_limit +
    #: 1``), -1 at the cap or when it does not fit -- what a live stage move
    #: books when the live pool IS that shrunk form (a live re-seat made it)
    rows_shrunk: int = -1

    @property
    def mapped(self) -> int:
        return int(self.mamba_mapped) + int(self.expert_mapped) + int(self.kv_mapped)


def stage_vram_cells(
    slot_tensors: Sequence[SlotTensorGeom],
    row_tensors: Sequence[RowTensorGeom],
    kv_tensors: Sequence[KvTensorGeom],
    *,
    cap: int,
    pool_size: int,
    extra_max: int,
    stage_tokens: Sequence[int],
    boot_rows_on: int = 0,
    granule: int = GRANULE_DEFAULT,
) -> Dict[Tuple[int, int], StageCell]:
    """Every (n, j): the largest k whose expert pages, with the GDN slots of n
    seats and the KV prefix of stage j, stay within the BOOT form (n = cap,
    j = 0, ``boot_rows_on`` rows ON). The GDN pool shrinks only when that buys
    a row (else it keeps its cap form -- nothing to hand over, H95c's rule).
    ``kv_tensors=()``, one stage and ``boot_rows_on=0`` give H95c's k(n)."""
    cap = max(1, int(cap))
    stages = [int(t) for t in stage_tokens] or [0]

    def mamba(keep: int) -> int:
        return sum(span_bytes(slot_spans(g, keep, granule)) for g in slot_tensors)

    def expert(k: int) -> int:
        return sum(span_bytes(row_spans(g, g.rows_boot + k, granule)) for g in row_tensors)

    x_max = max(0, int(extra_max))
    if row_tensors:
        x_max = min(x_max, min(int(g.rows_max) - int(g.rows_boot) for g in row_tensors))
    else:
        x_max = 0
    k_boot = max(0, min(int(boot_rows_on), x_max))
    m_cap = mamba(int(pool_size) + 1)
    budget = m_cap + expert(k_boot) + kv_mapped_bytes(kv_tensors, stages[0], granule)
    ex = [expert(k) for k in range(x_max + 1)]

    def best_k(fixed: int) -> int:
        k = -1
        while k < x_max and fixed + ex[k + 1] <= budget:
            k += 1
        return k

    out: Dict[Tuple[int, int], StageCell] = {}
    for n in range(1, cap + 1):
        lim = phase_slot_limit(pool_size, n, cap)
        m_n = mamba(lim + 1)
        for j, tokens in enumerate(stages):
            kv = kv_mapped_bytes(kv_tensors, tokens, granule)
            k_shrunk = best_k(m_n + kv) if n < cap else -1
            k_full = best_k(m_cap + kv)
            if k_shrunk > k_full:
                k, keep, m = k_shrunk, lim + 1, m_n
            else:
                k, keep, m = k_full, int(pool_size) + 1, m_cap
            out[(n, j)] = StageCell(
                n=n, stage=j, tokens=tokens, slot_limit=lim, mamba_keep=keep,
                extra_rows=max(0, k), mamba_mapped=m, expert_mapped=ex[max(0, k)],
                kv_mapped=kv, cap_mapped=budget, feasible=k >= 0, rows_full=k_full,
                rows_shrunk=k_shrunk)
    return out


@dataclass(frozen=True)
class StageChoice:
    stage: int
    tokens: int
    demand: Optional[int]
    over: bool  # demand above the chosen (= highest usable) stage: the youngest parks
    reason: str


def choose_stage(
    cells: Dict[Tuple[int, int], StageCell],
    n: int,
    demand_tokens: Optional[int],
    *,
    min_rows_on: int = 0,
) -> StageChoice:
    """#251c (Nutzer: Leistung geht vor): the SMALLEST stage whose tokens hold
    the phase's demand, among the stages the n-seat form can fund with at least
    ``min_rows_on`` expert rows ON (the captured waves' floor). The stage rises
    only when the next wake's demand exceeds the KV and falls as soon as it
    fits again; above the highest usable stage the highest one is taken and
    the youngest parks (H91, ``over``). A pure function of the wake request --
    every rank takes the same stage. No demand on the wake (an older front):
    S0, the boot form."""
    usable = sorted(
        j for (m, j), c in cells.items()
        if m == n and c.feasible and c.extra_rows >= int(min_rows_on)
    )
    if not usable:
        usable = [0]
    t = {j: cells[(n, j)].tokens for j in usable if (n, j) in cells}
    if demand_tokens is None:
        j0 = 0 if 0 in t else usable[0]
        return StageChoice(j0, t.get(j0, 0), None, False, "no demand on the wake: S0")
    d = int(demand_tokens)
    for j in usable:
        if t.get(j, 0) >= d:
            return StageChoice(j, t[j], d, False, "smallest stage holding the demand")
    top = usable[-1]
    return StageChoice(top, t.get(top, 0), d, True,
                       "demand above the highest usable stage: the youngest parks")


@dataclass(frozen=True)
class StageForm:
    """#251c, REPLICATED: the launcher's form values -- the stages' tokens,
    the stage rows ON in the boot form, and the highest stage a phase of n
    seats may take. Every rank of D reads the same --env-d, so every rank
    chooses the same stage from these and the wake request alone (the 3080
    workers hold no KV and no cells; TP0 checks the table against its pages,
    ``check_form_against_cells``)."""

    tokens: Tuple[int, ...]
    rows_on: int = 0
    max_by_seats: Tuple[int, ...] = ()
    #: #251d (FLLIPER_PDFLIP_D_KV_STAGE_BY_DEMAND): no table -- every stage is
    #: open to every n, the wake's demand alone picks it (choose_form_stage)
    by_demand: bool = False

    def max_stage(self, n: int) -> int:
        top = len(self.tokens) - 1
        if self.by_demand or not self.max_by_seats:
            return top
        i = min(max(1, int(n)), len(self.max_by_seats)) - 1
        return max(0, min(top, int(self.max_by_seats[i])))


def _ints(raw) -> Tuple[int, ...]:
    out = []
    for p in str(raw or "").split(","):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(int(p))
        except ValueError:
            return ()
    return tuple(out)


def stage_form(env: Optional[Dict[str, str]] = None) -> Optional[StageForm]:
    """The armed stage form of this D process, or None (off, not group D, or
    fewer than two stages -- then everything is H95c, byte-identical)."""
    if not armed(env):
        return None
    from flliper.srt.environ import envs

    tokens = tuple(sorted({t for t in _ints(envs.FLLIPER_PDFLIP_D_KV_STAGE_TOKENS.get()) if t > 0}))
    if len(tokens) < 2:
        return None
    rows_on = max(0, int(envs.FLLIPER_PDFLIP_D_KV_STAGE_ROWS.get() or 0))
    by_rank = _ints(envs.FLLIPER_PDFLIP_D_KV_STAGE_ROWS_BY_RANK.get())
    if by_rank:
        # #239 S3g: every KV rank funds its own stages; this rank's entry
        r = _this_rank()
        rows_on = max(0, int(by_rank[r])) if 0 <= r < len(by_rank) else 0
    return StageForm(
        tokens=tokens,
        rows_on=rows_on,
        max_by_seats=_ints(envs.FLLIPER_PDFLIP_D_KV_STAGE_MAX_BY_SEATS.get()),
        by_demand=bool(envs.FLLIPER_PDFLIP_D_KV_STAGE_BY_DEMAND.get()),
    )


def form_seats_text(form: StageForm):
    """The highest stage per seat count as the capture/stage lines name it."""
    if form.by_demand:
        return "by demand (every stage at every n)"
    return list(form.max_by_seats) or "all"


def choose_form_stage(form: StageForm, n: int, demand_tokens: Optional[int]) -> StageChoice:
    """``choose_stage`` over the replicated form: stages 0..max_stage(n)."""
    top = form.max_stage(n)
    if demand_tokens is None:
        return StageChoice(0, form.tokens[0], None, False, "no demand on the wake: S0")
    d = int(demand_tokens)
    for j in range(top + 1):
        if form.tokens[j] >= d:
            return StageChoice(j, form.tokens[j], d, False, "smallest stage holding the demand")
    return StageChoice(top, form.tokens[top], d, True,
                       "demand above the highest usable stage: the youngest parks")


def check_form_against_cells(form: StageForm, cells: Dict[Tuple[int, int], StageCell],
                             cap: int) -> None:
    """TP0 at its first wake: every stage the form lets a phase of n seats
    take must be fundable by this rank's pages. The workers choose from the
    same form without cells, so a table TP0 cannot honour is refused by name
    here instead of being quietly lowered on one rank (RAENGE-NIE-UNEINS)."""
    bad = [(n, j) for n in range(1, int(cap) + 1) for j in range(form.max_stage(n) + 1)
           if not cells.get((n, j), None) or not cells[(n, j)].feasible]
    if bad:
        raise PdFlipDSeatVramRefused(
            "%s: the stage form promises (n seats, stage) %s that this rank's pages "
            "cannot fund (stage tokens %s, stage rows %d) -- the launcher's table and "
            "the attention host's geometry disagree; nothing was mapped"
            % (LINE_MARK, bad[:6], list(form.tokens), form.rows_on))


def capture_floors(form: StageForm, cells: Dict[Tuple[int, int], StageCell],
                   cap: int) -> Tuple[int, ...]:
    """#251c, the capture floor per batch size b = 1..cap: the fewest expert
    rows ON (k) in any phase a batch of b can replay in -- n >= b seats at a
    stage the form lets n take. The graph of b is captured with these rows
    counted as ON (``expert_pool_device.pool_row_capacity``), so its wave count
    holds in every such phase; the launcher's ``max_by_seats`` decides how low
    it goes (a stage that would add a wave is simply not allowed at n)."""
    out = []
    for b in range(1, int(cap) + 1):
        ks = [cells[(n, j)].extra_rows for n in range(b, int(cap) + 1)
              for j in range(form.max_stage(n) + 1) if (n, j) in cells]
        out.append(max(0, min(ks)) if ks else 0)
    return tuple(out)


#: S3g floor: the refusal of a rank that carries stage rows but staged no KV
STAGE_ROWS_NO_KV_CODE = "W-STAGE-ROWS-NO-KV"


def check_stage_rows_have_kv(form: Optional[StageForm],
                             cells: Dict[Tuple[int, int], StageCell],
                             per_rank: Optional[bool] = None) -> None:
    """#239 S3g floor, before the capture counts anything: a rank the launcher
    gave its own stage rows (``form.rows_on`` > 0) must have a KV tensor born
    trimmed to S0 -- the cells are its stages. Without cells no capture floor
    exists, the captured step counts the stage rows OFF while the planner
    priced them ON in S0 (rc12z30b 28.09. 20:09:07, TP2: C 45 x 2 waves < 92
    ids, 'Step ids exceed the LRU rows plus the staging rows'). Named here
    instead of that anonymous death inside the capture. Only S3g's per-rank
    rows (FLLIPER_PDFLIP_D_KV_STAGE_ROWS_BY_RANK) are a rank's OWN rows; without
    them ``rows_on`` is the attention host's (#251c), which a Form A worker
    reads but never carries (its seat rows are 0)."""
    if per_rank is None:
        from flliper.srt.environ import envs

        per_rank = bool(_ints(envs.FLLIPER_PDFLIP_D_KV_STAGE_ROWS_BY_RANK.get()))
    if form is None or not per_rank or int(form.rows_on) <= 0 or cells:
        return
    raise PdFlipDSeatVramRefused(
        "%s %s: this rank carries %d stage rows (FLLIPER_PDFLIP_D_KV_STAGE_ROWS_BY_RANK / "
        "_ROWS) but no KV tensor was born trimmed to S0 -- nothing here grows at a stage, "
        "and the captured decode steps would count those rows OFF while the planner priced "
        "them ON in S0. A Form A worker holds KV only under the token cut (its FA share; "
        "never QSA keys) -- the launcher must give it no stage rows otherwise"
        % (LINE_MARK, STAGE_ROWS_NO_KV_CODE, int(form.rows_on)))


#: #251c: the model runner whose pools the capture floor is computed over
#: (``note_capture_context``, after the KV pool exists, before the capture),
#: and the floors, computed once at the first captured MoE step.
_CAPTURE: Dict[str, object] = {"runner": None, "floors": None}


def note_capture_context(runner) -> None:
    """The KV mixin, once its pools exist: the capture that follows may ask
    for the floors (``capture_floor_rows``). A no-op without a stage form."""
    if stage_form() is None:
        return
    _CAPTURE["runner"] = runner
    _CAPTURE["floors"] = None


def _capture_floor_table() -> Tuple[int, ...]:
    floors = _CAPTURE.get("floors")
    if floors is not None:
        return floors  # type: ignore[return-value]
    runner = _CAPTURE.get("runner")
    floors = ()
    if runner is not None:
        sa = getattr(runner, "server_args", None)
        cap = int(getattr(sa, "max_running_requests", 0) or 0)
        # a refusal of the form (check_form_against_cells) stops the boot here,
        # at the capture, by name -- before any phase could run on it
        ctl = SeatVram.from_runtime(cap=max(1, cap),
                                    req_to_token_pool=getattr(runner, "req_to_token_pool", None),
                                    model=getattr(runner, "model", None))
        check_stage_rows_have_kv(ctl.form, ctl.cells)
        if ctl.form is not None and ctl.cells:
            floors = capture_floors(ctl.form, ctl.cells, ctl.cap)
            logger.info("%s CAPTURE-FLOOR rows ON per bs %s (stages %s, max_by_seats %s): the "
                        "captured decode steps count these seat/stage rows as ON",
                        STAGE_MARK, list(floors), list(ctl.form.tokens),
                        form_seats_text(ctl.form))
    _CAPTURE["floors"] = floors
    return floors


def capture_floor_rows(n_ids: int, per_seat_ids: Optional[int]) -> int:
    """#251c: the rows ON the captured MoE step of ``n_ids`` routed ids may
    count on. The batch is ``ceil(n_ids / per_seat_ids)`` (verify tokens x
    top-k per seat; a draft step routes fewer ids per seat, its batch reads
    SMALLER and its floor lower -- never too high). Past the seat cap (an
    extend shape) or without a form/cells: 0, the live count."""
    if not per_seat_ids or int(per_seat_ids) <= 0 or stage_form() is None:
        return 0
    floors = _capture_floor_table()
    if not floors:
        return 0
    b = max(1, -(-int(n_ids) // int(per_seat_ids)))
    return int(floors[b - 1]) if b <= len(floors) else 0


def _form_a_worker() -> bool:
    """This process is a Form A expert worker (rank_role's installed plan)."""
    try:
        from flliper.srt.rank_role import this_rank_is_form_a_worker
    except ImportError:
        return False
    return bool(this_rank_is_form_a_worker())


def _form_a_worker_holds_kv() -> bool:
    """#239: this process is a Form A worker with a full-attention KV share."""
    try:
        from flliper.srt.rank_role import form_a_worker_holds_kv
    except ImportError:
        return False
    return bool(form_a_worker_holds_kv())


def _this_rank() -> int:
    """#239 S3g: this process's D TP rank (the rank role's, else the TP
    group's; 0 when neither is up -- a classic single-rank boot)."""
    from flliper.srt import rank_role

    if getattr(rank_role, "_INSTALLED_PLAN", None) is not None:
        return int(getattr(rank_role, "_INSTALLED_RANK", 0))
    try:
        from flliper.srt.distributed import get_tensor_model_parallel_rank

        return int(get_tensor_model_parallel_rank())
    except Exception:  # noqa: BLE001 -- no TP group: a single rank
        return 0


def kv_owner_block() -> Tuple[int, int]:
    """#239 S3g: (cp_S, cp_ratio) of this rank's owner share under the token
    cut, (0, 0) off the weighted lane -- the geometry ``KvTensorGeom`` maps a
    stage's global tokens to this rank's compacted rows with."""
    try:
        from flliper.srt.distributed.utils import uneven_dcp_owner_bounds
    except ImportError:
        return (0, 0)
    b = uneven_dcp_owner_bounds()
    if b is None:
        return (0, 0)
    S, lo, hi = (int(x) for x in b)
    return (S, hi - lo) if hi > lo else (0, 0)


def _worker_stages_here() -> bool:
    """#239 S3g: a Form A worker the launcher gave its own stage rows
    (FLLIPER_PDFLIP_D_KV_STAGE_ROWS_BY_RANK, this rank's entry > 0) stages its
    KV like the host: born at S0, grown at the wake from those rows."""
    from flliper.srt.environ import envs

    form = stage_form()
    return (form is not None and int(form.rows_on) > 0
            and bool(_ints(envs.FLLIPER_PDFLIP_D_KV_STAGE_ROWS_BY_RANK.get())))


def owner_block_for(pool_size: int) -> Tuple[int, int]:
    """#239 S3g: (cp_S, cp_ratio) when ``pool_size`` is this rank's compacted
    owner share of the top stage -- ``dcp_compact_pool_rows(top)`` = (top // S
    + 1) x ratio rows of a token-cut FA pool (a worker's, or the host's under a
    cut that gives it a share); (0, 0) for every pool that holds the whole
    context (Form A's host KV, the QSA keys, the draft KV) -- byte-identical."""
    form = stage_form()
    S, ratio = kv_owner_block()
    if form is None or S <= 0 or ratio <= 0 or ratio >= S:
        return (0, 0)
    return (S, ratio) if int(pool_size) == (int(form.tokens[-1]) // S + 1) * ratio else (0, 0)


def kv_stage_trims_here(pool_size: int) -> bool:
    """#251c: this rank trims its KV tensors to S0 at birth -- the stage form
    is armed, the pool is the top stage's, and this is NOT a Form A worker.
    rc12z11 (590fa56a02, 28.09. 09:01Z and 09:09Z): the workers' QSA keys were
    trimmed like TP0's KV. The stage pages are TP0's only (the planner prices
    them there); a worker's tensors stay H95c's, byte for byte.

    #239 S3g: a compacted token-cut FA pool (``owner_block_for``) is the top
    stage's too. A worker trims only with its own stage rows
    (``_worker_stages_here``) -- then every KV tensor it holds, its QSA keys
    included: they follow the stage as the planner books them (S0), grown at
    the wake from the worker's rows. rc12z11's fault (a worker trimming with
    no rows to grow from, and under the fill kernel) cannot recur: no rows,
    no trim; the birth syncs before a page goes (``_sync_before_unmap``)."""
    form = stage_form()
    if form is None:
        return False
    top = int(pool_size) >= form.tokens[-1] or owner_block_for(pool_size) != (0, 0)
    if not _form_a_worker():
        return top
    return top and _worker_stages_here()


def kv_stage_boot_rows(pool_size: int, page_size: int) -> Optional[int]:
    """#251c: the rows a stage-form pool keeps mapped from its birth to the
    first wake (S0's tokens + the page), None where this rank trims nothing.
    #239 S3g: a compacted owner-share pool keeps S0's compacted rows + the page."""
    if not kv_stage_trims_here(pool_size):
        return None
    form = stage_form()
    return stage_mapped_rows(pool_size, int(form.tokens[0]), page_size)


def bound_stage_rows(pool, rows: int) -> None:
    """#251c: tell a KV pool (and a hybrid pool's full-attention pool) which
    rows its stage keeps mapped -- through MHATokenToKVPool.set_stage_backed_rows,
    whose ``safe_zero_rows`` / ``_committed_row_bound`` read it. Never assign
    ``safe_zero_rows``: it is a property (rc12z13 died on exactly that)."""
    from flliper.srt.mem_cache.memory_pool import HybridLinearKVPool, MHATokenToKVPool

    subs = [pool]
    if isinstance(pool, HybridLinearKVPool):
        subs.append(pool.full_kv_pool)
    for sub in subs:
        if isinstance(sub, MHATokenToKVPool):
            sub.set_stage_backed_rows(rows)


def stage_mapped_rows(pool_size: int, tokens: int, page_size: int) -> int:
    """#239 S3g: the rows a pool of ``pool_size`` keeps mapped at a stage of
    ``tokens`` -- the stage's tokens + the page (#251c), or on a token-cut
    FA pool (``owner_block_for``) the stage's compacted rows + the page."""
    S, ratio = owner_block_for(pool_size)
    if S > 0:
        return (int(tokens) // S + 1) * ratio + int(page_size)
    return int(tokens) + int(page_size)


def bound_stage_tokens(pool, tokens: int, page_size: int) -> None:
    """#239 S3g: ``bound_stage_rows`` with each (sub-)pool's own geometry --
    a hybrid pool's FA sub-pool may be a compacted owner share while the
    draft's pool holds the whole context."""
    from flliper.srt.mem_cache.memory_pool import HybridLinearKVPool, MHATokenToKVPool

    subs = [pool]
    if isinstance(pool, HybridLinearKVPool):
        subs.append(pool.full_kv_pool)
    for sub in subs:
        if isinstance(sub, MHATokenToKVPool):
            sub.set_stage_backed_rows(
                stage_mapped_rows(int(getattr(sub, "size", 0) or 0), tokens, page_size))


def kv_stage_pool_tokens(max_tokens: int, *, is_form_a_worker: bool = False,
                         is_draft_worker: bool = False, draft_shares_slots: bool = True) -> int:
    """#251c ``_config_from_budget``: the KV pool's rows are the TOP stage's
    (virtual -- the graphs keep one address range); the pages behind them are
    S0's (``kv_stage_born``). The budget must hold S0 on the attention host:
    below it the boot form the planner priced does not fit, refused by name.
    A Form A worker holds no KV: its (replicated) allocator takes the same
    rows without a budget check -- its page ids stay in step with TP0's --
    and none of its tensors is trimmed (``kv_stage_trims_here``). Off:
    ``max_tokens`` unchanged."""
    form = stage_form()
    if form is None:
        return int(max_tokens)
    if is_draft_worker and not draft_shares_slots:
        # review 28.09. (#251c (c)): a draft pool is sized at the top stage and
        # born with only S0's pages, but only the TARGET's allocator is capped
        # to S0 (kv_stage_boot_cap). That is sound only while the draft writes
        # at the target's slot ids (MTP/EAGLE share the allocator). A draft
        # with its OWN allocator (the DFlash solo host) would hand out ids above
        # S0 onto unmapped pages -- refused by name at the boot instead.
        raise PdFlipDSeatVramRefused(
            "%s: the KV stage form needs the draft to write at the target's slot ids; this "
            "draft worker has its own allocator (solo host), whose ids are not capped to "
            "stage S0 -- turn the stage form off (FLLIPER_PDFLIP_D_KV_STAGE_TOKENS) for this "
            "draft form" % LINE_MARK)
    is_form_a_worker = bool(is_form_a_worker) or _form_a_worker()
    if is_form_a_worker and _form_a_worker_holds_kv() and not _worker_stages_here():
        # 28.09. (M1 prep): this worker holds full-attention KV under the token
        # cut; its compacted FA pool follows max_total, and nothing here trims
        # it -- the top stage would be mapped whole (+100 % FA KV at S0 x 2).
        raise PdFlipDSeatVramRefused(
            "%s: the KV stage form %s reached a Form A worker that holds full-attention "
            "KV under the token cut -- its pool would be mapped at the top stage, "
            "untrimmed; stages per KV rank are S3g. Run the cut with one stage "
            "(FLLIPER_PDFLIP_D_KV_STAGE_TOKENS=<kv tokens>) until then"
            % (LINE_MARK, list(form.tokens)))
    if not is_form_a_worker and int(max_tokens) < form.tokens[0]:
        raise PdFlipDSeatVramRefused(
            "%s: the KV budget holds %d tokens, below stage S0 = %d -- the boot form "
            "the planner priced does not fit this rank" % (LINE_MARK, int(max_tokens),
                                                         form.tokens[0]))
    return int(form.tokens[-1])


#: the attention host's KV tensors, trimmed to S0 the moment they are born
#: (``kv_stage_born``), in birth order: (data_ptr, geometry). Process-local;
#: TP0 only (a Form A worker trims none, ``kv_stage_trims_here``).
_KV_BORN: List[Tuple[int, KvTensorGeom]] = []
#: the span plan each of them was born with, keyed (data_ptr, geometry, stage
#: tokens, granule) -- a key that also names what the plan was computed from:
#: the saver mapped one extent per range, the controller checks them at its build
_KV_BORN_SPANS: Dict[tuple, Tuple[Tuple[int, int], ...]] = {}
#: S77-A: the born tensors themselves, keyed by data_ptr (``KvPageCopier`` moves
#: pages in exactly the tensors whose pages the stage form unmaps)
_KV_BORN_T: Dict[int, object] = {}


def kv_stage_born(t, *, pool_size: int, page_size: int, name: str = "kv",
                  tokens_per_slot: int = 1, layers: int = 1, slots: Optional[int] = None,
                  spans: Optional["TmsSpans"] = None, granule: Optional[int] = None):
    """#251c: a KV tensor of a stage-form pool keeps its TOP-stage range but
    only S0's pages -- trimmed right here, one tensor at a time, so the boot
    never holds the top stage's pages (~3.5 GiB on NF) even for an instant.
    Every other tensor passes through untouched (no form, another pool size).
    ``layers``/``slots``: a layer-major ``[L, slots x ...]`` tensor (the QSA
    compressed keys: one slot per ``tokens_per_slot`` tokens)."""
    form = stage_form()
    if form is None or t is None or not t.numel() or not kv_stage_trims_here(pool_size):
        return t
    spans = tms() if spans is None else spans
    info = spans.info(t.data_ptr()) if spans.available else None
    if info is None:
        raise PdFlipDSeatVramRefused(
            "%s: the KV tensor %s (%.1f MiB) is not a saver allocation -- its "
            "top-stage pages could not be unmapped and would sit on the card "
            "unpriced. Turn FLLIPER_OPT_PDFLIP_D_SEAT_VRAM off for this boot."
            % (LINE_MARK, name, t.numel() * t.element_size() / _MIB))
    g = int(granule or granule_for(t.device))
    n_slots = int(slots if slots is not None else t.shape[0])
    slot_bytes = (t.numel() * t.element_size()) // max(1, int(layers) * n_slots)
    geom = KvTensorGeom(
        SlotTensorGeom(name, int(layers), n_slots, int(slot_bytes), int(info.size)),
        token_ratio=int(tokens_per_slot), token_pad=int(page_size),
        # #239 S3g: a token-cut FA pool holds its owner share, compacted
        owner_block=owner_block_for(pool_size))
    # the tensor was born by torch.zeros: its fill kernel may still be in
    # flight. Unmapping under it is a GPU fault that surfaces at the next
    # sync (rc12z11 TP1/TP2: KvRowCap._apply, exit -6) or as a dead context
    # at the next launch (rc12z11 TP0: segfault in fill_kernel_cuda). H95c's
    # apply_stage synchronizes before a page goes; the birth must as well.
    _sync_before_unmap(t)
    # PA review of 1bab093912: born UNCUT, the per-layer ranges of a small
    # layer-major tensor (the QSA keys: layers far below the 2 MiB granule)
    # coalesce into one extent across the stage lattice, and the first live
    # KV move (S0 -> S1 with the KV mapped) stops with W-SEAT-WIPE mid-run.
    # Born at the lattice (``SeatVram.kv_spans``' cuts), every extent is a cell.
    born = slot_spans(geom.geom, geom.slots_for(form.tokens[0]), g,
                      cuts=[geom.slots_for(tk) for tk in form.tokens])
    rc = spans.set_spans(t.data_ptr(), born, now=True)
    if rc != 0:
        raise PdFlipDSeatVramRefused("%s: trimming the KV tensor %s to stage S0 failed "
                                   "(tms_set_spans rc=%d)" % (LINE_MARK, name, rc))
    _KV_BORN.append((int(t.data_ptr()), geom))
    _KV_BORN_T[int(t.data_ptr())] = t
    _KV_BORN_SPANS[(int(t.data_ptr()), geom, tuple(form.tokens), g)] = tuple(born)
    return t


def kv_stage_boot_cap(allocator, page_size: int) -> Optional[int]:
    """#251c: the allocator of a stage-form pool hands out only S0's pages
    until the first wake says otherwise (every rank, replicated). Returns the
    page cap, None when off."""
    form = stage_form()
    if form is None or allocator is None:
        return None
    pages = _engage_kv_cap(allocator, form.tokens[0], page_size)
    log_kv_stage_boot(form, allocator, pages)
    return pages


def log_kv_stage_boot(form: "StageForm", allocator, cap_pages: int) -> None:
    """One line per rank at the boot (rc12z13: no line said whether TP0 had
    trimmed to S0 and the workers had not): the form, whether this rank trims,
    the KV tensors born trimmed with their mapped bytes against the VA the
    top stage reserves, and the allocator cap."""
    spans = tms()
    va = mapped = 0
    for ptr, _geom in _KV_BORN:
        info = spans.info(ptr) if spans.available else None
        if info is not None:
            va += int(info.size)
            mapped += int(info.mapped)
    logger.info(
        "#251c KV-STAGE form=%s trims_here=%s born=%d mapped=%.1f/%.1f MiB cap_pages=%d/%d "
        "(S0 %d tokens; a Form A worker trims only its token-cut FA pool, S3g)",
        "/".join(str(t) for t in form.tokens),
        bool(_KV_BORN) or kv_stage_trims_here(form.tokens[-1]), len(_KV_BORN),
        mapped / _MIB, va / _MIB,
        int(cap_pages), int(getattr(allocator, "num_pages", 0) or 0), form.tokens[0])


def _sync_before_unmap(t) -> None:
    """Every kernel that may still write ``t`` is done before a page of it is
    unmapped (a CPU tensor -- the unit tests' -- has none)."""
    if getattr(t, "is_cuda", False):
        import torch

        torch.cuda.synchronize(t.device)


def _engage_kv_cap(allocator, tokens: int, page_size: int) -> int:
    from flliper.srt.managers.kv_backing_relief import KvRowCap

    cap = getattr(allocator, "_pdflip_kv_stage_cap", None)
    if cap is None:
        cap = KvRowCap(allocator)
        allocator._pdflip_kv_stage_cap = cap
    pages = int(tokens) // max(1, int(page_size))
    if pages >= int(getattr(allocator, "num_pages", pages)):
        if cap.engaged:
            cap.release()
        return pages
    # KvRowCap.engage only ever withholds MORE (it moves free ids above the
    # cap out of the free lists); a stage that rises must first hand back
    # what the lower cap held, then withhold above the new one
    if cap.engaged and cap.cap is not None and pages > int(cap.cap):
        cap.release()
    cap.engage(pages)
    return pages


def max_live_page(allocator) -> int:
    """The highest page id a request holds (0 = none): every id that is in
    no free list and not withheld. Replicated like the allocator itself."""
    import torch

    n = int(getattr(allocator, "num_pages", 0) or 0)
    if n <= 0:
        return 0
    live = torch.ones(n + 1, dtype=torch.bool)
    live[0] = False
    for name in ("free_pages", "release_pages"):
        ids = getattr(allocator, name, None)
        if ids is not None and ids.numel():
            live[ids.detach().to("cpu", torch.int64)] = False
    cap = getattr(allocator, "_pdflip_kv_stage_cap", None)
    held = getattr(cap, "_withheld", None) if cap is not None else None
    if held is not None and held.numel():
        live[held.to("cpu", torch.int64)] = False
    idx = torch.nonzero(live).flatten()
    return int(idx.max()) if idx.numel() else 0


class KvCopyRefused(RuntimeError):
    """S77-A: a KV page move this rank cannot carry out completely (named)."""


def page_slot_block(geom: "KvTensorGeom", page_size: int) -> Tuple[int, int]:
    """S77-A: (slots per page, slot of page 0's first token) -> page ``p`` lives in
    slots ``[p * per, (p + 1) * per)`` of every layer of a born KV tensor.

    * a paged K/V buffer (``token_ratio`` 1): one slot per token;
    * QSA compressed keys (``token_ratio`` = ratio): ``page // ratio`` slots;
    * a token-cut FA owner share (#239 S3g, ``owner_block`` (S, ratio)): global slot
      L at row (L // S) * ratio + (L % S - lo) -- a page of P global tokens is the
      contiguous block of (P // S) * ratio rows from (p * P // S) * ratio, whatever
      ``lo`` is; the owner class of a token (L % S) is its offset in the page, so a
      page moved whole keeps every token on the rank that owns it.
    The page must be a multiple of the ratio / of S, else the move is refused."""
    P = int(page_size)
    S, ratio = (int(x) for x in geom.owner_block)
    if S > 0 and ratio > 0:
        if P % S:
            raise KvCopyRefused("page %d not a multiple of the owner block %d (%s)"
                                % (P, S, geom.geom.name))
        return (P // S) * ratio, 0
    tps = max(1, int(geom.token_ratio))
    if P % tps:
        raise KvCopyRefused("page %d not a multiple of %d tokens per slot (%s)"
                            % (P, tps, geom.geom.name))
    return P // tps, 0


class KvPageCopier:
    """S77-A: moves whole KV pages (device copy) in EVERY tensor the stage form
    trims on this rank (``_KV_BORN``: the FA K/V or their owner share, the QSA
    compressed keys, the draft K/V) -- exactly the tensors whose pages a stage
    shrink unmaps, viewed as bytes ``[layers, slots, slot_bytes]`` with the
    geometry they were born with."""

    def __init__(self, born: Sequence[Tuple[int, "KvTensorGeom"]], tensors: Dict[int, object],
                 page_size: int):
        self.page_size = int(page_size)
        self.items = []
        for ptr, geom in born:
            t = tensors.get(int(ptr))
            if t is None:
                raise KvCopyRefused("born KV tensor %s has no tensor reference" % geom.geom.name)
            g = geom.geom
            if int(t.numel()) * int(t.element_size()) != int(g.layers) * int(g.slots) * int(g.slot_bytes):
                raise KvCopyRefused("born KV tensor %s is not [layers, slots, slot_bytes] (%d B vs "
                                    "%d x %d x %d)" % (g.name, t.numel() * t.element_size(),
                                                       g.layers, g.slots, g.slot_bytes))
            if not t.is_contiguous():
                raise KvCopyRefused("born KV tensor %s is not contiguous" % g.name)
            per, base = page_slot_block(geom, self.page_size)
            self.items.append((t, geom, per, base))

    @classmethod
    def from_born(cls, page_size: int) -> "KvPageCopier":
        return cls(list(_KV_BORN), _KV_BORN_T, page_size)

    @classmethod
    def from_buffers(cls, bufs, page_size: int) -> "KvPageCopier":
        """S77-A N1: the buffers ``kv_page_buffers`` found, each with its pool's geometry."""
        born = [(int(t.data_ptr()), g) for _n, t, g in bufs]
        return cls(born, {int(t.data_ptr()): t for _n, t, _g in bufs}, page_size)

    def ptrs(self) -> set:
        return {int(t.data_ptr()) for t, _g, _p, _b in self.items}

    def copy(self, src_pages: Sequence[int], dst_pages: Sequence[int]) -> int:
        """Copy every page ``src_pages[i]`` onto ``dst_pages[i]`` in every tensor
        (gather then scatter; the two sets are disjoint). Returns bytes copied.
        Ordered on the current stream; the caller synchronizes before it re-points
        the tree and frees the sources."""
        import torch

        if len(src_pages) != len(dst_pages):
            raise KvCopyRefused("%d sources for %d destinations" % (len(src_pages), len(dst_pages)))
        if not src_pages:
            return 0
        nbytes = 0
        for t, geom, per, base in self.items:
            g = geom.geom
            u8 = t.view(-1).view(torch.uint8).view(int(g.layers), int(g.slots), int(g.slot_bytes))
            off = torch.arange(per, dtype=torch.int64)
            src = (torch.tensor([int(p) for p in src_pages], dtype=torch.int64)[:, None] * per
                   + base + off[None, :]).reshape(-1)
            dst = (torch.tensor([int(p) for p in dst_pages], dtype=torch.int64)[:, None] * per
                   + base + off[None, :]).reshape(-1)
            hi = int(max(int(src.max()), int(dst.max())))
            if hi >= int(g.slots):
                raise KvCopyRefused("page slot %d beyond %s's %d slots" % (hi, g.name, g.slots))
            src, dst = src.to(t.device), dst.to(t.device)
            u8[:, dst] = u8[:, src]
            nbytes += int(src.numel()) * int(g.layers) * int(g.slot_bytes)
        return nbytes


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
            from flliper.srt.utils.torch_memory_saver_adapter import _pdflip_ring_symbol

            symbol = _pdflip_ring_symbol
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

        from flliper.srt.mem_cache.kv_vmm_backing import query_granularity

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
    rank = int(getattr(layer, "moe_tp_rank", 0) or 0)
    x = extra_rows_for_rank(rank)
    if x <= 0:
        return 0
    if not tms().available:
        logger.warning(
            "%s: FLLIPER_PDFLIP_D_SEAT_EXPERT_ROWS asks for %d seat rows but the running "
            "saver has no span map (tms_set_spans, tms_csrc patch 3) -- the bank stays "
            "at its cap form, no seat rows", LINE_MARK, x)
        return 0
    # D-TRANSIENT-LEND: a virtual head above the planner's seat rows (VA only);
    # the capture floor and the stage cells never count it (seat_lend_head)
    head = lend_head_rows_for_rank(rank)
    try:
        layer._pdflip_seat_lend_head = int(head)
    except AttributeError:  # a slotted test double: no head
        head = 0
    return int(x) + int(head)


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
        raise PdFlipDSeatVramRefused(
            "%s: the seat-form expert bank %s (%d + %d rows x %d B) is not one saver "
            "allocation (%s, padded %d B, in tag pool %s) -- its %d seat rows (%.1f MiB) "
            "could not be unmapped and would sit on the card unpriced. Turn "
            "FLLIPER_OPT_PDFLIP_D_SEAT_VRAM off for this boot." % (
                LINE_MARK, name, int(rows), int(extra), row_bytes,
                "no allocation at this address" if info is None
                else "allocation of %d B" % info.size, padded, in_tag_pool,
                int(extra), int(extra) * row_bytes / _MIB))
    rc = spans.set_spans(flat.data_ptr(), row_spans(
        RowTensorGeom(name, int(rows), rows_max, row_bytes, padded), int(rows), g), now=True)
    if rc != 0:
        raise PdFlipDSeatVramRefused(
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
    #: #251c: the phase's KV stage (None without a stage form), its tokens,
    #: the wake's demand and whether it exceeds the highest usable stage
    stage: Optional[int] = None
    stage_tokens: Optional[int] = None
    demand: Optional[int] = None
    over: bool = False
    #: D-SEAT-REWAKE: this rank's measured ms of the wake's page apply
    apply_ms: Optional[float] = None


@dataclass
class PhaseApplied:
    """RANK-LOCAL: the pages this rank mapped for the phase."""

    n: int
    extra_rows: int
    mamba_mapped: int
    expert_mapped: int
    cap_mapped: int
    note: str = ""
    #: #251c: the stage this rank mapped and its KV prefix
    stage: Optional[int] = None
    kv_mapped: int = 0


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
    #: NF1c 09291739: the GDN slots that HAVE pages -- the keep of the last
    #: plan actually set on the (paused) pool, None = every slot. A live apply
    #: never re-plans the pool, so it never moves this (``_plan_mamba``).
    mamba_keep: Optional[int] = None
    #: #251c: the KV tensors born trimmed (``kv_stage_born``), the stage form,
    #: the KV pools whose flush must stay in the mapped prefix, the cells
    kv_tensors: List[_Managed] = field(default_factory=list)
    form: Optional[StageForm] = None
    kv_pools: List[object] = field(default_factory=list)
    cells: Dict[Tuple[int, int], StageCell] = field(default_factory=dict)
    #: the expert rows ON as last applied (the bank is born at k = 0)
    rows_on: int = 0
    #: S1-Wisch (rc12z17 28.09.): every k a phase of this rank can take -- the
    #: bank's span plans are cut at ``rows_boot + k`` for all of them
    row_cut_ks: Tuple[int, ...] = ()
    #: D-MEM-SCHED (27B condition 1): rows displaced / turned on again by a
    #: live apply, and the milliseconds its stream syncs took
    evicted_rows: int = 0
    refilled_rows: int = 0
    sync_ms: float = 0.0
    #: coldest-first moves (27B): bank bytes copied and the milliseconds the
    #: copy + table writes took up to the sync that lands them (> 1 ms per
    #: round: the copy goes to the copy stream with an event)
    bytes_copied: int = 0
    copy_ms: float = 0.0
    #: D-SEAT-REWAKE: the last live re-seat's pause in parts (ms)
    last_reseat_parts: Dict[str, float] = field(default_factory=dict)
    #: D-TRANSIENT-LEND: the virtual head above the planner's seat rows (VA
    #: only, never in a cell), the bank's top, the rows ON a lend started
    #: from (None = nothing lent) and the rows lent / given back in total
    lend_head: int = 0
    x_max: int = 0
    lent_from: Optional[int] = None
    lent_rows: int = 0
    returned_rows: int = 0
    #: the last span plan set on each managed allocation; after its tag's
    #: resume or a live apply it IS the saver's extents (one per range)
    spans_by_ptr: Dict[int, Tuple[Tuple[int, int], ...]] = field(default_factory=dict)

    def __post_init__(self):
        x_max = min((int(getattr(c, "seat_rows", 0)) for c in self.caches), default=0)
        # D-TRANSIENT-LEND: the head is VA for a lend only -- the phase table
        # and the stage cells see the planner's rows (byte-identical form)
        head = min((int(getattr(c, "seat_lend_head", 0) or 0) for c in self.caches), default=0)
        self.lend_head = max(0, min(int(head), x_max))
        self.x_max = int(x_max)
        x_cells = x_max - self.lend_head
        self.rows = seat_vram_rows(
            [m.geom for m in self.slot_tensors], [m.geom for m in self.row_tensors],
            cap=self.cap, pool_size=self.pool_size, extra_max=x_cells, granule=self.granule)
        if self.form is not None and self.kv_tensors:
            self.cells = stage_vram_cells(
                [m.geom for m in self.slot_tensors], [m.geom for m in self.row_tensors],
                [m.geom for m in self.kv_tensors], cap=self.cap, pool_size=self.pool_size,
                extra_max=x_cells, stage_tokens=self.form.tokens,
                boot_rows_on=self.form.rows_on, granule=self.granule)
            check_form_against_cells(self.form, self.cells, self.cap)
        # F5 (27B 28.09.): a rank with its own stage rows but no KV cells is
        # refused at the wake too, not only at the capture
        check_stage_rows_have_kv(self.form, self.cells)
        ks = {0, int(self.form.rows_on) if self.form is not None else 0}
        ks.update(int(r.extra_rows) for r in self.rows)
        for c in self.cells.values():
            ks.add(int(c.extra_rows))
            if int(c.rows_full) >= 0:
                ks.add(int(c.rows_full))
        if lend_armed():
            # D-TRANSIENT-LEND: a lend ends on the lattice, so it and its
            # return release whole cells only (S1-Wisch) from any phase's k
            from flliper.srt.environ import envs

            ks.update(lend_lattice(x_max, int(envs.FLLIPER_PDFLIP_D_LEND_STEP_ROWS.get())))
        self.row_cut_ks = tuple(sorted(k for k in ks if 0 <= k <= max(0, x_max)))
        # the extents the birth trims left (seat_expert_buffer: the cap form's
        # rows; kv_stage_born: stage S0), both one range per span, no cuts
        for m in self.row_tensors:
            self.spans_by_ptr.setdefault(int(m.ptr), row_spans(m.geom, m.geom.rows_boot, self.granule))
        if self.form is not None:
            for m in self.kv_tensors:
                born = _KV_BORN_SPANS.get(
                    (int(m.ptr), m.geom, tuple(self.form.tokens), int(self.granule)))
                if born is None:
                    born = slot_spans(m.geom.geom, m.geom.slots_for(self.form.tokens[0]),
                                      self.granule)
                self.spans_by_ptr.setdefault(int(m.ptr), tuple(born))
                # PA review: a birth extent across the lattice would stop the
                # first live KV move mid-run -- stop at the build instead
                cut = straddling(self.spans_by_ptr[int(m.ptr)],
                                 self.kv_spans(m, self.form.tokens[0]))
                if cut is not None:
                    raise PdFlipDSeatVramWipe(
                        "%s %s at the controller's build: the KV tensor %s was born with the "
                        "extent [%d, %d) across the stage lattice -- the first live KV move "
                        "would release it and map it fresh. Stopped before the first phase."
                        % (LINE_MARK, WIPE_CODE, m.geom.geom.name, cut[0], cut[1]))

    def bank_spans(self, m: "_Managed", k: int) -> Tuple[Tuple[int, int], ...]:
        """The bank tensor's plan for ``k`` rows ON, cut at every k a phase
        can take (``row_spans`` cuts)."""
        return row_spans(m.geom, m.geom.rows_boot + int(k), self.granule,
                         cuts=[m.geom.rows_boot + c for c in self.row_cut_ks])

    def kv_spans(self, m: "_Managed", tokens: int) -> Tuple[Tuple[int, int], ...]:
        """A KV tensor's plan for a stage of ``tokens``, cut at every stage."""
        cuts = [m.geom.slots_for(t) for t in self.form.tokens] if self.form is not None else []
        return slot_spans(m.geom.geom, m.geom.slots_for(tokens), self.granule, cuts=cuts)

    def refuse_wipes(self, plans: Sequence[Tuple[int, str, Tuple[Tuple[int, int], ...], bool]]) -> None:
        """Before this apply moves anything: a live plan that would release an
        extent holding bytes the plan keeps stops by name (the saver maps a
        released range FRESH -- rc12z17's wiped bank). This apply's plans,
        tables and pages are untouched then; what the wake did before the
        apply (``on_wake``: the phase limit, the KV cap) stays done."""
        for ptr, name, spans, live in plans:
            prev = self.spans_by_ptr.get(int(ptr))
            if not live or prev is None:
                continue
            cut = straddling(prev, spans)
            if cut is not None:
                raise PdFlipDSeatVramWipe(
                    "%s %s: a live apply would release the mapped extent [%d, %d) of %s, "
                    "whose bytes the new plan keeps -- the saver maps a released range "
                    "fresh (rc12z17 10:51:24: the whole expert bank). No plan, table or "
                    "page of this apply moved; the wake's phase limit and KV cap were "
                    "already set." % (LINE_MARK, WIPE_CODE, cut[0], cut[1], name))

    def set_plan(self, ptr: int, name: str, spans: Tuple[Tuple[int, int], ...], *,
                 live: bool, census: Optional[Dict[str, int]] = None) -> None:
        """``tms_set_spans`` of one managed allocation, recorded; a live one
        counts its extents kept / cells freed / cells mapped into ``census``."""
        prev = self.spans_by_ptr.get(int(ptr))
        if live and census is not None and prev is not None:
            kept = [e for e in prev if any(lo <= e[0] and e[1] <= hi for lo, hi in spans)]
            census["kept"] += len(kept)
            census["freed"] += sum(1 for a, b in prev if all(b <= lo or a >= hi for lo, hi in spans))
            census["mapped"] += sum(1 for r in spans if r not in kept)
            census["tensors"] += 1
        rc = self.spans.set_spans(ptr, spans, now=live)
        if rc != 0:
            cls = PdFlipDSeatVramNoMemory if (live and rc == TMS_RC_OUT_OF_MEMORY) else PdFlipDSeatVramRefused
            raise cls("%s: tms_set_spans(%s, now=%s) rc=%d" % (LINE_MARK, name, live, rc))
        self.spans_by_ptr[int(ptr)] = tuple(spans)

    @classmethod
    def from_runtime(cls, *, cap: int, req_to_token_pool, model, spans: Optional[TmsSpans] = None,
                     granule: Optional[int] = None, kv_pools: Sequence[object] = ()) -> "SeatVram":
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
        form = stage_form()
        kv_tensors = [_Managed(ptr, geom) for ptr, geom in _KV_BORN] if form is not None else []
        return cls(cap=int(cap), pool_size=pool_size, slot_tensors=slot_tensors,
                   row_tensors=row_tensors, caches=caches, fixed_bytes=int(fixed),
                   spans=spans, granule=g, mamba_pool=pool, kv_tensors=kv_tensors,
                   form=form, kv_pools=list(kv_pools))

    def row_for(self, n: int) -> SeatVramRow:
        return self.rows[max(1, min(int(n), self.cap)) - 1]

    def _plan_mamba(self, infos, keep: int) -> None:
        """The GDN pool's plan for ``keep`` slots -- set on a PAUSED pool only
        (mapped at its tag's resume); a live pool keeps the pages it has.

        ``MambaPool.reset_state`` (the flush at the wake and before the next
        sleep) zeroes only ``[:, :_pdflip_seat_keep]``, so that value must name
        the pages the pool HAS, not the form this apply wanted: NF1c 09291739
        the D-MEM-SCHED tick's live S1 -> S0 wrote None ("keeps its cap form")
        over a pool mapped for one seat (slots 1..7); the sleep flush zeroed
        the whole temporal -> FillFunctor<BFloat16> device exception on the
        unmapped tail, 58 GB GPU coredump, abort in torch.cuda.synchronize."""
        planned = False
        rewake = _rewake_on()
        for m, info in zip(self.slot_tensors, infos):
            if info is not None and not info.active:
                # D-SEAT-REWAKE: the plan is cut at every keep a phase can take,
                # so a live shrink frees whole cells and never wipes a kept one
                sp = (slot_spans(m.geom, keep, self.granule, cuts=self.mamba_cut_keeps())
                      if rewake else slot_spans(m.geom, keep, self.granule))
                rc = self.spans.set_spans(m.ptr, sp, now=False)
                if rc != 0:
                    raise PdFlipDSeatVramRefused("%s: tms_set_spans(%s) rc=%d"
                                               % (LINE_MARK, m.geom.name, rc))
                if rewake:
                    self.spans_by_ptr[int(m.ptr)] = tuple(sp)
                planned = True
        if planned:
            self.mamba_keep = None if keep > self.pool_size else int(keep)
        if self.mamba_pool is not None and self.slot_tensors:
            self.mamba_pool._pdflip_seat_keep = self.mamba_keep

    def lend_keep_target(self, k_cell: int, kv_need_bytes: int, keep_need_bytes: Optional[int],
                         card: Optional["CardRoom"] = None) -> int:
        """S106 STAGE-KEEP: the rows a live phase move keeps ON of an open D-TRANSIENT-LEND.

        Metal (NF rc22-rc26, 20.10., D logs ``..._1020_094747`` .. ``..._1020_130752``): every
        D-MEM-SCHED stage move ended the lend (``lent_from = None``) -- rows OFF coldest first,
        sync, the lent cells released -- and the lend verdict mapped the SAME cells again a
        few rounds later (rc26 TP0: 134 re-lends after a lend-ending stage move, 133 of them
        back at or above the rows the move had dropped; group gap 82.4 s in 0.44 h). The
        card held the lend beside the new stage all along.

        Rule: the largest lattice k in ``[k_cell, rows_on]`` (no straddled extent) whose
        released cells leave the card -- measured now (``cudaMemGetInfo``) -- at least
        ``keep_need_bytes`` free after the move maps ``kv_need_bytes`` fresh:
        ``free + release(k) - kv_need >= keep_need``. ``keep_need`` is the lend's own price
        (``d_transient_lend.stage_keep_need_bytes``: the lend floor = the card ledger's
        near-OOM edge, plus the KEEP-SHORT price of a pending short extend); no new number,
        no reserve. ``k_cell`` (the old behaviour: the lend ends) when nothing is lent, no
        price is given (a big or unpriced extend pending: its full return already ran),
        or no card is measured (CPU, no context). Pure arithmetic over one card reading."""
        cur = int(self.rows_on)
        k_cell = int(k_cell)
        if self.lent_from is None or keep_need_bytes is None or cur <= k_cell:
            return k_cell
        if not self.row_tensors or not self._bank_live():
            return k_cell
        card = _CARD if card is None else card
        free = card.free()
        if free is None:
            return k_cell
        for c in sorted({cur} | {int(x) for x in self.row_cut_ks if k_cell < int(x) < cur}, reverse=True):
            if c < cur and self._bank_straddles(c):
                continue
            rel = self.bank_release_bytes(c) if c < cur else 0
            if int(free) + int(rel) - int(kv_need_bytes) >= int(keep_need_bytes):
                return int(c)
        return k_cell

    def _gdn_map_bytes(self, mamba_plans) -> int:
        """S106: the bytes a LIVE re-seat maps fresh for the GDN pool (``kv_map_bytes``' rule)."""
        total = 0
        for m, sp in mamba_plans:
            info = self.spans.info(m.ptr)
            if info is None or not info.active:
                continue
            total += _fresh_bytes(self.spans_by_ptr.get(int(m.ptr)), sp, int(info.mapped))
        return int(total)

    def _log_lend_keep(self, *, move: str, rows_from: int, kept: int, k_cell: int, kv_need: int,
                       keep_need: Optional[int]) -> None:
        """S106 line: what the phase move did with an open lend (one per move that met one)."""
        logger.info(
            "%s rank=%d stage-keep %s rows_on %d->%d phase_rows=%d lent_from=%s kv_need_mib=%d "
            "keep_need_mib=%s -- %s (S106)", LEND_MARK, _this_rank(), move, rows_from, kept, k_cell,
            "-" if self.lent_from is None else int(self.lent_from), int(int(kv_need) / _MIB),
            "-" if keep_need is None else int(int(keep_need) / _MIB),
            "the lend rides on above the new phase rows" if kept > k_cell
            else "the lend ends with the move (no price, or the card does not hold it)")

    def apply_stage(self, n: int, stage: int, keep_lend_need: Optional[int] = None) -> PhaseApplied:
        """#251c: the pages of a phase of ``n`` seats at KV stage ``stage``
        (``stage_vram_cells``): the GDN slots, the KV prefix and the expert
        rows ON move together, the total never above the boot form. Paused
        allocations get a plan (mapped at their tag's resume), mapped ones move
        in place. A LIVE bank that shrinks turns its rows OFF in the tables
        first and synchronizes before a page goes (a row a kernel may still
        read is never unmapped); one that grows maps first, then turns ON.
        Every plan is cut at the lattice of all phases (``bank_spans`` /
        ``kv_spans``): a live move releases only whole cells above the new
        end -- rc12z17's single-extent bank was released WHOLE by the shrink
        33 -> 23 and mapped fresh (``refuse_wipes`` stops that by name)."""
        import torch

        n = max(1, min(int(n), self.cap))
        j = max(0, min(int(stage), len(self.form.tokens) - 1))
        cell, k, keep, notes, infos = self._stage_rows(n, j)
        tokens = int(self.form.tokens[j])
        # S106 STAGE-KEEP: an open lend rides on above the new phase rows as far as the
        # card holds it beside the KV this move maps (``keep_lend_need`` = the lend's price;
        # None = the lend ends with the move, as before)
        k_cell, lent_base, rows_from = int(k), None, int(self.rows_on)
        if self.lent_from is not None and keep_lend_need is not None and rows_from > k_cell:
            kv_need = self.kv_map_bytes(tokens)
            k_keep = self.lend_keep_target(k_cell, kv_need, keep_lend_need)
            self._log_lend_keep(move="S%s->S%d" % ("-" if self.applied is None else self.applied.stage, j),
                                rows_from=rows_from, kept=k_keep, k_cell=k_cell, kv_need=kv_need,
                                keep_need=keep_lend_need)
            if k_keep > k_cell:
                k, lent_base = int(k_keep), k_cell
                notes.append("S106: the lend keeps rows %d above the phase's %d" % (k_keep, k_cell))
        # S77-B (NF rc17 1019_171636, D TP0 18:07:42Z): a live KV grow maps its
        # pages only into bytes the card HAS -- measured, the allocator cache
        # trimmed, this rank's own expert rows released first if still short
        room = self.grow_room(n, j, k=k)
        if room is not None:
            if not room.ok:
                raise PdFlipDSeatVramNoMemory(
                    "%s %s: S%d at n=%d maps %d KV bytes fresh; the card has %d free after "
                    "the cache trim and the rows ON down to 0 release %d -- refused before "
                    "anything moved" % (LINE_MARK, GROW_ROOM_MARK, j, n, room.need,
                                        room.free_after, room.release_all))
            if room.rows_to is not None and room.rows_to < k:
                notes.append("GROW-ROOM: rows %d -> %d pay the KV the card did not have"
                             % (k, room.rows_to))
                k = int(room.rows_to)
        return self._apply_stage_rows(n, j, cell, k, keep, notes, infos, tokens, lent_base=lent_base)

    def kv_map_bytes(self, tokens: int) -> int:
        """S77-B: the bytes a LIVE KV move to ``tokens`` maps FRESH -- every range of
        the new plan minus the extents it keeps (``set_plan``'s census rule: an extent
        is kept only wholly inside one range). Paused tensors map at their resume."""
        total = 0
        for m in self.kv_tensors:
            info = self.spans.info(m.ptr)
            if info is None or not info.active:
                continue
            total += _fresh_bytes(self.spans_by_ptr.get(int(m.ptr)), self.kv_spans(m, tokens),
                                  int(info.mapped))
        return int(total)

    def bank_release_bytes(self, k: int) -> int:
        """S77-B: the bytes a LIVE bank gives back going from its mapped rows to ``k``
        rows ON (whole lattice cells above the new end; a paused bank gives none)."""
        total = 0
        for m in self.row_tensors:
            info = self.spans.info(m.ptr)
            if info is None or not info.active:
                continue
            total += _released_bytes(self.spans_by_ptr.get(int(m.ptr)), self.bank_spans(m, k),
                                     int(info.mapped))
        return int(total)

    def _bank_straddles(self, k: int) -> bool:
        """``refuse_wipes``' rule for a live bank going to ``k`` rows ON."""
        for m in self.row_tensors:
            info = self.spans.info(m.ptr)
            prev = self.spans_by_ptr.get(int(m.ptr))
            if info is None or not info.active or prev is None:
                continue
            if straddling(prev, self.bank_spans(m, k)) is not None:
                return True
        return False

    def grow_room(self, n: int, j: int, *, k: Optional[int] = None,
                  card: Optional["CardRoom"] = None) -> Optional["GrowRoom"]:
        """S77-B: can THIS rank map the live KV of stage ``j`` at ``n`` seats -- measured?

        NF rc17 1019_171636 D TP0 18:07:42Z: S8 -> S9 mid D-prefill, the 5090's cell
        books the same 6 rows at S8 and S9 (its slack pays the KV), so ``apply_stage``
        mapped KV FIRST (``order=kv-first``) -- into a card whose free bytes the last
        chunk's transient still sat on in the caching allocator (``PDFLIP-VRAM-PEAK
        ... reserved 32980 allocated 30126 card_free_mib=109``; the extend trim runs
        only BEFORE the next chunk). The draft k0 (~68 MiB) mapped, v0 did not:
        ``tms_set_spans(v0, now=True) rc=2`` -> rank dead, TP1/TP2 (bank-first,
        applied) dead in the next collective. The cell table is the PLAN; the card
        is measured here: ``need`` = the bytes the KV maps fresh, ``free`` =
        ``cudaMemGetInfo``; short -> the allocator cache is trimmed (synchronize +
        empty_cache, the extend trim's own move) and measured again; still short ->
        this rank's own expert rows go OFF first (elastic by law: the KV is demand,
        the rows step aside), the largest k' whose released cells pay the rest.
        ``ok=False`` only when even all rows OFF cannot pay -- a real wall, which
        the runtime tick turns into a GROUP refusal before any rank moves.
        None when nothing is measured (no live KV grows, or no CUDA card).

        H2 (review S77-B): the wake (``on_wake``) applies its stage to PAUSED KV
        tensors (mapped at their tag's resume) -- ``need`` is 0 there and nothing
        is voted. A wake path that ever maps KV LIVE must take the same group vote
        as ``runtime_tick`` (``_grow_room_refused``) before its ``apply_stage``;
        without it a short rank stops by name here, before anything moved."""
        n = max(1, min(int(n), self.cap))
        j = max(0, min(int(j), len(self.form.tokens) - 1))
        if k is None:
            _cell, k, _keep, _notes, _infos = self._stage_rows(n, j)
        k = int(k)
        need = self.kv_map_bytes(int(self.form.tokens[j]))
        if need <= 0:
            return None
        card = _CARD if card is None else card
        free0 = card.free()
        if free0 is None:
            return None
        start = min(k, int(self.rows_on))
        rel = self.bank_release_bytes(start) if start < int(self.rows_on) else 0
        trimmed = 0
        free1 = int(free0)
        if free1 + rel < need:
            trimmed = int(card.trim() or 0)
            f = card.free()
            free1 = int(free0) if f is None else int(f)
        rows_to = None
        ok = free1 + rel >= need
        if not ok:
            # only a k on the rows' lattice (S1-Wisch: a live shrink releases whole
            # cells and never straddles a kept extent) -- the largest that pays
            for kk in sorted((c for c in self.row_cut_ks if 0 <= c < start), reverse=True):
                if self._bank_straddles(kk):
                    continue
                if free1 + self.bank_release_bytes(kk) >= need:
                    rows_to, ok = int(kk), True
                    break
        release_all = self.bank_release_bytes(0)
        room = GrowRoom(ok=bool(ok), need=int(need), free_before=int(free0), free_after=int(free1),
                        trimmed=int(trimmed), rows_from=int(self.rows_on), rows_cell=k,
                        rows_to=rows_to, release_all=int(release_all))
        if trimmed or rows_to is not None or not ok:
            logger.info(room.line(n=n, stage=j))
        return room

    def _stage_rows(self, n: int, j: int):
        """The expert rows ON and the GDN keep of the cell (n, j) as ``apply_stage``
        books them beside a LIVE GDN pool -- (cell, k, keep, notes, infos)."""
        cell = self.cells[(n, j)]
        notes = []
        k, keep = int(cell.extra_rows), int(cell.mamba_keep)
        infos = [self.spans.info(m.ptr) for m in self.slot_tensors]
        live_gdn = any(i is not None and i.active for i in infos)
        if live_gdn and _rewake_on():
            # D-SEAT-REWAKE: the live GDN pool is whatever the wake or a live
            # re-seat mapped for THIS phase's n -- the stage move books the
            # rows that fit beside exactly those pages (never the cap form's
            # GDN, whose cells a SHRINK gave back to the experts) and moves
            # no GDN page (the re-seat alone does, at its own round)
            live_keep = self.pool_size + 1 if self.mamba_keep is None else int(self.mamba_keep)
            if live_keep <= self.pool_size and cell.rows_shrunk >= 0 and live_keep == cell.slot_limit + 1:
                k, keep = int(cell.rows_shrunk), live_keep
                notes.append("the GDN pool is live for %d seats: rows beside it" % n)
            else:
                # the cap form (or a keep this n does not name): the rows that
                # fit beside the cap form's GDN -- never more than the pages hold
                if cell.rows_full < 0:
                    raise PdFlipDSeatVramRefused(
                        "%s: stage S%d at n=%d needs the GDN pages of the empty seats but the "
                        "Mamba pool is mapped (live) in its cap form" % (LINE_MARK, j, n))
                k, keep = int(cell.rows_full), live_keep
                if live_keep <= self.pool_size:
                    notes.append("the live GDN pool keeps %d slots, not n=%d's %d: rows of the cap "
                                 "form" % (live_keep - 1, n, cell.slot_limit))
        elif keep <= self.pool_size and live_gdn:
            if cell.rows_full < 0:
                raise PdFlipDSeatVramRefused(
                    "%s: stage S%d at n=%d needs the GDN pages of the empty seats but the "
                    "Mamba pool is mapped (live) -- the stage was chosen for a wake whose "
                    "pool must already be paused" % (LINE_MARK, j, n))
            notes.append("the Mamba pool is mapped (live): it keeps its cap form")
            k, keep = int(cell.rows_full), self.pool_size + 1
        return cell, k, keep, notes, infos

    def _apply_stage_rows(self, n: int, j: int, cell, k: int, keep: int, notes, infos,
                          tokens: int, lent_base: Optional[int] = None) -> "PhaseApplied":
        import torch

        experts_live = any(
            (lambda i: i is not None and i.active)(self.spans.info(m.ptr))
            for m in self.row_tensors)
        kv_plans = []
        for m in self.kv_tensors:
            info = self.spans.info(m.ptr)
            kv_plans.append((m, self.kv_spans(m, tokens), info is not None and info.active))
        bank_plans = []
        for m in self.row_tensors:
            info = self.spans.info(m.ptr)
            bank_plans.append((m, self.bank_spans(m, k), info is not None and info.active))
        self.refuse_wipes([(m.ptr, m.geom.geom.name, sp, lv) for m, sp, lv in kv_plans]
                          + [(m.ptr, m.geom.name, sp, lv) for m, sp, lv in bank_plans])
        shrink = k < int(self.rows_on)
        if shrink:
            self._rows_off_live(k, experts_live)
        self._plan_mamba(infos, keep)
        census = {"kept": 0, "freed": 0, "mapped": 0, "tensors": 0}
        # D-MEM-SCHED: a LIVE KV prefix that shrinks between rounds -- every
        # kernel that may still read a page above the new end is done first
        # (the runtime tick only shrinks below pages no request holds)
        was = self.applied.stage if self.applied is not None else None
        if (was is not None and tokens < int(self.form.tokens[int(was)])
                and any(lv for _m, _sp, lv in kv_plans)
                and torch.cuda.is_initialized()):
            self._timed_sync()
        # RB (NF y3w e033a931db, D 01:45:51, S7 -> S8 at incoming=77427):
        # RELEASE BEFORE MAP. The KV prefix used to be mapped first and the
        # expert bank released after it, so a grow that the planner paid
        # with expert rows (rows_on 15/12 -> 6) needed the rows' pages AND
        # the new KV pages at once. TP0/TP1 had the slack (card_free 2535 MiB
        # on TP0), TP2 (3080, card_free 758 MiB) did not:
        # ``PDFLIP-TMS-RESUME cuMemCreate FAILED rc=2 (out of memory)
        # size=37748736`` -> PdFlipDSeatVramRefused -> W17 group dead. The
        # cell table never books more than the boot form, so with the
        # shrinking side released first the growing side always fits:
        # a bank that shrinks gives its cells back before the KV maps, a KV
        # that shrinks (the loop order below, unchanged) before the bank maps.
        if shrink:
            self._set_bank_plans(bank_plans, census)
        for m, sp, live in kv_plans:
            self.set_plan(m.ptr, m.geom.geom.name, sp, live=live, census=census)
        page = int(self.kv_tensors[0].geom.token_pad) if self.kv_tensors else 0
        for pool in self.kv_pools:
            # zero_kv_data_buffers (the idle flush) writes only mapped rows --
            # #239 S3g: each pool's own (a token-cut FA pool: compacted)
            bound_stage_tokens(pool, tokens, page)
        if not shrink:
            k = self._grow_bank_elastic(bank_plans, k, census)
            for cache in self.caches:
                cache.set_seat_rows_on(k, device_write=experts_live)
        log_live_spans(census, n=n, stage=j, rows_from=int(self.rows_on), rows_to=k,
                       mamba_keep=self.mamba_keep, order="bank-first" if shrink else "kv-first")
        if experts_live:
            if k < int(self.rows_on):
                self.evicted_rows += int(self.rows_on) - k
            else:
                self.refilled_rows += k - int(self.rows_on)
        self.rows_on = k
        # D-TRANSIENT-LEND: a phase move ends any lend -- S106: unless the move kept lent
        # rows above the new phase's rows (``lent_base``): then the lend stays open from there
        self.lent_from = int(lent_base) if (lent_base is not None and k > int(lent_base)) else None
        applied = PhaseApplied(
            n=n, extra_rows=k,
            mamba_mapped=sum(span_bytes(slot_spans(m.geom, keep, self.granule))
                             for m in self.slot_tensors),
            expert_mapped=sum(span_bytes(row_spans(m.geom, m.geom.rows_boot + k, self.granule))
                              for m in self.row_tensors),
            cap_mapped=int(cell.cap_mapped), note="; ".join(notes), stage=j,
            kv_mapped=int(cell.kv_mapped))
        self.applied = applied
        return applied

    def _set_bank_plans(self, bank_plans, census) -> None:
        for m, sp, live in bank_plans:
            self.set_plan(m.ptr, m.geom.name, sp, live=live, census=census)

    def _grow_bank_elastic(self, bank_plans, k: int, census) -> int:
        """A growing bank takes the rows the card CAN back (y4g b8e559c3e3, D TP1
        07:48:04, S8 -> S6 with rows_on 6 -> k: the KV prefix released first
        (RB), yet ``PDFLIP-TMS-RESUME cuMemCreate FAILED rc=2 size=4194304`` on
        the bank -> PdFlipDSeatVramRefused -> W17 group dead; the same death in
        e033a931db 09300130). Expert rows are elastic by law: a row the card
        cannot map stays in the host store and the rank keeps decoding. On
        rc=2 the rows ON above the current ``rows_on`` are halved until the map
        fits; tensors already mapped for the larger count give the extra cells
        back (their rows were never turned ON). At ``rows_on`` nothing new is
        mapped, so a refusal there is a real one and stops by name."""
        floor, want = int(self.rows_on), int(k)
        while True:
            plans = bank_plans if k == want else [
                (m, self.bank_spans(m, k), live) for m, _sp, live in bank_plans]
            try:
                self._set_bank_plans(plans, census)
            except PdFlipDSeatVramNoMemory as exc:
                if k <= floor:
                    raise
                k_next = floor + (k - floor) // 2
                logger.warning(
                    "%s BANK-GROW-ELASTIC rows_on %d->%d refused (%s); retrying with %d rows "
                    "(the rows above stay in the host store)", LINE_MARK, floor, k, exc, k_next)
                k = k_next
                continue
            if k != want:
                logger.warning("%s BANK-GROW-ELASTIC rows_on %d -> %d of %d wanted (card short)",
                               LINE_MARK, floor, k, want)
            return k

    def _rows_off_live(self, k: int, experts_live: bool) -> None:
        """Turn bank rows OFF (coldest first) on a live bank: sync, then the
        copies and table writes of every layer, then a sync that lands them
        before the next forward and before a page goes."""
        import time

        import torch

        live_cuda = experts_live and any(
            getattr(getattr(getattr(c, "_pool_tables", None), "row_key", None),
                    "is_cuda", False) for c in self.caches)
        # D-MEM-SCHED (27B 29.09.): every kernel that may still read a
        # row -- source or coldest-first destination -- is done before a
        # byte moves; the copies and the table writes of EVERY layer then
        # land (second sync) before the next forward reads the routing
        # and before a page of the dropped rows goes
        if live_cuda:
            self._timed_sync()
        moved0 = self._bank_bytes_copied()
        t0 = time.perf_counter()
        for cache in self.caches:
            cache.set_seat_rows_on(k, device_write=experts_live)
        if live_cuda:
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
        moved = self._bank_bytes_copied() - moved0
        if moved > 0:
            self.bytes_copied += moved
            self.copy_ms += ms
        elif live_cuda:
            self.sync_ms += ms

    def _bank_bytes_copied(self) -> int:
        return sum(int(getattr(c, "_pdflip_bank_bytes_copied", 0) or 0) for c in self.caches)

    def warm_rows(self) -> int:
        """KV-STAGE warm refill: rows the grows refilled with the experts the
        shrink before sent away (summed over the layers)."""
        return sum(int(getattr(c, "_pdflip_seat_warmed", 0)) for c in self.caches)

    def _timed_sync(self) -> None:
        import time

        import torch

        t0 = time.perf_counter()
        torch.cuda.synchronize()
        self.sync_ms += (time.perf_counter() - t0) * 1000.0

    def _bank_bytes_at(self, k: int) -> int:
        """The bytes the bank's pages hold with ``k`` seat rows ON."""
        return sum(span_bytes(self.bank_spans(m, k)) for m in self.row_tensors)

    def _bank_live(self) -> bool:
        return any((lambda i: i is not None and i.active)(self.spans.info(m.ptr))
                   for m in self.row_tensors)

    def lend_target(self, budget_bytes: int) -> int:
        """D-TRANSIENT-LEND: the largest lattice k above ``rows_on`` whose bank
        pages cost at most ``budget_bytes`` more than now; ``rows_on`` when no
        step fits (pure arithmetic, no CUDA)."""
        base = int(self.rows_on)
        have = self._bank_bytes_at(base)
        best = base
        for c in self.row_cut_ks:
            if c <= base or c > int(self.x_max):
                continue
            if self._bank_bytes_at(c) - have <= int(budget_bytes):
                best = int(c)
        return best

    def lend(self, budget_bytes: int) -> int:
        """D-TRANSIENT-LEND (user law 10:20Z): card bytes free between two
        extends become expert rows. Only a LIVE bank of an applied phase
        lends; the rows grow like every grow of this class -- map first, then
        ON (no sync, no kernel reads a moved byte), halved if the card says
        no (``_grow_bank_elastic``). The phase cell is not touched: the lend
        rides above it until ``unlend`` or the next phase move. Returns the
        rows added."""
        if self.applied is None or not self.row_tensors or not self._bank_live():
            return 0
        base = int(self.rows_on)
        want = self.lend_target(budget_bytes)
        if want <= base:
            return 0
        plans = [(m, self.bank_spans(m, want), True) for m in self.row_tensors]
        self.refuse_wipes([(m.ptr, m.geom.name, sp, lv) for m, sp, lv in plans])
        census = {"kept": 0, "freed": 0, "mapped": 0, "tensors": 0}
        k = self._grow_bank_elastic(plans, want, census)
        for cache in self.caches:
            cache.set_seat_rows_on(k, device_write=True)
        if k > base:
            if self.lent_from is None:
                self.lent_from = base
            self.refilled_rows += k - base
            self.lent_rows += k - base
        self.rows_on = int(k)
        applied = self.applied
        log_live_spans(census, n=applied.n, stage=applied.stage, rows_from=base,
                       rows_to=int(k), mamba_keep=self.mamba_keep, order="lend")
        return int(k) - base

    def unlend(self) -> int:
        """D-TRANSIENT-LEND: the lent rows go back before anything that needs
        the bytes (an extend, a stage move): rows OFF coldest first, sync,
        then the cells above the phase's k are released (whole lattice cells,
        S1-Wisch). Returns the rows given back."""
        if self.lent_from is None:
            return 0
        base = int(self.lent_from)
        cur = int(self.rows_on)
        self.lent_from = None
        if cur <= base:
            return 0
        live = self._bank_live()
        plans = [(m, self.bank_spans(m, base), live) for m in self.row_tensors]
        self.refuse_wipes([(m.ptr, m.geom.name, sp, lv) for m, sp, lv in plans])
        self._rows_off_live(base, live)
        census = {"kept": 0, "freed": 0, "mapped": 0, "tensors": 0}
        self._set_bank_plans(plans, census)
        self.evicted_rows += cur - base
        self.returned_rows += cur - base
        self.rows_on = base
        applied = self.applied
        log_live_spans(census, n=applied.n if applied is not None else 0,
                       stage=applied.stage if applied is not None else None,
                       rows_from=cur, rows_to=base, mamba_keep=self.mamba_keep, order="return")
        return cur - base

    def unlend_target(self, need_free_bytes: int, free_bytes: int) -> int:
        """S78 KEEP-SHORT: the largest lattice k in ``[lent_from, rows_on]`` whose
        release leaves the card at least ``need_free_bytes`` free -- the cells
        above k go back, the rows up to k stay ON (pure arithmetic, no CUDA; the
        mirror of :meth:`lend_target`). ``rows_on`` when the card already holds
        the need, ``lent_from`` when no lent cell can stay; ``rows_on`` with
        nothing lent."""
        cur = int(self.rows_on)
        if self.lent_from is None:
            return cur
        base = int(self.lent_from)
        if int(free_bytes) >= int(need_free_bytes):
            return cur
        have = self._bank_bytes_at(cur)
        best = base
        for c in self.row_cut_ks:
            if c <= base or c >= cur:
                continue
            if int(free_bytes) + (have - self._bank_bytes_at(c)) >= int(need_free_bytes):
                best = max(best, int(c))
        return best

    def unlend_to(self, k: int) -> int:
        """S78 KEEP-SHORT: give back only the lent cells above lattice row ``k``
        (rows OFF coldest first, sync, the cells released -- :meth:`unlend`'s
        order and plans, cut at ``k`` instead of the phase's rows). The lend
        stays open (``lent_from`` kept) while rows above the phase remain;
        ``k <= lent_from`` is the full :meth:`unlend`. Returns the rows given back."""
        if self.lent_from is None:
            return 0
        base = int(self.lent_from)
        cur = int(self.rows_on)
        k = int(k)
        if k <= base:
            return self.unlend()
        if k >= cur:
            return 0
        live = self._bank_live()
        plans = [(m, self.bank_spans(m, k), live) for m in self.row_tensors]
        self.refuse_wipes([(m.ptr, m.geom.name, sp, lv) for m, sp, lv in plans])
        self._rows_off_live(k, live)
        census = {"kept": 0, "freed": 0, "mapped": 0, "tensors": 0}
        self._set_bank_plans(plans, census)
        self.evicted_rows += cur - k
        self.returned_rows += cur - k
        self.rows_on = k
        applied = self.applied
        log_live_spans(census, n=applied.n if applied is not None else 0,
                       stage=applied.stage if applied is not None else None,
                       rows_from=cur, rows_to=k, mamba_keep=self.mamba_keep, order="return-short")
        return cur - k

    def apply(self, n: int) -> PhaseApplied:
        """The pages of a phase of ``n`` seats (``n`` = cap: the cap form).
        Called BEFORE the saver resumes the tag of a paused allocation (its
        plan is set) and at any time for a mapped one (the expert bank grows
        or shrinks in place). A live shrink goes as in ``apply_stage``: rows
        OFF in the tables first, synchronize, then the pages (PA review of
        1bab093912: the pages went first here)."""
        import torch

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
        self._plan_mamba(infos, keep)
        bank_plans = []
        for m in self.row_tensors:
            info = self.spans.info(m.ptr)
            bank_plans.append((m, self.bank_spans(m, k), info is not None and info.active))
        self.refuse_wipes([(m.ptr, m.geom.name, sp, lv) for m, sp, lv in bank_plans])
        experts_live = any(lv for _m, _sp, lv in bank_plans)
        shrink = int(k) < int(self.rows_on)
        if shrink:
            for cache in self.caches:
                cache.set_seat_rows_on(k, device_write=experts_live)
            if experts_live and any(
                    getattr(getattr(getattr(c, "_pool_tables", None), "row_key", None),
                            "is_cuda", False) for c in self.caches):
                torch.cuda.synchronize()
        census = {"kept": 0, "freed": 0, "mapped": 0, "tensors": 0}
        for m, sp, live in bank_plans:
            self.set_plan(m.ptr, m.geom.name, sp, live=live, census=census)
        # the tables: written now when the bank is mapped (the wake's rearm
        # already ran), else recorded for the rearm's reinit
        if not shrink:
            for cache in self.caches:
                cache.set_seat_rows_on(k, device_write=experts_live)
        log_live_spans(census, n=n, stage=None, rows_from=int(self.rows_on), rows_to=int(k),
                       mamba_keep=self.mamba_keep)
        self.rows_on = int(k)
        self.lent_from = None  # D-TRANSIENT-LEND: a phase move ends any lend
        applied = PhaseApplied(
            n=n, extra_rows=int(k),
            mamba_mapped=sum(span_bytes(slot_spans(m.geom, keep, self.granule))
                             for m in self.slot_tensors),
            expert_mapped=sum(span_bytes(row_spans(m.geom, m.geom.rows_boot + k, self.granule))
                              for m in self.row_tensors),
            cap_mapped=int(self.row_for(self.cap).cap_mapped), note="; ".join(notes))
        self.applied = applied
        return applied

    def mamba_cut_keeps(self) -> Tuple[int, ...]:
        """D-SEAT-REWAKE: every GDN keep a phase of this rank can take (n =
        1..cap, every stage cell) -- the lattice the Mamba plans are cut at."""
        ks = {int(r.slot_limit) + 1 for r in self.rows}
        ks.update(int(c.mamba_keep) for c in self.cells.values())
        return tuple(sorted(k for k in ks if 0 < k <= self.pool_size))

    def reseat_target(self, n: int, stage: Optional[int]) -> Tuple[int, int]:
        """``(k, keep)`` of a phase of ``n`` seats: the expert rows ON and the
        GDN slots per layer mapped (``pool_size + 1`` = every slot) -- the stage
        cell when the stage form holds, else the H95c row."""
        n = max(1, min(int(n), self.cap))
        if stage is not None and self.cells:
            cell = self.cells[(n, int(stage))]
            return int(cell.extra_rows), int(cell.mamba_keep)
        row = self.row_for(n)
        k = int(row.extra_rows) if (self.slot_tensors and self.row_tensors) else 0
        return k, ((int(row.slot_limit) + 1) if k > 0 else self.pool_size + 1)

    def reseat_live(self, n: int, stage: Optional[int], keep_lend_need: Optional[int] = None) -> PhaseApplied:
        """D-SEAT-REWAKE: the pages of a LIVE phase move from n seats to ``n``
        (Nutzer 30.09.: "D sleeped ohne wirklich runterzufahren und waket
        sofort wieder mit mehr sitzen"). Nothing sleeps: the GDN pool keeps the
        cells of its slots in use and maps / frees only the cells of the seats
        that come or go; the expert bank gives rows back or takes them.

        GROW (more GDN slots): the bank's rows go OFF coldest first (sync,
        copies, sync) and its cells are released BEFORE the GDN cells map --
        release before map (RB), the card never holds both. SHRINK: a sync
        (no kernel reads a freed slot), the GDN cells above the new keep go,
        then the bank grows elastically (a row the card cannot back stays in
        the host store). Every plan is cut at the lattice (``refuse_wipes``
        stops a plan that would release a kept cell before anything moves)."""
        n = max(1, min(int(n), self.cap))
        k, keep = self.reseat_target(n, stage)
        cur_keep = self.pool_size + 1 if self.mamba_keep is None else int(self.mamba_keep)
        keep_eff = min(int(keep), self.pool_size + 1)
        grow = keep_eff > cur_keep
        mamba_plans = [(m, slot_spans(m.geom, keep_eff, self.granule, cuts=self.mamba_cut_keeps()))
                       for m in self.slot_tensors]
        # S106 STAGE-KEEP on a live re-seat: an open lend rides on above the new phase rows
        # as far as the card holds it beside what the move maps fresh -- a SHRINK maps
        # nothing (GDN cells only go), a GROW maps the new seats' GDN cells (priced here,
        # RB order unchanged: the bank's released cells go before the GDN maps)
        k_cell, lent_base = int(k), None
        if self.lent_from is not None and keep_lend_need is not None and int(self.rows_on) > k_cell:
            gdn_need = self._gdn_map_bytes(mamba_plans) if grow else 0
            k_keep = self.lend_keep_target(k_cell, gdn_need, keep_lend_need)
            self._log_lend_keep(move="reseat n->%d" % n, rows_from=int(self.rows_on), kept=k_keep,
                                k_cell=k_cell, kv_need=gdn_need, keep_need=keep_lend_need)
            if k_keep > k_cell:
                k, lent_base = int(k_keep), k_cell
        bank_plans = [(m, self.bank_spans(m, k), True) for m in self.row_tensors]
        self.refuse_wipes([(m.ptr, m.geom.name, sp, True) for m, sp in mamba_plans]
                          + [(m.ptr, m.geom.name, sp, lv) for m, sp, lv in bank_plans])
        census = {"kept": 0, "freed": 0, "mapped": 0, "tensors": 0}
        rows_from = int(self.rows_on)
        import time as _time

        # the decode pause in parts (the metal's 200 ms bound, per term):
        # off_copy = rows OFF (coldest-first copies + table writes), unmap =
        # cells released, map = cells mapped (+ rows ON), sync = device syncs
        parts = {"off_copy_ms": 0.0, "unmap_ms": 0.0, "map_ms": 0.0, "sync_ms": 0.0}

        def _off(k_to: int) -> None:
            s0, t = self.sync_ms, _time.perf_counter()
            self._rows_off_live(k_to, True)
            wall = (_time.perf_counter() - t) * 1000.0
            ds = max(0.0, self.sync_ms - s0)
            parts["sync_ms"] += ds
            parts["off_copy_ms"] += max(0.0, wall - ds)

        def _timed(key: str, fn) -> None:
            t = _time.perf_counter()
            fn()
            parts[key] += (_time.perf_counter() - t) * 1000.0

        def _map_gdn() -> None:
            for m, sp in mamba_plans:
                self.set_plan(m.ptr, m.geom.name, sp, live=True, census=census)

        if grow:
            if k < rows_from:
                _off(k)
                _timed("unmap_ms", lambda: self._set_bank_plans(bank_plans, census))
            _timed("map_ms", _map_gdn)
        else:
            import torch

            if torch.cuda.is_initialized():
                s0 = self.sync_ms
                self._timed_sync()  # no kernel may still read a slot whose cells go
                parts["sync_ms"] += self.sync_ms - s0
            _timed("unmap_ms", _map_gdn)
            if k > rows_from:
                t_map = _time.perf_counter()
                k = self._grow_bank_elastic(bank_plans, k, census)
                for cache in self.caches:
                    cache.set_seat_rows_on(k, device_write=True)
                parts["map_ms"] += (_time.perf_counter() - t_map) * 1000.0
            elif k < rows_from:
                _off(k)
                _timed("unmap_ms", lambda: self._set_bank_plans(bank_plans, census))
        self.last_reseat_parts = dict(parts)
        self.mamba_keep = None if keep_eff > self.pool_size else int(keep_eff)
        if self.mamba_pool is not None and self.slot_tensors:
            self.mamba_pool._pdflip_seat_keep = self.mamba_keep
        if k < rows_from:
            self.evicted_rows += rows_from - k
        elif k > rows_from:
            self.refilled_rows += k - rows_from
        self.rows_on = int(k)
        # D-TRANSIENT-LEND: a phase move ends any lend -- S106: unless a shrink kept lent rows
        self.lent_from = int(lent_base) if (lent_base is not None and int(k) > int(lent_base)) else None
        log_live_spans(census, n=n, stage=stage, rows_from=rows_from, rows_to=int(k),
                       mamba_keep=self.mamba_keep, order="bank-first" if grow else "mamba-first")
        prev = self.applied
        applied = PhaseApplied(
            n=n, extra_rows=int(k),
            mamba_mapped=sum(span_bytes(slot_spans(m.geom, keep_eff, self.granule))
                             for m in self.slot_tensors),
            expert_mapped=sum(span_bytes(row_spans(m.geom, m.geom.rows_boot + k, self.granule))
                              for m in self.row_tensors),
            cap_mapped=int(prev.cap_mapped) if prev is not None else 0,
            note="live reseat", stage=stage if stage is None else int(stage),
            kv_mapped=int(prev.kv_mapped) if prev is not None else 0)
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


def log_live_spans(census: Dict[str, int], *, n: int, stage: Optional[int], rows_from: int,
                   rows_to: int, mamba_keep: Optional[int] = None, order: str = "-") -> None:
    """The metal marker of a live apply (none when every allocation was
    paused: a plan moves no byte). ``wiped=0`` by construction --
    ``refuse_wipes`` stopped the apply otherwise."""
    if not census.get("tensors"):
        return
    logger.info(
        "%s n=%d stage=%s rows_on %d->%d tensors=%d extents_kept=%d cells_freed=%d "
        "cells_mapped=%d wiped=0 mamba_keep=%s order=%s (a live move releases whole lattice "
        "cells only; mamba_keep = the GDN slots with pages = the flush's reset range, all = "
        "cap form -- a live apply never moves it; order = which side moved first, RB: the "
        "shrinking side always releases before the growing side maps)",
        LIVE_MARK, int(n), "-" if stage is None else "S%d" % int(stage), int(rows_from),
        int(rows_to), census["tensors"], census["kept"], census["freed"], census["mapped"],
        "all" if mamba_keep is None else int(mamba_keep), order)


# ---------------------------------------------------------------------------
# the scheduler's entry points (d_park_runtime delegates here)
# ---------------------------------------------------------------------------

PHASE_ATTR = "_pdflip_d_seat_phase"
CTL_ATTR = "_pdflip_d_seat_vram"


def _cap_of(sched) -> int:
    return int(getattr(getattr(sched, "server_args", None), "max_running_requests", 0) or 1)


def _req_pool(sched):
    runner = getattr(getattr(sched, "tp_worker", None), "model_runner", None)
    return getattr(runner, "req_to_token_pool", None) or getattr(sched, "req_to_token_pool", None)


def _kv_pools(sched) -> List[object]:
    """#251c: the KV pools of this rank's target and draft runners."""
    out = []
    runners = [getattr(getattr(sched, "tp_worker", None), "model_runner", None)]
    draft = getattr(sched, "draft_worker", None)
    for attr in ("model_runner", "draft_model_runner"):
        runners.append(getattr(draft, attr, None))
    for r in runners:
        pool = getattr(r, "token_to_kv_pool", None)
        if pool is not None and all(pool is not q for q in out):
            out.append(pool)
    return out


def _kv_allocator(sched):
    runner = getattr(getattr(sched, "tp_worker", None), "model_runner", None)
    return (getattr(sched, "token_to_kv_pool_allocator", None)
            or getattr(runner, "token_to_kv_pool_allocator", None))


def _page_size(sched) -> int:
    runner = getattr(getattr(sched, "tp_worker", None), "model_runner", None)
    return int(getattr(sched, "page_size", None) or getattr(runner, "page_size", None) or 1)


def controller(sched) -> Optional[SeatVram]:
    """This rank's page controller, built on first use; None when nothing on
    this rank is seat-proportional (the 3080 workers) or a piece is missing."""
    ctl = getattr(sched, CTL_ATTR, None)
    if ctl is not None:
        return ctl if ctl is not False else None
    runner = getattr(getattr(sched, "tp_worker", None), "model_runner", None)
    try:
        ctl = SeatVram.from_runtime(cap=_cap_of(sched), req_to_token_pool=_req_pool(sched),
                                    model=getattr(runner, "model", None),
                                    kv_pools=_kv_pools(sched))
        if not ctl.slot_tensors and not ctl.row_tensors:
            ctl = None
    except PdFlipDSeatVramWipe:
        raise  # pages that would be remapped fresh: a stop, never "no controller"
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
        if allocator.set_phase_limit(want, seats=target):
            limit = want
        else:
            st.note = ("slots above %s still in use at the wake: slot limit NOT applied, "
                       "the phase keeps the cap form" % want)
            n_phys = st.cap
            allocator.set_phase_limit(None)
    st.n, st.slot_limit, st.done = target, limit, True
    st.has_n = n is not None
    # 1b. REPLICATED (#251c): the KV stage -- the form values and the wake's
    # demand only, and the allocator's page cap that makes it safe
    form = stage_form()
    stage = None
    if form is not None:
        demand = getattr(recv_req, "phase_kv_tokens", None) if n is not None else None
        choice = choose_form_stage(form, n_phys, demand)
        stage = choice.stage
        kv_alloc = _kv_allocator(sched)
        page = _page_size(sched)
        live = max_live_page(kv_alloc) if kv_alloc is not None else 0
        while stage < len(form.tokens) - 1 and form.tokens[stage] // page < live:
            stage += 1
        if stage > form.max_stage(n_phys):
            # the captured waves only hold in stages the form lets n take
            # (capture_floors); replicated inputs -> every rank stops alike
            raise PdFlipDSeatVramRefused(
                "%s: KV pages up to %d are still held at the wake and need stage S%d, "
                "above the S%d the form allows %d seats -- the captured decode waves "
                "would not hold there; nothing was mapped" % (
                    LINE_MARK, live, stage, form.max_stage(n_phys), n_phys))
        if stage != choice.stage:
            st.note = ("KV pages up to %d still held at the wake: stage S%d instead of S%d"
                       % (live, stage, choice.stage))
        if kv_alloc is not None:
            _engage_kv_cap(kv_alloc, form.tokens[stage], page)
        st.stage, st.stage_tokens = stage, form.tokens[stage]
        st.demand, st.over = choice.demand, bool(choice.over)
    # 2. RANK-LOCAL: the pages
    import time as _time

    _t_apply = _time.perf_counter()
    ctl = controller(sched)
    if ctl is None:
        applied = None
    elif stage is not None and ctl.cells:
        applied = ctl.apply_stage(n_phys, stage)
    else:
        applied = ctl.apply(n_phys)
    # D-SEAT-REWAKE: the wake's own page re-plan, measured (the first price)
    st.apply_ms = (_time.perf_counter() - _t_apply) * 1000.0 if ctl is not None else None
    logger.info("%s", phase_line(st, applied, ctl))
    if _rewake_on() and st.apply_ms is not None:
        logger.info("PDFLIP D-SEAT-REWAKE WAKE-APPLY n=%d ms=%.1f rows_on=%s mamba_keep=%s (the wake's page "
                    "re-plan, measured: the first price of a live re-plan)", st.n, st.apply_ms,
                    getattr(ctl, "rows_on", None), getattr(ctl, "mamba_keep", None))
    if stage is not None:
        line = stage_line(st, applied)
        if form.by_demand:
            line += " form=demand"  # #251d: no table, the demand alone chose
        logger.info("%s", line)
    return st


STAGE_MARK = "#251 WAKE-RESHARD"


def stage_line(st: PhaseState, applied: Optional[PhaseApplied]) -> str:
    """The #251c metal marker, one per wake that carries a stage."""
    return "%s n=%d stage=S%d tokens=%d demand=%s over=%s kv_mib=%s rows_on=%s epoch=%s" % (
        STAGE_MARK, st.n, int(st.stage), int(st.stage_tokens or 0),
        "-" if st.demand is None else int(st.demand), "yes" if st.over else "no",
        "-" if applied is None else "%.1f" % (applied.kv_mapped / _MIB),
        "-" if applied is None else int(applied.extra_rows), st.epoch)


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
        raise PdFlipDSeatOverrun(
            "%s: a forward batch of %d requests in a D phase of n=%d seats (epoch %s) -- "
            "the posts of seats above n are not mapped (slots 1..%s); the admission cap "
            "should have held it" % (REFUSAL_CODE, bs, cap, st.epoch, st.slot_limit))


# ---------------------------------------------------------------------------
# D-MEM-SCHED (29.09.): the stage BETWEEN wakes -- the gap (a) of #251c/d
# ---------------------------------------------------------------------------

MEM_SCHED_ATTR = "_pdflip_d_mem_sched"


def elastic_on() -> bool:
    """User law 10:20Z: default ON; the one switch is a diagnostic stop."""
    from flliper.srt.environ import envs

    return not bool(envs.FLLIPER_PDFLIP_DISABLE_D_ELASTIC_ROWS.get())


def _req_tokens(req) -> int:
    return len(getattr(req, "origin_input_ids", None) or ()) + len(
        getattr(req, "output_ids", None) or ())


#: y3j 09291933: the tick found an extend batch its merge had not taken yet
UNMERGED_MARK = "PDFLIP D-MEM-SCHED UNMERGED-EXTEND"
_UNMERGED_N = [0]


def _unmerged_extend(sched, running) -> list:
    """The requests of the extend batch that ran last and is not merged yet.

    The tick runs at the top of ``get_next_batch_to_run``, BEFORE the merge
    of ``last_batch`` into the running batch. After every extend its
    requests are in neither the running batch nor the queue, so the census
    read them as gone: y3j 09291933 D 20:02:20 -- the wake's extend held
    pdflip-27-76 (23168) and the needle pdflip-40-151 (64072), the tick saw
    used=0, an END event and a pending shrink to S0 (cap 32768) while live
    pages reached 87296; the bs2 decode found no page below the cap, the
    retraction emptied the batch (solo-OOM on the last) and the pressure
    park left the needle behind the older agent turn until its client gave
    up (MISS, empty answer). The same shape at 19:41:57 and 19:43:52.
    Replicated like the rest of the census: every D rank ran the same batch."""
    last = getattr(sched, "last_batch", None)
    mode = getattr(last, "forward_mode", None)
    if last is None or mode is None or not mode.is_extend():
        return []
    have = {id(r) for r in running}
    out = []
    for r in list(getattr(last, "reqs", None) or ()):
        if id(r) in have:
            continue
        fin = getattr(r, "finished", None)
        if callable(fin) and fin():
            continue
        out.append(r)
        have.add(id(r))
    return out


def _demand_lists(sched) -> Tuple[list, list]:
    """REPLICATED (27B condition 2): the running requests (with the chunked
    one and an extend batch the merge has not taken yet) and the queued
    requests the next admission can take -- the request lists are the same
    on every D rank, the page lists are not (#603)."""
    running = list(getattr(getattr(sched, "running_batch", None), "reqs", None) or ())
    chunked = getattr(sched, "chunked_req", None)
    if chunked is not None and all(chunked is not r for r in running):
        running.append(chunked)
    extra = _unmerged_extend(sched, running)
    if extra:
        running.extend(extra)
        _UNMERGED_N[0] += 1
        n = _UNMERGED_N[0]
        if n <= 20 or n % 200 == 0:
            logger.info(
                "%s n=%d rids=%s tokens=%d -- the extend batch that ran last is not merged "
                "yet at this tick: its requests hold their pages (no end event, no shrink "
                "below them)", UNMERGED_MARK, n, [str(getattr(r, "rid", "")) for r in extra],
                sum(_req_tokens(r) for r in extra))
    queue = getattr(sched, "waiting_queue", None) or ()
    return running, _admissible_queue(queue, _cap_of(sched) - len(running))


def global_demand(sched) -> Tuple[int, int, frozenset]:
    """REPLICATED (27B condition 2): the tokens the running requests hold,
    the tokens the queue head wants, the running rids -- the request lists
    are the same on every D rank, the page lists are not (#603). The
    requests of an extend batch the merge has not taken yet run too."""
    running, admissible = _demand_lists(sched)
    used = sum(_req_tokens(r) for r in running)
    incoming = sum(_req_tokens(r) for r in admissible)
    return used, incoming, frozenset(getattr(r, "rid", id(r)) for r in running)


def _clip_max_new() -> int:
    """``CLIP_MAX_NEW_TOKENS`` of ``schedule_policy`` (the same env var)."""
    try:
        from flliper.srt.environ import envs

        return int(envs.FLLIPER_CLIP_MAX_NEW_TOKENS_ESTIMATION.get())
    except Exception:  # noqa: BLE001 -- a desk double without environ: the default
        return int(os.environ.get("FLLIPER_CLIP_MAX_NEW_TOKENS_ESTIMATION", "4096") or 4096)


def _req_max_new(req, clip: int) -> int:
    sp = getattr(req, "sampling_params", None)
    mx = getattr(sp, "max_new_tokens", None)
    if mx is None:
        return 0
    out = len(getattr(req, "output_ids", None) or ())
    return max(0, min(int(mx) - out, int(clip)))


def _new_token_ratio(sched) -> float:
    tracker = getattr(sched, "new_token_ratio_tracker", None)
    v = getattr(tracker, "current", None)
    if v is None:
        v = getattr(sched, "new_token_ratio", None)
    try:
        return max(0.0, float(v)) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


#: NF y3z 0930_023308 ep44/ep58: the stage the census held while the gate refused
GATE_PRICE_MARK = "PDFLIP D-MEM-SCHED GATE-PRICE"


def admission_reserve(sched, running, admissible) -> int:
    """REPLICATED: what the admission gate charges beyond the census while a
    request is queued -- ``PrefillAdder``'s ``rem_total_token_offset`` (every
    running request's ``min(max_new - out, CLIP) * new_token_ratio``) plus
    each candidate's ``min(max_new - out, CLIP) + page`` (``add_one_req``).

    NF y3z 0930_023308 (30.09.): ep44 02:53:25 at S6 (229376) the census
    needed 144571 + 78477 + air 4120 = 227168 and held; the attention host
    priced pdflip-42-101 at 82637 against a budget of 78336 (5 decode
    reservations, 6469) -- NO_TOKEN at a free seat, batch-full until
    pdflip-41-100 ended 4 s later. ep58 02:58:12 the same at S5 with two free
    seats (63071 > 59392, 8,1 s). S7..S9 were on the ladder (S7 at 02:52:15).
    Zero with nothing queued: a reservation never moves an expert row on its
    own. Inputs are request fields and the scheduler's ratio. The ratio is
    RANK-LOCAL (y6h 10011531, see :func:`_agree_reserve`), so the caller
    agrees the result over the group before the machine steps on it."""
    if not admissible:
        return 0
    clip = _clip_max_new()
    page = max(1, _page_size(sched))
    ratio = _new_token_ratio(sched)
    run = sum(int(math.ceil(_req_max_new(r, clip) * ratio)) for r in running)
    return run + sum(_req_max_new(r, clip) + page for r in admissible)


#: y6h 10011531: the group's reservation differed from this rank's
RESERVE_AGREE_MARK = "PDFLIP D-MEM-SCHED RESERVE-AGREE"
_RESERVE_AGREE_N = [0]


def _agree_reserve(sched, admissible, reserve: int) -> int:
    """The gate reservation as ONE group number: MAX over the D ranks.

    NF y6h 10011531 (01.10. 16:07:53): ``new_token_ratio`` is rank-local --
    TP0 priced the queue at 4964, TP1/TP2 at 6592 for the same requests, so
    the replicated machine split: TP0 S6->S7 ("grow to hold 261597"), the
    workers S6->S8 ("grow to hold 263225"). From there the ranks took
    different control paths and every D rank wedged at 16:07:56 in the
    ``drain_retired_prefetch`` all_reduce (outstanding=4, no token, until the
    VORSTOPP). MAX is the gate's own direction: the rank that charges more
    is the one whose admission would refuse.

    Entered only with something queued (``admissible`` is a request list,
    the same on every D rank), so a decode round with an empty queue pays no
    collective."""
    if not admissible:
        return int(reserve)
    gmin = getattr(sched, "_pdflip_group_min_ints", None)
    if gmin is None:
        return int(reserve)
    agreed = -int(gmin([-int(reserve)])[0])
    if agreed != int(reserve):
        _RESERVE_AGREE_N[0] += 1
        n = _RESERVE_AGREE_N[0]
        if n <= 20 or n % 200 == 0:
            logger.info("%s n=%d local=%d group=%d ratio=%.4f -- the reservation the stage "
                        "machine steps on is the group's, not this rank's",
                        RESERVE_AGREE_MARK, n, int(reserve), agreed, _new_token_ratio(sched))
    return agreed


def kv_ladder_reading(sched) -> Optional[Dict[str, int]]:
    """NF-STAU (29.09.): D's KV LADDER for the front's KV fit test (ARRIVAL-SEAT),
    published on ``/server_info`` ``pdflip_kv`` -- IPC, never a log line.

    ``ladder_ceiling``: the highest stage D-MEM-SCHED may grow to this phase
    (its ``stage_tokens[-1]``; before its first tick the stage form's
    ``max_stage(n)``), ``ladder_stage``: the mapped stage, ``ladder_used``:
    the global tokens the running requests hold (:func:`global_demand`, the
    same census the machine steps on). The front tested against the MAPPED
    stage (available + evictable) -- y3m 22:03:44 "kv need=130563 >
    free=54016" at S3/S4 with ~77k used while the ladder reached 524288:
    no KV shortage, a wrong size. None when no ladder runs (off, asleep, no
    stage form, elastic stop) -- the reading then stays what it was."""
    if not armed() or not elastic_on() or getattr(sched, "pdflip_dormant", False):
        return None
    form = stage_form()
    if form is None:
        return None
    ms = getattr(sched, MEM_SCHED_ATTR, None)
    st = getattr(sched, PHASE_ATTR, None)
    if ms is not None and getattr(ms, "stage_tokens", None):
        ceiling = int(ms.stage_tokens[-1])
        stage = int(ms.stage_tokens[min(int(ms.stage), len(ms.stage_tokens) - 1)])
    else:
        n = 1
        if st is not None and int(getattr(st, "cap", 0) or 0) > 0:
            n = max(1, min(int(st.n or st.cap), int(st.cap)))
        ceiling = int(form.tokens[form.max_stage(n)])
        stage = int(getattr(st, "stage_tokens", 0) or form.tokens[0]) if st is not None else int(form.tokens[0])
    used, _incoming, _rids = global_demand(sched)
    return {"ladder_ceiling": ceiling, "ladder_stage": stage, "ladder_used": int(used)}


#: y3j 09291933: the admission gate a grown stage re-opened
REOPEN_MARK = "PDFLIP D-MEM-SCHED ADMIT-REOPEN"
_REOPEN_N = [0]


def _reopen_admission(sched) -> None:
    """A grown (or cap-lifted) stage has room the last admission did not
    see: its NO_TOKEN set ``batch_is_full``, and upstream clears that only
    when a request finishes -- a whole decode later (y3j: 6 s per request
    after the wake, 32k bs4). The grow is the event that frees the room, so
    it re-opens the gate; the admission in this same iteration re-asks."""
    rb = getattr(sched, "running_batch", None)
    if rb is None or not getattr(rb, "batch_is_full", False):
        return
    rb.batch_is_full = False
    _REOPEN_N[0] += 1
    n = _REOPEN_N[0]
    if n <= 20 or n % 200 == 0:
        queue = getattr(sched, "waiting_queue", None) or ()
        logger.info("%s n=%d queue=%d -- the stage grew: the batch-full gate of the last "
                    "NO_TOKEN no longer holds", REOPEN_MARK, n, len(queue))


def _admissible_queue(queue, free_seats: int) -> list:
    """The queued requests the next admission can take: one per free seat,
    at least the head. y3j 09291933 (bs4 x 32k after the wake): with only the
    head as ``incoming`` the stage grew by ONE request per NO_TOKEN, and the
    batch-full gate held the next one until a whole decode had ended -- the
    four first tokens came 6 s apart (flip time 19-26 s for the last)."""
    return list(queue[: max(1, int(free_seats))]) if len(queue) else []


def _air(sched) -> int:
    from flliper.srt.pdflip.d_mem_sched import air_tokens

    args = getattr(sched, "server_args", None)
    chunk = int(getattr(args, "chunked_prefill_size", 0) or 0)
    verify = int(getattr(args, "speculative_num_draft_tokens", 0) or 1)
    return air_tokens(max(0, chunk), _cap_of(sched), verify)


def _group_floor_tokens(sched) -> int:
    """The group's highest live KV page in tokens -- MAX over the ranks, one
    collective, entered only when the replicated machine may shrink."""
    alloc = _kv_allocator(sched)
    local = max_live_page(alloc) * _page_size(sched) if alloc is not None else 0
    gmin = getattr(sched, "_pdflip_group_min_ints", None)
    if gmin is None:
        return int(local)
    return -int(gmin([-int(local)])[0])


#: NF1d 09291811: the line a pending shrink's cap is handed back to the
#: mapped stage because the ids below it cannot pay the group's demand
CAP_LIFT_MARK = "PDFLIP D-MEM-SCHED CAP-LIFT"


def _chunked_rest(sched) -> int:
    """REPLICATED (27B condition 2): the tokens of the live ``chunked_req``
    that are not prefilled yet -- the whole request (``_req_tokens``, what
    ``used`` counts) minus what its chunks hold. The chunked continuation is
    the same request on every rank, as are the prefix length and the fill
    boundary; of the two the SMALLER counts as done (a parked chunk whose
    ``extend_range`` ran ahead of the prefix is not done): an overestimate is
    the safe side, the lift only ends on a fresh verdict. 0 without one."""
    req = getattr(sched, "chunked_req", None)
    if req is None:
        return 0
    done = []
    pre = getattr(req, "prefix_indices", None)
    if pre is not None:
        try:
            done.append(len(pre))
        except TypeError:
            pass
    ext = getattr(req, "extend_range", None)
    end = getattr(ext, "end", None)
    if end is not None:
        done.append(int(end))
    return max(0, _req_tokens(req) - (min(done) if done else 0))


def _queued_load_back(admissible) -> int:
    """The host load-back tokens the queued requests the next admission takes
    will put on the device in ONE step (``pp_load_back_extent``, stamped at the
    request's own match of an earlier pass; 0 until it was matched -- the
    guard then still stands). Rank-LOCAL by construction (``#1046``: each rank
    chose its extent at its own match): it enters only this rank's room
    (``free ids below the cap - need``), whose verdict is the group MIN, so a
    rank with more to load back is exactly the one that must lift the cap.
    Each request's extent is clamped to its own prompt."""
    total = 0
    for r in admissible:
        ext = getattr(r, "pp_load_back_extent", None)
        if ext:
            total += max(0, min(int(ext), _req_tokens(r)))
    return total


def free_tokens_below(allocator, tokens: int, page_size: int) -> Optional[int]:
    """Rank-local: the tokens of the FREE page ids at or below the cap of
    ``tokens`` (the only ids an allocation under that cap can take); None
    when the allocator shows no free lists (then nothing is decided)."""
    page = max(1, int(page_size))
    lim = int(tokens) // page
    seen, n = False, 0
    for name in ("free_pages", "release_pages"):
        ids = getattr(allocator, name, None)
        if ids is None or not hasattr(ids, "numel"):
            continue
        seen = True
        if ids.numel():
            n += int((ids <= lim).sum())
    return n * page if seen else None


def _group_room_below(sched, alloc, tokens: int, need: int) -> bool:
    """REPLICATED verdict (MIN over the ranks, one collective): may a pending
    shrink keep its cap at ``tokens``, i.e. do the free ids below it pay
    ``need`` on EVERY rank? Entered only under a replicated condition
    (pending shrink AND demand), so every rank enters it alike. Evictable
    tree pages below the cap are NOT counted: the peel frees leaves in LRU
    order, and the leaves of a long cached sequence sit ABOVE the cap
    (NF1d: the pool received 0 of 246080 evictable tokens)."""
    room = free_tokens_below(alloc, tokens, _page_size(sched))
    ok = 1 if (room is None or room >= int(need)) else 0
    gmin = getattr(sched, "_pdflip_group_min_ints", None)
    if gmin is None:
        return bool(ok)
    return bool(int(gmin([ok])[0]))


#: NF y6k 01.10.: the re-check census line (one per FLOOR_CHECK_EVERY ticks)
FLOOR_CHECK_MARK = "PDFLIP D-MEM-SCHED FLOOR-CHECK"
FLOOR_CHECK_EVERY = 512
_SLACK_CAP = 1 << 40


def _recheck_max() -> int:
    from flliper.srt.environ import envs

    try:
        return max(1, int(envs.FLLIPER_PDFLIP_D_MEM_RECHECK_ROUNDS.get()))
    except (TypeError, ValueError):
        return 1


def _floor_room_state(ms) -> dict:
    """The re-check bookkeeping rides on the (replicated) machine: a new wake
    epoch builds a new machine and so starts from fresh reads."""
    fr = getattr(ms, "_pdflip_floor_room", None)
    if fr is None:
        fr = {"floor": None, "room": None, "lifted_since": False, "ticks": 0,
              "floor_reads": 0, "floor_cached": 0, "room_reads": 0, "room_cached": 0}
        ms._pdflip_floor_room = fr
    return fr


def _floor_for_shrink(sched, ms, fr, ended: bool) -> int:
    """The floor ``MemSched.step`` asks for -- only on a path that shrinks.

    NF y6k 01.10.: a pending shrink re-read the group floor every decode
    round (max_live_page copies the free lists device->host = a sync of the
    decode stream, plus one group collective): 32 % of the D-TP0 scheduler,
    host gap 9 ms/round. While the SAME shrink stays pending the floor can
    only FALL (new pages go below the pending cap, which sits below the
    floor), so the last read is an upper bound and blocks nothing it should
    not. It is re-read after 1, 2, 4, ... up to FLLIPER_PDFLIP_D_MEM_RECHECK_ROUNDS
    rounds (all inputs replicated: the round count, the pending stage), and at
    once on an end event. While the cap is lifted to the stage (pages may go
    above the old floor) the cached rounds answer with the mapped stage
    itself: no shrink, never one below a live page."""
    k = _recheck_max()
    rnd = int(getattr(ms, "_round", 0))
    c = fr.get("floor")
    if (k > 1 and not ended and ms.pending is not None and c is not None
            and c["pending"] == ms.pending and rnd < c["next"]):
        fr["floor_cached"] += 1
        if fr.get("lifted_since"):
            return int(ms.stage_tokens[ms.stage])
        return int(c["value"])
    value = int(_group_floor_tokens(sched))
    fr["floor_reads"] += 1
    gap = 1
    if c is not None and c["pending"] == ms.pending and c["value"] == value:
        gap = min(k, max(1, int(c["gap"]) * 2))
    fr["floor"] = {"pending": ms.pending, "value": value, "next": rnd + gap, "gap": gap}
    fr["lifted_since"] = bool(getattr(ms, "_cap_lifted", False))
    return value


def _room_ok(sched, ms, fr, alloc, cap_tokens: int, need: int, incoming: int,
             n_running: int, page: int, demand: Optional[tuple] = None) -> bool:
    """``_group_room_below`` with the same replicated re-check: a pure decode
    round (no incoming) under an unchanged pending cap reuses the last group
    verdict while the group's MIN slack (free tokens below the cap minus the
    need) cannot have been used up -- each running request takes at most
    ceil(r * R / page) new pages in r rounds of R tokens (R = verify tokens
    + 1). A new demand, a new running set or a new cap re-reads at once.

    S72 (rc16 1019_121629, D TP0 13:28:43-13:30:58, bs 1, ``#queue-req: 1``):
    a HELD demand -- the same queued request, nothing admitted, DQH holding
    the pass -- is not a new demand, but ``incoming > 0`` bypassed the cache in
    every pass: ``FLOOR-CHECK room_reads`` +512 per 512 ticks, ``room_cached=0``,
    each read a device->host copy of the free lists behind the schedule
    stream's WAR wait on the previous forward plus one group collective.
    ``DECODE-HOST-SPLIT sched_ms`` 34.9 (3.7 without the reads), ``sync_ms`` 0:
    the overlap was gone, the round 42 ms wall / 33.5 "gpu-ms" against
    29.7 / 25.6 (3157 rounds of pdflip-132-465 = the whole rc16 "bs1 64-160k"
    bucket). ``demand`` is the REPLICATED fingerprint of what the admission
    can take (sorted queued rids, the live chunked rid); with it, a NOT-OK
    group verdict (the cap stays lifted to the mapped stage: more room, never
    less) is reused under an unchanged (pending, cap, running count, incoming
    tokens, demand) on the same back-off as a decode round (1, 2, 4 .. K
    passes). The rank-local ``need`` (host load-back extents, #1046) is NOT in
    that key -- every input of the hit test is replicated, so every rank
    enters the collective on the same passes. An OK verdict under demand is
    still read every pass (an admission on this pass must not ride a stale
    "the pending cap pays it" -- NF1d). ``demand=None`` (direct callers) keeps
    the old rule."""
    k = _recheck_max()
    rnd = int(getattr(ms, "_round", 0))
    key = (ms.pending, int(cap_tokens), int(need), int(n_running))
    dkey = None
    if demand is not None and int(incoming) > 0:
        dkey = (ms.pending, int(cap_tokens), int(n_running), int(incoming), tuple(demand))
    c = fr.get("room")
    if k > 1 and int(incoming) <= 0 and c is not None and c["key"] == key and rnd < c["next"]:
        fr["room_cached"] += 1
        return bool(c["ok"])
    if (k > 1 and dkey is not None and c is not None and not c["ok"]
            and c.get("dkey") == dkey and rnd < c["next"]):
        fr["room_cached"] += 1
        return False
    room = free_tokens_below(alloc, cap_tokens, _page_size(sched))
    slack = _SLACK_CAP if room is None else min(_SLACK_CAP, int(room) - int(need))
    ok_local = 1 if slack >= 0 else 0
    gmin = getattr(sched, "_pdflip_group_min_ints", None)
    res = [ok_local, slack] if gmin is None else list(gmin([ok_local, slack]))
    fr["room_reads"] += 1
    ok = bool(int(res[0]))
    min_slack = int(res[1]) if len(res) > 1 else 0
    # S72: under a demand key the back-off grows on the REPLICATED key only
    # (``key`` carries the rank-local ``need`` there), so ``next`` -- the hit
    # test above -- is the same number on every rank
    if dkey is not None:
        prev = c["gap"] if (c is not None and c.get("dkey") == dkey) else 0
    else:
        prev = c["gap"] if (c is not None and c["key"] == key) else 0
    gap = 0
    if ok and min_slack > 0:
        pg = max(1, int(page))
        args = getattr(sched, "server_args", None)
        per_round = max(1, int(getattr(args, "speculative_num_draft_tokens", 0) or 1)) + 1
        n = max(1, int(n_running))
        pages_each = min_slack // (n * pg)          # new pages each request may still take
        safe = (pages_each * pg) // per_round        # rounds before any could need more
        gap = min(k, safe, max(1, int(prev) * 2))
    elif not ok:
        # a lift keeps the cap at the mapped stage (more room, not less):
        # re-read on the same back-off, the lift ends on a fresh verdict
        gap = min(k, max(1, int(prev) * 2))
    fr["room"] = {"key": key, "dkey": dkey, "ok": ok, "next": rnd + gap, "gap": max(1, gap)}
    return ok


#: 27B 29.09.: iterations the tick left at its first checks (P, the 27B, a
#: D without stage form) -- no collective, no sync, no allocation there
TICK_NOOP_ATTR = "_pdflip_d_mem_tick_noop"


def _tick_noop(sched) -> None:
    try:
        setattr(sched, TICK_NOOP_ATTR, int(getattr(sched, TICK_NOOP_ATTR, 0) or 0) + 1)
    except (AttributeError, TypeError):  # a slotted test double: nothing to count on
        pass
    return None


#: Q-701: the epoch of the phase a D serves in from its boot, before any wake
BOOT_EPOCH = "boot"
BOOT_PHASE_MARK = "Q-701 D-MEM-SCHED BOOT-PHASE"


def boot_phase(sched) -> Optional[PhaseState]:
    """Q-701 (NF y9nf boot 2, D 00:49:37Z serving, never woken): the phase
    state is made only by a wake (``on_wake``), so a D that SERVES FROM ITS
    BOOT had none and the runtime tick -- the only thing that moves the KV
    stage between wakes -- was a no-op. D stayed at the boot cap S0 (32768
    tokens, ``#1045 FLOOR PUBLISHED floor=32768``) while the front sent it
    store-cached requests on the X route (uncached 204..1934, load-back
    88768 / 159040): ``H105 RU FORM-A ADMISSION WAIT ... host_price=163404
    host_budget=32768`` 5604 times, 4 queued / 0 running, no forward, no
    flip (the front saw nothing to flip for). The boot form IS a phase: the
    cap's seats at S0, exactly what ``kv_stage_boot_cap`` engaged -- seeded
    here once (replicated: the form and the cap), so the tick grows the stage
    for the demand as it does after any wake. None without a stage form."""
    form = stage_form()
    if form is None:
        return None
    st = PhaseState(cap=_cap_of(sched), epoch=BOOT_EPOCH, done=True,
                    stage=0, stage_tokens=int(form.tokens[0]))
    setattr(sched, PHASE_ATTR, st)
    logger.info("%s cap=%d stage=S0 tokens=%d: this D serves from its boot without a wake; the "
                "boot form is its phase, the runtime tick moves the KV stage for the demand",
                BOOT_PHASE_MARK, st.cap, st.stage_tokens)
    return st


def _demand_fingerprint(sched, admissible) -> tuple:
    """S72: what the admission of this pass can take, as REPLICATED values --
    the sorted rids of the queued requests the next admission may take (the
    request lists are the same on every D rank, #603) and the rid of a live
    chunked_req. Token counts ride beside it in ``_room_ok``'s key (``incoming``,
    replicated); nothing rank-local (page lists, prefix_indices, load-back
    extents) enters."""
    queued = tuple(sorted(str(getattr(r, "rid", "")) for r in (admissible or ())))
    chunked = getattr(sched, "chunked_req", None)
    return queued + (("~chunked", str(getattr(chunked, "rid", ""))) if chunked is not None else ())


def _cap_lift_verdict(sched, ms, fr, alloc, tokens, used: int, incoming: int, admissible, rids,
                      page: int):
    """NF1d / 1528: does a pending shrink's cap go back up to the mapped stage
    for the demand of this iteration? Sets ``ms._cap_lifted`` and the floor
    cache; returns (lifted, need, rest). The group verdict (``_room_ok``) is
    replicated, so every rank calls this in the same iterations. Shared by
    ``runtime_tick`` and ``cap_lift_after_seat_move``."""
    from flliper.srt.pdflip.park import chunk_admit_tokens

    lifted_before = bool(getattr(ms, "_cap_lifted", False))
    lifted = False
    need = 0
    rest = 0
    if ms.pending is not None and alloc is not None and (incoming > 0 or used > 0):
        chunk = int(getattr(getattr(sched, "server_args", None), "chunked_prefill_size", 0) or 0)
        # NF 1540 (cand3 bce16a6ddf 23:45:03Z, pdflip-62-415: 26048 host-backed
        # prefix + 1572): the host load-back is NOT chunked -- ``add_one_req`` /
        # ``chunk_admit_tokens`` (Q-700) charge it WHOLE, the allocator takes it
        # at admission. ``first`` as one chunk counted 4096 against room 26880,
        # the load-back took 26048 of it and the 1572 extend found 832.
        first = chunk_admit_tokens(int(incoming), chunk, load_back=_queued_load_back(admissible))
        # NF 1528 (K2 20:49:25Z, pdflip-54-290: 2176 + 830): the request after its
        # first chunk is the chunked_req, in ``used`` and no longer in
        # ``incoming`` -- the lift paid for chunk 1 only, ended itself between
        # the chunks (need=64) and chunk 2 found 192 free ids under the cap.
        # The next chunk of a live chunked_req is demand like a first chunk.
        rest = _chunked_rest(sched)
        nxt = min(rest, chunk) if chunk > 0 else rest
        need = first + nxt + len(rids) * max(1, page)
        # NF 1536: the cache bypass of ``_room_ok`` (``incoming <= 0``) must be a
        # RANK-UNIFORM bit. ``rest`` is read from the rank's own prefix_indices /
        # extend_range: 0 on one rank and > 0 on another would send one rank to the
        # group collective and the other to its cache (collective #0 without partners).
        # A live chunked_req is replicated (it is in ``used``), so the bit is that.
        chunk_live = 1 if getattr(sched, "chunked_req", None) is not None else 0
        lifted = not _room_ok(sched, ms, fr, alloc, tokens[ms.pending], need,
                              incoming + chunk_live, len(rids), page,
                              demand=_demand_fingerprint(sched, admissible))
    ms._cap_lifted = lifted
    if lifted:
        fr["lifted_since"] = True
    elif lifted_before:
        fr["floor"] = None                      # the lift ended: read the floor afresh
    return lifted, need, rest


def cap_lift_after_seat_move(sched) -> None:
    """NF 1534 F6 (4p5am6 22:10:15Z TP0: ``D-SEAT-REWAKE GROW n=2->3`` between chunk 2
    and chunk 3 of pdflip-40-295 at a pending S3): the iteration of a D-SEAT-REWAKE
    GROW/SHRINK skips the stage tick (``d_seat_rewake.round_boundary``) but admission
    runs on in the same iteration, so a pending shrink's cap stayed down for the next
    chunk (K2: 13 of 25 GROWs at a pending cap with lifted=no). This runs ONLY the
    cap-lift verdict of the tick, on the machine the last tick built (no stage step, no
    shrink, no apply_stage; a missing machine or one built for another seat count waits
    for the next tick). Same gates as the tick: not armed / dormant / no stage form ->
    the plain noop count, so P, the 27B and a D without a form never reach the verdict."""
    if not armed() or not elastic_on() or getattr(sched, "pdflip_dormant", False):
        return _tick_noop(sched)
    st = getattr(sched, PHASE_ATTR, None)
    form = stage_form()
    ms = getattr(sched, MEM_SCHED_ATTR, None)
    if st is None or st.stage is None or not st.done or form is None or ms is None:
        return _tick_noop(sched)
    n = max(1, min(int(st.n or st.cap), int(st.cap)))
    tokens = tuple(form.tokens[: form.max_stage(n) + 1])
    if ms.stage_tokens != tokens or getattr(ms, "_epoch", None) != st.epoch or ms.pending is None:
        return _tick_noop(sched)
    alloc = _kv_allocator(sched)
    if alloc is None:
        return _tick_noop(sched)
    page = _page_size(sched)
    running, admissible = _demand_lists(sched)
    used = sum(_req_tokens(r) for r in running)
    incoming = sum(_req_tokens(r) for r in admissible)
    rids = frozenset(getattr(r, "rid", id(r)) for r in running)
    fr = _floor_room_state(ms)
    lifted_before = bool(getattr(ms, "_cap_lifted", False))
    lifted, need, rest = _cap_lift_verdict(sched, ms, fr, alloc, tokens, used, incoming, admissible, rids, page)
    if lifted == lifted_before:
        return None
    _engage_kv_cap(alloc, tokens[ms.stage if lifted else ms.pending], page)
    if lifted:
        _reopen_admission(sched)
    room = free_tokens_below(alloc, tokens[ms.pending], page)
    logger.info(
        "%s pending=S%s stage=S%d cap=%d used=%d incoming=%d rest=%d need=%d room=%s lifted=%s "
        "-- seat move iteration (no stage tick): %s", CAP_LIFT_MARK, ms.pending, ms.stage,
        tokens[ms.stage if lifted else ms.pending], used, incoming, rest, need,
        "-" if room is None else room, "yes" if lifted else "no",
        "the free ids below the pending cap do not pay the demand: the mapped stage pays it"
        if lifted else "no demand left: the pending cap again")
    return None


#: S77-A: the D-MEM-SCHED KV compaction (``pdflip/d_kv_compact.py``)
KV_COMPACT_MARK = "PDFLIP D-MEM-SCHED KV-COMPACT"
_KV_COMPACT_N = [0]


class PdFlipKvCompactDivergence(RuntimeError):
    """S77-A N3: the ranks' KV-COMPACT plans or outcomes differ -- a named crash-stop
    (RAENGE-NIE-UNEINS: never "refuse and go on" after a rank-local mutation)."""


#: per-REQUEST rings a page move does not touch (indexed ``req_pool_idx * ratio``)
_KV_REQUEST_RINGS = ("qsa_key_state_buffer_pool", "qsa_rope_position_buffer")
#: per-layer lists whose FIRST dim is the token slot (rows), with the page's tokens
_KV_ROW_LISTS = ("k_buffer", "v_buffer", "k_scale_buffer", "v_scale_buffer",
                 "kv_buffer", "kv_scale_buffer")
#: sub-objects that are no token-indexed KV (GDN states: slots per request, moved by
#: d_seat_compact; the saver adapter; device descriptors)
_KV_SKIP_ATTRS = ("mamba_pool", "memory_saver_adapter", "_kv_buffer_descs", "layer_transfer_counter")


def _is_pool(x) -> bool:
    if x is None or isinstance(x, (int, float, str, bytes, bool)):
        return False
    try:
        from flliper.srt.mem_cache.memory_pool import KVCache

        if isinstance(x, KVCache):
            return True
    except Exception:  # noqa: BLE001
        pass
    return any(hasattr(x, a) for a in ("k_buffer", "kv_buffer", "qsa_compressed_flat"))


def _runner_pools(obj, out: list, seen_ids: set, depth: int = 0, poolless=None) -> None:
    """Every ``token_to_kv_pool`` behind a draft worker, whatever its shape:
    ``model_runner``, ``draft_runner``, ``draft_model_runner`` and the nested
    ``draft_worker`` (EAGLEWorkerV2: ``draft_worker.draft_worker`` is an
    EagleDraftWorker with ``draft_runner`` and ``draft_worker.model_runner``).
    S98: every runner WITHOUT a pool goes into ``poolless`` (once per object), so
    the caller can tell a declared solo shadow from a pool it cannot find."""
    if obj is None or depth > 3 or id(obj) in seen_ids:
        return
    seen_ids.add(id(obj))
    for attr in ("model_runner", "draft_runner", "draft_model_runner"):
        r = getattr(obj, attr, None)
        pool = getattr(r, "token_to_kv_pool", None) if r is not None else None
        if pool is not None:
            out.append(pool)
        elif r is not None and poolless is not None and all(r is not x for x in poolless):
            poolless.append(r)
    _runner_pools(getattr(obj, "draft_worker", None), out, seen_ids, depth + 1, poolless)


def _declared_solo_shadow(runner) -> bool:
    """S98: ``runner`` declares itself a solo-draft SHADOW (``ModelRunner.
    is_draft_solo_shadow``, ``compute_draft_solo_role``: "meta-device draft, no draft
    weights/KV/graphs"). Read as a declaration, never inferred from the missing pool:
    ``kv_cache_builder.get_draft_kv_pool`` answers None for a shadow by design."""
    return getattr(runner, "is_draft_solo_shadow", False) is True


def _kv_pools_of(sched) -> list:
    """(name, pool) of the target's KV cache and of EVERY draft pool (MTP/EAGLE write
    at the target's page ids). A draft worker whose pool cannot be found is named
    by a ``(\"draft?\", None)`` entry -- the walker refuses on it (fail-closed).

    S98 (NF rc22 9c40fe40a2, boot 1020_094747): a solo-draft SHADOW holds no draft KV
    by design -- ``Draft-solo placement: rank 1 is a draft SHADOW (meta-device draft,
    no draft weights/KV/graphs ...)``; the solo host TP0 holds the draft rows of every
    global token (``#tokens: 524288``) and moves them. rc22 refused every compaction on
    TP1/TP2 with ``draft?: pool not found``. A draft side whose EVERY pool-less runner
    declares ``is_draft_solo_shadow`` therefore has nothing token-indexed here; one
    pool-less runner that declares nothing still makes the entry (fail-closed), and a
    pool a declared shadow does hold is walked like any other."""
    out = []
    alloc = _kv_allocator(sched)
    kv = None
    if alloc is not None:
        get = getattr(alloc, "get_kvcache", None)
        kv = get() if callable(get) else getattr(alloc, "_kvcache", None)
    runner = getattr(getattr(sched, "tp_worker", None), "model_runner", None)
    if kv is None:
        kv = getattr(runner, "token_to_kv_pool", None)
    out.append(("target", kv))
    dw = getattr(sched, "draft_worker", None)
    if dw is not None:
        pools: list = []
        poolless: list = []
        _runner_pools(dw, pools, set(), poolless=poolless)
        uniq = []
        for p in pools:
            if p is not kv and all(p is not q for q in uniq):
                uniq.append(p)
        shadow_only = bool(poolless) and all(_declared_solo_shadow(r) for r in poolless)
        if not uniq and not any(p is kv for p in pools) and not shadow_only:
            out.append(("draft?", None))
        for i, p in enumerate(uniq):
            out.append(("draft" if i == 0 else "draft%d" % i, p))
    return out


def _slot_counts(alloc, pool) -> set:
    """The sizes a token-slot dimension of ``pool`` can have (exact): the global
    slot space (+ page), its pages, a compacted owner share, QSA groups."""
    P = int(getattr(pool, "page_size", 0) or getattr(alloc, "page_size", 0) or 1)
    out = set()
    for size in (getattr(alloc, "size", None), getattr(pool, "size", None),
                 getattr(pool, "qsa_slot_space", None)):
        if size:
            out.update({int(size), int(size) + P, (int(size) + P) // P, int(size) // P + 1})
    cap = getattr(pool, "qsa_compressed_capacity", None)
    if cap:
        out.add(int(cap))
    return {x for x in out if x > 1}


#: S98: ``_row_geom``'s answer for a buffer that holds no token (see there)
KV_ROWS_NO_TOKEN = "no-token"


def _row_geom(name: str, t, pool, alloc, P: int):
    """A per-layer buffer whose FIRST dim is the token slot -> (KvTensorGeom, why);
    ``(None, KV_ROWS_NO_TOKEN)`` for a buffer that provably holds no token.

    S98 (NF rc22 9c40fe40a2, boot 1020_094747, TP0): ``target.full_kv_pool.k_buffer[0]:
    64 rows match no slot geometry of the 524352-token space`` -- every compaction
    refused. TP0 is the Form A attention host with owner share 0 (``#239 S3a ZERO-SHARE
    CELL: rank 0 owns no full-attention token (vector [0, 3, 5])``):
    ``dcp_compact_pool_rows`` sizes its FA pool at 0 tokens (``KV Cache is allocated
    ... #tokens: 0``) and an MHA pool always carries ``size + page_size`` rows. Those
    rows are the pad page alone (page 0, never handed out, never a move's source or
    destination); every allocator page id >= 1 lies past the buffer, so no token can
    live in it -- structurally, whatever the form. Exactly ``pool.size == 0`` and at
    most one page of rows (a page-major layout: one row); anything more still refuses
    by name below."""
    rows = int(t.shape[0])
    size = getattr(pool, "size", None)
    if size is not None and int(size) == 0 and 0 < rows <= P:
        return None, KV_ROWS_NO_TOKEN
    nbytes = int(t.numel()) * int(t.element_size())
    if rows <= 0 or nbytes % rows:
        return None, "%s: %d bytes not whole rows of %d" % (name, nbytes, rows)
    sb = nbytes // rows
    G = int(getattr(alloc, "size", 0) or 0) + P            # the global slot space (+ pad page)
    S, ratio = owner_block_for(int(getattr(pool, "size", 0) or 0))
    if S > 0 and ratio > 0:                                # #239 S3g: this rank's owner share
        return KvTensorGeom(SlotTensorGeom(name, 1, rows, sb, nbytes), 1, P, (S, ratio)), None
    if rows == G:                                          # one row per global token
        return KvTensorGeom(SlotTensorGeom(name, 1, rows, sb, nbytes), 1, P), None
    if rows * P == G:                                      # page-major: one row per page
        return KvTensorGeom(SlotTensorGeom(name, 1, rows, sb, nbytes), P, P), None
    return None, "%s: %d rows match no slot geometry of the %d-token space" % (name, rows, G)


def kv_listener_refusal(alloc) -> Optional[str]:
    """S84: may a page move run under this allocator's slot-lifetime listeners?
    None = yes; else why not (named).

    By the listener's DECLARED role (``allocator.base.mark_listener``), never by
    its existence -- rc19 NF (boot 1019_230333) refused all 17 compactions per rank
    with "allocator._free_listeners armed" although the only listener was the KV
    stage cap itself (``KvRowCap``, engaged at the boot by ``kv_stage_boot_cap``):
    an id-set filter that follows every free()/clear(). Such a CAP FILTER is
    compatible: ``d_kv_compact.survey`` takes no destination above its cap and
    counts its withheld ids as neither live nor free, and the sources go back
    through ``free()``, where the filter withholds those above the cap again.
    A SLOT MIRROR (DFlash ``DraftKVSlotMapper``) refuses, and so does any listener
    that declares nothing (fail-closed) and any radix-dedup alias listener (only
    the DFlash window pool's draft-row carry subscribes there). Replicated: every
    D rank registers the same listeners (server config); a rank that differs says
    "no" into the group MIN of ``_kv_compact_for_shrink`` -- no rank moves."""
    from flliper.srt.mem_cache.allocator.base import (
        LISTENER_ROLE_CAP_FILTER,
        LISTENER_ROLE_SLOT_MIRROR,
        listener_role,
    )

    if getattr(alloc, "_alias_listeners", None):
        return ("allocator._alias_listeners armed (the radix-dedup carry of a slot-lifetime "
                "mirror, DFlash DraftKVSlotMapper) -- its rows would not follow a page move")
    for on_free, on_clear in list(getattr(alloc, "_free_listeners", None) or ()):
        role = listener_role(on_free)
        if role == LISTENER_ROLE_CAP_FILTER and (
                on_clear is None or listener_role(on_clear) == LISTENER_ROLE_CAP_FILTER):
            continue
        name = getattr(on_free, "__qualname__", None) or type(on_free).__name__
        if role == LISTENER_ROLE_SLOT_MIRROR:
            return ("allocator._free_listeners: a slot-lifetime mirror is armed (%s, e.g. DFlash "
                    "DraftKVSlotMapper) -- its per-slot rows would not follow a page move" % name)
        return ("allocator._free_listeners: listener %s declares no role (role=%s, clear role=%s) "
                "-- unknown semantics, no page move (fail-closed)"
                % (name, role, listener_role(on_clear)))
    return None


def kv_page_buffers(sched):
    """S77-A N1/N2: EVERY token-indexed device buffer of every KV pool of this rank,
    each with the geometry its POOL gives it -- (list of (name, tensor, geom), why).

    Structural, fail-closed: every tensor attribute / tensor list of every pool
    (sub-pools recursed) is looked at; a buffer with a token-slot dimension is
    either one the walker knows how to move (the per-layer row lists, the QSA
    compressed keys) or the walk REFUSES by its name. Named exemptions only: the
    per-request QSA rings, the GDN pool, the pad page of a size-0 pool (an empty Form A
    owner share, S98) and a draft side of declared solo shadows (S98). Also refused: a
    draft worker whose pool is not found, a slot-lifetime mirror (DFlash ``DraftKVSlotMapper`` free/alias
    listeners) or a listener of unknown role (``kv_listener_refusal``, S84; the KV
    stage cap's own id-set filter passes), HiSparse's index mapping. "Stage-born" plays no part here: TP0 on
    NF holds the QSA index and the draft for every global token without a single
    born tensor (#251c born=0, #239 S3a ZERO-SHARE CELL)."""
    import torch

    alloc = _kv_allocator(sched)
    if alloc is None:
        return [], "no allocator"
    P = int(getattr(alloc, "page_size", 0) or _page_size(sched))
    why = kv_listener_refusal(alloc)
    if why is not None:
        return [], why
    if getattr(alloc, "full_to_hisparse_device_index_mapping", None) is not None or \
            "HiSparse" in type(alloc).__name__:
        return [], "HiSparse device index mapping (%s)" % type(alloc).__name__
    found, seen_t, seen_p = [], set(), set()

    def walk(pool, name):
        if pool is None:
            raise KvCopyRefused("%s: pool not found" % name)
        if id(pool) in seen_p:
            return
        seen_p.add(id(pool))
        counts = _slot_counts(alloc, pool)
        flat = getattr(pool, "qsa_compressed_flat", None)
        flat_st = None
        if isinstance(flat, torch.Tensor) and flat.numel():
            flat_st = flat.untyped_storage().data_ptr()
        if getattr(pool, "full_to_hisparse_device_index_mapping", None) is not None:
            raise KvCopyRefused("%s: HiSparse index mapping" % name)
        for attr, val in sorted(vars(pool).items()):
            here = "%s.%s" % (name, attr)
            if attr in _KV_SKIP_ATTRS or attr in _KV_REQUEST_RINGS:
                continue
            if _is_pool(val):
                walk(val, here)
                continue
            if isinstance(val, dict):
                items = [("%s[%r]" % (here, k), v) for k, v in val.items()]
            elif isinstance(val, (list, tuple)):
                items = [("%s[%d]" % (here, i), v) for i, v in enumerate(val)]
            else:
                items = [(here, val)]
            for nm, t in items:
                if _is_pool(t):
                    walk(t, nm)
                    continue
                if not isinstance(t, torch.Tensor) or t.numel() == 0:
                    continue
                if id(t) in seen_t:
                    continue
                seen_t.add(id(t))
                if attr in _KV_ROW_LISTS:
                    g, why = _row_geom(nm, t, pool, alloc, P)
                    if g is None and why == KV_ROWS_NO_TOKEN:
                        continue   # S98: an empty owner share's pad page -- nothing to move
                    if g is None:
                        raise KvCopyRefused(why)
                    found.append((nm, t, g))
                    continue
                if attr == "qsa_compressed_flat":
                    cap = int(getattr(pool, "qsa_compressed_capacity", 0) or 0)
                    r = int(getattr(pool, "qsa_compress_ratio", 0) or 0)
                    nb = int(t.numel()) * int(t.element_size())
                    L = int(t.shape[0])
                    if cap <= 0 or r <= 0 or nb % (L * cap):
                        raise KvCopyRefused("%s: QSA geometry unreadable" % nm)
                    found.append((nm, t, KvTensorGeom(SlotTensorGeom(nm, L, cap, nb // (L * cap), nb), r, P)))
                    continue
                if attr == "qsa_compressed_k_buffer_pool" and flat_st is not None and \
                        t.untyped_storage().data_ptr() == flat_st:
                    continue  # per-layer views of qsa_compressed_flat
                if any(int(d) in counts for d in t.shape):
                    raise KvCopyRefused("%s %s: a token-slot dimension the page move does not "
                                        "know how to follow" % (nm, tuple(t.shape)))
    try:
        for name, pool in _kv_pools_of(sched):
            walk(pool, name)
    except KvCopyRefused as exc:
        return [], str(exc)
    return found, None


def kv_compact_unregistered(sched, born_ptrs=None) -> Optional[str]:
    """None when every token-indexed KV buffer of this rank is one the page move
    follows (``kv_page_buffers``); else why not. ``born_ptrs`` is ignored (kept for
    callers of the first form): being stage-born is not the question (review N1)."""
    return kv_page_buffers(sched)[1]


def _kv_compact_tree_ok(sched) -> bool:
    """S77-A: this D runs the unified tree KV-COMPACT moves pages in (config)."""
    from flliper.srt.pdflip import d_kv_compact as KC

    return KC.tree_why(getattr(sched, "tree_cache", None)) is None


#: S98c: how often the ranks' page plans differed (uniform refusal, nothing moved)
_KV_COMPACT_PLAN_DIFF = [0]


def _kv_compact_rank_survey(sched, floor_page: int):
    """Rank-local: (survey, why, copier) -- the survey above ``floor_page`` or why
    this rank cannot take part (no allocator, a buffer the move cannot follow, ...)."""
    from flliper.srt.pdflip import d_kv_compact as KC

    page = _page_size(sched)
    alloc = _kv_allocator(sched)
    if alloc is None:
        return None, "no allocator", None
    bufs, why = kv_page_buffers(sched)
    if why is not None:
        return None, why, None
    try:
        copier = KvPageCopier.from_buffers(bufs, page)
    except KvCopyRefused as exc:
        return None, "copier: %s" % exc, None
    sv = KC.survey(getattr(sched, "tree_cache", None), alloc, page, int(floor_page))
    return sv, sv.why, copier


def _kv_compact_vote(sv, why, floor_page: int, top_page: int) -> List[int]:
    """S98: this rank's ONE vote vector (fixed length from the replicated floor/top):
    [ok] + destinations (MIN) + page codes (MIN) + negated page codes (-> MAX)."""
    from flliper.srt.pdflip import d_kv_compact as KC

    B, T = int(floor_page), int(top_page)
    if sv is None:
        return [0] + [0] * B + [KC.CODE_BLOCK] * (T - B) + [-KC.CODE_BLOCK] * (T - B)
    dest, above, w = KC.page_codes(sv, T)
    ok = 1 if (why is None and w is None) else 0
    return [ok] + dest + above + [-c for c in above]


def _kv_compact_group_vote(sched, vec: List[int]) -> List[int]:
    """The element-wise group MIN of the vote (the ONE collective of the survey step;
    local without a group). A seam of its own so a test can play a peer."""
    gmin = getattr(sched, "_pdflip_group_min_ints", None)
    out = list(vec) if gmin is None else [int(x) for x in gmin(list(vec))]
    if len(out) != len(vec):
        return [0] * len(vec)
    return out


def _kv_compact_plans_from_vote(sv, group: List[int], floor_page: int, top_page: int,
                                boundaries: Sequence[int]):
    """(plans, why) per candidate boundary, planned on the GROUP-agreed survey."""
    from flliper.srt.pdflip import d_kv_compact as KC

    B, T = int(floor_page), int(top_page)
    if sv is None or int(group[0]) != 1:
        return [None] * len(boundaries), ""
    dest = group[1:1 + B]
    lo = group[1 + B:1 + B + (T - B)]
    hi = [-int(x) for x in group[1 + B + (T - B):]]
    agreed = KC.agreed_survey(sv, T, dest, lo, hi)
    plans, why = _Plans(), ""
    for b in boundaries:
        pl, w = KC.plan_for(agreed, int(b))
        plans.append(pl)
        plans.whys.append("" if pl is not None else w)
        if pl is None and not why:
            why = w
    return plans, why


class _Plans(list):
    """S98e: the plans per candidate, carrying each candidate's reason (``whys``) so
    the refusal names every candidate stage, not only the first."""

    def __init__(self, *a):
        super().__init__(*a)
        self.whys: List[str] = []


def _kv_compact_stage_reasons(cands, whys) -> str:
    """'S1..S7: <reason>; S0: <reason>' -- consecutive equal reasons folded."""
    out, run = [], []
    for jj, w in zip(cands, whys):
        w = re.sub(r"\b\d{3,}\b", "N", w or "plan ok")[:110]
        if run and run[-1][1] == w:
            run[-1][0].append(jj)
        else:
            run.append(([jj], w))
    for js, w in run:
        out.append("S%d%s: %s" % (js[0], "" if len(js) == 1 else "..S%d" % js[-1], w))
    return "; ".join(out)


def _kv_compact_page_witness(sched, sv, page: int, vec=None, floor_page: int = 0) -> str:
    """S98e: what holds ``page`` on THIS rank, read where it lives -- rank-local, no
    collective, logged only on a refusal. Kinds: free / release list / staged free /
    withheld by a cap / tree node (id, lock, write-through, load-back, backed, two
    owners) / a running or waiting request (rid, from req_to_token) / nobody known."""
    import torch

    from flliper.srt.pdflip import d_kv_compact as KC

    P = _page_size(sched)
    alloc = _kv_allocator(sched)
    p = int(page)
    kinds = []
    if vec is not None and p > int(floor_page):
        B = int(floor_page)
        T = (len(vec) - 1 - B) // 2 + B
        if p <= T:
            kinds.append("vote=%d" % int(vec[1 + B + (p - B - 1)]))
    for name in ("free_pages", "release_pages"):
        ids = KC._id_tensor(getattr(alloc, name, None))
        if bool((ids == p).any()):
            kinds.append(name)
    for t in (getattr(alloc, "free_group", None) or []):
        if bool((KC._id_tensor(t) // P == p).any()):
            kinds.append("staged-free(free_group)")
            break
    for cap in KC.cap_filters(alloc):
        if bool((KC._id_tensor(getattr(cap, "_withheld", None)) == p).any()):
            kinds.append("withheld(cap=%s)" % getattr(cap, "cap", "?"))
    np_ = getattr(sv, "owner", {}).get(p) if sv is not None else None
    if np_ is not None:
        node = np_.node
        nid = getattr(node, "id", "?")
        tree = getattr(sched, "tree_cache", None)
        locks = [int(getattr(cd, "lock_ref", 0) or 0) for cd in node.component_data]
        kinds.append("tree(node=%s locks=%s wt=%s lb=%s backed=%s pinned=%s)" % (
            nid, locks, nid in (getattr(tree, "ongoing_write_through", None) or {}) or
            getattr(node, "write_through_pending_id", None) is not None,
            nid in (getattr(tree, "ongoing_load_back", None) or {}), np_.backed, np_.pinned))
    rtt = getattr(_req_pool(sched), "req_to_token", None)
    reqs = list(getattr(getattr(sched, "running_batch", None), "reqs", None) or [])
    for r in reqs + list(getattr(sched, "waiting_queue", None) or []):
        idx = getattr(r, "req_pool_idx", None)
        if rtt is None or idx is None:
            continue
        try:
            n = int(len(getattr(r, "origin_input_ids", ()) or ()) + len(getattr(r, "output_ids", ()) or ()))
            row = rtt[int(idx), :max(1, n)].to("cpu", torch.int64) // P
            if bool((row == p).any()):
                kinds.append("request(rid=%s %s)" % (getattr(r, "rid", "?"),
                                                    "running" if r in reqs else "waiting"))
        except Exception:  # noqa: BLE001 -- a witness never fails the tick
            continue
    return ", ".join(kinds) or "nobody known here (live, no tree node, no request row)"


def _kv_compact_rank_plans(sched, boundaries: Sequence[int], top_page: Optional[int] = None):
    """Single-rank form (no group): (plans, why, copier) -- one plan (or None) per
    candidate boundary, planned on this rank's own vote as if it were the group's."""
    B = min(int(b) for b in boundaries)
    alloc = _kv_allocator(sched)
    T = int(top_page) if top_page is not None else int(getattr(alloc, "num_pages", 0) or 0)
    T = max(T, max(int(b) for b in boundaries))
    sv, why, copier = _kv_compact_rank_survey(sched, B)
    vec = _kv_compact_vote(sv, why, B, T)
    if vec[0] != 1:
        return [None] * len(boundaries), why or "this rank cannot vote", copier
    plans, w = _kv_compact_plans_from_vote(sv, vec, B, T, boundaries)
    return plans, w, copier


def _group_same(sched, value: int) -> Tuple[int, int]:
    """(MIN, MAX) of one int over the group (one collective); local without a group."""
    gmin = getattr(sched, "_pdflip_group_min_ints", None)
    if gmin is None:
        return int(value), int(value)
    r = [int(x) for x in gmin([int(value), -int(value)])]
    return r[0], -r[1]


def _kv_compact_for_shrink(sched, ms, fr, want: int, end: bool, floor_tokens: int) -> Optional[int]:
    """S77-A: the floor blocks the shrink to ``want``: move the tree's idle pages
    above the boundary down (``d_kv_compact``), then return the GROUP floor read
    afresh -- or None when no rank moved.

    REPLICATED entry: MemSched calls this only on a shrink the GROUP floor blocks.
    Every input of the decision is replicated (want, end, the floor, the stage); a
    refused (or fruitless) attempt is not repeated for the same (want, stage)
    before the next end event -- also while the shrink is pending.

    Collectives, all entered by every rank on the same path:
    1. S98: the VOTE, one MIN over [ok, destinations free here, page codes, negated
       page codes] (``_kv_compact_vote``); every rank plans every candidate on the
       agreed survey (``d_kv_compact.agreed_survey``: backed = group AND) and picks
       the lowest stage it clears (moves only first) -- replicated input, same pick;
    2. N3 the plan DIGEST (jj, page pairs, evicted pages) as MIN/MAX, entered by
       every rank after a yes vote (-1 without a plan): S98c unequal = a uniform,
       named refusal (nothing moved yet), no longer a crash-stop;
    3. N3 the MOVES' outcome as MIN/MAX: all moved -> go on; none moved (each
       rolled back) -> refused, nothing changed anywhere; mixed -> crash-stop;
       S98c: any other exception in the move step enters as -1 (state unknown) ->
       the named crash-stop on every rank, never a rank that skips the collective;
    4. only then the evictions (the d_seat_compact order: move first), outcome
       MIN/MAX: anything but "all evicted" -> crash-stop;
    5. the floor re-read (MAX)."""
    from flliper.srt.pdflip import d_kv_compact as KC

    tokens = ms.stage_tokens
    key = (int(want), int(ms.stage))
    if not end and fr.get("compact_refused") == key:
        fr["compact_skipped"] = int(fr.get("compact_skipped", 0)) + 1
        return None
    f = ms._smallest_holding(int(floor_tokens))
    top = int(ms.stage) if f is None else min(int(ms.stage), int(f))
    cands = list(range(int(want), top))
    if not cands:
        return None
    page = _page_size(sched)
    t0 = time.perf_counter()
    bounds = [int(tokens[jj]) // page for jj in cands]
    floor_page = min(bounds)
    top_page = int(tokens[int(ms.stage)]) // page
    # S98: ONE collective -- the vote. Every rank plans on the group-agreed survey
    # (backed = group AND, blocking = group OR, destinations = free on every rank),
    # so the pick, the moves and the evictions are the same on every rank whose tree
    # holds the same nodes; the digest below still checks exactly that.
    try:
        sv, why, copier = _kv_compact_rank_survey(sched, floor_page)
    except Exception as exc:  # noqa: BLE001 -- a survey failure is this rank's "no"
        sv, why, copier = None, "%s: %s" % (type(exc).__name__, exc), None
    vec = _kv_compact_vote(sv, why, floor_page, top_page)
    if vec[0] != 1 and not why:
        why = "this rank cannot vote"
    group = _kv_compact_group_vote(sched, vec)
    try:
        plans, pwhy = _kv_compact_plans_from_vote(sv, group, floor_page, top_page, bounds)
    except Exception as exc:  # noqa: BLE001 -- replicated input: every rank fails alike
        plans, pwhy = [None] * len(cands), "%s: %s" % (type(exc).__name__, exc)
    if not why:
        why = pwhy if int(group[0]) == 1 else ""
    k = len(cands)
    anys = [1 if p is not None else 0 for p in plans]
    pick = next((i for i in range(k) if plans[i] is not None and not plans[i].evicts), None)
    if pick is None:
        pick = next((i for i in range(k) if plans[i] is not None), None)
    _KV_COMPACT_N[0] += 1
    n = _KV_COMPACT_N[0]

    def _refused():
        fr["compact_refused"] = key
        if n <= 16 or (n & (n - 1)) == 0:
            logger.info("%s n=%d verdict=refused want=S%d stage=S%d floor=%d end=%s local=%s -- %s",
                        KV_COMPACT_MARK, n, int(want), int(ms.stage), int(floor_tokens),
                        "yes" if end else "no", "".join(str(x) for x in anys),
                        why or "a peer rank cannot clear the boundary")
            _refusal_detail()
        return None

    def _refusal_detail():
        # S98e: every candidate stage's reason (the line above names the first only), and
        # for a named page what holds it HERE -- rank-local, no collective
        try:
            whys = getattr(plans, "whys", None)
            if whys is not None and len(whys) == len(cands):
                logger.info("%s n=%d STAGES %s", KV_COMPACT_MARK, n,
                            _kv_compact_stage_reasons(cands, whys))
            pages = sorted({int(x) for x in re.findall(r"page (\d+)", why or "")})
            for pg in pages[:2]:
                logger.info("%s n=%d WITNESS page=%d (tokens from %d) here: %s", KV_COMPACT_MARK, n,
                            pg, pg * page, _kv_compact_page_witness(sched, sv, pg, vec, floor_page))
        except Exception as exc:  # noqa: BLE001 -- an instrument never fails the tick
            logger.info("%s n=%d WITNESS failed: %s: %s", KV_COMPACT_MARK, n, type(exc).__name__, exc)

    if int(group[0]) != 1:          # the group said no: replicated, every rank returns here
        return _refused()
    # S98: the pick comes from the agreed survey. The digest collective is entered on
    # EVERY rank once the group said yes -- with a plan or with the sentinel -1 -- so no
    # rank leaves its peers waiting in a collective it skipped.
    # S98c (rang review nf-review-s98-1020.md, Auflage 3): the digest is over PAGES
    # (d_kv_compact.plan_digest); unequal means the ranks would do different things to
    # the pages -- but NOTHING has moved yet and lo/hi are the reduced values every
    # rank sees alike: a uniform, named refusal, not a crash-stop. The crash-stop stays
    # for what cannot be undone: a mixed move outcome, a failed eviction.
    jj, plan = (cands[pick], plans[pick]) if pick is not None else (-1, None)
    digest = KC.plan_digest(jj, plan) if plan is not None else -1
    lo, hi = _group_same(sched, digest)
    if lo != hi:
        why = ("the ranks' plans for S%d differ (digest here %d, group %d..%d: other node cuts "
               "or another fallback eviction) -- refused on every rank, nothing moved"
               % (jj, digest, lo, hi))
        _KV_COMPACT_PLAN_DIFF[0] += 1
        d = _KV_COMPACT_PLAN_DIFF[0]
        fr["compact_refused"] = key
        if d <= 16 or (d & (d - 1)) == 0:
            logger.warning("%s n=%d verdict=refused-plan-diff diffs=%d want=S%d stage=S%d floor=%d -- %s",
                           KV_COMPACT_MARK, n, d, int(want), int(ms.stage), int(floor_tokens), why)
            _refusal_detail()
        return None
    if plan is None:
        return _refused()
    res = None
    try:
        res = KC.execute_moves(getattr(sched, "tree_cache", None), _kv_allocator(sched), plan, copier)
        ok = 1
    except KC.CompactRefused as exc:   # rolled back on THIS rank
        logger.warning("%s n=%d moves refused here (rolled back): %s", KV_COMPACT_MARK, n, exc)
        ok = 0
    except Exception as exc:  # noqa: BLE001 -- S98c review hint: never leave the peers waiting
        # Anything else (e.g. the final allocator.free of the sources, outside the
        # rollback) leaves THIS rank's state unknown -- nodes possibly re-pointed, sources
        # not freed. The rank still enters the outcome collective, with -1: not 0, which
        # means "rolled back cleanly" (a uniform 0 reads as "nothing moved anywhere").
        # Any -1 in the group is the named crash-stop on EVERY rank, in the same pass --
        # no rank leaves before the collective, so no peer hangs to the gloo timeout.
        logger.error("%s n=%d moves FAILED here (state unknown, not rolled back): %s: %s",
                     KV_COMPACT_MARK, n, type(exc).__name__, exc)
        ok = -1
    lo, hi = _group_same(sched, ok)
    if lo < 0:
        raise PdFlipKvCompactDivergence(
            "%s n=%d: the moves for S%d FAILED on a rank (here: %s) -- page placement there "
            "is unknown and cannot be undone; every rank stops in the same pass" % (
                KV_COMPACT_MARK, n, jj, {1: "moved", 0: "rolled back", -1: "failed"}[ok]))
    if lo != hi:
        raise PdFlipKvCompactDivergence(
            "%s n=%d: the moves for S%d went through on some ranks and were rolled back on "
            "others (here: %s) -- page placement differs" % (KV_COMPACT_MARK, n, jj,
                                                            "moved" if ok else "rolled back"))
    if not ok:
        fr["compact_refused"] = key
        logger.warning("%s n=%d verdict=refused-all target=S%d: every rank rolled back, nothing moved",
                       KV_COMPACT_MARK, n, jj)
        return None
    evicted = (0, 0)
    if plan.evicts:
        try:
            evicted = KC.execute_evicts(getattr(sched, "tree_cache", None), plan)
            ok = 1
        except Exception as exc:  # noqa: BLE001 -- named below as a group stop
            logger.error("%s n=%d eviction failed here: %s: %s", KV_COMPACT_MARK, n,
                         type(exc).__name__, exc)
            ok = 0
        lo, hi = _group_same(sched, ok)
        if lo != 1 or hi != 1:
            raise PdFlipKvCompactDivergence(
                "%s n=%d: the fallback evictions for S%d did not go through on every rank "
                "(here: %s) -- device presence of the tree differs" % (
                    KV_COMPACT_MARK, n, jj, "ok" if ok else "failed"))
    new_floor = int(_group_floor_tokens(sched))
    fr["floor"] = None
    if new_floor >= int(floor_tokens):
        fr["compact_refused"] = key
    else:
        fr.pop("compact_refused", None)
    logger.info(
        "%s n=%d verdict=moved want=S%d target=S%d boundary=%d floor %d->%d moved_nodes=%d "
        "moved_pages=%d evicted_nodes=%d evicted_tokens=%d bytes=%d buffers=%d digest=%d ms=%.1f "
        "(the tree's idle pages above the boundary moved down in every token-indexed KV buffer, "
        "node values re-pointed; evicted = backed nodes with no free page below, host copy stays)",
        KV_COMPACT_MARK, n, int(want), jj, int(tokens[jj]), int(floor_tokens), new_floor,
        res.moved_nodes, res.moved_pages, evicted[0], evicted[1], res.bytes,
        len(getattr(copier, "items", ()) or ()), digest, (time.perf_counter() - t0) * 1000.0)
    return new_floor


def _grow_room_refused(sched, ctl, n: int, stage: int) -> Optional[str]:
    """S77-B: the GROUP verdict on a KV grow, before any rank moves (MIN over the
    ranks, one collective). Entered on every rank alike: the grow itself is
    replicated (MemSched), a rank without a controller or without a measured card
    answers yes. Each rank's own answer is ``SeatVram.grow_room`` -- measured
    bytes, the cache trimmed, its own rows OFF counted in. None = grow; else why not."""
    room = None
    if ctl is not None and getattr(ctl, "cells", None):
        try:
            room = ctl.grow_room(n, stage)
        except PdFlipDSeatVramRefused as exc:
            room = None
            logger.warning("%s n=%d stage=S%d not measured: %s", GROW_ROOM_MARK, n, stage, exc)
    ok = 1 if (room is None or room.ok) else 0
    gmin = getattr(sched, "_pdflip_group_min_ints", None)
    group = ok if gmin is None else int(gmin([ok])[0])
    if group:
        return None
    if room is not None and not room.ok:
        return ("grow to S%d refused: this rank maps %.1f MiB fresh, has %.1f MiB after the cache "
                "trim and %.1f MiB with every expert row OFF -- no rank moves, the admission waits"
                % (stage, room.need / _MIB, room.free_after / _MIB,
                   (room.free_after + room.release_all) / _MIB))
    return "grow to S%d refused: a peer rank has no room for its KV -- no rank moves" % stage


def runtime_tick(sched):
    """Once per scheduler iteration of an AWAKE D (after the write-through
    acks are flushed): the KV stage follows the global demand between wakes
    -- grow when used + incoming + air reaches the stage, shrink at once on
    a finish / abort / park (only below pages no request and no unbacked
    tree node holds), otherwise after K rounds with a stage gap. The same
    stage form and ``apply_stage`` as the wake (#251c/d); the expert rows of
    the stage's cell go OFF coldest first and come back lazily. Returns the
    step, or None when nothing runs (not armed, the stop, asleep, no stage)."""
    if not armed() or not elastic_on() or getattr(sched, "pdflip_dormant", False):
        return _tick_noop(sched)
    st = getattr(sched, PHASE_ATTR, None)
    if st is None:
        st = boot_phase(sched)
    if st is None or st.stage is None or not st.done:
        return _tick_noop(sched)
    form = stage_form()
    if form is None:
        return _tick_noop(sched)
    from flliper.srt.pdflip.d_mem_sched import MemSched
    from flliper.srt.pdflip.park import chunk_admit_tokens

    n = max(1, min(int(st.n or st.cap), int(st.cap)))
    tokens = tuple(form.tokens[: form.max_stage(n) + 1])
    ms = getattr(sched, MEM_SCHED_ATTR, None)
    if ms is None or ms.stage_tokens != tokens or getattr(ms, "_epoch", None) != st.epoch:
        counters = dict(ms.counters) if ms is not None else None
        ms = MemSched(stage_tokens=tokens, air_tokens=_air(sched),
                      stage=min(int(st.stage), len(tokens) - 1))
        if counters:
            ms.counters.update(counters)
        ms._epoch = st.epoch
        ms._rids = None
        setattr(sched, MEM_SCHED_ATTR, ms)
    running, admissible = _demand_lists(sched)
    used = sum(_req_tokens(r) for r in running)
    incoming = sum(_req_tokens(r) for r in admissible)
    rids = frozenset(getattr(r, "rid", id(r)) for r in running)
    # NF y3z ep44/ep58 (30.09.): the machine grows for what the GATE charges,
    # not for the bare census -- else it holds a stage the gate refuses in
    reserve = _agree_reserve(sched, admissible, admission_reserve(sched, running, admissible))
    ended = ms._rids is not None and bool(ms._rids - rids)
    ms._rids = rids
    fr = _floor_room_state(ms)
    fr["ticks"] += 1
    floor_box = [0]

    def _floor():
        floor_box[0] = _floor_for_shrink(sched, ms, fr, ended)
        return floor_box[0]

    before, pending_before = ms.stage, ms.pending
    # S77-B review A1: a grow the group refuses moved nothing -- it must not count
    up_before = (int(ms.counters.get("stage_up", 0)), int(ms.counters.get("stage_flap", 0)))
    # S77-A: a floor that blocks a decided shrink is moved down first (KV-COMPACT) --
    # on the tree that holds the pages (a server-config property: the same on every
    # D rank, so every rank installs it alike and enters its collectives alike)
    ms.compactor = (
        (lambda want, end, cur: _kv_compact_for_shrink(sched, ms, fr, want, end, cur))
        if _kv_compact_tree_ok(sched) else None)
    # the floor is asked for only on a path that shrinks (MemSched calls it)
    step = ms.step(used, incoming + reserve, ended=ended, floor_tokens=_floor)
    floor = floor_box[0]
    if step.changed:
        fr["floor"] = fr["room"] = None
    alloc = _kv_allocator(sched)
    page = _page_size(sched)
    # NF1d 09291811 (z30y3f): a pending shrink caps new pages below the wanted
    # stage -- sound only while the ids below it can pay the demand. After the
    # 245k cell the end event left pending S0 (cap 32768, floor 248832: the
    # retained tree); the first bs2 cell's 2113 tokens fit S0, so nothing
    # cancelled the pending shrink, but every id below 32768 was the tree's
    # and every leaf the peel freed sat above the cap: "EVICTION
    # UNDER-DELIVERED ... the pool received 0", Prefill out of memory on all
    # three D ranks. The stage is still MAPPED above the cap, so while the
    # group has demand the cap goes back to the stage unless the free ids
    # below it pay (replicated verdict); with no demand it returns to pending.
    lifted_before = bool(getattr(ms, "_cap_lifted", False))
    lifted, need, rest = _cap_lift_verdict(sched, ms, fr, alloc, tokens, used, incoming, admissible, rids, page)
    if fr["ticks"] % FLOOR_CHECK_EVERY == 0:
        logger.info(
            "%s ticks=%d floor_reads=%d floor_cached=%d room_reads=%d room_cached=%d "
            "collectives=%d pending=S%s lifted=%s recheck_max=%d used=%d chunk_live=%d",
            FLOOR_CHECK_MARK, fr["ticks"], fr["floor_reads"], fr["floor_cached"],
            fr["room_reads"], fr["room_cached"], fr["floor_reads"] + fr["room_reads"],
            "-" if ms.pending is None else ms.pending, "yes" if lifted else "no",
            _recheck_max(), used,
            # nf-next-1006-30 (observation only, nf-next-1006-28 Fix A tripwire): this
            # rank's own ``used`` and live-chunked_req bit, appended so a later rank
            # comparison can show a split the group verdict would not (no collective).
            1 if getattr(sched, "chunked_req", None) is not None else 0)
    if not step.changed:
        if lifted != lifted_before and alloc is not None:
            want = ms.stage if lifted else ms.pending
            if want is None:
                want = ms.stage
            _engage_kv_cap(alloc, tokens[want], page)
            if lifted:
                _reopen_admission(sched)
            # room= is rank-local (the verdict itself is the group MIN): the free
            # tokens at or below the PENDING cap, i.e. what the lift was weighed on
            room = (free_tokens_below(alloc, tokens[ms.pending], page)
                    if ms.pending is not None else None)
            logger.info(
                "%s pending=S%s stage=S%d cap=%d used=%d incoming=%d rest=%d need=%d "
                "room=%s lifted=%s -- %s",
                CAP_LIFT_MARK, "-" if ms.pending is None else ms.pending, ms.stage,
                tokens[want], used, incoming, rest, need,
                "-" if room is None else room, "yes" if lifted else "no",
                "the free ids below the pending cap do not pay the demand: the mapped "
                "stage pays it" if lifted else "no demand left: the pending cap again")
            return step
        if ms.pending != pending_before and alloc is not None:
            # a shrink the floor blocks: new pages go below the wanted stage.
            # A pending shrink the need CANCELS (pending -> None, the stage
            # holds) hands the cap back up to the mapped stage -- rc12z30y2
            # 09291559 D: the end-event shrink S1->S0 waited on pdflip-2-10's
            # retained page 517, the cap went to 32768, the queue head
            # pdflip-4-11 (33633 tokens, 33024 its locked prefix) cleared the
            # pending shrink and the cap stayed at 32768 -- available_size 0
            # on every rank, #1045 floor 0, H105 host_budget 0 for 13 min.
            want = ms.pending if (ms.pending is not None and not lifted) else ms.stage
            _engage_kv_cap(alloc, tokens[want], page)
            logger.info(
                "%s cap=%d pending S%s->S%s stage=S%d used=%d incoming=%d floor=%d -- %s",
                ms.line(), tokens[want],
                "-" if pending_before is None else pending_before,
                "-" if ms.pending is None else ms.pending, ms.stage, used, incoming,
                floor, step.reason)
        return step
    ctl = controller(sched)
    if ms.stage > before:
        # S77-B: a grow moves no rank until EVERY rank measured the room for it
        # H1 (review S77-B): the vote measured this rank's card; ``apply_stage`` below
        # measures it AGAIN (the vote's own collective is the only work in between) and
        # pays a drift with more of its own rows -- a wall there would need a drift
        # beyond free + every row, which the vote has just excluded
        refused = _grow_room_refused(sched, ctl, n, ms.stage)
        if refused is not None:
            ms.stage = before
            ms.counters["stage_up"], ms.counters["stage_flap"] = up_before
            ms.counters["stage_up_refused_room"] = ms.counters.get("stage_up_refused_room", 0) + 1
            if alloc is not None:
                _engage_kv_cap(alloc, tokens[before], page)
            logger.warning("%s %s from=S%d to=S%d used=%d incoming=%d refused=%d -- %s",
                           ms.line(), GROW_ROOM_MARK, before, before + (step.stage - before),
                           used, incoming, ms.counters["stage_up_refused_room"], refused)
            from flliper.srt.pdflip.d_mem_sched import StageStep

            return StageStep(before, tokens[before], False, True, refused)
    if alloc is not None:
        _engage_kv_cap(alloc, tokens[ms.pending if (ms.pending is not None and not lifted)
                                     else ms.stage], page)
    if ctl is not None and ctl.cells:
        # S106 STAGE-KEEP: the lend's own price (rank-local amount; the move's verdict is the
        # group's, replicated above -- nothing here enters a collective)
        from flliper.srt.pdflip import d_transient_lend as _dtl

        need = _dtl.stage_keep_need_bytes(sched, ctl)
        if need is None:
            applied = ctl.apply_stage(n, ms.stage)
        else:
            lend_rows_before = int(getattr(ctl, "rows_on", 0) or 0)
            applied = ctl.apply_stage(n, ms.stage, keep_lend_need=need)
            _dtl.after_phase_move(sched, ctl, lend_rows_before)
    else:
        applied = None
    st.stage, st.stage_tokens = ms.stage, tokens[ms.stage]
    if ms.stage > before:
        _reopen_admission(sched)
    logger.info("%s from=S%d to=S%d used=%d incoming=%d ended=%s floor=%d -- %s%s gate_reserve=%d",
                ms.line(), before, ms.stage, used, incoming, "yes" if ended else "no", floor,
                step.reason,
                "" if ctl is None else (" evicted_rows=%d refilled_rows=%d warm_rows=%d "
                                        "sync_ms=%.1f bytes_copied=%d copy_ms=%.1f rows_on=%s")
                % (ctl.evicted_rows, ctl.refilled_rows, ctl.warm_rows(), ctl.sync_ms,
                   ctl.bytes_copied, ctl.copy_ms,
                   "-" if applied is None else int(applied.extra_rows)),
                int(reserve))
    return step
