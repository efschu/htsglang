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
from dataclasses import dataclass, field
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

    def slots_for(self, tokens: int) -> int:
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
                kv_mapped=kv, cap_mapped=budget, feasible=k >= 0, rows_full=k_full)
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

    def max_stage(self, n: int) -> int:
        top = len(self.tokens) - 1
        if not self.max_by_seats:
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
    from sglang.srt.environ import envs

    tokens = tuple(sorted({t for t in _ints(envs.SGLANG_WEG2_D_KV_STAGE_TOKENS.get()) if t > 0}))
    if len(tokens) < 2:
        return None
    return StageForm(
        tokens=tokens,
        rows_on=max(0, int(envs.SGLANG_WEG2_D_KV_STAGE_ROWS.get() or 0)),
        max_by_seats=_ints(envs.SGLANG_WEG2_D_KV_STAGE_MAX_BY_SEATS.get()),
    )


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
        raise Weg2DSeatVramRefused(
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
        if ctl.form is not None and ctl.cells:
            floors = capture_floors(ctl.form, ctl.cells, ctl.cap)
            logger.info("%s CAPTURE-FLOOR rows ON per bs %s (stages %s, max_by_seats %s): the "
                        "captured decode steps count these seat/stage rows as ON",
                        STAGE_MARK, list(floors), list(ctl.form.tokens),
                        list(ctl.form.max_by_seats) or "all")
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


def kv_stage_pool_tokens(max_tokens: int, *, is_form_a_worker: bool = False,
                         is_draft_worker: bool = False, draft_shares_slots: bool = True) -> int:
    """#251c ``_config_from_budget``: the KV pool's rows are the TOP stage's
    (virtual -- the graphs keep one address range); the pages behind them are
    S0's (``kv_stage_born``). The budget must hold S0 on the attention host:
    below it the boot form the planner priced does not fit, refused by name.
    A Form A worker holds no KV: its (replicated) allocator takes the same
    rows without a budget check. Off: ``max_tokens`` unchanged."""
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
        raise Weg2DSeatVramRefused(
            "%s: the KV stage form needs the draft to write at the target's slot ids; this "
            "draft worker has its own allocator (solo host), whose ids are not capped to "
            "stage S0 -- turn the stage form off (SGLANG_WEG2_D_KV_STAGE_TOKENS) for this "
            "draft form" % LINE_MARK)
    if not is_form_a_worker and int(max_tokens) < form.tokens[0]:
        raise Weg2DSeatVramRefused(
            "%s: the KV budget holds %d tokens, below stage S0 = %d -- the boot form "
            "the planner priced does not fit this rank" % (LINE_MARK, int(max_tokens),
                                                         form.tokens[0]))
    return int(form.tokens[-1])


#: the attention host's KV tensors, trimmed to S0 the moment they are born
#: (``kv_stage_born``), in birth order: (data_ptr, geometry). Process-local;
#: TP0 only (a Form A worker allocates none).
_KV_BORN: List[Tuple[int, KvTensorGeom]] = []


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
    if form is None or t is None or not t.numel() or int(pool_size) < form.tokens[-1]:
        return t
    spans = tms() if spans is None else spans
    info = spans.info(t.data_ptr()) if spans.available else None
    if info is None:
        raise Weg2DSeatVramRefused(
            "%s: the KV tensor %s (%.1f MiB) is not a saver allocation -- its "
            "top-stage pages could not be unmapped and would sit on the card "
            "unpriced. Turn SGLANG_OPT_WEG2_D_SEAT_VRAM off for this boot."
            % (LINE_MARK, name, t.numel() * t.element_size() / _MIB))
    g = int(granule or granule_for(t.device))
    n_slots = int(slots if slots is not None else t.shape[0])
    slot_bytes = (t.numel() * t.element_size()) // max(1, int(layers) * n_slots)
    geom = KvTensorGeom(
        SlotTensorGeom(name, int(layers), n_slots, int(slot_bytes), int(info.size)),
        token_ratio=int(tokens_per_slot), token_pad=int(page_size))
    rc = spans.set_spans(t.data_ptr(), slot_spans(geom.geom, geom.slots_for(form.tokens[0]), g),
                         now=True)
    if rc != 0:
        raise Weg2DSeatVramRefused("%s: trimming the KV tensor %s to stage S0 failed "
                                   "(tms_set_spans rc=%d)" % (LINE_MARK, name, rc))
    _KV_BORN.append((int(t.data_ptr()), geom))
    return t


def kv_stage_boot_cap(allocator, page_size: int) -> Optional[int]:
    """#251c: the allocator of a stage-form pool hands out only S0's pages
    until the first wake says otherwise (every rank, replicated). Returns the
    page cap, None when off."""
    form = stage_form()
    if form is None or allocator is None:
        return None
    return _engage_kv_cap(allocator, form.tokens[0], page_size)


def _engage_kv_cap(allocator, tokens: int, page_size: int) -> int:
    from sglang.srt.managers.kv_backing_relief import KvRowCap

    cap = getattr(allocator, "_weg2_kv_stage_cap", None)
    if cap is None:
        cap = KvRowCap(allocator)
        allocator._weg2_kv_stage_cap = cap
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
    cap = getattr(allocator, "_weg2_kv_stage_cap", None)
    held = getattr(cap, "_withheld", None) if cap is not None else None
    if held is not None and held.numel():
        live[held.to("cpu", torch.int64)] = False
    idx = torch.nonzero(live).flatten()
    return int(idx.max()) if idx.numel() else 0


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
    #: #251c: the phase's KV stage (None without a stage form), its tokens,
    #: the wake's demand and whether it exceeds the highest usable stage
    stage: Optional[int] = None
    stage_tokens: Optional[int] = None
    demand: Optional[int] = None
    over: bool = False


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
    #: #251c: the KV tensors born trimmed (``kv_stage_born``), the stage form,
    #: the KV pools whose flush must stay in the mapped prefix, the cells
    kv_tensors: List[_Managed] = field(default_factory=list)
    form: Optional[StageForm] = None
    kv_pools: List[object] = field(default_factory=list)
    cells: Dict[Tuple[int, int], StageCell] = field(default_factory=dict)
    #: the expert rows ON as last applied (the bank is born at k = 0)
    rows_on: int = 0

    def __post_init__(self):
        x_max = min((int(getattr(c, "seat_rows", 0)) for c in self.caches), default=0)
        self.rows = seat_vram_rows(
            [m.geom for m in self.slot_tensors], [m.geom for m in self.row_tensors],
            cap=self.cap, pool_size=self.pool_size, extra_max=x_max, granule=self.granule)
        if self.form is not None and self.kv_tensors:
            self.cells = stage_vram_cells(
                [m.geom for m in self.slot_tensors], [m.geom for m in self.row_tensors],
                [m.geom for m in self.kv_tensors], cap=self.cap, pool_size=self.pool_size,
                extra_max=x_max, stage_tokens=self.form.tokens,
                boot_rows_on=self.form.rows_on, granule=self.granule)
            check_form_against_cells(self.form, self.cells, self.cap)

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

    def apply_stage(self, n: int, stage: int) -> PhaseApplied:
        """#251c: the pages of a phase of ``n`` seats at KV stage ``stage``
        (``stage_vram_cells``): the GDN slots, the KV prefix and the expert
        rows ON move together, the total never above the boot form. Paused
        allocations get a plan (mapped at their tag's resume), mapped ones move
        in place. A LIVE bank that shrinks turns its rows OFF in the tables
        first and synchronizes before a page goes (a row a kernel may still
        read is never unmapped); one that grows maps first, then turns ON."""
        import torch

        n = max(1, min(int(n), self.cap))
        j = max(0, min(int(stage), len(self.form.tokens) - 1))
        cell = self.cells[(n, j)]
        notes = []
        k, keep = int(cell.extra_rows), int(cell.mamba_keep)
        infos = [self.spans.info(m.ptr) for m in self.slot_tensors]
        if keep <= self.pool_size and any(i is not None and i.active for i in infos):
            if cell.rows_full < 0:
                raise Weg2DSeatVramRefused(
                    "%s: stage S%d at n=%d needs the GDN pages of the empty seats but the "
                    "Mamba pool is mapped (live) -- the stage was chosen for a wake whose "
                    "pool must already be paused" % (LINE_MARK, j, n))
            notes.append("the Mamba pool is mapped (live): it keeps its cap form")
            k, keep = int(cell.rows_full), self.pool_size + 1
        experts_live = any(
            (lambda i: i is not None and i.active)(self.spans.info(m.ptr))
            for m in self.row_tensors)
        shrink = k < int(self.rows_on)
        if shrink:
            for cache in self.caches:
                cache.set_seat_rows_on(k, device_write=experts_live)
            # the OFF writes are stream-ordered device writes: every kernel
            # that could still read a row that goes must be done first
            if experts_live and any(
                    getattr(getattr(getattr(c, "_pool_tables", None), "row_key", None),
                            "is_cuda", False) for c in self.caches):
                torch.cuda.synchronize()
        for m, info in zip(self.slot_tensors, infos):
            if info is not None and not info.active:
                rc = self.spans.set_spans(m.ptr, slot_spans(m.geom, keep, self.granule), now=False)
                if rc != 0:
                    raise Weg2DSeatVramRefused("%s: tms_set_spans(%s) rc=%d"
                                               % (LINE_MARK, m.geom.name, rc))
        if self.mamba_pool is not None and self.slot_tensors:
            self.mamba_pool._weg2_seat_keep = None if keep > self.pool_size else int(keep)
        tokens = int(self.form.tokens[j])
        for m in self.kv_tensors:
            info = self.spans.info(m.ptr)
            live = info is not None and info.active
            rc = self.spans.set_spans(
                m.ptr, slot_spans(m.geom.geom, m.geom.slots_for(tokens), self.granule), now=live)
            if rc != 0:
                raise Weg2DSeatVramRefused("%s: tms_set_spans(%s, now=%s) rc=%d"
                                           % (LINE_MARK, m.geom.geom.name, live, rc))
        page = int(self.kv_tensors[0].geom.token_pad) if self.kv_tensors else 0
        for pool in self.kv_pools:
            # zero_kv_data_buffers (the idle flush) writes only mapped rows
            for sub in (pool, getattr(pool, "full_kv_pool", None)):
                if sub is not None:
                    sub.safe_zero_rows = tokens + page
        for m in self.row_tensors:
            info = self.spans.info(m.ptr)
            live = info is not None and info.active
            rc = self.spans.set_spans(
                m.ptr, row_spans(m.geom, m.geom.rows_boot + k, self.granule), now=live)
            if rc != 0:
                raise Weg2DSeatVramRefused("%s: tms_set_spans(%s, now=%s) rc=%d"
                                           % (LINE_MARK, m.geom.name, live, rc))
        if not shrink:
            for cache in self.caches:
                cache.set_seat_rows_on(k, device_write=experts_live)
        self.rows_on = k
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
            raise Weg2DSeatVramRefused(
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
    ctl = controller(sched)
    if ctl is None:
        applied = None
    elif stage is not None and ctl.cells:
        applied = ctl.apply_stage(n_phys, stage)
    else:
        applied = ctl.apply(n_phys)
    logger.info("%s", phase_line(st, applied, ctl))
    if stage is not None:
        logger.info("%s", stage_line(st, applied))
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
        raise Weg2DSeatOverrun(
            "%s: a forward batch of %d requests in a D phase of n=%d seats (epoch %s) -- "
            "the posts of seats above n are not mapped (slots 1..%s); the admission cap "
            "should have held it" % (REFUSAL_CODE, bs, cap, st.epoch, st.slot_limit))
