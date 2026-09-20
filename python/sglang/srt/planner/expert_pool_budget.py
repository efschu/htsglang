# SPDX-License-Identifier: Apache-2.0
"""Tasks #14/#48 -- the expert pool's LRU rows come from the BUDGET, not the hand.

WHAT THIS PRICES, AND WHAT IT LEARNED NOT TO
--------------------------------------------
Four configuration boots on 2026-09-20, each one a subtraction:

* **fn8ak** (``SGLANG_MOE_SCRATCH_SLOTS=175,60,60``) died before ``/health``:
  ``pool mode requires buffer_size == R+C (97 != 43+60)``.  Rank 2 OWNS 97
  experts and ``buffer_size`` is ``min(R + C, E_local)``
  (``expert_offload.py:504``, checked at ``:3895-3898``): rows above the owned
  count are not a bigger pool, they are a contradiction.
* **fn8ak2** (``175,60,54``) booted and then rank 0 went OOM inside the first
  8192-token prefill chunk (``254 MiB free ... 30.88 GiB in use``).
* **fn8ak3** (``145,60,54`` -- the 5090 untouched, the 3080s +24/+18 rows) died
  in the needle prefill on rank 2: ``73.5 MiB free, 19.36 GiB in use; 18.60 GiB
  allocated by PyTorch, 1.79 GiB in private pools (CUDA Graphs)`` with
  ``expandable_segments:True`` already on.  Not fragmentation: at the real
  prefill peak the 3080 is full.
* **fn8am** carried ``--max-total-tokens 90816`` and refused the needle at
  90810 -- see "THE FLAG IS THE WORLD CAP" below.

THERE IS NO KV CREDIT TO HARVEST (the correction that gutted this term)
-----------------------------------------------------------------------
An earlier version of this module read the ``KV pool sizing`` line --
``available_bytes=4.73 GiB -> max_total_num_tokens=358784`` -- and the
``local capacity 269995`` of the ``Uneven-DCP token sizing`` line as the pool a
rank had ALLOCATED, and concluded that shrinking it to the DCP need would free
~2.37 GiB per rank.  **Both numbers are CAPACITY, not allocation.**  Under the
weighted owner rule the pool is already compacted to
``dcp_compact_pool_rows(C, S, ratio_r) = (C // S + 1) * ratio_r``
(``layers/dcp/owner.py:155``), and the boot says so itself::

    fn8aj: KV Cache is allocated ... #tokens: 90123   (ratio 11, C = 262151)
           KV Cache is allocated ... #tokens: 81930   (ratio 10)
    fn8am: KV Cache is allocated ... #tokens: 31229   (ratio 11, C = 90816)
           KV Cache is allocated ... #tokens: 28390   (ratio 10)

``(262151 // 32 + 1) * 11 = 90123`` exactly.  fn8aj's pool was ALREADY the
1.19 GiB its DCP share needs, so the credit was zero and the 2.37 GiB never
existed -- measured independently: fn8am with the cap had 5.0-5.4 GiB free per
3080 after the pools, the same as without it.  This module now reads the
ALLOCATION line and charges ``(rows_now - rows_needed) * cell``, which is zero
whenever C already matches the context and NEGATIVE when the pool has to grow.

THE FLAG IS THE WORLD CAP, BY DESIGN
------------------------------------
``--max-total-tokens`` is C, the GLOBAL context budget; the per-rank pool is
compacted from it.  fn8am set it to one rank's physical row count (90816) and
so cut the context every request may have -- ``Input length (259415 tokens)
exceeds the maximum allowed length (90810 tokens)``.  That is the flag working
as designed on a wrong input, not a defect: :meth:`Plan.env` emits
``MAX_TOTAL_TOKENS`` as the WORLD pool and nothing else.

WHICH TRANSIENT (three readings, 4x apart, and only one binds)
---------------------------------------------------------------
* ``peak - allocated`` -- what the ``[vram-peak]`` line itself calls the
  transient: 1.92 GiB on fn8am's 3080s.  **Underbooks by 2.3x.**  It counts the
  allocator's live high-water only.
* ``peak_allocated_deep - allocated_at_rest`` (minus the rows the measuring
  boot carried beyond this census) -- **what this module uses**.  fn8ak3 rank 2:
  ``18.60 - 12.06 - 18 rows x 116 MiB = ~4.4 GiB``.
* ``card_free_at_rest - card_free_at_worst`` -- 5.37 - 0.08 = **5.29 GiB** on
  fn8am rank 1.  Larger again, because the allocator RESERVES more than it
  holds live and never returns it to the driver.  Reported beside the figure
  used, with the delta named, because it is the stricter bound and the one the
  OOM actually hit.

So::

    transient_r     = peak_alloc_deep_r - alloc_at_rest_r - extra_rows_r * row_bytes_r
    kv_credit_r     = (rows_now_r - rows_needed_r) * cell_size        # 0, or negative
    headroom_r      = card_free_after_pools_r + kv_credit_r - transient_r - floor
    rows_r          = floor(headroom_r / row_bytes_r)
    C_r             = min(C_now_r + rows_r, E_local_r - R_r)

``row_bytes`` is read from ``[vram-census] experts / (R+C)``, never from a
nominal per-expert size times a layer count: fn8aj/fn8am, three ranks with
three different arenas, give 115.98 / 116.05 / 116.01 MiB -- a 0.06 % spread.

USER LAWS THIS OBEYS
--------------------
* Reserves NEVER ("nicht ein Byte"): :data:`DEFAULT_FLOOR_BYTES` is 0.
* "KEIN BINDENDER RANG": ranks are priced independently.
* Under uneven DCP the pool is quoted as the WORLD total
  (:attr:`Plan.kv_tokens_world`), never as one rank's share.
* Refusal by name: an infeasible rank raises :class:`ExpertPoolBudgetRefused`
  naming the rank, every post and the shortfall.  A census without a measured
  DEEP prefill peak is refused outright (``strict=True``) -- the shallow
  high-water sample is a lower bound, and three boots died on reading one as a
  maximum.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "ExpertPoolBudgetRefused",
    "RankCensus",
    "RankPlan",
    "Plan",
    "GIB",
    "MIB",
    "DEFAULT_FLOOR_BYTES",
    "kv_rows_for_rank",
    "row_bytes_from_census",
    "plan_expert_pool",
    "survey_ranks",
    "verify_scratch_vector",
    "parse_boot_log",
    "dry_run",
]

GIB = float(1 << 30)
MIB = float(1 << 20)

#: User law 2026-09-19, verbatim "Reserven NIE, nicht ein Byte": this term
#: books no safety margin of its own.  A caller may pass ``floor_bytes`` when it
#: has a NAMED reason; the default is zero and stays zero.
DEFAULT_FLOOR_BYTES = 0


class ExpertPoolBudgetRefused(RuntimeError):
    """A rank cannot hold its DCP KV share inside what the card has free.

    Carries the arithmetic so no caller has to re-derive it: which rank, what
    was reachable, every post, and by how much it went negative.
    """

    def __init__(
        self,
        rank: int,
        headroom_bytes: float,
        posts: Sequence[Tuple[str, float]],
        shortfall_bytes: float,
    ):
        self.rank = int(rank)
        self.headroom_bytes = float(headroom_bytes)
        self.posts = tuple((str(n), float(v)) for n, v in posts)
        self.shortfall_bytes = float(shortfall_bytes)
        posted = ", ".join(f"{n}={v / GIB:+.3f} GiB" for n, v in self.posts)
        super().__init__(
            f"rank{self.rank} is infeasible: posts ({posted}) leave "
            f"{self.headroom_bytes / GIB:.3f} GiB, short by "
            f"{self.shortfall_bytes / GIB:.3f} GiB. Nothing is clamped -- lower the "
            f"context, this rank's DCP ratio, or its frozen resident set."
        )


@dataclass(frozen=True)
class RankCensus:
    """Everything about one rank that a previous boot already printed.

    Field -> log line, all from the boot's own instruments:

    ``available_bytes`` / ``cell_size`` / ``page_size``
        ``KV pool sizing: available_bytes=%d ... cell_size=%d, page_size=%d``
        (``pool_configurator.py:577``).  CAPACITY, not allocation.
    ``kv_rows_now``
        ``KV Cache is allocated. ... #tokens: %d`` -- the main pool's
        ALLOCATION.  This is the one that says what the pool costs.
    ``dcp_ratio``
        this rank's entry of the ``Uneven-DCP token sizing`` ratio vector.
    ``card_free_after_pools_gib`` / ``alloc_at_rest_gib``
        the ``[vram-peak] high-water (1 rows)`` line: the at-rest state once
        every pool exists.
    ``card_free_worst_gib`` / ``card_total_gib``
        the minimum ``card free`` over all this rank's ``[vram-peak]`` lines.
    ``expert_tensor_gib``
        ``[vram-census] ... {experts %s, ...}`` of the MAIN model (never the
        ``-draft`` census -- those experts are not rows of this arena).
    ``residents`` / ``lru_rows`` / ``staging`` / ``spill_rows``
        ``MoE expert pool on layer 0: residents %d, LRU rows %d, staging %d,
        spill rows %d``.
    """

    rank: int
    dcp_ratio: int
    available_bytes: int
    cell_size: int
    page_size: int
    #: PHYSICAL rows this rank's main KV pool holds, from
    #: ``KV Cache is allocated ... #tokens: %d`` -- the ALLOCATION, never the
    #: ``KV pool sizing`` capacity or the ``local capacity`` reconstruction.
    #: Equals ``dcp_compact_pool_rows(C, S, ratio_r)`` (owner.py:155).
    kv_rows_now: int
    #: card free at rest, once every pool exists: the ``[vram-peak]
    #: high-water (1 rows)`` line. This is the budget rows and the transient
    #: come out of.
    card_free_after_pools_gib: float
    #: torch allocated at that same at-rest moment.
    alloc_at_rest_gib: float
    #: the WORST card free this rank was ever observed at (minimum over its
    #: ``[vram-peak]`` lines). Reported as the stricter cross-check.
    card_free_worst_gib: float
    card_total_gib: float
    expert_tensor_gib: float
    residents: int
    lru_rows: int
    staging: int
    spill_rows: int
    #: ``torch.cuda.max_memory_allocated()`` at the DEEPEST prefill chunk, GiB.
    #: ``None`` -> a strict plan REFUSES: the shallow high-water sample is a
    #: lower bound on the draw and fn8ak3 died on reading one as a maximum.
    prefill_peak_allocated_gib: Optional[float] = None
    #: rows the boot that measured that peak carried BEYOND this census's C.
    #: Subtracted so the transient is the transient and not somebody's arena.
    peak_boot_extra_rows: int = 0
    #: CUDA-graph private pools, GiB -- chunk-INDEPENDENT (fn8ak3: 1.79).
    graph_private_gib: float = 0.0
    #: the prefill chunk ``prefill_peak_allocated_gib`` was measured at.
    prefill_chunk_tokens: int = 0

    @property
    def scratch(self) -> int:
        """C as the boot ran it: the device-LRU region plus the staging rows
        (``expert_offload.py:3866`` -- the last S of the C rows are staging)."""
        return int(self.lru_rows) + int(self.staging)

    @property
    def buffer_rows(self) -> int:
        """``buffer_size`` = R + C (``expert_offload.py:504``)."""
        return int(self.residents) + self.scratch

    @property
    def owned_experts(self) -> int:
        """E_local.  A spill row exists for every owned expert outside R, so
        ``E_local = R + spill`` -- the count ``install_pool`` checks against."""
        return int(self.residents) + int(self.spill_rows)

    @property
    def kv_bytes_now(self) -> int:
        """Bytes the pool ACTUALLY holds: compacted rows x cell."""
        return int(self.kv_rows_now) * int(self.cell_size)

    @property
    def transient_card_gib(self) -> float:
        """The stricter transient reading: how far the CARD's free column fell
        between the at-rest sample and the worst one.  Larger than the
        allocated-delta reading because the allocator reserves more than it
        holds live and returns none of it to the driver (fn8am rank 1: 5.37 ->
        0.08 GiB while allocated moved 11.85 -> 12.06)."""
        return max(
            0.0, float(self.card_free_after_pools_gib) - float(self.card_free_worst_gib)
        )

    def __post_init__(self):
        if self.cell_size <= 0:
            raise ValueError(
                f"rank{self.rank}: cell_size={self.cell_size}. A KV-less stage "
                "cannot be priced here -- pp_cut.py:2904 refuses the same shape, "
                "for the same reason (its capacity would be unbounded)."
            )
        if self.page_size <= 0:
            raise ValueError(f"rank{self.rank}: page_size={self.page_size} must be > 0")
        if self.dcp_ratio <= 0:
            raise ValueError(f"rank{self.rank}: dcp_ratio={self.dcp_ratio} must be > 0")
        if self.card_free_after_pools_gib < 0.0:
            raise ValueError(f"rank{self.rank}: card_free_after_pools_gib must be >= 0")
        if self.kv_rows_now <= 0:
            raise ValueError(
                f"rank{self.rank}: kv_rows_now={self.kv_rows_now}. This is the "
                "ALLOCATION ('KV Cache is allocated ... #tokens'), not the "
                "'KV pool sizing' capacity -- a census without it cannot say "
                "whether the pool has to grow or shrink."
            )
        if self.buffer_rows > self.owned_experts:
            raise ValueError(
                f"rank{self.rank}: census is self-inconsistent -- buffer R+C="
                f"{self.buffer_rows} exceeds owned experts {self.owned_experts}"
            )


@dataclass(frozen=True)
class RankPlan:
    """The per-rank answer.  Every field is a number a reader can check."""

    rank: int
    kv_tokens: int
    kv_bytes: int
    kv_credit_bytes: float
    card_free_bytes: float
    headroom_bytes: float
    row_bytes: float
    rows_affordable: int
    scratch: int
    scratch_delta: int
    buffer_rows: int
    owned_experts: int
    ownership_surplus_rows: int
    leftover_bytes: float
    transient_book_mib: int
    bound_by: str

    @property
    def kv_gib(self) -> float:
        return self.kv_bytes / GIB

    def line(self) -> str:
        return (
            f"rank{self.rank}: KV {self.kv_tokens:>7d} tok = {self.kv_gib:5.3f} GiB "
            f"(credit {self.kv_credit_bytes / GIB:+5.3f}) | card free "
            f"{self.card_free_bytes / GIB:4.2f} | headroom "
            f"{self.headroom_bytes / GIB:5.3f} GiB | row {self.row_bytes / MIB:6.2f} "
            f"MiB | rows +{self.rows_affordable:<3d} -> C {self.scratch:<4d} "
            f"(was {self.scratch - self.scratch_delta}, buffer {self.buffer_rows}/"
            f"{self.owned_experts}) | leftover {self.leftover_bytes / MIB:7.1f} MiB "
            f"| bound by {self.bound_by}"
        )


@dataclass(frozen=True)
class Plan:
    ranks: Tuple[RankPlan, ...]
    ctx_tokens: int
    spec_tokens: int
    dcp_ratios: Tuple[int, ...]
    unit_tokens: int
    max_total_tokens: int
    notes: Tuple[str, ...] = field(default=())

    @property
    def kv_tokens_world(self) -> int:
        """The GLOBAL pool under uneven DCP -- the number to quote (user law
        "unter uneven DCP den GESAMTPOOL nennen").  Unit times the ratio sum,
        i.e. what the boot prints as ``EFFECTIVE max_total_num_tokens``."""
        return int(self.unit_tokens) * int(sum(self.dcp_ratios))

    def env(self) -> Dict[str, str]:
        """The vector that REPLACES the hand pins.

        * ``SGLANG_MOE_SCRATCH_SLOTS`` -- C per rank, budget-derived.
        * ``MAX_TOTAL_TOKENS`` -- ``--max-total-tokens``, which IS C, the
          GLOBAL context budget; the per-rank pool is compacted from it by the
          runtime.  Never a rank's share: fn8am set it to one rank's physical
          row count and refused a 259415-token needle at 90810.
        * ``SGLANG_KV_BUDGET_PREFILL_TRANSIENT_MIB`` -- the residual post that
          books, in the sizer's own ledger, the bytes the new rows and the
          prefill transient will take.  Without it the sizer re-derives
          ``available_bytes`` from a budget it thinks is free and hands them
          back to KV; with it the pool comes out at the need by construction
          and the cap above is a belt over braces.
        """
        return {
            "SGLANG_MOE_SCRATCH_SLOTS": ",".join(str(r.scratch) for r in self.ranks),
            "MAX_TOTAL_TOKENS": str(self.max_total_tokens),
            "SGLANG_KV_BUDGET_PREFILL_TRANSIENT_MIB": ",".join(
                str(r.transient_book_mib) for r in self.ranks
            ),
        }

    def report(self) -> str:
        head = (
            f"context {self.ctx_tokens} + spec {self.spec_tokens} over DCP vector "
            f"{list(self.dcp_ratios)} -> unit {self.unit_tokens}; WORLD pool "
            f"{self.kv_tokens_world} tokens (the per-rank figures below are SHARES "
            f"of it, never the pool; --max-total-tokens is the WORLD number -- "
            f"giving it a rank's share refuses long requests, fn8am)"
        )
        body = "\n".join(r.line() for r in self.ranks)
        env = "\n".join(f"  {k}={v}" for k, v in self.env().items())
        tail = "\n".join(f"  NOTE {n}" for n in self.notes)
        out = f"{head}\n{body}\n\nenv:\n{env}"
        return f"{out}\n\n{tail}" if tail else out


# ---------------------------------------------------------------------------
# the terms
# ---------------------------------------------------------------------------


def kv_rows_for_rank(
    *,
    ctx_tokens: int,
    spec_tokens: int,
    dcp_ratios: Sequence[int],
    rank: int,
) -> Tuple[int, int]:
    """PHYSICAL rows rank ``rank`` must hold for a world context.  ``(C, rows)``.

    This is the runtime's own compaction rule, not a model of it
    (``layers/dcp/owner.py:155``)::

        C    = ctx + spec                      # the GLOBAL context budget
        rows = (C // S + 1) * ratio_r          # S = sum(ratios)

    The ``+1`` is a CEIL to a whole owner block and is not cosmetic: allocator
    slot ids reach ``C`` itself, and a slot in a trailing partial block
    compacts one block past a floored sizing.  Flooring it cost an async
    illegal memory access once already; this module will not re-derive it.

    Verified against two boots: ``(262151 // 32 + 1) * 11 = 90123`` and
    ``(90816 // 32 + 1) * 11 = 31229``, both exactly the ``KV Cache is
    allocated ... #tokens`` line.

    The speculative rows are a WORLD term: draft and verify tokens land in the
    same sharded pool, so charging them per rank charges them ``len(ratios)``
    times.
    """
    ratios = [int(x) for x in dcp_ratios]
    if not ratios or any(x <= 0 for x in ratios):
        raise ValueError(f"dcp_ratios must be positive, got {list(dcp_ratios)}")
    if not 0 <= int(rank) < len(ratios):
        raise ValueError(f"rank {rank} outside DCP vector of length {len(ratios)}")
    world = int(ctx_tokens) + int(spec_tokens)
    if world <= 0:
        raise ValueError(f"ctx_tokens + spec_tokens must be > 0, got {world}")
    S = sum(ratios)
    return world, (world // S + 1) * ratios[int(rank)]


def row_bytes_from_census(expert_tensor_gib: float, buffer_rows: int) -> float:
    """Bytes ONE pool row costs, across all MoE layers -- MEASURED, not modelled.

    A row is one expert slot of the ``[R + C]`` arena and it exists on every MoE
    layer, so its cost is ``per_expert_per_layer * num_moe_layers``.  Both
    factors depend on the checkpoint's quantization and on the cut, and both are
    already multiplied together in the census the boot prints, so the census is
    read instead::

        row_bytes = experts_bytes / (R + C)

    fn8aj, three ranks with three different ``R``/``C``: 16.65 GiB/147,
    8.50 GiB/75, 8.95 GiB/79 -> 115.98, 116.05, 116.01 MiB, a 0.06 % spread.
    The nominal 2.45 MiB per expert per layer over that boot's 48 MoE layers is
    117.6 MiB -- within 1.4 %, which is why the nominal is a cross-check and
    not the source.
    """
    rows = int(buffer_rows)
    if rows <= 0:
        raise ValueError(f"buffer_rows must be > 0, got {buffer_rows}")
    if float(expert_tensor_gib) <= 0.0:
        raise ValueError(
            f"expert_tensor_gib must be > 0, got {expert_tensor_gib}; without a "
            "census the row cost is unmeasured and this term refuses to guess"
        )
    return float(expert_tensor_gib) * GIB / rows


def rank_headroom_bytes(
    c: RankCensus,
    *,
    kv_credit_bytes: float,
    row_bytes: float,
    prefill_chunk_tokens: Optional[int] = None,
    floor_bytes: int = DEFAULT_FLOOR_BYTES,
    strict: bool = True,
) -> Tuple[float, float, Optional[str]]:
    """Bytes rank ``c`` has free for new pool rows.  ``(headroom, transient, note)``.

    ``headroom = card_free_after_pools + kv_credit - transient(chunk) - floor``

    The transient is ``peak_alloc_deep - alloc_at_rest``, minus the rows the
    measuring boot carried beyond this census (otherwise somebody else's arena
    is charged as a transient).  Only the non-graph part scales with the chunk:
    the CUDA-graph private pools are captured once.

    The cross-check :attr:`RankCensus.transient_card_gib` is computed too and a
    NOTE fires when it is larger -- it is the stricter bound and the one the
    fn8ak3 OOM hit, because the allocator reserves more than it holds live.
    """
    note: Optional[str] = None
    if c.prefill_peak_allocated_gib is None:
        if strict:
            raise ExpertPoolBudgetRefused(
                c.rank,
                0.0,
                [
                    (
                        "card free after pools",
                        float(c.card_free_after_pools_gib) * GIB,
                    ),
                    ("transient at the DEEPEST chunk", float("nan")),
                ],
                0.0,
            )
        note = (
            f"rank{c.rank}: no measured DEEP prefill peak -- only the shallow "
            f"high-water sample ({c.transient_card_gib:.2f} GiB of card free lost) "
            "is known, and that is a LOWER bound on the draw. The row count is an "
            "UPPER BOUND, not a budget"
        )
        true_transient = c.transient_card_gib
    else:
        true_transient = max(
            0.0,
            float(c.prefill_peak_allocated_gib)
            - float(c.alloc_at_rest_gib)
            - float(c.peak_boot_extra_rows) * row_bytes / GIB,
        )
    target = true_transient
    if (
        prefill_chunk_tokens is not None
        and int(c.prefill_chunk_tokens) > 0
        and c.prefill_peak_allocated_gib is not None
    ):
        scalable = max(0.0, true_transient - float(c.graph_private_gib))
        target = float(c.graph_private_gib) + scalable * (
            float(prefill_chunk_tokens) / float(c.prefill_chunk_tokens)
        )
    headroom = (
        float(c.card_free_after_pools_gib) * GIB
        + float(kv_credit_bytes)
        - target * GIB
        - float(floor_bytes)
    )
    if c.transient_card_gib > target:
        extra = (
            f"rank{c.rank}: the CARD-side transient reading is "
            f"{c.transient_card_gib:.2f} GiB against the {target:.2f} GiB priced "
            f"here -- {(c.transient_card_gib - target) * GIB / MIB:.0f} MiB of "
            "allocator reservation that never returns to the driver. The stricter "
            "reading allows "
            f"{max(0, int((headroom - (c.transient_card_gib - target) * GIB) // row_bytes))} "
            "row(s)"
        )
        note = (note + "; " + extra) if note else extra
    return headroom, target, note


def plan_expert_pool(
    census: Sequence[RankCensus],
    *,
    ctx_tokens: int,
    spec_tokens: int = 0,
    floor_bytes: int = DEFAULT_FLOOR_BYTES,
    max_total_tokens: Optional[int] = None,
    prefill_chunk_tokens: Optional[int] = None,
    strict: bool = True,
) -> Plan:
    """The planner term.  See the module docstring for the chain it inverts.

    ``max_total_tokens`` is the SCALAR ``--max-total-tokens`` the boot carries.
    Left ``None`` it is derived as the largest per-rank need, because the flag
    is a single int (``server_args.py:1339``) applied to every rank and a cap
    below any rank's need would starve the world unit.  A rank whose own need
    is smaller therefore budgets for the CAP -- on fn8aj that over-charges
    rank 2 by 8192 tokens (110 MiB, one row).  Naming it is the point.
    """
    ranks = sorted(census, key=lambda c: c.rank)
    if not ranks:
        raise ValueError("census is empty")
    if [c.rank for c in ranks] != list(range(len(ranks))):
        raise ValueError(
            f"census must cover ranks 0..{len(ranks) - 1} exactly, got "
            f"{[c.rank for c in ranks]}"
        )
    ratios = tuple(c.dcp_ratio for c in ranks)
    page = ranks[0].page_size
    if any(c.page_size != page for c in ranks):
        raise ValueError(
            "RAENGE NIE UNEINS: ranks disagree on page_size "
            f"{[c.page_size for c in ranks]} -- refusing to pick one"
        )

    world = 0
    needs: List[int] = []
    for c in ranks:
        world, rows = kv_rows_for_rank(
            ctx_tokens=ctx_tokens,
            spec_tokens=spec_tokens,
            dcp_ratios=ratios,
            rank=c.rank,
        )
        needs.append(rows)
    unit = world // sum(ratios)
    cap = int(max_total_tokens) if max_total_tokens is not None else world
    notes: List[str] = []
    if cap < world:
        notes.append(
            f"--max-total-tokens {cap} is below the world context {world}: the flag "
            "IS C, so requests longer than that are refused however big the pools "
            "are (fn8am refused a 259415-token needle at 90810)"
        )

    plans: List[RankPlan] = []
    for c, need in zip(ranks, needs):
        row_bytes = row_bytes_from_census(c.expert_tensor_gib, c.buffer_rows)
        kv_bytes = int(need) * int(c.cell_size)
        # CREDIT, and it is usually ZERO: the pool is already compacted to
        # C x ratio_r / S. It is NEGATIVE when C has to grow to serve the
        # context -- then restoring the pool costs rows, it does not fund them.
        credit = float(c.kv_bytes_now - kv_bytes)
        headroom, transient_gib, note = rank_headroom_bytes(
            c,
            kv_credit_bytes=credit,
            row_bytes=row_bytes,
            prefill_chunk_tokens=prefill_chunk_tokens,
            floor_bytes=floor_bytes,
            strict=strict,
        )
        if note:
            notes.append(note)
        if headroom < 0.0:
            raise ExpertPoolBudgetRefused(
                c.rank,
                headroom,
                [
                    (
                        "card free after pools",
                        float(c.card_free_after_pools_gib) * GIB,
                    ),
                    ("KV pool credit (allocated now - DCP need)", credit),
                    ("prefill transient at this chunk", -transient_gib * GIB),
                    ("floor", -float(floor_bytes)),
                ],
                -headroom,
            )
        rows = int(math.floor(headroom / row_bytes))
        ownable = c.owned_experts - c.residents - c.scratch
        if rows <= ownable:
            bound_by = "budget"
            surplus = 0
            taken = rows
        else:
            bound_by = "ownership (buffer_size == R+C <= E_local)"
            surplus = rows - ownable
            taken = ownable
        scratch = c.scratch + taken
        book = (float(c.available_bytes) - kv_bytes - taken * row_bytes) / MIB
        plans.append(
            RankPlan(
                rank=c.rank,
                kv_tokens=int(need),
                kv_bytes=kv_bytes,
                kv_credit_bytes=credit,
                card_free_bytes=float(c.card_free_after_pools_gib) * GIB,
                headroom_bytes=headroom,
                row_bytes=row_bytes,
                rows_affordable=taken,
                scratch=scratch,
                scratch_delta=taken,
                buffer_rows=c.residents + scratch,
                owned_experts=c.owned_experts,
                ownership_surplus_rows=surplus,
                leftover_bytes=headroom - taken * row_bytes,
                transient_book_mib=int(max(0.0, math.floor(book))),
                bound_by=bound_by,
            )
        )
        if surplus:
            notes.append(
                f"rank{c.rank} can afford {surplus} more row(s) than it owns "
                f"experts for (E_local={c.owned_experts}, R={c.residents}): give it "
                f"more experts via --rank-moe-ratio instead of more C -- "
                f"C={c.scratch + rows} would die with 'pool mode requires "
                f"buffer_size == R+C ({c.owned_experts} != {c.residents}+"
                f"{c.scratch + rows})'"
            )

    return Plan(
        ranks=tuple(plans),
        ctx_tokens=int(ctx_tokens),
        spec_tokens=int(spec_tokens),
        dcp_ratios=ratios,
        unit_tokens=int(unit),
        max_total_tokens=int(cap),
        notes=tuple(notes),
    )


def survey_ranks(
    census: Sequence[RankCensus],
    *,
    ctx_tokens: int,
    spec_tokens: int = 0,
    floor_bytes: int = DEFAULT_FLOOR_BYTES,
    prefill_chunk_tokens: Optional[int] = None,
    strict: bool = True,
) -> Tuple[str, ...]:
    """One line per rank, refusals INCLUDED instead of raising on the first.

    :func:`plan_expert_pool` refuses by name and stops, which is right for a
    caller that wants a vector.  A reader wants to see every rank -- especially
    when one refuses, because "rank 0 is short by 0.6 GiB" says nothing about
    whether the 3080s have room.  Same arithmetic, no exception.
    """
    ranks = sorted(census, key=lambda c: c.rank)
    ratios = tuple(c.dcp_ratio for c in ranks)
    out: List[str] = []
    for c in ranks:
        _, need = kv_rows_for_rank(
            ctx_tokens=ctx_tokens,
            spec_tokens=spec_tokens,
            dcp_ratios=ratios,
            rank=c.rank,
        )
        row_bytes = row_bytes_from_census(c.expert_tensor_gib, c.buffer_rows)
        credit = float(c.kv_bytes_now - need * int(c.cell_size))
        try:
            headroom, transient, note = rank_headroom_bytes(
                c,
                kv_credit_bytes=credit,
                row_bytes=row_bytes,
                prefill_chunk_tokens=prefill_chunk_tokens,
                floor_bytes=floor_bytes,
                strict=strict,
            )
        except ExpertPoolBudgetRefused as exc:
            out.append(f"rank{c.rank}: REFUSED -- {exc}")
            continue
        ownable = c.owned_experts - c.residents - c.scratch
        rows = int(math.floor(headroom / row_bytes)) if headroom >= 0 else 0
        taken = min(rows, ownable)
        verdict = (
            f"+{taken} row(s) -> C {c.scratch + taken}"
            if headroom >= 0
            else f"SHORT by {-headroom / MIB:.0f} MiB -- no rows, and "
            f"{int(-headroom // row_bytes) + 1} of the current {c.scratch} would "
            "have to go back"
        )
        out.append(
            f"rank{c.rank}: pool {c.kv_rows_now} -> {need} rows "
            f"(credit {credit / GIB:+.3f} GiB) | free at rest "
            f"{c.card_free_after_pools_gib:.2f} | transient {transient:.2f} "
            f"(card reading {c.transient_card_gib:.2f}) | row "
            f"{row_bytes / MIB:.1f} MiB | headroom {headroom / GIB:+.3f} GiB | "
            f"{verdict}"
        )
        if note:
            out.append(f"    NOTE {note}")
    return tuple(out)


def verify_scratch_vector(
    census: Sequence[RankCensus],
    scratch: Sequence[int],
    *,
    ctx_tokens: int,
    spec_tokens: int = 0,
    floor_bytes: int = DEFAULT_FLOOR_BYTES,
    max_total_tokens: Optional[int] = None,
    prefill_chunk_tokens: Optional[int] = None,
    strict: bool = True,
) -> Tuple[str, ...]:
    """Check a PROPOSED ``SGLANG_MOE_SCRATCH_SLOTS`` vector against the census.

    Returns one verdict string per rank that fails, empty when the vector fits.
    This is the retrodiction door: fed fn8aj's census and fn8ak's ``175,60,60``
    it names the ownership contradiction rank 2 died on, and fed fn8ak2's
    ``175,60,54`` it names the rows rank 0 was over budget by.
    """
    ranks = sorted(census, key=lambda x: x.rank)
    proposed = [int(x) for x in scratch]
    if len(proposed) != len(ranks):
        raise ValueError(
            f"scratch vector has {len(proposed)} entries for {len(ranks)} ranks"
        )
    # The pool identity needs NO budget, so it is checked first and separately:
    # a rank the budget cannot price still has a verdict on whether its C is
    # even representable (fn8ak died on exactly this, before any memory ran
    # out).
    identity: List[str] = []
    for c, want in zip(ranks, proposed):
        if c.residents + want > c.owned_experts:
            identity.append(
                f"rank{c.rank}: C={want} breaks the pool identity -- "
                f"'pool mode requires buffer_size == R+C "
                f"({c.owned_experts} != {c.residents}+{want})'"
            )
    try:
        plan = plan_expert_pool(
            census,
            ctx_tokens=ctx_tokens,
            spec_tokens=spec_tokens,
            floor_bytes=floor_bytes,
            max_total_tokens=max_total_tokens,
            prefill_chunk_tokens=prefill_chunk_tokens,
            strict=strict,
        )
    except ExpertPoolBudgetRefused as exc:
        return tuple(identity) + (
            f"rank{exc.rank}: the budget cannot be priced, so the remaining "
            f"ranks are unchecked -- {exc}",
        )
    out: List[str] = list(identity)
    for c, p, want in zip(ranks, plan.ranks, proposed):
        if c.residents + want > c.owned_experts:
            continue
        over = (want - c.scratch) - p.rows_affordable
        if over > 0:
            out.append(
                f"rank{c.rank}: C={want} is {over} row(s) = "
                f"{over * p.row_bytes / MIB:.0f} MiB beyond the "
                f"{p.headroom_bytes / GIB:.3f} GiB this rank has free; the "
                f"budget-derived value is {p.scratch}"
            )
    return tuple(out)


# ---------------------------------------------------------------------------
# dry run: read a boot log's own instruments
# ---------------------------------------------------------------------------

_RE_TP = re.compile(r"\bTP(\d+)\]")
_RE_SIZING = re.compile(
    r"KV pool sizing: available_bytes=(\d+) .*?cell_size=(\d+), page_size=(\d+)"
)
_RE_PEAK = re.compile(
    r"allocator peak since pools ([\d.]+) GiB, allocated now ([\d.]+), "
    r"reserved ([\d.]+), card free ([\d.]+) of ([\d.]+) GiB"
)
_RE_CENSUS = re.compile(
    r"\[vram-census\] pp\d+tp(\d+) after (?:load|pools): .*?\{experts ([\d.]+)"
)
_RE_POOL = re.compile(
    r"MoE expert pool on layer 0: residents (\d+), LRU rows (\d+), "
    r"staging (\d+), spill rows (\d+)"
)
_RE_DCP = re.compile(r"Uneven-DCP token sizing: rank \d+ .*?vector \[([\d, ]+)\]")
#: The ALLOCATION, not the capacity. A boot allocates several pools (draft/MTP,
#: QSA index, the main one); the MAIN one is the one with the largest K size,
#: never the one with the largest token count -- on fn8am the 90816-token pool
#: is the 1-layer draft pool at 0.02 GB while the main pool is 31229 tokens at
#: 0.18 GB.
_RE_ALLOC = re.compile(r"KV Cache is allocated\..*?#tokens: (\d+), K size: ([\d.]+) GB")

_REQUIRED = (
    "available_bytes",
    "cell_size",
    "page_size",
    "kv_rows_now",
    "card_free_after_pools_gib",
    "alloc_at_rest_gib",
    "card_free_worst_gib",
    "card_total_gib",
    "expert_tensor_gib",
    "residents",
    "lru_rows",
    "staging",
    "spill_rows",
)


def parse_boot_log(text: str) -> Tuple[RankCensus, ...]:
    """Build the per-rank census from a boot log's own lines.

    Reads ONLY instruments the boot already emits; nothing is derived here that
    the log does not state.  Two choices are load-bearing and both are the
    conservative one:

    * the AT-REST sample is the ``[vram-peak] high-water (1 rows)`` line --
      every pool exists and no traffic has drawn anything yet -- while
      ``card_free_worst`` is the MINIMUM over every ``[vram-peak]`` line, the
      worst load state the deployment actually served;
    * the pool row count comes from the ``KV Cache is allocated`` line with the
      LARGEST K size, so a small draft pool with a big token count cannot be
      mistaken for the main one;
    * ``-draft`` census lines are skipped, because the draft model's experts
      are not rows of this arena.
    """
    fields: Dict[int, Dict[str, float]] = {}

    def slot(rank: int) -> Dict[str, float]:
        return fields.setdefault(int(rank), {})

    ratios: Tuple[int, ...] = ()
    for line in text.splitlines():
        tp = _RE_TP.search(line)
        rank = int(tp.group(1)) if tp else None

        m = _RE_SIZING.search(line)
        if m and rank is not None:
            s = slot(rank)
            s["available_bytes"] = float(m.group(1))
            s["cell_size"] = float(m.group(2))
            s["page_size"] = float(m.group(3))
            continue
        m = _RE_PEAK.search(line)
        if m and rank is not None:
            s = slot(rank)
            free = float(m.group(4))
            s["card_total_gib"] = float(m.group(5))
            if free < s.get("card_free_worst_gib", float("inf")):
                s["card_free_worst_gib"] = free
            # the AT-REST sample is the '(1 rows)' one: every pool exists and
            # no traffic has drawn anything yet.
            if " (1 rows)" in line:
                s["card_free_after_pools_gib"] = free
                s["alloc_at_rest_gib"] = float(m.group(2))
            continue
        m = _RE_CENSUS.search(line)
        if m and "-draft" not in line:
            slot(int(m.group(1)))["expert_tensor_gib"] = float(m.group(2))
            continue
        m = _RE_POOL.search(line)
        if m and rank is not None:
            s = slot(rank)
            s["residents"] = float(m.group(1))
            s["lru_rows"] = float(m.group(2))
            s["staging"] = float(m.group(3))
            s["spill_rows"] = float(m.group(4))
            continue
        m = _RE_ALLOC.search(line)
        if m and rank is not None:
            s = slot(rank)
            k_gb = float(m.group(2))
            if k_gb > s.get("_k_gb", -1.0):
                s["_k_gb"] = k_gb
                s["kv_rows_now"] = float(m.group(1))
            continue
        m = _RE_DCP.search(line)
        if m and not ratios:
            ratios = tuple(int(x) for x in m.group(1).split(","))

    if not ratios:
        raise ValueError(
            "no 'Uneven-DCP token sizing' line in this log: the DCP ratio vector "
            "is the unit of the KV term and cannot be assumed"
        )
    out: List[RankCensus] = []
    for rank in sorted(fields):
        s = fields[rank]
        missing = [k for k in _REQUIRED if k not in s]
        if missing:
            raise ValueError(
                f"rank{rank}: the log is missing {missing}. A partial census is not "
                "a census -- boot with the instruments on rather than defaulting "
                "the gaps to zero"
            )
        out.append(
            RankCensus(
                rank=rank,
                dcp_ratio=ratios[rank],
                available_bytes=int(s["available_bytes"]),
                cell_size=int(s["cell_size"]),
                page_size=int(s["page_size"]),
                kv_rows_now=int(s["kv_rows_now"]),
                card_free_after_pools_gib=s["card_free_after_pools_gib"],
                alloc_at_rest_gib=s["alloc_at_rest_gib"],
                card_free_worst_gib=s["card_free_worst_gib"],
                card_total_gib=s["card_total_gib"],
                expert_tensor_gib=s["expert_tensor_gib"],
                residents=int(s["residents"]),
                lru_rows=int(s["lru_rows"]),
                staging=int(s["staging"]),
                spill_rows=int(s["spill_rows"]),
            )
        )
    return tuple(out)


#: The prefill chunk sizes the dry run prices side by side.  Halving CHUNK
#: halves the chunk-scaling half of the transient, and on fn8ak3's rank 2 that
#: is the difference between a refusal and ~12 rows -- at a TTFT cost this
#: module does not price.
DRY_RUN_CHUNKS = (8192, 4096)


def dry_run(
    text: str,
    *,
    ctx_tokens: int = 262144,
    spec_tokens: int = 0,
    floor_bytes: int = DEFAULT_FLOOR_BYTES,
    chunks: Sequence[int] = DRY_RUN_CHUNKS,
    strict: bool = True,
) -> str:
    """Parse a boot log and print the numbers the NEXT boot should carry, once
    per prefill chunk size in ``chunks``.

    Input is exactly the ``[vram-census]``, ``[vram-peak]``, ``KV pool
    sizing``, ``Uneven-DCP token sizing`` and ``MoE expert pool`` lines of a
    log -- the whole file works, so does a grep of those five shapes.

    A section is printed for every chunk, including the ones that REFUSE: a
    chunk size the rig cannot afford is an answer, and hiding it would leave
    the reader with only the affordable half of the trade.
    """
    census = parse_boot_log(text)
    out: List[str] = []
    for chunk in chunks:
        head = f"=== prefill chunk {chunk} " + "=" * 40
        try:
            plan = plan_expert_pool(
                census,
                ctx_tokens=ctx_tokens,
                spec_tokens=spec_tokens,
                floor_bytes=floor_bytes,
                prefill_chunk_tokens=chunk,
                strict=strict,
            )
            out.append(f"{head}\n{plan.report()}")
        except ExpertPoolBudgetRefused as exc:
            survey = "\n".join(
                survey_ranks(
                    census,
                    ctx_tokens=ctx_tokens,
                    spec_tokens=spec_tokens,
                    floor_bytes=floor_bytes,
                    prefill_chunk_tokens=chunk,
                    strict=strict,
                )
            )
            out.append(f"{head}\nREFUSED: {exc}\n{survey}")
    return "\n\n".join(out)


def _main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    import argparse
    import sys

    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("log", help="a boot server log, or '-' for stdin")
    ap.add_argument("--ctx-tokens", type=int, default=262144)
    ap.add_argument("--spec-tokens", type=int, default=0)
    ap.add_argument(
        "--verify",
        default=None,
        help="check a proposed SGLANG_MOE_SCRATCH_SLOTS vector instead of deriving one",
    )
    ap.add_argument(
        "--chunks",
        default=",".join(str(c) for c in DRY_RUN_CHUNKS),
        help="prefill chunk sizes to price side by side",
    )
    ap.add_argument(
        "--no-strict",
        action="store_true",
        help=(
            "price on the LATCHED [vram-peak] sample when the log has no measured "
            "prefill peak. The result is an upper bound, not a budget -- three "
            "boots died on exactly that reading"
        ),
    )
    args = ap.parse_args(argv)
    text = (
        sys.stdin.read()
        if args.log == "-"
        else open(args.log, "r", errors="replace").read()
    )
    try:
        if args.verify:
            bad = verify_scratch_vector(
                parse_boot_log(text),
                [int(x) for x in args.verify.split(",")],
                ctx_tokens=args.ctx_tokens,
                spec_tokens=args.spec_tokens,
                prefill_chunk_tokens=int(args.chunks.split(",")[0]),
                strict=not args.no_strict,
            )
            for line in bad:
                print(f"REFUSED {line}")
            print("VECTOR OK" if not bad else f"{len(bad)} rank(s) refused")
            return 1 if bad else 0
        print(
            dry_run(
                text,
                ctx_tokens=args.ctx_tokens,
                spec_tokens=args.spec_tokens,
                chunks=[int(c) for c in args.chunks.split(",")],
                strict=not args.no_strict,
            )
        )
    except (ExpertPoolBudgetRefused, ValueError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
