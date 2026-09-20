# SPDX-License-Identifier: Apache-2.0
"""Tasks #14/#48 -- the expert pool's LRU rows come from the BUDGET, not the hand.

WHY THIS EXISTS
---------------
Boot fn8aj (2026-09-20) sized the KV pool per rank out of what was left after
the weights::

    KV pool sizing: available_bytes=5074821120 (4.726 GiB), cell_size=14143,
    page_size=64 -> max_total_num_tokens=358784        (rank 0, one line per rank)

and then capped it by hand with ``--max-total-tokens 270000``.  But the DCP
token vector was ``[11, 11, 10]``, so what a rank has to hold for a
262144-token context is ``ceil(262144/32) * ratio_r`` = 90112 / 90112 / 81920
tokens -- 1.19 GiB at ``cell_size`` 14143, not the 3.56 GiB each rank got.

The expert residency on the other side was a pure HAND PIN:
``--rank-moe-resident-fraction 0.004,0.37,0.44`` froze R from the hot-set file
and ``SGLANG_MOE_SCRATCH_SLOTS=145,36,36`` set C by taste.  Two configuration
boots then failed, and BOTH failures are this module's acceptance tests:

* **fn8ak** (``175,60,60``) died before ``/health`` with
  ``RuntimeError: pool mode requires buffer_size == R+C (97 != 43+60)``.
  Rank 2 OWNS 97 experts and ``buffer_size`` is ``min(R + C, E_local)``
  (``expert_offload.py:504``, checked at ``:3895-3898``): rows above the owned
  count are not a bigger pool, they are a contradiction.
* **fn8ak2** (``175,60,54``) booted -- KV pools 1.215/2.031/2.293 GiB -- and
  then rank 0 (the 5090) went OOM inside the FIRST 8192-token prefill chunk:
  ``MoE offload gather fetch failed (OutOfMemoryError ... 254 MiB free ...
  30.88 GiB in use)``.  The 30 extra rows on that rank had eaten the memory
  the prefill needed.

WHICH TRANSIENT INSTRUMENT (the correction that decides this term)
------------------------------------------------------------------
The ``[vram-peak]`` line offers three readings of "how much does this rank
need beyond its steady state", and they disagree by 4x.  fn8aj rank 0::

    [vram-peak] extend (8192 rows): allocator peak since pools 24.77 GiB,
    allocated now 23.09, reserved 29.82, card free 0.55 of 31.34 GiB
    -> transient headroom used = peak - allocated 1.69 GiB

* ``peak - allocated`` = 1.69 GiB (2.47 at decode).  This is what the line
  itself names, and it **UNDERBOOKS**: it counts only the allocator's live
  high-water, not the cached-but-unreturned blocks the next allocation has to
  find contiguously.  fn8ak2 was planned on it and died.
* ``reserved - allocated`` = 6.73 GiB.  The allocator's cache high-water.  It
  **OVERBOOKS**: on fn8aj it exceeds rank 0's whole reachable budget, so a
  term using it refuses a configuration that demonstrably ran -- and it also
  forbids the +24/+18 rows the 3080s demonstrably took in fn8ak3.
* ``card free`` = 0.55 GiB.  **This is the base this module uses.**  It is the
  residual after the transient, the allocator cache, the fragmentation, the
  CUDA context and any foreign process -- i.e. it already contains every term
  the other two readings argue about, measured rather than modelled.

AND THE SAMPLE ITSELF IS NOT A MAXIMUM (fn8ak3, the third death)
-----------------------------------------------------------------
``maybe_log_vram_peak`` LATCHED: ``st["extend"]`` was set by the first extend
of >= 2048 rows, which under chunked prefill is the first chunk of the first
request.  The transient a deep prefill draws is not constant across chunks --
the attention workspace and the partials grow with the cached context a chunk
attends over -- so the worst chunk of a 259k needle comes minutes after the
latch and was never sampled.  Measured: fn8aj printed ``allocated now 13.29``
for rank 2; fn8ak3 (145,60,54 -- that rank +18 rows) died in the needle prefill
with ``18.60 GiB allocated by PyTorch, 1.79 GiB in private pools (CUDA Graphs),
73.5 MiB free``, and ``expandable_segments:True`` was already set, so it is not
fragmentation: at the real peak the 3080 is simply full.  The ~3.5 GiB that
look idle on a 3080 ARE the prefill transient at CHUNK 8192; they are free only
in decode.

``vram_family_census.py`` now re-emits on every new allocator high-water, so
the next boot MEASURES that number.  Until a log carries it, this module
REFUSES to size a pool (``strict=True``) rather than read the latched sample as
a maximum -- ``strict=False`` gives the upper bound, labelled as one.

So the headroom a rank has for more rows is::

    transient_r(chunk) = graph_private_r + (peak_alloc_r - alloc_r - graph_private_r)
                         * chunk / chunk_measured
    free_r(chunk)      = card_free_r - (peak_alloc_r - peak_r) + transient_r(measured)
                         - transient_r(chunk)
    headroom_r         = free_r(chunk) + (kv_bytes_now_r - kv_bytes_need_r)
    rows_r             = floor(headroom_r / row_bytes_r)
    C_r                = min(C_now_r + rows_r, E_local_r - R_r)

The KV term is a CREDIT, because shrinking the pool to its DCP need gives those
bytes back to the card before anything else runs.  The CUDA-graph private pools
are chunk-INDEPENDENT, so halving CHUNK halves only the other half of the
transient -- and that is a real lever: on fn8ak3's rank 2 it is the difference
between a refusal at 8192 and ~12 rows at 4096, at a TTFT cost this module does
not price.  :func:`dry_run` therefore prints both.

RETRODICTIONS, all from fn8aj's census, before anything new is booted:
fn8ak is refused by the pool identity on rank 2; fn8ak2 is refused by 5 rows on
rank 0 (0.57 GiB -- the OOM it hit reported 254 MiB free); and fn8ak3, which
the LATCHED reading accepts, is refused once rank 2's measured peak is in the
census.  The last one is the standing argument for ``strict=True``.

THE SIZING CHAIN THIS INVERTS (file:line @ 546dd2bd36)
------------------------------------------------------
1. ``model_runner_kv_cache_mixin.py:864-870`` -- the rank BUDGET is the
   absolute ``--rank-gpu-memory-mib``; the first post is the MEASURED
   ``weights + runtime state`` delta.  The expert arena (``R + C`` rows over
   all MoE layers) is inside that post, which is why a row is paid for in KV
   tokens and in nothing else.
2. ``model_runner_kv_cache_mixin.py:872-874`` -- corridor holdback (0 here).
3. ``model_runner_kv_cache_mixin.py:875-892`` -- the #48 C prefill transient,
   ``SGLANG_KV_BUDGET_PREFILL_TRANSIENT_MIB``, default 0 = nothing booked.
   In fn8aj it was UNSET, so the transient was funded by whatever the sizer
   happened to leave -- which is the defect, not a policy.
4. ``model_runner_kv_cache_mixin.py:905+`` -- mamba / speculative posts, then
   ``rest``, then a per-rank reserve, then ``available_bytes``.
5. ``pool_configurator.py:549-586`` -- ``available_bytes // cell_size``, then
   ``// page_size * page_size``, then the ``KV pool sizing`` line.
6. ``model_runner_kv_cache_mixin.py:5266-5305`` -- ``--max-total-tokens`` and
   the hybrid cap bind; the per-rank unit is ``local // ratio_r``, the world
   takes the MIN unit, ``EFFECTIVE`` is that unit times the ratio sum.
7. ``expert_offload.py:700-715`` / ``:3866`` / ``:3895`` --
   ``SGLANG_MOE_SCRATCH_SLOTS`` (scalar or per-rank vector) is C; the arena is
   ``[0,R)`` residents, ``[R, R+C-S)`` device LRU, last S staging.

USER LAWS THIS OBEYS
--------------------
* Reserves NEVER ("nicht ein Byte"): :data:`DEFAULT_FLOOR_BYTES` is 0 and the
  only non-zero floor is one a caller passes in with a named reason.
* Transients are BOOKED EXPLICITLY: :meth:`Plan.env` emits
  ``SGLANG_KV_BUDGET_PREFILL_TRANSIENT_MIB`` as the residual that keeps the
  sizer's hands off the bytes the rows and the prefill need -- the sizer then
  produces exactly the DCP need instead of "whatever is left".
* "KEIN BINDENDER RANG": ranks are priced independently; no rank's row count
  is copied to another.
* Under uneven DCP the pool is quoted as the WORLD total
  (:attr:`Plan.kv_tokens_world`), never as one rank's share.
* Refusal by name: an infeasible rank raises :class:`ExpertPoolBudgetRefused`
  naming the rank, every post and the shortfall.  Nothing is clamped silently.
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
    "kv_tokens_for_rank",
    "row_bytes_from_census",
    "plan_expert_pool",
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
        (``pool_configurator.py:577``).
    ``kv_tokens_now``
        ``Uneven-DCP token sizing: rank %d local capacity %d tokens``
        (``model_runner_kv_cache_mixin.py:5291``) -- the pool this rank
        actually built, AFTER every cap.
    ``dcp_ratio``
        this rank's entry of that line's ratio vector.
    ``card_free_gib`` / ``card_total_gib``
        ``[vram-peak] ... card free %s of %s GiB``, taken at the rank's WORST
        observed load state (minimum free).  This is the headroom instrument.
    ``reserved_gib`` / ``allocated_gib`` / ``peak_gib``
        the other three columns of that same worst line, kept only so the
        report can show why the two cheaper readings of "the transient"
        (``peak - allocated``, ``reserved - allocated``) are not used.
    ``expert_tensor_gib``
        ``[vram-census] ... model tensors on device ... {experts %s, ...}`` of
        the MAIN model (never the ``-draft`` census -- the draft's experts are
        not rows of this arena).
    ``residents`` / ``lru_rows`` / ``staging`` / ``spill_rows``
        ``MoE expert pool on layer 0: residents %d, LRU rows %d, staging %d,
        spill rows %d``.
    """

    rank: int
    dcp_ratio: int
    available_bytes: int
    cell_size: int
    page_size: int
    kv_tokens_now: int
    card_free_gib: float
    card_total_gib: float
    reserved_gib: float
    allocated_gib: float
    peak_gib: float
    expert_tensor_gib: float
    residents: int
    lru_rows: int
    staging: int
    spill_rows: int
    #: 20.09. (fn8ak3): the TRUE ``torch.cuda.max_memory_allocated()`` at the
    #: DEEPEST prefill chunk, GiB.  ``peak_gib`` above is NOT that number --
    #: ``maybe_log_vram_peak`` latched on the first extend of >= 2048 rows, so
    #: on a 259k needle it reported the first chunk of the first request.
    #: Measured gap on fn8aj/fn8ak3 rank 2: 14.98 reported against 18.60 real.
    #: ``None`` means the log could not measure it, and a strict plan REFUSES
    #: rather than treating the latched sample as a maximum.
    prefill_peak_allocated_gib: Optional[float] = None
    #: CUDA-graph private pools, GiB.  Chunk-INDEPENDENT, so it does not scale
    #: when the prefill chunk is halved (fn8ak3 rank 2: 1.79 GiB of the 19.36
    #: in use).
    graph_private_gib: float = 0.0
    #: the prefill chunk size ``prefill_peak_allocated_gib`` was measured at.
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
        return int(self.kv_tokens_now) * int(self.cell_size)

    @property
    def allocator_cache_gib(self) -> float:
        """``reserved - allocated``: cached blocks the allocator holds and does
        not return to the driver.  Reported, never subtracted -- it is already
        inside ``card_free``."""
        return float(self.reserved_gib) - float(self.allocated_gib)

    @property
    def peak_minus_allocated_gib(self) -> float:
        """What the ``[vram-peak]`` line calls the transient.  Reported as the
        instrument that UNDERBOOKS (fn8ak2 was planned on it)."""
        return float(self.peak_gib) - float(self.allocated_gib)

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
        if self.card_free_gib < 0.0:
            raise ValueError(f"rank{self.rank}: card_free_gib must be >= 0")
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
        * ``MAX_TOTAL_TOKENS`` -- the scalar ``--max-total-tokens`` that stops
          the pool at the DCP need instead of at the rest.
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
            f"of it, never the pool)"
        )
        body = "\n".join(r.line() for r in self.ranks)
        env = "\n".join(f"  {k}={v}" for k, v in self.env().items())
        tail = "\n".join(f"  NOTE {n}" for n in self.notes)
        out = f"{head}\n{body}\n\nenv:\n{env}"
        return f"{out}\n\n{tail}" if tail else out


# ---------------------------------------------------------------------------
# the terms
# ---------------------------------------------------------------------------


def kv_tokens_for_rank(
    *,
    ctx_tokens: int,
    spec_tokens: int,
    dcp_ratios: Sequence[int],
    rank: int,
    page_size: int,
) -> Tuple[int, int]:
    """KV tokens rank ``rank`` must hold -- FROM THE DCP UNIT, not from the rest.

    Returns ``(unit, local_tokens)``.

    The runtime's relation (``model_runner_kv_cache_mixin.py:5291``) is
    ``unit = local_capacity // ratio_r``, and the world pool is the MIN unit
    times ``sum(ratios)``.  Inverted for a REQUIRED world pool ``ctx + spec``::

        unit    = ceil((ctx + spec) / sum(ratios))
        local_r = page_up(unit * ratio_r)

    Both roundings are the opposite of the sizer's, deliberately:

    * ``ceil`` on the unit -- the sizer floors, so a unit taken with ``//``
      gives a world pool up to one ratio-sum short of the context.
    * ``page_up`` on the local count -- ``pool_configurator.py:534`` rounds the
      pool DOWN to a whole page, so a need that is not page-aligned is not met
      by a pool sized exactly to it.

    The speculative rows are a WORLD term, not a per-rank addend: draft and
    verify tokens land in the same DCP-sharded pool as everything else, so
    charging them per rank would charge them ``len(ratios)`` times.
    """
    ratios = [int(x) for x in dcp_ratios]
    if not ratios or any(x <= 0 for x in ratios):
        raise ValueError(f"dcp_ratios must be positive, got {list(dcp_ratios)}")
    if not 0 <= int(rank) < len(ratios):
        raise ValueError(f"rank {rank} outside DCP vector of length {len(ratios)}")
    world = int(ctx_tokens) + int(spec_tokens)
    if world <= 0:
        raise ValueError(f"ctx_tokens + spec_tokens must be > 0, got {world}")
    unit = -(-world // sum(ratios))  # ceil
    page = int(page_size)
    local = -(-(unit * ratios[int(rank)]) // page) * page  # page_up
    return unit, int(local)


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
    prefill_chunk_tokens: Optional[int] = None,
    floor_bytes: int = DEFAULT_FLOOR_BYTES,
    strict: bool = True,
) -> Tuple[float, float, Optional[str]]:
    """Bytes rank ``c`` has free for new pool rows.  ``(headroom, transient, note)``.

    THREE CORRECTIONS, one per dead boot, all of them subtractions:

    1. ``card free`` is the residual after the transient, the allocator cache,
       the fragmentation and the CUDA context -- so it is the headroom
       instrument, not ``peak - allocated`` (fn8ak2 was planned on that and
       died) and not ``reserved - allocated`` (which forbids boots that ran).
    2. But the ``card free`` the log prints was sampled at the moment
       ``maybe_log_vram_peak`` latched, which under chunked prefill is the
       FIRST chunk, not the deepest one.  The true peak is bigger by
       ``prefill_peak_allocated - peak_gib``, and the free column at that
       moment is smaller by the same amount.  Without a measured true peak a
       strict plan REFUSES -- the latched sample is a lower bound on the draw,
       and calling it a maximum is what killed fn8ak3.
    3. The transient is not one number per rank, it is a function of the
       prefill chunk.  ``graph_private_gib`` (CUDA-graph private pools) does
       not scale; the rest -- wave buffers, partials, the attention workspace
       that grows with the cached context a chunk attends over -- does, near
       enough to linearly to be worth pricing.  Halving CHUNK is therefore a
       real lever on the row count, and the dry run prints both.
    """
    instrument = float(c.peak_gib) - float(c.allocated_gib)
    note: Optional[str] = None
    if c.prefill_peak_allocated_gib is None:
        if strict:
            raise ExpertPoolBudgetRefused(
                c.rank,
                0.0,
                [
                    ("card free at the LATCHED sample", float(c.card_free_gib) * GIB),
                    ("transient the latched sample saw", instrument * GIB),
                ],
                0.0,
            )
        note = (
            f"rank{c.rank}: no measured prefill peak -- the latched "
            f"[vram-peak] sample ({instrument:.2f} GiB drawn) is a LOWER bound "
            "on the real draw, so the row count below is an UPPER BOUND and not "
            "a budget. Boot the high-water instrument first"
        )
        true_transient = instrument
    else:
        true_transient = float(c.prefill_peak_allocated_gib) - float(c.allocated_gib)
    # The free column was read while only `instrument` was drawn.
    free_gib = float(c.card_free_gib) - (true_transient - instrument)
    target = true_transient
    if prefill_chunk_tokens is not None and (
        int(c.prefill_chunk_tokens) <= 0 or c.prefill_peak_allocated_gib is None
    ):
        note = (note + "; " if note else f"rank{c.rank}: ") + (
            f"chunk {prefill_chunk_tokens} NOT priced -- this census carries no "
            "measured peak at a known chunk size, so the number below is the same "
            "at every chunk"
        )
    if (
        prefill_chunk_tokens is not None
        and int(c.prefill_chunk_tokens) > 0
        and c.prefill_peak_allocated_gib is not None
    ):
        scalable = max(0.0, true_transient - float(c.graph_private_gib))
        target = float(c.graph_private_gib) + scalable * (
            float(prefill_chunk_tokens) / float(c.prefill_chunk_tokens)
        )
        free_gib += true_transient - target
    headroom = free_gib * GIB + float(kv_credit_bytes) - float(floor_bytes)
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

    unit = 0
    needs: List[int] = []
    for c in ranks:
        unit, local = kv_tokens_for_rank(
            ctx_tokens=ctx_tokens,
            spec_tokens=spec_tokens,
            dcp_ratios=ratios,
            rank=c.rank,
            page_size=page,
        )
        needs.append(local)
    cap = int(max_total_tokens) if max_total_tokens is not None else max(needs)
    notes: List[str] = []

    plans: List[RankPlan] = []
    for c, need in zip(ranks, needs):
        # The rank allocates ``min(its own pool ceiling, cap)`` and the ceiling
        # is not known before the boot -- so the CAP is charged, never the
        # (possibly smaller) need.
        kv_tokens = (int(cap) // page) * page
        if need > kv_tokens:
            notes.append(
                f"rank{c.rank} needs {need} tokens for its DCP share but the cap "
                f"allows {kv_tokens}: the world unit binds at "
                f"{kv_tokens // c.dcp_ratio} and the served context drops to "
                f"{(kv_tokens // c.dcp_ratio) * sum(ratios)}"
            )
        kv_bytes = kv_tokens * int(c.cell_size)
        # CREDIT, not a post: shrinking the pool to the DCP need hands those
        # bytes back to the card before anything else runs.
        credit = float(c.kv_bytes_now - kv_bytes)
        card_free = float(c.card_free_gib) * GIB
        row_bytes = row_bytes_from_census(c.expert_tensor_gib, c.buffer_rows)
        headroom, transient_gib, note = rank_headroom_bytes(
            c,
            kv_credit_bytes=credit,
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
                    ("card free at worst observed state", card_free),
                    (
                        "prefill transient (measured, at this chunk)",
                        -transient_gib * GIB,
                    ),
                    ("KV pool credit (now - DCP need)", credit),
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
            # The fn8ak finding: R + C above E_local is not a bigger pool, it is
            # the RuntimeError at expert_offload.py:3898.  The excess belongs to
            # the OWNERSHIP vector (--rank-moe-ratio), not to this arena.
            bound_by = "ownership (buffer_size == R+C <= E_local)"
            surplus = rows - ownable
            taken = ownable
        scratch = c.scratch + taken
        # What the sizer must NOT hand back to KV next boot: everything between
        # this rank's available_bytes and the KV need, minus what the new rows
        # already take out of the same budget post.
        book = (float(c.available_bytes) - kv_bytes - taken * row_bytes) / MIB
        plans.append(
            RankPlan(
                rank=c.rank,
                kv_tokens=kv_tokens,
                kv_bytes=kv_bytes,
                kv_credit_bytes=credit,
                card_free_bytes=card_free,
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
    plan = plan_expert_pool(
        census,
        ctx_tokens=ctx_tokens,
        spec_tokens=spec_tokens,
        floor_bytes=floor_bytes,
        max_total_tokens=max_total_tokens,
        prefill_chunk_tokens=prefill_chunk_tokens,
        strict=strict,
    )
    proposed = [int(x) for x in scratch]
    if len(proposed) != len(plan.ranks):
        raise ValueError(
            f"scratch vector has {len(proposed)} entries for "
            f"{len(plan.ranks)} ranks"
        )
    out: List[str] = []
    for c, p, want in zip(sorted(census, key=lambda x: x.rank), plan.ranks, proposed):
        if c.residents + want > c.owned_experts:
            out.append(
                f"rank{c.rank}: C={want} breaks the pool identity -- "
                f"'pool mode requires buffer_size == R+C "
                f"({c.owned_experts} != {c.residents}+{want})'"
            )
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
_RE_DCP = re.compile(
    r"Uneven-DCP token sizing: rank (\d+) local capacity (\d+) tokens / ratio "
    r"(\d+)\b.*?vector \[([\d, ]+)\]"
)

_REQUIRED = (
    "available_bytes",
    "cell_size",
    "page_size",
    "kv_tokens_now",
    "card_free_gib",
    "card_total_gib",
    "reserved_gib",
    "allocated_gib",
    "peak_gib",
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

    * the ``[vram-peak]`` line kept per rank is the one with the MINIMUM
      ``card free`` -- the worst load state, which is the one a deployment
      serves;
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
            if free <= s.get("card_free_gib", float("inf")):
                s["peak_gib"] = float(m.group(1))
                s["allocated_gib"] = float(m.group(2))
                s["reserved_gib"] = float(m.group(3))
                s["card_free_gib"] = free
                s["card_total_gib"] = float(m.group(5))
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
        m = _RE_DCP.search(line)
        if m:
            s = slot(int(m.group(1)))
            s["kv_tokens_now"] = float(m.group(2))
            if not ratios:
                ratios = tuple(int(x) for x in m.group(4).split(","))

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
                kv_tokens_now=int(s["kv_tokens_now"]),
                card_free_gib=s["card_free_gib"],
                card_total_gib=s["card_total_gib"],
                reserved_gib=s["reserved_gib"],
                allocated_gib=s["allocated_gib"],
                peak_gib=s["peak_gib"],
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
            out.append(f"{head}\nREFUSED: {exc}")
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
