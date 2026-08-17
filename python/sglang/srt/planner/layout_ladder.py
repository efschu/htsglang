"""#704 part A: the prefill layout ladder, solved from measured inputs.

The PP layer cut is a pressure axis. Deep cuts (many layers on the fast rank)
shorten the pipeline but shrink the KV pool, because under PP prefill the pool
is layer-sharded and a rank's capacity is divided by its own attention-layer
count (#702). Speed and capacity therefore oppose along the cut -- but the
opposition only BINDS when the pool is full. At low fill the capacity
constraint is slack and the speed is free.

So the cut becomes a ladder: run a deep fast rung while the pool is empty, and
step back toward pool-max rungs as fill rises.

Everything here is SOLVED, never tabulated. Rungs, admission thresholds,
hysteresis bands and move costs are functions of measured inputs:

  * ``PhasePoolModel``    -- Slot-2's canonical pool model (rev5): NVML free
                             bytes, checkpoint weight bytes, the KV cell taken
                             FROM CONFIG, the per-layout arming floor and the
                             GDN residency. This module does not reimplement
                             it; an earlier duplicate of mine was deleted when
                             the attention-only divisor was adjudicated;
  * ``layer_families``    -- the checkpoint's own layer_types vector, so a
                             0-GDN pure-attention model is the degenerate
                             family split rather than a special case;
  * ``arming_floor_for``  -- the per-LAYOUT arming floor. Required, because a
                             constant floor is the error that over-predicted an
                             unchanged rank's measured capacity by 4.8%;
  * ``ms_per_layer``      -- the per-rank per-layer prefill self-probe;
  * ``link_mib_per_s``    -- the measured pair-matrix link reach per rank;
  * ``min_pool_tokens``   -- the corridor floor the deployment must serve;
  * ``prefill_tokens_per_s`` -- the measured fill rate, which is what turns a
                             move TIME into a fill headroom requirement.

No rig constant appears below. The rank0 depth cap is not typed in: it falls
out of the same weights-plus-corridor arithmetic that prices every rung, so it
moves when the card moves.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Callable, Sequence

from sglang.srt.planner.pp_cut import (
    PhasePoolModel,
    PrefillTiming,
    _enumerate_cuts,
    attention_counts,
    pipelined_prefill_ms,
    pp_phase_pool,
    serial_prefill_ms,
)


@dataclasses.dataclass(frozen=True)
class LadderInputs:
    """Measured inputs. Every field is probed; none is a policy constant."""

    pool: PhasePoolModel
    layer_families: tuple[str, ...]
    ms_per_layer: tuple[float, ...]
    fixed_ms: tuple[float, ...]
    link_mib_per_s: tuple[float, ...]
    min_pool_tokens: float
    prefill_tokens_per_s: float
    # Per-rank arming floor FOR A GIVEN CUT. The floor moves with the layout
    # (measured 2255/1728/2467 MiB on [32,16,16] against 1728/1825/2467 on
    # [28,20,16]), and treating it as a constant carried ~84% of the +9.6%
    # common-mode over-prediction that sank three boots. So the ladder cannot
    # hold one floor: it asks per rung.
    #
    # KNOWN GAP, inherited from the #676 machinery: that solver derives the
    # floor from a MEASURED seam draw, so a cut that has never booted has no
    # solved floor. Every unbooted rung therefore carries its proxy's floor
    # uncertainty -- about +-500 MiB, which at 8 attention layers is +-32,000
    # tokens (~7%). A ladder is mostly unbooted rungs, so this is the dominant
    # error term in every pool number below, and it is the reason a rung's
    # predicted pool is not a boot gate on its own.
    arming_floor_for: Callable[[Sequence[int]], Sequence[float]] = None  # type: ignore[assignment]
    # Fraction of a rung's pool held back as the admission ceiling. It is a
    # deployment input (the corridor's own safety statement), not a tuning
    # constant invented here.
    admit_fraction: float = 0.95

    def __post_init__(self) -> None:
        if self.arming_floor_for is None:
            raise ValueError(
                "arming_floor_for is required: the arming floor is per-layout, "
                "and pricing a ladder with one constant floor is the error that "
                "produced a 4.8% over-prediction of an UNCHANGED rank's measured "
                "capacity. Pass the #676 solver "
                "(phase_flip_seam_reserve.arming_floor_target_bytes), or an "
                "explicit proxy whose uncertainty you are willing to carry."
            )

    def timing(self) -> PrefillTiming:
        return PrefillTiming(
            ms_per_layer=tuple(self.ms_per_layer), fixed_ms=tuple(self.fixed_ms)
        )

    def pool_model_for(self, counts: Sequence[int]) -> PhasePoolModel:
        """The pool model with THIS layout's arming floor substituted in."""
        return dataclasses.replace(
            self.pool,
            arming_floor_mib=tuple(float(x) for x in self.arming_floor_for(counts)),
        )


def _pool_for(counts: Sequence[int], inputs: LadderInputs) -> float:
    """PP-phase pool for one cut, priced with that cut's own arming floor."""
    attn = attention_counts(inputs.layer_families, counts)
    return pp_phase_pool(counts, attn, inputs.pool_model_for(counts))


@dataclasses.dataclass(frozen=True)
class Rung:
    counts: tuple[int, ...]
    attn_counts: tuple[int, ...]
    pool_tokens: float
    serial_speedup: float
    pipelined_speedup: float
    admit_up_to_tokens: float


@dataclasses.dataclass(frozen=True)
class Transition:
    """A step between two adjacent rungs, priced on the measured link.

    ``shallower`` is the roomier/slower rung, ``deeper`` the faster/tighter one.
    Descending means moving layers ONTO the fast rank; ascending means giving
    them back.
    """

    shallower: tuple[int, ...]
    deeper: tuple[int, ...]
    weight_move_mib: float
    weight_move_ms: float
    attn_layers_moved: int
    kv_move_mib_per_token: float
    descend_below_tokens: float
    ascend_above_tokens: float


@dataclasses.dataclass(frozen=True)
class Ladder:
    rungs: tuple[Rung, ...]
    transitions: tuple[Transition, ...]


def _boundaries(counts: Sequence[int]) -> list[int]:
    out: list[int] = []
    s = 0
    for c in counts[:-1]:
        s += int(c)
        out.append(s)
    return out


def _moved_layers(
    a: Sequence[int], b: Sequence[int], families: Sequence[str]
) -> tuple[int, int]:
    """Layers and attention-layers crossing a boundary between two cuts."""
    total = 0
    attn = 0
    for x, y in zip(_boundaries(a), _boundaries(b)):
        lo, hi = min(x, y), max(x, y)
        total += hi - lo
        attn += sum(1 for i in range(lo, hi) if families[i] == "full_attention")
    return total, attn


def solve_layout_ladder(inputs: LadderInputs) -> Ladder:
    """Enumerate cuts, keep the Pareto frontier, and price every step.

    The frontier is the ladder: ordered from most pool / least speed to least
    pool / most speed, with every rung clearing the corridor floor. A cut that
    is beaten on both axes is not a rung.
    """
    model = inputs.pool
    families = inputs.layer_families
    depth = len(families)
    n_ranks = len(model.free_mib)
    if len(inputs.ms_per_layer) != n_ranks:
        raise ValueError(
            f"the timing probe covers {len(inputs.ms_per_layer)} ranks but the "
            f"memory census covers {n_ranks}."
        )
    if len(inputs.link_mib_per_s) != n_ranks:
        raise ValueError(
            f"the link matrix covers {len(inputs.link_mib_per_s)} ranks but the "
            f"memory census covers {n_ranks}."
        )
    timing = inputs.timing()

    # The reference for "speedup" is the best cut this hardware can serve at
    # full pool -- i.e. the ladder's own top rung, resolved below. Enumerate
    # first, then normalize, so no incumbent layout is baked in.
    priced: list[tuple[tuple[int, ...], float, float, float]] = []
    for counts in _enumerate_cuts(depth, n_ranks, 1):
        try:
            pool = _pool_for(counts, inputs)
        except ValueError:
            # Infeasible on weights, or a stage with no attention layer at all.
            # Both are refusals, not candidates. The rank0 depth cap lives here
            # and nowhere else: it is whatever the arithmetic permits.
            continue
        if pool < float(inputs.min_pool_tokens):
            continue
        priced.append(
            (
                tuple(int(c) for c in counts),
                pool,
                serial_prefill_ms(counts, timing),
                pipelined_prefill_ms(counts, timing),
            )
        )
    if not priced:
        raise ValueError(
            "no cut clears the corridor floor of "
            f"{float(inputs.min_pool_tokens):,.0f} tokens on this hardware; the "
            "ladder has no rungs and the layout cannot be chosen by pressure."
        )

    # Pareto frontier: sort by pool descending, keep strict speed improvements.
    priced.sort(key=lambda p: (-p[1], p[3]))
    front: list[tuple[tuple[int, ...], float, float, float]] = []
    best_pipe: float | None = None
    for counts, pool, serial_ms, pipe in priced:
        if best_pipe is None or pipe < best_pipe - 1e-12:
            best_pipe = pipe
            front.append((counts, pool, serial_ms, pipe))

    base_ser = front[0][2]
    base_pipe = front[0][3]
    rungs = tuple(
        Rung(
            counts=counts,
            attn_counts=attention_counts(families, counts),
            pool_tokens=pool,
            serial_speedup=base_ser / serial_ms,
            pipelined_speedup=base_pipe / pipe,
            admit_up_to_tokens=pool * float(inputs.admit_fraction),
        )
        for counts, pool, serial_ms, pipe in front
    )

    transitions = _solve_transitions(rungs, inputs)
    return Ladder(rungs=rungs, transitions=transitions)


def _move_ms_at(
    tokens: float, weight_ms: float, kv_per_token: float, gating_link: float
) -> float:
    """Total move time at a given live-token count.

    The weight half is fixed; the KV half is whatever must follow its layer at
    that occupancy. Under part B's decoupled layout ``kv_per_token`` is zero by
    construction and the move time stops depending on fill entirely.
    """
    return weight_ms + (kv_per_token * tokens) / gating_link * 1000.0


def _solve_transitions(
    rungs: Sequence[Rung], inputs: LadderInputs
) -> tuple[Transition, ...]:
    """Price each adjacent step and derive its two thresholds.

    The bands are not chosen; they are consequences:

    * **Ascend** (retreat to the roomier rung) must START while the current
      rung still has headroom for the tokens that will arrive during the move.
      So the trigger sits a full move-window below the rung's admission
      ceiling: ``admit - fill_rate * move_time``.
    * **Descend** (advance to the faster rung) may only fire when the live set
      fits the deeper rung's ceiling with the same move-window to spare.

    If those two cross, the pair is not a step: one fill level would demand
    both moves, which is the oscillation this solver exists to prevent. Such a
    pair is dropped, and a link slow enough collapses the ladder to nothing --
    which is the honest answer, not a ladder that thrashes.
    """
    families = inputs.layer_families
    out: list[Transition] = []
    fill_rate = float(inputs.prefill_tokens_per_s)
    for a, b in itertools.pairwise(rungs):
        moved, attn_moved = _moved_layers(a.counts, b.counts, families)
        weight_mib = moved * float(inputs.pool.weight_mib_per_layer)
        # The moved layers land on whichever rank gains them; charge the move
        # to the slowest link involved, which is the one that gates it.
        gating_link = min(float(x) for x in inputs.link_mib_per_s)
        if gating_link <= 0.0:
            raise ValueError("a measured link bandwidth of zero cannot move a layer.")
        weight_ms = weight_mib / gating_link * 1000.0
        # KV that must follow its layer, per live token, under the coupled
        # (part A) layout. Part B drives this term to zero by construction.
        kv_per_token = attn_moved * float(inputs.pool.kv_mib_per_token_per_attn_layer)

        # Ascend: leave the deeper rung before it runs out.
        ascend = b.admit_up_to_tokens
        for _ in range(8):  # KV term depends on the fill it is evaluated at
            move_ms = _move_ms_at(ascend, weight_ms, kv_per_token, gating_link)
            ascend = b.admit_up_to_tokens - fill_rate * move_ms / 1000.0
        # Descend: only into a rung that can hold the live set plus the window.
        descend = (
            b.admit_up_to_tokens
            - fill_rate
            * _move_ms_at(max(ascend, 0.0), weight_ms, kv_per_token, gating_link)
            / 1000.0
            * 2.0
        )
        if not 0.0 <= descend < ascend:
            # Either the bands cross, or they land below zero fill -- which
            # says the move consumes more headroom than the rung has at ANY
            # occupancy, so there is no fill level at which the controller
            # could sit on the deeper rung. Both are refusals: a step that
            # cannot be funded is not a step, and emitting it would produce a
            # ladder that thrashes on metal.
            continue
        out.append(
            Transition(
                shallower=a.counts,
                deeper=b.counts,
                weight_move_mib=weight_mib,
                weight_move_ms=weight_ms,
                attn_layers_moved=attn_moved,
                kv_move_mib_per_token=kv_per_token,
                descend_below_tokens=descend,
                ascend_above_tokens=ascend,
            )
        )
    return tuple(out)


# ---------------------------------------------------------------------------
# #704 Slice 1 route: the arena ladder.
#
# There is NO cross-rank runtime weight mover in this tree, and its absence is
# explicit rather than accidental: `regime_stages.py:100` defines
# REACH_NO_WEIGHT_MOVER ("no runtime actuator moves weights -- switching arms
# needs a restart"), and `regime_act.py:121-130` wires exactly three axes, kv /
# vram / phase, never weight.
#
# What DOES exist is `model_executor/weights_arena.py` plus
# `managers/phase_flip_boot.py:361`: a fixed-address VRAM arena per rank,
# refilled by a contiguous host->device memcpy from a boot-baked pinned image,
# checksum-verified, with a rollback arm. It performs no collective and moves
# nothing between ranks -- it swaps which pre-loaded layout occupies a rank's
# own arena.
#
# That is enough for the ladder, and on this rig it is the BETTER primitive:
# with no P2P (all PHB), a rank-to-rank weight transfer would stage through
# host memory anyway, so an H2D refill from pinned host memory costs the same
# link time without any new collective. It also keeps the tensor's VRAM address
# fixed, which keeps captured CUDA graphs valid across a rung change.
#
# The price is the constraint modelled here: one arena per rank, sized for the
# DEEPEST rung the ladder can reach. A rank does not get its weight bytes back
# when the ladder sits on a shallow rung, so free memory for KV is pinned at
# the deepest rung's weight cost at EVERY rung.
#
# The structural consequence is not obvious and it is what makes this model
# worth having: since free memory no longer varies across rungs, a rung's pool
# depends only on its ATTENTION count. Two rungs with the same attention
# profile therefore have exactly equal pool, and the faster of them strictly
# dominates -- so they are not a ladder pair at all. Under an arena the ladder's
# real axis is the attention-count vector, not the raw layer cut.
# ---------------------------------------------------------------------------


def arena_layers_for(rungs: Sequence[Sequence[int]]) -> tuple[int, ...]:
    """Per-rank arena size, in layers: the max over every reachable rung."""
    if not rungs:
        raise ValueError("an empty rung set has no arena to size.")
    n_ranks = len(rungs[0])
    return tuple(max(int(r[i]) for r in rungs) for i in range(n_ranks))


def rung_family(
    inputs: LadderInputs, shallowest_rank0_layers: int, deepest_rank0_layers: int
) -> tuple[tuple[int, ...], ...]:
    """The candidate rungs: one per rank0 depth in the requested span.

    For a fixed rank0 depth the remaining layers are split across the trailing
    stages to minimise pipelined time. That choice depends only on the timing
    probe, never on the pool, which is what keeps the arena non-circular: the
    family is fixed first, then the arena it implies prices every member.
    """
    model = inputs.pool
    depth = len(inputs.layer_families)
    n_ranks = len(model.free_mib)
    timing = inputs.timing()
    lo, hi = int(shallowest_rank0_layers), int(deepest_rank0_layers)
    if not 1 <= lo <= hi:
        raise ValueError(f"rank0 span [{lo}, {hi}] is not a valid ladder range.")
    out: list[tuple[int, ...]] = []
    for n0 in range(lo, hi + 1):
        rest = depth - n0
        if rest < n_ranks - 1:
            break
        best: tuple[int, ...] | None = None
        best_pipe = float("inf")
        for tail in _enumerate_cuts(rest, n_ranks - 1, 1):
            counts = (n0, *(int(t) for t in tail))
            pipe = pipelined_prefill_ms(counts, timing)
            if pipe < best_pipe - 1e-12:
                best_pipe = pipe
                best = counts
        if best is not None:
            out.append(best)
    if not out:
        raise ValueError(
            f"no feasible cut exists for a rank0 depth in [{lo}, {hi}] over "
            f"{depth} layers and {n_ranks} stages."
        )
    return tuple(out)


def solve_arena_ladder(
    inputs: LadderInputs,
    deepest_rank0_layers: int,
    shallowest_rank0_layers: int | None = None,
) -> Ladder:
    """Solve the ladder under one shared, resident weight arena per rank.

    ``deepest_rank0_layers`` is the depth the deployment wants to be able to
    reach. It sets the arena, and the arena then prices every rung -- so asking
    for a deeper ladder makes the shallow rungs poorer, which is a real trade
    the caller must see rather than a detail to hide.
    """
    model = inputs.pool
    families = inputs.layer_families
    n_ranks = len(model.free_mib)
    timing = inputs.timing()

    if shallowest_rank0_layers is None:
        shallowest_rank0_layers = int(deepest_rank0_layers)
    candidates = list(
        rung_family(inputs, shallowest_rank0_layers, deepest_rank0_layers)
    )
    arena = arena_layers_for(candidates)

    # Free memory after the arena, which does NOT vary with the current rung.
    free_after_arena = []
    for r in range(n_ranks):
        free = float(model.free_mib[r]) - float(model.weight_mib_per_layer) * arena[r]
        if free < 0.0:
            raise ValueError(
                f"the arena for rank{r} needs {arena[r]} layers "
                f"({float(model.weight_mib_per_layer) * arena[r]:,.1f} MiB) against "
                f"{float(model.free_mib[r]):,.1f} MiB free: "
                f"{-free:,.1f} MiB short. A shallower ladder is required."
            )
        free_after_arena.append(free)

    priced: list[tuple[tuple[int, ...], float, float, float]] = []
    for counts in candidates:
        attn = attention_counts(families, counts)
        if any(a <= 0 for a in attn):
            continue
        # GDN state and the arming floor are token-independent but BOTH follow
        # the layout, so they are charged per rung even though the arena is not.
        floor = tuple(float(x) for x in inputs.arming_floor_for(counts))
        caps = []
        for r in range(n_ranks):
            linear = counts[r] - attn[r]
            free = (
                free_after_arena[r]
                - float(model.mamba_mib_per_linear_layer_per_slot)
                * linear
                * int(model.mamba_slots)
                - floor[r]
            )
            if free < 0.0:
                break
            caps.append(free / (attn[r] * float(model.kv_mib_per_token_per_attn_layer)))
        if len(caps) != n_ranks:
            continue
        pool = min(caps)
        if pool < float(inputs.min_pool_tokens):
            continue
        priced.append(
            (
                counts,
                pool,
                serial_prefill_ms(counts, timing),
                pipelined_prefill_ms(counts, timing),
            )
        )
    if not priced:
        raise ValueError(
            "no cut clears the corridor floor of "
            f"{float(inputs.min_pool_tokens):,.0f} tokens once the arena for a "
            f"{int(deepest_rank0_layers)}-layer rank0 is resident."
        )

    priced.sort(key=lambda p: (-p[1], p[3]))
    front: list[tuple[tuple[int, ...], float, float, float]] = []
    best_pipe: float | None = None
    for counts, pool, serial_ms, pipe in priced:
        if best_pipe is None or pipe < best_pipe - 1e-12:
            best_pipe = pipe
            front.append((counts, pool, serial_ms, pipe))

    base_serial = front[0][2]
    base_pipe = front[0][3]
    rungs = tuple(
        Rung(
            counts=counts,
            attn_counts=attention_counts(families, counts),
            pool_tokens=pool,
            serial_speedup=base_serial / serial_ms,
            pipelined_speedup=base_pipe / pipe,
            admit_up_to_tokens=pool * float(inputs.admit_fraction),
        )
        for counts, pool, serial_ms, pipe in front
    )
    return Ladder(rungs=rungs, transitions=_solve_transitions(rungs, inputs))
