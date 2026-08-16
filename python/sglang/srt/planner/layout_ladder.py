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

  * ``FamilyPoolModel``   -- NVML free bytes, checkpoint weight bytes, the KV
                             geometry, and the checkpoint's own layer_types
                             vector (so a 0-GDN pure-attention model is just
                             the degenerate family split, not a special case);
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
from collections.abc import Sequence

from sglang.srt.planner.pp_cut import (
    FamilyPoolModel,
    PrefillTiming,
    _enumerate_cuts,
    attention_counts,
    family_phase_pool,
    pipelined_prefill_ms,
    serial_prefill_ms,
)


@dataclasses.dataclass(frozen=True)
class LadderInputs:
    """Measured inputs. Every field is probed; none is a policy constant."""

    pool: FamilyPoolModel
    ms_per_layer: tuple[float, ...]
    fixed_ms: tuple[float, ...]
    link_mib_per_s: tuple[float, ...]
    min_pool_tokens: float
    prefill_tokens_per_s: float
    # Fraction of a rung's pool held back as the admission ceiling. It is a
    # deployment input (the corridor's own safety statement), not a tuning
    # constant invented here.
    admit_fraction: float = 0.95

    def timing(self) -> PrefillTiming:
        return PrefillTiming(
            ms_per_layer=tuple(self.ms_per_layer), fixed_ms=tuple(self.fixed_ms)
        )


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
    families = model.layer_families
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
            pool = family_phase_pool(counts, model)
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
    families = inputs.pool.layer_families
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
