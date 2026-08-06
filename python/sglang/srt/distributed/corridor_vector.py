"""Corridor-constrained uneven-DCP token-vector solver (#602).

Under the weighted uneven-DCP owner rule the reported context budget is

    C(v) = min_r(E_r // v_r) * sum(v)

where ``v`` is the token-ownership vector and ``E_r`` is the number of KV
tokens rank ``r`` may physically hold. Rank ``r`` then stores exactly
``unit * v_r`` tokens with ``unit = min_r(E_r // v_r)``, so every rank whose
``v_r`` is not tight against its own capacity leaves ``E_r - unit * v_r``
tokens unallocated. Measured on the tp3 rig (2026-08-06): ~2.7 GiB of VRAM
sat unallocated across three cards purely because of that quantisation.

Two things are wrong with picking ``v`` by proportional rounding of the
profiled capacities alone, which is what the pre-#602 hint did:

1. Proportional rounding at a fixed grain is not the argmax of ``C``. The
   argmax has a closed form (see :func:`solve_token_vector`).
2. The profiled capacity ``P_r`` is a *budget-model* number. When the model
   over-states what the card can give, driving every rank tight against
   ``P_r`` drives the card BELOW the operator's free-VRAM floor. The floor is
   a hard constraint, not something to trade against tokens.

This module keeps those concerns separate. The floor enters as a *capacity*
``Q_r`` (the tokens that still leave the card at or above its reserve), the
solver runs on ``E_r = min(P_r, Q_r)``, and the objective is pure token
count. Because ``unit <= E_r // v_r`` for every rank by construction,
``unit * v_r <= E_r`` holds for every rank and every vector this module can
return -- the floor cannot be violated by any solution, so it never has to
appear as a penalty term in the objective.

Everything here is a pure function of its arguments so that every rank
derives the identical vector from the identical all-gathered inputs (the
rank-uniform install invariant the collective path depends on).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

# Granularity of the token vector. sum(vector) is the virtual block the
# weighted owner rule cycles over and the factor the KV page size is inflated
# by (see cp_token_split_factor), so it is deliberately bounded: a finer grain
# buys a fraction of a percent of context and costs allocation granularity on
# every request. 64 matches the pre-existing _CP_TOKEN_UNITS convention.
CORRIDOR_GRAIN = 64


class CorridorInfeasible(ValueError):
    """No vector can satisfy the free-VRAM floor on every card.

    Raised instead of silently shrinking the pool or falling back to the
    unconstrained vector: a card that cannot fund the operator reserve plus
    its own post-sizing demand is a configuration error, and the message
    names the card, the reserve, the demand and the shortfall.
    """


@dataclass(frozen=True)
class RankCapacity:
    """One DCP rank's admissible KV token count, from both sources.

    ``profiled_tokens`` is ``P_r``: what the budget model says fits.
    ``corridor_tokens`` is ``Q_r``: what still leaves this rank's physical
    card at or above its free-VRAM reserve once the pool is allocated and the
    modeled post-sizing demand has materialized. ``None`` means the corridor
    was not measured for this rank, in which case only the budget model
    constrains it.
    """

    dcp_rank: int
    profiled_tokens: int
    corridor_tokens: Optional[int] = None

    @property
    def effective_tokens(self) -> int:
        if self.corridor_tokens is None:
            return int(self.profiled_tokens)
        return min(int(self.profiled_tokens), int(self.corridor_tokens))

    @property
    def corridor_binds(self) -> bool:
        return self.corridor_tokens is not None and int(self.corridor_tokens) < int(
            self.profiled_tokens
        )


@dataclass(frozen=True)
class CorridorSolution:
    """The solved vector plus everything needed to justify it in a log line."""

    vector: List[int]
    context_tokens: int
    unit: int
    capacities: List[int]
    per_rank_tokens: List[int]
    grain: int

    @property
    def waste_tokens(self) -> List[int]:
        return [c - t for c, t in zip(self.capacities, self.per_rank_tokens)]

    @property
    def total_waste_tokens(self) -> int:
        return sum(self.waste_tokens)


def context_budget(vector: Sequence[int], capacities: Sequence[int]) -> int:
    """``C(v) = min_r(E_r // v_r) * sum(v)`` -- the objective, verbatim.

    Zero for a degenerate vector (a non-positive entry) so callers can score
    candidates without special-casing.
    """
    if len(vector) != len(capacities) or not vector:
        return 0
    if any(v <= 0 for v in vector):
        return 0
    return min(c // v for c, v in zip(capacities, vector)) * sum(vector)


def _unit_candidates(capacities: Sequence[int]) -> List[int]:
    """Every ``unit`` value at which the optimum can sit.

    For a fixed unit ``u`` the best vector is ``v_r = E_r // u`` (the largest
    ``v_r`` that still admits ``u``), giving ``C(u) = u * sum(E_r // u)``.
    Within a block of ``u`` where all the floors ``E_r // u`` are constant,
    ``C(u)`` is strictly increasing, so the optimum of every block sits at the
    block's right-hand end. Those ends are exactly the values ``E_r // k``,
    which is the standard divisor-block enumeration: O(n * sqrt(max E))
    candidates instead of O(max E).
    """
    ends = set()
    for cap in capacities:
        cap = int(cap)
        if cap <= 0:
            continue
        k = 1
        while k * k <= cap:
            ends.add(cap // k)
            ends.add(k)
            k += 1
        ends.add(cap)
    ends.discard(0)
    return sorted(ends)


def solve_token_vector(
    capacities: Sequence[int], grain: int = CORRIDOR_GRAIN
) -> CorridorSolution:
    """Maximize ``C(v)`` over integer vectors with ``v_r >= 1`` and
    ``sum(v) <= grain``.

    The floor is NOT part of the objective. It is already baked into
    ``capacities`` (``E_r``), and the returned vector satisfies
    ``unit * v_r <= E_r`` for every rank by construction, which is exactly
    the floor condition. Maximizing tokens can therefore never trade the
    floor away.

    Ties are broken toward the smaller ``sum(v)`` (coarser page inflation is
    cheaper) and then lexicographically, so the result is a deterministic
    pure function of the inputs -- every rank computes the same vector.
    """
    caps = [int(c) for c in capacities]
    n = len(caps)
    if n == 0:
        raise CorridorInfeasible("empty capacity vector")
    if any(c <= 0 for c in caps):
        raise CorridorInfeasible(
            "non-positive KV token capacity on DCP rank(s) "
            f"{[r for r, c in enumerate(caps) if c <= 0]} (capacities {caps}); "
            "the free-VRAM floor leaves no room for a KV pool on that card"
        )
    if grain < n:
        raise CorridorInfeasible(f"grain {grain} cannot give {n} ranks one unit each")

    from sglang.srt.distributed.utils import partition_units

    best_vec: Optional[List[int]] = None
    best_key = None

    def _consider(vec: List[int]) -> None:
        nonlocal best_vec, best_key
        if len(vec) != n or any(v <= 0 for v in vec):
            return
        total = sum(vec)
        if total > grain:
            return
        c = context_budget(vec, caps)
        key = (-c, total, tuple(vec))
        if best_key is None or key < best_key:
            best_key, best_vec = key, list(vec)

    # The closed-form family: one candidate per divisor block.
    for u in _unit_candidates(caps):
        vec = [c // u for c in caps]
        if any(v <= 0 for v in vec):
            continue
        if sum(vec) <= grain:
            _consider(vec)
        else:
            # Too fine for the grain: fall back to the proportional
            # projection of this shape onto the grain.
            _consider(partition_units(grain, vec))

    # The pre-#602 heuristic, kept as a candidate so the solver can never
    # return a WORSE vector than the one it replaces.
    _consider(partition_units(grain, caps))

    if best_vec is None:  # pragma: no cover - defended by the guards above
        raise CorridorInfeasible(f"no admissible vector for capacities {caps}")

    # gcd reduction never lowers C (coarser divisors floor less) and always
    # lowers sum(v), so it is applied unconditionally and the objective is
    # re-evaluated on the reduced form.
    g = math.gcd(*best_vec) if n > 1 else best_vec[0]
    if g > 1:
        best_vec = [v // g for v in best_vec]

    unit = min(c // v for c, v in zip(caps, best_vec))
    return CorridorSolution(
        vector=list(best_vec),
        context_tokens=unit * sum(best_vec),
        unit=unit,
        capacities=caps,
        per_rank_tokens=[unit * v for v in best_vec],
        grain=grain,
    )


def solve_corridor_vector(
    ranks: Sequence[RankCapacity], grain: int = CORRIDOR_GRAIN
) -> CorridorSolution:
    """:func:`solve_token_vector` on ``E_r = min(P_r, Q_r)``.

    ``ranks`` must cover every DCP rank exactly once; the solution is indexed
    by ``dcp_rank``.
    """
    order = sorted(ranks, key=lambda r: r.dcp_rank)
    expected = list(range(len(order)))
    if [r.dcp_rank for r in order] != expected:
        raise CorridorInfeasible(
            f"capacity list does not cover DCP ranks {expected} exactly once: "
            f"{[r.dcp_rank for r in ranks]}"
        )
    return solve_token_vector([r.effective_tokens for r in order], grain=grain)


def corridor_pool_bytes(
    free_bytes: int,
    reserve_mib: int,
    post_sizing_mib: int,
    colocated_ranks: int = 1,
) -> int:
    """Bytes ONE rank may put into its KV pool without breaking the floor.

    ``free_bytes`` is the driver/NVML free memory of the rank's PHYSICAL card
    measured after every rank on it has loaded its weights and before any pool
    is allocated. ``reserve_mib`` is the operator's free-VRAM floor for that
    card. ``post_sizing_mib`` is the modeled demand on that card that has not
    materialized yet at the measuring point (graph capture, activation peak,
    attention workspaces, ...) -- the pool must leave room for it too, or the
    floor is broken later rather than now.

    The remainder is split evenly among the ranks sharing the card, which is
    the correct split under pure TP/DCP co-location: co-located ranks are
    structurally symmetric.

    Returns 0 (never negative) so the caller can report an infeasible card
    with its own message and numbers.
    """
    if colocated_ranks < 1:
        raise ValueError(f"colocated_ranks must be >= 1, got {colocated_ranks}")
    allow = int(free_bytes) - (int(reserve_mib) << 20) - (int(post_sizing_mib) << 20)
    if allow <= 0:
        return 0
    return allow // colocated_ranks
