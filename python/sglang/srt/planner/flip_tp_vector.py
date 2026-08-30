"""#1017: planner-solved TP weight vector for the phase flip.

``--phase-flip-tp-vector`` declares how the TP decode phase splits the model's
width across the ranks the PP topology already owns. It has been a HAND-PIN
since the flip existed -- ``32,16,16`` on the reference rig -- and it is the
last large hand-set axis on the performance surface.

Provenance, so this module's reason to exist is checkable and does not have to
be rediscovered (the ANALYSE_799 sec 5.4 lesson: a document does not reach a
build on its own, so the build cites the document):

* ``/spinning/evidence-665-f1/PROVENANCE_32_16_16_boot.md``, gap **G10**:
  "``--phase-flip-tp-vector 32,16,16`` | Typed TP-phase weight shard.
  Coincidentally equals the new cut; unrelated axis. | solve() from per-rank
  compute capability."  That last clause is this module's specification.
* ``/spinning/gpu-arb/HANDOVER_855_0830.md`` sec 4 item 4 records the measured
  candidate space -- ``32,16,16`` / ``34,15,15`` / ``42,11,11``, "sum must stay
  64; ``43,11,11`` was rejected, sum 65 fails all three unitless dims" -- and,
  decisively for the cost model, "``kv_heads [2,1,1]`` is invariant across
  candidates, so the KV cell does not move and the whole prize is weight mass."
* ``/spinning/gpu-arb/progress.corridor-planner`` (2026-08-21) is the desk
  proof that this axis is derivable rather than only measurable:
  ``scoped_tp_partition_ratios([32,16,16])`` into ``plan_arena_layout``
  reproduced the boot-measured per-rank TP arena to **zero MiB** error, and
  records the verdict this module acts on -- "cut and vector are one joint
  solve, and the planner is the place for it."

WHAT THE VECTOR ACTUALLY IS -- the unit trap, caught by execution.
The entries are **scale-free weights**, not head counts. The reference rig's
checkpoint declares 24 query heads while the pin ``32,16,16`` sums to 64, so
reading the vector as heads is wrong by construction.
``distributed/utils.partition_sizes`` applies it two ways, and the second is
where the sum stops being arbitrary:

* for a dimension WITH a unit count (attention heads): largest-remainder over
  indivisible units, every rank at least one -- "any positive weights work";
* for a dimension WITHOUT one: ``total % sum(weights)`` must be zero, or it
  raises -- "choose weights whose sum divides every sharded dimension".

So ``sum == 64`` is not a convention and not a magic number: it is the
constraint that the sum must DIVIDE every unitless sharded dimension, and it
is exactly why HANDOVER_855 records ``43,11,11`` (sum 65) as rejected against
"all three unitless dims". This solver derives that admissible sum from the
model's own geometry instead of inheriting it.

WHAT THIS SOLVER IS NOT ALLOWED TO BECOME (#1010, commit ``949bd6e305``).
``apply_auto_performance``'s MLP-vector solve is REFUSED under ``pp_size > 1``
and that refusal was deliberately upheld, not bridged. Two of its recorded
reasons bind here, and are why this is its own solver rather than a new lane
in ``uneven_perf``:

1. ``apply_auto_performance`` indexes budgets by ``tp_size`` and scores by
   ``--rank-gpu-id``, which is WORLD-length under PP -- any vector it solved
   would be indexed wrong.
2. "On a phase-flip boot the TP phase's weight split is already declared by
   ``--phase-flip-tp-vector``" -- a second mechanism writing that split would
   be a second authority for one payload.

This solver writes ``phase_flip_tp_vector`` and nothing else, is the only
writer of it besides the operator's own flag, never touches ``rank_mlp_ratio``,
and must never be reached from ``apply_auto_performance``.

THE HAND-PIN REMAINS A VALIDATED OVERRIDE: passing the vector explicitly still
wins -- by being the only one passed, not by silently overruling a solved
vector (the ``--pp-solve-cut`` rule in ``server_args._handle_pp_solve_cut``).

MEASUREMENT HONESTY -- READ BEFORE TREATING THE OUTPUT AS CALIBRATED.
The cost model is a CAPABILITY model, not a measurement of the TP phase,
because the measurement does not exist: on ``boot_855_census1028i`` the
compute-honest instrument (the ``gpu-ms (compute, wait)`` suffix on ``Prefill
rank batch``) carries 39 rows and **all 39 are ``phase=pp``**; the 32
``phase=tp`` prefill rows carry no gpu_ms at all, and TP ``decode`` reports one
identical ``gen_tps`` on all three ranks -- a group aggregate, not a per-rank
cost. Pricing this vector off the PP-phase numbers would be the
``instrument-text-luegt`` class-B error (a wrong conclusion from a correct
instrument), and is refused here rather than done quietly. ``describe()`` says
so in the emitted provenance line, every time.
"""

from __future__ import annotations

import dataclasses
from math import gcd
from typing import Iterator, List, Optional, Sequence, Tuple

MIB = 1024.0 * 1024.0

#: The sum the reference rig's incumbent vector carries. Used only as the
#: PREFERRED choice when the model's geometry admits it, never as a default
#: that survives a geometry that rejects it.
INCUMBENT_RATIO_TOTAL = 64


@dataclasses.dataclass(frozen=True)
class RankCapability:
    """One rank's side of the solve, in the units the census already speaks."""

    #: Human label, e.g. ``"rank0-NVIDIA GeForce RTX 5090"``.
    label: str
    #: Per-rank compute score. Any positive scale works -- only RATIOS are
    #: used -- so ``_pp_cut_card_rates``' GEMM TFLOPs drop straight in.
    gemm_score: float
    #: Memory this rank may spend on its TP-phase shard, MiB.
    budget_mib: float
    #: Memory already spoken for before the shard lands, MiB (census
    #: residual: CUDA context, allocator floor).
    fixed_overhead_mib: float = 0.0


def admissible_ratio_totals(
    unitless_dims: Sequence[int],
    n_ranks: int,
    max_total: int = 512,
) -> List[int]:
    """Every vector sum that ``partition_sizes`` will accept, ascending.

    A sum is admissible exactly when it divides EVERY unitless sharded
    dimension -- i.e. when it divides their gcd -- and leaves at least one
    unit per rank. This is the constraint that rejected ``43,11,11`` (sum 65)
    on the reference rig, and deriving it is the difference between a solver
    and a solver that inherited someone's 64.
    """
    dims = [int(d) for d in unitless_dims if int(d) > 0]
    if not dims:
        # No unitless dimension constrains the sum; any total >= n_ranks works.
        return [t for t in range(n_ranks, max_total + 1)]
    g = 0
    for d in dims:
        g = gcd(g, d)
    return [t for t in range(n_ranks, min(g, max_total) + 1) if g % t == 0]


@dataclasses.dataclass(frozen=True)
class FlipTPVectorInputs:
    """Everything the vector solve consumes. No torch, no CUDA, no I/O."""

    ranks: Tuple[RankCapability, ...]
    #: The sum the emitted vector must carry. MUST divide every entry of
    #: ``unitless_dims`` -- see :func:`admissible_ratio_totals`.
    ratio_total: int
    #: Sharded dimensions that carry no unit count, whose divisibility the sum
    #: must respect. Validated here so an inadmissible sum is refused at the
    #: desk instead of raising deep inside a weight load.
    unitless_dims: Tuple[int, ...] = ()
    #: The model's query-head count, used to REPORT the realized head split
    #: the ratio produces. Not the vector's unit.
    head_units: int = 0
    #: Total KV heads, for the same reporting purpose.
    total_kv_heads: int = 0
    #: Weight bytes that are split across ranks in proportion to the vector.
    shardable_bytes: float = 0.0
    #: Weight bytes every rank holds regardless of its share.
    replicated_bytes: float = 0.0
    #: Smallest share any rank may receive. One, not zero: a rank with no
    #: width is not a thinner rank, it is a rank that is not in the group.
    min_units: int = 1

    def __post_init__(self) -> None:
        if len(self.ranks) < 2:
            raise ValueError(
                f"a TP split needs at least 2 ranks, got {len(self.ranks)}"
            )
        if self.ratio_total < self.min_units * len(self.ranks):
            raise ValueError(
                f"ratio_total {self.ratio_total} cannot give every one of "
                f"{len(self.ranks)} ranks its minimum {self.min_units}"
            )
        bad = [d for d in self.unitless_dims if int(d) % self.ratio_total]
        if bad:
            raise ValueError(
                f"ratio_total {self.ratio_total} does not divide unitless "
                f"sharded dimension(s) {bad}. partition_sizes would raise on "
                f"the first such dimension at weight-load time; refused here "
                f"instead. Admissible sums: "
                f"{admissible_ratio_totals(self.unitless_dims, len(self.ranks))}"
            )
        for rank in self.ranks:
            if rank.gemm_score <= 0:
                raise ValueError(
                    f"{rank.label}: gemm_score must be positive, got "
                    f"{rank.gemm_score}. An unpriced rank does not read as "
                    f"'unknown', it reads as 'free' (the C38/C39 lesson)."
                )


@dataclasses.dataclass(frozen=True)
class FlipTPVectorSolution:
    """The emitted vector plus every number that chose it."""

    counts: Tuple[int, ...]
    #: The head split this ratio actually realizes (largest remainder over
    #: indivisible heads), or () when no head count was supplied.
    realized_heads: Tuple[int, ...]
    kv_heads: Tuple[int, ...]
    #: Predicted TP-phase makespan, ``max_r share_r/score_r``. Comparable only
    #: BETWEEN candidates of one solve, never across boots.
    makespan: float
    bottleneck_rank: int
    shard_mib: Tuple[float, ...]
    headroom_mib: Tuple[float, ...]
    #: How far the integer vector sits from the unrounded capability ideal.
    quantization_gap: float
    candidates_considered: int
    ideal: Tuple[float, ...]
    ratio_total: int

    def vector_string(self) -> str:
        return ",".join(str(c) for c in self.counts)

    def summary(self) -> str:
        return (
            f"sum {self.ratio_total}, makespan {self.makespan:.4f} "
            f"(bottleneck rank {self.bottleneck_rank}), realized heads "
            f"{list(self.realized_heads)}, kv heads {list(self.kv_heads)}, "
            f"min headroom {min(self.headroom_mib):.1f} MiB, quantization "
            f"gap {self.quantization_gap * 100:.1f}% off the capability "
            f"ideal {[round(v, 2) for v in self.ideal]}, "
            f"{self.candidates_considered} candidate(s) considered"
        )

    def describe(self) -> str:
        """The honesty clause that rides with every emission."""
        return (
            "BASIS: per-rank compute CAPABILITY (GEMM score) x width mass, "
            "not a measured TP-phase cost -- no per-rank compute instrument "
            "exists inside the TP phase today (boot_855_census1028i: all 39 "
            "compute-honest prefill rows are phase=pp; TP decode gen_tps is "
            "one group aggregate repeated per rank). Treat this vector as "
            "DESK-DERIVED and confirm it against a TP-phase measurement "
            "before calling it calibrated."
        )


class FlipTPVectorRefused(ValueError):
    """No vector satisfies the constraints. Carries every reason."""

    def __init__(self, reasons: Sequence[str]) -> None:
        self.reasons: Tuple[str, ...] = tuple(reasons)
        super().__init__(
            "no --phase-flip-tp-vector satisfies this rig: " + "; ".join(reasons)
        )


def partition_units_largest_remainder(
    units: int, weights: Sequence[int]
) -> Tuple[int, ...]:
    """Largest-remainder split of ``units`` indivisible units by ``weights``.

    Mirrors what ``distributed/utils.partition_units`` will do at load time --
    every rank at least one unit -- so the solver reports the split the
    runtime will actually build rather than a fractional one that cannot
    exist. Kept independent of the distributed module so the planner stays
    importable with no torch present.
    """
    n = len(weights)
    if units <= 0:
        return tuple(0 for _ in weights)
    if units < n:
        # Fewer units than ranks: the runtime replicates rather than split.
        return tuple(units for _ in weights)
    denom = sum(weights)
    exact = [units * w / denom for w in weights]
    out = [max(1, int(v)) for v in exact]
    short = units - sum(out)
    order = sorted(range(n), key=lambda i: exact[i] - int(exact[i]), reverse=True)
    i = 0
    while short > 0:
        out[order[i % n]] += 1
        short -= 1
        i += 1
    while short < 0:
        biggest = max(range(n), key=lambda k: out[k])
        if out[biggest] <= 1:
            break
        out[biggest] -= 1
        short += 1
    return tuple(out)


def _compositions(total: int, n: int, minimum: int) -> Iterator[Tuple[int, ...]]:
    """Every way to write ``total`` as ``n`` integers each >= ``minimum``.

    Enumerated rather than solved in closed form on purpose: the space is tiny
    (sum 64 over 3 ranks is 1953 candidates) and an exhaustive search is
    checkable by eye, which a water-filling argument on an integer grid with a
    floor is not.
    """
    slack = total - minimum * n
    if slack < 0:
        return

    def walk(remaining: int, depth: int, acc: Tuple[int, ...]):
        if depth == n - 1:
            yield acc + (remaining + minimum,)
            return
        for take in range(remaining + 1):
            yield from walk(remaining - take, depth + 1, acc + (take + minimum,))

    yield from walk(slack, 0, ())


def solve_flip_tp_vector(inputs: FlipTPVectorInputs) -> FlipTPVectorSolution:
    """Emit the TP weight vector from per-rank capability (#1017, gap G10).

    Objective: minimize the predicted TP-phase makespan ``max_r share_r /
    score_r``, the makespan of a width-parallel phase where every rank does
    work proportional to its share and none may leave the barrier early -- the
    ``langsamster-rang-taktgeber`` rule applied per barrier. Ties break toward
    the larger minimum memory headroom, so of two equally fast vectors the one
    with more room to survive a transient wins.

    Raises :class:`FlipTPVectorRefused` when nothing fits -- never a clamp,
    never a silent fallback to the incumbent.
    """
    n = len(inputs.ranks)
    scores = [r.gemm_score for r in inputs.ranks]
    score_sum = sum(scores)
    ideal = tuple(inputs.ratio_total * s / score_sum for s in scores)

    best: Optional[FlipTPVectorSolution] = None
    considered = 0
    closest_miss: Optional[Tuple[str, float]] = None

    for counts in _compositions(inputs.ratio_total, n, inputs.min_units):
        considered += 1
        shares = [c / inputs.ratio_total for c in counts]
        shard_mib = tuple(
            (inputs.replicated_bytes + inputs.shardable_bytes * s) / MIB
            for s in shares
        )
        headroom = tuple(
            inputs.ranks[i].budget_mib
            - inputs.ranks[i].fixed_overhead_mib
            - shard_mib[i]
            for i in range(n)
        )
        if min(headroom) < 0:
            worst = min(range(n), key=lambda i: headroom[i])
            shortfall = -headroom[worst]
            # The SMALLEST shortfall over infeasible candidates is the
            # informative one: it is how much memory would unlock a solution,
            # which is the number the operator can act on.
            if closest_miss is None or shortfall < closest_miss[1]:
                closest_miss = (inputs.ranks[worst].label, shortfall)
            continue

        per_rank = [shares[i] / scores[i] for i in range(n)]
        makespan = max(per_rank)
        candidate = FlipTPVectorSolution(
            counts=tuple(counts),
            realized_heads=(
                partition_units_largest_remainder(inputs.head_units, counts)
                if inputs.head_units
                else ()
            ),
            kv_heads=(
                partition_units_largest_remainder(inputs.total_kv_heads, counts)
                if inputs.total_kv_heads
                else ()
            ),
            makespan=makespan,
            bottleneck_rank=max(range(n), key=lambda i: per_rank[i]),
            shard_mib=shard_mib,
            headroom_mib=headroom,
            quantization_gap=max(
                abs(counts[i] - ideal[i]) / ideal[i] for i in range(n)
            ),
            candidates_considered=0,
            ideal=ideal,
            ratio_total=inputs.ratio_total,
        )
        if best is None or (
            candidate.makespan,
            -min(candidate.headroom_mib),
        ) < (best.makespan, -min(best.headroom_mib)):
            best = candidate

    if best is None:
        reasons = [
            f"all {considered} candidate vector(s) of sum "
            f"{inputs.ratio_total} exceed a rank's memory budget"
        ]
        if closest_miss is not None:
            reasons.append(
                f"the closest miss is {closest_miss[0]}, short by "
                f"{closest_miss[1]:.1f} MiB"
            )
        reasons.append(
            "raise that rank's --rank-gpu-memory-mib, or pass "
            "--phase-flip-tp-vector explicitly to override the solve"
        )
        raise FlipTPVectorRefused(reasons)

    return dataclasses.replace(best, candidates_considered=considered)
