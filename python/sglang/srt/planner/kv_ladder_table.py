# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""KV pressure ladder STEP TABLE, computed once from a rig/model profile
(DESIGN_201 Nachtrag-13 Ergaenzung 9, CPU phase).

Ergaenzung 9: "the step table is computed by the planner (#272 key solver) IN
ADVANCE per model/rig -- at runtime the system only FLIPS, it never plans."
This module is that computation. It is purely additive to the planner: it
constructs no new cost model, it reuses the ones that exist
(``key_solver.nesting_hull`` for the family question, the fork's parse-time
capacity predictor for the token figures) and, where an input is simply
missing, it says so with a PROVENANCE LABEL instead of inventing a number --
the same honesty discipline as the HTCCL path dispatcher's rate profiles and
the planner package's own "no estimated tok/s" rule.

WHAT IT COMPUTES, AND FROM WHAT:

* THE RUNGS. Base = the profile's first (coarsest, fastest) geometry. Then
  the enabled relief features in their canonical cheapness order
  (``RELIEF_ORDER``, a SERVICE-cost order per the #287 user directive): DCP
  token-ratio flip (#320: sum-preserving KV-vector redistribution is
  service-neutral), admission cap (costs served sessions), KV spill,
  weightless rank, session offload -- referenced by name, never implemented
  here. Then the remaining geometries, coarse to fine. Then the
  out-of-family (Nachtrag-14) steps.
  The resulting order is exactly the one ``PressureLadder`` enforces, so the
  generator cannot produce a table the runtime would reject.

* THE FAMILY CHECK. Geometry rungs must be a REFINEMENT CHAIN -- that is the
  whole reason a step is a plan flip instead of a reshard (down-set property:
  the finest cut already holds every coarser geometry in the same bytes).
  ``key_solver.nesting_hull`` answers exactly that question and is what runs
  here; a profile whose geometries do not nest is a hard error naming the
  pair, not a silently accepted ladder that would need weight movement.

* THE CAPACITY PER RUNG. From declared card totals, the per-rank budget
  fraction, the model's weight bytes and its per-token KV bytes. Provenance
  ``solver`` when those inputs exist, ``placeholder`` (value ``None``) with
  the missing input named when they do not. Relief gains are placeholders by
  default -- what a DCP-ratio nudge or a spill tier is worth in tokens is a
  Messpflicht figure, not an argument.

* THE COST PER RUNG. Placeholder by default. A relative slowdown per rung is
  precisely what the R7c measurement chain produces; this module accepts it
  (``cost_fn``) and refuses to guess it.

The whole module is CPU-computable and imports nothing heavy at module
scope (``key_solver`` and the capacity predictor are function-local, as
everywhere in this package).
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from sglang.srt.model_executor.kv_pressure_ladder import (
    DEFAULT_EXTERNAL_HYSTERESIS_ROUNDS,
    HANDOVER_BACKGROUND_MIGRATE,
    HANDOVER_SPILL_RELOAD,
    OP_PHASE_DECODE,
    OP_PHASE_PREFILL,
    OP_PHASES,
    PROVENANCE_MEASURED,
    PROVENANCE_PLACEHOLDER,
    PROVENANCE_SOLVER,
    RELIEF_FEATURES,
    RELIEF_ORDER,
    STEP_BASE,
    STEP_EXTERNAL,
    STEP_GEOMETRY,
    STEP_RELIEF,
    LadderStep,
    OperatingPoint,
    PressureLadder,
    StageOperatingGrid,
)

__all__ = [
    "CardSpec",
    "ExternalRungSpec",
    "GeometryRungSpec",
    "RankScoreProfile",
    "RigModelProfile",
    "build_ladder_table",
    "capacity_from_report",
    "check_geometry_family",
    "solve_operating_grid",
    "solver_capacity_tokens",
]

#: Default depth/fill bins of the operating grid: the two poles plus three
#: interior bins (user directive: "also in between, not only at the poles").
#: The low pole sits at 0.05, not 0.0: at exactly zero context every KV
#: vector has the same lockstep time (the depth term vanishes) and the
#: "optimum" would be a tie-break artifact -- at 5% fill the time model is
#: non-degenerate and the perf pole is genuinely solved.
DEFAULT_DEPTH_BINS: Tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 1.0)

MIB = 1 << 20

#: Provenance source labels. Every figure the table carries names WHERE it
#: came from, so a later reader never has to guess whether a number was
#: measured, solved or parked.
SOURCE_SOLVER = (
    "planner.kv_ladder_table.solver_capacity_tokens (declared profile inputs)"
)
SOURCE_REPORT = "planner.capacity.predict_capacity (parse-time predictor, estimate)"


@dataclasses.dataclass(frozen=True)
class CardSpec:
    """One physical card as the table generator needs it."""

    index: int
    name: str
    total_mib: int
    #: The rank budget on this card in MiB, when the plan pins one
    #: (``--rank-gpu-memory-mib`` style). ``None`` = derive from
    #: ``total_mib * budget_fraction``.
    budget_mib: Optional[int] = None

    def __post_init__(self):
        if self.total_mib <= 0:
            raise ValueError(
                f"card {self.index} ({self.name!r}) declares total_mib="
                f"{self.total_mib}; a card has positive VRAM."
            )
        if self.budget_mib is not None and self.budget_mib <= 0:
            raise ValueError(
                f"card {self.index} ({self.name!r}) declares budget_mib="
                f"{self.budget_mib}; a budget is positive."
            )


@dataclasses.dataclass(frozen=True)
class GeometryRungSpec:
    """One geometry of the nesting family, coarse to fine.

    ``ratio`` is the per-rank MLP/weight ratio (``--rank-tp-ratio``),
    ``gpus`` the physical card per rank. ``graphs_precaptured`` says whether
    this rung's graphs are captured at boot -- a rung without them may not be
    a flip target, and the table carries the fact rather than hiding it.
    """

    key: str
    ratio: Tuple[int, ...]
    gpus: Tuple[int, ...]
    graphs_precaptured: bool = True
    handover: str = HANDOVER_BACKGROUND_MIGRATE

    def __post_init__(self):
        if not self.ratio:
            raise ValueError(f"geometry {self.key!r} has an empty ratio")
        if len(self.ratio) != len(self.gpus):
            raise ValueError(
                f"geometry {self.key!r} has {len(self.ratio)} ranks but "
                f"{len(self.gpus)} card entries; one card per rank."
            )
        if any(int(u) <= 0 for u in self.ratio):
            raise ValueError(
                f"geometry {self.key!r} has a non-positive rank share in "
                f"{self.ratio}; every rank carries weight."
            )


@dataclasses.dataclass(frozen=True)
class ExternalRungSpec:
    """One out-of-family rung (Nachtrag-14 warm standby + handover)."""

    key: str
    handover: str = HANDOVER_SPILL_RELOAD
    graphs_precaptured: bool = True
    min_hysteresis_rounds: Optional[int] = None


@dataclasses.dataclass(frozen=True)
class RankScoreProfile:
    """Measured per-CARD rates the depth/format-aware operating grid
    consumes (user directive, #287).

    The FORMAT factors come out of the measured planner profile -- the
    per-(rank, family) GEMM scores of #324 -- and are mapped card-by-card by
    the caller; nothing here is a hardcoded arch table, and a missing rate
    is a missing rate (the grid then stays unsolved with a note), never a
    guess.

    * ``card_prefill_tflops``: effective prefill compute rate per card,
      already family-blended by the ``sum(p/r)`` arithmetic
      (``uneven_perf.effective_prefill_tflops``) so the format axis of a
      MIXED checkpoint survives the reduction to one number per card.
    * ``card_attn_tflops``: the rate of the lane the attention/GDN family
      runs on that card (``GemmScores.for_family("attn_gdn")``) -- the
      depth-proportional prefill term divides by THIS, because
      attention-over-context is that family's work.
    * ``card_membw_gbs``: memory bandwidth per card; the decode phase and
      its depth term are bandwidth-bound.
    """

    card_prefill_tflops: Dict[int, float]
    card_attn_tflops: Dict[int, float]
    card_membw_gbs: Dict[int, float]
    provenance: str = PROVENANCE_MEASURED
    source: str = ""

    def __post_init__(self):
        for label, table in (
            ("card_prefill_tflops", self.card_prefill_tflops),
            ("card_attn_tflops", self.card_attn_tflops),
            ("card_membw_gbs", self.card_membw_gbs),
        ):
            for card, value in table.items():
                if float(value) <= 0.0:
                    raise ValueError(
                        f"{label}[{card}] must be > 0, got {value}; a missing "
                        f"rate is expressed by omitting the card, not by a "
                        f"zero."
                    )


@dataclasses.dataclass(frozen=True)
class RigModelProfile:
    """Everything the table generator is allowed to look at.

    Deliberately a flat, declared record rather than a live probe: the table
    is computed ONCE per model/rig, and every input that is absent has to
    surface as a labelled placeholder rather than as a probe the generator
    triggers behind the caller's back (the planner package never probes).

    ``kv_bytes_per_token`` is the MODEL-WIDE KV cost of one context token,
    summed over all ranks -- under the fork's token-axis (uneven DCP)
    sharding a token's KV lives entirely on its owner rank, so the total
    token capacity is ``sum_r free_bytes_r / kv_bytes_per_token``.
    """

    cards: Tuple[CardSpec, ...]
    geometries: Tuple[GeometryRungSpec, ...]
    reliefs: Tuple[str, ...] = ()
    externals: Tuple[ExternalRungSpec, ...] = ()
    #: Model-wide KV bytes of one context token. None = placeholder.
    kv_bytes_per_token: Optional[int] = None
    #: Total resident weight bytes of the model (all ranks). None =
    #: placeholder.
    weight_bytes_total: Optional[int] = None
    #: Fixed per-rank overhead (CUDA context, activation reserve, graphs) in
    #: MiB, subtracted before the KV fits.
    overhead_mib_per_rank: int = 0
    #: Fraction of a card's total the rank budget takes, when a card carries
    #: no explicit ``budget_mib``.
    budget_fraction: float = 0.9
    #: Indivisible MLP unit count of the checkpoint -- the family check needs
    #: it (``PerfCostModel.mlp_units``). None = family check skipped, and the
    #: table says so.
    mlp_units: Optional[int] = None
    #: Measured per-card rates for the depth/format-aware operating grid
    #: (#324 scores, mapped per card by the caller). None = no grid is
    #: solved and every rung says so.
    rank_scores: Optional[RankScoreProfile] = None
    #: FLOPs of attention-over-context per (new token, context token) pair,
    #: derived from the model dims by the caller (~4 * full_attn_layers *
    #: q_heads * head_dim for the two context GEMMs). None = the prefill
    #: depth term cannot be sized and the grid stays unsolved.
    attn_context_flops_per_token_pair: Optional[float] = None

    def __post_init__(self):
        if not self.cards:
            raise ValueError("a rig/model profile needs at least one card")
        if not self.geometries:
            raise ValueError(
                "a rig/model profile needs at least one geometry (the base "
                "rung the server boots in)"
            )
        seen_cards = set()
        for card in self.cards:
            if card.index in seen_cards:
                raise ValueError(f"duplicate card index {card.index}")
            seen_cards.add(card.index)
        seen_keys = set()
        for geom in self.geometries:
            if geom.key in seen_keys:
                raise ValueError(f"duplicate geometry key {geom.key!r}")
            seen_keys.add(geom.key)
            for gpu in geom.gpus:
                if int(gpu) not in seen_cards:
                    raise ValueError(
                        f"geometry {geom.key!r} uses card {gpu}, which the "
                        f"profile does not declare (known: "
                        f"{sorted(seen_cards)})."
                    )
        for ext in self.externals:
            if ext.key in seen_keys:
                raise ValueError(f"duplicate rung key {ext.key!r}")
            seen_keys.add(ext.key)
        for relief in self.reliefs:
            if relief not in RELIEF_FEATURES:
                raise ValueError(
                    f"profile lists unknown relief feature {relief!r}; the "
                    f"ladder only ORDERS existing features. Known: "
                    f"{', '.join(sorted(RELIEF_FEATURES))}."
                )
        if len(set(self.reliefs)) != len(self.reliefs):
            raise ValueError(f"duplicate relief feature in {self.reliefs}")
        if not (0.0 < self.budget_fraction <= 1.0):
            raise ValueError(
                f"budget_fraction must be within (0, 1], got {self.budget_fraction}"
            )
        if self.overhead_mib_per_rank < 0:
            raise ValueError("overhead_mib_per_rank must be >= 0")
        if self.kv_bytes_per_token is not None and self.kv_bytes_per_token <= 0:
            raise ValueError("kv_bytes_per_token must be > 0 when given")
        if self.weight_bytes_total is not None and self.weight_bytes_total < 0:
            raise ValueError("weight_bytes_total must be >= 0 when given")
        if self.mlp_units is not None and self.mlp_units <= 0:
            raise ValueError("mlp_units must be > 0 when given")
        if (
            self.attn_context_flops_per_token_pair is not None
            and self.attn_context_flops_per_token_pair <= 0
        ):
            raise ValueError("attn_context_flops_per_token_pair must be > 0 when given")

    def card(self, index: int) -> CardSpec:
        for c in self.cards:
            if c.index == int(index):
                return c
        raise KeyError(f"no card with index {index}")


# ---------------------------------------------------------------------------
# 1. The family check -- why a geometry step is a flip and not a reshard
# ---------------------------------------------------------------------------


def check_geometry_family(profile: RigModelProfile) -> Tuple[bool, List[str]]:
    """Do the profile's geometries form ONE nesting family?

    Runs the planner's own set-wise check (``key_solver.nesting_hull``) over
    the geometries as lanes, with the MLP dimension as the probe. A step
    inside the family is a plan flip precisely because the finer cut's shards
    are exact unions of the coarser one's -- if that fails, the "step" would
    be a weight reshard and the whole Ergaenzung-9 premise is gone for that
    pair.

    Returns ``(ok, notes)``. Without ``mlp_units`` the check cannot run and
    ``notes`` says so instead of returning a confident ``True``.
    """
    if profile.mlp_units is None:
        return (
            True,
            [
                "family check SKIPPED: profile carries no mlp_units "
                "(PerfCostModel.mlp_units); the nesting property is asserted "
                "by the caller, not verified here"
            ],
        )
    if len(profile.geometries) < 2:
        return (True, ["family check trivial: a single geometry"])

    from sglang.srt.planner.key_solver import HullProbe, LaneKey, nesting_hull

    lanes = [
        LaneKey(
            key=geom.key,
            ratio=tuple(int(u) for u in geom.ratio),
            gpus=tuple(int(g) for g in geom.gpus),
        )
        for geom in profile.geometries
    ]
    probes = [HullProbe(what="mlp", units=int(profile.mlp_units), family="mlp")]
    hull = nesting_hull(lanes, probes)
    return (bool(hull.ok), list(hull.failures))


# ---------------------------------------------------------------------------
# 2. The capacity per rung
# ---------------------------------------------------------------------------


def solver_capacity_tokens(
    profile: RigModelProfile, geom: GeometryRungSpec
) -> Tuple[Optional[int], str, str]:
    """KV token capacity of one geometry from the declared profile inputs.

    Per rank: ``budget - weight_share - overhead`` is what is left for KV;
    summed over the ranks and divided by the model-wide per-token KV bytes.
    The weight share follows the rank ratio, which is what the ratio MEANS
    (``PerfCostModel``'s affine weight model: ``W_r = A_r + m * u_r``; the
    rank-independent part ``A_r`` is folded into the overhead here, and that
    simplification is stated rather than hidden -- it is why the provenance
    is ``solver`` and never ``measured``).

    Returns ``(tokens_or_None, provenance, source)``.
    """
    if profile.kv_bytes_per_token is None:
        return (
            None,
            PROVENANCE_PLACEHOLDER,
            "placeholder: profile carries no kv_bytes_per_token",
        )
    if profile.weight_bytes_total is None:
        return (
            None,
            PROVENANCE_PLACEHOLDER,
            "placeholder: profile carries no weight_bytes_total",
        )
    ratio_sum = float(sum(int(u) for u in geom.ratio))
    free_total = 0.0
    for rank, gpu in enumerate(geom.gpus):
        card = profile.card(int(gpu))
        if card.budget_mib is not None:
            budget_bytes = float(card.budget_mib) * MIB
        else:
            budget_bytes = float(card.total_mib) * MIB * profile.budget_fraction
        weight_bytes = profile.weight_bytes_total * (int(geom.ratio[rank]) / ratio_sum)
        overhead_bytes = float(profile.overhead_mib_per_rank) * MIB
        free_total += budget_bytes - weight_bytes - overhead_bytes
    if free_total <= 0.0:
        return (0, PROVENANCE_SOLVER, SOURCE_SOLVER)
    return (
        int(free_total // profile.kv_bytes_per_token),
        PROVENANCE_SOLVER,
        SOURCE_SOLVER,
    )


def _free_kv_bytes_per_rank(
    profile: RigModelProfile, geom: GeometryRungSpec
) -> Optional[List[float]]:
    """Per-rank bytes left for KV under ``geom`` (same arithmetic as
    ``solver_capacity_tokens``, kept in one shape so the two cannot
    drift), or ``None`` when the weight bytes are unknown."""
    if profile.weight_bytes_total is None:
        return None
    ratio_sum = float(sum(int(u) for u in geom.ratio))
    out: List[float] = []
    for rank, gpu in enumerate(geom.gpus):
        card = profile.card(int(gpu))
        if card.budget_mib is not None:
            budget_bytes = float(card.budget_mib) * MIB
        else:
            budget_bytes = float(card.total_mib) * MIB * profile.budget_fraction
        weight_bytes = profile.weight_bytes_total * (int(geom.ratio[rank]) / ratio_sum)
        out.append(
            budget_bytes - weight_bytes - float(profile.overhead_mib_per_rank) * MIB
        )
    return out


def _kv_vector_candidates(n: int, top: int) -> List[Tuple[int, ...]]:
    """Deterministic small-integer KV-vector candidate set, gcd-reduced and
    sorted. Exhaustive for the group sizes this fork serves (n <= 4); larger
    groups fall back to the even vector so the solver never explodes -- the
    fallback is named in the grid notes by the caller."""
    import itertools
    import math as _math

    if top**n <= 1296 * 6:
        seen = set()
        for vec in itertools.product(range(1, top + 1), repeat=n):
            g = _math.gcd(*vec)
            seen.add(tuple(v // g for v in vec))
        return sorted(seen)
    return [tuple([1] * n)]


def solve_operating_grid(
    profile: RigModelProfile,
    geom: GeometryRungSpec,
    *,
    depth_bins: Sequence[float] = DEFAULT_DEPTH_BINS,
    kv_candidate_top: int = 6,
) -> Tuple[Optional[StageOperatingGrid], List[str]]:
    """The depth/format-aware (phase x depth) KV-vector optima of one rung
    (user directive, #287).

    Per phase (prefill compute-bound, decode bandwidth-bound) and per
    depth/fill bin ``d`` the solver picks the KV token vector minimizing the
    lockstep per-rank time

    * prefill: ``t_r = Wbytes_r / rate_r + d*T * c_attn * share_r / attn_r``
    * decode:  ``t_r = Wbytes_r / membw_r + d*T * kv_bytes * share_r / membw_r``

    subject to the FIT constraint ``d*T * kv_bytes * share_r <= free_r``
    (a vector that cannot hold the fill it is solved for is not an optimum
    at that fill). ``T`` is the rung's solver capacity, ``share_r`` the
    candidate's normalized ownership share. At ``d = 0`` the constraint is
    void and the time-balance (perf) pole wins; as ``d`` grows the feasible
    set shrinks towards the capacity pole -- the grid IS the trajectory
    between the #296 extrema, with the #320 depth measurement as the
    licence that the sum-preserving redistribution itself is service-neutral.
    The layer split stays the rung's own ratio: Stage 1 never reshards
    weights, so the split axis moves only by flipping rungs.

    Format awareness enters exclusively through ``profile.rank_scores``
    (#324 per-(rank, family) scores mapped per card); the weight-time proxy
    ``Wbytes_r / rate`` treats bytes as the mass unit, a stated
    simplification of the ``solver`` provenance. Returns ``(grid, notes)``;
    an unsolvable grid is ``(None, notes)`` with the missing input named,
    never a guessed table.
    """
    notes: List[str] = []
    scores = profile.rank_scores
    if scores is None:
        return (None, ["no rank_scores in profile -- operating grid not solved"])
    if profile.kv_bytes_per_token is None:
        return (None, ["no kv_bytes_per_token -- operating grid not solved"])
    free = _free_kv_bytes_per_rank(profile, geom)
    if free is None:
        return (None, ["no weight_bytes_total -- operating grid not solved"])
    if min(free) <= 0.0:
        return (
            None,
            [
                f"rank {free.index(min(free))} has no free KV bytes under "
                f"geometry {geom.key!r} -- operating grid not solved"
            ],
        )
    if profile.attn_context_flops_per_token_pair is None:
        return (
            None,
            [
                "no attn_context_flops_per_token_pair -- the prefill depth "
                "term cannot be sized, operating grid not solved"
            ],
        )
    total_tokens, prov, _src = solver_capacity_tokens(profile, geom)
    if not total_tokens:
        return (
            None,
            [f"rung capacity unavailable ({prov}) -- operating grid not solved"],
        )
    n = len(geom.gpus)
    rates: Dict[str, List[float]] = {}
    for label, table in (
        ("prefill", scores.card_prefill_tflops),
        ("attn", scores.card_attn_tflops),
        ("membw", scores.card_membw_gbs),
    ):
        vec = []
        for gpu in geom.gpus:
            if int(gpu) not in table:
                return (
                    None,
                    [
                        f"card {int(gpu)} has no {label} rate in rank_scores "
                        f"-- operating grid not solved (a missing rate is "
                        f"named, not guessed)"
                    ],
                )
            vec.append(float(table[int(gpu)]))
        rates[label] = vec
    bins = sorted(set(float(b) for b in depth_bins))
    if not bins or bins[0] < 0.0 or bins[-1] > 1.0:
        raise ValueError(f"depth_bins must lie within [0, 1], got {depth_bins}")

    ratio_sum = float(sum(int(u) for u in geom.ratio))
    # Narrowing for the type checker: _free_kv_bytes_per_rank above already
    # returned None (and this function with it) when the weight bytes are
    # unknown.
    assert profile.weight_bytes_total is not None
    w_bytes = [
        profile.weight_bytes_total * (int(geom.ratio[r]) / ratio_sum) for r in range(n)
    ]
    kv_bytes = float(profile.kv_bytes_per_token)
    c_attn = float(profile.attn_context_flops_per_token_pair)
    candidates = _kv_vector_candidates(n, kv_candidate_top)
    if len(candidates) == 1:
        notes.append(
            f"kv candidate set degenerated to the even vector (group size "
            f"{n} too large for the exhaustive grid)"
        )

    points: List[OperatingPoint] = []
    for phase in OP_PHASES:
        for d in bins:
            ctx_tokens = d * float(total_tokens)
            best: Optional[Tuple[float, Tuple[int, ...]]] = None
            for cand in candidates:
                share_sum = float(sum(cand))
                shares = [c / share_sum for c in cand]
                if any(ctx_tokens * kv_bytes * shares[r] > free[r] for r in range(n)):
                    continue
                t_max = 0.0
                for r in range(n):
                    if phase == OP_PHASE_PREFILL:
                        t = w_bytes[r] / (rates["prefill"][r] * 1e12) + (
                            ctx_tokens * c_attn * shares[r] / (rates["attn"][r] * 1e12)
                        )
                    else:
                        t = (w_bytes[r] + ctx_tokens * kv_bytes * shares[r]) / (
                            rates["membw"][r] * 1e9
                        )
                    t_max = max(t_max, t)
                if best is None or (t_max, cand) < best:
                    best = (t_max, cand)
            if best is not None:
                source = (
                    f"solver: argmin lockstep {phase} time over "
                    f"{len(candidates)} kv candidates at fill {d:g} "
                    f"(fit-constrained; #324 card rates)"
                )
                vector = best[1]
            else:
                # No candidate holds this fill: name the capacity pole
                # (argmax of the fit) rather than inventing a fit.
                cap_best = max(
                    candidates,
                    key=lambda cand: (
                        min(
                            free[r] * sum(cand) / (kv_bytes * cand[r]) for r in range(n)
                        ),
                        tuple(-c for c in cand),
                    ),
                )
                vector = cap_best
                source = (
                    f"solver: NO candidate holds fill {d:g} under geometry "
                    f"{geom.key!r}; naming the capacity pole instead"
                )
                notes.append(source)
            points.append(
                OperatingPoint(
                    phase=phase,
                    depth_fraction=d,
                    layer_split=tuple(int(u) for u in geom.ratio),
                    kv_vector=tuple(int(v) for v in vector),
                    provenance=PROVENANCE_SOLVER,
                    source=source,
                )
            )
    return (StageOperatingGrid(points), notes)


def capacity_from_report(report) -> Tuple[Optional[int], str, str]:
    """Adapter for the fork's parse-time capacity predictor
    (``planner.capacity.CapacityReport``).

    Provenance is ``solver``, never ``measured``: the predictor's own
    docstring says every absolute token number is an estimate and only
    candidate-over-candidate ratios are exact. An infeasible report yields
    ``None`` rather than a negative capacity -- a rung that does not fit is
    not a rung.
    """
    if report is None:
        return (None, PROVENANCE_PLACEHOLDER, "placeholder: no capacity report")
    if not getattr(report, "feasible", False):
        return (
            None,
            PROVENANCE_PLACEHOLDER,
            "placeholder: capacity report says the plan does not fit",
        )
    tokens = getattr(report, "max_context_tokens", None)
    if tokens is None or tokens < 0:
        return (
            None,
            PROVENANCE_PLACEHOLDER,
            "placeholder: capacity report carries no usable token figure",
        )
    return (int(tokens), PROVENANCE_SOLVER, SOURCE_REPORT)


# ---------------------------------------------------------------------------
# 3. The table
# ---------------------------------------------------------------------------

CapacityFn = Callable[
    [RigModelProfile, GeometryRungSpec], Tuple[Optional[int], str, str]
]
ReliefGainFn = Callable[[str, Optional[int]], Tuple[Optional[int], str, str]]
CostFn = Callable[[str, str], Tuple[Optional[float], str, str]]


def _default_relief_gain(
    feature: str, base_tokens: Optional[int]
) -> Tuple[Optional[int], str, str]:
    """What a relief feature is worth in tokens: unknown until measured.

    Deliberately NOT a heuristic. "KV spill buys you X %" is exactly the kind
    of number that has to come off the R7c chain; a guessed one would enter
    the ladder's monotonicity check as if it were real and could silently
    reorder the rungs.
    """
    return (
        None,
        PROVENANCE_PLACEHOLDER,
        f"placeholder: token gain of relief {feature!r} "
        f"({RELIEF_FEATURES[feature]}) is a Messpflicht figure",
    )


def _default_cost(name: str, step_type: str) -> Tuple[Optional[float], str, str]:
    """Relative round-time cost of a rung: unknown until measured."""
    return (
        None,
        PROVENANCE_PLACEHOLDER,
        f"placeholder: relative cost of {step_type} rung {name!r} comes from "
        f"the ms/round measurement chain",
    )


def _merge_provenance(*labels: str) -> str:
    """Weakest label wins: one placeholder makes the whole rung a
    placeholder, one solver figure keeps it at solver. Never upgrades."""
    if PROVENANCE_PLACEHOLDER in labels:
        return PROVENANCE_PLACEHOLDER
    if PROVENANCE_SOLVER in labels:
        return PROVENANCE_SOLVER
    return PROVENANCE_MEASURED


def build_ladder_table(
    profile: RigModelProfile,
    *,
    capacity_fn: Optional[CapacityFn] = None,
    relief_gain_fn: Optional[ReliefGainFn] = None,
    cost_fn: Optional[CostFn] = None,
    external_min_hysteresis_rounds: int = DEFAULT_EXTERNAL_HYSTERESIS_ROUNDS,
    require_family: bool = True,
) -> PressureLadder:
    """Compute the pressure ladder's step table for one rig/model profile.

    ``capacity_fn`` / ``relief_gain_fn`` / ``cost_fn`` are injectable so the
    GPU/measurement phase can feed real figures without touching this
    function, and so hermetic tests can exercise the ordering logic with a
    fake profile.

    ``require_family=True`` (the default) refuses a profile whose geometries
    do not form a refinement chain: without that property a "step" would be a
    weight reshard, which is the one thing Ergaenzung 9 rules out. Pass
    ``False`` only to inspect a broken profile.
    """
    capacity_fn = capacity_fn or solver_capacity_tokens
    relief_gain_fn = relief_gain_fn or _default_relief_gain
    cost_fn = cost_fn or _default_cost

    ok, notes = check_geometry_family(profile)
    if not ok and require_family:
        raise ValueError(
            "the profile's geometries are not one nesting family, so a step "
            "between them would be a weight RESHARD and not a plan flip "
            "(DESIGN_201 Erg. 9 point 1). Failures: " + "; ".join(notes)
        )
    family_note = "; ".join(notes) if notes else "family verified by nesting_hull"

    steps: List[LadderStep] = []

    # --- rung 0: the base geometry, the performance-optimal state ----------
    base_geom = profile.geometries[0]
    base_tokens, base_prov, base_src = capacity_fn(profile, base_geom)
    base_cost, base_cost_prov, base_cost_src = cost_fn(base_geom.key, STEP_BASE)
    # Depth/format-aware operating grid (user directive): solved per rung
    # from the #324 card rates when the profile carries them; an unsolved
    # grid is None WITH its reason in the step source, never a guess.
    base_grid, base_grid_notes = solve_operating_grid(profile, base_geom)
    base_grid_note = (
        "grid: " + "; ".join(base_grid_notes)
        if base_grid_notes
        else ("grid: solved" if base_grid is not None else "grid: none")
    )
    if base_cost is None:
        # The base rung IS the reference of the cost axis; that is a
        # definition, not an estimate, so it is the one cost figure this
        # module may fill in.
        base_cost, base_cost_prov, base_cost_src = (
            1.0,
            PROVENANCE_SOLVER,
            "definition: the base rung is the cost reference (1.0)",
        )
    steps.append(
        LadderStep(
            name=base_geom.key,
            step_type=STEP_BASE,
            geometry_key=base_geom.key,
            expected_kv_tokens=base_tokens,
            expected_cost_factor=base_cost,
            graphs_precaptured=True,
            provenance=_merge_provenance(base_prov, base_cost_prov),
            source=f"{base_src} | {base_cost_src} | {family_note} | {base_grid_note}",
            operating_grid=base_grid,
        )
    )

    # --- relief rungs, cheapest first (canonical order) ---------------------
    ordered_reliefs = [f for f in RELIEF_ORDER if f in profile.reliefs]
    running_tokens = base_tokens
    for feature in ordered_reliefs:
        gain, gain_prov, gain_src = relief_gain_fn(feature, running_tokens)
        tokens: Optional[int] = None
        if gain is not None and running_tokens is not None:
            tokens = int(running_tokens) + int(gain)
            running_tokens = tokens
        cost, cost_prov, cost_src = cost_fn(feature, STEP_RELIEF)
        # The dcp_ratio rung IS the KV-vector flip: its actuator picks a new
        # ownership vector at the UNCHANGED base layer split, so the base
        # geometry's grid is exactly its (phase x depth) target table. The
        # other relief features move no vectors and carry no grid.
        relief_grid = base_grid if feature == "dcp_ratio" else None
        relief_grid_note = f" | {base_grid_note}" if feature == "dcp_ratio" else ""
        steps.append(
            LadderStep(
                name=feature,
                step_type=STEP_RELIEF,
                relief_feature=feature,
                expected_kv_tokens=tokens,
                expected_cost_factor=cost,
                graphs_precaptured=True,
                provenance=_merge_provenance(gain_prov, cost_prov),
                source=f"{gain_src} | {cost_src}{relief_grid_note}",
                operating_grid=relief_grid,
            )
        )

    # --- geometry rungs, coarse to fine ------------------------------------
    for geom in profile.geometries[1:]:
        tokens, prov, src = capacity_fn(profile, geom)
        cost, cost_prov, cost_src = cost_fn(geom.key, STEP_GEOMETRY)
        geom_grid, geom_grid_notes = solve_operating_grid(profile, geom)
        geom_grid_note = (
            "grid: " + "; ".join(geom_grid_notes)
            if geom_grid_notes
            else ("grid: solved" if geom_grid is not None else "grid: none")
        )
        steps.append(
            LadderStep(
                name=geom.key,
                step_type=STEP_GEOMETRY,
                geometry_key=geom.key,
                expected_kv_tokens=tokens,
                expected_cost_factor=cost,
                graphs_precaptured=bool(geom.graphs_precaptured),
                handover=geom.handover,
                provenance=_merge_provenance(prov, cost_prov),
                source=f"{src} | {cost_src} | {geom_grid_note}",
                operating_grid=geom_grid,
            )
        )

    # --- external rungs (Nachtrag-14 warm standby) --------------------------
    for ext in profile.externals:
        cost, cost_prov, cost_src = cost_fn(ext.key, STEP_EXTERNAL)
        steps.append(
            LadderStep(
                name=ext.key,
                step_type=STEP_EXTERNAL,
                expected_kv_tokens=None,
                expected_cost_factor=cost,
                graphs_precaptured=bool(ext.graphs_precaptured),
                handover=ext.handover,
                min_hysteresis_rounds=(
                    ext.min_hysteresis_rounds
                    if ext.min_hysteresis_rounds is not None
                    else external_min_hysteresis_rounds
                ),
                provenance=_merge_provenance(PROVENANCE_PLACEHOLDER, cost_prov),
                source=(
                    "placeholder: out-of-family capacity is the Nachtrag-14 "
                    f"warm-standby path, not sized here | {cost_src}"
                ),
            )
        )

    return PressureLadder(
        steps, external_min_hysteresis_rounds=external_min_hysteresis_rounds
    )


def describe_table(ladder: PressureLadder) -> List[Dict[str, object]]:
    """The table as plain records (CLI / dashboard / issue text)."""
    return ladder.describe()


def profile_from_cards(
    cards: Sequence[CardSpec],
    geometries: Sequence[GeometryRungSpec],
    **kwargs,
) -> RigModelProfile:
    """Convenience constructor keeping the tuple conversions in one place."""
    return RigModelProfile(
        cards=tuple(cards),
        geometries=tuple(geometries),
        reliefs=tuple(kwargs.pop("reliefs", ())),
        externals=tuple(kwargs.pop("externals", ())),
        **kwargs,
    )
