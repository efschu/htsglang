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
"""key_solver — the distribution key per goal, COMPUTED instead of guessed (#272).

The rig profile already holds every input a distribution key needs: per-card
compute and bandwidth rates, the card-to-card pair matrix, host transfer
rates, the fixed-post ledger and the model geometry. What was missing was the
tool that turns those inputs into the key. This module is that tool.

It is deliberately NOT a search over pre-canned working points. Every goal
below reduces to the SAME small convex problem, which has a closed-form
solution; the optimizer is a bisection over one scalar, and the derivation
fits in this docstring.


1. THE VARIABLE
---------------
A key is an allocation of the model's MLP mass over the ranks. The fork
realizes it on an integer grid of ``U`` indivisible MLP units
(``PerfCostModel.mlp_units``: the quant group count for a dense checkpoint —
136 for Qwen3.6-27B-FP8 at group size 128 — or the expert count for a MoE
one; the fork's ``partition_units`` keeps them 16-element aligned, which is
why a 16er grid is the natural granularity). The solver's variable is

    u = (u_0, ..., u_{R-1}),   u_r >= 0,   sum_r u_r = U

i.e. a point of the scaled simplex. Ratios (``--rank-mlp-ratio``) are the
user-facing form of the same thing; the solver reports both.

Everything else in the plan — attention heads, GDN units, vocab, the KV token
vector — follows the BASE plan and is held fixed, because those axes are flat
under decode (M22) and their movement is a different feature. They enter as
constants ``A_r`` below.


2. THE PHASE COST MODEL
-----------------------
Per-rank resident weight bytes are AFFINE in the variable:

    W_r(u) = A_r + m * u_r,      m = mlp_family_bytes / U

``A_r`` is the sum of every family that does not follow the MLP key on rank r
(attention, GDN, vocab, the replicated draft rung); ``m`` is exact — it is one
unit's share of the MLP family. This affinity is what makes the whole problem
solvable in closed form, and it is a property of the partition, not an
approximation.

DECODE (bs=1, weight-bandwidth bound, lockstep over ranks)

    t_dec,r(u) = W_r(u) / B_r
    T_w(u)     = max_r t_dec,r(u)                      [the lockstep max]
    T_dec(u)   = T_w(u) + T_coll,dec + T_host,dec

``B_r`` is the EFFECTIVE decode bandwidth, not the probed peak: it is
``PerfCostModel.effective_decode_bw`` (the probe's decode-shaped GEMV rate
raised to the fitted residual exponent, with a named fallback to the streaming
peak). Taking peak for achieved is what let the 6,1,1 vector through at a
measured 16.5 % decode cost, so the same divisor is reused here rather than
re-derived.

``T_coll,dec`` is new here, and comes from the PAIR MATRIX rather than a
scalar link figure. Two all-reduces per layer, ring schedule:

    T_coll,dec = 2 * L * [ (R-1) * lat + (2(R-1)/R) * H * b / BW_min ]

with ``lat`` the worst ORDERED-pair latency among the ranks that talk,
``BW_min`` the narrowest ordered-pair bandwidth, ``H`` the hidden size and
``b`` the activation dtype width. On the reference rig (host-staged pairs,
19.5-22.4 us, 4.32-6.91 GB/s) this lands at a few ms per round and is
split-INVARIANT: it does not move with ``u``, so it changes the absolute
prediction and never the ranking. When no pair matrix is on disk the term is
``absent`` — it is not invented.

``T_host,dec`` is the host tier (expert offload / KV spill): offloaded bytes
per round over the measured pinned H2D rate of that rank's card. Absent when
no host probe exists or nothing is offloaded.

Reported decode STEP delta uses the calibrated form of the existing model, so
that the four fitted scalars keep their meaning:

    step_ratio(u) = f_dec + (1 - f_dec) * T_w(u) / T_w(base),   f_dec = 0.35

PREFILL (compute bound + collective + a split-invariant remainder)

    t_pre,r(u) = 2 * P_r(u) / (F_r * eta),   P_r(u) = D_r + p * u_r
    T_pre(u)   = max_r t_pre,r(u) + T_ar + inv

``P_r`` is the sharded parameter count (the flops proxy), ``F_r`` the probed
GEMM rate of that rank's card, ``eta`` the achieved-GEMM efficiency. ``inv``
is the eager launch/non-GEMM remainder, charged at
``prefill_invariant_fraction`` of the BASE plan's sharded time — constant
across candidates, so it deflates reported gains without touching the
ranking. Same construction as ``PerfCostModel.prefill_time_model``, whose
constants were fitted against the #216 campaign.

CAPACITY

    free_r(u) = budget_r - W_r(u) - mamba_r - overhead - role_post_r
    P^kv_r(u) = free_r(u) / cell
    ctx(u)    = min( sum_r P^kv_r(u), 64 * min_r P^kv_r(u) )

``cell`` is the KV bytes per token and counts FULL-ATTENTION layers only —
see §5. Note the identity that decides how the maxkv goal behaves:

    sum_r P^kv_r(u) = ( sum_r budget_r - sum_r A_r - m*U - ... ) / cell

is INDEPENDENT of ``u``, because ``sum_r u_r = U`` is fixed. The MLP key
does not create KV, it only moves it between cards. The key therefore
matters for maxkv exactly through the second term, ``64 * min_r P^kv_r``:
maximizing usable context means maximizing the WEAKEST rank's KV capacity.
That is stated as a result, not assumed.


3. THE OPTIMIZER
----------------
Every goal in ``GOALS`` reduces to one problem:

    minimize_u  max_r ( a_r + b_r * u_r )
    subject to  sum_r u_r = U,  lo_r <= u_r <= hi_r,  b_r > 0

(a max-min goal becomes a min-max by negation, see the table in ``_terms``.)
The solution is water-filling. For a level ``T`` define

    u_r(T) = clip( (T - a_r) / b_r , lo_r , hi_r )
    S(T)   = sum_r u_r(T)

``S`` is continuous and non-decreasing in ``T``, so there is a unique level
``T*`` with ``S(T*) = U`` (up to the flat pieces, which are handled by
splitting the surplus among the ranks that are strictly interior). Bisection
on ``T`` finds it to machine precision in ~60 iterations. That is the closed
form; there is no search over vectors.

Goal -> (a_r, b_r):

    dec       a_r = A_r / B_r                 b_r = m / B_r
    enc       a_r = 2 * D_r / (F_r * eta)     b_r = 2 * p / (F_r * eta)
    maxkv     a_r = -free_r(0)                b_r = m         [max-min]
    sessions  as maxkv, then capped by the u-invariant state-slot bound

The continuous optimum is projected onto the integer grid by largest
remainder (which preserves ``sum u_r = U`` exactly) followed by a bounded
repair pass: single-unit moves are accepted only while the objective strictly
improves, at most ``R * U`` of them and in practice a handful. The repair
exists because the grid is coarse (136 units for FP8 dense, ~68 for a GGUF
K-quant) and the rounded point can sit on the wrong side of a knee.

ROLES are a small discrete search on top, and they are literally the bounds
``lo_r``/``hi_r`` of the same problem — which is why the whole 0-100 %
continuum is representable without a special case:

    shard      lo=0,   hi=U     the ordinary TP rank
    kv_donor   lo=0,   hi=0     0 % of the weights: the weightless KV lane
                                (#115/#127/#131/#133/#143), a built randpunkt
    replica    a separate topology, not a simplex point: the card holds the
               WHOLE model and takes part opportunistically. Priced by the
               ledger and evaluated with its own (collective-free) formula;
               offered only when the model fits on that one card.

With R <= 6 the role vectors are enumerated exhaustively (3^R <= 729) and
each is solved by the same water-fill, so a mixed rig can genuinely be part
shard, part donor, part replica.


4. COMBINED GOALS
-----------------
Two forms, both bounded, neither an N-dimensional optimization theatre:

(a) CONSTRAINT form — "maximize A subject to B >= threshold". The candidate
    set is the union of the two single-goal optima, the segment between them
    (41 steps, each rounded and sum-repaired), and a coarse simplex grid.
    Every point is scored on ALL goals; the answer is the feasible argmax
    of A.

(b) PARETO front for a goal PAIR. The non-dominated subset of the same
    candidate set, thinned to 3-5 points: both endpoints plus the KNEE, the
    point of maximum perpendicular distance to the chord joining the
    endpoints in normalized objective space. Every returned candidate carries
    the predicted value of EVERY goal, including the sacrificed ones.

The candidate set is a bounded search, not a proof of global optimality, and
this module says so in its output (``caveats``).


5. HYBRID GEOMETRY, AND WHY IT IS ITS OWN SECTION
-------------------------------------------------
``num_hidden_layers`` sizes a hybrid checkpoint WRONG. Qwen3.6-27B has 64
layers of which only 16 are full attention (interval 4); the other 48 are
linear-attention/GDN and carry a per-SEQUENCE recurrent state instead of a
growing KV. The KV cell must therefore be counted over full-attention layers
only, and a layer WINDOW (the PP case) must be counted by intersecting the
window with the full-attention positions — #201 slice 2 found a 14/10 layer
split whose full-attention content was 3/3, i.e. equal, where a proportional
reading would have said 14:10. ``full_attention_layers_in_window`` is the
helper, and the regression store pins it.


6. VALIDATION
-------------
``REGRESSION_ANCHORS`` holds the measured points this model must reproduce,
with their source. ``check_regressions`` re-derives each one and reports
predicted vs measured, so the model error is visible rather than asserted.
The tolerances are stated per anchor and justified there — a model that
cannot reproduce a measurement it was built from is wrong, and this is where
that shows.

7. MORE THAN ONE INSTANCE
-------------------------
A rig can carry several serving sources at once — an extra solo server beside
the main group, a PD prefill lane, a replicated drafter, a full replica, a
prefill satellite on a second rig. All of them obey ONE mechanic, and none of
them gets its own branch:

    aggregate_X = sum_i X_i     IF the coexistence bracket closes
    bracket: for every physical GPU g,  sum_i resident_i(g) <= total(g) - reserve(g)

``resident_i(g)`` is the #260 co-residence computation — weights, state pool,
the modelled per-rank overhead and a minimum KV pool, plus the fixed process
post for every instance beyond the first on that card. When the bracket does
not close the aggregate is NOT reported: it names the overflowing GPU and the
MiB. An aggregate over instances that cannot coexist is precisely the number
that makes "just add a second server" look free.

The absolute prefill rate this needs is built from first principles
(``KeyCostModel.prefill_seconds``): lockstep GEMM at the probed rates plus
the pair-matrix collective, and deliberately WITHOUT the relative model's
invariant fraction, which was fitted to absorb the very collective that is
now explicit. Measured error on three arms, un-fitted: -19 %, +15 %, +11 %
(see ``ADDITIVE_ANCHOR``).


8. VALIDATION LOOP
------------------
Every prediction carries an ``estimate`` label and a ``remeasure`` hook
naming the instrument that would turn it into a ``measured`` one
(``split_probe`` for split/throughput, ``card_probe`` for rates).


9. N LANES: THE HULL TREE AND THE PRIORITY CLASSES
--------------------------------------------------
The multi-group runtime (#274) puts SEVERAL lanes over one resident weight
set, so two things that were pairwise become set-wise.

NESTING. ``nesting_bounds`` bounds the lane being solved against ONE resident
lane; ``nesting_bounds_over`` intersects the ceilings of all of them, which
is the same statement unrolled. But a lane set can pass every such check and
still be unbuildable, because nesting is not transitive across SIBLINGS: on
the rig's own ``[6,1,1]`` group the lanes ``[6,2]`` and ``[7,1]`` each nest in
the group at almost every unit count and in each other at none, since one
cuts after rank 0's units and the other after rank 1's. ``nesting_hull``
is the set-wise question — does a refinement forest over all of them exist —
and it is what a caller must ask before believing that N lanes share bytes.

WHERE THIS AND ``distributed/dual_group.py`` DIFFER, deliberately and
measurably. ``dual_group`` is given a segmentation and asks whether THAT
grouping nests; the hull, by default, asks whether ANY grouping does. Pin a
lane's ``shared_segments`` and the two agree at every unit count (67 of 497
for ``[6,1,1] -> [6,2]``; DESIGN #121 records 65, which is the same set
minus the 2 counts too small to split over 3 ranks at all). Leave it free and
the hull additionally accepts units 3-6, where the grouping ``[0,1],[2]``
nests although ``[0],[1,2]`` does not. Both answers are right for their
question; a caller that has already fixed a segmentation — Slice B installs
one — must pin it, and ``dual_group`` remains the authority for the runtime.

PRIORITY. PRIO-Nachtrag 5 in the N-lane form of Nachtrag 8d: lanes carry an
ordered ``priority_class`` and ``coresident_budget_plan`` serves the classes
on a shared card in ascending order, each in full before the next sees
anything. A class reached with nothing left is NAMED, never given an invented
share. With one class the award is the even split the two-lane mapping
already made, which is why no configuration that does not use the feature
moves.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sglang.srt.planner import cost_model
from sglang.srt.planner.bench_factors import ABSENT, ESTIMATE, MEASURED, Remeasure

__all__ = [
    "GOALS",
    "GOAL_LABELS",
    "ROLES",
    "REGRESSION_ANCHORS",
    "RoleSpec",
    "RigRates",
    "Candidate",
    "SolverAnswer",
    "KeyCostModel",
    "build_cost_model",
    "rates_from_probe",
    "water_fill",
    "project_to_grid",
    "nesting_bounds",
    "nesting_bounds_over",
    "nesting_hull",
    "partition_cuts",
    "rank_map_over_cards",
    "OuterLane",
    "LaneKey",
    "HullProbe",
    "HullTree",
    "UnitBound",
    "full_attention_layers",
    "full_attention_layers_in_window",
    "solve",
    "check_regressions",
    "check_additive_regression",
    "InstanceSpec",
    "InstanceEstimate",
    "estimate_instance",
    "coexistence",
    "coresident_budgets",
    "coresident_budget_plan",
    "CoresidentPlan",
    "aggregate",
    "LaneTarget",
    "LaneSolution",
    "solve_lanes",
    "gemm_dtype_for_checkpoint",
    "ADDITIVE_ANCHOR",
    "COLLECTIVE_EFFICIENCY",
    "NOISE_FLOOR_PCT",
]


# ---------------------------------------------------------------------------
# Goals
# ---------------------------------------------------------------------------

#: The v1 goals. ``ttft_at_n`` and ``session_max`` are deliberately NOT here:
#: they are interfaces (see ``GOAL_INTERFACES``), reserved so the shape does
#: not have to change when their falsifiers come back, and reported as
#: ``absent`` rather than guessed.
GOALS: Tuple[str, ...] = ("maxkv", "sessions", "dec", "enc")

GOAL_LABELS: Dict[str, str] = {
    "maxkv": "context that fits",
    "sessions": "parallel sessions at the target context",
    "dec": "decode rate",
    "enc": "prefill rate",
}

#: Higher is better, per goal (decode is reported as a RATE, so all four are).
GOAL_HIGHER_IS_BETTER: Dict[str, bool] = {g: True for g in GOALS}

GOAL_UNITS: Dict[str, str] = {
    "maxkv": "tokens",
    "sessions": "sessions",
    "dec": "tok/s",
    "enc": "tok/s",
}

#: Reserved goal names with no implementation yet. They exist in the output
#: so a caller can bind the UI once; each says what it is waiting on.
GOAL_INTERFACES: Dict[str, str] = {
    "ttft_at_n": (
        "reserved interface: TTFT-optimal placement parameterized by the "
        "expected number of concurrent requests. Needs the load model plus "
        "the tipping-point machinery; the #212 idle/loaded pair (0.257 s vs "
        "0.604 s) is the anchor it will be fitted against. Not implemented — "
        "no number is produced here."
    ),
    "session_max": (
        "reserved interface: maximum real parallel sessions via replicated "
        "GDN shards. Blocked on one prior question — whether the scheduler can "
        "bind a session to a replicated state pool (affinity). Not "
        "implemented — no number is produced here."
    ),
}

#: Whether one #82-style co-location process fits is a ledger question, not a
#: solver one; the solver only prices the post. Reference values from the
#: 28.5 GiB lesson (CUDA context + graphs + activations per extra process).
FIXED_PROCESS_POST_MIB: float = 1536.0


#: Ring efficiency for the pair-matrix collective term. A ring all-reduce
#: moves 2(R-1)/R of the payload per rank. One definition for every planner
#: (#348b): the shard planners price the same schedule.
_ring_factor = cost_model.ring_factor


#: Achieved fraction of the probed GEMM peak in a real prefill step. Same
#: value the existing prefill model uses; imported lazily to keep this module
#: importable without the heavy chain.
def _gemm_efficiency() -> float:
    from sglang.srt import uneven_perf

    return float(uneven_perf._PREDICT_GEMM_EFF)


#: Efficiency of a GROUP collective relative to what the pair matrix
#: measured. Shipped at 1.0, i.e. UNFITTED — the pair matrix is taken at face
#: value.
#:
#: The direction of its error is known and stated rather than tuned away: the
#: pair matrix measures one ORDERED pair at a time, while a real all-reduce
#: has every rank on the wire simultaneously and, on a rig without peer
#: access, through one shared pinned-host path. So 1.0 UNDER-states what a
#: group collective costs, and the absolute prefill model inherits that.
#:
#: Measured against three arms with this default and nothing else changed
#: (``check_regressions``): Q3_K_M solo 5090 -19 %, Q3_K_M TP=3 +15 %,
#: FP8 TP=3 +11 %. The residual is one-sided in exactly the predicted way
#: (the two collective-bearing arms are predicted too FAST). Refit recipe:
#: measure one prefill arm with a collective and one without on the same
#: checkpoint, then
#:     eta_coll = (T_group_measured - T_compute_predicted) / T_coll_predicted
#: A refit belongs in this constant, not in the code.
COLLECTIVE_EFFICIENCY: float = 1.0


def gemm_dtype_for_checkpoint(model_path: str) -> str:
    """Which probed GEMM rate this checkpoint's prefill actually reaches.

    ``"fp8"`` only for a checkpoint whose matmuls run on the fp8 tensor path;
    ``"bf16"`` for everything else, GGUF K-quants included — a K-quant is
    dequantized into a bf16/fp16 GEMM, so reading a 5090's fp8 rate for it
    over-states that rank by 2.4x. Unreadable or unfamiliar configs answer
    ``"bf16"``: the conservative side is the one that does not invent
    throughput.
    """
    import json
    import os

    path = str(model_path or "")
    if path.lower().endswith(".gguf"):
        return "bf16"
    cfg_path = os.path.join(path, "config.json") if os.path.isdir(path) else path
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return "bf16"
    if os.path.isdir(path) and any(
        n.lower().endswith(".gguf") for n in os.listdir(path)
    ):
        return "bf16"
    quant = (cfg.get("quantization_config") or {}) if isinstance(cfg, dict) else {}
    method = str(quant.get("quant_method") or "").lower()
    fmt = str(quant.get("fmt") or quant.get("format") or "").lower()
    if "fp8" in method or "fp8" in fmt or "e4m3" in fmt:
        return "fp8"
    return "bf16"


#: Benchmark noise floor of this project's harness, in percent. Anything the
#: solver reports below it is reported WITH the caveat, never as a win.
NOISE_FLOOR_PCT: float = 4.2


# ---------------------------------------------------------------------------
# Roles — the 0 % .. 100 % continuum, expressed as bounds
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RoleSpec:
    """One role a card can play, priced.

    ``weight_lo``/``weight_hi`` are the share of the TOTAL MLP units this
    rank may take, so the ordinary shard rank is simply ``[0, 1]`` and the
    two continuum ends are bounds rather than special cases. ``topology`` is
    ``"simplex"`` for roles that are points of the partition and ``"solo"``
    for the replica, which is a different process rather than a share.
    """

    key: str
    label: str
    weight_lo: float
    weight_hi: float
    fixed_post_mib: float
    topology: str
    note: str


ROLES: Tuple[RoleSpec, ...] = (
    RoleSpec(
        key="shard",
        label="tensor-parallel shard",
        weight_lo=0.0,
        weight_hi=1.0,
        fixed_post_mib=0.0,
        topology="simplex",
        note="the ordinary rank; its MLP share is what the solver sets",
    ),
    RoleSpec(
        key="kv_donor",
        label="KV donor (0 % of the weights)",
        weight_lo=0.0,
        weight_hi=0.0,
        fixed_post_mib=0.0,
        topology="simplex",
        note=(
            "the built randpunkt of the continuum: the weightless KV lane "
            "(#115/#127/#131/#133/#143). The card keeps attention and its KV "
            "pool and holds no MLP mass, so its whole budget funds KV. Costs "
            "the lock-step: the donor still takes part in every round."
        ),
    ),
    RoleSpec(
        key="replica",
        label="full replica (100 % of the weights)",
        weight_lo=1.0,
        weight_hi=1.0,
        fixed_post_mib=FIXED_PROCESS_POST_MIB,
        topology="solo",
        note=(
            "the other randpunkt: the card holds the WHOLE model and serves "
            "on its own — no collective tick at all. A real operating point "
            "for small checkpoints on 20 GiB cards; sorted out by the ledger, "
            "not by a rule, when the model does not fit."
        ),
    ),
)

_ROLE_BY_KEY: Dict[str, RoleSpec] = {r.key: r for r in ROLES}

#: Cap on the exhaustive role enumeration. 3 roles ^ 6 ranks = 729 solves,
#: each a bisection — milliseconds. Above that the search is skipped and only
#: the all-shard assignment is solved (reported in ``caveats``).
MAX_ROLE_SEARCH_RANKS: int = 6


# ---------------------------------------------------------------------------
# Hybrid geometry (§5)
# ---------------------------------------------------------------------------


def full_attention_layers(layer_types: Optional[Sequence[str]]) -> Optional[int]:
    """How many layers actually carry a growing KV cache.

    ``None`` when the checkpoint declares no ``layer_types`` (a pure-attention
    model, where the caller's ``num_hidden_layers`` is already correct).
    """
    if not layer_types:
        return None
    return sum(1 for t in layer_types if t == "full_attention")


def full_attention_layers_in_window(
    layer_types: Optional[Sequence[str]], start: int, end: int
) -> Optional[int]:
    """Full-attention layers inside the half-open window ``[start, end)``.

    The PP / layer-split case. #201 slice 2: a 14/10 split of a period-4
    hybrid holds 3 and 3 full-attention layers, not 14 and 10 — so a stage's
    KV pool is sized by this intersection and never by the layer COUNT.
    Sizing a hybrid stage from ``num_hidden_layers`` alone is the documented
    way to get it wrong in both directions at once.
    """
    if not layer_types:
        return None
    lo = max(0, int(start))
    hi = min(len(layer_types), int(end))
    if hi <= lo:
        return 0
    return sum(1 for t in layer_types[lo:hi] if t == "full_attention")


# ---------------------------------------------------------------------------
# Rates
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RigRates:
    """The measured inputs, in RANK order (not card order).

    Anything the profile does not hold is ``None`` and lands in ``absent``.
    Nothing here is ever defaulted to a plausible number: a missing pair
    matrix means the collective term is not predicted, it does not mean the
    collective is free.
    """

    #: Probed streaming bandwidth per rank's card, GB/s. ``None`` for a card
    #: the probe never scored -- never ``0.0`` (#359): a zero survives every
    #: arithmetic it enters and reads downstream as an extremely slow but
    #: valid card. Use :meth:`require_membw_gbs` to consume it.
    membw_gbs: List[Optional[float]]
    #: Probed decode-shaped GEMV rate per rank's card, GB/s (None = absent).
    gemv_gbs: Optional[List[float]]
    #: Probed GEMM rate per rank's card, TFLOP/s, RESOLVED for the compute
    #: format of the checkpoint being planned (see ``resolve_gemm_format``).
    #: Same absence contract as ``membw_gbs``.
    gemm_tflops: List[Optional[float]]
    #: The two raw rates, kept so the resolution can be redone once the
    #: checkpoint is known. A card without an fp8 tensor path has ``None``.
    gemm_bf16_tflops: List[Optional[float]] = dataclasses.field(default_factory=list)
    gemm_fp8_tflops: List[Optional[float]] = dataclasses.field(default_factory=list)
    #: Narrowest ordered-pair bandwidth among the participating cards, GB/s.
    link_bw_gbs: Optional[float] = None
    #: Worst ordered-pair latency among the participating cards, us.
    link_latency_us: Optional[float] = None
    #: Pinned host-to-device rate per rank's card, GB/s.
    h2d_gbs: Optional[List[float]] = None
    #: Pinned device-to-host rate per rank's card, GB/s.
    d2h_gbs: Optional[List[float]] = None
    #: Where each block of numbers came from.
    basis: Dict[str, str] = dataclasses.field(default_factory=dict)
    #: Names of the inputs that are missing (each one costs a prediction).
    absent: List[str] = dataclasses.field(default_factory=list)
    #: Where two artifacts measured the same fact and disagreed. Not an
    #: absence -- both numbers exist -- and not something to average away, so
    #: it travels as its own list for the caller to surface (#359).
    divergences: List[str] = dataclasses.field(default_factory=list)
    #: The rank's card entry from the probe, widened with whatever GEMM lanes
    #: the hardware profile holds for the same UUID. This is what the #324
    #: per-(rank, family) resolution reads; kept so the resolution can be
    #: redone once the checkpoint is known.
    rank_entries: Tuple[Dict, ...] = ()
    #: Human-readable rank label per position, for absence messages.
    rank_keys: Tuple[str, ...] = ()
    #: The checkpoint format ``gemm_tflops`` was resolved for (#298a key).
    gemm_format: str = "bf16"
    #: GEMM lane label per rank, as #324 resolved it.
    gemm_lanes: Tuple[str, ...] = ()
    #: family -> per-rank rate, only for a #324 family whose format diverges
    #: from the checkpoint-wide one. Empty on every single-scheme checkpoint.
    gemm_family_tflops: Dict[str, List[float]] = dataclasses.field(default_factory=dict)
    #: Loud wrong-lane fallbacks from the resolution, verbatim from #324.
    gemm_warnings: Tuple[str, ...] = ()

    @property
    def ranks(self) -> int:
        return len(self.membw_gbs)

    def require_membw_gbs(self) -> List[float]:
        """The streaming rates, or :class:`cost_model.AbsentRate` naming the
        first rank that has none."""
        return self._require(self.membw_gbs, "streaming bandwidth")

    def require_gemm_tflops(self) -> List[float]:
        """The GEMM rates, or :class:`cost_model.AbsentRate` naming the first
        rank that has none."""
        return self._require(self.gemm_tflops, "GEMM rate")

    def _require(self, values: Sequence[Optional[float]], what: str) -> List[float]:
        out: List[float] = []
        for r, value in enumerate(values):
            if value is None:
                key = self.rank_keys[r] if r < len(self.rank_keys) else f"rank {r}"
                raise cost_model.AbsentRate(
                    f"{what} for {key}",
                    "the card probe never scored this card, so every term "
                    "that divides by this rate is unpredictable. Re-run the "
                    "card probe (POST /api/card_probe).",
                )
            out.append(float(value))
        return out

    def resolve_gemm_format(
        self,
        fmt: str,
        family_formats: Optional[Dict[str, str]] = None,
    ) -> RigRates:
        """Re-score every rank on the lane the CHECKPOINT's format dispatches
        to, through the #324 per-(rank, family) resolution (#359).

        This replaces :meth:`resolve_gemm_dtype`, whose binary ``"fp8"`` /
        ``"bf16"`` answer was right for exactly two formats. An int8 W8A8, an
        NVFP4 or a W4A16 checkpoint answered ``"bf16"`` and was then priced at
        the dense rate -- a measured number on the wrong lane, which is worse
        than an absent one because it looks trustworthy. The reference rig's
        int8 lane is 2.9x its own best fp8 lane on the 3080s and 1.2x on the
        5090; scoring all three dense compresses a 3.68:1 rank ratio to
        3.54:1 and the ladder solves a different vector for it.

        The resolution is not re-derived here: ``rank_gemm_family_scores`` is
        the resolver, wrapped for provenance by ``cost_model``. A format with
        no lane table, or a lane the artifacts never measured, falls back to
        the dense rate WITH the warning #324 already emits -- the same number
        the binary classifier produced, now labelled.
        """
        if not self.rank_entries:
            return self
        rates = cost_model.compute_rates_from_entries(
            self.rank_entries,
            self.rank_keys or tuple(f"rank {r}" for r in range(self.ranks)),
            fmt=fmt,
            family_formats=family_formats,
        )
        picked = [rate.or_none() for rate in rates.rates]
        families = {
            family: [r.or_none() for r in vec] for family, vec in rates.families.items()
        }
        basis = dict(self.basis)
        basis["gemm"] = f"rig artifacts, {fmt} GEMM lane resolution (#324)"
        return dataclasses.replace(
            self,
            gemm_tflops=picked,
            gemm_format=fmt,
            gemm_lanes=tuple(rate.label for rate in rates.rates),
            gemm_family_tflops={k: v for k, v in families.items() if all(v)},
            gemm_warnings=tuple(rates.warnings),
            basis=basis,
        )

    def resolve_gemm_dtype(self, dtype: str) -> RigRates:
        """Re-pick the per-rank GEMM rate for the checkpoint's compute dtype.

        This is not cosmetic. Taking the fp8 rate for a card that has one
        while planning a GGUF K-quant reads the 5090 at 566.88 instead of
        231.97 TFLOP/s and makes the whole prefill model wrong by 2.4x on
        that rank; a card without an fp8 path falls back to bf16 either way,
        so the error only appears on the mixed rig — i.e. exactly where the
        planner is used.
        """
        if dtype not in ("fp8", "bf16"):
            raise ValueError(f"gemm dtype must be 'fp8' or 'bf16', got {dtype!r}")
        if not self.gemm_bf16_tflops:
            return self
        picked: List[Optional[float]] = []
        for r in range(self.ranks):
            bf = self.gemm_bf16_tflops[r]
            fp = self.gemm_fp8_tflops[r] if self.gemm_fp8_tflops else None
            rate = fp if (dtype == "fp8" and fp) else bf
            # A card the probe never scored keeps its named absence (#359);
            # it used to become a 0.0 that every divisor downstream accepted.
            picked.append(float(rate) if rate else None)
        basis = dict(self.basis)
        basis["gemm"] = (
            f"card probe, {dtype} GEMM rate (bf16 where the card has no fp8 path)"
        )
        return dataclasses.replace(self, gemm_tflops=picked, basis=basis)


def rates_from_probe(
    probe: Optional[dict],
    rank_gpu_id: Sequence[int],
    *,
    cuda_order: bool = True,
    hardware_profile: Optional[dict] = None,
) -> RigRates:
    """Assemble :class:`RigRates` from a ``card_probe`` artifact.

    ``probe`` is the JSON of ``~/.cache/sglang/card_probe-*.json`` (cards +
    ordered pair matrix). ``rank_gpu_id`` maps rank -> device index in the
    SAME order the probe's ``cuda_index`` uses when ``cuda_order`` is true —
    that is the ``--rank-gpu-id`` space, and mixing it with the NVML order is
    the device-order trap the fork has walked into before.

    ``hardware_profile`` is the rig profile (``profile["gpus"][uuid]``), whose
    v3 ``gemm_lanes`` map holds the quantized GEMM lanes the card probe cannot
    measure -- fp8 Marlin, fp8 W8A16, int8 native. Passing it is what lets the
    #324 lane resolution price an int8 / nvfp4 / W4A16 checkpoint on the lane
    it will actually dispatch to instead of on the dense bf16 probe (#359).
    Without it the solver keeps exactly the two lanes the card probe measures
    and any other format falls back to dense WITH a warning.

    Raises ``ValueError`` only for a structurally impossible mapping (a rank
    naming a card the probe does not know); everything merely missing becomes
    ``absent``.
    """
    absent: List[str] = []
    divergences: List[str] = []
    basis: Dict[str, str] = {}
    if not probe or not probe.get("cards"):
        raise ValueError(
            "no card probe on disk: the solver needs per-card rates. Run the "
            "card probe (POST /api/card_probe, ~5 s over three cards)."
        )
    cards = list(probe["cards"])
    by_index: Dict[int, dict] = {}
    for pos, c in enumerate(cards):
        idx = c.get("cuda_index") if cuda_order else c.get("index")
        by_index[int(idx if idx is not None else pos)] = c

    membw: List[Optional[float]] = []
    gemv: List[float] = []
    tflops: List[Optional[float]] = []
    bf16: List[Optional[float]] = []
    fp8: List[Optional[float]] = []
    h2d: List[float] = []
    d2h: List[float] = []
    gemv_ok, h2d_ok = True, True
    rank_cards: List[dict] = []
    rank_keys: List[str] = []
    for r, gpu in enumerate(rank_gpu_id):
        c = by_index.get(int(gpu))
        if c is None:
            raise ValueError(
                f"rank {r} is mapped to GPU {gpu}, which the card probe does "
                f"not know (it holds {sorted(by_index)}). Re-probe, or fix "
                "--rank-gpu-id."
            )
        rank_cards.append(c)
        rank_keys.append(f"rank {r} (GPU {gpu}, {c.get('name') or 'unnamed'})")
        streaming = c.get("membw_read_gbs") or c.get("membw_gbs")
        membw.append(float(streaming) if streaming else None)
        g = c.get("membw_gemv_gbs")
        if g:
            gemv.append(float(g))
        else:
            gemv_ok = False
        # Both rates are kept; which one is used depends on the CHECKPOINT
        # (resolve_gemm_format). The default here is bf16 — the path every
        # card has — so a caller that forgets to resolve under-states a 5090
        # on an fp8 model rather than over-stating it on a GGUF one.
        b16 = c.get("gemm_bf16_tflops") or c.get("gemm_tflops")
        f8 = c.get("gemm_fp8_tflops")
        bf16.append(float(b16) if b16 else None)
        fp8.append(float(f8) if f8 else None)
        dense = b16 or f8
        tflops.append(float(dense) if dense else None)
        if c.get("h2d_gbs") and c.get("d2h_gbs"):
            h2d.append(float(c["h2d_gbs"]))
            d2h.append(float(c["d2h_gbs"]))
        else:
            h2d_ok = False

    # The class docstring promises nothing here is defaulted to a plausible
    # number. #348b named the two holes the loops above left; #359 stops
    # filling them: a card the probe never scored now carries ``None`` all the
    # way to the consumer, which raises ``AbsentRate`` naming it, instead of a
    # 0.0 that gets clamped to 1e-9 and read as an extremely slow valid card.
    # On a complete probe -- every reference config -- both lists come back
    # empty and nothing moves.
    membw_holes = [
        rate.source
        for rate in cost_model.memory_rates_from_entries(rank_cards, rank_keys, "membw")
        if rate.is_absent
    ]
    if membw_holes:
        absent.append(
            f"per-card streaming bandwidth on {len(membw_holes)} rank(s) — "
            "the decode term cannot be predicted there: " + "; ".join(membw_holes[:3])
        )
    gemm_holes = [key for key, rate in zip(rank_keys, tflops) if rate is None]
    if gemm_holes:
        absent.append(
            f"per-card GEMM rate on {len(gemm_holes)} rank(s) — every prefill "
            "term there is unpredictable: " + "; ".join(gemm_holes[:3])
        )

    basis["membw"] = "card probe, streaming read rate"
    if gemv_ok:
        basis["gemv"] = "card probe, decode-shaped GEMV rate"
    else:
        gemv = []
        absent.append(
            "decode GEMV rate (profile predates it) — the decode roofline "
            "falls back to the streaming peak with its own exponent"
        )
    basis["gemm"] = "card probe, bf16 GEMM rate (unresolved default)"

    # Hop cost comes from the shared cost library (#348b), not from a reader
    # private to this module: the video and diffusion planners price the same
    # wires, and a second parser is a second set of assumptions. The library
    # also rejects a same-card row, which the hand-rolled loop here trusted the
    # probe never to emit.
    link_bw: Optional[float] = None
    link_lat: Optional[float] = None
    used = {int(g) for g in rank_gpu_id}
    if len(used) > 1:
        uuid_of_index: Dict[int, str] = {}
        for pos, c in enumerate(cards):
            idx = c.get("cuda_index") if cuda_order else c.get("index")
            if c.get("uuid"):
                uuid_of_index[int(idx if idx is not None else pos)] = str(c["uuid"])
        card_keys = [str(g) for g in sorted(used)]
        # ONE boundary for both on-disk shapes (#359): the ordered card probe
        # wins a contested pair, the unordered NCCL link map fills what it
        # does not cover, and a disagreement is reported rather than silently
        # decided by whichever reader ran.
        matrix, link_divergences = cost_model.load_pair_matrix(
            card_keys,
            card_probe=probe,
            hardware_profile=hardware_profile,
            uuid_of_key={str(g): uuid_of_index.get(int(g), "") for g in sorted(used)},
            read_disk=False,
        )
        divergences.extend(link_divergences)
        bw_rate = matrix.narrowest_bandwidth_gbs()
        lat_rate = matrix.worst_latency_us()
        # Both or neither: the collective formula needs a payload term AND a
        # latency term, and half of it is not a prediction.
        if not bw_rate.is_absent and not lat_rate.is_absent:
            link_bw = bw_rate.require("narrowest ordered pair")
            link_lat = lat_rate.require("worst ordered-pair latency")
            basis["link"] = bw_rate.source
        for line in matrix.rejected:
            absent.append(f"pair matrix row rejected — {line}")
    if link_bw is None and len(used) > 1:
        absent.append(
            "card-to-card pair matrix — the collective term of both phases "
            "cannot be predicted and is reported absent"
        )

    if h2d_ok and h2d:
        basis["host"] = "card probe, pinned host transfer rates"
    else:
        h2d, d2h = [], []
        absent.append(
            "pinned host H2D/D2H rates — any host-tier term (expert offload, "
            "KV spill) is reported absent"
        )

    # The lane-bearing entries the #324 resolution reads: the probe's own two
    # lanes, widened with whatever the hardware profile measured for the same
    # UUID. Assembled once here so ``resolve_gemm_format`` is a pure re-score.
    lane_entries, lane_notes = cost_model.gemm_lane_entries(
        rank_cards, hardware_profile=hardware_profile
    )
    divergences.extend(lane_notes)

    return RigRates(
        membw_gbs=membw,
        gemv_gbs=gemv or None,
        gemm_tflops=tflops,
        gemm_bf16_tflops=bf16,
        gemm_fp8_tflops=fp8,
        link_bw_gbs=link_bw,
        link_latency_us=link_lat,
        h2d_gbs=h2d or None,
        d2h_gbs=d2h or None,
        basis=basis,
        absent=absent,
        divergences=divergences,
        rank_entries=tuple(lane_entries),
        rank_keys=tuple(rank_keys),
    )


# ---------------------------------------------------------------------------
# The water-fill primitive (§3)
# ---------------------------------------------------------------------------


def water_fill(
    a: Sequence[float],
    b: Sequence[float],
    total: float,
    lo: Sequence[float],
    hi: Sequence[float],
    *,
    iterations: int = 200,
) -> List[float]:
    """Minimize ``max_r (a_r + b_r * u_r)`` on the box-bounded simplex.

    Returns the continuous optimum ``u`` with ``sum u = total`` and
    ``lo_r <= u_r <= hi_r``. See the module docstring §3 for the derivation:
    ``S(T) = sum_r clip((T - a_r)/b_r, lo_r, hi_r)`` is continuous and
    non-decreasing, so the level ``T*`` with ``S(T*) = total`` is unique and
    bisection finds it.

    Two degenerate inputs are handled explicitly rather than by accident:
    ``sum(lo) > total`` or ``sum(hi) < total`` has no feasible point (a role
    assignment that cannot hold the model), and is reported by raising; a rank
    with ``b_r <= 0`` would make the level meaningless and is rejected the
    same way.
    """
    n = len(a)
    if not (n == len(b) == len(lo) == len(hi)):
        raise ValueError("water_fill: a, b, lo, hi must have the same length")
    if any(x <= 0 for x in b):
        raise ValueError("water_fill: every slope b_r must be strictly positive")
    slo, shi = sum(lo), sum(hi)
    if total < slo - 1e-9 or total > shi + 1e-9:
        raise ValueError(
            f"water_fill: {total:g} units do not fit the bounds "
            f"[{slo:g}, {shi:g}] — this role assignment cannot hold the model"
        )
    total = min(max(total, slo), shi)

    def level_sum(t: float) -> float:
        s = 0.0
        for i in range(n):
            s += min(max((t - a[i]) / b[i], lo[i]), hi[i])
        return s

    t_lo = min(a[i] + b[i] * lo[i] for i in range(n))
    t_hi = max(a[i] + b[i] * hi[i] for i in range(n))
    if level_sum(t_lo) >= total:
        return [float(x) for x in lo]
    for _ in range(iterations):
        mid = 0.5 * (t_lo + t_hi)
        if level_sum(mid) < total:
            t_lo = mid
        else:
            t_hi = mid
        if t_hi - t_lo <= 1e-12 * max(1.0, abs(t_hi)):
            break
    t = 0.5 * (t_lo + t_hi)
    u = [min(max((t - a[i]) / b[i], lo[i]), hi[i]) for i in range(n)]
    # A flat piece of S can leave a residue; hand it to the ranks that are
    # strictly interior, proportionally to 1/b (they all sit at the same
    # level, so this keeps the max where it is).
    residue = total - sum(u)
    if abs(residue) > 1e-9:
        interior = [i for i in range(n) if lo[i] + 1e-12 < u[i] < hi[i] - 1e-12]
        if not interior:
            interior = [i for i in range(n) if hi[i] > lo[i]]
        w = sum(1.0 / b[i] for i in interior) or 1.0
        for i in interior:
            u[i] = min(max(u[i] + residue * (1.0 / b[i]) / w, lo[i]), hi[i])
    return u


def project_to_grid(
    u: Sequence[float],
    total: int,
    lo: Sequence[int],
    hi: Sequence[int],
) -> List[int]:
    """Largest-remainder projection of a continuous allocation onto integers.

    Preserves ``sum == total`` exactly and respects the bounds. Deterministic:
    ties break on the lower rank index, so the same inputs always give the
    same key (a solver whose answer depends on dict order is not usable as a
    launch-flag generator).
    """
    n = len(u)
    floors = [min(max(int(math.floor(x)), lo[i]), hi[i]) for i, x in enumerate(u)]
    short = total - sum(floors)
    if short > 0:
        order = sorted(range(n), key=lambda i: (-(u[i] - floors[i]), i))
        k = 0
        while short > 0 and k < n * 4:
            i = order[k % n]
            if floors[i] < hi[i]:
                floors[i] += 1
                short -= 1
            k += 1
    elif short < 0:
        order = sorted(range(n), key=lambda i: (u[i] - floors[i], i))
        k = 0
        while short < 0 and k < n * 4:
            i = order[k % n]
            if floors[i] > lo[i]:
                floors[i] -= 1
                short += 1
            k += 1
    return floors


def _repair(
    vec: Sequence[int],
    objective,
    lo: Sequence[int],
    hi: Sequence[int],
    *,
    max_moves: int = 64,
) -> List[int]:
    """Bounded local repair: move single units while the objective strictly
    improves. Rounding onto a coarse grid can land on the wrong side of a
    knee; this walks back off it without turning the solver into a search."""
    cur = list(vec)
    best = objective(cur)
    for _ in range(max_moves):
        moved = False
        for i in range(len(cur)):
            for j in range(len(cur)):
                if i == j or cur[i] <= lo[i] or cur[j] >= hi[j]:
                    continue
                cand = list(cur)
                cand[i] -= 1
                cand[j] += 1
                val = objective(cand)
                if val < best - 1e-12:
                    cur, best, moved = cand, val, True
                    break
            if moved:
                break
        if not moved:
            break
    return cur


# ---------------------------------------------------------------------------
# The cost model
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class KeyCostModel:
    """Per-phase analytic costs for one (model, rig, budget) point.

    Wraps the fork's ``PerfCostModel`` for everything already calibrated
    against measurements — family bytes, the effective decode bandwidth, the
    capacity budget, the two fitted invariant fractions — and adds what the
    solver needs on top: the affine decomposition of the per-rank byte model
    (so the optimum is a closed form rather than a sweep), the pair-matrix
    collective term, and the host-tier term.
    """

    perf: Any
    rates: RigRates
    base_plan: List[int]
    #: Total MLP units on the grid the runtime actually partitions.
    units: int
    #: Bytes of one MLP unit.
    unit_bytes: float
    #: Parameters of one MLP unit.
    unit_params: float
    #: Per-rank bytes of every family that does NOT follow the MLP key.
    fixed_bytes: List[float]
    #: Per-rank params of every sharded family that does NOT follow the key.
    fixed_params: List[float]

    # -- affine pieces ----------------------------------------------------

    def weight_bytes(self, units: Sequence[int]) -> List[float]:
        return [
            self.fixed_bytes[r] + self.unit_bytes * units[r] for r in range(len(units))
        ]

    def decode_weight_time(self, units: Sequence[int]) -> float:
        bw = self.perf.effective_decode_bw(
            self.rates.require_membw_gbs(), self.rates.gemv_gbs
        )
        w = self.weight_bytes(units)
        return max(w[r] / bw[r] for r in range(len(units)))

    def prefill_compute_time(self, units: Sequence[int]) -> float:
        eta = _gemm_efficiency()
        gemm = self.rates.require_gemm_tflops()
        out = 0.0
        for r in range(len(units)):
            p = self.fixed_params[r] + self.unit_params * units[r]
            out = max(out, 2.0 * p / (gemm[r] * 1e12 * eta))
        return out

    # -- collective + host terms (pair matrix / host probe) ----------------

    def collective_decode_s(self) -> Optional[float]:
        """Per decode round, all ranks. ``None`` when no pair matrix exists.

        Two all-reduces per layer of ``hidden * dtype`` bytes at bs=1, ring
        schedule. Split-invariant by construction, so it shifts the absolute
        prediction and never the ranking.
        """
        r = self.rates.ranks
        if r < 2 or self.rates.link_bw_gbs is None:
            return None if r >= 2 else 0.0
        per_ar = cost_model.allreduce_seconds(
            float(self.perf.hidden) * 2.0,  # bf16 activations
            r,
            self.rates.link_bw_gbs,
            float(self.rates.link_latency_us or 0.0),
            efficiency=COLLECTIVE_EFFICIENCY,
        )
        return 2.0 * self.perf.n_layers * per_ar

    def collective_prefill_s(self, tokens: int) -> Optional[float]:
        """Same schedule, ``tokens`` activations wide."""
        one = self.collective_decode_s()
        if one is None:
            return None
        r = self.rates.ranks
        if r < 2:
            return 0.0
        per_ar = cost_model.allreduce_seconds(
            float(self.perf.hidden) * 2.0 * max(int(tokens), 1),
            r,
            float(self.rates.link_bw_gbs or 1.0),
            float(self.rates.link_latency_us or 0.0),
            efficiency=COLLECTIVE_EFFICIENCY,
        )
        return 2.0 * self.perf.n_layers * per_ar

    def prefill_seconds(self, units: Sequence[int], tokens: int) -> Optional[float]:
        """ABSOLUTE prefill wall time for ``tokens`` tokens, from first
        principles. ``None`` when the collective term is absent (no pair
        matrix) and there is more than one rank — an absolute number that
        silently omits the dominant term is worse than no number.

            T = tokens * max_r [ 2 * P_r / (F_r * eta) ]  +  T_ar(tokens)

        Deliberately WITHOUT the ``prefill_invariant_fraction`` of the
        relative model. That constant was fitted to absorb everything the old
        model did not represent — and on this rig the collective is most of
        it (8.1 s of an 18.4 s group prefill of 20k tokens). Charging both
        would count the same seconds twice. The relative model keeps its
        invariant and its calibration untouched; this is a second, separately
        checkable quantity, and ``check_regressions`` reports its error on
        three measured arms.
        """
        coll = self.collective_prefill_s(tokens)
        if coll is None:
            return None
        eta = _gemm_efficiency()
        gemm = self.rates.require_gemm_tflops()
        comp = 0.0
        for r in range(len(units)):
            p = self.fixed_params[r] + self.unit_params * units[r]
            rate = gemm[r]
            if rate <= 0:
                return None
            comp = max(comp, 2.0 * p * max(int(tokens), 1) / (rate * 1e12 * eta))
        return comp + coll

    def host_tier_decode_s(self, units: Sequence[int]) -> Optional[float]:
        """Host-served weight bytes per decode round over the pinned H2D
        rate. ``None`` when no host probe exists; ``0.0`` when nothing is
        offloadable (a dense checkpoint)."""
        offl = self.perf.per_rank_offloadable_weight_bytes(list(units))
        if not any(offl):
            return 0.0
        if not self.rates.h2d_gbs:
            return None
        return max(offl[r] / (self.rates.h2d_gbs[r] * 1e9) for r in range(len(units)))

    # -- capacity ---------------------------------------------------------

    def free_bytes_at_zero(self) -> List[float]:
        """Per-rank bytes fundable for KV if a rank held NO MLP units.

        The intercept of the affine free-byte model, and the one quantity the
        max-min goals need that ``predict_capacity`` cannot be asked for
        directly (an all-zero vector is not a partition). Recovered from the
        base plan by linearity rather than re-derived, so the budget posts
        (reserve, state pool, overhead, the solo-host extra) keep living in
        exactly one place::

            free_r(0) = free_r(base) + unit_bytes * base_units_r

        This is exact, not an approximation: every post except the weights is
        independent of the MLP vector, and the weight term is linear in it.
        """
        base_units = self.perf.mlp_unit_partition(list(self.base_plan))
        cell = float(self.perf.kv_cell_bytes)
        p_base = self.perf.predict_capacity(list(self.base_plan))["p"]
        return [
            p_base[r] * cell + self.unit_bytes * base_units[r]
            for r in range(len(base_units))
        ]

    def capacity(self, units: Sequence[int], role_post_mib: Sequence[float]) -> dict:
        """Per-rank KV token capacity and usable context for a unit vector.

        Delegates the budget arithmetic to ``PerfCostModel.predict_capacity``
        (so the reserve/mamba/overhead posts stay in one place) and then
        subtracts any per-rank role post on top.
        """
        pred = self.perf.predict_capacity(list(units))
        p = list(pred["p"])
        cell = float(self.perf.kv_cell_bytes)
        for r in range(len(p)):
            if role_post_mib[r]:
                p[r] -= role_post_mib[r] * 2**20 / cell
        from sglang.srt import uneven_perf as _up

        feasible = all(x >= _up._PREDICT_MIN_RANK_TOKENS for x in p)
        ctx = min(sum(p), _up._PREDICT_TOKEN_UNITS * min(p)) if feasible else 0.0
        return {
            "p": p,
            "ctx": ctx,
            "feasible": feasible,
            "min_p": min(p) if p else 0.0,
            "sum_p": sum(p),
        }


def build_cost_model(
    plan_inputs,
    base_plan: Sequence[int],
    budgets_mib: Sequence[int],
    rates: RigRates,
    *,
    measured: Optional[List[dict]] = None,
    measured_mlp_vector: Optional[List[int]] = None,
) -> KeyCostModel:
    """Assemble a :class:`KeyCostModel`.

    The affine decomposition is read out of the family model rather than
    re-derived: ``fixed_bytes[r]`` is the whole-rank byte total at
    ``units[r] = 0``, and ``unit_bytes`` is the difference one unit makes.
    Both are exact for the family model, which is why the optimizer's
    closed form is exact for it too.
    """
    from sglang.srt import uneven_perf
    from sglang.srt.uneven_perf import PerfCostModel

    # #359: the checkpoint's own compute format, resolved per (rank, family)
    # through #324, replaces the binary fp8/bf16 classifier this module used to
    # run. ``gemm_dtype_for_checkpoint`` stays exported for the callers that
    # only want the dtype question answered; it is no longer what prices a
    # plan, because it cannot name int8, nvfp4, W4A16 or a mixed checkpoint.
    fmt, _desc, family_formats = uneven_perf.checkpoint_compute_format_families(
        plan_inputs.model_path
    )
    rates = rates.resolve_gemm_format(fmt, family_formats)
    perf = PerfCostModel(
        plan_inputs,
        list(base_plan),
        list(budgets_mib),
        measured=measured,
        measured_mlp_vector=measured_mlp_vector,
    )
    n = perf.tp_size
    units = int(perf.mlp_units)

    mlp_fam = perf.families.get("mlp")
    mlp_bytes = float(mlp_fam.bytes) if mlp_fam is not None else 0.0
    mlp_params = float(mlp_fam.params) if mlp_fam is not None else 0.0
    # The draft's MLP rung follows the same key; fold it into the unit so the
    # affine model matches ``streamed_bytes`` exactly.
    draft_mlp = perf.families.get("draft_mlp")
    if draft_mlp is not None and draft_mlp.params > 0:
        mlp_bytes += float(draft_mlp.bytes)
        mlp_params += float(draft_mlp.params)
    unit_bytes = mlp_bytes / units if units else 0.0
    unit_params = mlp_params / units if units else 0.0

    fixed_bytes = [0.0] * n
    fixed_params = [0.0] * n
    for name, fam in perf.families.items():
        if fam.params <= 0 or fam.shard == "mlp":
            continue
        fracs = perf._shard_fractions(fam.shard, list(base_plan))
        for r in range(n):
            fixed_bytes[r] += fam.bytes * fracs[r]
            if fam.shard != "replicated":
                fixed_params[r] += fam.params * fracs[r]

    return KeyCostModel(
        perf=perf,
        rates=rates,
        base_plan=list(base_plan),
        units=units,
        unit_bytes=unit_bytes,
        unit_params=unit_params,
        fixed_bytes=fixed_bytes,
        fixed_params=fixed_params,
    )


# ---------------------------------------------------------------------------
# Goal terms (§3, the (a_r, b_r) table)
# ---------------------------------------------------------------------------


def _terms(
    goal: str, model: KeyCostModel, role_post_mib: Sequence[float]
) -> Tuple[List[float], List[float]]:
    """``(a, b)`` of the min-max form for one goal.

    ``dec``      lockstep weight-read time; minimize the max.
    ``enc``      lockstep GEMM time; minimize the max.
    ``maxkv``    maximize the weakest rank's free bytes. ``sum_r`` free bytes
                 is invariant under the constraint (module docstring §2), so
                 the usable context is decided by ``min_r`` alone — the
                 max-min is the goal, not a proxy for it.
    ``sessions`` identical objective to ``maxkv`` in the variable: the state
                 bound is set by the GDN unit partition, which follows the
                 BASE plan and does not move with the MLP key. The difference
                 between the two goals is the SCORE, not the argmax, and the
                 solver says so instead of pretending they optimize apart.
    """
    n = model.perf.tp_size
    if goal == "dec":
        bw = model.perf.effective_decode_bw(
            model.rates.require_membw_gbs(), model.rates.gemv_gbs
        )
        a = [model.fixed_bytes[r] / bw[r] for r in range(n)]
        b = [model.unit_bytes / bw[r] for r in range(n)]
        return a, b
    if goal == "enc":
        eta = _gemm_efficiency()
        gemm = model.rates.require_gemm_tflops()
        a = [2.0 * model.fixed_params[r] / (gemm[r] * 1e12 * eta) for r in range(n)]
        b = [2.0 * model.unit_params / (gemm[r] * 1e12 * eta) for r in range(n)]
        return a, b
    if goal in ("maxkv", "sessions"):
        # free_r(u) = free_r(0) - unit_bytes * u_r  ->  maximize min_r free_r
        # == minimize max_r ( -free_r(0) + unit_bytes * u_r ).
        free0 = model.free_bytes_at_zero()
        a = [-(free0[r] - role_post_mib[r] * 2**20) for r in range(n)]
        b = [model.unit_bytes] * n
        return a, b
    raise ValueError(f"unknown goal {goal!r}; known: {GOALS}")


def _objective_value(
    goal: str, model: KeyCostModel, units: Sequence[int], role_post_mib: Sequence[float]
) -> float:
    """The min-max objective at an integer point (lower is better)."""
    a, b = _terms(goal, model, role_post_mib)
    return max(a[r] + b[r] * units[r] for r in range(len(units)))


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------


def _cell(
    value: Optional[float],
    provenance: str,
    basis: str,
    *,
    unit: str = "",
    study: Optional[str] = None,
) -> dict:
    """One number with the label that says where it came from.

    Shape-identical to ``wizard.cell`` on purpose (the Guide tab renders both
    with one component); duplicated rather than imported so the wizard can
    depend on the solver and not the other way round.
    """
    if provenance == ABSENT:
        value = None
    return {
        "value": value,
        "available": value is not None,
        "provenance": provenance,
        "basis": basis,
        "unit": unit,
        "study": study,
    }


@dataclasses.dataclass
class _Anchors:
    """Absolute anchors that turn relative model output into tok/s.

    Ratios are what the cost model is calibrated for; absolute rates need one
    measured point per phase. When ``split_probe`` has a row for this model
    the anchor is MEASURED and the derived numbers say so; otherwise the
    anchor itself is an estimate built from the roofline, and everything
    downstream inherits that.
    """

    decode_tok_s: Optional[float] = None
    prefill_tok_s: Optional[float] = None
    vector: Optional[List[int]] = None
    provenance: str = ABSENT
    source: str = ""


def _load_anchors(model_path: str, tp_size: int) -> _Anchors:
    """Newest measured baseline row for this checkpoint, if there is one."""
    try:
        from sglang.srt.planner import split_probe as sp

        store = sp.SplitProbeStore.load(sp.default_store_path())
    except Exception:
        return _Anchors()
    best: Optional[Any] = None
    for row in store.entries():
        if row.model_path != model_path or int(row.tp_size) != int(tp_size):
            continue
        if row.unbootable or not row.has_numbers():
            continue
        if best is None or float(row.timestamp or 0) > float(best.timestamp or 0):
            best = row
    if best is None:
        return _Anchors()
    vec: Optional[List[int]] = None
    if best.chosen_vector:
        try:
            vec = [int(x) for x in str(best.chosen_vector).split(",")]
        except ValueError:
            vec = None
    return _Anchors(
        decode_tok_s=float(best.decode_tok_s),
        prefill_tok_s=float(best.prefill_tok_s),
        vector=vec,
        provenance=MEASURED,
        source=f"split_probe row {best.candidate!r}",
    )


def _predict_all(
    model: KeyCostModel,
    units: Sequence[int],
    role_post_mib: Sequence[float],
    anchors: _Anchors,
    *,
    target_context: int,
    state_bound_sessions: Optional[int],
) -> Tuple[Dict[str, dict], Dict[str, float]]:
    """Every goal's predicted value for one key. Returns (cells, raw)."""
    n = model.perf.tp_size
    cells: Dict[str, dict] = {}
    raw: Dict[str, float] = {}

    cap = model.capacity(units, role_post_mib)
    ctx = float(cap["ctx"])
    raw["maxkv"] = ctx
    cells["maxkv"] = _cell(
        round(ctx) if cap["feasible"] else None,
        ESTIMATE if cap["feasible"] else ABSENT,
        (
            "budget minus weights, mamba pool and reserves over the KV cell "
            f"({model.perf.kv_cell_bytes / 1024:.0f} KiB/token, counted over "
            f"{model.perf.full_layers} full-attention layers of "
            f"{model.perf.n_layers}); usable context is "
            "min(sum, 64 x weakest rank)"
            if cap["feasible"]
            else "this key leaves at least one rank without a usable KV pool"
        ),
        unit="tokens",
    )

    sessions_val: Optional[float] = None
    if cap["feasible"] and target_context > 0:
        kv_bound = ctx / float(target_context)
        sessions_val = kv_bound
        basis = (
            f"KV-bound sessions at {target_context} tokens "
            f"({ctx:.0f} / {target_context})"
        )
        if state_bound_sessions is not None:
            sessions_val = min(kv_bound, float(state_bound_sessions))
            basis += (
                f"; capped by the GDN state pool at {state_bound_sessions} "
                "sessions (mrr geometry, set by the base plan — the MLP key "
                "does not move it)"
            )
        raw["sessions"] = sessions_val
    cells["sessions"] = _cell(
        math.floor(sessions_val) if sessions_val is not None else None,
        ESTIMATE if sessions_val is not None else ABSENT,
        basis if sessions_val is not None else "no feasible KV pool for this key",
        unit="sessions",
    )
    if sessions_val is None:
        raw["sessions"] = 0.0

    # Decode: the calibrated step-ratio form against the base plan, anchored
    # on a measured rate when one exists.
    base_units = model.perf.mlp_unit_partition(list(model.base_plan))
    tw = model.decode_weight_time(units)
    tw_base = model.decode_weight_time(base_units)
    f = model.perf.calibration.nonweight_fraction
    step_ratio = f + (1.0 - f) * (tw / tw_base if tw_base else 1.0)
    if anchors.decode_tok_s:
        dec_val: Optional[float] = anchors.decode_tok_s / step_ratio
        dec_basis = (
            f"measured baseline {anchors.decode_tok_s:.1f} tok/s "
            f"({anchors.source}) scaled by the predicted step ratio "
            f"{step_ratio:.3f} = weight term {tw / tw_base:.3f} folded with "
            f"the {f:.0%} non-weight share of a bs=1 step"
        )
    else:
        dec_val = None
        dec_basis = (
            "no measured decode baseline for this checkpoint on this rig; "
            f"the predicted step ratio against the base plan is {step_ratio:.3f}"
        )
    raw["dec"] = (1.0 / step_ratio) if step_ratio else 0.0
    cells["dec"] = _cell(
        round(dec_val, 1) if dec_val is not None else None,
        ESTIMATE if dec_val is not None else ABSENT,
        dec_basis,
        unit="tok/s",
    )

    # Absent pair matrix: the collective term is NOT PRICED and the ratio
    # below is a COMPUTE-ONLY one (#359, replacing the 1e-3 GB/s placeholder
    # that ``_prefill_sharded_time``'s own 0.1 floor silently swallowed before
    # it ever reached the arithmetic). The omitted term is split-invariant --
    # n_layers * hidden * ranks, never the candidate vector -- so it shifts
    # both times by the same constant and the ORDER over candidates is the
    # order at any link rate. What it is not is a magnitude a reader can act
    # on, and the basis text says so instead of printing a bare number.
    _link = model.rates.link_bw_gbs
    _gemm = model.rates.require_gemm_tflops()
    _families = model.rates.gemm_family_tflops or None
    tp = model.perf.prefill_time_model(list(units), _gemm, _link, _families)
    tp_base = model.perf.prefill_time_model(list(base_units), _gemm, _link, _families)
    pre_ratio = (tp_base / tp) if tp else 1.0
    compute_only = _link is None and model.rates.ranks > 1
    lane_txt = (
        f" on the {model.rates.gemm_format} GEMM lane of each card"
        if model.rates.gemm_format
        else ""
    )
    if compute_only:
        # No absolute figure and no magnitude: the ratio orders candidates and
        # that is all it is allowed to do here.
        pre_val = None
        pre_basis = (
            "no pair matrix for these cards, so the per-layer all-reduce is "
            f"not priced{lane_txt}. The candidates are still ORDERED by their "
            "compute term (the collective is split-invariant), but no prefill "
            "magnitude is reported from a comparison that omits it. Run the "
            "card probe (POST /api/card_probe) to resolve it."
        )
    elif anchors.prefill_tok_s:
        pre_val: Optional[float] = anchors.prefill_tok_s * pre_ratio
        pre_basis = (
            f"measured baseline {anchors.prefill_tok_s:.0f} tok/s "
            f"({anchors.source}) scaled by the predicted prefill ratio "
            f"{pre_ratio:.3f} (lockstep GEMM{lane_txt} + the per-layer "
            "all-reduce, with the split-invariant launch/non-GEMM remainder "
            f"held at {model.perf.calibration.prefill_invariant:.0%} of the "
            "base step)"
        )
    else:
        pre_val = None
        pre_basis = (
            "no measured prefill baseline for this checkpoint on this rig; "
            f"the predicted ratio against the base plan is {pre_ratio:.3f}"
        )
    raw["enc"] = pre_ratio
    cells["enc"] = _cell(
        round(pre_val, 0) if pre_val is not None else None,
        ESTIMATE if pre_val is not None else ABSENT,
        pre_basis,
        unit="tok/s",
    )

    # Named side terms, always reported so the caller can see what the phase
    # model charged and what it could not charge at all.
    coll = model.collective_decode_s()
    cells["collective_decode_ms"] = _cell(
        round(coll * 1e3, 3) if coll is not None else None,
        ESTIMATE if coll is not None else ABSENT,
        (
            f"pair matrix: {2 * model.perf.n_layers} all-reduces of "
            f"{model.perf.hidden * 2 / 1024:.1f} KiB, ring factor "
            f"{_ring_factor(n):.2f}, narrowest pair "
            f"{model.rates.link_bw_gbs:.2f} GB/s at "
            f"{model.rates.link_latency_us:.1f} us"
            if coll is not None
            else "no pair matrix on disk — the collective term is not guessed"
        ),
        unit="ms",
    )
    host = model.host_tier_decode_s(units)
    cells["host_tier_decode_ms"] = _cell(
        round(host * 1e3, 3) if host is not None else None,
        ESTIMATE if host is not None else ABSENT,
        (
            "offloadable routed-expert bytes over the measured pinned H2D rate"
            if host is not None
            else "no host probe on disk — the host tier is not guessed"
        ),
        unit="ms",
    )
    return cells, raw


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Candidate:
    """One key, with everything needed to judge and to launch it."""

    units: List[int]
    mlp_ratio: List[int]
    roles: List[str]
    feasible: bool
    reasons: List[str]
    predictions: Dict[str, dict]
    raw: Dict[str, float]
    tradeoff: Dict[str, Any]
    remeasure: Dict[str, Any]
    label: str = ""
    launch_flags: List[str] = dataclasses.field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "key": {
                "mlp_units": list(self.units),
                "rank_mlp_ratio": list(self.mlp_ratio),
                "roles": list(self.roles),
            },
            "label": self.label,
            "feasible": self.feasible,
            "reasons": list(self.reasons),
            "launch_flags": list(self.launch_flags),
            "predictions": self.predictions,
            "tradeoff": self.tradeoff,
            "remeasure": self.remeasure,
        }


def _ratio_of(units: Sequence[int]) -> List[int]:
    """The smallest integer ratio that realizes this unit partition.

    ``--rank-mlp-ratio`` is what the user types; the units are what the
    runtime partitions. Reducing by the gcd keeps the flag readable without
    changing what it resolves to.
    """
    vals = [max(int(u), 0) for u in units]
    g = 0
    for v in vals:
        g = math.gcd(g, v)
    if g <= 1:
        return vals
    return [v // g for v in vals]


def _remeasure_hook(model_path: str, units: Sequence[int], tp_size: int) -> dict:
    """The "measure it" hook for a predicted key — a split_probe job pinned
    to exactly this vector, so the measurement lands next to the prediction
    and the model error becomes visible instead of asserted."""
    ratio = _ratio_of(units)
    return Remeasure(
        kind="job",
        label=f"boot and measure the key {','.join(str(v) for v in ratio)}",
        path="/api/split_probe",
        status_path="/api/split_probe/status",
        body={
            "candidate": ",".join(str(v) for v in ratio),
            "tp_size": int(tp_size),
            "model_path": model_path,
        },
        cost="one cold boot to READY plus a prefill and a decode window",
    ).to_json()


def _tradeoff(
    cand_raw: Dict[str, float],
    ref_raw: Dict[str, float],
    cells: Dict[str, dict],
    roles: Sequence[str],
) -> Dict[str, Any]:
    """The three-part trade-off line: who gains, at whose cost, worth it for.

    A gain smaller than the harness noise floor is reported as "inside the
    noise floor" and never as a win — the same rule the measurement side
    obeys.
    """
    gains: List[dict] = []
    costs: List[dict] = []
    for goal in GOALS:
        a, b = ref_raw.get(goal), cand_raw.get(goal)
        if not a or not b:
            continue
        delta = (b / a - 1.0) * 100.0
        entry = {
            "goal": goal,
            "label": GOAL_LABELS[goal],
            "delta_pct": round(delta, 2),
            "value": cells.get(goal, {}).get("value"),
            "unit": GOAL_UNITS[goal],
            "provenance": cells.get(goal, {}).get("provenance", ABSENT),
            "below_noise_floor": abs(delta) < NOISE_FLOOR_PCT,
        }
        if delta > 0:
            gains.append(entry)
        elif delta < 0:
            costs.append(entry)
    gains.sort(key=lambda e: -e["delta_pct"])
    costs.sort(key=lambda e: e["delta_pct"])

    for role in set(roles):
        spec = _ROLE_BY_KEY.get(role)
        if spec is not None and spec.fixed_post_mib:
            costs.append(
                {
                    "goal": "ledger",
                    "label": f"fixed post for the {spec.label}",
                    "delta_pct": None,
                    "value": spec.fixed_post_mib,
                    "unit": "MiB",
                    "provenance": ESTIMATE,
                    "below_noise_floor": False,
                }
            )

    worth: List[str] = []
    top = gains[0]["goal"] if gains else None
    if top == "enc":
        worth.append("prefill-heavy work: long fresh prompts, document intake")
    elif top == "dec":
        worth.append("decode-heavy work: long generations, low concurrency")
    elif top == "maxkv":
        worth.append("KV-hungry work: very long single conversations")
    elif top == "sessions":
        worth.append("many concurrent sessions at a moderate context")

    def _fmt(entries: List[dict], sign: str) -> str:
        if not entries:
            return "nothing measurable"
        parts = []
        for e in entries[:3]:
            if e["delta_pct"] is None:
                parts.append(f"{e['label']} {e['value']:.0f} {e['unit']}")
            else:
                tail = " (inside the noise floor)" if e["below_noise_floor"] else ""
                parts.append(f"{e['label']} {sign}{abs(e['delta_pct']):.1f} %{tail}")
        return ", ".join(parts)

    line = (
        f"gains {_fmt(gains, '+')}; costs {_fmt(costs, '-')}; "
        f"worth it for {', '.join(worth) if worth else 'no clear workload'}"
    )
    return {"gains": gains, "costs": costs, "worth_it_for": worth, "line": line}


# ---------------------------------------------------------------------------
# The solve entry point
# ---------------------------------------------------------------------------


UnitBound = Tuple[Optional[int], Optional[int]]


@dataclasses.dataclass(frozen=True)
class OuterLane:
    """One already-resident lane the lane being solved has to nest inside.

    ``rank_of[i]`` is this lane's rank on the card the solved lane's rank
    ``i`` sits on, or ``None`` for a card this lane does not use.
    """

    key: str
    units: Sequence[int]
    rank_of: Sequence[Optional[int]]


def rank_map_over_cards(
    inner_gpus: Sequence[int], outer_gpus: Sequence[int]
) -> List[Optional[int]]:
    """``rank_of`` for a pair of lanes, derived from their card maps.

    When the outer lane puts SEVERAL ranks on one card the first is taken and
    that is a real ambiguity, not a detail: which of the co-located outer
    shards the inner lane reuses is a placement decision the card map does not
    carry. Callers that co-locate ranks must pass ``rank_of`` explicitly
    instead of deriving it here.
    """
    first: Dict[int, int] = {}
    for r, g in enumerate(outer_gpus):
        first.setdefault(int(g), r)
    return [first.get(int(g)) for g in inner_gpus]


def nesting_bounds_over(outers: Sequence[OuterLane]) -> List[UnitBound]:
    """Box bounds from a SET of resident lanes: the intersection of ceilings.

    The set-wise form of ``nesting_bounds``. Nesting inside several resident
    lanes at once is nesting inside the SMALLEST of them on every card, so the
    boxes intersect — ``hi_i = min`` over the outer lanes that use card ``i``,
    unbounded where none does. A lane's shard cannot be a subset of two
    different shards without being a subset of the smaller one, so this is the
    definition unrolled, not a heuristic.

    The intersection is NECESSARY, never sufficient (see ``nesting_bounds``),
    and it is not the whole set-wise question either: it bounds ONE lane
    against lanes already fixed. Whether the whole set can be arranged in a
    refinement chain at all is ``nesting_hull``, and a satisfied box says
    nothing about that.
    """
    if not outers:
        return []
    width = max(len(o.rank_of) for o in outers)
    out: List[UnitBound] = []
    for i in range(width):
        hi: Optional[int] = None
        for o in outers:
            if i >= len(o.rank_of):
                continue
            r = o.rank_of[i]
            if r is None:
                continue
            v = int(o.units[r])
            hi = v if hi is None else min(hi, v)
        out.append((None, hi))
    return out


def nesting_bounds(
    outer_units: Sequence[int],
    outer_rank_of: Sequence[Optional[int]],
) -> List[UnitBound]:
    """Box bounds that force one lane's shards to be SUBSETS of another's.

    The dual-group case: an outer lane (say TP=2 over two cards) is already
    resident, and an inner lane (TP=3 over three) must reuse it, so on every
    shared card the inner lane's shard has to be contained in the outer
    lane's. Containment of the unit SETS implies ``u_r <= v_r`` on the shared
    cards, and that is exactly a box on the inner problem — the same bounds
    the roles already express, so the solver needs no new machinery, only a
    way to be told them.

    ``outer_rank_of[i]`` is the outer lane's rank on the card inner rank ``i``
    sits on, or ``None`` for a card the outer lane does not use (the
    complement card, whose shard is the only genuinely new resident bytes).

    This is the ``N=2`` case of ``nesting_bounds_over`` and is implemented as
    that call, so the pairwise and set-wise answers cannot drift apart.

    NOT a free lunch, and the caller has to know it: the bound is a NECESSARY
    condition for reuse, not a sufficient one. Containment must also hold for
    the axes this vector does not carry — the attention, GDN and vocab shards
    — and the unit ranges have to be laid out CONTIGUOUSLY so that the inner
    shard is a prefix/interval of the outer one rather than merely smaller.
    Both are properties of the partition layout, not of the count, and the
    solver models counts. State them as configuration requirements; do not
    read a satisfied box as proof that the bytes really are shared.
    """
    return nesting_bounds_over(
        [OuterLane(key="outer", units=list(outer_units), rank_of=list(outer_rank_of))]
    )


# ---------------------------------------------------------------------------
# The hull tree — set-wise nesting over N lanes (PRIO-Nachtrag 8b)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LaneKey:
    """One lane's distribution key, as the shared-bytes algebra sees it.

    ``ratio`` is the lane's ``--rank-tp-ratio``; ``family_ratios`` carries the
    per-family overrides (``mlp``, ``moe``, ``vocab``) exactly as
    ``set_tp_partition_ratios`` takes them, so a solved ``--rank-mlp-ratio``
    enters here as ``(("mlp", units),)``. ``gpus`` is the physical card per
    rank: two lanes only have to nest where they actually meet.

    ``nests_in`` names the COARSER lane whose shards contain this lane's —
    the ``outer`` of ``nesting_bounds``, and the FAST group of
    ``dual_group.NestedGroupPlan`` with this lane in the BIG role.
    ``shared_segments``, when given, PINS that relation and is read exactly as
    ``dual_group.NestedGroupPlan.segments``: indexed by the ranks of whichever
    of the two lanes turns out to be the COARSER, listing the finer lane's
    ranks each covers. Which side is coarser is decided from the splits and
    not from the direction of ``nests_in``, so a caller cannot pin the
    relation upside down. Left ``None`` the hull DERIVES the segmentation,
    which answers the weaker question "is there ANY grouping under which these
    nest" — see ``nesting_hull``.
    """

    key: str
    ratio: Tuple[int, ...]
    gpus: Tuple[int, ...] = ()
    family_ratios: Tuple[Tuple[str, Tuple[int, ...]], ...] = ()
    priority_class: int = 0
    nests_in: Optional[str] = None
    shared_segments: Optional[Tuple[Tuple[int, ...], ...]] = None

    def ratio_for(self, family: Optional[str] = None) -> Tuple[int, ...]:
        if family is not None:
            for name, vec in self.family_ratios:
                if name == family:
                    return tuple(int(x) for x in vec)
        return tuple(int(x) for x in self.ratio)


@dataclasses.dataclass(frozen=True)
class HullProbe:
    """One sharded dimension the hull has to survive.

    The planner-side twin of ``dual_group.NestingProbe``: ``units`` is the
    indivisible unit count of the dimension, ``groups`` the kv-head-group
    alignment of the Q dimension (#116) and ``None`` elsewhere, ``family`` the
    per-family vector the dimension shards by.
    """

    what: str
    units: int
    groups: Optional[int] = None
    family: Optional[str] = None


@dataclasses.dataclass
class HullTree:
    """The result of the set-wise check: a refinement forest, or why not."""

    ok: bool
    #: Coarsest first, per connected component of the "shares a card" graph.
    order: List[str]
    #: lane key -> the next COARSER lane it nests inside (None = a root).
    parent: Dict[str, Optional[str]]
    #: (child, parent) -> the derived segmentation, per probe that pinned it.
    segments: Dict[Tuple[str, str], Tuple[Tuple[int, ...], ...]]
    #: Pairs that had to nest and did not, each naming probe, split and cut.
    failures: List[str]
    #: Pairs that were checked at all (they share at least one card).
    checked: List[Tuple[str, str]]
    #: Pairs that share no card and were therefore NOT required to nest.
    disjoint: List[Tuple[str, str]]

    def to_json(self) -> dict:
        return {
            "ok": self.ok,
            "order": list(self.order),
            "parent": dict(self.parent),
            "segments": {
                f"{c}<{p}": [list(s) for s in segs]
                for (c, p), segs in self.segments.items()
            },
            "failures": list(self.failures),
            "checked": [list(p) for p in self.checked],
            "disjoint": [list(p) for p in self.disjoint],
        }


def partition_cuts(
    units: int, ratio: Sequence[int], groups: Optional[int] = None
) -> Tuple[int, ...]:
    """The cut points of a lane's split: ``(0, ..., units)``.

    The whole nesting algebra lives in these numbers. A contiguous ordered
    partition IS its cut set, one partition refines another exactly when its
    cut set is a SUPERSET, and the shared shards are the same bytes exactly
    when the cuts that bound them coincide — which is the prefix-sum argument
    of DESIGN #121 §3.2, restated so that it composes over more than two
    lanes.

    A vector that ALREADY sums to ``units`` is taken as the split itself and
    not re-partitioned. That is not a convenience: the solver's answer is a
    unit vector, and ``partition_units`` cannot reproduce it, because it
    gives every rank at least one unit while the solver's ``kv_donor`` role
    is precisely a rank with zero. Re-splitting a solved key would compare
    the lane against a partition nobody installs. Anything else is a ratio
    and goes through ``partition_units``, rounding and all.
    """
    from sglang.srt.distributed.utils import partition_units

    vec = [int(w) for w in ratio]
    if sum(vec) != int(units) or any(w < 0 for w in vec):
        vec = list(partition_units(int(units), vec, groups))
    out = [0]
    for s in vec:
        out.append(out[-1] + int(s))
    return tuple(out)


def _segments_from_cuts(
    coarse_cuts: Sequence[int], fine_cuts: Sequence[int]
) -> Optional[Tuple[Tuple[int, ...], ...]]:
    """For each COARSE rank, the FINE ranks it covers — ``dual_group``'s
    ``segments``. ``None`` when the coarse split is not a union of fine
    shards, i.e. when the two do not nest at all."""
    index = {c: i for i, c in enumerate(fine_cuts)}
    segs: List[Tuple[int, ...]] = []
    for f in range(len(coarse_cuts) - 1):
        a, b = coarse_cuts[f], coarse_cuts[f + 1]
        if a not in index or b not in index:
            return None
        segs.append(tuple(range(index[a], index[b])))
    return tuple(segs)


def nesting_hull(
    lanes: Sequence[LaneKey],
    probes: Sequence[HullProbe],
    *,
    require_disjoint_lanes_to_nest: bool = False,
) -> HullTree:
    """Can these N lanes live over ONE set of resident bytes?

    ``nesting_bounds`` answers the pairwise question — does the lane I am
    solving fit inside the one lane that is already resident. With N lanes
    that is not enough, and the gap is not a corner case: two lanes can each
    nest perfectly inside the same coarse lane and still be incompatible with
    EACH OTHER, because their cuts fall in different places. On the rig's own
    ``[6,1,1]`` group, the two obvious two-rank lanes ``[6,2]`` (sharing rank
    0) and ``[7,1]`` (sharing rank 2) nest in the group at almost every unit
    count and nest in each other at NONE of them: one cuts after rank 0's
    units, the other after rank 1's, and rank 1 always holds at least one
    unit. Pairwise-against-the-root is therefore not a proof, and this
    function is what replaces it.

    The check, per pair of lanes that share at least one card, per probe:

    * split both dimensions with the fork's own ``partition_units``, so the
      rounding is the rounding the runtime will really do;
    * require the two cut sets to be COMPARABLE (one a superset of the other).
      Comparable means one partition refines the other, i.e. every shard of
      the coarser lane is an exact union of shards of the finer one — that is
      what makes the bytes the same bytes and not merely the same count;
    * require the direction to be CONSISTENT across probes: a lane that is
      coarser on the MLP dimension and finer on the vocabulary dimension has
      no place in any tree, and that contradiction is reported as such;
    * where ``segments_in_parent`` is pinned, require exactly that grouping —
      the question ``distributed/dual_group.py`` asks, since Slice B installs
      a segmentation rather than discovering one.

    Lanes that share no card are NOT required to nest (they hold different
    bytes on different silicon, so there is nothing to share); pass
    ``require_disjoint_lanes_to_nest`` to check them anyway, which is what a
    caller wanting ONE key family across the whole rig would want.

    Returns the forest and, when it does not exist, every failure named down
    to the probe, the two splits and the cut that broke it. It never repairs
    a set into one that nests: which lane to move is a plan decision.
    """
    keys = [lane.key for lane in lanes]
    if len(set(keys)) != len(keys):
        return HullTree(
            ok=False,
            order=[],
            parent={},
            segments={},
            failures=["duplicate lane keys: the hull would be ambiguous"],
            checked=[],
            disjoint=[],
        )
    by_key = {lane.key: lane for lane in lanes}
    failures: List[str] = []
    checked: List[Tuple[str, str]] = []
    disjoint: List[Tuple[str, str]] = []
    #: (a, b) -> True when a is COARSER than b (b refines a).
    coarser: Dict[Tuple[str, str], bool] = {}
    segments: Dict[Tuple[str, str], Tuple[Tuple[int, ...], ...]] = {}

    for ia in range(len(lanes)):
        for ib in range(ia + 1, len(lanes)):
            a, b = lanes[ia], lanes[ib]
            shared = sorted(set(int(g) for g in a.gpus) & set(int(g) for g in b.gpus))
            if not shared and not require_disjoint_lanes_to_nest:
                disjoint.append((a.key, b.key))
                continue
            checked.append((a.key, b.key))
            direction: Optional[str] = None
            for probe in probes:
                ra = a.ratio_for(probe.family)
                rb = b.ratio_for(probe.family)
                fam = f" family {probe.family!r}" if probe.family else ""
                try:
                    ca = partition_cuts(probe.units, ra, probe.groups)
                    cb = partition_cuts(probe.units, rb, probe.groups)
                except ValueError as exc:
                    failures.append(
                        f"{a.key} vs {b.key}: {probe.what}{fam} with "
                        f"{probe.units} units cannot be split at all — {exc}"
                    )
                    continue
                sa, sb = set(ca), set(cb)
                if sa == sb:
                    this = "same"
                elif sa >= sb:
                    this = "b_coarser"
                elif sb >= sa:
                    this = "a_coarser"
                else:
                    only_a = sorted(sa - sb)
                    only_b = sorted(sb - sa)
                    failures.append(
                        f"{a.key} vs {b.key} (sharing card(s) "
                        f"{shared}): {probe.what}{fam} splits into "
                        f"{list(ca)} and {list(cb)} over {probe.units} units "
                        f"— neither cut set contains the other ({a.key} cuts "
                        f"at {only_a} where {b.key} does not, {b.key} at "
                        f"{only_b} where {a.key} does not). Neither lane "
                        "refines the other, so no shard of one is a union of "
                        "shards of the other and the bytes cannot be shared. "
                        "This is the set-wise failure: each lane may still "
                        "nest perfectly inside a third, coarser lane."
                    )
                    continue
                if this == "same":
                    continue
                if direction is None:
                    direction = this
                elif direction != this:
                    failures.append(
                        f"{a.key} vs {b.key}: the refinement direction FLIPS "
                        f"between dimensions — {probe.what}{fam} makes "
                        f"{'a' if this == 'a_coarser' else 'b'} the coarser "
                        "lane while an earlier dimension made the other one "
                        "coarser. A hull tree needs one direction per pair; "
                        "this set has none."
                    )
                    direction = None
                    break
            if direction == "a_coarser":
                coarser[(a.key, b.key)] = True
            elif direction == "b_coarser":
                coarser[(b.key, a.key)] = True

    # -- pinned segmentations: the question dual_group asks -----------------
    for lane in lanes:
        if lane.shared_segments is None or lane.nests_in is None:
            continue
        other = by_key.get(lane.nests_in)
        if other is None:
            failures.append(
                f"{lane.key} declares nests_in={lane.nests_in!r}, which is not "
                "in the lane set"
            )
            continue
        want = tuple(tuple(int(r) for r in seg) for seg in lane.shared_segments)
        for probe in probes:
            fam = f" family {probe.family!r}" if probe.family else ""
            try:
                oc = partition_cuts(
                    probe.units, other.ratio_for(probe.family), probe.groups
                )
                lc = partition_cuts(
                    probe.units, lane.ratio_for(probe.family), probe.groups
                )
            except ValueError:
                continue
            # Which side is coarser is a property of the splits, not of the
            # direction the caller wrote the relation in.
            if len(set(oc)) <= len(set(lc)):
                got = _segments_from_cuts(oc, lc)
                coarse_key, fine_key = other.key, lane.key
            else:
                got = _segments_from_cuts(lc, oc)
                coarse_key, fine_key = lane.key, other.key
            if got != want:
                failures.append(
                    f"{fine_key} in {coarse_key}: {probe.what}{fam} over "
                    f"{probe.units} units nests as "
                    f"{[list(s) for s in got] if got else 'not at all'}, but "
                    f"the pinned segmentation is {[list(s) for s in want]}. "
                    f"The lane would read a shard {coarse_key}'s rank does "
                    "not own. (A free segmentation may still exist — that is "
                    "the weaker question this check deliberately does not "
                    "ask, and it is where this module and dual_group differ.)"
                )

    # -- the forest: coarsest first, per connected component ----------------
    order: List[str] = []
    remaining = list(keys)
    guard = 0
    while remaining and guard <= len(keys):
        guard += 1
        free = [
            k
            for k in remaining
            if not any(coarser.get((o, k)) for o in remaining if o != k)
        ]
        if not free:
            failures.append(
                "the refinement relation has a cycle over "
                f"{sorted(remaining)} — no lane is coarsest, so there is no "
                "hull tree"
            )
            break
        for k in sorted(free):
            order.append(k)
            remaining.remove(k)

    parent: Dict[str, Optional[str]] = {}
    for i, k in enumerate(order):
        chosen: Optional[str] = None
        for cand in order[:i]:
            if coarser.get((cand, k)):
                chosen = cand
        parent[k] = chosen
        if chosen is not None:
            for probe in probes:
                try:
                    pc = partition_cuts(
                        probe.units,
                        by_key[chosen].ratio_for(probe.family),
                        probe.groups,
                    )
                    cc = partition_cuts(
                        probe.units, by_key[k].ratio_for(probe.family), probe.groups
                    )
                except ValueError:
                    continue
                segs = _segments_from_cuts(pc, cc)
                if segs is not None:
                    segments[(k, chosen)] = segs
                    break

    return HullTree(
        ok=not failures,
        order=order,
        parent=parent,
        segments=segments,
        failures=failures,
        checked=checked,
        disjoint=disjoint,
    )


def _role_bounds(
    roles: Sequence[str],
    units: int,
    unit_bounds: Optional[Sequence[UnitBound]] = None,
) -> Tuple[List[int], List[int], List[float]]:
    """Per-rank ``(lo, hi, fixed post)`` on the unit grid.

    Role bounds and any explicit ``unit_bounds`` are INTERSECTED: a kv_donor
    that is also nested stays a donor, and a nesting ceiling never widens a
    role. An empty intersection (lo > hi) is left as-is and fails loudly in
    ``water_fill`` rather than being silently repaired into something the
    caller did not ask for.
    """
    lo: List[int] = []
    hi: List[int] = []
    post: List[float] = []
    for i, key in enumerate(roles):
        spec = _ROLE_BY_KEY[key]
        rank_lo = int(math.ceil(spec.weight_lo * units))
        rank_hi = int(math.floor(spec.weight_hi * units))
        if unit_bounds is not None and i < len(unit_bounds):
            b_lo, b_hi = unit_bounds[i]
            if b_lo is not None:
                rank_lo = max(rank_lo, int(b_lo))
            if b_hi is not None:
                rank_hi = min(rank_hi, int(b_hi))
        lo.append(rank_lo)
        hi.append(rank_hi)
        post.append(spec.fixed_post_mib)
    return lo, hi, post


def _solve_single(
    goal: str,
    model: KeyCostModel,
    roles: Sequence[str],
    unit_bounds: Optional[Sequence[UnitBound]] = None,
) -> Optional[List[int]]:
    """The closed-form optimum for one goal under one role assignment."""
    lo, hi, post = _role_bounds(roles, model.units, unit_bounds)
    if sum(lo) > model.units or sum(hi) < model.units:
        return None
    a, b = _terms(goal, model, post)
    try:
        cont = water_fill(
            a, b, float(model.units), [float(x) for x in lo], [float(x) for x in hi]
        )
    except ValueError:
        return None
    vec = project_to_grid(cont, model.units, lo, hi)
    return _repair(vec, lambda u: _objective_value(goal, model, u, post), lo, hi)


def _candidate_set(
    model: KeyCostModel,
    roles: Sequence[str],
    anchor_vectors: Sequence[Sequence[int]],
    *,
    segment_steps: int = 40,
    grid_steps: int = 8,
    unit_bounds: Optional[Sequence[UnitBound]] = None,
) -> List[Tuple[int, ...]]:
    """The bounded search set for the combined goals (§4).

    Union of: the single-goal optima, the segment between each pair of them
    (rounded and sum-repaired at every step), and a coarse simplex grid. The
    grid catches off-segment points; the segment gives the front its
    resolution where it matters.
    """
    lo, hi, _ = _role_bounds(roles, model.units, unit_bounds)
    out: Dict[Tuple[int, ...], None] = {}

    def add(vec: Sequence[float]) -> None:
        v = project_to_grid(list(vec), model.units, lo, hi)
        if sum(v) == model.units:
            out[tuple(v)] = None

    for v in anchor_vectors:
        add(v)
    for i in range(len(anchor_vectors)):
        for j in range(i + 1, len(anchor_vectors)):
            va, vb = anchor_vectors[i], anchor_vectors[j]
            for s in range(segment_steps + 1):
                t = s / segment_steps
                add([va[r] + t * (vb[r] - va[r]) for r in range(len(va))])

    n = model.perf.tp_size
    if n <= 4 and grid_steps > 0:
        step = max(model.units // grid_steps, 1)

        def rec(r: int, left: int, acc: List[int]) -> None:
            if r == n - 1:
                add(acc + [left])
                return
            k = 0
            while k <= left:
                rec(r + 1, left - k, acc + [k])
                k += step

        rec(0, model.units, [])
    return list(out)


def _knee(points: List[Tuple[float, float]]) -> int:
    """Index of the point furthest from the chord joining the two endpoints,
    in the normalized objective plane. The standard knee: it is where buying
    one more unit of A starts costing disproportionately much B."""
    if len(points) < 3:
        return 0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    dx = (x1 - x0) or 1.0
    dy = (y1 - y0) or 1.0
    norm = [((p[0] - x0) / dx, (p[1] - y0) / dy) for p in points]
    ax, ay = norm[0]
    bx, by = norm[-1]
    seg = math.hypot(bx - ax, by - ay) or 1.0
    best, best_i = -1.0, 0
    for i, (px, py) in enumerate(norm):
        d = abs((bx - ax) * (ay - py) - (ax - px) * (by - ay)) / seg
        if d > best:
            best, best_i = d, i
    return best_i


def solve(
    plan_inputs,
    base_plan: Sequence[int],
    budgets_mib: Sequence[int],
    rates: RigRates,
    *,
    goal: str = "maxkv",
    goal_b: Optional[str] = None,
    constraints: Optional[Dict[str, float]] = None,
    target_context: int = 8192,
    roles: Optional[Sequence[str]] = None,
    search_roles: bool = True,
    front_size: int = 5,
    unit_bounds: Optional[Sequence[UnitBound]] = None,
) -> SolverAnswer:
    """Compute the key.

    ``goal`` alone      -> one candidate, the closed-form optimum.
    ``goal`` + ``constraints`` -> the constrained optimum (§4a).
    ``goal`` + ``goal_b``      -> a 3-5 point Pareto front including the knee
                                  (§4b); every candidate carries every goal's
                                  predicted value, the sacrificed ones too.
    """
    if goal not in GOALS:
        raise ValueError(f"unknown goal {goal!r}; known: {GOALS}")
    if goal_b is not None and goal_b not in GOALS:
        raise ValueError(f"unknown goal {goal_b!r}; known: {GOALS}")

    model = build_cost_model(plan_inputs, base_plan, budgets_mib, rates)
    n = model.perf.tp_size
    caveats: List[str] = list(rates.absent)

    role_sets: List[Tuple[str, ...]]
    if roles is not None:
        role_sets = [tuple(roles)]
    elif search_roles and n <= MAX_ROLE_SEARCH_RANKS:
        role_sets = _enumerate_roles(n, model)
    else:
        role_sets = [tuple(["shard"] * n)]
        if search_roles:
            caveats.append(
                f"role search skipped: {n} ranks exceeds the "
                f"{MAX_ROLE_SEARCH_RANKS}-rank enumeration cap; only the "
                "all-shard assignment was solved"
            )

    anchors = _load_anchors(plan_inputs.model_path, plan_inputs.tp_size)
    state_bound = _state_bound_sessions(plan_inputs, base_plan, budgets_mib)

    def evaluate(units: Sequence[int], rset: Sequence[str]) -> Candidate:
        _, _, post = _role_bounds(rset, model.units, unit_bounds)
        cells, raw = _predict_all(
            model,
            units,
            post,
            anchors,
            target_context=target_context,
            state_bound_sessions=state_bound,
        )
        feasible = cells["maxkv"]["available"]
        reasons: List[str] = []
        if not feasible:
            reasons.append(
                "at least one rank is left without a usable KV pool under "
                "this key — the configuration would not boot"
            )
        return Candidate(
            units=list(units),
            mlp_ratio=_ratio_of(units),
            roles=list(rset),
            feasible=bool(feasible),
            reasons=reasons,
            predictions=cells,
            raw=raw,
            tradeoff={},
            remeasure=_remeasure_hook(
                plan_inputs.model_path, units, plan_inputs.tp_size
            ),
            launch_flags=[
                "--rank-mlp-ratio",
                ",".join(str(v) for v in _ratio_of(units)),
            ],
        )

    base_units = model.perf.mlp_unit_partition(list(base_plan))
    reference = evaluate(base_units, ["shard"] * n)
    reference.label = "base plan (reference)"
    reference.tradeoff = _tradeoff(
        reference.raw, reference.raw, reference.predictions, reference.roles
    )

    # -- single goal, per role set ---------------------------------------
    best_by_role: Dict[Tuple[str, ...], Dict[str, List[int]]] = {}
    for rset in role_sets:
        got: Dict[str, List[int]] = {}
        for g in GOALS:
            v = _solve_single(g, model, rset, unit_bounds)
            if v is not None:
                got[g] = v
        if got:
            best_by_role[rset] = got
    if not best_by_role:
        return SolverAnswer(
            ok=False,
            goal=goal,
            goal_b=goal_b,
            mode="single",
            candidates=[],
            reference=reference,
            caveats=caveats,
            reasons=[
                "no role assignment can hold the model on this rig under the "
                "given budgets"
            ],
            rates=rates,
            model_units=model.units,
            target_context=target_context,
            anchors=anchors,
        )

    def value_of(c: Candidate, g: str) -> float:
        """The number a constraint and the front are expressed in.

        The predicted cell (tok/s, tokens, sessions) when there is one, so a
        threshold reads as "at least 40000 tokens" rather than as an internal
        ratio; the relative quantity only when no absolute anchor exists.
        Both are monotone in the same direction, so the ranking is the same
        either way — this is about what the caller has to type."""
        cellv = c.predictions.get(g, {}).get("value")
        if cellv is not None:
            return float(cellv)
        return float(c.raw.get(g, 0.0))

    def score(c: Candidate, g: str) -> float:
        return value_of(c, g) if c.feasible else -math.inf

    if goal_b is None and not constraints:
        best: Optional[Candidate] = None
        for rset, got in best_by_role.items():
            if goal not in got:
                continue
            c = evaluate(got[goal], rset)
            if best is None or score(c, goal) > score(best, goal):
                best = c
        if best is None:
            best = reference
        best.label = f"optimum for {GOAL_LABELS[goal]}"
        best.tradeoff = _tradeoff(best.raw, reference.raw, best.predictions, best.roles)
        ref_val = reference.raw.get(goal, 0.0)
        if ref_val and abs(best.raw.get(goal, 0.0) / ref_val - 1.0) < 1e-6:
            # A degenerate goal is a RESULT, not a failure, and hiding it
            # would sell a key that changes nothing as an optimum. For maxkv
            # this is the invariance derived in §2: with the budgets fixed,
            # sum_r P_r does not move with the MLP key, so unless the
            # 64 x weakest-rank term binds, every key holds the same total
            # context and the solver has optimized the TIE-BREAK (the
            # weakest rank's own KV) instead.
            caveats.append(
                f"{GOAL_LABELS[goal]} does not move with the MLP key on this "
                "rig under these budgets — every representable key gives the "
                f"same {GOAL_LABELS[goal]}. The returned key is the tie-break "
                "optimum (the weakest rank's own capacity); the other goals' "
                "deltas in the trade-off line are what actually differs."
            )
        return SolverAnswer(
            ok=True,
            goal=goal,
            goal_b=None,
            mode="single",
            candidates=[best],
            reference=reference,
            caveats=caveats,
            reasons=[],
            rates=rates,
            model_units=model.units,
            target_context=target_context,
            anchors=anchors,
        )

    # -- combined: build the bounded candidate set (§4) --------------------
    pool: List[Candidate] = []
    for rset, got in best_by_role.items():
        anchor_vecs = list(got.values())
        for vec in _candidate_set(model, rset, anchor_vecs, unit_bounds=unit_bounds):
            pool.append(evaluate(list(vec), rset))
    pool = [c for c in pool if c.feasible]
    if not pool:
        pool = [reference]

    if constraints:
        mode = "constraint"
        feasible = [
            c
            for c in pool
            if all(value_of(c, k) >= float(v) for k, v in constraints.items())
        ]
        if not feasible:
            best_other = max(
                pool,
                key=lambda c: min(
                    value_of(c, k) / float(v) if float(v) else 1.0
                    for k, v in constraints.items()
                ),
            )
            reasons = [
                "no key on this rig satisfies "
                + ", ".join(
                    f"{GOAL_LABELS.get(k, k)} >= {v:g}" for k, v in constraints.items()
                )
                + "; the closest key is returned with its shortfall named"
            ]
            best_other.label = "closest key (constraint not satisfiable)"
            best_other.tradeoff = _tradeoff(
                best_other.raw,
                reference.raw,
                best_other.predictions,
                best_other.roles,
            )
            return SolverAnswer(
                ok=True,
                goal=goal,
                goal_b=goal_b,
                mode=mode,
                candidates=[best_other],
                reference=reference,
                caveats=caveats,
                reasons=reasons,
                rates=rates,
                model_units=model.units,
                target_context=target_context,
                anchors=anchors,
            )
        winner = max(feasible, key=lambda c: value_of(c, goal))
        winner.label = f"max {GOAL_LABELS[goal]} under " + ", ".join(
            f"{GOAL_LABELS.get(k, k)} >= {v:g}" for k, v in constraints.items()
        )
        winner.tradeoff = _tradeoff(
            winner.raw, reference.raw, winner.predictions, winner.roles
        )
        return SolverAnswer(
            ok=True,
            goal=goal,
            goal_b=goal_b,
            mode=mode,
            candidates=[winner],
            reference=reference,
            caveats=caveats,
            reasons=[],
            rates=rates,
            model_units=model.units,
            target_context=target_context,
            anchors=anchors,
        )

    # -- Pareto front ------------------------------------------------------
    assert goal_b is not None

    def dominated(c: Candidate, others: List[Candidate]) -> bool:
        for o in others:
            if o is c:
                continue
            ge = value_of(o, goal) >= value_of(c, goal) and value_of(
                o, goal_b
            ) >= value_of(c, goal_b)
            gt = value_of(o, goal) > value_of(c, goal) or value_of(
                o, goal_b
            ) > value_of(c, goal_b)
            if ge and gt:
                return True
        return False

    seen: Dict[Tuple[int, ...], Candidate] = {}
    for c in pool:
        seen.setdefault(tuple(c.units), c)
    uniq = list(seen.values())
    front = [c for c in uniq if not dominated(c, uniq)]
    front.sort(key=lambda c: (value_of(c, goal), value_of(c, goal_b)))
    if not front:
        front = [reference]

    picks: List[Candidate] = []
    if len(front) <= front_size:
        picks = list(front)
    else:
        pts = [(value_of(c, goal), value_of(c, goal_b)) for c in front]
        k = _knee(pts)
        idx = {0, len(front) - 1, k}
        while len(idx) < front_size:
            gap_at = max(
                (i for i in range(1, len(front)) if i not in idx),
                key=lambda i: abs(pts[i][0] - pts[i - 1][0]),
                default=None,
            )
            if gap_at is None:
                break
            idx.add(gap_at)
        picks = [front[i] for i in sorted(idx)]
        knee_vec = tuple(front[k].units)
        for c in picks:
            if tuple(c.units) == knee_vec:
                c.label = "knee"
    for i, c in enumerate(picks):
        if not c.label:
            if i == 0:
                c.label = f"endpoint: max {GOAL_LABELS[goal_b]}"
            elif i == len(picks) - 1:
                c.label = f"endpoint: max {GOAL_LABELS[goal]}"
            else:
                c.label = "front point"
        c.tradeoff = _tradeoff(c.raw, reference.raw, c.predictions, c.roles)

    caveats.append(
        "the front is the non-dominated subset of a BOUNDED candidate set "
        "(both single-goal optima, the segment between them, and a coarse "
        "simplex grid) — it is not a proof of global optimality"
    )
    if len(front) < 2:
        caveats.append(
            f"one key is best at BOTH {GOAL_LABELS[goal]} and "
            f"{GOAL_LABELS[goal_b]} on this rig — the front is a single "
            "point, so there is no trade to make between these two goals "
            "here. Nothing was sacrificed and nothing has to be chosen."
        )
    for g in (goal, goal_b):
        if len({round(value_of(c, g), 6) for c in uniq}) <= 1:
            caveats.append(
                f"{GOAL_LABELS[g]} takes the SAME value at EVERY representable "
                "key on this rig under these budgets — there is nothing to "
                "trade against it, so the front degenerates to the other "
                "goal's optimum. That is a result of the geometry (module "
                "docstring §2), not a failure of the search."
            )
    return SolverAnswer(
        ok=True,
        goal=goal,
        goal_b=goal_b,
        mode="pareto",
        candidates=picks,
        reference=reference,
        caveats=caveats,
        reasons=[],
        rates=rates,
        model_units=model.units,
        target_context=target_context,
        anchors=anchors,
    )


def _enumerate_roles(n: int, model: KeyCostModel) -> List[Tuple[str, ...]]:
    """Role vectors worth solving.

    Rules, all of them structural rather than preferential:
      * at least one shard rank must exist (someone has to hold the MLP);
      * a ``replica`` is only offered when the WHOLE model fits in that
        rank's budget — the ledger sorts it out, no rule excludes it;
      * the all-shard vector is always first, so a rig with nothing to gain
        from roles gets the plain answer.
    """
    out: List[Tuple[str, ...]] = [tuple(["shard"] * n)]
    total_bytes = sum(f.bytes for f in model.perf.families.values() if f.params > 0)
    replica_ok = [
        (model.perf.budgets_mib[r] * 2**20)
        > total_bytes + FIXED_PROCESS_POST_MIB * 2**20
        for r in range(n)
    ]
    if n > MAX_ROLE_SEARCH_RANKS:
        return out
    choices: List[List[str]] = []
    for r in range(n):
        opts = ["shard", "kv_donor"]
        if replica_ok[r]:
            opts.append("replica")
        choices.append(opts)

    def rec(r: int, acc: List[str]) -> None:
        if r == n:
            if "shard" in acc and tuple(acc) not in out:
                out.append(tuple(acc))
            return
        for o in choices[r]:
            rec(r + 1, acc + [o])

    rec(0, [])
    return out


def _state_bound_sessions(
    plan_inputs, base_plan: Sequence[int], budgets_mib: Sequence[int]
) -> Optional[int]:
    """Sessions the GDN state pool admits, from the mrr geometry. ``None``
    for a non-hybrid checkpoint (no state pool -> no bound)."""
    try:
        from sglang.srt.planner import mrr_balance

        report = mrr_balance.balance_report(
            plan_inputs, list(base_plan), list(budgets_mib)
        )
    except Exception:
        return None
    if report is None:
        return None
    return int(
        mrr_balance.admitted_sessions(int(plan_inputs.max_running_requests or 16))
    )


@dataclasses.dataclass
class SolverAnswer:
    """The solver's answer, JSON-ready."""

    ok: bool
    goal: str
    goal_b: Optional[str]
    mode: str
    candidates: List[Candidate]
    reference: Candidate
    caveats: List[str]
    reasons: List[str]
    rates: RigRates
    model_units: int
    target_context: int
    anchors: _Anchors

    def to_json(self) -> dict:
        return {
            "ok": self.ok,
            "goal": self.goal,
            "goal_b": self.goal_b,
            "mode": self.mode,
            "goals": [
                {
                    "key": g,
                    "label": GOAL_LABELS[g],
                    "unit": GOAL_UNITS[g],
                    "higher_is_better": GOAL_HIGHER_IS_BETTER[g],
                }
                for g in GOALS
            ],
            "goal_interfaces": dict(GOAL_INTERFACES),
            "roles": [dataclasses.asdict(r) for r in ROLES],
            "mlp_units": self.model_units,
            "target_context_tokens": self.target_context,
            "anchor": {
                "decode_tok_s": self.anchors.decode_tok_s,
                "prefill_tok_s": self.anchors.prefill_tok_s,
                "provenance": self.anchors.provenance,
                "source": self.anchors.source,
            },
            "rates_basis": dict(self.rates.basis),
            "absent": list(self.rates.absent),
            "reference": self.reference.to_json(),
            "candidates": [c.to_json() for c in self.candidates],
            "caveats": list(self.caveats),
            "reasons": list(self.reasons),
        }


# ---------------------------------------------------------------------------
# Additive multi-instance mechanics (§7)
# ---------------------------------------------------------------------------
#
# ONE mechanic, no per-family logic. Everything that adds a serving source to
# a rig — an extra solo server beside the main group, a PD prefill lane, a
# replicated drafter, a full replica, a foreign-rig prefill satellite — is an
# INSTANCE. The rig's aggregate throughput is the SUM over the instances, and
# it counts only if the instances jointly fit:
#
#     aggregate_X = sum_i X_i          IF the coexistence bracket closes
#     bracket:  for every physical GPU g,
#               sum_i resident_i(g)  <=  total(g) - reserve(g)
#
# ``resident_i(g)`` is the #260 co-residence computation: the instance's
# weights on that card, its state/mamba pool, the modelled per-rank overhead,
# a minimum KV pool (a server with no KV is not a serving source), and — for
# every instance beyond the first on that card — the fixed process post
# (CUDA context, graphs, activations; the 28.5 GiB lesson).
#
# The bracket closes or it does not. When it does not, the overflow is named
# per GPU in MiB and the aggregate is NOT reported — an aggregate over
# instances that cannot coexist is the exact number that made "two servers"
# look free.


@dataclasses.dataclass(frozen=True)
class InstanceSpec:
    """One serving source occupying part of the rig.

    The same type carries every family: a TP group is one instance over
    several ranks, an extra solo server is a one-rank instance, a prefill
    satellite on another rig is an instance whose ``rank_gpu_id`` names no
    local card at all (``local=False``), a replicated drafter is an instance
    with its own small model path. Nothing about the family is encoded here
    beyond ``kind``, which is a label for the report.
    """

    key: str
    model_path: str
    tp_size: int
    rank_gpu_id: List[int]
    budgets_mib: List[int]
    base_plan: Optional[List[int]] = None
    mlp_vector: Optional[List[int]] = None
    kv_cache_dtype: str = "auto"
    speculative_algorithm: Optional[str] = None
    speculative_num_draft_tokens: Optional[int] = None
    max_running_requests: Optional[int] = None
    #: Free label: "serving" | "prefill_lane" | "drafter" | "satellite" | ...
    kind: str = "serving"
    #: False for an instance that lives on another rig: it adds throughput
    #: and claims none of this rig's VRAM.
    local: bool = True
    #: Minimum KV pool that makes this instance a serving source at all.
    min_kv_tokens: int = 4096
    #: Instances that REUSE each other's weight bytes name the same share
    #: group. This is the dual-group runtime (PD with rank reuse, nested
    #: shards): the lane and the group are two schedulers over ONE resident
    #: weight set on the cards they overlap on. Within a share group the
    #: bracket counts weights ONCE per GPU (see ``coexistence``); pools,
    #: overhead and the process post stay per instance, because those really
    #: are duplicated. ``None`` = naive duplication, weights counted twice.
    share_group: Optional[str] = None
    #: Priority class under co-residence (PRIO-Nachtrag 5, generalized from
    #: "PD beats Main" to ordered classes over N lanes by Nachtrag 8d).
    #: LOWER is more protected: class 0 is served first and in full, every
    #: higher class is a work-conserving scavenger over what the classes
    #: below it left. Lanes of the SAME class share equally, which is why
    #: the default (everything 0) is exactly the even split the two-lane
    #: mapping already made.
    priority_class: int = 0


@dataclasses.dataclass
class InstanceEstimate:
    """What one instance contributes and what it claims."""

    key: str
    kind: str
    local: bool
    feasible: bool
    reasons: List[str]
    prefill_tok_s: Optional[float]
    decode_tok_s: Optional[float]
    max_kv_tokens: Optional[float]
    #: physical GPU -> WEIGHT MiB resident there. Kept apart from the rest
    #: because weights are the only post a share group may count once.
    weights_mib: Dict[int, float]
    #: physical GPU -> every other resident MiB (state pool, overhead,
    #: minimum KV). Always per instance, never shared.
    other_mib: Dict[int, float]
    #: The per-post breakdown, so an overflow can be argued with.
    posts_mib: Dict[str, float]
    #: Share group this instance's weights belong to (None = its own).
    share_group: Optional[str] = None

    @property
    def resident_mib(self) -> Dict[int, float]:
        """Total claim per GPU, ignoring sharing (the naive reading)."""
        gpus = set(self.weights_mib) | set(self.other_mib)
        return {
            g: self.weights_mib.get(g, 0.0) + self.other_mib.get(g, 0.0) for g in gpus
        }

    def to_json(self) -> dict:
        return {
            "key": self.key,
            "kind": self.kind,
            "local": self.local,
            "feasible": self.feasible,
            "reasons": list(self.reasons),
            "prefill_tok_s": self.prefill_tok_s,
            "decode_tok_s": self.decode_tok_s,
            "max_kv_tokens": self.max_kv_tokens,
            "share_group": self.share_group,
            "weights_mib": {str(k): round(v, 1) for k, v in self.weights_mib.items()},
            "other_mib": {str(k): round(v, 1) for k, v in self.other_mib.items()},
            "resident_mib": {str(k): round(v, 1) for k, v in self.resident_mib.items()},
            "posts_mib": {k: round(v, 1) for k, v in self.posts_mib.items()},
        }


def estimate_instance(
    spec: InstanceSpec,
    probe: dict,
    *,
    prefill_tokens: int = 20000,
) -> InstanceEstimate:
    """Size one instance: what it contributes, and what it holds where."""
    from sglang.srt.uneven_perf import PlanInputs

    reasons: List[str] = []
    budgets = list(spec.budgets_mib)
    base_plan = list(spec.base_plan or budgets)
    inputs = PlanInputs(
        tp_size=spec.tp_size,
        model_path=spec.model_path,
        kv_cache_dtype=spec.kv_cache_dtype,
        speculative_algorithm=spec.speculative_algorithm,
        speculative_num_draft_tokens=spec.speculative_num_draft_tokens,
        max_running_requests=spec.max_running_requests,
        rank_gpu_id=list(spec.rank_gpu_id),
        effective_vram_mib=budgets,
    )
    rates = rates_from_probe(probe, spec.rank_gpu_id)
    model = build_cost_model(inputs, base_plan, budgets, rates)
    units = model.perf.mlp_unit_partition(list(spec.mlp_vector or base_plan))

    cap = model.capacity(units, [0.0] * spec.tp_size)
    weights = model.weight_bytes(units)
    from sglang.srt import uneven_perf as _up

    overhead_mib = float(_up._PREDICT_OVERHEAD_MIB + _up._PREDICT_MAMBA_ACT_RESERVE_MIB)
    min_kv_mib = spec.min_kv_tokens * float(model.perf.kv_cell_bytes) / 2**20

    weights_mib: Dict[int, float] = {}
    other_mib: Dict[int, float] = {}
    posts = {
        "weights_mib": 0.0,
        "state_pool_mib": 0.0,
        "overhead_mib": 0.0,
        "min_kv_mib": 0.0,
    }
    if spec.local:
        for r, gpu in enumerate(spec.rank_gpu_id):
            w = weights[r] / 2**20
            s = model.perf.mamba_pool_bytes[r] / 2**20
            g = int(gpu)
            weights_mib[g] = weights_mib.get(g, 0.0) + w
            other_mib[g] = other_mib.get(g, 0.0) + s + overhead_mib + min_kv_mib
            posts["weights_mib"] += w
            posts["state_pool_mib"] += s
            posts["overhead_mib"] += overhead_mib
            posts["min_kv_mib"] += min_kv_mib

    t_pre = model.prefill_seconds(units, prefill_tokens)
    prefill_tok_s = (prefill_tokens / t_pre) if t_pre else None
    if t_pre is None:
        reasons.append(
            "no pair matrix for this instance's cards — its prefill rate "
            "cannot be predicted and it therefore cannot enter an aggregate"
        )

    anchors = _load_anchors(spec.model_path, spec.tp_size)
    decode_tok_s: Optional[float] = None
    if anchors.decode_tok_s:
        base_units = model.perf.mlp_unit_partition(base_plan)
        f = model.perf.calibration.nonweight_fraction
        tw = model.decode_weight_time(units)
        tw_b = model.decode_weight_time(base_units)
        ratio = f + (1.0 - f) * (tw / tw_b if tw_b else 1.0)
        decode_tok_s = anchors.decode_tok_s / ratio if ratio else None
    else:
        reasons.append(
            "no measured decode baseline for this instance's checkpoint — its "
            "decode contribution is absent, not estimated"
        )

    if not cap["feasible"]:
        reasons.append(
            "this instance alone leaves at least one of its ranks without a "
            "usable KV pool"
        )

    return InstanceEstimate(
        key=spec.key,
        kind=spec.kind,
        local=spec.local,
        feasible=bool(cap["feasible"]),
        reasons=reasons,
        prefill_tok_s=prefill_tok_s,
        decode_tok_s=decode_tok_s,
        max_kv_tokens=(float(cap["ctx"]) if cap["feasible"] else None),
        weights_mib=weights_mib,
        other_mib=other_mib,
        posts_mib=posts,
        share_group=spec.share_group,
    )


def coexistence(
    estimates: Sequence[InstanceEstimate],
    gpu_total_mib: Dict[int, int],
    *,
    reserve_mib: Optional[Dict[int, int]] = None,
    process_post_mib: float = FIXED_PROCESS_POST_MIB,
    shared_process: bool = False,
) -> dict:
    """The #260 bracket: do these instances jointly fit?

    Per physical GPU::

        claimed(g) = SUM_i other_i(g)                       # pools, overhead, min KV
                   + SUM_{share groups G} MAX_{i in G} weights_i(g)
                   + (processes_on(g) - 1) * process_post

    Two rules, and the second is the dual-group runtime:

    * everything that is genuinely duplicated is SUMMED — each instance
      brings its own state pool, its own overhead and its own KV pool, and
      every instance beyond the first on a card pays the fixed process post
      (a second engine process brings its own CUDA context, graph pool and
      activation scratch: the 1.5-3 GiB that decided the 28.5 GiB case);

    * WEIGHTS inside one share group are counted ONCE per card, as the
      maximum rather than the sum. That is what rank reuse is: the PD lane
      and the main group are two schedulers over ONE resident weight set,
      the lane's nested complementary shard plus the shared tp1 shard being
      the same bytes the group already holds. The union of two nested shard
      sets is the larger of them, so ``max`` is the union, not an optimistic
      stand-in for it. An instance with ``share_group=None`` shares with
      nobody and is counted on its own — naive duplication stays priced as
      naive duplication.

    ``shared_process`` switches the third term off. Two lanes of ONE engine
    process (the dual-group runtime) share the CUDA context, the graph pool
    and the activation scratch, so charging a second post would invent
    1.5-3 GiB per shared card that nobody allocates. Two independent servers
    do pay it, which is the default. This is a property of the runtime, not
    of the plan, so it is a parameter and not a guess: on the FP8
    tp2-in-tp3 evaluation it moved every cell of the feasibility sweep by
    3072 MiB, i.e. it decides three of its corners.

    Returns the verdict AND the per-GPU arithmetic, so a "does not fit" can
    be argued with instead of believed.
    """
    reserve = dict(reserve_mib or {})
    gpus: set = set()
    for est in estimates:
        if est.local:
            gpus |= set(est.weights_mib) | set(est.other_mib)

    rows: List[dict] = []
    fits = True
    for gpu in sorted(gpus):
        here = [
            e
            for e in estimates
            if e.local and (e.weights_mib.get(gpu) or e.other_mib.get(gpu))
        ]
        other_total = sum(e.other_mib.get(gpu, 0.0) for e in here)
        groups: Dict[str, float] = {}
        shared_note: List[str] = []
        for e in here:
            gkey = e.share_group or f"__own__{e.key}"
            w = e.weights_mib.get(gpu, 0.0)
            if gkey in groups:
                shared_note.append(e.key)
            groups[gkey] = max(groups.get(gkey, 0.0), w)
        weights_total = sum(groups.values())
        extra_processes = 0 if shared_process else max(len(here) - 1, 0)
        posts = extra_processes * process_post_mib
        claimed = other_total + weights_total + posts
        total = float(gpu_total_mib.get(gpu, 0))
        avail = total - float(reserve.get(gpu, 0))
        over = claimed - avail
        if over > 0:
            fits = False
        breakdown = " + ".join(
            f"{e.key} {e.weights_mib.get(gpu, 0.0):.0f} w / "
            f"{e.other_mib.get(gpu, 0.0):.0f} other"
            for e in here
        )
        shared_saving = sum(e.weights_mib.get(gpu, 0.0) for e in here) - weights_total
        rows.append(
            {
                "gpu": gpu,
                "instances": [e.key for e in here],
                "weights_mib": round(weights_total, 1),
                "other_mib": round(other_total, 1),
                "process_posts_mib": round(posts, 1),
                "shared_process": bool(shared_process),
                "lanes": len(here),
                "shared_weight_saving_mib": round(shared_saving, 1),
                "claimed_mib": round(claimed, 1),
                "available_mib": round(avail, 1),
                "headroom_mib": round(-over, 1),
                "fits": over <= 0,
                "note": (
                    (
                        f"GPU {gpu} is over by {over:.0f} MiB: {breakdown}"
                        if over > 0
                        else f"GPU {gpu} has {-over:.0f} MiB of headroom left"
                    )
                    + (
                        f"; {shared_saving:.0f} MiB of weights counted once "
                        f"(rank reuse: {', '.join(sorted(set(shared_note)))})"
                        if shared_saving > 0.5
                        else ""
                    )
                    + (
                        f"; {extra_processes} extra process post(s) {posts:.0f} MiB"
                        if extra_processes
                        else ""
                    )
                ),
            }
        )
    return {"fits": fits, "per_gpu": rows}


@dataclasses.dataclass
class CoresidentPlan:
    """The co-residence budget mapping, with the priority arithmetic shown."""

    #: lane key -> per-rank budget MiB (the thing the sizing model consumes).
    budgets: Dict[str, List[int]]
    #: physical GPU -> what each class asked for and what it got.
    per_gpu: List[dict]
    #: Lanes whose class was reached with nothing left to give. They are NOT
    #: assigned a made-up number; they keep their own resident footprint and
    #: are named here so the caller can report the cell ``absent``.
    starved: List[str]
    notes: List[str]

    def to_json(self) -> dict:
        return {
            "budgets": {k: list(v) for k, v in self.budgets.items()},
            "per_gpu": list(self.per_gpu),
            "starved": list(self.starved),
            "notes": list(self.notes),
        }


def _award_leftover(
    here: Sequence[InstanceEstimate],
    want: Dict[str, float],
    klass: Dict[str, int],
    leftover: float,
) -> Tuple[Dict[str, float], List[dict], List[str]]:
    """Split one card's leftover over the priority classes on it.

    Ascending class order; a class is served in FULL before the next one sees
    anything, which is what "guaranteed" means when the guarantee has to
    survive N lanes rather than two. Within a class the split is by what each
    lane asked for, and the residue after every class is satisfied goes to the
    LEAST protected non-empty class — a scavenger's whole job is to use what
    is lying idle, and handing it back to a protected lane that already has
    its budget would be inventing demand.

    Equal classes reduce to the even split of the two-lane mapping, and that
    is deliberate: no rig that does not USE priority classes may see its
    numbers move.
    """
    rows: List[dict] = []
    starved: List[str] = []
    out: Dict[str, float] = {e.key: 0.0 for e in here}
    classes = sorted({klass[e.key] for e in here})
    if len(classes) <= 1:
        n = len(here)
        for e in here:
            out[e.key] = leftover / n if n else 0.0
        rows.append(
            {
                "class": classes[0] if classes else 0,
                "lanes": [e.key for e in here],
                "wanted_mib": round(sum(want[e.key] for e in here), 1),
                "granted_mib": round(leftover, 1),
                "policy": "even split (one priority class on this card)",
            }
        )
        return out, rows, starved

    remaining = float(leftover)
    served: List[int] = []
    for c in classes:
        members = [e for e in here if klass[e.key] == c]
        need = sum(want[e.key] for e in members)
        if remaining <= 0.0:
            for e in members:
                starved.append(e.key)
            rows.append(
                {
                    "class": c,
                    "lanes": [e.key for e in members],
                    "wanted_mib": round(need, 1),
                    "granted_mib": 0.0,
                    "policy": (
                        "absent: every MiB of this card was taken by the "
                        f"protected class(es) {served}"
                    ),
                }
            )
            continue
        if need <= remaining:
            for e in members:
                out[e.key] += want[e.key]
            remaining -= need
            policy = "granted in full (protected)"
        else:
            if need > 0:
                for e in members:
                    out[e.key] += remaining * want[e.key] / need
            else:
                for e in members:
                    out[e.key] += remaining / len(members)
            policy = (
                "partial: the class asked for more than the classes below it "
                "left, so it is split in proportion to the requests"
            )
            need, remaining = min(need, remaining), 0.0
        served.append(c)
        rows.append(
            {
                "class": c,
                "lanes": [e.key for e in members],
                "wanted_mib": round(sum(want[e.key] for e in members), 1),
                "granted_mib": round(need, 1),
                "policy": policy,
            }
        )
    if remaining > 0.0:
        last = [e for e in here if klass[e.key] == classes[-1]]
        for e in last:
            out[e.key] += remaining / len(last)
        rows.append(
            {
                "class": classes[-1],
                "lanes": [e.key for e in last],
                "wanted_mib": 0.0,
                "granted_mib": round(remaining, 1),
                "policy": (
                    "residue: every class had its request met, the rest goes "
                    "to the least protected class rather than lying idle"
                ),
            }
        )
    return out, rows, starved


def coresident_budget_plan(
    specs: Sequence[InstanceSpec],
    estimates: Sequence[InstanceEstimate],
    gpu_total_mib: Dict[int, int],
    *,
    reserve_mib: Optional[Dict[int, int]] = None,
    process_post_mib: float = FIXED_PROCESS_POST_MIB,
    shared_process: bool = False,
) -> Optional[CoresidentPlan]:
    """Per-lane, per-rank budgets under co-residence (the #260 mapping).

    ``estimate_instance`` sizes a lane's KV capacity against the budget the
    caller passed for it, which is that lane's share of the card *as if it
    were alone*. Two lanes sharing a card then both count the same free
    bytes, and summing their capacities over-states the rig by whatever the
    overlap is worth — measured at 3.3x and 4.8x on the FP8 tp2-in-tp3
    candidates, 1.14M tokens against a real 240-343k. That is the bug this
    function exists to remove.

    The mapping, per physical GPU ``g``::

        leftover(g) = total(g) - reserve(g)
                    - SUM_{share groups} MAX weights(g)   # shared bytes once
                    - SUM_lanes other(g)                  # pools, overhead
                    - process posts
        budget_i(g) = weights_i(g) + other_i(g) + award_i(g)

    A lane keeps what it actually holds and is awarded a share of what is
    left. The award is where the PRIORITY CLASSES enter (Nachtrag 5, in the
    N-lane form of Nachtrag 8d): the classes on a card are served in
    ascending order, each in full before the next sees anything, so a
    protected class gets the budget it asked for and the scavengers divide
    the remainder. With ONE class on a card — which is every configuration
    that does not use the feature — the award is the even split, unchanged.

    Two properties are worth being explicit about, because they decide
    how far the result can be trusted:

    * The SPLIT of the leftover is a POLICY, not a derivation — even between
      equals, waterfall between classes. But the sum of the budgets is fixed
      regardless of the split, so the per-rank capacity TOTAL is
      split-invariant; only its division between lanes is a choice. The
      usable-context figure is not strictly invariant, because
      ``min(sum P, 64 x min P)`` is not linear, and the caller is told so.
      Priority therefore moves capacity BETWEEN lanes; it does not create
      any.
    * A lane whose weights are the smaller half of a share group is charged
      only its OWN weights here, which is correct: the bytes it reuses were
      already paid for by the group, and the saving flows into ``leftover``
      where both lanes get a share of it.
    * Consequently the budgets do NOT sum to the card. They sum to
      ``available - posts + shared_weight_saving``, because the shared bytes
      are handed to every lane that reads them — each lane's capacity model
      subtracts its own weight view, so each has to be given it, and the
      physical books stay straight because ``leftover`` charged those bytes
      only once. Reading the total as "available minus the saving" is the
      natural mistake and gets the sign backwards.

    Returns ``None`` when the mapping cannot be built — an overflowing card
    (a negative leftover has no honest division) or a lane with no local
    footprint at all. The caller then reports the KV cell ``absent`` rather
    than dividing a number it does not have. A lane that is merely STARVED
    (its class was reached with nothing left) is a different case: it is
    named in ``starved`` and keeps its own footprint, because "this class
    got nothing on this card" is a finding, not a hole.
    """
    reserve = dict(reserve_mib or {})
    by_key = {e.key: e for e in estimates}
    if len(by_key) != len(estimates):
        return None  # duplicate keys: the mapping would be ambiguous

    klass = {s.key: int(s.priority_class) for s in specs}
    lanes_on: Dict[int, List[InstanceEstimate]] = {}
    #: physical GPU -> lane key -> awarded MiB out of that card's leftover.
    award: Dict[int, Dict[str, float]] = {}
    per_gpu: List[dict] = []
    starved: List[str] = []
    notes: List[str] = []
    gpus: set = set()
    for est in estimates:
        if est.local:
            gpus |= set(est.weights_mib) | set(est.other_mib)
    for g in sorted(gpus):
        here = [
            e
            for e in estimates
            if e.local and (e.weights_mib.get(g) or e.other_mib.get(g))
        ]
        if not here:
            continue
        lanes_on[g] = here
        groups: Dict[str, float] = {}
        for e in here:
            gk = e.share_group or f"__own__{e.key}"
            groups[gk] = max(groups.get(gk, 0.0), e.weights_mib.get(g, 0.0))
        posts = 0.0 if shared_process else max(len(here) - 1, 0) * process_post_mib
        avail = float(gpu_total_mib.get(g, 0)) - float(reserve.get(g, 0))
        left = (
            avail
            - sum(groups.values())
            - sum(e.other_mib.get(g, 0.0) for e in here)
            - posts
        )
        if left < 0:
            return None  # does not fit; there is nothing to divide
        # What each lane ASKED for on this card, beyond what it already
        # holds there: the sum of the solo budgets of its ranks on g, minus
        # its own resident footprint. Never negative — a lane whose solo
        # budget is already below its footprint is asking for nothing extra,
        # not for a refund.
        want: Dict[str, float] = {}
        by_spec = {s.key: s for s in specs}
        for e in here:
            spec = by_spec.get(e.key)
            floor = e.weights_mib.get(g, 0.0) + e.other_mib.get(g, 0.0)
            asked = 0.0
            if spec is not None:
                asked = sum(
                    float(spec.budgets_mib[r])
                    for r, x in enumerate(spec.rank_gpu_id)
                    if int(x) == g and r < len(spec.budgets_mib)
                )
            want[e.key] = max(asked - floor, 0.0)
        got, rows, card_starved = _award_leftover(
            here, want, {e.key: klass.get(e.key, 0) for e in here}, left
        )
        award[g] = got
        starved.extend(f"{k}@gpu{g}" for k in card_starved)
        per_gpu.append(
            {
                "gpu": g,
                "leftover_mib": round(left, 1),
                "lanes": len(here),
                "classes": rows,
            }
        )
        if len({klass.get(e.key, 0) for e in here}) > 1:
            notes.append(
                f"GPU {g} carries {len(here)} lanes in "
                f"{len({klass.get(e.key, 0) for e in here})} priority classes; "
                "the leftover was awarded class by class, lowest class first"
            )

    out: Dict[str, List[int]] = {}
    for spec in specs:
        found: Optional[InstanceEstimate] = by_key.get(spec.key)
        if found is None or not found.local:
            continue
        est = found
        budgets: List[int] = []
        for gpu in spec.rank_gpu_id:
            g = int(gpu)
            if not lanes_on.get(g):
                return None
            ranks_here = sum(1 for x in spec.rank_gpu_id if int(x) == g)
            share = (
                est.weights_mib.get(g, 0.0)
                + est.other_mib.get(g, 0.0)
                + award.get(g, {}).get(spec.key, 0.0)
            ) / max(ranks_here, 1)
            budgets.append(int(share))
        out[spec.key] = budgets
    if not out:
        return None
    return CoresidentPlan(budgets=out, per_gpu=per_gpu, starved=starved, notes=notes)


def coresident_budgets(
    specs: Sequence[InstanceSpec],
    estimates: Sequence[InstanceEstimate],
    gpu_total_mib: Dict[int, int],
    *,
    reserve_mib: Optional[Dict[int, int]] = None,
    process_post_mib: float = FIXED_PROCESS_POST_MIB,
    shared_process: bool = False,
) -> Optional[Dict[str, List[int]]]:
    """``coresident_budget_plan`` reduced to the budget map alone.

    The mapping every caller that does not care about the priority arithmetic
    wants, and the shape ``aggregate`` consumed before classes existed.
    """
    plan = coresident_budget_plan(
        specs,
        estimates,
        gpu_total_mib,
        reserve_mib=reserve_mib,
        process_post_mib=process_post_mib,
        shared_process=shared_process,
    )
    return plan.budgets if plan is not None else None


def _capacity_under(
    spec: InstanceSpec, probe: dict, budgets_mib: Sequence[int]
) -> Optional[float]:
    """One lane's usable context under a REPLACED budget. ``None`` when the
    lane cannot fund a KV pool at that budget (which is a finding, not a
    hole: it means the co-residence share is too small to serve)."""
    from sglang.srt.uneven_perf import PlanInputs

    b = [int(x) for x in budgets_mib]
    base_plan = list(spec.base_plan or spec.budgets_mib)
    inputs = PlanInputs(
        tp_size=spec.tp_size,
        model_path=spec.model_path,
        kv_cache_dtype=spec.kv_cache_dtype,
        speculative_algorithm=spec.speculative_algorithm,
        speculative_num_draft_tokens=spec.speculative_num_draft_tokens,
        max_running_requests=spec.max_running_requests,
        rank_gpu_id=list(spec.rank_gpu_id),
        effective_vram_mib=b,
    )
    try:
        rates = rates_from_probe(probe, spec.rank_gpu_id)
        model = build_cost_model(inputs, base_plan, b, rates)
        units = model.perf.mlp_unit_partition(list(spec.mlp_vector or base_plan))
        cap = model.capacity(units, [0.0] * spec.tp_size)
    except (ValueError, KeyError, ZeroDivisionError):
        return None
    return float(cap["ctx"]) if cap["feasible"] else None


def aggregate(
    specs: Sequence[InstanceSpec],
    probe: dict,
    gpu_total_mib: Dict[int, int],
    *,
    reserve_mib: Optional[Dict[int, int]] = None,
    prefill_tokens: int = 20000,
    shared_process: bool = False,
) -> dict:
    """Aggregate throughput of several coexisting instances.

    The whole mechanic in four lines: size each instance, close the bracket,
    and sum — or refuse to sum and say which GPU overflowed by how much.
    Every family that adds a source goes through here; there is no per-family
    branch anywhere in this function, which is the point.

    KV is the one quantity that is NOT simply summed. Throughput is additive
    because two lanes really do produce tokens in parallel (subject to the
    interference caveat every caller is told about), but capacity is not:
    lanes that share a card would each count the same free bytes. So when any
    card carries more than one lane, every lane is re-sized against its
    co-residence share (``coresident_budgets``) before the sum, and the cell
    says it was. Where that mapping cannot be built the cell is ``absent``.
    """
    estimates = [
        estimate_instance(s, probe, prefill_tokens=prefill_tokens) for s in specs
    ]
    bracket = coexistence(
        estimates,
        gpu_total_mib,
        reserve_mib=reserve_mib,
        shared_process=shared_process,
    )

    reasons: List[str] = []
    for est in estimates:
        reasons.extend(f"{est.key}: {r}" for r in est.reasons)

    cells: Dict[str, dict] = {}
    if not bracket["fits"]:
        why = "; ".join(row["note"] for row in bracket["per_gpu"] if not row["fits"])
        for key, unit in (
            ("prefill_tok_s", "tok/s"),
            ("decode_tok_s", "tok/s"),
            ("max_kv_tokens", "tokens"),
        ):
            cells[key] = _cell(
                None,
                ABSENT,
                f"these instances do not coexist on this rig — {why}",
                unit=unit,
            )
        return {
            "ok": True,
            "fits": False,
            "shares_a_card": any(r["lanes"] > 1 for r in bracket["per_gpu"]),
            "shared_process": bool(shared_process),
            "instances": [
                {
                    **e.to_json(),
                    "coresident_kv_tokens": None,
                    "coresident_budget_mib": None,
                }
                for e in estimates
            ],
            "coexistence": bracket,
            "coresident_plan": None,
            "aggregate": cells,
            "reasons": reasons,
        }

    def _sum(field: str, unit: str, what: str) -> dict:
        vals = [getattr(e, field) for e in estimates]
        if any(v is None for v in vals):
            missing = [e.key for e, v in zip(estimates, vals) if v is None]
            return _cell(
                None,
                ABSENT,
                f"{what} is absent for {', '.join(missing)}, so the sum "
                "would silently under-count — no aggregate is reported",
                unit=unit,
            )
        total = sum(float(v) for v in vals if v is not None)
        parts = ", ".join(f"{e.key} {float(v):.0f}" for e, v in zip(estimates, vals))
        return _cell(
            round(total, 1),
            ESTIMATE,
            f"additive over instances that jointly fit: {parts}",
            unit=unit,
        )

    cells["prefill_tok_s"] = _sum("prefill_tok_s", "tok/s", "a prefill rate")
    cells["decode_tok_s"] = _sum("decode_tok_s", "tok/s", "a decode rate")

    # -- KV: sum only what the lanes can hold AT THE SAME TIME --------------
    shares_a_card = any(row["lanes"] > 1 for row in bracket["per_gpu"])
    coresident: Optional[Dict[str, List[int]]] = None
    coresident_plan: Optional[CoresidentPlan] = None
    per_lane_kv: Dict[str, Optional[float]] = {
        e.key: e.max_kv_tokens for e in estimates
    }
    if not shares_a_card:
        cells["max_kv_tokens"] = _sum("max_kv_tokens", "tokens", "a KV capacity")
    else:
        coresident_plan = coresident_budget_plan(
            specs,
            estimates,
            gpu_total_mib,
            reserve_mib=reserve_mib,
            shared_process=shared_process,
        )
        coresident = coresident_plan.budgets if coresident_plan else None
        if coresident_plan is not None:
            reasons.extend(coresident_plan.notes)
            reasons.extend(
                f"{k}: its priority class was reached with nothing left on "
                "that card — it holds only its own resident footprint there"
                for k in coresident_plan.starved
            )
        if coresident is None:
            cells["max_kv_tokens"] = _cell(
                None,
                ABSENT,
                "these lanes share at least one card, so their solo KV "
                "capacities cannot be added — and the co-residence budget "
                "mapping could not be built for them (an overflowing card or "
                "a lane with no local footprint). Rather than sum numbers "
                "that each assume the whole card, no KV aggregate is "
                "reported; size the lanes one at a time",
                unit="tokens",
            )
        else:
            recomputed = {
                s.key: _capacity_under(s, probe, coresident[s.key])
                for s in specs
                if s.key in coresident
            }
            per_lane_kv = {
                e.key: recomputed.get(e.key, e.max_kv_tokens) for e in estimates
            }
            starved = [k for k, v in recomputed.items() if v is None]
            if starved:
                cells["max_kv_tokens"] = _cell(
                    None,
                    ABSENT,
                    "co-resident sizing leaves "
                    + ", ".join(starved)
                    + " without a fundable KV pool on its share of the "
                    "shared card(s) — the configuration fits in bytes but "
                    "does not serve, so there is no capacity to report",
                    unit="tokens",
                )
            else:
                total = sum(float(v) for v in recomputed.values() if v)
                solo = sum(float(e.max_kv_tokens or 0.0) for e in estimates)
                parts = ", ".join(
                    f"{k} {float(v):.0f}" for k, v in recomputed.items() if v
                )
                cells["max_kv_tokens"] = _cell(
                    round(total, 1),
                    ESTIMATE,
                    (
                        "co-resident capacity: every lane re-sized against "
                        "its share of the cards it shares (#260 mapping, "
                        "leftover split evenly), then summed — "
                        f"{parts}. The solo figures would have summed to "
                        f"{solo:.0f}, a {solo / total:.2f}x over-count, "
                        "because each lane sized its KV as if it owned the "
                        "whole card. The per-rank capacity total is "
                        "invariant under how the leftover is split; the "
                        "usable-context figure is not exactly, because "
                        "min(sum, 64 x weakest) is not linear"
                    ),
                    unit="tokens",
                )

    out_instances = []
    for e in estimates:
        row = e.to_json()
        row["coresident_kv_tokens"] = per_lane_kv.get(e.key)
        row["coresident_budget_mib"] = coresident.get(e.key) if coresident else None
        out_instances.append(row)

    return {
        "ok": True,
        "fits": True,
        "shares_a_card": shares_a_card,
        "shared_process": bool(shared_process),
        "instances": out_instances,
        "coexistence": bracket,
        "coresident_plan": (
            coresident_plan.to_json() if coresident_plan is not None else None
        ),
        "aggregate": cells,
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# The N-lane entry point (PRIO-Nachtrag 8c)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LaneTarget:
    """One lane's goal in an N-lane solve: what it is for, and where it lives.

    Everything the single-lane ``solve`` takes, plus the three things only a
    SET of lanes has: the class that decides who gets the shared card's bytes
    first, the coarser lane this one has to nest inside, and the share group
    whose weight bytes it reuses.
    """

    key: str
    plan_inputs: Any
    base_plan: Sequence[int]
    budgets_mib: Sequence[int]
    goal: str = "maxkv"
    goal_b: Optional[str] = None
    constraints: Optional[Dict[str, float]] = None
    target_context: int = 8192
    roles: Optional[Sequence[str]] = None
    search_roles: bool = True
    priority_class: int = 0
    #: Key of the coarser lane whose shards must contain this lane's.
    nests_in: Optional[str] = None
    share_group: Optional[str] = None
    kind: str = "serving"
    min_kv_tokens: int = 4096
    #: Explicit ``rank_of`` against ``nests_in`` when the card map is
    #: ambiguous (a lane with two ranks on one card).
    outer_rank_of: Optional[Sequence[Optional[int]]] = None


@dataclasses.dataclass
class LaneSolution:
    """The answer for a whole set of lanes.

    ``ok`` means the PLAN is realizable: every lane got a key, the keys
    jointly fit, and the set has a hull tree. It does not mean every cell of
    the aggregate carries a number — a lane whose priority class was reached
    with nothing left, or a checkpoint with no measured decode baseline,
    still yields ``absent`` cells, and those are findings the caller has to
    read rather than failures of the plan.
    """

    ok: bool
    #: Solve order, coarsest/least-constrained first.
    order: List[str]
    answers: Dict[str, SolverAnswer]
    keys: Dict[str, List[int]]
    specs: List[InstanceSpec]
    #: The ``aggregate`` bracket over the solved keys (None when a lane had
    #: no feasible key at all — there is nothing to bracket).
    aggregate: Optional[dict]
    hull: Optional[HullTree]
    caveats: List[str]
    reasons: List[str]

    def to_json(self) -> dict:
        return {
            "ok": self.ok,
            "order": list(self.order),
            "keys": {k: list(v) for k, v in self.keys.items()},
            "answers": {k: a.to_json() for k, a in self.answers.items()},
            "aggregate": self.aggregate,
            "hull": self.hull.to_json() if self.hull else None,
            "caveats": list(self.caveats),
            "reasons": list(self.reasons),
        }


def _lane_order(lanes: Sequence[LaneTarget]) -> Optional[List[str]]:
    """Lanes ordered so that every ``nests_in`` target comes first. ``None``
    on a cycle or a dangling reference — both are plan errors, not solver
    ones, and inventing an order would hide them."""
    keys = [t.key for t in lanes]
    if len(set(keys)) != len(keys):
        return None
    by_key = {t.key: t for t in lanes}
    order: List[str] = []
    state: Dict[str, int] = {}

    def visit(k: str) -> bool:
        if state.get(k) == 2:
            return True
        if state.get(k) == 1:
            return False
        if k not in by_key:
            return False
        state[k] = 1
        parent = by_key[k].nests_in
        if parent is not None and not visit(parent):
            return False
        state[k] = 2
        order.append(k)
        return True

    for k in keys:
        if not visit(k):
            return None
    return order


def solve_lanes(
    lanes: Sequence[LaneTarget],
    probe: dict,
    gpu_total_mib: Dict[int, int],
    *,
    reserve_mib: Optional[Dict[int, int]] = None,
    shared_process: bool = False,
    prefill_tokens: int = 20000,
    hull_probes: Optional[Sequence[HullProbe]] = None,
) -> LaneSolution:
    """Solve the key of EVERY lane of a multi-group runtime, jointly bracketed.

    The N-lane form of ``solve``. What it does, and deliberately only this:

    1. order the lanes so a lane is solved after the lane it nests inside;
    2. solve each lane with ``solve``, under the set-wise box from EVERY
       coarser lane already fixed (``nesting_bounds_over``) — so a lane that
       reuses two resident lanes is bounded by the smaller of them per card,
       not by whichever one the caller happened to pass;
    3. close the feasibility bracket over the solved keys with ``aggregate``,
       which prices the shared weight bytes once per share group, charges the
       process posts unless ``shared_process``, and hands each lane its
       priority-class share of what is left;
    4. check the whole SET for nestability with ``nesting_hull``, because
       step 2's boxes are pairwise-against-the-ancestors and that is not the
       same question.

    What it does NOT do, and this is the scope line, not an omission: it
    never invents lanes. The lane count, the card sets, the roles and the
    nesting relations come from the CALLER; the solver evaluates and
    optimizes the keys under them. Enumerating lane STRUCTURES — how many
    lanes, over which cards, in which direction spread — is the dispatcher's
    search space (Slice D) and would be a combinatorial explosion bolted onto
    a closed-form solver.

    The lanes are solved in sequence, so an earlier lane does not see the
    later ones' demand. That is a real limitation and it is named in the
    caveats: the order is the priority order, so the protected lane is the
    one solved unconstrained, which is the intended asymmetry rather than an
    accident — but a joint optimum over all lanes at once this is not.
    """
    caveats: List[str] = []
    reasons: List[str] = []
    order = _lane_order(lanes)
    if order is None:
        return LaneSolution(
            ok=False,
            order=[],
            answers={},
            keys={},
            specs=[],
            aggregate=None,
            hull=None,
            caveats=caveats,
            reasons=[
                "the lanes' nests_in relations do not form a forest (a cycle, "
                "a dangling reference or a duplicate key). No lane can be "
                "solved before the lane it must nest inside, so no order "
                "exists and none is invented"
            ],
        )
    by_key = {t.key: t for t in lanes}
    # Protected classes first among the lanes the nesting relation leaves
    # unordered: the lane whose budget is guaranteed is the one that should be
    # solved unconstrained.
    order = _stable_priority_order(order, by_key)

    answers: Dict[str, SolverAnswer] = {}
    keys: Dict[str, List[int]] = {}
    gpus_of: Dict[str, List[int]] = {
        t.key: [int(g) for g in t.plan_inputs.rank_gpu_id] for t in lanes
    }

    for k in order:
        target = by_key[k]
        outers: List[OuterLane] = []
        # Every ANCESTOR bounds the lane, not only the direct parent: the
        # boxes intersect, and a chain of three lanes over one card is the
        # first configuration where taking only the parent would be wrong.
        ancestor = target.nests_in
        walked: set = set()
        while ancestor is not None and ancestor not in walked:
            walked.add(ancestor)
            if ancestor in keys:
                rank_of = (
                    list(target.outer_rank_of)
                    if target.outer_rank_of is not None and ancestor == target.nests_in
                    else rank_map_over_cards(gpus_of[k], gpus_of[ancestor])
                )
                outers.append(
                    OuterLane(key=ancestor, units=list(keys[ancestor]), rank_of=rank_of)
                )
            ancestor = by_key[ancestor].nests_in if ancestor in by_key else None
        bounds = nesting_bounds_over(outers) if outers else None
        rates = rates_from_probe(probe, target.plan_inputs.rank_gpu_id)
        answer = solve(
            target.plan_inputs,
            list(target.base_plan),
            list(target.budgets_mib),
            rates,
            goal=target.goal,
            goal_b=target.goal_b,
            constraints=target.constraints,
            target_context=target.target_context,
            roles=target.roles,
            search_roles=target.search_roles,
            unit_bounds=bounds,
        )
        answers[k] = answer
        if outers:
            caveats.append(
                f"{k}: bounded by the resident shards of "
                + ", ".join(o.key for o in outers)
                + " (per-card minimum of their unit counts)"
            )
        if not answer.ok or not answer.candidates:
            reasons.append(
                f"{k}: no key satisfies this lane under the bounds it "
                "inherited from the lanes it nests inside — "
                + "; ".join(answer.reasons)
            )
            continue
        keys[k] = list(answer.candidates[0].units)

    specs: List[InstanceSpec] = []
    for k in order:
        if k not in keys:
            continue
        t = by_key[k]
        pi = t.plan_inputs
        specs.append(
            InstanceSpec(
                key=k,
                model_path=pi.model_path,
                tp_size=pi.tp_size,
                rank_gpu_id=list(pi.rank_gpu_id),
                budgets_mib=[int(x) for x in t.budgets_mib],
                base_plan=list(t.base_plan),
                mlp_vector=list(keys[k]),
                kv_cache_dtype=pi.kv_cache_dtype,
                speculative_algorithm=pi.speculative_algorithm,
                speculative_num_draft_tokens=pi.speculative_num_draft_tokens,
                max_running_requests=pi.max_running_requests,
                kind=t.kind,
                min_kv_tokens=t.min_kv_tokens,
                share_group=t.share_group,
                priority_class=t.priority_class,
            )
        )

    agg: Optional[dict] = None
    if len(specs) == len(lanes) and specs:
        agg = aggregate(
            specs,
            probe,
            gpu_total_mib,
            reserve_mib=reserve_mib,
            prefill_tokens=prefill_tokens,
            shared_process=shared_process,
        )
        reasons.extend(agg.get("reasons", []))

    hull: Optional[HullTree] = None
    if len(keys) == len(lanes):
        lane_keys = [
            LaneKey(
                key=k,
                ratio=tuple(int(x) for x in by_key[k].base_plan),
                gpus=tuple(gpus_of[k]),
                family_ratios=(("mlp", tuple(int(x) for x in keys[k])),),
                priority_class=by_key[k].priority_class,
                nests_in=by_key[k].nests_in,
            )
            for k in order
        ]
        probes = list(hull_probes) if hull_probes is not None else None
        if probes is None:
            units = {len(keys[k]): sum(keys[k]) for k in order}
            total = sorted(set(units.values()))
            probes = [
                HullProbe(what="MLP units", units=int(u), family="mlp") for u in total
            ]
            caveats.append(
                "the hull was checked on the MLP dimension only — pass "
                "hull_probes with the attention, GDN and vocabulary unit "
                "counts of the checkpoint to check the axes the key vector "
                "does not carry. A hull that holds on one dimension is not a "
                "hull"
            )
        hull = nesting_hull(lane_keys, probes)
        if not hull.ok:
            reasons.extend(hull.failures)

    caveats.append(
        "lanes are solved in priority order, each against the ones already "
        "fixed — not jointly. The protected lane is therefore solved "
        "unconstrained and the scavengers see its key as a bound, which is "
        "the intended asymmetry; a joint Pareto optimum over all lanes at "
        "once is not what this computes"
    )
    ok = (
        len(keys) == len(lanes)
        and (agg is not None and agg.get("fits"))
        and (hull is None or hull.ok)
    )
    return LaneSolution(
        ok=bool(ok),
        order=order,
        answers=answers,
        keys=keys,
        specs=specs,
        aggregate=agg,
        hull=hull,
        caveats=caveats,
        reasons=reasons,
    )


def _stable_priority_order(
    order: Sequence[str], by_key: Dict[str, LaneTarget]
) -> List[str]:
    """Reorder within the nesting constraint so lower priority classes come
    first. A lane never moves ahead of the lane it nests inside — the box has
    to exist before it can bound anything — so priority only breaks ties the
    nesting relation leaves open."""
    remaining = list(order)
    done: set = set()
    out: List[str] = []
    while remaining:
        ready = [
            k
            for k in remaining
            if by_key[k].nests_in is None or by_key[k].nests_in in done
        ]
        if not ready:  # cannot happen: `order` is already a valid topo order
            out.extend(remaining)
            break
        pick = min(ready, key=lambda k: (by_key[k].priority_class, order.index(k)))
        out.append(pick)
        done.add(pick)
        remaining.remove(pick)
    return out


# ---------------------------------------------------------------------------
# Regression store (§6)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RegressionAnchor:
    """One measured point the model must reproduce, with its tolerance and
    the reason that tolerance is what it is."""

    key: str
    what: str
    source: str
    #: Measured deltas in percent (throughput), candidate vs reference.
    measured: Dict[str, float]
    #: Allowed absolute deviation of the prediction, in percentage points.
    tolerance_pct: Dict[str, float]
    tolerance_reason: str


REGRESSION_ANCHORS: Tuple[RegressionAnchor, ...] = (
    RegressionAnchor(
        key="264_611_net_negative",
        what=(
            "Qwen3.6-27B-FP8, uneven TP=3 on 5090 + 2x3080: concentrating the "
            "dense MLP to 6,1,1 against the auto 2,1,1 baseline"
        ),
        source=(
            "#264 A/B, split_probe rows 'auto' (1149.6 tok/s prefill, 93.71 "
            "tok/s decode) and '6,1,1' (1244.1 / 80.9)"
        ),
        measured={"enc": +8.2, "dec": -13.7},
        tolerance_pct={"enc": 4.0, "dec": 4.0},
        tolerance_reason=(
            "the harness noise floor on this rig is 2.7-4.2 % per arm, and "
            "both arms are single cold boots, so a prediction inside 4 "
            "percentage points is inside the measurement itself. The SIGNS "
            "and the net verdict are asserted exactly — a model that gets "
            "'net negative' wrong is wrong regardless of the magnitude."
        ),
    ),
    RegressionAnchor(
        key="phasedual_252_prefill_only",
        what=(
            "same rig and checkpoint, the moderate 2,5,2 vector "
            "(+9.6 points of MLP onto the 5090) against the auto base plan"
        ),
        source=(
            "phase-dual falsifier 2026-07-28, /tmp/phasendual: arm P vs arm D, "
            "prefill +5.7 % (first pass, 3 paired windows) and +3.4 % "
            "(replicate over three documents); decode -2.5 % paired"
        ),
        measured={"enc": +4.5, "dec": -2.5},
        tolerance_pct={"enc": 4.0, "dec": 4.0},
        tolerance_reason=(
            "the two prefill passes disagree by 2.3 points (+5.7 vs +3.4) "
            "because throughput follows output content, so the measured "
            "value is the band 3.4-5.7 % and its midpoint 4.5 % is the "
            "anchor. The decode arm sat at -2.5 %, BELOW the 4.2 % noise "
            "floor, i.e. measured as zero; the requirement is therefore that "
            "the prediction stays small, not that it matches a number the "
            "measurement could not resolve."
        ),
    ),
)


@dataclasses.dataclass(frozen=True)
class AdditiveAnchor:
    """The multi-instance anchor (#107 follow-up 2, the Q3_K_M trade).

    Three things at once, because they are three faces of one mechanic:
    (a) a solo instance and a group instance priced against each other,
    (b) a coexistence bracket that must CLOSE NEGATIVE, and
    (c) an aggregate that must be the sum when the bracket does close.
    """

    source: str
    #: (a) solo vs group on the same checkpoint.
    solo_prefill_tok_s: float
    group_prefill_tok_s: float
    solo_kv_tokens: int
    group_kv_tokens: int
    #: A third arm, not used to derive anything — a pure cross-check.
    fp8_group_prefill_tok_s: float
    fp8_group_prefill_tokens: int
    #: Allowed relative deviation of an absolute predicted rate.
    rate_tolerance: float
    #: Allowed relative deviation of a predicted ratio.
    ratio_tolerance: float
    tolerance_reason: str


ADDITIVE_ANCHOR = AdditiveAnchor(
    source=(
        "INTEGRATION_R3_VALIDATION, '#107 follow-up 2 — Q3_K_M single-request "
        "trade': Qwen3.6-27B Q3_K_M GGUF, arm S = TP=1 solo 5090 "
        "(mem-fraction 0.82), arm B = TP=3 uneven DCP (reserve 3000/2700/2700), "
        "20k cold prefill; plus the FP8 TP=3 arm of the same section "
        "(1149.6 tok/s over ~26k) as an independent third point"
    ),
    solo_prefill_tok_s=3202.8,
    group_prefill_tok_s=1089.4,
    solo_kv_tokens=155830,
    group_kv_tokens=655520,
    fp8_group_prefill_tok_s=1149.6,
    fp8_group_prefill_tokens=26000,
    rate_tolerance=0.25,
    ratio_tolerance=0.35,
    tolerance_reason=(
        "the absolute prefill model has NO fitted scalar of its own — it uses "
        "the existing GEMM efficiency and takes the pair matrix at face value "
        "(COLLECTIVE_EFFICIENCY = 1.0). 25 % on a rate is what an un-fitted "
        "two-term roofline is worth, and the three arms land at -19 %, +15 % "
        "and +11 %, all one-sided in the predicted direction (the "
        "collective-bearing arms come out too fast, because the pair matrix "
        "measured one ordered pair at a time and a group collective "
        "contends). The ratio tolerance is wider still, at 35 %, because the "
        "two Q3 arms ran under DIFFERENT reserve regimes — the source says so "
        "and calls the edge conservative in the group's favour — so the "
        "measured 2.94x is itself a bound, not a point. What is asserted "
        "EXACTLY is the direction (solo prefills faster, the group holds more "
        "KV) and the coexistence verdict, neither of which has any tolerance."
    ),
)


def check_additive_regression(
    q3_gguf_path: str,
    fp8_model_dir: str,
    probe: dict,
    *,
    small_model_path: Optional[str] = None,
    anchor: AdditiveAnchor = ADDITIVE_ANCHOR,
) -> dict:
    """Re-derive the multi-instance anchor: rates, bracket, aggregate.

    (a) the solo/group prefill trade and the KV renunciation that buys it;
    (b) 27B-Q3 solo BESIDE the 27B-Q3 group — the bracket must not close;
    (c) with ``small_model_path``, a second instance that DOES fit, whose
        aggregate prefill is the sum of the two.
    """
    gpu_total = {0: 32607, 1: 20480, 2: 20480}
    solo = InstanceSpec(
        key="q3-solo-5090",
        model_path=q3_gguf_path,
        tp_size=1,
        rank_gpu_id=[0],
        budgets_mib=[int(32607 * 0.82)],
        kv_cache_dtype="fp8_e4m3",
        speculative_algorithm="NEXTN",
        speculative_num_draft_tokens=4,
        # The source records that sglang clamped the solo arm to 6.
        max_running_requests=6,
    )
    group = InstanceSpec(
        key="q3-tp3-group",
        model_path=q3_gguf_path,
        tp_size=3,
        rank_gpu_id=[0, 1, 2],
        budgets_mib=[32607 - 3000, 20480 - 2700, 20480 - 2700],
        kv_cache_dtype="fp8_e4m3",
        speculative_algorithm="NEXTN",
        speculative_num_draft_tokens=4,
        max_running_requests=16,
    )
    e_solo = estimate_instance(solo, probe, prefill_tokens=20000)
    e_group = estimate_instance(group, probe, prefill_tokens=20000)

    def dev(pred: Optional[float], meas: float) -> Optional[float]:
        return None if pred is None else (pred / meas - 1.0)

    a_rows: List[Dict[str, Any]] = [
        {
            "arm": "q3 solo 5090, 20k prefill",
            "predicted_tok_s": (
                round(e_solo.prefill_tok_s, 1) if e_solo.prefill_tok_s else None
            ),
            "measured_tok_s": anchor.solo_prefill_tok_s,
            "deviation": dev(e_solo.prefill_tok_s, anchor.solo_prefill_tok_s),
        },
        {
            "arm": "q3 TP=3 group, 20k prefill",
            "predicted_tok_s": (
                round(e_group.prefill_tok_s, 1) if e_group.prefill_tok_s else None
            ),
            "measured_tok_s": anchor.group_prefill_tok_s,
            "deviation": dev(e_group.prefill_tok_s, anchor.group_prefill_tok_s),
        },
    ]
    # Independent third arm: a different checkpoint and quant path.
    fp8_group = InstanceSpec(
        key="fp8-tp3-group",
        model_path=fp8_model_dir,
        tp_size=3,
        rank_gpu_id=[0, 1, 2],
        budgets_mib=[32607 - 3000, 20480 - 2700, 20480 - 2700],
        base_plan=[32607 - 3000, 20480 - 2700, 20480 - 2700],
        mlp_vector=[2, 1, 1],
        kv_cache_dtype="fp8_e4m3",
        speculative_algorithm="NEXTN",
        speculative_num_draft_tokens=4,
        max_running_requests=16,
    )
    e_fp8 = estimate_instance(
        fp8_group, probe, prefill_tokens=anchor.fp8_group_prefill_tokens
    )
    a_rows.append(
        {
            "arm": f"fp8 TP=3 group, {anchor.fp8_group_prefill_tokens} prefill",
            "predicted_tok_s": (
                round(e_fp8.prefill_tok_s, 1) if e_fp8.prefill_tok_s else None
            ),
            "measured_tok_s": anchor.fp8_group_prefill_tok_s,
            "deviation": dev(e_fp8.prefill_tok_s, anchor.fp8_group_prefill_tok_s),
        }
    )

    prefill_ratio = (
        e_solo.prefill_tok_s / e_group.prefill_tok_s
        if (e_solo.prefill_tok_s and e_group.prefill_tok_s)
        else None
    )
    kv_ratio = (
        e_group.max_kv_tokens / e_solo.max_kv_tokens
        if (e_group.max_kv_tokens and e_solo.max_kv_tokens)
        else None
    )
    measured_prefill_ratio = anchor.solo_prefill_tok_s / anchor.group_prefill_tok_s
    measured_kv_ratio = anchor.group_kv_tokens / anchor.solo_kv_tokens

    part_a = {
        "arms": a_rows,
        "prefill_ratio_solo_over_group": (
            round(prefill_ratio, 2) if prefill_ratio else None
        ),
        "measured_prefill_ratio": round(measured_prefill_ratio, 2),
        "kv_ratio_group_over_solo": round(kv_ratio, 2) if kv_ratio else None,
        "measured_kv_ratio": round(measured_kv_ratio, 2),
        "direction_ok": bool(
            prefill_ratio and prefill_ratio > 1.0 and kv_ratio and kv_ratio > 1.0
        ),
        "rates_within_tolerance": all(
            r["deviation"] is not None
            and abs(float(r["deviation"])) <= anchor.rate_tolerance
            for r in a_rows
        ),
        "ratios_within_tolerance": bool(
            prefill_ratio
            and kv_ratio
            and abs(prefill_ratio / measured_prefill_ratio - 1.0)
            <= anchor.ratio_tolerance
            and abs(kv_ratio / measured_kv_ratio - 1.0) <= anchor.ratio_tolerance
        ),
    }

    # (b) the two Q3 instances beside each other, twice: naive duplication
    # (must NOT fit) and the dual-group runtime with rank reuse (must fit,
    # because the 12.9 GiB of weights live on the 5090 exactly once).
    naive = aggregate([group, solo], probe, gpu_total, prefill_tokens=20000)
    reuse_group = dataclasses.replace(group, share_group="dual")
    reuse_solo = dataclasses.replace(solo, key="q3-solo-5090-reuse", share_group="dual")
    reused = aggregate(
        [reuse_group, reuse_solo], probe, gpu_total, prefill_tokens=20000
    )
    reuse_rows = {r["key"]: r for r in reused["instances"]}
    group_only = e_group.prefill_tok_s
    reuse_agg = reused["aggregate"]["prefill_tok_s"]["value"]
    measured_agg_ratio = (
        anchor.solo_prefill_tok_s + anchor.group_prefill_tok_s
    ) / anchor.group_prefill_tok_s
    part_b = {
        "naive": {
            "instances": ["q3-tp3-group", "q3-solo-5090"],
            "fits": naive["fits"],
            "coexistence": naive["coexistence"],
            "aggregate_reported": naive["aggregate"]["prefill_tok_s"]["available"],
        },
        "rank_reuse": {
            "instances": list(reuse_rows),
            "fits": reused["fits"],
            "coexistence": reused["coexistence"],
            "aggregate_prefill": reused["aggregate"]["prefill_tok_s"],
            "aggregate_over_group": (
                round(reuse_agg / group_only, 2) if (reuse_agg and group_only) else None
            ),
            "measured_aggregate_over_group": round(measured_agg_ratio, 2),
            "ratio_within_tolerance": bool(
                reuse_agg
                and group_only
                and abs((reuse_agg / group_only) / measured_agg_ratio - 1.0)
                <= anchor.ratio_tolerance
            ),
        },
        "verdict_ok": bool(not naive["fits"] and reused["fits"]),
    }

    # (c) a smaller second instance that does fit -> additive aggregate.
    part_c: Dict[str, Any] = {"ran": False}
    if small_model_path:
        # On the 5090, where the group's own claim leaves the headroom. That
        # is also where such a lane would really be put; the bracket is what
        # decides it, not the placement being declared "the PD case".
        small = InstanceSpec(
            key="small-pd-lane",
            model_path=small_model_path,
            tp_size=1,
            rank_gpu_id=[0],
            budgets_mib=[14000],
            kv_cache_dtype="fp8_e4m3",
            kind="prefill_lane",
            max_running_requests=8,
        )
        agg = aggregate([group, small], probe, gpu_total, prefill_tokens=20000)
        rows = {r["key"]: r for r in agg["instances"]}
        summed = None
        if agg["fits"] and all(rows[k]["prefill_tok_s"] is not None for k in rows):
            summed = sum(rows[k]["prefill_tok_s"] for k in rows)
        part_c = {
            "ran": True,
            "instances": list(rows),
            "fits": agg["fits"],
            "per_instance_prefill_tok_s": {k: rows[k]["prefill_tok_s"] for k in rows},
            "aggregate_prefill": agg["aggregate"]["prefill_tok_s"],
            "sum_matches": (
                summed is not None
                and agg["aggregate"]["prefill_tok_s"]["value"] is not None
                and abs(agg["aggregate"]["prefill_tok_s"]["value"] - summed) <= 0.5
            ),
            "coexistence": agg["coexistence"],
        }

    return {
        "source": anchor.source,
        "tolerance_reason": anchor.tolerance_reason,
        "rate_tolerance": anchor.rate_tolerance,
        "ratio_tolerance": anchor.ratio_tolerance,
        "a_trade": part_a,
        "b_coexistence": part_b,
        "c_additive": part_c,
    }


def check_regressions(
    model_dir: str,
    probe: dict,
    *,
    anchors: Sequence[RegressionAnchor] = REGRESSION_ANCHORS,
) -> List[dict]:
    """Re-derive every anchor and report predicted vs measured.

    Returns one row per anchor with the deltas and whether the prediction is
    inside the stated tolerance. Used by the tests and exposed by the
    endpoint, because a calibration nobody can see is a calibration nobody
    can refute.
    """
    from sglang.srt.uneven_perf import PlanInputs

    rows: List[dict] = []
    for anchor in anchors:
        if anchor.key == "264_611_net_negative":
            budgets = [32607 - 3000, 20480 - 2700, 20480 - 2700]
            rank_gpu_id = [0, 1, 2]
            ref_vec, cand_vec = [2, 1, 1], [6, 1, 1]
        else:
            budgets = [20480 - 3000, 32607 - 2200, 20480 - 2200]
            rank_gpu_id = [1, 0, 2]
            ref_vec, cand_vec = list(budgets), [2, 5, 2]
        pi = PlanInputs(
            tp_size=3,
            model_path=model_dir,
            kv_cache_dtype="fp8_e4m3",
            speculative_algorithm="NEXTN",
            speculative_num_draft_tokens=4,
            max_running_requests=16,
            rank_gpu_id=rank_gpu_id,
            effective_vram_mib=list(budgets),
        )
        rates = rates_from_probe(probe, rank_gpu_id)
        model = build_cost_model(pi, list(budgets), list(budgets), rates)
        f = model.perf.calibration.nonweight_fraction
        tw_ref = model.decode_weight_time(model.perf.mlp_unit_partition(ref_vec))
        tw_cand = model.decode_weight_time(model.perf.mlp_unit_partition(cand_vec))
        step_ratio = f + (1.0 - f) * (tw_cand / tw_ref)
        dec_pred = (1.0 / step_ratio - 1.0) * 100.0
        # A measured MAGNITUDE can only be checked against a prediction that
        # prices the same terms. With no pair matrix the prefill prediction
        # omits the collective, so "enc" is reported as not comparable rather
        # than compared through a stand-in link rate (#359).
        link = rates.link_bw_gbs
        gemm = rates.require_gemm_tflops()
        families = rates.gemm_family_tflops or None
        p_ref = model.perf.prefill_time_model(ref_vec, gemm, link, families)
        p_cand = model.perf.prefill_time_model(cand_vec, gemm, link, families)
        enc_pred = (p_ref / p_cand - 1.0) * 100.0
        predicted = {"enc": enc_pred, "dec": dec_pred}
        not_priced = ["enc"] if (link is None and rates.ranks > 1) else []
        comparable = [k for k in anchor.measured if k not in not_priced]
        deviations = {k: predicted[k] - anchor.measured[k] for k in comparable}
        rows.append(
            {
                "key": anchor.key,
                "what": anchor.what,
                "source": anchor.source,
                "reference_vector": ref_vec,
                "candidate_vector": cand_vec,
                "gemm_format": rates.gemm_format,
                "gemm_lanes": list(rates.gemm_lanes),
                "measured_pct": dict(anchor.measured),
                "predicted_pct": {k: round(v, 2) for k, v in predicted.items()},
                "deviation_pct": {k: round(v, 2) for k, v in deviations.items()},
                "not_comparable": not_priced,
                "tolerance_pct": dict(anchor.tolerance_pct),
                "tolerance_reason": anchor.tolerance_reason,
                "within_tolerance": all(
                    abs(deviations[k]) <= anchor.tolerance_pct[k] for k in comparable
                ),
                "signs_match": all(
                    (predicted[k] > 0) == (anchor.measured[k] > 0)
                    for k in comparable
                    if abs(anchor.measured[k]) >= NOISE_FLOOR_PCT
                ),
            }
        )
    return rows


#: The JSON/HTTP surface lives in ``sglang.srt.planner.solver_api`` so this
#: module stays a pure library: it reads nothing from disk except through the
#: caller's inputs and has no notion of an endpoint.
