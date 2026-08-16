"""Planner-solved pipeline-parallel layer cut, per layer FAMILY (#485).

Why this module exists
----------------------
``derive_pp_layer_split`` (``srt/distributed/utils.py``) turns ONE per-stage
score vector into per-stage layer counts. On a hybrid checkpoint it derives
two targets from that single vector:

    target_full   = round(n_full   * cum_score / total)   # KV / bandwidth mass
    target_layers = round(n_layers * cum_score / total)   # compute / weight mass

and then clamps ``target_layers`` into the layer window that yields exactly
``target_full`` full-attention layers on the left. Both targets come from the
same fraction, so on a period-P hybrid ``target_layers`` lands at
``P * target_full`` -- the bottom of that window -- whenever the fraction sits
near a multiple of ``1/n_full``. That is the whole origin of the "the stage
boundary can only fall on a multiple of 4" observation in the 631 bench log
(PROD_BRINGUP_BENCH.md sec. 1e): it is a property of deriving two targets from
one number, NOT a property of the hardware, the KV pool, or the flip maps.

The consequence is that the two families cannot be traded against each other.
On a period-4 hybrid, one step of the single-score lever moves FOUR layers --
three linear/GDN plus one full-attention -- so relieving a memory-bound stage
by a fraction of a layer-block is unrepresentable, and buying compute relief
always drags KV mass along with it.

This module removes that coupling. Attention mass and linear mass are cut
INDEPENDENTLY, and the cut is SOLVED against measured per-rank rates rather
than supplied by hand as a ratio.

What stays the same
-------------------
The cut is still CONTIGUOUS: stage ``i`` owns the half-open layer range
``[bounds[i-1], bounds[i])``. That is deliberate and is not a shortcut.
Non-contiguous ownership would require replacing the ``layer_id -
self.start_layer`` offset arithmetic throughout ``mem_cache/memory_pool.py``,
the ``PPMissingLayer`` padding in ``utils/common.py:make_layers``, and the
contiguity assertions in ``managers/phase_flip_runtime.py:870-874`` and
``managers/gdn_flip_mover.py:65-69``. It is also unnecessary: because a
boundary may sit anywhere INSIDE a full-attention period, a contiguous cut
already spans the useful decoupled space. On the 64-layer / period-4
reference checkpoint, holding the attention split at ``[7, 5, 4]`` admits
layer counts from ``[28, 20, 16]`` through ``[31, 17, 16]`` -- i.e. up to
three linear layers may move off a memory-bound stage at ZERO KV cost.

Objective
---------
Prefill runs in lockstep, so a stage's wall time is the pipeline's cost only
through the SLOWEST stage. The solver therefore minimizes the makespan

    T = max_i ( attention_core_seconds_i + dense_compute_seconds_i )

subject to a hard per-rank memory feasibility constraint. Both terms are
per-stage sums over the layers that stage owns, which is what makes the
family decoupling meaningful: the attention term scales with the stage's
FULL-ATTENTION layer count, while the dense term scales with ALL of its
layers.

The attention core is priced as a ROOFLINE, ``max(flops/gemm, bytes/bw)``,
and not as a bandwidth term. That is a measured correction, not a
refinement. At a prefill chunk of C query tokens against KV depth D, one
attention layer does ``4 * C * D * q_heads * head_dim`` FLOPs while reading
``D * kv_heads * head_dim * 2 * dtype_bytes`` bytes of KV, so its arithmetic
intensity is INDEPENDENT of depth and equal to
``2 * C * q_heads / (kv_heads * dtype_bytes)``. On the reference geometry
that is ~24 600 FLOP/byte at C=2048, against ridge points of 151 (5090) and
91 (3080) -- compute-bound by more than two orders of magnitude. The term
only crosses into bandwidth-bound below a chunk of ~13 tokens, i.e. in
decode, never in prefill. Pricing deep-prefill attention as
bandwidth-proportional is therefore wrong, and a cut derived that way skews
the wrong way: it apportions attention on the 2.14x memory-bandwidth spread
when the binding spread is the 3.54x bf16 GEMM spread.

The roofline is kept rather than replaced by a bare FLOP count precisely so
this module stays honest at the other end of the chunk axis -- a decode-
shaped stage flips to the bandwidth side automatically, and ``StageCost``
reports which side bound it.

This module is stdlib-only on purpose: it must stay importable and testable
without torch, like ``planner/cost_model.py``. Everything that needs the
checkpoint or the measured probe is fed in as plain numbers; the adapter that
builds those numbers from ``PerfCostModel`` lives in ``pp_cut_adapter.py``.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "LAYER_FAMILY_ATTENTION",
    "LAYER_FAMILY_LINEAR",
    "MIB",
    "RankResources",
    "PPCutInputs",
    "StageCost",
    "PPCutSolution",
    "layer_families_from_config",
    "attention_counts",
    "token_shares_from_vector",
    "solve_pp_cut",
    "validate_pp_cut",
]

#: Layer-family tags. These match the strings a hybrid checkpoint puts in
#: ``layer_types`` / ``layers_block_type`` (see ``configs/qwen3_next.py:259``),
#: so a config-derived list needs no translation.
LAYER_FAMILY_ATTENTION = "full_attention"
LAYER_FAMILY_LINEAR = "linear_attention"

MIB = 1024.0 * 1024.0


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RankResources:
    """One pipeline stage's host card, as MEASURED.

    ``attn_bw_gbs`` is the KV-read rate for the attention term. Feed it the
    probe's GEMV-shaped rate (``membw_gemv_gbs``) rather than the streaming
    number when both exist -- the decode/attention roofline uses the GEMV
    basis (``uneven_perf.PerfCostModel.decode_bw_basis``), and on a 5090 the
    streaming figure is optimistic for this access pattern.

    ``budget_mib`` is everything the stage may occupy on that card: weights,
    its KV arena, and its transient working set. ``transient_mib`` is carved
    out of it separately because the transient is a property of the REQUEST
    and the geometry, not of the pool (PROD_BRINGUP_BENCH.md sec. 1f), so it
    must not scale with the cut.

    THE TRANSIENT IS PER LOAD STATE (law 31). The paragraph above is right
    that the transient is a property of the request rather than the pool, and
    it was read for three shifts as if that made it ONE number. It does not:
    "the load" is not one state, and the same rank on the same rig drew

        deep-prefill A/B     ->   956 MiB below at-rest
        22-minute mixed soak ->  1989 MiB (planner cut) / 3148 MiB (ship)

    A gate fed the 956 admitted cuts that metal then broke the corridor on,
    twice. So supply ``transient_by_load_state`` -- a measured mapping from
    load-state name to drawn MiB, as ``planner/transient_census.py`` writes
    it -- and the gate funds the WORST state, because every state in that
    table is one the deployment will serve. The scalar ``transient_mib``
    remains for callers with a single measured state; supplying both is
    refused rather than silently reconciled.
    """

    label: str
    attn_bw_gbs: float
    gemm_tflops: float
    budget_mib: float
    transient_mib: float = 0.0
    #: Measured transient draw per load state, e.g.
    #: ``{"DECODE": 1204.0, "EXTEND": 1989.0}``. The gate charges the worst
    #: entry and names it in any refusal, so the operator learns WHICH state
    #: could not be funded rather than only that something could not.
    transient_by_load_state: Optional[Mapping[str, float]] = None
    #: Everything resident on the card that the cut does NOT move: CUDA
    #: context, the mamba/GDN state pool, graph capture, allocator
    #: fragmentation, the weights-arena high-water. Calibrate it from a
    #: measured at-rest boot -- ``resident_at_rest - weights - kv``. Left at
    #: zero the memory verdict is a LOWER BOUND on real occupancy and will
    #: read as feasible where metal is not.
    fixed_overhead_mib: float = 0.0

    #: TRANSIENT headroom this rank must still have FREE once everything
    #: above is resident, because some mechanism needs peak memory that no
    #: at-rest measurement contains. On this rig that mechanism is the #631
    #: phase flip's seam staging: at a cutover the rank holds packed send
    #: buffers, pre-allocated receive buffers and the retained local leg at
    #: once (``phase_flip_runtime._staging_bytes``).
    #:
    #: WHY THIS FIELD EXISTS AT ALL (law 23, C34). Without it this model
    #: priced weights + KV + a measured fixed overhead, declared the #485
    #: planner cut feasible with 2617 MiB to spare, and was RIGHT -- the
    #: configuration does fit at rest. It wedged anyway, because the flip
    #: wanted 4881 MiB of staging on that same rank and nothing here had a
    #: term for it. A gate that certifies a cut which then cannot run is
    #: worse than no gate: it launders an unrunnable configuration through a
    #: calibrated-looking number. "Fits at rest" and "can run" are different
    #: predicates and this field is the difference.
    #:
    #: A PER-RANK SCALAR, AND THE SHAPE IS THE OPEN QUESTION. Two demands
    #: have been measured, both at a cutover that refused:
    #:
    #:     4881 MiB   rank0, attention 10/16, pool 340000
    #:     4343 MiB   rank2, attention  6/16, pool 280000
    #:
    #: which is not enough to separate "scales with the stage's attention
    #: count" from "scales with the arena's row count" -- and successor 49's
    #: confound boot showed the demand follows the ARENA (the token vector),
    #: not the attention split, which is the opposite of what the layer-count
    #: reading predicted. So NO formula is applied here on purpose: a derived
    #: term whose mechanism was wrong last week is exactly the kind of
    #: calibrated-looking number this field exists to prevent. Supply the
    #: measured value; left at zero the verdict is a residency verdict again
    #: and says nothing about runnability.
    seam_staging_mib: float = 0.0

    def __post_init__(self) -> None:
        if self.attn_bw_gbs <= 0.0:
            raise ValueError(
                f"rank {self.label!r}: attn_bw_gbs must be > 0, got "
                f"{self.attn_bw_gbs!r}. A missing measured rate must be "
                f"refused, never defaulted -- see planner/cost_model.AbsentRate."
            )
        if self.gemm_tflops <= 0.0:
            raise ValueError(
                f"rank {self.label!r}: gemm_tflops must be > 0, got "
                f"{self.gemm_tflops!r}."
            )
        table = self.transient_by_load_state
        if table is not None:
            if not table:
                raise ValueError(
                    f"rank {self.label!r}: transient_by_load_state is empty. "
                    f"An empty table is not 'no transient' -- it is an "
                    f"unmeasured one, and an unpriced term reads as free "
                    f"memory. Pass None to mean 'use the scalar', or measure "
                    f"a state."
                )
            for state, value in table.items():
                if value < 0.0:
                    raise ValueError(
                        f"rank {self.label!r}: transient for load state "
                        f"{state!r} is negative ({value!r})."
                    )
            if self.transient_mib:
                raise ValueError(
                    f"rank {self.label!r}: both transient_mib "
                    f"({self.transient_mib!r}) and transient_by_load_state "
                    f"({dict(table)!r}) were supplied. Two sources for one "
                    f"term is how a load state gets silently swapped for "
                    f"another (law 31); pass exactly one."
                )

    @property
    def governing_load_state(self) -> Optional[str]:
        """The load state whose transient binds, when a table was measured."""
        table = self.transient_by_load_state
        if not table:
            return None
        return max(table, key=lambda k: table[k])

    @property
    def worst_transient_mib(self) -> float:
        """The transient the gate charges: the worst state that will be served.

        Not the mean and not the most recent -- a cut is admitted only if the
        WORST load state it will serve is funded, because that state arrives
        whether or not the gate priced it.
        """
        table = self.transient_by_load_state
        if not table:
            return self.transient_mib
        return max(table.values())


@dataclasses.dataclass(frozen=True)
class PPCutInputs:
    """Everything the cut is solved against.

    Per-layer masses are given per FAMILY so the solver can price a stage
    from its family census alone.
    """

    #: Per-layer family tags, index-aligned with the model's layers.
    layer_families: Tuple[str, ...]

    #: Resident weight bytes of ONE layer of each family, already at the
    #: checkpoint's bytes-per-param (so a quantized checkpoint is priced as
    #: it actually loads).
    #:
    #: MEASURE THESE FROM THE CHECKPOINT, do not derive them from the config's
    #: parameter formulas. On the reference checkpoint the formula-derived
    #: attention layer is 325.0 MiB and the real one is 355.1 MiB, because
    #: ``attn_output_gate`` adds a second q-sized projection that the formula
    #: does not know about -- 482 MiB of silent error across 16 layers, and
    #: CUT-SHAPED, so it lands on whichever stage the solver was choosing.
    attn_layer_weight_bytes: float
    linear_layer_weight_bytes: float

    #: FLOPs per prompt token for ONE layer of each family, projections and
    #: MLP included. The attention CORE (the depth-dependent part) is NOT
    #: here; it is priced separately by the roofline below.
    attn_layer_flops_per_token: float
    linear_layer_flops_per_token: float

    #: Attention-core FLOPs per (query token x KV depth token) for ONE
    #: full-attention layer: ``4 * q_heads * head_dim`` (2 for QK^T, 2 for
    #: A@V). Depth-independent by construction, which is what makes the
    #: term's arithmetic intensity depth-independent too.
    attn_core_flops_per_token_pair: float

    #: KV bytes one full-attention layer stores per token, at the serving
    #: kv-cache dtype. On the reference checkpoint: 4 kv heads x 256 head_dim
    #: x 2 (K and V) x 1 byte (fp8_e4m3) = 2048.
    kv_bytes_per_token_per_attn_layer: float

    #: The KV depth the cut is optimized FOR -- the depth of the request the
    #: pipeline is being timed on. This drives the attention CORE term only.
    kv_depth_tokens: int

    #: Prompt tokens per prefill chunk. Sets the compute term's scale.
    prefill_chunk_tokens: int

    ranks: Tuple[RankResources, ...]

    #: NON-LAYER WEIGHTS, by the stage ROLE that owns them (C38).
    #:
    #: The three fields below exist because pricing "a per-layer census of the
    #: transformer layers" is not the same as pricing the checkpoint. On the
    #: reference model these payloads are ~5.7 GiB and every one of them is
    #: bf16: a compressed-tensors quantizer only ever targets Linear modules,
    #: so the input embeddings are never a candidate, and ``lm_head`` plus the
    #: whole visual tower sit in the config's explicit ``ignore`` list. They
    #: were invisible to this ledger for three shifts, and a second error of
    #: the opposite sign (see ``tp_token_shares``) cancelled them on the
    #: shipping cut -- which is exactly why nobody caught it: the gate was
    #: only ever checked on the configuration where its errors cancel.
    #:
    #: They are ROLE-scoped, not per-rank scalars, because that is how the
    #: pipeline places them: the input embedding is resident on the FIRST
    #: stage, ``lm_head`` on the LAST, and anything replicated (the vision
    #: tower, per-process loader constants) on EVERY stage. A cut that moves
    #: the stage boundaries does not move these -- but a cut that changes
    #: which rank IS the first or last stage does, and a per-rank scalar
    #: could not express that.
    #:
    #: Measure them from the checkpoint's tensor headers, or from a boot's
    #: residency census (``planner/residency_census.py``), which reports
    #: exactly these groups. Left at zero they are simply unpriced, and the
    #: verdict is optimistic by their size.
    embedding_weight_bytes: float = 0.0
    lm_head_weight_bytes: float = 0.0
    replicated_weight_bytes: float = 0.0

    #: Resident state bytes the runtime allocates PER LINEAR (GDN/mamba)
    #: LAYER a stage owns -- the recurrent state pool, sized by the mamba
    #: cache slots and not by the KV pool.
    #:
    #: It is here rather than inside ``RankResources.fixed_overhead_mib``
    #: because it is CUT-SHAPED: moving eleven linear layers onto a stage
    #: moves this term with them. Folding it into the per-rank overhead is
    #: what made that overhead look cut-invariant when it was not (C35's
    #: card conclusion stands; its cut-invariance conclusion does not).
    #: Calibrate from the census; zero prices it away.
    state_bytes_per_linear_layer: float = 0.0

    #: Tokens the KV ARENA is sized for, i.e. ``--max-total-tokens``. This
    #: drives the memory term only, and it is a different quantity from
    #: ``kv_depth_tokens``: the arena is provisioned for the whole pool while
    #: one request reads only its own depth. Conflating them either prices a
    #: 600k-token arena as if it were one 179k request (far too optimistic on
    #: memory) or times a request as if it swept the whole pool (far too
    #: pessimistic on speed). Zero falls back to ``kv_depth_tokens``.
    kv_pool_tokens: int = 0

    #: Decode-phase TOKEN shares per rank, when this PP layout shares its KV
    #: arena with a TP layout (the #631 phase flip). The arena is then sized
    #: ``max(PP layer share, TP token share)`` per rank, because one arena
    #: backs both layouts rather than two -- the relation the 631 bench log
    #: validates to the byte against two boots (PROD_BRINGUP_BENCH.md sec. 2),
    #: and which a boot log confirms directly: every rank allocates all 16
    #: full-attention layers' rows for ITS OWN token slice, and the three
    #: slices are the token vector exactly.
    #: ``None`` for a pure-PP deployment.
    #:
    #: FEED THIS THE TOKEN VECTOR, NEVER THE FLIP WEIGHT VECTOR (C38). The
    #: #631 flip carries two different per-rank ratios and they are not
    #: interchangeable: ``--phase-flip-tp-vector`` splits the WEIGHT shard by
    #: compute, while ``SGLANG_UNEVEN_TOKEN_VECTOR`` splits the KV ARENA by
    #: each rank's remaining memory. This field sizes the ARENA, so it takes
    #: the second one. Feeding it the first is worth 547-664 MiB per rank on
    #: the reference rig, in the optimistic direction on the cut the solver
    #: prefers, and it is half of the pair of errors that cancelled on the
    #: shipping cut. Build it with :func:`token_shares_from_vector` from
    #: ``phase_flip_boot.parse_flip_token_vector`` -- one resolver, so the
    #: gate cannot drift away from what the allocator actually does.
    tp_token_shares: Optional[Tuple[float, ...]] = None

    #: How many times a chunk sweeps the stage's KV. 1.0 models a
    #: well-tiled flash kernel reading K/V once per chunk. It scales the
    #: attention term against the compute term, so it moves the balance
    #: point; it is an explicit MODELLING ASSUMPTION and the measurement arm
    #: is what falsifies it.
    kv_sweeps_per_chunk: float = 1.0

    #: Free VRAM that must remain on every card. The rig's standing corridor.
    corridor_mib: float = 1024.0

    def __post_init__(self) -> None:
        n = len(self.layer_families)
        if n == 0:
            raise ValueError("PPCutInputs: layer_families is empty.")
        bad = sorted(
            set(self.layer_families) - {LAYER_FAMILY_ATTENTION, LAYER_FAMILY_LINEAR}
        )
        if bad:
            raise ValueError(
                f"PPCutInputs: unknown layer families {bad}; expected only "
                f"{LAYER_FAMILY_ATTENTION!r} and {LAYER_FAMILY_LINEAR!r}."
            )
        if not self.ranks:
            raise ValueError("PPCutInputs: no ranks.")
        if len(self.ranks) > n:
            raise ValueError(
                f"PPCutInputs: {len(self.ranks)} stages cannot split {n} "
                f"layers (every stage needs at least one)."
            )
        if self.kv_depth_tokens < 0 or self.prefill_chunk_tokens <= 0:
            raise ValueError(
                "PPCutInputs: kv_depth_tokens must be >= 0 and "
                "prefill_chunk_tokens > 0."
            )
        if self.kv_pool_tokens < 0:
            raise ValueError("PPCutInputs: kv_pool_tokens must be >= 0.")
        if self.tp_token_shares is not None:
            if len(self.tp_token_shares) != len(self.ranks):
                raise ValueError(
                    f"PPCutInputs: tp_token_shares has "
                    f"{len(self.tp_token_shares)} entries for "
                    f"{len(self.ranks)} ranks."
                )
            if any(s < 0.0 for s in self.tp_token_shares):
                raise ValueError(
                    f"PPCutInputs: tp_token_shares must be non-negative, got "
                    f"{list(self.tp_token_shares)}."
                )

    @property
    def arena_tokens(self) -> int:
        return self.kv_pool_tokens or self.kv_depth_tokens

    @property
    def n_layers(self) -> int:
        return len(self.layer_families)

    @property
    def pp_size(self) -> int:
        return len(self.ranks)

    @property
    def n_full_attention(self) -> int:
        return sum(1 for f in self.layer_families if f == LAYER_FAMILY_ATTENTION)


# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class StageCost:
    """One stage's priced load. ``feasible`` is the hard memory verdict."""

    rank: str
    start_layer: int
    end_layer: int
    n_layers: int
    n_attention: int
    #: Attention core, priced by roofline: ``max(flops/gemm, bytes/bw)``.
    attn_seconds: float
    #: Dense projections + MLP over every layer the stage owns.
    compute_seconds: float
    weight_mib: float
    kv_mib: float
    transient_mib: float
    budget_mib: float
    #: Non-layer weights this stage's ROLE owns (embedding on the first
    #: stage, lm_head on the last, replicated payloads everywhere). Kept as
    #: its own post rather than folded into ``weight_mib`` so a future
    #: mismatch names the term that moved.
    nonlayer_weight_mib: float = 0.0
    #: Recurrent (GDN/mamba) state pool for the linear layers this stage owns.
    state_mib: float = 0.0
    #: Which side of the roofline bound the attention core: ``"compute"``,
    #: ``"bandwidth"``, or ``"none"`` when the stage owns no attention layer.
    attn_bound_by: str = "none"
    #: Peak transient this stage must be able to reach on top of residency.
    #: See ``RankResources.seam_staging_mib``.
    seam_staging_mib: float = 0.0
    #: Which measured load state's transient was charged, when the rank
    #: carried a per-load-state table. ``None`` means a single scalar was
    #: supplied and the gate cannot say which state it describes -- which is
    #: exactly the ambiguity law 31 was written about.
    transient_load_state: Optional[str] = None

    @property
    def total_seconds(self) -> float:
        return self.attn_seconds + self.compute_seconds

    @property
    def resident_mib(self) -> float:
        return (
            self.weight_mib
            + self.nonlayer_weight_mib
            + self.state_mib
            + self.kv_mib
            + self.transient_mib
        )

    @property
    def headroom_mib(self) -> float:
        """Budget left after weights, KV and the transient.

        RESIDENCY ONLY. This is what the stage does not occupy at rest; it is
        NOT what the stage can spend at a cutover. Use
        :attr:`runnable_headroom_mib` for the feasibility question.
        """
        return self.budget_mib - self.resident_mib

    @property
    def runnable_headroom_mib(self) -> float:
        """Headroom left once the peak transient is also funded.

        The quantity the verdict is actually about: a stage that fits at rest
        and cannot reach its seam is a stage that boots and then wedges.
        """
        return self.headroom_mib - self.seam_staging_mib

    @property
    def feasible(self) -> bool:
        return self.runnable_headroom_mib >= 0.0


@dataclasses.dataclass(frozen=True)
class PPCutSolution:
    counts: Tuple[int, ...]
    bounds: Tuple[int, ...]
    attention_counts: Tuple[int, ...]
    stages: Tuple[StageCost, ...]
    makespan_seconds: float
    bottleneck_stage: int
    min_headroom_mib: float
    feasible: bool
    refusals: Tuple[str, ...]
    candidates_considered: int

    def as_layer_ratio(self) -> List[int]:
        """The value ``--pp-layer-ratio`` / ``SGLANG_PP_LAYER_PARTITION`` wants."""
        return list(self.counts)

    def summary(self) -> str:
        parts = [
            f"layers={list(self.counts)}",
            f"attn={list(self.attention_counts)}",
            f"makespan={self.makespan_seconds * 1e3:.1f}ms",
            f"pacer=stage{self.bottleneck_stage}",
            f"min_headroom={self.min_headroom_mib:.0f}MiB",
        ]
        return "  ".join(parts)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def layer_families_from_config(
    config: Dict,
    num_hidden_layers: Optional[int] = None,
) -> Tuple[str, ...]:
    """Per-layer family tags from a checkpoint config dict.

    Resolution order mirrors the advisory probe in ``server_args`` and the
    model's own rule (``configs/qwen3_next.py:259-269``):

      1. an explicit ``layer_types`` / ``layers_block_type`` list;
      2. ``full_attention_interval`` -- layer ``l`` is full attention when
         ``(l + 1) % interval == 0``;
      3. no hybrid marker at all -> every layer is full attention.

    A ``text_config`` wrapper (VL checkpoints) is unwrapped first.
    """
    cfg = config.get("text_config") or config
    n = int(num_hidden_layers or cfg.get("num_hidden_layers") or 0)
    if n <= 0:
        raise ValueError(
            "layer_families_from_config: num_hidden_layers is missing or "
            "non-positive; a cut cannot be derived from an unknown depth."
        )

    explicit = cfg.get("layer_types") or cfg.get("layers_block_type")
    if explicit:
        if len(explicit) != n:
            raise ValueError(
                f"layer_families_from_config: layer_types has "
                f"{len(explicit)} entries but num_hidden_layers is {n}."
            )
        return tuple(
            (
                LAYER_FAMILY_ATTENTION
                if str(t) == LAYER_FAMILY_ATTENTION
                else LAYER_FAMILY_LINEAR
            )
            for t in explicit
        )

    interval = cfg.get("full_attention_interval")
    if interval:
        interval = int(interval)
        return tuple(
            LAYER_FAMILY_ATTENTION if (idx + 1) % interval == 0 else LAYER_FAMILY_LINEAR
            for idx in range(n)
        )

    return tuple([LAYER_FAMILY_ATTENTION] * n)


def attention_counts(
    layer_families: Sequence[str], counts: Sequence[int]
) -> Tuple[int, ...]:
    """Full-attention layers per stage, for a contiguous split ``counts``."""
    out: List[int] = []
    start = 0
    for c in counts:
        out.append(
            sum(
                1
                for f in layer_families[start : start + c]
                if f == LAYER_FAMILY_ATTENTION
            )
        )
        start += c
    return tuple(out)


def token_shares_from_vector(token_vector: Sequence[int]) -> Tuple[float, ...]:
    """Normalize a KV TOKEN vector into per-rank shares for the gate.

    The one supported way to build :attr:`PPCutInputs.tp_token_shares`. It
    exists so the arena's ratio has a single source: pass what
    ``phase_flip_boot.parse_flip_token_vector`` returns -- the resolved
    ``SGLANG_UNEVEN_TOKEN_VECTOR``, or the flip vector when that env is unset
    and the two genuinely coincide -- and never the flip WEIGHT vector read
    straight off ``--phase-flip-tp-vector``.

    Note that a gcd-reduced vector is the SAME vector: ``14,10,8`` and
    ``7,5,4`` normalize identically, which is exactly the equivalence
    ``resolve_cp_token_split`` applies. Two configurations that differ only
    there are one configuration.
    """
    vec = [float(x) for x in token_vector]
    if not vec:
        raise ValueError("token_shares_from_vector: empty token vector.")
    if any(x <= 0.0 for x in vec):
        raise ValueError(
            f"token_shares_from_vector: every rank needs a positive token "
            f"ratio, got {list(token_vector)}. A rank with ratio 0 owns no KV "
            f"rows while still holding a weight shard."
        )
    total = sum(vec)
    return tuple(x / total for x in vec)


def _prefix_attention(layer_families: Sequence[str]) -> List[int]:
    """``pref[i]`` = full-attention layers in ``[0, i)``."""
    pref = [0]
    for f in layer_families:
        pref.append(pref[-1] + (1 if f == LAYER_FAMILY_ATTENTION else 0))
    return pref


def _price_stage(
    inputs: PPCutInputs,
    stage: int,
    start: int,
    end: int,
    pref_attn: Sequence[int],
) -> StageCost:
    rank = inputs.ranks[stage]
    n_layers = end - start
    n_attn = pref_attn[end] - pref_attn[start]
    n_linear = n_layers - n_attn

    # Attention core, as a roofline over the stage's attention layers. The
    # compute side is the chunk x depth score-and-weight product; the
    # bandwidth side is one sweep of the stage's KV. Whichever is slower is
    # what the stage actually waits on. This term depends ONLY on n_attn --
    # never on n_layers -- which is exactly the coupling the family cut
    # breaks.
    kv_bytes = (
        float(n_attn)
        * inputs.kv_bytes_per_token_per_attn_layer
        * float(inputs.kv_depth_tokens)
    )
    core_flops = (
        float(n_attn)
        * float(inputs.prefill_chunk_tokens)
        * float(inputs.kv_depth_tokens)
        * inputs.attn_core_flops_per_token_pair
    )
    attn_compute_s = core_flops / (rank.gemm_tflops * 1e12)
    attn_bw_s = kv_bytes * inputs.kv_sweeps_per_chunk / (rank.attn_bw_gbs * 1e9)
    if n_attn == 0:
        attn_seconds, attn_bound_by = 0.0, "none"
    elif attn_compute_s >= attn_bw_s:
        attn_seconds, attn_bound_by = attn_compute_s, "compute"
    else:
        attn_seconds, attn_bound_by = attn_bw_s, "bandwidth"

    # Dense term: projections and MLP for the chunk over the measured rate.
    flops = float(inputs.prefill_chunk_tokens) * (
        n_attn * inputs.attn_layer_flops_per_token
        + n_linear * inputs.linear_layer_flops_per_token
    )
    compute_seconds = flops / (rank.gemm_tflops * 1e12)

    weight_mib = (
        n_attn * inputs.attn_layer_weight_bytes
        + n_linear * inputs.linear_layer_weight_bytes
    ) / MIB

    # Non-layer weights, charged to the stage ROLE that holds them: the input
    # embedding to the first stage, lm_head to the last, replicated payloads
    # (vision tower, loader constants) to every stage. These do not move when
    # the cut moves, which is precisely why leaving them out was invisible on
    # a single cut and ~3.6 GiB wrong the moment the cut changed.
    nonlayer_bytes = inputs.replicated_weight_bytes
    if stage == 0:
        nonlayer_bytes += inputs.embedding_weight_bytes
    if stage == inputs.pp_size - 1:
        nonlayer_bytes += inputs.lm_head_weight_bytes
    nonlayer_weight_mib = nonlayer_bytes / MIB

    # Recurrent-state pool: cut-shaped, one share per linear layer owned.
    state_mib = (n_linear * inputs.state_bytes_per_linear_layer) / MIB

    # KV ARENA, which is sized for the whole pool and not for one request.
    # When a TP layout shares this arena (the phase flip), the stage pays the
    # LARGER of its PP layer share and its TP token share -- one arena backs
    # both layouts, so the rig pays max(), not the sum and not the PP term
    # alone.
    n_full_total = inputs.n_full_attention
    arena_layers = float(n_attn)
    if inputs.tp_token_shares is not None and n_full_total > 0:
        arena_layers = max(
            arena_layers, inputs.tp_token_shares[stage] * float(n_full_total)
        )
    kv_mib = (
        arena_layers
        * inputs.kv_bytes_per_token_per_attn_layer
        * float(inputs.arena_tokens)
    ) / MIB

    return StageCost(
        rank=rank.label,
        start_layer=start,
        end_layer=end,
        n_layers=n_layers,
        n_attention=n_attn,
        attn_seconds=attn_seconds,
        compute_seconds=compute_seconds,
        weight_mib=weight_mib,
        nonlayer_weight_mib=nonlayer_weight_mib,
        state_mib=state_mib,
        kv_mib=kv_mib,
        # The transient and the cut-invariant overhead both occupy the card
        # alongside weights and KV, so they are charged together here. The
        # transient charged is the WORST measured load state (law 31): a cut
        # that only fits in the gentlest state it will serve does not fit.
        transient_mib=rank.worst_transient_mib + rank.fixed_overhead_mib,
        transient_load_state=rank.governing_load_state,
        seam_staging_mib=rank.seam_staging_mib,
        # The corridor is subtracted here, once, so every downstream
        # comparison is against usable bytes.
        budget_mib=rank.budget_mib - inputs.corridor_mib,
        attn_bound_by=attn_bound_by,
    )


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

#: Relative slack on the optimal makespan when the secondary objective
#: (maximize the tightest stage's headroom) picks among near-optimal cuts.
#: A cut 0.5 % slower that leaves hundreds of MiB more on the binding card is
#: the better cut on this rig -- the corridor is a hard law, the half percent
#: is inside the A-vs-A noise floor.
_MAKESPAN_SLACK = 1.005


def solve_pp_cut(
    inputs: PPCutInputs,
    *,
    require_attention_per_stage: bool = True,
    makespan_slack: float = _MAKESPAN_SLACK,
) -> PPCutSolution:
    """Exactly minimize the lockstep makespan over contiguous cuts.

    Two passes, both exact dynamic programs over stage boundaries:

      1. minimize ``max_i stage_seconds_i`` over memory-FEASIBLE cuts;
      2. among cuts within ``_MAKESPAN_SLACK`` of that optimum, maximize
         ``min_i headroom_i``.

    Splitting it this way keeps each pass a single-aggregate DP, which is
    exactly solvable; a one-pass lexicographic DP would not be, because a
    prefix that is worse on the primary key can still lead to the better
    overall cut.

    The search space is every contiguous split, so the answer is the true
    optimum over the representable space -- not a rounded ratio and not a
    local repair. ``O(pp_size * n_layers^2)`` stage prices, which is 12k
    evaluations at the reference geometry.

    ``makespan_slack`` is how much the second pass may spend on headroom;
    pass ``1.0`` to get the strictly makespan-optimal cut and nothing else.

    Infeasible input does not fall back to an even split (the #202 lesson):
    it returns ``feasible=False`` with a named reason per rank.
    """
    if makespan_slack < 1.0:
        raise ValueError(
            f"solve_pp_cut: makespan_slack must be >= 1.0, got "
            f"{makespan_slack!r} -- a value below 1 would exclude the "
            f"optimum itself."
        )
    n = inputs.n_layers
    k = inputs.pp_size
    pref_attn = _prefix_attention(inputs.layer_families)
    hybrid = 0 < inputs.n_full_attention < n

    # Price every (stage, start, end) once.
    cache: Dict[Tuple[int, int, int], StageCost] = {}

    def price(stage: int, start: int, end: int) -> StageCost:
        key = (stage, start, end)
        got = cache.get(key)
        if got is None:
            got = _price_stage(inputs, stage, start, end, pref_attn)
            cache[key] = got
        return got

    def admissible(cost: StageCost) -> bool:
        if not cost.feasible:
            return False
        if require_attention_per_stage and hybrid and cost.n_attention == 0:
            # A stage with no full-attention layer has an EMPTY KV pool.
            # Same refusal as derive_pp_layer_split (#201 slice 2).
            return False
        return True

    inf = math.inf

    # ---- pass 1: minimize the makespan -----------------------------------
    # best[s][b] = minimal achievable max-stage-seconds using stages
    # 0..s-1 to cover layers [0, b).
    best: List[List[float]] = [[inf] * (n + 1) for _ in range(k + 1)]
    best[0][0] = 0.0
    considered = 0
    for s in range(1, k + 1):
        # Stage s-1 must leave at least one layer for each remaining stage.
        hi = n - (k - s)
        for b in range(s, hi + 1):
            row = best[s - 1]
            acc = inf
            for prev in range(s - 1, b):
                base = row[prev]
                if base is inf or base >= acc:
                    continue
                cost = price(s - 1, prev, b)
                considered += 1
                if not admissible(cost):
                    continue
                cand = cost.total_seconds
                val = base if base > cand else cand
                if val < acc:
                    acc = val
            best[s][b] = acc

    if best[k][n] is inf:
        return _infeasible(inputs, pref_attn, hybrid, considered)

    optimal = best[k][n]
    ceiling = optimal * makespan_slack

    # ---- pass 2: among near-optimal cuts, maximize the tightest headroom --
    hbest: List[List[float]] = [[-inf] * (n + 1) for _ in range(k + 1)]
    choice: List[List[int]] = [[-1] * (n + 1) for _ in range(k + 1)]
    hbest[0][0] = inf
    for s in range(1, k + 1):
        hi = n - (k - s)
        for b in range(s, hi + 1):
            row = hbest[s - 1]
            acc = -inf
            arg = -1
            for prev in range(s - 1, b):
                base = row[prev]
                if base == -inf:
                    continue
                cost = price(s - 1, prev, b)
                if not admissible(cost) or cost.total_seconds > ceiling:
                    continue
                cand_h = cost.runnable_headroom_mib
                val = base if base < cand_h else cand_h
                # Deterministic tie-break: the earliest boundary wins, so the
                # same inputs always yield the same cut.
                if val > acc:
                    acc = val
                    arg = prev
            hbest[s][b] = acc
            choice[s][b] = arg

    if hbest[k][n] == -inf:
        # Cannot happen while ceiling >= optimal, but never silently guess.
        return _infeasible(inputs, pref_attn, hybrid, considered)

    bounds_rev: List[int] = []
    b = n
    for s in range(k, 0, -1):
        bounds_rev.append(b)
        b = choice[s][b]
    bounds = tuple(reversed(bounds_rev))
    starts = (0,) + bounds[:-1]
    stages = tuple(price(i, starts[i], bounds[i]) for i in range(k))
    counts = tuple(s.n_layers for s in stages)
    times = [s.total_seconds for s in stages]
    makespan = max(times)

    return PPCutSolution(
        counts=counts,
        bounds=bounds,
        attention_counts=tuple(s.n_attention for s in stages),
        stages=stages,
        makespan_seconds=makespan,
        bottleneck_stage=times.index(makespan),
        min_headroom_mib=min(s.runnable_headroom_mib for s in stages),
        feasible=True,
        refusals=(),
        candidates_considered=considered,
    )


def _infeasible(
    inputs: PPCutInputs,
    pref_attn: Sequence[int],
    hybrid: bool,
    considered: int,
) -> PPCutSolution:
    """Name WHY no cut fits, per rank, with the numbers that decide it.

    Fail-fast with the physical quantities in the message; a late OOM or a
    silent even split is what this replaces.
    """
    n = inputs.n_layers
    k = inputs.pp_size
    reasons: List[str] = []

    # The cheapest possible stage is one linear layer plus its KV share of
    # zero; if even that does not fit a rank, that rank is the reason.
    lightest_mib = (
        min(inputs.attn_layer_weight_bytes, inputs.linear_layer_weight_bytes) / MIB
    )
    for r in inputs.ranks:
        # The replicated non-layer weights are on every stage whatever the
        # cut does, so they belong in the "before a single layer lands"
        # figure; the role-scoped ones (embedding, lm_head) are not, because
        # this loop does not know which rank ends up holding them.
        overhead = (
            r.worst_transient_mib
            + r.fixed_overhead_mib
            + inputs.replicated_weight_bytes / MIB
        )
        usable = r.budget_mib - inputs.corridor_mib - overhead
        if usable < lightest_mib:
            reasons.append(
                f"rank {r.label}: budget {r.budget_mib:.0f} MiB minus corridor "
                f"{inputs.corridor_mib:.0f} MiB minus transient+overhead "
                f"{overhead:.0f} MiB leaves {usable:.0f} MiB, which "
                f"cannot hold even one layer ({lightest_mib:.0f} MiB)."
            )

    if not reasons:
        total_weight_mib = (
            pref_attn[n] * inputs.attn_layer_weight_bytes
            + (n - pref_attn[n]) * inputs.linear_layer_weight_bytes
            + inputs.embedding_weight_bytes
            + inputs.lm_head_weight_bytes
            + inputs.pp_size * inputs.replicated_weight_bytes
        ) / MIB
        total_kv_mib = (
            inputs.n_full_attention
            * inputs.kv_bytes_per_token_per_attn_layer
            * inputs.arena_tokens
        ) / MIB
        usable = (
            sum(
                r.budget_mib
                - inputs.corridor_mib
                - r.worst_transient_mib
                - r.fixed_overhead_mib
                for r in inputs.ranks
            )
            - (
                inputs.pp_size * inputs.replicated_weight_bytes
                + inputs.embedding_weight_bytes
                + inputs.lm_head_weight_bytes
            )
            / MIB
        )
        reasons.append(
            f"no contiguous {k}-stage cut of {n} layers fits: the model needs "
            f"{total_weight_mib:.0f} MiB of weights plus {total_kv_mib:.0f} MiB "
            f"of KV for a {inputs.arena_tokens}-token arena, against "
            f"{usable:.0f} MiB usable across {k} ranks after the "
            f"{inputs.corridor_mib:.0f} MiB corridor and per-rank transients. "
            f"Lower --max-total-tokens, raise the per-rank budget, or reduce "
            f"the optimization depth."
        )
        if hybrid and inputs.n_full_attention < k:
            reasons.append(
                f"the checkpoint has only {inputs.n_full_attention} "
                f"full-attention layers for {k} stages, so at least one stage "
                f"would hold an empty KV pool."
            )

    return PPCutSolution(
        counts=(),
        bounds=(),
        attention_counts=(),
        stages=(),
        makespan_seconds=math.inf,
        bottleneck_stage=-1,
        min_headroom_mib=-math.inf,
        feasible=False,
        refusals=tuple(reasons),
        candidates_considered=considered,
    )


# ---------------------------------------------------------------------------
# Validation of a hand-supplied cut
# ---------------------------------------------------------------------------


def validate_pp_cut(
    counts: Sequence[int],
    inputs: PPCutInputs,
    *,
    require_attention_per_stage: bool = True,
) -> Tuple[PPCutSolution, Tuple[str, ...]]:
    """Price an EXPLICIT per-stage layer split and report every violation.

    This is what turns ``--pp-layer-ratio`` from a planner bypass into a
    planner-checked override: the flag still wins, but it is priced against
    the same measured model the solver uses, and an infeasible list is
    refused loudly at parse time instead of becoming a runtime OOM or a
    corridor breach twenty minutes into a soak.

    Returns the priced solution and the violation list. An empty violation
    tuple means the split is admissible; the solution is still returned when
    violations exist so the caller can print the numbers alongside them.
    """
    n = inputs.n_layers
    k = inputs.pp_size
    violations: List[str] = []

    if len(counts) != k:
        raise ValueError(
            f"validate_pp_cut: got {len(counts)} stage counts for {k} ranks."
        )
    if any(int(c) < 1 for c in counts):
        raise ValueError(
            f"validate_pp_cut: every stage needs at least one layer, got "
            f"{list(counts)}."
        )
    if sum(int(c) for c in counts) != n:
        raise ValueError(
            f"validate_pp_cut: the split {list(counts)} sums to "
            f"{sum(int(c) for c in counts)}, but the model has {n} layers."
        )

    pref_attn = _prefix_attention(inputs.layer_families)
    hybrid = 0 < inputs.n_full_attention < n

    stages: List[StageCost] = []
    start = 0
    for i, c in enumerate(counts):
        cost = _price_stage(inputs, i, start, start + int(c), pref_attn)
        stages.append(cost)
        start += int(c)

    for cost in stages:
        if not cost.feasible:
            violations.append(
                f"stage {cost.start_layer}-{cost.end_layer} on rank "
                f"{cost.rank}: needs {cost.resident_mib:.0f} MiB "
                f"({cost.weight_mib:.0f} layer weights + "
                f"{cost.nonlayer_weight_mib:.0f} non-layer weights + "
                f"{cost.state_mib:.0f} recurrent state + "
                f"{cost.kv_mib:.0f} KV + "
                f"{cost.transient_mib:.0f} transient"
                + (
                    f", worst load state {cost.transient_load_state!r}"
                    if cost.transient_load_state
                    else ""
                )
                + f") plus "
                f"{cost.seam_staging_mib:.0f} MiB of peak seam staging it "
                f"must still be able to reach, but only "
                f"{cost.budget_mib:.0f} MiB is usable after the "
                f"{inputs.corridor_mib:.0f} MiB corridor -- over by "
                f"{-cost.runnable_headroom_mib:.0f} MiB."
                + (
                    ""
                    if cost.headroom_mib < 0
                    else f" It FITS AT REST ({cost.headroom_mib:.0f} MiB spare) "
                    f"and still cannot run: the staging is transient and no "
                    f"at-rest measurement contains it."
                )
            )
        if require_attention_per_stage and hybrid and cost.n_attention == 0:
            violations.append(
                f"stage {cost.start_layer}-{cost.end_layer} on rank "
                f"{cost.rank} holds zero of the model's "
                f"{inputs.n_full_attention} full-attention layers -- its KV "
                f"pool would be empty. A hybrid model splits its KV after "
                f"full-attention layers, not after layers."
            )

    times = [s.total_seconds for s in stages]
    makespan = max(times)
    solution = PPCutSolution(
        counts=tuple(int(c) for c in counts),
        bounds=tuple(s.end_layer for s in stages),
        attention_counts=tuple(s.n_attention for s in stages),
        stages=tuple(stages),
        makespan_seconds=makespan,
        bottleneck_stage=times.index(makespan),
        min_headroom_mib=min(s.runnable_headroom_mib for s in stages),
        feasible=not violations,
        refusals=tuple(violations),
        candidates_considered=k,
    )
    return solution, tuple(violations)


# ---------------------------------------------------------------------------
# #602 term 2: solve the cut for the KV FLOOR
# ---------------------------------------------------------------------------
#
# Under a pipeline the KV token count is necessarily UNIFORM across stages: a
# request's tokens occupy KV on every stage, each in its own layers. So
# `model_runner_kv_cache_mixin.py` min-reduces the per-stage capacities into one
# world value, and every stage above that minimum strands its surplus. Measured
# on the 2026-08-16 boot: 78362 tokens stranded on PP0 and 55255 on PP2, about
# 1.4 GiB that no post holds and no rank can spend.
#
# THAT SURPLUS IS NOT RECLAIMABLE WHERE IT SITS. It can only be converted by
# moving LAYERS: give the roomy stage more of them and the binding stage's
# per-token cost falls, so the world minimum rises. Hence a second objective
# over the same contiguous search space.
#
# WHY NOT REUSE solve_pp_cut's SECOND PASS. That pass maximizes the tightest
# `runnable_headroom_mib`. Headroom is not capacity: each stage converts MiB
# into tokens at a rate set by its OWN attention-layer count, so the cut that
# leaves the most MiB on the tightest card is not the cut that lets the pipeline
# address the most tokens. The two objectives genuinely differ and this one is
# the fill-side question.


@dataclasses.dataclass(frozen=True)
class KvFloorSolution:
    """The cut that maximizes the world KV floor, with its justification."""

    counts: Tuple[int, ...]
    bounds: Tuple[int, ...]
    attention_counts: Tuple[int, ...]
    stages: Tuple[StageCost, ...]
    #: min_r(capacity_r) -- the token count the pipeline can actually address.
    floor_tokens: float
    #: Per-stage capacity, same order as ``stages``.
    stage_tokens: Tuple[float, ...]
    feasible: bool
    refusals: Tuple[str, ...] = ()
    candidates_considered: int = 0

    def as_layer_ratio(self) -> List[int]:
        return list(self.counts)

    def summary(self) -> str:
        if not self.feasible:
            return "KV-floor cut: INFEASIBLE\n  " + "\n  ".join(self.refusals)
        rows = "\n".join(
            f"  stage {i} {c.rank}: layers {c.n_layers} (attn {c.n_attention}), "
            f"capacity {t:.0f} tokens, runnable headroom "
            f"{c.runnable_headroom_mib:.0f} MiB"
            for i, (c, t) in enumerate(zip(self.stages, self.stage_tokens))
        )
        return (
            f"KV-floor cut {list(self.counts)}: world floor "
            f"{self.floor_tokens:.0f} tokens\n{rows}"
        )


def _bounds_from_counts(counts: Sequence[int]) -> List[int]:
    out, acc = [], 0
    for c in counts:
        acc += int(c)
        out.append(acc)
    return out


def stage_kv_capacity(inputs: PPCutInputs, cost: StageCost) -> Optional[float]:
    """Tokens this stage could hold, or ``None`` when it holds none.

    Inverts the pricing rather than re-deriving it: ``kv_mib`` was priced at
    ``inputs.arena_tokens``, so the per-token cost is exact and any future
    change to the arena model (the TP-share ``max()``, a new family) is picked
    up here for free instead of drifting in a second copy.

    ``runnable_headroom_mib + kv_mib`` is what the stage could spend on KV once
    the corridor, the transient, the fixed overhead and the seam staging are all
    funded -- the hard constraints stay hard and are simply not part of the
    spendable pot.

    ``None`` for a stage with no full-attention layer: its KV pool is empty, so
    its capacity is unbounded and it would win every maximin while contributing
    no KV at all. That is the same refusal ``solve_pp_cut`` already makes.
    """
    arena = float(inputs.arena_tokens)
    if arena <= 0.0 or cost.kv_mib <= 0.0:
        return None
    spendable = cost.runnable_headroom_mib + cost.kv_mib
    if spendable <= 0.0:
        return None
    return spendable * arena / cost.kv_mib


def stage_kv_capacities(
    counts: Sequence[int], inputs: PPCutInputs
) -> Tuple[Optional[float], ...]:
    """Per-stage capacity for an ARBITRARY cut -- the incumbent included.

    An optimiser that cannot score the cut it proposes to replace cannot be
    shown to beat it, so this is public and takes plain layer counts.
    """
    if len(counts) != inputs.pp_size:
        raise ValueError(
            f"stage_kv_capacities: {len(counts)} counts for "
            f"{inputs.pp_size} stages."
        )
    if sum(int(c) for c in counts) != inputs.n_layers:
        raise ValueError(
            f"stage_kv_capacities: counts {list(counts)} cover "
            f"{sum(int(c) for c in counts)} layers, model has "
            f"{inputs.n_layers}."
        )
    pref_attn = _prefix_attention(inputs.layer_families)
    bounds = _bounds_from_counts(counts)
    starts = [0] + bounds[:-1]
    out: List[Optional[float]] = []
    for i in range(inputs.pp_size):
        cost = _price_stage(inputs, i, starts[i], bounds[i], pref_attn)
        out.append(stage_kv_capacity(inputs, cost))
    return tuple(out)


def world_kv_floor(
    counts: Sequence[int],
    inputs: PPCutInputs,
    *,
    require_attention_per_stage: bool = True,
) -> Optional[float]:
    """``min_r(capacity_r)`` for a given cut, or ``None`` if inadmissible.

    ``None`` rather than 0.0 or -inf: a cut that cannot fund the corridor and
    the seam has no capacity to report, and returning a number would let a
    caller rank it against cuts that can.
    """
    pref_attn = _prefix_attention(inputs.layer_families)
    hybrid = 0 < inputs.n_full_attention < inputs.n_layers
    bounds = _bounds_from_counts(counts)
    if len(counts) != inputs.pp_size or bounds[-1] != inputs.n_layers:
        return None
    if any(int(c) < 1 for c in counts):
        return None
    starts = [0] + bounds[:-1]
    floor = math.inf
    for i in range(inputs.pp_size):
        cost = _price_stage(inputs, i, starts[i], bounds[i], pref_attn)
        if not cost.feasible:
            return None
        if require_attention_per_stage and hybrid and cost.n_attention == 0:
            return None
        cap = stage_kv_capacity(inputs, cost)
        if cap is None:
            return None
        floor = min(floor, cap)
    return None if floor is math.inf else floor


def solve_pp_cut_for_kv_floor(
    inputs: PPCutInputs,
    *,
    require_attention_per_stage: bool = True,
) -> KvFloorSolution:
    """Exactly maximize ``min_r(capacity_r)`` over contiguous cuts.

    One maximin dynamic program over stage boundaries, the same shape as
    ``solve_pp_cut``'s second pass and exactly solvable for the same reason:
    the objective is a single aggregate (a minimum) over per-stage values that
    depend only on ``(stage, start, end)``.

    Infeasible input does not fall back to an even split (the #202 lesson): it
    returns ``feasible=False`` with a named reason per rank.

    TWO MODELLING LIMITS, NAMED RATHER THAN GUESSED:

    * The NEXTN / draft head is not modelled anywhere in this module, so a
      deployment that places a draft head on one stage pays bytes this solve
      does not see. On the flip-target layout that placement is a real term.
    * ``seam_staging_mib`` is a per-rank SCALAR here, but the live seam reserve
      is ``fixed + per-token`` (measured 227 MiB + 2360.1 B/token on rank 0),
      and the per-token part scales with the very arena this function solves
      for. Feeding a scalar measured at one arena size makes this a solve at
      that operating point, not a fixed point. Solving the fixed point needs a
      seam model this module does not have; until then, re-measure the scalar
      at the arena the solve proposes and re-run.
    """
    n = inputs.n_layers
    k = inputs.pp_size
    pref_attn = _prefix_attention(inputs.layer_families)
    hybrid = 0 < inputs.n_full_attention < n

    cache: Dict[Tuple[int, int, int], StageCost] = {}

    def price(stage: int, start: int, end: int) -> StageCost:
        key = (stage, start, end)
        got = cache.get(key)
        if got is None:
            got = _price_stage(inputs, stage, start, end, pref_attn)
            cache[key] = got
        return got

    def value(stage: int, start: int, end: int) -> Optional[float]:
        cost = price(stage, start, end)
        if not cost.feasible:
            return None
        if require_attention_per_stage and hybrid and cost.n_attention == 0:
            return None
        return stage_kv_capacity(inputs, cost)

    considered = 0
    best: List[List[float]] = [[-math.inf] * (n + 1) for _ in range(k + 1)]
    choice: List[List[int]] = [[-1] * (n + 1) for _ in range(k + 1)]
    best[0][0] = math.inf
    for s in range(1, k + 1):
        hi = n - (k - s)
        for b in range(s, hi + 1):
            row = best[s - 1]
            acc, arg = -math.inf, -1
            for prev in range(s - 1, b):
                base = row[prev]
                if base == -math.inf:
                    continue
                considered += 1
                cand = value(s - 1, prev, b)
                if cand is None:
                    continue
                val = base if base < cand else cand
                # Deterministic tie-break: the earliest boundary wins, so the
                # same inputs always yield the same cut.
                if val > acc:
                    acc, arg = val, prev
            best[s][b] = acc
            choice[s][b] = arg

    if best[k][n] == -math.inf:
        fallback = _infeasible(inputs, pref_attn, hybrid, considered)
        return KvFloorSolution(
            counts=fallback.counts,
            bounds=fallback.bounds,
            attention_counts=fallback.attention_counts,
            stages=fallback.stages,
            floor_tokens=0.0,
            stage_tokens=(),
            feasible=False,
            refusals=fallback.refusals,
            candidates_considered=considered,
        )

    bounds_rev: List[int] = []
    b = n
    for s in range(k, 0, -1):
        bounds_rev.append(b)
        b = choice[s][b]
    bounds = tuple(reversed(bounds_rev))
    starts = (0,) + bounds[:-1]
    stages = tuple(price(i, starts[i], bounds[i]) for i in range(k))
    caps = tuple(float(stage_kv_capacity(inputs, c) or 0.0) for c in stages)
    return KvFloorSolution(
        counts=tuple(s.n_layers for s in stages),
        bounds=bounds,
        attention_counts=tuple(s.n_attention for s in stages),
        stages=stages,
        floor_tokens=min(caps),
        stage_tokens=caps,
        feasible=True,
        refusals=(),
        candidates_considered=considered,
    )
