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
import json
import math
import os
import re
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

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

    #: #602: what the NEXTN / draft runner actually costs this rank, NET of
    #: the measured inter-runner overlap credit. See
    #: :func:`draft_residency_from_flight`, which is where this should come
    #: from -- the raw ``weights_draft`` post is the WRONG number and charging
    #: it prices the running configuration as infeasible.
    #:
    #: ``None`` means NOT MEASURED, which is not the same as zero and is
    #: refused by ``PPCutInputs`` whenever a draft runner is declared present.
    #: Zero is reserved for "this deployment has no draft runner".
    draft_residency_mib: Optional[float] = None

    #: #1009(a): what the DRIVER says this stage's card holds in total, as the
    #: census recorded it. Supplied so the corridor reserve is charged ONCE.
    #:
    #: THE RESERVE WAS BOOKED TWICE. ``budget_mib`` is the operator's
    #: ``--rank-gpu-memory-mib`` cap when they set one, and that cap is
    #: ITSELF a reserve: on this rig it is 18800 MiB against a 20054.9 MiB
    #: card, so the operator has already held 1254.9 MiB back. The gate then
    #: subtracted ``corridor_mib`` from that cap as well, which targets
    #: 2278.9 MiB free on the card. The corridor law is a BAND -- 819-1229
    #: MiB NVML-free per card under load -- and 2278.9 is ABOVE it, which
    #: that law scores as a FAILED acceptance in its own right. So the second
    #: subtraction did not buy safety; it drove the card out of the band on
    #: the other side and refused cuts for a reserve nothing asked for.
    #:
    #: With this set, the budget becomes ``min(cap, card_total - corridor)``:
    #: never more than the operator allowed, and never so much that fewer
    #: than ``corridor_mib`` MiB remain free on the card. Both constraints
    #: are still enforced, in full -- what is dropped is only the double
    #: charge where they overlap.
    #:
    #: ``None`` means the caller could not say, and the gate then keeps the
    #: strictly more conservative ``cap - corridor``. The default is the safe
    #: direction on purpose.
    #:
    #: ONE RANK PER CARD is assumed here, which is this dataclass's stated
    #: model ("One pipeline stage's host card"). Were two stages ever to
    #: share a card, ``card_total - corridor`` would have to be split between
    #: them rather than offered to each, and this field would need the
    #: co-tenant count.
    card_total_mib: Optional[float] = None

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

    #: #1009a: the same term PER STAGE, where the census measured it per
    #: stage. Indexed by stage; empty means "use the scalar for every stage",
    #: which is the previous behaviour exactly.
    #:
    #: The scalar above is a cross-rank MEAN, and on the #855 gdncov census
    #: the per-rank values spread 317.4 / 255.9 / 366.9 MiB per linear layer.
    #: Applying the mean back to each rank mispriced the calibrated cut by
    #: +799.7 MiB on stage1 and -541 MiB on stage2 -- and an under-charge is
    #: the direction that OOMs. See
    #: ``pp_cut_calibration.CensusCalibration.state_per_linear_mib_by_rank``
    #: for the measurement and for the honest limit (exact on the calibrated
    #: cut, an assumption off it, separable only with a second cut family).
    state_bytes_per_linear_layer_by_stage: Tuple[float, ...] = ()

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

    #: Free VRAM that must remain on every card AT REST. The rig's standing
    #: corridor, and specifically the USER-FREEDOM reserve of the reserve
    #: semantics law: 1024 MiB per card that nothing resident may consume.
    corridor_mib: float = 1024.0

    #: #1009a, operator decision 2026-08-30: how far the MEASURED WORST
    #: TRANSIENT may dip into the corridor, in MiB of free VRAM.
    #:
    #: TWO LAWS, TWO FLOORS, AND THEY WERE BEING CONFLATED. ``corridor_mib``
    #: is the at-rest user-freedom reserve (reserve semantics: 1024 MiB per
    #: card, default). The CORRIDOR law is a different statement: it defines
    #: 819-1229 MiB free per card as the acceptance BAND UNDER LOAD. Charging
    #: the transient peak against the 1024 at-rest reserve applied the wrong
    #: law to the peak and refused boots the corridor law passes.
    #:
    #: MEASURED, and this is the case that forced the decision: on the #855
    #: gdncov boot stage 0 holds 909.7 MiB free at its measured worst
    #: transient (30464.8 MiB at rest + 714.0 MiB EXTEND draw against a
    #: 32088.5 MiB card). 909.7 is INSIDE the 819-1229 band -- the running
    #: boot passes acceptance -- while the gate refused it for being below
    #: 1024. The band is not softened here: 819 is its published lower edge.
    #:
    #: SO THE PREDICATE IS TWO-SIDED, not one budget. At-rest residency must
    #: still leave ``corridor_mib`` free, because that reserve is the user's
    #: and a peak is not permission to spend it permanently. The peak on top
    #: of it may dip to ``corridor_dip_floor_mib`` and no further. A dip that
    #: lands in [floor, corridor) is LOGGED with both numbers rather than
    #: passed silently -- it is admissible, and it is also the operator's
    #: signal that the card is running at the band's lower edge.
    corridor_dip_floor_mib: float = 819.0

    #: #602: does this deployment run a NEXTN / draft runner? Declaring it
    #: makes ``RankResources.draft_residency_mib`` MANDATORY on every rank.
    #:
    #: THE FLAG EXISTS SO ABSENCE CANNOT BE MISTAKEN FOR ZERO (#606). Both
    #: readings are legitimate -- a deployment may genuinely have no draft
    #: runner -- and they are numerically identical, so only the caller can
    #: separate them. Leaving the term unset while a draft runner is in fact
    #: resident is what produced a solver that priced 18 GiB of weights at
    #: nothing; requiring the declaration turns that into a refusal at
    #: construction instead of a confident wrong answer at the end.
    draft_runner_present: bool = False

    def __post_init__(self) -> None:
        n = len(self.layer_families)
        if n == 0:
            raise ValueError("PPCutInputs: layer_families is empty.")
        if self.draft_runner_present:
            missing = [r.label for r in self.ranks if r.draft_residency_mib is None]
            if missing:
                raise ValueError(
                    "PPCutInputs: draft_runner_present=True but "
                    f"draft_residency_mib is not measured on rank(s) {missing}. "
                    "Supply it from draft_residency_from_flight() -- the NET "
                    "figure, weights_draft minus the inter-runner overlap "
                    "credit. Defaulting it to zero here would price a resident "
                    "draft runner at nothing, which is the calibration defect "
                    "this flag exists to prevent."
                )
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
    #: The AT-REST budget: what may be occupied with the user-freedom reserve
    #: (``PPCutInputs.corridor_mib``) still free on the card.
    budget_mib: float
    #: #1009a: the PEAK budget -- what may be occupied at the measured worst
    #: transient, with only the corridor band's lower edge
    #: (``PPCutInputs.corridor_dip_floor_mib``) still free. Always >=
    #: ``budget_mib``. Defaults to ``budget_mib`` so a caller that builds a
    #: StageCost by hand keeps the old, stricter single-budget behaviour.
    peak_budget_mib: float = 0.0
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
    #: #602: the cut-invariant residency that is ALWAYS resident -- CUDA
    #: context, NCCL, workspace, graph capture, boot tail. Split out of
    #: ``transient_mib``, which used to carry their SUM.
    #:
    #: WHY THE SPLIT EXISTS. The two are charged by different questions. This
    #: one is occupied at rest, so the live pool sizer charges it and any
    #: sizer-equivalent prediction must too. ``transient_mib`` is a PEAK the
    #: sizer never sees, because in this regime the worst load state is a SEAM
    #: and the pool is sized before any seam has run. Summed into one field,
    #: no consumer could charge one without the other, and the pool prediction
    #: came out 23 % under a measured boot.
    fixed_overhead_mib: float = 0.0
    #: #602: the draft runner's NET residency on this rank, its own post so a
    #: future mismatch names the term that moved rather than hiding inside
    #: ``transient_mib``. Cut-INVARIANT on this deployment: the draft model's
    #: placement follows its own vector, not ``pp_layer_ratio``.
    draft_mib: float = 0.0

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
            + self.fixed_overhead_mib
            + self.draft_mib
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
    def at_rest_mib(self) -> float:
        """Occupancy with no load on the stage: everything but the transient.

        The transient is a PEAK the stage reaches and gives back, so it is
        not part of what the stage holds at rest -- and the user-freedom
        reserve is a statement about rest.
        """
        return self.resident_mib - self.transient_mib

    @property
    def at_rest_headroom_mib(self) -> float:
        """Room left under the AT-REST budget once the stage is resident."""
        return self.budget_mib - self.at_rest_mib

    @property
    def effective_peak_budget_mib(self) -> float:
        """The peak budget, falling back to the at-rest one when unset."""
        return max(self.budget_mib, self.peak_budget_mib)

    @property
    def peak_headroom_mib(self) -> float:
        """Room left at the measured peak, against the dip floor.

        Charges the seam only where it exceeds the transient already funded
        -- see :attr:`runnable_headroom_mib` for why that subtraction is a
        max and not a sum.
        """
        return (
            self.effective_peak_budget_mib
            - self.resident_mib
            - max(0.0, self.seam_staging_mib - self.transient_mib)
        )

    @property
    def runnable_headroom_mib(self) -> float:
        """Headroom left once the peak transient is also funded.

        The quantity the verdict is actually about: a stage that fits at rest
        and cannot reach its seam is a stage that boots and then wedges.

        #1009(a): THE PEAK IS CHARGED ONCE, NOT TWICE. ``resident_mib``
        already contains ``transient_mib``, and on the wired path that is
        ``RankResources.worst_transient_mib`` -- the max over EVERY load state
        in the measured table. ``server_args._pp_cut_seam_staging`` builds
        ``seam_staging_mib`` by FILTERING THAT SAME TABLE to its
        ``SEAM_``-prefixed keys, so the seam states are a SUBSET of the states
        the transient already maxed over and subtracting the seam again books
        one measurement twice.

        Measured on the #855 gdncov census (census-855-v2, stage1): the table
        is ``{DECODE 1526.0, EXTEND 1390.1, SEAM_PP_TO_TP 1342.1,
        SEAM_TP_TO_PP 1526.0}``. ``worst_transient_mib`` returns 1526.0 and
        ``seam_staging_mib`` returns 1526.0 -- the SAME entry -- and the old
        expression charged 3052.0 MiB for a peak the instrument measured at
        1526.0. That is 1526 MiB of phantom demand on the binding rank.

        A rank occupies ONE load state at a time, so the honest bound on
        demand above at-rest residency is the MAX over states, never the sum
        of two of them. This charges the max: the seam only adds what it
        exceeds the already-charged transient by.

        NOT A RELAXATION OF THE SEAM LAW. When a caller supplies
        ``seam_staging_mib`` from an INDEPENDENT measurement that the
        transient table does not contain -- the hand-fed path the field was
        added for, where the census predates the seam feed -- that excess is
        still charged in full. What is removed is only the portion already
        funded once. The verdict can therefore never become more permissive
        than "fund the worst state this rank actually served", which is the
        predicate law 31 asks for.

        #1009a: TWO FLOORS, BOTH BINDING. The verdict is the tighter of two
        separate laws, not one budget doing double duty:

          at rest -- must leave ``corridor_mib`` (1024 MiB) free, the user's
                     reserve, which a transient peak is not permission to
                     spend permanently;
          at peak -- must leave ``corridor_dip_floor_mib`` (819 MiB) free,
                     the corridor band's published lower edge under load.

        Neither is softened; the change is only that the PEAK is now measured
        against the law that governs peaks. See
        ``PPCutInputs.corridor_dip_floor_mib`` for the measured case
        (909.7 MiB free at the worst transient -- inside the band, refused by
        the at-rest reserve).
        """
        return min(self.at_rest_headroom_mib, self.peak_headroom_mib)

    @property
    def corridor_dip_note(self) -> Optional[str]:
        """One line when this stage's peak dips into the corridor band.

        #1009a: a dip to between the band's lower edge and the at-rest
        reserve is ADMISSIBLE and is not passed silently -- it is the
        operator's signal that this card runs at the band's lower edge, and
        it names BOTH numbers so the reader never has to reconstruct which
        floor was applied. Returns None when the stage does not dip, or when
        no card total was supplied (no dip allowance was granted, so there is
        nothing to disclose).
        """
        if not self.peak_budget_mib or self.peak_budget_mib <= self.budget_mib:
            return None
        # Free VRAM at the peak, expressed against the at-rest floor. The
        # dip allowance is (peak_budget - budget) = corridor - dip_floor.
        overshoot = self.resident_mib - self.budget_mib
        if overshoot <= 0.0:
            return None
        allowance = self.peak_budget_mib - self.budget_mib
        return (
            f"{self.rank}: measured worst transient dips into the corridor "
            f"band -- peak occupancy is {overshoot:.1f} MiB above the at-rest "
            f"reserve, leaving {allowance - overshoot:.1f} MiB of the "
            f"{allowance:.0f} MiB dip allowance. Admissible: the peak is "
            f"governed by the corridor band's lower edge, at rest the full "
            f"reserve is still free ({self.at_rest_headroom_mib:.1f} MiB "
            f"spare). transient={self.transient_mib:.1f} MiB "
            f"(state {self.transient_load_state or 'unnamed'})."
        )

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
    # #1009a: per-stage where the census measured it per stage, otherwise the
    # cross-rank scalar. The mean destroyed a 43 % per-rank spread the
    # artifact already carried and under-charged one rank by 541 MiB.
    by_stage = inputs.state_bytes_per_linear_layer_by_stage
    state_bytes_per_linear = (
        by_stage[stage]
        if stage < len(by_stage)
        else inputs.state_bytes_per_linear_layer
    )
    state_mib = (n_linear * state_bytes_per_linear) / MIB

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
        # #602: NOT summed any more. The peak and the at-rest overhead are
        # charged by different questions; see StageCost.fixed_overhead_mib.
        transient_mib=rank.worst_transient_mib,
        fixed_overhead_mib=rank.fixed_overhead_mib,
        transient_load_state=rank.governing_load_state,
        seam_staging_mib=rank.seam_staging_mib,
        draft_mib=float(rank.draft_residency_mib or 0.0),
        # The corridor is subtracted here, once, so every downstream
        # comparison is against usable bytes.
        #
        # #1009(a): ONCE means once. Both constraints bind -- the stage may
        # not exceed the operator's cap, and the card may not be left with
        # less than the corridor free -- so the budget is the tighter of the
        # two, not the cap minus the corridor. See
        # RankResources.card_total_mib for the measured case this fixes: a
        # cap already 1254.9 MiB below the card total had the corridor
        # subtracted a second time, targeting 2278.9 MiB free against a
        # 819-1229 MiB band. Without a card total the old, strictly more
        # conservative form is kept.
        budget_mib=(
            rank.budget_mib - inputs.corridor_mib
            if rank.card_total_mib is None
            else min(
                rank.budget_mib,
                float(rank.card_total_mib) - inputs.corridor_mib,
            )
        ),
        # #1009a: the peak budget uses the corridor band's lower edge instead
        # of the at-rest reserve. Same two constraints as above -- never over
        # the operator's cap, never below the floor of free VRAM -- only the
        # floor differs, because a peak is governed by the corridor law and
        # rest is governed by the reserve law.
        # Without a card total the gate cannot say where the card's free
        # VRAM actually sits, so the dip allowance is NOT granted: the peak
        # budget collapses onto the at-rest budget and the old, stricter
        # single-budget behaviour is kept. The permissive direction is never
        # the default for an unmeasured term.
        peak_budget_mib=(
            rank.budget_mib - inputs.corridor_mib
            if rank.card_total_mib is None
            else min(
                rank.budget_mib,
                float(rank.card_total_mib) - inputs.corridor_dip_floor_mib,
            )
        ),
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
            f"stage_kv_capacities: {len(counts)} counts for {inputs.pp_size} stages."
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


# ---------------------------------------------------------------------------
# #602: the draft runner's residency, measured from the flight recorder
# ---------------------------------------------------------------------------


class DraftResidencyUnavailable(RuntimeError):
    """The recorder holds no draft posts for this boot.

    Raised instead of returning zero. Zero is a REAL value in this model --
    it means the deployment has no draft runner -- so substituting it for "not
    measured" would let an uncalibrated solve present itself as calibrated.
    That is the #606 getattr lesson applied to a number instead of an
    attribute: a defensive default that hides its own trigger.
    """


@dataclasses.dataclass(frozen=True)
class DraftResidency:
    """What the NEXTN / draft runner costs one rank, and where that was read.

    ``weights_draft_mib`` is what the draft runner ALLOCATED.
    ``overlap_credit_mib`` is what the process RELEASED between the two
    runners (the recorder's ``inter_runner_gap``, sign-flipped to a credit).

    THE NET IS THE ONLY FIGURE THE SOLVER MAY USE. Charging the gross weights
    prices the reference rig's RUNNING configuration as infeasible -- 10796
    MiB of draft weights against an 18800 MiB budget that also carries the
    target model -- because the two runners do not both occupy the card at
    once. The gross number is kept beside the net one so a reader can see the
    size of the correction rather than being handed a small number to trust.
    """

    pid: int
    card_uuid: str
    boot_id: str
    source: str
    weights_draft_mib: float
    overlap_credit_mib: float

    @property
    def net_mib(self) -> float:
        return self.weights_draft_mib - self.overlap_credit_mib


#: Recorder transitions this term is differenced from, keyed as
#: ``(closing phase, draft_worker)`` exactly as the fill-side report names
#: posts. Kept next to the reader so the two cannot drift into naming the same
#: bytes differently.
_DRAFT_WEIGHT_KEY = ("weights_loaded", True)
_INTER_RUNNER_GAP_KEY = ("pre_weight_load", True)


def draft_residency_from_flight(
    directory: str, *, boot: Optional[str] = None
) -> Dict[int, DraftResidency]:
    """Measure each process's draft-runner residency from the flight recorder.

    Returns ``{pid: DraftResidency}`` for ONE boot (the latest by default),
    grouped by pid rather than by rank for the reason
    ``flight_recorder.read_marks`` documents: under ``--tp-size 1 --pp-size 3``
    all three processes file their marks under TP rank 0, so keying on the rank
    field merges three cards into one timeline.

    Raises :class:`DraftResidencyUnavailable` when the directory holds no marks
    or the boot has no draft posts at all -- never a zero-filled result.
    """
    try:
        from sglang.srt.mem_ledger import flight_recorder
    except ImportError as exc:  # pragma: no cover - packaging accident
        raise DraftResidencyUnavailable(
            f"flight recorder module is not importable: {exc}"
        ) from exc

    try:
        by_pid = flight_recorder.read_marks(directory, boot=boot)
    except Exception as exc:
        raise DraftResidencyUnavailable(
            f"could not read flight marks from {directory!r}: {exc}"
        ) from exc

    if not by_pid:
        raise DraftResidencyUnavailable(
            f"no flight marks under {directory!r}"
            + (f" for boot {boot!r}" if boot else "")
            + ". The draft residency cannot be defaulted; arm the recorder on "
            "a boot of this configuration and re-read."
        )

    out: Dict[int, DraftResidency] = {}
    for pid, marks in by_pid.items():
        ordered = sorted(marks, key=lambda m: m.get("monotonic") or 0.0)
        weights = 0.0
        gap = 0.0
        seen_draft = False
        for prev, cur in zip(ordered, ordered[1:]):
            flag = (cur.get("extra") or {}).get("draft_worker")
            flag = None if flag is None else bool(flag)
            key = (str(cur.get("phase")), flag)
            delta = (
                int(cur.get("nvml_self_bytes") or 0)
                - int(prev.get("nvml_self_bytes") or 0)
            ) / MIB
            if key == _DRAFT_WEIGHT_KEY:
                weights += delta
                seen_draft = True
            elif key == _INTER_RUNNER_GAP_KEY:
                gap += delta
                seen_draft = True
        if not seen_draft:
            continue
        uuids = [m.get("card_uuid") for m in ordered if m.get("card_uuid")]
        boots = [m.get("boot_id") for m in ordered if m.get("boot_id")]
        out[int(pid)] = DraftResidency(
            pid=int(pid),
            card_uuid=str(sorted(uuids)[0]) if uuids else "?",
            boot_id=str(boots[0]) if boots else "?",
            source=str(directory),
            weights_draft_mib=weights,
            # Sign-flipped: the recorder writes the release as negative, and a
            # credit that reads positive is the one a caller can subtract
            # without having to remember which way the sign ran.
            overlap_credit_mib=-gap,
        )

    if not out:
        raise DraftResidencyUnavailable(
            f"the boot under {directory!r} has no draft-runner posts. If this "
            "deployment genuinely runs no draft runner, say so with "
            "draft_runner_present=False rather than reading a zero out of a "
            "recorder that never saw one."
        )
    return out


# ---------------------------------------------------------------------------
# #602: the seam is a FIXED POINT, because it flips the cut
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class KvFloorFixedPoint:
    """A KV-floor solve whose seam demand is consistent with its own arena."""

    solution: KvFloorSolution
    seam_staging_mib: Tuple[float, ...]
    iterations: int
    converged: bool


def solve_pp_cut_for_kv_floor_at_seam_fixed_point(
    inputs: PPCutInputs,
    *,
    seam_fixed_mib: Sequence[float],
    seam_slope_bytes_per_token: Sequence[float],
    tolerance_tokens: float = 50.0,
    max_iterations: int = 40,
) -> KvFloorFixedPoint:
    """Solve the cut with the seam priced at the arena the solve itself implies.

    WHY THIS IS NOT OPTIONAL, MEASURED RATHER THAN ASSUMED. The seam reserve is
    ``fixed + slope * tokens`` (rank 0: 227 MiB + 2360.1 B/token), so pricing it
    at the CURRENT arena while solving for a LARGER one understates it on
    exactly the stage the solve wants to load. On the reference rig that is not
    a rounding difference, it changes the ANSWER:

        seam priced at the live arena (471638)   ->  cut [31, 16, 17]
        seam priced at its own fixed point       ->  cut [30, 17, 17]

    A one-shot solve therefore returns a cut that is optimal for a seam demand
    the cut itself invalidates. The iteration below re-prices the seam at each
    candidate arena until the arena stops moving, which is the only operating
    point at which the answer is self-consistent.

    NON-CONVERGENCE IS REPORTED, NOT SMOOTHED. ``converged=False`` comes back
    with the last iterate rather than an exception so a caller can see what it
    was oscillating between, but the flag must be checked: an unconverged fixed
    point is a cut whose seam demand does not match its own arena, which is the
    defect this function exists to remove.
    """
    if (
        len(seam_fixed_mib) != inputs.pp_size
        or len(seam_slope_bytes_per_token) != inputs.pp_size
    ):
        raise ValueError(
            "solve_pp_cut_for_kv_floor_at_seam_fixed_point: seam terms must "
            f"cover all {inputs.pp_size} stages."
        )

    def _with_seam(tokens: float) -> PPCutInputs:
        seam = tuple(
            float(f) + float(s) * float(tokens) / MIB
            for f, s in zip(seam_fixed_mib, seam_slope_bytes_per_token)
        )
        ranks = tuple(
            dataclasses.replace(r, seam_staging_mib=seam[i])
            for i, r in enumerate(inputs.ranks)
        )
        return dataclasses.replace(inputs, ranks=ranks)

    tokens = float(inputs.arena_tokens)
    solution = None
    seam_used: Tuple[float, ...] = ()
    for it in range(1, max_iterations + 1):
        scoped = _with_seam(tokens)
        seam_used = tuple(r.seam_staging_mib for r in scoped.ranks)
        solution = solve_pp_cut_for_kv_floor(scoped)
        if not solution.feasible:
            return KvFloorFixedPoint(solution, seam_used, it, False)
        if abs(solution.floor_tokens - tokens) <= tolerance_tokens:
            return KvFloorFixedPoint(solution, seam_used, it, True)
        tokens = solution.floor_tokens
    return KvFloorFixedPoint(solution, seam_used, max_iterations, False)


def world_kv_floor_at_seam_fixed_point(
    counts: Sequence[int],
    inputs: PPCutInputs,
    *,
    seam_fixed_mib: Sequence[float],
    seam_slope_bytes_per_token: Sequence[float],
    tolerance_tokens: float = 50.0,
    max_iterations: int = 40,
) -> Optional[float]:
    """The same fixed point for a GIVEN cut, so the incumbent is scored on the
    same terms as the candidate. Comparing a fixed-point solve against a
    one-shot incumbent would credit the new cut with the correction."""
    tokens = float(inputs.arena_tokens)
    floor: Optional[float] = None
    for _ in range(max_iterations):
        seam = tuple(
            float(f) + float(s) * tokens / MIB
            for f, s in zip(seam_fixed_mib, seam_slope_bytes_per_token)
        )
        ranks = tuple(
            dataclasses.replace(r, seam_staging_mib=seam[i])
            for i, r in enumerate(inputs.ranks)
        )
        floor = world_kv_floor(counts, dataclasses.replace(inputs, ranks=ranks))
        if floor is None:
            return None
        if abs(floor - tokens) <= tolerance_tokens:
            return floor
        tokens = floor
    return floor


# ---------------------------------------------------------------------------
# #602: the remaining residency terms, measured from the flight recorder
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class MeasuredResidency:
    """One rank's cut-invariant overhead and its OBSERVED transient draw.

    ``fixed_overhead_mib`` is an at-rest quantity and the boot marks settle it
    exactly: what is resident at ``boot_complete`` that is not weights, not the
    KV pool and not the draft runner -- CUDA context, the NCCL init, the
    attention workspace, graph capture, the boot tail.

    ``observed_transient_mib`` is NOT settled by the recorder and the name says
    so. It is the peak draw over the serving window the recorder happened to
    cover, which on a short window is far gentler than the worst state the
    deployment will serve (law 31: 742 MiB observed here against 1989-3148 MiB
    on a 22-minute soak). ``covers_worst_load_state`` is therefore always False
    from this reader: only a soak-length window could justify True, and nothing
    in the marks proves the window was one. Charge
    ``max(observed, worst known)`` -- a gate that gets cheaper by looking at a
    shorter window is the C1 defect this field is shaped to prevent.
    """

    pid: int
    card_uuid: str
    boot_id: str
    source: str
    fixed_overhead_mib: float
    observed_transient_mib: float
    serving_samples: int
    serving_window_seconds: float
    covers_worst_load_state: bool = False


def residency_terms_from_flight(
    directory: str, *, boot: Optional[str] = None
) -> Dict[int, MeasuredResidency]:
    """Measure per-rank fixed overhead and observed transient, by pid.

    Raises :class:`DraftResidencyUnavailable` when the boot marks, the
    ``boot_complete`` mark, or the serving marks are missing. None of the three
    may be defaulted: zero overhead and zero transient are both meaningful
    values, so producing them from an absent measurement is the #606 defect.
    """
    try:
        from sglang.srt.mem_ledger import flight_recorder
    except ImportError as exc:  # pragma: no cover - packaging accident
        raise DraftResidencyUnavailable(
            f"flight recorder module is not importable: {exc}"
        ) from exc

    try:
        boot_by_pid = flight_recorder.read_marks(directory, boot=boot)
    except Exception as exc:
        raise DraftResidencyUnavailable(
            f"could not read flight marks from {directory!r}: {exc}"
        ) from exc
    if not boot_by_pid:
        raise DraftResidencyUnavailable(f"no flight marks under {directory!r}")

    try:
        serving_by_pid = flight_recorder.read_serving_marks(directory, boot=boot)
    except Exception:
        serving_by_pid = {}

    out: Dict[int, MeasuredResidency] = {}
    for pid, marks in boot_by_pid.items():
        ordered = sorted(marks, key=lambda m: m.get("monotonic") or 0.0)
        complete = [m for m in ordered if str(m.get("phase")) == "boot_complete"]
        if not complete:
            raise DraftResidencyUnavailable(
                f"pid {pid} under {directory!r} has no boot_complete mark, so "
                "its at-rest residency is not established. The overhead cannot "
                "be differenced from an unfinished boot."
            )
        at_rest = int(complete[-1].get("nvml_self_bytes") or 0) / MIB

        sums: Dict[Tuple[str, Optional[bool]], float] = {}
        for prev, cur in zip(ordered, ordered[1:]):
            flag = (cur.get("extra") or {}).get("draft_worker")
            flag = None if flag is None else bool(flag)
            delta = (
                int(cur.get("nvml_self_bytes") or 0)
                - int(prev.get("nvml_self_bytes") or 0)
            ) / MIB
            key = (str(cur.get("phase")), flag)
            sums[key] = sums.get(key, 0.0) + delta

        weights_target = sums.get(("weights_loaded", False), 0.0)
        kv_target = sums.get(("kv_pool_sized", False), 0.0)
        weights_draft = sums.get(_DRAFT_WEIGHT_KEY, 0.0)
        gap = sums.get(_INTER_RUNNER_GAP_KEY, 0.0)
        net_draft = weights_draft + gap
        # The draft runner's own KV motion (``kv_pool_sized`` with draft=True,
        # -740 MiB on rank 1 of the reference boot) is deliberately NOT
        # subtracted: the cost model prices only the TARGET KV arena, so a term
        # removed here would be represented nowhere and the reconstruction
        # would sit 740 MiB high on that rank. Letting the overhead absorb it
        # keeps the identity
        #     at_rest = weights + kv + net_draft + overhead
        # exact against what the model can actually price.
        overhead = at_rest - weights_target - kv_target - net_draft

        serving = serving_by_pid.get(pid) or []
        if not serving:
            raise DraftResidencyUnavailable(
                f"pid {pid} under {directory!r} has no serving marks, so no "
                "transient draw was observed at all. Zero would be a claim "
                "that nothing is drawn; arm the serving recorder and re-read."
            )
        s_ordered = sorted(serving, key=lambda m: m.get("monotonic") or 0.0)
        peak = max(int(m.get("nvml_self_bytes") or 0) for m in s_ordered) / MIB
        span = float(s_ordered[-1].get("monotonic") or 0.0) - float(
            s_ordered[0].get("monotonic") or 0.0
        )

        uuids = [m.get("card_uuid") for m in ordered if m.get("card_uuid")]
        boots = [m.get("boot_id") for m in ordered if m.get("boot_id")]
        out[int(pid)] = MeasuredResidency(
            pid=int(pid),
            card_uuid=str(sorted(uuids)[0]) if uuids else "?",
            boot_id=str(boots[0]) if boots else "?",
            source=str(directory),
            fixed_overhead_mib=overhead,
            observed_transient_mib=max(0.0, peak - at_rest),
            serving_samples=len(s_ordered),
            serving_window_seconds=span,
            # Never True from a reader: nothing in the marks proves the window
            # covered the worst state the deployment will serve.
            covers_worst_load_state=False,
        )
    return out


# ---------------------------------------------------------------------------
# #602: the weight terms, measured from the checkpoint
# ---------------------------------------------------------------------------

#: Bytes per element, by safetensors dtype tag.
_SAFETENSORS_DTYPE_BYTES: Dict[str, int] = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


@dataclasses.dataclass(frozen=True)
class CheckpointWeightTerms:
    """Per-family layer weights and the non-layer payloads, from the headers.

    Read from the safetensors HEADERS only -- no payload is touched, so this is
    instant and needs no accelerator. `PPCutInputs` already instructs callers to
    measure these rather than derive them from the config's parameter formulas,
    because the formula-derived attention layer was 30 MiB per layer wrong on
    the reference checkpoint and the error is CUT-SHAPED.

    ``replicated_weight_bytes`` is the payload every stage carries: on the
    reference checkpoint a vision tower and an MTP head. Adding those two is
    what closes the per-stage weight identity against the recorder to 0.1 %;
    modelling them as first-stage-only leaves stages 1 and 2 short by exactly
    their size, which is how they were found.
    """

    source: str
    n_layers: int
    attention_layer_indices: Tuple[int, ...]
    attn_layer_weight_bytes: float
    linear_layer_weight_bytes: float
    embedding_weight_bytes: float
    lm_head_weight_bytes: float
    replicated_weight_bytes: float
    replicated_breakdown: Dict[str, float]


def checkpoint_weight_terms(model_path: str) -> CheckpointWeightTerms:
    """Measure the weight terms `PPCutInputs` needs from a safetensors dir.

    Raises :class:`DraftResidencyUnavailable` when the directory holds no
    safetensors shards or no transformer layers -- never a zero-filled result,
    for the same reason the residency readers refuse: zero non-layer weight is
    a real value (tied embeddings, no vision tower) and must not stand in for
    "not measured".
    """
    import glob as _glob
    import struct as _struct

    shards = sorted(_glob.glob(os.path.join(model_path, "*.safetensors")))
    if not shards:
        raise DraftResidencyUnavailable(
            f"no *.safetensors under {model_path!r}; the weight terms cannot "
            "be defaulted."
        )

    sizes: Dict[str, float] = {}
    for shard in shards:
        try:
            with open(shard, "rb") as fh:
                (header_len,) = _struct.unpack("<Q", fh.read(8))
                header = json.loads(fh.read(header_len))
        except Exception as exc:
            raise DraftResidencyUnavailable(
                f"could not read the safetensors header of {shard!r}: {exc}"
            ) from exc
        for name, meta in header.items():
            if name == "__metadata__" or not isinstance(meta, dict):
                continue
            elements = 1
            for dim in meta.get("shape", []):
                elements *= int(dim)
            sizes[name] = float(elements) * _SAFETENSORS_DTYPE_BYTES.get(
                str(meta.get("dtype")), 2
            )

    layer_bytes: Dict[int, float] = {}
    layer_family: Dict[int, str] = {}
    embedding = lm_head = 0.0
    replicated: Dict[str, float] = {}
    layer_re = re.compile(r"layers\.(\d+)\.")
    for name, size in sizes.items():
        if name.startswith("mtp"):
            replicated["mtp"] = replicated.get("mtp", 0.0) + size
            continue
        if ".visual." in name or name.startswith("model.visual"):
            replicated["visual"] = replicated.get("visual", 0.0) + size
            continue
        if name == "lm_head.weight" or name.endswith(".lm_head.weight"):
            lm_head += size
            continue
        if "embed_tokens" in name:
            embedding += size
            continue
        found = layer_re.search(name)
        if found is None:
            continue
        index = int(found.group(1))
        layer_bytes[index] = layer_bytes.get(index, 0.0) + size
        if ".self_attn." in name:
            layer_family[index] = LAYER_FAMILY_ATTENTION
        elif index not in layer_family:
            layer_family[index] = LAYER_FAMILY_LINEAR

    if not layer_bytes:
        raise DraftResidencyUnavailable(
            f"{model_path!r} holds safetensors but no transformer layers; the "
            "per-family weights cannot be measured from it."
        )

    attn_idx = tuple(
        sorted(i for i in layer_bytes if layer_family.get(i) == LAYER_FAMILY_ATTENTION)
    )
    lin_idx = [i for i in layer_bytes if layer_family.get(i) != LAYER_FAMILY_ATTENTION]
    attn_mean = (
        sum(layer_bytes[i] for i in attn_idx) / len(attn_idx) if attn_idx else 0.0
    )
    lin_mean = sum(layer_bytes[i] for i in lin_idx) / len(lin_idx) if lin_idx else 0.0
    return CheckpointWeightTerms(
        source=str(model_path),
        n_layers=len(layer_bytes),
        attention_layer_indices=attn_idx,
        attn_layer_weight_bytes=attn_mean,
        linear_layer_weight_bytes=lin_mean,
        embedding_weight_bytes=embedding,
        lm_head_weight_bytes=lm_head,
        replicated_weight_bytes=sum(replicated.values()),
        replicated_breakdown=dict(replicated),
    )


# ---------------------------------------------------------------------------
# #685 desk half: where the rank-0 seam slope's 5.6x comes from
# ---------------------------------------------------------------------------


def seam_slope_bytes_per_token(
    flip_tp_vector: Sequence[float],
    attention_counts: Sequence[int],
    kv_bytes_per_token_per_attn_layer: float,
    n_attention_total: int,
    baseline_bytes_per_token: Sequence[float] = (),
) -> Tuple[float, ...]:
    """Desk-side alias of :func:`managers.seam_slope.derive_seam_slope_bytes_per_token`.

    #685. The derivation lives in ``srt/managers/seam_slope.py`` -- a
    dependency-free module the funding path can read without importing a
    solver -- and this is the name the planner already calls it by. Delegating
    rather than keeping a second copy is the point: two implementations of a
    formula that moves in whole-layer steps would drift, and the whole finding
    is that the vector is NOT a constant.

    See that module for the mechanism (the incoming leg of
    ``phase_flip_runtime._staging_bytes``), for why rank 0's 5.6x is one
    received layer rather than a pathology, and for why a frozen vector cannot
    be carried across cuts.
    """
    from sglang.srt.managers.seam_slope import derive_seam_slope_bytes_per_token

    return derive_seam_slope_bytes_per_token(
        flip_tp_vector,
        attention_counts,
        kv_bytes_per_token_per_attn_layer,
        n_attention_total,
        baseline_bytes_per_token,
    )


# ---------------------------------------------------------------------------
# #602: the SIZER-equivalent pool, kept apart from the corridor-safe floor
# ---------------------------------------------------------------------------
#
# TWO QUESTIONS, ~29 % APART ON THIS RIG, AND THEY WERE ONE NUMBER.
#
#   world_kv_floor        "the largest pool that stays corridor-safe while a
#                          cutover is in flight" -- funds the WORST measured
#                          load transient (law 31).
#   world_predicted_pool  "the pool the live sizer will actually produce" --
#                          funds no load transient at all, because the sizer
#                          runs BEFORE any seam has ever run and therefore
#                          never sees one.
#
# In F4-r4's census regime the worst load state is a SEAM on every rank
# (SEAM_TP_TO_PP 2168 MiB on rank 0; SEAM_PP_TO_TP 700 / 932 on ranks 1 / 2),
# two to three times the prefill-triggered scalars. Holding the corridor-safe
# floor to a measured pool therefore read -23.3 % and a metal arm was correctly
# refused on it. The defect was the comparison, not a constant.
#
# EVERYTHING ELSE STAYS CHARGED BY BOTH. The corridor, the weights, the draft
# residency, the cut-invariant overhead and the seam RESERVE (which is held at
# rest, so the sizer does charge it) bind both numbers. Only the load-transient
# PEAK separates them.


def stage_costs(counts: Sequence[int], inputs: PPCutInputs) -> Tuple[StageCost, ...]:
    """Priced stages for an arbitrary cut -- the incumbent included."""
    if len(counts) != inputs.pp_size:
        raise ValueError(
            f"stage_costs: {len(counts)} counts for {inputs.pp_size} stages."
        )
    if sum(int(c) for c in counts) != inputs.n_layers:
        raise ValueError(
            f"stage_costs: counts {list(counts)} cover "
            f"{sum(int(c) for c in counts)} layers, model has {inputs.n_layers}."
        )
    pref_attn = _prefix_attention(inputs.layer_families)
    bounds = _bounds_from_counts(counts)
    starts = [0] + bounds[:-1]
    return tuple(
        _price_stage(inputs, i, starts[i], bounds[i], pref_attn)
        for i in range(inputs.pp_size)
    )


def stage_pool_capacity(inputs: PPCutInputs, cost: StageCost) -> Optional[float]:
    """Tokens the SIZER would give this stage: the corridor-safe capacity with
    the worst-load transient added back.

    ``stage_kv_capacity`` subtracts that peak; the sizer never does. Adding it
    back here -- rather than re-deriving the whole spendable pot -- keeps the
    two functions provably one term apart, which is the property the tests
    pin.
    """
    safe = stage_kv_capacity(inputs, cost)
    if safe is None:
        return None
    arena = float(inputs.arena_tokens)
    if arena <= 0.0 or cost.kv_mib <= 0.0:
        return None
    return safe + float(cost.transient_mib) * arena / cost.kv_mib


def stage_pool_capacities(
    counts: Sequence[int], inputs: PPCutInputs
) -> Tuple[Optional[float], ...]:
    return tuple(
        stage_pool_capacity(inputs, cost) for cost in stage_costs(counts, inputs)
    )


def world_predicted_pool(
    counts: Sequence[int],
    inputs: PPCutInputs,
    *,
    require_attention_per_stage: bool = True,
) -> Optional[float]:
    """``min_r`` of the sizer-equivalent capacity -- the number a MEASURED pool
    may be compared against.

    This is the only output of this module that belongs in a gate against a
    boot's ``max_total_num_tokens``. ``world_kv_floor`` answers a corridor
    question and will read far below a measured pool wherever the worst load
    state is a seam; that is not an error in either number.
    """
    pref_attn = _prefix_attention(inputs.layer_families)
    hybrid = 0 < inputs.n_full_attention < inputs.n_layers
    bounds = _bounds_from_counts(counts)
    if len(counts) != inputs.pp_size or bounds[-1] != inputs.n_layers:
        return None
    if any(int(c) < 1 for c in counts):
        return None
    starts = [0] + bounds[:-1]
    best = math.inf
    for i in range(inputs.pp_size):
        cost = _price_stage(inputs, i, starts[i], bounds[i], pref_attn)
        if require_attention_per_stage and hybrid and cost.n_attention == 0:
            return None
        cap = stage_pool_capacity(inputs, cost)
        if cap is None:
            return None
        best = min(best, cap)
    return None if best is math.inf else best


def world_corridor_safe_floor(
    counts: Sequence[int],
    inputs: PPCutInputs,
    *,
    require_attention_per_stage: bool = True,
) -> Optional[float]:
    """``min_r`` of the corridor-safe capacity -- the SCORING counterpart of
    :func:`world_predicted_pool`, gated identically so the two can be compared.

    NOT the same function as :func:`world_kv_floor`, and the difference is the
    reason this exists. ``world_kv_floor`` additionally refuses a stage whose
    ``runnable_headroom_mib`` is negative AT THE INPUT ARENA, which is correct
    for the solver -- a cut that does not fit the arena being searched is not a
    candidate -- and wrong for scoring an incumbent: a MEASURED arena is
    already full by construction, so headroom sits at ~0 and any model error
    of either sign flips it to "infeasible" and the capacity, which is
    perfectly well defined and smaller, cannot be reported at all.

    So the solver keeps its gate and scoring gets a function without it. The
    only term separating this from ``world_predicted_pool`` is then the
    worst-load transient, which is exactly the property the tests pin.
    """
    pref_attn = _prefix_attention(inputs.layer_families)
    hybrid = 0 < inputs.n_full_attention < inputs.n_layers
    bounds = _bounds_from_counts(counts)
    if len(counts) != inputs.pp_size or bounds[-1] != inputs.n_layers:
        return None
    if any(int(c) < 1 for c in counts):
        return None
    starts = [0] + bounds[:-1]
    best = math.inf
    for i in range(inputs.pp_size):
        cost = _price_stage(inputs, i, starts[i], bounds[i], pref_attn)
        if require_attention_per_stage and hybrid and cost.n_attention == 0:
            return None
        cap = stage_kv_capacity(inputs, cost)
        if cap is None:
            return None
        best = min(best, cap)
    return None if best is math.inf else best


# ---------------------------------------------------------------------------
# #702 -- the PREFILL-SPEED objective.
#
# The #602 solve above answers "which cut maximises the KV pool". That is a
# capacity question, and on the censused regime it correctly withdrew the cut.
# The question the user actually asked -- "more prefill on the 5090" -- is a
# TIME question, and these two objectives do not agree. Both are exported so
# the trade is decided on numbers rather than on which solver was run.
#
# Two time objectives, also not equivalent to each other:
#   SERIAL     sum_r  stage_ms_r   -- today's non-pipelined prefill cost.
#   PIPELINED  max_r  stage_ms_r   -- once stages overlap, the slowest stage
#                                     is the throughput bound.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PrefillTiming:
    """Per-stage prefill cost model: ``fixed_ms[r] + ms_per_layer[r] * n_r``.

    CALIBRATION LIMIT, stated because it is load-bearing: a single measured cut
    gives ONE (layer-count, time) point per rank. That cannot separate the
    per-layer slope from a fixed per-stage cost -- both fit the point exactly.
    ``fixed_ms`` therefore defaults to all-zero, which is the OPTIMISTIC end of
    the family: it credits a reallocation with the full per-layer saving. A
    second measured cut per rank would pin the intercept; until one exists,
    every speedup produced here is an upper bound, not a prediction.
    """

    ms_per_layer: Tuple[float, ...]
    fixed_ms: Tuple[float, ...]

    def stage_ms(self, counts: Sequence[int]) -> Tuple[float, ...]:
        if len(counts) != len(self.ms_per_layer):
            raise ValueError(
                f"counts has {len(counts)} stages but the timing model has "
                f"{len(self.ms_per_layer)}."
            )
        return tuple(
            f + m * float(n)
            for f, m, n in zip(self.fixed_ms, self.ms_per_layer, counts)
        )


def prefill_timing_from_measurement(
    counts: Sequence[int],
    stage_ms: Sequence[float],
    fixed_fraction: float = 0.0,
) -> PrefillTiming:
    """Calibrate from one measured cut.

    ``fixed_fraction`` is the share of each measured stage time attributed to
    fixed per-stage cost rather than to layers. It is a SENSITIVITY DIAL, not a
    measurement: any value in [0, 1) reproduces the calibration point exactly
    (see ``PrefillTiming``), and larger values shrink the modelled benefit of
    moving layers between ranks.
    """
    if len(counts) != len(stage_ms):
        raise ValueError(
            f"counts ({len(counts)}) and stage_ms ({len(stage_ms)}) disagree "
            "on the number of stages."
        )
    if not 0.0 <= fixed_fraction < 1.0:
        raise ValueError(f"fixed_fraction must be in [0, 1), got {fixed_fraction}.")
    slopes: List[float] = []
    fixed: List[float] = []
    for r, (n, ms) in enumerate(zip(counts, stage_ms)):
        if int(n) <= 0:
            raise ValueError(
                f"stage {r} holds zero layers in the calibration cut, so its "
                "measured time cannot be turned into a per-layer cost "
                "(division by zero)."
            )
        f = fixed_fraction * float(ms)
        fixed.append(f)
        slopes.append((float(ms) - f) / float(n))
    return PrefillTiming(ms_per_layer=tuple(slopes), fixed_ms=tuple(fixed))


def _check_total(counts: Sequence[int], total_layers: Optional[int]) -> None:
    if total_layers is not None and int(sum(counts)) != int(total_layers):
        raise ValueError(
            f"stage counts {tuple(counts)} sum to {int(sum(counts))}, not the "
            f"declared total of {int(total_layers)} layers."
        )


def serial_prefill_ms(
    counts: Sequence[int],
    timing: PrefillTiming,
    total_layers: Optional[int] = None,
) -> float:
    """Non-pipelined prefill cost: the SUM over stages."""
    _check_total(counts, total_layers)
    return float(sum(timing.stage_ms(counts)))


def pipelined_prefill_ms(
    counts: Sequence[int],
    timing: PrefillTiming,
    total_layers: Optional[int] = None,
) -> float:
    """Pipelined prefill bound: the MAX over stages (the slowest stage)."""
    _check_total(counts, total_layers)
    return float(max(timing.stage_ms(counts)))


@dataclasses.dataclass(frozen=True)
class PrefillCutCandidate:
    counts: Tuple[int, ...]
    serial_ms: float
    pipelined_ms: float
    serial_speedup: float
    pipelined_speedup: float
    pool_tokens: Optional[float]


def _enumerate_cuts(
    total_layers: int, stages: int, min_per_stage: int
) -> List[Tuple[int, ...]]:
    out: List[Tuple[int, ...]] = []

    def rec(remaining: int, left: int, acc: Tuple[int, ...]) -> None:
        if left == 1:
            if remaining >= min_per_stage:
                out.append(acc + (remaining,))
            return
        # Leave at least min_per_stage for each of the remaining stages.
        hi = remaining - min_per_stage * (left - 1)
        for n in range(min_per_stage, hi + 1):
            rec(remaining - n, left - 1, acc + (n,))

    rec(int(total_layers), int(stages), ())
    return out


def solve_pp_cut_for_prefill_speed(
    total_layers: int,
    timing: PrefillTiming,
    incumbent: Sequence[int],
    max_rank0_layers: int,
    min_layers_per_stage: int = 1,
    pool_fn: Optional[Callable[[Sequence[int]], Optional[float]]] = None,
    top: Optional[int] = None,
) -> List[PrefillCutCandidate]:
    """Enumerate feasible cuts and price each on BOTH time objectives.

    ``max_rank0_layers`` is a hard VRAM constraint, not a preference: rank0's
    weights plus arena must fit its budget, and a cut above the cap is
    arithmetic that ignores it.

    ``pool_fn`` supplies the capacity cost of a cut, in tokens. It is injected
    rather than computed here so this solver stays free of ``PPCutInputs``; when
    it is absent, ``pool_tokens`` is ``None`` and NO pool number is invented.
    Every candidate carries both speedups and the pool cost so the trade is
    decided on numbers.
    """
    stages = len(incumbent)
    _check_total(incumbent, total_layers)
    base_serial = serial_prefill_ms(incumbent, timing)
    base_pipelined = pipelined_prefill_ms(incumbent, timing)
    cands: List[PrefillCutCandidate] = []
    for counts in _enumerate_cuts(total_layers, stages, min_layers_per_stage):
        if counts[0] > int(max_rank0_layers):
            continue
        s = serial_prefill_ms(counts, timing)
        p = pipelined_prefill_ms(counts, timing)
        cands.append(
            PrefillCutCandidate(
                counts=counts,
                serial_ms=s,
                pipelined_ms=p,
                serial_speedup=base_serial / s,
                pipelined_speedup=base_pipelined / p,
                pool_tokens=(None if pool_fn is None else pool_fn(counts)),
            )
        )
    cands.sort(key=lambda c: c.serial_ms)
    return cands if top is None else cands[: int(top)]


# ---------------------------------------------------------------------------
# #702 revision 4 -- the PP pool divides by ATTENTION layers.
#
# Rev 2 used the SUM of per-rank capacities (the TP rule); arm B was armed on it
# and OOM'd. Rev 3 corrected SUM -> MIN but kept dividing by the rank's TOTAL
# layer count. Rev 4 divides by its FULL-ATTENTION layer count, because that is
# where token-scaling KV actually lives:
#
#   HybridLinearKVPool -- "KV cache with separate pools for full and linear
#   attention layers" (mem_cache/memory_pool.py:3606); full_kv_pool is built
#   with layer_num=self.full_layer_nums (:3688), and
#   full_layer_nums = len(full_attention_layer_ids) (:3637).
#
# The GDN/linear layers hold no token-scaling KV at all. They hold per-SEQUENCE
# MambaPool slots, which is a RESIDENCY term (subtract from free) and not a
# divisor.
#
# The metal discriminator that forced rev 4: [28,20,16] solves 434,878 tokens
# live, [32,16,16] solves 416,796 (boot_armC2.log). Metal says the move LOSES
# 4.2 %; rev 3 predicted it WINS 9.2 %. Rev 3 had the sign backwards, which is
# fatal for a solver whose whole output is a ranking. Rev 4 reproduces both
# points, rank1 binding at the incumbent and rank0 at [32,16,16].
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PhasePoolModel:
    """Per-rank memory facts for the PP-phase pool.

    ``free_mib`` is free memory BEFORE layer weights, mamba states and the
    arming floor. ``kv_mib_per_token_per_attn_layer`` prices the FULL-ATTENTION
    layers only (fp8_e4m3 KV on this checkpoint: 2 x 4 kv-heads x 256 head_dim
    = 2048 B).

    ``arming_floor_mib`` is PER RANK and PER LAYOUT, and it has no default on
    purpose. It moves with the cut -- measured 2255/1728/2467 MiB on [32,16,16]
    against 1728/1825/2467 on [28,20,16] -- and treating it as a constant was
    responsible for ~84 % of the +9.6 % common-mode over-prediction that both
    competing KV-scaling rules shared. Callers must pass the value solved for
    THIS layout by the #676 machinery
    (``phase_flip_seam_reserve.arming_floor_target_bytes``), never a constant.

    KNOWN GAP: that solver derives the floor from a MEASURED seam draw, so a
    layout that has never booted has no solved floor. Predicting one needs a
    draw-versus-layout model that does not exist yet, and until it does every
    row for an unbooted cut carries the floor uncertainty of its proxy -- about
    +-500 MiB, which at 8 attention layers is +-32,000 tokens (~7 %).
    """

    free_mib: Tuple[float, ...]
    weight_mib_per_layer: float
    kv_mib_per_token_per_attn_layer: float
    arming_floor_mib: Tuple[float, ...]
    mamba_mib_per_linear_layer_per_slot: float = 0.0
    mamba_slots: int = 0


def stage_pp_capacities(
    counts: Sequence[int],
    attn_counts: Sequence[int],
    model: PhasePoolModel,
) -> Tuple[float, ...]:
    """Per-rank PP-phase token capacity.

    ``free_i / (attention_layers_i * kv_per_token_per_attn_layer)``, with the
    weights of ALL the rank's layers and its per-sequence mamba states removed
    from free first.
    """
    caps: List[float] = []
    for r, (n, a) in enumerate(zip(counts, attn_counts)):
        n, a = int(n), int(a)
        linear = n - a
        free = (
            float(model.free_mib[r])
            - float(model.weight_mib_per_layer) * n
            - float(model.mamba_mib_per_linear_layer_per_slot)
            * linear
            * int(model.mamba_slots)
            - float(model.arming_floor_mib[r])
        )
        if free < 0.0:
            raise ValueError(
                f"cut {tuple(counts)} is infeasible on rank{r}: {n} layers of "
                f"weights, its mamba states and a "
                f"{float(model.arming_floor_mib[r]):,.1f} MiB arming floor "
                f"exceed the {float(model.free_mib[r]):,.1f} MiB free by "
                f"{-free:,.1f} MiB."
            )
        if a <= 0:
            raise ValueError(
                f"rank{r} holds no full-attention layer, so it carries no "
                "token-scaling KV and its token capacity is unbounded. That is a "
                "modelling artifact, not a real configuration: refusing to price "
                f"cut {tuple(counts)} with attention counts {tuple(attn_counts)}."
            )
        caps.append(free / (a * float(model.kv_mib_per_token_per_attn_layer)))
    return tuple(caps)


def pp_phase_pool(
    counts: Sequence[int],
    attn_counts: Sequence[int],
    model: PhasePoolModel,
    **forbidden,
) -> float:
    """PP-phase pool bound: the MIN over ranks.

    Takes NO KV-vector argument. Under PP the pool is layer-sharded, so a rank's
    footprint is ``max_total_tokens * attention_layers_r`` and the token vector
    cannot relieve it -- proven on metal in #702, where cutting rank0's vector
    share 4.3x moved its memory by zero.
    """
    if forbidden:
        raise TypeError(
            "pp_phase_pool takes no KV-token-vector argument. Under PP prefill "
            "the pool is LAYER-sharded, so the vector does not enter; passing "
            "one is the TP-phase rule applied to the PP phase, which is the bug "
            f"this signature exists to prevent. Refused: {sorted(forbidden)}"
        )
    return min(stage_pp_capacities(counts, attn_counts, model))


def tp_phase_pool(total_attn_layers: int, n_ranks: int, model: PhasePoolModel) -> float:
    """TP-phase pool: the SUM over ranks, independent of any PP cut.

    Under TP every rank holds a width shard of every layer, so its per-token
    cost is ``total_attention_layers / n_ranks`` layers' worth. This column IS
    vector-relievable; the PP column is not.
    """
    per_token = (
        float(total_attn_layers)
        * float(model.kv_mib_per_token_per_attn_layer)
        / float(n_ranks)
    )
    return sum(float(f) / per_token for f in model.free_mib)


# ---------------------------------------------------------------------------
# #704 D1: the KV cell is CONSUMED from config, never fitted.
#
# The review gate's binding lesson: a constant calibrated against the incumbent
# silently absorbs exactly the layout-varying terms a new layout then exposes.
# The per-token KV cell is not a free parameter at all -- it is
# `2 (K and V) x num_key_value_heads x head_dim x dtype_width`, and the boot log
# confirms it byte-exactly: at 436,766 tokens the [28,20,16] boot logs K sizes
# 2.92 / 2.08 / 1.67 GB against attention counts 7 / 5 / 4, i.e. exactly
# `attn_i x 1024 B` per token for K, so 2048 B for K+V under fp8_e4m3.
#
# Fitting this cell against an observed pool is what produced the "0.83 of
# observed" fudge in an earlier revision of DESIGN_704, and with it a bf16
# reading of a checkpoint that ships fp8_e4m3.
#
# Converged from feat/704-prefill-ladder (Slot-3) into this module by
# Slot-2, verbatim, so the KV cell has ONE implementation rather than two
# that must agree.
# ---------------------------------------------------------------------------

_KV_DTYPE_WIDTH_BYTES = {
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "int8": 1,
    "bf16": 2,
    "bfloat16": 2,
    "fp16": 2,
    "float16": 2,
    "half": 2,
    "fp32": 4,
    "float32": 4,
    "auto": None,  # resolved from the model dtype by the caller, never guessed
}


def kv_dtype_width_bytes(kv_cache_dtype: str) -> int:
    """Byte width of one KV element. Unknown names are refused, not defaulted.

    A wrong default here is a silent 2x on every pool number, so there is no
    fallback: an unrecognised dtype raises.
    """
    key = str(kv_cache_dtype).strip().lower().removeprefix("torch.")
    width = _KV_DTYPE_WIDTH_BYTES.get(key)
    if width is None:
        if key == "auto":
            raise ValueError(
                "kv_cache_dtype='auto' does not name a width: resolve it to the "
                "model's own dtype before pricing a pool."
            )
        raise ValueError(
            f"unknown kv_cache_dtype {kv_cache_dtype!r}; refusing to guess a "
            f"width. Known: {sorted(k for k in _KV_DTYPE_WIDTH_BYTES if k != 'auto')}"
        )
    return int(width)


def kv_mib_per_token_per_attn_layer_from_config(
    config: Dict,
    kv_cache_dtype: str,
    num_hidden_layers: Optional[int] = None,
) -> float:
    """The per-token KV cell for ONE full-attention layer, from config alone.

    ``2 x num_key_value_heads x head_dim x dtype_width``. ``head_dim`` falls
    back to ``hidden_size / num_attention_heads`` only when the checkpoint does
    not state it, which is the same resolution order the model itself uses.
    """
    cfg = config.get("text_config") or config
    kv_heads = cfg.get("num_key_value_heads") or cfg.get("num_attention_heads")
    if not kv_heads:
        raise ValueError(
            "config states neither num_key_value_heads nor num_attention_heads; "
            "the KV cell cannot be derived and must not be fitted."
        )
    head_dim = cfg.get("head_dim")
    if not head_dim:
        hidden = cfg.get("hidden_size")
        heads = cfg.get("num_attention_heads")
        if not hidden or not heads:
            raise ValueError(
                "config states no head_dim and no hidden_size/num_attention_heads "
                "to derive one from; refusing to fit the KV cell."
            )
        head_dim = float(hidden) / float(heads)
    width = kv_dtype_width_bytes(kv_cache_dtype)
    return 2.0 * float(kv_heads) * float(head_dim) * float(width) / (1024.0 * 1024.0)


def decoupled_phase_pool(
    counts: Sequence[int],
    attn_counts: Sequence[int],
    total_attn_layers: int,
    model: PhasePoolModel,
) -> float:
    """#704 part B: the pool once attention KV is token-sharded across ranks.

    With the attention layers' KV distributed by free bytes instead of pinned
    to the layer owner, a rank's token-scaling footprint stops depending on
    which layers it holds: every rank carries a share of ALL attention layers.
    The bound is then the SUM of per-rank capacities priced over the world's
    attention layers.

    It is very nearly independent of the cut -- the weight total is constant
    however the layers are split -- but NOT exactly, because the per-layout
    arming floor and the mamba residency still follow the layers a rank holds.
    Those two terms are the reason this takes ``counts`` at all.
    """
    if int(total_attn_layers) <= 0:
        raise ValueError("no full-attention layers: there is no KV pool to shard.")
    n_ranks = len(model.free_mib)
    per_token = float(total_attn_layers) * float(model.kv_mib_per_token_per_attn_layer)
    total = 0.0
    for r, (n, a) in enumerate(zip(counts, attn_counts)):
        linear = int(n) - int(a)
        free = (
            float(model.free_mib[r])
            - float(model.weight_mib_per_layer) * int(n)
            - float(model.mamba_mib_per_linear_layer_per_slot)
            * linear
            * int(model.mamba_slots)
            - float(model.arming_floor_mib[r])
        )
        if free < 0.0:
            raise ValueError(
                f"cut {tuple(counts)} is infeasible on rank{r} before any KV is "
                f"placed: {-free:,.1f} MiB short on weights, mamba state and the "
                "arming floor."
            )
        total += free / per_token
    return total
