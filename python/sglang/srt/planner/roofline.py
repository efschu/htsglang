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
"""Roofline throughput ESTIMATE (design #145) — a rough, clearly-labelled
prefill + decode tok/s ballpark, WITHOUT MTP.

WHY THIS IS SEPARATE FROM EVERYTHING ELSE IN THE PLANNER
--------------------------------------------------------
The rest of the planner is capacity/feasibility only and REFUSES to emit an
invented throughput number (advantage.py §3.4: "emitting an invented perf
number is made structurally impossible"). That stance is correct: only the
runtime measures the real rate. This module does NOT overturn it — it is an
ADDITIVE, explicitly-separate estimate that:

  * carries ``provenance = "planner-estimate"`` (results_store.py rejects that
    provenance from the MEASURED store — an estimate can never be smuggled in
    as a measurement);
  * is surfaced under a big "rough ballpark, NOT measured" banner with the
    inputs shown so the user can sanity-check the derivation;
  * yields to a MEASURED entry whenever one exists (``measured_available`` ->
    the caller renders the roofline as secondary).

THE TWO ROOFLINES
-----------------
DECODE (bs=1, memory-bandwidth bound):

    decode_tok/s ~ eff_decode * interconnect_discount
                   / max_over_ranks( t_rank )
    t_rank = (resident_active_weight_bytes + kv_bytes_read) / peak_membw_rank
             + host_offloaded_active_expert_bytes / pcie_fetch_bw

  - "active bytes/token" reuses the fork's own sizing (uneven_perf): dense =
    all weights, MoE = shared expert + top-k routed experts only, hybrid =
    per-layer. MTP/draft and the vision tower are excluded (WITHOUT MTP;
    text decode does not stream the encoder).
  - KV bytes GROW with context (each new token attends to the whole prefix),
    so decode/token DROPS as the context fills — we report a range from the
    weight-bound ceiling (short context) to the value at a reference context.
  - MoE host-RAM expert offload (#77): the offloaded active experts are
    fetched over PCIe on demand, so their bytes move from the ~1.8 TB/s VRAM
    term to the ~single-digit-GB/s PCIe term — this is exactly why the
    122B-A10B measured at ~5-7 tok/s, orders of magnitude below its
    VRAM-resident roofline.

PREFILL (compute bound):

    prefill_tok/s ~ eff_prefill * min_over_ranks( peak_flops_rank
                                                  / (2 * active_params_rank) )

  gated by the slowest rank (uneven shards -> one rank finishes last). The
  O(n^2) attention term makes prefill/token drop at long context — noted as a
  caveat, not folded into the headline.

THE INTERCONNECT DISCOUNT (the honest core)
-------------------------------------------
A single-card roofline OVER-predicts multi-GPU TP because cross-rank
collectives add latency the roofline ignores. We apply a crude, explicitly
-labelled discount (mainly to decode) keyed on the interconnect:

  * solo (1 physical card)                 -> 1.00 (no cross-card collective)
  * NVLink                                 -> ~0.90 (mild)
  * PCIe with P2P                          -> ~0.70 (moderate)
  * PCIe host-staging, NO P2P (this rig)   -> STRONG, ~0.5/0.35/0.28 for
    2/3/4+ cross cards (calibrated crudely against our own measured
    weightless-lane / hetero-TP decode, which ran far below the single-card
    roofline; see MEMORY rig-interconnect-p2p: all pairs PHB, P2P=Not
    Supported, GPU0 on x4).

KV-DONOR / weightless-lane surcharge: if the plan concentrates the weights on
one card and uses the OTHER cards mainly as "KV donors" (they hold KV tokens
but ~no weights), every decode step must move attention/KV partials across the
interconnect for zero weight-locality benefit — an EXTRA ~0.7 on the
no-P2P/PCIe path, with a plain note. This is the "shift the KV split and pay
for it over PCIe" effect made visible.

Every factor above is a HEURISTIC. The numbers are order-of-magnitude, not
precise; the surfacing says so loudly.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Dict, List, Optional, Sequence

__all__ = [
    "RankRoofline",
    "RooflineEstimate",
    "estimate_roofline",
    "CardEnergy",
    "RooflineEnergyEstimate",
    "roofline_energy",
    "EFF_DECODE",
    "EFF_PREFILL",
    "IDLE_FRACTION_OF_TDP",
    "ROOFLINE_PROVENANCE",
]

#: This estimate's provenance tag. results_store.MEASURED_PROVENANCE does NOT
#: include it, so an ingest of a roofline number into the measured store is
#: rejected by construction (design §5B.3a).
ROOFLINE_PROVENANCE = "planner-estimate"

#: Efficiency factors (documented, tunable). A real kernel reaches only a
#: fraction of the nameplate roofline: decode is launch/overlap-limited even
#: when bandwidth-bound (~0.7-0.8); prefill loses more to attention, softmax,
#: norm, and imperfect GEMM tiling (~0.5-0.7). We take the middle of each band.
EFF_DECODE = 0.75
EFF_PREFILL = 0.60

#: Idle-power floor as a fraction of each card's TDP, for the ENERGY estimate
#: (roofline_energy). A GPU that is powered and holding a CUDA context never
#: drops to 0 W; under a live server process its floor sits well above the
#: desktop-idle few-watts. We model P_idle = IDLE_FRACTION_OF_TDP * TDP and let
#: the operating power ride between that floor and TDP with utilization
#: (P_op = P_idle + (TDP - P_idle) * util). This is a CRUDE, documented constant
#: — the single knob most worth calibrating against measured board power. See
#: the calibration note in roofline_energy(). Calibrated loosely against the
#: measured #146 board power: at 0.10 the PACE-SETTING (bottleneck) card lands
#: within ~10% and the rig-total watts within a few %; a fixed FRACTION of TDP
#: over-charges idle for a very-high-TDP card that a memory-bound decode never
#: loads (the RTX 5090's absolute watts come out ~30% high), which is the
#: dominant residual error and the reason this is flagged as a rough estimate.
IDLE_FRACTION_OF_TDP = 0.10

#: Mild extra decode discount for GGUF/marlin: weights are dequantized on the
#: fly, so decode is pushed part-way toward compute-bound (the pure-bandwidth
#: roofline slightly over-predicts). A documented, crude knock-down.
_GGUF_MARLIN_DECODE_DISCOUNT = 0.90

#: Families the roofline IGNORES: MTP/draft (WITHOUT MTP, by request) and the
#: vision tower (text decode/prefill does not stream the encoder per token).
_SKIP_FAMILIES = ("vision", "draft_attn", "draft_mlp", "draft_repl")

#: Only the lm_head half of the ``vocab`` family (embed + lm_head) is a
#: per-token matmul; the embedding table is a gather (not streamed / no MACs).
_VOCAB_STREAM_FRACTION = 0.5

#: Interconnect discount tiers (crude heuristics — shown to the user).
_NVLINK_DISCOUNT = 0.90
_PCIE_P2P_DISCOUNT = 0.70
_PCIE_NOP2P_BY_CROSS_CARDS = {2: 0.50, 3: 0.35}  # >=4 -> 0.28
_PCIE_NOP2P_MANY = 0.28
#: KV-donor / weightless-lane surcharge. When the plan places KV on DIFFERENT
#: cards than the weights (manual --rank-mlp-ratio / KV-token shift, up to a
#: pure "KV donor" card), every decode step moves attention/KV partials across
#: the interconnect for no weight-locality benefit. We grade it by the total
#: -variation distance between the per-card WEIGHT distribution and the per-card
#: KV distribution: 0 (aligned) -> no surcharge; 1 (fully disjoint) -> the max
#: knock-down below. Only applied on a cross-card, non-NVLink path.
_KV_DONOR_MAX_KNOCKDOWN = 0.40   # ×0.60 at full weight/KV divergence
#: Divergence above this is called out as an explicit "KV-donor layout".
_DONOR_DIVERGENCE_NOTE = 0.25

#: Effective PCIe host-staging efficiency vs the theoretical lane rate (real
#: H2D copies under no-P2P staging land well below theoretical).
_PCIE_STAGING_EFFICIENCY = 0.60

#: Effective expert-WAVE-PREFIX residency for MoE host-offload (#77). The
#: runtime's fixed-resident+scratch design (MEMORY moe-expert-offload) pins a
#: PREFIX of experts [0, R) on the GPU that is REUSED across every offload wave
#: and NEVER re-fetched — only the cold scratch region [R, R+C) spills over
#: PCIe, and even those cold misses are prefetched/double-buffered (#136a/#136b)
#: so they rarely stall the decode. So the fraction of per-token ACTIVE experts
#: that actually stalls on a host fetch is small: this knob is (1 - that
#: effective miss fraction). Because the resident prefix is chosen to be the HOT
#: experts, the effective hit-rate is far above the naive resident fraction.
#: Calibrated against our real-rig MEASURED point: 122B-A10B TP=3 on the live
#: 5090+2x3080 (no NVLink, GPU0 on a x4 slot) reached ~16.3 tok/s decode with
#: CUDA-graph + wave-prefix residency + H2D-prefetch; at this value the estimate
#: lands near it on the live probe. Hit-rate 0 (every active expert stalls on
#: PCIe, i.e. NO wave-prefix reuse) predicts ~4-5 tok/s — that matches only the
#: launch-overhead-bound eager path, not the end-state, so it would be too
#: harsh. A crude knob, documented as such.
_MOE_HOTSET_HIT_RATE = 0.85
#: GB/s per PCIe lane by generation (decimal GB, matching the GB/s convention).
_PCIE_GBS_PER_LANE = {3: 0.985, 4: 1.969, 5: 3.938}
_PCIE_DEFAULT_GEN, _PCIE_DEFAULT_WIDTH = 4, 8  # when the topology is unknown


# ---------------------------------------------------------------------------
# Result types.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RankRoofline:
    """Per-rank roofline inputs — the transparency block: what membw / FLOPS
    peak went in, whether it was measured or a nameplate spec, and the active
    bytes/params the formulas divided by."""

    rank: int
    gpu_index: Optional[int]
    gpu_name: str
    peak_membw_gbs: float
    membw_source: str  # "measured" | "nameplate-peak"
    peak_flops_tflops: float
    flops_source: str  # "measured" | "nameplate-peak(fp8)" | "nameplate-peak(fp16)"
    #: Active weight bytes streamed per decode token that stay resident in VRAM.
    resident_active_weight_gib: float
    #: Active MoE routed-expert bytes/token served from HOST RAM over PCIe
    #: (0.0 unless the plan is in the RAM-offload state).
    offloaded_active_weight_gib: float
    #: KV bytes read per decode token AT the reference context (grows with it).
    kv_bytes_per_token_gib: float
    #: Active params/token (for the prefill compute bound), in billions.
    active_gparams: float


@dataclasses.dataclass(frozen=True)
class RooflineEstimate:
    """A rough, explicitly-labelled prefill+decode tok/s ballpark (NOT
    measured). ``provenance == "planner-estimate"`` — never admissible into the
    measured results store."""

    provenance: str
    #: Decode tok/s as a range: high = weight-bound ceiling (short context),
    #: low = at the reference context (KV reads included). Both after all
    #: efficiency + interconnect discounts.
    decode_tok_s_high: float
    decode_tok_s_low: float
    #: Prefill tok/s (compute-bound, gated by the slowest rank).
    prefill_tok_s: float

    # -- transparency: the inputs used ----------------------------------
    reference_context_tokens: int
    eff_decode: float
    eff_prefill: float
    compute_dtype: str  # "fp8" | "fp16"
    gguf_marlin_dequant: bool
    interconnect: str  # solo | nvlink | pcie-p2p | pcie-host-staging
    interconnect_discount: float
    interconnect_note: str
    kv_donor_layout: bool
    offload_note: Optional[str]
    pcie_fetch_gbs: Optional[float]
    bottleneck_rank: int
    per_rank: List[RankRoofline]
    caveats: List[str]
    #: True only when a MEASURED end-to-end entry exists for this config (then
    #: the caller renders this estimate as SECONDARY). Default False on this rig
    #: (the measured store is empty) — the estimate then stands alone, labelled.
    measured_available: bool = False


# ---------------------------------------------------------------------------
# Energy estimate (design #148) — J/token total + per-card, from TDP + compute
# + membw + compute_share, reusing the throughput roofline for the wall time.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CardEnergy:
    """Per-PHYSICAL-CARD estimated energy for one phase. Ranks co-located on the
    same physical GPU are aggregated (their memory/compute activity adds on the
    one board that draws the power). ``compute_share`` is the uneven-DCP work
    weight (fraction of the model on this card) — the SAME weight the measured
    wpw column normalizes by, so measured and estimated read consistently."""

    gpu_index: int
    gpu_name: str
    #: Fraction of the model's work on this card (aggregated rank tp-ratio).
    compute_share: float
    #: Estimated average board power during the op (W), between idle and TDP.
    watts: float
    #: Estimated energy for one token on this card (J) = watts * shared t/token.
    j_per_token: float
    #: Utilization heuristic that set the power (busy fraction, in [0, 1]).
    util: float
    tdp_w: float
    idle_w: float
    #: Where the idle/active power ANCHORS came from. ``"estimate-tdp"`` (the
    #: default): idle = IDLE_FRACTION_OF_TDP x TDP and the dynamic ceiling is
    #: TDP — the crude heuristic, for a VIRTUAL / unmeasured card. ``"measured"``:
    #: the anchors are this exact physical card's NVML-measured board power
    #: (power_calibration), so the estimate is exact for the real rig and only
    #: VIRTUAL cards stay estimated (measured-overrides-estimate).
    power_source: str = "estimate-tdp"
    #: Upper power anchor the util rides toward (W): TDP for the heuristic, or
    #: the measured membw/gemm active draw for a measured card.
    active_anchor_w: Optional[float] = None


@dataclasses.dataclass(frozen=True)
class RooflineEnergyEstimate:
    """A rough, explicitly-labelled per-card + total J/token ENERGY estimate
    (design #148), the energy sibling of ``RooflineEstimate``. Same
    ``provenance == "planner-estimate"`` discipline — a render-time derivation,
    NEVER a measured entry, NEVER admissible into the measured results store.

    The physics (all order-of-magnitude, documented in ``roofline_energy``):
        J_per_token[card] = P_op[card] * t_per_token
      * ``t_per_token`` is the SHARED lockstep wall time = 1 / (rig roofline
        throughput of that phase, WITH the interconnect discount) — identical
        for every card (all ranks step together).
      * ``P_op[card]`` rides between an idle floor and TDP with a utilization
        heuristic (decode ~ membw utilization, prefill ~ FLOPS utilization);
        a fast card that finishes its shard early and waits on the slowest rank
        draws less -> its per-work energy (J / compute_share) is lower -> the
        heterogeneous efficiency gap is PREDICTED, not just measured."""

    provenance: str
    #: Totals (Σ over cards) — the rig J for one token of each phase.
    j_per_prefill_token_total: float
    j_per_decode_token_total: float
    #: Shared lockstep wall time per token (s) for each phase (1 / rig tok/s).
    t_prefill_token_s: float
    t_decode_token_s: float
    prefill: List[CardEnergy]
    decode: List[CardEnergy]
    idle_fraction_of_tdp: float
    caveats: List[str]
    #: Mirror of the throughput estimate: True only when a MEASURED energy entry
    #: exists for this config (then the caller renders this as SECONDARY).
    measured_available: bool = False


def _compute_share_by_card(inputs, per_rank) -> Dict[int, float]:
    """{gpu_index: fraction of the model's work on that card}, from the uneven
    -TP ratio aggregated per PHYSICAL card (co-located ranks add). Even/no-ratio
    -> each rank weighs 1/tp. This mirrors energy._compute_share_by_nvml (the
    measured wpw normalizer) so measured vs estimated efficiency read the same
    scale."""
    tp = inputs.tp_size
    ratio = inputs.rank_tp_ratio or [1] * tp
    total = float(sum(ratio)) or 1.0
    share: Dict[int, float] = {}
    for r in range(tp):
        gid = per_rank[r].gpu_index
        share[gid] = share.get(gid, 0.0) + ratio[r] / total
    return share


def _phase_cards(per_rank, share_by_card, tdp_by_card, activity, eff, t_wall,
                 idle_fraction, anchors_by_card=None) -> List[CardEnergy]:
    """Build the per-physical-card energy list for one phase.

    ``activity[r]`` is rank r's operation time at its OWN roofline (memory time
    for decode, compute time for prefill) — the *undiscounted* activity time.
    The busy fraction (utilization) is relative to the pace-setter:

        util[rank] = eff * activity[rank] / max_rank(activity)      (in [0, 1])

    so the SLOWEST (pace-setting) rank runs at ``eff`` (a real kernel reaches
    only a fraction of nameplate even flat-out) and a faster co-rank that
    finishes early and waits draws proportionally less. Deliberately does NOT
    fold the interconnect discount into utilization: during collective waits a
    card is blocked, near idle — that lost time lengthens ``t_wall`` (and hence
    the energy) but should not raise the busy-power. Co-located ranks share one
    board, so their utilizations add before the idle->active interpolation, then
    the board is clamped at its active ceiling.

        P_op[card] = P_idle[card] + (P_active[card] - P_idle[card]) * util_card
        J_per_token[card] = P_op[card] * t_wall           (shared lockstep time)

    ``anchors_by_card[gid] = (idle_w, active_w, source)`` supplies the two power
    ANCHORS per physical card. When a card was MEASURED (power_calibration),
    ``idle_w`` / ``active_w`` are its NVML-measured idle and membw/gemm active
    draw — so the real rig reads exact. When absent, the anchors fall back to the
    crude heuristic ``idle = idle_fraction x TDP`` and ``active = TDP`` — the
    estimate for a VIRTUAL / unmeasured card (measured-overrides-estimate).
    """
    anchors_by_card = anchors_by_card or {}
    max_act = max(activity) if activity else 0.0
    # Aggregate per physical card.
    util_by_card: Dict[int, float] = {}
    name_by_card: Dict[int, str] = {}
    for r, pr in enumerate(per_rank):
        gid = pr.gpu_index
        u = (eff * activity[r] / max_act) if max_act > 0 else 0.0
        util_by_card[gid] = util_by_card.get(gid, 0.0) + u
        name_by_card[gid] = pr.gpu_name

    out: List[CardEnergy] = []
    for gid in sorted(util_by_card):
        tdp = float(tdp_by_card[gid])
        anchor = anchors_by_card.get(gid)
        if anchor is not None:
            idle, active, source = anchor
        else:
            # Heuristic fallback for a virtual / unmeasured card.
            idle, active, source = idle_fraction * tdp, tdp, "estimate-tdp"
        util = min(util_by_card[gid], 1.0)  # bounded [idle, active]
        watts = idle + (active - idle) * util
        out.append(
            CardEnergy(
                gpu_index=gid,
                gpu_name=name_by_card[gid],
                compute_share=share_by_card.get(gid, 0.0),
                watts=watts,
                j_per_token=watts * t_wall,
                util=util,
                tdp_w=tdp,
                idle_w=idle,
                power_source=source,
                active_anchor_w=active,
            )
        )
    return out


#: Sentinel: roofline_energy auto-loads the MEASURED power table from the
#: default persisted path. Pass ``power_profile=None`` (or an explicit dict) to
#: override — tests pass ``{}`` to force the heuristic hermetically.
_AUTO_POWER = object()

#: Hardware sources that describe a LIVE, physically-present rig — only then is
#: an arch-level (not UUID-exact) power match allowed. A ``manual`` / ``json``
#: spec (virtual rig, unit tests) never picks up a measured entry by arch, so
#: the heuristic path stays hermetic regardless of what is cached on disk.
_LIVE_HW_SOURCES = ("pynvml", "nvml", "nvidia-smi")


def _measured_power_by_card(hardware, per_rank, library, power_profile, phase):
    """{gpu_index: (idle_w, active_w, "measured")} for cards that have a MEASURED
    power row, matched by NVML UUID (always) or, on a live rig, by sm-arch. The
    active anchor is the phase-relevant measured draw: membw (decode) / gemm
    (prefill). Cards with no measurement are simply absent -> the caller keeps
    the heuristic for them. Never raises."""
    if not power_profile:
        return {}
    try:
        from sglang.srt.planner.power_calibration import (
            _norm_uuid,
            power_profile_by_arch,
        )
    except Exception:
        return {}

    by_uuid = {_norm_uuid(u): m for u, m in power_profile.items()}
    use_arch = getattr(hardware, "source", None) in _LIVE_HW_SOURCES
    by_arch = power_profile_by_arch(power_profile) if use_arch else {}

    out: Dict[int, tuple] = {}
    for pr in per_rank:
        gid = pr.gpu_index
        if gid in out:
            continue
        try:
            card = hardware.gpu(gid)
        except Exception:
            continue
        m = None
        uu = getattr(card, "uuid", None)
        if uu:
            m = by_uuid.get(_norm_uuid(uu))
        if m is None and by_arch:
            prof = _profile_for(library, card.name)
            arch = getattr(prof, "sm_arch", None) if prof else None
            if arch:
                m = by_arch.get(arch)
        if m is None:
            continue
        active = m.p_gemm_w if phase == "prefill" else m.p_membw_w
        out[gid] = (float(m.p_idle_w), float(active), "measured")
    return out


def roofline_energy(
    estimate: Optional[RooflineEstimate],
    inputs,
    hardware,
    *,
    library=None,
    measured_available: bool = False,
    power_profile=_AUTO_POWER,
) -> Optional[RooflineEnergyEstimate]:
    """Estimated J/token (total + per physical card) for a planned config, the
    ENERGY sibling of ``estimate_roofline``. Returns None when it cannot be sized
    (no throughput estimate, or a card missing a TDP in the card library).
    NEVER raises into the plan flow — callers guard it.

    Reuses ``estimate`` (the throughput roofline) for BOTH the shared per-token
    wall time (1 / tok_s, WITH all discounts) and the per-rank activity terms
    (bytes streamed / FLOPs per token, and the peak membw / FLOPS that were
    used) — it does not re-derive throughput. TDP comes from the ``CardSpec``
    catalog (``tdp_w``); ``compute_share`` from the uneven-TP ratio."""
    from sglang.srt.planner.card_library import CardLibrary

    if estimate is None:
        return None
    library = library or CardLibrary()
    per_rank = estimate.per_rank
    tp = inputs.tp_size
    if not per_rank or len(per_rank) != tp:
        return None

    # MEASURED power table (power_calibration): auto-load from disk unless the
    # caller supplied one (tests pass {} to force the heuristic hermetically).
    # It OVERRIDES the TDP heuristic for the cards it recognises (by UUID/arch);
    # every other card keeps the estimate. Best-effort — never fatal.
    if power_profile is _AUTO_POWER:
        try:
            from sglang.srt.planner.power_calibration import load_power_profile

            power_profile = load_power_profile()
        except Exception:
            power_profile = {}

    # TDP per physical card (from the catalog). A card without a known TDP means
    # no energy estimate for this config (like an unknown peak drops throughput).
    tdp_by_card: Dict[int, float] = {}
    for pr in per_rank:
        if pr.gpu_index in tdp_by_card:
            continue
        prof = _profile_for(library, pr.gpu_name)
        if prof is None or not getattr(prof, "tdp_w", None):
            return None
        tdp_by_card[pr.gpu_index] = float(prof.tdp_w)

    share_by_card = _compute_share_by_card(inputs, per_rank)

    # -- per-rank UNDISCOUNTED activity time (s) at its own roofline ----------
    # decode: memory time = VRAM bytes streamed / peak membw (resident weights +
    #   KV at the reference context; the offloaded PCIe slice is not a membw
    #   term). prefill: compute time = per-token FLOPs / peak FLOPS.
    decode_act: List[float] = []
    prefill_act: List[float] = []
    for pr in per_rank:
        vram_bytes = (pr.resident_active_weight_gib + pr.kv_bytes_per_token_gib) * 2**30
        membw = pr.peak_membw_gbs * 1e9
        decode_act.append(vram_bytes / membw if membw > 0 else 0.0)
        per_tok_flops = 2.0 * pr.active_gparams * 1e9
        flops = pr.peak_flops_tflops * 1e12
        prefill_act.append(per_tok_flops / flops if flops > 0 else 0.0)

    # Shared lockstep wall time per token = 1 / rig throughput, WITH every
    # discount (interconnect, efficiency) already baked into the roofline tok/s.
    # Decode uses the at-reference-context (low) rate to match the KV-included
    # activity above. A zero rate (un-sizable) -> no energy estimate.
    if estimate.decode_tok_s_low <= 0 or estimate.prefill_tok_s <= 0:
        return None
    t_decode = 1.0 / estimate.decode_tok_s_low
    t_prefill = 1.0 / estimate.prefill_tok_s

    decode_anchors = _measured_power_by_card(
        hardware, per_rank, library, power_profile, "decode")
    prefill_anchors = _measured_power_by_card(
        hardware, per_rank, library, power_profile, "prefill")
    any_measured = bool(decode_anchors or prefill_anchors)

    decode_cards = _phase_cards(
        per_rank, share_by_card, tdp_by_card, decode_act,
        estimate.eff_decode, t_decode, IDLE_FRACTION_OF_TDP,
        anchors_by_card=decode_anchors,
    )
    prefill_cards = _phase_cards(
        per_rank, share_by_card, tdp_by_card, prefill_act,
        estimate.eff_prefill, t_prefill, IDLE_FRACTION_OF_TDP,
        anchors_by_card=prefill_anchors,
    )

    caveats = [
        "ROUGH ENERGY BALLPARK, NOT measured — board power is estimated from "
        "TDP and a utilization heuristic, not read from NVML. Treat J/token as "
        "order-of-magnitude; the runtime energy module measures the real value.",
        f"power model: P_op = idle + (TDP - idle) x util, idle = "
        f"{IDLE_FRACTION_OF_TDP:.0%} of TDP; the pace-setting (slowest) card "
        f"runs at util = eff (decode x{estimate.eff_decode}, prefill "
        f"x{estimate.eff_prefill}), a faster co-rank that waits draws less.",
        "decode util ~ membw utilization (memory-bound); prefill util ~ FLOPS "
        "utilization (compute-bound). Utilization is bounded to [idle, TDP].",
        "CALIBRATION vs measured #146: the bottleneck card lands within ~10% "
        "and rig-total watts within a few %, but a fixed idle-FRACTION of TDP "
        "over-charges idle on a very-high-TDP card a memory-bound decode never "
        "loads (RTX 5090 absolute watts ~30% high) — the dominant residual.",
    ]
    if any_measured:
        measured_gids = sorted(set(decode_anchors) | set(prefill_anchors))
        caveats.insert(0, (
            "MEASURED power anchors in use for card(s) "
            f"{measured_gids}: their idle / active watts are NVML board-power "
            "measurements (power_calibration), NOT the TDP heuristic — the "
            "power for these physical cards is exact (measured-overrides"
            "-estimate); any remaining card still uses the TDP estimate. The "
            "wall-time (tok/s) is still a roofline estimate, so J/token remains "
            "an estimate, but a far tighter one on the measured cards."
        ))

    return RooflineEnergyEstimate(
        provenance=ROOFLINE_PROVENANCE,
        j_per_prefill_token_total=sum(c.j_per_token for c in prefill_cards),
        j_per_decode_token_total=sum(c.j_per_token for c in decode_cards),
        t_prefill_token_s=t_prefill,
        t_decode_token_s=t_decode,
        prefill=prefill_cards,
        decode=decode_cards,
        idle_fraction_of_tdp=IDLE_FRACTION_OF_TDP,
        caveats=caveats,
        measured_available=measured_available,
    )


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _infer_compute(model_path: str, cfg: dict):
    """(compute_dtype, gguf_or_marlin) for the FLOPS-peak selection.

    fp8 checkpoint -> the fp8 tensor path (on cards that have fp8 cores);
    everything quantized-to-int (AWQ/GPTQ/GGUF/marlin) dequantizes to fp16 for
    the matmul -> the fp16 tensor path, flagged so decode takes the mild
    on-the-fly-dequant discount."""
    q = cfg.get("quantization_config") or {}
    method = str(q.get("quant_method", "")).lower()
    fmt = str(q.get("fmt", "")).lower()
    name = os.path.basename(str(model_path)).lower()
    is_gguf = str(model_path).lower().endswith(".gguf")
    if "fp8" in method or "fp8" in fmt or ("fp8" in name and not is_gguf):
        return "fp8", False
    marlinish = is_gguf or any(
        k in (method + " " + name)
        for k in ("awq", "gptq", "marlin", "gguf", "int4", "q4", "q3", "q2")
    )
    return "fp16", marlinish


def _profile_for(library, name):
    """Look up a CardSpec by card name, tolerating the vendor/marketing prefix
    that live NVML reports ("NVIDIA GeForce RTX 5090") but the catalog omits
    ("RTX 5090"). Returns None when the card is not in the library (then the
    roofline has no nameplate fallback for it — measured probe or nothing)."""
    if library.has(name):
        return library.get(name)
    alt = (
        str(name).replace("NVIDIA", "").replace("GeForce", "")
        .replace("geforce", "").strip()
    )
    if alt and library.has(alt):
        return library.get(alt)
    return None


def _measured_by_index(hardware, measured_scores):
    """{gpu_index: (membw_gbs, gemm_tflops)} of MEASURED micro-probe scores.

    Preference: match the live rig's cached probe by GPU **UUID** (the profile
    is keyed by UUID, and ``HardwareSpec`` carries the NVML UUID per card) — this
    sidesteps the torch-cuda-vs-NVML device-order trap that a bare index map
    would fall into. ``measured_scores`` (advantage.MeasuredCardScore) is folded
    in as a secondary source for indices the UUID match didn't cover."""
    out = {}
    for s in measured_scores or []:
        mb = getattr(s, "membw_gbs", None)
        gf = getattr(s, "gemm_tflops", None)
        if mb and gf:
            out[s.gpu_index] = (float(mb), float(gf))
    # UUID-matched live probe (authoritative when this IS the booted rig).
    if hardware.source in ("pynvml", "nvidia-smi", "nvml"):
        try:
            from sglang.srt.uneven_perf import get_cached_hardware_profile

            profile, _ = get_cached_hardware_profile()
            if profile is not None:
                by_uuid = profile.get("gpus", {})
                for g in hardware.gpus:
                    ent = by_uuid.get(getattr(g, "uuid", None) or "")
                    if ent and ent.get("membw_gbs") and ent.get("gemm_tflops"):
                        out[g.index] = (
                            float(ent["membw_gbs"]),
                            float(ent["gemm_tflops"]),
                        )
        except Exception:
            pass
    return out


def _rank_peaks(model, hardware, library, measured_scores, dtype):
    """Per-rank (peak_membw_gbs, membw_source, peak_flops_tflops, flops_source).
    MEASURED probe scores WIN over the nameplate spec (the whole point: a
    custom/OC/higher-binned card on-box drives the roofline with its REAL
    bandwidth, not the reference table); a rank whose card has neither is a hard
    miss (returns None -> no roofline)."""
    from sglang.srt.planner.card_library import CardLibrary

    library = library or CardLibrary()
    measured_by_idx = _measured_by_index(hardware, measured_scores)

    out = []
    for r in range(model.tp_size):
        gid = model.plan_inputs.rank_gpu_id[r] if model.plan_inputs.rank_gpu_id else r
        card = hardware.gpu(gid)
        prof = _profile_for(library, card.name)
        ms = measured_by_idx.get(gid)

        membw = membw_src = None
        if ms is not None:
            membw, membw_src = ms[0], "measured"
        elif prof is not None and prof.peak_membw_gbs:
            membw, membw_src = float(prof.peak_membw_gbs), "nameplate-peak"

        flops = flops_src = None
        if ms is not None:
            # The measured GEMM probe is one bf16 number; it is the real
            # achievable compute, so it wins over the dtype-specific nameplate.
            flops, flops_src = ms[1], "measured"
        elif prof is not None:
            if dtype == "fp8" and prof.peak_gemm_tflops_fp8:
                flops, flops_src = float(prof.peak_gemm_tflops_fp8), "nameplate-peak(fp8)"
            elif prof.peak_gemm_tflops_fp16:
                # fp8 model on a no-fp8-cores card up-casts to the fp16 path.
                flops, flops_src = float(prof.peak_gemm_tflops_fp16), "nameplate-peak(fp16)"

        if membw is None or flops is None:
            return None, card.name
        out.append((gid, card.name, membw, membw_src, flops, flops_src))
    return out, None


def _active_weight_terms(model, mlp_vector):
    """Per-rank (resident_active_weight_bytes, active_params, routed_active_bytes)
    streamed per token, reusing the cost model's family sharding.

    dense -> all weights; MoE -> shared expert + top-k routed experts only;
    hybrid -> per-layer via the families. MTP/draft and vision are skipped.
    ``routed_active_bytes`` is the offloadable slice (active routed experts)."""
    tp = model.tp_size
    res_bytes = [0.0] * tp
    params = [0.0] * tp
    routed_active_bytes = [0.0] * tp

    routed_share = 0.0
    active_frac = 0.0
    if model.num_experts > 0:
        moe_i = model.moe_intermediate or model.intermediate
        routed = model.num_experts * 3 * model.hidden * moe_i
        shared = 3 * model.hidden * model.shared_expert_intermediate
        routed_share = routed / (routed + shared) if (routed + shared) else 1.0
        active_frac = model.num_experts_per_tok / model.num_experts

    for name, fam in model.families.items():
        if name in _SKIP_FAMILIES or fam.params <= 0:
            continue
        mult = _VOCAB_STREAM_FRACTION if name == "vocab" else 1.0
        fracs = model._shard_fractions(fam.shard, mlp_vector)
        for r in range(tp):
            fam_b = fam.bytes * fracs[r] * mult
            fam_p = fam.params * fracs[r] * mult
            if name == "mlp" and model.num_experts > 0:
                # active = always-on shared expert + top-k routed experts.
                shared_b = fam_b * (1.0 - routed_share)
                shared_p = fam_p * (1.0 - routed_share)
                routed_b = fam_b * routed_share * active_frac
                routed_p = fam_p * routed_share * active_frac
                res_bytes[r] += shared_b + routed_b
                params[r] += shared_p + routed_p
                routed_active_bytes[r] += routed_b
            else:
                res_bytes[r] += fam_b
                params[r] += fam_p
    return res_bytes, params, routed_active_bytes


def _kv_shares(model, capacity) -> List[float]:
    """Per-rank share of the KV/token mass (weighted-DCP token split). Uses the
    plan's token vector when present, else the tp ratio, else even."""
    tp = model.tp_size
    vec = None
    if capacity is not None and getattr(capacity, "token_vector", None):
        vec = list(capacity.token_vector)
    elif model.plan_inputs.kv_token_vector:
        vec = list(model.plan_inputs.kv_token_vector)
    elif model.plan_inputs.rank_tp_ratio:
        vec = list(model.plan_inputs.rank_tp_ratio)
    if not vec or len(vec) != tp or sum(vec) <= 0:
        return [1.0 / tp] * tp
    tot = float(sum(vec))
    return [v / tot for v in vec]


def _pcie_fetch_gbs(hardware, used_gids) -> float:
    """Crude effective PCIe host-staging bandwidth (GB/s) — the MIN across the
    used cards' links (the bottleneck), at ``_PCIE_STAGING_EFFICIENCY`` of the
    theoretical lane rate. Falls back to gen4 x8 when the topology is unknown."""
    best = None
    for gid in used_gids:
        try:
            card = hardware.gpu(gid)
        except ValueError:
            continue
        gen = card.pcie_gen or _PCIE_DEFAULT_GEN
        width = card.pcie_width or _PCIE_DEFAULT_WIDTH
        gbs = _PCIE_GBS_PER_LANE.get(gen, _PCIE_GBS_PER_LANE[_PCIE_DEFAULT_GEN]) * width
        best = gbs if best is None else min(best, gbs)
    if best is None:
        best = _PCIE_GBS_PER_LANE[_PCIE_DEFAULT_GEN] * _PCIE_DEFAULT_WIDTH
    return best * _PCIE_STAGING_EFFICIENCY


def _interconnect(hardware, library, used_gids, n_cross, weight_share_by_card,
                  kv_share_by_card):
    """(tier, tier_discount, donor_surcharge, kv_donor_layout, note). The tier
    discount is the TP-collective knock-down (applies to decode AND prefill);
    the donor surcharge is the KV-traffic knock-down (decode only). Detects
    NVLink vs PCIe and the KV-donor surcharge from the weight/KV distribution."""
    from sglang.srt.planner.card_library import CardLibrary

    library = library or CardLibrary()
    if n_cross <= 1:
        return "solo", 1.0, 1.0, False, (
            "solo / single physical card in the plan — no cross-card collective"
        )

    all_nvlink = True
    for gid in used_gids:
        try:
            card = hardware.gpu(gid)
        except ValueError:
            all_nvlink = False
            break
        prof = _profile_for(library, card.name)
        if prof is None or not prof.nvlink:
            all_nvlink = False
            break

    if all_nvlink:
        tier, base = "nvlink", _NVLINK_DISCOUNT
        note = f"x{base:.2f}: TP over NVLink ({n_cross} cards) — mild collective cost"
    else:
        # Consumer multi-gen default: no GPUDirect-P2P -> host-staged collectives.
        base = _PCIE_NOP2P_BY_CROSS_CARDS.get(n_cross, _PCIE_NOP2P_MANY)
        tier = "pcie-host-staging"
        note = (
            f"x{base:.2f}: TP over PCIe, no P2P (host-staged collectives), "
            f"{n_cross} cross cards"
        )

    # KV-donor / weightless-lane surcharge (user-visible effect: shifting the KV
    # split onto cards that don't hold the weights pays cross-interconnect
    # traffic every step). Graded by weight-vs-KV total-variation divergence.
    cards = set(weight_share_by_card) | set(kv_share_by_card)
    divergence = 0.5 * sum(
        abs(kv_share_by_card.get(c, 0.0) - weight_share_by_card.get(c, 0.0))
        for c in cards
    )
    donor = divergence >= _DONOR_DIVERGENCE_NOTE
    surcharge = 1.0
    if divergence > 0 and tier != "nvlink":
        surcharge = 1.0 - _KV_DONOR_MAX_KNOCKDOWN * divergence
        if donor:
            note += (
                f"; x{surcharge:.2f} KV-donor layout (weight/KV divergence "
                f"{divergence:.2f}: cards hold KV but little weight -> "
                "attention/KV crosses the interconnect each step)"
            )
        else:
            note += f"; x{surcharge:.2f} mild weight/KV divergence {divergence:.2f}"
    elif donor:
        note += "; KV-donor layout (mild over NVLink)"
    return tier, base, surcharge, donor, note


# ---------------------------------------------------------------------------
# The estimate.
# ---------------------------------------------------------------------------


def estimate_roofline(
    inputs,
    hardware,
    capacity,
    offload=None,
    *,
    library=None,
    measured_scores=None,
    context_tokens: int = 4096,
    measured_available: bool = False,
) -> Optional[RooflineEstimate]:
    """Produce the roofline estimate for a planned config, or None when it
    cannot be sized (a card without known peak specs, or a geometry the cost
    model cannot shard). NEVER raises into the plan flow — callers guard it.

    ``capacity`` is the CapacityReport (feasible OR ram-offloadable); ``offload``
    is the OffloadAssessment (its ``ram_offload`` state routes active experts to
    the PCIe-fetch term). ``measured_scores`` are advantage.MeasuredCardScore
    (cached micro-probe) — used in preference to the nameplate peaks.
    """
    from sglang.srt.planner.card_library import CardLibrary
    from sglang.srt.uneven_perf import PerfCostModel

    library = library or CardLibrary()
    tp = inputs.tp_size
    budgets = inputs.effective_vram_mib or [1024] * tp
    base_plan = inputs.rank_tp_ratio or [1] * tp
    mlp_vector = inputs.rank_mlp_ratio or base_plan

    model = PerfCostModel(inputs, list(base_plan), list(budgets))
    cfg = PerfCostModel._load_config(inputs.model_path)
    dtype, gguf_marlin = _infer_compute(inputs.model_path, cfg)

    peaks, missing = _rank_peaks(model, hardware, library, measured_scores, dtype)
    if peaks is None:
        return None  # a card lacks both measured + nameplate specs.

    res_bytes, params, routed_active = _active_weight_terms(model, mlp_vector)

    # KV cell WITHOUT MTP (the spec/draft layers are excluded by request).
    kv_cell = model.kv_cell_bytes_per_layer * model.full_layers
    kv_shares = _kv_shares(model, capacity)
    kv_total_per_token = context_tokens * kv_cell  # full prefix, one new token

    # Host-offload split: fraction of each rank's routed experts on host RAM.
    used_gids = []
    for gid, *_ in peaks:
        if gid not in used_gids:
            used_gids.append(gid)
    n_cross = len(used_gids)
    pcie_fetch = _pcie_fetch_gbs(hardware, used_gids)

    off_frac = [0.0] * tp
    offload_note = None
    if offload is not None and offload.status == "ram_offload":
        offloadable = model.per_rank_offloadable_weight_bytes(mlp_vector)
        per_rank_off = offload.per_rank_offloaded_gib or [0.0] * tp
        for r in range(tp):
            cap_b = offloadable[r]
            off_b = per_rank_off[r] * 2**30 if r < len(per_rank_off) else 0.0
            # Fraction of routed experts on host, KNOCKED DOWN by the hot-set
            # hit-rate: hot experts stay resident, so only the cold minority of
            # the per-token active experts actually cross PCIe.
            resident_miss = min(off_b / cap_b, 1.0) if cap_b > 0 else 0.0
            off_frac[r] = resident_miss * (1.0 - _MOE_HOTSET_HIT_RATE)
        offload_note = (
            f"~{offload.offloaded_gib:.1f} GiB of MoE routed experts on host RAM: "
            f"their ACTIVE slice is fetched over PCIe (~{pcie_fetch:.0f} GB/s "
            "host-staging), so decode gains a PCIe-fetch term on top of the VRAM "
            "roofline. CALIBRATION: our 122B-A10B TP=3 offload (5090+2x3080, no "
            "NVLink) achieved ~16.3 tok/s decode WITH CUDA-graph + hot-expert "
            "residency (the launch-overhead-bound eager 6.97 is NOT the ceiling). "
            "Hot-set residency keeps most ACTIVE experts resident, so a real "
            "graph+hotset run lands well above this conservative all-active-fetch "
            "estimate."
        )

    # Aggregate weight / KV share PER PHYSICAL CARD (for the KV-donor check).
    total_w = sum(res_bytes) + sum(routed_active[r] * off_frac[r] for r in range(tp))
    total_w = total_w or 1.0
    weight_share_by_card: Dict[int, float] = {}
    kv_share_by_card: Dict[int, float] = {}
    for r in range(tp):
        gid = peaks[r][0]
        w_r = res_bytes[r] + routed_active[r] * off_frac[r]
        weight_share_by_card[gid] = weight_share_by_card.get(gid, 0.0) + w_r / total_w
        kv_share_by_card[gid] = kv_share_by_card.get(gid, 0.0) + kv_shares[r]

    tier, tier_disc, donor_surcharge, donor, ic_note = _interconnect(
        hardware, library, used_gids, n_cross, weight_share_by_card, kv_share_by_card
    )
    # Decode sees the TP-collective tier discount AND the KV-donor surcharge
    # (KV traffic is a decode-per-step cost); prefill sees only the tier
    # discount (compute-bound, far less latency-sensitive). GGUF/marlin adds the
    # mild dequant knock-down to decode.
    discount = tier_disc * donor_surcharge
    decode_discount = discount * (_GGUF_MARLIN_DECODE_DISCOUNT if gguf_marlin else 1.0)

    # -- per-rank decode time (s/token) at short + reference context --------
    per_rank: List[RankRoofline] = []
    t_high: List[float] = []
    t_low: List[float] = []
    prefill_rate_by_rank: List[float] = []
    for r in range(tp):
        gid, name, membw, membw_src, flops, flops_src = peaks[r]
        vram_bw = membw * 1e9  # GB/s -> bytes/s (decimal, matches nameplate)
        offloaded_b = routed_active[r] * off_frac[r]
        resident_b = res_bytes[r] - offloaded_b
        kv_b = kv_total_per_token * kv_shares[r]
        pcie_term = offloaded_b / (pcie_fetch * 1e9) if pcie_fetch > 0 else 0.0
        t_high.append(resident_b / vram_bw + pcie_term)
        t_low.append((resident_b + kv_b) / vram_bw + pcie_term)
        # prefill: compute-bound; 2 FLOPs per active param per token.
        flops_s = flops * 1e12
        per_tok_flops = 2.0 * params[r]
        prefill_rate_by_rank.append(flops_s / per_tok_flops if per_tok_flops > 0 else 0.0)
        per_rank.append(
            RankRoofline(
                rank=r,
                gpu_index=gid,
                gpu_name=name,
                peak_membw_gbs=membw,
                membw_source=membw_src,
                peak_flops_tflops=flops,
                flops_source=flops_src,
                resident_active_weight_gib=resident_b / 2**30,
                offloaded_active_weight_gib=offloaded_b / 2**30,
                kv_bytes_per_token_gib=kv_b / 2**30,
                active_gparams=params[r] / 1e9,
            )
        )

    slowest_high = max(t_high) if t_high else 0.0
    slowest_low = max(t_low) if t_low else 0.0
    decode_high = (EFF_DECODE * decode_discount / slowest_high) if slowest_high > 0 else 0.0
    decode_low = (EFF_DECODE * decode_discount / slowest_low) if slowest_low > 0 else 0.0
    bottleneck = t_low.index(slowest_low) if t_low else 0

    # prefill gated by the slowest rank; the TP-collective tier discount applies
    # (prefill all-reduces per layer too) but NOT the KV-donor surcharge.
    prefill_rate = min(prefill_rate_by_rank) if prefill_rate_by_rank else 0.0
    prefill_tok_s = EFF_PREFILL * tier_disc * prefill_rate

    caveats = [
        "ROUGH BALLPARK, NOT MEASURED — the runtime measures the real number. "
        "Expect this to be off by a large factor; it is order-of-magnitude only.",
        f"efficiency assumed: decode x{EFF_DECODE}, prefill x{EFF_PREFILL} "
        "(nameplate roofline is never fully reached).",
        f"decode range spans short-context (weight-bound ceiling) to the "
        f"reference {context_tokens:,}-token context; it drops further as the "
        "context fills (KV grows linearly).",
        "prefill ignores the O(n^2) attention term — prefill/token drops at "
        "long context.",
        "prefill is the COLD-CACHE compute rate: it does NOT model prefix "
        "caching (--enable-prefix-caching: cached prefix tokens skip prefill "
        "entirely, so real effective prefill on a cache hit is much higher — "
        "this is a lower-bound recompute rate) nor chunked-prefill batching. "
        "Prefix caching affects PREFILL ONLY — it has exactly zero effect on "
        "decode (a separate concept from the MoE expert wave-prefix, which IS "
        "modelled in the decode offload term above).",
    ]
    if any(p[4] and p[5].startswith("nameplate") for p in peaks):
        caveats.append(
            "FLOPS/membw are NAMEPLATE peaks (theoretical ceilings), not this "
            "rig's measured probe — real silicon runs below them."
        )
    if gguf_marlin:
        caveats.append(
            f"GGUF/marlin: on-the-fly dequant pushes decode toward compute — "
            f"extra x{_GGUF_MARLIN_DECODE_DISCOUNT} applied."
        )
    if tier == "pcie-host-staging":
        caveats.append(
            "the PCIe/no-P2P interconnect discount is a CRUDE heuristic, "
            "calibrated loosely against our own weightless-lane / hetero-TP "
            "decode; treat it as a rough knock-down, not a model."
        )

    return RooflineEstimate(
        provenance=ROOFLINE_PROVENANCE,
        decode_tok_s_high=decode_high,
        decode_tok_s_low=decode_low,
        prefill_tok_s=prefill_tok_s,
        reference_context_tokens=context_tokens,
        eff_decode=EFF_DECODE,
        eff_prefill=EFF_PREFILL,
        compute_dtype=dtype,
        gguf_marlin_dequant=gguf_marlin,
        interconnect=tier,
        interconnect_discount=discount,
        interconnect_note=ic_note,
        kv_donor_layout=donor,
        offload_note=offload_note,
        pcie_fetch_gbs=(pcie_fetch if offload_note else None),
        bottleneck_rank=bottleneck,
        per_rank=per_rank,
        caveats=caveats,
        measured_available=measured_available,
    )
