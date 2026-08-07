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
"""Granular per-card placement engine for the config-planner dashboard.

WHAT THIS COMPUTES
------------------
Given a model config dict (the ``_gguf_config_and_families`` / HF-config shape
the cost model already builds) PLUS the fork's placement flags
(``rank_gpu_id``, ``rank_tp_ratio``, ``rank_mlp_ratio``, ``rank_moe_ratio``,
``rank_vocab_ratio``, ``dcp_size`` / token vector, ``kv_cache_dtype``,
``context_length``, ``max_running_requests``), this returns a JSON-able,
per-rank / per-card placement structure usable for BOTH

  * the RUNNING config (landing page shows where things actually live), and
  * a PROSPECTIVE plan (runner tab, live-updated as flags change).

GUIDING INTERPRETATION (design_dashboard_v2 "Guiding interpretation")
---------------------------------------------------------------------
Granular placement -- which Q/K/V head, which token's KV, which expert, what
adaptive would offload -- is COMPUTED from the model config + the fork's ratio
flags. sglang exposes no live per-head / per-token physical addresses, and none
are needed: the fork's split is deterministic from the ratios (the SAME
``partition_units`` largest-remainder rounding the runtime uses). Offload is the
RULE adaptive would apply (with index / numbering), not a live memory dump.

REUSE DISCIPLINE
----------------
The byte / geometry math is NOT re-implemented here. This module constructs the
exact ``PerfCostModel`` the server builds at parse time (design "single source
of truth") and reads back its family byte model, per-rank weight bytes, mamba
pool, KV cell, and DCP token vector. The per-card MiB breakdown decomposes the
cost model's OWN per-family shard fractions, so summing the buckets reproduces
``PerfCostModel.per_rank_weight_bytes`` byte-for-byte -- the per-card view
reconciles with the fits-in-VRAM numbers by construction.

The only place this module computes a finer granularity than the cost model is
the sub-split WITHIN a family (Q vs K vs V vs O heads, individual expert
indices, vocab shard widths). Those sub-splits sum back to the family total in
the symmetric (GQA, even) cases; the two documented asymmetric cases
(replicated KV under TP > kv_heads, and an uneven ``rank_vocab_ratio``) are
flagged in the returned ``notes``.
"""

from __future__ import annotations

import dataclasses
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

from sglang.srt.planner.placement_overrides import (
    PlacementOverrideConflict,
    apply_expert_constraints,
    describe_overrides,
    parse_placement_overrides,
    resolve_expert_constraints,
)

__all__ = [
    "compute_placement",
    "compute_placement_struct",
    "PlacementFlags",
    "Placement",
]

_MIB = 2**20
_GIB = 2**30


def _auto_tp_ratio(rank_gpu_id, budgets, card_total_mib, tp):
    """Reproduce the PARSE-TIME proportional split that ``--rank-tp-ratio auto``
    derives from per-card VRAM. Returns per-rank integer weights (the raw MiB
    budgets -- ``partition_units`` only needs relative weights and splits exactly
    as the runtime would) or ``None`` when no budget info is available, in which
    case the caller falls back to an even split.
    """
    per_rank = None
    if budgets and len(budgets) == tp:
        per_rank = list(budgets)
    elif card_total_mib and rank_gpu_id and len(rank_gpu_id) == tp:
        per_rank = [card_total_mib.get(g) for g in rank_gpu_id]
    if not per_rank or any(x is None or int(x) <= 0 for x in per_rank):
        return None
    return [int(x) for x in per_rank]


# ---------------------------------------------------------------------------
# Flags contract (a thin, JSON-able mirror of the plan()/ServerArgs knobs).
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class PlacementFlags:
    """Normalized placement inputs. Every list-shaped ratio may be ``None``
    (auto / even). ``from_dict`` accepts the loose dict the webui hands over."""

    tp_size: int
    rank_gpu_id: Optional[List[int]] = None
    #: Per-rank absolute VRAM budget in MiB. Scalar (broadcast) or per-rank
    #: list. Needed only for KV-token CAPACITY numbers; placement geometry
    #: (heads / experts / token ranges) does not depend on it.
    rank_gpu_memory_mib: Optional[List[int]] = None
    rank_tp_ratio: Optional[List[int]] = None
    rank_mlp_ratio: Optional[List[int]] = None
    rank_moe_ratio: Optional[List[int]] = None
    rank_vocab_ratio: Optional[List[int]] = None
    dcp_size: Optional[int] = None
    #: Explicit uneven-DCP KV-token ownership vector (SGLANG_UNEVEN_TOKEN_VECTOR
    #: / --rank-kv-ratio). When absent the parse-time ESTIMATE is derived.
    kv_token_vector: Optional[List[int]] = None
    kv_cache_dtype: str = "auto"
    context_length: Optional[int] = None
    max_running_requests: Optional[int] = None
    speculative_algorithm: Optional[str] = None
    speculative_num_draft_tokens: Optional[int] = None
    #: Ladder shape for the CUDA-graph segment (adaptive = the #93/#102
    #: high-accept ladder: one draft-graph pair PER RUNG, itemized).
    speculative_num_steps: Optional[int] = None
    speculative_adaptive: bool = False
    #: Config dict of an EXTERNAL --speculative-draft-model-path (checkpoints
    #: without their own MTP head): sized into the draft-weights segment.
    #: The webui resolves the path to this config; placement stays pure.
    speculative_draft_model_cfg: Optional[dict] = None
    #: Precomputed CUDA-graph memory (measured live parse or boot-log
    #: anchor, graphmem shapes). None -> the calibrated heuristic.
    graph_mem_override: Optional[dict] = None
    #: Render with STOCK sglang semantics (the side-by-side normal-TP
    #: column): KV replication under tp >= kv_heads is stock head
    #: replication (num_kv_head_replicas = tp/kv, KV cache duplicated),
    #: not the fork's token-split.
    stock_semantics: bool = False
    include_vision: bool = True
    #: Fraction of routed experts adaptive keeps GPU-resident (the hot set);
    #: the remaining tail spills to host RAM. ``None`` -> read the env, else
    #: report the full routed pool as the offloadable class without pinning
    #: specific indices.
    moe_resident_expert_fraction: Optional[float] = None
    #: #396(b): declarative ``regex=target`` placement rules, in command-line
    #: order (first match wins). Fed to the solve as CONSTRAINTS -- the plan is
    #: still computed and still checked, and an unmeetable rule is refused by
    #: name rather than relaxed. ``None`` (the default, and what every caller
    #: passes today) leaves the offload rule field-for-field what it was.
    expert_placement_override: Optional[List[str]] = None
    #: Optional physical-card inventory {gpu_index: total_mib} (or name) so the
    #: per-card breakdown can show the card name + a physical-fit check. Purely
    #: informational; never required.
    #:
    #: INDEX SPACE: keys are CUDA-ORDER indices -- the same space as
    #: ``rank_gpu_id`` (whose values land in CUDA_VISIBLE_DEVICES) and as the
    #: default identity map ``rank i -> gpu i``. NVML/nvidia-smi enumeration
    #: order DIVERGES from CUDA order on mixed rigs; callers must translate
    #: NVML-sampled inventories through registry.nvml.IdentityMap before keying
    #: these dicts (an NVML-keyed inventory attributes ranks to the WRONG
    #: cards -- on the reference box rank 0 = cuda:0 is the 32 GiB 5090, not
    #: the 20 GiB 3080 that sits at nvml:0).
    card_total_mib: Optional[Dict[int, int]] = None
    card_name: Optional[Dict[int, str]] = None

    @staticmethod
    def _as_int_list(v) -> Optional[List[int]]:
        if v is None:
            return None
        if isinstance(v, str) and v.strip() in ("auto", "auto-performance"):
            return None  # string sentinel: resolved to a split by from_dict
        if isinstance(v, (list, tuple)):
            return [int(x) for x in v]
        # scalar -> single-element (broadcast handled by the caller)
        return [int(v)]

    @classmethod
    def from_dict(cls, d: dict) -> "PlacementFlags":
        d = dict(d or {})
        tp = int(d.get("tp_size") or d.get("tensor_parallel_size") or 1)
        budgets = d.get("rank_gpu_memory_mib", d.get("effective_vram_mib"))
        budgets = cls._as_int_list(budgets)
        if budgets is not None and len(budgets) == 1 and tp > 1:
            budgets = budgets * tp  # scalar broadcast, per the fork's semantics
        rgi = cls._as_int_list(d.get("rank_gpu_id"))
        ctm = (
            {int(k): int(v) for k, v in d["card_total_mib"].items()}
            if d.get("card_total_mib")
            else None
        )
        # rank_tp_ratio "auto"/"auto-performance": reproduce the parse-time
        # PROPORTIONAL split from per-rank VRAM budget (rank_gpu_memory_mib, else
        # the card each rank lands on) -- so an uneven-auto config renders uneven,
        # not even. The runtime installs the EXACT vector after profiling (same
        # caveat as the DCP token vector); this is the estimate.
        raw_rtr = d.get("rank_tp_ratio")
        if isinstance(raw_rtr, str) and raw_rtr.strip() in ("auto", "auto-performance"):
            rtr = _auto_tp_ratio(rgi, budgets, ctm, tp)
        else:
            rtr = cls._as_int_list(raw_rtr)
        return cls(
            tp_size=tp,
            rank_gpu_id=rgi,
            rank_gpu_memory_mib=budgets,
            rank_tp_ratio=rtr,
            rank_mlp_ratio=cls._as_int_list(d.get("rank_mlp_ratio")),
            rank_moe_ratio=cls._as_int_list(d.get("rank_moe_ratio")),
            rank_vocab_ratio=cls._as_int_list(d.get("rank_vocab_ratio")),
            dcp_size=(int(d["dcp_size"]) if d.get("dcp_size") else None),
            kv_token_vector=cls._as_int_list(
                d.get("kv_token_vector", d.get("rank_kv_ratio"))
            ),
            kv_cache_dtype=str(d.get("kv_cache_dtype") or "auto"),
            context_length=(
                int(d["context_length"])
                if d.get("context_length")
                else (int(d["max_model_len"]) if d.get("max_model_len") else None)
            ),
            max_running_requests=(
                int(d["max_running_requests"])
                if d.get("max_running_requests")
                else None
            ),
            speculative_algorithm=d.get("speculative_algorithm"),
            speculative_num_draft_tokens=(
                int(d["speculative_num_draft_tokens"])
                if d.get("speculative_num_draft_tokens")
                else None
            ),
            speculative_num_steps=(
                int(d["speculative_num_steps"])
                if d.get("speculative_num_steps")
                else None
            ),
            speculative_adaptive=bool(d.get("speculative_adaptive")),
            speculative_draft_model_cfg=(
                d.get("speculative_draft_model_cfg")
                if isinstance(d.get("speculative_draft_model_cfg"), dict)
                else None
            ),
            graph_mem_override=(
                d.get("graph_mem_override")
                if isinstance(d.get("graph_mem_override"), dict)
                else None
            ),
            stock_semantics=bool(d.get("stock_semantics")),
            include_vision=bool(d.get("include_vision", True)),
            moe_resident_expert_fraction=(
                float(d["moe_resident_expert_fraction"])
                if d.get("moe_resident_expert_fraction") is not None
                else None
            ),
            expert_placement_override=(
                [str(x) for x in d["expert_placement_override"]]
                if isinstance(d.get("expert_placement_override"), (list, tuple))
                else (
                    [str(d["expert_placement_override"])]
                    if d.get("expert_placement_override")
                    else None
                )
            ),
            card_total_mib=ctm,
            card_name=(
                {int(k): str(v) for k, v in d["card_name"].items()}
                if d.get("card_name")
                else None
            ),
        )


# ---------------------------------------------------------------------------
# Result dataclasses (all JSON-able via dataclasses.asdict).
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class AttnHeadPlacement:
    """Full-attention head assignment for one rank, Q / K / V / O reported
    separately as NUMBERED index ranges + per-rank MiB."""

    rank: int
    gpu_index: Optional[int]
    q_head_start: int
    q_head_end: int
    q_heads: int
    k_head_start: int
    k_head_end: int
    k_heads: int
    v_head_start: int
    v_head_end: int
    v_heads: int
    #: True when TP > num_kv_heads: the KV heads are REPLICATED on every rank
    #: and the token axis is sharded instead (uneven DCP). The K/V ranges then
    #: cover the full [0, kv_heads) on every rank.
    kv_replicated: bool
    q_mib: float
    k_mib: float
    v_mib: float
    o_mib: float
    #: The cost model's lumped attn-family share for this rank (reconciling
    #: figure). Equals q+k+v+o in the GQA case; under replicated KV the precise
    #: q/k/v/o sum exceeds this, because the cost model approximates K/V as
    #: sharded (see notes).
    attn_family_mib: float


@dataclasses.dataclass
class GdnHeadPlacement:
    """Linear-attention (GDN / SSM) head assignment for one rank -- the fork's
    hybrid models carry num_key_heads != num_value_heads (e.g. 16 key / 48
    value), so key and value heads are reported SEPARATELY and granularly."""

    rank: int
    gpu_index: Optional[int]
    k_head_start: int
    k_head_end: int
    k_heads: int
    v_head_start: int
    v_head_end: int
    v_heads: int
    replicated: bool
    mib: float


@dataclasses.dataclass
class KvTokenPlacement:
    """Where the SEQUENCE's KV lives under uneven DCP. The owner rule is
    CYCLIC, not linear (distributed/utils.py, weighted prefix-range rule):
    global slot L belongs to rank ``rank`` iff ``L % cycle_len`` falls in
    [cycle_start, cycle_end). With token vector [33,13,18] (cycle 64), rank 0
    owns positions 0..32 of EVERY 64-token block, interleaved across the whole
    context -- NOT one contiguous range. ``tokens_owned`` is the rank's total
    over the full context under that rule."""

    rank: int
    gpu_index: Optional[int]
    cycle_len: int  # S = sum(token vector); 0 when no vector/context
    cycle_start: int  # lo of this rank's within-cycle prefix range
    cycle_end: int  # hi (exclusive)
    tokens_owned: int
    #: KV bytes for the owned positions at full context (owned_tokens * cell).
    kv_mib_owned: float
    #: Predicted max KV tokens this rank COULD hold on its budget (ESTIMATE;
    #: None when no per-rank budget was supplied).
    kv_token_capacity: Optional[float]
    kv_mib_capacity: Optional[float]


@dataclasses.dataclass
class ExpertPlacement:
    """Routed-MoE expert assignment for one rank. The expert index range is the
    SAME on every MoE layer; per-rank MiB is summed over all MoE layers."""

    rank: int
    gpu_index: Optional[int]
    expert_start: int
    expert_end: int
    num_experts: int
    routed_mib: float
    #: The always-on shared expert's share for this rank (sharded by the same
    #: MoE ratio, matching how the cost model sizes it). 0.0 when the model has
    #: no shared expert.
    shared_mib: float
    per_expert_mib: float


@dataclasses.dataclass
class MtpPlacement:
    present: bool
    num_layers: int
    #: Layer indices the nextn / MTP block occupies (appended after the base
    #: layers): [layer_start, layer_end).
    layer_start: int
    layer_end: int
    per_rank_mib: List[float]


@dataclasses.dataclass
class OffloadRankRule:
    rank: int
    gpu_index: Optional[int]
    #: Routed-expert mass this rank COULD serve from host RAM (#77), MiB.
    offloadable_mib: float
    #: When a resident fraction is known: the numbered tail of experts held in
    #: host RAM and their MiB (a RULE, not a live dump).
    host_expert_start: Optional[int]
    host_expert_end: Optional[int]
    host_expert_count: int
    host_mib: float
    resident_expert_count: int
    resident_mib: float
    #: #396(b): the EXPLICIT id lists, populated only when placement overrides
    #: are in force. Without overrides the host set is the numbered tail and
    #: ``host_expert_start..host_expert_end`` says everything there is to say,
    #: so these stay ``None`` and the rule serializes exactly as it did.
    resident_ids: Optional[Tuple[int, ...]] = None
    host_ids: Optional[Tuple[int, ...]] = None


@dataclasses.dataclass
class OffloadRule:
    offloadable_class: str
    resident_fraction: Optional[float]
    per_rank: List[OffloadRankRule]
    total_offloadable_mib: float
    total_host_mib: float
    #: MTP / nextn graphs adaptive can also park in host RAM when the draft is
    #: not the bottleneck (per-rank MiB from the MTP placement).
    mtp_offloadable_mib: List[float]
    note: str


@dataclasses.dataclass
class RankBreakdown:
    """Per-rank MiB budget decomposition. weight_* sum to the cost model's
    ``per_rank_weight_bytes`` for this rank (byte-identical when vocab is even).
    """

    rank: int
    gpu_index: Optional[int]
    budget_mib: Optional[int]
    attn_mib: float
    ffn_experts_mib: float
    vocab_mib: float
    gdn_mib: float
    vision_mib: float
    mtp_mib: float
    weight_mib: float
    kv_mib: float
    mamba_mib: float
    overhead_mib: float
    total_mib: float
    # -- fine-grained sub-segments (sums reconcile with the lumps above) --
    #: attn family split by parameter proportion: q(+gate) vs k+v+o.
    #: attn_q_mib + attn_kv_mib == attn_mib byte-exact.
    attn_q_mib: float = 0.0
    attn_kv_mib: float = 0.0
    #: mlp family split: dense MLP (+ always-on shared expert) vs routed
    #: experts. mlp_dense_mib + experts_mib == ffn_experts_mib.
    mlp_dense_mib: float = 0.0
    experts_mib: float = 0.0
    #: Draft/speculative weights: the checkpoint's own MTP layers
    #: (== mtp_mib) OR an external --speculative-draft-model-path shard.
    draft_w_mib: float = 0.0
    draft_w_source: Optional[str] = None
    #: KV cache split: target layers vs draft/MTP layers (the cost model's
    #: cell spans full+mtp layers when spec is on). Sums to kv_mib.
    kv_target_mib: float = 0.0
    kv_draft_mib: float = 0.0
    #: GDN/mamba STATE pool: per-session state x session slots == mamba_mib.
    mamba_per_session_mib: float = 0.0
    mamba_sessions: int = 0
    #: Overhead split: CUDA-graph share (estimate or measured; clamped into
    #: the overhead reserve) + the workspace/context rest.
    graphs_mib: float = 0.0
    overhead_rest_mib: float = 0.0


@dataclasses.dataclass
class CardBreakdown:
    """Aggregate of every rank co-located on one physical GPU (duplicate
    rank_gpu_id => co-location => summed here)."""

    gpu_index: int
    card_name: Optional[str]
    card_total_mib: Optional[int]
    ranks: List[int]
    attn_mib: float
    ffn_experts_mib: float
    vocab_mib: float
    gdn_mib: float
    vision_mib: float
    mtp_mib: float
    weight_mib: float
    kv_mib: float
    mamba_mib: float
    overhead_mib: float
    total_mib: float
    budget_mib: Optional[int]
    #: Physical-impossibility flag: sum of co-located rank budgets exceeds the
    #: card's NVML total (only checked when card_total_mib is supplied).
    physical_overcommit: Optional[bool]
    #: Fine-grained bar segments for the renderer: ordered
    #: [{id, label, mib, detail, replicated, replication_factor,
    #:   replication_reason}] -- ONE source of truth for the segment bar AND
    #: its hover tooltips (webui renders, never re-derives). Segment MiB sums
    #: to total_mib.
    segments: List[dict] = dataclasses.field(default_factory=list)
    graphs_mib: float = 0.0
    overhead_rest_mib: float = 0.0


@dataclasses.dataclass
class Placement:
    model: dict
    tp_size: int
    dcp_size: int
    ranks_per_gpu: Dict[int, List[int]]
    token_vector: List[int]
    token_vector_source: str
    attn_heads: List[AttnHeadPlacement]
    gdn_heads: Optional[List[GdnHeadPlacement]]
    kv_tokens: List[KvTokenPlacement]
    experts: Optional[List[ExpertPlacement]]
    shared_expert: Optional[dict]
    mtp: MtpPlacement
    offload: Optional[OffloadRule]
    ranks: List[RankBreakdown]
    cards: List[CardBreakdown]
    notes: List[str]
    #: CUDA-graph memory block: {provenance: "measured"|"heuristic",
    #: per_rank_mib, items: [{label, mib, formula}...], error_band_pct,
    #: formula} -- itemized per graph kind AND ladder rung.
    graph_mem: Optional[dict] = None
    #: Explicit KV-replication marker: {replicated, factor, source:
    #: "fork-uneven-dcp"|"stock-replication", reason, cache_duplicated}.
    kv_replication: Optional[dict] = None
    #: Draft/speculative summary: {weights_source: "mtp"|"external"|None,
    #: per_rank_mib, kv_draft_per_rank_mib, note}.
    draft: Optional[dict] = None


# ---------------------------------------------------------------------------
# Cost-model construction with an injected config (no disk read).
# ---------------------------------------------------------------------------
def _build_cost_model(model_cfg: dict, flags: PlacementFlags, budgets_mib, base_plan):
    """Construct the fork's ``PerfCostModel`` on the given config dict.

    The cost model normally reads the config from ``model_path`` on disk; here
    we inject the caller-supplied ``model_cfg`` (the same shape
    ``_gguf_config_and_families`` returns) so placement is a pure function of
    config + flags -- no checkpoint access, no GPU.
    """
    from sglang.srt.uneven_perf import PerfCostModel, PlanInputs

    class _InjectedCostModel(PerfCostModel):
        def __init__(self, plan_inputs, base, budg, cfg):
            self._injected_cfg = cfg
            super().__init__(plan_inputs, base, budg)

        def _load_config(self, model_path):  # overrides the static loader
            return self._injected_cfg

    plan_inputs = PlanInputs(
        tp_size=flags.tp_size,
        model_path="<in-memory-config>",
        kv_cache_dtype=flags.kv_cache_dtype,
        speculative_algorithm=flags.speculative_algorithm,
        speculative_num_draft_tokens=flags.speculative_num_draft_tokens,
        max_running_requests=flags.max_running_requests,
        include_vision=flags.include_vision,
        rank_gpu_id=list(flags.rank_gpu_id) if flags.rank_gpu_id else None,
        effective_vram_mib=list(budgets_mib),
        rank_tp_ratio=list(base_plan),
        rank_mlp_ratio=flags.rank_mlp_ratio,
        rank_moe_ratio=flags.rank_moe_ratio,
        rank_vocab_ratio=flags.rank_vocab_ratio,
        dcp_size=flags.dcp_size,
        kv_token_vector=flags.kv_token_vector,
    )
    return _InjectedCostModel(
        plan_inputs, list(base_plan), list(budgets_mib), model_cfg
    )


# ---------------------------------------------------------------------------
# Small partition helpers.
# ---------------------------------------------------------------------------
def _prefix_ranges(counts: Sequence[int]):
    """[(start, end), ...] contiguous half-open ranges from a per-rank count
    vector."""
    out, c = [], 0
    for n in counts:
        out.append((c, c + int(n)))
        c += int(n)
    return out


def _largest_remainder(total: int, weights: Sequence[int]) -> List[int]:
    """Split ``total`` items across ranks proportional to ``weights`` with
    largest-remainder rounding; sums to EXACTLY ``total`` with no gaps or
    overlaps (a rank may receive 0 when its weight rounds down)."""
    n = len(weights)
    sw = sum(weights)
    if sw <= 0 or total <= 0:
        base = [0] * n
        if total > 0:
            base[0] = total
        return base
    quotas = [total * w / sw for w in weights]
    sizes = [int(q) for q in quotas]
    rem = total - sum(sizes)
    order = sorted(
        range(n), key=lambda r: (quotas[r] - int(quotas[r]), -r), reverse=True
    )
    for k in range(rem):
        sizes[order[k % n]] += 1
    assert sum(sizes) == total
    return sizes


# ---------------------------------------------------------------------------
# Public entry points.
# ---------------------------------------------------------------------------
def compute_placement(model_cfg: dict, flags) -> dict:
    """Compute the granular per-rank / per-card placement (JSON-able dict).

    ``model_cfg`` is the ``_gguf_config_and_families`` / HF-config shaped dict
    (``text_config`` with num_attention_heads, num_key_value_heads, head_dim,
    num_hidden_layers, num_experts, moe_intermediate_size, mtp_num_hidden_layers,
    the GDN linear-head geometry, etc.). ``flags`` is a :class:`PlacementFlags`
    or a loose dict accepted by :meth:`PlacementFlags.from_dict`.
    """
    return dataclasses.asdict(compute_placement_struct(model_cfg, flags))


def compute_placement_struct(model_cfg: dict, flags) -> Placement:
    if not isinstance(flags, PlacementFlags):
        flags = PlacementFlags.from_dict(flags)
    tp = flags.tp_size
    notes: List[str] = []

    # ---- rank -> GPU map (duplicates = co-location) ----------------------
    if flags.rank_gpu_id and len(flags.rank_gpu_id) == tp:
        rank_gpu = list(flags.rank_gpu_id)
    else:
        if flags.rank_gpu_id:
            notes.append(
                f"rank_gpu_id length {len(flags.rank_gpu_id)} != tp_size {tp}; "
                "defaulted to one rank per card (0..tp-1)."
            )
        rank_gpu = list(range(tp))
    ranks_per_gpu: Dict[int, List[int]] = {}
    for r, g in enumerate(rank_gpu):
        ranks_per_gpu.setdefault(g, []).append(r)

    # ---- base plan (attention / KV / DCP split) + the MLP/MoE vector -----
    base_plan = flags.rank_tp_ratio if flags.rank_tp_ratio else [1] * tp
    if len(base_plan) != tp:
        notes.append(
            f"rank_tp_ratio length {len(base_plan)} != tp_size {tp}; using even."
        )
        base_plan = [1] * tp

    # ---- budgets (optional; only KV CAPACITY needs them) -----------------
    budgets = flags.rank_gpu_memory_mib
    have_budgets = bool(budgets and len(budgets) == tp)
    if budgets and len(budgets) != tp:
        notes.append(
            f"rank_gpu_memory_mib length {len(budgets)} != tp_size {tp}; "
            "KV-capacity numbers omitted."
        )
    budgets_for_model = list(budgets) if have_budgets else [1] * tp

    # Whether this is a MoE checkpoint decides which ratio drives the FFN
    # family: rank_moe_ratio (MoE) vs rank_mlp_ratio (dense). We feed the SAME
    # vector to the cost model that we use to place the experts, so the
    # per-card breakdown reconciles with per_rank_weight_bytes by construction.
    text = model_cfg.get("text_config", model_cfg)
    num_experts = int(text.get("num_experts", text.get("n_routed_experts", 0)) or 0)
    if num_experts > 0:
        ffn_vector = flags.rank_moe_ratio or flags.rank_mlp_ratio or base_plan
    else:
        ffn_vector = flags.rank_mlp_ratio or base_plan
    if len(ffn_vector) != tp:
        ffn_vector = list(base_plan)

    model = _build_cost_model(model_cfg, flags, budgets_for_model, base_plan)

    # ---- family byte decomposition (reconciling per-rank/per-card) -------
    # per-family per-rank bytes using the cost model's OWN shard fractions, so
    # the sum equals per_rank_weight_bytes(ffn_vector) exactly.
    fam_frac = {
        name: model._shard_fractions(fam.shard, ffn_vector)
        for name, fam in model.families.items()
        if fam.params > 0
    }
    fam_bytes = {
        name: [model.families[name].bytes * fam_frac[name][r] for r in range(tp)]
        for name in fam_frac
    }

    def fam_r(name: str, r: int) -> float:
        return fam_bytes.get(name, [0.0] * tp)[r]

    # ---- attention head placement ---------------------------------------
    attn_heads = _compute_attn_heads(
        model, base_plan, rank_gpu, fam_bytes, notes, flags=flags
    )

    # ---- GDN / linear-attention head placement ---------------------------
    gdn_heads = _compute_gdn_heads(model, base_plan, rank_gpu, fam_bytes, notes)

    # ---- MoE expert placement + shared expert ----------------------------
    experts, shared_expert = _compute_experts(
        model, ffn_vector, rank_gpu, num_experts, notes
    )

    # ---- MTP / nextn placement -------------------------------------------
    mtp = _compute_mtp(model, fam_bytes, tp)

    # ---- DCP token-range placement + per-rank KV capacity ----------------
    token_vector, tv_source, capacity = _resolve_token_vector(
        model, flags, base_plan, have_budgets, notes
    )
    kv_tokens = _compute_kv_tokens(
        model, flags, rank_gpu, token_vector, capacity, tp, notes
    )

    # ---- KV-replication marker + draft weights + CUDA-graph estimate ----
    kv_replication = _kv_replication_info(model, flags, base_plan, attn_heads)
    draft_source, draft_w, draft_note = _draft_weights(
        model, flags, mtp, base_plan, tp, notes
    )
    graph_mem = _graph_mem(model, flags, tp)

    # ---- per-rank + per-card MiB breakdown -------------------------------
    ranks, cards = _compute_breakdown(
        model,
        flags,
        rank_gpu,
        ranks_per_gpu,
        fam_r,
        capacity,
        budgets if have_budgets else None,
        kv_tokens,
        tp,
        attn_heads=attn_heads,
        experts=experts,
        kv_replication=kv_replication,
        draft_source=draft_source,
        draft_w=draft_w,
        graph_mem=graph_mem,
    )

    # ---- offload rule ----------------------------------------------------
    offload = _compute_offload(
        model, flags, ffn_vector, rank_gpu, experts, mtp, num_experts, notes
    )

    dcp_size = flags.dcp_size or tp
    model_summary = {
        "hidden_size": model.hidden,
        "num_attention_heads": model.q_heads,
        "num_key_value_heads": model.kv_heads,
        "head_dim": model.head_dim,
        "num_hidden_layers": model.n_layers,
        "full_attention_layers": model.full_layers,
        "gdn_layers": model.gdn_layers,
        "num_experts": num_experts,
        "num_experts_per_tok": model.num_experts_per_tok,
        "moe_intermediate_size": model.moe_intermediate,
        "shared_expert_intermediate_size": model.shared_expert_intermediate,
        "mtp_num_hidden_layers": model.mtp_layers,
        "vocab_size": model.vocab,
        "linear_num_key_heads": model.gdn_k_heads,
        "linear_num_value_heads": model.gdn_v_heads,
        "kv_cache_dtype": flags.kv_cache_dtype,
        "kv_cell_bytes_per_token": model.kv_cell_bytes,
    }

    return Placement(
        model=model_summary,
        tp_size=tp,
        dcp_size=dcp_size,
        ranks_per_gpu=ranks_per_gpu,
        token_vector=list(token_vector),
        token_vector_source=tv_source,
        attn_heads=attn_heads,
        gdn_heads=gdn_heads,
        kv_tokens=kv_tokens,
        experts=experts,
        shared_expert=shared_expert,
        mtp=mtp,
        offload=offload,
        ranks=ranks,
        cards=cards,
        notes=notes,
        graph_mem=graph_mem,
        kv_replication=kv_replication,
        draft=(
            {
                "weights_source": draft_source,
                "per_rank_mib": list(draft_w),
                "kv_draft_per_rank_mib": [r.kv_draft_mib for r in ranks],
                "note": draft_note,
            }
            if draft_source
            else None
        ),
    )


# ---------------------------------------------------------------------------
# Sub-computations.
# ---------------------------------------------------------------------------
def _uneven_dcp_active(flags, base_plan) -> bool:
    """Plan-time mirror of the runtime's ``uneven_dcp_kv_replicated``.

    The body used to be spelled out here; #503 moved it next to the runtime
    predicate it mirrors (``distributed/utils.py``) so the two cannot drift,
    and so the phase-prefill enumerator reads the SAME function instead of a
    third spelling of it. Kept as a named wrapper because this module's own
    readers and tests know the condition by this name, and because
    ``placement`` imports ``distributed.utils`` lazily throughout.
    """
    from sglang.srt.distributed.utils import plan_uneven_dcp_kv_replicated

    return plan_uneven_dcp_kv_replicated(flags, base_plan)


def _compute_attn_heads(model, base_plan, rank_gpu, fam_bytes, notes, flags=None):
    from sglang.srt.distributed.utils import partition_units

    tp = model.tp_size
    q_heads, kv_heads, head_dim = model.q_heads, model.kv_heads, model.head_dim
    gate = 2 if model.attn_gate else 1

    # Per-family attn bytes (base layers) already reconciled in fam_bytes;
    # decompose into q/k/v/o by parameter proportion of one attn layer.
    H = model.hidden
    bpp = model.families["attn"].bytes_per_param
    layers = model.full_layers
    q_bytes = H * q_heads * head_dim * gate * bpp * layers
    k_bytes = H * kv_heads * head_dim * bpp * layers
    v_bytes = H * kv_heads * head_dim * bpp * layers
    o_bytes = q_heads * head_dim * H * bpp * layers

    attn_family = fam_bytes.get("attn", [0.0] * tp)

    # KV heads are replicated on TWO paths (matching the runtime's
    # uneven_dcp_kv_replicated): (a) TP > kv_heads (head-sharding impossible),
    # and (b) uneven DCP in force (dcp spans the TP group with a non-uniform
    # ratio) -- there the KV cache is split on the TOKEN axis and every rank
    # keeps the FULL kv-head set, EVEN when kv_heads >= tp. Showing 4:2:2
    # head-sharded KV for the reference uneven-DCP config was wrong.
    dcp_replicated = _uneven_dcp_active(flags, base_plan)
    replicated = kv_heads < tp or dcp_replicated
    if not replicated:
        # GQA / MHA (coupled head-sharding, no DCP): shard the KV-head grid;
        # Q heads follow at the group size.
        kv_units = partition_units(kv_heads, base_plan)
        group = max(q_heads // kv_heads, 1)
        q_per = [u * group for u in kv_units]
        k_per = list(kv_units)
        v_per = list(kv_units)
        q_ranges = _prefix_ranges(q_per)
        k_ranges = _prefix_ranges(k_per)
        v_ranges = k_ranges
    else:
        # Replicate the KV heads on every rank, shard the Q heads on the
        # q-head grid (kv-group aligned, matching the runtime).
        reason = (
            f"TP={tp} > num_key_value_heads={kv_heads}"
            if kv_heads < tp
            else f"uneven DCP in force (non-uniform ratio, kv_heads={kv_heads} >= TP={tp})"
        )
        notes.append(
            f"{reason}: KV heads are REPLICATED "
            "on every rank and the token axis is sharded (uneven DCP). Granular "
            "K/V MiB counts the full replicated heads per rank, so it exceeds "
            "the cost model's lumped attn share (which approximates K/V as "
            "sharded on the q-head grid)."
        )
        grid = max(q_heads, tp)
        q_per = partition_units(
            grid, base_plan, groups=(kv_heads if kv_heads >= 2 else None)
        )
        q_ranges = _prefix_ranges(q_per)
        k_per = [kv_heads] * tp
        v_per = [kv_heads] * tp
        k_ranges = [(0, kv_heads)] * tp
        v_ranges = [(0, kv_heads)] * tp

    out = []
    for r in range(tp):
        qf = q_per[r] / q_heads if q_heads else 0.0
        if replicated:
            kf = 1.0  # full replicated K/V
            vf = 1.0
        else:
            kf = k_per[r] / kv_heads if kv_heads else 0.0
            vf = v_per[r] / kv_heads if kv_heads else 0.0
        out.append(
            AttnHeadPlacement(
                rank=r,
                gpu_index=rank_gpu[r],
                q_head_start=q_ranges[r][0],
                q_head_end=q_ranges[r][1],
                q_heads=q_per[r],
                k_head_start=k_ranges[r][0],
                k_head_end=k_ranges[r][1],
                k_heads=k_per[r],
                v_head_start=v_ranges[r][0],
                v_head_end=v_ranges[r][1],
                v_heads=v_per[r],
                kv_replicated=replicated,
                q_mib=qf * q_bytes / _MIB,
                k_mib=kf * k_bytes / _MIB,
                v_mib=vf * v_bytes / _MIB,
                o_mib=qf * o_bytes / _MIB,
                attn_family_mib=attn_family[r] / _MIB,
            )
        )
    return out


def _compute_gdn_heads(model, base_plan, rank_gpu, fam_bytes, notes):
    from sglang.srt.distributed.utils import partition_units

    if not model.gdn_layers or not model.gdn_k_heads:
        return None
    tp = model.tp_size
    k_heads, v_heads = model.gdn_k_heads, model.gdn_v_heads
    gdn_family = fam_bytes.get("gdn", [0.0] * tp)

    if k_heads >= tp:
        k_units = partition_units(k_heads, base_plan)
        vgroup = max(v_heads // max(k_heads, 1), 1)
        v_per = [u * vgroup for u in k_units]
        k_ranges = _prefix_ranges(k_units)
        v_ranges = _prefix_ranges(v_per)
        replicated = False
    else:
        notes.append(
            f"TP={tp} > linear_num_key_heads={k_heads}: GDN/SSM state is "
            "replicated across ranks (heads are indivisible below the rank "
            "count)."
        )
        k_units = [k_heads] * tp
        v_per = [v_heads] * tp
        k_ranges = [(0, k_heads)] * tp
        v_ranges = [(0, v_heads)] * tp
        replicated = True

    return [
        GdnHeadPlacement(
            rank=r,
            gpu_index=rank_gpu[r],
            k_head_start=k_ranges[r][0],
            k_head_end=k_ranges[r][1],
            k_heads=k_units[r],
            v_head_start=v_ranges[r][0],
            v_head_end=v_ranges[r][1],
            v_heads=v_per[r],
            replicated=replicated,
            mib=gdn_family[r] / _MIB,
        )
        for r in range(tp)
    ]


def _compute_experts(model, ffn_vector, rank_gpu, num_experts, notes):
    from sglang.srt.distributed.utils import partition_units

    if num_experts <= 0:
        return None, None
    tp = model.tp_size
    if num_experts < tp:
        notes.append(
            f"num_experts={num_experts} < tp_size={tp}: fewer experts than "
            "ranks; expert placement falls back to an even (possibly empty) "
            "split."
        )
        counts = _largest_remainder(num_experts, ffn_vector)
    else:
        counts = partition_units(num_experts, ffn_vector)

    H = model.hidden
    moe_i = model.moe_intermediate or model.intermediate
    bpp = model.families["mlp"].bytes_per_param
    layers = model.n_layers
    per_expert_bytes = 3 * H * moe_i * bpp * layers
    shared_total = 3 * H * model.shared_expert_intermediate * bpp * layers

    ranges = _prefix_ranges(counts)
    out = []
    for r in range(tp):
        efrac = counts[r] / num_experts if num_experts else 0.0
        out.append(
            ExpertPlacement(
                rank=r,
                gpu_index=rank_gpu[r],
                expert_start=ranges[r][0],
                expert_end=ranges[r][1],
                num_experts=counts[r],
                routed_mib=counts[r] * per_expert_bytes / _MIB,
                shared_mib=shared_total * efrac / _MIB,
                per_expert_mib=per_expert_bytes / _MIB,
            )
        )
    shared = None
    if model.shared_expert_intermediate > 0:
        shared = {
            "intermediate_size": model.shared_expert_intermediate,
            "handling": "always-on; sharded by the same MoE ratio (matches the "
            "cost model), summed over all MoE layers.",
            "per_rank_mib": [o.shared_mib for o in out],
            "total_mib": shared_total / _MIB,
        }
    return out, shared


def _compute_mtp(model, fam_bytes, tp):
    present = bool(model.mtp_layers and model.spec_active)
    per_rank = [0.0] * tp
    if present:
        for name in ("draft_attn", "draft_mlp", "draft_repl"):
            fb = fam_bytes.get(name)
            if fb:
                for r in range(tp):
                    per_rank[r] += fb[r] / _MIB
    return MtpPlacement(
        present=present,
        num_layers=model.mtp_layers,
        layer_start=model.n_layers,
        layer_end=model.n_layers + model.mtp_layers,
        per_rank_mib=per_rank,
    )


def _resolve_token_vector(model, flags, base_plan, have_budgets, notes):
    """Effective uneven-DCP token-ownership vector + its provenance.

    Precedence (mirrors resolve_cp_token_ratios): explicit kv_token_vector >
    parse-time capacity estimate (predict_capacity) > gcd-reduced rank_tp_ratio
    > even. HONEST NOTE: the runtime installs a MEASURED optimal vector AFTER
    post-weight-load KV profiling (capacity mode); from flags alone only the
    parse-time ESTIMATE is recoverable. That caveat is added to ``notes``.
    """
    tp = model.tp_size
    capacity = None
    if have_budgets:
        try:
            mlp_vec = flags.rank_mlp_ratio or list(base_plan)
            if len(mlp_vec) != tp:
                mlp_vec = list(base_plan)
            capacity = model.predict_capacity(mlp_vec)
        except Exception:
            capacity = None

    if flags.kv_token_vector and len(flags.kv_token_vector) == tp:
        return list(flags.kv_token_vector), "explicit_flag", capacity

    if capacity and capacity.get("token_vector"):
        notes.append(
            "DCP token vector is the PARSE-TIME estimate (proportional to each "
            "rank's predicted free-VRAM KV capacity). The runtime replaces it "
            "with the measured optimum after profiling; not recoverable from "
            "flags alone."
        )
        return list(capacity["token_vector"]), "capacity_estimate", capacity

    if len(set(base_plan)) > 1:
        g = math.gcd(*base_plan)
        return [w // g for w in base_plan], "rank_tp_ratio", capacity

    return [1] * tp, "even", capacity


def _compute_kv_tokens(model, flags, rank_gpu, token_vector, capacity, tp, notes):
    """CYCLIC weighted prefix-range owner rule, mirroring the runtime
    (distributed/utils.py): S = sum(token_vector); rank r owns the within-cycle
    prefix range [lo_r, hi_r); global slot L belongs to r iff L % S in
    [lo_r, hi_r). tokens_owned counts that rule exactly over the context --
    the earlier linear "rank r owns tokens a..b" rendering was WRONG."""
    context = flags.context_length or 0
    cell = model.kv_cell_bytes
    cycle = sum(token_vector) if token_vector else 0
    ranges = _prefix_ranges(list(token_vector)) if token_vector else [(0, 0)] * tp
    if not context:
        notes.append(
            "context_length not supplied: KV token placement reports the "
            "within-cycle ownership ranges only (no absolute totals)."
        )
    out = []
    for r in range(tp):
        lo, hi = ranges[r]
        if context and cycle:
            full, rem = divmod(context, cycle)
            owned = full * (hi - lo) + max(0, min(rem, hi) - lo)
        else:
            owned = 0
        cap = capacity["p"][r] if capacity else None
        out.append(
            KvTokenPlacement(
                rank=r,
                gpu_index=rank_gpu[r],
                cycle_len=cycle,
                cycle_start=lo,
                cycle_end=hi,
                tokens_owned=owned,
                kv_mib_owned=owned * cell / _MIB,
                kv_token_capacity=(float(cap) if cap is not None else None),
                kv_mib_capacity=(float(cap) * cell / _MIB if cap is not None else None),
            )
        )
    return out


def _kv_replication_info(model, flags, base_plan, attn_heads):
    """Explicit KV-replication marker for the VRAM display. Two roots:

    * fork path: uneven DCP in force, or TP > kv_heads -- every rank keeps
      the FULL kv-head set, the KV CACHE is sharded on the token axis
      (weights replicated, cache NOT duplicated);
    * stock path (``stock_semantics``): tp >= kv with tp % kv == 0 --
      QKVParallelLinear replicates each kv head tp/kv times
      (num_kv_head_replicas) AND each rank keeps the KV cache for its
      replicated head (weights + cache duplicated).
    """
    tp = model.tp_size
    kv = model.kv_heads
    replicated = bool(attn_heads and attn_heads[0].kv_replicated)
    if not replicated:
        return {
            "replicated": False,
            "factor": 1,
            "source": None,
            "cache_duplicated": False,
            "reason": (
                f"KV heads sharded normally (kv_heads={kv} >= TP={tp}, "
                "uniform plan; every head lives on exactly one rank)."
            ),
        }
    if flags.stock_semantics and kv and tp >= kv and tp % kv == 0:
        factor = tp // kv
        return {
            "replicated": True,
            "factor": factor,
            "source": "stock-replication",
            "cache_duplicated": True,
            "reason": (
                f"stock KV-head replication: tp={tp} >= kv_heads={kv} with "
                f"tp % kv == 0 -> num_kv_head_replicas={factor} "
                "(QKVParallelLinear); the replicated heads' KV WEIGHTS AND "
                "KV CACHE are duplicated on every rank of a replica group."
            ),
        }
    why = (
        f"TP={tp} > num_key_value_heads={kv}"
        if kv < tp
        else "uneven DCP in force (non-uniform rank ratio)"
    )
    return {
        "replicated": True,
        "factor": tp,
        "source": "fork-uneven-dcp",
        "cache_duplicated": False,
        "reason": (
            f"fork uneven-DCP token-split: {why} -> every rank keeps the "
            f"FULL kv-head set ({kv} heads, KV weights replicated x{tp}); "
            "the KV CACHE is sharded on the token axis instead of the head "
            "axis, so cache tokens are NOT duplicated."
        ),
    }


def _draft_weights(model, flags, mtp, base_plan, tp, notes):
    """Draft-weight segment source + per-rank MiB: the checkpoint's own MTP
    layers (already in the family model), or an EXTERNAL
    --speculative-draft-model-path config sized through the same cost model
    (tp=1 total, sharded by the base plan -- an estimate, noted)."""
    if mtp.present:
        return (
            "mtp",
            list(mtp.per_rank_mib),
            (
                f"draft = the checkpoint's own {mtp.num_layers} MTP/nextn "
                f"layer(s), appended at layers [{mtp.layer_start}.."
                f"{mtp.layer_end})."
            ),
        )
    if flags.speculative_algorithm and flags.speculative_draft_model_cfg:
        try:
            draft_flags = PlacementFlags(
                tp_size=1,
                kv_cache_dtype=flags.kv_cache_dtype,
                include_vision=False,
            )
            dm = _build_cost_model(
                flags.speculative_draft_model_cfg, draft_flags, [1], [1]
            )
            total = float(dm.per_rank_weight_bytes([1])[0])
        except Exception as e:  # pragma: no cover - defensive
            notes.append(f"external draft model could not be sized: {e}")
            return None, [0.0] * tp, ""
        sw = sum(base_plan) or tp
        per_rank = [total * w / sw / _MIB for w in base_plan]
        notes.append(
            "draft weights are an EXTERNAL --speculative-draft-model-path sized "
            "from its config and sharded by the base plan (estimate)."
        )
        return (
            "external",
            per_rank,
            (
                "draft = external --speculative-draft-model-path "
                f"({total / _GIB:.2f} GiB total), sharded by the base TP plan "
                "(estimate)."
            ),
        )
    return None, [0.0] * tp, ""


def _graph_mem(model, flags, tp):
    """CUDA-graph memory block: the caller-supplied override (measured live
    parse or boot-log anchor) wins; else the calibrated heuristic
    (graphmem) -- ladder-aware, so changing k/adaptive changes the number."""
    if flags.graph_mem_override:
        d = dict(flags.graph_mem_override)
        d.setdefault("provenance", "measured")
        return d
    from sglang.srt.planner import graphmem

    geom = {
        "hidden_size": model.hidden,
        "num_hidden_layers": model.n_layers,
        "tp_size": tp,
        "max_running_requests": flags.max_running_requests or 16,
    }
    spec = {
        "speculative_algorithm": flags.speculative_algorithm,
        "speculative_num_steps": flags.speculative_num_steps,
        "speculative_num_draft_tokens": flags.speculative_num_draft_tokens,
        "speculative_adaptive": flags.speculative_adaptive,
    }
    return graphmem.heuristic_estimate(geom, spec)


def _graph_mib_for_rank(graph_mem, r):
    """Per-rank graph MiB from a graph_mem block: an explicit per-rank map
    (measured parses carry one; draft graphs may sit on rank 0 only) wins
    over the uniform per_rank_mib scalar."""
    if not graph_mem:
        return 0.0
    m = graph_mem.get("per_rank_map")
    if m:
        return float(m.get(str(r), m.get(r, 0.0)))
    return float(graph_mem.get("per_rank_mib") or 0.0)


def _mamba_sessions(model, flags):
    """Session-slot count behind the mamba pool. MIRRORS the sizing in
    ``PerfCostModel._mamba_pool_bytes`` (slots = ceil(min(target,48) * 5 *
    1.25) + spec pad) so per_session * sessions == pool byte-exact -- the
    slot arithmetic itself lives in ``mrr_balance.state_slot_count``, called
    here with the predictor's target clamp so there is one copy of it."""
    from sglang.srt.planner.mrr_balance import (
        PREDICTOR_TARGET_CLAMP,
        state_slot_count,
    )

    return state_slot_count(
        flags.max_running_requests or 16,
        draft_tokens=model.spec_draft_tokens if model.spec_active else 0,
        clamp=PREDICTOR_TARGET_CLAMP,
    )


def _compute_breakdown(
    model,
    flags,
    rank_gpu,
    ranks_per_gpu,
    fam_r,
    capacity,
    budgets,
    kv_tokens,
    tp,
    attn_heads=None,
    experts=None,
    kv_replication=None,
    draft_source=None,
    draft_w=None,
    graph_mem=None,
):
    overhead_mib = _overhead_mib()
    cell = model.kv_cell_bytes
    kv_repl = kv_replication or {}
    draft_w = draft_w or [0.0] * tp

    # KV cell layer split: with spec on, the cell spans full + MTP layers --
    # the MTP share IS the draft-KV/verify footprint per token.
    cell_layers = model.full_layers + (model.mtp_layers if model.spec_active else 0)
    draft_layer_frac = (
        model.mtp_layers / cell_layers
        if (model.spec_active and model.mtp_layers and cell_layers)
        else 0.0
    )

    sessions = _mamba_sessions(model, flags)

    ranks = []
    for r in range(tp):
        attn = (fam_r("attn", r)) / _MIB
        ffn = fam_r("mlp", r) / _MIB
        vocab = fam_r("vocab", r) / _MIB
        gdn = fam_r("gdn", r) / _MIB
        vision = fam_r("vision", r) / _MIB
        mtp = (
            fam_r("draft_attn", r) + fam_r("draft_mlp", r) + fam_r("draft_repl", r)
        ) / _MIB
        weight = attn + ffn + vocab + gdn + vision + mtp

        # attn family sub-split: q(+gate) vs k+v+o, scaled onto the family
        # share so attn_q + attn_kv == attn byte-exact (in the symmetric GQA
        # case the scale is 1.0 and the granular numbers match unscaled).
        ah = attn_heads[r] if attn_heads else None
        if ah is not None:
            raw_q = ah.q_mib
            raw_kvo = ah.k_mib + ah.v_mib + ah.o_mib
            raw = raw_q + raw_kvo
            scale = (attn / raw) if raw > 0 else 0.0
            attn_q = raw_q * scale
            attn_kv = raw_kvo * scale
        else:
            attn_q, attn_kv = attn, 0.0

        # mlp family sub-split: routed experts vs dense (+ shared expert).
        ex = experts[r] if experts else None
        experts_mib = min(ex.routed_mib, ffn) if ex else 0.0
        mlp_dense = max(ffn - experts_mib, 0.0)

        # external draft weights add ON TOP of the family model (the cost
        # model only carries the checkpoint's own MTP families).
        draft_mib = draft_w[r] if draft_source else 0.0
        if draft_source == "external":
            weight += draft_mib
        elif draft_source == "mtp":
            draft_mib = mtp

        if capacity and capacity["p"][r] > 0:
            kv = capacity["p"][r] * cell / _MIB
        else:
            kv = 0.0
        kv_draft = kv * draft_layer_frac
        kv_target = kv - kv_draft

        mamba = model.mamba_pool_bytes[r] / _MIB
        per_session = mamba / sessions if sessions else 0.0

        graphs = _graph_mib_for_rank(graph_mem, r)
        # The graph share is displayed INSIDE the fixed overhead reserve
        # (the capacity math already budgets it there); the raw estimate
        # stays un-clamped in the graph_mem block.
        graphs_shown = min(graphs, float(overhead_mib))
        rest = float(overhead_mib) - graphs_shown

        budget = int(budgets[r]) if budgets else None
        ranks.append(
            RankBreakdown(
                rank=r,
                gpu_index=rank_gpu[r],
                budget_mib=budget,
                attn_mib=attn,
                ffn_experts_mib=ffn,
                vocab_mib=vocab,
                gdn_mib=gdn,
                vision_mib=vision,
                mtp_mib=mtp,
                weight_mib=weight,
                kv_mib=kv,
                mamba_mib=mamba,
                overhead_mib=float(overhead_mib),
                total_mib=weight + kv + mamba + overhead_mib,
                attn_q_mib=attn_q,
                attn_kv_mib=attn_kv,
                mlp_dense_mib=mlp_dense,
                experts_mib=experts_mib,
                draft_w_mib=draft_mib,
                draft_w_source=draft_source,
                kv_target_mib=kv_target,
                kv_draft_mib=kv_draft,
                mamba_per_session_mib=per_session,
                mamba_sessions=sessions,
                graphs_mib=graphs_shown,
                overhead_rest_mib=rest,
            )
        )

    cards = []
    for gpu_index in sorted(ranks_per_gpu):
        rs = ranks_per_gpu[gpu_index]
        agg = {
            k: sum(getattr(ranks[r], k) for r in rs)
            for k in (
                "attn_mib",
                "ffn_experts_mib",
                "vocab_mib",
                "gdn_mib",
                "vision_mib",
                "mtp_mib",
                "weight_mib",
                "kv_mib",
                "mamba_mib",
                "overhead_mib",
                "total_mib",
                "graphs_mib",
                "overhead_rest_mib",
            )
        }
        card_total = None
        if flags.card_total_mib and gpu_index in flags.card_total_mib:
            card_total = int(flags.card_total_mib[gpu_index])
        card_name = None
        if flags.card_name and gpu_index in flags.card_name:
            card_name = flags.card_name[gpu_index]
        budget = sum(int(budgets[r]) for r in rs) if budgets else None
        overcommit = None
        if card_total is not None and budget is not None:
            overcommit = budget > card_total
        segments = _card_segments(model, flags, ranks, rs, kv_repl, graph_mem, sessions)
        cards.append(
            CardBreakdown(
                gpu_index=gpu_index,
                card_name=card_name,
                card_total_mib=card_total,
                ranks=list(rs),
                budget_mib=budget,
                physical_overcommit=overcommit,
                segments=segments,
                **agg,
            )
        )
    return ranks, cards


def _card_segments(model, flags, ranks, rs, kv_repl, graph_mem, sessions):
    """Ordered fine-grained segment list for one card (sum of the co-located
    ranks). Every segment carries its hover ``detail`` (the deeper numbers /
    formula) so the renderer never re-derives math. Sums to the card total.
    """

    def s(field):
        return sum(getattr(ranks[r], field) for r in rs)

    m = model
    tp = m.tp_size
    q_heads_txt = (
        f"{m.q_heads} Q heads x {m.head_dim} head_dim x {m.hidden} hidden "
        f"x {m.full_layers} attn layers (q_proj+o_proj q-sharded)"
    )
    kv_heads_txt = (
        f"{m.kv_heads} KV heads x {m.head_dim} head_dim x {m.hidden} hidden "
        f"x {m.full_layers} attn layers (k_proj+v_proj)"
    )
    repl = bool(kv_repl.get("replicated"))
    cache_dup = bool(kv_repl.get("cache_duplicated"))
    factor = int(kv_repl.get("factor") or 1)
    reason = kv_repl.get("reason") or ""

    def seg(id_, label, mib, detail, replicated=False):
        return {
            "id": id_,
            "label": label,
            "mib": mib,
            "detail": detail,
            "replicated": replicated,
            "replication_factor": factor if replicated else 1,
            "replication_reason": reason if replicated else None,
        }

    kv_cell = m.kv_cell_bytes
    draft_src = ranks[rs[0]].draft_w_source if rs else None
    per_session = ranks[rs[0]].mamba_per_session_mib if rs else 0.0
    gm = graph_mem or {}
    graph_items = gm.get("items") or []
    graph_detail = (
        f"CUDA graphs ({gm.get('provenance', 'heuristic')}"
        + (
            f", +-{gm.get('error_band_pct')}%"
            if gm.get("provenance") == "heuristic"
            else ""
        )
        + "): "
        + "; ".join(f"{it['label']} {it['mib']:.0f} MiB" for it in graph_items)
        + (". " + str(gm.get("formula") or "") if gm.get("formula") else "")
    )

    out = [
        seg("attn_q", "attn Q", s("attn_q_mib"), q_heads_txt),
        seg(
            "attn_kv",
            "attn KV",
            s("attn_kv_mib"),
            kv_heads_txt + (f"; REPLICATED x{factor}: {reason}" if repl else ""),
            replicated=repl,
        ),
        seg(
            "mlp",
            "MLP",
            s("mlp_dense_mib"),
            f"dense MLP (gate/up/down, intermediate {m.intermediate}) "
            f"x {m.n_layers} layers"
            + (
                f" + shared expert (intermediate " f"{m.shared_expert_intermediate})"
                if m.shared_expert_intermediate
                else ""
            ),
        ),
        seg(
            "experts",
            "experts",
            s("experts_mib"),
            f"routed MoE experts: per-expert 3 x {m.hidden} x "
            f"{m.moe_intermediate or m.intermediate} x layers",
        ),
        seg(
            "vocab",
            "vocab",
            s("vocab_mib"),
            f"embedding + lm_head, vocab {m.vocab} x hidden {m.hidden}",
        ),
        seg(
            "gdn_w",
            "GDN wts",
            s("gdn_mib"),
            f"linear-attention (GDN/SSM) weights: {m.gdn_k_heads}K/"
            f"{m.gdn_v_heads}V heads x {m.gdn_layers} layers",
        ),
        seg("vision", "vision", s("vision_mib"), "vision tower (replicated)"),
        seg(
            "draft_w",
            "draft wts",
            s("draft_w_mib"),
            (
                (
                    f"draft weights ({draft_src}): "
                    + (
                        f"{m.mtp_layers} MTP/nextn layer(s) appended after "
                        f"layer {m.n_layers}"
                        if draft_src == "mtp"
                        else "external --speculative-draft-model-path (estimate)"
                    )
                )
                if draft_src
                else "no draft weights"
            ),
        ),
        seg(
            "kv",
            "KV cache",
            s("kv_target_mib"),
            f"target KV: {kv_cell:.0f} B/token cell "
            f"({m.kv_cell_bytes_per_layer:.0f} B/layer x layers), "
            "tokens = predicted per-rank capacity"
            + (
                f"; CACHE DUPLICATED x{factor}: {reason}"
                if cache_dup
                else (
                    "; token-sharded across ranks (uneven DCP), not " "duplicated"
                    if repl
                    else ""
                )
            ),
            replicated=cache_dup,
        ),
        seg(
            "kv_draft",
            "draft KV",
            s("kv_draft_mib"),
            (
                f"draft/MTP KV + verify share: {m.mtp_layers} MTP layer(s) "
                f"of the {m.full_layers + m.mtp_layers}-layer KV cell; "
                "scales with speculative_num_draft_tokens "
                f"({flags.speculative_num_draft_tokens or 0})"
            ),
        ),
        seg(
            "mamba",
            "GDN state",
            s("mamba_mib"),
            f"GDN/mamba state pool: {per_session:.1f} MiB per session slot "
            f"x {sessions} slots (grows with concurrent sessions; "
            f"max_running_requests {flags.max_running_requests or 16} + "
            "overlap/spec pad)",
        ),
        seg("graphs", "CUDA graphs", s("graphs_mib"), graph_detail),
        seg(
            "ovh",
            "overhead",
            s("overhead_rest_mib"),
            "workspace / CUDA context / allocator rest of the fixed "
            "overhead reserve (total reserve "
            f"{_overhead_mib()} MiB per rank, minus the graph share)",
        ),
    ]
    return [x for x in out if x["mib"] > 0.005 or x["id"] in ("kv", "ovh")]


def _identity_map_or_none():
    """The live #397 IdentityMap, or ``None`` on a host with no NVML.

    A planner run is frequently a DESK run -- the wizard sizing a rig that is
    not this box -- so the absence of NVML must not make a plan unanswerable.
    What it costs is the card-exists check on ``gpu:`` targets; the grammar,
    the conflict rules and the feasibility arithmetic are all unaffected.
    ``allow_cuda_init`` stays False: a planner must not create a CUDA context
    (registry/rank_cards.py:43 on why that is a few hundred MiB).
    """
    try:
        from sglang.srt.registry.nvml import identity_map

        return identity_map()
    except Exception:  # noqa: BLE001 - no NVML is a legal desk state
        return None


def _compute_offload(
    model, flags, ffn_vector, rank_gpu, experts, mtp, num_experts, notes
):
    tp = model.tp_size
    offloadable = model.per_rank_offloadable_weight_bytes(ffn_vector)
    if num_experts <= 0 or sum(offloadable) <= 0:
        return None

    # Resident fraction: explicit flag > env > None (report pool only).
    # The env may be a comma-list, one entry per rank (#417 follow-up). A bare
    # float() on that raises, and the old except-branch turned the raise into
    # frac=None, which silently dropped the whole offload section from the
    # report -- an unbooked pool in the one place whose job is to book it.
    fracs: Optional[List[float]] = None
    if flags.moe_resident_expert_fraction is not None:
        fracs = [float(flags.moe_resident_expert_fraction)] * tp
    else:
        env = os.environ.get("SGLANG_MOE_RESIDENT_EXPERT_FRACTION")
        if env:
            try:
                parsed = [float(p) for p in env.split(",") if p.strip()]
            except ValueError:
                parsed = []
            if len(parsed) == 1:
                fracs = parsed * tp
            elif len(parsed) == tp:
                fracs = parsed
            elif parsed:
                raise ValueError(
                    f"SGLANG_MOE_RESIDENT_EXPERT_FRACTION has {len(parsed)} "
                    f"entries but this plan is for {tp} ranks; give one entry "
                    f"per rank or a single value."
                )
    if fracs is not None:
        fracs = [min(max(f, 0.0), 1.0) for f in fracs]
    # Kept for the report text below, which speaks about the plan as a whole.
    frac = fracs[0] if fracs else None

    H = model.hidden
    moe_i = model.moe_intermediate or model.intermediate
    bpp = model.families["mlp"].bytes_per_param
    per_expert_bytes = 3 * H * moe_i * bpp * model.n_layers

    # #396(b): operator constraints enter HERE -- inside the solve, after the
    # resident fraction has decided how many slots exist, and before any rule
    # is published. They can move WHICH experts fill those slots; they cannot
    # create slots, which is why apply_expert_constraints refuses rather than
    # widening the fraction behind the operator's back.
    #
    # The identity map is resolved ONLY when there is something to resolve.
    # Building it queries NVML, and a planner solve is run in tight loops by
    # the wizard and the explorer; paying an NVML sweep per solve on the
    # default path -- where no override exists to check a card name against --
    # would be a cost this feature has no right to impose on plans that do not
    # use it.
    override_specs = getattr(flags, "expert_placement_override", None)
    overrides = (
        parse_placement_overrides(override_specs, identity=_identity_map_or_none())
        if override_specs
        else ()
    )
    if overrides and fracs is None:
        raise PlacementOverrideConflict(
            "placement overrides were given but no resident expert fraction "
            "is pinned, so this plan has no residency split for them to "
            "constrain: every expert is GPU-resident and the overrides would "
            "silently do nothing. Supply moe_resident_expert_fraction (or "
            "SGLANG_MOE_RESIDENT_EXPERT_FRACTION) alongside them."
        )
    if overrides:
        notes.append(
            "placement overrides (#396) in force as CONSTRAINTS on the expert "
            "residency solve: " + describe_overrides(overrides)
        )

    per_rank = []
    total_host = 0.0
    for r in range(tp):
        off_mib = offloadable[r] / _MIB
        e = experts[r] if experts else None
        count = e.num_experts if e else 0
        rank_frac = fracs[r] if fracs else None
        resident_ids = host_ids = None
        if rank_frac is not None and e is not None:
            resident = math.ceil(rank_frac * count)
            host_count = count - resident
            host_start = e.expert_start + resident
            host_end = e.expert_end
            if overrides:
                constraints = resolve_expert_constraints(
                    overrides,
                    rank=r,
                    gpu_index=rank_gpu[r],
                    expert_start=e.expert_start,
                    expert_end=e.expert_end,
                    num_layers=model.n_layers,
                )
                if not constraints.is_empty:
                    resident_ids, host_ids = apply_expert_constraints(
                        constraints, e.expert_start, e.expert_end, resident
                    )
                    resident = len(resident_ids)
                    host_count = len(host_ids)
                    # The tail range is only meaningful while the host set IS
                    # a tail. Under overrides it is an explicit id list, and
                    # reporting a range that no longer describes it would be
                    # worse than reporting none.
                    contiguous = list(host_ids) == list(
                        range(e.expert_end - host_count, e.expert_end)
                    )
                    host_start = e.expert_end - host_count if contiguous else None
                    host_end = e.expert_end if contiguous else None
            host_mib = host_count * per_expert_bytes / _MIB
            resident_mib = resident * per_expert_bytes / _MIB
        else:
            resident = count
            host_count = 0
            host_start = host_end = None
            host_mib = 0.0
            resident_mib = count * per_expert_bytes / _MIB
        total_host += host_mib
        per_rank.append(
            OffloadRankRule(
                rank=r,
                gpu_index=rank_gpu[r],
                offloadable_mib=off_mib,
                host_expert_start=host_start,
                host_expert_end=host_end,
                host_expert_count=host_count,
                host_mib=host_mib,
                resident_expert_count=resident,
                resident_mib=resident_mib,
                resident_ids=resident_ids,
                host_ids=host_ids,
            )
        )

    if frac is None:
        note = (
            "RULE: the routed-expert stack (#77) is the only host-offloadable "
            "weight class. Adaptive keeps a hot resident subset on the GPU and "
            "spills the rest to a pinned host pool. No resident fraction was "
            "pinned, so only the offloadable pool per rank is reported; supply "
            "moe_resident_expert_fraction (or SGLANG_MOE_RESIDENT_EXPERT_FRACTION) "
            "to enumerate the host-held expert indices."
        )
    else:
        note = (
            f"RULE: with resident fraction {frac:.2f}, each rank keeps the first "
            "ceil(fraction * count) experts of its range GPU-resident and holds "
            "the numbered tail in host RAM (per-rank host_expert_start.."
            "host_expert_end). MTP / nextn graphs are additionally host-parkable "
            "when the draft is not the bottleneck."
        )

    return OffloadRule(
        offloadable_class="moe_routed_experts",
        resident_fraction=frac,
        per_rank=per_rank,
        total_offloadable_mib=sum(offloadable) / _MIB,
        total_host_mib=total_host,
        mtp_offloadable_mib=list(mtp.per_rank_mib) if mtp.present else [0.0] * tp,
        note=note,
    )


def _overhead_mib() -> int:
    from sglang.srt import uneven_perf

    return int(
        uneven_perf._PREDICT_OVERHEAD_MIB + uneven_perf._PREDICT_MAMBA_ACT_RESERVE_MIB
    )
