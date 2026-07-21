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
"""Authoritative flag/env catalog for the config-planner dashboard (design
"Module: flags.py").

This module is the single source of truth for EVERY sglang ``ServerArgs`` flag
plus EVERY fork-specific flag/env var, together with the mutual-exclusion,
dependency, model-compatibility and value-constraint logic the Runner tab uses
to grey fields, drive hover text and keep dropdowns self-consistent, and a
profile generator (single-gpu / normal-TP / co-location / uneven-split) whose
outputs are user-saveable to a small JSON store.

Design of the catalog
----------------------
The upstream flag set is discovered LIVE from the deployed ``ServerArgs``
dataclass (``typing.get_type_hints(..., include_extras=True)`` gives the
``Annotated[T, Arg(...)]`` help/choices), so the catalog can never drift from
the build that is actually running -- the same feature-detection discipline
``webui.discover_knobs`` already uses. On top of that auto-discovered base we
overlay a hand-curated layer (``_CURATED``) that adds the real hover text,
verified ``mutually_exclusive_with`` / ``requires`` edges, ``model_compat``
predicates and enumerable ``allowed`` values for the flags that carry real
planner logic (all fork flags + the upstream flags they interact with). Fork
env vars that are NOT ``ServerArgs`` fields (``SGLANG_UNEVEN_*``,
``SGLANG_MOE_RESIDENT_EXPERT_FRACTION`` ...) are added explicitly.

CRITICAL fork-capability note
-----------------------------
The fork CAN do uneven Tensor Parallelism with ``tp_size`` GREATER than the
model's KV-head count (KV heads are replicated and the KV cache is token-
sharded on the DCP axis) and with ``tp_size`` GREATER than the physical card
count (several ranks co-locate on one card via CUDA MPS). Neither of those is
marked model-incompatible here -- stock sglang cannot do them, this fork can.
``model_compat`` only flags GENUINELY impossible combinations (e.g. a
MoE-expert split on a dense model), each with a clear hover reason.
"""

from __future__ import annotations

import dataclasses
import json
import os
import typing
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "FlagSpec",
    "Profile",
    "catalog",
    "upstream_count",
    "fork_count",
    "resolve",
    "cross_field_errors",
    "profiles",
    "profile_argv",
    "profile_env",
    "validate_profile",
    "ProfileStore",
    "DEFAULT_STORE_PATH",
    "model_is_moe",
    "model_kv_heads",
    "model_is_hybrid",
    "model_has_mtp",
    "model_quant_kind",
    "rig_has_sm86",
]


# ===========================================================================
# model_cfg accessors (flat OR nested-under-"text_config", the shape the cost
# model / _gguf_config_and_families already build).
# ===========================================================================


def _cfg_get(model_cfg: Optional[dict], key: str, default=None):
    """Read ``key`` from a model config that may be flat or nested under a
    ``text_config`` block (both shapes are produced elsewhere in the planner)."""
    if not model_cfg:
        return default
    if key in model_cfg and model_cfg[key] is not None:
        return model_cfg[key]
    tc = model_cfg.get("text_config")
    if isinstance(tc, dict) and tc.get(key) is not None:
        return tc[key]
    return default


def model_is_moe(model_cfg: Optional[dict]) -> bool:
    """True when the model has routed experts (a MoE model)."""
    n = _cfg_get(model_cfg, "num_experts", 0) or _cfg_get(
        model_cfg, "n_routed_experts", 0
    )
    try:
        return int(n) > 0
    except (TypeError, ValueError):
        return False


def model_kv_heads(model_cfg: Optional[dict]) -> Optional[int]:
    """KV-head count (``num_key_value_heads``), or None when unknown."""
    v = _cfg_get(model_cfg, "num_key_value_heads")
    if v is None:
        v = _cfg_get(model_cfg, "num_attention_heads")
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def model_is_hybrid(model_cfg: Optional[dict]) -> bool:
    """True for a hybrid attention/linear-attention (GDN/mamba) model, e.g.
    Qwen3.5/3.6: ``layer_types`` contains 'linear_attention', or the config
    carries ``linear_num_value_heads`` / ``mamba_*`` keys. Hybrid models need
    ``--hicache-mem-layout page_first_direct`` under the hierarchical cache
    (MambaPoolHost rejects every other layout) and honor the
    ``SGLANG_MAMBA_SSM_DTYPE`` env var."""
    lt = _cfg_get(model_cfg, "layer_types")
    if isinstance(lt, (list, tuple)) and "linear_attention" in lt:
        return True
    if _cfg_get(model_cfg, "linear_num_value_heads") is not None:
        return True
    if _cfg_get(model_cfg, "mamba_ssm_dtype") is not None or _cfg_get(
        model_cfg, "mamba_d_state"
    ):
        return True
    return False


def model_has_mtp(model_cfg: Optional[dict]) -> bool:
    """True when the checkpoint ships MTP draft layers (NEXTN speculative
    decoding is possible): ``mtp_num_hidden_layers`` or
    ``num_nextn_predict_layers`` > 0."""
    v = _cfg_get(model_cfg, "mtp_num_hidden_layers")
    if v is None:
        v = _cfg_get(model_cfg, "num_nextn_predict_layers")
    try:
        return v is not None and int(v) > 0
    except (TypeError, ValueError):
        return False


def model_quant_kind(model_cfg: Optional[dict]) -> Optional[str]:
    """Coarse checkpoint-quant classification for the calibration table:
    'fp8' (self-declaring fp8 quant_method), 'awq' (compressed-tensors / awq
    quant_method — on this rig the compressed-tensors checkpoint IS the AWQ
    one), or None when the config does not self-declare."""
    qc = _cfg_get(model_cfg, "quantization_config")
    if not isinstance(qc, dict):
        return None
    method = str(qc.get("quant_method") or "").lower()
    if method == "fp8":
        return "fp8"
    if method in ("compressed-tensors", "awq"):
        return "awq"
    return None


#: GPU-name fragments that identify sm86 (Ampere GA10x) cards. fp8_e4m3 KV is
#: NOT supported on sm86 — any hit forces kv-cache-dtype fp8_e5m2.
_SM86_NAME_FRAGMENTS = (
    "rtx 30",
    "rtx30",
    "a10",
    "a16",
    "a40",
    "a2 ",
    "rtx a2000",
    "rtx a4000",
    "rtx a4500",
    "rtx a5000",
    "rtx a5500",
    "rtx a6000",
)


def rig_has_sm86(gpus: Sequence) -> bool:
    """True when any GPU in the rig is (or looks like) an sm86 card. Uses an
    explicit ``sm`` / ``compute_capability`` field when present, else a
    conservative name match (RTX 30xx / Ampere workstation cards)."""
    for g in gpus or ():
        if isinstance(g, dict):
            sm = g.get("sm") or g.get("compute_capability")
            name = str(g.get("name") or "")
        else:
            sm = getattr(g, "sm", None) or getattr(g, "compute_capability", None)
            name = str(getattr(g, "name", "") or "")
        if sm is not None:
            try:
                if int(float(str(sm).replace("sm_", "").replace("8.6", "86"))) == 86:
                    return True
                continue
            except (TypeError, ValueError):
                pass
        low = name.lower()
        if any(f in low for f in _SM86_NAME_FRAGMENTS):
            return True
    return False


# ===========================================================================
# FlagSpec: one catalog entry.
# ===========================================================================


@dataclasses.dataclass(frozen=True)
class FlagSpec:
    """One flag / env var in the catalog.

    Fields:
      * ``id``          -- stable catalog key (the ServerArgs field name, or the
        env-var name for env-only flags).
      * ``name``        -- the user-facing CLI flag (``--tp-size``) or env var.
      * ``type``        -- one of bool / int / float / enum / list / str.
      * ``default``     -- the inactive/default value.
      * ``allowed``     -- enumerable values for a dropdown, else None.
      * ``group``       -- UI section.
      * ``help``        -- one-line help.
      * ``hover``       -- longer hover explanation.
      * ``mutually_exclusive_with`` -- ids that cannot be active at the same
        time (treated symmetrically by :func:`resolve`).
      * ``requires``    -- ids that MUST be active for this to be legal
        (auto-set by :func:`resolve` when this flag is active).
      * ``requires_any`` -- list of alternative-groups; each group is satisfied
        when ANY of its ids is active, and :func:`resolve` auto-sets the FIRST
        member of an unsatisfied group.
      * ``source``      -- "upstream" | "fork" | "env".
      * ``is_env``      -- True for environment variables (not CLI flags).
      * ``tuple_len_flag`` -- when set, a list value's length must equal the
        current value of this other flag (e.g. ``tp_size``).
      * ``_compat``     -- optional ``model_cfg -> (ok, reason)`` predicate.
    """

    id: str
    name: str
    type: str
    default: Any = None
    allowed: Optional[Tuple[Any, ...]] = None
    group: str = "other"
    help: str = ""
    hover: str = ""
    mutually_exclusive_with: Tuple[str, ...] = ()
    requires: Tuple[str, ...] = ()
    requires_any: Tuple[Tuple[str, ...], ...] = ()
    source: str = "upstream"
    is_env: bool = False
    tuple_len_flag: Optional[str] = None
    _compat: Optional[Callable[[Optional[dict]], Tuple[bool, str]]] = None

    def model_compat(self, model_cfg: Optional[dict]) -> Tuple[bool, str]:
        """Return ``(ok, reason)`` for this flag against a model config. Default
        is always-ok; curated flags supply a real predicate."""
        if self._compat is None:
            return True, ""
        try:
            return self._compat(model_cfg)
        except Exception as e:  # pragma: no cover - defensive
            return True, f"(compat check skipped: {e})"


# ===========================================================================
# Curated overlay: real hover / exclusion / dependency / compat logic.
#
# Every edge below was verified against server_args.py's own validation
# (ServerArgs.__post_init__ / _resolve_rank_gpu_mapping): the fork rejects
# --rank-gpu-id together with pp/dp/ep > 1, nnodes > 1, an explicit
# --mem-fraction-static, and a non-default --base-gpu-id/--gpu-id-step; it
# requires --rank-gpu-memory-mib (or --rank-tp-ratio auto) whenever
# --rank-gpu-id is set; --rank-{tp,mlp,vocab,moe,kv}-ratio all require an
# active plan as noted in their help; the weightless-KV lane requires its
# fast-lane switch and hard-rejects speculative decoding + mixed-chunk.
# ===========================================================================


def _dense_only(reason_flag: str):
    def _c(cfg):
        if model_is_moe(cfg):
            return True, ""
        return (
            False,
            f"{reason_flag} only affects routed experts; this model is dense "
            "(num_experts == 0), so the flag is a no-op / not applicable.",
        )

    return _c


# id -> dict of curated overrides merged onto the auto-discovered base.
_CURATED: Dict[str, dict] = {
    # ---- parallelism (upstream, but the fork's exclusion partners) --------
    "tp_size": dict(
        group="parallelism",
        hover="Tensor-parallel rank count. The fork additionally allows "
        "tp_size > model KV-head count (KV replicated + token-sharded on the "
        "DCP axis) and tp_size > physical card count (ranks co-locate on one "
        "card via CUDA MPS) -- neither is rejected here.",
    ),
    "pp_size": dict(
        group="parallelism",
        mutually_exclusive_with=("rank_gpu_id",),
        hover="Pipeline-parallel size. The fork's heterogeneous rank->GPU "
        "mapping is pure single-node TENSOR parallelism only; pp_size > 1 is "
        "hard-rejected together with --rank-gpu-id.",
    ),
    "dp_size": dict(
        group="parallelism",
        mutually_exclusive_with=("rank_gpu_id",),
        hover="Data-parallel size. Rejected together with --rank-gpu-id "
        "(pure-TP-only mode).",
    ),
    "ep_size": dict(
        group="parallelism",
        mutually_exclusive_with=("rank_gpu_id",),
        hover="Expert-parallel size. Rejected together with --rank-gpu-id "
        "(pure-TP-only mode); use --rank-moe-ratio to split experts unevenly "
        "across TP ranks instead.",
    ),
    "nnodes": dict(
        group="parallelism",
        mutually_exclusive_with=("rank_gpu_id",),
        hover="Node count. The heterogeneous mapping is single-node only.",
    ),
    "dcp_size": dict(
        group="parallelism",
        hover="Decode context-parallel size. Under the fork's uneven-DCP the "
        "KV cache is token-sharded across this many ranks; it is auto-set to "
        "tp_size when a weighted token vector is active.",
    ),
    "base_gpu_id": dict(
        group="device",
        mutually_exclusive_with=("rank_gpu_id",),
        hover="First GPU index used. Replaced by the explicit --rank-gpu-id "
        "map; a non-zero value together with --rank-gpu-id is rejected.",
    ),
    "gpu_id_step": dict(
        group="device",
        mutually_exclusive_with=("rank_gpu_id",),
        hover="Stride between rank GPU indices. Replaced by --rank-gpu-id; a "
        "non-default value together with --rank-gpu-id is rejected.",
    ),
    "mem_fraction_static": dict(
        group="memory",
        mutually_exclusive_with=("rank_gpu_id", "rank_gpu_memory_mib"),
        hover="Global static-memory fraction (0..1). In the heterogeneous "
        "mode --rank-gpu-memory-mib supplies each rank's ABSOLUTE budget and "
        "an explicit --mem-fraction-static is DISABLED (hard-rejected, not "
        "merely ignored): the same MiB maps to a different fraction on each "
        "physical card. (This is sglang's analogue of vLLM's "
        "--gpu-memory-utilization.)",
    ),
    # ---- fork: heterogeneous rank -> GPU mapping --------------------------
    "rank_gpu_id": dict(
        source="fork",
        group="uneven-tp",
        type="list",
        tuple_len_flag="tp_size",
        requires_any=(("rank_gpu_memory_mib", "rank_tp_ratio"),),
        mutually_exclusive_with=(
            "mem_fraction_static",
            "pp_size",
            "dp_size",
            "ep_size",
            "nnodes",
            "base_gpu_id",
            "gpu_id_step",
        ),
        hover="One physical GPU index per rank (length == tp_size). Duplicates "
        "co-locate several ranks on one physical card (MPS-isolated). Requires "
        "--rank-gpu-memory-mib, OR --rank-tp-ratio auto to derive the budgets "
        "from NVML totals. Enables the heterogeneous-TP path.",
    ),
    "rank_gpu_memory_mib": dict(
        source="fork",
        group="uneven-tp",
        type="list",
        requires=("rank_gpu_id",),
        mutually_exclusive_with=("mem_fraction_static",),
        hover="Absolute per-rank VRAM budget in MiB. A scalar applies to every "
        "rank (even TP: shards are structurally equal); a per-rank list needs "
        "--rank-tp-ratio (or the weightless-KV lane) since shards then differ. "
        "This is the rank's ENTIRE budget -- no extra utilization ceiling or "
        "safety margin is layered on; the MiB->fraction conversion is per "
        "physical card. Requires --rank-gpu-id.",
    ),
    "rank_tp_ratio": dict(
        source="fork",
        group="uneven-tp",
        type="list",
        tuple_len_flag="tp_size",
        allowed=("auto", "auto-performance"),
        requires=("rank_gpu_id",),
        hover="Uneven-TP weight vector (one positive int per rank, sum must "
        "divide every sharded dim), or 'auto' (weights from the VRAM budgets) "
        "or 'auto-performance' (VRAM split + a measured MLP-family vector for "
        "throughput). Attention splits in whole KV-head units. Requires "
        "--rank-gpu-id. NOTE: this is enumerable ('auto'/'auto-performance') "
        "OR a free integer list -- the dropdown offers the two auto modes and "
        "a 'custom vector' entry.",
    ),
    "rank_auto_reserve_mib": dict(
        source="fork",
        group="uneven-tp",
        type="str",
        default="auto",
        requires=("rank_tp_ratio",),
        hover="Headroom (MiB) subtracted from NVML totals when --rank-tp-ratio "
        "auto derives budgets. 'auto' (default) derives the runtime reserve + "
        "CUDA-graph capture demand. Only valid with --rank-tp-ratio auto.",
    ),
    "rank_perf_loose_ctx_percent": dict(
        source="fork",
        group="uneven-tp",
        type="float",
        default=0.0,
        requires=("rank_tp_ratio",),
        hover="Context floor for --rank-tp-ratio auto-performance: only accept "
        "splits whose predicted max context stays >= (100 - X)%% of the "
        "VRAM-auto split. X=0 (default) takes only free gains. Only valid "
        "with auto-performance.",
    ),
    "rank_perf_tune": dict(
        source="fork",
        group="uneven-tp",
        type="enum",
        default="both",
        allowed=("both", "dec", "enc"),
        requires=("rank_tp_ratio",),
        hover="auto-performance tuning target: 'enc' maximizes prefill, 'both' "
        "(default) prefill+concurrent throughput, 'dec' a near-no-op (decode "
        "is flat across splits). Only valid with auto-performance.",
    ),
    "rank_mlp_ratio": dict(
        source="fork",
        group="uneven-tp",
        type="list",
        tuple_len_flag="tp_size",
        requires=("rank_tp_ratio",),
        hover="Per-rank integer weights for the dense-MLP (FFN) family ONLY, "
        "on the MLP unit grid. Shifts FFN mass between ranks to grow the "
        "min-synced KV pool; attention/KV keep following --rank-tp-ratio. Env "
        "SGLANG_UNEVEN_MLP_VECTOR overrides it. Requires an active "
        "--rank-tp-ratio plan.",
    ),
    "rank_moe_ratio": dict(
        source="fork",
        group="uneven-tp",
        type="list",
        tuple_len_flag="tp_size",
        requires=("rank_tp_ratio",),
        _compat=_dense_only("--rank-moe-ratio"),
        hover="Per-rank integer weights for the fused expert (MoE) family "
        "ONLY (whole experts per rank). The main lever on MoE models; "
        "attention/KV keep following --rank-tp-ratio. Env "
        "SGLANG_UNEVEN_MOE_VECTOR overrides. Requires an active "
        "--rank-tp-ratio plan and a MoE model.",
    ),
    "rank_vocab_ratio": dict(
        source="fork",
        group="uneven-tp",
        type="list",
        tuple_len_flag="tp_size",
        allowed=("auto",),
        requires=("rank_tp_ratio",),
        hover="Ratio-weighted vocab/lm_head sharding (64-row padded units): "
        "'auto' (from the bandwidth profile) or a per-rank integer vector, to "
        "balance the lm_head matvec READ TIME on heterogeneous cards. Env "
        "SGLANG_UNEVEN_VOCAB_VECTOR overrides. Requires an active "
        "--rank-tp-ratio plan; default OFF (even).",
    ),
    "rank_kv_ratio": dict(
        source="fork",
        group="uneven-tp",
        type="list",
        default="coupled",
        allowed=("coupled", "capacity", "auto"),
        requires=("rank_gpu_id", "rank_tp_ratio"),
        hover="Uneven-DCP KV-TOKEN ownership vector, DECOUPLED from the weight "
        "split. 'coupled' (default) = previous behavior; 'capacity'/'auto' = "
        "measured capacity-weighted ownership (maximizes max_total_num_tokens, "
        "shifts deep-context attention work to bigger cards); or an explicit "
        "per-rank integer vector. Requires --rank-gpu-id with a NON-uniform "
        "--rank-tp-ratio plan. Env SGLANG_UNEVEN_TOKEN_VECTOR overrides.",
    ),
    # ---- fork: weightless-KV fast lane (Variant C) ------------------------
    "weightless_kv_fastlane": dict(
        source="fork",
        group="weightless-kv",
        type="bool",
        default=False,
        mutually_exclusive_with=("speculative_algorithm", "enable_mixed_chunk"),
        hover="EXPERIMENTAL: one 'head rank' holds the full weights and runs "
        "collective-free TP=1; the other DCP ranks are weightless (KV "
        "token-shard + attention only), so a large KV lives on smaller cards "
        "while single-stream speed stays near the head card's rate. Requires "
        "tp_size == dcp_size and a flashinfer attention backend. Speculative "
        "decoding and --enable-mixed-chunk are hard-rejected.",
    ),
    "weightless_kv_head_rank": dict(
        source="fork",
        group="weightless-kv",
        type="int",
        default=0,
        requires=("weightless_kv_fastlane",),
        hover="The rank that holds all weights under the weightless-KV lane "
        "(default 0). Every other rank is weightless.",
    ),
    "weightless_kv_chunked_block_size": dict(
        source="fork",
        group="weightless-kv",
        type="int",
        default=0,
        requires=("weightless_kv_fastlane",),
        hover="Block-decode staging size for the weightless-KV lane. 0 (OFF) = "
        "single monolithic flashinfer call; >0 iterates the owned KV shard in "
        "blocks with online-LSE merge. Requires --weightless-kv-fastlane.",
    ),
    "weightless_kv_host_spill_tokens": dict(
        source="fork",
        group="weightless-kv",
        type="int",
        default=0,
        requires=("weightless_kv_fastlane", "weightless_kv_chunked_block_size"),
        hover="Per-rank pinned-host KV overflow slots for the weightless-KV "
        "lane. 0 (OFF) = fully device-resident; >0 appends a host tier so a "
        "single sequence's KV can exceed VRAM (PCIe-streamed, eager-only). "
        "Requires the fast lane AND a non-zero chunked block size.",
    ),
    "weightless_kv_spill_device_cap": dict(
        source="fork",
        group="weightless-kv",
        type="int",
        default=0,
        requires=("weightless_kv_host_spill_tokens",),
        hover="DEBUG cap on allocatable device-resident KV slots per rank "
        "(0 = no cap); forces the host-streaming path at small contexts for "
        "byte-parity testing. Pairs with --weightless-kv-host-spill-tokens.",
    ),
    # ---- fork: hibernate (#89) --------------------------------------------
    "hibernate_dir": dict(
        source="fork",
        group="memory",
        type="str",
        hover="Directory for hibernate (suspend-to-disk) of per-rank weights, "
        "surviving process exit (fork #89). Speeds a cold reload; None = OFF.",
    ),
    # ---- fork: adaptive speculative decoding ------------------------------
    "speculative_adaptive": dict(
        source="fork",
        group="speculative",
        type="bool",
        default=False,
        requires=("speculative_algorithm",),
        hover="Adaptive speculative decoding: dynamically adjusts the draft "
        "length (k) from the running acceptance rate. Requires a speculative "
        "algorithm to be selected.",
    ),
    # ---- upstream flags the profiles/UI care about ------------------------
    "speculative_algorithm": dict(
        group="speculative",
        type="enum",
        allowed=("EAGLE", "EAGLE3", "NEXTN", "STANDALONE", "NGRAM"),
        hover="Speculative-decoding algorithm (EAGLE/EAGLE3/NEXTN/...). "
        "Mutually exclusive with the weightless-KV fast lane.",
    ),
    "enable_mixed_chunk": dict(
        group="scheduling",
        hover="Mix chunked-prefill and decode in one batch. Hard-rejected by "
        "the weightless-KV fast lane.",
    ),
    "attention_backend": dict(
        group="attention",
        hover="Attention kernel backend. The weightless-KV lane and several "
        "fork paths require a flashinfer backend (fa3/flashinfer).",
    ),
    "kv_cache_dtype": dict(group="memory"),
    "quantization": dict(group="model"),
    "enable_memory_saver": dict(
        group="memory",
        hover="Allow release/resume of GPU memory occupation (suspend-to-RAM). "
        "Pairs with the fork's SGLANG_MEMORY_SAVER_CUDA_GRAPH env var.",
    ),
}


# ---- env-only fork flags (not ServerArgs fields) --------------------------
_ENV_FLAGS: List[FlagSpec] = [
    FlagSpec(
        id="SGLANG_UNEVEN_TOKEN_VECTOR",
        name="SGLANG_UNEVEN_TOKEN_VECTOR",
        type="list",
        source="env",
        is_env=True,
        group="uneven-tp",
        requires=("rank_tp_ratio",),
        help="Explicit KV-token ownership vector (overrides --rank-kv-ratio).",
        hover="Comma-separated integer token-vector on the "
        "page_size x sum(ratios) virtual owner blocks. Takes precedence over "
        "--rank-kv-ratio. Requires an active uneven-TP plan.",
    ),
    FlagSpec(
        id="SGLANG_UNEVEN_MLP_VECTOR",
        name="SGLANG_UNEVEN_MLP_VECTOR",
        type="list",
        source="env",
        is_env=True,
        group="uneven-tp",
        requires=("rank_tp_ratio",),
        help="Explicit dense-MLP split vector (overrides --rank-mlp-ratio).",
        hover="Per-rank integer MLP-family weights; takes precedence over "
        "--rank-mlp-ratio. Requires an active uneven-TP plan.",
    ),
    FlagSpec(
        id="SGLANG_UNEVEN_MOE_VECTOR",
        name="SGLANG_UNEVEN_MOE_VECTOR",
        type="list",
        source="env",
        is_env=True,
        group="uneven-tp",
        requires=("rank_tp_ratio",),
        _compat=_dense_only("SGLANG_UNEVEN_MOE_VECTOR"),
        help="Explicit MoE-expert split vector (overrides --rank-moe-ratio).",
        hover="Per-rank integer expert-family weights; takes precedence over "
        "--rank-moe-ratio. Requires an active uneven-TP plan and a MoE model.",
    ),
    FlagSpec(
        id="SGLANG_UNEVEN_VOCAB_VECTOR",
        name="SGLANG_UNEVEN_VOCAB_VECTOR",
        type="list",
        source="env",
        is_env=True,
        group="uneven-tp",
        requires=("rank_tp_ratio",),
        help="Explicit vocab split vector (overrides --rank-vocab-ratio).",
        hover="Per-rank integer vocab-family weights; takes precedence over "
        "--rank-vocab-ratio. Requires an active uneven-TP plan.",
    ),
    FlagSpec(
        id="SGLANG_UNEVEN_DCP",
        name="SGLANG_UNEVEN_DCP",
        type="bool",
        default=False,
        source="env",
        is_env=True,
        group="uneven-tp",
        help="Legacy gate for the uneven-DCP estimate token vector.",
        hover="Legacy env gate enabling the uneven-DCP token-axis KV sharding "
        "with an estimate vector. Superseded by --rank-kv-ratio (which does "
        "not need this env). Kept for backward compatibility.",
    ),
    FlagSpec(
        id="SGLANG_UNEVEN_DCP_WEIGHTED",
        name="SGLANG_UNEVEN_DCP_WEIGHTED",
        type="bool",
        default=False,
        source="env",
        is_env=True,
        group="uneven-tp",
        requires=("SGLANG_UNEVEN_DCP",),
        help="Legacy gate for the WEIGHTED uneven-DCP vector.",
        hover="Legacy env gate for the weighted (vs even-modulo) uneven-DCP "
        "token vector. Superseded by --rank-kv-ratio.",
    ),
    FlagSpec(
        id="SGLANG_MAMBA_SSM_DTYPE",
        name="SGLANG_MAMBA_SSM_DTYPE",
        type="enum",
        default=None,
        allowed=("bfloat16", "float32"),
        source="env",
        is_env=True,
        group="memory",
        help="Dtype of the mamba/GDN SSM state on hybrid models.",
        hover="Overrides the hybrid (GDN/mamba) SSM-state dtype. 'bfloat16' "
        "halves the per-request mamba state vs float32 and is the validated "
        "setting for Qwen3.5/3.6-class hybrids on this fork. Only meaningful "
        "on a hybrid/linear-attention model.",
        _compat=lambda cfg: (
            (True, "")
            if cfg is None or model_is_hybrid(cfg)
            else (
                False,
                "SGLANG_MAMBA_SSM_DTYPE only affects hybrid GDN/mamba models; "
                "this model has no linear-attention layers.",
            )
        ),
    ),
    FlagSpec(
        id="SGLANG_MOE_RESIDENT_EXPERT_FRACTION",
        name="SGLANG_MOE_RESIDENT_EXPERT_FRACTION",
        type="float",
        default=1.0,
        source="env",
        is_env=True,
        group="offload",
        _compat=_dense_only("SGLANG_MOE_RESIDENT_EXPERT_FRACTION"),
        help="Fraction of MoE experts kept resident in VRAM (rest offloaded).",
        hover="0.0..1.0 fraction of routed experts kept resident on the GPU; "
        "the remainder is offloaded to host RAM and streamed on demand "
        "(adaptive expert offload). Only meaningful on a MoE model.",
    ),
    FlagSpec(
        id="SGLANG_MEMORY_SAVER_CUDA_GRAPH",
        name="SGLANG_MEMORY_SAVER_CUDA_GRAPH",
        type="bool",
        default=False,
        source="env",
        is_env=True,
        group="memory",
        requires=("enable_memory_saver",),
        help="Include CUDA-graph memory in the memory-saver release/resume.",
        hover="Extends --enable-memory-saver to also release/resume the "
        "captured CUDA-graph pool. Requires --enable-memory-saver.",
    ),
]

#: ids that are fork additions even though they are ServerArgs fields (used for
#: the upstream-vs-fork count). Env flags are counted as fork separately.
_FORK_FIELD_IDS = {
    "rank_gpu_id",
    "rank_gpu_memory_mib",
    "rank_tp_ratio",
    "rank_auto_reserve_mib",
    "rank_perf_loose_ctx_percent",
    "rank_perf_tune",
    "rank_mlp_ratio",
    "rank_moe_ratio",
    "rank_vocab_ratio",
    "rank_kv_ratio",
    "weightless_kv_fastlane",
    "weightless_kv_head_rank",
    "weightless_kv_chunked_block_size",
    "weightless_kv_host_spill_tokens",
    "weightless_kv_spill_device_cap",
    "hibernate_dir",
    "speculative_adaptive",
    "speculative_adaptive_config",
    "speculative_adaptive_graph_memory",
}


# ===========================================================================
# Catalog assembly: auto-discover ServerArgs + overlay curated + env flags.
# ===========================================================================


def _infer_type(base_type, default, allowed) -> str:
    if allowed:
        return "enum"
    s = str(base_type)
    if base_type is bool or default is True or default is False:
        return "bool"
    if "List" in s or "list" in s or isinstance(default, list):
        return "list"
    if base_type is int or (default is not None and isinstance(default, int)
                            and not isinstance(default, bool)):
        return "int"
    if base_type is float or isinstance(default, float):
        return "float"
    # Optional[Union[int, List[int]]] etc.
    if "int" in s and "List" in s:
        return "list"
    if "int" in s:
        return "int"
    if "float" in s:
        return "float"
    return "str"


def _discover_upstream() -> Dict[str, dict]:
    """Discover every ``ServerArgs`` field as a base catalog dict (id -> attrs).
    Returns an empty dict if ServerArgs cannot be imported (keeps the module
    importable in a bare-CPU test env, though it is available here)."""
    try:
        from sglang.srt.arg_groups.arg_utils import Arg
        from sglang.srt.server_args import ServerArgs
    except Exception:  # pragma: no cover - defensive
        return {}

    hints = typing.get_type_hints(ServerArgs, include_extras=True)
    defaults = {f.name: f for f in dataclasses.fields(ServerArgs)}
    base: Dict[str, dict] = {}
    for name, f in defaults.items():
        hint = hints.get(name)
        base_type: Any = str
        help_ = ""
        choices = None
        cli_name = "--" + name.replace("_", "-")
        if hint is not None:
            args = typing.get_args(hint)
            if args:
                base_type = args[0]
                meta = args[1] if len(args) > 1 else None
                if isinstance(meta, str):
                    help_ = meta
                elif isinstance(meta, Arg):
                    if getattr(meta, "no_cli", False):
                        # internal-only field, not a user flag; skip.
                        continue
                    help_ = getattr(meta, "help", "") or ""
                    choices = getattr(meta, "choices", None)
                    aliases = getattr(meta, "aliases", None)
                    if aliases:
                        cli_name = aliases[0]
        if f.default is dataclasses.MISSING:
            default = None
        else:
            default = f.default
        allowed = tuple(choices) if choices else None
        base[name] = dict(
            id=name,
            name=cli_name,
            type=_infer_type(base_type, default, allowed),
            default=default,
            allowed=allowed,
            group="other",
            help=(help_.split("\n")[0][:200] if help_ else ""),
            hover=help_,
            source="fork" if name in _FORK_FIELD_IDS else "upstream",
        )
    return base


_CATALOG_CACHE: Optional[Dict[str, FlagSpec]] = None


def catalog(refresh: bool = False) -> Dict[str, FlagSpec]:
    """Return the full ``{id: FlagSpec}`` catalog (built once, cached).

    Upstream flags are auto-discovered from the deployed ``ServerArgs`` so the
    catalog matches the running build; the curated overlay + env flags add the
    hover/exclusion/dependency/compat logic.
    """
    global _CATALOG_CACHE
    if _CATALOG_CACHE is not None and not refresh:
        return _CATALOG_CACHE

    base = _discover_upstream()
    # Overlay curated attrs.
    for cid, over in _CURATED.items():
        if cid in base:
            base[cid].update(over)
        else:
            # A curated flag whose ServerArgs field is absent in this build:
            # include it anyway so the catalog is complete for the UI.
            entry = dict(
                id=cid,
                name="--" + cid.replace("_", "-"),
                type="str",
                default=None,
                allowed=None,
                group="other",
                help="",
                hover="",
                source="fork" if cid in _FORK_FIELD_IDS else "upstream",
            )
            entry.update(over)
            base[cid] = entry

    specs: Dict[str, FlagSpec] = {}
    valid = {f.name for f in dataclasses.fields(FlagSpec)}
    for cid, attrs in base.items():
        kwargs = {k: v for k, v in attrs.items() if k in valid}
        # tuple-ify sequence fields for the frozen dataclass.
        for tup in ("allowed", "mutually_exclusive_with", "requires"):
            if isinstance(kwargs.get(tup), list):
                kwargs[tup] = tuple(kwargs[tup])
        if isinstance(kwargs.get("requires_any"), list):
            kwargs["requires_any"] = tuple(
                tuple(g) for g in kwargs["requires_any"]
            )
        specs[cid] = FlagSpec(**kwargs)

    for spec in _ENV_FLAGS:
        specs[spec.id] = spec

    _CATALOG_CACHE = specs
    return specs


def upstream_count() -> int:
    return sum(1 for s in catalog().values() if s.source == "upstream")


def fork_count() -> int:
    return sum(1 for s in catalog().values() if s.source in ("fork", "env"))


# ===========================================================================
# Cross-field constraints: the richer predicates a flat requires/exclusion
# graph cannot express (numeric equality, enum-value membership, vector
# NON-uniformity). Machine-enforced through resolve()/validate_profile(),
# not just hover text. Every rule below was verified against
# server_args.py __post_init__ (_handle_weightless_kv_fastlane,
# _handle_dcp_validation, _handle_uneven_tp) or MambaPoolHost.
# ===========================================================================

#: attention backends the weightless-KV lane accepts (server_args:
#: ``attention_backend not in (None, "flashinfer", "fa3")`` is rejected).
_FLASHINFER_BACKENDS = (None, "", "flashinfer", "fa3")


def _int_or(v, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _ratio_vector(v) -> Optional[List[int]]:
    """An explicit integer vector from a ratio value, or None for unset /
    'auto' / 'auto-performance' / other sentinels."""
    if v is None or v == "":
        return None
    if isinstance(v, str) and not v.replace(" ", "").replace(",", "").isdigit():
        return None  # sentinel ('auto', 'auto-performance', 'coupled', ...)
    return _as_int_list(v)


def _uniform_explicit_ratio(v) -> bool:
    """True when ``v`` is an EXPLICIT vector with all-identical entries (the
    even split). Sentinels/unset return False (runtime-resolved)."""
    lst = _ratio_vector(v)
    return lst is not None and len(set(lst)) <= 1


def _c_weightless_backend(values: dict, model_cfg) -> Optional[Tuple[str, str]]:
    # enum-value predicate: fastlane -> flashinfer attention backend.
    if not values.get("weightless_kv_fastlane"):
        return None
    ab = values.get("attention_backend")
    if (ab or None) in _FLASHINFER_BACKENDS:
        return None
    return (
        "weightless_kv_fastlane",
        "--weightless-kv-fastlane requires a flashinfer attention backend "
        f"(--attention-backend flashinfer or fa3), got {ab!r} "
        "(hard-rejected by ServerArgs).",
    )


def _c_weightless_tp_eq_dcp(values: dict, model_cfg) -> Optional[Tuple[str, str]]:
    # numeric-equality predicate: fastlane -> tp_size == dcp_size.
    if not values.get("weightless_kv_fastlane"):
        return None
    tp = _int_or(values.get("tp_size"), 1)
    dcp = _int_or(values.get("dcp_size"), 1)
    if dcp == tp:
        return None
    return (
        "weightless_kv_fastlane",
        "--weightless-kv-fastlane requires --dcp-size == --tp-size (the KV "
        f"cache is token-sharded across the whole TP group); got tp_size={tp}, "
        f"dcp_size={dcp} (hard-rejected by ServerArgs).",
    )


def _c_dcp_spec(values: dict, model_cfg) -> Optional[Tuple[str, str]]:
    # An EXPLICIT --dcp-size > 1 together with a speculative algorithm is
    # hard-rejected on CUDA unless the full uneven-weighted-DCP condition
    # holds (env pair or non-'coupled' rank_kv_ratio, non-uniform
    # rank_tp_ratio, dcp == tp). Uneven DCP is env-driven: never pass
    # --dcp-size > 1 with speculation; ServerArgs auto-sets dcp_size itself.
    dcp = _int_or(values.get("dcp_size"), 1)
    if dcp <= 1 or not values.get("speculative_algorithm"):
        return None
    tp = _int_or(values.get("tp_size"), 1)
    rtr = values.get("rank_tp_ratio")
    weighted = (
        bool(values.get("SGLANG_UNEVEN_DCP"))
        and bool(values.get("SGLANG_UNEVEN_DCP_WEIGHTED"))
    ) or values.get("rank_kv_ratio") not in (None, "", "coupled")
    non_uniform = _is_active(catalog()["rank_tp_ratio"], rtr) and not (
        _uniform_explicit_ratio(rtr)
    )
    if weighted and non_uniform and dcp == tp:
        return None
    return (
        "dcp_size",
        "--decode-context-parallel-size/--dcp-size > 1 together with a "
        "speculative algorithm is hard-rejected on CUDA. Do not pass the "
        "flag: enable uneven DCP via SGLANG_UNEVEN_DCP=1 + "
        "SGLANG_UNEVEN_DCP_WEIGHTED=1 on a non-uniform --rank-tp-ratio plan "
        "(ServerArgs then auto-sets dcp_size = tp_size itself).",
    )


def _c_kv_ratio_non_uniform(values: dict, model_cfg) -> Optional[Tuple[str, str]]:
    # non-uniformity predicate: a decoupled KV-token ownership only exists
    # under a NON-uniform rank_tp_ratio plan.
    rkv = values.get("rank_kv_ratio")
    if rkv in (None, "", "coupled"):
        return None
    rtr = values.get("rank_tp_ratio")
    if rtr in (None, "") or _uniform_explicit_ratio(rtr):
        return (
            "rank_kv_ratio",
            "--rank-kv-ratio (other than 'coupled') requires --rank-gpu-id "
            "with a NON-uniform --rank-tp-ratio plan: the token-axis KV "
            "sharding it rebalances only exists under uneven TP "
            "(ServerArgs rejects an explicit vector on the even split).",
        )
    return None


def _c_uniform_tp_ratio(values: dict, model_cfg) -> Optional[Tuple[str, str]]:
    # ServerArgs rejects an explicit all-identical --rank-tp-ratio vector.
    if _uniform_explicit_ratio(values.get("rank_tp_ratio")):
        return (
            "rank_tp_ratio",
            "--rank-tp-ratio with identical entries is the even split — "
            "omit the flag instead (hard-rejected by ServerArgs).",
        )
    return None


def _c_weighted_needs_dcp_env(values: dict, model_cfg) -> Optional[Tuple[str, str]]:
    if values.get("SGLANG_UNEVEN_DCP_WEIGHTED") and not values.get(
        "SGLANG_UNEVEN_DCP"
    ):
        return (
            "SGLANG_UNEVEN_DCP_WEIGHTED",
            "SGLANG_UNEVEN_DCP_WEIGHTED=1 requires SGLANG_UNEVEN_DCP=1 (the "
            "weighted vector refines the uneven-DCP token sharding; alone it "
            "is dead).",
        )
    return None


def _c_adaptive_needs_spec(values: dict, model_cfg) -> Optional[Tuple[str, str]]:
    if values.get("speculative_adaptive") and not values.get(
        "speculative_algorithm"
    ):
        return (
            "speculative_adaptive",
            "--speculative-adaptive requires a --speculative-algorithm (it "
            "adapts the draft length of an active speculative run).",
        )
    return None


def _c_rank_gpu_id_budget(values: dict, model_cfg) -> Optional[Tuple[str, str]]:
    # --rank-gpu-id needs --rank-gpu-memory-mib OR --rank-tp-ratio auto* (the
    # auto modes derive the budgets from NVML totals). An explicit ratio
    # VECTOR does not derive budgets.
    if not _is_active(catalog()["rank_gpu_id"], values.get("rank_gpu_id")):
        return None
    if _is_active(
        catalog()["rank_gpu_memory_mib"], values.get("rank_gpu_memory_mib")
    ):
        return None
    if values.get("rank_tp_ratio") in ("auto", "auto-performance"):
        return None
    return (
        "rank_gpu_id",
        "--rank-gpu-id requires --rank-gpu-memory-mib, or --rank-tp-ratio "
        "auto / auto-performance (which derive the budgets from NVML totals, "
        "optionally minus --rank-auto-reserve-mib). An explicit ratio vector "
        "does not supply budgets (hard-rejected by ServerArgs).",
    )


def _c_hicache_hybrid_layout(values: dict, model_cfg) -> Optional[Tuple[str, str]]:
    # MambaPoolHost (hybrid/mamba models) only supports page_first_direct.
    if not values.get("enable_hierarchical_cache"):
        return None
    if not model_is_hybrid(model_cfg):
        return None
    if values.get("hicache_mem_layout") == "page_first_direct":
        return None
    return (
        "hicache_mem_layout",
        "the hierarchical cache on a hybrid/mamba model requires "
        "--hicache-mem-layout page_first_direct: MambaPoolHost rejects every "
        f"other layout (got {values.get('hicache_mem_layout')!r}).",
    )


#: All cross-field constraints, applied by :func:`resolve` (on the
#: post-auto-set value map) and :func:`validate_profile` (also on the RAW
#: profile settings, so an auto-set cannot mask a broken saved profile).
_CROSS_CONSTRAINTS: Tuple[
    Callable[[dict, Optional[dict]], Optional[Tuple[str, str]]], ...
] = (
    _c_weightless_backend,
    _c_weightless_tp_eq_dcp,
    _c_dcp_spec,
    _c_kv_ratio_non_uniform,
    _c_uniform_tp_ratio,
    _c_weighted_needs_dcp_env,
    _c_adaptive_needs_spec,
    _c_rank_gpu_id_budget,
    _c_hicache_hybrid_layout,
)


def cross_field_errors(
    settings: Optional[dict], model_cfg: Optional[dict] = None
) -> List[Tuple[str, str]]:
    """Evaluate every cross-field constraint against a raw ``settings`` dict
    (no auto-set applied). Returns ``[(flag_id, error), ...]``."""
    values = dict(settings or {})
    out: List[Tuple[str, str]] = []
    for check in _CROSS_CONSTRAINTS:
        hit = check(values, model_cfg)
        if hit is not None:
            out.append(hit)
    return out


# ===========================================================================
# resolve(): per-field {enabled, options, value, disabled_reason, error}.
# ===========================================================================


def _as_int_list(v) -> Optional[List[int]]:
    if v is None or v == "":
        return None
    if isinstance(v, (list, tuple)):
        try:
            return [int(x) for x in v]
        except (TypeError, ValueError):
            return None
    if isinstance(v, str):
        parts = [p for p in v.replace(" ", "").split(",") if p != ""]
        try:
            return [int(p) for p in parts]
        except ValueError:
            return None
    return None


def _is_active(spec: FlagSpec, value) -> bool:
    """Is a flag "active" (set to something other than its inactive default)?"""
    if value is None:
        return False
    if spec.type == "bool":
        return bool(value)
    if isinstance(value, str) and value == "":
        return False
    # int/float/str/list/enum: active when it differs from the inactive default.
    return value != spec.default


def _activation_value(spec: FlagSpec, settings: dict) -> Any:
    """A sensible value that makes a dependency ACTIVE, for auto-set."""
    if spec.type == "bool":
        return True
    if spec.id == "rank_gpu_id":
        tp = int(settings.get("tp_size", 1) or 1)
        return list(range(tp))
    if spec.id == "rank_tp_ratio":
        return "auto"
    if spec.allowed:
        # first non-default allowed value.
        for a in spec.allowed:
            if a != spec.default:
                return a
        return spec.allowed[0]
    if spec.type in ("int",):
        return 1
    if spec.type == "float":
        return 1.0
    if spec.type == "list":
        tp = int(settings.get("tp_size", 1) or 1)
        return list(range(tp)) if tp > 1 else [0]
    return "auto"


def resolve(
    settings: Optional[dict], model_cfg: Optional[dict] = None
) -> Dict[str, dict]:
    """Resolve a user ``settings`` dict against the catalog + model, returning
    ``{id: {enabled, options, value, disabled_reason, error, auto_set}}`` for
    EVERY catalog flag.

    Applies, in order:
      1. model-compat: an incompatible flag is disabled with its reason.
      2. mutual exclusion (symmetric): when one flag of an exclusive pair is
         active, the other is disabled with a reason; if both are active, both
         are flagged as conflicting.
      3. dependency auto-set: when an active flag ``requires`` others, each
         unmet requirement is auto-set to an activation value; a ``requires_any``
         group is satisfied by any member, else its FIRST member is auto-set.
         Already-active compatible requirements are left untouched.
      4. value-constraint propagation: a list flag whose length must equal
         ``tuple_len_flag`` (e.g. tp_size) reports an ``error`` when it does not.
    """
    settings = dict(settings or {})
    cat = catalog()

    # Working value map (dependency auto-set may add to it).
    values = {cid: settings.get(cid, cat[cid].default) for cid in cat}
    out = {
        cid: {
            "enabled": True,
            "options": list(cat[cid].allowed) if cat[cid].allowed else None,
            "value": values[cid],
            "disabled_reason": None,
            "error": None,
            "auto_set": False,
        }
        for cid in cat
    }

    # -- 1. model compatibility --------------------------------------------
    for cid, spec in cat.items():
        ok, reason = spec.model_compat(model_cfg)
        if not ok:
            out[cid]["enabled"] = False
            out[cid]["disabled_reason"] = reason

    # -- 3(pre). dependency auto-set (before exclusion so an auto-set dep can
    #            itself trigger nothing contradictory; exclusion wins later) --
    def _active(cid: str) -> bool:
        return _is_active(cat[cid], values.get(cid))

    # iterate to a fixed point (a dep may itself have deps).
    for _ in range(len(cat) + 1):
        changed = False
        for cid, spec in cat.items():
            if not _active(cid):
                continue
            reqs = list(spec.requires)
            for group in spec.requires_any:
                if not any(g in cat and _active(g) for g in group):
                    # satisfy the first existing member.
                    for g in group:
                        if g in cat:
                            reqs.append(g)
                            break
            for dep in reqs:
                if dep not in cat:
                    continue
                if _active(dep):
                    continue  # already-set compatible value: leave untouched.
                values[dep] = _activation_value(cat[dep], values)
                out[dep]["value"] = values[dep]
                out[dep]["auto_set"] = True
                changed = True
        if not changed:
            break

    # -- 2. mutual exclusion (symmetric) -----------------------------------
    # Build undirected edge set.
    edges = set()
    for cid, spec in cat.items():
        for other in spec.mutually_exclusive_with:
            if other in cat:
                edges.add(frozenset((cid, other)))
    for edge in edges:
        a, b = tuple(edge) if len(edge) == 2 else (next(iter(edge)),) * 2
        aa, ab = _active(a), _active(b)
        if aa and ab:
            for x, y in ((a, b), (b, a)):
                out[x]["enabled"] = False
                out[x]["disabled_reason"] = (
                    f"conflicts with {cat[y].name} ({y}) -- both are set but "
                    "they are mutually exclusive; unset one."
                )
        elif aa and not ab:
            out[b]["enabled"] = False
            out[b]["disabled_reason"] = (
                f"disabled while {cat[a].name} ({a}) is set "
                "(mutually exclusive)."
            )
        elif ab and not aa:
            out[a]["enabled"] = False
            out[a]["disabled_reason"] = (
                f"disabled while {cat[b].name} ({b}) is set "
                "(mutually exclusive)."
            )

    # -- 4. value-constraint propagation (tuple lengths) -------------------
    for cid, spec in cat.items():
        if not spec.tuple_len_flag:
            continue
        v = values.get(cid)
        if v is None:
            continue
        # enum sentinels ('auto', 'auto-performance', 'coupled', 'capacity')
        # are not length-constrained.
        if isinstance(v, str) and (spec.allowed and v in spec.allowed):
            continue
        lst = _as_int_list(v)
        if lst is None:
            continue
        need = int(values.get(spec.tuple_len_flag, 1) or 1)
        if len(lst) != need:
            out[cid]["error"] = (
                f"{spec.name} must have exactly {need} entries to match "
                f"{cat[spec.tuple_len_flag].name}={need} (got {len(lst)}: "
                f"{lst})."
            )

    # -- 5. cross-field constraints (numeric-equality / enum-value /
    #       non-uniformity predicates), evaluated on the RESOLVED value map
    #       (auto-set applied) so they are machine-enforced, not hover text --
    for cid, msg in cross_field_errors(values, model_cfg):
        if cid not in out:
            continue
        out[cid]["error"] = (
            msg if not out[cid]["error"] else out[cid]["error"] + " " + msg
        )

    return out


# ===========================================================================
# Profile generation.
# ===========================================================================


@dataclasses.dataclass
class Profile:
    """A fully-populated named configuration: ``settings`` sets EVERY catalog
    flag. ``info`` carries UI notes (e.g. an MPS-required flag).

    ``env`` carries LAUNCH environment variables that are NOT catalog flags
    (LD_LIBRARY_PATH so the re-exec'd NEXTN draft worker finds
    libnvrtc.so.13, PYTHONPATH, ...). Env vars that ARE catalog entries
    (``SGLANG_UNEVEN_DCP`` ...) live in ``settings`` like every other flag;
    :func:`profile_env` merges both into the mapping the launcher must apply
    to the server process."""

    name: str
    kind: str  # single | normal-tp | colocation | uneven-max-tokens | uneven-max-perf
    settings: Dict[str, Any]
    info: List[str] = dataclasses.field(default_factory=list)
    env: Dict[str, str] = dataclasses.field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "settings": self.settings,
            "info": self.info,
            "env": self.env,
        }

    @classmethod
    def from_json(cls, d: dict) -> "Profile":
        return cls(
            name=d["name"],
            kind=d.get("kind", "custom"),
            settings=dict(d.get("settings", {})),
            info=list(d.get("info", [])),
            env={str(k): str(v) for k, v in (d.get("env") or {}).items()},
        )


def profile_env(profile: Profile) -> Dict[str, str]:
    """The COMPLETE environment mapping the launcher must apply for this
    profile: every ACTIVE env-typed catalog flag from ``settings`` (bools
    render as '1', vectors comma-join), overlaid with the profile's extra
    ``env`` entries (LD_LIBRARY_PATH, PYTHONPATH, ...). Without the
    SGLANG_UNEVEN_DCP pair a spec+DCP boot is hard-rejected; without
    LD_LIBRARY_PATH the NEXTN draft worker dies importing deep_gemm
    (libnvrtc.so.13)."""
    cat = catalog()
    env: Dict[str, str] = {}
    for cid, spec in cat.items():
        if not spec.is_env:
            continue
        v = profile.settings.get(cid, spec.default)
        if not _is_active(spec, v):
            continue
        if spec.type == "bool":
            env[spec.name] = "1"
        elif isinstance(v, (list, tuple)):
            env[spec.name] = ",".join(str(x) for x in v)
        else:
            env[spec.name] = str(v)
    env.update(profile.env)
    return env


#: argv emission order for the flags the reference profile pins; everything
#: else follows in catalog order. Purely cosmetic (argparse is order-free),
#: but keeps generated commands diffable against the validated reference.
_ARGV_ORDER = (
    "model_path",
    "tokenizer_path",
    "served_model_name",
    "load_format",
    "quantization",
    "tp_size",
    "rank_gpu_id",
    "rank_tp_ratio",
    "rank_auto_reserve_mib",
    "rank_gpu_memory_mib",
    "kv_cache_dtype",
    "context_length",
    "max_running_requests",
    "speculative_algorithm",
    "speculative_num_steps",
    "speculative_eagle_topk",
    "speculative_num_draft_tokens",
    "speculative_adaptive",
    "reasoning_parser",
    "tool_call_parser",
    "chat_template",
    "trust_remote_code",
    "enable_metrics",
    "host",
    "port",
    "enable_hierarchical_cache",
    "hicache_size",
    "hicache_storage_backend",
    "hicache_mem_layout",
)

#: serving-identity flags emitted even when they equal the catalog default
#: (a saved profile must pin its port/host, not inherit a future default).
_ARGV_ALWAYS_EMIT = ("host", "port")


def profile_argv(profile: Profile) -> List[str]:
    """Build the ``sglang.launch_server`` CLI argv for a profile: every ACTIVE
    non-env setting as ``--flag [value]`` using the CANONICAL flag spelling
    (``--`` + field name, which argparse always registers; aliases like
    ``--tp`` parse to the same dest). Bool True (store_true) emits the bare
    flag; vectors comma-join. Env-typed entries are excluded — apply
    :func:`profile_env` to the process environment instead."""
    cat = catalog()
    ordered = [cid for cid in _ARGV_ORDER if cid in cat]
    rest = [cid for cid in cat if cid not in _ARGV_ORDER]
    argv: List[str] = []
    for cid in ordered + rest:
        spec = cat[cid]
        if spec.is_env:
            continue
        v = profile.settings.get(cid, spec.default)
        if v is None:
            continue
        # emit only what DIFFERS from the deployed default (a default-True
        # bool like dllm_fdfo is "active" per _is_active but needs no flag),
        # plus the pinned serving-identity flags.
        if v == spec.default and cid not in _ARGV_ALWAYS_EMIT:
            continue
        if not _is_active(spec, v) and cid not in _ARGV_ALWAYS_EMIT:
            continue
        name = "--" + cid.replace("_", "-")
        if spec.type == "bool" or isinstance(v, bool):
            if v and not spec.default:
                argv.append(name)
            # False (or a default-True bool) cannot be expressed as a bare
            # store_true flag; the default applies by omission.
            continue
        argv.append(name)
        if isinstance(v, (list, tuple)):
            argv.append(",".join(str(x) for x in v))
        else:
            argv.append(str(v))
    return argv


def _base_settings() -> Dict[str, Any]:
    """Every catalog flag set to its default (a fully-populated baseline)."""
    return {cid: spec.default for cid, spec in catalog().items()}


def _gpu_totals(gpus: Sequence) -> List[int]:
    out = []
    for g in gpus:
        if isinstance(g, dict):
            out.append(int(g.get("total_mib") or 0))
        else:
            out.append(int(getattr(g, "total_mib", 0) or 0))
    return out


def _gpu_names(gpus: Sequence) -> List[str]:
    out = []
    for g in gpus:
        if isinstance(g, dict):
            out.append(str(g.get("name") or ""))
        else:
            out.append(str(getattr(g, "name", "") or ""))
    return out


# ---------------------------------------------------------------------------
# Rig calibrations: measured, battery-stable per-quant values for known rigs
# (matrix/htsglang_local_run.sh, MATRIX_PLAN 3.3). Order-sensitive: the vectors
# are per-RANK with rank i on physical GPU i (--rank-gpu-id 0,1,2), so they only
# apply to the enumeration they were MEASURED on. That box enumerates (NVML /
# nvidia-smi order) as [3080, 5090, 3080] -- the 5090 sits at index 1, NOT 0.
# Hence tokvec 33,13,18 gives the middle rank (the 5090) the SMALLEST KV share:
# it carries the largest weight/compute shard, so it has the least room for KV.
# Any other enumeration gets no calibration (the caller falls back to a derived
# estimate and says so) -- never remap measured per-rank vectors onto a rig we
# did not measure.
# ---------------------------------------------------------------------------

def _match_calibration(gpus: Sequence, quant: Optional[str]) -> Optional[dict]:
    """A measured calibration for this exact rig shape + checkpoint quant, or
    None (callers must then fall back to a derived estimate AND say so)."""
    names = [n.lower() for n in _gpu_names(gpus)]
    if (
        len(names) == 3
        and "3080" in names[0]
        and "5090" in names[1]
        and "3080" in names[2]
    ):
        if quant == "fp8":
            return {
                "tokvec": [33, 13, 18],
                "reserve": "3000,2200,2200",
                "label": "fp8 (MATRIX_PLAN 3.3, battery-stable to 530944 ctx)",
            }
        if quant == "awq":
            return {
                "tokvec": [31, 15, 18],
                "reserve": "1500",
                "label": "awq (MATRIX_PLAN 3.3, 886080 ctx)",
            }
    return None


def _cuda_lib_dirs(python_exe: str) -> List[str]:
    """The interpreter venv's bundled NVIDIA/torch shared-lib dirs (energy.py
    helper): libnvrtc.so.13 lives in nvidia/cu13/lib and the re-exec'd NEXTN
    draft worker needs it on LD_LIBRARY_PATH at interpreter startup."""
    try:
        from sglang.srt.planner.energy import _venv_cuda_lib_dirs

        return _venv_cuda_lib_dirs(python_exe)
    except Exception:  # pragma: no cover - defensive
        return []


def _fork_launch_env(python_exe: Optional[str], info: List[str]) -> Dict[str, str]:
    """LD_LIBRARY_PATH + PYTHONPATH for the fork profiles (rank-gpu-id
    re-exec'd workers + the NEXTN draft worker's deep_gemm import)."""
    import sys

    env: Dict[str, str] = {}
    exe = python_exe or sys.executable
    dirs = _cuda_lib_dirs(exe)
    if dirs:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(dirs)
    else:
        info.append(
            "WARNING: no bundled nvidia/torch lib dirs found for "
            f"{exe!r} -- set LD_LIBRARY_PATH so the NEXTN draft worker "
            "finds libnvrtc.so.13 (deep_gemm import dies without it)."
        )
    try:
        import sglang

        env["PYTHONPATH"] = os.path.dirname(
            os.path.dirname(os.path.abspath(sglang.__file__))
        )
    except Exception:  # pragma: no cover - defensive
        pass
    return env


def profiles(
    model_cfg: Optional[dict],
    gpus: Sequence,
    base: Optional[dict] = None,
    python_exe: Optional[str] = None,
) -> List[Profile]:
    """Generate LAUNCHABLE profiles for ``gpus`` (a list of ``{name,
    total_mib}`` dicts or GpuDescriptor-like objects), each with ALL fields
    set, plus the launch env (:func:`profile_env`) and argv
    (:func:`profile_argv`).

    ``base`` merges user identity settings (model_path, port, context_length,
    hicache toggles, ...) into every profile before the profile-specific
    correctness rules run, so the rules can react to them (e.g. hicache on a
    hybrid model is forced to the only layout MambaPoolHost accepts).

    Always: single-gpu, and (for >1 GPU) normal-TP using ONLY stock upstream
    options. When the GPU set exceeds the stock TP/kv-head rule (more GPUs
    than KV heads, or heterogeneous VRAM, or a co-location request),
    additionally: a co-location / TP>card-count profile (MPS-required,
    flagged in ``info``) and two full uneven-split profiles (max-tokens,
    max-perf) -- carrying the SGLANG_UNEVEN_DCP env pair (spec+DCP is ONLY
    supported on that path), the measured per-quant calibration when this
    rig/quant is calibrated (fallbacks are called out in ``info``, never
    silently invented), the sm86-safe fp8 KV dtype, and the validated NEXTN
    speculative shape when the checkpoint has MTP layers."""
    totals = _gpu_totals(gpus)
    ngpu = len(totals)
    kvh = model_kv_heads(model_cfg)
    heterogeneous = len(set(totals)) > 1
    hybrid = model_is_hybrid(model_cfg)
    quant = model_quant_kind(model_cfg)
    sm86 = rig_has_sm86(gpus)
    cat = catalog()

    base = dict(base or {})
    base.pop("python_exe", None)
    dropped = sorted(k for k in base if k not in cat)
    base_clean = {k: v for k, v in base.items() if k in cat}
    base_notes: List[str] = []
    if dropped:
        base_notes.append(
            "ignored unknown base settings: " + ", ".join(dropped) + "."
        )

    def _mk(overrides: Dict[str, Any], info: List[str]) -> Dict[str, Any]:
        """base settings + user base + profile overrides, then the
        model-level correctness rules every profile must obey."""
        s = _base_settings()
        s.update(base_clean)
        s.update(overrides)
        # hicache on a hybrid/mamba model: MambaPoolHost accepts ONLY the
        # page_first_direct host layout -- force it (verified rule).
        if s.get("enable_hierarchical_cache") and hybrid:
            if s.get("hicache_mem_layout") != "page_first_direct":
                s["hicache_mem_layout"] = "page_first_direct"
                info.append(
                    "hicache on a hybrid/mamba model: forced "
                    "--hicache-mem-layout page_first_direct (MambaPoolHost "
                    "rejects every other layout)."
                )
        # compressed-tensors checkpoints boot with the explicit
        # --quantization compressed-tensors (the validated matrix flag);
        # fp8 checkpoints self-declare and need no flag.
        if quant == "awq" and not s.get("quantization"):
            s["quantization"] = "compressed-tensors"
            info.append(
                "compressed-tensors checkpoint: set --quantization "
                "compressed-tensors explicitly (validated launch shape)."
            )
        return s

    def _apply_fork_rules(s: Dict[str, Any], info: List[str]) -> None:
        """Rules for the fork (co-location / uneven) profiles: sm86-safe fp8
        KV dtype, validated NEXTN spec shape, hybrid SSM dtype."""
        # KV dtype: fp8 KV maximizes capacity; sm86 (RTX 3080-class) cannot
        # do fp8_e4m3 -> e5m2 whenever ANY sm86 card is in the rig.
        cur = s.get("kv_cache_dtype")
        if cur in (None, "auto"):
            s["kv_cache_dtype"] = "fp8_e5m2" if sm86 else "fp8_e4m3"
            info.append(
                "KV cache dtype "
                + s["kv_cache_dtype"]
                + (
                    " (sm86/RTX-3080-class card present: e4m3 KV is "
                    "unsupported on sm86)."
                    if sm86
                    else " (no sm86 card detected)."
                )
            )
        elif sm86 and cur == "fp8_e4m3":
            s["kv_cache_dtype"] = "fp8_e5m2"
            info.append(
                "overrode kv-cache-dtype fp8_e4m3 -> fp8_e5m2: an sm86 "
                "(RTX 3080-class) card is in the rig and cannot run e4m3 KV."
            )
        # Speculative decoding: the validated fork shape, only when the
        # checkpoint actually ships MTP draft layers.
        if model_has_mtp(model_cfg):
            s.update(
                speculative_algorithm="NEXTN",
                speculative_num_steps=3,
                speculative_eagle_topk=1,
                speculative_num_draft_tokens=4,
                speculative_adaptive=True,
            )
            info.append(
                "speculative decoding: NEXTN chain, steps 3, topk 1, draft "
                "4, adaptive (the validated shape; topk stays 1 -- tree "
                "spec is rejected under uneven DCP)."
            )
        # Hybrid models: bfloat16 SSM state (validated; halves mamba state).
        if hybrid:
            s["SGLANG_MAMBA_SSM_DTYPE"] = "bfloat16"
            info.append(
                "hybrid GDN/mamba model: SGLANG_MAMBA_SSM_DTYPE=bfloat16 "
                "(validated reference setting)."
            )

    def _apply_uneven_env(
        s: Dict[str, Any], info: List[str], max_tokens: bool
    ) -> None:
        """The uneven-DCP env pair + per-quant calibration for the uneven
        profiles. spec+DCP is ONLY supported on the uneven-WEIGHTED path, so
        the env pair is mandatory here."""
        s["SGLANG_UNEVEN_DCP"] = True
        s["SGLANG_UNEVEN_DCP_WEIGHTED"] = True
        cal = _match_calibration(gpus, quant)
        if cal is not None:
            s["SGLANG_UNEVEN_TOKEN_VECTOR"] = list(cal["tokvec"])
            s["rank_auto_reserve_mib"] = cal["reserve"]
            info.append(
                "calibrated for this rig/quant: SGLANG_UNEVEN_TOKEN_VECTOR="
                + ",".join(str(x) for x in cal["tokvec"])
                + ", --rank-auto-reserve-mib "
                + cal["reserve"]
                + " -- "
                + cal["label"]
                + "."
            )
        else:
            info.append(
                "NO measured calibration for this rig/quant combination: "
                "token vector and reserve are runtime-derived ESTIMATES, "
                "not calibrated values. Run a calibration boot, read the "
                "TOKVEC hint line, then pin SGLANG_UNEVEN_TOKEN_VECTOR + "
                "--rank-auto-reserve-mib."
            )
            if max_tokens:
                # one-boot measured convergence toward the capacity optimum.
                s["rank_kv_ratio"] = "capacity"

    out: List[Profile] = []
    launch_env_notes: List[str] = []
    fork_env = _fork_launch_env(python_exe, launch_env_notes)

    # --- single-gpu -------------------------------------------------------
    info = list(base_notes) + ["Single GPU, no tensor parallelism."]
    s = _mk({"tp_size": 1}, info)
    out.append(Profile("single-gpu", "single", s, info=info))

    if ngpu <= 1:
        return out

    # normal (stock) TP is representable only when tp_size divides the KV-head
    # count (the stock even-split rule) and there is one rank per card.
    stock_tp_ok = kvh is None or (kvh % ngpu == 0)
    # The fork-only profiles (co-location, uneven split) are offered exactly
    # when the GPU set EXCEEDS the stock TP/kv-head rule: heterogeneous VRAM,
    # tp does not divide the KV heads, or more ranks than KV heads are wanted.
    needs_fork = (
        heterogeneous
        or (not stock_tp_ok)
        or (kvh is not None and ngpu > kvh)
    )

    # --- normal-TP (stock sglang, only upstream options) ------------------
    if stock_tp_ok and not heterogeneous:
        info = list(base_notes) + [
            f"Even tensor parallelism across {ngpu} identical GPUs, "
            "the stock sglang way (no fork flags).",
        ]
        s = _mk({"tp_size": ngpu}, info)
        out.append(Profile("normal-TP", "normal-tp", s, info=info))
    else:
        reason = []
        if heterogeneous:
            reason.append("GPUs have different VRAM totals")
        if not stock_tp_ok:
            reason.append(
                f"tp_size={ngpu} does not divide the model's KV-head count "
                f"({kvh})"
            )
        # still offer a normal-TP entry, but flagged that stock cannot run it
        # cleanly (the fork's uneven path is the intended route).
        info = list(base_notes) + [
            "Stock even TP is imperfect here: " + "; ".join(reason)
            + ". The fork's uneven-TP profiles below handle it "
            "properly (KV replicate + token-shard).",
        ]
        s = _mk({"tp_size": ngpu}, info)
        out.append(
            Profile("normal-TP (stock, limited)", "normal-tp", s, info=info)
        )

    if not needs_fork:
        # A homogeneous rig whose card count divides the KV heads: stock even
        # TP is the right answer; no fork-only profiles are needed.
        return out

    # --- co-location / TP > card-count (MPS required) ---------------------
    # Add ranks beyond the physical card count OR when tp>kv-heads is the point:
    # place two ranks on the largest card.
    largest = totals.index(max(totals)) if totals else 0
    colo_tp = ngpu + 1
    rank_gpu = list(range(ngpu)) + [largest]  # extra rank co-locates on largest
    # even budget: the scalar must fit BOTH the smallest solo card (minus a
    # context allowance) and half of the shared largest card (minus a shared
    # 2 GiB allowance) -- otherwise the co-located pair is physically
    # impossible on the big card.
    per_rank = max(
        1024, min(min(totals) - 1024, (max(totals) - 2048) // 2)
    )
    info = list(base_notes) + [
        f"tp_size={colo_tp} on {ngpu} cards: rank {colo_tp - 1} "
        f"co-locates with rank {largest} on physical GPU {largest}. "
        "REQUIRES CUDA MPS (two ranks share one card). "
        f"Uniform per-rank budget {per_rank} MiB; ensure "
        f"2 x {per_rank} MiB fits GPU {largest}'s total "
        f"({max(totals)} MiB) -- headroom is the user's responsibility.",
        "Stock sglang cannot do this; it is a fork capability.",
    ] + list(launch_env_notes)
    s = _mk(
        {
            "tp_size": colo_tp,
            "rank_gpu_id": rank_gpu,
            "rank_gpu_memory_mib": per_rank,
        },
        info,
    )
    _apply_fork_rules(s, info)
    out.append(
        Profile(
            "co-location (TP>cards, MPS)",
            "colocation",
            s,
            info=info,
            env=dict(fork_env),
        )
    )

    # --- uneven full-split: max-tokens (VRAM-auto) ------------------------
    info = list(base_notes) + [
        "Uneven TP with VRAM-proportional shards (--rank-tp-ratio auto) "
        "and weighted uneven-DCP KV-token sharding (SGLANG_UNEVEN_DCP env "
        "pair) to MAXIMIZE total KV tokens / context.",
    ] + list(launch_env_notes)
    s = _mk(
        {
            "tp_size": ngpu,
            "rank_gpu_id": list(range(ngpu)),
            "rank_tp_ratio": "auto",
        },
        info,
    )
    _apply_fork_rules(s, info)
    _apply_uneven_env(s, info, max_tokens=True)
    out.append(
        Profile(
            "uneven-max-tokens",
            "uneven-max-tokens",
            s,
            info=info,
            env=dict(fork_env),
        )
    )

    # --- uneven full-split: max-perf (auto-performance) ------------------
    coincide = not heterogeneous
    info = list(base_notes) + [
        "Uneven TP tuned for throughput (--rank-tp-ratio auto-performance): "
        "VRAM-auto split plus a measured MLP-family shift toward the "
        "compute-strong card.",
    ] + list(launch_env_notes)
    if coincide:
        info.append(
            "NOTE: on these homogeneous cards max-perf and max-tokens "
            "coincide (the even split is already optimal for both)."
        )
    s = _mk(
        {
            "tp_size": ngpu,
            "rank_gpu_id": list(range(ngpu)),
            "rank_tp_ratio": "auto-performance",
        },
        info,
    )
    _apply_fork_rules(s, info)
    _apply_uneven_env(s, info, max_tokens=False)
    out.append(
        Profile(
            "uneven-max-perf",
            "uneven-max-perf",
            s,
            info=info,
            env=dict(fork_env),
        )
    )

    return out


def validate_profile(
    profile: Profile, model_cfg: Optional[dict] = None
) -> Tuple[bool, List[str]]:
    """Validate a profile: every catalog flag must be present in ``settings``,
    :func:`resolve` must report no conflict / compat / length error among the
    flags the profile actively sets, and the cross-field constraints must hold
    on the RAW settings (a saved profile launches as-is -- resolve()'s
    dependency auto-set must not mask a missing requirement).
    Returns ``(ok, errors)``."""
    cat = catalog()
    errors: List[str] = []
    for cid in cat:
        if cid not in profile.settings:
            errors.append(f"missing flag {cid}")
    res = resolve(profile.settings, model_cfg)
    for cid, spec in cat.items():
        v = profile.settings.get(cid, spec.default)
        if not _is_active(spec, v):
            continue
        r = res[cid]
        if not r["enabled"] and r["disabled_reason"]:
            errors.append(f"{cid}: {r['disabled_reason']}")
        if r["error"]:
            errors.append(f"{cid}: {r['error']}")
    seen = set(errors)
    for cid, msg in cross_field_errors(profile.settings, model_cfg):
        e = f"{cid}: {msg}"
        if e not in seen:
            errors.append(e)
            seen.add(e)
    return (len(errors) == 0, errors)


# ===========================================================================
# Profile store (user-saveable, JSON).
# ===========================================================================

DEFAULT_STORE_PATH = os.path.expanduser(
    os.environ.get(
        "SGLANG_PLANNER_PROFILES",
        os.path.join("~", ".cache", "sglang", "planner_profiles.json"),
    )
)


class ProfileStore:
    """A tiny JSON-backed store of user-saved :class:`Profile` objects, keyed
    by name. No server, no locking beyond an atomic replace."""

    def __init__(self, path: str = DEFAULT_STORE_PATH):
        self.path = os.path.expanduser(path)

    def _read(self) -> Dict[str, dict]:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
        return {p["name"]: p for p in data.get("profiles", [])}

    def _write(self, by_name: Dict[str, dict]) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"profiles": list(by_name.values())}, f, indent=1)
        os.replace(tmp, self.path)

    def save(self, profile: Profile, overwrite: bool = True) -> None:
        """Persist ``profile`` (overwriting a same-named entry by default)."""
        by_name = self._read()
        if profile.name in by_name and not overwrite:
            raise KeyError(f"profile {profile.name!r} already exists")
        by_name[profile.name] = profile.to_json()
        self._write(by_name)

    def load(self, name: str) -> Profile:
        by_name = self._read()
        if name not in by_name:
            raise KeyError(f"unknown profile {name!r}")
        return Profile.from_json(by_name[name])

    def list(self) -> List[str]:
        return sorted(self._read().keys())

    def load_all(self) -> List[Profile]:
        return [Profile.from_json(d) for d in self._read().values()]

    def delete(self, name: str) -> bool:
        by_name = self._read()
        if name not in by_name:
            return False
        del by_name[name]
        self._write(by_name)
        return True
