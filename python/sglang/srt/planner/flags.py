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
import logging
import os
import re
import typing
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from sglang.srt.registry.nvml import DeviceOrderUnresolvedError

logger = logging.getLogger(__name__)

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
    "model_bundles_dspark_draft",
    "model_quant_kind",
    "model_max_context",
    "rig_has_sm86",
    "find_draft_models",
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


def model_q_heads(model_cfg: Optional[dict]) -> Optional[int]:
    v = _cfg_get(model_cfg, "num_attention_heads")
    try:
        return int(v) if v else None
    except (TypeError, ValueError):
        return None


def stock_tp_legal(model_cfg: Optional[dict], tp: int) -> Tuple[bool, str]:
    """VERIFIED stock-sglang TP rule (QKVParallelLinear): a tp value is
    expressible in stock exactly when

      * q_heads % tp == 0 (Q heads shard evenly), AND
      * kv_heads % tp == 0 when tp <= kv_heads (head sharding), OR
        tp % kv_heads == 0 when tp > kv_heads (kv replication,
        num_kv_head_replicas = divide(tp, kv)).

    Non-divisible combinations are NOT a stock option -- they need the fork's
    uneven path (KV replicate + token-shard). Returns ``(ok, reason)`` with a
    NEUTRAL reason string (a fact about stock's rule, not a verdict). Unknown
    head counts (no config) are treated as legal."""
    if tp <= 1:
        return True, ""
    q = model_q_heads(model_cfg)
    kv = model_kv_heads(model_cfg)
    if q is not None and q % tp != 0:
        return False, (
            f"stock requires q_heads % tp == 0 (q_heads={q}, tp={tp})"
        )
    if kv is None:
        return True, ""
    if tp <= kv and kv % tp == 0:
        return True, ""
    if tp > kv and tp % kv == 0:
        return True, ""
    return False, (
        f"stock requires kv_heads % tp == 0 or tp % kv_heads == 0 "
        f"(kv_heads={kv}, tp={tp}); this combination needs the fork path"
    )


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


def model_bundles_dspark_draft(model_cfg: Optional[dict]) -> bool:
    """True when the checkpoint bundles a DSpark draft head, marked by the
    prefixed ``dspark_*`` keys on the target config. Mirrors
    ``speculative.dspark_components.dspark_config.checkpoint_bundles_dspark_draft``
    on the planner's flat/nested dict shape (the planner reads config.json
    directly and must not import the runtime)."""
    return any(
        _cfg_get(model_cfg, key) is not None
        for key in (
            "dspark_block_size",
            "dspark_markov_rank",
            "dspark_noise_token_id",
            "dspark_target_layer_ids",
        )
    )


def model_has_mtp(model_cfg: Optional[dict]) -> bool:
    """True when the checkpoint ships MTP draft layers (NEXTN speculative
    decoding is possible): ``mtp_num_hidden_layers`` or
    ``num_nextn_predict_layers`` > 0.

    A DSpark-bundled checkpoint is excluded even when it declares
    ``num_nextn_predict_layers``. Every published DeepSeek-V4 config carries
    ``num_nextn_predict_layers: 1``, but on the DSpark variants the ``mtp.*``
    namespace holds the three DSpark stages (``markov_head.*``,
    ``confidence_head.proj``, ``main_proj``) and no NextN block at all --
    checked against the safetensors index of ``DeepSeek-V4-Flash`` (1 stage,
    ``enorm``/``hnorm`` = NextN, no ``dspark_*`` keys) versus
    ``DeepSeek-V4-Flash-DSpark`` and ``DeepSeek-V4-Flash-0731`` (3 stages,
    markov/confidence, ``dspark_*`` keys). Reading the field as "NEXTN is
    possible" on the latter produces a preset whose draft weights cannot
    load."""
    if model_bundles_dspark_draft(model_cfg):
        return False
    v = _cfg_get(model_cfg, "mtp_num_hidden_layers")
    if v is None:
        v = _cfg_get(model_cfg, "num_nextn_predict_layers")
    try:
        return v is not None and int(v) > 0
    except (TypeError, ValueError):
        return False


def model_max_context(model_cfg: Optional[dict]) -> Optional[int]:
    """The model's OWN maximum context length (native positional limit, e.g.
    ``max_position_embeddings``), or None when the config does not declare
    one. The capacity rules never set ``--context-length`` above this."""
    for key in (
        "max_position_embeddings",
        "max_seq_len",
        "max_sequence_length",
        "n_ctx",
        "seq_length",
        "model_max_length",
    ):
        v = _cfg_get(model_cfg, key)
        if v is None:
            continue
        try:
            iv = int(v)
        except (TypeError, ValueError):
            continue
        if iv > 0:
            return iv
    return None


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
# Draft-model matcher: speculative decoding for checkpoints WITHOUT an MTP
# head via a separate local EAGLE/EAGLE3 draft model
# (--speculative-algorithm + --speculative-draft-model-path).
# ===========================================================================

#: a model name that carries one of these markers is a draft/speculator head,
#: never a base model (case-insensitive substring match on the full name).
_DRAFT_MARKER_RE = re.compile(r"eagle|speculator|draft", re.IGNORECASE)

#: tokens that carry NO family identity (quant tags, format/role suffixes,
#: vendor prefixes): dropped from the BASE tokens so "Qwen3.6-27B-AWQ" still
#: requires only {qwen3.6, 27b} of a draft name -- and never REQUIRED of the
#: draft (a draft is allowed to carry any of them).
_FAMILY_NOISE_TOKENS = frozenset({
    "it", "instruct", "chat", "base", "hf", "sglang", "gguf", "awq", "gptq",
    "bnb", "bf16", "fp16", "fp8", "f16", "f32", "int2", "int4", "int8",
    "nvfp4", "mxfp4", "4bit", "8bit", "dynamic", "quantized", "compressed",
    "tensors", "unsloth", "ud", "vl", "eagle", "eagle3", "speculator",
    "draft", "mtp", "spec", "head",
})
#: quant-shaped tokens (Q4_K_M splits into q4/k/m on the '_' separator) --
#: also noise, matched by shape instead of enumeration.
_QUANT_TOKEN_RE = re.compile(
    r"^(i?q\d\w*|w\d+a\d+\w*|k|m|s|l|xl|xs|xxs)$", re.IGNORECASE
)


def _name_tokens(name: str) -> List[str]:
    """Lowercase identity tokens of a model name: split on separators but NOT
    on '.' (Qwen3.6 must stay one token; a version-truncated 'qwen3' must
    never match a 'qwen3.6' draft)."""
    return [t for t in re.split(r"[\s/\\_,+-]+", str(name or "").lower()) if t]


def _family_tokens(name: str) -> List[str]:
    """The tokens a DRAFT model's name must contain to count as the same
    family: the base name's tokens minus quant/format/role noise."""
    base = os.path.basename(str(name or "").rstrip("/"))
    return [
        t
        for t in _name_tokens(base)
        if t not in _FAMILY_NOISE_TOKENS and not _QUANT_TOKEN_RE.match(t)
    ]


def _draft_config_arch(path: str) -> str:
    """Lowercased ``architectures`` string of a candidate's config.json (empty
    when unreadable) -- used to detect a draft head whose NAME lacks a marker
    and to refine the EAGLE vs EAGLE3 kind."""
    try:
        with open(os.path.join(path, "config.json")) as f:
            cfg = json.load(f)
        archs = cfg.get("architectures")
        if isinstance(archs, list):
            return " ".join(str(a) for a in archs).lower()
    except Exception:
        pass
    return ""


def find_draft_models(
    base_ref: Optional[str], models: Sequence
) -> List[Dict[str, str]]:
    """Find local DRAFT-model candidates for ``base_ref`` among ``models``
    (``server_manager.DiscoveredModel``-likes or dicts with name/path).

    A candidate must (a) be a draft/speculator head -- its name contains
    eagle/eagle3/speculator/draft, or its config architectures contain Eagle
    -- AND (b) match the base model's family: EVERY non-noise token of the
    base name (quant tags like AWQ/BF16/INT4 dropped) appears verbatim in the
    candidate's name tokens. Deliberately conservative: a wrong draft is
    worse than none, so token matching is exact (a 'qwen3' base never matches
    a 'qwen3.6' draft and a gemma draft is never offered for a Qwen base).

    Returns ``[{"name", "path", "algorithm"}]`` with algorithm EAGLE3 / EAGLE
    (from the name/config kind markers) or STANDALONE when the candidate is a
    draft by name only, EAGLE3 candidates first."""
    required = _family_tokens(base_ref or "")
    if not required or not models:
        return []
    base_real = os.path.realpath(str(base_ref)) if base_ref else None
    out: List[Dict[str, str]] = []
    for m in models:
        if isinstance(m, dict):
            name = str(m.get("name") or "")
            path = str(m.get("path") or "")
            broken = m.get("error")
        else:
            name = str(getattr(m, "name", "") or "")
            path = str(getattr(m, "path", "") or "")
            broken = getattr(m, "error", None)
        if broken:
            continue  # a broken draft is worse than none
        if not name and path:
            name = os.path.basename(path.rstrip("/"))
        if not name:
            continue
        if base_real and path and os.path.realpath(path) == base_real:
            continue  # the base model itself is never its own draft
        toks = set(_name_tokens(name))
        if path:
            toks |= set(_name_tokens(os.path.basename(path.rstrip("/"))))
        if not all(t in toks for t in required):
            continue
        hay = name.lower()
        if not _DRAFT_MARKER_RE.search(hay):
            arch = _draft_config_arch(path)
            if "eagle" not in arch:
                continue  # neither name nor config marks it as a draft head
            hay += " " + arch
        # kind: eagle3 > eagle from the name; a bare draft/speculator marker
        # falls back to the config architectures, else STANDALONE.
        if "eagle" not in hay:
            hay += " " + _draft_config_arch(path)
        if "eagle3" in hay:
            algo = "EAGLE3"
        elif "eagle" in hay:
            algo = "EAGLE"
        else:
            algo = "STANDALONE"
        out.append({"name": name, "path": path or name, "algorithm": algo})
    rank = {"EAGLE3": 0, "EAGLE": 1, "STANDALONE": 2}
    out.sort(key=lambda c: (rank.get(c["algorithm"], 9), c["name"].lower()))
    return out


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
      * ``tuple_len_times_flag`` -- optional SECOND factor for that length, so
        a WORLD-length vector can be expressed (``--rank-gpu-id`` under a
        pipeline wants ``pp_size * tp_size`` entries, in world-rank order
        ``pp_rank * tp_size + tp_rank`` -- ``server_args.py:9698-9712``).
        Absent (or 1) reduces to the plain ``tuple_len_flag`` rule.
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
    tuple_len_times_flag: Optional[str] = None
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
# --rank-gpu-id together with dp/ep > 1, nnodes > 1, an explicit
# --mem-fraction-static, and a non-default --base-gpu-id/--gpu-id-step; it
# requires --rank-gpu-memory-mib (or --rank-tp-ratio auto) whenever
# --rank-gpu-id is set; --rank-{mlp,vocab,moe,kv}-ratio all require an
# active plan as noted in their help; the weightless-KV lane requires its
# fast-lane switch and hard-rejects speculative decoding + mixed-chunk.
#
# "Verified" now means EXECUTED, not read (#500-I1/I2/I3). Two of these edges
# had drifted the other way -- pp_size was listed as exclusive with
# --rank-gpu-id where the runtime REQUIRES it, and --rank-tp-ratio was listed
# as requiring a placement where the runtime deliberately decouples them -- so
# the dashboard forbade two configurations the server supports. Every edge on
# the uneven-TP flags is now asserted against the real predicate by driving
# ServerArgs: test/registered/unit/planner/test_flag_registry_contract_500.py.
# Add an edge there when you add one here.
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
        # The reverse half of #500-I2: `resolve` treats exclusion
        # SYMMETRICALLY, so leaving the edge here would keep greying
        # --rank-gpu-id under a pipeline even with the forward edge removed.
        hover="Pipeline-parallel size. Combines WITH the fork's heterogeneous "
        "rank->GPU mapping: an uneven plan under pp_size > 1 REQUIRES "
        "--rank-gpu-id, given as pp_size x tp_size entries in world-rank order "
        "(pp_rank * tp_size + tp_rank) so each stage owns its own group of "
        "physical cards. Without a placement every stage would shard by the "
        "same ratio onto the same cards, which is what the runtime refuses "
        "(server_args.py:9596).",
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
        tuple_len_times_flag="pp_size",
        requires_any=(("rank_gpu_memory_mib", "rank_tp_ratio"),),
        # pp_size is NOT here (#500-I2). The runtime does the opposite: with a
        # pipeline and an uneven plan it REQUIRES --rank-gpu-id
        # (server_args.py:9596, "requires --rank-gpu-id to give each pipeline
        # stage its own group of physical GPUs") and validates the world-length
        # form (:9698-9712). Listing it as exclusive made the dashboard unable
        # to express §1's TPxPPxTP feature at all. dp/ep/nnodes/base_gpu_id/
        # gpu_id_step ARE refused by the runtime -- each verified by driving
        # _handle_uneven_tp, see test_flag_registry_contract_500.
        mutually_exclusive_with=(
            "mem_fraction_static",
            "dp_size",
            "ep_size",
            "nnodes",
            "base_gpu_id",
            "gpu_id_step",
        ),
        hover="One physical GPU index per rank: length == tp_size without a "
        "pipeline, pp_size x tp_size WITH one (world-rank order "
        "pp_rank * tp_size + tp_rank, no card shared between stages). "
        "Duplicates co-locate several ranks on one physical card "
        "(MPS-isolated). Requires --rank-gpu-memory-mib, OR --rank-tp-ratio "
        "auto to derive the budgets from NVML totals. Enables the "
        "heterogeneous-TP path, and is REQUIRED (not forbidden) when an "
        "uneven plan runs under --pp-size > 1.",
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
        # NO `requires=("rank_gpu_id",)` (#500-I3). The vector describes the
        # PARTITION, not the placement, and the runtime decouples them on
        # purpose (server_args.py:9540-9548): the whole --rank-tp-ratio /
        # --rank-kv-ratio validation block is hoisted ABOVE the
        # `if self.rank_gpu_id is None: ... return` early return, because the
        # cross-vendor two-launcher bring-up (--nnodes 2, one CUDA venv + one
        # ROCm venv) places each rank itself and --rank-gpu-id could not
        # describe the AMD rank at all -- it resolves devices through NVML.
        # The 'auto' MODES do need a placement to derive budgets from (:8971),
        # which is a value-level rule and lives in cross_field_errors.
        hover="Uneven-TP weight vector (one positive int per rank, sum must "
        "divide every sharded dim), or 'auto' (weights from the VRAM budgets) "
        "or 'auto-performance' (VRAM split + a measured MLP-family vector for "
        "throughput). Attention splits in whole KV-head units. An EXPLICIT "
        "vector does NOT require --rank-gpu-id (it is a partition, not a "
        "placement; this is what lets another launcher place the ranks); the "
        "'auto' modes do, since they derive the weights from per-card budgets, "
        "and so does any pipeline (--pp-size > 1). NOTE: this is enumerable "
        "('auto'/'auto-performance') OR a free integer list -- the dropdown "
        "offers the two auto modes and a 'custom vector' entry.",
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
        allowed=(
            "both",
            "dec",
            "enc",
            "maxkv",
            "phase-prefill",
            "phase-decode",
        ),
        requires=("rank_tp_ratio",),
        hover="auto-performance tuning target: 'enc' maximizes prefill, 'both' "
        "(default) prefill+concurrent throughput, 'maxkv' maximum context "
        "(selects --rank-kv-ratio capacity), 'dec' decode -- a near-no-op on "
        "the WEIGHT split (decode is flat across splits) but it selects "
        "--rank-kv-ratio speed, which is where the decode lever actually sits. "
        "'phase-prefill'/'phase-decode' are the two arms of the phase-optimal "
        "recipe (#354/#357): one weight vector per phase, decode-knee guard "
        "advisory on the prefill arm, fundability still binding. "
        "Only valid with auto-performance.",
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
        allowed=("coupled", "capacity", "auto", "speed"),
        requires=("rank_gpu_id", "rank_tp_ratio"),
        hover="Uneven-DCP KV-TOKEN ownership vector, DECOUPLED from the weight "
        "split. 'coupled' (default) = previous behavior; 'capacity'/'auto' = "
        "measured capacity-weighted ownership (maximizes max_total_num_tokens, "
        "shifts deep-context attention work to bigger cards); 'speed' = the "
        "counterpart, ownership shifted toward the memory-bandwidth "
        "proportion as far as --rank-perf-loose-ctx-percent allows (measured "
        "-24.5 % on the context-dependent part of the decode step, #210); or "
        "an explicit per-rank integer vector. Requires --rank-gpu-id with a "
        "NON-uniform --rank-tp-ratio plan. Env SGLANG_UNEVEN_TOKEN_VECTOR "
        "overrides.",
    ),
    # ---- fork: weightless-KV fast lane (Variant C) ------------------------
    "weightless_kv_fastlane": dict(
        source="fork",
        group="weightless-kv",
        type="bool",
        default=False,
        mutually_exclusive_with=("enable_mixed_chunk",),
        hover="EXPERIMENTAL: one 'head rank' holds the full weights and runs "
        "collective-free TP=1; the other DCP ranks are weightless (KV "
        "token-shard + attention only), so a large KV lives on smaller cards "
        "while single-stream speed stays near the head card's rate. Requires "
        "tp_size == dcp_size and a flashinfer attention backend. "
        "--enable-mixed-chunk is hard-rejected. Speculative decoding is "
        "supported for CHAIN drafts only (#143): EAGLE/EAGLE3/NEXTN at "
        "--speculative-eagle-topk 1, with --speculative-draft-placement solo "
        "hosted on the head rank and a fixed --speculative-num-steps; tree "
        "verify, adaptive k and the block-decode/host-spill lane stay "
        "hard-rejected alongside it.",
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
        # #379: a UI PICK-LIST, deliberately narrower than what the server
        # accepts -- these are the algorithms the profiles and the UI have
        # something to say about. It is NOT a validation list and must not
        # become one: SpeculativeAlgorithm.known_names() is the single source
        # of truth for what --speculative-algorithm may say, and it includes
        # plugin-registered names this static tuple cannot know. A test pins
        # that every entry here is still resolvable, so the pick-list cannot
        # drift into offering something the server would reject.
        allowed=("EAGLE", "EAGLE3", "NEXTN", "STANDALONE", "NGRAM", "DSPARK"),
        hover="Speculative-decoding algorithm (EAGLE/EAGLE3/NEXTN/...). "
        "DSPARK is the block draft bundled in DeepSeek-V4 checkpoints "
        "(dspark_* config keys); it needs --speculative-dspark-block-size "
        "semantics, not the NEXTN chain shape. Mutually exclusive with the "
        "weightless-KV fast lane.",
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
    if values.get("weightless_kv_fastlane"):
        # #143: the lane is the second supported spec+DCP config (chain drafts
        # only). Its own admission rules are checked by ServerArgs
        # (_reject_unsupported_weightless_spec); this constraint only guards the
        # blanket CUDA door, which the lane no longer sits behind. The lane also
        # REQUIRES dcp_size == tp_size, so an explicit --dcp-size here is
        # expected rather than suspicious.
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


def _c_rank_tp_ratio_placement(values: dict, model_cfg) -> Optional[Tuple[str, str]]:
    """The two value-level cases where --rank-tp-ratio DOES need a placement.

    The flat ``requires=("rank_gpu_id",)`` edge this replaces was wrong for the
    common case (#500-I3): an explicit vector describes the partition and is
    legal alone, which is what the cross-vendor two-launcher bring-up needs.
    Two narrower rules are real, and both are runtime raises:

    * the ``auto`` modes derive the weights from per-card budgets, so they need
      the placement -- ``server_args.py:8971``, "--rank-tp-ratio auto requires
      --rank-gpu-id";
    * under a pipeline every stage must own its own group of physical cards,
      or all stages shard by the same vector onto the same cards --
      ``server_args.py:9596``.
    """
    ratio = values.get("rank_tp_ratio")
    if not _is_active(catalog()["rank_tp_ratio"], ratio):
        return None
    if _is_active(catalog()["rank_gpu_id"], values.get("rank_gpu_id")):
        return None
    if ratio in ("auto", "auto-performance"):
        return (
            "rank_tp_ratio",
            f"--rank-tp-ratio {ratio} derives the per-rank weights from the "
            "cards' budgets, so it requires --rank-gpu-id. An EXPLICIT integer "
            "vector does not (it is a partition, not a placement).",
        )
    if _int_or(values.get("pp_size"), 1) > 1:
        return (
            "rank_gpu_id",
            "--rank-tp-ratio with --pp-size > 1 requires --rank-gpu-id: pass "
            "pp_size x tp_size entries in world-rank order "
            "(pp_rank * tp_size + tp_rank) so each pipeline stage owns its own "
            "group of physical GPUs. Without it every stage would shard by the "
            "same ratio onto the same cards.",
        )
    return None


#: Host layouts a hybrid/mamba model can actually run.
#:
#: This list USED to be page_first_direct alone, on the grounds that
#: "MambaPoolHost rejects every other layout". That was true of the pool's old
#: constructor guard and is no longer true of anything: MambaPoolHost accepts
#: layer_first, and every one of its methods already carried a layer-first
#: branch. Keeping the old rule would have made the planner -- the config
#: authority in this repo -- steer hybrid models onto page_first_direct, the one
#: route whose host write-back segfaults on CUDA (#760), and refuse the layout
#: the server itself falls back to. A constraint that contradicts the runtime is
#: worse than no constraint, because it is believed.
#:
#: page_first stays out: it is the value that arms the write-back staging
#: kernel MambaPoolHost cannot drive. page_head stays out on #441a's evidence.
_HYBRID_HICACHE_LAYOUTS = ("layer_first", "page_first_direct")


def _c_hicache_hybrid_layout(values: dict, model_cfg) -> Optional[Tuple[str, str]]:
    if not values.get("enable_hierarchical_cache"):
        return None
    if not model_is_hybrid(model_cfg):
        return None
    if values.get("hicache_mem_layout") in _HYBRID_HICACHE_LAYOUTS:
        return None
    return (
        "hicache_mem_layout",
        "the hierarchical cache on a hybrid/mamba model requires "
        f"--hicache-mem-layout one of {list(_HYBRID_HICACHE_LAYOUTS)}: "
        "MambaPoolHost cannot drive the write-back staging kernel that "
        "page_first arms, and page_head takes the lf_ph route #441a refuses "
        f"(got {values.get('hicache_mem_layout')!r}).",
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
    _c_rank_tp_ratio_placement,
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
        basis = f"{cat[spec.tuple_len_flag].name}={need}"
        if spec.tuple_len_times_flag:
            factor = int(values.get(spec.tuple_len_times_flag, 1) or 1)
            if factor > 1:
                basis = (
                    f"{cat[spec.tuple_len_times_flag].name}={factor} x "
                    f"{cat[spec.tuple_len_flag].name}={need}"
                )
                need *= factor
        if len(lst) != need:
            out[cid]["error"] = (
                f"{spec.name} must have exactly {need} entries to match "
                f"{basis} (got {len(lst)}: {lst})."
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
    "speculative_draft_model_path",
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
# Capacity rules: every generated preset picks --context-length and
# --max-running-requests from the planner's EXISTING sizing path
# (feasibility.plan -> the server's own PerfCostModel) instead of leaving
# them to the form defaults. No new sizing math lives here.
# ---------------------------------------------------------------------------


def _gpu_flops(g) -> float:
    """Best-effort FLOPs for the single-GPU tiebreak: a measured/explicit value
    on the descriptor wins; else the SEED_CARDS peak table matched by name;
    else 0 (unknown cards tie and fall through to first-index)."""
    name = str((g.get("name") if isinstance(g, dict) else getattr(g, "name", "")) or "")
    for key in ("gemm_tflops", "tflops", "peak_gemm_tflops_fp16"):
        v = g.get(key) if isinstance(g, dict) else getattr(g, key, None)
        if v:
            return float(v)
    try:
        from sglang.srt.planner.card_library import SEED_CARDS

        low = name.lower()
        for p in SEED_CARDS.values():
            if p.name.lower() in low or low in p.name.lower():
                return float(p.peak_gemm_tflops_fp16 or 0.0)
    except Exception:
        pass
    return 0.0


def _cuda_index_at(gpus: Sequence, pos: int) -> Tuple[int, str]:
    """The CUDA-order index (the space --base-gpu-id / --rank-gpu-id /
    CUDA_VISIBLE_DEVICES are interpreted in) of the card at LIST position
    ``pos``, plus how it was resolved.

    The inventory list itself is NVML/detect-ordered (PCI bus), which
    DIVERGES from CUDA's FASTEST_FIRST order on mixed rigs -- list positions
    must never be written into cuda-space flags. Resolution:
      1. an explicit ``cuda_index`` on the descriptor (the detect_hardware
         payload carries it, bridged by the #331 identity map) -> "bridged";
      2. else the card's UUID looked up in that same identity map through the
         cached planner adapter -> "bridged";
      3. else, for an inventory that carries NO card identity at all -- the
         offline ``--gpu NAME:MIB`` specs -- the list position, under the
         convention #392 already fixed for manual ``HardwareSpec``s: a spec
         that declares no CUDA order means its own order -> "declared".

    A card that DOES carry identity and still cannot be placed raises
    :class:`~sglang.srt.registry.nvml.DeviceOrderUnresolvedError` (#397).
    Before that, this function emulated FASTEST_FIRST by sorting on the
    SEED_CARDS fp16 peak and labelled the result "heuristic"; unknown cards
    ranked 0.0 and silently kept NVML order, so the emulation wrote a
    confident CUDA index for a rig it had never seen into a launch flag.
    Callers now omit the pin instead of guessing it.
    """

    def _field(g, key):
        return g.get(key) if isinstance(g, dict) else getattr(g, key, None)

    def _cu(g):
        v = _field(g, "cuda_index")
        return int(v) if v is not None else None

    v = _cu(gpus[pos])
    if v is not None:
        return v, "bridged"

    uuid = _field(gpus[pos], "uuid")
    if not uuid:
        # No identity on ANY card: an offline/manual inventory. There is no
        # live order to get wrong, so the declared order is the meaning.
        if not any(_field(g, "uuid") for g in gpus):
            return pos, "declared"
    else:
        try:
            from sglang.srt.planner.device_map import device_map

            bridged = device_map().cuda_for_uuid(uuid)
        except Exception:  # pragma: no cover - defensive
            bridged = None
        if bridged is not None:
            return int(bridged), "bridged"

    from sglang.srt.registry.nvml import (
        DeviceOrderUnresolvedError,
        cuda_bridge_diagnosis,
    )

    name = _gpu_names(gpus)[pos] if pos < len(_gpu_names(gpus)) else "?"
    raise DeviceOrderUnresolvedError(
        f"the flag planner: card at inventory position {pos} ({name}"
        + (f", uuid {uuid}" if uuid else ", no uuid")
        + ") has no CUDA-order index, so no --base-gpu-id / --rank-gpu-id "
        "value can be written for it. Refusing to guess: the list position "
        "is the NVML/PCI order, which differs from CUDA's on a mixed rig "
        f"(#397). Reason: {cuda_bridge_diagnosis()}"
    )


def _pick_stock_subset(gpus: Sequence, totals: Sequence[int], n: int):
    """Pick ``n`` cards for a stock normal-TP subset preset. Preference:
    a set of IDENTICAL-VRAM cards (the stock even split fits them cleanly;
    largest such total wins -- on the reference box the two 3080s), else the
    ``n`` largest cards. Returns ``(list_positions, cuda_indices, why)`` or
    None when the rig has fewer than ``n`` cards, or when the CUDA order of a
    picked card cannot be resolved -- the preset's whole content is a set of
    cuda-space pins, so an unplaceable card means there is no preset to emit
    (#397), not a preset with guessed pins."""
    if len(totals) < n:
        return None
    by_total: Dict[int, List[int]] = {}
    for pos, t in enumerate(totals):
        by_total.setdefault(int(t), []).append(pos)
    identical = [t for t, ps in by_total.items() if len(ps) >= n]
    if identical:
        t = max(identical)
        positions = by_total[t][:n]
        why = f"Identical-VRAM pick ({t} MiB each): the even split fits cleanly."
    else:
        order = sorted(range(len(totals)), key=lambda p: -int(totals[p]))
        positions = sorted(order[:n])
        why = (
            "No identical-VRAM set of this size; picked the largest cards "
            "(the even split is sized by the smallest of them)."
        )
    try:
        cuda_idx = [_cuda_index_at(gpus, p)[0] for p in positions]
    except DeviceOrderUnresolvedError as e:
        logger.warning("stock subset preset omitted: %s", e)
        return None
    return positions, cuda_idx, why


def _pick_single_gpu(gpus: Sequence):
    """Default card for the single-gpu preset:
    ``(list_pos, cuda_index, reason)`` or None.

    Rule: largest total VRAM; VRAM tie -> higher FLOPs; full tie -> first.
    ``list_pos`` indexes the given inventory list (for sizing against that
    card); ``cuda_index`` is what --base-gpu-id must pin -- the two are
    DIFFERENT spaces (list = NVML/detect order, flag = CUDA order).

    ``cuda_index`` is None when the card's CUDA order cannot be resolved
    (#397). The pick still stands -- it decides which card the preset is
    SIZED against, which is a list-space question -- but no pin is emitted,
    and the reason says so. Writing the list position into --base-gpu-id
    there is the defect: on the reference box that pins a 3080 while sizing
    for the 5090."""
    if not gpus:
        return None

    def _mib(g) -> int:
        v = g.get("total_mib") if isinstance(g, dict) else getattr(g, "total_mib", 0)
        return int(v or 0)

    def _name(g) -> str:
        return str((g.get("name") if isinstance(g, dict) else getattr(g, "name", "")) or "?")

    best = 0
    for i in range(1, len(gpus)):
        if _mib(gpus[i]) > _mib(gpus[best]):
            best = i
        elif _mib(gpus[i]) == _mib(gpus[best]) and _gpu_flops(gpus[i]) > _gpu_flops(gpus[best]):
            best = i
        # full tie: keep the earlier index
    try:
        cuda_idx, cuda_src = _cuda_index_at(gpus, best)
    except DeviceOrderUnresolvedError as e:
        logger.warning("single-gpu preset emits no --base-gpu-id: %s", e)
        return (
            best,
            None,
            f"GPU pick: {_name(gpus[best])} ({_mib(gpus[best])} MiB) -- "
            "largest VRAM, ties broken by FLOPs, then first. NOT pinned: "
            "this card's CUDA-order index could not be resolved, and "
            "--base-gpu-id is a CUDA-order index, so any value written here "
            f"would be a guess. {e}",
        )
    why = (
        f"GPU pick: {_name(gpus[best])} ({_mib(gpus[best])} MiB) at "
        f"cuda:{cuda_idx} -- largest VRAM, ties broken by FLOPs, then first. "
        "Selectable in the UI; pinned via the stock --base-gpu-id flag, "
        "which is a CUDA-order index (FASTEST_FIRST), NOT the NVML/"
        "nvidia-smi index."
        + (
            " (cuda index is the DECLARED order of this offline inventory: "
            "the spec carries no card identity, so its own order is its "
            "meaning -- verify it against nvidia-smi before launching)"
            if cuda_src == "declared"
            else ""
        )
    )
    return best, cuda_idx, why


# ---------------------------------------------------------------------------
# Co-location rank-distribution helpers: shared between the colocation
# preset below and the Runner's per-card rank steppers (webui.py mirrors
# coloDefaultCounts / coloRankGpuIdFromCounts / coloParseRankGpuId in JS --
# keep the two sides in sync). All indices are CUDA-order (the --rank-gpu-id
# space), never NVML/inventory positions.
# ---------------------------------------------------------------------------


def colocation_rank_counts(
    tp: int, cards: Sequence[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """Default per-card rank counts when ``tp`` ranks land on ``cards``
    (sequence of ``(cuda_index, total_mib)``): VRAM-proportional floor
    allocation, then the remaining ranks go to the LARGEST card first
    (descending VRAM, tie: lowest cuda index, cycling). On the reference box
    (5090 + 2x3080) tp=4 puts the extra rank on the 5090: 2/1/1.

    Returns ``[(cuda_index, count)]`` ascending by cuda index. Deterministic
    by construction -- the UI re-derives the same distribution on every
    tp_size change."""
    cards = sorted(((int(c), max(0, int(t or 0))) for c, t in cards))
    if tp <= 0 or not cards:
        return [(c, 0) for c, _t in cards]
    total = sum(t for _c, t in cards) or 1
    counts = {c: (tp * t) // total for c, t in cards}
    assigned = sum(counts.values())
    order = sorted(cards, key=lambda ct: (-ct[1], ct[0]))
    i = 0
    while assigned < tp:
        counts[order[i % len(order)][0]] += 1
        assigned += 1
        i += 1
    return [(c, counts[c]) for c, _t in cards]


def rank_gpu_id_from_counts(counts) -> List[int]:
    """Canonical ``--rank-gpu-id`` list from per-card rank counts (a dict
    ``{cuda_index: count}`` or a sequence of ``(cuda_index, count)`` in any
    order): cards ascending by CUDA index, each card's index repeated its
    count -- 2 ranks on cuda:0 + 1 each on cuda:1/2 -> ``[0, 0, 1, 2]``."""
    pairs = counts.items() if isinstance(counts, dict) else counts
    out: List[int] = []
    for cuda, count in sorted((int(c), int(n)) for c, n in pairs):
        out.extend([cuda] * max(0, count))
    return out


def rank_counts_from_gpu_id(v, enabled_cuda) -> Optional[Dict[int, int]]:
    """Reverse mapping for the UI steppers: parse a ``--rank-gpu-id`` value
    (int list or comma string) into ``{cuda_index: rank count}``. Returns
    None when the value is empty, any entry is not an integer, or an entry
    names a card outside ``enabled_cuda`` -- the UI then greys the steppers
    ('manual rank_gpu_id active') instead of guessing."""
    lst = _as_int_list(v)
    if not lst:
        return None
    legal = {int(c) for c in enabled_cuda}
    counts: Dict[int, int] = {}
    for g in lst:
        if g not in legal:
            return None
        counts[g] = counts.get(g, 0) + 1
    return counts


def _hardware_spec(gpus: Sequence):
    """A manual :class:`HardwareSpec` from the profile generator's GPU list
    (dicts or GpuDescriptor-likes), for the capacity sizing below."""
    from sglang.srt.planner.hardware import GpuDescriptor, HardwareSpec

    descs = []
    for i, g in enumerate(gpus):
        if isinstance(g, dict):
            idx = g.get("index", i)
            name = str(g.get("name") or f"GPU{i}")
            total = int(g.get("total_mib") or 0)
        else:
            idx = getattr(g, "index", i)
            name = str(getattr(g, "name", "") or f"GPU{i}")
            total = int(getattr(g, "total_mib", 0) or 0)
        descs.append(
            GpuDescriptor(index=int(idx), name=name, total_mib=total)
        )
    return HardwareSpec(gpus=tuple(descs), source="manual")


def _capacity_plan(model_path: str, hardware, **kwargs):
    """Monkeypatchable seam: run the planner's existing sizing path
    (``feasibility.plan`` -> the server's own ``PerfCostModel``) for one
    preset's flag set. NOT a new sizing model — the capacity numbers the
    presets use are exactly the ones the plan tab / CLI show."""
    from sglang.srt.planner.feasibility import plan
    from sglang.srt.planner.model import resolve_model_ref

    try:
        model_path = resolve_model_ref(model_path)
    except Exception:
        pass  # use the path as given; plan() fails loudly if unusable.
    return plan(model_path, hardware, with_advantage=False, **kwargs)


def _apply_capacity_rules(
    s: Dict[str, Any],
    info: List[str],
    model_cfg: Optional[dict],
    gpus: Sequence,
    stock_even: bool = False,
) -> None:
    """CAPACITY RULES for a generated preset — pick ``context_length`` and
    ``max_running_requests`` from the preset's actual KV capacity:

      1. context_length = min(model native max, what this preset's KV
         capacity actually fits) — never above the model's own limit.
      2. (KV dtype: always fp8 — chosen in ``_mk`` for every preset, with
         the sm86 e5m2 rule intact.)
      3. max_running_requests = the largest N such that EVERY session still
         gets the full context: N x context_length <= the CONCURRENCY-AWARE
         capacity at N. The GDN/mamba pool grows with N and SHRINKS KV, so
         each candidate N is re-planned (naive division would overshoot on
         hybrids). If not even one full-context session fits, context_length
         falls back to the one-session fit and N = 1.
      4. A preset that only fits by offloading MoE experts to host RAM
         (offload.status == 'ram_offload') pins context_length = the model's
         native max and N = 1, always.

    The numbers come from :func:`_capacity_plan` (``feasibility.plan`` ->
    the server's own cost model). A preset whose capacity computation fails
    keeps the form defaults and says so in ``info`` — profile generation
    never crashes on an unsizable model. ``stock_even`` passes an explicit
    even --rank-tp-ratio so the stock-TP presets are sized as the even split
    (not the fork's VRAM-proportional auto split).
    """
    model_path = s.get("model_path")
    if not model_path:
        info.append(
            "capacity rules skipped: no model path in the base settings, "
            "so the KV capacity cannot be sized -- context_length / "
            "max_running_requests keep the form defaults."
        )
        return
    model_max = model_max_context(model_cfg)
    try:
        hardware = _hardware_spec(gpus)
        tp = int(s.get("tp_size") or 1)
        budgets = s.get("rank_gpu_memory_mib")
        if budgets is not None and not isinstance(budgets, (list, tuple)):
            budgets = [int(budgets)] * tp
        ratio = s.get("rank_tp_ratio")
        if isinstance(ratio, (list, tuple)):
            ratio = list(ratio)
        elif stock_even and tp > 1:
            ratio = [1] * tp  # size the stock preset as the even split.
        else:
            # sentinels ('auto', 'auto-performance') -> None: plan() then
            # derives the VRAM-proportional auto split itself.
            ratio = None
        kwargs = dict(
            tp_size=tp,
            rank_gpu_id=(
                list(s["rank_gpu_id"]) if s.get("rank_gpu_id") else None
            ),
            rank_gpu_memory_mib=budgets,
            rank_tp_ratio=ratio,
            kv_cache_dtype=s.get("kv_cache_dtype") or "auto",
            speculative_algorithm=s.get("speculative_algorithm") or None,
            speculative_num_draft_tokens=(
                s.get("speculative_num_draft_tokens") or None
            ),
        )

        def _cap_at(n: int):
            r = _capacity_plan(
                model_path, hardware, max_running_requests=n, **kwargs
            )
            cap = getattr(r, "capacity", None)
            tokens = (
                float(getattr(cap, "max_context_tokens", 0.0) or 0.0)
                if cap is not None
                else 0.0
            )
            return r, tokens

        r1, cap1 = _cap_at(1)
    except Exception as e:
        info.append(
            "capacity rules skipped (the cost model cannot size this "
            f"preset): {e} -- context_length / max_running_requests keep "
            "the form defaults."
        )
        return

    offload = getattr(r1, "offload", None)
    if getattr(offload, "status", None) == "ram_offload":
        # Rule 4: expert-offload preset -> full native context, N = 1.
        s["max_running_requests"] = 1
        if model_max:
            s["context_length"] = int(model_max)
        off_gib = float(getattr(offload, "offloaded_gib", 0.0) or 0.0)
        info.append(
            "capacity rule (expert-offload): the model only fits by "
            f"offloading ~{off_gib:.1f} GiB of MoE experts to host RAM -> "
            "context_length "
            + (
                str(int(model_max))
                if model_max
                else "left unset (the server derives the model's native max)"
            )
            + " (= the model's own maximum), max_running_requests 1."
        )
        return

    if not getattr(r1, "fits", False) or cap1 <= 0:
        reasons = list(getattr(r1, "infeasible_reasons", []) or [])
        info.append(
            "capacity rules skipped: this preset does not fit (no KV "
            "capacity to distribute)"
            + ((": " + reasons[0]) if reasons else "")
            + " -- context_length / max_running_requests keep the form "
            "defaults."
        )
        return

    cap1_i = int(cap1)
    if model_max is not None and model_max <= cap1_i:
        # Rule 1, model-max branch: the capacity holds at least one session
        # at the model's own maximum context. Rule 3: binary-search the
        # largest N whose N x ctx still fits the capacity AT that N —
        # cap(n) - n*ctx is monotone-decreasing in n (the mamba/GDN pool
        # grows with n, so cap(n) never grows), so the search is sound.
        ctx = int(model_max)
        cap_n = cap1_i
        lo, hi = 1, max(cap1_i // ctx, 1)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            try:
                rm, capm = _cap_at(mid)
                ok = bool(getattr(rm, "fits", False)) and capm >= mid * ctx
            except Exception:
                ok = False
            if ok:
                lo = mid
                cap_n = int(capm)
            else:
                hi = mid - 1
        n = lo
        s["context_length"] = ctx
        s["max_running_requests"] = n
        info.append(
            f"capacity rule (model-max): KV capacity ~{cap1_i} tokens at "
            f"N=1 covers the model's native max ({ctx}) -> context_length "
            f"{ctx}; max_running_requests {n} = the largest N with "
            f"N x {ctx} <= the concurrency-aware capacity (~{cap_n} tokens "
            f"at N={n}; the mamba/GDN pool grows with N)."
        )
    else:
        # Rule 1 capacity-cap branch == rule 3's N=1 fallback: the capacity
        # is below the model's own max, so ONE full-capacity session is the
        # optimum (N=2 would need twice the one-session capacity, which a
        # non-growing cap(n) can never supply).
        s["context_length"] = cap1_i
        s["max_running_requests"] = 1
        info.append(
            f"capacity rule (capacity-cap): KV capacity ~{cap1_i} tokens "
            "at N=1 is below the model's native max ("
            + (str(model_max) if model_max is not None else "not declared")
            + f") -> context_length {cap1_i} (the one-session fit), "
            "max_running_requests 1."
        )


# ---------------------------------------------------------------------------
# Rig calibrations: measured, battery-stable per-quant values for known rigs
# (matrix/htsglang_local_run.sh, MATRIX_PLAN 3.3).
#
# INDEX SPACES: the vectors are per-RANK, i.e. in RANK space == CUDA
# enumeration order. The validated reference launches with --rank-gpu-id
# 0,1,2 and no CUDA_DEVICE_ORDER, so rank i = cuda:i, and CUDA's default
# FASTEST_FIRST puts the RTX 5090 at cuda:0 on the reference box -- rank 0
# IS the 5090. Hence fp8 tokvec 33,13,18 / reserve 3000,2200,2200: rank 0
# (the 5090, most free VRAM) owns the LARGEST KV share (33) and the biggest
# reserve (3000); ranks 1/2 are the two 3080s. The NVML/nvidia-smi inventory
# order ([3080, 5090, 3080] on that box) has NO bearing on rank order, which
# is why the gate below matches the rig by NAME MULTISET, order-insensitive:
# any listing order of exactly one 5090 + two 3080s is the same physical rig
# and gets the same rank-space vectors. Any other rig gets no calibration
# (the caller falls back to a derived estimate and says so) -- never remap
# measured per-rank vectors onto a rig we did not measure.
#
# The gate also checks the VRAM TOTALS, not only the model names (#434). A
# token vector and a reserve are budget-space quantities: they were solved
# against 32607 / 20480 / 20480 MiB. The reference rig's 3080s are the 20 GB
# variant, and the STOCK RTX 3080 is a 10 GB card (see
# ``card_library.SEED_CARDS``) -- a name-only gate hands a stock 5090 + 2x
# 3080 10 GB rig a KV split and a 3000 MiB reserve solved for cards with
# twice the memory, which is a boot-time OOM rather than a calibration.
# ---------------------------------------------------------------------------

#: NVML totals (MiB) of the calibrated rig, in the same NAME order the gate
#: counts: the 5090 first, then the two 20 GB 3080s.
_CALIBRATED_RIG_TOTALS_MIB = {"5090": 32607, "3080": 20480}

#: How far a card's NVML total may sit from the calibrated one and still count
#: as the same card. NVML totals of nominally identical cards differ by a few
#: MiB (ECC state, driver version, vBIOS), so an exact match would reject the
#: reference rig itself on a driver upgrade; a whole VRAM tier is far outside
#: this band (10240 vs 20480 is 100 % away).
_CALIBRATED_RIG_TOTAL_TOLERANCE = 0.05


def _match_calibration(gpus: Sequence, quant: Optional[str]) -> Optional[dict]:
    """A measured calibration for this exact rig shape + checkpoint quant, or
    None (callers must then fall back to a derived estimate AND say so).

    The gate is ORDER-INSENSITIVE (exact name multiset: one 5090 + two
    3080s): inventory listing order is NVML/detect order, while the vectors
    are rank-space (cuda order) -- see the block comment above. It is also
    TOTAL-SENSITIVE: a card whose NVML total is a different VRAM tier from
    the calibrated one is a different card for budget purposes, whatever its
    model name says."""
    names = [n.lower() for n in _gpu_names(gpus)]
    totals = _gpu_totals(gpus)

    def _totals_match() -> bool:
        if len(totals) != len(names):
            return False
        for name, total in zip(names, totals):
            for key, want in _CALIBRATED_RIG_TOTALS_MIB.items():
                if key in name:
                    if not total:
                        # No total to check against: the inventory is a manual
                        # or offline spec. Refuse rather than assume -- the
                        # whole point of the check is not to guess a budget.
                        return False
                    if abs(total - want) > want * _CALIBRATED_RIG_TOTAL_TOLERANCE:
                        return False
        return True

    if (
        len(names) == 3
        and sum(1 for n in names if "5090" in n) == 1
        and sum(1 for n in names if "3080" in n) == 2
        and _totals_match()
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


def _apply_crossover_knees(
    s: Dict[str, Any],
    info: List[str],
    tp: int,
    finding,
    prompt_to_output_ratio: Optional[float],
) -> None:
    """Fold this rig's measured MLP-split knee points into the max-perf
    preset.

    The knee (the prompt:output ratio at which an MLP concentration starts
    to pay) is a property of a machine — card mix, interconnect, model,
    quantisation lane — so NOTHING here is a built-in number: every figure
    comes from the rig-local crossover store (``crossover.load_finding``) and
    is emitted with its rig label and age. Three outcomes, mirroring the
    lever logic (levers._resolve_crossover):

      * a usable local finding + a workload ratio above a knee -> the
        measured vector is applied (``--rank-mlp-ratio`` directly, with
        ``--rank-tp-ratio auto``: the auto-performance optimizer's decode
        -knee guard rejects exactly the concentration a prompt-dominated
        workload justifies, so naming both would emit a command whose halves
        disagree);
      * a usable finding without a workload ratio (or one below every knee)
        -> the knee table is stated, nothing is applied;
      * no usable finding -> the preset says the knees are UNMEASURED and
        names the study that measures them. A finding from another rig or a
        stale one never selects anything.
    """
    from sglang.srt.planner.crossover import STUDY_TIERS

    if finding is None:
        info.append(
            "MLP-split knee points: NOT measured on this rig. Where "
            "concentration starts to pay depends on the card mix, model and "
            "quant lane, so no vector is proposed. Measure with the "
            "mlp_split_crossover study ("
            + "; ".join(
                f"{t.label} ~{t.est_runtime_min} min" for t in STUDY_TIERS
            )
            + ")."
        )
        return
    if not finding.usable_for_advice():
        caveats = finding.caveats()
        info.append(
            "MLP-split knee points: a finding exists but does not select a "
            "vector ("
            + (caveats[0] if caveats else "not usable for advice")
            + ") Run the mlp_split_crossover study on this rig for a usable "
            "one."
        )
        return
    rows = [
        r
        for r in finding.break_even_table()
        if r["proposable"] and r["break_even_prompt_to_output"] is not None
    ]
    if not rows:
        info.append(
            "MLP-split knee points measured on this rig ("
            + finding.rig.label()
            + "): no concentration vector is ever the best positive-net "
            "choice; the base split stands at every prompt:output mix."
        )
        return
    def _pct(v: Optional[float]) -> str:
        return "unmeasured" if v is None else f"{v:+.1f} %"

    info.append(
        "MLP-split knee points (measured on this rig: "
        + finding.rig.label()
        + f", {finding.age_s() / 86400.0:.0f} days ago): "
        + "; ".join(
            f"{r['vector']} turns at "
            f"{r['break_even_prompt_to_output']:.1f}:1 prompt:output "
            f"(prefill {_pct(r['prefill_gain_pct'])}, decode "
            f"{_pct(r['decode_cost_pct'])})"
            for r in rows
        )
        + "."
    )
    if prompt_to_output_ratio is None:
        info.append(
            "No workload prompt:output ratio given -- knee points stated, "
            "no concentration applied."
        )
        return
    best = finding.best_for_ratio(float(prompt_to_output_ratio))
    if best is None:
        info.append(
            f"At {float(prompt_to_output_ratio):.1f}:1 prompt:output every "
            "measured concentration is a net loss -- the base split IS the "
            "measured answer; nothing applied."
        )
        return
    if len(best.vector) != tp:
        info.append(
            f"The measured vector {best.label()} is for TP="
            f"{len(best.vector)}, this preset is TP={tp} -- not applicable, "
            "nothing applied."
        )
        return
    s["rank_tp_ratio"] = "auto"
    s["rank_mlp_ratio"] = list(best.vector)
    info.append(
        f"Applied --rank-mlp-ratio {best.label()} from the measured "
        f"crossover: at {float(prompt_to_output_ratio):.1f}:1 prompt:output "
        f"it nets {best.net_ms_per_output_token(float(prompt_to_output_ratio)):.3f} "
        f"ms per output token over the base split (turns at "
        f"{best.break_even_ratio:.1f}:1). Set with --rank-tp-ratio auto, "
        "not auto-performance: the optimizer's decode-knee guard would "
        "reject this measured, workload-justified concentration."
    )


def profiles(
    model_cfg: Optional[dict],
    gpus: Sequence,
    base: Optional[dict] = None,
    python_exe: Optional[str] = None,
    draft_models: Optional[Sequence[Dict[str, str]]] = None,
    crossover_finding=None,
    prompt_to_output_ratio: Optional[float] = None,
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
    speculative shape when the checkpoint has MTP layers. A checkpoint
    WITHOUT MTP layers gets the same conservative speculative shape via a
    matching local draft model when ``draft_models`` (find_draft_models
    output) is non-empty; otherwise spec stays off with an explicit note.

    Every preset additionally runs the CAPACITY RULES
    (:func:`_apply_capacity_rules`): context_length = min(model native max,
    what the preset's KV capacity actually fits), fp8 KV always,
    max_running_requests = the largest N whose N x context still fits the
    concurrency-aware capacity (expert-offload presets pin the native max +
    N=1). The numbers come from ``feasibility.plan`` and need
    ``base['model_path']``; without it (or when the cost model cannot size
    the preset) the form defaults are kept and ``info`` says so.

    ``crossover_finding`` is this rig's measured MLP-split crossover
    (``crossover.load_finding()``, or None) and ``prompt_to_output_ratio``
    the workload mix, both folded into the max-perf preset by
    ``_apply_crossover_knees``: the knee points are rig measurements, so the
    generator states them (or their absence + the study that measures them)
    and only ever applies a vector that a LOCAL, usable finding selects for
    the given mix. This function stays free of filesystem reads -- the
    caller loads the finding (webui.config_profiles_get does)."""
    totals = _gpu_totals(gpus)
    ngpu = len(totals)
    kvh = model_kv_heads(model_cfg)
    heterogeneous = len(set(totals)) > 1
    hybrid = model_is_hybrid(model_cfg)
    quant = model_quant_kind(model_cfg)
    sm86 = rig_has_sm86(gpus)
    cat = catalog()
    # first (best-ranked) matching local draft model, for checkpoints WITHOUT
    # their own MTP head (find_draft_models output; EAGLE3 candidates first).
    draft = dict(draft_models[0]) if draft_models else None

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
        # hicache on a hybrid/mamba model: only some host layouts are drivable.
        # The forced value is layer_first, NOT page_first_direct as it was:
        # page_first_direct's host write-back reaches
        # transfer_kv_all_layer_direct_lf_pf -> cudaMemcpyBatchAsync, which
        # segfaulted on three separate boots (#760), and ServerArgs now falls
        # back away from it on CUDA. A profile emitting the layout the server
        # then rewrites is a profile nobody can validate against the boot.
        if s.get("enable_hierarchical_cache") and hybrid:
            if s.get("hicache_mem_layout") not in _HYBRID_HICACHE_LAYOUTS:
                s["hicache_mem_layout"] = "layer_first"
                info.append(
                    "hicache on a hybrid/mamba model: forced "
                    "--hicache-mem-layout layer_first (page_first arms the "
                    "staging kernel MambaPoolHost cannot drive; page_head "
                    "takes the route #441a refuses)."
                )
            elif s.get("hicache_mem_layout") == "page_first_direct":
                s["hicache_mem_layout"] = "layer_first"
                info.append(
                    "hicache on a hybrid/mamba model: moved "
                    "--hicache-mem-layout page_first_direct -> layer_first; "
                    "its host write-back segfaults on CUDA (#760) and the "
                    "server gates it anyway, so the profile now states what "
                    "the boot will actually run."
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
        # Speculative decoding: EVERY generated preset ships the validated
        # adaptive-MTP shape (NEXTN chain, adaptive) whenever the checkpoint
        # actually carries MTP draft layers -- including the stock single-gpu
        # and normal-TP presets, not only the fork profiles. A checkpoint
        # without draft layers cannot run NEXTN; there a MATCHING local
        # draft model (find_draft_models) enables spec via
        # --speculative-draft-model-path instead, in the same conservative
        # shape. Without either, spec stays off with an explicit note.
        if not s.get("speculative_algorithm"):
            if model_has_mtp(model_cfg):
                s.update(
                    speculative_algorithm="NEXTN",
                    speculative_num_steps=3,
                    speculative_eagle_topk=1,
                    speculative_num_draft_tokens=4,
                    speculative_adaptive=True,
                )
                info.append(
                    "speculative decoding: NEXTN chain, steps 3, topk 1, "
                    "draft 4, adaptive (the validated shape; topk stays 1 "
                    "-- tree spec is rejected under uneven DCP)."
                )
            elif model_bundles_dspark_draft(model_cfg):
                # DeepSeek-V4-Flash-0731 ships a DSpark block draft under
                # mtp.* and NO NextN head, so neither the NEXTN branch above
                # nor a local-draft match applies. DSPARK is not auto-enabled:
                # it uses the block shape (num_steps 1, num_draft_tokens =
                # gamma + 1, see arg_groups/speculative_hook.py) rather than
                # the NEXTN chain this generator emits, and DSpark on
                # DeepSeek-V4 is not boot-proven in this fork yet (#447).
                info.append(
                    "checkpoint bundles a DSpark draft head (dspark_* config "
                    "keys) and has no NextN head: speculative decoding needs "
                    "--speculative-algorithm DSPARK plus the block shape "
                    "(--speculative-dspark-block-size), which this generator "
                    "does not emit -- spec stays off."
                )
            elif draft is not None:
                algo = str(draft.get("algorithm") or "EAGLE3")
                # adaptive spec is VERIFIED legal only for EAGLE/EAGLE3 (+
                # FROZEN_KV_MTP, and multi-layer EAGLE since #138):
                # adaptive_spec_params.adaptive_unsupported_reason rejects every
                # other algorithm, so a STANDALONE draft keeps adaptive off.
                # The planner never picks multi-layer EAGLE itself (it is
                # auto-enabled by the MiMoV2 / Step3p5 override registry), so
                # this branch stays EAGLE/EAGLE3-only.
                adaptive = algo in ("EAGLE", "EAGLE3")
                s.update(
                    speculative_algorithm=algo,
                    speculative_draft_model_path=draft.get("path"),
                    speculative_num_steps=3,
                    speculative_eagle_topk=1,
                    speculative_num_draft_tokens=4,
                    speculative_adaptive=adaptive,
                )
                info.append(
                    "no MTP head: speculative decoding via the local draft "
                    "model " + str(draft.get("name") or draft.get("path"))
                    + " (" + algo + ", steps 3, topk 1, draft 4, "
                    + ("adaptive" if adaptive else
                       "adaptive OFF -- adaptive spec supports only "
                       "EAGLE/EAGLE3")
                    + "; topk stays 1 -- tree spec is rejected under "
                    "uneven DCP)."
                )
            elif model_cfg:
                info.append(
                    "no MTP head and no matching local draft model found - "
                    "speculative decoding unavailable."
                )
        # KV dtype (capacity rule 2): fp8 KV ALWAYS, for EVERY preset (stock
        # single-gpu / normal-TP included) -- fp8 KV maximizes KV capacity.
        # sm86 (RTX 3080-class) cannot do fp8_e4m3 -> e5m2 whenever ANY sm86
        # card is in the rig; e4m3 only on an sm86-free rig.
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
        return s

    def _apply_fork_rules(s: Dict[str, Any], info: List[str]) -> None:
        """Rules for the fork (co-location / uneven) profiles: hybrid SSM
        dtype (KV dtype and spec/MTP are preset-wide, set in _mk)."""
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
    # GPU pick rule (user-mandated): largest VRAM wins; on a VRAM tie the
    # higher-FLOPs card wins; on a full tie the first (lowest index) wins.
    # The pick is a DEFAULT -- the UI offers a selector; base_gpu_id is the
    # stock upstream flag for pinning the card, so this preset stays stock.
    # base_gpu_id is a CUDA-ORDER index (it lands in CUDA_VISIBLE_DEVICES),
    # NOT the position in the NVML-ordered inventory list -- the pick emits
    # the chosen card's cuda_index (on the reference box the 5090 is list
    # position 1 but cuda:0).
    pick = _pick_single_gpu(gpus)
    info = list(base_notes) + ["Single GPU, no tensor parallelism."]
    overrides: Dict[str, Any] = {"tp_size": 1}
    if pick is not None:
        _pos, cuda_idx, why = pick
        # None = the card's CUDA order is unknown (#397): size against it,
        # but emit no pin rather than a guessed one.
        if cuda_idx is not None and cuda_idx != 0:
            overrides["base_gpu_id"] = cuda_idx
        info.append(why)
    s = _mk(overrides, info)
    # Size against the CHOSEN card only (by list position), not gpus[0].
    cap_gpus = [gpus[pick[0]]] if pick is not None else gpus
    _apply_capacity_rules(s, info, model_cfg, cap_gpus, stock_even=True)
    out.append(Profile("single-gpu", "single", s, info=info))

    if ngpu <= 1:
        return out

    # normal (stock) TP is representable only under the verified stock rule
    # (q % tp == 0 and kv % tp == 0 / tp % kv == 0 -- head-shard or
    # kv-replication) and with one rank per card.
    stock_tp_ok, stock_tp_reason = stock_tp_legal(model_cfg, ngpu)
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
        _apply_capacity_rules(s, info, model_cfg, gpus, stock_even=True)
        out.append(Profile("normal-TP", "normal-tp", s, info=info))
    else:
        # NEUTRAL framing (no scare wording): state plain facts about what
        # stock's rule allows here; the fork profiles below are alternatives
        # with their own numbers, not a verdict on this one.
        reason = []
        if heterogeneous:
            reason.append(
                "the cards have different VRAM totals, so the stock even "
                "split is sized by the smallest card"
            )
        if not stock_tp_ok:
            reason.append(stock_tp_reason)
        info = list(base_notes) + [
            f"Plain even tensor parallelism across all {ngpu} cards, stock "
            "sglang options only. " + "; ".join(reason) + ".",
        ]
        s = _mk({"tp_size": ngpu}, info)
        _apply_capacity_rules(s, info, model_cfg, gpus, stock_even=True)
        out.append(
            Profile("normal-TP (stock, limited)", "normal-tp", s, info=info)
        )

    # --- stock normal-TP subset presets: 2 cards (and 4 when 4 exist) -----
    # Sensible plain-TP options on a bigger/mixed rig: tp=2 on the best pair
    # and tp=4 on the best quad -- ONLY for card counts the rig actually has
    # (never fabricate a card) and ONLY for tp values stock can express
    # (stock_tp_legal); a non-divisible tp is not emitted at all (the fork
    # profiles cover that space).
    for n_sub, kind_sub in ((2, "normal-tp-2"), (4, "normal-tp-4")):
        if ngpu <= 2 or ngpu < n_sub:
            continue  # subset must be a real subset of existing cards
        if n_sub == ngpu and stock_tp_ok and not heterogeneous:
            continue  # identical to the plain normal-TP preset above
        legal, why_not = stock_tp_legal(model_cfg, n_sub)
        if not legal:
            continue  # falls to the fork profiles; never emit illegal stock
        pick = _pick_stock_subset(gpus, totals, n_sub)
        if pick is None:
            continue
        positions, cuda_idx, why = pick
        info = list(base_notes) + [
            f"Plain stock tensor parallelism, tp={n_sub}, on "
            f"{n_sub} of the {ngpu} cards: "
            + ", ".join(
                f"{_gpu_names(gpus)[p]} (cuda:{c})"
                for p, c in zip(positions, cuda_idx)
            )
            + f". {why} Stock sglang options only (no fork flags).",
        ]
        overrides_sub: Dict[str, Any] = {"tp_size": n_sub}
        env_sub: Dict[str, str] = {}
        lo = min(cuda_idx)
        if sorted(cuda_idx) == list(range(lo, lo + n_sub)):
            # contiguous in CUDA order: the stock --base-gpu-id pin suffices.
            if lo != 0:
                overrides_sub["base_gpu_id"] = lo
        else:
            # non-contiguous pick: restrict visibility instead (stock-
            # compatible; base_gpu_id stays 0 inside the restricted set).
            env_sub["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(c) for c in sorted(cuda_idx)
            )
            info.append(
                "cards are not contiguous in CUDA order: pinned via "
                "CUDA_VISIBLE_DEVICES="
                + env_sub["CUDA_VISIBLE_DEVICES"] + "."
            )
        s = _mk(overrides_sub, info)
        cap_gpus_sub = [gpus[p] for p in positions]
        _apply_capacity_rules(
            s, info, model_cfg, cap_gpus_sub, stock_even=True
        )
        out.append(
            Profile(
                f"normal-TP-{n_sub} (stock subset)",
                kind_sub,
                s,
                info=info,
                env=env_sub,
            )
        )

    if not needs_fork:
        # A homogeneous rig whose card count divides the KV heads: stock even
        # TP is the right answer; no fork-only profiles are needed.
        return out

    # --- co-location / TP > card-count (MPS required) ---------------------
    # Add ranks beyond the physical card count OR when tp>kv-heads is the point:
    # place two ranks on the largest card. rank_gpu_id entries are CUDA-ORDER
    # indices (the space CUDA_VISIBLE_DEVICES uses): list(range(ngpu)) covers
    # every card regardless of inventory order, and the duplicated entry must
    # be the largest card's cuda_index, NOT its position in the NVML-ordered
    # inventory list (on the reference box the 5090 is list position 1 but
    # cuda:0).
    largest_pos = totals.index(max(totals)) if totals else 0
    try:
        largest_cuda, largest_cuda_src = _cuda_index_at(gpus, largest_pos)
    except DeviceOrderUnresolvedError as e:
        # The preset IS a --rank-gpu-id vector; without the shared card's
        # CUDA index there is nothing correct to emit (#397).
        logger.warning("co-location preset omitted: %s", e)
        largest_cuda = largest_cuda_src = None
    if largest_cuda is not None:
        colo_tp = ngpu + 1
        # One rank per card + the extra rank on the largest card (cuda-space
        # ids), emitted in the CANONICAL counts order (rank_gpu_id_from_counts:
        # ascending cuda index, each index repeated its count) -- exactly what
        # the Runner's per-card rank steppers derive, so applying this preset
        # and then touching a stepper round-trips without rewriting the flag.
        colo_counts = {c: 1 for c in range(ngpu)}
        colo_counts[largest_cuda] = colo_counts.get(largest_cuda, 0) + 1
        rank_gpu = rank_gpu_id_from_counts(colo_counts)
        colo_ranks = [i for i, g in enumerate(rank_gpu) if g == largest_cuda]
        # even budget: the scalar must fit BOTH the smallest solo card (minus a
        # context allowance) and half of the shared largest card (minus a shared
        # 2 GiB allowance) -- otherwise the co-located pair is physically
        # impossible on the big card.
        per_rank = max(
            1024, min(min(totals) - 1024, (max(totals) - 2048) // 2)
        )
        info = list(base_notes) + [
            f"tp_size={colo_tp} on {ngpu} cards: ranks "
            + "+".join(str(r) for r in colo_ranks)
            + " co-locate on the largest card "
            f"({_gpu_names(gpus)[largest_pos]}, cuda:{largest_cuda}). "
            "REQUIRES CUDA MPS (two ranks share one card). "
            f"Uniform per-rank budget {per_rank} MiB; ensure "
            f"2 x {per_rank} MiB fits that card's total "
            f"({max(totals)} MiB) -- headroom is the user's responsibility."
            + (
                " (cuda indices are the DECLARED order of this offline "
                "inventory -- verify against nvidia-smi before launching)"
                if largest_cuda_src == "declared"
                else ""
            ),
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
        _apply_capacity_rules(s, info, model_cfg, gpus)
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
    # rank_gpu_id list(range(ngpu)) is CUDA-space identity: rank i on cuda:i,
    # covering every card whatever the NVML inventory order (the validated
    # reference command's 0,1,2 shape).
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
    _apply_capacity_rules(s, info, model_cfg, gpus)
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
    _apply_crossover_knees(
        s, info, ngpu, crossover_finding, prompt_to_output_ratio
    )
    _apply_capacity_rules(s, info, model_cfg, gpus)
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


def _live_identity_map():
    """The canonical UUID <-> ordinal resolver, or ``None`` without NVML.

    ``allow_cuda_init`` because this is the one consumer that genuinely needs
    the CUDA side: ``--rank-gpu-id`` is expressed in CUDA ordinals, so a saved
    profile cannot be checked against the hardware without asking torch which
    ordinal each card is now. The planner is a desk process that boots no
    model; the cost is one CUDA context, and the alternative is relaunching a
    server onto the wrong cards.
    """
    from sglang.srt.registry import nvml

    if not nvml.is_available():
        return None
    return nvml.identity_map(allow_cuda_init=True)


#: Where the physical identity of a profile's ``--rank-gpu-id`` cards is kept.
#: Not a catalog flag and never passed to the server: it exists so a saved
#: profile can be checked against the hardware it was written for.
CARD_UUIDS_KEY = "_card_uuids"

#: The settings that name physical cards as a per-rank list. Their entries are
#: CUDA ordinals -- ``server_args`` hands ``--rank-gpu-id`` straight to
#: ``torch.cuda`` -- so they migrate under ``order="cuda"``, not NVML order.
#: Duplicates are meaningful (two ranks co-located on one card) and survive
#: the round trip as duplicate UUIDs. The scalar ``base_gpu_id`` is not here:
#: it is an offset into whatever the process can see, not a card reference.
_CARD_INDEX_FLAGS = ("rank_gpu_id",)


class ProfileStore:
    """A tiny JSON-backed store of user-saved :class:`Profile` objects, keyed
    by name. No server, no locking beyond an atomic replace.

    A saved profile can name physical cards (``--rank-gpu-id`` is one CUDA
    ordinal per rank), and it is read back on a later boot -- so the ordinal
    it stored is exactly the kind of reference AUDIT #331 removes: CUDA
    enumeration is ``FASTEST_FIRST`` and shifts with driver state, so
    ``0,1,1,2`` written in June can name a different card layout in August.
    On save the store therefore resolves those ordinals to NVML UUIDs and
    keeps them beside the settings; on load it resolves the UUIDs back to the
    ordinals valid *now*, rewrites the profile when they moved, and raises a
    named error when a card is simply gone. A profile saved before #331 has
    no UUIDs, is loaded unchanged with a warning, and is stamped on its next
    save.
    """

    def __init__(self, path: str = DEFAULT_STORE_PATH, identity=None):
        self.path = os.path.expanduser(path)
        #: Injectable for tests; returns an ``IdentityMap`` or ``None``.
        self.identity = identity or _live_identity_map

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
        record = profile.to_json()
        uuids = self._stamp_card_uuids(record.get("settings") or {})
        if uuids is not None:
            record[CARD_UUIDS_KEY] = uuids
        by_name[profile.name] = record
        self._write(by_name)

    def load(self, name: str) -> Profile:
        by_name = self._read()
        if name not in by_name:
            raise KeyError(f"unknown profile {name!r}")
        record, migrated = self._resolve_card_uuids(by_name[name])
        if migrated:
            by_name[name] = record
            self._write(by_name)
        return Profile.from_json(record)

    def list(self) -> List[str]:
        return sorted(self._read().keys())

    def load_all(self) -> List[Profile]:
        by_name = self._read()
        out: List[Profile] = []
        migrated_any = False
        for name, record in list(by_name.items()):
            resolved, migrated = self._resolve_card_uuids(record)
            by_name[name] = resolved
            migrated_any = migrated_any or migrated
            out.append(Profile.from_json(resolved))
        if migrated_any:
            self._write(by_name)
        return out

    # -- card identity (AUDIT #331) ----------------------------------------

    def _stamp_card_uuids(self, settings: Dict[str, Any]) -> Optional[Dict[str, list]]:
        """``{flag: [uuid, ...]}`` for every card-naming flag that is set."""
        wanted = {
            flag: settings[flag]
            for flag in _CARD_INDEX_FLAGS
            if isinstance(settings.get(flag), (list, tuple)) and settings[flag]
        }
        if not wanted:
            return None
        imap = self._identity_map()
        if imap is None:
            return None
        out: Dict[str, list] = {}
        for flag, ordinals in wanted.items():
            try:
                out[flag] = imap.adopt_legacy_indices(
                    [int(v) for v in ordinals], order="cuda"
                )
            except Exception as exc:  # noqa: BLE001 - saving must not fail
                logger.warning(
                    "planner profiles: could not resolve %s=%s to card "
                    "identities (%s); the profile is saved index-only and "
                    "will not survive a re-enumeration (#331).",
                    flag,
                    ordinals,
                    exc,
                )
        return out or None

    def _resolve_card_uuids(self, record: dict) -> Tuple[dict, bool]:
        """Rewrite a record's card ordinals from its stored UUIDs.

        Returns ``(record, migrated)``; ``migrated`` is True when the ordinals
        on disk no longer matched the live enumeration and the file needs
        rewriting.
        """
        settings = record.get("settings") or {}
        if not any(settings.get(flag) for flag in _CARD_INDEX_FLAGS):
            return record, False
        stored = record.get(CARD_UUIDS_KEY)
        if not isinstance(stored, dict) or not stored:
            logger.warning(
                "planner profile %r names cards by index but predates card "
                "identity stamping (#331); its indices are used as written "
                "and cannot be checked against this host's cards.",
                record.get("name"),
            )
            return record, False
        imap = self._identity_map()
        if imap is None:
            return record, False
        out = dict(record)
        out["settings"] = dict(settings)
        migrated = False
        for flag, uuids in stored.items():
            if not isinstance(uuids, list) or not uuids:
                continue
            ordinals = [imap.cuda_ordinal_of(u) for u in uuids]
            if [int(v) for v in (settings.get(flag) or [])] != ordinals:
                logger.warning(
                    "planner profile %r: %s was saved as %s but those cards "
                    "are CUDA ordinals %s now; rewriting the profile from the "
                    "stored card identities (#331).",
                    record.get("name"),
                    flag,
                    settings.get(flag),
                    ordinals,
                )
                out["settings"][flag] = ordinals
                migrated = True
        return out, migrated

    def _identity_map(self):
        try:
            return self.identity()
        except Exception as exc:  # noqa: BLE001 - a desk host has no NVML
            logger.debug("planner profiles: identity map unavailable (%s)", exc)
            return None

    def delete(self, name: str) -> bool:
        by_name = self._read()
        if name not in by_name:
            return False
        del by_name[name]
        self._write(by_name)
        return True


# ---------------------------------------------------------------------------
# Usability parsers (#531): every generated boot command carries the
# model-family reasoning + tool-call parser.
#
# User standing order 2026-08-03: --reasoning-parser and --tool-call-parser are
# STANDARD boot settings, not tuning knobs -- so that the models can actually
# be USED (the user's wording, translated). Without them a reasoning
# checkpoint's chain-of-thought lands in
# `content` as raw `</think>` text and a tool call arrives as a JSON-looking
# STRING instead of a structured `tool_calls` entry, so every agentic client
# silently degrades while the server reports HTTP 200.
#
# The mapping is keyed on identity TOKENS of the architecture/name, and the
# emitted names are CHECKED against the live parser registries (see
# `validate_usability_parsers`) rather than trusted -- a registry rename must
# turn this red, not ship a boot command with a flag value the server rejects.
# ---------------------------------------------------------------------------

#: (identity tokens that must ALL appear, reasoning parser, tool-call parser).
#: Ordered: the first match wins, so more specific rows come first.
_USABILITY_PARSER_TABLE: Tuple[Tuple[Tuple[str, ...], Optional[str], Optional[str]], ...] = (
    # DeepSeek: the V4 family has its own detector pair; V3.x shares deepseek-v3
    # reasoning with a point-release tool-call parser.
    (("deepseek", "v4"), "deepseek-v4", "deepseekv4"),
    (("deepseek", "v32"), "deepseek-v3", "deepseekv32"),
    (("deepseek", "v31"), "deepseek-v3", "deepseekv31"),
    (("deepseek", "v3"), "deepseek-v3", "deepseekv3"),
    (("deepseek", "r1"), "deepseek-r1", "deepseekv3"),
    # Qwen3.x (this rig's default family). Token stays undotted-joined because
    # _name_tokens does not split on '.', so "qwen3.6" contains "qwen3".
    (("qwen3",), "qwen3", "qwen3_coder"),
    (("qwen2",), None, "qwen25"),
    # Others with a registered pair.
    (("glm", "47"), "glm45", "glm47"),
    (("glm", "45"), "glm45", "glm45"),
    (("gpt", "oss"), "gpt-oss", "gpt-oss"),
    (("kimi", "k2"), "kimi_k2", "kimi_k2"),
    (("minimax", "m3"), "minimax-m3", "minimax-m3"),
    # Written "Step-3.5" on disk, which squashes to "step35"; the registered
    # parser names spell it "step3p5".
    (("step35",), "step3p5", "step3p5"),
    (("gemma4",), "gemma4", "gemma4"),
    (("hunyuan",), "hunyuan", "hunyuan"),
    (("mistral",), "mistral", "mistral"),
    (("llama3",), None, "llama3"),
)


def _usability_identity(model_path: str, model_cfg: Optional[dict] = None) -> str:
    """One SQUASHED lowercase identity string for parser selection: the
    checkpoint's ``architectures`` (authoritative -- a renamed directory cannot
    fool it) concatenated with the path basename, with every separator AND dot
    removed.

    Squashing rather than tokenising is deliberate: the family marker is
    written five different ways across real checkpoints -- ``Qwen3.6-27B``,
    ``qwen3_5``, ``Qwen3_5ForConditionalGeneration``, ``DeepSeek-V4-Flash``,
    ``glm-4.7`` -- and only a separator-free form makes ``qwen3``, ``v4`` and
    ``glm47`` all findable by one rule. Table rows are matched as SUBSTRINGS of
    this string, which is why the table is ordered specific-first (``v32``
    before ``v3``): ``v3`` is a substring of ``v32``.
    """
    raw: List[str] = []
    archs = _cfg_get(model_cfg, "architectures")
    if isinstance(archs, list):
        raw.extend(str(a) for a in archs)
    elif isinstance(archs, str):
        raw.append(archs)
    if not raw:
        raw.append(_draft_config_arch(str(model_path or "")))
    raw.append(os.path.basename(str(model_path or "").rstrip("/")))
    joined = " ".join(chunk for chunk in raw if chunk)
    return re.sub(r"[^a-z0-9]+", "", joined.lower())


def usability_parsers(
    model_path: str, model_cfg: Optional[dict] = None
) -> Tuple[Optional[str], Optional[str], str]:
    """Resolve ``(reasoning_parser, tool_call_parser, note)`` for a checkpoint.

    An unrecognised family returns ``(None, None, <named hint>)`` -- the
    caller must surface the hint rather than emit a bare command, because a
    silently parser-less boot is exactly the failure this exists to prevent.
    """
    identity = _usability_identity(model_path, model_cfg)
    for needed, reasoning, toolcall in _USABILITY_PARSER_TABLE:
        if all(n in identity for n in needed):
            note = f"parser family resolved from {'+'.join(needed)}"
            if reasoning is None:
                note += " (no reasoning parser registered for this family)"
            return reasoning, toolcall, note
    return (
        None,
        None,
        "UNKNOWN MODEL FAMILY for --reasoning-parser/--tool-call-parser: "
        f"{os.path.basename(str(model_path or '')) or model_path!r} matched no "
        "entry in the parser table. Pick the parsers by hand from the "
        "registered names (sglang.srt.parser.reasoning_parser.ReasoningParser."
        "DetectorMap and sglang.srt.function_call.function_call_parser."
        "FunctionCallParser.ToolCallParserEnum) and add the family here. "
        "Booting WITHOUT them serves the model with its chain-of-thought as "
        "raw text in `content` and tool calls as unparsed strings.",
    )


def usability_argv(
    model_path: str, model_cfg: Optional[dict] = None
) -> Tuple[List[str], str]:
    """``usability_parsers`` rendered as argv plus the note to print."""
    reasoning, toolcall, note = usability_parsers(model_path, model_cfg)
    argv: List[str] = []
    if reasoning:
        argv += ["--reasoning-parser", reasoning]
    if toolcall:
        argv += ["--tool-call-parser", toolcall]
    return argv, note


def validate_usability_parsers() -> List[str]:
    """Every name in the table must exist in the live registries. Returns the
    list of problems (empty = consistent). Import-guarded: a registry that
    cannot be imported yields no verdict rather than a false failure."""
    try:
        from sglang.srt.function_call.function_call_parser import FunctionCallParser
        from sglang.srt.parser.reasoning_parser import ReasoningParser
    except Exception as exc:  # noqa: BLE001 - absence is not a mapping error
        return [f"registries unavailable, mapping unchecked: {exc}"]
    known_reasoning = set(ReasoningParser.DetectorMap)
    known_toolcall = set(FunctionCallParser.ToolCallParserEnum)
    problems: List[str] = []
    for needed, reasoning, toolcall in _USABILITY_PARSER_TABLE:
        key = "+".join(needed)
        if reasoning is not None and reasoning not in known_reasoning:
            problems.append(f"{key}: reasoning parser {reasoning!r} is not registered")
        if toolcall is not None and toolcall not in known_toolcall:
            problems.append(f"{key}: tool-call parser {toolcall!r} is not registered")
    return problems
