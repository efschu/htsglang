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
"""Thin web UI for the offline config planner (design #97 stage S3, §2.6/§6).

A self-contained stdlib HTTP server (same no-CDN style as
``tools/rig_dashboard/server.py``) that serves one HTML page + a narrow
planner API. The page is a THIN CLIENT: every edit re-POSTs the whole form
to ``/api/plan``, which runs the SAME ``feasibility.plan()`` the CLI runs, so
the UI never diverges from the engine. No GPU, no server boot — pure offline
math.

Endpoints:
  GET  /                -> the single HTML page (inline CSS/JS, no CDN)
  GET  /api/knobs       -> the editable-knob list, FEATURE-DETECTED from the
                           deployed ServerArgs schema (§2.6†: never offer a
                           knob the running build cannot honor)
  POST /api/plan        -> {model, hardware, overrides} -> fit / split /
                           per-card VRAM / capacity-% / launch flags / honest
                           advantage
  POST /api/issue       -> {kind: results|bug, ...plan...} -> S2 issue text +
                           prefilled GitHub URL

HONESTY (design §3.4): the API returns only what the planner returns —
capacity / feasibility / split. There is NO estimated-tok/s field; measured
scores appear only when a cached hardware profile exists (surfaced by S1's
advantage module). The HTML has no widget that renders an estimated rate.
"""

from __future__ import annotations

import dataclasses
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional

__all__ = [
    "discover_knobs",
    "detect_hardware",
    "gguf_options_for",
    "plan_from_payload",
    "issue_from_payload",
    "matrix_from_payload",
    "landscape_from_payload",
    "list_profiles",
    "hicache_saved_read",
    "hicache_saved_record",
    "serve",
]


# ===========================================================================
# Hardware auto-detect + GGUF quant picker (design §PART 2 / §PART 1a).
# ===========================================================================


def detect_hardware() -> dict:
    """Auto-detect the local GPUs (NVML/nvidia-smi) for the selectable card
    list. Returns ``{ok, gpus:[{index,name,total_mib,free_mib}], host_ram_mib,
    source}`` or ``{ok:False, error}`` on a GPU-less host — the UI then shows
    only the manual/virtual add-card path (still fully usable offline)."""
    from sglang.srt.planner import hardware as hwmod

    try:
        spec = hwmod.hardware_from_nvml()
    except hwmod.HardwareUnavailable as e:
        return {
            "ok": False,
            "error": str(e),
            "gpus": [],
            "host_ram_mib": hwmod._host_ram_mib(),
        }
    return {
        "ok": True,
        "source": spec.source,
        "host_ram_mib": spec.host_ram_mib,
        "gpus": [
            {
                "index": g.index,
                "name": g.name,
                "total_mib": g.total_mib,
                "free_mib": g.free_mib,
            }
            for g in spec.gpus
        ],
    }


def gguf_options_for(payload: dict) -> dict:
    """List the selectable ``.gguf`` checkpoints for the quant dropdown, for a
    LOCAL multi-quant export directory OR an HF-hub GGUF repo id (metadata-only
    file listing, no weight download). Empty list for a single-file /
    safetensors model."""
    from sglang.srt.planner.model import list_gguf_options_any

    ref = (payload or {}).get("model", "").strip()
    if not ref:
        return {"ok": True, "options": []}
    try:
        return {"ok": True, "options": list_gguf_options_any(ref)}
    except Exception as e:  # pragma: no cover - defensive
        return {"ok": False, "error": str(e), "options": []}


# ===========================================================================
# Feature-detection of the editable knobs (design §2.6 / §2.6†).
# ===========================================================================

#: The curated knob catalog: every entry maps to a REAL server-arg with a
#: real granularity (design §2.6 table). Each is emitted ONLY when the
#: deployed ServerArgs actually declares its field — so the UI never offers a
#: degree of freedom the running build cannot honor (rank_kv_ratio, the
#: weightless-KV flags and the offload env live on other branches and are
#: absent here; grep confirms). This IS the single-source-of-truth
#: discipline of §5: the knob list is the arg schema.
_KNOB_CATALOG = [
    {
        "id": "plan_free_reserve_gb",
        "server_arg": None,  # planner-only external headroom (§2.5)
        "label": "Per-card free VRAM to leave (GiB)",
        "kind": "per_card_float",
        "help": "External headroom carved out per physical card (a display, "
        "another process). Every capacity number becomes 'max under your "
        "budget'. Maps to --rank-auto-reserve-mib / mem-fraction at boot.",
        "always": True,
    },
    {
        "id": "rank_gpu_id",
        "server_arg": "rank_gpu_id",
        "label": "rank -> physical GPU map",
        "kind": "int_list",
        "help": "One physical GPU id per rank; duplicates co-locate ranks on "
        "one card (e.g. 0,0,1,2). Sum of co-located budgets must fit the card.",
    },
    {
        "id": "rank_gpu_memory_mib",
        "server_arg": "rank_gpu_memory_mib",
        "label": "Per-rank VRAM budget (MiB)",
        "kind": "int_list_or_scalar",
        "help": "Absolute MiB per rank (a per-rank list needs a tp ratio; a "
        "scalar applies to all). Unset = auto-derived from totals - reserves.",
    },
    {
        "id": "rank_tp_ratio",
        "server_arg": "rank_tp_ratio",
        "label": "Overall TP shard ratio",
        "kind": "int_list",
        "help": "Positive integer weight per rank; sum must divide every "
        "sharded dim; attention splits in whole KV-head units.",
    },
    {
        "id": "rank_mlp_ratio",
        "server_arg": "rank_mlp_ratio",
        "label": "Dense-MLP (FFN) split",
        "kind": "int_list",
        "help": "Integer weights per rank on the MLP unit grid "
        "(intermediate/group_size), not arbitrary widths.",
    },
    {
        "id": "rank_moe_ratio",
        "server_arg": "rank_moe_ratio",
        "label": "MoE expert-intermediate split",
        "kind": "int_list",
        "help": "Integer weights per rank; whole experts per rank.",
    },
    {
        "id": "rank_vocab_ratio",
        "server_arg": "rank_vocab_ratio",
        "label": "Vocab / lm_head split",
        "kind": "int_list",
        "help": "Integer weights per rank on 64-row padded vocab units.",
    },
    {
        "id": "dcp_size",
        "server_arg": "dcp_size",
        "label": "DCP degree",
        "kind": "int",
        "help": "1..tp_size; auto-set to tp_size under uneven-DCP token "
        "sharding.",
    },
    {
        "id": "kv_token_vector",
        "server_arg": None,  # env SGLANG_UNEVEN_TOKEN_VECTOR
        "label": "KV-token split across ranks",
        "kind": "int_list",
        "env": "SGLANG_UNEVEN_TOKEN_VECTOR",
        "help": "Integer token-vector on page_size x sum(ratios) virtual "
        "blocks; owner-rule granularity, not per-token.",
    },
]

#: Knobs deliberately NOT offered (design §2.6 "NOT expressible"): free-canvas
#: per-layer scatter, non-proportional/non-contiguous expert or head sets,
#: off-grid shard widths. Surfaced to the UI as an explanatory note so the
#: absence is intentional and explained, never a silent omission.
_NOT_EXPRESSIBLE = [
    "Arbitrary per-layer placement (the runtime shards every layer the same "
    "way by proportional TP + token-axis KV — not per-layer scatter).",
    "Non-proportional / non-contiguous expert or head assignment (experts and "
    "heads go to ranks in contiguous, whole-unit, ratio-proportional blocks).",
    "Off-grid shard widths (a ratio whose sum doesn't divide a sharded dim, or "
    "an FFN width off the MLP unit grid — rejected with the reason).",
]


def discover_knobs() -> dict:
    """Return the editable-knob list actually honorable by the deployed build,
    by intersecting the curated catalog with the live ServerArgs fields
    (§2.6†). Also reports the env-var knobs and the not-expressible notes."""
    from sglang.srt.server_args import ServerArgs

    fields = {f.name for f in dataclasses.fields(ServerArgs)}
    knobs = []
    for spec in _KNOB_CATALOG:
        arg = spec.get("server_arg")
        detected = arg is None or arg in fields
        if not detected and not spec.get("always"):
            continue
        knobs.append(
            {
                "id": spec["id"],
                "label": spec["label"],
                "kind": spec["kind"],
                "help": spec["help"],
                "server_arg": arg,
                "env": spec.get("env"),
                "source": "planner" if arg is None else "server_args-field",
            }
        )
    return {
        "knobs": knobs,
        "not_expressible": _NOT_EXPRESSIBLE,
        "detected_fields": sorted(
            f for f in fields if f.startswith("rank_") or f == "dcp_size"
        ),
    }


# ===========================================================================
# Payload -> planner call -> JSON (the thin-client contract).
# ===========================================================================


def _int_list(v):
    if v is None or v == "":
        return None
    if isinstance(v, list):
        return [int(x) for x in v]
    return [int(x) for x in str(v).replace(" ", "").split(",") if x != ""]


def _float_list(v):
    if v is None or v == "":
        return None
    if isinstance(v, list):
        return [float(x) for x in v]
    return [float(x) for x in str(v).replace(" ", "").split(",") if x != ""]


def _cards_hardware_and_reserve(hw):
    """Build a HardwareSpec + per-card free-reserve (MiB) from the structured
    card list the UI posts (design §PART 2/§PART 3). Each entry:
    ``{name, total_mib, include, reserve_gb, virtual}``. Only INCLUDED cards
    are planned; they are re-indexed 0..k-1 (physical ids the --rank-gpu-id map
    refers to). ``virtual`` cards (hypothetical/future GPUs the user typed) are
    treated identically — the whole point of an offline planner. Returns
    ``(HardwareSpec, reserve_mib_list_or_None)``."""
    from sglang.srt.planner import hardware as hwmod

    cards = [c for c in hw.get("cards", []) if c.get("include", True)]
    if not cards:
        raise ValueError(
            "No cards selected — tick at least one detected GPU or add a "
            "card (real or hypothetical) to plan against."
        )
    gpus = []
    reserve_mib = []
    any_reserve = False
    for i, c in enumerate(cards):
        total = int(c["total_mib"])
        if total <= 0:
            raise ValueError(f"card {c.get('name', i)!r}: VRAM must be positive.")
        gpus.append(
            hwmod.GpuDescriptor(
                index=i,
                name=str(c.get("name") or f"gpu{i}"),
                total_mib=total,
                # Plan against the declared total (offline semantics); a
                # detected free_mib is not carried so virtual/future cards and
                # real ones are handled identically.
            )
        )
        r = float(c.get("reserve_gb") or 0)
        if r < 0:
            raise ValueError(f"card {c.get('name', i)!r}: reserve must be >= 0.")
        if r > 0:
            any_reserve = True
        reserve_mib.append(int(r * 1024))
    host_ram = hw.get("host_ram_mib")
    spec = hwmod.HardwareSpec(
        gpus=tuple(gpus),
        source="manual",
        host_ram_mib=int(host_ram) if host_ram else hwmod._host_ram_mib(),
    )
    return spec, (reserve_mib if any_reserve else None)


def _hardware_and_reserve(payload):
    """Resolve (HardwareSpec, per-card free-reserve MiB) from a plan payload,
    supporting BOTH the new structured card list (source="cards", per-card
    reserve — design §PART 2/3) and the legacy modes (manual "NAME:MIB" list /
    nvml / json_inline, with the top-level ``plan_free_reserve_gb`` knob). The
    legacy path is untouched so existing CLI/test payloads keep working."""
    hw = payload.get("hardware", {}) or {}
    if hw.get("source") == "cards":
        return _cards_hardware_and_reserve(hw)
    hardware = _load_hardware_from_payload(hw)
    reserve = _float_list(payload.get("plan_free_reserve_gb"))
    reserve_mib = None
    if reserve is not None:
        if len(reserve) == 1:
            reserve = reserve * len(hardware.gpus)
        reserve_mib = [int(v * 1024) for v in reserve]
    return hardware, reserve_mib


def _load_hardware_from_payload(hw):
    from sglang.srt.planner import hardware as hwmod

    source = (hw or {}).get("source", "manual")
    if source == "cards":
        return _cards_hardware_and_reserve(hw)[0]
    if source == "nvml":
        return hwmod.hardware_from_nvml()
    if source == "json_inline":
        # A composed/declared inventory posted inline.
        gpus = hw["gpus"]
        spec_gpus = tuple(
            hwmod.GpuDescriptor(
                index=int(g.get("index", i)),
                name=str(g.get("name", f"gpu{i}")),
                total_mib=int(g["total_mib"]),
                free_mib=g.get("free_mib"),
                pcie_gen=g.get("pcie_gen"),
                pcie_width=g.get("pcie_width"),
            )
            for i, g in enumerate(gpus)
        )
        return hwmod.HardwareSpec(gpus=spec_gpus, source="manual")
    # default: manual "NAME:MIB" list
    items = hw.get("gpus", [])
    return hwmod.hardware_from_manual(items)


def _plan_to_dict(result, model_path: str) -> dict:
    cap = result.capacity
    adv = result.advantage
    return {
        "valid": True,
        "model": model_path,
        "fits": result.fits,
        "infeasible_reasons": result.infeasible_reasons,
        "launch_flags": result.launch_flags,
        "hardware": dataclasses.asdict(result.hardware),
        "inputs": dataclasses.asdict(result.inputs),
        "capacity": dataclasses.asdict(cap) if cap is not None else None,
        "advantage": dataclasses.asdict(adv) if adv is not None else None,
        "offload": (
            dataclasses.asdict(result.offload)
            if result.offload is not None
            else None
        ),
        "roofline_estimate": (
            dataclasses.asdict(result.roofline)
            if getattr(result, "roofline", None) is not None
            else None
        ),
    }


def plan_from_payload(payload: dict) -> dict:
    """Run ``feasibility.plan()`` for a UI form payload; return a JSON-able
    dict (or ``{valid: False, reasons: [...]}`` for a rejected manual edit)."""
    from sglang.srt.planner.feasibility import PlanRejected, plan
    from sglang.srt.planner.model import resolve_model_ref

    try:
        model_path = resolve_model_ref(
            payload["model"], gguf_choice=payload.get("gguf_choice") or None
        )
    except (ValueError, KeyError) as e:
        return {"valid": False, "reasons": [f"model: {e}"]}

    hw_payload = payload.get("hardware", {})
    try:
        hardware, reserve_mib = _hardware_and_reserve(payload)
    except (ValueError, KeyError) as e:
        return {"valid": False, "reasons": [f"hardware: {e}"]}

    mem = _int_list(payload.get("rank_gpu_memory_mib"))
    rank_gpu_id = _int_list(payload.get("rank_gpu_id"))
    if mem is not None and len(mem) == 1:
        tp_guess = (
            len(rank_gpu_id)
            if rank_gpu_id
            else (payload.get("tp_size") or len(hardware.gpus))
        )
        mem = mem * int(tp_guess)

    host_ram_mib = payload.get("host_ram_mib") or hw_payload.get("host_ram_mib")
    # Concurrency default = 1 (single-user), matching how these models are
    # actually launched (--max-num-seqs 1 in the reference command). A blank
    # field therefore shows the honest single-user KV capacity instead of an
    # arbitrary many-slot default that inflates the GDN/mamba pool and makes a
    # single-user config look infeasible. The user can raise it and SEE the KV
    # shrink (kv_by_concurrency below).
    concurrency = payload.get("max_running_requests") or 1
    # Vision tower toggle: default OFF (text-only) so the sized footprint is the
    # text-serving resident set most rigs actually run; ticking it adds the
    # vision encoder (HF: the unquantized visual blocks; GGUF: the mmproj
    # sidecar) back into the budget.
    include_vision = bool(payload.get("include_vision", False))
    try:
        result = plan(
            model_path,
            hardware,
            tp_size=payload.get("tp_size"),
            rank_gpu_id=rank_gpu_id,
            rank_gpu_memory_mib=mem,
            rank_tp_ratio=_int_list(payload.get("rank_tp_ratio")),
            rank_mlp_ratio=_int_list(payload.get("rank_mlp_ratio")),
            rank_moe_ratio=_int_list(payload.get("rank_moe_ratio")),
            rank_vocab_ratio=_int_list(payload.get("rank_vocab_ratio")),
            dcp_size=payload.get("dcp_size") or None,
            kv_token_vector=_int_list(payload.get("kv_token_vector")),
            user_free_reserve_mib=reserve_mib,
            kv_cache_dtype=payload.get("kv_cache_dtype") or "auto",
            speculative_algorithm=payload.get("speculative_algorithm") or None,
            speculative_num_draft_tokens=payload.get(
                "speculative_num_draft_tokens"
            )
            or None,
            speculative_draft_model_path=payload.get(
                "speculative_draft_model_path"
            )
            or None,
            max_running_requests=concurrency,
            include_vision=include_vision,
            host_ram_mib=int(host_ram_mib) if host_ram_mib else None,
        )
    except PlanRejected as e:
        return {"valid": False, "reasons": e.reasons}
    except ValueError as e:
        return {"valid": False, "reasons": [str(e)]}

    out = _plan_to_dict(result, model_path)
    out["measured"] = _measured_for_plan(payload, model_path, result)
    if out["measured"] and out.get("roofline_estimate"):
        # A MEASURED entry exists -> the roofline renders as secondary.
        out["roofline_estimate"]["measured_available"] = True
    out["concurrency"] = concurrency
    out["include_vision"] = include_vision
    out["kv_by_concurrency"] = _kv_by_concurrency(
        model_path, hardware, payload, mem, rank_gpu_id, reserve_mib,
        host_ram_mib, include_vision,
    )
    return out


def _measured_for_plan(payload, model_path, result) -> Optional[dict]:
    """Look up MEASURED perf/energy (S2.5 energy module) for this config and
    return it for the UI, or None. Prefers the results.jsonl path in the
    payload, falling back to the energy module's default store. Matches on
    quant bits + tp_size (+ model basename when it lines up) so the measured
    numbers surface next to (and preferred over) the roofline estimate.

    kWh-per-1M-token is derived here (J/token / 3.6) — a pure render-time
    conversion of the measured J/token, never stored as a measured input."""
    from sglang.srt.planner.results_store import ResultsStore

    try:
        from sglang.srt.planner.energy import DEFAULT_RESULTS_STORE
    except Exception:
        DEFAULT_RESULTS_STORE = None
    store_path = payload.get("results_store") or DEFAULT_RESULTS_STORE
    if not store_path or not os.path.exists(store_path):
        return None
    try:
        store = ResultsStore.load(store_path)
    except Exception:
        return None

    tp_size = result.inputs.tp_size
    base = os.path.basename(str(model_path).rstrip("/"))

    def _norm(s):
        return "".join(ch for ch in str(s).lower() if ch.isalnum())

    nb = _norm(base)

    def _kwh_1m(d):
        return {int(k): (float(v) / 3.6) for k, v in (d or {}).items()}

    # Group by config_label (e.g. "no-MTP baseline" vs "MTP+adaptive") so the
    # dashboard shows every sibling row for this model/hardware side by side.
    groups: dict = {}
    for e in store.entries():
        if not e.has_measured_perf():
            continue
        if e.tp_config and not e.tp_config.startswith(f"tp{tp_size}"):
            continue
        # Model match: normalized-alnum substring either way (served name vs
        # checkpoint basename), e.g. "Qwen3.6-27B" <-> "Qwen3.6-27B-FP8".
        ne = _norm(str(e.model).split("/")[-1])
        if not (ne and (ne in nb or nb in ne)):
            continue
        label = e.config_label or "measured"
        groups.setdefault(label, []).append({
            "workload": e.workload or "default",
            "tp_config": e.tp_config,
            "kv_cache_dtype": e.kv_cache_dtype,
            "provenance": e.provenance,
            "config_label": label,
            "spec_accept_length_by_bucket": e.spec_accept_length_by_bucket or {},
            "prefill_tok_s_by_bucket": e.prefill_tok_s_by_bucket or {},
            "decode_tok_s_by_bucket": e.decode_tok_s_by_bucket or {},
            "peak_prefill_tok_s": e.peak_prefill_tok_s,
            "peak_decode_tok_s": e.peak_decode_tok_s,
            "j_per_prefill_token_by_bucket": e.j_per_prefill_token_by_bucket or {},
            "j_per_decode_token_by_bucket": e.j_per_decode_token_by_bucket or {},
            "kwh_per_1m_prefill_by_bucket": _kwh_1m(e.j_per_prefill_token_by_bucket),
            "kwh_per_1m_decode_by_bucket": _kwh_1m(e.j_per_decode_token_by_bucket),
            "per_card_energy": e.per_card_energy,
        })
    if not groups:
        return None
    # Baseline first, then the rest (MTP etc.), so the multiplier reads left->right.
    def _order(lbl):
        return (0 if "baseline" in lbl.lower() or "no-mtp" in lbl.lower() else 1, lbl)
    rows = [{"config_label": lbl, "workloads": groups[lbl]}
            for lbl in sorted(groups, key=_order)]
    return {"store_path": store_path, "rows": rows,
            "workloads": rows[0]["workloads"]}  # back-compat: first row


#: Concurrency ladder shown so the user SEES the KV<->concurrency<->mamba
#: tradeoff. 1 = single-user (huge KV); higher = parallel (mamba pool grows,
#: KV shrinks). The GDN/mamba pool genuinely scales with concurrency -- this
#: surfaces it honestly instead of hiding it behind one default.
_CONCURRENCY_LADDER = (1, 4, 16, 32, 64)


def _kv_by_concurrency(
    model_path, hardware, payload, mem, rank_gpu_id, reserve_mib,
    host_ram_mib, include_vision,
):
    """Max-KV-tokens (and mamba pool, fit) at a ladder of concurrencies, so the
    UI can show how ``max_running_requests`` trades KV cache for parallel
    slots. Cheap re-plans; failures degrade to an absent row."""
    from sglang.srt.planner.feasibility import plan as _plan

    rows = []
    for c in _CONCURRENCY_LADDER:
        try:
            r = _plan(
                model_path, hardware,
                tp_size=payload.get("tp_size"),
                rank_gpu_id=rank_gpu_id,
                rank_gpu_memory_mib=mem,
                rank_tp_ratio=_int_list(payload.get("rank_tp_ratio")),
                dcp_size=payload.get("dcp_size") or None,
                user_free_reserve_mib=reserve_mib,
                kv_cache_dtype=payload.get("kv_cache_dtype") or "auto",
                max_running_requests=c,
                include_vision=include_vision,
                host_ram_mib=int(host_ram_mib) if host_ram_mib else None,
                with_advantage=False,
            )
        except (PlanRejected, ValueError):
            continue
        cap = r.capacity
        # The AGGREGATE context (max_context_tokens), NOT the min per-rank cap:
        # under uneven DCP the token axis is split across ranks, so the servable
        # context is the sum the converged optimum reports (== the headline
        # "max context (KV)"). Reporting min per-rank here understated it ~4x
        # and contradicted the headline (e.g. showed 97k while the header and
        # the per-rank column summed to 408k).
        ctx = int(cap.max_context_tokens) if (cap and r.fits) else 0
        mamba = max((rc.mamba_gib for rc in cap.per_rank), default=0.0) if cap else 0.0
        rows.append({
            "concurrency": c,
            "kv_tokens": max(ctx, 0),
            "mamba_gib": round(mamba, 2),
            "fits": bool(r.fits),
        })
    return rows


def issue_from_payload(payload: dict) -> dict:
    """Re-plan, then render the S2 RESULTS or BUG issue text + prefilled URL
    (the 'Submit config' / 'Report bug' buttons)."""
    from sglang.srt.planner.feasibility import PlanRejected, plan
    from sglang.srt.planner.issue_text import bug_from_plan, results_from_plan
    from sglang.srt.planner.model import resolve_model_ref

    kind = payload.get("kind", "results")
    try:
        model_path = resolve_model_ref(payload["model"])
        hardware = _load_hardware_from_payload(payload.get("hardware", {}))
    except (ValueError, KeyError) as e:
        return {"ok": False, "error": str(e)}

    mem = _int_list(payload.get("rank_gpu_memory_mib"))
    rank_gpu_id = _int_list(payload.get("rank_gpu_id"))
    if mem is not None and len(mem) == 1:
        tp_guess = len(rank_gpu_id) if rank_gpu_id else (
            payload.get("tp_size") or len(hardware.gpus)
        )
        mem = mem * int(tp_guess)
    reserve = _float_list(payload.get("plan_free_reserve_gb"))
    reserve_mib = None
    if reserve is not None:
        if len(reserve) == 1:
            reserve = reserve * len(hardware.gpus)
        reserve_mib = [int(v * 1024) for v in reserve]

    try:
        result = plan(
            model_path,
            hardware,
            tp_size=payload.get("tp_size"),
            rank_gpu_id=rank_gpu_id,
            rank_gpu_memory_mib=mem,
            rank_tp_ratio=_int_list(payload.get("rank_tp_ratio")),
            rank_mlp_ratio=_int_list(payload.get("rank_mlp_ratio")),
            rank_moe_ratio=_int_list(payload.get("rank_moe_ratio")),
            rank_vocab_ratio=_int_list(payload.get("rank_vocab_ratio")),
            dcp_size=payload.get("dcp_size") or None,
            kv_token_vector=_int_list(payload.get("kv_token_vector")),
            user_free_reserve_mib=reserve_mib,
            kv_cache_dtype=payload.get("kv_cache_dtype") or "auto",
        )
    except (PlanRejected, ValueError) as e:
        reasons = getattr(e, "reasons", [str(e)])
        return {"ok": False, "error": "; ".join(reasons)}

    quant = payload.get("quant") or None
    group_size = payload.get("group_size") or None
    repo = payload.get("issue_repo", "efschu/htsglang")
    if kind == "bug":
        issue = bug_from_plan(
            result,
            symptom=payload.get("symptom", "(describe what happened)"),
            log_text=payload.get("log_text") or None,
            quant=quant,
            group_size=group_size,
            owner_repo=repo,
        )
    else:
        issue = results_from_plan(
            result, quant=quant, group_size=group_size, owner_repo=repo
        )
    return {
        "ok": True,
        "kind": kind,
        "title": issue.title,
        "markdown": issue.markdown,
        "url": issue.url,
        "url_within_budget": issue.url_within_budget,
    }


# ===========================================================================
# S4 explorer: profile library + model x rig matrix.
# ===========================================================================


def list_profiles() -> dict:
    """The hardware-profile library (design §2.7), for the explorer's rig
    composer."""
    from sglang.srt.planner.profiles import ProfileLibrary

    lib = ProfileLibrary()
    return {
        "profiles": [
            {
                "name": lib.get(n).name,
                "total_mib": lib.get(n).total_mib,
                "sm_arch": lib.get(n).sm_arch,
                "nvlink": lib.get(n).nvlink,
                "tdp_w": lib.get(n).tdp_w,
            }
            for n in lib.names()
        ]
    }


def matrix_from_payload(payload: dict) -> dict:
    """Build a model x rig matrix (design §2.7). ``models`` = [{label, model}];
    ``rigs`` = [{name, profiles: [profile-name,...]}] (composed from the
    library) or [{name, source:"nvml"}] for the live rig. Composed rigs are
    stamped as estimates (design §8)."""
    from sglang.srt.planner.explorer import plan_matrix
    from sglang.srt.planner.hardware import (
        HardwareUnavailable,
        hardware_from_nvml,
    )
    from sglang.srt.planner.model import resolve_model_ref
    from sglang.srt.planner.profiles import ProfileLibrary, compose_rig

    lib = ProfileLibrary()
    models = []
    for m in payload.get("models", []):
        label = m.get("label") or m["model"]
        try:
            models.append((label, resolve_model_ref(m["model"])))
        except (ValueError, KeyError) as e:
            return {"ok": False, "error": f"model {m.get('model')!r}: {e}"}
    rigs = []
    for r in payload.get("rigs", []):
        if r.get("source") == "nvml":
            try:
                rigs.append(hardware_from_nvml())
            except HardwareUnavailable as e:
                return {"ok": False, "error": f"live rig: {e}"}
            continue
        try:
            rigs.append(compose_rig(r["profiles"], library=lib))
        except (KeyError, ValueError) as e:
            return {"ok": False, "error": f"rig {r.get('name')!r}: {e}"}
    if not models or not rigs:
        return {"ok": False, "error": "need at least one model and one rig"}

    matrix = plan_matrix(models, rigs)
    return {
        "ok": True,
        "models": matrix.models,
        "rigs": matrix.rigs,
        "cells": [dataclasses.asdict(c) for c in matrix.cells],
    }


def landscape_from_payload(payload: dict) -> dict:
    """Build a Mode-A (model, quant) landscape (design §5B.3): measured store
    rows (preferred) + planner feasibility rows for the composed rigs. The
    measured perf/energy columns stay empty until the energy module (S2.5)."""
    import dataclasses as _dc

    from sglang.srt.planner.landscape import build_mode_a
    from sglang.srt.planner.model import resolve_model_ref
    from sglang.srt.planner.profiles import ProfileLibrary, compose_rig
    from sglang.srt.planner.results_store import QuantDescriptor, ResultsStore

    try:
        model_path = resolve_model_ref(payload["model"])
    except (ValueError, KeyError) as e:
        return {"ok": False, "error": f"model: {e}"}
    model_label = payload.get("model_label") or payload["model"].rstrip(
        "/"
    ).rsplit("/", 1)[-1]
    quant = QuantDescriptor.parse(payload.get("quant", "bf16"))

    store = ResultsStore()
    store_path = payload.get("results_store")
    if store_path:
        try:
            store = ResultsStore.load(store_path)
        except Exception as e:
            return {"ok": False, "error": f"results store: {e}"}

    lib = ProfileLibrary()
    planner_rigs = []
    for r in payload.get("rigs", []):
        try:
            planner_rigs.append(
                (model_path, compose_rig(r["profiles"], library=lib))
            )
        except (KeyError, ValueError) as e:
            return {"ok": False, "error": f"rig {r.get('name')!r}: {e}"}

    ls = build_mode_a(
        model_label,
        quant,
        store=store,
        planner_rigs=planner_rigs,
        bucket=payload.get("bucket"),
        similar=bool(payload.get("similar")),
    )
    out = _dc.asdict(ls)
    return {"ok": True, "landscape": out}


# ===========================================================================
# HTTP server (stdlib only; same shape as the rig-dashboard).
# ===========================================================================


# ===========================================================================
# #149 / #150 — energy live-monitoring + scenario-builder route payloads.
# These are thin adapters over sglang.srt.planner.energy; the heavy logic
# (tagging schema, sweep restore, delta-rate math, scenario expansion, the
# cache-flush gate) lives there and is unit-tested without a GPU.
# ===========================================================================


def gpu_state_payload() -> dict:
    """Live per-card power-state tags (#149 Ebene-4 refresh button)."""
    from sglang.srt.planner.energy import read_gpu_power_states

    states = read_gpu_power_states()
    return {"ok": True, "cards": [s.to_json() for s in states]}


def live_snapshot_payload(payload: dict) -> dict:
    """One /metrics scrape of a RUNNING target server (#149 live widget /
    'measure against a running server'). The client polls this at its chosen
    resolution and computes delta rates locally — the token counters are
    already Prometheus counters, so prefill/s + decode/s are pure client deltas.

    Also echoes the CLAMPED resolution so the widget cannot poll below the
    ~30ms floor."""
    from sglang.srt.planner.energy import (
        clamp_live_resolution_ms,
        fetch_live_snapshot,
    )

    target = (payload.get("target") or "").strip()
    if not target:
        return {"ok": False, "error": "no target server (host:port) given"}
    if not target.startswith("http"):
        target = "http://" + target
    resolution_ms = clamp_live_resolution_ms(
        payload.get("resolution_ms", 250.0))
    try:
        snap = fetch_live_snapshot(target)
    except Exception as e:
        return {"ok": False, "error": f"scrape of {target}/metrics failed: {e}"}
    return {
        "ok": True,
        "resolution_ms": resolution_ms,
        "snapshot": {
            "t": snap.t,
            "prompt_tokens_total": snap.prompt_tokens_total,
            "generation_tokens_total": snap.generation_tokens_total,
            "spec_accept_rate": snap.spec_accept_rate,
            "spec_num_steps": snap.spec_num_steps,
            "spec_ema_accept_len": snap.spec_ema_accept_len,
            "gen_throughput": snap.gen_throughput,
        },
    }


def scenario_payload(payload: dict) -> dict:
    """Expand a scenario-builder config (#150) into the concrete list of
    phase/behavior run units, and attach the cache-flush warning gate so the UI
    can require confirmation BEFORE a cold-prefill flush."""
    from sglang.srt.planner.energy import ScenarioConfig, cache_flush_warning

    fields = {
        k: payload[k]
        for k in (
            "scale", "phases", "concurrency", "behaviors", "prefill_tokens",
            "decode_tokens", "duration_s", "multiturn", "turns",
            "turn_growth_tokens", "cold_prefill",
        )
        if k in payload and payload[k] is not None
    }
    try:
        cfg = ScenarioConfig(**fields)
    except (TypeError, ValueError) as e:
        return {"ok": False, "error": str(e)}
    units = cfg.expand()
    will_flush = any(u.get("cold") for u in units)
    warning = cache_flush_warning(
        will_flush=will_flush,
        target_running_server=bool(payload.get("target_running_server")),
    )
    return {
        "ok": True,
        "units": units,
        "cache_flush_warning": warning,
        "summary": {
            "prefill_tokens": cfg.scaled_prefill_tokens(),
            "decode_tokens": cfg.scaled_decode_tokens(),
            "concurrency": cfg.concurrency,
            "n_units": len(units),
        },
    }


def cache_flush_warning_payload(payload: dict) -> dict:
    """Standalone cache-flush gate (#150) — the UI calls this before ANY flush
    (cold-prefill or benchmark) against a target to know whether a blocking
    confirmation is mandatory."""
    from sglang.srt.planner.energy import cache_flush_warning

    return {
        "ok": True,
        **cache_flush_warning(
            will_flush=bool(payload.get("will_flush", True)),
            target_running_server=bool(payload.get("target_running_server")),
        ),
    }


# ===========================================================================
# #147 — persistent "HiCache energy-saved" accumulator (RAM/disk prefix cache).
# ===========================================================================
#
# The HiCache serves prefill tokens from RAM/disk instead of recomputing them.
# Every such token is an avoided prefill recompute -> energy saved. We accumulate
# the recovered-token counter per (model, config_label) and convert the running
# total into a kWh / €-ct saving BAND (von–bis over the measured J/prefill-token
# buckets). The recovery costs NOTHING extra (RAM/disk power is sunk cost), so
# nothing is ever deducted — the saving is the pure avoided recompute.


def _price_ct(payload: dict) -> float:
    try:
        p = float(payload.get("price_ct_per_kwh", 30.0))
        return p if p >= 0 else 30.0
    except (TypeError, ValueError):
        return 30.0


def _hicache_store_path(payload: dict) -> str:
    from sglang.srt.planner.hicache_savings import DEFAULT_HICACHE_STORE

    return payload.get("hicache_store") or DEFAULT_HICACHE_STORE


def _measured_prefill_band(model: str, config_label: str, results_store_path):
    """(lo, hi) measured J/prefill-token band for a (model, config_label), read
    from the measured results store — MIN/MAX across buckets. None when no
    matching MEASURED entry exists (never fabricated). Matching mirrors the
    measured-panel lookup: normalized-alnum model substring + config_label."""
    from sglang.srt.planner.hicache_savings import band_from_buckets
    from sglang.srt.planner.results_store import ResultsStore

    try:
        from sglang.srt.planner.energy import DEFAULT_RESULTS_STORE
    except Exception:
        DEFAULT_RESULTS_STORE = None
    path = results_store_path or DEFAULT_RESULTS_STORE
    if not path or not os.path.exists(path):
        return None
    try:
        store = ResultsStore.load(path)
    except Exception:
        return None

    def _norm(s):
        return "".join(ch for ch in str(s).lower() if ch.isalnum())

    nm = _norm(str(model).split("/")[-1])
    merged: dict = {}
    for e in store.entries():
        if not e.has_measured_perf() or not e.j_per_prefill_token_by_bucket:
            continue
        ne = _norm(str(e.model).split("/")[-1])
        if not (ne and nm and (ne in nm or nm in ne)):
            continue
        if config_label and (e.config_label or "measured") != config_label:
            continue
        merged.update(e.j_per_prefill_token_by_bucket)
    return band_from_buckets(merged)


def hicache_saved_read(payload: Optional[dict] = None) -> dict:
    """GET side of ``/api/hicache_saved``: read the persisted accumulator and
    return every (model, config_label) record with its derived kWh/ct saving
    band plus the grand total. Pure read — never boots a server."""
    from sglang.srt.planner.hicache_savings import HiCacheSavingsStore

    payload = payload or {}
    path = _hicache_store_path(payload)
    price = _price_ct(payload)
    store = HiCacheSavingsStore.load(path)
    return {"ok": True, "store_path": path, **store.to_view(price)}


def hicache_saved_record(payload: dict) -> dict:
    """POST side of ``/api/hicache_saved``: record newly-recovered prefill tokens
    for a (model, config_label) and persist the grown total.

    Two input modes (both accumulate, neither resets):
      * ``target``: scrape the running server's ``/metrics`` and record the
        ABSOLUTE ``sglang:cached_tokens_total`` (HiCache RAM/disk tiers only) as a
        counter snapshot — the store adds only the DELTA since the last snapshot.
      * ``recovered_tokens``: a manual delta to add directly (no server).

    The measured J/prefill-token band is (re)attached from the measured results
    store when available; a measured band is never overwritten by an estimate."""
    from sglang.srt.planner.hicache_savings import (
        HiCacheSavingsStore,
        hicache_recovered_from_metrics,
    )

    model = (payload.get("model") or "").strip()
    config_label = (payload.get("config_label") or "measured").strip()
    if not model:
        return {"ok": False, "error": "model is required to key the saving record"}

    path = _hicache_store_path(payload)
    price = _price_ct(payload)
    store = HiCacheSavingsStore.load(path)
    rec = store.get_or_create(model, config_label)

    # (Re)attach the measured J/prefill-token band (provenance-guarded inside).
    band = _measured_prefill_band(
        model, config_label, payload.get("results_store")
    )
    if band is not None:
        rec.set_band(band[0], band[1], provenance="measured")

    delta = None
    target = (payload.get("target") or "").strip()
    if target:
        if not target.startswith("http"):
            target = "http://" + target
        url = target.rstrip("/") + "/metrics"
        try:
            import urllib.request

            with urllib.request.urlopen(url, timeout=5.0) as r:
                text = r.read().decode("utf-8", "replace")
        except Exception as e:
            return {"ok": False, "error": f"scrape of {url} failed: {e}"}
        total = hicache_recovered_from_metrics(text)
        delta = rec.record_counter_snapshot(total)
    elif payload.get("recovered_tokens") is not None:
        try:
            n = float(payload["recovered_tokens"])
        except (TypeError, ValueError):
            return {"ok": False, "error": "recovered_tokens must be a number"}
        try:
            rec.add_tokens(n)
        except ValueError as e:
            return {"ok": False, "error": str(e)}
        delta = n
    else:
        return {
            "ok": False,
            "error": "provide either 'target' (server host:port to scrape) or "
            "'recovered_tokens' (manual delta)",
        }

    store.save(path)
    return {
        "ok": True,
        "store_path": path,
        "recorded_delta_tokens": delta,
        "record": rec.to_view(price),
        **store.to_view(price),
    }


class _Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj), "application/json")

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw or b"{}")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, INDEX_HTML, "text/html; charset=utf-8")
            return
        if self.path.startswith("/api/knobs"):
            try:
                self._json(200, discover_knobs())
            except Exception as e:  # pragma: no cover - defensive
                self._json(500, {"error": str(e)})
            return
        if self.path.startswith("/api/profiles"):
            try:
                self._json(200, list_profiles())
            except Exception as e:  # pragma: no cover - defensive
                self._json(500, {"error": str(e)})
            return
        if self.path.startswith("/api/detect"):
            try:
                self._json(200, detect_hardware())
            except Exception as e:  # pragma: no cover - defensive
                self._json(500, {"ok": False, "error": str(e)})
            return
        if self.path.startswith("/api/gpu_state"):
            try:
                self._json(200, gpu_state_payload())
            except Exception as e:  # pragma: no cover - defensive
                self._json(500, {"ok": False, "error": str(e)})
            return
        if self.path.startswith("/api/hicache_saved"):
            try:
                from urllib.parse import parse_qs, urlsplit

                q = {k: v[0] for k, v in parse_qs(urlsplit(self.path).query).items()}
                self._json(200, hicache_saved_read(q))
            except Exception as e:  # pragma: no cover - defensive
                self._json(500, {"ok": False, "error": str(e)})
            return
        self._send(404, "not found", "text/plain")

    def do_POST(self):
        try:
            payload = self._read_json()
        except Exception as e:
            self._json(400, {"error": f"bad json: {e}"})
            return
        try:
            if self.path.startswith("/api/gguf_options"):
                self._json(200, gguf_options_for(payload))
                return
            if self.path.startswith("/api/plan"):
                self._json(200, plan_from_payload(payload))
                return
            if self.path.startswith("/api/issue"):
                self._json(200, issue_from_payload(payload))
                return
            if self.path.startswith("/api/matrix"):
                self._json(200, matrix_from_payload(payload))
                return
            if self.path.startswith("/api/landscape"):
                self._json(200, landscape_from_payload(payload))
                return
            if self.path.startswith("/api/live"):
                self._json(200, live_snapshot_payload(payload))
                return
            if self.path.startswith("/api/scenario"):
                self._json(200, scenario_payload(payload))
                return
            if self.path.startswith("/api/cache_flush_warning"):
                self._json(200, cache_flush_warning_payload(payload))
                return
            if self.path.startswith("/api/hicache_saved"):
                self._json(200, hicache_saved_record(payload))
                return
        except Exception as e:  # pragma: no cover - defensive
            self._json(500, {"error": str(e)})
            return
        self._json(404, {"error": "not found"})

    def log_message(self, *a):  # quiet
        pass


def serve(host: str = "127.0.0.1", port: int = 8780) -> None:
    srv = ThreadingHTTPServer((host, port), _Handler)
    print(f"Config-Planner UI on http://{host}:{port}/")
    print("  offline planner — no GPU, no server boot; every edit re-plans.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


# ===========================================================================
# The single HTML page (inline CSS + JS, no CDN, self-contained).
# ===========================================================================

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>htsglang config planner</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
         margin: 0; padding: 1.2rem; line-height: 1.4;
         background: #0e1116; color: #d7dde5; }
  h1 { font-size: 1.15rem; margin: 0 0 .2rem; }
  .sub { color: #8b96a5; font-size: .8rem; margin-bottom: 1rem; }
  .cols { display: grid; grid-template-columns: 360px 1fr; gap: 1.2rem; }
  @media (max-width: 820px) { .cols { grid-template-columns: 1fr; } }
  fieldset { border: 1px solid #2a323d; border-radius: 8px; margin: 0 0 .9rem;
             padding: .7rem .8rem; }
  legend { color: #9fb0c3; font-size: .78rem; padding: 0 .35rem; }
  label { display: block; font-size: .72rem; color: #9aa6b4; margin: .5rem 0 .15rem; }
  input, select, textarea {
    width: 100%; background: #161b22; color: #e6edf3; border: 1px solid #303a46;
    border-radius: 5px; padding: .32rem .45rem; font: inherit; font-size: .8rem; }
  input::placeholder { color: #5b6675; }
  .knob-help { color: #6b7686; font-size: .66rem; margin: .1rem 0 .2rem; }
  button { background: #1f6feb; color: #fff; border: 0; border-radius: 6px;
           padding: .5rem .9rem; font: inherit; font-size: .8rem; cursor: pointer; }
  button.secondary { background: #30363d; }
  button.mini { padding: .18rem .5rem; font-size: .68rem; }
  .cardlist { display: flex; flex-direction: column; gap: .35rem; }
  .cardrow { display: grid; grid-template-columns: auto 1fr auto auto;
             gap: .4rem; align-items: center; background: #12171e;
             border: 1px solid #263041; border-radius: 6px; padding: .3rem .45rem; }
  .cardrow input[type=text] { padding: .2rem .35rem; font-size: .74rem; }
  .cardrow .vram { width: 5.2rem; text-align: right; }
  .cardrow .resv { width: 4rem; text-align: right; }
  .cardrow .cap { font-size: .64rem; color: #7f8b99; }
  .cardrow.excluded { opacity: .45; }
  .cardrow input[type=checkbox] { width: auto; }
  .verdict.offload { background: #33280f; color: #e3b341; border: 1px solid #7a5c14; }
  .verdict { font-size: 1rem; font-weight: 700; padding: .55rem .7rem;
             border-radius: 7px; margin-bottom: .8rem; }
  .fit { background: #10331d; color: #56d364; border: 1px solid #1c6b34; }
  .nofit { background: #3a1417; color: #ff7b72; border: 1px solid #7d2a2a; }
  table { border-collapse: collapse; width: 100%; font-size: .76rem; }
  th, td { text-align: right; padding: .3rem .5rem; border-bottom: 1px solid #232b35; }
  th:first-child, td:first-child { text-align: left; }
  .bar { height: 9px; background: #232b35; border-radius: 4px; overflow: hidden; }
  .bar > span { display: block; height: 100%; background: #2f81f7; }
  .pricebar { margin-top: .9rem; font-size: .78rem; color: #adbac7; }
  .pricebar input { width: 5rem; padding: .2rem .4rem; background: #0d1117;
                    color: #e6edf3; border: 1px solid #30363d; border-radius: 5px; }
  .measured { margin-top: .9rem; background: #0f1a12; border: 1px solid #2ea043;
              border-radius: 8px; padding: .7rem .8rem; }
  .ms-title { font-weight: 700; color: #56d364; font-size: .9rem;
              text-transform: uppercase; letter-spacing: .02em; }
  .ms-note { font-size: .72rem; color: #9fb0a4; margin: .35rem 0 .5rem; }
  .ms-wl { margin: .5rem 0; }
  .ms-mult { font-size: .8rem; color: #d2f7dd; background: #10251a;
             border: 1px solid #2ea043; border-radius: 6px; padding: .4rem .6rem;
             margin: .4rem 0; }
  .ms-row { border-top: 1px solid #21402b; margin-top: .6rem; padding-top: .4rem; }
  .ms-row-h { font-size: .8rem; color: #adbac7; }
  .ms-wl-h { font-size: .74rem; color: #8b98a5; margin-top: .35rem; }
  .ms-phases { display: flex; gap: 1rem; flex-wrap: wrap; }
  .ms-phase { flex: 1 1 320px; min-width: 300px; border-radius: 6px;
              padding: .35rem .5rem; }
  .ms-prefill { background: #0d1a22; border: 1px solid #1f3a4d; }
  .ms-decode  { background: #10251a; border: 1px solid #235c34; }
  .ms-ph-h { font-weight: 700; font-size: .74rem; letter-spacing: .02em;
             margin-bottom: .2rem; }
  .ms-prefill .ms-ph-h { color: #6cb6ff; }
  .ms-decode .ms-ph-h { color: #56d364; }
  .ms-phase table, .ms-wl table { border-collapse: collapse; margin: .3rem 0;
                                  font-size: .7rem; width: 100%; }
  .ms-phase th, .ms-phase td, .ms-wl th, .ms-wl td {
      border: 1px solid #21402b; padding: .16rem .45rem; text-align: right; }
  .ms-phase th, .ms-wl th { color: #7f8b99; font-weight: 600; }
  .roofline { margin-top: .9rem; background: #191410; border: 1px dashed #7a5c14;
              border-radius: 8px; padding: .7rem .8rem; }
  .rf-title { font-weight: 700; color: #e3b341; font-size: .9rem;
              text-transform: uppercase; letter-spacing: .02em; }
  .rf-nums { display: flex; gap: 1.2rem; margin: .5rem 0; flex-wrap: wrap; }
  .rf-num { background: #12171e; border: 1px solid #263041; border-radius: 6px;
            padding: .4rem .7rem; }
  .rf-num span { display: block; font-size: .66rem; color: #7f8b99;
                 text-transform: uppercase; }
  .rf-num b { font-size: 1.15rem; color: #e6edf3; }
  .rf-num small { display: block; font-size: .62rem; color: #7f8b99; }
  .rf-meas { color: #56d364; font-weight: 600; }
  .rf-name { color: #d29922; }
  .rf-caveats { margin: .5rem 0 0; padding-left: 1.1rem; font-size: .68rem;
                color: #8b949e; }
  .rf-caveats li { margin: .15rem 0; }
  pre { background: #161b22; border: 1px solid #232b35; border-radius: 6px;
        padding: .6rem; overflow-x: auto; font-size: .74rem; white-space: pre-wrap;
        word-break: break-word; }
  .muted { color: #8b96a5; font-size: .72rem; }
  .est { color: #d29922; font-size: .68rem; }
  .reasons li { color: #ff9d96; font-size: .76rem; margin: .2rem 0; }
  .adv { border-left: 3px solid #2f81f7; padding: .3rem .6rem; margin: .5rem 0; }
  .knoblist { font-size: .68rem; color: #7f8b99; }
  .pill { display: inline-block; background: #21262d; border: 1px solid #30363d;
          border-radius: 10px; padding: .05rem .5rem; margin: .1rem .15rem 0 0;
          font-size: .66rem; }
  .actions { display: flex; gap: .5rem; flex-wrap: wrap; margin-top: .3rem; }
  a { color: #58a6ff; }
  .tabs { display: flex; gap: .4rem; margin-bottom: .8rem; }
  .tab { background: #21262d; color: #9aa6b4; }
  .tab.active { background: #1f6feb; color: #fff; }
  .mx td, .mx th { text-align: center; }
  .mx .estcell { outline: 1px dashed #6e5417; }
  .mx .fitc { color: #56d364; } .mx .nofitc { color: #ff7b72; }
  .legend { color: #8b96a5; font-size: .68rem; margin-top: .5rem; }
</style>
</head>
<body>
<h1>htsglang offline config planner</h1>
<div class="sub">capacity / feasibility / split &mdash; never estimated throughput.
  Every edit re-runs the same planner the server runs. No GPU touched.</div>

<div class="tabs">
  <button id="tab_plan" class="tab active" onclick="showTab('plan')">Plan a model</button>
  <button id="tab_explore" class="tab" onclick="showTab('explore')">Explore rigs (matrix)</button>
  <button id="tab_landscape" class="tab" onclick="showTab('landscape')">Landscape (benchmark DB)</button>
  <button id="tab_energy" class="tab" onclick="showTab('energy')">Energy &amp; live</button>
</div>

<div id="view_landscape" style="display:none">
  <div class="sub">Per (model, quant): measured cross-rig spread (Mode A).
    <b>Measured</b> rows come from a results.jsonl of real runs; the rest are
    planner/composed <b>feasibility estimates</b>. Perf/energy columns are
    MEASURED-only &mdash; empty until the energy module (S2.5); no number is
    invented.</div>
  <div class="cols">
    <div>
      <fieldset>
        <legend>model + quant</legend>
        <label>Model (path / HF id)</label>
        <input id="ls_model" placeholder="/path/to/Qwen3.6-27B-AWQ">
        <label>quant descriptor</label>
        <input id="ls_quant" placeholder="4b/AWQ/g128 | Q4_K_M | fp8">
        <label>efficiency bucket (batch size)</label>
        <input id="ls_bucket" placeholder="(optional, e.g. 16)">
        <label>results.jsonl (measured; optional)</label>
        <input id="ls_store" placeholder="(path to a measured store)">
        <label><input type="checkbox" id="ls_similar" style="width:auto"> similar-quant (by bits — approximate)</label>
      </fieldset>
      <fieldset>
        <legend>rigs to feasibility-check (NAME=profile,profile,...)</legend>
        <textarea id="ls_rigs" rows="4" placeholder="hetero=RTX 5090,RTX 3080 20GB,RTX 3080 20GB&#10;4x4090=RTX 4090,RTX 4090,RTX 4090,RTX 4090"></textarea>
      </fieldset>
      <button onclick="doLandscape()">Build landscape</button>
    </div>
    <div><div id="ls_out"></div></div>
  </div>
</div>

<div id="view_energy" style="display:none">
  <div class="sub">Live GPU power-state tags (#149), a live throughput/MTP
    widget that measures against a <b>running</b> server (no boot/teardown),
    and a configurable measurement <b>scenario</b> builder (#150). Efficiency
    (J/token) is only comparable at the <b>same</b> power-limit context &mdash;
    the tags below make that context explicit.</div>
  <div class="cols">
    <div>
      <fieldset>
        <legend>GPU power state
          <button class="secondary mini" onclick="refreshGpuState()" title="re-query NVML live">refresh</button></legend>
        <div id="gpu_state_out" class="muted">click refresh…</div>
      </fieldset>
      <fieldset>
        <legend>live monitor (measure against a running server)</legend>
        <label>target server (host:port)</label>
        <input id="live_target" placeholder="127.0.0.1:31000">
        <label>resolution (ms) &mdash; floor ~30ms</label>
        <input id="live_res" value="250">
        <div style="margin-top:.5rem">
          <button onclick="toggleLive()" id="live_btn">start live</button>
          <button class="secondary" onclick="remeasureNow()" title="single immediate scrape">re-measure now</button>
        </div>
        <div id="live_out" class="muted" style="margin-top:.6rem">idle.</div>
      </fieldset>
    </div>
    <div>
      <fieldset>
        <legend>scenario builder (#150)</legend>
        <div class="cols" style="grid-template-columns:1fr 1fr;gap:.6rem">
          <div>
            <label>scale (1.0 = 10k prefill / 1k decode)</label>
            <input id="sc_scale" value="1.0">
            <label>phases</label>
            <select id="sc_phases"><option value="both">both</option>
              <option value="prefill">prefill-only</option>
              <option value="decode">decode-only</option></select>
            <label>concurrency</label>
            <input id="sc_conc" value="1">
          </div>
          <div>
            <label>behaviors (decode split)</label>
            <select id="sc_behav"><option value="code,prose">code + prose</option>
              <option value="code">code only</option>
              <option value="prose">prose only</option></select>
            <label><input type="checkbox" id="sc_multi" style="width:auto"> multiturn (growing context)</label>
            <label>turns (multiturn)</label>
            <input id="sc_turns" value="3">
            <label><input type="checkbox" id="sc_cold" checked style="width:auto"> cold prefill (flush + fresh prompt)</label>
            <label><input type="checkbox" id="sc_running" style="width:auto"> target is a RUNNING (production) server</label>
          </div>
        </div>
        <div style="margin-top:.5rem"><button onclick="previewScenario()">preview scenario</button></div>
        <div id="sc_out" class="muted" style="margin-top:.6rem"></div>
      </fieldset>
    </div>
  </div>
</div>

<div id="view_explore" style="display:none">
  <div class="sub">Compose rigs from the hardware-profile library (or add the
    live one) and see which models fit on each. <b>Composed rigs are
    ESTIMATES</b> &mdash; no measured free-VRAM or interconnect (§8).</div>
  <div class="cols">
    <div>
      <fieldset>
        <legend>models (one per line: LABEL=path, or just a path)</legend>
        <textarea id="mx_models" rows="4" placeholder="27B-AWQ=/path/to/Qwen3.6-27B-AWQ&#10;35B-A3B=/path/to/model"></textarea>
      </fieldset>
      <fieldset>
        <legend>rigs (one per line: NAME=profile,profile,... — or 'live')</legend>
        <textarea id="mx_rigs" rows="4" placeholder="hetero=RTX 5090,RTX 3080 20GB,RTX 3080 20GB&#10;4x4090=RTX 4090,RTX 4090,RTX 4090,RTX 4090"></textarea>
        <div id="mx_profiles" class="knoblist" style="margin-top:.5rem"></div>
      </fieldset>
      <button onclick="doMatrix()">Build matrix</button>
    </div>
    <div><div id="mx_out"></div></div>
  </div>
</div>

<div id="view_plan">
<div class="cols">
  <div>
    <fieldset>
      <legend>model</legend>
      <label>Model (config.json dir, .gguf file, GGUF dir, or HF id)</label>
      <input id="model" placeholder="/path/to/Qwen3.6-27B-AWQ or org/model" onchange="onModelChange()">
      <div id="gguf_pick" style="display:none">
        <label>GGUF quant (several available &mdash; pick one)</label>
        <select id="gguf_choice" onchange="doPlan()"></select>
      </div>
    </fieldset>

    <fieldset>
      <legend>hardware &mdash; tick the cards to plan on
        <button class="secondary mini" onclick="detectGPUs()" title="re-query NVML">detect</button></legend>
      <div id="detect_note" class="muted" style="margin-bottom:.4rem">detecting local GPUs…</div>
      <div id="cardlist" class="cardlist"></div>
      <div class="actions" style="margin-top:.4rem">
        <button class="secondary mini" onclick="addCard(false)">+ add card</button>
        <button class="secondary mini" onclick="addCard(true)" title="a hypothetical GPU you don't own yet">+ add future GPU</button>
      </div>
      <div class="muted" style="margin-top:.4rem">Tick = include in the plan.
        &ldquo;keep free&rdquo; is per-card headroom (a display / another
        process) carved off before sizing. Add real or hypothetical cards to
        see what a future rig could run.</div>
      <label style="margin-top:.5rem">Host RAM total (GiB) &mdash; for RAM-offload fit</label>
      <input id="host_ram_gb" placeholder="(auto-detected; override to plan another host)">
    </fieldset>

    <fieldset>
      <legend>plan &mdash; blank = auto-derive</legend>
      <div id="knobs"></div>
      <label>tp-size</label>
      <input id="tp_size" placeholder="(= #included cards, or set)">
      <label style="margin-top:.6rem"><b>max concurrent requests</b>
        &mdash; drives max KV cache &amp; the GDN/mamba pool</label>
      <input id="max_running_requests" placeholder="(default 1 = single-user &rarr; largest KV)"
        onchange="doPlan()">
      <div class="muted" style="margin:.2rem 0 .4rem">1 = single-user (biggest KV);
        raise it for parallel requests &mdash; the mamba/SSM pool grows and max
        KV shrinks. See the KV-vs-concurrency table in the result.</div>
      <label><input type="checkbox" id="include_vision" onchange="doPlan()">
        include vision tower <span class="muted">(VL models; off = text-only,
        frees VRAM for KV)</span></label>
      <label style="margin-top:.5rem"><b>KV-cache quantization</b>
        &mdash; halves/doubles the KV budget</label>
      <select id="kv_cache_dtype" onchange="doPlan()">
        <option value="auto">auto (model dtype, ~2 B/cell)</option>
        <option value="fp8_e4m3">fp8_e4m3 (1 B/cell &rarr; ~2x context)</option>
        <option value="fp8_e5m2">fp8_e5m2 (1 B/cell &rarr; ~2x context)</option>
      </select>
      <div class="muted" style="margin:.2rem 0 .4rem">fp8 KV halves the
        per-token cell &rarr; ~2x the max context for the same VRAM. Match this
        to your launch <code>--kv-cache-dtype</code>.</div>
      <label>quant descriptor (for the issue text)</label>
      <input id="quant" placeholder="compressed-tensors / Q4_K_M / fp8">
    </fieldset>

    <div class="actions">
      <button onclick="doPlan()">Plan / re-validate</button>
      <button class="secondary" onclick="doIssue('results')">Submit config (RESULTS issue)</button>
      <button class="secondary" onclick="doIssue('bug')">Report bug</button>
    </div>
    <div id="notexpr" class="knoblist" style="margin-top:.8rem"></div>
  </div>

  <div>
    <div id="verdict"></div>
    <div id="split"></div>
    <div id="cards"></div>
    <div id="advantage"></div>
    <div id="pricebar" class="pricebar">
      energy price
      <input id="kwh_price" type="number" step="1" min="0" value="30"
             oninput="if (window.__lastMeasured !== undefined) renderMeasured(window.__lastMeasured); if (window.__lastHicache !== undefined) renderHicacheSaved(window.__lastHicache)">
      <span class="muted">ct/kWh (&euro;-cent; currency-agnostic &mdash; just a
      multiplier). Costs below are in <b>ct</b>. Applies to the measured cost,
      the HiCache kWh-saved band (#147) and, later, the virtual-rig estimate
      (#148).</span>
    </div>
    <div id="measured"></div>
    <div id="hicache_saved"></div>
    <div id="roofline"></div>
    <div id="flags"></div>
    <div id="issue"></div>
  </div>
</div>
</div>

<script>
const $ = id => document.getElementById(id);
let KNOBS = [];

async function loadKnobs() {
  const r = await fetch('/api/knobs'); const d = await r.json();
  // plan_free_reserve_gb is now a PER-CARD input in the hardware panel
  // (design §PART 3), so drop it from the generic knob list to avoid two
  // places to set the same thing.
  KNOBS = d.knobs.filter(k => k.id !== 'plan_free_reserve_gb');
  const box = $('knobs'); box.innerHTML = '';
  for (const k of KNOBS) {
    if (k.id === 'kv_token_vector' && k.env) k.label += ' (env ' + k.env + ')';
    const lab = document.createElement('label'); lab.textContent = k.label;
    const inp = document.createElement('input');
    inp.id = 'knob_' + k.id; inp.placeholder = 'auto';
    const help = document.createElement('div'); help.className = 'knob-help';
    help.textContent = k.help + '  [' + k.source + ']';
    box.appendChild(lab); box.appendChild(inp); box.appendChild(help);
  }
  $('notexpr').innerHTML = '<b>Not offered (the runtime cannot honor these):</b><br>'
    + d.not_expressible.map(x => '&bull; ' + x).join('<br>')
    + '<br><span class="muted">detected fields: '
    + d.detected_fields.map(f => '<span class="pill">'+f+'</span>').join('') + '</span>';
}

// ---- hardware card list (detect + select + per-card reserve + virtual) ----
let CARDS = [];          // {name,total_mib,include,reserve_gb,virtual,free_mib}
let HOST_RAM_MIB = null;

async function detectGPUs() {
  const r = await fetch('/api/detect'); const d = await r.json();
  if (d.host_ram_mib) { HOST_RAM_MIB = d.host_ram_mib;
    if (!$('host_ram_gb').value) $('host_ram_gb').placeholder =
      '(detected '+(d.host_ram_mib/1024).toFixed(0)+' GiB; override to plan another host)'; }
  if (d.ok && d.gpus && d.gpus.length) {
    // Replace any previously-detected rows; keep user-added virtual cards.
    CARDS = CARDS.filter(c => c.virtual);
    for (const g of d.gpus) CARDS.unshift({
      name: g.name, total_mib: g.total_mib, include: true,
      reserve_gb: 0, virtual: false, free_mib: g.free_mib });
    $('detect_note').textContent = 'detected '+d.gpus.length+' GPU(s) via '+(d.source||'nvml');
  } else {
    $('detect_note').textContent = 'no GPU detected'+(d.error?' ('+d.error+')':'')
      + ' — add cards (real or hypothetical) below.';
    if (!CARDS.length) addCard(false);
  }
  renderCards();
}

function addCard(virtual) {
  CARDS.push({ name: virtual ? 'Future GPU' : 'RTX ????',
    total_mib: virtual ? 32768 : 16384, include: true, reserve_gb: 0,
    virtual: !!virtual, free_mib: null });
  renderCards();
}

function renderCards() {
  const box = $('cardlist'); box.innerHTML = '';
  CARDS.forEach((c, i) => {
    const row = document.createElement('div');
    row.className = 'cardrow' + (c.include ? '' : ' excluded');
    // checkbox (include)
    const cb = document.createElement('input');
    cb.type = 'checkbox'; cb.checked = c.include;
    cb.onchange = () => { c.include = cb.checked; renderCards(); };
    // name + capability tag
    const nameWrap = document.createElement('span');
    const name = document.createElement('input');
    name.type = 'text'; name.value = c.name;
    name.onchange = () => { c.name = name.value; };
    nameWrap.appendChild(name);
    const tag = document.createElement('span'); tag.className = 'cap';
    tag.textContent = c.virtual ? ' virtual'
      : (c.free_mib != null ? ' ' + (c.free_mib/1024).toFixed(1) + 'G free' : '');
    nameWrap.appendChild(tag);
    // VRAM (MiB)
    const vramWrap = document.createElement('span'); vramWrap.title = 'total VRAM (MiB)';
    const vram = document.createElement('input');
    vram.type = 'text'; vram.className = 'vram'; vram.value = c.total_mib;
    vram.onchange = () => { c.total_mib = parseInt(vram.value) || 0; };
    vramWrap.appendChild(vram); vramWrap.appendChild(document.createTextNode(' MiB'));
    // per-card reserve (keep N GiB free)
    const resWrap = document.createElement('span');
    resWrap.title = 'keep this many GiB free on this card';
    resWrap.appendChild(document.createTextNode('keep '));
    const res = document.createElement('input');
    res.type = 'text'; res.className = 'resv'; res.value = c.reserve_gb;
    res.onchange = () => { c.reserve_gb = parseFloat(res.value) || 0; };
    resWrap.appendChild(res); resWrap.appendChild(document.createTextNode(' G'));
    row.appendChild(cb); row.appendChild(nameWrap);
    row.appendChild(vramWrap); row.appendChild(resWrap);
    box.appendChild(row);
  });
}

async function onModelChange() {
  const m = $('model').value.trim();
  $('gguf_pick').style.display = 'none';
  if (!m) return;
  const r = await fetch('/api/gguf_options', {method:'POST', body: JSON.stringify({model:m})});
  const d = await r.json();
  if (d.ok && d.options && d.options.length > 1) {
    const sel = $('gguf_choice');
    sel.innerHTML = d.options.map(o=>'<option value="'+esc(o)+'">'+esc(o)+'</option>').join('');
    $('gguf_pick').style.display = '';
  }
}

function payload() {
  const cards = CARDS.map(c => ({ name: c.name, total_mib: c.total_mib,
    include: c.include, reserve_gb: c.reserve_gb, virtual: c.virtual }));
  const hostGb = $('host_ram_gb').value.trim();
  const host_ram_mib = hostGb ? Math.round(parseFloat(hostGb)*1024) : HOST_RAM_MIB;
  const p = {
    model: $('model').value.trim(),
    hardware: { source: 'cards', cards, host_ram_mib },
    tp_size: $('tp_size').value ? parseInt($('tp_size').value) : null,
    max_running_requests: $('max_running_requests').value ? parseInt($('max_running_requests').value) : null,
    include_vision: $('include_vision').checked,
    kv_cache_dtype: $('kv_cache_dtype').value.trim(),
    quant: $('quant').value.trim(),
  };
  if ($('gguf_pick').style.display !== 'none') p.gguf_choice = $('gguf_choice').value;
  for (const k of KNOBS) {
    const v = $('knob_' + k.id).value.trim();
    if (v !== '') p[k.id] = v;
  }
  return p;
}

function bar(pct) {
  const c = Math.max(0, Math.min(100, pct));
  return '<div class="bar"><span style="width:'+c+'%"></span></div>';
}

// KV<->concurrency<->mamba tradeoff: the honest picture that the GDN/mamba
// pool grows with parallel slots, shrinking max KV. The chosen concurrency
// row is highlighted so single-user (1) vs parallel is obvious.
function concurrencyTable(d) {
  const rows = d.kv_by_concurrency;
  if (!rows || !rows.length) return '';
  const cur = d.concurrency;
  let body = rows.map(r =>
    '<tr'+(r.concurrency===cur?' style="font-weight:600;background:rgba(80,160,255,.12)"':'')+'>'
    +'<td>'+r.concurrency+(r.concurrency===cur?' &#9664;':'')+'</td>'
    +'<td>'+(r.fits? r.kv_tokens.toLocaleString() : '&mdash;')+'</td>'
    +'<td>'+r.mamba_gib.toFixed(1)+'</td>'
    +'<td>'+(r.fits?'&check;':'no fit')+'</td></tr>').join('');
  return '<p class="muted" style="margin:.6rem 0 .2rem">max context (KV) vs. '
    +'concurrency <span class="est">(aggregate servable context across the '
    +'uneven-DCP ranks; GDN/mamba pool grows with parallel slots &rarr; context '
    +'shrinks; ESTIMATE)</span></p>'
    +'<table><tr><th>max concurrent</th><th>max context (KV)</th>'
    +'<th>mamba GiB</th><th>fits</th></tr>'+body+'</table>';
}

async function doPlan() {
  $('issue').innerHTML = '';
  const r = await fetch('/api/plan', {method:'POST', body: JSON.stringify(payload())});
  const d = await r.json();
  render(d);
}

function render(d) {
  if (d.valid === false) {
    $('verdict').innerHTML = '<div class="verdict nofit">PLAN REJECTED</div>';
    $('split').innerHTML = '<ul class="reasons">' +
      d.reasons.map(x=>'<li>'+esc(x)+'</li>').join('') + '</ul>';
    $('cards').innerHTML = $('advantage').innerHTML = $('flags').innerHTML = '';
    $('roofline').innerHTML = '';
    return;
  }
  const cap = d.capacity;
  const off = d.offload;
  // Three honest states: fits-in-VRAM (fast) / fits-with-RAM-offload (slower,
  // PCIe-bound) / genuinely cannot fit (design §PART 4).
  if (d.fits) {
    $('verdict').innerHTML = '<div class="verdict fit">FITS IN VRAM &check; '
      + '<span class="est">(estimate &mdash; the runtime measures real free bytes)</span></div>';
  } else if (off && off.status === 'ram_offload') {
    $('verdict').innerHTML = '<div class="verdict offload">FITS WITH ~'
      + off.offloaded_gib.toFixed(1) + ' GiB ON HOST RAM &mdash; slower (PCIe-bound)</div>';
  } else {
    $('verdict').innerHTML = '<div class="verdict nofit">DOES NOT FIT</div>';
  }
  // offload detail line
  let offHtml = '';
  if (off && off.status !== 'vram') {
    offHtml = '<div class="adv"><b>RAM-offload:</b> ' + esc(off.note);
    if (off.per_rank_offloaded_gib && off.offloaded_gib > 0)
      offHtml += '<br><span class="muted">per-rank to host: ['
        + off.per_rank_offloaded_gib.map(x=>x.toFixed(1)).join(', ') + '] GiB'
        + (off.host_ram_total_mib ? ' · host total '+(off.host_ram_total_mib/1024).toFixed(0)+' GiB' : '')
        + '</span>';
    offHtml += '</div>';
  }
  if (!d.fits && off && off.status === 'ram_offload') {
    // Show capacity as usual (it fits with offload) plus the offload note.
    $('split').innerHTML = offHtml +
      '<ul class="reasons"><li>Does not fit in VRAM alone:</li>' +
      (d.infeasible_reasons||[]).map(x=>'<li>'+esc(x)+'</li>').join('') + '</ul>';
  } else if (!d.fits) {
    $('split').innerHTML = offHtml + '<ul class="reasons">' +
      (d.infeasible_reasons||[]).map(x=>'<li>'+esc(x)+'</li>').join('') + '</ul>';
  } else if (offHtml) {
    $('split').innerHTML = offHtml;
  } else {
    let s = '';
    if (cap) {
      s += '<p class="muted">max context (KV): <b>~'+Math.round(cap.max_context_tokens).toLocaleString()
        +'</b> tokens <span class="est">(converged weighted-DCP optimum; ESTIMATE)</span></p>';
      if (cap.token_vector) s += '<p class="muted">KV token-split hint: ['+cap.token_vector.join(', ')+']</p>';
    }
    $('split').innerHTML = s;
  }
  // per-card breakdown
  if (cap && cap.per_rank) {
    let rows = cap.per_rank.map(rc =>
      '<tr><td>rank '+rc.rank+' &rarr; GPU '+rc.gpu_index+'</td>'
      +'<td>'+rc.budget_mib+'</td><td>'+rc.weight_gib.toFixed(1)+'</td>'
      +'<td>'+rc.mamba_gib.toFixed(1)+'</td>'
      +'<td>'+Math.round(Math.max(0,rc.kv_tokens)).toLocaleString()+'</td>'
      +'<td style="min-width:90px">'+bar(rc.budget_used_pct)+' '+rc.budget_used_pct.toFixed(0)+'%</td></tr>').join('');
    $('cards').innerHTML =
      '<table><tr><th>rank</th><th>budget MiB</th><th>weights GiB</th>'
      +'<th>mamba GiB</th><th>KV tok</th><th>budget used</th></tr>'+rows+'</table>'
      +(cap.mlp_units ? '<p class="muted">MLP unit partition: ['+cap.mlp_units.join(', ')+']</p>' : '');
    $('cards').innerHTML += concurrencyTable(d);
  } else { $('cards').innerHTML = ''; }
  // advantage (honest: feasibility + capacity-% + measured-only scores)
  const a = d.advantage; let av='';
  if (a) {
    av += '<div class="adv"><b>vs stock even-TP</b> (capacity/feasibility &mdash; no throughput guess)<br>';
    if (a.stock.runs) {
      av += 'stock runs; max context ~'+Math.round(a.stock.max_context_tokens).toLocaleString()+' tokens.';
      if (a.capacity_pct_range) {
        const [lo,hi]=a.capacity_pct_range;
        av += '<br>capacity: <b>'+(lo>=0?'+':'')+lo+'% .. '+(hi>=0?'+':'')+hi+'%</b> KV/context '
          +'<span class="est">(ratio of two same-model estimates)</span>';
      }
    } else {
      av += 'stock even-TP <b>DOES NOT RUN</b>:<ul class="reasons">'
        + a.stock.reasons.map(x=>'<li>'+esc(x)+'</li>').join('')
        + '</ul>' + (d.fits ? '&rArr; the advantage is feasibility itself.' : '');
    }
    if (a.measured) {
      av += '<br><span class="muted">measured card scores (cached profile, not estimated): '
        + a.measured.map(s=>'GPU '+s.gpu_index+' '+s.gemm_tflops.toFixed(0)+' TFLOPS/'+s.membw_gbs.toFixed(0)+' GB/s').join(' · ')+'</span>';
    } else {
      av += '<br><span class="muted">measured perf: absent (no cached hardware profile) &mdash; not guessed.</span>';
    }
    av += '</div>';
  }
  $('advantage').innerHTML = av;
  // MEASURED perf/energy (S2.5 energy module) — preferred over the roofline
  renderMeasured(d.measured);
  // #147 persistent HiCache energy-saved accumulator (read-only here)
  loadHicacheSaved();
  // roofline throughput ESTIMATE — loudly labelled rough ballpark, NOT measured
  renderRoofline(d.roofline_estimate);
  // launch flags
  $('flags').innerHTML = (d.launch_flags && d.launch_flags.length)
    ? '<p class="muted">launch flags (copy into your command):</p><pre>'
      + d.launch_flags.map(esc).join('\n') + '</pre>' : '';
}

// -- SHARED energy->money layer (ct/kWh) --------------------------------------
// Reused by the measured cost below AND (later) the HiCache kWh-saved band
// (#147 -> money saved) and the virtual-rig estimate (#148 -> estimated cost).
// Keep the conversion here, not buried per-metric, so those hooks plug in.
function getKwhPriceCt() {  // ct per kWh, default 30
  const el = $('kwh_price');
  const v = el ? parseFloat(el.value) : NaN;
  return isFinite(v) && v >= 0 ? v : 30;
}
function kwhToCt(kwh, priceCt) {          // kWh * (ct/kWh) -> ct
  if (kwh == null || priceCt == null) return null;
  return kwh * priceCt;
}
function jPerTokToCtPer1M(jPerTok, priceCt) {  // J/token -> ct per 1M tokens
  if (jPerTok == null) return null;
  return kwhToCt(jPerTok / 3.6, priceCt);      // kWh/1M = J/tok / 3.6
}
// Hook for #147 (kWh-saved is a [lo,hi] band): band of kWh -> band of ct.
function kwhBandToCt(band, priceCt) {
  if (!band) return null;
  return band.map(x => kwhToCt(x, priceCt));
}

const _f1 = x => (x==null ? '—' : Math.round(x).toLocaleString());
const _f2 = x => (x==null ? '—' : Number(x).toFixed(2));
const _f3 = x => (x==null ? '—' : Number(x).toFixed(3));
const _fct = x => (x==null ? '—' : Number(x).toFixed(3) + ' ct');

// One self-contained PHASE table (PREFILL or DECODE): tok/s + J/tok + kWh/1M +
// ct/1M per bucket. Prefill and decode are NEVER combined — two separate blocks.
function phaseTable(w, bs, priceCt, phase) {
  const isP = phase === 'prefill';
  const tps = isP ? (w.prefill_tok_s_by_bucket||{}) : (w.decode_tok_s_by_bucket||{});
  const jb  = isP ? (w.j_per_prefill_token_by_bucket||{})
                  : (w.j_per_decode_token_by_bucket||{});
  const kwh = isP ? (w.kwh_per_1m_prefill_by_bucket||{})
                  : (w.kwh_per_1m_decode_by_bucket||{});
  const acc = w.spec_accept_length_by_bucket || {};
  const hasAcc = !isP && Object.keys(acc).length > 0;
  const label = isP ? 'PREFILL' : 'DECODE';
  const sub = isP ? 'compute-bound, amortized over the prompt'
                  : 'memory-bound, per OUTPUT token';
  let h = '<div class="ms-phase ms-' + phase + '"><div class="ms-ph-h">' + label
    + ' <span class="muted">&mdash; ' + sub + '</span></div>';
  h += '<table><tr><th>bs</th><th>tok/s</th><th>J/tok</th><th>kWh/1M</th>'
    + '<th>ct/1M</th>' + (hasAcc ? '<th>accept-len</th>' : '') + '</tr>';
  for (const b of bs) {
    const jt = jb[b];
    h += '<tr><td>' + b + '</td><td>' + _f1(tps[b]) + '</td><td>' + _f3(jt)
      + '</td><td>' + _f2(kwh[b]) + '</td><td>' + _fct(jPerTokToCtPer1M(jt, priceCt))
      + '</td>' + (hasAcc ? '<td>' + _f2(acc[b]) + '</td>' : '') + '</tr>';
  }
  h += '</table><p class="muted">peak ' + label.toLowerCase() + ' <b>'
    + _f1(isP ? w.peak_prefill_tok_s : w.peak_decode_tok_s) + '</b> tok/s</p>';
  h += perCardPhase(w, bs, priceCt, phase);
  h += '</div>';
  return h;
}

// Per-card energy for ONE phase @ the largest bucket: each card's OWN NVML power.
function perCardPhase(w, bs, priceCt, phase) {
  const pce = w.per_card_energy;
  if (!pce || !pce.gpu_names || !pce.gpu_names.length) return '';
  const isP = phase === 'prefill';
  const b = bs[bs.length - 1];
  const names = pce.gpu_names, share = pce.compute_share || [];
  const wl = ((isP ? pce.prefill_watts_by_bucket : pce.decode_watts_by_bucket)||{})[b]||[];
  const jl = ((isP ? pce.prefill_j_per_token_by_bucket
                   : pce.decode_j_per_token_by_bucket)||{})[b]||[];
  const totW = wl.reduce((a,x)=>a+(x||0), 0);
  const totJ = jl.reduce((a,x)=>a+(x||0), 0);
  // work-normalized efficiency: (compute-share / J-per-token), rescaled so the
  // LEAST efficient card = 1.0x. Raw per-card J/tok is ~uniform across cards
  // (lockstep: every card sees the SAME tokens), so it is NOT efficiency — only
  // dividing by each card's actual uneven-DCP work-share reveals the hetero gap.
  const effs = names.map((_,i) => (share[i]!=null && jl[i]) ? share[i]/jl[i] : null);
  const effMin = Math.min(...effs.filter(x=>x!=null && x>0));
  let h = '<p class="muted" style="margin:.3rem 0 .1rem">per-card @ bs=' + b
    + ' (own NVML power, not total/N). <b>J/tok is RAW</b> (lockstep &mdash; same '
    + 'tokens on every card, NOT efficiency); <b>wpw(rel)</b> = work-per-watt '
    + 'normalized by uneven-DCP compute-share = the real hetero-efficiency.</p>';
  h += '<table><tr><th>card</th><th>compute</th><th>power</th><th>W</th>'
    + '<th>J/tok (raw)</th><th>wpw(rel)</th><th>kWh/1M</th><th>ct/1M</th></tr>';
  for (let i = 0; i < names.length; i++) {
    const pw = totW > 0 ? (wl[i]||0)/totW : null;
    const rel = (effs[i]!=null && effMin>0) ? (effs[i]/effMin) : null;
    h += '<tr><td style="text-align:left">' + esc(names[i]) + '</td><td>'
      + (share[i]!=null ? Math.round(share[i]*100)+'%' : '—') + '</td><td>'
      + (pw!=null ? Math.round(pw*100)+'%' : '—') + '</td><td>' + _f1(wl[i])
      + '</td><td>' + _f3(jl[i]) + '</td><td>' + (rel!=null ? rel.toFixed(2)+'&times;' : '—')
      + '</td><td>' + _f2(jl[i]!=null?jl[i]/3.6:null)
      + '</td><td>' + _fct(jPerTokToCtPer1M(jl[i], priceCt)) + '</td></tr>';
  }
  h += '<tr><td style="text-align:left"><b>TOTAL</b></td><td>100%</td><td>100%</td>'
    + '<td>' + _f1(totW) + '</td><td>' + _f3(totJ) + '</td><td>&mdash;</td><td>' + _f2(totJ/3.6)
    + '</td><td>' + _fct(jPerTokToCtPer1M(totJ, priceCt)) + '</td></tr></table>';
  return h;
}

function _bucketsOf(workloads) {
  const s = new Set();
  for (const w of workloads)
    for (const k of Object.keys(w.decode_tok_s_by_bucket||{})) s.add(+k);
  return [...s].sort((a,b)=>a-b);
}

// Peak decode tok/s of a config row (max over workloads/buckets) — for the mult.
function _peakDecode(row) {
  let mx = null;
  for (const w of row.workloads) {
    const v = w.peak_decode_tok_s;
    if (v != null && (mx == null || v > mx)) mx = v;
  }
  return mx;
}

function renderMeasured(mz) {
  window.__lastMeasured = mz;  // cached so a price edit re-renders in place
  // back-compat: older payload had {workloads}; normalize to {rows:[{...}]}.
  let rows = mz && mz.rows;
  if (!rows && mz && mz.workloads)
    rows = [{config_label: 'measured', workloads: mz.workloads}];
  if (!rows || !rows.length) { $('measured').innerHTML = ''; return; }
  const priceCt = getKwhPriceCt();

  let h = '<div class="measured"><div class="ms-title">MEASURED throughput '
    + '&amp; energy &mdash; real run, CUDA graphs ON (S2.5 energy module)</div>'
    + '<p class="ms-note">MEASURED (whole-rig NVML board power integrated over '
    + 'the prefill vs decode window, temp 0). OVERRIDES the roofline below. '
    + 'PREFILL and DECODE are shown as SEPARATE operations (never summed) &mdash; '
    + 'prefill is cheap/compute-amortized, decode is the expensive per-token cost. '
    + 'kWh/1M = J/token &divide; 3.6; cost = kWh &times; ' + priceCt
    + '&nbsp;ct/kWh. <b>Energy = GPU (NVML) power only, summed across cards '
    + '&mdash; EXCLUDES CPU/RAM/PSU losses; NOT wall-socket power.</b></p>';

  // MTP multiplier: fastest non-baseline row's peak decode vs baseline's.
  const base = rows.find(r => /baseline|no-mtp/i.test(r.config_label));
  const spec = rows.find(r => !/baseline|no-mtp/i.test(r.config_label));
  if (base && spec) {
    const pb = _peakDecode(base), ps = _peakDecode(spec);
    if (pb && ps)
      h += '<p class="ms-mult"><b>MTP multiplier:</b> peak decode ' + _f1(ps)
        + ' vs ' + _f1(pb) + ' tok/s &rarr; <b>' + (ps/pb).toFixed(2)
        + '&times;</b> (MTP is faster AND cheaper per output token &mdash; the '
        + 'target forward is amortized across accepted tokens).</p>';
  }

  for (const row of rows) {
    const bs = _bucketsOf(row.workloads);
    h += '<div class="ms-row"><div class="ms-row-h">config: <b>'
      + esc(row.config_label) + '</b> <span class="muted">'
      + esc((row.workloads[0]||{}).tp_config||'') + ' &middot; kv '
      + esc((row.workloads[0]||{}).kv_cache_dtype||'') + '</span></div>';
    for (const w of row.workloads) {
      h += '<div class="ms-wl"><div class="ms-wl-h">workload: <b>'
        + esc(w.workload) + '</b></div><div class="ms-phases">';
      h += phaseTable(w, bs, priceCt, 'prefill');
      h += phaseTable(w, bs, priceCt, 'decode');
      h += '</div></div>';
    }
    h += '</div>';
  }
  h += '<p class="muted">source: ' + esc(mz.store_path) + '</p></div>';
  $('measured').innerHTML = h;
}

// -- #147 persistent HiCache energy-saved band -------------------------------
// The HiCache serves prefill tokens from RAM/disk instead of recomputing them;
// every recovered token is an avoided prefill recompute -> energy saved. The
// saving is a von–bis BAND (J/prefill-token varies by bucket) and NOTHING is
// deducted for the fetch (RAM/disk power is sunk cost). Read-only view of the
// persisted per-(model,config) accumulator + grand total.
async function loadHicacheSaved() {
  try {
    const r = await fetch('/api/hicache_saved?price_ct_per_kwh=' + getKwhPriceCt());
    const d = await r.json();
    renderHicacheSaved(d);
  } catch (e) { /* offline / no store yet -> leave the block empty */ }
}

function _kwhBand(b) {   // [lo,hi] kWh -> "lo–hi kWh" (or em-dash if absent)
  if (!b) return '<span class="muted">no measured J/prefill-token band yet</span>';
  return _f3(b[0]) + '&ndash;' + _f3(b[1]) + ' kWh';
}
function _ctBand(b) {    // [lo,hi] ct -> "lo–hi ct"
  if (!b) return '&mdash;';
  return _f2(b[0]) + '&ndash;' + _f2(b[1]) + ' ct';
}

function renderHicacheSaved(d) {
  window.__lastHicache = d;   // cached so a price edit re-renders in place
  const el = $('hicache_saved');
  if (!el) return;
  // Re-derive the ct band client-side from the kWh band at the CURRENT price so
  // editing the price bar updates instantly without a refetch.
  const priceCt = getKwhPriceCt();
  const recs = (d && d.records) || [];
  if (!recs.length) { el.innerHTML = ''; return; }
  const ctOf = kwhB => (kwhB ? kwhBandToCt(kwhB, priceCt) : null);

  let h = '<div class="measured"><div class="ms-title">HiCache energy SAVED '
    + '&mdash; cumulative, per model/config (#147)</div>'
    + '<p class="ms-note">Prefill tokens served from the RAM/disk prefix cache '
    + 'instead of being recomputed. Each is an <b>avoided prefill recompute</b> '
    + '&rarr; energy saved = recovered&nbsp;tokens &times; J/prefill-token. The '
    + 'fetch itself costs <b>nothing extra</b> (RAM/disk draw is sunk cost), so '
    + 'nothing is deducted. Shown as a <b>von&ndash;bis band</b> (J/prefill-token '
    + 'varies by batch bucket: band = min&ndash;max measured bucket). Accumulates '
    + 'across sessions. kWh = J &divide; 3.6e6; ct = kWh &times; ' + priceCt
    + '&nbsp;ct/kWh.</p>';
  h += '<table><tr><th>model</th><th>config</th><th>recovered prefill tok</th>'
    + '<th>J/prefill-token (band)</th><th>kWh saved (band)</th>'
    + '<th>ct saved (band)</th></tr>';
  for (const r of recs) {
    const jb = r.j_per_prefill_token_band;
    h += '<tr><td style="text-align:left">' + esc(r.model) + '</td>'
      + '<td style="text-align:left">' + esc(r.config_label) + '</td>'
      + '<td>' + _f1(r.recovered_prefill_tokens) + '</td>'
      + '<td>' + (jb ? _f3(jb[0]) + '&ndash;' + _f3(jb[1]) : '<span class="muted">&mdash;</span>')
      + (jb && r.band_provenance ? ' <span class="muted">(' + esc(r.band_provenance) + ')</span>' : '')
      + '</td>'
      + '<td>' + _kwhBand(r.saved_kwh_band) + '</td>'
      + '<td>' + _ctBand(ctOf(r.saved_kwh_band)) + '</td></tr>';
  }
  const gt = (d && d.grand_total) || {};
  h += '<tr><td style="text-align:left"><b>GRAND TOTAL</b></td><td>&mdash;</td>'
    + '<td><b>' + _f1(gt.recovered_prefill_tokens) + '</b></td><td>&mdash;</td>'
    + '<td><b>' + _kwhBand(gt.saved_kwh_band) + '</b></td>'
    + '<td><b>' + _ctBand(ctOf(gt.saved_kwh_band)) + '</b></td></tr>';
  h += '</table><p class="muted">source: ' + esc(d.store_path || '')
    + ' &mdash; persistent, appends across sessions.</p></div>';
  el.innerHTML = h;
}

function renderRoofline(rf) {
  if (!rf) { $('roofline').innerHTML = ''; return; }
  const fmt = x => Math.round(x).toLocaleString();
  let h = '<div class="roofline"><div class="rf-title">Roofline estimate '
    + '(no MTP) &mdash; ROUGH BALLPARK, NOT measured</div>'
    + '<p class="est">The runtime measures the real number; this is an '
    + 'order-of-magnitude ceiling with a large error bar.</p>';
  if (rf.measured_available)
    h += '<p class="est"><b>A measured entry exists for this config (shown '
      + 'above) &mdash; this roofline is secondary.</b></p>';
  h += '<div class="rf-nums">'
    + '<div class="rf-num"><span>prefill</span><b>~' + fmt(rf.prefill_tok_s)
    + '</b> tok/s<small>compute-bound, cold cache</small></div>'
    + '<div class="rf-num"><span>decode</span><b>~' + fmt(rf.decode_tok_s_low)
    + '&ndash;' + fmt(rf.decode_tok_s_high) + '</b> tok/s<small>at '
    + rf.reference_context_tokens.toLocaleString() + '-tok ctx .. short-ctx '
    + 'ceiling</small></div></div>';
  h += '<p class="muted">derivation: compute dtype <b>' + esc(rf.compute_dtype)
    + '</b>; eff decode &times;' + rf.eff_decode + ', prefill &times;'
    + rf.eff_prefill + '; interconnect <b>' + esc(rf.interconnect) + ' &times;'
    + rf.interconnect_discount.toFixed(2) + '</b><br><span class="est">'
    + esc(rf.interconnect_note) + '</span></p>';
  if (rf.offload_note)
    h += '<p class="muted">MoE offload: <span class="est">'
      + esc(rf.offload_note) + '</span></p>';
  h += '<table><tr><th>rank</th><th>GPU</th><th>membw</th><th>FLOPS</th>'
    + '<th>active weight/token</th></tr>';
  for (const pr of rf.per_rank) {
    const src = s => '<span class="'
      + (s === 'measured' ? 'rf-meas' : 'rf-name') + '">' + esc(s) + '</span>';
    h += '<tr><td>' + pr.rank + '</td><td>GPU ' + pr.gpu_index + ' '
      + esc(pr.gpu_name) + '</td><td>' + fmt(pr.peak_membw_gbs) + ' GB/s '
      + src(pr.membw_source) + '</td><td>' + fmt(pr.peak_flops_tflops)
      + ' TFLOPS ' + src(pr.flops_source) + '</td><td>'
      + pr.resident_active_weight_gib.toFixed(2) + ' GiB'
      + (pr.offloaded_active_weight_gib > 0
          ? ' + ' + pr.offloaded_active_weight_gib.toFixed(2) + ' via PCIe' : '')
      + '</td></tr>';
  }
  h += '</table><ul class="rf-caveats">'
    + rf.caveats.map(c => '<li>' + esc(c) + '</li>').join('') + '</ul></div>';
  $('roofline').innerHTML = h;
}

async function doIssue(kind) {
  const p = payload(); p.kind = kind;
  if (kind === 'bug') p.symptom = prompt('What happened? (e.g. OOM at load)') || '(unspecified)';
  const r = await fetch('/api/issue', {method:'POST', body: JSON.stringify(p)});
  const d = await r.json();
  if (!d.ok) { $('issue').innerHTML = '<p class="reasons">issue error: '+esc(d.error)+'</p>'; return; }
  let h = '<hr><p class="muted">'+kind.toUpperCase()+' issue (opt-in &mdash; nothing is sent; copy or click):</p>';
  h += '<pre>'+esc(d.markdown)+'</pre>';
  h += d.url_within_budget
    ? '<p><a href="'+esc(d.url)+'" target="_blank" rel="noopener">Open prefilled GitHub issue &rarr;</a></p>'
    : '<p class="est">(prefilled URL exceeds the ~6 KB budget &mdash; use the copy-paste block above.)</p>';
  $('issue').innerHTML = h;
}

function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}

function showTab(t) {
  for (const v of ['plan','explore','landscape','energy'])
    $('view_'+v).style.display = t===v ? '' : 'none';
  for (const v of ['plan','explore','landscape','energy'])
    $('tab_'+v).classList.toggle('active', t===v);
  if ((t==='explore'||t==='landscape') && !window._profLoaded) loadProfiles();
  if (t==='energy' && !window._energyInit) { window._energyInit=true; refreshGpuState(); }
}

async function loadProfiles() {
  window._profLoaded = true;
  const r = await fetch('/api/profiles'); const d = await r.json();
  $('mx_profiles').innerHTML = '<b>library:</b> ' +
    d.profiles.map(p=>'<span class="pill">'+esc(p.name)+' '+Math.round(p.total_mib/1024)+'GB</span>').join('');
}

function parseMxModels() {
  return $('mx_models').value.split('\n').map(s=>s.trim()).filter(Boolean).map(line=>{
    const i = line.indexOf('=');
    return i>0 ? {label:line.slice(0,i).trim(), model:line.slice(i+1).trim()}
               : {model:line};
  });
}
function parseMxRigs() {
  return $('mx_rigs').value.split('\n').map(s=>s.trim()).filter(Boolean).map(line=>{
    if (line.toLowerCase()==='live') return {name:'live', source:'nvml'};
    const i = line.indexOf('=');
    const name = i>0 ? line.slice(0,i).trim() : line;
    const rest = i>0 ? line.slice(i+1) : line;
    return {name, profiles: rest.split(',').map(s=>s.trim()).filter(Boolean)};
  });
}

async function doMatrix() {
  $('mx_out').innerHTML = 'planning…';
  const body = {models: parseMxModels(), rigs: parseMxRigs()};
  const r = await fetch('/api/matrix', {method:'POST', body: JSON.stringify(body)});
  const d = await r.json();
  if (!d.ok) { $('mx_out').innerHTML = '<p class="reasons">'+esc(d.error)+'</p>'; return; }
  const cellOf = (m,rig)=>d.cells.find(c=>c.model===m && c.rig===rig);
  let h = '<table class="mx"><tr><th>model \\ rig</th>' +
    d.rigs.map(r=>'<th>'+esc(r)+'</th>').join('') + '</tr>';
  for (const m of d.models) {
    h += '<tr><td style="text-align:left">'+esc(m)+'</td>';
    for (const rig of d.rigs) {
      const c = cellOf(m,rig); let inner, cls = c.estimate ? 'estcell ' : '';
      if (!c.fits) { inner = '<span class="nofitc">no fit</span>'; }
      else {
        const ctx = c.max_context_tokens ? '~'+Math.round(c.max_context_tokens/1000)+'k' : '';
        let pct = '';
        if (c.capacity_pct_range) { const [lo,hi]=c.capacity_pct_range; pct = '<br>'+(lo>=0?'+':'')+lo+'..'+hi+'%'; }
        inner = '<span class="fitc">fit'+(c.estimate?'*':'')+'</span> '+ctx+pct;
      }
      h += '<td class="'+cls+'" title="'+esc(c.estimate_note||c.provenance)+'">'+inner+'</td>';
    }
    h += '</tr>';
  }
  h += '</table>';
  h += '<div class="legend">&bull; <b>*</b> / dashed = COMPOSED rig &rarr; '
     + 'ESTIMATE (assumes pcie/nvlink topology; not measured, §8). '
     + 'Solid = real rig (live NVML / declared).<br>&bull; capacity % is vs '
     + 'stock even-TP (ratio of two same-model estimates); no throughput is shown.</div>';
  $('mx_out').innerHTML = h;
}

async function doLandscape() {
  $('ls_out').innerHTML = 'building…';
  const rigs = $('ls_rigs').value.split('\n').map(s=>s.trim()).filter(Boolean).map(line=>{
    const i=line.indexOf('='); const name=i>0?line.slice(0,i).trim():line;
    const rest=i>0?line.slice(i+1):line;
    return {name, profiles: rest.split(',').map(s=>s.trim()).filter(Boolean)};
  });
  const body = {model: $('ls_model').value.trim(), quant: $('ls_quant').value.trim(),
    bucket: $('ls_bucket').value ? parseInt($('ls_bucket').value) : null,
    results_store: $('ls_store').value.trim() || null,
    similar: $('ls_similar').checked, rigs};
  const r = await fetch('/api/landscape', {method:'POST', body: JSON.stringify(body)});
  const d = await r.json();
  if (!d.ok) { $('ls_out').innerHTML = '<p class="reasons">'+esc(d.error)+'</p>'; return; }
  const L = d.landscape;
  let h = '<h3 style="margin:.2rem 0">'+esc(L.model)+' @ '+esc(L.quant)
    + (L.bucket? ' <span class="muted">(efficiency @ batch '+L.bucket+')</span>':'')+'</h3>';
  h += '<table class="mx"><tr><th>rig</th><th>provenance</th><th>fit</th>'
     + '<th>max ctx</th><th>J/dec-tok</th><th>peak dec tok/s</th></tr>';
  for (const c of L.rows) {
    const cls = c.is_measured ? '' : 'estcell ';
    const provtag = c.is_measured ? '<span class="fitc">'+esc(c.provenance)+'</span>'
                                  : '<span class="est">'+esc(c.provenance)+'*</span>';
    const ctx = c.max_context ? '~'+Math.round(c.max_context.value/1000)+'k' : (c.fits?'':'—');
    const jd = c.j_per_decode_token!=null ? c.j_per_decode_token
             : (c.is_measured ? '(no data)' : '—');
    const pd = c.peak_decode ? (c.peak_decode.value+' @ '+esc(c.peak_decode.at)) : '—';
    h += '<tr><td style="text-align:left" title="'+esc((c.config||[]).join(' '))+'">'+esc(c.rig)
       + (c.fits?'':' <span class="nofitc">(no fit)</span>')+'</td>'
       + '<td class="'+cls+'">'+provtag+'</td><td>'+(c.fits?'yes':'NO')+'</td>'
       + '<td>'+ctx+'</td><td>'+jd+'</td><td>'+pd+'</td></tr>';
    if (c.config && c.config.length)
      h += '<tr><td colspan="6" class="legend" style="text-align:left">reproduce: <code>'
         + esc(c.config.join('  ')) + '</code></td></tr>';
  }
  h += '</table><div class="legend">'+esc(L.note)+'</div>';
  $('ls_out').innerHTML = h;
}

// ---- #149 GPU power-state tags ----
async function refreshGpuState() {
  $('gpu_state_out').innerHTML = 'querying NVML…';
  try {
    const r = await fetch('/api/gpu_state'); const d = await r.json();
    if (!d.ok) { $('gpu_state_out').innerHTML = '<span class="reasons">'+esc(d.error)+'</span>'; return; }
    let h = '<table class="mx"><tr><th>#</th><th>card</th><th>W now</th><th>limit</th>'
          + '<th>%TDP</th><th>SM/MEM MHz</th><th>°C</th><th>state</th></tr>';
    for (const c of d.cards) {
      const tag = c.oc_uv==='stock' ? c.oc_uv
                : '<b style="color:#e3a008">'+esc(c.oc_uv)+'</b>';
      h += '<tr><td>'+c.nvml_index+'</td><td style="text-align:left">'+esc(c.name)+'</td>'
        + '<td>'+c.power_watts.toFixed(0)+'</td>'
        + '<td>'+c.power_limit_w.toFixed(0)+'/'+c.default_limit_w.toFixed(0)+'</td>'
        + '<td>'+(c.limit_pct_of_default*100).toFixed(0)+'%</td>'
        + '<td>'+c.sm_clock_mhz+'/'+c.mem_clock_mhz+'</td>'
        + '<td>'+c.temperature_c+'</td><td>'+tag+'</td></tr>';
    }
    h += '</table><div class="legend">non-"stock" = efficiency not directly '
       + 'comparable to a stock card without normalizing the power-limit axis.</div>';
    $('gpu_state_out').innerHTML = h;
  } catch(e) { $('gpu_state_out').innerHTML = '<span class="reasons">'+esc(''+e)+'</span>'; }
}

// ---- #149 live widget (client-side delta rates off /metrics) ----
window._liveTimer = null; window._livePrev = null;
async function liveScrape() {
  const target = $('live_target').value.trim();
  const res = parseFloat($('live_res').value) || 250;
  const r = await fetch('/api/live', {method:'POST', body: JSON.stringify({target, resolution_ms:res})});
  const d = await r.json();
  if (!d.ok) { $('live_out').innerHTML = '<span class="reasons">'+esc(d.error)+'</span>'; return d; }
  const s = d.snapshot; let rates = null;
  if (window._livePrev) {
    const dt = s.t - window._livePrev.t;
    const pfx = dt>0 ? Math.max(0, s.prompt_tokens_total - window._livePrev.prompt_tokens_total)/dt : 0;
    const dec = dt>0 ? Math.max(0, s.generation_tokens_total - window._livePrev.generation_tokens_total)/dt : 0;
    rates = {dt, pfx, dec};
  }
  window._livePrev = s;
  let h = '<b>resolution:</b> '+d.resolution_ms+'ms (floor 30ms)<br>';
  if (rates) {
    h += '<b>prefill:</b> '+rates.pfx.toFixed(1)+' tok/s &nbsp; '
       + '<b>decode:</b> '+rates.dec.toFixed(1)+' tok/s<br>';
  } else { h += '<span class="muted">(first sample — rates on next tick)</span><br>'; }
  h += '<b>MTP accept rate:</b> '+(s.spec_accept_rate*100).toFixed(1)+'% &nbsp; '
     + '<b>adaptive-k:</b> '+s.spec_num_steps+' &nbsp; '
     + '<b>ema accept len:</b> '+s.spec_ema_accept_len.toFixed(2);
  $('live_out').innerHTML = h;
  return d;
}
async function toggleLive() {
  if (window._liveTimer) {
    clearInterval(window._liveTimer); window._liveTimer=null; window._livePrev=null;
    $('live_btn').textContent = 'start live'; return;
  }
  const res = Math.max(30, parseFloat($('live_res').value) || 250);
  window._livePrev = null;
  $('live_btn').textContent = 'stop live';
  await liveScrape();
  window._liveTimer = setInterval(liveScrape, res);
}
async function remeasureNow() { window._livePrev = null; await liveScrape(); }

// ---- #150 scenario builder + cache-flush confirm gate ----
async function previewScenario() {
  $('sc_out').innerHTML = 'expanding…';
  const body = {
    scale: parseFloat($('sc_scale').value)||1.0,
    phases: $('sc_phases').value,
    concurrency: parseInt($('sc_conc').value)||1,
    behaviors: $('sc_behav').value.split(','),
    multiturn: $('sc_multi').checked,
    turns: parseInt($('sc_turns').value)||1,
    cold_prefill: $('sc_cold').checked,
    target_running_server: $('sc_running').checked,
  };
  const r = await fetch('/api/scenario', {method:'POST', body: JSON.stringify(body)});
  const d = await r.json();
  if (!d.ok) { $('sc_out').innerHTML = '<span class="reasons">'+esc(d.error)+'</span>'; return; }
  const w = d.cache_flush_warning;
  let h = '';
  if (w.warn) {
    const cls = w.mandatory ? 'reasons' : 'muted';
    h += '<div class="'+cls+'" style="border:1px solid #e3a008;padding:.4rem;border-radius:6px;margin-bottom:.5rem">'
       + '⚠ '+esc(w.message)
       + (w.mandatory ? '<br><button class="mini" onclick="confirmFlush(true)">Continue (clear cache)</button> '
                       + '<button class="mini secondary" onclick="confirmFlush(false)">Cancel</button>' : '')
       + '</div>';
  }
  h += '<b>'+d.summary.n_units+' run units</b> — prefill '+d.summary.prefill_tokens
     + ' tok / decode '+d.summary.decode_tokens+' tok / conc '+d.summary.concurrency;
  h += '<table class="mx" style="margin-top:.4rem"><tr><th>#</th><th>phase</th><th>behavior</th>'
     + '<th>prompt tok</th><th>decode tok</th><th>turn</th><th>cold</th></tr>';
  d.units.forEach((u,i)=>{ h += '<tr><td>'+(i+1)+'</td><td>'+esc(u.phase)+'</td><td>'+esc(u.behavior)+'</td>'
     + '<td>'+u.prompt_tokens+'</td><td>'+u.decode_tokens+'</td><td>'+u.turn+'</td>'
     + '<td>'+(u.cold?'yes':'—')+'</td></tr>'; });
  h += '</table>';
  $('sc_out').innerHTML = h;
}
function confirmFlush(ok) {
  alert(ok ? 'Confirmed — a real run would now flush the cache and measure.'
           : 'Cancelled — cache preserved, no run started.');
}

loadKnobs();
detectGPUs();
</script>
</body>
</html>
"""
