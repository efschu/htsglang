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
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

__all__ = [
    "discover_knobs",
    "detect_hardware",
    "gguf_options_for",
    "plan_from_payload",
    "issue_from_payload",
    "matrix_from_payload",
    "landscape_from_payload",
    "list_cards",
    "hicache_saved_read",
    "hicache_saved_record",
    "list_models_payload",
    "server_start_payload",
    "server_stop_payload",
    "server_restart_payload",
    "server_status_payload",
    "download_targets_payload",
    "model_download_payload",
    "power_profile_payload",
    "measure_power_payload",
    "quality_run_payload",
    "quality_save_payload",
    "quality_shots_payload",
    "landing_snapshot_payload",
    "detect_endpoint_payload",
    "bench_probe_payload",
    "bench_run_events",
    "share_preview_payload",
    "share_submit_payload",
    "serve",
]


# ===========================================================================
# Hardware auto-detect + GGUF quant picker (design §PART 2 / §PART 1a).
# ===========================================================================


def detect_hardware() -> dict:
    """Auto-detect the local GPUs (NVML/nvidia-smi) for the selectable card
    list. Returns ``{ok, gpus:[{index,name,total_mib,free_mib,cuda_index}],
    host_ram_mib, source, cuda_index_source}`` or ``{ok:False, error}`` on a
    GPU-less host — the UI then shows only the manual/virtual add-card path
    (still fully usable offline).

    ``index`` is the NVML/PCI-bus index (telemetry space); ``cuda_index`` is
    the SAME card's CUDA-order index — the space every engine flag
    (--rank-gpu-id / --base-gpu-id) is interpreted in. ``cuda_index_source``
    is "torch" (exact UUID bridge) or "heuristic" (FASTEST_FIRST emulation;
    the UI must say so), or None when unbridged."""
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
        "cuda_index_source": spec.cuda_index_source,
        "host_ram_mib": spec.host_ram_mib,
        "gpus": [
            {
                "index": g.index,
                "name": g.name,
                "total_mib": g.total_mib,
                "free_mib": g.free_mib,
                "cuda_index": g.cuda_index,
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


#: rank_tp_ratio (and the other rank_*_ratio flags) accept the string sentinels
#: "auto" / "auto-performance" as well as an explicit int tuple; pass those
#: through verbatim instead of int-parsing them into "a,u,t,o".
_RATIO_SENTINELS = ("auto", "auto-performance")


def _int_list(v):
    if v is None or v == "":
        return None
    if isinstance(v, str) and v.strip() in _RATIO_SENTINELS:
        # PLAN path: "auto"/"auto-performance" means "derive the split" -- and
        # deriving splits is exactly what the planner does, so pass None. Never
        # forward the bare string: feasibility.plan() len()s the ratio, and
        # len("auto")==4 / len("auto-performance")==16 produced the absurd
        # '--rank-tp-ratio length (16) must equal --tp-size (3)' rejection.
        # (The LAUNCH path keeps the sentinel via _launch_settings_from_payload.)
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
    are planned; they are re-indexed 0..k-1 -- and those positions ARE the
    ``--rank-gpu-id`` space, i.e. CUDA enumeration order, NOT NVML/nvidia-smi
    order (the client posts detected cards sorted by their bridged
    cuda_index; see planner.device_map). ``virtual`` cards (hypothetical/
    future GPUs the user typed) are treated identically — the whole point of
    an offline planner. Returns ``(HardwareSpec, reserve_mib_list_or_None)``."""
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
        "roofline_energy": (
            dataclasses.asdict(result.roofline_energy)
            if getattr(result, "roofline_energy", None) is not None
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
    if out["measured"] and out.get("roofline_energy"):
        # Likewise the estimated energy yields to a measured energy entry.
        out["roofline_energy"]["measured_available"] = True
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
    # PlanRejected is caught below; without this import the rung raised
    # NameError instead of degrading to an absent row, which is the exact
    # opposite of what the handler was written to do.
    from sglang.srt.planner.feasibility import PlanRejected
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
# S4 explorer: card library + model x rig matrix.
# ===========================================================================


def list_cards() -> dict:
    """The GPU-model card library (design §2.7), for the explorer's rig
    composer."""
    from sglang.srt.planner.card_library import CardLibrary

    lib = CardLibrary()
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
    from sglang.srt.planner.card_library import CardLibrary, compose_rig

    lib = CardLibrary()
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
    from sglang.srt.planner.card_library import CardLibrary, compose_rig
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

    lib = CardLibrary()
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
    """Live per-card power-state tags (#149 Ebene-4 refresh button). Cards
    are NVML-sampled (nvml_index); each row is additionally annotated with
    the card's CUDA-order index via the device_map bridge so the UI can
    label both spaces."""
    from sglang.srt.planner.energy import read_gpu_power_states

    states = read_gpu_power_states()
    cards = [s.to_json() for s in states]
    try:
        from sglang.srt.planner.device_map import device_map

        n2c = device_map().nvml_to_cuda()
        for c in cards:
            c["cuda_index"] = n2c.get(c.get("nvml_index"))
    except Exception:  # pragma: no cover - defensive
        pass
    return {"ok": True, "cards": cards}


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


# ===========================================================================
# Model-Manager tab (#S3 model manager) — discover / supervise / download.
#
# These wire the already-committed control-plane core (server_manager.py). The
# UI is a THIN client: it never builds the sglang argv itself, it POSTs the
# launch knobs and the backend maps them onto ``LaunchSettings`` (which reuses
# the proven ``energy.MeasurementConfig`` argv builder). The supervisor owns
# exactly ONE managed sglang child; a "launch / restart" REPLACES it (the box
# is VRAM-bound). Restart is guarded while a live job is in-flight (is_busy).
# ===========================================================================

#: The single long-lived supervisor the dashboard drives. Lazily created so the
#: module imports GPU-free; tests replace it with a fake via ``_set_supervisor``.
_SUPERVISOR = None


def _supervisor():
    global _SUPERVISOR
    if _SUPERVISOR is None:
        from sglang.srt.planner.server_manager import SglangSupervisor

        _SUPERVISOR = SglangSupervisor()
    return _SUPERVISOR


def _set_supervisor(sup) -> None:
    """Test seam: inject a fake supervisor (never boots a real server)."""
    global _SUPERVISOR
    _SUPERVISOR = sup


def _model_to_json(m) -> dict:
    return {
        "name": m.name,
        "path": m.path,
        "format": m.format,
        "quant_method": m.quant_method,
        "vision": bool(m.vision),
        "size_bytes": int(m.size_bytes or 0),
        "size_gib": round((m.size_bytes or 0) / 2**30, 1),
        "error": m.error,
        "gguf_variants": [
            {
                "quant": v.quant,
                "filename": v.filename,
                "path": v.path,
                "size_bytes": int(v.size_bytes or 0),
                "size_gib": round((v.size_bytes or 0) / 2**30, 1),
            }
            for v in (m.gguf_variants or [])
        ],
    }


def list_models_payload(payload: Optional[dict] = None) -> dict:
    """Enumerate every local model the dashboard can serve (config-authoritative
    quant, gguf variants, vision-capable, size). Pure discovery — no GPU."""
    from sglang.srt.planner.server_manager import (
        discover_models,
        model_roots,
    )

    payload = payload or {}
    extra = payload.get("extra_roots") or None
    roots = list(model_roots()) + list(extra or [])
    try:
        models = discover_models(extra_roots=extra)
    except Exception as e:  # pragma: no cover - defensive
        return {"ok": False, "error": str(e), "roots": roots}
    # ``roots`` is what was scanned; ``roots_present`` says which of them
    # actually exist, so an empty list is diagnosable from the payload alone
    # (a mistyped --model-root shows up as present: false, not as "no models").
    return {
        "ok": True,
        "roots": roots,
        "roots_present": [
            {"path": r, "exists": os.path.isdir(os.path.expanduser(r))}
            for r in roots
        ],
        "count": len(models),
        "models": [_model_to_json(m) for m in models],
    }


def _launch_settings_from_payload(payload: dict):
    """Map the UI launch knobs onto a validated ``LaunchSettings``."""
    from sglang.srt.planner.server_manager import LaunchSettings

    def _int_or_none(v):
        if v in (None, ""):
            return None
        return int(v)

    def _float_or_none(v):
        if v in (None, ""):
            return None
        return float(v)

    def _ints(v):
        if v is None or v == "":
            return None
        if isinstance(v, str):
            if v.strip() in _RATIO_SENTINELS:
                return v.strip()
            return [int(x) for x in v.replace(",", " ").split()]
        return [int(x) for x in v]

    ls = LaunchSettings(
        model_path=(payload.get("model_path") or payload.get("model") or "").strip(),
        format=payload.get("format", "hf"),
        gguf_variant=payload.get("gguf_variant") or None,
        served_model_name=(payload.get("served_model_name") or "model").strip(),
        tp_size=int(payload.get("tp_size") or 1),
        rank_gpu_id=_ints(payload.get("rank_gpu_id")),
        rank_tp_ratio=_ints(payload.get("rank_tp_ratio")),
        rank_gpu_memory_mib=_ints(payload.get("rank_gpu_memory_mib")),
        kv_cache_dtype=payload.get("kv_cache_dtype", "auto"),
        context_length=int(payload.get("context_length") or 8192),
        max_running_requests=int(payload.get("max_running_requests") or 16),
        host=str(payload.get("host") or "127.0.0.1"),
        max_num_seqs=(
            int(payload["max_num_seqs"])
            if payload.get("max_num_seqs") not in (None, "")
            else None
        ),
        spec_mode=payload.get("spec_mode", "off"),
        # Speculative DEPTH is part of the configuration, not a detail: a
        # NEXTN boot is defined by steps / topk / draft-tokens, and dropping
        # them silently produced a differently-configured server than the one
        # that was asked for.
        speculative_num_steps=_int_or_none(payload.get("speculative_num_steps")),
        speculative_eagle_topk=_int_or_none(payload.get("speculative_eagle_topk")),
        speculative_num_draft_tokens=_int_or_none(
            payload.get("speculative_num_draft_tokens")),
        speculative_draft_model_path=(
            payload.get("speculative_draft_model_path") or None),
        mem_fraction_static=_float_or_none(payload.get("mem_fraction_static")),
        # Loader identity -- what makes a GGUF boot a GGUF boot.
        tokenizer_path=payload.get("tokenizer_path") or None,
        load_format=payload.get("load_format") or None,
        quantization=payload.get("quantization") or None,
        rank_auto_reserve_mib=_int_or_none(payload.get("rank_auto_reserve_mib")),
        chat_template=payload.get("chat_template") or None,
        tool_call_parser=payload.get("tool_call_parser") or None,
        reasoning_parser=payload.get("reasoning_parser") or None,
        trust_remote_code=bool(payload.get("trust_remote_code", True)),
        extra_flags=[str(a) for a in (payload.get("extra_flags") or [])],
        vision=bool(payload.get("vision")),
        port=int(payload.get("port") or 30000),
    )
    extra_env = payload.get("env")
    if isinstance(extra_env, dict) and extra_env:
        # Profile launch env (flags.profile_env): SGLANG_UNEVEN_* pair,
        # LD_LIBRARY_PATH, PYTHONPATH, ... -- applied by the supervisor ON TOP
        # of its defaults so a launched profile matches the reference command.
        ls.extra_env = {str(k): str(v) for k, v in extra_env.items()}
    return ls.validate()


def _display_env(settings) -> dict:
    """The launch env echoed back for DISPLAY, with credential-looking values
    redacted (a launch env sometimes carries HF_TOKEN)."""
    from sglang.srt.planner.github_share import _redact_env_value

    env = getattr(settings, "extra_env", None) or {}
    return {k: _redact_env_value(k, str(v)) for k, v in env.items()}


def _drop_flags(argv: List[str], drop: set) -> List[str]:
    """Remove ``drop`` flags (and their values) from a flag list."""
    out: List[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if not tok.startswith("--"):
            out.append(tok)
            i += 1
            continue
        has_value = i + 1 < len(argv) and not argv[i + 1].startswith("--")
        if tok not in drop:
            out.append(tok)
            if has_value:
                out.append(argv[i + 1])
        i += 2 if has_value else 1
    return out


def _argv_from_payload(payload: dict, settings) -> Optional[list]:
    """The optional argv override: an explicit full ``argv`` wins (test seam);
    else a ``profile_argv`` flag list (flags.profile_argv) is prefixed with
    the interpreter so the launched command IS the profile's exact argv."""
    argv = payload.get("argv")
    if argv is not None:
        return list(argv)
    profile_argv = payload.get("profile_argv")
    if not profile_argv:
        return None
    # A profile's argv is a FLAG SET from the catalog; it carries no serving
    # identity (--model-path, --served-model-name, --context-length, ...),
    # which lives in the launch form. Using it alone produced a command sglang
    # rejects outright ("the following arguments are required: --model-path").
    # MERGE: profile flags first (they win on conflict), then every flag from
    # the LaunchSettings command the profile does not already set.
    # SERVING IDENTITY belongs to the launch form, not the profile: a profile
    # pins placeholder values (e.g. --host 127.0.0.1, a default port) and would
    # otherwise silently override what the user typed. A too-large
    # --max-running-requests here is not cosmetic: it OOMs during CUDA-graph
    # capture. So the form wins for these, the profile wins for everything else.
    identity = {
        "--model-path", "--model", "--served-model-name", "--context-length",
        "--max-running-requests", "--host", "--port", "--tokenizer-path",
    }
    prof = [
        str(a) for a in _drop_flags([str(a) for a in profile_argv], identity)
    ]
    prof_flags = {a for a in prof if a.startswith("--")}
    base = list(settings.launch_command())
    head, rest = base[:3], base[3:]  # [python, -m, sglang.launch_server]
    extra: List[str] = []
    i = 0
    while i < len(rest):
        tok = rest[i]
        if not tok.startswith("--"):  # stray value, keep position-safe
            i += 1
            continue
        has_value = i + 1 < len(rest) and not rest[i + 1].startswith("--")
        if tok not in prof_flags:
            extra.append(tok)
            if has_value:
                extra.append(rest[i + 1])
        i += 2 if has_value else 1
    return [*head, *prof, *extra]


def _force_enable_metrics(argv: Optional[list]) -> Optional[list]:
    """Guarantee ``--enable-metrics`` on every server this dashboard boots.

    There is deliberately no opt-out. The flag changes observability only,
    never inference: without it /metrics is absent, and the whole live half of
    this dashboard -- token rates, per-session throughput, MTP acceptance,
    cache-hit, tokens per watt -- has nothing to read. A server booted from
    here is a server that can be watched.

    ``LaunchSettings.launch_command()`` already appends it, so this only has
    to cover the argv OVERRIDE path (an explicit argv, or a profile's exact
    argv), where a caller-supplied command would otherwise decide.
    """
    if argv is None:
        return None
    out = list(argv)
    if "--enable-metrics" not in out:
        out.append("--enable-metrics")
    return out


def server_start_payload(payload: dict) -> dict:
    """Boot the managed sglang server from the UI launch knobs. Builds a
    ``LaunchSettings`` (fail-fast validated) and hands it to the supervisor.
    ``wait_ready`` defaults False so the UI stays responsive and tails the boot
    log; the client polls ``/api/server_status`` for readiness."""
    try:
        settings = _launch_settings_from_payload(payload)
    except (ValueError, TypeError) as e:
        return {"ok": False, "error": str(e)}
    sup = _supervisor()
    if sup.is_running():
        return {
            "ok": False,
            "error": "a server is already running; stop or restart it (a restart "
            "REPLACES the single managed instance).",
            "status": sup.status(),
        }
    argv = _force_enable_metrics(_argv_from_payload(payload, settings))
    try:
        status = sup.start(
            settings,
            wait_ready=bool(payload.get("wait_ready", False)),
            argv=argv,
        )
    except Exception as e:
        return {"ok": False, "error": str(e), "status": sup.status()}
    return {
        "ok": True,
        "launch_command": argv if argv is not None else settings.launch_command(),
        "env_applied": _display_env(settings),
        "status": status,
    }


def server_stop_payload(payload: Optional[dict] = None) -> dict:
    sup = _supervisor()
    try:
        report = sup.stop(wait_vram=bool((payload or {}).get("wait_vram", True)))
    except Exception as e:
        return {"ok": False, "error": str(e), "status": sup.status()}
    return {"ok": True, "report": report, "status": sup.status()}


def server_restart_payload(payload: dict) -> dict:
    """Guarded restart: REPLACES the single managed instance. Refused while a
    live job is in-flight (the supervisor's ``is_busy`` guard)."""
    from sglang.srt.planner.server_manager import SupervisorBusyError

    try:
        settings = _launch_settings_from_payload(payload)
    except (ValueError, TypeError) as e:
        return {"ok": False, "error": str(e)}
    sup = _supervisor()
    argv = _argv_from_payload(payload, settings)
    kwargs = {"wait_ready": bool(payload.get("wait_ready", False))}
    if argv is not None:
        kwargs["argv"] = argv
    try:
        status = sup.restart(settings, **kwargs)
    except SupervisorBusyError as e:
        return {"ok": False, "busy": True, "error": str(e), "status": sup.status()}
    except Exception as e:
        return {"ok": False, "error": str(e), "status": sup.status()}
    return {
        "ok": True,
        "launch_command": argv if argv is not None else settings.launch_command(),
        "env_applied": _display_env(settings),
        "status": status,
    }


def server_status_payload(payload: Optional[dict] = None) -> dict:
    sup = _supervisor()
    try:
        return {"ok": True, "running": sup.is_running(), "status": sup.status()}
    except Exception as e:  # pragma: no cover - defensive
        return {"ok": False, "error": str(e)}


def download_targets_payload(payload: dict) -> dict:
    """Writability probe + (optional) HF repo variant listing for the download
    control. The button is GATED on ``model_root_writable`` — a read-only mount
    disables it. When a ``repo_id`` is given its GGUF quant variants are listed
    for the quant dropdown (no file bytes fetched)."""
    from sglang.srt.planner.server_manager import (
        DEFAULT_MODEL_ROOTS,
        available_downloads,
        model_root_writable,
    )

    payload = payload or {}
    root = (payload.get("root") or "").strip() or os.path.expanduser(
        DEFAULT_MODEL_ROOTS[0]
    )
    writable = model_root_writable(root)
    out = {
        "ok": True,
        "root": root,
        "writable": writable,
        "note": None
        if writable
        else "mount read-only — remount rw to enable downloads (not forced).",
    }
    repo_id = (payload.get("repo_id") or "").strip()
    if repo_id:
        try:
            info = available_downloads(repo_id)
            out["repo_id"] = repo_id
            out["is_gguf"] = info.is_gguf
            out["gguf_variants"] = [
                {"quant": v.quant, "filename": v.filename} for v in info.gguf_variants
            ]
            out["files"] = info.files
        except Exception as e:
            out["repo_error"] = str(e)
    return out


def model_download_payload(payload: dict) -> dict:
    """Pull a model into the mounted model root — ONLY when the root is writable
    (hard PermissionError otherwise). For a GGUF ``quant`` only the one chosen
    quant file is fetched. This hits an EXTERNAL service (Hugging Face); the UI
    must PREVIEW (size) + CONFIRM before calling this."""
    from sglang.srt.planner.server_manager import (
        DEFAULT_MODEL_ROOTS,
        download_model,
    )

    repo_id = (payload.get("repo_id") or "").strip()
    if not repo_id:
        return {"ok": False, "error": "repo_id is required"}
    root = (payload.get("root") or "").strip() or os.path.expanduser(
        DEFAULT_MODEL_ROOTS[0]
    )
    quant = payload.get("quant") or None
    try:
        local = download_model(repo_id, quant=quant, root=root)
    except PermissionError as e:
        return {"ok": False, "writable": False, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "path": local, "repo_id": repo_id, "quant": quant}


# ===========================================================================
# Power-measurement button (#S3 power calibration).
#
# One click -> ``measure_all_cards()`` (each card in its own
# CUDA_VISIBLE_DEVICES=<uuid> subprocess; busy cards are SKIPPED, never
# contended; the result is persisted automatically). WARN before running: it
# briefly loads each free GPU with short micro-benchmarks.
# ===========================================================================


def _card_power_to_json(c) -> dict:
    return {
        "uuid": c.uuid,
        "name": c.name,
        "arch": c.arch,
        "total_mib": c.total_mib,
        "p_idle_w": c.p_idle_w,
        "p_membw_w": c.p_membw_w,
        "p_gemm_w": c.p_gemm_w,
        "membw_gbs": c.membw_gbs,
        "gemm_tflops": c.gemm_tflops,
        "provenance": c.provenance,
        "driver": c.driver,
        "measured_at": c.measured_at,
    }


def power_profile_payload(payload: Optional[dict] = None) -> dict:
    """GET side: the currently-loaded PERSISTED power profile (if any), with its
    provenance/driver — shown before/without a fresh measurement."""
    from sglang.srt.planner import power_calibration

    payload = payload or {}
    path = payload.get("path") or power_calibration.DEFAULT_POWER_PROFILE_PATH
    try:
        profile = power_calibration.load_power_profile(path)
    except Exception as e:  # pragma: no cover - defensive
        return {"ok": False, "error": str(e)}
    cards = [_card_power_to_json(c) for c in profile.values()]
    return {
        "ok": True,
        "path": path,
        "loaded": bool(cards),
        "cards": cards,
        "driver": next((c["driver"] for c in cards if c["driver"]), None),
    }


def measure_power_payload(payload: Optional[dict] = None) -> dict:
    """Run the per-card power calibration (``measure_all_cards``) and return the
    measured table + any skipped-because-busy notes. Persists automatically.
    This TOUCHES the GPUs (short micro-benchmarks); busy cards are skipped by
    the backend, never contended."""
    from sglang.srt.planner import power_calibration

    payload = payload or {}
    kwargs = {}
    if payload.get("path"):
        kwargs["path"] = payload["path"]
    if payload.get("only_uuids"):
        kwargs["only_uuids"] = payload["only_uuids"]
    try:
        result = power_calibration.measure_all_cards(**kwargs)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {
        "ok": True,
        "driver": result.driver,
        "created": result.created,
        "cards": [_card_power_to_json(c) for c in result.cards],
        "skipped": list(result.skipped),
    }


# ===========================================================================
# Quality tab (#S3 chess-SVG quality benchmark).
#
# ONESHOT. The model is called BACKEND-SIDE (never from the browser — CORS + we
# want the raw response's token usage), the SVG is extracted, and
# ``quality_chess.validate`` grades it deterministically. Saving a shot is
# OPTIONAL (a JSONL history the right-hand slider reads).
#
# sglang thinking control (verified against this build's OpenAI protocol):
#   * thinking on/off -> ``chat_template_kwargs`` carries BOTH ``enable_thinking``
#     (qwen3/glm45/nemotron/interns1) and ``thinking`` (deepseek-v3/kimi_k2);
#     different chat templates read different keys, so we set both.
#   * thinking budget -> an INTEGER in the request's ``custom_params`` under
#     ``thinking_budget`` (grammar_manager caps ``grammar.max_think_tokens`` to
#     it). NOTE/uncertainty: the budget is only honored by templates/grammars
#     that expose a </think> stop and is a soft cap; models without a thinking
#     grammar ignore it silently.
# ===========================================================================

#: Where saved quality shots accumulate (JSONL, keyed by model+quant+timestamp).
QUALITY_SHOTS_PATH = os.path.expanduser("~/.cache/sglang/quality_shots.jsonl")

_SVG_EXTRACT_RE = None


def _extract_svg(text: str) -> Optional[str]:
    """Pull the first ``<svg>…</svg>`` block out of a chat response (handles a
    fenced ```svg / ```xml code block or a bare inline SVG)."""
    import re

    global _SVG_EXTRACT_RE
    if _SVG_EXTRACT_RE is None:
        _SVG_EXTRACT_RE = re.compile(r"<svg\b.*?</svg>", re.IGNORECASE | re.DOTALL)
    if not text:
        return None
    m = _SVG_EXTRACT_RE.search(text)
    return m.group(0) if m else None


def _chat_completion(
    endpoint: str,
    model: str,
    prompt: str,
    thinking: bool,
    thinking_budget: Optional[int],
    timeout_s: float = 300.0,
) -> dict:
    """Backend-side OpenAI-compatible chat completion (NEVER called from the
    browser). Returns the parsed JSON response. Factored out so tests inject a
    fake and no network/model is touched."""
    import urllib.request

    base = endpoint.strip().rstrip("/")
    if not base.startswith("http"):
        base = "http://" + base
    if not base.endswith("/chat/completions"):
        base = base + ("/v1/chat/completions" if "/v1" not in base else "/chat/completions")
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        # sglang thinking control (see module comment above).
        "chat_template_kwargs": {"enable_thinking": thinking, "thinking": thinking},
    }
    if thinking and thinking_budget:
        body["custom_params"] = {"thinking_budget": int(thinking_budget)}
    req = urllib.request.Request(
        base,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read())


def quality_run_payload(payload: dict) -> dict:
    """ONESHOT quality run: send ``CHESS_PROMPT`` to the target server BACKEND-
    SIDE, extract the SVG, grade it with ``quality_chess.validate``, and capture
    the token usage. The model is NEVER called from the browser."""
    from sglang.srt.planner import quality_chess

    endpoint = (payload.get("endpoint") or "").strip()
    model = (payload.get("model") or "").strip()
    if not endpoint:
        return {"ok": False, "error": "endpoint (OpenAI-compatible base URL) required"}
    if not model:
        return {"ok": False, "error": "model name required"}
    thinking = bool(payload.get("thinking"))
    budget = payload.get("thinking_budget")
    try:
        budget = int(budget) if budget not in (None, "") else None
    except (TypeError, ValueError):
        budget = None
    try:
        resp = _chat_completion(
            endpoint, model, quality_chess.CHESS_PROMPT, thinking, budget
        )
    except Exception as e:
        return {"ok": False, "error": f"chat completion failed: {e}"}

    try:
        content = resp["choices"][0]["message"]["content"] or ""
    except Exception:
        content = ""
    usage = resp.get("usage") or {}
    tokens = {
        "prompt": usage.get("prompt_tokens"),
        "completion": usage.get("completion_tokens"),
        "total": usage.get("total_tokens"),
    }

    svg = _extract_svg(content)
    if not svg:
        return {
            "ok": True,
            "svg": None,
            "raw": content,
            "verdict": "broken",
            "report": "No <svg>…</svg> block found in the model response.",
            "piece_diff": [],
            "offer_download": True,
            "tokens": tokens,
            "representation": "unparseable",
        }

    res = quality_chess.validate(svg).as_dict()
    return {
        "ok": True,
        "svg": svg,
        "raw": content,
        "verdict": res["verdict"],
        "report": res["report"],
        "piece_diff": res["piece_diff"],
        "highlight_squares": res["highlight_squares"],
        "offer_download": res["offer_download"],
        "representation": res["representation"],
        "render_error": res.get("render_error"),
        "tokens": tokens,
    }


def _quality_shots_path(payload: Optional[dict]) -> str:
    return (payload or {}).get("path") or QUALITY_SHOTS_PATH


def quality_save_payload(payload: dict) -> dict:
    """Append ONE shot to the JSONL history (keyed by model+quant+timestamp).
    Optional — only called when the user toggles 'save this shot'."""
    import time

    path = _quality_shots_path(payload)
    if not (payload.get("save", True)):
        return {"ok": True, "saved": False, "note": "save toggle off"}
    record = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": payload.get("model"),
        "quant": payload.get("quant"),
        "verdict": payload.get("verdict"),
        "tokens": payload.get("tokens"),
        "svg": payload.get("svg"),
        "report": payload.get("report"),
        "config": payload.get("config"),
        "prompt": payload.get("prompt"),
    }
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "saved": True, "path": path}


def quality_shots_payload(payload: Optional[dict] = None) -> dict:
    """Read the saved-shot history for the right-hand slider (newest last)."""
    path = _quality_shots_path(payload)
    shots = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    shots.append(json.loads(line))
                except Exception:
                    continue
    except FileNotFoundError:
        pass
    except Exception as e:  # pragma: no cover - defensive
        return {"ok": False, "error": str(e)}
    return {"ok": True, "path": path, "shots": shots}


def _reference_png_bytes() -> Optional[bytes]:
    """The chess ground-truth reference image, served as a static asset."""
    p = os.path.join(os.path.dirname(__file__), "assets", "quality_chess_reference.png")
    try:
        with open(p, "rb") as f:
            return f.read()
    except Exception:
        return None


# ===========================================================================
# Dashboard v2 (PHASE 2) — Landing live-monitor + Runner (models+planner).
#
# THIN adapters over the three committed backend modules:
#   * placement.py     -> granular per-card placement (running + prospective).
#   * flags.py         -> full flag catalog, resolve (greying/auto-set),
#                         profiles + user ProfileStore.
#   * live_metrics.py  -> running-server + NVML live snapshot for the landing.
# Every heavy computation lives in those modules; these functions only resolve
# the model reference, forward the call, and JSON-shape the result.
# ===========================================================================

#: The delta ``state`` live_metrics.snapshot returns between polls. Kept ONLY
#: module-side for delta math (rates / cache-hit); nothing is persisted to disk
#: and the client owns the 60s ring buffers.
_LANDING_SNAPSHOT_STATE = None

#: Which monitor target the delta state belongs to -- switching targets resets
#: the delta math (counters from different servers must never be subtracted).
_LANDING_TARGET_KEY = None

#: Ports probed for a reachable sglang server when neither an explicit
#: endpoint nor a managed instance exists (the user launches servers BY HAND;
#: the landing page must attach to them regardless of who started them).
#: Deliberately SHORT -- this list runs on every landing poll.
_MONITOR_DETECT_PORTS = (30000, 30001, 30100, 8000)

#: The explicit 'detect' button sweeps the whole documented sglang port range
#: plus the common OpenAI-API ports. Reached only on demand, and gated behind a
#: TCP pre-scan, so the width costs nothing when the ports are closed.
_DETECT_SWEEP_PORTS = tuple(range(30000, 30101)) + (8000, 8080)

#: Cache of the last auto-detected endpoint (re-verified first on each poll so
#: a stable hand-started server is one cheap probe, not a port sweep).
_DETECTED_ENDPOINT = None


def _model_cfg_from_ref(model_ref: str, gguf_choice: Optional[str] = None):
    """Resolve a model reference to (model_cfg_dict, model_path) the SAME way
    the plan path does: ``resolve_model_ref`` -> the cost-model config dict
    (``PerfCostModel._load_config``: config.json for HF, GGUF header for GGUF).
    Pure config read -- no checkpoint bytes, no GPU."""
    from sglang.srt.planner.model import resolve_model_ref
    from sglang.srt.uneven_perf import PerfCostModel

    model_path = resolve_model_ref(model_ref, gguf_choice=gguf_choice or None)
    cfg = PerfCostModel._load_config(model_path)
    return cfg, model_path


def _resolve_model_cfg_from_payload(payload: dict):
    """Return (model_cfg or None, error or None). Accepts an explicit
    ``model_cfg`` dict, else a ``model`` reference resolved to its config."""
    model_cfg = payload.get("model_cfg")
    if isinstance(model_cfg, dict) and model_cfg:
        return model_cfg, None
    ref = payload.get("model")
    if not ref:
        return None, None
    try:
        cfg, _ = _model_cfg_from_ref(ref, payload.get("gguf_choice"))
    except (ValueError, KeyError, OSError) as e:
        return None, f"model: {e}"
    return cfg, None


def placement_payload(payload: dict) -> dict:
    """POST /api/placement {model|model_cfg, flags} -> compute_placement.

    ONE granular renderer feeds BOTH the landing page (the RUNNING config's
    placement) and the runner tab (the PROSPECTIVE placement) -- the caller
    supplies the flag dict from either source; the geometry is identical.

    Extras handled here (webui resolves references; placement stays pure):
      * ``flags.speculative_draft_model_path`` -> resolved to the draft
        model's config dict (external-draft weights segment);
      * CUDA-graph memory: an explicit ``flags.graph_mem_override`` (the
        landing page's LIVE boot-log parse) wins; else the measured-anchor
        store (graphmem, self-populated from the boot logs on disk); else
        placement falls back to the calibrated heuristic;
      * ``stock_compare: true`` -> a SECOND, plain normal-TP placement for
        the side-by-side view (neutral framing; when stock cannot express
        any tp on these cards, the reason is stated instead of numbers).
    """
    from sglang.srt.planner import placement as placementmod

    model_cfg, err = _resolve_model_cfg_from_payload(payload)
    if err:
        return {"ok": False, "error": err}
    if not model_cfg:
        return {"ok": False, "error": "no model or model_cfg given"}
    flags_dict = dict(payload.get("flags") or {})

    # external draft model reference -> config dict for the draft segment.
    draft_ref = flags_dict.get("speculative_draft_model_path")
    if draft_ref and not flags_dict.get("speculative_draft_model_cfg"):
        try:
            dcfg, _ = _model_cfg_from_ref(str(draft_ref))
            flags_dict["speculative_draft_model_cfg"] = dcfg
        except (ValueError, KeyError, OSError):
            pass  # unsizable draft ref: placement notes the absence

    # measured-anchor graph memory (measured-overrides-estimate): only when
    # the caller did not already inject a live parse.
    if not flags_dict.get("graph_mem_override") and payload.get("model"):
        try:
            from sglang.srt.planner import graphmem

            est = graphmem.estimate(
                {
                    "tp_size": flags_dict.get("tp_size") or 1,
                    "max_running_requests": flags_dict.get(
                        "max_running_requests"
                    ),
                },
                {
                    "speculative_algorithm": flags_dict.get(
                        "speculative_algorithm"
                    ),
                    "speculative_num_steps": flags_dict.get(
                        "speculative_num_steps"
                    ),
                    "speculative_num_draft_tokens": flags_dict.get(
                        "speculative_num_draft_tokens"
                    ),
                    "speculative_adaptive": flags_dict.get(
                        "speculative_adaptive"
                    ),
                },
                model_path=str(payload.get("model")),
                kv_cache_dtype=flags_dict.get("kv_cache_dtype"),
            )
            if est.get("provenance") == "measured":
                est["provenance"] = "measured (boot-log anchor)"
                flags_dict["graph_mem_override"] = est
        except Exception:
            pass  # anchor store is an enhancement, never a blocker

    try:
        result = placementmod.compute_placement(model_cfg, flags_dict)
    except Exception as e:  # pragma: no cover - defensive
        return {"ok": False, "error": str(e)}
    out = {"ok": True, "placement": result}

    if payload.get("stock_compare"):
        out["stock"] = _stock_comparison(model_cfg, flags_dict, placementmod)
    return out


def _stock_comparison(model_cfg, flags_dict, placementmod) -> dict:
    """Side-by-side plain normal-TP configuration next to a planned (fork)
    one -- SAME granular card view, its own honest numbers, NEUTRAL wording.
    Picks the largest stock-legal tp <= card count (stock_tp_legal: q % tp
    == 0 and kv % tp == 0 / tp % kv == 0); when none exists, only the rule
    is stated ("stock requires ..."), phrased as a fact, not a verdict."""
    from sglang.srt.planner import flags as flagsmod

    ct = flags_dict.get("card_total_mib") or {}
    cards = sorted((int(k), int(v)) for k, v in ct.items())
    ncards = len(cards) or int(flags_dict.get("tp_size") or 1)
    tried = []
    tp_stock = None
    # candidates: all cards, else the 4- / 2-card subsets. tp=1 is NOT a
    # candidate -- a single-GPU run is not a normal-TP comparison.
    for n in sorted({ncards, 4, 2}, reverse=True):
        if n < 2 or n > ncards:
            continue
        ok, reason = flagsmod.stock_tp_legal(model_cfg, n)
        if ok:
            tp_stock = n
            break
        tried.append(reason)
    if tp_stock is None:
        return {
            "legal": False,
            "tp": None,
            "placement": None,
            "note": (
                "no stock-expressible tp on these cards: "
                + "; ".join(dict.fromkeys(tried))
            ),
        }
    # Largest cards first (identical-VRAM preference mirrors the preset
    # picker); rank i lands on the chosen card's cuda index so the card
    # blocks attribute correctly.
    totals = [t for _, t in cards]
    pick = flagsmod._pick_stock_subset(
        [{"name": "", "total_mib": t, "cuda_index": i} for i, t in cards],
        totals,
        tp_stock,
    ) if cards else None
    if pick is not None:
        positions, cuda_idx, _why = pick
        chosen = [cards[p] for p in positions]
    else:
        cuda_idx = list(range(tp_stock))
        chosen = [(i, 0) for i in cuda_idx]
    chosen_totals = [t for _, t in chosen if t]
    even_budget = min(chosen_totals) if chosen_totals else None
    stock_flags = {
        "tp_size": tp_stock,
        "rank_gpu_id": list(cuda_idx),
        "stock_semantics": True,
        "kv_cache_dtype": flags_dict.get("kv_cache_dtype", "auto"),
        "context_length": flags_dict.get("context_length"),
        "max_running_requests": flags_dict.get("max_running_requests"),
        "speculative_algorithm": flags_dict.get("speculative_algorithm"),
        "speculative_num_draft_tokens": flags_dict.get(
            "speculative_num_draft_tokens"
        ),
        "speculative_num_steps": flags_dict.get("speculative_num_steps"),
        "speculative_adaptive": flags_dict.get("speculative_adaptive"),
        "include_vision": flags_dict.get("include_vision", True),
        "card_total_mib": {str(i): t for i, t in chosen},
    }
    names = flags_dict.get("card_name") or {}
    cn = {
        str(i): names.get(str(i)) or names.get(i)
        for i, _ in chosen
        if (names.get(str(i)) or names.get(i))
    }
    if cn:
        stock_flags["card_name"] = cn
    if even_budget:
        # Stock's even split is sized by the smallest chosen card (same
        # mem fraction everywhere) -- a fact of the even split, not a flaw.
        stock_flags["rank_gpu_memory_mib"] = [even_budget] * tp_stock
    try:
        pl = placementmod.compute_placement(model_cfg, stock_flags)
    except Exception as e:  # pragma: no cover - defensive
        return {"legal": True, "tp": tp_stock, "placement": None,
                "note": f"stock placement failed: {e}"}
    return {
        "legal": True,
        "tp": tp_stock,
        "placement": pl,
        "note": (
            f"plain stock normal-TP, tp={tp_stock}, even split "
            "(sized by the smallest card used); same granular view, its "
            "own numbers."
        ),
    }


def resolve_flags_payload(payload: dict) -> dict:
    """POST /api/resolve_flags {settings, model} -> per-field state JSON.

    Wraps ``flags.resolve`` (model-compat greying, mutual-exclusion disable,
    dependency auto-set, tuple-length errors, and the MACHINE-ENFORCED
    cross-field constraints -- weightless-backend, dcp+spec, rank-gpu-id
    budget rule, hicache hybrid layout, ...). The cross-field violations land
    both on the field (``fields[id].error``) and in a flat ``warnings``
    summary list so the UI can render them as blocking banners, not hover
    text."""
    from sglang.srt.planner import flags as flagsmod

    settings = payload.get("settings")
    if settings is None:
        settings = payload.get("flags")  # tolerate either key
    settings = settings or {}
    model_cfg, _ = _resolve_model_cfg_from_payload(payload)
    try:
        fields = flagsmod.resolve(settings, model_cfg)
    except Exception as e:  # pragma: no cover - defensive
        return {"ok": False, "error": str(e)}
    warnings = [
        {"id": cid, "level": "error", "message": st["error"]}
        for cid, st in fields.items()
        if st.get("error")
    ]
    return {
        "ok": True,
        "fields": fields,
        "warnings": warnings,
    }


def recompute_payload(payload: dict) -> dict:
    """POST /api/recompute -> plan + placement + resolved field states, once.

    Every dimension the fork exposes propagates to every dependent value, and
    all of that arithmetic already exists server-side (feasibility.plan,
    placement.compute_placement, flags.resolve). What did not exist was a way
    to ask for all three about the SAME configuration.

    The UI used to fire ``/api/plan``, ``/api/placement`` and
    ``/api/resolve_flags`` in parallel on every edit. Three calls, three
    independent latencies, no ordering: a slow placement answer could land on
    top of a newer one and leave the panels describing two different
    configurations at once. One call cannot disagree with itself, and it is
    one round trip instead of three.

    ``sections`` selects what to compute -- the simple view asks only for the
    parts it shows. Each section reports its own error rather than failing
    the whole answer, because a rejected plan must still be able to show the
    per-field reasons that explain the rejection.
    """
    payload = payload or {}
    want = payload.get("sections") or ["plan", "placement", "fields"]
    out: Dict[str, Any] = {"ok": True}

    if "plan" in want:
        try:
            out["plan"] = plan_from_payload(payload)
        except Exception as e:  # pragma: no cover - defensive
            out["plan"] = {"valid": False, "reasons": [str(e)]}

    if "placement" in want:
        pl = dict(payload.get("placement_request") or {})
        pl.setdefault("model", payload.get("model"))
        pl.setdefault("model_cfg", payload.get("model_cfg"))
        pl.setdefault("gguf_choice", payload.get("gguf_choice"))
        pl.setdefault("flags", payload.get("flags") or {})
        try:
            out["placement"] = placement_payload(pl)
        except Exception as e:  # pragma: no cover - defensive
            out["placement"] = {"ok": False, "error": str(e)}

    if "fields" in want:
        try:
            out["fields"] = resolve_flags_payload(
                {
                    "settings": payload.get("settings") or payload.get("flags") or {},
                    "model": payload.get("model"),
                    "model_cfg": payload.get("model_cfg"),
                    "gguf_choice": payload.get("gguf_choice"),
                }
            )
        except Exception as e:  # pragma: no cover - defensive
            out["fields"] = {"ok": False, "error": str(e)}

    return out


def _measured_figures() -> Dict[str, str]:
    """Short figures this rig actually measured, for the trade-off tooltips.

    Only findings measured HERE produce a number. A finding carried over from
    another rig, or none at all, leaves the entry absent, and
    ``tooltips.describe`` then says the study has not been run rather than
    quoting somebody else's hardware.
    """
    out: Dict[str, str] = {}
    try:
        from sglang.srt.planner import crossover as crossovermod

        finding = crossovermod.load_finding()
        if finding is not None and finding.provenance == crossovermod.MEASURED_HERE:
            rows = [r for r in finding.break_even_table() if r.get("proposable")]
            if rows:
                best = max(rows, key=lambda r: r.get("prefill_gain_pct") or 0.0)
                out["mlp_crossover"] = (
                    f"{best['prefill_gain_pct']:+.1f}% prefill / "
                    f"{best['decode_cost_pct']:+.1f}% decode at {best['vector']}"
                )
    except Exception:  # pragma: no cover - a missing finding is normal
        pass
    return out


def tooltips_payload(payload: Optional[dict] = None) -> dict:
    """GET /api/tooltips -> every control's what-it-gives / what-it-costs line.

    One registry (``planner/tooltips.py``) feeds the flag surface, the
    templates, the view switch and the simple-view sliders alike, so a
    control's stated cost is written once and cannot drift between the four
    places it is drawn. Served on its own endpoint as well as merged into the
    flag catalog, so a CLI reads exactly the text the UI shows.
    """
    from sglang.srt.planner import tooltips as tipsmod

    return {"ok": True, "tooltips": tipsmod.tooltip_map(_measured_figures())}


def discussion_preview_payload(payload: dict) -> dict:
    """POST /api/discussion_preview -> the exact Markdown, plus whether it could
    be sent. Pure: no network, no token read beyond checking one exists."""
    from sglang.srt.planner import discussion_export as dx

    payload = payload or {}
    try:
        return dx.preview(
            payload.get("data") or {},
            payload.get("bundle") or "bench_system",
            energy_groups=payload.get("energy_groups"),
            target=payload.get("target"),
        )
    except dx.DiscussionError as e:
        return {"ok": False, "error": str(e)}


def discussion_submit_payload(payload: dict) -> dict:
    """POST /api/discussion_submit -> post the previewed Markdown, if armed.

    Gated by design: with no discussion configured this returns the preview
    and reports "no target configured". Nothing is created automatically --
    creating a discussion to post into is a decision for a human, not a side
    effect of pressing a button.
    """
    from sglang.srt.planner import discussion_export as dx

    payload = payload or {}
    try:
        return dx.submit(
            payload.get("data") or {},
            payload.get("bundle") or "bench_system",
            energy_groups=payload.get("energy_groups"),
            target=payload.get("target"),
            confirmed=bool(payload.get("confirmed")),
        )
    except dx.DiscussionError as e:
        # Already token-redacted by the module.
        return {"ok": False, "error": str(e)}


#: Last engine scrape per endpoint, so the lead metrics are a delta between
#: two calls rather than a blocking sleep inside one. Same shape the landing
#: strip uses client-side; kept here because ms-per-round needs the
#: phase-labelled forward-time counter, which only the host-side scraper reads.
_LEAD_PREV: Dict[str, tuple] = {}


def bench_lead_metrics_payload(payload: Optional[dict] = None) -> dict:
    """POST /api/bench_lead_metrics {endpoint} -> ms per round, from the engine.

    ms/verify-round and ms/1k-prefill-tokens are the yardstick, not tok/s:
    they say how long a round actually takes, which is what a change to the
    split or the speculation depth moves. They come from the engine's
    phase-labelled forward-time counter differenced across a window.

    The window is the gap between two calls to this endpoint. Sampling twice
    with a sleep in between would block a request for the length of the
    window, and the caller is already polling.

    A metric the engine did not export is ABSENT, never zero, and ``notes``
    says why -- an empty column has to read as "the device timer was off"
    rather than as "the round took no time".
    """
    from sglang.srt.rigmon.rates import phase_seconds, round_time
    from sglang.srt.rigmon.sources import EngineScraper

    endpoint = _norm_endpoint((payload or {}).get("endpoint") or "")
    if not endpoint:
        return {"ok": False, "error": "no endpoint given"}

    scraper = EngineScraper(endpoint)
    now = time.time()
    try:
        sample = scraper.scrape()
    except Exception as e:
        return {"ok": False, "error": f"scrape of {endpoint} failed: {e}"}
    if not sample.up:
        return {"ok": False, "error": sample.reason or "engine not reachable"}

    prev = _LEAD_PREV.get(endpoint)
    _LEAD_PREV[endpoint] = (now, sample)
    if prev is None:
        return {
            "ok": True,
            "metrics": {},
            "notes": ["First sample: the round times appear on the next poll, "
                      "because they are a delta across a window."],
            "window_s": None,
        }

    prev_t, prev_sample = prev
    dt = max(now - prev_t, 1e-6)
    delta = phase_seconds(sample.per_rank_phase, prev_sample.per_rank_phase)
    notes = []
    if not delta:
        notes.append(
            "No phase-labelled forward time in this window: the round times "
            "are absent, not zero. Boot the server with "
            "SGLANG_ENABLE_METRICS_DEVICE_TIMER=1 (and "
            "--enable-metrics-for-all-schedulers for the per-rank split)."
        )
    rt = round_time(sample.metrics, prev_sample.metrics, delta, dt)
    metrics = {}
    for key in ("ms_per_verify_round", "ms_per_decode_round",
                "ms_per_1k_prefill_tokens", "ms_per_draft_pass",
                "accept_length", "verify_ct"):
        v = getattr(rt, key, None)
        if v is not None:
            metrics[key] = float(v)
    if not metrics and not notes:
        notes.append("Nothing moved in this window: the server was idle.")
    return {
        "ok": True,
        "metrics": metrics,
        "notes": notes,
        "window_s": round(dt, 2),
    }


# ===========================================================================
# Task #214 -- rig pairing. THIN adapters only.
#
# The flow itself lives in rigmon/pairing.py and runs on the host; these
# functions map an HTTP call onto it and JSON-shape the answer. One endpoint
# per step, each taking a small body, so `curl` is a complete client and the
# dashboard has no privileged path. Nothing here computes anything.
# ===========================================================================


def rig_pair_start_payload(payload: dict) -> dict:
    """POST /api/rig_pair/start {target} -> a session, held on the host."""
    from sglang.srt.rigmon import pairing

    target = ((payload or {}).get("target") or "").strip()
    if not target:
        return {
            "ok": False,
            "error": "no target given",
            "remedy": "Pass the far rig's rigmon aggregator as host:port.",
        }
    try:
        s = pairing.STORE.create(target)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "session": s.to_json()}


def rig_pair_advance_payload(payload: dict) -> dict:
    """POST /api/rig_pair/advance {session_id[, step]} -> returns immediately.

    The step is started and the call returns; the caller polls
    ``/api/rig_pair/status``. A slow or unreachable far rig therefore never
    holds an HTTP request open, and a browser reload mid-step loses nothing
    because the state is here rather than in the page.
    """
    from sglang.srt.rigmon import pairing

    payload = payload or {}
    sid = (payload.get("session_id") or "").strip()
    try:
        s = pairing.STORE.advance(sid, payload.get("step") or None)
    except KeyError:
        return {"ok": False, "error": f"no such pairing session: {sid}"}
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "session": s.to_json()}


def rig_pair_status_payload(payload: Optional[dict] = None) -> dict:
    """GET /api/rig_pair/status?session_id=... -> the whole flow state."""
    from sglang.srt.rigmon import pairing

    sid = ((payload or {}).get("session_id") or "").strip()
    if not sid:
        return {
            "ok": True,
            "sessions": [s.to_json() for s in pairing.STORE.list()],
        }
    s = pairing.STORE.get(sid)
    if s is None:
        return {"ok": False, "error": f"no such pairing session: {sid}"}
    return {"ok": True, "session": s.to_json()}


def rig_pair_reset_payload(payload: dict) -> dict:
    """POST /api/rig_pair/reset {session_id} -> clear the results, keep target.

    Retrying after fixing whatever a remedy named is the normal path, so it
    gets its own verb rather than making the caller start a fresh session and
    retype the target.
    """
    from sglang.srt.rigmon import pairing

    sid = ((payload or {}).get("session_id") or "").strip()
    s = pairing.STORE.reset(sid)
    if s is None:
        return {"ok": False, "error": f"no such pairing session: {sid}"}
    return {"ok": True, "session": s.to_json()}


def flag_catalog_payload(payload: Optional[dict] = None) -> dict:
    """GET /api/flag_catalog -> the full flag catalog metadata, grouped, for
    rendering the runner-tab flag surface (help / hover / dropdown options).

    Each entry also carries its ``tradeoff`` line -- what the knob buys and
    what it costs -- resolved from the central registry in ``tooltips.py``,
    plus ``tradeoff_by_value`` for enums whose options pull in opposite
    directions (``rank_kv_ratio`` capacity against speed, say). The UI renders
    what it is given and holds no copy of the text.
    """
    from sglang.srt.planner import flags as flagsmod
    from sglang.srt.planner import tooltips as tipsmod

    measured = _measured_figures()
    cat = flagsmod.catalog()
    groups: Dict[str, List[dict]] = {}
    for cid, spec in cat.items():
        by_value = {}
        for allowed in (spec.allowed or ()):
            txt = tipsmod.describe(f"{spec.id}={allowed}", measurements=measured)
            if txt:
                by_value[str(allowed)] = txt
        groups.setdefault(spec.group, []).append({
            "tradeoff": tipsmod.describe(spec.id, measurements=measured),
            "tradeoff_by_value": by_value,
            "id": spec.id,
            "name": spec.name,
            "type": spec.type,
            "default": spec.default,
            "allowed": list(spec.allowed) if spec.allowed else None,
            "group": spec.group,
            "help": spec.help,
            "hover": spec.hover,
            "source": spec.source,
            "is_env": spec.is_env,
            "tuple_len_flag": spec.tuple_len_flag,
            "mutually_exclusive_with": list(spec.mutually_exclusive_with),
            "requires": list(spec.requires),
        })
    for g in groups:
        groups[g].sort(key=lambda f: (f["source"] != "upstream", f["id"]))
    ordered = {g: groups[g] for g in sorted(groups)}
    return {
        "ok": True,
        "groups": ordered,
        "tooltips": tipsmod.tooltip_map(measured),
        "upstream_count": flagsmod.upstream_count(),
        "fork_count": flagsmod.fork_count(),
    }


def _serves_metrics(base_url: str, timeout: float = 1.0) -> bool:
    """True when the server exposes Prometheus /metrics, i.e. was started with
    ``--enable-metrics``. A detected foreign server without it is a REPORTED
    state, not a dashboard that quietly shows nothing."""
    import urllib.request

    try:
        with urllib.request.urlopen(
            base_url.rstrip("/") + "/metrics", timeout=timeout
        ) as r:
            return r.getcode() == 200
    except Exception:
        return False


def _probe_sglang(base_url: str, timeout: float = 0.8) -> bool:
    """True when ``base_url`` answers 200 on /get_model_info or /metrics --
    the cheap 'is this an sglang server?' reachability probe (read-only)."""
    import urllib.request

    for path in ("/get_model_info", "/metrics"):
        try:
            with urllib.request.urlopen(
                base_url.rstrip("/") + path, timeout=timeout
            ) as r:
                if r.getcode() == 200:
                    return True
        except Exception:
            continue
    return False


def _detect_external_endpoint(
    ports=None, host: str = "127.0.0.1", timeout: float = 0.8
) -> Optional[str]:
    """Auto-detect a reachable sglang server on the common local ports. The
    last hit is cached and re-verified FIRST, so a stable hand-started server
    costs one probe per poll instead of a sweep. Returns a base URL or None."""
    global _DETECTED_ENDPOINT
    candidates: List[str] = []
    if _DETECTED_ENDPOINT:
        candidates.append(_DETECTED_ENDPOINT)
    for p in ports or _MONITOR_DETECT_PORTS:
        url = f"http://{host}:{p}"
        if url not in candidates:
            candidates.append(url)
    for url in candidates:
        if _probe_sglang(url, timeout=timeout):
            _DETECTED_ENDPOINT = url
            return url
    _DETECTED_ENDPOINT = None
    return None


def _tcp_open(host: str, port: int, timeout: float = 0.15) -> bool:
    """True when something accepts a TCP connection on host:port. A closed port
    refuses instantly, so this pre-scan makes a 100-port sweep cheap; only the
    ports that answer get the (much more expensive) HTTP probe."""
    import socket

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _split_host_port(text: str, default_port: int = 30000):
    """'host:port' / 'http://host:port' / 'host' / ':port' -> (host, port)."""
    from urllib.parse import urlsplit

    t = (text or "").strip()
    if not t:
        return None, None
    if "://" not in t:
        t = "http://" + t
    parts = urlsplit(t)
    host = parts.hostname or "127.0.0.1"
    port = parts.port or default_port
    return host, int(port)


def detect_endpoint_payload(payload: Optional[dict] = None) -> dict:
    """GET/POST /api/detect_endpoint -> find a reachable sglang server (the
    landing page's 'detect' button). Read-only probes; never boots anything.

    Accepted payload keys (all optional):
      ``endpoint``  an explicit ``host:port`` to verify directly -- when given,
                    only that one target is probed and no sweep runs;
      ``host``      the host to sweep (default 127.0.0.1);
      ``ports``     an explicit port list, else the full sglang range
                    30000-30100 plus the common OpenAI-API ports.
    """
    global _DETECTED_ENDPOINT
    payload = payload or {}
    explicit = (payload.get("endpoint") or "").strip()
    if explicit:
        host, port = _split_host_port(explicit)
        url = f"http://{host}:{port}"
        ok = _probe_sglang(url, timeout=1.5)
        _DETECTED_ENDPOINT = url if ok else None
        return {
            "ok": True,
            "endpoint": url if ok else None,
            "reachable": [url] if ok else [],
            "probed": [port],
            "host": host,
            "explicit": True,
            "metrics": _serves_metrics(url) if ok else None,
            "error": None if ok else f"no sglang server answering at {url}",
        }

    host = (payload.get("host") or "127.0.0.1").strip() or "127.0.0.1"
    ports = [int(p) for p in (payload.get("ports") or _DETECT_SWEEP_PORTS)]
    # Two stages: a cheap threaded TCP scan, then the HTTP identity probe only
    # on the ports that actually accept a connection.
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=32) as ex:
        open_ports = [
            p for p, is_open in zip(ports, ex.map(lambda p: _tcp_open(host, p), ports))
            if is_open
        ]
    reachable = []
    for p in open_ports:
        url = f"http://{host}:{p}"
        if _probe_sglang(url):
            reachable.append(url)
    _DETECTED_ENDPOINT = reachable[0] if reachable else None
    return {
        "ok": True,
        "endpoint": reachable[0] if reachable else None,
        "reachable": reachable,
        "probed": ports,
        "host": host,
        "tcp_open": open_ports,
        "explicit": False,
        "metrics": _serves_metrics(reachable[0]) if reachable else None,
    }


def landing_snapshot_payload(payload: Optional[dict] = None) -> dict:
    """GET /api/live_snapshot[?endpoint=...] -> one live_metrics.snapshot of
    the resolved MONITOR TARGET + local NVML.

    Target resolution (the landing page is DECOUPLED from the supervisor --
    the user usually launches servers by hand):

      1. an explicit user-entered ``endpoint`` (query param) -- always wins;
      2. the supervisor-managed instance, when one is running;
      3. an auto-detected reachable sglang server on the common local ports.

    Only when NONE of the three exists does the page get the clean
    ``running: False`` placeholder. The response names WHICH target is being
    monitored and whether it is managed or external; an external target has
    no LaunchSettings, so live_metrics falls back to /get_server_info for the
    start config. The delta ``state`` is kept module-side only (never
    persisted) and is reset whenever the target changes."""
    from sglang.srt.planner import live_metrics

    global _LANDING_SNAPSHOT_STATE, _LANDING_TARGET_KEY
    payload = payload or {}
    explicit = (payload.get("endpoint") or "").strip()

    sup = _supervisor()
    try:
        running_managed = sup.is_running()
    except Exception:
        running_managed = False
    settings = getattr(sup, "settings", None)
    try:
        status = sup.status()
    except Exception:
        status = None

    target = None
    kind = None
    label = None
    if explicit:
        label = explicit if explicit.startswith("http") else "http://" + explicit
        target = label
        kind = "explicit"
    elif running_managed and settings is not None:
        target = sup
        kind = "managed"
        host = getattr(settings, "host", "127.0.0.1") or "127.0.0.1"
        label = f"http://{host}:{getattr(settings, 'port', '')}"
    else:
        det = _detect_external_endpoint()
        if det:
            target = det
            kind = "detected"
            label = det

    if target is None:
        _LANDING_SNAPSHOT_STATE = None  # reset delta math for the next target
        _LANDING_TARGET_KEY = None
        return {
            "ok": True,
            "running": False,
            "snapshot": None,
            "target": None,
            "status": status,
        }

    key = (kind, label)
    if key != _LANDING_TARGET_KEY:
        # Never subtract counters from two different servers.
        _LANDING_SNAPSHOT_STATE = None
        _LANDING_TARGET_KEY = key
    target_info = {
        "endpoint": label,
        "kind": kind,
        "managed": kind == "managed",
    }
    try:
        snap, new_state = live_metrics.snapshot(
            target, prev_state=_LANDING_SNAPSHOT_STATE
        )
        _LANDING_SNAPSHOT_STATE = new_state
    except Exception as e:  # pragma: no cover - defensive
        return {"ok": False, "running": True, "error": str(e),
                "target": target_info}
    return {
        "ok": True,
        "running": True,
        "snapshot": snap,
        "target": target_info,
        "status": status,
        # LIVE CUDA-graph memory of the running server: managed boot-log
        # parse first (per-kind + per-ladder-rung itemization), then the
        # orchestrator's conventional /tmp log for detected servers, then
        # /get_server_info's memory_usage.graph; honest null otherwise.
        "graph_capture": _live_graph_capture(
            sup if kind == "managed" else None, label, snap
        ),
    }


def _live_graph_capture(sup, label, snap) -> dict:
    """Measured CUDA-graph MiB of the RUNNING server, best source first:

      1. managed supervisor boot log (full per-capture lines: target
         decode / prefill / verify + every draft ladder rung separately);
      2. the orchestrator's conventional ``/tmp/sglang_boot_<port>.log``
         for a detected external server (same parse);
      3. ``/get_server_info`` ``internal_states[].memory_usage.graph``
         (rank-0 target graphs only, GB -- coarser but external-safe);
      4. none of those -> ``source: None`` with the honest reason.
    """
    from sglang.srt.planner import graphmem

    log_path = None
    if sup is not None:
        log_path = getattr(sup, "_log_path", None)
    if not log_path and label:
        m = re.search(r":(\d+)$", str(label).rstrip("/"))
        if m:
            cand = f"/tmp/sglang_boot_{m.group(1)}.log"
            if os.path.exists(cand):
                log_path = cand
    if log_path and os.path.exists(log_path):
        try:
            with open(log_path, errors="replace") as f:
                text = f.read()
            entries = graphmem.parse_capture_lines(text)
            if entries:
                summary = graphmem.summarize_captures(entries)
                return {
                    "source": "boot-log",
                    "log_path": log_path,
                    "summary": summary,
                }
        except OSError:
            pass
    si = ((snap or {}).get("server_info") or {}).get("server_info") or {}
    states = si.get("internal_states") or []
    for st in states:
        mu = (st or {}).get("memory_usage") or {}
        if mu.get("graph") is not None:
            return {
                "source": "server_info",
                "total_mib": float(mu["graph"]) * 1024.0,
                "note": "target graphs, scheduler rank 0 "
                "(/get_server_info memory_usage.graph)",
            }
    return {
        "source": None,
        "reason": "n/a (external server without a boot log or "
        "/get_server_info graph memory)",
    }


def _profile_store():
    """The user ProfileStore, honoring an env override so tests can point it at
    a temp file (the default is ~/.cache/sglang/planner_profiles.json)."""
    from sglang.srt.planner import flags as flagsmod

    path = os.environ.get("SGLANG_PLANNER_PROFILES") or flagsmod.DEFAULT_STORE_PATH
    return flagsmod.ProfileStore(path)


def config_profiles_get(payload: Optional[dict] = None) -> dict:
    """GET /api/config_profiles -> generated (flags.profiles) + user-saved
    (ProfileStore) config profiles for the runner-tab picker.

    NOTE: distinct from ``/api/cards`` (the GPU-model catalogue the
    explore/landscape tabs use) -- these are full flag-set config profiles."""
    from sglang.srt.planner import flags as flagsmod

    payload = payload or {}
    model_cfg, _ = _resolve_model_cfg_from_payload(payload)
    gpus = payload.get("gpus")
    if not gpus:
        det = detect_hardware()
        gpus = det.get("gpus") or []

    def _prof_json(p) -> dict:
        # Enrich every profile with its EXACT launch surface: the argv
        # (flags.profile_argv) and the COMPLETE env mapping (flags.profile_env
        # = active env-typed settings + the profile's extra env). The runner
        # tab displays both and passes them through the launch path.
        d = p.to_json()
        try:
            d["argv"] = flagsmod.profile_argv(p)
        except Exception:  # pragma: no cover - defensive
            d["argv"] = None
        try:
            d["launch_env"] = flagsmod.profile_env(p)
        except Exception:  # pragma: no cover - defensive
            d["launch_env"] = dict(p.env or {})
        return d

    # Draft-model candidates for the selected base model (speculative
    # decoding WITHOUT an own MTP head): matched against the local model
    # inventory. Also returned to the UI, which feeds the Speculative
    # section's draft selector + no-MTP hint from them.
    has_mtp = flagsmod.model_has_mtp(model_cfg) if model_cfg else None
    draft_candidates: List[dict] = []
    if payload.get("model"):
        try:
            from sglang.srt.planner.server_manager import discover_models

            draft_candidates = flagsmod.find_draft_models(
                payload["model"], discover_models()
            )
        except Exception:  # pragma: no cover - defensive
            draft_candidates = []

    # The generator's capacity rules (context_length / max_running_requests)
    # size KV via feasibility.plan and need the checkpoint path -- pass the
    # selected model ref through as the base model_path (the rules degrade
    # to the form defaults, with an info note, when it cannot be sized).
    base = {"model_path": payload["model"]} if payload.get("model") else None

    # This rig's measured MLP-split crossover (or None): the max-perf preset
    # states the knee points / their absence, and applies a vector only when
    # a LOCAL usable finding selects one for a given prompt:output mix.
    # profiles() itself never touches the filesystem — the load lives here.
    try:
        from sglang.srt.planner.crossover import load_finding

        knee_finding = load_finding()
    except Exception:  # pragma: no cover - defensive
        knee_finding = None
    p2o = payload.get("prompt_to_output_ratio")
    try:
        p2o = float(p2o) if p2o is not None else None
    except (TypeError, ValueError):
        p2o = None

    generated: List[dict] = []
    try:
        generated = [
            _prof_json(p)
            for p in flagsmod.profiles(
                model_cfg, gpus, base=base,
                draft_models=draft_candidates or None,
                crossover_finding=knee_finding,
                prompt_to_output_ratio=p2o,
            )
        ]
    except Exception as e:  # pragma: no cover - defensive
        generated = []
        gen_error = str(e)
    else:
        gen_error = None
    try:
        saved = [_prof_json(p) for p in _profile_store().load_all()]
    except Exception:
        saved = []
    return {
        "ok": True,
        "generated": generated,
        "saved": saved,
        "gen_error": gen_error,
        "upstream_count": flagsmod.upstream_count(),
        "fork_count": flagsmod.fork_count(),
        "model_has_mtp": has_mtp,
        "draft_candidates": draft_candidates,
    }


def config_profiles_save(payload: dict) -> dict:
    """POST /api/config_profiles -> save the current settings as a named user
    profile (ProfileStore)."""
    from sglang.srt.planner import flags as flagsmod

    name = (payload.get("name") or "").strip()
    if not name:
        return {"ok": False, "error": "a profile name is required"}
    settings = payload.get("settings")
    if not isinstance(settings, dict) or not settings:
        return {"ok": False, "error": "a non-empty settings dict is required"}
    env = payload.get("env")
    prof = flagsmod.Profile(
        name=name,
        kind=payload.get("kind", "custom"),
        settings=settings,
        info=list(payload.get("info") or []),
        env=(
            {str(k): str(v) for k, v in env.items()}
            if isinstance(env, dict)
            else {}
        ),
    )
    try:
        _profile_store().save(prof, overwrite=bool(payload.get("overwrite", True)))
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "name": name}


def config_profiles_delete(payload: dict) -> dict:
    """DELETE /api/config_profiles -> remove a user-saved profile by name."""
    name = (payload.get("name") or "").strip()
    if not name:
        return {"ok": False, "error": "a profile name is required"}
    try:
        deleted = _profile_store().delete(name)
    except Exception as e:  # pragma: no cover - defensive
        return {"ok": False, "error": str(e)}
    return {"ok": deleted, "name": name, "deleted": deleted}


# ===========================================================================
# #151 -- Benchmark/quality suite routes (thin adapters over bench_suite.py).
# BACKEND-DRIVEN by design: every model request runs server-side (CORS + the
# raw responses are needed for the verifiers); the browser only renders the
# streamed per-test results. This layer never boots or restarts a server.
# ===========================================================================


def _norm_endpoint(v: Optional[str]) -> str:
    v = (v or "").strip()
    if v and not v.startswith("http"):
        v = "http://" + v
    return v


def bench_probe_payload(payload: Optional[dict] = None) -> dict:
    """POST /api/bench_probe {endpoint?, force?} -> the capability/gating
    state + the full test catalog with each test's gate decision. Without an
    endpoint only the catalog + presets are returned (nothing probed)."""
    from sglang.srt.planner import bench_suite

    payload = payload or {}
    endpoint = _norm_endpoint(payload.get("endpoint"))
    caps = None
    probe_error = None
    if endpoint:
        try:
            caps = bench_suite.probe_capabilities(endpoint)
        except Exception as e:  # pragma: no cover - defensive
            probe_error = str(e)
    tests = []
    for tid in sorted(bench_suite.TEST_CATALOG):
        spec = bench_suite.TEST_CATALOG[tid]
        gd = bench_suite.gate_test(spec, caps, force=bool(payload.get("force")))
        tests.append({
            "test_id": tid,
            "key": spec.key,
            "label": spec.label,
            "optional": spec.optional,
            "crash_prone": spec.crash_prone,
            "deps": spec.deps_json(),
            "gate_status": gd.status,      # None -> runnable
            "gate_reason": gd.reason,
            "expected_fail_note": gd.expected_fail_note,
            "rungs": gd.rungs,
        })
    return {
        "ok": True,
        "endpoint": endpoint or None,
        "probe_error": probe_error,
        "capabilities": caps.to_json() if caps else None,
        "tests": tests,
        "presets": {k: list(v) for k, v in bench_suite.PRESETS.items()},
    }


def bench_run_events(payload: dict):
    """Iterator of SSE event dicts for one suite run: ``start``, one
    ``result`` per finished test (bench_suite.run_suite is itself an
    iterator), then ``done`` with the status counts. The HTTP handler streams
    each event the moment it exists."""
    from sglang.srt.planner import bench_suite

    payload = payload or {}
    endpoint = _norm_endpoint(payload.get("endpoint"))
    model = (payload.get("model") or "").strip()
    if not endpoint:
        yield {"event": "error", "error": "endpoint is required"}
        return
    caps = None
    if isinstance(payload.get("capabilities"), dict):
        try:
            caps = bench_suite.Capabilities(**payload["capabilities"])
        except TypeError as e:
            yield {"event": "error", "error": f"bad capabilities: {e}"}
            return
    selected = payload.get("selected")
    preset = payload.get("preset")
    yield {
        "event": "start",
        "endpoint": endpoint,
        "model": model or None,
        "selected": list(selected) if selected else None,
        "preset": preset,
    }
    counts: dict = {}
    results: List[dict] = []
    # run_suite records every model exchange on its context; the sink hands it
    # back so the finished run can be stored WITH what was asked and answered.
    sink: List[Any] = []
    started = time.time()
    try:
        for res in bench_suite.run_suite(
            endpoint,
            model,
            selected=selected,
            capabilities=caps,
            preset=preset,
            force=bool(payload.get("force")),
            transcript_sink=sink,
        ):
            counts[res.get("status")] = counts.get(res.get("status"), 0) + 1
            results.append(res)
            yield {"event": "result", "result": res}
    except Exception as e:  # pragma: no cover - defensive
        _save_bench_run(endpoint, model, preset, caps, results, sink, started,
                        error=str(e))
        yield {"event": "error", "error": str(e)}
        return
    run_id = _save_bench_run(endpoint, model, preset, caps, results, sink,
                             started)
    yield {"event": "done", "counts": counts, "run_id": run_id}


def _save_bench_run(endpoint, model, preset, caps, results, sink, started,
                    error=None) -> Optional[str]:
    """Persist one finished (or crashed) run to the per-model history.

    A run that ends in an exception is stored too: the transcript up to the
    failure is usually the only record of what killed it. Storing must never
    take the run down with it, so a write failure is reported in the payload
    rather than raised at the client.
    """
    from sglang.srt.planner import bench_history

    transcript = []
    for ctx in sink:
        transcript.extend(getattr(ctx, "transcript", []) or [])
    record = {
        "endpoint": endpoint,
        "model": model or None,
        "preset": preset,
        "started_at": started,
        "duration_s": round(time.time() - started, 3),
        "capabilities": (
            dataclasses.asdict(caps) if dataclasses.is_dataclass(caps) else None
        ),
        "results": results,
        "transcript": transcript,
        "error": error,
    }
    try:
        return bench_history.save_run(record)
    except Exception:  # pragma: no cover - defensive
        return None


def bench_history_payload(payload: Optional[dict] = None) -> dict:
    """GET /api/bench_history[?model=...] -> newest-first run summaries."""
    from sglang.srt.planner import bench_history

    payload = payload or {}
    model = (payload.get("model") or "").strip() or None
    try:
        limit = int(payload.get("limit") or 50)
    except (TypeError, ValueError):
        limit = 50
    try:
        runs = bench_history.list_runs(model=model, limit=limit)
    except Exception as e:  # pragma: no cover - defensive
        return {"ok": False, "error": str(e), "runs": []}
    return {
        "ok": True,
        "model": model,
        "root": bench_history.history_root(),
        "runs": runs,
    }


def bench_run_payload(payload: Optional[dict] = None) -> dict:
    """GET /api/bench_run_detail?run_id=... -> one stored run, transcript and
    all. This is what the download button fetches."""
    from sglang.srt.planner import bench_history

    run_id = ((payload or {}).get("run_id") or "").strip()
    if not run_id:
        return {"ok": False, "error": "run_id is required"}
    rec = bench_history.load_run(run_id)
    if rec is None:
        return {"ok": False, "error": f"no stored run {run_id!r}"}
    return {"ok": True, "run": rec}


# ===========================================================================
# #152 -- GitHub results sharing (thin adapters over github_share.py).
# Preview-first + explicit confirm; the PAT is per-use, never persisted,
# never logged, never echoed, and redacted from every error message.
# ===========================================================================


def share_preview_payload(payload: dict) -> dict:
    """POST /api/share_preview -> the EXACT markdown that would be posted
    (github_share.build_report). Pure render; sends nothing."""
    from sglang.srt.planner import github_share

    payload = payload or {}
    body = payload.get("payload")
    if not isinstance(body, dict):
        body = payload
    try:
        report = github_share.build_report(body or {})
    except Exception as e:  # pragma: no cover - defensive
        return {"ok": False, "error": str(e)}
    return {
        "ok": True,
        "report": report,
        "default_repo": github_share.DEFAULT_REPO,
    }


def share_submit_payload(payload: dict) -> dict:
    """POST /api/share_submit {report, token, repo?, existing_issue?,
    confirmed} -> create-or-update the user's share issue. Refuses without
    ``confirmed`` (github_share.submit performs no network call in that
    case). The token is never stored and never appears in any response --
    every error string has passed github_share.redact."""
    from sglang.srt.planner import github_share

    payload = payload or {}
    token = payload.get("token") or ""
    report = payload.get("report") or ""
    if not report.strip():
        return {"ok": False,
                "error": "no report: build + confirm the preview first"}
    existing = payload.get("existing_issue")
    try:
        existing = int(existing) if existing not in (None, "") else None
    except (TypeError, ValueError):
        return {"ok": False, "error": "existing_issue must be an issue number"}
    try:
        out = github_share.submit(
            report,
            token,
            repo=(payload.get("repo") or "").strip() or github_share.DEFAULT_REPO,
            existing_issue=existing,
            confirmed=bool(payload.get("confirmed")),
        )
    except github_share.GitHubShareError as e:
        return {"ok": False, "error": str(e)}  # message is token-redacted
    except Exception as e:  # pragma: no cover - defensive
        return {"ok": False, "error": github_share.redact(str(e), token)}
    return {"ok": True, **out}


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
        if self.path.startswith("/api/cards"):
            try:
                self._json(200, list_cards())
            except Exception as e:  # pragma: no cover - defensive
                self._json(500, {"error": str(e)})
            return
        if self.path.startswith("/api/detect_endpoint"):
            # MUST precede the /api/detect prefix check below.
            try:
                from urllib.parse import parse_qs, urlsplit

                q = {k: v[0] for k, v in parse_qs(urlsplit(self.path).query).items()}
                if q.get("ports"):
                    q["ports"] = [
                        int(x) for x in q["ports"].replace(",", " ").split()
                    ]
                self._json(200, detect_endpoint_payload(q))
            except Exception as e:  # pragma: no cover - defensive
                self._json(500, {"ok": False, "error": str(e)})
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
        if self.path.startswith("/api/models"):
            try:
                self._json(200, list_models_payload())
            except Exception as e:  # pragma: no cover - defensive
                self._json(500, {"ok": False, "error": str(e)})
            return
        if self.path.startswith("/api/live_snapshot"):
            try:
                from urllib.parse import parse_qs, urlsplit

                q = {k: v[0] for k, v in parse_qs(urlsplit(self.path).query).items()}
                self._json(200, landing_snapshot_payload(q))
            except Exception as e:  # pragma: no cover - defensive
                self._json(500, {"ok": False, "error": str(e)})
            return
        if self.path.startswith("/api/rig_pair/status"):
            try:
                from urllib.parse import parse_qs, urlsplit

                q = {k: v[0] for k, v in parse_qs(urlsplit(self.path).query).items()}
                self._json(200, rig_pair_status_payload(q))
            except Exception as e:  # pragma: no cover - defensive
                self._json(500, {"ok": False, "error": str(e)})
            return
        if self.path.startswith("/api/tooltips"):
            try:
                self._json(200, tooltips_payload())
            except Exception as e:  # pragma: no cover - defensive
                self._json(500, {"ok": False, "error": str(e)})
            return
        if self.path.startswith("/api/flag_catalog"):
            try:
                self._json(200, flag_catalog_payload())
            except Exception as e:  # pragma: no cover - defensive
                self._json(500, {"ok": False, "error": str(e)})
            return
        if self.path.startswith("/api/config_profiles"):
            try:
                from urllib.parse import parse_qs, urlsplit

                q = {k: v[0] for k, v in parse_qs(urlsplit(self.path).query).items()}
                self._json(200, config_profiles_get(q))
            except Exception as e:  # pragma: no cover - defensive
                self._json(500, {"ok": False, "error": str(e)})
            return
        if self.path.startswith("/api/server_status"):
            try:
                self._json(200, server_status_payload())
            except Exception as e:  # pragma: no cover - defensive
                self._json(500, {"ok": False, "error": str(e)})
            return
        if self.path.startswith("/api/power_profile"):
            try:
                self._json(200, power_profile_payload())
            except Exception as e:  # pragma: no cover - defensive
                self._json(500, {"ok": False, "error": str(e)})
            return
        if self.path.startswith("/api/bench_history"):
            try:
                from urllib.parse import parse_qs, urlsplit

                q = {k: v[0] for k, v in parse_qs(urlsplit(self.path).query).items()}
                self._json(200, bench_history_payload(q))
            except Exception as e:  # pragma: no cover - defensive
                self._json(500, {"ok": False, "error": str(e)})
            return
        if self.path.startswith("/api/bench_run_detail"):
            # Served as a DOWNLOAD: the point of keeping the transcript is
            # being able to take the whole request/answer series away.
            try:
                from urllib.parse import parse_qs, urlsplit

                q = {k: v[0] for k, v in parse_qs(urlsplit(self.path).query).items()}
                d = bench_run_payload(q)
                body = json.dumps(d, ensure_ascii=False, indent=1)
                if d.get("ok") and q.get("download"):
                    data = body.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header(
                        "Content-Disposition",
                        'attachment; filename="bench-%s.json"'
                        % (q.get("run_id") or "run"),
                    )
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                self._send(200, body, "application/json")
            except Exception as e:  # pragma: no cover - defensive
                self._json(500, {"ok": False, "error": str(e)})
            return
        if self.path.startswith("/api/quality_shots"):
            try:
                self._json(200, quality_shots_payload())
            except Exception as e:  # pragma: no cover - defensive
                self._json(500, {"ok": False, "error": str(e)})
            return
        if self.path.startswith("/assets/"):
            name = self.path[len("/assets/"):].split("?", 1)[0]
            if name == "quality_chess_reference.png":
                data = _reference_png_bytes()
                if data is not None:
                    self._send(200, data, "image/png")
                    return
            self._send(404, "not found", "text/plain")
            return
        self._send(404, "not found", "text/plain")

    def _sse_stream(self, events) -> None:
        """Stream an iterator of event dicts as Server-Sent Events (one
        ``data:`` frame per event, flushed immediately)."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            for ev in events:
                self.wfile.write(
                    ("data: " + json.dumps(ev) + "\n\n").encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # client went away mid-run; the suite iterator stops with us
        except Exception as e:  # pragma: no cover - defensive
            try:
                self.wfile.write(
                    ("data: " + json.dumps(
                        {"event": "error", "error": str(e)}) + "\n\n").encode())
                self.wfile.flush()
            except Exception:
                pass

    def do_POST(self):
        try:
            payload = self._read_json()
        except Exception as e:
            self._json(400, {"error": f"bad json: {e}"})
            return
        if self.path.startswith("/api/bench_run"):
            # streamed (SSE), not a one-shot JSON body.
            self._sse_stream(bench_run_events(payload))
            return
        try:
            # Detect is reachable by BOTH methods: GET (no body) for the plain
            # button, POST for a body-carrying probe (explicit host:port, custom
            # port list). The prefix order mirrors do_GET -- the longer
            # "/api/detect_endpoint" must be tested first.
            if self.path.startswith("/api/detect_endpoint"):
                self._json(200, detect_endpoint_payload(payload))
                return
            if self.path.startswith("/api/detect"):
                self._json(200, detect_hardware())
                return
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
            if self.path.startswith("/api/scenario"):
                self._json(200, scenario_payload(payload))
                return
            if self.path.startswith("/api/cache_flush_warning"):
                self._json(200, cache_flush_warning_payload(payload))
                return
            if self.path.startswith("/api/hicache_saved"):
                self._json(200, hicache_saved_record(payload))
                return
            if self.path.startswith("/api/server_start"):
                self._json(200, server_start_payload(payload))
                return
            if self.path.startswith("/api/server_stop"):
                self._json(200, server_stop_payload(payload))
                return
            if self.path.startswith("/api/server_restart"):
                self._json(200, server_restart_payload(payload))
                return
            if self.path.startswith("/api/download_targets"):
                self._json(200, download_targets_payload(payload))
                return
            if self.path.startswith("/api/model_download"):
                self._json(200, model_download_payload(payload))
                return
            if self.path.startswith("/api/measure_power"):
                self._json(200, measure_power_payload(payload))
                return
            if self.path.startswith("/api/quality_run"):
                self._json(200, quality_run_payload(payload))
                return
            if self.path.startswith("/api/quality_save"):
                self._json(200, quality_save_payload(payload))
                return
            if self.path.startswith("/api/placement"):
                self._json(200, placement_payload(payload))
                return
            if self.path.startswith("/api/resolve_flags"):
                self._json(200, resolve_flags_payload(payload))
                return
            if self.path.startswith("/api/discussion_preview"):
                self._json(200, discussion_preview_payload(payload))
                return
            if self.path.startswith("/api/discussion_submit"):
                self._json(200, discussion_submit_payload(payload))
                return
            if self.path.startswith("/api/bench_lead_metrics"):
                self._json(200, bench_lead_metrics_payload(payload))
                return
            if self.path.startswith("/api/rig_pair/start"):
                self._json(200, rig_pair_start_payload(payload))
                return
            if self.path.startswith("/api/rig_pair/advance"):
                self._json(200, rig_pair_advance_payload(payload))
                return
            if self.path.startswith("/api/rig_pair/reset"):
                self._json(200, rig_pair_reset_payload(payload))
                return
            if self.path.startswith("/api/recompute"):
                self._json(200, recompute_payload(payload))
                return
            if self.path.startswith("/api/config_profiles"):
                self._json(200, config_profiles_save(payload))
                return
            if self.path.startswith("/api/bench_probe"):
                self._json(200, bench_probe_payload(payload))
                return
            if self.path.startswith("/api/share_preview"):
                self._json(200, share_preview_payload(payload))
                return
            if self.path.startswith("/api/share_submit"):
                self._json(200, share_submit_payload(payload))
                return
        except Exception as e:  # pragma: no cover - defensive
            self._json(500, {"error": str(e)})
            return
        self._json(404, {"error": "not found"})

    def do_DELETE(self):
        try:
            payload = self._read_json()
        except Exception:
            payload = {}
        try:
            if self.path.startswith("/api/config_profiles"):
                from urllib.parse import parse_qs, urlsplit

                q = {k: v[0] for k, v in parse_qs(urlsplit(self.path).query).items()}
                q.update(payload or {})
                self._json(200, config_profiles_delete(q))
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

_ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")


def _vendored_asset(name: str) -> str:
    """Read a vendored front-end asset for inlining into the page.

    The dashboard serves one self-contained page and makes no external
    requests, so third-party JavaScript and CSS are inlined rather than
    linked. See ``assets/README.md`` for what is vendored and why.
    """
    with open(os.path.join(_ASSET_DIR, name), encoding="utf-8") as f:
        return f.read()


_INDEX_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>htsglang config planner</title>
<style>
/*__VENDOR_NORMALIZE__*/
  /* ---- design tokens -------------------------------------------------------
     Values are Grafana's published dark theme (grafana-data/src/themes:
     palette.ts, createColors.ts, createSpacing.ts, createTypography.ts), not
     invented ones. Grafana is the reference for this class of page -- a dense,
     dark, panel-based monitoring UI -- and its scales are what make the
     spacing come out even: one 8px grid unit, three surface levels, three
     border weights, and colour used only to carry state.
     The cross-browser reset above is vendored modern-normalize (MIT), inlined
     like morphdom; see assets/README.md. */
  :root {
    color-scheme: dark;
    /* surfaces: canvas < panel < elevated */
    --bg-canvas:   #111217;
    --bg-panel:    #181b1f;
    --bg-elevated: #22252b;
    --bg-input:    #0d1014;
    /* borders: weak = decoration, medium = widget outline, strong = focus */
    --bd-weak:   #363940;
    --bd-medium: #44464e;
    --bd-strong: #555760;
    /* text */
    --fg:          #ccccdc;
    --fg-strong:   #e6e9f0;
    --fg-muted:    rgba(204,204,220,.65);
    --fg-disabled: rgba(204,204,220,.45);
    /* interaction */
    --accent:        #3d71d9;
    --accent-text:   #6e9fff;
    --hover:         rgba(204,204,220,.10);
    --selected:      rgba(204,204,220,.16);
    /* state -- the ONLY decorative use of colour on this page is none */
    --ok:      #73BF69; --ok-dim:   #37872D;
    --warn:    #EAB839; --warn-dim: #E0B400;
    --bad:     #F2495C; --bad-dim:  #C4162A;
    --info:    #5794F2;
    /* 8px grid */
    --s1: 4px; --s2: 8px; --s3: 12px; --s4: 16px; --s5: 24px; --s6: 32px;
    /* type scale: 14 base, 12 for tile metadata */
    --t-xs: 11px; --t-sm: 12px; --t-md: 13px; --t-lg: 14px; --t-xl: 18px;
    --radius: 4px; --radius-lg: 6px;
    --mono: ui-monospace, SFMono-Regular, "Roboto Mono", Menlo, monospace;
    /* the page never grows past this; the gutter stays constant either side */
    --page-max: 1680px;
    --gutter: var(--s4);
  }
  * { box-sizing: border-box; }
  body { font-family: var(--mono); font-size: var(--t-md); line-height: 1.45;
         background: var(--bg-canvas); color: var(--fg);
         margin: 0; padding: 0; }
  /* Every top-level block sits in the same centred column, so the left and
     right margins are equal on every tab and do not change with content. */
  body > .hdr, body > .tabs, body > div[id^="view_"] {
    max-width: var(--page-max); margin-left: auto; margin-right: auto;
    padding-left: var(--gutter); padding-right: var(--gutter); }
  h1 { font-size: var(--t-xl); font-weight: 500; margin: 0; letter-spacing: 0; }
  .sub { color: var(--fg-muted); font-size: var(--t-sm); margin: var(--s1) 0 0;
         max-width: 96ch; }
  .cols { display: grid; grid-template-columns: 380px 1fr; gap: var(--s4);
          align-items: start; }
  /* runner: full page width, settings left / results right */
  .cols.runner { grid-template-columns: minmax(420px, 5fr) 7fr; }
  @media (max-width: 820px) { .cols, .cols.runner { grid-template-columns: 1fr; } }
  @media (max-width: 1100px) { .cols.runner { grid-template-columns: 1fr; } }

  /* ---- panels --------------------------------------------------------------
     A section is separated by a surface shift plus one weak border, never by
     whitespace -- the convention every dense monitoring UI converges on. */
  fieldset { border: 1px solid var(--bd-weak); border-radius: var(--radius-lg);
             background: var(--bg-panel);
             margin: 0 0 var(--s3); padding: var(--s3); }
  legend { color: var(--fg-muted); font-size: var(--t-sm); font-weight: 500;
           padding: 0 var(--s1); text-transform: none; }
  label { display: block; font-size: var(--t-sm); color: var(--fg-muted);
          margin: var(--s2) 0 var(--s1); }
  input, select, textarea {
    width: 100%; background: var(--bg-input); color: var(--fg-strong);
    border: 1px solid var(--bd-medium); border-radius: var(--radius);
    padding: 5px 8px; font: inherit; font-size: var(--t-sm); }
  input:focus, select:focus, textarea:focus {
    outline: none; border-color: var(--bd-strong); }
  input::placeholder { color: var(--fg-disabled); }
  .knob-help { color: var(--fg-muted); font-size: var(--t-xs);
               margin: 2px 0 var(--s1); }
  button { background: var(--accent); color: #fff; border: 1px solid transparent;
           border-radius: var(--radius); padding: 6px 12px; font: inherit;
           font-size: var(--t-sm); line-height: 1.4; cursor: pointer; }
  button:hover { filter: brightness(1.12); }
  button:disabled { opacity: .5; cursor: default; filter: none; }
  button.secondary { background: var(--bg-elevated); color: var(--fg);
                     border-color: var(--bd-medium); }
  button.secondary:hover { background: var(--selected); filter: none; }
  button.mini { padding: 3px 8px; font-size: var(--t-xs); }

  /* ---- header + tab bar ----------------------------------------------------
     A conservative admin tab bar: one row, fixed 36px height, active tab
     marked by a 2px underline in the accent colour and a brighter label,
     hover by a flat tint. No pills, no colour per tab, no shadow. The bar's
     own bottom border runs the full width so the tabs read as a strip rather
     than as a row of loose buttons. */
  .hdr { display: flex; align-items: flex-start; justify-content: space-between;
         gap: var(--s4); flex-wrap: wrap;
         padding-top: var(--s4); padding-bottom: var(--s3); }
  /* The bar WRAPS; it never scrolls. A scrollbar in a navigation strip hides
     destinations behind a gesture, which is exactly what navigation must not
     do. Labels are one word each so wrapping is rare in the first place. */
  .tabs { display: flex; flex-wrap: wrap; gap: 0; margin: 0 auto var(--s4);
          border-bottom: 1px solid var(--bd-weak); overflow: visible; }
  .tab, button.tab { background: transparent; color: var(--fg-muted);
    border: 0; border-bottom: 2px solid transparent; border-radius: 0;
    padding: 0 var(--s3); height: 36px; font-size: var(--t-md);
    white-space: nowrap; flex: none; cursor: pointer; }
  .tab:hover { background: var(--hover); color: var(--fg); filter: none; }
  .tab.active, button.tab.active { background: transparent;
    color: var(--fg-strong); border-bottom-color: var(--accent-text); }
  .tab.active:hover { background: var(--hover); }

  /* ---- tables --------------------------------------------------------------
     Numbers are the content, so they get tabular figures and right alignment;
     the label column stays left. */
  table { border-collapse: collapse; width: 100%; font-size: var(--t-sm); }
  th, td { text-align: right; padding: 4px 8px;
           border-bottom: 1px solid var(--bd-weak);
           font-variant-numeric: tabular-nums; }
  th { color: var(--fg-muted); font-weight: 500; }
  th:first-child, td:first-child { text-align: left;
                                   font-variant-numeric: normal; }

  /* ---- meters --------------------------------------------------------------
     One bar, filled left to right, numeric label beside it -- never a stack of
     decorative segments for a single quantity. */
  .bar { height: 8px; background: var(--bg-input); border-radius: 2px;
         overflow: hidden; border: 1px solid var(--bd-weak); }
  .bar > span { display: block; height: 100%; background: var(--info); }
  .verdict { font-size: var(--t-lg); font-weight: 500; padding: var(--s2) var(--s3);
             border-radius: var(--radius); margin-bottom: var(--s3); }
  .verdict.offload { background: rgba(234,184,57,.10); color: var(--warn);
                     border: 1px solid var(--warn-dim); }
  .fit { background: rgba(115,191,105,.10); color: var(--ok);
         border: 1px solid var(--ok-dim); }
  .nofit { background: rgba(242,73,92,.10); color: var(--bad);
           border: 1px solid var(--bad-dim); }
  .pricebar { margin-top: var(--s3); font-size: var(--t-sm);
              color: var(--fg-muted); }
  .pricebar input { width: 5rem; padding: 3px 6px; }
  .measured { margin-top: var(--s3); background: var(--bg-panel);
              border: 1px solid var(--ok-dim); border-radius: var(--radius-lg);
              padding: var(--s3); }
  .ms-title { font-weight: 500; color: var(--ok); font-size: var(--t-lg);
              text-transform: uppercase; letter-spacing: .04em; }
  .ms-note { font-size: var(--t-sm); color: var(--fg-muted);
             margin: var(--s1) 0 var(--s2); }
  .ms-wl { margin: var(--s2) 0; }
  .ms-mult { font-size: var(--t-sm); color: var(--fg-strong);
             background: var(--bg-elevated); border: 1px solid var(--ok-dim);
             border-radius: var(--radius); padding: var(--s1) var(--s2);
             margin: var(--s1) 0; }
  .ms-row { border-top: 1px solid var(--bd-weak); margin-top: var(--s2);
            padding-top: var(--s1); }
  .ms-row-h { font-size: var(--t-md); color: var(--fg); }
  .ms-wl-h { font-size: var(--t-sm); color: var(--fg-muted);
             margin-top: var(--s1); }
  .ms-phases { display: flex; gap: var(--s3); flex-wrap: wrap; }
  .ms-phase { flex: 1 1 320px; min-width: 300px; border-radius: var(--radius);
              padding: var(--s2); }
  .ms-prefill { background: var(--bg-elevated);
                border: 1px solid var(--bd-weak); }
  .ms-decode  { background: var(--bg-elevated);
                border: 1px solid var(--bd-weak); }
  .ms-ph-h { font-weight: 500; font-size: var(--t-sm); letter-spacing: .04em;
             margin-bottom: var(--s1); text-transform: uppercase; }
  .ms-prefill .ms-ph-h { color: var(--info); }
  .ms-decode .ms-ph-h { color: var(--ok); }
  .ms-phase table, .ms-wl table { border-collapse: collapse; margin: var(--s1) 0;
                                  font-size: var(--t-xs); width: 100%; }
  .ms-phase th, .ms-phase td, .ms-wl th, .ms-wl td {
      border: 1px solid var(--bd-weak); padding: 2px 6px; text-align: right;
      font-variant-numeric: tabular-nums; }
  .ms-phase th, .ms-wl th { color: var(--fg-muted); font-weight: 500; }
  .roofline { margin-top: var(--s3); background: var(--bg-panel);
              border: 1px solid var(--warn-dim); border-radius: var(--radius-lg);
              padding: var(--s3); }
  .rf-title { font-weight: 500; color: var(--warn); font-size: var(--t-lg);
              text-transform: uppercase; letter-spacing: .04em; }
  .rf-nums { display: flex; gap: var(--s3); margin: var(--s2) 0;
             flex-wrap: wrap; }
  .rf-num { background: var(--bg-elevated); border: 1px solid var(--bd-weak);
            border-radius: var(--radius); padding: var(--s2) var(--s3); }
  .rf-num span { display: block; font-size: var(--t-xs); color: var(--fg-muted);
                 text-transform: uppercase; letter-spacing: .04em; }
  .rf-num b { font-size: var(--t-xl); color: var(--fg-strong); font-weight: 500;
              font-variant-numeric: tabular-nums; }
  .rf-num small { display: block; font-size: var(--t-xs);
                  color: var(--fg-muted); }
  .rf-meas { color: var(--ok); font-weight: 500; }
  .rf-name { color: var(--warn); }
  .rf-caveats { margin: var(--s2) 0 0; padding-left: var(--s4);
                font-size: var(--t-sm); color: var(--fg-muted); }
  .rf-caveats li { margin: 2px 0; }
  .rf-energy { margin-top: var(--s3); padding-top: var(--s2);
               border-top: 1px solid var(--bd-weak); }
  pre { background: var(--bg-input); border: 1px solid var(--bd-weak);
        border-radius: var(--radius); padding: var(--s2); overflow-x: auto;
        font-size: var(--t-sm); white-space: pre-wrap; word-break: break-word;
        margin: var(--s2) 0; }
  code { font-family: var(--mono); }
  .muted { color: var(--fg-muted); font-size: var(--t-sm); }
  .est { color: var(--warn); font-size: var(--t-xs); }
  .reasons li { color: var(--bad); font-size: var(--t-sm); margin: 2px 0; }
  .adv { border-left: 2px solid var(--accent); padding: var(--s1) var(--s3);
         margin: var(--s2) 0; }
  .knoblist { font-size: var(--t-xs); color: var(--fg-muted); }
  .pill { display: inline-block; background: var(--bg-elevated);
          border: 1px solid var(--bd-weak); border-radius: 9999px;
          padding: 1px 8px; margin: 2px 2px 0 0; font-size: var(--t-xs); }
  .actions { display: flex; gap: var(--s2); flex-wrap: wrap;
             margin-top: var(--s2); }
  a { color: var(--accent-text); }
  /* ---- benchmark test buttons ---------------------------------------------
     The tests are the tab's content, so they are the thing you click. A
     selected test is a pressed button (accent border + tinted surface), a
     gated one is disabled and carries its reason on hover. Fixed-width tiles
     keep the grid even however long a label is. */
  .testbtn { display: flex; flex-direction: column; align-items: flex-start;
             gap: 2px; width: 232px; min-height: 52px; text-align: left;
             background: var(--bg-elevated); color: var(--fg);
             border: 1px solid var(--bd-medium); border-radius: var(--radius);
             padding: var(--s2); font: inherit; font-size: var(--t-sm);
             cursor: pointer; }
  .testbtn:hover { background: var(--selected); filter: none; }
  .testbtn.on { border-color: var(--accent-text); background: rgba(61,113,217,.16);
                color: var(--fg-strong); }
  .testbtn.on .tb-n { color: var(--accent-text); }
  .testbtn.gated { opacity: .5; cursor: not-allowed; }
  .testbtn .tb-n { font-size: var(--t-xs); color: var(--fg-muted);
                   font-variant-numeric: tabular-nums; }
  .testbtn .tb-l { line-height: 1.3; }
  .testbtn .tb-t { font-size: var(--t-xs); color: var(--fg-muted); }
  .testbtn .tb-t.warn { color: var(--warn); }
  .mx td, .mx th { text-align: center; }
  .mx .estcell { outline: 1px dashed var(--warn-dim); }
  .mx .fitc { color: var(--ok); } .mx .nofitc { color: var(--bad); }
  .legend { color: var(--fg-muted); font-size: var(--t-xs);
            margin-top: var(--s2); }
  .cardlist { display: flex; flex-direction: column; gap: var(--s1); }
  .cardrow { display: grid; grid-template-columns: auto 1fr auto auto;
             gap: var(--s2); align-items: center; background: var(--bg-elevated);
             border: 1px solid var(--bd-weak); border-radius: var(--radius);
             padding: var(--s1) var(--s2); }
  .cardrow input[type=text] { padding: 2px 6px; font-size: var(--t-sm); }
  .cardrow .vram { width: 5.4rem; text-align: right; }
  .cardrow .resv { width: 4.2rem; text-align: right; }
  .cardrow .cap { font-size: var(--t-xs); color: var(--fg-muted); }
  .cardrow.excluded { opacity: .45; }
  .cardrow input[type=checkbox] { width: auto; }
  .cardblock { border: 1px solid var(--bd-weak); border-radius: var(--radius-lg);
               padding: var(--s2) var(--s3); margin: var(--s2) 0;
               background: var(--bg-panel); }
  .segbar { display: flex; height: 14px; border-radius: 2px; overflow: hidden;
            background: var(--bg-input); border: 1px solid var(--bd-weak);
            margin: var(--s1) 0; }
  .segbar span { display: block; height: 100%; }
  /* replicated segments (KV heads synced/duplicated): hatched overlay so the
     not-the-normal case is unmissable in the bar itself. */
  .segbar span.hatch { background-image: repeating-linear-gradient(
      45deg, rgba(255,255,255,.28) 0 3px, transparent 3px 7px); }
  .seglegend { font-size: var(--t-xs); color: var(--fg-muted); margin: 2px 0; }
  .repltag { font-size: var(--t-xs); color: var(--warn);
             border: 1px solid var(--warn-dim); border-radius: 2px;
             padding: 0 4px; margin-left: var(--s1); }
  .graphline { font-size: var(--t-xs); color: var(--fg-muted); margin: 2px 0; }
  /* side-by-side planned vs plain normal-TP (runner): two equal columns,
     stacking on narrow screens. */
  .sxs { display: grid; grid-template-columns: repeat(auto-fit,
         minmax(340px, 1fr)); gap: var(--s3); align-items: start; }
  .sxs-h { font-size: var(--t-sm); font-weight: 500; color: var(--fg);
           margin-bottom: var(--s1); text-transform: uppercase;
           letter-spacing: .04em; }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 2px;
         margin-right: var(--s1); }
  .rankline { color: var(--fg-muted); font-size: var(--t-xs);
              margin: 2px 0 0 var(--s1); }
  .st-pass { color: var(--ok); } .st-warn { color: var(--warn); }
  .st-fail { color: var(--bad); } .st-skip { color: var(--fg-muted); }
  .st-blocked { color: var(--fg-muted); }
  .cfgrow { margin: 2px 0; }
  .cfgrow .cfgk { display: inline-block; width: 7.5rem; color: var(--fg-muted);
                  font-size: var(--t-sm); vertical-align: top; }
  /* -- settings rows: label left, control right, ? at the end --------------- */
  .setrow { display: flex; align-items: center; gap: var(--s2);
            margin: var(--s1) 0; font-size: var(--t-sm); }
  .setrow .lbl { flex: 1; min-width: 0; color: var(--fg); display: flex;
                 align-items: center; gap: var(--s1); overflow: hidden;
                 white-space: nowrap; text-overflow: ellipsis; }
  .setrow input[type=text], .setrow input[type=number], .setrow select {
    width: auto; max-width: 42%; padding: 2px 6px; font-size: var(--t-sm); }
  .setrow input[type=number].num { width: 6.4rem;
                                   font-variant-numeric: tabular-nums; }
  .setrow input[type=range] { flex: 1.2; width: auto; max-width: none;
    padding: 0; accent-color: var(--accent); background: transparent; border: 0; }
  .qmark { flex: none; width: 15px; height: 15px; border-radius: 50%;
           border: 1px solid var(--bd-medium); color: var(--fg-muted);
           font-size: var(--t-xs); line-height: 1; display: inline-flex;
           align-items: center; justify-content: center; cursor: help;
           user-select: none; }
  .chg { display: none; width: 6px; height: 6px; border-radius: 50%;
         background: var(--warn); flex: none; }
  .setrow.changed .chg { display: inline-block; }
  /* toggle switch (pure CSS; the real input stays a hidden checkbox) */
  .switch { position: relative; display: inline-block; width: 34px;
            height: 18px; flex: none; }
  .switch input { opacity: 0; width: 0; height: 0; position: absolute;
                  margin: 0; padding: 0; border: 0; }
  .switch .track { position: absolute; inset: 0; background: var(--bg-elevated);
    border-radius: 9999px; transition: .15s; cursor: pointer;
    border: 1px solid var(--bd-medium); }
  .switch .track:before { content: ""; position: absolute; width: 12px;
    height: 12px; left: 2px; top: 2px; background: var(--fg-muted);
    border-radius: 50%; transition: .15s; }
  .switch input:checked + .track { background: var(--accent);
                                   border-color: var(--accent); }
  .switch input:checked + .track:before { transform: translateX(16px);
    background: #fff; }
  .switch input:disabled + .track { opacity: .45; cursor: default; }
  /* collapsible config sections in a fixed order */
  .cfg-section { border: 1px solid var(--bd-weak); border-radius: var(--radius);
                 margin: var(--s1) 0; background: var(--bg-elevated); }
  .cfg-section > summary { cursor: pointer; padding: var(--s2);
    font-size: var(--t-sm); color: var(--fg); list-style-position: inside; }
  .cfg-section > summary:hover { background: var(--hover); }
  .cfg-section > summary b { color: var(--fg-strong); font-weight: 500; }
  .cfg-section .sec-sum { color: var(--fg-muted); font-size: var(--t-xs);
                          margin-left: var(--s1); }
  .cfg-section .sec-body { padding: 0 var(--s2) var(--s2); }
  .advrow { border: 1px dashed var(--bd-medium); border-radius: var(--radius);
            margin: var(--s2) 0; padding: var(--s2); }
  /* sticky action bar (load / eject / restart + status chip) */
  .actionbar { position: sticky; bottom: 0; z-index: 5;
    background: var(--bg-panel); border: 1px solid var(--bd-weak);
    border-radius: var(--radius-lg); padding: var(--s2) var(--s3);
    margin: var(--s3) 0; box-shadow: 0 -6px 12px rgba(0,0,0,.45); }
  .chip { display: inline-flex; align-items: center; gap: var(--s1);
    border-radius: 9999px; padding: 2px 10px; font-size: var(--t-xs);
    border: 1px solid var(--bd-medium); background: var(--bg-elevated);
    color: var(--fg-muted); }
  .chip:before { content: ""; width: 7px; height: 7px; border-radius: 50%;
                 background: var(--fg-disabled); }
  .chip.ready { color: var(--ok); border-color: var(--ok-dim); }
  .chip.ready:before { background: var(--ok); }
  .chip.loading { color: var(--warn); border-color: var(--warn-dim); }
  .chip.loading:before { background: var(--warn); }
  .chip.error { color: var(--bad); border-color: var(--bad-dim); }
  .chip.error:before { background: var(--bad); }
  /* hardware: per-card VRAM bar spanning the row */
  .cardrow .cardbar { grid-column: 1 / -1; }
  /* landing top strip: ONE full-width row of live metric tiles (label, big
     value, secondary line, 60s sparkline). Normal flow -- the strip pushes
     the rest of the page down, nothing overlaps; tiles wrap to a second row
     on narrow screens instead of overflowing. */
  .mstrip { display: flex; flex-wrap: wrap; gap: var(--s2); width: 100%;
            margin: 0 0 var(--s3); }
  .mtile { flex: 1 1 172px; min-width: 172px; background: var(--bg-panel);
           border: 1px solid var(--bd-weak); border-radius: var(--radius-lg);
           padding: var(--s2); }
  .mtile .mt-l { font-size: var(--t-xs); color: var(--fg-muted);
                 text-transform: uppercase; letter-spacing: .04em;
                 white-space: nowrap; overflow: hidden;
                 text-overflow: ellipsis; }
  .mtile .mt-v { font-size: 20px; font-weight: 500; color: var(--fg-strong);
                 line-height: 1.3; white-space: nowrap;
                 font-variant-numeric: tabular-nums; }
  .mtile .mt-v small { font-size: var(--t-xs); font-weight: 400;
                       color: var(--fg-muted); }
  .mtile .mt-s { font-size: var(--t-xs); color: var(--fg-muted);
                 line-height: 1.4; min-height: 1.7em;
                 font-variant-numeric: tabular-nums; }
  /* fixed-pixel sparkline: sized in JS as samples*px so ONE sample = ONE
     pixel bucket; never stretched by CSS (stretching would alias samples). */
  .mtile svg.spk { display: block; margin-top: var(--s1);
                   background: var(--bg-input); border-radius: 2px;
                   max-width: 100%; }

  /* ---- refresh affordance -------------------------------------------------
     A panel waiting on a slow backend call keeps showing its previous
     numbers, dimmed. Stale-while-revalidate, not a spinner that blanks the
     panel and throws away what the reader was looking at. The transition is
     short enough that a fast answer never registers as a flicker. */
  .stale { opacity: .55; transition: opacity .12s linear; }
  .stale-note { font-size: var(--t-xs); color: var(--fg-muted); }
  /* A backend call that failed or timed out. The panel keeps its last good
     content; this line says why it is no longer moving. */
  .rev-error { font-size: var(--t-sm); color: var(--warn);
               margin-top: var(--s1); }

  /* ---- simple / expert -----------------------------------------------------
     Two densities of the same page, not two different pages. Expert ADDS
     dimensions; it never changes what a control shared by both means. */
  .vmode { display: flex; border: 1px solid var(--bd-medium);
           border-radius: var(--radius); overflow: hidden; flex: none; }
  .vmode .vm { background: var(--bg-elevated); color: var(--fg-muted);
               border: 0; border-radius: 0; padding: 5px 14px; font: inherit;
               font-size: var(--t-sm); cursor: pointer; }
  .vmode .vm.on { background: var(--accent); color: #fff; }
  .vmode .vm:not(.on):hover { color: var(--fg-strong);
                              background: var(--selected); filter: none; }
  /* Visibility is driven by one class on <body>, so no panel has to know
     which mode it is in and a mode switch touches no backend call. */
  body.mode-simple .expert-only { display: none !important; }
  body.mode-expert .simple-only { display: none !important; }

  /* ---- simple view: one bar and one budget slider per card ---------------- */
  .cardsimple { border: 1px solid var(--bd-weak); border-radius: var(--radius-lg);
                padding: var(--s2) var(--s3); margin: var(--s2) 0;
                background: var(--bg-panel); }
  .cardsimple .cs-h { display: flex; justify-content: space-between;
                      align-items: baseline; gap: var(--s2); flex-wrap: wrap; }
  .cardsimple .cs-n { font-weight: 500; color: var(--fg-strong); }
  .cardsimple .cs-u { font-size: var(--t-sm); color: var(--fg-muted);
                      white-space: nowrap; font-variant-numeric: tabular-nums; }
  /* One bar = everything the configuration puts on this card. The granular
     split is the expert view's job; here the number is the whole point.
     Fill colour is state, not decoration: normal / tight / over. */
  .csbar { position: relative; height: 18px; border-radius: 2px;
           background: var(--bg-input); border: 1px solid var(--bd-weak);
           overflow: hidden; margin: var(--s1) 0; }
  .csbar .fill { position: absolute; inset: 0 auto 0 0; background: var(--info); }
  .csbar .fill.over { background: var(--bad); }
  .csbar .fill.tight { background: var(--warn); }
  /* The budget the user set, drawn as a line across the bar: the reader can
     see at a glance whether the configuration sits under it or over it. */
  .csbar .cap { position: absolute; top: 0; bottom: 0; width: 2px;
                background: var(--fg-strong); }
  .csrow { display: flex; align-items: center; gap: var(--s2); }
  .csrow input[type=range] { flex: 1 1 auto; min-width: 8rem; }
  .csrow .cs-v { font-size: var(--t-sm); color: var(--fg-muted);
                 min-width: 7.5rem; text-align: right; white-space: nowrap;
                 font-variant-numeric: tabular-nums; }
</style>
</head>
<body>
<div class="hdr">
  <div>
    <h1>htsglang offline config planner</h1>
    <div class="sub">capacity / feasibility / split &mdash; never estimated throughput.
      Every edit re-runs the same planner the server runs. No GPU touched.</div>
  </div>
  <!-- Simple vs expert. Not a hiding place for broken controls: expert adds
       dimensions, it never changes what a shared control means. The choice
       persists, so the page opens the way it was left. -->
  <div class="vmode" id="view_mode">
    <button id="vm_simple" class="vm" onclick="setViewMode('simple')"
            title="per-card VRAM as one number, one budget slider per card">simple</button>
    <button id="vm_expert" class="vm" onclick="setViewMode('expert')"
            title="the granular VRAM breakdown and every dimension the fork exposes">expert</button>
  </div>
</div>

<!-- One word per tab, so the bar fits on a normal window without ever
     scrolling; the long form is the title attribute. -->
<div class="tabs">
  <button id="tab_landing" class="tab active" onclick="showTab('landing')"
    title="live monitor of any reachable sglang server">Monitor</button>
  <button id="tab_runner" class="tab" onclick="showTab('runner')"
    title="models, capacity planner and server launch">Planner</button>
  <button id="tab_bench" class="tab" onclick="showTab('bench')"
    title="behavioural benchmark / quality suite">Benchmark</button>
  <button id="tab_explore" class="tab" onclick="showTab('explore')"
    title="model x rig capacity matrix">Rigs</button>
  <button id="tab_landscape" class="tab" onclick="showTab('landscape')"
    title="recorded benchmark database">Landscape</button>
  <button id="tab_energy" class="tab" onclick="showTab('energy')"
    title="per-card power calibration">Energy</button>
  <button id="tab_quality" class="tab" onclick="showTab('quality')"
    title="rendering / instruction-following checks">Quality</button>
  <button id="tab_pair" class="tab" onclick="showTab('pair')"
    title="couple a second rig">Pair rig</button>
</div>

<div id="view_pair" style="display:none">
  <div class="sub">Couple a second rig, one step at a time. Every step runs on
    this host &mdash; reachability, the compatibility gate, the transport
    choice and the configuration are all computed here, exactly like telemetry
    collection and plan computation are. This page only sends the command and
    shows what came back, so the same flow is drivable from a script, and
    reloading the browser resumes rather than restarts.
    <b>Nothing is booted across rigs</b>: the flow ends at a configuration.</div>
  <fieldset>
    <legend>far rig</legend>
    <div style="display:flex;gap:.4rem;align-items:center;flex-wrap:wrap">
      <input id="pair_target" placeholder="far rig's rigmon aggregator — host:port"
        style="max-width:24rem">
      <button class="mini" onclick="pairStart()">couple</button>
      <button class="mini secondary" onclick="pairReset()"
        title="clear the results and run the steps again against the same rig">retry</button>
      <span id="pair_note" class="muted"></span>
    </div>
    <div class="legend" style="margin-top:.3rem">The far rig needs its rigmon
      aggregator running (<code>python -m sglang.srt.rigmon serve</code>). Only
      its read-only endpoints are touched; nothing is pushed or changed there.</div>
  </fieldset>
  <div id="pair_steps"><span class="muted">enter a rig above to begin.</span></div>
</div>

<div id="view_landing">
  <div class="sub">Live monitor. Attaches to ANY reachable sglang server:
    an explicit endpoint wins, else the managed instance, else a server
    auto-detected on the common local ports &mdash; hand-started servers are
    monitored exactly like managed ones. Charts keep a client-side 60s ring
    buffer; <b>nothing is persisted</b>.</div>
  <fieldset>
    <legend>monitor target</legend>
    <div style="display:flex;gap:.4rem;align-items:center;flex-wrap:wrap">
      <input id="land_endpoint" placeholder="host:port (blank = managed instance, else auto-detect)" style="max-width:22rem">
      <button class="mini" onclick="setLandingEndpoint()">use</button>
      <button class="mini secondary" onclick="clearLandingEndpoint()" title="clear the explicit endpoint; fall back to managed / auto-detect">auto</button>
      <button class="mini secondary" onclick="detectLandingEndpoint()" title="empty box: sweep the local sglang ports (30000-30100, 8000, 8080). Box filled: verify that host:port.">detect</button>
      <span id="land_target_note" class="muted">resolving&hellip;</span>
    </div>
  </fieldset>
  <!-- Always present, in every state. Living inside the "a server is
       running" container is what let a warning outlive the server it was
       about: the container was hidden, the stale text stayed. -->
  <fieldset id="live_panel">
    <legend>VRAM in use, throughput per session, tokens per watt (live)</legend>
    <div id="live_note" class="muted">looking for a running server&hellip;</div>
    <div id="live_strip" class="mstrip" style="display:none"></div>
    <div id="live_cards"></div>
    <div class="legend" id="live_legend" style="display:none">Bars are the
      <b>measured</b> NVML occupancy of each card, drawn with the same bar the
      planner uses for its projection, so the two are comparable by looking.
      Per-session rates divide the server-wide rate by the concurrent requests
      being served. Tokens per watt is the live token rate over NVML power
      draw &mdash; per card it says what that card costs to keep in the group,
      the total is the figure for the whole configuration.</div>
  </fieldset>
  <div id="landing_none" class="muted" style="display:none;padding:1rem;border:1px solid #30363d;border-radius:8px">
    No reachable sglang server: no explicit endpoint, no managed instance,
    nothing detected on the common ports. Enter an endpoint above, hit
    <b>detect</b>, or launch a server in the Runner tab.
    <span id="landing_none_status" class="muted"></span></div>
  <div id="landing_live" style="display:none">
    <!-- full-width top strip: the live headline metrics (throughput / spec /
         cache / energy / cost) with 60s freeze-at-zero activity graphs. This
         REPLACES the former "throughput / spec / cache" block below -- the
         strip is the only place these numbers appear on the landing page. -->
    <div id="landing_strip" class="mstrip"><span class="muted">waiting for first snapshot&hellip;</span></div>
    <fieldset>
      <legend>running model + start config</legend>
      <div id="landing_config" class="muted">waiting for first snapshot&hellip;</div>
    </fieldset>
    <!-- ONE block per physical card: what occupies its VRAM (placement bar)
         AND how it is doing live (compact telemetry line + 60s util/power
         sparklines). The former separate standalone per-GPU chart grid is
         MERGED into these card blocks -- nothing is rendered twice. -->
    <fieldset>
      <legend>per-card VRAM placement + live telemetry (RUNNING config, 60s)</legend>
      <div id="landing_placement" class="muted">computing&hellip;</div>
    </fieldset>
    <fieldset>
      <legend>energy &mdash; live GPU power state (per-card power CALIBRATION lives in the Energy tab)</legend>
      <div style="margin-bottom:.4rem">
        <button class="secondary mini" onclick="refreshGpuState()" title="re-query NVML live">refresh power state</button>
      </div>
      <div id="gpu_state_out" class="muted">click refresh&hellip;</div>
    </fieldset>
  </div>
</div>

<div id="view_bench" style="display:none">
  <div class="sub">club-3090 behavioral benchmark / quality suite (#151).
    Every probe runs <b>backend-side</b> against the target's OpenAI API
    &mdash; the browser never calls the model. Chat-template / parser /
    spec-decode gating decides which tests can run; crash-prone long-context
    tests always run LAST. Results stream in per test.</div>
  <div class="cols">
    <div>
      <fieldset>
        <legend>target + capabilities</legend>
        <label>endpoint (OpenAI-compatible base URL)</label>
        <div style="display:flex;gap:.4rem">
          <input id="bn_endpoint" placeholder="127.0.0.1:30000">
          <button class="mini secondary" onclick="benchUseMonitor()" title="copy the landing monitor target">monitor</button>
        </div>
        <label>model (blank = probed from /v1/models)</label>
        <input id="bn_model" placeholder="(auto)">
        <div style="margin-top:.5rem"><button onclick="benchProbe(false)" id="bn_probe_btn">Probe capabilities</button></div>
        <div id="bn_caps" class="muted" style="margin-top:.5rem">probe to see the gating state&hellip;</div>
      </fieldset>
      <fieldset>
        <legend>share results on GitHub (#152)</legend>
        <div class="muted" style="margin-bottom:.4rem">Builds the EXACT
          markdown to post as a preview &mdash; nothing is sent until you
          confirm. One issue per user, updated in place on re-submit. The PAT
          is entered per-use only: <b>never persisted, never logged</b>, and
          redacted from every error message.</div>
        <label><input type="checkbox" id="sh_inc_metrics" checked style="width:auto"> measured rates + per-card power (landing monitor)</label>
        <label><input type="checkbox" id="sh_inc_bench" checked style="width:auto"> benchmark results (last run)</label>
        <label><input type="checkbox" id="sh_inc_quality" style="width:auto"> quality shot (SVG + verdict + tokens, Quality tab)</label>
        <label>notes (optional)</label>
        <textarea id="sh_notes" rows="2"></textarea>
        <div style="margin-top:.4rem"><button class="secondary" onclick="sharePreview()">Build preview</button></div>
        <div id="sh_preview_wrap" style="display:none;margin-top:.5rem">
          <div class="muted">this exact markdown will be posted:</div>
          <pre id="sh_preview" style="max-height:280px;overflow:auto"></pre>
          <label>GitHub PAT <span class="muted">(used once for this submit; never stored)</span></label>
          <input id="sh_token" type="password" autocomplete="off">
          <label>repository</label>
          <input id="sh_repo">
          <label>existing issue number <span class="muted">(blank = find-or-create your issue)</span></label>
          <input id="sh_issue" placeholder="(optional)">
          <div style="margin-top:.4rem"><button onclick="shareSubmit()" id="sh_btn">Confirm + submit to GitHub</button></div>
        </div>
        <div id="sh_out" class="muted" style="margin-top:.4rem"></div>
      </fieldset>
      <fieldset>
        <legend>share in a GitHub discussion</legend>
        <div class="muted" style="margin-bottom:.4rem">Pick a bundle, read the
          preview, then send. System details always pass through the
          redaction: card models and driver versions go, host names, addresses
          and paths do not. <b>Nothing is created automatically</b> &mdash;
          with no discussion configured this builds the preview and says so.</div>
        <label>bundle</label>
        <select id="dx_bundle" onchange="discussionPreview()"></select>
        <div id="dx_bundle_note" class="muted" style="font-size:.68rem;margin:.2rem 0"></div>
        <label>energy metrics</label>
        <div id="dx_groups" class="muted" style="font-size:.7rem"></div>
        <div style="margin-top:.4rem"><button class="secondary" onclick="discussionPreview()">Build preview</button></div>
        <div id="dx_wrap" style="display:none;margin-top:.5rem">
          <div class="muted">this exact markdown would be posted:</div>
          <pre id="dx_preview" style="max-height:300px;overflow:auto"></pre>
          <div id="dx_gate" class="muted"></div>
          <div class="actions" style="margin-top:.4rem">
            <button onclick="discussionSubmit()" id="dx_btn">Confirm + post</button>
          </div>
        </div>
        <div id="dx_out" class="muted" style="margin-top:.4rem"></div>
      </fieldset>
    </div>
    <div>
      <!-- The tests ARE the tab. They get the full width and are picked by
           clicking them, not by hunting checkboxes in a narrow column: a
           selected test is a pressed button, a gated one is disabled and says
           why on hover. -->
      <fieldset>
        <legend>tests &mdash; click to select</legend>
        <div class="actions" id="bn_presets" style="margin-bottom:var(--s2)"></div>
        <div id="bn_tests" class="muted">loading test catalog&hellip;</div>
        <label style="margin:var(--s2) 0"><input type="checkbox" id="bn_force" style="width:auto" onchange="benchReGate()">
          force-run tool tests despite a missing tool parser
          <span class="muted">(deliberately surfaces the tool-call cascade)</span></label>
        <div class="actions">
          <button onclick="benchRun()" id="bn_run_btn">Run selected</button>
          <span id="bn_sel_note" class="muted"></span>
        </div>
      </fieldset>
      <!-- Past runs, per model. A verdict table is not reviewable on its own;
           the stored run carries every request and every answer, and the
           download is the whole series. -->
      <fieldset>
        <legend>history &mdash; past runs for this model</legend>
        <div class="actions" style="margin-bottom:var(--s2)">
          <button class="mini secondary" onclick="benchHistory()">refresh</button>
          <label style="display:inline-flex;align-items:center;gap:var(--s1);margin:0">
            <input type="checkbox" id="bn_hist_all" style="width:auto" onchange="benchHistory()">
            <span class="muted">all models</span></label>
          <span id="bn_hist_note" class="muted"></span>
        </div>
        <div id="bn_history" class="muted">no runs recorded yet.</div>
      </fieldset>
      <!-- Lead metrics: how long a round TAKES. tok/s hides which phase moved;
           ms per verify round and ms per 1k prefill tokens do not. Absent is
           shown as absent, never as zero. -->
      <fieldset>
        <legend>lead metrics &mdash; ms per round</legend>
        <div id="bn_lead" class="muted">start a run, or point at a busy server&hellip;</div>
      </fieldset>
      <!-- Running and finished are separated, so a finished table is never
           silently a partial one. -->
      <fieldset id="bn_running_box" style="display:none">
        <legend>running</legend>
        <div id="bn_running"></div>
      </fieldset>
      <fieldset>
        <legend>finished runs</legend>
        <div id="bn_out" class="muted">no run yet.</div>
      </fieldset>
    </div>
  </div>
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
        <legend>rigs to feasibility-check (NAME=card,card,...)</legend>
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
        <legend>per-card power calibration (measured)</legend>
        <div class="muted" style="margin-bottom:.4rem">Measures each free card's
          board power at idle / membw / GEMM (short micro-benchmarks that briefly
          load the GPU). <b>Do not run against a busy card</b> — the backend
          SKIPS any card another process owns and reports it. Result is persisted
          and overrides the energy model's power heuristic. Live GPU power-state
          + throughput monitoring moved to the Landing tab.</div>
        <button onclick="measurePower()" id="pw_btn">Measure per-card power</button>
        <div id="pw_out" class="muted" style="margin-top:.6rem"></div>
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


<div id="view_quality" style="display:none">
  <!-- User-facing reminder (self-note, intentionally prominent): the test
       design stems from the linked reddit thread; before making this
       feature public, ask there whether using the test is OK. -->
  <div id="quality_permission_note" style="border:1px solid #e3a008;
       background:#241a06; color:#e3b341; border-radius:8px;
       padding:.5rem .7rem; margin:.2rem 0 .6rem; font-size:.8rem">
    <b>ERINNERUNG:</b> Vor Veroeffentlichung im
    <a href="https://www.reddit.com/r/LocalLLaMA/comments/1t53dhp/quality_comparison_between_qwen_36_27b/"
       target="_blank">Reddit-Beitrag</a> nachfragen, ob die Nutzung dieses
    Tests ok ist.
  </div>
  <div class="sub">ONESHOT chess-SVG quality benchmark (<a href="https://www.reddit.com/r/LocalLLaMA/comments/1t53dhp/quality_comparison_between_qwen_36_27b/" target="_blank">reddit reference</a>).
    The model is asked to render the position after <code>7. h4</code> as SVG,
    highlighting the last move. The model is called <b>backend-side</b> (never
    from the browser); the SVG is graded deterministically against the true
    position. Verdict is honest: an unverifiable-but-rendering board is
    <b>renders-unverifiable</b>, never correct.</div>
  <div class="cols">
    <div>
      <fieldset>
        <legend>run config</legend>
        <label>server endpoint (OpenAI-compatible base URL)</label>
        <input id="q_endpoint" placeholder="127.0.0.1:30000">
        <label>model name</label>
        <input id="q_model" placeholder="served-model-name">
        <label>quant (for the saved-shot key)</label>
        <input id="q_quant" placeholder="Q4_K_M / AWQ / fp8">
        <label><input type="checkbox" id="q_think" style="width:auto"> thinking on</label>
        <label>thinking budget (tokens; blank = unbounded)</label>
        <input id="q_budget" placeholder="(optional int)">
        <label><input type="checkbox" id="q_save" style="width:auto"> save this shot</label>
        <div style="margin-top:.5rem"><button onclick="qualityRun()" id="q_btn">Run quality check</button></div>
        <div id="q_status" class="muted" style="margin-top:.5rem"></div>
      </fieldset>
    </div>
    <div>
      <div class="cols" style="grid-template-columns:1fr 1fr 1fr;gap:.6rem">
        <fieldset>
          <legend>reference (ground truth)</legend>
          <img src="/assets/quality_chess_reference.png" alt="reference board" style="max-width:100%;border-radius:6px">
        </fieldset>
        <fieldset>
          <legend>generated SVG</legend>
          <div id="q_svg" class="muted">run a check…</div>
        </fieldset>
        <fieldset>
          <legend>history <span id="q_slide_lbl" class="muted"></span></legend>
          <input type="range" id="q_slider" min="0" max="0" value="0" style="width:100%" oninput="showShot()">
          <div id="q_shot" class="muted" style="margin-top:.4rem">no saved shots.</div>
        </fieldset>
      </div>
      <div id="q_result" style="margin-top:.6rem"></div>
    </div>
  </div>
</div>

<div id="view_explore" style="display:none">
  <div class="sub">Compose rigs from the GPU-model card library (or add the
    live one) and see which models fit on each. <b>Composed rigs are
    ESTIMATES</b> &mdash; no measured free-VRAM or interconnect (§8).</div>
  <div class="cols">
    <div>
      <fieldset>
        <legend>models (one per line: LABEL=path, or just a path)</legend>
        <textarea id="mx_models" rows="4" placeholder="27B-AWQ=/path/to/Qwen3.6-27B-AWQ&#10;35B-A3B=/path/to/model"></textarea>
      </fieldset>
      <fieldset>
        <legend>rigs (one per line: NAME=card,card,... — or 'live')</legend>
        <textarea id="mx_rigs" rows="4" placeholder="hetero=RTX 5090,RTX 3080 20GB,RTX 3080 20GB&#10;4x4090=RTX 4090,RTX 4090,RTX 4090,RTX 4090"></textarea>
        <div id="mx_cards" class="knoblist" style="margin-top:.5rem"></div>
      </fieldset>
      <button onclick="doMatrix()">Build matrix</button>
    </div>
    <div><div id="mx_out"></div></div>
  </div>
</div>

<div id="view_runner" style="display:none">
<div class="sub">Pick the model ONCE at the top &mdash; it drives everything:
  profiles, flag resolve, plan, placement and launch. Below it: profile &rarr;
  serving identity &rarr; the one flag surface &rarr; launch. No field appears
  twice. The page opens on the configuration that is currently loaded, when
  there is one; live telemetry is the Monitor tab's job.</div>
<div id="loaded_cfg" class="adv" style="margin-bottom:var(--s3)">
  <span class="muted">reading the running configuration&hellip;</span></div>

<fieldset>
  <legend>model &mdash; the ONE selector (drives plan, profiles, placement, launch)
    <button class="secondary mini" onclick="loadModels()" title="re-scan roots">rescan</button></legend>
  <div style="display:flex;gap:1rem;flex-wrap:wrap;align-items:flex-start">
    <div style="flex:2;min-width:280px">
      <label>discovered local models</label>
      <input id="model_search" placeholder="search models (name / format / quant)&hellip;"
        oninput="renderModelOptions()" style="margin-bottom:.3rem">
      <div id="models_out" class="muted">scanning&hellip;</div>
    </div>
    <div style="flex:2;min-width:280px">
      <label>or type a path / HF id (config.json dir, .gguf file, GGUF dir)</label>
      <input id="model" placeholder="/path/to/Qwen3.6-27B-AWQ or org/model" onchange="onModelChange(); onRunnerModel()">
      <label style="margin-top:.4rem"><input type="checkbox" id="include_vision" style="width:auto"
        onchange="doPlan(); refreshRunnerPlacement()">
        vision tower on <span class="muted">(VL models; off = text-only sizing +
        launch, frees VRAM for KV)</span></label>
    </div>
    <div id="gguf_pick" style="flex:1;min-width:220px;display:none">
      <label>GGUF quant (several available &mdash; pick one)</label>
      <select id="gguf_choice" onchange="onRunnerModel()"></select>
    </div>
  </div>
  <div id="model_info" class="muted" style="margin-top:.4rem;word-break:break-all"></div>
</fieldset>


<!-- ===================================================================== -->
<!-- SIMPLE view: per card, how much VRAM this configuration uses, and one -->
<!-- slider for how much it is ALLOWED to use. Everything else is derived  -->
<!-- server-side and shown as the verdict. The granular breakdown, the     -->
<!-- ratios and the full flag surface are the expert view's business.      -->
<!-- ===================================================================== -->
<div class="simple-only">
  <fieldset>
    <legend>VRAM per card &mdash; usage and budget</legend>
    <div id="simple_cards" class="muted">pick a model above&hellip;</div>
    <div class="legend" style="margin-top:.35rem">The slider is each card's
      budget: how much of it this server may take. What the configuration
      actually needs is the bar. A budget below the bar is not refused here
      &mdash; it is reported, with the reason, in the verdict.</div>
  </fieldset>
  <fieldset>
    <legend>verdict</legend>
    <div id="simple_verdict" class="muted">waiting for a model&hellip;</div>
  </fieldset>
  <div class="actions" style="align-items:center;margin:.5rem 0">
    <button onclick="serverStart()">Load model</button>
    <button class="secondary" onclick="serverStop()">Eject</button>
    <span class="chip" id="simple_status_chip">stopped</span>
    <span id="simple_status_note" class="muted"></span>
  </div>
</div>

<div class="cols runner expert-only">
  <div>
    <!-- preset bar: dropdown + save, directly above the settings panel -->
    <fieldset>
      <legend>preset &mdash; fills the settings rows below; adaptive MTP is
        always on when the checkpoint has draft layers</legend>
      <div id="profile_pick" class="muted">select a model to list presets&hellip;</div>
      <!-- Tuning objective: the same trio the fork's --rank-perf-tune takes.
           A template is a STARTING POINT. Applying one seeds the controls;
           the controls keep their full range afterwards, and the limits stay
           the hard rejects, never the value the template happened to set. -->
      <div style="margin-top:.5rem">
        <label>tuning objective <span class="muted">(template &mdash; a starting
          point, not a ceiling)</span></label>
        <div id="tune_pick" style="display:flex;gap:.3rem;flex-wrap:wrap;margin-top:.2rem">
          <button class="mini secondary" id="tune_both" onclick="applyTune('both')"
            title="prefill and concurrent throughput together (the default)">balanced</button>
          <button class="mini secondary" id="tune_maxkv" onclick="applyTune('maxkv')"
            title="maximum context: give the KV cache everything that is left">max KV</button>
          <button class="mini secondary" id="tune_dec" onclick="applyTune('dec')"
            title="decode throughput">max decode</button>
          <button class="mini secondary" id="tune_enc" onclick="applyTune('enc')"
            title="prefill throughput">max prefill</button>
        </div>
        <div id="tune_note" class="muted" style="font-size:.68rem;margin-top:.2rem"></div>
      </div>
      <div style="display:flex;gap:.4rem;margin-top:.4rem;align-items:center">
        <input id="profile_save_name" placeholder="save current settings as&hellip;" style="max-width:60%">
        <button class="secondary mini" onclick="saveProfile()">save preset</button>
      </div>
      <div id="profile_msg" class="muted" style="margin-top:.3rem"></div>
      <div id="profile_env_box" style="margin-top:.3rem"></div>
    </fieldset>

    <!-- load settings: labeled rows in fixed, collapsible sections -->
    <fieldset>
      <legend>load settings
        (<span id="flag_counts" class="muted"></span>)</legend>
      <input id="flag_search" placeholder="search all settings (name / help)&hellip;" oninput="filterFlags()">
      <div class="muted" style="margin:.3rem 0 .4rem">Every sglang + fork flag,
        in fixed sections; a dot marks a row changed from the applied preset.
        Each field re-resolves on change: greyed + hover-? when excluded /
        incompatible / auto-set. Model identity is set above, not here.</div>
      <div id="flag_warnings"></div>
      <div id="flag_surface">

      <details class="cfg-section" id="sec_context">
        <summary><b>Context</b><span class="sec-sum" id="sum_context"></span></summary>
        <div class="sec-body">
          <div class="setrow" id="row_sv_ctx" data-hay="context-length context length tokens kv max">
            <span class="lbl">context-length
              <span class="chg" title="changed from preset"></span></span>
            <input type="range" id="sv_ctx_slider" min="512" max="262144" step="512"
              value="8192" oninput="ctxFromSlider()">
            <input type="number" id="sv_ctx" class="num" value="8192" onchange="ctxFromNum()">
            <span class="muted" id="sv_ctx_max" style="flex:none;font-size:.64rem"></span>
            <span class="qmark" title="Max tokens per request (prompt + output). The slider is capped at the plan's computed KV capacity for the current config; the numeric field is free.">?</span>
          </div>
          <div class="setrow" id="row_mrr" data-hay="max-running-requests max running requests concurrency seqs">
            <span class="lbl">max-running-requests
              <span class="chg" title="changed from preset"></span></span>
            <input type="range" id="mrr_slider" min="1" max="64" step="1" value="1"
              oninput="mrrFromSlider()">
            <span id="mrr_max" class="muted" style="font-size:.62rem"></span>
            <input type="number" id="max_running_requests" class="num"
              placeholder="auto" onchange="mrrFromNum()">
            <span class="qmark" title="Concurrent request slots. 1 = single-user = largest KV; raising it grows the GDN/mamba pool and shrinks max context (see the KV-vs-concurrency table in the plan result). The slider range follows what the plan reports as fitting; the numeric field is free, and a value that does not fit is REJECTED with a reason rather than silently clamped. Blank: plan 1, launch 16.">?</span>
          </div>
          <div id="secflags_context"></div>
        </div>
      </details>

      <details class="cfg-section" id="sec_gpu">
        <summary><b>GPU offload / split</b><span class="sec-sum" id="sum_gpu"></span></summary>
        <div class="sec-body">
          <div class="setrow" id="row_tp_count"
            data-hay="tensor parallel tp size rank count co-location colocation">
            <span class="lbl">tensor-parallel ranks (tp)
              <span class="chg" title="changed from preset"></span></span>
            <input type="number" class="num" id="tp_count" min="1" step="1"
              onchange="tpCountChanged()">
            <span class="qmark" title="Rank count for tensor parallelism (mirrors --tp-size; the tp-size row below stays the authoritative field). More ranks than enabled cards is allowed: the extra ranks then CO-LOCATE -- several ranks share one physical card. That needs CUDA MPS and is a fork-only capability; the per-card assignment controls appear below when tp exceeds the enabled card count.">?</span>
          </div>
          <div id="colo_box" style="display:none">
            <div class="muted" id="colo_note" style="margin:.2rem 0 .3rem;color:#e3a008">
              tp exceeds the enabled card count &mdash; the extra ranks
              CO-LOCATE (share a physical card). REQUIRES CUDA MPS on the
              shared card(s); co-location is a fork-only capability (stock
              sglang cannot co-locate ranks at all). Set how many ranks each
              card hosts; the sum must equal tp. The derived --rank-gpu-id
              (below) stays the authoritative flag.</div>
            <div id="colo_rows"></div>
            <div id="colo_sum" class="muted" style="margin:.15rem 0"></div>
            <div id="colo_err" class="reasons" style="display:none;margin:.15rem 0"></div>
            <div id="colo_manual_note" class="muted"
              style="display:none;color:#e3a008;margin:.15rem 0">
              manual rank_gpu_id active &mdash; the free-text rank-gpu-id
              field holds a value these steppers cannot represent (non-integer
              or an id outside the enabled cards); edit or clear it there to
              re-enable the steppers.</div>
          </div>
          <div class="setrow" id="row_split_mode" style="display:none"
            data-hay="even uneven sharding split rank-tp-ratio auto performance">
            <span class="lbl">shard split across ranks
              <span class="chg" title="changed from preset"></span></span>
            <select id="split_mode_select" onchange="splitModeChanged()">
              <option value="even">even (uniform shards)</option>
              <option value="auto">uneven auto (max KV)</option>
              <option value="auto-performance">uneven auto-performance</option>
              <option value="custom">custom vector (rank-tp-ratio field)</option>
            </select>
            <span class="qmark" title="even: uniform equal-size shards on every rank (clears --rank-tp-ratio). With --rank-gpu-id set this still runs the FORK path, just with a uniform split -- stock sglang cannot co-locate ranks at all. uneven auto: VRAM-proportional shards (--rank-tp-ratio auto, budgets from NVML totals minus --rank-auto-reserve-mib). uneven auto-performance: the VRAM split plus a measured MLP-family shift toward the compute-strong card. custom: an explicit per-rank integer vector typed into the rank-tp-ratio field below, which stays authoritative.">?</span>
          </div>
          <div id="gpu_placement" class="muted" style="margin:.25rem 0 .4rem">
            plan a model to see the prospective placement&hellip;</div>
          <div class="setrow" id="row_gpu_pick" style="display:none"
            data-hay="gpu card select base-gpu-id single-gpu pick">
            <span class="lbl">GPU (single-card run)
              <span class="chg" title="changed from preset"></span></span>
            <select id="gpu_pick_select" onchange="gpuPickChanged()"></select>
            <span class="qmark" title="Which physical card a single-GPU (tp=1) run uses. Default = the preset's rule pick (largest VRAM, then higher FLOPs, then first index); writes the stock --base-gpu-id flag, a CUDA-order index (FASTEST_FIRST) -- NOT the nvidia-smi/NVML index; both are labeled. Hidden for tp > 1, where rank-gpu-id owns the placement.">?</span>
          </div>
          <div id="secflags_gpu"></div>
        </div>
      </details>

      <details class="cfg-section" id="sec_speculative">
        <summary><b>Speculative decoding</b><span class="sec-sum" id="sum_speculative"></span></summary>
        <div class="sec-body">
          <div id="spec_draft_hint" class="muted"
            style="display:none;color:#e3a008;margin:.2rem 0 .3rem"></div>
          <div class="setrow" id="row_draft_pick"
            data-hay="draft model eagle eagle3 speculator speculative-draft-model local">
            <span class="lbl">draft model (local)
              <span class="chg" title="changed from preset"></span></span>
            <select id="draft_model_select" onchange="draftPickChanged()">
              <option value="">(none)</option></select>
            <span class="qmark" title="Matching LOCAL draft/speculator models for the selected base model (name-family match, EAGLE3/EAGLE heads). Picking one fills --speculative-draft-model-path below; the free-text field stays authoritative and accepts any path. Needed for speculative decoding when the checkpoint has no MTP head of its own.">?</span>
          </div>
          <div id="secflags_speculative"></div>
        </div>
      </details>

      <details class="cfg-section" id="sec_cache">
        <summary><b>Cache</b><span class="sec-sum" id="sum_cache"></span></summary>
        <div class="sec-body"><div id="secflags_cache"></div></div>
      </details>

      <details class="cfg-section" id="sec_serving">
        <summary><b>Serving</b><span class="sec-sum" id="sum_serving"></span></summary>
        <div class="sec-body">
          <div class="muted" style="margin:.2rem 0 .3rem">AUTHORITATIVE
            serving identity &mdash; always wins over a preset's argv on
            launch.</div>
          <div class="setrow" id="row_sv_served" data-hay="served-model-name served model name">
            <span class="lbl">served-model-name
              <span class="chg" title="changed from preset"></span></span>
            <input id="sv_served" placeholder="(default: model)" onchange="onServingEdit()">
            <span class="qmark" title="The name the OpenAI API serves this model under.">?</span>
          </div>
          <div class="setrow" id="row_sv_host" data-hay="host bind address">
            <span class="lbl">host
              <span class="chg" title="changed from preset"></span></span>
            <input id="sv_host" placeholder="127.0.0.1" onchange="onServingEdit()">
            <span class="qmark" title="Bind address of the HTTP server.">?</span>
          </div>
          <div class="setrow" id="row_sv_port" data-hay="port">
            <span class="lbl">port
              <span class="chg" title="changed from preset"></span></span>
            <input id="sv_port" value="30000" onchange="onServingEdit()">
            <span class="qmark" title="TCP port of the HTTP server.">?</span>
          </div>
          <div id="secflags_serving"></div>
        </div>
      </details>

      <div class="advrow">
        <div class="setrow" style="margin:0" id="row_advanced">
          <span class="lbl">Show advanced settings
            <span class="muted" id="adv_count"></span></span>
          <label class="switch"><input type="checkbox" id="advanced_toggle"
            onchange="toggleAdvanced()"><span class="track"></span></label>
        </div>
        <div id="sec_advanced" style="display:none;margin-top:.3rem"></div>
      </div>

      </div>
    </fieldset>

    <!-- sticky action bar: load / eject / restart + status chip + boot log -->
    <div class="actionbar" id="action_bar">
      <div class="actions" style="margin-top:0;align-items:center">
        <button onclick="serverStart()">Load model</button>
        <button class="secondary" onclick="serverStop()">Eject</button>
        <button class="secondary" onclick="serverRestart()">Restart (replace)</button>
        <span class="chip" id="status_chip">stopped</span>
        <button class="secondary mini" onclick="doPlan()">Plan / re-validate</button>
        <button class="secondary mini" onclick="refreshServerStatus()">status</button>
      </div>
      <details id="boot_log" style="margin-top:.4rem">
        <summary class="muted" style="cursor:pointer">boot log / status detail</summary>
        <div id="sv_out" class="muted" style="margin-top:.3rem"></div>
      </details>
      <details style="margin-top:.3rem">
        <summary class="muted" style="cursor:pointer">GitHub issue &mdash; submit results / report bug</summary>
        <label>quant descriptor (for the issue text)</label>
        <input id="quant" placeholder="compressed-tensors / Q4_K_M / fp8">
        <div class="actions" style="margin-top:.4rem">
          <button class="secondary" onclick="doIssue('results')">Submit config (RESULTS issue)</button>
          <button class="secondary" onclick="doIssue('bug')">Report bug</button>
        </div>
      </details>
    </div>

    <details style="margin-top:.6rem">
      <summary style="cursor:pointer"><b>hardware</b> <span class="muted">&mdash;
        cards to plan on (auto-detected; open to edit or add hypothetical cards)</span></summary>
      <fieldset style="margin-top:.4rem">
        <legend>cards
          <button class="secondary mini" onclick="detectGPUs()" title="re-query NVML">detect</button></legend>
        <div id="detect_note" class="muted" style="margin-bottom:.4rem">detecting local GPUs…</div>
        <div id="cardlist" class="cardlist"></div>
        <div class="actions" style="margin-top:.4rem">
          <button class="secondary mini" onclick="addCard(false)">+ add card</button>
          <button class="secondary mini" onclick="addCard(true)" title="a hypothetical GPU you don't own yet">+ add future GPU</button>
        </div>
        <div class="muted" style="margin-top:.4rem">Tick = include in the plan.
          &ldquo;keep free&rdquo; is per-card headroom (a display / another
          process) carved off before sizing.</div>
        <label style="margin-top:.5rem">Host RAM total (GiB) &mdash; for RAM-offload fit</label>
        <input id="host_ram_gb" placeholder="(auto-detected; override to plan another host)">
      </fieldset>
    </details>

    <details style="margin-top:.6rem">
      <summary style="cursor:pointer"><b>download a model</b> <span class="muted">&mdash;
        external HF fetch (occasional task)</span></summary>
      <fieldset style="margin-top:.4rem">
        <legend>download</legend>
        <label>repo id (org/name)</label>
        <input id="dl_repo" placeholder="unsloth/Qwen3.6-27B-GGUF" onchange="dlTargets()">
        <label>model root (mount)</label>
        <input id="dl_root" placeholder="(default first root)" onchange="dlTargets()">
        <div id="dl_variants" style="display:none">
          <label>GGUF quant</label>
          <select id="dl_quant"></select>
        </div>
        <div style="margin-top:.5rem">
          <button onclick="dlPreview()" id="dl_btn">Preview + download</button>
        </div>
        <div id="dl_out" class="muted" style="margin-top:.5rem"></div>
      </fieldset>
    </details>
  </div>

  <div>
    <fieldset>
      <legend>prospective placement + per-card fit (updates as flags change)</legend>
      <div id="runner_placement" class="muted">plan a model to see the granular placement&hellip;</div>
    </fieldset>
    <div id="verdict"></div>
    <div id="split"></div>
    <div id="cards"></div>
    <div id="advantage"></div>
    <div id="pricebar" class="pricebar">
      energy price
      <input id="kwh_price" type="number" step="1" min="0" value="30"
             oninput="if (window.__lastMeasured !== undefined) renderMeasured(window.__lastMeasured); if (window.__lastHicache !== undefined) renderHicacheSaved(window.__lastHicache); if (window.__lastRoofline !== undefined) renderRoofline(window.__lastRoofline, window.__lastRooflineEnergy)">
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
/*__VENDOR_MORPHDOM__*/
const $ = id => document.getElementById(id);

// ===========================================================================
// Update foundation.
//
// The rule this section exists to enforce: a refresh changes numbers and
// nothing else. It never closes a collapse the reader opened, never discards
// a scroll position, never overwrites a field that is being typed into, and
// never lets a slow answer land on top of a newer one.
//
// Panels are still described as HTML strings -- that part of the existing
// design is fine and keeps rendering readable -- but the string is now
// diffed against the live tree instead of replacing it. Surviving nodes keep
// their identity, and with it every piece of state the browser hangs off a
// node rather than off its markup: <details open>, scrollTop, focus,
// selection range, and the current value of an input.
//
// The tree diff is morphdom (MIT, vendored, inlined above -- see
// assets/README.md). It supplies the algorithm; the rules below about what
// must survive an update are ours and live in the onBeforeElUpdated hook.
// ===========================================================================

// Fields carry live state that markup does not describe.
function _isField(n){
  const t=n&&n.tagName;
  return t==='INPUT'||t==='TEXTAREA'||t==='SELECT';
}

// True when the node is, or contains, whatever the user is working in.
function _busy(node){
  const a=document.activeElement;
  if(!a||a===document.body) return false;
  return node===a||(node.contains&&node.contains(a));
}

// Node identity for re-matching across renders: morphdom keys on id by
// default; data-key extends that to nodes that have no business owning a
// global id (the two collapses inside the landing start-config block).
function _nodeKey(n){
  if(!n||n.nodeType!==1) return '';
  return n.id||n.getAttribute('data-key')||'';
}

// The policy hook. Returning false tells morphdom to leave the node and its
// subtree exactly as they are.
function _beforeElUpdated(fromEl,toEl){
  // A field being edited is untouchable -- re-enabling or re-labelling it
  // mid-keystroke is exactly the sort of false update this rework removes.
  if(_isField(fromEl)&&_busy(fromEl)) return false;
  // The open state of a collapse belongs to the reader, not to the renderer.
  // Deliberate opens go through openDetails(), which is not undone here.
  if(fromEl.tagName==='DETAILS'&&fromEl.open!==toEl.open){
    if(fromEl.open) toEl.setAttribute('open',''); else toEl.removeAttribute('open');
  }
  // Scroll positions survive: a <pre> the reader scrolled through must not
  // jump back to the top because a number above it changed.
  if(fromEl.scrollTop||fromEl.scrollLeft){
    const t=fromEl.scrollTop, l=fromEl.scrollLeft;
    _restore.push(()=>{ fromEl.scrollTop=t; fromEl.scrollLeft=l; });
  }
  return true;
}
let _restore=[];

// The replacement for `el.innerHTML = html`.
//
// Two short-circuits before any DOM work: identical markup does nothing at
// all (this is what makes a 2 s poll free when nothing moved), and a panel
// the user is typing inside is deferred until focus leaves it.
function setHTML(el,html){
  if(!el) return;
  if(el._lastHTML===html) return;
  if(_busy(el)&&_isField(document.activeElement)){
    el._pendingHTML=html;
    if(!el._flushBound){
      el._flushBound=true;
      el.addEventListener('focusout',()=>{
        setTimeout(()=>{
          if(el._pendingHTML!=null&&!_busy(el)){
            const h=el._pendingHTML; el._pendingHTML=null; setHTML(el,h);
          }
        },0);
      });
    }
    return;
  }
  el._pendingHTML=null;
  const act=document.activeElement;
  const sel=(act&&_isField(act)&&act.selectionStart!=null)
    ?{s:act.selectionStart,e:act.selectionEnd}:null;
  // childrenOnly: `el` itself is the container the caller owns; only its
  // contents are described by `html`.
  const holder=document.createElement(el.tagName==='SELECT'?'select':'div');
  holder.innerHTML=html;
  // Rebuilding a <select>'s own option list: the options are markup, the
  // selection is state. morphdom's SELECT handler derives selectedIndex from
  // the incoming markup, so the selection is put back explicitly afterwards
  // if it is still on offer.
  const wasSel=(el.tagName==='SELECT')?el.value:null;
  _restore=[];
  morphdom(el,holder,{childrenOnly:true, getNodeKey:_nodeKey,
                      onBeforeElUpdated:_beforeElUpdated});
  for(const f of _restore) f();
  _restore=[];
  if(wasSel!=null&&Array.prototype.some.call(el.options,o=>o.value===wasSel))
    el.value=wasSel;
  if(act&&document.contains(act)&&document.activeElement!==act){
    try{
      act.focus();
      if(sel&&act.setSelectionRange) act.setSelectionRange(sel.s,sel.e);
    }catch(e){}
  }
  el._lastHTML=html;
}

// A deliberate open, e.g. "the boot failed, show the log". Recorded so a
// later poll does not keep re-opening what the reader closed again.
function openDetails(el){
  if(!el||el._autoOpened) return;
  el._autoOpened=true;
  el.open=true;
}

// ---- backend calls --------------------------------------------------------
// Every call is bounded and cancellable. `key` names a logical slot: issuing
// a second call under the same key aborts the first, so the answer that
// lands is always the answer to the newest question.
const API_TIMEOUT_MS=15000;
window._inflight={};
async function api(path,opts){
  opts=opts||{};
  const key=opts.key||path;
  const prev=window._inflight[key];
  if(prev){ try{ prev.abort(); }catch(e){} }
  const ac=new AbortController();
  window._inflight[key]=ac;
  const timer=setTimeout(()=>{ try{ ac.abort(); }catch(e){} },
                         opts.timeout||API_TIMEOUT_MS);
  try{
    const init={signal:ac.signal};
    if(opts.body!==undefined){
      init.method=opts.method||'POST';
      init.body=JSON.stringify(opts.body);
    } else if(opts.method) init.method=opts.method;
    const r=await fetch(path,init);
    return await r.json();
  } finally {
    clearTimeout(timer);
    if(window._inflight[key]===ac) delete window._inflight[key];
  }
}
// An aborted call is a superseded or timed-out call, never a real failure --
// callers use this to stay quiet instead of painting an error over good data.
function apiAborted(e){
  return !!e&&(e.name==='AbortError'||e.name==='TimeoutError');
}
function apiError(e){ return apiAborted(e)?'':(''+(e&&e.message||e)); }

// Mark a panel as refreshing without replacing what it shows.
//
// The dim engages only after STALE_AFTER_MS, so a fast answer -- which is
// the normal case -- produces no visible change at all. Flashing a panel on
// every 2 s poll would be its own kind of jitter, and a spinner that blanks
// the panel would throw away the numbers the reader is looking at.
const STALE_AFTER_MS=250;
function stale(el,on){
  if(!el) return;
  if(on===false){
    if(el._staleTimer){ clearTimeout(el._staleTimer); el._staleTimer=null; }
    el.classList.remove('stale');
    return;
  }
  if(el._staleTimer) return;
  el._staleTimer=setTimeout(()=>{ el._staleTimer=null; el.classList.add('stale'); },
                            STALE_AFTER_MS);
}

// One debounce implementation. Every control that triggers a backend call
// on each tick (sliders above all) goes through it.
function debounce(fn,ms){
  let t=null;
  return function(){
    const args=arguments, self=this;
    if(t) clearTimeout(t);
    t=setTimeout(()=>{ t=null; fn.apply(self,args); },ms);
  };
}

// ===========================================================================
// Simple / expert.
//
// Two densities of one page, not two pages. Visibility is a single class on
// <body>, so no panel has to know which mode it is in and switching costs no
// backend call. Controls hidden in simple mode keep their values and are
// still read when the configuration is assembled -- expert ADDS dimensions,
// it never changes what a shared control means.
// ===========================================================================
const VIEW_MODES=['simple','expert'];
function viewMode(){
  try{
    const m=localStorage.getItem('view_mode');
    if(VIEW_MODES.indexOf(m)>=0) return m;
  }catch(e){}
  return 'simple';
}
function setViewMode(m){
  if(VIEW_MODES.indexOf(m)<0) m='simple';
  try{ localStorage.setItem('view_mode',m); }catch(e){}
  applyViewMode();
}
function applyViewMode(){
  const m=viewMode();
  document.body.classList.toggle('mode-simple', m==='simple');
  document.body.classList.toggle('mode-expert', m==='expert');
  for(const k of VIEW_MODES){
    const b=$('vm_'+k); if(b) b.classList.toggle('on', k===m);
  }
  // The mode decides which panels are worth computing, so a switch asks for
  // the sections the newly visible panels need.
  scheduleRecompute();
}

// Serving identity is owned by the MODEL selector + the SERVING form group,
// NOT by the flag surface -- these catalog ids are hidden there and routed to
// their single authoritative field instead (no fact is entered twice).
const SERVING_OWNED = {model_path:1, tokenizer_path:1, served_model_name:1,
                       port:1, host:1, context_length:1, max_running_requests:1};

// Debounced re-plan: any flag/model edit re-plans without hammering the API.
window._planTimer = null;
function schedulePlan() {
  if (!$('model').value.trim()) return;
  if (window._planTimer) clearTimeout(window._planTimer);
  window._planTimer = setTimeout(doPlan, 400);
}

// ---- hardware card list (detect + select + per-card reserve + virtual) ----
// {name,total_mib,include,reserve_gb,virtual,free_mib,nvml_index,cuda_index}
// TWO index spaces per detected card: nvml_index (NVML/PCI, telemetry) and
// cuda_index (CUDA/FASTEST_FIRST -- the space --rank-gpu-id/--base-gpu-id
// are interpreted in). They diverge on mixed rigs; card rows label BOTH.
let CARDS = [];
let HOST_RAM_MIB = null;
// cuda_index -> nvml_index (for dual labels wherever only cuda is known),
// maintained from the detect payload and the live snapshot.
window._cudaNvml = {};
window._cudaMapHeuristic = false;
function noteCudaMap(gpus, source){
  for (const g of (gpus||[]))
    if (g.cuda_index != null)
      window._cudaNvml[g.cuda_index] = (g.nvml_index != null ? g.nvml_index : g.index);
  if (source === 'heuristic') window._cudaMapHeuristic = true;
}
// Dual-space device label: 'cuda:0 / nvml:1' (either half degrades alone).
function devLabel(cudaIdx, nvmlIdx){
  if (cudaIdx == null && nvmlIdx == null) return '';
  if (cudaIdx != null && nvmlIdx == null && window._cudaNvml[cudaIdx] != null)
    nvmlIdx = window._cudaNvml[cudaIdx];
  const parts = [];
  if (cudaIdx != null) parts.push('cuda:'+cudaIdx+(window._cudaMapHeuristic?'?':''));
  if (nvmlIdx != null) parts.push('nvml:'+nvmlIdx);
  return parts.join(' / ');
}

async function detectGPUs() {
  const r = await fetch('/api/detect'); const d = await r.json();
  if (d.host_ram_mib) { HOST_RAM_MIB = d.host_ram_mib;
    if (!$('host_ram_gb').value) $('host_ram_gb').placeholder =
      '(detected '+(d.host_ram_mib/1024).toFixed(0)+' GiB; override to plan another host)'; }
  if (d.ok && d.gpus && d.gpus.length) {
    // Replace any previously-detected rows; keep user-added virtual cards.
    // Detected rows are ordered by CUDA index (the --rank-gpu-id space) so
    // list position == cuda index wherever the bridge resolved; the old
    // per-item unshift additionally REVERSED the detect order.
    const det = d.gpus.map(g => ({
      name: g.name, total_mib: g.total_mib, include: true,
      reserve_gb: 0, virtual: false, free_mib: g.free_mib,
      nvml_index: g.index, cuda_index: (g.cuda_index!=null?g.cuda_index:null) }))
      .sort((a,b)=>((a.cuda_index!=null?a.cuda_index:1e9)
                   -(b.cuda_index!=null?b.cuda_index:1e9)) || (a.nvml_index-b.nvml_index));
    CARDS = det.concat(CARDS.filter(c => c.virtual));
    noteCudaMap(d.gpus, d.cuda_index_source);
    $('detect_note').textContent = 'detected '+d.gpus.length+' GPU(s) via '+(d.source||'nvml')
      +(d.cuda_index_source==='heuristic'
        ? ' — cuda indices are a FASTEST_FIRST emulation (no torch bridge), marked "?"'
        : '');
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
    // Detected cards: label BOTH index spaces so a number is never ambiguous
    // (cuda = the --rank-gpu-id/--base-gpu-id space, nvml = telemetry space).
    const idxLbl = devLabel(c.cuda_index, c.nvml_index);
    if (idxLbl) {
      const idxTag = document.createElement('span'); idxTag.className = 'cap';
      idxTag.title = 'cuda: CUDA enumeration order (FASTEST_FIRST) — the space '
        + '--rank-gpu-id/--base-gpu-id use; nvml: NVML/PCI-bus order (nvidia-smi, telemetry)';
      idxTag.textContent = ' [' + idxLbl + ']';
      nameWrap.appendChild(idxTag);
    }
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
    res.onchange = () => { c.reserve_gb = parseFloat(res.value) || 0; renderCards(); };
    resWrap.appendChild(res); resWrap.appendChild(document.createTextNode(' G'));
    row.appendChild(cb); row.appendChild(nameWrap);
    row.appendChild(vramWrap); row.appendChild(resWrap);
    // per-card VRAM bar (LM-Studio hardware rows): used (detected) /
    // reserve (kept free on purpose) / free, proportional to the total.
    const total = Math.max(1, c.total_mib||1);
    const usedMib = (c.free_mib!=null && !c.virtual)
      ? Math.max(0, c.total_mib - c.free_mib) : 0;
    const resvMib = Math.min(total, (c.reserve_gb||0)*1024);
    const barWrap = document.createElement('div');
    barWrap.className = 'cardbar';
    barWrap.innerHTML = '<div class="segbar" style="height:8px" title="'
      +(usedMib? 'in use '+fmtMib(usedMib)+' / ':'')
      +(resvMib? 'kept free '+fmtMib(resvMib)+' / ':'')
      +'free '+fmtMib(Math.max(0, total-usedMib-resvMib))
      +' of '+fmtMib(total)+'">'
      +(usedMib? '<span style="width:'+(usedMib/total*100).toFixed(1)+'%;background:#2f81f7"></span>':'')
      +(resvMib? '<span style="width:'+(resvMib/total*100).toFixed(1)+'%;background:#e3a008"></span>':'')
      +'</div>';
    row.appendChild(barWrap);
    box.appendChild(row);
  });
  updateGpuPick(); updateColoUI();
}

// The ONE model selector's state: the free-text field is the reference; the
// discovered-models dropdown fills it; _modelMeta remembers format + GGUF
// variants (from discovery or from /api/gguf_options for a typed ref).
async function onModelChange() {
  const m = $('model').value.trim();
  $('gguf_pick').style.display = 'none';
  window._modelMeta = null;
  if (!m) return;
  const r = await fetch('/api/gguf_options', {method:'POST', body: JSON.stringify({model:m})});
  const d = await r.json();
  const opts = (d.ok && d.options) || [];
  if (opts.length) {
    window._modelMeta = {format:'gguf', variants: opts.map(o=>({filename:o}))};
    if (opts.length > 1) {
      setHTML($('gguf_choice'), opts.map(o=>'<option value="'+esc(o)+'">'+esc(o)+'</option>').join(''));
      $('gguf_pick').style.display = '';
    }
  }
}

// {path, format, gguf_variant} derived from the single selector -- the launch
// path and "save current as profile" read model identity ONLY from here.
function modelState() {
  const path = $('model').value.trim();
  const meta = window._modelMeta || {};
  const shown = $('gguf_pick').style.display !== 'none';
  let variant = null;
  if (shown) variant = $('gguf_choice').value || null;
  else if (meta.variants && meta.variants.length === 1)
    variant = meta.variants[0].filename || null;
  const format = (shown || meta.format === 'gguf' || /\.gguf$/i.test(path))
    ? 'gguf' : 'hf';
  return {path, format, gguf_variant: format === 'gguf' ? variant : null};
}

// Plan-relevant flags come straight from the ONE flag surface (fl_* /
// _flagSettings); the kv-token env flag maps onto the planner's knob name.
const PLAN_FLAG_KEYS = ['rank_gpu_id','rank_gpu_memory_mib','rank_tp_ratio',
  'rank_mlp_ratio','rank_moe_ratio','rank_vocab_ratio','dcp_size'];
function payload() {
  const s = window._flagSettings || {};
  // The posted card list is POSITIONAL: the backend re-indexes included
  // cards 0..k-1 and those positions ARE the --rank-gpu-id space (CUDA
  // order). Post detected cards sorted by cuda_index (detectGPUs already
  // orders CARDS that way; the sort keeps it true after manual edits),
  // virtual cards after.
  const cards = CARDS.slice()
    .sort((a,b)=>((a.cuda_index!=null?a.cuda_index:1e9)
                 -(b.cuda_index!=null?b.cuda_index:1e9)))
    .map(c => ({ name: c.name, total_mib: c.total_mib,
    include: c.include, reserve_gb: c.reserve_gb, virtual: c.virtual }));
  const hostGb = $('host_ram_gb').value.trim();
  const host_ram_mib = hostGb ? Math.round(parseFloat(hostGb)*1024) : HOST_RAM_MIB;
  const p = {
    model: $('model').value.trim(),
    hardware: { source: 'cards', cards, host_ram_mib },
    tp_size: s.tp_size ? parseInt(s.tp_size) : null,
    max_running_requests: $('max_running_requests').value ? parseInt($('max_running_requests').value) : null,
    include_vision: $('include_vision').checked,
    kv_cache_dtype: s.kv_cache_dtype || 'auto',
    quant: $('quant').value.trim(),
  };
  if ($('gguf_pick').style.display !== 'none') p.gguf_choice = $('gguf_choice').value;
  for (const k of PLAN_FLAG_KEYS)
    if (s[k] != null && s[k] !== '') p[k] = s[k];
  if (s.SGLANG_UNEVEN_TOKEN_VECTOR != null && s.SGLANG_UNEVEN_TOKEN_VECTOR !== '')
    p.kv_token_vector = s.SGLANG_UNEVEN_TOKEN_VECTOR;
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
  // Co-location gate: an incomplete per-card rank assignment (sum != tp)
  // blocks planning with the same inline error the section shows.
  const coloErr = coloBlockError();
  if (coloErr) {
    $('verdict').innerHTML = '<div class="verdict nofit">PLAN BLOCKED</div>';
    $('split').innerHTML = '<ul class="reasons"><li>' + esc(coloErr) + '</li></ul>';
    return;
  }
  // Clear any prior verdict immediately so a slow sizing (e.g. a first-time
  // GGUF header fetch) never leaves a stale REJECTED on screen — the exact
  // "re-validate does nothing" confusion.
  $('verdict').innerHTML = '<div class="verdict">sizing…</div>';
  $('split').innerHTML = '';
  try {
    const d = await api('/api/plan', {key:'plan', body:payload()});
    render(d);
  } catch (e) {
    // Superseded by a newer edit: the newer plan owns the panel, say nothing.
    if (apiAborted(e)) return;
    $('verdict').innerHTML = '<div class="verdict nofit">PLAN ERROR</div>';
    $('split').innerHTML = '<ul class="reasons"><li>' + esc(apiError(e)) + '</li></ul>';
  }
}

function render(d) {
  if (d.valid === false) {
    $('verdict').innerHTML = '<div class="verdict nofit">PLAN REJECTED</div>';
    $('split').innerHTML = '<ul class="reasons">' +
      d.reasons.map(x=>'<li>'+esc(x)+'</li>').join('') + '</ul>';
    $('cards').innerHTML = $('advantage').innerHTML = $('flags').innerHTML = '';
    $('roofline').innerHTML = '';
    setCtxCap(null);
    return;
  }
  const cap = d.capacity;
  const off = d.offload;
  // context-length slider: clamp to the plan's capacity max when known.
  setCtxCap(cap ? cap.max_context_tokens : null);
  // concurrency slider: the range follows the highest rung the planner
  // reports as fitting, doubled so the reader can walk past it and see the
  // rejection with its reason. A limit is a hard reject, not a preset value.
  const rungs=(d.kv_by_concurrency||[]).filter(r=>r.fits).map(r=>r.concurrency);
  setMrrCap(rungs.length? Math.max(64, Math.max.apply(null,rungs)*2) : 64);
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
    // NEUTRAL side-by-side framing: when stock runs, its numbers stand next
    // to the planned config without a verdict; when stock genuinely cannot
    // express the shape, the divisibility rule is stated as a fact.
    av += '<div class="adv"><b>alongside: stock even-TP</b> (capacity/feasibility &mdash; no throughput guess)<br>';
    if (a.stock.runs) {
      av += 'stock even-TP runs here; max context ~'+Math.round(a.stock.max_context_tokens).toLocaleString()+' tokens.';
      if (a.capacity_pct_range) {
        const [lo,hi]=a.capacity_pct_range;
        av += '<br>capacity: <b>'+(lo>=0?'+':'')+lo+'% .. '+(hi>=0?'+':'')+hi+'%</b> KV/context '
          +'<span class="est">(ratio of two same-model estimates)</span>';
      }
    } else {
      av += 'stock even-TP is not expressible for this shape:<ul class="reasons">'
        + a.stock.reasons.map(x=>'<li>'+esc(x)+'</li>').join('')
        + '</ul>' + (d.fits ? 'This configuration uses the fork path.' : '');
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
  renderRoofline(d.roofline_estimate, d.roofline_energy);
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

function renderRoofline(rf, re) {
  window.__lastRoofline = rf;         // cached so a price edit re-renders energy
  window.__lastRooflineEnergy = re;
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
    + rf.caveats.map(c => '<li>' + esc(c) + '</li>').join('') + '</ul>';
  h += rooflineEnergyHtml(re);
  h += '</div>';
  $('roofline').innerHTML = h;
}

// -- #148 estimated J/token (total + per card) for a VIRTUAL / planned rig ----
// The ENERGY sibling of the throughput roofline: predicts the heterogeneous
// efficiency gap (a virtual 5090 vs 3080) from TDP + compute + membw + the
// uneven-DCP compute-share, BEFORE any measurement. Mirrors the MEASURED
// per-card table layout (raw J/tok + wpw(rel)) so estimated vs measured read on
// the same scale — but is styled as an ESTIMATE (amber, planner-estimate), NOT
// the measured green panel. Respects the kWh-price bar for estimated ct/1M.
function rooflineEnergyHtml(re) {
  if (!re) return '';
  const priceCt = getKwhPriceCt();
  let h = '<div class="rf-energy"><div class="rf-title">Estimated energy '
    + '(J/token) &mdash; ESTIMATE, planner-estimate provenance, NOT measured</div>'
    + '<p class="est">Predicted from each card\'s TDP + compute + membw and its '
    + 'uneven-DCP compute-share (shared lockstep step-time from the roofline '
    + 'above). This PREDICTS the hetero efficiency gap before any run &mdash; a '
    + 'faster card that finishes its shard early and waits draws less, so its '
    + 'work-per-watt is higher. kWh/1M = J/token &divide; 3.6; cost = kWh &times; '
    + priceCt + '&nbsp;ct/kWh.</p>';
  if (re.measured_available)
    h += '<p class="est"><b>A measured energy entry exists for this config '
      + '(shown above) &mdash; this estimate is secondary.</b></p>';
  h += rfEnergyPhase(re, 'prefill', priceCt);
  h += rfEnergyPhase(re, 'decode', priceCt);
  h += '<p class="est">idle floor = ' + Math.round(re.idle_fraction_of_tdp*100)
    + '% of TDP (heuristic). ' + esc(re.caveats[re.caveats.length-1]) + '</p>';
  h += '</div>';
  return h;
}

// One phase (PREFILL or DECODE) of the estimated per-card energy table, laid
// out to MATCH the measured perCardPhase: card | compute | W | J/tok (raw) |
// wpw(rel) | kWh/1M | ct/1M, with a TOTAL row. wpw(rel) = compute-share /
// J-per-token, rescaled so the LEAST efficient card = 1.0x (same as measured).
function rfEnergyPhase(re, phase, priceCt) {
  const cards = phase === 'prefill' ? re.prefill : re.decode;
  if (!cards || !cards.length) return '';
  const totJ = phase === 'prefill' ? re.j_per_prefill_token_total
                                    : re.j_per_decode_token_total;
  const t = phase === 'prefill' ? re.t_prefill_token_s : re.t_decode_token_s;
  const effs = cards.map(c => (c.compute_share > 0 && c.j_per_token > 0)
    ? c.compute_share / c.j_per_token : null);
  const effMin = Math.min(...effs.filter(x => x != null && x > 0));
  let h = '<p class="muted" style="margin:.4rem 0 .1rem"><b>' + phase.toUpperCase()
    + '</b> &mdash; est. per card @ shared ' + (t*1000).toFixed(2)
    + ' ms/token. <b>J/tok is RAW</b> (lockstep &mdash; same tokens on every '
    + 'card, NOT efficiency); <b>wpw(rel)</b> = work-per-watt normalized by '
    + 'compute-share = the predicted hetero-efficiency (least efficient = 1.0&times;).</p>';
  h += '<table><tr><th>card</th><th>compute</th><th>util</th><th>W (est)</th>'
    + '<th>J/tok (raw)</th><th>wpw(rel)</th><th>kWh/1M</th><th>ct/1M</th></tr>';
  for (let i = 0; i < cards.length; i++) {
    const c = cards[i];
    const rel = (effs[i] != null && effMin > 0) ? (effs[i]/effMin) : null;
    h += '<tr><td style="text-align:left">GPU ' + c.gpu_index + ' ' + esc(c.gpu_name)
      + '</td><td>' + Math.round(c.compute_share*100) + '%</td><td>'
      + Math.round(c.util*100) + '%</td><td>' + _f1(c.watts) + '</td><td>'
      + _f3(c.j_per_token) + '</td><td>' + (rel != null ? rel.toFixed(2)+'&times;' : '—')
      + '</td><td>' + _f2(c.j_per_token/3.6) + '</td><td>'
      + _fct(jPerTokToCtPer1M(c.j_per_token, priceCt)) + '</td></tr>';
  }
  h += '<tr><td style="text-align:left"><b>TOTAL</b></td><td>&mdash;</td><td>&mdash;</td>'
    + '<td>&mdash;</td><td><b>' + _f3(totJ) + '</b></td><td>&mdash;</td><td>'
    + _f2(totJ/3.6) + '</td><td>' + _fct(jPerTokToCtPer1M(totJ, priceCt))
    + '</td></tr></table>';
  return h;
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

function esc(s){return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}

function showTab(t) {
  const TABS = ['landing','runner','bench','explore','landscape','energy','quality','pair'];
  for (const v of TABS) $('view_'+v).style.display = t===v ? '' : 'none';
  for (const v of TABS) $('tab_'+v).classList.toggle('active', t===v);
  if ((t==='explore'||t==='landscape') && !window._profLoaded) loadProfiles();
  if (t==='energy' && !window._energyInit) { window._energyInit=true; loadPowerProfile(); }
  if (t==='quality') { autofillQuality(); if(!window._qualityInit){window._qualityInit=true; loadShots();} }
  if (t==='runner' && !window._runnerInit) {
    window._runnerInit=true; detectGPUs(); loadModels();
    loadFlagCatalog(); refreshServerStatus();
  }
  if (t==='bench' && !window._benchInit) { window._benchInit=true; benchInit(); }
  // The lead-metric poll runs only while its tab is visible.
  if (t!=='bench') benchLeadStop();
  // Pairing state lives on the host, so entering the tab READS it rather than
  // creating anything -- a flow started from a script shows up here.
  if (t==='pair') pairRefresh(); else pairStopPoll();
  // The landing page live-poll runs only while its tab is visible.
  if (t==='landing') startLanding(); else stopLanding();
  // The Planner opens on the RUNNING configuration when there is one: the
  // useful default is what is loaded, not an empty form.
  if (t==='runner') prefillFromRunning();
}

// ===========================================================================
// Task #214 -- couple a rig. STEERING ONLY.
//
// Every decision in this flow is made on the host: whether the far rig is
// reachable, whether the two are compatible, which transport the measurements
// support, what the resulting configuration is. This code posts a command and
// renders the state that comes back. It contains no rule about what makes a
// pairing valid, and it must not grow one -- the same flow has to behave
// identically when driven by a shell script, and a rule that lives here would
// simply be absent there.
//
// Consequently the session id is the only client-side state, and it is kept
// in localStorage so a reload rejoins a flow already running on the host.
// ===========================================================================
const PAIR_POLL_MS=1200;
window._pairTimer=null;
function pairSession(){
  try{ return localStorage.getItem('pair_session')||''; }catch(e){ return ''; }
}
function setPairSession(id){
  try{ if(id) localStorage.setItem('pair_session',id);
       else localStorage.removeItem('pair_session'); }catch(e){}
}
function pairStopPoll(){
  if(window._pairTimer){ clearInterval(window._pairTimer); window._pairTimer=null; }
}
function pairStartPoll(){
  if(window._pairTimer) return;
  window._pairTimer=setInterval(pairRefresh, PAIR_POLL_MS);
}
async function pairStart(){
  const target=$('pair_target').value.trim();
  if(!target){ $('pair_note').textContent='enter the far rig as host:port.'; return; }
  $('pair_note').textContent='creating session…';
  try{
    const d=await api('/api/rig_pair/start',{key:'pair', body:{target}});
    if(!d.ok){ $('pair_note').textContent=d.error+(d.remedy?(' — '+d.remedy):''); return; }
    setPairSession(d.session.session_id);
    renderPair(d.session);
    pairAdvance();
  }catch(e){ if(!apiAborted(e)) $('pair_note').textContent=apiError(e); }
}
async function pairAdvance(step){
  const sid=pairSession(); if(!sid) return;
  try{
    const d=await api('/api/rig_pair/advance',
      {key:'pair_advance', body:{session_id:sid, step:step||null}});
    if(d.ok){ renderPair(d.session); pairStartPoll(); }
    else $('pair_note').textContent=d.error;
  }catch(e){ if(!apiAborted(e)) $('pair_note').textContent=apiError(e); }
}
async function pairReset(){
  const sid=pairSession(); if(!sid) return;
  try{
    const d=await api('/api/rig_pair/reset',{key:'pair', body:{session_id:sid}});
    if(d.ok) renderPair(d.session);
  }catch(e){ if(!apiAborted(e)) $('pair_note').textContent=apiError(e); }
}
async function pairRefresh(){
  const sid=pairSession();
  if(!sid){ setHTML($('pair_steps'),'<span class="muted">enter a rig above to begin.</span>'); return; }
  try{
    const d=await api('/api/rig_pair/status?session_id='+encodeURIComponent(sid),
                      {key:'pair_status', timeout:PAIR_POLL_MS-200});
    if(!d.ok){ setPairSession(''); pairStopPoll(); return; }
    renderPair(d.session);
  }catch(e){}
}
const PAIR_TITLES={
  reach:'1 — reach the far rig',
  gate:'2 — compatibility gate',
  transport:'3 — transport',
  config:'4 — start configuration',
};
function pairStateClass(st){
  if(st==='ok') return 'fitc';
  if(st==='blocked'||st==='error') return 'nofitc';
  return 'muted';
}
function renderPair(s){
  if(!s) return;
  if($('pair_target')!==document.activeElement && !$('pair_target').value)
    $('pair_target').value=s.target||'';
  $('pair_note').innerHTML='session <code>'+esc(s.session_id)+'</code> &mdash; '
    +(s.complete?'complete':('next: '+esc(s.next_step||'—')));
  // The host says which step runs next, so the poll simply stops when it
  // says there is nothing running.
  const running=(s.steps||[]).some(x=>x.state==='running');
  if(!running) pairStopPoll();
  let h='';
  for(const st of (s.steps||[])){
    const t=PAIR_TITLES[st.step]||st.step;
    h+='<fieldset data-key="pairstep_'+st.step+'"><legend>'+esc(t)
      +' <span class="'+pairStateClass(st.state)+'">'+esc(st.state)+'</span></legend>';
    if(st.error)
      h+='<div class="reasons">'+esc(st.error)+'</div>';
    if(st.remedy)
      h+='<div class="adv"><b>remedy:</b> '+esc(st.remedy)+'</div>';
    h+=pairStepBody(st);
    if(st.state==='pending'||st.state==='blocked'||st.state==='error')
      h+='<div class="actions" style="margin-top:.4rem">'
        +'<button class="mini" onclick="pairAdvance(\''+st.step+'\')">'
        +(st.state==='pending'?'run this step':'run again')+'</button></div>';
    h+='</fieldset>';
  }
  setHTML($('pair_steps'), h);
}
function pairStepBody(st){
  const d=st.detail||{};
  if(st.step==='reach'){
    if(!d.url) return '';
    let h='<div class="cfgrow"><span class="cfgk">endpoint</span>'
      +'<span class="pill">'+esc(d.url)+'</span>';
    if(d.rtt_ms!=null) h+='<span class="pill">'+d.rtt_ms+' ms</span>';
    if((d.node_ids||[]).length) h+='<span class="pill">nodes: '+esc(d.node_ids.join(', '))+'</span>';
    h+='</div>';
    if(d.reachable && !d.identity)
      h+='<div class="muted">reachable, but the far rig reports no identity yet.</div>';
    return h;
  }
  if(st.step==='gate'){
    const rows=d.rows||[];
    if(!rows.length) return '';
    // Every unmet row shows its reason AND its remedy. Nothing is greyed out
    // without saying why, which is the capability table's rule.
    let h='<table class="mx"><tr><th>check</th><th>verdict</th><th>this rig</th>'
      +'<th>far rig</th><th style="text-align:left">why / what to do</th></tr>';
    for(const r of rows){
      h+='<tr><td style="text-align:left">'+esc(r.label||r.key)+'</td>'
        +'<td class="'+pairStateClass(r.verdict==='ok'?'ok':(r.verdict==='block'?'blocked':''))
        +'"><b>'+esc(r.verdict)+'</b></td>'
        +'<td>'+esc(r.local||'')+'</td><td>'+esc(r.remote||'')+'</td>'
        +'<td style="text-align:left">'+esc(r.reason||'')
        +(r.remedy?('<br><span class="muted">'+esc(r.remedy)+'</span>'):'')
        +'</td></tr>';
    }
    return h+'</table>';
  }
  if(st.step==='transport'){
    let h='<div>'+esc(d.note||'')+'</div>';
    if(d.offer)
      h+='<div class="adv"><b>not measured yet.</b> '+esc(d.offer.what||'')
        +' <span class="muted">'+esc(d.offer.cost||'')+'</span>'
        +'<div class="actions" style="margin-top:.3rem">'
        +'<button class="mini secondary" onclick="showTab(\'energy\')" '
        +'title="the study machinery lives in the measurement tabs; this flow never starts a run by itself">'
        +'open the measurement tools</button></div></div>';
    const pairs=(d.pairs||[]).filter(p=>!p.same_node);
    if(pairs.length){
      h+='<table class="mx"><tr><th>pair</th><th>chosen</th>'
        +'<th style="text-align:left">options</th></tr>';
      for(const p of pairs)
        h+='<tr><td style="text-align:left">'+esc(p.src)+' &rarr; '+esc(p.dst)+'</td>'
          +'<td>'+(p.chosen?('<b>'+esc(p.chosen)+'</b>'):'<span class="muted">unknown</span>')+'</td>'
          +'<td style="text-align:left">'+(p.options||[]).map(o=>
              '<span class="pill" title="'+esc(o.reason||'')+'">'+esc(o.key)+': '+esc(o.verdict)+'</span>'
            ).join(' ')+'</td></tr>';
      h+='</table>';
    }
    return h;
  }
  if(st.step==='config'){
    if(!d.env) return '';
    let h='';
    for(const n of (d.notes||[])) h+='<div class="muted">'+esc(n)+'</div>';
    h+='<pre>'+esc(Object.keys(d.env).map(k=>k+'='+d.env[k]).join('\n'))+'</pre>';
    h+='<div class="legend">Placeholders, not this machine\'s values: the '
      +'block is safe to paste anywhere.</div>';
    return h;
  }
  return '';
}

// ===========================================================================
// Shared granular placement renderer (used by BOTH the landing running-config
// view and the runner prospective view -- ONE renderer, two data sources).
// ===========================================================================
function fmtMib(x){ return (x==null)? '—' : (x>=1024? (x/1024).toFixed(2)+'G' : x.toFixed(0)+'M'); }
// One row/block per PHYSICAL CARD: name + total VRAM, a proportional bar of
// what occupies it (weights / KV / mamba / overhead / free), the co-located
// ranks, and the granular per-rank detail (head/token/expert index ranges,
// MTP, host-RAM offload) as compact secondary lines. Used by both the
// landing (running config) and the runner (prospective config); the landing
// additionally passes the live snapshot's gpus[] so each card block carries
// its CURRENT telemetry (util/clock/power/temp + 60s util/power sparklines)
// -- one place per card explains BOTH what occupies its VRAM and how it is
// doing live.
// Fine-grained segment palette (dark, one distinct hue per family; tiny
// segments below 1% of the card collapse into the neutral "other" sliver).
const SEG_COLORS = {
  attn_q:'#1f6feb', attn_kv:'#58a6ff', mlp:'#8957e5', experts:'#bc8cff',
  vocab:'#db6d28', gdn_w:'#9e6a03', vision:'#484f58', draft_w:'#f778ba',
  kv:'#2ea043', kv_draft:'#56d364', mamba:'#e3b341', graphs:'#ff7b72',
  ovh:'#6e7681', other:'#30363d',
  // legacy lump ids (fallback when a placement has no segments list)
  weights:'#2f81f7'};
// The live gpus[] row for a placement card: cards are CUDA-keyed
// (c.gpu_index == the rank_gpu_id space); gpus[] rows are NVML-sampled and
// carry the UUID-resolved device-map bridge cuda_index -- match on that
// (nvml_index only as the unbridged fallback, same degradation as devLabel).
function _liveGpuForCard(liveGpus, cudaIdx){
  for(const g of (liveGpus||[])){
    const k=(g.cuda_index!=null? g.cuda_index : g.nvml_index);
    if(k===cudaIdx) return g;
  }
  return null;
}
// Compact per-card live line: current numbers + ONE small sparkline row
// (60s util + power; ring keys are nvml-keyed, pushed by renderLanding).
// mem stays a live used/total text -- the placement bar above already shows
// the VRAM breakdown.
function cardLiveHtml(g){
  const key='gpu'+g.nvml_index;
  return '<div class="cardlive" style="margin:.15rem 0;font-size:.72rem" '
    +'title="live NVML telemetry (60s rings; util freezes at zero like the top strip)">'
    +'util <b>'+g.utilization_pct+'%</b> '+sparkline(key+'_util',80,18)
    +' &nbsp; pow <b>'+g.power_watts.toFixed(0)+'</b>/'+g.power_limit_w.toFixed(0)+'W '
    +sparkline(key+'_pow',80,18)
    +' &nbsp; SM '+g.sm_clock_mhz+'MHz &nbsp; '+g.temperature_c+'C'
    +' &nbsp; mem '+(g.mem_used_mib/1024).toFixed(1)+'/'+(g.mem_total_mib/1024).toFixed(1)+'G live'
    +'</div>';
}
// One card's fine-grained segment bar + legend. Segments come READY-MADE from
// the placement JSON (id/label/mib/detail/replicated) -- the hover tooltip is
// the segment's own detail string (title attr: no layout shift), replicated
// segments get the hatched overlay + "replicated xN" tag, and segments under
// 1% of the card collapse into ONE neutral "other" sliver whose hover lists
// them, so the bar stays readable.
function segBarHtml(c, total, free){
  const segs=c.segments && c.segments.length ? c.segments
    // legacy fallback (older placement payloads): the coarse lumps.
    : [{id:'weights',label:'weights',mib:c.weight_mib,detail:''},
       {id:'kv',label:'KV',mib:c.kv_mib,detail:''},
       {id:'mamba',label:'mamba',mib:c.mamba_mib,detail:''},
       {id:'ovh',label:'overhead',mib:c.overhead_mib,detail:''}];
  const main=[], tiny=[];
  for(const sg of segs){
    if(!(sg.mib>0)) continue;
    ((sg.mib/total*100) < 1 ? tiny : main).push(sg);
  }
  let bar='<div class="segbar">';
  for(const sg of main){
    const pct=Math.min(100, sg.mib/total*100);
    bar+='<span '+(sg.replicated?'class="hatch" ':'')
      +'style="width:'+pct.toFixed(2)+'%;background:'+(SEG_COLORS[sg.id]||'#30363d')+'" '
      +'title="'+esc(sg.label+' '+fmtMib(sg.mib)
        +(sg.replicated?' [replicated x'+sg.replication_factor+']':'')
        +' — '+(sg.detail||''))+'"></span>';
  }
  const tinySum=tiny.reduce((a,s)=>a+s.mib,0);
  if(tinySum>0)
    bar+='<span style="width:'+Math.max(tinySum/total*100,0.6).toFixed(2)
      +'%;background:'+SEG_COLORS.other+'" title="'
      +esc('other ('+fmtMib(tinySum)+'): '
        +tiny.map(s=>s.label+' '+fmtMib(s.mib)).join(' · '))+'"></span>';
  bar+='</div>';
  let leg='<div class="seglegend">';
  for(const sg of main)
    leg+='<span class="dot" style="background:'+(SEG_COLORS[sg.id]||'#30363d')
      +'" title="'+esc(sg.detail||'')+'"></span>'
      +esc(sg.label)+' '+fmtMib(sg.mib)
      +(sg.replicated?('<span class="repltag" title="'
        +esc(sg.replication_reason||'')+'">replicated x'
        +sg.replication_factor+'</span>'):'')
      +' &nbsp;';
  if(tinySum>0)
    leg+='<span class="dot" style="background:'+SEG_COLORS.other+'"></span>'
      +'other '+fmtMib(tinySum)+' &nbsp;';
  leg+='<span class="dot" style="background:#0d1117;border:1px solid #30363d"></span>'
    +'free '+fmtMib(free)+' &nbsp;·&nbsp; used <b>'+fmtMib(c.total_mib)+'</b></div>';
  return bar+leg;
}
// The CUDA-graph line under a placement: the placement's own graph_mem block
// (estimate or anchor, with provenance + error band), plus -- landing only --
// the LIVE measured capture totals (opts.graphCapture from the boot-log
// parse / server_info; honest n/a when neither exists).
function graphMemHtml(pl, opts){
  let h='';
  const gm=pl.graph_mem;
  if(gm){
    const items=(gm.items||[]).map(i=>i.label+' '+fmtMib(i.mib)).join(' · ');
    h+='<div class="graphline" title="'+esc((gm.formula||'')+(items?(' — '+items):''))+'">'
      +'CUDA graphs: <b>'+fmtMib(gm.per_rank_mib)+'</b>/rank '
      +'<span class="muted">('+esc(gm.provenance||'heuristic')
      +(gm.provenance==='heuristic'?(', estimate &plusmn;'+(gm.error_band_pct||30)+'%'):'')
      +')</span>'
      +(items?(' <span class="muted">— '+esc(items)+'</span>'):'')
      +'</div>';
  }
  const gc=opts && opts.graphCapture;
  if(gc){
    if(gc.source==='boot-log' && gc.summary){
      const s=gc.summary;
      const per=Object.entries(s.per_rank_mib||{}).map(([r,v])=>'TP'+r+' '+fmtMib(v)).join(' · ');
      h+='<div class="graphline" title="'
        +esc((s.items||[]).map(i=>i.label+' '+fmtMib(i.total_mib)).join(' · '))+'">'
        +'CUDA graphs (LIVE, measured): <b>'+fmtMib(s.total_mib)+'</b> total'
        +(per?(' <span class="muted">('+per+')</span>'):'')
        +' <span class="muted">— boot-log capture lines ('+s.n_captures+' graphs)</span></div>';
    } else if(gc.source==='server_info'){
      h+='<div class="graphline">CUDA graphs (LIVE, measured): <b>'
        +fmtMib(gc.total_mib)+'</b> <span class="muted">— '+esc(gc.note||'server_info')+'</span></div>';
    } else {
      h+='<div class="graphline muted">CUDA graphs (live): '
        +esc(gc.reason||'n/a')+'</div>';
    }
  }
  return h;
}
function renderPlacement(pl, liveGpus, opts){
  opts=opts||{};
  if(!pl) return '<span class="muted">no placement.</span>';
  const m=pl.model||{}; let h='';
  h+='<div class="legend" style="margin-top:0">TP '+pl.tp_size+' · DCP '+pl.dcp_size
    +' · heads Q'+m.num_attention_heads+'/KV'+m.num_key_value_heads
    +' · layers '+m.num_hidden_layers
    +(m.num_experts?(' · experts '+m.num_experts):'')+'</div>';
  // Explicit KV-replication banner (fork uneven-DCP or stock replication):
  // the hatched segment marks it in the bar; this names the WHY once.
  if(pl.kv_replication && pl.kv_replication.replicated)
    h+='<div class="graphline" title="'+esc(pl.kv_replication.reason||'')+'">'
      +'KV heads <span class="repltag">replicated x'+pl.kv_replication.factor+'</span> '
      +'<span class="muted">('+esc(pl.kv_replication.source||'')
      +(pl.kv_replication.cache_duplicated?', KV cache duplicated per rank':', cache token-sharded, not duplicated')
      +')</span></div>';
  const byRank={};
  const put=(list,f)=>{ for(const x of (list||[])) (byRank[x.rank]=byRank[x.rank]||{})[f]=x; };
  put(pl.attn_heads,'attn'); put(pl.gdn_heads,'gdn');
  put(pl.kv_tokens,'kv'); put(pl.experts,'ex');
  const offByRank={};
  if(pl.offload && pl.offload.per_rank)
    for(const r of pl.offload.per_rank) offByRank[r.rank]=r;
  for(const c of (pl.cards||[])){
    const total=c.card_total_mib || Math.max(c.total_mib, c.budget_mib||0) || 1;
    const free=Math.max(0, total-c.total_mib);
    const over=c.physical_overcommit
      ? ' <b class="nofitc">EXCEEDS CARD (physically impossible)</b>' : '';
    // c.gpu_index is CUDA-space (== the rank_gpu_id space); label the nvml
    // index too via the bridge map so the number is never misread as the
    // nvidia-smi index.
    h+='<div class="cardblock">'
      +'<div><b>'+(devLabel(c.gpu_index, null)||('GPU'+c.gpu_index))+'</b> '+esc(c.card_name||'')
      +' <span class="muted">'
      +(c.card_total_mib?fmtMib(c.card_total_mib)+' total · ':'')
      +'ranks ['+c.ranks.join(',')+']'
      +(c.budget_mib!=null?' · budget '+fmtMib(c.budget_mib):'')
      +'</span>'+over+'</div>'
      +segBarHtml(c, total, free);
    const lg=_liveGpuForCard(liveGpus, c.gpu_index);
    if(lg) h+=cardLiveHtml(lg);
    for(const r of c.ranks){
      const d=byRank[r]||{}; const bits=[];
      if(d.attn) bits.push('Q['+d.attn.q_head_start+'..'+d.attn.q_head_end+') '
        +'K['+d.attn.k_head_start+'..'+d.attn.k_head_end+')'+(d.attn.kv_replicated?' repl':'')
        +' V['+d.attn.v_head_start+'..'+d.attn.v_head_end+') '
        +fmtMib(d.attn.q_mib+d.attn.k_mib+d.attn.v_mib+d.attn.o_mib));
      if(d.gdn) bits.push('GDN K['+d.gdn.k_head_start+'..'+d.gdn.k_head_end
        +') V['+d.gdn.v_head_start+'..'+d.gdn.v_head_end+')');
      // Cyclic owner rule: positions [lo..hi) of EVERY cycle_len-token block
      // (interleaved across the whole context), never one contiguous range.
      if(d.kv) bits.push('KV pos ['+d.kv.cycle_start+'..'+d.kv.cycle_end+') of every '
        +d.kv.cycle_len+'-tok block, '+d.kv.tokens_owned
        +(d.kv.kv_token_capacity!=null?(' (cap '+Math.round(d.kv.kv_token_capacity)+')'):''));
      if(d.ex) bits.push('experts ['+d.ex.expert_start+'..'+d.ex.expert_end+') '+d.ex.num_experts);
      if(pl.mtp && pl.mtp.present && pl.mtp.per_rank_mib && pl.mtp.per_rank_mib[r]!=null)
        bits.push('MTP '+fmtMib(pl.mtp.per_rank_mib[r]));
      const off=offByRank[r];
      if(off && off.host_expert_start!=null)
        bits.push('host-RAM experts ['+off.host_expert_start+'..'+off.host_expert_end
          +') '+fmtMib(off.host_mib));
      h+='<div class="rankline">rank '+r+': '+(bits.join(' · ')||'(no detail)')+'</div>';
    }
    h+='</div>';
  }
  h+=graphMemHtml(pl, opts);
  h+='<div class="legend">KV token vector ['+pl.token_vector.join(',')
    +'] ('+esc(pl.token_vector_source)+')'
    +(pl.mtp&&pl.mtp.present?(' · MTP layers ['+pl.mtp.layer_start+'..'+pl.mtp.layer_end+')'):'')+'</div>';
  if(pl.offload){
    const o=pl.offload;
    h+='<div class="legend"><b>host-RAM offload rule</b> ('+esc(o.offloadable_class)
      +'): offloadable '+fmtMib(o.total_offloadable_mib)
      +', host-held '+fmtMib(o.total_host_mib)
      +(o.resident_fraction!=null?(', resident frac '+o.resident_fraction.toFixed(2)):'')
      +' &mdash; '+esc(o.note)+'</div>';
  }
  if(pl.notes && pl.notes.length)
    h+='<div class="legend">'+pl.notes.map(n=>'&bull; '+esc(n)).join('<br>')+'</div>';
  return h;
}

// ===========================================================================
// Landing page: live monitor of the managed server (client 60s ring buffers).
// ===========================================================================
window._landTimer=null; window._ring={};
const RING_SECONDS=60;
//: settable resolution hook: the landing poll cadence IS the strip sample
//: resolution (one sample per poll; one pixel bucket per sample below).
const LAND_POLL_MS=2000;
function pushRing(key,t,v){
  if(v==null||isNaN(v)) return;
  const a=window._ring[key]||(window._ring[key]=[]);
  a.push({t,v});
  const cutoff=t-RING_SECONDS;
  while(a.length && a[0].t<cutoff) a.shift();
}
// -- top-strip ring semantics: FREEZE at zero -------------------------------
// A strip graph shows the last 60s of ACTIVITY. When a stream goes idle the
// curve drops to 0 ONCE (a zero sample is accepted only within the grace
// window after the last nonzero sample) and then the buffer FREEZES -- no
// eternally scrolling flatline of zeros. New activity resumes appending; the
// wall-time gap is allowed (pushRing's cutoff then drops pre-gap history).
const STRIP_FREEZE_GRACE_S=2;  // grace after the last nonzero sample
window._stripActive={};        // ring key -> t of the last NONZERO sample
function stripPush(key,t,v){
  if(v==null||isNaN(v)) return;
  if(v>0) window._stripActive[key]=t;
  else if(window._stripActive[key]==null
          || (t-window._stripActive[key])>STRIP_FREEZE_GRACE_S)
    return;  // FROZEN: the drop-to-zero was already recorded (or never active)
  pushRing(key,t,v);
}
// -- top-strip sparkline: one pixel bucket per sample, NO smoothing ---------
// Sample i occupies one fixed STRIP_PX_PER_SAMPLE-wide x bucket (newest at
// the right edge); the SVG is sized samples*px so nothing is interpolated,
// stretched, or aliased -- the graph never invents data between polls.
const STRIP_PX_PER_SAMPLE=5;
const STRIP_SAMPLES=Math.round(RING_SECONDS*1000/LAND_POLL_MS);
const STRIP_W=STRIP_SAMPLES*STRIP_PX_PER_SAMPLE;
const STRIP_H=26;
// -- column rendering -------------------------------------------------------
// One filled column per sample, drawn from the LEFT and growing rightwards as
// samples arrive, each column anchored on the baseline. A polyline was the
// wrong mark here twice over: with two samples it drew a single stroke across
// half an empty field, which reads as data that is not there, and it
// interpolated between samples taken seconds apart. A column stands for
// exactly one sample and for nothing in between.
function _columns(a, w, h, max){
  if(!a || !a.length) return '';
  const n=Math.max(1, Math.floor(w/STRIP_PX_PER_SAMPLE));
  const bw=Math.max(1, STRIP_PX_PER_SAMPLE-1);
  let mx=max;
  if(mx==null){ mx=-Infinity; for(const p of a) if(p.v>mx) mx=p.v; }
  if(!(mx>0)) mx=1;
  let out='';
  const shown=a.slice(-n);
  for(let i=0;i<shown.length;i++){
    const v=Math.max(0, shown[i].v);
    const bh=Math.max(1, Math.round((h-2)*Math.min(1, v/mx)));
    out+='<rect x="'+(i*STRIP_PX_PER_SAMPLE)+'" y="'+(h-bh)+'" width="'+bw
      +'" height="'+bh+'" fill="var(--info)"/>';
  }
  return out;
}
function stripSpark(key){
  const a=(window._ring[key]||[]).slice(-STRIP_SAMPLES);
  return '<svg class="spk" width="'+STRIP_W+'" height="'+STRIP_H+'">'
    +_columns(a, STRIP_W, STRIP_H)+'</svg>';
}
function sparkline(key,w,hh){
  const a=window._ring[key]||[]; if(!a.length) return '';
  return '<svg width="'+w+'" height="'+hh+'" style="vertical-align:middle;'
    +'background:var(--bg-input);border-radius:2px">'
    +_columns(a, w, hh)+'</svg>';
}
function landingEndpoint(){
  try{ return (localStorage.getItem('land_endpoint')||'').trim(); }catch(e){ return ''; }
}
function resetLandingRings(){
  window._ring={}; window._stripActive={}; window._stripHit=null;
}
function setLandingEndpoint(){
  try{ localStorage.setItem('land_endpoint', $('land_endpoint').value.trim()); }catch(e){}
  resetLandingRings(); landingPoll();
}
function clearLandingEndpoint(){
  try{ localStorage.removeItem('land_endpoint'); }catch(e){}
  $('land_endpoint').value=''; resetLandingRings(); landingPoll();
}
// 'detect' with something typed in the endpoint box VERIFIES that one target;
// with the box empty it sweeps the local sglang port range. Same endpoint,
// same button -- the input decides which of the two the user asked for.
async function detectLandingEndpoint(){
  const typed=($('land_endpoint').value||'').trim();
  $('land_target_note').textContent = typed
    ? ('probing '+typed+'…') : 'sweeping the local sglang ports…';
  try{
    const r=await fetch('/api/detect_endpoint',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify(typed?{endpoint:typed}:{})});
    const d=await r.json();
    if(d.endpoint){
      $('land_endpoint').value=d.endpoint.replace(/^https?:\/\//,'');
      setLandingEndpoint();
      if(d.metrics===false)
        $('land_target_note').innerHTML='<span class="est">found '
          +esc(d.endpoint)+', but it serves no /metrics &mdash; started '
          +'without --enable-metrics, so live rates stay unavailable.</span>';
    } else if(d.explicit){
      $('land_target_note').textContent=d.error||('no sglang server at '+typed);
    } else {
      const pr=d.probed||[];
      const span=pr.length>4?(pr[0]+'-'+pr[pr.length-1]+' ('+pr.length+' ports)')
                            :pr.join(', ');
      $('land_target_note').textContent='nothing reachable on '+span;
    }
  }catch(e){ $('land_target_note').textContent=''+e; }
}
async function startLanding(){
  if(window._landTimer) return;
  if(!$('land_endpoint').value) $('land_endpoint').value=landingEndpoint();
  await landingPoll();
  window._landTimer=setInterval(landingPoll, LAND_POLL_MS);
}
function stopLanding(){ if(window._landTimer){ clearInterval(window._landTimer); window._landTimer=null; } }
async function landingPoll(){
  let d;
  const ep=landingEndpoint();
  const live=$('landing_live');
  try{
    // Keyed on 'landing': a poll that is still outstanding when the next
    // tick fires is aborted, so two snapshots can never race to paint the
    // same panels. The timeout is just under the poll period for the same
    // reason. While it is out, the panel dims rather than blanks.
    stale(live,true);
    d=await api('/api/live_snapshot'+(ep?('?endpoint='+encodeURIComponent(ep)):''),
                {key:'landing', timeout:LAND_POLL_MS-200});
  }catch(e){
    if(!apiAborted(e)) $('land_target_note').textContent='snapshot unavailable: '+apiError(e);
    return;
  } finally { stale(live,false); }
  window._lastTarget=d && d.target;
  if(d && d.target){
    $('land_target_note').innerHTML='monitoring <b>'+esc(d.target.endpoint)+'</b> ('
      +(d.target.managed?'managed instance':esc(d.target.kind)+' / external')+')';
  } else {
    $('land_target_note').textContent='no target';
  }
  if(!d || !d.running || !d.snapshot){
    // State 1: nothing is running. Rendered from THIS tick, so a warning
    // about a server that has since gone away cannot survive it.
    renderLivePanel(null, d);
    $('landing_none').style.display=''; $('landing_live').style.display='none';
    const st=d && d.status;
    $('landing_none_status').textContent = (d && d.error)? ' ('+d.error+')'
      : (st && st.state && st.state!=='stopped'? ' (managed server state: '+st.state+')':'');
    return;
  }
  $('landing_none').style.display='none'; $('landing_live').style.display='';
  // LIVE CUDA-graph capture info (boot-log parse / server_info; may be an
  // honest n/a) -- consumed by renderLandingPlacement below.
  window._lastGraphCapture = d.graph_capture || null;
  renderLanding(d.snapshot, d.target);
}
// Normalize the start config from EITHER source: the managed LaunchSettings
// (exact dashboard flags) or, for an external/hand-started server, its own
// /get_server_info + /get_model_info view.
function normalizeStartConfig(s){
  const cfg=s.launch_config;
  if(cfg && cfg.model_path)
    return {src:'managed launch settings', cfg:cfg, argv:cfg.launch_argv||null,
            env:cfg.extra_env||null, raw:null};
  const w=s.server_info||{};
  const si=w.server_info||{};
  const mi=w.model_info||{};
  if(!Object.keys(si).length && !Object.keys(mi).length) return null;
  return {src:'/get_server_info (external server)', raw:s.server_info, argv:null, env:null, cfg:{
    model_path: si.model_path||mi.model_path,
    served_model_name: si.served_model_name,
    tp_size: si.tp_size, rank_gpu_id: si.rank_gpu_id,
    rank_tp_ratio: si.rank_tp_ratio, rank_gpu_memory_mib: si.rank_gpu_memory_mib,
    kv_cache_dtype: si.kv_cache_dtype, context_length: si.context_length,
    max_num_seqs: si.max_running_requests,
    spec_mode: si.speculative_algorithm||null,
    speculative_num_steps: si.speculative_num_steps,
    speculative_num_draft_tokens: si.speculative_num_draft_tokens,
    speculative_eagle_topk: si.speculative_eagle_topk,
    speculative_adaptive: si.speculative_adaptive,
    tool_call_parser: si.tool_call_parser, reasoning_parser: si.reasoning_parser,
    chat_template: si.chat_template, port: si.port, gguf_variant: null,
  }};
}
// Compact, grouped start-config summary: what is running + the key flags,
// with the full argv/env and raw server_info COLLAPSED (no wall of text).
function renderStartConfig(s, tgt){
  const n=normalizeStartConfig(s);
  if(!n) return '<span class="muted">no start config available (neither launch settings nor /get_server_info).</span>';
  const c=n.cfg;
  const pill=(k,v)=>{
    if(v==null||v===''||v===false) return '';
    return '<span class="pill"><span class="muted">'+esc(k)+'</span> '
      +esc(String(Array.isArray(v)?v.join(','):v))+'</span> ';
  };
  const modelName=(c.served_model_name&&c.served_model_name!=='model')
    ? c.served_model_name : String(c.model_path||'').split('/').pop();
  let h='<div style="font-size:1rem;margin-bottom:.35rem"><b>'+esc(modelName||'unknown model')+'</b>'
    +(tgt?' <span class="muted">@ '+esc(tgt.endpoint)+' ('+(tgt.managed?'managed':'external')+')</span>':'')
    +'</div>';
  const groups=[
    ['model', pill('path',c.model_path)+pill('gguf',c.gguf_variant)+pill('served-as',c.served_model_name)],
    ['parallelism', pill('tp',c.tp_size)+pill('rank-gpu-id',c.rank_gpu_id)
      +pill('rank-tp-ratio',c.rank_tp_ratio)+pill('rank-mem-mib',c.rank_gpu_memory_mib)],
    ['memory / KV', pill('kv-dtype',c.kv_cache_dtype)+pill('context',c.context_length)+pill('max-seqs',c.max_num_seqs)],
    ['speculative', (c.spec_mode&&c.spec_mode!=='off')
      ? pill('algo',c.spec_mode)+pill('steps',c.speculative_num_steps)
        +pill('draft-tokens',c.speculative_num_draft_tokens)+pill('topk',c.speculative_eagle_topk)
      : '<span class="pill muted">off</span>'],
    ['serving', pill('port',c.port)+pill('tool-parser',c.tool_call_parser)
      +pill('reasoning-parser',c.reasoning_parser)
      +pill('chat-template', c.chat_template?String(c.chat_template).split('/').pop():null)],
  ];
  for(const [name,row] of groups)
    if(row.trim()) h+='<div class="cfgrow"><span class="cfgk">'+esc(name)+'</span>'+row+'</div>';
  h+='<div class="legend" style="margin-top:.25rem">source: '+esc(n.src)+'</div>';
  // data-key gives the patcher a stable identity for these two collapses, so
  // they are re-matched across polls even when the groups above them change
  // shape. Without it they would fall back to positional matching, which is
  // correct but fragile once a group appears or disappears mid-run.
  if(n.argv||n.env){
    h+='<details data-key="cfg_launch" style="margin-top:.3rem"><summary class="muted" style="cursor:pointer">full launch command + env</summary><pre>';
    if(n.env) for(const k of Object.keys(n.env)) h+=esc(k+'='+n.env[k])+'\n';
    if(n.argv) h+=esc(n.argv.join(' '));
    h+='</pre></details>';
  }
  if(n.raw)
    h+='<details data-key="cfg_raw" style="margin-top:.3rem"><summary class="muted" style="cursor:pointer">raw server_info</summary>'
      +'<pre data-key="cfg_raw_pre" style="max-height:260px;overflow:auto">'+esc(JSON.stringify(n.raw,null,1))+'</pre></details>';
  return h;
}
function renderLanding(s, tgt){
  window._lastSnapshot=s;
  renderLivePanel(s);
  // Patched, not replaced: this block carries two <details> ("full launch
  // command + env", "raw server_info") and a scrollable <pre>. Rewriting it
  // wholesale on every 2 s tick is what used to slam them shut.
  setHTML($('landing_config'), renderStartConfig(s, tgt)
    +(s.metrics_error?noMetricsBanner(s.metrics_error, s):''));

  // Per-card 60s telemetry rings (rendered INSIDE the placement card blocks,
  // renderPlacement's live line -- the old standalone per-GPU chart grid is
  // gone). util is an ACTIVITY stream -> strip freeze-at-zero semantics;
  // power idles at a nonzero wattage -> plain ring.
  for(const g of (s.gpus||[])){
    const key='gpu'+g.nvml_index;
    stripPush(key+'_util',s.t,g.utilization_pct);
    pushRing(key+'_pow',s.t,g.power_watts);
  }

  if(s.rates) window._lastRates=s.rates;
  renderLandingStrip(s);

  renderLandingPlacement(s);
}
// ===========================================================================
// Landing TOP STRIP: full-width headline tiles + 60s freeze-at-zero activity
// graphs. Data: /api/live_snapshot only, plus the #147 /api/hicache_saved
// accumulator (throttled) for the SAVED tile. The former lower
// "throughput / spec / cache" block moved INTO this strip -- nothing is
// rendered twice on the landing page.
// ===========================================================================
const STRIP_DASH='<span class="muted">--</span>';
function stripTile(id,label,value,sub,sparkKey,tip){
  return '<div class="mtile" id="'+id+'"'+(tip?' title="'+esc(tip)+'"':'')+'>'
    +'<div class="mt-l">'+label+'</div>'
    +'<div class="mt-v">'+value+'</div>'
    +'<div class="mt-s">'+(sub||'&nbsp;')+'</div>'
    +(sparkKey?stripSpark(sparkKey)
      :'<svg class="spk" width="'+STRIP_W+'" height="'+STRIP_H+'"></svg>')
    +'</div>';
}
function stripFetchSaved(){  // throttled read of the #147 savings accumulator
  const now=Date.now();
  if(window._stripSavedT && now-window._stripSavedT<10000) return;
  window._stripSavedT=now;
  fetch('/api/hicache_saved?price_ct_per_kwh='+getKwhPriceCt())
    .then(r=>r.json()).then(d=>{ window._stripSaved=d; }).catch(e=>{});
}
function renderLandingStrip(s){
  const t=s.t, EPS=0.05;   // tok/s below EPS counts as "phase not active"
  const noMetrics=!!s.metrics_error;
  const hint='token metrics unavailable &mdash; needs --enable-metrics on the server';
  const rates=s.rates||null, spec=s.spec||null;
  const dec=rates?rates.decode_tok_s:null, pfx=rates?rates.prefill_tok_s:null;
  const num=(v,dgt)=>(v==null?STRIP_DASH:Number(v).toFixed(dgt==null?1:dgt));
  const sub=(txt)=>noMetrics?('<span class="est">'+hint+'</span>'):txt;
  // throughput streams: freeze-at-zero ring pushes (drop to 0 once, then hold)
  stripPush('st_dec',t,dec); stripPush('st_pfx',t,pfx);
  // live board power: per-card NVML watts summed (works without /metrics)
  let watts=null;
  for(const g of (s.gpus||[])) if(g.power_watts!=null) watts=(watts||0)+g.power_watts;
  // spec accept is a GAUGE (holds its last value while idle) -- sample it only
  // while decode is active so the graph freezes instead of scrolling a stale
  // flat line; no fake zero is invented for a gauge.
  if(spec&&spec.accept_rate!=null&&dec!=null&&dec>EPS)
    stripPush('st_acc',t,spec.accept_rate*100);
  // per-tier cache hit: fresh only within a prompt window; keep the last known
  // numbers on display, but push samples only when a fresh window exists.
  if(s.cache_hit_rates) window._stripHit=s.cache_hit_rates;
  const hit=window._stripHit||null;
  if(s.cache_hit_rates&&s.cache_hit_rates.overall!=null)
    stripPush('st_hit',t,s.cache_hit_rates.overall*100);
  let hitBig=STRIP_DASH, tierTxt='no prompt window yet';
  if(hit){
    hitBig=((hit.overall||0)*100).toFixed(0)+'<small>%</small>';
    tierTxt=Object.keys(hit).filter(k=>k!=='overall')
      .map(k=>esc(k)+' <b>'+((hit[k]||0)*100).toFixed(0)+'%</b>').join(' &middot; ')
      ||'no per-tier counters';
  }
  if(s.hicache)
    tierTxt+=' &middot; host pool '+((s.hicache.host_used_frac||0)*100).toFixed(0)+'% used';
  // live energy: J/token = summed NVML watts / active-phase tok/s. The phase
  // is whichever rate is active (decode wins when both run, chunked prefill).
  // Undefined while idle -> '--' and a FROZEN graph (no divide-by-zero junk).
  let phase=null, tokS=null;
  if(dec!=null&&dec>EPS){phase='decode';tokS=dec;}
  else if(pfx!=null&&pfx>EPS){phase='prefill';tokS=pfx;}
  const jtok=(watts!=null&&tokS)?watts/tokS:null;
  const tokKwh=jtok?3.6e6/jtok:null;
  if(jtok!=null) stripPush('st_j',t,jtok);
  // live cost per phase (ct / 1M tok at the shared kWh price)
  const priceCt=getKwhPriceCt();
  const decCt=(watts!=null&&dec!=null&&dec>EPS)?jPerTokToCtPer1M(watts/dec,priceCt):null;
  const pfxCt=(watts!=null&&pfx!=null&&pfx>EPS)?jPerTokToCtPer1M(watts/pfx,priceCt):null;
  if(decCt!=null) stripPush('st_cost',t,decCt);
  // SAVED tile: #147 accumulator = host-RAM + disk tiers ONLY (the pipeline
  // already excludes the device/VRAM hot tier). Activity graph = tok/s served
  // from the non-device tiers (per-tier hit fraction x gross prompt rate).
  stripFetchSaved();
  if(s.cache_hit_rates&&rates&&rates.prefill_tok_s_gross!=null){
    let f=0;
    for(const k of Object.keys(s.cache_hit_rates))
      if(k!=='overall'&&k!=='device'&&s.cache_hit_rates[k]!=null)
        f+=s.cache_hit_rates[k];
    stripPush('st_saved',t,f*rates.prefill_tok_s_gross);
  }
  const sd=window._stripSaved||null;
  const sgt=(sd&&sd.grand_total)||null;
  let savedVal=STRIP_DASH, savedSub='no savings recorded yet (#147 accumulator)';
  if(sgt&&sgt.saved_kwh_band){
    const cb=kwhBandToCt(sgt.saved_kwh_band,priceCt);
    savedVal=_f2(cb[0])+'&ndash;'+_f2(cb[1])+'<small> ct</small>';
    savedSub=_f1(sgt.recovered_prefill_tokens)
      +' tok recovered &middot; TOTAL across sessions';
  }
  $('landing_strip').innerHTML=
    stripTile('strip_decode','decode tok/s',
      noMetrics?STRIP_DASH:num(dec)+'<small> tok/s</small>',
      sub(rates?('server gauge '+num(rates.gen_throughput_server)+' tok/s')
               :'(rates on next tick)'),
      'st_dec')
   +stripTile('strip_prefill','prefill tok/s (non-cached)',
      noMetrics?STRIP_DASH:num(pfx)+'<small> tok/s</small>',
      sub(rates?('gross incl. cache-served: '+num(rates.prefill_tok_s_gross)+' tok/s')
               :'(rates on next tick)'),
      'st_pfx',
      'REAL computed prefill only: cache-served tokens are subtracted. '
      +'The gross figure incl. cached tokens is the small line, never the headline.')
   +stripTile('strip_spec','MTP accept',
      (noMetrics||!spec)?STRIP_DASH:((spec.accept_rate||0)*100).toFixed(1)+'<small>%</small>',
      sub(spec?('adaptive-k <b>'+(spec.adaptive_k!=null?spec.adaptive_k:'--')
        +'</b> &middot; ema accept-len '
        +(spec.ema_accept_len!=null?Number(spec.ema_accept_len).toFixed(2):'--'))
        :'spec decode off / no data'),
      'st_acc')
   +stripTile('strip_cache','cache hit per tier',
      noMetrics?STRIP_DASH:hitBig, sub(tierTxt), 'st_hit',
      'Per-tier hit rate over the last prompt window: device = VRAM hot tier, '
      +'host = system RAM, storage = disk. Headline = overall across tiers.')
   +stripTile('strip_energy','energy now',
      noMetrics?STRIP_DASH
        :(jtok!=null?num(jtok)+'<small> J/tok ('+phase+')</small>':STRIP_DASH),
      sub((tokKwh!=null?_f1(tokKwh):'--')+' tok/kWh &middot; '
        +(watts!=null?Math.round(watts)+' W (NVML sum)':'-- W')),
      'st_j')
   +stripTile('strip_cost','cost now (ct/1M tok)',
      noMetrics?STRIP_DASH
        :(decCt!=null?num(decCt,2)+'<small> ct/1M decode</small>':STRIP_DASH),
      sub('prefill '+(pfxCt!=null?Number(pfxCt).toFixed(2):'--')
        +' ct/1M &middot; @'+priceCt+' ct/kWh'),
      'st_cost')
   +stripTile('strip_saved','saved by RAM+disk cache',
      savedVal, savedSub, 'st_saved',
      'Avoided prefill recompute cost from the HiCache host-RAM + disk tiers '
      +'ONLY -- the device/VRAM hot tier is explicitly excluded (#147). '
      +'Cumulative total across sessions at the current kWh price.');
}
async function renderLandingPlacement(s){
  const n=normalizeStartConfig(s);
  const cfg=(n&&n.cfg)||{}; const model=cfg.model_path;
  if(!model){
    // No placement computable (external server without /get_server_info):
    // still show the per-card LIVE telemetry blocks -- the merged card view
    // degrades to telemetry-only, never to nothing.
    let gh='';
    for(const g of (s.gpus||[]))
      gh+='<div class="cardblock"><div><b>'
        +(devLabel(g.cuda_index, g.nvml_index)||('#'+g.nvml_index))+'</b> '
        +esc(g.name)+'</div>'+cardLiveHtml(g)+'</div>';
    setHTML($('landing_placement'),
      '<span class="muted">no start config for placement (external server without /get_server_info).</span>'+gh);
    return;
  }
  const flags={
    tp_size: cfg.tp_size||1, rank_gpu_id: cfg.rank_gpu_id||null, rank_tp_ratio: cfg.rank_tp_ratio||null,
    rank_gpu_memory_mib: cfg.rank_gpu_memory_mib||null, kv_cache_dtype: cfg.kv_cache_dtype||'auto',
    context_length: cfg.context_length||null, max_running_requests: cfg.max_num_seqs||null,
    speculative_algorithm: (cfg.spec_mode&&cfg.spec_mode!=='off')?cfg.spec_mode:null,
    speculative_num_steps: cfg.speculative_num_steps||null,
    speculative_num_draft_tokens: cfg.speculative_num_draft_tokens||null,
    speculative_adaptive: cfg.spec_mode==='adaptive' || !!cfg.speculative_adaptive,
  };
  // LIVE measured CUDA-graph memory (boot-log capture parse) overrides the
  // planner estimate in the placement's graph segment -- measured wins.
  const gcap=window._lastGraphCapture;
  if(gcap && gcap.source==='boot-log' && gcap.summary && gcap.summary.per_rank_mib){
    const s=gcap.summary;
    flags.graph_mem_override={provenance:'measured (live boot log)',
      per_rank_map:s.per_rank_mib,
      per_rank_mib:Math.max.apply(null,Object.values(s.per_rank_mib)),
      items:(s.items||[]).map(i=>({label:i.label,mib:i.total_mib,
        formula:'measured capture line (live boot log)'})),
      error_band_pct:0,
      formula:'measured capture lines from the running server boot log'};
  }
  // Card inventory keyed in CUDA space: rank_gpu_id (and the default
  // rank i -> gpu i identity when it is unset, e.g. an external server
  // without the flag) is CUDA-order, while s.gpus is NVML-sampled. Keying
  // by nvml_index attributed rank 0 (= cuda:0, the 5090 on this rig) to
  // the 3080 sitting at nvml:0. cuda_index comes from the snapshot's
  // device_map bridge; an unbridged card falls back to nvml_index (labeled
  // ambiguity beats a dropped card).
  noteCudaMap(s.gpus, (s.gpus||[]).some(g=>g.cuda_index_source==='heuristic')?'heuristic':null);
  const ct={},cn={};
  (s.gpus||[]).forEach(g=>{ const k=(g.cuda_index!=null?g.cuda_index:g.nvml_index);
    ct[k]=g.mem_total_mib; cn[k]=g.name; });
  flags.card_total_mib=ct; flags.card_name=cn;
  try{
    const d=await api('/api/placement',{key:'placement_landing',
      body:{model, gguf_choice: cfg.gguf_variant||null, flags}});
    setHTML($('landing_placement'), d.ok
      ? renderPlacement(d.placement, s.gpus, {graphCapture: window._lastGraphCapture})
      : '<span class="reasons">'+esc(d.error)+'</span>'
        +((s.gpus&&s.gpus.length)?'':(s.nvml_error?'<div class="muted">no NVML cards ('+esc(s.nvml_error)+')</div>':'')));
  }catch(e){
    // A superseded call is not a failure; the panel keeps the newer answer.
    if(apiAborted(e)) return;
    setHTML($('landing_placement'),'<span class="reasons">'+esc(apiError(e))+'</span>');
  }
}

// ===========================================================================
// Runner tab: full flag surface (resolve greying/auto-set) + config profiles +
// live prospective placement (shared renderer).
// ===========================================================================
window._flagCat=null; window._flagSettings={}; window._profiles=[];
window._flagSection={};   // catalog id -> section key (built at render time)
async function loadFlagCatalog(){
  try{
    const d=await api('/api/flag_catalog',{key:'flag_catalog'});
    if(!d.ok) return;
    window._flagCat=d;
    // The trade-off registry rides along with the catalog: one fetch, one
    // source. Controls that are not flags (the view switch, the objective
    // templates, the budget sliders) look themselves up in it by key.
    window._tips=d.tooltips||{};
    $('flag_counts').textContent=d.upstream_count+' upstream + '+d.fork_count+' fork';
    renderFlagSurface();
    applyStaticTips();
  }catch(e){}
}
// ---- what it gives / what it costs, for the controls that are not flags ---
// The text lives server-side in planner/tooltips.py, keyed the same way. This
// only puts it on the elements; it never composes a sentence of its own, so
// there is exactly one place to edit when a control's cost changes.
const STATIC_TIPS={
  vm_simple:'view_mode.simple', vm_expert:'view_mode.expert',
  tune_both:'tune.both', tune_maxkv:'tune.maxkv',
  tune_dec:'tune.dec', tune_enc:'tune.enc',
  sv_ctx:'context_length', sv_ctx_slider:'context_length',
  row_sv_ctx:'context_length',
  max_running_requests:'max_running_requests', mrr_slider:'max_running_requests',
  row_mrr:'max_running_requests',
};
function tip(key){ return (window._tips||{})[key]||''; }
function applyStaticTips(){
  for(const id of Object.keys(STATIC_TIPS)){
    const el=$(id); if(!el) continue;
    const t=tip(STATIC_TIPS[id]);
    if(t) el.title=t;
  }
}
function _surfaceSpecs(g){
  // The serving-identity ids are owned by the MODEL/SERVING sections above
  // the surface -- hiding them here keeps every fact in exactly one place.
  return window._flagCat.groups[g].filter(f=>!SERVING_OWNED[f.id]);
}
// ---- LM-Studio-style section map: the catalog stays the single source of
// truth; these are VIEWS onto it. Every flag lands in EXACTLY ONE section
// (first matching rule wins); everything unmatched goes behind the ONE
// "Show advanced settings" toggle.
const SECTION_KEYS=['context','gpu','speculative','cache','serving','advanced'];
const SEC_CACHE_IDS={disable_radix_cache:1, radix_cache_backend:1,
  radix_eviction_policy:1, enable_session_radix_cache:1,
  enable_hierarchical_cache:1, enable_lmcache:1, lmcache_config_file:1,
  enable_flexkv:1, flexkv_config_file:1, page_size:1,
  enable_page_major_kv_layout:1, disable_chunked_prefix_cache:1,
  hibernate_dir:1};
const SEC_SERVING_IDS={chat_template:1, hf_chat_template_name:1,
  completion_template:1, tool_call_parser:1, reasoning_parser:1};
const SEC_GPU_IDS={mem_fraction_static:1, cpu_offload_gb:1};
const SEC_GPU_GROUPS={'uneven-tp':1, parallelism:1, device:1, offload:1};
function flagSection(f){
  if(f.id==='kv_cache_dtype') return 'context';
  if(SEC_GPU_GROUPS[f.group] || SEC_GPU_IDS[f.id]) return 'gpu';
  if(f.group==='speculative' || f.id.indexOf('speculative_')===0) return 'speculative';
  if(SEC_CACHE_IDS[f.id] || f.id.indexOf('hicache_')===0) return 'cache';
  if(SEC_SERVING_IDS[f.id]) return 'serving';
  return 'advanced';
}
// One labeled row per flag: label left, LM-Studio-like control right (toggle
// switch for bool / compact dropdown for enum / numeric or text input), a
// pure-CSS "?" hover at the row end. Ids flrow_/fl_/flq_/flh_ are unchanged
// so resolve()-driven greying, auto-set notes and profiles keep working.
// The row markup describes the CURRENT value, not an empty control.
// _flagSettings is the single source of truth for what a knob is set to, and
// the row now says so out loud. That matters twice over: the patcher treats
// markup as the description of state, and the old full rebuild used to drop
// every entered value because the markup claimed the field was empty.
function _flagValue(id){
  const v=(window._flagSettings||{})[id];
  return (v==null||v===false)?'':String(Array.isArray(v)?v.join(','):v);
}
// Hover text for a flag: its trade-off line if one is written, the value's
// own line when the selected value pulls a different way, else the catalog's
// hover. All of it arrives from the server; the browser stores no copy.
function flagTip(f, cur){
  const byVal=f.tradeoff_by_value||{};
  return (cur && byVal[cur]) || f.tradeoff || f.hover || f.help || '';
}
function flagRowHtml(f){
  const src=f.source!=='upstream'? ' <span class="pill">'+esc(f.source)+'</span>':'';
  const cur=_flagValue(f.id);
  const tip=flagTip(f, cur);
  let h='<div class="setrow" id="flrow_'+f.id+'" title="'+esc(tip)+'">'
    +'<span class="lbl" title="'+esc(tip)+'">'+esc(f.name)+src
    +' <span class="flag-q" id="flq_'+f.id+'" style="color:#e3a008"></span>'
    +'<span class="chg" title="changed from preset"></span></span>';
  if(f.type==='bool')
    h+='<label class="switch"><input type="checkbox" id="fl_'+f.id+'"'
      +(cur?' checked':'')+' onchange="onFlagChange(\''+f.id+'\')"><span class="track"></span></label>';
  else if(f.allowed)
    h+='<select id="fl_'+f.id+'" onchange="onFlagChange(\''+f.id+'\')">'
      +'<option value=""'+(cur===''?' selected':'')+'>(default)</option>'
      +f.allowed.map(a=>'<option value="'+esc(String(a))+'"'
        +(String(a)===cur?' selected':'')+'>'+esc(String(a))+'</option>').join('')+'</select>';
  else if(f.type==='int'||f.type==='float')
    h+='<input type="number" class="num" id="fl_'+f.id+'" placeholder="'
      +(f.default!=null?esc(String(f.default)):'auto')+'" value="'+esc(cur)+'" '
      +(f.type==='float'?'step="any" ':'')+'onchange="onFlagChange(\''+f.id+'\')">';
  else
    h+='<input type="text" id="fl_'+f.id+'" placeholder="'
      +(f.default!=null?esc(String(f.default)):'default')+'" value="'+esc(cur)
      +'" onchange="onFlagChange(\''+f.id+'\')">';
  h+='<span class="qmark" title="'+esc(f.help||f.hover||'')+'">?</span></div>'
    +'<div class="knob-help" id="flh_'+f.id+'"></div>';
  return h;
}
function renderFlagSurface(){
  const d=window._flagCat; if(!d) return;
  window._flagSection={};
  const bySec={context:[],gpu:[],speculative:[],cache:[],serving:[],advanced:[]};
  for(const g of Object.keys(d.groups))
    for(const f of _surfaceSpecs(g)){
      const sec=flagSection(f);
      window._flagSection[f.id]=sec;
      bySec[sec].push(f);
    }
  // in-section order: fork/env flags after upstream, then by id (stable).
  for(const sec of SECTION_KEYS)
    bySec[sec].sort((a,b)=>(a.source!=='upstream')-(b.source!=='upstream')
      || (a.id<b.id?-1:1));
  // Patched: the rows carry the user's current values, and the patcher keeps
  // them. The old full rebuild dropped every entered value on the floor --
  // window._flagSettings held the truth but nothing wrote it back into the
  // DOM afterwards.
  for(const sec of ['context','gpu','speculative','cache','serving'])
    setHTML($('secflags_'+sec), bySec[sec].map(flagRowHtml).join(''));
  // advanced: everything else, still grouped by catalog group for
  // navigability, all behind the ONE toggle.
  const byGrp={};
  for(const f of bySec.advanced) (byGrp[f.group]=byGrp[f.group]||[]).push(f);
  let ah=''; let gi=0;
  for(const g of Object.keys(byGrp).sort()){
    ah+='<details id="flgrp_'+(gi++)+'" style="margin:.2rem 0"><summary style="cursor:pointer"><b>'
      +esc(g)+'</b> <span class="muted">('+byGrp[g].length+')</span></summary>'
      +byGrp[g].map(flagRowHtml).join('')+'</details>';
  }
  setHTML($('sec_advanced'), ah);
  $('adv_count').textContent='('+bySec.advanced.length+' more flags)';
  filterFlags();
  updateSectionSummaries();
  markPresetDrift();
}
function toggleAdvanced(){
  const q=($('flag_search').value||'').trim();
  if(!q) $('sec_advanced').style.display=$('advanced_toggle').checked?'':'none';
}
// Search/filter across ALL sections (static rows included): matching rows
// stay, sections with hits auto-open, the advanced block is revealed while a
// query has hits there; an empty query restores the collapsed default.
function filterFlags(){
  const d=window._flagCat; if(!d) return;
  const q=($('flag_search').value||'').trim().toLowerCase();
  const secHits={context:0,gpu:0,speculative:0,cache:0,serving:0,advanced:0};
  // static (serving-owned) rows carry their haystack in data-hay.
  // row_split_mode / row_gpu_pick / colo_box are STATE-driven (updateColoUI /
  // updateGpuPick own their visibility) and stay out of the search filter.
  const _staticSec={row_sv_ctx:'context', row_mrr:'context', row_tp_count:'gpu',
    row_sv_served:'serving', row_sv_host:'serving', row_sv_port:'serving'};
  for(const rid of Object.keys(_staticSec)){
    const row=$(rid); if(!row) continue;
    const hit=!q || (row.getAttribute('data-hay')||'').indexOf(q)>=0;
    row.style.display=hit?'':'none';
    if(hit) secHits[_staticSec[rid]]++;
  }
  for(const g of Object.keys(d.groups)){
    for(const f of _surfaceSpecs(g)){
      const row=$('flrow_'+f.id); if(!row) continue;
      const hay=(f.name+' '+f.id+' '+(f.help||'')+' '+(f.hover||'')).toLowerCase();
      const hit=!q || hay.indexOf(q)>=0;
      row.style.display=hit?'':'none';
      const hlp=$('flh_'+f.id); if(hlp) hlp.style.display=hit?'':'none';
      if(hit) secHits[window._flagSection[f.id]||'advanced']++;
    }
  }
  // A search opens the sections that have hits, so the reader can see what
  // matched. Clearing the search must NOT slam everything shut again: only
  // the sections the search itself opened are put back, and a section the
  // reader opened by hand is left alone. That is what _searchOpened records.
  for(const sec of ['context','gpu','speculative','cache','serving']){
    const det=$('sec_'+sec); if(!det) continue;
    if(q){
      det.style.display=secHits[sec]?'':'none';
      if(secHits[sec]>0 && !det.open){ det.open=true; det._searchOpened=true; }
    } else {
      det.style.display='';
      if(det._searchOpened){ det.open=false; det._searchOpened=false; }
    }
  }
  // advanced groups: show only those with hits while searching.
  let gi=0; let det;
  while((det=$('flgrp_'+(gi++)))){
    let vis=0;
    for(const row of det.querySelectorAll('.setrow'))
      if(row.style.display!=='none') vis++;
    det.style.display=vis?'':'none';
    if(q){
      if(vis>0 && !det.open){ det.open=true; det._searchOpened=true; }
    } else if(det._searchOpened){ det.open=false; det._searchOpened=false; }
  }
  $('sec_advanced').style.display =
    q? (secHits.advanced?'':'none') : ($('advanced_toggle').checked?'':'none');
}
// Collapsed-section summary line: the effective (non-default) values.
function _sumPush(list, name, v){
  if(v==null||v===''||v===false) return;
  list.push(name+'='+String(Array.isArray(v)?v.join(','):v));
}
function updateSectionSummaries(){
  const s=window._flagSettings||{};
  const by={context:[],gpu:[],speculative:[],cache:[],serving:[],advanced:[]};
  const ctx=$('sv_ctx').value;
  if(ctx) by.context.push('ctx '+parseInt(ctx).toLocaleString()
    +(window._ctxCap?' / max '+window._ctxCap.toLocaleString():''));
  _sumPush(by.context,'reqs',$('max_running_requests').value);
  const host=$('sv_host').value||'127.0.0.1', port=$('sv_port').value;
  if(port) by.serving.push(host+':'+port);
  _sumPush(by.serving,'served',$('sv_served').value);
  for(const id of Object.keys(s)){
    const sec=window._flagSection[id]||'advanced';
    _sumPush(by[sec], id, s[id]);
  }
  for(const sec of ['context','gpu','speculative','cache','serving']){
    const el=$('sum_'+sec); if(!el) continue;
    const items=by[sec];
    el.textContent=items.length
      ? '— '+items.slice(0,3).join(' · ')+(items.length>3?' +'+(items.length-3):'')
      : '';
  }
}
function _flagSpec(id){
  const d=window._flagCat; if(!d) return null;
  for(const g of Object.keys(d.groups)){ const f=d.groups[g].find(x=>x.id===id); if(f) return f; }
  return null;
}
function onFlagChange(id){
  const el=$('fl_'+id); const spec=_flagSpec(id); if(!el||!spec) return;
  const v = spec.type==='bool'? el.checked : el.value.trim();
  if(v===''||v===false) delete window._flagSettings[id];
  else window._flagSettings[id]=v;
  // A manual edit invalidates the applied profile's EXACT argv (the launch
  // then falls back to the form fields; the profile env stays applied).
  if(window._profileArgv){ window._profileArgv=null; window._profileDirty=true; renderProfileLaunch(); }
  if(id==='speculative_draft_model_path'){ renderDraftPick(); updateSpecDraftHint(); }
  // tp_size change -> the co-location steppers re-derive the DEFAULT
  // VRAM-proportional distribution (spec'd behavior; a manual-unparseable
  // rank_gpu_id is never overwritten).
  if(id==='tp_size') window._coloRedistribute=true;
  updateGpuPick(); updateColoUI(); markPresetDrift(); updateSectionSummaries();
  markTune();
  // ONE consistent recompute rather than three calls racing each other.
  scheduleRecompute();
}
// ---- context-length / max-running-requests: slider+numeric pairs ----------
// The numeric field (sv_ctx / max_running_requests) stays authoritative --
// every existing reader keeps its id. The slider clamps to the plan's
// computed KV-capacity max when a plan result exists; the numeric is free.
window._ctxCap=null;
function setCtxCap(cap){
  window._ctxCap=cap? Math.round(cap):null;
  const sl=$('sv_ctx_slider'); if(!sl) return;
  sl.max=window._ctxCap||262144;
  $('sv_ctx_max').textContent=window._ctxCap
    ? 'max '+window._ctxCap.toLocaleString():'';
  const v=parseInt($('sv_ctx').value)||0;
  sl.value=Math.min(v||8192, parseInt(sl.max));
  updateSectionSummaries();
}
function _reflow(){ scheduleRecompute(); }
function ctxFromSlider(){
  $('sv_ctx').value=$('sv_ctx_slider').value;
  onServingEdit(); _reflow();
}
function ctxFromNum(){
  const sl=$('sv_ctx_slider');
  sl.value=Math.min(parseInt($('sv_ctx').value)||8192, parseInt(sl.max));
  onServingEdit(); _reflow();
}
// Sliders fire on every pixel of travel. The label follows the thumb at once
// (that has to feel immediate), the backend work is debounced behind it.
function _replan(){ scheduleRecompute(); }
function mrrFromSlider(){
  $('max_running_requests').value=$('mrr_slider').value;
  onServingEdit(); _replan();
}
function mrrFromNum(){
  const v=parseInt($('max_running_requests').value);
  // The numeric field is authoritative and unbounded. A hard 64 here used to
  // silently swallow anything larger; the real limit is the plan's reject,
  // which says WHY, so the slider grows to represent what was typed instead.
  if(v) setMrrCap(Math.max(window._mrrCap||64, v));
  if(v) $('mrr_slider').value=v;
  onServingEdit(); _replan();
}
// Slider bounds follow the hard limits the planner reports, never a preset's
// value: applying a template is a starting point, not a ceiling.
window._mrrCap=null;
function setMrrCap(cap){
  cap=Math.max(1, Math.round(cap||64));
  window._mrrCap=cap;
  const sl=$('mrr_slider'); if(!sl) return;
  sl.max=cap;
  const note=$('mrr_max');
  if(note) note.textContent='max '+cap.toLocaleString();
}
function onServingEdit(){ markPresetDrift(); updateSectionSummaries(); }
// ---- single-GPU card selector (tp=1): writes the stock --base-gpu-id ------
function _effectiveTp(){
  const s=window._flagSettings||{};
  return (s.tp_size? parseInt(s.tp_size):null)
    || (CARDS.filter(c=>c.include).length||1);
}
function updateGpuPick(){
  const row=$('row_gpu_pick'); if(!row) return;
  // Only DETECTED cards carry a real device index --base-gpu-id can address
  // (CARDS order is display order; virtual cards have no physical index).
  // --base-gpu-id is a CUDA-ORDER index (it lands in CUDA_VISIBLE_DEVICES),
  // so the option VALUES are cuda indices, never nvml indices -- on this
  // rig cuda:0 is the 5090 even though it sits at nvml:1.
  const detected=CARDS.filter(c=>c.cuda_index!=null || c.nvml_index!=null)
    .map(c=>({card:c, cuda:(c.cuda_index!=null?c.cuda_index:c.nvml_index),
              exact:c.cuda_index!=null}))
    .sort((a,b)=>a.cuda-b.cuda);
  const show=_effectiveTp()===1 && detected.length>0;
  row.style.display=show?'':'none';
  if(!show) return;
  const s=window._flagSettings||{};
  const sel=$('gpu_pick_select');
  sel.innerHTML=detected.map(d=>'<option value="'+d.cuda+'">'
    +(d.exact? devLabel(d.cuda, d.card.nvml_index)
             : ('gpu '+d.cuda+' (unbridged index!)'))
    +' &mdash; '+esc(d.card.name)+' ('+(d.card.total_mib/1024).toFixed(0)
    +'G)</option>').join('');
  const cur=(s.base_gpu_id!=null && s.base_gpu_id!=='')? String(s.base_gpu_id):'0';
  sel.value=cur;
  if(sel.selectedIndex<0) sel.selectedIndex=0;
}
function gpuPickChanged(){
  const el=$('fl_base_gpu_id');
  if(el){ el.value=$('gpu_pick_select').value; onFlagChange('base_gpu_id'); }
}
// ---- co-location controls (GPU offload/split section) ---------------------
// tp_count mirrors fl_tp_size (authoritative). When tp exceeds the enabled
// card count the per-card rank steppers activate: one row per enabled card
// (CUDA order, dual cuda/nvml label). Derivation rule (canonical, mirrors
// flags.rank_gpu_id_from_counts): cards ascending by cuda index, each card's
// index repeated its rank count -- 2 on cuda:0 + 1 each on cuda:1/2 ->
// rank_gpu_id 0,0,1,2. fl_rank_gpu_id stays THE authoritative field; manual
// edits there reverse-populate the steppers when parseable, else the
// steppers grey out with the 'manual rank_gpu_id active' note.
function coloCards(){
  // Enabled cards keyed in CUDA space: detected cards use the bridged
  // cuda_index; virtual/undetected cards fill the lowest unused keys (the
  // exact rule the plan payload / runnerFlags inventory uses).
  const incl=CARDS.filter(c=>c.include);
  const used={};
  incl.forEach(c=>{ if(c.cuda_index!=null) used[c.cuda_index]=1; });
  let nextFree=0;
  const out=incl.map(c=>{
    let k=c.cuda_index;
    if(k==null){ while(used[nextFree]) nextFree++; k=nextFree; used[k]=1; }
    return {cuda:k, nvml:c.nvml_index, name:c.name, total_mib:c.total_mib||0};
  });
  out.sort((a,b)=>a.cuda-b.cuda);
  return out;
}
function coloDefaultCounts(tp, cards){
  // Default distribution: VRAM-proportional floor allocation, remaining
  // ranks to the LARGEST card first (descending VRAM, tie: lowest cuda,
  // cycling) -- on the reference box tp=4 puts the extra rank on the 5090
  // (2/1/1). Mirrors flags.colocation_rank_counts; keep the two in sync.
  const sum=cards.reduce((a,c)=>a+Math.max(0,c.total_mib),0)||1;
  const counts={}; let assigned=0;
  for(const c of cards){
    const n=Math.floor(tp*Math.max(0,c.total_mib)/sum);
    counts[c.cuda]=n; assigned+=n;
  }
  const order=cards.slice().sort((a,b)=>(b.total_mib-a.total_mib)||(a.cuda-b.cuda));
  let i=0;
  while(assigned<tp && order.length){ counts[order[i%order.length].cuda]++; assigned++; i++; }
  return counts;
}
function coloRankGpuIdFromCounts(counts, cards){
  // canonical order: ascending cuda, each index repeated its count.
  const out=[];
  for(const c of cards) for(let i=0;i<(counts[c.cuda]||0);i++) out.push(c.cuda);
  return out.join(',');
}
function coloParseRankGpuId(v, cards){
  // -> {cuda: count} when every entry is an integer naming an enabled card,
  // else null (manual mode). Mirrors flags.rank_counts_from_gpu_id.
  const toks=String(v==null?'':v).replace(/[\s\[\]]/g,'').split(',').filter(x=>x!=='');
  if(!toks.length) return null;
  const legal={}; cards.forEach(c=>{legal[c.cuda]=1;});
  const counts={};
  for(const t of toks){
    if(!/^\d+$/.test(t)) return null;
    const k=parseInt(t);
    if(!legal[k]) return null;
    counts[k]=(counts[k]||0)+1;
  }
  return counts;
}
function tpCountChanged(){
  const el=$('fl_tp_size'); if(!el) return;
  el.value=$('tp_count').value.trim();
  onFlagChange('tp_size');  // sets _coloRedistribute (default spread on tp change)
}
function coloRanksChanged(){
  // Self-updating by construction: the changed stepper stands, the sum /
  // free-remainder line updates, and the canonical rank_gpu_id is written
  // to the authoritative field (a wrong sum surfaces as the inline error +
  // resolve()'s tuple-length error and blocks Launch/Plan).
  const cards=coloCards(); const counts={};
  for(const c of cards){
    const el=$('colo_ranks_'+c.cuda);
    counts[c.cuda]=el? Math.max(0, parseInt(el.value)||0) : 0;
  }
  const el=$('fl_rank_gpu_id');
  if(el){ el.value=coloRankGpuIdFromCounts(counts, cards); onFlagChange('rank_gpu_id'); }
}
function splitModeChanged(){
  const v=$('split_mode_select').value;
  if(v==='custom') return;  // the free-text rank-tp-ratio field is authoritative
  const el=$('fl_rank_tp_ratio'); if(!el) return;
  el.value=(v==='even')? '' : v;   // even = uniform split: ratio cleared
  onFlagChange('rank_tp_ratio');
}
function coloBlockError(){
  // The Launch/Plan gate: while the co-location steppers are active their
  // sum must equal tp_size (manual rank_gpu_id mode gates via resolve()).
  const box=$('colo_box');
  if(!box || box.style.display==='none' || window._coloManual) return null;
  const tp=_effectiveTp(); const cards=coloCards();
  let sum=0;
  for(const c of cards){
    const el=$('colo_ranks_'+c.cuda); sum+=el? (parseInt(el.value)||0):0;
  }
  if(sum===tp) return null;
  return 'co-location rank assignment: '+sum+' of '+tp+' ranks placed ('
    +(sum<tp? (tp-sum)+' still free' : (sum-tp)+' too many')
    +') -- the per-card sum must equal tp before Launch/Plan.';
}
function updateColoUI(){
  const box=$('colo_box'); if(!box) return;
  const s=window._flagSettings||{};
  const cards=coloCards();
  const tp=_effectiveTp();
  // tp mirror (authoritative field: fl_tp_size).
  const tpc=$('tp_count');
  if(tpc){
    tpc.value=(s.tp_size!=null && s.tp_size!=='')? String(s.tp_size):'';
    tpc.placeholder='(cards: '+(cards.length||1)+')';
  }
  // even/uneven selector state from the authoritative fl_rank_tp_ratio.
  const smRow=$('row_split_mode');
  if(smRow){
    smRow.style.display=(tp>1)?'':'none';
    const rtr=(s.rank_tp_ratio!=null && s.rank_tp_ratio!=='')? String(s.rank_tp_ratio):'';
    const sel=$('split_mode_select');
    sel.value=(rtr==='')?'even':(rtr==='auto'||rtr==='auto-performance')?rtr:'custom';
  }
  const active=tp>cards.length && cards.length>0;
  box.style.display=active?'':'none';
  if(!active){ window._coloRedistribute=false; window._coloManual=false; return; }
  const raw=s.rank_gpu_id;
  const has=(raw!=null && String(raw)!=='');
  let counts=has? coloParseRankGpuId(raw, cards):null;
  let manual=has && counts==null;
  if((!has || (window._coloRedistribute && !manual))){
    // no mapping yet, or tp_size changed: apply the default VRAM-
    // proportional distribution and write the derived canonical
    // rank_gpu_id into the authoritative field (no onFlagChange recursion:
    // the caller's resolve/replan pass picks the new value up).
    counts=coloDefaultCounts(tp, cards);
    const v=coloRankGpuIdFromCounts(counts, cards);
    const el=$('fl_rank_gpu_id'); if(el) el.value=v;
    window._flagSettings=window._flagSettings||{};
    if(v==='') delete window._flagSettings.rank_gpu_id;
    else window._flagSettings.rank_gpu_id=v;
    manual=false;
  }
  window._coloRedistribute=false;
  window._coloManual=manual;
  // one stepper row per enabled card, CUDA-ordered, dual cuda/nvml label.
  $('colo_rows').innerHTML=cards.map(c=>
    '<div class="setrow" id="colorow_'+c.cuda+'">'
    +'<span class="lbl">ranks on ['+ (devLabel(c.cuda, c.nvml)||('gpu '+c.cuda)) +'] '
    +esc(c.name)+' <span class="muted">('+(c.total_mib/1024).toFixed(0)+'G)</span></span>'
    +'<input type="number" class="num" id="colo_ranks_'+c.cuda+'" min="0" step="1"'
    +(manual?' disabled':'')+' value="'+((counts&&counts[c.cuda])||0)+'"'
    +' onchange="coloRanksChanged()">'
    +'</div>').join('');
  $('colo_rows').style.opacity=manual?'0.45':'1';
  $('colo_manual_note').style.display=manual?'':'none';
  let sum=0; if(counts) for(const k of Object.keys(counts)) sum+=counts[k];
  $('colo_sum').textContent=manual? ''
    : 'assigned '+sum+' of '+tp+' ranks -- free: '+(tp-sum);
  const err=coloBlockError();
  $('colo_err').style.display=err?'':'none';
  $('colo_err').textContent=err||'';
}
// ---- draft-model selector (Speculative section): matching LOCAL draft
// heads for the selected base model; picking one writes the authoritative
// fl_speculative_draft_model_path free-text field. -------------------------
function renderDraftPick(){
  const sel=$('draft_model_select'); if(!sel) return;
  const cands=window._draftCandidates||[];
  const cur=((window._flagSettings||{}).speculative_draft_model_path)||'';
  sel.innerHTML='<option value="">(none)</option>'
    +cands.map(c=>'<option value="'+esc(c.path)+'">'+esc(c.name)
      +' ['+esc(c.algorithm)+']</option>').join('');
  sel.value=cands.some(c=>c.path===cur)? cur : '';
}
function draftPickChanged(){
  const el=$('fl_speculative_draft_model_path');
  if(el){ el.value=$('draft_model_select').value;
    onFlagChange('speculative_draft_model_path'); }
  updateSpecDraftHint();
}
// Subtle amber hint (not a blocking banner): the selected base model has NO
// MTP head and no draft model is chosen -> spec needs a draft model; name
// the best local suggestion when one matched, else say none was found.
function updateSpecDraftHint(){
  const el=$('spec_draft_hint'); if(!el) return;
  const chosen=((window._flagSettings||{}).speculative_draft_model_path)
    || ($('fl_speculative_draft_model_path')&&$('fl_speculative_draft_model_path').value.trim());
  const cands=window._draftCandidates||[];
  if(window._modelHasMtp===false && !chosen){
    el.style.display='';
    el.textContent='this model has no MTP head - pick a draft model for '
      +'speculative decoding '
      +(cands.length? '(suggestion: '+cands[0].name+')'
                    : '(none matching found locally)');
  } else { el.style.display='none'; el.textContent=''; }
}
// ---- changed-from-preset markers ------------------------------------------
// Snapshot every row's value when a preset is applied; a differing row gets
// the subtle amber dot. No preset applied -> no markers.
const _STATIC_ROWS={sv_ctx:'row_sv_ctx', max_running_requests:'row_mrr',
  tp_count:'row_tp_count',
  sv_served:'row_sv_served', sv_host:'row_sv_host', sv_port:'row_sv_port'};
function _rowValue(el){
  if(!el) return '';
  return el.type==='checkbox'? (el.checked?'1':'') : String(el.value||'');
}
function presetSnapshot(){
  const snap={};
  for(const fid of Object.keys(_STATIC_ROWS)) snap[fid]=_rowValue($(fid));
  for(const id of Object.keys(window._flagSection))
    snap['fl_'+id]=_rowValue($('fl_'+id));
  return snap;
}
function markPresetDrift(){
  const base=window._presetBase;
  const mark=(rowId, changed)=>{
    const row=$(rowId); if(row) row.classList.toggle('changed', !!changed);
  };
  for(const fid of Object.keys(_STATIC_ROWS))
    mark(_STATIC_ROWS[fid], base && _rowValue($(fid))!==base[fid]);
  for(const id of Object.keys(window._flagSection))
    mark('flrow_'+id, base && _rowValue($('fl_'+id))!==base['fl_'+id]);
}
function collectFlagSettings(){
  // The flag surface IS the flag truth; only the single model selector and
  // the SERVING identity group add their (authoritative) fields on top.
  const s=Object.assign({}, window._flagSettings);
  const ms=modelState();
  if(ms.path) s.model_path=ms.path;
  const served=$('sv_served').value.trim(); if(served) s.served_model_name=served;
  const host=$('sv_host').value.trim(); if(host) s.host=host;
  if($('sv_port').value.trim()) s.port=parseInt($('sv_port').value)||30000;
  if($('sv_ctx').value.trim()) s.context_length=parseInt($('sv_ctx').value)||8192;
  if($('max_running_requests').value.trim())
    s.max_running_requests=parseInt($('max_running_requests').value)||1;
  return s;
}
async function resolveFlags(){
  if(!window._flagCat) return;
  const model=$('model').value.trim();
  try{
    const d=await api('/api/resolve_flags',{key:'resolve_flags',
      body:{settings:collectFlagSettings(), model}});
    if(!d.ok) return;
    applyFieldStates(d.fields); renderFlagWarnings(d.warnings);
  }catch(e){}
}
function applyFieldStates(fields){
  for(const id of Object.keys(fields)){
    const st=fields[id]; const el=$('fl_'+id); if(!el) continue;
    el.disabled=!st.enabled;
    const row=$('flrow_'+id); if(row) row.style.opacity=st.enabled?'1':'0.45';
    const q=$('flq_'+id); if(q){ q.textContent=st.disabled_reason?'(?)':''; q.title=st.disabled_reason||''; }
    const help=$('flh_'+id);
    if(help){
      let parts=[];
      if(st.auto_set) parts.push('auto-set: '+JSON.stringify(st.value));
      if(st.disabled_reason) parts.push(st.disabled_reason);
      if(st.error) parts.push(st.error);
      help.innerHTML = parts.length? '<span class="'+(st.error?'nofitc':'muted')+'">'+esc(parts.join(' · '))+'</span>':'';
    }
    if(st.options && el.tagName==='SELECT')
      setHTML(el, '<option value="">(default)</option>'
        +st.options.map(a=>'<option value="'+esc(String(a))+'">'+esc(String(a))+'</option>').join(''));
  }
}
function renderFlagWarnings(ws){
  const box=$('flag_warnings'); if(!box) return;
  if(!ws||!ws.length){ box.innerHTML=''; return; }
  box.innerHTML=ws.map(w=>'<div class="'+(w.level==='error'?'reasons':'muted')
    +'" style="border:1px solid #e3a008;padding:.3rem;border-radius:6px;margin:.2rem 0">'
    +(w.level==='error'?'BLOCK: ':'NOTE: ')+esc(w.message)+'</div>').join('');
}
async function onRunnerModel(){
  const model=$('model').value.trim(); if(!model) return;
  loadConfigProfiles(); scheduleRecompute();
}

// ===========================================================================
// Live propagation: ONE call, ONE consistent answer.
//
// Every dimension the fork exposes feeds every dependent value, and all of
// that arithmetic lives server-side already. What the page used to do was ask
// three separate questions about the same configuration -- /api/plan,
// /api/placement and /api/resolve_flags, fired together on every edit -- and
// paint whichever answer arrived last. With three independent latencies the
// panels could end up describing two different configurations at once.
//
// /api/recompute answers all three about one configuration. One call cannot
// disagree with itself. It is also one round trip instead of three, which is
// most of why editing feels immediate now.
// ===========================================================================
function recomputeSections(){
  // Only what the visible mode actually shows.
  return viewMode()==='simple'
    ? ['plan','placement']
    : ['plan','placement','fields'];
}
async function recomputeNow(){
  const model=$('model').value.trim();
  if(!model){ renderSimpleCards(null); return; }
  const coloErr=coloBlockError();
  if(coloErr){
    $('verdict').innerHTML='<div class="verdict nofit">PLAN BLOCKED</div>';
    $('split').innerHTML='<ul class="reasons"><li>'+esc(coloErr)+'</li></ul>';
    renderSimpleVerdict({valid:false, reasons:[coloErr]});
    return;
  }
  const flags=runnerFlags();
  const body=Object.assign({}, payload(), {
    sections: recomputeSections(),
    settings: collectFlagSettings(),
    placement_request: {
      model,
      gguf_choice: ($('gguf_pick').style.display!=='none'?$('gguf_choice').value:null),
      flags: flags,
      stock_compare: flagsAreForkish(flags),
    },
  });
  const rp=$('runner_placement');
  stale(rp,true);
  let d;
  try{
    d=await api('/api/recompute',{key:'recompute', body:body, timeout:20000});
  }catch(e){
    // Superseded by a newer edit: the newer answer owns the panels.
    if(apiAborted(e)) return;
    $('verdict').innerHTML='<div class="verdict nofit">PLAN ERROR</div>';
    $('split').innerHTML='<ul class="reasons"><li>'+esc(apiError(e))+'</li></ul>';
    return;
  } finally { stale(rp,false); }
  if(d.fields && d.fields.ok){
    applyFieldStates(d.fields.fields); renderFlagWarnings(d.fields.warnings);
  }
  if(d.placement) applyPlacementResult(d.placement);
  if(d.plan){ render(d.plan); renderSimpleVerdict(d.plan); }
  renderSimpleCards(d.placement && d.placement.ok ? d.placement.placement : null);
}
const scheduleRecompute=debounce(recomputeNow, 180);

// ===========================================================================
// Simple view: per card, what this configuration puts on it, and one slider
// for how much of the card it may have. No breakdown, no ratios -- those are
// the expert view. The numbers come from the same placement result the
// expert view renders granularly; nothing here is computed in the browser.
// ===========================================================================
function cardBudgetMib(c){
  // The budget slider writes the existing per-card reserve: budget = total
  // minus what is kept free. One mechanism, two presentations.
  const total=c.total_mib||0;
  const keep=Math.round((c.reserve_gb||0)*1024);
  return Math.max(0, total-keep);
}
function setCardBudgetMib(i, mib){
  const c=CARDS[i]; if(!c) return;
  const total=c.total_mib||0;
  c.reserve_gb=Math.max(0, (total-Math.min(mib,total)))/1024;
}
function simpleBudgetInput(i){
  const c=CARDS[i]; if(!c) return;
  const sl=$('csb_'+i); if(!sl) return;
  setCardBudgetMib(i, parseInt(sl.value)||0);
  // The label follows the thumb at once; the re-plan is debounced behind it.
  const lbl=$('csv_'+i);
  if(lbl) lbl.textContent=fmtGiB(cardBudgetMib(c))+' of '+fmtGiB(c.total_mib);
  scheduleRecompute();
}
// Distinct from the placement renderer's fmtMib: the simple view spells the
// unit out, because it shows one number per card and has the room.
function fmtGiB(m){
  if(m==null||isNaN(m)) return 'n/a';
  return (m/1024).toFixed(1)+' GiB';
}

// ===========================================================================
// LIVE panel (runner tab, both densities).
//
// The planner draws what a configuration WOULD occupy. This draws what the
// running server DOES occupy, with the same bar, so the two are comparable by
// looking rather than by arithmetic. Everything here comes from one
// /api/live_snapshot poll -- NVML per card, the Prometheus scrape for rates.
//
// Fill colour is state, using the thresholds these tools converge on: under
// 80% of the card normal, 80-90% tight, above 90% critical. Nothing on this
// panel is coloured for decoration.
// ===========================================================================
const LIVE_POLL_MS=2000;
window._liveTimer=null;
// A server started WITHOUT --enable-metrics serves no /metrics, so every rate
// on this page is unavailable -- not zero, not pending. Say which of the two
// it is, and say what to do about it; an empty widget that never fills is the
// worst of the three states to be shown.
// WHICH server is meant has to be on the banner. A state line that names no
// target reads as a statement about the server the reader has in mind, and a
// stray process on a nearby port then looks like the real one.
function targetLabel(s){
  const n=s?normalizeStartConfig(s):null;
  const cfg=(n&&n.cfg)||{};
  const st=(s&&s.status)||null;
  const name=cfg.served_model_name||cfg.model_path||(st&&st.served_model_name)
             ||(st&&st.model_path)||'unnamed model';
  let port=cfg.port||(st&&st.port)||null;
  if(!port && s && s.endpoint){
    const m=(''+s.endpoint).match(/:(\d+)\s*$/); if(m) port=m[1];
  }
  const pid=(st&&st.pid)||null;
  return esc(String(name))+(port?(' @ :'+esc(String(port))):'')
        +(pid?(' &middot; pid '+esc(String(pid))):'');
}
function noMetricsBanner(err, s){
  return '<div class="verdict offload" style="font-size:var(--t-sm);'
    +'font-weight:400;margin:var(--s2) 0 0">'
    +'<b>Server started without --enable-metrics</b>'
    +(s?(' &mdash; <b>'+targetLabel(s)+'</b>'):'')
    +' &mdash; live rates (decode / prefill tok/s, per-session throughput, '
    +'MTP acceptance, cache hit) are not available from this server. Per-card '
    +'VRAM, power and utilisation come from NVML and keep working. Restart '
    +'the server with <code>--enable-metrics</code>; a server booted from '
    +'this dashboard always has it.'
    +(err?'<div class="muted" style="margin-top:var(--s1)">'+esc(err)+'</div>':'')
    +'</div>';
}
function vramClass(frac){
  if(frac==null) return '';
  if(frac>=0.90) return ' over';
  if(frac>=0.80) return ' tight';
  return '';
}
// One card's measured occupancy, on the planner's own bar.
function liveCardHtml(g, tokS){
  const frac=(g.mem_total_mib? g.mem_used_mib/g.mem_total_mib : null);
  const pct=frac==null?0:Math.min(100,frac*100);
  const idx=devLabel(g.cuda_index, g.nvml_index)||('nvml:'+g.nvml_index);
  const key='gpu'+g.nvml_index;
  pushRing(key+'_util', window._liveT, g.utilization_pct);
  pushRing(key+'_pow', window._liveT, g.power_watts);
  pushRing(key+'_mem', window._liveT, frac==null?null:frac*100);
  // Tokens per watt for ONE card: the server-wide token rate over what this
  // card is drawing. It is what this card costs to keep in the group -- the
  // tokens are produced by the whole TP group, never by one card alone, and
  // the label says so rather than implying a per-card share of the work.
  const tpw=(tokS!=null&&g.power_watts>0)?(tokS/g.power_watts):null;
  return '<div class="cardsimple" data-key="live_'+g.nvml_index+'">'
    +'<div class="cs-h"><span class="cs-n">'+esc(g.name)
    +' <span class="muted" style="font-weight:400">['+esc(idx)+']</span></span>'
    +'<span class="cs-u">'+fmtGiB(g.mem_used_mib)+' of '+fmtGiB(g.mem_total_mib)
    +' &middot; '+pct.toFixed(0)+'%</span></div>'
    +'<div class="csbar" title="measured NVML occupancy of this card">'
    +'<div class="fill'+vramClass(frac)+'" style="width:'+pct.toFixed(1)+'%"></div></div>'
    +'<div class="csrow" style="font-size:var(--t-xs);color:var(--fg-muted);'
    +'flex-wrap:wrap;gap:var(--s3)">'
    +'<span>util <b>'+g.utilization_pct+'%</b> '+sparkline(key+'_util',72,16)+'</span>'
    +'<span>power <b>'+g.power_watts.toFixed(0)+'</b>/'+g.power_limit_w.toFixed(0)
    +' W '+sparkline(key+'_pow',72,16)+'</span>'
    +'<span>VRAM '+sparkline(key+'_mem',72,16)+'</span>'
    +'<span>'+g.temperature_c+' &deg;C</span>'
    +'<span>SM '+g.sm_clock_mhz+' MHz</span>'
    +'<span title="server-wide token rate over this card\'s NVML power draw">'
    +'tok/W <b>'+(tpw!=null?tpw.toFixed(2):'--')+'</b></span>'
    +'</div></div>';
}
function liveTile(label, value, sub, spark){
  return '<div class="mtile" data-key="'+label.replace(/[^a-z]/gi,'')+'">'
    +'<div class="mt-l">'+label+'</div><div class="mt-v">'+value+'</div>'
    +'<div class="mt-s">'+(sub||'&nbsp;')+'</div>'
    +(spark?stripSpark(spark):'')+'</div>';
}
// Exactly three states, and the panel is in one of them after EVERY poll:
//   1. no inference server running        -- neutral, not an error;
//   2. a server that serves no /metrics   -- warning, names the target;
//   3. a server with metrics              -- the live readings.
// The banner is a function of the current tick, never something set once.
function renderLivePanel(s, envelope){
  const note=$('live_note'), strip=$('live_strip'), cards=$('live_cards');
  if(!note) return;
  if(!s||!s.ok||!s.gpus){
    note.style.display=''; strip.style.display='none';
    $('live_legend').style.display='none';
    setHTML(cards,'');
    const st=(envelope&&envelope.status)||null;
    const err=(s&&s.error)||(envelope&&envelope.error)||null;
    if(err){
      setHTML(note,'<span class="rev-error">'+esc(err)+'</span>');
    } else if(st && st.state && st.state!=='stopped'){
      setHTML(note,'<span class="muted">Managed server is <b>'+esc(st.state)
        +'</b>'+(st.model_path?(' &mdash; '+esc(st.model_path)):'')
        +(st.port?(' @ :'+esc(String(st.port))):'')
        +' &mdash; readings appear once it is ready.</span>');
    } else {
      setHTML(note,'<span class="muted"><b>No inference server running.</b> '
        +'Nothing to read here yet &mdash; start one in the Planner tab, or '
        +'point the monitor at a running one above.</span>');
    }
    return;
  }
  window._liveT=s.t;
  const noMetrics=!!s.metrics_error;
  const rates=s.rates||null;
  const dec=rates?rates.decode_tok_s:null, pfx=rates?rates.prefill_tok_s:null;
  const running=(s.num_running_reqs!=null)?s.num_running_reqs:null;
  const queued=(s.num_queue_reqs!=null)?s.num_queue_reqs:null;
  // Per session = the server-wide rate divided by the requests it is actually
  // serving. With nothing running the per-session figure is undefined, not 0.
  const per=(v)=>(v==null||running==null||running<1)?null:v/running;
  const decPer=per(dec), pfxPer=per(pfx);
  let watts=null;
  for(const g of s.gpus) if(g.power_watts!=null) watts=(watts||0)+g.power_watts;
  const EPS=0.05;
  let tokS=null;
  if(dec!=null&&dec>EPS) tokS=dec; else if(pfx!=null&&pfx>EPS) tokS=pfx;
  const tpwTotal=(tokS!=null&&watts)?tokS/watts:null;
  stripPush('lv_dec',s.t,dec); stripPush('lv_pfx',s.t,pfx);
  stripPush('lv_tpw',s.t,tpwTotal);
  // Without /metrics a rate is UNAVAILABLE, which is a different fact from
  // "idle at 0" -- the tiles must not be able to be read as the latter.
  const NA='<span class="muted">n/a</span>';
  const n1=(v)=>noMetrics?NA:(v==null?'&ndash;':Number(v).toFixed(1));
  const n2=(v)=>noMetrics?NA:(v==null?'&ndash;':Number(v).toFixed(2));
  note.style.display=noMetrics?'':'none';
  if(noMetrics) setHTML(note, noMetricsBanner(s.metrics_error, s));
  strip.style.display=''; $('live_legend').style.display='';
  setHTML(strip,
    liveTile('decode per session',
      n1(decPer)+'<small> tok/s</small>',
      'server total '+n1(dec)+' tok/s &middot; '
      +(running==null?'sessions n/a':running+' running')
      +(queued?(' &middot; '+queued+' queued'):''), 'lv_dec')
   +liveTile('prefill per session',
      n1(pfxPer)+'<small> tok/s</small>',
      'server total '+n1(pfx)+' tok/s (non-cached)', 'lv_pfx')
   +liveTile('tokens per watt (total)',
      n2(tpwTotal)+'<small> tok/W</small>',
      (watts!=null?Math.round(watts)+' W across '+s.gpus.length+' cards':'-- W')
      +(tokS!=null?(' &middot; '+n1(tokS)+' tok/s'):' &middot; idle'), 'lv_tpw')
   +liveTile('energy per token',
      (tpwTotal?n1(1/tpwTotal):(noMetrics?NA:'&ndash;'))+'<small> J/tok</small>',
      tpwTotal?'at the current phase rate'
        :(noMetrics?'needs /metrics for the token rate':'undefined while idle'),
      ''));
  setHTML(cards, s.gpus.map(g=>liveCardHtml(g, tokS)).join(''));
}
// The Planner's default is the configuration that is CURRENTLY LOADED. An
// empty form is only the right starting point when nothing is running; when a
// server is up, the thing the reader wants to edit is what that server was
// started with. Runs once per visit and never overwrites a form the user has
// already typed into.
window._prefilledFrom=null;
async function prefillFromRunning(){
  if(window._prefillBusy) return;
  window._prefillBusy=true;
  try{
    const ep=landingEndpoint();
    const d=await api('/api/live_snapshot'+(ep?('?endpoint='+encodeURIComponent(ep)):''),
                      {key:'prefill', timeout:4000});
    const s=(d&&d.running)?d.snapshot:null;
    const n=s?normalizeStartConfig(s):null;
    const cfg=(n&&n.cfg)||null;
    if(!cfg||!cfg.model_path){ renderLoadedConfigNote(null); return; }
    const key=cfg.model_path+'|'+(cfg.port||'');
    renderLoadedConfigNote(cfg, n.src);
    if(window._prefilledFrom===key) return;      // already applied this one
    if($('model').value.trim()) return;          // never overwrite the user
    window._prefilledFrom=key;
    $('model').value=cfg.model_path;
    applyRunningConfigToForm(cfg);
    onModelChange(); onRunnerModel();
  }catch(e){ /* the form simply stays empty */ }
  finally{ window._prefillBusy=false; }
}
function applyRunningConfigToForm(cfg){
  const direct={served_model_name:'sv_served', host:'sv_host', port:'sv_port',
                context_length:'sv_ctx', max_running_requests:'max_running_requests'};
  for(const k of Object.keys(direct)){
    const el=$(direct[k]);
    if(el && cfg[k]!=null && cfg[k]!=='') el.value=String(cfg[k]);
  }
  window._flagSettings=window._flagSettings||{};
  const flags={tp_size:'tp_size', rank_gpu_id:'rank_gpu_id',
               rank_tp_ratio:'rank_tp_ratio', rank_gpu_memory_mib:'rank_gpu_memory_mib',
               kv_cache_dtype:'kv_cache_dtype'};
  for(const k of Object.keys(flags)){
    const v=cfg[k]; if(v==null||v==='') continue;
    const str=Array.isArray(v)?v.join(','):String(v);
    const el=$('fl_'+flags[k]); if(el) el.value=str;
    window._flagSettings[flags[k]]=v;
  }
  window._presetBase=presetSnapshot();
  markPresetDrift(); updateSectionSummaries(); updateGpuPick(); updateColoUI();
  scheduleRecompute();
}
// The Planner states which configuration it is showing, always -- "the
// running one", "the running one, edited", or "nothing loaded".
function renderLoadedConfigNote(cfg, src){
  const box=$('loaded_cfg'); if(!box) return;
  if(!cfg){
    setHTML(box,'<span class="muted">No server running &mdash; this is a fresh '
      +'configuration. Pick a model below.</span>');
    return;
  }
  const bits=[];
  if(cfg.tp_size) bits.push('tp '+cfg.tp_size);
  if(cfg.rank_gpu_id) bits.push('ranks on '+[].concat(cfg.rank_gpu_id).join(','));
  if(cfg.context_length) bits.push('ctx '+cfg.context_length);
  if(cfg.kv_cache_dtype) bits.push('kv '+cfg.kv_cache_dtype);
  if(cfg.spec_mode) bits.push('spec '+cfg.spec_mode);
  setHTML(box,'<b>currently loaded:</b> '+esc(cfg.model_path||'(unnamed)')
    +(bits.length?' <span class="muted">&middot; '+esc(bits.join(' &middot; '))+'</span>':'')
    +' <span class="muted">&mdash; read from '+esc(src||'the running server')
    +'; the rows below start from it.</span>');
}
function renderSimpleCards(placement){
  const box=$('simple_cards'); if(!box) return;
  const incl=CARDS.filter(c=>c.include);
  if(!incl.length){ setHTML(box,'<span class="muted">no cards selected.</span>'); return; }
  // Placement reports per PHYSICAL card, keyed in CUDA space; map back onto
  // the card rows so a number is never attributed to the wrong card.
  const used={};
  (placement&&placement.cards||[]).forEach(pc=>{ used[pc.gpu_index]=pc; });
  let h='';
  CARDS.forEach((c,i)=>{
    if(!c.include) return;
    const key=(c.cuda_index!=null?c.cuda_index:i);
    const pc=used[key];
    const total=c.total_mib||0;
    const budget=cardBudgetMib(c);
    const need=pc?pc.total_mib:null;
    const pctNeed=(need!=null&&total)?Math.min(100, need/total*100):0;
    const pctCap=total?Math.min(100, budget/total*100):100;
    let cls='';
    if(need!=null&&need>budget) cls=' over';
    else if(need!=null&&budget&&need>budget*0.95) cls=' tight';
    const idx=devLabel(c.cuda_index, c.nvml_index);
    h+='<div class="cardsimple" data-key="cs_'+i+'">'
      +'<div class="cs-h"><span class="cs-n">'+esc(c.name||('card '+i))
      +(idx?' <span class="muted" style="font-weight:400">['+esc(idx)+']</span>':'')+'</span>'
      +'<span class="cs-u">'+(need!=null
          ? esc(fmtGiB(need))+' used of '+esc(fmtGiB(total))
          : 'usage not computed yet')+'</span></div>'
      +'<div class="csbar"><div class="fill'+cls+'" style="width:'+pctNeed.toFixed(1)+'%"></div>'
      +'<div class="cap" style="left:'+pctCap.toFixed(1)+'%" title="budget"></div></div>'
      +'<div class="csrow"><label class="muted" style="font-size:.7rem;min-width:9rem">'
      +'maximum VRAM to use</label>'
      +'<input type="range" id="csb_'+i+'" min="0" max="'+total+'" step="256" value="'+budget+'"'
      +' title="'+esc(tip('card_budget'))+'" oninput="simpleBudgetInput('+i+')">'
      +'<span class="cs-v" id="csv_'+i+'">'+esc(fmtGiB(budget))+' of '+esc(fmtGiB(total))+'</span>'
      +'</div>';
    if(need!=null&&need>budget)
      h+='<div class="reasons" style="font-size:.7rem;margin-top:.2rem">'
        +'needs '+esc(fmtGiB(need-budget))+' more than this budget allows.</div>';
    h+='</div>';
  });
  setHTML(box,h);
}
function renderSimpleVerdict(d){
  const box=$('simple_verdict'); if(!box||!d) return;
  if(d.valid===false){
    setHTML(box,'<div class="verdict nofit">REJECTED</div><ul class="reasons">'
      +(d.reasons||[]).map(x=>'<li>'+esc(x)+'</li>').join('')+'</ul>');
    return;
  }
  const cap=d.capacity, off=d.offload;
  let h;
  if(d.fits) h='<div class="verdict fit">FITS IN VRAM &check;</div>';
  else if(off&&off.status==='ram_offload')
    h='<div class="verdict offload">FITS WITH ~'+off.offloaded_gib.toFixed(1)
      +' GiB ON HOST RAM &mdash; slower (PCIe-bound)</div>';
  else h='<div class="verdict nofit">DOES NOT FIT</div>';
  if(cap&&cap.max_context_tokens)
    h+='<div class="adv">context that fits: <b>'
      +Math.round(cap.max_context_tokens).toLocaleString()+'</b> tokens</div>';
  if(!d.fits&&(d.infeasible_reasons||[]).length)
    h+='<ul class="reasons">'
      +d.infeasible_reasons.map(x=>'<li>'+esc(x)+'</li>').join('')+'</ul>';
  h+='<div class="legend">Switch to <b>expert</b> for the per-card breakdown '
    +'and every dimension this fork exposes.</div>';
  setHTML(box,h);
}
// ---- tuning objective template -------------------------------------------
// Sets --rank-perf-tune and nothing else. The point of a template here is to
// move the starting point, not to fence the controls in: everything stays
// adjustable afterwards, and a value that genuinely cannot work is refused
// by the planner with a reason rather than by a slider that stops early.
const TUNE_LABELS={both:'balanced', maxkv:'max KV', dec:'max decode', enc:'max prefill'};
function applyTune(mode){
  window._flagSettings=window._flagSettings||{};
  if(mode==='both') delete window._flagSettings.rank_perf_tune;
  else window._flagSettings.rank_perf_tune=mode;
  const el=$('fl_rank_perf_tune');
  if(el) el.value=(mode==='both'?'':mode);
  markTune();
  markPresetDrift(); updateSectionSummaries(); scheduleRecompute();
}
function markTune(){
  const cur=(window._flagSettings||{}).rank_perf_tune||'both';
  for(const k of Object.keys(TUNE_LABELS)){
    const b=$('tune_'+k); if(!b) continue;
    b.className='mini'+(k===cur?'':' secondary');
  }
  const note=$('tune_note');
  if(note) note.innerHTML='objective: <b>'+esc(TUNE_LABELS[cur]||cur)+'</b>'
    +' &mdash; sets --rank-perf-tune only; every control below stays free to '
    +'move past it. This is not a second preset: a preset fills every row, '
    +'an objective moves one.';
}
function runnerFlags(){
  const s=window._flagSettings||{};
  const incl=CARDS.filter(c=>c.include);
  const f={
    tp_size: (s.tp_size? parseInt(s.tp_size):null) || (incl.length||1),
    rank_gpu_id: (s.rank_gpu_id!=null && s.rank_gpu_id!=='')? s.rank_gpu_id:null,
    rank_tp_ratio: (s.rank_tp_ratio!=null && s.rank_tp_ratio!=='')? s.rank_tp_ratio:null,
    kv_cache_dtype: s.kv_cache_dtype||'auto',
    context_length: $('sv_ctx').value? parseInt($('sv_ctx').value):null,
    max_running_requests: $('max_running_requests').value? parseInt($('max_running_requests').value):null,
    include_vision: $('include_vision').checked,
  };
  // Card inventory for the placement view, keyed in CUDA space (the space
  // rank_gpu_id / the default rank i -> gpu i identity live in). Detected
  // cards use their bridged cuda_index; undetected/virtual cards fill the
  // lowest unused keys (offline planning, where indices are abstract).
  const ct={},cn={};
  const used={};
  const incl2=CARDS.filter(c=>c.include);
  incl2.forEach(c=>{ if(c.cuda_index!=null) used[c.cuda_index]=1; });
  let nextFree=0;
  incl2.forEach(c=>{
    let k=c.cuda_index;
    if(k==null){ while(used[nextFree]) nextFree++; k=nextFree; used[k]=1; }
    ct[k]=c.total_mib; cn[k]=c.name;
  });
  f.card_total_mib=ct; f.card_name=cn;
  for(const k of ['rank_mlp_ratio','rank_moe_ratio','rank_vocab_ratio','dcp_size',
    'rank_gpu_memory_mib','rank_kv_ratio','speculative_algorithm',
    'speculative_num_draft_tokens','speculative_num_steps',
    'speculative_adaptive','speculative_draft_model_path',
    'moe_resident_expert_fraction'])
    if(s[k]!=null && s[k]!=='') f[k]=s[k];
  if(s.SGLANG_UNEVEN_TOKEN_VECTOR!=null && s.SGLANG_UNEVEN_TOKEN_VECTOR!=='')
    f.kv_token_vector=s.SGLANG_UNEVEN_TOKEN_VECTOR;
  return f;
}
// A config is "fork-shaped" when it uses the uneven/fork levers -- exactly
// then the prospective view shows a SECOND, plain normal-TP configuration
// NEXT TO it (side by side, same granular card view, neutral wording).
function flagsAreForkish(f){
  const rtr=f.rank_tp_ratio;
  const nonuniform = !!rtr && (typeof rtr==='string'
    ? true
    : new Set(String(rtr).split(',').map(x=>x.trim()).filter(x=>x)).size>1);
  let dup=false;
  if(f.rank_gpu_id){
    const a=String(f.rank_gpu_id).split(',').map(x=>x.trim()).filter(x=>x);
    dup=new Set(a).size!==a.length;
  }
  return !!(nonuniform || dup || f.dcp_size || f.kv_token_vector || f.rank_kv_ratio);
}
async function refreshRunnerPlacement(){
  const model=$('model').value.trim(); if(!model) return;
  const rp=$('runner_placement');
  try{
    const flags=runnerFlags();
    const wantStock=flagsAreForkish(flags);
    // Keyed 'placement_runner': the previous call is aborted, so the panel
    // always shows the answer to the CURRENT flag set. Dimming while it is
    // out keeps the previous numbers readable instead of blanking them.
    stale(rp,true);
    const d=await api('/api/placement',{key:'placement_runner',
      body:{model, gguf_choice: ($('gguf_pick').style.display!=='none'?$('gguf_choice').value:null),
            flags, stock_compare: wantStock}});
    let html;
    if(d.ok){
      const mainH=renderPlacement(d.placement);
      const stock=d.stock;
      if(stock){
        // Side-by-side: the planned (fork) config and a plain normal-TP one,
        // each with its own honest numbers. Neutral wording throughout: when
        // stock cannot express any tp here, the divisibility rule is stated
        // as a fact ("stock requires tp % kv == 0 ..."), not a verdict.
        let stockH;
        if(stock.legal && stock.placement)
          stockH='<div class="sxs-h">plain normal-TP (stock, tp='+stock.tp+')</div>'
            +'<div class="muted" style="font-size:.68rem;margin-bottom:.2rem">'+esc(stock.note||'')+'</div>'
            +renderPlacement(stock.placement);
        else
          stockH='<div class="sxs-h">plain normal-TP (stock)</div>'
            +'<div class="muted" style="font-size:.7rem">'+esc(stock.note||'')+'</div>';
        html='<div class="sxs"><div><div class="sxs-h">planned config</div>'+mainH+'</div>'
          +'<div>'+stockH+'</div></div>';
      } else html=mainH;
    } else html='<span class="reasons">'+esc(d.error)+'</span>';
    // ONE renderer, mirrored: the right-column overview AND the GPU
    // offload/split section (config + its effect live together).
    setHTML(rp, html);
    setHTML($('gpu_placement'), html);
  }catch(e){
    if(apiAborted(e)) return;
    setHTML(rp,'<span class="reasons">'+esc(apiError(e))+'</span>');
  } finally { stale(rp,false); }
}
// ONE renderer for a placement answer, whichever call produced it: the
// right-column overview AND the GPU offload/split section, always in step.
function applyPlacementResult(d){
  let html;
  if(d && d.ok){
    const mainH=renderPlacement(d.placement);
    const stock=d.stock;
    if(stock){
      let stockH;
      if(stock.legal && stock.placement)
        stockH='<div class="sxs-h">plain normal-TP (stock, tp='+stock.tp+')</div>'
          +'<div class="muted" style="font-size:.68rem;margin-bottom:.2rem">'+esc(stock.note||'')+'</div>'
          +renderPlacement(stock.placement);
      else
        stockH='<div class="sxs-h">plain normal-TP (stock)</div>'
          +'<div class="muted" style="font-size:.7rem">'+esc(stock.note||'')+'</div>';
      html='<div class="sxs"><div><div class="sxs-h">planned config</div>'+mainH+'</div>'
        +'<div>'+stockH+'</div></div>';
    } else html=mainH;
  } else html='<span class="reasons">'+esc((d&&d.error)||'placement unavailable')+'</span>';
  setHTML($('runner_placement'), html);
  setHTML($('gpu_placement'), html);
}
async function loadConfigProfiles(){
  const model=$('model').value.trim();
  try{
    const q=model? ('?model='+encodeURIComponent(model)):'';
    const d=await api('/api/config_profiles'+q,{key:'profiles'});
    if(!d.ok){ setHTML($('profile_pick'),'<span class="reasons">'+esc(d.error||'error')+'</span>'); return; }
    // Draft-model candidates + MTP-head fact for the Speculative section's
    // selector and no-MTP hint (matched server-side per selected model).
    window._draftCandidates=d.draft_candidates||[];
    window._modelHasMtp=(d.model_has_mtp===undefined?null:d.model_has_mtp);
    renderDraftPick(); updateSpecDraftHint();
    const nGen=(d.generated||[]).length;
    window._profiles=(d.generated||[]).concat(d.saved||[]);
    // LM-Studio-style preset DROPDOWN: generated presets first, user-saved
    // after (marked); picking one applies it and fills the settings rows.
    let h='<label for="profile_select">preset</label>'
      +'<select id="profile_select" onchange="applyProfileSel()">'
      +'<option value="">&mdash; apply a preset &mdash;</option>'
      +window._profiles.map((p,i)=>'<option value="'+i+'" title="'
        +esc((p.info||[]).join(' | '))+'">'+esc(p.name)
        +(i>=nGen?' (saved)':'')+'</option>').join('')
      +'</select>';
    setHTML($('profile_pick'), h);
  }catch(e){
    if(apiAborted(e)) return;
    setHTML($('profile_pick'),'<span class="reasons">'+esc(apiError(e))+'</span>');
  }
}
// Mirror a profile's serving-identity value into its ONE authoritative field
// (the SERVING group / the model selector) -- the user can still override it
// afterwards, and the form always wins on launch (identity merge).
function applyServingValue(id, v){
  if(v==null||v==='') return;
  const map={model_path:'model', served_model_name:'sv_served', host:'sv_host',
             port:'sv_port', context_length:'sv_ctx',
             max_running_requests:'max_running_requests'};
  const el=$(map[id]); if(!el) return;
  el.value=String(v);
  if(id==='model_path') onModelChange();
}
function applyProfileSel(){
  const v=$('profile_select').value;
  if(v!=='') applyProfile(parseInt(v));
}
function applyProfile(i){
  const p=window._profiles[i]; if(!p) return;
  window._flagSettings={};
  // Reset every rendered flag control first so a previously-applied preset's
  // leftovers don't linger in rows this preset does not set.
  for(const id of Object.keys(window._flagSection)){
    const el=$('fl_'+id); if(!el) continue;
    if(el.type==='checkbox') el.checked=false; else el.value='';
  }
  for(const id of Object.keys(p.settings)){
    const v=p.settings[id];
    if(SERVING_OWNED[id]){ applyServingValue(id, v); continue; }
    // Everything else lands in the ONE flag surface. "auto" /
    // "auto-performance" are valid rank_tp_ratio values (NVML-derived
    // budgets) and MUST land in the field -- dropping them leaves
    // rank_gpu_id orphaned, which the fork rejects at launch.
    const el=$('fl_'+id);
    if(el){ if(el.type==='checkbox') el.checked=!!v; else el.value=(v==null?'':String(Array.isArray(v)?v.join(','):v)); }
    if(v!=null && v!=='' && v!==false){ const spec=_flagSpec(id);
      if(!spec || v!==spec.default) window._flagSettings[id]=v; }
  }
  // The profile's EXACT launch surface: env (applied to the server process)
  // and argv (used verbatim on Launch until a flag is edited manually).
  window._profileEnv=p.launch_env||p.env||null;
  window._profileArgv=p.argv||null;
  window._profileDirty=false;
  renderProfileLaunch();
  // A preset that lands silently is indistinguishable from a dead control.
  // Report what it actually wrote, and -- the case that made it look broken --
  // say plainly when nothing downstream can move because no model is picked.
  const nSet=Object.keys(window._flagSettings||{}).length;
  const noModel=!$('model').value.trim();
  $('profile_msg').innerHTML='<span class="muted">applied <b>'+esc(p.name)+'</b> &middot; '
    +nSet+' setting'+(nSet===1?'':'s')+' written to the rows below'
    +((p.info&&p.info.length)?' &middot; '+esc(p.info.join(' | ')):'')+'</span>'
    +(noModel?'<div class="rev-error">No model selected, so the plan and the '
      +'per-card bars cannot be recomputed yet. Pick a model above &mdash; the '
      +'preset stays applied.</div>':'');
  // Sync the slider pairs to the applied values, then snapshot for the
  // changed-from-preset markers (the snapshot IS the applied state).
  const sl=$('sv_ctx_slider');
  sl.value=Math.min(parseInt($('sv_ctx').value)||8192, parseInt(sl.max));
  const mv=parseInt($('max_running_requests').value);
  if(mv){ setMrrCap(Math.max(window._mrrCap||64, mv)); $('mrr_slider').value=mv; }
  // Preset prefill reaches the co-location controls too: the profile's
  // tp_size / rank_gpu_id map back onto the tp mirror + per-card steppers
  // (reverse-populated, never redistributed).
  window._coloRedistribute=false;
  updateGpuPick(); updateColoUI(); renderDraftPick(); updateSpecDraftHint();
  window._presetBase=presetSnapshot();
  markPresetDrift(); updateSectionSummaries(); markTune();
  // One consistent recompute for the whole applied preset.
  scheduleRecompute();
}
// DISPLAY the applied profile's launch env + exact argv (not just the CLI
// flags): a launched profile must match the reference command exactly, and
// the env half (SGLANG_UNEVEN_* pair, LD_LIBRARY_PATH, PYTHONPATH) is as
// load-bearing as the argv half.
function renderProfileLaunch(){
  const box=$('profile_env_box'); if(!box) return;
  const env=window._profileEnv, argv=window._profileArgv;
  if(!env && !argv && !window._profileDirty){ box.innerHTML=''; return; }
  let h='';
  if(env && Object.keys(env).length){
    h+='<div class="muted"><b>launch env</b> (applied to the server process on Launch):</div>'
      +'<pre style="margin:.2rem 0">'+Object.keys(env).map(k=>esc(k+'='+env[k])).join('\n')+'</pre>';
  }
  if(argv){
    h+='<details><summary class="muted" style="cursor:pointer">exact profile argv (used verbatim on Launch)</summary>'
      +'<pre style="margin:.2rem 0">'+esc('python -m sglang.launch_server '+argv.join(' '))+'</pre></details>';
  } else if(window._profileDirty){
    h+='<div class="muted" style="color:#e3a008">flags edited after applying the profile — '
      +'the exact profile argv is no longer used; Launch builds the command from the form fields '
      +'(the profile env above still applies).</div>';
  }
  box.innerHTML=h;
}
async function saveProfile(){
  const name=$('profile_save_name').value.trim();
  if(!name){ $('profile_msg').innerHTML='<span class="reasons">enter a name</span>'; return; }
  try{
    const r=await fetch('/api/config_profiles',{method:'POST',
      body:JSON.stringify({name, settings:collectFlagSettings(), kind:'custom'})});
    const d=await r.json();
    $('profile_msg').innerHTML=d.ok?'<span class="fitc">saved '+esc(name)+'</span>'
      :'<span class="reasons">'+esc(d.error)+'</span>';
    if(d.ok) loadConfigProfiles();
  }catch(e){ $('profile_msg').innerHTML='<span class="reasons">'+esc(''+e)+'</span>'; }
}

async function loadProfiles() {
  window._profLoaded = true;
  const r = await fetch('/api/cards'); const d = await r.json();
  $('mx_cards').innerHTML = '<b>library:</b> ' +
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
    noteCudaMap(d.cards, null);
    let h = '<table class="mx"><tr><th>cuda/nvml</th><th>card</th><th>W now</th><th>limit</th>'
          + '<th>%TDP</th><th>SM/MEM MHz</th><th>°C</th><th>state</th></tr>';
    for (const c of d.cards) {
      const tag = c.oc_uv==='stock' ? c.oc_uv
                : '<b style="color:#e3a008">'+esc(c.oc_uv)+'</b>';
      h += '<tr><td>'+(devLabel(c.cuda_index, c.nvml_index)||c.nvml_index)+'</td><td style="text-align:left">'+esc(c.name)+'</td>'
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

// The #149 live widget that used to sit here is gone. It addressed
// live_target / live_res / live_btn / live_out, none of which have existed
// in the markup since the landing strip replaced it, so its interval fetched
// nothing and its output went nowhere. The landing top strip (stripTile,
// LAND_POLL_MS) is the live rate view. POST /api/live went with it.

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

// ---- Models tab ----
async function loadModels() {
  $('models_out').innerHTML = 'scanning…';
  try {
    const r = await fetch('/api/models'); const d = await r.json();
    if (!d.ok) { $('models_out').innerHTML = '<span class="reasons">'+esc(d.error)+'</span>'; return; }
    window._models = d.models;
    window._modelRoots = d.roots;
    renderModelOptions();
  } catch(e) { $('models_out').innerHTML = '<span class="reasons">'+esc(''+e)+'</span>'; }
}
// Search-as-you-type over the discovered models (name / format / quant); the
// option value stays the ORIGINAL index into window._models, so filtering
// never changes what pickFromDropdown() resolves.
function renderModelOptions() {
  const models = window._models; if (!models) return;
  const q = ($('model_search').value||'').trim().toLowerCase();
  const cur = $('model').value.trim();
  let shown = 0;
  const opts = models.map((m,i)=>{
    const quant = (m.quant_method && m.quant_method!=='None') ? m.quant_method : '';
    if (q && (m.name+' '+m.format+' '+quant).toLowerCase().indexOf(q)<0) return '';
    shown++;
    const qb = quant ? ' · '+esc(quant) : '';
    const vv = (m.gguf_variants && m.gguf_variants.length>1) ? ' · '+m.gguf_variants.length+' quants' : '';
    const err = m.error ? ' · (err)' : '';
    return '<option value="'+i+'"'+(cur && m.path===cur?' selected':'')+'>'
      +esc(m.name)+'  ['+esc(m.format)+qb+' · '+m.size_gib+'G]'+vv+err+'</option>';
  }).join('');
  $('models_out').innerHTML =
    '<select id="model_select" style="width:100%;max-width:100%" onchange="pickFromDropdown()">'
    + '<option value="">— select a model —</option>' + opts + '</select>'
    + '<div class="muted" style="margin-top:.3rem">'+shown+'/'+models.length
    + ' models · '+esc((window._modelRoots||[]).join(', '))+'</div>';
}
function pickFromDropdown() {
  const i = $('model_select').value;
  if (i==='') { $('gguf_pick').style.display='none'; $('model_info').innerHTML='';
    window._modelMeta=null; return; }
  const m = window._models[+i];
  // The dropdown FILLS the one model field -- there is no second model input.
  $('model').value = m.path;
  window._modelMeta = {format: m.format, variants: m.gguf_variants||[]};
  if (m.gguf_variants && m.gguf_variants.length>1) {
    setHTML($('gguf_choice'), m.gguf_variants
      .map(v=>'<option value="'+esc(v.filename)+'">'+esc(v.quant)+' ('+v.size_gib+'G)</option>').join(''));
    $('gguf_pick').style.display='';
  } else {
    $('gguf_pick').style.display='none';
  }
  const info = [];
  if (m.quant_method && m.quant_method!=='None') info.push('quant '+esc(m.quant_method));
  info.push(m.size_gib+' GiB'); if (m.vision) info.push('vision');
  if (m.error) info.push('<span class="nofitc" title="'+esc(m.error)+'">error</span>');
  $('model_info').innerHTML = '<b>'+esc(m.name)+'</b> — '+info.join(' · ')+'<br>'+esc(m.path);
  onRunnerModel();
}
function serverSettings() {
  // Launch reads model identity from the ONE selector, serving identity from
  // the SERVING group, and every tuning flag from the ONE flag surface.
  const ms = modelState();
  const s = window._flagSettings || {};
  const sval = k => (s[k]!=null && s[k]!=='') ? s[k] : null;
  return {
    model_path: ms.path,
    format: ms.format,
    gguf_variant: ms.gguf_variant,
    served_model_name: $('sv_served').value.trim() || 'model',
    host: $('sv_host').value.trim() || '127.0.0.1',
    port: parseInt($('sv_port').value)||30000,
    context_length: parseInt($('sv_ctx').value)||8192,
    max_running_requests: $('max_running_requests').value.trim() || null,
    tp_size: parseInt(s.tp_size)||1,
    rank_gpu_id: sval('rank_gpu_id'),
    rank_tp_ratio: sval('rank_tp_ratio'),
    rank_gpu_memory_mib: sval('rank_gpu_memory_mib'),
    kv_cache_dtype: s.kv_cache_dtype || 'auto',
    spec_mode: s.speculative_algorithm ? (s.speculative_adaptive ? 'adaptive' : 'mtp') : 'off',
    chat_template: sval('chat_template'),
    tool_call_parser: sval('tool_call_parser'),
    reasoning_parser: sval('reasoning_parser'),
    vision: $('include_vision').checked,
  };
}
// Sticky-bar status chip: stopped (grey) / loading (amber, while booting) /
// ready (green) / error (red). Fed by every status render below.
function updateStatusChip(state) {
  const s=String(state||'stopped');
  let cls='chip', txt=s;
  if(s==='ready'){ cls='chip ready'; txt='ready'; }
  else if(s==='booting'){ cls='chip loading'; txt='loading…'; }
  else if(s==='error'){ cls='chip error'; txt='error'; }
  // Both modes show the same state; simple mode simply has fewer buttons
  // around it.
  for(const id of ['status_chip','simple_status_chip']){
    const chip=$(id); if(!chip) continue;
    chip.className=cls; chip.textContent=txt;
  }
}
// While the managed server boots, poll status until it settles so the chip
// flips loading -> ready/error without manual refreshes (bounded, 2s tick).
function pollStatusUntilSettled(){
  if(window._bootPoll) clearInterval(window._bootPoll);
  let left=90;  // up to 3 minutes of boot-watching
  window._bootPoll=setInterval(async ()=>{
    if(--left<0){ clearInterval(window._bootPoll); window._bootPoll=null; return; }
    try{
      const d=await api('/api/server_status',{key:'boot_status', timeout:1800});
      setHTML($('sv_out'), renderServerStatus(d));
      const st=(d.status&&d.status.state)||d.state;
      if(st!=='booting'){ clearInterval(window._bootPoll); window._bootPoll=null; }
    }catch(e){}
  }, 2000);
}
function renderServerStatus(d) {
  if (!d) return '';
  const st=(d.status&&d.status.state)||d.state;
  updateStatusChip(st);
  // Open the boot log ONCE when the boot starts or fails. openDetails records
  // that it was opened for the reader, so re-collapsing it sticks instead of
  // being undone by the next 2 s poll.
  if(st==='error'||st==='booting') openDetails($('boot_log'));
  if (!d.ok && d.error) {
    let h = '<div class="reasons">'+esc(d.error)+'</div>';
    if (d.busy) h += '<div class="muted">restart is guarded while a live job is in-flight.</div>';
    if (d.status) h += renderStatusBox(d.status);
    return h;
  }
  let h = '';
  if (d.env_applied && Object.keys(d.env_applied).length)
    h += '<div class="legend">env: <code>'
      +esc(Object.keys(d.env_applied).map(k=>k+'='+d.env_applied[k]).join(' '))+'</code></div>';
  if (d.launch_command) h += '<div class="legend">launch: <code>'+esc(d.launch_command.join(' '))+'</code></div>';
  h += renderStatusBox(d.status || d);
  return h;
}
function renderStatusBox(s) {
  if (!s) return '';
  let h = '<div style="margin-top:.4rem"><b>state:</b> '+esc(s.state||'?')
        + (s.pid?' pid '+s.pid:'') + (s.busy?' <b style="color:#e3a008">BUSY</b>':'')
        + (s.port?' :'+s.port:'') + (s.uptime_s?' up '+s.uptime_s.toFixed(0)+'s':'') + '</div>';
  if (s.error) h += '<div class="reasons">'+esc(s.error)+'</div>';
  if (s.log_tail) h += '<pre style="white-space:pre-wrap;font-size:.66rem;max-height:200px;overflow:auto;background:#0d1117;padding:.4rem;border-radius:6px">'+esc(s.log_tail)+'</pre>';
  return h;
}
async function serverPost(path, body) {
  $('sv_out').innerHTML = 'working…';
  try {
    const r = await fetch(path, {method:'POST', body: JSON.stringify(body||{})});
    const d = await r.json();
    $('sv_out').innerHTML = renderServerStatus(d);
  } catch(e) { $('sv_out').innerHTML = '<span class="reasons">'+esc(''+e)+'</span>'; }
}
function launchBody() {
  // Launch settings + the applied profile's env (always) and exact argv
  // (until a manual flag edit invalidated it).
  const body = serverSettings();
  if (window._profileEnv && Object.keys(window._profileEnv).length) body.env = window._profileEnv;
  if (window._profileArgv) body.profile_argv = window._profileArgv;
  return body;
}
// Co-location gate for Launch/Restart: refuse while the per-card rank
// assignment does not sum to tp (the section shows the same inline error).
function launchGate() {
  const coloErr = coloBlockError();
  if (!coloErr) return true;
  $('sv_out').innerHTML = '<div class="reasons">' + esc(coloErr) + '</div>';
  updateColoUI();
  return false;
}
function serverStart() {
  if (!launchGate()) return;
  updateStatusChip('booting');
  serverPost('/api/server_start', launchBody());
  pollStatusUntilSettled();
}
function serverRestart() {
  if (!launchGate()) return;
  if (!confirm('Restart REPLACES the single managed instance (stops the running model). Continue?')) return;
  updateStatusChip('booting');
  serverPost('/api/server_restart', launchBody());
  pollStatusUntilSettled();
}
function serverStop() { serverPost('/api/server_stop', {}); }
async function refreshServerStatus() {
  try {
    const r = await fetch('/api/server_status'); const d = await r.json();
    $('sv_out').innerHTML = renderServerStatus(d);
  } catch(e) { $('sv_out').innerHTML = '<span class="reasons">'+esc(''+e)+'</span>'; }
}
async function dlTargets() {
  const repo = $('dl_repo').value.trim(); if (!repo) return;
  try {
    const r = await fetch('/api/download_targets', {method:'POST',
      body: JSON.stringify({repo_id:repo, root:$('dl_root').value.trim()||null})});
    const d = await r.json();
    window._dlTargets = d;
    let note = 'root <b>'+esc(d.root)+'</b> — '+(d.writable?'<span class="fitc">writable</span>':'<span class="nofitc">'+esc(d.note||'read-only')+'</span>');
    $('dl_btn').disabled = !d.writable;
    if (d.gguf_variants && d.gguf_variants.length) {
      $('dl_variants').style.display='';
      setHTML($('dl_quant'), d.gguf_variants.map(v=>'<option value="'+esc(v.quant)+'">'+esc(v.quant)+' ('+esc(v.filename)+')</option>').join(''));
    } else { $('dl_variants').style.display='none'; }
    if (d.repo_error) note += '<br><span class="reasons">'+esc(d.repo_error)+'</span>';
    $('dl_out').innerHTML = note;
  } catch(e) { $('dl_out').innerHTML = '<span class="reasons">'+esc(''+e)+'</span>'; }
}
async function dlPreview() {
  const t = window._dlTargets;
  if (!t || !t.writable) { $('dl_out').innerHTML = '<span class="reasons">model root not writable — remount rw.</span>'; return; }
  const quant = ($('dl_variants').style.display!=='none') ? $('dl_quant').value : null;
  if (!confirm('Download '+t.repo_id+(quant?' ['+quant+']':'')+' from Hugging Face into '+t.root+'? This fetches from an EXTERNAL service.')) return;
  $('dl_out').innerHTML = 'downloading… (external HF fetch, may take a while)';
  try {
    const r = await fetch('/api/model_download', {method:'POST',
      body: JSON.stringify({repo_id:t.repo_id, root:t.root, quant:quant})});
    const d = await r.json();
    $('dl_out').innerHTML = d.ok ? '<span class="fitc">downloaded to '+esc(d.path)+'</span>'
                                 : '<span class="reasons">'+esc(d.error)+'</span>';
    if (d.ok) loadModels();
  } catch(e) { $('dl_out').innerHTML = '<span class="reasons">'+esc(''+e)+'</span>'; }
}

// ---- power measurement (Energy tab) ----
async function loadPowerProfile() {
  try {
    const r = await fetch('/api/power_profile'); const d = await r.json();
    if (d.ok && d.loaded) $('pw_out').innerHTML = 'persisted profile ('+esc(d.driver||'?')+'):'+powerTable(d.cards, []);
  } catch(e) {}
}
function powerTable(cards, skipped) {
  let h = '<table class="mx"><tr><th>card</th><th>arch</th><th>idle W</th><th>membw W</th>'
        + '<th>gemm W</th><th>GB/s</th><th>TFLOP/s</th></tr>';
  for (const c of cards) {
    h += '<tr><td style="text-align:left" title="'+esc(c.uuid)+'">'+esc(c.name)+'</td>'
       + '<td>'+esc(c.arch)+'</td><td>'+c.p_idle_w+'</td><td>'+c.p_membw_w+'</td>'
       + '<td>'+c.p_gemm_w+'</td><td>'+c.membw_gbs+'</td><td>'+c.gemm_tflops+'</td></tr>';
  }
  h += '</table>';
  for (const s of (skipped||[]))
    h += '<div class="muted">SKIPPED '+esc(s.name||'?')+' ('+esc(s.reason)+'): '+esc(s.detail||'')+'</div>';
  return h;
}
async function measurePower() {
  if (!confirm('This briefly LOADS each free GPU with short micro-benchmarks to measure board power. Busy cards are skipped automatically. Continue?')) return;
  $('pw_btn').disabled = true;
  $('pw_out').innerHTML = 'measuring (each card in its own subprocess)…';
  try {
    const r = await fetch('/api/measure_power', {method:'POST', body: '{}'});
    const d = await r.json();
    if (!d.ok) { $('pw_out').innerHTML = '<span class="reasons">'+esc(d.error)+'</span>'; }
    else $('pw_out').innerHTML = 'measured '+esc(d.created||'')+' ('+esc(d.driver||'?')+'):'+powerTable(d.cards, d.skipped);
  } catch(e) { $('pw_out').innerHTML = '<span class="reasons">'+esc(''+e)+'</span>'; }
  finally { $('pw_btn').disabled = false; }
}

// ---- Quality tab ----
function verdictClass(v) {
  return v==='correct' ? 'fit' : (v==='wrong-position'||v==='broken' ? 'nofit' : 'offload');
}
async function autofillQuality() {
  // Use the currently running managed server (endpoint + served model) so the
  // Quality run targets whatever is loaded, without retyping it.
  // Autofill FILLS; it does not overwrite. Switching to this tab used to
  // stamp over whatever endpoint or model name had been typed by hand.
  try {
    const d = await api('/api/server_status',{key:'quality_autofill', timeout:3000});
    const s = (d && d.status) || {};
    if (s.state === 'ready' && s.port) {
      if (!$('q_endpoint').value.trim()) $('q_endpoint').value = '127.0.0.1:' + s.port;
      if (s.served_model_name && !$('q_model').value.trim())
        $('q_model').value = s.served_model_name;
    }
  } catch(e) {}
}
// The chess suite follows the same line as the benchmark window: a run is
// visibly RUNNING until it is finished, and the outcome is a table of
// measure and value rather than a paragraph. The verdict stays a verdict --
// it is a judgement, not a measurement, and reads as one.
function qualityTableHtml(d){
  const tk=d.tokens||{};
  const rows=[];
  rows.push({k:'verdict', v:d.verdict, cls:verdictClass(d.verdict)});
  if(d.representation) rows.push({k:'representation', v:d.representation});
  if(tk.prompt!=null) rows.push({k:'prompt tokens', v:tk.prompt});
  if(tk.completion!=null) rows.push({k:'completion tokens', v:tk.completion});
  if(tk.total!=null) rows.push({k:'total tokens', v:tk.total});
  if(d.pieces_correct!=null) rows.push({k:'pieces correct', v:d.pieces_correct?'yes':'no'});
  if(d.highlight_ok!=null) rows.push({k:'move highlighted', v:d.highlight_ok?'yes':'no'});
  return '<table class="mx"><tr><th style="text-align:left">measure</th>'
    +'<th style="text-align:left">value</th></tr>'
    +rows.map(r=>'<tr data-key="q_'+r.k.replace(/ /g,'_')+'">'
      +'<td style="text-align:left">'+esc(r.k)+'</td>'
      +'<td style="text-align:left" class="'+(r.cls||'')+'"><b>'+esc(String(r.v))+'</b></td></tr>'
      ).join('')+'</table>';
}
async function qualityRun() {
  if (!$('q_endpoint').value.trim() || !$('q_model').value.trim()) await autofillQuality();
  const endpoint = $('q_endpoint').value.trim();
  const model = $('q_model').value.trim();
  if (!endpoint || !model) { setHTML($('q_status'), '<span class="reasons">endpoint + model required</span>'); return; }
  $('q_btn').disabled = true;
  setHTML($('q_status'), '<span class="chip loading">running</span> calling the model (backend-side)…');
  setHTML($('q_result'), '<span class="muted">waiting for the model…</span>');
  const budget = $('q_budget').value.trim();
  try {
    // Not routed through api(): a quality run is a deliberate act and must
    // not be aborted because something else asked a newer question.
    const r = await fetch('/api/quality_run', {method:'POST', body: JSON.stringify({
      endpoint, model, thinking: $('q_think').checked,
      thinking_budget: budget!==''?parseInt(budget):null})});
    const d = await r.json();
    if (!d.ok) {
      setHTML($('q_status'), '<span class="reasons">'+esc(d.error)+'</span>');
      setHTML($('q_result'), '');
      return;
    }
    window._lastQuality = d;
    $('q_svg').innerHTML = d.svg ? d.svg : '<span class="muted">no SVG extracted</span>';
    let h = qualityTableHtml(d);
    if (d.report) h += '<div class="legend" style="margin-top:.4rem">'+esc(d.report)+'</div>';
    if (d.offer_download && d.svg)
      h += '<div class="actions" style="margin-top:.4rem"><button class="mini secondary" onclick="dlSvg()">download raw SVG</button></div>';
    else if (d.offer_download && d.raw)
      h += '<div class="actions" style="margin-top:.4rem"><button class="mini secondary" onclick="dlRaw()">download raw answer</button></div>';
    setHTML($('q_result'), h);
    setHTML($('q_status'), '<span class="chip ready">finished</span>');
    if ($('q_save').checked) await saveShot(d);
  } catch(e) { setHTML($('q_status'), '<span class="reasons">'+esc(''+e)+'</span>'); }
  finally { $('q_btn').disabled = false; }
}
function dlSvg() { const d=window._lastQuality; if(d&&d.svg) dlBlob(d.svg, 'chess.svg', 'image/svg+xml'); }
function dlRaw() { const d=window._lastQuality; if(d&&d.raw) dlBlob(d.raw, 'chess_answer.txt', 'text/plain'); }
function dlBlob(text, name, mime) {
  const b = new Blob([text], {type: mime}); const u = URL.createObjectURL(b);
  const a = document.createElement('a'); a.href = u; a.download = name; a.click(); URL.revokeObjectURL(u);
}
async function saveShot(d) {
  try {
    await fetch('/api/quality_save', {method:'POST', body: JSON.stringify({
      save: true, model: $('q_model').value.trim(), quant: $('q_quant').value.trim(),
      verdict: d.verdict, tokens: d.tokens, svg: d.svg, report: d.report,
      prompt: 'chess', config: {thinking: $('q_think').checked, budget: $('q_budget').value.trim()||null}})});
    loadShots();
  } catch(e) {}
}
async function loadShots() {
  try {
    const r = await fetch('/api/quality_shots'); const d = await r.json();
    window._shots = (d.ok && d.shots) ? d.shots : [];
    const sl = $('q_slider'); sl.max = Math.max(0, window._shots.length-1);
    sl.value = sl.max;
    showShot();
  } catch(e) {}
}
function showShot() {
  const shots = window._shots || [];
  if (!shots.length) { $('q_shot').innerHTML = 'no saved shots.'; $('q_slide_lbl').textContent=''; return; }
  const i = Math.min(parseInt($('q_slider').value)||0, shots.length-1);
  const s = shots[i];
  $('q_slide_lbl').textContent = '('+(i+1)+'/'+shots.length+')';
  const tk = s.tokens||{};
  let h = '<div class="legend">'+esc(s.ts||'')+' — '+esc(s.model||'?')+' '+esc(s.quant||'')+'</div>';
  h += '<div class="verdict '+verdictClass(s.verdict)+'" style="font-size:.8rem">'+esc(s.verdict||'?')+'</div>';
  h += '<div class="legend">tokens total '+(tk.total??'?')+'</div>';
  if (s.svg) h += '<div style="max-width:100%;overflow:auto">'+s.svg+'</div>';
  $('q_shot').innerHTML = h;
}

// ---- Benchmark tab (#151) -- backend-driven suite, streamed results ----
window._benchCatalog=null; window._benchResults=null;
function benchInit(){
  const t=window._lastTarget;
  if(t && t.endpoint && !$('bn_endpoint').value.trim())
    $('bn_endpoint').value=t.endpoint.replace(/^https?:\/\//,'');
  benchProbe(true);  // catalog + presets only; nothing probed without endpoint
  benchHistory();
}
function benchUseMonitor(){
  const t=window._lastTarget;
  if(t && t.endpoint) $('bn_endpoint').value=t.endpoint.replace(/^https?:\/\//,'');
}
function benchReGate(){ if($('bn_endpoint').value.trim()) benchProbe(false); }
async function benchProbe(catalogOnly){
  const endpoint=catalogOnly? '' : $('bn_endpoint').value.trim();
  if(!catalogOnly) $('bn_caps').innerHTML='probing (backend-side)&hellip;';
  try{
    const r=await fetch('/api/bench_probe',{method:'POST',
      body:JSON.stringify({endpoint, force:$('bn_force').checked})});
    const d=await r.json();
    if(!d.ok){ $('bn_caps').innerHTML='<span class="reasons">'+esc(d.error||'probe failed')+'</span>'; return; }
    window._benchCatalog=d;
    if(d.capabilities){
      const c=d.capabilities;
      if(c.model && !$('bn_model').value.trim()) $('bn_model').value=c.model;
      const chip=(ok,txt)=>'<span class="pill" style="border-color:'+(ok?'#2ea043':'#7d2a2a')
        +'">'+(ok?'OK ':'NO ')+esc(txt)+'</span> ';
      $('bn_caps').innerHTML=
        chip(c.chat_template_basic,'basic chat template')
        +chip(!!c.tool_parser,'tool parser'+(c.tool_parser?' ('+c.tool_parser+')':''))
        +chip(!!c.reasoning_parser,'reasoning parser'+(c.reasoning_parser?' ('+c.reasoning_parser+')':''))
        +chip(c.spec_decode,'spec decode ('+c.spec_mode+')')
        +(c.max_model_len?'<span class="pill">ctx '+c.max_model_len+'</span>':'')
        +(d.probe_error?'<div class="reasons">'+esc(d.probe_error)+'</div>':'');
    } else if(!catalogOnly){
      $('bn_caps').innerHTML='<span class="reasons">'+esc(d.probe_error||'no endpoint given')+'</span>';
    }
    renderBenchTests();
  }catch(e){ $('bn_caps').innerHTML='<span class="reasons">'+esc(''+e)+'</span>'; }
}
// Which tests are selected. Held in one set rather than read back out of
// the DOM, so a re-gate (which re-renders the buttons) cannot silently drop
// a selection the user made.
window._benchSel=window._benchSel||new Set();
function benchGated(t){ return t.gate_status!=null; }
function renderBenchTests(){
  const d=window._benchCatalog; if(!d) return;
  $('bn_presets').innerHTML='<span class="muted">presets:</span> '
    +Object.keys(d.presets).map(p=>'<button class="mini secondary" onclick="benchPreset(\''+p+'\')">'+esc(p)+'</button>').join(' ')
    +' <button class="mini secondary" onclick="benchSelectAll(true)">all runnable</button>'
    +' <button class="mini secondary" onclick="benchSelectAll(false)">none</button>';
  let h='<div style="display:flex;flex-wrap:wrap;gap:var(--s2)">';
  for(const t of d.tests){
    const gated=benchGated(t);
    const on=!gated&&window._benchSel.has(t.test_id);
    const why=gated?(t.gate_status+': '+(t.gate_reason||''))
                   :(t.expected_fail_note?('expected-fail: '+t.expected_fail_note):'');
    h+='<button type="button" id="bnt_'+t.test_id+'" class="testbtn'
      +(on?' on':'')+(gated?' gated':'')+'"'+(gated?' disabled':'')
      +' onclick="benchToggle('+t.test_id+')" title="'+esc(why)+'">'
      +'<span class="tb-n">'+t.test_id+'</span>'
      +'<span class="tb-l">'+esc(t.label)+'</span>'
      +(t.crash_prone?'<span class="tb-t" title="runs last">crash-prone</span>':'')
      +(t.optional?'<span class="tb-t">optional</span>':'')
      +(gated?'<span class="tb-t warn">'+esc(t.gate_status)+'</span>':'')
      +(!gated&&t.expected_fail_note?'<span class="tb-t warn">expected-fail</span>':'')
      +'</button>';
  }
  h+='</div>';
  $('bn_tests').innerHTML=h;
  benchSelNote();
}
function benchSelNote(){
  const n=window._benchSel.size;
  const el=$('bn_sel_note'); if(!el) return;
  el.textContent=n?(n+' test'+(n===1?'':'s')+' selected'):'nothing selected yet';
}
function benchToggle(id){
  if(window._benchSel.has(id)) window._benchSel.delete(id);
  else window._benchSel.add(id);
  const b=$('bnt_'+id); if(b) b.classList.toggle('on', window._benchSel.has(id));
  benchSelNote();
}
function benchSelectAll(on){
  const d=window._benchCatalog; if(!d) return;
  window._benchSel=new Set();
  if(on) for(const t of d.tests) if(!benchGated(t)) window._benchSel.add(t.test_id);
  renderBenchTests();
}
function benchPreset(p){
  const d=window._benchCatalog; if(!d) return;
  const ids=d.presets[p]||[];
  window._benchSel=new Set();
  for(const t of d.tests)
    if(!benchGated(t) && ids.includes(t.test_id)) window._benchSel.add(t.test_id);
  renderBenchTests();
}
// ---- run history ----------------------------------------------------------
function benchHistoryModel(){
  return $('bn_hist_all').checked ? '' : ($('bn_model').value.trim()||'');
}
async function benchHistory(){
  const m=benchHistoryModel();
  try{
    const d=await (await fetch('/api/bench_history'+(m?('?model='+encodeURIComponent(m)):''))).json();
    if(!d.ok){ setHTML($('bn_history'),'<span class="reasons">'+esc(d.error||'error')+'</span>'); return; }
    $('bn_hist_note').textContent=(d.runs.length||0)+' run'
      +((d.runs.length===1)?'':'s')+' in '+d.root;
    if(!d.runs.length){
      setHTML($('bn_history'), m
        ? 'no runs recorded for this model yet.'
        : 'no runs recorded yet.');
      return;
    }
    let h='<table><tr><th>when</th><th>model</th><th>tests</th>'
      +'<th>pass</th><th>fail</th><th>warn</th><th>skip</th>'
      +'<th>exchanges</th><th></th></tr>';
    for(const r of d.runs){
      const c=r.counts||{};
      const when=new Date((r.started_at||0)*1000).toLocaleString();
      h+='<tr><td>'+esc(when)+'</td>'
        +'<td>'+esc(r.model||'(unnamed)')+'</td>'
        +'<td>'+(r.n_tests||0)+'</td>'
        +'<td class="st-pass">'+(c.pass||0)+'</td>'
        +'<td class="st-fail">'+(c.fail||0)+'</td>'
        +'<td class="st-warn">'+(c.warn||0)+'</td>'
        +'<td class="st-skip">'+(c.skip||0)+'</td>'
        +'<td>'+(r.n_exchanges||0)+'</td>'
        +'<td><button class="mini secondary" onclick="benchShowRun(\''+esc(r.run_id)+'\')">view</button> '
        +'<a class="pill" href="/api/bench_run_detail?download=1&run_id='
        +encodeURIComponent(r.run_id)+'" download>download</a></td></tr>';
    }
    h+='</table><div id="bn_run_detail"></div>';
    setHTML($('bn_history'), h);
  }catch(e){ setHTML($('bn_history'),'<span class="reasons">'+esc(''+e)+'</span>'); }
}
// The whole point of storing the transcript: read back what was asked and
// what the model answered, per test, without leaving the page.
async function benchShowRun(id){
  const box=$('bn_run_detail'); if(!box) return;
  setHTML(box,'<span class="muted">loading run&hellip;</span>');
  try{
    const d=await (await fetch('/api/bench_run_detail?run_id='+encodeURIComponent(id))).json();
    if(!d.ok){ setHTML(box,'<span class="reasons">'+esc(d.error||'error')+'</span>'); return; }
    const r=d.run;
    let h='<div class="cardblock"><div class="sxs-h">run '+esc(r.run_id)+'</div>'
      +'<div class="muted">'+esc(r.model||'(unnamed)')+' &middot; '+esc(r.endpoint||'')
      +' &middot; '+(r.duration_s||0)+' s'
      +(r.error?' &middot; <span class="st-fail">ended with: '+esc(r.error)+'</span>':'')
      +'</div>';
    for(const t of (r.results||[])){
      h+='<div style="margin-top:var(--s2)"><b>'+t.test_id+'. '+esc(t.label||'')+'</b> '
        +'<span class="st-'+esc(t.status||'skip')+'">'+esc(t.status||'')+'</span>'
        +(t.reason?' <span class="muted">'+esc(t.reason)+'</span>':'')+'</div>';
      const ex=(r.transcript||[]).filter(e=>e.test_id===t.test_id);
      for(const e of ex){
        const req=(((e.request||{}).messages)||[])
          .map(m=>(m.role||'?')+': '+(typeof m.content==='string'?m.content:JSON.stringify(m.content)))
          .join('\n\n');
        h+='<details style="margin:var(--s1) 0"><summary class="muted">'
          +'exchange &middot; HTTP '+(e.http_code==null?'--':e.http_code)
          +(e.wall_ms!=null?(' &middot; '+Math.round(e.wall_ms)+' ms'):'')+'</summary>'
          +'<div class="muted" style="margin-top:var(--s1)">request</div>'
          +'<pre style="max-height:220px;overflow:auto">'+esc(req||JSON.stringify(e.request||{},null,1))+'</pre>'
          +'<div class="muted">answer</div>'
          +'<pre style="max-height:280px;overflow:auto">'+esc(e.answer==null?'(none)':e.answer)+'</pre>'
          +'</details>';
      }
    }
    h+='</div>';
    setHTML(box,h);
  }catch(e){ setHTML(box,'<span class="reasons">'+esc(''+e)+'</span>'); }
}
// ===========================================================================
// Benchmark window.
//
// Running and finished are separate panels: a table that is still filling
// must never be mistaken for a complete result. Each finished run keeps its
// own table of configuration / measure / value, so two runs can be read
// against each other instead of one overwriting the other.
//
// ttft_ms and prefill tok/s are recorded by bench_suite per test and were
// previously dropped on the floor here -- benchEvent read only status and
// metric. They are measures, so they go in the measure column.
// ===========================================================================
const BENCH_STATUS_CLASS={pass:'st-pass',warn:'st-warn',fail:'st-fail',
                          skip:'st-skip',blocked:'st-blocked'};
function benchMeasures(r){
  // Every number this test produced, one row each. Absent stays absent.
  const out=[];
  const m=r.metric||{};
  if(m.name && m.name!=='none' && m.value!=null)
    out.push({k:m.name, v:m.value, u:m.unit||''});
  const d=r.detail||{};
  if(d.ttft_ms!=null) out.push({k:'time to first token', v:(+d.ttft_ms).toFixed(1), u:'ms'});
  if(d.prefill_tps!=null) out.push({k:'prefill', v:(+d.prefill_tps).toFixed(1), u:'tok/s'});
  if(d.prompt_tokens!=null) out.push({k:'prompt', v:d.prompt_tokens, u:'tokens'});
  return out;
}
function benchRowHtml(r){
  const cls=BENCH_STATUS_CLASS[r.status]||'';
  const ms=benchMeasures(r);
  const note=r.reason||(r.detail&&r.detail.http_code!=null?('http '+r.detail.http_code):'')||'';
  return '<tr data-key="bnr_'+r.test_id+'"><td>'+r.test_id+'</td>'
    +'<td style="text-align:left">'+esc(r.label||'')+'</td>'
    +'<td class="'+cls+'"><b>'+esc(r.status)+'</b></td>'
    +'<td style="text-align:left">'+(ms.length
        ? ms.map(x=>esc(x.k)+' <b>'+esc(String(x.v))+'</b>'+(x.u?' '+esc(x.u):'')).join('<br>')
        : '<span class="muted">&mdash;</span>')+'</td>'
    +'<td style="text-align:left" class="muted">'+esc(note)+'</td></tr>';
}
function benchTableHtml(rows){
  return '<table class="mx"><tr><th>#</th><th>test</th><th>status</th>'
    +'<th style="text-align:left">measure / value</th><th style="text-align:left">note</th></tr>'
    +rows.map(benchRowHtml).join('')+'</table>';
}
function renderBenchRunning(){
  const rows=window._benchResults||[];
  const box=$('bn_running_box');
  if(!window._benchActive){ if(box) box.style.display='none'; return; }
  if(box) box.style.display='';
  setHTML($('bn_running'),
    '<div class="muted">'+rows.length+' of '+(window._benchSelected||0)
    +' test(s) done&hellip;</div>'+benchTableHtml(rows));
}
function renderBenchFinished(){
  const runs=window._benchRuns||[];
  if(!runs.length){ setHTML($('bn_out'),'<span class="muted">no run yet.</span>'); return; }
  let h='';
  // Newest first: the run just finished is the one being read.
  for(let i=runs.length-1;i>=0;i--){
    const run=runs[i];
    h+='<details data-key="bnrun_'+run.id+'"'+(i===runs.length-1?' open':'')+'>'
      +'<summary style="cursor:pointer"><b>'+esc(run.label)+'</b> '
      +'<span class="muted">'+esc(run.summary)+'</span></summary>'
      +'<div class="legend">configuration: '+esc(run.config)+'</div>'
      +benchTableHtml(run.results)+'</details>';
  }
  setHTML($('bn_out'), h);
}
async function benchRun(){
  const d=window._benchCatalog;
  const endpoint=$('bn_endpoint').value.trim();
  if(!endpoint){ setHTML($('bn_out'),'<span class="reasons">endpoint required</span>'); return; }
  const selected=[];
  if(d) for(const t of d.tests) if(window._benchSel.has(t.test_id)) selected.push(t.test_id);
  if(!selected.length){ setHTML($('bn_out'),'<span class="reasons">select at least one test (or a preset).</span>'); return; }
  window._benchResults=[];
  window._benchSelected=selected.length;
  window._benchActive=true;
  window._benchError=null;
  $('bn_run_btn').disabled=true;
  renderBenchRunning();
  benchLeadStart(endpoint);
  const model=$('bn_model').value.trim();
  try{
    const body={endpoint, model, selected, force:$('bn_force').checked};
    if(d && d.capabilities) body.capabilities=d.capabilities;
    // Streamed, so it is deliberately NOT routed through api(): aborting a
    // benchmark because a newer request arrived would be wrong.
    const resp=await fetch('/api/bench_run',{method:'POST',body:JSON.stringify(body)});
    const reader=resp.body.getReader(); const dec=new TextDecoder(); let buf='';
    while(true){
      const {done,value}=await reader.read(); if(done) break;
      buf+=dec.decode(value,{stream:true});
      let i;
      while((i=buf.indexOf('\n\n'))>=0){
        const line=buf.slice(0,i).trim(); buf=buf.slice(i+2);
        if(line.startsWith('data:')) benchEvent(JSON.parse(line.slice(5)));
      }
    }
  }catch(e){ window._benchError=''+e; }
  finally{
    $('bn_run_btn').disabled=false;
    benchFinish(endpoint, model);
  }
}
function benchFinish(endpoint, model){
  window._benchActive=false;
  const rows=window._benchResults||[];
  const counts=window._benchCounts||{};
  const summary=window._benchError
    ? ('failed — '+window._benchError)
    : (Object.keys(counts).map(k=>k+' '+counts[k]).join(' · ')||(rows.length+' test(s)'));
  window._benchRuns=(window._benchRuns||[]);
  window._benchRuns.push({
    id: Date.now(),
    label: new Date().toLocaleTimeString(),
    summary: summary,
    config: endpoint+(model?(' · '+model):'')+' · '+rows.length+' test(s)',
    results: rows.slice(),
  });
  window._benchCounts=null;
  renderBenchRunning();
  renderBenchFinished();
}
function benchEvent(ev){
  if(ev.event==='result' && ev.result){
    window._benchResults.push(ev.result);
    renderBenchRunning();
  } else if(ev.event==='done'){
    window._benchCounts=ev.counts||{};
  } else if(ev.event==='error'){
    window._benchError=ev.error;
  }
}
// ---- lead metrics: ms per round, polled off the engine's device timer -----
// A delta between two polls, computed host-side. The first poll only seeds
// the window, which the panel says rather than showing an empty table.
window._benchLeadTimer=null;
const BENCH_LEAD_MS=2000;
const LEAD_LABELS={
  ms_per_verify_round:'ms / verify round',
  ms_per_decode_round:'ms / decode round',
  ms_per_1k_prefill_tokens:'ms / 1k prefill tokens',
  ms_per_draft_pass:'ms / draft pass',
  accept_length:'accepted tokens per round',
  verify_ct:'verify rounds in window',
};
function benchLeadStart(endpoint){
  benchLeadStop();
  benchLeadPoll(endpoint);
  window._benchLeadTimer=setInterval(()=>benchLeadPoll(endpoint), BENCH_LEAD_MS);
}
function benchLeadStop(){
  if(window._benchLeadTimer){ clearInterval(window._benchLeadTimer); window._benchLeadTimer=null; }
}
async function benchLeadPoll(endpoint){
  try{
    const d=await api('/api/bench_lead_metrics',
      {key:'lead', body:{endpoint}, timeout:BENCH_LEAD_MS-200});
    if(!d.ok){ setHTML($('bn_lead'),'<span class="muted">'+esc(d.error||'')+'</span>'); return; }
    const m=d.metrics||{};
    // Kept for the discussion export: the round times are the lead numbers
    // worth sharing, and they exist only while a server is busy.
    if(Object.keys(m).length) window._leadMetrics=m;
    const keys=Object.keys(LEAD_LABELS).filter(k=>m[k]!=null);
    let h='';
    if(keys.length)
      h='<table class="mx"><tr><th style="text-align:left">measure</th><th>value</th></tr>'
        +keys.map(k=>'<tr data-key="lead_'+k+'"><td style="text-align:left">'+esc(LEAD_LABELS[k])
          +'</td><td><b>'+(+m[k]).toFixed(2)+'</b></td></tr>').join('')+'</table>';
    // A metric the engine did not export is absent, never zero -- the note
    // says which, so an empty panel never reads as "the round took no time".
    for(const n of (d.notes||[])) h+='<div class="muted">'+esc(n)+'</div>';
    if(d.window_s!=null) h+='<div class="legend">window '+d.window_s+' s</div>';
    setHTML($('bn_lead'), h||'<span class="muted">nothing reported.</span>');
  }catch(e){}
}

// ===========================================================================
// Discussion export: bundle composer, preview, gated send.
//
// Steering only, like the pairing tab: the bundles, the redaction and the
// Markdown all come from planner/discussion_export.py. This assembles the
// data the page already holds and renders what the server returns -- it does
// not compose Markdown and it does not decide what may be shared.
// ===========================================================================
function discussionData(){
  // Whatever this session actually measured. Sections with no data are
  // simply absent from the report; nothing is invented to fill a bundle.
  const s=window._lastSnapshot||null;
  const n=s? normalizeStartConfig(s) : null;
  const c=(n&&n.cfg)||{};
  const runs=window._benchRuns||[];
  const last=runs.length? runs[runs.length-1] : null;
  const out={};
  if(last) out.bench_results=last.results;
  if(window._leadMetrics) out.lead_metrics=window._leadMetrics;
  const gpus=(s&&s.gpus)||[];
  out.system={
    cards: gpus.map(g=>g.name),
    model: c.model_path||$('bn_model').value.trim()||null,
    quant: c.kv_cache_dtype||null,
  };
  if(n&&n.argv) out.launch_flags=n.argv;
  if(window._lastQuality) out.quality=window._lastQuality;
  const notes=$('sh_notes'); if(notes && notes.value.trim()) out.notes=notes.value.trim();
  return out;
}
function discussionGroups(){
  const out=[];
  for(const el of document.querySelectorAll('#dx_groups input[type=checkbox]'))
    if(el.checked) out.push(el.getAttribute('data-group'));
  return out;
}
async function discussionPreview(){
  setHTML($('dx_out'),'building…');
  try{
    const d=await api('/api/discussion_preview',{key:'dx', body:{
      data:discussionData(), bundle:($('dx_bundle').value||'bench_system'),
      energy_groups:discussionGroups()}});
    if(!d.ok){ setHTML($('dx_out'),'<span class="reasons">'+esc(d.error||'')+'</span>'); return; }
    renderDiscussionOptions(d);
    $('dx_preview').textContent=d.markdown;
    $('dx_wrap').style.display='';
    // The gate is stated plainly rather than by disabling a button with no
    // explanation: "no target configured" is information, not a failure.
    setHTML($('dx_gate'), d.can_send
      ? '<span class="fitc">ready to post to '+esc(d.target||'')+'</span>'
      : '<span class="reasons">'+esc(d.reason||'')+'</span>');
    $('dx_btn').disabled=!d.can_send;
    setHTML($('dx_out'),'');
  }catch(e){ if(!apiAborted(e)) setHTML($('dx_out'),'<span class="reasons">'+esc(apiError(e))+'</span>'); }
}
function renderDiscussionOptions(d){
  const sel=$('dx_bundle');
  if(sel && !sel.options.length && (d.bundles||[]).length){
    setHTML(sel, d.bundles.map(b=>'<option value="'+esc(b.key)+'">'+esc(b.label)+'</option>').join(''));
    sel.value='bench_system';
  }
  const cur=(d.bundles||[]).find(b=>b.key===(sel&&sel.value));
  if(cur) $('dx_bundle_note').textContent=cur.note||'';
  const box=$('dx_groups');
  if(box && !box.children.length && (d.energy_groups||[]).length)
    setHTML(box, d.energy_groups.map(g=>
      '<label style="display:inline-block;margin-right:.6rem"><input type="checkbox" checked '
      +'style="width:auto" data-group="'+esc(g.key)+'" onchange="discussionPreview()"> '
      +esc(g.label)+'</label>').join(''));
}
async function discussionSubmit(){
  if(!confirm('Post the previewed report to the configured GitHub discussion? '
              +'This sends data to an EXTERNAL service.')) return;
  $('dx_btn').disabled=true; setHTML($('dx_out'),'posting…');
  try{
    const d=await api('/api/discussion_submit',{key:'dx_submit', body:{
      data:discussionData(), bundle:($('dx_bundle').value||'bench_system'),
      energy_groups:discussionGroups(), confirmed:true}, timeout:40000});
    setHTML($('dx_out'), d.sent
      ? ('<span class="fitc">'+esc(d.action||'posted')+'</span> — '
         +'<a href="'+esc(d.url||'#')+'" target="_blank">'+esc(d.url||'')+'</a>')
      : '<span class="reasons">'+esc(d.reason||d.error||'not sent')+'</span>');
  }catch(e){ if(!apiAborted(e)) setHTML($('dx_out'),'<span class="reasons">'+esc(apiError(e))+'</span>'); }
  finally{ $('dx_btn').disabled=false; }
}

// ---- GitHub share (#152) -- preview first, explicit confirm, PAT per-use ----
function shareCollect(){
  const s=window._lastSnapshot||null;
  const n=s? normalizeStartConfig(s) : null;
  const c=(n&&n.cfg)||{};
  const payload={};
  payload.model=c.served_model_name||c.model_path||$('bn_model').value.trim()||null;
  const gpus=(s&&s.gpus)||[];
  if(gpus.length) payload.hardware=gpus.map(g=>g.name).join(' + ');
  if(n) payload.command={argv:n.argv||[], env:n.env||{}};
  if($('sh_inc_metrics').checked){
    const mset={};
    const r=window._lastRates;
    if(r){
      if(r.decode_tok_s!=null) mset.decode_tok_s=r.decode_tok_s;
      if(r.prefill_tok_s!=null) mset.prefill_tok_s=r.prefill_tok_s;
    }
    const pc={};
    gpus.forEach(g=>{ pc[g.name+' #'+g.nvml_index]={
      power_w:g.power_watts, util_pct:g.utilization_pct, mem_used_mib:g.mem_used_mib}; });
    if(Object.keys(pc).length) mset.per_card=pc;
    if(Object.keys(mset).length) payload.metrics=mset;
  }
  if($('sh_inc_bench').checked && window._benchResults && window._benchResults.length)
    payload.bench_results=window._benchResults;
  if($('sh_inc_quality').checked && window._lastQuality){
    const q=window._lastQuality;
    payload.quality={svg:q.svg, verdict:q.verdict, tokens:q.tokens, report:q.report};
  }
  const notes=$('sh_notes').value.trim(); if(notes) payload.notes=notes;
  return payload;
}
async function sharePreview(){
  $('sh_out').textContent='building preview…';
  try{
    const r=await fetch('/api/share_preview',{method:'POST',
      body:JSON.stringify({payload:shareCollect()})});
    const d=await r.json();
    if(!d.ok){ $('sh_out').innerHTML='<span class="reasons">'+esc(d.error)+'</span>'; return; }
    window._shareReport=d.report;
    $('sh_preview').textContent=d.report;
    if(!$('sh_repo').value.trim()) $('sh_repo').value=d.default_repo||'';
    $('sh_preview_wrap').style.display='';
    $('sh_out').textContent='review the preview, then confirm to post.';
  }catch(e){ $('sh_out').innerHTML='<span class="reasons">'+esc(''+e)+'</span>'; }
}
async function shareSubmit(){
  const token=$('sh_token').value;
  if(!token){ $('sh_out').innerHTML='<span class="reasons">enter the PAT (used once, never stored).</span>'; return; }
  if(!window._shareReport){ $('sh_out').innerHTML='<span class="reasons">build the preview first.</span>'; return; }
  if(!confirm('Post the previewed report to GitHub now? This sends data to an EXTERNAL service.')) return;
  $('sh_btn').disabled=true; $('sh_out').textContent='submitting…';
  try{
    const r=await fetch('/api/share_submit',{method:'POST',body:JSON.stringify({
      report:window._shareReport, token, repo:$('sh_repo').value.trim(),
      existing_issue:$('sh_issue').value.trim()||null, confirmed:true})});
    const d=await r.json();
    $('sh_out').innerHTML=d.ok
      ?('<span class="fitc">'+esc(d.action)+' issue #'+d.number+'</span> — '
        +'<a href="'+esc(d.url||'#')+'" target="_blank">'+esc(d.url||'')+'</a>')
      :'<span class="reasons">'+esc(d.error)+'</span>';
  }catch(e){ $('sh_out').innerHTML='<span class="reasons">'+esc(''+e)+'</span>'; }
  finally{ $('sh_btn').disabled=false; $('sh_token').value=''; }
}

// The page opens the way it was left: simple unless expert was chosen.
applyViewMode();
// Landing is the default view (live monitor of any reachable server).
showTab('landing');
</script>
</body>
</html>
"""

# The page is assembled once, at import time, with every vendored asset
# inlined. INDEX_HTML stays a plain string constant: the handler serves it
# verbatim and the tests read it directly, exactly as before.
INDEX_HTML = _INDEX_TEMPLATE.replace(
    "/*__VENDOR_MORPHDOM__*/", _vendored_asset("morphdom-umd.min.js")
).replace(
    "/*__VENDOR_NORMALIZE__*/", _vendored_asset("modern-normalize.css")
)
