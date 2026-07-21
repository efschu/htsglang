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


#: rank_tp_ratio (and the other rank_*_ratio flags) accept the string sentinels
#: "auto" / "auto-performance" as well as an explicit int tuple; pass those
#: through verbatim instead of int-parsing them into "a,u,t,o".
_RATIO_SENTINELS = ("auto", "auto-performance")


def _int_list(v):
    if v is None or v == "":
        return None
    if isinstance(v, str) and v.strip() in _RATIO_SENTINELS:
        return v.strip()
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
        DEFAULT_MODEL_ROOTS,
        discover_models,
    )

    payload = payload or {}
    extra = payload.get("extra_roots") or None
    try:
        models = discover_models(extra_roots=extra)
    except Exception as e:  # pragma: no cover - defensive
        return {"ok": False, "error": str(e)}
    return {
        "ok": True,
        "roots": list(DEFAULT_MODEL_ROOTS),
        "models": [_model_to_json(m) for m in models],
    }


def _launch_settings_from_payload(payload: dict):
    """Map the UI launch knobs onto a validated ``LaunchSettings``."""
    from sglang.srt.planner.server_manager import LaunchSettings

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
        chat_template=payload.get("chat_template") or None,
        tool_call_parser=payload.get("tool_call_parser") or None,
        reasoning_parser=payload.get("reasoning_parser") or None,
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
    argv = _argv_from_payload(payload, settings)
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
_MONITOR_DETECT_PORTS = (30000, 30100, 8000)

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
    supplies the flag dict from either source; the geometry is identical."""
    from sglang.srt.planner import placement as placementmod

    model_cfg, err = _resolve_model_cfg_from_payload(payload)
    if err:
        return {"ok": False, "error": err}
    if not model_cfg:
        return {"ok": False, "error": "no model or model_cfg given"}
    flags_dict = payload.get("flags") or {}
    try:
        result = placementmod.compute_placement(model_cfg, flags_dict)
    except Exception as e:  # pragma: no cover - defensive
        return {"ok": False, "error": str(e)}
    return {"ok": True, "placement": result}


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


def flag_catalog_payload(payload: Optional[dict] = None) -> dict:
    """GET /api/flag_catalog -> the full flag catalog metadata, grouped, for
    rendering the runner-tab flag surface (help / hover / dropdown options)."""
    from sglang.srt.planner import flags as flagsmod

    cat = flagsmod.catalog()
    groups: Dict[str, List[dict]] = {}
    for cid, spec in cat.items():
        groups.setdefault(spec.group, []).append({
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
        "upstream_count": flagsmod.upstream_count(),
        "fork_count": flagsmod.fork_count(),
    }


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


def detect_endpoint_payload(payload: Optional[dict] = None) -> dict:
    """GET /api/detect_endpoint -> fresh sweep of the common ports (the
    landing page's 'detect' button). Read-only probes; never boots anything."""
    ports = list((payload or {}).get("ports") or _MONITOR_DETECT_PORTS)
    reachable = []
    for p in ports:
        url = f"http://127.0.0.1:{p}"
        if _probe_sglang(url):
            reachable.append(url)
    global _DETECTED_ENDPOINT
    _DETECTED_ENDPOINT = reachable[0] if reachable else None
    return {
        "ok": True,
        "endpoint": reachable[0] if reachable else None,
        "reachable": reachable,
        "probed": ports,
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

    NOTE: distinct from ``/api/profiles`` (the hardware-rig library the
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

    # The generator's capacity rules (context_length / max_running_requests)
    # size KV via feasibility.plan and need the checkpoint path -- pass the
    # selected model ref through as the base model_path (the rules degrade
    # to the form defaults, with an info note, when it cannot be sized).
    base = {"model_path": payload["model"]} if payload.get("model") else None
    generated: List[dict] = []
    try:
        generated = [
            _prof_json(p)
            for p in flagsmod.profiles(model_cfg, gpus, base=base)
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
    try:
        for res in bench_suite.run_suite(
            endpoint,
            model,
            selected=selected,
            capabilities=caps,
            preset=preset,
            force=bool(payload.get("force")),
        ):
            counts[res.get("status")] = counts.get(res.get("status"), 0) + 1
            yield {"event": "result", "result": res}
    except Exception as e:  # pragma: no cover - defensive
        yield {"event": "error", "error": str(e)}
        return
    yield {"event": "done", "counts": counts}


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
        if self.path.startswith("/api/profiles"):
            try:
                self._json(200, list_profiles())
            except Exception as e:  # pragma: no cover - defensive
                self._json(500, {"error": str(e)})
            return
        if self.path.startswith("/api/detect_endpoint"):
            # MUST precede the /api/detect prefix check below.
            try:
                self._json(200, detect_endpoint_payload())
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
  /* runner: full page width, settings left / results right */
  .cols.runner { grid-template-columns: minmax(420px, 5fr) 7fr; }
  @media (max-width: 820px) { .cols, .cols.runner { grid-template-columns: 1fr; } }
  @media (max-width: 1100px) { .cols.runner { grid-template-columns: 1fr; } }
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
  .rf-energy { margin-top: .8rem; padding-top: .6rem;
               border-top: 1px dashed #7a5c14; }
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
  .cardblock { border: 1px solid #263041; border-radius: 8px;
               padding: .45rem .6rem; margin: .4rem 0; background: #12171e; }
  .segbar { display: flex; height: 14px; border-radius: 4px; overflow: hidden;
            background: #0d1117; border: 1px solid #263041; margin: .25rem 0; }
  .segbar span { display: block; height: 100%; }
  .seglegend { font-size: .66rem; color: #8b96a5; margin: .1rem 0; }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 2px;
         margin-right: .25rem; }
  .rankline { color: #8b96a5; font-size: .68rem; margin: .12rem 0 0 .2rem; }
  .st-pass { color: #56d364; } .st-warn { color: #e3a008; }
  .st-fail { color: #ff7b72; } .st-skip { color: #8b96a5; }
  .st-blocked { color: #8b96a5; }
  .cfgrow { margin: .15rem 0; }
  .cfgrow .cfgk { display: inline-block; width: 7.5rem; color: #8b96a5;
                  font-size: .72rem; vertical-align: top; }
</style>
</head>
<body>
<h1>htsglang offline config planner</h1>
<div class="sub">capacity / feasibility / split &mdash; never estimated throughput.
  Every edit re-runs the same planner the server runs. No GPU touched.</div>

<div class="tabs">
  <button id="tab_landing" class="tab active" onclick="showTab('landing')">Landing (live monitor)</button>
  <button id="tab_runner" class="tab" onclick="showTab('runner')">Runner (models + planner)</button>
  <button id="tab_bench" class="tab" onclick="showTab('bench')">Benchmark</button>
  <button id="tab_explore" class="tab" onclick="showTab('explore')">Explore rigs (matrix)</button>
  <button id="tab_landscape" class="tab" onclick="showTab('landscape')">Landscape (benchmark DB)</button>
  <button id="tab_energy" class="tab" onclick="showTab('energy')">Energy (calibration)</button>
  <button id="tab_quality" class="tab" onclick="showTab('quality')">Quality</button>
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
      <button class="mini secondary" onclick="detectLandingEndpoint()" title="probe the common local ports for a reachable sglang server">detect</button>
      <span id="land_target_note" class="muted">resolving&hellip;</span>
    </div>
  </fieldset>
  <div id="landing_none" class="muted" style="display:none;padding:1rem;border:1px solid #30363d;border-radius:8px">
    No reachable sglang server: no explicit endpoint, no managed instance,
    nothing detected on the common ports. Enter an endpoint above, hit
    <b>detect</b>, or launch a server in the Runner tab.
    <span id="landing_none_status" class="muted"></span></div>
  <div id="landing_live" style="display:none">
    <fieldset>
      <legend>running model + start config</legend>
      <div id="landing_config" class="muted">waiting for first snapshot&hellip;</div>
    </fieldset>
    <div class="cols">
      <div>
        <fieldset>
          <legend>throughput / spec / cache (live + 60s)</legend>
          <div id="landing_rates" class="muted">waiting&hellip;</div>
        </fieldset>
        <fieldset>
          <legend>per-card GPU (live + 60s)</legend>
          <div id="landing_gpus" class="muted">waiting&hellip;</div>
        </fieldset>
      </div>
      <div>
        <fieldset>
          <legend>where it sits on the cards (RUNNING config placement)</legend>
          <div id="landing_placement" class="muted">computing&hellip;</div>
        </fieldset>
      </div>
    </div>
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
        <legend>tests &mdash; presets + per-test selection</legend>
        <div class="actions" id="bn_presets"></div>
        <label style="margin:.4rem 0"><input type="checkbox" id="bn_force" style="width:auto" onchange="benchReGate()">
          force-run tool tests despite a missing tool parser
          <span class="muted">(deliberately surfaces the tool-call cascade)</span></label>
        <div id="bn_tests" class="muted">loading test catalog&hellip;</div>
        <div style="margin-top:.5rem"><button onclick="benchRun()" id="bn_run_btn">Run selected</button></div>
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
    </div>
    <div>
      <fieldset>
        <legend>results (streamed per test)</legend>
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

<div id="view_runner" style="display:none">
<div class="sub">Pick the model ONCE at the top &mdash; it drives everything:
  profiles, flag resolve, plan, placement and launch. Below it: profile &rarr;
  serving identity &rarr; the one flag surface &rarr; launch. No field appears
  twice.</div>

<fieldset>
  <legend>model &mdash; the ONE selector (drives plan, profiles, placement, launch)
    <button class="secondary mini" onclick="loadModels()" title="re-scan roots">rescan</button></legend>
  <div style="display:flex;gap:1rem;flex-wrap:wrap;align-items:flex-start">
    <div style="flex:2;min-width:280px">
      <label>discovered local models</label>
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

<div class="cols runner">
  <div>
    <fieldset>
      <legend>config profile (preset) &mdash; fills the flag surface below;
        adaptive MTP is always on when the checkpoint has draft layers</legend>
      <div id="profile_pick" class="muted">select a model to list profiles&hellip;</div>
      <div style="margin-top:.4rem">
        <input id="profile_save_name" placeholder="name to save current settings as" style="max-width:60%">
        <button class="secondary mini" onclick="saveProfile()">save current as profile</button>
      </div>
      <div id="profile_msg" class="muted" style="margin-top:.3rem"></div>
      <div id="profile_env_box" style="margin-top:.3rem"></div>
    </fieldset>

    <fieldset>
      <legend>serving &mdash; AUTHORITATIVE identity (always wins over a
        profile's argv on launch)</legend>
      <div style="display:flex;gap:.6rem;flex-wrap:wrap">
        <div style="flex:2;min-width:130px">
          <label>served-model-name</label>
          <input id="sv_served" placeholder="(default: model)">
        </div>
        <div style="flex:1;min-width:90px">
          <label>host</label>
          <input id="sv_host" placeholder="127.0.0.1">
        </div>
        <div style="flex:1;min-width:70px">
          <label>port</label>
          <input id="sv_port" value="30000">
        </div>
        <div style="flex:1;min-width:90px">
          <label>context-length</label>
          <input id="sv_ctx" value="8192" onchange="refreshRunnerPlacement()">
        </div>
        <div style="flex:1;min-width:120px">
          <label>max-running-requests</label>
          <input id="max_running_requests" placeholder="(blank: plan 1, launch 16)"
            onchange="doPlan(); refreshRunnerPlacement()">
        </div>
      </div>
      <div class="muted" style="margin-top:.3rem">These five fields own the
        serving identity &mdash; an applied profile keeps its tuning flags but
        never overrides them. 1 concurrent request = single-user = largest KV;
        raising it grows the GDN/mamba pool and shrinks max context (see the
        KV-vs-concurrency table in the result).</div>
    </fieldset>

    <fieldset>
      <legend>flags &mdash; the ONE flag surface
        (<span id="flag_counts" class="muted"></span>)</legend>
      <input id="flag_search" placeholder="search flags (name / help)&hellip;" oninput="filterFlags()">
      <div class="muted" style="margin:.3rem 0 .4rem">Every sglang + fork flag,
        grouped and collapsed. Each field re-resolves on change: greyed +
        hover-? when excluded / incompatible / auto-set; dropdowns self-update;
        incompatible values blocked. Model + serving identity are set above,
        not here.</div>
      <div id="flag_warnings"></div>
      <div id="flag_surface" class="muted">select a model to populate&hellip;</div>
    </fieldset>

    <fieldset>
      <legend>actions &mdash; plan / launch / status (Launch and Restart
        REPLACE the single managed server)</legend>
      <div class="actions">
        <button onclick="doPlan()">Plan / re-validate</button>
        <button onclick="serverStart()">Launch</button>
        <button class="secondary" onclick="serverRestart()">Restart (replace)</button>
        <button class="secondary" onclick="serverStop()">Stop</button>
        <button class="secondary mini" onclick="refreshServerStatus()">status</button>
      </div>
      <div id="sv_out" class="muted" style="margin-top:.6rem"></div>
      <details style="margin-top:.5rem">
        <summary class="muted" style="cursor:pointer">GitHub issue &mdash; submit results / report bug</summary>
        <label>quant descriptor (for the issue text)</label>
        <input id="quant" placeholder="compressed-tensors / Q4_K_M / fp8">
        <div class="actions" style="margin-top:.4rem">
          <button class="secondary" onclick="doIssue('results')">Submit config (RESULTS issue)</button>
          <button class="secondary" onclick="doIssue('bug')">Report bug</button>
        </div>
      </details>
    </fieldset>

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
const $ = id => document.getElementById(id);

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
      $('gguf_choice').innerHTML = opts.map(o=>'<option value="'+esc(o)+'">'+esc(o)+'</option>').join('');
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
  const cards = CARDS.map(c => ({ name: c.name, total_mib: c.total_mib,
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
  // Clear any prior verdict immediately so a slow sizing (e.g. a first-time
  // GGUF header fetch) never leaves a stale REJECTED on screen — the exact
  // "re-validate does nothing" confusion.
  $('verdict').innerHTML = '<div class="verdict">sizing…</div>';
  $('split').innerHTML = '';
  try {
    const r = await fetch('/api/plan', {method:'POST', body: JSON.stringify(payload())});
    const d = await r.json();
    render(d);
  } catch (e) {
    $('verdict').innerHTML = '<div class="verdict nofit">PLAN ERROR</div>';
    $('split').innerHTML = '<ul class="reasons"><li>' + esc(String(e)) + '</li></ul>';
  }
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

function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}

function showTab(t) {
  const TABS = ['landing','runner','bench','explore','landscape','energy','quality'];
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
  // The landing page live-poll runs only while its tab is visible.
  if (t==='landing') startLanding(); else stopLanding();
}

// ===========================================================================
// Shared granular placement renderer (used by BOTH the landing running-config
// view and the runner prospective view -- ONE renderer, two data sources).
// ===========================================================================
function fmtMib(x){ return (x==null)? '—' : (x>=1024? (x/1024).toFixed(2)+'G' : x.toFixed(0)+'M'); }
// One row/block per PHYSICAL CARD: name + total VRAM, a proportional bar of
// what occupies it (weights / KV / mamba / overhead / free), the co-located
// ranks, and the granular per-rank detail (head/token/expert index ranges,
// MTP, host-RAM offload) as compact secondary lines. Used UNCHANGED by both
// the landing (running config) and the runner (prospective config).
const SEG_COLORS = {weights:'#2f81f7', kv:'#2ea043', mamba:'#a371f7', ovh:'#6e7681'};
function renderPlacement(pl){
  if(!pl) return '<span class="muted">no placement.</span>';
  const m=pl.model||{}; let h='';
  h+='<div class="legend" style="margin-top:0">TP '+pl.tp_size+' · DCP '+pl.dcp_size
    +' · heads Q'+m.num_attention_heads+'/KV'+m.num_key_value_heads
    +' · layers '+m.num_hidden_layers
    +(m.num_experts?(' · experts '+m.num_experts):'')+'</div>';
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
    const segs=[['weights',c.weight_mib],['kv',c.kv_mib],['mamba',c.mamba_mib],['ovh',c.overhead_mib]];
    let bar='<div class="segbar" title="weights '+fmtMib(c.weight_mib)+' / KV '+fmtMib(c.kv_mib)
      +' / mamba '+fmtMib(c.mamba_mib)+' / overhead '+fmtMib(c.overhead_mib)+' / free '+fmtMib(free)+'">';
    for(const [k,v] of segs){
      const pct=Math.min(100, v/total*100);
      if(pct>0.3) bar+='<span style="width:'+pct.toFixed(1)+'%;background:'+SEG_COLORS[k]+'"></span>';
    }
    bar+='</div>';
    const over=c.physical_overcommit
      ? ' <b class="nofitc">EXCEEDS CARD (physically impossible)</b>' : '';
    h+='<div class="cardblock">'
      +'<div><b>GPU'+c.gpu_index+'</b> '+esc(c.card_name||'')
      +' <span class="muted">'
      +(c.card_total_mib?fmtMib(c.card_total_mib)+' total · ':'')
      +'ranks ['+c.ranks.join(',')+']'
      +(c.budget_mib!=null?' · budget '+fmtMib(c.budget_mib):'')
      +'</span>'+over+'</div>'
      +bar
      +'<div class="seglegend">'
      +'<span class="dot" style="background:'+SEG_COLORS.weights+'"></span>weights '+fmtMib(c.weight_mib)+' &nbsp;'
      +'<span class="dot" style="background:'+SEG_COLORS.kv+'"></span>KV '+fmtMib(c.kv_mib)+' &nbsp;'
      +(c.mamba_mib>0?('<span class="dot" style="background:'+SEG_COLORS.mamba+'"></span>mamba '+fmtMib(c.mamba_mib)+' &nbsp;'):'')
      +'<span class="dot" style="background:'+SEG_COLORS.ovh+'"></span>overhead '+fmtMib(c.overhead_mib)+' &nbsp;'
      +'<span class="dot" style="background:#0d1117;border:1px solid #30363d"></span>free '+fmtMib(free)
      +' &nbsp;·&nbsp; used <b>'+fmtMib(c.total_mib)+'</b></div>';
    for(const r of c.ranks){
      const d=byRank[r]||{}; const bits=[];
      if(d.attn) bits.push('Q['+d.attn.q_head_start+'..'+d.attn.q_head_end+') '
        +'K['+d.attn.k_head_start+'..'+d.attn.k_head_end+')'+(d.attn.kv_replicated?' repl':'')
        +' V['+d.attn.v_head_start+'..'+d.attn.v_head_end+') '
        +fmtMib(d.attn.q_mib+d.attn.k_mib+d.attn.v_mib+d.attn.o_mib));
      if(d.gdn) bits.push('GDN K['+d.gdn.k_head_start+'..'+d.gdn.k_head_end
        +') V['+d.gdn.v_head_start+'..'+d.gdn.v_head_end+')');
      if(d.kv) bits.push('KV tok ['+d.kv.pos_start+'..'+d.kv.pos_end+') '+d.kv.tokens_owned
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
function pushRing(key,t,v){
  if(v==null||isNaN(v)) return;
  const a=window._ring[key]||(window._ring[key]=[]);
  a.push({t,v});
  const cutoff=t-RING_SECONDS;
  while(a.length && a[0].t<cutoff) a.shift();
}
function sparkline(key,w,hh){
  const a=window._ring[key]||[]; if(a.length<2) return '';
  let mn=Infinity,mx=-Infinity;
  for(const p of a){ if(p.v<mn)mn=p.v; if(p.v>mx)mx=p.v; }
  if(mx<=mn) mx=mn+1;
  const t0=a[0].t, t1=a[a.length-1].t, span=(t1-t0)||1;
  const pts=a.map(p=>((p.t-t0)/span*w).toFixed(1)+','+(hh-((p.v-mn)/(mx-mn))*hh).toFixed(1)).join(' ');
  return '<svg width="'+w+'" height="'+hh+'" style="vertical-align:middle;background:#0d1117;border-radius:3px">'
    +'<polyline fill="none" stroke="#1f6feb" stroke-width="1.5" points="'+pts+'"/></svg>';
}
function landingEndpoint(){
  try{ return (localStorage.getItem('land_endpoint')||'').trim(); }catch(e){ return ''; }
}
function setLandingEndpoint(){
  try{ localStorage.setItem('land_endpoint', $('land_endpoint').value.trim()); }catch(e){}
  window._ring={}; landingPoll();
}
function clearLandingEndpoint(){
  try{ localStorage.removeItem('land_endpoint'); }catch(e){}
  $('land_endpoint').value=''; window._ring={}; landingPoll();
}
async function detectLandingEndpoint(){
  $('land_target_note').textContent='probing the common local ports…';
  try{
    const d=await (await fetch('/api/detect_endpoint')).json();
    if(d.endpoint){
      $('land_endpoint').value=d.endpoint.replace(/^https?:\/\//,'');
      setLandingEndpoint();
    } else {
      $('land_target_note').textContent='nothing reachable on ports '+(d.probed||[]).join(', ');
    }
  }catch(e){ $('land_target_note').textContent=''+e; }
}
async function startLanding(){
  if(window._landTimer) return;
  if(!$('land_endpoint').value) $('land_endpoint').value=landingEndpoint();
  await landingPoll();
  window._landTimer=setInterval(landingPoll, 2000);
}
function stopLanding(){ if(window._landTimer){ clearInterval(window._landTimer); window._landTimer=null; } }
async function landingPoll(){
  let d;
  const ep=landingEndpoint();
  try{
    const r=await fetch('/api/live_snapshot'+(ep?('?endpoint='+encodeURIComponent(ep)):''));
    d=await r.json();
  }catch(e){ return; }
  window._lastTarget=d && d.target;
  if(d && d.target){
    $('land_target_note').innerHTML='monitoring <b>'+esc(d.target.endpoint)+'</b> ('
      +(d.target.managed?'managed instance':esc(d.target.kind)+' / external')+')';
  } else {
    $('land_target_note').textContent='no target';
  }
  if(!d || !d.running || !d.snapshot){
    $('landing_none').style.display=''; $('landing_live').style.display='none';
    const st=d && d.status;
    $('landing_none_status').textContent = (d && d.error)? ' ('+d.error+')'
      : (st && st.state && st.state!=='stopped'? ' (managed server state: '+st.state+')':'');
    return;
  }
  $('landing_none').style.display='none'; $('landing_live').style.display='';
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
  if(n.argv||n.env){
    h+='<details style="margin-top:.3rem"><summary class="muted" style="cursor:pointer">full launch command + env</summary><pre>';
    if(n.env) for(const k of Object.keys(n.env)) h+=esc(k+'='+n.env[k])+'\n';
    if(n.argv) h+=esc(n.argv.join(' '));
    h+='</pre></details>';
  }
  if(n.raw)
    h+='<details style="margin-top:.3rem"><summary class="muted" style="cursor:pointer">raw server_info</summary>'
      +'<pre style="max-height:260px;overflow:auto">'+esc(JSON.stringify(n.raw,null,1))+'</pre></details>';
  return h;
}
function renderLanding(s, tgt){
  window._lastSnapshot=s;
  $('landing_config').innerHTML=renderStartConfig(s, tgt)
    +(s.metrics_error?'<div class="reasons" style="margin-top:.3rem">'+esc(s.metrics_error)
      +' &mdash; token rates need --enable-metrics on the server.</div>':'');

  let gh='';
  for(const g of (s.gpus||[])){
    const key='gpu'+g.nvml_index;
    pushRing(key+'_util',s.t,g.utilization_pct); pushRing(key+'_pow',s.t,g.power_watts);
    pushRing(key+'_temp',s.t,g.temperature_c); pushRing(key+'_mem',s.t,g.mem_used_mib);
    gh+='<div style="margin:.3rem 0;border-bottom:1px solid #21262d;padding-bottom:.3rem">'
      +'<b>'+esc(g.name)+'</b> <span class="muted">#'+g.nvml_index+'</span><br>'
      +'util '+g.utilization_pct+'% '+sparkline(key+'_util',80,18)+' &nbsp; '
      +'pow '+g.power_watts.toFixed(0)+'/'+g.power_limit_w.toFixed(0)+'W '+sparkline(key+'_pow',80,18)+'<br>'
      +'SM/mem '+g.sm_clock_mhz+'/'+g.mem_clock_mhz+'MHz &nbsp; temp '+g.temperature_c+'C '+sparkline(key+'_temp',60,18)+' &nbsp; '
      +'mem '+(g.mem_used_mib/1024).toFixed(1)+'/'+(g.mem_total_mib/1024).toFixed(1)+'G '+sparkline(key+'_mem',60,18)
      +'</div>';
  }
  $('landing_gpus').innerHTML = gh||'<span class="muted">no NVML cards'+(s.nvml_error?' ('+esc(s.nvml_error)+')':'')+'</span>';

  let rh=''; const rates=s.rates;
  if(rates){
    window._lastRates=rates;
    pushRing('dec',s.t,rates.decode_tok_s); pushRing('pfx',s.t,rates.prefill_tok_s);
    rh+='<div>decode <b>'+(rates.decode_tok_s!=null?rates.decode_tok_s.toFixed(1):'—')+'</b> tok/s '+sparkline('dec',120,20)+'</div>'
      +'<div>prefill (non-cached) <b>'+(rates.prefill_tok_s!=null?rates.prefill_tok_s.toFixed(1):'—')+'</b> tok/s '+sparkline('pfx',120,20)+'</div>';
  } else { rh+='<div class="muted">(rates on next tick)</div>'; }
  if(s.spec){
    pushRing('acc',s.t,(s.spec.accept_rate!=null?s.spec.accept_rate*100:null));
    rh+='<div>MTP accept <b>'+((s.spec.accept_rate||0)*100).toFixed(1)+'%</b> '+sparkline('acc',120,20)
      +' &nbsp; adaptive-k '+(s.spec.adaptive_k!=null?s.spec.adaptive_k:'—')+'</div>';
  }
  if(s.cache_hit_rates)
    rh+='<div class="legend">cache-hit per tier: '+Object.keys(s.cache_hit_rates)
      .map(k=>k+' '+((s.cache_hit_rates[k]||0)*100).toFixed(0)+'%').join(' · ')+'</div>';
  if(s.hicache)
    rh+='<div class="legend">HiCache host RAM: '+(s.hicache.host_used_tokens||0).toFixed(0)+'/'
      +(s.hicache.host_total_tokens||0).toFixed(0)+' tok ('+((s.hicache.host_used_frac||0)*100).toFixed(0)+'%)</div>';
  $('landing_rates').innerHTML=rh;

  renderLandingPlacement(s);
}
async function renderLandingPlacement(s){
  const n=normalizeStartConfig(s);
  const cfg=(n&&n.cfg)||{}; const model=cfg.model_path;
  if(!model){ $('landing_placement').innerHTML='<span class="muted">no start config for placement (external server without /get_server_info).</span>'; return; }
  const flags={
    tp_size: cfg.tp_size||1, rank_gpu_id: cfg.rank_gpu_id||null, rank_tp_ratio: cfg.rank_tp_ratio||null,
    rank_gpu_memory_mib: cfg.rank_gpu_memory_mib||null, kv_cache_dtype: cfg.kv_cache_dtype||'auto',
    context_length: cfg.context_length||null, max_running_requests: cfg.max_num_seqs||null,
    speculative_algorithm: (cfg.spec_mode&&cfg.spec_mode!=='off')?cfg.spec_mode:null,
  };
  const ct={},cn={};
  (s.gpus||[]).forEach(g=>{ ct[g.nvml_index]=g.mem_total_mib; cn[g.nvml_index]=g.name; });
  flags.card_total_mib=ct; flags.card_name=cn;
  try{
    const r=await fetch('/api/placement',{method:'POST',body:JSON.stringify(
      {model, gguf_choice: cfg.gguf_variant||null, flags})});
    const d=await r.json();
    $('landing_placement').innerHTML = d.ok? renderPlacement(d.placement) : '<span class="reasons">'+esc(d.error)+'</span>';
  }catch(e){ $('landing_placement').innerHTML='<span class="reasons">'+esc(''+e)+'</span>'; }
}

// ===========================================================================
// Runner tab: full flag surface (resolve greying/auto-set) + config profiles +
// live prospective placement (shared renderer).
// ===========================================================================
window._flagCat=null; window._flagSettings={}; window._profiles=[];
async function loadFlagCatalog(){
  try{
    const r=await fetch('/api/flag_catalog'); const d=await r.json();
    if(!d.ok) return;
    window._flagCat=d;
    $('flag_counts').textContent=d.upstream_count+' upstream + '+d.fork_count+' fork';
    renderFlagSurface();
  }catch(e){}
}
function _surfaceSpecs(g){
  // The serving-identity ids are owned by the MODEL/SERVING sections above
  // the surface -- hiding them here keeps every fact in exactly one place.
  return window._flagCat.groups[g].filter(f=>!SERVING_OWNED[f.id]);
}
function renderFlagSurface(){
  const d=window._flagCat; if(!d) return;
  let h=''; let gi=0;
  for(const g of Object.keys(d.groups)){
    const specs=_surfaceSpecs(g);
    if(!specs.length) continue;
    h+='<details id="flgrp_'+(gi++)+'" style="margin:.2rem 0"><summary style="cursor:pointer"><b>'+esc(g)
      +'</b> <span class="muted">('+specs.length+')</span></summary>';
    for(const f of specs){
      const src=f.source!=='upstream'? ' <span class="pill">'+esc(f.source)+'</span>':'';
      h+='<div class="flagrow" id="flrow_'+f.id+'" style="margin:.15rem 0">'
        +'<label style="display:flex;align-items:center;gap:.3rem" title="'+esc(f.hover||f.help||'')+'">'
        +'<span style="flex:1;min-width:0">'+esc(f.name)+src
        +' <span class="flag-q" id="flq_'+f.id+'" style="color:#e3a008"></span></span>';
      if(f.type==='bool')
        h+='<input type="checkbox" id="fl_'+f.id+'" style="width:auto" onchange="onFlagChange(\''+f.id+'\')">';
      else if(f.allowed)
        h+='<select id="fl_'+f.id+'" style="max-width:45%" onchange="onFlagChange(\''+f.id+'\')"><option value="">(default)</option>'
          +f.allowed.map(a=>'<option value="'+esc(String(a))+'">'+esc(String(a))+'</option>').join('')+'</select>';
      else
        h+='<input id="fl_'+f.id+'" style="max-width:45%" placeholder="'
          +(f.default!=null?esc(String(f.default)):'default')+'" onchange="onFlagChange(\''+f.id+'\')">';
      h+='</label><div class="knob-help" id="flh_'+f.id+'"></div></div>';
    }
    h+='</details>';
  }
  $('flag_surface').innerHTML=h;
  filterFlags();
}
// Search/filter over the ONE flag surface: matching rows stay, matching
// groups auto-open while a query is active; an empty query re-collapses.
function filterFlags(){
  const d=window._flagCat; if(!d) return;
  const box=$('flag_search');
  const q=(box?box.value:'').trim().toLowerCase();
  let gi=0;
  for(const g of Object.keys(d.groups)){
    const specs=_surfaceSpecs(g);
    if(!specs.length) continue;
    const det=$('flgrp_'+(gi++)); if(!det) continue;
    let vis=0;
    for(const f of specs){
      const row=$('flrow_'+f.id); if(!row) continue;
      const hay=(f.name+' '+f.id+' '+(f.help||'')+' '+(f.hover||'')).toLowerCase();
      const hit=!q || hay.indexOf(q)>=0;
      row.style.display=hit?'':'none';
      if(hit) vis++;
    }
    det.style.display=vis?'':'none';
    det.open=!!q && vis>0;
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
  resolveFlags(); refreshRunnerPlacement(); schedulePlan();
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
    const r=await fetch('/api/resolve_flags',{method:'POST',
      body:JSON.stringify({settings:collectFlagSettings(), model})});
    const d=await r.json(); if(!d.ok) return;
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
    if(st.options && el.tagName==='SELECT'){
      const cur=el.value;
      el.innerHTML='<option value="">(default)</option>'
        +st.options.map(a=>'<option value="'+esc(String(a))+'">'+esc(String(a))+'</option>').join('');
      el.value=cur;
    }
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
  loadConfigProfiles(); resolveFlags(); refreshRunnerPlacement(); schedulePlan();
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
  const ct={},cn={};
  CARDS.forEach((c,i)=>{ ct[i]=c.total_mib; cn[i]=c.name; });
  f.card_total_mib=ct; f.card_name=cn;
  for(const k of ['rank_mlp_ratio','rank_moe_ratio','rank_vocab_ratio','dcp_size',
    'rank_gpu_memory_mib','rank_kv_ratio','speculative_algorithm',
    'speculative_num_draft_tokens','moe_resident_expert_fraction'])
    if(s[k]!=null && s[k]!=='') f[k]=s[k];
  if(s.SGLANG_UNEVEN_TOKEN_VECTOR!=null && s.SGLANG_UNEVEN_TOKEN_VECTOR!=='')
    f.kv_token_vector=s.SGLANG_UNEVEN_TOKEN_VECTOR;
  return f;
}
async function refreshRunnerPlacement(){
  const model=$('model').value.trim(); if(!model) return;
  try{
    const r=await fetch('/api/placement',{method:'POST',body:JSON.stringify(
      {model, gguf_choice: ($('gguf_pick').style.display!=='none'?$('gguf_choice').value:null),
       flags: runnerFlags()})});
    const d=await r.json();
    $('runner_placement').innerHTML = d.ok? renderPlacement(d.placement)
      : '<span class="reasons">'+esc(d.error)+'</span>';
  }catch(e){ $('runner_placement').innerHTML='<span class="reasons">'+esc(''+e)+'</span>'; }
}
async function loadConfigProfiles(){
  const model=$('model').value.trim();
  try{
    const q=model? ('?model='+encodeURIComponent(model)):'';
    const r=await fetch('/api/config_profiles'+q); const d=await r.json();
    if(!d.ok){ $('profile_pick').innerHTML='<span class="reasons">'+esc(d.error||'error')+'</span>'; return; }
    window._profiles=(d.generated||[]).concat(d.saved||[]);
    let h='<b>profiles:</b> '+window._profiles.map((p,i)=>'<button class="mini secondary" onclick="applyProfile('
      +i+')" title="'+esc((p.info||[]).join(' | '))+'">'+esc(p.name)+'</button>').join(' ');
    if(d.saved && d.saved.length)
      h+=' <span class="muted">(saved: '+d.saved.map(p=>esc(p.name)).join(', ')+')</span>';
    $('profile_pick').innerHTML=h;
  }catch(e){ $('profile_pick').innerHTML='<span class="reasons">'+esc(''+e)+'</span>'; }
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
function applyProfile(i){
  const p=window._profiles[i]; if(!p) return;
  window._flagSettings={};
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
  $('profile_msg').innerHTML='<span class="muted">applied <b>'+esc(p.name)+'</b>'
    +((p.info&&p.info.length)?' — '+esc(p.info.join(' | ')):'')+'</span>';
  resolveFlags(); refreshRunnerPlacement(); doPlan();
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

// ---- Models tab ----
async function loadModels() {
  $('models_out').innerHTML = 'scanning…';
  try {
    const r = await fetch('/api/models'); const d = await r.json();
    if (!d.ok) { $('models_out').innerHTML = '<span class="reasons">'+esc(d.error)+'</span>'; return; }
    window._models = d.models;
    const opts = d.models.map((m,i)=>{
      const q = (m.quant_method && m.quant_method!=='None') ? ' · '+esc(m.quant_method) : '';
      const vv = (m.gguf_variants && m.gguf_variants.length>1) ? ' · '+m.gguf_variants.length+' quants' : '';
      const err = m.error ? ' · (err)' : '';
      return '<option value="'+i+'">'+esc(m.name)+'  ['+esc(m.format)+q+' · '+m.size_gib+'G]'+vv+err+'</option>';
    }).join('');
    $('models_out').innerHTML =
      '<select id="model_select" style="width:100%;max-width:100%" onchange="pickFromDropdown()">'
      + '<option value="">— select a model —</option>' + opts + '</select>'
      + '<div class="muted" style="margin-top:.3rem">'+d.models.length+' models · '+esc(d.roots.join(', '))+'</div>';
  } catch(e) { $('models_out').innerHTML = '<span class="reasons">'+esc(''+e)+'</span>'; }
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
    $('gguf_choice').innerHTML = m.gguf_variants
      .map(v=>'<option value="'+esc(v.filename)+'">'+esc(v.quant)+' ('+v.size_gib+'G)</option>').join('');
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
function renderServerStatus(d) {
  if (!d) return '';
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
function serverStart() { serverPost('/api/server_start', launchBody()); }
function serverRestart() {
  if (!confirm('Restart REPLACES the single managed instance (stops the running model). Continue?')) return;
  serverPost('/api/server_restart', launchBody());
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
      $('dl_quant').innerHTML = d.gguf_variants.map(v=>'<option value="'+esc(v.quant)+'">'+esc(v.quant)+' ('+esc(v.filename)+')</option>').join('');
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
  try {
    const s = (await (await fetch('/api/server_status')).json()).status || {};
    if (s.state === 'ready' && s.port) {
      $('q_endpoint').value = '127.0.0.1:' + s.port;
      if (s.served_model_name) $('q_model').value = s.served_model_name;
    }
  } catch(e) {}
}
async function qualityRun() {
  if (!$('q_endpoint').value.trim() || !$('q_model').value.trim()) await autofillQuality();
  const endpoint = $('q_endpoint').value.trim();
  const model = $('q_model').value.trim();
  if (!endpoint || !model) { $('q_status').innerHTML = '<span class="reasons">endpoint + model required</span>'; return; }
  $('q_btn').disabled = true; $('q_status').innerHTML = 'calling the model (backend-side)…';
  const budget = $('q_budget').value.trim();
  try {
    const r = await fetch('/api/quality_run', {method:'POST', body: JSON.stringify({
      endpoint, model, thinking: $('q_think').checked,
      thinking_budget: budget!==''?parseInt(budget):null})});
    const d = await r.json();
    if (!d.ok) { $('q_status').innerHTML = '<span class="reasons">'+esc(d.error)+'</span>'; return; }
    window._lastQuality = d;
    $('q_svg').innerHTML = d.svg ? d.svg : '<span class="muted">no SVG extracted</span>';
    const tk = d.tokens||{};
    let h = '<div class="verdict '+verdictClass(d.verdict)+'">'+esc(d.verdict)+'</div>';
    h += '<div class="legend"><b>tokens:</b> prompt '+(tk.prompt??'?')+' / completion '+(tk.completion??'?')+' / total '+(tk.total??'?')
       + ' &nbsp; <b>representation:</b> '+esc(d.representation||'?')+'</div>';
    h += '<div style="margin:.4rem 0">'+esc(d.report||'')+'</div>';
    if (d.offer_download && d.svg)
      h += '<button class="mini secondary" onclick="dlSvg()">download raw SVG</button>';
    else if (d.offer_download && d.raw)
      h += '<button class="mini secondary" onclick="dlRaw()">download raw answer</button>';
    $('q_result').innerHTML = h;
    $('q_status').innerHTML = 'done.';
    if ($('q_save').checked) await saveShot(d);
  } catch(e) { $('q_status').innerHTML = '<span class="reasons">'+esc(''+e)+'</span>'; }
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
function renderBenchTests(){
  const d=window._benchCatalog; if(!d) return;
  $('bn_presets').innerHTML='<span class="muted">presets:</span> '
    +Object.keys(d.presets).map(p=>'<button class="mini secondary" onclick="benchPreset(\''+p+'\')">'+esc(p)+'</button>').join(' ');
  let h='';
  for(const t of d.tests){
    const gated=t.gate_status!=null;  // blocked / skip: greyed with the reason
    h+='<div style="margin:.12rem 0;opacity:'+(gated?'.45':'1')+'" title="'
      +esc(t.gate_reason||t.expected_fail_note||'')+'">'
      +'<label style="display:flex;gap:.35rem;align-items:flex-start;margin:0">'
      +'<input type="checkbox" id="bnt_'+t.test_id+'" style="width:auto;margin-top:.15rem" '+(gated?'disabled':'')+'>'
      +'<span>'+t.test_id+'. '+esc(t.label)
      +(t.optional?' <span class="muted">(opt)</span>':'')
      +(t.crash_prone?' <span style="color:#e3a008">(crash-prone, runs last)</span>':'')
      +(gated?' <span class="muted">['+esc(t.gate_status)+': '+esc(t.gate_reason||'')+']</span>':'')
      +(!gated&&t.expected_fail_note?' <span style="color:#e3a008">[expected-fail: '+esc(t.expected_fail_note)+']</span>':'')
      +'</span></label></div>';
  }
  $('bn_tests').innerHTML=h;
}
function benchPreset(p){
  const d=window._benchCatalog; if(!d) return;
  const ids=d.presets[p]||[];
  for(const t of d.tests){
    const el=$('bnt_'+t.test_id);
    if(el && !el.disabled) el.checked=ids.includes(t.test_id);
  }
}
async function benchRun(){
  const d=window._benchCatalog;
  const endpoint=$('bn_endpoint').value.trim();
  if(!endpoint){ $('bn_out').innerHTML='<span class="reasons">endpoint required</span>'; return; }
  const selected=[];
  if(d) for(const t of d.tests){ const el=$('bnt_'+t.test_id); if(el&&el.checked) selected.push(t.test_id); }
  if(!selected.length){ $('bn_out').innerHTML='<span class="reasons">select at least one test (or a preset).</span>'; return; }
  window._benchResults=[];
  $('bn_run_btn').disabled=true;
  $('bn_out').innerHTML='<table class="mx" id="bn_table"><tr><th>#</th><th>test</th><th>status</th>'
    +'<th>metric</th><th>note</th></tr></table><div id="bn_done" class="muted">running&hellip;</div>';
  try{
    const body={endpoint, model:$('bn_model').value.trim(), selected, force:$('bn_force').checked};
    if(d && d.capabilities) body.capabilities=d.capabilities;
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
  }catch(e){ const el=$('bn_done'); if(el) el.innerHTML='<span class="reasons">'+esc(''+e)+'</span>'; }
  finally{ $('bn_run_btn').disabled=false; }
}
function benchEvent(ev){
  if(ev.event==='result' && ev.result){
    const r=ev.result; window._benchResults.push(r);
    const m=r.metric||{};
    const mtxt=(m.name && m.name!=='none' && m.value!=null)
      ? (m.name+'='+m.value+(m.unit?' '+m.unit:'')) : '';
    const cls={pass:'st-pass',warn:'st-warn',fail:'st-fail',skip:'st-skip',blocked:'st-blocked'}[r.status]||'';
    const row=document.createElement('tr');
    row.innerHTML='<td>'+r.test_id+'</td><td style="text-align:left">'+esc(r.label||'')+'</td>'
      +'<td class="'+cls+'"><b>'+esc(r.status)+'</b></td><td>'+esc(mtxt)+'</td>'
      +'<td style="text-align:left" class="muted">'
      +esc(r.reason||(r.detail&&r.detail.http_code!=null?('http '+r.detail.http_code):'')||'')+'</td>';
    const tb=$('bn_table'); if(tb) tb.appendChild(row);
  } else if(ev.event==='done'){
    const el=$('bn_done');
    if(el) el.innerHTML='done &mdash; '+Object.keys(ev.counts||{})
      .map(k=>k+' '+ev.counts[k]).join(' · ');
  } else if(ev.event==='error'){
    const el=$('bn_done');
    if(el) el.innerHTML='<span class="reasons">'+esc(ev.error)+'</span>';
  }
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

// Landing is the default view (live monitor of any reachable server).
showTab('landing');
</script>
</body>
</html>
"""
