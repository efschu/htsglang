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
    "plan_from_payload",
    "issue_from_payload",
    "matrix_from_payload",
    "list_profiles",
    "serve",
]


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


def _load_hardware_from_payload(hw):
    from sglang.srt.planner import hardware as hwmod

    source = (hw or {}).get("source", "manual")
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
    }


def plan_from_payload(payload: dict) -> dict:
    """Run ``feasibility.plan()`` for a UI form payload; return a JSON-able
    dict (or ``{valid: False, reasons: [...]}`` for a rejected manual edit)."""
    from sglang.srt.planner.feasibility import PlanRejected, plan
    from sglang.srt.planner.model import resolve_model_ref

    try:
        model_path = resolve_model_ref(payload["model"])
    except (ValueError, KeyError) as e:
        return {"valid": False, "reasons": [f"model: {e}"]}

    hardware = _load_hardware_from_payload(payload.get("hardware", {}))

    mem = _int_list(payload.get("rank_gpu_memory_mib"))
    rank_gpu_id = _int_list(payload.get("rank_gpu_id"))
    if mem is not None and len(mem) == 1:
        tp_guess = (
            len(rank_gpu_id)
            if rank_gpu_id
            else (payload.get("tp_size") or len(hardware.gpus))
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
            speculative_algorithm=payload.get("speculative_algorithm") or None,
            speculative_num_draft_tokens=payload.get(
                "speculative_num_draft_tokens"
            )
            or None,
            speculative_draft_model_path=payload.get(
                "speculative_draft_model_path"
            )
            or None,
            max_running_requests=payload.get("max_running_requests") or None,
        )
    except PlanRejected as e:
        return {"valid": False, "reasons": e.reasons}
    except ValueError as e:
        return {"valid": False, "reasons": [str(e)]}

    return _plan_to_dict(result, model_path)


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


# ===========================================================================
# HTTP server (stdlib only; same shape as the rig-dashboard).
# ===========================================================================


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
        self._send(404, "not found", "text/plain")

    def do_POST(self):
        try:
            payload = self._read_json()
        except Exception as e:
            self._json(400, {"error": f"bad json: {e}"})
            return
        try:
            if self.path.startswith("/api/plan"):
                self._json(200, plan_from_payload(payload))
                return
            if self.path.startswith("/api/issue"):
                self._json(200, issue_from_payload(payload))
                return
            if self.path.startswith("/api/matrix"):
                self._json(200, matrix_from_payload(payload))
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
  .verdict { font-size: 1rem; font-weight: 700; padding: .55rem .7rem;
             border-radius: 7px; margin-bottom: .8rem; }
  .fit { background: #10331d; color: #56d364; border: 1px solid #1c6b34; }
  .nofit { background: #3a1417; color: #ff7b72; border: 1px solid #7d2a2a; }
  table { border-collapse: collapse; width: 100%; font-size: .76rem; }
  th, td { text-align: right; padding: .3rem .5rem; border-bottom: 1px solid #232b35; }
  th:first-child, td:first-child { text-align: left; }
  .bar { height: 9px; background: #232b35; border-radius: 4px; overflow: hidden; }
  .bar > span { display: block; height: 100%; background: #2f81f7; }
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
      <legend>model + hardware</legend>
      <label>Model (config.json dir, .gguf file, or HF id)</label>
      <input id="model" placeholder="/path/to/Qwen3.6-27B-AWQ or org/model">
      <label>Hardware</label>
      <select id="hw_source">
        <option value="manual">manual (declare cards)</option>
        <option value="nvml">live NVML / nvidia-smi</option>
      </select>
      <label>Cards (one per line: NAME:TOTAL_MIB, or 20g for GiB)</label>
      <textarea id="gpus" rows="3" placeholder="RTX 5090:32607&#10;RTX 3080:20480&#10;RTX 3080:20480"></textarea>
    </fieldset>

    <fieldset>
      <legend>plan &mdash; blank = auto-derive</legend>
      <div id="knobs"></div>
      <label>tp-size</label>
      <input id="tp_size" placeholder="(= #cards, or set)">
      <label>kv-cache-dtype</label>
      <input id="kv_cache_dtype" placeholder="auto | fp8_e4m3">
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
  KNOBS = d.knobs;
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

function payload() {
  const p = {
    model: $('model').value.trim(),
    hardware: { source: $('hw_source').value,
      gpus: $('gpus').value.split('\n').map(s=>s.trim()).filter(Boolean) },
    tp_size: $('tp_size').value ? parseInt($('tp_size').value) : null,
    kv_cache_dtype: $('kv_cache_dtype').value.trim(),
    quant: $('quant').value.trim(),
  };
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
    return;
  }
  const cap = d.capacity;
  $('verdict').innerHTML = d.fits
    ? '<div class="verdict fit">FITS &check; <span class="est">(estimate &mdash; the runtime measures real free bytes)</span></div>'
    : '<div class="verdict nofit">DOES NOT FIT</div>';
  if (!d.fits) {
    $('split').innerHTML = '<ul class="reasons">' +
      (d.infeasible_reasons||[]).map(x=>'<li>'+esc(x)+'</li>').join('') + '</ul>';
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
  // launch flags
  $('flags').innerHTML = (d.launch_flags && d.launch_flags.length)
    ? '<p class="muted">launch flags (copy into your command):</p><pre>'
      + d.launch_flags.map(esc).join('\n') + '</pre>' : '';
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
  $('view_plan').style.display = t==='plan' ? '' : 'none';
  $('view_explore').style.display = t==='explore' ? '' : 'none';
  $('tab_plan').classList.toggle('active', t==='plan');
  $('tab_explore').classList.toggle('active', t==='explore');
  if (t==='explore' && !window._profLoaded) loadProfiles();
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

loadKnobs();
</script>
</body>
</html>
"""
