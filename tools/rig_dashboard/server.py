#!/usr/bin/env python3
"""Rig-Dashboard -- live view of the heterogeneous 3-GPU rig.

A single self-contained process: it polls NVML for live per-card VRAM / temp /
util / power / PCIe, optionally scrapes a running sglang server
(``/get_server_info`` + ``/metrics``), parses the boot-log uneven-TP plan, and
serves one static HTML page that renders it all as SVG.

No third-party web deps: stdlib ``http.server`` + ``urllib`` + optional
``pynvml`` (falls back to parsing ``nvidia-smi`` if pynvml is absent).

    python3 server.py --port 8770 \
        --sglang http://127.0.0.1:30010 \
        --boot-log /path/to/server.log

Then open http://localhost:8770/ .  The page auto-refreshes ~1 Hz.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan_parser import derive_head_split, largest_remainder, parse_plan_file

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- model geometry (defaults for Qwen3.6-27B; override via CLI) -----------
# These are the *structural* counts used to draw the head/unit distribution.
# The authoritative per-rank splits for MLP/vocab/tokens come from the boot log;
# Q/KV/GDN head splits are derived here from the shard ratio + these totals.
DEFAULT_GEOMETRY = {
    "q_heads": 24,        # total attention query heads
    "kv_heads": 4,        # total key/value heads (GQA groups)
    "gdn_v_heads": 32,    # linear-attn (GDN) value heads
    "gdn_k_heads": 16,    # linear-attn (GDN) key heads
    "vocab_units": 25,    # abstract vocab units the vocab ratio sums to
}

_G = {}          # active geometry (filled in main)
_CFG = {}        # runtime config (sglang url, boot log path)


# ===========================================================================
# NVML live sampling
# ===========================================================================
def _nvml_via_pynvml():
    import pynvml

    pynvml.nvmlInit()
    try:
        out = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)

            def _try(fn, *a, default=None):
                try:
                    return fn(*a)
                except Exception:
                    return default

            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            util = _try(pynvml.nvmlDeviceGetUtilizationRates, h)
            out.append(
                {
                    "index": i,
                    "name": name,
                    "uuid": (
                        (lambda u: u.decode() if isinstance(u, bytes) else u)(
                            pynvml.nvmlDeviceGetUUID(h)
                        )
                    ),
                    "mem_total_mib": mem.total // (1024 * 1024),
                    "mem_used_mib": mem.used // (1024 * 1024),
                    "temp_c": _try(
                        pynvml.nvmlDeviceGetTemperature, h, pynvml.NVML_TEMPERATURE_GPU
                    ),
                    "util_pct": getattr(util, "gpu", None),
                    "power_w": (
                        lambda p: round(p / 1000.0, 1) if p is not None else None
                    )(_try(pynvml.nvmlDeviceGetPowerUsage, h)),
                    "pcie_gen": _try(
                        pynvml.nvmlDeviceGetCurrPcieLinkGeneration, h
                    ),
                    "pcie_width": _try(pynvml.nvmlDeviceGetCurrPcieLinkWidth, h),
                }
            )
        return out
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _nvml_via_smi():
    q = (
        "index,name,uuid,memory.total,memory.used,temperature.gpu,"
        "utilization.gpu,power.draw,pcie.link.gen.current,pcie.link.width.current"
    )
    txt = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
        text=True,
    )
    out = []
    for line in txt.strip().splitlines():
        c = [x.strip() for x in line.split(",")]

        def _i(v):
            try:
                return int(float(v))
            except Exception:
                return None

        def _f(v):
            try:
                return float(v)
            except Exception:
                return None

        out.append(
            {
                "index": _i(c[0]),
                "name": c[1],
                "uuid": c[2],
                "mem_total_mib": _i(c[3]),
                "mem_used_mib": _i(c[4]),
                "temp_c": _i(c[5]),
                "util_pct": _i(c[6]),
                "power_w": _f(c[7]),
                "pcie_gen": _i(c[8]),
                "pcie_width": _i(c[9]),
            }
        )
    return out


def sample_nvml():
    try:
        return _nvml_via_pynvml(), "pynvml"
    except Exception:
        pass
    try:
        return _nvml_via_smi(), "nvidia-smi"
    except Exception as e:
        return [], f"unavailable: {e}"


# ===========================================================================
# sglang live scraping (robust to server-down)
# ===========================================================================
def _get(url, timeout=1.5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def scrape_server_info():
    base = _CFG.get("sglang")
    if not base:
        return None
    for path in ("/get_server_info", "/server_info"):
        try:
            return json.loads(_get(base + path))
        except Exception:
            continue
    return None


_METRIC = re.compile(r'^(sglang:[a-z_]+)(?:\{[^}]*\})?\s+([0-9eE.+-]+)\s*$')


def scrape_metrics():
    base = _CFG.get("sglang")
    if not base:
        return None
    try:
        txt = _get(base + "/metrics")
    except Exception:
        return None
    vals = {}
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        m = _METRIC.match(line.strip())
        if not m:
            continue
        # keep last value seen for each bare metric name
        try:
            vals[m.group(1)] = float(m.group(2))
        except ValueError:
            pass
    keep = {
        "gen_throughput": "sglang:gen_throughput",
        "num_running_reqs": "sglang:num_running_reqs",
        "num_queue_reqs": "sglang:num_queue_reqs",
        "num_paused_reqs": "sglang:num_paused_reqs",
        "token_usage": "sglang:token_usage",
        "kv_used_tokens": "sglang:kv_used_tokens",
        "kv_available_tokens": "sglang:kv_available_tokens",
        "mamba_used_tokens": "sglang:mamba_used_tokens",
        "max_total_num_tokens": "sglang:max_total_num_tokens",
        "cache_hit_rate": "sglang:cache_hit_rate",
        "spec_accept_length": "sglang:spec_accept_length",
        "num_requests_total": "sglang:num_requests_total",
    }
    return {k: vals[v] for k, v in keep.items() if v in vals}


# ===========================================================================
# plan + geometry -> per-GPU VRAM segments and unit distribution
# ===========================================================================
def _param_family_weights():
    """Relative parameter mass per weight family, used only to *estimate* how a
    rank's measured weight_gb splits across families (labelled 'estimated')."""
    g = _G
    # rough per-layer param proportions (attention qkv+o, GDN in/out, MLP,
    # plus a global embed+lm_head term). Absolute scale cancels out.
    return {
        "attn": g["q_heads"] + 2 * g["kv_heads"],
        "gdn": g["gdn_v_heads"] + g["gdn_k_heads"],
        "mlp": 48,  # dominant; MLP intermediate is the bulk of a dense layer
        "embed": 8,  # embed + lm_head (vocab projection)
    }


def build_plan_view(plan):
    """Turn the raw parsed plan into a render-ready structure."""
    if not plan:
        return {"available": False}

    ratio = plan.get("rank_tp_ratio") or plan.get("rank_gpu_memory_mib") or []
    rank_gpu = plan.get("rank_gpu_id") or [g["gpu_index"] for g in plan.get("gpus", [])]
    tp = plan.get("tp_size") or len(ratio) or len(rank_gpu)
    ranks = plan.get("ranks", {})

    # ---- Q/KV head split ----
    # Preferred source: the materialized REPLICATED-KV log line (authoritative,
    # e.g. q split [8,4,4] in units of kv_total). Fallback: derive from the
    # shard ratio + model geometry (whole-GQA-group rule).
    kv_total, q_total = _G["kv_heads"], _G["q_heads"]
    heads = {}
    rep = plan.get("attn_replicated")
    if rep:
        heads = {
            "kv_mode": "replicated",
            "kv_split": [rep["kv_heads"]] * len(rep["q_split"]),
            "q_split": rep["q_split"],
            "unit": rep["unit"],
            "source": "materialized (boot log)",
        }
        q_total = sum(rep["q_split"])
        kv_total = rep["kv_heads"]
    elif ratio and kv_total >= tp:
        kv_split = largest_remainder(kv_total, ratio)
        grp = max(1, q_total // max(1, kv_total))
        q_split = [k * grp for k in kv_split]
        # top up any rounding gap on rank 0
        if sum(q_split) != q_total and q_split:
            q_split[0] += q_total - sum(q_split)
        heads = {
            "kv_mode": "sharded",
            "kv_split": kv_split,
            "q_split": q_split,
            "source": "derived (ratio + geometry)",
        }
    elif ratio:
        # TP > kv_heads: KV replicated on every rank, Q head-sharded
        heads = {
            "kv_mode": "replicated",
            "kv_split": [kv_total] * len(ratio),
            "q_split": derive_head_split(q_total, ratio),
            "source": "derived (ratio + geometry)",
        }

    if heads:
        heads["q_total"] = sum(heads["q_split"])
        heads["kv_total"] = kv_total

    gdn_v = derive_head_split(_G["gdn_v_heads"], ratio) if ratio else []
    gdn_k = derive_head_split(_G["gdn_k_heads"], ratio) if ratio else []

    # ---- MoE expert ownership (materialized, GGUF MoE uneven TP) ----
    moe = plan.get("moe_experts") or {}
    moe_split = None
    if moe:
        n = max(moe) + 1
        moe_split = {
            "counts": [
                (moe[r]["end"] - moe[r]["start"]) if r in moe else None
                for r in range(max(n, tp))
            ],
            "ranges": [
                [moe[r]["start"], moe[r]["end"]] if r in moe else None
                for r in range(max(n, tp))
            ],
            "total": next(iter(moe.values()))["total"],
        }

    # ---- per-rank VRAM segments (measured GB from the boot log) ----
    fam = _param_family_weights()
    fam_sum = sum(fam.values())
    rank_view = []
    for r in range(tp):
        rd = ranks.get(r, {})
        w = rd.get("weight_gb", 0.0)
        seg = {
            "rank": r,
            "gpu": rank_gpu[r] if r < len(rank_gpu) else None,
            "weight_gb": w,
            "draft_gb": rd.get("draft_weight_gb", 0.0),
            "kv_gb": rd.get("kv_gb", 0.0),
            "kv_tokens": rd.get("kv_tokens", 0),
            "draft_kv_gb": rd.get("draft_kv_gb", 0.0),
            "mamba_gb": rd.get("mamba_gb", 0.0),
            "free_gb_end": rd.get("free_gb_end"),
            # estimated weight-by-family split of the measured weight_gb
            "weight_family_est": {
                k: round(w * v / fam_sum, 3) for k, v in fam.items()
            },
            "mlp_units": (plan.get("mlp_units") or [None] * tp)[r]
            if plan.get("mlp_units")
            else None,
            "vocab_units": (plan.get("rank_vocab_ratio") or [None] * tp)[r]
            if plan.get("rank_vocab_ratio")
            else None,
            "token_units": (plan.get("token_vector") or [None] * tp)[r]
            if plan.get("token_vector")
            else None,
            "q_heads": heads.get("q_split", [None] * tp)[r] if heads else None,
            "kv_heads": heads.get("kv_split", [None] * tp)[r] if heads else None,
            "gdn_v": gdn_v[r] if r < len(gdn_v) else None,
        }
        rank_view.append(seg)

    return {
        "available": True,
        "model": (plan.get("model") or "").split("/")[-1],
        "quant": plan.get("quant"),
        "kv_cache_dtype": plan.get("kv_cache_dtype"),
        "spec_algorithm": plan.get("spec_algorithm"),
        "tp_size": tp,
        "dcp_size": plan.get("dcp_size"),
        "rank_gpu_id": rank_gpu,
        "rank_tp_ratio": ratio,
        "memory_budgets_mib": plan.get("memory_budgets_mib"),
        "reserve_mib": plan.get("reserve_mib"),
        "gpus": plan.get("gpus", []),
        "mlp_vector": plan.get("mlp_vector"),
        "mlp_units": plan.get("mlp_units"),
        "vocab_ratio": plan.get("rank_vocab_ratio"),
        "token_vector": plan.get("token_vector"),
        "token_block": plan.get("token_units"),
        "profiled_capacity": plan.get("profiled_capacity"),
        "max_total_num_tokens": plan.get("max_total_num_tokens"),
        "heads": heads,
        "moe": moe_split,
        "gdn_v": gdn_v,
        "gdn_k": gdn_k,
        "geometry": dict(_G),
        "ranks": rank_view,
        # duplicates in rank_gpu_id => co-located ranks share a physical GPU
        "colocation": _colocation(rank_gpu),
    }


def _colocation(rank_gpu):
    by = {}
    for r, g in enumerate(rank_gpu):
        by.setdefault(g, []).append(r)
    return {str(g): rs for g, rs in by.items()}


# Parsed-plan cache: the boot log can be huge and still growing (crash-loop
# logs of 17 GB have been observed live), so never reparse per request.
# Reparse at most every _PLAN_TTL seconds while the boot is in progress; once
# the parser saw the "fired up and ready" line the plan is final and is only
# reparsed if the file shrinks (i.e. it was replaced by a new boot).
_PLAN_TTL = 15.0
_plan_cache = {}  # path -> {"ts", "size", "complete", "view"}


def load_plan():
    import time as _t

    path = _CFG.get("boot_log")
    if not path or not os.path.exists(path):
        return {"available": False, "reason": "no boot log configured"}
    try:
        size = os.path.getsize(path)
        c = _plan_cache.get(path)
        if c is not None:
            fresh = (_t.time() - c["ts"]) < _PLAN_TTL
            final = c["complete"] and size >= c["size"]
            if fresh or final:
                return c["view"]
        raw = parse_plan_file(path)
        raw["_source"] = path
        view = build_plan_view(raw)
        _plan_cache[path] = {
            "ts": _t.time(),
            "size": size,
            "complete": bool(raw.get("boot_complete")),
            "view": view,
        }
        return view
    except Exception as e:
        return {"available": False, "reason": f"parse error: {e}"}


def map_plan_gpus_to_nvml(plan, nvml):
    """Map the boot log's GPU indices onto live NVML card indices.

    The server's boot-time enumeration and the current NVML order can diverge
    (observed live: sglang's GPU 0 = RTX 5090, system NVML index 1). Match by
    card NAME when the auto-performance block logged it (unique names only),
    then by per-rank memory budget best-fit (a 29607 MiB budget can only live
    on the 32 GB card). Falls back to identity for anything unresolved.
    """
    if not plan.get("available") or not nvml:
        return {}
    names = {g["gpu_index"]: g["name"] for g in plan.get("gpus", [])}
    budgets = {}
    mib = plan.get("memory_budgets_mib") or []
    for r, g in enumerate(plan.get("rank_gpu_id") or []):
        if r < len(mib):
            budgets[g] = max(budgets.get(g, 0), mib[r])
    plan_gpus = sorted({g for g in (plan.get("rank_gpu_id") or [])})
    mapping, used = {}, set()
    # pass 1: unique-name matches (e.g. the single RTX 5090)
    for pg in plan_gpus:
        nm = names.get(pg)
        if not nm:
            continue
        same = [c for c in nvml if c["name"] == nm]
        if len(same) == 1 and same[0]["index"] not in used:
            mapping[pg] = same[0]["index"]
            used.add(same[0]["index"])
    # pass 2: budget best-fit, largest budget first
    rest = sorted(
        (pg for pg in plan_gpus if pg not in mapping),
        key=lambda g: -budgets.get(g, 0),
    )
    for pg in rest:
        cands = [
            c
            for c in nvml
            if c["index"] not in used
            and c["mem_total_mib"] >= budgets.get(pg, 0)
            and (pg not in names or c["name"] == names[pg])
        ]
        cands.sort(key=lambda c: c["mem_total_mib"])  # tightest fit
        if not cands:
            cands = sorted(
                (c for c in nvml if c["index"] not in used),
                key=lambda c: -c["mem_total_mib"],
            )
        if cands:
            mapping[pg] = cands[0]["index"]
            used.add(cands[0]["index"])
    return mapping


# ===========================================================================
# MLP-split crossover: this rig's finding, or the offer to measure it
# ===========================================================================
_CROSSOVER_MOD: list = []   # memoised: [module] or [None]


def _crossover_module():
    """``sglang.srt.planner.crossover``, or None when it cannot be reached.

    Two ways in, because the dashboard is a standalone process that is often
    run against a checkout whose sglang is not the installed one: the package
    import first, then the module file next to this checkout loaded directly.
    The module is stdlib-only, so the direct load has nothing to satisfy.
    The panel degrades to "not available" rather than taking the server down.
    """
    if _CROSSOVER_MOD:
        return _CROSSOVER_MOD[0]
    mod = None
    try:
        import sglang.srt.planner.crossover as mod  # noqa: PLC0415
    except Exception:
        import importlib.util  # noqa: PLC0415

        src = os.path.join(
            HERE, "..", "..", "python", "sglang", "srt", "planner", "crossover.py"
        )
        if os.path.isfile(src):
            try:
                spec = importlib.util.spec_from_file_location(
                    "_rig_dashboard_crossover", src
                )
                mod = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = mod
                spec.loader.exec_module(mod)
            except Exception:
                mod = None
    _CROSSOVER_MOD.append(mod)
    return mod


def crossover_state(path=None):
    """What the dashboard may say about MLP concentration on this rig.

    Three outcomes, kept apart because they mean different things: no finding
    at all, a finding that exists but must not be used (stale, taken under
    throttling with no cache proof, or measured somewhere else), and a usable
    one. The first two carry the offer to run the measurement; none of them
    silently substitutes a number from another rig.
    """
    m = _crossover_module()
    if m is None:
        return {
            "state": "unavailable",
            "usable": False,
            "caveats": ["sglang.srt.planner.crossover is not importable here"],
            "offer": [],
        }
    return m.describe_evidence(m.load_finding(path))


def build_state():
    nvml, nvml_src = sample_nvml()
    info = scrape_server_info()
    metrics = scrape_metrics()
    plan = load_plan()

    # extract a few live server scalars from server_info if present
    server = None
    if info:
        sched = info.get("internal_states") or []
        s0 = sched[0] if isinstance(sched, list) and sched else {}
        server = {
            "up": True,
            "model": (info.get("served_model_name") or "").split("/")[-1],
            "tp_size": info.get("tp_size"),
            "dcp_size": info.get("dcp_size"),
            "rank_gpu_id": info.get("rank_gpu_id"),
            "rank_tp_ratio": info.get("rank_tp_ratio"),
            "rank_mlp_ratio": info.get("rank_mlp_ratio"),
            "rank_vocab_ratio": info.get("rank_vocab_ratio"),
            "max_total_num_tokens": info.get("max_total_num_tokens")
            or s0.get("max_total_num_tokens"),
            "version": info.get("version"),
        }
    return {
        "ts": __import__("time").time(),
        "nvml": nvml,
        "nvml_source": nvml_src,
        "server": server or {"up": False},
        "metrics": metrics,
        "plan": plan,
        # boot-time GPU index -> live NVML index (enumeration can diverge)
        "gpu_map": map_plan_gpus_to_nvml(plan, nvml),
        "crossover": crossover_state(_CFG.get("crossover_file")),
    }


# ===========================================================================
# HTTP
# ===========================================================================
class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/api/state"):
            try:
                self._send(200, json.dumps(build_state()), "application/json")
            except Exception as e:
                self._send(500, json.dumps({"error": str(e)}), "application/json")
            return
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
            return
        self._send(404, "not found", "text/plain")

    def log_message(self, *a):  # quiet
        pass


def main():
    ap = argparse.ArgumentParser(description="Rig-Dashboard server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument(
        "--sglang",
        default="http://127.0.0.1:30010",
        help="base URL of the sglang server (blank to disable scraping)",
    )
    ap.add_argument(
        "--boot-log",
        default=os.environ.get("RIG_BOOT_LOG", ""),
        help="path to the sglang boot log for the uneven-TP plan",
    )
    ap.add_argument(
        "--crossover-file",
        default=os.environ.get("RIG_CROSSOVER_FILE", ""),
        help="path to this rig's measured MLP-split crossover "
        "(default ~/.cache/sglang/mlp_crossover.json)",
    )
    for k in DEFAULT_GEOMETRY:
        ap.add_argument(f"--{k.replace('_', '-')}", type=int, default=None)
    args = ap.parse_args()

    _G.update(DEFAULT_GEOMETRY)
    for k in DEFAULT_GEOMETRY:
        v = getattr(args, k)
        if v is not None:
            _G[k] = v
    _CFG["sglang"] = args.sglang.rstrip("/") if args.sglang else ""
    _CFG["boot_log"] = args.boot_log
    _CFG["crossover_file"] = args.crossover_file

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Rig-Dashboard on http://{args.host}:{args.port}/")
    print(f"  sglang   : {_CFG['sglang'] or '(disabled)'}")
    print(f"  boot log : {_CFG['boot_log'] or '(none)'}")
    print(f"  crossover: {_CFG['crossover_file'] or '(default cache path)'}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
