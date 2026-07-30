"""Boot-log parser for the uneven-TP "materialized plan".

The heterogeneous rank->GPU split (rank_tp_ratio / rank_mlp_ratio /
rank_vocab_ratio / DCP token vector) is decided at boot time and printed to the
sglang server log. Those materialized numbers do NOT live in ServerArgs, so
``/server_info`` cannot expose them (they are computed inside the TP worker
processes). This module recovers them by parsing a boot log, which means the
dashboard works against *any* server -- modified or stock -- as long as it has
its log file.

Pure stdlib, no sglang imports, so it is trivially unit-testable against the
real logs under matrix_logs/ without a GPU.

Public entry point: ``parse_plan(text) -> dict``.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

# ---- line patterns (anchored to the real M38 boot-log format) --------------

# strip the "[2026-07-17 05:20:13 TP0]" / "[2026-07-17 05:20:13]" prefix,
# returning (tp_rank_or_None, rest)
_PREFIX = re.compile(r"^\[[\d\- :]+?(?:\s+TP(\d+))?\]\s*(.*)$")


def _ints(s: str) -> List[int]:
    return [int(x) for x in re.findall(r"-?\d+", s)]


def _floats(s: str) -> List[float]:
    return [float(x) for x in re.findall(r"-?\d+\.?\d*", s)]


def _split_prefix(line: str):
    m = _PREFIX.match(line.strip())
    if not m:
        return None, line.strip()
    return (int(m.group(1)) if m.group(1) is not None else None), m.group(2)


def _ensure(ranks: Dict[int, dict], r: int) -> dict:
    return ranks.setdefault(r, {"rank": r})


# All plan-relevant lines appear during boot; once the server logs this line
# the plan is final and parsing can stop (critical for huge, still-growing
# logs -- a crash-looping boot has been observed to spew a 17 GB log).
_BOOT_DONE = "The server is fired up and ready to roll"

#: hard byte cap for logs that never reach the ready line (crash loops)
MAX_PARSE_BYTES = 128 * 1024 * 1024


def parse_plan(text: str) -> dict:
    return parse_plan_lines(text.splitlines())


def parse_plan_lines(lines) -> dict:
    """Parse a full sglang boot log into a structured uneven-TP plan.

    Returns a dict with keys (any may be missing if the log predates that
    stage / is not an uneven-TP run):

      model, quant, tp_size, dcp_size
      rank_gpu_id            [int]   physical GPU index per rank
      rank_tp_ratio          [int]
      rank_mlp_ratio         [int]
      rank_vocab_ratio       [int]
      memory_budgets_mib     [int]   per-rank VRAM budget
      reserve_mib            {gpu:int}
      gpus                   [{rank,gpu_index,name,gemm_tflops,membw_gbs}]
      mlp_vector             [int]   the chosen MLP concentration lever
      mlp_units              [int]   materialized MLP units per rank
      token_vector           [int]   active per-64 token ownership per rank
      token_units            int     block size the vector sums to (usually 64)
      profiled_capacity      [int]   per-rank profiled token capacity
      max_total_num_tokens   int
      ranks                  {r: {weight_gb, draft_weight_gb, kv_tokens,
                                  kv_gb, mamba_gb, free_gb_end, ...}}
    """
    plan: dict = {}
    ranks: Dict[int, dict] = {}

    for raw in lines:
        if _BOOT_DONE in raw:
            plan["boot_complete"] = True
            break
        tp, body = _split_prefix(raw)

        # --- global (main-process) planning lines ---------------------------
        if "derived memory budgets" in body:
            m = re.search(r"derived memory budgets \[([\d,\s]+)\]", body)
            if m:
                plan["memory_budgets_mib"] = _ints(m.group(1))
            m2 = re.search(r"reserve per GPU: \{([^}]*)\}", body)
            if m2:
                plan["reserve_mib"] = {
                    int(k): int(v)
                    for k, v in re.findall(r"(\d+)\s*:\s*(\d+)", m2.group(1))
                }
            continue

        # The GEMM figure carries the LANE it was measured in from #298a
        # ("[fp8 native (_scaled_mm)]"); older logs have no bracket, so the
        # group is optional and the dashboard keeps parsing both.
        m = re.match(
            r"rank (\d+) -> GPU (\d+) \(([^)]+)\): GEMM ([\d.]+) TFLOPS"
            r"(?: \[(.+?)\])?, membw ([\d.]+) GB/s",
            body,
        )
        if m:
            plan.setdefault("gpus", []).append(
                {
                    "rank": int(m.group(1)),
                    "gpu_index": int(m.group(2)),
                    "name": m.group(3),
                    "gemm_tflops": float(m.group(4)),
                    "gemm_lane": m.group(5),
                    "membw_gbs": float(m.group(6)),
                }
            )
            continue

        if "CHOSEN MLP vector" in body:
            mv = re.search(r"CHOSEN MLP vector:\s*([\d,\s]+?)\s*\(", body)
            if mv:
                plan["mlp_vector"] = _ints(mv.group(1))
            mu = re.search(r"materialized units \[([\d,\s]+)\]", body)
            if mu:
                plan["mlp_units"] = _ints(mu.group(1))
            pc = re.search(r"per-rank capacity \[([\d,\s]+)\]", body)
            if pc:
                plan["profiled_capacity"] = _ints(pc.group(1))
            continue

        # fallback MLP source when no optimizer ran (VRAM-auto reference line)
        if "materialized MLP units" in body and "mlp_units" not in plan:
            mu = re.search(r"materialized MLP units \[([\d,\s]+)\]", body)
            if mu:
                plan["mlp_units"] = _ints(mu.group(1))
            continue

        if "rank-vocab-ratio auto ->" in body:
            m = re.search(r"auto ->\s*([\d,\s]+?)\s*\(", body)
            if m:
                plan["rank_vocab_ratio"] = _ints(m.group(1))
            continue

        if "Uneven DCP: restart with SGLANG_UNEVEN_TOKEN_VECTOR" in body:
            # The "active vector [...]" is the CURRENTLY materialized per-64
            # token ownership; the SGLANG_UNEVEN_TOKEN_VECTOR value is only a
            # tuning *recommendation* for a future restart -- do not use it.
            av = re.search(r"active vector \[([\d,\s]+)\]", body)
            if av:
                vec = _ints(av.group(1))
                plan["token_vector"] = vec
                plan["token_units"] = sum(vec)
            pc = re.search(r"profiled capacity \[([\d,\s]+)\]", body)
            if pc and "profiled_capacity" not in plan:
                plan["profiled_capacity"] = _ints(pc.group(1))
            mm = re.search(r"raise max_total_num_tokens from (\d+)", body)
            if mm:
                plan.setdefault("max_total_num_tokens", int(mm.group(1)))
            continue

        if body.startswith("server_args=ServerArgs("):
            _parse_server_args(body, plan)
            continue

        # Materialized attention geometry under replicated KV (TP > kv_heads):
        # "REPLICATED-KV geometry active for attention layer 3: kv_heads=2 <
        #  tp_size=3; all kv heads on every rank, q heads split [8, 4, 4]
        #  (units of 2)."  -- authoritative, overrides any derived split.
        if "REPLICATED-KV geometry active" in body and "attn_replicated" not in plan:
            m = re.search(
                r"kv_heads=(\d+) < tp_size=(\d+).*?"
                r"q heads split \[([\d,\s]+)\] \(units of (\d+)\)",
                body,
            )
            if m:
                plan["attn_replicated"] = {
                    "kv_heads": int(m.group(1)),
                    "tp_size": int(m.group(2)),
                    "q_split": _ints(m.group(3)),
                    "unit": int(m.group(4)),
                }
            continue

        # Materialized MoE expert ownership (GGUF MoE uneven TP):
        # "... rank 1 owns experts [114, 185) of 256 (full intermediate ...)"
        if "owns experts [" in body:
            m = re.search(r"rank (\d+) owns experts \[(\d+), (\d+)\) of (\d+)", body)
            if m:
                r, a, b, tot = (int(m.group(i)) for i in range(1, 5))
                moe = plan.setdefault("moe_experts", {})
                moe[r] = {"start": a, "end": b, "total": tot}
            continue

        # --- per-rank (TP worker) lines -------------------------------------
        if tp is None:
            continue
        rk = _ensure(ranks, tp)

        if "Load weight end" in body:
            mu = re.search(r"mem usage=([\d.]+) GB", body)
            gb = float(mu.group(1)) if mu else 0.0
            if "MTP" in body or "Draft" in body or "CausalLMMTP" in body:
                rk["draft_weight_gb"] = rk.get("draft_weight_gb", 0.0) + gb
            else:
                rk["weight_gb"] = rk.get("weight_gb", 0.0) + gb
            continue

        if "KV Cache is allocated" in body:
            tk = re.search(r"#tokens:\s*(\d+)", body)
            ks = re.search(r"K size:\s*([\d.]+) GB", body)
            vs = re.search(r"V size:\s*([\d.]+) GB", body)
            tokens = int(tk.group(1)) if tk else 0
            gb = (float(ks.group(1)) if ks else 0.0) + (
                float(vs.group(1)) if vs else 0.0
            )
            # Largest allocation per rank is the real KV pool; smaller later
            # allocations are the 1-layer draft/MTP KV pool.
            if gb >= rk.get("kv_gb", 0.0):
                if "kv_gb" in rk:  # demote the previous (smaller) pool to draft
                    rk["draft_kv_gb"] = rk.get("draft_kv_gb", 0.0) + rk["kv_gb"]
                rk["kv_tokens"] = tokens
                rk["kv_gb"] = gb
            else:
                rk["draft_kv_gb"] = rk.get("draft_kv_gb", 0.0) + gb
            continue

        if "Mamba Cache is allocated" in body:
            comps = re.findall(r"([\d.]+)\s*GB", body)
            rk["mamba_gb"] = sum(float(c) for c in comps)
            ms = re.search(r"max_mamba_cache_size:\s*(\d+)", body)
            if ms:
                rk["mamba_slots"] = int(ms.group(1))
            continue

        if "Memory pool end" in body:
            m = re.search(r"avail mem=([\d.]+) GB", body)
            if m:
                rk["free_gb_end"] = float(m.group(1))  # last wins => final free
            continue

        if "Load weight begin" in body and "avail_gb_start" not in rk:
            m = re.search(r"avail mem=([\d.]+) GB", body)
            if m:
                rk["avail_gb_start"] = float(m.group(1))
            continue

    if ranks:
        plan["ranks"] = {r: ranks[r] for r in sorted(ranks)}
    return plan


def _parse_server_args(body: str, plan: dict) -> None:
    def _grab_list(key: str) -> Optional[List[int]]:
        m = re.search(rf"{key}=\[([\d,\s]*)\]", body)
        return _ints(m.group(1)) if m and m.group(1).strip() else None

    def _grab_scalar(key: str):
        m = re.search(rf"{key}=([^\s,)]+)", body)
        return m.group(1).strip("'\"") if m else None

    for key, dst in [
        ("rank_gpu_id", "rank_gpu_id"),
        ("rank_gpu_memory_mib", "rank_gpu_memory_mib"),
        ("rank_tp_ratio", "rank_tp_ratio"),
        ("rank_mlp_ratio", "rank_mlp_ratio"),
        ("rank_vocab_ratio", "rank_vocab_ratio"),
    ]:
        v = _grab_list(key)
        if v is not None:
            plan[dst] = v

    for key in ["tp_size", "dcp_size", "pp_size", "dp_size", "ep_size"]:
        s = _grab_scalar(key)
        if s and s.lstrip("-").isdigit():
            plan[key] = int(s)

    for key, dst in [
        ("served_model_name", "model"),
        ("model_path", "model_path"),
        ("quantization", "quant"),
        ("kv_cache_dtype", "kv_cache_dtype"),
        ("speculative_algorithm", "spec_algorithm"),
    ]:
        s = _grab_scalar(key)
        if s and s != "None":
            plan.setdefault(dst, s)


def parse_plan_file(path: str, max_bytes: int = MAX_PARSE_BYTES) -> dict:
    """Stream-parse a boot log. Never slurps the file: live crash-loop logs
    can reach tens of GB, which would OOM a naive read(). Stops at the
    boot-complete line or after ``max_bytes``, whichever comes first."""

    def _lines():
        seen = 0
        with open(path, "r", errors="replace") as f:
            for line in f:
                seen += len(line)
                yield line
                if seen > max_bytes:
                    return

    return parse_plan_lines(_lines())


# ---- geometry helpers (derive per-rank head/unit splits) -------------------

def largest_remainder(total: int, weights: List[float]) -> List[int]:
    """Split ``total`` into per-rank integer shares proportional to ``weights``
    using the largest-remainder method (sums exactly to ``total``)."""
    s = float(sum(weights)) or 1.0
    exact = [total * w / s for w in weights]
    floor = [int(x) for x in exact]
    rem = total - sum(floor)
    order = sorted(range(len(weights)), key=lambda i: exact[i] - floor[i], reverse=True)
    for i in range(rem):
        floor[order[i]] += 1
    return floor


def derive_head_split(total_heads: int, ratio: List[int]) -> List[int]:
    """Per-rank whole-head split proportional to the shard ratio."""
    if not ratio:
        return []
    return largest_remainder(total_heads, ratio)


if __name__ == "__main__":  # tiny CLI: python plan_parser.py <logfile>
    import json
    import sys

    print(json.dumps(parse_plan_file(sys.argv[1]), indent=2))
