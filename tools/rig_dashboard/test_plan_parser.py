"""Unit tests for the boot-log plan parser, run against the REAL M38 boot logs.

    python3 -m pytest test_plan_parser.py -q      # or:
    python3 test_plan_parser.py                    # plain-stdlib runner
"""

import os

from plan_parser import derive_head_split, largest_remainder, parse_plan, parse_plan_file

LOGDIR = "/root/.claude/jobs/1481bb40/tmp/matrix_logs"


def _log(name):
    return os.path.join(LOGDIR, name, name + "_server.log")


# ---------------------------------------------------------------------------
def test_awq_full_plan():
    p = parse_plan_file(_log("htsglang_awq_tp3_V2"))
    assert p["tp_size"] == 3 and p["dcp_size"] == 3
    assert p["rank_gpu_id"] == [0, 1, 2]
    assert p["rank_tp_ratio"] == [29607, 18280, 18280]
    assert p["memory_budgets_mib"] == [29607, 18280, 18280]
    assert p["reserve_mib"] == {0: 3000, 1: 2200, 2: 2200}
    assert p["mlp_vector"] == [5, 1, 1]
    assert p["mlp_units"] == [388, 78, 78]
    assert p["rank_vocab_ratio"] == [13, 6, 6]
    # active per-64 token ownership (NOT the SGLANG_UNEVEN_TOKEN_VECTOR hint)
    assert p["token_vector"] == [30, 17, 17]
    assert p["token_units"] == 64
    assert p["profiled_capacity"] == [235938, 158388, 201452]
    assert p["max_total_num_tokens"] == 441536
    # per-rank measured VRAM (5090 rank0 carries the fat weight shard)
    assert p["ranks"][0]["weight_gb"] == 14.91
    assert p["ranks"][0]["draft_weight_gb"] == 2.72
    assert p["ranks"][0]["kv_tokens"] == 206970
    assert round(p["ranks"][0]["kv_gb"], 2) == 6.32
    assert p["ranks"][1]["weight_gb"] == 6.48
    assert p["ranks"][2]["kv_tokens"] == 117283
    # gpu inventory from the auto-performance block
    names = [g["name"] for g in p["gpus"]]
    assert "NVIDIA GeForce RTX 5090" in names[0]
    assert p["gpus"][0]["membw_gbs"] == 1558.0


def test_fp8_plan():
    p = parse_plan_file(_log("htsglang_fp8_tp3_V2"))
    assert p["mlp_units"] == [91, 23, 22]
    assert p["token_vector"] == [30, 17, 17]
    assert p["ranks"][0]["weight_gb"] == 16.88  # 14.10 main + 2.78? summed loads


def test_gguf_pinned_partial():
    # gguf run used a PIN HINT: no MLP-optimizer block, no auto-perf gpus.
    # Parser must degrade gracefully and still recover the token vector + ranks.
    p = parse_plan_file(_log("htsglang_q6_tp3_V2"))
    assert p["tp_size"] == 3
    assert p.get("mlp_units") is None
    assert p["token_vector"] == [29, 17, 18]
    assert p["ranks"][0]["kv_tokens"] == 109301
    assert p.get("gpus", []) == []  # no auto-performance block in a pinned boot


def test_v1_minimal():
    # earliest boot: no vocab/token block logged at all -> still parses ranks
    p = parse_plan_file(_log("htsglang_q6_tp3_V1"))
    assert p["tp_size"] == 3
    assert p.get("token_vector") is None
    assert p["ranks"][0]["weight_gb"] == 12.41


def test_token_vector_uses_active_not_recommendation():
    # guard: the log line contains BOTH SGLANG_UNEVEN_TOKEN_VECTOR=21,19,24
    # (a restart hint) and active vector [30,17,17]; we must take the active one.
    line = (
        "[2026-07-17 05:20:13 TP0] Uneven DCP: restart with "
        "SGLANG_UNEVEN_TOKEN_VECTOR=21,19,24 to raise max_total_num_tokens from "
        "441536 to ~630784 (per-rank profiled capacity [206995, 189558, 240912]; "
        "active vector [30, 17, 17] leaves ranks idle)."
    )
    p = parse_plan(line)
    assert p["token_vector"] == [30, 17, 17]
    assert p["profiled_capacity"] == [206995, 189558, 240912]


def test_streaming_stops_at_boot_complete_and_byte_cap(tmp_path=None):
    # parser must stop at the ready line (live logs keep growing after it --
    # a 17 GB crash-loop log was observed live) and honor the byte cap.
    import tempfile

    from plan_parser import MAX_PARSE_BYTES  # noqa: F401  (exists)

    pre = "[2026-07-17 05:20:14 TP0] KV Cache is allocated. dtype: torch.float8_e4m3fn, #tokens: 100, K size: 1.00 GB, V size: 1.00 GB\n"
    done = "[2026-07-17 05:20:22] The server is fired up and ready to roll!\n"
    post = "[2026-07-17 05:20:23 TP0] KV Cache is allocated. dtype: x, #tokens: 999999, K size: 99.00 GB, V size: 99.00 GB\n"
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
        f.write(pre + done + post)
        path = f.name
    p = parse_plan_file(path)
    assert p["boot_complete"] is True
    assert p["ranks"][0]["kv_tokens"] == 100  # post-ready line ignored
    # byte cap: parsing stops after the cap (first line always yielded), so
    # neither the ready line nor the post-ready line is ever seen
    p2 = parse_plan_file(path, max_bytes=10)
    assert p2.get("boot_complete") is None
    assert p2["ranks"][0]["kv_tokens"] == 100
    os.unlink(path)


def test_replicated_kv_and_moe_expert_lines():
    # exact line shapes from the LIVE battery boot (m33_a3b_gguf_boot10.log)
    text = "\n".join(
        [
            "[2026-07-17 14:46:33 TP1] REPLICATED-KV geometry active for "
            "attention layer 3: kv_heads=2 < tp_size=3; all kv heads on every "
            "rank, q heads split [8, 4, 4] (units of 2).",
            "[2026-07-17 14:46:33 TP0] GGUF MoE uneven TP: expert-dim sharding "
            "active — rank 0 owns experts [0, 114) of 256 (full intermediate "
            "512 per expert).",
            "[2026-07-17 14:46:33 TP1] GGUF MoE uneven TP: expert-dim sharding "
            "active — rank 1 owns experts [114, 185) of 256 (full intermediate "
            "512 per expert).",
            "[2026-07-17 14:46:33 TP2] GGUF MoE uneven TP: expert-dim sharding "
            "active — rank 2 owns experts [185, 256) of 256 (full intermediate "
            "512 per expert).",
        ]
    )
    p = parse_plan(text)
    assert p["attn_replicated"] == {
        "kv_heads": 2,
        "tp_size": 3,
        "q_split": [8, 4, 4],
        "unit": 2,
    }
    assert p["moe_experts"][0] == {"start": 0, "end": 114, "total": 256}
    assert p["moe_experts"][2] == {"start": 185, "end": 256, "total": 256}


def test_live_a3b_boot10_end_to_end():
    # the actual live-battery log, if still present (skip silently if rotated)
    path = "/root/.claude/jobs/1481bb40/tmp/m33_a3b_gguf_boot10.log"
    if not os.path.exists(path):
        return
    p = parse_plan_file(path)
    assert p["attn_replicated"]["q_split"] == [8, 4, 4]
    assert p["moe_experts"][1]["end"] - p["moe_experts"][1]["start"] == 71


def test_largest_remainder_and_head_split():
    assert largest_remainder(64, [30, 17, 17]) == [30, 17, 17]
    assert sum(largest_remainder(24, [29607, 18280, 18280])) == 24
    # whole-GQA groups: kv=4 over ratio -> [2,1,1]; q=24 -> 6 per kv -> [12,6,6]
    kv = largest_remainder(4, [29607, 18280, 18280])
    assert kv == [2, 1, 1]
    q = [k * (24 // 4) for k in kv]
    assert q == [12, 6, 6]
    assert derive_head_split(0, []) == []


def test_gpu_map_resolves_enumeration_divergence():
    # Live-observed: boot-time GPU 0 = RTX 5090, but system NVML has the 5090
    # at index 1. Budget 29607 MiB can only fit the 32 GB card.
    from server import map_plan_gpus_to_nvml

    nvml = [
        {"index": 0, "name": "NVIDIA GeForce RTX 3080", "mem_total_mib": 20480},
        {"index": 1, "name": "NVIDIA GeForce RTX 5090", "mem_total_mib": 32607},
        {"index": 2, "name": "NVIDIA GeForce RTX 3080", "mem_total_mib": 20480},
    ]
    # with names from the auto-performance block (unique 5090 name match)
    plan = {
        "available": True,
        "rank_gpu_id": [0, 1, 2],
        "memory_budgets_mib": [29607, 18280, 18280],
        "gpus": [
            {"gpu_index": 0, "name": "NVIDIA GeForce RTX 5090"},
            {"gpu_index": 1, "name": "NVIDIA GeForce RTX 3080"},
            {"gpu_index": 2, "name": "NVIDIA GeForce RTX 3080"},
        ],
    }
    m = map_plan_gpus_to_nvml(plan, nvml)
    assert m[0] == 1  # 5090 by unique name
    assert sorted([m[1], m[2]]) == [0, 2]
    # without names (pinned boot): budget best-fit still finds the 5090
    plan2 = {
        "available": True,
        "rank_gpu_id": [0, 1, 2],
        "memory_budgets_mib": [29607, 18280, 18280],
    }
    m2 = map_plan_gpus_to_nvml(plan2, nvml)
    assert m2[0] == 1
    assert map_plan_gpus_to_nvml({"available": False}, nvml) == {}


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            ok += 1
        except Exception:
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{ok}/{len(fns)} passed")
    raise SystemExit(0 if ok == len(fns) else 1)
