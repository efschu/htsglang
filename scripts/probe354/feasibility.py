"""Feasibility arithmetic for a PINNED MLP vector (--rank-mlp-ratio).

The pin path in apply_auto_performance skips probe AND optimizer, so a
pinned phase-optimal vector gets none of the fundability checks the
optimizer applies to its own candidates. This reproduces those numbers
outside the boot: per-rank weight bytes, predicted KV capacity, and the
residual free VRAM the #264 check compares against the derived reserve
demand.

Usage:
  python feasibility.py <model-path> <vec> [<vec> ...] [-- <extra flags>]
where <vec> is "16,1,1" (or "base" for the plain VRAM-auto split).
"""

import logging
import sys

logging.basicConfig(level=logging.WARNING)


def main() -> int:
    argv = list(sys.argv[1:])
    extra = []
    if "--" in argv:
        i = argv.index("--")
        extra = argv[i + 1 :]
        argv = argv[:i]
    model_path, vecs = argv[0], argv[1:]

    from sglang.srt.server_args import prepare_server_args
    from sglang.srt import uneven_perf as up

    base_argv = [
        "--model-path",
        model_path,
        "--tp-size",
        "3",
        "--rank-gpu-id",
        "0,1,2",
        "--rank-tp-ratio",
        "auto",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--context-length",
        "32768",
        "--trust-remote-code",
        "--max-running-requests",
        "16",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--enable-metrics",
    ] + extra

    sa = prepare_server_args(base_argv)
    base_plan = list(sa.rank_tp_ratio)
    budgets = list(sa.rank_gpu_memory_mib)
    _profile, gpus = up.get_cached_hardware_profile()
    totals = [
        next(g["total_mib"] for g in gpus if g["cuda_index"] == gid)
        for gid in sa.rank_gpu_id
    ]
    counts = [sa.rank_gpu_id.count(gid) for gid in sa.rank_gpu_id]
    demand = [
        int(sa.derived_rank_auto_reserve_mib(totals[r], counts[r]))
        for r, gid in enumerate(sa.rank_gpu_id)
    ]

    model = up.PerfCostModel(up.PlanInputs.from_server_args(sa), base_plan, budgets)

    print(f"base plan   : {base_plan}")
    print(f"budgets MiB : {budgets}")
    print(f"NVML totals : {totals}")
    print(f"reserve dmd : {demand}")
    print()
    hdr = (
        f"{'vector':>12} {'MLP units':>16} {'weights GiB/rank':>28} "
        f"{'cap tokens':>26} {'residual MiB':>22} verdict"
    )
    print(hdr)
    for spec in vecs:
        vec = base_plan if spec == "base" else [int(x) for x in spec.split(",")]
        units = model.mlp_unit_partition(vec)
        wb = [b / 2**30 for b in model.per_rank_weight_bytes(vec)]
        pred = model.predict_capacity(vec)
        tv = pred["token_vector"] or [1] * len(vec)
        res = model.residual_free_mib(vec, totals, counts, tv)
        # Hard physical check: weights alone must fit the rank's budget.
        over = [r for r in range(len(vec)) if wb[r] * 1024 > budgets[r]]
        under = [r for r in range(len(vec)) if res[r] < demand[r]]
        verdict = "OK"
        if over:
            verdict = f"WEIGHTS-OVERFLOW rank {over}"
        elif under:
            verdict = f"under reserve demand on rank {under}"
        print(
            f"{spec:>12} {str(units):>16} "
            f"{str([round(x, 2) for x in wb]):>28} "
            f"{str([int(x) for x in pred['p']]):>26} "
            f"{str([int(x) for x in res]):>22} {verdict}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
