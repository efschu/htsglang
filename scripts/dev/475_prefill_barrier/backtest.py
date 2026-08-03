"""#475 backtest: predicted prefill gain against the measured prefill window.

Desk-only (``CUDA_VISIBLE_DEVICES=99``). Prices the four recorded
2026-08-02 concentration arms with the pre-#475 cost model (lockstep max once
per step) and with the shipped one (lockstep max per barrier), and prints
both against the GPU-window delta that ``window_accounting.py`` extracts from
the same boots' CollectiveClock lines.

    CUDA_VISIBLE_DEVICES=99 PYTHONPATH=python \\
        python scripts/dev/475_prefill_barrier/backtest.py

The measured column is the prefill WINDOW (summed ``gpu-ms`` over the full
2048-token prefill batches of the s=1 probe, normalised per 1000 prompt
tokens), not the probe's host-side tok/s -- the two disagree by up to 9
points on the same arm pair, and the window is the quantity the cost model
predicts.
"""

import argparse
import os
import sys

CACHE_DEFAULT = "/spinning/llm_stuff/club-3090/models-cache"

#: Both boots' derived per-rank budgets, which are also the base MLP plan.
BUDGETS = [28447, 16320, 16320]
MIN_LINK = 5.1

#: (label, checkpoint dir, probed per-rank GEMM TFLOPS, candidate vector,
#:  measured base window ms/1k tok, measured candidate window ms/1k tok,
#:  provenance)
ARMS = (
    (
        "FP8  base -> 10,1,1  (#424, NCCL)",
        "Qwen3.6-27B-FP8", [563.1, 57.6, 60.8], [10, 1, 1], 803.3, 697.2,
        "2026-08-02_424_phase_record_bench: fp8_decode vs fp8_prefill",
    ),
    (
        "FP8  base -> 10,1,1  (#435, BAR1)",
        "Qwen3.6-27B-FP8", [563.1, 57.6, 60.8], [10, 1, 1], 744.2, 630.8,
        "2026-08-02_435_coupling_fp8bar1: fp8_decode_bar1 vs fp8_prefill_bar1",
    ),
    (
        "INT8 base -> 10,1,1  (#424)",
        "Qwen3.6-27B-INT8-W8A8", [681.4, 187.6, 183.8], [10, 1, 1], 527.4, 533.8,
        "2026-08-02_424_phase_record_bench: int8_decode vs int8_prefill",
    ),
    (
        "INT8 base ->  8,1,1  (#433, solved)",
        "Qwen3.6-27B-INT8-W8A8", [681.4, 187.6, 183.8], [8, 1, 1], 527.4, 518.1,
        "#424 int8_decode vs 2026-08-02_433_int8_prefill: int8_prefill_solved",
    ),
)


def build(model_path):
    from sglang.srt.uneven_perf import PerfCostModel, PlanInputs

    pi = PlanInputs(
        tp_size=3, model_path=model_path, kv_cache_dtype="fp8_e4m3",
        speculative_algorithm="NEXTN", speculative_num_draft_tokens=4,
        rank_gpu_id=[0, 1, 2], effective_vram_mib=list(BUDGETS),
        rank_tp_ratio=list(BUDGETS),
    )
    return PerfCostModel(pi, list(BUDGETS), list(BUDGETS))


def gains(m, vec, gemm):
    """(pre-#475 gain %, shipped gain %, barrier skew ms per 1000 tokens)."""
    k = m.calibration.prefill_invariant / (1.0 - m.calibration.prefill_invariant)
    t_ar = m.n_layers * 2 * m.hidden * 2 * 2 * 2 / 3 / (MIN_LINK * 1e9)

    def old(v):
        return max(m.per_rank_prefill_compute_times(list(v), gemm), default=0.0)

    def new(v):
        return m.prefill_lockstep_compute_time(list(v), gemm)

    out = []
    for term in (old, new):
        sb, sc = term(BUDGETS) + t_ar, term(vec) + t_ar
        out.append(((sb * (1 + k)) / (sc + k * sb) - 1.0) * 100.0)
    skew = (m.prefill_barrier_skew(list(vec), gemm)
            - m.prefill_barrier_skew(list(BUDGETS), gemm)) * 1e6
    return out[0], out[1], skew


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-cache", default=CACHE_DEFAULT)
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "99")

    print(f"{'arm':38s} {'pre-#475':>9s} {'shipped':>9s} {'measured':>9s} "
          f"{'skew ms/1k':>11s}")
    err_old, err_new = [], []
    for label, ckpt, gemm, vec, base_ms, cand_ms, _src in ARMS:
        path = os.path.join(args.models_cache, ckpt)
        if not os.path.isdir(path):
            print(f"{label:38s}   (checkpoint not present: {path})")
            continue
        m = build(path)
        g_old, g_new, skew = gains(m, vec, gemm)
        meas = (base_ms / cand_ms - 1.0) * 100.0
        err_old.append(g_old - meas)
        err_new.append(g_new - meas)
        print(f"{label:38s} {g_old:+8.1f}% {g_new:+8.1f}% {meas:+8.1f}% "
              f"{skew:10.1f}")
    if err_new:
        def rms(xs):
            return (sum(x * x for x in xs) / len(xs)) ** 0.5
        print(f"\n{'rms error (points)':38s} {rms(err_old):9.1f} "
              f"{rms(err_new):9.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
