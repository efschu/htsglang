"""#485 backtest: the joint per-family cut, against every measured point.

Extended by #492, which corrects slice 1's central mistake: sections 1-4
search HEAD partitions only, and on a checkpoint whose kv-head count does not
divide across the ranks that space is empty, which #485 wrote down as "the
attention family is grid-pinned". Sections 5-6 price the axis it forgot --
replication + token-sharding, the fork's own #62/#116 machinery -- and the
head-only space is executed as the falsifier that it cannot move the family
by construction. Sections 1-4 must print exactly what they printed before.

Desk-only (``CUDA_VISIBLE_DEVICES=99``). Sections, in the order a
reader should refuse to believe them:

1. REGRESSION. The four measured 2026-08-02 concentration arms of #475,
   re-priced with the joint machinery present but no attention vector in
   play. Every number must equal the #475 column to the last digit -- a
   joint solver that moves the single-family points has broken the anchor it
   is built on, and nothing after this section is worth reading.
2. JOINT. The pair space, for both checkpoints: the best MLP-only vector the
   #475 model finds, the best (MLP, attention/GDN) pair, and the per-family
   barrier pacers of each.
3. FALSIFIER. A deliberately DETUNED attention vector paired with the
   optimal MLP vector must price WORSE than the aligned pair. If it does
   not, the objective is not reading the attention family at all.
4. LANE BRACKET. Which RATE paces the attention barrier.
5. REPLICATION AXIS (#492). Which VECTOR distributes it. Prints how many
   distinct attention partitions the whole #485 head space realizes (one, on
   this checkpoint -- the executed falsifier), the geometric core/projection
   crossover depth, and the CORE-FREE / CORE-PACED bracket.
6. FALSIFIER. A detuned TOKEN vector must price WORSE than the aligned one.

    CUDA_VISIBLE_DEVICES=99 PYTHONPATH=python \\
        python scripts/dev/485_joint_phase/backtest_joint.py

Everything is the prefill WINDOW model, the quantity #475 anchored against
the CollectiveClock lines; no boot was run for any of it, and every number
printed here is DESK/PREDICTED.
"""

import argparse
import math
import os
import sys

CACHE_DEFAULT = "/spinning/llm_stuff/club-3090/models-cache"

#: Both boots' derived per-rank budgets, which are also the base MLP plan.
BUDGETS = [28447, 16320, 16320]
MIN_LINK = 5.1

#: Per-rank GEMM lane rates, from the two boots' own plan logs (#475 §2).
GEMM = {
    "INT8": [681.4, 187.6, 183.8],
    "FP8": [563.1, 57.6, 60.8],
}
CHECKPOINT = {
    "INT8": "Qwen3.6-27B-INT8-W8A8",
    "FP8": "Qwen3.6-27B-FP8",
}
#: Measured decode-shaped GEMV rate of the same three cards (#231 probe
#: group "membw"), used ONLY for the lane bracket -- never as a mass.
GEMV = [900.0, 420.0, 420.0]
#: The generic dense-bf16 GEMM lane of the same three cards
#: (test_prefill_calibration's refreshed stage-0 fixture). The GDN family is
#: BF16-resident in both checkpoints, so this is its physical lane.
BF16_DENSE = [233.91, 63.17, 61.24]

#: (label, checkpoint key, candidate vector, measured base window ms/1k tok,
#:  measured candidate window ms/1k tok)
ARMS = (
    ("FP8  base -> 10,1,1  (#424, NCCL)", "FP8", [10, 1, 1], 803.3, 697.2),
    ("FP8  base -> 10,1,1  (#435, BAR1)", "FP8", [10, 1, 1], 744.2, 630.8),
    ("INT8 base -> 10,1,1  (#424)", "INT8", [10, 1, 1], 527.4, 533.8),
    ("INT8 base ->  8,1,1  (#433, solved)", "INT8", [8, 1, 1], 527.4, 518.1),
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


def gain(m, vec, gemm, attn=None):
    base = m.prefill_time_model(list(BUDGETS), gemm, MIN_LINK)
    cand = m.prefill_time_model(list(vec), gemm, MIN_LINK, None, attn)
    return base / cand - 1.0


def old_gain(m, vec, gemm):
    """The pre-#475 term (one max at the end of the step), for the columns
    NOTE_475 §4 prints."""
    k = m.calibration.prefill_invariant / (1.0 - m.calibration.prefill_invariant)
    t_ar = m.n_layers * 2 * m.hidden * 2 * 2 * 2 / 3 / (MIN_LINK * 1e9)
    sb = max(m.per_rank_prefill_compute_times(list(BUDGETS), gemm)) + t_ar
    sc = max(m.per_rank_prefill_compute_times(list(vec), gemm)) + t_ar
    return (sb * (1 + k)) / (sc + k * sb) - 1.0


def pacers(m, vec, gemm, attn=None):
    fam = m.per_family_prefill_compute_times(list(vec), gemm, None, attn)
    return "/".join(
        f"{n}:r{t.index(max(t))}" for n, t in sorted(fam.items()) if max(t) > 0
    )


def best_pair(m, gemm, mlp_grid, attn_grid):
    best = None
    for a in attn_grid:
        for v in mlp_grid:
            g = gain(m, v, gemm, a)
            if best is None or g > best[0]:
                best = (g, v, a)
    return best


def main():
    from sglang.srt.uneven_perf import _attn_candidates, _mlp_candidates

    ap = argparse.ArgumentParser()
    ap.add_argument("--models-cache", default=CACHE_DEFAULT)
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "99")

    models = {}
    for key, ckpt in CHECKPOINT.items():
        path = os.path.join(args.models_cache, ckpt)
        if os.path.isdir(path):
            models[key] = build(path)
        else:
            print(f"(checkpoint not present: {path})")

    print("== 1. REGRESSION: the #475 arms, joint machinery present, no "
          "attention vector\n")
    print(f"{'arm':38s} {'pre-#475':>9s} {'shipped':>9s} {'measured':>9s} "
          f"{'skew ms/1k':>11s}")
    errs = []
    for label, key, vec, base_ms, cand_ms in ARMS:
        if key not in models:
            continue
        m, gemm = models[key], GEMM[key]
        g_old, g_new = old_gain(m, vec, gemm) * 100, gain(m, vec, gemm) * 100
        skew = (
            m.prefill_barrier_skew(list(vec), gemm)
            - m.prefill_barrier_skew(list(BUDGETS), gemm)
        ) * 1e6
        meas = (base_ms / cand_ms - 1.0) * 100.0
        errs.append(g_new - meas)
        print(f"{label:38s} {g_old:+8.1f}% {g_new:+8.1f}% {meas:+8.1f}% "
              f"{skew:10.1f}")
    if errs:
        rms = (sum(e * e for e in errs) / len(errs)) ** 0.5
        print(f"\n{'rms error (points)':38s} {'':9s} {rms:9.1f}")

    print("\n== 2. JOINT: single-family best vs pair best\n")
    for key, m in models.items():
        gemm = GEMM[key]
        mlp_grid = [list(c) for c in _mlp_candidates(m, gemm, BUDGETS)]
        attn_grid = [list(c) for c in _attn_candidates(m, gemm, BUDGETS)]
        g1, v1, _ = best_pair(m, gemm, mlp_grid, [None])
        g2, v2, a2 = best_pair(m, gemm, mlp_grid, attn_grid)
        skew1 = m.prefill_barrier_skew(v1, gemm) * 1e6
        skew2 = m.prefill_barrier_skew(v2, gemm, None, a2) * 1e6
        skew0 = m.prefill_barrier_skew(list(BUDGETS), gemm) * 1e6
        print(f"{key}: attention/GDN grids {m.attn_units} kv-head units, "
              f"{m.gdn_units} GDN k-head units; "
              f"{len(attn_grid)} attention candidates {attn_grid}")
        print(f"  base           {str(BUDGETS):20s}            "
              f"skew {skew0:6.2f} us/tok  pacers {pacers(m, BUDGETS, gemm)}")
        print(f"  MLP-only  best {str(v1):20s} {g1 * 100:+6.2f}%  "
              f"skew {skew1:6.2f} us/tok  pacers {pacers(m, v1, gemm)}")
        print(f"  JOINT     best {str(v2):20s} {g2 * 100:+6.2f}%  "
              f"skew {skew2:6.2f} us/tok  pacers "
              f"{pacers(m, v2, gemm, a2)}  + attn/GDN {a2} "
              f"-> GDN units {m.gdn_unit_partition(a2)}")
        print(f"  joint delta over the single-family cut: "
              f"{(g2 - g1) * 100:+.2f} points\n")

    print("== 3. FALSIFIER: a detuned attention vector must price WORSE\n")
    ok = True
    for key, m in models.items():
        gemm = GEMM[key]
        mlp_grid = [list(c) for c in _mlp_candidates(m, gemm, BUDGETS)]
        attn_grid = [list(c) for c in _attn_candidates(m, gemm, BUDGETS)]
        g2, v2, a2 = best_pair(m, gemm, mlp_grid, attn_grid)
        # Detuned: the aligned vector REVERSED, i.e. attention mass pushed
        # onto the ranks the lane rates say are slowest. Same grid, same unit
        # count, same MLP half -- only the direction is wrong.
        detuned = list(reversed(a2))
        g_det = gain(m, v2, gemm, detuned)
        verdict = "PASS" if g_det < g2 - 1e-9 else "FAIL"
        ok &= verdict == "PASS"
        print(f"{key}: aligned {a2} {g2 * 100:+.2f}%  vs  detuned "
              f"{detuned} {g_det * 100:+.2f}%  -> {verdict} "
              f"({(g2 - g_det) * 100:+.2f} points)")
        print(f"     detuned pacers {pacers(m, v2, gemm, detuned)}, skew "
              f"{m.prefill_barrier_skew(v2, gemm, None, detuned) * 1e6:.2f} "
              "us/tok")
    print("\nfalsifier:", "PASS" if ok else "FAIL")

    print("\n== 4. LANE BRACKET: how big the joint lever is depends on a "
          "quantity nobody measured\n")
    print("The attention/GDN barrier is part GEMM (qkv/o and the GDN "
          "in/out projections)\nand part bandwidth (flash / chunked scan). "
          "The mass split between them depends\non the context length, which "
          "this parse-time model does not carry, so it is\nBRACKETED rather "
          "than estimated. A third, physical point sits inside the\nbracket: "
          "the GDN family is BF16-resident in both checkpoints (2.0 B/param "
          "in\nthe family table), so its real lane is the dense bf16 probe, "
          "not the\ncheckpoint-wide quantized one #324 assigns it.\n")
    from sglang.srt.uneven_perf import _attn_lane_bracket

    for key, m in models.items():
        gemm = GEMM[key]
        bracket = _attn_lane_bracket(m, None, gemm, GEMV)
        gemm_lane, bw_lane = bracket
        bf16_lane = {
            n: (list(BF16_DENSE) if n in ("gdn", "vision") else list(gemm))
            for n in m.families
        }
        mlp_grid = [list(c) for c in _mlp_candidates(m, gemm, BUDGETS)]
        attn_grid = [list(c) for c in _attn_candidates(m, gemm, BUDGETS)]
        print(f"{key}:")
        for name, rates in (
            ("GEMM lane (shipped)", gemm_lane),
            ("bf16-resident GDN  ", bf16_lane),
            ("bandwidth lane     ", bw_lane),
        ):
            ref = m.prefill_time_model(list(BUDGETS), gemm, MIN_LINK, rates)

            def g(v, a, ref=ref, rates=rates):
                return ref / m.prefill_time_model(
                    v, gemm, MIN_LINK, rates, a
                ) - 1.0

            j = max(
                ((g(v, a), v, a) for v in mlp_grid for a in attn_grid),
                key=lambda x: x[0],
            )
            s1 = max(((g(v, None), v) for v in mlp_grid), key=lambda x: x[0])
            print(f"  {name}  MLP-only {s1[1]!s:12s} {s1[0] * 100:+6.2f}%   "
                  f"JOINT {j[1]!s:12s} + {j[2]!s:10s} {j[0] * 100:+6.2f}%   "
                  f"delta {(j[0] - s1[0]) * 100:+5.2f} pts")
        print()

    ok &= section5(models)
    return 0 if ok else 1


def section5(models):
    """#492: the axis section 2 could not see.

    Section 2 searched head partitions only, found that the 4-kv-head grid at
    tp=3 admits exactly one of them, and NOTE_485 wrote the attention family
    down as grid-pinned. The fork's own #62/#116 machinery clones kv heads and
    shards the token axis, so the attention CORE's per-rank mass follows the
    DCP token vector -- continuous, no grid. Three things are printed:

    1. the falsifier, executed: how many distinct attention partitions the
       WHOLE #485 candidate space realizes. One means the head axis is empty
       and a search restricted to it cannot move the family by construction.
    2. the geometric crossover depth, so a reader can place an operating point
       between the two bracket endpoints.
    3. the bracket itself, at both endpoints, with the argmax verdict.
    """
    from sglang.srt.uneven_perf import (
        _attn_candidates,
        _attn_token_candidates,
        _mlp_candidates,
        _replication_axis_lines,
        AttnCorePlan,
    )

    print("== 5. REPLICATION AXIS (#492): the attention family is NOT pinned\n")
    ok = True
    for key, m in models.items():
        gemm = GEMM[key]
        attn_grid = [list(c) for c in _attn_candidates(m, gemm, BUDGETS)]
        realized = {
            tuple(m._shard_fractions("attn", BUDGETS, list(v)))
            for v in [list(BUDGETS)] + attn_grid
        }
        base_tok = [
            v // math.gcd(*BUDGETS) for v in BUDGETS
        ]
        tok_grid = [list(c) for c in _attn_token_candidates(m, gemm, base_tok)]
        print(f"{key}: HEAD axis -- {len(attn_grid)} candidates on a "
              f"{m.attn_units}-kv-head grid realize {len(realized)} distinct "
              f"attention partition(s) {sorted(realized)}")
        print(f"      TOKEN axis -- {len(tok_grid)} candidates {tok_grid}, "
              "grid-free (the owner rule takes any positive integer per rank)")
        verdict = "PASS" if len(realized) == 1 and tok_grid else "FAIL"
        ok &= verdict == "PASS"
        print(f"      falsifier (head-only space cannot move the family): "
              f"{verdict}")
        print(f"      core/projection crossover: "
              f"{m.attn_core_crossover_tokens():,.0f} attended tokens\n")

    for key, m in models.items():
        gemm = GEMM[key]
        mlp_grid = [list(c) for c in _mlp_candidates(m, gemm, BUDGETS)]
        attn_grid = [list(c) for c in _attn_candidates(m, gemm, BUDGETS)]
        print(f"{key}:")
        for line in _replication_axis_lines(
            m, list(BUDGETS), mlp_grid + [tuple(BUDGETS)], attn_grid,
            gemm, gemm, None, MIN_LINK,
        ):
            print("  " + line.strip())
        print()

    print("== 6. FALSIFIER: a detuned TOKEN vector must price WORSE\n")
    for key, m in models.items():
        gemm = GEMM[key]
        mlp_grid = [list(c) for c in _mlp_candidates(m, gemm, BUDGETS)]
        attn_grid = [list(c) for c in _attn_candidates(m, gemm, BUDGETS)]
        base_tok = tuple(
            v // math.gcd(*BUDGETS) for v in BUDGETS
        )
        tok_grid = [list(c) for c in _attn_token_candidates(m, gemm, base_tok)]
        if not tok_grid:
            # A symmetric lane proposes nothing, which is the correct
            # generalization and not a failure -- say so instead of raising.
            print(f"{key}: symmetric attention lane, no token candidate to "
                  "detune")
            continue
        base_core = AttnCorePlan(base_tok, 1.0)
        ref = m.prefill_time_model(
            list(BUDGETS), gemm, MIN_LINK, None, None, base_core, base_core
        )

        def g(v, a, t):
            return ref / m.prefill_time_model(
                v, gemm, MIN_LINK, None, a,
                AttnCorePlan(tuple(t), 1.0), base_core,
            ) - 1.0

        best = max(
            ((g(v, a, t), v, a, t)
             for v in mlp_grid for a in attn_grid for t in tok_grid),
            key=lambda x: x[0],
        )
        det = list(reversed(best[3]))
        g_det = g(best[1], best[2], det)
        verdict = "PASS" if g_det < best[0] - 1e-9 else "FAIL"
        ok &= verdict == "PASS"
        print(f"{key}: aligned tokens {best[3]} {best[0] * 100:+.2f}%  vs  "
              f"detuned {det} {g_det * 100:+.2f}%  -> {verdict} "
              f"({(best[0] - g_det) * 100:+.2f} points)")
    print("\nreplication-axis falsifiers:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(main())
