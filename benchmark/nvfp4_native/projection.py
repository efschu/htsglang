#!/usr/bin/env python3
"""HOCHRECHNUNG (not a measurement): P-prefill 8k and the D GEMM term under bs6
for native-mixed NVFP4 (Backlog #38). Every input is named with its source.

Model P (PP3, one layer = MLP [NVFP4] + one attention-family projection set [FP8]):
  t_stage(card) = other(card) + FLOP_mlp / R_mlp(card) + FLOP_fp8 / R_fp8(card)
  per layer per 512-token chunk. "other" is DERIVED from the measured INT8 stage
  (#PGAP RC4 medians, BACKLOG_NVFP4_NATIVE_LAYOUT.md point 3) minus its INT8 GEMM
  time at the measured INT8 lane rate. Cut: L5*t5 = L3*t3, L5 + 2*L3 = 64.
  Throughput is ANCHORED on measured boots: tok/s_new = tok/s_meas * T_meas/T_new,
  with T = max stage time of the anchor's own cut, computed by the same model.
  Two anchors (INT8 RC2-final 8905 @8k, NVFP4-Marlin RC7b 4246 @8k) bracket the
  model error.

Model D (GEMM term only, per verify round, per rank):
  sum over the rank's linears of the MEASURED kernel time at M (bench JSON), or,
  for the 3080 until N4A reports, a stated rate. Printed as ms per round, not tok/s.
"""

from __future__ import annotations

import argparse
import json
import math

H, I, L = 5120, 17408, 64
FLOP_MLP = 2 * 512 * 3 * H * I  # 273.8 GF per layer per 512 chunk
# FP8 family: 48 GDN layers (qkvz 16384 + out 5120x6144), 16 attn (qkv 14336 + o 5120x6144)
FLOP_GDN = 2 * 512 * (16384 * H + H * 6144)
FLOP_ATT = 2 * 512 * (14336 * H + H * 6144)
FLOP_FP8 = (48 * FLOP_GDN + 16 * FLOP_ATT) / 64

# Lane rates, TFLOPS. Source unless overridden by --bench: INTEGRATION_R3_VALIDATION.md
# :15960-15978 (31.07., M=2048) and uneven_perf.py:754-757 (#327: int8 5090 678, 3080 ~180).
R = {
    "5090": {"int8": 678.0, "nvfp4_native": 1304.0, "nvfp4_marlin": 233.0, "fp8_native": 568.0, "fp8_marlin": 215.0},
    "3080": {"int8": 180.0, "nvfp4_marlin": 63.0, "fp8_marlin": 60.0},
}
# Measured stage medians (ms / layer / 512 chunk), BACKLOG point 3.
STAGE = {"int8": {"5090": 1.8, "3080": 6.1}, "nvfp4_marlin": {"5090": 2.9, "3080": 11.0}}
ANCHOR = {"int8": (8905.0, None), "nvfp4_marlin": (4246.0, (42, 11, 11))}


def other(card):
    return STAGE["int8"][card] - (FLOP_MLP + FLOP_FP8) / (R[card]["int8"] * 1e12) * 1e3


def t_stage(card, r_mlp, r_fp8, extra_ms=0.0):
    return other(card) + FLOP_MLP / (r_mlp * 1e12) * 1e3 + FLOP_FP8 / (r_fp8 * 1e12) * 1e3 + extra_ms


def balanced_T(t5, t3):
    l5 = L / (1 + 2 * t5 / t3)
    return l5 * t5, l5


def int_cut_T(t5, t3):
    best = None
    for l5 in range(1, L):
        rest = L - l5
        a, b = rest // 2, rest - rest // 2
        T = max(l5 * t5, b * t3)
        if best is None or T < best[0]:
            best = (T, (l5, b, a))
    return best


def anchor_T(name):
    t5, t3 = STAGE[name]["5090"], STAGE[name]["3080"]
    cut = ANCHOR[name][1]
    if cut:
        return max(cut[0] * t5, max(cut[1], cut[2]) * t3)
    return int_cut_T(t5, t3)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", help="bench_5090_nvfp4.py JSON; overrides the 5090 rates at M=512")
    ap.add_argument("--w4a8-tops", type=float, nargs="*", default=[120.0, 155.0],
                    help="3080 W4A8 effective rate(s) until N4A measures (user estimate 65%% of 238 = 155)")
    ap.add_argument("--q-overhead-ms", type=float, default=0.03,
                    help="5090 fp4 activation quant per layer per chunk (2 calls)")
    a = ap.parse_args()
    if a.bench:
        rows = json.load(open(a.bench))["rows"]

        def rate(shape, kern, M=512):
            for r in rows:
                if r.get("shape") == shape and r.get("kernel") == kern and r.get("M") == M and "us" in r:
                    return r
            return None
        fl = 0.0
        us = 0.0
        for sh in ("P.gate_up", "P.down"):
            r = rate(sh, "apply_nat")
            if r:
                us += r["us"]; fl += 2 * 512 * r["N"] * r["K"]
        if us:
            R["5090"]["nvfp4_native"] = fl / (us * 1e-6) / 1e12
            a.q_overhead_ms = 0.0  # apply_nat includes the quant
        fl = us = 0.0
        for sh in ("P.gate_up", "P.down"):
            r = rate(sh, "apply_mar")
            if r:
                us += r["us"]; fl += 2 * 512 * r["N"] * r["K"]
        if us:
            R["5090"]["nvfp4_marlin"] = fl / (us * 1e-6) / 1e12
        for kern, key in (("f8_nat", "fp8_native"), ("f8_mar", "fp8_marlin")):
            fl = us = 0.0
            for sh in ("F.qkvz", "F.o"):
                r = rate(sh, kern)
                if r:
                    us += r["us"]; fl += 2 * 512 * r["N"] * r["K"]
            if us:
                R["5090"][key] = fl / (us * 1e-6) / 1e12
    print("HOCHRECHNUNG -- keine Messung")
    print(f"FLOP/Schicht/512er-Chunk: MLP {FLOP_MLP/1e9:.1f} GF, FP8-Familie {FLOP_FP8/1e9:.1f} GF")
    print(f"'other' abgeleitet: 5090 {other('5090'):.2f} ms, 3080 {other('3080'):.2f} ms")
    print("5090-Raten (TFLOPS):", {k: round(v) for k, v in R["5090"].items()})
    Ti, Tn = anchor_T("int8"), anchor_T("nvfp4_marlin")
    print(f"Anker: INT8 T={Ti:.1f} ms (8905 tok/s), NVFP4-Marlin T={Tn:.1f} ms (4246 tok/s, Schnitt 42/11/11)")
    t5_nat = t_stage("5090", R["5090"]["nvfp4_native"], R["5090"]["fp8_native"], a.q_overhead_ms)
    t5_nat_fp8m = t_stage("5090", R["5090"]["nvfp4_native"], R["5090"]["fp8_marlin"], a.q_overhead_ms)
    for w in a.w4a8_tops:
        t3 = t_stage("3080", w, R["3080"]["fp8_marlin"])
        for label, t5 in (("5090 FP4+FP8 nativ", t5_nat), ("5090 FP4 nativ, FP8 Marlin", t5_nat_fp8m)):
            T, cut = int_cut_T(t5, t3)
            lo, hi = 4246 * Tn / T, 8905 * Ti / T
            print(f"  W4A8 {w:.0f} TOPS | {label}: t5={t5:.2f} t3={t3:.2f} ms, Schnitt {cut}, "
                  f"T={T:.1f} ms -> P8k ~{min(lo,hi)/1e3:.1f}-{max(lo,hi)/1e3:.1f}k tok/s")
    # what 10k would need
    Tneed = 8905 * Ti / 10000
    if a.bench:
        print("D, 5090-Rang, GEMM-Anteil je Verify-Runde (ms), gemessen je Kernel, summiert:")
        for M, res in d_gemm_term(a.bench).items():
            print(f"  M={M}: {res}")
    print(f"  10k tok/s braucht T<={Tneed:.1f} ms; bei t5={t5_nat:.2f}: L5<={Tneed/t5_nat:.1f}, "
          f"3080-Stufe <= {Tneed/((L-Tneed/t5_nat)/2):.2f} ms/Schicht (other allein {other('3080'):.2f})")


def d_gemm_term(bench_path, Ms=(8, 16, 48)):
    """5090 D-rank GEMM time per verify round (ms): all 64 layers' MLP shards
    (measured D shapes) + the FP8 family scaled by the 5090 share (full-width
    shapes measured, scaled linearly -- an approximation), lm_head excluded."""
    rows = json.load(open(bench_path))["rows"]

    def us(shape, kern, M):
        for r in rows:
            if r.get("shape") == shape and r.get("kernel") == kern and r.get("M") == M and "us" in r:
                return r["us"]
        return None
    share = 73 / 136
    out = {}
    for M in Ms:
        res = {}
        for label, k4, k8 in (("heute Marlin", "apply_mar", "f8_mar"), ("nativ", "apply_nat", "f8_nat")):
            g, d = us("D.gate_up", k4, M), us("D.down", k4, M)
            fq, fo, fa = us("F.qkvz", k8, M), us("F.o", k8, M), us("F.qkv", k8, M)
            if None in (g, d, fq, fo, fa):
                res[label] = None
                continue
            mlp = 64 * (g + d)
            fp8 = share * (48 * (fq + fo) + 16 * (fa + fo))
            res[label] = round((mlp + fp8) / 1e3, 2)
        out[M] = res
    return out


if __name__ == "__main__":
    main()
