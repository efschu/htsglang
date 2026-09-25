#!/usr/bin/env python3
"""HOCHRECHNUNG (not a measurement): P-prefill 8k and the D GEMM term under bs6
for native-mixed NVFP4 (Backlog #38). Every input is named with its source.

Model P (PP3): per rank and 512-token chunk starting at position p (k tokens)
  t = n_layers * const(card) + GEMM(rates) + slope(card) * n_attn * p
  const and slope come from linear fits of the MEASURED #PGAP gpu_fwd_ms of the
  INT8 boot weg2rc4 (minus its INT8 GEMM at the lane rate); the 3080 Marlin rate
  at M=512 from the NVFP4 boot weg2rc4n4. 8k wall = (16 chunks + 2) * max rank
  time at mean p = 3.75k. The model reproduces INT8 8760 (measured 8905) and
  NVFP4-Marlin 4468 (measured 4246); both calibration factors bracket the answer.

Model D (GEMM term only, per verify round, per rank):
  sum over the rank's linears of the MEASURED kernel time at M (bench JSON), or,
  for the 3080 until N4A reports, a stated rate. Printed as ms per round, not tok/s.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

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
# MEASURED (#PGAP gpu_fwd_ms per 512-token chunk, linear fit over chunk start
# position p in k tokens, p <= 64k; RC4 boots, cut 42/11/11, attn 10/3/3; parser
# benchmark/nvfp4_native/pgap_fit.py):
#   weg2rc4   (INT8):        PP0 48.6 + 0.894 p | PP1 38.1 + 1.083 p | PP2 42.1 + 1.111 p
#   weg2rc4n4 (NVFP4-Marlin): PP0 94.6 + 1.133 p | PP1 93.8 + 1.113 p | PP2 94.3 + 1.116 p
FIT_INT8 = {0: (48.6, 0.894), 1: (38.1, 1.083), 2: (42.1, 1.111)}
FIT_NV = {0: (94.6, 1.133), 1: (93.8, 1.113), 2: (94.3, 1.116)}
CUT0 = (42, 11, 11)
MEAS_8K = {"int8": 8905.0, "nvfp4_marlin": 4246.0}
N_CHUNKS_8K, P_MEAN_8K = 16, 3.75  # 8k prompt: 16 chunks, mean start position 3.75k


def attn_count(lo, hi):
    return sum(1 for i in range(lo, hi) if i % 4 == 3)


def stage_flops(lo, hi):
    n = hi - lo
    na = attn_count(lo, hi)
    return n * FLOP_MLP, na * FLOP_ATT + (n - na) * FLOP_GDN


def ranks(cut):
    b = [0, cut[0], cut[0] + cut[1], L]
    return [(b[i], b[i + 1]) for i in range(3)]


def derive_constants():
    """Non-GEMM constant per layer and attention slope per attn-layer, per card."""
    out = {}
    for r, (lo, hi) in enumerate(ranks(CUT0)):
        card = "5090" if r == 0 else "3080"
        fm, fa = stage_flops(lo, hi)
        gemm = (fm + fa) / (R[card]["int8"] * 1e12) * 1e3
        a, b = FIT_INT8[r]
        out[r] = {"card": card, "const_per_layer": (a - gemm) / (hi - lo),
                  "slope_per_attn": b / attn_count(lo, hi), "gemm_int8": gemm, "a": a}
    return out


def model_wall(cut, rates, consts, extra_last=None):
    """8k-prompt pipeline wall (s) = (chunks + stages - 1) x max stage time at mean p."""
    if extra_last is None:
        extra_last = consts[2]["const_per_layer"] * 11 - consts[1]["const_per_layer"] * 11
    ts = []
    for r, (lo, hi) in enumerate(ranks(cut)):
        card = "5090" if r == 0 else "3080"
        cl = consts[0 if r == 0 else 1]
        fm, fa = stage_flops(lo, hi)
        t = (hi - lo) * cl["const_per_layer"] + fm / (rates[card][0] * 1e12) * 1e3 \
            + fa / (rates[card][1] * 1e12) * 1e3 + cl["slope_per_attn"] * attn_count(lo, hi) * P_MEAN_8K
        if r == 2:
            t += extra_last
        ts.append(t)
    return (N_CHUNKS_8K + 2) * max(ts) / 1e3, ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", help="bench_5090_nvfp4.py JSON; overrides the 5090 rates at M=512")
    ap.add_argument("--w4a8-tops", type=float, nargs="*", default=[100.0, 120.0, 155.0],
                    help="3080 W4A8 effective rate(s) at M=512 until N4A measures")
    ap.add_argument("--marlin3080-m512", type=float, default=None,
                    help="3080 FP8-Marlin rate at M=512; default: derived from the NVFP4 PGAP fit")
    a = ap.parse_args()
    if a.bench:
        rows = json.load(open(a.bench))["rows"]

        def rate(shapes, kern, M=512):
            us = fl = 0.0
            for sh in shapes:
                for r in rows:
                    if r.get("shape") == sh and r.get("kernel") == kern and r.get("M") == M and "us" in r:
                        us += r["us"]; fl += 2 * M * r["N"] * r["K"]
                        break
            return fl / (us * 1e-6) / 1e12 if us else None
        for key, shapes, kern in (("nvfp4_native", ("P.gate_up", "P.down"), "apply_nat"),
                                  ("nvfp4_marlin", ("P.gate_up", "P.down"), "apply_mar"),
                                  ("fp8_native", ("F.qkvz", "F.o", "F.qkv"), "f8_nat"),
                                  ("fp8_marlin", ("F.qkvz", "F.o", "F.qkv"), "f8_mar")):
            v = rate(shapes, kern)
            if v:
                R["5090"][key] = v
    C = derive_constants()
    print("HOCHRECHNUNG -- keine Messung (Eingaben: #PGAP-Fits = Messung, Lane-Raten = Messung, Rest = Modell)")
    for r in (0, 1, 2):
        c = C[r]
        print(f"  PP{r} ({c['card']}): gpu_fwd(p=0) {c['a']:.1f} ms = INT8-GEMM {c['gemm_int8']:.1f} (Lane-Rate) "
              f"+ Nicht-GEMM {c['a']-c['gemm_int8']:.1f} ms ({c['const_per_layer']:.2f} ms/Schicht); "
              f"Attention {c['slope_per_attn']:.3f} ms je Attn-Schicht je 1k Kontext")
    # 3080 Marlin at M=512, derived from the NVFP4 fit of PP1 (same constants)
    lo, hi = ranks(CUT0)[1]
    fm, fa = stage_flops(lo, hi)
    g_nv = FIT_NV[1][0] - (C[1]["a"] - C[1]["gemm_int8"])
    m3080 = a.marlin3080_m512 or (fm + fa) / (g_nv * 1e-3) / 1e12
    print(f"  3080 Marlin (FP4+FP8) bei M=512 aus PP1-NVFP4-Fit: {m3080:.1f} TFLOPS (Lane-Wert M=2048: 63/60)")
    rates_int8 = {"5090": (R["5090"]["int8"], R["5090"]["int8"]), "3080": (R["3080"]["int8"], R["3080"]["int8"])}
    rates_nv = {"5090": (R["5090"]["nvfp4_marlin"], R["5090"]["fp8_marlin"]), "3080": (m3080, m3080)}
    w_i, _ = model_wall(CUT0, rates_int8, C)
    w_n, _ = model_wall(CUT0, rates_nv, C)
    cal_i, cal_n = (8192 / w_i) / MEAS_8K["int8"], (8192 / w_n) / MEAS_8K["nvfp4_marlin"]
    print(f"  Modellprobe 8k: INT8 {8192/w_i:.0f} (gemessen 8905, Faktor {1/cal_i:.3f}), "
          f"NVFP4-Marlin {8192/w_n:.0f} (gemessen 4246, Faktor {1/cal_n:.3f})")
    print("  5090-Raten (TFLOPS):", {k: round(v) for k, v in R["5090"].items()})
    for w in a.w4a8_tops:
        for label, r5 in (("5090 FP4+FP8 nativ", (R["5090"]["nvfp4_native"], R["5090"]["fp8_native"])),
                          ("5090 FP4 nativ/FP8 Marlin", (R["5090"]["nvfp4_native"], R["5090"]["fp8_marlin"]))):
            rates = {"5090": r5, "3080": (w, m3080)}
            best = None
            for l0 in range(30, 56):
                rest = L - l0
                cut = (l0, rest - rest // 2, rest // 2)
                wall, ts = model_wall(cut, rates, C)
                if best is None or wall < best[0]:
                    best = (wall, cut, ts)
            wall, cut, ts = best
            tps = 8192 / wall
            print(f"  W4A8 {w:.0f} TOPS | {label}: Schnitt {cut}, Stufen {[round(t,1) for t in ts]} ms -> "
                  f"P8k ~{tps/cal_n/1e3:.1f}-{tps/cal_i/1e3:.1f}k tok/s (kalibriert NVFP4/INT8)")
    if a.bench:
        print("D, 5090-Rang, GEMM-Anteil je Verify-Runde (ms), gemessen je Kernel, summiert:")
        for M, res in d_gemm_term(a.bench).items():
            print(f"  M={M}: {res}")


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
        for label, k4, k8 in (("heute Marlin", "apply_mar", "f8_mar"), ("nativ", "apply_nat", "f8_torch"), ("FP4 nativ, FP8 Marlin", "apply_nat", "f8_mar")):
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


if __name__ == "__main__" and "--window" not in sys.argv:
    main()


# ---------------------------------------------------------------------------
# v3: every per-kernel term MEASURED in window zcx7pv (analyze_window.py); only
# the composition, the 3080 W4A8 rate and the cut are HOCHRECHNUNG.
# ---------------------------------------------------------------------------
def window_projection(d, w4a8_tops=(117.0, 150.0)):
    import os
    def rows(p):
        return json.load(open(p))["rows"] if os.path.exists(p) else []
    l5, c5 = rows(f"{d}/b5090_all.json"), rows(f"{d}/b5090_comp.json")
    l3, c3 = rows(f"{d}/b3080_lanes.json"), rows(f"{d}/b3080_comp.json")
    u = lambda R, s, k: next(r["us"] for r in R if r.get("shape") == s and r.get("kernel") == k and r.get("M") == 512 and "us" in r) / 1e3  # noqa: E731
    c = lambda R, n, **kw: next(r["us"] for r in R if r.get("comp") == n and all(r.get(a) == b for a, b in kw.items())) / 1e3  # noqa: E731
    P = 3.75
    def attn_ms(C):
        a1, a4 = c(C, "attn_pre", prefix=1024), c(C, "attn_pre", prefix=4096)
        return a1 + (a4 - a1) * (P * 1024 - 1024) / (3072) + c(C, "attn_diag")
    card = {
        "5090": {"gdn": c(c5, "gdn"), "attn": attn_ms(c5), "elt": 2 * c(c5, "rmsnorm") + c(c5, "silu_mul"),
                 "mlp": {"nat": u(l5, "P.gate_up", "apply_nat") + u(l5, "P.down", "apply_nat"),
                         "int8": u(l5, "P.gate_up", "i8_q") + u(l5, "P.gate_up", "i8_mm") + u(l5, "P.down", "i8_q") + u(l5, "P.down", "i8_mm"),
                         "mar": u(l5, "P.gate_up", "apply_mar") + u(l5, "P.down", "apply_mar")},
                 "proj_g": {k: u(l5, "F.qkvz", v) + u(l5, "F.o", v) for k, v in (("f8nat", "f8_torch"), ("f8mar", "f8_mar"), ("int8", "i8_q+mm"))},
                 "proj_a": {k: u(l5, "F.qkv", v) + u(l5, "F.o", v) for k, v in (("f8nat", "f8_torch"), ("f8mar", "f8_mar"), ("int8", "i8_q+mm"))}},
        "3080": {"gdn": c(c3, "gdn"), "attn": attn_ms(c3), "elt": 2 * c(c3, "rmsnorm") + c(c3, "silu_mul"),
                 "mlp": {"int8": u(l3, "P.gate_up", "i8_q") + u(l3, "P.gate_up", "i8_mm") + u(l3, "P.down", "i8_q") + u(l3, "P.down", "i8_mm"),
                         "mar": u(l3, "P.gate_up", "apply_mar") + u(l3, "P.down", "apply_mar")},
                 "proj_g": {k: u(l3, "F.qkvz", v) + u(l3, "F.o", v) for k, v in (("f8mar", "f8_mar"), ("int8", "i8_q+mm"))},
                 "proj_a": {k: u(l3, "F.qkv", v) + u(l3, "F.o", v) for k, v in (("f8mar", "f8_mar"), ("int8", "i8_q+mm"))}},
    }
    for t in w4a8_tops:
        card["3080"]["mlp"][f"w4a8_{int(t)}"] = FLOP_MLP / (t * 1e12) * 1e3

    def layer(cd, i, mlp, proj):
        k = card[cd]
        is_attn = i % 4 == 3
        return k["mlp"][mlp] + (k["proj_a"][proj] + k["attn"] if is_attn else k["proj_g"][proj] + k["gdn"]) + k["elt"]

    # unexplained per layer = #PGAP INT8 stage at p=3.75k minus the measured INT8 layer sum (42/11/11 cut)
    stage_meas = {0: 48.6 + 0.894 * P, 1: 38.1 + 1.083 * P, 2: 42.1 + 1.111 * P}
    sums = {r: sum(layer("5090" if r == 0 else "3080", i, "int8", "int8") for i in range(lo, hi))
            for r, (lo, hi) in enumerate(ranks(CUT0))}
    unexpl = {r: (stage_meas[r] - sums[r]) / (hi - lo) for r, (lo, hi) in enumerate(ranks(CUT0))}
    extra_last = (unexpl[2] - unexpl[1]) * 11
    print("\n== v3: Kerne GEMESSEN (zcx7pv), Zusammensetzung/Schnitt/W4A8 = HOCHRECHNUNG ==")
    for r in (0, 1, 2):
        lo, hi = ranks(CUT0)[r]
        print(f"  PP{r}: gemessene Kernsumme INT8 {sums[r]:.1f} ms vs #PGAP {stage_meas[r]:.1f} ms "
              f"-> unerklaert {unexpl[r]:.3f} ms/Schicht ({100*(stage_meas[r]-sums[r])/stage_meas[r]:.0f} %)")

    def stage_times(cut, m5, p5, m3, p3):
        ts = []
        for r, (lo, hi) in enumerate(ranks(cut)):
            cd = "5090" if r == 0 else "3080"
            mlp, proj = (m5, p5) if r == 0 else (m3, p3)
            t = sum(layer(cd, i, mlp, proj) for i in range(lo, hi)) + (hi - lo) * unexpl[0 if r == 0 else 1]
            if r == 2:
                t += extra_last
            ts.append(t)
        return ts

    def best(m5, p5, m3, p3):
        out = None
        for l0 in range(30, 58):
            rest = L - l0
            cut = (l0, rest - rest // 2, rest // 2)
            ts = stage_times(cut, m5, p5, m3, p3)
            if out is None or max(ts) < max(out[1]):
                out = (cut, ts)
        return out
    ref_i = stage_times(CUT0, "int8", "int8", "int8", "int8")
    ref_n = stage_times(CUT0, "mar", "f8mar", "mar", "f8mar")
    cal_i = (8192 / ((N_CHUNKS_8K + 2) * max(ref_i) / 1e3)) / 8905.0
    cal_n = (8192 / ((N_CHUNKS_8K + 2) * max(ref_n) / 1e3)) / 4246.0
    print(f"  Probe: INT8 42/11/11 -> {8192/((N_CHUNKS_8K+2)*max(ref_i)/1e3):.0f} tok/s (gem. 8905); "
          f"NVFP4-Marlin 42/11/11 -> {8192/((N_CHUNKS_8K+2)*max(ref_n)/1e3):.0f} (gem. 4246)")
    # N4C (#38, 3080 = Marlin W4A16 on the shared layout): the MEASURED Marlin MLP
    # of the 3080, plus the in-place band penalty -- the banded apply concatenates
    # the band outputs of every banded linear (P gate_up [512, 34816] bf16 = 35.6 MB,
    # read+write ~0.12 ms at ~600 GB/s on the 3080; down is one band at 16 MiB? no:
    # 44.6 MB -> 3 bands, [512, 5120] out = 5 MB, ~0.02 ms). RECHNUNG, not measured.
    card["3080"]["mlp"]["mar_banded"] = card["3080"]["mlp"]["mar"] + 0.14
    res = []
    for p5, lab in (("f8nat", "FP8 nativ (cuBLASLt)"), ("f8mar", "FP8 Marlin")):
        for m3, lab3 in (("mar", "Marlin W4A16"), ("mar_banded", "Marlin W4A16 + Band-Concat")):
            cut, ts = best("nat", p5, m3, "f8mar")
            tps = 8192 / ((N_CHUNKS_8K + 2) * max(ts) / 1e3)
            lo_, hi_ = sorted((tps / cal_i, tps / cal_n))
            vram = vram_5090(cut[0])
            print(f"  3080 {lab3} | 5090 NVFP4 nativ + {lab}: Schnitt {cut}, Stufen "
                  f"{[round(x,1) for x in ts]} ms -> P8k {tps/1e3:.2f}k (kalibriert {lo_/1e3:.1f}-{hi_/1e3:.1f}k) tok/s; "
                  f"5090-VRAM P {vram}")
            res.append((m3, lab, cut, ts, tps))
    for t in w4a8_tops:
        for p5, lab in (("f8nat", "FP8 nativ (cuBLASLt)"), ("f8mar", "FP8 Marlin")):
            cut, ts = best("nat", p5, f"w4a8_{int(t)}", "f8mar")
            tps = 8192 / ((N_CHUNKS_8K + 2) * max(ts) / 1e3)
            lo_, hi_ = sorted((tps / cal_i, tps / cal_n))
            vram = vram_5090(cut[0])
            print(f"  3080 W4A8 {t:.0f} TOPS | 5090 NVFP4 nativ + {lab}: Schnitt {cut}, Stufen "
                  f"{[round(x,1) for x in ts]} ms -> P8k {tps/1e3:.2f}k (kalibriert {lo_/1e3:.1f}-{hi_/1e3:.1f}k) tok/s; "
                  f"5090-VRAM P {vram}")
            res.append((t, lab, cut, ts, tps))
    return card, res


def vram_5090(l0, ctx_tokens=262144):
    """P-group VRAM on the 5090 for layers [0, l0) (GiB). Weights per layer from the
    checkpoint (NVFP4 MLP 150.4 MB; FP8 proj GDN 115.3 + ~1.2 MB bf16, attn 104.9 MB),
    base (embed etc.) 2.35 GiB = weg2rc7n4 PP0 13.18 GiB minus its 42 layers; mamba
    36.6 MB per GDN layer (1.17 GB / 32); KV fp8 2048 B/token per attn layer + 5 draft
    layers (cell 30720 at 10 attn layers); graph+reserve 1.2 GiB; avail 30.54 GiB."""
    na = sum(1 for i in range(l0) if i % 4 == 3)
    ng = l0 - na
    w = 2.35 + (ng * (150.4 + 116.5) + na * (150.4 + 105.0)) / 1024
    mamba = 0.0366 * ng
    cell = 2048 * (na + 5)
    kv_ctx = cell * ctx_tokens / 2**30
    fixed = w + mamba + 1.2
    free_for_kv = 30.54 - fixed
    return (f"Gewichte {w:.1f} + Mamba {mamba:.2f} + Graph/Reserve 1.2 = {fixed:.1f} GiB; "
            f"KV fuer 262k Kontext {kv_ctx:.1f} GiB (Zelle {cell} B); frei fuer KV {free_for_kv:.1f} GiB "
            f"= {int(free_for_kv*2**30/cell/1000)}k Token")


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "--window":
    window_projection(sys.argv[2])
