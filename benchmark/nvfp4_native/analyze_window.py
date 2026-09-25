#!/usr/bin/env python3
"""Turn the window's bench JSONs into (A) a kernel table and (B) a MEASURED layer sum
per 512-token chunk, compared with the #PGAP stage time per layer. No projection here."""

from __future__ import annotations

import json
import os
import sys

PEAK = {  # dense datasheet peaks (TFLOPS/TOPS); GeForce FP32-accumulate halving NOT applied
    "5090": {"fp4": 1676.0, "fp8": 838.0, "int8": 838.0, "bf16": 419.0},
    "3080": {"int8": 238.0, "bf16": 59.5, "fp8": 59.5, "fp4": 59.5},
}
KFAM = {"q_fi": None, "q_sgl": None, "mm_sgl": "fp4", "mm_fi_cutlass": "fp4", "mm_fi_cudnn": "fp4",
        "mm_fi_b12x": "fp4", "apply_nat": "fp4", "apply_mar": "bf16", "i8_q": None, "i8_mm": "int8",
        "bf16_mm": "bf16", "f8_nat": "fp8", "f8_mar": "bf16", "f8_deq_bf16": "bf16", "i8_q+mm": "int8"}
# #PGAP fits (weg2rc4, INT8) at the 8k mean start position 3.75k, per layer:
STAGE_PER_LAYER = {"5090": (48.6 + 0.894 * 3.75) / 42, "3080": (38.1 + 1.083 * 3.75) / 11}
MIX = {"5090": (10, 32), "3080": (3, 8)}  # (attn, gdn) layers in the 42/11/11 cut


def load(path):
    if not os.path.exists(path):
        return []
    try:
        return json.load(open(path))["rows"]
    except Exception:  # noqa: BLE001
        rows = []
        for line in open(path.replace(".json", ".log"), errors="replace"):
            if line.startswith("{") and '"kernel"' in line or '"comp"' in line:
                try:
                    rows.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    pass
        return rows


def us(rows, shape, kern, M):
    for r in rows:
        if r.get("shape") == shape and r.get("kernel") == kern and r.get("M") == M and "us" in r:
            return r["us"]
    return None


def comp(rows, name, **kw):
    for r in rows:
        if r.get("comp") == name and all(r.get(k) == v for k, v in kw.items()) and "us" in r:
            return r["us"]
    return None


def kernel_table(card, rows, Ms=(1, 8, 16, 48, 512, 4096)):
    print(f"\n### A. Kernel ({card}) -- MESSUNG, µs / TFLOPS / Anteil an der Datenblatt-Spitze")
    shapes = []
    for r in rows:
        if "us" in r and r.get("shape") and r["shape"] not in shapes:
            shapes.append(r["shape"])
    for sh in shapes:
        kerns = []
        for r in rows:
            if r.get("shape") == sh and "us" in r and r["kernel"] not in kerns:
                kerns.append(r["kernel"])
        print(f"\n{sh}  | " + " | ".join(f"M={m}" for m in Ms))
        for k in kerns:
            cells = []
            for m in Ms:
                rr = next((r for r in rows if r.get("shape") == sh and r.get("kernel") == k and r.get("M") == m and "us" in r), None)
                if rr is None:
                    cells.append("-")
                    continue
                fam = KFAM.get(k)
                pk = PEAK[card].get(fam) if fam else None
                pct = f" {100*rr['tflops']/pk:.0f}%" if pk and rr.get("tflops") else ""
                cells.append(f"{rr['us']:.1f}µs {rr.get('tflops', 0):.0f}T{pct}")
            print(f"  {k:<13} | " + " | ".join(cells))
    errs = [r for r in rows if "error" in r]
    if errs:
        print(f"  Fehler ({len(errs)}): " + "; ".join(f"{e.get('shape','')}/{e.get('kernel', e.get('comp'))}@{e.get('M','')}: {e['error'][:80]}" for e in errs[:8]))


def layer_sum(card, lanes, comps, M=512):
    print(f"\n### B. Schichtsumme {card}, 512er-Chunk, Kontext 3,75k -- MESSUNG je Kern, Summe = Rechnung")
    na, ng = MIX[card]
    L = na + ng
    g = lambda s, k: us(lanes, s, k, M)  # noqa: E731
    variants = {
        "INT8": (lambda: (g("P.gate_up", "i8_q") or 0) + g("P.gate_up", "i8_mm") + (us(lanes, "P.down", "i8_q", M) or 0) + g("P.down", "i8_mm"),
                 "i8_q+mm"),
        "NVFP4 nativ": (lambda: g("P.gate_up", "apply_nat") + g("P.down", "apply_nat"), "f8_nat"),
        "Marlin (heute)": (lambda: g("P.gate_up", "apply_mar") + g("P.down", "apply_mar"), "f8_mar"),
    }
    attn_pre = comp(comps, "attn_pre", prefix=4096)
    a1 = comp(comps, "attn_pre", prefix=1024)
    if attn_pre is not None and a1 is not None:
        attn_pre = a1 + (attn_pre - a1) * (3750 - 1024) / (4096 - 1024)
    parts = {
        "gdn": comp(comps, "gdn"), "attn_pre@3.75k": attn_pre, "attn_diag": comp(comps, "attn_diag"),
        "rmsnorm": comp(comps, "rmsnorm"), "silu_mul": comp(comps, "silu_mul"),
    }
    print("  Rest-Kerne (µs):", {k: (round(v, 1) if v is not None else None) for k, v in parts.items()})
    nongemm = None
    if None not in parts.values():
        nongemm = (ng * parts["gdn"] + na * (parts["attn_pre@3.75k"] + parts["attn_diag"])
                   + L * (2 * parts["rmsnorm"] + parts["silu_mul"])) / L
    for name, (mlp_f, f8k) in variants.items():
        try:
            mlp = mlp_f()
            att = g("F.qkv", f8k) + g("F.o", f8k)
            gdnp = g("F.qkvz", f8k) + g("F.o", f8k)
        except TypeError:
            print(f"  {name}: unvollständig")
            continue
        gemm = mlp + (na * att + ng * gdnp) / L
        tot = gemm + (nongemm or 0)
        print(f"  {name:<15} GEMM {gemm/1e3:.3f} ms/Schicht (MLP {mlp/1e3:.3f}, Proj {(na*att+ng*gdnp)/L/1e3:.3f})"
              + (f" + Nicht-GEMM {nongemm/1e3:.3f} = {tot/1e3:.3f} ms" if nongemm else ""))
    print(f"  gemessene Stufe (#PGAP, INT8, 8k-Mittel): {STAGE_PER_LAYER[card]:.3f} ms/Schicht")


if __name__ == "__main__":
    d = sys.argv[1]
    l5, c5 = load(f"{d}/b5090_lanes.json"), load(f"{d}/b5090_comp.json")
    l3, c3 = load(f"{d}/b3080_lanes.json"), load(f"{d}/b3080_comp.json")
    b12 = load(f"{d}/b5090_b12x.json")
    kernel_table("5090", l5 + [r for r in b12 if r.get("kernel") == "mm_fi_b12x"])
    kernel_table("3080", l3)
    for card, c in (("5090", c5), ("3080", c3)):
        print(f"\n### Komponenten {card} (MESSUNG):")
        for r in c:
            print("  ", r)
    layer_sum("5090", l5, c5)
    layer_sum("3080", l3, c3)
