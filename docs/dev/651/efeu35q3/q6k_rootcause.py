#!/usr/bin/env python
"""efeu-TP14: root-cause harness for the gfx1103 K-quant per-launch fault.

Prior state (FINAL_651 §2, gpu_sanity_guard_v2.py): ~1.6 % of dequantize
launches return whole contiguous runs of 32/64/128 wrong fp16 elements, small
in magnitude (1e-2..3e-2), q6_K > q5_K >> q4_K. Ruled out then: unwritten
memory (sentinel), D2H copy, cold start; a native gfx1103 build was reported at
the same 3/25 rate.

This harness characterises EACH faulty run instead of counting launches, so the
mechanism can be read off the numbers:
  * position: block index i, element offset in the 256-block, wave (32-lane
    group) and which of the four per-thread stores (y[0]/y[32]/y[64]/y[96])
  * the implied d' = y_bad / (sc*q) per element: if it is constant over the run
    and equals another block's d, the wave read the WRONG block's scale header;
    if it varies, the quant bits or the scales were wrong
  * the ratio y_bad / y_good
and it runs the same bytes through dequant / mmvq / mmq / moe kernels with
DISTINCT per-expert contents (a replicated expert stack hides index bugs).

    python q6k_rootcause.py --src <gguf> --tensor output.weight --rows 4096 \
        --launches 200 [--module sglang_gguf_rocm] [--json out.json]

Run it under each build/override variant; the variant is printed from the env.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time

import numpy as np
import torch
import gguf
from gguf import GGUFReader
from gguf.constants import GGMLQuantizationType as QT


def load_rows(src, tname, rows):
    r = GGUFReader(src)
    for t in r.tensors:
        if t.name == tname:
            d = t.data
            if d.ndim == 3:
                d = d.reshape(-1, d.shape[-1])
            return t.tensor_type, np.ascontiguousarray(d[:rows])
    raise SystemExit(f"{tname} not in {src}")


def q6k_fields(raw):
    """Per-block d (fp32), int8 scales [16] and the 6-bit q in element order."""
    nb = raw.shape[0] * raw.shape[1] // 210
    b = raw.reshape(nb, 210)
    ql, qh, sc = b[:, :128], b[:, 128:192], b[:, 192:208].view(np.int8)
    d = b[:, 208:210].copy().view(np.float16).astype(np.float32)[:, 0]
    q = np.zeros((nb, 256), np.int32)
    for ip in range(2):
        for il in range(32):
            l0 = ql[:, 64 * ip + il].astype(np.int32)
            l1 = ql[:, 64 * ip + il + 32].astype(np.int32)
            h = qh[:, 32 * ip + il].astype(np.int32)
            base = 128 * ip + il
            q[:, base + 0] = ((l0 & 0xF) | (((h >> 0) & 3) << 4)) - 32
            q[:, base + 32] = ((l1 & 0xF) | (((h >> 2) & 3) << 4)) - 32
            q[:, base + 64] = ((l0 >> 4) | (((h >> 4) & 3) << 4)) - 32
            q[:, base + 96] = ((l1 >> 4) | (((h >> 6) & 3) << 4)) - 32
    scale_of = np.array([(e // 16) for e in range(256)])  # 16 scales x 16 elems
    return d, sc.astype(np.int32), q, scale_of


def runs_of(mask_flat):
    idx = np.flatnonzero(mask_flat)
    if idx.size == 0:
        return []
    out, s, p = [], idx[0], idx[0]
    for x in idx[1:]:
        if x != p + 1:
            out.append((s, p + 1))
            s = x
        p = x
    out.append((s, p + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--tensor", default="output.weight")
    ap.add_argument("--rows", type=int, default=4096)
    ap.add_argument("--launches", type=int, default=200)
    ap.add_argument("--module", default=os.environ.get("GGUF_MODULE", "sglang_gguf_rocm"))
    ap.add_argument("--ops", default="dequant,mmvq,mmq,moe_vec")
    ap.add_argument("--json")
    a = ap.parse_args()

    K = importlib.import_module(a.module)
    variant = {
        "module_file": K.__file__,
        "HSA_OVERRIDE_GFX_VERSION": os.environ.get("HSA_OVERRIDE_GFX_VERSION"),
        "gcnArchName": torch.cuda.get_device_properties(0).gcnArchName,
        "sclk_level": open("/sys/class/drm/card1/device/power_dpm_force_performance_level").read().strip(),
    }
    print("VARIANT", json.dumps(variant))
    qt, raw = load_rows(a.src, a.tensor, a.rows)
    rows = raw.shape[0]
    cols = raw.shape[1] // gguf.GGML_QUANT_SIZES[qt][1] * gguf.GGML_QUANT_SIZES[qt][0]
    ref = gguf.quants.dequantize(raw, qt).astype(np.float32)
    print(f"tensor {a.tensor} {qt.name} rows={rows} cols={cols} blocks={rows * cols // 256}")
    W = torch.from_numpy(raw).cuda()
    res = {"variant": variant, "type": qt.name, "rows": rows, "cols": cols, "ops": {}}

    # ---- dequant: per-launch fault characterisation against the CONSENSUS ----
    if "dequant" in a.ops:
        outs = []
        t0 = time.time()
        for _ in range(a.launches):
            o = K.ggml_dequantize(W, int(qt), rows, cols, torch.float16, None)
            torch.cuda.synchronize()
            outs.append(o.cpu().numpy())
            del o
        stack = np.stack(outs)
        cons = np.median(stack.astype(np.float32), axis=0).astype(np.float16)
        cons_err = float(np.abs(cons.astype(np.float32) - ref).max())
        bad_launch = 0
        faults = []
        fields = q6k_fields(raw) if qt == QT.Q6_K else None
        for li, o in enumerate(outs):
            m = o != cons
            if not m.any():
                continue
            bad_launch += 1
            flat_o, flat_c = o.reshape(-1), cons.reshape(-1)
            for s, e in runs_of(m.reshape(-1)):
                blk, off = divmod(int(s), 256)
                rec = {"launch": li, "start": int(s), "len": int(e - s), "block": blk,
                       "off": off, "maxdev": float(np.abs(flat_o[s:e].astype(np.float32) - flat_c[s:e].astype(np.float32)).max())}
                good = flat_c[s:e].astype(np.float32)
                badv = flat_o[s:e].astype(np.float32)
                nz = np.abs(good) > 1e-6
                if nz.any():
                    ratio = badv[nz] / good[nz]
                    rec["ratio_mean"] = float(ratio.mean())
                    rec["ratio_std"] = float(ratio.std())
                if fields is not None:
                    d, sc, q, scale_of = fields
                    offs = np.arange(off, off + (e - s)) % 256
                    blks = (np.arange(s, e)) // 256
                    den = sc[blks, scale_of[offs]] * q[blks, offs]
                    ok = den != 0
                    if ok.any():
                        dimp = badv[ok] / den[ok]
                        rec["d_true"] = float(d[blk])
                        rec["d_implied_mean"] = float(dimp.mean())
                        rec["d_implied_std"] = float(dimp.std())
                        # which block's d matches the implied d best?
                        cand = np.argmin(np.abs(d - dimp.mean()))
                        rec["d_best_block"] = int(cand)
                        rec["d_best_delta"] = int(cand - blk)
                faults.append(rec)
        res["ops"]["dequant"] = {"launches": a.launches, "bad_launches": bad_launch,
                                 "consensus_vs_oracle": cons_err, "faults": faults[:200],
                                 "n_fault_runs": len(faults),
                                 "run_len_hist": {str(k): int(v) for k, v in zip(*np.unique([f["len"] for f in faults], return_counts=True))} if faults else {},
                                 "wall_s": time.time() - t0}
        print(f"dequant: {bad_launch}/{a.launches} launches faulty, {len(faults)} runs, "
              f"consensus-vs-oracle {cons_err:.3e}")
        for f in faults[:12]:
            print("   ", json.dumps(f))

    # ---- matmul kernels: determinism + error vs fp64 oracle -----------------
    rng = np.random.default_rng(7)
    def det(name, call, rf, n):
        outs = []
        for _ in range(n):
            o = call()
            torch.cuda.synchronize()
            outs.append(o.float().cpu().numpy().reshape(rf.shape))
        diffs = [float(np.abs(o - outs[0]).max()) for o in outs[1:]]
        nbad = sum(1 for x in diffs if x > 0)
        worst = max(float(np.abs(np.nan_to_num(o) - rf).max()) for o in outs)
        scale = float(np.abs(rf).max())
        res["ops"][name] = {"n": n, "nondet_launches": nbad, "worst_abs": worst, "rel": worst / scale}
        print(f"{name:8s} nondet {nbad}/{n - 1}  worst|d| {worst:.3e} rel {worst / scale:.2e}")

    nmm = max(20, a.launches // 4)
    rows_mm = min(rows, 2048)
    W2 = torch.from_numpy(np.ascontiguousarray(raw[:rows_mm])).cuda()
    ref2 = ref[:rows_mm].astype(np.float64)
    if "mmvq" in a.ops:
        X1 = torch.from_numpy(rng.standard_normal((1, cols), dtype=np.float32) * 0.1).cuda().half()
        det("mmvq", lambda: K.ggml_mul_mat_vec_a8(W2, X1, int(qt), rows_mm),
            X1.float().cpu().numpy().astype(np.float64) @ ref2.T, nmm)
    if "mmq" in a.ops:
        Xb = torch.from_numpy(rng.standard_normal((64, cols), dtype=np.float32) * 0.1).cuda().half()
        det("mmq", lambda: K.ggml_mul_mat_a8(W2, Xb, int(qt), rows_mm),
            Xb.float().cpu().numpy().astype(np.float64) @ ref2.T, nmm)
    if "moe_vec" in a.ops:
        # DISTINCT experts: split the rows into E expert matrices of R rows each
        E, R, T, TOPK = 8, rows_mm // 8, 4, 2
        Wm = torch.from_numpy(np.ascontiguousarray(raw[: E * R].reshape(E, R, -1))).cuda()
        refm = ref[: E * R].reshape(E, R, cols).astype(np.float64)
        Xt = torch.from_numpy(rng.standard_normal((T, cols), dtype=np.float32) * 0.1).cuda().half()
        ids = rng.choice(E, size=(T, TOPK), replace=True).astype(np.int32)
        ids[:, 1] = (ids[:, 0] + 3) % E
        topk = torch.from_numpy(ids).cuda()
        xt = Xt.float().cpu().numpy().astype(np.float64)
        rf = np.stack([np.stack([xt[t] @ refm[ids[t, k]].T for k in range(TOPK)]) for t in range(T)])
        det("moe_vec", lambda: K.ggml_moe_a8_vec(Xt, Wm, topk, TOPK, int(qt), R, T), rf, nmm)

    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
