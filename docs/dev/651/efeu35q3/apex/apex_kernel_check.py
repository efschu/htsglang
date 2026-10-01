#!/usr/bin/env python
"""efeu-TP14: gfx1103 GGUF kernel correctness for the APEX-I-MiniPlus-V2.1
quant mix (+ the MTP Q4_0 head), with the CURRENT ext_v2 build
(no real-true16, WARP_SIZE_GGUF).

For every quant type present in the given GGUF files it takes REAL tensor
slices (for IQ3_XXS: a routed-expert tensor of layers 10-29) and runs, 8x
with fixed inputs, against the numpy gguf dequantize oracle (float64):
  dequant              ggml_dequantize
  mmvq@M=1,2,4,8       ggml_mul_mat_vec_a8   (M=2..8 = the tuned K-quant path
                                              that had the WARP_SIZE bug)
  mmq@M=16             ggml_mul_mat_a8       (prefill GEMM)
  moe_a8@T=8           ggml_moe_a8           (MoE prefill, MMQ types only)
  moe_vec@T=8          ggml_moe_a8_vec       (MoE decode / IQ prefill)
clean = all runs byte-identical AND rel err < TOL AND all finite.

--load N   : DISABLED (refuses). The 21:16 run with 6 hogs caused a global
             OOM that killed the user's desktop session.
--runs R   : runs per op (default 8).
Gate: refuses to start (and stops between ops) when the live service on
:31651 has inflight > 0 -- the iGPU is the user's.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import urllib.request

import numpy as np
import torch

sys.path.insert(0, os.environ.get("GGUF_EXT_DIR", "/root/efeu35q3/ext_v2"))
import sglang_gguf_rocm as K  # noqa: E402
import gguf  # noqa: E402
from gguf import GGUFReader  # noqa: E402
from gguf.constants import GGMLQuantizationType as QT  # noqa: E402

ROWS = 512
E, T = 4, 8
TOL_REL = 5e-2
MMQ_TYPES = {"Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"}


def service_busy():
    try:
        d = json.load(urllib.request.urlopen("http://127.0.0.1:31651/ondemand/status", timeout=5))
        return d.get("inflight", 0) > 0
    except Exception:
        return False


def hog(stop):
    a = np.ones(64 << 20, dtype=np.float64)  # 512 MB
    b = np.empty_like(a)
    while not stop.is_set():
        np.copyto(b, a)
        np.copyto(a, b)


def pick(paths, prefer_core_expert=True):
    picked = {}
    for path in paths:
        r = GGUFReader(path)
        for t in r.tensors:
            tt = t.tensor_type.name
            if tt in ("F32", "F16", "BF16"):
                continue
            core = any(f"blk.{i}." in t.name for i in range(10, 30)) and "exps" in t.name
            if tt in picked and not (prefer_core_expert and core and not picked[tt][2]):
                continue
            d = t.data
            if d.ndim == 3:
                d = d[0]
            d = d.reshape(-1, d.shape[-1])
            if d.shape[0] < 64:
                continue
            picked[tt] = (int(t.tensor_type), np.ascontiguousarray(d[:ROWS]), core, t.name, os.path.basename(path))
    return picked


def run(call, ref, runs):
    outs = []
    for _ in range(runs):
        o = call()
        torch.cuda.synchronize()
        outs.append(o.float().cpu().numpy().reshape(ref.shape))
    det = all(np.array_equal(np.nan_to_num(outs[0]), np.nan_to_num(o)) for o in outs[1:])
    worst = max(np.abs(np.nan_to_num(o, posinf=0, neginf=0) - ref).max() for o in outs)
    rel = worst / (np.abs(ref).max() + 1e-12)
    nonfin = max(int((~np.isfinite(o)).sum()) for o in outs)
    return det and rel < TOL_REL and nonfin == 0, det, rel, nonfin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ggufs", nargs="+")
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--load", type=int, default=0)
    ap.add_argument("--types", default="")
    a = ap.parse_args()
    if a.load:
        # 01.10. 21:16: 6 hogs (~1 GB anon each) beside the 17 GB service drove the
        # laptop into a global OOM that killed the user's GNOME session. Never again.
        print("REFUSED: --load (memory-bandwidth hogs) is disabled on this machine")
        return 4
    if service_busy():
        print("REFUSED: live service has inflight > 0")
        return 3
    fx = pick(a.ggufs)
    if a.types:
        fx = {k: v for k, v in fx.items() if k in a.types.split(",")}
    stop = mp.Event()
    hogs = [mp.Process(target=hog, args=(stop,), daemon=True) for _ in range(a.load)]
    for h in hogs:
        h.start()
    if hogs:
        time.sleep(2)
    rng = np.random.default_rng(29)
    bad = []
    try:
        for name, (tid, raw, core, tname, fname) in sorted(fx.items()):
            if service_busy():
                print("STOPPED: user request arrived")
                return 3
            ref = gguf.quants.dequantize(raw, QT(tid)).astype(np.float64)
            rows, cols = ref.shape
            W = torch.from_numpy(raw).cuda()
            Wm = torch.from_numpy(np.repeat(raw[None], E, axis=0).copy()).cuda()
            print(f"# {name}: {fname}:{tname} {rows}x{cols}" + (" (core expert)" if core else ""))
            ops = {"dequant": (lambda: K.ggml_dequantize(W, tid, rows, cols, torch.float16, None), ref)}
            for m in (1, 2, 4, 8):
                X = torch.from_numpy(rng.standard_normal((m, cols), dtype=np.float32) * 0.1).cuda().half()
                ops[f"mmvq@{m}"] = ((lambda X=X: K.ggml_mul_mat_vec_a8(W, X, tid, rows)),
                                   X.float().cpu().numpy().astype(np.float64) @ ref.T)
            Xb = torch.from_numpy(rng.standard_normal((16, cols), dtype=np.float32) * 0.1).cuda().half()
            ops["mmq@16"] = ((lambda: K.ggml_mul_mat_a8(W, Xb, tid, rows)),
                             Xb.float().cpu().numpy().astype(np.float64) @ ref.T)
            Xt = torch.from_numpy(rng.standard_normal((T, cols), dtype=np.float32) * 0.1).cuda().half()
            reft = Xt.float().cpu().numpy().astype(np.float64) @ ref.T
            topk = torch.from_numpy(rng.integers(0, E, size=(T, 1)).astype(np.int32)).cuda()
            ops["moe_vec@8"] = ((lambda: K.ggml_moe_a8_vec(Xt, Wm, topk, 1, tid, rows, T)), reft)
            b = int(K.ggml_moe_get_block_size(tid))
            if b > 0:
                npad = ((T + b - 1) // b) * b
                sids = torch.full((npad,), T, dtype=torch.int32)
                sids[:T] = torch.arange(T, dtype=torch.int32)
                eids = torch.full((npad // b,), 2, dtype=torch.int32)
                npost = torch.tensor([npad], dtype=torch.int32)
                sids, eids, npost = sids.cuda(), eids.cuda(), npost.cuda()
                ops["moe_a8@8"] = ((lambda: K.ggml_moe_a8(Xt, Wm, sids, eids, npost, tid, rows, 1, T)), reft)
            for op, (call, rf) in ops.items():
                served = not (op.startswith("mmq") and name not in MMQ_TYPES) and not (
                    op.startswith("moe_a8") and name not in MMQ_TYPES)
                try:
                    ok, det, rel, nf = run(call, rf, a.runs)
                    detail = f"det={det} rel={rel:.2e} nonfin={nf}"
                except Exception as exc:  # noqa: BLE001
                    ok, detail = False, f"EXC {type(exc).__name__}: {str(exc)[:80]}"
                tag = "clean" if ok else ("BROKEN" if served else "n/a (no kernel; never routed)")
                if not ok and served:
                    bad.append((name, op))
                print(f"{name:<8} {op:<10} {tag:<8} {detail}", flush=True)
    finally:
        stop.set()
        for h in hogs:
            h.join(timeout=5)
    print()
    print(f"load hogs: {a.load}, runs/op: {a.runs}")
    print("SERVED-PATH BROKEN: " + (str(bad) if bad else "none"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
