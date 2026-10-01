"""Correctness audit of every GPU kernel x every quant type in the checkpoint.

The recovered harness validated Q4_K/Q5_K/Q6_K only; Q8_0 (259 of 753 tensors,
most experts) was never validated on this GPU. Slices are real tensor bytes
read straight from the GGUF; oracle is gguf.quants.dequantize (numpy).
Ops: dequantize, MMVQ, MMQ, moe_a8, moe_a8_vec -- 4-run determinism + error."""
import sys
import numpy as np
import torch
import gguf
from gguf import GGUFReader
from gguf.constants import GGMLQuantizationType as QT
import gguf_rocm_probe as K

SRC = sys.argv[1] if len(sys.argv) > 1 else "/root/lh/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
ROWS = 512
N = 4
E, T = 4, 8
rng = np.random.default_rng(23)
reader = GGUFReader(SRC)

# one representative 2D-sliceable tensor per quant type
picked = {}
for t in reader.tensors:
    tt = t.tensor_type
    if tt.name in ("F32", "BF16") or tt.name in picked:
        continue
    d = t.data
    if d.ndim == 3:
        d = d.reshape(-1, d.shape[-1])
    if d.ndim != 2 or d.shape[0] < ROWS:
        continue
    picked[tt.name] = (t.name, int(tt), d[:ROWS].copy())

for name, (tname, tid, raw) in sorted(picked.items()):
    ref = gguf.quants.dequantize(raw, QT(tid)).astype(np.float64)
    rows, cols = ref.shape
    W = torch.from_numpy(raw).cuda()
    X1 = torch.from_numpy((rng.standard_normal((1, cols), dtype=np.float32) * 0.1)).cuda().half()
    Xb = torch.from_numpy((rng.standard_normal((16, cols), dtype=np.float32) * 0.1)).cuda().half()
    Wm = torch.from_numpy(np.repeat(raw[None], E, axis=0).copy()).cuda()
    Xt = torch.from_numpy((rng.standard_normal((T, cols), dtype=np.float32) * 0.1)).cuda().half()
    topk = torch.from_numpy(rng.integers(0, E, size=(T, 1)).astype(np.int32)).cuda()
    b = int(K.ggml_moe_get_block_size(tid))
    has_moe = b > 0
    npad = ((T + b - 1) // b) * b if has_moe else T
    sids = torch.full((npad,), T, dtype=torch.int32); sids[:T] = torch.arange(T, dtype=torch.int32)
    eids = torch.full((max(npad // b, 1) if has_moe else 1,), 2, dtype=torch.int32)
    npost = torch.tensor([npad], dtype=torch.int32)
    sids, eids, npost = sids.cuda(), eids.cuda(), npost.cuda()

    ref1 = X1.float().cpu().numpy().astype(np.float64) @ ref.T
    refb = Xb.float().cpu().numpy().astype(np.float64) @ ref.T
    reft = Xt.float().cpu().numpy().astype(np.float64) @ ref.T

    ops = {
        "dequant":    (lambda: K.ggml_dequantize(W, tid, rows, cols, torch.float16, None), ref),
        "mmvq":       (lambda: K.ggml_mul_mat_vec_a8(W, X1, tid, rows), ref1),
        "mmq":        (lambda: K.ggml_mul_mat_a8(W, Xb, tid, rows), refb),
        "moe_a8":     (lambda: K.ggml_moe_a8(Xt, Wm, sids, eids, npost, tid, rows, 1, T), reft),
        "moe_a8_vec": (lambda: K.ggml_moe_a8_vec(Xt, Wm, topk, 1, tid, rows, T), reft),
    }
    if not has_moe:
        ops.pop("moe_a8"); ops.pop("moe_a8_vec")
    print(f"=== {name} ({tname}, {rows}x{cols}) ===")
    for op, (call, rf) in ops.items():
        try:
            outs = []
            for _ in range(N):
                o = call(); torch.cuda.synchronize()
                outs.append(o.float().cpu().numpy().reshape(np.asarray(rf).shape))
            ident = all(np.array_equal(np.nan_to_num(outs[0]), np.nan_to_num(o)) for o in outs[1:])
            worst = max(np.abs(np.nan_to_num(o, posinf=0, neginf=0) - rf).max() for o in outs)
            scale = np.abs(rf).max() + 1e-12
            nf = max(int((~np.isfinite(o)).sum()) for o in outs)
            print(f"  {op:<11} det={ident!s:<5} worst|d| {worst:.3e} (rel {worst/scale:.2e}) nonfin {nf}")
        except Exception as e:
            print(f"  {op:<11} ERROR {type(e).__name__}: {str(e)[:90]}")
