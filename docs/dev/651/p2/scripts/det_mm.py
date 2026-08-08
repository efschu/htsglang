"""8-run determinism of MMVQ and MMQ per K-quant type, fixed inputs.

determinism_all.py covers dequantize only. Serving decode runs MMVQ and the
Q6_K containment pins the lm_head to MMQ, so these kernels make the logits.
Inputs are CPU-sampled with a fixed seed and moved to device."""
import json
import numpy as np
import torch
import gguf
from gguf.constants import GGMLQuantizationType as QT
import gguf_rocm_probe as K

QT_BY_ID = {12: QT.Q4_K, 13: QT.Q5_K, 14: QT.Q6_K}
N = 8
rng = np.random.default_rng(7)

for s in json.load(open("slices.json")):
    rows, cols, tid = s["rows"], s["cols"], s["type"]
    raw = np.fromfile(s["path"], dtype=np.uint8).reshape(rows, s["row_bytes"])
    ref = gguf.quants.dequantize(raw, QT_BY_ID[tid]).astype(np.float64)
    W = torch.from_numpy(raw).cuda()
    X1 = torch.from_numpy((rng.standard_normal((1, cols), dtype=np.float32) * 0.1)).cuda().half()
    Xb = torch.from_numpy((rng.standard_normal((16, cols), dtype=np.float32) * 0.1)).cuda().half()
    ref1 = X1.float().cpu().numpy().astype(np.float64) @ ref.T
    refb = Xb.float().cpu().numpy().astype(np.float64) @ ref.T
    for name, call, rf in (
        ("MMVQ", lambda: K.ggml_mul_mat_vec_a8(W, X1, tid, rows), ref1),
        ("MMQ",  lambda: K.ggml_mul_mat_a8(W, Xb, tid, rows), refb),
    ):
        outs = []
        for _ in range(N):
            o = call(); torch.cuda.synchronize()
            outs.append(o.float().cpu().numpy())
        ident = all(np.array_equal(np.nan_to_num(outs[0]), np.nan_to_num(o)) for o in outs[1:])
        spread = max((np.abs(outs[i] - outs[0]).max() for i in range(1, N)), default=0.0)
        worst = max(np.abs(np.nan_to_num(o, posinf=0, neginf=0).astype(np.float64) - rf).max() for o in outs)
        nf = max(int((~np.isfinite(o)).sum()) for o in outs)
        print(f"{s['name']:<6} {name:<4} {N} runs: identical={ident!s:<5} "
              f"run-spread {spread:.3e}  worst|d|vs-ref {worst:.3e}  non-finite {nf}")
