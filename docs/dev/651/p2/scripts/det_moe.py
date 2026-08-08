"""8-run determinism of ggml_moe_a8_vec per K-quant type.

The Q6_K containment covers the LINEAR path (MMQ pin). The checkpoint carries
Q6_K on three MoE expert down-proj stacks (blk.34/38/39), which dispatch to the
MoE kernels instead -- this asks whether the Q6_K defect reaches them."""
import json
import numpy as np
import torch
import gguf
from gguf.constants import GGMLQuantizationType as QT
import gguf_rocm_probe as K

QT_BY_ID = {12: QT.Q4_K, 13: QT.Q5_K, 14: QT.Q6_K}
N = 8
E = 8       # experts
T = 8       # tokens
rng = np.random.default_rng(11)

for s in json.load(open("slices.json")):
    rows, cols, tid = s["rows"], s["cols"], s["type"]
    raw = np.fromfile(s["path"], dtype=np.uint8).reshape(rows, s["row_bytes"])
    ref = gguf.quants.dequantize(raw, QT_BY_ID[tid]).astype(np.float64)
    W = torch.from_numpy(np.repeat(raw[None, :, :], E, axis=0).copy()).cuda()
    X = torch.from_numpy((rng.standard_normal((T, cols), dtype=np.float32) * 0.1)).cuda().half()
    topk = torch.from_numpy(rng.integers(0, E, size=(T, 1)).astype(np.int32)).cuda()
    refout = X.float().cpu().numpy().astype(np.float64) @ ref.T  # same for every expert
    outs = []
    for _ in range(N):
        o = K.ggml_moe_a8_vec(X, W, topk, 1, tid, rows, T)
        torch.cuda.synchronize()
        outs.append(o.float().cpu().numpy().reshape(T, rows))
    ident = all(np.array_equal(np.nan_to_num(outs[0]), np.nan_to_num(o)) for o in outs[1:])
    spread = max(np.abs(outs[i] - outs[0]).max() for i in range(1, N))
    worst = max(np.abs(np.nan_to_num(o, posinf=0, neginf=0) - refout).max() for o in outs)
    nf = max(int((~np.isfinite(o)).sum()) for o in outs)
    nm = s["name"]
    print(f"{nm:<6} moe_a8_vec {N} runs: identical={ident!s:<5} run-spread {spread:.3e}  worst|d|vs-ref {worst:.3e}  non-finite {nf}")
