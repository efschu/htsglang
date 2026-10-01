#!/usr/bin/env python
"""efeu-TP14: repro for the warmup launch failure in _forward_input_proj.

Q3_K_M quantizes ssm_alpha / ssm_beta (in_proj_ba: 2 shards x 32 rows x 2048).
Run every GGUF matmul kernel on exactly that tensor shape for a ladder of token
counts, one launch at a time with a sync after each, and compare to the numpy
oracle -- which kernel/M faults or computes garbage?

    HIP_LAUNCH_BLOCKING=1 python small_rows_repro.py <gguf> [tensor] [M,...]
"""

import importlib
import os
import sys

import numpy as np
import torch
import gguf
from gguf import GGUFReader

src = sys.argv[1]
tname = sys.argv[2] if len(sys.argv) > 2 else "blk.0.ssm_beta.weight"
Ms = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "1,2,3,4,8,16,32,64,128,256").split(",")]
K = importlib.import_module(os.environ.get("GGUF_MODULE", "sglang_gguf_rocm"))
r = GGUFReader(src)
t = [x for x in r.tensors if x.name == tname][0]
qt = t.tensor_type
raw = np.array(t.data.reshape(-1, t.data.shape[-1]))
rows = raw.shape[0]
bs, ts = gguf.GGML_QUANT_SIZES[qt]
cols = raw.shape[1] // ts * bs
ref = gguf.quants.dequantize(raw, qt).astype(np.float64)
print(f"{tname} {qt.name} rows={rows} cols={cols} raw={raw.shape}", flush=True)
W = torch.from_numpy(raw).cuda()
rng = np.random.default_rng(1)
for M in Ms:
    x = torch.from_numpy(rng.standard_normal((M, cols)).astype(np.float32) * 0.1).cuda().to(torch.bfloat16)
    rf = x.float().cpu().numpy().astype(np.float64) @ ref.T
    for name, fn in (("mmvq", lambda: K.ggml_mul_mat_vec_a8(W, x, int(qt), rows)),
                     ("mmq", lambda: K.ggml_mul_mat_a8(W, x, int(qt), rows)),
                     ("dequant", lambda: x @ K.ggml_dequantize(W, int(qt), rows, cols, x.dtype, None).T)):
        if name == "mmvq" and M > 16:
            continue
        try:
            y = fn()
            torch.cuda.synchronize()
            e = float(np.abs(y.float().cpu().numpy() - rf).max() / (np.abs(rf).max() + 1e-12))
            print(f"  M={M:4d} {name:8s} rel err {e:.2e}", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"  M={M:4d} {name:8s} FAILED {type(ex).__name__}: {str(ex)[:120]}", flush=True)
            raise SystemExit(1)
print("ALL OK")
