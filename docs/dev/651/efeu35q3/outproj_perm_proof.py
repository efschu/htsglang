#!/usr/bin/env python
"""efeu-TP14: exactness proof of the GDN out_proj activation permutation, CPU only.

For every GDN layer of the served GGUF: dequantize out_proj (Q3_K, tiled column
order), build the grouped weight exactly as the old dense path did
(GGUFQwen35Adapter._undo_v_tiling along dim 1), and compare
    W_grouped @ x            (old dense path, in float64)
    W_tiled   @ x[..., inv]  (new packed path's math, in float64)
for random x. Identical up to float64 rounding <=> the permutation is right; a
wrong permutation gives a relative error of order 1.

    python outproj_perm_proof.py <gguf>
"""
import sys

import numpy as np
import gguf
from gguf import GGUFReader

src = sys.argv[1]
num_k, num_v, head_v = 16, 32, 128
vpk = num_v // num_k
V = num_v * head_v
idx = np.arange(V).reshape(vpk, num_k, head_v).transpose(1, 0, 2).reshape(-1)
inv = np.argsort(idx)
rng = np.random.default_rng(0)
x = rng.standard_normal((4, V))
worst = 0.0
n = 0
for t in GGUFReader(src).tensors:
    if not t.name.endswith("ssm_out.weight"):
        continue
    W = gguf.quants.dequantize(np.array(t.data), t.tensor_type).astype(np.float64)  # [hidden, V] tiled
    # old path: _undo_v_tiling(weight, dim=1, head_units=head_v)
    Wg = W.reshape(W.shape[0], vpk, num_k, head_v).transpose(0, 2, 1, 3).reshape(W.shape[0], V)
    a = x @ Wg.T
    b = x[:, inv] @ W.T
    rel = np.abs(a - b).max() / np.abs(a).max()
    worst = max(worst, rel)
    n += 1
print(f"{n} ssm_out layers ({t.tensor_type.name}); worst relative difference {worst:.2e}")
print("EXACT" if worst < 1e-12 else "MISMATCH")
