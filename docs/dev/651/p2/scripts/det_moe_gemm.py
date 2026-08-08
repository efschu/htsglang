"""8-run determinism of ggml_moe_a8 (sorted/padded GEMM MoE variant) per type.

If this is deterministic-correct for Q6_K, it is the containment target for
the Q6_K expert stacks (blk.34/38/39) the same way MMQ is for the linear path.
All T tokens are routed to one expert; padding follows the vLLM convention
(sentinel = numel for token slots, -1 for expert slots is NOT used here since
serving masks >=E to -1 -- padding blocks beyond the real tokens keep their
expert id but sentinel token ids)."""
import json
import numpy as np
import torch
import gguf
from gguf.constants import GGMLQuantizationType as QT
import gguf_rocm_probe as K

QT_BY_ID = {12: QT.Q4_K, 13: QT.Q5_K, 14: QT.Q6_K}
N = 8
E = 8
T = 8
TOPK = 1
rng = np.random.default_rng(13)

for s in json.load(open("slices.json")):
    rows, cols, tid = s["rows"], s["cols"], s["type"]
    raw = np.fromfile(s["path"], dtype=np.uint8).reshape(rows, s["row_bytes"])
    ref = gguf.quants.dequantize(raw, QT_BY_ID[tid]).astype(np.float64)
    W = torch.from_numpy(np.repeat(raw[None, :, :], E, axis=0).copy()).cuda()
    X = torch.from_numpy((rng.standard_normal((T, cols), dtype=np.float32) * 0.1)).cuda().half()
    b = int(K.ggml_moe_get_block_size(tid))
    npad = ((T + b - 1) // b) * b
    sorted_ids = torch.full((npad,), T * TOPK, dtype=torch.int32)
    sorted_ids[:T] = torch.arange(T, dtype=torch.int32)
    expert_ids = torch.full((npad // b,), 3, dtype=torch.int32)
    num_post = torch.tensor([npad], dtype=torch.int32)
    sorted_ids, expert_ids, num_post = sorted_ids.cuda(), expert_ids.cuda(), num_post.cuda()
    refout = X.float().cpu().numpy().astype(np.float64) @ ref.T
    outs = []
    for _ in range(N):
        o = K.ggml_moe_a8(X, W, sorted_ids, expert_ids, num_post, tid, rows, TOPK, T)
        torch.cuda.synchronize()
        outs.append(o.float().cpu().numpy().reshape(T, rows))
    ident = all(np.array_equal(np.nan_to_num(outs[0]), np.nan_to_num(o)) for o in outs[1:])
    spread = max(np.abs(outs[i] - outs[0]).max() for i in range(1, N))
    worst = max(np.abs(np.nan_to_num(o, posinf=0, neginf=0) - refout).max() for o in outs)
    nf = max(int((~np.isfinite(o)).sum()) for o in outs)
    nm = s["name"]
    print(f"{nm:<6} moe_a8(b={b}) {N} runs: identical={ident!s:<5} run-spread {spread:.3e}  worst|d|vs-ref {worst:.3e}  non-finite {nf}")
