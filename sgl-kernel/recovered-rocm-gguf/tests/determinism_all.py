"""Do Q4_K / Q5_K / Q8_0 dequantize share the Q6_K race, or is it Q6_K alone?"""
import json
import numpy as np
import torch
import gguf
from gguf.constants import GGMLQuantizationType as QT
import gguf_rocm_probe as K

QT_BY_ID = {12: QT.Q4_K, 13: QT.Q5_K, 14: QT.Q6_K}
for s in json.load(open("slices.json")):
    r, c, tid = s["rows"], s["cols"], s["type"]
    raw = np.fromfile(s["path"], dtype=np.uint8).reshape(r, s["row_bytes"])
    ref = gguf.quants.dequantize(raw, QT_BY_ID[tid]).astype(np.float32)
    W = torch.from_numpy(raw).cuda()
    outs = []
    for _ in range(8):
        outs.append(K.ggml_dequantize(W, tid, r, c, torch.float16, None).float().cpu().numpy())
        torch.cuda.synchronize()
    ident = all(np.array_equal(np.nan_to_num(outs[0]), np.nan_to_num(o)) for o in outs[1:])
    worst = max(np.abs(np.nan_to_num(o, posinf=0, neginf=0) - ref).max() for o in outs)
    nf = max(int((~np.isfinite(o)).sum()) for o in outs)
    print(f"{s['name']:<6} {r}x{c:<5} 8 runs: identical={ident!s:<5} "
          f"worst max|d| {worst:.3e}  worst non-finite {nf}")
