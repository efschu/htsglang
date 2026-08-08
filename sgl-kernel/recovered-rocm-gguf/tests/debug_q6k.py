"""Characterise the Q6_K dequantize failure on gfx1103."""
import json
import numpy as np
import torch
import gguf
from gguf.constants import GGMLQuantizationType as QT
import gguf_rocm_probe as K

s = [x for x in json.load(open("slices.json")) if x["name"] == "q6_K"][0]
rows, cols, rb = s["rows"], s["cols"], s["row_bytes"]
raw = np.fromfile(s["path"], dtype=np.uint8).reshape(rows, rb)
ref = gguf.quants.dequantize(raw, QT.Q6_K).astype(np.float32)
W = torch.from_numpy(raw).cuda()

for tag, r, c in (("full 512x512", rows, cols), ("256x512", 256, cols), ("1x512", 1, cols)):
    out = K.ggml_dequantize(W[:r].contiguous(), 14, r, c, torch.float16, None)
    a = out.float().cpu().numpy()
    bad = ~np.isfinite(a)
    print(f"{tag:<14} shape {tuple(a.shape)}  non-finite {bad.sum():6d}/{a.size} "
          f"({100.0*bad.sum()/a.size:5.2f}%)")
    if bad.any():
        idx = np.argwhere(bad)
        print(f"   first bad at row {idx[0][0]} col {idx[0][1]}; "
              f"bad cols unique: {sorted(set(idx[:,1].tolist()))[:16]}")
        print(f"   bad per row (first 8 rows): {[int(bad[i].sum()) for i in range(min(8,r))]}")
    fin = np.isfinite(a)
    if fin.any():
        print(f"   finite part max|d| {np.abs(a[fin]-ref[:r][fin]).max():.3e}")

# fp32 output instead of fp16
out32 = K.ggml_dequantize(W, 14, rows, cols, torch.float32, None)
a32 = out32.cpu().numpy()
print(f"\nfp32 out: non-finite {int((~np.isfinite(a32)).sum())}/{a32.size}")
if np.isfinite(a32).all():
    print(f"   max|d| vs ref {np.abs(a32-ref).max():.3e}  -> fp16-specific")
