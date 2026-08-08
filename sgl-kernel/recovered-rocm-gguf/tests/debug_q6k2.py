"""Is the Q6_K dequantize defect a race (run-to-run varying) or deterministic?"""
import json
import numpy as np
import torch
import gguf
from gguf.constants import GGMLQuantizationType as QT
import gguf_rocm_probe as K

s = [x for x in json.load(open("slices.json")) if x["name"] == "q6_K"][0]
raw = np.fromfile(s["path"], dtype=np.uint8).reshape(s["rows"], s["row_bytes"])
ref = gguf.quants.dequantize(raw, QT.Q6_K).astype(np.float32)
W = torch.from_numpy(raw).cuda()
r, c = s["rows"], s["cols"]

runs = []
for i in range(5):
    a = K.ggml_dequantize(W, 14, r, c, torch.float16, None).float().cpu().numpy()
    torch.cuda.synchronize()
    runs.append(a)
    print(f"run {i}: non-finite {int((~np.isfinite(a)).sum()):4d}  "
          f"finite max|d| {np.abs(np.nan_to_num(a,posinf=0,neginf=0)-ref).max():.3e}")

same = all(np.array_equal(np.nan_to_num(runs[0]), np.nan_to_num(runs[i])) for i in range(1, 5))
print(f"\nrun-to-run identical: {same}  -> {'deterministic bug' if same else 'RACE CONDITION'}")

# Does an oversized (padded) input allocation change it? Tests the OOB-read hypothesis.
pad = torch.zeros(r * s["row_bytes"] + 4096, dtype=torch.uint8, device="cuda")
pad[: r * s["row_bytes"]] = W.flatten()
Wp = pad[: r * s["row_bytes"]].view(r, s["row_bytes"])
ap = K.ggml_dequantize(Wp, 14, r, c, torch.float16, None).float().cpu().numpy()
print(f"with 4 KiB trailing pad: non-finite {int((~np.isfinite(ap)).sum())}  "
      f"finite max|d| {np.abs(np.nan_to_num(ap,posinf=0,neginf=0)-ref).max():.3e}")

# Sweep row counts to find the onset.
print("\nrow-count sweep:")
for rr in (64, 128, 192, 256, 320, 384, 448, 512):
    a = K.ggml_dequantize(W[:rr].contiguous(), 14, rr, c, torch.float16, None).float().cpu().numpy()
    nf = int((~np.isfinite(a)).sum())
    md = np.abs(np.nan_to_num(a,posinf=0,neginf=0) - ref[:rr]).max()
    print(f"  rows {rr:4d}: non-finite {nf:4d}  max|d| {md:.3e}")
