"""One dequantize call in a fresh process: is the defect present from the first
call, or does it appear only after GPU state accumulates?"""
import json, sys
import numpy as np, torch, gguf
from gguf.constants import GGMLQuantizationType as QT
import gguf_rocm_probe as K

nm = sys.argv[1]
s = [x for x in json.load(open("slices.json")) if x["name"] == nm][0]
r, c, tid = s["rows"], s["cols"], s["type"]
raw = np.fromfile(s["path"], dtype=np.uint8).reshape(r, s["row_bytes"])
ref = gguf.quants.dequantize(raw, {12: QT.Q4_K, 13: QT.Q5_K, 14: QT.Q6_K}[tid]).astype(np.float32)
W = torch.from_numpy(raw).cuda(); torch.cuda.synchronize()
a = K.ggml_dequantize(W, tid, r, c, torch.float16, None).cpu().numpy().astype(np.float32)
nf = int((~np.isfinite(a)).sum())
print(f"{np.abs(np.nan_to_num(a,posinf=0,neginf=0)-ref).max():.3e} {nf}")
