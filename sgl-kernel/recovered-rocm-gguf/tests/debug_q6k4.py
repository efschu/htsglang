"""Narrow the Q6_K defect: in-kernel hazard, or launch-ordering/visibility?"""
import json, os
import numpy as np, torch, gguf
from gguf.constants import GGMLQuantizationType as QT
import gguf_rocm_probe as K

s = [x for x in json.load(open("slices.json")) if x["name"] == "q6_K"][0]
r, c = s["rows"], s["cols"]
raw = np.fromfile(s["path"], dtype=np.uint8).reshape(r, s["row_bytes"])
ref = gguf.quants.dequantize(raw, QT.Q6_K).astype(np.float32)
W = torch.from_numpy(raw).cuda(); torch.cuda.synchronize()

def once(sync_around):
    if sync_around: torch.cuda.synchronize()
    out = torch.empty((r, c), dtype=torch.float16, device="cuda")
    if sync_around: torch.cuda.synchronize()
    a = K.ggml_dequantize(W, 14, r, c, torch.float16, out)
    if sync_around: torch.cuda.synchronize()
    a = a.float().cpu().numpy()
    return np.abs(np.nan_to_num(a, posinf=0, neginf=0) - ref).max()

print(f"AMD_SERIALIZE_KERNEL={os.environ.get('AMD_SERIALIZE_KERNEL')}")
print("no extra sync :", " ".join(f"{once(False):.2e}" for _ in range(6)))
print("sync around   :", " ".join(f"{once(True):.2e}" for _ in range(6)))

# Same input, same output buffer, kernel run repeatedly WITHOUT reallocating.
out = torch.empty((r, c), dtype=torch.float16, device="cuda")
errs = []
for _ in range(6):
    K.ggml_dequantize(W, 14, r, c, torch.float16, out)
    torch.cuda.synchronize()
    a = out.float().cpu().numpy()
    errs.append(np.abs(np.nan_to_num(a, posinf=0, neginf=0) - ref).max())
print("one fixed buf :", " ".join(f"{e:.2e}" for e in errs))

# Smaller launch: chunk the work into <=256-row pieces, which were always clean.
outc = torch.empty((r, c), dtype=torch.float16, device="cuda")
for _ in range(3):
    for i in range(0, r, 256):
        part = K.ggml_dequantize(W[i:i+256].contiguous(), 14, 256, c, torch.float16, None)
        outc[i:i+256] = part
    torch.cuda.synchronize()
    a = outc.float().cpu().numpy()
    print(f"chunked 256   : {np.abs(np.nan_to_num(a,posinf=0,neginf=0)-ref).max():.2e}")
