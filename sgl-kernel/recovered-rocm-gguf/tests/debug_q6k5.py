"""Is the Q6_K corruption in DEVICE memory, or in the device->host read-back?

On an APU the GPU writes into GTT, which is host memory. A GPU->CPU coherence
gap would look exactly like a flaky kernel. This distinguishes them: reduce the
output ON THE GPU (never crossing to host) and compare that checksum's
stability against the stability of the host-side copy.
"""
import json
import numpy as np, torch, gguf
from gguf.constants import GGMLQuantizationType as QT
import gguf_rocm_probe as K

s = [x for x in json.load(open("slices.json")) if x["name"] == "q6_K"][0]
r, c = s["rows"], s["cols"]
raw = np.fromfile(s["path"], dtype=np.uint8).reshape(r, s["row_bytes"])
ref = gguf.quants.dequantize(raw, QT.Q6_K).astype(np.float32)
ref_t = torch.from_numpy(ref).cuda()
W = torch.from_numpy(raw).cuda()
torch.cuda.synchronize()

print(f"{'run':>3} {'gpu-side max|d|':>16} {'gpu-side nonfinite':>19} "
      f"{'host-side max|d|':>17} {'host nonfinite':>15} {'gpu sum':>16}")
for i in range(8):
    out = K.ggml_dequantize(W, 14, r, c, torch.float16, None)
    # Reduce on device: this never leaves the GPU.
    d_gpu = (out.float() - ref_t).abs().max().item()
    nf_gpu = int((~torch.isfinite(out.float())).sum().item())
    gsum = float(torch.nan_to_num(out.float(), posinf=0, neginf=0).sum().item())
    # Now the same buffer read back to host.
    a = out.float().cpu().numpy()
    d_host = float(np.abs(np.nan_to_num(a, posinf=0, neginf=0) - ref).max())
    nf_host = int((~np.isfinite(a)).sum())
    print(f"{i:>3} {d_gpu:>16.3e} {nf_gpu:>19d} {d_host:>17.3e} {nf_host:>15d} {gsum:>16.4f}")

print("\nIf gpu-side and host-side agree run by run, the corruption is in device")
print("memory (kernel). If gpu-side is stable while host-side varies, it is the")
print("device->host path.")

# Second read of the SAME buffer, no re-run of the kernel.
out = K.ggml_dequantize(W, 14, r, c, torch.float16, None)
torch.cuda.synchronize()
reads = [out.float().cpu().numpy() for _ in range(5)]
same = all(np.array_equal(np.nan_to_num(reads[0]), np.nan_to_num(x)) for x in reads[1:])
print(f"\n5 repeated host reads of ONE kernel result identical: {same}")
