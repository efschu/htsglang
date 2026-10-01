"""#651: GDN triton kernels (the SERVING path) vs the torch-naive reference.

The CPU-CI test targets the sgl_kernel AOT ops (absent here). This drives the
fla triton `chunk_gated_delta_rule` (prefill) at the real 35B geometry
(HK=16, HV=32, K=V=128) against `torch_chunk_gated_delta_rule` from
test_mamba.py, plus an 8-run determinism check. Inputs sampled on CPU."""
import sys
sys.path.insert(0, "/root/651-p2/scripts")
import numpy as np
import torch
from test_mamba import chunk_gated_delta_rule_update
from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule

torch.manual_seed(3)
B, T, HK, HV, K, V = 1, 128, 16, 32, 128, 128
q = torch.randn(B, T, HK, K, dtype=torch.bfloat16)
k = torch.randn(B, T, HK, K, dtype=torch.bfloat16)
v = torch.randn(B, T, HV, V, dtype=torch.bfloat16)
g = torch.nn.functional.logsigmoid(torch.randn(B, T, HV, dtype=torch.float32))
beta = torch.rand(B, T, HV, dtype=torch.bfloat16)
h0 = torch.zeros(B, HV, K, V, dtype=torch.float32)

cu = torch.tensor([0, T], dtype=torch.long)
ref_out, ref_state = chunk_gated_delta_rule_update(
    q.float(), k.float(), v.float(), g.float(), beta.float(),
    cu, h0.transpose(-1, -2).contiguous(), True)

dev = "cuda"
POOL = 4
SLOT = 2
outs, states = [], []
for i in range(8):
    pool = torch.zeros(POOL, HV, K, V, dtype=torch.float32, device=dev)
    res = chunk_gated_delta_rule(
        q.view(1, T, HK, K).to(dev), k.view(1, T, HK, K).to(dev),
        v.view(1, T, HV, V).to(dev), g.view(1, T, HV).to(dev),
        beta.view(1, T, HV).to(dev),
        initial_state=pool,
        initial_state_indices=torch.tensor([SLOT], dtype=torch.int32, device=dev),
        cu_seqlens=torch.tensor([0, T], dtype=torch.int32, device=dev),
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()
    o = res[0] if isinstance(res, tuple) else res
    outs.append(o.float().cpu().view(T, HV, V))
    states.append(pool[SLOT].float().cpu())

ident = all(torch.equal(outs[0], o) for o in outs[1:])
ro = ref_out.float().view(T, HV, V)
err = (outs[0] - ro).abs().max().item()
scale = ro.abs().max().item()
nf = int((~torch.isfinite(outs[0])).sum().item())
spread = max(((outs[i] - outs[0]).abs().max().item()) for i in range(1, 8))
print(f"kernel out absmax {outs[0].abs().max().item():.4f}, ref absmax {ro.abs().max().item():.4f}, corr {torch.corrcoef(torch.stack([outs[0].flatten(), ro.flatten()]))[0,1].item():.4f}")
print(f"chunk_gated_delta_rule (triton, 35B geometry {B}x{T}x{HK}/{HV}x{K}):")
print(f"  8-run identical={ident} run-spread {spread:.3e}")
print(f"  vs naive ref: max|d| {err:.3e} (ref absmax {scale:.3f}, rel {err/scale:.2e}) nonfin {nf}")
verdict_ok = ident and err / scale < 5e-2 and nf == 0
print("VERDICT:", "GDN-CHUNK CLEAN" if verdict_ok else "GDN-CHUNK SUSPECT")
