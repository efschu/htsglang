"""#651: the REAL serving MoE path (fused_moe_gguf) at REAL geometry with REAL
distinct experts vs the validated numpy oracle.

The synthetic kernel audit used replicated expert stacks (any expert-indexing
bug is invisible when all experts are identical) — this is the falsifier that
closes that blind spot, using the exact dumped inputs of the incoherent
serving forward (layer 0, 5 tokens)."""
import numpy as np, torch, gguf
from gguf import GGUFReader
from gguf.constants import GGMLQuantizationType as QT
from sglang.srt.server_args import ServerArgs
from sglang.srt.runtime_context import _CONTEXT
_CONTEXT.set_server_args(ServerArgs(
    model_path="/root/651-p2/models/Qwen3.6-35B-A3B-UD-Q4KM-noQ6K.gguf",
    tokenizer_path="/root/lh/models", load_format="gguf", quantization="gguf",
    device="cuda", disable_cuda_graph=True, mamba_radix_cache_strategy="no_buffer", disable_overlap_schedule=True, disable_radix_cache=True, attention_backend="triton", sampling_backend="pytorch"))
from sglang.srt.layers.quantization.gguf import fused_moe_gguf

d = torch.load("/root/651-p2/dumps/TP0_PP0_Rank0_pid14808/Pass00000.pt", map_location="cpu", weights_only=False)
x = d["model.layers.0.post_attention_layernorm"]
if isinstance(x, (tuple, list)): x = x[0]
w, ids, _ = d["model.layers.0.mlp.topk"]
ours_serving = d["model.layers.0.mlp.experts"]
if isinstance(ours_serving, (tuple, list)): ours_serving = ours_serving[0]

r = GGUFReader("/root/651-p2/models/Qwen3.6-35B-A3B-UD-Q4KM-noQ6K.gguf")
t = {t.name: t for t in r.tensors}
gate_p = np.asarray(t["blk.0.ffn_gate_exps.weight"].data)  # (256, 512, 1152)
up_p   = np.asarray(t["blk.0.ffn_up_exps.weight"].data)
down_p = np.asarray(t["blk.0.ffn_down_exps.weight"].data)  # (256, 2048, 352)
print("packed shapes:", gate_p.shape, up_p.shape, down_p.shape)
w13 = np.concatenate([gate_p, up_p], axis=1)               # (256, 1024, 1152)
qt1 = int(t["blk.0.ffn_gate_exps.weight"].tensor_type)     # Q4_K
qt2 = int(t["blk.0.ffn_down_exps.weight"].tensor_type)     # Q5_K
print("types:", qt1, qt2)

dev = "cuda"
outs = {}
for label, ids_t in (("int32-ids", ids.to(torch.int32)), ("int64-ids", ids.to(torch.int64))):
    outs[label] = None
out = fused_moe_gguf(
    x.to(dev).half() if x.dtype == torch.float32 else x.to(dev),
    torch.from_numpy(w13.copy()).to(dev),
    torch.from_numpy(down_p.copy()).to(dev),
    w.to(dev), ids.to(dev).to(torch.int32),
    qt1, qt2, "silu",
)
torch.cuda.synchronize()
out64 = fused_moe_gguf(
    x.to(dev),
    torch.from_numpy(w13.copy()).to(dev),
    torch.from_numpy(down_p.copy()).to(dev),
    w.to(dev), ids.to(dev).to(torch.int64),
    qt1, qt2, "silu",
)
torch.cuda.synchronize()
out64 = out64.float().cpu().numpy()
out = out.float().cpu().numpy()

# numpy oracle (validated against llama.cpp)
def dq(name):
    tt = t[name]
    return gguf.quants.dequantize(tt.data.reshape(-1, tt.data.shape[-1]), QT(int(tt.tensor_type))).astype(np.float64)
E, FF, H = 256, 512, 2048
gate = dq("blk.0.ffn_gate_exps.weight").reshape(E, FF, H)
up   = dq("blk.0.ffn_up_exps.weight").reshape(E, FF, H)
down = dq("blk.0.ffn_down_exps.weight").reshape(E, H, FF)
def silu(a): return a / (1.0 + np.exp(-a))
xr = x.float().numpy().astype(np.float64)
ref = np.zeros((5, H))
for tok in range(5):
    for k in range(8):
        e = int(ids[tok, k]); wk = float(w[tok, k])
        ref[tok] += wk * (down[e] @ (silu(gate[e] @ xr[tok]) * (up[e] @ xr[tok])))

for name, o in (("standalone int32-ids", out), ("standalone int64-ids", out64), ("serving dump", ours_serving.float().numpy())):
    corr = np.corrcoef(ref.flatten(), o.flatten())[0, 1]
    rel = np.abs(o - ref).max() / np.abs(ref).max()
    print(f"{name}: corr {corr:.4f} rel {rel:.3f} sum {o.sum():.4f} (ref {ref.sum():.4f})")
