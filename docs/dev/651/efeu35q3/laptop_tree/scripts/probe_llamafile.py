import os, sys, time, traceback
import torch
from kt_kernel.utils.loader import GGUFLoader

GGUF = "/root/651-p2/models/Qwen3.6-35B-A3B-UD-Q4KM-noQ6K.gguf"

ld = GGUFLoader(GGUF)
md = ld.metadata
for k in sorted(md):
    if any(s in k for s in ("embedding_length", "expert", "block_count", "feed_forward")):
        print("META", k, md[k])

for name in ("blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight"):
    info = ld.tensor_info.get(name)
    print("TENSOR", name, info["shape"] if info else None, info["dtype"].name if info else None)

ARCH = str(md.get("general.architecture"))
if isinstance(md.get("general.architecture"), (bytes,)):
    ARCH = md["general.architecture"].decode()
ARCH = ARCH.strip("[]'\" ")


def mdi(suffix, default=0):
    v = md.get(f"{ARCH}.{suffix}", default)
    try:
        return int(v)
    except TypeError:
        return int(v[0])


HIDDEN = mdi("embedding_length")
INTER = mdi("expert_feed_forward_length")
NEXP = mdi("expert_count")
TOPK = mdi("expert_used_count")
print("DERIVED hidden=%s inter=%s nexp=%s topk=%s" % (HIDDEN, INTER, NEXP, TOPK))
if not all((HIDDEN, INTER, NEXP, TOPK)):
    print("METADATA INCOMPLETE; all keys:", sorted(md)[:80])
    sys.exit(2)

NGPU = int(os.environ.get("NGPU", "0"))
mask = torch.zeros(NEXP, dtype=torch.bool)
if NGPU:
    mask[:NGPU] = True

from kt_kernel import KTMoEWrapper

t0 = time.time()
w = KTMoEWrapper(
    layer_idx=0,
    num_experts=NEXP,
    num_experts_per_tok=TOPK,
    hidden_size=HIDDEN,
    moe_intermediate_size=INTER,
    gpu_experts_mask=mask,
    cpuinfer_threads=int(os.environ.get("CPUINFER", "8")),
    threadpool_count=1,
    weight_path=GGUF,
    chunked_prefill_size=256,
    method="LLAMAFILE",
    max_deferred_experts_per_token=int(os.environ.get("DEFER", "0")),
)
print("WRAPPER OK type=%s ctor=%.1fs" % (type(w).__name__, time.time() - t0))

t0 = time.time()
w.load_weights(torch.arange(NEXP, dtype=torch.int32))
print("LOAD_WEIGHTS OK %.1fs" % (time.time() - t0))

BS = int(os.environ.get("BS", "1"))
torch.manual_seed(0)
x_cpu = torch.randn(BS, HIDDEN, dtype=torch.bfloat16)
ids_cpu = torch.stack([torch.randperm(NEXP)[:TOPK] for _ in range(BS)]).to(torch.int64)
wt_cpu = torch.rand(BS, TOPK, dtype=torch.float32)
wt_cpu = wt_cpu / wt_cpu.sum(-1, keepdim=True)

# --- CPU-only sync path first (no stream involved) ---
try:
    out = w.forward(x_cpu, ids_cpu, wt_cpu)
    print("FORWARD(cpu-sync) OK out", tuple(out.shape), out.dtype, out.device,
          "absmean=%.5f" % out.float().abs().mean().item())
except Exception:
    print("FORWARD(cpu-sync) FAILED")
    traceback.print_exc()

# --- stream path, the one sglang actually uses ---
if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    x = x_cpu.to(dev)
    ids = ids_cpu.to(dev)
    wts = wt_cpu.to(dev)
    st = torch.cuda.current_stream(dev).cuda_stream
    print("STREAM handle", st)
    try:
        w.submit_forward(x, ids, wts, st)
        o = w.sync_forward(x, st)
        torch.cuda.synchronize()
        print("SUBMIT/SYNC OK out", tuple(o.shape), o.dtype, o.device,
              "absmean=%.5f" % o.float().abs().mean().item())
        # timing
        N = int(os.environ.get("ITERS", "20"))
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(N):
            w.submit_forward(x, ids, wts, st)
            o = w.sync_forward(x, st)
        torch.cuda.synchronize()
        dt = (time.time() - t0) / N
        print("PERLAYER ms=%.3f  (x40 layers -> %.1f ms/token)" % (dt * 1e3, dt * 1e3 * 40))
    except Exception:
        print("SUBMIT/SYNC FAILED")
        traceback.print_exc()
else:
    print("no torch.cuda")
