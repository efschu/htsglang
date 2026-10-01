"""Probe: does the kt LLAMAFILE CPU expert path actually COMPUTE on this ROCm box?

The stream path (submit_with_cuda_stream) returned all-zeros in probe 1, with a
stream handle of 0. This probe drives the same MOE object through the plain
cpu_infer.submit/sync pair instead -- the same pair kt itself uses for
load_weights -- so a nonzero result here isolates the failure to the CUDA-stream
bridge rather than to the kernel.
"""

import os, sys, time, traceback
import torch
from kt_kernel.utils.loader import GGUFLoader
from kt_kernel.experts_base import BaseMoEWrapper, KExpertsCPUBuffer

GGUF = "/root/651-p2/models/Qwen3.6-35B-A3B-UD-Q4KM-noQ6K.gguf"
ld = GGUFLoader(GGUF)
md = ld.metadata
ARCH = "qwen35moe"


def mdi(s):
    v = md[f"{ARCH}.{s}"]
    try:
        return int(v)
    except TypeError:
        return int(v[0])


HIDDEN, INTER, NEXP, TOPK = mdi("embedding_length"), mdi("expert_feed_forward_length"), mdi("expert_count"), mdi("expert_used_count")
print("cfg hidden=%d inter=%d nexp=%d topk=%d" % (HIDDEN, INTER, NEXP, TOPK))

NGPU = int(os.environ.get("NGPU", "0"))
mask = torch.zeros(NEXP, dtype=torch.bool)
if NGPU:
    mask[:NGPU] = True

from kt_kernel import KTMoEWrapper

w = KTMoEWrapper(
    layer_idx=0, num_experts=NEXP, num_experts_per_tok=TOPK,
    hidden_size=HIDDEN, moe_intermediate_size=INTER,
    gpu_experts_mask=mask,
    cpuinfer_threads=int(os.environ.get("CPUINFER", "8")),
    threadpool_count=1, weight_path=GGUF,
    chunked_prefill_size=256, method="LLAMAFILE",
    max_deferred_experts_per_token=0,
)
w.load_weights(torch.arange(NEXP, dtype=torch.int32))
print("loaded")


def cpu_only_forward(wrapper, x_cpu, ids_cpu, wt_cpu):
    """submit_forward's body, but with cpu_infer.submit/sync (no cuda stream)."""
    flat = x_cpu.view(-1, x_cpu.shape[-1])
    (inp, imm, defr, wts, out_cpu, bsz, out_gpu) = KExpertsCPUBuffer.get_buffer(flat, wrapper.num_experts_per_tok)
    slot = wrapper.layer_idx % KExpertsCPUBuffer.buffer_depth
    inp[slot].copy_(flat)
    wts[slot].copy_(wt_cpu)
    imm[slot].copy_(ids_cpu.to(torch.long))
    out_cpu[slot].zero_()
    wrapper.cpu_infer.submit(
        wrapper.moe.forward_task(
            bsz[slot].data_ptr(), imm[slot].size(-1), imm[slot].data_ptr(),
            wts[slot].data_ptr(), inp[slot].data_ptr(), out_cpu[slot].data_ptr(), False,
        )
    )
    wrapper.cpu_infer.sync()
    return out_cpu[slot]


for BS in [int(b) for b in os.environ.get("BSLIST", "1,8,64").split(",")]:
    torch.manual_seed(0)
    x = torch.randn(BS, HIDDEN, dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(NEXP)[:TOPK] for _ in range(BS)]).to(torch.int64)
    wt = torch.rand(BS, TOPK, dtype=torch.float32)
    wt = wt / wt.sum(-1, keepdim=True)
    try:
        o = cpu_only_forward(w, x, ids, wt)
        am = o.float().abs().mean().item()
        n = int(os.environ.get("ITERS", "20"))
        t0 = time.time()
        for _ in range(n):
            cpu_only_forward(w, x, ids, wt)
        dt = (time.time() - t0) / n
        print("BS=%-4d absmean=%.6f nonzero=%d/%d  %.3f ms/layer -> %.1f ms for 40 layers"
              % (BS, am, int(o.ne(0).sum()), o.numel(), dt * 1e3, dt * 1e3 * 40))
    except Exception:
        print("BS=%d FAILED" % BS)
        traceback.print_exc()
sys.stdout.flush()
