"""Validate the JIT moe_align kernel (laptop-patched moe_align_kernel.cu,
compile-fixed for HIP but never result-validated) against invariants.

Serving routes through it on this host because the AOT kernel is absent.
Wrong sorted ids = tokens multiplied by the WRONG experts weights =
deterministic fluent-degenerate output, invisible to per-kernel GEMM audits."""
import numpy as np
import torch
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import moe_align_block_size

rng = np.random.default_rng(3)

def check(T, topk, E, bs):
    ids = np.stack([rng.choice(E, size=topk, replace=False) for _ in range(T)])
    topk_ids = torch.from_numpy(ids.astype(np.int32)).cuda()
    sorted_ids, expert_ids, npost = moe_align_block_size(topk_ids, bs, E)
    torch.cuda.synchronize()
    numel = T * topk
    npost_v = int(npost.item())
    s = sorted_ids.cpu().numpy()
    e = expert_ids.cpu().numpy()
    flat = ids.flatten()
    ok = True
    if npost_v % bs != 0:
        ok = False; print(f"  FAIL npost {npost_v} not divisible by {bs}")
    valid_positions = []
    for bi in range(npost_v // bs):
        exp = e[bi]
        for j in range(bi * bs, (bi + 1) * bs):
            tokpos = s[j]
            if tokpos < numel:
                valid_positions.append(tokpos)
                if flat[tokpos] != exp:
                    ok = False
                    if len(valid_positions) < 8:
                        print(f"  FAIL slot {j}: token-pos {tokpos} has expert {flat[tokpos]}, block says {exp}")
    vp = np.sort(np.array(valid_positions))
    if not np.array_equal(vp, np.arange(numel)):
        ok = False; print(f"  FAIL coverage: {len(vp)} valid entries, {len(np.unique(vp))} unique, expected {numel}")
    print(f"T={T} topk={topk} E={E} bs={bs}: npost={npost_v} {'PASS' if ok else 'FAIL'}")
    return ok

allok = True
for T, topk in ((1, 8), (8, 8), (64, 8), (1024, 8)):
    allok &= check(T, topk, 256, 8)
print("VERDICT:", "moe_align OK" if allok else "moe_align BROKEN")
