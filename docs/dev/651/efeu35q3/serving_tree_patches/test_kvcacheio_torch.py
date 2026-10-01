"""kvcacheio_torch vs a literal port of the C++ per-page loops (transfer.cu).

    python test_kvcacheio_torch.py [cuda|cpu]
"""

import sys

import torch

import kvcacheio_torch as T

dev = sys.argv[1] if len(sys.argv) > 1 else "cuda"
g = torch.Generator().manual_seed(0)


def ref_page(src, dst, s, d, ps):
    dst[d:d + ps].copy_(src[s:s + ps])


def ref_lf_pf(src_ptrs, dst_ptrs, si, di, ps, start_layer=0):
    is_mla = len(dst_ptrs) == 1
    L = len(src_ptrs) if is_mla else len(src_ptrs) // 2
    for i in range(si.numel() // ps):
        s, d = int(si[i * ps]), int(di[i * ps]) // ps
        for j in range(L):
            ref_page(src_ptrs[j], dst_ptrs[0][d][start_layer + j], s, 0, ps)
            if not is_mla:
                ref_page(src_ptrs[j + L], dst_ptrs[1][d][start_layer + j], s, 0, ps)


def ref_pf_lf(src_ptrs, dst_ptrs, si, di, layer_id, ps):
    is_mla = len(src_ptrs) == 1
    L = len(dst_ptrs) if is_mla else len(dst_ptrs) // 2
    for i in range(si.numel() // ps):
        s, d = int(si[i * ps]) // ps, int(di[i * ps])
        for j in range(L):
            ref_page(src_ptrs[0][s][layer_id + j], dst_ptrs[j], 0, d, ps)
            if not is_mla:
                ref_page(src_ptrs[1][s][layer_id + j], dst_ptrs[j + L], 0, d, ps)


def pages(n_pages_total, n, ps, device):
    p = torch.randperm(n_pages_total, generator=g)[:n]
    return (p[:, None] * ps + torch.arange(ps)).reshape(-1).to(device)


ok = True
for ps in (1, 4):
    for mla in (False, True):
        L, H, D, NTOK, NPG = 3, 2, 8, 256, 64
        shape = (NTOK, 1, D) if mla else (NTOK, H, D)
        dev_layers = [torch.randn(shape, generator=g).to(dev) for _ in range(L if mla else 2 * L)]
        host_shape = (NPG, L, ps) + shape[1:]
        host = [torch.zeros(host_shape) for _ in range(1 if mla else 2)]
        host_ref = [h.clone() for h in host]
        n = 10
        si = pages(NTOK // ps, n, ps, dev)
        di = pages(NPG, n, ps, "cpu")
        T.transfer_kv_all_layer_direct_lf_pf(dev_layers, host, si, di, ps)
        ref_lf_pf([x.cpu() for x in dev_layers], host_ref, si.cpu(), di, ps)
        torch.cuda.synchronize() if dev == "cuda" else None
        good = all(torch.equal(a, b) for a, b in zip(host, host_ref))
        print(f"lf->pf ps={ps} mla={mla}: {'OK' if good else 'MISMATCH'}")
        ok &= good
        # back: host page-first -> fresh device layers, one layer at a time
        back = [torch.zeros_like(x) for x in dev_layers]
        back_ref = [torch.zeros_like(x).cpu() for x in dev_layers]
        di2 = pages(NTOK // ps, n, ps, dev)
        for layer in range(L):
            dst = [back[layer]] if mla else [back[layer], back[layer + L]]
            dst_r = [back_ref[layer]] if mla else [back_ref[layer], back_ref[layer + L]]
            T.transfer_kv_per_layer_direct_pf_lf(host, dst, di, di2, layer, ps)
            ref_pf_lf(host_ref, dst_r, di, di2.cpu(), layer, ps)
        torch.cuda.synchronize() if dev == "cuda" else None
        good = all(torch.equal(a.cpu(), b) for a, b in zip(back, back_ref))
        print(f"pf->lf ps={ps} mla={mla}: {'OK' if good else 'MISMATCH'}")
        ok &= good
# transfer_kv_direct
src = [torch.randn(64, 2, 8, generator=g).to(dev) for _ in range(4)]
dst = [torch.zeros(64, 2, 8) for _ in range(4)]
si = torch.randperm(64, generator=g)[:20].to(dev)
di = torch.randperm(64, generator=g)[:20]
T.transfer_kv_direct(src, dst, si, di, 1)
torch.cuda.synchronize() if dev == "cuda" else None
good = all(torch.equal(d[di], s.cpu()[si.cpu()]) for s, d in zip(src, dst))
print(f"direct: {'OK' if good else 'MISMATCH'}")
ok &= good
print("ALL OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
