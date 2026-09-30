#!/usr/bin/env python3
"""DUAL-TP3PP3: does CUDA IPC weight sharing (P maps D's shard) work between two
processes, with and without MPS?  Parent owns a bf16 'shard', child maps it via
torch.multiprocessing (cudaIpc) and runs a GEMM on it; the checksum must match.
Prints one JSON line."""
import json, sys, time
import torch
import torch.multiprocessing as mp


def child(q, r):
    t = q.get()
    x = torch.ones(8, t.shape[0], dtype=t.dtype, device=t.device)
    y = (x @ t).float().sum().item()
    r.put({"child_sum": y, "child_ptr": t.data_ptr()})
    q.get()


def main():
    mp.set_start_method("spawn")
    torch.manual_seed(0)
    w = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
    ref = (torch.ones(8, 4096, dtype=w.dtype, device="cuda") @ w).float().sum().item()
    q, r = mp.Queue(), mp.Queue()
    p = mp.Process(target=child, args=(q, r))
    p.start()
    t0 = time.time()
    q.put(w)
    try:
        out = r.get(timeout=60)
        ok = abs(out["child_sum"] - ref) <= 1e-3 * max(1.0, abs(ref))
        res = {"ipc_ok": ok, "ref": ref, **out, "parent_ptr": w.data_ptr(), "s": time.time() - t0}
    except Exception as e:
        res = {"ipc_ok": False, "error": repr(e)}
    q.put(None)
    p.join(timeout=30)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
