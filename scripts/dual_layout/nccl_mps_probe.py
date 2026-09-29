#!/usr/bin/env python3
"""DUAL-TP3PP3: NCCL all_reduce between two processes on two cards when both
processes are MPS clients (the P and D groups keep their own communicators;
each group has one rank per card).  One JSON line from rank 0."""
import json, os, sys, time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run(rank, world, initf, q):
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", init_method=f"file://{initf}", rank=rank, world_size=world)
        x = torch.full((1 << 20,), float(rank + 1), device="cuda")
        for _ in range(3):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(20):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        ok = bool(torch.all(x == x[0]).item())
        if rank == 0:
            q.put({"nccl_ok": ok, "ms_per_ar_4MiB": (time.time() - t0) / 20 * 1e3,
                   "nccl": ".".join(map(str, torch.cuda.nccl.version()))})
        dist.destroy_process_group()
    except Exception as e:  # noqa: BLE001
        if rank == 0:
            q.put({"nccl_ok": False, "error": repr(e)[:300]})


if __name__ == "__main__":
    initf = f"/tmp/nccl-mps-probe-{os.getpid()}"
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=run, args=(r, 2, initf, q)) for r in range(2)]
    [p.start() for p in ps]
    try:
        res = q.get(timeout=120)
    except Exception as e:  # noqa: BLE001
        res = {"nccl_ok": False, "error": "timeout " + repr(e)}
    [p.join(timeout=30) for p in ps]
    [p.kill() for p in ps if p.is_alive()]
    res["mps"] = bool(os.environ.get("CUDA_MPS_PIPE_DIRECTORY"))
    try:
        os.unlink(initf)
    except OSError:
        pass
    print(json.dumps(res))
