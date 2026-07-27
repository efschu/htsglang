"""Cross-rig collective latency, one harness, three link configurations.

Task #204 / FEATURES_VS_UPSTREAM HTCCL transport line.

WHY THIS EXISTS
---------------
The 1 GbE figure previously in circulation (78 us) was a raw TCP round-trip
taken during network bring-up. It is NOT a collective latency and must not be
compared against the 5.5 / 26.6 us UCX numbers. This script measures the SAME
operations at the SAME sizes through each transport so the comparison is real.

Three configurations, which separate the software stack from the wire:

  gloo-1g    torch.distributed gloo over the 1 GbE LAN   (<RIG_LAN>.x)
  gloo-roce  torch.distributed gloo over the RoCE NIC's IP (<RDMA_NET>.x, TCP)
  ucx        HTCCL/UCX native RDMA over the same RoCE NIC

gloo-1g vs ucx is the deployment question ("how much does RDMA buy").
gloo-roce sits between them and shows how much of the gap is the wire and how
much is the transport stack.

CPU tensors only -- no GPU is touched, so this can run while a GPU window is
busy on either rig.

Usage: link_lat.py --mode {gloo-1g,gloo-roce,ucx} --rank R --world 2
                   [--comm-dir DIR] [--iters N]
"""

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
import types

SIZES = [8 * 1024, 64 * 1024, 512 * 1024, 4 * 1024 * 1024]  # bytes


def load_transport(comm_dir):
    """Import htccl_ucx{,_bindings} from a checkout without importing sglang."""
    for name in ("sglang", "sglang.srt", "sglang.srt.distributed",
                 "sglang.srt.distributed.device_communicators"):
        if name not in sys.modules or not hasattr(sys.modules[name], "__path__"):
            stub = types.ModuleType(name)
            stub.__path__ = []
            sys.modules[name] = stub

    def _load(mod_name, path):
        spec = importlib.util.spec_from_file_location(mod_name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod

    base = "sglang.srt.distributed.device_communicators."
    _load(base + "htccl_ucx_bindings",
          os.path.join(comm_dir, "htccl_ucx_bindings.py"))
    return _load(base + "htccl_ucx", os.path.join(comm_dir, "htccl_ucx.py"))


def stats(samples):
    """Median plus p5/p95 -- distribution, not just a mean."""
    s = sorted(samples)
    n = len(s)
    return {
        "median_us": round(statistics.median(s) * 1e6, 2),
        "p5_us": round(s[max(0, int(0.05 * n))] * 1e6, 2),
        "p95_us": round(s[min(n - 1, int(0.95 * n))] * 1e6, 2),
        "min_us": round(s[0] * 1e6, 2),
        "n": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["gloo-1g", "gloo-roce", "ucx"])
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--comm-dir", default="")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import torch

    res = {"mode": a.mode, "rank": a.rank, "world": a.world,
           "iters": a.iters, "cells": {}}

    if a.mode.startswith("gloo"):
        import torch.distributed as dist
        dist.init_process_group(backend="gloo", rank=a.rank,
                                world_size=a.world)

        def do_barrier():
            dist.barrier()

        def do_all_reduce(x):
            dist.all_reduce(x)

        def do_all_gather(x):
            out = [torch.empty_like(x) for _ in range(a.world)]
            dist.all_gather(out, x)

        fini = dist.destroy_process_group
    else:
        # UCX still rendezvouses over a gloo group; only the DATA plane is RDMA.
        import torch.distributed as dist
        dist.init_process_group(backend="gloo", rank=a.rank,
                                world_size=a.world)
        mod = load_transport(a.comm_dir)
        bindings = sys.modules[
            "sglang.srt.distributed.device_communicators.htccl_ucx_bindings"]
        lib = bindings.UcpLibrary.instance()
        print(f"[rank {a.rank}] UCX {lib.version_string()} from {lib.path}",
              file=sys.stderr, flush=True)
        t = mod.HTCCLUcxTransport(cpu_group=dist.group.WORLD,
                                  device=torch.device("cpu"))

        class _Comm:
            """Stand-in for HTCCLCommunicator: a fresh output tensor per call."""

            def _get_out_buf(self, ref):
                return torch.empty_like(ref)

        comm = _Comm()

        def do_barrier():
            t.barrier()

        def do_all_reduce(x):
            t.htccl_all_reduce(comm, x)

        def do_all_gather(x):
            t.htccl_all_gather(comm, x, 0)

        fini = dist.destroy_process_group

    # ---- barrier ---------------------------------------------------------
    for _ in range(a.warmup):
        do_barrier()
    samp = []
    for _ in range(a.iters):
        do_barrier()                      # keep ranks in lockstep
        t0 = time.perf_counter()
        do_barrier()
        samp.append(time.perf_counter() - t0)
    res["cells"]["barrier"] = stats(samp)

    # ---- all_reduce / all_gather at each size ----------------------------
    for nbytes in SIZES:
        n = nbytes // 4
        for op, fn in (("all_reduce", do_all_reduce),
                       ("all_gather", do_all_gather)):
            x = torch.ones(n, dtype=torch.float32)
            for _ in range(a.warmup):
                fn(x.clone())
            samp = []
            for _ in range(a.iters):
                y = x.clone()
                do_barrier()
                t0 = time.perf_counter()
                fn(y)
                samp.append(time.perf_counter() - t0)
            st = stats(samp)
            # effective wire rate for the payload actually moved
            st["gbit_s"] = round(nbytes * 8 / (st["median_us"] * 1e-6) / 1e9, 3)
            res["cells"][f"{op}/{nbytes // 1024}KiB"] = st

    fini()

    if a.rank == 0:
        print(json.dumps(res, indent=1))
        if a.out:
            with open(a.out, "w") as f:
                json.dump(res, f, indent=1)
        print(f"\n{'cell':24s} {'median_us':>10s} {'p5':>9s} {'p95':>9s} {'Gbit/s':>8s}")
        for k, v in res["cells"].items():
            print(f"{k:24s} {v['median_us']:10.2f} {v['p5_us']:9.2f} "
                  f"{v['p95_us']:9.2f} {v.get('gbit_s', ''):>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
