"""Baseline collective cells for the comm suite: gloo, NCCL, barlink/shm.

Task #271.

WHY THIS EXISTS
---------------
``link_collective_cost.py`` measures the barlink/UCX transport and nothing else
-- it wraps that transport's own call sites to split a collective into
stage/post/wait/finish. A transport figure with no reference point is not
interpretable: "37 us at 20 KiB" only means something next to what the stock
backends cost on the same box, in the same process shape, at the same sizes.

This worker measures those reference cells and writes the SAME JSON shape
(``{"cells": {label: stats}}``, ``stats`` from ``link_lat.stats``), so the
suite parses one schema and the arms are comparable by construction rather
than by a conversion the reader has to trust.

Backends:

``gloo``
    ``torch.distributed`` over TCP. The CPU reference every rig has.
``nccl``
    ``torch.distributed`` on CUDA tensors, one rank per visible card. The
    intra-rig reference. Touches the GPU -- the suite runs it only inside a
    card window.
``barlink_shm``
    ``BarlinkShmTransport``: single-node shared memory, all_reduce only (the
    transport implements no all_gather, and this worker does not add one --
    a cell that exists here but not in the transport would be measuring this
    file).

One process per rank, launched by the caller. CPU backends touch no GPU, so
they run while a GPU window is busy.

Usage: comm_suite_worker.py --backend gloo --rank R --world W [--sizes KiB,...]
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from link_lat import stats  # noqa: E402

#: Same decode-shaped default ladder link_collective_cost.py sweeps, in KiB:
#: 20 KiB = one hidden row of Qwen3.6-27B in fp32 staging (bs=1 decode),
#: 80 KiB = a 4-token MTP verify, 256 KiB = a small prefill chunk.
DEFAULT_SIZES_KIB = (20, 80, 256)


def _load_shm_transport(comm_dir):
    """Import ``barlink_shm`` from a checkout without importing sglang.

    Same trick as ``link_lat.load_transport``, kept separate because the shm
    module has no bindings sibling to load first.
    """
    import importlib.util
    import types

    for name in ("sglang", "sglang.srt", "sglang.srt.distributed",
                 "sglang.srt.distributed.device_communicators"):
        if name not in sys.modules or not hasattr(sys.modules[name], "__path__"):
            stub = types.ModuleType(name)
            stub.__path__ = []
            sys.modules[name] = stub
    mod_name = "sglang.srt.distributed.device_communicators.barlink_shm"
    spec = importlib.util.spec_from_file_location(
        mod_name, os.path.join(comm_dir, "barlink_shm.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


class _Comm:
    """The scrap of communicator surface the barlink transports call back into.

    Returns a FRESH buffer every time on purpose: a shared scratch buffer is
    the recurring shape behind the returned-buffer bug family, and a
    measurement harness must not be the place that hides it.
    """

    def _get_out_buf(self, ref):
        import torch

        return torch.empty_like(ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True,
                    choices=("gloo", "nccl", "barlink_shm"))
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--sizes", default="",
                    help="comma-separated KiB list (default: 20,80,256)")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--comm-dir", default="",
                    help="device_communicators dir, barlink_shm only")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import torch
    import torch.distributed as dist

    sizes = [int(x) for x in a.sizes.split(",")] if a.sizes \
        else list(DEFAULT_SIZES_KIB)

    device = torch.device("cpu")
    if a.backend == "nccl":
        # One rank per visible card. The caller sets CUDA_VISIBLE_DEVICES or
        # relies on rank == device index; both land on a distinct card.
        torch.cuda.set_device(a.rank % max(torch.cuda.device_count(), 1))
        device = torch.device("cuda", a.rank % max(torch.cuda.device_count(), 1))

    dist.init_process_group(
        backend="nccl" if a.backend == "nccl" else "gloo",
        rank=a.rank, world_size=a.world)

    res = {
        "backend": a.backend,
        "rank": a.rank,
        "world": a.world,
        "iters": a.iters,
        "cells": {},
    }

    transport = None
    if a.backend == "barlink_shm":
        mod = _load_shm_transport(a.comm_dir)
        # The slot has to hold the largest payload this run moves, plus the
        # ragged element the correctness check appends. Sizing it from the
        # sweep rather than from a constant keeps the segment honest when the
        # caller passes a bigger ladder.
        slot_bytes = max(sizes) * 1024 + 4096
        transport = mod.BarlinkShmTransport(cpu_group=dist.group.WORLD,
                                          device=device,
                                          slot_bytes=slot_bytes)
        res["transport"] = "barlink_shm"
        res["slot_bytes"] = slot_bytes

    comm = _Comm()

    def _ops():
        """(label, callable) per collective this backend actually implements.

        BarlinkShmTransport has no all_gather, so the shm arm reports all_reduce
        only rather than a cell this file would have to synthesize.
        """
        if a.backend == "barlink_shm":
            return [("all_reduce", lambda y: transport.barlink_all_reduce(comm, y))]

        def ar(y):
            dist.all_reduce(y)
            return y

        def ag(y):
            out = [torch.empty_like(y) for _ in range(a.world)]
            dist.all_gather(out, y)
            return out

        return [("all_reduce", ar), ("all_gather", ag)]

    for op_name, fn in _ops():
        for kib in sizes:
            n = kib * 1024 // 4
            x = torch.ones(n, dtype=torch.float32, device=device)
            for _ in range(a.warmup):
                fn(x.clone())
            if a.backend == "nccl":
                torch.cuda.synchronize()
            samp = []
            for _ in range(a.iters):
                y = x.clone()
                dist.barrier()
                t0 = time.perf_counter()
                fn(y)
                if a.backend == "nccl":
                    torch.cuda.synchronize()
                samp.append(time.perf_counter() - t0)
            st = stats(samp)
            st["gbit_s"] = round(
                kib * 1024 * 8 / (st["median_us"] * 1e-6) / 1e9, 3)
            # Spread as a fraction of the median: the per-cell noise the
            # suite quotes next to every number (harness rule 5).
            st["spread_pct"] = round(
                100.0 * (st["p95_us"] - st["p5_us"]) / max(st["median_us"], 1e-9), 1)
            res["cells"][f"{op_name}/{kib}KiB"] = st
            if a.rank == 0:
                print(f"  {a.backend:10s} {op_name}/{kib}KiB "
                      f"{st['median_us']:9.2f} us  "
                      f"{st['gbit_s']:7.3f} Gbit/s  "
                      f"spread {st['spread_pct']:5.1f} %", flush=True)

    # ---- correctness line: the same exactness check the UCX gate runs, so
    # every backend carries a right/wrong bit next to its timings.
    bad = 0
    for kib in sizes:
        n = kib * 1024 // 4 + 1  # ragged on purpose: the padding path
        x = (torch.arange(n, dtype=torch.float32, device=device)
             * (a.rank + 1) + 1.0)
        want = (torch.arange(n, dtype=torch.float32, device=device)
                * (a.world * (a.world + 1) // 2) + float(a.world))
        if a.backend == "barlink_shm":
            got = transport.barlink_all_reduce(comm, x.clone())
        else:
            got = x.clone()
            dist.all_reduce(got)
        if not torch.equal(got, want):
            bad += 1
    res["exact_mismatches"] = bad

    if transport is not None:
        transport.close()
    dist.destroy_process_group()

    if a.rank == 0 and a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
