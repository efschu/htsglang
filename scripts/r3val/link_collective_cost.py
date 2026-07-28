"""Per-collective fixed cost of the HTCCL/UCX transport at decode payload sizes.

Task #244.

WHY THIS EXISTS
---------------
The transport figures in circulation (26.6 us all_reduce, 27.0 us all_gather)
were taken at 8 KiB and world 2. A cross-rig TP=4 decode does neither:

  * the payload is one hidden row per token in fp32 staging -- 20 KiB at bs=1,
    80 KiB for a 4-token verify (5120 x 4 B x {1,4});
  * world is 4, so the flat exchange puts (W-1) payloads in each direction
    across the ONE RoCE link per collective, not one;
  * 8 KiB is below UCX's automatic rendezvous threshold and 20/80 KiB is
    above it, so the production path runs a different UCX protocol than the
    number that was used to declare the transport "3 % of the socket".

This harness measures the same operations at the sizes and the world size
that actually run, splits every collective into stage / post / wait / finish,
and separates fixed cost per collective from cost per byte by moving the SAME
total bytes as one big collective and as many small ones.

CPU tensors only -- no GPU is touched, so it runs while a GPU window is busy.

Usage: link_collective_cost.py --rank R --world W --comm-dir DIR [--iters N]
"""

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from link_lat import load_transport, stats  # noqa: E402

# Decode-shaped payloads for Qwen3.6-27B (hidden 5120) with the default
# fp32 reduce staging: bs=1 decode, and a 4-token MTP verify.
DECODE_BYTES = 5120 * 4
VERIFY_BYTES = 5120 * 4 * 4


class Phases:
    """Stage / post / wait / finish split, taken from inside the transport.

    Wraps the two ctypes entry points the transport calls in a fixed order
    (first post_recv, then wait), so the timestamps come from the real call
    sites without editing the transport.
    """

    def __init__(self, transport):
        self.t = transport
        self.w = transport.worker
        self._orig_recv = self.w.post_recv
        self._orig_wait = self.w.wait
        self.reset()
        self.w.post_recv = self._recv
        self.w.wait = self._wait

    def reset(self):
        self.t0 = 0.0
        self.first_post = None
        self.wait_in = None
        self.wait_out = None

    def _recv(self, *a, **kw):
        if self.first_post is None:
            self.first_post = time.perf_counter()
        return self._orig_recv(*a, **kw)

    def _wait(self, *a, **kw):
        if self.wait_in is None:
            self.wait_in = time.perf_counter()
        r = self._orig_wait(*a, **kw)
        self.wait_out = time.perf_counter()
        return r

    def restore(self):
        self.w.post_recv = self._orig_recv
        self.w.wait = self._orig_wait

    def split(self, t0, t1):
        """(stage, post, wait, finish) in seconds for one collective."""
        fp = self.first_post if self.first_post is not None else t1
        wi = self.wait_in if self.wait_in is not None else fp
        wo = self.wait_out if self.wait_out is not None else wi
        return (fp - t0, wi - fp, wo - wi, t1 - wo)


def median_split(splits):
    return {
        k: round(statistics.median(v) * 1e6, 2)
        for k, v in zip(
            ("stage_us", "post_us", "wait_us", "finish_us"), zip(*splits)
        )
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--comm-dir", required=True)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--sizes", default="",
                    help="comma-separated KiB list; replaces the default "
                         "decode-shaped cells and skips the burst sweep")
    ap.add_argument("--op", default="all_reduce",
                    choices=("all_reduce", "all_gather"),
                    help="collective for the --sizes sweep")
    ap.add_argument("--gate", action="store_true",
                    help="exactness gate across the ring threshold instead "
                         "of a timing run: every collective is checked "
                         "against the computed reference at atol 0")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import torch
    import torch.distributed as dist

    dist.init_process_group(backend="gloo", rank=a.rank, world_size=a.world)
    mod = load_transport(a.comm_dir)
    bindings = sys.modules[
        "sglang.srt.distributed.device_communicators.htccl_ucx_bindings"]
    lib = bindings.UcpLibrary.instance()
    print(f"[rank {a.rank}] UCX {lib.version_string()} from {lib.path}",
          file=sys.stderr, flush=True)
    t = mod.HTCCLUcxTransport(cpu_group=dist.group.WORLD,
                              device=torch.device("cpu"))

    class _Comm:
        def _get_out_buf(self, ref):
            return torch.empty_like(ref)

    comm = _Comm()
    ph = Phases(t)
    res = {
        "rank": a.rank, "world": a.world, "iters": a.iters,
        "rndv_thresh": os.environ.get("UCX_RNDV_THRESH", "auto(unset)"),
        "fp32_reduce": os.environ.get("SGLANG_HTCCL_FP32_REDUCE", "1(default)"),
        "cells": {},
    }

    def measure(label, nbytes, op):
        n = nbytes // 4
        x = torch.ones(n, dtype=torch.float32)
        fn = (lambda y: t.htccl_all_reduce(comm, y)) if op == "all_reduce" \
            else (lambda y: t.htccl_all_gather(comm, y, 0))
        for _ in range(a.warmup):
            fn(x.clone())
        samp, splits = [], []
        for _ in range(a.iters):
            y = x.clone()
            t.barrier()
            ph.reset()
            t0 = time.perf_counter()
            fn(y)
            t1 = time.perf_counter()
            samp.append(t1 - t0)
            splits.append(ph.split(t0, t1))
        st = stats(samp)
        st["gbit_s"] = round(nbytes * 8 / (st["median_us"] * 1e-6) / 1e9, 3)
        st.update(median_split(splits))
        res["cells"][label] = st
        if a.rank == 0:
            print(f"  {label:34s} {st['median_us']:9.2f} us  "
                  f"stage {st['stage_us']:7.2f} post {st['post_us']:7.2f} "
                  f"wait {st['wait_us']:8.2f} finish {st['finish_us']:6.2f}",
                  flush=True)

    if a.gate:
        # Sizes straddling the ring threshold, including the ragged case the
        # ring pads (numel not a multiple of world) and the exact boundary.
        bad = 0
        for kib in (8, 16, 23, 24, 25, 32, 80, 128):
            for extra in (0, 1, 3):
                n = kib * 1024 // 4 + extra
                x = torch.arange(n, dtype=torch.float32) * (a.rank + 1) + 1.0
                got = t.htccl_all_reduce(comm, x.clone())
                want = torch.arange(n, dtype=torch.float32) * (
                    a.world * (a.world + 1) // 2) + float(a.world)
                if not torch.equal(got, want):
                    bad += 1
                    print(f"  MISMATCH all_reduce n={n} ({kib}KiB+{extra})",
                          flush=True)
                g = t.htccl_all_gather(comm, x.clone(), 0)
                wg = torch.cat([
                    torch.arange(n, dtype=torch.float32) * (r + 1) + 1.0
                    for r in range(a.world)])
                if not torch.equal(g, wg):
                    bad += 1
                    print(f"  MISMATCH all_gather n={n}", flush=True)
        # Read the thresholds off the transport, not off the environment: the
        # defaults live in the module, so an env-derived line would report
        # "off" for a gate run that did exercise the ring.
        ag_kib = t.ag_ring_bytes // 1024
        print(f"[rank {a.rank}] gate: {bad} mismatches "
              f"(all_reduce ring {t.ring_bytes // 1024} KiB, all_gather ring "
              f"{str(ag_kib) + ' KiB' if ag_kib else 'off'}, "
              f"world {a.world})", flush=True)
        t.close()
        dist.destroy_process_group()
        return 1 if bad else 0

    if a.rank == 0:
        print(f"== decode-shaped, world {a.world}, "
              f"RNDV_THRESH={res['rndv_thresh']} ==", flush=True)
    if a.sizes:
        for kib in [int(x) for x in a.sizes.split(",")]:
            measure(f"{a.op}/{kib}KiB", kib * 1024, a.op)
        t.close()
        dist.destroy_process_group()
        if a.rank == 0 and a.out:
            with open(a.out, "w") as f:
                json.dump(res, f, indent=1)
        return 0
    for op in ("all_reduce", "all_gather"):
        for nb, tag in ((8192, "8KiB"), (DECODE_BYTES, "20KiB-decode"),
                        (VERIFY_BYTES, "80KiB-verify")):
            measure(f"{op}/{tag}", nb, op)

    # ---- fixed cost per collective: same bytes, different collective count
    if a.rank == 0:
        print("== fixed cost: 640 KiB moved as 1 / 8 / 64 collectives ==",
              flush=True)
    TOTAL = 64 * 10240
    for count in (1, 8, 64):
        nb = TOTAL // count
        n = nb // 4
        x = torch.ones(n, dtype=torch.float32)
        for _ in range(a.warmup):
            t.htccl_all_reduce(comm, x.clone())
        samp = []
        for _ in range(max(a.iters // 4, 10)):
            ys = [x.clone() for _ in range(count)]
            t.barrier()
            t0 = time.perf_counter()
            for y in ys:
                t.htccl_all_reduce(comm, y)
            samp.append(time.perf_counter() - t0)
        st = stats(samp)
        st["per_collective_us"] = round(st["median_us"] / count, 2)
        res["cells"][f"burst/{count}x{nb // 1024}KiB"] = st
        if a.rank == 0:
            print(f"  {count:3d} x {nb // 1024:5d} KiB  total "
                  f"{st['median_us']:9.2f} us  "
                  f"per collective {st['per_collective_us']:8.2f} us",
                  flush=True)

    t.close()
    dist.destroy_process_group()
    if a.rank == 0 and a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
