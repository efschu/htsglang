#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Nordstern L1: cross-rig correctness + throughput for the HTCCL UCX transport.

Runs one rank per host. CPU tensors only -- this validates the transport
(rendezvous, version parity, tag matching, chunking, every collective) over the
real RDMA link without touching a GPU, so it can run while a GPU window is busy
on either rig.

The modules under test are loaded straight from a checkout by path, with the
`sglang` package tree stubbed out. That is deliberate twice over: the second
rig has no sglang install, and it proves the transport has no hidden
dependency on the rest of the runtime.

Launch via l1_ucx_crossrig.sh, which sets the environment both sides need.
"""

import argparse
import importlib.util
import os
import statistics
import sys
import time
import types


def load_transport(comm_dir: str):
    """Import htccl_ucx{,_bindings} from `comm_dir` without importing sglang."""
    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.distributed",
        "sglang.srt.distributed.device_communicators",
    ):
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
    _load(base + "htccl_ucx_bindings", os.path.join(comm_dir, "htccl_ucx_bindings.py"))
    return _load(base + "htccl_ucx", os.path.join(comm_dir, "htccl_ucx.py"))


class _Comm:
    """Stand-in for HTCCLCommunicator: a FRESH output tensor per call."""

    def _get_out_buf(self, ref):
        import torch

        return torch.empty_like(ref)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--master-addr", required=True)
    ap.add_argument("--master-port", type=int, default=29677)
    ap.add_argument("--comm-dir", required=True,
                    help="path to .../distributed/device_communicators")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--expect-version-mismatch", action="store_true",
                    help="succeed only if the parity check REJECTS the group")
    args = ap.parse_args()

    import torch
    import torch.distributed as dist

    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=args.rank,
        world_size=args.world,
    )
    mod = load_transport(args.comm_dir)
    bindings = sys.modules[
        "sglang.srt.distributed.device_communicators.htccl_ucx_bindings"
    ]
    lib = bindings.UcpLibrary.instance()
    print(f"[rank {args.rank}] UCX {lib.version_string()} from {lib.path}", flush=True)

    if args.expect_version_mismatch:
        try:
            mod.HTCCLUcxTransport(cpu_group=dist.group.WORLD,
                                  device=torch.device("cpu"))
        except bindings.UcxVersionMismatch as e:
            print(f"[rank {args.rank}] REJECTED as designed:\n{e}", flush=True)
            return 0
        print(f"[rank {args.rank}] FAIL: mismatched group was accepted", flush=True)
        return 1

    t0 = time.monotonic()
    t = mod.HTCCLUcxTransport(cpu_group=dist.group.WORLD, device=torch.device("cpu"))
    print(f"[rank {args.rank}] rendezvous+wireup {time.monotonic()-t0:.3f}s", flush=True)

    comm = _Comm()
    fails = []
    W, R = args.world, args.rank

    def chk(name, got, want, atol=0.0):
        d = (got - want).abs().max().item() if got.numel() else 0.0
        if d > atol:
            fails.append(f"{name}: max|delta|={d}")
            print(f"[rank {R}] FAIL {name}: max|delta|={d}", flush=True)
        else:
            print(f"[rank {R}] ok   {name}", flush=True)

    torch.manual_seed(20260726)
    parts = [torch.randn(97, 131) for _ in range(W)]

    chk("all_reduce", t.htccl_all_reduce(comm, parts[R].clone()), sum(parts), 1e-4)

    a = t.htccl_all_reduce(comm, parts[R].clone())
    a_snapshot = a.clone()
    b = t.htccl_all_reduce(comm, (parts[R] * 3).clone())
    if a.data_ptr() == b.data_ptr():
        fails.append("all_reduce returned an aliased buffer")
    chk("all_reduce/no-aliasing", a, a_snapshot, 0.0)

    t.ring_bytes = 1024  # force the ring branch (no-op at world=2 by design)
    big = [torch.randn(8, 512, 64) for _ in range(W)]
    chk("all_reduce/large", t.htccl_all_reduce(comm, big[R].clone()), sum(big), 1e-3)
    t.ring_bytes = 1 << 30

    bf = [p.bfloat16() for p in parts]
    chk("all_reduce/bf16", t.htccl_all_reduce(comm, bf[R].clone()).float(),
        sum(p.float() for p in bf), 3e-1)

    for dim in (0, 1, -1):
        chk(f"all_gather/dim={dim}",
            t.htccl_all_gather(comm, parts[R].clone(), dim),
            torch.cat(parts, dim=dim))

    three = [torch.randn(2, 3, 5) for _ in range(W)]
    chk("all_gather/3d dim=2", t.htccl_all_gather(comm, three[R].clone(), 2),
        torch.cat(three, dim=2))

    for src in range(W):
        payload = torch.arange(257, dtype=torch.float32) + src * 1000
        tensor = payload.clone() if R == src else torch.zeros(257)
        chk(f"broadcast/src={src}", t.htccl_broadcast(comm, tensor, src), payload)

    rs = [torch.randn(2 * W, 3, 4 * W) for _ in range(W)]
    total = sum(rs)
    for dim in (0, 2):
        moved = total.movedim(dim, 0).contiguous()
        c = moved.shape[0] // W
        want = moved[R * c:(R + 1) * c].movedim(0, dim).contiguous()
        chk(f"reduce_scatter/dim={dim}",
            t.htccl_reduce_scatter(comm, rs[R].clone(), dim), want, 1e-4)

    t.chunk_bytes = 8192  # force multi-chunk transfers
    chk("all_reduce/chunked", t.htccl_all_reduce(comm, parts[R].clone()),
        sum(parts), 1e-4)
    t.chunk_bytes = 4 << 20

    for _ in range(5):
        t.barrier()
    print(f"[rank {R}] ok   barrier x5", flush=True)

    if args.bench:
        print(f"[rank {R}] --- throughput (all_reduce, world={W}) ---", flush=True)
        for mib in (0.0078125, 0.0625, 0.5, 4, 32):
            n = max(int(mib * 1024 * 1024 / 4), 2)
            x = torch.randn(n)
            for _ in range(3):
                t.htccl_all_reduce(comm, x)
            iters = 50 if mib < 1 else 10
            samples = []
            for _ in range(iters):
                t0 = time.perf_counter()
                t.htccl_all_reduce(comm, x)
                samples.append(time.perf_counter() - t0)
            med = statistics.median(samples)
            nbytes = n * 4
            gbps = nbytes * 8 / med / 1e9
            print(f"[rank {R}] all_reduce {nbytes/1024:9.1f} KiB  "
                  f"median {med*1e6:9.1f} us  {gbps:6.2f} Gbit/s", flush=True)

        bar = []
        for _ in range(200):
            t0 = time.perf_counter()
            t.barrier()
            bar.append(time.perf_counter() - t0)
        print(f"[rank {R}] barrier median {statistics.median(bar)*1e6:.1f} us  "
              f"min {min(bar)*1e6:.1f} us", flush=True)

    t.close()
    dist.destroy_process_group()
    print(f"[rank {R}] RESULT: {'FAIL' if fails else 'ALL PASS'}", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
