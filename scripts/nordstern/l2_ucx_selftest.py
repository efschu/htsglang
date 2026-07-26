#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Nordstern L2: sglang-free self-test for the HTCCL UCX transport.

Same assertions as ``test/registered/unit/distributed/test_htccl_ucx_collectives.py``
-- real multi-process ranks over a real UCX worker on loopback (self/sm/tcp) --
but loaded the way ``l1_ucx_crossrig.py`` loads it: the two transport modules
by path, with the ``sglang`` package tree stubbed out.

That is not a convenience. The registered unit test needs the whole sglang
runtime importable, which neither rig in this fleet has; this file needs
nothing but torch and a loadable ``libucp``. It is therefore the version that
can actually be run where the hardware is, and it is the one that guards the
pipelined all_reduce on the machines that will carry it.

Usage:
    python3 scripts/nordstern/l2_ucx_selftest.py            # world 2 and 3
    UCX_TLS=self,sm,tcp python3 ... --world 3
"""

import argparse
import importlib.util
import multiprocessing as mp
import os
import sys
import tempfile
import types

REPO_COMM = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "python", "sglang", "srt", "distributed", "device_communicators",
)


def load_transport(comm_dir):
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


def _worker(rank, world, store, comm_dir, q):
    import torch
    import torch.distributed as dist

    fails = []
    try:
        dist.init_process_group(
            backend="gloo", init_method=f"file://{store}",
            rank=rank, world_size=world,
        )
        mod = load_transport(comm_dir)
        comm = _Comm()
        t = mod.HTCCLUcxTransport(
            cpu_group=dist.group.WORLD, device=torch.device("cpu")
        )

        def chk(name, got, want, atol=0.0):
            d = (got - want).abs().max().item() if got.numel() else 0.0
            if d > atol:
                fails.append(f"{name}: max|delta|={d} (atol {atol})")

        torch.manual_seed(4242)
        parts = [torch.randn(37, 53) for _ in range(world)]

        chk("all_reduce", t.htccl_all_reduce(comm, parts[rank].clone()),
            sum(parts), 1e-4)

        a = t.htccl_all_reduce(comm, parts[rank].clone())
        snapshot = a.clone()
        b = t.htccl_all_reduce(comm, (parts[rank] * 5).clone())
        if a.data_ptr() == b.data_ptr():
            fails.append("all_reduce returned an aliased buffer")
        chk("all_reduce/first-result-intact", a, snapshot, 0.0)

        t.ring_bytes = 1024  # ring branch (world>2 only, by construction)
        big = [torch.randn(3, 128, 33) for _ in range(world)]
        chk("all_reduce/ring", t.htccl_all_reduce(comm, big[rank].clone()),
            sum(big), 1e-3)
        t.ring_bytes = 1 << 30

        bf = [p.bfloat16() for p in parts]
        chk("all_reduce/bf16", t.htccl_all_reduce(comm, bf[rank].clone()).float(),
            sum(p.float() for p in bf), 3e-1)

        for dim in (0, 1, -1):
            chk(f"all_gather/dim={dim}",
                t.htccl_all_gather(comm, parts[rank].clone(), dim),
                torch.cat(parts, dim=dim))

        three = [torch.randn(2, 3, 5) for _ in range(world)]
        chk("all_gather/3d-dim2", t.htccl_all_gather(comm, three[rank].clone(), 2),
            torch.cat(three, dim=2))

        for src in range(world):
            payload = torch.arange(129, dtype=torch.float32) + src * 977
            tensor = payload.clone() if rank == src else torch.zeros(129)
            ret = t.htccl_broadcast(comm, tensor, src)
            chk(f"broadcast/src={src}", ret, payload)
            if ret.data_ptr() != tensor.data_ptr():
                fails.append(f"broadcast/src={src} was not in-place")

        rs = [torch.randn(2 * world, 3, 4 * world) for _ in range(world)]
        total = sum(rs)
        for dim in (0, 2):
            moved = total.movedim(dim, 0).contiguous()
            c = moved.shape[0] // world
            want = moved[rank * c:(rank + 1) * c].movedim(0, dim).contiguous()
            chk(f"reduce_scatter/dim={dim}",
                t.htccl_reduce_scatter(comm, rs[rank].clone(), dim), want, 1e-4)

        t.chunk_bytes = 4096
        chk("all_reduce/chunked", t.htccl_all_reduce(comm, parts[rank].clone()),
            sum(parts), 1e-4)
        chk("all_gather/chunked",
            t.htccl_all_gather(comm, parts[rank].clone(), 0),
            torch.cat(parts, dim=0))
        t.chunk_bytes = 4 << 20

        # ---- pipelining (task #198) -------------------------------------
        #
        # The pipelined all_reduce writes its result chunk by chunk out of two
        # rotating slot sets. A `randn` reference would not notice a chunk
        # landing at the wrong offset or two chunks swapped, as long as the
        # sums stayed close -- so these payloads are ramps whose every element
        # is distinct and position-dependent. A misplaced chunk then shows up
        # as a large max|delta|, not as noise.
        t.chunk_bytes = 4096  # 1024 fp32 elements per chunk
        for label, count in (
            ("aligned-4chunks", 4096),      # exact multiple of the chunk
            ("ragged-tail", 4096 + 7),      # short final chunk
            ("one-past-boundary", 1025),    # 2 chunks, second holds 1 element
            ("exact-one-chunk", 1024),      # boundary: no pipelining at all
            ("sub-chunk", 3),               # smaller than one chunk
            ("odd", 4999),
        ):
            ramp = [
                torch.arange(count, dtype=torch.float32) * (r + 1) + r * 1e5
                for r in range(world)
            ]
            chk(f"all_reduce/pipeline/{label}",
                t.htccl_all_reduce(comm, ramp[rank].clone()), sum(ramp), 1e-2)

        t.progress_bytes = 512  # exercise every sub-block boundary
        ramp = [
            torch.arange(5000, dtype=torch.float32) * (r + 3) + r * 7e4
            for r in range(world)
        ]
        chk("all_reduce/pipeline/fine-progress",
            t.htccl_all_reduce(comm, ramp[rank].clone()), sum(ramp), 1e-2)
        t.progress_bytes = 256 * 1024

        # ---- all_gather pipelining (task #198, block 2) ------------------
        #
        # all_gather is pure data movement -- no arithmetic -- so the
        # reference comparison is EXACT (atol 0.0): a chunk landing at the
        # wrong offset, in the wrong rank's row, or out of a stale slot is a
        # hard failure, never tolerance noise. Ramps for the same reason as
        # above: every element is distinct and position-dependent.
        for label, count in (
            ("aligned-4chunks", 4096),
            ("ragged-tail", 4096 + 7),
            ("one-past-boundary", 1025),
            ("exact-one-chunk", 1024),
            ("sub-chunk", 3),
            ("odd", 4999),
        ):
            ramp = [
                torch.arange(count, dtype=torch.float32) * (r + 1) + r * 1e5
                for r in range(world)
            ]
            chk(f"all_gather/pipeline/{label}",
                t.htccl_all_gather(comm, ramp[rank].clone(), 0),
                torch.cat(ramp, dim=0), 0.0)

        # Non-zero gather dim with multi-chunk payloads: the chunking is over
        # the FLAT input, the axis handling over the output -- a mix-up
        # between the two scrambles rows across ranks.
        ramp2 = [
            (torch.arange(6000, dtype=torch.float32) * (r + 2)).reshape(6, 1000)
            for r in range(world)
        ]
        for dim in (0, 1, -1):
            chk(f"all_gather/pipeline/2d-dim={dim}",
                t.htccl_all_gather(comm, ramp2[rank].clone(), dim),
                torch.cat(ramp2, dim=dim), 0.0)

        # bf16: chunk length is counted in INPUT elements (2 bytes) -- an
        # element-size mix-up shows up as a misplaced tail.
        bramp2 = [
            (torch.arange(3000, dtype=torch.float32) * 0.01 + r).bfloat16()
            for r in range(world)
        ]
        chk("all_gather/pipeline/bf16",
            t.htccl_all_gather(comm, bramp2[rank].clone(), 0),
            torch.cat(bramp2, dim=0), 0.0)

        # Back-to-back: slot parities rotate across calls; a stale slot from
        # the first call corrupting the second, or an aliased output, fails.
        g1 = t.htccl_all_gather(comm, ramp[rank].clone(), 0)
        g1_snapshot = g1.clone()
        g2 = t.htccl_all_gather(comm, (ramp[rank] * 2).clone(), 0)
        chk("all_gather/pipeline/back-to-back-1", g1, g1_snapshot, 0.0)
        chk("all_gather/pipeline/back-to-back-2", g2,
            torch.cat([r * 2 for r in ramp], dim=0), 0.0)
        if g1.data_ptr() == g2.data_ptr():
            fails.append("pipelined all_gather returned an aliased buffer")

        # Fine progress granularity across every sub-block boundary.
        t.progress_bytes = 512
        chk("all_gather/pipeline/fine-progress",
            t.htccl_all_gather(comm, ramp[rank].clone(), 0),
            torch.cat(ramp, dim=0), 0.0)
        t.progress_bytes = 256 * 1024

        # Pipelined and unpipelined must agree bit for bit (both are pure
        # copies, so anything else is a routing bug).
        gp = t.htccl_all_gather(comm, ramp[rank].clone(), 0)
        was_ag = t.pipeline  # restore, never assign True back
        t.pipeline = False
        gu = t.htccl_all_gather(comm, ramp[rank].clone(), 0)
        t.pipeline = was_ag
        chk("all_gather/pipeline/matches-unpipelined", gp, gu, 0.0)

        # Single-chunk FAST path vs the generic path: same exactness bar,
        # including a non-zero dim so the axis handling is compared too.
        small = [
            (torch.arange(500, dtype=torch.float32) * (r + 1)).reshape(20, 25)
            for r in range(world)
        ]
        for dim in (0, 1):
            gf = t.htccl_all_gather(comm, small[rank].clone(), dim)
            was_ag = t.pipeline
            t.pipeline = False
            gg = t.htccl_all_gather(comm, small[rank].clone(), dim)
            t.pipeline = was_ag
            chk(f"all_gather/fastpath/matches-generic-dim={dim}", gf, gg, 0.0)
            chk(f"all_gather/fastpath/reference-dim={dim}", gf,
                torch.cat(small, dim=dim), 0.0)

        # Back-to-back multi-chunk collectives: the slot parities rotate
        # across calls, so a stale slot would corrupt this one.
        first = t.htccl_all_reduce(comm, ramp[rank].clone())
        first_snapshot = first.clone()
        second = t.htccl_all_reduce(comm, (ramp[rank] * 2).clone())
        chk("all_reduce/pipeline/back-to-back-1", first, first_snapshot, 0.0)
        chk("all_reduce/pipeline/back-to-back-2", second,
            sum(r * 2 for r in ramp), 1e-2)
        if first.data_ptr() == second.data_ptr():
            fails.append("pipelined all_reduce returned an aliased buffer")

        # bf16 in, fp32 on the wire: the chunk length is counted in REDUCE
        # elements, so an input-element-size mix-up shows up as a misplaced
        # tail.
        bramp = [
            (torch.arange(3000, dtype=torch.float32) * 0.01 + r).bfloat16()
            for r in range(world)
        ]
        chk("all_reduce/pipeline/bf16",
            t.htccl_all_reduce(comm, bramp[rank].clone()).float(),
            sum(b.float() for b in bramp), 3e-1)

        # Pipelined and unpipelined must agree bit for bit: same accumulation
        # order, same result.
        payload = torch.arange(4096 + 5, dtype=torch.float32) * (rank + 1)
        piped = t.htccl_all_reduce(comm, payload.clone())
        was = t.pipeline  # restore, never assign True back
        t.pipeline = False
        plain = t.htccl_all_reduce(comm, payload.clone())
        t.pipeline = was
        chk("all_reduce/pipeline/matches-unpipelined", piped, plain, 0.0)
        t.chunk_bytes = 4 << 20

        for _ in range(3):
            t.barrier()

        t.close()
        t.close()  # idempotent
        dist.destroy_process_group()
    except Exception as e:
        import traceback

        fails.append(f"exception: {e}\n{traceback.format_exc()}")
    q.put((rank, fails))


def run_world(world, comm_dir):
    ctx = mp.get_context("spawn")
    os.environ.setdefault("UCX_TLS", "self,sm,tcp")
    with tempfile.TemporaryDirectory() as d:
        store = os.path.join(d, "store")
        q = ctx.Queue()
        procs = [
            ctx.Process(target=_worker, args=(r, world, store, comm_dir, q))
            for r in range(world)
        ]
        for p in procs:
            p.start()
        try:
            results = [q.get(timeout=300) for _ in range(world)]
        finally:
            for p in procs:
                p.join(timeout=60)
                if p.is_alive():
                    p.terminate()
    bad = False
    for rank, fails in sorted(results):
        if fails:
            bad = True
            for f in fails:
                print(f"  world={world} rank={rank} FAIL {f}")
        else:
            print(f"  world={world} rank={rank} ok")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, action="append",
                    help="repeatable; default 2 and 3")
    ap.add_argument("--comm-dir", default=os.path.normpath(REPO_COMM))
    a = ap.parse_args()
    worlds = a.world or [2, 3]
    bad = False
    for w in worlds:
        print(f"== world {w} ==")
        bad |= run_world(w, a.comm_dir)
    print("RESULT:", "FAIL" if bad else "ALL PASS")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
