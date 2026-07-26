# SPDX-License-Identifier: Apache-2.0
"""HTCCL UCX transport: collective correctness + registry wiring (task #117).

CPU-only and single-host. The ranks are real processes talking over a real UCX
worker (loopback ``self``/``sm``/``tcp`` transports), so this exercises the
actual rendezvous, tag matching, chunking and teardown -- everything the
cross-rig RDMA run does except the wire underneath. The RDMA leg needs two
hosts and lives in ``scripts/nordstern/l1_ucx_crossrig.py``.

world=3 is not redundant with world=2: it is the smallest size at which the
ring all_reduce and the multi-peer exchange differ from a straight pairwise
swap.
"""

import multiprocessing as mp
import os
import tempfile
import unittest

import torch

from sglang.srt.distributed.device_communicators.htccl import (
    TRANSPORT_REGISTRY,
    _NO_FALLBACK,
    HTCCLCommunicator,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="base-a-test-cpu")


def _ucx_available():
    try:
        from sglang.srt.distributed.device_communicators.htccl_ucx_bindings import (
            UcpLibrary,
        )

        UcpLibrary.instance()
        return True
    except Exception:
        return False


class _Comm:
    """Stand-in for HTCCLCommunicator: a FRESH output tensor per call.

    Mirrors HTCCLCommunicator._get_out_buf exactly. The real communicator
    cannot be constructed here because its __init__ opens a CUDA stream.
    """

    def _get_out_buf(self, ref):
        return torch.empty_like(ref)


def _worker(rank, world, store, q):
    """One rank: run every collective against a locally computed reference."""
    import torch.distributed as dist

    fails = []
    try:
        dist.init_process_group(
            backend="gloo", init_method=f"file://{store}",
            rank=rank, world_size=world,
        )
        from sglang.srt.distributed.device_communicators.htccl_ucx import (
            HTCCLUcxTransport,
        )

        comm = _Comm()
        t = HTCCLUcxTransport(cpu_group=dist.group.WORLD, device=torch.device("cpu"))

        def chk(name, got, want, atol=0.0):
            d = (got - want).abs().max().item() if got.numel() else 0.0
            if d > atol:
                fails.append(f"{name}: max|delta|={d}")

        torch.manual_seed(4242)
        parts = [torch.randn(37, 53) for _ in range(world)]

        chk("all_reduce", t.htccl_all_reduce(comm, parts[rank].clone()),
            sum(parts), 1e-4)

        # The buffer-aliasing regression that once corrupted the forward
        # silently: two same-shape results must be independent tensors.
        a = t.htccl_all_reduce(comm, parts[rank].clone())
        snapshot = a.clone()
        b = t.htccl_all_reduce(comm, (parts[rank] * 5).clone())
        if a.data_ptr() == b.data_ptr():
            fails.append("all_reduce returned an aliased buffer")
        chk("all_reduce/first-result-intact", a, snapshot, 0.0)

        # Ring branch (world>2 only, by construction).
        t.ring_bytes = 1024
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

        # dim=2 is the orientation that a movedim(0, dim) mix-up gets wrong
        # while every shape assertion still passes.
        rs = [torch.randn(2 * world, 3, 4 * world) for _ in range(world)]
        total = sum(rs)
        for dim in (0, 2):
            moved = total.movedim(dim, 0).contiguous()
            c = moved.shape[0] // world
            want = moved[rank * c:(rank + 1) * c].movedim(0, dim).contiguous()
            chk(f"reduce_scatter/dim={dim}",
                t.htccl_reduce_scatter(comm, rs[rank].clone(), dim), want, 1e-4)

        # Force many chunks through one collective.
        t.chunk_bytes = 4096
        chk("all_reduce/chunked", t.htccl_all_reduce(comm, parts[rank].clone()),
            sum(parts), 1e-4)
        chk("all_gather/chunked",
            t.htccl_all_gather(comm, parts[rank].clone(), 0),
            torch.cat(parts, dim=0))
        t.chunk_bytes = 4 << 20

        # ---- pipelining (task #198) -------------------------------------
        #
        # The pipelined all_reduce writes the result chunk by chunk out of two
        # rotating slot sets. A reference built from `randn` would not notice
        # a chunk landing at the wrong offset, or two chunks swapped, as long
        # as the sums happened to be close -- so these use payloads whose
        # every element is DISTINCT and position-dependent. A misplaced chunk
        # then shows up as a large max|delta|, not as noise.
        t.chunk_bytes = 4096  # 1024 fp32 elements per chunk
        for label, count in (
            ("aligned-4chunks", 4096),      # exact multiple of the chunk
            ("ragged-tail", 4096 + 7),      # short final chunk
            ("one-past-boundary", 1025),    # 2 chunks, second holds 1 element
            ("exact-one-chunk", 1024),      # boundary case: no pipelining at all
            ("sub-chunk", 3),               # smaller than one chunk
            ("odd", 4999),
        ):
            ramp = [
                torch.arange(count, dtype=torch.float32) * (r + 1) + r * 1e5
                for r in range(world)
            ]
            chk(f"all_reduce/pipeline/{label}",
                t.htccl_all_reduce(comm, ramp[rank].clone()), sum(ramp), 1e-2)

        # Same, with a progress interleave so short that every sub-block
        # boundary inside a chunk is exercised too.
        t.progress_bytes = 512
        ramp = [
            torch.arange(5000, dtype=torch.float32) * (r + 3) + r * 7e4
            for r in range(world)
        ]
        chk("all_reduce/pipeline/fine-progress",
            t.htccl_all_reduce(comm, ramp[rank].clone()), sum(ramp), 1e-2)
        t.progress_bytes = 256 * 1024

        # Back-to-back multi-chunk collectives: the slot parities rotate
        # across calls, so a stale slot from the previous collective would
        # corrupt this one. Both results must stay independent and correct.
        first = t.htccl_all_reduce(comm, ramp[rank].clone())
        first_snapshot = first.clone()
        second = t.htccl_all_reduce(comm, (ramp[rank] * 2).clone())
        chk("all_reduce/pipeline/back-to-back-1", first, first_snapshot, 0.0)
        chk("all_reduce/pipeline/back-to-back-2", second,
            sum(r * 2 for r in ramp), 1e-2)
        if first.data_ptr() == second.data_ptr():
            fails.append("pipelined all_reduce returned an aliased buffer")

        # bf16 in, fp32 on the wire: the chunk length is counted in REDUCE
        # elements, so an input-element-size mix-up shows up here as a
        # misplaced tail.
        bramp = [
            (torch.arange(3000, dtype=torch.float32) * 0.01 + r).bfloat16()
            for r in range(world)
        ]
        chk("all_reduce/pipeline/bf16",
            t.htccl_all_reduce(comm, bramp[rank].clone()).float(),
            sum(b.float() for b in bramp), 3e-1)

        # The pipelined and unpipelined paths must agree exactly -- same
        # accumulation order, same result, byte for byte.
        payload = torch.arange(4096 + 5, dtype=torch.float32) * (rank + 1)
        piped = t.htccl_all_reduce(comm, payload.clone())
        was = t.pipeline  # restore, never assign True back
        t.pipeline = False
        plain = t.htccl_all_reduce(comm, payload.clone())
        t.pipeline = was
        chk("all_reduce/pipeline/matches-unpipelined", piped, plain, 0.0)

        # ---- all_gather pipelining (task #198, block 2) ------------------
        #
        # Pure data movement, so every reference here is EXACT (atol 0.0):
        # a chunk landing at the wrong offset, in the wrong rank's row, or
        # out of a stale slot is a hard failure, never tolerance noise.
        for label, count in (
            ("aligned-4chunks", 4096),
            ("ragged-tail", 4096 + 7),
            ("one-past-boundary", 1025),
            ("sub-chunk", 3),
            ("odd", 4999),
        ):
            gramp = [
                torch.arange(count, dtype=torch.float32) * (r + 1) + r * 1e5
                for r in range(world)
            ]
            chk(f"all_gather/pipeline/{label}",
                t.htccl_all_gather(comm, gramp[rank].clone(), 0),
                torch.cat(gramp, dim=0), 0.0)

        # Chunking is over the FLAT input, axis handling over the output; a
        # mix-up between the two scrambles rows across ranks.
        gramp2 = [
            (torch.arange(6000, dtype=torch.float32) * (r + 2)).reshape(6, 1000)
            for r in range(world)
        ]
        for dim in (0, 1, -1):
            chk(f"all_gather/pipeline/2d-dim={dim}",
                t.htccl_all_gather(comm, gramp2[rank].clone(), dim),
                torch.cat(gramp2, dim=dim), 0.0)

        # Pipelined vs unpipelined, and the single-chunk fast path vs the
        # generic path: all bit-identical, they are all pure copies.
        gramp = [torch.arange(9000, dtype=torch.float32) * (r + 3)
                 for r in range(world)]
        g_piped = t.htccl_all_gather(comm, gramp[rank].clone(), 0)
        was = t.pipeline  # restore, never assign True back
        t.pipeline = False
        g_plain = t.htccl_all_gather(comm, gramp[rank].clone(), 0)
        t.pipeline = was
        chk("all_gather/pipeline/matches-unpipelined", g_piped, g_plain, 0.0)

        small = [
            (torch.arange(500, dtype=torch.float32) * (r + 1)).reshape(20, 25)
            for r in range(world)
        ]
        for dim in (0, 1):
            g_fast = t.htccl_all_gather(comm, small[rank].clone(), dim)
            was = t.pipeline
            t.pipeline = False
            g_gen = t.htccl_all_gather(comm, small[rank].clone(), dim)
            t.pipeline = was
            chk(f"all_gather/fastpath/matches-generic-dim={dim}",
                g_fast, g_gen, 0.0)
            chk(f"all_gather/fastpath/reference-dim={dim}", g_fast,
                torch.cat(small, dim=dim), 0.0)
        t.chunk_bytes = 4 << 20

        for _ in range(3):
            t.barrier()

        t.close()
        t.close()  # idempotent
        dist.destroy_process_group()
    except Exception as e:  # pragma: no cover - reported through the queue
        import traceback

        fails.append(f"exception: {e}\n{traceback.format_exc()}")
    q.put((rank, fails))


@unittest.skipUnless(_ucx_available(), "libucp not loadable (apt install libucx0)")
class TestUcxCollectives(CustomTestCase):
    def _run_world(self, world):
        ctx = mp.get_context("spawn")
        env = dict(os.environ)
        env.setdefault("UCX_TLS", "self,sm,tcp")
        os.environ.update(env)
        with tempfile.TemporaryDirectory() as d:
            store = os.path.join(d, "store")
            q = ctx.Queue()
            procs = [
                ctx.Process(target=_worker, args=(r, world, store, q))
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
            for rank, fails in sorted(results):
                self.assertEqual(fails, [], f"rank {rank} of {world}: {fails}")
            for p in procs:
                self.assertEqual(p.exitcode, 0)

    def test_world_2(self):
        self._run_world(2)

    def test_world_3(self):
        """Smallest world where the ring and the multi-peer exchange differ."""
        self._run_world(3)


class TestUcxRegistryWiring(CustomTestCase):
    """No transport-specific condition may leak back to a call site."""

    def test_registered_under_ucx(self):
        self.assertIn("ucx", TRANSPORT_REGISTRY)

    def test_may_fall_back_to_gloo(self):
        """Unlike `device`, ucx is CPU-orchestrated, so gloo can stand in.

        `_NO_FALLBACK` exists because the device transport's presence is what
        licensed CUDA-graph capture; a silent CPU replacement would be captured
        and crash later. That reasoning does not apply here -- ucx synchronises
        with the host exactly like gloo does.
        """
        self.assertNotIn("ucx", _NO_FALLBACK)

    def test_handles_is_size_independent(self):
        """Routing must not depend on payload size.

        The communicator asks each rank separately. A size-dependent answer
        would let two ranks disagree about whether a collective goes over UCX
        or over gloo, and the group would hang with half of it waiting on a tag
        the other half never sends.
        """
        from sglang.srt.distributed.device_communicators.htccl_ucx import (
            HTCCLUcxTransport,
        )

        fake = HTCCLUcxTransport.__new__(HTCCLUcxTransport)
        for op in ("all_reduce", "all_gather", "broadcast", "reduce_scatter"):
            for nbytes in (0, 1, 1 << 10, 1 << 30):
                self.assertTrue(HTCCLUcxTransport.handles(fake, op, nbytes))
        for nbytes in (0, 1 << 30):
            self.assertFalse(HTCCLUcxTransport.handles(fake, "nonsense", nbytes))

    def test_dispatch_selects_ucx(self):
        """A registered transport is dispatchable without touching call sites."""
        from sglang.srt.distributed.device_communicators.htccl_ucx import (
            HTCCLUcxTransport,
        )

        fake = HTCCLUcxTransport.__new__(HTCCLUcxTransport)
        comm = HTCCLCommunicator.__new__(HTCCLCommunicator)
        comm.transport = fake
        self.assertIs(HTCCLCommunicator._select(comm, "all_reduce", 1 << 20), fake)
        self.assertIsNone(HTCCLCommunicator._select(comm, "nonsense", 8))


class TestUcxVersionParity(CustomTestCase):
    """The mismatch this transport exists to diagnose."""

    def _check(self, gathered):
        from sglang.srt.distributed.device_communicators.htccl_ucx import (
            HTCCLUcxTransport,
        )

        fake = HTCCLUcxTransport.__new__(HTCCLUcxTransport)
        return HTCCLUcxTransport._check_version_parity(fake, gathered)

    @staticmethod
    def _rank(rank, version, path="libucp.so.0"):
        return {
            "rank": rank,
            "version": version,
            "version_string": ".".join(str(v) for v in version),
            "lib_path": path,
            "address": b"",
        }

    def test_uniform_group_accepted(self):
        self._check([self._rank(0, (1, 16, 0)), self._rank(1, (1, 16, 0))])

    def test_mismatch_rejected_with_actionable_message(self):
        from sglang.srt.distributed.device_communicators.htccl_ucx_bindings import (
            UcxVersionMismatch,
        )

        with self.assertRaises(UcxVersionMismatch) as cm:
            self._check([
                self._rank(0, (1, 18, 1)),
                self._rank(1, (1, 16, 0), "/opt/ucx116/lib/libucp.so.0"),
            ])
        msg = str(cm.exception)
        # Naming both versions and the remedy is the whole point: the raw UCX
        # failure ('invalid bandwidth 0.00') identifies neither.
        self.assertIn("1.18.1", msg)
        self.assertIn("1.16.0", msg)
        self.assertIn("SGLANG_HTCCL_UCX_LIB", msg)
        self.assertIn("/opt/ucx116/lib/libucp.so.0", msg)

    def test_patch_level_mismatch_also_rejected(self):
        """UCX's wire format is not guaranteed stable across patch levels."""
        from sglang.srt.distributed.device_communicators.htccl_ucx_bindings import (
            UcxVersionMismatch,
        )

        with self.assertRaises(UcxVersionMismatch):
            self._check([self._rank(0, (1, 16, 0)), self._rank(1, (1, 16, 1))])


if __name__ == "__main__":
    unittest.main()
