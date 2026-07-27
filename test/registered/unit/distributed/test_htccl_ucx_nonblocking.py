# SPDX-License-Identifier: Apache-2.0
"""HTCCL UCX transport: non_blocking copy-out discipline (task #246).

The ucx transport pays one irreducible host sync per collective (the D2H
stage-in: UCX reads the pinned buffer host-side, outside stream order). Every
OTHER device-boundary copy -- the copy-outs of a result into device memory --
is consumed only by the compute stream and must be non_blocking, guarded by
per-slot events so a slot is never rewritten while the device may still be
reading it. Blocking those copies cost 416 stream drains per cross-rig verify
forward (analysis #244); after #246 it is one per collective.

These tests drive the single-chunk (decode-shaped) paths with a fake UCX
worker and fake CUDA events, so the BEHAVIOR -- which copy blocks, which does
not, and the sync-before-rewrite ordering -- is asserted directly, without a
GPU. Numeric correctness of the same paths runs against real UCX loopback in
``test_htccl_ucx_collectives.py``.
"""

import threading
import unittest

import torch

from sglang.srt.distributed.device_communicators.htccl_ucx import (
    HTCCLUcxTransport,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class _Comm:
    """Stand-in for HTCCLCommunicator: a FRESH output tensor per call."""

    def _get_out_buf(self, ref):
        return torch.empty_like(ref)


class _FakeEvent:
    """Records record/synchronize calls into a shared ordered log."""

    def __init__(self, log):
        self.log = log
        self.recorded = 0
        self.synced = 0

    def record(self, *args):
        self.recorded += 1
        self.log.append(("record", id(self)))

    def synchronize(self):
        self.synced += 1
        self.log.append(("sync", id(self)))

    def query(self):
        return True


class _FakeWorker:
    """UcpWorker stand-in: accepts posts, completes them instantly."""

    def __init__(self, log):
        self.log = log

    def post_recv(self, ptr, nbytes, tag):
        self.log.append(("post_recv", ptr))
        return object()

    def post_send(self, peer, ptr, nbytes, tag):
        self.log.append(("post_send", ptr))
        return object()

    def wait(self, reqs):
        self.log.append(("wait",))

    def progress(self):
        pass


class _CopySpy:
    """Context manager recording every Tensor.copy_ with its non_blocking flag."""

    def __init__(self):
        self.calls = []

    def __enter__(self):
        self._orig = torch.Tensor.copy_
        spy = self

        def wrapper(tensor, src, non_blocking=False):
            spy.calls.append(
                (
                    tensor.data_ptr(),
                    src.data_ptr() if isinstance(src, torch.Tensor) else None,
                    tensor.dtype,
                    src.dtype if isinstance(src, torch.Tensor) else None,
                    bool(non_blocking),
                )
            )
            return spy._orig(tensor, src, non_blocking)

        torch.Tensor.copy_ = wrapper
        return self

    def __exit__(self, *exc):
        torch.Tensor.copy_ = self._orig
        return False

    def to_dst(self, ptr):
        return [c for c in self.calls if c[0] == ptr]

    def from_src(self, ptr):
        return [c for c in self.calls if c[1] == ptr]


def _make_transport(log, world=2, rank=0):
    """A transport over the fake worker; mirrors the __init__ bookkeeping."""
    t = HTCCLUcxTransport.__new__(HTCCLUcxTransport)
    t.cpu_group = None
    t.device = torch.device("cpu")
    t.world_size = world
    t.rank = rank
    t.chunk_bytes = 4 << 20
    t.ring_bytes = 1 << 30
    t.progress_bytes = 0
    t.pipeline = True
    t._lock = threading.Lock()
    t._seq = 0
    t._staging_bufs = {}
    t._view_cache = {}
    t._use_cuda = False
    t._h2d_events = {}
    t._async_free = {}
    t._closed = False
    t._peers = tuple(p for p in range(world) if p != rank)
    t._ar_keys = tuple(
        (f"ar_s{par}", tuple(f"ar_r{par}_{p}" for p in t._peers))
        for par in (0, 1)
    )
    t._ag_keys = tuple(
        (f"ag_s{par}", tuple(f"ag_r{par}_{p}" for p in t._peers))
        for par in (0, 1)
    )
    t._bar_slots = None
    t.worker = _FakeWorker(log)
    # Force the copy-out fast path: the device is CPU here, but the point of
    # these tests is the EVENT AND FLAG bookkeeping of the CUDA branch.
    t._async_h2d_ok = lambda dst: True
    return t


class _EventPatch:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        self._orig = torch.cuda.Event
        log = self.log
        torch.cuda.Event = lambda *a, **k: _FakeEvent(log)
        return self

    def __exit__(self, *exc):
        torch.cuda.Event = self._orig
        return False


class TestUcxNonBlockingCopyOut(CustomTestCase):
    def test_allgather_stage_blocks_and_peer_rows_do_not(self):
        """Stage-in D2H must block (UCX reads it host-side); the W-1 peer-row
        copy-outs -- 3 of the 4 drains of every decode all_gather -- must be
        non_blocking."""
        log = []
        t = _make_transport(log, world=3, rank=0)
        inp = torch.arange(8, dtype=torch.float32)
        with _EventPatch(log), _CopySpy() as spy:
            t.htccl_all_gather(_Comm(), inp.clone(), 0)

        send_ptr = t._slot("ag_s0", 8, torch.float32)[1]
        stage = spy.to_dst(send_ptr)
        self.assertEqual(len(stage), 1)
        self.assertFalse(
            stage[0][4], "the stage-in copy must stay BLOCKING: UCX reads "
            "the pinned buffer outside stream order"
        )
        for peer in (1, 2):
            rptr = t._slot(f"ag_r0_{peer}", 8, torch.float32)[1]
            outs = spy.from_src(rptr)
            self.assertEqual(len(outs), 1, f"peer {peer} row not copied out")
            self.assertTrue(
                outs[0][4],
                f"peer {peer} row copy-out must be non_blocking -- a "
                "blocking copy_ is a cudaStreamSynchronize per peer",
            )
        # every non_blocking read left a completion event on its slot
        for peer in (1, 2):
            self.assertIn(f"ag_r0_{peer}", t._h2d_events)
            self.assertEqual(t._h2d_events[f"ag_r0_{peer}"].recorded, 1)

    def test_allgather_syncs_slot_event_before_reposting_recv(self):
        """The NIC writes a recv slot from the moment the recv is posted, so
        the previous collective's async read of that slot must be waited on
        BEFORE the repost -- not merely assumed complete by stream order."""
        log = []
        t = _make_transport(log, world=3, rank=0)
        inp = torch.arange(8, dtype=torch.float32)
        with _EventPatch(log), _CopySpy():
            t.htccl_all_gather(_Comm(), inp.clone(), 0)
            events = {
                peer: t._h2d_events[f"ag_r0_{peer}"] for peer in (1, 2)
            }
            log.append(("second",))
            t.htccl_all_gather(_Comm(), inp.clone(), 0)

        second_at = log.index(("second",))
        tail = log[second_at:]
        for peer in (1, 2):
            rptr = t._slot(f"ag_r0_{peer}", 8, torch.float32)[1]
            sync_at = tail.index(("sync", id(events[peer])))
            post_at = tail.index(("post_recv", rptr))
            self.assertLess(
                sync_at, post_at,
                f"slot ag_r0_{peer} was reposted before its in-flight "
                "read-out was synchronized",
            )

    def test_allreduce_copyout_nonblocking_with_pinned_downcast(self):
        """The all_reduce copy-out (the second of its two drains) must be
        non_blocking; with fp32 reduction the downcast must happen on the CPU
        into a pinned slot first, because a cross-dtype copy_ converts through
        a PAGEABLE temporary whose cudaMemcpyAsync drains the stream."""
        log = []
        t = _make_transport(log, world=2, rank=0)
        inp = torch.arange(8, dtype=torch.float32).bfloat16()
        with _EventPatch(log), _CopySpy() as spy:
            out = t.htccl_all_reduce(_Comm(), inp.clone())

        stage_ptr = t._slot("ar_s0", 8, torch.float32)[1]
        stage = spy.to_dst(stage_ptr)
        self.assertEqual(len(stage), 1)
        self.assertFalse(stage[0][4], "stage-in must stay blocking")

        final = spy.to_dst(out.data_ptr())
        self.assertEqual(len(final), 1)
        self.assertTrue(final[0][4], "copy-out must be non_blocking")
        self.assertEqual(
            final[0][3], torch.bfloat16,
            "the H2D copy must be same-dtype: the fp32->bf16 downcast "
            "belongs on the CPU, in a pinned slot",
        )
        dn_ptr = t._slot("ar_s0:dn", 8, torch.bfloat16)[1]
        self.assertEqual(final[0][1], dn_ptr)
        self.assertIn("ar_s0:dn", t._h2d_events)

    def test_broadcast_receiver_copyout_nonblocking(self):
        log = []
        t = _make_transport(log, world=2, rank=1)
        tensor = torch.zeros(8, dtype=torch.float32)
        with _EventPatch(log), _CopySpy() as spy:
            t.htccl_broadcast(_Comm(), tensor, src=0)

        outs = spy.to_dst(tensor.data_ptr())
        self.assertEqual(len(outs), 1)
        self.assertTrue(outs[0][4], "receiver copy-out must be non_blocking")
        self.assertIn("bc", t._h2d_events)

    def test_wait_async_releases_pool_slots_with_completion_event(self):
        """Pool slots freed by wait_async carry the completion event of the
        in-flight result copy; _pool_acquire must wait on it before handing
        the buffer back out."""
        log = []
        t = _make_transport(log, world=2, rank=0)
        inp = torch.arange(8, dtype=torch.float32).bfloat16()
        with _EventPatch(log), _CopySpy() as spy:
            handle = t.all_reduce_async(_Comm(), inp.clone())
            issue_copy = spy.calls[0]
            self.assertFalse(
                issue_copy[4],
                "the ISSUE stage-in must stay blocking (the posts read the "
                "buffer host-side); making it async is the deferred-post "
                "follow-up, not a flag change",
            )
            out = t.wait_async(handle)
            self.assertTrue(spy.to_dst(out.data_ptr())[0][4])

            freed = [rec for recs in t._async_free.values() for rec in recs]
            self.assertGreater(len(freed), 0)
            events = {id(ev) for _, ev in freed if ev is not None}
            self.assertEqual(
                len(events), 1,
                "every slot of the handle must be released with the ONE "
                "completion event of its result copy",
            )
            for _, ev in freed:
                self.assertIsNotNone(ev)

            # reacquiring a freed buffer must wait for that event first
            ev = next(e for _, e in freed if e is not None)
            before = ev.synced
            t._pool_acquire(8, torch.float32)
            self.assertGreater(
                ev.synced, before,
                "_pool_acquire handed out a buffer whose previous read may "
                "still be in flight",
            )

    def test_pinned_downcast_matches_blocking_convert(self):
        """Equivalence pin (green before and after #246): the pinned-slot CPU
        downcast produces the same bytes as the converting copy_ it replaces
        -- both run the identical CPU convert kernel."""
        src = torch.cat(
            [
                torch.tensor(
                    [0.0, -0.0, 1.0, -1.0, 1e-40, -1e-40, 65504.0, 3.14159],
                    dtype=torch.float32,
                ),
                torch.linspace(-4.0, 4.0, 4096, dtype=torch.float32),
                torch.arange(4096, dtype=torch.float32) * (1.0 + 2 ** -10),
            ]
        )
        for dtype, bits in ((torch.bfloat16, torch.int16), (torch.float16, torch.int16)):
            via_slot = torch.empty(src.numel(), dtype=dtype)
            via_slot.copy_(src)  # what _h2d_async's pinned downcast runs
            via_to = src.to(dtype)  # what the blocking converting copy_ ran
            self.assertTrue(
                torch.equal(via_slot.view(bits), via_to.view(bits)),
                f"CPU downcast to {dtype} diverged from the converting copy_",
            )


if __name__ == "__main__":
    unittest.main()
