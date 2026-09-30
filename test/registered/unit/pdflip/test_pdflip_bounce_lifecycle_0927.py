# SPDX-License-Identifier: Apache-2.0
"""weight_exchange_bounce.py resource lifecycle (27.09.2026, review
agentlast w-ac450cdc2218/out/bounce_lifecycle.md, checked against the tree).

(1) the per-call lane stream of run_sequential_units: ALREADY destroyed on the tree (N4E, 25.09.,
    LaneStreamIsDestroyedN4E in test_pdflip_sequential_transport_1378.py) -- the review's corpus
    (e914e89fde) predates it. Not repeated here.
(2) the per-call SemSet: every call's sem_open handles (one fd + one mapping each) were never
    sem_close'd -- SemSet has no __del__. Pinned by counting this process's open fds.
(3) run_bounce_leg: the two streams and the bounce buffer were created BEFORE the try whose finally
    releases them -- a raise in between (a second create_stream, prime_band_drain) leaked them.
(4) LayerBounce.__init__: a raise after os.open (ftruncate/mmap/ledger event) orphaned fd/mm/pin.
(7) LayerBounce.close(): mm.close() and os.close() shared one try -- a raising munmap skipped the fd.
"""

import os
import tempfile
import unittest
from unittest import mock

from flliper.srt.pdflip import weight_exchange_bounce as bx
from flliper.srt.pdflip import weight_exchange_region as xr
from flliper.srt.pdflip import weight_exchange_transport as tp

try:
    from test_pdflip_sequential_transport_1378 import _desc, _MemOps, _TransportHarness  # noqa: E402
except ImportError:  # run from the repo root
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_pdflip_sequential_transport_1378 import _desc, _MemOps, _TransportHarness  # noqa: E402


def _fd_count() -> int:
    return len(os.listdir("/proc/self/fd"))


class SemSetIsClosedPerCall(_TransportHarness):
    def _descs(self, n=2):
        out = []
        for i in range(n):
            src, dst = self._vram_pair(64, i)
            self.ops.write(src, bytes([0x40 + i]) * 64)
            out.append(_desc(f"s{i}", 64, src_ptr=src, dst_ptr=dst))
        return out

    def test_every_semset_is_closed_on_both_phases(self):
        closed = []
        orig = tp.SemSet

        class Rec(orig):
            def close(self):
                closed.append(len(self._handles))
                super().close()

        descs = self._descs()
        with mock.patch.object(bx.tp, "SemSet", Rec):
            self.assertEqual(self._deposit(descs), "")
            self.assertEqual(self._collect(descs), "")
        self.assertEqual(len(closed), 2, "each run_sequential_units call must close its SemSet")
        self.assertTrue(all(n >= 1 for n in closed), closed)  # it had opened handles to close

    def test_open_fds_do_not_grow_over_many_lane_calls(self):
        descs = self._descs()
        self._deposit(descs)
        self._collect(descs)  # warm: first-use caches, the lane file
        before = _fd_count()
        for _ in range(8):
            self.assertEqual(self._deposit(descs), "")
            self.assertEqual(self._collect(descs), "")
        self.assertLessEqual(_fd_count() - before, 0, "sem_open fds leak per lane call")

    def test_semset_closed_when_the_lane_raises(self):
        closed = []
        orig = tp.SemSet

        class Rec(orig):
            def close(self):
                closed.append(True)
                super().close()

        class Boom(_MemOps):
            def memcpy_async(self, dst, src, nbytes, stream):
                raise RuntimeError("copy failed")

        descs = self._descs()
        self.ops = Boom()
        for i in range(2):
            src, dst = 0x1000 + i * 512, 0x8000 + i * 512
            self.ops.hook(src, 64)
            self.ops.hook(dst, 64)
        with mock.patch.object(bx.tp, "SemSet", Rec):
            with self.assertRaises(RuntimeError):
                self._deposit(descs)
        self.assertEqual(closed, [True])


class _LegOps:
    """Stand-in ops for run_bounce_leg's pre-try window."""

    def __init__(self, fail_on_stream=None):
        self.created, self.destroyed = [], []
        self.fail_on_stream = fail_on_stream
        self.registered, self.unregistered = [], []

    def create_stream(self, device):
        n = len(self.created) + 1
        if self.fail_on_stream == n:
            raise RuntimeError("stream %d refused" % n)
        self.created.append(0x7000 + n)
        return 0x7000 + n

    def destroy_stream(self, s):
        self.destroyed.append(s)

    def host_register(self, ptr, nbytes, flags):
        self.registered.append(ptr)

    def host_unregister(self, ptr):
        self.unregistered.append(ptr)

    def synchronize(self, s=0):
        return None


class RunBounceLegReleasesThePreTryWindow(unittest.TestCase):
    """Real LayerBounce, the transport tests' FakeDeviceOps, one flat unit."""

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="pdflip-bleg-")
        self.nonce = f"bleg0927{os.getpid()}"

    def tearDown(self):
        import shutil

        shutil.rmtree(self.root, ignore_errors=True)

    def _ops(self, fail_on_stream=None):
        from test_pdflip_xchg_transport_1273 import FakeDeviceOps

        class Ops(FakeDeviceOps):
            def __init__(self_inner, *a, **k):
                super().__init__(*a, **k)
                self_inner.created, self_inner.destroyed = [], []

            def create_stream(self_inner, device):
                if fail_on_stream == len(self_inner.created) + 1:
                    raise RuntimeError("stream refused")
                s = super().create_stream(device)
                self_inner.created.append(s)
                return s

            def destroy_stream(self_inner, s):
                self_inner.destroyed.append(s)
                return super().destroy_stream(s)

        return Ops(self.root, rank=0)

    def _leg(self, ops):
        from flliper.srt.pdflip import weight_exchange as wx

        n = 4096
        src, dst = ops.raw_malloc(0, n), ops.raw_malloc(0, n)
        self.fds = _fd_count()  # after the fake device's own backing files
        d = wx.XchgDesc(tag="t0", src_rank=0, dst_rank=0, param_name="model.layers.0.w", src_ptr=src,
                        dst_ptr=dst, kind=wx.FLAT, nbytes=n, rows=1, run_bytes=n, spitch=0, dpitch=0,
                        src_off=0, dst_off=0)
        return bx.run_bounce_leg([d], ops, self.nonce, slot_bytes=n, depth=1, mode=wx.INJECT_AUTHORITATIVE,
                                 shm_root=self.root, lane="c0")

    def test_a_second_stream_that_fails_releases_the_first_and_the_buffer(self):
        ops = self._ops(fail_on_stream=2)
        closes = []
        orig_close = bx.LayerBounce.close

        def rec_close(self):
            closes.append(self.path)
            orig_close(self)

        with mock.patch.object(bx.LayerBounce, "close", rec_close):
            with self.assertRaises(RuntimeError):
                self._leg(ops)
        self.assertEqual(len(ops.created), 1)
        self.assertEqual(ops.destroyed, ops.created, "the first stream must be destroyed")
        self.assertEqual(len(closes), 1, "the bounce buffer must be closed")
        self.assertLessEqual(_fd_count(), self.fds, "the bounce fd leaked")

    def test_the_normal_leg_still_releases_everything(self):
        ops = self._ops()
        self._leg(ops)
        self.assertEqual(len(ops.created), 2)
        self.assertEqual(sorted(ops.destroyed), sorted(ops.created))
        self.assertLessEqual(_fd_count(), self.fds)


class LayerBounceInitAndCloseNeverOrphanTheFd(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="pdflip-lb-")
        self.nonce = f"lb0927{os.getpid()}"

    def tearDown(self):
        import shutil

        shutil.rmtree(self.root, ignore_errors=True)

    def test_a_failing_mmap_closes_the_fd(self):
        fds = _fd_count()
        with mock.patch.object(bx._mmap, "mmap", side_effect=OSError(28, "No space left on device")):
            with self.assertRaises(OSError):
                bx.LayerBounce(_LegOps(), self.nonce, slot_bytes=4096, depth=1, shm_root=self.root, lane="c0")
        self.assertLessEqual(_fd_count(), fds, "the fd opened before the mmap leaked")

    def test_a_failing_slot_event_unpins_unmaps_and_closes(self):
        ops = _LegOps()
        fds = _fd_count()
        with mock.patch.object(bx, "_host_slot_event", side_effect=RuntimeError("ledger event failed")):
            with self.assertRaises(RuntimeError):
                bx.LayerBounce(ops, self.nonce, slot_bytes=4096, depth=1, shm_root=self.root, lane="c0")
        self.assertEqual(ops.unregistered, ops.registered)
        self.assertLessEqual(_fd_count(), fds)

    def test_close_closes_the_fd_even_when_the_munmap_raises(self):
        b = bx.LayerBounce(_LegOps(), self.nonce, slot_bytes=4096, depth=1, shm_root=self.root, lane="c0")
        fd = b._fd
        real_mm = b._mm

        class BadMM:
            def close(self_inner):
                real_mm.close()
                raise BufferError("cannot close exported pointers exist")

        b._mm = BadMM()
        with self.assertRaises(BufferError):
            b.close()
        with self.assertRaises(OSError):
            os.fstat(fd)  # closed


if __name__ == "__main__":
    unittest.main()
