"""27b-lanes3x80-1023: desk END-TO-END test of the MULTI-REGION BAR1 ring.

Drives the REAL ``Bar1Lanes.open_window`` (receiver, group D, rank 1) and the
REAL ``Bar1Lanes.connect_peer`` (depositor, group P, rank 0) of two objects in
one process over a REAL AF_UNIX socket (SCM_RIGHTS fd passing included) under
a tempdir root. Fakes sit ONLY at the driver boundary:

* the CUDA wrapper (``Bar1Lanes._cu``): ``dmabuf_fd`` hands out a real fd per
  region, ``register_io`` / ``dev_ptr`` / ``malloc`` / ``memcpy`` are
  recorded stand-ins (the readback probe reads are fast or slow on demand);
* ``bar1_window`` (a 256 MiB small BAR at a fixed base);
* the dmabuf holder (``Holder.hold`` returns an sg table inside that BAR,
  distinct per fd, in buffer order; ``release`` is recorded);
* the ``resource1_wc`` open (a sparse temp file, so the real ``mmap`` works).

The receiver is a 3080 D process: group windows world:0 15 MiB, tp:0 31 MiB,
dcp:0 39 MiB (+512 KiB remainder each), ONE receiving lane -> the multi-region
plan is 4 slots x 15 MiB, two in dcp:0 and two in tp:0.

CPU-only, no card.
"""

import ctypes
import mmap as real_mmap
import os
import shutil
import socket
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from flliper.srt.distributed.device_communicators import barlink_bar1 as b1
from flliper.srt.pdflip import bar1_lanes as bl
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

MIB = 1 << 20
BAR_BASE = 0xD0000000
BAR_SIZE = 256 * MIB
DST_BDF = "0000:18:00.0"
SRC_BDF = "0000:15:00.0"

# the receiver's own group windows: name -> (dptr, handle, full)
DCP_DPTR, TP_DPTR, WORLD_DPTR = 0xA000000000, 0xB000000000, 0xC000000000
OWN = {
    "dcp:0": (DCP_DPTR, 11, 40 * MIB),
    "tp:0": (TP_DPTR, 12, 32 * MIB),
    "world:0": (WORLD_DPTR, 13, 16 * MIB),
}
WINDOWS = {"world:0": 15 * MIB + (512 << 10), "tp:0": 31 * MIB + (512 << 10),
           "dcp:0": 39 * MIB + (512 << 10)}


class _OsProxy:
    """``bar1_lanes.os``: everything real, except the ``resource1_wc`` open and
    a spy on ``close``."""

    def __init__(self, resource_file: str):
        self._resource_file = resource_file
        self.closed: list = []
        self.opened_resource: list = []

    def __getattr__(self, name):
        return getattr(os, name)

    def open(self, path, flags, *a, **k):
        if str(path).endswith("resource1_wc"):
            self.opened_resource.append(str(path))
            return os.open(self._resource_file, os.O_RDWR)
        return os.open(path, flags, *a, **k)

    def close(self, fd):
        self.closed.append(int(fd))
        return os.close(fd)


class _MmapProxy:
    """``bar1_lanes.mmap``: everything real, every created map recorded."""

    def __init__(self):
        self.maps: list = []

    def __getattr__(self, name):
        return getattr(real_mmap, name)

    def mmap(self, *a, **k):
        m = real_mmap.mmap(*a, **k)
        self.maps.append(m)
        return m


class _FakeCu:
    def __init__(self, tmpdir: str, targets, read_s: float = 0.0):
        self.tmpdir, self.targets, self.read_s = tmpdir, list(targets), read_s
        self.dmabufs: list = []        # (dptr, handle, full, ordinal, fd, inode)
        self.registered: list = []     # (address, length)
        self.unregistered: list = []
        self.dev_map: dict = {}
        self.reads_from: dict = {}
        self.freed: list = []

    def vmm_alloc(self, *_a):
        raise AssertionError("vmm_alloc is not used on a small BAR")

    def dmabuf_fd(self, dptr, handle, size, ordinal):
        fd = os.open(os.path.join(self.tmpdir, f"dmabuf{len(self.dmabufs)}"),
                     os.O_RDWR | os.O_CREAT, 0o600)
        self.dmabufs.append((int(dptr), int(handle), int(size), int(ordinal), fd,
                             os.fstat(fd).st_ino))
        return fd, [], "fake"

    def register_io(self, address, length):
        self.registered.append((int(address), int(length)))
        k = len(self.registered) - 1
        # the depositor's k-th mapping stands for the receiver's k-th region
        self.dev_map[int(address)] = (self.targets[k] if k < len(self.targets)
                                      else int(address) + 0x100000000)

    def unregister(self, address):
        self.unregistered.append(int(address))

    def dev_ptr(self, host_address):
        return self.dev_map[int(host_address)]

    def malloc(self, n):
        return 0x1000

    def free(self, p):
        self.freed.append(p)

    def memcpy(self, dst, src, n):
        if src in self.targets:
            self.reads_from[src] = self.reads_from.get(src, 0) + 1
            if self.read_s:
                time.sleep(self.read_s)
        else:
            ctypes.memmove(dst, b"\x00" * n, n)


class _FakeHolder:
    def __init__(self, fail_on: int = 0):
        self.fail_on = fail_on
        self.held: list = []           # (fd, bdf, handle, inode)
        self.released: list = []
        self.calls = 0

    def hold(self, fd, bdf, max_entries=1024):
        self.calls += 1
        if self.fail_on and self.calls == self.fail_on:
            raise RuntimeError(f"fake holder: hold #{self.calls} refused")
        k = self.calls - 1
        start = BAR_BASE + 4 * MIB + k * 64 * MIB
        sg = [b1.SgEntry(start + j * 16 * MIB, 16 * MIB) for j in range(3)]   # 48 MiB, ascending
        handle = 100 + k
        self.held.append((int(fd), str(bdf), handle, os.fstat(fd).st_ino))
        return handle, sg, 48 * MIB

    def release(self, handle):
        self.released.append(int(handle))


class _Rig:
    """Both Bar1Lanes objects plus every patch; ``close()`` undoes them."""

    def __init__(self, env: dict, read_s: float = 0.0, hold_fail_on: int = 0):
        self.tmp = tempfile.mkdtemp(prefix="blm")
        self.d_log, self.p_log = [], []
        res = os.path.join(self.tmp, "resource1_wc")
        with open(res, "wb") as fh:
            fh.truncate(BAR_SIZE + 4 * MIB)            # sparse
        self.osx = _OsProxy(res)
        self.mm = _MmapProxy()
        self.cu = _FakeCu(self.tmp, [DCP_DPTR, TP_DPTR], read_s)
        self.holder = _FakeHolder(hold_fail_on)
        self.recv_fds_seen: list = []
        self.send_fds_seen: list = []
        self.dst = bl.Bar1Lanes("nonce", "D", 1, 1, [(0, 1)], log=self.d_log.append, root=self.tmp)
        self.src = bl.Bar1Lanes("nonce", "P", 0, 0, [(0, 1)], log=self.p_log.append, root=self.tmp)
        self.dst._bdf, self.src._bdf = DST_BDF, SRC_BDF
        self.dst._dst_lanes = ["p0"]
        self.dst.group_windows = lambda: dict(WINDOWS)
        self.dst._group_own.update(OWN)
        self.dst.big_cards = lambda: []
        real_recv, real_send = socket.recv_fds, socket.send_fds

        def recv_spy(sock, bufsize, maxfds, *a, **k):
            r = real_recv(sock, bufsize, maxfds, *a, **k)
            self.recv_fds_seen.append(list(r[1]))
            return r

        def send_spy(sock, buffers, fds, *a, **k):
            self.send_fds_seen.append(list(fds))
            return real_send(sock, buffers, fds, *a, **k)

        bar1_window = lambda bdf: b1.Bar1Window(bdf=str(bdf), base=BAR_BASE, size=BAR_SIZE)  # noqa: E731
        self.patches = [
            mock.patch.dict(os.environ, env),
            mock.patch.object(bl.Bar1Lanes, "_cu", lambda _self: self.cu),
            mock.patch.object(b1, "bar1_window", bar1_window),
            mock.patch.object(b1, "Holder", lambda: self.holder),
            mock.patch.object(bl, "os", self.osx),
            mock.patch.object(bl, "mmap", self.mm),
            mock.patch.object(socket, "recv_fds", recv_spy),
            mock.patch.object(socket, "send_fds", send_spy),
        ]
        for p in self.patches:
            p.start()

    def close(self):
        for p in reversed(self.patches):
            p.stop()
        for w in list(self.dst.recv.values()):
            for c in (w.conn,):
                try:
                    c.close()
                except Exception:  # noqa: BLE001
                    pass
            try:
                w.listener.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                w.listener.close()
            except OSError:
                pass
        for p in list(self.src.peers.values()):
            for c in (p.sock,):
                try:
                    c.close()
                except Exception:  # noqa: BLE001
                    pass
            for m in [p.mmap_obj] + [e[1] for e in p.extra]:
                try:
                    m.close()
                except Exception:  # noqa: BLE001
                    pass
        for m in self.mm.maps:
            try:
                m.close()
            except Exception:  # noqa: BLE001
                pass
        for d in self.cu.dmabufs:
            try:
                os.close(d[4])
            except OSError:
                pass
        shutil.rmtree(self.tmp, ignore_errors=True)

    # the received fds are the ones recv_fds returned, whatever their count
    def received(self):
        return [fd for batch in self.recv_fds_seen for fd in batch]


_ENV = {bl.ENV_ON: "1", bl.ENV_WRITE_ORDER: "1"}


class TestMultiRegionEndToEnd(CustomTestCase):
    def _rig(self, env_extra=None, **kw):
        env = dict(_ENV)
        env.update(env_extra or {})
        rig = _Rig(env, **kw)
        self.addCleanup(rig.close)
        return rig

    # -- 1: the multi-region ring end to end ----------------------------------
    def test_two_regions_served_mapped_and_tabled(self):
        rig = self._rig({bl.ENV_MULTI_REGION: "1"})
        w = rig.dst.open_window("p0")
        self.assertIsNotNone(w, rig.dst.refusals)
        S = 15 * MIB
        self.assertEqual((w.ring, w.slot_bytes, w.size), (4, S, 4 * S))
        self.assertEqual(w.borrowed, "dcp:0+tp:0")
        self.assertEqual(len(w.slot_addrs), 4)
        self.assertEqual(w.slot_addrs, [DCP_DPTR, DCP_DPTR + S, TP_DPTR, TP_DPTR + S])
        in_dcp = [a for a in w.slot_addrs if DCP_DPTR <= a < DCP_DPTR + WINDOWS["dcp:0"]]
        in_tp = [a for a in w.slot_addrs if TP_DPTR <= a < TP_DPTR + WINDOWS["tp:0"]]
        self.assertEqual((len(in_dcp), len(in_tp)), (2, 2))
        # world:0 is never borrowed; each region is its own dma-buf, dcp:0 first
        self.assertEqual([(d[0], d[1], d[2], d[3]) for d in rig.cu.dmabufs],
                         [(DCP_DPTR, 11, 40 * MIB, 1), (TP_DPTR, 12, 32 * MIB, 1)])
        self.assertEqual(len(w.extra_fds), 1)

        p = rig.src.connect_peer("p0", timeout_s=5.0)
        self.assertIsNotNone(p, rig.src.refusals)

        # two fds sent, two received, each a distinct open file in ring order
        self.assertEqual([len(f) for f in rig.send_fds_seen], [2])
        self.assertEqual([len(f) for f in rig.recv_fds_seen], [2])
        self.assertEqual(len(set(rig.received())), 2)
        self.assertEqual([h[3] for h in rig.holder.held], [d[5] for d in rig.cu.dmabufs])
        self.assertEqual([h[1] for h in rig.holder.held], [SRC_BDF, SRC_BDF])
        # every received fd was closed again
        self.assertTrue(set(rig.received()) <= set(rig.osx.closed), (rig.received(), rig.osx.closed))
        # the resource file was opened once per region, on the receiver's bdf
        self.assertEqual(rig.osx.opened_resource,
                         [f"/sys/bus/pci/devices/{DST_BDF}/resource1_wc"] * 2)

        # depositor side: 4 slot addresses, 15 MiB apart inside a region, the
        # second region derived from ITS OWN device pointer
        (a0, l0), (a1, l1) = rig.cu.registered
        self.assertEqual((l0, l1), (2 * S, 2 * S))
        dev0, dev1 = rig.cu.dev_map[a0], rig.cu.dev_map[a1]
        self.assertNotEqual(dev0, dev1)
        self.assertEqual(p.dev_ptr, dev0)
        self.assertEqual((p.ring, p.slot_bytes, p.size), (4, S, 4 * S))
        self.assertEqual(len(p.slot_addrs), 4)
        self.assertEqual(p.slot_addrs[1] - p.slot_addrs[0], S)
        self.assertEqual(p.slot_addrs[0], dev0)
        self.assertEqual(p.slot_addrs[2], dev1)
        self.assertEqual(p.slot_addrs[3] - p.slot_addrs[2], S)
        self.assertEqual((p.holder_handle, [(e[0], e[2]) for e in p.extra]),
                         (100, [(101, a1)]))
        self.assertEqual(p.reg_address, a0)
        self.assertEqual(rig.mm.maps[0].closed, False)
        self.assertEqual(rig.mm.maps[1].closed, False)

        # the slot tables: each side's own tuple; the fake maps the depositor's
        # k-th mapping onto the receiver's k-th region, so they coincide
        self.assertEqual(rig.dst.slot_table("p0", "dst"), tuple(w.slot_addrs))
        self.assertEqual(rig.src.slot_table("p0", "src"), tuple(p.slot_addrs))
        self.assertEqual(rig.dst.slot_table("p0", "dst"), rig.src.slot_table("p0", "src"))
        self.assertIsNone(rig.dst.slot_table("p0", "src"))
        self.assertIsNone(rig.src.slot_table("p0", "dst"))
        self.assertTrue(rig.dst.peer_mapped("p0"))

        # the readback probe ran on region 0 only
        self.assertEqual(rig.cu.reads_from, {dev0: bl.READBACK_PROBE_N_DEFAULT})
        # log lines on both sides
        d_line = [m for m in rig.d_log if "role=dst" in m and "window=" in m]
        p_line = [m for m in rig.p_log if "role=src" in m and "mapped" in m]
        self.assertEqual((len(d_line), len(p_line)), (1, 1), (rig.d_log, rig.p_log))
        self.assertIn("multi_region=on", d_line[0])
        self.assertIn("multi_region=on", p_line[0])
        self.assertIn("regions=2", d_line[0])
        self.assertIn("regions=2", p_line[0])
        self.assertIn("readback_probe_ms=", p_line[0])

    # -- 2: switch off -> the single-region plan byte for byte ------------------
    def test_multi_region_off_is_one_region_ring_4x9(self):
        rig = self._rig({bl.ENV_MULTI_REGION: "0"})
        w = rig.dst.open_window("p0")
        self.assertIsNotNone(w, rig.dst.refusals)
        S = 9 * MIB
        self.assertEqual((w.ring, w.slot_bytes, w.size), (4, S, 4 * S))
        self.assertEqual(w.borrowed, "dcp:0")
        self.assertIsNone(w.slot_addrs)
        self.assertEqual(w.extra_fds, [])
        self.assertEqual(w.dptr, DCP_DPTR)
        p = rig.src.connect_peer("p0", timeout_s=5.0)
        self.assertIsNotNone(p, rig.src.refusals)
        self.assertEqual([len(f) for f in rig.send_fds_seen], [1])
        self.assertEqual([len(f) for f in rig.recv_fds_seen], [1])
        self.assertTrue(set(rig.received()) <= set(rig.osx.closed))
        self.assertIsNone(rig.dst.slot_table("p0", "dst"))
        self.assertIsNone(rig.src.slot_table("p0", "src"))
        self.assertIsNone(p.slot_addrs)
        self.assertEqual(p.extra, [])
        self.assertEqual((p.ring, p.slot_bytes, p.size), (4, S, 4 * S))
        self.assertEqual(len(rig.cu.registered), 1)
        self.assertEqual(len(rig.cu.dmabufs), 1)
        self.assertEqual(p.dev_ptr, DCP_DPTR)
        for m in rig.d_log + rig.p_log:
            self.assertNotIn("multi_region=on", m)

    # -- 3: the second region cannot be held ----------------------------------
    def test_second_region_hold_failure_unwinds_the_first(self):
        rig = self._rig({bl.ENV_MULTI_REGION: "1"}, hold_fail_on=2)
        self.assertIsNotNone(rig.dst.open_window("p0"))
        p = rig.src.connect_peer("p0", timeout_s=5.0)
        self.assertIsNone(p)
        why = rig.src.refusals["p0"]
        self.assertTrue(why.startswith("map:"), why)
        self.assertIn("hold #2 refused", why)
        # region 0 was undone: holder released, unregistered, mmap closed
        self.assertEqual(rig.holder.released, [100])
        (a0, _l0), = rig.cu.registered
        self.assertEqual(rig.cu.unregistered, [a0])
        self.assertEqual(len(rig.mm.maps), 1)
        self.assertTrue(rig.mm.maps[0].closed)
        # nothing published
        self.assertNotIn("p0", rig.src.peers)
        self.assertIsNone(rig.src.slot_table("p0", "src"))
        self.assertFalse(rig.dst.peer_mapped("p0"))
        self.assertFalse(any("mapped" in m and "role=src" in m and "REFUSED" not in m
                             for m in rig.p_log), rig.p_log)
        self.assertTrue(any("REFUSED map:" in m for m in rig.p_log), rig.p_log)
        # both received fds closed all the same
        self.assertEqual([len(f) for f in rig.recv_fds_seen], [2])
        self.assertTrue(set(rig.received()) <= set(rig.osx.closed))

    # -- 4: the readback probe refuses a two-region mapping ---------------------
    def test_readback_probe_refusal_releases_both_regions(self):
        rig = self._rig({bl.ENV_MULTI_REGION: "1"}, read_s=0.008)
        self.assertIsNotNone(rig.dst.open_window("p0"))
        p = rig.src.connect_peer("p0", timeout_s=5.0)
        self.assertIsNone(p)
        why = rig.src.refusals["p0"]
        self.assertIn("readback stalls", why)
        self.assertEqual(sorted(rig.holder.released), [100, 101])
        regs = [a for a, _l in rig.cu.registered]
        self.assertEqual(len(regs), 2)
        self.assertEqual(sorted(rig.cu.unregistered), sorted(regs))
        self.assertEqual(len(rig.mm.maps), 2)
        self.assertTrue(all(m.closed for m in rig.mm.maps))
        self.assertNotIn("p0", rig.src.peers)
        self.assertFalse(rig.dst.peer_mapped("p0"))
        self.assertTrue(any("role=src REFUSED" in m and "readback stalls" in m
                            for m in rig.p_log), rig.p_log)
        self.assertEqual([len(f) for f in rig.recv_fds_seen], [2])
        self.assertTrue(set(rig.received()) <= set(rig.osx.closed))
        # the probe stopped at the second stalled read (READBACK-CONFIRM=2)
        self.assertEqual(sum(rig.cu.reads_from.values()), 2)


if __name__ == "__main__":
    unittest.main()
