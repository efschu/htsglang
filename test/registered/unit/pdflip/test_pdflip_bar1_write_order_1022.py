"""Fix A (22.10.) on the flip lanes: the deposited bytes must have ARRIVED and be
VISIBLE before the collector copies them out -- without assuming PCIe write order.

switch-p2p-wurzel-1022.md: a receiving card's engines can see a peer's posted
writes late and out of order (3080 WRITES_ORDERING = NONE, Relaxed Ordering on,
PEX switch in every path). The lanes' credit goes via the host, so "stream sync,
then socket" said nothing about arrival. Now:

* depositor: a 16-byte READ of the peer window on the batch's stream, behind its
  copies and before the sync that precedes the ``full`` credit (a PCIe read may
  not pass a posted write on its path);
* collector: ``cudaDeviceFlushGPUDirectRDMAWrites(current, TO_OWNER)`` after the
  credit and before the copy out (NVIDIA's documented visibility step).

Desk: the ORDER of those calls against the copies, syncs and credits, on the
lanes' own fakes. Not proven here: the metal (no lane harness without a server).
"""
from __future__ import annotations

import ctypes
import os
import threading
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import bar1_lanes as b1  # noqa: E402
from flliper.srt.pdflip import weight_exchange_transport as tp  # noqa: E402

PAIRS = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))
SLOT = 4096


import pytest


@pytest.fixture(autouse=True)
def _lanes3x80_call_counting(monkeypatch):
    """27b-lanes3x80-1023: this file pins exact copy/sync/probe call sequences
    of the lane; LANE-INTEGRITY adds sampled D2H reads + syncs per tag and
    PREPLAN-RECHECK a second probe pass -- both have their own tests in
    test_pdflip_bar1_lanes3x80_1023.py, so they are held off here."""
    monkeypatch.setenv("FLLIPER_PDFLIP_BAR1_INTEGRITY_SAMPLES", "0")
    monkeypatch.setenv("FLLIPER_PDFLIP_BAR1_PREPLAN_RECHECK_MS", "1e9")


def _addr(buf):
    return ctypes.addressof(ctypes.c_char.from_buffer(buf))


class _Ops:
    def __init__(self, name):
        self.name = name
        self.events = []
        self._n = 0
        self._keep = []

    def create_stream(self, device):
        self._n += 1
        return 100 + self._n

    def synchronize(self, stream):
        self.events.append(("sync", stream))

    def memcpy_async(self, dst, src, n, stream):
        self.events.append(("copy", stream, n))
        ctypes.memmove(dst, src, n)

    def memcpy2d_async(self, dst, dpitch, src, spitch, width, height, stream):
        self.events.append(("copy2d", stream))
        for r in range(height):
            ctypes.memmove(dst + r * dpitch, src + r * spitch, width)

    def ptr_attrs(self, addr):
        return (0, 2, 0)


class _DepOps(_Ops):
    def raw_malloc(self, device, nbytes):
        buf = bytearray(nbytes)
        self._keep.append(buf)
        return _addr(buf)


class _ColOps(_Ops):
    def rdma_flush(self, device):
        self.events.append(("flush", device))


def _run(tmp_path, ring, n, dep_ops, col_ops, env=None):
    b1._RB_SCRATCH.clear()
    window = bytearray(SLOT * ring)
    dep = b1.Bar1Lanes("n1", "P", 0, 0, PAIRS, log=lambda *_a: None, root=str(tmp_path))
    col = b1.Bar1Lanes("n1", "D", 1, 0, PAIRS, log=lambda *_a: None, root=str(tmp_path))
    col.recv["p0"] = SimpleNamespace(dptr=_addr(window), slot_bytes=SLOT, ring=ring)
    dep.peers["p0"] = SimpleNamespace(dev_ptr=_addr(window), slot_bytes=SLOT, ring=ring)
    srcs = [bytearray(os.urandom(SLOT)) for _ in range(n)]
    dsts = [bytearray(SLOT) for _ in range(n)]
    descs = [SimpleNamespace(kind=tp.FLAT, nbytes=SLOT, src_off=0, dst_off=0,
                             param_name=f"w{i}", tag="weights_9",
                             src_ptr=_addr(srcs[i]), dst_ptr=_addr(dsts[i]))
             for i in range(n)]
    out, dlog, clog = {}, [], []

    def _collect():
        out["c"] = b1.run_bar1_units(descs, col_ops, lanes=col, lane_key="p0", role="dst",
                                     seq="5-weights_9", phase="collect", budget_s=10.0,
                                     log=clog.append)

    th = threading.Thread(target=_collect)
    th.start()
    out["d"] = b1.run_bar1_units(descs, dep_ops, lanes=dep, lane_key="p0", role="src",
                                 seq="5-weights_9", phase="deposit", budget_s=10.0,
                                 log=dlog.append)
    th.join(20)
    assert not th.is_alive(), "collector hung"
    assert out == {"c": "", "d": ""}, out
    assert all(bytes(s) == bytes(d) for s, d in zip(srcs, dsts))
    return dlog, clog


def test_switch_defaults_on():
    assert b1.write_order_on({})
    assert not b1.write_order_on({b1.ENV_WRITE_ORDER: "0"})


def test_depositor_reads_back_every_batch_before_its_sync(tmp_path):
    dep, col = _DepOps("dep"), _ColOps("col")
    dlog, _ = _run(tmp_path, 3, 7, dep, col)
    # per stream: every SLOT-sized deposit copy is followed by a 16-byte read on
    # the same stream before that stream's next sync
    per = {}
    for ev in dep.events:
        per.setdefault(ev[1], []).append(ev)
    reads = 0
    for evs in per.values():
        pending = None
        for ev in evs:
            if ev[0] == "copy" and ev[2] == SLOT:
                assert pending is None, f"two batches on one stream without a readback: {evs}"
                pending = "deposit"
            elif ev[0] == "copy" and ev[2] == b1.READBACK_BYTES:
                assert pending == "deposit", "readback without a deposit before it"
                pending = "read"
                reads += 1
            elif ev[0] == "sync":
                assert pending in (None, "read"), f"sync before the readback: {evs}"
                pending = None
    assert reads == 7
    assert any("write_order=readbackx7" in line for line in dlog), dlog


def test_collector_flushes_after_the_credit_and_before_each_copy(tmp_path):
    dep, col = _DepOps("dep"), _ColOps("col")
    _, clog = _run(tmp_path, 3, 7, dep, col)
    kinds = [e[0] for e in col.events if e[0] in ("flush", "copy")]
    # strictly alternating: flush, copy, flush, copy, ...
    assert kinds == ["flush", "copy"] * 7, kinds
    assert any("write_order=flushx7" in line for line in clog), clog


def test_ops_without_the_methods_are_named_not_silent(tmp_path):
    dep, col = _Ops("plain"), _Ops("plain")
    dlog, clog = _run(tmp_path, 3, 4, dep, col)
    assert any("write_order=OFF why=ops_plain_has_no_raw_malloc" in line for line in dlog)
    assert any("write_order=OFF why=ops_plain_has_no_rdma_flush" in line for line in clog)


def test_switch_off_is_named(tmp_path, monkeypatch):
    monkeypatch.setenv(b1.ENV_WRITE_ORDER, "0")
    dep, col = _DepOps("dep"), _ColOps("col")
    dlog, clog = _run(tmp_path, 3, 4, dep, col)
    assert not [e for e in col.events if e[0] == "flush"]
    assert any(f"write_order=OFF why={b1.ENV_WRITE_ORDER}=0" in line for line in clog)


def test_the_real_ops_carry_the_flush():
    src = open(tp.__file__, encoding="utf-8").read()
    assert "def rdma_flush(self, device: int)" in src
    assert "cudaDeviceFlushGPUDirectRDMAWrites" in src
    assert "fn(0, 100)" in src          # TargetCurrentDevice, ToOwner
    assert callable(getattr(tp.CudartDeviceOps, "rdma_flush", None))
    assert callable(getattr(tp.CudartDeviceOps, "raw_malloc", None))
