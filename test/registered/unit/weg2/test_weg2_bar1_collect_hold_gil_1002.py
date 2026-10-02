"""FLIP-LEGS 02.10.: the BAR1 collector's copy-out keeps the GIL.

MEASURED (N4p f405217a61 / N4q 58361d5471, WEG2-BAR1 lane-time): collect
issue_ms 240-490 ms per lane per flip on the 5090 rank (P->D TP0 p2/p4, D->P
PP0 p2/p4 -- p2 is the x4 card's whole D->P flow), 350-490 ms on every rank in
the slow P->D flip (N4q epoch 8, legs 2425 ms), while the depositors waited
1.1-1.8 s in credit_ms for the slots those collectors free. Per call that is
70-270 us for a cudaMemcpyAsync that only queues a device-to-device copy.

Cause: ``CudartDeviceOps`` calls cudart through ``ctypes.CDLL``, which drops
the GIL for every call and must win it back behind the rank's other Python
threads. Measured on this rig's Python 3.12 (libc getpid, 2000 calls): idle
1.5 ms total; one busy thread 1811 ms (worst 5.1 ms); three busy threads
8687 ms (worst 349 ms); the same calls through ``ctypes.PyDLL`` 0.8-1.2 ms.

Pinned here (red before): the cudart adapter offers GIL-holding entry points
for the two async copies (PyDLL), the collector issues its copy-out through
them by default (SGLANG_WEG2_BAR1_COLLECT_HOLD_GIL=0 restores CDLL), a fake
without them keeps the plain calls, the bytes land identically, and the
lane-time line names the path (issue_gil=held|released).
"""
from __future__ import annotations

import ctypes
import os
import threading
from types import SimpleNamespace

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import bar1_lanes as b1  # noqa: E402
from sglang.srt.weg2 import weight_exchange_transport as tp  # noqa: E402

PAIRS = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))
SLOT = 4096


def _addr(buf):
    return ctypes.addressof(ctypes.c_char.from_buffer(buf))


class _HostOps:
    def create_stream(self, device):
        return 0

    def synchronize(self, stream):
        pass

    def memcpy_async(self, dst, src, n, stream):
        ctypes.memmove(dst, src, n)

    def memcpy2d_async(self, dst, dpitch, src, spitch, width, height, stream):
        for r in range(height):
            ctypes.memmove(dst + r * dpitch, src + r * spitch, width)


class _HeldOps(_HostOps):
    held_gil = True

    def __init__(self):
        self.calls = []

    def memcpy_async(self, dst, src, n, stream):
        self.calls.append("released")
        super().memcpy_async(dst, src, n, stream)

    def memcpy2d_async(self, *a):
        self.calls.append("released2d")
        super().memcpy2d_async(*a)

    def memcpy_async_held(self, dst, src, n, stream):
        self.calls.append("held")
        _HostOps.memcpy_async(self, dst, src, n, stream)

    def memcpy2d_async_held(self, *a):
        self.calls.append("held2d")
        _HostOps.memcpy2d_async(self, *a)


def _run(tmp_path, descs, col_ops, ring=4):
    window = bytearray(SLOT * ring)
    dep = b1.Bar1Lanes("n1", "P", 0, 0, PAIRS, log=lambda *_a: None, root=str(tmp_path))
    col = b1.Bar1Lanes("n1", "D", 1, 0, PAIRS, log=lambda *_a: None, root=str(tmp_path))
    col.recv["p0"] = SimpleNamespace(dptr=_addr(window), slot_bytes=SLOT, ring=ring)
    dep.peers["p0"] = SimpleNamespace(dev_ptr=_addr(window), slot_bytes=SLOT, ring=ring)
    out, logs = {}, []

    def _collect():
        out["c"] = b1.run_bar1_units(descs, col_ops, lanes=col, lane_key="p0", role="dst",
                                     seq="9-weights_1", phase="collect", budget_s=10.0,
                                     log=logs.append)

    th = threading.Thread(target=_collect)
    th.start()
    out["d"] = b1.run_bar1_units(descs, _HostOps(), lanes=dep, lane_key="p0", role="src",
                                 seq="9-weights_1", phase="deposit", budget_s=10.0,
                                 log=lambda *_a: None)
    th.join(20)
    assert not th.is_alive(), "collector hung"
    return out, logs


def _flat_descs(n, nbytes):
    srcs = [bytearray(os.urandom(nbytes)) for _ in range(n)]
    dsts = [bytearray(nbytes) for _ in range(n)]
    descs = [SimpleNamespace(kind=tp.FLAT, nbytes=nbytes, src_off=0, dst_off=0,
                             param_name=f"w{i}", tag="weights_1",
                             src_ptr=_addr(srcs[i]), dst_ptr=_addr(dsts[i]))
             for i in range(n)]
    return descs, srcs, dsts


def test_switch_default_on_and_off():
    assert b1.collect_hold_gil_on({})
    for off in ("0", "false", "no", "off"):
        assert not b1.collect_hold_gil_on({b1.ENV_COLLECT_HOLD_GIL: off})


def test_copier_choice():
    ops = _HeldOps()
    cp, cp2, mode = b1.collect_copiers(ops, {})
    assert mode == "held" and cp == ops.memcpy_async_held and cp2 == ops.memcpy2d_async_held
    cp, cp2, mode = b1.collect_copiers(ops, {b1.ENV_COLLECT_HOLD_GIL: "0"})
    assert mode == "released" and cp == ops.memcpy_async
    plain = _HostOps()                        # a fake without the entry points
    cp, cp2, mode = b1.collect_copiers(plain, {})
    assert mode == "released" and cp == plain.memcpy_async


def test_the_base_ops_fall_back_to_the_plain_call():
    seen = []

    class _O(tp.DeviceOps):
        def memcpy_async(self, *a):
            seen.append("m")

        def memcpy2d_async(self, *a):
            seen.append("m2")

    o = _O()
    assert o.held_gil is False
    o.memcpy_async_held(1, 2, 3, 0)
    o.memcpy2d_async_held(1, 2, 3, 4, 5, 6, 0)
    assert seen == ["m", "m2"]


@pytest.mark.parametrize("preplan", ["1", "0"])
def test_the_collector_issues_through_the_held_entry_points(tmp_path, monkeypatch, preplan):
    monkeypatch.setenv(b1.ENV_COLLECT_PREPLAN, preplan)
    monkeypatch.delenv(b1.ENV_COLLECT_HOLD_GIL, raising=False)
    descs, srcs, dsts = _flat_descs(9, SLOT)
    ops = _HeldOps()
    out, logs = _run(tmp_path, descs, ops)
    assert out == {"c": "", "d": ""}, out
    assert ops.calls and set(ops.calls) == {"held"}, ops.calls
    assert all(bytes(a) == bytes(b) for a, b in zip(srcs, dsts))
    line = [l for l in logs if "WEG2-BAR1 lane-time" in l]
    assert line and "issue_gil=held" in line[-1], line


def test_switch_off_keeps_cdll(tmp_path, monkeypatch):
    monkeypatch.setenv(b1.ENV_COLLECT_HOLD_GIL, "0")
    descs, srcs, dsts = _flat_descs(5, SLOT)
    ops = _HeldOps()
    out, logs = _run(tmp_path, descs, ops)
    assert out == {"c": "", "d": ""}, out
    assert set(ops.calls) == {"released"}, ops.calls
    assert all(bytes(a) == bytes(b) for a, b in zip(srcs, dsts))
    assert "issue_gil=released" in [l for l in logs if "lane-time" in l][-1]


def test_cudart_adapter_binds_the_held_calls_through_pydll():
    try:
        ops = tp.CudartDeviceOps()
    except OSError as e:                      # no libcudart on this desk
        pytest.skip(f"libcudart not loadable: {e}")
    assert ops.held_gil is True
    assert isinstance(ops.lib_held, ctypes.PyDLL)
    assert not isinstance(ops.lib, ctypes.PyDLL)
    assert ops.lib_held.cudaMemcpyAsync.argtypes == ops.lib.cudaMemcpyAsync.argtypes
    assert ops.lib_held.cudaMemcpy2DAsync.argtypes == ops.lib.cudaMemcpy2DAsync.argtypes
