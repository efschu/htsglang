"""FLIP-LEGS 02.10. (follow-up to 095772644f): the GIL-held pattern on the
remaining NON-BLOCKING lane calls, each behind its own switch, each named on
its lane-time line so a boot can compare held against released:

* the BAR1 depositor's copies into the peer window (device -> registered IO
  memory with a device pointer: async DMA) -- FLLIPER_PDFLIP_BAR1_DEPOSIT_HOLD_GIL,
  ``issue_gil=held|released|sm-copier`` on the deposit line;
* the per-batch synchronize asks ``cudaStreamQuery`` first, GIL held; a drained
  stream skips the blocking call (a busy one still synchronizes with the GIL
  released -- a synchronizing call never holds it) --
  FLLIPER_PDFLIP_LANE_SYNC_QUERY_HELD, ``sync_query=held|off sync_skipped=k/n``;
* the on-card SEQ lane's copies when it is IPC-staged (device to device) --
  FLLIPER_PDFLIP_SEQ_IPC_HOLD_GIL, ``issue_gil=`` on the PDFLIP-SEQ lane-time line.

The SEQ ``record_ms`` is file I/O (one JSON record per unit) and a polling
wait, not a CUDA issue call: deliberately NOT under this pattern.
"""
from __future__ import annotations

import ctypes
import os
import threading
from types import SimpleNamespace

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import bar1_lanes as b1  # noqa: E402
from flliper.srt.pdflip import weight_exchange_bounce as bx  # noqa: E402
from flliper.srt.pdflip import weight_exchange_transport as tp  # noqa: E402

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


class _Ops(_HostOps):
    held_gil = True

    def __init__(self, done=True):
        self.calls = []
        self.done = done

    def synchronize(self, stream):
        self.calls.append("sync")

    def stream_done_held(self, stream):
        self.calls.append("query")
        return self.done

    def memcpy_async(self, dst, src, n, stream):
        self.calls.append("released")
        super().memcpy_async(dst, src, n, stream)

    def memcpy_async_held(self, dst, src, n, stream):
        self.calls.append("held")
        _HostOps.memcpy_async(self, dst, src, n, stream)

    def memcpy2d_async_held(self, *a):
        self.calls.append("held2d")
        _HostOps.memcpy2d_async(self, *a)


def _run(tmp_path, descs, dep_ops, col_ops, ring=4):
    window = bytearray(SLOT * ring)
    dep = b1.Bar1Lanes("n1", "P", 0, 0, PAIRS, log=lambda *_a: None, root=str(tmp_path))
    col = b1.Bar1Lanes("n1", "D", 1, 0, PAIRS, log=lambda *_a: None, root=str(tmp_path))
    col.recv["p0"] = SimpleNamespace(dptr=_addr(window), slot_bytes=SLOT, ring=ring)
    dep.peers["p0"] = SimpleNamespace(dev_ptr=_addr(window), slot_bytes=SLOT, ring=ring)
    out, clog, dlog = {}, [], []

    def _collect():
        out["c"] = b1.run_bar1_units(descs, col_ops, lanes=col, lane_key="p0", role="dst",
                                     seq="9-weights_2", phase="collect", budget_s=10.0,
                                     log=clog.append)

    th = threading.Thread(target=_collect)
    th.start()
    out["d"] = b1.run_bar1_units(descs, dep_ops, lanes=dep, lane_key="p0", role="src",
                                 seq="9-weights_2", phase="deposit", budget_s=10.0,
                                 log=dlog.append)
    th.join(20)
    assert not th.is_alive(), "collector hung"
    return out, clog, dlog


def _flat_descs(n, nbytes):
    srcs = [bytearray(os.urandom(nbytes)) for _ in range(n)]
    dsts = [bytearray(nbytes) for _ in range(n)]
    descs = [SimpleNamespace(kind=tp.FLAT, nbytes=nbytes, src_off=0, dst_off=0,
                             param_name=f"w{i}", tag="weights_2",
                             src_ptr=_addr(srcs[i]), dst_ptr=_addr(dsts[i]))
             for i in range(n)]
    return descs, srcs, dsts


def _line(logs):
    return [l for l in logs if "PDFLIP-BAR1 lane-time" in l][-1]


def test_switches_default_on():
    assert b1.deposit_hold_gil_on({}) and b1.sync_query_held_on({})
    assert bx.seq_ipc_hold_gil_on({})
    assert not b1.deposit_hold_gil_on({b1.ENV_DEPOSIT_HOLD_GIL: "0"})
    assert not b1.sync_query_held_on({b1.ENV_SYNC_QUERY_HELD: "off"})
    assert not bx.seq_ipc_hold_gil_on({bx.ENV_SEQ_IPC_HOLD_GIL: "no"})


def test_the_depositor_issues_held_and_says_so(tmp_path, monkeypatch):
    for k in (b1.ENV_DEPOSIT_HOLD_GIL, b1.ENV_SYNC_QUERY_HELD):
        monkeypatch.delenv(k, raising=False)
    descs, srcs, dsts = _flat_descs(9, SLOT)
    dep = _Ops()
    out, clog, dlog = _run(tmp_path, descs, dep, _HostOps())
    assert out == {"c": "", "d": ""}, out
    copies = [c for c in dep.calls if c in ("held", "released")]
    assert copies and set(copies) == {"held"}, dep.calls
    assert all(bytes(a) == bytes(b) for a, b in zip(srcs, dsts))
    assert "issue_gil=held" in _line(dlog)


def test_the_depositor_switch_off_keeps_cdll(tmp_path, monkeypatch):
    monkeypatch.setenv(b1.ENV_DEPOSIT_HOLD_GIL, "0")
    descs, srcs, dsts = _flat_descs(5, SLOT)
    dep = _Ops()
    out, _, dlog = _run(tmp_path, descs, dep, _HostOps())
    assert out == {"c": "", "d": ""}
    assert set(c for c in dep.calls if c in ("held", "released")) == {"released"}
    assert "issue_gil=released" in _line(dlog)


def test_a_drained_stream_skips_the_blocking_sync(tmp_path, monkeypatch):
    monkeypatch.delenv(b1.ENV_SYNC_QUERY_HELD, raising=False)
    descs, srcs, dsts = _flat_descs(9, SLOT)
    dep, col = _Ops(done=True), _Ops(done=True)
    out, clog, dlog = _run(tmp_path, descs, dep, col)
    assert out == {"c": "", "d": ""}
    for ops, logs in ((dep, dlog), (col, clog)):
        assert "sync" not in ops.calls and ops.calls.count("query") >= 9, ops.calls
        assert "sync_query=held sync_skipped=9/9" in _line(logs), _line(logs)
    assert all(bytes(a) == bytes(b) for a, b in zip(srcs, dsts))


@pytest.mark.parametrize("answer", [False, None])
def test_a_busy_or_unknown_stream_still_synchronizes(tmp_path, monkeypatch, answer):
    monkeypatch.delenv(b1.ENV_SYNC_QUERY_HELD, raising=False)
    descs, _, _ = _flat_descs(6, SLOT)
    dep = _Ops(done=answer)
    out, _, dlog = _run(tmp_path, descs, dep, _HostOps())
    assert out == {"c": "", "d": ""}
    assert dep.calls.count("sync") == dep.calls.count("query") == 6, dep.calls
    assert "sync_skipped=0/6" in _line(dlog)


def test_sync_query_off_never_asks(tmp_path, monkeypatch):
    monkeypatch.setenv(b1.ENV_SYNC_QUERY_HELD, "0")
    descs, _, _ = _flat_descs(4, SLOT)
    dep = _Ops(done=True)
    out, _, dlog = _run(tmp_path, descs, dep, _HostOps())
    assert out == {"c": "", "d": ""}
    assert "query" not in dep.calls and dep.calls.count("sync") == 4
    assert "sync_query=off sync_skipped=0/4" in _line(dlog)


def test_a_fake_without_query_keeps_the_plain_sync(tmp_path, monkeypatch):
    monkeypatch.delenv(b1.ENV_SYNC_QUERY_HELD, raising=False)
    assert b1.sync_query_mode(_HostOps(), {}) is False
    assert tp.DeviceOps().stream_done_held(0) is None


class _FakeLib:
    def __init__(self, rc):
        self.rc = rc
        self.cleared = 0

    def cudaStreamQuery(self, s):
        return self.rc

    def cudaGetLastError(self):
        self.cleared += 1
        return 0


@pytest.mark.parametrize("rc,want,cleared", [(0, True, 0), (tp.CUDA_ERROR_NOT_READY, False, 1),
                                             (700, None, 0)])
def test_cudart_query_maps_codes_and_clears_not_ready(rc, want, cleared):
    ops = tp.CudartDeviceOps.__new__(tp.CudartDeviceOps)
    ops.lib_held = _FakeLib(rc)
    assert ops.stream_done_held(123) is want
    assert ops.lib_held.cleared == cleared


def test_cudart_binds_the_query_through_pydll():
    try:
        ops = tp.CudartDeviceOps()
    except OSError as e:
        pytest.skip(f"libcudart not loadable: {e}")
    assert isinstance(ops.lib_held, ctypes.PyDLL)
    assert ops.lib_held.cudaStreamQuery.argtypes == [ctypes.c_void_p]


def test_seq_copiers_held_only_device_to_device(monkeypatch):
    monkeypatch.delenv(bx.ENV_SEQ_IPC_HOLD_GIL, raising=False)
    ops = _Ops()
    assert bx._seq_copiers(ops, True) == (ops.memcpy_async_held, ops.memcpy2d_async_held)
    assert bx._seq_copiers(ops, False) == (ops.memcpy_async, ops.memcpy2d_async)
    plain = _HostOps()
    assert bx._seq_copiers(plain, True) == (plain.memcpy_async, plain.memcpy2d_async)
    monkeypatch.setenv(bx.ENV_SEQ_IPC_HOLD_GIL, "0")
    assert bx._seq_copiers(ops, True) == (ops.memcpy_async, ops.memcpy2d_async)
