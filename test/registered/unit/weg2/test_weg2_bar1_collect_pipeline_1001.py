"""01.10. (y6k legs, BAR1-Legs 2-5x unter Link-Physik): the collector was the
lane's metronome. Per batch it waited for ``full``, then ran the per-piece
work (no-write filter, the x33 pointer probe, address arithmetic), issued
the copies and SYNCED the stream before it sent ``free`` -- lag 0, while the
depositor already ran one behind. Measured on the lane-time lines: collect
issue_ms up to 140 of 151 ms (TP0 p4 11-weights_14, 131 units) and the
depositor of that lane 598 ms in credit_ms; deposit credit_ms ~50 % of the
P->D deposit time.

Two changes, each behind its own switch (default on):

* LAG -- the collector syncs and frees one batch behind like the depositor.
  Deadlock-free for ring >= 3: a depositor at batch h waits free(h - ring),
  and a collector that holds full(h - 2) has freed h - 3.
* PREPLAN -- the copy list and the pointer probes are built for the whole tag
  before the plan arrives; the loop between two credits only issues copies.

The lane-time line keeps wait_ms / credit_ms / issue_ms and adds lag= and
prep_ms=.
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


def _lanes(tmp_path, group, rank):
    return b1.Bar1Lanes("n1", group, rank, 0, PAIRS, log=lambda *_a: None, root=str(tmp_path))


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


class _RecordingOps(_HostOps):
    """Host copies with an ordered event log: which stream a copy went to,
    which stream was synced, and every destination probe."""

    def __init__(self):
        self.events = []
        self._n = 0

    def create_stream(self, device):
        self._n += 1
        return 100 + self._n

    def synchronize(self, stream):
        self.events.append(("sync", stream))

    def ptr_attrs(self, addr):
        self.events.append(("probe", addr))
        return (0, 2, 0)

    def memcpy_async(self, dst, src, n, stream):
        self.events.append(("copy", stream))
        super().memcpy_async(dst, src, n, stream)


def _run(tmp_path, ring, descs, col_ops, seq="5-weights_9"):
    window = bytearray(SLOT * ring)
    dep = _lanes(tmp_path, "P", 0)
    col = _lanes(tmp_path, "D", 1)
    col.recv["p0"] = SimpleNamespace(dptr=_addr(window), slot_bytes=SLOT, ring=ring)
    dep.peers["p0"] = SimpleNamespace(dev_ptr=_addr(window), slot_bytes=SLOT, ring=ring)
    out, logs = {}, []

    def _collect():
        out["c"] = b1.run_bar1_units(descs, col_ops, lanes=col, lane_key="p0", role="dst",
                                     seq=seq, phase="collect", budget_s=10.0,
                                     log=logs.append)

    th = threading.Thread(target=_collect)
    th.start()
    out["d"] = b1.run_bar1_units(descs, _HostOps(), lanes=dep, lane_key="p0", role="src",
                                 seq=seq, phase="deposit", budget_s=10.0, log=lambda *_a: None)
    th.join(20)
    assert not th.is_alive(), "collector hung"
    return out, logs


def _flat_descs(n, nbytes):
    srcs = [bytearray(os.urandom(nbytes)) for _ in range(n)]
    dsts = [bytearray(nbytes) for _ in range(n)]
    descs = [SimpleNamespace(kind=tp.FLAT, nbytes=nbytes, src_off=0, dst_off=0,
                             param_name=f"w{i}", tag="weights_9",
                             src_ptr=_addr(srcs[i]), dst_ptr=_addr(dsts[i]))
             for i in range(n)]
    return descs, srcs, dsts


def _sync_positions(events):
    """For each sync: how many copies were issued before it."""
    copies, out = 0, []
    for kind, _ in events:
        if kind == "copy":
            copies += 1
        elif kind == "sync":
            out.append(copies)
    return out


def test_knobs_default_on_and_switch_off():
    assert b1.collect_lag_on({}) and b1.collect_preplan_on({})
    assert not b1.collect_lag_on({b1.ENV_COLLECT_LAG: "0"})
    assert not b1.collect_preplan_on({b1.ENV_COLLECT_PREPLAN: "off"})


@pytest.mark.parametrize("ring", [3, 4])
def test_collector_frees_one_batch_behind(tmp_path, ring):
    # one FLAT unit per slot: batch g = copy g
    descs, srcs, dsts = _flat_descs(8, SLOT)
    ops = _RecordingOps()
    out, _ = _run(tmp_path, ring, descs, ops)
    assert out == {"c": "", "d": ""}
    assert all(bytes(d) == bytes(s) for d, s in zip(dsts, srcs))
    # lag 1: the first sync comes after the SECOND batch's copies, every
    # later one trails its batch by one; the tail drains the last batch
    assert _sync_positions(ops.events) == [2, 3, 4, 5, 6, 7, 8, 8]


def test_lag_switch_off_restores_sync_per_batch(tmp_path, monkeypatch):
    monkeypatch.setenv(b1.ENV_COLLECT_LAG, "0")
    descs, srcs, dsts = _flat_descs(6, SLOT)
    ops = _RecordingOps()
    out, logs = _run(tmp_path, 4, descs, ops)
    assert out == {"c": "", "d": ""}
    assert all(bytes(d) == bytes(s) for d, s in zip(dsts, srcs))
    assert _sync_positions(ops.events) == [1, 2, 3, 4, 5, 6]
    assert any("lane-time" in s and " lag=0 " in s for s in logs)


def test_ring_two_keeps_lag_zero(tmp_path):
    # ring 2 cannot pipeline both sides: the collector stays at lag 0
    descs, srcs, dsts = _flat_descs(5, SLOT)
    ops = _RecordingOps()
    out, _ = _run(tmp_path, 2, descs, ops)
    assert out == {"c": "", "d": ""}
    assert all(bytes(d) == bytes(s) for d, s in zip(dsts, srcs))
    assert _sync_positions(ops.events) == [1, 2, 3, 4, 5]


def test_every_probe_runs_before_the_first_copy(tmp_path):
    # units larger than a slot: unit i's first piece sits in a later batch
    descs, srcs, dsts = _flat_descs(4, 3 * SLOT // 2)
    ops = _RecordingOps()
    out, _ = _run(tmp_path, 4, descs, ops)
    assert out == {"c": "", "d": ""}
    assert all(bytes(d) == bytes(s) for d, s in zip(dsts, srcs))
    kinds = [k for k, _ in ops.events]
    first_copy = kinds.index("copy")
    probes = [i for i, k in enumerate(kinds) if k == "probe"]
    assert len(probes) == 4                       # one per destination, as before
    assert max(probes) < first_copy


def test_preplan_switch_off_probes_inside_the_loop(tmp_path, monkeypatch):
    monkeypatch.setenv(b1.ENV_COLLECT_PREPLAN, "0")
    descs, srcs, dsts = _flat_descs(4, 3 * SLOT // 2)
    ops = _RecordingOps()
    out, _ = _run(tmp_path, 4, descs, ops)
    assert out == {"c": "", "d": ""}
    assert all(bytes(d) == bytes(s) for d, s in zip(dsts, srcs))
    kinds = [k for k, _ in ops.events]
    assert max(i for i, k in enumerate(kinds) if k == "probe") > kinds.index("copy")


def test_strided_and_no_write_units_through_the_plan(tmp_path):
    rows, run, spitch, dpitch = 9, 1000, 1536, 2048
    s_src = bytearray(os.urandom(rows * spitch))
    s_dst = bytearray(rows * dpitch)
    f_src, f_dst = bytearray(os.urandom(5000)), bytearray(5000)
    nw_src, nw_dst = bytearray(os.urandom(700)), bytearray(700)
    descs = [
        SimpleNamespace(kind=tp.FLAT, nbytes=5000, src_off=0, dst_off=0, param_name="a",
                        tag="weights_9", src_ptr=_addr(f_src), dst_ptr=_addr(f_dst)),
        SimpleNamespace(kind=tp.STRIDED2D, nbytes=rows * run, rows=rows, run_bytes=run,
                        spitch=spitch, dpitch=dpitch, src_off=0, dst_off=0,
                        param_name="b", tag="weights_9",
                        src_ptr=_addr(s_src), dst_ptr=_addr(s_dst)),
        SimpleNamespace(kind=tp.FLAT, nbytes=700, src_off=0, dst_off=0, param_name="skip",
                        tag="weights_9", src_ptr=_addr(nw_src), dst_ptr=_addr(nw_dst)),
    ]
    window = bytearray(SLOT * 3)
    dep = _lanes(tmp_path, "P", 0)
    col = _lanes(tmp_path, "D", 1)
    col.recv["p0"] = SimpleNamespace(dptr=_addr(window), slot_bytes=SLOT, ring=3)
    dep.peers["p0"] = SimpleNamespace(dev_ptr=_addr(window), slot_bytes=SLOT, ring=3)
    out = {}

    def _collect():
        out["c"] = b1.run_bar1_units(descs, _HostOps(), lanes=col, lane_key="p0", role="dst",
                                     seq=3, phase="collect", no_write={"skip"}, budget_s=10.0,
                                     log=lambda *_a: None)

    th = threading.Thread(target=_collect)
    th.start()
    assert b1.run_bar1_units(descs, _HostOps(), lanes=dep, lane_key="p0", role="src", seq=3,
                             phase="deposit", no_write={"skip"}, budget_s=10.0,
                             log=lambda *_a: None) == ""
    th.join(20)
    assert out["c"] == ""
    assert bytes(f_dst) == bytes(f_src)
    for r in range(rows):
        assert s_dst[r * dpitch:r * dpitch + run] == s_src[r * spitch:r * spitch + run]
    assert bytes(nw_dst) == bytes(700)


def test_lane_time_line_keeps_its_fields_and_names_lag_and_prep(tmp_path):
    descs, _, _ = _flat_descs(4, SLOT)
    _, logs = _run(tmp_path, 4, descs, _RecordingOps())
    line = next(s for s in logs if "WEG2-BAR1 lane-time" in s)
    for f in ("wait_ms=", "credit_ms=", "issue_ms=", "copy_sync_ms=", "overlap_ms="):
        assert f in line, f
    assert " lag=1 " in line and "prep_ms=" in line
