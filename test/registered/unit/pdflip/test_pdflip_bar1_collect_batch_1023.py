"""The BAR1 collector issues each batch's copy-out as ONE batched runtime call
(barlink-lanes2-1023, FLLIPER_PDFLIP_BAR1_COLLECT_BATCH, default on).

Metal (boot dkr27bnvfp4duorebarbar1duo10230932): 17 of the 49 BAR1 tags under
10 GB/s were collector CPU -- 176-355 pieces per tag, issue_ms 0.01-0.25 ms per
piece, spikes under GIL competition (p1 5-weights_7 issue 84 ms, p0
5-weights_1 52 ms). The collector now builds every batch's copy list in the
preplan and hands it to the ops' batch copier (cudaMemcpyBatchAsync +
cudaMemcpy3DBatchAsync behind a device canary, weight_exchange_transport).

Pinned here with a host-emulated copier (desk, no GPU):
* bytes land exactly as on the per-piece path, flat and strided, no-write
  pieces untouched; ONE call per batch, zero per-piece calls;
* the rows handed over are the per-piece call's arguments (2D: dpitch=a,
  spitch=width=run_bytes, height=rows);
* every refusal is the per-piece path, named on the lane-time line: switch
  off, ops without a copier, the copier refused, preplan off, stream 0, a
  failing call mid-tag (the rest goes per piece, data intact);
* the depositor never uses it.
"""
from __future__ import annotations

import ctypes
import os
import threading
from types import SimpleNamespace

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import bar1_lanes as b1  # noqa: E402
from flliper.srt.pdflip import weight_exchange_transport as tp  # noqa: E402

PAIRS = ((0, 1), (1, 0))
SLOT = 4096


def _addr(buf):
    return ctypes.addressof(ctypes.c_char.from_buffer(buf))


class _HostOps:
    name = "host"

    def __init__(self):
        self.piece_calls = 0
        self._n = 0

    def create_stream(self, device):
        self._n += 1
        return 100 + self._n

    def synchronize(self, stream):
        pass

    def memcpy_async(self, dst, src, n, stream):
        self.piece_calls += 1
        ctypes.memmove(dst, src, n)

    def memcpy2d_async(self, dst, dpitch, src, spitch, width, height, stream):
        self.piece_calls += 1
        for r in range(height):
            ctypes.memmove(dst + r * dpitch, src + r * spitch, width)


class _Copier:
    """Host emulation of weight_exchange_transport.CudartBatchCopier."""

    def __init__(self, fail_at=None):
        self.issued = []
        self.prepared = []
        self.fail_at = fail_at

    def prepare(self, rows):
        rows = list(rows)
        self.prepared.append(rows)
        return rows

    def issue(self, prepared, stream):
        if not stream:
            raise ValueError("stream 0")
        if self.fail_at is not None and len(self.issued) == self.fail_at:
            self.issued.append(None)
            raise RuntimeError("cudaMemcpyBatchAsync rc=1 invalid argument")
        self.issued.append((stream, len(prepared)))
        for kind, dst, src, n, dpitch, spitch, rows in prepared:
            if kind == tp.FLAT:
                ctypes.memmove(dst, src, n)
            else:
                for r in range(rows):
                    ctypes.memmove(dst + r * dpitch, src + r * spitch, n)


class _BatchOps(_HostOps):
    def __init__(self, copier=None, why=""):
        super().__init__()
        self.copier = copier
        self.why = why
        self.asked = []

    def batch_copier(self, device):
        self.asked.append(device)
        return self.copier, self.why


def _mixed_descs():
    rows, run, spitch, dpitch = 9, 1000, 1536, 2048
    s_src = bytearray(os.urandom(rows * spitch))
    s_dst = bytearray(rows * dpitch)
    f_src, f_dst = bytearray(os.urandom(5000)), bytearray(5000)
    g_src, g_dst = bytearray(os.urandom(9000)), bytearray(9000)
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
        SimpleNamespace(kind=tp.FLAT, nbytes=9000, src_off=0, dst_off=0, param_name="c",
                        tag="weights_9", src_ptr=_addr(g_src), dst_ptr=_addr(g_dst)),
    ]
    bufs = dict(rows=rows, run=run, spitch=spitch, dpitch=dpitch, s_src=s_src, s_dst=s_dst,
                f=(f_src, f_dst), g=(g_src, g_dst), nw=(nw_src, nw_dst))
    return descs, bufs


def _check(bufs):
    assert bytes(bufs["f"][1]) == bytes(bufs["f"][0])
    assert bytes(bufs["g"][1]) == bytes(bufs["g"][0])
    r_, run, sp, dp = bufs["rows"], bufs["run"], bufs["spitch"], bufs["dpitch"]
    for r in range(r_):
        assert bufs["s_dst"][r * dp:r * dp + run] == bufs["s_src"][r * sp:r * sp + run]
    assert bytes(bufs["nw"][1]) == bytes(700)


def _run(tmp_path, col_ops, dep_ops=None, ring=3):
    descs, bufs = _mixed_descs()
    window = bytearray(SLOT * ring)
    dep = b1.Bar1Lanes("n1", "P", 0, 0, PAIRS, log=lambda *_a: None, root=str(tmp_path))
    col = b1.Bar1Lanes("n1", "D", 1, 0, PAIRS, log=lambda *_a: None, root=str(tmp_path))
    col.recv["p0"] = SimpleNamespace(dptr=_addr(window), slot_bytes=SLOT, ring=ring)
    dep.peers["p0"] = SimpleNamespace(dev_ptr=_addr(window), slot_bytes=SLOT, ring=ring)
    out, clog, dlog = {}, [], []

    def _collect():
        out["c"] = b1.run_bar1_units(descs, col_ops, lanes=col, lane_key="p0", role="dst",
                                     seq="5-weights_9", phase="collect", no_write={"skip"},
                                     budget_s=10.0, log=clog.append)

    th = threading.Thread(target=_collect)
    th.start()
    out["d"] = b1.run_bar1_units(descs, dep_ops or _HostOps(), lanes=dep, lane_key="p0",
                                 role="src", seq="5-weights_9", phase="deposit",
                                 no_write={"skip"}, budget_s=10.0, log=dlog.append)
    th.join(20)
    assert not th.is_alive(), "collector hung"
    assert out == {"c": "", "d": ""}, out
    _check(bufs)
    cline = next(s for s in clog if "lane-time" in s)
    dline = next(s for s in dlog if "lane-time" in s)
    nb = int(cline.split(" batches=")[1].split()[0])
    return cline, dline, nb


def test_switch_default_on():
    assert b1.collect_batch_on({}) is True
    assert b1.collect_batch_on({b1.ENV_COLLECT_BATCH: "0"}) is False
    assert b1.ENV_COLLECT_BATCH == "FLLIPER_PDFLIP_BAR1_COLLECT_BATCH"


def test_one_call_per_batch_and_no_piece_calls(tmp_path, monkeypatch):
    monkeypatch.delenv(b1.ENV_COLLECT_BATCH, raising=False)
    cp = _Copier()
    ops = _BatchOps(cp)
    cline, dline, nb = _run(tmp_path, ops)
    assert ops.piece_calls == 0
    assert len(cp.issued) == nb >= 4          # every batch has a copy (no batch is all no-write)
    assert all(st >= 100 for st, _n in cp.issued)
    assert f" collect_batch=on calls={nb}" in cline
    assert "collect_batch" not in dline       # the depositor never batches


def test_rows_are_the_per_piece_arguments(tmp_path, monkeypatch):
    monkeypatch.delenv(b1.ENV_COLLECT_BATCH, raising=False)
    cp = _Copier()
    _run(tmp_path, _BatchOps(cp))
    rows = [r for batch in cp.prepared for r in batch]
    flat = [r for r in rows if r[0] == tp.FLAT]
    two = [r for r in rows if r[0] != tp.FLAT]
    assert two and flat
    assert all(r[4:] == (0, 0, 0) for r in flat)
    for kind, dst, src, width, dpitch, spitch, nrows in two:
        assert dpitch == 2048 and spitch == width and width == 1000 and nrows >= 1
    assert sum(r[3] for r in flat) == 5000 + 9000        # the no-write piece is not there
    assert sum(r[3] * r[6] for r in two) == 9 * 1000


def test_batch_rows_shape():
    row = [(tp.FLAT, 10, 5, 7, 0, 0), (tp.STRIDED2D, 20, 64, 2048, 1000, 3)]
    assert b1.batch_rows(row, 1000) == [
        (tp.FLAT, 10, 1005, 7, 0, 0, 0),
        (tp.STRIDED2D, 20, 1064, 1000, 2048, 1000, 3),
    ]
    assert b1.batch_rows([(tp.FLAT, 10, 5, 0, 0, 0), (tp.STRIDED2D, 20, 64, 2048, 1000, 0)], 0) == []


@pytest.mark.parametrize("case", ["switch_off", "no_method", "refused", "preplan_off"])
def test_refusals_are_the_per_piece_path_named(tmp_path, monkeypatch, case):
    monkeypatch.delenv(b1.ENV_COLLECT_BATCH, raising=False)
    cp = _Copier()
    ops = _BatchOps(cp)
    want = ""
    if case == "switch_off":
        monkeypatch.setenv(b1.ENV_COLLECT_BATCH, "0")
        want = "why=FLLIPER_PDFLIP_BAR1_COLLECT_BATCH=0"
    elif case == "no_method":
        ops = _HostOps()
        want = "why=ops_host_has_no_batch_copier"
    elif case == "refused":
        ops = _BatchOps(None, "canary mismatch at byte 17")
        want = "why=canary_mismatch_at_byte_17"
    elif case == "preplan_off":
        monkeypatch.setenv(b1.ENV_COLLECT_PREPLAN, "0")
        want = "why=FLLIPER_PDFLIP_BAR1_COLLECT_PREPLAN=0"
    cline, _d, nb = _run(tmp_path, ops)
    assert cp.issued == []
    assert ops.piece_calls >= nb
    assert " collect_batch=OFF calls=0 " in cline and want in cline, cline


def test_a_failing_call_mid_tag_goes_per_piece_with_data_intact(tmp_path, monkeypatch):
    monkeypatch.delenv(b1.ENV_COLLECT_BATCH, raising=False)
    cp = _Copier(fail_at=1)
    ops = _BatchOps(cp)
    cline, _d, nb = _run(tmp_path, ops)     # _run checks every byte
    assert cp.issued[0] is not None and cp.issued[1] is None and len(cp.issued) == 2
    assert ops.piece_calls >= 1
    assert " collect_batch=OFF calls=1 " in cline and "why=issue_RuntimeError" in cline


def test_stream_zero_goes_per_piece(tmp_path, monkeypatch):
    """Desk fakes yield stream 0; the API forbids the legacy NULL stream."""
    monkeypatch.delenv(b1.ENV_COLLECT_BATCH, raising=False)

    class _Zero(_BatchOps):
        def create_stream(self, device):
            return 0

    cp = _Copier()
    ops = _Zero(cp)
    cline, _d, nb = _run(tmp_path, ops)
    assert cp.issued == [] and ops.piece_calls >= nb
    assert " collect_batch=OFF calls=0 why=stream_0" in cline


def test_collect_batch_copier_names_exceptions():
    class _Boom:
        name = "boom"

        def batch_copier(self, device):
            raise OSError("libcudart gone")

    cp, why = b1.collect_batch_copier(_Boom(), 0, {})
    assert cp is None and why == "batch_copier raised OSError: libcudart gone"



# -- review barlink-lanes2-1023: the canary at boot, not in the first collect ---

class _Lanes:
    def __init__(self, recv):
        self.recv = recv

    def _ordinal(self):
        return 0


def test_prewarm_runs_the_canary_once_at_boot(monkeypatch):
    monkeypatch.delenv(b1.ENV_COLLECT_BATCH, raising=False)
    monkeypatch.delenv(b1.ENV_COLLECT_PREPLAN, raising=False)
    made, logs = [], []

    def factory():
        o = _BatchOps(_Copier())
        made.append(o)
        return o

    line = b1.prewarm_collect_batch(_Lanes({"p0": object()}), factory, logs.append)
    assert len(made) == 1 and made[0].asked == [0]
    assert line == logs[0] and "collect-batch canary device=0 OK" in line and "['p0']" in line
    refused = b1.prewarm_collect_batch(
        _Lanes({"p1": object()}), lambda: _BatchOps(None, "canary byte mismatch"), logs.append)
    assert "REFUSED why=canary_byte_mismatch" in refused


@pytest.mark.parametrize("case", ["no_recv", "switch_off", "preplan_off"])
def test_prewarm_does_nothing_without_a_receiving_lane_or_switch(monkeypatch, case):
    monkeypatch.delenv(b1.ENV_COLLECT_BATCH, raising=False)
    monkeypatch.delenv(b1.ENV_COLLECT_PREPLAN, raising=False)
    lanes = _Lanes({"p0": object()})
    if case == "no_recv":
        lanes = _Lanes({})
    elif case == "switch_off":
        monkeypatch.setenv(b1.ENV_COLLECT_BATCH, "0")
    else:
        monkeypatch.setenv(b1.ENV_COLLECT_PREPLAN, "0")
    made = []
    assert b1.prewarm_collect_batch(lanes, lambda: made.append(1) or _BatchOps(_Copier()),
                                    lambda *_a: None) == ""
    assert made == []


def test_prewarm_never_raises():
    def boom():
        raise OSError("no libcudart")

    line = b1.prewarm_collect_batch(_Lanes({"p0": object()}), boom, lambda *_a: None)
    assert "canary n/a (OSError: no libcudart)" in line


def test_the_setup_thread_prewarms():
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[4]
           / "python/flliper/srt/managers/scheduler_components/weight_updater.py").read_text()
    assert "b1.prewarm_collect_batch(reg, log=logger.info)" in src
    assert 'target=_setup_then_canary' in src


def test_preplan_off_does_not_ask_for_a_copier(tmp_path, monkeypatch):
    monkeypatch.delenv(b1.ENV_COLLECT_BATCH, raising=False)
    monkeypatch.setenv(b1.ENV_COLLECT_PREPLAN, "0")
    ops = _BatchOps(_Copier())
    _run(tmp_path, ops)
    assert ops.asked == []
