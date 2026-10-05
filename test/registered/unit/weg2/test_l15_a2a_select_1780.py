# SPDX-License-Identifier: Apache-2.0
"""1780/1800: the L15 park's all_to_all picks its transport from a RANK-UNIFORM
size (``select_bytes``), not from the rank-local send sum.

The wedge (boot i8l 2026-10-05, epoch 8): in a park chunk only the SOURCE rank
has bytes to send; the others have ``sum(send_bytes) == 0``, below barlink's
``a2a_min_bytes``.  ``BarlinkCommunicator.all_to_all_single`` chose BAR1 (+ a
gloo ``group_max``) on the source rank and the gloo ``all_to_all_single``
fallback on the others: two different gloo collectives that never pair
(120 s CollectiveTimeoutError on all three ranks).

Hermetic: no CUDA, no process group, no network.  The REAL chain runs on three
threads: ``l15_park.run_park`` / ``run_anchor_park`` -> ``_group_io.a2a`` ->
``GroupCoordinator.all_to_all_single_v`` -> ``BarlinkCommunicator.
all_to_all_single`` (real ``_select``); only the transport (BAR1) and the two
gloo calls (``_group_max``, ``dist.all_to_all_single``) are fakes that meet on
one shared ``threading.Barrier(timeout=2)`` and fail with BrokenBarrierError
when ranks enter DIFFERENT collective kinds (the wedge, in 2 s instead of 120).

Run (own worktree, capped):
  cd /spinning/wt-27b-l15-a2a-select-1005 && /spinning/gpu-arb/pytest_gedeckelt.sh \
    test/registered/unit/weg2/test_l15_a2a_select_1780.py
"""

from __future__ import annotations

import inspect
import threading
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.distributed import parallel_state as PS
from sglang.srt.distributed.device_communicators import barlink
from sglang.srt.weg2 import l15_park as P
from sglang.srt.weg2 import l15_pool_anchor as PA
from sglang.srt.weg2.l15_park import ParkPiece

import test_weg2_l15_park_1002 as T  # _World, _entries, _run_ranks

ROW = 1 << 18                       # row bytes: 256 KiB -> 4 rows per 1 MiB chunk
ENV = {"SGLANG_WEG2_L15_PARK_CHUNK_MIB": "1"}
N = 3


class _Sim:
    """Three ranks, one barrier; every collective entry is recorded per rank."""

    def __init__(self):
        self.barrier = threading.Barrier(N, timeout=2)
        self.posts = [None] * N
        self.proto = [[] for _ in range(N)]     # collective kinds entered, per rank
        self.sel = [[] for _ in range(N)]       # nbytes _select was asked, per rank
        self.tl = threading.local()

    def rendezvous(self, kind, payload):
        r = self.tl.rank
        self.proto[r].append(kind)
        self.posts[r] = (kind, payload)
        self.barrier.wait()
        snap = list(self.posts)
        if len({p[0] for p in snap}) > 1:       # ranks in different collectives
            self.barrier.abort()
            raise threading.BrokenBarrierError("kind mismatch %s" % [p[0] for p in snap])
        return snap

    def done(self):
        self.barrier.wait()

    def exchange(self, kind, out, inp, in_rows, out_rows):
        r = self.tl.rank
        snap = self.rendezvous(kind, (inp, list(in_rows)))
        off = 0
        for i in range(N):
            k = out_rows[i]
            src_inp, src_in_rows = snap[i][1]
            if k:
                soff = sum(src_in_rows[:r])
                out[off:off + k].copy_(src_inp[soff:soff + k])
            off += k
        self.done()


class _FakeBar1:
    """handles/supports/rounds like barlink_bar1 (a2a_min_bytes = 16)."""

    def __init__(self, sim):
        self.sim = sim

    def handles(self, op, nbytes):
        return op == "all_to_all" and nbytes >= 16

    def supports_a2a(self, largest):
        return True

    def a2a_rounds_for(self, largest):
        return 1

    def barlink_all_to_all_single(self, comm, output, inp, send_bytes, recv_bytes, rounds=None):
        # the real extension's TORCH_CHECK (barlink_bar1_ext.py, a2a): an rank that
        # is neither source nor destination of a piece used to hand in TWO empty
        # views at row 0 -> same data_ptr -> RuntimeError in round 1 (j2, 05.10.)
        if inp.data_ptr() == output.data_ptr():
            raise RuntimeError("barlink-bar1 a2a: in and out must not be the same")
        rb = int(output.shape[1]) * output.element_size()
        self.sim.exchange("bar1", output, inp, [b // rb for b in send_bytes],
                          [b // rb for b in recv_bytes])
        return output


@pytest.fixture
def sim(monkeypatch):
    s = _Sim()

    def fake_group_max(value, cpu_group, table=None):
        snap = s.rendezvous("group_max", int(value))
        s.done()
        return max(p[1] for p in snap)

    def fake_gloo_a2a(host_out, host_in, output_split_sizes=None, input_split_sizes=None,
                      group=None, async_op=False):
        s.exchange("gloo_a2a", host_out, host_in, input_split_sizes, output_split_sizes)

    real_empty = torch.empty

    def empty_nopin(*a, **k):
        k.pop("pin_memory", None)
        return real_empty(*a, **k)

    monkeypatch.setattr(barlink, "_group_max", fake_group_max)
    monkeypatch.setattr(barlink.dist, "all_to_all_single", fake_gloo_a2a)
    monkeypatch.setattr(barlink.barlink_liveness, "bounded_collective",
                        lambda issue, label, **kw: issue())
    monkeypatch.setattr(torch, "empty", empty_nopin)

    def rank_group(r):
        comm = barlink.BarlinkCommunicator.__new__(barlink.BarlinkCommunicator)
        comm.disabled = False
        comm.world_size = N
        comm.rank = r
        comm._closed = False
        comm._path_dispatcher = None
        comm.cpu_group = None
        comm._peer_table = None
        comm.transport = _FakeBar1(s)
        real_select = barlink.BarlinkCommunicator._select

        def select(op, nbytes):
            s.sel[r].append(int(nbytes))
            return real_select(comm, op, nbytes)

        comm._select = select
        g = PS.GroupCoordinator.__new__(PS.GroupCoordinator)
        g.world_size = N
        g.rank_in_group = r
        g.barlink_comm = comm
        g.cpu_group = None
        g._census_wire = False
        return g

    def get_tp_group():
        return s.tl.g

    monkeypatch.setattr("sglang.srt.distributed.get_tp_group", get_tp_group)
    monkeypatch.setattr(PS, "_CLOCK_A2A", False)
    s.rank_group = rank_group
    return s


def _run(s, fn):
    """fn(rank) on 3 threads; returns [(result, exception)] per rank."""
    out = [None] * N

    def body(r):
        s.tl.rank = r
        s.tl.g = s.rank_group(r)
        try:
            out[r] = (fn(r), None)
        except BaseException as e:  # noqa: BLE001
            out[r] = (None, e)

    th = [threading.Thread(target=body, args=(r,)) for r in range(N)]
    [t.start() for t in th]
    [t.join() for t in th]
    return out


def _bufs(fill_rank=None, rows=20, nbuf=2):
    out = []
    for layer in range(nbuf):
        b = torch.zeros(rows, ROW, dtype=torch.uint8)
        if fill_rank is not None:
            for r in range(rows):
                b[r] = (fill_rank * 17 + layer * 50 + r) % 251 + 1
        out.append(b)
    return out


def _ok(res):
    assert all(e is None for _, e in res), [repr(e) for _, e in res]


def _park(s, direction, pieces, bufs, uniform=True):
    def fn(r):
        _, _, _, a2a = P._group_io(None)
        return P.run_park(direction, pieces, r, N, bufs[r], a2a, ENV, uniform=uniform)

    return _run(s, fn)


# -- 1. the wedge, pinned on the gate-off path ---------------------------------------------------


def test_gate_off_rank_local_select_is_the_collective_mismatch(sim):
    bufs = {r: _bufs(r) for r in range(N)}
    res = _park(sim, "out", [ParkPiece(0, 1, 0, 0, 10)], bufs, uniform=False)
    # the gloo fallback wraps the barrier error (a RuntimeError) into NotImplementedError
    assert all(isinstance(e, threading.BrokenBarrierError)
               or isinstance(getattr(e, "__cause__", None), threading.BrokenBarrierError)
               for _, e in res), [repr(e) for _, e in res]
    assert sim.proto[0] == ["group_max"]                 # source rank: BAR1 branch (group_max first)
    assert sim.proto[1] == ["gloo_a2a"] and sim.proto[2] == ["gloo_a2a"]
    # the old figure is rank-local: n*row_bytes on the source, 0 elsewhere
    assert sim.sel[0] == [4 * ROW] and sim.sel[1] == [0] and sim.sel[2] == [0]


# -- 2./3./4. the fix: every rank takes the same path, data arrives ------------------------------


@pytest.mark.parametrize("src,dst", [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)])
def test_uniform_out_and_back_all_ranks_bar1_same_select(sim, src, dst):
    piece = ParkPiece(src, dst, 1, 3, 10)                # 10 rows -> chunks 4,4,2 per buffer
    bufs = {r: _bufs(r) for r in range(N)}
    src_orig = [b[1:11].clone() for b in bufs[src]]
    res = _park(sim, "out", [piece], bufs)
    _ok(res)
    expect = [4 * ROW, 4 * ROW, 2 * ROW] * 2
    for r in range(N):
        assert sim.proto[r] == ["bar1"] * 6, (r, sim.proto[r])
        assert sim.sel[r] == expect, (r, sim.sel[r])     # one figure, every rank
    for L in range(2):
        assert torch.equal(bufs[dst][L][3:13], src_orig[L])
    # wake: dst -> src (the way back), the source rows were lost meanwhile
    for b in bufs[src]:
        b[1:11] = 0
    sim.proto = [[] for _ in range(N)]
    sim.sel = [[] for _ in range(N)]
    sim.barrier = threading.Barrier(N, timeout=2)
    res = _park(sim, "back", [piece], bufs)
    _ok(res)
    for r in range(N):
        assert sim.proto[r] == ["bar1"] * 6 and sim.sel[r] == expect
    for L in range(2):
        assert torch.equal(bufs[src][L][1:11], src_orig[L])


def test_two_pieces_with_different_third_ranks(sim):
    pieces = [ParkPiece(0, 1, 0, 0, 5), ParkPiece(2, 1, 0, 5, 3)]
    bufs = {r: _bufs(r) for r in range(N)}
    o0 = [b[0:5].clone() for b in bufs[0]]
    o2 = [b[0:3].clone() for b in bufs[2]]
    res = _park(sim, "out", pieces, bufs)
    _ok(res)
    for r in range(N):
        # piece 0: 5 rows = chunks 4,1; piece 1: 3 rows = chunk 3; x 2 buffers
        assert sim.proto[r] == ["bar1"] * 6, (r, sim.proto[r])
        assert sim.sel[r] == [4 * ROW, 1 * ROW] * 2 + [3 * ROW] * 2
    for L in range(2):
        assert torch.equal(bufs[1][L][0:5], o0[L])
        assert torch.equal(bufs[1][L][5:8], o2[L])


# -- 5. the anchor park (S4) -----------------------------------------------------------------------


def _anchor_views(r):
    # mamba: 8 slots x 512 KiB; KV hold rows: 20 x 256 KiB
    m = [torch.zeros(8, 2 * ROW, dtype=torch.uint8)]
    kv = [torch.zeros(20, ROW, dtype=torch.uint8)]
    for s in range(8):
        m[0][s] = (r * 31 + s * 7) % 251 + 1
    return m, kv


@pytest.mark.parametrize("owner,host", [(0, 1), (2, 1), (1, 0)])
def test_anchor_park_out_and_back_uniform(sim, owner, host):
    n_a = 3
    nbytes = n_a * 2 * ROW                               # 1.5 MiB
    h_rows = -(-nbytes // ROW)                           # 6 -> chunks 4,2
    g = (owner, host, 0, n_a, 3, h_rows, nbytes)
    views = {r: _anchor_views(r) for r in range(N)}
    want = PA.gather_stream(views[owner][0], PA._slots_of(g)).clone()

    def go(direction):
        def fn(r):
            _, _, _, a2a = P._group_io(None)
            m, kv = views[r]
            return PA.run_anchor_park(direction, [g], r, N, m, kv, a2a, ENV, uniform=True)
        return _run(sim, fn)

    _ok(go("out"))
    for r in range(N):
        assert sim.proto[r] == ["bar1", "bar1"], (r, sim.proto[r])
        assert sim.sel[r] == [4 * ROW, 2 * ROW]
    got = views[host][1][0][3:3 + h_rows].reshape(-1)[:nbytes]
    assert torch.equal(got, want.reshape(-1))
    # wake
    views[owner][0][0][1:1 + n_a] = 0
    sim.proto = [[] for _ in range(N)]
    sim.sel = [[] for _ in range(N)]
    sim.barrier = threading.Barrier(N, timeout=2)
    _ok(go("back"))
    for r in range(N):
        assert sim.proto[r] == ["bar1", "bar1"] and sim.sel[r] == [4 * ROW, 2 * ROW]
    assert torch.equal(PA.gather_stream(views[owner][0], PA._slots_of(g)), want)


# -- 6. gate: default None keeps the old call byte for byte --------------------------------------


def test_signatures_default_select_bytes_none_and_other_callers_untouched():
    for fn in (barlink.BarlinkCommunicator.all_to_all_single, PS.GroupCoordinator.all_to_all_single_v):
        assert inspect.signature(fn).parameters["select_bytes"].default is None
    # the DCP merge caller passes no select_bytes
    from sglang.srt.layers.dcp import comm as dcp_comm

    assert "select_bytes" not in inspect.getsource(dcp_comm._a2a_v)


def test_group_io_a2a_forwards_rows_only_when_given(monkeypatch):
    calls = []

    class _G:
        world_size = 3
        rank_in_group = 0
        cpu_group = None

        def all_to_all_single_v(self, *a, **k):
            calls.append((a, k))

    monkeypatch.setattr("sglang.srt.distributed.get_tp_group", lambda: _G())
    _, _, _, a2a = P._group_io(None)
    out = torch.zeros(0, 8, dtype=torch.uint8)
    a2a(out, out, [0, 0, 0], [0, 0, 0])
    assert calls[-1][1] == {} and len(calls[-1][0]) == 4
    a2a(out, out, [0, 0, 0], [0, 0, 0], rows=3)
    assert calls[-1][1] == {"largest_block_rows": 3, "select_bytes": 24}


def test_group_coordinator_default_call_passes_no_new_kwargs(monkeypatch):
    seen = []

    class _C:
        def all_to_all_single(self, out, inp, osp, isp, **kw):
            seen.append(kw)
            return out

    g = PS.GroupCoordinator.__new__(PS.GroupCoordinator)
    g.world_size = 3
    g.barlink_comm = _C()
    g._census_wire = False
    monkeypatch.setattr(PS, "_CLOCK_A2A", False)
    x = torch.zeros(0, 4, dtype=torch.uint8)
    g.all_to_all_single_v(x, x, [0, 0, 0], [0, 0, 0])
    g.all_to_all_single_v(x, x, [0, 0, 0], [0, 0, 0], largest_block_rows=2)
    g.all_to_all_single_v(x, x, [0, 0, 0], [0, 0, 0], largest_block_rows=2, select_bytes=8)
    g.all_to_all_single_v(x, x, [0, 0, 0], [0, 0, 0], select_bytes=8)
    assert seen == [{}, {"largest_block_rows": 2},
                    {"largest_block_rows": 2, "select_bytes": 8}, {"select_bytes": 8}]


# -- 7. entries: the pooled park announces rows, the plain park does not ------------------------


def _record_rows(monkeypatch, rec):
    orig = P._group_io

    def wrapped(sched):
        r, w, gather, a2a = orig(sched)

        def a2a2(out, inp, osp, isp, rows=None):
            rec.append(rows)
            return a2a(out, inp, osp, isp, rows=rows)

        return r, w, gather, a2a2

    monkeypatch.setattr(P, "_group_io", wrapped)


def test_pooled_entries_pass_rows_plain_park_passes_none(monkeypatch, tmp_path):
    import test_weg2_l15_pool_s2_1004 as S2

    m = S2._manifest()
    for pooled in (True, False):
        bufs = {0: T._bufs(0, fill=3), 1: T._bufs(1), 2: T._bufs(2)}
        w, env, scheds = T._entries(monkeypatch, tmp_path, [m, m, m], bufs, [0, 15, 30])
        env.update(S2._pool_env(tmp_path, on=pooled))
        rec = []
        _record_rows(monkeypatch, rec)
        logs = []
        T._run_ranks(lambda r: P.park_at_release(scheds[r], env, logs.append))
        T._run_ranks(lambda r: P.park_back_at_wake(scheds[r], env, logs.append,
                                                   epoch=5, group_ok=True))
        assert rec, "the park moved rows"
        if pooled:
            assert all(x is not None and x > 0 for x in rec), rec
        else:
            assert all(x is None for x in rec), rec
