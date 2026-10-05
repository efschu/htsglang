# SPDX-License-Identifier: Apache-2.0
"""1860: a FAILED pooled L15 park must end the BAR1 kernels the other ranks had
already queued, and must not let the group run another a2a.

The wedge (boot j2 05.10. 01:17Z): rank 2 raised in round 1 of the park (in/out
one data_ptr), ranks 0/1 had queued ~94 a2a kernels that wait for rank 2's flag;
the host raised ``Bar1CollectiveStalled`` after 60 s but nothing told the device:
every queued kernel spins its own device deadline (3e11 cycles, ~150 s) and the
flip's next stream sync (``move_indices``) waited for all of them (>10 min).
``Bar1CollectiveStalled`` does not set the host abort word.

Hermetic: no CUDA, no process group.  Park side: the real ``park_at_release`` /
``park_back_at_wake`` on three threads with a fake a2a and a fake transport that
counts ``abort_and_rearm``.  Transport side: the real ``abort_and_rearm`` against
a stub (the suite's unbound-method pattern) with a fake event/stream.

Run (own worktree, capped):
  cd /spinning/wt-27b-park-abort-1005 && /spinning/gpu-arb/pytest_gedeckelt.sh \
    test/registered/unit/weg2/test_weg2_l15_park_abort_1860.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.distributed.device_communicators import barlink_bar1, barlink_liveness
from sglang.srt.weg2 import l15_park as P

import test_weg2_l15_park_1002 as T  # _World, _bufs, _run_ranks
import test_weg2_l15_park_restamp_1005 as R  # _setup (real manifest files)


class _Stalled(RuntimeError):
    """Stands in for Bar1CollectiveStalled (raised by the host after the stall)."""


class _FakeTransport:
    def __init__(self):
        self.queued = 0           # kernels queued on the stream
        self.calls = 0            # abort_and_rearm calls
        self.reasons = []

    def abort_and_rearm(self, reason, drain_timeout_s=30.0):
        self.calls += 1
        self.reasons.append(reason)
        self.queued = 0           # the trip ends every queued kernel
        return "aborted, drained, rearmed"


@pytest.fixture(autouse=True)
def _reset_latch():
    P._A2A_DESYNC = None
    yield
    P._A2A_DESYNC = None


def _wire(monkeypatch, scheds, w, box, a2a_of, transports):
    monkeypatch.setattr(P, "_group_io", lambda sched: (
        sched.rank, 3, w.gather_for(sched.rank, box), a2a_of(sched.rank)))
    monkeypatch.setattr(P, "_group_transport", lambda sched: transports[sched.rank])


def _stall_a2a(transports, calls):
    """j2: rank 2 raises in round 1 and never arrives; ranks 0/1 queue kernels
    round after round (host async) until the host stall raise (one call is
    enough here: the park of this size has few rounds)."""

    def a2a_of(rank):
        def a2a(out, inp, osp, isp, rows=None):
            calls[rank] += 1
            if rank == 2:
                raise RuntimeError("barlink-bar1 a2a: in and out must not be the same")
            transports[rank].queued += 10      # rounds queued before the host raise
            raise _Stalled("collective stalled")
        return a2a
    return a2a_of


def test_failed_pooled_park_aborts_the_queued_kernels_and_latches_the_group(
        monkeypatch, tmp_path):
    bufs, w, env, scheds, paths = R._setup(monkeypatch, tmp_path, 12)   # POOL=1
    transports = [_FakeTransport() for _ in range(3)]
    calls = [0, 0, 0]
    _wire(monkeypatch, scheds, w, [None] * 3, _stall_a2a(transports, calls), transports)
    logs = []
    assert T._run_ranks(lambda r: P.park_at_release(scheds[r], env, logs.append)) == [None] * 3
    assert sum("result=FAILED" in x for x in logs) == 3
    # every rank ended its kernels (ranks 0/1 had them queued, rank 2 none)
    assert [t.calls for t in transports] == [1, 1, 1]
    assert [t.queued for t in transports] == [0, 0, 0]
    assert all("L15 park failed" in t.reasons[0] for t in transports)
    assert P.a2a_desync_reason() is not None
    assert sum("L15-PARK abort" in x for x in logs) == 3
    # no sidecar anywhere: the wake refills from L2
    assert not list(tmp_path.glob("weg2_l15_park.*"))

    # the group is latched: the next park (next flip) does not start a single a2a
    before = list(calls)
    logs2 = []
    assert T._run_ranks(lambda r: P.park_at_release(scheds[r], env, logs2.append)) == [None] * 3
    assert calls == before
    assert sum("desynced" in x for x in logs2) >= 1
    assert [t.calls for t in transports] == [1, 1, 1]


def test_not_pooled_branch_is_unchanged(monkeypatch, tmp_path):
    bufs, w, env, scheds, paths = R._setup(monkeypatch, tmp_path, 12)
    env.pop("SGLANG_WEG2_L15_POOL")                                    # per-card park
    transports = [_FakeTransport() for _ in range(3)]
    calls = [0, 0, 0]
    _wire(monkeypatch, scheds, w, [None] * 3, _stall_a2a(transports, calls), transports)
    logs = []
    assert T._run_ranks(lambda r: P.park_at_release(scheds[r], env, logs.append)) == [None] * 3
    assert [t.calls for t in transports] == [0, 0, 0]
    assert P.a2a_desync_reason() is None
    assert not any("L15-PARK abort" in x for x in logs)


def test_a_clean_refusal_does_not_latch(monkeypatch, tmp_path):
    """No manifest on one rank = nobody starts a collective = nothing to abort."""
    bufs, w, env, scheds, paths = R._setup(monkeypatch, tmp_path, 12)
    transports = [_FakeTransport() for _ in range(3)]
    calls = [0, 0, 0]
    _wire(monkeypatch, scheds, w, [None] * 3, _stall_a2a(transports, calls), transports)
    import os

    os.unlink(paths[1])
    logs = []
    assert T._run_ranks(lambda r: P.park_at_release(scheds[r], env, logs.append)) == [None] * 3
    assert calls == [0, 0, 0] and [t.calls for t in transports] == [0, 0, 0]
    assert P.a2a_desync_reason() is None


def _sleep_ok_then(monkeypatch, tmp_path, back_a2a_of):
    bufs, w, env, scheds, paths = R._setup(monkeypatch, tmp_path, 12)
    transports = [_FakeTransport() for _ in range(3)]
    box = [None] * 3
    logs = []
    _wire(monkeypatch, scheds, w, box, w.a2a_for, transports)
    assert T._run_ranks(lambda r: P.park_at_release(scheds[r], env, logs.append))[0] > 0
    for b in bufs[0]:
        b.zero_()                                                     # KV pause
    _wire(monkeypatch, scheds, w, box, back_a2a_of, transports)
    ep = int(R._read_file(paths[0]).epoch)
    back = T._run_ranks(lambda r: P.park_back_at_wake(
        scheds[r], env, logs.append, epoch=ep, group_ok=True))
    return back, transports, logs


def test_wake_transport_failure_aborts_and_latches(monkeypatch, tmp_path):
    def boom(rank):
        def a2a(out, inp, osp, isp, rows=None):
            raise _Stalled("collective stalled")
        return a2a

    back, transports, logs = _sleep_ok_then(monkeypatch, tmp_path, boom)
    assert back == [False] * 3
    assert [t.calls for t in transports] == [1, 1, 1]
    assert P.a2a_desync_reason() is not None


def test_wake_checksum_mismatch_is_not_a_transport_failure(monkeypatch, tmp_path):
    def silent(rank):                       # a2a "works" but brings nothing back
        return lambda out, inp, osp, isp, rows=None: None

    back, transports, logs = _sleep_ok_then(monkeypatch, tmp_path, silent)
    assert back == [False] * 3
    assert any("round-trip checksum" in x for x in logs)
    assert [t.calls for t in transports] == [0, 0, 0]
    assert P.a2a_desync_reason() is None


# -- the transport side: the real abort_and_rearm against a stub ---------------


class _Win:
    def __init__(self):
        self.tripped = False
        self.rearmed = 0

    def trip(self, reason):
        self.tripped = True

    def rearm(self):
        self.tripped = False
        self.rearmed += 1


def _stub(drained_after):
    ev = {"n": 0}

    class _Ev:
        def record(self, stream):
            pass

        def query(self):
            ev["n"] += 1
            return ev["n"] > drained_after

    stub = SimpleNamespace(
        _abort_window=_Win(), _ctl_dev=torch.tensor([1, 7], dtype=torch.int32),
        _ctl_inflight=True, _ctl_lag=5, _ctl_stall_run=3, _unchecked_launches=9,
        _deferred_launches=9, _abort_code_seen=1, device=None, group="tp:0", rank=0)
    return stub, _Ev


def test_abort_and_rearm_trips_drains_then_clears(monkeypatch):
    stub, Ev = _stub(drained_after=3)
    monkeypatch.setattr(torch.cuda, "Event", Ev)
    monkeypatch.setattr(torch.cuda, "current_stream",
                        lambda *a, **k: SimpleNamespace(synchronize=lambda: None))
    out = barlink_bar1.BarlinkBar1Transport.abort_and_rearm(stub, "x", drain_timeout_s=5.0)
    assert out == "aborted, drained, rearmed"
    w = stub._abort_window
    assert w.rearmed == 1 and not w.tripped
    assert stub._ctl_dev.tolist() == [0, 7]            # only the status word, not word 1
    assert (stub._ctl_inflight, stub._ctl_lag, stub._ctl_stall_run) == (False, 0, 0)
    assert (stub._unchecked_launches, stub._deferred_launches, stub._abort_code_seen) == (0, 0, 0)


def test_abort_and_rearm_leaves_the_word_tripped_when_the_stream_never_drains(monkeypatch):
    stub, Ev = _stub(drained_after=10 ** 9)
    monkeypatch.setattr(torch.cuda, "Event", Ev)
    monkeypatch.setattr(torch.cuda, "current_stream",
                        lambda *a, **k: SimpleNamespace(synchronize=lambda: None))
    out = barlink_bar1.BarlinkBar1Transport.abort_and_rearm(stub, "x", drain_timeout_s=0.05)
    assert out.startswith("drain timeout")
    assert stub._abort_window.tripped and stub._abort_window.rearmed == 0
    assert stub._ctl_dev.tolist() == [1, 7] and stub._abort_code_seen == 1


def test_abort_and_rearm_without_a_window_is_a_noop():
    stub, _ = _stub(0)
    stub._abort_window = None
    assert barlink_bar1.BarlinkBar1Transport.abort_and_rearm(stub, "x") == "no abort window"


def test_abort_window_rearm_clears_word_and_reason():
    w = barlink_liveness.AbortWindow.__new__(barlink_liveness.AbortWindow)
    w._buf = torch.zeros(16, dtype=torch.int32)
    w._tripped_reason = None
    w.trip("x")
    assert w.tripped and int(w._buf[0]) == 1
    w.rearm()
    assert not w.tripped and int(w._buf[0]) == 0
    w.trip("again")                                      # trips again after a rearm
    assert w.tripped and int(w._buf[0]) == 1
