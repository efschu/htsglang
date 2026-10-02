"""DP-NACHLAUF 02.10.: WEG2-FIRST-FWD-TIMING -- device events around the first
forwards' per-layer waits on the load producer, and around the H2D issue.

Pinned (red before): armed at a wake, the next N batches are recorded (one
pair per layer, a repeated wait on the same layer is not re-recorded); the
H2D begin/end issued before the batch is attached to it; harvest never
blocks (an incomplete event keeps the record pending); the line carries the
wait sum, the worst layer, the H2D duration and the offsets; the switch off
records nothing; wiring: LayerDoneCounter routes waits through the
instrument only while a record is open, the kv resume arms it before
WAKE-PRELOAD, start_loading marks the load stream.
"""
from __future__ import annotations

import inspect
import logging
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import first_fwd_timing as fft  # noqa: E402


class _Clock:
    t = 0.0


class _Ev:
    def __init__(self, clock):
        self.clock, self.t, self.done = clock, None, True

    def record(self, stream=None):
        self.t = self.clock.t

    def query(self):
        return self.done

    def elapsed_time(self, other):
        return (other.t - self.t) * 1000.0


class _Loading:
    def __init__(self, clock, ready_at):
        self.clock, self.ready_at, self.waits = clock, ready_at, []

    def wait(self, layer):
        self.waits.append(layer)
        self.clock.t = max(self.clock.t, self.ready_at[layer])


def _setup(monkeypatch, on=True, n="2"):
    clk = _Clock()
    monkeypatch.setattr(fft, "S", fft._State())
    fft.S.event_factory = lambda: _Ev(clk)
    fft.S.available = lambda: True
    if on:
        monkeypatch.delenv(fft.ENV, raising=False)
    else:
        monkeypatch.setenv(fft.ENV, "0")
    monkeypatch.setenv(fft.ENV_N, n)
    return clk


def test_records_layers_h2d_and_logs(monkeypatch, caplog):
    clk = _setup(monkeypatch)
    fft.arm("kv_resume")
    clk.t = 0.000
    fft.on_load("load_stream", "begin")
    clk.t = 0.200
    fft.on_load("load_stream", "end")            # H2D 200 ms on the device
    clk.t = 0.050
    fft.on_set_consumer(4)
    lev = _Loading(clk, ready_at={0: 0.060, 1: 0.120, 2: 0.200})
    for layer in (0, 0, 1, 2):                    # K and V of layer 0 both wait
        fft.timed_wait(lev, layer)
        clk.t += 0.005
    assert sorted(fft.S.cur["layers"]) == [0, 1, 2] and lev.waits == [0, 0, 1, 2]
    with caplog.at_level(logging.INFO):
        fft.on_set_consumer(-1)                   # the next batch closes and harvests
    line = next(r.getMessage() for r in caplog.records if "WEG2-FIRST-FWD-TIMING" in r.getMessage())
    assert "consumer=4 layers=3" in line and "h2d_ms=200.0" in line
    assert "max_wait_ms=" in line and "first_wait_after_h2d_begin_ms=50.0" in line
    # 10 + 50 + 75 ms of stall (the repeated layer-0 wait adds no record)
    assert "load_wait_ms=135.0" in line and "max_wait_ms=75.0@L2" in line


def test_n_per_wake_and_pending_until_done(monkeypatch, caplog):
    clk = _setup(monkeypatch, n="1")
    fft.arm("kv_resume")
    fft.on_set_consumer(2)
    lev = _Loading(clk, ready_at={0: 0.0})
    fft.timed_wait(lev, 0)
    fft.S.cur["layers"][0][1].done = False        # the device has not reached it yet
    with caplog.at_level(logging.INFO):
        fft.on_set_consumer(-1)
    assert not [r for r in caplog.records if "WEG2-FIRST-FWD-TIMING" in r.getMessage()]
    assert len(fft.S.pending) == 1 and fft.S.cur is None   # N=1: the second batch is not recorded
    fft.S.pending[0]["layers"][0][1].done = True
    assert fft.harvest() == 1 and fft.S.pending == []


def test_switch_off_records_nothing(monkeypatch):
    clk = _setup(monkeypatch, on=False)
    fft.arm("kv_resume")
    fft.on_set_consumer(3)
    assert fft.S.cur is None and not fft.armed()
    lev = _Loading(clk, ready_at={0: 0.0})
    fft.timed_wait(lev, 0)
    assert lev.waits == [0]


def test_wiring():
    from sglang.srt.managers import cache_controller as cc
    from sglang.srt.managers.scheduler_components import weight_updater as wu
    from sglang.srt.mem_cache.hybrid_cache import hybrid_cache_controller as hcc

    wsrc = inspect.getsource(cc.LayerDoneCounter.wait_until)
    assert "if _fft.S.cur is not None:" in wsrc and "_fft.timed_wait(" in wsrc
    assert "_fft.on_set_consumer(index)" in inspect.getsource(cc.LayerDoneCounter.set_consumer)
    usrc = inspect.getsource(wu)
    assert usrc.index('_fft.arm("kv_resume")') < usrc.index("_wpl.run(self, l15_hold_aware=")
    ssrc = inspect.getsource(hcc.HybridCacheController.start_loading)
    assert ssrc.index('_fft.on_load(self.load_stream, "begin")') < ssrc.index('_fft.on_load(self.load_stream, "end")')
