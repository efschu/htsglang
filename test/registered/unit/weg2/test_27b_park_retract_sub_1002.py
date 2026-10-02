"""FLIP-EDGE 2 (02.10.): the park's ``retract`` lap, split into contiguous
named sub-laps (``WEG2-D-PARK RETRACT-SUB``).

N5d D log: retract_ms 0.07-0.76 s, linear in the parked sequence length and
not in the host bytes the write-through took (e18 seq 210263 -> 667 ms with
511 host slots; e24 seq 188367 -> 756 ms). The bare radix insert costs 2 ms
at 171k tokens on the CPU, so the term sits in a step around it, and the log
had no word for which. The instrument names every step; the laps are
contiguous, so their sum is the retract lap and nothing is left out.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import re
import types
from pathlib import Path

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import park_retract_laps as prl  # noqa: E402

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_park_timing_1001", _HERE / "test_27b_park_timing_1001.py")
pt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pt)

clk = pt.clk  # the PARK-TIMING fixture: one fake monotonic clock for the park


class _Fake:
    def __init__(self):
        self.t = 10.0

    def __call__(self):
        return self.t


def test_laps_are_contiguous_and_the_tail_is_named_rest(monkeypatch):
    c = _Fake()
    monkeypatch.setattr(prl, "_clock", c)
    monkeypatch.delenv("SGLANG_WEG2_PARK_RETRACT_LAPS", raising=False)
    prl.mark("ignored")  # unarmed: nothing recorded
    prl.arm()
    c.t += 0.002
    prl.mark("pre")
    c.t += 0.300
    prl.acc("wb", 0.120)
    prl.mark("insert")
    c.t += 0.050
    prl.mark("pre")  # a second request: the same step accumulates
    c.t += 0.010
    laps, nested = prl.disarm()
    assert list(laps) == ["pre", "insert", "rest"]
    assert laps["pre"] == pytest.approx(0.052)
    assert laps["insert"] == pytest.approx(0.300)
    assert laps["rest"] == pytest.approx(0.010)
    assert sum(laps.values()) == pytest.approx(c.t - 10.0)
    assert nested == {"wb": (pytest.approx(0.120), 1)}
    assert not prl.armed()
    prl.mark("after")
    assert prl.disarm() == ({}, {})
    line = prl.describe(laps, nested)
    assert "pre_ms=52.0" in line and "insert_ms=300.0" in line and "sum_ms=362.0" in line
    assert "wb_ms=120.0 wb_n=1" in line


def test_env_zero_never_arms(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_PARK_RETRACT_LAPS", "0")
    prl.arm()
    assert not prl.armed()
    prl.mark("x")
    assert prl.disarm() == ({}, {})


def test_park_prints_the_retract_split_whose_sum_is_the_retract_lap(clk, caplog, monkeypatch):
    from sglang.srt.managers.io_struct import Weg2ParkRunningReqInput

    monkeypatch.delenv("SGLANG_WEG2_PARK_RETRACT_LAPS", raising=False)
    monkeypatch.setattr(prl, "_clock", clk)
    steps = (("pre", 0.001), ("key", 0.040), ("insert", 0.200), ("publish", 0.300), ("reset", 0.002))

    def _retract_all(self, server_args, offload_kv=True, retain=False):
        assert prl.armed(), "the park arms the sub-laps around the retraction"
        for _req in self.reqs:
            for name, s in steps:
                self.tree.clk.adv(s)
                if name == "insert":
                    prl.acc("wb", 0.05)
                prl.mark(name)
        self.tree.clk.adv(0.003)  # after the last step: the lap "rest"
        out, self.reqs = self.reqs, []
        return out

    monkeypatch.setattr(pt._Batch, "retract_all", _retract_all)
    reqs = [pt._req("weg2-24-89", 132000, 694), pt._req("weg2-24-91", 55000, 673)]
    s = pt._Sched(reqs, clk, 0)
    with caplog.at_level(logging.INFO):
        out = pt.rt.park_running(s, Weg2ParkRunningReqInput(epoch=24, reason="immediate-over-x"))
    assert out.success
    assert not prl.armed(), "disarmed after the retraction"
    msgs = [r.getMessage() for r in caplog.records]
    sub = [m for m in msgs if "WEG2-D-PARK RETRACT-SUB" in m]
    timing = [m for m in msgs if "WEG2-D-PARK TIMING" in m]
    assert len(sub) == 1 and len(timing) == 1
    f = dict(re.findall(r"(\w+)=(\S+)", sub[0]))
    t = dict(re.findall(r"(\w+)=(\S+)", timing[0]))
    assert f["epoch"] == "24" and f["rank"] == "0" and f["seq_tokens"] == t["seq_tokens"]
    assert float(f["publish_ms"]) == pytest.approx(2 * 300.0)
    assert float(f["insert_ms"]) == pytest.approx(2 * 200.0)
    assert float(f["rest_ms"]) == pytest.approx(3.0)
    assert float(f["wb_ms"]) == pytest.approx(100.0) and f["wb_n"] == "2"
    # contiguous: the split adds up to the TIMING line's retract lap
    assert float(f["sum_ms"]) == pytest.approx(float(t["retract_ms"]), abs=0.2)


def test_no_running_request_prints_no_split(clk, caplog, monkeypatch):
    from sglang.srt.managers.io_struct import Weg2ParkRunningReqInput

    monkeypatch.setattr(prl, "_clock", clk)
    s = pt._Sched([], clk, 0)
    with caplog.at_level(logging.INFO):
        assert pt.rt.park_running(s, Weg2ParkRunningReqInput(epoch=2, reason="x")).success
    assert not [r for r in caplog.records if "RETRACT-SUB" in r.getMessage()]
    assert not prl.armed()


def test_cache_finished_req_marks_every_step_in_order():
    """The production marks sit in UnifiedRadixCache.cache_finished_req in
    step order (source check: the method needs a live pool to run)."""
    import inspect

    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    src = inspect.getsource(UnifiedRadixCache.cache_finished_req)
    names = re.findall(r'_prl\.mark\("(\w+)"\)', src)
    assert names == ["head", "turn", "commit", "ids", "prepare", "rows", "trim", "key", "insert",
                     "free", "anchor", "cap", "handoff", "publish", "release_free", "dec", "cleanup"]
    from sglang.srt.managers import schedule_batch as sb

    rel = re.findall(r'_prl\.mark\("(\w+)"\)', inspect.getsource(sb.release_req))
    assert rel == ["pre", "release_tail", "reset"]
