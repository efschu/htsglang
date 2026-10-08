"""FEHLT 7 (Dashboard-Plausi-Audit 30.09., Zeile 4): rankstats.prefill without
the wait behind the predecessor.

Befund: ``compute_ms`` holds, on PP0, the wait behind the previous chunk -- a
67-token chunk measured 1876 ms and ended 0.13 s after the 16k chunk before it;
its event window opened while the predecessor still ran. The dashboard's
"Rang-Zeit-Rate" was therefore only a lower bound.

Now the prefill timer hands each duration the device time since the PREVIOUS
interval's end event; ``own = min(t, since_prev_end)`` is the chunk's own span,
``compute_only = own - collective wait`` (split known). ``prefill.last`` carries
``own_ms``, ``behind_prev_ms``, ``compute_only_ms``; the cumulative block
``own_ms``, ``compute_only_ms``, ``own_n``.
"""
from __future__ import annotations

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.managers.scheduler_components import metrics_reporter as M  # noqa: E402
from flliper.srt.pdflip import rankstats  # noqa: E402


class _Timer:
    def _report(self):
        pass


class _Clock:
    def __init__(self, wait_s):
        self.wait_s = wait_s

    def harvest_detail(self, slot):
        return types.SimpleNamespace(total_s=self.wait_s, families={})


def _log(wait_s=None):
    log = M.RankPrefillLog()
    log.timer = _Timer()
    if wait_s is not None:
        log.clock = _Clock(wait_s)
    return log


def _slot():
    return types.SimpleNamespace(graph_capture_skipped=False)


def test_pp0_the_67_token_chunk_behind_the_16k_one():
    log = _log()
    log.record(16384, 0, timed=True)
    log._on_duration(t=2.9)                                  # the rank's first chunk: no predecessor
    log.flush()
    assert log.cum["last"]["own_ms"] is None and log.cum["own_n"] == 0
    log.record(67, 0, timed=True)
    log._on_duration(t=1.876, since_prev_end=0.13)           # ended 0.13 s after the 16k chunk
    log.flush()
    last = log.cum["last"]
    assert last["gpu_ms"] == 1876.0 and last["own_ms"] == 130.0 and last["behind_prev_ms"] == 1746.0
    assert last["compute_only_ms"] is None                   # split unknown (no collective clock)
    assert log.cum["own_n"] == 1 and round(log.cum["own_ms"], 1) == 130.0


def test_compute_only_subtracts_the_collective_wait():
    log = _log(wait_s=0.02)
    log.record(512, 0, timed=True)
    log._on_duration(t=0.2, collective_slot=_slot(), since_prev_end=0.5)   # started after prev end
    log.flush()
    last = log.cum["last"]
    assert last["own_ms"] == 200.0 and last["behind_prev_ms"] == 0.0
    assert last["compute_only_ms"] == 180.0 and round(log.cum["compute_only_ms"], 1) == 180.0


def test_the_timer_hands_the_span_since_the_previous_end():
    got = []

    class _Ev:
        def __init__(self, t):
            self.t = t

        def query(self):
            return True

        def elapsed_time(self, other):
            return (other.t - self.t) * 1000.0

    def iv(start, end):
        return types.SimpleNamespace(start_event=_Ev(start), end_event=_Ev(end), metadata={},
                                     elapsed_time=lambda: (end - start) * 1000.0)

    timer = M.MissWindowTimer(reporter=lambda **kw: got.append(kw), clock=None)
    timer._intervals.extend([iv(0.0, 2.9), iv(1.02, 3.03)])
    timer._report()
    assert got[0]["since_prev_end"] is None
    assert abs(got[1]["t"] - 2.01) < 1e-9 and abs(got[1]["since_prev_end"] - 0.13) < 1e-9


def test_rankstats_carries_and_rounds_them():
    log = _log()
    log.record(67, 0, timed=True)
    log._on_duration(t=1.87612, since_prev_end=0.130004)
    log.flush()
    block = rankstats._prefill_block(types.SimpleNamespace(rank_prefill_log=log))
    assert block["own_n"] == 1 and block["own_ms"] == 130.0 and block["last"]["own_ms"] == 130.0
