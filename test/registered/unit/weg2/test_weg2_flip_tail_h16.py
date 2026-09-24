# SPDX-License-Identifier: Apache-2.0
"""fnFL2 H16 -- der Heap-Zensus aus dem kritischen Pfad des Flips, und die
Lane-Zeile, die GIL/CPU-Stau von DMA-Zeit trennt.

DER BEFUND (Boot fnFL2x127, a1dab24409): der Sleep-RPC einer Phase endet mit
WEG2-SLEEP-HOST-HEAP, dessen gc-Walk auf den Sleeps 1, 2, 4, 8 im RPC laeuft.
Die Front startet den Wake erst nach der RPC-Antwort. Schwanz letzter
Deposit-Tag -> GROUP-FENCE: P PP0 538/505 ms MIT Walk (Sleep 2, 4), 18/23 ms
ohne (Sleep 3, 5); D TP0 616/639/558 ms mit, 57 ms ohne. Der Walk laeuft jetzt
per Default hinter der Antwort (Timer), seine Zeile traegt census_ms.

Die Lane-Zeile: x127 PP0 weights_9 lief mit wait_ms 2-5 und copy_sync_ms
404-410 -- kein Kredit-Warten, aber copy_sync_ms kann eine laufende DMA nicht
von einem Thread unterscheiden, der die GIL/CPU nach dem Sync spaet
zurueckbekommt. sync_cpu_ms tut das (Spin-Sync: CPU ~ Wand; geparkt: CPU << Wand).
"""

from __future__ import annotations

import ctypes
import os
import threading
import time
from types import SimpleNamespace

from sglang.srt.environ import Weg2HeapCensus, envs
from sglang.srt.weg2 import bar1_lanes as b1
from sglang.srt.weg2 import heap_census as hc
from sglang.srt.weg2 import weight_exchange_transport as tp
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PAIRS = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))


# -- the census: cadence, mode, and that the default never walks inline -------


def test_census_cadence_is_the_xsn297_one():
    due = [n for n in range(0, 20) if hc.census_due(sleep_count=n)]
    assert due == [1, 2, 4, 8, 16]


def test_default_mode_is_off_and_every_mode_respects_the_cadence():
    # x131/x132: DEFER lands in D's post-wake extend (+0.2-0.3 s per census
    # flip), INLINE is 500 ms on P's tail once the extend shrinks -- OFF.
    assert envs.SGLANG_WEG2_SLEEP_HEAP_CENSUS.get() == Weg2HeapCensus.OFF
    assert hc.census_action(mode=Weg2HeapCensus.DEFER, sleep_count=2) == "defer"
    assert hc.census_action(mode=Weg2HeapCensus.INLINE, sleep_count=4) == "inline"
    assert hc.census_action(mode=Weg2HeapCensus.OFF, sleep_count=2) == "off"
    # a sleep off the cadence never walks, whatever the mode (x127 sleeps 3, 5)
    for mode in Weg2HeapCensus:
        assert hc.census_action(mode=mode, sleep_count=3) == "off"


class _FakeCensus:
    """heap_census stand-in: records whether the walk ran synchronously."""

    DEFER_S = hc.DEFER_S

    def __init__(self):
        self.walked = 0
        self.deferred = []

    def timed_census(self):
        self.walked += 1
        return "dict:1x0MiB", 3.0

    def defer_census(self, *, sleep_count, log):
        self.deferred.append(sleep_count)


def test_the_sleep_path_does_not_walk_the_heap_under_the_default():
    """x127: the inline walk cost 505-639 ms inside the sleep RPC. Under DEFER
    the sleep path only arms the timer; the walk itself is not called."""
    from sglang.srt.managers.scheduler_components.weight_updater import (
        SchedulerWeightUpdaterManager as M,
    )

    fake = _FakeCensus()
    suffix = M._weg2_census_now(heap_census=fake, action="defer", n=2)
    assert fake.walked == 0 and fake.deferred == [2]
    assert "deferred" in suffix
    suffix = M._weg2_census_now(heap_census=fake, action="inline", n=4)
    assert fake.walked == 1 and "census_ms=3" in suffix
    assert M._weg2_census_now(heap_census=fake, action="off", n=3) == ""
    assert fake.walked == 1 and fake.deferred == [2]


class _Timer:
    def __init__(self, delay, fn):
        self.delay, self.fn, self.daemon, self.started = delay, fn, False, False

    def start(self):
        self.started = True


def test_deferred_census_is_a_daemon_timer_whose_line_carries_its_cost():
    lines = []
    t = hc.defer_census(
        sleep_count=4,
        log=lines.append,
        timer_factory=_Timer,
        objects=lambda: [{}, {}, [], "x"],
    )
    assert t.started and t.daemon and t.delay == hc.DEFER_S
    assert lines == []  # nothing ran at arming time
    t.fn()
    assert len(lines) == 1
    line = lines[0]
    assert line.startswith("WEG2-SLEEP-HOST-HEAP-CENSUS sleep=4 ")
    assert "census_ms=" in line and "dict:2x" in line


def test_a_failing_walk_logs_na_and_never_raises():
    lines = []

    def _boom():
        raise MemoryError("walk")

    t = hc.defer_census(
        sleep_count=1, log=lines.append, timer_factory=_Timer, objects=_boom
    )
    t.fn()
    assert "census=n/a(MemoryError)" in lines[0]


# -- the lane line: sync CPU separates a parked thread from a running DMA ------


class _SlowSyncOps:
    """host memcpy; synchronize either sleeps (a thread parked: no CPU) or
    spins (a spinning cudaStreamSynchronize while the DMA runs: CPU = wall)."""

    def __init__(self, sync_s, spin):
        self.sync_s, self.spin = sync_s, spin

    def create_stream(self, device):
        return 0

    def synchronize(self, stream):
        if self.spin:
            end = time.perf_counter() + self.sync_s
            while time.perf_counter() < end:
                pass
        else:
            time.sleep(self.sync_s)

    def memcpy_async(self, dst, src, n, stream):
        ctypes.memmove(dst, src, n)

    def memcpy2d_async(self, dst, dpitch, src, spitch, width, height, stream):
        for r in range(height):
            ctypes.memmove(dst + r * dpitch, src + r * spitch, width)


def _addr(buf):
    return ctypes.addressof(ctypes.c_char.from_buffer(buf))


def _run_pair(tmp_path, ops):
    slot, ring = 4096, 4
    window = bytearray(slot * ring)
    dep = b1.Bar1Lanes("n1", "P", 0, 0, PAIRS, log=lambda *_a: None, root=str(tmp_path))
    col = b1.Bar1Lanes("n1", "D", 1, 0, PAIRS, log=lambda *_a: None, root=str(tmp_path))
    col.recv["p0"] = SimpleNamespace(dptr=_addr(window), slot_bytes=slot, ring=ring)
    dep.peers["p0"] = SimpleNamespace(dev_ptr=_addr(window), slot_bytes=slot, ring=ring)
    src = bytearray(os.urandom(slot * 6))
    dst = bytearray(slot * 6)
    descs = [
        SimpleNamespace(
            kind=tp.FLAT,
            nbytes=len(src),
            src_off=0,
            dst_off=0,
            param_name="w",
            tag="t0",
            src_ptr=_addr(src),
            dst_ptr=_addr(dst),
        )
    ]
    lines, out = [], {}

    def _collect():
        out["c"] = b1.run_bar1_units(
            descs,
            _SlowSyncOps(0.0, False),
            lanes=col,
            lane_key="p0",
            role="dst",
            seq=1,
            phase="collect",
            budget_s=5.0,
            log=lambda *_a: None,
        )

    th = threading.Thread(target=_collect)
    th.start()
    out["d"] = b1.run_bar1_units(
        descs,
        ops,
        lanes=dep,
        lane_key="p0",
        role="src",
        seq=1,
        phase="deposit",
        budget_s=5.0,
        log=lines.append,
    )
    th.join(10)
    assert out == {"d": "", "c": ""} and bytes(dst) == bytes(src)
    line = [ln for ln in lines if "lane-time" in ln][0]
    return dict(kv.split("=", 1) for kv in line.split() if "=" in kv)


def test_lane_line_splits_credit_issue_sync_and_sync_cpu(tmp_path):
    parked = _run_pair(tmp_path, _SlowSyncOps(0.02, spin=False))
    # the old fields stay, the new split rides beside them
    assert parked["wait_ms"] == parked["credit_ms"]
    assert parked["copy_sync_ms"] == parked["sync_ms"]
    assert {"issue_ms", "send_ms", "max_sync_ms", "sync_cpu_ms"} <= set(parked)
    # 6 batches x 20 ms: the wall is in the syncs, the CPU is not
    assert float(parked["sync_ms"]) >= 100
    assert float(parked["sync_cpu_ms"]) < 0.5 * float(parked["sync_ms"])
    assert float(parked["max_sync_ms"]) >= 19


def test_a_spinning_sync_reads_as_cpu_time(tmp_path):
    spun = _run_pair(tmp_path, _SlowSyncOps(0.02, spin=True))
    assert float(spun["sync_cpu_ms"]) >= 0.7 * float(spun["sync_ms"])
