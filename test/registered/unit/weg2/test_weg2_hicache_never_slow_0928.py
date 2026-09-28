# SPDX-License-Identifier: Apache-2.0
"""HICACHE-NEVER-SLOW (28.09.): the L3 write-behind yields to a queued load.

27B release-draft 13:51:00-21: ``WEG2-LOAD-DEVICE rid=weg2-1-17 tokens=70142
ms=17271 queue_ms=17016 read_ms=227`` -- the read took 0.2 s, the load waited
17 s while six ``L3-REUSE WRITE-BEHIND pass ... ms~1400`` ran (gate=open): an
awake P with work queued and no prefill for 21 s. User law: HiCache never
brakes the prefill; the write-behind is background work.

Guarded here:
* the gate is quiet while a store->host load is queued or in flight
  (``load``), open again when none is;
* the controller counts the prefetch queue, the aux thread's buffer and the
  operation in flight;
* a broken probe never closes the gate.
Hermetic.
"""
from __future__ import annotations

import os
import types
from queue import Queue

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.mem_cache import l3_write_behind as wb  # noqa: E402


@pytest.fixture(autouse=True)
def _clean():
    wb._reset_for_tests()
    yield
    wb._reset_for_tests()


def test_the_write_behind_yields_while_a_load_is_queued():
    q = {"n": 0}
    wb.register_load_probe(lambda: q["n"])
    assert wb.quiet_reason() is None
    q["n"] = 1
    assert wb.quiet_reason() == "load"
    q["n"] = 0
    assert wb.quiet_reason() is None


def test_the_controller_counts_queued_and_in_flight_loads():
    from sglang.srt.managers.cache_controller import HiCacheController

    c = types.SimpleNamespace(prefetch_queue=Queue(), prefetch_buffer=Queue(), _prefetch_io_current=None)
    assert HiCacheController.storage_loads_pending(c) == 0
    c.prefetch_buffer.put(object())
    c.prefetch_queue.put(object())
    c._prefetch_io_current = object()
    assert HiCacheController.storage_loads_pending(c) == 3


def test_a_broken_probe_never_closes_the_gate():
    def boom():
        raise RuntimeError("x")

    wb.register_load_probe(boom)
    assert wb.quiet_reason() is None


def test_a_dead_controller_drops_out_of_the_gate():
    class Ctl:
        def pending(self):
            return 1

    c = Ctl()
    wb.register_load_probe(c.pending)
    assert wb.quiet_reason() == "load"
    del c
    import gc

    gc.collect()
    assert wb.quiet_reason() is None
