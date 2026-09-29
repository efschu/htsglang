# SPDX-License-Identifier: Apache-2.0
"""BOOTZEIT 3 Stufe 1 (29.09.): the per-layer presplit on its own serial thread.

rc12z30o3: the presplit summed to PP0 26.8 of 60 s and TP0 33.4 of 62.8 s on
the loader thread, and reading/consuming stood still while it ran. With
SGLANG_LOAD_PRESPLIT_THREAD=1 it runs on ONE thread that mirrors the loader's
TMS thread-local config; off, the fnFL2x31 behaviour stays byte-identical.
"""

import threading
import time
from unittest import mock

import pytest
import torch

from sglang.srt.managers import weg2_memory_saver as wms
from sglang.srt.model_loader import load_consumer as lc


def _stub_layer(param, fired):
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    class Stub:
        _ct_stream_note = FusedMoE._ct_stream_note

        def __init__(self):
            self._ct_stream_presplit = {
                "done": False, "lock": threading.Lock(),
                "names": {id(param): "w13"}, "expected": {"w13": 2}, "seen": {},
            }

        def _ct_stream_presplit_now(self, state):
            fired.append(threading.get_ident())

    return Stub()


def test_switch_is_off_by_default():
    assert lc.presplit_thread_enabled() is False
    pool = lc.ExpertLoadPool(2)
    assert pool.presplit_mode == "loader"
    pool.close()


def test_presplit_runs_on_its_own_thread_when_switched_on():
    param = torch.zeros(1)
    fired = []
    layer = _stub_layer(param, fired)
    with lc.ExpertLoadPool(2, presplit_thread=True) as pool:
        assert pool.presplit_mode == "thread"
        pool.submit(layer._ct_stream_note, param)
        pool.submit(layer._ct_stream_note, param)
    # drain() on exit waited for it; it ran exactly once, NOT on the loader
    # thread and not on a consumer
    assert len(fired) == 1
    assert fired[0] != threading.get_ident()
    assert pool.presplit_busy_s >= 0.0


def test_off_keeps_the_presplit_on_the_loader_thread():
    param = torch.zeros(1)
    fired = []
    layer = _stub_layer(param, fired)
    with lc.ExpertLoadPool(2, presplit_thread=False):
        pass
    with lc.ExpertLoadPool(2, presplit_thread=False) as pool:
        pool.submit(layer._ct_stream_note, param)
        pool.submit(layer._ct_stream_note, param)
    assert fired == [threading.get_ident()]


def test_serial_in_order_and_drain_waits():
    order = []
    active = []
    w = lc.PresplitWorker(device_index=None, region=None, depth=4)

    def job(i):
        def run():
            active.append(i)
            assert len(active) == 1, "presplits must never overlap"
            time.sleep(0.01)
            order.append(i)
            active.remove(i)
        return run

    for i in range(6):
        w.submit(job(i))
    w.drain()
    w.close()
    assert order == list(range(6))
    assert w.ran == 6


def test_depth_bounds_the_queue_and_blocks_the_submitter():
    gate = threading.Event()
    w = lc.PresplitWorker(device_index=None, region=None, depth=1)
    w.submit(gate.wait)            # running
    w.submit(lambda: None)         # queued (depth 1)
    threading.Timer(0.15, gate.set).start()
    w.submit(lambda: None)         # blocks until the first one finishes
    w.drain()
    w.close()
    assert w.submit_wait_s >= 0.1


def test_first_failure_resurfaces_and_refuses_further_work():
    ran = []
    w = lc.PresplitWorker(device_index=None, region=None, depth=2)
    w.submit(lambda: (_ for _ in ()).throw(ValueError("repack broke")))
    w.submit(lambda: ran.append(1))
    with pytest.raises(RuntimeError, match="PRESPLIT-THREAD failed"):
        w.drain()
    with pytest.raises(RuntimeError):
        w.submit(lambda: None)
    w.close()
    assert ran == [], "nothing may run after a failed presplit"


class _FakeCdll:
    def __init__(self, interesting=False, tag=b"default"):
        self.interesting = interesting
        self.tag = tag
        self.backup = False

    def tms_get_interesting_region(self):
        return self.interesting

    def tms_set_interesting_region(self, v):
        self.interesting = bool(v)

    def tms_set_current_tag(self, t):
        self.tag = t

    def tms_get_enable_cpu_backup(self):
        return self.backup

    def tms_set_enable_cpu_backup(self, v):
        self.backup = bool(v)


def test_mirror_sets_and_restores_the_thread_local_config():
    cdll = _FakeCdll()
    with wms.mirrored_weights_region((cdll, "weights", True)) as on:
        assert on is True
        assert cdll.interesting is True and cdll.tag == b"weights"
        assert cdll.backup is True
    assert cdll.interesting is False and cdll.tag == b"default"
    with wms.mirrored_weights_region(None) as on:
        assert on is False


def test_mirror_refuses_a_thread_that_already_has_a_region():
    with pytest.raises(RuntimeError):
        with wms.mirrored_weights_region((_FakeCdll(interesting=True), "weights", False)):
            pass


def test_capture_only_for_the_banded_base_region():
    cdll = _FakeCdll(interesting=True)
    with mock.patch.object(wms, "_tms_cdll_in_region", return_value=cdll):
        with mock.patch.object(wms, "weight_chunk_geometry", return_value=(4, 12)):
            assert wms.capture_weights_region_for_thread() == (cdll, "weights", False)
            with wms.weights_region_tag(wms.GPU_MEMORY_TYPE_WEIGHTS_DRAFT):
                assert wms.capture_weights_region_for_thread() is None
        with mock.patch.object(wms, "weight_chunk_geometry", return_value=(0, 0)):
            assert wms.capture_weights_region_for_thread() is None
    with mock.patch.object(wms, "_tms_cdll_in_region", return_value=None):
        assert wms.capture_weights_region_for_thread() is None


def test_unmirrorable_region_keeps_the_presplit_on_the_loader():
    with mock.patch.object(lc, "_tms_in_use", return_value=True), \
            mock.patch.object(wms, "capture_weights_region_for_thread", return_value=None):
        pool = lc.ExpertLoadPool(2, presplit_thread=True)
    assert pool.presplit_mode.startswith("loader")
    pool.close()


def test_the_worker_thread_carries_the_mirrored_config():
    cdll = _FakeCdll()
    seen = []
    w = lc.PresplitWorker(device_index=None, region=(cdll, "weights", False), depth=1)
    w.submit(lambda: seen.append((cdll.interesting, cdll.tag)))
    w.drain()
    w.close()
    assert seen == [(True, b"weights")]
    assert cdll.interesting is False
