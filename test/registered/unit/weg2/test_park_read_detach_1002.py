# SPDX-License-Identifier: Apache-2.0
"""PARK-READ-DETACH (02.10.2026): the sleep flush's reset no longer joins the
store probe of a PARKED request synchronously.

L15 boot dac8b62b8c (...10021440_dac8b62b8c_1002_144019.D.log) D->P 14:43:49:
drain+quiesce 1866 ms, D flush_cache 1820 ms, 'WEG2-SLEEP-SUB alloc_clear=1414'
= '#1068 RESET JOIN terminated_ops=1 joined_s=1.29' (TP0) / 0.94 (TP1): the
reset joined the running 133k-key probe of the parked weg2-6-6 (one
batch_exists_v2 over the span, no batch boundary inside). The read is
terminated as before; its thread is joined by a reaper that restarts the
pipeline. Switch SGLANG_WEG2_PARK_READ_DETACH. Hermetic, CPU, no store;
red_* red on 446d67df99.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import types
from queue import Queue

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=6, suite="stage-a-test-cpu")

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers import cache_controller as CC  # noqa: E402


class _Op:
    def __init__(self, rid):
        self.request_id = rid
        self._t = False

    def mark_terminate(self):
        self._t = True

    def is_terminated(self):
        return self._t


def _ctl(rid="weg2-6-6", probe_s=0.8, parked=("weg2-6-6",), aux_busy=False):
    """A controller whose prefetch thread sits in a probe that ignores the stop
    (as batch_exists_v2 over a 133k span does) for ``probe_s``."""
    c = CC.HiCacheController.__new__(CC.HiCacheController)
    c.enable_storage = True
    c.storage_stop_event = threading.Event()
    c.prefetch_queue, c.backup_queue, c.prefetch_buffer = Queue(), Queue(), Queue()
    c.host_mem_release_queue, c.prefetch_revoke_queue = Queue(), Queue()
    c.write_queue, c.load_queue, c.ack_write_queue, c.ack_load_queue = [], [], [], []
    c._prefetch_drained_after_stop = c._prefetch_io_drained_after_stop = 0
    c.prefetch_tokens_occupied = 0
    op = _Op(rid)
    c._prefetch_current = op
    c._prefetch_io_inflight = {id(op): op} if aux_busy else {}
    c._prefetch_io_current = op if aux_busy else None
    c._weg2_reset_detach_rids = frozenset(parked)
    c.starts = []
    t_end = time.monotonic() + probe_s

    def _probe():
        while time.monotonic() < t_end:   # the probe does not look at the stop
            time.sleep(0.01)
        c.host_mem_release_queue.put("stale-slots-of-the-cleared-pool")

    def _quick():
        c.storage_stop_event.wait()

    c.prefetch_thread = threading.Thread(target=_probe, daemon=True)
    c.backup_thread = threading.Thread(target=_quick, daemon=True)
    c.prefetch_io_aux_threads = [threading.Thread(target=_quick, daemon=True) for _ in range(2)]
    for t in [c.prefetch_thread, c.backup_thread] + c.prefetch_io_aux_threads:
        t.start()

    def _start():
        c.starts.append(time.monotonic())
        c.prefetch_thread = threading.Thread(target=lambda: None, daemon=True)
        c.prefetch_thread.start()

    c._start_storage_threads = _start
    return c, op


def test_red_the_reset_does_not_wait_for_the_parked_probe(caplog):
    c, op = _ctl()
    t0 = time.monotonic()
    with caplog.at_level(logging.INFO):
        c.reset()
        took = time.monotonic() - t0
        assert took < 0.4, took                              # base: ~0.8 s, the probe's length
        assert op.is_terminated()
        assert "detached=prefetch" in caplog.text
        assert c.weg2_reset_reaper_alive() and not c.starts  # restart waits for the old thread
        c._weg2_reset_reaper.join(3)
    assert len(c.starts) == 1 and not c.storage_stop_event.is_set()
    assert c.host_mem_release_queue.empty()                  # the old thread's release: discarded
    assert "PARK-READ-DETACH reaped" in caplog.text


def test_red_a_second_reset_merges_and_prefetch_waits_for_the_reaper(caplog):
    c, _ = _ctl()
    c.reset()
    with caplog.at_level(logging.INFO):
        c.reset()                                            # the release flush's reset
    assert "RESET MERGED" in caplog.text
    c.weg2_await_reset_reaper("prefetch")
    assert not c.weg2_reset_reaper_alive() and len(c.starts) == 1


def test_a_read_of_an_unparked_request_is_joined_as_before():
    c, _ = _ctl(parked=("someone-else",), probe_s=0.4)
    t0 = time.monotonic()
    c.reset()
    assert time.monotonic() - t0 >= 0.35 and len(c.starts) == 1
    assert getattr(c, "_weg2_reset_reaper", None) is None


def test_a_page_transfer_in_flight_is_joined_as_before():
    c, _ = _ctl(aux_busy=True, probe_s=0.4)
    t0 = time.monotonic()
    c.reset()
    assert time.monotonic() - t0 >= 0.35
    assert getattr(c, "_weg2_reset_reaper", None) is None


def test_switch_off_joins_as_before():
    c, _ = _ctl(probe_s=0.4)
    with envs.SGLANG_WEG2_PARK_READ_DETACH.override(False):
        t0 = time.monotonic()
        c.reset()
    assert time.monotonic() - t0 >= 0.35


def test_red_the_radix_drain_skips_while_the_old_pipeline_ends():
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    q = Queue()
    q.put("stale")
    cc = types.SimpleNamespace(weg2_reset_reaper_alive=lambda: True, prefetch_revoke_queue=q,
                               host_mem_release_queue=q, ack_backup_queue=q)
    ns = types.SimpleNamespace(cache_controller=cc, ongoing_prefetch={})
    UnifiedRadixCache._drain_storage_control_queues_impl(ns, None, None, None, None, False)
    assert q.qsize() == 1
