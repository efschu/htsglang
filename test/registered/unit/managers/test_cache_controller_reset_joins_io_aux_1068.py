"""#1068 WEG 1 slice 2 (WEG1_BUILD_SPEC_0901 section 4.2, graft G6):
``HiCacheController.reset()`` quiesces the storage pipeline through the ONE
join authority (``_stop_storage_threads`` / ``_start_storage_threads``).

THE DEFECT. ``reset()`` hand-rolled its own joins: ``prefetch_thread`` and
``backup_thread``, never ``prefetch_io_aux_thread`` -- the thread that
``prefetch_thread_func`` starts and that carries the actual page transfers.
``_stop_storage_threads`` (runtime detach, and #1025's cutover quiesce) joins
all three; #1025's own docstring names the half ``reset()`` "hand-rolls and
omits". A tree reset at the cutover therefore left the aux thread alive on
the OUTGOING binding while the pools swapped -- the state that killed it with
a StrayHostIndexError or a page-shape RuntimeError (boot 22, #1052).

Hermetic: a controller shell whose three threads are recording stubs; the
thread constructor of the controller module is replaced for the restart half
so no real thread runs. What is under test is WHICH threads are joined, the
stop-event bracket around them, and that the pipeline is started again.

    CUDA_VISIBLE_DEVICES='' python -m pytest \\
        test/registered/unit/managers/test_cache_controller_reset_joins_io_aux_1068.py -q
"""

import inspect
import threading
import types
import unittest
from queue import Queue
from unittest import mock

from sglang.srt.managers import cache_controller as cc_module
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _ThreadStub:
    """A live thread of the OUTGOING pipeline: records whether the stop event
    was already set when it was joined."""

    def __init__(self, name, event, log):
        self.name = name
        self._event = event
        self._log = log
        self._alive = True

    def join(self, timeout=None):
        self._log.append((self.name, self._event.is_set(), timeout))
        self._alive = False

    def is_alive(self):
        return self._alive


class _FakeThread:
    """Stands in for ``threading.Thread`` inside the controller module so the
    restart builds recordable objects instead of running the real loops."""

    started = []

    def __init__(self, target=None, daemon=None):
        self.target = target
        self.daemon = daemon
        self.started_flag = False

    def start(self):
        self.started_flag = True
        _FakeThread.started.append(self)

    def is_alive(self):
        return self.started_flag

    def join(self, timeout=None):
        pass


def _controller(stop_event, joins):
    cc = HiCacheController.__new__(HiCacheController)
    cc.enable_storage = True
    cc.storage_stop_event = stop_event
    cc.write_queue = [1]
    cc.load_queue = [1]
    cc.ack_write_queue = [1]
    cc.ack_load_queue = [1]
    cc.prefetch_queue = Queue()
    cc.backup_queue = Queue()
    cc.prefetch_buffer = Queue()
    cc.prefetch_revoke_queue = Queue()
    cc.ack_backup_queue = Queue()
    cc.host_mem_release_queue = Queue()
    cc.prefetch_thread = _ThreadStub("prefetch", stop_event, joins)
    cc.backup_thread = _ThreadStub("backup", stop_event, joins)
    cc.prefetch_io_aux_thread = _ThreadStub("prefetch_io_aux", stop_event, joins)
    cc.prefetch_tokens_occupied = 4096
    return cc


class TestResetUsesTheOneJoinAuthority(CustomTestCase):
    def setUp(self):
        _FakeThread.started = []

    def test_reset_joins_all_three_threads_and_restarts_the_pipeline(self):
        stop_event = threading.Event()
        joins = []
        cc = _controller(stop_event, joins)
        old_prefetch, old_backup = cc.prefetch_thread, cc.backup_thread
        fake_threading = types.SimpleNamespace(Thread=_FakeThread, Event=threading.Event)
        with mock.patch.object(cc_module, "threading", fake_threading):
            cc.reset()
        names = sorted(name for name, _, _ in joins)
        self.assertEqual(names, ["backup", "prefetch", "prefetch_io_aux"])
        # The stop event is SET while the threads are joined ...
        self.assertTrue(all(was_set for _, was_set, _ in joins), joins)
        # ... and CLEARED afterwards so the restart can arm a fresh pipeline.
        self.assertFalse(stop_event.is_set())
        # Fresh threads, started; the outgoing ones are gone.
        self.assertIsNot(cc.prefetch_thread, old_prefetch)
        self.assertIsNot(cc.backup_thread, old_backup)
        self.assertTrue(cc.prefetch_thread.started_flag)
        self.assertTrue(cc.backup_thread.started_flag)
        self.assertEqual(len(_FakeThread.started), 2)
        # The instrument counter is reset with the pipeline (N2 resolved:
        # _start_storage_threads does not touch it, so reset() must).
        self.assertEqual(cc.prefetch_tokens_occupied, 0)
        # The four non-storage queues are still emptied.
        for q in (cc.write_queue, cc.load_queue, cc.ack_write_queue, cc.ack_load_queue):
            self.assertEqual(q, [])

    def test_reset_without_storage_joins_nothing_and_starts_nothing(self):
        """Characterisation pin, green on the parent (228a66db32): the
        no-storage path of reset() was already correct and must stay
        byte-equivalent while the storage path moves onto the helpers."""
        stop_event = threading.Event()
        joins = []
        cc = _controller(stop_event, joins)
        cc.enable_storage = False
        fake_threading = types.SimpleNamespace(Thread=_FakeThread, Event=threading.Event)
        with mock.patch.object(cc_module, "threading", fake_threading):
            cc.reset()
        self.assertEqual(_FakeThread.started, [])
        self.assertFalse(stop_event.is_set())

    def test_reset_uses_the_helpers_not_hand_rolled_joins(self):
        src = inspect.getsource(HiCacheController.reset)
        self.assertIn("self._stop_storage_threads()", src)
        self.assertIn("self._start_storage_threads()", src)
        self.assertNotIn("prefetch_thread.join", src)
        self.assertNotIn("backup_thread.join", src)


if __name__ == "__main__":
    unittest.main()
