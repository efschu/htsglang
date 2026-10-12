"""NF D-HANG 1011: the wedge watchdog writes the scheduler thread's stack on every rank, and the
rank that ran ahead gets its own FORWARD-HANG signal.

SPECIMEN (NF int24d, 11.10. 10:25:45Z, group D): TP1/TP2 logged "ADMISSION-WEDGE: 1 queued, 0
running" + "CLASS=A-PIPELINE-DEAD" and wrote no stack, so the line they stood in is unknown. TP0 had
an empty queue and stood in the extend forward: `queued > 0 and running == 0` cannot fire there, it
logged nothing at all (``test_tp0_shape_...`` reproduces that blind spot and its closure).

Hermetic: no CUDA, no scheduler boot; the "scheduler thread" is a real thread parked on an Event so
the dump is read off a live frame.
"""

import logging
import os
import shutil
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from flliper.srt.managers.scheduler_components import invariant_checker as IC
from flliper.srt.managers.scheduler_components.invariant_checker import (
    FORWARD_HANG,
    FORWARD_HANG_SECONDS,
    WEDGE_STACK,
    WEDGE_STACK_MAX_PER_PROCESS,
    check_admission_wedge_once,
    check_forward_hang_once,
    make_admission_wedge_poller,
    write_scheduler_stack,
)
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def setUpModule():  # the poller publishes its verdict file: keep it out of /run
    global _TMP
    _TMP = tempfile.mkdtemp(prefix="wedge1011_")
    os.environ["FLLIPER_WEDGE_STATUS_DIR"] = _TMP


def tearDownModule():
    os.environ.pop("FLLIPER_WEDGE_STATUS_DIR", None)
    shutil.rmtree(_TMP, ignore_errors=True)


class _SchedulerThread:
    """A live thread that stands in a recognisable function, like a rank waiting in a collective."""

    def __init__(self):
        self.release = threading.Event()
        self.started = threading.Event()
        self.thread = threading.Thread(target=self._waiting_for_peers_in_group_min, name="sched")
        self.thread.start()
        self.started.wait(5)

    def _waiting_for_peers_in_group_min(self):
        self.started.set()
        self.release.wait(30)

    def stop(self):
        self.release.set()
        self.thread.join(5)

    @property
    def ident(self):
        return self.thread.ident


class _Sched:
    """Only what the poller / forward-hang check read."""

    def __init__(self, *, queued=0, running=0, forward_ct=100, batch=object(), age=0.0, ident=None):
        self.is_initializing = False
        self.waiting_queue = [object()] * queued
        self.running_batch = SimpleNamespace(reqs=[object()] * running)
        self.last_first_token_progress_time = time.perf_counter() - age
        self.forward_ct = forward_ct
        self.cur_batch_for_debug = batch
        self.ps = SimpleNamespace(pp_rank=0, tp_rank=0, tp_size=3)
        self.server_args = SimpleNamespace(enable_phase_flip=False)
        self.session_controller = SimpleNamespace(maybe_reap=lambda now: None)
        self.flush_wrapper = SimpleNamespace(check_pending=lambda: None)
        self.external_corpus_manager = None
        self.return_health_check_ipcs = []
        if ident is not None:
            self._wedge_sched_thread_ident = ident


class WriteSchedulerStackTest(CustomTestCase):
    def test_stack_names_the_waiting_frame_with_file_and_line_on_the_scheduler_thread_only(self):
        st = _SchedulerThread()
        try:
            s = _Sched(ident=st.ident)
            with self.assertLogs(IC.logger, level="ERROR") as cm:
                self.assertTrue(write_scheduler_stack(s, "first-alarm"))
            text = "\n".join(cm.output)
            self.assertIn(f"{WEDGE_STACK} rank=pp0-tp0 why=first-alarm", text)
            self.assertIn("_waiting_for_peers_in_group_min", text)
            self.assertIn("test_wedge_stack_forward_hang_1011.py", text)  # file:line, not just a name
            self.assertIn("line ", text)
            # only the scheduler thread: the test's own (main) thread is not in the dump
            self.assertNotIn("test_stack_names_the_waiting_frame", text)
        finally:
            st.stop()

    def test_once_per_reason_and_bounded_per_process(self):
        st = _SchedulerThread()
        try:
            s = _Sched(ident=st.ident)
            with self.assertLogs(IC.logger, level="ERROR"):
                self.assertTrue(write_scheduler_stack(s, "first-alarm"))
            self.assertFalse(write_scheduler_stack(s, "first-alarm"))  # same reason, same wedge
            with self.assertLogs(IC.logger, level="ERROR"):
                self.assertTrue(write_scheduler_stack(s, "class-A"))
            wrote = 2
            for i in range(WEDGE_STACK_MAX_PER_PROCESS + 5):
                with self.assertLogs(IC.logger, level="ERROR") as cm:
                    logging.getLogger(IC.logger.name).error("sentinel")
                    ok = write_scheduler_stack(s, f"r{i}")
                wrote += 1 if ok else 0
            self.assertEqual(wrote, WEDGE_STACK_MAX_PER_PROCESS)
        finally:
            st.stop()

    def test_a_missing_thread_degrades_to_a_header_and_never_raises(self):
        s = _Sched(ident=-1)
        with self.assertLogs(IC.logger, level="ERROR") as cm:
            self.assertTrue(write_scheduler_stack(s, "first-alarm"))
        self.assertIn(WEDGE_STACK, "\n".join(cm.output))


class PollerStackTest(CustomTestCase):
    def _poller(self, s):
        # no recovery driver traffic: the post/settle machinery is not under test here
        return make_admission_wedge_poller(s)

    def test_first_alarm_dumps_once_class_a_and_unrecovered_add_one_each_and_clear_rearms(self):
        st = _SchedulerThread()
        try:
            # TP1/TP2 shape: 1 queued, nothing running, no first token for far longer than the threshold
            s = _Sched(queued=1, running=0, age=500.0, ident=st.ident)
            poll = self._poller(s)
            with self.assertLogs(IC.logger, level="ERROR") as cm:
                self.assertTrue(poll())
            first = [m for m in cm.output if WEDGE_STACK in m]
            self.assertEqual(len(first), 1)
            self.assertIn("why=first-alarm", first[0])
            self.assertIn("_waiting_for_peers_in_group_min", first[0])
            # the same wedge, next poll: no second first-alarm dump
            with self.assertLogs(IC.logger, level="ERROR") as cm:
                poll()
            self.assertEqual([m for m in cm.output if "why=first-alarm" in m], [])
            # the wedge is classed A-PIPELINE-DEAD (the class text rides on the alarm detail)
            IC._wedge_stack_poll(s, True, "CLASS=A-PIPELINE-DEAD (...)", None)
            self.assertIn("class-A-PIPELINE-DEAD", s._wedge_stack_done)
            IC._wedge_stack_poll(s, True, "x", {"escalated": True})
            self.assertIn("unrecovered", s._wedge_stack_done)
            # the alarm clears: the reasons re-arm for the next wedge
            IC._wedge_stack_poll(s, False, "", None)
            self.assertEqual(s._wedge_stack_done, set())
        finally:
            st.stop()

    def test_no_alarm_writes_nothing(self):
        s = _Sched(queued=0, running=1, age=0.0)
        poll = self._poller(s)
        logging.getLogger(IC.logger.name)
        with self.assertNoLogs(IC.logger, level="ERROR"):
            self.assertFalse(poll())


class ForwardHangTest(CustomTestCase):
    def test_bound_is_not_below_the_wedge_alarm_threshold(self):
        self.assertGreaterEqual(FORWARD_HANG_SECONDS, 30.0)

    def test_moves_reset_and_a_frozen_counter_with_a_batch_set_fires_after_the_bound(self):
        s = _Sched(forward_ct=100)
        self.assertEqual(check_forward_hang_once(s, now=1000.0)[0], False)  # first sight
        self.assertEqual(check_forward_hang_once(s, now=1000.0 + 5)[0], False)  # below the bound
        hang, detail = check_forward_hang_once(s, now=1000.0 + FORWARD_HANG_SECONDS + 1)
        self.assertTrue(hang)
        self.assertIn("forward_ct=100 has not moved", detail)
        s.forward_ct = 101  # a forward completed
        self.assertFalse(check_forward_hang_once(s, now=1100.0)[0])
        self.assertFalse(check_forward_hang_once(s, now=1100.0 + 5)[0])

    def test_not_a_hang_when_idle_paused_dormant_or_initializing(self):
        for mutate in (
            lambda x: setattr(x, "cur_batch_for_debug", None),
            lambda x: setattr(x, "_engine_paused", True),
            lambda x: setattr(x, "pdflip_dormant", True),
            lambda x: setattr(x, "is_initializing", True),
        ):
            s = _Sched(forward_ct=7)
            check_forward_hang_once(s, now=1.0)
            mutate(s)
            self.assertFalse(check_forward_hang_once(s, now=1.0 + 3600)[0])

    def test_tp0_shape_empty_queue_in_the_forward_is_blind_to_the_wedge_alarm_but_not_here(self):
        """TP0 of the specimen: queue empty, nothing 'running' by the queue's count, a batch handed to the
        loop and the counter stuck. The ADMISSION-WEDGE predicate cannot fire; FORWARD-HANG does."""
        st = _SchedulerThread()
        try:
            s = _Sched(queued=0, running=0, age=500.0, forward_ct=2529, ident=st.ident)
            alarm, _ = check_admission_wedge_once(s)
            self.assertFalse(alarm)  # the blind spot: nothing to report on this rank
            t0 = time.monotonic()
            with self.assertNoLogs(IC.logger, level="ERROR"):
                IC._forward_hang_poll(s, now=t0)  # first sight of this counter value
            with self.assertLogs(IC.logger, level="ERROR") as cm:
                IC._forward_hang_poll(s, now=t0 + FORWARD_HANG_SECONDS + 5)
            text = "\n".join(cm.output)
            self.assertIn(f"{FORWARD_HANG} rank=pp0-tp0 forward_ct=2529 has not moved", text)
            self.assertIn(f"{WEDGE_STACK} rank=pp0-tp0 why=forward-hang@2529", text)
            self.assertIn("_waiting_for_peers_in_group_min", text)
            # once per frozen counter value
            with self.assertNoLogs(IC.logger, level="ERROR"):
                IC._forward_hang_poll(s, now=t0 + FORWARD_HANG_SECONDS + 15)
            # and the watchdog's poll really calls it (the wiring, not just the function)
            with mock.patch.object(IC, "_forward_hang_poll") as fh, mock.patch.object(
                IC, "_wedge_stack_poll"
            ) as wp:
                make_admission_wedge_poller(s)()
            fh.assert_called_once_with(s)
            self.assertEqual(wp.call_count, 2)  # before and after the recovery step
        finally:
            st.stop()


if __name__ == "__main__":
    unittest.main()
