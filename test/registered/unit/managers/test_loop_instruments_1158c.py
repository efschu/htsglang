"""#1158c instruments 1-3 (review-1158c-quiesce-1010 §7): log lines only, no behaviour.

  1. EVENT-LOOP-ENTER once per rank in dispatch_event_loop, before any loop is chosen;
  2. LOOP-ITER-SLOW: an overlap-loop iteration past THRESHOLD_S prints its recv/lane/war/sched/run/result split, bounded
     to MAX_LINES per process; a fast iteration prints nothing;
  3. #1458 CTRL-SEND at the zmq send of the Weg-2 control requests (flush/release/resume) in FanOutCommunicator, never for
     other kinds, and the send itself is unchanged.
"""

import asyncio
import inspect
import logging
import time
import unittest

from flliper.srt.managers import communicator as C
from flliper.srt.managers.io_struct import FlushCacheReqInput
from flliper.srt.managers.scheduler_components.loop_iter_slow import MAX_LINES, THRESHOLD_S, LoopIterSlow

LOG = logging.getLogger("test_loop_instruments_1158c")


class TestLoopIterSlow(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual((THRESHOLD_S, MAX_LINES), (2.0, 20))

    def test_fast_iteration_is_silent(self):
        s = LoopIterSlow()
        s.begin()
        s.span("recv_ms", time.perf_counter())
        with self.assertNoLogs(LOG, level="DEBUG"):
            s.end(LOG)

    def test_slow_iteration_prints_the_split_and_the_cap_holds(self):
        s = LoopIterSlow(threshold_s=0.0, max_lines=2)
        with self.assertLogs(LOG, level="WARNING") as cm:
            for _ in range(5):
                s.begin()
                t0 = time.perf_counter()
                s.span("recv_ms", t0)
                s.span("war_ms", t0)
                s.span("result_ms", t0)
                s.span("result_ms", t0)  # pop_and_process may run twice in one iteration: accumulated
                s.end(LOG)
        self.assertEqual(len(cm.output), 2)
        self.assertRegex(cm.output[0], r"LOOP-ITER-SLOW iter=1 total_ms=\S+ recv_ms=\S+ war_ms=\S+ result_ms=\S+ t=")
        self.assertIn("iter=2 ", cm.output[1])
        self.assertEqual(s.iter, 5)

    def test_overlap_loop_is_wired(self):
        from flliper.srt.managers.scheduler import Scheduler

        src = inspect.getsource(Scheduler.event_loop_overlap)
        for frag in ("_slow = LoopIterSlow()", "_slow.begin()", "_slow.end(logger)", '"recv_ms"', '"lane_ms"',
                     '"war_ms"', '"sched_ms"', '"run_ms"', '"result_ms"'):
            self.assertIn(frag, src)
        self.assertLess(src.index("_slow.begin()"), src.index("self.request_receiver.recv_requests()"))


class TestEventLoopEnter(unittest.TestCase):
    def test_logged_before_any_loop_is_chosen(self):
        from flliper.srt.managers import scheduler as S

        src = inspect.getsource(S.dispatch_event_loop)
        self.assertIn('"EVENT-LOOP-ENTER disagg=%s', src)
        self.assertLess(src.index("EVENT-LOOP-ENTER"), src.index("scheduler.event_loop_"))


class TestCtrlSend(unittest.TestCase):
    def _call(self, obj, mode):
        sent = []
        comm = C.FanOutCommunicator(sent.append, fan_out=1, mode=mode)

        async def run():
            task = asyncio.ensure_future(comm(obj))
            await asyncio.sleep(0)
            comm.handle_recv("r")
            return await task

        return asyncio.run(run()), sent

    def test_flush_is_stamped_and_sent_unchanged(self):
        for mode in ("queueing", "watching"):
            with self.subTest(mode=mode):
                obj = FlushCacheReqInput()
                with self.assertLogs(C.logger, level="INFO") as cm:
                    res, sent = self._call(obj, mode)
                self.assertEqual(res, ["r"])
                self.assertEqual(sent, [obj])
                self.assertEqual(len(cm.output), 1)
                self.assertRegex(cm.output[0], r"#1458 CTRL-SEND kind=FlushCacheReqInput t=\d+\.\d{3}$")

    def test_other_kinds_are_not_stamped(self):
        with self.assertNoLogs(C.logger, level="DEBUG"):
            res, sent = self._call("not-a-control-request", "queueing")
        self.assertEqual(sent, ["not-a-control-request"])

    def test_kinds(self):
        self.assertEqual(C._CTRL_SEND_KINDS, ("FlushCacheReqInput", "ReleaseMemoryOccupationReqInput",
                                             "ResumeMemoryOccupationReqInput"))


if __name__ == "__main__":
    unittest.main()
