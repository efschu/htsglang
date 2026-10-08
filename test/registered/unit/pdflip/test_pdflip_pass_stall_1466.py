"""#1466: PASS-STALL -- the phase terms of a scheduler pass, and the line."""
import inspect
import os
import types
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.managers import scheduler as sched_mod
from flliper.srt.managers import scheduler_pp_mixin as ppm
from flliper.srt.managers import pdflip_pass_timer as pt
from flliper.srt.managers.scheduler_components import request_receiver as rr
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-pdflip-unit")


class Test1466(unittest.TestCase):
    def test_timed_records_ms_on_the_holder_and_read_resets(self):
        class H:
            @pt.timed("_x_ms")
            def work(self, n):
                return n * 2
        h = H()
        self.assertEqual(h.work(21), 42)
        self.assertGreaterEqual(h._x_ms, 0.0)
        v = pt.read_ms(h, "_x_ms")
        self.assertGreaterEqual(v, 0.0)
        self.assertEqual(pt.read_ms(h, "_x_ms"), 0.0)          # reset
        self.assertEqual(pt.read_ms(types.SimpleNamespace(), "_never"), 0.0)

    def test_timed_survives_a_bare_namespace_and_exceptions(self):
        @pt.timed("_y_ms")
        def boom(self):
            raise KeyError("x")
        ns = types.SimpleNamespace()
        with self.assertRaises(KeyError):
            boom(ns)
        self.assertGreaterEqual(ns._y_ms, 0.0)               # recorded in finally
        self.assertEqual(boom.__wrapped__.__name__, "boom")

    def test_the_four_phases_are_decorated(self):
        self.assertTrue(hasattr(rr.SchedulerRequestReceiver.recv_requests, "__wrapped__"))
        self.assertTrue(hasattr(sched_mod.Scheduler.get_next_batch_to_run, "__wrapped__"))
        self.assertTrue(hasattr(sched_mod.Scheduler.run_batch, "__wrapped__"))
        self.assertTrue(hasattr(ppm.SchedulerPPMixin._pp_forward_and_process_input_requests, "__wrapped__"))

    def test_the_line_is_emitted_per_pass_after_the_flip_tick(self):
        # inspect.getsource stops at the docstring of this method (56 lines);
        # slice the file between the two loop definitions instead
        text = open(inspect.getsourcefile(ppm)).read()
        src = text[text.index("    def event_loop_pp(self"):text.index("    def event_loop_pp_disagg_prefill(")]
        self.assertIn("#1466 PASS-STALL", src)
        self.assertLess(src.index("self._pp_flip_pass_tick(mb_id)"), src.index("#1466 PASS-STALL"))
        self.assertIn("if _1466_pass - _1466_run >= 300.0:", src)


    def test_1447_pass_tail_keeps_commit_and_recv_for_the_pass_stall_line(self):
        # nf12 (1447): _pp_process_batch_result zeroed the two terms before PASS-STALL read them,
        # so PASS-STALL printed commit_ms=0/proxy_recv_ms=0 and the wait fell into other_ms.
        seen = {}

        class H:
            _1463_recv_ms = 79.0
            _1463_commit_ms = 1087.0
            ps = types.SimpleNamespace(pp_rank=2)

            def process_batch_result(self, batch, result):
                seen["called"] = True

        h = H()
        batch = types.SimpleNamespace(reqs=[])
        ppm.SchedulerPPMixin._pp_process_batch_result(h, batch, None)
        self.assertTrue(seen.get("called"))
        # the PASS-TAIL consumer zeroed its own terms ...
        self.assertEqual(h._1463_recv_ms, 0.0)
        self.assertEqual(h._1463_commit_ms, 0.0)
        # ... and kept the copies PASS-STALL adds back
        self.assertEqual(h._1466_recv_keep, 79.0)
        self.assertEqual(h._1466_commit_keep, 1087.0)
        # two tail passes between two PASS-STALL reads sum up
        h._1463_recv_ms, h._1463_commit_ms = 1.0, 2.0
        ppm.SchedulerPPMixin._pp_process_batch_result(h, batch, None)
        self.assertEqual((h._1466_recv_keep, h._1466_commit_keep), (80.0, 1089.0))

    def test_1447_the_pass_stall_block_adds_the_kept_terms_and_resets_them(self):
        text = open(inspect.getsourcefile(ppm)).read()
        src = text[text.index("    def event_loop_pp(self"):text.index("    def event_loop_pp_disagg_prefill(")]
        self.assertIn('getattr(self, "_1466_recv_keep", 0.0)', src)
        self.assertIn('getattr(self, "_1466_commit_keep", 0.0)', src)
        self.assertIn("self._1466_recv_keep = 0.0", src)
        self.assertIn("self._1466_commit_keep = 0.0", src)
        # the keep is read before the line is built and reset after it
        self.assertLess(src.index('getattr(self, "_1466_commit_keep", 0.0)'), src.index("#1466 PASS-STALL pp_rank"))
        self.assertLess(src.index("#1466 PASS-STALL pp_rank"), src.index("self._1466_commit_keep = 0.0"))


if __name__ == "__main__":
    unittest.main()
