"""#1466: PASS-STALL -- the phase terms of a scheduler pass, and the line."""
import inspect
import os
import types
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers import scheduler as sched_mod
from sglang.srt.managers import scheduler_pp_mixin as ppm
from sglang.srt.managers import weg2_pass_timer as pt
from sglang.srt.managers.scheduler_components import request_receiver as rr
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-weg2-unit")


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


if __name__ == "__main__":
    unittest.main()
