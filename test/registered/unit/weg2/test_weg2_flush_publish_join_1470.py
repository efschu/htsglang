"""#1470: the sleep flush publishes every un-backed node and joins the
write-throughs before the tree is reset."""
import inspect
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers import scheduler as sched_mod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-weg2-unit")


class Test1470(unittest.TestCase):
    def test_sweep_loops_and_joins_before_reset(self):
        src = inspect.getsource(sched_mod.Scheduler.flush_cache)
        i_sweep = src.index("_sweep(max_issue=256)")
        i_join = src.index("_wc(write_back=True)")
        i_reset = src.index("self.tree_cache.reset()")
        self.assertLess(i_sweep, i_join)
        self.assertLess(i_join, i_reset)
        self.assertIn('os.environ.get("SGLANG_HICACHE_FLUSH_PUBLISH_SWEEP", "1") != "0"', src)
        self.assertNotIn("len(self.waiting_queue) == 0", src[:i_reset])
        self.assertIn("#1470 FLUSH-PUBLISH", src)


if __name__ == "__main__":
    unittest.main()
