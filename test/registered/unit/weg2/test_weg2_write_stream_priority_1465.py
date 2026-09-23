"""#1465: group P's write-through stream at high CUDA priority (env-gated)."""
import inspect
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers import cache_controller as cc
from sglang.srt.mem_cache import unified_radix_cache as urc
from sglang.srt.weg2 import launcher
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-weg2-unit")


class Test1465(unittest.TestCase):
    def test_priority_unset_is_default_and_bad_values_fall_back(self):
        for raw, want in (("", 0), ("-1", -1), ("  -2 ", -2), ("high", 0), ("0", 0)):
            os.environ[cc.HICACHE_WRITE_STREAM_PRIORITY_ENV] = raw
            try:
                self.assertEqual(cc.hicache_write_stream_priority(), want, raw)
            finally:
                os.environ.pop(cc.HICACHE_WRITE_STREAM_PRIORITY_ENV, None)
        self.assertEqual(cc.hicache_write_stream_priority(), 0)

    def test_write_stream_is_built_with_the_priority_and_load_stream_is_not(self):
        src = inspect.getsource(cc.HiCacheController.__init__)
        self.assertIn("device_module.Stream(priority=hicache_write_stream_priority())", src)
        self.assertIn("self.load_stream = device_module.Stream()", src)

    def test_launcher_sets_it_for_group_p_only(self):
        src = inspect.getsource(launcher)
        self.assertIn('env_p.setdefault("SGLANG_HICACHE_WRITE_STREAM_PRIORITY", "-1")', src)
        self.assertNotIn('env_d.setdefault("SGLANG_HICACHE_WRITE_STREAM_PRIORITY"', src)
        self.assertNotIn('env_d["SGLANG_HICACHE_WRITE_STREAM_PRIORITY"]', src)

    def test_flush_drain_stamp_is_before_the_return(self):
        src = inspect.getsource(urc.UnifiedRadixCache.writing_check)
        a = src.index("assert len(self.ongoing_write_through) == 0")
        stamp = src.index("#1465 WRITE-BACK DRAIN")
        ret = src.index("            return\n", a)
        self.assertLess(a, stamp)
        self.assertLess(stamp, ret)


if __name__ == "__main__":
    unittest.main()
