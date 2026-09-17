"""#1469: the node-value trail (RETAIN / FREE / EVICT) exists and is capped."""
import inspect
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.mem_cache import unified_radix_cache as urc
from sglang.srt.mem_cache.unified_cache_components import mamba_component as mc
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-weg2-unit")


class Test1469(unittest.TestCase):
    def test_note_is_capped_and_never_raises(self):
        mc._1469_N = mc._1469_CAP - 1
        with self.assertLogs(mc.logger, level="INFO") as cm:
            mc._1469_note("RETAIN", rid="r", value=True)
        self.assertEqual(len(cm.records), 1)
        self.assertIn("#1469 RETAIN rid=r value=True", cm.records[0].getMessage())
        mc._1469_note("RETAIN", rid="r2")            # over the cap: silent
        mc._1469_N = 0

    def test_the_three_sites_are_wired(self):
        src = inspect.getsource(mc.MambaComponent.prepare_for_caching_req)
        self.assertIn('_1469_note("RETAIN"', src)
        self.assertIn("_prepare_for_caching_req_impl(", src)
        self.assertIn('_1469_note("FREE"', inspect.getsource(mc.MambaComponent._free_mamba_value))
        self.assertIn('_1469_note("EVICT"', inspect.getsource(urc.UnifiedRadixCache._evict_component_and_detach_lru))


if __name__ == "__main__":
    unittest.main()
