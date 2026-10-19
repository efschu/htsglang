"""S73 (rc16 bf9ca46f45): the PUBLISH-SWEEP line names its own refusals.

rc16 D TP1/TP2 14:13:15 (park of pdflip-174-648): the flush sweeps read
``unbacked=6 issued=0 refused=6`` five times with pins 0/7 and
``#1470 FLUSH-PUBLISH unbacked_left=6`` went into the reset -- while the
``#1421 BACKUP-REFUSED`` lines, sampled per reason, printed none of those 39
refusals (n=2225 -> n=2265). The real sweep over the full-arena fixture of
SWEEP-FULL-ARENA (1007) must name every refusal by reason in its stats and
its line, and collect nothing outside a sweep."""

from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(__file__)

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.test.test_utils import CustomTestCase

from test_sweep_full_arena_skip_1007 import CHAIN, _DirectPool, _sweep, _tree


class TestSweepNamesItsRefusals(CustomTestCase):
    def test_full_arena_refusals_are_named_in_the_stats_and_the_line(self):
        cache = _tree()
        pool = _DirectPool(room=())
        with self.assertLogs("flliper.srt.mem_cache.unified_radix_cache", level="WARNING") as cm:
            stats = _sweep(cache, pool, on=False)
        self.assertEqual((stats["unbacked"], stats["issued"], stats["refused"]), (2 * CHAIN, 0, 2 * CHAIN))
        # every node the sweep refused has its innermost reason counted
        self.assertIn("arena_claim", stats["refused_by"])
        self.assertGreaterEqual(sum(stats["refused_by"].values()), stats["refused"])
        lines = [m for m in cm.output if "PUBLISH-SWEEP" in m]
        self.assertTrue(lines)
        self.assertIn(" refused_by=", lines[-1])
        self.assertIn("arena_claim:%d" % stats["refused_by"]["arena_claim"], lines[-1])
        self.assertIsNone(getattr(cache, "_s73_sweep_why", None), "collected only inside a sweep")

    def test_a_sweep_that_refuses_nothing_prints_no_census(self):
        cache = _tree()
        pool = _DirectPool(room=[n.hash_value[0] for n in cache._collect_all_nodes()
                                 if n is not cache.root_node])
        with self.assertLogs("flliper.srt.mem_cache.unified_radix_cache", level="WARNING") as cm:
            stats = _sweep(cache, pool, on=False)
        self.assertEqual((stats["issued"], stats["refused"]), (2 * CHAIN, 0))
        self.assertEqual(stats["refused_by"], {})
        self.assertFalse(any("refused_by=" in m for m in cm.output if "PUBLISH-SWEEP" in m))
