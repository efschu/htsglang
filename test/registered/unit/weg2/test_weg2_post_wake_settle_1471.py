"""#1471: a held request whose store read is still short is parked after the
wake and released when its re-read completes (or the bound lapses)."""
import inspect
import os
import types
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers import scheduler as sched_mod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-weg2-unit")

S = sched_mod.Scheduler


def _holder(states):
    """A bare holder with the four methods bound; `states[rid]` drives
    _weg2_refetch_one's verdict."""
    h = types.SimpleNamespace()
    h.waiting_queue = []
    h.weg2_dormant_hold = []
    h.tree_cache = types.SimpleNamespace(cache_controller=types.SimpleNamespace(weg2_hold_rids=set()))
    h.WEG2_POST_WAKE_SETTLE_S = S.WEG2_POST_WAKE_SETTLE_S
    h.ps = types.SimpleNamespace(tp_size=1)
    for n in ("_weg2_release_dormant_hold", "_weg2_post_wake_settle_tick", "_weg2_group_min_flags"):
        setattr(h, n, types.MethodType(getattr(S, n), h))
    h._weg2_refetch_one = lambda req, now: states[req.rid]
    return h


class Test1471(unittest.TestCase):
    def test_wake_queues_complete_reads_and_parks_short_ones(self):
        r1, r2 = types.SimpleNamespace(rid="a"), types.SimpleNamespace(rid="b")
        h = _holder({"a": "complete", "b": "reissued"})
        h.weg2_dormant_hold = [r1, r2]
        h.tree_cache.cache_controller.weg2_hold_rids = {"a", "b"}
        n = h._weg2_release_dormant_hold()
        self.assertEqual(n, 1)
        self.assertEqual([r.rid for r in h.waiting_queue], ["a"])
        self.assertEqual([r.rid for r in h.weg2_post_wake_settle], ["b"])
        self.assertEqual(h.tree_cache.cache_controller.weg2_hold_rids, {"b"})   # claim kept while parked

    def test_settle_tick_releases_on_complete_or_lapse(self):
        r = types.SimpleNamespace(rid="b", _1471_since=0.0)   # since epoch 0 -> lapsed
        h = _holder({"b": "wait"})
        h.weg2_post_wake_settle = [r]
        h.tree_cache.cache_controller.weg2_hold_rids = {"b"}
        self.assertEqual(h._weg2_post_wake_settle_tick(), 1)                  # lapsed -> released anyway
        self.assertEqual([x.rid for x in h.waiting_queue], ["b"])
        self.assertEqual(h.weg2_post_wake_settle, [])
        self.assertEqual(h.tree_cache.cache_controller.weg2_hold_rids, set())
        import time
        r2 = types.SimpleNamespace(rid="c", _1471_since=time.monotonic())
        h2 = _holder({"c": "reading"})
        h2.weg2_post_wake_settle = [r2]
        self.assertEqual(h2._weg2_post_wake_settle_tick(), 0)                 # still reading, not lapsed
        self.assertEqual([x.rid for x in h2.weg2_post_wake_settle], ["c"])

    def test_group_min_flags_is_local_on_one_rank(self):
        h = _holder({})
        self.assertEqual(h._weg2_group_min_flags([True, False, True]), [1, 0, 1])
        self.assertEqual(h._weg2_group_min_flags([]), [])

    def test_tick_is_wired_into_every_scheduling_pass(self):
        src = inspect.getsource(S.get_next_batch_to_run)
        self.assertIn("_weg2_post_wake_settle_tick()", src)


if __name__ == "__main__":
    unittest.main()
