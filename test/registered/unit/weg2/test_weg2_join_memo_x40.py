"""23.09. (fnFL2x40): the first flip of the boot spent 6.3-7.7 s in the FIRST
deposit of every rank, every later tag 0.1-0.4 s -- and D's first collect moved
its whole tag in 92 ms, so the bytes were not it.

The manifest join is a pure function of the boot's manifest files. The first
leg recomputed it on its critical path once per region in the hook
(``leg_plan_from_join``) and once more in every lane thread of the first tag
(``_weg2_seq_lane_descs``), under one GIL. Measured on x40's own manifests,
CPU only: one join 0.55-0.57 s, three concurrent joins 2.09 s wall.

``join_manifests`` now keeps each distinct input once per process, concurrent
callers of one input wait for the first, and a boot-time warm-up joins the two
inputs the first leg needs. The cases below pin what makes that correct.
"""
import inspect
import os
import threading
import time
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import xchg_manifest as xm  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=5, suite="stage-a-weg2-unit")


def _piece(name, tag, rows=4):
    return xm.ManifestPiece(param_name=name, tensor_class="linear", rows_full=rows,
                            cols_full=8, itemsize=2, tag=tag, nbytes=rows * 16)


def _manifests(rows=4):
    """Two groups, one rank each, a main-region and a draft-region piece."""
    out = []
    for group in ("P", "D"):
        out.append(xm.RankManifest(
            group=group, rank=0, card=0, region_tag="weights", boot_token="b",
            pieces=(_piece("model.layers.0.w", "weights_0", rows),
                    _piece("mtp.fc.w", "weights_draft"))))
    return tuple(out)


class _CountingJoin:
    """Stands in for the real join: counts calls, overlaps them on purpose."""

    def __init__(self, delay_s=0.05, fail=False):
        self.calls = 0
        self.delay_s = delay_s
        self.fail = fail
        self._lock = threading.Lock()

    def __call__(self, manifests, *, pp_group="P", tp_group="D"):
        with self._lock:
            self.calls += 1
        time.sleep(self.delay_s)
        if self.fail:
            raise xm.wx.Weg2XchgSourceMissing("W74 test refusal")
        return ("joined", tuple(m.group for m in manifests), len(manifests))


class _MemoCase(unittest.TestCase):
    def setUp(self):
        self._orig = xm._join_manifests_uncached
        xm._JOIN_MEMO.clear()
        xm._JOIN_INFLIGHT.clear()

    def tearDown(self):
        xm._join_manifests_uncached = self._orig
        xm._JOIN_MEMO.clear()
        xm._JOIN_INFLIGHT.clear()

    def _install(self, stub):
        xm._join_manifests_uncached = stub
        return stub


class ConcurrentCallersJoinOnce(_MemoCase):
    def test_three_lane_threads_share_one_join(self):
        stub = self._install(_CountingJoin(delay_s=0.1))
        out = []

        def lane():
            # every lane re-reads the files: equal manifests, new objects
            out.append(xm.join_manifests(_manifests()))

        threads = [threading.Thread(target=lane) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(stub.calls, 1)
        self.assertEqual(len(out), 3)
        self.assertTrue(all(o is out[0] for o in out))

    def test_a_rewritten_manifest_is_a_new_key_never_a_stale_hit(self):
        stub = self._install(_CountingJoin(delay_s=0.0))
        first = xm.join_manifests(_manifests(rows=4))
        second = xm.join_manifests(_manifests(rows=6))  # the drafter's rewrite
        self.assertEqual(stub.calls, 2)
        self.assertIsNot(first, second)

    def test_a_failed_join_is_not_stored(self):
        self._install(_CountingJoin(delay_s=0.0, fail=True))
        with self.assertRaises(xm.wx.Weg2XchgSourceMissing):
            xm.join_manifests(_manifests())
        stub = self._install(_CountingJoin(delay_s=0.0))
        xm.join_manifests(_manifests())
        self.assertEqual(stub.calls, 1)


class TheWarmUpJoinsWhatTheFirstLegJoins(_MemoCase):
    def test_after_the_warm_up_the_lane_and_the_hook_inputs_are_hits(self):
        stub = self._install(_CountingJoin(delay_s=0.0))
        orig_mfb = xm.manifests_for_boot
        xm.manifests_for_boot = lambda **kw: (_manifests(), "")
        try:
            ms = xm.prewarm_joins()
        finally:
            xm.manifests_for_boot = orig_mfb
        self.assertEqual(set(ms), {"full", "region:weights"})
        calls = stub.calls
        # the lane derivation: the whole set
        xm.join_manifests(_manifests())
        # the hook: the main region, narrowed by the SAME function it uses
        narrowed, *_ = xm.narrow_manifests_to_region(_manifests(),
                                                     region_tag="weights")
        xm.join_manifests(narrowed)
        self.assertEqual(stub.calls, calls)

    def test_the_hook_narrows_through_the_shared_function(self):
        src = inspect.getsource(xm.leg_plan_from_join)
        self.assertIn("narrow_manifests_to_region(", src)
        self.assertNotIn("_empty_on_arrival", src)


if __name__ == "__main__":
    unittest.main()
