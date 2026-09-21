"""fnFL2 v45-v49 (21.09.): the load transients must leave the private tag pool.

Under ``--flip-weights family`` the weights load inside
``tag_pool_scope``'s private ``torch.cuda.MemPool``, and that pool is CACHED
per tag -- so it is not left until the whole load is over, and
``torch.cuda.empty_cache()`` does not reach a live private pool.  Every
layer's Marlin repack transient therefore stayed pinned.

MEASURED (boots v47 vs v48, same ranks, two very different expert bookings):
the per-rank overhead was identical across bookings (7.58 GiB on the 8-layer
stage, 10.10 on the 11-layer one) and rose 0.84 GiB per layer.  Group P's
29-layer stage 0 pays ~25 GiB of a 32.6 GiB card before one expert is
resident, and the boots died in cu_mem_create at layer 28 of 29.
"""

import unittest
from unittest import mock

from sglang.srt.managers import weg2_memory_saver as MS


class ReleaseActiveTagPools(unittest.TestCase):
    def setUp(self):
        self._saved = dict(MS._TAG_MEM_POOLS)
        MS._TAG_MEM_POOLS.clear()

    def tearDown(self):
        MS._TAG_MEM_POOLS.clear()
        MS._TAG_MEM_POOLS.update(self._saved)

    def test_no_pools_is_a_zero_not_a_call(self):
        with mock.patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(MS.release_active_tag_pools(), 0)

    def test_every_pool_is_released_and_the_cache_emptied_after(self):
        MS._TAG_MEM_POOLS["weights_0"] = mock.Mock(id=(0, 1))
        MS._TAG_MEM_POOLS["weights_1"] = mock.Mock(id=(0, 2))
        calls = []
        with mock.patch("torch.cuda.is_available", return_value=True), \
             mock.patch("torch.cuda.current_device", return_value=0), \
             mock.patch("torch.cuda.memory._cuda_releasePool",
                        side_effect=lambda d, i: calls.append(("release", d, i))), \
             mock.patch("torch.cuda.empty_cache",
                        side_effect=lambda: calls.append(("empty",))):
            n = MS.release_active_tag_pools(reason="test")
        self.assertEqual(n, 2)
        self.assertEqual([c[0] for c in calls], ["release", "release", "empty"])
        self.assertEqual({c[2] for c in calls if c[0] == "release"}, {(0, 1), (0, 2)})

    def test_one_failing_pool_is_named_and_does_not_stop_the_others(self):
        MS._TAG_MEM_POOLS["bad"] = mock.Mock(id=(0, 1))
        MS._TAG_MEM_POOLS["good"] = mock.Mock(id=(0, 2))

        def _rp(d, i):
            if i == (0, 1):
                raise RuntimeError("nope")

        with mock.patch("torch.cuda.is_available", return_value=True), \
             mock.patch("torch.cuda.current_device", return_value=0), \
             mock.patch("torch.cuda.memory._cuda_releasePool", side_effect=_rp), \
             mock.patch("torch.cuda.empty_cache"), \
             self.assertLogs(MS.logger, level="WARNING") as log:
            n = MS.release_active_tag_pools()
        self.assertEqual(n, 1)
        self.assertIn("bad", "\n".join(log.output))


class ThePresplitPathCallsIt(unittest.TestCase):
    def test_the_release_stands_before_the_empty_cache(self):
        import inspect

        from sglang.srt.layers.moe.fused_moe_triton import layer as L

        src = inspect.getsource(L.FusedMoE._ct_stream_presplit_now)
        self.assertIn('release_active_tag_pools(reason="ct-stream-presplit")', src)
        self.assertLess(
            src.index("release_active_tag_pools(reason="),
            src.rindex("torch.cuda.empty_cache()"),
        )


if __name__ == "__main__":
    unittest.main()
