"""fnFL2 v57 (21.09.): ask the allocator whether a tag pool CAN be dropped.

v56 dropped one without asking and died in ``c10::AcceleratorError -- CUDA
error: invalid argument``, because tensors its copy had not covered still
held blocks.  ``active_blocks == 0`` is the precondition; ``inactive_gib``
is the prize (9.29 GiB on PP1, 7.91 on PP2 -- the per-rank load overhead
that killed v45 to v56).
"""

import unittest
from unittest import mock

from sglang.srt.managers import weg2_memory_saver as MS

_SEGS = [
    {"blocks": [
        {"size": 4 * 1024**3, "state": "active_allocated"},
        {"size": 2 * 1024**3, "state": "inactive"},
    ]},
    {"blocks": [{"size": 1024**3, "state": "inactive"}]},
]


class TagPoolOccupancy(unittest.TestCase):
    def setUp(self):
        self._saved = dict(MS._TAG_MEM_POOLS)
        MS._TAG_MEM_POOLS.clear()

    def tearDown(self):
        MS._TAG_MEM_POOLS.clear()
        MS._TAG_MEM_POOLS.update(self._saved)

    def test_an_unknown_tag_answers_None_not_zero(self):
        """Absent is not empty -- a None must not read as 'nothing in the way'."""
        with mock.patch("torch.cuda.is_available", return_value=True):
            self.assertIsNone(MS.tag_pool_occupancy("nope"))

    def test_it_separates_active_from_inactive(self):
        MS._TAG_MEM_POOLS["weights"] = mock.Mock(
            snapshot=mock.Mock(return_value=_SEGS)
        )
        with mock.patch("torch.cuda.is_available", return_value=True):
            occ = MS.tag_pool_occupancy("weights")
        self.assertEqual(occ["segments"], 2)
        self.assertEqual(occ["active_blocks"], 1)
        self.assertAlmostEqual(occ["active_gib"], 4.0, places=3)
        self.assertAlmostEqual(occ["inactive_gib"], 3.0, places=3)

    def test_an_unreadable_snapshot_is_None_and_warns(self):
        MS._TAG_MEM_POOLS["weights"] = mock.Mock(
            snapshot=mock.Mock(side_effect=RuntimeError("nope"))
        )
        with mock.patch("torch.cuda.is_available", return_value=True), \
             self.assertLogs(MS.logger, level="WARNING"):
            self.assertIsNone(MS.tag_pool_occupancy("weights"))

    def test_the_log_line_carries_both_numbers(self):
        MS._TAG_MEM_POOLS["weights"] = mock.Mock(
            snapshot=mock.Mock(return_value=_SEGS)
        )
        with mock.patch("torch.cuda.is_available", return_value=True), \
             self.assertLogs(MS.logger, level="INFO") as log:
            MS.log_tag_pool_occupancy("weights", when="after-load")
        line = "\n".join(log.output)
        self.assertIn("active_blocks=1", line)
        self.assertIn("inactive_gib=3.00", line)
        self.assertIn("when=after-load", line)


class TheRunnerProbesAfterLoading(unittest.TestCase):
    def test_it_runs_before_the_coverage_arm(self):
        import inspect

        from sglang.srt.model_executor import model_runner as MR

        src = inspect.getsource(MR)
        self.assertIn('log_tag_pool_occupancy(weights_tag, when="after-load")', src)
        self.assertLess(
            src.index("log_tag_pool_occupancy(weights_tag"),
            src.index("from sglang.srt.weg2.weight_exchange import arm_coverage_at_load"),
        )


if __name__ == "__main__":
    unittest.main()
