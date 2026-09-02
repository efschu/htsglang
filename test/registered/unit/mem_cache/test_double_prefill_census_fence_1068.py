"""#1068 slice 3 (T15, G9): the #939 census carries `fence_proceeds`.

A flip whose write-back fence stayed incomplete for `_WRITEBACK_DEFER_LIMIT`
consecutive defers PROCEEDS (phase_flip_runtime, '#1028 WRITEBACK DEFER LIMIT
reached'). The residents it re-admits may then miss the store and recompute
in full -- a loss term the #939 acceptance line must carry beside its bound,
or a `within_bound=false` cannot be told apart from a store defect.

RED on 846c6797b9: `DoublePrefillCensus` has no `fence_proceeds` field and no
`note_fence_proceed`; `log_fields()` ends with `readmitted`.

Lifetime rule (named, because the ORDER in `_execute_body` forces it): the
fence verdict is taken BEFORE `_post_cutover_readmit` ends the previous
wave's census, so a proceed noted at module level is SEEDED into the census
of the wave that follows the reset, not into the wave that is being closed.
"""

import os
import unittest
from unittest import mock

from sglang.srt.mem_cache import producer_phase_census as pc
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestTheCensusCarriesFenceProceeds(CustomTestCase):
    def setUp(self):
        pc.reset_for_test()

    def tearDown(self):
        pc.reset_for_test()

    def test_note_fence_proceed_is_the_last_field(self):
        # T15
        c = pc.DoublePrefillCensus()
        c.note_fence_proceed()
        c.note_fence_proceed()
        fields = c.log_fields()
        self.assertEqual(fields["fence_proceeds"], 2)
        self.assertEqual(list(fields)[-1], "fence_proceeds")
        self.assertTrue(c.format_line().endswith("fence_proceeds=2"), c.format_line())

    def test_a_fresh_census_reads_zero(self):
        c = pc.DoublePrefillCensus()
        self.assertEqual(c.log_fields()["fence_proceeds"], 0)

    def test_a_module_level_proceed_seeds_the_next_wave(self):
        # ORDER: fence proceed -> reset (previous wave ends) -> readmit wave
        # records. The proceed belongs to the wave being re-admitted.
        with mock.patch.dict(os.environ, {"SGLANG_MATCH_REFUSAL_CENSUS_EVERY": "1"}):
            pc.note_fence_proceed()
            pc.reset_double_prefill_census()
            pc.note_double_prefill("r1", already_computed=4096, recovered=4096)
            census = pc.double_prefill_census()
            self.assertIsNotNone(census)
            self.assertEqual(census.fence_proceeds, 1)
            self.assertEqual(census.log_fields()["readmitted"], 1)
            # the NEXT reset starts a wave without a proceed
            pc.reset_double_prefill_census()
            pc.note_double_prefill("r2", already_computed=4096, recovered=4096)
            self.assertEqual(pc.double_prefill_census().fence_proceeds, 0)

    def test_a_proceed_for_a_wave_that_never_came_does_not_leak(self):
        # cutover A proceeds, re-admits nothing (no census); cutover B does
        # not proceed and re-admits: B's census must read 0, not A's 1.
        with mock.patch.dict(os.environ, {"SGLANG_MATCH_REFUSAL_CENSUS_EVERY": "1"}):
            pc.note_fence_proceed()
            pc.reset_double_prefill_census()
            # no readmit in wave A
            pc.reset_double_prefill_census()
            pc.note_double_prefill("r1", already_computed=1, recovered=1)
            self.assertEqual(pc.double_prefill_census().fence_proceeds, 0)


if __name__ == "__main__":
    unittest.main()
