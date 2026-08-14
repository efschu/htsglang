# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#363 defect 7: the first stage measurement must be takeable.

THE DEADLOCK, measured on metal after defects 3, 4 and 6 were cleared.

``_intra_phase_decide`` returned ``None`` whenever the boot stage table was
absent, and that suppressed the ms/round SPLIT along with the decision. The
split is a property of the BOUNDARY, not of a choice, so suppressing it closed
a loop:

    no stage table
      -> ms_decision = None on every verdict row
      -> stage_measure_pass refuses: "no boundary carries an ms/round split"
      -> no measurement in the canon
      -> StageTable refuses the stage for carrying no measurement (#578)
      -> no stage table

Every rig starts with an empty table, so the FIRST measurement could never be
taken anywhere. The measurement pass's own comment read the empty split as
"a boot without --regime-stage-clock" -- one way to get there, and not the one
that happens.

THE FIX. The split is recorded whenever the clock is wired. Only the DECISION
needs a table; without one the record carries ``target=None``, the measurement
fields, and a reason that says so. ``wants_flip`` is False, and act mode is
unaffected because ``_act_interlocks`` refuses on the missing table
independently.
"""

from __future__ import annotations

import unittest


class _Window:
    def __init__(self, total, share):
        self.mean_total_ms = total
        self.mean_wait_share = share


class _Clock:
    def __init__(self, total=123.5, share=0.25):
        self.window = _Window(total, share)
        self.observed = []

    def observe_round(self, r, compute, wait):
        self.observed.append((r, compute, wait))

    def decide(self, current, candidates):  # pragma: no cover - not reached
        raise AssertionError("decide() must not be called without a table")


class _Runtime:
    """The two attributes `_intra_phase_decide` actually reads, plus a clock."""

    def __init__(self, table=None, current_stage=None):
        from sglang.srt.managers.regime_runtime import RegimeObserver

        self._intra_phase_decide = RegimeObserver._intra_phase_decide.__get__(self)
        self._table = table
        self._current_stage = current_stage
        self._stage_clock = _Clock()
        self._tp_size = 1
        self._collective_min = None
        self._round = 42
        self._ms_compute_sum = 90.0
        self._ms_wait_sum = 30.0
        self._ms_split_n = 1


class TestTheSplitIsRecordedWithoutAStageTable(unittest.TestCase):
    def test_a_boundary_with_no_table_still_carries_the_split(self):
        """The arm that was red: this returned None, and the whole chain
        downstream had nothing to measure."""
        d = _Runtime(table=None, current_stage=None)._intra_phase_decide(None)
        self.assertIsNotNone(
            d,
            "no record at all -- stage_measure_pass will refuse with 'no "
            "boundary carries an ms/round split' and the canon stays empty",
        )
        self.assertIsNotNone(d.mean_total_ms, "the split is the point")
        self.assertIsNotNone(d.mean_wait_share)

    def test_the_record_is_a_measurement_and_not_a_choice(self):
        d = _Runtime(table=None, current_stage=None)._intra_phase_decide(None)
        self.assertIsNone(d.target, "a decision without a table is a fabrication")
        self.assertFalse(d.wants_flip)
        self.assertIsNone(d.signal_pct)
        self.assertIsNone(d.threshold_pct)

    def test_the_reason_says_why_there_is_no_decision(self):
        d = _Runtime(table=None, current_stage=None)._intra_phase_decide(None)
        self.assertIn("measurement only", d.reason)
        self.assertIn("stage table", d.reason)

    def test_the_window_is_still_fed(self):
        """The clock must see the round even when nothing is decided, or the
        means it reports are taken over a window with holes in it."""
        rt = _Runtime(table=None, current_stage=None)
        rt._intra_phase_decide(None)
        self.assertEqual(len(rt._stage_clock.observed), 1)

    def test_a_missing_current_stage_takes_the_same_path(self):
        d = _Runtime(table=object(), current_stage=None)._intra_phase_decide(None)
        self.assertIsNotNone(d)
        self.assertIsNone(d.target)


class TestThePassAcceptsSuchARow(unittest.TestCase):
    """The other end of the loop: the pass must read the record we now write."""

    def test_the_pass_reads_mean_total_ms_from_the_record(self):
        from sglang.srt.managers.regime_ms_clock import MsDecision

        d = MsDecision(
            target=None,
            reason="measurement only",
            mean_total_ms=123.5,
            mean_wait_share=0.25,
        ).as_dict()
        # These two keys are exactly what stage_measure_pass consumes.
        self.assertEqual(d["mean_total_ms"], 123.5)
        self.assertEqual(d["mean_wait_share"], 0.25)
        self.assertIsNone(d["target"])

    def test_the_pass_no_longer_claims_an_empty_split_means_no_clock(self):
        import inspect

        from sglang.srt.planner import stage_measure_pass

        src = inspect.getsource(stage_measure_pass)
        self.assertIn("stage table was absent", src)


if __name__ == "__main__":
    unittest.main()
