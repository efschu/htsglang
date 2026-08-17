"""#631: pins for the cutover's per-stage memory census.

WHAT THESE PROTECT. The census sits ON the cutover path, which is the
no-return region: after ``pre_write_fns`` the source pool's pages are
gone and there is no abandoning left. So the load-bearing property of
this instrument is not its arithmetic, it is that it CANNOT be the
reason a flip dies. Several pins below exist only to hold that line --
they assert that broken probes, missing censuses and double-begins all
degrade to a missing log line rather than an exception.

The arithmetic pins matter for a second reason: the number this
produces is the one a successor will size a pre-flight against, and a
trough attributed to the wrong stage sends the fix to the wrong module.
"""

import unittest

from sglang.srt.managers import phase_flip_seam_census as census

MIB = 1024 * 1024


def _probe_sequence(*samples):
    """A probe returning each (free, reserved, allocated) triple in turn.

    Sizes are given in MiB for legibility and converted here; a test that
    writes out 5705*1024*1024 three times per stage stops being readable
    at exactly the point where the numbers start mattering.
    """
    queue = list(samples)

    def probe():
        if not queue:
            raise AssertionError("probe called more times than the test scripted")
        sample = queue.pop(0)
        if sample is None:
            return None
        free, reserved, allocated = sample
        return (free * MIB, reserved * MIB, allocated * MIB)

    return probe


class TestTheInstrumentCannotKillAFlip(unittest.TestCase):
    """The no-return-path contract. Each of these would be a crash in the
    seam if the module raised instead of degrading."""

    def setUp(self):
        census.reset()

    def tearDown(self):
        census.reset()

    def test_marking_with_no_open_census_is_a_no_op(self):
        # The cutover marks unconditionally; a flip that somehow reaches a
        # mark without a begin (an abandoned flip retried, an exception
        # path) must not take the instance down.
        census.mark("kv_pack")
        self.assertIsNone(census.active())

    def test_end_with_no_open_census_returns_none(self):
        self.assertIsNone(census.end())

    def test_a_probe_that_raises_does_not_escape(self):
        def exploding_probe():
            raise RuntimeError("driver query failed")

        census.begin("pp_to_tp", 0, probe=exploding_probe)
        # begin() itself marks "entry"; neither it nor a later mark may
        # propagate the failure.
        census.mark("kv_pack")
        result = census.end()
        self.assertIsNotNone(result)

    def test_a_probe_returning_none_is_recorded_as_a_gap_not_dropped(self):
        # A silently-dropped row would close the record up and make the
        # NEXT stage appear to have spent the missing stage's memory.
        probe = _probe_sequence((5000, 100, 90), None, (4000, 100, 90))
        census.begin("pp_to_tp", 1, probe=probe)
        census.mark("kv_pack")
        census.mark("kv_exchange")
        result = census.active()
        self.assertEqual(len(result.stages), 3)
        self.assertEqual(result.stages[1][1], -1)
        self.assertIn("probe-failed", result.format_line())

    def test_double_begin_does_not_raise_and_starts_a_fresh_record(self):
        census.begin("pp_to_tp", 0, probe=_probe_sequence((5000, 0, 0)))
        census.begin("tp_to_pp", 0, probe=_probe_sequence((4000, 0, 0)))
        active = census.active()
        self.assertEqual(active.direction, "tp_to_pp")
        self.assertEqual(len(active.stages), 1)


class TestAttribution(unittest.TestCase):
    """The arithmetic that decides which module gets the fix."""

    def setUp(self):
        census.reset()

    def tearDown(self):
        census.reset()

    def test_the_trough_names_the_stage_that_spent_the_memory(self):
        # Shape of the real measurement: flat, then one stage takes a
        # large bite, then it is handed back.
        probe = _probe_sequence(
            (5705, 2000, 1900),  # entry
            (5700, 2000, 1900),  # plan
            (5698, 2000, 1900),  # kv_pack -- 0.5 MiB carried, near-flat
            (2745, 2000, 1900),  # gdn_state -- the bite
            (5685, 2000, 1900),  # done
        )
        census.begin("pp_to_tp", 0, probe=probe)
        census.mark("plan")
        census.mark("kv_pack")
        census.mark("gdn_state")
        result = census.end()

        self.assertEqual(result.trough()[0], "gdn_state")
        self.assertEqual(result.peak_bytes(), (5705 - 2745) * MIB)

    def test_the_transient_is_measured_against_the_baseline_not_the_end(self):
        # A flip that ends with LESS free memory than it started (a real
        # phase asymmetry) must still report the transient relative to the
        # entry baseline; measuring against the end would under-report a
        # peak that a corridor check has to survive.
        probe = _probe_sequence(
            (6000, 0, 0),
            (2000, 0, 0),
            (4000, 0, 0),
        )
        census.begin("pp_to_tp", 0, probe=probe)
        census.mark("backing_restore")
        result = census.end()
        self.assertEqual(result.peak_bytes(), 4000 * MIB)

    def test_slack_is_reported_because_it_explains_the_driver_visible_cost(self):
        # The driver is asked for (what the stage wants) - (torch's slack),
        # so a line without slack shows the symptom and hides the cause.
        probe = _probe_sequence((5000, 3000, 1000), (4000, 3000, 1000))
        census.begin("pp_to_tp", 2, probe=probe)
        census.mark("backing_restore")
        line = census.end().format_line()
        self.assertIn("slack=2000", line)

    def test_the_line_names_direction_rank_and_trough_stage(self):
        probe = _probe_sequence((5000, 0, 0), (1000, 0, 0), (5000, 0, 0))
        census.begin("tp_to_pp", 2, probe=probe)
        census.mark("backing_restore")
        line = census.end().format_line()
        self.assertIn("tp_to_pp", line)
        self.assertIn("rank 2", line)
        self.assertIn("backing_restore", line)
        self.assertIn("4000 MiB", line)

    def test_per_stage_steps_are_reported_so_one_stage_can_be_blamed(self):
        # The delta BETWEEN consecutive stages is what attributes cost;
        # absolute free per stage leaves that arithmetic to the reader.
        probe = _probe_sequence((5000, 0, 0), (5000, 0, 0), (2000, 0, 0))
        census.begin("pp_to_tp", 0, probe=probe)
        census.mark("kv_pack")
        census.mark("gdn_state")
        line = census.end().format_line()
        self.assertIn("step-3000", line)

    def test_reset_drops_an_open_census(self):
        census.begin("pp_to_tp", 0, probe=_probe_sequence((5000, 0, 0)))
        census.reset()
        self.assertIsNone(census.active())


class TestTheLabelCannotReSeedTheOriginalError(unittest.TestCase):
    """A flip that ENDS lower than it started spent a PHASE STEP, not a
    transient. Conflating the two is precisely what put a withdrawn
    1.4-3.0 GiB "cutover cost" into three handoffs, so the instrument is
    pinned against re-seeding it."""

    def setUp(self):
        census.reset()

    def tearDown(self):
        census.reset()

    def test_a_flip_that_ends_at_its_low_reports_a_phase_step(self):
        # Shape measured on metal for tp_to_pp rank 0: free RISES at the
        # backing release, falls at the restore, and the flip ends there.
        probe = _probe_sequence(
            (6819, 0, 0),  # entry
            (10335, 0, 0),  # backing_release -- source pages handed back
            (5663, 0, 0),  # backing_restore -- destination committed
        )
        census.begin("tp_to_pp", 0, probe=probe)
        census.mark("backing_release")
        census.mark("backing_restore")
        line = census.end().format_line()
        self.assertIn("phase step", line)
        self.assertNotIn("transient", line)

    def test_a_genuine_dip_in_the_middle_is_still_a_transient(self):
        probe = _probe_sequence(
            (6000, 0, 0),
            (2000, 0, 0),  # a real dip
            (6000, 0, 0),  # given back
        )
        census.begin("pp_to_tp", 0, probe=probe)
        census.mark("gdn_state")
        line = census.end().format_line()
        self.assertIn("transient", line)
        self.assertNotIn("phase step", line)


if __name__ == "__main__":
    unittest.main()
