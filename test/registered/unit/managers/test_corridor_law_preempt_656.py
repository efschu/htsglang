"""#656: the corridor law is carried by the walk that takes the bytes.

WHAT THIS PROTECTS, AND WHY IT IS TWO MECHANISMS AND NOT ONE.

On 2026-08-12 a ``pp_to_tp`` cutover on PP1 entered with 3006 MiB free, drew a
2066 MiB transient through the arena's page-commit walk, and sat at **940 MiB
free for 1.5 s** -- 84 MiB under the 1024 MiB corridor law. Two independent
failures had to line up for that to become a silent breach:

  1. NOTHING FUNDED THE WALK. ``_mem_create_reclaiming`` already knew the
     remedy -- torch sits on reserved-but-unused blocks and ``empty_cache``
     returns them to the driver -- but its trigger was
     ``CUDA_ERROR_OUT_OF_MEMORY``, i.e. free memory reaching ZERO. The law
     floor is 1024 MiB above zero, so the walk crossed it long before anything
     was refused and the remedy never ran. The census recorded ``slack=1054``
     at the trough: the bytes to stay legal were there the whole time.

  2. NOTHING RECOGNISED THE CROSSING. The seam census sampled the exact NVML
     observable at every stage, named the 1024 MiB floor in its own docstring,
     and never compared the two -- and its summary line is emitted only AFTER
     the flip completes. The breach was found hours later in an external CSV.

So there are pins here for an ACTUATOR and for a RECOGNISER, and they fail
independently. A recogniser alone would have produced a well-documented
breach; an actuator alone would have produced a silent near-miss whose margin
nobody could audit.

EVERY LOAD-BEARING ASSERTION HAS A SIBLING ARM. The failure mode this corpus
keeps shipping is a mechanism that cannot fire being indistinguishable from
one that fired correctly, so each "it acts" pin is paired with an "it stays
out of the way" pin driven through the SAME function.
"""

import logging
import unittest
from unittest import mock

from sglang.srt.managers import phase_flip_seam_census as census

MIB = 1024 * 1024


def _probe_sequence(*samples):
    """A probe returning each (free, reserved, allocated) triple in turn.

    Sizes in MiB for legibility, converted here -- the same helper shape the
    sibling census pins use, so the two files read alike.
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


class SeamCensusRecognisesTheLaw(unittest.TestCase):
    """The recogniser: the census must READ the number it already holds."""

    def tearDown(self):
        census.reset()

    def test_a_walk_that_crosses_the_law_is_named_at_the_stage_that_crossed(self):
        # The real descent, in the real order: the walk ratchets down in
        # ~24 MiB commit steps and then sits flat at the trough for three
        # stages that neither allocate nor free.
        probe = _probe_sequence(
            (3006, 4000, 2946),  # entry
            (1048, 4000, 2946),  # last legal sample
            (1004, 4000, 2946),  # CROSSES
            (940, 4000, 2946),  # trough, held
            (1128, 4000, 2946),  # weights_refill recovers
        )
        with self.assertLogs(census.logger, level=logging.ERROR) as captured:
            c = census.begin("pp_to_tp", 1, probe=probe)
            census.mark("backing_restore_span")
            census.mark("backing_restore_span")
            census.mark("kv_write")
            census.mark("weights_refill")

        self.assertEqual(
            [lbl for lbl, _free in c.below_law], ["backing_restore_span", "kv_write"]
        )
        # ONE announcement, not one per stage: this runs on the no-return
        # path and 60 identical ERROR lines would be its own outage.
        self.assertEqual(len(captured.output), 1)
        joined = captured.output[0]
        self.assertIn("CORRIDOR LAW BROKEN", joined)
        self.assertIn("1004", joined)  # the stage that crossed, not the trough
        self.assertIn("1054", joined)  # the slack it could have spent

        line = c.format_line()
        self.assertIn("CORRIDOR LAW BROKEN", line)
        self.assertIn("deepest 940 MiB", line)

    def test_a_walk_that_stays_above_the_law_says_nothing_at_all(self):
        # SIBLING ARM. Same function, same stage labels, same depth ORDER --
        # only the level differs. Without this pin a check hard-wired to
        # "always report" would pass the test above.
        probe = _probe_sequence(
            (3006, 4000, 2946),
            (1600, 4000, 2946),
            (1104, 4000, 2946),  # 80 MiB above the floor: legal
            (1128, 4000, 2946),
        )
        c = census.begin("pp_to_tp", 1, probe=probe)
        census.mark("backing_restore_span")
        census.mark("backing_restore_span")
        census.mark("weights_refill")

        self.assertEqual(c.below_law, [])
        self.assertNotIn("CORRIDOR LAW BROKEN", c.format_line())
        # The trough is still REPORTED -- suppressing the breach marker must
        # not suppress the measurement it is derived from.
        self.assertIn("trough 1104 MiB", c.format_line())

    def test_a_probe_that_fails_mid_walk_does_not_reach_the_law_check(self):
        # A failed probe is recorded as a gap (-1) and must never be read as
        # "free memory is minus one megabyte", which is below every floor.
        probe = _probe_sequence((3006, 4000, 2946), None, (2000, 4000, 2946))
        c = census.begin("tp_to_pp", 2, probe=probe)
        census.mark("backing_restore_span")
        census.mark("done_ish")
        self.assertEqual(c.below_law, [])
        self.assertNotIn("CORRIDOR LAW BROKEN", c.format_line())

    def test_the_floor_is_the_law_floor_and_zero_disables_the_check(self):
        # The arming floor (1536) belongs to the admission gate and carries a
        # margin ON TOP of the law; reporting a breach against it would cry
        # wolf on every cutover that legally spends its margin.
        with mock.patch.dict("os.environ", {"SGLANG_CORRIDOR_LAW_FLOOR_MIB": "1024"}):
            self.assertEqual(census.law_floor_bytes(), 1024 * MIB)
        with mock.patch.dict("os.environ", {"SGLANG_CORRIDOR_LAW_FLOOR_MIB": "0"}):
            self.assertEqual(census.law_floor_bytes(), 0)
            probe = _probe_sequence((3006, 4000, 2946), (940, 4000, 2946))
            c = census.begin("pp_to_tp", 1, probe=probe)
            census.mark("kv_write")
            self.assertEqual(c.below_law, [])
        # A junk value must not disable the law by accident.
        with mock.patch.dict("os.environ", {"SGLANG_CORRIDOR_LAW_FLOOR_MIB": "nope"}):
            self.assertEqual(census.law_floor_bytes(), 1024 * MIB)


class ArenaPreemptsTheCrossing(unittest.TestCase):
    """The actuator: spend torch's cache BEFORE the commit that would cross."""

    def _backing(self):
        from sglang.srt.mem_cache import kv_vmm_backing

        return kv_vmm_backing

    def _torch_stub(self, free_mib, reserved_mib, allocated_mib):
        """A torch whose CUDA memory numbers are scripted.

        ``empty_cache`` records that it was called instead of doing anything,
        because what is under test is the DECISION to spend the cache, not
        torch's implementation of spending it.
        """
        calls = {"empty_cache": 0}

        class _CudaStub:
            @staticmethod
            def mem_get_info():
                return (free_mib * MIB, 20480 * MIB)

            @staticmethod
            def memory_reserved():
                return reserved_mib * MIB

            @staticmethod
            def memory_allocated():
                return allocated_mib * MIB

            @staticmethod
            def empty_cache():
                calls["empty_cache"] += 1

        stub = mock.MagicMock()
        stub.cuda = _CudaStub
        return stub, calls

    def test_a_commit_that_would_cross_the_law_spends_the_cache_first(self):
        mod = self._backing()
        # 1048 MiB free, a 24 MiB commit step -> 1024 exactly is the floor and
        # 1048-24 = 1024 is NOT below it, so push one step further: 1040-24.
        stub, calls = self._torch_stub(1040, 4000, 2946)  # slack 1054 MiB
        reclaimed = {"n": 0}

        def reclaim():
            reclaimed["n"] += 1
            return 0

        with mock.patch.object(mod, "torch", stub):
            with self.assertLogs(mod.logger, level=logging.WARNING) as captured:
                mod._corridor_preempt(24 * MIB, "cuMemCreate", reclaim)

        self.assertEqual(calls["empty_cache"], 1)
        self.assertEqual(reclaimed["n"], 1)
        self.assertIn("corridor law floor", captured.output[0])
        self.assertIn("1054", captured.output[0])

    def test_a_commit_with_room_to_spare_does_not_touch_the_cache(self):
        # SIBLING ARM, and the one that matters for cost: this path runs on
        # every page commit of every flip. If it spent the cache here it
        # would re-warm the allocator thousands of times per window.
        mod = self._backing()
        stub, calls = self._torch_stub(3006, 4000, 2946)
        with mock.patch.object(mod, "torch", stub):
            mod._corridor_preempt(24 * MIB, "cuMemCreate", None)
        self.assertEqual(calls["empty_cache"], 0)

    def test_a_crossing_with_no_slack_to_spend_leaves_the_evidence_alone(self):
        # SIBLING ARM on the OTHER precondition. An unfundable walk is a real
        # finding about the configuration's budget; churning empty_cache for
        # a handful of MiB would burn the allocator and hide it.
        mod = self._backing()
        stub, calls = self._torch_stub(1040, 3000, 2990)  # slack 10 MiB
        with mock.patch.object(mod, "torch", stub):
            mod._corridor_preempt(24 * MIB, "cuMemCreate", None)
        self.assertEqual(calls["empty_cache"], 0)

    def test_the_preemption_can_be_disabled_and_then_never_fires(self):
        mod = self._backing()
        stub, calls = self._torch_stub(100, 4000, 2946)  # deeply sub-law
        with mock.patch.dict("os.environ", {"SGLANG_CORRIDOR_LAW_FLOOR_MIB": "0"}):
            with mock.patch.object(mod, "torch", stub):
                mod._corridor_preempt(24 * MIB, "cuMemCreate", None)
        self.assertEqual(calls["empty_cache"], 0)

    def test_a_broken_probe_degrades_to_doing_nothing_not_to_raising(self):
        # This sits on the cutover's no-return path. It may cost a missing
        # reclaim; it may not cost the instance.
        mod = self._backing()
        stub = mock.MagicMock()
        stub.cuda.mem_get_info.side_effect = RuntimeError("driver went away")
        with mock.patch.object(mod, "torch", stub):
            mod._corridor_preempt(24 * MIB, "cuMemCreate", None)  # must not raise

    def test_a_reclaim_callback_that_raises_does_not_escape(self):
        mod = self._backing()
        stub, _calls = self._torch_stub(1040, 4000, 2946)

        def reclaim():
            raise RuntimeError("the park is gone")

        with mock.patch.object(mod, "torch", stub):
            with self.assertLogs(mod.logger, level=logging.WARNING):
                mod._corridor_preempt(24 * MIB, "cuMemCreate", reclaim)


class TheTwoWindowsReplayed(unittest.TestCase):
    """The corpus's own numbers, so a regression is recognisable as one.

    These are not synthetic: they are the entry/draw/trough triples the seam
    census recorded in the two acceptance windows that motivated this change.
    """

    def tearDown(self):
        census.reset()

    def test_s42_the_breach_is_reported_where_it_happened(self):
        probe = _probe_sequence((3006, 4210, 3156), (940, 4210, 3156))
        with self.assertLogs(census.logger, level=logging.ERROR):
            c = census.begin("pp_to_tp", 1, probe=probe)
            census.mark("kv_write")
        self.assertEqual(c.peak_bytes(), (3006 - 940) * MIB)  # the 2066 MiB draw
        self.assertTrue(c.below_law)

    def test_s38_the_green_window_stays_green_and_says_so(self):
        # SIBLING ARM drawn from the corpus rather than invented: s38 entered
        # at 3469 and troughed at 1083, 59 MiB clear. It must NOT be reported
        # as a breach -- if it were, the mechanism would be re-labelling the
        # baseline it is supposed to protect.
        probe = _probe_sequence((3469, 4210, 3156), (1083, 4210, 3156))
        c = census.begin("pp_to_tp", 1, probe=probe)
        census.mark("kv_write")
        self.assertEqual(c.peak_bytes(), (3469 - 1083) * MIB)  # the 2386 MiB draw
        self.assertEqual(c.below_law, [])
        self.assertNotIn("CORRIDOR LAW BROKEN", c.format_line())


if __name__ == "__main__":
    unittest.main()
