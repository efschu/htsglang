"""#739: the detector must not alarm on a box that is visibly prefilling.

THE FALSE-POSITIVE SPECIMEN, 20:08. A ~500k-token backlog was chunking at
``chunked_prefill_size=512``. Prefill batches ran visibly the whole time, and
no request reached a first token for 135 s. The #699 detector alarmed, because
its only progress clock is the FIRST-TOKEN clock: a mega-prefill produces no
first token for minutes by construction, so "no first token for 135 s" is
exactly what healthy progress looks like here.

The shape is indistinguishable from a real wedge on the three numbers #699
reads (queued > 0, running == 0, large first-token age) -- during chunked
prefill the request stays in the waiting queue and nothing is decoding. So a
SECOND progress signal is required; the first three numbers cannot separate
these cases even in principle.

THE SIGNAL IS AN EVENT TIMESTAMP, NOT A COUNTER. The obvious candidate was a
delta of the pending-token count, but that is exactly the counter #731 shows
double-billed (51369 -> 102307), and a fix to it is in flight on another lane.
A detector keyed to a counter under repair would inherit its noise and its
rebases. Instead this stamps a clock when a chunked request's middle chunk
COMPLETES -- one event per completed chunk, the same shape as the existing
first-token clock, with no shared counter and so no coupling to #731.

THE RULE: alarm only when NEITHER a first token NOR a prefill chunk landed in
the window. Either one is progress.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.managers.scheduler_components.invariant_checker import (
    ADMISSION_WEDGE_SECONDS,
    admission_wedge_verdict,
)
from sglang.test.test_utils import CustomTestCase


class TestTheMegaPrefillSpecimenIsSilent(CustomTestCase):
    """The new red case."""

    def test_the_2008_specimen_does_not_alarm(self):
        """135 s without a first token, but chunks are landing."""
        alarm, detail = admission_wedge_verdict(
            1, 0, 135.0, seconds_since_prefill_progress=0.4
        )
        self.assertFalse(alarm, detail)
        self.assertIn("prefill", detail.lower())

    def test_it_stays_silent_across_the_whole_mega_prefill(self):
        """A 500k backlog at 512 tokens/chunk is ~1000 chunks. Sampled the
        whole way down, with a chunk landing between samples every time, the
        detector must never alarm -- one alarm in that sweep is the bug."""
        for age in range(0, 600, 15):
            with self.subTest(first_token_age=age):
                alarm, _ = admission_wedge_verdict(
                    1, 0, float(age), seconds_since_prefill_progress=0.7
                )
                self.assertFalse(alarm)

    def test_a_chunk_just_inside_the_window_still_counts_as_progress(self):
        just_inside = ADMISSION_WEDGE_SECONDS * 0.99
        alarm, _ = admission_wedge_verdict(
            1, 0, 900.0, seconds_since_prefill_progress=just_inside
        )
        self.assertFalse(alarm)


class TestTheRealWedgesStillFire(CustomTestCase):
    """A wedge is the case where NOTHING progressed. All 17 true hits have a
    stale prefill clock too, because nothing was prefilling either."""

    def test_the_31_second_specimen_still_alarms(self):
        alarm, detail = admission_wedge_verdict(
            1, 0, 31.64, seconds_since_prefill_progress=31.64
        )
        self.assertTrue(alarm, detail)
        self.assertIn("ADMISSION-WEDGE", detail)

    def test_a_stalled_prefill_clock_does_not_excuse_a_wedge(self):
        """The suppression must be keyed to RECENT prefill, not to the mere
        presence of the signal -- otherwise wiring the clock would silence the
        detector permanently, which is the failure mode that matters."""
        stale = ADMISSION_WEDGE_SECONDS * 3
        alarm, _ = admission_wedge_verdict(
            1, 0, stale, seconds_since_prefill_progress=stale
        )
        self.assertTrue(alarm)

    def test_a_chunk_just_outside_the_window_does_not_suppress(self):
        just_outside = ADMISSION_WEDGE_SECONDS * 1.01
        alarm, _ = admission_wedge_verdict(
            1, 0, 900.0, seconds_since_prefill_progress=just_outside
        )
        self.assertTrue(alarm)

    def test_the_sampled_sequence_still_alarms_once_past_threshold(self):
        seen = [
            admission_wedge_verdict(1, 0, t, seconds_since_prefill_progress=t)[0]
            for t in range(0, 34, 2)
        ]
        self.assertTrue(seen[-1])


class TestTheDefaultPathIsUnchanged(CustomTestCase):
    """Omitting the new argument must reproduce #699 exactly, so the existing
    specimen suite is a regression test rather than a rewritten one."""

    def test_absent_signal_reproduces_the_old_verdict(self):
        for queued in (0, 1, 3):
            for running in (0, 1, 2):
                for age in (0.0, 5.0, 31.64, 135.0):
                    with self.subTest(q=queued, r=running, age=age):
                        old = admission_wedge_verdict(queued, running, age)
                        new = admission_wedge_verdict(
                            queued, running, age, seconds_since_prefill_progress=None
                        )
                        self.assertEqual(old, new)

    def test_a_serving_box_is_still_silent_regardless_of_the_new_signal(self):
        for prefill_age in (None, 0.1, 999.0):
            with self.subTest(prefill_age=prefill_age):
                alarm, _ = admission_wedge_verdict(
                    3, 2, 120.0, seconds_since_prefill_progress=prefill_age
                )
                self.assertFalse(alarm)

    def test_an_empty_queue_is_still_silent(self):
        alarm, _ = admission_wedge_verdict(
            0, 0, 999.0, seconds_since_prefill_progress=999.0
        )
        self.assertFalse(alarm)


class TestTheDetailExplainsWhichClockSpokeUp(CustomTestCase):
    def test_suppression_names_the_prefill_age(self):
        _, detail = admission_wedge_verdict(
            1, 0, 135.0, seconds_since_prefill_progress=0.4
        )
        self.assertIn("0.4", detail)
        self.assertIn("135", detail)

    def test_the_alarm_reports_both_clocks(self):
        _, detail = admission_wedge_verdict(
            1, 0, 40.0, seconds_since_prefill_progress=40.0
        )
        self.assertIn("40.0", detail)
        self.assertIn("prefill", detail.lower())


class TestTheWiringCarriesTheSecondClock(CustomTestCase):
    """Source pins: the verdict is only as good as what reaches it."""

    def test_scheduler_has_a_prefill_progress_clock(self):
        import inspect

        from sglang.srt.managers.scheduler import Scheduler

        src = inspect.getsource(Scheduler)
        self.assertIn("last_prefill_progress_time", src)
        self.assertIn("def note_prefill_progress", src)

    def test_the_check_passes_the_prefill_age(self):
        import inspect

        from sglang.srt.managers.scheduler_components.invariant_checker import (
            check_admission_wedge_once,
        )

        src = inspect.getsource(check_admission_wedge_once)
        self.assertIn("last_prefill_progress_time", src)
        self.assertIn("seconds_since_prefill_progress", src)

    def test_the_chunk_completion_stamps_it(self):
        """Stamped where a middle chunk is retired, NOT on a forward pass --
        the same distinction #699 made for the first-token clock."""
        import inspect

        from sglang.srt.managers.scheduler_components import batch_result_processor

        src = inspect.getsource(batch_result_processor)
        # The CALL, not the field declaration: asserting the bare name passed
        # with every call site deleted, because the `record_prefill_progress:
        # Callable` default still matched it. A pin must state the invariant.
        self.assertEqual(src.count("self.record_prefill_progress()"), 2)
        # Both middle-chunk retirement sites stamp: the generation path and
        # the embedding/reward path. One of them alone leaves a blind arm.
        self.assertEqual(src.count("inflight_middle_chunks -= 1"), 2)


if __name__ == "__main__":
    unittest.main()
