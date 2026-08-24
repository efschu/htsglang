"""#846 (#831 O-1) -- a remembered diagnostic figure must say how old it is.

THE DEFECT, verified in the tree before this was written.
``_staging_affordable`` (phase_flip_runtime.py:7434) remembers three figures
for a census that runs later and cannot re-measure them:

    :7495  self._last_staging_bytes         -- always, on every probe
    :7512  self._last_cache_promised_bytes  -- ONLY inside the reclaim branch
    :7513  self._last_cache_delivered_bytes -- ONLY inside the reclaim branch

``_funding_post_census`` (:8453) reads the last two and states its contract in
its own comment:

    "None when no reclaim was attempted in this pass, which law 2 reads as
     'unobserved -- trust it once', so a refusal that never tried the cache is
     priced exactly as it was before."

**That contract is unreachable after the first reclaim.** Nothing in the module
ever assigns those attributes back to ``None`` -- verified,
``grep -c '_last_cache_delivered_bytes = None'`` is 0 -- and the write sits
inside ``if cached_free > 0 and (...)``. So from the first reclaim in a process
onward the pair is never None again, and every later pass that attempts no
reclaim quotes an EARLIER pass's measurement as though it were this one's.

WHAT IS AND IS NOT AT STAKE, because it decides the size of this fix.
``_funding_post_census`` returns a STRING (``-> str``, and its except arm
returns ``""`` because "a census must not raise"). The stale pair therefore
mis-states a funding verdict in a LOG LINE; it does not gate a flip. This is
the "#831 O-1" item, and it is diagnostic -- which is exactly why the remedy
here is to make the age visible and to change no verdict at all.

It is also the failure class that has cost this strand two passes: an
indicator that cannot say how old it is (#843's decline line, logged at DEBUG
on an INFO boot; #833's ticket title asserting a mechanism the tree had already
refuted). A number that outlives the measurement it came from and does not say
so is the same defect wearing a third coat.

SCOPE, stated so the next reader does not have to infer it: this ships the
COUNTER, not an actuator. The census keeps taking exactly the branch it took
before; it only now says how old the figure it quoted is. Whether a stale
reading should instead be treated as unobserved -- i.e. whether the documented
contract should be restored -- is a change to what the census asserts, and it
belongs to whoever owns #828's law 2. #777 made the same split deliberately
("built the counter and deliberately left the actuator to the planner") and it
was the right call there.
"""

import unittest

from sglang.srt.managers.phase_flip_runtime import (
    seam_probe_reading_age,
    seam_probe_age_phrase,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

# ~1s: two pure functions, no torch, no scheduler, no accelerator.
register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestSeamProbeReadingAge(CustomTestCase):
    def test_a_figure_measured_in_this_pass_is_age_zero(self):
        self.assertEqual(seam_probe_reading_age(7, 7), 0)

    def test_a_figure_from_an_earlier_pass_reports_its_distance(self):
        self.assertEqual(seam_probe_reading_age(9, 7), 2)

    def test_never_measured_is_None_not_zero(self):
        """The distinction the whole item is about.

        ``None`` must not collapse into 0: "never measured" and "measured just
        now" are the two readings a census most needs to tell apart, and
        ``getattr(self, '_last_...', 0)`` collapsing them is how a default
        reads as a measurement.
        """
        self.assertIsNone(seam_probe_reading_age(9, None))

    def test_no_probe_has_run_at_all(self):
        self.assertIsNone(seam_probe_reading_age(None, None))
        self.assertIsNone(seam_probe_reading_age(None, 3))

    def test_a_stamp_from_the_future_is_not_negative(self):
        """Defensive: a counter reset (or a stand-in that never increments)
        must not print a negative age, which would read as a corrupt census
        rather than as the missing counter it is."""
        self.assertIsNone(seam_probe_reading_age(2, 5))


class TestSeamProbeAgePhrase(CustomTestCase):
    """The phrase is what reaches the log, so it is pinned too."""

    def test_this_pass_says_so_explicitly(self):
        self.assertEqual(seam_probe_age_phrase(0), "measured this pass")

    def test_one_pass_ago_is_singular(self):
        self.assertEqual(seam_probe_age_phrase(1), "measured 1 pass ago")

    def test_older_readings_are_plural_and_named(self):
        self.assertEqual(seam_probe_age_phrase(4), "measured 4 passes ago")

    def test_unmeasured_never_prints_a_number(self):
        self.assertEqual(seam_probe_age_phrase(None), "never measured")

    def test_the_phrase_never_returns_empty(self):
        """An empty phrase would restore the exact silence this fixes."""
        for age in (None, 0, 1, 2, 99):
            self.assertTrue(seam_probe_age_phrase(age).strip())


class TestTheAgeIsActuallyWired(CustomTestCase):
    """Pure functions nobody calls would be the same silence in a new place.

    Source-level, because driving ``_funding_post_census`` for real needs a
    scheduler, a rung and a CUDA probe, and a stub of those would pin the stub.
    What must be true is the WIRING, and that is what these read.
    """

    def _src(self, name):
        import inspect

        from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

        return inspect.getsource(getattr(PhaseFlipRuntime, name))

    def test_the_probe_pass_is_counted_and_the_reclaim_figures_stamped(self):
        src = self._src("_staging_affordable")
        self.assertIn(
            "_seam_probe_seq", src, "the probe pass must be counted somewhere"
        )
        self.assertIn(
            "_last_cache_bytes_seq",
            src,
            "the reclaim figures must be stamped where they are written; "
            "unstamped, the census cannot tell this pass's measurement from "
            "one several passes old.",
        )

    def test_the_census_states_the_age_it_priced_from(self):
        src = self._src("_funding_post_census")
        self.assertIn("seam_probe_reading_age", src)
        self.assertIn(
            "seam_probe_age_phrase",
            src,
            "the age must reach the RETURNED STRING -- computing it and not "
            "printing it is the #843 shape (a decline observable only at "
            "DEBUG on an INFO boot).",
        )

    def test_the_census_still_returns_a_string_and_cannot_raise(self):
        """The one property this ticket must not break.

        `_funding_post_census` is a census: its contract is that it returns
        text and never raises. An age stamp that could raise would turn a
        diagnostic into an outage.
        """
        src = self._src("_funding_post_census")
        self.assertIn("except Exception", src)
        self.assertIn('return ""', src)


if __name__ == "__main__":
    unittest.main()
