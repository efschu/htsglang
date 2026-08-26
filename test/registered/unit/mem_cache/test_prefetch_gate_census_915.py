"""#915: a prefetch that was never attempted left no trace at all.

THE SECOND HALF OF THE SAME ZERO. #914 answered "why did the match refuse".
It does not answer the question that follows immediately: a walk that matches
nothing SHOULD fall through to an L3 storage prefetch, so "the match refused"
and "no prefetch was attempted" are two different failures and only the first
was instrumented.

MEASURED, 0826 window R7. Prefetch was attempted on 264 of 675 census-sampled
match walks (138 with completed_local=0, 126 with completed_local=4096). The
other 411 declined inside `UnifiedRadixCache.prefetch_from_storage` and left
NO counter and NO log line. The gate is a three-term conjunction:

    eligible = (locally_eligible
                and prefetch_length >= self.prefetch_threshold
                and not self.cache_controller.prefetch_rate_limited())

so 411 declines are three unrelated verdicts wearing one boolean, and the
remedies point at three different files:

  anchor        the caller's local gate, `last_host_node.backuped`
                (scheduler.py:4933, which admits the ROOT on purpose -- so a
                fully-refused match does NOT decline here, and anyone assuming
                the mamba refusal starves the prefetch is guessing)
  too_short     fewer than prefetch_threshold (256) new tokens to fetch
  rate_limited  prefetch_tokens_occupied >= prefetch_capacity_limit, and that
                limit is `0.5 * mem_pool_host.size` (cache_controller.py:729)
                -- which across a phase flip is not one number. #905 measured
                the two host tiers at 703472 rows (PP) and 30518 (TP), 23x
                apart, putting the TP-phase budget at ~15259 tokens: under
                four prefetches of the 4096 this window actually completed.

That last one is a HYPOTHESIS with an arithmetic fit, not a finding, and it is
written here as such. The counter is what turns it into either on the next
boot. This ticket deliberately changes no behaviour: it records the verdict the
code was already reaching.

WHY IT IS NOT ARMED BEHIND AN ENV FLAG, unlike the #904 match census. That one
builds an object and walks validators a second time, so it pays for itself only
when asked. This is one integer increment on a path that already builds a
RadixKey and takes a host lock. And a gate that counts only when someone
remembered to arm it cannot answer "was it ever tried" -- which is the whole
question.
"""

import unittest

from sglang.srt.mem_cache.match_refusal_census import (
    PREFETCH_GATE_COUNTS,
    format_prefetch_gate,
    note_prefetch_gate,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _CleanCounts(CustomTestCase):
    def setUp(self):
        PREFETCH_GATE_COUNTS.clear()

    def tearDown(self):
        PREFETCH_GATE_COUNTS.clear()


class TestTheThreeDeclinesAreSeparated(_CleanCounts):
    def test_each_reason_is_counted_under_its_own_name(self):
        note_prefetch_gate("anchor")
        note_prefetch_gate("too_short")
        note_prefetch_gate("rate_limited")
        note_prefetch_gate("rate_limited")
        self.assertEqual(PREFETCH_GATE_COUNTS["anchor"], 1)
        self.assertEqual(PREFETCH_GATE_COUNTS["too_short"], 1)
        self.assertEqual(PREFETCH_GATE_COUNTS["rate_limited"], 2)

    def test_an_attempt_is_counted_too_so_the_denominator_is_local(self):
        """#873: a denominator reconstructed from a different log is how a
        narrowed candidate set gets read as a decomposition."""
        note_prefetch_gate(None)
        note_prefetch_gate("anchor")
        self.assertEqual(PREFETCH_GATE_COUNTS["attempted"], 1)

    def test_the_parts_sum_to_every_call(self):
        for reason in (None, None, "anchor", "too_short", "rate_limited", None):
            note_prefetch_gate(reason)
        total = sum(
            v for k, v in PREFETCH_GATE_COUNTS.items() if not k.endswith("_tokens")
        )
        self.assertEqual(total, 6)

    def test_tokens_are_tracked_apart_from_counts(self):
        note_prefetch_gate("rate_limited", 4096)
        note_prefetch_gate("rate_limited", 4096)
        self.assertEqual(PREFETCH_GATE_COUNTS["rate_limited"], 2)
        self.assertEqual(PREFETCH_GATE_COUNTS["rate_limited_tokens"], 8192)

    def test_zero_tokens_does_not_invent_a_token_key(self):
        note_prefetch_gate("anchor", 0)
        self.assertNotIn("anchor_tokens", PREFETCH_GATE_COUNTS)


class TestTheInstrumentCanSayItDidNotMeasure(_CleanCounts):
    """#829/INDIKATOR-GESETZ: an instrument that cannot say 'I did not measure'
    is indistinguishable from one that measured nothing."""

    def test_no_observation_is_stated_not_implied_as_zero(self):
        self.assertEqual(format_prefetch_gate(), "[#915 prefetch-gate] no observation")

    def test_a_recorded_verdict_produces_a_greppable_line(self):
        note_prefetch_gate("rate_limited", 4096)
        line = format_prefetch_gate()
        self.assertIn("[#915 prefetch-gate]", line)
        self.assertIn("rate_limited=1", line)
        self.assertIn("rate_limited_tokens=4096", line)


class TestTheGateIsWiredAndOrdered(CustomTestCase):
    """PRESENT-AND-VERDRAHTET. A counter nothing calls is the middle of the
    three delivery states and the most expensive to mistake for either end."""

    def _src(self):
        import inspect

        from sglang.srt.mem_cache import unified_radix_cache

        return inspect.getsource(
            unified_radix_cache.UnifiedRadixCache.prefetch_from_storage
        )

    def test_the_gate_records_its_verdict(self):
        self.assertIn("_note_prefetch_gate(reason", self._src())

    def test_all_three_reasons_are_reachable_from_the_gate(self):
        src = self._src()
        for reason in ('"anchor"', '"too_short"', '"rate_limited"'):
            self.assertIn(reason, src)

    def test_eligibility_is_derived_from_the_reason_not_computed_twice(self):
        """Two expressions of one rule drift; #747 records these very match
        lineages doing it. `eligible` must BE the reason's absence."""
        self.assertIn("eligible = reason is None", self._src())

    def test_the_first_failing_term_is_named_not_all_of_them(self):
        """A request can trip several. Summing them would double-count exactly
        the way refused_tokens_by_component is documented to."""
        src = self._src()
        self.assertIn("if not locally_eligible:", src)
        self.assertIn("elif prefetch_length < self.prefetch_threshold:", src)
        self.assertIn("elif self.cache_controller.prefetch_rate_limited():", src)

    def test_the_rate_limit_check_is_still_called_at_most_once(self):
        """It reads a live counter; calling it twice per gate would be a second
        reading of a moving quantity, and the two could disagree.

        COMMENT LINES ARE STRIPPED FIRST, and that is not incidental. Counting
        occurrences in raw source counts the prose too -- this very function
        carries a pre-existing comment naming `prefetch_rate_limited()` at
        :2674, which made the naive assertion read 2 and fail for a reason that
        has nothing to do with the code. Matching on prose to reach a verdict
        about code is the #908 substring defect; a test may not commit it
        either.
        """
        code = "\n".join(
            line
            for line in self._src().splitlines()
            if not line.lstrip().startswith("#")
        )
        self.assertEqual(code.count("prefetch_rate_limited()"), 1)


class TestBehaviourIsUnchanged(CustomTestCase):
    """This ticket is an instrument. It must not move a single prefetch."""

    def test_the_three_terms_are_the_same_three_as_before(self):
        src = TestTheGateIsWiredAndOrdered()._src()
        self.assertIn("locally_eligible", src)
        self.assertIn("self.prefetch_threshold", src)
        self.assertIn("prefetch_rate_limited", src)

    def test_the_symmetric_escape_is_untouched(self):
        """#580: under `symmetric` a locally ineligible rank must still enter
        the collective, or gloo aborts the peer that posted the vote alone."""
        self.assertIn(
            "if not eligible and not symmetric:", TestTheGateIsWiredAndOrdered()._src()
        )


if __name__ == "__main__":
    unittest.main()
