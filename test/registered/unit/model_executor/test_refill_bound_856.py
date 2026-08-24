"""#856(a): the refill leg says WHICH HALF was slow, or says it cannot.

THE DEFECT. The weights-arena refill leg is 91% of a `tp_to_pp` flip -- W25's
own seam census reads `worst 'refill_highwater->weights_refill' 9516.2 ms
(91% of the walk)` against a 10466.8 ms total -- and it reported ONE aggregate
MiB/s. The leg is a synchronous `preadv` pipelined against an async H2D DMA,
so that aggregate is `min(read_rate, h2d_rate)` and cannot say which bound it
hit.

That mattered immediately. W25 measured, on the SAME rank and within 2.7% of
the same bytes:

    pp_to_tp   15925.8 MiB   3214-3915 MiB/s   4.07-4.96 s
    tp_to_pp   16362.7 MiB   1351-1723 MiB/s   9.50-12.11 s

a 2.5x rate gap that two independent readers could not attribute from the
code. The obvious readings were all checked and all fail: both directions
take `_staged_file_refill` (`SGLANG_PHASE_FLIP_REFILL_STAGED` defaults True);
both `#802` fallback warnings appear ZERO times in the 3.45 MB capture against
9 `FILE-BACKED` registrations; the O_DIRECT alignment cliff cannot fire
because offsets are 32 MiB multiples of a 4096 alignment; and #802's own
fault-path signature (rank rates CONVERGE) is absent -- W25's rates diverge
with the link in BOTH directions.

So the instrument was the thing missing, not the mechanism: one number with
several meanings, the #851 class, inside the dominant term of the seam.

THE CAN-FAIL DIRECTION, and it is the whole risk. A phrase that always names
a bound would satisfy every "it says something" assertion while being exactly
as useless as the aggregate it replaces. "unattributed" and "MIXED" must
therefore be REACHABLE, and they are asserted here as first-class outcomes.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest

from sglang.srt.model_executor.weights_arena import (
    RefillLegTiming,
    refill_bound_phrase,
)
from sglang.test.test_utils import CustomTestCase


class TestTheBoundIsNamed(CustomTestCase):
    def test_a_read_dominated_leg_reads_storage_bound(self):
        t = RefillLegTiming(read_s=9.0, h2d_wait_s=0.5, drain_s=0.1, chunks=512)
        phrase = refill_bound_phrase(t)
        self.assertIn("STORAGE-BOUND", phrase)
        self.assertNotIn("LINK-BOUND", phrase)

    def test_a_wait_dominated_leg_reads_link_bound(self):
        t = RefillLegTiming(read_s=0.5, h2d_wait_s=9.0, drain_s=0.1, chunks=512)
        phrase = refill_bound_phrase(t)
        self.assertIn("LINK-BOUND", phrase)
        self.assertNotIn("STORAGE-BOUND", phrase)

    def test_the_phrase_carries_the_numbers_not_just_the_verdict(self):
        # A verdict with no figures behind it is the shape that made the
        # aggregate rate unusable in the first place.
        t = RefillLegTiming(read_s=9.0, h2d_wait_s=0.5, drain_s=0.25, chunks=511)
        phrase = refill_bound_phrase(t)
        self.assertIn("9.000", phrase)
        self.assertIn("0.500", phrase)
        self.assertIn("511", phrase)
        self.assertIn("0.250", phrase)


class TestItRefusesToGuess(CustomTestCase):
    """The can-fail direction: both non-verdicts must be reachable."""

    def test_an_uninstrumented_leg_is_unattributed(self):
        self.assertIn("unattributed", refill_bound_phrase(None))
        self.assertIn("unattributed", refill_bound_phrase(RefillLegTiming()))

    def test_a_leg_with_no_time_accounted_is_unattributed(self):
        t = RefillLegTiming(read_s=0.0, h2d_wait_s=0.0, chunks=8)
        self.assertIn("unattributed", refill_bound_phrase(t))

    def test_an_even_split_is_MIXED_and_names_neither(self):
        t = RefillLegTiming(read_s=5.0, h2d_wait_s=5.0, chunks=64)
        phrase = refill_bound_phrase(t)
        self.assertIn("MIXED", phrase)
        self.assertNotIn("STORAGE-BOUND", phrase)
        self.assertNotIn("LINK-BOUND", phrase)

    def test_the_verdict_bands_do_not_overlap_or_leave_a_hole(self):
        # Sweep the whole read-share range: every point must land in exactly
        # one of the three verdicts. A band that overlapped would make the
        # phrase ambiguous; a hole would make it empty.
        for i in range(0, 101):
            read = float(i)
            wait = float(100 - i)
            t = RefillLegTiming(read_s=read, h2d_wait_s=wait, chunks=1)
            phrase = refill_bound_phrase(t)
            hits = sum(
                (
                    "STORAGE-BOUND" in phrase,
                    "LINK-BOUND" in phrase,
                    "MIXED" in phrase,
                )
            )
            with self.subTest(read_share=i):
                self.assertEqual(hits, 1, f"{i}: {phrase}")


class TestTheW25GapWouldBeAttributable(CustomTestCase):
    """The point of the ticket, stated as an assertion.

    These are the two shapes the next proof window can produce for the SAME
    leg. Whichever it emits names the root that desk analysis could not.
    """

    def test_a_storage_bound_tp_to_pp_would_indict_the_read_path(self):
        # 16362.7 MiB at ~1585 MiB/s = ~10.3 s. If nearly all of it is preadv,
        # the pool/ARC is the bound and the fix is on the storage side.
        t = RefillLegTiming(read_s=9.8, h2d_wait_s=0.4, drain_s=0.1, chunks=512)
        self.assertIn("STORAGE-BOUND", refill_bound_phrase(t))

    def test_a_link_bound_tp_to_pp_would_indict_the_h2d_path(self):
        # Same leg, same wall time, opposite attribution.
        t = RefillLegTiming(read_s=0.4, h2d_wait_s=9.8, drain_s=0.1, chunks=512)
        self.assertIn("LINK-BOUND", refill_bound_phrase(t))


if __name__ == "__main__":
    unittest.main()
