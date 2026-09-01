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
    BOUND_MIN_COVERAGE,
    RefillLegTiming,
    refill_bound_phrase,
)
from sglang.test.test_utils import CustomTestCase


def _timing(**kw) -> RefillLegTiming:
    """A leg timing whose LEG TOTAL is supplied, defaulting to full coverage.

    #1082 gave ``refill_bound_phrase`` a denominator: the two halves it weighs
    must explain at least ``BOUND_MIN_COVERAGE`` of the LEG before it may name
    a winner. Every fixture in this file is about the read-vs-wait split, so
    each one declares a leg total just large enough to be fully covered, and
    the coverage rule itself is exercised in TestItRefusesOnThinCoverage below.
    """
    kw.setdefault("leg_total_s", float(kw.get("read_s", 0.0)) + float(kw.get("h2d_wait_s", 0.0)))
    return RefillLegTiming(**kw)



class TestTheBoundIsNamed(CustomTestCase):
    def test_a_read_dominated_leg_reads_storage_bound(self):
        t = _timing(read_s=9.0, h2d_wait_s=0.5, drain_s=0.1, chunks=512)
        phrase = refill_bound_phrase(t)
        self.assertIn("STORAGE-BOUND", phrase)
        self.assertNotIn("LINK-BOUND", phrase)

    def test_a_wait_dominated_leg_reads_link_bound(self):
        t = _timing(read_s=0.5, h2d_wait_s=9.0, drain_s=0.1, chunks=512)
        phrase = refill_bound_phrase(t)
        self.assertIn("LINK-BOUND", phrase)
        self.assertNotIn("STORAGE-BOUND", phrase)

    def test_the_phrase_carries_the_numbers_not_just_the_verdict(self):
        # A verdict with no figures behind it is the shape that made the
        # aggregate rate unusable in the first place.
        t = _timing(read_s=9.0, h2d_wait_s=0.5, drain_s=0.25, chunks=511)
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
        t = _timing(read_s=0.0, h2d_wait_s=0.0, chunks=8)
        self.assertIn("unattributed", refill_bound_phrase(t))

    def test_an_even_split_is_MIXED_and_names_neither(self):
        t = _timing(read_s=5.0, h2d_wait_s=5.0, chunks=64)
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
            t = _timing(read_s=read, h2d_wait_s=wait, chunks=1)
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
        t = _timing(read_s=9.8, h2d_wait_s=0.4, drain_s=0.1, chunks=512)
        self.assertIn("STORAGE-BOUND", refill_bound_phrase(t))

    def test_a_link_bound_tp_to_pp_would_indict_the_h2d_path(self):
        # Same leg, same wall time, opposite attribution.
        t = _timing(read_s=0.4, h2d_wait_s=9.8, drain_s=0.1, chunks=512)
        self.assertIn("LINK-BOUND", refill_bound_phrase(t))


class TestItRefusesOnThinCoverage(CustomTestCase):
    """#1082: the verdict needs a DENOMINATOR, and it did not have one.

    THE DEFECT, measured. ``read_s`` has exactly one writer in the tree, inside
    ``_staged_file_refill``. The flip's weights leg does not go through that
    function -- it goes through ``rotate_arena`` -- so on that path ``read_s``
    is always 0.0, ``read_share`` is always 0.0, and the verdict is always
    LINK-BOUND. It printed exactly that on boot_855_1078spec: `LINK-BOUND (read 0.000s /
    h2d-wait 0.016s over 429 chunk(s))` on a leg that took 63.931 s. A verdict
    with one reachable value is not a verdict.
    """

    #: The real line from boot_855_1078spec, PP0 pp_to_tp.
    MEASURED_READ_S = 0.000
    MEASURED_WAIT_S = 0.016
    MEASURED_LEG_S = 63.931

    def test_the_measured_specimen_is_REFUSED_not_named(self):
        t = RefillLegTiming(
            read_s=self.MEASURED_READ_S,
            h2d_wait_s=self.MEASURED_WAIT_S,
            chunks=429,
            leg_total_s=self.MEASURED_LEG_S,
        )
        phrase = refill_bound_phrase(t)
        self.assertIn("REFUSED", phrase)
        self.assertNotIn("LINK-BOUND", phrase)
        self.assertNotIn("STORAGE-BOUND", phrase)

    def test_the_refusal_prints_the_coverage_that_caused_it(self):
        # A refusal that does not say HOW thin the coverage was sends the next
        # reader back to the same guess.
        t = RefillLegTiming(
            read_s=self.MEASURED_READ_S,
            h2d_wait_s=self.MEASURED_WAIT_S,
            chunks=429,
            leg_total_s=self.MEASURED_LEG_S,
        )
        phrase = refill_bound_phrase(t)
        self.assertIn("63.931", phrase)
        self.assertIn("%", phrase)

    def test_a_leg_with_no_total_is_refused_rather_than_guessed(self):
        # THE SILENT-FALLBACK CAN-FAIL. Making the coverage rule conditional on
        # a total being present would restore the constant verdict for every
        # caller that forgets to set one -- the #742 silently-inert class. A
        # missing denominator must therefore read as "cannot judge".
        t = RefillLegTiming(read_s=0.0, h2d_wait_s=9.0, chunks=64)
        phrase = refill_bound_phrase(t)
        self.assertIn("unattributed", phrase)
        self.assertNotIn("LINK-BOUND", phrase)

    def test_well_covered_legs_still_get_a_real_verdict(self):
        # The rule must not swallow the cases the instrument was built for:
        # both named bounds stay REACHABLE above the coverage floor.
        storage = RefillLegTiming(
            read_s=9.0, h2d_wait_s=0.5, chunks=512, leg_total_s=10.0
        )
        link = RefillLegTiming(
            read_s=0.5, h2d_wait_s=9.0, chunks=512, leg_total_s=10.0
        )
        self.assertIn("STORAGE-BOUND", refill_bound_phrase(storage))
        self.assertIn("LINK-BOUND", refill_bound_phrase(link))

    def test_the_floor_is_where_the_constant_says_it_is(self):
        # Pin the threshold to the named constant rather than to a literal, so
        # moving the constant moves the behaviour and not just the docstring.
        just_under = RefillLegTiming(
            read_s=0.0,
            h2d_wait_s=BOUND_MIN_COVERAGE * 10.0 - 0.01,
            chunks=8,
            leg_total_s=10.0,
        )
        just_over = RefillLegTiming(
            read_s=0.0,
            h2d_wait_s=BOUND_MIN_COVERAGE * 10.0 + 0.01,
            chunks=8,
            leg_total_s=10.0,
        )
        self.assertIn("REFUSED", refill_bound_phrase(just_under))
        self.assertIn("LINK-BOUND", refill_bound_phrase(just_over))


class TestTheRotationPathReachesTheRefusal(CustomTestCase):
    """Reachability, not just arithmetic: the executor that HAS the defect must
    itself produce the refusal, or the rule is desk-only (#1082).

    This is the Kompensator-Erreichbarkeit check -- a bound proven only on a
    hand-built fixture says nothing about the path the defect lives on.
    """

    def test_a_real_cpu_rotation_refuses_instead_of_saying_LINK_BOUND(self):
        import torch

        from sglang.srt.mem_cache.read_buffer_pool import ReadBufferPool
        from sglang.srt.model_executor.rotation_executor import (
            TorchRotationOps,
            rotate_arena,
        )
        from sglang.srt.model_executor.weights_arena import (
            _CHECKSUM_BYTES,
            uint8_checksum,
        )

        chunk, depth = 4096, 4
        incoming, outgoing = 9 * chunk + 137, 11 * chunk + 41
        span = max(incoming, outgoing)
        arena = torch.zeros(span, dtype=torch.uint8)
        image = torch.zeros(span + _CHECKSUM_BYTES, dtype=torch.uint8)
        g = torch.Generator().manual_seed(22)
        image[:incoming] = torch.randint(
            0, 256, (incoming,), generator=g, dtype=torch.uint8
        )
        image[incoming : incoming + _CHECKSUM_BYTES] = torch.tensor(
            [uint8_checksum(image[:incoming])], dtype=torch.int64
        ).view(torch.uint8)
        ring = ReadBufferPool(
            name="test_1082_ring",
            flag="--test",
            capacity=depth,
            page_bytes=chunk,
            factory=lambda: torch.empty(chunk, dtype=torch.uint8),
            register=False,
        )
        timing = RefillLegTiming()
        rotate_arena(
            arena=arena,
            host_image=image,
            incoming_bytes=incoming,
            outgoing_bytes=outgoing,
            chunk_bytes=chunk,
            depth=depth,
            ring=ring,
            ops=TorchRotationOps(),
            timing=timing,
        )
        # The executor now hands the phrase a denominator...
        self.assertGreater(timing.leg_total_s, 0.0)
        # ...and on this path read_s still has no writer, which is the finding.
        self.assertEqual(timing.read_s, 0.0)
        self.assertNotIn("LINK-BOUND", refill_bound_phrase(timing))


if __name__ == "__main__":
    unittest.main()
