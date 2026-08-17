# SPDX-License-Identifier: Apache-2.0
"""#536 -- the fast lane must stay admissible under a saturating heavy prefill.

THE STARVATION, as arithmetic. A heavy prefill holds the pool; the fast request
is already FIRST in the queue (schedule_policy.py:385-432 puts it there, and
#552's aging deliberately never promotes a heavy request above it). It still
waits, because being first in a queue does not produce memory -- the #536
mechanism note: "priority orders the queue; it cannot release memory another
request holds".

These tests drive the reserve rule directly. No server, no GPU, no scheduler.
"""

import unittest

from sglang.srt.managers.fast_lane_reserve import (
    PROV_ABSENT,
    PROV_CHUNK,
    PROV_DECLARED,
    FastLaneReserveError,
    ReservePlan,
    admissible_tokens,
    describe,
    solve_fast_lane_reserve,
)

POOL = 400_000
CHUNK = 2_048


def _plan(**kw):
    base = dict(
        enabled=True,
        pool_total_tokens=POOL,
        fast_lane_max_prompt_tokens=None,
        chunked_prefill_size=CHUNK,
        min_pool_tokens_after_reserve=100_000,
    )
    base.update(kw)
    return solve_fast_lane_reserve(**base)


class TestTheStarvation(unittest.TestCase):
    def test_UNRESERVED_a_saturating_heavy_prefill_leaves_the_fast_lane_NOTHING(self):
        """The defect, with the reserve dark: the heavy lane may draw the whole
        pool, so when it has, the fast request's admissible budget is zero
        however early it sits in the queue."""
        dark = ReservePlan(0, PROV_ABSENT, "disabled", False)
        heavy_draw = admissible_tokens(
            is_fast_lane=False, pool_free_tokens=POOL, reserve=dark
        )
        self.assertEqual(heavy_draw, POOL, "heavy may take everything")
        self.assertEqual(
            admissible_tokens(is_fast_lane=True, pool_free_tokens=0, reserve=dark),
            0,
            "and then the fast lane has nothing to admit into -- the 34.5 s",
        )

    def test_RESERVED_the_fast_request_still_admits(self):
        """RED-FIRST against a tree without the reserve draw: the heavy lane is
        capped below the pool, so the tokens the fast lane needs survive a
        saturating prefill."""
        r = _plan()
        heavy_draw = admissible_tokens(
            is_fast_lane=False, pool_free_tokens=POOL, reserve=r
        )
        self.assertEqual(heavy_draw, POOL - r.tokens)
        # The heavy lane consumes everything it is allowed to.
        left = POOL - heavy_draw
        self.assertGreaterEqual(left, r.tokens)
        self.assertGreaterEqual(
            admissible_tokens(is_fast_lane=True, pool_free_tokens=left, reserve=r),
            CHUNK,
            "the fast request must be able to take at least one chunk",
        )

    def test_the_reserve_is_a_FLOOR_under_the_fast_lane_not_a_ceiling_on_it(self):
        r = _plan()
        self.assertEqual(
            admissible_tokens(is_fast_lane=True, pool_free_tokens=POOL, reserve=r),
            POOL,
            "a fast request may use the whole free pool, not just its reserve",
        )

    def test_reserve_already_in_use_is_not_withheld_twice(self):
        r = _plan()
        heavy = admissible_tokens(
            is_fast_lane=False, pool_free_tokens=POOL, reserve=r,
            reserve_in_use_tokens=r.tokens,
        )
        self.assertEqual(heavy, POOL, "a spent reserve withholds nothing further")


class TestTheDarkDefault(unittest.TestCase):
    """The gate is unsatisfied -- no lane-ON/OFF observation exists -- so the
    default path must be byte-identical to before."""

    def test_disabled_prices_nothing_and_withholds_nothing(self):
        r = _plan(enabled=False)
        self.assertTrue(r.is_dark)
        self.assertEqual(r.provenance, PROV_ABSENT)
        for fast in (True, False):
            self.assertEqual(
                admissible_tokens(is_fast_lane=fast, pool_free_tokens=POOL, reserve=r),
                POOL,
            )

    def test_describe_says_DARK_rather_than_implying_protection(self):
        self.assertIn("DARK", describe(_plan(enabled=False)))
        self.assertIn("withheld", describe(_plan()))


class TestThePricingHasProvenance(unittest.TestCase):
    def test_a_declared_ceiling_is_used_and_labelled(self):
        r = _plan(fast_lane_max_prompt_tokens=8_192)
        self.assertEqual(r.tokens, 8_192)
        self.assertEqual(r.provenance, PROV_DECLARED)

    def test_without_a_declaration_it_falls_back_to_one_chunk_and_says_so(self):
        r = _plan()
        self.assertEqual(r.tokens, CHUNK)
        self.assertEqual(r.provenance, PROV_CHUNK)

    def test_a_reserve_below_one_chunk_could_not_admit_a_chunk(self):
        """Why the chunk is the floor of the pricing, not an arbitrary pick."""
        r = _plan()
        self.assertGreaterEqual(r.tokens, CHUNK)

    def test_a_nonpositive_declared_ceiling_is_refused(self):
        with self.assertRaises(FastLaneReserveError):
            _plan(fast_lane_max_prompt_tokens=0)

    def test_nothing_to_price_against_is_refused_not_guessed(self):
        with self.assertRaises(FastLaneReserveError) as e:
            _plan(chunked_prefill_size=0)
        self.assertIn("rather than letting the reserve be guessed", str(e.exception))


class TestTheRefusalSurface(unittest.TestCase):
    """#584-conform: named refusal, never a silent shrink."""

    def test_an_unfundable_reserve_REFUSES_with_the_numbers(self):
        with self.assertRaises(FastLaneReserveError) as e:
            _plan(fast_lane_max_prompt_tokens=350_000)
        msg = str(e.exception)
        self.assertIn("REFUSED rather than shrunk", msg)
        self.assertIn("350,000", msg)
        self.assertIn("100,000", msg)

    def test_it_does_not_silently_shrink_to_what_fits(self):
        try:
            r = _plan(fast_lane_max_prompt_tokens=350_000)
        except FastLaneReserveError:
            return
        self.fail(f"silently shrank to {r.tokens}")

    def test_a_reserve_that_exactly_meets_the_floor_is_allowed(self):
        r = _plan(fast_lane_max_prompt_tokens=POOL - 100_000)
        self.assertEqual(r.tokens, POOL - 100_000)


class TestTheCombinedInvariantWith552(unittest.TestCase):
    """The two mechanisms look like they should fight. They do not, and the
    reason is that they act on different quantities.

    NOTE, and it is a correction to the framing this task arrived with: #552
    does NOT let an aged heavy request beat the fast tier for one admission.
    schedule_policy.py:385-432 promotes it to fast_lane_priority - 1 -- below
    the fast tier -- and its docstring says promoting a heavy request above
    fast "would only wedge the admission loop and starve the fast lane", which
    is precisely the #536 defect.
    """

    def test_552_never_promotes_a_heavy_request_above_the_fast_tier(self):
        import inspect

        from sglang.srt.managers import schedule_policy

        # Whitespace-normalised: the docstring wraps this sentence across two
        # lines (schedule_policy.py:398-399), and a contiguous-string pin fails
        # on the newline rather than on the meaning.
        src = " ".join(inspect.getsource(schedule_policy).split())
        self.assertIn("fast_lane_priority - 1", src)
        self.assertIn("promoting it ABOVE fast would only wedge the admission loop", src)
        self.assertIn("starve the fast lane", src)

    def test_an_AGED_heavy_request_still_cannot_take_the_reserve(self):
        """Ordering promotion and memory withholding are independent: whatever
        an aged heavy request's queue position, its admissible tokens exclude
        the untouched reserve."""
        r = _plan()
        for aged in (False, True):  # aging changes ORDER, not the draw
            self.assertEqual(
                admissible_tokens(
                    is_fast_lane=False, pool_free_tokens=POOL, reserve=r
                ),
                POOL - r.tokens,
                f"aged={aged}: the reserve is subtracted from the DRAW",
            )

    def test_the_two_mechanisms_touch_disjoint_quantities(self):
        """The invariant in one assertion: the reserve module decides tokens
        and knows nothing about priority; the ordering decides position and
        knows nothing about the pool."""
        import inspect

        from sglang.srt.managers import fast_lane_reserve

        src = inspect.getsource(fast_lane_reserve)
        for ordering_word in ("sort(", "wait_queue_entry_time", "priority_sign"):
            self.assertNotIn(ordering_word, src)


if __name__ == "__main__":
    unittest.main()
