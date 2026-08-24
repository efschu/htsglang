"""#856: a retracted request must not push the policy over the bar.

THE DEFECT THIS PREVENTS, and it is a REPEAT of a measured one.

The phase-flip seam retracts resident requests instead of carrying them --
the carry is what made #825's tree reset crash (`dec_lock_ref` walking off an
orphaned root, all three ranks, 2026-08-23), and HiCache read-through makes
the carry unnecessary. But retraction puts each request's FULL prompt back in
the waiting queue, and `Scheduler._pending_prefill_tokens` sums
`len(req.origin_input_ids)`.

That figure is compared against `N = C / (1/X - 1/P)`, where X and P are
UNCACHED prefill throughputs. The retracted tokens are not uncached: their KV
was computed and the fence persisted it, so re-prefilling them is a cache read
-- at a cost that is the same in TP and in PP. Equal cost on both sides of an
inequality cancels. Pricing them as cold prefill would hand the policy a large
backlog after EVERY cutover, in both directions.

#731 measured exactly that outcome from a different cause (one prompt counted
twice across a cutover):

    "51,369 -> 102,307 tokens across one cutover, within rounding of exactly
     2x. The inflated backlog drove the flip policy past its threshold --
     six cutovers, nothing served."

#731's fix -- make the carry consume the queue entry -- cannot catch this
route, because nothing is double-counted here. The tokens are counted ONCE, at
the wrong price.

RED-FIRST: `uncached_prompt_tokens` did not exist, and the shipped sum priced
eight retracted 20k-token requests at 160k against a live bar of 18614 --
reproduced below as `test_the_shipped_sum_reproduces_the_731_shape`.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

import types
import unittest

from sglang.srt.managers.scheduler import uncached_prompt_tokens
from sglang.test.test_utils import CustomTestCase

#: W25's shape: max_running_requests=8, prompts 16-24k, live bar N=18614.
W25_BAR = 18614
W25_PROMPT = 20000
W25_RESIDENT = 8


def _req(prompt_tokens, *, retracted=False, credit=None, output=0):
    r = types.SimpleNamespace(
        origin_input_ids=[0] * prompt_tokens,
        output_ids=[0] * output,
        is_retracted=retracted,
    )
    if credit is not None:
        r.cached_prompt_tokens_at_retract = credit
    return r


class TestTheShippedSumReproducesThe731Shape(CustomTestCase):
    """The defect, in the numbers this rig actually runs."""

    def test_the_shipped_sum_reproduces_the_731_shape(self):
        reqs = [_req(W25_PROMPT) for _ in range(W25_RESIDENT)]
        naive = sum(len(r.origin_input_ids) for r in reqs)
        self.assertEqual(naive, 160000)
        self.assertGreater(
            naive,
            W25_BAR,
            "eight retracted 20k prompts priced as cold prefill clear the "
            "live break-even by 8.6x -- that is the thrash pathway",
        )


class TestARetractedRequestIsPricedAtItsResidency(CustomTestCase):
    def test_a_fully_prefilled_retracted_request_contributes_nothing(self):
        # It was decoding, so its whole prompt is computed and persisted.
        r = _req(W25_PROMPT, retracted=True, credit=W25_PROMPT)
        self.assertEqual(uncached_prompt_tokens(r), 0)

    def test_the_w25_cutover_no_longer_clears_the_bar(self):
        reqs = [
            _req(W25_PROMPT, retracted=True, credit=W25_PROMPT)
            for _ in range(W25_RESIDENT)
        ]
        priced = sum(uncached_prompt_tokens(r) for r in reqs)
        self.assertEqual(priced, 0)
        self.assertLess(
            priced,
            W25_BAR,
            "a cutover that retracts fully-cached requests must not, by "
            "itself, argue for flipping straight back",
        )

    def test_a_partly_prefilled_request_pays_for_its_remainder(self):
        # Retracted mid-chunked-prefill: 6000 of 20000 computed. The other
        # 14000 are real, uncached work and must still count.
        r = _req(W25_PROMPT, retracted=True, credit=6000)
        self.assertEqual(uncached_prompt_tokens(r), 14000)


class TestTheDefaultPathIsUnchanged(CustomTestCase):
    """Every request that was never retracted counts in full, token for
    token. This is what makes the change safe for non-flip deployments."""

    def test_a_normal_queued_request_counts_in_full(self):
        self.assertEqual(uncached_prompt_tokens(_req(1234)), 1234)

    def test_a_retracted_request_without_a_stamp_counts_in_full(self):
        # THE CAN-FAIL DIRECTION. An implementation that zeroed anything
        # flagged retracted would pass every test above while deleting real
        # backlog -- and a request retracted for a reason OTHER than the flip
        # seam (priority preemption, #731's own path) has no fence behind it.
        self.assertEqual(uncached_prompt_tokens(_req(1234, retracted=True)), 1234)

    def test_the_sum_over_a_mixed_queue_is_the_honest_one(self):
        queue = [
            _req(20000, retracted=True, credit=20000),  # cached, free
            _req(20000, retracted=True, credit=6000),  # 14000 real
            _req(5000),  # never retracted
        ]
        self.assertEqual(sum(uncached_prompt_tokens(r) for r in queue), 19000)


class TestItCannotProduceANegativeBacklog(CustomTestCase):
    """This value is summed into a backlog; a negative would silently offset
    other requests' real work."""

    def test_a_credit_larger_than_the_prompt_floors_at_zero(self):
        self.assertEqual(
            uncached_prompt_tokens(_req(100, retracted=True, credit=999)), 0
        )

    def test_a_negative_credit_cannot_inflate_the_backlog(self):
        self.assertEqual(
            uncached_prompt_tokens(_req(100, retracted=True, credit=-50)), 100
        )

    def test_an_unreadable_credit_counts_the_prompt_in_full(self):
        for bad in ("n/a", object(), None):
            with self.subTest(credit=bad):
                r = _req(100, retracted=True)
                r.cached_prompt_tokens_at_retract = bad
                self.assertEqual(uncached_prompt_tokens(r), 100)

    def test_a_malformed_request_is_zero_not_an_exception(self):
        # It runs inside a scheduler round; an instrument may never be the
        # thing that breaks one.
        self.assertEqual(uncached_prompt_tokens(types.SimpleNamespace()), 0)


class TestTheStampIsTakenBeforeItIsCleared(CustomTestCase):
    """`reset_for_retract` wipes prefix_indices, num_matched_prefix_tokens and
    extend_range. The credit must be captured ahead of that, or it is
    unrecoverable."""

    def test_reset_for_retract_records_the_credit(self):
        import inspect

        from sglang.srt.managers.schedule_batch import Req

        src = inspect.getsource(Req.reset_for_retract)
        stamp = src.index("cached_prompt_tokens_at_retract")
        cleared = src.index("self.extend_range = None")
        self.assertLess(
            stamp,
            cleared,
            "the credit must be stamped BEFORE extend_range is cleared; "
            "afterwards the information does not exist anywhere",
        )


if __name__ == "__main__":
    unittest.main()
