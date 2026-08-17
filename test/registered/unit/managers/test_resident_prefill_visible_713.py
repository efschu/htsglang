"""#713 (a): an ADMITTED but unprefilled request must be visible to the policy.

MEASURED SPECIMEN, 2026-08-17. Arm D2 arrived 06:53:59.344, INSIDE the
:59 -> :01 PP window with ~1.7 s of PP remaining -- ample for a 10-token prompt.
It was not served. pp_to_tp armed at :01 and D2's first token did not come until
:04.879, a full ~5.5 s cycle late.

The policy states the contradiction in its own arm reason:

    IDLE-LOCKED: no batch of either work class can be built in the pp layout
    (1 REQ RESIDENT, 0 TOK PREFILL PENDING)

A resident request with zero pending prefill. ``_pending_prefill_tokens`` counted
only three sources -- the waiting queue, the chunked slot, and the recv batch --
and an admitted request has LEFT the waiting queue. If it is not the chunked_req
it is invisible, so pp_to_tp armed correctly by its own inputs and took the
layout away from work it could not see.

THE FIX'S DANGEROUS DIRECTION IS OVER-COUNTING, which is why (b) exists: if a
resident request whose prefill is DONE were counted, pending would never reach
0, the policy would pull and hold toward PP permanently, and decode would starve
-- bounded only by the SLO cap. Unknown progress therefore counts as ZERO.
"""

import unittest
from types import SimpleNamespace

from sglang.test.test_utils import CustomTestCase


class _Range:
    def __init__(self, start, end):
        self.start = start
        self.end = end


def _req(total, filled, rid="r"):
    return SimpleNamespace(
        origin_input_ids=list(range(total)),
        extend_range=None if filled is None else _Range(0, filled),
        rid=rid,
    )


def _sched(resident=(), waiting=(), chunked=None):
    from sglang.srt.managers.scheduler import Scheduler

    s = Scheduler.__new__(Scheduler)
    s.waiting_queue = list(waiting)
    s.chunked_req = chunked
    s.running_batch = SimpleNamespace(reqs=list(resident))
    return s


class TestResidentPrefillIsVisible713(CustomTestCase):
    def test_a_resident_unprefilled_request_is_counted(self):
        """(a) THE D2 SHAPE. Admitted, unprefilled, empty waiting queue --
        pending must not read 0. Red before the fix."""
        s = _sched(resident=[_req(22, 0)])
        self.assertEqual(s._pending_prefill_tokens(), 22)

    def test_a_partially_prefilled_resident_counts_only_the_remainder(self):
        s = _sched(resident=[_req(1000, 400)])
        self.assertEqual(s._pending_prefill_tokens(), 600)

    def test_a_fully_prefilled_resident_counts_zero(self):
        """(b) THE STARVATION GUARD. A decoding request must contribute
        nothing, or pending never reaches 0 and the policy holds toward PP
        forever."""
        s = _sched(resident=[_req(500, 500)])
        self.assertEqual(s._pending_prefill_tokens(), 0)

    def test_unknown_progress_counts_as_zero_not_as_a_whole_prompt(self):
        """The chosen failure direction, pinned: a request with no
        extend_range is treated as fully prefilled. Under-counting restores
        today's behaviour; over-counting starves decode."""
        s = _sched(resident=[_req(9999, None)])
        self.assertEqual(s._pending_prefill_tokens(), 0)

    def test_the_chunked_req_is_not_double_counted(self):
        """(c) The chunked term already prices the chunked request. Counting it
        again in the resident term would inflate pending and bias every flip
        decision toward PP."""
        chunked = _req(1000, 400, rid="chunked")
        s = _sched(resident=[chunked], chunked=chunked)
        self.assertEqual(
            s._pending_prefill_tokens(),
            600,
            "600 from the chunked term only -- not 1200",
        )

    def test_waiting_queue_and_resident_sum(self):
        s = _sched(resident=[_req(22, 0)], waiting=[_req(100, 0)])
        self.assertEqual(s._pending_prefill_tokens(), 122)

    def test_no_resident_batch_is_harmless(self):
        from sglang.srt.managers.scheduler import Scheduler

        s = Scheduler.__new__(Scheduler)
        s.waiting_queue = []
        s.chunked_req = None
        s.running_batch = None
        self.assertEqual(s._pending_prefill_tokens(), 0)

    def test_a_broken_resident_entry_cannot_break_the_round(self):
        """An observation must never raise into the scheduler loop."""
        bad = SimpleNamespace(extend_range=_Range(0, 0))  # no origin_input_ids
        s = _sched(resident=[bad])
        self.assertIsInstance(s._pending_prefill_tokens(), int)


if __name__ == "__main__":
    unittest.main()
