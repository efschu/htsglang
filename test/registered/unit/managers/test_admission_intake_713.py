"""#713: the flip policy must count work that has ARRIVED but is not yet queued.

ROOT CAUSE, traced 2026-08-17.

``recv_requests`` (``scheduler_components/request_receiver.py:104-129``) does
this, in this order:

    recv_reqs = self._pull_raw_reqs()        # the new request is HERE
    ...
    policy_req = self.phase_policy_hook()    # the policy evaluates NOW
    ...                                       # -> returned to the scheduler
                                              # -> waiting_queue.append(req)
                                              #    (scheduler.py:4089)

So when ``maybe_arm_phase_policy`` runs, the request that just arrived is still
sitting in the local ``recv_reqs`` list and has NOT reached
``self.waiting_queue``. ``_pending_prefill_tokens`` sums the waiting queue plus
the chunked remainder (``scheduler.py:8533-8539``), so on an idle box it reads
**0** at exactly the moment the policy asks whether the target layout could
admit anything.

``_layout_admits("pp", ...)`` then early-falses on its FIRST line::

    if int(pending_tokens) <= 0:
        return False

which is why the specimen's own diagnostic showed both PP terms holding
(72033 rows >= need, 3 mamba slots) while the verdict was still False: the
function never reached them. ``tp`` refuses too (nothing resident), so
``_idle_locked_inputs`` returns ``(nothing_can_run=True, target_can_admit=False)``
and the policy declines the flip -- the 31.64 s TTFT on an idle box.

WHICH NUMBER IS WRONG FOR AN IDLE BOX. The 22 is the truth and the 0 is the
defect. The work exists -- it has been received and will be queued in this same
round; the queue simply has not been updated yet. The rule was proven correct on
replayed inputs (``test_idle_locked_terms_713.py``), so the fix is to the INPUT,
never to the rule.

THE COUNTERWEIGHTS BELOW ARE NOT DECORATION. A fix that made admission easier
in general would trade a 31 s TTFT for a wedge somewhere else, so every genuine
cause of refusal is pinned to still refuse.
"""

import unittest
from types import SimpleNamespace

from sglang.test.test_utils import CustomTestCase

PENDING = 22
ROWS_AVAIL = 72_033
MAMBA_SLOTS = 3


def _arriving(n_tokens=PENDING):
    """A just-received request: TokenizedGenerateReqInput carries ``input_ids``
    (io_struct.py:798), NOT ``origin_input_ids`` -- that is the Req the
    scheduler builds later."""
    return SimpleNamespace(input_ids=list(range(n_tokens)))


def _queued(n_tokens=PENDING):
    """A request that HAS reached the waiting queue: a Req, with
    ``origin_input_ids``."""
    return SimpleNamespace(origin_input_ids=list(range(n_tokens)))


def _sched(avail=ROWS_AVAIL, slots=MAMBA_SLOTS, evictable=0, chunk=512, queue=()):
    from sglang.srt.managers.scheduler import Scheduler

    s = Scheduler.__new__(Scheduler)
    s.server_args = SimpleNamespace(chunked_prefill_size=chunk)
    s.token_to_kv_pool_allocator = SimpleNamespace(available_size=lambda: avail)
    s.tree_cache = SimpleNamespace(full_evictable_size=lambda: evictable)
    s.req_to_token_pool = SimpleNamespace(
        mamba_allocator=SimpleNamespace(available_size=lambda: slots)
    )
    s.waiting_queue = list(queue)
    s.chunked_req = None
    s._round_built_nothing = True
    s.phase_flip_active_stack = "tp"
    return s


class TestArrivingWorkIsCounted(CustomTestCase):
    def test_THE_DIVERGENCE_the_queue_is_empty_while_the_work_exists(self):
        """RED-FIRST. The specimen's 0-vs-22 in one assertion.

        Same round, same box: the queue reads 0 because the request has not
        been appended yet, and the arriving intake holds the 22 tokens the
        refusal message reported.
        """
        s = _sched()
        self.assertEqual(s._pending_prefill_tokens(), 0, "the queue is empty")
        self.assertEqual(
            s._pending_prefill_tokens([_arriving()]),
            PENDING,
            "arriving work must be counted: it is why the policy was woken",
        )

    def test_the_idle_box_ARMS_once_the_arriving_request_is_counted(self):
        """End to end: the verdict the specimen should have produced."""
        s = _sched()
        pending = s._pending_prefill_tokens([_arriving()])
        nothing_can_run, target_can_admit = s._idle_locked_inputs(0, pending)
        self.assertTrue(nothing_can_run)
        self.assertTrue(
            target_can_admit,
            "with the arriving 22 tokens counted, pp can prefill and the flip "
            "must arm -- this is the 31.64 s TTFT",
        )

    def test_the_uncounted_state_is_exactly_the_reported_refusal(self):
        """Pins the DEFECT itself, so a regression is recognisable.

        Without the arriving work the pair is (True, False) -- nothing can run
        and the target 'cannot' admit -- which is the BOTH BLOCKED the specimen
        logged.
        """
        s = _sched()
        self.assertEqual(
            s._idle_locked_inputs(0, s._pending_prefill_tokens()), (True, False)
        )

    def test_the_early_false_is_what_hides_the_holding_terms(self):
        """Why the diagnostic showed both PP terms holding and still refused."""
        s = _sched()
        self.assertFalse(s._layout_admits("pp", 0, 0), "early-false on pending<=0")
        self.assertTrue(
            s._layout_admits("pp", 0, PENDING),
            "the very same box admits once the pending count is right",
        )


class TestTheObserverPathIsUnchanged(CustomTestCase):
    """The no-argument call is the #363 observer and the break-even N
    denominator. It must keep meaning exactly what it meant."""

    def test_no_arg_behaviour_is_byte_identical(self):
        s = _sched(queue=[_queued(7), _queued(5)])
        self.assertEqual(s._pending_prefill_tokens(), 12)

    def test_the_chunked_remainder_still_counts(self):
        s = _sched(queue=[_queued(4)])
        s.chunked_req = SimpleNamespace(
            origin_input_ids=list(range(100)), extend_range=SimpleNamespace(end=30)
        )
        self.assertEqual(s._pending_prefill_tokens(), 4 + 70)

    def test_arriving_work_ADDS_to_the_queue_it_has_not_joined(self):
        s = _sched(queue=[_queued(4)])
        self.assertEqual(s._pending_prefill_tokens([_arriving(6)]), 10)


class TestRefusalStaysReachableForItsGenuineCauses(CustomTestCase):
    """The fix must not weaken admission where the hold is CORRECT."""

    def test_a_truly_empty_box_still_refuses(self):
        s = _sched()
        self.assertEqual(s._pending_prefill_tokens([]), 0)
        self.assertEqual(s._idle_locked_inputs(0, 0), (True, False))

    def test_a_starved_pool_still_refuses_even_with_arriving_work(self):
        s = _sched(avail=3)
        pending = s._pending_prefill_tokens([_arriving()])
        self.assertEqual(pending, PENDING)
        self.assertFalse(s._layout_admits("pp", 0, pending), "rows < need")
        self.assertEqual(s._idle_locked_inputs(0, pending), (True, False))

    def test_no_state_slot_still_refuses_even_with_arriving_work(self):
        s = _sched(slots=0)
        pending = s._pending_prefill_tokens([_arriving()])
        self.assertFalse(s._layout_admits("pp", 0, pending), "no GDN slot")
        self.assertEqual(s._idle_locked_inputs(0, pending), (True, False))

    def test_a_round_that_BUILT_something_is_never_idle_locked(self):
        s = _sched()
        s._round_built_nothing = False
        self.assertEqual(
            s._idle_locked_inputs(0, s._pending_prefill_tokens([_arriving()])),
            (False, False),
        )

    def test_non_prefill_intake_is_not_counted_as_work(self):
        """recv_reqs carries control messages too -- an abort or a flip arm is
        not prefill work, and counting it would arm on nothing."""
        s = _sched()
        control = SimpleNamespace(rid="abort-1")
        self.assertEqual(s._pending_prefill_tokens([control]), 0)
        self.assertEqual(
            s._pending_prefill_tokens([control, _arriving()]),
            PENDING,
            "the generate request still counts alongside control traffic",
        )

    def test_an_empty_prompt_is_not_work(self):
        s = _sched()
        self.assertEqual(s._pending_prefill_tokens([_arriving(0)]), 0)

    def test_a_raising_intake_item_cannot_break_the_round(self):
        """A probe inside the admission path must never fault the scheduler --
        the #715 lesson, where an observation died inside the crash it existed
        to explain."""

        class Hostile:
            @property
            def input_ids(self):
                raise RuntimeError("boom")

        s = _sched()
        self.assertEqual(s._pending_prefill_tokens([Hostile(), _arriving()]), PENDING)


if __name__ == "__main__":
    unittest.main()
