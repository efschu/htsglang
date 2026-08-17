# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#731: a request must exist in exactly ONE place after a cutover.

THE CAUSAL CHAIN, measured 2026-08-17 across two boots:

    the carry re-homes a request into running_batch/running_mbs[0] and left
    waiting_queue untouched
      -> the request exists TWICE: resident (invisible to the policy as
         runnable) and queued (counted)
      -> _pending_prefill_tokens sums the queue and the resident set without
         excluding their intersection, so the same prompt is billed twice:
         51,369 -> 102,307 tokens across one cutover, within rounding of 2x
      -> the inflated backlog drives the flip policy past its threshold
      -> six cutovers, FLIP-CARRY reporting a resident carry while the policy
         reads `running bs 0`, the #699 detector reporting "1 queued, 0
         running", and the warmup generation never served.

THE FLIP CHURN WAS A SYMPTOM. Six cutovers looked like a flip defect and was
not one; it was the policy responding correctly to a number that was wrong.
Nobody should "fix" the churn separately.

Three fixes, three sets of tests here: the carry consumes the queue entry
(state), the counter de-duplicates at the intersection (number), and the
duplicate guard's universe grows to include the queue (so this class cannot go
silent again).
"""

import types
import unittest

from sglang.srt.managers.phase_flip_resident_carry import duplicate_resident_reqs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2)


class _Req:
    def __init__(self, rid, ntok):
        self.rid = rid
        self.origin_input_ids = list(range(ntok))
        self.extend_range = None


class _Batch:
    def __init__(self, reqs):
        self.reqs = list(reqs)


def _consume(scheduler, merged):
    from sglang.srt.managers.phase_flip_resident_carry import (
        _consume_carried_from_waiting_queue,
    )

    return _consume_carried_from_waiting_queue(scheduler, merged)


class TheCarryConsumesTheQueueEntry(unittest.TestCase):
    """STATE: after a cutover the resident set owns the request, alone."""

    def test_a_carried_request_leaves_the_queue(self):
        req = _Req("r1", 51369)
        sched = types.SimpleNamespace(waiting_queue=[req])
        removed = _consume(sched, _Batch([req]))
        self.assertEqual(1, removed)
        self.assertEqual([], sched.waiting_queue)

    def test_a_merely_queued_request_is_untouched(self):
        """CAN-FAIL: the reverse edge. Only what was re-homed may be removed."""
        carried, queued_only = _Req("carried", 10), _Req("queued", 20)
        sched = types.SimpleNamespace(waiting_queue=[carried, queued_only])
        _consume(sched, _Batch([carried]))
        self.assertEqual([queued_only], sched.waiting_queue)

    def test_consuming_twice_is_idempotent(self):
        """A carry over an already-consumed queue is a no-op, not a corruption."""
        req = _Req("r1", 10)
        sched = types.SimpleNamespace(waiting_queue=[req])
        self.assertEqual(1, _consume(sched, _Batch([req])))
        self.assertEqual(0, _consume(sched, _Batch([req])))
        self.assertEqual([], sched.waiting_queue)

    def test_no_queue_and_no_merge_are_both_safe(self):
        self.assertEqual(0, _consume(types.SimpleNamespace(waiting_queue=[]), None))
        self.assertEqual(
            0, _consume(types.SimpleNamespace(waiting_queue=None), _Batch([_Req("x", 1)]))
        )

    def test_the_carry_ITSELF_consumes_the_queue(self):
        """WIRING, not just the helper.

        The first version of this class only called
        `_consume_carried_from_waiting_queue` directly, so deleting the call
        from `install_resident_set` left every test green -- the helper worked
        and nothing used it. That is the documented-but-inert shape this repo
        has paid for repeatedly, so the real entry point is exercised here.
        """
        from sglang.srt.managers.phase_flip_resident_carry import install_resident_set

        req = _Req("carried", 51369)
        sched = types.SimpleNamespace(
            waiting_queue=[req],
            running_mbs=[],
            running_batch=None,
            last_batch=None,
            last_mbs=[],
        )
        install_resident_set(sched, [_Batch([req])], to_tp=True)
        self.assertEqual(
            [], sched.waiting_queue, "install_resident_set must consume the queue entry"
        )

    def test_a_broken_scheduler_does_not_kill_the_cutover(self):
        """Bookkeeping must never take down a flip (#715 lesson)."""

        class _Hostile:
            @property
            def waiting_queue(self):
                raise RuntimeError("boom")

        self.assertEqual(0, _consume(_Hostile(), _Batch([_Req("x", 1)])))


class TheGuardSeesTheQueueNow(unittest.TestCase):
    """GUARD: the universe that excluded half the places a request can live."""

    def test_resident_and_queued_is_reported(self):
        """RED before the fix: this shape was invisible."""
        req = _Req("dup", 5)
        dups = duplicate_resident_reqs([_Batch([req])], waiting_queue=[req])
        self.assertEqual(["queued:dup"], dups)

    def test_the_original_resident_vs_resident_shape_still_reports(self):
        """CAN-FAIL: growing the universe must not lose the 2026-08-09 case."""
        req = _Req("dup", 5)
        dups = duplicate_resident_reqs([_Batch([req]), _Batch([req])])
        self.assertEqual(["dup"], dups)

    def test_the_two_shapes_stay_distinguishable(self):
        req = _Req("dup", 5)
        both = duplicate_resident_reqs([_Batch([req]), _Batch([req])], [req])
        self.assertIn("dup", both)
        self.assertIn("queued:dup", both)

    def test_no_duplication_reports_nothing(self):
        a, b = _Req("a", 1), _Req("b", 1)
        self.assertEqual([], duplicate_resident_reqs([_Batch([a])], waiting_queue=[b]))

    def test_omitting_the_queue_keeps_the_old_behaviour(self):
        """Existing callers must not change meaning."""
        req = _Req("x", 1)
        self.assertEqual([], duplicate_resident_reqs([_Batch([req])]))


class TheCounterBillsEachPromptOnce(unittest.TestCase):
    """NUMBER: the half that made the duplication visible, and mine to fix.

    The #713(a) resident term did not create the duplicate state -- the carry
    did -- but a counter that sums two sets without excluding their
    intersection is the second half of the defect.
    """

    def _pending(self, *, queued, running):
        from sglang.srt.managers.scheduler import Scheduler

        holder = types.SimpleNamespace(
            waiting_queue=list(queued),
            chunked_req=None,
            running_batch=_Batch(running),
        )
        return Scheduler._pending_prefill_tokens(holder)

    def test_a_request_in_both_sets_is_counted_once(self):
        """RED before the fix: this returned 2x the prompt (51369 -> 102307).

        `extend_range` is set DELIBERATELY. The #713(a) resident term skips a
        request whose progress is unknown (`extend_range is None` counts as
        zero), so a duplicate without it contributes nothing to the resident
        side and the intersection is never exercised -- the first version of
        this test passed with the dedup deleted, which is no test at all.
        """
        req = _Req("dup", 51369)
        req.extend_range = types.SimpleNamespace(end=0)
        self.assertEqual(51369, self._pending(queued=[req], running=[req]))

    def test_distinct_requests_still_add_up(self):
        """CAN-FAIL: the fix must not swallow genuinely separate work."""
        a, b = _Req("a", 100), _Req("b", 200)
        b.extend_range = types.SimpleNamespace(end=0)
        self.assertEqual(300, self._pending(queued=[a], running=[b]))

    def test_queued_only_is_counted(self):
        self.assertEqual(70, self._pending(queued=[_Req("q", 70)], running=[]))

    def test_resident_only_with_unprefilled_tail_is_counted(self):
        """The #713(a) term itself still works -- dedup is at the intersection."""
        r = _Req("r", 90)
        r.extend_range = types.SimpleNamespace(end=40)
        self.assertEqual(50, self._pending(queued=[], running=[r]))


class TheArrivingTermEnforcesItsOwnInvariant(unittest.TestCase):
    """Its docstring asserted "ARRIVED but not yet on the queue"; nothing checked.

    The only inflight-bearing call site runs pre-queue, so the invariant holds
    today. It is checked rather than trusted because an asserted-never-enforced
    invariant is exactly how the resident-vs-queued double count stayed silent.
    """

    def _arriving(self, inflight, already_queued=None):
        from sglang.srt.managers.scheduler import _arriving_prefill_tokens

        return _arriving_prefill_tokens(inflight, already_queued)

    def test_an_already_queued_arrival_is_not_counted_again(self):
        item = types.SimpleNamespace(input_ids=list(range(500)))
        self.assertEqual(0, self._arriving([item], {id(item)}))

    def test_a_genuine_arrival_is_still_counted(self):
        """CAN-FAIL: the check must not silence the term it hardens."""
        item = types.SimpleNamespace(input_ids=list(range(500)))
        self.assertEqual(500, self._arriving([item], set()))
        self.assertEqual(500, self._arriving([item]))

    def test_control_traffic_still_contributes_nothing(self):
        self.assertEqual(0, self._arriving([types.SimpleNamespace(abort=True)]))


if __name__ == "__main__":
    unittest.main()
