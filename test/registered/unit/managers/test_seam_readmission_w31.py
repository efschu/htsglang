"""W31: the #856 seam must PUT BACK what it retracts.

THE DEFECT, measured twice before it was named. The seam's own log line has
promised on every flip of every boot that "their KV is in the canonical store
from the fence; the new layout RE-ADMITS them and serves the prefix by
read-through". Nothing did the re-admitting. `retract_all` returns the list it
retracted, `_release_residents_for_cutover` returned it upward, and its caller
discarded it.

W31 arm 2 (SPECIMEN_w31_a2_residents_never_readmitted.log):
  * 28 distinct non-health rids admitted, each appearing EXACTLY 3 times --
    one ADMIT line per rank, i.e. every request admitted exactly once ever;
  * 42 batches of `#new-token: 4096` / 3 ranks = 14 requests, prefilled once;
  * 78 requests retracted by the seam across 42 pp_to_tp cutovers;
  * ZERO completions -- all 28 clients waited out the 600 s timeout.
And three code facts, checked rather than inferred: `retract_all` never
touches `waiting_queue`; the caller at phase_flip_runtime.py discarded the
returned list; no seam path called `_add_request_to_queue`.

W30 and W31 were both read as a flip "livelock". They were the flip
ping-ponging over an instance whose work it had already dropped on the floor.

WHAT THIS FILE PINS
  * every retracted resident comes back, exactly once;
  * #1068 (spec 4.3, G3/G4): the WHOLE run-willing population -- retracted
    residents AND queue occupants -- is re-issued through the one intake
    site in `kv_arrival_seq` order; occupants are NOT stamped as retracts;
    the old front-of-queue block sort is gone (the queue is rebuilt as a
    whole in arrival order, so the oldest work is first by construction);
  * CAN-FAIL: dropping the return value again turns this red;
  * the abort path: there is no window where the list is computed and lost;
  * rank-uniformity: the rebuilt queue is a pure function of rank-uniform
    inputs.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

import types
import unittest

from sglang.srt.managers.scheduler import Scheduler
from sglang.test.test_utils import CustomTestCase


class _Req:
    def __init__(self, rid, arrival, *, seam_epoch=7, done=False):
        self.rid = rid
        self.kv_arrival_seq = arrival
        self.seam_readmit_epoch = seam_epoch
        self._done = done
        self.time_stats = types.SimpleNamespace(
            set_wait_queue_entry_time=lambda: None,
            set_retract_time=lambda: None,
        )

    def finished(self):
        return self._done


def _method(name):
    # Tolerant binding (same shape as test_prefetch_deferral_1068._method):
    # on a tree that predates the slice-3 fix the method is absent and the
    # stand-in must still import, so the red-first run fails in the TEST
    # that reaches the missing method, not at collection.
    fn = getattr(Scheduler, name, None)
    if fn is not None:
        return fn

    def _missing(self, *a, **k):
        raise AssertionError(f"Scheduler.{name} does not exist (slice 3 fix not built)")

    return _missing


class _Sched:
    """A scheduler stand-in exposing only what the re-admission touches.

    `readmit_seam_residents` is taken from the REAL class and bound here, so
    this exercises the shipped code rather than a restatement of it.
    """

    readmit_seam_residents = Scheduler.readmit_seam_residents
    # slice-3 fix (review round 1): the cutover CLEARER of the A12.2
    # deferral mark runs inside readmit_seam_residents; bound from the real
    # class too, so the stand-in exercises the shipped re-issue path whole.
    _clear_prefetch_deferral_for_reissue = _method("_clear_prefetch_deferral_for_reissue")
    _clear_prefetch_deferral_fields = _method("_clear_prefetch_deferral_fields")

    def __init__(self, queue=()):
        self.waiting_queue = list(queue)
        self.added = []
        self.refuse = set()
        self.retract_sites = []
        self.verdicts = {}

    def _add_request_to_queue(self, req, is_retracted=False):
        # Models the real one's contract: it MAY refuse (priority validation,
        # queued-limit abort), it appends rather than front-inserts, and it
        # stamps the intake prefetch verdict (`_969c_verdict`, #1068).
        self.added.append((req.rid, is_retracted))
        if req.rid in self.refuse:
            return
        req._969c_verdict = self.verdicts.get(req.rid, "issued")
        self.waiting_queue.append(req)

    def _969ad_note_retract(self, req, site):
        # The #969AD probe the real `readmit_seam_residents` calls per
        # requeued request: a bounded per-rid site recorder, no scheduling
        # effect. Recorded so the stand-in stays faithful, never swallowed.
        self.retract_sites.append((req.rid, site))


class TestEverythingRetractedComesBack(CustomTestCase):
    def test_all_of_them_exactly_once(self):
        reqs = [_Req(f"r{i}", arrival=i) for i in range(5)]
        s = _Sched()
        self.assertEqual(s.readmit_seam_residents(reqs), 5)
        self.assertEqual(
            [r.rid for r in s.waiting_queue], ["r0", "r1", "r2", "r3", "r4"]
        )
        self.assertEqual(len(s.waiting_queue), len(set(id(r) for r in s.waiting_queue)))

    def test_they_are_queued_as_retracted(self):
        # `is_retracted=True` is what keeps the original arrival position and
        # stamps the retract time instead of a fresh wait-queue entry.
        reqs = [_Req("a", 0)]
        s = _Sched()
        s.readmit_seam_residents(reqs)
        self.assertEqual(s.added, [("a", True)])

    def test_an_empty_release_is_a_no_op(self):
        s = _Sched()
        self.assertEqual(s.readmit_seam_residents([]), 0)
        self.assertEqual(s.waiting_queue, [])

    def test_the_l5_line_is_emitted_for_an_empty_population(self):
        # slice-3 fix: spec 4.3 says `Log L5` unconditionally and section
        # 10 counts ONE L5 per cutover; a cutover whose whole population
        # finished() (k+m == 0) printed nothing on 0ad85647cb (the
        # `if population:` guard was an unnamed deviation). RED there.
        from sglang.srt.managers import scheduler as sched_mod

        s = _Sched()
        with self.assertLogs(sched_mod.logger, level="INFO") as caught:
            self.assertEqual(s.readmit_seam_residents([], requeue_waiting=True), 0)
        line = [ln for ln in caught.output if "SEAM RE-ADMISSION" in ln]
        self.assertEqual(len(line), 1, caught.output)
        self.assertIn("0 retracted resident(s) + 0 queue occupant(s)", line[0])
        self.assertIn("queue 0 -> 0", line[0])
        self.assertIn("dropped_by_queue_limit=0", line[0])


class TestArrivalOrderOverAll(CustomTestCase):
    """#1068 spec 4.3 (G4): issue order = kv_arrival_seq over the WHOLE
    population, residents and occupants alike. The pre-#1068 pin put the
    retracted block in FRONT of every occupant; an occupant that arrived
    BEFORE a resident now precedes it, because the queue is rebuilt as one
    arrival-ordered sequence."""

    def test_an_older_occupant_precedes_a_younger_resident(self):
        # THE INVERSION of the retired front-of-queue pin: occupant q(3) sits
        # between residents old(1) and new(5).
        s = _Sched([_Req("q", 3)])
        s.readmit_seam_residents([_Req("new", 5), _Req("old", 1)])
        self.assertEqual([r.rid for r in s.waiting_queue], ["old", "q", "new"])

    def test_arrival_order_is_restored_even_if_released_out_of_order(self):
        # The seam enumerates residents by slot, which is not arrival order.
        s = _Sched()
        s.readmit_seam_residents([_Req("c", 3), _Req("a", 1), _Req("b", 2)])
        self.assertEqual([r.rid for r in s.waiting_queue], ["a", "b", "c"])

    def test_a_request_the_queue_refused_is_not_conjured_in(self):
        # `_add_request_to_queue` may legitimately refuse. The count reports
        # what LANDED, and a refused request must not appear -- inventing it
        # would be worse than dropping it.
        s = _Sched()
        s.refuse = {"b"}
        n = s.readmit_seam_residents([_Req("a", 1), _Req("b", 2), _Req("c", 3)])
        self.assertEqual([r.rid for r in s.waiting_queue], ["a", "c"])
        self.assertEqual(n, 2, "the count must report what LANDED, not what was tried")

    def test_a_client_that_gave_up_is_not_re_admitted(self):
        # Re-admitting a finished request puts work in a queue nobody is
        # waiting on.
        s = _Sched()
        n = s.readmit_seam_residents([_Req("live", 1), _Req("gone", 2, done=True)])
        self.assertEqual([r.rid for r in s.waiting_queue], ["live"])
        self.assertEqual(n, 1)


class TestQueueOccupantsAreReissued(CustomTestCase):
    """#1068 spec 4.3 (G3): the cutover nulls everything (tree, prefetch
    records, host pool), so a queue occupant's intake prefetch is gone too.
    It must be RE-ISSUED through the same intake site, in arrival order, and
    it is NOT a retract (no #969AD stamp, `is_retracted=False`)."""

    def test_queue_occupants_are_reissued_not_stamped_as_retracts(self):
        # T11. RED on 846c6797b9: readmit_seam_residents has no
        # requeue_waiting parameter and iterates only `reqs`.
        q1 = _Req("q1", 2)
        r1 = _Req("r1", 1)
        s = _Sched([q1])
        n = s.readmit_seam_residents([r1], requeue_waiting=True)
        self.assertEqual(n, 1, "the return value counts RESIDENTS only")
        self.assertEqual(s.added, [("r1", True), ("q1", False)])
        self.assertEqual(s.retract_sites, [("r1", "readmit_seam_residents")])
        self.assertEqual(q1._969c_population, "queue")
        self.assertEqual(r1._969c_population, "retract")
        self.assertEqual([r.rid for r in s.waiting_queue], ["r1", "q1"])
        summary = s.last_seam_readmit
        self.assertEqual(summary["retracted"], 1)
        self.assertEqual(summary["requeued"], 1)
        self.assertEqual(summary["residents"], 1)
        self.assertEqual(summary["occupants"], 1)
        self.assertEqual(summary["queue_before"], 1)
        self.assertEqual(summary["queue_after"], 2)
        self.assertEqual(summary["dropped_by_queue_limit"], 0)
        self.assertEqual(summary["verdicts"], {"issued": 2})

    def test_issue_order_is_arrival_order(self):
        # T12. RED on 846c6797b9: the loop runs in retract order and the
        # sort happens only after the loop, on the block alone.
        new, old, q = _Req("new", 5), _Req("old", 1), _Req("q", 3)
        s = _Sched([q])
        s.readmit_seam_residents([new, old], requeue_waiting=True)
        self.assertEqual([rid for rid, _ in s.added], ["old", "q", "new"])
        self.assertEqual([r.rid for r in s.waiting_queue], ["old", "q", "new"])

    def test_requeue_waiting_false_leaves_occupants_untouched(self):
        # The knob is honoured: occupants stay where they are, and they are
        # NOT re-issued.
        q = _Req("q", 3)
        s = _Sched([q])
        s.readmit_seam_residents([_Req("r", 1)], requeue_waiting=False)
        self.assertEqual(s.added, [("r", True)])
        self.assertIsNone(getattr(q, "_969c_population", None))
        self.assertEqual(s.last_seam_readmit["requeued"], 0)
        self.assertEqual(s.last_seam_readmit["occupants"], 0)

    def test_a_refused_occupant_is_named_as_dropped_not_requeued(self):
        # R8: under a queued limit an occupant can be refused by the intake;
        # it is counted as dropped_by_queue_limit, never silently.
        q = _Req("q", 3)
        s = _Sched([q])
        s.refuse = {"q"}
        s.readmit_seam_residents([_Req("r", 1)], requeue_waiting=True)
        self.assertEqual(s.last_seam_readmit["requeued"], 0)
        self.assertEqual(s.last_seam_readmit["dropped_by_queue_limit"], 1)

    def test_no_residents_still_reissues_the_occupants(self):
        # A cutover with nothing retracted still dropped every occupant's
        # prefetch record at _reset_full; the occupants must be re-issued.
        q = _Req("q", 3)
        s = _Sched([q])
        n = s.readmit_seam_residents([], requeue_waiting=True)
        self.assertEqual(n, 0)
        self.assertEqual(s.added, [("q", False)])
        self.assertEqual(s.last_seam_readmit["requeued"], 1)

    def test_the_verdict_histogram_names_declines(self):
        s = _Sched([_Req("q", 3)])
        s.verdicts = {"q": "declined:store_absent"}
        s.readmit_seam_residents([_Req("r", 1)], requeue_waiting=True)
        self.assertEqual(
            s.last_seam_readmit["verdicts"],
            {"issued": 1, "declined:store_absent": 1},
        )

    def test_the_l5_line_names_both_populations(self):
        from sglang.srt.managers import scheduler as sched_mod

        s = _Sched([_Req("q", 3)])
        with self.assertLogs(sched_mod.logger, level="INFO") as caught:
            s.readmit_seam_residents([_Req("r", 1)], requeue_waiting=True)
        line = [ln for ln in caught.output if "SEAM RE-ADMISSION" in ln]
        self.assertEqual(len(line), 1, caught.output)
        self.assertIn("1 retracted resident(s) + 1 queue occupant(s)", line[0])
        self.assertIn("queue 1 -> 2", line[0])
        self.assertIn("dropped_by_queue_limit=0", line[0])


class TestTheStampSurvivesTheRoundTrip(CustomTestCase):
    def test_requeued_requests_are_still_stamped(self):
        # The whole point: they must be admissible through the seam-transport
        # exemption in the TARGET layout, and that gate reads the stamp off
        # the waiting queue. A re-admission that stripped the stamp would put
        # them back somewhere strict purity still refuses to serve them.
        from sglang.srt.managers.phase_purity import seam_readmit_candidates

        s = _Sched()
        s.readmit_seam_residents([_Req("a", 1), _Req("b", 2)])
        self.assertEqual([r.rid for r in seam_readmit_candidates(s)], ["a", "b"])


class TestTheSeamActuallyCallsIt(CustomTestCase):
    """CAN-FAIL: dropping the return value again must turn this red.

    CONTRACT AS OF #1066 (8bac764f4b): the release site retracts and STASHES
    the released list (`_pending_seam_readmit`); the requeue runs in
    `_post_cutover_readmit`, after the stacks are swapped and the HiCache
    pools rebound, so the intake prefetch opens on the binding that will
    serve it. The pre-#1066 pins asserted the requeue inside the release
    method; that order was retired deliberately (its prefetch opened on the
    outgoing binding and refused stale every time) and the pins below follow
    the tree.
    """

    def test_the_release_site_stashes_what_it_released(self):
        import inspect

        from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

        src = inspect.getsource(PhaseFlipRuntime._release_residents_for_cutover)
        self.assertIn("_pending_seam_readmit", src)
        self.assertIn("released", src)

    def test_the_post_cutover_readmit_requeues_the_stash(self):
        import inspect

        from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

        src = inspect.getsource(PhaseFlipRuntime._post_cutover_readmit)
        self.assertIn("_pending_seam_readmit", src)
        self.assertIn("readmit_seam_residents", src)

    def test_the_seam_asserts_retracted_equals_readmitted(self):
        import inspect

        from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

        src = inspect.getsource(PhaseFlipRuntime._post_cutover_readmit)
        self.assertIn("RE-ADMISSION MISMATCH", src)

    def test_the_requeue_is_deferred_past_the_cutover_not_done_at_the_release_site(
        self,
    ):
        # INVERSION of the pre-#1066 pin. The old test wanted the requeue at
        # the release site so an abort mid-cutover could not strand the list.
        # #1066 measured the price of that order (intake prefetch opened on
        # the OUTGOING binding: 6/6 stale refusals, cached=0 on 90/90
        # prefills) and moved the requeue behind `_cutover_fn`; the abort
        # window is named in `_post_cutover_readmit`'s docstring as the
        # honest residual (a failed flip takes the instance down regardless).
        # Pinned here so the retired order cannot creep back one line at a
        # time.
        import inspect

        from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

        release_src = inspect.getsource(
            PhaseFlipRuntime._release_residents_for_cutover
        )
        self.assertNotIn(
            "readmit_seam_residents",
            release_src,
            "the requeue moved to _post_cutover_readmit in #1066; a requeue at "
            "the release site would open its intake prefetch on the outgoing "
            "binding again",
        )
        body_src = inspect.getsource(PhaseFlipRuntime._execute_body)
        release_at = body_src.find("self._release_residents_for_cutover(")
        readmit_at = body_src.find("self._post_cutover_readmit(")
        self.assertGreater(release_at, -1)
        self.assertGreater(readmit_at, -1)
        self.assertLess(release_at, readmit_at, "release first, then the readmit")

    def test_the_703_fence_coverage_is_asserted_not_assumed(self):
        import inspect

        from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

        src = inspect.getsource(PhaseFlipRuntime._release_residents_for_cutover)
        self.assertIn("_writeback_fence_ms", src)
        self.assertIn("WRITEBACK FENCE RECORDED", src)


class TestOrderAgainstTheLiveUniverse(CustomTestCase):
    """#731 shape: never live-referenced AND queued at the same time."""

    def test_consume_runs_before_the_requeue(self):
        import inspect

        from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

        # the consume happens inside `_retract_and_consume`, which is passed
        # to `release_residents_for_cutover` inside the release method; the
        # requeue lives in `_post_cutover_readmit` (#1066), and the cutover
        # body runs the release method before it.
        release_src = inspect.getsource(
            PhaseFlipRuntime._release_residents_for_cutover
        )
        self.assertGreater(release_src.find("release_residents_for_cutover("), -1)
        body_src = inspect.getsource(PhaseFlipRuntime._execute_body)
        release_at = body_src.find("self._release_residents_for_cutover(")
        readmit_at = body_src.find("self._post_cutover_readmit(")
        self.assertGreater(release_at, -1)
        self.assertLess(release_at, readmit_at)


class TestRankUniformity(CustomTestCase):
    def test_the_rebuilt_queue_is_a_pure_function_of_arrival_order(self):
        # Every rank retracts the same resident set (group-unanimous cutover)
        # and assigns `kv_arrival_seq` identically ("the admission order is
        # identical on every TP rank, so the counter is rank-uniform"). So
        # each rank must rebuild the SAME queue -- modelled here as two ranks
        # receiving the same requests in different slot-enumeration orders.
        rank_a = _Sched()
        rank_b = _Sched()
        rank_a.readmit_seam_residents([_Req("x", 5), _Req("y", 6), _Req("z", 7)])
        rank_b.readmit_seam_residents([_Req("z", 7), _Req("x", 5), _Req("y", 6)])
        self.assertEqual(
            [r.rid for r in rank_a.waiting_queue],
            [r.rid for r in rank_b.waiting_queue],
        )


if __name__ == "__main__":
    unittest.main()
