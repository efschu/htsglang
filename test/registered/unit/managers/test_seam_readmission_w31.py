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
  * at the FRONT, as a block, in original arrival order -- they are the oldest
    work and the flip's own justification, and appending them behind new
    arrivals lets a busy instance starve the bundle it just flipped for;
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


class _Sched:
    """A scheduler stand-in exposing only what the re-admission touches.

    `readmit_seam_residents` is taken from the REAL class and bound here, so
    this exercises the shipped code rather than a restatement of it.
    """

    readmit_seam_residents = Scheduler.readmit_seam_residents

    def __init__(self, queue=()):
        self.waiting_queue = list(queue)
        self.added = []
        self.refuse = set()

    def _add_request_to_queue(self, req, is_retracted=False):
        # Models the real one's contract: it MAY refuse (priority validation,
        # queued-limit abort), and it appends rather than front-inserts.
        self.added.append((req.rid, is_retracted))
        if req.rid in self.refuse:
            return
        self.waiting_queue.append(req)


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


class TestFrontOfQueueAsABlockInArrivalOrder(CustomTestCase):
    def test_they_go_in_front_of_newer_work(self):
        # THE STARVATION DIRECTION. Appended behind new arrivals, the bundle
        # the flip was armed for waits behind the work that arrived while it
        # flipped -- on a busy instance, for ever.
        newer = [_Req("new1", 100), _Req("new2", 101)]
        s = _Sched(newer)
        carried = [_Req("old1", 1), _Req("old2", 2)]
        s.readmit_seam_residents(carried)
        self.assertEqual(
            [r.rid for r in s.waiting_queue], ["old1", "old2", "new1", "new2"]
        )

    def test_arrival_order_is_restored_even_if_released_out_of_order(self):
        # The seam enumerates residents by slot, which is not arrival order.
        s = _Sched()
        s.readmit_seam_residents([_Req("c", 3), _Req("a", 1), _Req("b", 2)])
        self.assertEqual([r.rid for r in s.waiting_queue], ["a", "b", "c"])

    def test_a_request_the_queue_refused_is_not_conjured_in(self):
        # `_add_request_to_queue` may legitimately refuse. The front-insert is
        # identity-keyed on what actually landed, so a refused request must
        # not appear -- inventing it would be worse than dropping it.
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
    """CAN-FAIL: dropping the return value again must turn this red."""

    def test_the_release_site_requeues_what_it_released(self):
        import inspect

        from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

        src = inspect.getsource(PhaseFlipRuntime._release_residents_for_cutover)
        self.assertIn("readmit_seam_residents", src)
        self.assertIn("released", src)

    def test_the_seam_asserts_retracted_equals_readmitted(self):
        import inspect

        from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

        src = inspect.getsource(PhaseFlipRuntime._release_residents_for_cutover)
        self.assertIn("RE-ADMISSION MISMATCH", src)

    def test_the_requeue_happens_at_the_release_site_not_later(self):
        # THE ABORT PATH, pinned structurally. Requeuing here means there is
        # no window in which the list exists and is owned by nobody: if the
        # cutover raises after this point the flip abandons, the layout is
        # unchanged, and the requests are already back on the SOURCE layout's
        # queue. Deferring the requeue to the end of the cutover would
        # recreate the W31 defect for exactly the abort case.
        import inspect

        from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

        src = inspect.getsource(PhaseFlipRuntime._release_residents_for_cutover)
        readmit_at = src.find("readmit_seam_residents")
        return_at = src.rfind("return released")
        self.assertGreater(readmit_at, -1)
        self.assertLess(
            readmit_at, return_at, "the requeue must happen before the method returns"
        )

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

        src = inspect.getsource(PhaseFlipRuntime._release_residents_for_cutover)
        # the consume happens inside `_retract_and_consume`, which is passed
        # to `release_residents_for_cutover` -- both must precede the requeue
        release_at = src.find("release_residents_for_cutover(")
        readmit_at = src.find("readmit_seam_residents")
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
