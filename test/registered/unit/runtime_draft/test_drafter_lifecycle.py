"""The attach/detach state machine under synthetic in-flight load (#309).

The three hard cases the task names are all "a transition requested while the
engine is in a state that makes it unsafe", and all three are decidable from a
quiesce report -- so all three are pinned here, on CPU, against synthetic load
rather than against a real batch.

The regression that matters most is the last class: with no drafter attached
the machine must be inert, so a server that never attaches one behaves exactly
as a spec-less boot.
"""

import unittest

from sglang.srt.speculative.runtime_draft import (
    DrafterBusy,
    DrafterLifecycle,
    DrafterState,
    DrafterStateError,
    DrafterUnsupported,
    QuiesceReport,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

IDLE = QuiesceReport()
BUSY = QuiesceReport(running_requests=3)
VERIFYING = QuiesceReport(running_requests=1, spec_verify_in_flight=True)
CAPTURING = QuiesceReport(graph_active=True)

# Each blocking condition ISOLATED -- only that one field set. Without these
# a report that trips two conditions passes even when one of them has been
# removed from the predicate, which is how a safety condition rots unnoticed.
ONLY_VERIFY = QuiesceReport(running_requests=0, spec_verify_in_flight=True)
ONLY_GRAPH = QuiesceReport(running_requests=0, graph_active=True)
ONLY_RUNNING = QuiesceReport(running_requests=1)
SPEC = {"algorithm": "NEXTN", "k": 3}


class TestHappyPath(CustomTestCase):
    def test_attach_lands_at_the_boundary(self):
        lc = DrafterLifecycle()
        self.assertIs(lc.state, DrafterState.DETACHED)
        lc.request_attach(SPEC)
        self.assertIs(lc.state, DrafterState.ATTACH_PENDING)
        out = lc.step(IDLE)
        self.assertEqual(out.executed, "attach")
        self.assertIs(lc.state, DrafterState.ATTACHED)

    def test_detach_lands_at_the_boundary(self):
        lc = DrafterLifecycle(state=DrafterState.ATTACHED)
        lc.request_detach()
        out = lc.step(IDLE)
        self.assertEqual(out.executed, "detach")
        self.assertIs(lc.state, DrafterState.DETACHED)

    def test_a_boundary_with_nothing_pending_does_nothing(self):
        for state in (DrafterState.DETACHED, DrafterState.ATTACHED):
            with self.subTest(state=state):
                lc = DrafterLifecycle(state=state)
                out = lc.step(IDLE)
                self.assertIsNone(out.executed)
                self.assertIs(lc.state, state)

    def test_a_transition_executes_exactly_once(self):
        """A second boundary must not re-run the attach: re-loading weights
        over a live drafter is the double-execute this machine exists to make
        impossible."""
        lc = DrafterLifecycle()
        lc.request_attach(SPEC)
        self.assertEqual(lc.step(IDLE).executed, "attach")
        self.assertIsNone(lc.step(IDLE).executed)
        self.assertEqual(lc.history, ["attach"])


class TestAttachWhileRequestsInFlight(CustomTestCase):
    """Hard case 1: the attach is HELD, visibly, not dropped and not early."""

    def test_attach_defers_while_requests_run(self):
        lc = DrafterLifecycle()
        lc.request_attach(SPEC)
        out = lc.step(BUSY)
        self.assertIsNone(out.executed)
        self.assertIs(lc.state, DrafterState.ATTACH_PENDING)
        self.assertIn("still running", out.detail)

    def test_it_lands_on_the_first_quiesced_boundary(self):
        lc = DrafterLifecycle()
        lc.request_attach(SPEC)
        for _ in range(5):
            self.assertIsNone(lc.step(BUSY).executed)
        self.assertEqual(lc.step(IDLE).executed, "attach")

    def test_waiting_requests_alone_do_not_block(self):
        """A waiting request has not been scheduled, so it will pick up
        whatever configuration is live when it is."""
        lc = DrafterLifecycle()
        lc.request_attach(SPEC)
        out = lc.step(QuiesceReport(running_requests=0, waiting_requests=9))
        self.assertEqual(out.executed, "attach")

    def test_a_pending_attach_does_not_serve_drafts(self):
        """The weights are not in yet. Serving from ATTACH_PENDING would be a
        forward against a drafter that does not exist."""
        lc = DrafterLifecycle()
        lc.request_attach(SPEC)
        self.assertFalse(lc.serves_drafts)


class TestDetachMidVerify(CustomTestCase):
    """Hard case 2: freeing the draft pool under a verify that still reads it."""

    def test_detach_defers_while_a_verify_is_in_flight(self):
        lc = DrafterLifecycle(state=DrafterState.ATTACHED)
        lc.request_detach()
        out = lc.step(VERIFYING)
        self.assertIsNone(out.executed)
        self.assertIn("verify", out.detail)

    def test_a_pending_detach_STILL_serves_drafts(self):
        """The in-flight verify that is keeping us here is entitled to finish
        against the drafter. Getting this backwards frees the pool under it."""
        lc = DrafterLifecycle(state=DrafterState.ATTACHED)
        lc.request_detach()
        self.assertIs(lc.state, DrafterState.DETACH_PENDING)
        self.assertTrue(lc.serves_drafts)

    def test_detach_defers_while_a_graph_is_active(self):
        lc = DrafterLifecycle(state=DrafterState.ATTACHED)
        lc.request_detach()
        self.assertIsNone(lc.step(CAPTURING).executed)
        self.assertIn("CUDA graph", lc.step(CAPTURING).detail)

    def test_it_lands_once_the_verify_completes(self):
        lc = DrafterLifecycle(state=DrafterState.ATTACHED)
        lc.request_detach()
        self.assertIsNone(lc.step(VERIFYING).executed)
        self.assertEqual(lc.step(IDLE).executed, "detach")


class TestUnsupportedModelRefusal(CustomTestCase):
    """Hard case 3: the parse-time refusal, arriving at runtime."""

    def _lc(self, reason):
        return DrafterLifecycle(supports_spec=lambda spec: reason)

    def test_an_unsupported_spec_is_refused_by_name(self):
        lc = self._lc("model architecture has no MTP head")
        with self.assertRaises(DrafterUnsupported) as cm:
            lc.request_attach(SPEC)
        self.assertIn("no MTP head", str(cm.exception))

    def test_a_refused_attach_leaves_the_state_untouched(self):
        lc = self._lc("nope")
        with self.assertRaises(DrafterUnsupported):
            lc.request_attach(SPEC)
        self.assertIs(lc.state, DrafterState.DETACHED)
        self.assertIsNone(lc.pending)

    def test_the_refusal_type_says_retrying_will_not_help(self):
        """Distinct from DrafterBusy on purpose: one is 'later', the other is
        'never on this server', and an operator's next action differs."""
        lc = self._lc("unsupported")
        with self.assertRaises(DrafterUnsupported) as cm:
            lc.request_attach(SPEC)
        self.assertNotIsInstance(cm.exception, DrafterBusy)


class TestConcurrentAndInapplicableRequests(CustomTestCase):
    def test_two_pending_transitions_are_refused(self):
        lc = DrafterLifecycle()
        lc.request_attach(SPEC)
        with self.assertRaises(DrafterBusy):
            lc.request_attach(SPEC)
        with self.assertRaises(DrafterBusy):
            lc.request_detach()

    def test_attach_when_already_attached_is_refused_with_the_alternative(self):
        lc = DrafterLifecycle(state=DrafterState.ATTACHED)
        with self.assertRaises(DrafterStateError) as cm:
            lc.request_attach(SPEC)
        # points at the cheaper operation instead of just refusing
        self.assertIn("selection endpoint", str(cm.exception))

    def test_detach_when_nothing_is_attached_is_refused(self):
        with self.assertRaises(DrafterStateError):
            DrafterLifecycle().request_detach()

    def test_a_pending_transition_can_be_cancelled(self):
        """A boundary that never arrives under sustained load must be
        escapable without a reboot -- the thing this task removes."""
        lc = DrafterLifecycle()
        lc.request_attach(SPEC)
        self.assertIsNone(lc.step(BUSY).executed)
        req = lc.cancel_pending()
        self.assertEqual(req.action, "attach")
        self.assertIs(lc.state, DrafterState.DETACHED)
        self.assertIsNone(lc.step(IDLE).executed)

    def test_cancelling_a_detach_returns_to_attached(self):
        lc = DrafterLifecycle(state=DrafterState.ATTACHED)
        lc.request_detach()
        lc.cancel_pending()
        self.assertIs(lc.state, DrafterState.ATTACHED)

    def test_cancel_with_nothing_pending_is_a_no_op(self):
        self.assertIsNone(DrafterLifecycle().cancel_pending())


class TestBoundaryIsEnforced(CustomTestCase):
    """#364: the executor refuses to run outside the window it is given, which
    is what makes the placement enforced rather than documented."""

    def test_step_without_a_quiesce_report_is_a_type_error(self):
        lc = DrafterLifecycle()
        lc.request_attach(SPEC)
        for bad in (None, True, {"running_requests": 0}):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError) as cm:
                    lc.step(bad)
                self.assertIn("#364", str(cm.exception))

    def test_quiesce_predicate_matches_its_reasons(self):
        for report in (BUSY, VERIFYING, CAPTURING):
            with self.subTest(report=report):
                self.assertFalse(report.is_quiesced)
                self.assertNotEqual(report.why_not_quiesced(), "quiesced")
        self.assertTrue(IDLE.is_quiesced)
        self.assertEqual(IDLE.why_not_quiesced(), "quiesced")

    def test_every_blocking_condition_blocks_ON_ITS_OWN(self):
        """Each condition carries its own weight.

        A report that trips two conditions still fails the predicate when one
        of them is deleted, so a combined report cannot prove that either is
        load-bearing -- it only proves their disjunction is. These three pin
        them individually, which is what makes deleting any one of them a red
        test rather than a silent loss of a safety condition.
        """
        for report, needle in (
            (ONLY_RUNNING, "still running"),
            (ONLY_VERIFY, "verify"),
            (ONLY_GRAPH, "CUDA graph"),
        ):
            with self.subTest(report=report):
                self.assertFalse(report.is_quiesced)
                self.assertIn(needle, report.why_not_quiesced())

    def test_an_isolated_verify_defers_a_detach(self):
        """The dangerous shape: the batch is retired but a verify is still
        outstanding. Freeing the draft pool here reads-after-free."""
        lc = DrafterLifecycle(state=DrafterState.ATTACHED)
        lc.request_detach()
        self.assertIsNone(lc.step(ONLY_VERIFY).executed)
        self.assertEqual(lc.step(IDLE).executed, "detach")

    def test_an_isolated_graph_defers_a_transition(self):
        lc = DrafterLifecycle()
        lc.request_attach(SPEC)
        self.assertIsNone(lc.step(ONLY_GRAPH).executed)
        self.assertEqual(lc.step(IDLE).executed, "attach")

    def test_the_reason_mentions_waiting_load_without_blocking_on_it(self):
        r = QuiesceReport(running_requests=2, waiting_requests=17)
        self.assertIn("17 more waiting", r.why_not_quiesced())


class TestDetachedIsInert(CustomTestCase):
    """THE REGRESSION PIN: with no drafter attached, nothing here does
    anything, so the server behaves as a spec-less boot."""

    def test_a_detached_machine_never_serves_drafts(self):
        self.assertFalse(DrafterLifecycle().serves_drafts)

    def test_stepping_a_detached_machine_forever_changes_nothing(self):
        lc = DrafterLifecycle()
        for report in (IDLE, BUSY, VERIFYING, CAPTURING) * 3:
            out = lc.step(report)
            self.assertIsNone(out.executed)
        self.assertIs(lc.state, DrafterState.DETACHED)
        self.assertEqual(lc.history, [])

    def test_the_snapshot_of_an_untouched_server_is_all_negative(self):
        snap = DrafterLifecycle().snapshot()
        self.assertEqual(snap["state"], "detached")
        self.assertFalse(snap["serves_drafts"])
        self.assertIsNone(snap["pending"])
        self.assertEqual(snap["transitions"], [])


class TestSnapshot(CustomTestCase):
    def test_snapshot_reports_the_pending_action_and_id(self):
        lc = DrafterLifecycle()
        lc.request_attach(SPEC, request_id="req-7")
        snap = lc.snapshot()
        self.assertEqual(snap["pending"], "attach")
        self.assertEqual(snap["pending_request_id"], "req-7")

    def test_history_records_each_completed_transition(self):
        lc = DrafterLifecycle()
        lc.request_attach(SPEC)
        lc.step(IDLE)
        lc.request_detach()
        lc.step(IDLE)
        self.assertEqual(lc.snapshot()["transitions"], ["attach", "detach"])


if __name__ == "__main__":
    unittest.main()
