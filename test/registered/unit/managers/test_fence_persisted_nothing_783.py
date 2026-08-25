"""#783: a fence that RAN and persisted NOTHING must not read as a healthy fence.

THE SPECIMEN. `boot_w37g_0825_1326.log`, 39 cutovers in 5.5 min, zero
completions. Every fence in the whole boot:

    33 x eligible=0 staged=0 already_staged=0 acked=0 outstanding=0 elapsed=0.000s/2.000s
     3 x eligible=3 staged=0 already_staged=3 acked=0 outstanding=2 elapsed=0.000s/2.000s
     3 x eligible=1 staged=0 already_staged=1 acked=0 outstanding=1 elapsed=0.000s/2.000s

`acked=0` in every single one. Not one prefix was ever confirmed into the
canonical store, and `#cached-token: 0` on all 209 prefill batch lines follows.

WHY NOTHING SAW IT. The seam's guard (`phase_flip_runtime`, under the header
"#703 FENCE COVERAGE, ASSERTED RATHER THAN ASSUMED") warns only when
`_writeback_fence_ms(...) is None`. That helper returns `elapsed_s`, so the
33 empty fences returned **0.0, not None**, and the guard stayed silent on all
of them. `FlipWritebackReport.complete` is `outstanding == 0`, so those same 33
also logged at INFO as COMPLETE -- a fence that persisted nothing reported
success twice.

`test_writeback_fence_ms_856.py` names this exact outcome in its own docstring:
"a defaulted zero would report such a flip as fully fenced while nothing was
persisted". It then pins the wrong variable. It guards against a DEFAULTED
zero; W37-G produced a GENUINE zero, from an empty tree, and arrived at the
outcome that file exists to prevent by a path it never considered. A cheap
fence and a healthy fence are indistinguishable by cost alone, because the
healthy case is also cheap.

THE PREDICATE THIS FILE PINS. "Did a fence run" is not the question the seam
needs answered. The question is "is anything of these residents' prefixes
retrievable after the tree is dropped", and the report already carries the
counts to answer it -- nothing read them.

Deliberately NARROW. `persisted_nothing` is True only when NOTHING is
retrievable by either route: no storage acknowledgement AND no pre-existing
host copy. The six fences with `already_staged>0, acked=0, outstanding>0` are
NOT claimed by this predicate -- those nodes have a host copy that may still
serve the read-through, and they already draw the "deadline reached with
backups still in flight" warning, which did fire 6 times. Widening the
predicate to cover them would be a second crying-wolf gate, which the same law
that motivates this one forbids.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest

from sglang.srt.mem_cache.hicache_flip_writeback import FlipWritebackReport
from sglang.test.test_utils import CustomTestCase


def _w37g_empty_fence() -> FlipWritebackReport:
    """The shape 33 of W37-G's 39 fences had, verbatim from the log."""
    return FlipWritebackReport(
        eligible=0,
        staged=0,
        already_staged=0,
        acknowledged=0,
        outstanding=0,
        elapsed_s=0.000,
        deadline_s=2.000,
    )


def _w37g_lapsed_fence() -> FlipWritebackReport:
    """The shape the other 6 had: a host copy exists, storage never acked."""
    return FlipWritebackReport(
        eligible=3,
        staged=0,
        already_staged=3,
        acknowledged=0,
        outstanding=2,
        elapsed_s=0.000,
        deadline_s=2.000,
    )


def _healthy_fence() -> FlipWritebackReport:
    return FlipWritebackReport(
        eligible=12,
        staged=12,
        already_staged=0,
        acknowledged=12,
        outstanding=0,
        elapsed_s=0.0748,
        deadline_s=2.000,
    )


class TestAnEmptyFenceIsNotAHealthyFence(CustomTestCase):
    def test_the_w37g_empty_fence_reports_persisted_nothing(self):
        # THE RED. 33 of 39 cutovers looked exactly like this and every
        # instrument in the tree called them healthy.
        self.assertTrue(_w37g_empty_fence().persisted_nothing)

    def test_a_healthy_fence_does_not(self):
        self.assertFalse(_healthy_fence().persisted_nothing)

    def test_a_lapsed_fence_with_a_host_copy_is_not_claimed(self):
        # NARROWNESS, PINNED. These already draw the "deadline reached with
        # backups still in flight" warning; claiming them here would double-
        # report one condition and turn this into a crying-wolf gate.
        self.assertFalse(_w37g_lapsed_fence().persisted_nothing)

    def test_an_ack_alone_is_enough(self):
        self.assertFalse(
            FlipWritebackReport(
                eligible=1,
                staged=1,
                already_staged=0,
                acknowledged=1,
                outstanding=0,
                elapsed_s=0.01,
                deadline_s=2.0,
            ).persisted_nothing
        )

    def test_a_host_copy_alone_is_enough(self):
        self.assertFalse(
            FlipWritebackReport(
                eligible=1,
                staged=0,
                already_staged=1,
                acknowledged=0,
                outstanding=0,
                elapsed_s=0.01,
                deadline_s=2.0,
            ).persisted_nothing
        )


class TestCompleteWasNeverThisQuestion(CustomTestCase):
    """`complete` is `outstanding == 0`. It cannot answer this and must not be
    silently repurposed to -- pinned so a later reader does not delete
    `persisted_nothing` as a duplicate of it."""

    def test_the_empty_fence_is_complete_and_still_persisted_nothing(self):
        report = _w37g_empty_fence()
        self.assertTrue(report.complete)
        self.assertTrue(report.persisted_nothing)

    def test_the_lapsed_fence_is_incomplete_and_still_persisted_something(self):
        report = _w37g_lapsed_fence()
        self.assertFalse(report.complete)
        self.assertFalse(report.persisted_nothing)


class TestTheSeamActuallyWarns(CustomTestCase):
    """THE WIRING, EXECUTED -- not inspected.

    Every existing test of `_release_residents_for_cutover` (w31 readmission,
    tree-drop-returns-rows) asserts against `inspect.getsource`. Source
    inspection cannot distinguish a guard that runs from a guard that is
    unreachable, which is precisely the desk-written-never-executed shape. So
    this drives the real method and reads the real log record.
    """

    def _run_seam(self, report):
        import types
        from unittest import mock

        from sglang.srt.managers import phase_flip_runtime as pfr

        runtime = pfr.PhaseFlipRuntime.__new__(pfr.PhaseFlipRuntime)
        runtime._last_writeback_report = report

        readmitted = []
        scheduler = types.SimpleNamespace(
            readmit_seam_residents=lambda reqs: (readmitted.extend(reqs), len(reqs))[1]
        )
        runtime._census_scheduler = scheduler

        reqs = [types.SimpleNamespace(rid="r0"), types.SimpleNamespace(rid="r1")]

        with mock.patch.object(
            pfr, "_live_reqs", return_value=list(reqs)
        ), mock.patch.object(
            pfr, "build_cutover_release", return_value=(lambda rs: list(rs), lambda: 0)
        ), mock.patch.object(
            pfr, "consume_retracted_from_live_universe", return_value=0
        ):
            with self.assertLogs(pfr.logger, level="WARNING") as caught:
                # A DEBUG record so assertLogs never fails for want of any
                # record at all -- the assertion below is on CONTENT.
                pfr.logger.warning("probe: seam driven")
                runtime._release_residents_for_cutover("pp_to_tp")
        return "\n".join(caught.output), readmitted

    def test_the_empty_fence_makes_the_seam_warn(self):
        output, readmitted = self._run_seam(_w37g_empty_fence())
        self.assertIn("#783", output)
        self.assertIn("RAN AND PERSISTED NOTHING", output)
        # The warning must not cost the re-admission -- it is an instrument.
        self.assertEqual(len(readmitted), 2)

    def test_a_healthy_fence_makes_the_seam_silent(self):
        # THE CAN-FAIL DIRECTION. A gate that fires on everything is not a
        # gate; INDIKATOR-GESETZ requires both directions be shown.
        output, readmitted = self._run_seam(_healthy_fence())
        self.assertNotIn("PERSISTED NOTHING", output)
        self.assertEqual(len(readmitted), 2)

    def test_the_lapsed_fence_makes_the_seam_silent_on_this_gate(self):
        output, _ = self._run_seam(_w37g_lapsed_fence())
        self.assertNotIn("PERSISTED NOTHING", output)


if __name__ == "__main__":
    unittest.main()
