# SPDX-License-Identifier: Apache-2.0
"""#830 F5: the cutover sub-step marks must ATTRIBUTE the cutover term.

WHY THIS FILE EXISTS. ANALYSE_830 section 6.3 ran ``grep -c '#690'`` over six
boot logs, got 0, and concluded the #690 marks "are not present in these
boots" -- so the cutover term was filed as UNMEASURABLE (open item O4), and
F2 was aimed at #719's pool rebind on the strength of that gap.

Both halves of that were wrong, and re-measurement showed it:

* The marks WERE emitting, three per flip, in every one of those logs. The
  emitted line read ``PHASE-FLIP CUTOVER SUB-STEPS ...`` and never contained
  the token ``#690`` at all -- that token existed only in the source comment.
  The grep could not have matched. An instrument whose documented grep does
  not match its own output is not an instrument.
* Once the real string is grepped, the term attributes immediately, and NOT
  to the rebind. Per boot, PP0, mean ms:

      boot_735_nohc2   HiCache off   cutover   50.2   draft_state   18.9
      boot_735_hc1     HiCache on    cutover 3448.9   draft_state 3417.3
      boot_735_acc767  HiCache on    cutover 3773.0   draft_state 3741.2

  ``draft_state`` is 99.98% of the term. ``verify+publish+trace``, which is
  the step CONTAINING #719's rebind, measures 0.25-0.7 ms mean (max 2.4 ms)
  on both sides of the flag.

So the single ``draft_state`` bucket was actively misleading: it covered a
176-line span containing eight distinct calls, and it hid the only one that
moves behind a label naming one that does not. The two gates below are the
two ways that failure can come back.
"""

import re
import unittest
import unittest.mock

from sglang.srt.layers.dcp.phase_flip_plan import PP_TO_TP, TP_TO_PP
from sglang.srt.managers import phase_flip_runtime as _rt
from sglang.srt.managers.phase_flip_runtime import (
    PhaseFlipRuntime,
    build_production_flip_cutover,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

from test_phase_flip_protocol import (  # noqa: E402  (sibling harness)
    _ParallelStatePatch,
    _PublishedArgsPatch,
    _StubScheduler,
)

_LINE = re.compile(
    r"CUTOVER SUB-STEPS (?P<dir>\w+) total=(?P<total>[0-9.]+) ms \| (?P<steps>.*)"
)

# The calls the 7b..8 span makes, per leg. A label per call is the whole
# point: anything coarser cannot answer "which step moved".
_TP_LEG_STEPS = {"retune", "spill_rung2", "rung4_cold_stack", "draft_bootstrap"}
_PP_LEG_STEPS = {"retune", "spill_rung2", "spec_clear", "relay_reseed"}
_BOTH_LEGS = {"kv_recover", "abort_drain", "stack_swap"}


def _run_cutover(direction):
    """Drive the REAL production cutover and return (raw_line, steps, total)."""
    with _ParallelStatePatch(), _PublishedArgsPatch():
        sched = _StubScheduler(0)
        cutover = build_production_flip_cutover(sched)
        with unittest.mock.patch.object(
            __import__(
                "sglang.srt.managers.phase_flip_runtime", fromlist=["logger"]
            ).logger,
            "warning",
        ) as warn:
            cutover(direction)
        for call in warn.call_args_list:
            rendered = call.args[0] % tuple(call.args[1:])
            if "CUTOVER SUB-STEPS" in rendered:
                m = _LINE.search(rendered)
                assert m, rendered
                steps = {}
                for part in m.group("steps").split(", "):
                    k, v = part.split("=")
                    steps[k] = float(v)
                return rendered, steps, float(m.group("total"))
    raise AssertionError(f"no CUTOVER SUB-STEPS line emitted for {direction}")


class TestCutoverSubStepAttribution(CustomTestCase):
    def test_green_both_legs_emit_and_the_marks_partition_the_term(self):
        """The instrument exists, fires on both legs, and its parts sum to
        its whole -- so a term can be attributed rather than guessed at."""
        for direction, leg_steps in (
            (PP_TO_TP, _TP_LEG_STEPS),
            (TP_TO_PP, _PP_LEG_STEPS),
        ):
            line, steps, total = _run_cutover(direction)
            self.assertIn("CUTOVER SUB-STEPS", line)
            missing = (leg_steps | _BOTH_LEGS) - set(steps)
            self.assertEqual(
                set(), missing, f"{direction}: unmarked call(s) {missing}"
            )
            # The marks are consecutive perf_counter deltas over one span, so
            # they must account for the whole of it. A gap here means a call
            # is running inside no labelled step -- exactly the blind spot
            # this file exists to prevent.
            self.assertAlmostEqual(
                total, sum(steps.values()), delta=max(0.5, total * 0.02)
            )

    def test_can_fail_the_span_is_not_one_coarse_bucket(self):
        """RED ARM for the shape that hid the cost. ``draft_state`` was ONE
        label over eight calls and carried 99.98% of the term. If it ever
        returns as a single label, the attribution is blind again."""
        for direction in (PP_TO_TP, TP_TO_PP):
            _, steps, _ = _run_cutover(direction)
            self.assertNotIn(
                "draft_state",
                steps,
                f"{direction}: the 7b..8 span is a single bucket again; it "
                "spans eight calls and hid recover_kv_backing for a week",
            )
            # Nine boundaries, not two: the span's own call sites.
            self.assertGreaterEqual(len(steps), 9, f"{direction}: {steps}")

    def test_can_fail_the_documented_grep_matches_the_emitted_line(self):
        """RED ARM for the false indicator itself. ANALYSE_830's own
        reproduction command greps '#690'. If the token is not IN the line,
        the grep returns 0 against a fully working instrument and the next
        reader concludes the marks are missing -- which is what happened."""
        for direction in (PP_TO_TP, TP_TO_PP):
            line, _, _ = _run_cutover(direction)
            self.assertIn(
                "#690",
                line,
                "the emitted line must carry its own ticket token; a grep "
                "for '#690' returned 0 across six boots that were emitting "
                "this line three times per flip",
            )


class TestHostRamDeferIsCounted(CustomTestCase):
    """#830 F6: ANALYSE_830 open item O1 -- "#721 defer frequency: UNMEASURED.
    No log line counted."

    The path was never silent; it was UNCOUNTABLE. The ``HOST HEADROOM`` line
    fires on every flip regardless of verdict, so its count is the flip count.
    The defer's own text was folded into that same info-level line and carried
    no running total, so a boot log could not answer "how often did this fire".

    It matters because a defer makes the rank vote the flip unfit, which ends
    its armed window early and RANK-LOCALLY -- host RAM is measured per rank,
    so peers need not agree. That is the resume-slot divergence shape.
    """

    @staticmethod
    def _runtime():
        """A bare instance: these recorders touch only their own counters and
        the module logger, which is the point of lifting them out of
        ``_execute`` (that method needs a live group to run at all)."""
        return PhaseFlipRuntime.__new__(PhaseFlipRuntime)

    def _emit(self, fn, *args):
        with unittest.mock.patch.object(_rt.logger, "warning") as warn:
            total = fn(*args)
        lines = [c.args[0] % tuple(c.args[1:]) for c in warn.call_args_list]
        return total, lines

    def test_green_defers_are_counted_and_greppable(self):
        rt = self._runtime()
        rt._host_ram_defers = 1
        total, lines = self._emit(rt.record_host_ram_defer, "tp_to_pp", "detail")
        self.assertEqual(1, total)
        self.assertEqual(1, len(lines))
        self.assertIn("[#721] HOST-RAM DEFER", lines[0])
        self.assertIn("lifetime=1", lines[0])
        # The rank-local consequence must be IN the line: a reader counting
        # these is looking for early rank-local abandons, not for a memory
        # statistic.
        self.assertIn("RANK-LOCALLY", lines[0])

        rt._host_ram_defers = 2
        total, lines = self._emit(rt.record_host_ram_defer, "tp_to_pp", "detail")
        self.assertEqual(2, total, "the lifetime total must accumulate")
        self.assertIn("lifetime=2", lines[0])
        self.assertIn("consecutive=2 of 3", lines[0])

    def test_green_escalation_is_counted_separately_from_defer(self):
        """A defer retries; an escalation proceeds anyway under a floor the
        guard just failed. One merged counter would hide which the boot took."""
        rt = self._runtime()
        self._emit(rt.record_host_ram_defer, "pp_to_tp", "d")
        self._emit(rt.record_host_ram_defer, "pp_to_tp", "d")
        total, lines = self._emit(rt.record_host_ram_escalation, "pp_to_tp", "e")
        self.assertEqual(1, total)
        self.assertIn("[#721] HOST-RAM ESCALATION", lines[0])
        self.assertIn("lifetime=1", lines[0])
        self.assertEqual(2, rt._host_ram_defer_total, "counters must not share")

    def test_can_fail_the_defer_is_not_emitted_below_warning(self):
        """RED ARM. The reason O1 stayed open is that this path spoke at INFO
        inside a line that fires every flip. A defer that only whispers is a
        defer nobody counts."""
        rt = self._runtime()
        rt._host_ram_defers = 1
        with unittest.mock.patch.object(_rt.logger, "info") as info:
            with unittest.mock.patch.object(_rt.logger, "warning") as warn:
                rt.record_host_ram_defer("tp_to_pp", "detail")
        self.assertEqual(
            1, warn.call_count, "the counted defer line must be a WARNING"
        )
        self.assertEqual(
            0, info.call_count, "a defer must not be reported only at info"
        )

    def test_can_fail_the_call_site_still_routes_through_the_recorder(self):
        """RED ARM against re-inlining. The counter is only worth anything if
        the production path increments it; a call site that logs its own
        string again leaves the lifetime total at zero forever."""
        import inspect

        src = inspect.getsource(PhaseFlipRuntime._execute)
        self.assertIn("record_host_ram_defer", src)
        self.assertIn("record_host_ram_escalation", src)


class TestSeamDrainBudget(CustomTestCase):
    """#830 F4: "Nothing in the tree caps how long the seam may hold the ring.
    A named ceiling converts a silent 24-second exposure into a refusal that
    names itself."

    The exposure is measured, not hypothetical: single flips reaching 24892 ms
    and single cutovers 13742 ms, against a HiCache-off corpus in which 1014
    flips never exceeded 4155 ms total or 1094 ms of cutover.
    """

    def test_green_the_budget_refuses_by_name_above_it_and_allows_below(self):
        allow, escalated, detail = _rt.flip_seam_budget_verdict(500.0, 0, budget_ms=1094)
        self.assertTrue(allow)
        self.assertFalse(escalated)

        allow, escalated, detail = _rt.flip_seam_budget_verdict(
            5000.0, 0, budget_ms=1094
        )
        self.assertFalse(allow, "a drain over budget must refuse the arm")
        self.assertFalse(escalated)
        # NAMED is the whole feature. A refusal nobody can grep is the silent
        # exposure it replaces, wearing a different shape.
        self.assertIn(_rt.SEAM_BUDGET_REFUSED, detail)
        self.assertIn("5000.0", detail)
        self.assertIn("1094", detail)

    def test_green_the_refusal_is_bounded_and_escalates_rather_than_holding(self):
        """A permanent refusal would stop the instance alternating prefill and
        decode -- a certain outage traded for a latency hazard. #721 settled
        that trade; this guard inherits the settlement."""
        for defers in range(_rt.FLIP_SEAM_BUDGET_MAX_DEFERS):
            allow, _, _ = _rt.flip_seam_budget_verdict(
                9999.0, defers, budget_ms=1094
            )
            self.assertFalse(allow, f"defer {defers} should still refuse")
        allow, escalated, detail = _rt.flip_seam_budget_verdict(
            9999.0, _rt.FLIP_SEAM_BUDGET_MAX_DEFERS, budget_ms=1094
        )
        self.assertTrue(allow, "the guard must stand aside, never hold forever")
        self.assertTrue(escalated)
        self.assertIn("ESCALATED", detail)

    def test_green_an_unknown_never_produces_a_refusal(self):
        """#721's rule, applied unchanged: a refusal is the thing with a
        service cost, so it must never rest on a number we do not have."""
        allow, _, detail = _rt.flip_seam_budget_verdict(None, 0, budget_ms=1094)
        self.assertTrue(allow)
        self.assertIn("stood down", detail)

        allow, _, detail = _rt.flip_seam_budget_verdict(9999.0, 0, budget_ms=0)
        self.assertTrue(allow, "budget 0 disables the guard")
        self.assertIn("stood down", detail)

    def test_can_fail_the_budget_is_actually_compared(self):
        """RED ARM. Delete the comparison and this is a logging change wearing
        a guard's name -- it would allow every drain while still emitting a
        reassuring 'seam drain OK' line."""
        over = _rt.flip_seam_budget_verdict(1e9, 0, budget_ms=1094)
        under = _rt.flip_seam_budget_verdict(1.0, 0, budget_ms=1094)
        self.assertNotEqual(
            over[0],
            under[0],
            "the verdict must DEPEND on the projection vs the budget; if a "
            "drain of 1e9 ms and one of 1 ms get the same answer, the budget "
            "is decorative",
        )

    def test_can_fail_the_default_budget_is_derived_not_absent(self):
        """The default must exist and be the documented corpus maximum. A 0
        default would ship the guard permanently stood down, which is the
        silent exposure again."""
        self.assertEqual(1094, _rt.flip_seam_drain_budget_ms())

    def test_can_fail_the_guard_is_wired_into_the_arm_path(self):
        """RED ARM. A verdict function nothing calls caps nothing. The arm
        path must consult it AND vote through ``too_small``, which is what
        makes a refusal actually refuse."""
        import inspect

        src = inspect.getsource(PhaseFlipRuntime._execute)
        self.assertIn("flip_seam_budget_verdict", src)
        self.assertIn("_seam_budget_defers", src)
        # The vote is the actuator; without it the guard only narrates.
        seam_block = src[src.index("flip_seam_budget_verdict") :]
        self.assertIn("too_small.append(seam_detail)", seam_block)

    def test_can_fail_the_drain_measurement_reaches_the_seam_record(self):
        """RED ARM for F1. ``quiesce_device_io`` documents that it "Returns
        the wait in seconds; the caller logs it into the seam record" -- and
        the caller discarded it, so the drain vanished into the residual and
        F1 was filed against it on docstring evidence alone."""
        import inspect

        src = inspect.getsource(PhaseFlipRuntime._quiesce_hicache)
        self.assertIn("return drain_ms", src)
        self.assertIn("SEAM DRAIN", src)
        stats_src = inspect.getsource(PhaseFlipRuntime)
        self.assertIn('"drain_ms"', stats_src)

    def test_can_fail_the_drain_itself_is_still_performed_at_the_seam(self):
        """RED ARM, and the most important one in this file: CORRECTNESS
        BEFORE SPEED.

        The tempting version of F1 is to delete the drain -- it is a blocking
        wait in the no-return window, and on the current tree it measures
        0.0 ms, so it looks free to remove. It is not. It exists because of two
        live SIGSEGVs on 2026-08-19, each three seconds after a cutover, a
        HiCache copy reaching pool memory the seam had already released. Under
        ``layer_first`` the host layout equals the device layout, so the stale
        binding is SHAPE-IDENTICAL to the live one and ``check_shapes`` passes
        by construction -- no Python-side predicate can catch it.

        So the drain must still be CALLED at the no-return point. Its cost is
        now measured (above) and budgeted (F4); it is not removed."""
        import inspect

        src = inspect.getsource(PhaseFlipRuntime._execute)
        self.assertIn(
            "_quiesce_hicache",
            src,
            "the #760 device-tier drain must still run at the seam; removing "
            "it re-opens two SIGSEGVs that a shape check cannot catch",
        )
        # And it must be armed BEFORE the drain: the guard refuses NEW I/O,
        # the drain finishes OLD I/O, and both must hold before the first wave.
        self.assertLess(
            src.index("hicache_seam_active = True"),
            src.index("_quiesce_hicache"),
            "the seam guard must be up before the drain, or new I/O can be "
            "enqueued behind the drain it was supposed to cover",
        )


register_cpu_ci(__file__)

if __name__ == "__main__":
    unittest.main()
