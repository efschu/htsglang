"""#363 close-out: the ms/round decider HAS an actuator, and the chain stays whole.

THE QUESTION THIS ANSWERS. #578 found a #363 seam that looked wired and was
not: ``build_regime_stage_table`` called ``planner_candidates`` without
``solve_fn``, so the production feed silently produced nothing
(``AUDIT_421_UNWIRED.md`` B.9, now pinned by
``test_regime_act.py::TestPlannerFeed``). That was the FEED half. This file
pins the other half, asked the same way: does ``MsStageDecider``'s verdict
reach a real dial/reshard/flip call, or is it another armed-never-executing
shape?

THE ANSWER IS THAT IT IS WIRED, end to end, and every link is checked below:

    MsStageDecider.decide            regime_ms_clock.py
      -> RegimeObserver._intra_phase_decide
      -> overrides `target`          regime_runtime.py
      -> MODE_ACT + _act_interlocks
      -> self._commit_fn(target, ...)
      -> RegimeActuator.apply        regime_act.py
      -> _vram_apply   (#330 dial, kv_capacity_runtime.apply_budget_request)
         _reshard_arm  (#297,       kv_reshard_runtime.arm)
         _phase_flip_arm (#631,     scheduler.arm_phase_flip)

So the verdict is DORMANT BY FLAG, not unwired by defect -- a different state
from #578's, and the distinction is the whole point of asking. On this rig's
turnkey config (``deploy/turnkey/stack.rig3.toml``) neither
``--regime-stage-clock`` nor ``--regime-controller`` is passed, so the decider
is never constructed and no proposal is ever made. Nothing here changes that:
arming the controller on the serving path is a review-and-boot decision, not a
desk one.

WHAT THESE TESTS ARE FOR. They are a ratchet on the chain, not a claim that it
runs. If a future edit breaks any link -- the clock stops being consulted, the
override stops reaching ``_commit_fn``, or an axis quietly becomes a stub
instead of a named refusal -- #363 goes back to the state #578 found, and the
next reader would have to re-derive all of this to notice.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import inspect
import types
import unittest

from sglang.srt.managers import regime_act, regime_runtime
from sglang.test.test_utils import CustomTestCase


class TestTheDeciderReachesTheActuator(CustomTestCase):
    """Link by link, in source, because the chain crosses three modules and no
    hermetic test can run a scheduler."""

    def test_the_clock_verdict_is_consulted_at_the_boundary(self):
        src = inspect.getsource(regime_runtime)
        self.assertIn("ms_decision = self._intra_phase_decide(target)", src)

    def test_the_verdict_can_replace_the_target(self):
        """Not merely recorded. A decision that only lands in a trace row is an
        instrument, not a decider."""
        src = inspect.getsource(regime_runtime)
        self.assertIn("if ms_decision is not None and ms_decision.wants_flip:", src)
        self.assertIn("target = proposed", src)

    def test_the_target_reaches_the_commit_function(self):
        src = inspect.getsource(regime_runtime)
        self.assertIn("result = self._commit_fn(target, self._table", src)

    def test_the_commit_function_is_the_actuator(self):
        src = inspect.getsource(regime_runtime)
        self.assertIn("build_regime_actuator(scheduler, table_plan).apply", src)

    def test_committing_is_act_mode_only(self):
        """The property the no-actuator tests pin: observe holds no path to a
        move. Checked here too, because this file would otherwise read as if
        the chain were always live."""
        src = inspect.getsource(regime_runtime)
        self.assertIn("if self._mode == MODE_ACT and target is not None:", src)


class TestTheActuatorReachesRealCalls(CustomTestCase):
    """The #578 shape was a call that produced nothing. These are the three
    calls that make a move, named at their runtimes."""

    def test_the_vram_axis_calls_the_dial(self):
        src = inspect.getsource(regime_act.build_regime_actuator)
        self.assertIn("apply_budget_request", src)

    def test_the_kv_axis_calls_the_reshard_runtime(self):
        src = inspect.getsource(regime_act.build_regime_actuator)
        self.assertIn("_rt.arm(vector, source=source)", src)

    def test_the_phase_axis_calls_the_flip_arm(self):
        src = inspect.getsource(regime_act.build_regime_actuator)
        self.assertIn("_s.arm_phase_flip(direction, source)", src)

    def test_an_unwired_axis_is_none_and_not_a_stub(self):
        """``None`` rather than a no-op callable is what makes the refusal name
        the missing flag instead of the move half-succeeding in silence."""
        act = regime_act.build_regime_actuator(types.SimpleNamespace())
        self.assertIsNone(act._reshard_arm)
        self.assertIsNone(act._vram_apply)
        self.assertIsNone(act._phase_flip_arm)

    def test_a_fully_unwired_actuator_says_every_proposal_will_be_refused(self):
        act = regime_act.build_regime_actuator(types.SimpleNamespace())
        self.assertEqual(tuple(act.wired_axes), ())

    def test_a_wired_axis_is_reported_as_wired(self):
        """The falsifier for the test above: if wired_axes were always empty,
        the pin would pass against a broken binder."""
        scheduler = types.SimpleNamespace(
            kv_capacity_runtime=types.SimpleNamespace(
                apply_budget_request=lambda budget_mib: (True, "ok")
            )
        )
        act = regime_act.build_regime_actuator(scheduler)
        self.assertIsNotNone(act._vram_apply)
        self.assertTrue(act.wired_axes)


class TestTheRefusalsNameTheMissingFlag(CustomTestCase):
    """An unwired axis must say which flag would wire it. This is what turns a
    dormant controller into a diagnosable one rather than a silent one."""

    def test_the_kv_refusal_names_the_reshard_flag(self):
        src = inspect.getsource(regime_act)
        self.assertIn("--kv-reshard-vectors", src)

    def test_the_vram_refusal_names_the_dial_flag(self):
        src = inspect.getsource(regime_act)
        self.assertIn("--enable-vram-dial", src)

    def test_a_shrink_is_never_autonomous(self):
        """#330's rule, pinned here because the decider is what would
        otherwise trigger one: a shrink flushes the radix cache and only an
        explicit dial authorizes that."""
        src = inspect.getsource(regime_act)
        self.assertIn("would SHRINK a VRAM budget", src)


class TestTheClockIsAnInstrumentInEveryMode(CustomTestCase):
    """#363 defect 7's fix: the clock follows the FLAG, the admission gate stays
    act-only. Pinned because the previous coupling was a bootstrap deadlock --
    the canon needed observe rows that only act mode produced."""

    def test_the_clock_is_built_from_the_flag_not_the_mode(self):
        src = inspect.getsource(regime_runtime)
        self.assertIn(
            'if bool(getattr(server_args, "regime_stage_clock", False)):', src
        )

    def test_the_admission_gate_stays_act_only(self):
        src = inspect.getsource(regime_runtime)
        i = src.index('if bool(getattr(server_args, "regime_stage_clock", False)):')
        self.assertIn("if mode == MODE_ACT:", src[i : i + 900])

    def test_without_the_flag_there_is_no_decider_at_all(self):
        """Which is the state of this rig's turnkey config today: the chain
        above is whole and simply not switched on."""
        src = inspect.getsource(regime_runtime)
        self.assertIn("stage_clock = None", src)


if __name__ == "__main__":
    unittest.main()
