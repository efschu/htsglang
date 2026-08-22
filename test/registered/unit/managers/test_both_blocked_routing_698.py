"""#698: the BOTH-BLOCKED decline must actually invoke the remedy it names.

phase_policy's branch states, in prose, "Declining here is what routes the
caller to the evict rung instead of to a cutover." For 54 minutes on
2026-08-16 the caller did not: every decline took one throttled log line and
returned. The instance printed "this is an evict trigger and NOT a flip" 350
times while no eviction was ever attempted, health answered 200 throughout,
and three GPUs sat idle behind 10.5M queued tokens.

This is the #505 discipline applied to that: an invariant asserted only in a
comment is not an invariant. The test fails if the caller stops routing.
"""

import ast
import inspect
import unittest

from sglang.srt.managers import scheduler as scheduler_mod
from sglang.srt.managers.phase_policy import BOTH_BLOCKED
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5)


class TheDeclineMustRouteToTheRemedy(unittest.TestCase):
    def test_the_relief_hook_exists(self):
        self.assertTrue(
            hasattr(scheduler_mod.Scheduler, "_apply_both_blocked_relief"),
            "the BOTH-BLOCKED receipt promises an evict; the method that "
            "performs it is gone, so the promise is prose again.",
        )

    def test_the_arming_path_calls_it(self):
        src = inspect.getsource(scheduler_mod)
        tree = ast.parse(src)
        cls = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.ClassDef) and n.name == "Scheduler"
        )
        fn = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "maybe_arm_phase_policy"
        )
        # MATCH THE NAME, NOT THE CALL FORM. The routing is invoked through
        # getattr(self, "...", noop) so scheduler STAND-INS in the policy tests
        # do not raise; that is not an attribute call and an AST walk keyed on
        # ast.Attribute misses it. Keying on the identifier anywhere in the
        # function survives either spelling and still fails if the routing is
        # deleted -- which is the only thing this pin is for.
        body = ast.get_source_segment(src, fn) or ""
        self.assertIn(
            "_apply_both_blocked_relief",
            body,
            "maybe_arm_phase_policy no longer routes the decline to the evict "
            "rung. That is exactly the 2026-08-16 16:23 wedge: the diagnosis "
            "loops forever and the named remedy never runs.",
        )

    def test_the_relief_actually_calls_eviction(self):
        body = inspect.getsource(scheduler_mod.Scheduler._apply_both_blocked_relief)
        self.assertIn(
            "evict_from_tree_cache",
            body,
            "the relief must invoke eviction, not merely log that it would.",
        )

    def test_it_reports_a_zero_delivery(self):
        """'Ran and freed 0' and 'never ran' are the two states the outage
        could not tell apart. The receipt must separate them."""
        body = inspect.getsource(scheduler_mod.Scheduler._apply_both_blocked_relief)
        self.assertIn("freed", body)
        self.assertIn("NOTHING", body)

    def test_it_is_rate_limited(self):
        self.assertGreater(
            scheduler_mod.Scheduler.BOTH_BLOCKED_EVICT_INTERVAL_S,
            0.0,
            "the decline is evaluated every round; an unbounded evict there "
            "walks the whole tree in a tight loop on an already-wedged box.",
        )

    def test_the_constant_is_shared_not_respelled(self):
        self.assertEqual("BOTH BLOCKED", BOTH_BLOCKED)
        self.assertIn(
            "BOTH_BLOCKED",
            inspect.getsource(scheduler_mod.Scheduler._apply_both_blocked_relief),
        )


if __name__ == "__main__":
    unittest.main()
