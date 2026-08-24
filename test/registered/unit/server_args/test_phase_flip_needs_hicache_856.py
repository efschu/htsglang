"""#856: --enable-phase-flip with the hierarchical cache OFF is refused at launch.

THE DEFECT THIS PREVENTS. Since #856 the flip carries NO KV: at the seam it
retracts every resident request and DROPS the prefix tree, so the next phase
starts with an empty device tier and restores prefixes by read-through from
the hierarchical cache. With that cache off there is nothing to read through
-- the #703 flip-time writeback has nowhere to persist, the retracted
prefixes are gone, and every conversation re-prefills from scratch on every
flip.

That is a CORRECTNESS-SHAPED cost, not a tuning one, which is why it is
refused rather than warned about. It is also invisible at runtime: the flip
completes, the requests complete, and only the token bill and the latency say
anything happened. A silent, expensive, correct-looking failure is exactly
what a launch gate is for.

SAME SHAPE AND SAME PLACE AS #806, which refuses --enable-phase-flip x
--disable-radix-cache for the neighbouring reason that the flip cannot
ENUMERATE what it must move. This one says the flip cannot RESTORE what it
deliberately drops.

NO FALLBACK IS OFFERED, and the message says so. Reviving the KV mover for
this case would reintroduce the seam this ticket retired, the staging reserve
behind W25's 33 refused arms, and the resident carry that made #825's tree
reset crash three ranks.
"""

import unittest

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def _args(**kwargs):
    """model_path='dummy' short-circuits __post_init__ (the repo-wide
    convention for argument tests), so the validator is invoked explicitly --
    which also pins it as a validator of its own rather than a block buried in
    an unrelated handler."""
    args = ServerArgs(model_path="dummy", **kwargs)
    args._validate_phase_flip_needs_hierarchical_cache()
    return args


class TheCombinationIsRefusedTest(CustomTestCase):
    def test_flip_without_hierarchical_cache_is_refused(self):
        """THE FALSIFIER. Accepting this boots an instance that flips
        correctly and re-prefills every conversation from scratch, silently."""
        with self.assertRaises(ValueError) as ctx:
            _args(enable_phase_flip=True, enable_hierarchical_cache=False)
        self.assertIn("--enable-hierarchical-cache", str(ctx.exception))

    def test_the_message_names_both_exits(self):
        # Which exit is right depends on what the operator wanted and the
        # process cannot know; naming only one sends half its readers the
        # wrong way. Same rule #806's message follows.
        with self.assertRaises(ValueError) as ctx:
            _args(enable_phase_flip=True, enable_hierarchical_cache=False)
        message = str(ctx.exception)
        self.assertIn("--enable-hierarchical-cache", message)
        self.assertIn("--enable-phase-flip", message)

    def test_the_message_says_why_rather_than_only_what(self):
        # A gate that says "not allowed" and not "because the flip drops the
        # tree and reads it back" teaches nobody, and the next reader relaxes
        # it. This is the #856 contract in one sentence.
        with self.assertRaises(ValueError) as ctx:
            _args(enable_phase_flip=True, enable_hierarchical_cache=False)
        message = str(ctx.exception)
        self.assertIn("carries NO KV", message)
        self.assertIn("read-through", message)

    def test_the_message_forecloses_the_mover_fallback(self):
        # THE STANDING DOCTRINE, written into the refusal itself: every gap is
        # fixed inside the HiCache route, never by conditionally reviving the
        # seam that was retired.
        with self.assertRaises(ValueError) as ctx:
            _args(enable_phase_flip=True, enable_hierarchical_cache=False)
        self.assertIn("no fallback", str(ctx.exception))


class EachOptionAloneStaysLegalTest(CustomTestCase):
    """The both-directions half. The refusal must come from the COMBINATION,
    or it would reject launches that work today."""

    def test_the_flip_with_the_cache_on_is_untouched(self):
        args = _args(enable_phase_flip=True, enable_hierarchical_cache=True)
        self.assertTrue(args.enable_phase_flip)

    def test_the_cache_off_without_the_flip_is_untouched(self):
        args = _args(enable_phase_flip=False, enable_hierarchical_cache=False)
        self.assertFalse(args.enable_phase_flip)

    def test_neither_flag_is_untouched(self):
        self.assertIsNotNone(_args())


class TheValidatorRunsAtLaunchTest(CustomTestCase):
    """A validator nothing calls is the shape #806 existed to remove."""

    def test_post_init_invokes_it(self):
        import inspect

        src = inspect.getsource(ServerArgs.__post_init__)
        self.assertIn("_validate_phase_flip_needs_hierarchical_cache", src)

    def test_it_runs_after_the_declarations_are_materialized(self):
        # It must read the RESOLVED value: the hierarchical cache is switched
        # by handlers above, so a check placed earlier would pass a launch
        # this one refuses. Same ordering requirement #806 documents.
        import inspect

        src = inspect.getsource(ServerArgs.__post_init__)
        self.assertLess(
            src.index("materialize_declarations(self)"),
            src.index("_validate_phase_flip_needs_hierarchical_cache"),
        )


if __name__ == "__main__":
    unittest.main()
