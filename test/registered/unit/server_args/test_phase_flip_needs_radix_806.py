"""#806: --enable-phase-flip x --disable-radix-cache is refused at launch.

THE DEFECT, as it was run. Without a radix cache the scheduler builds a
``ChunkCache``, and no ChunkCache variant implements ``all_values_flatten``.
The flip's guard tests for exactly that method
(``phase_flip_runtime.py:1292``) and appends "tree cache ChunkCache (no
all_values_flatten enumeration)", so every flip attempt is refused for the
life of the process. Arm-1-v1 ran to completion that way: 15 runtime
refusals, a flip program that never flipped, and nothing at startup to say it
never could.

The runtime guard is correct and stays -- it is what keeps the flip from
moving KV it cannot enumerate. What it cannot do is say so in time.

Both directions of the gate, plus the mechanism itself: the last test pins
WHY the refusal exists, so that if a ChunkCache ever grows the enumeration the
refusal can be lifted with evidence rather than by guess.

    python -m pytest test/registered/unit/server_args/test_phase_flip_needs_radix_806.py -v
"""

import inspect
import unittest

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def _args(**kwargs):
    """ServerArgs with model_path='dummy' short-circuits __post_init__ (the
    repo-wide convention for argument tests), so the #806 validator is invoked
    explicitly -- which also pins it as a validator of its own rather than a
    block buried in an unrelated handler."""
    args = ServerArgs(model_path="dummy", **kwargs)
    args._validate_phase_flip_needs_a_tree_cache()
    return args


class TheCombinationIsRefusedTest(CustomTestCase):
    def test_flip_with_radix_disabled_is_refused(self):
        """THE FALSIFIER. Accepting this is the dead arm: it boots, it runs,
        and it refuses every flip in silence."""
        with self.assertRaises(ValueError) as ctx:
            _args(enable_phase_flip=True, disable_radix_cache=True)
        message = str(ctx.exception)
        self.assertIn("all_values_flatten", message)

    def test_the_message_names_both_exits(self):
        """Which exit is right depends on what the operator wanted, and the
        process cannot know. A refusal that named only one would send half its
        readers the wrong way."""
        with self.assertRaises(ValueError) as ctx:
            _args(enable_phase_flip=True, disable_radix_cache=True)
        message = str(ctx.exception)
        self.assertIn("--disable-radix-cache", message)
        self.assertIn("--enable-phase-flip", message)
        self.assertIn("leave the radix cache ON", message)

    def test_the_message_names_the_implicit_switches(self):
        """--disable-radix-cache is also set FOR the operator by four handlers.
        A reader who never typed the flag needs to be told why they are seeing
        this."""
        with self.assertRaises(ValueError) as ctx:
            _args(enable_phase_flip=True, disable_radix_cache=True)
        self.assertIn("--enable-mis", str(ctx.exception))


class EachOptionAloneStaysLegalTest(CustomTestCase):
    """The both-directions half. The refusal must come from the COMBINATION,
    not from either flag, or it would refuse launches that work today."""

    def test_the_flip_alone_is_untouched(self):
        args = _args(enable_phase_flip=True)
        self.assertTrue(args.enable_phase_flip)
        self.assertFalse(args.disable_radix_cache)

    def test_disabling_radix_alone_is_untouched(self):
        args = _args(disable_radix_cache=True)
        self.assertTrue(args.disable_radix_cache)
        self.assertFalse(args.enable_phase_flip)

    def test_neither_is_untouched(self):
        args = _args()
        self.assertFalse(args.enable_phase_flip)
        self.assertFalse(args.disable_radix_cache)


class TheValidatorRunsLastTest(CustomTestCase):
    """ORDERING, which is the subtle half of this fix.

    ``disable_radix_cache`` is not only a flag the operator types: it is set
    for them by ``--enable-mis``, by an HRM-text model, by Whisper, and by the
    dual_chunk_flash_attn backend -- all in handlers that run AFTER
    ``_handle_phase_flip``. A check placed with the flip handler would read
    False and pass a launch that ends up with no tree cache anyway, which is
    the same dead arm through another door.

    Pinned textually because the property IS the position in the dispatcher;
    there is no cheaper behavioural instrument for it, since the dummy-model
    boundary keeps the argument tests from running the tail of __post_init__.
    """

    def test_the_validator_is_called_after_the_declarations_are_materialized(self):
        source = inspect.getsource(ServerArgs.__post_init__)
        self.assertIn("_validate_phase_flip_needs_a_tree_cache()", source)
        self.assertLess(
            source.index("materialize_declarations(self)"),
            source.index("_validate_phase_flip_needs_a_tree_cache()"),
            "the #806 validator moved above the point where declarations are "
            "applied: it would read a pre-resolution disable_radix_cache and "
            "miss every implicit switch",
        )


class TheMechanismBehindTheRefusalTest(CustomTestCase):
    """WHY the refusal exists, pinned so it can be lifted with evidence.

    If a ChunkCache ever grows ``all_values_flatten``, this test fails and
    tells its reader that the refusal above is now removable -- rather than
    leaving a refusal in place whose reason nobody can re-derive."""

    def test_no_chunk_cache_can_enumerate_its_values(self):
        from sglang.srt.mem_cache.chunk_cache import (
            ChunkCache,
            PureSWAChunkCache,
            SWAChunkCache,
        )

        for cls in (ChunkCache, SWAChunkCache, PureSWAChunkCache):
            self.assertFalse(
                hasattr(cls, "all_values_flatten"),
                f"{cls.__name__} now enumerates its values; the #806 refusal "
                "may be liftable -- check the flip guard in "
                "phase_flip_runtime.py first",
            )

    def test_the_radix_caches_do_enumerate(self):
        """The other direction: the refusal must not be a claim about every
        cache, or it would be refusing something that is not the problem."""
        from sglang.srt.mem_cache.radix_cache import RadixCache

        self.assertTrue(hasattr(RadixCache, "all_values_flatten"))

    def test_the_flip_guard_still_tests_for_that_method(self):
        """The refusal is a shortcut for a runtime guard. If the guard stops
        keying on this method the shortcut is no longer equivalent."""
        from sglang.srt.managers import phase_flip_runtime

        source = inspect.getsource(phase_flip_runtime)
        self.assertIn('hasattr(scheduler.tree_cache, "all_values_flatten")', source)


if __name__ == "__main__":
    unittest.main()
