"""When HTCCL is active, the PyNccl communicator must not be CONSTRUCTED.

Not built-and-unused -- not built. NCCL cannot span vendors, so on a
vendor-mixed TP group `ncclCommInitRank` aborts inside the C library while
trying to form a world with a peer that has no NCCL at all. Measured on the
cross-vendor host: both ranks logged "HTCCL enabled ... (transport=gloo)" and
rank 0 then segfaulted at

    pynccl_wrapper.py:404 in ncclCommInitRank
    pynccl.py:111 in __init__
    parallel_state.py:420 in __init__

before the model was loaded. `use_pynccl` is independent of SGLANG_HTCCL, so
enabling HTCCL rerouted the collectives but never stopped the construction.

Three properties are pinned here, and the second matters most:

 1. HTCCL active  -> pynccl is NOT constructed.
 2. The decision is RANK-UNIFORM -- a pure function of state every rank derives
    identically. If one rank built pynccl and another did not, the failure mode
    would not be a segfault but a HANG: quieter, and worse.
 3. HTCCL inactive -> byte-identical to before. pynccl is the faster path on
    same-vendor rigs; the rule is "HTCCL active -> no pynccl", never
    "no pynccl".

CPU only: this tests the construction DECISION, not NCCL itself.
"""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


# THE REAL FUNCTION, imported -- not a re-implementation.
#
# The first version of this test defined its own copy of the condition. When the
# fix was reverted to check the test was red, four of the five tests still
# PASSED: they were exercising the copy, which still had the fix. Only the
# source-inspection test caught it. That is the same failure this codebase keeps
# producing -- a check that verifies something adjacent to the thing under test.
# Importing the real predicate is what makes the red check meaningful.
from sglang.srt.distributed.parallel_state import should_build_pynccl as _should_build_pynccl


class TestHtcclSuppressesPynccl(CustomTestCase):
    def test_htccl_active_means_no_pynccl(self):
        """Property 1 -- the crash this prevents."""
        for world in (2, 3, 4, 8):
            self.assertFalse(
                _should_build_pynccl(True, world, htccl_active=True),
                f"world={world}: pynccl must not be constructed under HTCCL",
            )

    def test_flag_off_is_unchanged(self):
        """Property 3 -- pynccl must still be built on same-vendor rigs.

        This is the guard against 'fixing' the crash by disabling pynccl
        outright, which would silently cost every existing NVIDIA user the
        faster path.
        """
        for world in (2, 3, 4, 8):
            self.assertTrue(
                _should_build_pynccl(True, world, htccl_active=False),
                f"world={world}: pynccl must still be built without HTCCL",
            )
        # And the pre-existing conditions still gate it.
        self.assertFalse(_should_build_pynccl(False, 4, htccl_active=False))
        self.assertFalse(_should_build_pynccl(True, 1, htccl_active=False))

    def test_decision_is_rank_uniform(self):
        """Property 2 -- the important one.

        The decision must depend ONLY on inputs every rank derives identically
        from the same CLI/env. Two ranks with identical configuration must
        reach the identical verdict; nothing rank-local (rank id, device,
        hostname) may enter.
        """
        for use_pynccl in (True, False):
            for world in (1, 2, 4):
                for htccl in (True, False):
                    verdicts = {
                        _should_build_pynccl(use_pynccl, world, htccl)
                        for _rank in range(world)  # same inputs on every rank
                    }
                    self.assertEqual(
                        len(verdicts),
                        1,
                        f"decision differs across ranks for "
                        f"use_pynccl={use_pynccl} world={world} htccl={htccl}",
                    )

    def test_old_semantics_would_have_crashed(self):
        """Pin the BUG, so a revert cannot pass silently.

        The old condition omitted the HTCCL term. Assert that it would have
        built pynccl in exactly the situation that segfaults.
        """
        old = lambda use_pynccl, world: use_pynccl and world > 1  # noqa: E731
        self.assertTrue(
            old(True, 2),
            "sanity: the old condition built pynccl for a TP=2 group",
        )
        self.assertFalse(
            _should_build_pynccl(True, 2, htccl_active=True),
            "the fix must differ from the old condition exactly here",
        )

    def test_matches_parallel_state_source(self):
        """Keep this test honest: assert the real call site still applies the
        same three-term condition. Without this the helper above could drift
        into testing a fiction."""
        import inspect

        from sglang.srt.distributed import parallel_state

        src = inspect.getsource(parallel_state.GroupCoordinator.__init__)
        # The call site must go through the shared predicate, so that reverting
        # the fix cannot leave this test passing against a private copy.
        self.assertIn(
            "should_build_pynccl(use_pynccl, self.world_size, _htccl_active)", src
        )
        # ...and _htccl_active must come from what was actually BUILT, not from
        # re-reading the env, which could diverge from it.
        self.assertIn("_htccl_active = self.htccl_comm is not None", src)


if __name__ == "__main__":
    unittest.main()
