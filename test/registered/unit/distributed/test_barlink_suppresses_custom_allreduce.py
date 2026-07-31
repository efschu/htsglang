"""When barlink is active, CustomAllreduce must not be CONSTRUCTED.

Sibling of test_barlink_suppresses_pynccl.py, and the failure mode is the worse
one: a HANG rather than a crash, because the constructor contains a COLLECTIVE
that only some ranks reach.

`CustomAllreduce.__init__` returns early -- before any collective -- when
`ops.IS_CUSTOM_AR_AVAILABLE` is false, which is the case wherever sgl_kernel's
custom-AR ops are absent (an sm75 build, a ROCm build). Where the ops ARE
present it calls `can_use_custom_all_reduce_with_nvlink`, whose very first act
is `in_the_same_node_as(group, source_rank=0)` -> `broadcast_object_list` over
the whole group. Mixed availability across a group therefore means: some ranks
inside a group-wide broadcast, the rest already past it.

Measured, Nordstern L0 (TP=5 across two hosts, 5090+2x3080 | 2080 Ti + Vega 64),
per-rank py-spy taken at the stall:

    rank 0,1,2  broadcast_object_list <- in_the_same_node_as
                <- can_use_custom_all_reduce_with_nvlink
                <- CustomAllreduce.__init__ <- GroupCoordinator.__init__
    rank 3,4    all_reduce <- get_available_gpu_memory <- init_torch_distributed

Both sides idle, no error on any rank, indefinite.

Note what does NOT save this: the call site wraps the constructor in
try/except with a warning. An exception is rank-local, so that handles a rank
that FAILS -- never a rank that never arrives.

CPU only: this tests the construction DECISION, not custom all-reduce itself.
"""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

# THE REAL FUNCTION, imported -- not a re-implementation. See the sibling test
# for why: a private copy kept four of five tests green across a revert.
from sglang.srt.distributed.parallel_state import (
    should_build_custom_allreduce as _should_build_ca,
)


class TestBarlinkSuppressesCustomAllreduce(CustomTestCase):
    def test_barlink_active_means_no_custom_allreduce(self):
        """Property 1 -- the hang this prevents."""
        for world in (2, 3, 4, 5, 8):
            self.assertFalse(
                _should_build_ca(True, world, barlink_active=True),
                f"world={world}: CustomAllreduce must not be constructed under barlink",
            )

    def test_flag_off_is_unchanged(self):
        """Property 3 -- same-vendor rigs must keep their fast path."""
        for world in (2, 3, 4, 5, 8):
            self.assertTrue(
                _should_build_ca(True, world, barlink_active=False),
                f"world={world}: CustomAllreduce must still be built without barlink",
            )
        self.assertFalse(_should_build_ca(False, 4, barlink_active=False))
        self.assertFalse(_should_build_ca(True, 1, barlink_active=False))

    def test_decision_is_rank_uniform(self):
        """Property 2 -- the important one.

        Nothing rank-local may enter. The thing that actually diverged in the
        field, `ops.IS_CUSTOM_AR_AVAILABLE`, is a property of the local build:
        it is exactly what must NOT be allowed to decide a collective.
        """
        for use_ca in (True, False):
            for world in (1, 2, 4, 5):
                for barlink in (True, False):
                    verdicts = {
                        _should_build_ca(use_ca, world, barlink)
                        for _rank in range(world)  # same inputs on every rank
                    }
                    self.assertEqual(
                        len(verdicts),
                        1,
                        f"decision differs across ranks for "
                        f"use_ca={use_ca} world={world} barlink={barlink}",
                    )

    def test_old_semantics_would_have_hung(self):
        """Pin the BUG, so a revert cannot pass silently."""
        old = lambda use_ca, world: use_ca and world > 1  # noqa: E731
        self.assertTrue(old(True, 5), "sanity: the old condition built it for TP=5")
        self.assertFalse(
            _should_build_ca(True, 5, barlink_active=True),
            "the fix must differ from the old condition exactly here",
        )

    def test_matches_parallel_state_source(self):
        """The call site must go through the shared predicate."""
        import inspect

        from sglang.srt.distributed import parallel_state

        src = inspect.getsource(parallel_state.GroupCoordinator.__init__)
        self.assertIn("should_build_custom_allreduce(", src)
        self.assertIn("use_custom_allreduce, self.world_size, _barlink_active", src)

    def test_constructor_really_does_collective_before_the_early_return(self):
        """Keep the RATIONALE honest, not just the decision.

        If upstream ever moved the nvlink probe behind the availability check,
        this fix would still be correct but its stated reason would be fiction.
        Assert the two facts the reason rests on:
          1. the constructor returns early on IS_CUSTOM_AR_AVAILABLE,
          2. and the nvlink probe -- which broadcasts -- comes after it.
        """
        import inspect

        from sglang.srt.distributed.device_communicators import custom_all_reduce
        from sglang.srt.distributed.device_communicators import custom_all_reduce_utils

        src = inspect.getsource(custom_all_reduce.CustomAllreduce.__init__)
        i_avail = src.find("IS_CUSTOM_AR_AVAILABLE")
        i_probe = src.find("can_use_custom_all_reduce_with_nvlink")
        self.assertNotEqual(i_avail, -1, "early return on availability is gone")
        self.assertNotEqual(i_probe, -1, "nvlink probe is gone")
        self.assertLess(
            i_avail,
            i_probe,
            "the availability early-return no longer precedes the nvlink probe -- "
            "re-check why this fix exists",
        )
        probe_src = inspect.getsource(
            custom_all_reduce_utils.can_use_custom_all_reduce_with_nvlink
        )
        self.assertIn(
            "in_the_same_node_as",
            probe_src,
            "the probe no longer performs the group-wide broadcast",
        )


if __name__ == "__main__":
    unittest.main()
