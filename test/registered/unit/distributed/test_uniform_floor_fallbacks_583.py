"""#583 follow-up: the uniform-floor FALLBACKS must not silently go local.

THE DEFECT THIS PINS CLOSED
---------------------------
#603 and the #583 retraction fix both work the same way: reduce the pool
headroom ONCE per scheduler iteration, then decide from the reduced value
instead of the rank-local ``available_size()``. Two places kept a fallback
for the case where the reduced value was not available:

  * ``Scheduler.uniform_min_avail``  -- ``getattr(self, "_uniform_min_avail",
    None)``, returning the local pool when the reduce had not run;
  * ``ScheduleBatch.decode_mem_avail`` -- returning the local pool when
    ``uniform_avail_floor`` was unset.

Both were justified as "correct for a single rank and for tests". They are.
The problem is what they do on the configuration that matters: on a
MULTI-RANK boot they hand back the exact rank-local predicate the two fixes
removed, so any path that reaches a decode-mem decision without setting the
floor silently restores the desync -- with the protection still visibly
present in the source.

That is the getattr-default trap (#606): a guard that reads as armed and is
absent at runtime. It is the same failure shape as a fixture string nobody
ever printed (#380) -- the code claims a property that nothing checks.

The fix keeps the single-rank fallback (there is nothing to diverge from
with one rank) and REFUSES loudly on a group.

Hermetic: no CUDA, no process group, no model.
"""

import types
import unittest
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import schedule_batch as sb  # noqa: E402
from sglang.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

LOCAL_AVAIL = 4242


class _FakeAllocator:
    page_size = 16

    def available_size(self):
        return LOCAL_AVAIL


class _FakeScheduler:
    """Only the surface ``uniform_min_avail`` touches."""

    def __init__(self, tp_size):
        self.token_to_kv_pool_allocator = _FakeAllocator()
        # `self.ps.tp_size`, mirroring the real Scheduler. A flat
        # `self.tp_size` here would be a stub inventing a surface the
        # production class does not have -- the exact mistake that let the
        # census ship dead (see test_census_attribute_surface_583.py).
        self.ps = types.SimpleNamespace(tp_size=tp_size)

    uniform_min_avail = Scheduler.uniform_min_avail


def _batch(tp_size, floor=None):
    b = ScheduleBatch.__new__(ScheduleBatch)
    b.token_to_kv_pool_allocator = _FakeAllocator()
    b.uniform_avail_floor = floor
    return b


class UniformFloorFallbackTest(unittest.TestCase):
    # -- THE FALSIFIER, first half: the pre-fix behaviour, spelled out -----

    def test_prefix_the_silent_local_fallback_is_what_we_are_removing(self):
        """Pinned so the assertions below cannot be mistaken for a no-op.

        The OLD code was exactly this expression on BOTH paths, for any world
        size -- a group got its own pool back and never knew.
        """
        old_scheduler_fallback = _FakeAllocator().available_size()
        self.assertEqual(old_scheduler_fallback, LOCAL_AVAIL)
        # ...and on a 3-rank group that value is one rank's pool, not the
        # group's minimum, which is the whole divergence.
        self.assertNotEqual(LOCAL_AVAIL, min(LOCAL_AVAIL, 100))

    # -- second half: a group now refuses instead of guessing --------------

    def test_scheduler_refuses_a_missing_reduce_on_a_group(self):
        sched = _FakeScheduler(tp_size=3)
        with self.assertRaises(RuntimeError) as ctx:
            sched.uniform_min_avail()
        msg = str(ctx.exception)
        self.assertIn("_update_uniform_pool_budget", msg)
        self.assertIn("multi-rank", msg)

    def test_batch_refuses_a_missing_floor_on_a_group(self):
        with mock.patch.object(sb, "_group_world_size", lambda: 3):
            with self.assertRaises(RuntimeError) as ctx:
                _batch(3).decode_mem_avail()
        self.assertIn("uniform_avail_floor", str(ctx.exception))

    # -- but a single rank must still work, or this is a regression --------

    def test_single_rank_scheduler_still_uses_the_local_value(self):
        self.assertEqual(_FakeScheduler(tp_size=1).uniform_min_avail(), LOCAL_AVAIL)

    def test_single_rank_batch_still_uses_the_local_value(self):
        with mock.patch.object(sb, "_group_world_size", lambda: 1):
            self.assertEqual(_batch(1).decode_mem_avail(), LOCAL_AVAIL)

    # -- and a supplied value is always preferred, group or not ------------

    def test_a_present_reduced_value_is_used_and_never_the_local_one(self):
        sched = _FakeScheduler(tp_size=3)
        sched._uniform_min_avail = 99
        self.assertEqual(sched.uniform_min_avail(), 99)

    def test_a_present_floor_is_used_and_never_the_local_one(self):
        with mock.patch.object(sb, "_group_world_size", lambda: 3):
            self.assertEqual(_batch(3, floor=99).decode_mem_avail(), 99)

    def test_a_zero_floor_is_honoured_and_not_treated_as_missing(self):
        """A full pool reads as 0, which must not fall through to the local
        value -- `if not floor` instead of `if floor is None` would be the
        same silent-local bug wearing a different hat."""
        with mock.patch.object(sb, "_group_world_size", lambda: 3):
            self.assertEqual(_batch(3, floor=0).decode_mem_avail(), 0)
        sched = _FakeScheduler(tp_size=3)
        sched._uniform_min_avail = 0
        self.assertEqual(sched.uniform_min_avail(), 0)

    # -- the world-size probe must never be the thing that fails -----------

    def test_the_world_size_probe_degrades_to_single_rank_never_raises(self):
        """A probe that can itself fail would move the silent-default problem
        one level down, so it answers 1 when no group is up."""
        self.assertEqual(sb._group_world_size(), 1)

    def test_the_probe_reads_the_parallel_state_when_there_is_one(self):
        fake = types.SimpleNamespace(tp_size=4)
        with mock.patch.object(sb, "get_parallel", lambda: fake):
            self.assertEqual(sb._group_world_size(), 4)


if __name__ == "__main__":
    unittest.main()
