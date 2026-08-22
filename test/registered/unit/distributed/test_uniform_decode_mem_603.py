"""#603 (B): the decode-OOM decision is rank-uniform on BOTH paths.

The defect this pins closed: ``Scheduler.get_next_batch_to_run`` gates a
collective-bearing ``run_batch`` on ``if batch:``, and the ``batch`` that
comes out of it depended -- when ``--enable-kv-session-offload`` was off --
on ``ScheduleBatch.check_decode_mem``, which compares the LOCAL
``available_size()`` against the replicated token demand.

Under uneven DCP/TP the ranks' pools differ by construction, so near the pool
boundary the binding rank flips to retract while the others still fit. The
two groups then take different branches of the event loop, and the branches
carry different collectives. On 2026-08-05 19:41 that produced the observed
crash: ranks 0/1 spun in a BAR1 ``all_to_all`` inside a CUDA-graph replay for
~30 s waiting for rank 2, which had gone round to ``recv_requests``; the spin
kernels then took their abort path.

The mechanism that fixes it already existed (``update_dcp_admission_state``'s
MIN-reduce) but was gated behind the offload manager -- a condition
orthogonal to the defect. These tests pin the reduced value being used on the
unguarded path too.

Deliberately hermetic: no CUDA, no real process group. The reduce is driven
through a fake that behaves like ``torch.distributed`` for a MIN all_reduce
over a fixed set of per-rank values.
"""

import types
import unittest
from unittest import mock

import torch

from sglang.srt.managers.scheduler import Scheduler

MiB = 1024 * 1024


class _FakeAllocator:
    def __init__(self, avail):
        self._avail = avail

    def available_size(self):
        return self._avail


class _FakeScheduler:
    """Only the surface ``_update_uniform_pool_budget`` / ``uniform_min_avail``
    actually touch. Built by hand rather than by constructing a Scheduler:
    the real constructor wants a model, a device and a process group, none of
    which this decision depends on."""

    def __init__(self, avail, world_avails=None, tp_size=None):
        self.token_to_kv_pool_allocator = _FakeAllocator(avail)
        self.kv_session_offload = None
        self.tp_cpu_group = object() if world_avails else None
        self._world_avails = world_avails
        # The real Scheduler carries its parallel identity on `self.ps`
        # (ParallelState), NOT as `self.tp_size` -- a stub that invents the
        # flat attribute is what let the census ship dead against
        # `self.tp_rank`. Model the real shape.
        self.ps = types.SimpleNamespace(
            tp_size=tp_size if tp_size is not None else len(world_avails or [1])
        )
        # _update_uniform_pool_budget now reads self.server_args.dcp_size
        # (scheduler.py:3503) to decide whether the uneven DCP lane is active.
        # dcp_size=1 means even TP (no uneven DCP), the default path.
        self.server_args = types.SimpleNamespace(dcp_size=1)
        # #791b: the reduce now also carries the prefetch ballot, so the
        # stub models the two inputs the ballot reads -- an empty waiting
        # queue and the storage flag off, under which the SHIPPED drain
        # (bound below) returns {} without touching a tree cache. The
        # ballot then contributes only neutral slots and the #603
        # quantities under test stay untouched.
        self.waiting_queue = []
        self.enable_hicache_storage = False

    _update_uniform_pool_budget = Scheduler._update_uniform_pool_budget
    uniform_min_avail = Scheduler.uniform_min_avail
    # #616g: the reduce now also publishes the rank-uniform eviction floor, so
    # the stub must carry that method too. A stub that models less surface than
    # the real class is exactly what this file's own `self.tp_rank` story warns
    # about; it has no `tree_cache`, so the publisher is a no-op here and the
    # #603 quantities under test are untouched.
    _publish_uniform_evict_floor = Scheduler._publish_uniform_evict_floor
    # #639: same reason, one release later -- the reduce also carries the
    # HOST-tier pair now. No `tree_cache` here either, so both publishers are
    # no-ops and the #603 quantities are untouched.
    _HOST_AVAIL_ABSENT = Scheduler._HOST_AVAIL_ABSENT
    _local_host_avail = Scheduler._local_host_avail
    _publish_uniform_host_floor = Scheduler._publish_uniform_host_floor
    # #639b: and the MAMBA pair, same reason one release later again. No
    # `req_to_token_pool` here, so `_local_mamba_avail` is the ABSENT
    # sentinel and the #603 quantities stay untouched.
    _MAMBA_AVAIL_ABSENT = Scheduler._MAMBA_AVAIL_ABSENT
    _local_mamba_avail = Scheduler._local_mamba_avail
    _publish_uniform_mamba_floor = Scheduler._publish_uniform_mamba_floor
    # #791b: and the PREFETCH BALLOT, same reason one release later again.
    # The shipped drain no-ops with the storage flag off (set in __init__).
    _drain_prefetch_progress = Scheduler._drain_prefetch_progress
    # #794 [#772]: and the CORRIDOR WIDTH CEILING, one release later again.
    # `_update_uniform_pool_budget` calls `self._local_corridor_width_ceiling()`
    # unconditionally while assembling the vote vector (scheduler.py:4896), so
    # a stub without it raises AttributeError inside the reduce -- this file's
    # baseline failure. Bind the SHIPPED function rather than stub a number:
    # it is written to be non-binding where it cannot price (every failure
    # path, including the AttributeError this stub's missing surface would
    # otherwise cause inside `get_prefill_admission_gate`, returns the
    # configured width), so this stub -- which has no admission gate -- always
    # contributes a non-binding vote and the #603 quantities under test stay
    # untouched. `chunked_prefill_size` only has to be a positive int; this
    # file's `_FakeDist.all_reduce` ignores payload content and sets `t[0]`
    # directly from `world_avails`, so the exact value is inert here and
    # `_CORRIDOR_WIDTH` is not shared with a payload mirror the way the
    # 616g/639/639b files need.
    chunked_prefill_size = 4096
    _local_corridor_width_ceiling = Scheduler._local_corridor_width_ceiling
    uniform_corridor_width = Scheduler.uniform_corridor_width


class _FakeDist:
    """A MIN all_reduce over a fixed group of per-rank values."""

    #: The real ``ReduceOp``, so the production call site keeps naming MIN
    #: through the same symbol it always did.
    ReduceOp = torch.distributed.ReduceOp

    def __init__(self, world_avails):
        self.world_avails = world_avails
        self.calls = 0

    def get_world_size(self, group):
        return len(self.world_avails)

    def all_reduce(self, t, op=None, group=None):
        self.calls += 1
        t[0] = min(self.world_avails)


class UniformPoolBudgetTest(unittest.TestCase):
    # -- the reduce itself --------------------------------------------------

    def test_every_rank_reads_the_same_available_size(self):
        """The three ranks of the crashing boot: uneven pools, one binding."""
        world = [12000, 9000, 4000]
        seen = []
        for own in world:
            sched = _FakeScheduler(own, world_avails=world)
            fake = _FakeDist(world)
            with mock.patch.object(torch, "distributed", fake):
                sched._update_uniform_pool_budget()
            seen.append(sched.uniform_min_avail())
        self.assertEqual(seen, [4000, 4000, 4000])
        # It is the MINIMUM that everyone gets -- the binding rank's number,
        # not an average and not each rank's own.
        self.assertNotEqual(seen, world)

    def test_the_retract_decision_agrees_across_ranks(self):
        """THE FALSIFIER. A token demand that sits BETWEEN the ranks' local
        headrooms is exactly the divergent case: rank 2 cannot fund it, ranks
        0 and 1 can.

        Before the fix each rank answered from its own pool -- the list below
        would read [False, False, True], the group would split, and the ranks
        that ran would wait for the rank that did not. After it, one answer.
        """
        world = [12000, 9000, 4000]
        demand = 6000  # > 4000, < 9000: the binding rank alone is short

        # The pre-fix predicate, spelled out, to show it really does diverge.
        local_answers = [avail < demand for avail in world]
        self.assertEqual(
            local_answers,
            [False, False, True],
            msg="the fixture must actually be a divergent case, "
            "otherwise the test below proves nothing",
        )

        uniform_answers = []
        for own in world:
            sched = _FakeScheduler(own, world_avails=world)
            with mock.patch.object(torch, "distributed", _FakeDist(world)):
                sched._update_uniform_pool_budget()
            uniform_answers.append(sched.uniform_min_avail() < demand)
        self.assertEqual(uniform_answers, [True, True, True])
        self.assertEqual(len(set(uniform_answers)), 1)

    def test_a_comfortable_demand_is_uniformly_funded(self):
        """The gate must be able to answer NO as well -- otherwise the test
        above would pass against a predicate that always retracts."""
        world = [12000, 9000, 4000]
        answers = []
        for own in world:
            sched = _FakeScheduler(own, world_avails=world)
            with mock.patch.object(torch, "distributed", _FakeDist(world)):
                sched._update_uniform_pool_budget()
            answers.append(sched.uniform_min_avail() < 100)
        self.assertEqual(answers, [False, False, False])

    # -- cost, and the paths that must NOT pay it ---------------------------

    def test_exactly_one_collective_per_iteration(self):
        world = [12000, 9000, 4000]
        sched = _FakeScheduler(12000, world_avails=world)
        fake = _FakeDist(world)
        with mock.patch.object(torch, "distributed", fake):
            sched._update_uniform_pool_budget()
        self.assertEqual(fake.calls, 1)

    def test_single_rank_takes_no_collective_at_all(self):
        """Byte-identical to the previous path off the uneven config."""
        sched = _FakeScheduler(12000, world_avails=None)
        fake = _FakeDist([12000])
        with mock.patch.object(torch, "distributed", fake):
            sched._update_uniform_pool_budget()
        self.assertEqual(fake.calls, 0)
        self.assertEqual(sched.uniform_min_avail(), 12000)

    def test_the_offload_manager_is_not_reduced_twice(self):
        """When session offload is on it has already reduced this iteration.
        A second reduce here would be a second chance for the collective
        counts to diverge -- the very failure being closed."""
        sched = _FakeScheduler(12000, world_avails=[12000, 9000, 4000])
        sched.kv_session_offload = mock.Mock(dcp_min_avail=lambda: 4000)
        fake = _FakeDist([12000, 9000, 4000])
        with mock.patch.object(torch, "distributed", fake):
            sched._update_uniform_pool_budget()
        self.assertEqual(fake.calls, 0)
        self.assertEqual(sched.uniform_min_avail(), 4000)

    # -- the fallback must not silently mean something else -----------------

    def test_before_any_reduce_the_local_value_is_used_and_not_a_stale_zero(self):
        sched = _FakeScheduler(777, world_avails=None)
        self.assertEqual(sched.uniform_min_avail(), 777)


if __name__ == "__main__":
    unittest.main()
