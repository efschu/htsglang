"""#426 -- the custom-group rank-config exchange must name its CPU group.

Upstream sgl-project/sglang#32751: ``create_custom_parallel_group`` builds a
``backend="gloo"`` group, but exchanges the per-rank rank-list with
``torch.distributed.all_gather_object(...)`` and no ``group=``. That defaults
to WORLD, and torch then picks the staging device for the size-exchange tensor
in ``_get_object_coll_device(group)``, whose documented last resort when WORLD
has several backends and none of them is CPU is:

    # No cpu in the backend list. Randomly pick the first backend
    return devices[0].type

So the one collective in the function whose entire purpose is CPU-side
coordination stages its metadata on whichever accelerator backend happens to be
listed first. The reporter saw an intermittent ``CUDA error: invalid argument``
about one launch in a few on MetaX/MACA during HiCache prefetch-sync group
creation; on stock NVIDIA it is silent and merely non-deterministic.

This is our device-identity family (#397, #406, #394): never let an implicit
enumeration decide which device a collective touches. The fix names the world
group's gloo ``cpu_group``, which removes the question instead of answering it.

GPU-free and driver-free: torch.distributed is fully injected.
"""

from __future__ import annotations

import unittest
from unittest import mock

import sglang.srt.distributed.parallel_state as parallel_state
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _FakeProcessGroup:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"<pg {self.name}>"


class _FakeWorldGroup:
    """A GroupCoordinator whose device group is NCCL and whose cpu_group is
    gloo -- exactly the shape that makes torch's fallback reachable."""

    def __init__(self):
        self.device_group = _FakeProcessGroup("world-nccl")
        self.cpu_group = _FakeProcessGroup("world-gloo")


class _DistRecorder:
    """Records the ``group=`` every all_gather_object was given."""

    def __init__(self, world_size, rank, configs):
        self.world_size = world_size
        self.rank = rank
        self.configs = configs
        self.object_groups = []
        self.new_groups = []

    def install(self, stack):
        dist = parallel_state.torch.distributed
        stack.enter_context(mock.patch.object(dist, "is_initialized", lambda: True))
        stack.enter_context(
            mock.patch.object(dist, "get_world_size", lambda: self.world_size)
        )
        stack.enter_context(mock.patch.object(dist, "get_rank", lambda: self.rank))
        stack.enter_context(
            mock.patch.object(dist, "all_gather_object", self._all_gather_object)
        )
        stack.enter_context(mock.patch.object(dist, "new_group", self._new_group))

    def _all_gather_object(self, out_list, obj, group=None):
        self.object_groups.append(group)
        for i, config in enumerate(self.configs):
            out_list[i] = list(config)

    def _new_group(self, ranks=None, backend=None):
        self.new_groups.append((tuple(ranks), backend))
        return _FakeProcessGroup(f"custom-{tuple(ranks)}")


def _run_create(world, world_size=4, rank=0, configs=None):
    import contextlib

    configs = configs or [[0, 1], [0, 1], [2, 3], [2, 3]]
    recorder = _DistRecorder(world_size, rank, configs)
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(parallel_state, "_WORLD", world))
        recorder.install(stack)
        result = parallel_state.create_custom_parallel_group(configs[rank])
    return recorder, result


class TestTheExchangeNamesTheCpuGroup(CustomTestCase):
    """The falsifier: unfixed, ``group`` is None and torch is left to guess."""

    def test_all_gather_object_receives_the_world_cpu_group(self):
        world = _FakeWorldGroup()
        recorder, _ = _run_create(world)
        self.assertEqual(len(recorder.object_groups), 1)
        self.assertIs(recorder.object_groups[0], world.cpu_group)

    def test_it_is_not_the_device_group(self):
        """Naming *a* group is not enough -- the accelerator group would keep
        the staging tensor on the device this exchange must stay off."""
        world = _FakeWorldGroup()
        recorder, _ = _run_create(world)
        group = recorder.object_groups[0]
        self.assertIsNotNone(group, "no group named: torch picks the device")
        self.assertIsNot(group, world.device_group)


class TestTheGroupsThemselvesAreUnchanged(CustomTestCase):
    """Control: only the exchange's group argument moved."""

    def test_the_same_custom_groups_are_created_in_the_same_order(self):
        world = _FakeWorldGroup()
        recorder, result = _run_create(world, rank=2)
        self.assertEqual(
            recorder.new_groups,
            [((0, 1), "gloo"), ((2, 3), "gloo")],
        )
        self.assertEqual(result.name, "custom-(2, 3)")

    def test_a_rank_outside_every_config_gets_none(self):
        world = _FakeWorldGroup()
        configs = [[0, 1], [0, 1], [2, 3], [2, 3]]
        recorder, result = _run_create(world, rank=0, configs=configs)
        self.assertEqual(result.name, "custom-(0, 1)")
        # ...and the exchange still ran exactly once.
        self.assertEqual(len(recorder.object_groups), 1)


class TestNoWorldGroupFallsBackToTorchDefault(CustomTestCase):
    """Without parallel state there is no cpu_group to name; passing None is
    then literally the previous behavior, not a new guess."""

    def test_group_is_none_when_the_world_group_is_absent(self):
        recorder, _ = _run_create(None)
        self.assertIsNone(recorder.object_groups[0])


if __name__ == "__main__":
    unittest.main()
