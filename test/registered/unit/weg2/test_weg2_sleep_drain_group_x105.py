"""fnFL2x105: the D->P sleep split group D; the drain and the flush are GROUP facts.

THE SPECIMEN, boot fnFL2x104 @ 5a96de48be (2026-09-23). Group D (TP=3, PP=1)
held a 97,841-token context that P had prefilled (loaded from the store) plus a
3,891-token needle it prefilled itself; the two share a 3,008-token prefix, so
the needle's insert SPLIT the loaded node. Then the decode probe forced the
flip D->P:

* the sleep flush's publish sweep issued 3 store writes on TP1/TP2
  (nodes 33, 34, 32) but 2 on TP0 (34, 32): on the workers the split parent
  (node 33) read as un-backed -- their host rows had been released after the
  load-back and ``_split_node`` did not carry ``l3_present`` to the new parent --
  while on TP0 (arena rows) the same parent was ``backuped``;
* the storage acks drain by a MIN over the group, so TP1/TP2 kept
  ``hicache_backup(1)`` for ever while TP0 went idle;
* ``/flush_cache``: TP0 "Cache flushed successfully!", TP1/TP2 "not-idle
  because: hicache_backup(1) | single-rank verdict (pp_size=1)"; only TP0's 200
  left the group;
* ``release_memory_occupation``: TP0 was idle, skipped the drain, slept and sat
  in the fence; TP1/TP2 polled ``check_hicache_events`` -- an all_reduce over the
  attention group TP0 had left. 120 s later the fence expired
  ("Ranks 1, 2 failed to pass monitoredBarrier in 120000 ms").

Boot fnFL2x105 (40ac644fe0) died the same way with a 4,521-token context: the
condition is "a store-loaded prefix split by a later D request, then a D->P
sleep", not the context size.

Each case below is red on 5a96de48be, except the drainable-group negative
branch, which is green on both sides by design.
"""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import torch

from sglang.srt.managers.io_struct import FlushCacheReqInput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.flush_wrapper import (
    SchedulerFlushWrapper,
)
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_COLLECTIVE_TIMEOUT_S = 2.0


class _Group:
    """A 3-rank all_reduce over threads. A rank that never arrives leaves the
    others in ``Barrier.wait`` until the timeout -- the gloo shape of x104."""

    def __init__(self, world):
        self.world = world
        self._slots = [None] * world
        self._out = None
        self._barrier = threading.Barrier(world, timeout=_COLLECTIVE_TIMEOUT_S)
        self._barrier2 = threading.Barrier(world, timeout=_COLLECTIVE_TIMEOUT_S)

    def all_reduce(self, rank, values, op):
        self._slots[rank] = list(values)
        if self._barrier.wait() == 0:
            self._out = [op(col) for col in zip(*self._slots)]
        self._barrier2.wait()
        return list(self._out)


class _RankTree:
    """One rank's HiCache: ``ongoing_backup`` plus the acks that have landed.

    ``check_hicache_events`` drains acks by a MIN over the group -- the rule of
    ``UnifiedRadixCache.drain_storage_control_queues`` -- so a rank with more
    storage ops than its peers keeps the surplus in flight."""

    def __init__(self, group, rank, backups):
        self.group = group
        self.rank = rank
        self.ongoing_backup = {op: object() for op in range(backups)}
        self.acks_landed = backups
        self.polls = 0

    def check_hicache_events(self):
        self.polls += 1
        (n,) = self.group.all_reduce(self.rank, [self.acks_landed], min)
        for op in sorted(self.ongoing_backup)[:n]:
            del self.ongoing_backup[op]
        self.acks_landed -= n

    def hicache_group_max(self, values, *, label):
        return self.group.all_reduce(self.rank, values, max)


def _rank_updater(tree):
    sch = SimpleNamespace(tree_cache=tree, enable_hierarchical_cache=True)
    sch.idle_blockers = lambda: (
        [f"hicache_backup({len(tree.ongoing_backup)})"] if tree.ongoing_backup else []
    )
    sch.is_fully_idle = lambda: not tree.ongoing_backup
    wu = SchedulerWeightUpdaterManager.__new__(SchedulerWeightUpdaterManager)
    wu.scheduler = sch
    return wu


def _run_sleep_drain(backups_per_rank, bound_s):
    """Run every rank's pre-sleep drain on its own thread, as group D does."""
    group = _Group(len(backups_per_rank))
    trees = [_RankTree(group, r, b) for r, b in enumerate(backups_per_rank)]
    outcome = [None] * len(trees)

    def rank_main(r):
        try:
            _rank_updater(trees[r])._weg2_drain_hicache_before_sleep(bound_s=bound_s)
            outcome[r] = "returned"
        except Exception as exc:  # noqa: BLE001 -- the outcome IS the finding
            outcome[r] = type(exc).__name__

    threads = [threading.Thread(target=rank_main, args=(r,)) for r in range(len(trees))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    return outcome, [t.polls for t in trees]


class TheSleepDrainIsAGroupLoop(CustomTestCase):
    def test_x104_idle_tp0_and_stuck_workers_refuse_together(self):
        """RED ON 5a96de48be: TP0 returns (goes to sleep) without a single
        collective, TP1/TP2 wait for it in the ack drain's all_reduce until the
        transport gives up -- the 120 s monitoredBarrier death, in miniature.

        Fixed: every rank reads the same group verdict each pass, polls equally
        often, and all three refuse by name (W120) when the bound expires."""
        outcome, polls = _run_sleep_drain([0, 1, 1], bound_s=0.2)
        self.assertEqual(
            outcome,
            ["Weg2SleepDrainRefused"] * 3,
            f"ranks disagree on the sleep: {outcome} (polls {polls})",
        )
        self.assertEqual(len(set(polls)), 1, f"unequal collective counts {polls}")

    def test_a_drainable_group_sleeps_on_every_rank(self):
        """The negative branch: equal op counts drain in one group poll and no
        rank refuses -- the fix must not turn a drainable flush into a stop."""
        outcome, polls = _run_sleep_drain([2, 2, 2], bound_s=5.0)
        self.assertEqual(outcome, ["returned"] * 3)
        self.assertEqual(polls, [1, 1, 1])

    def test_the_release_leg_calls_the_drain_unconditionally(self):
        """The rank-local gate `if not self.is_fully_idle():` in front of the
        drain is what let TP0 skip the group's collectives."""
        import inspect

        src = inspect.getsource(
            SchedulerWeightUpdaterManager.release_memory_occupation
        )
        i = src.index("self._weg2_drain_hicache_before_sleep()")
        line_start = src.rindex("\n", 0, i) + 1
        prev_line = src[src.rindex("\n", 0, line_start - 1) + 1 : line_start]
        self.assertNotIn("is_fully_idle", prev_line)


class TheFlushVerdictBindsTheTpAxis(CustomTestCase):
    def _sched(self, *, idle, group_blocked, pp_size=1):
        calls = []

        def group_max(values, *, label):
            calls.append(list(values))
            return [max(values[0], group_blocked)]

        s = SimpleNamespace(
            is_fully_idle=lambda: idle,
            idle_blockers=lambda: [] if idle else ["hicache_backup(1)"],
            enable_hierarchical_cache=True,
            tree_cache=SimpleNamespace(hicache_group_max=group_max),
            ps=SimpleNamespace(pp_rank=0, pp_size=pp_size),
        )
        s.group_idle_verdict = Scheduler.group_idle_verdict.__get__(s)
        return s, calls

    def test_x104_tp0_idle_while_workers_block_is_not_idle(self):
        """RED ON 5a96de48be: TP0 answered the /flush_cache 200 on its own."""
        s, calls = self._sched(idle=True, group_blocked=1)
        ok, detail = s.group_idle_verdict(tp_group_verdict=True)
        self.assertFalse(ok, detail)
        self.assertEqual(calls, [[0]])
        self.assertIn("tp-group verdict", detail)

    def test_pp_groups_keep_the_vote_and_take_no_tp_reduce(self):
        """On a PP group the answer path must stay collective-free (#1268:
        sb2/sb3 deadlocked on a reduce here); the TP reduce is pp_size<=1 only."""
        s, calls = self._sched(idle=True, group_blocked=1, pp_size=3)
        s._weg2_vote_verdict = None
        ok, detail = s.group_idle_verdict(tp_group_verdict=True)
        self.assertFalse(ok)
        self.assertIn("PENDING", detail)
        self.assertEqual(calls, [])

    def test_only_the_immediate_rpc_asks_for_the_group_verdict(self):
        """The immediate RPC reaches every TP rank in one pass; the deferred
        path flushes on a rank-local idle test and must not post a collective."""
        flush = MagicMock(return_value=True)
        wrapper = SchedulerFlushWrapper(
            flush_cache=flush, is_fully_idle=lambda: True, ipc_channels=MagicMock()
        )
        wrapper.handle(FlushCacheReqInput(timeout_s=0.0))
        flush.assert_called_once_with(tp_group_verdict=True)
        flush.reset_mock()
        wrapper.handle(FlushCacheReqInput(timeout_s=5.0))
        flush.assert_called_once_with()


class TheSplitParentOfAStoredSpanIsStored(CustomTestCase):
    def test_split_carries_l3_present_to_the_new_parent(self):
        """RED ON 5a96de48be: node 33 of x104 -- the upper half of a loaded,
        store-present node -- came out of the split as neither backed nor
        stored, and the sleep sweep wrote it again on the workers only."""
        import os
        import sys

        sys.path.insert(
            0,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mem_cache"),
        )
        from unittest.mock import patch

        from test_unified_radix_cache_unittest import CacheConfig, build_fixture

        # Hermetic (CUDA_VISIBLE_DEVICES=): the fixture's device probe finds no
        # accelerator and `is_cpu()` is cached before this test can name the
        # CPU engine, so the probe is answered directly.
        with patch("test_unified_radix_cache_unittest.get_device", return_value="cpu"):
            cache, allocator, _ = build_fixture(
                CacheConfig(page_size=1, components=(ComponentType.FULL,))
            )
        cache.insert(
            InsertParams(
                key=RadixKey(list(range(1, 33)), None),
                value=allocator.alloc(32).to(dtype=torch.int64),
            )
        )
        loaded = next(iter(cache.root_node.children.values()))
        loaded.l3_present = True  # its pages came out of the store
        self.assertFalse(loaded.backuped)  # transit host rows released

        cache.insert(
            InsertParams(
                key=RadixKey(list(range(1, 17)) + list(range(101, 117)), None),
                value=allocator.alloc(32).to(dtype=torch.int64),
            )
        )
        parent = next(iter(cache.root_node.children.values()))
        self.assertIsNot(parent, loaded, "the insert must split the loaded node")
        self.assertEqual(len(parent.key), 16)
        self.assertTrue(
            parent.l3_present,
            "the split parent of a store-present span reads as un-backed: the "
            "sleep sweep re-writes it on the ranks whose host rows were released "
            "and not on the arena rank",
        )
        self.assertTrue(loaded.l3_present)


if __name__ == "__main__":
    unittest.main()
