"""W35: a host release must never outlive the binding it was enqueued on.

THE SPECIMEN (SPECIMEN_w35_rebind_armed_then_crash.log, 2026-08-25, all three
ranks, seconds after the first rebind this tree has ever armed):

    AssertionError: Double-free detected: slots not currently allocated:
      [0, 1, 2, ...]
    get_next_batch_to_run -> get_new_batch_prefill -> _get_new_batch_prefill_raw
      -> check_hicache_events -> drain_storage_control_queues -> _drain_release
      -> memory_pool_host.HostPoolGroup.free -> pool_host/base.py free

`cc.host_mem_release_queue` holds bare index tensors naming slots allocated
from the OUTGOING pool. `rebind` re-points `mem_pool_host`. The next ordinary
scheduler round drains those entries against a pool that never handed the ids
out. The assertion is CORRECT and caught a real corruption loudly.

THE CRITERION, DECIDED BY READING. Dropping stale entries would be right only
if the outgoing pool died with them. It does not: `_stamp` only re-points
readers, nothing tears the outgoing pool down, and `phase_flip_host_pools`
holds BOTH phases for process life so the flip can alternate. The outgoing
pool survives with live allocations and returns on the next flip -- so a
dropped release is a recurring host-slot LEAK. Route, do not drop.

SETTLING RATHER THAN ROUTING-AT-DRAIN keeps ONE authority: routing later would
need a per-entry generation stamp and a generation->pool map beside the #719
generation -- a second bookkeeping scheme, which is the W32 defect. Settling
makes the invariant true by construction: the binding changes in exactly one
place, so at that instant every queued entry belongs to the binding still
installed.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

import queue as _queue
import types
import unittest

import torch

from sglang.srt.mem_cache.hicache_phase_binding import (
    RebindRefused,
    settle_pending_releases,
)
from sglang.test.test_utils import CustomTestCase


class _Pool:
    """Models the one property the real assertion enforces: a pool refuses to
    free ids it did not hand out. `pool_host/base.py` raises exactly this."""

    def __init__(self, name, owned):
        self.name = name
        self.owned = set(owned)
        self.freed = []

    def free(self, indices):
        ids = [int(i) for i in indices.tolist()]
        bad = [i for i in ids if i not in self.owned]
        if bad:
            raise AssertionError(
                f"Double-free detected: slots not currently allocated: {bad}"
            )
        for i in ids:
            self.owned.discard(i)
        self.freed.extend(ids)


def _sched(pending=(), extras=None, owned=range(64)):
    q = _queue.Queue()
    for block in pending:
        q.put(torch.tensor(block, dtype=torch.int64))
    cc = types.SimpleNamespace(
        mem_pool_host=_Pool("pp", owned),
        host_mem_release_queue=q,
        extra_host_mem_release_queues=extras or {},
    )
    return types.SimpleNamespace(tree_cache=types.SimpleNamespace(cache_controller=cc))


class TestTheSpecimenShape(CustomTestCase):
    def test_pending_releases_are_settled_against_the_outgoing_pool(self):
        s = _sched(pending=[[0, 1, 2], [3, 4]])
        self.assertEqual(settle_pending_releases(s), 2)
        cc = s.tree_cache.cache_controller
        self.assertEqual(sorted(cc.mem_pool_host.freed), [0, 1, 2, 3, 4])
        self.assertTrue(cc.host_mem_release_queue.empty())

    def test_after_settling_the_incoming_pool_is_never_asked_to_free_them(self):
        # THE SPECIMEN, end to end: settle on the OUTGOING pool, then swap, then
        # drain. The incoming pool owns a disjoint id space, so if anything had
        # survived the settle it would double-free here exactly as on metal.
        s = _sched(pending=[[0, 1, 2]])
        cc = s.tree_cache.cache_controller
        settle_pending_releases(s)
        cc.mem_pool_host = _Pool("tp", owned=range(1000, 1064))  # the rebind
        leftovers = []
        while not cc.host_mem_release_queue.empty():
            leftovers.append(cc.host_mem_release_queue.get_nowait())
        self.assertEqual(leftovers, [], "nothing may outlive the binding")

    def test_without_settling_the_drain_double_frees(self):
        # RED-FIRST, modelled exactly: this is what W35 did on metal. If this
        # ever stops raising, the double is no longer modelling the real pool
        # and every other assertion here is worthless.
        s = _sched(pending=[[0, 1, 2]])
        cc = s.tree_cache.cache_controller
        cc.mem_pool_host = _Pool("tp", owned=range(1000, 1064))  # rebind first
        blocks = [cc.host_mem_release_queue.get_nowait()]
        with self.assertRaises(AssertionError) as caught:
            cc.mem_pool_host.free(torch.cat(blocks, dim=0))
        self.assertIn("Double-free detected", str(caught.exception))

    def test_an_empty_queue_is_a_no_op(self):
        self.assertEqual(settle_pending_releases(_sched()), 0)


class TestLoudInBothWrongDirections(CustomTestCase):
    """Dropping a needed release is a LEAK; draining it against the wrong pool
    is CORRUPTION. Neither may ever happen quietly."""

    def test_unsettleable_auxiliary_queues_refuse_the_rebind(self):
        aux = _queue.Queue()
        aux.put(torch.tensor([7], dtype=torch.int64))
        s = _sched(extras={"mamba": aux})
        with self.assertRaises(RebindRefused) as caught:
            settle_pending_releases(s)
        msg = str(caught.exception)
        self.assertIn("mamba", msg, "the refusal must NAME the queue")
        self.assertIn("leaks host slots", msg)

    def test_a_failing_settle_refuses_rather_than_proceeding(self):
        s = _sched(pending=[[999]])  # not owned by the outgoing pool
        with self.assertRaises(RebindRefused):
            settle_pending_releases(s)

    def test_nothing_is_discarded_silently(self):
        import inspect

        from sglang.srt.mem_cache import hicache_phase_binding

        src = inspect.getsource(hicache_phase_binding.settle_pending_releases)
        self.assertIn("RebindRefused", src)
        self.assertNotIn("pass  # drop", src)


class TestItRunsBeforeTheSwap(CustomTestCase):
    """Order is the whole fix: after the swap it would free against the pool
    that never allocated the ids, which is the crash."""

    def test_the_settle_precedes_the_rebind(self):
        import inspect

        from sglang.srt.mem_cache import hicache_phase_binding

        src = inspect.getsource(hicache_phase_binding.rebind_for_cutover)
        settle_at = src.find("settle_pending_releases")
        rebind_at = src.find("generation = rebind(")
        self.assertGreater(settle_at, -1)
        self.assertGreater(rebind_at, -1)
        self.assertLess(settle_at, rebind_at)

    def test_it_uses_the_one_generation_authority_not_a_second_scheme(self):
        # Ein-Job-ein-Mover: no parallel per-entry stamp / generation->pool map.
        import inspect

        from sglang.srt.mem_cache import hicache_phase_binding

        src = inspect.getsource(hicache_phase_binding.settle_pending_releases)
        for rival in ("generation_of_entry", "_gen_map", "stamp_release"):
            self.assertNotIn(rival, src)


class TestAgainstTheRealClasses(CustomTestCase):
    """The six-instance lesson: assert against what the live boot actually
    uses, not against a double that agrees by construction."""

    def test_the_real_drain_path_frees_through_mem_pool_host(self):
        import inspect

        from sglang.srt.mem_cache import unified_radix_cache

        src = inspect.getsource(unified_radix_cache)
        self.assertIn("cc.mem_pool_host.free(torch.cat(host_indices_list", src)
        self.assertIn("cc.host_mem_release_queue", src)

    def test_the_real_host_group_refuses_unowned_frees(self):
        # `HostPoolGroup.free` delegates into pool_host/base.py, which is where
        # the metal assertion came from. Pin that the raiser still exists.
        import inspect

        from sglang.srt.mem_cache.pool_host import base

        self.assertIn("Double-free detected", inspect.getsource(base))

    def test_the_rebind_repoints_mem_pool_host(self):
        # Why settling must happen first: this is the attribute the drain reads.
        import inspect

        from sglang.srt.mem_cache import hicache_phase_binding

        self.assertIn(
            '"mem_pool_host"', inspect.getsource(hicache_phase_binding._stamp)
        )


if __name__ == "__main__":
    unittest.main()
