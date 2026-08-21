"""#616g: the CACHE-MUTATION triggers are rank-uniform, so the radix replicas
stay identical and the extend token count stays rank-uniform with them.

The defect these pin closed, measured on the 2026-08-06 21:52:25 wedge:

  #603 and #610 made the decode-mem and prefill-admission DECISIONS uniform,
  but both left eviction as an explicitly local side effect ("Still evict
  locally for the space"). ``evict_from_tree_cache`` fires on
  ``available_size() < num_tokens`` where the DEMAND is replicated
  (``batch.extend_num_tokens``) and the AVAILABILITY is this rank's own pool
  shard -- 179825 / 143860 / 136667 tokens on that boot. The roomy rank
  declines an eviction the tight ranks take, the trees stop being replicas,
  ``match_prefix`` returns a rank-dependent prefix, and
  ``prepare_for_extend`` turns that straight into a rank-dependent
  ``extend_num_tokens``. All three ranks then parked in the SAME all_reduce
  at the SAME ``layer_idx`` with rank 0 reducing 1690 tokens against 1818 on
  its peers -- shapes that can never pair, so the BAR1 spin waits forever
  with a CLEAN abort word.

  hicache load-back is the second source and needs no eviction to fire: it
  EXTENDS the device prefix only on the ranks whose own free space accepts
  it.

Deliberately hermetic: no CUDA, no real process group, no model. The reduce
is driven through a fake that performs a real elementwise MIN across a fixed
set of per-rank payloads.
"""

import types
import unittest
from unittest import mock

import torch

from sglang.srt.managers.prefetch_ballot import build_prefetch_ballot_payload
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.common import (
    evict_from_tree_cache,
    uniform_avail_for_evict,
)

# The three pools of the crashing boot, in tokens (from the KV-cache
# allocation lines of the 21:56 restart, same geometry as 21:52).
BOOT_POOLS = [179825, 143860, 136667]


class _FakeAllocator:
    def __init__(self, avail):
        self._avail = avail

    def available_size(self):
        return self._avail


class _FakeTreeCache:
    """Only the surface ``evict_from_tree_cache`` touches."""

    uniform_avail_floor = None
    uniform_mamba_avail_floor = None

    def __init__(self, avail):
        self.token_to_kv_pool_allocator = _FakeAllocator(avail)
        self.evictions = []

    def is_chunk_cache(self):
        return False

    def evict(self, params):
        self.evictions.append(params.num_tokens)


class _FakeScheduler:
    """Only the surface the publisher and the reduce touch. Built by hand:
    the real constructor wants a model, a device and a process group, none of
    which this decision depends on."""

    def __init__(self, avail, world_avails=None, tree_cache=None):
        self.token_to_kv_pool_allocator = _FakeAllocator(avail)
        self.kv_session_offload = None
        self.tree_cache = tree_cache
        self.tp_cpu_group = object() if world_avails else None
        self.server_args = types.SimpleNamespace(dcp_size=1)
        # #791b: the reduce now also carries the prefetch ballot; model the
        # two inputs it reads. Storage off makes the SHIPPED drain (bound
        # below) return {} without touching a tree cache, so the ballot
        # contributes only neutral slots and the quantities under test stay
        # untouched.
        self.waiting_queue = []
        self.enable_hicache_storage = False
        self.ps = types.SimpleNamespace(tp_size=len(world_avails or [1]))

    _update_uniform_pool_budget = Scheduler._update_uniform_pool_budget
    _publish_uniform_evict_floor = Scheduler._publish_uniform_evict_floor
    uniform_min_avail = Scheduler.uniform_min_avail
    uniform_budget_deficit = Scheduler.uniform_budget_deficit
    # #639: the same reduce now also carries the HOST-tier pair, so the stub
    # has to model that surface too. This fixture's tree cache has no
    # `cache_controller`, so `_local_host_avail` returns the ABSENT sentinel
    # on every rank, no host floor is published, and the #616g quantities
    # under test are untouched -- which is the point of pinning it here.
    _HOST_AVAIL_ABSENT = Scheduler._HOST_AVAIL_ABSENT
    _local_host_avail = Scheduler._local_host_avail
    _publish_uniform_host_floor = Scheduler._publish_uniform_host_floor
    # #639b: and now the MAMBA pair as well, on the same argument. This
    # fixture's scheduler has no `req_to_token_pool`, so `_local_mamba_avail`
    # returns the ABSENT sentinel on every rank, no mamba floor is published,
    # and the #616g quantities under test stay untouched.
    _MAMBA_AVAIL_ABSENT = Scheduler._MAMBA_AVAIL_ABSENT
    _local_mamba_avail = Scheduler._local_mamba_avail
    _publish_uniform_mamba_floor = Scheduler._publish_uniform_mamba_floor
    # #791b: and the PREFETCH BALLOT, same reason one release later again.
    _drain_prefetch_progress = Scheduler._drain_prefetch_progress


class _FakeDist:
    """A real elementwise MIN all_reduce over a fixed group of per-rank
    payloads. Each rank's payload is derived from its own availability by the
    production code, so the fake only has to reproduce the reduction."""

    ReduceOp = torch.distributed.ReduceOp

    def __init__(self, world_avails, pin_admission=False):
        self.world_avails = world_avails
        self.pin_admission = pin_admission
        self.calls = 0
        self.widths = []

    def get_world_size(self, group):
        return len(self.world_avails)

    def _payload(self, avail):
        # #639 appends the host pair and #639b the mamba pair; both ABSENT on
        # every rank in this fixture.
        absent = Scheduler._HOST_AVAIL_ABSENT
        m_absent = Scheduler._MAMBA_AVAIL_ABSENT
        tail = [absent, -absent, m_absent, -m_absent]
        # #791b: the prefetch ballot rides behind the mamba pair; every
        # fixture rank has an empty queue, so its contribution is the
        # neutral one -- digest-0 pair plus all-done slots.
        tail = tail + build_prefetch_ballot_payload([], {})
        if self.pin_admission:
            return [avail, avail, -avail] + tail
        return [avail, -avail] + tail

    def all_reduce(self, t, op=None, group=None):
        self.calls += 1
        self.widths.append(t.numel())
        assert op is self.ReduceOp.MIN, "the pin is a MIN ballot"
        payloads = [self._payload(a) for a in self.world_avails]
        for i in range(t.numel()):
            t[i] = min(p[i] for p in payloads)


def _publish_for(own, world):
    """Run the production reduce for one rank of `world` and hand back its
    tree cache, floor published."""
    tree = _FakeTreeCache(own)
    sched = _FakeScheduler(own, world_avails=world, tree_cache=tree)
    with mock.patch.object(torch, "distributed", _FakeDist(world)):
        sched._update_uniform_pool_budget()
    return sched, tree


class UniformEvictFloorTest(unittest.TestCase):
    # -- the eviction trigger -----------------------------------------------

    def test_the_eviction_trigger_agrees_across_ranks(self):
        """THE FALSIFIER.

        A replicated demand that sits BETWEEN the ranks' local headrooms is
        exactly the divergent case. Before the fix each rank answered from its
        own pool, so the ranks evicted different sets, their radix trees
        stopped being replicas, and the next prefill was composed with
        rank-dependent prefixes.

        Reverting the fix -- i.e. reading ``available_size()`` in
        ``evict_from_tree_cache`` again -- turns the second assertion back
        into [False, True, True] and fails this test.
        """
        world = BOOT_POOLS
        demand = 150_000  # < 179825, > 143860 and > 136667

        # The fixture must really be divergent, or the assertion below proves
        # nothing.
        self.assertEqual(
            [avail < demand for avail in world],
            [False, True, True],
        )

        evicted = []
        for own in world:
            _, tree = _publish_for(own, world)
            evict_from_tree_cache(tree, demand)
            evicted.append(bool(tree.evictions))

        self.assertEqual(evicted, [True, True, True])
        self.assertEqual(len(set(evicted)), 1, "the ranks must not split")

    def test_the_trigger_can_still_answer_no(self):
        """Otherwise the test above would pass against a predicate that
        always evicts -- which would be uniform but would destroy the cache
        on every allocation."""
        world = BOOT_POOLS
        demand = 1000  # every rank can fund this out of headroom

        evicted = []
        for own in world:
            _, tree = _publish_for(own, world)
            evict_from_tree_cache(tree, demand)
            evicted.append(bool(tree.evictions))

        self.assertEqual(evicted, [False, False, False])

    def test_the_floor_never_evicts_less_often_than_the_local_test(self):
        """DIRECTION SAFETY: min <= local, so every rank evicts at least as
        often as it did before the fix. Under-eviction would surface as an
        allocator OOM, and this is what makes that arithmetically impossible.
        """
        world = BOOT_POOLS
        for demand in (1, 1000, 136_000, 136_667, 143_860, 150_000, 179_825, 10**7):
            for own in world:
                _, tree = _publish_for(own, world)
                evict_from_tree_cache(tree, demand)
                fixed = bool(tree.evictions)
                local = own < demand
                if local:
                    self.assertTrue(
                        fixed,
                        msg=f"demand={demand} own={own}: fix must not evict LESS",
                    )

    # -- the publisher ------------------------------------------------------

    def test_even_pools_leave_the_floor_off(self):
        """The default path stays byte-identical: when every rank owns the
        same pool there is nothing to diverge from, the floor is not
        published, and the trigger reads its live local value exactly as
        before."""
        world = [143_860, 143_860, 143_860]
        for own in world:
            _, tree = _publish_for(own, world)
            self.assertIsNone(tree.uniform_avail_floor)
            self.assertEqual(
                uniform_avail_for_evict(tree, tree.token_to_kv_pool_allocator),
                own,
            )

    def test_uneven_pools_publish_the_group_minimum(self):
        for own in BOOT_POOLS:
            _, tree = _publish_for(own, BOOT_POOLS)
            self.assertEqual(tree.uniform_avail_floor, min(BOOT_POOLS))

    def test_single_rank_leaves_the_floor_off(self):
        tree = _FakeTreeCache(BOOT_POOLS[0])
        sched = _FakeScheduler(BOOT_POOLS[0], world_avails=None, tree_cache=tree)
        sched._update_uniform_pool_budget()
        self.assertIsNone(tree.uniform_avail_floor)
        self.assertEqual(sched.uniform_min_avail(), BOOT_POOLS[0])

    def test_the_610_admission_pin_is_unchanged(self):
        """The extra element rides the SAME reduce; the #610 quantities must
        come out exactly as they did, and there must still be exactly ONE
        collective."""
        world = BOOT_POOLS
        for own in world:
            tree = _FakeTreeCache(own)
            sched = _FakeScheduler(own, world_avails=world, tree_cache=tree)
            fake = _FakeDist(world)
            with mock.patch.object(torch, "distributed", fake):
                sched._update_uniform_pool_budget()
            self.assertEqual(fake.calls, 1, "no second reduce may appear")
            self.assertEqual(sched.uniform_min_avail(), min(world))
            self.assertEqual(sched.uniform_budget_deficit(), 0)

    def test_the_payload_width_is_rank_uniform(self):
        """A width that varied per rank would hang the reduce itself."""
        world = BOOT_POOLS
        widths = []
        for own in world:
            tree = _FakeTreeCache(own)
            sched = _FakeScheduler(own, world_avails=world, tree_cache=tree)
            fake = _FakeDist(world)
            with mock.patch.object(torch, "distributed", fake):
                sched._update_uniform_pool_budget()
            widths.extend(fake.widths)
        self.assertEqual(len(set(widths)), 1, f"widths diverged: {widths}")

    # -- the helper ---------------------------------------------------------

    def test_helper_prefers_the_floor_and_falls_back_locally(self):
        tree = _FakeTreeCache(179_825)
        alloc = tree.token_to_kv_pool_allocator
        self.assertEqual(uniform_avail_for_evict(tree, alloc), 179_825)
        tree.uniform_avail_floor = 136_667
        self.assertEqual(uniform_avail_for_evict(tree, alloc), 136_667)
        tree.uniform_avail_floor = 0
        self.assertEqual(
            uniform_avail_for_evict(tree, alloc),
            0,
            msg="a published floor of 0 is a value, not a missing floor",
        )

    # -- load-back, the second trigger --------------------------------------

    def test_the_load_back_decision_agrees_across_ranks(self):
        """Load-back extends the DEVICE prefix and needs no eviction to
        diverge: the roomy rank loads a prefix back, the tight rank gives up,
        and the trees stop being replicas from that moment.

        Modelled here as the predicate the production sites now evaluate --
        `floor < kv_tokens` instead of `available_size() < kv_tokens` -- on the
        three real pool sizes.
        """
        world = BOOT_POOLS
        kv_tokens = 150_000

        self.assertEqual(
            [avail < kv_tokens for avail in world],
            [False, True, True],
            msg="fixture must be a divergent case",
        )

        decisions = []
        for own in world:
            _, tree = _publish_for(own, world)
            floor = tree.uniform_avail_floor
            decisions.append((floor if floor is not None else own) < kv_tokens)
        self.assertEqual(decisions, [True, True, True])

    def test_a_cleared_floor_cannot_make_a_rank_evict_below_the_group(self):
        """`floor >= kv_tokens` implies THIS rank's own availability is >=
        kv_tokens, because the floor is a MIN. That is what makes the local
        eviction retry inside load_back unreachable under the pin rather than
        merely unlikely -- worth pinning, because the comment claims it."""
        world = BOOT_POOLS
        for own in world:
            _, tree = _publish_for(own, world)
            floor = tree.uniform_avail_floor
            self.assertIsNotNone(floor)
            self.assertLessEqual(floor, own)

    def test_both_load_back_sites_consult_the_floor(self):
        """Deletion falsifier for the two load-back gates.

        These sites cannot be driven hermetically without standing up a cache
        controller, host pool, lock refs and transfer plans, so what is pinned
        here is that the gate EXISTS and is consulted before the rank-local
        availability read. If someone removes it, this fails loudly instead of
        the defect returning silently under load.
        """
        import inspect

        from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        for func, name in (
            (UnifiedRadixCache.load_back, "UnifiedRadixCache.load_back"),
            (HiRadixCache.load_back, "HiRadixCache.load_back"),
        ):
            src = inspect.getsource(func)
            self.assertIn(
                "uniform_avail_floor",
                src,
                msg=f"{name} no longer consults the rank-uniform floor",
            )
            floor_at = src.index("uniform_avail_floor")
            local_at = src.find("available_size()")
            if local_at != -1:
                self.assertLess(
                    floor_at,
                    local_at,
                    msg=f"{name} reads its local pool before the group floor",
                )

    def test_the_attribute_is_declared_on_the_base_class(self):
        """Declared in the type, not conjured by whichever path happens to
        set it (#606): a reader must be able to see the pin exists."""
        from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache

        self.assertIn("uniform_avail_floor", BasePrefixCache.__annotations__)
        self.assertIsNone(BasePrefixCache.uniform_avail_floor)


if __name__ == "__main__":
    unittest.main()
