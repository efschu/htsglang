"""#639b: the MAMBA-slot eviction budget is rank-uniform, so a node's mamba
state is tombstoned on every rank or on none -- which is what keeps the radix
replicas identical and the extend prefix vector rank-uniform with them.

THE DEFECT, measured
--------------------
Two production crashes, 2026-08-07 07:45 and 10:04, both
``PrefixLensRankDivergence`` out of ``prepare_for_extend``:

    rank 0: n=1 sum=19711 has_prefix=True [19711]
    rank 1: n=1 sum=16957 has_prefix=True [16957]

One request, and the two ranks disagree about its cached prefix by 2754
tokens. Everything downstream of that vector -- ``extend_num_tokens``, and
with it the shape of every per-layer TP collective in the forward -- is
computed from it, so the ranks cannot pair.

Two floors were already in place by then and neither binds here. #616g pinned
the DEVICE KV eviction trigger (``uniform_avail_for_evict``,
``common.evict_from_tree_cache``) and the #639 first pass pinned the HOST
backup admission (``uniform_host_avail_for_backup``,
``UnifiedRadixCache.write_backup``). Both govern the KV token axis. The MAMBA
slot pool has a third, independent eviction path and it had no floor at all.

The uncovered trigger is ``common.alloc_req_slots``:

    mamba_available_size = req_to_token_pool.mamba_allocator.schedulable_available_size()
    mamba_state_needed = num_reqs * factor
    if mamba_available_size < mamba_state_needed:
        mamba_num = max(0, mamba_state_needed - mamba_available_size)
        tree_cache.evict(EvictParams(num_tokens=0, mamba_num=mamba_num))

``mamba_state_needed`` is REPLICATED -- ``num_reqs`` comes from the batch and
``factor`` from replicated server args. ``mamba_available_size`` is this
rank's own occupancy. So the roomy rank skips an eviction the tight rank
takes; and when both do evict they evict DIFFERENT AMOUNTS, because
``mamba_num`` is itself derived from the local availability. The trigger and
the magnitude are both rank-local.

WHY THE TOMBSTONE REACHES THE MATCHER
-------------------------------------
``MambaComponent.evict_component`` clears the mamba component only:

    if EvictLayer.DEVICE in target and cd.value is not None:
        self._free_mamba_value(cd.value)
        ...
        cd.value = None

The node keeps its KV and stays in the tree. But
``MambaComponent.create_match_validator(match_device_only=True)`` is

    lambda node: node.component_data[ct].value is not None

and ``UnifiedRadixCache._match_prefix_helper`` advances the match only while
EVERY component validates (``_all_valid``). So the node is still there on both
ranks and only the rank that did NOT evict will match through it. That is the
whole mechanism: a rank-local eviction becomes a rank-local prefix length.

WHY "THE POOL SIZE IS MIN-REDUCED" DOES NOT SAVE IT
---------------------------------------------------
``MambaRadixCache._alloc_mamba_slot`` carried a docstring claiming this path
is "rank-uniform without a collective", because ``max_mamba_cache_size`` is
min-reduced across ranks at startup. That claim is false and this change
removes it. A uniform pool SIZE is not a uniform eviction OUTCOME: the
startup reduce equalises how many slots each rank HAS, not how many are FREE.
Occupancy diverges from rank-local lock_ref history (``get_lru_no_lock`` skips
locked nodes, so equal-sized pools at different occupancy pick different
victims) and from the rank-local degrade branch (an exhausted pool makes that
rank alone SKIP a cache insert). ``test_equal_pool_size_does_not_imply_equal_
eviction`` below is that claim, falsified directly.

It also compounds. A shorter match extends more tokens, which takes more
slots, which forces more eviction, which shortens the next match -- which is
how one request's two ranks got 2754 tokens apart.

WHAT THIS PINS
--------------
The group MIN of the mamba allocator's ``available_size()``, published once
per scheduler iteration by ``Scheduler._publish_uniform_mamba_floor`` from the
SAME ``all_reduce(MIN)`` on ``tp_cpu_group`` that already carries the device
and host pairs -- no new collective, and no floor at all when the ranks'
occupancy agrees or the world size is 1.

THE NUMBERS BELOW are chosen to land exactly on the production pair: with the
floor off the two ranks match 19711 and 16957 tokens, and with it on they both
match 16957.
"""

import types
import unittest
from unittest import mock

import torch

try:
    from sglang.test.ci.ci_register import register_cpu_ci
except ImportError:  # pragma: no cover - registration is a CI-time marker

    def register_cpu_ci(*args, **kwargs):
        return None


from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.common import (
    peer_needs_mamba_evict,
    uniform_mamba_avail_for_evict,
)
from sglang.srt.mem_cache.unified_cache_components.mamba_component import MambaComponent
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


#: The chain of radix nodes on the crashing request's prefix path, in tokens.
#: The tail node is the 2754 the two ranks disagreed about; everything before
#: it sums to 16957, and the whole chain to 19711 -- the two numbers the
#: production detector printed.
NODE_LENS = [2048, 2048, 2048, 2048, 2048, 2048, 4669, 2754]

FULL_MATCH = sum(NODE_LENS)  # 19711
SHORT_MATCH = sum(NODE_LENS[:-1])  # 16957

#: One mamba state per node on this path.
MAMBA_PER_NODE = 1

#: Both ranks' mamba pools are the SAME SIZE -- `max_mamba_cache_size` is
#: min-reduced at startup. This is exactly the premise the false docstring
#: reasoned from, so the test grants it.
MAMBA_POOL_SIZE = 96

#: ...and their OCCUPANCY still differs, because a different number of slots
#: is pinned by running requests on each rank (`mamba_lock_ref > 0`, which
#: `get_lru_no_lock` skips). One slot apart is enough.
RANK_FREE = [12, 11]

#: `num_reqs * factor` from `alloc_req_slots`. REPLICATED: it is computed from
#: the batch and from server args, both of which are the same on every rank.
#: Sits exactly on rank 0's free count so the pre-fix branch splits the group:
#: 12 < 12 is False, 11 < 12 is True.
MAMBA_STATE_NEEDED = 12


# ---------------------------------------------------------------------------
# The 2026-08-07 10:49 specimen, three ranks, ALL-OR-NOTHING. Verified against
# /spinning/CRASH_20260807_1049_watchdog.log (traceback ~line 10923):
#
#     rank 0: n=1 sum=16063 has_prefix=True  [16063]
#     rank 1: n=1 sum=0     has_prefix=False [0]
#     rank 2: n=1 sum=0     has_prefix=False [0]
#
# Sharper than the 10:04 pair. A PARTIAL length split is consistent with
# several causes; a 16063-vs-0 split is not. Ranks 1 and 2 failed the
# device-residency validator at the FIRST node of the walk while rank 0 walked
# the entire prefix -- which is what a whole-component device tombstone looks
# like, and the mamba component is the only one this deployment tombstones
# without a uniform floor (`evict_component` nulls `cd.value` for MAMBA and
# leaves FULL's KV resident, so the node survives and only its mamba validator
# refuses).
#
# It also crosses a threshold the partial case cannot: at length 0
# `weightless_has_prefix` flips to False, so ranks 1 and 2 would have taken
# `_forward_extend_dcp`'s `if not has_prefix: ... return` and skipped the LSE
# all-gather that rank 0 entered alone. The detector's own message names that
# consequence. So the floor has to keep `has_prefix` uniform BY CONSTRUCTION,
# not merely keep the lengths close.
#
# Corroboration from the same log: the MAMBA-PIN-TRACE lines run at
# `mamba_avail` 0-3 for the whole window (`protected=4 evictable=7
# mamba_avail=2` on all three ranks at tick 27500, two seconds before the
# raise), i.e. the pool sits permanently in the regime where the unpinned
# `alloc(1) is None` trigger fires. The trace samples every 50 ticks and never
# catches the transient itself, so it corroborates the REGIME, not the
# individual eviction.
# ---------------------------------------------------------------------------

#: Rank 0's whole matched prefix in the 10:49 specimen, as one node -- the
#: divergence is at the FIRST node, so the chain's internal shape is not what
#: the case is about.
ALLNOTHING_NODE_LENS = [16063]

#: Three ranks. Rank 0 roomy, the two tight ranks exactly out of slots, which
#: is what makes their eviction tombstone the head of the walk.
ALLNOTHING_RANK_FREE = [4, 0, 0]

#: Replicated demand, above every rank's free count except rank 0's.
ALLNOTHING_STATE_NEEDED = 4


class _FakeMambaAllocator:
    def __init__(self, avail):
        self._avail = avail
        self.size = MAMBA_POOL_SIZE

    def available_size(self):
        return self._avail

    def schedulable_available_size(self):
        return self._avail


class _FakeReqToTokenPool:
    def __init__(self, mamba_avail):
        self.mamba_allocator = _FakeMambaAllocator(mamba_avail)


class _NodeData:
    """`UnifiedTreeNode.component_data[ct]`, reduced to what the mamba
    validator and the mamba tombstone touch."""

    def __init__(self, value):
        self.value = value
        self.host_value = None
        self.lock_ref = 0
        self.host_lock_ref = 0


class _FakeNode:
    def __init__(self, idx, length, mamba_locked):
        self.id = idx
        self.length = length
        # FULL keeps its KV throughout: the tombstone clears MAMBA only, which
        # is precisely why the node survives on both ranks and still splits
        # the match.
        self.component_data = {
            ComponentType.FULL: _NodeData(object()),
            ComponentType.MAMBA: _NodeData(object()),
        }
        self.mamba_locked = mamba_locked


class _FakeTreeCache:
    """The surface the publisher writes and the mamba eviction trigger reads.

    ``evict`` reproduces ``MambaComponent.drive_eviction``'s contract: walk
    this rank's OWN mamba LRU, skip locked nodes (``get_lru_no_lock``), and
    tombstone the DEVICE mamba value of each victim (``cd.value = None``)
    while leaving its KV in place.
    """

    uniform_avail_floor = None
    uniform_host_avail_floor = None
    uniform_mamba_avail_floor = None

    def __init__(self, nodes, mamba_avail):
        self.nodes = nodes
        self.req_to_token_pool = _FakeReqToTokenPool(mamba_avail)
        self.token_to_kv_pool_allocator = _FakeMambaAllocator(mamba_avail)
        self.cache_controller = None
        self.evicted_ids = []

    def is_chunk_cache(self):
        return False

    def supports_mamba(self):
        return True

    def evict(self, params):
        freed = 0
        # LRU order: deepest node first. The chain is a single path, so the
        # least recently used end of it is its tail.
        for node in reversed(self.nodes):
            if freed >= params.mamba_num:
                break
            if node.mamba_locked:
                continue  # get_lru_no_lock skips a pinned checkpoint
            cd = node.component_data[ComponentType.MAMBA]
            if cd.value is None:
                continue
            cd.value = None  # the tombstone: mamba cleared, KV left in place
            self.evicted_ids.append(node.id)
            freed += MAMBA_PER_NODE
        return freed


class _FakeScheduler:
    """Built by hand: the real constructor wants a model, a device and a
    process group, none of which this decision depends on."""

    def __init__(self, mamba_avail, world=None, tree_cache=None):
        self.token_to_kv_pool_allocator = _FakeMambaAllocator(mamba_avail)
        self.req_to_token_pool = _FakeReqToTokenPool(mamba_avail)
        self.kv_session_offload = None
        self.tree_cache = tree_cache
        self.tp_cpu_group = object() if world else None
        self.server_args = types.SimpleNamespace(dcp_size=1)
        self.ps = types.SimpleNamespace(tp_size=len(world or [1]))

    _HOST_AVAIL_ABSENT = Scheduler._HOST_AVAIL_ABSENT
    _MAMBA_AVAIL_ABSENT = Scheduler._MAMBA_AVAIL_ABSENT
    _update_uniform_pool_budget = Scheduler._update_uniform_pool_budget
    _publish_uniform_evict_floor = Scheduler._publish_uniform_evict_floor
    _publish_uniform_host_floor = Scheduler._publish_uniform_host_floor
    _publish_uniform_mamba_floor = Scheduler._publish_uniform_mamba_floor
    _local_host_avail = Scheduler._local_host_avail
    _local_mamba_avail = Scheduler._local_mamba_avail
    uniform_min_avail = Scheduler.uniform_min_avail
    uniform_budget_deficit = Scheduler.uniform_budget_deficit


class _FakeDist:
    """A real elementwise MIN all_reduce over a fixed group of per-rank
    payloads. Each rank's payload is built by the production code from its own
    pools, so the fake only has to reproduce the reduction."""

    ReduceOp = torch.distributed.ReduceOp

    def __init__(self, world_mamba, world_device=None):
        self.world_mamba = world_mamba
        # The device KV pools are EQUAL here on purpose: it isolates the mamba
        # axis, and it proves the mamba floor activates on its own rather than
        # riding the #616g activation predicate.
        self.world_device = world_device or [190400] * len(world_mamba)
        self.calls = 0
        self.widths = []

    def get_world_size(self, group):
        return len(self.world_mamba)

    def _payload(self, device_avail, mamba_avail):
        # Mirrors the production packing for pin_admission=False:
        #   [avail, -avail, host, -host, mamba, -mamba]
        absent = Scheduler._HOST_AVAIL_ABSENT
        return [
            device_avail,
            -device_avail,
            absent,
            -absent,
            mamba_avail,
            -mamba_avail,
        ]

    def all_reduce(self, t, op=None, group=None):
        self.calls += 1
        self.widths.append(t.numel())
        assert op is self.ReduceOp.MIN, "the pin is a MIN ballot"
        payloads = [
            self._payload(d, m) for d, m in zip(self.world_device, self.world_mamba)
        ]
        for i in range(t.numel()):
            t[i] = min(p[i] for p in payloads)


def _build_nodes(locked_ids):
    """The prefix chain as one rank sees it. ``locked_ids`` is that rank's own
    lock history -- which nodes a running request currently pins."""
    return [
        _FakeNode(i, length, mamba_locked=(i in locked_ids))
        for i, length in enumerate(NODE_LENS)
    ]


def _match_len(nodes):
    """``UnifiedRadixCache._match_prefix_helper``, reduced to its decision.

    The real walk advances while ``_all_valid(validators, node)`` holds over
    every component and stops at the first node that fails. The mamba
    validator is the REAL one, taken off ``MambaComponent`` -- this test does
    not paraphrase the predicate it is about.
    """
    component = object.__new__(MambaComponent)
    # #747: validators take (node, depth) and gate on the checkpoint grid;
    # this test is about lock floors, not the grid, so the grid is off.
    component.mamba_checkpoint_interval = None
    mamba_validator = MambaComponent.create_match_validator(
        component, match_device_only=True
    )

    def full_validator(node):
        return node.component_data[ComponentType.FULL].value is not None

    matched = 0
    for node in nodes:
        if not (full_validator(node) and mamba_validator(node, matched + node.length)):
            break
        matched += node.length
    return matched


def _publish_for(rank, world_mamba=None):
    """Run the production reduce for one rank and hand back its tree cache
    with the mamba floor published."""
    world_mamba = world_mamba or RANK_FREE
    nodes = _build_nodes(LOCK_HISTORY[rank])
    tree = _FakeTreeCache(nodes, world_mamba[rank])
    sched = _FakeScheduler(world_mamba[rank], world=world_mamba, tree_cache=tree)
    fake = _FakeDist(world_mamba)
    with mock.patch.object(torch, "distributed", fake):
        sched._update_uniform_pool_budget()
    return sched, tree, fake


#: The two ranks' DIFFERENT rank-local lock histories. Both pin one checkpoint
#: each, but not the same one, and neither pins the tail node -- so the LRU
#: victim is the tail on both ranks and the ONLY thing that can make their
#: tombstone sets differ is how many slots each decides to free.
LOCK_HISTORY = [{1}, {3}]


def _run_mamba_evict_trigger(tree, *, use_floor):
    """``common.alloc_req_slots``'s mamba branch, verbatim in structure.

    ``use_floor=False`` is the pre-#639b source: the local occupancy decides
    both the trigger and the magnitude.
    """
    local = tree.req_to_token_pool.mamba_allocator.schedulable_available_size()
    avail = uniform_mamba_avail_for_evict(tree, local) if use_floor else local
    if avail < MAMBA_STATE_NEEDED:
        tree.evict(types.SimpleNamespace(mamba_num=max(0, MAMBA_STATE_NEEDED - avail)))
    return avail


class UniformMambaEvictFloorTest(CustomTestCase):
    """The floor, on the boot that crashed."""

    def test_the_mamba_eviction_agrees_across_ranks(self):
        """THE REGRESSION. Both ranks reach the same prefix length."""
        matches = []
        tombstones = []
        for rank in range(len(RANK_FREE)):
            _, tree, _ = _publish_for(rank)
            _run_mamba_evict_trigger(tree, use_floor=True)
            matches.append(_match_len(tree.nodes))
            tombstones.append(sorted(tree.evicted_ids))

        self.assertEqual(
            matches[0],
            matches[1],
            "the extend prefix vector must be rank-uniform; this is the "
            f"PrefixLensRankDivergence pair {matches}",
        )
        self.assertEqual(
            tombstones[0],
            tombstones[1],
            "the two ranks must tombstone the SAME mamba nodes",
        )
        # Both land on the tight rank's answer, never on the roomy rank's:
        # the floor is a MIN, so it can only evict more.
        self.assertEqual(matches[0], SHORT_MATCH)

    def test_the_unpinned_trigger_reproduces_the_production_divergence(self):
        """THE DEFECT, with the floor bypassed. Documents what the fix closes;
        this is the pre-#639b behaviour and it is expected to diverge."""
        matches = []
        for rank in range(len(RANK_FREE)):
            _, tree, _ = _publish_for(rank)
            _run_mamba_evict_trigger(tree, use_floor=False)
            matches.append(_match_len(tree.nodes))

        self.assertNotEqual(matches[0], matches[1])
        self.assertEqual(
            matches,
            [FULL_MATCH, SHORT_MATCH],
            "the reconstruction must land on the numbers the crash printed",
        )

    def test_equal_pool_size_does_not_imply_equal_eviction(self):
        """The false docstring, falsified directly.

        Both ranks are given the SAME `max_mamba_cache_size` -- the premise
        the removed comment reasoned from -- and their eviction outcomes still
        differ, because occupancy and lock history are not the pool size.
        """
        sizes = set()
        outcomes = []
        for rank in range(len(RANK_FREE)):
            _, tree, _ = _publish_for(rank)
            sizes.add(tree.req_to_token_pool.mamba_allocator.size)
            _run_mamba_evict_trigger(tree, use_floor=False)
            outcomes.append(sorted(tree.evicted_ids))

        self.assertEqual(len(sizes), 1, "the pool sizes ARE equal")
        self.assertNotEqual(
            outcomes[0],
            outcomes[1],
            "...and the tombstone sets are still not, which is the claim the "
            "removed comment made",
        )

    def test_the_tombstone_leaves_kv_in_place(self):
        """The reason a mamba eviction is invisible to the two KV floors: it
        does not remove the node, so nothing on the KV axis notices."""
        _, tree, _ = _publish_for(1)
        _run_mamba_evict_trigger(tree, use_floor=False)
        self.assertTrue(tree.evicted_ids, "the tight rank must have evicted")
        for node_id in tree.evicted_ids:
            node = tree.nodes[node_id]
            self.assertIsNone(node.component_data[ComponentType.MAMBA].value)
            self.assertIsNotNone(
                node.component_data[ComponentType.FULL].value,
                "the KV must survive the mamba tombstone",
            )

    def test_the_floor_is_the_group_minimum(self):
        for rank in range(len(RANK_FREE)):
            _, tree, _ = _publish_for(rank)
            self.assertEqual(tree.uniform_mamba_avail_floor, min(RANK_FREE))

    def test_even_mamba_occupancy_leaves_the_floor_off(self):
        """Byte-identical default path: when the ranks agree there is nothing
        to pin, so every trigger reads its live local value exactly as before."""
        for rank in range(2):
            _, tree, _ = _publish_for(rank, world_mamba=[11, 11])
            self.assertIsNone(tree.uniform_mamba_avail_floor)

    def test_single_rank_leaves_the_floor_off(self):
        nodes = _build_nodes(set())
        tree = _FakeTreeCache(nodes, RANK_FREE[0])
        sched = _FakeScheduler(RANK_FREE[0], world=None, tree_cache=tree)
        sched._update_uniform_pool_budget()
        self.assertIsNone(tree.uniform_mamba_avail_floor)

    def test_no_mamba_pool_leaves_the_floor_off(self):
        """A boot with no mamba allocator still contributes its pair, so the
        payload width never depends on a per-rank capability -- and the
        sentinel keeps the floor off rather than pinning to 2**62."""
        nodes = _build_nodes(set())
        tree = _FakeTreeCache(nodes, RANK_FREE[0])
        sched = _FakeScheduler(RANK_FREE[0], world=RANK_FREE, tree_cache=tree)
        sched.req_to_token_pool = None
        absent = Scheduler._MAMBA_AVAIL_ABSENT
        self.assertEqual(sched._local_mamba_avail(), absent)
        fake = _FakeDist([absent, absent])
        with mock.patch.object(torch, "distributed", fake):
            sched._update_uniform_pool_budget()
        self.assertIsNone(tree.uniform_mamba_avail_floor)

    def test_the_floor_never_evicts_less_than_the_local_test(self):
        """Direction safety, exhaustively over the neighbourhood: the pinned
        availability is never ABOVE the local one, so no rank can be made to
        evict less than it did before the fix (which would be a new
        starvation, the only way this pin could itself become a fault)."""
        for local in range(0, 40):
            for floor in range(0, 40):
                tree = _FakeTreeCache(_build_nodes(set()), local)
                tree.uniform_mamba_avail_floor = floor
                self.assertLessEqual(uniform_mamba_avail_for_evict(tree, local), local)

    def test_the_pinned_value_is_identical_on_every_rank(self):
        """Uniformity is exact, not approximate: with the floor published,
        every rank's `min(local, floor)` collapses to the same number even
        though the callers read two different local quantities."""
        floor = min(RANK_FREE)
        seen = set()
        for local in RANK_FREE:
            tree = _FakeTreeCache(_build_nodes(set()), local)
            tree.uniform_mamba_avail_floor = floor
            seen.add(uniform_mamba_avail_for_evict(tree, local))
            # ...and also for a caller reading the LARGER schedulable view.
            seen.add(uniform_mamba_avail_for_evict(tree, local + 7))
        self.assertEqual(seen, {floor})

    def test_the_reduce_is_one_min_ballot_of_uniform_width(self):
        """No new collective: the mamba pair rides the reduce that already
        ran, and the payload width does not depend on any per-rank quantity."""
        widths = set()
        for rank in range(len(RANK_FREE)):
            _, _, fake = _publish_for(rank)
            self.assertEqual(fake.calls, 1, "exactly one all_reduce per iteration")
            widths.update(fake.widths)
        self.assertEqual(widths, {6})

    def test_the_616g_and_639_quantities_are_unchanged(self):
        """The mamba pair was APPENDED. The device and host floors must still
        read their own slots -- appending at the tail moved what `t[-2]` and
        `t[-1]` pointed at, which is the bug this asserts against."""
        # Device pools uneven, host absent, mamba uneven: each floor must
        # answer for its own axis and no other.
        nodes = _build_nodes(set())
        tree = _FakeTreeCache(nodes, RANK_FREE[0])
        sched = _FakeScheduler(RANK_FREE[0], world=RANK_FREE, tree_cache=tree)
        fake = _FakeDist(RANK_FREE, world_device=[190400, 143840])
        with mock.patch.object(torch, "distributed", fake):
            sched._update_uniform_pool_budget()
        self.assertEqual(tree.uniform_avail_floor, 143840)
        self.assertIsNone(
            tree.uniform_host_avail_floor,
            "no host tier => the host floor stays off; if this reads a mamba "
            "number instead, the tail indices regressed",
        )
        self.assertEqual(tree.uniform_mamba_avail_floor, min(RANK_FREE))

    def test_the_offload_branch_leaves_the_mamba_floor_off_by_design(self):
        """Named gap, not a silent one: under kv-session-offload this branch
        takes no reduce of its own, so the mamba floor is off there."""
        nodes = _build_nodes(set())
        tree = _FakeTreeCache(nodes, RANK_FREE[0])
        tree.uniform_mamba_avail_floor = 5  # stale value from a prior iteration
        sched = _FakeScheduler(RANK_FREE[0], world=RANK_FREE, tree_cache=tree)
        sched.kv_session_offload = types.SimpleNamespace(dcp_min_avail=lambda: 4242)
        sched._update_uniform_pool_budget()
        self.assertIsNone(
            tree.uniform_mamba_avail_floor,
            "the stale floor must be cleared, not carried into the next " "iteration",
        )


class PeerNeedsMambaEvictTest(CustomTestCase):
    """The companion gate for the ``alloc(1) is None`` sites."""

    def test_it_is_off_whenever_no_floor_was_published(self):
        """The default path must be untouched: with no floor there is no extra
        eviction, so a single-rank or even-occupancy boot behaves exactly as
        it did before #639b."""
        tree = _FakeTreeCache(_build_nodes(set()), RANK_FREE[0])
        self.assertIsNone(tree.uniform_mamba_avail_floor)
        self.assertFalse(peer_needs_mamba_evict(tree))

    def test_it_fires_only_when_the_group_minimum_is_dry(self):
        tree = _FakeTreeCache(_build_nodes(set()), 5)
        tree.uniform_mamba_avail_floor = 0
        self.assertTrue(peer_needs_mamba_evict(tree))
        tree.uniform_mamba_avail_floor = 1
        self.assertFalse(peer_needs_mamba_evict(tree))

    def test_the_roomy_rank_matches_the_dry_rank_tombstone_count(self):
        """The point of the gate: rank 0's own alloc succeeds, rank 1's does
        not. Without it rank 1 tombstones and rank 0 does not."""
        roomy = _FakeTreeCache(_build_nodes(LOCK_HISTORY[0]), 5)
        dry = _FakeTreeCache(_build_nodes(LOCK_HISTORY[1]), 0)
        for tree in (roomy, dry):
            tree.uniform_mamba_avail_floor = 0

        # dry rank: alloc(1) failed -> the pre-existing branch evicts.
        dry.evict(types.SimpleNamespace(mamba_num=1))
        # roomy rank: alloc(1) succeeded -> only the new gate can make it evict.
        if peer_needs_mamba_evict(roomy):
            roomy.evict(types.SimpleNamespace(mamba_num=1))

        self.assertEqual(len(roomy.evicted_ids), len(dry.evicted_ids))
        self.assertEqual(_match_len(roomy.nodes), _match_len(dry.nodes))


class WiredIntoTheProductionSitesTest(CustomTestCase):
    """The pins exist where the eviction actually happens."""

    def test_alloc_req_slots_consults_the_floor(self):
        import inspect

        from sglang.srt.mem_cache import common

        src = inspect.getsource(common.alloc_req_slots)
        self.assertIn("uniform_mamba_avail_for_evict", src)
        # ...and the evicted MAGNITUDE is derived from the pinned value, not
        # from the raw local one.
        self.assertIn("mamba_state_needed - mamba_available_size", src)

    def test_the_alloc_failure_sites_consult_the_peer_gate(self):
        import inspect

        from sglang.srt.mem_cache import mamba_radix_cache
        from sglang.srt.mem_cache.unified_cache_components import mamba_component

        # The ACTIVE implementation for hybrid SSM under hierarchical cache.
        src = inspect.getsource(mamba_component.MambaComponent)
        self.assertEqual(
            src.count("peer_needs_mamba_evict"),
            3,
            "all three mamba_allocator sites must be gated: _alloc_mamba_slot, "
            "the prefix-resume COW, and the host load-back COW",
        )
        src = inspect.getsource(mamba_radix_cache.MambaRadixCache._alloc_mamba_slot)
        self.assertIn("peer_needs_mamba_evict", src)

    def test_the_false_rank_uniform_claim_is_gone(self):
        import inspect

        from sglang.srt.mem_cache import mamba_radix_cache

        doc = inspect.getdoc(mamba_radix_cache.MambaRadixCache._alloc_mamba_slot)
        self.assertNotIn("Rank-uniform without a collective:", doc)
        self.assertIn("NOT rank-uniform on its own", doc)

    def test_the_floor_is_declared_on_the_base_class(self):
        from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache

        self.assertIsNone(BasePrefixCache.uniform_mamba_avail_floor)


def _publish_allnothing(rank):
    """The 10:49 three-rank layout, with the floor published for one rank."""
    nodes = [_FakeNode(0, ALLNOTHING_NODE_LENS[0], mamba_locked=False)]
    tree = _FakeTreeCache(nodes, ALLNOTHING_RANK_FREE[rank])
    sched = _FakeScheduler(
        ALLNOTHING_RANK_FREE[rank], world=ALLNOTHING_RANK_FREE, tree_cache=tree
    )
    fake = _FakeDist(ALLNOTHING_RANK_FREE)
    with mock.patch.object(torch, "distributed", fake):
        sched._update_uniform_pool_budget()
    return tree


def _run_allnothing_trigger(tree, *, use_floor):
    local = tree.req_to_token_pool.mamba_allocator.schedulable_available_size()
    avail = uniform_mamba_avail_for_evict(tree, local) if use_floor else local
    if avail < ALLNOTHING_STATE_NEEDED:
        tree.evict(
            types.SimpleNamespace(mamba_num=max(0, ALLNOTHING_STATE_NEEDED - avail))
        )
    return avail


def _has_prefix(prefix_lens):
    """The REAL predicate the detector's message names, not a paraphrase."""
    from sglang.srt.layers.dcp.lockstep import weightless_has_prefix

    return weightless_has_prefix(False, prefix_lens)


class AllOrNothingSpecimen1049Test(CustomTestCase):
    """The 2026-08-07 10:49 crash: rank 0 16063, ranks 1 and 2 exactly zero."""

    def test_the_unpinned_trigger_reproduces_the_1049_vector(self):
        """THE DEFECT, three ranks. Expected to diverge: this is pre-#639b."""
        matches = []
        for rank in range(len(ALLNOTHING_RANK_FREE)):
            tree = _publish_allnothing(rank)
            _run_allnothing_trigger(tree, use_floor=False)
            matches.append(_match_len(tree.nodes))

        self.assertEqual(
            matches,
            [16063, 0, 0],
            "the reconstruction must land on the vector the crash printed",
        )

    def test_has_prefix_flips_across_ranks_without_the_floor(self):
        """The consequence the partial-length specimen cannot show: at zero the
        predicate that gates `_forward_extend_dcp`'s LSE all-gather flips, so
        rank 0 would have entered a collective its peers returned early from."""
        flags = []
        for rank in range(len(ALLNOTHING_RANK_FREE)):
            tree = _publish_allnothing(rank)
            _run_allnothing_trigger(tree, use_floor=False)
            flags.append(_has_prefix([_match_len(tree.nodes)]))

        self.assertEqual(flags, [True, False, False])

    def test_the_floor_makes_the_1049_vector_uniform(self):
        """THE REGRESSION for this specimen."""
        matches = []
        tombstones = []
        for rank in range(len(ALLNOTHING_RANK_FREE)):
            tree = _publish_allnothing(rank)
            _run_allnothing_trigger(tree, use_floor=True)
            matches.append(_match_len(tree.nodes))
            tombstones.append(sorted(tree.evicted_ids))

        self.assertEqual(
            len(set(matches)),
            1,
            f"the extend prefix vector must be rank-uniform; got {matches}",
        )
        self.assertEqual(tombstones[0], tombstones[1])
        self.assertEqual(tombstones[1], tombstones[2])

    def test_has_prefix_is_uniform_with_the_floor(self):
        """BY CONSTRUCTION, not by luck: every rank derives the predicate from
        the same length, so it cannot split the group across
        `if not has_prefix: return`."""
        flags = []
        for rank in range(len(ALLNOTHING_RANK_FREE)):
            tree = _publish_allnothing(rank)
            _run_allnothing_trigger(tree, use_floor=True)
            flags.append(_has_prefix([_match_len(tree.nodes)]))

        self.assertEqual(len(set(flags)), 1, f"has_prefix split the group: {flags}")

    def test_the_floor_is_the_group_minimum_of_three_ranks(self):
        for rank in range(len(ALLNOTHING_RANK_FREE)):
            tree = _publish_allnothing(rank)
            self.assertEqual(tree.uniform_mamba_avail_floor, min(ALLNOTHING_RANK_FREE))

    def test_the_tombstone_is_at_the_head_of_the_walk(self):
        """Why the match is exactly ZERO rather than short: the evicted node is
        the first one the walk visits, so `_all_valid` fails immediately."""
        tree = _publish_allnothing(1)
        _run_allnothing_trigger(tree, use_floor=False)
        self.assertEqual(tree.evicted_ids, [0])
        self.assertEqual(_match_len(tree.nodes), 0)
        # ...and the KV is still there, which is why no KV-axis floor saw it.
        self.assertIsNotNone(tree.nodes[0].component_data[ComponentType.FULL].value)


class TheDetectorStaysADetectorTest(CustomTestCase):
    """#639b decision (a): the source is fixed, the detector still REFUSES.

    The alternative considered was to additionally MIN-reduce the prefix
    vector in ``prepare_for_extend`` so a surviving divergence no longer kills
    the server. It is rejected, and not only on the "a second silent floor
    would mask the next source" argument -- there is a mechanical reason it
    would be WRONG here specifically.

    Clamping the length vector does not clamp the state that was chosen with
    it. ``MambaComponent.finalize_match_result`` sets
    ``req.mamba_cow_src_index`` from ``last_node.component_data[MAMBA].value``
    -- the SSM checkpoint at the depth the match actually reached. Truncating
    ``prefix_indices`` to the group minimum afterwards would leave the request
    resuming an SSM state that has already consumed N tokens while its KV
    prefix and token stream describe M < N. That is not a shorter-but-correct
    forward, it is a silently wrong one: the two ranks would then AGREE on a
    vector and disagree on the hidden state behind it, which is strictly worse
    than the crash because nothing downstream can detect it.

    A correction that is actually safe would have to re-run ``match_prefix``
    against the clamped depth so the mamba checkpoint, the lock refs and
    ``last_node`` are chosen together. That is a different change from a MIN
    on a length vector, and it is not what this crash needs now that the
    divergence SOURCE is pinned.
    """

    def test_the_check_refuses_rather_than_clamping(self):
        import inspect

        from sglang.srt.layers.dcp import prefix_lens_check

        src = inspect.getsource(prefix_lens_check.assert_prefix_lens_rank_uniform)
        self.assertIn("raise PrefixLensRankDivergence", src)
        # No write-back of a reduced vector: the ballot is read, never applied.
        self.assertNotIn("prefix_lens[:]", src)
        self.assertNotIn("prefix_lens =", src)

    def test_prepare_for_extend_does_not_min_reduce_the_vector(self):
        import inspect

        from sglang.srt.managers.schedule_batch import ScheduleBatch

        src = inspect.getsource(ScheduleBatch.prepare_for_extend)
        head = src[: src.index("assert_prefix_lens_rank_uniform(prefix_lens)")]
        tail = src[src.index("assert_prefix_lens_rank_uniform(prefix_lens)") :]
        # The vector handed to the detector is the one built from the radix
        # match, and it is not rewritten after the detector returns.
        self.assertIn("prefix_lens = [len(r.prefix_indices) for r in reqs]", head)
        self.assertNotIn("prefix_lens = [min(", tail)
        self.assertNotIn("ReduceOp.MIN", tail)

    def test_the_mamba_checkpoint_is_chosen_with_the_matched_depth(self):
        """The fact that makes a bare length clamp unsafe: the SSM checkpoint
        is bound to the node the match reached, not to a length."""
        import inspect

        from sglang.srt.mem_cache.unified_cache_components import mamba_component

        src = inspect.getsource(mamba_component.MambaComponent.finalize_match_result)
        self.assertIn("mamba_value = last_node.component_data", src)
        self.assertIn("req.mamba_cow_src_index = mamba_value", src)


if __name__ == "__main__":
    unittest.main()
