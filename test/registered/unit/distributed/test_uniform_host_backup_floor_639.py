"""#639: the HOST-tier backup admission is rank-uniform, so a node is either
backed up on every rank or on none -- which is what keeps the device radix
replicas identical and the extend token count rank-uniform with them.

THE DEFECT, measured
--------------------
Four production wedges carry one signature, and it is always the same shape:

    2026-08-06 21:54   rank 0 1690 tokens   peers 1818
    2026-08-06 22:09   rank 0  828 tokens   peers 2048
    2026-08-07 07:45   rank 0  912 tokens   peers 2048
    2026-08-07 07:54   rank 0  914 tokens   peers 2048

Rank 0 always LOW, and the two peers always agreeing with each other. That
correlation is not about rank ids: it tracks POOL SIZE. On the crashing boot
the host KV pools are 359652 / 287722 / 273336 slots -- ranks 1 and 2 within
5% of each other, rank 0 25-32% larger -- so ranks 1 and 2 cross a
pool-sized boundary together and rank 0 does not.

The boundary is ``UnifiedRadixCache.write_backup``:

    host_avail = self.cache_controller.mem_pool_host.available_size()
    if host_avail < kv_tokens:
        needed = kv_tokens - host_avail
        evicted = self.evict_host(needed)
        if evicted < needed:
            return 0

``kv_tokens`` is replicated (it is the node's own length); ``host_avail`` is
this rank's own host shard. The roomy rank backs the node up and the tight
ranks return 0, so ``node.backuped`` diverges.

``node.backuped`` is not a bookkeeping flag. It selects the eviction
STRATEGY, in ``UnifiedRadixCache._evict_device_leaf``:

    if not node.backuped:
        if ... write_policy == "write_back":
            ...
        else:
            # Write-through: node has no backup, delete entirely.
            ...
            self._remove_leaf_from_parent(node)
            return
    self._evict_to_host(node, tracker)

So under this boot's ``write_through`` policy a backed-up node is DEMOTED and
stays in the tree (still matchable, and restorable by ``load_back``), while a
node without a backup is REMOVED FROM THE TREE. The rank with the roomiest
host pool therefore keeps a prefix its peers have deleted, ``match_prefix``
hands it a longer ``prefix_indices``, and ``prepare_for_extend``
(schedule_batch.py:2235) turns that straight into a smaller token axis:

    input_ids = [r.get_fill_ids()[len(r.prefix_indices):] for r in reqs]
    extend_num_tokens = sum(len(ids) for ids in input_ids)

Every per-layer TP all_reduce of that forward is then entered with a
rank-dependent shape and BAR1 spins on a contribution that cannot arrive --
abort word CLEAN, which is a stall, not a trip.

WHY THE #616B FLOOR DID NOT BIND
--------------------------------
#616B pinned the two DEVICE-side triggers (``evict_from_tree_cache`` and
both ``load_back`` sites) to the group MIN of the DEVICE pool. Both pins are
deployed and both are live on this boot. Neither can reach this defect: the
divergence here is decided against the HOST pool, one tier down, and it is
decided BEFORE the device-side question is asked. ``load_back``'s floor gates
whether a load-back is ATTEMPTED given host content; it cannot make host
content EXIST on a rank whose backup was refused.

NOTE_616g named this surface and dropped a drafted pin for it, with a stated
reason that does not survive reading ``_evict_device_leaf``: "under this
boot's write_through policy the eviction path gates on write_policy ==
'write_back', not on backup state, so the chain to the device tree is NOT
established". The quoted branch above shows the eviction path gating on
``node.backuped`` FIRST and on ``write_policy`` only inside that branch --
and the write_through arm is precisely the one that deletes the node. The
chain is established, and it is established only on the write_through path.

Deliberately hermetic: no CUDA, no real process group, no model, no cache
controller. The reduce is driven through a fake that performs a real
elementwise MIN over a fixed set of per-rank payloads, exactly as the #616g
suite does.
"""

import inspect
import types
import unittest
from unittest import mock

import torch

try:
    from sglang.test.ci.ci_register import register_cpu_ci
except ImportError:  # pragma: no cover - registration is a CI-time marker

    def register_cpu_ci(*args, **kwargs):
        return None


from sglang.srt.managers.prefetch_ballot import build_prefetch_ballot_payload
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.common import uniform_host_avail_for_backup
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


#: The three DEVICE pools of the crashing boot, in tokens. Rank 0 is the
#: roomiest, which is why it is the rank that keeps prefixes its peers drop.
BOOT_DEVICE_POOLS = [190400, 143840, 143906]

#: The three HOST KV pools of the same boot, in slots, from the
#: "HiCache host KV pool (N tokens)" lines of the 07:12 boot:
#:   TP0 359652   TP1 287722   TP2 273336
#: This is the axis the defect is decided on.
BOOT_HOST_POOLS = [359652, 287722, 273336]


class _FakeAllocator:
    def __init__(self, avail):
        self._avail = avail

    def available_size(self):
        return self._avail


class _FakeHostPool:
    def __init__(self, avail):
        self._avail = avail

    def available_size(self):
        return self._avail


class _FakeCacheController:
    def __init__(self, host_avail):
        self.mem_pool_host = _FakeHostPool(host_avail)
        self.write_policy = "write_through"


class _FakeTreeCache:
    """Only the surface the publisher and the backup gate touch."""

    uniform_avail_floor = None
    uniform_host_avail_floor = None
    uniform_mamba_avail_floor = None

    def __init__(self, device_avail, host_avail):
        self.token_to_kv_pool_allocator = _FakeAllocator(device_avail)
        self.cache_controller = _FakeCacheController(host_avail)

    def is_chunk_cache(self):
        return False

    def evict(self, params):  # pragma: no cover - not exercised here
        raise AssertionError("the host-tier pin must not touch device eviction")


class _FakeScheduler:
    """Built by hand: the real constructor wants a model, a device and a
    process group, none of which this decision depends on."""

    def __init__(self, device_avail, host_avail, world=None, tree_cache=None):
        self.token_to_kv_pool_allocator = _FakeAllocator(device_avail)
        self.kv_session_offload = None
        self.tree_cache = tree_cache
        self.tp_cpu_group = object() if world else None
        self.server_args = types.SimpleNamespace(dcp_size=1)
        # #791b: the reduce now also carries the prefetch ballot; model the
        # two inputs it reads. Storage off makes the SHIPPED drain (bound
        # below) return {} without touching a tree cache, so the ballot
        # contributes only neutral slots and the quantities under test stay
        # untouched.
        self.waiting_queue = []
        self.enable_hicache_storage = False
        self.ps = types.SimpleNamespace(tp_size=len(world or [1]))

    _HOST_AVAIL_ABSENT = Scheduler._HOST_AVAIL_ABSENT
    # #639b: the mamba pair rides the same reduce. This fixture has no
    # `req_to_token_pool`, so it contributes the ABSENT sentinel and no mamba
    # floor is published -- the #639 host quantities stay untouched.
    _MAMBA_AVAIL_ABSENT = Scheduler._MAMBA_AVAIL_ABSENT
    _local_mamba_avail = Scheduler._local_mamba_avail
    _publish_uniform_mamba_floor = Scheduler._publish_uniform_mamba_floor
    _update_uniform_pool_budget = Scheduler._update_uniform_pool_budget
    _publish_uniform_evict_floor = Scheduler._publish_uniform_evict_floor
    _publish_uniform_host_floor = Scheduler._publish_uniform_host_floor
    _local_host_avail = Scheduler._local_host_avail
    uniform_min_avail = Scheduler.uniform_min_avail
    uniform_budget_deficit = Scheduler.uniform_budget_deficit
    # #791b: and the PREFETCH BALLOT, same reason one release later again.
    _drain_prefetch_progress = Scheduler._drain_prefetch_progress


class _FakeDist:
    """A real elementwise MIN all_reduce over a fixed group of per-rank
    payloads. Each rank's payload is built by the production code from its own
    pools, so the fake only has to reproduce the reduction."""

    ReduceOp = torch.distributed.ReduceOp

    def __init__(self, world_device, world_host):
        self.world_device = world_device
        self.world_host = world_host
        self.calls = 0
        self.widths = []

    def get_world_size(self, group):
        return len(self.world_device)

    def _payload(self, device_avail, host_avail):
        # Mirrors the production packing for pin_admission=False. #639b
        # appended the mamba pair AFTER the host pair, which is why the
        # publisher stopped reading the host term off `t[-2]`/`t[-1]`:
        #   [avail, -avail, host, -host, mamba, -mamba]
        m_absent = Scheduler._MAMBA_AVAIL_ABSENT
        return [
            device_avail,
            -device_avail,
            host_avail,
            -host_avail,
            m_absent,
            -m_absent,
            # #791b: the prefetch ballot rides behind the mamba pair;
            # neutral on every fixture rank (empty queue).
            *build_prefetch_ballot_payload([], {}),
        ]

    def all_reduce(self, t, op=None, group=None):
        self.calls += 1
        self.widths.append(t.numel())
        assert op is self.ReduceOp.MIN, "the pin is a MIN ballot"
        payloads = [
            self._payload(d, h) for d, h in zip(self.world_device, self.world_host)
        ]
        for i in range(t.numel()):
            t[i] = min(p[i] for p in payloads)


def _publish_for(rank, world_device=None, world_host=None):
    """Run the production reduce for one rank and hand back its tree cache
    with both floors published."""
    world_device = world_device or BOOT_DEVICE_POOLS
    world_host = world_host or BOOT_HOST_POOLS
    tree = _FakeTreeCache(world_device[rank], world_host[rank])
    sched = _FakeScheduler(
        world_device[rank], world_host[rank], world=world_device, tree_cache=tree
    )
    fake = _FakeDist(world_device, world_host)
    with mock.patch.object(torch, "distributed", fake):
        sched._update_uniform_pool_budget()
    return sched, tree, fake


class _StubComponent:
    """The base component, reduced to the two calls ``write_backup`` makes on
    it. Auxiliary components are absent (``_components_tuple`` is empty), so
    the transfer plan is the KV one and nothing else."""

    component_type = None  # set to BASE_COMPONENT_TYPE at construction

    def __init__(self, base_type):
        self.component_type = base_type
        self.commits = 0

    def build_hicache_transfers(self, node, phase, **kwargs):  # pragma: no cover
        return []

    def commit_hicache_transfer(self, node, phase, transfers=None):
        self.commits += 1


class _NodeData:
    def __init__(self, value):
        self.value = value


class _FakeNode:
    def __init__(self, base_type, kv_tokens, parent):
        self.id = 1
        self.parent = parent
        self.backuped = False
        self.component_data = {base_type: _NodeData(list(range(kv_tokens)))}


class _FakeUnifiedCache:
    """Drives the REAL ``UnifiedRadixCache.write_backup``.

    Only the surface that function touches is supplied. It is bound unbound
    rather than reimplemented on purpose: a fixture that re-spelled the gate
    would pass whatever the gate said, which is the failure mode the #616g
    suite's deletion falsifier exists to catch. Reverting the production gate
    to ``mem_pool_host.available_size()`` makes the falsifier below fail
    because this really is that function running.
    """

    def __init__(self, base_type, host_avail, host_evictable=0):
        self.cache_controller = _FakeCacheController(host_avail)
        self.cache_controller.write = self._write
        self.root_node = object()
        self.uniform_host_avail_floor = None
        self.uniform_avail_floor = None
        self._base_type = base_type
        self.components = {base_type: _StubComponent(base_type)}
        self._components_tuple = ()
        self._host_evictable = host_evictable
        self.evicted_host = 0
        self.tracked = 0

    # -- the surface write_backup calls, and nothing else ------------------
    def _build_sidecar_transfers(self, phase, kv_xfer, comp_xfers):
        return []

    def evict_host(self, num_tokens, component_type=None):
        freed = min(num_tokens, self._host_evictable)
        self._host_evictable -= freed
        self.evicted_host += freed
        return freed

    def _write(self, device_indices, node_id=None, extra_pools=None):
        return list(device_indices)

    def inc_lock_ref(self, node):
        return types.SimpleNamespace(to_dec_params=lambda: None)

    def _track_write_through_node(self, node, lock_params):
        self.tracked += 1

    write_backup = None  # bound below, after UnifiedRadixCache is imported


def _backup_admitted(tree, kv_tokens):
    """Run the REAL ``UnifiedRadixCache.write_backup`` for this rank and
    report whether the node got a host backup. True = backed up."""
    from sglang.srt.mem_cache.unified_cache_components import BASE_COMPONENT_TYPE
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    cache = _FakeUnifiedCache(
        BASE_COMPONENT_TYPE, tree.cache_controller.mem_pool_host.available_size()
    )
    # The floor the scheduler published for this rank is the whole point of
    # the fixture; carry it onto the object write_backup actually reads.
    cache.uniform_host_avail_floor = tree.uniform_host_avail_floor
    node = _FakeNode(BASE_COMPONENT_TYPE, kv_tokens, cache.root_node)
    written = UnifiedRadixCache.write_backup(cache, node)
    return written > 0


class UniformHostBackupFloorTest(CustomTestCase):
    # -- the falsifier ------------------------------------------------------

    def test_the_backup_admission_agrees_across_ranks(self):
        """THE FALSIFIER.

        A replicated node length that sits BETWEEN the ranks' host headrooms
        is exactly the divergent case, and on this boot's pools it is an
        ordinary one: any node between 287722 and 359652 slots splits the
        group. Before the fix each rank answered from its own host pool, so
        ``node.backuped`` diverged, ``_evict_device_leaf`` then deleted the
        node on the tight ranks and demoted it on the roomy one, and the next
        ``match_prefix`` returned a rank-dependent prefix.

        Reverting the fix -- reading ``mem_pool_host.available_size()`` in
        ``write_backup`` again -- turns the second assertion back into
        [True, False, False] and fails this test.
        """
        kv_tokens = 300_000  # < 359652, > 287722 and > 273336

        # The fixture must really be divergent, or the assertion below proves
        # nothing.
        self.assertEqual(
            [avail >= kv_tokens for avail in BOOT_HOST_POOLS],
            [True, False, False],
            msg="fixture is not divergent; the test would prove nothing",
        )

        # POSITIVE CONTROL, through the REAL write_backup: with no floor
        # published -- which is what the pre-#639 tree does on every
        # iteration -- the same three ranks really do split. Without this the
        # assertion below could be passing against a fixture that never
        # diverged in the first place.
        unpinned = []
        for rank in range(3):
            _, tree, _ = _publish_for(rank)
            tree.uniform_host_avail_floor = None
            unpinned.append(_backup_admitted(tree, kv_tokens))
        self.assertEqual(unpinned, [True, False, False])

        admitted = []
        for rank in range(3):
            _, tree, _ = _publish_for(rank)
            admitted.append(_backup_admitted(tree, kv_tokens))

        self.assertEqual(admitted, [False, False, False])
        self.assertEqual(len(set(admitted)), 1, "the ranks must not split")

    def test_a_divergent_backup_verdict_diverges_the_token_axis(self):
        """The falsifier above pins the DECISION; this pins the CONSEQUENCE,
        so the decision cannot be dismissed as bookkeeping.

        The arithmetic is ``prepare_for_extend``'s own (schedule_batch.py:2236,
        pinned structurally by
        ``test_the_extend_token_axis_still_reads_the_matched_prefix``): a rank
        that keeps a backed-up prefix matches it and computes that many fewer
        new tokens. With the split verdict the ranks produce two different
        token counts for one logical collective -- which is the wedge. With
        the uniform verdict they produce one.
        """
        seq_len = 3000
        prefix_without_the_kept_node = 952
        kept_node_len = 1136  # what the roomy rank keeps and the peers delete

        def extend_num_tokens(keeps_node: bool) -> int:
            prefix = prefix_without_the_kept_node + (
                kept_node_len if keeps_node else 0
            )
            return seq_len - prefix

        # Unpinned: rank 0 keeps the node, peers do not.
        split = [extend_num_tokens(v) for v in (True, False, False)]
        self.assertEqual(split, [912, 2048, 2048])
        self.assertGreater(
            len(set(split)),
            1,
            msg="the unpinned verdict must produce a divergent token axis",
        )

        # Pinned: the group floor refuses the backup on every rank.
        kv_tokens = 300_000
        verdicts = []
        for rank in range(3):
            _, tree, _ = _publish_for(rank)
            verdicts.append(_backup_admitted(tree, kv_tokens))
        pinned = [extend_num_tokens(v) for v in verdicts]
        self.assertEqual(
            len(set(pinned)), 1, "the pinned verdict must produce ONE token axis"
        )

    def test_the_gate_can_still_answer_yes(self):
        """Otherwise the falsifier would pass against a gate that refuses
        every backup -- uniform, but it would disable the host tier."""
        kv_tokens = 1000  # every rank's host pool can hold this
        admitted = []
        for rank in range(3):
            _, tree, _ = _publish_for(rank)
            admitted.append(_backup_admitted(tree, kv_tokens))
        self.assertEqual(admitted, [True, True, True])

    def test_the_floor_never_admits_more_than_the_local_test(self):
        """DIRECTION SAFETY: min <= local, so the pinned gate refuses whenever
        the local gate refused, and sometimes when it did not. Over-admission
        into the host pool -- which is what would turn this pin into a new
        failure -- is arithmetically impossible."""
        for kv_tokens in (1, 1000, 273_336, 273_337, 287_722, 300_000, 359_652, 10**7):
            for rank in range(3):
                _, tree, _ = _publish_for(rank)
                pinned = _backup_admitted(tree, kv_tokens)
                local = BOOT_HOST_POOLS[rank] >= kv_tokens
                if pinned:
                    self.assertTrue(
                        local,
                        msg=(
                            f"kv_tokens={kv_tokens} rank={rank}: the pin admitted "
                            "a backup the local pool cannot hold"
                        ),
                    )

    # -- the publisher ------------------------------------------------------

    def test_even_host_pools_leave_the_floor_off(self):
        """The default path stays byte-identical: when every rank owns the
        same host pool there is nothing to diverge from, no floor is
        published, and the gate reads its live local value exactly as before.
        """
        world_host = [287_722, 287_722, 287_722]
        for rank in range(3):
            _, tree, _ = _publish_for(rank, world_host=world_host)
            self.assertIsNone(tree.uniform_host_avail_floor)
            self.assertEqual(
                uniform_host_avail_for_backup(
                    tree, tree.cache_controller.mem_pool_host
                ),
                world_host[rank],
            )

    def test_uneven_host_pools_publish_the_group_minimum(self):
        for rank in range(3):
            _, tree, _ = _publish_for(rank)
            self.assertEqual(tree.uniform_host_avail_floor, min(BOOT_HOST_POOLS))

    def test_single_rank_leaves_the_floor_off(self):
        tree = _FakeTreeCache(BOOT_DEVICE_POOLS[0], BOOT_HOST_POOLS[0])
        sched = _FakeScheduler(
            BOOT_DEVICE_POOLS[0], BOOT_HOST_POOLS[0], world=None, tree_cache=tree
        )
        sched._update_uniform_pool_budget()
        self.assertIsNone(tree.uniform_host_avail_floor)
        self.assertIsNone(tree.uniform_avail_floor)

    def test_no_host_tier_leaves_the_floor_off(self):
        """A boot without --enable-hierarchical-cache has no host pool at all.
        The element still rides the reduce (width must not vary per rank), but
        it carries the ABSENT sentinel on every rank and no floor is
        published."""
        for rank in range(3):
            tree = _FakeTreeCache(BOOT_DEVICE_POOLS[rank], BOOT_HOST_POOLS[rank])
            tree.cache_controller = None
            sched = _FakeScheduler(
                BOOT_DEVICE_POOLS[rank],
                BOOT_HOST_POOLS[rank],
                world=BOOT_DEVICE_POOLS,
                tree_cache=tree,
            )
            absent = Scheduler._HOST_AVAIL_ABSENT
            fake = _FakeDist(BOOT_DEVICE_POOLS, [absent] * 3)
            with mock.patch.object(torch, "distributed", fake):
                sched._update_uniform_pool_budget()
            self.assertIsNone(tree.uniform_host_avail_floor)

    # -- the reduce it rides on --------------------------------------------

    def test_the_616g_and_610_quantities_are_unchanged(self):
        """The host elements ride the SAME reduce. The device-side floor and
        the #610 admission deficit must come out exactly as they did, and
        there must still be exactly ONE collective."""
        for rank in range(3):
            sched, tree, fake = _publish_for(rank)
            self.assertEqual(fake.calls, 1, "no second reduce may appear")
            self.assertEqual(sched.uniform_min_avail(), min(BOOT_DEVICE_POOLS))
            self.assertEqual(sched.uniform_budget_deficit(), 0)
            self.assertEqual(tree.uniform_avail_floor, min(BOOT_DEVICE_POOLS))

    def test_the_payload_width_is_rank_uniform(self):
        """A width that varied per rank would hang the reduce it rides on."""
        widths = []
        for rank in range(3):
            _, _, fake = _publish_for(rank)
            widths.extend(fake.widths)
        self.assertEqual(len(set(widths)), 1, f"payload width varies: {widths}")

    def test_the_offload_branch_leaves_the_host_floor_off_by_design(self):
        """NAMED GAP, pinned so it cannot become an unnoticed one.

        Under kv-session-offload the device minimum comes from the offload
        manager's own reduce and this branch takes none of its own -- an
        invariant test_uniform_decode_mem_603 pins by name. Carrying the host
        term would mean adding a reduce there, trading a divergence that
        branch HAS closed for one it has not, on a path no CPU test can
        exercise. So the host floor stays off and `write_backup` keeps its
        pre-#639 local gate there. The close belongs in
        `update_dcp_admission_state`'s existing packed reduce.
        """
        tree = _FakeTreeCache(BOOT_DEVICE_POOLS[0], BOOT_HOST_POOLS[0])
        sched = _FakeScheduler(
            BOOT_DEVICE_POOLS[0],
            BOOT_HOST_POOLS[0],
            world=BOOT_DEVICE_POOLS,
            tree_cache=tree,
        )
        sched.kv_session_offload = types.SimpleNamespace(
            dcp_min_avail=lambda: min(BOOT_DEVICE_POOLS)
        )
        fake = _FakeDist(BOOT_DEVICE_POOLS, BOOT_HOST_POOLS)
        with mock.patch.object(torch, "distributed", fake):
            sched._update_uniform_pool_budget()
        self.assertEqual(fake.calls, 0, "the offload branch must take no reduce")
        self.assertIsNone(tree.uniform_host_avail_floor)

    # -- deletion falsifiers ------------------------------------------------

    def test_write_backup_consults_the_floor_in_both_cache_classes(self):
        """``write_backup`` cannot be driven hermetically without standing up
        a cache controller, a host pool, transfer plans and lock refs, so what
        is pinned here is that the gate EXISTS and is consulted BEFORE the
        rank-local availability read -- in both classes.

        Both, because #616g's load-back pin was first written into the class
        this rig does not instantiate and half the fix sat dead for a boot.
        The rig runs ``UnifiedRadixCache``; ``HiRadixCache`` carries the same
        gate for the deployments that use it.
        """
        from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        for func, name in (
            (UnifiedRadixCache.write_backup, "UnifiedRadixCache.write_backup"),
            (HiRadixCache.write_backup, "HiRadixCache.write_backup"),
        ):
            src = inspect.getsource(func)
            self.assertIn(
                "uniform_host_avail_for_backup",
                src,
                msg=f"{name} no longer consults the rank-uniform host floor",
            )
            floor_at = src.index("uniform_host_avail_for_backup")
            local_at = src.find("mem_pool_host.available_size()")
            if local_at != -1:
                self.assertLess(
                    floor_at,
                    local_at,
                    msg=f"{name} reads its local host pool before the group floor",
                )

    def test_the_eviction_strategy_still_branches_on_backuped(self):
        """The link that turns a backup verdict into a TREE difference. If
        ``_evict_device_leaf`` stops deleting the un-backed-up node under
        write_through, the chain this pin closes no longer exists -- and the
        pin's justification would have to be rewritten rather than silently
        kept.
        """
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        src = inspect.getsource(UnifiedRadixCache._evict_device_leaf)
        self.assertIn("if not node.backuped:", src)
        self.assertIn("_remove_leaf_from_parent(node)", src)
        self.assertIn("_evict_to_host(node, tracker)", src)

    def test_the_extend_token_axis_still_reads_the_matched_prefix(self):
        """The other end of the chain: a rank-dependent ``prefix_indices``
        becomes a rank-dependent collective shape here and nowhere else."""
        from sglang.srt.managers.schedule_batch import ScheduleBatch

        src = inspect.getsource(ScheduleBatch.prepare_for_extend)
        self.assertIn("len(r.prefix_indices)", src)
        self.assertIn("extend_num_tokens = sum(len(ids) for ids in input_ids)", src)

    def test_the_attribute_is_declared_on_the_base_class(self):
        """Declared in the type, not conjured by whichever path happens to set
        it (#606): a reader must be able to see the pin exists."""
        from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache

        self.assertIn("uniform_host_avail_floor", BasePrefixCache.__annotations__)
        self.assertIsNone(BasePrefixCache.uniform_host_avail_floor)


class HasPrefixIsNotRankUniformTest(CustomTestCase):
    """#639, the 08:26 DEGRADED specimen: the same rank-local vector also
    decides WHICH COLLECTIVES RUN, not only how big they are.

    Caught live with all three ranks alive, seven lines apart in one layer
    body and on two different group collectives:

        TP0        qwen3_5.py:1241 -> prepare_mlp -> _gather_hidden_states...
                   -> attention_tensor_model_parallel_all_reduce   ALL_REDUCE
        TP1, TP2   qwen3_5.py:1234 -> self_attention -> radix_attention
                   -> _forward_extend_dcp (flashinfer_backend.py:5905)
                   -> cp_lse_ag_out_ar_mha_uneven -> _ag_lse   ALL_GATHER

    TP0 is already past attention; its peers are inside an LSE all-gather it
    never joins. The branch that lets a rank leave early is
    ``_forward_extend_dcp``'s

        if not has_prefix:
            ...
            return o_cur.contiguous().view(...)

    guarded only by a comment: "has_prefix is a global (rank-uniform)
    property, so every rank takes the same branch -> the DCP collectives
    below stay balanced (no deadlock)". The three branches BELOW that return
    each carry their own note that they are rank-local but reach the
    collective identically -- the author defended those and left this one
    undefended. #505 class; #131 is the same property failing before.

    THE LINK TO THE TOKEN-COUNT SPECIMENS, established by construction rather
    than by correlation. ``weightless_has_prefix`` reduces to
    ``any(extend_prefix_lens_cpu)`` (lockstep.py:143), and that vector's
    provenance is one line:

        schedule_batch.py:2239  prefix_lens = [len(r.prefix_indices) for r in reqs]
                        :2264  self.prefix_lens = prefix_lens
        forward_batch_info.py:855  ret.extend_prefix_lens_cpu = extend_prefix_lens

    which is the SAME ``len(r.prefix_indices)`` that schedule_batch.py:2235
    turns into ``extend_num_tokens``. So ``has_prefix`` and the token axis are
    two consumers of one rank-local vector: any prefix divergence produces
    both symptoms, and a shape mismatch and a sequence mismatch are the same
    defect seen at two amplitudes. The docstring's claim that the vector is
    replicated rests on it being "a host-side length vector, not a per-rank
    tensor" -- true about its type, false about its content.
    """

    def test_has_prefix_is_a_pure_function_of_a_rank_local_vector(self):
        """THE FALSIFIER for the 08:26 specimen: drive the REAL predicate with
        the rank-divergent input the dumps imply and watch the group split.

        TP0 sees no prefix and takes the early return; its peers see one and
        enter the LSE all-gather. Nothing here is mocked -- this is
        ``weightless_has_prefix`` itself, on three length vectors that differ
        the way three radix replicas differ once they stop being replicas.
        """
        from sglang.srt.layers.dcp.lockstep import weightless_has_prefix

        # The 08:26 shape: TP0's request carries no matched prefix, its peers'
        # carries one. `forces_prefix` is False -- this is an EXTEND, not a
        # target-verify (the dumps' _execute_extend confirms it).
        per_rank_prefix_lens = [[0], [2048], [2048]]

        verdicts = [
            weightless_has_prefix(False, lens) for lens in per_rank_prefix_lens
        ]
        self.assertEqual(verdicts, [False, True, True])
        self.assertGreater(
            len(set(verdicts)),
            1,
            msg=(
                "the predicate must be shown to SPLIT before anything claims "
                "it is rank-uniform"
            ),
        )

    def test_the_split_is_reachable_from_a_one_token_prefix_difference(self):
        """It does not take a large divergence. ONE token of prefix on one
        rank and none on another flips the branch, so any prefix divergence at
        all -- not merely a large one -- is enough to desync the group."""
        from sglang.srt.layers.dcp.lockstep import weightless_has_prefix

        self.assertFalse(weightless_has_prefix(False, [0]))
        self.assertTrue(weightless_has_prefix(False, [1]))

    def test_verify_still_forces_the_prefix_branch(self):
        """The one case that IS pinned today stays pinned: a target-verify
        batch forces the branch regardless of the length vector, which is
        #180's rule and must not regress."""
        from sglang.srt.layers.dcp.lockstep import weightless_has_prefix

        self.assertTrue(weightless_has_prefix(True, [0]))
        self.assertTrue(weightless_has_prefix(True, None))

    def test_the_early_return_still_precedes_the_collective(self):
        """The structural half: if the early return is ever moved below
        ``cp_lse_ag_out_ar_mha_uneven`` -- or removed -- this hazard is gone
        and the analysis above has to be rewritten rather than silently kept.
        """
        from sglang.srt.layers.attention.flashinfer_backend import (
            FlashInferAttnBackend,
        )

        src = inspect.getsource(FlashInferAttnBackend._forward_extend_dcp)
        self.assertIn("if not has_prefix:", src)
        early_at = src.index("if not has_prefix:")
        collective_at = src.index("cp_lse_ag_out_ar_mha_uneven")
        self.assertLess(
            early_at,
            collective_at,
            msg="the early return no longer precedes the LSE all-gather",
        )

    def test_the_predicate_and_the_token_axis_read_the_same_vector(self):
        """The LINK, pinned so it cannot be quietly broken in either
        direction: ``has_prefix`` and ``extend_num_tokens`` are both functions
        of ``len(r.prefix_indices)``."""
        from sglang.srt.layers.dcp.lockstep import weightless_has_prefix
        from sglang.srt.managers.schedule_batch import ScheduleBatch

        batch_src = inspect.getsource(ScheduleBatch.prepare_for_extend)
        self.assertIn(
            "prefix_lens = [len(r.prefix_indices) for r in reqs]", batch_src
        )
        self.assertIn("extend_num_tokens = sum(len(ids) for ids in input_ids)", batch_src)

        pred_src = inspect.getsource(weightless_has_prefix)
        self.assertIn("return any(extend_prefix_lens_cpu)", pred_src)


if __name__ == "__main__":
    unittest.main()
