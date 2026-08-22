"""#645: the write-backup path performs NO rank-local host eviction, so the
radix replicas stay identical and ``match_prefix`` stays rank-uniform.

THE DEFECT, measured
--------------------
Two production wedges on 2026-08-07, both with
``impl=UnifiedRadixCache hybrid_ssm=True hierarchical=True`` in the boot
banner (CRASH_20260807_1215_watchdog.log:600 and
CRASH_20260807_1326_watchdog.log), both under
``--hicache-write-policy write_through``:

    12:15   rank 0 [2047]    rank 1/2 [10238]     roomy rank SHORTER
    13:26   rank 0 [22014]   rank 1/2 [19967]     roomy rank LONGER

The 13:26 difference is exactly 22014 - 19967 = 2047 tokens, one
``chunked_prefill_size`` chunk: rank 0 matched exactly one chunk node more
than its peers. Rank 0 is the 5090, the roomiest pool on the rig.

BOTH DIRECTIONS COME OUT OF ONE BRANCH, which is the point of this suite.
``UnifiedRadixCache.write_backup`` (unified_radix_cache.py:1801-1809) reads::

    host_avail = uniform_host_avail_for_backup(self, ...)
    if host_avail < kv_tokens:
        needed = kv_tokens - host_avail
        evicted = self.evict_host(needed)
        if evicted < needed:
            return 0

#639's floor made ``host_avail`` REPLICATED, so the ``if`` is a rank-uniform
branch and every rank enters the eviction together with the same ``needed``.
What the floor did not make uniform is what happens INSIDE:

  * WHICH nodes ``evict_host`` deletes. Its victims are H-leaves, and
    ``_is_host_leaf`` (unified_radix_cache.py:1653-1665) requires
    ``node.evicted`` -- device-evicted. The device pools are rank-sized
    (190400 / 143840 / 143906 on this boot), so the ranks device-evict at
    different times and each rank's H-leaf set is a different set of nodes.
    ``_evict_host_leaf`` then calls ``self._remove_leaf_from_parent(node)``
    (unified_radix_cache.py:1751), so this is a TREE EDIT, not bookkeeping.
    The rank that deletes an old chunk node matches SHORTER there.
    -> roomy rank LONGER, the 13:26 specimen.

  * WHETHER ``evicted >= needed``. A rank with few H-leaves cannot raise the
    tokens, returns 0, and the node gets no backup. Under ``write_through``
    an un-backed-up node is DELETED at its next device eviction while a
    backed-up one is demoted and stays matchable (the #639 chain). The rank
    that FAILED to evict loses the NEW node instead.
    -> roomy rank SHORTER, the 12:15 specimen.

So one unguarded call produces a rank that keeps old prefixes its peers
dropped AND a rank that drops a new prefix its peers kept, in the same
iteration. Which of the two the detector reports depends only on which node
the next request happens to match. That is why the two specimens point in
opposite directions and are nevertheless the same defect.

WHY THE FLOOR CANNOT SIMPLY BE PUSHED INTO ``evict_host``
---------------------------------------------------------
The floor works for an ADMISSION question because both sides of the compare
can be made replicated. Eviction is a SELECTION question: the candidate set
is ``evictable_host_leaves``, and membership turns on ``node.evicted``, a
rank-local fact about a rank-sized device pool. No arithmetic on a published
scalar can make two ranks pick the same nodes out of two different candidate
sets. Making the selection uniform needs a group-agreed victim list, i.e. a
new collective on the hot path. So the pin here is the other direction: under
an active floor the backup path does not evict at all, and refuses uniformly
instead. See ``test_a_uniform_floor_cannot_make_the_victim_set_uniform``.

Deliberately hermetic: no CUDA, no process group, no model, no pools. The
REAL ``write_backup`` of both cache classes is driven on a fixture that
supplies only the surface those functions touch.
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


from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


#: ``chunked_prefill_size`` on the crashing boot. The 13:26 specimen's
#: rank gap is exactly this minus one (22014 - 19967 = 2047), i.e. one node.
CHUNK = 2048

#: The three DEVICE pools of the crashing boot, in tokens. Rank 0 is the
#: 5090 and the roomiest.
BOOT_DEVICE_POOLS = [190400, 143840, 143906]

#: The H-leaf CANDIDATE SET per rank -- which chunk nodes this rank may
#: delete. Not a free parameter, and the membership matters more than the
#: count: ``_is_host_leaf`` requires ``node.evicted``, so a rank's candidates
#: are exactly the nodes IT has device-evicted. Rank 0 is the 5090 with the
#: largest device pool (190400) and has evicted nothing yet; ranks 1 and 2
#: have device pools within 0.05% of each other (143840 / 143906) and so
#: agree with each other -- which is the shape both specimens show, rank 0
#: apart and ranks 1/2 identical.
BOOT_HOST_LEAF_SETS = [
    [],
    ["old3", "old4", "old5"],
    ["old3", "old4", "old5"],
]

#: Host availability REMAINING late in a write_through run, in tokens. The
#: absolute pools are 359652 / 287722 / 273336, but the defective branch is
#: only reached once the host tier is saturated -- which write_through
#: reaches, because it backs up every insert. These are the remaining-free
#: values in that regime; the floor is their group MIN.
LATE_RUN_HOST_FREE = [3000, 1000, 1000]


class _FakeHostPool:
    """A host pool with real alloc semantics: it can run out."""

    def __init__(self, avail):
        self._avail = avail

    def available_size(self):
        return self._avail

    def alloc(self, n):
        if n > self._avail:
            return None
        self._avail -= n
        # A tensor, because production calls `.clone()` on what comes back.
        return torch.arange(n)

    def free(self, n):
        self._avail += n


class _FakeCacheController:
    def __init__(self, host_avail):
        self.mem_pool_host = _FakeHostPool(host_avail)
        self.write_policy = "write_through"

    def write(self, device_indices=None, node_id=None, extra_pools=None, **kwargs):
        return self.mem_pool_host.alloc(len(device_indices))


class _StubComponent:
    """The base component, reduced to the calls write_backup makes on it."""

    def __init__(self, base_type):
        self.component_type = base_type

    def build_hicache_transfers(self, node, phase, **kwargs):  # pragma: no cover
        return []

    def commit_hicache_transfer(self, node, phase, transfers=None):
        return None


class _NodeData:
    def __init__(self, value):
        self.value = value


class _FakeNode:
    """A chunk node. ``node_key`` identifies it across ranks, so 'did the
    ranks delete the same nodes' is a well-posed question."""

    _counter = 100

    def __init__(self, base_type, kv_tokens, parent, node_key="new"):
        _FakeNode._counter += 1
        self.id = _FakeNode._counter
        self.node_key = node_key
        self.parent = parent
        self.host_value = None
        self.write_through_pending_id = None
        self.component_data = {base_type: _NodeData(list(range(kv_tokens)))}

    @property
    def backuped(self):
        return self.host_value is not None


class _RankFixture:
    """Drives the REAL ``UnifiedRadixCache.write_backup`` for one rank.

    Bound unbound rather than reimplemented, for the reason the #639 suite
    states: a fixture that re-spelled the gate would pass whatever the gate
    said. Reverting the production guard makes the falsifiers below fail
    because this really is that function running.

    ``evict_host`` is modelled, not real -- the real one needs a component
    stack, an LRU and a host pool allocator. What is modelled is only the
    property the defect turns on: it consumes a RANK-LOCAL candidate list and
    removes those nodes from the tree. That the real one really does remove
    them from the tree is pinned separately, by reading its source, in
    ``test_host_eviction_is_still_a_tree_edit``.
    """

    def __init__(self, base_type, host_free, candidates, rank):
        self.rank = rank
        self.cache_controller = _FakeCacheController(host_free)
        self.root_node = object()
        self.uniform_host_avail_floor = None
        self.uniform_host_admitted_since_floor = 0
        self.uniform_avail_floor = None
        self._base_type = base_type
        self.components = {base_type: _StubComponent(base_type)}
        self._components_tuple = ()
        self.ongoing_write_through = {}
        self.tree_components = (base_type,)

        # Rank-local eviction candidates, named by NODE so cross-rank
        # comparison is meaningful. Rank r's list is the nodes r has
        # device-evicted, in r's own priority order.
        self._candidates = list(candidates)
        self.deleted_from_tree = []

    # -- the surface write_backup touches, and nothing else ----------------
    #
    # #773/#581 (703b05c723) added a gate to that surface: `write_backup`
    # asks `self._mamba_write_through_pin_admissible(node, ...)` at
    # unified_radix_cache.py:1921 before taking a write-through pin, and this
    # fixture -- which IS the `self` that `_run_backup` passes in -- did not
    # carry it.
    #
    # The SHIPPED predicate is bound rather than stubbed True, so the real
    # admission rule keeps running here. `_mamba_pin_budget` is pinned to -1,
    # which is not an invented number: it is the value the production property
    # itself computes when there is no mamba pool to protect
    # (unified_radix_cache.py:3426), and the predicate's first budget branch
    # reads `budget < 0 -> admissible`. This fixture has a KV-only tree, so
    # that is the branch a real cache of this shape would take, and the host
    # evict floor under test sees exactly the admissions it saw before #773.
    # Delegated rather than bound as a class attribute, because this file
    # imports the cache inside the functions that need it (see `_run_backup`
    # and `_build_group`) and a class body cannot reach that.
    _mamba_pin_budget = -1

    def _mamba_write_through_pin_admissible(self, node, write_back=False):
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        return UnifiedRadixCache._mamba_write_through_pin_admissible(
            self, node, write_back=write_back
        )

    def _build_sidecar_transfers(self, phase, kv_xfer, comp_xfers):
        return []

    def evict_host(self, num_tokens, component_type=None):
        freed = 0
        while freed < num_tokens and self._candidates:
            victim = self._candidates.pop(0)
            self.deleted_from_tree.append(victim)
            self.cache_controller.mem_pool_host.free(CHUNK)
            freed += CHUNK
        return freed

    def inc_lock_ref(self, node):
        return types.SimpleNamespace(to_dec_params=lambda: None)

    def _track_write_through_node(self, node, lock_params):
        self.ongoing_write_through[node.id] = (node, lock_params, [node])

    def _record_remove_event(self, node, medium=None):
        return None


def _run_backup(fixture, kv_tokens=CHUNK):
    """Run the REAL ``UnifiedRadixCache.write_backup`` on this rank and report
    whether the node was admitted to the host tier."""
    from sglang.srt.mem_cache.unified_cache_components import BASE_COMPONENT_TYPE
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    node = _FakeNode(BASE_COMPONENT_TYPE, kv_tokens, fixture.root_node)
    written = UnifiedRadixCache.write_backup(fixture, node)
    return written > 0


def _build_group(floor_active=True, host_free=None, leaf_sets=None):
    from sglang.srt.mem_cache.unified_cache_components import BASE_COMPONENT_TYPE

    host_free = LATE_RUN_HOST_FREE if host_free is None else host_free
    leaf_sets = BOOT_HOST_LEAF_SETS if leaf_sets is None else leaf_sets
    ranks = [
        _RankFixture(BASE_COMPONENT_TYPE, free, cands, r)
        for r, (free, cands) in enumerate(zip(host_free, leaf_sets))
    ]
    if floor_active:
        # What the scheduler publishes: the group MIN, once per iteration.
        floor = min(host_free)
        for r in ranks:
            r.uniform_host_avail_floor = floor
    return ranks


class _HiRadixFixture:
    """Drives the REAL ``HiRadixCache.write_backup`` for one rank.

    Same discipline as ``_RankFixture``: the production function is bound
    unbound, only the surface it touches is supplied, and ``evict_host`` is
    modelled down to the one property the defect turns on -- it consumes a
    rank-local candidate list and removes those nodes from the tree.
    """

    def __init__(self, host_free, candidates):
        self.cache_controller = _FakeCacheController(host_free)
        self.root_node = object()
        self.uniform_host_avail_floor = None
        self.uniform_host_admitted_since_floor = 0
        self.ongoing_write_through = {}
        self._candidates = list(candidates)
        self.deleted_from_tree = []

    def _get_extra_pools(self):
        return {}

    def evict_host(self, num_tokens):
        freed = 0
        while freed < num_tokens and self._candidates:
            self.deleted_from_tree.append(self._candidates.pop(0))
            self.cache_controller.mem_pool_host.free(CHUNK)
            freed += CHUNK
        return freed

    def inc_lock_ref(self, node):
        return types.SimpleNamespace(delta=0)

    def _track_write_through_node(self, node, backup_len):
        self.ongoing_write_through[node.id] = (node, backup_len, [node])


class _HiRadixNode:
    """The node surface ``HiRadixCache.write_backup`` reads. ``value`` and the
    host indices are torch tensors because the production code calls
    ``.clone()`` on them."""

    def __init__(self, kv_tokens, parent):
        self.id = 7
        self.parent = parent
        self.value = torch.arange(kv_tokens)
        self.key = list(range(kv_tokens))
        self.host_value = None
        self.write_through_pending_id = None

    @property
    def backuped(self):
        return self.host_value is not None


def _build_hiradix_group(floor_active=True, host_free=None, floor=None):
    """``floor`` is set independently of ``host_free`` on purpose: the whole
    point is that the published number is a START-OF-ITERATION snapshot and
    the live pools have moved on under it."""
    host_free = LATE_RUN_HOST_FREE if host_free is None else host_free
    ranks = [_HiRadixFixture(free, ["old3", "old4", "old5"]) for free in host_free]
    if floor_active:
        published = min(host_free) if floor is None else floor
        for r in ranks:
            r.uniform_host_avail_floor = published
    return ranks


def _run_hiradix_backup(fixture, kv_tokens=CHUNK):
    """Run the REAL ``HiRadixCache.write_backup`` and report admission."""
    from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

    node = _HiRadixNode(kv_tokens, fixture.root_node)
    return HiRadixCache.write_backup(fixture, node) > 0


class UniformHostEvictFloorTest(CustomTestCase):
    # -- the falsifiers -----------------------------------------------------

    def test_the_backup_path_deletes_the_same_tree_nodes_on_every_rank(self):
        """THE FALSIFIER, 13:26 direction (roomy rank LONGER).

        Every rank enters ``write_backup`` with the same replicated floor and
        the same node length, so the eviction branch is taken by all three.
        Each then deletes nodes out of its OWN candidate set. At the
        pre-change base ranks 1/2 delete 'old3' -- a node rank 0 has not
        device-evicted and therefore keeps -- so the surviving trees differ
        by one chunk node.

        A rank whose tree still holds a chunk node its peers deleted matches
        exactly one chunk further: 22014 vs 19967, difference 2047.
        """
        ranks = _build_group(floor_active=True)
        for r in ranks:
            _run_backup(r)

        deleted = [tuple(r.deleted_from_tree) for r in ranks]
        self.assertEqual(
            len(set(deleted)),
            1,
            msg=(
                "the write-backup path deleted a DIFFERENT set of tree nodes on "
                f"different ranks: {deleted}. Each deletion is a "
                "_remove_leaf_from_parent, so match_prefix now returns a "
                "rank-dependent prefix -- the 13:26 signature "
                "(rank 0 [22014] vs peers [19967], exactly one 2047-token chunk)."
            ),
        )

    def test_the_backup_verdict_agrees_across_ranks(self):
        """THE FALSIFIER, 12:15 direction (roomy rank SHORTER).

        Same branch, the other observable. ``evicted >= needed`` is decided
        against a rank-local candidate set, so the rank with the ROOMIEST
        DEVICE pool -- which has device-evicted the fewest nodes and so has
        the fewest H-leaves -- is the one that cannot raise the tokens and
        returns 0. Its node gets no host backup and is DELETED at its next
        device eviction while its peers demote theirs and keep it matchable,
        so the roomy rank matches SHORTER: rank 0 [2047] vs peers [10238].
        """
        ranks = _build_group(floor_active=True)
        admitted = [_run_backup(r) for r in ranks]

        self.assertEqual(
            len(set(admitted)),
            1,
            msg=(
                f"the host backup verdict is not rank-uniform: {admitted}. Under "
                "write_through an un-backed-up node is deleted at its next "
                "device eviction and a backed-up one is demoted and stays "
                "matchable, so this is a divergent TREE -- the 12:15 signature."
            ),
        )

    def test_both_specimen_directions_come_out_of_the_same_branch(self):
        """The two production specimens point in OPPOSITE directions. This
        pins that one unguarded call produces both at once, so a fix that
        closed only one of them would be incomplete.

        At the base, in a single iteration: at least one rank deletes an old
        chunk node that another rank keeps (roomy LONGER), AND at least one
        rank is refused a backup that another rank was granted (roomy
        SHORTER).
        """
        ranks = _build_group(floor_active=True)
        admitted = [_run_backup(r) for r in ranks]
        deleted = [set(r.deleted_from_tree) for r in ranks]

        old_node_divergence = any(deleted[0] ^ d for d in deleted[1:])
        verdict_divergence = len(set(admitted)) > 1

        self.assertFalse(
            old_node_divergence or verdict_divergence,
            msg=(
                "one iteration produced BOTH crash directions: "
                f"old-node deletion divergence={old_node_divergence} "
                f"(13:26, roomy LONGER), backup-verdict divergence="
                f"{verdict_divergence} (12:15, roomy SHORTER). deleted={deleted} "
                f"admitted={admitted}"
            ),
        )

    # -- the stale-snapshot half ---------------------------------------------

    def test_the_admission_count_agrees_across_ranks_over_one_iteration(self):
        """The floor is published at the TOP of the iteration and read at the
        END of it, so on its own it counts as free every slot the iteration's
        own earlier backups have already taken.

        Three ranks, floor 6000, chunk nodes of 2048. Charging admissions
        against the floor makes all three stop after the same node. Without
        the charge the floor still reads 6000 on the third insert, the roomy
        rank's pool really does have room and the tight ranks' does not, and
        the verdict splits -- which is the divergence, one iteration deep.
        """
        counts = self._admissions_over_one_iteration()
        self.assertEqual(
            len(set(counts)),
            1,
            msg=(
                f"ranks admitted different numbers of backups in one "
                f"iteration: {counts}. The floor is a start-of-iteration "
                "snapshot; without charging admissions against it the tight "
                "rank exhausts its real pool while its peers read the same "
                "optimistic number."
            ),
        )

    def test_the_ledger_is_what_makes_that_agreement_hold(self):
        """THE CAN-FAIL PROOF for the test above.

        Restoring the pre-#645 arithmetic -- read the published floor, ignore
        what this iteration already spent -- must bring the divergence back.
        If it does not, the test above is passing for some other reason and
        proves nothing about the ledger.

        Note this reaches the split WITHOUT going through the eviction guard:
        the stale floor says there is room, so the path runs straight into a
        write that fails only on the tight ranks. The guard and the ledger
        close two different doors.
        """

        def stale(tree, pool):
            """The pre-#645 arithmetic: the published floor, uncharged."""
            return int(tree.uniform_host_avail_floor)

        with mock.patch(
            "sglang.srt.mem_cache.unified_radix_cache.uniform_host_avail_for_backup",
            stale,
        ):
            counts = self._admissions_over_one_iteration()

        self.assertGreater(
            len(set(counts)),
            1,
            msg=(
                "with the ledger disabled the admission counts still agreed "
                f"({counts}); the falsifier above cannot fail and does not "
                "pin the ledger"
            ),
        )

    def _admissions_over_one_iteration(self):
        """Run one iteration's worth of inserts on each rank and report how
        many backups each rank admitted."""
        ranks = _build_group(
            floor_active=True,
            host_free=[12000, 6000, 6000],
            leaf_sets=[[], [], []],
        )
        counts = []
        for r in ranks:
            admitted = 0
            for _ in range(3):
                if _run_backup(r):
                    admitted += 1
            counts.append(admitted)
        return counts

    # -- behaviour neutrality ------------------------------------------------

    def test_no_floor_leaves_the_eviction_retry_exactly_as_it_was(self):
        """The guard may only bind when the scheduler published a floor.

        With ``uniform_host_avail_floor is None`` -- single rank, host pools
        that agree, or no host tier -- the path must still read the LOCAL
        availability, still call ``evict_host``, and still admit the node.
        This is the byte-identical case and it is the one the reference boot
        runs on an even rig.
        """
        (solo,) = _build_group(
            floor_active=False,
            host_free=[1000],
            leaf_sets=[["old3", "old4", "old5"]],
        )
        self.assertIsNone(solo.uniform_host_avail_floor)

        admitted = _run_backup(solo)

        self.assertTrue(
            admitted,
            msg="without a floor the backup must still succeed via evict_host",
        )
        self.assertEqual(
            solo.deleted_from_tree,
            ["old3"],
            msg=(
                "without a floor the eviction retry must run exactly as before; "
                f"got {solo.deleted_from_tree}"
            ),
        )

    # -- the sibling class, driven for real ----------------------------------

    def test_hiradix_makes_no_rank_local_host_eviction_under_a_floor(self):
        """The #639 suite pinned ``HiRadixCache`` by SOURCE TEXT only, on the
        stated grounds that it "cannot be driven hermetically without standing
        up a cache controller, a host pool, transfer plans and lock refs".
        It can; this is that fixture. The class matters because #616g's
        load-back fix spent a boot sitting in the class deployment never
        built, and a source-text pin cannot tell a guard that binds from one
        that is merely present.

        Its retry differs from the sibling's: it is reached on a RANK-LOCAL
        condition (this rank's ``cache_controller.write`` returned None), so
        the rank that takes it evicts and deletes tree nodes while its peers,
        whose write succeeded, do not.

        The scenario is a floor that is stale by MORE than the ledger can
        account for -- published at 2048 while the tight ranks' live pools
        hold 100 -- because that is the only way to reach this retry at all
        once admissions are charged. What is asserted is what the fix
        guarantees: no rank edits its tree. See
        ``test_the_refusal_residual_is_named_not_hidden`` for what is
        deliberately NOT claimed here.
        """
        ranks = _build_hiradix_group(
            floor_active=True, host_free=[12000, 100, 100], floor=2048
        )
        for r in ranks:
            _run_hiradix_backup(r)
        deleted = [tuple(r.deleted_from_tree) for r in ranks]

        self.assertEqual(
            len(set(deleted)),
            1,
            msg=(
                f"HiRadixCache.write_backup evicted rank-locally: {deleted}. "
                "evict_host pops parent.children, so the ranks' trees and "
                "their match_prefix results now differ."
            ),
        )
        self.assertEqual(
            deleted[0],
            (),
            msg="under an active floor no rank may delete a tree node here",
        )

    def test_the_refusal_residual_is_named_not_hidden(self):
        """WHAT THIS FIX DOES NOT CLOSE, pinned so it cannot be forgotten.

        When the write fails under an active floor -- i.e. the floor was
        stale by more than this gate's own admissions, which takes an
        allocation from OUTSIDE the gate -- the refusing rank returns 0 while
        a rank whose write succeeded returns >0. The backup VERDICT still
        diverges there, and under write_through a divergent verdict is a
        divergent tree one device-eviction later (the #639 chain).

        The fix converts an immediate, silent, structural tree edit into a
        deferred and LOGGED one-node divergence. That is strictly better, not
        complete. Closing it needs the backup verdict itself to be
        group-agreed, which needs a collective this path does not have.

        This test documents the residual by asserting it is still present;
        if a later change makes the verdict uniform, this test fails and the
        follow-up ticket can be closed with it.
        """
        ranks = _build_hiradix_group(
            floor_active=True, host_free=[12000, 100, 100], floor=2048
        )
        admitted = [_run_hiradix_backup(r) for r in ranks]

        self.assertEqual(
            admitted,
            [True, False, False],
            msg=(
                "the known residual changed shape; re-read the follow-up "
                f"ticket before adjusting this expectation. got {admitted}"
            ),
        )

    def test_hiradix_without_a_floor_keeps_its_eviction_retry(self):
        """Behaviour neutrality for the sibling class: no floor, no change."""
        (solo,) = _build_hiradix_group(floor_active=False, host_free=[100])

        admitted = _run_hiradix_backup(solo)

        self.assertTrue(
            admitted,
            msg="without a floor HiRadixCache must still retry via evict_host",
        )
        self.assertEqual(
            solo.deleted_from_tree,
            ["old3"],
            msg=(
                "without a floor the retry must run exactly as before; got "
                f"{solo.deleted_from_tree}"
            ),
        )

    def test_an_active_floor_that_fits_still_admits_without_evicting(self):
        """The guard must not turn into a blanket refusal. When the published
        floor covers the node, every rank admits it and nobody evicts."""
        ranks = _build_group(floor_active=True, host_free=[9000, 9000, 9000])
        # Uneven pools are what activates the floor; force it on explicitly
        # so this case is about the floor FITTING, not about it being absent.
        for r in ranks:
            r.uniform_host_avail_floor = 9000

        admitted = [_run_backup(r) for r in ranks]

        self.assertEqual(admitted, [True, True, True])
        for r in ranks:
            self.assertEqual(
                r.deleted_from_tree,
                [],
                msg="a fitting floor must not trigger any host eviction",
            )

    # -- deletion falsifiers / mechanism pins --------------------------------

    def test_host_eviction_is_still_a_tree_edit(self):
        """The link that makes the modelled ``evict_host`` faithful. If
        ``_evict_host_leaf`` stops removing the node from its parent, the
        chain this pin closes no longer exists and the justification would
        have to be rewritten rather than silently kept."""
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        src = inspect.getsource(UnifiedRadixCache._evict_host_leaf)
        self.assertIn(
            "_remove_leaf_from_parent",
            src,
            msg="host eviction no longer edits the tree; #645's premise is gone",
        )

    def test_a_uniform_floor_cannot_make_the_victim_set_uniform(self):
        """Why the fix refuses instead of evicting under a floor.

        ``_is_host_leaf`` gates candidacy on ``node.evicted`` -- a rank-local
        fact about a rank-sized device pool. A published scalar cannot make
        two ranks select the same nodes out of two different candidate sets,
        so 'push the floor into evict_host' is not an available fix.
        """
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        src = inspect.getsource(UnifiedRadixCache._is_host_leaf)
        self.assertIn(
            "node.evicted",
            src,
            msg=(
                "host-leaf candidacy no longer depends on rank-local device "
                "residency; the refuse-instead-of-evict choice would need "
                "revisiting"
            ),
        )

    def test_both_cache_classes_guard_the_eviction_from_the_backup_path(self):
        """Both, because #616g's load-back fix was first written into the
        class this rig does not instantiate and half of it sat dead for a
        boot. The 12:15 and 13:26 boots both report
        ``impl=UnifiedRadixCache``; ``HiRadixCache`` carries the same defect
        at its own ``evict_host`` retry and is fixed with it.
        """
        from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        for func, name in (
            (UnifiedRadixCache.write_backup, "UnifiedRadixCache.write_backup"),
            (HiRadixCache.write_backup, "HiRadixCache.write_backup"),
        ):
            src = inspect.getsource(func)
            self.assertIn(
                "uniform_host_floor_active",
                src,
                msg=(
                    f"{name} no longer guards its host eviction against the "
                    "rank-uniform floor (#645)"
                ),
            )
            guard_at = src.index("uniform_host_floor_active")
            evict_at = src.find("self.evict_host(")
            self.assertNotEqual(
                evict_at, -1, msg=f"{name} lost its evict_host call entirely"
            )
            self.assertLess(
                guard_at,
                evict_at,
                msg=f"{name} evicts before consulting the #645 guard",
            )


if __name__ == "__main__":
    unittest.main()
