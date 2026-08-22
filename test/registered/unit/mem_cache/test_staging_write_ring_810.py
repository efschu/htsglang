"""#810: write-through into a STAGING host tier is bounded, counted, and never
re-inflates the pinned footprint the role exists to shrink.

WHAT THIS PINS, and why it is not the #720 shape
------------------------------------------------
``ReadBufferPool.acquire`` (``read_buffer_pool.py:99-105``) answers exhaustion
by allocating a fresh UNCOUNTED pinned buffer, on purpose: for a prefetch
worker "stalling ... to save memory would trade a bounded spike for an
unbounded latency". Reused verbatim on the write path that choice re-inflates,
page by page, exactly the pinned bytes ``--hicache-host-role staging`` was set
to remove. So the falsifier here is not "does it bound" but "does it bound
WITHOUT allocating": :class:`StagingWriteRing` has no factory, no allocation
call, and one refusal outcome.

The other half is the production edge. A ring that only its own unit test
calls proves nothing about ``write_backup``, so both real implementations --
``HiRadixCache.write_backup`` and ``UnifiedRadixCache.write_backup``, the only
two with a construction site (``registry.py:115`` / ``registry.py:191``;
``HiMambaRadixCache`` has none) -- are driven unbound over a fixture, in both
directions: with a ring that has room (admits, exactly as before) and with a
ring that is full (refuses, and does NOT reach the rank-local host eviction).

Deliberately hermetic: no CUDA, no process group, no model, no pools.
"""

import types
import unittest

import torch

try:
    from sglang.test.ci.ci_register import register_cpu_ci
except ImportError:  # pragma: no cover - registration is a CI-time marker

    def register_cpu_ci(*args, **kwargs):
        return None


from sglang.srt.mem_cache.staging_write_ring import (
    StagingWriteRing,
    build_staging_write_ring,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


#: One chunk of the standing boot, in tokens.
CHUNK = 2048


class StagingWriteRingUnitTest(CustomTestCase):
    """The bound itself."""

    def test_a_full_ring_refuses_instead_of_allocating(self):
        """THE FALSIFIER against the #720 shape.

        #720 answers exhaustion with a fresh uncounted buffer. If this ring
        did that, ``admit`` would keep returning True past its capacity and
        the pinned footprint would grow without bound -- the precise failure
        ``--hicache-host-role staging`` exists to prevent.
        """
        ring = StagingWriteRing(capacity_tokens=2 * CHUNK)
        self.assertTrue(ring.admit("a", CHUNK))
        self.assertTrue(ring.admit("b", CHUNK))
        self.assertFalse(
            ring.admit("c", CHUNK),
            "a full staging ring admitted a third page: exhaustion re-inflated "
            "the pinned footprint instead of applying backpressure",
        )
        self.assertEqual(ring.occupied_tokens, 2 * CHUNK)
        self.assertEqual(ring.refused, 1)
        self.assertEqual(ring.refused_tokens, CHUNK)

    def test_the_other_direction_a_ring_with_room_admits(self):
        """A bound that refuses everything is not a bound, it is an outage."""
        ring = StagingWriteRing(capacity_tokens=2 * CHUNK)
        self.assertTrue(ring.admit("a", CHUNK))
        self.assertEqual(ring.refused, 0)
        self.assertEqual(ring.admitted, 1)
        self.assertEqual(ring.available_tokens, CHUNK)

    def test_the_refusal_is_counted_not_silent(self):
        """Today's exhaustion is a ``None`` read back out of a failed alloc and
        nothing else. A slowdown nobody can see is the reason #810 exists."""
        ring = StagingWriteRing(capacity_tokens=CHUNK)
        ring.admit("a", CHUNK)
        for _ in range(3):
            ring.admit("b", CHUNK)
        stats = ring.stats()
        self.assertEqual(stats["refused"], 3)
        self.assertEqual(stats["refused_tokens"], 3 * CHUNK)
        self.assertEqual(stats["occupied_tokens"], CHUNK)
        self.assertEqual(stats["peak_occupied_tokens"], CHUNK)

    def test_a_drained_page_gives_its_room_back(self):
        ring = StagingWriteRing(capacity_tokens=CHUNK)
        self.assertTrue(ring.admit("a", CHUNK))
        self.assertFalse(ring.admit("b", CHUNK))
        ring.release("a")
        self.assertEqual(ring.occupied_tokens, 0)
        self.assertTrue(
            ring.admit("b", CHUNK),
            "the ring stayed full after its only page drained: backpressure "
            "that never lifts is a stalled write-through, not a bound",
        )
        self.assertEqual(ring.released, 1)

    def test_releasing_twice_cannot_inflate_the_ring(self):
        """Occupancy is the sum of a dict, not a free-running counter: the
        release edges are the same ones that retire ``ongoing_backup``,
        including a forced release on detach, and a double subtraction there
        would let the ring admit more than its capacity forever."""
        ring = StagingWriteRing(capacity_tokens=2 * CHUNK)
        ring.admit("a", CHUNK)
        ring.release("a")
        ring.release("a")
        ring.release("never-admitted")
        self.assertEqual(ring.occupied_tokens, 0)
        ring.admit("b", 2 * CHUNK)
        self.assertFalse(ring.admit("c", 1))

    def test_readmitting_a_live_page_does_not_charge_twice(self):
        """A node offered again by a later insert before its first backup
        drained must not inflate the occupancy of a page already counted."""
        ring = StagingWriteRing(capacity_tokens=2 * CHUNK)
        ring.admit("a", CHUNK)
        ring.admit("a", CHUNK)
        self.assertEqual(ring.occupied_tokens, CHUNK)
        self.assertEqual(ring.readmitted, 1)

    def test_an_aborted_admission_is_given_back(self):
        """``write_backup`` can still fail AFTER the ring admitted it (the host
        allocation is the next statement). A charge left standing there leaks."""
        ring = StagingWriteRing(capacity_tokens=CHUNK)
        ring.admit("a", CHUNK)
        ring.abort("a")
        self.assertEqual(ring.occupied_tokens, 0)
        self.assertEqual(ring.aborted, 1)
        self.assertTrue(ring.admit("b", CHUNK))

    def test_the_drain_phase_is_charged_and_cannot_be_refused(self):
        """The page is already resident when the storage hand-off happens, so
        refusing there would free nothing and only hide the drain queue from
        the next admission."""
        ring = StagingWriteRing(capacity_tokens=CHUNK)
        ring.occupy("op-1", CHUNK)
        ring.occupy("op-2", CHUNK)
        self.assertEqual(ring.occupied_tokens, 2 * CHUNK)
        self.assertFalse(
            ring.admit("next", 1),
            "an admission was granted while the drain queue already exceeded "
            "the ring: the drain window is the quantity the tier is sized from",
        )
        ring.release("op-1")
        ring.release("op-2")
        self.assertEqual(ring.occupied_tokens, 0)

    def test_a_ring_that_can_admit_nothing_is_refused_at_construction(self):
        with self.assertRaises(ValueError):
            StagingWriteRing(capacity_tokens=0)


class StagingRingConstructionTest(CustomTestCase):
    """Who gets a ring, and how big."""

    @staticmethod
    def _controller(size=100_000, prefetch_limit=50_000):
        return types.SimpleNamespace(
            mem_pool_host=types.SimpleNamespace(size=size),
            prefetch_capacity_limit=prefetch_limit,
        )

    def test_the_default_role_gets_no_ring_at_all(self):
        """The whole backward-compatibility claim: under ``retention`` -- the
        default -- every call site is a single ``is None`` test."""
        args = types.SimpleNamespace(hicache_host_role="retention")
        self.assertIsNone(build_staging_write_ring(args, self._controller()))

    def test_a_server_args_without_the_flag_gets_no_ring(self):
        self.assertIsNone(
            build_staging_write_ring(types.SimpleNamespace(), self._controller())
        )

    def test_the_staging_role_gets_a_ring(self):
        args = types.SimpleNamespace(hicache_host_role="staging")
        ring = build_staging_write_ring(args, self._controller())
        self.assertIsNotNone(ring)

    def test_the_capacity_is_the_complement_of_the_prefetch_reservation(self):
        """Not a new number. The read consumer is already bounded at runtime by
        ``prefetch_capacity_limit``; the two consumers share one tier, so the
        write bound is that number's complement and this module introduces no
        constant of its own."""
        args = types.SimpleNamespace(hicache_host_role="staging")
        ring = build_staging_write_ring(
            args, self._controller(size=100_000, prefetch_limit=50_000)
        )
        self.assertEqual(ring.capacity_tokens, 50_000)
        ring = build_staging_write_ring(
            args, self._controller(size=80_000, prefetch_limit=30_000)
        )
        self.assertEqual(ring.capacity_tokens, 50_000)

    def test_no_ring_when_the_reservation_already_claims_the_tier(self):
        """A ring of zero would stop write-through rather than throttle it."""
        args = types.SimpleNamespace(hicache_host_role="staging")
        self.assertIsNone(
            build_staging_write_ring(
                args, self._controller(size=50_000, prefetch_limit=50_000)
            )
        )

    def test_no_controller_no_ring(self):
        args = types.SimpleNamespace(hicache_host_role="staging")
        self.assertIsNone(build_staging_write_ring(args, None))
        self.assertIsNone(
            build_staging_write_ring(args, types.SimpleNamespace(mem_pool_host=None))
        )


# --------------------------------------------------------------------------
# The production edge. Both real ``write_backup`` implementations, driven
# unbound over a fixture that supplies only the surface they touch.
# --------------------------------------------------------------------------


class _FakeMemPoolHost:
    def __init__(self, free_tokens):
        self._free = free_tokens
        self.size = 100_000

    def available_size(self):
        return self._free

    def free(self, n):
        self._free += n


class _FakeCacheController:
    def __init__(self, free_tokens, fail_write=False):
        self.mem_pool_host = _FakeMemPoolHost(free_tokens)
        self.writes = 0
        self.fail_write = fail_write
        self.prefetch_capacity_limit = 50_000

    def write(self, *args, **kwargs):
        self.writes += 1
        if self.fail_write or self.mem_pool_host.available_size() < CHUNK:
            return None
        self.mem_pool_host._free -= CHUNK
        return torch.arange(CHUNK)


class _HiRadixNode:
    def __init__(self, node_id, kv_tokens, parent):
        self.id = node_id
        self.parent = parent
        self.value = torch.arange(kv_tokens)
        self.key = list(range(kv_tokens))
        self.host_value = None
        self.write_through_pending_id = None

    @property
    def backuped(self):
        return self.host_value is not None


class _HiRadixFixture:
    """The surface ``HiRadixCache.write_backup`` touches, and nothing else."""

    def __init__(self, ring, free_tokens=100_000, fail_write=False):
        self.cache_controller = _FakeCacheController(free_tokens, fail_write)
        self.root_node = object()
        self.uniform_host_avail_floor = None
        self.uniform_host_admitted_since_floor = 0
        self.ongoing_write_through = {}
        self.staging_write_ring = ring
        self.evictions = 0
        self.locks = 0

    def _get_extra_pools(self):
        return {}

    def evict_host(self, num_tokens):
        self.evictions += 1
        return 0

    def inc_lock_ref(self, node):
        self.locks += 1
        return types.SimpleNamespace(delta=0)

    def _track_write_through_node(self, node, backup_len):
        self.ongoing_write_through[node.id] = (node, backup_len, [node])


def _run_hiradix_backup(fixture, node_id=7, kv_tokens=CHUNK):
    from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

    node = _HiRadixNode(node_id, kv_tokens, fixture.root_node)
    return HiRadixCache.write_backup(fixture, node)


class _UnifiedNode:
    def __init__(self, node_id, kv_tokens, parent):
        self.id = node_id
        self.parent = parent
        self.component_data = {}
        self.key = list(range(kv_tokens))
        self.write_through_pending_id = None
        self.backuped = False
        self._kv_tokens = kv_tokens


class _UnifiedFixture:
    def __init__(self, ring, free_tokens=100_000, fail_write=False):
        self.cache_controller = _FakeCacheController(free_tokens, fail_write)
        self.root_node = object()
        self.uniform_host_avail_floor = None
        self.uniform_host_admitted_since_floor = 0
        self.ongoing_write_through = {}
        self.staging_write_ring = ring
        self.evictions = 0
        self._components_tuple = ()
        from sglang.srt.mem_cache.unified_cache_components import BASE_COMPONENT_TYPE

        self.components = {
            BASE_COMPONENT_TYPE: types.SimpleNamespace(
                commit_hicache_transfer=lambda *a, **k: None
            )
        }

    def _mamba_write_through_pin_admissible(self, node, write_back=False):
        return True

    def _note_mamba_pin_skipped(self):
        pass

    def _build_sidecar_transfers(self, phase, kv_xfer, comp_xfers):
        return []

    def evict_host(self, num_tokens):
        self.evictions += 1
        return 0

    def inc_lock_ref(self, node):
        return types.SimpleNamespace(to_dec_params=lambda: None)

    def _track_write_through_node(self, node, lock_params):
        self.ongoing_write_through[node.id] = (node, lock_params, [node])


def _run_unified_backup(fixture, node_id=7, kv_tokens=CHUNK):
    from sglang.srt.mem_cache.unified_cache_components import BASE_COMPONENT_TYPE
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    node = _UnifiedNode(node_id, kv_tokens, fixture.root_node)
    node.component_data[BASE_COMPONENT_TYPE] = types.SimpleNamespace(
        value=torch.arange(kv_tokens),
        host_value=None,
    )
    return UnifiedRadixCache.write_backup(fixture, node)


class WriteBackupIsBoundedTest(CustomTestCase):
    """The gate as the production functions actually reach it."""

    def test_hiradix_backup_is_refused_when_the_ring_is_full(self):
        ring = StagingWriteRing(capacity_tokens=CHUNK)
        fixture = _HiRadixFixture(ring)
        self.assertGreater(_run_hiradix_backup(fixture, node_id=1), 0)
        self.assertEqual(
            _run_hiradix_backup(fixture, node_id=2),
            0,
            "HiRadixCache.write_backup admitted a second page into a full "
            "staging ring: the runtime bound is not on the production path",
        )
        self.assertEqual(ring.refused, 1)

    def test_hiradix_backup_is_admitted_when_the_ring_has_room(self):
        """The other direction: the gate must not refuse what fits."""
        ring = StagingWriteRing(capacity_tokens=4 * CHUNK)
        fixture = _HiRadixFixture(ring)
        for node_id in (1, 2, 3):
            self.assertGreater(_run_hiradix_backup(fixture, node_id=node_id), 0)
        self.assertEqual(ring.refused, 0)
        self.assertEqual(ring.occupied_tokens, 3 * CHUNK)

    def test_hiradix_without_a_ring_behaves_exactly_as_before(self):
        """The default role. Three backups, three writes, no gate."""
        fixture = _HiRadixFixture(None)
        for node_id in (1, 2, 3):
            self.assertGreater(_run_hiradix_backup(fixture, node_id=node_id), 0)
        self.assertEqual(fixture.cache_controller.writes, 3)

    def test_a_refused_backup_never_reaches_the_rank_local_host_eviction(self):
        """The #645 path. Today exhaustion is learned from a failed allocation
        and answered with ``evict_host`` -- a rank-local TREE EDIT that
        diverges the radix replicas. The ring refuses before the allocation, so
        that branch is not reached at all."""
        ring = StagingWriteRing(capacity_tokens=CHUNK)
        fixture = _HiRadixFixture(ring)
        _run_hiradix_backup(fixture, node_id=1)
        _run_hiradix_backup(fixture, node_id=2)
        self.assertEqual(
            fixture.evictions,
            0,
            "a ring refusal still reached evict_host: the bound was taken "
            "after the allocation failed rather than before it happened",
        )

    def test_hiradix_never_allocates_past_the_ring(self):
        """The #720 falsifier at the production edge: a refusal must not turn
        into a host write at all."""
        ring = StagingWriteRing(capacity_tokens=CHUNK)
        fixture = _HiRadixFixture(ring)
        _run_hiradix_backup(fixture, node_id=1)
        writes_after_first = fixture.cache_controller.writes
        _run_hiradix_backup(fixture, node_id=2)
        self.assertEqual(
            fixture.cache_controller.writes,
            writes_after_first,
            "the refused backup still called cache_controller.write",
        )

    def test_hiradix_gives_the_admission_back_when_the_write_fails(self):
        """Admitted, then the host allocation failed anyway. A charge left
        standing here shrinks the ring for the rest of the process's life."""
        ring = StagingWriteRing(capacity_tokens=4 * CHUNK)
        fixture = _HiRadixFixture(ring, free_tokens=0)
        fixture.uniform_host_avail_floor = None
        self.assertEqual(_run_hiradix_backup(fixture, node_id=1), 0)
        self.assertEqual(
            ring.occupied_tokens,
            0,
            "a failed write left its ring admission charged",
        )
        self.assertEqual(ring.aborted, 1)

    def test_unified_backup_is_refused_when_the_ring_is_full(self):
        ring = StagingWriteRing(capacity_tokens=CHUNK)
        fixture = _UnifiedFixture(ring)
        self.assertGreater(_run_unified_backup(fixture, node_id=1), 0)
        self.assertEqual(
            _run_unified_backup(fixture, node_id=2),
            0,
            "UnifiedRadixCache.write_backup admitted a second page into a full "
            "staging ring: the runtime bound is not on that production path",
        )
        self.assertEqual(ring.refused, 1)

    def test_unified_backup_is_admitted_when_the_ring_has_room(self):
        ring = StagingWriteRing(capacity_tokens=4 * CHUNK)
        fixture = _UnifiedFixture(ring)
        for node_id in (1, 2, 3):
            self.assertGreater(_run_unified_backup(fixture, node_id=node_id), 0)
        self.assertEqual(ring.refused, 0)

    def test_unified_without_a_ring_behaves_exactly_as_before(self):
        fixture = _UnifiedFixture(None)
        for node_id in (1, 2, 3):
            self.assertGreater(_run_unified_backup(fixture, node_id=node_id), 0)
        self.assertEqual(fixture.cache_controller.writes, 3)

    def test_unified_gives_the_admission_back_when_the_write_fails(self):
        """The tier reported room -- so the pre-eviction check above passed and
        the ring admitted -- and the write failed anyway. That is the window in
        which a charge can be stranded."""
        ring = StagingWriteRing(capacity_tokens=4 * CHUNK)
        fixture = _UnifiedFixture(ring, fail_write=True)
        self.assertEqual(_run_unified_backup(fixture, node_id=1), 0)
        self.assertEqual(ring.occupied_tokens, 0)
        self.assertEqual(ring.aborted, 1)


class _StorageFixture(_HiRadixFixture):
    """Adds the surface the DRAIN-phase edges touch to the backup fixture."""

    def __init__(self, ring):
        super().__init__(ring)
        self.ongoing_backup = {}
        self.enable_storage = True
        self.enable_storage_metrics = False
        self.hicache_storage_pass_prefix_keys = False
        self.storage_events = []
        self.next_operation_id = 100
        self.cache_controller.write_storage = self._write_storage

    def _write_storage(self, *args, **kwargs):
        self.next_operation_id += 1
        return self.next_operation_id

    def _record_store_event(self, node, medium=None):
        self.storage_events.append(node.id)

    def dec_lock_ref(self, node):
        pass

    def write_backup_storage(self, node, backup_len=None):
        """The REAL hand-off, so the phase hand-over is exercised end to end."""
        from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

        return HiRadixCache.write_backup_storage(self, node, backup_len)


class _StorageNode(_HiRadixNode):
    def __init__(self, node_id, kv_tokens, parent):
        super().__init__(node_id, kv_tokens, parent)
        self.hash_value = None
        self.host_value = torch.arange(kv_tokens)
        self.protected = 0

    def protect_host(self):
        self.protected += 1

    def release_host(self):
        self.protected -= 1


class DrainPhaseIsChargedTest(CustomTestCase):
    """The residency the tier is actually sized from.

    ``planner/hicache_staging.write_staging_bytes`` sizes the tier from the
    DRAIN window -- the page is pinned from the storage hand-off until the
    backup acks. A ring that only spanned the device->host copy would bound a
    window one or two orders of magnitude shorter than the one that fills the
    tier, i.e. it would report a bound it does not have.
    """

    def test_the_storage_handoff_charges_the_ring(self):
        from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

        ring = StagingWriteRing(capacity_tokens=4 * CHUNK)
        fixture = _StorageFixture(ring)
        node = _StorageNode(1, CHUNK, fixture.root_node)
        HiRadixCache.write_backup_storage(fixture, node, CHUNK)
        self.assertEqual(
            ring.occupied_tokens,
            CHUNK,
            "the page was handed to the storage backend and protected on the "
            "host without being charged: the drain queue is invisible to the "
            "next admission",
        )
        self.assertEqual(ring.occupied, 1)
        self.assertEqual(node.protected, 1)

    def test_the_backup_ack_gives_the_room_back(self):
        from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

        ring = StagingWriteRing(capacity_tokens=4 * CHUNK)
        fixture = _StorageFixture(ring)
        node = _StorageNode(1, CHUNK, fixture.root_node)
        HiRadixCache.write_backup_storage(fixture, node, CHUNK)
        (operation_id,) = fixture.ongoing_backup.keys()
        fixture.cache_controller.ack_backup_queue = _AckQueue(
            [types.SimpleNamespace(id=operation_id, completed_tokens=CHUNK)]
        )
        fixture.cache_controller.prefetch_revoke_queue = _AckQueue([])
        fixture.cache_controller.host_mem_release_queue = _AckQueue([])
        HiRadixCache._drain_storage_control_queues_impl(
            fixture, n_revoke=None, n_backup=None, n_release=None, log_metrics=False
        )
        self.assertEqual(
            ring.occupied_tokens,
            0,
            "the drained page kept its charge: the ring shrinks permanently "
            "and write-through stalls",
        )

    def test_the_two_phases_do_not_double_count_one_page(self):
        """The admitted charge is retired at the device->host ack, BEFORE the
        storage hand-off takes its own. Overlapping them would halve the ring."""
        from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

        ring = StagingWriteRing(capacity_tokens=4 * CHUNK)
        fixture = _StorageFixture(ring)
        node = _StorageNode(1, CHUNK, fixture.root_node)
        ring.admit(node.id, CHUNK)
        fixture.ongoing_write_through[node.id] = (node, CHUNK, [node])
        node.write_through_pending_id = node.id
        HiRadixCache._finish_write_through_ack(fixture, node.id, release_lock=False)
        self.assertEqual(
            ring.occupied_tokens,
            CHUNK,
            "one page occupied two charges across the phase hand-over",
        )

    def test_a_forced_release_on_detach_gives_the_room_back(self):
        """The other exit from ``ongoing_backup``. A charge skipped here would
        shrink the ring for the rest of the process's life -- the failure mode
        that is strictly worse than the overshoot it protects against."""
        from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

        ring = StagingWriteRing(capacity_tokens=4 * CHUNK)
        fixture = _StorageFixture(ring)
        node = _StorageNode(1, CHUNK, fixture.root_node)
        HiRadixCache.write_backup_storage(fixture, node, CHUNK)
        fixture.ongoing_prefetch = {}
        HiRadixCache._force_release_pending_storage_ops(fixture)
        self.assertEqual(ring.occupied_tokens, 0)
        self.assertEqual(fixture.ongoing_backup, {})

    def test_the_floor_refusal_gives_the_admission_back(self):
        """The #645 branch: an active rank-uniform floor answers a failed write
        by refusing rather than evicting. It is still a path out of
        ``write_backup`` after the ring admitted."""
        ring = StagingWriteRing(capacity_tokens=4 * CHUNK)
        fixture = _HiRadixFixture(ring, fail_write=True)
        fixture.uniform_host_avail_floor = 100_000
        self.assertEqual(_run_hiradix_backup(fixture, node_id=1), 0)
        self.assertEqual(
            ring.occupied_tokens,
            0,
            "the floor refusal left the admission charged",
        )
        self.assertEqual(fixture.evictions, 0)


class UnifiedDrainPhaseTest(CustomTestCase):
    """The same two drain edges on the other production class."""

    def _fixture(self, ring):
        from sglang.srt.mem_cache.unified_cache_components import BASE_COMPONENT_TYPE

        fixture = _UnifiedFixture(ring)
        fixture.ongoing_backup = {}
        fixture.enable_storage = True
        fixture.enable_storage_metrics = False
        fixture.storage_metrics_collector = None
        fixture.hicache_storage_pass_prefix_keys = False
        fixture.is_eagle = False
        fixture.next_operation_id = 200
        fixture.inc_host_lock_ref = lambda node: types.SimpleNamespace(
            to_dec_params=lambda: None
        )
        fixture.dec_host_lock_ref = lambda node, params: None

        def _write_storage(*args, **kwargs):
            fixture.next_operation_id += 1
            return fixture.next_operation_id

        fixture.cache_controller.write_storage = _write_storage
        fixture.cache_controller._dcp_owner_ctx = lambda: None
        fixture._record_store_event = lambda node, medium=None: None
        fixture.dec_lock_ref = lambda node, params: None

        def _write_backup_storage(node):
            from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

            return UnifiedRadixCache.write_backup_storage(fixture, node)

        fixture.write_backup_storage = _write_backup_storage
        node = _UnifiedNode(1, CHUNK, fixture.root_node)
        node.backuped = True
        node.component_data[BASE_COMPONENT_TYPE] = types.SimpleNamespace(
            value=torch.arange(CHUNK),
            host_value=torch.arange(CHUNK),
        )
        node.hash_value = None
        node.key = types.SimpleNamespace(token_ids=list(range(CHUNK)))
        return fixture, node

    def test_the_storage_handoff_charges_the_ring(self):
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        ring = StagingWriteRing(capacity_tokens=4 * CHUNK)
        fixture, node = self._fixture(ring)
        UnifiedRadixCache.write_backup_storage(fixture, node)
        self.assertEqual(ring.occupied_tokens, CHUNK)
        self.assertEqual(ring.occupied, 1)

    def test_the_backup_ack_gives_the_room_back(self):
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        ring = StagingWriteRing(capacity_tokens=4 * CHUNK)
        fixture, node = self._fixture(ring)
        UnifiedRadixCache.write_backup_storage(fixture, node)
        (operation_id,) = fixture.ongoing_backup.keys()
        cc = fixture.cache_controller
        cc.ack_backup_queue = _AckQueue(
            [types.SimpleNamespace(id=operation_id, completed_tokens=CHUNK)]
        )
        cc.prefetch_revoke_queue = _AckQueue([])
        cc.host_mem_release_queue = _AckQueue([])
        UnifiedRadixCache._drain_storage_control_queues_impl(
            fixture,
            n_revoke=None,
            n_backup=None,
            n_release=None,
            extra_release_counts=None,
            log_metrics=False,
        )
        self.assertEqual(ring.occupied_tokens, 0)

    def test_the_two_phases_do_not_double_count_one_page(self):
        """The admitted charge is retired at the device->host ack, BEFORE the
        storage hand-off takes its own. Overlapping them would halve the ring,
        and leaving the admitted one standing would leak it: after a node
        split the storage backups are keyed by OPERATION, so nothing downstream
        ever retires a node-keyed charge."""
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        ring = StagingWriteRing(capacity_tokens=4 * CHUNK)
        fixture, node = self._fixture(ring)
        ring.admit(node.id, CHUNK)
        node.write_through_pending_id = node.id
        fixture.ongoing_write_through[node.id] = (node, None, [node])
        UnifiedRadixCache._finish_write_through_ack(fixture, node.id)
        self.assertEqual(
            ring.occupied_tokens,
            CHUNK,
            "one page occupied two charges across the phase hand-over",
        )


class _AckQueue:
    """The `get_nowait`/`Empty` surface the drain helper consumes."""

    def __init__(self, items):
        self._items = list(items)

    def get_nowait(self):
        from queue import Empty

        if not self._items:
            raise Empty
        return self._items.pop(0)


if __name__ == "__main__":
    unittest.main()
