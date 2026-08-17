"""Hermetic (CPU-only) tests for the #581 mamba pin trace on the class
production actually runs: UnifiedRadixCache with the MAMBA component.

Boot 26 logged

    Tree cache initialized: source=default impl=UnifiedRadixCache
      hybrid_swa=False hybrid_ssm=True hierarchical=True

and emitted ZERO trace lines with SGLANG_MAMBA_PIN_TRACE=50 armed, because
the first trace landed in `HiMambaRadixCache`. `registry.py:106-109` routes
`--enable-hierarchical-cache` + hybrid SSM to `_create_unified_radix_cache`
unconditionally, so the hierarchical MAMBA path is UnifiedRadixCache and the
Mamba* classes never see this configuration.

The trace reports the same ledger as the Hi variant, with the pin counts
resolved through the unified registries (`_OngoingWriteThrough` /
`_OngoingLoadBack` carry the acquire's skip set, so an entry that skipped
MAMBA is correctly NOT counted as holding a mamba pin).
"""

# The unified fixture lives beside this file; the suite runs without a
# package __init__, so load it by path rather than by relative import.
import importlib.util
import os
import threading
import unittest
from array import array
from collections import Counter

from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci

_spec = importlib.util.spec_from_file_location(
    "_unified_fixture",
    os.path.join(os.path.dirname(__file__), "test_unified_radix_cache_unittest.py"),
)
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)
# The shared fixture builds on the accelerator; these tests are CPU-only.
_fixture.get_device = lambda *args, **kwargs: "cpu"
CacheConfig = _fixture.CacheConfig
build_fixture = _fixture.build_fixture

register_cpu_ci(est_time=15)

TRACE_LOGGER = "sglang.srt.mem_cache.unified_radix_cache"


def _mamba_cfg() -> CacheConfig:
    return CacheConfig(components=(ComponentType.FULL, ComponentType.MAMBA))


def _key(token_ids) -> RadixKey:
    return RadixKey(array("q", token_ids))


def _insert(cache, allocator, pool, token_ids):
    slot = pool.mamba_allocator.alloc(1)
    assert slot is not None, "test setup: mamba pool exhausted"
    cache.insert(
        InsertParams(
            key=_key(token_ids),
            value=allocator.alloc(len(token_ids)),
            mamba_value=slot,
        )
    )
    return cache.match_prefix(MatchPrefixParams(key=_key(token_ids))).last_device_node


class TestUnifiedPinTrace(unittest.TestCase):
    def test_trace_line_renders_with_the_pin_ledger(self):
        with envs.SGLANG_MAMBA_PIN_TRACE.override(1):
            cache, allocator, pool = build_fixture(_mamba_cfg())
            self.assertEqual(cache._pin_trace_every, 1)
            node = _insert(cache, allocator, pool, list(range(100, 116)))
            cache.inc_lock_ref(node)
            with self.assertLogs(TRACE_LOGGER, level="INFO") as captured:
                cache.check_hicache_events()

        line = next(m for m in captured.output if "MAMBA-PIN-TRACE" in m)
        for field in (
            "impl=unified",
            "tick=",
            "ack_write=",
            "ack_load=",
            "wt_mamba_pins=",
            "lb_mamba_pins=",
            "ongoing_wt=",
            "ongoing_lb=",
            "ongoing_backup=",
            "protected=",
            "evictable=",
            "mamba_avail=",
            "ops[",
        ):
            self.assertIn(field, line)

        # The lock this test took is attributed to THIS function, and it moved
        # a real mamba ref (the node carries a checkpoint).
        self.assertIn("inc@test_trace_line_renders_with_the_pin_ledger=1", line)
        self.assertIn("inc_mamba@test_trace_line_renders_with_the_pin_ledger=1", line)
        # One checkpoint, locked -> protected, nothing evictable.
        self.assertIn("protected=1", line)
        self.assertIn("evictable=0", line)

    def test_release_is_attributed_to_its_own_site(self):
        with envs.SGLANG_MAMBA_PIN_TRACE.override(1):
            cache, allocator, pool = build_fixture(_mamba_cfg())
            node = _insert(cache, allocator, pool, list(range(100, 116)))
            cache.inc_lock_ref(node)
            self._release(cache, node)
            with self.assertLogs(TRACE_LOGGER, level="INFO") as captured:
                cache.check_hicache_events()

        line = next(m for m in captured.output if "MAMBA-PIN-TRACE" in m)
        self.assertIn("dec@_release=1", line)
        self.assertIn("dec_mamba@_release=1", line)
        # Released -> the checkpoint is cache again.
        self.assertIn("protected=0", line)
        self.assertIn("evictable=1", line)

    def _release(self, cache, node):
        cache.dec_lock_ref(node)

    def test_a_tombstone_lock_is_counted_as_a_call_but_not_as_a_mamba_ref(self):
        """The inc/inc_mamba split is what separates 'lock traffic' from
        'pool pressure': only the latter can exhaust the state pool."""
        with envs.SGLANG_MAMBA_PIN_TRACE.override(1):
            cache, allocator, pool = build_fixture(_mamba_cfg())
            token_ids = list(range(100, 116))
            node = _insert(cache, allocator, pool, token_ids)
            # Drop the checkpoint, keeping the node: a mamba tombstone.
            cd = node.component_data[ComponentType.MAMBA]
            pool.mamba_allocator.free(cd.value)
            cache.component_evictable_size_[ComponentType.MAMBA] -= len(cd.value)
            cd.value = None

            cache.inc_lock_ref(node)
            with self.assertLogs(TRACE_LOGGER, level="INFO") as captured:
                cache.check_hicache_events()

        line = next(m for m in captured.output if "MAMBA-PIN-TRACE" in m)
        self.assertIn(
            "inc@test_a_tombstone_lock_is_counted_as_a_call_but_not_as_a_mamba_ref=1",
            line,
        )
        self.assertNotIn("inc_mamba@", line)

    def test_counters_reset_between_lines(self):
        with envs.SGLANG_MAMBA_PIN_TRACE.override(1):
            cache, allocator, pool = build_fixture(_mamba_cfg())
            node = _insert(cache, allocator, pool, list(range(100, 116)))
            cache.inc_lock_ref(node)
            with self.assertLogs(TRACE_LOGGER, level="INFO") as first:
                cache.check_hicache_events()
            with self.assertLogs(TRACE_LOGGER, level="INFO") as second:
                cache.check_hicache_events()

        self.assertIn("inc_mamba@", first.output[0])
        self.assertIn("ops[]", next(m for m in second.output if "MAMBA-PIN-TRACE" in m))

    def test_interval_throttles_the_line(self):
        with envs.SGLANG_MAMBA_PIN_TRACE.override(3):
            cache, _, _ = build_fixture(_mamba_cfg())
            with self.assertLogs(TRACE_LOGGER, level="INFO") as captured:
                for _ in range(6):
                    cache.check_hicache_events()
        self.assertEqual(sum(1 for m in captured.output if "MAMBA-PIN-TRACE" in m), 2)

    def test_default_is_off(self):
        cache, _, _ = build_fixture(_mamba_cfg())
        self.assertEqual(cache._pin_trace_every, 0)
        cache.check_hicache_events()
        self.assertEqual(cache._pin_trace_ops, Counter())

    def test_pin_count_ignores_entries_whose_acquire_skipped_mamba(self):
        """A write-through/load-back lock taken on a mamba TOMBSTONE holds no
        mamba pin; counting it as one would hide the real pin pressure."""
        cache, allocator, pool = build_fixture(_mamba_cfg())
        node = _insert(cache, allocator, pool, list(range(100, 116)))

        holds_pin = _FakeEntry(node, DecLockRefParams())
        skipped = _FakeEntry(
            node,
            DecLockRefParams(skip_lock_node_ids={ComponentType.MAMBA: {node.id}}),
        )
        no_lock = _FakeEntry(node, None)

        self.assertEqual(cache._mamba_pins_in({1: holds_pin}), 1)
        self.assertEqual(cache._mamba_pins_in({1: skipped}), 0)
        self.assertEqual(cache._mamba_pins_in({1: no_lock}), 0)
        self.assertEqual(
            cache._mamba_pins_in({1: holds_pin, 2: skipped, 3: no_lock}), 1
        )


class _FakeEntry:
    """Shape of the `_Ongoing*` NamedTuples the trace reads."""

    def __init__(self, node, lock_params):
        self.node = node
        self.lock_params = lock_params


if __name__ == "__main__":
    unittest.main()


class _FakeEvent:
    def __init__(self, ready: bool = True):
        self.ready = ready

    def query(self) -> bool:
        return self.ready

    def synchronize(self) -> None:
        pass


class _FakeController:
    """Only the two ack queues writing_check / loading_check poll."""

    def __init__(self):
        self.ack_write_queue = []
        self.ack_load_queue = []

    def queue_write(self, node_id: int, ready: bool = True) -> None:
        self.ack_write_queue.append((None, _FakeEvent(ready), [node_id]))

    def queue_load(self, node_id: int, ready: bool = True) -> None:
        self.ack_load_queue.append((None, _FakeEvent(ready), [node_id]))


class _MinAllReduce:
    """MIN all_reduce over N threads, one per simulated TP rank."""

    def __init__(self, n: int):
        self.barrier = threading.Barrier(n, timeout=30)
        self.values = [None] * n
        self.local = threading.local()

    def __call__(self, tensor, op=None, label=None):
        self.values[self.local.rank] = int(tensor.item())
        self.barrier.wait()
        reduced = min(v for v in self.values if v is not None)
        self.barrier.wait()
        tensor.fill_(reduced)


class TestAckDrainAcrossRanks(unittest.TestCase):
    """FALSIFIERS for the #581 write-through drain freeze in the LIVE class.

    Measured on the instrumented boot: `ack_write == wt_mamba_pins ==
    ongoing_wt == protected` climbed together on TP0 (10 -> 71) and TP1
    (-> 31, then plateau) while TP2 stayed at 0 and never took a
    write-backup pin. When the load stopped the queues did NOT drain -- TP0
    frozen at 71 across thousands of idle ticks. A MIN over rank-local ready
    counts pinned at 0 by the idle rank produces exactly that.
    """

    RANKS = 3
    QUEUED = 4

    def _run(self, queued_per_rank, drain_rounds=1, load_queue=False):
        allreduce = _MinAllReduce(self.RANKS)
        caches, drained, errors = [], {}, {}
        for rank in range(self.RANKS):
            cache, _, _ = build_fixture(_mamba_cfg())
            cache.cache_controller = _FakeController()
            cache.tp_world_size = self.RANKS
            cache.pp_rank = 0
            cache._all_reduce = allreduce
            for i in range(queued_per_rank[rank]):
                if load_queue:
                    cache.cache_controller.queue_load(i)
                else:
                    cache.cache_controller.queue_write(i)
            # The scheduling of the drain is under test, not the per-ack
            # bookkeeping: record instead of replaying real registry entries.
            seen = []
            if load_queue:
                cache.ongoing_load_back = _RecordingPop(seen)
                cache.dec_lock_ref = lambda *a, **k: None
                cache.dec_host_lock_ref = lambda *a, **k: None
            else:
                cache._finish_write_through_ack = seen.append
            drained[rank] = seen
            caches.append(cache)

        def run(rank):
            allreduce.local.rank = rank
            try:
                for _ in range(drain_rounds):
                    if load_queue:
                        caches[rank].loading_check()
                    else:
                        caches[rank].writing_check()
            except BaseException as exc:
                errors[rank] = exc
                raise

        threads = [threading.Thread(target=run, args=(r,)) for r in range(self.RANKS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        self.assertEqual(errors, {}, f"rank thread raised: {errors}")
        self.assertFalse(any(t.is_alive() for t in threads), "rank thread hung")
        return caches, drained

    def test_an_idle_rank_does_not_freeze_the_write_drain(self):
        """FALSIFIER: TP2 has nothing queued; TP0/TP1 must still drain.

        Before the fix every rank drained 0 and the queues stayed full --
        the live 71/31/0 freeze.
        """
        caches, drained = self._run([self.QUEUED, self.QUEUED, 0])

        self.assertEqual(len(drained[0]), self.QUEUED, "backing-up rank never drained")
        self.assertEqual(len(drained[1]), self.QUEUED)
        self.assertEqual(len(drained[2]), 0)
        for rank in range(self.RANKS):
            self.assertEqual(caches[rank].cache_controller.ack_write_queue, [])

    def test_an_idle_rank_does_not_freeze_the_load_drain(self):
        """The loading_check sibling has the identical shape."""
        caches, drained = self._run([self.QUEUED, self.QUEUED, 0], load_queue=True)
        self.assertEqual(len(drained[0]), self.QUEUED)
        self.assertEqual(len(drained[2]), 0)
        for rank in range(self.RANKS):
            self.assertEqual(caches[rank].cache_controller.ack_load_queue, [])

    def test_queues_empty_when_no_new_backups_arrive(self):
        """Idle-drain regression: the live falsifier is that the frozen
        queues must return to ~0 once the load stops."""
        caches, _ = self._run([self.QUEUED, 2, 0], drain_rounds=4)
        for rank in range(self.RANKS):
            self.assertEqual(
                caches[rank].cache_controller.ack_write_queue,
                [],
                f"rank{rank} still holds acks after idle rounds",
            )

    def test_a_slow_rank_no_longer_throttles_the_others(self):
        """THE THROTTLE IS WITHDRAWN BY DESIGN (#737), and this records it.

        This asserted the opposite -- that a rank whose head event is not ready
        holds every other rank at zero. That property came from MIN-reducing the
        ready count across the group, and that reduction was the #737 deadlock:
        it sat inside the per-microbatch path of a pipeline, whose stages are at
        different offsets by construction, so PP0/PP1 waited in it for a PP2
        that was blocked in `_pp_recv_proxy_tensors` waiting for them.

        The reduction is gone and counting is rank-local. What that costs is
        PACING, not correctness: ranks may now run further apart in ack
        processing. Correctness is owned by #706's per-page completeness marker
        -- an incomplete page reads as a MISS, never as wrong bytes -- and a
        replacement backpressure bound is FILED rather than guessed, with a
        drain-depth observability line so the first real fast-rank pressure
        specimen is attributable.

        What #581 actually needed SURVIVES and is asserted below: a rank that
        cannot drain must not stop the ranks that can. That was the exhaustion
        this file exists for, and rank-local counting makes it unreachable
        rather than merely compensated.
        """
        allreduce = _MinAllReduce(self.RANKS)
        caches, drained, errors = [], {}, {}
        for rank in range(self.RANKS):
            cache, _, _ = build_fixture(_mamba_cfg())
            cache.cache_controller = _FakeController()
            cache.tp_world_size = self.RANKS
            cache.pp_rank = 0
            cache._all_reduce = allreduce
            # rank 2 has a queue whose head is still in flight -> 0 ready.
            for i in range(3):
                cache.cache_controller.queue_write(i, ready=(rank != 2))
            seen = []
            cache._finish_write_through_ack = seen.append
            drained[rank] = seen
            caches.append(cache)

        def run(rank):
            allreduce.local.rank = rank
            try:
                caches[rank].writing_check()
            except BaseException as exc:
                errors[rank] = exc
                raise

        threads = [threading.Thread(target=run, args=(r,)) for r in range(self.RANKS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        self.assertEqual(errors, {}, f"rank thread raised: {errors}")
        # rank 2's head is still in flight, so it drains nothing...
        self.assertEqual(len(drained[2]), 0, "rank2's unready head must not drain")
        # ...and that no longer holds anybody else back. This is the #581
        # property that mattered, now true by construction.
        for rank in (0, 1):
            self.assertEqual(
                len(drained[rank]),
                3,
                f"rank{rank} was throttled by a peer it no longer consults",
            )


class _RecordingPop(dict):
    """ongoing_load_back stand-in: records the ack ids the drain pops."""

    def __init__(self, sink):
        super().__init__()
        self._sink = sink

    def pop(self, ack_id, *args):
        self._sink.append(ack_id)
        return (None, None, None)
