# SPDX-License-Identifier: Apache-2.0
"""27B HiCache drain spikes (boot xsn429, 24.09.): the cause, and the bounded drain.

MEASURED. `check_hicache_events` drains the storage control queues on the
scheduler thread every round; its drain term spiked after every terminated or
revoked store prefetch, linearly in the span given back (P PP0: 8189 tokens ->
13.3 ms, 32766 -> 59.8 ms, 94795 -> 171.9 ms, 255726 -> 401.1 ms; D: 4314 ->
8.3-8.8 ms), i.e. ~1.7 us per token. The span is released through
`append_host_mem_release`; the hybrid controller splits it by the host pool's
page size -- 1 on the arena host pool -- so the drain pays one Queue.get_nowait,
append and len() per TOKEN plus a torch.cat over as many one-row tensors, to free
rows `HostPoolGroup.free` then drops (placeholders / arena ids beyond staging).

SGLANG_HICACHE_DRAIN_BUDGET=<rows per round> (default off): one queue entry per
release, at most <rows> rows (and ENTRY_BUDGET entries) per release queue and
round with the rest re-queued at the front, at most N acks per round. What must
hold: off is the unchanged path; on, a round never touches more than the
budget, no row or ack is lost or freed twice, the rest stays where every
settle/clear/fence path already looks (the queue), and a TP group drains the
same rows on the same rounds.
"""

import os
import types
import unittest
from queue import Queue
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache import unified_radix_cache as u
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import HybridCacheController
from sglang.srt.mem_cache.memory_pool_host import HostPoolGroup, PoolEntry
from sglang.srt.mem_cache.pool_host.arena_pool import ArenaMHAHostPool
from sglang.test.test_utils import CustomTestCase

# literals, not the module's names: the same file runs (red) against the
# parent tree, where hicache_drain_budget does not exist yet
ENV_BUDGET = "SGLANG_HICACHE_DRAIN_BUDGET"
ENV_ACK = "SGLANG_HICACHE_DRAIN_ACK_BUDGET"
ENTRY_BUDGET = 512
DEFAULT_ACK_BUDGET = 32
_ENVS = (ENV_BUDGET, ENV_ACK, "SGLANG_HICACHE_ROUND_TIMING",
         "SGLANG_HICACHE_DRAIN_AGREE_EVERY", "SGLANG_WEG2_RELEASE_DRAIN_CAP")

#: P's geometry in xsn429 (ARENA-PRESENT stats): staging rows, arena slots
_STAGING, _SLOTS = 2442, 720896


class _EnvCase(CustomTestCase):
    def setUp(self):
        super().setUp()
        self._saved = {k: os.environ.pop(k, None) for k in _ENVS}

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        super().tearDown()


def _arena_pool():
    """The live P host pool: an ArenaMHAHostPool bound to the arena geometry
    (only what the release path reads -- its staging size and id space)."""
    p = ArenaMHAHostPool.__new__(ArenaMHAHostPool)
    p.size, p.page_size, p.layout, p.device = _STAGING, 1, "layer_first", "cpu"
    p._arena_init_fields()
    p.arena_slots = _SLOTS
    p.id_space = _STAGING + _SLOTS + (1 << 22)
    return p


class _RecordingPool:
    """A host pool that records every free (conservation checks)."""

    def __init__(self, page_size=1):
        self.page_size = page_size
        self.freed = []

    def free(self, idx):
        self.freed.append(idx.clone())
        return int(idx.numel())


def _hybrid_cc(pool):
    cc = HybridCacheController.__new__(HybridCacheController)
    cc.mem_pool_host = pool
    cc.host_mem_release_queue = Queue()
    cc.extra_host_mem_release_queues = {}
    cc.prefetch_revoke_queue = Queue()
    cc.ack_backup_queue = Queue()
    return cc


def _real_p_cc():
    pool = _arena_pool()
    group = HostPoolGroup([PoolEntry(name=PoolName.KV, host_pool=pool, device_pool=None,
                                     layer_mapping={}, is_primary_index_anchor=True)])
    return pool, _hybrid_cc(group)


def _cache(cc, tp_world_size=1):
    c = u.UnifiedRadixCache.__new__(u.UnifiedRadixCache)
    c.cache_controller = cc
    c.attn_cp_group = c.attn_tp_group = None
    c.tp_world_size = tp_world_size
    c.enable_storage = True
    c.enable_storage_metrics = False
    c.storage_metrics_collector = None
    c.ongoing_prefetch, c.ongoing_backup = {}, {}
    c.staging_write_ring = None
    c._pin_trace_every = 0
    c._drain_async_work = lambda: None
    c.writing_check = lambda *a, **k: None
    c.loading_check = lambda: None
    return c


def _queued_rows(q):
    return [int(t.numel()) for t in list(q.queue)]


class TheDefaultPathIsUnchanged(_EnvCase):
    """The cause, on the unchanged path: page size 1 = one queue entry per
    token, and one round drains the whole agreed count."""

    def test_a_span_is_one_entry_per_token_and_one_round_drains_it_all(self):
        pool, cc = _real_p_cc()
        c = _cache(cc)
        span = pool.alloc_read(4314)  # a prefetch registration's placeholders
        cc.append_host_mem_release(host_indices=span)
        self.assertEqual(cc.host_mem_release_queue.qsize(), 4314)
        c.check_hicache_events()
        self.assertEqual(cc.host_mem_release_queue.qsize(), 0)

    def test_garbage_and_nonpositive_budgets_are_off(self):
        from sglang.srt.mem_cache import hicache_drain_budget as hdb

        for v in ("x", "0", "-5", ""):
            os.environ[ENV_BUDGET] = v
            self.assertEqual(hdb.drain_budget_tokens(), 0, v)
            self.assertFalse(hdb.coalesce_host_releases(), v)
            self.assertEqual(hdb.drain_ack_budget(), 0, v)

    def test_the_default_call_passes_no_budget(self):
        c = _cache(_hybrid_cc(_RecordingPool()))
        c._drain_storage_control_queues_impl = mock.Mock()
        c.drain_storage_control_queues()
        self.assertNotIn("budget", c._drain_storage_control_queues_impl.call_args.kwargs)

    def test_instrument_off_records_nothing(self):
        pool, cc = _real_p_cc()
        c = _cache(cc)
        cc.append_host_mem_release(host_indices=pool.alloc_read(64))
        c.check_hicache_events()
        self.assertIsNone(c._hc_parts)
        self.assertFalse(hasattr(c, "_hc_round_acc"))


class TheReleaseIsOneEntry(_EnvCase):
    def test_hybrid_producer_puts_one_entry(self):
        os.environ[ENV_BUDGET] = "16384"
        pool, cc = _real_p_cc()
        span = pool.alloc_read(94795)
        cc.append_host_mem_release(host_indices=span)
        self.assertEqual(cc.host_mem_release_queue.qsize(), 1)
        self.assertTrue(torch.equal(cc.host_mem_release_queue.queue[0], span))

    def test_base_producer_puts_one_entry_with_provenance(self):
        os.environ[ENV_BUDGET] = "16384"
        cc = HiCacheController.__new__(HiCacheController)
        cc.mem_pool_host = types.SimpleNamespace(page_size=1)
        cc.host_mem_release_queue = Queue()
        cc.host_release_provenance = {}
        idx = torch.arange(100, 5100, dtype=torch.int64)
        cc.append_host_mem_release(idx)
        self.assertEqual(cc.host_mem_release_queue.qsize(), 1)
        self.assertIn(100, cc.host_release_provenance)

    def test_an_unaligned_span_keeps_the_page_split(self):
        os.environ[ENV_BUDGET] = "16384"
        cc = _hybrid_cc(_RecordingPool(page_size=4))
        cc._append_host_mem_release_pages(cc.host_mem_release_queue, torch.arange(10), 4)
        self.assertEqual(_queued_rows(cc.host_mem_release_queue), [4, 4, 2])


class ARoundTouchesAtMostTheBudget(_EnvCase):
    def test_p_shape_95k_prefetch_span_drains_in_budget_slices(self):
        """xsn429 PP0 19:00:00: a revoked 94115-token prefetch, drained in ONE
        round (~170 ms on metal). Budget 16384: six rounds, each <= budget."""
        os.environ[ENV_BUDGET] = "16384"
        pool, cc = _real_p_cc()
        c = _cache(cc)
        n = 94795
        cc.append_host_mem_release(host_indices=pool.alloc_read(n))
        left = [n]
        for _ in range(10):
            c.check_hicache_events()
            rows = sum(_queued_rows(cc.host_mem_release_queue))
            self.assertGreaterEqual(left[-1] - rows, 0)
            self.assertLessEqual(left[-1] - rows, 16384)
            left.append(rows)
            if rows == 0:
                break
        self.assertEqual(left[-1], 0)
        self.assertEqual(len(left) - 1, 6)  # ceil(94795 / 16384)

    def test_every_row_reaches_free_exactly_once_in_order(self):
        os.environ[ENV_BUDGET] = "1000"
        pool = _RecordingPool()
        cc = _hybrid_cc(pool)
        c = _cache(cc)
        a = torch.arange(0, 2500, dtype=torch.int64)
        b = torch.arange(10000, 10700, dtype=torch.int64)
        cc.append_host_mem_release(host_indices=a)
        cc.append_host_mem_release(host_indices=b)
        for _ in range(4):
            c.drain_storage_control_queues()
        freed = torch.cat(pool.freed)
        self.assertTrue(torch.equal(freed, torch.cat([a, b])))
        self.assertEqual([int(t.numel()) for t in pool.freed], [1000, 1000, 1000, 200])

    def test_the_rest_stays_in_the_queue_for_every_settle_path(self):
        """No second holder: after a budgeted round the rest is a queue entry,
        so the fence / detach drain (and the rebind settle, which empties the
        same queue) frees it."""
        os.environ[ENV_BUDGET] = "1000"
        pool = _RecordingPool()
        cc = _hybrid_cc(pool)
        c = _cache(cc)
        span = torch.arange(0, 3500, dtype=torch.int64)
        cc.append_host_mem_release(host_indices=span)
        c.drain_storage_control_queues()
        self.assertEqual(_queued_rows(cc.host_mem_release_queue), [2500])
        c._drain_storage_control_queues_local()
        self.assertEqual(cc.host_mem_release_queue.qsize(), 0)
        self.assertTrue(torch.equal(torch.cat(pool.freed), span))

    def test_the_cut_is_page_aligned(self):
        os.environ[ENV_BUDGET] = "10"
        pool = _RecordingPool(page_size=4)
        cc = _hybrid_cc(pool)
        c = _cache(cc)
        cc.append_host_mem_release(host_indices=torch.arange(24))
        c.drain_storage_control_queues()
        self.assertEqual(int(pool.freed[0].numel()), 8)
        self.assertEqual(_queued_rows(cc.host_mem_release_queue), [16])

    def test_page_sized_entries_are_bounded_by_the_entry_budget(self):
        """A producer that still enqueues one entry per page (not coalesced)
        is bounded by ENTRY_BUDGET entries per round."""
        os.environ[ENV_BUDGET] = "100000"
        pool = _RecordingPool()
        cc = _hybrid_cc(pool)
        c = _cache(cc)
        for i in range(2000):
            cc.host_mem_release_queue.put(torch.tensor([i]))
        c.drain_storage_control_queues()
        self.assertEqual(int(pool.freed[0].numel()), ENTRY_BUDGET)
        self.assertEqual(cc.host_mem_release_queue.qsize(), 2000 - ENTRY_BUDGET)

    def test_extra_pool_releases_are_budgeted_too(self):
        os.environ[ENV_BUDGET] = "300"
        pool = _RecordingPool()
        mamba = _RecordingPool()
        cc = _hybrid_cc(pool)
        cc.extra_host_mem_release_queues = {PoolName.MAMBA: Queue()}
        cc.entry_for_extra_release = lambda name: types.SimpleNamespace(host_pool=mamba)
        c = _cache(cc)
        cc.extra_host_mem_release_queues[PoolName.MAMBA].put(torch.arange(700))
        for _ in range(3):
            c.drain_storage_control_queues()
        self.assertEqual([int(t.numel()) for t in mamba.freed], [300, 300, 100])
        self.assertTrue(torch.equal(torch.cat(mamba.freed), torch.arange(700)))


class AcksBeyondTheBudgetWaitInTheirQueue(_EnvCase):
    def _backup_cache(self, n):
        cc = _hybrid_cc(_RecordingPool())
        c = _cache(cc)
        released = []
        c.staging_write_ring = types.SimpleNamespace(release=released.append)
        c.dec_host_lock_ref = lambda node, params: None
        c._weg2_rebind_host_to_arena = lambda node: True
        for i in range(n):
            node = types.SimpleNamespace(l3_present=False)
            c.ongoing_backup[i] = (node, None)
            cc.ack_backup_queue.put(types.SimpleNamespace(id=i, completed_tokens=4096))
        return c, cc, released

    def test_default_drains_every_agreed_ack_in_one_round(self):
        c, cc, released = self._backup_cache(100)
        c.drain_storage_control_queues()
        self.assertEqual(len(released), 100)

    def test_budget_drains_at_most_n_acks_per_round_and_loses_none(self):
        os.environ[ENV_BUDGET] = "16384"
        os.environ[ENV_ACK] = "32"
        c, cc, released = self._backup_cache(100)
        per_round = []
        for _ in range(5):
            before = len(released)
            c.drain_storage_control_queues()
            per_round.append(len(released) - before)
        self.assertEqual(per_round, [32, 32, 32, 4, 0])
        self.assertEqual(released, list(range(100)))  # in order, each once
        self.assertEqual(c.ongoing_backup, {})

    def test_default_ack_budget_applies_when_only_the_row_budget_is_set(self):
        from sglang.srt.mem_cache import hicache_drain_budget as hdb

        os.environ[ENV_BUDGET] = "16384"
        self.assertEqual(hdb.drain_ack_budget(), DEFAULT_ACK_BUDGET)
        os.environ[ENV_ACK] = "0"
        self.assertEqual(hdb.drain_ack_budget(), 0)  # 0 = every agreed ack


class TheBudgetIsRankUniform(_EnvCase):
    """Group D (TP 3): the same releases reach the three ranks at different
    rounds. The MIN agreement plus constant allowances must free the same rows
    on the same rounds on every rank -- also under the gated cadence."""

    def _run(self, every):
        os.environ[ENV_BUDGET] = "1000"
        os.environ["SGLANG_HICACHE_DRAIN_AGREE_EVERY"] = str(every)
        pools = [_RecordingPool() for _ in range(3)]
        ranks = [_cache(_hybrid_cc(p), tp_world_size=3) for p in pools]
        agreed = {}

        def reduce(tensor, op, label=""):  # the gloo MIN over the three ranks
            tensor.copy_(agreed["min"])

        for r in ranks:
            r._all_reduce_attn_groups = reduce
        a = torch.arange(0, 2500, dtype=torch.int64)
        b = torch.arange(50000, 50700, dtype=torch.int64)
        freed_per_round = [[] for _ in range(3)]
        for rnd in range(1, 41):
            for i, r in enumerate(ranks):
                if rnd == 1 + 2 * i:
                    r.cache_controller.append_host_mem_release(host_indices=a)
                if rnd == 9 + i:
                    r.cache_controller.append_host_mem_release(host_indices=b)
            vecs = [torch.tensor([0, 0, r.cache_controller.host_mem_release_queue.qsize()]
                                 + [0] * u._POOL_SLOT_COUNT, dtype=torch.int) for r in ranks]
            agreed["min"] = torch.minimum(torch.minimum(vecs[0], vecs[1]), vecs[2])
            for i, r in enumerate(ranks):
                before = len(pools[i].freed)
                r.check_hicache_events()
                freed_per_round[i].append([t.tolist() for t in pools[i].freed[before:]])
        return pools, freed_per_round, a, b

    def test_same_rows_same_rounds_every_round(self):
        pools, per_round, a, b = self._run(every=1)
        self.assertEqual(per_round[0], per_round[1])
        self.assertEqual(per_round[1], per_round[2])
        for p in pools:
            self.assertTrue(torch.equal(torch.cat(p.freed), torch.cat([a, b])))

    def test_same_rows_same_rounds_under_the_gated_cadence(self):
        pools, per_round, a, b = self._run(every=8)
        self.assertEqual(per_round[0], per_round[1])
        self.assertEqual(per_round[1], per_round[2])
        for p in pools:
            self.assertTrue(torch.equal(torch.cat(p.freed), torch.cat([a, b])))
        rows = [sum(len(x) for x in r) for r in per_round[0]]
        self.assertEqual(sum(rows), 3200)
        self.assertLessEqual(max(rows), 1000)  # never more than the budget in a round
        # the leftovers keep the gate hot: consecutive rounds, not every 8th
        busy = [i for i, n in enumerate(rows) if n]
        self.assertEqual(busy, list(range(busy[0], busy[0] + len(busy))))


class TheInstrumentNamesThePart(_EnvCase):
    def test_the_timing_line_carries_the_parts_of_the_max_round(self):
        os.environ["SGLANG_HICACHE_ROUND_TIMING"] = "4"
        pool, cc = _real_p_cc()
        c = _cache(cc)
        c.check_hicache_events()
        cc.append_host_mem_release(host_indices=pool.alloc_read(8189))
        with self.assertLogs(u.logger, level="INFO") as cm:
            for _ in range(3):
                c.check_hicache_events()
        lines = [m for m in cm.output if "HICACHE-ROUND-TIMING" in m]
        self.assertEqual(len(lines), 1)
        self.assertIn("rounds=4", lines[0])
        self.assertIn("max-round agree=", lines[0])
        self.assertIn("release=", lines[0])
        self.assertIn("(entries 8189 rows 8189)", lines[0])  # the unchanged path: 1 entry per token
        self.assertIn("budget rows=0 acks=0", lines[0])
        self.assertIsNone(c._hc_parts)

    def test_with_the_budget_the_max_round_names_one_entry_and_the_slice(self):
        os.environ["SGLANG_HICACHE_ROUND_TIMING"] = "4"
        os.environ[ENV_BUDGET] = "16384"
        pool, cc = _real_p_cc()
        c = _cache(cc)
        cc.append_host_mem_release(host_indices=pool.alloc_read(94795))
        with self.assertLogs(u.logger, level="INFO") as cm:
            for _ in range(4):
                c.check_hicache_events()
        line = [m for m in cm.output if "HICACHE-ROUND-TIMING" in m][0]
        self.assertIn("(entries 1 rows 16384)", line)  # one slice of the one entry
        self.assertIn("rows=65536", line)
        self.assertIn("budget rows=16384 acks=32 release_q=1", line)

    def test_the_agreement_is_timed_apart_from_the_drains(self):
        os.environ["SGLANG_HICACHE_ROUND_TIMING"] = "1"
        c = _cache(_hybrid_cc(_RecordingPool()), tp_world_size=3)

        def slow_reduce(tensor, op, label=""):
            import time as _t

            _t.sleep(0.02)  # a peer late into the gloo MIN

        c._all_reduce_attn_groups = slow_reduce
        with self.assertLogs(u.logger, level="INFO") as cm:
            c.check_hicache_events()
        line = [m for m in cm.output if "HICACHE-ROUND-TIMING" in m][0]
        agree_ms = float(line.split("max-round agree=")[1].split()[0])
        release_ms = float(line.split("max-round agree=")[1].split("release=")[1].split()[0])
        self.assertGreaterEqual(agree_ms, 15.0)
        self.assertLess(release_ms, 15.0)


if __name__ == "__main__":
    unittest.main()
