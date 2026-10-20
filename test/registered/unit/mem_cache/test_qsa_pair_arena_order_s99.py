"""QSA-PAIR (S99): a direct-written KV page becomes readable only together
with its QSA index.

Metal (NF rc22 9c40fe40a2, P log 10:19:45, PP0/PP1/PP2 alike)::

    #1028B FETCH CAP n=7: kv=952 claimed=760 lost=192
        caps={qsa_indexer: 760, mamba: 760} keys=968
    PDFLIP-CLAIM-SHORT rid=pdflip-12-238 pp=0 asked=968 kv=952 claimed=760
        lost=192 short=208 by=mamba,qsa_indexer
    PDFLIP P-WAKE-PROBE rid=pdflip-12-238 have=48640 store_now=60928
        (10:19:47: the index had landed two seconds later)

D (Form A, TP0 = the only QSA writer) had direct-written the KV of pages
760..951 into the shared arena (#1427): ``_pdflip_direct_complete`` made the
slots COMPLETE at the write-through ack, while their QSA index -- a plain
sidecar without an arena home -- was only QUEUED on the backup thread
(``_pdflip_write_plain_sidecars``, #106S) behind D's flip backlog
(``hicache_backup(34)`` on TP0, 18 on the workers). A read in that window met
KV without index and the ALL_PAGES cap took the pages.

Pinned (RED on 7336200fde, the rc22 head):
  1. after the write-through ack the KV slot is NOT complete while its index
     write is still queued; it completes at that write's ack drain -- a
     reader of the store never sees KV without index;
  2. the index write goes ahead of the queued KV backups (FIFO among index
     writes, same queue);
  3. every exit settles the waiting KV slot exactly once: a failed index
     write (named), the K12 abandon (before the host lock goes), the reset,
     a split (both fragments);
  4. nothing changes where no own claim-capping index rides along (a Form A
     worker's byteless sidecars, an anchor-only write, a staging write).

Hermetic: no CUDA, no process group, no boot.
"""

from __future__ import annotations

import logging
import os
import types
from queue import Queue

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.managers.cache_controller import StorageOperation  # noqa: E402
from flliper.srt.mem_cache import pdflip_backup_guard as guard  # noqa: E402
from flliper.srt.mem_cache.hicache_storage import (  # noqa: E402
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from flliper.srt.mem_cache.hybrid_cache import hybrid_cache_controller as hcc  # noqa: E402
from flliper.srt.mem_cache.unified_cache_components import BASE_COMPONENT_TYPE  # noqa: E402
from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

P = 64            # tokens per page
S = 4096          # staging rows: arena rows start here
QSA_LOG = "flliper.srt.mem_cache.unified_radix_cache"


class _Node:
    _next = 100

    def __init__(self, pages, first_slot=10, anchor_only=False):
        _Node._next += 1
        self.id = _Node._next
        self.write_through_pending_id = None
        self.hash_value = [f"h{self.id}-{i}" for i in range(pages)]
        rows = torch.arange(S + first_slot * P, S + (first_slot + pages) * P, dtype=torch.int64)
        self.component_data = {BASE_COMPONENT_TYPE: types.SimpleNamespace(host_value=rows, value=None)}
        self.key = types.SimpleNamespace(token_ids=list(range(pages * P)))
        self.parent = None
        self.l3_present = False
        self.anchor_only = anchor_only

    @property
    def backuped(self):
        return self.component_data[BASE_COMPONENT_TYPE].host_value is not None


class _Store:
    """What a reader of the shared store sees: KV pages whose slot is
    COMPLETE, index pages whose store write ran."""

    def __init__(self):
        self.kv = set()
        self.qsa = set()
        self.order = []

    def unpaired(self):
        return sorted(self.kv - self.qsa)


def _cache(store, *, own_index=True, sidecars=True):
    cache = object.__new__(UnifiedRadixCache)
    cc = object.__new__(hcc.HybridCacheController)
    cc.backup_queue = Queue()
    cc.ack_backup_queue = Queue()
    cc.prefetch_revoke_queue = Queue()
    cc.host_mem_release_queue = Queue()
    cc.mem_pool_host = types.SimpleNamespace(arena=object(), staging_rows=S, arena_tokens=1 << 20)
    cc.storage_backend = types.SimpleNamespace(registered_pools={
        PoolName.QSA_INDEXER: types.SimpleNamespace(arena_read=False),   # plain V4 sidecar
        PoolName.MAMBA: types.SimpleNamespace(arena_read=False),         # a worker's shadow pool
    })
    cache.cache_controller = cc
    cache.enable_storage = True
    cache.enable_storage_metrics = False
    cache.storage_metrics_collector = None
    cache.hicache_storage_pass_prefix_keys = False
    cache.staging_write_ring = None
    cache.ongoing_write_through = {}
    cache.ongoing_backup = {}
    cache._pdflip_anchor_only_ids = set()
    cache.locks = []
    cache._record_store_event = lambda node, medium=None: None
    cache.dec_lock_ref = lambda node, params: None
    cache._pdflip_cap_drain = lambda: 0

    def _inc_host(node):
        cache.locks.append(("inc", node.id))
        return types.SimpleNamespace(to_dec_params=lambda: None)

    cache.inc_host_lock_ref = _inc_host
    cache.dec_host_lock_ref = lambda node, params: cache.locks.append(("dec", node.id))
    cache.write_backup_storage = lambda node: store.order.append(("full_backup", node.id))

    def _complete(node):
        if node.anchor_only:
            store.order.append(("anchor_complete", node.id))
            return True
        store.kv.update(node.hash_value)
        store.order.append(("kv_complete", node.id))
        return True

    cache._pdflip_direct_complete = _complete

    cache._components_tuple = ()   # the sidecar specs below are the whole transfer set

    def _sidecars(phase, kv_xfer, comp_xfers):
        if not sidecars:
            return []
        name = PoolName.QSA_INDEXER if own_index else PoolName.MAMBA
        pol = PoolHitPolicy.ALL_PAGES if own_index else PoolHitPolicy.TRAILING_PAGES
        return [PoolTransfer(name=name, keys=kv_xfer.keys, host_indices=kv_xfer.host_indices,
                             indices_from_pool=PoolName.KV, hit_policy=pol)]

    cache._build_sidecar_transfers = _sidecars
    return cache


def _ack(cache, node):
    node.write_through_pending_id = node.id
    cache.ongoing_write_through[node.id] = (node, None, [node])
    UnifiedRadixCache._finish_write_through_ack(cache, node.id)


def _run_backup_thread(cache, store, n=None, fail=False):
    """The backup thread: execute queued operations in queue order."""
    q = cache.cache_controller.backup_queue
    done = 0
    while not q.empty() and (n is None or done < n):
        op = q.get_nowait()
        if op is None:
            continue
        done += 1
        if fail:
            guard.mark_failed(op, "test")
        elif getattr(op, "sidecar_only", False):
            store.qsa.update(op.hash_value)
            store.order.append(("qsa_written", op.id))
        else:
            store.order.append(("kv_backup_written", op.id))
        cache.cache_controller.ack_backup_queue.put(op)
    return done


def _drain(cache):
    UnifiedRadixCache._drain_storage_control_queues_impl(
        cache, n_revoke=None, n_backup=None, n_release=None,
        extra_release_counts=None, log_metrics=False,
    )


# ---- 1. the order -----------------------------------------------------------

def test_kv_slot_is_not_readable_before_its_index_is_stored():
    store = _Store()
    cache = _cache(store)
    node = _Node(pages=3)
    _ack(cache, node)
    # the index write is queued, the backup thread has not run: a reader
    # must not see the KV pages (12-238: it saw 192 of them and capped)
    assert store.unpaired() == [], "KV readable while its QSA index is only queued"
    assert store.kv == set()
    _run_backup_thread(cache, store)
    assert store.unpaired() == []
    _drain(cache)
    assert store.kv == set(node.hash_value) == store.qsa
    kinds = [k for k, _ in store.order]
    assert kinds.index("qsa_written") < kinds.index("kv_complete")


def test_kv_completes_exactly_once_and_the_host_lock_goes_after_it():
    store = _Store()
    cache = _cache(store)
    node = _Node(pages=2)
    _ack(cache, node)
    _run_backup_thread(cache, store)
    _drain(cache)
    _drain(cache)
    assert [k for k, _ in store.order].count("kv_complete") == 1
    assert cache.__dict__.get("_qsa_pair_deferred") in (None, {})
    assert cache.ongoing_backup == {}


# ---- 2. the index write jumps the backlog -------------------------------------

def test_index_writes_go_ahead_of_queued_kv_backups_fifo_among_themselves():
    store = _Store()
    cache = _cache(store)
    cc = cache.cache_controller
    backlog = []
    for _ in range(3):   # D's flip backlog: full KV backups already queued
        op = StorageOperation(torch.arange(P), list(range(P)), hash_value=["x"])
        cc.backup_queue.put(op)
        backlog.append(op.id)
    a, b = _Node(pages=1, first_slot=20), _Node(pages=1, first_slot=30)
    _ack(cache, a)
    _ack(cache, b)
    queued = [op.id for op in list(cc.backup_queue.queue)]
    side = [op.id for op in list(cc.backup_queue.queue) if getattr(op, "sidecar_only", False)]
    assert len(side) == 2
    assert queued == side + backlog, f"index writes queued behind the backlog: {queued}"
    # one thread step writes the first index: its KV is readable after the drain
    _run_backup_thread(cache, store, n=1)
    _drain(cache)
    assert store.kv == set(a.hash_value)
    assert store.unpaired() == []


def test_put_backup_ahead_keeps_the_queue_counters():
    q = Queue()
    q.put("kv1")
    op = types.SimpleNamespace(id=1)
    assert hcc.put_backup_ahead(q, op) == 0
    assert q.qsize() == 2 and q.get_nowait() is op and q.get_nowait() == "kv1"
    q.task_done()
    q.task_done()   # unfinished_tasks was counted for the inserted item


# ---- 3. every exit settles ----------------------------------------------------

def test_failed_index_write_still_completes_the_kv_slot_named(caplog):
    store = _Store()
    cache = _cache(store)
    node = _Node(pages=2)
    _ack(cache, node)
    _run_backup_thread(cache, store, fail=True)
    with caplog.at_level(logging.WARNING, logger=QSA_LOG):
        _drain(cache)
    assert store.kv == set(node.hash_value), "a failed index write left the KV slot WRITING (leak)"
    assert any("QSA-PAIR KV-COMPLETE WITHOUT INDEX" in r.getMessage() and "why=failed" in r.getMessage()
               for r in caplog.records)


def test_short_index_write_is_named(caplog):
    store = _Store()
    cache = _cache(store)
    node = _Node(pages=3)
    _ack(cache, node)
    op = cache.cache_controller.backup_queue.get_nowait()
    op.pool_storage_result = types.SimpleNamespace(extra_pool_hit_pages={PoolName.QSA_INDEXER: 1})
    cache.cache_controller.ack_backup_queue.put(op)
    with caplog.at_level(logging.WARNING, logger=QSA_LOG):
        _drain(cache)
    assert any("index_written=1" in r.getMessage() and "pages=3" in r.getMessage() for r in caplog.records)


def test_abandon_completes_the_kv_slot_before_the_host_lock_goes():
    store = _Store()
    cache = _cache(store)
    node = _Node(pages=2)
    _ack(cache, node)
    seen = []
    cache.dec_host_lock_ref = lambda n, p: seen.append(("dec", n.id, set(store.kv)))
    abandoned, kept = guard.abandon(cache, "test")
    assert abandoned == 1 and kept == 0
    assert store.kv == set(node.hash_value)
    assert seen and seen[0][2] == set(node.hash_value), "host lock released before the KV completed"
    # the cancelled op is acked unwritten later: no second completion
    _run_backup_thread(cache, store, fail=True)
    _drain(cache)
    assert [k for k, _ in store.order].count("kv_complete") == 1


def test_reset_completes_a_waiting_kv_slot():
    store = _Store()
    cache = _cache(store)
    node = _Node(pages=2)
    _ack(cache, node)
    assert UnifiedRadixCache._qsa_pair_settle_all(cache, "reset") == 1
    assert store.kv == set(node.hash_value)
    assert cache.__dict__["_qsa_pair_deferred"] == {}


def test_reset_full_settles_first():
    import inspect

    src = inspect.getsource(UnifiedRadixCache._reset_full)
    assert src.index('_qsa_pair_settle_all(self, "reset")') < src.index(
        "self._release_host_values_before_reset()")


def test_a_split_hands_the_wait_to_both_fragments():
    store = _Store()
    cache = _cache(store)
    node = _Node(pages=4)
    _ack(cache, node)
    upper = _Node(pages=1, first_slot=60)   # the fragment a split creates above `node`
    UnifiedRadixCache._qsa_pair_on_split(cache, node, upper)
    _run_backup_thread(cache, store)
    _drain(cache)
    done = [i for k, i in store.order if k == "kv_complete"]
    assert sorted(done) == sorted([node.id, upper.id])


def test_split_node_calls_the_pair_hook():
    import inspect

    src = inspect.getsource(UnifiedRadixCache._split_node)
    assert "_qsa_pair_on_split(self, child, new_node)" in src


# ---- 4. unchanged where no own index rides along ------------------------------

def test_worker_without_an_own_index_completes_at_the_ack():
    store = _Store()
    cache = _cache(store, own_index=False)   # TP1/TP2: plain sidecar 'mamba' (byteless)
    node = _Node(pages=2)
    _ack(cache, node)
    assert store.kv == set(node.hash_value)
    assert [k for k, _ in store.order][0] == "kv_complete"


def test_host_state_pools_of_a_worker_do_not_defer(monkeypatch):
    from flliper.srt.managers import cache_controller as ccm

    monkeypatch.setattr(ccm, "split_host_state_pools", lambda c, t: ([], list(t)))
    store = _Store()
    cache = _cache(store)
    node = _Node(pages=1)
    _ack(cache, node)
    assert store.kv == set(node.hash_value)


def test_no_sidecars_completes_at_the_ack():
    store = _Store()
    cache = _cache(store, sidecars=False)
    node = _Node(pages=2)
    _ack(cache, node)
    assert store.kv == set(node.hash_value)
    assert cache.cache_controller.backup_queue.empty()


def test_anchor_only_write_is_not_deferred():
    store = _Store()
    cache = _cache(store)
    node = _Node(pages=2, anchor_only=True)
    cache._pdflip_anchor_only_ids.add(node.id)
    _ack(cache, node)
    assert ("anchor_complete", node.id) == store.order[0]


def test_staging_rows_are_not_deferred():
    store = _Store()
    cache = _cache(store)
    node = _Node(pages=1)
    node.component_data[BASE_COMPONENT_TYPE].host_value = torch.arange(P, dtype=torch.int64)
    _ack(cache, node)
    assert store.order[0] == ("kv_complete", node.id)


def test_storage_off_is_not_deferred():
    store = _Store()
    cache = _cache(store)
    cache.enable_storage = False
    node = _Node(pages=1)
    _ack(cache, node)
    assert store.order == [("kv_complete", node.id)]


def test_refused_index_write_completes_at_once():
    store = _Store()
    cache = _cache(store)
    node = _Node(pages=1)
    # an A2-orphan id above the arena: K12 refuses the store write, nothing is issued
    node.component_data[BASE_COMPONENT_TYPE].host_value = torch.full((P,), S + (1 << 20) + 5, dtype=torch.int64)
    _ack(cache, node)
    assert ("kv_complete", node.id) in store.order
    assert cache.__dict__.get("_qsa_pair_deferred") in (None, {})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
