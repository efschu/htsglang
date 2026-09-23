"""fnFL2x62: two defects of the paged arena form (Task #107), both measured.

1. SIDECARS WITHOUT AN ARENA HOME WERE NEVER WRITTEN.  A direct write completes
   the KV page in the arena, the write-through ack then skips
   ``write_backup_storage`` -- and with it the plain V4 sidecars (Next Flash's
   QSA index, the paged draft).  Group P issued 0 store writes (x58: 12), D
   found 0 QSA anchors and capped every claim to 0::

       #1035c ZERO-ANSWER PARTITION ... cause=CAPPED ... claimed=0 by=qsa_indexer

   Guarded: ``_plain_sidecar_transfers`` keeps exactly the registered pools
   without an arena; ``sidecar_only`` persists them and acks without the KV
   pass.

2. NO-KV PEERS SYNCED TO THE STAGING RING.  The Form A expert workers (0 B per
   token) take the group MIN of the fixed pool size; under the arena form the
   attention host's ``size`` is its 4096-row staging ring, so the workers held
   4096 ids for a 4480-token re-read and declined it (``host_pool_exhausted``)
   while TP0 issued it -- a rank-local exit before the claim vote, gloo
   ``8 vs 4``.  Guarded: the arena pool bids its id space to the second sync.

Hermetic: no CUDA, no process group, no boot.
"""

import os
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
    StorageOperation,
)
from sglang.srt.mem_cache.pool_host import arena_pool as ap
from sglang.srt.mem_cache.pool_host.base import NO_KV_RANK_TOKENS
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache


# ---- 1. sidecar filter and sidecar-only backup -----------------------------

def _xfer(name):
    return PoolTransfer(name=name, keys=["k1", "k2"], indices_from_pool=PoolName.KV)


def test_the_filter_keeps_registered_pools_without_an_arena_only():
    pools = {
        PoolName.KV: SimpleNamespace(arena_read=True),
        PoolName.MAMBA: SimpleNamespace(arena_read=True),      # its own arena
        PoolName.QSA_INDEXER: SimpleNamespace(arena_read=False),  # plain V4 sidecar
        # DRAFT deliberately NOT registered: the backend would KeyError on it
    }
    xfers = [_xfer(PoolName.KV), _xfer(PoolName.MAMBA), _xfer(PoolName.QSA_INDEXER), _xfer(PoolName.DRAFT)]
    kept = UnifiedRadixCache._plain_sidecar_transfers(xfers, pools)
    assert [t.name for t in kept] == [PoolName.QSA_INDEXER]


def test_the_filter_is_empty_without_a_registry():
    assert UnifiedRadixCache._plain_sidecar_transfers([_xfer(PoolName.QSA_INDEXER)], None) == []
    assert UnifiedRadixCache._plain_sidecar_transfers([], {}) == []


class _Backend:
    def __init__(self):
        self.v2_calls = []

    def batch_set_v2(self, transfers):
        self.v2_calls.append([t.name for t in transfers])
        return {t.name: [True] * len(t.keys or []) for t in transfers}

    def batch_set(self, *a, **k):  # the KV pass -- must never run for sidecar_only
        raise AssertionError("KV pages were written on a sidecar_only operation")


def _controller():
    cc = object.__new__(HybridCacheController)
    cc.page_size = 64
    cc.storage_backend = _Backend()
    cc.page_set_func = lambda *a, **k: cc.storage_backend.batch_set()
    return cc


def test_sidecar_only_persists_the_sidecars_and_acks_every_token():
    cc = _controller()
    hv = torch.arange(6976, 6976 + 128, dtype=torch.int64)  # two arena pages
    op = StorageOperation(hv, list(range(128)), hash_value=["h1", "h2"],
                          pool_transfers=[_xfer(PoolName.QSA_INDEXER)])
    op.sidecar_only = True
    cc._page_backup(op)
    assert cc.storage_backend.v2_calls == [[PoolName.QSA_INDEXER]]
    assert op.pool_transfers[0].host_indices is hv  # derived from the KV rows
    assert op.completed_tokens == 128


def test_a_regular_operation_still_takes_the_kv_pass():
    cc = _controller()
    hv = torch.arange(128, dtype=torch.int64)
    op = StorageOperation(hv, list(range(128)), hash_value=["h1", "h2"],
                          pool_transfers=[_xfer(PoolName.QSA_INDEXER)])
    try:
        cc._page_backup(op)
    except AssertionError as e:
        assert "KV pages were written" in str(e)
    else:
        raise AssertionError("the KV pass did not run for a regular operation")


# ---- 2. the no-KV peers' capacity ------------------------------------------

def test_a_plain_pool_bids_its_own_size_and_keeps_it():
    p = object.__new__(MHATokenToKVPoolHost)
    p.size_per_token = 12288
    assert p._carrier_capacity_bid(325568) == 325568
    assert p._sync_no_kv_peers(325568, 4.0) == 325568  # no process group: unchanged


def test_a_no_kv_rank_takes_the_synced_value():
    p = object.__new__(MHATokenToKVPoolHost)
    p.size_per_token = 0
    # without a process group the sync hands the bid back: the sentinel, never
    # the 4096-row ring of x62
    assert p._sync_no_kv_peers(4096, 4.0) == NO_KV_RANK_TOKENS


def test_the_arena_pool_bids_its_id_space_not_its_ring():
    p = object.__new__(ap.ArenaMHAHostPool)
    p.page_size = 64
    p.size_per_token = 12288  # 12 attention layers x 1024 B (Next Flash fp8)
    with mock.patch.dict(os.environ, {"SGLANG_HICACHE_ARENA_GIB": "2"}, clear=False):
        os.environ.pop("SGLANG_HICACHE_ARENA_KV_PAGE_BYTES", None)
        slots = ap.planned_arena_slots(12288 * 64)
        assert slots == max(1024, (2 << 30) // 786432) == 2730
        assert p._carrier_capacity_bid(4096) == 4096 + 2730 * 64
        # the pool itself keeps its ring: the second sync changes only peers
        assert p._sync_no_kv_peers(4096, 0.05) == 4096


def test_an_arena_pool_without_bytes_bids_the_sentinel():
    p = object.__new__(ap.ArenaMHAHostPool)
    p.page_size = 64
    p.size_per_token = 0
    assert p._carrier_capacity_bid(4096) == NO_KV_RANK_TOKENS
