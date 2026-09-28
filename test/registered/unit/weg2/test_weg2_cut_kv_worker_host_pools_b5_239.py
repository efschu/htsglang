"""#239 Blocker 5: under the Form A token cut D decoded nothing.

Metal (rc12z30d -st-cut-vsync, 37c76c65e9, D log 21:11:02, cut [0,32,32], F14
OWNER-ROWS page 64 S=2, TP1 rows [0,1), TP2 [1,2)):

* TP1/TP2 ``#1028B FETCH CAP n=4: kv=264 claimed=0 lost=264 caps={} ...
  anchors_in_range {mamba: (0, -1)}`` and ``#1035c ZERO-ANSWER PARTITION
  cause=CAPPED ... claimed=0 by=mamba`` -- the KV-row worker asked its own
  store for the mamba anchor, which is the host's state (the worker's mamba
  pool is byteless, its store holds none), so its KV claim was capped to 0,
  the group MIN was 0, ``#1471w SETTLE-NO-WRITER remainder=16919``, W50.
* TP1/TP2 136x ``WEG2-TAIL stage refused rid=... (IndexError: list index out
  of range)`` -- ``held_shapes`` read the QSA compressed row of a worker whose
  compressed list is empty (no indexer on the worker, ``qsa_index_on_rank``).

The fix keeps S3h's rule: the worker answers for its KV rows only; the host's
pools (mamba anchor, sidecars) are the host's to answer. The host itself is
unchanged: its claim still stops at its deepest anchor.

Hermetic: a tempdir file store with the metal's 264 KV pages, CPU tensors,
stand-in controllers. RED on ca2a9706ec, GREEN with the fix.
"""

from __future__ import annotations

import tempfile
from types import SimpleNamespace

import pytest
import torch

from sglang.srt import rank_role
from sglang.srt.managers import cache_controller as cc
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.weg2 import tail_adopt as ta

KV_PAGES = 264  # the metal: '#1028B FETCH CAP n=4: kv=264 claimed=0 lost=264'
KEYS = [f"b5{i:06d}" for i in range(KV_PAGES)]


@pytest.fixture
def kv_worker(monkeypatch):
    """This rank is a Form A worker that owns token rows under the cut."""
    monkeypatch.setattr(rank_role, "form_a_worker_holds_kv", lambda: True)


@pytest.fixture
def host(monkeypatch):
    monkeypatch.setattr(rank_role, "form_a_worker_holds_kv", lambda: False)


def _store(d):
    cfg = HiCacheStorageConfig(
        tp_rank=1, tp_size=1, pp_rank=0, pp_size=1, attn_cp_rank=0, attn_cp_size=1,
        is_mla_model=False, enable_storage_metrics=False, is_page_first_layout=False,
        model_name="unit-b5-239",
    )
    store = HiCacheFile(storage_config=cfg, file_path=d)
    for k in KEYS:  # the worker's KV rows are in its store; no mamba blob anywhere
        store.set(k, torch.zeros(64, dtype=torch.uint8))
    return store


def _mamba():
    return PoolTransfer(name=PoolName.MAMBA, keys=["__placeholder__"],
                        hit_policy=PoolHitPolicy.TRAILING_PAGES)


def _query(store, transfers):
    me = SimpleNamespace(
        get_hash_str=lambda *a, **k: list(KEYS),
        storage_backend=store,
        page_size=1,
        extra_host_mem_release_entries=None,
        _draft_presence_transfer=lambda: None,
    )
    op = SimpleNamespace(
        is_terminated=lambda: False, token_ids=list(range(KV_PAGES)), last_hash=None,
        prefix_keys=None, pool_transfers=transfers, request_id="weg2-0-6",
        pool_storage_result=PoolTransferResult.empty(),
    )
    return HybridCacheController._storage_hit_query(me, op), op


# ------------------------------------------------------------------ the claim
def test_the_kv_worker_claims_its_264_kv_pages_not_zero(kv_worker):
    with tempfile.TemporaryDirectory() as d:
        (keys, tokens), op = _query(_store(d), [_mamba()])
    assert tokens == KV_PAGES and len(keys) == KV_PAGES  # metal: claimed=0
    # the host's anchor is reported present at the KV end on this rank, so the
    # group MIN of the mamba slot is the host's own answer, never this rank's 0
    assert op.pool_storage_result.kv_hit_pages == KV_PAGES


def test_the_host_without_an_anchor_is_still_capped(host):
    with tempfile.TemporaryDirectory() as d:
        (_keys, tokens), _op = _query(_store(d), [_mamba()])
    assert tokens == 0  # unchanged: the host's claim stops at its deepest anchor


def test_split_is_identity_off_the_kv_worker(host):
    xs = [_mamba()]
    assert cc.split_host_state_pools(SimpleNamespace(storage_backend=None), xs) == (xs, [])


def test_split_hands_every_non_kv_pool_to_the_host(kv_worker):
    kv = PoolTransfer(name=PoolName.KV, keys=["a"])
    m = _mamba()
    own, host_pools = cc.split_host_state_pools(SimpleNamespace(storage_backend=None), [kv, m])
    assert own == [kv] and host_pools == [m]
    # a byteless tier abstains in both arms -- it is not a KV-row worker
    null = SimpleNamespace(storage_backend=SimpleNamespace(abstains_from_claim_vote=True))
    assert cc.split_host_state_pools(null, [m]) == ([m], [])


# ------------------------------------------------------------------ probes
def _ctl_with_mamba_pool():
    return SimpleNamespace(storage_backend=SimpleNamespace(registered_pools={PoolName.MAMBA: object()}))


def test_presence_probe_asks_no_anchor_of_a_kv_worker(kv_worker):
    # store_presence_pages, the #1416 told clamp and the #257 anchor reach all
    # read this list; on a KV-row worker it held the host's anchor -> 0
    assert cc.HiCacheController._presence_pool_transfers(_ctl_with_mamba_pool()) is None


def test_presence_probe_on_the_host_is_unchanged(host):
    out = cc.HiCacheController._presence_pool_transfers(_ctl_with_mamba_pool())
    assert [t.name for t in out] == [PoolName.MAMBA]


# ------------------------------------------------------------------ the read
class _RecordingStore:
    abstains_from_claim_vote = False

    def __init__(self):
        self.gets, self.sets = [], []

    def batch_get_v2(self, transfers, extra_info=None):
        self.gets.append([t.name for t in transfers])
        return {t.name: [True] * len(t.keys or []) for t in transfers}

    def batch_set_v2(self, transfers, extra_info=None):
        self.sets.append([t.name for t in transfers])
        return {t.name: [True] * len(t.keys or []) for t in transfers}


def _hybrid(store):
    me = object.__new__(HybridCacheController)
    me.storage_backend = store
    me.page_size = 1
    return me


def _op():
    return SimpleNamespace(
        hash_value=list(KEYS), completed_tokens=0, pool_transfers=[_mamba()],
        pool_storage_result=PoolTransferResult.empty(), pool_transfers_done=False,
    )


def test_the_worker_moves_no_host_pool_and_the_group_min_stays_the_hosts(kv_worker, monkeypatch):
    monkeypatch.setattr(cc.HiCacheController, "_page_transfer",
                        lambda self, op: setattr(op, "completed_tokens", len(op.hash_value)))
    store = _RecordingStore()
    op = _op()
    HybridCacheController._page_transfer(_hybrid(store), op)
    assert store.gets == []  # nothing read for the byteless anchor
    assert op.pool_storage_result.extra_pool_hit_pages[PoolName.MAMBA] == 1  # the null tier's 1
    assert op.pool_transfers_done


def test_the_worker_never_writes_the_hosts_anchor(kv_worker, monkeypatch):
    monkeypatch.setattr(cc.HiCacheController, "_page_backup", lambda self, op: None)
    store = _RecordingStore()
    op = _op()
    HybridCacheController._page_backup(_hybrid(store), op)
    assert store.sets == []


def test_the_host_still_reads_its_anchor(host, monkeypatch):
    monkeypatch.setattr(cc.HiCacheController, "_page_transfer",
                        lambda self, op: setattr(op, "completed_tokens", len(op.hash_value)))
    store = _RecordingStore()
    HybridCacheController._page_transfer(_hybrid(store), _op())
    assert store.gets == [[PoolName.MAMBA]]


# ------------------------------------------------------------------ the tail
TP1 = (2, 0, 1, 1)  # F14 OWNER-ROWS page 64, S=2, rows [0, 1)


def _worker_pool():
    from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

    pool = object.__new__(QSATokenToKVPool)
    pool.full_kv_pool = SimpleNamespace(k_buffer=[torch.zeros(8, 2, 4)], v_buffer=[torch.zeros(8, 2, 4)])
    pool.full_attention_layer_id_mapping = {3: 0}
    pool.qsa_compress_ratio = 4
    pool.qsa_compressed_k_buffer_pool = []  # qsa_index_on_rank=False: no indexer on a worker
    pool.qsa_key_state_buffer_pool = [torch.zeros(8, 1, 4)]
    pool.qsa_rope_position_buffer = torch.zeros(8, 3, dtype=torch.int64)
    return pool


def test_a_cut_worker_holds_kv_rows_only_no_index_error(monkeypatch):
    from sglang.srt.distributed import utils as du

    monkeypatch.setattr(du, "uneven_dcp_active", lambda: True)
    monkeypatch.setattr(ta, "cut_owner", lambda: TP1)
    held = ta.held_shapes(_worker_pool(), SimpleNamespace())  # metal: IndexError
    assert len(held.fa[3]) == 2 and held.qsa_ratio == 0
    assert held.owner == TP1 and ta.cut_gate(held) == ""  # E1 by owner rows
    assert held.ring  # the host's ring stays named: E2 is refused 'cut_ring_on_worker'


def test_a_cut_worker_takes_kv_of_ps_three_row_part(monkeypatch):
    from sglang.srt.weg2 import tail_handoff as th

    held = ta.HeldShapes(fa={3: [ta.RowSpec(shape=[2, 4], dtype="torch.float32")] * 2}, gdn={},
                         qsa_ratio=0, dcp=True, owner=TP1)
    spec = th.TailSpec(rid="weg2-0-6", n_tokens=16918, page_prefix=16896, cut=16904, key="k")
    rows = spec.rows
    hdr = th.TailHeader(spec=spec, part="pp0", fa_layers=[3], gdn_layers=[],
                        fa_row_shapes={"3": [2, 4]}, gdn_row_shapes={}, fa_digest="", gdn_digest="",
                        nbytes=0)
    k, v = torch.ones(rows, 2, 4), -torch.ones(rows, 2, 4)
    comp = torch.zeros(rows // 4, 1, 4)  # P holds the indexer: K, V and the compressed row
    monkeypatch.setattr(th, "read_part", lambda h, check_digest=False: ({"fa": {"3": (k, v, comp)}, "gdn": {}}, ""))
    st = ta._stage_e1([hdr], held, False, [])
    assert st.verdict == "ready", st.verdict  # base: fa_arity:3:3!=2
    assert len(st.fa[3]) == 2 and torch.equal(st.fa[3][0], k)
