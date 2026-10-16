"""K12 index 72085: the QSA mirror row a backup asked for is the A2 ORPHAN id.

D TP0, boot ..._226aad7105_1015_124245, D.log:
* 271588 (13:35:33) ``A2 LOST-ROWS n=1 pages=30 repointed=6 orphaned=24`` -- a
  late write ack: ``ArenaMHAHostPool.settle_lost_rows`` moved 24 pages of the
  node's KV host ids to the ORPHAN id ``id_space`` = staging (4096) + arena
  tokens (6485 * 64) + PLACEHOLDERS (2**22), "deliberately outside every pool";
* 271625-271645 the store write of that node: the ``qsa_indexer`` sidecar is
  derived from the KV host ids (``indices_from_pool=KV``), ``get_data_page``
  divides by 64 -> row (4096 + 415040 + 4194304) / 64 = **72085** of a mirror
  of (4096 + 415040) / 64 = **6549** rows (D.log 5589) -> IndexError, the backup
  thread died, W120 at the sleep 10 s later (K12).

The mirror is sized right -- it covers exactly the KV host id space. The id is
the A2 sentinel, and the store-write issuers did not know it. RED on
97e49033c2 / 226aad7105: the orphaned node's store write is issued and its
sidecar transfer raises. GREEN: the issuers refuse such a node by name (no
host bytes, nothing to write), and the file backend refuses an out-of-range
paged-mirror row by name instead of raising.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.mem_cache import hicache_storage as hs  # noqa: E402
from sglang.srt.mem_cache.pool_host import arena_pool as ap  # noqa: E402
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache as U  # noqa: E402

PAGE = 64
STAGING = 4096
SLOTS = 6485
ARENA_TOKENS = SLOTS * PAGE
ORPHAN = STAGING + ARENA_TOKENS + ap.PLACEHOLDERS  # settle_lost_rows' id_space
MIRROR_ROWS = (STAGING + ARENA_TOKENS) // PAGE


def test_the_numbers_of_the_death():
    assert MIRROR_ROWS == 6549
    assert ORPHAN // PAGE == 72085


class _Tree:
    write_backup_storage = U.write_backup_storage
    _weg2_write_plain_sidecars = U._weg2_write_plain_sidecars
    _plain_sidecar_transfers = staticmethod(U._plain_sidecar_transfers)
    _k12_unreal_host_rows = getattr(U, "_k12_unreal_host_rows", None)
    _k12_refuse_unreal_backup = getattr(U, "_k12_refuse_unreal_backup", None)

    def __init__(self):
        self.enable_storage = True
        self.hicache_storage_pass_prefix_keys = False
        self._components_tuple = ()
        self.staging_write_ring = None
        self.ongoing_backup = {}
        self.writes = []
        self.cache_controller = SimpleNamespace(
            mem_pool_host=SimpleNamespace(arena=object(), staging_rows=STAGING,
                                          arena_tokens=ARENA_TOKENS),
            page_owner_mask_ctx=lambda: None,
            write_storage=lambda hv, *a, **k: self.writes.append(hv) or 99,
            storage_backend=SimpleNamespace(registered_pools={}),
        )

    def _build_sidecar_transfers(self, phase, kv_xfer, comp_xfers):
        # the QSA mirror transfer, derived from the KV host ids
        return [SimpleNamespace(name="qsa_indexer", host_indices=kv_xfer.host_indices,
                                keys=kv_xfer.keys)]

    def inc_host_lock_ref(self, node):
        return SimpleNamespace(to_dec_params=lambda: None)


class _Key:
    def __init__(self, ids):
        self.token_ids = ids
        self.extra_key = None

    def __len__(self):
        return len(self.token_ids)


def _node(hv):
    return SimpleNamespace(
        id=7, backuped=True, hash_value=[f"h{i}" for i in range(hv.numel() // PAGE)],
        key=_Key(list(range(hv.numel()))),
        component_data=[SimpleNamespace(host_value=hv)], parent=None,
    )


def _orphaned_hv():
    """Two pages: one real arena page, one A2-orphaned page."""
    real = torch.arange(STAGING + 5 * PAGE, STAGING + 6 * PAGE)
    orphan = torch.full((PAGE,), ORPHAN, dtype=torch.int64)
    return torch.cat([real, orphan])


def test_an_orphaned_node_is_not_written_to_the_store():
    t = _Tree()
    assert t._k12_refuse_unreal_backup is not None, "K12 orphan guard absent"
    t.write_backup_storage(_node(_orphaned_hv()))
    assert t.writes == [] and t.ongoing_backup == {}


def test_an_orphaned_node_gets_no_plain_sidecar_write():
    t = _Tree()
    assert t._k12_refuse_unreal_backup is not None, "K12 orphan guard absent"
    t.cache_controller.storage_backend.registered_pools = {"qsa_indexer": SimpleNamespace()}
    t._weg2_write_plain_sidecars(_node(_orphaned_hv()))
    assert t.writes == []


def test_a_real_node_is_still_written():
    t = _Tree()
    if t._k12_refuse_unreal_backup is None:
        pytest.skip("guard absent (base): the positive case is the old behaviour")
    hv = torch.arange(STAGING, STAGING + 2 * PAGE)
    t.write_backup_storage(_node(hv))
    assert len(t.writes) == 1 and 99 in t.ongoing_backup


def _file_batch_set_v2():
    for name in dir(hs):
        cls = getattr(hs, name)
        fn = getattr(cls, "batch_set_v2", None) if isinstance(cls, type) else None
        if fn is not None and "get_data_pages" in getattr(getattr(fn, "__code__", None), "co_consts", ()):
            return fn
    raise AssertionError("no file-backend batch_set_v2 found")


def test_the_file_backend_refuses_an_out_of_range_mirror_row_by_name():
    """RED on the base: the IndexError of D.log 271645 escapes."""
    mirror = SimpleNamespace(page_size=PAGE, slot_page_size=PAGE, num_host_pages=MIRROR_ROWS,
                             get_data_pages=lambda starts: [
                                 (_ for _ in ()).throw(IndexError(
                                     f"index {s // PAGE} is out of bounds for dimension 0 with size {MIRROR_ROWS}"))
                                 if s // PAGE >= MIRROR_ROWS else torch.zeros(1) for s in starts])
    backend = SimpleNamespace(
        registered_pools={"qsa_indexer": mirror},
        _log_key=lambda name, key: f"{key}.{name}",
        _batch_set_each=lambda keys, pages: [True] * len(keys),
        _qsa_sidecar_trace=lambda *a, **k: None,
    )
    xfer = SimpleNamespace(name="qsa_indexer", keys=["k0"],
                           host_indices=torch.full((PAGE,), ORPHAN, dtype=torch.int64))
    res = _file_batch_set_v2()(backend, [xfer])
    assert res == {"qsa_indexer": [False]}
    ok = SimpleNamespace(name="qsa_indexer", keys=["k1"],
                         host_indices=torch.arange(STAGING, STAGING + PAGE))
    assert _file_batch_set_v2()(backend, [ok]) == {"qsa_indexer": [True]}


# ---------------------------------------------------------------- review k12c A1
def test_a_staging_node_is_written():
    """Review M3: a node on STAGING rows [0, 2P) (every non-direct write-through
    lands there) must still be written -- a guard that also flags rows < S would
    silently drop every ordinary store write."""
    t = _Tree()
    t.write_backup_storage(_node(torch.arange(0, 2 * PAGE)))
    assert len(t.writes) == 1 and 99 in t.ongoing_backup


@pytest.mark.parametrize("where", ["first", "middle"])
def test_an_orphan_in_the_first_or_a_middle_page_is_refused(where):
    """Review M2/M4: the guard reads EVERY id, not one end of the node."""
    real = lambda lo: torch.arange(lo, lo + PAGE)  # noqa: E731
    orphan = torch.full((PAGE,), ORPHAN, dtype=torch.int64)
    pages = [real(STAGING), real(STAGING + PAGE), real(STAGING + 2 * PAGE)]
    pages[0 if where == "first" else 1] = orphan
    t = _Tree()
    t.write_backup_storage(_node(torch.cat(pages)))
    assert t.writes == [] and t.ongoing_backup == {}


def test_the_belt_at_row_num_host_pages_returns_false_without_raising():
    """Review: the off-by-one direction -- row == num_host_pages (6549) is past
    the mirror; row 6548 is its last real row."""
    def page(s):
        if s // PAGE >= MIRROR_ROWS:
            raise IndexError(f"index {s // PAGE} is out of bounds for dimension 0 with size {MIRROR_ROWS}")
        return torch.zeros(1)

    mirror = SimpleNamespace(page_size=PAGE, slot_page_size=PAGE, num_host_pages=MIRROR_ROWS,
                             get_data_pages=lambda starts: [page(s) for s in starts])
    backend = SimpleNamespace(
        registered_pools={"qsa_indexer": mirror},
        _log_key=lambda name, key: f"{key}.{name}",
        _batch_set_each=lambda keys, pages: [True] * len(keys),
        _qsa_sidecar_trace=lambda *a, **k: None,
    )
    fn = _file_batch_set_v2()
    at = SimpleNamespace(name="qsa_indexer", keys=["k"],
                         host_indices=torch.arange(MIRROR_ROWS * PAGE, (MIRROR_ROWS + 1) * PAGE))
    assert fn(backend, [at]) == {"qsa_indexer": [False]}
    last = SimpleNamespace(name="qsa_indexer", keys=["k"],
                           host_indices=torch.arange((MIRROR_ROWS - 1) * PAGE, MIRROR_ROWS * PAGE))
    assert fn(backend, [last]) == {"qsa_indexer": [True]}


def test_the_first_id_past_the_real_rows_is_refused():
    """Review M1 (``>`` for ``>=``): id S + A is the first non-row (the first
    read placeholder) -- refused even when it is the only one."""
    hv = torch.arange(STAGING, STAGING + 2 * PAGE)
    hv[PAGE] = STAGING + ARENA_TOKENS
    t = _Tree()
    t.write_backup_storage(_node(hv))
    assert t.writes == []
