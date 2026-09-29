"""#760 (rc12z4-vis, D-Log 06:24:34-06:24:47): a Form A expert worker's KV
host pool is byteless (0 kv-heads, JG) and, under R12, keeps every row until
the reset. #249 lets such a pool GROW its id space instead of refusing:

    [06:24:34 TP1/TP2] #249 BYTELESS-GROW pool=MHATokenToKVPoolHost
                       rows 353600 -> 436032 (+82432, 0 B each; synced 353600)

Thirteen seconds later both workers died in the backup of the finished
pdflip-0-14 (``#1469 RETAIN ... token_ids_len=62091``):

    KvTransferShapeMismatch: #760 ... dst indices out of bounds:
    max=365887 >= capacity 353600

365887 = 353600 + 12287 is a row of the GROWN range -- the worker's own,
legal id. ``_regrow_byteless_buffer`` re-shaped ``kv_buffer``, but the
per-layer views ``k_data_refs``/``v_data_refs`` (and their pointer vectors),
cut once in ``__init__``, still viewed the old 353600-row buffer, and the #760
seam guard reads its host capacity from exactly those views.

Fix: one binding of views + pointers, used by ``__init__`` and by every
byteless re-shape (grow, and the ``clear()`` back to the synced size).

Built on the real ``MHATokenToKVPoolHost`` bookkeeping (alloc, #249 grow,
clear, backup + #760 guard); only the constructor's allocation plumbing is
bypassed, and the state it leaves (views of the initial buffer) is set up the
way ``__init__`` sets it."""

from __future__ import annotations

import os
import threading
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.mem_cache.kv_transfer_guard import KvTransferShapeMismatch  # noqa: E402
from flliper.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost  # noqa: E402

SYNCED = 353600      # the worker's synced size = TP0 staging 4096 + 5461 slots x 64
GROW = 82432         # rc12z4-vis: 353600 -> 436032
BACKUP = 12288       # the backup whose highest host row was 365887
PAGE = 64
LAYERS = 2


def _post_init_views(p):
    """What ``MHATokenToKVPoolHost.__init__`` leaves: per-layer views of the
    buffer it was constructed with, and their pointer vectors."""
    p.k_data_refs = [p.kv_buffer[0][i] for i in range(p.layer_num)]
    p.v_data_refs = [p.kv_buffer[1][i] for i in range(p.layer_num)]
    p.k_data_ptrs = torch.tensor([x.data_ptr() for x in p.k_data_refs], dtype=torch.uint64)
    p.v_data_ptrs = torch.tensor([x.data_ptr() for x in p.v_data_refs], dtype=torch.uint64)


def _worker_pool() -> MHATokenToKVPoolHost:
    """Form A worker KV host pool: 0 kv-heads, layer_first (the layout whose
    host capacity the #760 guard checks), every synced row held (R12)."""
    p = object.__new__(MHATokenToKVPoolHost)
    p.page_size = PAGE
    p.size = SYNCED
    p.page_num = SYNCED // PAGE
    p.byteless = True
    p.lock = threading.RLock()
    p.layout = "layer_first"
    p.layer_num, p.head_num, p.head_dim = LAYERS, 0, 8
    p.dtype = torch.bfloat16
    p.device = "cpu"
    p.budget_label = "form-a-worker-kv"
    p.token_stride_size = 0
    p.device_pool = types.SimpleNamespace(device="cpu")
    p.kv_buffer = torch.empty((2, LAYERS, SYNCED, 0, 8), dtype=p.dtype)
    _post_init_views(p)
    p.clear()
    assert p.alloc(SYNCED) is not None  # R12: the worker keeps every row
    return p


def _device_pool(rows: int):
    """The worker's device KV pool: also 0 kv-heads."""
    return types.SimpleNamespace(
        k_buffer=[torch.empty((rows, 0, 8), dtype=torch.bfloat16) for _ in range(LAYERS)],
        k_data_ptrs=torch.zeros(LAYERS, dtype=torch.uint64),
        v_data_ptrs=torch.zeros(LAYERS, dtype=torch.uint64),
        token_stride_size=0,
    )


def _grow_and_take(p):
    ids = p.alloc(GROW)
    assert ids is not None
    assert p.size == SYNCED + GROW == 436032
    return ids


def test_rc12z4_vis_backup_into_grown_row_365887():
    """RED on 0eb7829efc: the backup of rows 353600..365887 (all the worker's
    own, allocated from its grown range) dies at the #760 guard with
    'max=365887 >= capacity 353600'. GREEN: the guard sees the grown capacity
    and the byteless backup returns."""
    p = _worker_pool()
    ids = _grow_and_take(p)
    host = ids[:BACKUP]
    assert int(host.max()) == 365887
    dev = torch.arange(BACKUP, dtype=torch.int64)
    p.backup_from_device_all_layer(_device_pool(BACKUP), host, dev, "kernel")


def test_views_follow_every_byteless_reshape():
    """The views' row count IS the pool's id space after the grow and after
    the clear() back to the synced size -- never the buffer of another size."""
    p = _worker_pool()
    _grow_and_take(p)
    assert p.kv_buffer.shape[2] == 436032
    assert {int(r.shape[0]) for r in p.k_data_refs + p.v_data_refs} == {436032}
    assert int(p.k_data_ptrs.numel()) == LAYERS == int(p.v_data_ptrs.numel())
    p.clear()
    assert p.size == SYNCED
    assert {int(r.shape[0]) for r in p.k_data_refs + p.v_data_refs} == {SYNCED}


def test_backup_does_not_touch_the_allocation_census():
    """The fix re-binds views only: which rows are allocated (slot_used), the
    free list and the size -- the length the worker votes and follows TP0 with
    (H98/#249) -- are exactly what alloc left. TP0 and the workers cannot come
    apart through this path."""
    p = _worker_pool()
    ids = _grow_and_take(p)
    used = p.slot_used.clone()
    free = p.free_slots.clone()
    size = p.size
    p.backup_from_device_all_layer(
        _device_pool(BACKUP), ids[:BACKUP], torch.arange(BACKUP), "kernel"
    )
    assert p.size == size
    assert torch.equal(p.slot_used, used)
    assert torch.equal(p.free_slots, free)
    assert int(p.slot_used.sum()) == SYNCED + GROW
    assert p.available_size() == 0


def test_a_genuinely_foreign_row_is_still_refused():
    """The guard is not loosened: a row beyond the GROWN id space (never
    allocated by this pool) still dies by name."""
    p = _worker_pool()
    _grow_and_take(p)
    host = torch.tensor([SYNCED + GROW], dtype=torch.int64)
    with pytest.raises(KvTransferShapeMismatch, match="436032"):
        p.backup_from_device_all_layer(_device_pool(1), host, torch.arange(1), "kernel")
