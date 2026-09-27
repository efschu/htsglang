"""#1424 (rc12m-dpr D-TP0 11:49:05): a batch loading the SAME host rows for
two requests died in the paged arena load.

weg2-17-71 and weg2-11-56 both loaded the 384-token node [22656, 23040)
(START-LOADING nodes=3 tokens=37376 = 36608 + 384 + 384). ``move_indices``
(io_backend direct, layer_first) sorts the merged host indices, so each
duplicated row sits next to its twin -- r0,r0,r1,r1,... -- and
``_page_slots`` raised 'a page's token rows are not consecutive from its
first id'. Loading one row into two device rows is legal; only the
whole-page fast path cannot express it. The control batch at 11:45:38
(weg2-0-3 + weg2-5-28, same prefix 34304) survived because the second
request found the shared nodes already on the device and loaded only its
own 3904 tokens -- no duplicate rows. 27B's arena is unpaged (P == 1).

Hermetic: the real ``ArenaMHAHostPool._load_arena`` and ``_transfer_paged``
on a bare instance with CPU tensors."""
from __future__ import annotations

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from sglang.srt.mem_cache.pool_host import arena_pool as ap  # noqa: E402

P = 4
SLOTS = 3
CELL = 2
LAYERS = 2


def _pool():
    pool = object.__new__(ap.ArenaMHAHostPool)
    pool._arena_page_tokens = P
    pool._page_view = object()
    pool._page_key_hint = None
    pool._page_loaded_key = None
    pool.row_slot = None
    # arena[layer][slot, token, cell] = distinct values per row
    base = torch.arange(SLOTS * P * CELL, dtype=torch.float32).view(SLOTS, P, CELL)
    pool.arena_k_refs = [base + 1000 * layer for layer in range(LAYERS)]
    pool.arena_v_refs = [-(base + 1000 * layer) for layer in range(LAYERS)]
    pool.page_loads = []
    pool.pin_slots = lambda slots: 0
    pool._arena_load_guard = lambda *a, **k: None
    pool._load_pages_all_layers = lambda dp, slots, di: pool.page_loads.append(slots.tolist())
    return pool


def _device(n):
    return types.SimpleNamespace(
        k_buffer=[torch.zeros(n, CELL) for _ in range(LAYERS)],
        v_buffer=[torch.zeros(n, CELL) for _ in range(LAYERS)],
    )


def test_metal_shape_duplicated_rows_after_the_sort_load_row_by_row():
    """RED on b6e2235d8a: RuntimeError '#1424 paged arena load ...'."""
    pool = _pool()
    page = torch.arange(P, P + P)                      # slot 1, rows 4..7
    rows, _ = torch.cat([page, page]).sort()           # 4,4,5,5,6,6,7,7
    dev_idx = torch.arange(2 * P)
    dp = _device(2 * P)
    for layer in range(LAYERS):
        pool._load_arena(dp, rows, dev_idx, layer)
    for layer in range(LAYERS):
        want = pool.arena_k_refs[layer].view(-1, CELL)[rows]
        assert torch.equal(dp.k_buffer[layer], want)
        assert torch.equal(dp.v_buffer[layer], pool.arena_v_refs[layer].view(-1, CELL)[rows])
    assert pool.page_loads == [], "the whole-page path cannot express duplicated rows"


def test_whole_consecutive_pages_keep_the_page_path():
    pool = _pool()
    rows = torch.arange(0, 2 * P)                      # slots 0 and 1, whole pages
    pool._load_arena(_device(2 * P), rows, torch.arange(2 * P), 0)
    assert pool.page_loads == [[0, 1]]


def test_the_fallback_is_named():
    pool = _pool()
    rows, _ = torch.cat([torch.arange(P), torch.arange(P)]).sort()
    before = ap._PAGE_FALLBACK_N[0]
    pool._load_arena(_device(2 * P), rows, torch.arange(2 * P), 0)
    assert ap._PAGE_FALLBACK_N[0] == before + 1
