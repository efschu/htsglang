"""F22 Flipzeit (29.09.): the owner load under the token cut takes whole pages.

MEASURED (marker audit x178 vs z30w-park vs z30x2-kvdemand, 29.09.): the
posten that x178 did not have is the Form A worker's loadback after the wake.
TP0's ``prepare_ms`` of the first post-wake pass equals the slowest worker's
``WEG2-START-LOADING kv_issue_ms`` flip by flip (kvdemand 3717 / 3734 ms,
1967 / 1988, 682 / 716; z30w-park median 998 / 1033 ms, n=56) -- TP0 waits in
``prepare_for_extend`` for the workers. x178 (no cut): TP0 page load
``mode=dma``, ``kv=5`` ms. Under the cut (#239 S4b F13 3/7, 89164b8a59) a
worker's owner rows never take the whole-page path; ``_transfer_paged`` runs a
CPU fancy-index gather of (slot, token) cells PER LAYER out of the mapped arena
and a BLOCKING pageable H2D per K and V (TP1 kvdemand: 213096 rows in 3.73 s).

The fix: when the owner rows are whole owner groups (every page's owned lanes,
in page order -- the shape a page-aligned loadback hands over), the pages go
whole into the device stage like TP0's page load (dma / kernel / cpu) and only
the owned lanes are scattered on the device; a tail that is not whole groups,
and any other shape (the #1424 duplicate-row case), keeps the gather.

Hermetic: a tmp arena file, no CUDA. RED on 895559fed2: the owner branch
always calls ``_transfer_paged``, once per layer, with every row.
"""

import os
import shutil
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch

from sglang.srt.mem_cache.canonical_kv_page import CanonicalPageSpec
from sglang.srt.mem_cache.canonical_page_store import (
    CanonicalPageWindow,
    owner_row_window,
    owner_token_runs,
)
from sglang.srt.mem_cache.pool_host.arena_pool import ArenaMHAHostPool
from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

L, H, D = 2, 1, 2          # attention layers, kv heads, head dim (uint8: cell = 2 B)
CELL = H * D
P = 8                      # tokens per page
S = 4                      # the owner split: two owner runs per page
BLOCK = P * CELL
PAGE = 2 * L * BLOCK       # [K l0 | K l1 | V l0 | V l1], token-minor
STAGING = 5
NSLOTS = 8
SPEC = CanonicalPageSpec(num_attn_layers=L, kv_bytes_per_token_per_attn_layer=2 * BLOCK)
WHOLE = CanonicalPageWindow(spec=SPEC, first_slot=0, num_slots=L)
W1 = (P, S, 0, 3)          # lanes 0,1,2,4,5,6
W2 = (P, S, 3, 4)          # lanes 3,7


def _pool(arena, owner):
    p = object.__new__(ArenaMHAHostPool)
    p.layout = "layer_first"; p.page_size = P; p.layer_num = L; p.head_num = H; p.head_dim = D
    p.dtype = torch.uint8; p.device = "cpu"; p.pin_memory = False; p.size = STAGING
    p.element_dim = H * D; p.can_use_jit = True
    p.free_slots = torch.arange(STAGING, dtype=torch.int64)
    p.slot_used = torch.zeros(STAGING, dtype=torch.bool)
    p.kv_buffer = torch.zeros(2, L, STAGING, H, D, dtype=torch.uint8)
    p._arena_init_fields()
    p.bind(arena, owner_row_window(WHOLE, P, owner_token_runs(*owner)), role="kv", pin=False,
           owner_rows=owner)
    return p


def _val(kv, layer, slot, tok):
    return (1 + 7 * kv + 3 * layer + 11 * slot + tok) % 251


def _fill(pool, slots):
    for s in slots:
        page = pool._page_view[s].view(2, L, P, CELL)
        for kv in range(2):
            for l in range(L):
                for t in range(P):
                    page[kv, l, t] = _val(kv, l, s, t)


def _dev(rows):
    return types.SimpleNamespace(
        k_buffer=[torch.zeros(rows, H, D, dtype=torch.uint8) for _ in range(L)],
        v_buffer=[torch.zeros(rows, H, D, dtype=torch.uint8) for _ in range(L)])


def _owner_rows(pool, slots, lanes=None):
    lanes = pool._owner_tok.tolist() if lanes is None else lanes
    return [STAGING + s * P + t for s in slots for t in lanes]


def _load(pool, host_ids, dev_rows, counter):
    dev = _dev(max(dev_rows) + 1)
    hi, di = torch.tensor(host_ids, dtype=torch.int64), torch.tensor(dev_rows, dtype=torch.int64)
    orig = ArenaMHAHostPool._transfer_paged

    def _counted(self, device_pool, rows, device_indices, layer_id):
        counter.append((int(layer_id), int(rows.numel())))
        return orig(self, device_pool, rows, device_indices, layer_id)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ArenaMHAHostPool, "_transfer_paged", _counted)
        for l in range(L):
            pool.load_to_device_per_layer(dev, hi, di, l, "direct")
    return dev


def _expect(dev, host_ids, dev_rows):
    for h, r in zip(host_ids, dev_rows):
        s, t = divmod(h - STAGING, P)
        for l in range(L):
            assert dev.k_buffer[l][r].flatten().tolist() == [_val(0, l, s, t)] * CELL, (h, r, l)
            assert dev.v_buffer[l][r].flatten().tolist() == [_val(1, l, s, t)] * CELL, (h, r, l)


@pytest.fixture
def arena(tmp_path):
    return ShmArena(str(tmp_path / "kv.bin"), PAGE, NSLOTS)


def test_whole_owner_groups_take_the_page_load_not_the_per_layer_gather(arena):
    w1 = _pool(arena, W1)
    assert w1._owner_tok.tolist() == [0, 1, 2, 4, 5, 6]
    slots = [1, 2, 3, 6]
    _fill(w1, slots)
    host = _owner_rows(w1, slots)
    dev_rows = list(range(3, 3 + len(host)))   # compact device rows, any placement
    calls = []
    dev = _load(w1, host, dev_rows, calls)
    assert calls == []                        # RED on 895559fed2: L calls with every row
    _expect(dev, host, dev_rows)
    # nothing but the asked rows
    for l in range(L):
        assert bool((dev.k_buffer[l][:3] == 0).all()) and bool((dev.v_buffer[l][:3] == 0).all())


def test_the_minority_owner_loads_its_two_lanes_per_page(arena):
    w2 = _pool(arena, W2)
    assert w2._owner_tok.tolist() == [3, 7]
    slots = [0, 4, 5]
    _fill(w2, slots)
    host = _owner_rows(w2, slots)
    dev_rows = [9, 2, 7, 0, 4, 1]
    calls = []
    dev = _load(w2, host, dev_rows, calls)
    assert calls == []
    _expect(dev, host, dev_rows)


def test_a_tail_that_is_not_whole_groups_keeps_the_gather_for_the_tail_only(arena):
    w1 = _pool(arena, W1)
    slots = [2, 3]
    _fill(w1, slots + [4])
    host = _owner_rows(w1, slots) + _owner_rows(w1, [4], lanes=[0, 1])   # page 4: two of six lanes
    dev_rows = list(range(len(host)))
    calls = []
    dev = _load(w1, host, dev_rows, calls)
    assert calls == [(l, 2) for l in range(L)]   # the 2 tail rows, per layer; RED: every row
    _expect(dev, host, dev_rows)


def test_duplicate_rows_keep_the_gather(arena):
    """#1424 (rc12m-dpr): the same host rows loaded twice come sorted as twins
    -- not whole groups, so the gather addresses every row on its own."""
    w1 = _pool(arena, W1)
    _fill(w1, [1])
    one = _owner_rows(w1, [1])
    host = [h for h in one for _ in (0, 1)]
    dev_rows = list(range(len(host)))
    calls = []
    dev = _load(w1, host, dev_rows, calls)
    assert calls == [(l, len(host)) for l in range(L)]
    _expect(dev, host, dev_rows)


def test_registered_slots_take_the_dma_runs(arena, monkeypatch):
    """The production form: the arena pre-pinned at bind (#1436), an NF page
    above 32 KiB -> "dma" (one copy per run of consecutive slots). Forced here
    by the mode env on the desk's tiny page; the lane scatter is the same."""
    monkeypatch.setenv("SGLANG_WEG2_ARENA_PAGE_LOAD_MODE", "dma")
    w2 = _pool(arena, W2)
    w2._all_pinned = True
    slots = [2, 3, 4, 7]
    _fill(w2, slots)
    host = _owner_rows(w2, slots)
    dev_rows = list(range(len(host)))[::-1]
    calls = []
    dev = _load(w2, host, dev_rows, calls)
    assert calls == []
    _expect(dev, host, dev_rows)


# --------------------------------------------------------------- 27B: byte-equal
# 27B (29.09., z30y2 pick): uneven DCP, no Form A workers, page size 1. The
# owner rows exist only for PAGED pools (``canonical_kv_owner_rows_for``
# returns None at page 1 -- the "page-1 owner form"; 27B D log 09290020: 0 x
# "#239 F14 OWNER-ROWS"), so a 27B arena pool never holds ``_owner_tok`` and
# its load takes the whole-page path with ``lanes=None`` -- the same call and
# the same bytes as on 895559fed2.

P1, L1, H1, D1 = 1, 2, 2, 4
CELL1 = H1 * D1
PAGE1 = 2 * L1 * CELL1


class _Win27:
    total_bytes = PAGE1
    extents = ((0, L1 * CELL1), (L1 * CELL1, L1 * CELL1))


def _pool27(tmp_path):
    p = object.__new__(ArenaMHAHostPool)
    p.layout = "layer_first"; p.page_size = P1; p.layer_num = L1; p.head_num = H1; p.head_dim = D1
    p.dtype = torch.uint8; p.device = "cpu"; p.pin_memory = False; p.size = STAGING
    p.element_dim = H1 * D1; p.can_use_jit = True
    p.free_slots = torch.arange(STAGING, dtype=torch.int64)
    p.slot_used = torch.zeros(STAGING, dtype=torch.bool)
    p.kv_buffer = torch.zeros(2, L1, STAGING, H1, D1, dtype=torch.uint8)
    p._arena_init_fields()
    p.bind(ShmArena(str(tmp_path / "kv27.bin"), PAGE1, NSLOTS), _Win27(), role="kv", pin=False)
    p._pinned[:] = True
    return p


def test_27b_page_one_has_no_owner_rows():
    from sglang.srt.managers.cache_controller import canonical_kv_owner_rows_for

    assert canonical_kv_owner_rows_for((3, 0, 2), 1, object()) is None
    assert canonical_kv_owner_rows_for((3, 0, 2), 64, object()) == (64, 3, 0, 2)  # NF: paged


def test_27b_load_is_the_unchanged_whole_page_call_and_bytes(tmp_path):
    p = _pool27(tmp_path)
    assert p._owner_tok is None and p._arena_page_tokens == 1
    slots = [0, 3, 4, 7]
    for s in slots:
        page = p._page_view[s].view(2, L1, CELL1)
        for kv in range(2):
            for l in range(L1):
                page[kv, l] = _val(kv, l, s, 0)
    host = [STAGING + s for s in slots]
    dev_rows = [6, 1, 4, 2]
    dev = types.SimpleNamespace(
        k_buffer=[torch.zeros(8, H1, D1, dtype=torch.uint8) for _ in range(L1)],
        v_buffer=[torch.zeros(8, H1, D1, dtype=torch.uint8) for _ in range(L1)])
    seen, owner_calls = [], []
    orig = ArenaMHAHostPool._load_pages_all_layers

    def _spy(self, device_pool, slots_, device_indices, *a, **k):
        seen.append((slots_.tolist(), a, k))
        return orig(self, device_pool, slots_, device_indices, *a, **k)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ArenaMHAHostPool, "_load_pages_all_layers", _spy)
        mp.setattr(ArenaMHAHostPool, "_owner_page_prefix_load",
                   lambda *a, **k: owner_calls.append(1) or 0, raising=False)
        hi, di = torch.tensor(host), torch.tensor(dev_rows)
        for l in range(L1):
            p.load_to_device_per_layer(dev, hi, di, l, "direct")
    assert owner_calls == []
    assert seen == [(slots, (), {})]      # one page load at layer 0, called exactly as on the base
    for s, r in zip(slots, dev_rows):
        for l in range(L1):
            assert dev.k_buffer[l][r].flatten().tolist() == [_val(0, l, s, 0)] * CELL1
            assert dev.v_buffer[l][r].flatten().tolist() == [_val(1, l, s, 0)] * CELL1


def test_page_load_switch_off_keeps_the_gather(arena, monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_ARENA_PAGE_LOAD", "0")
    w1 = _pool(arena, W1)
    _fill(w1, [5])
    host = _owner_rows(w1, [5])
    dev_rows = list(range(len(host)))
    calls = []
    dev = _load(w1, host, dev_rows, calls)
    assert calls == [(l, len(host)) for l in range(L)]
    _expect(dev, host, dev_rows)
