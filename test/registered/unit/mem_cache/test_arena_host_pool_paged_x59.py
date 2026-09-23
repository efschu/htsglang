"""x59 (23.09.2026, Task #107): the arena host pool is PAGED -- one arena slot
per page of P tokens, so Next Flash (page 64) rides ONE shared L2 like the 27B
(P == 1) instead of six private 4-GB pinned pools (fnFL2x23's fallback).

Hermetic like test 1424: a tmp arena file, no CUDA, torch paths. What must
hold: ids count tokens (S + slot*P + t), a resolved run is P ids per slot, a
page load lands every token of this rank's layers on the device rows of its
page, a direct write puts this rank's K/V blocks into the slot at the
canonical offsets, and the draft role refuses pages.
"""
import os
import shutil
import threading
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch

from sglang.srt.mem_cache.pool_host import arena_pool as ap
from sglang.srt.mem_cache.pool_host.arena_pool import PLACEHOLDERS, ArenaMHAHostPool
from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(__file__)

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

P, L, LT, H, D = 4, 2, 3, 2, 4   # page tokens, local layers, model layers, heads, head_dim (uint8)
CELL = H * D                     # one token's K (or V) bytes of one layer
BLOCK = P * CELL                 # one layer's K block of a page
PAGE = 2 * LT * BLOCK            # the canonical page: [K L0 L1 L2][V L0 L1 L2], token-minor
S = 8                            # staging rows (two pages)
A = 6                            # arena slots


class _Win:
    """This rank holds model layers 1..2 of 3: K at 1*BLOCK, V at PAGE/2 + 1*BLOCK."""
    total_bytes = PAGE
    extents = ((1 * BLOCK, L * BLOCK), (PAGE // 2 + 1 * BLOCK, L * BLOCK))


class _Backend:
    def _get_suffixed_key(self, h):
        return f"{h}.sfx"

    def _arena_evict_to_disk(self, arena, want):
        return 0


def _pool(tmp_path):
    p = object.__new__(ArenaMHAHostPool)
    p.layout = "layer_first"; p.page_size = P; p.layer_num = L; p.head_num = H; p.head_dim = D
    p.dtype = torch.uint8; p.device = "cpu"; p.pin_memory = False; p.size = S
    p.element_dim = H * D; p.can_use_jit = False; p.token_stride_size = CELL
    p.lock = threading.Lock()
    p.free_slots = torch.arange(S, dtype=torch.int64); p.slot_used = torch.zeros(S, dtype=torch.bool)
    p.kv_buffer = torch.zeros(2, L, S, H, D, dtype=torch.uint8)
    p._arena_init_fields()
    arena = ShmArena(str(tmp_path / "kv.bin"), PAGE, A)
    p.bind(arena, _Win(), role="kv", pin=False)
    p._pinned[:] = True  # the load guard reads the registration bitmap; no CUDA here
    p._backend = _Backend()
    return p, arena


def _page_bytes(tag):
    return ((torch.arange(PAGE) + tag) % 251).to(torch.uint8)


def _write_page(arena, stem, tag):
    pay = _page_bytes(tag)
    assert arena.write([stem], [PAGE], [((0, PAGE),)], [pay.data_ptr()]) == [1]


def _dev_pool(rows=32):
    return types.SimpleNamespace(
        k_buffer=[torch.zeros(rows, H, D, dtype=torch.uint8) for _ in range(L)],
        v_buffer=[torch.zeros(rows, H, D, dtype=torch.uint8) for _ in range(L)],
    )


def test_ids_count_tokens_and_a_resolve_is_p_ids_per_slot(tmp_path):
    p, arena = _pool(tmp_path)
    assert p.arena_tokens == A * P
    assert p.id_space == S + A * P + PLACEHOLDERS
    assert p.prefetch_capacity_tokens == A * P
    assert p.is_arena_id(S) and p.is_arena_id(S + A * P - 1)
    assert p.is_placeholder(S + A * P) and not p.is_arena_id(S - 1)
    assert p.arena_ids([2]).tolist() == [S + 8, S + 9, S + 10, S + 11]
    ph = p.alloc_read(2 * P)
    assert all(p.is_placeholder(int(i)) for i in ph)
    _write_page(arena, "pg0", 10); _write_page(arena, "pg1", 90)
    slots = [s for s, _ in arena.find_slots(["pg0", "pg1"])]
    host = ph.clone()
    p.resolve_rows(host, slots)
    assert host.tolist() == p.arena_ids(slots).tolist()
    assert p._slots_of(host) == slots
    assert p.available_size() % P == 0
    with pytest.raises(ValueError):
        p.resolve_rows(ph[: P - 1].clone(), slots)  # a run that does not fit the registration


def test_a_page_load_lands_every_token_of_this_ranks_layers(tmp_path, monkeypatch):
    monkeypatch.setenv(ap.ARENA_PAGE_LOAD_BLOCK_ENV, "64")
    p, arena = _pool(tmp_path)
    _write_page(arena, "pg0", 10); _write_page(arena, "pg1", 90)
    slots = [s for s, _ in arena.find_slots(["pg0", "pg1"])]
    host = p.arena_ids(slots)
    dst = torch.tensor([4, 5, 6, 7, 20, 21, 22, 23])
    dev = _dev_pool()
    p.load_to_device_per_layer(dev, host, dst, 0, "kernel")
    p.load_to_device_per_layer(dev, host, dst, 1, "kernel")  # came with the page load at layer 0
    for l in range(L):
        ko = (1 + l) * BLOCK
        vo = PAGE // 2 + (1 + l) * BLOCK
        for i, tag in enumerate((10, 90)):
            page = _page_bytes(tag)
            for t in range(P):
                row = int(dst[i * P + t])
                assert dev.k_buffer[l][row].flatten().tolist() == page[ko + t * CELL: ko + (t + 1) * CELL].tolist(), \
                    f"K layer {l} page {i} token {t}"
                assert dev.v_buffer[l][row].flatten().tolist() == page[vo + t * CELL: vo + (t + 1) * CELL].tolist(), \
                    f"V layer {l} page {i} token {t}"
    # rows this pool never loaded stay untouched
    assert not dev.k_buffer[0][0].any() and not dev.v_buffer[1][31].any()


def test_a_direct_write_puts_this_ranks_blocks_into_the_slots(tmp_path):
    p, arena = _pool(tmp_path)
    ids = p.alloc_write(["h0", "h1"])
    assert ids is not None and ids.numel() == 2 * P and all(p.is_arena_id(int(i)) for i in ids)
    slots = p._slots_of(ids)
    dst = torch.tensor([3, 4, 5, 6, 10, 11, 12, 13])
    dev = _dev_pool()
    for l in range(L):
        for j, row in enumerate(dst.tolist()):
            dev.k_buffer[l][row] = torch.full((H, D), 1 + 10 * l + j, dtype=torch.uint8)
            dev.v_buffer[l][row] = torch.full((H, D), 101 + 10 * l + j, dtype=torch.uint8)
    p.backup_from_device_all_layer(dev, ids, dst, "kernel")
    for l in range(L):
        ko = (1 + l) * BLOCK
        vo = PAGE // 2 + (1 + l) * BLOCK
        for i, slot in enumerate(slots):
            want_k = [v for t in range(P) for v in [1 + 10 * l + i * P + t] * CELL]
            want_v = [v for t in range(P) for v in [101 + 10 * l + i * P + t] * CELL]
            assert p._page_view[slot, ko:ko + BLOCK].tolist() == want_k, f"K layer {l} slot {slot}"
            assert p._page_view[slot, vo:vo + BLOCK].tolist() == want_v, f"V layer {l} slot {slot}"
        # the other layer shard's blocks (ordinal 0) are untouched
        assert not p._page_view[slots[0], 0:BLOCK].any()
    p.complete_write(ids)  # merges this rank's extents (the page is not full: ordinal 0 is missing)
    assert p.free(ids) == 2 * P


def test_the_draft_role_refuses_pages(tmp_path):
    p = object.__new__(ArenaMHAHostPool)
    p.layout = "layer_first"; p.page_size = P; p.layer_num = L; p.head_num = H; p.head_dim = D
    p.dtype = torch.uint8; p.device = "cpu"; p.pin_memory = False; p.size = S
    p.element_dim = H * D; p.can_use_jit = False
    p.free_slots = torch.arange(S, dtype=torch.int64); p.slot_used = torch.zeros(S, dtype=torch.bool)
    p.kv_buffer = torch.zeros(2, L, S, H, D, dtype=torch.uint8)
    p._arena_init_fields()
    arena = ShmArena(str(tmp_path / "d.bin"), PAGE, A)
    with pytest.raises(ValueError):
        p.bind(arena, _Win(), role="draft", pin=False)


def test_planned_capacity_matches_the_backends_slot_rule(monkeypatch):
    monkeypatch.setenv("SGLANG_HICACHE_ARENA_GIB", "1")
    monkeypatch.delenv(ap.ENV_ARENA_KV_PAGE_BYTES, raising=False)
    slots = ap.planned_arena_slots(PAGE)
    assert slots == max(1024, (1 << 30) // PAGE)
    assert ap.planned_id_space_tokens(S, P, PAGE) == S + slots * P
    monkeypatch.setenv(ap.ENV_ARENA_KV_PAGE_BYTES, str(PAGE * 2))
    assert ap.planned_arena_slots(PAGE) == max(1024, (1 << 30) // (PAGE * 2)), "the launcher's canonical page wins"
