"""H12 (fnFL2x104, 90k needle): the paged arena loadback after the wake.

The first scheduler pass after the P->D wake spent init_new 448 ms on D TP0,
347 ms of it in the page load's cpu stage: a CPU index_select of 1528 pages
(1.2 GB) in the scheduler thread at 3.5 GB/s while the expert workers waited
in the extend's first all_reduce. The pages of one request lie in consecutive
arena slots (x104: slot=[72,1599]), so the "dma" mode copies each run of
consecutive slots straight out of the registered arena -- no gather, no JIT,
no pinned stage.

What must hold: a run never crosses a cudaHostRegister piece (one copy is one
DMA from page-locked memory), the stage keeps the caller's page order, the
bytes that land are the cpu stage's bytes, and "dma" never allocates the cpu
mode's 2 x 256 MiB pinned host stages. Hermetic: tmp arena file, no CUDA.
"""
import os
import shutil
import threading
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch

from sglang.srt.mem_cache.pool_host import arena_pool as ap
from sglang.srt.mem_cache.pool_host.arena_pool import ArenaMHAHostPool, page_dma_runs
from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(__file__)

P, L, LT, H, D = 4, 2, 3, 2, 4   # page tokens, local layers, model layers, heads, head_dim (uint8)
CELL = H * D
BLOCK = P * CELL
PAGE = 2 * LT * BLOCK
S, A = 8, 6


def _covered(runs):
    return [first + i for _row, first, count in runs for i in range(count)]


def test_runs_merge_consecutive_slots_and_never_cross_a_registration_piece():
    # x104's shape: 1528 consecutive pages from slot 72, pre-pin pieces of
    # 1365 pages (1 GiB // 786432 B) -> one copy per piece the run touches
    slots = list(range(72, 1600))
    runs = page_dma_runs(slots, piece_pages=1365)
    assert runs == [(0, 72, 1365 - 72), (1365 - 72, 1365, 1600 - 1365)]
    # out of order, a gap, a descending pair: the stage keeps the caller's order
    slots = [5, 6, 7, 3, 9, 8, 10, 11]
    runs = page_dma_runs(slots, piece_pages=4)
    assert _covered(runs) == slots
    assert [row for row, _f, _c in runs] == [0, 3, 4, 5, 6]
    for _row, first, count in runs:
        assert first // 4 == (first + count - 1) // 4, f"run {first}+{count} crosses a piece"
    # the lazy pin registers whole slots: piece 1 is one copy per page
    assert page_dma_runs([2, 3, 4], piece_pages=1) == [(0, 2, 1), (1, 3, 1), (2, 4, 1)]
    assert page_dma_runs([], piece_pages=8) == []


class _Win:
    total_bytes = PAGE
    extents = ((1 * BLOCK, L * BLOCK), (PAGE // 2 + 1 * BLOCK, L * BLOCK))


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
    p._pinned[:] = True
    return p, arena


def _dev_pool(rows=32):
    return types.SimpleNamespace(
        k_buffer=[torch.zeros(rows, H, D, dtype=torch.uint8) for _ in range(L)],
        v_buffer=[torch.zeros(rows, H, D, dtype=torch.uint8) for _ in range(L)],
    )


def _load(p, slots, mode, monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_ARENA_PAGE_LOAD_MODE", mode)
    p._page_mode = None
    p._page_loaded_key = None
    dst = torch.arange(len(slots) * P, dtype=torch.int64) + 3
    dev = _dev_pool()
    p.load_to_device_per_layer(dev, p.arena_ids(slots), dst, 0, "kernel")
    return dev


@pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")
def test_dma_lands_the_cpu_stages_bytes_without_a_pinned_stage(tmp_path, monkeypatch):
    monkeypatch.setenv(ap.ARENA_PAGE_LOAD_BLOCK_ENV, "64")
    p, arena = _pool(tmp_path)
    stems = [f"pg{i}" for i in range(A)]
    for i, stem in enumerate(stems):
        pay = ((torch.arange(PAGE) * (i + 3) + 7 * i) % 251).to(torch.uint8)
        assert arena.write([stem], [PAGE], [((0, PAGE),)], [pay.data_ptr()]) == [1]
    by_stem = dict(zip(stems, [s for s, _ in arena.find_slots(stems)]))
    # consecutive, a jump back, and a piece boundary inside the consecutive run
    slots = [by_stem["pg1"], by_stem["pg2"], by_stem["pg3"], by_stem["pg0"], by_stem["pg5"]]
    p._dma_piece_pages = 2
    want = _load(p, slots, "cpu", monkeypatch)
    p._page_stages = None
    got = _load(p, slots, "dma", monkeypatch)
    for l in range(L):
        assert torch.equal(got.k_buffer[l], want.k_buffer[l]), f"K layer {l}"
        assert torch.equal(got.v_buffer[l], want.v_buffer[l]), f"V layer {l}"
    assert got.k_buffer[0][3:3 + len(slots) * P].any()
    assert p._page_stages is None, "dma allocated the cpu mode's pinned host stages"
