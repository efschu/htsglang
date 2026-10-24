# SPDX-License-Identifier: Apache-2.0
"""Review #217 ADMIT-PEAK (desk/27b-admit-peak-review-1024, 24.10.): two pins the author's battery leaves open.

Danger direction of both: DATA CORRUPTION (bytes landing in the device pool that are not the arena's).

1. The 64-MiB KV device stage makes a MULTI-BLOCK load the common case (27B: from 2049 rank pages on; NF: from 86).
   The only multi-block byte check of the page load (xsn305) runs the "cpu" mode on a CPU box; a stage refilled for
   block i+1 before block i's scatter read it (review mutant R3) survived every "dma" test. Pinned here: a "dma" load
   of 2100 pages at the DEFAULT #217 geometry (two blocks of 2048 + 52) lands the bytes of the single-block "cpu" load.
2. The vectorised mamba split was only checked with ONE block (b == stage rows). Pinned here: several blocks with a
   PARTIAL last block (3-row blocks, 3 + 3 + 1 slots) land the per-layer loop's bytes in the same rows.
Hermetic: tmp arena file, no CUDA.
"""
import os
import shutil
import threading
import types
import unittest.mock as mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.mem_cache.pool_host import arena_mamba_pool as AMP  # noqa: E402
from flliper.srt.mem_cache.pool_host import arena_pool as AP  # noqa: E402
from flliper.srt.mem_cache.pool_host.arena_pool import ArenaMHAHostPool  # noqa: E402
from flliper.srt.mem_cache.storage.file.hicache_arena import ShmArena  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(__file__)

P, L, LT, H, D = 1, 2, 3, 2, 4   # page tokens, local layers, model layers, heads, head_dim (uint8)
CELL = H * D
BLOCK = P * CELL
PAGE = 2 * LT * BLOCK
N_PAGES = 2100                   # > 2048: two blocks under the #217 device stage, one under the cpu mode's 8192


def _clean_env(**kw):
    clean = {k: v for k, v in os.environ.items()
             if k not in (AP.ARENA_PAGE_LOAD_BLOCK_ENV, AP.ENV_ADMIT_PEAK_STAGES, AMP.ENV_STATE_LOAD_BLOCK_BYTES,
                          AMP.ENV_STATE_SPLIT_VEC, "FLLIPER_PDFLIP_ARENA_PAGE_LOAD_MODE")}
    clean.update(kw)
    return mock.patch.dict(os.environ, clean, clear=True)


class _Win:
    total_bytes = PAGE
    extents = ((1 * BLOCK, L * BLOCK), (PAGE // 2 + 1 * BLOCK, L * BLOCK))


def _kv_pool(tmp_path):
    p = object.__new__(ArenaMHAHostPool)
    p.layout = "layer_first"; p.page_size = P; p.layer_num = L; p.head_num = H; p.head_dim = D
    p.dtype = torch.uint8; p.device = "cpu"; p.pin_memory = False; p.size = 8
    p.element_dim = H * D; p.can_use_jit = False; p.token_stride_size = CELL
    p.lock = threading.Lock()
    p.free_slots = torch.arange(8, dtype=torch.int64); p.slot_used = torch.zeros(8, dtype=torch.bool)
    p.kv_buffer = torch.zeros(2, L, 8, H, D, dtype=torch.uint8)
    p._arena_init_fields()
    arena = ShmArena(str(tmp_path / "kv.bin"), PAGE, N_PAGES)
    p.bind(arena, _Win(), role="kv", pin=False)
    p._pinned[:] = True
    return p, arena


def _kv_load(p, slots, mode):
    rows = len(slots) * P + 3
    dev = types.SimpleNamespace(
        k_buffer=[torch.zeros(rows, H, D, dtype=torch.uint8) for _ in range(L)],
        v_buffer=[torch.zeros(rows, H, D, dtype=torch.uint8) for _ in range(L)])
    with _clean_env(FLLIPER_PDFLIP_ARENA_PAGE_LOAD_MODE=mode):
        p._page_mode = None
        p._page_loaded_key = None
        p._page_stages = None
        assert AP._arena_page_load_block(PAGE, mode) == (8192 if mode == "cpu" else 2048)
        p.load_to_device_per_layer(dev, p.arena_ids(slots), torch.arange(len(slots) * P, dtype=torch.int64) + 3,
                                   0, "kernel")
    return dev


@pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")
def test_a_multi_block_dma_load_at_the_217_geometry_lands_the_single_block_bytes(tmp_path):
    p, arena = _kv_pool(tmp_path)
    stems = [f"pg{i}" for i in range(N_PAGES)]
    for i, stem in enumerate(stems):
        pay = ((torch.arange(PAGE) * (i % 97 + 3) + 7 * i) % 251).to(torch.uint8)
        assert arena.write([stem], [PAGE], [((0, PAGE),)], [pay.data_ptr()]) == [1]
    by_stem = dict(zip(stems, [s for s, _ in arena.find_slots(stems)]))
    # a permutation (reversed halves) so the block boundary at 2048 falls inside runs and between them
    order = list(range(N_PAGES // 2, N_PAGES)) + list(range(N_PAGES // 2))
    slots = [by_stem[stems[i]] for i in order]
    p._dma_piece_pages = 512
    want = _kv_load(p, slots, "cpu")       # one block of 2100
    got = _kv_load(p, slots, "dma")        # blocks 2048 + 52 through ONE 64-MiB-geometry stage
    for l in range(L):
        assert torch.equal(got.k_buffer[l], want.k_buffer[l]), f"K layer {l}"
        assert torch.equal(got.v_buffer[l], want.v_buffer[l]), f"V layer {l}"
    assert got.k_buffer[0][3 + 2048:].any() and got.k_buffer[0][3:3 + 2048].any()


# ---- mamba: several blocks, partial last block --------------------------------------------------------------------

ML, MA, T_SHAPE, WIDTH, CONV_SHAPE, E = 3, 9, (2, 4), 3, (6, 3), 2
T_ROW = 8 * E
SPLIT = (2, 3, 1)


def _mamba_pool():
    t_ext, c_ext, off = [], [], 0
    for _l in range(ML):
        off += 5
        t_ext.append((off, T_ROW)); off += T_ROW
        off += 3
        segs = []
        for j, n_j in enumerate(SPLIT):
            if j == 2:
                off += 2
            segs.append((off, n_j * WIDTH * E)); off += n_j * WIDTH * E
        c_ext.append(segs)
    g = torch.Generator().manual_seed(1024)
    blob = torch.randint(0, 255, (MA, off), dtype=torch.uint8, generator=g)
    pool = AMP.ArenaMambaPoolHost.__new__(AMP.ArenaMambaPoolHost)
    pool._slot_view = blob
    pool._page_bytes = off            # < 32 KiB: the non-dma (pinned-stage) path, the one with multi-row blocks
    pool._state_stage = None
    pool.temporal_dtype = torch.bfloat16
    pool.conv_dtype = torch.bfloat16
    pool._layout = {"L": ML, "t_shape": T_SHAPE, "conv_shape": CONV_SHAPE, "width": WIDTH, "t_ext": t_ext,
                    "c_ext": c_ext}
    return pool


def _mamba_dev(rows=16):
    return types.SimpleNamespace(mamba_cache=types.SimpleNamespace(
        temporal=torch.zeros((ML, rows) + T_SHAPE, dtype=torch.bfloat16),
        conv=[torch.zeros((ML, rows) + CONV_SHAPE, dtype=torch.bfloat16)]))


def _u8(t):
    return t.contiguous().view(torch.uint8)


def test_the_vectorised_split_with_a_partial_last_block_lands_the_loops_bytes():
    slots = torch.tensor([8, 1, 6, 2, 0, 5, 3]); didx = torch.tensor([4, 15, 0, 9, 11, 2, 7])
    out = {}
    for vec in ("1", "0"):
        pool = _mamba_pool()
        dp = _mamba_dev()
        # 3 rows per block (the explicit byte env has a 1-MiB floor, too coarse for this fixture): blocks 3 + 3 + 1
        with _clean_env(**{AMP.ENV_STATE_SPLIT_VEC: vec}), \
                mock.patch.object(AMP, "_state_load_block_rows", lambda rb, dma, env=None: 3):
            pool._load_states_all_layers(dp, slots, didx)
            if vec == "1":
                assert pool._state_split_vec(dp, pool._compact[0], pool._compact[1]) is not None
        assert tuple(pool._state_dev_stage.shape)[0] == 3
        out[vec] = dp
    assert torch.equal(_u8(out["1"].mamba_cache.temporal), _u8(out["0"].mamba_cache.temporal))
    assert torch.equal(_u8(out["1"].mamba_cache.conv[0]), _u8(out["0"].mamba_cache.conv[0]))
    assert _u8(out["1"].mamba_cache.temporal[:, didx[-1]]).any()     # the partial block's slot landed
    untouched = [r for r in range(16) if r not in didx.tolist()]
    assert not _u8(out["1"].mamba_cache.temporal[:, untouched]).any()
