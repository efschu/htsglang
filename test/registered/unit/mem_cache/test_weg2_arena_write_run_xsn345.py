"""xsn345: the arena write's run mode -- pure decisions (extents contiguity,
pointer/stride plan, env knobs); the kernel itself is exercised by the boot."""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import arena_write as aw  # noqa: E402


def test_env_defaults():
    assert aw.write_mode({}) == "run"
    assert aw.write_mode({aw.MODE_ENV: "cell"}) == "cell"
    assert aw.write_mode({aw.MODE_ENV: "x"}) == "run"
    assert aw.write_block_quota({}) == 16
    assert aw.write_block_quota({aw.QUOTA_ENV: "0"}) is None
    assert aw.write_block_quota({aw.QUOTA_ENV: "8"}) == 8
    assert aw.write_block_quota({aw.QUOTA_ENV: "x"}) == 16


def test_two_extent_form_is_one_run_per_side():
    cell = 1024
    L = 12
    k = [10240 + l * cell for l in range(L)]
    v = [26624 + l * cell for l in range(L)]
    assert aw.contiguous_runs(k, v, cell) == (10240, 26624, 12 * cell)


def test_per_layer_interleaved_extents_keep_the_cell_kernel():
    cell = 256
    # the canonical draft page: K/V interleaved per layer
    k = [640 + l * 2 * cell for l in range(4)]
    v = [896 + l * 2 * cell for l in range(4)]
    assert aw.contiguous_runs(k, v, cell) is None
    assert aw.contiguous_runs([], [], cell) is None
    # overlapping runs are refused
    assert aw.contiguous_runs([0, cell], [cell, 2 * cell], cell) is None


def test_run_pointers_address_both_runs_of_a_page():
    dst, src, stride = aw.run_pointers(1_000_000, 10240, 26624, 12288, 5_000)
    assert dst == [1_010_240, 1_026_624]
    assert src == [5_000, 5_000 + 12288]
    assert stride == 2 * 12288


def test_mamba_pieces_group_by_element_size_and_stride():
    assert aw.mamba_write_mode({}) == "kernel"
    assert aw.mamba_write_mode({aw.MAMBA_MODE_ENV: "copy"}) == "copy"
    pieces = [(4096, 10, 20, 4096), (4096, 11, 21, 4096), (256, 12, 22, 768), (512, 13, 23, 768), (256, 14, 24, 768)]
    g = aw.group_pieces(pieces)
    assert list(g.keys()) == [(4096, 4096), (256, 768), (512, 768)]
    assert g[(4096, 4096)] == ([10, 11], [20, 21])
    assert g[(256, 768)] == ([12, 14], [22, 24])


def test_mamba_pieces_split_into_cached_1k_elements():
    assert aw.mamba_element_bytes({}) == 1024
    kern, rest = aw.split_pieces([(3072, 100, 500, 4096), (1000, 200, 600, 4096)], 1024)
    assert kern == [(1024, 100, 500, 4096), (1024, 1124, 1524, 4096), (1024, 2148, 2548, 4096)]
    assert rest == [(1000, 200, 600, 4096)]
    g = aw.group_pieces(kern)
    assert list(g.keys()) == [(1024, 4096)] and g[(1024, 4096)][0] == [100, 1124, 2148]


# --- fnFL2 H47: the run is written in 1-KiB items, never as one element ------

def _emulate_all_layer_mla(mem, ptr_dst, idx_dst, ptr_src, idx_src, src_stride, dst_stride, element):
    """hicache.cuh hicache_transfer_all_layer<kIsMLA=true> on a flat byte
    buffer: for every item i and pseudo-layer l, copy `element` bytes from
    ptr_src[l] + idx_src[i] * src_stride to ptr_dst[l] + idx_dst[i] * dst_stride."""
    for i in range(len(idx_src)):
        for l in range(len(ptr_src)):
            s = int(ptr_src[l]) + int(idx_src[i]) * src_stride
            d = int(ptr_dst[l]) + int(idx_dst[i]) * dst_stride
            mem[d:d + element] = mem[s:s + element]


def test_split_run_writes_the_same_bytes_as_the_whole_run():
    import numpy as np

    E = 1024
    L, block = 3, 2048                 # 3 layers x 2 KiB block -> run 6 KiB
    run = L * block
    page = 4 * run                     # arena page (multiple of E)
    k_off, v_off = 1024, 1024 + 2 * run
    n_slots, b = 6, 3
    slots = [4, 0, 2]
    stage_base = 0
    data_base = b * 2 * run + 4096     # arena region after the stage
    size = data_base + n_slots * page
    rng = np.random.default_rng(47)
    stage = rng.integers(0, 256, size=b * 2 * run, dtype=np.uint8)

    whole = np.zeros(size, dtype=np.uint8)
    whole[:stage.size] = stage
    dst, src, stride = aw.run_pointers(data_base, k_off, v_off, run, stage_base)
    _emulate_all_layer_mla(whole, dst, slots, src, list(range(b)), stride, page, run)

    split = np.zeros(size, dtype=np.uint8)
    split[:stage.size] = stage
    n = aw.run_split_elements(run, page, E)
    assert n == run // E
    idx_dst, idx_src = aw.run_split_indices(slots, n, page, E)
    assert idx_dst.numel() == idx_src.numel() == b * n
    _emulate_all_layer_mla(split, dst, idx_dst.tolist(), src, idx_src.tolist(), E, E, E)

    assert np.array_equal(whole, split)
    # and both put page p's K run / V run into slot_p
    for p, s in enumerate(slots):
        base = data_base + s * page
        assert np.array_equal(split[base + k_off:base + k_off + run], stage[p * 2 * run:p * 2 * run + run])
        assert np.array_equal(split[base + v_off:base + v_off + run], stage[p * 2 * run + run:(p + 1) * 2 * run])


def test_split_refuses_a_non_whole_element_run_or_page():
    assert aw.run_split_elements(229376, 786432) == 224          # PP0 of the NF canonical page
    assert aw.run_split_elements(98304, 786432) == 96            # PP1
    assert aw.run_split_elements(65536, 786432) == 64            # PP2
    assert aw.run_split_elements(1536, 786432, 1024) is None
    assert aw.run_split_elements(229376, 786433, 1024) is None
    assert aw.run_split_elements(0, 786432) is None
    assert aw.run_split_elements(229376, 786432, 0) is None


def test_run_element_keeps_the_thread_storage_small():
    # the measured law (fnFL2x151): stack = run / 32 - 64 B for the whole-run
    # element (unroll 1) -- 7104 / 3008 / 1984 B on PP0 / PP1 / PP2
    for run, stack in ((229376, 7104), (98304, 3008), (65536, 1984)):
        assert aw.kernel_thread_bytes(run, 1) - 64 == stack
    # the split element: the 1-KiB module (unroll 2), 64 B per thread, the one
    # the mamba write already launches without growing the 1248-B base stack
    assert aw.RUN_ELEMENT_BYTES == aw.mamba_element_bytes({}) == 1024
    assert aw.kernel_thread_bytes(aw.RUN_ELEMENT_BYTES, 2) == 64
