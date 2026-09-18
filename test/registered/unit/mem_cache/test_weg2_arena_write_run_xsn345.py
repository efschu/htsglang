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
