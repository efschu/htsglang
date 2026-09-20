"""Task #47 build step 1: the shared expert host store (rows = global ids).
Hermetic: CUDA_VISIBLE_DEVICES="" -> the store is a plain MAP_SHARED file."""
import os

import torch

from sglang.srt.layers.moe import expert_store as es


def test_global_rows_skip_the_pad_expert():
    assert es.global_rows([0, 1, 2, 5], lo=312) == {1: 312, 2: 313, 5: 316}


def test_two_ranks_write_disjoint_ranges_into_one_file(tmp_path):
    d = str(tmp_path)
    E, row = 8, (3, 4)
    # rank 0 owns global [0, 5) as local 1..5 ; rank 1 owns [5, 8) as local 1..3
    src0 = torch.arange(6 * 12, dtype=torch.float32).reshape(6, *row)  # local 0 = pad
    src1 = -torch.arange(4 * 12, dtype=torch.float32).reshape(4, *row)
    s0, created0 = es.open_store(d, "L3", "w2_weight", E, row, torch.float32, register=False)
    s1, created1 = es.open_store(d, "L3", "w2_weight", E, row, torch.float32, register=False)
    assert created0 and not created1
    r0 = es.write_rows(s0, src0, [1, 2, 3, 4, 5], lo=0)
    r1 = es.write_rows(s1, src1, [1, 2, 3], lo=5)
    assert r0 == {1: 0, 2: 1, 3: 2, 4: 3, 5: 4} and r1 == {1: 5, 2: 6, 3: 7}
    # the second mapping sees the first writer's bytes and vice versa
    assert torch.equal(s1[0:5], src0[1:6])
    assert torch.equal(s0[5:8], src1[1:4])
    es.mark_rows_written(d, "L3", "w2_weight", 0, r0.values())
    es.mark_rows_written(d, "L3", "w2_weight", 1, r1.values())
    written = es.rows_written(d, "L3", "w2_weight", world=2)
    assert written == {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 1}
    assert os.path.exists(es.store_path(d, "L3", "w2_weight"))


def test_store_is_off_without_the_env(monkeypatch):
    monkeypatch.delenv(es.STORE_DIR_ENV, raising=False)
    assert not es.store_enabled()
    monkeypatch.setenv(es.STORE_DIR_ENV, "/dev/shm/x")
    assert es.store_enabled() and es.store_dir() == "/dev/shm/x"


def test_unsharded_layer_maps_local_to_global_identity():
    assert es.global_rows([0, 1, 2], lo=0, pad=False) == {0: 0, 1: 1, 2: 2}


def test_write_rows_partial_ids_take_the_per_row_path(tmp_path):
    src = torch.arange(5 * 4, dtype=torch.float32).reshape(5, 4)
    s, _ = es.open_store(str(tmp_path), "L0", "w2", 10, (4,), torch.float32, register=False)
    rows = es.write_rows(s, src, [1, 3], lo=6)
    assert rows == {1: 6, 3: 8}
    assert torch.equal(s[6], src[1]) and torch.equal(s[8], src[3]) and float(s[7].abs().sum()) == 0.0
