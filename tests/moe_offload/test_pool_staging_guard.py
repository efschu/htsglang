"""Task #40: the staging table is bounded by the LRU, not by the id width.

A miss takes an LRU victim (a row not used this step) or a staging row; hits
protect at most `len(ids)` LRU rows, so `len(ids) <= lru + staging` makes an
overflow impossible. With that bound the staging rows shrink from 30 (= the
id width) to a handful, and the freed rows become LRU cache."""

import pytest
import torch

from sglang.srt.layers.moe import expert_pool_device as ep


def _pool(E=12, R=2, lru=3, staging=2):
    rows = R + lru + staging
    host_row = [-1] * R + list(range(E - R))
    t = ep.allocate_pool_tables("cpu", E, rows, R, staging, {e: e for e in range(R)}, host_row)
    b = ep.allocate_step_buffers("cpu", E, 16)
    return t, b


def test_ids_may_exceed_the_staging_rows_up_to_the_lru_bound():
    t, b = _pool(lru=3, staging=2)  # bound: 5 ids
    ids = torch.tensor([2, 3, 4, 5, 6], dtype=torch.int32)  # 5 misses, no hits
    ep.step_reference(t, ids, b)
    assert int(b.gather_count[0]) == 5 and int(b.staged_count[0]) == 2
    assert int(t.error[0]) == 0
    ep.check_pool_error(t)


def test_more_ids_than_lru_plus_staging_is_refused_before_the_step():
    t, b = _pool(lru=3, staging=2)
    with pytest.raises(ValueError, match="LRU rows plus the staging rows"):
        ep.step_reference(t, torch.tensor([2, 3, 4, 5, 6, 7], dtype=torch.int32), b)
    with pytest.raises(ValueError, match="LRU rows plus the staging rows"):
        ep.step(t, torch.tensor([2, 3, 4, 5, 6, 7], dtype=torch.int32), b)


def test_an_overflow_under_protect_recent_sets_the_sticky_error():
    t, b = _pool(lru=3, staging=1)  # bound: 4 ids
    t.protect_recent.fill_(1)
    # step 1 fills the three LRU rows
    ep.step_reference(t, torch.tensor([2, 3, 4], dtype=torch.int32), b)
    # step 2: four new misses; protect_recent keeps all three rows (used at
    # clock-1), one staging row -> the 2nd..4th miss cannot be placed
    ep.step_reference(t, torch.tensor([5, 6, 7, 8], dtype=torch.int32), b)
    assert int(t.error[0]) == 1
    assert int(b.staged_count[0]) == 1
    with pytest.raises(RuntimeError, match="sticky device error"):
        ep.check_pool_error(t, "test")


def test_the_kernel_caps_the_staged_loop_at_the_staging_count():
    import inspect

    src = inspect.getsource(ep._step_kernel)
    assert "if staged < n_staging:" in src
    assert "tl.store(error_ptr, 1)" in src
    assert "n_staging" in inspect.getsource(ep.step)
