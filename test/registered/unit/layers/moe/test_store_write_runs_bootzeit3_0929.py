"""BOOTZEIT 3 (29.09., z30r3): the presplit's device -> store write in runs.

z30r3 PP0: store_write 16.7 s of a 51.1 s presplit -- one synchronous D2H
copy per expert row. With SGLANG_EXPERT_STORE_WRITE_RUNS the rows go in runs
where both the store rows and the source rows are consecutive: same bytes,
same rows, a handful of copies per tensor.
"""

import os
from unittest import mock

import torch

from sglang.srt.layers.moe import expert_store as es


def test_runs_need_both_sides_consecutive():
    rows = {0: 10, 1: 11, 2: 12, 5: 13, 6: 20, 7: 21}
    assert es.row_runs(rows) == [(10, 0, 3), (13, 5, 1), (20, 6, 2)]


def test_runs_do_not_bridge_a_hole_on_either_side():
    assert es.row_runs({0: 0, 2: 1}) == [(0, 0, 1), (1, 2, 1)]
    assert es.row_runs({0: 0, 1: 2}) == [(0, 0, 1), (2, 1, 1)]
    assert es.row_runs({}) == []


def test_run_copy_writes_the_same_bytes_as_the_row_copy():
    g = torch.Generator().manual_seed(3)
    src = torch.randint(0, 255, (9, 4, 6), dtype=torch.uint8, generator=g)
    rows = {0: 3, 1: 4, 2: 5, 4: 0, 5: 1, 8: 7}
    ref = torch.zeros(8, 4, 6, dtype=torch.uint8)
    for e, r in rows.items():
        ref[r].copy_(src[e])
    got = torch.zeros_like(ref)
    es._copy_row_runs(got, src, es.row_runs(rows), non_blocking=False)
    assert torch.equal(ref, got)


def test_switch_off_keeps_the_per_row_path(monkeypatch):
    monkeypatch.delenv("SGLANG_EXPERT_STORE_WRITE_RUNS", raising=False)
    from sglang.srt.environ import envs

    assert envs.SGLANG_EXPERT_STORE_WRITE_RUNS.get() is False
    called = []
    monkeypatch.setattr(es, "_copy_row_runs", lambda *a, **k: called.append(1))
    store = torch.zeros(4, 2)
    src = torch.ones(4, 2)
    # a CPU source takes the index_copy_ branch in either mode; the run copy
    # belongs to the device branch only
    with mock.patch.dict(os.environ, {"SGLANG_EXPERT_STORE_WRITE_RUNS": "1"}):
        es.write_rows(store, src, [0, 1], 0, rows={0: 0, 1: 1})
    assert called == []
    assert torch.equal(store[:2], torch.ones(2, 2))
