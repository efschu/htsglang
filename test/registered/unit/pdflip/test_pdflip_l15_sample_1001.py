# SPDX-License-Identifier: Apache-2.0
"""L15-12c-E1C: the sample-check loader (plan L15-12-PART3-PLAN sec 4).

l15_sample loads the L2 SOURCE of a few sampled held rows into SCRATCH
device rows (never the live rows) with the same one-call
_load_pages_all_layers machinery as l15_refill.refill, and reads the rows
back as plain tensors, so the wake can compare device-vs-L2 byte for byte
and fold bad>0 into the F11 refusal. CPU fakes only.
"""

from __future__ import annotations

import pathlib
import sys

import torch

REPO = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "python"))

from flliper.srt.pdflip import l15_sample  # noqa: E402

LAYERS = 2
COLUMNS = 8
SCRATCH_ROWS = 6   # >= the 5 sampled rows of _plan() at k=64
LIVE_ROWS = 6


class _FakeDevicePool:
    """k_buffer/v_buffer lists, one (rows, cols) tensor per layer."""

    def __init__(self, rows):
        self.k_buffer = [
            torch.zeros(rows, COLUMNS, dtype=torch.int64) for _ in range(LAYERS)
        ]
        self.v_buffer = [
            torch.zeros(rows, COLUMNS, dtype=torch.int64) for _ in range(LAYERS)
        ]


class _FakeHostPool:
    """_load_pages_all_layers that writes KNOWN bytes: the source slot into
    every byte of the k row, slot + 10000 into the v row, per layer."""

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def _load_pages_all_layers(self, device_pool, slots, device_indices,
                                lanes=None, mode=None):
        if self.fail:
            raise RuntimeError("boom")
        self.calls.append((tuple(int(s) for s in slots),
                           tuple(int(d) for d in device_indices)))
        for slot, dst in zip(
            (int(s) for s in slots), (int(d) for d in device_indices)
        ):
            for l in range(LAYERS):
                device_pool.k_buffer[l][dst].fill_(slot)
                device_pool.v_buffer[l][dst].fill_(slot + 10000)


def _plan():
    # (rid, compact_row, l2_slot, l2_gen); "c" has NO L2 source (slot -1)
    return [
        ("a", 0, 10, 5), ("b", 1, 11, 6), ("c", 2, -1, 7),
        ("d", 3, 13, 8), ("e", 4, 14, 9), ("f", 5, 15, 10),
    ]


def test_sample_plan_is_deterministic_and_excludes_no_source_rows():
    p1 = l15_sample.sample_plan(_plan(), k=64)
    p2 = l15_sample.sample_plan(_plan(), k=64)
    assert p1 == p2
    # all rows are sampled (n <= k) except the slot -1 row "c"
    assert [t[0] for t in p1] == ["a", "b", "d", "e", "f"]
    assert all(t[2] >= 0 for t in p1)
    assert ("c", 2, -1, 7) not in p1


def test_sample_plan_small_k_is_an_even_subset():
    s = l15_sample.sample_plan(_plan(), k=4)
    # deterministic: the same call twice, and the rows follow
    # l15_restore.sample_rows' even spacing (here rows {0, 1, 3, 4})
    assert [t[1] for t in l15_sample.sample_plan(_plan(), k=4)] == [t[1] for t in s]
    rows = [t[1] for t in s]
    assert len(rows) == len(set(rows)) and len(rows) <= 4
    assert all(t[2] >= 0 for t in s)


def test_load_into_scratch_only_scratch_rows_and_live_untouched():
    sampled = l15_sample.sample_plan(_plan(), k=64)
    host = _FakeHostPool()
    scratch = _FakeDevicePool(SCRATCH_ROWS)
    live = _FakeDevicePool(LIVE_ROWS)
    ids = l15_sample.load_into_scratch(sampled, host, scratch, page_tokens=1)
    assert ids == list(range(len(sampled)))
    # the one call carried the sampled slots -> scratch rows 0..n-1
    assert host.calls == [(tuple(t[2] for t in sampled), tuple(ids))]
    n = len(sampled)
    for l in range(LAYERS):
        for r in range(n):
            assert int(scratch.k_buffer[l][r, 0]) == sampled[r][2]
            assert int(scratch.v_buffer[l][r, 0]) == sampled[r][2] + 10000
        for r in range(n, SCRATCH_ROWS):          # unused scratch stays zero
            assert bool((scratch.k_buffer[l][r] == 0).all())
            assert bool((scratch.v_buffer[l][r] == 0).all())
    # the live pool was never a load target
    assert not host.calls or all(d < n for d in host.calls[0][1])
    for buf in (live.k_buffer, live.v_buffer):
        for l in range(LAYERS):
            assert bool((buf[l] == 0).all())


def test_read_rows_returns_equal_tensors_for_equal_bytes():
    a = _FakeDevicePool(2)
    b = _FakeDevicePool(2)
    for pool in (a, b):
        for l in range(LAYERS):
            pool.k_buffer[l].fill_(7)
            pool.v_buffer[l].fill_(9)
    ra = l15_sample.read_rows(a, [0, 1])
    rb = l15_sample.read_rows(b, [0, 1])
    assert len(ra) == 2 and all(torch.equal(x, y) for x, y in zip(ra, rb))
    # every layer's k then v bytes, in one 1-D tensor per row
    assert ra[0].numel() == 2 * LAYERS * COLUMNS
    assert bool((ra[0][: LAYERS * COLUMNS] == 7).all())
    assert bool((ra[0][LAYERS * COLUMNS :] == 9).all())
    # a differing byte shows up as a differing tensor
    b.k_buffer[1].fill_(8)
    assert not torch.equal(l15_sample.read_rows(b, [0])[0], ra[0])


def test_read_rows_unwraps_the_hybrid_full_pool():
    inner = _FakeDevicePool(2)
    inner.k_buffer[0].fill_(3)
    inner.v_buffer[0].fill_(4)
    outer = type("Hyb", (), {})()
    outer.full_kv_pool = inner
    rows = l15_sample.read_rows(outer, [0])
    # all layers' k first (lay 0 at [0, C)), then all layers' v
    assert bool((rows[0][: COLUMNS] == 3).all())
    assert bool(
        (rows[0][LAYERS * COLUMNS : (LAYERS + 1) * COLUMNS] == 4).all()
    )


def test_page_tokens_gt_1_raises_before_any_load():
    host = _FakeHostPool()
    scratch = _FakeDevicePool(SCRATCH_ROWS)
    try:
        l15_sample.load_into_scratch(
            [("a", 0, 10, 5)], host, scratch, page_tokens=2
        )
        raise AssertionError("expected L15SampleError for P>1")
    except l15_sample.L15SampleError:
        pass
    assert host.calls == []


def test_failing_load_raises_l15_sample_error():
    host = _FakeHostPool(fail=True)
    scratch = _FakeDevicePool(SCRATCH_ROWS)
    try:
        l15_sample.load_into_scratch(
            [("a", 0, 10, 5)], host, scratch, page_tokens=1
        )
        raise AssertionError("expected L15SampleError for a failing load")
    except l15_sample.L15SampleError:
        pass
