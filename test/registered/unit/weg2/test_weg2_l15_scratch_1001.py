# SPDX-License-Identifier: Apache-2.0
"""L15-W2B: the wake sample-check SCRATCH pool
(design /spinning/gpu-arb/docs/L15-WIRE2-NOTES.md sec 2).

make_scratch_pool(live_pool, rows) returns a one-shot, never-resident
ScratchKV that quacks like the live KV pool for exactly the attributes
the sample-check path touches:

* ArenaMHAHostPool._load_pages_all_layers reads
  device_pool.k_buffer[0].device (arena_pool.py:1625) and scatters
  through device_pool.k_buffer[l].index_copy_(0, dst, block) and
  device_pool.v_buffer[l].index_copy_(0, dst, block)
  (arena_pool.py:1715-1720);
* l15_sample.read_rows indexes b[r] on p.k_buffer / p.v_buffer after
  unwrapping a hybrid pool via .full_kv_pool (l15_sample.py:101).

CPU fakes only.
"""

from __future__ import annotations

import gc
import pathlib
import sys
import weakref

import torch

REPO = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "python"))

from sglang.srt.weg2 import l15_sample, l15_scratch  # noqa: E402

LAYERS = 3
HEADS = 2
HEAD_DIM = 4
LIVE_ROWS = 64
SCRATCH_ROWS = 6
ROW_SHAPE = (HEADS, HEAD_DIM)


class _FakeLivePool:
    """MHATokenToKVPool-shaped live pool: k_buffer/v_buffer are per-layer
    lists of (slots, heads, dim) float16 tensors, filled with distinct
    known bytes so any touch shows up."""

    def __init__(self, rows=LIVE_ROWS, dtype=torch.float16):
        self.k_buffer = [
            torch.full((rows,) + ROW_SHAPE, 100.0 + l, dtype=dtype)
            for l in range(LAYERS)
        ]
        self.v_buffer = [
            torch.full((rows,) + ROW_SHAPE, 200.0 + l, dtype=dtype)
            for l in range(LAYERS)
        ]


class _FakeHostPool:
    """_load_pages_all_layers that writes KNOWN bytes through the same
    attribute paths the real ArenaMHAHostPool._load_pages_all_layers
    touches: device_pool.k_buffer[0].device (arena_pool.py:1625), then
    per layer device_pool.k_buffer[l].index_copy_(0, dst, block) and
    device_pool.v_buffer[l].index_copy_(0, dst, block)
    (arena_pool.py:1715-1720)."""

    def __init__(self, n_src=16):
        self.source_k = [
            torch.stack([torch.full(ROW_SHAPE, float(s * 10 + l))
                         for s in range(n_src)])
            for l in range(LAYERS)
        ]
        self.source_v = [
            torch.stack([torch.full(ROW_SHAPE, float(s * 10 + l + 100))
                         for s in range(n_src)])
            for l in range(LAYERS)
        ]
        self.calls = 0

    def _load_pages_all_layers(self, device_pool, slots, device_indices,
                               lanes=None, mode=None):
        self.calls += 1
        dev = device_pool.k_buffer[0].device
        dst = device_indices.to(device=dev, dtype=torch.int64)
        slots = slots.to(device=dev, dtype=torch.int64)
        for l in range(len(device_pool.k_buffer)):
            # the real loader converts host dtype -> device dtype on the
            # way in (JIT kernel / staging copy); mirror that here
            device_pool.k_buffer[l].index_copy_(
                0, dst, self.source_k[l].index_select(0, slots)
                .to(device_pool.k_buffer[l].dtype))
            device_pool.v_buffer[l].index_copy_(
                0, dst, self.source_v[l].index_select(0, slots)
                .to(device_pool.v_buffer[l].dtype))


def _sampled():
    # (rid, compact_row, l2_slot, l2_gen); the loads land on scratch
    # rows 0, 1, 2 in this order
    return [("a", 3, 5, 0), ("b", 7, 2, 0), ("d", 9, 11, 0)]


def test_scratch_mirrors_the_live_per_layer_shape_dtype_device():
    live = _FakeLivePool()
    scratch = l15_scratch.make_scratch_pool(live, SCRATCH_ROWS)
    assert isinstance(scratch, l15_scratch.ScratchKV)
    assert len(scratch.k_buffer) == len(live.k_buffer) == LAYERS
    assert len(scratch.v_buffer) == len(live.v_buffer) == LAYERS
    for l in range(LAYERS):
        for sb, lb in ((scratch.k_buffer[l], live.k_buffer[l]),
                       (scratch.v_buffer[l], live.v_buffer[l])):
            assert tuple(sb.shape) == (SCRATCH_ROWS,) + ROW_SHAPE
            assert sb.dtype == lb.dtype
            assert sb.device == lb.device


def test_make_scratch_pool_unwraps_the_hybrid_wrapper_like_read_rows():
    inner = _FakeLivePool()
    outer = type("Hyb", (), {})()
    # a WRONG shape on the wrapper itself: only the .full_kv_pool unwrap
    # (the same one read_rows does, l15_sample.py:101) yields the truth
    outer.k_buffer = [torch.zeros((1, 1), dtype=torch.float32)]
    outer.v_buffer = [torch.zeros((1, 1), dtype=torch.float32)]
    outer.full_kv_pool = inner
    scratch = l15_scratch.make_scratch_pool(outer, SCRATCH_ROWS)
    assert len(scratch.k_buffer) == len(inner.k_buffer) == LAYERS
    for l in range(LAYERS):
        assert tuple(scratch.k_buffer[l].shape) == (SCRATCH_ROWS,) + ROW_SHAPE
        assert scratch.k_buffer[l].dtype == inner.k_buffer[l].dtype


def test_load_into_scratch_lands_source_bytes_in_scratch_rows():
    live = _FakeLivePool()
    live_before = [b.clone() for b in live.k_buffer + live.v_buffer]
    host = _FakeHostPool()
    scratch = l15_scratch.make_scratch_pool(live, SCRATCH_ROWS)
    sampled = _sampled()
    ids = l15_sample.load_into_scratch(sampled, host, scratch, page_tokens=1)
    assert ids == [0, 1, 2]
    assert host.calls == 1
    for r, t in enumerate(sampled):
        slot = t[2]
        for l in range(LAYERS):
            assert bool((scratch.k_buffer[l][r] == slot * 10 + l).all())
            assert bool((scratch.v_buffer[l][r] == slot * 10 + l + 100).all())
    # the live pool is only ever READ by the check -- never a load target
    for lb, before in zip(live.k_buffer + live.v_buffer, live_before):
        assert torch.equal(lb, before)


def test_read_rows_works_on_the_scratch_and_matches_the_l2_source():
    host = _FakeHostPool()
    scratch = l15_scratch.make_scratch_pool(_FakeLivePool(), SCRATCH_ROWS)
    sampled = _sampled()
    ids = l15_sample.load_into_scratch(sampled, host, scratch, page_tokens=1)
    got = l15_sample.read_rows(scratch, ids)
    assert len(got) == len(sampled)
    for row, t in zip(got, sampled):
        slot = t[2]
        ks = torch.cat([host.source_k[l][slot].reshape(-1)
                        for l in range(LAYERS)])
        vs = torch.cat([host.source_v[l][slot].reshape(-1)
                        for l in range(LAYERS)])
        assert row.numel() == 2 * LAYERS * HEADS * HEAD_DIM
        assert torch.equal(row[: ks.numel()], ks)
        assert torch.equal(row[ks.numel():], vs)
    # read_rows reaches the scratch through a .full_kv_pool wrapper too
    outer = type("Hyb", (), {})()
    outer.full_kv_pool = scratch
    assert all(torch.equal(a, b) for a, b in
               zip(l15_sample.read_rows(outer, ids), got))


def test_nbytes_counts_every_scratch_byte():
    scratch = l15_scratch.make_scratch_pool(_FakeLivePool(), SCRATCH_ROWS)
    esz = scratch.k_buffer[0].element_size()
    per_layer = SCRATCH_ROWS * HEADS * HEAD_DIM * esz
    assert scratch.nbytes() == 2 * LAYERS * per_layer


def test_free_drops_the_tensors_and_is_idempotent():
    scratch = l15_scratch.make_scratch_pool(_FakeLivePool(), SCRATCH_ROWS)
    ref_k = weakref.ref(scratch.k_buffer[0])
    ref_v = weakref.ref(scratch.v_buffer[-1])
    scratch.free()
    gc.collect()
    assert ref_k() is None and ref_v() is None
    assert scratch.nbytes() == 0
    scratch.free()  # idempotent, second call is a no-op


def test_rejects_bogus_arguments():
    live = _FakeLivePool()
    for rows in (0, -3):
        try:
            l15_scratch.make_scratch_pool(live, rows)
            raise AssertionError("expected ValueError for rows=%r" % (rows,))
        except ValueError:
            pass
    try:
        l15_scratch.make_scratch_pool(object(), 4)
        raise AssertionError("expected ValueError without k_buffer/v_buffer")
    except ValueError:
        pass
