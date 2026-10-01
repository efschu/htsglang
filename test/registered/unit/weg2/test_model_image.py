"""Dual-model: the model image (a sleeping model's weights in a file).

Pinned without a GPU: the plan (sequential 2 MiB-aligned layout, refusals at
save), the wake check (same allocation set, all resumed), and a byte-exact
save/load round trip through the real file path with CPU tensors standing in
for device memory -- including a corrupted file caught by the checksum.
"""
import os
import tempfile

import pytest
import torch

from sglang.srt.weg2.model_image import (
    ALIGN,
    Alloc,
    ImagePlan,
    ModelImageRefused,
    check_wake,
    load,
    plan_image,
    save,
)


def A(ptr, size, mapped=None, active=True):
    return Alloc(ptr, size, size if mapped is None else mapped, active)


def test_plan_is_sorted_aligned_and_sums():
    p = plan_image({"weights_1": [A(0x5000, 100)], "weights_0": [A(0x9000, ALIGN + 1), A(0x1000, 10)]})
    assert [(e.tag, e.ptr) for e in p.entries] == [("weights_0", 0x1000), ("weights_0", 0x9000), ("weights_1", 0x5000)]
    assert [e.offset for e in p.entries] == [0, ALIGN, 3 * ALIGN]
    assert p.total_bytes == 10 + ALIGN + 1 + 100 and p.file_bytes == 4 * ALIGN


def test_plan_refuses_holes_paused_and_empty_tags():
    with pytest.raises(ModelImageRefused, match="span plan"):
        plan_image({"w": [A(0x1000, 100, mapped=50)]})
    with pytest.raises(ModelImageRefused, match="PAUSED"):
        plan_image({"w": [A(0x1000, 100, active=False)]})
    with pytest.raises(ModelImageRefused, match="no allocations"):
        plan_image({"w": []})


def test_wake_check_names_moved_and_paused_allocations():
    p = plan_image({"w": [A(0x1000, 100), A(0x2000, 100)]})
    check_wake(p, {"w": [A(0x2000, 100), A(0x1000, 100)]})
    with pytest.raises(ModelImageRefused, match="missing"):
        check_wake(p, {"w": [A(0x1000, 100), A(0x3000, 100)]})
    with pytest.raises(ModelImageRefused, match="still PAUSED"):
        check_wake(p, {"w": [A(0x1000, 100), A(0x2000, 100, active=False)]})
    with pytest.raises(ModelImageRefused, match="'extra'"):
        check_wake(p, {"w": [A(0x1000, 100), A(0x2000, 100)], "extra": [A(0x9000, 1)]})


def _fake_device(nbytes, seed):
    g = torch.Generator().manual_seed(seed)
    mem = torch.randint(0, 256, (nbytes,), dtype=torch.uint8, generator=g)
    return mem, (lambda ptr, n: mem[ptr:ptr + n])


def test_round_trip_is_byte_exact_with_small_chunks():
    mem, view = _fake_device(1 << 20, 1)
    allocs = {"weights_0": [A(0, 300_000), A(400_000, 123_457)], "weights": [A(600_000, 400_000)]}
    p = plan_image(allocs, meta={"boot": "t"})
    orig = mem.clone()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "r0.img")
        r = save(path, p, view=view, pin=False, chunk=65_536)
        assert r["bytes"] == p.total_bytes
        mem.zero_()                                  # the pause/resume left garbage
        load(path, allocs, view=view, pin=False, chunk=50_000)
    for rows in allocs.values():
        for a in rows:
            assert torch.equal(mem[a.ptr:a.ptr + a.size], orig[a.ptr:a.ptr + a.size])


def test_corrupted_file_is_caught_by_the_checksum():
    mem, view = _fake_device(1 << 18, 2)
    allocs = {"w": [A(0, 100_000)]}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "r0.img")
        save(path, plan_image(allocs), view=view, pin=False, chunk=32_768)
        with open(path, "r+b") as f:
            f.seek(777)
            b = f.read(1)
            f.seek(777)
            f.write(bytes([(b[0] + 1) % 256]))
        with pytest.raises(ModelImageRefused, match="checksum"):
            load(path, allocs, view=view, pin=False)


def test_manifest_schema_is_checked():
    with pytest.raises(ModelImageRefused, match="schema"):
        ImagePlan.from_manifest({"schema": "x", "entries": [], "total_bytes": 0, "file_bytes": 0})


def test_load_refuses_a_moved_allocation_before_writing_a_byte():
    mem, view = _fake_device(1 << 18, 3)
    allocs = {"w": [A(0, 50_000)]}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "r0.img")
        save(path, plan_image(allocs), view=view, pin=False, chunk=32_768)
        mem.zero_()
        with pytest.raises(ModelImageRefused, match="missing"):
            load(path, {"w": [A(4096, 50_000)]}, view=view, pin=False)
        assert int(mem.sum()) == 0
