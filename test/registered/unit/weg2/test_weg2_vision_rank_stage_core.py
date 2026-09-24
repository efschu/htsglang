"""The in-rank vision stage, model-neutral core (user design 2026-09-24).

Hermetic, CPU. Pins: the KV tail comes out of the free list only when it is
wholly free and goes back to the END; the tail's bytes are the buffers' own
rows (both layouts); tensors never straddle a segment; a meta-built module's
parameters become views on the slab (buffers stay real); the checkpoint is
read straight into the views -- O_DIRECT when the filesystem takes it, a
NAMED buffered fallback otherwise -- and the pairing refuses unmapped or
unfilled names.
"""

import errno
import os
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.weg2 import vision_rank_stage as vrs


class _Alloc:
    def __init__(self, num_pages, page_size=1, need_sort=True):
        self.num_pages = num_pages
        self.page_size = page_size
        self.size = num_pages * page_size
        self.need_sort = need_sort
        self.free_pages = torch.arange(1, num_pages + 1, dtype=torch.int64)
        self.release_pages = torch.empty((0,), dtype=torch.int64)


def test_tail_reserved_only_when_wholly_free_and_returned_at_the_end():
    a = _Alloc(100)
    res = vrs.reserve_tail_pages(a, 10)
    assert (res.lo_page, res.hi_page, res.slots) == (91, 100, 10)
    assert int(a.free_pages.max()) == 90 and a.free_pages.numel() == 90
    vrs.return_tail_pages(a, res)
    assert a.free_pages.numel() == 100
    assert a.free_pages[-10:].tolist() == list(range(91, 101))  # tail stays last
    # a request holds page 95 -> the tail is not wholly free -> refused
    a.free_pages = a.free_pages[a.free_pages != 95]
    before = a.free_pages.clone()
    assert vrs.reserve_tail_pages(a, 10) is None
    assert torch.equal(a.free_pages, before)  # untouched on refusal
    # pages pending in release_pages count as free
    a2 = _Alloc(20)
    a2.free_pages = torch.arange(1, 16, dtype=torch.int64)
    a2.release_pages = torch.arange(16, 21, dtype=torch.int64)
    res2 = vrs.reserve_tail_pages(a2, 5)
    assert res2 is not None and a2.release_pages.numel() == 0
    assert a2.free_pages.tolist() == list(range(1, 16))


@pytest.mark.parametrize("page_size,layout", [(1, "rows"), (64, "rows"), (64, "vec")])
def test_tail_segments_alias_the_buffers_rows(page_size, layout):
    num_pages = 8
    heads, dim = 2, 16
    if layout == "rows":
        shape = ((num_pages + 1) * page_size, heads, dim)
    else:
        shape = (num_pages + 1, heads, dim // 4, page_size, 4)
    bufs = [torch.zeros(shape, dtype=torch.uint8) for _ in range(4)]
    res = vrs.TailReservation(lo_page=7, hi_page=8, page_size=page_size)
    segs = vrs.tail_segments(bufs, res, num_pages)
    per_page = bufs[0].numel() // (num_pages + 1)
    assert all(s.numel() == 2 * per_page for s in segs)
    segs[1].fill_(7)
    flat = bufs[1].reshape(-1)
    assert int(flat[: 7 * per_page].sum()) == 0
    assert bool((flat[7 * per_page:] == 7).all())
    assert vrs.bytes_per_slot(bufs, num_pages, page_size) == 4 * per_page // page_size


def test_slab_never_straddles_and_refuses_overflow():
    segs = [torch.zeros(1000, dtype=torch.uint8), torch.zeros(1000, dtype=torch.uint8)]
    slab = vrs.SlabAllocator(segs, align=256)
    a = slab.take(600)
    b = slab.take(600)  # does not fit behind a (256-aligned 768 + 600 > 1000) -> segment 2
    assert a.data_ptr() == segs[0].data_ptr()
    assert b.data_ptr() == segs[1].data_ptr()
    with pytest.raises(vrs.VisionRankStageRefused):
        slab.take(600)


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(8, 4)
        self.norm = torch.nn.LayerNorm(4)
        self.register_buffer("scale", torch.full((4,), 2.0), persistent=False)


def test_meta_built_module_lands_on_the_slab_buffers_stay_real():
    with vrs.params_on_meta():
        m = _Tiny()
    assert all(p.device.type == "meta" for p in m.parameters())
    assert m.scale.device.type == "cpu" and float(m.scale[0]) == 2.0
    seg = torch.zeros(4096, dtype=torch.uint8)
    views = vrs.place_parameters(m, vrs.SlabAllocator([seg]))
    assert set(views) == {"fc.weight", "fc.bias", "norm.weight", "norm.bias"}
    lo, hi = seg.data_ptr(), seg.data_ptr() + seg.numel()
    for name, p in m.named_parameters():
        assert lo <= p.data_ptr() < hi, name
        assert p.data_ptr() == views[name].data_ptr()
    views["fc.weight"].fill_(1.0)
    assert float(m.fc.weight.sum()) == 32.0  # the module reads the slab


def _write_ckpt(path, tensors):
    from safetensors.torch import save_file

    save_file(tensors, str(path))


def _fake_view_dict(tensors):
    return {k: torch.empty_like(v) for k, v in tensors.items()}


@pytest.mark.parametrize("direct_ok", [True, False])
def test_read_into_fills_views_exactly_and_names_the_route(tmp_path, monkeypatch, direct_ok):
    torch.manual_seed(0)
    tensors = {
        "model.visual.blocks.0.attn.qkv.weight": torch.randn(48, 16).to(torch.bfloat16),
        "model.visual.blocks.0.attn.qkv.bias": torch.randn(48).to(torch.bfloat16),
        "model.language_model.x": torch.randn(7, 3),
        "model.visual.merger.linear_fc1.weight": torch.randn(33, 5).to(torch.bfloat16),
    }
    shard = tmp_path / "model-00001-of-00001.safetensors"
    _write_ckpt(shard, tensors)
    cks = vrs.checkpoint_tensors(str(shard), lambda n: "visual" in n)
    assert {c.name for c in cks} == {n for n in tensors if "visual" in n}
    assert [c.file_offset for c in cks] == sorted(c.file_offset for c in cks)
    views = {c.name: torch.empty(c.shape, dtype=c.dtype) for c in cks}
    plan = vrs.plan_checkpoint_into(views, cks, lambda n: n)

    real_open = os.open
    calls = []

    def fake_open(path, flags, *a):
        calls.append(flags)
        if flags & os.O_DIRECT:
            if not direct_ok:
                raise OSError(errno.EINVAL, "no O_DIRECT here")
            return real_open(path, flags & ~os.O_DIRECT, *a)  # hermetic: tmpfs
        return real_open(path, flags, *a)

    monkeypatch.setattr(vrs.os, "open", fake_open)
    rep = vrs.read_into(str(shard), plan, bounce_bytes=8192)
    assert rep.direct is direct_ok
    assert any(f & os.O_DIRECT for f in calls)  # O_DIRECT is always tried first
    for c in cks:
        assert torch.equal(views[c.name], tensors[c.name]), c.name
    assert rep.bytes_read >= sum(c.nbytes for c in cks)


def test_small_bounce_splits_tensors_across_chunks(tmp_path):
    t = {"visual.big": torch.arange(20000, dtype=torch.int32)}
    shard = tmp_path / "s.safetensors"
    _write_ckpt(shard, t)
    cks = vrs.checkpoint_tensors(str(shard), lambda n: True)
    dst = {"visual.big": torch.zeros(20000, dtype=torch.int32)}
    rep = vrs.read_into(str(shard), vrs.plan_checkpoint_into(dst, cks, lambda n: n),
                        bounce_bytes=8192)
    assert rep.chunks >= 10
    assert torch.equal(dst["visual.big"], t["visual.big"])


def test_pairing_refuses_unmapped_and_unfilled(tmp_path):
    t = {"visual.a": torch.zeros(4), "visual.b": torch.zeros(2)}
    shard = tmp_path / "s.safetensors"
    _write_ckpt(shard, t)
    cks = vrs.checkpoint_tensors(str(shard), lambda n: True)
    with pytest.raises(vrs.VisionRankStageRefused, match="has no view"):
        vrs.plan_checkpoint_into({"visual.a": torch.zeros(4)}, cks, lambda n: n)
    with pytest.raises(vrs.VisionRankStageRefused, match="no checkpoint tensor"):
        vrs.plan_checkpoint_into({"visual.a": torch.zeros(4), "visual.b": torch.zeros(2),
                                  "visual.c": torch.zeros(1)}, cks, lambda n: n)
    with pytest.raises(vrs.VisionRankStageRefused, match="checkpoint"):
        vrs.plan_checkpoint_into({"visual.a": torch.zeros(5), "visual.b": torch.zeros(2)},
                                 cks, lambda n: n)


def test_slots_for_matches_the_placement():
    num_pages, page = 64, 1
    bufs = [torch.zeros((num_pages + 1) * page, 4, 32, dtype=torch.uint8) for _ in range(6)]
    sizes = [300, 700, 128, 64, 900]
    pages = vrs.slots_for(sizes, bufs, num_pages, page, align=64)
    res = vrs.TailReservation(num_pages - pages + 1, num_pages, page)
    slab = vrs.SlabAllocator(vrs.tail_segments(bufs, res, num_pages), align=64)
    for n in sizes:
        slab.take(n)  # fits by construction
    if pages > 1:
        res_small = vrs.TailReservation(num_pages - pages + 2, num_pages, page)
        slab2 = vrs.SlabAllocator(vrs.tail_segments(bufs, res_small, num_pages), align=64)
        with pytest.raises(vrs.VisionRankStageRefused):
            for n in sizes:
                slab2.take(n)


def test_attention_kv_buffers_walks_the_hybrid_wrapper():
    inner = SimpleNamespace(k_buffer=[torch.zeros(2)], v_buffer=[torch.ones(2)])
    outer = SimpleNamespace(full_kv_pool=inner)
    bufs = vrs.attention_kv_buffers(outer)
    assert len(bufs) == 2 and float(bufs[1][0]) == 1.0
    with pytest.raises(vrs.VisionRankStageRefused):
        vrs.attention_kv_buffers(SimpleNamespace())
