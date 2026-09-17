"""weg2xsn267 (17.09.2026): the DFlash draft arena on group D never bound.

    TP0  #1424 arena host pool bind failed (role=draft): ValueError('#1424
         expected a K and a V extent, got [(0, 640), (1024, 640), (2048, 640), ...
    TP1  ... got [(640, 256), (1664, 256), (2688, 256), ...]

A HEAD-SHARDED draft window names its K and V extent PER LAYER: the canonical
draft page is [K L0 | V L0 | K L1 | ...] (5 layers x 2 x 8 heads x 128 x 1 B =
10240 B) and each D rank owns a head slice of every layer (TP0 5 of 8 heads:
640 B at offset 0 / 1024 / ...; TP1/TP2 2 heads: 256 B at 640 / 1664 / ...).
`ArenaMHAHostPool.bind` accepted only the whole-page form (two extents, K all
layers then V all layers). Now 2*L extents are read as per-layer K/V pairs;
the whole-page form is the special case. Both ways of naming the same
whole page must yield identical per-layer references.
"""

from __future__ import annotations

import os
import types

import pytest
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.mem_cache.pool_host import arena_pool as ap  # noqa: E402


class _Arena:
    def __init__(self, slots, page_bytes, data_off=64):
        self.slots = slots
        self._page = page_bytes
        self._off = data_off
        self._mm = bytearray(data_off + slots * page_bytes)
        self._pinned_slots = None

    def data_offset(self):
        return self._off


class _Window:
    def __init__(self, extents, total):
        self.extents = extents
        self.total_bytes = total


def _pool(*, layers, heads, head_dim, staging_rows=8):
    p = ap.ArenaMHAHostPool.__new__(ap.ArenaMHAHostPool)
    p.layer_num = layers
    p.head_num = heads
    p.head_dim = head_dim
    p.dtype = torch.uint8
    p.size = staging_rows
    p.page_size = 1
    p.pin_memory = False
    ap.ArenaMHAHostPool._arena_init_fields(p)      # reads self.size
    return p


def _bind(p, extents, total, slots=4):
    arena = _Arena(slots, total)
    monkey_env = os.environ.get("SGLANG_HICACHE_ARENA_PREPIN")
    os.environ["SGLANG_HICACHE_ARENA_PREPIN"] = "0"
    try:
        p.bind(arena, _Window(extents, total), role="draft", pin=False)
    finally:
        if monkey_env is None:
            os.environ.pop("SGLANG_HICACHE_ARENA_PREPIN", None)
        else:
            os.environ["SGLANG_HICACHE_ARENA_PREPIN"] = monkey_env
    return arena


def _offsets(p):
    e = 1
    base = p.arena_k_refs[0].storage_offset() - p._k_offs[0] // e
    return ([r.storage_offset() - base for r in p.arena_k_refs],
            [r.storage_offset() - base for r in p.arena_v_refs])


def test_d_tp0_five_of_eight_heads_binds_with_per_layer_extents():
    L, H, D = 5, 5, 128
    page = 5 * 2 * 8 * 128                       # 10240
    ext = [((2 * l + kv) * 1024, 640) for l in range(L) for kv in (0, 1)]
    p = _pool(layers=L, heads=H, head_dim=D)
    _bind(p, ext, page)
    assert p.arena is not None and p.row_slot is not None
    assert p._k_offs == [0, 2048, 4096, 6144, 8192]
    assert p._v_offs == [1024, 3072, 5120, 7168, 9216]
    k, v = _offsets(p)
    assert k == p._k_offs and v == p._v_offs
    assert tuple(p.arena_k_refs[0].shape) == (4, H, D)
    assert p.arena_k_refs[1].stride() == (page, D, 1)


def test_d_tp1_two_heads_at_offset_640():
    L, H, D = 5, 2, 128
    page = 10240
    ext = [((2 * l + kv) * 1024 + 640, 256) for l in range(L) for kv in (0, 1)]
    p = _pool(layers=L, heads=H, head_dim=D)
    _bind(p, ext, page)
    assert p._k_offs[0] == 640 and p._v_offs[0] == 1664
    assert p._k_offs[4] == 8832 and p._v_offs[4] == 9856


def test_the_whole_page_named_both_ways_is_the_same_binding():
    L, H, D = 3, 4, 8
    cell = H * D
    page = 2 * L * cell
    whole = _pool(layers=L, heads=H, head_dim=D)
    _bind(whole, [(0, page)], page)
    per_layer = _pool(layers=L, heads=H, head_dim=D)
    # K-major whole page: [K L0..L2][V L0..L2]
    ext = []
    for l in range(L):
        ext.append((l * cell, cell))
        ext.append((L * cell + l * cell, cell))
    _bind(per_layer, ext, page)
    assert whole._k_offs == per_layer._k_offs
    assert whole._v_offs == per_layer._v_offs
    assert _offsets(whole) == _offsets(per_layer)


def test_a_per_layer_extent_of_the_wrong_width_is_refused():
    L, H, D = 2, 2, 8
    page = 2 * L * 8 * 8
    ext = [(0, 16), (64, 16), (128, 24), (192, 16)]   # layer 1's K is 24, cell is 16
    p = _pool(layers=L, heads=H, head_dim=D)
    with pytest.raises(ValueError) as exc:
        _bind(p, ext, page)
    assert "per-layer window extents" in str(exc.value)


def test_three_extents_are_still_refused_by_name():
    p = _pool(layers=2, heads=2, head_dim=8)
    with pytest.raises(ValueError) as exc:
        _bind(p, [(0, 16), (16, 16), (32, 16)], 64)
    assert "expected a K and a V extent" in str(exc.value)
