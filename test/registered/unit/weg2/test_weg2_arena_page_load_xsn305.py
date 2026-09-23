"""Posten 2 (18.09.): whole-page loadback -- pages fetched once, layers split
on the device; later layer calls are no-ops for the same load."""
from __future__ import annotations

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch  # noqa: E402

from sglang.srt.mem_cache.pool_host import arena_pool as ap  # noqa: E402


def _stand_in(A=16, L=3, H=2, D=4, dtype=torch.bfloat16):
    e = dtype.itemsize
    cell = H * D * e
    pb = 2 * L * cell
    page = torch.randint(0, 255, (A, pb), dtype=torch.uint8)
    s = types.SimpleNamespace(
        _page_view=page, _page_bytes=pb,
        _k_offs_b=[2 * l * cell for l in range(L)], _v_offs_b=[(2 * l + 1) * cell for l in range(L)],
        _page_loaded_key=None, _page_stages=None, head_num=H, head_dim=D, dtype=dtype,
        row_slot=None, pin_slots=lambda slots: 0, _arena_load_guard=lambda *a, **k: None,
    )
    pool = types.SimpleNamespace(
        k_buffer=[torch.zeros((32, H, D), dtype=dtype) for _ in range(L)],
        v_buffer=[torch.zeros((32, H, D), dtype=dtype) for _ in range(L)],
    )
    return s, pool, page, cell


def test_pages_land_layer_split_on_the_device_in_blocks():
    os.environ[ap.ARENA_PAGE_LOAD_BLOCK_ENV] = "64"
    s, pool, page, cell = _stand_in()
    slots = torch.tensor([3, 9, 0, 15, 7])
    dst = torch.tensor([20, 21, 22, 23, 24])
    ap.ArenaMHAHostPool._load_pages_all_layers(s, pool, slots, dst)
    for l in range(3):
        for i, sl in enumerate(slots.tolist()):
            ko, vo = s._k_offs_b[l], s._v_offs_b[l]
            # compare BYTES: random bit patterns include NaNs, and NaN != NaN
            assert torch.equal(pool.k_buffer[l][dst[i]].reshape(-1).view(torch.uint8), page[sl, ko:ko + cell])
            assert torch.equal(pool.v_buffer[l][dst[i]].reshape(-1).view(torch.uint8), page[sl, vo:vo + cell])
    assert int(pool.k_buffer[0][0].abs().sum()) == 0  # untouched rows stay


def test_layer_zero_loads_everything_and_later_layers_are_no_ops():
    calls = []
    s, pool, page, cell = _stand_in()
    s._load_pages_all_layers = lambda p, sl, d: calls.append(int(sl.numel()))
    rows = torch.tensor([1, 2, 3]); dst = torch.tensor([5, 6, 7])
    os.environ[ap.ARENA_PAGE_LOAD_ENV] = "1"
    ap.ArenaMHAHostPool._load_arena(s, pool, rows, dst, 0)
    ap.ArenaMHAHostPool._load_arena(s, pool, rows, dst, 1)
    ap.ArenaMHAHostPool._load_arena(s, pool, rows, dst, 2)
    assert calls == [3] and s._page_loaded_key is not None
    # a different load at layer 0 stages again
    rows2 = torch.tensor([4]); ap.ArenaMHAHostPool._load_arena(s, pool, rows2, torch.tensor([8]), 0)
    assert calls == [3, 1]


def test_knob_off_keeps_the_per_layer_path():
    os.environ[ap.ARENA_PAGE_LOAD_ENV] = "0"
    try:
        assert not ap._arena_page_load_on()
    finally:
        os.environ.pop(ap.ARENA_PAGE_LOAD_ENV, None)
    assert ap._arena_page_load_on()
    assert ap._arena_page_load_block() >= 64


def test_the_layer_zero_key_is_the_callers_objects_not_the_per_layer_temporaries():
    """xsn398 (19.09.): `host_indices - S` is a NEW tensor per layer; keying the
    'page-loaded at layer 0' check on its id relied on CPython reusing the freed
    id. With the caller's hint the later layers are no-ops even when every
    layer hands in a fresh temporary; the per-layer kernel never runs."""
    calls, transfers = [], []
    s, pool, page, cell = _stand_in()
    s._load_pages_all_layers = lambda p, sl, d: calls.append(int(sl.numel()))
    s._transfer = lambda *a, **k: transfers.append(1)
    s.row_slot = None
    os.environ[ap.ARENA_PAGE_LOAD_ENV] = "1"
    host = torch.tensor([11, 12, 13]); dst = torch.tensor([5, 6, 7])
    s._page_key_hint = (id(host), id(dst), 3, 3)          # what load_to_device_per_layer sets
    for layer in range(3):
        rows = host - 10                                   # a fresh temporary every layer
        ap.ArenaMHAHostPool._load_arena(s, pool, rows, dst, layer)
    assert calls == [3] and transfers == []
    # a new caller object (the next start_loading) stages again
    host2 = torch.tensor([14]); dst2 = torch.tensor([8])
    s._page_key_hint = (id(host2), id(dst2), 1, 1)
    ap.ArenaMHAHostPool._load_arena(s, pool, host2 - 10, dst2, 0)
    ap.ArenaMHAHostPool._load_arena(s, pool, host2 - 10, dst2, 1)
    assert calls == [3, 1] and transfers == []
