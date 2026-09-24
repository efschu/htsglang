# SPDX-License-Identifier: Apache-2.0
"""fnFL2 H47 -- the arena page load's "kernel" mode never launches a page as
the kernel element when that spills to local memory.

An NF page is 786432 B; as ONE element of the pointer/stride kernel (unroll 1)
it is 24576 B of LocalStorage per thread, and the driver grows the card's LMEM
reservation to it for every resident thread and keeps it (~6.1 GiB on the
5090). The run write's 229376-B element did exactly that on fnFL2x151 (stack
7104 B, +1458 MiB). The page goes as 1-KiB element rows instead, on the module
the run/mamba writes already build (no JIT build at the wake, x66)."""

from __future__ import annotations

import types

import pytest
import torch

from sglang.jit_kernel import hicache as hc
from sglang.srt.mem_cache.pool_host import arena_pool as ap
from sglang.srt.weg2 import arena_write as aw
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

NF_PAGE = 786432          # 12 layers x 64 tokens x 1 KiB
B27_PAGE = 32768


def test_page_load_element():
    assert aw.page_load_element(NF_PAGE) == (1024, 768)
    assert aw.page_load_element(B27_PAGE) == (1024, 32)          # 1024 B/thread whole: split too
    assert aw.page_load_element(8192) == (8192, 1)               # exactly 256 B/thread: kept
    assert aw.page_load_element(512) == (512, 1)
    with pytest.raises(ValueError, match="H47"):
        aw.page_load_element(NF_PAGE + 128)
    # the element launched never spills: at most the warning threshold
    for pb in (NF_PAGE, B27_PAGE, 8192, 512):
        e, _ = aw.page_load_element(pb)
        assert hc.local_bytes_per_thread(e, hc._default_unroll(e)) <= hc.LOCAL_BYTES_WARN_THRESHOLD
    # the whole NF page would be the ~6.1 GiB post the H47 warning names
    assert hc.local_bytes_per_thread(NF_PAGE, hc._default_unroll(NF_PAGE)) == 24576
    assert "LMEM-Reservierung 6120 MiB" in hc.lmem_warning_line(NF_PAGE, 1, 170 * 1536)


def test_split_row_indices():
    idx = aw.split_row_indices(torch.tensor([3, 0, 5]), 4)
    assert idx.dtype == torch.int64
    assert idx.tolist() == [12, 13, 14, 15, 0, 1, 2, 3, 20, 21, 22, 23]


def _emulated_one_layer_mla(calls):
    """jit_kernel.hicache.transfer_hicache_one_layer_mla on CPU tensors: rows of
    element_dim, dst[indices_dst[i]] = src[indices_src[i]]."""
    def fake(*, cache_dst, indices_dst, cache_src, indices_src, element_dim=None,
             unroll=None, block_quota=None):
        calls.append({"element_dim": element_dim, "block_quota": block_quota,
                      "items": int(indices_src.numel())})
        dst = cache_dst.view(-1, element_dim)
        src = cache_src.view(-1, element_dim)
        dst[indices_dst] = src[indices_src]
    return fake


@pytest.mark.parametrize("pb", [NF_PAGE, B27_PAGE, 4096])
def test_the_kernel_mode_gathers_the_same_pages_in_non_spilling_elements(pb, monkeypatch):
    monkeypatch.delenv("SGLANG_HICACHE_ARENA_LOAD_BLOCK_QUOTA", raising=False)
    calls = []
    monkeypatch.setattr(hc, "transfer_hicache_one_layer_mla", _emulated_one_layer_mla(calls))
    A, b = 6, 3
    g = torch.Generator().manual_seed(47)
    view = torch.randint(0, 256, (A, pb), dtype=torch.uint8, generator=g)
    pool = types.SimpleNamespace(_page_bytes=pb, _page_view=view)
    stage = torch.zeros((4, pb), dtype=torch.uint8)
    slots = torch.tensor([4, 1, 5], dtype=torch.int64)

    elem = ap._page_block_kernel(pool, stage, slots, b)

    assert torch.equal(stage[:b], view[slots])                   # page order kept
    assert int(stage[b:].sum()) == 0
    assert len(calls) == 1 and calls[0]["element_dim"] == elem
    assert hc.local_bytes_per_thread(elem, hc._default_unroll(elem)) <= hc.LOCAL_BYTES_WARN_THRESHOLD
    if pb > 8192:
        # split: 1-KiB rows on the write's module (element 1024, quota 16)
        assert elem == 1024 and calls[0]["items"] == b * pb // 1024
        assert calls[0]["block_quota"] == aw.write_block_quota() == 16
    else:
        assert elem == pb and calls[0]["items"] == b


def test_an_explicit_load_quota_still_wins(monkeypatch):
    monkeypatch.setenv("SGLANG_HICACHE_ARENA_LOAD_BLOCK_QUOTA", "8")
    calls = []
    monkeypatch.setattr(hc, "transfer_hicache_one_layer_mla", _emulated_one_layer_mla(calls))
    view = torch.zeros((2, NF_PAGE), dtype=torch.uint8)
    pool = types.SimpleNamespace(_page_bytes=NF_PAGE, _page_view=view)
    ap._page_block_kernel(pool, torch.zeros((1, NF_PAGE), dtype=torch.uint8), torch.tensor([1]), 1)
    assert calls[0]["block_quota"] == 8 and calls[0]["element_dim"] == 1024


def test_the_load_path_goes_through_the_split_helper():
    import inspect

    src = inspect.getsource(ap)
    body = src[src.index("if mode == \"kernel\":\n                try:"):]
    body = body[:body.index("except Exception")]
    assert "_page_block_kernel(self, dev_stage, src_idx, b)" in body
    assert "element_dim=pb" not in body
