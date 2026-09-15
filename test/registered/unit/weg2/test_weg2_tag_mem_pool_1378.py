"""#1378 xsn66 -- one caching-allocator pool per weights tag.

MEASURED on weg2xsn66 (846bf739ec): after ``resume(weights_0)`` on P rank 0
the driver read ``model.layers.0.input_layernorm.weight`` (10240 B) as
type=0 (unmapped) and the first copy-out of lane c0 died with SIGSEGV; on P
rank 2 ``model.layers.52.linear_attn.dt_bias`` read type=2 after
``resume(weights_6)``. torch_memory_saver tags per cudaMalloc, i.e. per
caching-allocator SEGMENT, and small tensors share segments across tags:
rank 0 opened its first small segment under the base tag (embed_tokens and
its weight_scale come before layer 0), rank 2 under ``weights_6``.

The fix routes every weights tag into its OWN ``torch.cuda.MemPool`` -- the
base region and every layer band -- so a tag's segments hold that tag's
tensors and nothing else. These tests fake the pool API and the saver's C
entry point exactly as ``test_weg2_dormant_vram_1y._tagged_region`` does, on
a box with no CUDA.

Mutant: the shipped scope, which only called ``tms_set_current_tag``.
"""
from __future__ import annotations

import contextlib
import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS  # noqa: E402
from sglang.srt.managers import weg2_memory_saver as ms  # noqa: E402


class _Pool:
    n = 0

    def __init__(self):
        _Pool.n += 1
        self.id = _Pool.n


class _Cdll:
    def __init__(self):
        self.tags = []

    def tms_set_current_tag(self, raw):
        self.tags.append(raw.decode("utf-8"))


class _Adapter:
    @contextlib.contextmanager
    def region(self, tag, enable_cpu_backup=False):
        yield


@pytest.fixture
def fake_pools(monkeypatch):
    """torch.cuda.MemPool / use_mem_pool faked; entered pools recorded."""
    entered = []

    @contextlib.contextmanager
    def _use_mem_pool(pool):
        entered.append(("enter", pool.id))
        yield
        entered.append(("exit", pool.id))

    monkeypatch.setattr(torch.cuda, "MemPool", _Pool, raising=False)
    monkeypatch.setattr(torch.cuda, "use_mem_pool", _use_mem_pool, raising=False)
    monkeypatch.setattr(ms, "_TAG_MEM_POOLS", {})
    monkeypatch.setattr(ms, "_TAG_POOL_UNAVAILABLE_SAID", False)
    cdll = _Cdll()
    monkeypatch.setattr(ms, "_tms_cdll_in_region", lambda: cdll)
    monkeypatch.setenv("SGLANG_WEG2_WEIGHT_CHUNK_LAYERS", "8")
    monkeypatch.setenv("SGLANG_WEG2_WEIGHT_CHUNKS", "8")
    return entered, cdll


def test_the_base_region_enters_its_own_pool(fake_pools):
    entered, _ = fake_pools
    with ms.weights_region(_Adapter(), GPU_MEMORY_TYPE_WEIGHTS, enable_cpu_backup=False):
        assert entered == [("enter", 1)]
    assert entered == [("enter", 1), ("exit", 1)]
    assert set(ms._TAG_MEM_POOLS) == {GPU_MEMORY_TYPE_WEIGHTS}


def test_a_layer_band_enters_the_bands_pool_not_the_base_one(fake_pools):
    entered, cdll = fake_pools
    with ms.weights_region(_Adapter(), GPU_MEMORY_TYPE_WEIGHTS, enable_cpu_backup=False):
        with ms.weight_chunk_scope(0) as tag:
            assert tag == "weights_0"
            # the band's pool is a DIFFERENT pool from the base region's
            assert entered[-1] == ("enter", ms._TAG_MEM_POOLS["weights_0"].id)
            assert ms._TAG_MEM_POOLS["weights_0"] is not ms._TAG_MEM_POOLS[GPU_MEMORY_TYPE_WEIGHTS]
        # on exit the tag AND the pool return to the base region's
        assert cdll.tags[-1] == GPU_MEMORY_TYPE_WEIGHTS
        assert entered[-1] == ("exit", ms._TAG_MEM_POOLS["weights_0"].id)


def test_the_same_band_reuses_its_pool_and_bands_differ(fake_pools):
    entered, _ = fake_pools
    with ms.weights_region(_Adapter(), GPU_MEMORY_TYPE_WEIGHTS, enable_cpu_backup=False):
        with ms.weight_chunk_scope(3):
            p0a = ms._TAG_MEM_POOLS["weights_0"]
        with ms.weight_chunk_scope(7):
            p0b = ms._TAG_MEM_POOLS["weights_0"]
        with ms.weight_chunk_scope(8):
            p1 = ms._TAG_MEM_POOLS["weights_1"]
    assert p0a is p0b
    assert p1 is not p0a
    assert set(ms._TAG_MEM_POOLS) == {GPU_MEMORY_TYPE_WEIGHTS, "weights_0", "weights_1"}


def test_without_a_pool_api_the_scope_still_yields_and_says_so_once(monkeypatch, caplog):
    monkeypatch.delattr(torch.cuda, "MemPool", raising=False)
    monkeypatch.delattr(torch.cuda, "use_mem_pool", raising=False)
    monkeypatch.setattr(ms, "_TAG_MEM_POOLS", {})
    monkeypatch.setattr(ms, "_TAG_POOL_UNAVAILABLE_SAID", False)
    with caplog.at_level("WARNING"):
        with ms.tag_pool_scope("weights_0") as p:
            assert p is None
        with ms.tag_pool_scope("weights_1") as p:
            assert p is None
    said = [r for r in caplog.records if "WEG2-TAG-POOL UNAVAILABLE" in r.getMessage()]
    assert len(said) == 1, said
