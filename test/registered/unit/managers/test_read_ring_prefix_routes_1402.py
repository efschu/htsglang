"""#1402 (boot xsn131, 2026-09-15): the KV and draft page reads of a prefix
borrow their targets from the backend's #720 ring instead of pinning a fresh
page per read.

Measured: D read its store at 250-430 pages/s per rank with the ring switched
on, because the ring served only the extra-pool route (`_read_page`) while
`_generic_page_get` and `_draft_page_get_generic` -- every page of a prefix --
still called `get_dummy_flat_data_page()` per page (one cudaHostAlloc each).
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.read_buffer_pool import ReadBufferPool

PAGE = 64


class _HostPool:
    def __init__(self, rows=8):
        self.page_size = 1
        self.rows = {}
        self.fresh = 0

    def get_dummy_flat_data_page(self):
        self.fresh += 1
        return torch.zeros(PAGE, dtype=torch.uint8)

    def set_from_flat_data_page(self, index, page):
        self.rows[int(index)] = page.clone()


class _Backend:
    """Fills each target with a byte pattern; `missing` keys return None."""

    def __init__(self, ring_capacity=4, missing=(), raise_on=None):
        self._rings = {}
        self.capacity = ring_capacity
        self.missing = set(missing)
        self.raise_on = raise_on

    def _read_buffer_pool(self, pool_name, host_pool):
        if self.capacity <= 0:
            return None
        ring = self._rings.get(pool_name)
        if ring is None:
            ring = ReadBufferPool(
                name=f"test ring {pool_name}",
                flag="TEST",
                capacity=self.capacity,
                page_bytes=PAGE,
                factory=host_pool.get_dummy_flat_data_page,
                register=False,
            )
            self._rings[pool_name] = ring
        return ring

    def batch_get(self, keys, targets):
        if self.raise_on is not None and any(self.raise_on in k for k in keys):
            raise RuntimeError("read raised")
        out = []
        for k, t in zip(keys, targets):
            if k.split(".")[0] in self.missing:
                out.append(None)
            else:
                t.fill_(int(k.split(".")[0][-1]) + 1)  # key 'p3' -> bytes 4
                out.append(t)
        return out


class _Op:
    def __init__(self):
        self.request_id = "rid"
        self.completed_tokens = 0

    def increment(self, n):
        self.completed_tokens += n
        return True


class _Ctl:
    _generic_page_get = HiCacheController._generic_page_get
    _draft_page_get_generic = HiCacheController._draft_page_get_generic

    def __init__(self, backend):
        self.storage_backend = backend
        self.mem_pool_host = _HostPool()
        self.mem_pool_host_draft = _HostPool()
        self.page_size = 1
        self._draft_l3_hits = 0
        self._draft_l3_misses = 0

    def _draft_component_name(self):
        return "draft-abc"


def _idx(n):
    return torch.arange(n, dtype=torch.int64)


def test_kv_route_borrows_from_the_ring_and_returns_every_buffer():
    be = _Backend(ring_capacity=4)
    c = _Ctl(be)
    op = _Op()
    c._generic_page_get(op, ["p0", "p1", "p2"], _idx(3))
    ring = be._rings[PoolName.KV]
    assert ring.available == 4, "every borrowed buffer returned after the copy-out"
    assert ring.overflow_allocations == 0
    # the ring's 4 buffers are the ONLY fresh pages the host pool ever made
    assert c.mem_pool_host.fresh == 4
    assert op.completed_tokens == 3
    assert int(c.mem_pool_host.rows[2][0]) == 3
    # a second batch reuses the ring: no new allocation at all
    c._generic_page_get(op, ["p3", "p4"], _idx(2))
    assert c.mem_pool_host.fresh == 4
    assert int(c.mem_pool_host.rows[1][0]) == 5


def test_draft_miss_on_a_dirty_ring_buffer_still_writes_the_zero_page():
    be = _Backend(ring_capacity=2, missing={"p1"})
    c = _Ctl(be)
    # first batch dirties both ring buffers with p0/p2 bytes
    flags = c._draft_page_get_generic(["p0", "p2"], _idx(2))
    assert flags == [True, True]
    # second batch: p1 misses -- its borrowed buffer holds p0's or p2's bytes
    flags = c._draft_page_get_generic(["p1", "p0"], _idx(2))
    assert flags == [False, True]
    assert int(c.mem_pool_host_draft.rows[0].abs().sum()) == 0, "#993 zero page"
    assert int(c.mem_pool_host_draft.rows[1][0]) == 1
    assert be._rings["draft-abc"].available == 2
    assert c._draft_l3_hits == 3 and c._draft_l3_misses == 1


def test_backend_without_a_ring_keeps_the_fresh_page_path():
    class _NoRing(_Backend):
        _read_buffer_pool = None  # attribute exists but is not callable-ish

    be = _Backend(ring_capacity=0)  # returns None -> fresh pages
    c = _Ctl(be)
    op = _Op()
    c._generic_page_get(op, ["p0", "p1"], _idx(2))
    assert c.mem_pool_host.fresh == 2
    assert op.completed_tokens == 2


def test_a_raised_read_returns_the_borrowed_buffers():
    be = _Backend(ring_capacity=3, raise_on="p9")
    c = _Ctl(be)
    op = _Op()
    try:
        c._generic_page_get(op, ["p0", "p9"], _idx(2))
    except RuntimeError:
        pass
    else:
        raise AssertionError("the read should have raised")
    assert be._rings[PoolName.KV].available == 3


# ---- the batched host write (#1402, A3) ---------------------------------


def _mha_pool(layout, page_size, layer_num=4, head_num=2, head_dim=8, size=64):
    from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

    pool = object.__new__(MHATokenToKVPoolHost)
    pool.layout = layout
    pool.page_size = page_size
    pool.layer_num = layer_num
    pool.head_num = head_num
    pool.head_dim = head_dim
    pool.dtype = torch.bfloat16
    pool.device = "cpu"
    pool.pin_memory = False
    pool.size = size
    if layout == "layer_first":
        pool.kv_buffer = torch.zeros(2, layer_num, size, head_num, head_dim, dtype=torch.bfloat16)
    else:
        pool.kv_buffer = torch.zeros(2, size, layer_num, head_num, head_dim, dtype=torch.bfloat16)
    return pool


def test_batched_host_write_equals_the_per_page_loop():
    for layout in ("layer_first", "page_first"):
        for page_size in (1, 2):
            a = _mha_pool(layout, page_size)
            b = _mha_pool(layout, page_size)
            n_elem = 2 * a.layer_num * page_size * a.head_num * a.head_dim
            pages = [torch.randn(n_elem, dtype=torch.bfloat16) for _ in range(5)]
            starts = [0, 6, 2, 20, 10]  # unordered, non-contiguous, page-aligned
            for s, pg in zip(starts, pages):
                a.set_from_flat_data_page(s, pg)
            b.set_from_flat_data_pages(starts, pages)
            assert torch.equal(a.kv_buffer, b.kv_buffer), (layout, page_size)
            # and a single page takes the per-page route unchanged
            c = _mha_pool(layout, page_size)
            c.set_from_flat_data_pages([6], [pages[1]])
            assert torch.equal(c.kv_buffer[:, :, 6:6 + page_size] if layout == "layer_first" else c.kv_buffer[:, 6:6 + page_size], a.kv_buffer[:, :, 6:6 + page_size] if layout == "layer_first" else a.kv_buffer[:, 6:6 + page_size])


def test_kv_route_writes_the_served_prefix_in_one_batch_and_stops_at_the_miss():
    be = _Backend(ring_capacity=8, missing={"p2"})
    c = _Ctl(be)
    calls = []
    c.mem_pool_host.set_from_flat_data_pages = lambda idx, pages: calls.append((list(idx), len(pages)))
    op = _Op()
    c._generic_page_get(op, ["p0", "p1", "p2", "p3"], _idx(4))
    assert calls == [([0, 1], 2)], "one batched write of the served prefix only"
    assert op.completed_tokens == 2
