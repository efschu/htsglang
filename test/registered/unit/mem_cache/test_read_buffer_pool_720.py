"""#720: the read path's pinned spike, and the bounded ring that bounds it.

TODAY, RED: every storage read takes its target from
``host_pool.get_dummy_flat_data_page()``, which allocates a fresh tensor with
``pin_memory=self.pin_memory``. One pinned page per read, transient, and
invisible to the joint pinned budget -- the registry accounts POOLS declared at
attach, and this is neither declared nor a pool. It scales with concurrent
reads rather than tier size, which is the shape a steady-state budget cannot
see.

GREEN: a small ring, allocated once, declared to the registry through the same
door every other pinned consumer uses, and reused across reads.

The first two tests are the falsifier pair: the same number of reads, counted
allocations, with and without the ring. The rest pin the properties that make
the ring safe to turn on -- it is BOUNDED (exhaustion degrades to today's
behaviour instead of blocking a prefetch worker), it is CHARGED (the registry
sees it, and stops seeing it on close), and it is OFF by default.
"""

import unittest

import torch

from sglang.srt.mem_cache import pinned_host_budget as budget
from sglang.srt.mem_cache.read_buffer_pool import ReadBufferPool, borrowed
from sglang.test.test_utils import CustomTestCase

PAGE_BYTES = 512


class _CountingHostPool:
    """A host pool that records how many read targets were allocated."""

    def __init__(self):
        self.allocations = 0

    def get_dummy_flat_data_page(self):
        self.allocations += 1
        return torch.zeros(PAGE_BYTES, dtype=torch.uint8)


def _read_many(pool, host, n):
    for _ in range(n):
        with borrowed(pool, host.get_dummy_flat_data_page) as target:
            self_check = target.numel()
            assert self_check == PAGE_BYTES


class TestReadBufferPool(CustomTestCase):
    def tearDown(self):
        budget.unregister_pinned_post("test-ring")

    # -- the falsifier pair --------------------------------------------------

    def test_red_today_one_allocation_per_read(self):
        host = _CountingHostPool()
        _read_many(None, host, 25)
        self.assertEqual(host.allocations, 25)

    def test_green_the_ring_allocates_once_and_reuses(self):
        host = _CountingHostPool()
        ring = ReadBufferPool(
            name="test-ring",
            flag="SGLANG_HICACHE_READ_BUFFERS",
            capacity=4,
            page_bytes=PAGE_BYTES,
            factory=host.get_dummy_flat_data_page,
            register=False,
        )
        _read_many(ring, host, 25)
        self.assertEqual(host.allocations, 4)
        self.assertEqual(ring.overflow_allocations, 0)

    # -- bounded, not blocking ----------------------------------------------

    def test_concurrent_borrows_beyond_the_ring_fall_back(self):
        """Exhaustion must degrade to today's behaviour, not stall: blocking a
        prefetch worker to save memory trades a bounded spike for unbounded
        latency."""
        host = _CountingHostPool()
        ring = ReadBufferPool(
            name="test-ring",
            flag="f",
            capacity=2,
            page_bytes=PAGE_BYTES,
            factory=host.get_dummy_flat_data_page,
            register=False,
        )
        held = [ring.acquire() for _ in range(5)]
        self.assertEqual(host.allocations, 2 + 3)
        self.assertEqual(ring.overflow_allocations, 3)
        for buf in held:
            ring.release(buf)
        # The ring never grows past its declared capacity.
        self.assertEqual(len(ring._free), 2)

    def test_a_raised_read_still_returns_its_buffer(self):
        host = _CountingHostPool()
        ring = ReadBufferPool(
            name="test-ring",
            flag="f",
            capacity=1,
            page_bytes=PAGE_BYTES,
            factory=host.get_dummy_flat_data_page,
            register=False,
        )
        with self.assertRaises(RuntimeError):
            with borrowed(ring, host.get_dummy_flat_data_page):
                raise RuntimeError("read failed")
        self.assertEqual(len(ring._free), 1)
        _read_many(ring, host, 3)
        self.assertEqual(host.allocations, 1)

    # -- charged to the budget ----------------------------------------------

    def test_the_ring_is_declared_to_the_pinned_registry(self):
        host = _CountingHostPool()
        ring = ReadBufferPool(
            name="test-ring",
            flag="SGLANG_HICACHE_READ_BUFFERS",
            capacity=8,
            page_bytes=PAGE_BYTES,
            factory=host.get_dummy_flat_data_page,
        )
        posts = {p.name: p for p in budget.registered_posts()}
        self.assertIn("test-ring", posts)
        self.assertEqual(posts["test-ring"].nbytes, 8 * PAGE_BYTES)
        ring.close()
        self.assertNotIn("test-ring", {p.name for p in budget.registered_posts()})

    def test_capacity_must_be_positive(self):
        with self.assertRaises(ValueError):
            ReadBufferPool(
                name="test-ring",
                flag="f",
                capacity=0,
                page_bytes=PAGE_BYTES,
                factory=lambda: None,
                register=False,
            )

    # -- off by default ------------------------------------------------------

    def test_default_is_off(self):
        from sglang.srt.environ import envs

        self.assertEqual(int(envs.SGLANG_HICACHE_READ_BUFFERS.get() or 0), 0)


if __name__ == "__main__":
    unittest.main()
