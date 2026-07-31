# SPDX-License-Identifier: Apache-2.0
"""#352: a captured store_kvcache must still accept slot ids created by a
later #330 capacity growth.

``store_cache`` passes ``size_limit`` as a by-value kernel parameter, so a
CUDA graph replays the bound it saw AT CAPTURE TIME. Under #330 the KV pool's
``size`` grows after capture (physical pages are mapped into the boot VA
reservation, addresses never move, graphs are never re-captured), and the
token allocator starts handing out ids above the boot ceiling. A graph that
baked in the pre-growth bound then fails
``index >= 0 && index < size_limit`` on a LEGAL id -- an assert that reports
as KV corruption but is really the guard being stale.

This is the mechanism test for that, on real hardware and in seconds: capture
a graph over a buffer that is larger than the pool's current span, then write
through it at an index only a post-growth allocator would produce. It needs no
model, no TP group and no dial -- exactly the shape #352 escaped detection in,
because every hermetic test used a pool whose size never changed after capture.

    python -m pytest test/registered/mem_cache/test_kv_store_bound_after_growth.py -v
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu")

# The failing rank's geometry from the #330 card run: fp8 KV, 1 kv head x
# head_dim 256 = 256 row bytes (kElementBytes=256 in the crash signature).
ROW_DIM = 256
STORE_DTYPE = torch.float8_e4m3fn
PAGE_SIZE = 1
# Scaled-down stand-ins for C_boot = 251965 and the 522197-row VA reserve.
BOOT_ROWS = 1024
RESERVE_ROWS = 4096
# A slot id only a grown allocator hands out (251965 < id <= 341861 upstream).
GROWN_ID = 2048


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestKvStoreBoundAfterGrowth(CustomTestCase):
    def _buffers(self):
        """Allocate the KV buffers, THEN check that the JIT kernel is usable.

        Order matters: loading the kvcache JIT module before this process has
        allocated its first CUDA tensor yields a module bound to a context
        torch then replaces, and every later launch dies with "invalid
        resource handle". Allocating first is also what the production path
        does -- the pool exists long before the first store.
        """
        dev = torch.device("cuda")
        k_cache = torch.zeros(RESERVE_ROWS, ROW_DIM, dtype=STORE_DTYPE, device=dev)
        v_cache = torch.zeros(RESERVE_ROWS, ROW_DIM, dtype=STORE_DTYPE, device=dev)
        k = torch.ones(1, ROW_DIM, dtype=STORE_DTYPE, device=dev)
        v = torch.ones(1, ROW_DIM, dtype=STORE_DTYPE, device=dev)

        from sglang.jit_kernel.kvcache import can_use_store_cache

        if not can_use_store_cache(ROW_DIM * STORE_DTYPE.itemsize):
            # Without the JIT kernel _set_kv_buffer_impl takes the indexing
            # fallback, which carries no bound and cannot exercise #352.
            self.skipTest("store_cache JIT kernel unavailable for this row size")
        return k, v, k_cache, v_cache

    def _store(self, k, v, k_cache, v_cache, loc, size_limit):
        from sglang.srt.mem_cache.memory_pool import _set_kv_buffer_impl

        _set_kv_buffer_impl(
            k,
            v,
            k_cache,
            v_cache,
            loc,
            row_dim=ROW_DIM,
            store_dtype=STORE_DTYPE,
            device_module=torch.cuda,
            size_limit=size_limit,
            alt_stream=None,
            same_kv_dim=True,
        )

    def test_bound_helper_admits_every_reachable_capacity(self):
        from sglang.srt.mem_cache.memory_pool import graph_safe_store_bound

        # Pre-growth live bound, VA-reserve-sized buffers.
        bound = graph_safe_store_bound(BOOT_ROWS + PAGE_SIZE, RESERVE_ROWS)
        self.assertGreater(bound, GROWN_ID)
        # Off the dial lane (buffers exactly size + page) nothing widens.
        self.assertEqual(
            graph_safe_store_bound(BOOT_ROWS + PAGE_SIZE, BOOT_ROWS + PAGE_SIZE),
            BOOT_ROWS + PAGE_SIZE,
        )

    def test_captured_graph_accepts_a_post_growth_slot_id(self):
        """THE #352 regression. Pre-fix this replay dies with
        ``Assertion index >= 0 && index < size_limit failed`` and takes the
        CUDA context with it."""
        k, v, k_cache, v_cache = self._buffers()
        loc = torch.zeros(1, dtype=torch.int64, device="cuda")

        # Warm the JIT + allocator outside the capture, on a side stream.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._store(k, v, k_cache, v_cache, loc, BOOT_ROWS + PAGE_SIZE)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            # Captured with the PRE-GROWTH live bound, exactly as the pool
            # passes it: this is the number the graph keeps forever.
            self._store(k, v, k_cache, v_cache, loc, BOOT_ROWS + PAGE_SIZE)

        # ... the dial grows the pool; the allocator hands out GROWN_ID.
        loc.fill_(GROWN_ID)
        k_cache.zero_()
        graph.replay()
        torch.cuda.synchronize()

        written = k_cache[GROWN_ID].to(torch.float32)
        self.assertTrue(
            bool((written != 0).all()),
            f"row {GROWN_ID} was not written by the replayed graph",
        )
        # Nothing else moved.
        self.assertEqual(int((k_cache.to(torch.float32) != 0).any(dim=1).sum()), 1)

    def test_eager_store_at_a_post_growth_slot_id(self):
        """The same id through an eager launch (the non-captured prefill
        path), which re-reads the bound every time and was never broken."""
        k, v, k_cache, v_cache = self._buffers()
        loc = torch.full((1,), GROWN_ID, dtype=torch.int64, device="cuda")
        self._store(k, v, k_cache, v_cache, loc, GROWN_ID + PAGE_SIZE)
        torch.cuda.synchronize()
        self.assertTrue(bool((k_cache[GROWN_ID].to(torch.float32) != 0).all()))


if __name__ == "__main__":
    unittest.main()
