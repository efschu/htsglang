"""#583: alloc_req_slots must name the pool that actually ran out.

WHAT WENT WRONG
---------------
Boot 14 (2026-08-05 22:18:58) killed TP0 with

    alloc_req_slots runs out of memory. ... available_size()=4, num_reqs=1

That reads as a contradiction -- four slots free, one request wanted, and
it raised anyway -- and it cost real time to resolve.

The explanation is that ``available_size()`` counts REQUEST slots only. On a
``HybridReqToTokenPool`` the allocation also needs a mamba state (and, with
the extra buffer, a ping-pong slot), and ``alloc`` returns ``None`` if
EITHER is unavailable. The mamba pool was at 96/96. So the message reported
the one pool that was fine and stayed silent about the one that was not.

A fail-loud path that names the wrong resource is worse than a quiet one:
it does not merely fail to help, it actively points away from the cause.

These tests pin that the message names the mamba pool's numbers whenever the
pool is hybrid, and that the request-slot count is labelled rather than left
to be misread as "the" capacity.

Hermetic: no CUDA, no model. Fake pools with the real call under test.
"""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.mem_cache import common as mc  # noqa: E402
from sglang.srt.mem_cache.common import alloc_req_slots  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _MambaAllocator:
    def __init__(self, size, avail):
        self.size = size
        self._avail = avail

    def available_size(self):
        return self._avail

    def schedulable_available_size(self):
        return self._avail


class _PlainPool:
    """Stands in for a non-hybrid ReqToTokenPool: request slots only."""

    def __init__(self, avail):
        self._avail = avail

    def available_size(self):
        return self._avail

    def alloc(self, reqs):
        return None


class _HybridPool(_PlainPool):
    """Boot 14 exactly: request slots free, mamba pool empty."""

    def __init__(self, avail, mamba_size, mamba_avail):
        super().__init__(avail)
        self.mamba_allocator = _MambaAllocator(mamba_size, mamba_avail)
        self.enable_mamba_extra_buffer_lazy = False


class _TreeCache:
    def supports_mamba(self):
        return True

    def evict(self, params):
        return None


class AllocReqSlotsMessageTest(unittest.TestCase):
    def setUp(self):
        # `isinstance(pool, HybridReqToTokenPool)` is the real branch; point it
        # at the stand-in so the message logic can be exercised without a GPU
        # pool. The branch itself is production code.
        self._orig = mc.HybridReqToTokenPool
        mc.HybridReqToTokenPool = _HybridPool

    def tearDown(self):
        mc.HybridReqToTokenPool = self._orig

    # -- THE FALSIFIER: boot 14's exact numbers ---------------------------

    def test_the_message_names_the_mamba_pool_that_actually_ran_out(self):
        pool = _HybridPool(avail=4, mamba_size=96, mamba_avail=0)
        with self.assertRaises(RuntimeError) as ctx:
            alloc_req_slots(pool, [object()], _TreeCache())
        msg = str(ctx.exception)
        # The pool that was empty must appear, with its numbers.
        self.assertIn("mamba_available=0", msg)
        self.assertIn("mamba_total=96", msg)
        # ...and the count that was FINE must be labelled, so it cannot be
        # read as "the" capacity the way boot 14's line was.
        self.assertIn("request slots", msg)

    def test_prefix_the_old_message_alone_would_read_as_a_contradiction(self):
        """Pins why this matters: the request-slot count on its own says
        there was room, which is what sent the boot-14 diagnosis sideways."""
        pool = _HybridPool(avail=4, mamba_size=96, mamba_avail=0)
        with self.assertRaises(RuntimeError) as ctx:
            alloc_req_slots(pool, [object()], _TreeCache())
        msg = str(ctx.exception)
        self.assertIn("available_size()=4", msg)
        self.assertIn("num_reqs=1", msg)
        # 4 >= 1, so without the mamba numbers this is unexplainable.
        self.assertIn("mamba", msg)

    def test_it_points_at_the_unevictable_checkpoint_hypothesis(self):
        """The message should hand the reader the next place to look."""
        pool = _HybridPool(avail=4, mamba_size=96, mamba_avail=0)
        with self.assertRaises(RuntimeError) as ctx:
            alloc_req_slots(pool, [object()], _TreeCache())
        self.assertIn("max-mamba-cache-size", str(ctx.exception))

    # -- and the non-hybrid path must not grow mamba noise ----------------

    def test_a_plain_pool_message_is_unchanged_apart_from_the_label(self):
        mc.HybridReqToTokenPool = _HybridPool  # _PlainPool is not an instance
        pool = _PlainPool(avail=0)
        with self.assertRaises(RuntimeError) as ctx:
            alloc_req_slots(pool, [object()], _TreeCache())
        msg = str(ctx.exception)
        self.assertIn("available_size()=0", msg)
        self.assertNotIn("mamba", msg)

    # -- diagnostics must never mask the failure --------------------------

    def test_a_broken_mamba_allocator_still_raises_the_real_error(self):
        """If the extra numbers cannot be read, the OOM must still surface --
        a diagnostic that can throw would replace the real error with its
        own."""

        class _Broken(_MambaAllocator):
            def available_size(self):
                raise ValueError("allocator is in a bad state")

        pool = _HybridPool(avail=4, mamba_size=96, mamba_avail=0)
        pool.mamba_allocator = _Broken(96, 0)
        with self.assertRaises(RuntimeError) as ctx:
            alloc_req_slots(pool, [object()], _TreeCache())
        self.assertIn("alloc_req_slots runs out of memory", str(ctx.exception))
        self.assertIn("mamba_available=<unavailable>", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
