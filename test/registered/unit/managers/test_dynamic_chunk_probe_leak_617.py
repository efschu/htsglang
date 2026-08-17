# SPDX-License-Identifier: Apache-2.0
"""#617/#661: --enable-dynamic-chunking leaked every Mamba slot at boot.

MEASURED, F1's boot_fail_dynchunk.log, 2026-08-15 13:38:

    Profiling prefill latency for dynamic chunking:   9%| | 12/128
    mamba state slot pool exhausted and nothing evictable ... pool=12
    [PP Dynamic Chunk] [PP0] profiling failed (alloc_req_slots runs out of
      memory ...)
    ValueError: pool memory leak detected!
      [full]  total=509621, available=509621
      [mamba] total=12, available=0, leaked_mamba_pages={1,...,12}

TWO FACTS SETTLE THE DIAGNOSIS AND BOTH ARE IN THAT DUMP. It failed at exactly
12 of 128 -- the mamba pool size -- and the FULL pool came back untouched. So
this is not a chunk-size change racing the seam's slot accounting, and it is
not about the seam at all: the instance died in `on_idle` at boot, before it
served a request.

`ReqToTokenPool.free` releases the REQ slot only. The mamba state has its own
allocator and its own call, `free_mamba_cache`, and the profiler never made
it -- so each probe consumed one mamba slot permanently.

ORDER IS LOAD-BEARING: `free_mamba_cache` reads
`req_index_to_mamba_ping_pong_track_buffer_mapping[req.req_pool_idx]`, and
`free` nulls `req_pool_idx`. Freeing the req slot first either raises or frees
the wrong ping-pong buffer.
"""

import types
import unittest

from sglang.srt.managers.scheduler_pp_mixin import _release_dynamic_chunk_probe


class _Req:
    def __init__(self, idx=3, mamba=7, end=128):
        self.req_pool_idx = idx
        self.mamba_pool_idx = mamba
        self.extend_range = types.SimpleNamespace(end=end)


class _Pool:
    def __init__(self):
        self.freed_reqs = []
        self.freed_mamba = []
        self.order = []
        self.req_to_token = {}

    def __getitem__(self, key):  # pragma: no cover - not used
        return None

    def free(self, req):
        assert req.req_pool_idx is not None, "double free of the req slot"
        self.order.append("req")
        self.freed_reqs.append(req.req_pool_idx)
        req.req_pool_idx = None

    def free_mamba_cache(self, req):
        assert req.req_pool_idx is not None, (
            "free_mamba_cache needs req_pool_idx: it indexes the ping-pong "
            "buffer mapping by it, so the req slot must still be held"
        )
        self.order.append("mamba")
        self.freed_mamba.append(req.mamba_pool_idx)
        req.mamba_pool_idx = None


class _Alloc:
    def __init__(self):
        self.freed = 0

    def free(self, indices):
        self.freed += 1


class _Sched:
    def __init__(self):
        self.req_to_token_pool = _Pool()
        self.token_to_kv_pool_allocator = _Alloc()
        # req_to_token[idx, :end] -> anything sliceable
        self.req_to_token_pool.req_to_token = _Table()


class _Table:
    def __getitem__(self, key):
        return [0, 1, 2]


class TheProbeReleasesEverythingItTook(unittest.TestCase):
    def test_the_mamba_slot_is_released(self):
        """THE BUG. Twelve probes took twelve slots and returned none."""
        s, r = _Sched(), _Req()
        _release_dynamic_chunk_probe(s, r)
        self.assertEqual(s.req_to_token_pool.freed_mamba, [7])
        self.assertIsNone(r.mamba_pool_idx)

    def test_the_req_slot_and_kv_are_still_released(self):
        s, r = _Sched(), _Req()
        _release_dynamic_chunk_probe(s, r)
        self.assertEqual(s.req_to_token_pool.freed_reqs, [3])
        self.assertEqual(s.token_to_kv_pool_allocator.freed, 1)

    def test_mamba_is_released_BEFORE_the_req_slot(self):
        """The ordering constraint, asserted by the fake: free_mamba_cache
        indexes the ping-pong mapping by req_pool_idx, which free() nulls."""
        s, r = _Sched(), _Req()
        _release_dynamic_chunk_probe(s, r)
        self.assertEqual(s.req_to_token_pool.order, ["mamba", "req"])

    def test_twelve_probes_return_twelve_slots(self):
        """The exact shape of the failure: pool=12, and it died at probe 12."""
        s = _Sched()
        for i in range(12):
            _release_dynamic_chunk_probe(s, _Req(idx=i + 1, mamba=i + 1))
        self.assertEqual(len(s.req_to_token_pool.freed_mamba), 12)

    def test_a_partially_allocated_probe_is_safe(self):
        """The abort path: the request that RAISED may hold some slots and not
        others. Cleanup after a failure must not become a second failure."""
        s = _Sched()
        r = _Req(mamba=None)  # took a req slot, never got a mamba one
        _release_dynamic_chunk_probe(s, r)
        self.assertEqual(s.req_to_token_pool.freed_mamba, [])
        self.assertEqual(s.req_to_token_pool.freed_reqs, [3])

    def test_none_is_a_no_op(self):
        s = _Sched()
        _release_dynamic_chunk_probe(s, None)
        self.assertEqual(s.req_to_token_pool.freed_reqs, [])

    def test_it_is_idempotent(self):
        """It runs per iteration AND on the abort path; the two must not
        double-free."""
        s, r = _Sched(), _Req()
        _release_dynamic_chunk_probe(s, r)
        _release_dynamic_chunk_probe(s, r)
        self.assertEqual(s.req_to_token_pool.freed_reqs, [3])
        self.assertEqual(s.req_to_token_pool.freed_mamba, [7])

    def test_a_raising_pool_does_not_escape(self):
        class _Boom(_Pool):
            def free_mamba_cache(self, req):
                raise RuntimeError("allocator went away")

        s, r = _Sched(), _Req()
        s.req_to_token_pool = _Boom()
        s.req_to_token_pool.req_to_token = _Table()
        _release_dynamic_chunk_probe(s, r)  # must not raise
        self.assertEqual(s.req_to_token_pool.freed_reqs, [3])


if __name__ == "__main__":
    unittest.main()


class TheOldReleaseSequenceLeaked(unittest.TestCase):
    """The defect replayed, so the fix cannot be undone silently.

    The profiler's release block was exactly:

        kv_indices = req_to_token_pool.req_to_token[idx, :end]
        token_to_kv_pool_allocator.free(kv_indices)
        req_to_token_pool.free(req)

    Correct for KV and the req slot, silent about mamba. Twelve probes, twelve
    slots, none returned -- and the pool report proves the halves apart:
    [full] available == total, [mamba] available == 0.
    """

    def _old_release(self, sched, req):
        kv = sched.req_to_token_pool.req_to_token[
            req.req_pool_idx, : req.extend_range.end
        ]
        sched.token_to_kv_pool_allocator.free(kv)
        sched.req_to_token_pool.free(req)

    def test_the_old_sequence_returns_kv_and_the_req_slot(self):
        s, r = _Sched(), _Req()
        self._old_release(s, r)
        self.assertEqual(s.token_to_kv_pool_allocator.freed, 1)
        self.assertEqual(s.req_to_token_pool.freed_reqs, [3])

    def test_and_leaks_the_mamba_slot_every_time(self):
        s = _Sched()
        for i in range(12):
            self._old_release(s, _Req(idx=i + 1, mamba=i + 1))
        self.assertEqual(
            s.req_to_token_pool.freed_mamba,
            [],
            "twelve probes, twelve mamba slots, none returned -- which is the "
            "boot failure at exactly 12 of 128",
        )

    def test_the_fix_returns_what_the_old_sequence_kept(self):
        s = _Sched()
        for i in range(12):
            _release_dynamic_chunk_probe(s, _Req(idx=i + 1, mamba=i + 1))
        self.assertEqual(len(s.req_to_token_pool.freed_mamba), 12)
