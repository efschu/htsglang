# SPDX-License-Identifier: Apache-2.0
"""#662: the rung must read the depth a shrink can ACT on, not an average.

MEASURED on the 2048-chunk boot, 2026-08-15:

    shrink to 320217 rows reported 0 MiB ...   (x9, current read as 591872)
    KV-BACKING released 2364 MiB by backing 73345 rows instead of 149504  (x6)

Deep targets paid; shallow targets from a high `current` returned nothing.
`decommit_range` frees only extents lying wholly above the keep point PER
BUFFER, so what matters is the SHALLOWEST buffer, not the total.

`backed_bytes` is a sum across buffers. Dividing it by the all-buffers per-row
size yields an AVERAGE depth, which is only the real depth when the backing is
uniform -- and the waved seam releases and restores one layer at a time, so it
is not. The average overstates, the rung picks a keep point above the
shallowest watermark, and the shrink returns zero while looking large.

Same defect class as reading the configured `size` (#662-F4), one level down:
a number that is not the one the shrink acts on.
"""

import unittest


class _Desc:
    def __init__(self, row_bytes=1024, tokens_per_row=1):
        self.row_bytes = row_bytes
        self.tokens_per_row = tokens_per_row


class _Spec:
    def __init__(self, offset, row_bytes=1024):
        self.offset = offset
        self.desc = _Desc(row_bytes)


class _Arena:
    def __init__(self, committed):
        self._c = committed

    def committed_bytes(self, offset):
        return self._c.get(offset, 0)

    @property
    def backed_bytes(self):
        return sum(self._c.values())


class _Owner:
    """The real accessor under test, bound to fakes."""

    def __init__(self, committed):
        from sglang.srt.mem_cache.kv_vmm_backing import KvVmmBufferOwner

        self._arena = _Arena(committed)
        self._specs = [_Spec(off) for off in committed]
        self.uniform_backed_tokens = KvVmmBufferOwner.uniform_backed_tokens.fget(self)


class UniformDepthIsTheMinimum(unittest.TestCase):
    def test_uniform_backing_reads_that_depth(self):
        o = _Owner({0: 100 * 1024, 4096: 100 * 1024})
        self.assertEqual(o.uniform_backed_tokens, 100)

    def test_uneven_backing_reads_the_SHALLOWEST_not_the_average(self):
        """THE BUG. Average says 550; a shrink can only act on 100."""
        o = _Owner({0: 1000 * 1024, 4096: 100 * 1024})
        average = o._arena.backed_bytes // (2 * 1024)
        self.assertEqual(average, 550, "what backed_bytes//bytes_per_row gives")
        self.assertEqual(o.uniform_backed_tokens, 100, "what a shrink can act on")

    def test_a_fully_released_buffer_makes_the_depth_zero(self):
        """One emptied buffer means no shrink can release anywhere."""
        o = _Owner({0: 1000 * 1024, 4096: 0})
        self.assertEqual(o.uniform_backed_tokens, 0)

    def test_tokens_per_row_is_honoured(self):
        o = _Owner({0: 100 * 1024})
        o._specs[0].desc.tokens_per_row = 4
        from sglang.srt.mem_cache.kv_vmm_backing import KvVmmBufferOwner

        self.assertEqual(KvVmmBufferOwner.uniform_backed_tokens.fget(o), 400)

    def test_no_arena_reads_as_nothing_to_give(self):
        from sglang.srt.mem_cache.kv_vmm_backing import KvVmmBufferOwner

        class _Gone:
            _arena = None
            _specs = []

        self.assertEqual(KvVmmBufferOwner.uniform_backed_tokens.fget(_Gone()), 0)

    def test_the_measured_instant(self):
        """591872 by the average, and a keep at 320217 that released nothing:
        reproducible only if some buffer sat below 320217."""
        o = _Owner({0: 900 * 1024, 4096: 300 * 1024})
        self.assertGreater(o._arena.backed_bytes // (2 * 1024), 320)
        self.assertLess(o.uniform_backed_tokens, 320)


if __name__ == "__main__":
    unittest.main()
