"""#631: a span's byte range is normalised to the GRANULARITY, not the chunk.

WHAT THIS PINS, and it is a crash rather than an inefficiency.
``commit_span`` rounded its range outward to the COMMIT CHUNK while
``commit_range`` has always rounded to the allocation GRANULARITY. Buffer
VA extents inside the arena are laid out granularity-aligned, so a
chunk-rounded ``hi`` overshoots the end of its own buffer by up to
chunk-1 bytes and asks the driver to map over the NEXT buffer's live
mapping. cuMemMap answers CUDA_ERROR_INVALID_VALUE.

Measured 2026-08-10 18:07:40 on the first boot where the streamed seam
actually engaged: all three ranks raised out of
``_stream_wave -> restore_wave_span -> commit_span`` and the instance
died on SIGQUIT. The exception lands inside the flip's no-return region,
so there is no degraded mode -- it takes the server.

It hid for two reasons worth keeping: the legacy whole-pool path never
produces a span ending anywhere but at a buffer's own end, and the span
substrate's own tests never ran against a real arena with a NEIGHBOUR to
collide with. So "built and tested" was true and still insufficient.

The chunk remains the handle SIZE cap. It is not an alignment.
"""

from __future__ import annotations

import unittest

from sglang.srt.mem_cache.kv_vmm_backing import KvVmmArena
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GRAN = 2 << 20  # 2 MiB, the value this rig's cards report
CHUNK = 16 << 20


def bounds(lo, hi, outward):
    return KvVmmArena.span_bounds(lo, hi, GRAN, outward)


class OutwardRoundingStaysInsideTheBuffer(unittest.TestCase):
    def test_hi_rounds_to_granularity_not_to_the_chunk(self):
        """The crash, stated as an upper bound.

        A buffer whose reserved span ends 2 MiB past a chunk boundary must
        not have its commit rounded up by the remaining 14 MiB.
        """
        end = CHUNK + GRAN  # 18 MiB: granularity-aligned, NOT chunk-aligned
        _lo, hi = bounds(0, end, True)
        self.assertEqual(
            hi,
            end,
            "hi was rounded past the buffer's own VA extent, which maps over "
            "the neighbouring buffer's live pages (cuMemMap INVALID_VALUE)",
        )
        self.assertLessEqual(hi, end)

    def test_a_partial_granule_still_rounds_up(self):
        """Outward must still COVER every byte asked for."""
        _lo, hi = bounds(0, GRAN + 1, True)
        self.assertEqual(hi, 2 * GRAN)

    def test_lo_rounds_down(self):
        lo, _hi = bounds(GRAN + 1, 8 * GRAN, True)
        self.assertEqual(lo, GRAN)

    def test_an_already_aligned_span_is_unchanged(self):
        self.assertEqual(bounds(2 * GRAN, 6 * GRAN, True), (2 * GRAN, 6 * GRAN))


class InwardRoundingNeverCoversALiveGranule(unittest.TestCase):
    """Release is the direction where over-reach is silent corruption."""

    def test_lo_rounds_up_and_hi_rounds_down(self):
        lo, hi = bounds(GRAN + 1, 6 * GRAN - 1, False)
        self.assertEqual((lo, hi), (2 * GRAN, 5 * GRAN))

    def test_a_span_narrower_than_a_granule_releases_nothing(self):
        lo, hi = bounds(GRAN + 1, GRAN + 2, False)
        self.assertGreaterEqual(lo, hi, "an empty inward span must not invert")

    def test_negatives_are_clamped_rather_than_wrapped(self):
        self.assertEqual(bounds(-5, 4 * GRAN, True), (0, 4 * GRAN))


class TheTwoDirectionsDisagreeOnPurpose(unittest.TestCase):
    def test_outward_contains_inward(self):
        """Whatever release covers, commit must also have covered.

        If this ever inverts, a range would be released that was never
        mapped, or mapped and never releasable.
        """
        for lo, hi in ((1, 3 * GRAN + 7), (GRAN - 1, 9 * GRAN + 1), (0, GRAN)):
            olo, ohi = bounds(lo, hi, True)
            ilo, ihi = bounds(lo, hi, False)
            if ihi > ilo:
                self.assertLessEqual(olo, ilo)
                self.assertGreaterEqual(ohi, ihi)


if __name__ == "__main__":
    unittest.main()
