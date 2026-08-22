# SPDX-License-Identifier: Apache-2.0
"""#631: arbitrary-extent backing, the thing a streamed seam needs.

WHY THIS EXISTS. The phase flip's seam has to hand the source layout's
pages back while committing the destination's, and the corridor floor is
a CONTINUOUS minimum -- a peak that lasts milliseconds still counts. To
keep that peak bounded the exchange has to be STREAMED: back a slice of
the destination just ahead of the writes, release a slice of the source
just behind the reads.

The arena could not express that. ``commit_range`` grows a contiguous
watermark up from zero and ``decommit_range`` drops the tail above a keep
point, so both are PREFIX operations. That is not enough, and the reason
is the direction of travel rather than any missing tuning:

* rows ASCENDING -- the destination grows as a prefix [0, t), fine, but
  the source still owes rows [t, N), a SUFFIX, so it can release nothing
  until the end;
* rows DESCENDING -- the source shrinks as a prefix [0, t), fine, but the
  destination is written at its TOP, so it needs [0, N) backed from the
  first write.

Both peak at twice the layout. The source row list and the destination
row list are index-aligned and both ascending, so the two orders are
LOCKED -- one cannot take ascending on one side and descending on the
other. A suffix-capable commit is therefore required and is not
avoidable by scheduling.

Hermetic: a fake driver, no GPU, no stub build -- the same shape as
test_kv_arena_handle_retention_631.py.
"""

import types
import unittest
from unittest import mock

from sglang.test.test_utils import CustomTestCase


class _Result:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


SUCCESS = _Result("CUDA_SUCCESS")
OOM = _Result("CUDA_ERROR_OUT_OF_MEMORY")


class _FakeCUresult:
    CUDA_SUCCESS = SUCCESS
    # _mem_create_reclaiming compares against this on every create, so the
    # fake must carry it even though these tests never provoke an OOM.
    CUDA_ERROR_OUT_OF_MEMORY = OOM


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeDriver:
    CUresult = _FakeCUresult

    def __init__(self):
        self._next_handle = 5000
        self.creates = []
        self.releases = []
        self.maps = []
        self.unmaps = []

    def cuMemCreate(self, step, prop, flags):  # noqa: N802
        self.creates.append(step)
        self._next_handle += 1
        return (SUCCESS, self._next_handle)

    def cuMemMap(self, addr, size, offset, handle, flags):  # noqa: N802
        self.maps.append((addr, size, handle))
        return SUCCESS

    def cuMemSetAccess(self, addr, size, desc, count):  # noqa: N802
        return SUCCESS

    def cuMemUnmap(self, addr, size):  # noqa: N802
        self.unmaps.append((addr, size))
        return SUCCESS

    def cuMemRelease(self, handle):  # noqa: N802
        self.releases.append(handle)
        return SUCCESS


CHUNK = 1 << 20  # 1 MiB granule, so the arithmetic below reads directly
OFF = 0  # one buffer is enough; the seam drives each buffer independently


def _bare_arena(backing, chunk=CHUNK):
    arena = object.__new__(backing.KvVmmArena)
    arena.device_id = 0
    arena.granularity = CHUNK
    arena.base = 0x100000000
    arena.reserved = 1 << 40
    arena._prop = object()
    arena._access = object()
    arena._extents_by_offset = {}
    arena._committed_by_offset = {}
    arena._range_backed = 0
    arena._closed = False
    arena._chunk = chunk
    arena._retain_handles = False
    arena._retained = {}
    arena._retained_bytes = 0
    # #464: mirrors the real constructor's first line, which defaults this OFF
    # -- and off is the per-#330-chunk plan these span assertions are written
    # against. Set explicitly rather than resolved from the environment, so
    # flipping the #464 lever on for a measurement cannot change the shape of
    # the plan under test.
    arena._coalesce_resume = False
    return arena


class _SpanOpsBase(CustomTestCase):
    def setUp(self):
        import sglang.srt.mem_cache.kv_vmm_backing as backing

        self.backing = backing
        self.drv = _FakeDriver()

    def _patched(self):
        return mock.patch.multiple(
            self.backing,
            _driver=lambda: self.drv,
            torch=types.SimpleNamespace(
                cuda=types.SimpleNamespace(
                    device=lambda *a, **k: _NullCtx(),
                    synchronize=lambda *a, **k: None,
                    empty_cache=lambda *a, **k: None,
                    memory_reserved=lambda *a: 0,
                    memory_allocated=lambda *a: 0,
                )
            ),
        )

    def _covered(self, arena, offset=OFF):
        """The set of chunk-aligned starts currently mapped at ``offset``."""
        return sorted(
            rel for rel, _size, _h in arena._extents_by_offset.get(offset, [])
        )


class TestSuffixCommit(_SpanOpsBase):
    """The capability that does not exist today: back a MIDDLE/TOP range."""

    def test_commit_span_backs_a_range_that_does_not_start_at_zero(self):
        arena = _bare_arena(self.backing)
        with self._patched():
            arena.commit_span(OFF, 4 * CHUNK, 7 * CHUNK)
        self.assertEqual(
            self._covered(arena),
            [4 * CHUNK, 5 * CHUNK, 6 * CHUNK],
            "commit_span must back exactly the requested chunks and nothing "
            "below them -- a suffix is the whole point",
        )
        self.assertEqual(arena.backed_bytes, 3 * CHUNK)

    def test_a_suffix_then_the_prefix_below_it_ends_contiguous(self):
        """The seam finishes with the destination fully backed; getting
        there via a suffix must leave the same state as a plain commit."""
        arena = _bare_arena(self.backing)
        with self._patched():
            arena.commit_span(OFF, 4 * CHUNK, 8 * CHUNK)
            arena.commit_span(OFF, 0, 4 * CHUNK)
        self.assertEqual(self._covered(arena), [i * CHUNK for i in range(8)])
        self.assertEqual(arena.backed_bytes, 8 * CHUNK)
        self.assertEqual(
            arena.committed_bytes(OFF),
            8 * CHUNK,
            "once the holes are filled the watermark must agree with the "
            "extents, or the legacy prefix path will mis-skip",
        )

    def test_committing_an_already_backed_span_allocates_nothing(self):
        arena = _bare_arena(self.backing)
        with self._patched():
            arena.commit_span(OFF, 2 * CHUNK, 5 * CHUNK)
            before = len(self.drv.creates)
            arena.commit_span(OFF, 2 * CHUNK, 5 * CHUNK)
            arena.commit_span(OFF, 3 * CHUNK, 4 * CHUNK)
        self.assertEqual(
            len(self.drv.creates),
            before,
            "a re-commit of covered chunks must be free; otherwise the "
            "streamed seam allocates once per block",
        )


class TestSpanDecommit(_SpanOpsBase):
    def test_decommit_span_releases_only_extents_inside_the_range(self):
        arena = _bare_arena(self.backing)
        with self._patched():
            arena.commit_span(OFF, 0, 8 * CHUNK)
            released = arena.decommit_span(OFF, 2 * CHUNK, 5 * CHUNK)
        self.assertEqual(released, 3 * CHUNK)
        self.assertEqual(
            self._covered(arena),
            [0, CHUNK, 5 * CHUNK, 6 * CHUNK, 7 * CHUNK],
            "chunks outside the range must stay mapped -- releasing a "
            "neighbour would unmap rows the flip has not read yet",
        )
        self.assertEqual(arena.backed_bytes, 5 * CHUNK)

    def test_decommitting_an_unbacked_span_releases_nothing(self):
        arena = _bare_arena(self.backing)
        with self._patched():
            arena.commit_span(OFF, 0, 2 * CHUNK)
            released = arena.decommit_span(OFF, 4 * CHUNK, 9 * CHUNK)
        self.assertEqual(released, 0)
        self.assertEqual(self.drv.unmaps, [])


class TestRoundingDirectionIsSafe(_SpanOpsBase):
    """THE correctness property, and it is asymmetric on purpose.

    An unaligned request must never leave a row that will be WRITTEN
    unbacked, and must never release a row that will still be READ. Those
    two pull in opposite directions, so commit rounds OUTWARD and decommit
    rounds INWARD. Getting this backwards produces a fault or silent KV
    corruption at a chunk boundary -- rare, data-dependent, and very
    expensive to find on metal.
    """

    def test_commit_rounds_outward(self):
        arena = _bare_arena(self.backing)
        with self._patched():
            arena.commit_span(OFF, CHUNK + 1, 3 * CHUNK - 1)
        self.assertEqual(
            self._covered(arena),
            [CHUNK, 2 * CHUNK],
            "commit must cover every byte asked for, so lo rounds DOWN and "
            "hi rounds UP",
        )

    def test_decommit_rounds_inward(self):
        arena = _bare_arena(self.backing)
        with self._patched():
            arena.commit_span(OFF, 0, 8 * CHUNK)
            arena.decommit_span(OFF, CHUNK + 1, 5 * CHUNK - 1)
        self.assertEqual(
            self._covered(arena),
            [0, CHUNK, 4 * CHUNK, 5 * CHUNK, 6 * CHUNK, 7 * CHUNK],
            "decommit must release only chunks WHOLLY inside the range, so "
            "lo rounds UP and hi rounds DOWN -- a partially-covered chunk "
            "still holds live rows",
        )


class TestChunkIsRequired(_SpanOpsBase):
    def test_span_ops_refuse_a_monolithic_arena(self):
        """Without a commit chunk each buffer holds ONE extent, so a span
        op would either release far too much or nothing at all. Refuse
        loudly rather than degenerate silently."""
        arena = _bare_arena(self.backing, chunk=None)
        with self._patched():
            with self.assertRaises(RuntimeError) as ctx:
                arena.commit_span(OFF, 0, 2 * CHUNK)
            self.assertIn("chunk", str(ctx.exception).lower())


class TestLegacyPrefixPathAfterASpanRelease(_SpanOpsBase):
    """The hazard a coverage audit found, pinned.

    ``commit_range`` decided what to map from the contiguous watermark
    ALONE. That is right only while coverage is contiguous from zero.
    ``decommit_span`` can leave an interior HOLE, and the watermark then
    reports the prefix BELOW that hole -- so the streamed seam's own
    completion step (``restore_backing`` -> ``finalize`` ->
    ``commit_range``) would re-map extents that were never released:
    cuMemMap over live mappings, and ``backed_bytes`` counting them twice.

    Both calls now share ``_gaps_in``, so they cannot disagree about what
    is already backed.
    """

    def test_commit_range_does_not_remap_a_span_that_is_still_mapped(self):
        arena = _bare_arena(self.backing)
        with self._patched():
            arena.commit_span(OFF, 0, 8 * CHUNK)
            arena.decommit_span(OFF, 2 * CHUNK, 5 * CHUNK)  # interior hole
            self.assertEqual(arena.backed_bytes, 5 * CHUNK)
            maps_before = len(self.drv.maps)
            arena.commit_range(OFF, 8 * CHUNK)  # the completion step
        self.assertEqual(
            arena.backed_bytes,
            8 * CHUNK,
            "the pool must end FULLY backed and counted exactly once",
        )
        self.assertEqual(
            len(self.drv.maps) - maps_before,
            3,
            "only the 3 chunks of the hole may be mapped; re-mapping the "
            "5 that were never released is the bug this pins",
        )
        self.assertEqual(self._covered(arena), [i * CHUNK for i in range(8)])

    def test_the_watermark_reports_the_prefix_below_a_hole(self):
        """Why the bug was reachable at all. Pinned so a future change to
        the watermark's meaning shows up here rather than as a double map."""
        arena = _bare_arena(self.backing)
        with self._patched():
            arena.commit_span(OFF, 0, 8 * CHUNK)
            arena.decommit_span(OFF, 2 * CHUNK, 5 * CHUNK)
        self.assertEqual(arena.committed_bytes(OFF), 2 * CHUNK)


class TestStreamingPeakIsBounded(_SpanOpsBase):
    """The property the whole design turns on, in miniature.

    Two arenas stand in for the two layouts. Walking rows DESCENDING, the
    destination's backing grows as a SUFFIX just ahead of the writes while
    the source's shrinks as a prefix just behind the reads. Their COMBINED
    backed bytes must never exceed one resting layout plus a small
    constant -- that constant is what replaces successor 24's staging
    slope, and it must not scale with the span.
    """

    def _peak_for(self, n_chunks):
        src = _bare_arena(self.backing)
        dst = _bare_arena(self.backing)
        with self._patched():
            src.commit_span(OFF, 0, n_chunks * CHUNK)  # resting source
            resting = src.backed_bytes
            peak = resting
            for t in range(n_chunks, 0, -1):
                lo = (t - 1) * CHUNK
                dst.commit_span(OFF, lo, t * CHUNK)  # back just ahead of writes
                peak = max(peak, src.backed_bytes + dst.backed_bytes)
                src.decommit_span(OFF, lo, t * CHUNK)  # release just behind reads
                peak = max(peak, src.backed_bytes + dst.backed_bytes)
            self.assertEqual(src.backed_bytes, 0)
            self.assertEqual(dst.backed_bytes, resting)
        return resting, peak

    def test_peak_never_exceeds_resting_plus_one_chunk(self):
        resting, peak = self._peak_for(16)
        self.assertEqual(
            peak,
            resting + CHUNK,
            "the streamed seam must hold at most ONE extra chunk over the "
            "resting layout",
        )

    def test_the_overhead_does_not_grow_with_the_span(self):
        """A constant, not a slope. This is the whole capacity argument:
        successor 24's seam cost 4.517 MiB per 1000 live slots, and a
        constant is what lifts the pool ceiling past the 600000 floor."""
        small_rest, small_peak = self._peak_for(8)
        big_rest, big_peak = self._peak_for(64)
        self.assertEqual(small_peak - small_rest, big_peak - big_rest)
        self.assertEqual(
            big_peak - big_rest,
            CHUNK,
            "an 8x longer span must cost the SAME transient, or the slope "
            "is still there and the ceiling has not moved",
        )


if __name__ == "__main__":
    unittest.main()
