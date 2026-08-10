# SPDX-License-Identifier: Apache-2.0
"""#631: row-range backing on the KV pool, the seam's streaming unit.

The arena can now back an arbitrary extent range
(``commit_span``/``decommit_span``, test_kv_arena_span_ops_631.py). This
file pins the layer ABOVE it: turning a range of KV ROWS into the byte
span of each buffer, which is where an off-by-one stops being an
exception and becomes silent KV corruption at a chunk boundary.

THE ROUNDING IS ASYMMETRIC AND IT IS THE WHOLE POINT. A commit must
cover every row that will be WRITTEN, so it rounds OUTWARD. A release
must never drop a row that will still be READ, so it rounds INWARD. The
two directions are pinned separately below because a single "rounds to
the chunk" test would pass with both of them wrong in the same
direction.

Hermetic: a recording stub arena, no GPU, no driver.
"""

import unittest

from sglang.test.test_utils import CustomTestCase

PAGE = 16
TOKENS_PER_ROW = 1
ROW_BYTES = 2048


class _FakeDesc:
    """The parts of a KV buffer descriptor the span maths touches."""

    def __init__(self, name):
        self.name = name
        self.tokens_per_row = TOKENS_PER_ROW
        self.row_bytes = ROW_BYTES

    def _rows(self, num_tokens):
        n = max(int(num_tokens), 0)
        return (n + self.tokens_per_row - 1) // self.tokens_per_row

    def prefix_span_bytes(self, num_tokens, page_size):
        return self._rows(num_tokens) * self.row_bytes

    def final_span_bytes(self, num_tokens, page_size):
        return self._rows(max(int(num_tokens), 0) + page_size) * self.row_bytes


class _RecordingArena:
    """Records the BYTE ranges asked for, so the row->byte map is pinned."""

    def __init__(self):
        self.commits = []
        self.decommits = []
        self._watermark = {}

    def commit_span(self, offset, lo, hi):
        self.commits.append((offset, lo, hi))
        self._watermark[offset] = max(self._watermark.get(offset, 0), hi)
        return max(0, hi - lo)

    def decommit_span(self, offset, lo, hi):
        self.decommits.append((offset, lo, hi))
        return max(0, hi - lo)

    def committed_bytes(self, offset):
        return self._watermark.get(offset, 0)


def _bare_owner(backing, n_buffers=2):
    """A KvVmmBufferOwner with only the fields the span calls touch."""
    owner = object.__new__(backing.KvVmmBufferOwner)
    owner.page_size = PAGE
    owner._reserved_num_tokens = 1000
    owner._final_num_tokens = 1000
    specs = []
    for i in range(n_buffers):
        desc = _FakeDesc(f"buf{i}")
        spec = object.__new__(backing._BufferSpec)
        spec.desc = desc
        spec.offset = i * (1 << 24)
        spec.reserved_span = 1000 * ROW_BYTES
        spec.aligned_reserved = 1000 * ROW_BYTES
        spec.backed_to = 0
        specs.append(spec)
    owner._specs = specs
    owner._arena = _RecordingArena()
    return owner


class _Base(CustomTestCase):
    def setUp(self):
        import sglang.srt.mem_cache.kv_vmm_backing as backing

        self.backing = backing
        self.owner = _bare_owner(backing)
        self.arena = self.owner._arena


class TestRowToByteMap(_Base):
    def test_back_token_span_asks_for_the_rows_byte_range(self):
        self.owner.back_token_span(100, 200)
        self.assertEqual(len(self.arena.commits), 2, "both buffers must move")
        off0, lo, hi = self.arena.commits[0]
        self.assertEqual(off0, 0)
        self.assertEqual(lo, 100 * ROW_BYTES)
        self.assertEqual(
            hi,
            (200 + PAGE) * ROW_BYTES,
            "the top must include the padded page, or the last partial page "
            "of the range is written into unbacked memory",
        )

    def test_a_buffer_subset_moves_only_those_buffers(self):
        self.owner.back_token_span(0, 50, buffer_indices=[1])
        self.assertEqual(len(self.arena.commits), 1)
        self.assertEqual(self.arena.commits[0][0], self.owner._specs[1].offset)


class TestReleaseRoundsInward(_Base):
    def test_release_token_span_never_drops_a_row_still_in_range(self):
        self.owner.release_token_span(100, 200)
        off, lo, hi = self.arena.decommits[0]
        self.assertEqual(
            lo,
            (100 + PAGE) * ROW_BYTES,
            "the bottom of a release must round UP past the padded page: "
            "row 100's page may still hold rows below 100",
        )
        self.assertEqual(
            hi,
            200 * ROW_BYTES,
            "the top of a release must round DOWN, so a page straddling the "
            "boundary is kept rather than unmapped",
        )

    def test_commit_and_release_of_the_same_range_are_not_symmetric(self):
        """The asymmetry itself, pinned. If someone 'tidies' these into one
        helper the seam starts unmapping live rows."""
        self.owner.back_token_span(100, 200)
        self.owner.release_token_span(100, 200)
        _o, c_lo, c_hi = self.arena.commits[0]
        _o2, r_lo, r_hi = self.arena.decommits[0]
        self.assertLess(c_lo, r_lo, "commit must start at or below release")
        self.assertGreater(c_hi, r_hi, "commit must end at or above release")


class TestWatermarkIsCarriedBack(_Base):
    def test_backed_to_follows_the_arena_watermark(self):
        """``backed_to`` is what the legacy prefix path skips on. If a span
        op leaves it stale, a later whole-pool restore silently skips rows
        it never backed."""
        self.owner.back_token_span(0, 300)
        self.assertEqual(
            self.owner._specs[0].backed_to,
            self.arena.committed_bytes(0),
        )


if __name__ == "__main__":
    unittest.main()
