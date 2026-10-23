"""barlink-visibility-1023: the BAR1 window must be bound in BUFFER order.

The holder's sg table lists the exported buffer in buffer order
(``nv_dma_buf_map_pfns``), and RM maps it with ``BUS_MAP_FB_FLAGS_ALLOW_DISCONTIG``
(``rm_dma_buf_map_mem_handle``): when no single BAR1 range is free the pieces land
wherever BAR1 VA is free, not necessarily ascending. The pre-1023 binders
(``BarlinkBar1Transport._bind_region`` and the flip lanes' ``connect_peer``)
SORTED the entries by address and took the contiguous run from the lowest one.
Buffer pieces [p0, p1, p2] at BAR1 [X, X+2L, X+L] then passed as one contiguous
3L window, and window offset L reached buffer offset 2L -- a permutation the
64-KiB setup proof at offset 0 cannot see, and one that shows up exactly as
"a slot tile never verified" at fixed offsets.

DESK only (no card): the helper's arithmetic on small and on 32-GiB BAR1
windows (the 5090 after ReBAR), the binder's refusal on a permuted table (red on
the pre-1023 code: it went on to map), and a source invariant for the lanes.
"""

import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from flliper.srt.distributed.device_communicators import barlink_bar1 as b1
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_LANES = (
    Path(__file__).resolve().parents[4] / "python/flliper/srt/pdflip/bar1_lanes.py"
)

MIB = 1 << 20
L = 2 * MIB                      # one VMM page piece


def _sg(*pairs):
    return [b1.SgEntry(int(a), int(n)) for a, n in pairs]


SMALL = b1.Bar1Window(bdf="0000:18:00.0", base=0xD0000000, size=256 * MIB)
BIG = b1.Bar1Window(bdf="0000:15:00.0", base=0x7000000000, size=32 << 30)


class TestPrefix(CustomTestCase):
    def test_ascending_contiguous_is_the_whole_buffer(self):
        X = SMALL.base + 0x410000
        sg = _sg((X, L), (X + L, L), (X + 2 * L, L))
        self.assertEqual(b1.bar1_contiguous_prefix(sg, SMALL), (X, 3 * L, 3))

    def test_permuted_pieces_stop_at_the_first_jump(self):
        X = SMALL.base + 0x410000
        # buffer order p0, p1, p2 -> BAR1 X, X+2L, X+L: sorted it LOOKS contiguous
        sg = _sg((X, L), (X + 2 * L, L), (X + L, L))
        self.assertEqual(b1.bar1_contiguous_prefix(sg, SMALL), (X, L, 1))

    def test_first_piece_not_lowest(self):
        X = SMALL.base + 0x410000
        sg = _sg((X + L, L), (X, L))
        start, length, n = b1.bar1_contiguous_prefix(sg, SMALL)
        self.assertEqual((start, length, n), (X + L, L, 1))

    def test_first_entry_outside_the_window_gives_nothing(self):
        sg = _sg((0x1000, L), (SMALL.base, L))
        self.assertEqual(b1.bar1_contiguous_prefix(sg, SMALL)[1], 0)
        self.assertEqual(b1.bar1_contiguous_prefix([], SMALL), (0, 0, 0))

    def test_run_never_leaves_the_window(self):
        X = SMALL.end - L
        sg = _sg((X, L), (X + L, L))          # second piece would be past BAR1
        self.assertEqual(b1.bar1_contiguous_prefix(sg, SMALL), (X, L, 1))

    def test_big_bar_32_gib(self):
        # 5090 after ReBAR: 32 GiB BAR1 above 4 GiB, seg sizes up to 4 GiB.
        X = BIG.base + (1 << 30)
        sg = _sg((X, 4 << 30), (X + (4 << 30), 2 * L), (X + (4 << 30) + 2 * L, L))
        self.assertEqual(b1.bar1_contiguous_prefix(sg, BIG), (X, (4 << 30) + 3 * L, 3))
        perm = _sg((X, 4 << 30), (X + (4 << 30) + 2 * L, L), (X + (4 << 30), 2 * L))
        self.assertEqual(b1.bar1_contiguous_prefix(perm, BIG), (X, 4 << 30, 1))


class _Mapped(Exception):
    """The binder went past the contiguity check and tried to map."""


class TestBindRegionRefusesPermutation(CustomTestCase):
    def _bind(self, sg, minimum):
        holder = mock.Mock()
        holder.hold.return_value = (7, sg, sum(e.length for e in sg))
        fake = SimpleNamespace(_cuda=mock.Mock(), _holder=holder,
                               bdfs=["0000:15:00.0", "0000:18:00.0"], rank=0)
        with mock.patch.object(b1, "bar1_window", return_value=SMALL), \
                mock.patch.object(b1.os, "open", side_effect=_Mapped):
            return b1.BarlinkBar1Transport._bind_region(
                fake, 1, SMALL.bdf, 5, "payload", minimum)

    def test_permuted_table_is_refused_not_mapped(self):
        X = SMALL.base + 0x410000
        sg = _sg((X, L), (X + 2 * L, L), (X + L, L))
        # pre-1023: sorted -> "contiguous 3L" >= 3L -> went on to os.open (_Mapped)
        with self.assertRaises(b1.Bar1Unavailable):
            self._bind(sg, 3 * L)

    def test_ordered_table_still_maps(self):
        X = SMALL.base + 0x410000
        sg = _sg((X, L), (X + L, L), (X + 2 * L, L))
        with self.assertRaises(_Mapped):
            self._bind(sg, 3 * L)


class TestLanesUseTheSameRule(CustomTestCase):
    def test_connect_peer_walks_buffer_order(self):
        src = _LANES.read_text(encoding="utf-8")
        body = src[src.index("def connect_peer"):]
        body = body[:body.index("\n    def ", 10)]
        self.assertIn("bar1_contiguous_prefix(sg, window)", body)
        self.assertNotIn("key=lambda e: e.dma_address", body)


if __name__ == "__main__":
    unittest.main()
