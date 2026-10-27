"""P-CHUNK-MEM-GUARD (27.10.2026): die Wache kappt eine P-Chunkbreite > 2048 am freien Speicher von PP0.
Messung: Boot dkr27bbar1drei10272209, 16384 -> OOM auf der 5090 (107 MiB frei, 192 MiB angefordert)."""
import unittest

from flliper.srt.pdflip.p_chunk_policy import (
    P_MEM_GUARD_MIN_WIDTH,
    mem_guard_width,
)


class TestPChunkMemGuard(unittest.TestCase):
    def test_standard_width_untouched(self):
        for free in (0.0, 300.0, 5000.0):
            self.assertEqual(mem_guard_width(2048, free), 2048)
            self.assertEqual(mem_guard_width(512, free), 512)

    def test_crash_width_is_cut_on_the_measured_card(self):
        # PP0 hatte im 2048er-Betrieb ~1875 MiB frei: 16384 ist nicht tragbar
        w = mem_guard_width(16384, 1875.0)
        self.assertLess(w, 16384)
        self.assertGreaterEqual(w, P_MEM_GUARD_MIN_WIDTH)
        self.assertEqual(w % 512, 0)

    def test_plenty_of_memory_passes_the_width(self):
        self.assertEqual(mem_guard_width(16384, 9000.0), 16384)

    def test_never_below_the_floor_and_never_lifts(self):
        self.assertEqual(mem_guard_width(8192, 100.0), P_MEM_GUARD_MIN_WIDTH)
        self.assertEqual(mem_guard_width(4096, 100000.0), 4096)

    def test_rate_zero_is_a_noop(self):
        self.assertEqual(mem_guard_width(16384, 100.0, rate=0.0), 16384)


if __name__ == "__main__":
    unittest.main()
