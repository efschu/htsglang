"""27B INT8 night 22.10., D TP1 (RTX 3080): the driver-free minimum (#1028c BOUND driver_free) fell to 359.8 MiB (round 1,
trim 1200,700,700, 22:48:13Z) and 371.8 MiB (round 2, trim 1200,1100,800, 23:43:56Z) right after D-direct extends whose
chunk the width vote (PDFLIP-EXTEND-STUECKELUNG) had funded against a 1024 MiB line.

Root (extend_trim.width_vote): the vote funded the chunk with driver free PLUS torch's reserved-but-unallocated cache
(rc29, 121e4bf3f2). The rate it divides by is the measured RESERVED growth per row (EXTEND-RATE, read after the trim):
for this process every MiB of reserved growth is a MiB of driver free lost, and any cache reuse is already netted out
of it. Counting the cache again is a double count. The metal says the cache funded nothing: at 23:43:54Z the vote read
driver 882 + cache 860 MiB, funded 2263 rows; the 2162-row extend's live transient was 282 MiB, reserved grew 534 MiB,
driver 882 -> 372. Across all 80 TP1 extends >= 1024 rows of both rounds the driver drop stayed <= 0.3091 x rows (max
0.253 MiB/row) -- the start rate was never too small; the trim cannot help either: the ~850 MiB cache that survives every
trim (alloc_cache bound 853.7) is free blocks in segments that still hold live blocks.

Not the hypothesis "the forward's transients are not budgeted": they are (rate 0.3091 >= measured 0.2405 x 1.15 floor
and >= every measured driver drop per row); what was wrong is the funding side.
"""
import logging
import os
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-pdflip-unit")

from flliper.srt.pdflip import extend_trim as ET  # noqa: E402

MIB = 1 << 20
RATE = 0.3091            # FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB of both boots (rate_src=start)
MEASURED_DROP = 0.2533   # the largest driver drop per row of the 80 TP1 extends (round 1 + 2)
CONFIGURED = 4096
LAW = 700                # the corridor target named for the 3080 (the boots ran the 1024 default)


class _Cuda:
    def __init__(self, free, reserved, allocated):
        self.free, self.reserved, self.allocated = float(free), float(reserved), float(allocated)
        self.trims = 0

    def is_current_stream_capturing(self):
        return False

    def mem_get_info(self, *a):
        return int(self.free * MIB), int(20055 * MIB)

    def memory_reserved(self, *a):
        return int(self.reserved * MIB)

    def memory_allocated(self, *a):
        return int(self.allocated * MIB)

    def synchronize(self):
        pass

    def empty_cache(self):
        self.trims += 1


def _vote(cuda, law="1024", floor="1"):
    env = {"FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB": str(RATE), "FLLIPER_CORRIDOR_LAW_FLOOR_MIB": str(law)}
    if floor:
        env["FLLIPER_PDFLIP_EXTEND_CAP_FLOOR"] = floor
    with mock.patch.dict(os.environ, env, clear=False):
        for k in ("FLLIPER_PDFLIP_TORCH_CACHE_CAP", "FLLIPER_PDFLIP_EXTEND_TRIM_MIB", "FLLIPER_PDFLIP_EXTEND_RATE_MEASURE"):
            os.environ.pop(k, None)
        if not floor:
            os.environ.pop("FLLIPER_PDFLIP_EXTEND_CAP_FLOOR", None)
        ET.reset_for_tests()
        try:
            # NACHT2 22.10.: the line is the card's derived corridor floor (guard door, extend_trim._read_corridor_line);
            # here the card's floor is the named law (metal round 3c: TP1 graded 700, MEASURED-D)
            with mock.patch.object(ET, "_read_corridor_line", lambda: (float(law), "test:law")):
                return ET.width_vote(cuda, 1, CONFIGURED, 1, True)
        finally:
            ET.reset_for_tests()


def _r2_low_state():
    """Round 2, 23:43:54Z, D TP1 right after the trim (card_free_after=882) before pdflip-8-125 (2162 rows):
    reserved 18022 - 212 - 42 released = 17768 MiB, allocated 16908 -> cache 860 MiB (vote line: cache_free=860)."""
    return _Cuda(free=882, reserved=17768, allocated=16908)


class TheVoteFundsDriverFreeOnly(unittest.TestCase):
    def test_round2_low_state_keeps_the_corridor(self):
        # before: post = 882 + 860 = 1742 -> floor((1742 - 1024) / 0.3091) = 2322 rows -> 718 MiB growth -> 164 MiB
        # (metal 372). Driver alone under the 1024 line funds nothing; the Q-710 floor chunk (4096/16 = 256) is what D
        # runs: 256 x 0.3091 = 79 MiB priced (65 measured) -> 803 MiB.
        w = _vote(_r2_low_state())
        self.assertEqual(w, ET.floor_rows(CONFIGURED, 1))
        self.assertGreaterEqual(882 - w * RATE, LAW)
        self.assertGreaterEqual(882 - w * MEASURED_DROP, LAW)

    def test_round2_low_state_under_the_700_law_funds_to_the_law(self):
        # law 700: floor((882 - 700) / 0.3091) = 588 rows, the low at the priced rate is the law itself
        w = _vote(_r2_low_state(), law=LAW)
        self.assertEqual(w, 588)
        self.assertGreaterEqual(882 - w * RATE, LAW - 1)

    def test_the_cache_does_not_widen_the_chunk(self):
        # a card with ample driver free: the vote is the same with 0 or 1600 MiB idle in the cache
        a = _vote(_Cuda(free=1900, reserved=17000, allocated=17000), law=LAW)
        b = _vote(_Cuda(free=1900, reserved=18600, allocated=17000), law=LAW)
        self.assertEqual(a, b)
        self.assertEqual(a, 3882)      # floor((1900 - 700) / 0.3091)

    def test_every_round_metal_vote_would_have_kept_the_line(self):
        # (driver after trim, cache) of the four round-2 extends that ended lowest (330 / 336 / 338 / 340 MiB on the metal)
        for driver, cache in ((880, 871), (862, 919), (844, 933), (866, 917)):
            with self.subTest(driver=driver):
                w = _vote(_Cuda(free=driver, reserved=17800, allocated=17800 - cache), law=LAW)
                self.assertGreaterEqual(driver - w * RATE, LAW - 1)

    def test_line_names_the_funding(self):
        with self.assertLogs(ET.logger, level=logging.INFO) as cm:
            _vote(_r2_low_state(), law=LAW)
        line = next(m for m in cm.output if ET.CHUNKING_MARKER in m)
        self.assertIn("post=882 cache_free=860 funds=driver", line)

    def test_without_the_floor_the_vote_is_one_page(self):
        # FLLIPER_PDFLIP_EXTEND_CAP_FLOOR off: the line-respecting cap alone (one page), nothing guessed from the cache
        self.assertEqual(_vote(_r2_low_state(), floor=""), 1)


class TheRateWasNeverTheHole(unittest.TestCase):
    def test_start_rate_covers_the_measured_rate_with_its_safety(self):
        # EXTEND-RATE TP1 maximum of round 2 (23:43:22Z): 0.2405 -> x1.15 = 0.2766 < 0.3091: the start rate stays
        self.assertLess(ET._ceil4(0.2405 * ET.RATE_SAFETY), RATE)
        self.assertLess(MEASURED_DROP, RATE)


if __name__ == "__main__":
    unittest.main()
