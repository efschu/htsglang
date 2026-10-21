"""27B INT8 metal 21.10. (image rc29 on 041a871f4b, boot dkr27bbar1drei10210018, hang from 00:26:49Z, no death): six
parked requests held all six D seats, D refused each of them again every 30 s, P slept (no flip under X, SOLL R7).

Two roots, both D refusals of requests under X (R7: a defect, never a flip reason), reproduced here at the functions
the metal ran through:

(a) the kurz ones (pdflip-4-69..72, d_extent 234..797): D TP1 (3080) voted a chunk width of 63 -- the width vote counted
    the DRIVER's free bytes only (19.8 MiB) while torch held 1.6 GiB reserved-but-unallocated (allocated 17043 / reserved
    18650 MiB); EXTEND-CAP-FLOOR then refused every fresh D extend > 63 rows (Q-710) and W50 sent it to P.
(b) the halb ones (pdflip-4-40/42, u 5815/5953, front X 10000): sent to D unpriced with the MEASURE barrier (front_x 4998,
    front_priced 0); D refused them there (the measure), kept the streams and re-offered them every 30 s -- each time
    against the same barrier (limit 4999), and its refusal answer carried X=4999, so RESUME-VIA-P classed them gross.
"""
import os
import types
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-pdflip-unit")

from flliper.srt.pdflip import extend_trim as ET  # noqa: E402
from flliper.srt.pdflip import front_x as FX  # noqa: E402

MIB = 1 << 20
RATE = 0.3091
CONFIGURED = 4096


class _Cuda:
    """D TP1 at 00:26:45Z: driver free 19.8 MiB of 20480, torch allocated 17043 / reserved 18650 MiB."""

    def __init__(self, free=19.8, reserved=18650, allocated=17043):
        self.free, self.reserved, self.allocated = float(free), float(reserved), float(allocated)

    def is_current_stream_capturing(self):
        return False

    def mem_get_info(self, *a):
        return int(self.free * MIB), int(20480 * MIB)

    def memory_reserved(self, *a):
        return int(self.reserved * MIB)

    def memory_allocated(self, *a):
        return int(self.allocated * MIB)

    def synchronize(self):
        pass

    def empty_cache(self):
        pass


def _vote(cuda):
    env = {"FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB": str(RATE), "FLLIPER_PDFLIP_EXTEND_CAP_FLOOR": "1",
           "FLLIPER_CORRIDOR_LAW_FLOOR_MIB": "700"}
    with mock.patch.dict(os.environ, env, clear=False):
        os.environ.pop("FLLIPER_PDFLIP_TORCH_CACHE_CAP", None)
        ET.reset_for_tests()
        try:
            return ET.width_vote(cuda, 1, CONFIGURED, 1, True)
        finally:
            ET.reset_for_tests()


class WidthVoteCountsTheTorchCache(unittest.TestCase):
    def test_metal_tp1_with_1_6_gib_idle_cache_votes_a_wide_chunk_inside_the_law(self):
        # driver 19.8 + cache 1607 MiB fund floor((1626.8 - 700) / 0.3091) = 2998 rows: far above the Q-710 floor chunk
        # (256), so no refusal -- and the chunk's growth leaves the 700 MiB corridor law free (before: width 63, refused)
        w = _vote(_Cuda())
        self.assertEqual(w, 2998)
        self.assertGreater(w, ET.floor_rows(CONFIGURED, 1))
        self.assertGreaterEqual(19.8 + 1607 - w * RATE, 700 - 1)

    def test_without_idle_cache_the_card_still_narrows(self):
        # the danger direction stays: a card with nothing idle in the cache still votes narrow (Q-710 floor)
        w = _vote(_Cuda(free=19.8, reserved=18650, allocated=18650))
        self.assertIsNotNone(w)
        self.assertLess(w, ET.floor_rows(CONFIGURED, 1) + 1)

    def test_partial_cache_funds_only_what_it_holds(self):
        # 1000 MiB idle: floor((1019.8 - 700) / 0.3091) = 1034 rows -- the vote never funds more than the bytes that exist
        w = _vote(_Cuda(free=19.8, reserved=18650, allocated=17650))
        self.assertEqual(w, 1034)


class WidthVoteKeepsTheCorridorLaw(unittest.TestCase):
    """27B INT8 rc29b (and 0798c91fcd alike): D TP1 breached the 700 MiB law 17x (minimum 3-7 MiB) -- the vote funded the
    4096-row chunk down to the private 300 MiB halt line, the chunk's growth (0.3091 MiB/row) ate the rest. The vote now
    keeps the LAW in force (FLLIPER_CORRIDOR_LAW_FLOOR_MIB / corridor_guard.corridor_law_mib)."""

    def test_the_metal_after_chunk_state_narrows_to_what_keeps_the_law(self):
        # TP1 after a chunk (rc29b VRAM-PEAK minimum): driver + cache = 1406 MiB -> floor((1406 - 700) / 0.3091) = 2284
        w = _vote(_Cuda(free=600, reserved=18286, allocated=17480))
        self.assertEqual(w, 2284)
        self.assertGreaterEqual(1406 - w * RATE, 700 - 1)

    def test_the_line_is_the_law_in_force(self):
        with mock.patch.dict(os.environ, {"FLLIPER_CORRIDOR_LAW_FLOOR_MIB": "900"}):
            self.assertEqual(ET.corridor_line_mib(), 900.0)
        with mock.patch.dict(os.environ, {"FLLIPER_CORRIDOR_LAW_FLOOR_MIB": "100"}):
            self.assertEqual(ET.corridor_line_mib(), float(ET.CHUNKING_FLOOR_MIB))   # never under the halt line


def _req(output_ids=()):
    r = types.SimpleNamespace(rid="pdflip-4-40", output_ids=list(output_ids))
    setattr(r, FX.ATTR_X, 4998)         # the MEASURE barrier: halbX - 1 - one page
    setattr(r, FX.ATTR_PRICED, 0)
    setattr(r, FX.ATTR_HALFX, 5000)
    return r


class MeasureBarrierIsSpentByItsRefusal(unittest.TestCase):
    def setUp(self):
        self._p = mock.patch.object(FX, "enabled", lambda: True)
        self._p.start()

    def tearDown(self):
        self._p.stop()

    def test_first_offer_meets_the_barrier(self):
        lim, fx, pr = FX.resolve_limit(_req(), 12288, 1)
        self.assertEqual((lim, fx, pr), (4999, 4998, 0))
        self.assertTrue(FX.is_measure_barrier(fx, pr))
        self.assertGreater(5815, lim)    # the halb request is refused: the measure is taken

    def test_the_re_offer_is_bounded_by_the_fronts_x(self):
        r = _req()
        setattr(r, FX.ATTR_MEASURE_SPENT, True)          # set by D's X gate with the barrier's own refusal
        with mock.patch.dict(os.environ, {"FLLIPER_PDFLIP_HALF_X_FRAC": "0.5"}):
            lim, fx, pr = FX.resolve_limit(r, 12288, 1)
        self.assertEqual(lim, 10001)                      # front X 10000 + one page, under the static 12288
        self.assertLessEqual(5815, lim)                   # pdflip-4-40 is admitted on D (SOLL R7: under X is D's)
        self.assertLessEqual(5953, lim)                   # pdflip-4-42 too

    def test_a_priced_front_x_is_never_touched(self):
        r = _req()
        setattr(r, FX.ATTR_PRICED, 7000)
        setattr(r, FX.ATTR_MEASURE_SPENT, True)
        self.assertFalse(FX.is_measure_barrier(4998, 7000))
        self.assertEqual(FX.resolve_limit(r, 12288, 1)[0], 4999)

    def test_the_gate_marks_the_barrier_spent_on_every_rank(self):
        src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "python", "flliper",
                                "srt", "managers", "scheduler.py"), encoding="utf-8").read()
        i = src.index('verdict = "W31" if uncached > limit else "admit"')
        self.assertIn("setattr(req, _pdflip_front_x.ATTR_MEASURE_SPENT, True)", src[i:i + 400])


if __name__ == "__main__":
    unittest.main()
