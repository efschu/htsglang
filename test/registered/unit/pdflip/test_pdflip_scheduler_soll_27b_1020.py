"""SCHEDULER-SOLL v2 on the 27B line (port 20.10.): the rules R1-R9 of ``test_pdflip_scheduler_soll_1017`` (whose numbers are
SOLL §5's X = 6000) bound once more at the 27B's OWN X -- read from the 27B INT8 curve file the image carries
(docker/flliper/profiles_release/27b.xcurves.json, measured on the rig 06.10., boots dkr27browauthoritybar1fs10061132 +
...10061534), at four cache depths. Same interface (§4: ``phase_policy.set_verdict`` / ``classify_arrival``), no clock.

The SOLL file is not touched (only the user changes it); this file only re-states its rules at 27B numbers. Every member
size is a fraction of the depth's X, so each rule is tested on the same side of X / halbX as in §5.
"""
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import phase_policy as pp
from flliper.srt.pdflip import x_curves as xc
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-pdflip-unit")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
CURVES_27B = os.path.join(_ROOT, "docker", "flliper", "profiles_release", "27b.xcurves.json")
HALF, T_HALF = 0.5, 4.5
DEPTHS = (0, 32768, 65536, 131072)


def _x_at(depth: int) -> int:
    return int(xc.x_from_curves(depth=depth, open_requests=1, curves=xc.load(CURVES_27B)))


def M(rid, u, t=0.0, p_only=False):
    return pp.SetMember(rid=rid, uncached=int(u), t_arrive=t, p_only=p_only)


def V(x, members, *, now=1.0, t_open=0.0, d_decoding=True, seats_free=8, running=0, new_since_last=0):
    return pp.set_verdict(
        members, x_tokens=x, half_frac=HALF, t_half_s=T_HALF, now=now, t_open=t_open, d_decoding=d_decoding,
        seats_free=seats_free, d_prefill_running_uncached=running, new_since_last=new_since_last,
    )


class Soll27bCurves(unittest.TestCase):
    def test_the_27b_curve_file_is_the_27b_int8_form(self):
        c = xc.load(CURVES_27B)
        self.assertTrue(c.identity.model.startswith("Qwen3.8-27B-INT8"), c.identity)
        self.assertEqual(c.identity.hardware, "RTX5090,RTX3080,RTX3080")
        xs = [_x_at(d) for d in DEPTHS]
        # measured, not the NF numbers: X falls with depth on the 27B curves
        self.assertEqual(xs, sorted(xs, reverse=True), xs)
        self.assertTrue(all(4096 <= x <= 12288 for x in xs), xs)   # inside start-X .. --x-ceiling-tokens 12288

    def _each_x(self):
        for d in DEPTHS:
            x = _x_at(d)
            with self.subTest(depth=d, x=x):
                yield x

    def test_r1_set_sum_decides(self):
        for x in self._each_x():
            one = V(x, [M("a", 0.58 * x)], now=2.0)
            two = V(x, [M("a", 0.58 * x), M("b", 0.54 * x, 1.0)], now=2.0)
            self.assertNotEqual(one.verdict, "FLIP_P")
            self.assertEqual(two.verdict, "FLIP_P")
            self.assertEqual(two.half_x, int(HALF * x))

    def test_r2_short_is_kurz_never_in_a_set(self):
        for x in self._each_x():
            hx = int(HALF * x)
            self.assertEqual(pp.classify_arrival(hx - 1, x, HALF, False), "kurz")
            self.assertEqual(pp.classify_arrival(hx, x, HALF, False), "halb")
            self.assertEqual(pp.classify_arrival(x, x, HALF, False), "gross")

    def test_r3_half_collects_until_t_half(self):
        for x in self._each_x():
            self.assertEqual(V(x, [M("a", 0.58 * x)], now=2.0).verdict, "COLLECT")
            self.assertEqual(V(x, [M("a", 0.58 * x)], now=4.6).verdict, "D_AFTER_TIMER")

    def test_r4_large_waits_for_company_when_d_decodes(self):
        for x in self._each_x():
            self.assertEqual(V(x, [M("g", 1.5 * x)], now=1.0).verdict, "COLLECT")
            self.assertEqual(V(x, [M("g", 1.5 * x), M("b", 0.52 * x, 1.0)], now=1.0).verdict, "FLIP_P")
            self.assertEqual(V(x, [M("g", 1.5 * x)], now=4.6).verdict, "FLIP_P")
            self.assertEqual(V(x, [M("g", 1.5 * x)], now=0.0, d_decoding=False).verdict, "FLIP_P")

    def test_r5_no_seat_prefill_and_park(self):
        for x in self._each_x():
            v = V(x, [M("a", 0.58 * x), M("b", 0.54 * x, 1.0)], now=2.0, seats_free=0)
            self.assertEqual(v.verdict, "PARK_NO_SEAT")

    def test_r6_running_d_prefill_rerouted(self):
        for x in self._each_x():
            v = V(x, [M("n", 0.8 * x, 3.0)], now=3.0, running=int(0.7 * x), new_since_last=1)
            self.assertEqual(v.verdict, "REROUTE_P")
            self.assertNotEqual(V(x, [M("n", 0.8 * x, 3.0)], now=3.0, running=0).verdict, "REROUTE_P")

    def test_r7_never_a_flip_below_x(self):
        for x in self._each_x():
            for members in ([M("a", x - 1)], [M("a", 0.53 * x), M("b", 0.46 * x, 1.0)]):
                for decoding in (True, False):
                    for seats in (0, 8):
                        for now in (0.5, 4.6, 20.0):
                            v = V(x, members, now=now, d_decoding=decoding, seats_free=seats)
                            self.assertLess(v.s_tokens, x)
                            self.assertNotIn(v.verdict, ("FLIP_P", "REROUTE_P"))

    def test_r8_soft_timers_without_decode(self):
        for x in self._each_x():
            small = V(x, [M("a", 0.58 * x)], now=0.0, d_decoding=False)
            self.assertEqual(small.verdict, "D_AFTER_TIMER")
            big = V(x, [M("a", 0.58 * x), M("b", 0.54 * x)], now=0.0, d_decoding=False)
            self.assertEqual(big.verdict, "FLIP_P")

    def test_r9_p_only_counts_as_large(self):
        for x in self._each_x():
            self.assertEqual(pp.classify_arrival(100, x, HALF, True), "p_only")
            self.assertEqual(V(x, [M("v", 100, p_only=True)], now=1.0).verdict, "COLLECT")
            self.assertEqual(V(x, [M("v", 100, p_only=True)], now=0.0, d_decoding=False).verdict, "FLIP_P")


if __name__ == "__main__":
    unittest.main()
