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


def test_r6k_a_running_kurz_d_prefill_is_not_rerouted_and_not_in_s():
    """SOLL R6k (user 20.10. 21:30Z "b"): a running KURZ D prefill runs on D to its end -- its rest is not in S and
    triggers no flip; R6 stays for halb/gross. Same scene as the SOLL file's R6 front test (big member g waits its
    T_halb while D decodes; D began the extend of k and reports its real rest), once with k a halb (R6 redirects) and
    once with k classed kurz at its arrival (R6k: hold, nothing redirected)."""
    import time as _time
    from unittest import mock

    from test_pdflip_arrival_seat_rule_0929 import _front, _pending

    from flliper.srt.pdflip import d_extend_beacon as dxb

    env = {"FLLIPER_PDFLIP_HALF_FLIP_S": str(T_HALF), "FLLIPER_PDFLIP_HALF_X_FRAC": str(HALF),
           "FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_RULE": "1"}

    def scene(kurz: bool):
        f = _front(running=["dec"], n=6, kv={"available": 400000, "evictable": 0})
        f.tp_prefill_max_tokens = 7543
        now = _time.time()
        f.t_awake = now - 60.0
        g = _pending("g", 8793, now - 1.4)
        g.p_only = True
        f.queue = [g]
        f._d_pf_book().grant(rid="k", tokens=1426, now=now - 0.136)
        f._d_pf_book().enter_leg2(rid="k")
        f.groups["D"].outstanding["k"] = now - 0.1
        f._rvp_state()
        f.__dict__.setdefault("_d_prefill_legs", {})["k"] = now
        f._leg2_lookahead.add("k")
        f._d_ext_reading = lambda: {dxb.rid_key("k"): (_time.time_ns(), 1426)}
        if kurz:
            f._r6k_mark_kurz("k")
        return f

    with mock.patch.dict(os.environ, env):
        f = scene(kurz=False)
        assert f._dc_carried_running() == 1426
        assert f._decode_collect(_time.time()) == "P"            # R6: halb/gross running rest moves S over X
        assert f.counters["set_reroute_p"] == 1
        f = scene(kurz=True)
        assert f._dc_carried_running() == 0                      # R6k: the kurz rest is not in S
        assert f._decode_collect(_time.time()) == "hold"         # g waits its T_halb (R4), no flip by k
        assert f.counters["set_reroute_p"] == 0
        assert f.counters["r6k_kurz_stays_on_d"] >= 1


def _r2_scene(now, *, started=False):
    """Boot 1023_140032 14:21:46Z (27B AutoRound, X=4096, halbX=2048, T_halb=4,5 s): pdflip-20-118 (gross, u=23969) waits
    3,9 s in the set; on D only pdflip-20-119 (kurz, u=293, D_NOW 45,018), granted and not begun -- no decode on D."""
    from test_pdflip_arrival_seat_rule_0929 import _front, _pending

    f = _front(running=["k"], n=6, kv={"available": 400000, "evictable": 0})
    f.tp_prefill_max_tokens = 4096
    f.t_awake = now - 60.0
    g = _pending("g", 23969, now - 3.9)
    f.queue = [g]
    f._d_pf_book().grant(rid="k", tokens=293, now=now - 1.44)
    f._d_pf_book().enter_leg2(rid="k")
    f.groups["D"].outstanding["k"] = now - 1.4
    f._r6k_mark_kurz("k")
    f._d_ext_reading = (lambda: {}) if not started else None
    if started:
        import time as _t

        from flliper.srt.pdflip import d_extend_beacon as dxb

        f._d_ext_reading = lambda: {dxb.rid_key("k"): (_t.time_ns(), 293)}
    return f


def test_r2_a_d_granted_kurz_finishes_on_d_before_the_flip_then_r4_decides():
    """SOLL R2 (user 23.10. ~14:40Z: "unsere regeln sind eigentlich eindeutig" -- a kurz request is prefilled on D at once and
    without wait, and there is never a flip for it; R6k is not needed for the case) + R4/R8. Defect (boot 1023_140032
    14:21:46,456Z): R8 "no decode on D" decided FLIP_P while the D-granted kurz 20-119 had not begun; PARK-IMMEDIATE parked
    it and its decode stood 49 s. SOLL: the D-granted kurz prefills on D to its end before D quiesces; then it DECODES, and
    the big member waits the rest of its T_halb (R4)."""
    import time as _time
    from unittest import mock

    from flliper.srt.environ import envs

    env = {"FLLIPER_PDFLIP_HALF_FLIP_S": str(T_HALF), "FLLIPER_PDFLIP_HALF_X_FRAC": str(HALF),
           "FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_RULE": "1"}
    for decider in (True, False):
        with mock.patch.dict(os.environ, env), envs.FLLIPER_PDFLIP_DECIDER.override(decider):
            now = _time.time()
            # the defect, reproduced with the hold switched off: R8 flips over the granted kurz
            with envs.FLLIPER_PDFLIP_R2_KURZ_GRANT_HOLD.override(False):
                f = _r2_scene(now)
                assert not f._dc_decoding()                       # the R8 premise: a granted kurz is no decode
                assert f._decode_collect(now) == "P", decider
            for started in (False, True):                         # granted-not-begun (20-119) and running alike
                f = _r2_scene(now, started=started)
                assert f._decode_collect(now) == "hold", (decider, started)
                assert f.counters["r2_kurz_grant_hold"] == 1 and f.counters["set_flip_p"] == 0
                # the kurz prefill ended on D (first content): it DECODES now -> R4, the big member waits its T_halb
                f._d_pf_book().done(rid="k")
                assert f._dc_decoding()
                assert f._decode_collect(now + 0.1) == "hold"     # waited 4.0 s < 4.5 s: R4 COLLECT
                assert f.counters["r2_kurz_grant_hold_done"] == 0  # no flip verdict was held this time: R4 itself waits
                assert f._decode_collect(now + 0.7) == "P"        # T_halb over (4.6 s): R4 flips
                assert f.counters["set_flip_p"] == 1


def test_r2_a_parked_kurz_does_not_hold_the_flip():
    """The hold covers only kurz requests D is prefilling now: a kurz D parked (or holds for RESUME-VIA-P) is not
    prefilling -- R8 decides as before."""
    import time as _time
    from unittest import mock

    from flliper.srt.environ import envs

    env = {"FLLIPER_PDFLIP_HALF_FLIP_S": str(T_HALF), "FLLIPER_PDFLIP_HALF_X_FRAC": str(HALF),
           "FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_RULE": "1"}
    with mock.patch.dict(os.environ, env), envs.FLLIPER_PDFLIP_DECIDER.override(True):
        now = _time.time()
        f = _r2_scene(now)
        f._d_parked["k"] = now - 0.5
        assert f._r2_kurz_grants_open(now) == []
        assert f._decode_collect(now) == "P"
