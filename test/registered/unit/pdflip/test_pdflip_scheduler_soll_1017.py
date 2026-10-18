"""SCHEDULER-SOLL v2 (/spinning/gpu-arb/SCHEDULER-SOLL.md, GILT seit 17.10. 06:00Z): ein Test je Regel R1-R9 gegen die
verbindliche Schnittstelle §4 (``phase_policy.set_verdict`` / ``classify_arrival``, reine Funktionen, keine Uhr).

Nutzer 16.10.: "SOLL-Funktionalitaet darf nie wieder verloren gehen". Diese Datei ist das Netz dazu: sie bindet die Regeln an
die Aufrufschnittstelle, nicht an Hilfsfunktionen. Auf einem Stand OHNE ``set_verdict`` ist sie ROT (gewollt, AttributeError);
mit dem Kern (S1) wird sie gruen. Aenderung nur durch den Nutzer mit woertlichem, datiertem Zitat -- ein Test wird nie
angepasst, damit er zum Code passt, sondern der Code zum SOLL.

Zahlen: X = 6000, halbX = 3000 (half_frac 0,5), T_halb = 4,5 s, wie SOLL §5.
R10 (Flag + Instrument) und R11 (Front entscheidet allein) sind keine Eigenschaft der reinen Policy-Funktion und werden
in den Front-/Rig-Tests (S1/S2) gebunden.
"""
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import phase_policy as pp
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-pdflip-unit")

X, HALF, T_HALF = 6000, 0.5, 4.5


def M(rid, u, t=0.0, p_only=False):
    return pp.SetMember(rid=rid, uncached=u, t_arrive=t, p_only=p_only)


def V(members, *, now=1.0, t_open=0.0, d_decoding=True, seats_free=8, running=0, new_since_last=0):
    return pp.set_verdict(
        members, x_tokens=X, half_frac=HALF, t_half_s=T_HALF, now=now, t_open=t_open, d_decoding=d_decoding,
        seats_free=seats_free, d_prefill_running_uncached=running, new_since_last=new_since_last,
    )


class SollR1toR9(unittest.TestCase):
    def test_r1_set_sum_decides_never_a_single_request(self):
        """R1: Flip oder nicht faellt ueber S gegen X. Ein halber Request allein flippt nie, zwei halbe zusammen ja."""
        one = V([M("a", 3500)], now=2.0)
        two = V([M("a", 3500), M("b", 3200, 1.0)], now=2.0)
        self.assertEqual(one.s_tokens, 3500)
        self.assertNotEqual(one.verdict, "FLIP_P")
        self.assertEqual(two.s_tokens, 6700)
        self.assertEqual(two.verdict, "FLIP_P")
        self.assertEqual(two.half_x, 3000)

    def test_r2_short_goes_to_d_at_once(self):
        """R2: u < halbX = kurz, sofort D, nie Teil eines Sets (SOLL §5 Bsp. 1 und 4)."""
        self.assertEqual(pp.classify_arrival(300, X, HALF, False), "kurz")
        self.assertEqual(pp.classify_arrival(2999, X, HALF, False), "kurz")
        self.assertEqual(pp.classify_arrival(3000, X, HALF, False), "halb")
        # Bsp. 4: 3500, dann 300 (kurz, nicht im Set), dann 2800 -> S = 6300, nicht 6600
        v = V([M("a", 3500), M("c", 2800, 2.0)], now=2.5)
        self.assertEqual(v.s_tokens, 6300)
        self.assertEqual(v.verdict, "FLIP_P")

    def test_r3_half_collects_until_t_half(self):
        """R3: halb sammelt bis T_halb; S >= X innerhalb -> Flip; Ablauf mit S < X -> D, kein Flip (Bsp. 2 und 3)."""
        early = V([M("a", 3500)], now=2.0, t_open=0.0)
        self.assertEqual(early.verdict, "COLLECT")
        self.assertAlmostEqual(early.left_s, T_HALF - 2.0, places=3)
        late = V([M("a", 3500)], now=4.6, t_open=0.0)
        self.assertEqual(late.verdict, "D_AFTER_TIMER")
        self.assertEqual(V([M("a", 3500), M("b", 3200, 2.0)], now=2.0).verdict, "FLIP_P")

    def test_r4_large_waits_for_company_when_d_decodes(self):
        """R4: u >= X waehrend D decodet: T_halb auf Mitfahrer; Mitfahrer oder Ablauf -> Flip; ohne Decode sofort (Bsp. 5)."""
        self.assertEqual(V([M("g", 9000)], now=1.0).verdict, "COLLECT")
        self.assertEqual(V([M("g", 9000), M("b", 3100, 1.0)], now=1.0).verdict, "FLIP_P")
        self.assertEqual(V([M("g", 9000)], now=4.6).verdict, "FLIP_P")
        self.assertEqual(V([M("g", 9000)], now=0.0, d_decoding=False).verdict, "FLIP_P")

    def test_r5_no_seat_means_prefill_and_park(self):
        """R5: Flip nur mit Decode-Sitzen fuer das Set; ohne Sitz prefillen und parken (Bsp. 8)."""
        v = V([M("a", 3500), M("b", 3200, 1.0)], now=2.0, seats_free=0)
        self.assertEqual(v.verdict, "PARK_NO_SEAT")

    def test_r6_running_d_prefill_is_rerouted(self):
        """R6: wachsendes S inkl. Rest des laufenden D-Prefills ueber X -> Umleitung nach P (Bsp. 7)."""
        v = V([M("n", 5000, 3.0)], now=3.0, running=4000, new_since_last=1)
        self.assertEqual(v.s_tokens, 9000)
        self.assertEqual(v.verdict, "REROUTE_P")
        self.assertNotEqual(V([M("n", 5000, 3.0)], now=3.0, running=0).verdict, "REROUTE_P")

    def test_r6_only_a_really_running_d_prefill_counts(self):
        """R6 (Nutzerentscheid 18.10. ~21:15Z, woertlich "Nur echt laufend zählt (Recommended)"): R6 zaehlt nur einen
        D-Prefill, der schon rechnet; vergeben-aber-nicht-gestartet zaehlt nicht, ein grosser Request wartet seine
        T_halb (R4) zu Ende. Belegfall rc13 20:33:48 (S46): p_only 8793 bei X=7543 waited 1,4 s, kurz 1426 vergeben,
        D hatte den Extend noch nicht begonnen -> kein REROUTE_P, COLLECT; erst mit D's Extend-Beleg -> REROUTE_P."""
        import time as _time
        from unittest import mock
        from test_pdflip_arrival_seat_rule_0929 import _front, _pending
        from flliper.srt.pdflip import d_extend_beacon as dxb

        env = {"FLLIPER_PDFLIP_HALF_FLIP_S": str(T_HALF), "FLLIPER_PDFLIP_HALF_X_FRAC": str(HALF),
               "FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_RULE": "1"}
        with mock.patch.dict(os.environ, env):
            self._r6_only_really_running(_time, _front, _pending, dxb)

    def _r6_only_really_running(self, _time, _front, _pending, dxb):
        f = _front(running=["dec"], n=6, kv={"available": 400000, "evictable": 0})   # D decodes (R4 waits)
        f.tp_prefill_max_tokens = 7543
        now = _time.time()
        f.t_awake = now - 60.0
        g = _pending("g", 8793, now - 1.4)
        g.p_only = True
        f.queue = [g]
        f._d_pf_book().grant(rid="k", tokens=1426, now=now - 0.136)
        f._d_pf_book().enter_leg2(rid="k")
        f.groups["D"].outstanding["k"] = now - 0.1    # D holds the POST, its extend has not begun
        f._d_ext_reading = lambda: {}                 # D's beacon: no extend of "k" yet
        self.assertEqual(f._dc_carried_running(), 0)
        self.assertEqual(f._d_carried_tokens(), 1426, "the X-SUM price of the kurz path is unchanged")
        self.assertEqual(f._decode_collect(now), "hold", "R4: the big member waits its T_halb")
        self.assertEqual(f.counters["set_reroute_p"], 0)
        f._d_ext_reading = lambda: {dxb.rid_key("k"): _time.time_ns()}   # D began the extend of "k"
        self.assertEqual(f._dc_carried_running(), 1426)
        self.assertEqual(f._decode_collect(_time.time()), "P")
        self.assertEqual(f.counters["set_reroute_p"], 1)

    def test_r7_never_a_flip_below_x(self):
        """R7: kein Flip, kein Reroute fuer S < X -- in keinem Zustand (Decode ja/nein, Sitze, Zeit)."""
        for members in ([M("a", 5900)], [M("a", 3200), M("b", 2700, 1.0)]):
            for decoding in (True, False):
                for seats in (0, 8):
                    for now in (0.5, 4.6, 20.0):
                        v = V(members, now=now, d_decoding=decoding, seats_free=seats)
                        self.assertLess(v.s_tokens, X)
                        self.assertNotIn(v.verdict, ("FLIP_P", "REROUTE_P"), (decoding, seats, now))

    def test_r8_timers_are_soft_without_decode(self):
        """R8: ohne Decode auf D sofort entscheiden: D bei S < X, P bei S >= X (Bsp. 6)."""
        small = V([M("a", 3500)], now=0.0, t_open=0.0, d_decoding=False)
        self.assertEqual(small.verdict, "D_AFTER_TIMER")
        self.assertEqual(small.left_s, 0.0)
        self.assertEqual(V([M("g", 9000)], now=0.0, d_decoding=False).verdict, "FLIP_P")

    def test_r9_p_only_counts_as_large(self):
        """R9: p_only (Vision) zaehlt als gross -- auch mit 100 Tokens."""
        self.assertEqual(pp.classify_arrival(100, X, HALF, True), "p_only")
        self.assertEqual(V([M("v", 100, p_only=True)], now=1.0).verdict, "COLLECT")
        self.assertEqual(V([M("v", 100, p_only=True)], now=0.0, d_decoding=False).verdict, "FLIP_P")


if __name__ == "__main__":
    unittest.main()
