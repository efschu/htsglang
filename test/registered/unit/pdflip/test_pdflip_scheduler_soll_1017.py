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
        # D began the extend of "k" and reports its REAL rest (S56); the leg can move to P (lookahead open)
        f._rvp_state()
        f.__dict__.setdefault("_d_prefill_legs", {})["k"] = now
        f._leg2_lookahead.add("k")
        f._d_ext_reading = lambda: {dxb.rid_key("k"): (_time.time_ns(), 1426)}
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


def test_r6_s56_the_running_rest_is_ds_real_number_and_only_if_it_moves(monkeypatch):
    """R6 (S56, NF rc15 19.10., user "um 9:20:26 wird nur EIN request prefillt und zwar 4492 neue token"): the rest of
    the running D prefill in S is D's REAL rest (D-EXTEND-BEACON v2), not the front's arrival price, and it counts only
    when the REROUTE really moves it to P. rc15: 4492 + D's 977 = 5469 < X=6216 -> no flip (R7); with a real big
    rest (3000) R6 still reroutes and the prefill is handed to P."""
    import time as _time

    from test_pdflip_s56_r6_phantom_dprefill_1019 import _fx, _grant_611, _halb_612
    from test_pdflip_set_scheduler_1017 import _on

    _on(monkeypatch)
    now = _time.time()
    f = _fx(now)
    _grant_611(f, now, rest=977)
    _halb_612(f, now)
    assert f._decode_collect(now) == "hold" and f.counters["set_reroute_p"] == 0
    f = _fx(now)
    _grant_611(f, now, rest=3000)
    _halb_612(f, now)
    assert f._decode_collect(now) == "P" and f.counters["set_reroute_p"] == 1 and f.counters["r6_redirected"] == 1
    f = _fx(now)
    _grant_611(f, now, rest=3000, redirectable=False)
    _halb_612(f, now)
    assert f._decode_collect(now) == "hold" and f.counters["set_reroute_p"] == 0


def test_r9_s56_an_image_in_the_uncached_tail_is_p_only():
    """R9 (S56): D's probe finds an image past the covered prefix (D has no tower, W123) -> P-only, at P's number."""
    import numpy as _np

    from flliper.srt.pdflip import d_match_probe as _dmp

    a = _dmp.ProbeAnswer(ids=_np.arange(85590, dtype=_np.int32), device_tokens=83712, host_tokens=0,
                         key_match_depth=83712, mm_spans=((84000, 84882),))
    p = _dmp.exact_price(a, 79872)
    assert p.image_in_tail and _dmp.class_uncached(p) == 5718
    assert pp.classify_arrival(_dmp.class_uncached(p), X, HALF, p.image_in_tail) == pp.CLASS_P_ONLY


# ---------------------------------------------------------------------------------------------------------------- R6k (20.10.)
# SOLL R6k (Nutzerentscheid 20.10., Eintrag 21:30:15Z, woertlich "b"): ein laufender KURZER D-Prefill (bei Ankunft u < halbX, R2)
# wird NICHT nach P umgeleitet, rechnet auf D zu Ende, sein Rest zaehlt NICHT in S, loest keinen Flip aus und zieht keinen vor.
# R6 gilt nur fuer laufende Halb- und Gross-Prefills. R1a bleibt (wartende kurze fahren mit). Front-Kante, live-Zahlen rc28
# pdflip-12-274 (gross, u=9875) / pdflip-12-275 (kurz, u=1394), X=7016, halbX=3508, T_halb=4,5 s.
def _r6k_front(monkeypatch, tmp_path, *, member_u, klass, waited=3.9, ended=True, rest_start=1394, ext_len=768):
    import time as _time

    from test_nf_r6_rest_exakt_1020 import _front_live
    from test_pdflip_set_scheduler_1017 import _on

    _on(monkeypatch)
    now = _time.time()
    return now, _front_live(now, tmp_path, member_u=member_u, ended=ended, rest_start=rest_start, ext_len=ext_len,
                            klass=klass, waited=waited)


def _r6k_lines(caplog, token):
    return [m for m in caplog.messages if token in m]


def test_r6k_kurz_running_and_a_big_set_flip_without_the_kurz_it_stays_on_d(monkeypatch, caplog, tmp_path):
    """R6k + R4: kurzer laufender (12-275) + grosses Set. Vor T_halb: kein Flip (der kurze zieht keinen vor). Nach T_halb: Flip
    nach P OHNE den kurzen -- S = 9875 (nur das Mitglied), verdict kein REROUTE_P, kein R6-REDIRECT / PARK-HANDBACK des kurzen."""
    import logging as _lg

    from flliper.srt.environ import envs

    now, f = _r6k_front(monkeypatch, tmp_path, member_u=9875, klass="kurz", waited=3.9)
    with envs.FLLIPER_PDFLIP_DECIDER.override(True), caplog.at_level(_lg.INFO, logger="pdflip.front"):
        assert f._dc_carried_running() == 0
        assert f._decode_collect(now) == "hold"                       # R4 waits; the kurz rest pulls nothing forward
    caplog.clear()
    tp = tmp_path / "late"
    tp.mkdir()
    now, f = _r6k_front(monkeypatch, tp, member_u=9875, klass="kurz", waited=4.6)
    with envs.FLLIPER_PDFLIP_DECIDER.override(True), caplog.at_level(_lg.INFO, logger="pdflip.front"):
        assert f._decode_collect(now) == "P"                          # T_halb over: the big set flips
    sv = _r6k_lines(caplog, "PDFLIP SET-VERDICT")
    assert sv and all("verdict=REROUTE_P" not in m for m in sv), sv
    assert any("S=9875 " in m for m in sv), sv                        # the kurz rest is not in S
    assert not _r6k_lines(caplog, "R6-REDIRECT") and not _r6k_lines(caplog, "PARK-HANDBACK")
    assert f.counters["r6_redirected"] == 0 and "pdflip-12-275" not in f._rvp_requeue


def test_r6k_kurz_running_and_a_halb_set_under_x_no_flip(monkeypatch, caplog, tmp_path):
    """R6k + R3/R7: kurzer laufender (Rest 3400, Forward laeuft noch) + halbes Set u=4000: S = 4000 < X -> kein Flip. (Vor R6k:
    S = 7400 > X -> REROUTE_P.)"""
    import logging as _lg

    from flliper.srt.environ import envs

    now, f = _r6k_front(monkeypatch, tmp_path, member_u=4000, klass="kurz", ended=False, rest_start=3400, ext_len=2048)
    with envs.FLLIPER_PDFLIP_DECIDER.override(True), caplog.at_level(_lg.INFO, logger="pdflip.front"):
        assert f._dc_carried_running() == 0
        assert f._decode_collect(now) == "hold"
    assert all("verdict=REROUTE_P" not in m for m in _r6k_lines(caplog, "PDFLIP SET-VERDICT"))
    assert f.counters["set_reroute_p"] == 0 and f.counters["r6_redirected"] == 0


def test_r6k_halb_running_r6_as_before(monkeypatch, caplog, tmp_path):
    """R6 bleibt fuer laufende Halb-Prefills: halber laufender (Rest 6000, Chunk 2048 im Forward) + halbes Set u=3600: S = 3600 +
    3952 = 7552 > X -> REROUTE_P why=R6, der halbe wird nach P umgeleitet (R6-REDIRECT). NF-RC30-R6STALE (R6 + R7): S zaehlt
    nur, was nach P geht -- den Teil, fuer den D noch keinen Forward gestartet hat (der laufende Chunk endet auf D)."""
    import logging as _lg

    from flliper.srt.environ import envs

    now, f = _r6k_front(monkeypatch, tmp_path, member_u=3600, klass="halb", ended=False, rest_start=6000, ext_len=2048)
    with envs.FLLIPER_PDFLIP_DECIDER.override(True), caplog.at_level(_lg.INFO, logger="pdflip.front"):
        assert f._dc_carried_running() == 3952
        assert f._decode_collect(now) == "P"
    assert any("verdict=REROUTE_P" in m and "why=R6" in m for m in _r6k_lines(caplog, "PDFLIP SET-VERDICT"))
    assert f.counters["r6_redirected"] == 1


def test_r6k_halb_and_kurz_running_r6_moves_only_the_halb(monkeypatch, caplog, tmp_path):
    """R6k + R6 zusammen: ein halber UND ein kurzer laufen. S zaehlt nur den halben Rest; REROUTE_P leitet nur den halben um,
    der kurze rechnet auf D zu Ende."""
    import logging as _lg

    from test_nf_r6_rest_exakt_1020 import add_running
    from flliper.srt.environ import envs

    now, f = _r6k_front(monkeypatch, tmp_path, member_u=3600, klass="halb", ended=False, rest_start=6000, ext_len=2048)
    add_running(f, now, tmp_path, "pdflip-12-276", rest_start=900, ext_len=900, ended=False, klass="kurz", key=37870)
    with envs.FLLIPER_PDFLIP_DECIDER.override(True), caplog.at_level(_lg.INFO, logger="pdflip.front"):
        assert f._dc_carried_running() == 3952      # NF-RC30-R6STALE: the halb's unlaunched 6000-2048, the kurz not at all
        assert f._decode_collect(now) == "P"
    red = _r6k_lines(caplog, "R6-REDIRECT")
    assert red and "pdflip-12-275" in red[0] and "pdflip-12-276" not in red[0], red
    assert f.counters["r6_redirected"] == 1


def test_r6k_class_comes_from_the_arrival_classification():
    """R6k: die Klasse ist die der Ankunfts-Einstufung (``_dec_arrival`` CLASSIFY / D's Messung), keine Schaetzung."""
    import inspect

    from flliper.srt.pdflip import front as F

    src = inspect.getsource(F.Front._dec_arrival)
    assert "phase_policy.classify_arrival(int(remainder), _x6, _f6, bool(p_only)))" in src
    assert src.index("Front._r6k_note_class(self, rid, _k6") < src.index("now = time.monotonic()")   # review K2
    mr = inspect.getsource(F.Front._measure_return)
    assert "Front._r6k_note_class(self, rid, klass, _dn.SRC_D_EXTENT, hx)" in mr
    assert mr.index("Front._r6k_clear_class(self, rid)") < mr.index("sent = ")                     # review K1


def test_r6k_the_kurz_rest_ends_on_d_before_the_flip_not_parked_at_a_chunk_boundary(monkeypatch, tmp_path):
    """R6k "laeuft nach R2 auf D zu Ende" (Planer-Entscheid 20.10.): der D->P-Park wartet, bis der laufende kurze Prefill auf D
    fertig ist (R6K_HOLD, kein Park-RPC, Versuch nicht verbraucht); danach geht der Park raus. Begrenzt (T_halb / gemessene
    Restzeit), ohne Sync: die Front fragt im eigenen Takt neu."""
    import asyncio as _a

    from flliper.srt.pdflip import front as F
    from test_nf_r6_rest_exakt_1020 import REST_AFTER, RID_K, _park_rpc, end_forward

    now, f = _r6k_front(monkeypatch, tmp_path, member_u=9875, klass="kurz", waited=4.6, ended=False)
    calls = []
    _park_rpc(f, calls)
    loop = _a.new_event_loop()
    try:
        assert loop.run_until_complete(f._wait_bound_park(None, immediate=f.queue[0], cause="before-flip")) == F.R6K_HOLD
        assert calls == []
        ws, _ev = f._test_running[RID_K]
        for w in ws:
            w.note_extend(rids=[RID_K], rests=[REST_AFTER], extend_lens=[REST_AFTER])
        end_forward(f, RID_K)
        for w in ws:
            w.note_extend_done(rids=[RID_K])
        assert loop.run_until_complete(f._wait_bound_park(None, immediate=f.queue[0], cause="before-flip")) != F.R6K_HOLD
    finally:
        loop.close()
    assert calls


# ------------------------------------------------------------------------------------------------- R6 + R7 (NF-RC31-R6REST, 21.10.)
# SOLL R6 ("Läuft ein D-Prefill und wächst S (inkl. des laufenden Rests) über X, wird nach P geflippt") + R7 ("Kein Flip unter X")
# + R11. Live chain NF boot ..._82784e1138_1021_202608 (front/D log, /spinning/docker-acceptance/nf/evidence): pdflip-6-401 (halb,
# u=6550) prefills on D, chunk 1 (2240) ended 21:15:36,826 -> free 4310. pdflip-6-403 (halb, u=3786) collects. 36,982: REROUTE_P
# why=R6 S=8096 > X=7370 "with a D prefill of 4310 running". D's next recv step came before the abort (37,114): chunk 2 (2300)
# launched, the abort applied after it (40,07), D-PARK RETAINED tail 2069 -> real S = 3786 + 2010(+page) < X: a flip under X.
# The R6 verdict must stand on the rest that REALLY reaches P: the running prefill is stopped first, the verdict follows on D's
# halt record (the free at the applied abort).
_R6R_RUN, _R6R_SET, _R6R_X, _R6R_HX = "pdflip-6-401", "pdflip-6-403", 7370, 3685


def _r6rest_chain(monkeypatch, tmp_path):
    import time as _time

    from test_nf_r6_rest_exakt_1020 import _front_live
    from test_pdflip_set_scheduler_1017 import _on

    _on(monkeypatch)
    now = _time.time()
    f = _front_live(now, tmp_path, member_u=3786, ended=True, rest_start=6550, ext_len=2240, x=_R6R_X, klass="halb",
                    waited=0.1, rid_run=_R6R_RUN, half_x=_R6R_HX)
    f._set_st()["shorts"][_R6R_SET] = f._set_st()["shorts"].pop("pdflip-12-274")
    ws, _ev = f._test_running[_R6R_RUN]
    for w in ws:
        w.announce_halt()          # the D scheduler wires note_halt (``d_extend_beacon.build``)
    return now, f, ws


def _r6rest_d_applies_the_stop(ws, *, chunk2):
    """D's side after the verdict: ``chunk2`` > 0 = its next recv step came before the abort and launched chunk 2 (live 2300);
    then the abort is applied (``process_pending_chunked_abort``) on every rank: the halt record."""
    for w in ws:
        if chunk2:
            w.note_extend(rids=[_R6R_RUN], rests=[4310], extend_lens=[chunk2])
        assert w.note_halt(rid=_R6R_RUN)


def test_r6_r7_rc31_6401_the_verdict_stands_on_the_rest_that_reaches_p_not_on_the_free_read_before_the_abort(
        monkeypatch, caplog, tmp_path):
    """R6 + R7 (live 21:15:36,982 pdflip-6-401/-403): the verdict on S = 3786 + 4310 = 8096 > X stops the running prefill FIRST
    (no flip yet). D launched chunk 2 (2300) before the abort landed -> 2010 reach P -> S = 5796 < X: NO flip (R7), the stopped
    prefill goes back to D, the set (halb, T_halb running) keeps collecting. Before the fix: REROUTE_P on 8096 at once."""
    import logging as _lg

    from flliper.srt.environ import envs

    now, f, ws = _r6rest_chain(monkeypatch, tmp_path)
    with envs.FLLIPER_PDFLIP_DECIDER.override(True), caplog.at_level(_lg.INFO, logger="pdflip.front"):
        assert f._dc_carried_running() == 4310                         # the free D left at the read (exact at the read)
        assert f._decode_collect(now) == "hold"                         # stopped first, no flip on the read
        assert f.counters["set_reroute_p"] == 0 and f.counters["r6_halt_sent"] == 1
        assert _R6R_RUN in f._rvp_requeue                               # the leg is handed back (stream stays open)
        assert f._decode_collect(now + 0.05) == "hold"                  # D has not applied the stop yet: the verdict waits
        assert f.counters["set_reroute_p"] == 0
        _r6rest_d_applies_the_stop(ws, chunk2=2300)
        assert f._decode_collect(now + 0.1) == "hold"                   # S = 3786 + 2010 = 5796 < 7370: R3 collects on
    msgs = caplog.messages
    assert any("R6-HALT-CONFIRMED rid=%s counted=4310 halted_free=2010 launched_after=2300" % _R6R_RUN in m for m in msgs), msgs
    assert any("R6-HALT-STAYS-ON-D rid=%s halted_free=2010" % _R6R_RUN in m for m in msgs), msgs
    assert not any("verdict=REROUTE_P" in m for m in msgs if "SET-VERDICT" in m)
    assert f.counters["set_reroute_p"] == 0 and f.counters["r6_halt_to_d"] == 1 and f.counters["r6_halt_to_p"] == 0
    assert _R6R_RUN not in f._set_st()["go_p"]
    assert f._dc_carried_running() == 0                                 # released: no second count of the old leg


def test_r6_rc31_6401_without_a_chunk_after_the_verdict_the_halted_rest_flips_r6(monkeypatch, caplog, tmp_path):
    """R6 stays R6: D applied the stop before its next chunk -> 4310 reach P, S = 8096 > X -> REROUTE_P, the stopped prefill
    rides the flip to P."""
    import logging as _lg

    from flliper.srt.environ import envs

    now, f, ws = _r6rest_chain(monkeypatch, tmp_path)
    with envs.FLLIPER_PDFLIP_DECIDER.override(True), caplog.at_level(_lg.INFO, logger="pdflip.front"):
        assert f._decode_collect(now) == "hold"
        _r6rest_d_applies_the_stop(ws, chunk2=0)
        assert f._decode_collect(now + 0.1) == "P"
    assert any("verdict=REROUTE_P" in m and "S=8096" in m for m in caplog.messages if "SET-VERDICT" in m), caplog.messages
    assert f.counters["set_reroute_p"] == 1 and f.counters["r6_halt_to_p"] == 1
    assert {_R6R_RUN, _R6R_SET} <= f._set_st()["go_p"]
