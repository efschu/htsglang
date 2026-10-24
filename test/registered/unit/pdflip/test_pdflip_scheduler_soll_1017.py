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
        # FLIPTIME-1023 SOLL R2 (user 23.10. ~14:40Z): the D-granted kurz prefills on D to its END before D quiesces --
        # the flip verdict waits at the verdict (before: committed, and the park's R6k hold waited at most T_halb)
        assert f._decode_collect(now) == "hold"
        assert _r6k_lines(caplog, "R2-KURZ-BEFORE-FLIP outcome=hold")
        f._d_pf_book().done(rid="pdflip-12-275")                      # its first content: the kurz prefill ended on D
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
        # FLIPTIME-1023 SOLL R2 (user 23.10.): the running kurz prefills on D to its end before D quiesces for the flip
        assert f._decode_collect(now) == "hold"
        assert _r6k_lines(caplog, "R2-KURZ-BEFORE-FLIP outcome=hold rids=pdflip-12-276")
        f._d_pf_book().done(rid="pdflip-12-276")      # its first content: the kurz prefill ended on D
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


# ---------------------------------------------------------------------------------------------------------------------
# R2 / R7: D must be able to TAKE a kurz request. NF boot weg2..._1023_234711 (rc36p), D, 00:12:48-00:13:54Z: the corridor
# actuator cut the pass chunk to its 64-token liveness floor, MIXED-D took running_bs (2) of it for decode rows, and
# ``add_one_req`` aligned 62 down to the page (0): 64 x ``gate=trunc_len_zero`` for pdflip-6-179 (u=927), 180 (u=250) and
# 182 (u=200) never started (waited_s=66.2). The floor was a liveness property defeated one line later.
# R1a stays as it is: a decided flip takes the still-waiting kurz along (PARK-HANDBACK, test_pdflip_park_handback_1002).
# ---------------------------------------------------------------------------------------------------------------------
_PAGE = 64
_MAX_NEW = 4096


def _mixed_adder(rem_chunk, mixed, n_running=2, avail=10**6):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from flliper.srt.managers.schedule_policy import PrefillAdder
    from flliper.srt.mem_cache.base_prefix_cache import DecLockRefResult, IncLockRefResult
    from flliper.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    tc = MagicMock()
    tc.supports_mamba.return_value = False
    tc.evictable_size.return_value = 0
    tc.full_evictable_size.return_value = 0
    tc.swa_evictable_size.return_value = 0
    tc.disable = False
    tc.uniform_avail_floor = None
    tc.inc_lock_ref.return_value = IncLockRefResult()
    tc.dec_lock_ref.return_value = DecLockRefResult()
    alloc = MagicMock()
    alloc.available_size.return_value = avail
    alloc.full_available_size.return_value = avail
    alloc.swa_available_size.return_value = 0
    running = MagicMock()
    running.reqs = [SimpleNamespace(rid=f"run-{i}", output_ids=[],
                                    sampling_params=SimpleNamespace(max_new_tokens=_MAX_NEW)) for i in range(n_running)]
    return PrefillAdder(
        page_size=_PAGE, tree_cache=tc, token_to_kv_pool_allocator=alloc, running_batch=running, new_token_ratio=1.0,
        rem_input_tokens=10**9, rem_chunk_tokens=rem_chunk, num_mixed_decode_tokens=mixed,
        priority_scheduling_preemption_threshold=0,
    )


def _kurz_req(fill, rid="pdflip-6-179"):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from flliper.srt.managers.schedule_batch import Req

    req = MagicMock(spec=Req)
    req.rid = rid
    req.priority = 0
    req.prefix_indices = []
    req.full_untruncated_fill_ids = list(range(fill))
    req.output_ids = []
    req.sampling_params = SimpleNamespace(max_new_tokens=_MAX_NEW, ignore_eos=False)
    req.time_stats = SimpleNamespace(wait_queue_entry_time=0)
    req.retracted_stain = False
    req.finished.return_value = False
    req.needs_host_load_back.return_value = False
    req.host_hit_length = 0
    req.last_node = MagicMock()
    req.born_spilled = False
    req.born_spilled_deep = False
    return req


def test_r2_r7_the_corridor_floor_chunk_survives_the_mixed_decode_rows():
    """The specimen: chunk 64 (corridor liveness floor), 2 decode rows mixed in -> the chunk stays ONE PAGE and the kurz
    request (u=927) is admitted as a chunk instead of being refused ``trunc_len_zero`` pass after pass (R7)."""
    for running in (1, 2, 3, 4, 5):  # rem_chunk=63..59 in the boot's refusals
        adder = _mixed_adder(rem_chunk=64, mixed=running, n_running=running)
        assert adder.rem_chunk_tokens == _PAGE, running
        adder.add_one_req(_kurz_req(927), truncation_align_size=None)
        assert len(adder.can_run_list) == 1, "R2/R7: D takes the kurz request (running=%d)" % running


def test_r2_r7_the_old_subtraction_is_what_refused_it(monkeypatch):
    """Pin of the cause: with the page floor switched off the very same pass reproduces the specimen (62, refused)."""
    monkeypatch.setenv("FLLIPER_PDFLIP_MIXED_CHUNK_PAGE_FLOOR", "0")
    adder = _mixed_adder(rem_chunk=64, mixed=2)
    assert adder.rem_chunk_tokens == 62
    adder.add_one_req(_kurz_req(927), truncation_align_size=None)
    assert adder.can_run_list == []


def test_r2_r7_the_page_floor_changes_nothing_off_the_specimen():
    """Only a budget the subtraction pushes BELOW one page (from >= one page) is lifted; everything else is the old number."""
    from flliper.srt.managers.schedule_policy import _chunk_after_mixed_decode as f

    assert f(4096, 3, _PAGE) == 4093
    assert f(320, 5, _PAGE) == 315
    assert f(128, 64, _PAGE) == 64        # exactly one page left: untouched
    assert f(128, 65, _PAGE) == 64        # pushed below a page, lifted to the floor
    assert f(64, 0, _PAGE) == 64          # no mixed rows: nothing to defend
    assert f(60, 2, _PAGE) == 58          # configured below a page: not ours to lift
    assert f(64, 2, _PAGE) == 64


# ---------------------------------------------------------------------------------------------------------------------
# R2 / R7 (+ R5, R6k untouched): a kurz request is not starved behind a carried continuation. 27B int4ar boot
# weg2..._1023_223241 (D, TP0, X=4096 / halbX=2048 from the front): the corridor narrowed the pass chunk to 256
# (#794 GROUP-NARROWED 4096 -> 256); the continuation (pdflip-2-21, kurz u=1265; elsewhere halb D_AFTER_TIMER u=3805)
# took all 256 through ``add_chunked_req`` before the waiting loop, and every kurz behind it read ``rem_chunk=0``:
# ``D-ADMIT-REFUSED rid=pdflip-2-22 gate=trunc_len_zero rem_chunk=0 uncached=188`` (3 passes, 1.3 s), pdflip-20-190
# u=26 (1.3 s), pdflip-2-25 u=221 (10 passes, 4.3 s) -- 48 TP0 refusals for 18 kurz rids. The seat was granted (R5).
# The fix (pdflip/r2_kurz_chunk.py, wired at the ``lanes_p.chunk_scope`` seam): the kurz needs that fit WHOLE next to the
# continuation's floor are reserved before it; it is charged only what it took.
# ---------------------------------------------------------------------------------------------------------------------
_K_CHUNK = 256
_K_HALFX = 2048


def _k_adder(rem_chunk=_K_CHUNK, page_size=1, num_mixed=0):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from flliper.srt.managers.schedule_policy import PrefillAdder
    from flliper.srt.mem_cache.base_prefix_cache import DecLockRefResult, IncLockRefResult
    from flliper.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    tc = MagicMock()
    tc.supports_mamba.return_value = False
    tc.evictable_size.return_value = 0
    tc.full_evictable_size.return_value = 0
    tc.swa_evictable_size.return_value = 0
    tc.disable = False
    tc.uniform_avail_floor = None
    tc.inc_lock_ref.return_value = IncLockRefResult()
    tc.dec_lock_ref.return_value = DecLockRefResult()
    alloc = MagicMock()
    for name in ("available_size", "full_available_size", "swa_available_size"):
        getattr(alloc, name).return_value = 10**6
    running = MagicMock()
    running.reqs = [SimpleNamespace(rid=f"run-{i}", output_ids=[], sampling_params=SimpleNamespace(max_new_tokens=512))
                    for i in range(num_mixed)]
    return PrefillAdder(
        page_size=page_size, tree_cache=tc, token_to_kv_pool_allocator=alloc, running_batch=running,
        new_token_ratio=1.0, rem_input_tokens=10**9, rem_chunk_tokens=rem_chunk, num_mixed_decode_tokens=num_mixed,
        priority_scheduling_preemption_threshold=0,
    )


def _k_req(rid, fill, prefix=0, *, priced=None, halfx=_K_HALFX, xgate=None, front_x=4096):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from flliper.srt.managers.schedule_batch import Req
    from flliper.srt.utils.common import Range

    r = MagicMock(spec=Req)
    r.rid = rid
    r.priority = 0
    r.prefix_indices = list(range(prefix))
    r.full_untruncated_fill_ids = list(range(fill))
    r.output_ids = []
    r.host_hit_length = 0
    r.swa_host_hit_length = 0
    r.sampling_params = SimpleNamespace(max_new_tokens=512, ignore_eos=False)
    r.time_stats = SimpleNamespace(wait_queue_entry_time=0)
    r.retracted_stain = False
    r.born_spilled = False
    r.born_spilled_deep = False
    r.last_node = MagicMock()
    r.mamba_pool_idx = None
    r.finished.return_value = False
    r.needs_host_load_back.return_value = False
    r.set_extend_range = MagicMock(side_effect=lambda a, b: setattr(r, "extend_range", Range(a, b)))
    if priced is not None:  # the front's D leg headers (front_x.py) and D's own group-priced X-gate reading
        r.weg2_front_x = front_x
        r.weg2_front_priced = priced
        r.weg2_front_halfx = halfx
        r._pdflip_xgate_priced = xgate
    return r


def _k_sched(queue, x=12288):
    from types import SimpleNamespace

    return SimpleNamespace(server_args=SimpleNamespace(tp_prefill_max_tokens=x), waiting_queue=list(queue))


def _k_pass(adder, sched, cont):
    """One D pass as ``Scheduler.get_new_batch_prefill`` runs it: the continuation inside the two scopes, then the
    waiting loop (``add_one_req`` in queue order). Returns the continuation as the scheduler keeps it."""
    from flliper.srt.pdflip import lanes_p
    from flliper.srt.pdflip import r2_kurz_chunk as k

    with lanes_p.chunk_scope(adder, cont), k.reserve_scope(adder, sched, cont):
        kept = adder.add_chunked_req(cont)
    adder.chunked_req_outstanding = kept is not None
    for req in list(sched.waiting_queue):
        adder.add_one_req(req, truncation_align_size=None)
    return kept


def _k_cont():
    # pdflip-2-21 at 22:38:06: prefix 64993, 1265 uncached in all, one 256 chunk a pass
    return _k_req("pdflip-2-21", 64993 + 1265, prefix=64993)


def test_r2_kurz_rides_the_pass_of_a_continuation_that_held_the_whole_chunk():
    """The specimen: chunk 256, continuation with 1265 left; pdflip-2-22 (front priced 700 from P's store, D's own
    group-priced extent 188) and pdflip-20-190 (u=26) are admitted in THIS pass; the continuation keeps the rest."""
    for rid, u, priced in (("pdflip-2-22", 188, 700), ("pdflip-20-190", 26, 26)):
        adder = _k_adder()
        cont = _k_cont()
        kurz = _k_req(rid, u, priced=priced, xgate=(u, 4097))
        kept = _k_pass(adder, _k_sched([kurz]), cont)
        assert kurz in adder.can_run_list, "R2/R7: the kurz request rides this pass (%s)" % rid
        assert kept is cont, "the continuation stays chunked"
        need = u + 1  # D's extent + one page (page_size 1 on 27B)
        assert cont.extend_range.end - cont.extend_range.start == _K_CHUNK - need
        assert kurz.extend_range.end - kurz.extend_range.start == u


def test_r2_kurz_switch_off_is_the_specimen(monkeypatch):
    """Pin of the cause: with the switch off the continuation takes all 256 and the kurz request is refused
    ``trunc_len_zero`` (the boot's 48 refusals)."""
    monkeypatch.setenv("FLLIPER_PDFLIP_R2_KURZ_CHUNK_RESERVE", "0")
    adder = _k_adder()
    cont = _k_cont()
    kurz = _k_req("pdflip-2-22", 188, priced=700, xgate=(188, 4097))
    _k_pass(adder, _k_sched([kurz]), cont)
    assert cont.extend_range.end - cont.extend_range.start == _K_CHUNK
    assert adder.can_run_list == [cont]


def test_r2_kurz_reserve_only_for_the_fronts_kurz_and_never_below_the_continuation_floor():
    """Halb (front priced >= halbX) is never reserved for (R3 owns it); a kurz that does not fit next to the
    continuation's floor (one page) waits as before; several kurz are taken greedily in queue order."""
    from flliper.srt.pdflip import r2_kurz_chunk as k

    halb = _k_req("pdflip-14-122", 2286, priced=2286, xgate=(2286, 4097))
    big_kurz = _k_req("pdflip-4-62", 462, priced=462, xgate=(462, 4097))
    a = _k_req("a", 100, priced=100, xgate=(100, 4097))
    b = _k_req("b", 300, priced=300, xgate=(300, 4097))
    c = _k_req("c", 50, priced=50, xgate=(50, 4097))
    plan = k.plan_reserve([halb, big_kurz, a, b, c], _K_CHUNK, 1, 1)
    assert plan.rids == [("a", 101), ("c", 51)]
    assert [r for r, _ in plan.nofit] == ["pdflip-4-62", "b"]
    assert plan.reserve == 152
    # the continuation always keeps its floor: a kurz of exactly chunk-floor+1 needs does not fit
    edge = _k_req("e", _K_CHUNK - 1, priced=_K_CHUNK - 1, xgate=(_K_CHUNK - 1, 4097))   # need 256 > room 255
    assert k.plan_reserve([edge], _K_CHUNK, 1, 1).reserve == 0
    # the halb continuation still progresses with the floor when the kurz take the rest
    adder = _k_adder()
    cont = _k_cont()
    fill = _k_req("f", _K_CHUNK - 2, priced=_K_CHUNK - 2, xgate=(_K_CHUNK - 2, 4097))   # need 255 = room
    _k_pass(adder, _k_sched([fill]), cont)
    assert cont.extend_range.end - cont.extend_range.start == 1
    assert fill in adder.can_run_list


def test_r2_kurz_reserve_reads_ds_extent_the_measure_barrier_and_nothing_unpriced():
    """Size = D's group-priced X-gate extent (never the front's price); a MEASURE barrier (price 0) is kurz by D's extent
    under halbX; no X-gate reading yet / no carried halbX / an already decoding request = no reservation."""
    from flliper.srt.pdflip import r2_kurz_chunk as k

    assert k.kurz_need(_k_req("p", 188, priced=700, xgate=(188, 4097)), 1) == 189
    assert k.kurz_need(_k_req("m", 908, priced=0, halfx=2047, xgate=(908, 2047)), 1) == 909   # pdflip-38-298 shape
    assert k.kurz_need(_k_req("mh", 2100, priced=0, halfx=2047, xgate=(2100, 2047)), 1) is None
    assert k.kurz_need(_k_req("nx", 188, priced=188, xgate=None), 1) is None
    assert k.kurz_need(_k_req("nh", 188), 1) is None
    dec = _k_req("dec", 188, priced=188, xgate=(188, 4097))
    dec.output_ids = [7]
    assert k.kurz_need(dec, 1) is None
    assert k.kurz_need(_k_req("pg", 100, priced=100, xgate=(100, 4097)), 64) == 192   # ceil to the page + one page


def test_r2_kurz_reserve_is_bounded_group_d_only_and_restores_the_groups_charge():
    """MAX_TRIES caps a reservation another gate keeps refusing; group P (no riegel) and a forwarded schedule never
    reserve; the charge restored is what the continuation REALLY took (CHUNKGEOM group MIN cut it further)."""
    from flliper.srt.pdflip import r2_kurz_chunk as k

    kurz = _k_req("pdflip-30-275", 188, priced=188, xgate=(188, 4097))
    for _ in range(k.MAX_TRIES):
        adder = _k_adder()
        with k.reserve_scope(adder, _k_sched([kurz]), _k_cont()):
            assert adder.rem_chunk_tokens == _K_CHUNK - 189
    adder = _k_adder()
    with k.reserve_scope(adder, _k_sched([kurz]), _k_cont()):
        assert adder.rem_chunk_tokens == _K_CHUNK, "after MAX_TRIES the continuation is not cut any more"
    fresh = _k_req("x", 188, priced=188, xgate=(188, 4097))
    assert k.plan_for_pass(_k_adder(), _k_sched([fresh], x=0), _k_cont()) is None            # group P
    told = _k_adder()
    told.scheduled_extents = {"pdflip-2-21": 1}
    assert k.plan_for_pass(told, _k_sched([fresh]), _k_cont()) is None                         # executes a schedule
    class _Agree:  # a two-rank group whose other rank funds only 40 continuation tokens (CHUNKGEOM MIN)
        def __call__(self, req, gate, total_tokens, rem_total):
            return "ADMIT" if gate is None else gate  # both ranks admit what this rank admits

        def chunk_min(self, rid, v):
            return min(int(v), 40)

    adder = _k_adder()
    adder.tp_admission_agree = _Agree()
    cont = _k_cont()
    _k_pass(adder, _k_sched([fresh]), cont)
    assert cont.extend_range.end - cont.extend_range.start == 40
    assert fresh in adder.can_run_list
    assert adder.rem_chunk_tokens == _K_CHUNK - 40 - 188


def test_r2_kurz_reserve_is_wired_at_the_continuation_seam():
    """The admission pass itself is too large to drive here: pinned by text -- the reservation wraps the continuation's
    ``add_chunked_req`` (inside the lane scope) and therefore precedes the waiting-queue loop."""
    import flliper.srt.managers.scheduler as sch

    src = open(sch.__file__).read()
    i = src.index("_pdflip_r2_kurz_chunk.reserve_scope(")
    j = src.index("self.chunked_req = adder.add_chunked_req(self.chunked_req)", i)
    assert j - i < 200
    assert src.rfind("_lanes_p.chunk_scope(adder, self.chunked_req)", 0, i) > i - 200
    assert src.index("adder.chunked_req_outstanding = self.chunked_req is not None", j) > j


# -- R4OVER-1024: R4 "Flip erst, wenn ... T_halb ablaeuft" -- a kurz grant opened AFTER the verdict never delays the flip ----
def _r4over_scene(now, *, waited=4.6):
    """Boot int4ar 1023_223241 epoch 12 (27B, X=4096, halbX=2048, T_halb=4,5 s): pdflip-12-113 (gross, u=4476) COLLECT (R4)
    since 22:48:01,586, D decodes; at T_halb over the kurz pdflip-12-115 (u=181) is D-granted and prefilling -- the R2 hold
    (FLIPTIME-1023) waits for it. Then pdflip-12-116 (u=1222) arrives 1,0 s after the verdict."""
    from test_pdflip_arrival_seat_rule_0929 import _front, _pending

    f = _front(running=["dec", "k1"], n=6, kv={"available": 400000, "evictable": 0})   # "dec" decodes on D (R4 waits)
    f.tp_prefill_max_tokens = 4096
    f.t_awake = now - 60.0
    g = _pending("pdflip-12-113", 4476, now - waited)
    f.queue = [g]
    f._d_pf_book().grant(rid="k1", tokens=181, now=now - 0.2)
    f._d_pf_book().enter_leg2(rid="k1")
    f.groups["D"].outstanding["k1"] = now - 0.2
    f._r6k_mark_kurz("k1")
    f._d_ext_reading = lambda: {}
    return f, g


def _r4over_grant_late(f, now, rid="k2", tokens=1222):
    f._d_pf_book().grant(rid=rid, tokens=tokens, now=now)
    f._d_pf_book().enter_leg2(rid=rid)
    f.groups["D"].outstanding[rid] = now
    f._r6k_mark_kurz(rid)


def test_r4_t_half_over_a_kurz_grant_after_the_verdict_never_delays_the_flip():
    """R4 (SOLL v2: "Decodet D gerade: Flip erst, wenn ein weiterer Halb-/Gross-Request dazukommt oder T_halb ablaeuft") +
    R1a ("Ist ein Flip nach R3/R4/R6 beschlossen, prefillt P ALLE anstehenden Prefills, auch kurze") + R5 ("oder faehrt mit
    einem beschlossenen Flip mit"). Defect (boot 1023_223241, epochs 2/12): the R2 hold turned the decided FLIP_P into
    COLLECT, a new kurz arrival saw no decided flip, was granted D_NOW and was APPENDED to the hold -- pdflip-12-116 kept the
    R4 flip of pdflip-12-113 back 3,9 s (waited_s=9.2 instead of 4,6 + the rest of pdflip-12-115).
    SOLL: T_halb over = the flip is decided; it waits only for the kurz grant open at the verdict (R2/R6k), a later kurz
    arrival rides it; the flip stands at T_halb + that grant's rest + one tick."""
    import time as _time
    from unittest import mock

    from flliper.srt.environ import envs

    env = {"FLLIPER_PDFLIP_HALF_FLIP_S": str(T_HALF), "FLLIPER_PDFLIP_HALF_X_FRAC": str(HALF),
           "FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_RULE": "1"}
    for decider in (True, False):
        with mock.patch.dict(os.environ, env), envs.FLLIPER_PDFLIP_DECIDER.override(decider):
            now = _time.time()
            # before T_halb: R4 COLLECT, nothing decided -- a kurz arrival is D's (R2)
            f, _g = _r4over_scene(now, waited=4.0)
            assert f._decode_collect(now) == "hold", decider
            assert f.counters["r2_kurz_grant_hold"] == 0
            assert not f._flip_decided()
            # T_halb over (4,6 s): FLIP_P, held for the kurz granted BEFORE it (R2/R6k)
            f, _g = _r4over_scene(now)
            assert f._decode_collect(now) == "hold", decider
            assert f.counters["r2_kurz_grant_hold"] == 1 and f.counters["set_flip_p"] == 0
            # the flip is DECIDED while held: a new kurz arrival rides it (R1a), even with a free D seat
            assert f._flip_decided(), decider
            assert f._set_flip_foreseeable(now + 0.05), decider
            assert f._d_direct_yields("pdflip-12-116", uncached=1222), decider
            # a grant that slipped through anyway (race) is named, never waited on
            _r4over_grant_late(f, now + 0.1)
            assert f._decode_collect(now + 0.15) == "hold"            # k1 still prefills on D (R6k: it ends on D)
            assert f.counters["r2_kurz_grant_hold_late"] == 1
            f._d_pf_book().done(rid="k1")                               # k1's first content: its D prefill is over
            assert f._decode_collect(now + 0.2) == "P", decider         # T_halb + k1's rest + one tick; k2 still open
            assert f.counters["set_flip_p"] == 1
            st = f.__dict__["_r2_kurz_hold"]
            assert st["rids"] == ["k1"] and st["late"] == ["k2"] and st["closed"]


def test_r4_held_flip_is_decided_only_while_its_set_stands():
    """Danger direction of R4OVER-1024: "decided" must not outlive the held flip -- else a kurz arrival rides a flip that
    never comes (R2 "sofort auf D"). The big member left the queue, or the epoch moved: no decided flip, R2 again."""
    import time as _time
    from unittest import mock

    from flliper.srt.environ import envs

    env = {"FLLIPER_PDFLIP_HALF_FLIP_S": str(T_HALF), "FLLIPER_PDFLIP_HALF_X_FRAC": str(HALF),
           "FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_RULE": "1"}
    with mock.patch.dict(os.environ, env), envs.FLLIPER_PDFLIP_DECIDER.override(True):
        now = _time.time()
        f, g = _r4over_scene(now)
        assert f._decode_collect(now) == "hold" and f._flip_decided()
        g.fut._d = True                                                 # the big request is gone (client closed)
        assert not f._flip_decided() and not f._r2_flip_held()
        f, g = _r4over_scene(now)
        assert f._decode_collect(now) == "hold" and f._flip_decided()
        f.epoch += 1                                                    # the phase moved
        assert not f._r2_flip_held()
        # switch off: the old hold (FLIPTIME-1023) as before -- no decided flip while held
        with envs.FLLIPER_PDFLIP_R2_KURZ_GRANT_HOLD.override(False):
            f, g = _r4over_scene(now)
            assert f._decode_collect(now) == "P"


def test_r4_arrival_seat_pbound_flip_waits_only_for_grants_open_at_its_start():
    """R4OVER-1024 on the ARRIVAL-SEAT P-bound site (FLIPTIME-1023 hold outside a set): the same frozen wait."""
    import time as _time
    from unittest import mock

    from flliper.srt.environ import envs

    env = {"FLLIPER_PDFLIP_HALF_FLIP_S": str(T_HALF), "FLLIPER_PDFLIP_HALF_X_FRAC": str(HALF),
           "FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_RULE": "1"}
    with mock.patch.dict(os.environ, env), envs.FLLIPER_PDFLIP_DECIDER.override(True):
        now = _time.time()
        f, g = _r4over_scene(now)
        assert f._r2_pbound_wait(["k1"], [g]) == ["k1"]
        assert f._r2_flip_held() and f._flip_decided()
        assert f._r2_pbound_wait(["k1", "k2"], [g]) == ["k1"]          # k2 opened after: named, not waited on
        assert f.counters["r2_kurz_grant_hold_late"] == 1
        assert f._r2_pbound_wait(["k2"], [g]) == []                     # k1 over: the P-bound flip goes
        assert not f._r2_flip_held()


def test_r4_an_r6_commit_tick_does_not_void_the_held_flip():
    """R4OVER-1024 review A3: the one-tick R6 commit hold sits BEHIND the R2 hold -- it can only add to S, never turn a flip
    into a non-flip, so it never voids a held flip: in that tick the flip stays decided and a kurz arrival rides it (R1a)."""
    import time as _time
    from unittest import mock

    from flliper.srt.environ import envs
    from flliper.srt.pdflip import front as front_mod
    from flliper.srt.pdflip import phase_policy as _pp

    env = {"FLLIPER_PDFLIP_HALF_FLIP_S": str(T_HALF), "FLLIPER_PDFLIP_HALF_X_FRAC": str(HALF),
           "FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_RULE": "1"}
    with mock.patch.dict(os.environ, env), envs.FLLIPER_PDFLIP_DECIDER.override(True):
        now = _time.time()
        f, _g = _r4over_scene(now)
        assert f._decode_collect(now) == "hold" and f._flip_decided()
        with mock.patch.object(front_mod.Front, "_r6_commit_hold",
                               lambda self, sv: _pp.SetVerdict(_pp.V_COLLECT, "test: R6 commit tick", 0.0, sv.s_tokens,
                                                               sv.half_x)):
            assert f._decode_collect(now + 0.05) == "hold"
            assert f._flip_decided()                                   # still decided in the commit tick
            assert f._d_direct_yields("pdflip-12-117", uncached=300)   # the kurz arrival rides
            assert f.counters["r2_kurz_grant_hold_void"] == 0
        f._d_pf_book().done(rid="k1")
        assert f._decode_collect(now + 0.1) == "P"
        assert f.counters["r2_kurz_grant_hold"] == 1                   # one hold piece, not restarted


# -- R4OVER-1024 review probes P1-P3 (adopted from review/r4probe-1024, test_review_r4probe_1024.py) -------------------------
def test_probe_p1_stale_hold_after_client_gone_is_not_reused_by_the_next_flip():
    """Review A1 (SOLL R2/R6k): the held flip's member left (CLIENT-GONE) -> the set empties, the hold ends (void); a kurz
    granted while NO flip stood is R2's and the NEXT flip of the same epoch waits for it -- never "late"."""
    import time as _time
    from unittest import mock

    from test_pdflip_arrival_seat_rule_0929 import _pending

    from flliper.srt.environ import envs

    env = {"FLLIPER_PDFLIP_HALF_FLIP_S": str(T_HALF), "FLLIPER_PDFLIP_HALF_X_FRAC": str(HALF),
           "FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_RULE": "1"}
    with mock.patch.dict(os.environ, env), envs.FLLIPER_PDFLIP_DECIDER.override(True):
        now = _time.time()
        f, g = _r4over_scene(now)
        assert f._decode_collect(now) == "hold" and f._flip_decided()
        g.fut._d = True
        assert f._decode_collect(now + 0.05) is None
        assert not f._flip_decided()
        assert f.counters["r2_kurz_grant_hold_void"] == 1 and f.counters["r2_held_flip_no_flip"] == 1
        f._d_pf_book().done(rid="k1")
        _r4over_grant_late(f, now + 0.1, rid="k3", tokens=300)
        f.queue = [_pending("pdflip-12-200", 4476, now + 0.1 - 4.6)]
        got = f._decode_collect(now + 0.2)
        assert f.counters["r2_kurz_grant_hold_late"] == 0, (got, f.__dict__["_r2_kurz_hold"])
        assert got == "hold"
        assert f.__dict__["_r2_kurz_hold"]["rids"] == ["k3"]


def test_probe_p2_held_park_no_seat_is_never_a_decided_flip():
    """Review A4 (SOLL R5): a held PARK_NO_SEAT is no flip begin -- never a decided flip (kills mutant M1)."""
    import time as _time
    from unittest import mock

    from flliper.srt.environ import envs
    from flliper.srt.pdflip import phase_policy as _pp

    env = {"FLLIPER_PDFLIP_HALF_FLIP_S": str(T_HALF), "FLLIPER_PDFLIP_HALF_X_FRAC": str(HALF),
           "FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_RULE": "1"}
    with mock.patch.dict(os.environ, env), envs.FLLIPER_PDFLIP_DECIDER.override(True):
        now = _time.time()
        f, g = _r4over_scene(now)
        st = {"epoch": f.epoch, "t0": now, "t0_wall": now, "rids": ["k1"], "late": [],
              "last_verdict": _pp.V_PARK_NO_SEAT, "members": [g.rid]}
        f.__dict__["_r2_kurz_hold"] = st
        assert not f._r2_flip_held()
        assert not f._flip_decided()
        st["last_verdict"] = _pp.V_FLIP_P
        assert f._r2_flip_held() and f._flip_decided()


def test_probe_p3_member_reprobed_to_kurz_ends_the_decided_state():
    """Review A1: SET-REPROBE moves the held big member under halbX (d_eligible kurz, no set member) -- the flip can never
    come, "decided" ends (else every kurz arrival yields to a flip that never comes, R2)."""
    import time as _time
    from unittest import mock

    from flliper.srt.environ import envs

    env = {"FLLIPER_PDFLIP_HALF_FLIP_S": str(T_HALF), "FLLIPER_PDFLIP_HALF_X_FRAC": str(HALF),
           "FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_RULE": "1"}
    with mock.patch.dict(os.environ, env), envs.FLLIPER_PDFLIP_DECIDER.override(True):
        now = _time.time()
        f, g = _r4over_scene(now)
        assert f._decode_collect(now) == "hold" and f._flip_decided()
        g.est_uncached = 300
        g.d_eligible = True
        assert not f._flip_decided()                                    # the member alone ends it (A1, _r2_flip_held)
        f._decode_collect(now + 0.05)
        assert not f._flip_decided(), f.__dict__["_r2_kurz_hold"]


def test_r4_held_flip_is_not_decided_while_its_members_are_reprobed():
    """Review A5 (S63 F1): while a SET-REPROBE runs, the held flip is not decided on D's current numbers -- no rider."""
    import time as _time
    import types as _types
    from unittest import mock

    from flliper.srt.environ import envs

    env = {"FLLIPER_PDFLIP_HALF_FLIP_S": str(T_HALF), "FLLIPER_PDFLIP_HALF_X_FRAC": str(HALF),
           "FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_RULE": "1"}
    with mock.patch.dict(os.environ, env), envs.FLLIPER_PDFLIP_DECIDER.override(True):
        now = _time.time()
        f, g = _r4over_scene(now)
        assert f._decode_collect(now) == "hold" and f._flip_decided()
        task = _types.SimpleNamespace(done=lambda: False)
        f._set_st()["reprobe"] = _types.SimpleNamespace(task=task, rids=(g.rid,), t0=now, site="set-verdict")
        assert not f._flip_decided()
        task.done = lambda: True
        assert f._flip_decided()


def test_r4_a_running_late_grant_is_held_by_the_r6k_park_hold(monkeypatch, tmp_path):
    """Review A4 (SOLL R6k "rechnet auf D zu Ende"): a late kurz grant that RUNS when the held flip goes is not waited on
    by the R2 hold -- the park's R6k finish hold holds it (R6K_HOLD, no park RPC); the control is the frozen grant alone."""
    import asyncio as _a

    from flliper.srt.environ import envs
    from flliper.srt.pdflip import front as F
    from test_nf_r6_rest_exakt_1020 import REST_AFTER, RID_K, _park_rpc, add_running, end_forward

    now, f = _r6k_front(monkeypatch, tmp_path, member_u=9875, klass="kurz", waited=4.6, ended=False)
    calls = []
    _park_rpc(f, calls)
    with envs.FLLIPER_PDFLIP_DECIDER.override(True):
        assert f._decode_collect(now) == "hold"                       # held for RID_K (open at the verdict)
        add_running(f, now, tmp_path, "pdflip-12-279", rest_start=900, ext_len=512, ended=False, klass="kurz", key=37877)
        ws, _ev = f._test_running[RID_K]
        for w in ws:
            w.note_extend(rids=[RID_K], rests=[REST_AFTER], extend_lens=[REST_AFTER])
        end_forward(f, RID_K)
        for w in ws:
            w.note_extend_done(rids=[RID_K])
        f._d_pf_book().done(rid=RID_K)
        assert f._decode_collect(now + 0.1) == "P"                    # the late running grant is not waited on ...
        assert f.counters["r2_kurz_grant_hold_late"] == 1
    loop = _a.new_event_loop()
    try:
        # ... but the park holds it to its end on D (R6k), no park RPC
        assert loop.run_until_complete(f._wait_bound_park(None, immediate=f.queue[0], cause="before-flip")) == F.R6K_HOLD
        assert calls == []
    finally:
        loop.close()


def test_r4_a_request_at_the_seat_semaphore_when_the_flip_is_decided_rides_without_a_grant():
    """Review A2 (SOLL R1a/R6k): a kurz arrival passed both yield checks (no flip yet) and waited at D's seat semaphore; the
    R4 flip became decided (held) meanwhile -> it releases the permit and rides (None), no D grant (an UNBEGUN late grant
    the R6k finish hold could not see)."""
    import asyncio as _a
    import time as _time
    from unittest import mock

    from flliper.srt.environ import envs
    from flliper.srt.pdflip import front as front_mod

    env = {"FLLIPER_PDFLIP_HALF_FLIP_S": str(T_HALF), "FLLIPER_PDFLIP_HALF_X_FRAC": str(HALF),
           "FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_RULE": "0"}
    with mock.patch.dict(os.environ, env), envs.FLLIPER_PDFLIP_DECIDER.override(True):
        now = _time.time()
        f, g = _r4over_scene(now, waited=4.0)                          # R4 still collecting: nothing decided

        class _Sem:
            released = 0

            async def acquire(self):
                g.t_arrive = now - 4.6                                 # T_halb runs out while it waits
                assert f._decode_collect(time.time()) == "hold"        # FLIP_P held for k1: decided
                return True

            def release(self):
                _Sem.released += 1

        import time
        f._d_seat = _Sem()

        async def _no_reading(self, rid):
            return None

        with mock.patch.object(front_mod.Front, "_d_reading_kurz", _no_reading):
            loop = _a.new_event_loop()
            try:
                got = loop.run_until_complete(f._acquire_short_seat_core(
                    "pdflip-12-116", 110278, [], 64000, 1222, 4096, now))
            finally:
                loop.close()
        assert got is None
        assert _Sem.released == 1 and f.counters["short_seat_yield_after_sem"] == 1
        assert "pdflip-12-116" not in f._d_pf_book().open_rids(now=time.time(), live=f.groups["D"].outstanding,
                                                               idle=lambda r: False)


# ---------------------------------------------------------------------------------------------------------------------
# Review review-27b-r2kurz-mixeddflash-1024 (Opus, JA MIT AUFLAGEN): A1 -- the three combinations that MIXED-D under DFLASH
# makes live or that were only read (probe P1/P1b/P2/P3); A2 census; H1 END-ANCHOR riegel; H2 tries restart.
# ---------------------------------------------------------------------------------------------------------------------
def test_r2_kurz_a1a_reservation_after_the_mixed_decode_deduction_no_double_charge():
    """(a) MIXED-D armed: 6 decode rows take 6 of the 256 at the adder's init (250); the reservation works on the rest,
    the continuation gets 250 - 189, the kurz request rides, exactly the one-page slack is left; the largest kurz that
    still fits (need 249 = 250 - floor) leaves the continuation its floor of one page."""
    adder = _k_adder(num_mixed=6)
    assert adder.rem_chunk_tokens == 250
    cont = _k_cont()
    kurz = _k_req("pdflip-2-22", 188, priced=700, xgate=(188, 4097))
    assert _k_pass(adder, _k_sched([kurz]), cont) is cont
    assert kurz in adder.can_run_list
    assert cont.extend_range.end - cont.extend_range.start == 250 - 189
    assert adder.rem_chunk_tokens == 1
    adder = _k_adder(num_mixed=6)
    cont = _k_cont()
    big = _k_req("k248", 248, priced=248, xgate=(248, 4097))
    _k_pass(adder, _k_sched([big]), cont)
    assert big in adder.can_run_list and cont.extend_range.end - cont.extend_range.start == 1


def test_r2_kurz_a1b_two_ranks_with_different_plans_heal_through_the_shared_chunk_min():
    """(b) rank A carries the X-gate reading (reserves 189), rank B does not (plans the whole chunk): the CHUNKGEOM group
    MIN gives both continuations 67 rows, both ranks restore the same budget and admit the kurz request alike."""
    seen = {"A": 256 - 189, "B": 256}   # the gather: both locals known before either reads the MIN

    def run(xgate):
        adder = _k_adder()

        class _Agree:
            def __call__(self, req, gate, total_tokens, rem_total):
                return "ADMIT" if gate is None else gate

            def chunk_min(self, rid, v):
                return min(seen.values())

        adder.tp_admission_agree = _Agree()
        cont = _k_cont()
        kurz = _k_req("pdflip-20-190", 188, priced=188, xgate=xgate)
        _k_pass(adder, _k_sched([kurz]), cont)
        return adder, cont, kurz

    a, ca, ka = run((188, 4097))
    b, cb, kb = run(None)
    assert ca.extend_range.end - ca.extend_range.start == cb.extend_range.end - cb.extend_range.start == 67
    assert a.rem_chunk_tokens == b.rem_chunk_tokens
    assert (ka in a.can_run_list) and (kb in b.can_run_list)


def test_r2_kurz_a1c_nested_in_the_lane_cap_composes_to_outer_minus_taken():
    """(c) ``lanes_p.chunk_scope`` outside (lane cap 128 while a higher lane waits), the reservation inside: the
    continuation gets 128 - 51, the outer budget is charged exactly what it took."""
    from flliper.srt.pdflip import lanes_p
    from flliper.srt.pdflip import r2_kurz_chunk as k

    adder = _k_adder()
    adder.lane_chunk_cap_tokens = 128
    adder.lane_higher_waiting = 5
    cont = _k_cont()
    kurz = _k_req("c", 50, priced=50, xgate=(50, 4097))
    with lanes_p.chunk_scope(adder, cont), k.reserve_scope(adder, _k_sched([kurz]), cont):
        adder.add_chunked_req(cont)
    assert cont.extend_range.end - cont.extend_range.start == 128 - 51
    assert adder.rem_chunk_tokens == 256 - (128 - 51)


def test_r2_kurz_a2_census_counts_reserved_wasted_and_the_floor_streak():
    """A2: a reserved request that still waits the next pass is counted wasted (with its need, once); a continuation
    held at <= 1/4 of the chunk pass after pass is the floor streak."""
    from flliper.srt.pdflip import r2_kurz_chunk as k

    for key in ("reserve_passes", "reserved_tokens", "wasted", "wasted_tokens", "consec_floor", "max_consec_floor"):
        k.CENSUS[key] = 0
    k.CENSUS["floor_cont"] = ""
    kurz = _k_req("w", 230, priced=230, xgate=(230, 4097))     # need 231: the continuation keeps 25 <= 64
    sched = _k_sched([kurz])
    for _ in range(2):   # reserved, but nobody admits it (another gate): still waiting
        adder = _k_adder()
        with k.reserve_scope(adder, sched, _k_cont()):
            adder.add_chunked_req(_k_cont())
    assert k.CENSUS["reserve_passes"] == 2 and k.CENSUS["reserved_tokens"] == 2 * 231
    assert k.CENSUS["wasted"] == 1 and k.CENSUS["wasted_tokens"] == 231
    assert k.CENSUS["max_consec_floor"] == 2
    adder = _k_adder()
    cont = _k_cont()
    fresh = _k_req("v", 100, priced=100, xgate=(100, 4097))
    sched2 = _k_sched([fresh])
    _k_pass(adder, sched2, cont)                     # admitted in its pass
    sched2.waiting_queue = []                        # the batch took it out of the queue
    with k.reserve_scope(_k_adder(), sched2, _k_cont()):
        pass
    assert k.CENSUS["wasted"] == 1 and k.CENSUS["wasted_tokens"] == 231   # 'v' was admitted and left: no waste
    assert k.CENSUS["consec_floor"] == 0 and k.CENSUS["max_consec_floor"] == 2   # 'v' left the continuation 155 rows


def test_r2_kurz_h1_no_reservation_when_the_end_anchor_cut_would_refuse_it(monkeypatch):
    """H1: END-ANCHOR armed without multi-anchor tails -> the reserved request would be refused
    ``second_continuation_end_anchor``; no reservation at all."""
    from flliper.srt.managers import schedule_policy as sp
    from flliper.srt.pdflip import r2_kurz_chunk as k

    kurz = _k_req("e", 100, priced=100, xgate=(100, 4097))
    monkeypatch.setattr(sp, "_PDFLIP_END_ANCHOR", True)
    adder = _k_adder()
    adder.multi_anchor_tails = False
    assert k.plan_for_pass(adder, _k_sched([kurz]), _k_cont()) is None
    adder.multi_anchor_tails = True
    assert k.plan_for_pass(adder, _k_sched([kurz]), _k_cont()).reserve == 101


def test_r2_kurz_h2_tries_restart_behind_a_new_continuation_or_a_new_pricing():
    """H2: MAX_TRIES counts per (continuation, X-gate reading); behind the next continuation, or after D re-priced the
    request, it is reserved for again (the waste stays bounded per continuation)."""
    from flliper.srt.pdflip import r2_kurz_chunk as k

    kurz = _k_req("t", 100, priced=100, xgate=(100, 4097))
    sched = _k_sched([kurz])
    for _ in range(k.MAX_TRIES):
        with k.reserve_scope(_k_adder(), sched, _k_cont()):
            pass
    assert k.plan_for_pass(_k_adder(), sched, _k_cont()).reserve == 0
    other = _k_req("pdflip-2-23", 64993 + 3805, prefix=64993)
    assert k.plan_for_pass(_k_adder(), sched, other).reserve == 101
    kurz._pdflip_xgate_priced = (120, 4097)
    assert k.plan_for_pass(_k_adder(), sched, _k_cont()).reserve == 121

