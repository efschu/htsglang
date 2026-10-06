"""NF 06.10.2026 (Boot dkrnfint4h6ablbar1dauer10061420, boot-20261006T142041Z-3e55): von 47 D>P-Flips hatte VM nur 20
Punkte weg2_flip_user_view_ms{def="t2t",dir="D>P",part="total"}.  Nachgerechnet mit dem echten State (events.jsonl +
D-Log): 20 ok, 17 "fehlt" (Start), 9 "leerlauf", 1 "fehlt" (Ende) -- zwei Ursachen, beide in ipcboot.flip_views:

  A  D>P-Flip, in dessen D-Phase D KEINE Decode-Runde schrieb (D wurde geweckt, rechnete nur einen Prefill-Forward und
     schlief wieder ein, 11 Flips, Epochen 53, 59, 63, ... und 6 ohne DP-WAIT 7, 19, 29, 35, 39, 41): die Suche nach
     D's letzter Runde war auf die D-Phase begrenzt -> Start None -> "fehlt", nie ein Punkt.  Strikt ist es das letzte
     Decode-Token, das D je erzeugt hat -- auch aus einer frueheren Phase (27B: int8_matrix_lib.last_d_token).
  B  Die Front stempelt idle_flip am BEGIN (kein Wartender, kein Park).  9 Flips (Epochen 3, 5, 11, 37, 55, 61, 69, 77,
     87) hatten einen Request, der 0,00-0,07 s VOR dem Begin ankam (WEG2 DP-WAIT, Rennen mit der Wartelliste): der
     Request wartete auf den Flip, es stand Prefill an -> kein Leerlauf-Flip, aber kind="leerlauf", nie ein Punkt.

Echte Leerlauf-Flips (der Request kommt erst NACH flip_done, oder nichts ist bekannt) bleiben "leerlauf" und zaehlen
nicht."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from rigdash import flipzeit, ipcboot, vmpush  # noqa: E402

T = 1791296000.0


def _ev(typ, ts, **data):
    return {"type": typ, "ts": ts, "data": data}


def _ipc(idle=False, rid="weg2-2-16"):
    """Two flips of the NF boot's shape: P>D (epoch 2: begin +20.0, done +22.5), then D>P (epoch 3: begin +40.0,
    done +42.2, P's first forward on PP0 at +42.45)."""
    ev = [_ev("flip_begin", T + 20.0, flip_begin_ts=T + 20.0, sleep="P", wake="D", epoch_before=1),
          _ev("flip_done", T + 22.5, flip_begin_ts=T + 20.0, t=T + 22.5, flip_ms=2300, epoch=2, sleep="P", wake="D"),
          _ev("flip_begin", T + 40.0, flip_begin_ts=T + 40.0, sleep="D", wake="P", epoch_before=2),
          _ev("flip_done", T + 42.2, flip_begin_ts=T + 40.0, t=T + 42.2, flip_ms=2000, epoch=3, sleep="D", wake="P")]
    fw = [{"dir": "P>D", "flip_begin_ts": T + 20.0, "first_work_ts": T + 22.9, "what": "d_first_forward_done", "epoch": 2}]
    ut = [{"dir": "D>P", "epoch": 3, "rid": rid, "flip_user_ms": 2400, "idle_flip": idle,
           "start_ts": T + (40.06 if idle else 39.0), "start_source": "first_dispatch_after_idle_flip" if idle else "park_rpc_sent",
           "prefill_start_ts": T + 42.45, "prefill_start_source": "pp_first_forward"}]
    return {"ipc_events": ev, "flip_user_time": ut, "flip_first_work": fw}


SEGS = [{"s": T + 0.0, "e": T + 120.0, "k": "unknown"}]
NOW = T + 60.0


def _sum(x):
    return sum(x[k] for k in ipcboot.PARTS if x.get(k) is not None)


def _dp(rows):
    return next(x for x in rows if x["dir"] == "D>P")


class NoRoundInTheDPhase(unittest.TestCase):
    """A: D was woken at +20.0 and wrote no decode round before it slept again at +40.0."""
    ROUNDS_BEFORE = [(T + 9.0, T + 9.4), (T + 9.4, T + 9.8)]           # D's last token: +9.8, in the phase BEFORE
    LATER = (T + 50.0, T + 50.3)       # a later round in the log: the start is final (else the row is provisional)

    def test_start_without_a_later_round_is_provisional_and_not_counted_yet(self):
        # D's log writes late: until a round after flip_done is in it, the start is not final (PROVISIONAL_MAX_S)
        x = _dp(ipcboot.flip_views(SEGS, _ipc(), NOW, None, d_rounds=self.ROUNDS_BEFORE, arrivals={"weg2-2-16": T + 39.0}))
        self.assertTrue(x["provisional"])
        self.assertFalse(flipzeit.counted(x))
        y = _dp(ipcboot.flip_views(SEGS, _ipc(), T + 42.2 + ipcboot.PROVISIONAL_MAX_S + 1, None,
                                   d_rounds=self.ROUNDS_BEFORE, arrivals={"weg2-2-16": T + 39.0}))
        self.assertTrue(flipzeit.counted(y))

    def test_start_is_the_last_token_of_an_earlier_d_phase_and_the_flip_gets_a_point(self):
        x = _dp(ipcboot.flip_views(SEGS, _ipc(), NOW, None, d_rounds=self.ROUNDS_BEFORE + [self.LATER],
                                   arrivals={"weg2-2-16": T + 39.0}))
        self.assertEqual(x["kind"], "ok")
        # Nutzer 06.10. (Korrektur): Server-Leerlauf ist keine Flipzeit -> Start = max(letztes D-Token +9.8, Ankunft +39.0)
        self.assertAlmostEqual(x["start"], T + 39.0, places=3)
        self.assertAlmostEqual(x["start_last_d"], T + 9.8, places=3)
        self.assertTrue(x["start_prev_phase"])
        self.assertIn("D-Log letzte Decode-Runde", x["start_src"])
        self.assertAlmostEqual(x["total_ms"], (42.45 - 39.0) * 1000, delta=1)
        self.assertAlmostEqual(_sum(x), x["total_ms"], delta=1e-6)             # the partition still sums to the total
        # D's last token -> arrival (D idle, nothing pending) is NOT in the total, but stays visible, outside the sum
        self.assertAlmostEqual(x["leer_excl_ms"], (39.0 - 9.8) * 1000, delta=1)
        self.assertEqual(x["leer_ms"], 0.0)
        self.assertTrue(flipzeit.counted(x))
        pts = vmpush.flip_view_points([x], "NF", "boot-x", set())
        self.assertTrue(any('part="total"' in p and 'dir="D>P"' in p for p in pts), pts)
        self.assertTrue(any('part="leer_excl"' in p for p in pts), pts)

    def test_arrival_before_the_last_token_changes_nothing(self):
        """The request was already waiting when D produced its last token: the start is that token (no idle gap)."""
        rounds = self.ROUNDS_BEFORE + [self.LATER]
        x = _dp(ipcboot.flip_views(SEGS, _ipc(), NOW, None, d_rounds=rounds, arrivals={"weg2-2-16": T + 5.0}))
        self.assertAlmostEqual(x["start"], T + 9.8, places=3)
        self.assertIsNone(x["leer_excl_ms"])
        self.assertAlmostEqual(x["total_ms"], (42.45 - 9.8) * 1000, delta=1)
        self.assertAlmostEqual(_sum(x), x["total_ms"], delta=1e-6)

    def test_a_round_inside_the_phase_still_wins(self):
        rounds = self.ROUNDS_BEFORE + [(T + 30.0, T + 30.4), self.LATER]
        x = _dp(ipcboot.flip_views(SEGS, _ipc(), NOW, None, d_rounds=rounds, arrivals={"weg2-2-16": T + 25.0}))
        self.assertAlmostEqual(x["start"], T + 30.4, places=3)
        self.assertFalse(x.get("start_prev_phase"))
        y = _dp(ipcboot.flip_views(SEGS, _ipc(), NOW, None, d_rounds=rounds, arrivals={"weg2-2-16": T + 39.0}))
        self.assertAlmostEqual(y["start"], T + 39.0, places=3)                  # arrival after the last token: max()
        self.assertAlmostEqual(y["leer_excl_ms"], (39.0 - 30.4) * 1000, delta=1)

    def test_no_round_anywhere_is_still_missing_never_a_guess(self):
        for rounds in ([], [(T + 50.0, T + 50.3)]):                             # none / only AFTER this flip's done
            x = _dp(ipcboot.flip_views(SEGS, _ipc(), NOW, None, d_rounds=rounds, arrivals={}))
            self.assertEqual((x["kind"], x["total_ms"]), ("fehlt", None), rounds)
            self.assertFalse(flipzeit.counted(x))


class IdleFlipWithAWaitingRequest(unittest.TestCase):
    """B: the front's idle_flip (decided at the begin) and the request that arrived at the begin."""
    ROUNDS = [(T + 30.0, T + 30.4), (T + 50.0, T + 50.3)]

    def test_request_that_arrived_before_the_done_waited_for_the_flip(self):
        x = _dp(ipcboot.flip_views(SEGS, _ipc(idle=True), NOW, None, d_rounds=self.ROUNDS, arrivals={"weg2-2-16": T + 39.94}))
        self.assertEqual(x["kind"], "ok")
        self.assertTrue(x["idle_waited"])
        self.assertAlmostEqual(x["total_ms"], (42.45 - 39.94) * 1000, delta=1)     # from the arrival, not D's last token
        self.assertAlmostEqual(x["leer_excl_ms"], (39.94 - 30.4) * 1000, delta=1)
        self.assertAlmostEqual(_sum(x), x["total_ms"], delta=1e-6)
        self.assertTrue(flipzeit.counted(x))

    def test_request_that_arrives_after_the_done_leaves_a_real_idle_flip(self):
        x = _dp(ipcboot.flip_views(SEGS, _ipc(idle=True), NOW, None, d_rounds=self.ROUNDS, arrivals={"weg2-2-16": T + 47.0}))
        self.assertEqual(x["kind"], "leerlauf")
        self.assertFalse(flipzeit.counted(x))
        self.assertEqual(vmpush.flip_view_points([x], "NF", "boot-x", set()), [])

    def test_unknown_arrival_stays_an_idle_flip(self):
        for arr in ({}, None):
            x = _dp(ipcboot.flip_views(SEGS, _ipc(idle=True), NOW, None, d_rounds=self.ROUNDS, arrivals=arr))
            self.assertEqual(x["kind"], "leerlauf", arr)
            self.assertFalse(flipzeit.counted(x))


class DefinitionGuards(unittest.TestCase):
    """Zusatz zum NF-Patch (Branch desk/dash-flipzeit-1006): die Definition (Nutzer 06.10.) an ihren Raendern."""
    ROUNDS = [(T + 30.0, T + 30.4), (T + 50.0, T + 50.3)]

    def test_an_arrival_after_the_first_prefill_forward_is_missing_never_a_zero_total(self):
        x = _dp(ipcboot.flip_views(SEGS, _ipc(), NOW, None, d_rounds=self.ROUNDS, arrivals={"weg2-2-16": T + 44.0}))
        self.assertEqual((x["kind"], x["total_ms"]), ("fehlt", None))
        self.assertEqual(x["missing"], ipcboot.F_DP_ARRIVAL)
        self.assertFalse(flipzeit.counted(x))
        self.assertEqual(vmpush.flip_view_points([x], "NF", "boot-x", set()), [])

    def test_clock_skew_inside_the_tolerance_clamps_to_the_end_of_the_flip(self):
        x = _dp(ipcboot.flip_views(SEGS, _ipc(), NOW, None, d_rounds=self.ROUNDS, arrivals={"weg2-2-16": T + 42.6}))
        self.assertEqual(x["kind"], "ok")
        self.assertAlmostEqual(x["start"], T + 42.45, places=3)

    def test_unknown_arrival_is_named_on_the_row_the_idle_span_cannot_be_taken_out(self):
        x = _dp(ipcboot.flip_views(SEGS, _ipc(), NOW, None, d_rounds=self.ROUNDS, arrivals={}))
        self.assertTrue(x["arrival_unknown"])
        self.assertAlmostEqual(x["start"], T + 30.4, places=3)
        self.assertIsNone(x["leer_excl_ms"])

    def test_idle_span_is_never_in_the_total_and_never_in_the_points_total(self):
        x = _dp(ipcboot.flip_views(SEGS, _ipc(), NOW, None, d_rounds=self.ROUNDS, arrivals={"weg2-2-16": T + 39.0}))
        self.assertAlmostEqual(x["total_ms"], (42.45 - 39.0) * 1000, delta=1)
        self.assertGreater(x["leer_excl_ms"], 8000)
        tot = [p for p in vmpush.flip_view_points([x], "NF", "boot-x", set()) if 'part="total"' in p]
        self.assertEqual(len(tot), 1)


if __name__ == "__main__":
    unittest.main()
