"""Nutzer 02.10.: FLIPZEIT = vom LETZTEN Token der abgebenden Phase bis zum ERSTEN Token der annehmenden, beide
Richtungen, beide Modelle -- die eine Zahl; die Teile (Vorlauf, Layer, Wake-KV/DC, Nachlauf, Rest) summieren zu
ihr.  Zahlen aus NF y7l (boot ...120607Z-068d, Flip 2 P>D 12:10:08 und Flip 3 D>P 12:10:44, events.jsonl und
der rigdash-Ring als Vorlage):

* D>P endete bisher am Leg-1-Dispatch (2513 ms); der erste Forward auf PP2 kam erst 12:10:53 (PP2-forward_ct
  6 -> 7 zwischen den Rang-Takten 052,850 und 053,910) -- Flipzeit 9,8 s, Nachlauf 6,2 s + Rest (Takt) 1,06 s.
* P>D: D gab sein erstes Token 1,9 s VOR flip_done (waehrend wake-kv/dc) -- 2603 ms, nicht 4496 ms.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from rigdash import history, ipcboot, vmpush  # noqa: E402

T = 1790943000.0


def _ev(typ, ts, **data):
    return {"type": typ, "ts": ts, "data": data}


def _ring(pp2_rise_at=53.910, d_rise_at=None):
    """1-s samples 40..58 s: P.tp0pp0/P.tp0pp2 forward_ct (PP2 6 -> 7 at ``pp2_rise_at``), D.tp0pp0 counters."""
    out = []
    for i in range(19):
        t = T + 40.5 + i
        pp2_ts = T + 39.845 + i + (0.065 if i >= 14 else 0.0)        # the rank clock slipped 65 ms at the forward
        r = {"P.tp0pp0": {"ts": T + 39.74 + i, "fwd": 6.0 + (i >= 8) + (i >= 12)},
             "P.tp0pp2": {"ts": pp2_ts, "fwd": 6.0 + (pp2_ts >= T + pp2_rise_at - 1e-6)},
             "D.tp0pp0": {"ts": T + 39.95 + i, "fwd": 614.0, "dtok": 2455.0, "rounds": 608.0, "pnew": 210.0}}
        if d_rise_at is not None and r["D.tp0pp0"]["ts"] >= T + d_rise_at:
            r["D.tp0pp0"].update(dtok=2466.0, rounds=611.0)
        out.append({"t": t, "r": r})
    return out


def _ipc_dp(prefill_source="leg1_dispatch", prefill_ts=46.608):
    ev = [_ev("flip_begin", T + 44.319, flip_begin_ts=T + 44.317, sleep="D", wake="P", epoch_before=2),
          _ev("flip_done", T + 46.608, flip_begin_ts=T + 44.317, t=T + 46.605589, flip_ms=2288, epoch=3, sleep="D", wake="P")]
    ut = [{"dir": "D>P", "epoch": 3, "flip_user_ms": 2513, "idle_flip": False, "start_ts": T + 44.096,
           "start_source": "park_rpc_sent", "prefill_start_ts": T + prefill_ts, "prefill_start_source": prefill_source,
           "parts": {"first_chunk_ms": 3, "legs_ms": 2289, "park_rpc_ms": 219, "pre_begin_ms": 221}}]
    fw = [{"dir": "D>P", "flip_begin_ts": T + 44.317, "first_work_ts": T + 46.608, "flip_time_ms": 2291,
           "what": "p_leg1_dispatch", "epoch": 3}]
    return {"ipc_events": ev, "flip_user_time": ut, "flip_first_work": fw}


SEGS = [{"s": T + 0.0, "e": T + 60.0, "k": "unknown"}]


def _sum(x):
    return sum(x[k] for k in ipcboot.PARTS if x.get(k) is not None)


class FlipzeitDP(unittest.TestCase):
    def test_dp_ends_at_first_forward_of_last_pp_stage_not_leg1_dispatch(self):
        x = ipcboot.flip_views(SEGS, _ipc_dp(), T + 60.0, _ring())[0]
        self.assertEqual(x["kind"], "ok")
        self.assertAlmostEqual(x["end"], T + 53.910, places=3)          # PP2 forward_ct 6 -> 7 first seen
        self.assertAlmostEqual(x["total_ms"], (53.910 - 44.096) * 1000, delta=1)
        self.assertGreater(x["total_ms"], 9000)                           # never the 2513 of the leg-1 dispatch
        self.assertIn("P.tp0pp2", x["end_src"])
        # the partition: vorlauf 221 + layer 2288 + wake-kv/dc ~0 + nachlauf to the previous rank clock + rest (clock)
        self.assertAlmostEqual(x["vorlauf_ms"], 221, delta=1)
        self.assertAlmostEqual(x["layer_ms"], 2288, delta=1)
        self.assertAlmostEqual(x["nachlauf_ms"], (52.845 - 46.605589) * 1000, delta=2)
        self.assertAlmostEqual(x["rest_ms"], (53.910 - 52.845) * 1000, delta=2)      # the rank clock, as in y7l
        self.assertAlmostEqual(x["end_res_ms"], x["rest_ms"], delta=0.01)
        self.assertAlmostEqual(_sum(x), x["total_ms"], delta=1e-6)

    def test_dp_front_pp_last_forward_is_exact_and_preferred(self):
        x = ipcboot.flip_views(SEGS, _ipc_dp("pp_last_forward", 53.23), T + 60.0, _ring())[0]
        self.assertAlmostEqual(x["end"], T + 53.23, places=3)
        self.assertEqual(x["rest_ms"], 0.0)
        self.assertAlmostEqual(_sum(x), x["total_ms"], delta=1e-6)

    def test_dp_without_last_stage_counter_is_missing_not_small(self):
        ring = [{"t": s["t"], "r": {k: v for k, v in s["r"].items() if k.startswith("D")}} for s in _ring()]
        x = ipcboot.flip_views(SEGS, _ipc_dp(), T + 60.0, ring)[0]
        self.assertEqual(x["kind"], "fehlt")
        self.assertIsNone(x["total_ms"])
        self.assertIn("forward_ct", x["missing"])
        fl = ipcboot.flip_last([x])["D>P"]
        self.assertEqual((fl["n"], fl["newest"]["kind"]), (0, "fehlt"))
        ft = ipcboot.flip_times_of([x])
        self.assertIsNone(ft["D>P"]["last"])
        self.assertIn("forward_ct", ft["D>P"]["missing"])


def _ipc_pd(first_work=11.063):
    ev = [_ev("flip_begin", T + 8.465, flip_begin_ts=T + 8.461, sleep="P", wake="D", epoch_before=1),
          _ev("flip_done", T + 12.96, flip_begin_ts=T + 8.461, t=T + 12.9572518, flip_ms=2427, epoch=2, sleep="P", wake="D")]
    fw = [{"dir": "P>D", "flip_begin_ts": T + 8.461, "first_work_ts": T + first_work, "flip_time_ms": 2601,
           "flip_user_ms": 2602, "p_end_ts": T + 8.46, "p_end_source": "p_leg1_end", "what": "decode_token", "epoch": 2}]
    return {"ipc_events": ev, "flip_first_work": fw, "flip_user_time": []}


def _ring_pd():
    out = []
    for i in range(10):
        ts = T + 6.893 + i
        out.append({"t": ts + 0.6, "r": {"D.tp0pp0": {"ts": ts, "dtok": 1.0 + 17 * (ts > T + 12.5), "rounds": 1.0 + (ts > T + 11.5) + 4 * (ts > T + 12.5),
                                                      "pnew": 1.0 + 4 * (ts > T + 11.5)},
                                         "P.tp0pp2": {"ts": ts, "fwd": 6.0}}})
    return out


class FlipzeitPD(unittest.TestCase):
    def test_pd_first_token_during_wake_kv_dc_counts(self):
        x = ipcboot.flip_views(SEGS, _ipc_pd(), T + 60.0, _ring_pd())[0]
        self.assertEqual(x["kind"], "ok")
        self.assertAlmostEqual(x["total_ms"], 2603, delta=1)              # not 4496 (to flip_done)
        self.assertAlmostEqual(x["layer_ms"], 2427, delta=1)
        self.assertAlmostEqual(x["wake_kv_dc_ms"], 175, delta=1)          # clipped at the first token
        self.assertEqual(x["nachlauf_ms"], 0.0)
        self.assertAlmostEqual(_sum(x), x["total_ms"], delta=1e-6)

    def test_pd_early_fire_before_layers_falls_back_to_d_rank_counters(self):
        # the NF y6d class: a "decode_token" 0,1 s after flip_begin -- D's layers are not back yet
        x = ipcboot.flip_views(SEGS, _ipc_pd(first_work=8.56), T + 60.0, _ring_pd())[0]
        self.assertIn("rankstats D.tp0pp0", x["end_src"])
        self.assertAlmostEqual(x["end"], T + 11.893, places=3)            # rounds/pnew 1 -> 2/5 first seen
        self.assertGreater(x["rest_ms"], 0)
        self.assertAlmostEqual(_sum(x), x["total_ms"], delta=1e-6)


class FlipzeitPush(unittest.TestCase):
    def test_vm_gets_only_the_total_and_its_parts(self):
        x = ipcboot.flip_views(SEGS, _ipc_dp(), T + 60.0, _ring())[0]
        lines = vmpush.flip_view_points([x], "NF", "068d", set())
        parts = sorted(l.split('part="')[1].split('"')[0] for l in lines)
        self.assertEqual(parts, ["layer", "nachlauf", "rest", "total", "vorlauf", "wake_kv_dc"])
        self.assertTrue(all('def="t2t"' in l for l in lines))
        # the front's own small numbers are no longer pushed as weg2_flip_time_ms / weg2_flip_user_ms
        pts, newest = vmpush.flip_points(_ipc_dp(), "NF", 0.0)
        self.assertEqual(pts, [])
        self.assertGreater(newest, 0.0)

    def test_vm_flip_stats(self):
        st = vmpush.flip_stats_from({"D>P": [(1, 9814.0), (2, 8523.0), (3, 10191.0)]})["D>P"]
        self.assertEqual((st["n"], st["median"], st["max"]), (3, 9814.0, 10191.0))


class FlipzeitHistory(unittest.TestCase):
    def test_history_tiles_read_only_the_t2t_marks(self):
        db = history.HistoryDB(None)
        db.mark(T + 1, "NF", "flip_user", "D>P ipc", 2513.0)             # old: up to the leg-1 dispatch
        db.mark(T + 2, "NF", "flip", "P>D ipc", 2601.0)                  # old: the front's number
        db.mark(T + 3, "NF", "flip_t2t", "D>P v=221 l=2288 w=1 n=6305 r=1000 ipc", 9814.0)
        db.mark(T + 4, "NF", "flip_t2t", "P>D v=1 l=2427 w=175 n=0 r=0 ipc", 2603.0)
        v = history.view(db, None, "NF", "1h", lo_hi=(T, T + 10))
        fl = v["tiles"]["flip"]
        self.assertEqual((fl["D>P"]["last_ms"], fl["D>P"]["n"]), (9814.0, 1))
        self.assertEqual((fl["P>D"]["last_ms"], fl["P>D"]["n"]), (2603.0, 1))
        self.assertEqual(v["tiles"]["flip_last_ms"], 2603.0)
        vals = [(m["kind"], m["v"]) for m in v["marks"] if m["kind"].startswith("flip")]
        self.assertNotIn(("flip_user", 2513.0), vals)
        self.assertNotIn(("flip", 2601.0), vals)


if __name__ == "__main__":
    unittest.main()
