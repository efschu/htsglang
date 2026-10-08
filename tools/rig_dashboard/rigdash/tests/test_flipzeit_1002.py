"""Nutzer 02.10.: FLIPZEIT = vom LETZTEN Token der abgebenden Phase bis zum ERSTEN Token der annehmenden, beide
Richtungen, beide Modelle -- die eine Zahl; die Teile (Vorlauf, Layer, Wake-KV/DC, Nachlauf, Rest) summieren zu
ihr.  Zahlen aus NF y7l (boot ...120607Z-068d, Flip 2 P>D 12:10:08 und Flip 3 D>P 12:10:44, events.jsonl und
der rigdash-Ring als Vorlage):

* D>P endete bisher am Leg-1-Dispatch (2513 ms).  Ende = erster Prefill-Forward auf P (PP0, Nutzer 13:05Z
  "... prefill batch beginn"); der erste Forward auf PP2 (12:10:53) liegt ~6 s spaeter -- Pipeline-Fuellung,
  Prefill, keine Flipzeit.
* P>D: D gab sein erstes Token 1,9 s VOR flip_done (waehrend wake-kv/dc) -- 2603 ms, nicht 4496 ms.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from rigdash import flipzeit, history, ipcboot, vmpush  # noqa: E402

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
    ut = [{"dir": "D>P", "epoch": 3, "flip_user_ms": 2513, "idle_flip": False, "rid": "pdflip-1-1", "start_ts": T + 44.096,
           "start_source": "park_rpc_sent", "prefill_start_ts": T + prefill_ts, "prefill_start_source": prefill_source,
           "parts": {"first_chunk_ms": 3, "legs_ms": 2289, "park_rpc_ms": 219, "pre_begin_ms": 221}}]
    fw = [{"dir": "D>P", "flip_begin_ts": T + 44.317, "first_work_ts": T + 46.608, "flip_time_ms": 2291,
           "what": "p_leg1_dispatch", "epoch": 3}]
    return {"ipc_events": ev, "flip_user_time": ut, "flip_first_work": fw}


SEGS = [{"s": T + 0.0, "e": T + 60.0, "k": "unknown"}]
#: the waiter arrived before D's last token (Nutzer 06.10.: start = max(last D token, arrival))
ARR = {"pdflip-1-1": T + 44.0}
#: D's TP0 rounds (open, open + gpu-ms): the last one before the D>P flip ends at 44,096
D_ROUNDS = [(T + 43.9, T + 43.93), (T + 44.07, T + 44.096)]


def _sum(x):
    return sum(x[k] for k in ipcboot.PARTS if x.get(k) is not None)


class FlipzeitDP(unittest.TestCase):
    def test_dp_ends_at_first_prefill_forward_on_pp0_not_dispatch_not_pp_last(self):
        """Nutzer 02.10. ~13:05Z ("... prefill batch beginn"): D>P ends at the first forward on P's FIRST stage.
        y7l 12:10:44: PP0 forward_ct 6 -> 7 first seen 47,74 (ADMIT PP0 12:10:47); PP2 only at 53,91 -- the
        ~6 s between are PP0 + PP1 computing chunk 1 (pipeline fill = prefill, not flip)."""
        x = ipcboot.flip_views(SEGS, _ipc_dp(), T + 60.0, _ring(), d_rounds=D_ROUNDS, arrivals=ARR)[0]
        self.assertEqual(x["kind"], "ok")
        self.assertAlmostEqual(x["end"], T + 47.74, places=3)            # PP0 forward_ct 6 -> 7 first seen
        self.assertAlmostEqual(x["total_ms"], (47.74 - 44.096) * 1000, delta=1)
        self.assertIn("P.tp0pp0", x["end_src"])
        self.assertNotAlmostEqual(x["total_ms"], 2513, delta=50)          # never the leg-1 dispatch
        self.assertAlmostEqual(x["warmup_ms"], 221, delta=1)
        self.assertAlmostEqual(x["layer_ms"], 2288, delta=1)
        self.assertAlmostEqual(x["nachlauf_ms"], (46.74 - 46.605589) * 1000, delta=2)
        self.assertAlmostEqual(x["rest_ms"], 1000.0, delta=2)             # the rank clock (lo 46,74 -> seen 47,74)
        self.assertAlmostEqual(x["end_res_ms"], x["rest_ms"], delta=0.01)
        self.assertAlmostEqual(_sum(x), x["total_ms"], delta=1e-6)

    def test_dp_front_pp_first_forward_is_exact_and_preferred(self):
        ipc = _ipc_dp("pp_first_forward", 47.21)
        ipc["flip_user_time"][0]["pp_last_start_ts"] = T + 53.23
        x = ipcboot.flip_views(SEGS, ipc, T + 60.0, _ring(), d_rounds=D_ROUNDS, arrivals=ARR)[0]
        self.assertAlmostEqual(x["end"], T + 47.21, places=3)
        self.assertEqual(x["rest_ms"], 0.0)
        self.assertAlmostEqual(x["pp_last_start"], T + 53.23, places=3)   # named, not part of the flip
        self.assertAlmostEqual(_sum(x), x["total_ms"], delta=1e-6)
        # a stamp at the leg-1 dispatch is not the end
        y = ipcboot.flip_views(SEGS, _ipc_dp("leg1_dispatch", 46.608), T + 60.0, _ring(), d_rounds=D_ROUNDS, arrivals=ARR)[0]
        self.assertAlmostEqual(y["end"], T + 47.74, places=3)

    def test_phase_bar_draws_pipeline_fill_as_prefill(self):
        from rigdash import activity
        ipc = _ipc_dp()
        fd = [dict(e["data"]) for e in ipc["ipc_events"] if e["type"] == "flip_done"]
        m = activity.Model(_ring(), fd, ipc["flip_first_work"], None, ipc["flip_user_time"])
        tail = [t for t in m.tails() if t[2] == "D>P"]
        self.assertEqual(len(tail), 1)
        self.assertAlmostEqual(tail[0][1], T + 47.74, places=3)           # Nachlauf ends at PP0's first forward

    def test_dp_without_pp0_counter_is_missing_not_small(self):
        ring = [{"t": s["t"], "r": {k: v for k, v in s["r"].items() if k.startswith("D")}} for s in _ring()]
        # (no P rank in the ring: P's first stage cannot be read)
        x = ipcboot.flip_views(SEGS, _ipc_dp(), T + 60.0, ring, d_rounds=D_ROUNDS)[0]
        self.assertEqual(x["kind"], "fehlt")
        self.assertIsNone(x["total_ms"])
        self.assertIn("forward_ct", x["missing"])
        self.assertEqual(flipzeit.stats(flipzeit.from_views([x]), "D>P")["n"], 0)        # a missing end is never a point
        dg = ipcboot.flip_diag([x])["D>P"]
        self.assertEqual(dg["missing_n"], 1)
        self.assertIn("forward_ct", dg["missing"])


def _ipc_pd(first_work=11.063, what="decode_token"):
    ev = [_ev("flip_begin", T + 8.465, flip_begin_ts=T + 8.461, sleep="P", wake="D", epoch_before=1),
          _ev("flip_done", T + 12.96, flip_begin_ts=T + 8.461, t=T + 12.9572518, flip_ms=2427, epoch=2, sleep="P", wake="D")]
    fw = [{"dir": "P>D", "flip_begin_ts": T + 8.461, "first_work_ts": T + first_work, "flip_time_ms": 2601,
           "flip_user_ms": 2602, "p_end_ts": T + 8.46, "p_end_source": "p_leg1_end", "what": what, "epoch": 2}]
    return {"ipc_events": ev, "flip_first_work": fw, "flip_user_time": []}


SEGS_PD = [{"s": T + 0.0, "e": T + 8.46, "k": "P"}, {"s": T + 8.46, "e": T + 60.0, "k": "unknown"}]


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
        x = ipcboot.flip_views(SEGS_PD, _ipc_pd(), T + 60.0, _ring_pd(), d_rounds=None)[0]
        self.assertEqual(x["kind"], "ok")
        self.assertAlmostEqual(x["total_ms"], 2603, delta=1)              # not 4496 (to flip_done)
        self.assertAlmostEqual(x["layer_ms"], 2427, delta=1)
        self.assertAlmostEqual(x["wake_kv_dc_ms"], 175, delta=1)          # clipped at the first token
        self.assertEqual(x["nachlauf_ms"], 0.0)
        self.assertAlmostEqual(_sum(x), x["total_ms"], delta=1e-6)

    def test_pd_non_streaming_d_first_forward_is_the_exact_end(self):
        # 27B PDFLIP-E3 a0d03e9321: a non-streaming request stamps D's first forward from its beacon
        for what in ("d_first_forward_done", "d_first_forward_done_approx"):
            x = ipcboot.flip_views(SEGS_PD, _ipc_pd(what=what), T + 60.0, _ring_pd(), d_rounds=None)[0]
            self.assertEqual(x["kind"], "ok")
            self.assertIn(what, x["end_src"])
            self.assertAlmostEqual(x["total_ms"], 2603, delta=1)
            self.assertEqual(x["rest_ms"], 0.0)

    def test_pd_early_fire_before_layers_falls_back_to_d_rank_counters(self):
        # the NF y6d class: a "decode_token" 0,1 s after flip_begin -- D's layers are not back yet
        x = ipcboot.flip_views(SEGS_PD, _ipc_pd(first_work=8.56), T + 60.0, _ring_pd(), d_rounds=None)[0]
        self.assertIn("rankstats D.tp0pp0", x["end_src"])
        self.assertAlmostEqual(x["end"], T + 11.893, places=3)            # rounds/pnew 1 -> 2/5 first seen
        self.assertGreater(x["rest_ms"], 0)
        self.assertAlmostEqual(_sum(x), x["total_ms"], delta=1e-6)


class FlipzeitPush(unittest.TestCase):
    def test_vm_gets_only_the_total_and_its_parts(self):
        # D's log has a round of the next phase: the D>P start is final (a provisional one stays out of VM)
        x = ipcboot.flip_views(SEGS, _ipc_dp(), T + 60.0, _ring(), d_rounds=D_ROUNDS + [(T + 58.0, T + 58.03)],
                               arrivals=ARR)[0]
        lines = vmpush.flip_view_points([x], "NF", "068d", set())
        parts = sorted(l.split('part="')[1].split('"')[0] for l in lines)
        # with the waiter's arrival the Vorlauf split rides along (leer = 0: Server-Leerlauf is never in the total)
        self.assertEqual(parts, ["halt", "layer", "leer", "leer_d_prefill", "nachlauf", "park", "rest", "total",
                                 "vor_rest", "wake_kv_dc", "warmup"])    # sorted: F0-D rename vorlauf -> warmup moved the key behind wake_kv_dc
        self.assertTrue(all('def="t2t"' in l for l in lines))
        # the front's own small numbers are no longer pushed as pdflip_flip_time_ms / pdflip_flip_user_ms
        pts, newest = vmpush.flip_points(_ipc_dp(), "NF", 0.0)
        self.assertEqual(pts, [])
        self.assertGreater(newest, 0.0)


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
        self.assertNotIn("flip_last_ms", v["tiles"])                       # one tile payload: tiles["flip"]
        vals = [(m["kind"], m["v"]) for m in v["marks"] if m["kind"].startswith("flip")]
        self.assertNotIn(("flip_user", 2513.0), vals)
        self.assertNotIn(("flip", 2601.0), vals)


if __name__ == "__main__":
    unittest.main()
