"""Glatte, ehrliche Raten und Sitze (Nutzer 30.09. ~21:05Z / ~21:10Z).

Wortlaut: "der decode durchsatz wird noch falsch dargestellt, nicht durchgehend sondern extrem
sprunghaft - diese sprunghaftigkeit bei den übrigen dingen auch beachten" und "bei verlauf wo decode
je stream steht soll auch die anzahl oder das mittel der sitze".

Two causes, both measured on the live service 30.09. ~21:10Z (NF y5i e599):

1. The 1-s sampler slips (a poll takes > 1 s under /api/live load): ~28 % of the 1-s history rows
   held no sample and were written as gaps -- every curve broke into pieces every 3-4 s, although
   the work between two samples is placed by the rank clocks.
2. A 5-s bucket's rate was tokens / 5 s: a 2-s D-extend inside it halved the decode curve although D
   decoded at full speed the rest of the time (5-s rows 123, 127, 81, 27, 147 tok/s).  The rate WHILE
   decoding is tokens / the seconds D decoded (``busy``); pauses without any decode stay a gap.

Simulated D: rank writes every 1.0 s; 50 rounds/s of 20 ms each, 2 tokens per seat and round;
bs 4 in [10, 20), bs 2 in [20, 30); D-extends (2 s, 2000 tokens) at [14, 16) and [24, 26); idle after 30.
True: 400 tok/s at bs 4, 200 at bs 2, 100 per stream always.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rigdash import activity, history, ipcboot, server  # noqa: E402

PAUSES = ((14.0, 16.0), (24.0, 26.0))


def _dec_s(ts, lo, hi):
    """Decode seconds in [lo, min(ts, hi)) without the pauses."""
    e = min(ts, hi)
    if e <= lo:
        return 0.0
    tot = e - lo
    for a, b in PAUSES:
        tot -= max(0.0, min(e, b) - max(lo, a))
    return tot


def rec_d(ts):
    s4, s2 = _dec_s(ts, 10.0, 20.0), _dec_s(ts, 20.0, 30.0)
    ext = [(a, b) for a, b in PAUSES if b <= ts]
    last = ext[-1] if ext else None
    return {"schema": "pdflip.rankstats/1", "ts": ts,
            "prefill": {"chunks": len(ext), "new_tokens": 2000 * len(ext), "cached_tokens": 0,
                        "compute_ms": 2000.0 * len(ext),
                        "last": {"t": last[1], "gpu_ms": 2000.0, "new": 2000} if last else None},
            "decode": {"tokens": int(round(400 * s4 + 200 * s2)), "rounds": int(round(50 * (s4 + s2))),
                       "gpu_ms": 1000.0 * (s4 + s2),
                       "gpu_ms_by_bs": {"4": [int(round(50 * s4)), 1000.0 * s4], "2": [int(round(50 * s2)), 1000.0 * s2]},
                       "running": 4 if 10 <= ts < 20 else 2 if 20 <= ts < 30 else 0},
            "sched": {"full_token_usage": 0.5}}


def ring(t_end=40.0, slip=False, t0=0.0):
    out = []
    for k in range(int(t_end) + 1):
        if slip and k % 4 == 3:          # the service's slipped poll: this second has no sample
            continue
        ts = float(k)
        r = ipcboot.compact(rec_d(ts))
        r["ts"] += t0
        if r.get("plast_t") is not None:
            r["plast_t"] += t0
        out.append({"t": t0 + ts + 0.3, "r": {"D.tp0pp0": r}, "front": {"queue": 0, "outstanding": 0}})
    return out


class TestSlippedSampler(unittest.TestCase):
    """Cause 1: a second without its own sample is observed (between two samples <= 3 s apart)."""

    def test_no_gap_inside_a_watched_stretch(self):
        m = activity.Model(ring(slip=True), [], [])
        b = m.buckets(1.0, 38, 1.0)
        self.assertEqual([i + 1 for i, v in enumerate(b["dec_tps"]) if v is None], [])   # was every 4th second
        self.assertEqual([i + 1 for i, v in enumerate(b["kv_pct"]) if v is None], [])    # level: held
        # the values are the rank's, not the sampler's: 400 tok/s in every full bs-4 second
        for sec in (11, 12, 13, 16, 17, 18):
            self.assertAlmostEqual(b["dec_tps"][sec - 1], 400.0, delta=1.0, msg=sec)

    def test_a_stalled_sampler_loses_nothing(self):
        # Nutzer 30.09. ~21:40Z: the sampler stalls 6 s (no reading 20..27): the rank's counters bound the
        # interval by the RANK's clock, the Δ is spread over it -- no gap, every token once
        rg = [s for s in ring() if not (20 < s["t"] < 27)]
        b = activity.Model(rg, [], []).buckets(0.0, 40, 1.0)
        self.assertEqual([i for i in range(11, 30) if b["dec_tps"][i] is None], [])
        self.assertAlmostEqual(sum(v or 0 for v in b["dec_tps"]), 400 * 8 + 200 * 8, delta=2)
        self.assertEqual([i for i in range(11, 30) if b["kv_pct"][i] is None], [])
        self.assertTrue(any(b["held"][i] for i in range(21, 27)))      # the KV level there was held: counted

    def test_a_silent_rank_stays_a_gap(self):
        rg = [s for s in ring(80.0) if not (20 < s["t"] < 62)]          # the rank itself gave nothing for 42 s
        b = activity.Model(rg, [], []).buckets(0.0, 80, 1.0)
        self.assertIsNone(b["dec_tps"][40])


class TestRateWhileWorking(unittest.TestCase):
    """Cause 2: the curve drawn is tokens / decode time; the pause is a gap, not a dip."""

    def setUp(self):
        self.m = activity.Model(ring(), [], [])

    def test_five_second_rows_do_not_dip_at_an_extend(self):
        b = self.m.buckets(10.0, 4, 5.0)                        # [10,15) [15,20) [20,25) [25,30)
        self.assertAlmostEqual(b["dec_tps"][0], 320.0, delta=2)   # wall: 4 s of 400 in 5 s
        self.assertAlmostEqual(b["dec_rate"][0], 400.0, delta=2)  # while decoding
        self.assertAlmostEqual(b["dec_rate"][1], 400.0, delta=2)
        self.assertAlmostEqual(b["dec_rate"][2], 200.0, delta=2)
        self.assertAlmostEqual(b["dec_rate"][3], 200.0, delta=2)
        # busy share = decode seconds / bucket
        self.assertAlmostEqual(b["dec_busy"][0], 0.8, delta=0.01)

    def test_pause_is_a_gap_and_idle_is_no_rate(self):
        b = self.m.buckets(0.0, 40, 1.0)
        self.assertIsNone(b["dec_rate"][14])                     # D-extend second: no decode, no rate
        self.assertEqual(b["dec_tps"][14], 0.0)                  # the wall curve says 0 there
        self.assertIsNone(b["dec_rate"][35])                     # idle after 30 s
        self.assertAlmostEqual(b["d_rate"][14], 1000.0, delta=5)  # the extend itself: 2000 tok / 2 s

    def test_every_token_once(self):
        b = self.m.buckets(0.0, 40, 1.0)
        self.assertAlmostEqual(sum(v or 0 for v in b["dec_tps"]), 400 * 8 + 200 * 8, delta=2)


class TestSeats(unittest.TestCase):
    """Nachtrag 21:10Z: seats beside per stream, time-weighted over the decode time only."""

    def setUp(self):
        self.m = activity.Model(ring(), [], [])

    def test_seats_from_the_rounds(self):
        b = self.m.buckets(10.0, 2, 10.0)                        # [10,20) bs 4, [20,30) bs 2
        self.assertAlmostEqual(b["seats"][0], 4.0, places=3)
        self.assertAlmostEqual(b["seats"][1], 2.0, places=3)
        self.assertAlmostEqual(b["stream_tps"][0], 100.0, delta=0.5)
        self.assertAlmostEqual(b["stream_tps"][1], 100.0, delta=0.5)
        mixed = self.m.buckets(15.0, 1, 10.0)                    # [15,25): 4 s bs 4 + 4 s bs 2
        self.assertAlmostEqual(mixed["seats"][0], 3.0, delta=0.05)
        self.assertAlmostEqual(mixed["dec_rate"][0], 300.0, delta=2)
        self.assertAlmostEqual(mixed["stream_tps"][0], 100.0, delta=0.5)
        self.assertEqual((mixed["dec_bs_min"][0], mixed["dec_bs_max"][0]), (2.0, 4.0))
        # rate = per stream x seats, exactly (one denominator)
        self.assertAlmostEqual(mixed["stream_tps"][0] * mixed["seats"][0], mixed["dec_rate"][0], places=6)
        # sleep and pause are not "0 seats": over [10,40) (half of it idle) the mean stays 3
        whole = self.m.buckets(10.0, 1, 30.0)
        self.assertAlmostEqual(whole["seats"][0], 3.0, delta=0.05)

    def test_live_tile(self):
        dv = ipcboot.decode_view(self.m, "D", {}, 29.5)
        self.assertAlmostEqual(dv["per_stream"], 100.0, delta=1)
        self.assertLess(abs(dv["seats_mean"] - 3.0), 0.3)
        self.assertEqual((dv["seats_min"], dv["seats_max"]), (2.0, 4.0))
        self.assertIn("gpu_ms_by_bs", dv["seats_src"])
        self.assertAlmostEqual(dv["one_s"], 200.0, delta=2)
        ser = ipcboot.series_view(self.m, 40.0)
        self.assertIn("D_decode_rate", ser)
        self.assertIn("D_decode_seats", ser)

    def test_zoomed_live_series_is_finer(self):
        ser = ipcboot.series_view(self.m, 40.0, zoom=(10.0, 30.0))
        self.assertEqual(ser["step"], 1.0)
        self.assertEqual(len(ser["t"]), 20)
        self.assertAlmostEqual(ser["D_decode_rate"][2], 400.0, delta=2)
        self.assertEqual(server.parse_zoom("/api/live?zoom=10,30"), (10.0, 30.0))
        self.assertIsNone(server.parse_zoom("/api/live?zoom=x"))


class TestHistoryHonestTiers(unittest.TestCase):
    """The shares are stored per 1-s row, the division happens per shown bucket: right at every tier;
    the batch-size span folds by MIN / MAX; a zoomed stretch is re-read at its own step."""

    T0 = 1790000000.0

    def _db(self):
        from collections import deque
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "nf", "state")
            d = os.path.join(root, "nfx-boot-20260930T153426Z-051f")
            os.makedirs(d)
            import json
            with open(os.path.join(d, "state.json"), "w") as fh:
                json.dump({"schema": "pdflip.state/1", "boot_id": os.path.basename(d), "kind": "boot", "tag": "nfx",
                           "lifecycle": {"state": "serving"}, "front": {"awake": "D"}}, fh)
            ib = ipcboot.IpcBoots(roots=(root,))
            ib.poll(self.T0 + 0.1)
            ib.rings[os.path.basename(d)] = deque(ring(40.0, slip=True, t0=self.T0))
            db = history.HistoryDB(None)
            rec = history.Recorder(db, ib)
            rec.ingest_ipc(self.T0 + 200.0)
        return db

    def test_rows_and_derived_rates(self):
        db = self._db()
        one = history.view(db, None, "NF", "15m", now=self.T0 + 200.0, lo_hi=(self.T0, self.T0 + 40))
        self.assertEqual(one["step"], 1)
        self.assertEqual(one["zoom"], [self.T0, self.T0 + 40])
        s, t = one["series"], one["t"]
        i = t.index(int(self.T0) + 12)
        self.assertAlmostEqual(s["m.dec_rate"][i], 400.0, delta=2)
        self.assertAlmostEqual(s["m.seats"][i], 4.0, delta=0.01)
        gaps = [t[j] - int(self.T0) for j in range(len(t)) if 11 <= t[j] - self.T0 <= 29 and s["m.ipc"][j] is None]
        self.assertEqual(gaps, [])                                   # slipped samples: no gaps
        # compact to the 10-s tier and read a range that uses it
        db.compact(self.T0 + 400.0)
        ten = history.view(db, None, "NF", "4h", now=self.T0 + 400.0, lo_hi=(self.T0 - 14000, self.T0 + 400))
        self.assertEqual(ten["step"], 10)
        s, t = ten["series"], ten["t"]
        i = t.index(int(self.T0) + 10)
        self.assertAlmostEqual(s["m.dec_rate"][i], 400.0, delta=3)    # [10,20): 8 s decode, pause cut out
        self.assertAlmostEqual(s["m.dec_tps"][i], 320.0, delta=3)     # the wall rate keeps its meaning
        j = t.index(int(self.T0) + 20)
        self.assertAlmostEqual(s["m.seats"][j], 2.0, delta=0.01)
        self.assertAlmostEqual(s["m.stream_tps"][j], 100.0, delta=1)
        self.assertEqual((s["m.seats_min"][i], s["m.seats_max"][i]), (4.0, 4.0))
        tl = ten["tiles"]
        self.assertAlmostEqual(tl["seats_mean"], 3.0, delta=0.05)   # time-weighted, idle not counted
        self.assertEqual((tl["seats_min"], tl["seats_max"]), (2.0, 4.0))
        self.assertAlmostEqual(tl["stream_mean"], 100.0, delta=1)

    def test_min_max_fold(self):
        db = history.HistoryDB(None)
        db.put([("mi.NF.dec_bs_min", 100, 3.0), ("mi.NF.dec_bs_min", 101, 1.0), ("mi.NF.dec_bs_max", 100, 3.0),
                ("mi.NF.dec_bs_max", 101, 6.0), ("mi.NF.dec_tps", 100, 10.0), ("mi.NF.dec_tps", 101, 30.0)])
        got = db.query(["mi.NF.dec_bs_min", "mi.NF.dec_bs_max", "mi.NF.dec_tps"], 100, 110, 5, now=150)   # from p0
        self.assertEqual(got["mi.NF.dec_bs_min"][100], 1.0)
        self.assertEqual(got["mi.NF.dec_bs_max"][100], 6.0)
        self.assertEqual(got["mi.NF.dec_tps"][100], 20.0)
        db.compact(10000.0)
        with db.lock:
            rows = dict(db.db.execute("SELECT s.name, p.v FROM p1 p JOIN series s ON s.id = p.sid").fetchall())
        self.assertEqual((rows["mi.NF.dec_bs_min"], rows["mi.NF.dec_bs_max"], rows["mi.NF.dec_tps"]), (1.0, 6.0, 20.0))

    def test_tier_tokens_at_the_one_second_raster(self):
        db = history.HistoryDB(None)
        rec = history.Recorder(db, None)
        def st(dev):
            return {"front": {"served_tokens": {"P": {"n": 1, "prompt": 1000 + dev, "cached": dev, "cached_tier": {"device": dev}},
                                                "D": {"n": 0, "prompt": 0, "cached": 0},
                                                "D_after_P": {"n": 0, "prompt": 0, "cached": 0}}}}
        rec._tiers("k", "NF", st(0), 1000)
        rec._tiers("k", "NF", st(500), 1005)
        self.assertEqual(db.get("src.NF.tiers"), "ipc")
        v = history.view(db, None, "NF", "15m", now=1100.0)
        self.assertEqual(v["step"], 1)
        self.assertAlmostEqual(v["tiles"]["tiers"]["device"], 500.0, delta=1)   # was 100 (1/5)


class TestCounterBooking(unittest.TestCase):
    """Nutzer 30.09. ~21:40Z: "der probenehmer sollte doch nicht an zu viel last scheitern" -- a counter's
    Δ between two readings goes onto the seconds they cover; a late reading loses nothing."""

    def test_spread_and_complete_seconds(self):
        acc = {}
        history.spread_counter(acc, 10.5, 13.5, 300.0)            # 100 per second
        self.assertEqual(history.pop_complete(acc, 13.5), {10: 50.0, 11: 100.0, 12: 100.0})
        self.assertEqual(acc, {13: 50.0})                        # 13 is not complete yet
        history.spread_counter(acc, 13.5, 14.0, 50.0)
        self.assertEqual(history.pop_complete(acc, 14.0), {13: 100.0})

    def test_power_from_energy_with_a_late_reading(self):
        rec = history.Recorder(history.HistoryDB(None), None)
        rows = []
        # 250 W constant; readings at 100.2, 101.2, then the loop stalls 4 s, 105.3, 106.2
        for t in (100.2, 101.2, 105.3, 106.2):
            rows += rec.power_from_energy(1, t, 250.0 * 1000.0 * t)
        got = {ts: v for _, ts, v in rows}
        self.assertEqual(sorted(got), [101, 102, 103, 104, 105])   # no second lost, 100 only half covered
        for v in got.values():
            self.assertAlmostEqual(v, 250.0, places=6)
        self.assertEqual(rec.held["total"], 0)

    def test_levels_filled_and_counted(self):
        rec = history.Recorder(history.HistoryDB(None), None)
        rec.level_rows(100, [("g0.temp", 50.0)])
        rows = rec.level_rows(103, [("g0.temp", 53.0)])
        self.assertEqual(sorted((ts, v) for _, ts, v in rows), [(101, 50.0), (102, 50.0), (103, 53.0)])
        self.assertEqual(rec.held["nvml"], 2)
        self.assertEqual(rec.held_view()["last_5min"], 2)


class TestViewHoldCount(unittest.TestCase):
    def test_only_interior_fills_count(self):
        arr = [1.0, None, None, 2.0, None]
        self.assertEqual(history._hold(arr, 2), 2)                # the gap 1..2 counts, the live edge not
        self.assertEqual(arr, [1.0, 1.0, 1.0, 2.0, 2.0])


class TestStaticZoom(unittest.TestCase):
    """The zoom is on every chart: the history charts (uPlot), the boot cards' curves and phase bar."""

    def test_page_wires_the_zoom(self):
        st = server.STATIC
        z = open(os.path.join(st, "zoom.js"), encoding="utf-8").read()
        g = open(os.path.join(st, "grafik.js"), encoding="utf-8").read()
        h = open(os.path.join(st, "index.html"), encoding="utf-8").read()
        for w in ("pointerdown", "dblclick", "Escape", "Zoom zurück"):
            self.assertIn(w, z, w)
        self.assertIn("RigZoom", g)
        self.assertIn("from=", g)
        self.assertIn('src="zoom.js"', h)
        self.assertIn("zoom=", h)
        self.assertIn('data-zoom', h)
        self.assertIn("/zoom.js", server.STATIC_FILES)


if __name__ == "__main__":
    unittest.main()
