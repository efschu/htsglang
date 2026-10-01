"""Plausibility of the IPC instrument (Nutzer 30.09. ~16:45Z: "die ganze zeit 16k/s prefill").

A simulated NF-like boot, sampled like the real thing (every rank writes cumulative counters once a
second; a prefill chunk's tokens land in ``prefill.new_tokens`` only when it FINISHES):

  P, PP3, four 16k chunks back to back: PP0 3.0 s, PP1 2.5 s, PP2 1.0 s per chunk (pipelined),
  flip 26.0 -> 28.0 s, D re-extend of 1 token, then decode 28.5 -> 48.5 s, 2 streams x 100 tok/s.

True values: P burst = 65536 tok / (25.5 - 10.0) s = 4228 tok/s; per stream 100 tok/s; decode 200.
The old instrument (Δcounter / Δsample clock) shows 16384 tok/s in a 1-s sample and P and D busy at
the same time; these tests pin the new one to the true values.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rigdash import activity, ipcboot  # noqa: E402

CH = 16384
PP_MS = (3.0, 2.5, 1.0)


def p_chunks():
    """[(stage, k, start, end)] of the pipelined P burst."""
    out, prev_end = [], {}
    for k in range(4):
        t = 10.0 + 3.0 * k
        for st, dur in enumerate(PP_MS):
            s = max(t, prev_end.get(st, 0.0))
            e = s + dur
            out.append((st, k, s, e))
            prev_end[st] = e
            t = e
    return out


def rec_p(stage, ts, ch):
    done = [c for c in ch if c[0] == stage and c[3] <= ts]
    last = max(done, key=lambda c: c[3]) if done else None
    return {"schema": "weg2.rankstats/1", "ts": ts,
            "prefill": {"chunks": len(done), "new_tokens": CH * len(done), "cached_tokens": 0,
                        "compute_ms": 1000.0 * sum(c[3] - c[2] for c in done),
                        "last": {"t": last[3], "gpu_ms": 1000.0 * (last[3] - last[2]), "new": CH} if last else None},
            "decode": {"tokens": 0, "rounds": 0, "gpu_ms": 0.0, "running": None},
            "sched": {"full_token_usage": 0.25}}


def rec_d(ts):
    ext = ts >= 28.4
    dec = max(0.0, min(ts, 48.5) - 28.5)
    return {"schema": "weg2.rankstats/1", "ts": ts,
            "prefill": {"chunks": 1 if ext else 0, "new_tokens": 1 if ext else 0, "cached_tokens": 65535 if ext else 0,
                        "compute_ms": 200.0 if ext else 0.0,
                        "last": {"t": 28.4, "gpu_ms": 200.0, "new": 1} if ext else None},
            "decode": {"tokens": int(200 * dec), "rounds": int(50 * dec), "gpu_ms": 1000.0 * dec * 0.9,
                       "running": 2 if 28.5 <= ts <= 48.5 else 0},
            "sched": {"full_token_usage": 0.6 if ts >= 28.2 else 0.1}}


def ring_until(t_end):
    ch = p_chunks()
    ring = []
    t = 0.3
    while t <= t_end:
        ts = t - 0.1
        r = {"P.tp0pp%d" % st: ipcboot.compact(rec_p(st, ts, ch)) for st in range(3)}
        r["D.tp0pp0"] = ipcboot.compact(rec_d(ts))
        ring.append({"t": t, "r": r, "front": {"awake": "P" if t < 27 else "D"}})
        t += 1.0
    return ring


FLIP_DONE = [{"sleep": "P", "wake": "D", "flip_begin_ts": 26.0, "t": 28.0, "flip_ms": 2000}]
FIRST_WORK = [{"dir": "P>D", "flip_begin_ts": 26.0, "flip_time_ms": 2500, "what": "decode_token"}]


class TestInstrument(unittest.TestCase):
    def setUp(self):
        self.ring = ring_until(60.0)
        self.m = activity.Model(self.ring, FLIP_DONE, FIRST_WORK)

    def test_chunk_intervals_and_burst_rate(self):
        cs = self.m.pchunks["P"]
        self.assertEqual(len(cs), 4)
        self.assertAlmostEqual(cs[0]["s"], 10.0, places=3)                   # PP0 start of chunk 0
        self.assertAlmostEqual(cs[-1]["e"], 25.5, places=3)                  # PP2 end of chunk 3
        b = activity.bursts(cs)
        self.assertEqual(len(b), 1)
        self.assertAlmostEqual(b[0]["rate"], 4 * CH / 15.5, delta=1.0)      # 4228 tok/s, not 16384

    def test_curve_never_shows_a_whole_chunk_in_one_second(self):
        b = self.m.buckets(0.0, 60, 1.0)
        self.assertLess(max(v or 0 for v in b["p_tps"]), 6000)
        self.assertAlmostEqual(sum(v or 0 for v in b["p_tps"]), 4 * CH, delta=2)   # every token once
        five = self.m.buckets(0.0, 12, 5.0)
        self.assertLess(max(v or 0 for v in five["p_tps"]), 6000)                  # was 6550 / 9814

    def test_phases_do_not_overlap(self):
        self.assertEqual(self.m.overlap_s()["p_vs_d_s"], 0.0)
        segs = self.m.segments()
        for a, b in zip(segs, segs[1:]):
            self.assertLessEqual(a["e"], b["s"] + 1e-9)
        kinds = [s["k"] for s in segs]
        self.assertLess(kinds.index("P"), kinds.index("flip_pd"))
        self.assertLess(kinds.index("flip_pd"), kinds.index("dec"))
        b = self.m.buckets(0.0, 60, 1.0)
        both = [i for i in range(60) if (b["p_tps"][i] or 0) > 0 and ((b["dec_tps"][i] or 0) > 0 or (b["d_tps"][i] or 0) > 0)]
        self.assertEqual(both, [])

    def test_decode_per_stream(self):
        st = [x["stream"] for x in self.m.dec if x["stream"] is not None]
        self.assertTrue(st)
        for v in st:
            self.assertAlmostEqual(v, 100.0, delta=0.5)
        self.assertTrue(all(x["tok"] <= 200 * 1.01 * (x["e"] - x["s"]) + 1 for x in self.m.dec))

    def test_views(self):
        m = activity.Model(ring_until(20.0), [], [])
        pv = ipcboot.prefill_view(m, "P", 20.3)
        self.assertLess(pv["one_s"], 6000)            # running burst, not a chunk per sample
        self.assertGreater(pv["one_s"], 3000)
        full = ipcboot.prefill_view(self.m, "P", 60.3)
        self.assertEqual(full["one_s"], 0.0)           # nothing running
        self.assertAlmostEqual(full["last_burst"]["wall_tps"], 4 * CH / 15.5, delta=1.0)
        self.assertAlmostEqual(full["last_burst"]["tps_gpu"], CH / 3.0, delta=1.0)   # slowest rank PP0
        dv = ipcboot.prefill_view(self.m, "D", 60.3)
        self.assertIsNone(dv["last_burst"]["wall_tps"])   # 1-token re-extend: no rate, not "1 tok/s"
        dec = ipcboot.decode_view(self.m, "D", {}, 45.3)
        self.assertAlmostEqual(dec["gen_tps"], 200.0, delta=2)
        self.assertAlmostEqual(dec["per_stream"], 100.0, delta=1)
        self.assertLess(abs((dec["gen_tps_last"] or 0) - 200.0), 5)

    def test_timeline_flip_tail(self):
        segs = self.m.segments()
        tail = [s for s in segs if s["k"] == "flip_tail"]
        self.assertEqual(len(tail), 1)
        self.assertAlmostEqual(tail[0]["s"], 28.0, places=3)
        self.assertAlmostEqual(tail[0]["e"], 28.5, places=3)


class TestPhaseStates(unittest.TestCase):
    """Nutzer 30.09. ~18Z: "idle sieht aus wie flip oder flip nachlauf" -- every state from data.
    Leerlauf -> Flip P>D -> Nachlauf -> Decode -> Leerlauf gives exactly these segments."""

    def ring(self, queue=0, gap=None):
        out = []
        for i in range(41):
            t = float(i)
            if gap and gap[0] < t < gap[1]:
                continue
            dec = max(0.0, min(t, 25.0) - 13.0)
            out.append({"t": t, "front": {"queue": queue, "outstanding": {"P": 0, "D": 0}},
                        "r": {"D.tp0pp0": ipcboot.compact({"ts": t, "decode": {"tokens": int(100 * dec), "running": 1},
                                                            "prefill": {"chunks": 0, "new_tokens": 0},
                                                            "work": {"forward_ct": int(20 * dec)}})}})
        return out

    FD = [{"sleep": "P", "wake": "D", "flip_begin_ts": 10.0, "t": 12.0, "flip_ms": 2000}]
    FW = [{"dir": "P>D", "flip_begin_ts": 10.0, "flip_time_ms": 3000, "what": "decode_token", "first_work_ts": 13.0}]

    def test_sequence(self):
        m = activity.Model(self.ring(), self.FD, self.FW, life={"serving_since": -1.0})
        segs = [(x["k"], round(x["s"], 1), round(x["e"], 1)) for x in m.segments(40.0)]
        self.assertEqual([k for k, _, _ in segs], ["idle", "flip_pd", "flip_tail", "dec", "idle"])
        self.assertEqual(segs[1][1:], (10.0, 12.0))
        self.assertEqual(segs[2][1:], (12.0, 13.0))
        self.assertAlmostEqual(segs[3][2], 26.0, delta=1.0)

    def test_idle_needs_proof(self):
        # work queued but no rank moving: not idle, "unknown" with the reason
        m = activity.Model(self.ring(queue=2), [], [], life={"serving_since": -1.0})
        segs = m.segments(40.0)
        self.assertNotIn("idle", [x["k"] for x in segs])
        self.assertIn("queue/outstanding", [x for x in segs if x["k"] == "unknown"][0]["why"])
        # no samples at all: unknown, never flip or idle
        m = activity.Model(self.ring(gap=(30, 38)), self.FD, self.FW, life={"serving_since": -1.0})
        unk = [x for x in m.segments(40.0) if x["k"] == "unknown"]
        self.assertTrue(unk and unk[-1]["s"] <= 30.0 and unk[-1]["e"] >= 38.0)

    def test_dp_tail_runs_to_the_first_p_chunk(self):
        # D>P: the first-work stamp is only the leg-1 dispatch; the tail ends where P's first chunk starts
        ring = []
        for i in range(21):
            t = float(i)
            done = 1 if t >= 12.0 else 0
            ring.append({"t": t, "front": {"queue": 0, "outstanding": 0},
                         "r": {"P.tp0pp0": ipcboot.compact({"ts": t, "prefill": {"chunks": done, "new_tokens": 16384 * done,
                                                                                  "compute_ms": 2000.0 * done,
                                                                                  "last": {"t": 12.0, "gpu_ms": 2000.0, "new": 16384} if done else None}})}})
        fd = [{"sleep": "D", "wake": "P", "flip_begin_ts": 4.0, "t": 6.0}]
        fw = [{"dir": "D>P", "flip_begin_ts": 4.0, "first_work_ts": 6.1, "flip_time_ms": 2100, "what": "p_leg1_dispatch"}]
        m = activity.Model(ring, fd, fw, life={"serving_since": -1.0})
        segs = [(x["k"], round(x["s"], 1), round(x["e"], 1)) for x in m.segments(20.0)]
        self.assertIn(("flip_tail", 6.0, 10.0), segs)             # 10.0 = 12.0 - 2.0 s first chunk
        self.assertIn(("flip_dp", 4.0, 6.0), segs)

    def test_open_flip_and_off(self):
        begins = [{"flip_begin_ts": 30.0, "sleep": "D", "wake": "P"}]
        m = activity.Model(self.ring(), self.FD, self.FW, begins=begins,
                           life={"serving_since": 5.0, "terminal_since": None})
        segs = m.segments(40.0)
        self.assertEqual(segs[0]["k"], "off")                                  # before serving: lädt
        self.assertEqual(segs[-1]["k"], "flip_dp")                              # open D>P flip up to now
        self.assertEqual((segs[-1]["s"], segs[-1]["e"]), (30.0, 40.0))

    def test_legend_same_in_both_editions(self):
        from rigdash import server
        html = open(os.path.join(server.STATIC, "index.html"), encoding="utf-8").read()
        rel = server.edition_page(html, "release")
        for page in (html, rel):
            for k in activity.STATES:
                self.assertIn(".ph-k-%s {" % k, page)
            self.assertIn('const PHASE_ORDER = ["P", "D", "dec", "flip_pd", "flip_dp", "flip_tail", "idle", "off", "unknown"]', page)


class TestIdenticalBurst(unittest.TestCase):
    """dmatrix bench: 6 identical 65602-token prompts at once, all cached = 0 -- shown as a burst of
    identical prompts, and the hit rate is also given without it."""

    def test_synthetic_burst(self):
        def smp(t, n, prompt, cached):
            return {"t": t, "r": {}, "front": {"served_tokens": {"P": {"n": n, "prompt": prompt, "cached": cached}}}}
        ring = [smp(0.0, 10, 100000, 40000), smp(5.0, 11, 110000, 48000),       # ordinary leg: 8k of 10k cached
                smp(10.0, 17, 110000 + 6 * 65602, 48000),                           # burst: 6 x 65602, cached 0
                smp(15.0, 18, 110000 + 6 * 65602 + 10000, 56000)]
        bu = ipcboot.identical_bursts(ring)
        self.assertEqual([(b["n"], b["len"]) for b in bu], [(6, 65602)])
        c = ipcboot.cache_view(ring, set(), 16.0)["served_P"]
        self.assertEqual(c["bursts_n"], 1)
        self.assertAlmostEqual(c["ring"]["hit"], 16000 / (6 * 65602 + 20000), places=6)
        self.assertAlmostEqual(c["ring"]["hit_no_burst"], 16000 / 20000, places=6)   # 80 % without the burst
        # two different legs ending together are no burst of identical prompts when their sum does not split
        self.assertEqual(ipcboot.identical_bursts([smp(0, 0, 0, 0), smp(1, 2, 3001, 0)]), [])


class TestPrefillRoute(unittest.TestCase):
    """y5a (d222) front.served = {D: 96, P: 74}: 74 requests through P, 22 direct on D (front.log counted
    21 / 77 of 98 -- the mirror lags 5 s, so the derived split is labelled and the exact field proposed)."""

    def test_derived_and_exact(self):
        r = ipcboot.prefill_route({"served": {"D": 96, "P": 74}})
        self.assertEqual((r["via_p"], r["d_direct"], r["exact"], r["missing"]), (74, 22, False, "front.routes"))
        r = ipcboot.prefill_route({"routes": {"d_direct": 21, "via_p": 77, "d_drain": 0, "reroute_midstream": 0}})
        self.assertEqual((r["via_p"], r["d_direct"], r["exact"]), (77, 21, True))
        self.assertEqual(ipcboot.prefill_route({})["missing"], "front.routes")


class TestWhatNone(unittest.TestCase):
    """NF-Operator 30.09.: flip_first_work what="none" (flip without work after it) is never a Flipzeit --
    not in the tiles, the median, the marks' value or the flip tail -- even if flip_time_ms is set."""

    def test_excluded_everywhere(self):
        from rigdash import ipcfields
        fw = FIRST_WORK + [{"dir": "P>D", "flip_begin_ts": 50.0, "what": "none", "flip_time_ms": 9000,
                            "flip_total_ms": 9000, "reason": "next_flip_before_work"},
                           {"dir": "D>P", "flip_begin_ts": 55.0, "what": "none", "flip_time_ms": None,
                            "flip_total_ms": 1200, "reason": "front_stop"}]
        ft = ipcboot.flip_times_view(fw, FLIP_DONE, False)
        self.assertEqual((ft["P>D"]["n"], ft["P>D"]["median"], ft["P>D"]["no_work"]), (1, 2500.0, 1))
        self.assertEqual(ft["D>P"]["n"], 0)
        self.assertIn("flip_user_time", ft["D>P"]["missing"])      # no fallback on flip_first_work D>P
        ut = [{"dir": "D>P", "start_ts": 49.0, "prefill_start_ts": 51.3, "flip_user_ms": 2300,
               "prefill_start_source": "leg1_dispatch"}]
        ft2 = ipcboot.flip_times_view(fw, FLIP_DONE, False, ut)
        self.assertEqual((ft2["D>P"]["n"], ft2["D>P"]["last"], ft2["D>P"]["missing"]), (1, 2300.0, None))
        self.assertIn("Dispatch", ft2["D>P"]["src"])
        self.assertEqual([r["ms"] for r in ft["recent"]], [2500])
        m = activity.Model(ring_until(60.0), FLIP_DONE + [{"sleep": "D", "wake": "P", "flip_begin_ts": 50.0, "t": 51.0}], fw)
        self.assertEqual(len(m.tails()), 1)
        ev = [{"type": "flip_first_work", "data": x} for x in fw]
        f = ipcfields._flip_first_work({"ipc_events": ev}, "P>D")
        self.assertEqual((f["n"], f["last_ms"]), (1, 2500.0))


class TestKvSleepingGroup(unittest.TestCase):
    """Nutzer 01.10.: in the P<->D flip layout the sleeping group's KV curve drops to 0 while the other
    layout runs; in the PD dual layout (no flips) both curves stay as read."""

    def _kv(self, flip_done):
        m = activity.Model(ring_until(60.0), flip_done, FIRST_WORK if flip_done else [])
        b = m.buckets(0.0, 60, 1.0)
        return b["kv_pct"], b["kv_p_pct"]

    def test_flip_layout_sleeping_group_is_zero(self):
        kv_d, kv_p = self._kv(FLIP_DONE)
        self.assertEqual(kv_d[15], 0.0)            # D sleeps during the P burst
        self.assertAlmostEqual(kv_p[15], 25.0)
        self.assertEqual(kv_p[40], 0.0)            # P sleeps while D decodes
        self.assertAlmostEqual(kv_d[40], 60.0)

    def test_dual_layout_keeps_both(self):
        kv_d, kv_p = self._kv([])
        self.assertAlmostEqual(kv_d[15], 10.0)
        self.assertAlmostEqual(kv_p[40], 25.0)
        self.assertAlmostEqual(kv_d[40], 60.0)


if __name__ == "__main__":
    unittest.main()
