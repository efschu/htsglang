"""Nutzer 02.10. ~11:00Z: "dann muss der visiontower laden rechnen entladen auch mit in die phasenliste ins
dashboard".  Source: rankstats ``vision`` of P's PP0 (writer weg2/rank_timing.note_vision_*, desk/nf-vision-ipc-1002)
-- IPC, never the log line (test_no_new_log_parsers).

Metal shape (y7h-noH4, P PP0, 02.10.):
  run=1 10:54:28Z legs_ms=(build 69, load 488, encode 838, attach 0, teardown 439)   fresh image, weg2-0-1
  run=2 10:55:03Z legs_ms=(build 55, load 478, encode 20, attach 0, teardown 413)    cached image, weg2-1-5
The stage runs in P's admission after the D>P flip -- inside the flip tail, before the first P chunk.
"""
import unittest

from rigdash import activity, ipcboot

T1 = 1790938468.5          # 2026-10-02 10:54:28.5Z, the run-1 line (end of the stage)
T2 = 1790938503.0          # 10:55:03Z, run 2


def run(no, end, legs_ms, rid, reserve_ms=3.0):
    """A rankstats vision run as the writer lays it: legs back to back, ending at ``end``."""
    order = ["build", "reserve", "load", "encode", "attach", "teardown"]
    ms = dict(legs_ms, reserve=reserve_ms)
    t = end - sum(ms.values()) / 1e3
    legs = {}
    for k in order:
        legs[k] = [round(t, 3), round(t + ms[k] / 1e3, 3), ms[k]]
        t += ms[k] / 1e3
    return {"run": no, "ok": True, "code": "W102", "t0": legs["build"][0], "t1": end, "rids": [rid],
            "tower_mib": 856.3, "place": "kvtail", "card": 1, "legs": legs}


RUN1 = run(1, T1, {"build": 69, "load": 488, "encode": 838, "attach": 0, "teardown": 439}, "weg2-0-1")
RUN2 = run(2, T2, {"build": 55, "load": 478, "encode": 20, "attach": 0, "teardown": 413}, "weg2-1-5")


def pp0(ts, runs, live=None, fwd=0):
    return ipcboot.compact({"ts": ts, "vision": {"runs": len(runs), "live": live, "recent": runs},
                            "work": {"forward_ct": fwd}})


def ring_of(*samples):
    return [{"t": t, "r": {"P.tp0pp0": r}, "front": {}} for t, r in samples]


class VisionCompact(unittest.TestCase):
    def test_compact_keeps_runs_and_live(self):
        c = pp0(T1 + 1, [RUN1], live={"run": 2, "leg": "load", "since": T2 - 0.5, "rids": ["weg2-1-5"]})
        v = c["vis"]
        self.assertEqual(v["live"]["leg"], "load")
        self.assertEqual(v["recent"][0]["rids"], ["weg2-0-1"])
        self.assertEqual(v["recent"][0]["mib"], 856.3)
        self.assertEqual(set(v["recent"][0]["legs"]), {"build", "reserve", "load", "encode", "attach", "teardown"})

    def test_no_vision_block_no_key_value(self):
        self.assertIsNone(ipcboot.compact({"ts": 1.0})["vis"])


class VisionPhases(unittest.TestCase):
    def test_run1_laden_rechnen_entladen_in_the_phase_list(self):
        ring = ring_of((T1 - 3.0, pp0(T1 - 3.0, [])), (T1 + 0.5, pp0(T1 + 0.5, [RUN1])))
        m = activity.Model(ring, [], [])
        segs = [x for x in m.segments(T1 + 0.5) if x["k"].startswith("vis_")]
        self.assertEqual([x["k"] for x in segs], ["vis_load", "vis_enc", "vis_unload"])
        lo, enc, un = segs
        self.assertAlmostEqual(lo["e"] - lo["s"], (69 + 3 + 488) / 1e3, places=2)    # build + reserve + load
        self.assertAlmostEqual(enc["e"] - enc["s"], 838 / 1e3, places=2)            # encode (+ attach 0)
        self.assertAlmostEqual(un["e"] - un["s"], 439 / 1e3, places=2)
        self.assertAlmostEqual(un["e"], T1, places=2)

    def test_cached_image_run2_encode_is_a_sliver_not_a_gap(self):
        ring = ring_of((T2 - 3.0, pp0(T2 - 3.0, [RUN1])), (T2 + 0.5, pp0(T2 + 0.5, [RUN1, RUN2])))
        m = activity.Model(ring, [], [])
        segs = [x for x in m.segments(T2 + 0.5) if x["k"].startswith("vis_") and x["s"] > T1]
        self.assertEqual([x["k"] for x in segs], ["vis_load", "vis_enc", "vis_unload"])
        self.assertAlmostEqual(segs[1]["e"] - segs[1]["s"], 0.020, places=2)

    def test_vision_wins_over_the_flip_tail_it_sits_in(self):
        # D>P flip done 2 s before the stage; first P chunk after it: the tail is cut by the stage
        fd = [{"flip_begin_ts": T1 - 5.0, "t": T1 - 2.5, "sleep": "D", "wake": "P"}]
        fw = [{"flip_begin_ts": T1 - 5.0, "dir": "D>P", "first_work_ts": T1 + 0.4, "what": "prefill"}]
        # Nutzer 02.10. ~13:05Z: the D>P tail ends at P's first prefill forward (PP0 forward_ct), after the stage
        ring = ring_of((T1 - 6.0, pp0(T1 - 6.0, [])), (T1 - 3.0, pp0(T1 - 3.0, [])), (T1 + 0.5, pp0(T1 + 0.5, [RUN1], fwd=1)))
        m = activity.Model(ring, fd, fw)
        ks = [x["k"] for x in m.segments(T1 + 0.5) if x["e"] > T1 - 2.5]
        self.assertIn("vis_load", ks)
        self.assertIn("vis_enc", ks)
        self.assertEqual(ks[0], "flip_tail")                 # the tail before the stage stays a tail
        tl = [x for x in m.segments(T1 + 0.5) if x["k"] == "flip_tail"]
        self.assertTrue(all(x["e"] <= RUN1["t0"] + 1e-3 or x["s"] >= T1 - 1e-3 for x in tl))

    def test_live_leg_shows_now_and_in_the_active_frame(self):
        live = {"run": 2, "leg": "load", "since": T2 - 0.6, "rids": ["weg2-1-5"]}
        ring = ring_of((T2 - 2.0, pp0(T2 - 2.0, [RUN1])), (T2 - 0.1, pp0(T2 - 0.1, [RUN1], live=live)))
        m = activity.Model(ring, [], [])
        tl = ipcboot.timeline_view(m, True, "P", T2, T1 - 10.0)
        last = [x for x in tl["segs"] if x["k"] != "unknown"][-1]
        self.assertEqual(last["k"], "vis_load")
        self.assertTrue(last.get("running"))
        self.assertEqual(last.get("vis_live"), "load")
        pn = ipcboot.phase_now(tl["segs"], {}, {"state": "serving", "awake": "P"}, [], True, T2)
        self.assertEqual(pn["label"], "Vision load")
        self.assertIn("weg2-1-5", pn["sub"])

    def test_states_and_history_fractions_carry_the_vision_phases(self):
        for k in ("vis_load", "vis_enc", "vis_unload"):
            self.assertIn(k, activity.STATES)
        ring = ring_of((T1 - 3.0, pp0(T1 - 3.0, [])), (T1 + 0.5, pp0(T1 + 0.5, [RUN1])))
        m = activity.Model(ring, [], [])
        b = m.buckets(int(T1) - 3, 4, 1.0)
        self.assertGreater(sum(v or 0 for v in b["ph_vis_enc"]), 0.5)

    def test_annotated_segment_names_rid_mib_and_legs(self):
        ring = ring_of((T1 - 3.0, pp0(T1 - 3.0, [])), (T1 + 0.5, pp0(T1 + 0.5, [RUN1])))
        m = activity.Model(ring, [], [])
        tl = ipcboot.timeline_view(m, False, "P", T1 + 0.5, T1 - 10.0)
        enc = next(x for x in tl["segs"] if x["k"] == "vis_enc")
        self.assertEqual((enc["run"], enc["rids"], enc["mib"]), (1, ["weg2-0-1"], 856.3))
        self.assertEqual(enc["legs_ms"], {"encode": 838, "attach": 0})


if __name__ == "__main__":
    unittest.main()
