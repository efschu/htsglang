"""Auftrag 880 (Nutzer 03.10. ~19:50Z): VRAM-Balken und "6 token/s prefill in D???".

A. Kartenplaner-Balken (kartenplan.py + kartenplan.js):
   * jeder Posten trägt Phase (P/D/gemeinsam), Herkunft (gemessen/Planerwert) und eine Ein-Satz-Erklärung (Tooltip);
   * die Posten einer Phase liegen als EIN zusammenhängender Block (gemeinsam, dann P, dann D), stabil sortiert;
   * Summe über der Karte -> over_mib / hard_over_mib (Profil passt nicht) bzw. overlap_mib (Planer-Überbuchung des
     Rang-Budgets, die Karte schließt trotzdem);
   * (mit Playwright, sonst übersprungen) der Balken wächst im DOM über die Kante, Hinweis + Tooltip erscheinen.
B. D-Prefill-Rate: auf D beginnt jeder übernommene Request mit einem 1-Token-Extend (y8vb D.log: 3029 von 3051
   Prefill-Batches #new-token 1, stock "input throughput" 5,8 tok/s).  Das sind Admits, keine Prefill-Geschwindigkeit:
   getrennt gezählt, die Rate nur aus Chunks >= WIDE_MIN_TOK.
"""

import json
import os
import sys
import threading
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from rigdash import activity, ipcboot  # noqa: E402
from rigdash import kartenplan as K  # noqa: E402

TREE = os.path.join(HERE, "fixtures", "kartenplan", "planner_tree", "python")


def plan(profile):
    kp = K.Kartenplaner(tree=TREE)
    return kp.plan({"profile": profile, "cards": kp.catalog()["rig_preset"]["cards"]})["plan"]["einfach"]["bars"]


PROFILES = ("27b-int8", "nf-int4-abl", "27b-nvfp4-dual", "27b-fp8", "27b-gguf-iq4xs")


class TestSegmentDetails(unittest.TestCase):
    def test_every_segment_has_phase_origin_explanation(self):
        for prof in PROFILES:
            for b in plan(prof):
                for g in "PD":
                    for s in b[g]["segments"]:
                        self.assertIn(s["phase"], ("P", "D", "gemeinsam"), (prof, s["key"]))
                        self.assertIn(s["origin"], ("gemessen", "Planerwert"))
                        self.assertTrue(s["origin_note"])
                        self.assertGreater(len(s["what"]), 20, (prof, s["key"]))

    def test_awake_posts_belong_to_the_awake_phase_and_the_sleep_rest_to_the_other(self):
        for b in plan("27b-int8"):
            for g, other in (("P", "D"), ("D", "P")):
                segs = b[g]["segments"]
                for s in segs:
                    if s["key"] == "asleep":
                        self.assertEqual(s["phase"], other)
                    elif s["key"] == "carve":
                        self.assertEqual(s["phase"], "gemeinsam")
                    else:
                        self.assertEqual(s["phase"], g, s["key"])

    def test_measured_vs_planner_value(self):
        segs = {s["key"]: s for s in plan("27b-int8")[0]["D"]["segments"]}
        self.assertEqual(segs["weights"]["origin"], "gemessen")        # Rang-Log
        self.assertEqual(segs["carve"]["origin"], "gemessen")          # NVML
        self.assertEqual(segs["awake_rest"]["origin"], "Planerwert")


class TestPhaseBlocksContiguous(unittest.TestCase):
    def test_phases_form_one_block_each_in_a_fixed_order(self):
        order = list(K.PHASE_BLOCKS)
        for prof in PROFILES:
            for b in plan(prof):
                for g in "PD":
                    ph = [order.index(s["phase"]) for s in b[g]["segments"]]
                    self.assertEqual(ph, sorted(ph), (prof, b["ordinal"], g))     # never P, D, P again

    def test_order_is_stable_between_runs_and_within_a_block_by_post_kind(self):
        a, b = plan("nf-int4-abl"), plan("nf-int4-abl")
        self.assertEqual([[s["key"] for s in x["D"]["segments"]] for x in a], [[s["key"] for s in x["D"]["segments"]] for x in b])
        for x in a:
            keys = [s["key"] for s in x["D"]["segments"] if s["phase"] == "D"]
            idx = [K.SEG_ORDER.index(k) for k in keys]
            self.assertEqual(idx, sorted(idx))

    def test_annotate_unit_interleaved_input_is_grouped(self):
        raw = [K._seg("kv", "KV", 10, "vram_plan"), K._seg("asleep", "Schlaf", 5, "x"), K._seg("weights", "W", 20, "Rang-Log"),
               K._seg("carve", "Carve", 3, "x"), K._seg("asleep", "Schlaf2", 2, "x")]
        out = K._annotate_segments(raw, "D", dual=False)
        self.assertEqual([(s["phase"], s["key"]) for s in out],
                         [("gemeinsam", "carve"), ("P", "asleep"), ("P", "asleep"), ("D", "weights"), ("D", "kv")])


class TestOverflowFields(unittest.TestCase):
    def test_real_records_close_on_the_card(self):
        for prof in ("27b-int8", "nf-int4-abl", "27b-fp8", "27b-gguf-iq4xs"):
            for b in plan(prof):
                for g in "PD":
                    self.assertEqual(b[g]["hard_over_mib"], 0, (prof, b["ordinal"], g))
                    self.assertEqual(b[g]["over_mib"], 0)
                    self.assertEqual(b[g]["sum_mib"], sum(max(0, s["mib"]) for s in b[g]["segments"]))

    def test_dual_budget_overbooking_is_named_overlap_not_a_misfit(self):
        d = [b["D"] for b in plan("27b-nvfp4-dual")]
        self.assertTrue(all(x["over_mib"] > 0 and x["hard_over_mib"] == 0 and x["overlap_mib"] == x["over_mib"] for x in d))
        self.assertTrue(all(x["rest_mib"] < 8 for x in d))        # the planner still closes the card


# ----------------------------------------------------------------------------- B: D admit extends
def rec_d(ts, wide_at=22.0, wide_tok=2000, wide_ms=2000.0):
    w = 1 if ts >= wide_at else 0
    admits = int(ts) - w                              # one 1-token admit per second, 30 ms each; none in the wide second
    last = {"t": wide_at, "gpu_ms": wide_ms, "new": wide_tok} if (w and ts - wide_at < 1.0) else \
        {"t": float(admits), "gpu_ms": 30.0, "new": 1}
    return {"schema": "weg2.rankstats/1", "ts": ts,
            "prefill": {"chunks": admits + w, "new_tokens": admits + wide_tok * w, "cached_tokens": 14000 * admits,
                        "compute_ms": 30.0 * admits + wide_ms * w, "gpu_ms": 30.0 * admits + wide_ms * w, "last": last},
            "decode": {"tokens": 0, "rounds": 0, "gpu_ms": 0.0, "running": 0}, "sched": {"full_token_usage": 0.1}}


def ring(t_end=60):
    return [{"t": float(k) + 0.3, "r": {"D.tp0pp0": ipcboot.compact(rec_d(float(k)))}, "front": {"queue": 0, "outstanding": 0}}
            for k in range(1, t_end + 1)]


class TestDAdmitExtends(unittest.TestCase):
    def setUp(self):
        self.m = activity.Model(ring(), [], [])

    def test_is_wide(self):
        self.assertFalse(activity.is_wide(1, 1))
        self.assertFalse(activity.is_wide(63, 1))
        self.assertTrue(activity.is_wide(64, 1))
        self.assertTrue(activity.is_wide(2001, 2))                     # one wide chunk + one admit in one sample step
        self.assertFalse(activity.is_wide(40, 40))                     # 40 admits in a step: still admits

    def test_the_stock_style_burst_would_say_a_few_tok_per_s(self):
        """What the instrument said before: every chunk chained into ONE burst over the whole minute."""
        old = activity.bursts(self.m.pchunks["D"])
        self.assertEqual(len(old), 1)
        self.assertLess(old[0]["tok"] / old[0]["wall_s"], 100.0)

    def test_prefill_view_rates_only_the_wide_chunk(self):
        pv = ipcboot.prefill_view(self.m, "D", 61.0, None)
        lb = pv["last_burst"]
        self.assertEqual(lb["tokens"], 2000)
        self.assertGreater(lb["wall_tps"], 500.0)                       # ~1000 tok/s, not ~6
        self.assertEqual(pv["min_chunk_tok"], activity.WIDE_MIN_TOK)
        ad = pv["admit"]
        self.assertGreaterEqual(ad["ring"]["n"], 55)
        self.assertAlmostEqual(ad["ring"]["tok_per_req"], 1.0, delta=0.01)

    def test_no_wide_chunk_means_no_rate(self):
        m = activity.Model(ring(20), [], [])                            # the wide chunk (22 s) is not in yet
        pv = ipcboot.prefill_view(m, "D", 21.0, None)
        self.assertIsNone(pv["last_burst"])
        self.assertIsNone(pv["now"])
        self.assertGreater(pv["admit"]["ring"]["n"], 15)

    def test_curve_rate_is_none_where_only_admits_ran(self):
        b = self.m.buckets(0.0, 60, 1.0)
        self.assertIsNone(b["d_rate"][5])                               # admits only
        self.assertAlmostEqual(b["d_rate"][21], 1000.0, delta=60)       # the wide extend at [20, 22)

    def test_totals_split_and_rate(self):
        t = ipcboot.totals_view(ring(), set(self.m.keys), {}, 0.0, 61.0)
        sp = t["d_split"]
        self.assertEqual(sp["wide"]["chunks"], 1)
        self.assertEqual(sp["wide"]["tokens"], 2000)
        self.assertGreaterEqual(sp["admit"]["chunks"], 55)
        self.assertAlmostEqual(sp["admit"]["tok_per_req"], 1.0, delta=0.01)
        self.assertAlmostEqual(t["d_rate_gpu"], 1000.0, delta=1)        # 2000 tok / 2000 ms of the wide chunk only
        self.assertLess(t["d_mean_chunk"], 100)                         # the since-boot mean says: mostly admits

    def test_timeline_segment_names_the_admits(self):
        tl = ipcboot.timeline_view(self.m, False, None, 61.0, boot_t0=0.0, detail=False)
        d = [x for x in tl["segs"] if x["k"] == "D"]
        self.assertTrue(d)
        self.assertTrue(all("admit_n" in x for x in d))
        wide = [x for x in d if x["tps"] is not None]
        self.assertTrue(wide and all(x["tok"] >= 64 for x in wide))
        self.assertTrue(any(x["tps"] is None and x["admit_n"] for x in d) or len(d) == len(wide))


class TestFlipBootOnlyDSegmentsChange(unittest.TestCase):
    """NF y8c (a flip boot): only D segments carry the new admit keys / a None rate; P, decode, flip segments stay as they were."""

    def test_only_d_segments_are_touched(self):
        from rigdash.tests.boot_replay import replay_boot
        segs = replay_boot("y8c_0cf3")["segs"]
        self.assertEqual(len(segs), 125)
        for x in segs:
            if x["k"] != "D":
                self.assertNotIn("admit_n", x)
        adm = [x for x in segs if x.get("admit_n")]
        self.assertTrue(adm)
        for x in adm:
            self.assertEqual(x["k"], "D")
            self.assertLess(x["admit_tok"] / x["admit_n"], activity.WIDE_MIN_TOK)


# ----------------------------------------------------------------------------- A: the page (needs Playwright)
try:
    from playwright.sync_api import sync_playwright   # noqa: E402
except Exception:                                      # noqa: BLE001
    sync_playwright = None


@unittest.skipUnless(sync_playwright, "playwright nicht installiert")
class TestBarDom(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from http.server import ThreadingHTTPServer
        from rigdash import kartenplan_preview as KP
        kp = K.Kartenplaner(tree=TREE)
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), KP.make_handler(kp))
        cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def _page(self, pw, mutate=None, profile=None):
        br = pw.chromium.launch()
        pg = br.new_page(viewport={"width": 1200, "height": 1400})
        self.errs = []
        pg.on("pageerror", lambda e: self.errs.append(str(e)))
        if mutate:
            def h(route):
                resp = route.fetch()
                j = resp.json()
                mutate(j)
                route.fulfill(response=resp, body=json.dumps(j))
            pg.route("**/api/kartenplan/plan*", h)
        pg.goto("http://127.0.0.1:%d/#t=kartenplan" % self.port)
        pg.wait_for_selector(".kp-bar", timeout=20000)
        if profile:
            pg.select_option("#kp-profile", profile)
            pg.wait_for_timeout(1000)
        return br, pg

    def test_tooltip_on_hover_and_phase_blocks(self):
        with sync_playwright() as pw:
            br, pg = self._page(pw)
            try:
                self.assertEqual(pg.locator(".kp-bar").count(), 6)
                self.assertEqual(pg.locator(".kp-edge").count(), 0)         # real records do not overflow
                pg.locator(".kp-bar > i[data-s]:not([data-s=free])").first.hover()
                tip = pg.locator(".kp-tip").inner_text()
                for want in ("MiB", "% der Karte", "Herkunft:", "P-Phase"):
                    self.assertIn(want, tip)
                blocks = pg.locator(".kp-strip").first.locator(".kp-ph").all()
                self.assertEqual([b.get_attribute("title").split(":")[0] for b in blocks], ["P-Phase", "D-Phase"])
                self.assertEqual(self.errs, [])
            finally:
                br.close()

    def test_wrong_values_grow_past_the_card_edge_with_a_notice(self):
        def mutate(j):
            ph = j["plan"]["einfach"]["bars"][1]["D"]
            for s in ph["segments"]:
                if s["key"] == "kv":
                    s["mib"] += 9000
            ph["sum_mib"] = sum(max(0, s["mib"]) for s in ph["segments"])
            ph["over_mib"] = ph["hard_over_mib"] = ph["sum_mib"] - ph["total_mib"]
            ph["overlap_mib"] = 0
        with sync_playwright() as pw:
            br, pg = self._page(pw, mutate)
            try:
                self.assertEqual(pg.locator(".kp-edge").count(), 1)
                self.assertGreaterEqual(pg.locator(".kp-beyond").count(), 1)
                alert = pg.locator(".kp-over.bad").all_inner_texts()
                self.assertTrue(any("über dem VRAM – Profil passt nicht" in t and "Karte 2" in t and "Größte Posten" in t for t in alert), alert)
                pg.locator(".kp-beyond").first.hover()
                self.assertIn("hinter dem Kartenende", pg.locator(".kp-tip").inner_text())
            finally:
                br.close()

    def test_dual_overbooking_is_amber_not_a_misfit(self):
        with sync_playwright() as pw:
            br, pg = self._page(pw, profile="27b-nvfp4-dual")
            try:
                self.assertEqual(pg.locator(".kp-over.bad").count(), 0)
                self.assertGreaterEqual(pg.locator(".kp-over.warn").count(), 1)
                self.assertGreaterEqual(pg.locator(".kp-soft").count(), 1)
            finally:
                br.close()


if __name__ == "__main__":
    unittest.main()
