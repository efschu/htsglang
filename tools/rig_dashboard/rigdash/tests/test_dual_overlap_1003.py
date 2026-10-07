"""Nutzer 03.10. ~17:50Z: "beim dashboard muss im balkendiagramm beim 27b dual nvfp4 die prefill/decode schraffiert
übereinanderliegen und auch im tooltip beim hovern beides angezeigt werden in solch einer phase".

27B NVFP4 Dual runs P (PP3) and D (TP3) AT THE SAME TIME on their own cards, no flip.  The phase bar took the
one segment of highest priority per stretch, so a P prefill hid D's decode under it (live boot 9bfe 17:27Z:
192 s of P x D overlap, all drawn as P alone).  Now, in a dual boot only, a P segment in which D placed work too
carries ``co`` = "dec" / "D" with D's rate (``co_tps``); the bar hatches both colours, the hover names both.
Flip boots (27B INT8, NF) keep their segments byte for byte (no ``co``, no ph_Pdec/ph_PD history rows).
"""

import os
import shutil
import subprocess
import sys
import unittest

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from boot_replay import replay_boot  # noqa: E402
from rigdash import activity, ipcboot, server  # noqa: E402
from test_activity_0930 import CH, p_chunks, rec_p  # noqa: E402


def rec_d_dual(ts):
    """D decodes 5 -> 40 s at 200 tok/s (2 streams) while P prefills 10 -> 25.5 s; no flip, no extend."""
    dec = max(0.0, min(ts, 40.0) - 5.0)
    return {"schema": "weg2.rankstats/1", "ts": ts,
            "prefill": {"chunks": 0, "new_tokens": 0, "cached_tokens": 0, "compute_ms": 0.0, "last": None},
            "decode": {"tokens": int(200 * dec), "rounds": int(50 * dec), "gpu_ms": 1000.0 * dec * 0.9,
                       "running": 2 if 5.0 <= ts <= 40.0 else 0},
            "sched": {"full_token_usage": 0.5}}


def dual_ring(t_end=45.0):
    ch = p_chunks()
    ring, t = [], 0.3
    while t <= t_end:
        ts = t - 0.1
        r = {"P.tp0pp%d" % st: ipcboot.compact(rec_p(st, ts, ch)) for st in range(3)}
        r["D.tp0pp0"] = ipcboot.compact(rec_d_dual(ts))
        ring.append({"t": t, "r": r, "front": {"awake": "D", "queue": 0, "outstanding": {"P": 1, "D": 2}}})
        t += 1.0
    return ring


class TestOverlapModel(unittest.TestCase):
    def test_dual_p_segment_names_the_simultaneous_decode(self):
        segs = activity.Model(dual_ring(), [], [], dual=True).segments()
        co = [x for x in segs if x["k"] == "P" and x.get("co") == "dec"]
        self.assertTrue(co, segs)
        # the whole P burst (10.0 -> 25.5 s on the last stage) lies inside D's decode (5 -> 40 s)
        self.assertLess(abs(min(x["s"] for x in co) - 10.0), 0.2)
        self.assertLess(abs(max(x["e"] for x in co) - 25.5), 1.1)
        self.assertFalse([x for x in segs if x["k"] == "P" and not x.get("co")], segs)
        self.assertTrue(any(x["k"] == "dec" for x in segs))

    def test_without_dual_the_segments_are_unchanged(self):
        ring = dual_ring()
        old = activity.Model(ring, [], []).segments()
        self.assertFalse([x for x in old if "co" in x])
        new = activity.Model(ring, [], [], dual=True).segments()
        # same kinds over the same time: dual only annotates (and splits) P, never changes a kind
        cover = lambda ss, k: round(sum(x["e"] - x["s"] for x in ss if x["k"] == k), 6)  # noqa: E731
        for k in activity.STATES:
            self.assertEqual(cover(old, k), cover(new, k), k)

    def test_timeline_carries_both_rates(self):
        ring = dual_ring()
        m = activity.Model(ring, [], [], dual=True)
        tl = ipcboot.timeline_view(m, False, "D", ring[-1]["t"], ring[0]["t"])
        p = [x for x in tl["segs"] if x["k"] == "P" and x.get("co") == "dec"]
        tok = sum(x["tok"] for x in p)
        dur = sum(x["e"] - x["s"] for x in p)
        self.assertAlmostEqual(tok, 4 * CH, delta=1.0)
        # D decodes 200 tok/s throughout, also under the P burst
        self.assertAlmostEqual(sum(x["co_tok"] for x in p) / dur, 200.0, delta=15.0)
        self.assertTrue(all(x["co_tps"] > 150 for x in p if x["e"] - x["s"] > 1.0))
        self.assertTrue(all("co_by_bs" in x for x in p))

    def test_bucket_shares(self):
        ring = dual_ring()
        b = activity.Model(ring, [], [], dual=True).buckets(0, 45, 1.0)
        self.assertAlmostEqual(sum(v or 0 for v in b["ph_Pdec"]), sum(v or 0 for v in b["ph_P"]), delta=0.01)
        self.assertGreater(sum(v or 0 for v in b["ph_Pdec"]), 14.0)
        nb = activity.Model(ring, [], []).buckets(0, 45, 1.0)
        self.assertNotIn("ph_Pdec", nb)
        self.assertNotIn("ph_PD", nb)


class TestIsDual(unittest.TestCase):
    def test_profile_tag_counters_and_flips(self):
        self.assertTrue(ipcboot.is_dual({"profile": "27b-nvfp4-dual1m-psleep", "tag": "dkr27bnvfp4dual1mpsleepbar1fs10031727"}))
        self.assertTrue(ipcboot.is_dual({"profile": None, "front": {"counters": {"dual_passes": 3}}}))
        self.assertFalse(ipcboot.is_dual({"profile": "27b-row-authority", "tag": "dkr27browauthoritybar1fs10031707"}))
        self.assertFalse(ipcboot.is_dual({"profile": "nf-int4-h6-abl", "tag": "dkrnfint4h6ablbar1dauer10031352"}))
        # a boot that flipped is a flip layout, whatever its name
        self.assertFalse(ipcboot.is_dual({"profile": "27b-nvfp4-dual1m-psleep",
                                          "ipc_events": [{"type": "flip_done", "ts": 1.0, "data": {}}]}))


class TestDualReplay(unittest.TestCase):
    """The live dual boot 27bbf-boot-20261003T172745Z-9bfe (17:27Z, fixture fixtures/dual_9bfe)."""

    @classmethod
    def setUpClass(cls):
        cls.r = replay_boot("dual_9bfe")

    def test_fixture_is_the_dual_boot(self):
        self.assertEqual(self.r["ipc"].get("boot_id"), "27bbf-boot-20261003T172745Z-9bfe")
        self.assertTrue(ipcboot.is_dual(self.r["ipc"]))
        self.assertTrue(self.r["model"].dual)

    def test_p_and_decode_overlap_drawn_and_rated(self):
        co = [x for x in self.r["segs"] if x["k"] == "P" and x.get("co") == "dec"]
        self.assertGreater(sum(x["e"] - x["s"] for x in co), 100.0)
        long = [x for x in co if x["e"] - x["s"] > 2.0]
        self.assertTrue(long)
        for x in long:
            self.assertGreater(x["tps"], 0)
            self.assertGreater(x["co_tps"], 0)
        # the overlap seconds the audit counts (overlap_s) are now visible in the bar
        drawn = sum(x["e"] - x["s"] for x in self.r["segs"] if x["k"] == "P" and x.get("co"))
        self.assertGreater(drawn, 0.5 * self.r["view"]["timeline"]["overlap"]["p_vs_d_s"])


class TestFlipBootUnchanged(unittest.TestCase):
    """NF y8c (fixtures/y8c_0cf3, a flip boot with real P and D overlap measured 0): no co, no dual rows."""

    def test_no_co_in_a_flip_boot(self):
        r = replay_boot("y8c_0cf3")
        self.assertFalse(ipcboot.is_dual(r["ipc"]))
        self.assertFalse([x for x in r["segs"] if "co" in x or "co_tps" in x])
        m = r["model"]
        b = m.buckets(int(m.ring[0]["t"]), 60, 1.0)
        self.assertNotIn("ph_Pdec", b)


NODE = next((p for p in (shutil.which("node"), "/opt/node-v22.14.0-linux-x64/bin/node",
                         "/opt/node-v22.11.0-linux-x64/bin/node") if p and os.path.exists(p)), None)


def _run_js(expr):
    html = open(os.path.join(server.STATIC, "index.html"), encoding="utf-8").read()
    a, b = html.index("// ---- phase bar:"), html.index("// label = mean tok/s")
    prelude = ('const window = {i18nLang: () => "en", innerWidth: 1400};\n'
               'const fmt = (v, d = 0) => (v == null || !isFinite(v)) ? "—" : Number(v).toLocaleString("en-US", '
               '{maximumFractionDigits: d, minimumFractionDigits: d});\n'
               'const esc = (s) => String(s); const hhmm = (t) => "t" + t; const srcOf = () => "";\n')
    out = subprocess.run([NODE, "-e", prelude + html[a:b] + "\nprocess.stdout.write(" + expr + ");"],
                         capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return out.stdout


@pytest.mark.skipif(NODE is None, reason="kein node")
def test_hover_and_bar_show_both_phases():
    seg = ('{s: 100, e: 110, k: "P", co: "dec", tps: 4545.2, tok: 45452, co_tps: 82.4, co_tok: 824, '
           'co_by_bs: [{bs: 6, tps: 82.4, per_slot: 13.7, busy_s: 10}]}')
    tip = _run_js("phaseTip(%s, false)" % seg)
    assert "P prefill + D decode simultaneously" in tip
    assert "<b>P prefill</b> &middot; <b>4,545</b> tok/s" in tip
    assert "<b>D decode</b> &middot; <b>82.4</b> tok/s" in tip and "bs 6: <b>82.4</b> tok/s" in tip
    bar = _run_js('phaseHtml({stem: "x", timeline: {segs: [%s], span_s: 900, t1: 110}})' % seg)
    assert 'class="seg ph-k-P co-dec"' in bar and "P prefill + D decode simultaneously" in bar
    # a flip boot's P segment: one phase, as before
    plain = '{s: 100, e: 110, k: "P", tps: 4545.2, tok: 45452}'
    tip0 = _run_js("phaseTip(%s, false)" % plain)
    assert "gleichzeitig" not in tip0 and tip0.startswith("<b>P prefill</b> &middot; <b>4,545</b> tok/s")
    bar0 = _run_js('phaseHtml({stem: "x", timeline: {segs: [%s], span_s: 900, t1: 110}})' % plain)
    assert 'class="seg ph-k-P"' in bar0 and "gleichzeitig" not in bar0
