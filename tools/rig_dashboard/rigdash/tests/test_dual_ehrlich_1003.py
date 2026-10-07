"""Nutzer 03.10. (27B NVFP4 Dual, Balken P+D schraffiert): "fuer D nur 1-30 tok/s ... falsch".

Am Log geprueft (Koordinator):
* Segmente mit co="D": die D-"Prefill"-Chunks neben P sind Resume-Extends (#988 LOADBACK + 1-Token-Extend auf
  14-20k gecachten Tokens) = Uebernahme einer P-Anfrage aus dem Cache.  1-5 tok/s "chunks at compute time" sind dort
  keine Prefill-Leistung -> Tooltip nennt Anfragen / gecachte / neue Tokens, keine tok/s.
* Segmente mit co="dec": 13-60 tok/s sind echt -- die D-Runde wird unter vollem P-Prefill um ein Vielfaches
  langsamer -> Tooltip nennt die Rundenzeit (Delta decode.gpu_ms_by_bs) gegen die Runden ohne P.
* Flip-Boots bleiben Byte fuer Byte: kein co_*-Feld, Timeline-Hash wie vor der Aenderung.
"""

import hashlib
import json
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


class TestCoExtends(unittest.TestCase):
    def chunk(self, s, e, tok, cached, n=1):
        return {"s": s, "e": e, "tok": float(tok), "cached": float(cached), "n": n, "parts": [(s, e)]}

    def test_resume_extend_is_not_prefill(self):
        r = activity.co_extends([self.chunk(10.0, 10.5, 1, 18000), self.chunk(10.6, 11.0, 24, 14000)], 10.0, 11.0)
        self.assertTrue(r["resume"])
        self.assertEqual(r["n"], 2)
        self.assertAlmostEqual(r["cached"], 32000, delta=1)
        self.assertAlmostEqual(r["new"], 25, delta=0.5)

    def test_real_prefill_on_d_keeps_its_rate(self):
        r = activity.co_extends([self.chunk(10.0, 12.0, 6000, 0)], 10.0, 12.0)
        self.assertFalse(r["resume"])
        # a big new-token chunk beside a resume: the stretch is no hand-over
        r = activity.co_extends([self.chunk(10.0, 10.5, 1, 18000), self.chunk(10.6, 12.0, 6000, 2000)], 10.0, 12.0)
        self.assertFalse(r["resume"])

    def test_slice_between_chunk_windows_still_counts_the_burst(self):
        c = dict(self.chunk(10.0, 10.1, 1, 9000), parts=[(10.0, 11.0)])
        r = activity.co_extends([c], 10.5, 10.9)
        self.assertGreaterEqual(r["n"], 1)


class TestDecodeRounds(unittest.TestCase):
    def iv(self, s, e, rnd):
        return {"s": s, "e": e, "parts": [(s, e)], "dur": e - s, "rnd": rnd}

    def test_round_ms_against_the_rounds_without_p(self):
        solo = [self.iv(0, 10, {"3": (200.0, 9400.0)})]                 # 47 ms
        under = [self.iv(20, 24, {"3": (25.0, 4300.0)})]                # 172 ms
        r = activity.decode_rounds(solo + under, 20, 24, solo)
        self.assertAlmostEqual(r["ms"], 172.0, delta=0.1)
        self.assertAlmostEqual(r["solo_ms"], 47.0, delta=0.1)
        self.assertEqual(r["by_bs"][0]["bs"], 3)
        # a reference needs rounds (SOLO_MIN_ROUNDS): none -> no solo_ms, the round time stays
        r2 = activity.decode_rounds(under, 20, 24, [])
        self.assertIsNone(r2["solo_ms"])
        self.assertAlmostEqual(r2["ms"], 172.0, delta=0.1)

    def test_share_inside_the_segment(self):
        r = activity.decode_rounds([self.iv(0, 10, {"2": (100.0, 5000.0)})], 0, 5)
        self.assertAlmostEqual(r["n"], 50.0, delta=0.01)
        self.assertAlmostEqual(r["ms"], 50.0, delta=0.01)


class TestDualReplay(unittest.TestCase):
    """The live dual boot 27bbf-boot-20261003T172745Z-9bfe (fixtures/dual_9bfe)."""

    @classmethod
    def setUpClass(cls):
        cls.r = replay_boot("dual_9bfe")

    def test_d_prefill_beside_p_is_a_takeover_without_a_rate(self):
        co = [x for x in self.r["segs"] if x["k"] == "P" and x.get("co") == "D"]
        self.assertGreater(len(co), 10)
        res = [x for x in co if x.get("co_resume")]
        self.assertGreater(len(res), 0.5 * len(co))
        for x in res:
            self.assertIsNone(x["co_tps"], x)
            self.assertGreaterEqual(x["co_n"], 1, x)
            self.assertGreater(x["co_cached"], 0, x)
            self.assertLess(x["co_new"], 100, x)
        for x in co:
            self.assertIn("co_n", x)

    def test_decode_under_p_carries_the_round_time(self):
        dec = [x for x in self.r["segs"] if x["k"] == "P" and x.get("co") == "dec" and x.get("co_round_ms")]
        self.assertTrue(dec)
        for x in dec:
            self.assertGreater(x["co_tps"], 0)           # the tok/s stay (real)
            self.assertTrue(x["co_round"])
        long = max(dec, key=lambda x: x["e"] - x["s"])
        # the longest stretch under P: the round is several times slower than without P (rankstats, bs-weighted)
        self.assertIsNotNone(long["co_solo_ms"])
        self.assertGreater(long["co_round_ms"], 2.0 * long["co_solo_ms"])


class TestFlipBootUnchanged(unittest.TestCase):
    """NF y8c (fixtures/y8c_0cf3, a flip boot): segments and the whole timeline are the bytes the base release
    (e5a274ddb4) produced -- hashes taken from that release, EXCEPT the D segments of Auftrag 880 (Nutzer 03.10. "6 token/s
    prefill in D???"): a D segment whose chunks are all narrower than activity.WIDE_MIN_TOK (admit extends of 25..92 tokens)
    now has tps None and carries admit_n/admit_tok.  Verified by diff against the base release: 10 of 125 segments changed (all D: 8x tps 1..181 tok/s -> None, e.g. a "1.0 tok/s" one),
    admit keys added, tps of a segment with a wide chunk 80,9 -> 80,5 (no Dual/co key, no P/dec/flip segment touched).  Hashes below are the 880 state."""
    SEGS = "9a04f69aa116537ccb709ec35b8f77bf01d919e433a63c9a91240b62b29e9823"      # 07.10. English texts: only the strings why/src/reqs_src changed (verified by structured diff against 07c20a35e5, no key/number changed)
    TIMELINE = "1502f91e5bc7d65d4594211de32879354a60f436b8fc5a38520809e3e7d64710"

    def test_flip_boot_bytes(self):
        r = replay_boot("y8c_0cf3")
        self.assertFalse(ipcboot.is_dual(r["ipc"]))
        sha = lambda o: hashlib.sha256(json.dumps(o, sort_keys=True, default=repr).encode()).hexdigest()  # noqa: E731
        self.assertEqual(sha(r["segs"]), self.SEGS)
        self.assertEqual(sha(r["view"]["timeline"]), self.TIMELINE)
        self.assertFalse([x for x in r["segs"] if any(k.startswith("co") for k in x)])


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
def test_tooltip_resume_segment_names_takeover_not_rate():
    seg = ('{s: 100, e: 101, k: "P", co: "D", tps: 2308.8, tok: 2300, co_tps: null, co_tok: 0.9, co_resume: true, '
           'co_n: 3, co_cached: 59882.7, co_new: 3.4}')
    tip = _run_js("phaseTip(%s, false)" % seg)
    d_part = tip.split("</i> <b>D prefill/extend</b>")[1].split("<br>")[0]
    assert "takes over 3 requests from the cache" in d_part
    assert "59,883 tok cached, 3 new" in d_part
    assert "chunks at compute time" not in d_part and "<b>2</b> tok/s" not in d_part and "<b>1</b> tok/s" not in d_part
    assert "no prefill, no tok/s" in d_part
    bar = _run_js('phaseHtml({stem: "x", timeline: {segs: [%s], span_s: 900, t1: 110}})' % seg)
    assert "D takes over 3 from the cache" in bar


@pytest.mark.skipif(NODE is None, reason="kein node")
def test_tooltip_real_d_prefill_keeps_rate():
    seg = ('{s: 100, e: 110, k: "P", co: "D", tps: 2308.8, tok: 2300, co_tps: 1234.4, co_tok: 12344, '
           'co_n: 2, co_cached: 5000, co_new: 12344}')
    tip = _run_js("phaseTip(%s, false)" % seg)
    assert "<b>1,234</b> tok/s" in tip and "chunks at compute time" in tip and "2 requests, 5,000 tok cached" in tip


@pytest.mark.skipif(NODE is None, reason="kein node")
def test_tooltip_decode_under_p_shows_round_time():
    seg = ('{s: 100, e: 110, k: "P", co: "dec", tps: 4545.2, tok: 45452, co_tps: 17.8, co_tok: 178, '
           'co_by_bs: [{bs: 4, tps: 17.8, per_slot: 4.4, busy_s: 10}], co_round_ms: 172.1, co_solo_ms: 47.0, '
           'co_round: [{bs: 4, ms: 172.1, n: 42, solo_ms: 47.0}]}')
    tip = _run_js("phaseTip(%s, false)" % seg)
    assert "<b>17.8</b> tok/s" in tip                       # the rate stays: it is real
    assert "D round avg <b>172</b> ms" in tip and "without P avg 47 ms" in tip and "&times;3.7" in tip
    assert "P computes simultaneously" in tip and "bs 4: 172 ms (without P 47)" in tip
    # without the round data nothing is invented
    tip0 = _run_js("phaseTip(%s, false)" % '{s: 100, e: 110, k: "P", co: "dec", tps: 4545.2, tok: 1, co_tps: 17.8, co_tok: 178}')
    assert "D round" not in tip0


@pytest.mark.skipif(NODE is None, reason="kein node")
def test_flip_boot_tooltips_unchanged():
    for seg in ('{s: 100, e: 110, k: "P", tps: 4545.2, tok: 45452}', '{s: 100, e: 110, k: "D", tps: 900.5, tok: 9005}',
                '{s: 100, e: 110, k: "dec", tps: 71.3, tok: 713}'):
        tip = _run_js("phaseTip(%s, false)" % seg)
        assert "simultaneously" not in tip and "D round" not in tip and "takes over" not in tip
