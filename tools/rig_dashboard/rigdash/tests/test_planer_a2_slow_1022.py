"""PLANER A2 1022 (review done/planer-nfblock-review-1022.md, Auflage A2): the planner page marks a layout the launcher STARTS but whose decode its own solver
prices far above the reference rig as "startable, slow (estimate)" -- a named hint in the verdict (``verdikt.langsam``, code SLOW-ESTIMATE), no lock.

The hint is derived from the launcher solver's result (``plan.d_decode`` = ``propose_oracle.d_decode_figures``), never from model or card count:
  * estimated bs1 decode round (max over the D ranks) > ``SLOW_BS1_ROUND_MS`` = 2 x 27.1 ms (the upper measured bs1 round of the reference rig, nf-s40), or
  * the owned solve's nfblock wave fallback (no form at the Form-A standard waves, searched again at Form A's waves).

Figures below are those of the planner page's own dry runs, 22.10. (``ZielPlaner.plan`` with the real oracle, NF INT4 flip, this tree):
  2 x RTX 3090: round bs1 73.84/73.84 ms, 10 waves (standard 2), fallback Env 2 -> Form A 6     -> hint
  3 x RTX 3090: round bs1 27.26/24.66/23.96 ms, 4 waves, no fallback                             -> no hint
  rig (5090 + 2 x 3080): round bs1 35.56/9.52/28.28 ms, 2 waves, no fallback                     -> no hint
  4 x RTX 3090: Form A kept (no cut solve, no round), 2 waves                                    -> no hint
``RealChain`` repeats those dry runs (box-bound, ~2 min: it needs the models, the census and this tree's launcher).
"""

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

from rigdash import planer_ziel as Z  # noqa: E402
import test_planer_release_1022 as T  # noqa: E402

STATIC = os.path.join(os.path.dirname(HERE), "static")
REPO_ROOT = T.REPO_ROOT
SEED = "Saat UNMEASURED (plan_s3_251 §1, H29/x138)"

#: the decode figures of the page's dry runs (see the module docstring)
TWO = {"waves": 10, "waves_default": 2, "round_bs1_ms": [73.84, 73.84], "form_a_round_bs1_ms": [57.35, 106.15], "miss_cost": "0.1/0.2", "miss_cost_source": SEED,
       "wave_fallback": {"env_waves": 2, "form_a_waves": 6, "result": "9 tragbar"}}
THREE = {"waves": 4, "waves_default": 2, "round_bs1_ms": [27.26, 24.66, 23.96], "form_a_round_bs1_ms": [19.44, 32.5, 30.33], "miss_cost": "0.1/0.2",
         "miss_cost_source": SEED, "wave_fallback": None}
RIG = {"waves": 2, "waves_default": 2, "round_bs1_ms": [35.56, 9.52, 28.28], "form_a_round_bs1_ms": [32.8, 10.83, 31.77], "miss_cost": "0.1/0.2",
       "miss_cost_source": SEED, "wave_fallback": None}
FOUR = {"waves": 2, "waves_default": 2, "round_bs1_ms": None, "form_a_round_bs1_ms": None, "miss_cost": None, "miss_cost_source": None, "wave_fallback": None}


def hint(d, outcome="ok_with_force"):
    return Z.slow_hint(outcome, {"d_decode": d})


class TheThreshold(unittest.TestCase):
    def test_the_threshold_is_twice_the_measured_reference(self):
        self.assertEqual(Z.REFERENCE_BS1_ROUND_MS, (22.7, 27.1))                # nf-s40-decode-runde-1018.md, "Rundenebene", bs1 ALLE: rc12 / rc11
        self.assertEqual(Z.SLOW_BS1_ROUND_MS, 54.2)

    def test_the_boundary(self):                                                # the mutant direction: a moved threshold moves this line
        no_fb = dict(TWO, wave_fallback=None, waves=2)
        self.assertIsNone(hint(dict(no_fb, round_bs1_ms=[54.2, 10.0])))
        self.assertEqual(hint(dict(no_fb, round_bs1_ms=[10.0, 54.3]))["round_bs1_ms"], 54.3)     # the slowest rank decides (the round is synchronous)

    def test_the_page_cases_on_their_round_alone(self):
        # without the wave fallback the ROUND alone must decide right for both ends of the dry runs: 73.84 is slow, the rig's 35.56 is not
        self.assertIsNotNone(hint(dict(TWO, wave_fallback=None)))
        self.assertIsNone(hint(RIG))
        self.assertIsNone(hint(THREE))


class TheHint(unittest.TestCase):
    def test_two_rtx3090_carry_the_named_hint_with_the_figure(self):
        h = hint(TWO)
        self.assertEqual((h["code"], h["label"]), ("SLOW-ESTIMATE", "startable, slow (estimate)"))
        self.assertEqual((h["round_bs1_ms"], h["waves"], h["waves_default"]), (73.84, 10, 2))
        text = " ".join(h["saetze"])
        for frag in ("73.8 ms per bs1 decode round", "22.7 ms to 27.1 ms measured on the reference rig", "up to 10 overflow waves (Form A standard: 2)",
                     "not priced, so the figure is a lower bound", "standard 2 overflow waves; the solver searched again with up to 6", "unmeasured seed",
                     "No lock"):
            self.assertIn(frag, text)

    def test_three_rtx3090_the_rig_and_form_a_carry_none(self):
        for d in (THREE, RIG, FOUR):
            self.assertIsNone(hint(d), d)

    def test_the_wave_fallback_alone_is_enough(self):                           # the solver itself chose waves above the Form-A standard
        h = hint(dict(THREE, wave_fallback={"env_waves": 2, "form_a_waves": 5, "result": "3 tragbar"}))
        self.assertIsNotNone(h)
        self.assertIn("up to 5", " ".join(h["saetze"]))

    def test_never_on_a_refusal_or_without_figures(self):
        for oc in ("verweigert", "crash", None):
            self.assertIsNone(hint(TWO, oc))
        self.assertIsNone(Z.slow_hint("ok", None))
        self.assertIsNone(Z.slow_hint("ok", {}))                                # 27B: no expert pool, no d_decode
        self.assertIsNotNone(hint(TWO, "geht"))

    def test_no_lower_bound_sentence_at_the_standard_waves(self):
        h = hint(dict(TWO, waves=2, wave_fallback=None))
        self.assertNotIn("lower bound", " ".join(h["saetze"]))

    def test_the_verdict_carries_it(self):
        vd = {"outcome": "ok_with_force", "verdikte": [], "plan": {"d_decode": TWO}}
        self.assertEqual(Z.ZielPlaner._verdict(vd, {})["langsam"]["code"], "SLOW-ESTIMATE")
        self.assertIsNone(Z.ZielPlaner._verdict(dict(vd, plan={"d_decode": THREE}), {})["langsam"])
        self.assertIsNone(Z.ZielPlaner._verdict({"outcome": "ok", "verdikte": []}, {})["langsam"])     # an old oracle answer without d_decode


class German(unittest.TestCase):
    def test_every_sentence_is_german_with_the_decimal_comma(self):
        h = Z.localize(hint(TWO), "de")
        self.assertEqual(h["label"], "startbar, langsam (Hochrechnung)")
        self.assertEqual(h["code"], "SLOW-ESTIMATE")
        for s in h["saetze"]:
            self.assertEqual(T.english_words(s), [], s)
        text = " ".join(h["saetze"])
        self.assertIn("73,8 ms je bs1-Decode-Runde", text)
        self.assertIn("22,7 ms bis 27,1 ms", text)
        self.assertIn("über 54,2 ms", text)
        self.assertIn("bis zu 10 Überlaufwellen (Form-A-Standard: 2)", text)
        self.assertIn("Untergrenze", text)
        self.assertEqual(h["round_bs1_ms"], 73.84)                              # the figures stay numbers

    def test_the_english_view_keeps_the_point(self):
        h = hint(TWO)
        self.assertEqual(Z.localize(copy.deepcopy(h), "en"), h)
        self.assertIn("73.8 ms", h["saetze"][0])

    def test_the_ms_comma_touches_only_a_figure_before_ms(self):
        self.assertEqual(Z._de_gib("Runde 73.8 ms, Faktor 0.82"), "Runde 73,8 ms, Faktor 0.82")
        self.assertEqual(Z._de_gib("--x 1.5"), "--x 1.5")


HARNESS = r"""
const STATIC = process.argv[1], CASE = JSON.parse(process.argv[2]);
global.RigI18n = { lang: () => CASE.lang, loc: () => (CASE.lang === "de" ? "de-DE" : "en-US") };
const mod = require(STATIC + "/planer_ziel.js");
console.log(JSON.stringify({ plan: mod.renderPlan(CASE.plan) }));
"""


def render(plan, lang="en"):
    p = subprocess.run([T.NODE, "-e", HARNESS, STATIC, json.dumps({"plan": plan, "lang": lang})], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads(p.stdout.strip().splitlines()[-1])["plan"]


@unittest.skipUnless(T.NODE, "node missing")
class Page(unittest.TestCase):
    def plan(self, d):
        doc = copy.deepcopy(T.plan(2, "qwen38-nf-int4", "flip"))               # the recorded answer of the page, with the verdict this tree computes
        doc["verdikt"]["langsam"] = hint(d)
        return doc

    def test_the_result_shows_the_hint_after_the_verdict(self):
        h = render(self.plan(TWO))
        self.assertIn("pz-verdict slow", h)
        self.assertIn("startable, slow (estimate)", h)
        self.assertIn("SLOW-ESTIMATE", h)
        self.assertIn("73.8 ms per bs1 decode round", h)
        self.assertLess(h.index("pz-verdict ok"), h.index("pz-verdict slow"))

    def test_no_hint_no_box(self):
        self.assertNotIn("pz-verdict slow", render(self.plan(THREE)))
        doc = copy.deepcopy(T.plan(3, "qwen38-nf-int4", "flip"))               # an answer recorded before A2 (no ``langsam`` key at all)
        self.assertNotIn("pz-verdict slow", render(doc))

    def test_german_view_server_texts_behind_the_skip_marker(self):
        doc = Z.localize(self.plan(TWO), "de")
        h = render(doc, "de")
        self.assertIn('<span data-i18n="skip">startbar, langsam (Hochrechnung)</span>', h)
        self.assertIn("73,8 ms", h)


CENSUS = "/spinning/gpu-arb/weg2/census/xchg_census_fnFL2_graph.json"
REPO_PY = os.path.join(REPO_ROOT, "python")


@unittest.skipUnless(os.path.exists(CENSUS) and os.path.isdir(os.path.join(REPO_PY, "flliper")), "the real chain needs the rig box (census, models) and the planner tree")
class RealChain(unittest.TestCase):
    """The behaviour on the REAL chain: ``ZielPlaner.plan`` -> ``ProfilEditor.propose`` -> oracle child (planner + launcher dry run) -> ``slow_hint``.
    NF INT4 flip on 2 x RTX 3090 carries the hint; 3 x RTX 3090 and the reference rig (5090 + 2 x 3080) do not."""

    @classmethod
    def setUpClass(cls):
        from rigdash import kartenplan as K
        from rigdash import profil as P
        from rigdash import profile_oracle as ORA

        ck = os.path.join(REPO_ROOT, "test", "registered", "unit", "pdflip", "fixtures", "planer_1006", "checkpoints")
        snaps = {n: os.path.join(ck, n) for n in sorted(os.listdir(ck)) if os.path.isfile(os.path.join(ck, n, "manifest.json"))}
        cls.oracle = ORA.OracleService(REPO_PY, python=sys.executable, request_extra={"snapshots": snaps})
        cls.tmp = tempfile.mkdtemp(prefix="a2_chain_")
        ed = P.ProfilEditor(kartenplaner=K.Kartenplaner(tree=REPO_PY), release_dir=T.RELEASE_DIR, user_dir=os.path.join(cls.tmp, "usr"), tree=REPO_PY,
                            catalog_file=os.path.join(os.path.dirname(HERE), "profil_data", "catalog.json"), oracle=cls.oracle,
                            hardware=lambda: {"ok": False})
        cls.z = Z.ZielPlaner(ed)

    @classmethod
    def tearDownClass(cls):
        cls.oracle.close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def verdict(self, cards):
        p = self.z.plan({"model": "qwen38-nf-int4", "layout": "flip", "cards": cards})
        self.assertEqual(p["status"], "ok", p.get("hinweise"))
        return p["verdikt"]

    def test_two_rtx3090_carry_the_hint(self):
        v = self.verdict([{"card": "rtx3090-24", "count": 2}])
        self.assertIn(v["outcome"], ("geht", "ok", "ok_with_force"))           # startable: a hint, not a lock
        h = v["langsam"]
        self.assertIsNotNone(h)
        self.assertEqual(h["code"], "SLOW-ESTIMATE")
        self.assertGreater(h["round_bs1_ms"], Z.SLOW_BS1_ROUND_MS)
        self.assertGreater(h["waves"], h["waves_default"])
        self.assertIsNotNone(h["wave_fallback"])

    def test_three_rtx3090_carry_none(self):
        self.assertIsNone(self.verdict([{"card": "rtx3090-24", "count": 3}])["langsam"])

    def test_the_reference_rig_carries_none(self):
        self.assertIsNone(self.verdict([{"card": "rtx5090-32", "count": 1}, {"card": "rtx3080-20", "count": 2}])["langsam"])


if __name__ == "__main__":
    unittest.main()
