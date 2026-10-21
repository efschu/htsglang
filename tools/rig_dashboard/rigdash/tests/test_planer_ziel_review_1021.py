"""PLANER-ZIELBILD 1021 -- REVIEW tests (independent reviewer, desk/flliper-planer-zielbild-1021 @ 5382694e1d).

Two kinds of test here:

* GREEN pins the author's suite did not cover: the rig golden for EVERY model/layout the rig runs today (not only 27B INT8 flip), the
  "This machine" shortcut in NVML order (3080, 5090, 3080) still asks the oracle on the measured hardware, today's three rig containers
  (27B INT8 flip, 27B NVFP4 dual, NF flip) are never blocked by the layout pre-check with the inputs of their own release profiles.
* AUFLAGEN A1-A6b (``AuflagenClosed``): each one is a defect found in the review, written as the behaviour the fix reaches.  They were
  ``expectedFailure`` until the fixes landed (then "unexpected success", red; the markers came off and the tests stay as pins).  Each names its Auflage id
  (report /spinning/gpu-arb/deskq/done/planer-zielbild-1021-review.md).

Measured reference for A4 (no estimate): rig 27B INT8 flip, boot dkr27bbar1drei10210113 (21.10. 01:13Z, image c77da494f9), D log
``max_total_num_tokens`` per rank 245185 + 252097 + 281698 = 778980 tokens D pool (uneven DCP: the rank shares ADD UP, memory
d-kv-gesamtpool-nicht-je-rang).
"""

import copy
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from rigdash import planer_ziel as Z  # noqa: E402
from rigdash import profil as P  # noqa: E402

FIX = os.path.join(HERE, "fixtures", "planer_ziel_1021")
RIG = [{"card": "rtx5090-32", "count": 1}, {"card": "rtx3080-20", "count": 2}]
RIG_NVML_ORDER = [{"card": "rtx3080-20", "count": 2}, {"card": "rtx5090-32", "count": 1}]     # what katalog()["machine"] yields on the rig
X3090 = [{"card": "rtx3090-24", "count": 3}]
#: measured D pool of the rig, 27B INT8 flip (see module head)
RIG_27B_INT8_D_POOL_MEASURED = 245185 + 252097 + 281698


def hw_rig(measured_rates: bool = True):
    def c(i, name, mib, cc):
        row = {"nvml_index": i, "uuid": "GPU-%d" % i, "name": name, "cc": cc, "vram_total_mib": {"v": mib, "src": "NVML"}}
        if measured_rates:
            row["gemm_tflops"] = {"bf16": {"v": 50.0, "src": "gemessen", "at": 1.0, "probe": "card_probe-x.json"}}
        return row
    return {"schema": "flliper.hardware/1", "cards": [c(0, "NVIDIA GeForce RTX 3080", 20480, [8, 6]), c(1, "NVIDIA GeForce RTX 5090", 32607, [12, 0]),
                                                        c(2, "NVIDIA GeForce RTX 3080", 20480, [8, 6])]}


class StubEditor:
    _hw_cards_match = staticmethod(P.ProfilEditor._hw_cards_match)

    def __init__(self, hw=None, res=None):
        self.hw, self.res, self.calls = hw, res, []

    def _hardware_profile(self):
        return self.hw

    def propose(self, req):
        self.calls.append(copy.deepcopy(req))
        return copy.deepcopy(self.res)


def canned(name):
    with open(os.path.join(FIX, name), encoding="utf-8") as fh:
        return json.load(fh)


def fields(out):
    return {r["key"]: r for g in out["felder"] for r in g["zeilen"]}


def plan(fixture, model, layout, cards, hw=None):
    ed = StubEditor(hw, canned(fixture))
    out = Z.ZielPlaner(ed).plan({"model": model, "layout": layout, "cards": cards})
    return out, ed


def flag_value(res, flag):
    """The value of ``flag`` anywhere in the launcher's group flags or launch argv of a canned oracle answer (first hit)."""
    gf = ((res.get("verdict") or {}).get("plan") or {}).get("group_flags") or {}
    for g in ("P", "D", "front"):
        if (gf.get(g) or {}).get(flag):
            return gf[g][flag]
    argv = json.dumps(res.get("launch") or {})
    key = '"%s": "' % flag
    if key in argv:
        return argv.split(key, 1)[1].split('"', 1)[0]
    i = argv.find('"%s", "' % flag)
    if i >= 0:
        return argv[i:].split('", "', 1)[1].split('"', 1)[0]
    return None


# ---------------------------------------------------------------------------------------------------------------------------- GREEN pins
class RigGoldenEveryLayout(unittest.TestCase):
    """The rig golden is the oracle REQUEST: for every model/layout the page offers on the rig, the request is the one of the old page for
    'This rig' (release basis of the model/layout, goals empty, the NVML cards of the machine -> measured hardware)."""

    def test_every_possible_layout_on_the_rig_asks_the_old_request(self):
        res = canned("res_27b_int8_flip_rig.json")
        for cards in (RIG, RIG_NVML_ORDER):
            for m in Z.MODELS:
                for lid, prof in m["profiles"].items():
                    if not prof:
                        continue
                    ed = StubEditor(hw_rig(), res)
                    out = Z.ZielPlaner(ed).plan({"model": m["id"], "layout": lid, "cards": cards})
                    self.assertEqual(out["status"], "ok", (m["id"], lid, out.get("hinweise")))
                    self.assertEqual(out["hw"]["quelle"], "gemessen", (m["id"], lid))
                    req = ed.calls[0]
                    form = {"solo": "tp", "flip": "flip", "dual": "dual"}[lid]
                    self.assertEqual((req["basis"], req["form"], req["goals"]), ({"kind": "release", "name": prof}, form, {}), (m["id"], lid))
                    self.assertTrue(P.ProfilEditor._hw_cards_match(hw_rig(), [{"entry": Z.CAT.card(c["card"])} for c in req["inventory"]]))

    def test_the_machine_shortcut_in_nvml_order_is_measured(self):
        z = Z.ZielPlaner(StubEditor(hw_rig()))
        mine = z.katalog()["machine"]["cards"]
        self.assertEqual(z.layouts({"model": "qwen38-27b-int8", "cards": mine})["hw"]["quelle"], "gemessen")


class TodaysRigContainersAreNeverBlocked(unittest.TestCase):
    """The three containers of the rig (memory drei-container-portainer-1020) with the seats/ceiling of their own release profiles."""

    def test_27b_int8_flip_nvfp4_dual_nf_flip(self):
        z = Z.ZielPlaner(StubEditor(hw_rig()))
        for model, layout, seats, tokens in (("qwen38-27b-int8", "flip", 6, 262144), ("qwen38-27b-nvfp4", "dual", 6, 262144),
                                             ("qwen38-nf-int4", "flip", 6, 262144), ("qwen38-27b-nvfp4", "flip", 6, 262144)):
            lay = {l["id"]: l for l in z.layouts({"model": model, "cards": RIG, "seats": seats, "tokens": tokens})["layouts"]}
            self.assertTrue(lay[layout]["possible"], (model, layout, lay[layout]["why"]))


class SharedPoolAndSm86(unittest.TestCase):
    def test_seats_never_multiply_the_ceiling_in_the_pool_rows(self):
        out, _ = plan("res_27b_int8_flip_3090.json", "qwen38-27b-int8", "flip", X3090)
        f = fields(out)
        self.assertEqual(f["tokens"]["wert"], "262144")
        self.assertNotEqual(f["p_pool"]["wert"], str(6 * 262144))

    def test_nvfp4_is_offered_on_every_sm86_card_of_the_catalog(self):
        z = Z.ZielPlaner(StubEditor())
        for card in ("rtx3090-24", "rtx3080-20", "rtx3090ti-24"):
            lay = z.layouts({"model": "qwen38-27b-nvfp4", "cards": [{"card": card, "count": 3}]})
            self.assertFalse(any("fp4" in w.lower() for l in lay["layouts"] for w in l["why"]), card)


# ------------------------------------------------------------------------------------------------------------------------ AUFLAGEN (red)
class AuflagenClosed(unittest.TestCase):
    """The review conditions A1-A6b, closed (the ``expectedFailure`` markers came off when the fixes landed; the tests stay as pins)."""

    def test_A1_nf_decode_pool_is_the_argv_value_not_an_expert_blind_estimate(self):
        """A1: NF D has --max-total-tokens 262144 in the argv; the page shows an 'estimate' of 4.28 M tokens that ignores the resident
        experts (resident fraction 0.58/0.81 of 58 GiB experts sit in the same D budgets; memory GRUNDGESETZ VRAM=EXPERTEN)."""
        res = canned("res_nf_flip_3090.json")
        out, _ = plan("res_nf_flip_3090.json", "qwen38-nf-int4", "flip", X3090)
        want = flag_value(res, "--max-total-tokens")
        self.assertEqual(want, "262144")
        self.assertEqual(fields(out)["d_pool"]["wert"], want)

    def test_A2_prefill_pool_is_what_the_launcher_applies(self):
        """A2: the P pool row reads plan.pp_cut.chosen_pool; for NF that is 1496658 while P runs with --max-total-tokens 262144."""
        res = canned("res_nf_flip_3090.json")
        out, _ = plan("res_nf_flip_3090.json", "qwen38-nf-int4", "flip", X3090)
        self.assertEqual(fields(out)["p_pool"]["wert"], ((res["verdict"]["plan"]["group_flags"]["P"]) or {}).get("--max-total-tokens"))

    def test_A3_dual_pools_are_the_dual_kv_flags(self):
        """A3: dual sizes its pools with --dual-p-kv-max-tokens / --dual-d-kv-max-tokens (266240 / 1048576 in this answer); the page shows
        pp_cut.chosen_pool 681833 for P and an estimate 1100595 for D."""
        res = canned("res_nvfp4_dual_3090.json")
        out, _ = plan("res_nvfp4_dual_3090.json", "qwen38-27b-nvfp4", "dual", X3090)
        f = fields(out)
        self.assertEqual(f["p_pool"]["wert"], flag_value(res, "--dual-p-kv-max-tokens"))
        self.assertEqual(f["d_pool"]["wert"], flag_value(res, "--dual-d-kv-max-tokens"))

    def test_A4_rig_decode_pool_matches_the_measured_rig(self):
        """A4 (golden): on OUR rig (measured hardware) the D pool row must be the measured pool (778980 tokens, boot 21.10. 01:13Z) or an
        estimate within 10 % of it; today it is 962204 (+23.5 %), because the estimate subtracts only weights + state from the D budgets
        (no graphs, draft KV, L1.5 posts, activations)."""
        out, _ = plan("res_27b_int8_flip_rig.json", "qwen38-27b-int8", "flip", RIG, hw_rig())
        shown = int(fields(out)["d_pool"]["wert"])
        self.assertLessEqual(abs(shown - RIG_27B_INT8_D_POOL_MEASURED) / RIG_27B_INT8_D_POOL_MEASURED, 0.10, shown)

    def test_A5_measured_only_when_the_rates_are_measured(self):
        """A5: the chip says 'measured' (text: 'measured rates') as soon as the chosen cards equal the NVML cards of the machine -- also on a
        machine whose hardware profile was only CAPTURED (NVML names/sizes) and never rate-measured (every foreign 3 x 3090 box on day 1)."""
        z = Z.ZielPlaner(StubEditor(hw_rig(measured_rates=False)))
        hw = z.layouts({"model": "qwen38-27b-int8", "cards": RIG})["hw"]
        self.assertNotEqual(hw["quelle"], "gemessen", hw)

    def test_A6_the_pre_check_overhead_is_a_lower_bound_of_its_own_dry_run(self):
        """A6: the 'rough NECESSARY condition' blocks without force; its per-card constant (6000 MiB) is the P residue of the 3 x 3090 dry
        run (24576 - 18584, contains the user reserve and the L1.5 post).  The D residue of the SAME dry run is 24576 - 19792 = 4784 MiB.
        A necessary condition may only use the smaller one (or the measured fixed posts), never a reserve."""
        res = canned("res_27b_int8_flip_3090.json")
        gf = res["verdict"]["plan"]["group_flags"]
        resid = [24576 - int(x) for g in ("P", "D") for x in str(gf[g]["--rank-gpu-memory-mib"]).split(",")]
        self.assertLessEqual(Z.OVERHEAD_PER_CARD_MIB, min(resid))

    def test_A6_three_3080_20gb_nvfp4_dual_is_not_hard_blocked_by_the_reserve(self):
        """A6 consequence: 3 x RTX 3080 20 GB + 27B NVFP4 dual (6 seats, ceiling 262144) is 'blocked' by the pre-check, short by ~3 GiB --
        a status that 'cannot be forced here' and never reaches the oracle.  With the D residue of the same 3 x 3090 dry run (4784 MiB/card)
        instead of the P residue (6000) the margin is positive.  Our rig runs this very model in dual with two of these 3080s; whether it
        fits must be the planner's answer, not a reserve inside a 'necessary' condition."""
        z = Z.ZielPlaner(StubEditor())
        lay = {l["id"]: l for l in z.layouts({"model": "qwen38-27b-nvfp4", "cards": [{"card": "rtx3080-20", "count": 3}]})["layouts"]}
        self.assertTrue(lay["dual"]["possible"], (lay["dual"]["status"], lay["dual"]["margin_mib"], lay["dual"]["why"]))


# ------------------------------------------------------------------------------------------------------- Nachbesserung: more pins for the fixes
class FixesBeyondTheReviewTests(unittest.TestCase):
    def test_A4_estimate_without_a_flag_uses_the_fixed_posts_and_is_shown_as_an_estimate(self):
        out, _ = plan("res_27b_int8_flip_rig.json", "qwen38-27b-int8", "flip", RIG, hw_rig())
        r = fields(out)["d_pool"]
        self.assertTrue(r["schaetzung"])
        self.assertIn("fixed posts", r["warum"])
        self.assertIn("778980", r["warum"])                       # the measured pool the constant is calibrated on is named

    def test_the_calibration_constant_reproduces_the_measured_rig_pool_exactly(self):
        """D_FIXED_PER_CARD_MIB is derived from the boot's own numbers (budgets 27792+17384+17168, weights, draft, state): back-substituted it gives the pool."""
        s = Z.load_steckbrief()["qwen38-27b-int8"]
        free = (27792 + 17384 + 17168) - (s["weights_total_mib"] + s["draft_mib"]) - 6 * s["state_mib_per_seat_bf16"] - 3 * Z.D_FIXED_PER_CARD_MIB
        self.assertLess(abs(free * (1 << 20) / s["kv_bytes_per_token_fp8"] - RIG_27B_INT8_D_POOL_MEASURED) / RIG_27B_INT8_D_POOL_MEASURED, 0.001)

    def test_A1_without_the_flag_the_nf_estimate_counts_the_resident_experts(self):
        res = canned("res_nf_flip_3090.json")
        D = res["verdict"]["plan"]["group_flags"]["D"]
        D.pop("--max-total-tokens")
        st = Z.load_steckbrief()["qwen38-nf-int4"]
        est = Z.ZielPlaner._d_pool_estimate(st, "flip", [Z.CAT.card("rtx3090-24")] * 3, D, 6)
        blind = int((sum(int(x) for x in D["--rank-gpu-memory-mib"].split(",")) - st["weights_dense_mib"] - st["lm_head_mib"] - st["draft_mib"] - 6 * st["state_mib_per_seat_bf16"])
                    * (1 << 20) / st["kv_bytes_per_token_fp8"])
        self.assertLess(est["tokens"], blind / 3)                  # experts take most of the budgets: the blind figure was 4.3 M
        self.assertIn("resident experts", est["why"])
        D.pop("--rank-moe-resident-fraction")
        self.assertIsNone(Z.ZielPlaner._d_pool_estimate(st, "flip", [Z.CAT.card("rtx3090-24")] * 3, D, 6))   # no residency known: no estimate

    def test_pool_rows_name_the_flag_they_show(self):
        for fx, model, layout, p_flag, d_flag in (("res_nf_flip_3090.json", "qwen38-nf-int4", "flip", "--max-total-tokens", "--max-total-tokens"),
                                                  ("res_nvfp4_dual_3090.json", "qwen38-27b-nvfp4", "dual", "--dual-p-kv-max-tokens", "--dual-d-kv-max-tokens")):
            out, _ = plan(fx, model, layout, X3090)
            f = fields(out)
            self.assertEqual((f["p_pool"]["roh"], f["d_pool"]["roh"]), (p_flag, d_flag))
            self.assertFalse(f["p_pool"]["schaetzung"] or f["d_pool"]["schaetzung"])

    def test_B1_the_shortcut_is_a_named_reference_rig_not_ours(self):
        labels = [x["label"] for x in Z.SHORTCUTS]
        self.assertTrue(any(l.startswith("Reference rig") for l in labels), labels)
        self.assertFalse(any("Our rig" in l or "our rig" in l for l in labels), labels)

    def test_B4_supported_card_counts_come_from_the_topology(self):
        class Kp:
            @staticmethod
            def _mods():
                return None, object()

        class Ed(StubEditor):
            kp = Kp()

            def __init__(self, ok):
                super().__init__()
                self.ok = ok

            def _topology_verdict(self, n, tp, notes):
                if n == 6:
                    notes.append("not checked")                    # a count that cannot be checked is not offered
                    return None
                if n in self.ok:
                    return None
                return "HW-COUNT: %d cards: not proven" % n

        z = Z.ZielPlaner(Ed({2, 3, 4, 6}))
        self.assertEqual(z.supported_n(), (2, 3, 4))
        self.assertEqual(z.katalog()["supported_n"], [2, 3, 4])
        lay = {l["id"]: l for l in z.layouts({"model": "qwen38-27b-int8", "cards": [{"card": "rtx3090-24", "count": 4}]})["layouts"]}
        self.assertNotEqual(lay["flip"]["status"], "n_unsupported")
        lay = {l["id"]: l for l in z.layouts({"model": "qwen38-27b-int8", "cards": [{"card": "rtx3090-24", "count": 5}]})["layouts"]}
        self.assertEqual(lay["flip"]["status"], "n_unsupported")
        self.assertEqual(Z.ZielPlaner(StubEditor()).supported_n(), Z.SUPPORTED_N)        # no planner gate to ask: the fallback

    def test_B5_nf_weights_are_the_sum_of_their_parts(self):
        k = {m["id"]: m for m in Z.ZielPlaner(StubEditor()).katalog()["models"]}
        st = Z.load_steckbrief()["qwen38-nf-int4"]
        self.assertEqual(k["qwen38-nf-int4"]["weights_mib"], st["weights_dense_mib"] + st["lm_head_mib"] + st["experts"]["total_mib"])
        self.assertEqual(k["qwen38-27b-int8"]["weights_mib"], Z.load_steckbrief()["qwen38-27b-int8"]["weights_total_mib"])


class GermanPage(unittest.TestCase):
    """Review B2: the server answers in German when the page asks (``lang: de``): complete sentences, no half-translated mixtures."""

    ENGLISH = ("the ", " of ", " and ", " with ", "must fit", "work as", "does not fit", "rough")

    def _texts(self, o, out):
        if isinstance(o, str):
            out.append(o)
        elif isinstance(o, list):
            for x in o:
                self._texts(x, out)
        elif isinstance(o, dict):
            for k, v in o.items():
                if k not in ("roh", "intern", "key", "id"):
                    self._texts(v, out)
        return out

    def test_layouts_and_calculation_lines_are_german(self):
        z = Z.ZielPlaner(StubEditor(hw_rig()))
        for model in ("qwen38-27b-int8", "qwen38-27b-nvfp4", "qwen38-nf-int4"):
            out = z.layouts({"model": model, "cards": RIG, "lang": "de"})
            texts = self._texts([[l["satz"], l["why"], l["calc"]] for l in out["layouts"]] + [out["note"], out["hw"]["text"]], [])
            for t in texts:
                self.assertFalse(any(w in t.lower() for w in self.ENGLISH), t)
        l = {x["id"]: x for x in z.layouts({"model": "qwen38-27b-nvfp4", "cards": RIG, "lang": "de"})["layouts"]}["flip"]
        self.assertTrue(l["calc"][0].startswith("Anzahl Karten: 3"), l["calc"][0])
        self.assertIn("Gewichte:", l["calc"][1])
        self.assertTrue(any("EIN Request an der Decke" in x for x in l["calc"]))

    def test_english_and_unknown_languages_are_unchanged(self):
        z = Z.ZielPlaner(StubEditor(hw_rig()))
        en = z.layouts({"model": "qwen38-27b-int8", "cards": RIG})
        self.assertEqual(en, z.layouts({"model": "qwen38-27b-int8", "cards": RIG, "lang": "en"}))
        self.assertEqual(en, z.layouts({"model": "qwen38-27b-int8", "cards": RIG, "lang": "fr"}))
        self.assertTrue(en["layouts"][0]["calc"][0].startswith("cards: 3 cards"))

    def test_plan_rows_are_german_and_the_raw_flags_and_the_start_profile_are_not_touched(self):
        res = canned("res_27b_int8_flip_rig.json")
        z = Z.ZielPlaner(StubEditor(hw_rig(), res))
        de = z.plan({"model": "qwen38-27b-int8", "layout": "flip", "cards": RIG, "lang": "de"})
        en = z.plan({"model": "qwen38-27b-int8", "layout": "flip", "cards": RIG})
        self.assertEqual(de["intern"], en["intern"])
        fd, fe = fields(de), fields(en)
        self.assertEqual(sorted(fd), sorted(fe))
        for k in fd:
            self.assertEqual((fd[k]["roh"], fd[k]["wert"], fd[k]["je_karte"]), (fe[k]["roh"], fe[k]["wert"], fe[k]["je_karte"]), k)
        self.assertEqual(fd["d_pool"]["label"], "KV-Pool, Decode-Gruppe (von allen Sitzen geteilt)")
        self.assertIn("Festposten 3 x 1926 MiB", fd["d_pool"]["warum"])
        self.assertEqual(fd["p_pool"]["quelle"], "Launcher-Trockenlauf (--max-total-tokens)")
        self.assertEqual(de["verdikt"]["text"], "Der Trockenlauf des Launchers geht ohne Force durch.")
        self.assertEqual(de["verdikt"]["outcome"], en["verdikt"]["outcome"])

    def test_every_german_text_uses_only_placeholders_its_english_key_has(self):
        import re as _re
        de = json.load(open(Z.DE_FILE, encoding="utf-8"))["de"]
        for en, g in de.items():
            n = len(_re.findall(r"%[-.\d]*[dsrf]", en.replace("%%", "")))
            used = [int(x) for x in _re.findall(r"\{(\d+)\}", g)]
            self.assertTrue(all(i < n for i in used), (en, g))
            self.assertNotIn("%s", g, en)

    def test_the_launcher_outcome_geht_has_its_text(self):
        """The dry run of OUR rig ends with outcome ``geht``: the verdict line must say so, not "No launcher verdict (geht)"."""
        out, _ = plan("res_27b_int8_flip_rig.json", "qwen38-27b-int8", "flip", RIG, hw_rig())
        self.assertEqual(out["verdikt"]["text"], "The launcher's dry run passes without force.")

    def test_the_katalog_in_german(self):
        k = Z.ZielPlaner(StubEditor(hw_rig())).katalog("de")
        self.assertEqual(k["layouts"][0]["name"], "Solo (nur TP)")
        self.assertTrue(k["layouts"][0]["satz"].startswith("Alle Karten"))
        self.assertTrue(any(x["label"].startswith("Referenz-Rig") for x in k["shortcuts"]))
        self.assertEqual([m["id"] for m in k["models"]], [m["id"] for m in Z.MODELS])


if __name__ == "__main__":
    unittest.main()
