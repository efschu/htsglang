"""PLANER-ZIELBILD 1021: the planner page in the target picture of the user (21.10.): model + cards -> only the possible layouts (solo / flip / dual) ->
max seats + max tokens per seat -> the rest computed.

Gepinnt:
  * the model table (``MODELS``) names release profiles that exist, every model has a fact sheet row, the layouts map to forms ``ProfilEditor.propose`` serves;
  * ``layouts()``: the table of statuses for the rig, 3 x RTX 3090 and a foreign card, for every model; NVFP4 is NOT excluded on sm86 (our own kernels; user rebuke
    21.10.); a card count outside SUPPORTED_N (1 and 8) is "not supported yet" for every layout, 2 and 4 carry the named blocker METAL-UNPROVEN; an sm89 card is blocked by the image's arch gate; a calculation line for every impossible
    layout; the KV ceiling is never multiplied by the seats;
  * ``plan()``: GOLDEN of the rig -- the oracle request is exactly what the old page sent for this machine (``goals`` empty, the cards are the NVML cards of the machine,
    so the proposal runs on the measured hardware); seats/tokens go into the goals only when the user set them; a not-possible layout never asks the oracle;
    the internal profile names never reach the visible part of the answer;
  * the page: the planner card, the expert fold around the old editor, the script and the routes.
"""

import copy
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from rigdash import kartenplan_catalog as CAT  # noqa: E402
from rigdash import planer_ziel as Z  # noqa: E402
from rigdash import profil as P  # noqa: E402

STATIC = os.path.join(os.path.dirname(HERE), "static")
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
RELEASE_DIR = os.path.join(REPO_ROOT, "docker", "flliper", "profiles_release")
FIX = os.path.join(HERE, "fixtures", "planer_ziel_1021")

RIG = [{"card": "rtx5090-32", "count": 1}, {"card": "rtx3080-20", "count": 2}]
X3090 = [{"card": "rtx3090-24", "count": 3}]
MODELS = ("qwen38-27b-int8", "qwen38-27b-nvfp4", "qwen38-nf-int4")


def hw_rig(measured=True):
    """The hardware profile of our rig (NVML order 3080, 5090, 3080).  ``measured`` = the rates carry ``src: gemessen`` nodes as the measuring run writes them
    (``mem_gbs`` / ``compute``); without them the profile is only CAPTURED (NVML names and sizes)."""
    def c(i, name, mib, cc):
        row = {"nvml_index": i, "uuid": "GPU-%d" % i, "name": name, "cc": cc, "vram_total_mib": {"v": mib, "src": "NVML"},
               "mem_gbs": {"nameplate": {"v": 760.0, "src": "NVML"}, "nominal": {"v": 760.0, "src": "Datasheet"}}}
        if measured:
            row["mem_gbs"]["read"] = {"v": 700.0, "src": "gemessen"}
            row["compute"] = {"bf16": {"v": 50.0, "src": "gemessen"}}
        return row
    return {"schema": "flliper.hardware/1", "cards": [c(0, "NVIDIA GeForce RTX 3080", 20480, [8, 6]), c(1, "NVIDIA GeForce RTX 5090", 32607, [12, 0]),
                                                        c(2, "NVIDIA GeForce RTX 3080", 20480, [8, 6])]}


class StubEditor:
    """What ``ZielPlaner`` needs of ``ProfilEditor``: the hardware profile, the match rule (the real one), ``propose`` (records, answers canned)."""

    _hw_cards_match = staticmethod(P.ProfilEditor._hw_cards_match)

    def __init__(self, hw=None, res=None):
        self.hw, self.res, self.calls = hw, res, []

    def _hardware_profile(self):
        return self.hw

    def propose(self, req):
        self.calls.append(copy.deepcopy(req))
        return copy.deepcopy(self.res)


def zp(hw=None, res=None):
    ed = StubEditor(hw, res)
    return Z.ZielPlaner(ed), ed


def by_id(lay):
    return {l["id"]: l for l in lay["layouts"]}


class Pins(unittest.TestCase):
    def test_the_models_name_release_profiles_that_exist(self):
        for m in Z.MODELS:
            for lid, prof in m["profiles"].items():
                if prof:
                    self.assertTrue(os.path.isfile(os.path.join(RELEASE_DIR, prof + ".env")), (m["id"], lid, prof))
        self.assertEqual([m["id"] for m in Z.MODELS], list(MODELS))

    def test_every_model_has_a_fact_sheet_row_from_the_model_profile(self):
        steck = Z.load_steckbrief()
        for m in Z.MODELS:
            s = steck[m["id"]]
            for k in ("weights_total_mib", "weights_dense_mib", "lm_head_mib", "kv_bytes_per_token_fp8", "state_mib_per_seat_bf16", "attn_layers", "linear_layers", "draft_mib"):
                self.assertTrue(s.get(k), (m["id"], k))
        self.assertTrue(steck["qwen38-nf-int4"]["experts"]["n"] == 512)
        self.assertIsNone(steck["qwen38-27b-nvfp4"]["experts"])

    def test_the_layouts_map_to_forms_the_editor_serves(self):
        self.assertEqual({l["form"] for l in Z.LAYOUTS}, {"tp", "flip", "dual"})
        self.assertLessEqual({l["form"] for l in Z.LAYOUTS}, set(P.ProfilEditor.FORMS))
        self.assertEqual(Z.SUPPORTED_N[:3], (2, 3, 4))    # NCARDS 1021: 12 release profiles + a dry-run golden each (later counts are added behind them)
        self.assertEqual(Z.PROVEN_N, (3,))                # the only count a release boot has proven (topology.PROVEN_CARD_COUNTS)

    def test_page_routes_and_markup(self):
        html = open(os.path.join(STATIC, "index.html"), encoding="utf-8").read()
        self.assertIn('id="pz-root"', html)
        self.assertLess(html.index('<script src="planer_ziel.js">'), html.index('<script src="profil.js">'))
        self.assertIn('<details id="pf-expert"', html)                       # the old editor lives on, behind the expert fold
        self.assertLess(html.index('id="pz-card"'), html.index('id="pf-card"'))
        self.assertIn("window.RigPlaner.show()", html)
        srv = open(os.path.join(os.path.dirname(HERE), "server.py"), encoding="utf-8").read()
        for needle in ('"/planer_ziel.js"', "/api/profil/planer-katalog", "/api/profil/planer-layouts", "/api/profil/planer-plan"):
            self.assertIn(needle, srv)


class Katalog(unittest.TestCase):
    def test_models_by_name_and_cards_with_origin(self):
        k = zp()[0].katalog()
        self.assertEqual([m["name"] for m in k["models"]], ["Qwen3.8-27B INT8", "Qwen3.8-27B NVFP4", "Qwen3.8-Flash-Next INT4"])
        self.assertEqual([m["layouts"] for m in k["models"]], [["solo", "flip"], ["solo", "flip", "dual"], ["solo", "flip"]])
        self.assertEqual(k["supported_n"], list(Z.SUPPORTED_N))
        self.assertEqual([c["id"] for c in k["cards"] if c["preset"]], ["rtx5090-32", "rtx3090-24", "rtx3080-20"])
        self.assertIsNone(k["machine"])                                         # no hardware profile: no "this machine" shortcut

    def test_the_machine_shortcut_is_the_hardware_profile(self):
        k = zp(hw_rig())[0].katalog()
        self.assertEqual(sorted((r["card"], r["count"]) for r in k["machine"]["cards"]), [("rtx3080-20", 2), ("rtx5090-32", 1)])

    def test_no_internal_profile_name_is_visible(self):
        text = json.dumps(zp(hw_rig())[0].katalog())
        for m in Z.MODELS:
            for prof in m["profiles"].values():
                if prof:
                    self.assertNotIn('"%s"' % prof, text)
        self.assertNotIn("profiles", text)


class Layouts(unittest.TestCase):
    def status(self, model, cards, **kw):
        lay = zp()[0].layouts(dict({"model": model, "cards": cards}, **kw))
        return {k: v["status"] for k, v in by_id(lay).items()}

    def test_the_table_for_the_rig_and_three_3090(self):
        for cards in (RIG, X3090):
            self.assertEqual(self.status("qwen38-27b-int8", cards), {"solo": "ok", "flip": "ok", "dual": "blocked"})
            self.assertEqual(self.status("qwen38-27b-nvfp4", cards), {"solo": "ok", "flip": "ok", "dual": "ok"})
            self.assertEqual(self.status("qwen38-nf-int4", cards), {"solo": "ok", "flip": "ok", "dual": "blocked"})

    def test_nvfp4_runs_on_sm86_our_own_kernels(self):
        lay = by_id(zp()[0].layouts({"model": "qwen38-27b-nvfp4", "cards": X3090}))
        for l in lay.values():
            self.assertTrue(l["possible"], l["id"])
            self.assertNotIn("fp4", " ".join(l["why"]).lower())

    def test_card_counts_outside_two_to_four_are_not_supported_and_say_why(self):
        for n in (1, 8):
            lay = by_id(zp()[0].layouts({"model": "qwen38-27b-nvfp4", "cards": [{"card": "rtx3090-24", "count": n}]}))
            for l in lay.values():
                self.assertEqual(l["status"], "n_unsupported", (n, l["id"]))
                self.assertFalse(l["possible"])
                self.assertIn("not supported yet", " ".join(l["why"]))
                self.assertIn("%d card" % n, " ".join(l["why"]))

    def test_two_and_four_cards_are_calculated_and_say_that_no_boot_proved_them(self):
        for n in (2, 4):
            lay = by_id(zp()[0].layouts({"model": "qwen38-27b-nvfp4", "cards": [{"card": "rtx3090-24", "count": n}]}))
            for l in lay.values():
                self.assertEqual(l["proof"], "dry_run")
                self.assertIn("METAL-UNPROVEN", " ".join(l["why"]))                  # the named blocker is visible in every layout of the count
                if l["id"] != "dual" or n == 4:                                      # (2 x 24 GB cannot hold NVFP4 dual by the rough calculation: blocked, with its numbers)
                    self.assertIn(l["status"], ("ok", "tight"), (n, l["id"]))
                else:
                    self.assertEqual(l["status"], "blocked")
                    self.assertIn("short by", " ".join(l["why"]))
        for l in by_id(zp()[0].layouts({"model": "qwen38-27b-nvfp4", "cards": X3090})).values():
            self.assertEqual(l["proof"], "metal")
            self.assertNotIn("METAL-UNPROVEN", " ".join(l["why"]))

    def test_an_sm89_card_is_blocked_by_the_arch_gate_of_the_image(self):
        lay = by_id(zp()[0].layouts({"model": "qwen38-nf-int4", "cards": [{"card": "rtx4070-12", "count": 3}]}))
        for l in lay.values():
            self.assertEqual(l["status"], "blocked")
            self.assertIn("sm89", " ".join(l["why"]))

    def test_every_impossible_layout_has_a_calculation_with_numbers(self):
        lay = by_id(zp()[0].layouts({"model": "qwen38-27b-int8", "cards": X3090}))
        d = lay["dual"]
        self.assertFalse(d["possible"])
        self.assertTrue(any("INT8" in w for w in d["why"]), d["why"])                  # dual has no INT8 reference profile
        self.assertTrue(any("short by" in w for w in d["why"]), d["why"])              # AND the arithmetic
        self.assertTrue(any(l.startswith("needed ") and "available" in l for l in d["calc"]), d["calc"])
        self.assertLess(d["margin_mib"], 0)
        self.assertGreater(lay["flip"]["margin_mib"], 0)

    def test_dual_holds_both_layouts_resident(self):
        zpl = zp()[0]
        s = zpl.steck["qwen38-27b-nvfp4"]
        flip = zpl._resident_weights_mib("qwen38-27b-nvfp4", "flip", 3)["mib"]
        dual = zpl._resident_weights_mib("qwen38-27b-nvfp4", "dual", 3)["mib"]
        self.assertEqual(flip, s["weights_total_mib"] + s["draft_mib"])
        self.assertAlmostEqual(dual, flip * 5 / 3)

    def test_moe_keeps_its_experts_in_the_host(self):
        zpl = zp()[0]
        s = zpl.steck["qwen38-nf-int4"]
        w = zpl._resident_weights_mib("qwen38-nf-int4", "flip", 3)["mib"]
        self.assertEqual(w, s["weights_dense_mib"] + s["lm_head_mib"] + s["draft_mib"])
        self.assertLess(w, 10000)

    def test_the_ceiling_is_never_multiplied_by_the_seats(self):
        """User 21.10.: 'nicht jeder request braucht den vollen kv'. Seats move the margin only by their recurrent state; the tokens only by ONE request's KV."""
        base = zp()[0].layouts({"model": "qwen38-27b-int8", "cards": X3090, "seats": 6, "tokens": 262144})["layouts"][1]["margin_mib"]
        more_seats = zp()[0].layouts({"model": "qwen38-27b-int8", "cards": X3090, "seats": 60, "tokens": 262144})["layouts"][1]["margin_mib"]
        s = Z.load_steckbrief()["qwen38-27b-int8"]
        self.assertAlmostEqual(base - more_seats, 54 * s["state_mib_per_seat_bf16"], delta=1.5)      # 54 more seats: state only, NOT 54 x 8.5 GiB of KV
        half = zp()[0].layouts({"model": "qwen38-27b-int8", "cards": X3090, "seats": 6, "tokens": 131072})["layouts"][1]["margin_mib"]
        self.assertAlmostEqual(half - base, 131072 * s["kv_bytes_per_token_fp8"] / (1 << 20), delta=1.5)

    def test_hardware_measured_only_for_the_cards_of_the_machine(self):
        zr = zp(hw_rig())[0]
        self.assertEqual(zr.layouts({"model": "qwen38-27b-int8", "cards": RIG})["hw"]["quelle"], "gemessen")
        self.assertEqual(zr.layouts({"model": "qwen38-27b-int8", "cards": X3090})["hw"]["quelle"], "geschaetzt")
        self.assertEqual(zp()[0].layouts({"model": "qwen38-27b-int8", "cards": RIG})["hw"]["quelle"], "geschaetzt")        # no hardware profile at all

    def test_captured_but_never_measured_rates_are_not_measured_and_offer_the_measurement(self):
        """Review A5: a hardware profile that only has the NVML names/sizes says "rates estimated", never "measured", and names the call that measures."""
        hw = zp(hw_rig(measured=False))[0].layouts({"model": "qwen38-27b-int8", "cards": RIG})["hw"]
        self.assertEqual((hw["quelle"], hw["label"]), ("erfasst", "rates estimated"))
        self.assertEqual(hw["measure"], {"api": "api/hwprofil/measure", "cards": [0, 1, 2]})
        self.assertNotIn("measure", zp(hw_rig())[0].layouts({"model": "qwen38-27b-int8", "cards": RIG})["hw"])

    def test_bad_requests_are_refused_with_a_reason(self):
        z = zp()[0]
        for body, frag in (({"model": "nope", "cards": X3090}, "unknown model"), ({"model": MODELS[0], "cards": [{"card": "gtx1", "count": 1}]}, "card catalog"),
                           ({"model": MODELS[0], "cards": [{"card": "rtx3090-24", "count": 0}]}, "between 1 and 8"), ({"model": MODELS[0], "cards": []}, "at least one card"),
                           ({"model": MODELS[0], "cards": X3090, "seats": 0}, "seats must be between"), ({"model": MODELS[0], "cards": X3090, "tokens": "x"}, "whole number"),
                           ({"model": MODELS[0], "cards": [{"card": "rtx2080ti-11", "count": 3}]}, "card catalog")):
            with self.assertRaises(Z.ZielError) as cm:
                z.layouts(body)
            self.assertIn(frag, str(cm.exception))


def canned(name):
    with open(os.path.join(FIX, name), encoding="utf-8") as fh:
        return json.load(fh)


class Plan(unittest.TestCase):
    def test_golden_rig_the_oracle_request_is_what_the_old_page_sent(self):
        res = canned("res_27b_int8_flip_3090.json")
        z, ed = zp(hw_rig(), res)
        out = z.plan({"model": "qwen38-27b-int8", "layout": "flip", "cards": RIG})
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["hw"]["quelle"], "gemessen")
        self.assertEqual(len(ed.calls), 1)
        req = ed.calls[0]
        self.assertEqual((req["basis"], req["form"], req["goals"]), ({"kind": "release", "name": "27b"}, "flip", {}))
        # the cards ARE the NVML cards of the machine: ProfilEditor._inventar_for turns the list into {hardware: profile} (real UUIDs, measured rates)
        want = sorted((CAT.card(c["card"])["nvml_name"], CAT.card(c["card"])["usable_mib"]) for c in req["inventory"])
        have = sorted((c["name"], c["vram_total_mib"]["v"]) for c in hw_rig()["cards"])
        self.assertEqual(want, have)
        self.assertTrue(P.ProfilEditor._hw_cards_match(hw_rig(), [{"entry": CAT.card(c["card"])} for c in req["inventory"]]))

    def test_seats_and_tokens_become_goals_only_when_given(self):
        res = canned("res_27b_int8_flip_3090.json")
        z, ed = zp(None, res)
        z.plan({"model": "qwen38-27b-int8", "layout": "flip", "cards": X3090})
        z.plan({"model": "qwen38-27b-int8", "layout": "flip", "cards": X3090, "seats": 4})
        z.plan({"model": "qwen38-27b-int8", "layout": "flip", "cards": X3090, "seats": 4, "tokens": 65536})
        self.assertEqual([c["goals"] for c in ed.calls], [{}, {"seats": 4}, {"seats": 4, "kv_tokens": 65536}])

    def test_the_layout_picks_the_form_and_the_internal_profile(self):
        res = canned("res_27b_int8_flip_3090.json")
        z, ed = zp(None, res)
        for model, layout, form, prof in (("qwen38-27b-int8", "solo", "tp", "27b"), ("qwen38-27b-nvfp4", "dual", "dual", "27b-nvfp4-dual"),
                                          ("qwen38-nf-int4", "flip", "flip", "nf-int4"), ("qwen38-27b-nvfp4", "solo", "tp", "27b-nvfp4")):
            z.plan({"model": model, "layout": layout, "cards": X3090})
            self.assertEqual((ed.calls[-1]["form"], ed.calls[-1]["basis"]["name"]), (form, prof))

    def test_a_layout_that_is_not_possible_never_asks_the_oracle(self):
        z, ed = zp(None, canned("res_27b_int8_flip_3090.json"))
        for body in ({"model": "qwen38-27b-int8", "layout": "dual", "cards": X3090}, {"model": "qwen38-27b-nvfp4", "layout": "flip", "cards": [{"card": "rtx3090-24", "count": 8}]},
                     {"model": "qwen38-nf-int4", "layout": "solo", "cards": [{"card": "rtx4070-12", "count": 3}]}):
            out = z.plan(body)
            self.assertIn(out["status"], ("blocked", "n_unsupported"))
            self.assertEqual(out["felder"], [])
            self.assertTrue(out["hinweise"])
        self.assertEqual(ed.calls, [])

    def test_a_planner_error_is_a_status_not_a_crash(self):
        z, _ed = zp(None, {"ok": False, "error": "the oracle is down", "notes": []})
        out = z.plan({"model": "qwen38-27b-int8", "layout": "flip", "cards": X3090})
        self.assertEqual(out["status"], "error")
        self.assertIn("the oracle is down", out["hinweise"][0])

    def test_no_internal_profile_name_in_the_visible_answer(self):
        z, _ed = zp(None, canned("res_27b_int8_flip_3090.json"))
        out = z.plan({"model": "qwen38-27b-nvfp4", "layout": "dual", "cards": X3090})
        vis = dict(out)
        intern = vis.pop("intern")
        self.assertEqual(intern["basis"], "27b-nvfp4-dual")
        text = json.dumps(vis)
        for name in ('"27b-nvfp4-dual"', '"27b-nvfp4"', '"nf-int4"', '"27b"', "PROFILE_"):
            self.assertNotIn(name, text)


class Summary(unittest.TestCase):
    """The 'rest computed' fields read out of canned oracle answers (captured from the real oracle on 3 x RTX 3090, synthetic catalog cards)."""

    def fields(self, fixture, model, layout, **kw):
        z, _ed = zp(None, canned(fixture))
        out = z.plan(dict({"model": model, "layout": layout, "cards": X3090}, **kw))
        self.assertEqual(out["status"], "ok", out.get("hinweise"))
        return out, {r["key"]: r for g in out["felder"] for r in g["zeilen"]}

    def test_golden_rig_the_planner_changes_nothing_and_the_dry_run_passes_without_force(self):
        """Our rig (measured hardware): the proposal leaves every value of the release profile as it is (no argv value 'changed'), the launcher passes without
        force; the figures shown are the dry run's own (captured from the real oracle on the replayed NVML cards of the rig)."""
        res = canned("res_27b_int8_flip_rig.json")
        self.assertFalse([w["label"] for w in res["values"] if w["changed"] and w.get("in_argv") and not w.get("seed")])
        z, _ed = zp(hw_rig(), res)
        out = z.plan({"model": "qwen38-27b-int8", "layout": "flip", "cards": RIG})
        f = {r["key"]: r for g in out["felder"] for r in g["zeilen"]}
        self.assertEqual((out["hw"]["quelle"], out["verdikt"]["outcome"]), ("gemessen", "geht"))
        self.assertEqual(f["p_layers"]["je_karte"], ["40", "14", "10"])
        self.assertEqual(f["d_split"]["je_karte"], ["58", "25", "25"])          # the reference D split (d_reshard.RC9_BASE)
        self.assertEqual((f["p_pool"]["wert"], f["x"]["wert"]), ("289250", "12288"))

    def test_27b_int8_flip_on_3x3090(self):
        out, f = self.fields("res_27b_int8_flip_3090.json", "qwen38-27b-int8", "flip")
        self.assertEqual(f["p_layers"]["je_karte"], ["24", "24", "16"])
        self.assertEqual(f["p_pool"]["wert"], "307660")
        self.assertEqual(f["d_mem"]["je_karte"], ["19792"] * 3)
        self.assertEqual(f["chunk_p"]["wert"], "2048")
        self.assertEqual(f["x"]["wert"], "12288")
        self.assertEqual(f["mamba_host"]["wert"], "100")
        self.assertTrue(f["d_pool"]["schaetzung"])
        self.assertNotIn("experts", [g["id"] for g in out["felder"]])
        self.assertEqual(out["defaults"], {"seats": 6, "tokens": 262144})
        self.assertEqual(out["verdikt"]["outcome"], "ok_with_force")

    def test_nf_flip_has_the_expert_rows(self):
        out, f = self.fields("res_nf_flip_3090.json", "qwen38-nf-int4", "flip")
        self.assertIn("experts", [g["id"] for g in out["felder"]])
        self.assertEqual(f["--pp-cut-expert-lru-rows"]["je_karte"], ["32", "32", "32"])
        self.assertTrue(f["--rank-moe-ratio"]["je_karte"])
        self.assertEqual(f["mamba_host"]["wert"], "900")
        self.assertIn("host", f["expert_total"]["warum"])

    def test_dual_has_no_x_and_solo_has_no_prefill_group(self):
        _out, f = self.fields("res_nvfp4_dual_3090.json", "qwen38-27b-nvfp4", "dual")
        self.assertNotIn("x", f)
        _out, f = self.fields("res_27b_int8_solo_3090.json", "qwen38-27b-int8", "solo")
        for k in ("x", "p_layers", "p_mem", "p_pool", "chunk_p"):
            self.assertNotIn(k, f)
        self.assertIn("d_mem", f)

    def test_the_ceiling_row_says_it_is_not_multiplied(self):
        _out, f = self.fields("res_27b_int8_flip_3090.json", "qwen38-27b-int8", "flip")
        self.assertIn("not multiplied by the seats", f["tokens"]["warum"])
        self.assertIn("ceiling", f["tokens"]["label"])


if __name__ == "__main__":
    unittest.main()
