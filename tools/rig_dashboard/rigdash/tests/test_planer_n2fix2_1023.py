"""planer-n2fix2 1023: the dashboard half of the follow-up to the n2fix review (deskq/done/planer-n2fix-1023.md, "Review 23.10." + Nachtrag).

* B11 the start profile kept the base profile's model and draft although the request named ``model_path`` / ``draft_path`` (the int4ar seat set
      ``--model`` / ``--dflash-draft-path`` / ``PROFILE_*`` by hand, deskq/bl/27b-int4ar-1023/mkprofile.py).  ``ProfilEditor._path_edits`` puts the chosen
      directories into every row that names the base profile's ones (origin ``nutzer``: the user's choice, not a planner value); the oracle retargets the
      dry run the same way (``propose_verdict.retarget_paths``, pinned in test/registered/unit/pdflip/test_planer_n2fix2_1023.py).
* A4  "Huihui-Qwen3.8-27B-Abliterated-INT4-W4A16-AutoRound" is a planner model: model list, fact sheet row (from the planner's model profile + config.json),
      release profile ``27b-int4ar``.  The live request ``{"model": "qwen38-27b-int4ar", "layout": "flip", "cards": [{"card": "rtx3080-20", "count": 2}]}``
      answers with a verdict instead of "unknown model" (``RealInt4ar``: the real oracle child, two launcher dry runs).
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from rigdash import kartenplan as K  # noqa: E402
from rigdash import planer_ziel as Z  # noqa: E402
from rigdash import profil as P  # noqa: E402
from rigdash import profile_oracle as ORA  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
REPO_PY = os.path.join(REPO_ROOT, "python")
RELEASE_DIR = os.path.join(REPO_ROOT, "docker", "flliper", "profiles_release")
CATALOG = os.path.join(os.path.dirname(HERE), "profil_data", "catalog.json")
FIX_N2 = os.path.join(HERE, "fixtures", "planer_n2fix_1023")
CARD_LIBRARY = os.path.join(FIX_N2, "card_library_rig_0814.json")
RIG_HW = os.path.join(FIX_N2, "hwprofil_rig_1023.json")
MC = "/spinning/llm_stuff/club-3090/models-cache/"
INT4AR = MC + "Huihui-Qwen3.8-27B-Abliterated-INT4-W4A16-AutoRound-ct"
W8 = MC + "Qwen3.8-27B-DFlash2-W8-lued"
NVFP4 = MC + "Qwen3.8-27B-NVFP4-RadixArk"
NVFP4_DRAFT = MC + "Qwen3.8-27B-DFlash2-NVFP4-RTNcal"
HAS_LAUNCHER = os.path.isfile(os.path.join(REPO_PY, "flliper", "srt", "pdflip", "launcher.py"))
HAS_INT4AR = os.path.isfile(os.path.join(INT4AR, "config.json")) and os.path.isfile(os.path.join(W8, "config.json"))
MID = "qwen38-27b-int4ar"
TWO_3080 = [{"card": "rtx3080-20", "count": 2}]


def _vars(doc):
    return {v["name"]: (v.get("value") if "value" in v else " ".join(v.get("values") or [])) for v in doc.get("vars") or []}


def _flag(doc, name):
    for a in doc.get("args") or []:
        if a.get("flag") == name:
            return " ".join(a.get("values") or [])
    return None


def _editor(tmp, **kw):
    ed = P.ProfilEditor(kartenplaner=K.Kartenplaner(tree=REPO_PY), release_dir=RELEASE_DIR, user_dir=os.path.join(tmp, "usr"), tree=REPO_PY,
                        catalog_file=CATALOG, **kw)
    ed.check_path = lambda p, fld: p
    return ed


class PathEdits(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="n2fix2_")
        self.ed = _editor(self.tmp)
        self.base = self.ed._import_release("27b-nvfp4")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_every_row_naming_the_base_directories_names_the_chosen_ones(self):
        vals, notes = self.ed._path_edits(self.base, {"model_path": INT4AR, "draft_path": W8})
        self.assertEqual(vals["var:PROFILE_MODEL"], INT4AR)
        self.assertEqual(vals["flag:--model"], INT4AR)
        self.assertEqual(vals["var:PROFILE_DRAFT"], W8)
        self.assertEqual(vals["flag:--dflash-draft-path"], W8)
        self.assertIn(W8 + "/config.json", vals["var:PROFILE_REQUIRED_PATHS"])
        self.assertNotIn(NVFP4_DRAFT, json.dumps(vals))
        self.assertEqual(len(notes), 2)
        self.assertTrue(notes[0].startswith("Model %s chosen for this proposal" % INT4AR), notes)

    def test_the_base_paths_or_nothing_change_nothing(self):
        self.assertEqual(self.ed._path_edits(self.base, {"model_path": NVFP4 + "/", "draft_path": NVFP4_DRAFT}), ({}, []))
        self.assertEqual(self.ed._path_edits(self.base, {}), ({}, []))

    def test_a_longer_name_is_another_directory(self):
        pat = P.ProfilEditor._path_re(NVFP4)
        self.assertIsNone(pat.search(NVFP4 + "-ct/config.json"))
        self.assertIsNotNone(pat.search("--model=" + NVFP4))
        self.assertIsNotNone(pat.search(NVFP4 + "/config.json other"))

    def test_the_start_profile_carries_them_as_the_users_choice(self):
        chosen = self.ed._path_edits(self.base, {"model_path": INT4AR, "draft_path": W8})
        prop = {"values": [], "cards": [], "inventory": {"same_as_profile": True}}
        new, _keys, _skipped = self.ed._startprofil(self.base, prop, "27b-nvfp4", ({}, {}, []), chosen)
        pj, _ = self.ed.mods()
        self.assertEqual(_flag(new, "--model"), INT4AR)
        self.assertEqual(_flag(new, "--dflash-draft-path"), W8)
        v = _vars(new)
        self.assertEqual((v["PROFILE_MODEL"], v["PROFILE_DRAFT"]), (INT4AR, W8))
        for k in chosen[0]:
            self.assertEqual(new["meta"]["origins"][k], pj.ORIGIN_NUTZER, k)
            self.assertNotIn(k, new["meta"].get("planner") or {})
        text = pj.render_env(new)
        self.assertEqual(self.ed.verify_render(new, text), [])
        self.assertIn(INT4AR, text)
        self.assertNotIn(NVFP4_DRAFT, text)


class ModelInt4ar(unittest.TestCase):
    def test_model_list_profile_and_fact_sheet(self):
        m = {x["id"]: x for x in Z.MODELS}[MID]
        self.assertEqual((m["line"], m["format"], m["profiles"]["flip"], m["profiles"]["solo"], m["profiles"]["dual"]), ("27b", "int4", "27b-int4ar", "27b-int4ar", None))
        self.assertTrue(os.path.isfile(os.path.join(RELEASE_DIR, "27b-int4ar.env")))
        s = Z.load_steckbrief()[MID]
        self.assertEqual(s["checkpoint"], "Huihui-Qwen3.8-27B-Abliterated-INT4-W4A16-AutoRound-ct")
        self.assertEqual(s["draft_checkpoint"], "Qwen3.8-27B-DFlash2-W8-lued")
        q = s["quant"]
        self.assertEqual(q["method"], "compressed-tensors")
        self.assertEqual([(g["type"], g["num_bits"], g["group_size"], g["symmetric"], g["strategy"]) for g in q["weight_groups"]],
                         [("int", 4, 128, True, "group")])
        for pat in ("lm_head", "re:.*visual.*", "re:.*linear_attn[.]in_proj_a$", "re:.*linear_attn[.]in_proj_b$"):
            self.assertIn(pat, q["unquantized"])
        self.assertEqual(q["converted_from"]["quant_method"], "auto-round")
        self.assertAlmostEqual(s["vision_tower_mib"], 878.8, delta=0.5)       # 921460192 B (the seat's repack, model-visual.safetensors)
        self.assertGreater(s["mtp_mib"], 0)
        self.assertEqual((s["n_layers"], s["attn_layers"], s["linear_layers"]), (64, 16, 48))
        self.assertLess(s["weights_total_mib"], Z.load_steckbrief()["qwen38-27b-nvfp4"]["weights_total_mib"])

    def test_the_rough_check_offers_flip_and_solo_on_two_3080(self):
        tmp = tempfile.mkdtemp(prefix="n2fix2_")
        self.addCleanup(shutil.rmtree, tmp, True)
        zp = Z.ZielPlaner(_editor(tmp))                    # no hardware profile: catalog cards
        lay = {x["id"]: x for x in zp.layouts({"model": MID, "cards": TWO_3080, "tokens": 131072})["layouts"]}
        self.assertTrue(lay["flip"]["possible"], lay["flip"])
        self.assertEqual(lay["dual"]["status"], "blocked")
        k = zp.katalog()
        row = {x["id"]: x for x in k["models"]}[MID]
        self.assertEqual(row["layouts"], ["solo", "flip"])
        self.assertEqual(row["weights_mib"], Z.load_steckbrief()[MID]["weights_total_mib"])


@unittest.skipUnless(HAS_LAUNCHER and HAS_INT4AR, "needs this tree's launcher and the int4ar checkpoint + W8 draft headers on this box")
class RealInt4ar(unittest.TestCase):
    """The page's request on the rig's hardware profile (2 x RTX 3080 = 2 of its 3 cards, real UUIDs) with the REAL oracle child, and the propose request
    of the int4ar seat (basis 27b-nvfp4 + model_path/draft_path) whose dry run ran the NVFP4 weights before B11."""

    @classmethod
    def setUpClass(cls):
        cls._env = os.environ.get("FLLIPER_CARD_LIBRARY")
        os.environ["FLLIPER_CARD_LIBRARY"] = CARD_LIBRARY
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        cls.tmp = tempfile.mkdtemp(prefix="n2fix2_real_")
        cls.oracle = ORA.OracleService(REPO_PY, python=sys.executable)
        with open(RIG_HW, encoding="utf-8") as fh:
            hw = json.load(fh)
        cls.ed = _editor(cls.tmp, oracle=cls.oracle, hardware=lambda: {"ok": True, "profile": hw})
        cls.out = Z.ZielPlaner(cls.ed).plan({"model": MID, "layout": "flip", "cards": TWO_3080, "tokens": 131072})
        inv = [{"card": "rtx3080-20", "pcie": {"gen": 4, "lanes": 16}}] * 2
        cls.b11 = cls.ed.propose({"basis": {"kind": "release", "name": "27b-nvfp4"}, "form": "flip", "inventory": inv,
                                  "goals": {"kv_tokens": 131072}, "model_path": INT4AR, "draft_path": W8})

    @classmethod
    def tearDownClass(cls):
        cls.oracle.close()
        shutil.rmtree(cls.tmp, ignore_errors=True)
        if cls._env is None:
            os.environ.pop("FLLIPER_CARD_LIBRARY", None)
        else:
            os.environ["FLLIPER_CARD_LIBRARY"] = cls._env

    def test_a4_the_page_answers_with_a_verdict(self):
        out = self.out
        self.assertEqual(out.get("status"), "ok", out.get("hinweise"))
        self.assertEqual(out["model"]["id"], MID)
        vd = out["verdikt"]
        self.assertIn(vd["outcome"], ("geht", "ok_with_force"), vd)
        doc = out["intern"]["startprofil"]["doc"]
        self.assertEqual(_flag(doc, "--model"), INT4AR)
        self.assertEqual(_flag(doc, "--dflash-draft-path"), W8)
        v = _vars(doc)
        self.assertEqual((v["PROFILE_FORMAT"], v["PROFILE_CARD_COUNT"], v["PROFILE_INVENTORY"]), ("int4", "2", "RTX3080,RTX3080"))
        self.assertEqual(out["intern"]["basis"], "27b-int4ar")

    def test_b11_the_dry_run_and_the_start_profile_take_the_chosen_model(self):
        res = self.b11
        self.assertTrue(res.get("ok"), res.get("error"))
        self.assertEqual(res["verdict"]["profil"]["model"], INT4AR)           # the dry run was asked about the int4ar weights, not the NVFP4 ones
        self.assertEqual(res["verdict"]["profil"]["draft"], W8)
        self.assertIn("--model", res["launch"]["argv"])
        self.assertEqual(res["launch"]["argv"][res["launch"]["argv"].index("--model") + 1], INT4AR)
        doc = res["startprofil"]["doc"]
        self.assertEqual(_flag(doc, "--model"), INT4AR)
        self.assertEqual(_flag(doc, "--dflash-draft-path"), W8)
        self.assertEqual(_vars(doc)["PROFILE_MODEL"], INT4AR)
        self.assertTrue(res["startprofil"]["verifiziert"], res["startprofil"]["probleme"])
        self.assertTrue(any("replaces the base profile's model" in n for n in res["notes"]), res["notes"])


if __name__ == "__main__":
    unittest.main()
