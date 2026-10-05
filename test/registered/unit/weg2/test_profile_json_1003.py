"""PROFIL-EDITOR S1 (Auftrag 930): ``flliper.server/1`` -- .env <-> JSON round trip (golden), rows, origins, edits.

Gepinnt:
  * GOLDEN: fuer jedes Release-Profil gilt ``effective(f) == effective(render(import(f)))`` unter HTSGLANG_INSTRUMENTS 0 und 1
    (PROFILE_*-Variablen und -Arrays, PROFILE_ARGS, Exporte, ``_form``-Zeilen beider Funktionen). Bis das fuer alle gruen ist,
    bleibt die .env die Quelle (Nutzer-Setzung E3). Die Release-Profile liegen ausserhalb des Repos
    (/spinning/gpu-arb/docker/profiles_release); fehlen sie, laeuft nur der synthetische Teil.
  * Synthetische Profile decken, was die echten tragen: source, Funktionsneudefinition, Exporte, Instrumenten-Variante.
  * Ein verfaelschtes Rendering faellt auf (das Golden ist kein Tautologie-Test).
  * Bearbeiten schreibt in das besitzende Token zurueck (--env-p, --extra-p), Herkunft ``nutzer``, Zuruecksetzen auf Profil/Planer.
"""

import importlib.util
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
WEG2 = os.path.join(HERE, "..", "..", "..", "..", "python", "sglang", "srt", "weg2")
RELEASE_DIR = os.environ.get("PROFILES_RELEASE_DIR", "/spinning/gpu-arb/docker/profiles_release")


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(WEG2, fn))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PJ = _load("t_profile_json", "profile_json.py")

BASE_ENV = """\
# shellcheck shell=bash
PROFILE_NAME=base
PROFILE_LINE=nf
PROFILE_STATUS=abgenommen
PROFILE_CARD_COUNT=3
PROFILE_REQUIRED_PATHS=("/a b" /c)
PROFILE_DRAFT_X=${HTSGLANG_DRAFT:-dflt}
_x=1
PROFILE_ARGS=(--model /m --p-bs 2 --pp-stage-ratio 29,11,8 --p-hostgap
              "--extra-p=--rank-moe-ratio 183,137,168 --max-running-requests=2"
              --env-p "SGLANG_A=1;SGLANG_B=2,3" --env-d "SGLANG_A=4")
if [ "${HTSGLANG_INSTRUMENTS:-0}" = "1" ]; then PROFILE_ARGS+=(--p-chunk-policy dynamic); export SGLANG_INSTR_ONLY=yes; fi
export SGLANG_TOP=top
profile_form_env() {
  _form SGLANG_F1 1
  _form SGLANG_F2 "two words"
  _form SGLANG_TAGGED "pre-${HTSGLANG_TAG}-post"
}
profile_instr_env() {
  _form SGLANG_I1 1
}
"""

ALIAS_ENV = """\
source "$(dirname "${BASH_SOURCE[0]}")/base.env"
PROFILE_NAME=alias
PROFILE_STATUS=experimentell
profile_form_env() {
  _form SGLANG_F1 9
}
"""


def tmp_dir_with(**files):
    d = tempfile.mkdtemp(prefix="pj1003_")
    for k, v in files.items():
        with open(os.path.join(d, k.replace("__", "-") + ".env"), "w") as fh:
            fh.write(v)
    return d


class SyntheticRoundTrip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tmp_dir_with(base=BASE_ENV, alias=ALIAS_ENV)
        cls.addClassCleanup(shutil.rmtree, cls.dir, True)

    def path(self, n):
        return os.path.join(self.dir, n + ".env")

    def test_golden_synthetic(self):
        for n in ("base", "alias"):
            self.assertEqual(PJ.roundtrip_check(self.path(n)), [], n)

    def test_import_facts(self):
        d = PJ.import_env(self.path("base"))
        self.assertEqual(d["schema"], PJ.SCHEMA)
        self.assertEqual(d["name"], "base")
        self.assertEqual(d["line"], "nf")
        var = {v["name"]: v for v in d["vars"]}
        self.assertEqual(var["PROFILE_REQUIRED_PATHS"]["values"], ["/a b", "/c"])
        self.assertNotIn("_x", var)                                   # only PROFILE_* are facts
        self.assertEqual([e["name"] for e in d["exports"]], ["SGLANG_TOP"])
        self.assertEqual([(r["name"], r["value"]) for r in d["form"]],
                         [("SGLANG_F1", "1"), ("SGLANG_F2", "two words"), ("SGLANG_TAGGED", "pre-${HTSGLANG_TAG}-post")])
        self.assertEqual([r["name"] for r in d["instr"]], ["SGLANG_I1"])
        # the INSTRUMENTS=1 variant differs and is carried, not lost
        self.assertIn("instruments1", d)
        flags1 = [e.get("flag") for e in d["instruments1"]["args"]]
        self.assertIn("--p-chunk-policy", flags1)
        self.assertIn({"name": "SGLANG_INSTR_ONLY", "value": "yes"}, d["instruments1"]["exports"])

    def test_runtime_placeholders_survive_the_round_trip(self):
        """The boot tag is set by the entrypoint BEFORE it sources a profile: frozen at import it would be empty (found by the
        entrypoint equality test on nf-int4's SGLANG_MOE_COLD_TIER_INSTANCE). It stays an expansion, resolved at run time."""
        d = PJ.import_env(self.path("base"))
        text = PJ.render_env(d)
        self.assertIn('_form SGLANG_TAGGED pre-"${HTSGLANG_TAG}"-post', text)       # an expansion the entrypoint resolves, not a frozen value
        out = os.path.join(self.dir, "ph.env")
        with open(out, "w") as fh:
            fh.write(text)
        raw = PJ.dump_env(out, "0")
        self.assertEqual(dict(raw["form"])["SGLANG_TAGGED"], "pre-@@HTSGLANG_TAG@@-post")        # the dump runs with a sentinel
        env = dict(os.environ, HTSGLANG_TAG="dkr42")
        got = PJ.subprocess.run(["bash", "-c", 'source "$1"; _form() { echo "$1=$2"; }; profile_form_env', "x", out], capture_output=True, text=True, env=env).stdout
        self.assertIn("SGLANG_TAGGED=pre-dkr42-post", got)
        os.unlink(out)

    def test_caller_switches_are_named_not_hidden(self):
        d = PJ.import_env(self.path("base"))
        self.assertEqual(d["meta"]["caller_switches"], ["HTSGLANG_DRAFT"])
        self.assertEqual({v["name"]: v.get("value") for v in d["vars"]}["PROFILE_DRAFT_X"], "dflt")        # baked with its default
        self.assertEqual(PJ.import_env(self.path("alias"))["meta"]["caller_switches"], ["HTSGLANG_DRAFT"])   # found through source

    def test_alias_resolves_its_source_and_redefinition(self):
        d = PJ.import_env(self.path("alias"))
        self.assertEqual(d["name"], "alias")
        self.assertEqual({v["name"]: v.get("value") for v in d["vars"]}["PROFILE_STATUS"], "experimentell")
        self.assertEqual([(r["name"], r["value"]) for r in d["form"]], [("SGLANG_F1", "9")])
        self.assertTrue(any(e.get("flag") == "--pp-stage-ratio" for e in d["args"]))   # from the sourced base

    def test_args_tokens_are_lossless(self):
        toks = ["--model", "/m", "--p-bs", "2", "--p-hostgap", "--extra-p=--a b --c=1", "--env-p", "K=V;L=1,2", "stray"]
        self.assertEqual(PJ.render_args(PJ.parse_args(toks)), toks)
        specs = {"--extra-p": {"bare": False, "nargs": None}, "--p-hostgap": {"bare": True, "nargs": None}}
        toks2 = ["--extra-p", "--max-running-requests=2", "--p-hostgap"]
        ent = PJ.parse_args(toks2, specs)
        self.assertEqual(ent[0], {"flag": "--extra-p", "values": ["--max-running-requests=2"]})   # value starting with -- is the value
        self.assertEqual(PJ.render_args(ent), toks2)

    def test_a_falsified_rendering_is_caught(self):
        d = PJ.import_env(self.path("base"))
        text = PJ.render_env(d).replace("two words", "other words")
        p = os.path.join(self.dir, "falsified.env")
        with open(p, "w") as fh:
            fh.write(text)
        diffs = PJ.diff_effective(PJ.effective(self.path("base")), PJ.effective(p))
        self.assertTrue(diffs and any("form" in x for x in diffs))
        os.unlink(p)

    def test_expected_effective_matches_render_for_an_edited_doc(self):
        d = PJ.import_env(self.path("base"))
        PJ.freeze_profile_values(d)
        e = PJ.apply_edits(d, [{"key": "flag:--p-bs", "op": "set", "value": "5"},
                               {"key": "env:D:SGLANG_NEW", "op": "set", "value": "7"},
                               {"key": "form:SGLANG_F2", "op": "delete"}])
        p = os.path.join(self.dir, "edited.env")
        with open(p, "w") as fh:
            fh.write(PJ.render_env(e))
        self.assertEqual(PJ.diff_effective(PJ.expected_effective(e), PJ.effective(p)), [])
        os.unlink(p)


class RowsAndEdits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tmp_dir_with(base=BASE_ENV)
        cls.addClassCleanup(shutil.rmtree, cls.dir, True)
        cls.doc = PJ.import_env(os.path.join(cls.dir, "base.env"))
        PJ.freeze_profile_values(cls.doc)

    def rows(self, doc):
        return {r["key"]: r for r in PJ.rows(doc)}

    def test_keys_and_scopes(self):
        r = self.rows(self.doc)
        for k, scope, val in (("flag:--p-bs", "launcher", "2"), ("env:P:SGLANG_A", "P", "1"), ("env:P:SGLANG_B", "P", "2,3"),
                              ("env:D:SGLANG_A", "D", "4"), ("extra:P:--rank-moe-ratio", "P", "183,137,168"),
                              ("extra:P:--max-running-requests", "P", "2"), ("var:PROFILE_CARD_COUNT", "profile", "3"),
                              ("form:SGLANG_F2", "form", "two words"), ("export:SGLANG_TOP", "all", "top")):
            self.assertEqual((r[k]["scope"], r[k]["value"]), (scope, val), k)
        self.assertTrue(r["flag:--p-hostgap"]["bare"])

    def test_edit_writes_back_into_the_owning_token_and_marks_nutzer(self):
        e = PJ.apply_edits(self.doc, [{"key": "env:P:SGLANG_B", "op": "set", "value": "9,9"},
                                      {"key": "extra:P:--rank-moe-ratio", "op": "set", "value": "1,1,1"}])
        r = self.rows(e)
        self.assertEqual(r["env:P:SGLANG_B"]["value"], "9,9")
        self.assertEqual(r["env:P:SGLANG_A"]["value"], "1")                     # neighbours untouched
        self.assertEqual(r["env:D:SGLANG_A"]["value"], "4")
        self.assertEqual(r["extra:P:--max-running-requests"]["value"], "2")
        self.assertEqual(e["meta"]["origins"]["env:P:SGLANG_B"], "nutzer")
        v = PJ.view(e)
        self.assertEqual({x["key"] for x in v["rows"] if x["changed"]}, {"env:P:SGLANG_B", "extra:P:--rank-moe-ratio"})
        # the original document is not mutated
        self.assertEqual(self.rows(self.doc)["env:P:SGLANG_B"]["value"], "2,3")

    def test_reset_to_profile_and_planer_and_delete_and_add(self):
        e = PJ.apply_edits(self.doc, [{"key": "flag:--p-bs", "op": "set", "value": "7"}])
        back = PJ.apply_edits(e, [{"key": "flag:--p-bs", "op": "reset", "to": "profil"}])
        self.assertEqual(self.rows(back)["flag:--p-bs"]["value"], "2")
        self.assertEqual(back["meta"]["origins"]["flag:--p-bs"], "profil")
        e["meta"]["planner"] = {"flag:--p-bs": "3", "extra:D:--rank-gpu-memory-mib": "26000,17000,17000"}
        taken = PJ.apply_edits(e, [{"key": "flag:--p-bs", "op": "reset", "to": "planer"},
                                   {"key": "extra:D:--rank-gpu-memory-mib", "op": "reset", "to": "planer"}])
        r = self.rows(taken)
        self.assertEqual((r["flag:--p-bs"]["value"], r["extra:D:--rank-gpu-memory-mib"]["value"]), ("3", "26000,17000,17000"))
        self.assertEqual(taken["meta"]["origins"]["extra:D:--rank-gpu-memory-mib"], "planer")
        gone = PJ.apply_edits(self.doc, [{"key": "flag:--p-hostgap", "op": "delete"}])
        self.assertNotIn("flag:--p-hostgap", self.rows(gone))
        self.assertEqual([x["key"] for x in PJ.view(gone)["removed"]], ["flag:--p-hostgap"])
        again = PJ.apply_edits(gone, [{"key": "flag:--p-hostgap", "op": "reset", "to": "profil"}])
        self.assertIn("flag:--p-hostgap", self.rows(again))

    def test_a_bare_flag_with_specs_toggles(self):
        specs = {"--p-hostgap": {"bare": True, "nargs": None}}
        e = PJ.apply_edits(self.doc, [{"key": "flag:--p-hostgap", "op": "set", "value": "0"}], specs)
        self.assertNotIn("flag:--p-hostgap", self.rows(e))

    def test_explanation_sources_and_dependencies(self):
        cat = {"--p-bs": {"text": "Gleichzeitige Anfragen", "help": "K1 ...", "source": {"file": "launcher.py", "line": 1}, "status": "kuratiert",
                          "depends": [{"to": "--d-bs", "rel": "tauscht", "effect": "x", "calc": "text"}]},
               "--p-hostgap": {"help": "instrument", "source": {"file": "launcher.py", "line": 2}, "status": "geerntet", "depends": []}}
        v = PJ.view(self.doc, cat, {"--model": {"text": "weil", "source": "base.env:3"}})
        by = {r["key"]: r for r in v["rows"]}
        self.assertEqual(by["flag:--p-bs"]["explain"]["status"], "kuratiert")
        self.assertEqual(by["flag:--p-hostgap"]["explain"]["status"], "geerntet")
        self.assertEqual(by["flag:--model"]["explain"]["status"], "profil-kommentar")
        self.assertEqual(by["flag:--pp-stage-ratio"]["explain"]["status"], "unerklaert")
        dep = by["flag:--p-bs"]["explain"]["depends"][0]
        self.assertFalse(dep["present"])                                       # --d-bs is not set in this profile
        self.assertEqual(v["coverage"]["unerklaert"] + v["coverage"]["erklaert"], v["coverage"]["rows"])


@unittest.skipUnless(os.path.isdir(RELEASE_DIR), "release profiles are outside the repo (%s)" % RELEASE_DIR)
class ReleaseProfilesGolden(unittest.TestCase):
    def test_every_release_profile_round_trips(self):
        files = sorted(f for f in os.listdir(RELEASE_DIR) if f.endswith(".env"))
        self.assertGreaterEqual(len(files), 5)
        bad = {}
        for f in files:
            diffs = PJ.roundtrip_check(os.path.join(RELEASE_DIR, f))
            if diffs:
                bad[f] = diffs[:3]
        self.assertEqual(bad, {})


if __name__ == "__main__":
    unittest.main()
