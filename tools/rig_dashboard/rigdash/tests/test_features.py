"""Feature table (user order 29.09.): im Image from git, aktiv from state.json, gains per model.

stdlib unittest; run: python3 -m unittest discover -s tests (from tools/rig_dashboard/rigdash).
The git cases build a throw-away repo: a line with an ancestor commit, a feature branch
cherry-picked onto the line (new sha, same patch-id) and a branch that never landed.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rigdash import features, features_update  # noqa: E402


def _git(repo, *args):
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@t")
    return subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True, env=env).stdout.decode().strip()


def _commit(repo, name, text, msg):
    with open(os.path.join(repo, name), "w") as fh:
        fh.write(text)
    _git(repo, "add", name)
    _git(repo, "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")[:10]


def _boot(root, boot_id, lifecycle, groups=None, profile="nf-test", rev="", image="htsglang:cu130-weg2-rc12z30p-27b-nf"):
    d = os.path.join(root, boot_id)
    os.makedirs(d, exist_ok=True)
    st = {"schema": "weg2.state/1", "kind": "boot", "boot_id": boot_id, "rev": rev, "image": image,
          "profile": profile, "lifecycle": {"state": lifecycle}, "groups": groups or {}}
    with open(os.path.join(d, "state.json"), "w") as fh:
        json.dump(st, fh)
    return st


class GitCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        r = cls.repo = cls.tmp.name
        _git(r, "init", "-q", "-b", "line")
        cls.base = _commit(r, "a.txt", "a\n", "base")
        cls.anc = _commit(r, "b.txt", "b\n", "H1: ancestor feature")
        _git(r, "checkout", "-q", "-b", "feat")
        cls.picked_orig = _commit(r, "c.txt", "c\n", "H2: picked feature")
        _git(r, "checkout", "-q", "-b", "never", cls.anc)
        cls.never = _commit(r, "d.txt", "d\n", "H3: never landed")
        _git(r, "checkout", "-q", "line")
        _commit(r, "e.txt", "e\n", "line moves on")
        _git(r, "cherry-pick", cls.picked_orig)
        cls.rev = _git(r, "rev-parse", "HEAD")[:10]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def line(self):
        return features.LineIndex(self.repo, self.rev, "2000-01-01")

    def test_ancestor_is_in_image(self):
        im = features.in_image(self.repo, self.rev, [{"branch": "line", "sha": self.anc}], self.line())
        self.assertEqual((im["state"], im["how"]), ("ja", "vorfahr"))

    def test_cherry_pick_found_by_patch_id(self):
        # 27B picks NF branches under a new sha: the original sha is no ancestor, its patch is on the line
        im = features.in_image(self.repo, self.rev, [{"branch": "feat", "sha": self.picked_orig}], self.line())
        self.assertEqual((im["state"], im["how"]), ("ja", "patch-id"))
        self.assertNotEqual(im["line_sha"], self.picked_orig)

    def test_branch_that_never_landed_is_not_in_image(self):
        im = features.in_image(self.repo, self.rev, [{"branch": "never", "sha": self.never}], self.line())
        self.assertEqual(im["state"], "nein")

    def test_unknown_sha_and_missing_rev_are_unknown_not_no(self):
        self.assertEqual(features.in_image(self.repo, self.rev, [{"branch": "x", "sha": "0123456789"}],
                                           self.line())["state"], "unbekannt")
        self.assertEqual(features.in_image(self.repo, "", [{"branch": "x", "sha": self.anc}], None)["state"],
                         "unbekannt")
        self.assertEqual(features.in_image(self.repo, "fedcba9876", [{"branch": "x", "sha": self.anc}],
                                           None)["state"], "unbekannt")

    def test_view_end_to_end_cached_and_per_model(self):
        with tempfile.TemporaryDirectory() as t:
            nf, b27 = os.path.join(t, "nf"), os.path.join(t, "27b")
            _boot(nf, "nf-boot-1", "serving", rev=self.rev, groups={
                "D": {"state": "ready", "launch": {"argv": ["--x"], "env": {"SGLANG_A": "1"}}},
                "P": {"state": "ready", "launch": {"argv": [], "env": {}}}})
            _boot(b27, "27b-boot-1", "dead", rev=self.rev, image="htsglang:cu130-weg2-rc12z30j-27b-nf")
            os.symlink("27b-boot-1", os.path.join(b27, "current"))
            path = os.path.join(t, "features.json")
            with open(path, "w") as fh:
                json.dump({"features": [
                    {"id": "A", "modell": "NF", "titel": "a", "fertig": True,
                     "zweige": [{"branch": "line", "sha": self.anc}],
                     "schalter": [{"name": "SGLANG_A", "art": "env", "gruppe": "D", "an_wert": "1", "default": "aus"}]},
                    {"id": "B", "modell": "beide", "titel": "b", "fertig": True,
                     "zweige": [{"branch": "feat", "sha": self.picked_orig}],
                     "schalter": [{"name": "SGLANG_B", "art": "env", "gruppe": "P", "an_wert": "1", "default": "aus"}],
                     "aus_begruendung": "noch kein A/B",
                     "gewinn": [{"metrik": "Flipzeit", "nachher": "2.4", "einheit": "s", "art": "gemessen", "modell": "NF"},
                                {"metrik": "Decode", "nachher": "40", "einheit": "tok/s", "art": "gemessen"}]},
                ]}, fh)
            f = features.Features(path, repo=self.repo, state_roots={"NF": nf, "27B": b27}, background=False)
            v = f.view()
            by = {m["model"]: m for m in v["models"]}
            self.assertTrue(by["NF"]["boot"]["running"])
            self.assertEqual(by["NF"]["boot"]["rc"], "rc12z30p")
            # (D) the 27B table follows its LAST boot, it does not read "aus" because 27B is down
            self.assertFalse(by["27B"]["boot"]["running"])
            self.assertEqual(by["27B"]["boot"]["rc"], "rc12z30j")
            nf_rows = {r["id"]: r for r in by["NF"]["features"]}
            self.assertEqual(nf_rows["A"]["aktiv"]["state"], "an")
            self.assertEqual(nf_rows["B"]["im_image"]["how"], "patch-id")
            self.assertTrue(nf_rows["B"]["image_aber_aus"])
            # (C) gains strictly per model: the NF gain under NF only, the gain without modell never shown
            self.assertEqual([g["metrik"] for g in nf_rows["B"]["gewinn"]], ["Flipzeit"])
            b27 = {r["id"]: r for r in by["27B"]["features"]}
            self.assertEqual(list(b27), ["B"])
            self.assertEqual(b27["B"]["gewinn"], [])
            self.assertTrue(any("ohne Feld modell" in p for p in v["problems"]))
            t0 = time.time()
            f.view()
            self.assertLess(time.time() - t0, 0.5)   # git ran once per (model, rev, file)


class AktivCase(unittest.TestCase):
    ST = {"groups": {
        "P": {"launch": {"argv": ["--p-host-overlap", "--chunk", "16384"], "env": {"SGLANG_X": "1"}}},
        "D": {"launch": {"argv": ["--d-kv-token-cut=owned"], "env": {"SGLANG_X": "0", "SGLANG_Y": "ptx"}}}}}

    def sw(self, name, gruppe, an_wert="", default="aus", art=None):
        return features.switch_state({"name": name, "art": art or ("flag" if name.startswith("--") else "env"),
                                      "gruppe": gruppe, "an_wert": an_wert, "default": default}, self.ST, None)

    def test_env_per_group_and_both(self):
        self.assertEqual(self.sw("SGLANG_X", "P", "1")["state"], "an")
        self.assertEqual(self.sw("SGLANG_X", "D", "1")["state"], "aus")
        self.assertEqual(self.sw("SGLANG_X", "beide", "1")["state"], "teilweise")
        self.assertEqual(self.sw("SGLANG_Y", "D", "ptx")["state"], "an")

    def test_absent_switch_takes_its_default(self):
        self.assertEqual(self.sw("SGLANG_NONE", "D", "1", default="an")["state"], "an")
        self.assertEqual(self.sw("SGLANG_NONE", "D", "1", default="aus")["state"], "aus")

    def test_flags_presence_and_value(self):
        self.assertEqual(self.sw("--p-host-overlap", "P")["state"], "an")
        self.assertEqual(self.sw("--d-kv-token-cut", "D")["state"], "an")
        self.assertEqual(self.sw("--chunk", "P", "8192")["state"], "aus")

    def test_profile_fallback_ignores_comments(self):
        prof = ('NF_ENV_D="SGLANG_A=1;SGLANG_B=0"   # SGLANG_C=1 steht nur im Kommentar\n'
                '# SGLANG_D=1 ganz auskommentiert\n  --max-kv-per-request 262144\n')
        text = features.strip_comments(prof)
        st = lambda n, w="1", art="env": features.switch_state(
            {"name": n, "art": art, "gruppe": "D", "an_wert": w, "default": "aus"}, {"groups": {}}, text)
        self.assertEqual(st("SGLANG_A")["state"], "an")
        self.assertEqual(st("SGLANG_B")["state"], "aus")
        self.assertEqual(st("SGLANG_C")["state"], "aus")
        self.assertEqual(st("SGLANG_D")["state"], "aus")
        self.assertEqual(st("--max-kv-per-request", "524288", "flag")["state"], "aus")
        self.assertIn("Profil", st("SGLANG_A")["src"])

    def test_no_state_no_profile_is_unknown(self):
        r = features.switch_state({"name": "SGLANG_A", "art": "env", "gruppe": "D", "default": "aus"}, None, None)
        self.assertEqual(r["state"], "unbekannt")


class FileAndCliCase(unittest.TestCase):
    def test_reload_only_on_change_and_keep_last_good(self):
        with tempfile.TemporaryDirectory() as t:
            p = os.path.join(t, "f.json")
            with open(p, "w") as fh:
                json.dump({"features": [{"id": "A"}]}, fh)
            ff = features.FeatureFile(p)
            self.assertEqual([f["id"] for f in ff.load()[0]], ["A"])
            with open(p, "w") as fh:
                fh.write("{kaputt")
            os.utime(p, ns=(time.time_ns(), time.time_ns() + 10**9))
            feats, err, _ = ff.load()
            self.assertEqual([f["id"] for f in feats], ["A"])
            self.assertIsNotNone(err)

    def test_cli_upsert_gain_and_refusal(self):
        with tempfile.TemporaryDirectory() as t:
            p = os.path.join(t, "f.json")
            run = lambda *a: features_update.main(["--file", p, *a])
            run("set", "--id", "H1", "--modell", "beide", "--titel", "t", "--fertig", "ja",
                "--zweig", "desk/x=0123456789", "--schalter", "SGLANG_H1=env:D:1:aus")
            run("add-zweig", "--id", "H1", "--zweig", "desk/27b-unified-0926=abcdef0123")
            with self.assertRaises(SystemExit):   # modell=beide: a gain without modell is refused
                run("gewinn", "--id", "H1", "--metrik", "Flipzeit", "--nachher", "2.4", "--art", "gemessen")
            run("gewinn", "--id", "H1", "--modell", "NF", "--metrik", "Flipzeit", "--vorher", "3.1",
                "--nachher", "2.4", "--einheit", "s", "--art", "gemessen")
            run("gewinn", "--id", "H1", "--modell", "NF", "--metrik", "Flipzeit", "--nachher", "2.2",
                "--art", "gemessen")
            with open(p) as fh:
                d = json.load(fh)
            f = d["features"][0]
            self.assertEqual([z["sha"] for z in f["zweige"]], ["0123456789", "abcdef0123"])
            self.assertEqual([(g["metrik"], g["nachher"]) for g in f["gewinn"]], [("Flipzeit", "2.2")])
            self.assertEqual(f["schalter"][0], {"name": "SGLANG_H1", "art": "env", "gruppe": "D",
                                                "an_wert": "1", "default": "aus"})
            self.assertEqual(features.validate(d["features"]), [])


SAMPLE_27B = """# 27B-Ist-Werte (Stand 29.09.2026 ~07:10Z)

## F1–F22

| F | Feature | Status | Ist-Wert | Beleg / Grund |
|---|---|---|---|---|
| 1 | Flip | fertig+aktiv | flip_total **2167 ms** | z30j front.log `WEG2-FLIP` |
| 2 | Heterogen | INT8 fertig+aktiv; NVFP4 im Image aus | TP3 | z30j |
| 3 | nicht im Produkt | offen | – | – |

## Kreuztabelle 27B

| | tp | dcp | moe | pp | forma | kvonly |
|---|---|---|---|---|---|---|
| **tp** | am Metall belegt (24g) | nein | nein | nein | nur Desk | nur Desk |
| **dcp** | | am Metall belegt | nein | nein | nein | nur Desk (F15) |
| **moe** | | | nein | nein | nein | nein |
| **pp** | | | | am Metall belegt | nein | nein |
| **forma** | | | | | nur Desk | nur Desk |
| **kvonly** | | | | | | nur Desk |

### P1 Prefill

| Format / Form | Instrument | ≤8k | 8–32k | Beleg |
|---|---|---|---|---|
| INT8 PP3 | A, PP0 | 4452 (n=588) | 6564 | z30j P.log |

**Form ohne DCP/Schnitt, NVFP4 D-only (nvfp4form 09290630, rc12z30n):**

| Form | Tiefe | Text | bs1 | bs2 | bs3 | bs4 | bs5 | bs6 |
|---|---|---|---|---|---|---|---|---|
| 5090 solo | 2k | Code | 23,6 | 22,2 | 22,2 | 22,0 | 22,7 | 25,1 |
| 5090 solo | 10k | Prosa | 23,9 | ungültig | 23,0 | ungültig | ungültig | ungültig |

- Gemessene Runde (gpu-ms): bs1 27,8 (n=1961), bs6 62,0 (n=208).

| F | Marker (Quelle des Werts) | INT8 | NVFP4 | W4A8 |
|---|---|---|---|---|
| 2 Heterogen | state.json `groups.<P\\|D>.launch.argv` | ja | ja (argv) | ja (argv) |
| P2 Decode-Matrix | TP0 `Decode rank batch` | ja (Tiefe gemischt) | ja | ja |
"""


def _prod_doc():
    return {"features": [{"id": "B1", "modell": "beide", "titel": "Baustein", "fertig": True}],
            "produkt": [{"id": "F%d" % i, "nr": i, "titel": "t%d" % i, "soll": "s%d" % i, "ist": {},
                         "bausteine": ["B1"] if i == 1 else []} for i in (1, 2, 23, 24)]}


class ProduktCase(unittest.TestCase):
    def test_kreuz_key_is_axis_ordered_and_validated(self):
        self.assertEqual(features.kreuz_key("kvonly", "tp"), "tp+kvonly")
        d = _prod_doc()
        d["produkt"][1]["kreuztabelle"] = {"zellen": {"NF": {"kvonly+tp": {"status": "nein"},
                                                             "tp+dcp": {"status": "vielleicht"}}}}
        probs = features.validate_doc(d)
        self.assertTrue(any("kvonly+tp" in p for p in probs), probs)
        self.assertTrue(any("vielleicht" in p for p in probs), probs)

    def test_matrix_key_and_ist_status_validation(self):
        self.assertIsNone(features.matrix_key_error("Form A|1|kurz|code"))
        self.assertIn("bs", features.matrix_key_error("Form A|7|kurz|code"))
        self.assertIn("tiefe", features.matrix_key_error("Form A|1|99k|code"))
        d = _prod_doc()
        d["produkt"][0]["ist"] = {"NF": {"status": "halbfertig"}, "27B": {"status": "offen", "belegt_am": "gestern"}}
        probs = features.validate_doc(d)
        self.assertTrue(any("halbfertig" in p for p in probs), probs)
        self.assertTrue(any("belegt_am" in p for p in probs), probs)

    def test_unassigned_baustein_is_a_hint_not_a_save_blocker(self):
        d = _prod_doc()
        d["features"].append({"id": "B2", "modell": "NF", "titel": "neu"})
        self.assertEqual(features.validate_doc(d), [])
        self.assertEqual(len(features.unassigned_bausteine(d["produkt"], {"B1", "B2"})), 1)

    def test_view_joins_bausteine_and_marks_stale_ist(self):
        d = _prod_doc()
        d["produkt"][0]["ist"] = {"NF": {"status": "fertig+aktiv", "wert": "x", "belegt_am": "2026-09-29T05:48Z"},
                                  "27B": {"status": "fertig+aktiv", "wert": "y", "belegt_am": "2026-09-29T07:10Z"}}
        bs = {"NF": {"B1": {"titel": "Baustein", "fertig": True, "im_image": {"state": "ja"}, "aktiv": {"state": "an"},
                            "image_aber_aus": False, "aus_begruendung": None, "gewinn": [], "zweige": []}}}
        start = features.belegt_ts("2026-09-29T06:54Z")
        v = features.produkt_view(d["produkt"], bs, {"NF": start, "27B": start})
        f1 = v[0]
        self.assertEqual([p["nr"] for p in v], [1, 2, 23, 24])
        self.assertTrue(f1["ist"]["NF"]["veraltet"])
        self.assertFalse(f1["ist"]["27B"]["veraltet"])
        self.assertEqual(v[1]["ist"]["NF"]["status"], "unbelegt")
        self.assertIsNone(v[1]["ist"]["NF"].get("veraltet"))
        self.assertEqual(f1["bausteine"][0]["je_modell"]["NF"]["aktiv"], "an")

    def test_attach_current_value_or_named_marker(self):
        fv = {"models": [{"model": "NF", "boot": {"boot_s": 281, "format": "int4-mixed", "rc": "rc12z30u"}},
                         {"model": "27B", "boot": None}],
              "produkt": [{"id": "F17", "untertabelle": None}, {"id": "F9", "untertabelle": None},
                          {"id": "F12", "untertabelle": {"zeilen": [{"name": "INT4 (AutoRound)"}, {"name": "NVFP4"}]}}]}
        features.attach_current(fv, [], None)
        f17, f9, f12 = fv["produkt"]
        self.assertIn("281 s", f17["aktuell"]["NF"]["wert"])
        self.assertEqual(f17["aktuell"]["NF"]["je_format"], {"INT4": "dieser Boot", "NVFP4": "kein Boot in diesem Format"})
        self.assertEqual(f17["aktuell"]["27B"], {"leer": "kein Boot dieses Modells gefunden"})
        self.assertIn("Marker", f9["aktuell"]["NF"]["kein_instrument"])
        self.assertEqual([z["aktuell"]["NF"] for z in f12["untertabelle"]["zeilen"]],
                         ["läuft in diesem Boot", "kein Boot in diesem Format"])

    def test_formats_running_w4a8_rides_on_nvfp4_for_27b(self):
        self.assertEqual(features._formats_running("27B", "nvfp4-modelopt"), {"NVFP4", "W4A8"})
        self.assertEqual(features._formats_running("NF", "nvfp4-modelopt"), {"NVFP4"})
        self.assertEqual(features._formats_running("27B", None), set())

    def test_import_27b_fills_only_27b_with_source(self):
        with tempfile.TemporaryDirectory() as t:
            md = os.path.join(t, "features_27b_ist_0929.md")
            with open(md, "w") as fh:
                fh.write(SAMPLE_27B)
            d = _prod_doc()
            d["produkt"][0]["ist"]["NF"] = {"status": "offen"}
            features_update.import_27b(d, md)
            f1, f2, f23, f24 = d["produkt"]
            self.assertEqual(f1["ist"]["NF"], {"status": "offen"})
            self.assertEqual(f1["ist"]["27B"]["status"], "fertig+aktiv")
            self.assertEqual(f1["ist"]["27B"]["wert"], "flip_total 2167 ms")
            self.assertEqual(f1["ist"]["27B"]["quelle"], features_update.QUELLE_27B)
            self.assertEqual(f1["ist"]["27B"]["belegt_am"], "2026-09-29T07:10Z")
            self.assertEqual(f2["ist"]["27B"]["status"], "fertig+aktiv")   # earliest keyword wins
            k = f2["kreuztabelle"]["zellen"]["27B"]
            self.assertEqual(len(k), 21)
            self.assertEqual(k["tp+tp"], {"status": "am Metall belegt", "note": "24g", "quelle": features_update.QUELLE_27B})
            self.assertEqual(k["dcp+kvonly"]["note"], "F15")
            self.assertEqual(f23["untertabelle"]["zeilen"][0]["name"], "27B INT8 PP3")
            m = f24["matrix"]["zellen"]["27B"]
            self.assertEqual(m["NVFP4 5090 solo|1|2k|code"]["wert"], "23,6 ms")
            self.assertEqual(m["NVFP4 5090 solo|2|10k|prosa"]["status"], "ungültig")
            self.assertEqual(m["INT8 uneven DCP D TP3|6|gemischt|gemischt"]["wert"], "62,0 ms (n=208)")
            self.assertEqual(f2["marker"]["27B"]["marker"], "state.json groups.<P|D>.launch.argv")
            self.assertEqual(f24["marker"]["27B"]["je_format"]["INT8"], "ja (Tiefe gemischt)")
            self.assertEqual(features.validate_doc(d), [])

    def test_cli_produkt_ist_kreuz_matrix_override_and_md(self):
        with tempfile.TemporaryDirectory() as t:
            p = os.path.join(t, "f.json")
            with open(p, "w") as fh:
                json.dump(_prod_doc(), fh)
            run = lambda *a: features_update.main(["--file", p, *a])
            run("produkt-ist", "--id", "F1", "--modell", "NF", "--status", "fertig+aktiv", "--wert", "2,1 s",
                "--beleg", "Boot x", "--belegt-am", "2026-09-29T06:00Z")
            with self.assertRaises(SystemExit):
                run("produkt-ist", "--id", "F1", "--modell", "NF", "--status", "fertig+aktiv", "--belegt-am", "heute")
            run("kreuz", "--modell", "NF", "--a", "kvonly", "--b", "dcp", "--status", "nur Desk", "--note", "F15")
            run("matrix", "--id", "F24", "--modell", "NF", "--form", "Form A", "--bs", "1", "--tiefe", "kurz",
                "--text", "code", "--wert", "131,9 tok/s", "--boot", "x177", "--beleg", "Probe")
            run("matrix", "--id", "F24", "--modell", "NF", "--form", "Form A", "--bs", "2", "--tiefe", "kurz",
                "--text", "prosa", "--ungueltig", "--boot", "x177", "--beleg", "EOS < 500")
            run("zeile-ist", "--id", "F23", "--zeile", "97k", "--modell", "NF", "--status", "fertig+aktiv",
                "--wert", "24,18 s", "--beleg", "x175")
            run("set", "--id", "B2", "--modell", "NF", "--titel", "neu", "--produkt", "F2")
            run("boot-override", "--boot", "b-20260929T002052Z-86f6", "--lifecycle", "stopped (geplant)", "--beleg", "27B")
            with open(p) as fh:
                d = json.load(fh)
            P = {x["id"]: x for x in d["produkt"]}
            self.assertEqual(P["F1"]["ist"]["NF"]["belegt_am"], "2026-09-29T06:00Z")
            self.assertEqual(P["F2"]["kreuztabelle"]["zellen"]["NF"]["dcp+kvonly"], {"status": "nur Desk", "note": "F15"})
            self.assertEqual(P["F2"]["bausteine"], ["B2"])
            cells = P["F24"]["matrix"]["zellen"]["NF"]
            self.assertEqual(cells["Form A|2|kurz|prosa"]["status"], "ungültig")
            self.assertTrue(P["F23"]["untertabelle"]["zeilen"][0]["ist"]["NF"]["belegt_am"].startswith("20"))
            self.assertEqual(d["boot_overrides"]["b-20260929T002052Z-86f6"]["lifecycle"], "stopped (geplant)")
            out = os.path.join(t, "o.md")
            run("md", "--out", out, "--live-url", "")
            text = open(out).read()
            self.assertIn("| 1 | t1 | s1 |", text)
            self.assertIn("## F24 Decode-Matrix NF", text)
            self.assertIn("| kurz | code | 131,9 tok/s | ungemessen |", text)
            self.assertIn("| kurz | prosa | ungemessen | ungültig |", text)
            self.assertIn("## F24 Decode-Matrix 27B", text)
            self.assertIn("alle Zellen ungemessen", text)


if __name__ == "__main__":
    unittest.main()
