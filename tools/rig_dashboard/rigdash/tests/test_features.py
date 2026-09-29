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


if __name__ == "__main__":
    unittest.main()
