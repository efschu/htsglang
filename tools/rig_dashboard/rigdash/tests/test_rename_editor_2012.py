"""Auftrag 2012 (Baustopper L1): der Profil-Editor im umbenannten fLLiper-Baum.

Das Image traegt python/flliper, srt/pdflip, FLLIPER_PDFLIP_*; der Editor kannte nur sglang/srt/weg2 und SGLANG_WEG2_* (Katalog 902 Mal).
``docker/weg2-release/rename_rigdash.py`` wendet die Umbenennungsmaschine des Release-Kits auf das Editor-Paket an. Gepinnt, am
UMBENANNTEN Paket (Kopie in einem Wegwerf-Verzeichnis, das Original bleibt unberuehrt):

  * das Skript laeuft durch (Kit-Pruefer ``verify`` PASS, zweiter Durchlauf aendert nichts, keine Reste alter Namen);
  * das alte Paket findet den umbenannten Planer-Baum NICHT (der Fehler, den es zu beheben galt), das neue schon -- auch das Start-Skript
    (``entrypoint_rigdash.sh --check``: alt -> rc 3 "kein Editor", neu -> Kommandozeile);
  * Importe der Editor-Module und Laden der umbenannten Planer-Module (profile_json, refusals) im umbenannten Paket;
  * Katalog-Roundtrip: jeder Name des Katalogs ist der abgebildete alte Name (902 SGLANG_WEG2_* -> FLLIPER_PDFLIP_*, kein alter uebrig);
    ein Release-Profil wird geladen, bearbeitet, als .env exportiert und von bash gegen die Werte des Profils geprueft;
  * die Boot-Aufzeichnungen (kartenplan_data) bleiben byte-gleich (ihre plan_id ist ein sha256 ueber den Inhalt) und rechnen nach;
  * Kopplung zum Kit-Schritt 2 (ident_fix.py): die Woerter seiner Tabelle, die der Editor liest, sind benannt (nur ``blockiert`` als
    Dashboard-eigener Statuswert; ein neues Wort bricht den Test und muss gegen den Planer-Baum geprueft werden).

Ohne das Kit (``/spinning/flliper/tools`` bzw. ``RELEASE_KIT_TOOLS``) wird uebersprungen.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.dirname(HERE))            # tools/rig_dashboard
REPO = os.path.dirname(os.path.dirname(PKG))
WRAP = os.path.join(REPO, "docker", "weg2-release", "rename_rigdash.py")
KIT = os.environ.get("RELEASE_KIT_TOOLS") or "/spinning/flliper/tools"
KIT_OK = (os.path.isfile(os.path.join(KIT, "rename_to_flliper.py"))
          and os.path.isfile(os.path.join(KIT, "release", "data", "merged_0928.json")) and os.path.isfile(WRAP))
FIX_RENAMED = os.path.join("rigdash", "tests", "fixtures", "kartenplan", "planner_tree", "python")


def run_py(root, code, env_extra=None, timeout=120):
    env = dict(os.environ, PYTHONPATH=root, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    env.pop("KARTENPLAN_TREE", None)
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-c", code], cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, timeout=timeout)


def map_name(n):
    """Die Namensregeln des Kits fuer ENV-Namen und Flags (nur was im Katalog vorkommt)."""
    n = n.replace("WEG2", "PDFLIP").replace("weg2", "pdflip").replace("Weg2", "PdFlip")
    return re.sub(r"(?<![Hh][Tt])SGLANG", "FLLIPER", n)


@unittest.skipUnless(KIT_OK, "Release-Kit nicht vorhanden (RELEASE_KIT_TOOLS)")
class RenamedEditor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="rn2012_")
        cls.src = os.path.join(cls.tmp, "src")
        cls.out = os.path.join(cls.tmp, "out")
        shutil.copytree(PKG, cls.src, ignore=shutil.ignore_patterns("__pycache__", ".git", "test_rename_editor_2012.py"))   # der Test selbst schreibt beide Namen
        r = subprocess.run([sys.executable, WRAP, cls.src, "--out", cls.out, "--kit", KIT], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, timeout=240)
        cls.rc, cls.log = r.returncode, r.stdout

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # ------------------------------------------------------------------ Skript
    def test_script_runs_through(self):
        self.assertEqual(self.rc, 0, self.log)
        self.assertIn("rename_rigdash: ok", self.log)

    def test_layout_is_the_renamed_one(self):
        base = os.path.join(self.out, "rigdash", "tests", "fixtures", "kartenplan", "planner_tree", "python")
        self.assertTrue(os.path.isfile(os.path.join(base, "flliper", "srt", "pdflip", "profile_json.py")))
        self.assertFalse(os.path.exists(os.path.join(base, "sglang")))
        # der Paketname bleibt (Dockerfile-Importprobe, `python -m rigdash`, COPY tools/rigdash)
        self.assertTrue(os.path.isfile(os.path.join(self.out, "rigdash", "__main__.py")))

    def test_script_refuses_an_unlisted_collision(self):
        """Eine NEUE Datei mit altem UND neuem Namen nebeneinander bricht ab (Exit 3), statt still zu verschmelzen."""
        d = os.path.join(self.tmp, "newcol")
        shutil.copytree(os.path.join(self.src, "kartenplan_build"), os.path.join(d, "kartenplan_build"))
        with open(os.path.join(d, "kartenplan_build", "neu.py"), "w") as fh:
            fh.write('A = "sglang"\nB = "flliper"\n')
        r = subprocess.run([sys.executable, WRAP, d, "--out", os.path.join(self.tmp, "newcol_out"), "--kit", KIT, "--dry-run"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
        self.assertEqual(r.returncode, 3, r.stdout)
        self.assertIn("kartenplan_build/neu.py", r.stdout)

    # ------------------------------------------------------------------ der Fehler und seine Behebung
    def test_old_package_does_not_find_the_renamed_tree_and_the_new_one_does(self):
        code = ("import os, sys\n"
                "from rigdash import profil as P\n"
                "P.TREE_CANDIDATES = ()\n"
                "t = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else %r\n"
                "print('TREE', P.find_tree(%r))\n")
        tree = os.path.join(self.out, FIX_RENAMED)
        old = run_py(self.src, code % (tree, tree))
        new = run_py(self.out, code % (tree, tree))
        self.assertIn("TREE None", old.stdout, old.stdout)             # der Befund L1: "kein Editor" im umbenannten Baum
        self.assertIn("TREE " + tree, new.stdout, new.stdout + old.stdout)

    def test_start_script_old_refuses_renamed_tree_new_starts(self):
        home = os.path.join(self.tmp, "home")
        os.makedirs(os.path.join(home, "src-27b"))
        os.symlink(os.path.join(self.out, "rigdash", "tests", "fixtures", "kartenplan", "planner_tree", "python"),
                   os.path.join(home, "src-27b", "python"))

        def start(root):
            env = dict(os.environ, RIGDASH_DIR=root, HOME_DIR=home, HTSGLANG_RIGDASH_LINE="27b", PY="python3")
            env.pop("HTSGLANG_RIGDASH", None)
            return subprocess.run(["bash", os.path.join(root, "entrypoint_rigdash.sh"), "--check"], env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, timeout=60)
        old, new = start(self.src), start(self.out)
        self.assertEqual(old.returncode, 3, old.stdout)
        self.assertIn("kein Editor", old.stdout)
        self.assertEqual(new.returncode, 0, new.stdout)
        self.assertIn("--editor-only", new.stdout)
        self.assertIn("--profil-tree", new.stdout)

    # ------------------------------------------------------------------ Import und Planer-Module
    def test_imports_and_planner_modules_load_in_the_renamed_package(self):
        tree = os.path.join(self.out, FIX_RENAMED)
        code = ("import os\n"
                "import rigdash.server, rigdash.profil, rigdash.profil_recompute, rigdash.hwprofil, rigdash.modellprofil, rigdash.kartenplan\n"
                "from rigdash import profil as P\n"
                "P.TREE_CANDIDATES = ()\n"
                "ed = P.ProfilEditor(kartenplaner=None, release_dir='/nonexistent', user_dir='/nonexistent', tree=%r)\n"
                "pj, rf = ed.mods()\n"
                "print('MODS', pj.SCHEMA, bool(rf.public_register()))\n") % tree
        r = run_py(self.out, code)
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("MODS flliper.server/1 True", r.stdout)

    # ------------------------------------------------------------------ Katalog
    def test_catalog_is_the_mapped_old_catalog(self):
        with open(os.path.join(PKG, "rigdash", "profil_data", "catalog.json"), encoding="utf-8") as fh:
            old = json.load(fh)
        with open(os.path.join(self.out, "rigdash", "profil_data", "catalog.json"), encoding="utf-8") as fh:
            new = json.load(fh)
        oe, ne = set(old["entries"]), set(new["entries"])
        self.assertEqual({map_name(n) for n in oe}, ne)
        self.assertGreaterEqual(sum(1 for n in oe if n.startswith("SGLANG_WEG2_")), 200)       # 277 Eintraege am 05.10.
        self.assertEqual([n for n in ne if n.startswith("SGLANG_WEG2_")], [])
        self.assertGreaterEqual(sum(1 for n in ne if n.startswith("FLLIPER_PDFLIP_")), 200)
        # im Rohtext (Beschreibungen, Quellen, Abhaengigkeiten): 902 Vorkommen alt, keins neu
        raw_old = open(os.path.join(PKG, "rigdash", "profil_data", "catalog.json"), encoding="utf-8").read()
        raw_new = open(os.path.join(self.out, "rigdash", "profil_data", "catalog.json"), encoding="utf-8").read()
        self.assertGreaterEqual(raw_old.count("SGLANG_WEG2_"), 800)
        self.assertEqual(raw_new.count("SGLANG_WEG2_"), 0)
        self.assertEqual(raw_new.count("FLLIPER_PDFLIP_"), raw_old.count("SGLANG_WEG2_"))
        self.assertEqual(len(ne), len(oe))                       # die Abbildung ist injektiv: nichts verschmolzen
        self.assertTrue(any(n.startswith("--pdflip-") for n in ne))
        # HTSGLANG_* ist die Produktebene und bleibt
        self.assertEqual({n for n in oe if n.startswith("HTSGLANG_")}, {n for n in ne if n.startswith("HTSGLANG_")})

    def test_roundtrip_load_edit_export_in_the_renamed_editor(self):
        code = r'''
import os, sys, tempfile
from rigdash.tests import test_profil_930 as T
tmp = tempfile.mkdtemp(prefix="rt2012_")
ed, rel, usr = T.editor(tmp)
r = ed.load("release", "demo")
rows = {x["key"]: x for x in r["view"]["rows"]}
env_keys = sorted(k for k in rows if k.startswith("env:P:FLLIPER_"))
assert env_keys, sorted(rows)
x = ed.export_env(r["doc"])
assert x["verified"], x["problems"]
k = env_keys[0]
e = ed.edit(r["doc"], [{"key": k, "op": "set", "value": "80,50,50"}])
x2 = ed.export_env(e["doc"])
assert x2["verified"], x2["problems"]
name = k.split(":", 2)[2]
assert name + "=80,50,50" in x2["env"], x2["env"][-400:]
assert "SGLANG_WEG2" not in x2["env"]
ex = rows["flag:--pp-stage-ratio"]["explain"]
assert ex["status"] == "kuratiert", ex["status"]
assert any(d["rel"] == "tauscht" for d in ex["depends"])
print("ROUNDTRIP", name, ex["status"], len(ex["depends"]))
'''
        r = run_py(self.out, code, timeout=180)
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("ROUNDTRIP FLLIPER_", r.stdout)

    # ------------------------------------------------------------------ gesperrte Daten
    def test_boot_records_stay_byte_identical_and_recompute(self):
        d = os.path.join("rigdash", "kartenplan_data")
        names = sorted(os.listdir(os.path.join(PKG, d)))
        self.assertTrue(names)
        for n in names:
            with open(os.path.join(PKG, d, n), "rb") as a, open(os.path.join(self.out, d, n), "rb") as b:
                self.assertEqual(a.read(), b.read(), n)
        code = ("from rigdash import kartenplan as K\n"
                "for rid in ('27b-int8', 'nf-int4-abl', '27b-nvfp4-dual'):\n"
                "    assert K.plan_id_ok(K.load_record(rid)['vram_plan']), rid\n"
                "print('PLANID ok')\n")
        r = run_py(self.out, code)
        self.assertIn("PLANID ok", r.stdout, r.stdout)

    # ------------------------------------------------------------------ Kopplung zu ident_fix (Kit-Schritt 2)
    def test_identfix_words_the_editor_reads_are_named(self):
        with open(os.path.join(KIT, "release", "data", "identfix_map.json"), encoding="utf-8") as fh:
            words = [k for k in json.load(fh) if not k.startswith("_")]
        files = [os.path.join("rigdash", n) for n in ("profil.py", "profil_recompute.py", "hwprofil.py", "modellprofil.py",
                                                      "kartenplan.py", "kartenplan_gate.py", "kartenplan_catalog.py",
                                                      "kartenplan_transport.py", "kartenplan_preview.py")]
        files += [os.path.join("rigdash", "static", n) for n in os.listdir(os.path.join(PKG, "rigdash", "static")) if n.endswith(".js")]
        files += [os.path.join("kartenplan_build", n) for n in os.listdir(os.path.join(PKG, "kartenplan_build")) if n.endswith(".py")]
        found = {}
        for f in files:
            p = os.path.join(PKG, f)
            if not os.path.isfile(p):
                continue
            txt = open(p, encoding="utf-8").read()
            for w in words:
                if re.search(r"(?<![A-Za-z0-9_])" + re.escape(w) + r"(?![A-Za-z0-9_])", txt):
                    found.setdefault(w, []).append(f)
        # `blockiert`: Statuswert der Force-Sicht im Dashboard selbst (profil.force_verdict), kein Planer-Schluessel
        self.assertEqual(sorted(found), ["blockiert"], found)


if __name__ == "__main__":
    unittest.main()
