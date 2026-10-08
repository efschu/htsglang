"""Auftrag 930 (Profil-Editor S1): der Entrypoint-Patch (FLLIPER_PROFILE=<Nutzerprofil.json>, FLLIPER_FORCE=1) aendert OHNE diese
Variablen nichts.

Der Entrypoint (/spinning/gpu-arb/docker/entrypoint.sh) liegt ausserhalb des Repos; der Patch ist als
``entrypoint.sh.profil-force-staged`` gestaged (nach dem Einspielen: Original als ``entrypoint.sh.bak_930``). Dieser Test laeuft BEIDE Fassungen
als Kopie in einer Sandbox (HOME_DIR/tmp umgebogen, kein GPU, kein Docker, nichts ausserhalb eines Temp-Verzeichnisses) bis unmittelbar vor
``cd "$TREE"`` und vergleicht, was der Entrypoint dem Launcher uebergeben wuerde:

  * dasselbe Release-Profil -> dasselbe ``LAUNCH``-argv, dieselbe Umgebung, dieselben Log-Zeilen (Zeitstempel normalisiert), derselbe rc,
    fuer NF, 27B und Dual, mit und ohne experimentelle Freigabe, ueber FLLIPER_PROFILE wie ueber HTSGLANG_PROFILE;
  * auch die Verweigerungen sind dieselben (unbekanntes Profil, Status ohne Freigabe: gleicher Text, rc 3);
  * FLLIPER_FORCE=1 haengt nur ``--force`` an das argv (und die FORCE-Zeile), sonst nichts;
  * Grenze: ``run_preflight`` (nvidia-smi, shm, MemAvailable, Baum-Pruefung, Modellpfade) laeuft in der Sandbox NICHT (in beiden Kopien durch
    ``TRANSPORT=bar1`` ersetzt): die dort umgestellten Schwellen (SHM, MEMAVAIL, STORE, HW-COUNT/-UNCALIBRATED des Karten-Gates) sind hier
    nicht belegt, sondern nur am Metall (Probe-Boot) und durch Lesen des .diff;
  * ein Nutzerprofil (JSON, aus dem Release-Profil importiert) ergibt ueber den Exporter dasselbe argv und dieselben Schalter wie das .env selbst;
  * unbekannt als Release UND als JSON bleibt die alte Verweigerung.
Fehlen die Dateien (anderer Rechner), wird uebersprungen.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_PY = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "python"))
DOCKER = "/spinning/gpu-arb/docker"
LIVE = os.path.join(DOCKER, "entrypoint.sh")
STAGED = os.path.join(DOCKER, "entrypoint.sh.profil-force-staged")
BAK = os.path.join(DOCKER, "entrypoint.sh.bak_930")
PROFILES = os.environ.get("PROFILES_RELEASE_DIR", os.path.join(DOCKER, "profiles_release"))
ORIG = BAK if os.path.isfile(BAK) else LIVE
NEW = STAGED if os.path.isfile(STAGED) else LIVE
ANCHOR = '\ncd "$TREE"\n\nif [ "$SUB" = "dryrun" ]'
TS = re.compile(r"\[htsglang-pdflip \d\d:\d\d:\d\d[Z]?\]")


def _ok():
    return all(os.path.isfile(p) for p in (ORIG, NEW)) and os.path.isdir(PROFILES) and shutil.which("bash")


@unittest.skipUnless(_ok(), "entrypoint / release profiles are outside the repo")
class EntrypointEquality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = tempfile.mkdtemp(prefix="ep930_")
        cls.addClassCleanup(shutil.rmtree, cls.t, True)
        cls.opt = os.path.join(cls.t, "opt")
        for stand in ("27b", "nf"):
            pk = os.path.join(cls.opt, "src-" + stand, "python", "flliper", "srt")
            os.makedirs(os.path.join(pk, "pdflip"))
            for d in (os.path.join(cls.opt, "src-" + stand, "python", "flliper"), pk, os.path.join(pk, "pdflip")):
                open(os.path.join(d, "__init__.py"), "w").close()
            shutil.copy(os.path.join(REPO_PY, "flliper", "srt", "pdflip", "profile_json.py"), os.path.join(pk, "pdflip", "profile_json.py"))
        shutil.copytree(PROFILES, os.path.join(cls.opt, "profiles"), ignore=shutil.ignore_patterns("*.bak*", "*.diff", "*.staged", "*-staged", "*.stale*", "*.json", "*.hwgeneric*", "*.fixb*", "*.abl*", "*.new*", "*vor_*"))
        cls.venv = os.path.join(cls.t, "venv")
        os.makedirs(os.path.join(cls.venv, "bin"))
        os.symlink(sys.executable, os.path.join(cls.venv, "bin", "python"))
        cls.users = os.path.join(cls.t, "users")
        os.makedirs(cls.users)
        cls.run_dir = os.path.join(cls.t, "run")
        os.makedirs(cls.run_dir)
        cls.scripts = {}
        for tag, src in (("orig", ORIG), ("new", NEW)):
            s = open(src, encoding="utf-8").read()
            assert s.count(ANCHOR) == 1, src
            dump = ('\nprintf "%s\\0" "${LAUNCH[@]}" > "$EP_T/launch.$EP_TAG"; env -0 | sort -z > "$EP_T/env.$EP_TAG"; '
                    'echo "SANDBOX-STOP" >&2; exit 0\ncd "$TREE"\n\nif [ "$SUB" = "dryrun" ]')
            s = s.replace(ANCHOR, dump)
            # the preflight (nvidia-smi, shm, memory, source tree check, model paths) needs the box: out of the sandbox, the same stub in both copies
            assert s.count("\nrun_preflight\nresolve_transport\n") == 1, src
            s = s.replace("\nrun_preflight\nresolve_transport\n", "\nTRANSPORT=bar1\n")
            s = s.replace("HOME_DIR=/opt/htsglang\n", "HOME_DIR=%s\n" % cls.opt, 1).replace("/tmp/htsglang", cls.run_dir)
            p = os.path.join(cls.t, "ep.%s.sh" % tag)
            open(p, "w").write(s)
            cls.scripts[tag] = p

    def run_ep(self, tag, **env):
        base = {"PATH": "/usr/bin:/bin", "HOME": os.path.join(self.t, "home"), "MODE": "pdflip", "HTSGLANG_EXPECT_GPUS": "0",
                "FLLIPER_PDFLIP_VENV": self.venv, "FLLIPER_PDFLIP_TMS_OUT_DIR": os.path.join(self.t, "tms"), "HTSGLANG_TAG": "dkrfixed",
                "EP_T": self.t, "EP_TAG": tag, "HTSGLANG_ALLOW_EXPERIMENTAL": "1"}
        base.update(env)
        for f in ("launch", "env"):
            try:
                os.unlink(os.path.join(self.t, "%s.%s" % (f, tag)))
            except OSError:
                pass
        p = subprocess.run(["bash", self.scripts[tag]], capture_output=True, text=True, env=base, errors="replace", timeout=120)
        out = TS.sub("[htsglang-pdflip TS]", p.stderr)
        res = {"rc": p.returncode, "log": out, "launch": None, "env": None}
        for f in ("launch", "env"):
            fp = os.path.join(self.t, "%s.%s" % (f, tag))
            if os.path.isfile(fp):
                raw = open(fp, "rb").read()
                if f == "env":                                   # the harness's own variables are not the entrypoint's
                    raw = b"\0".join(x for x in raw.split(b"\0") if x and not x.startswith((b"EP_TAG=", b"EP_T=", b"_=", b"OLDPWD="))) + b"\0"
                res[f] = raw
        return res

    def both(self, **env):
        a, b = self.run_ep("orig", **env), self.run_ep("new", **env)
        return a, b

    def test_release_profiles_are_identical_without_the_new_variables(self):
        names = [f[:-4] for f in sorted(os.listdir(os.path.join(self.opt, "profiles"))) if f.endswith(".env")]
        self.assertGreaterEqual(len(names), 5)
        reached = 0
        for n in names:
            for var in ("HTSGLANG_PROFILE", "FLLIPER_PROFILE"):
                a, b = self.both(**{var: n})
                self.assertEqual(a["rc"], b["rc"], (n, var))
                self.assertEqual(a["launch"], b["launch"], (n, var))
                self.assertEqual(a["env"], b["env"], (n, var))
                self.assertEqual(a["log"], b["log"], (n, var))
                reached += a["launch"] is not None
        self.assertGreater(reached, 8)                                   # the sandbox really reached the LAUNCH line

    def test_refusals_are_unchanged(self):
        for env in ({"HTSGLANG_PROFILE": "nf-bogus"}, {"HTSGLANG_PROFILE": "bogus"}, {"FLLIPER_PROFILE": ""},
                    {"HTSGLANG_PROFILE": "27b-nvfp4", "HTSGLANG_ALLOW_EXPERIMENTAL": "0"},
                    {"HTSGLANG_PROFILE": "nf", "HTSGLANG_ALLOW_EXPERIMENTAL": "0"}):
            a, b = self.both(**env)
            self.assertEqual((a["rc"], a["log"]), (b["rc"], b["log"]), env)

    def test_force_adds_only_the_flag(self):
        a, b = self.both(HTSGLANG_PROFILE="nf-int4")
        f = self.run_ep("new", HTSGLANG_PROFILE="nf-int4", FLLIPER_FORCE="1")
        self.assertEqual(f["rc"], 0)
        self.assertEqual(f["launch"], b["launch"] + b"--force\0")
        self.assertIn("FORCE:", f["log"])
        envs = {x for x in f["env"].split(b"\0") if x}
        base = {x for x in b["env"].split(b"\0") if x}
        self.assertEqual(sorted(envs - base), [b"HTSGLANG_FORCE=1"])    # FLLIPER_FORCE was mapped to HTSGLANG_FORCE, nothing else moved
        self.assertEqual(base - envs, set())

    def test_force_lifts_the_status_refusal_and_says_so(self):
        a = self.run_ep("orig", HTSGLANG_PROFILE="27b-nvfp4", HTSGLANG_ALLOW_EXPERIMENTAL="0")
        self.assertEqual(a["rc"], 3)
        f = self.run_ep("new", HTSGLANG_PROFILE="27b-nvfp4", HTSGLANG_ALLOW_EXPERIMENTAL="0", FLLIPER_FORCE="1")
        self.assertEqual(f["rc"], 0, f["log"][-400:])
        self.assertIn("FORCED-PAST PROFIL-STATUS", f["log"])

    def test_a_user_profile_reaches_the_same_launch_as_its_release_profile(self):
        sys.path.insert(0, REPO_PY)
        import importlib.util

        spec = importlib.util.spec_from_file_location("pj930", os.path.join(REPO_PY, "flliper", "srt", "pdflip", "profile_json.py"))
        pj = importlib.util.module_from_spec(spec)
        sys.modules["pj930"] = pj
        spec.loader.exec_module(pj)
        import json

        for rel in ("nf-int4", "27b"):
            doc = pj.import_env(os.path.join(self.opt, "profiles", rel + ".env"))
            doc["name"] = "mein-" + rel
            with open(os.path.join(self.users, "mein-%s.json" % rel), "w") as fh:
                json.dump(doc, fh)
            base = self.run_ep("new", HTSGLANG_PROFILE=rel)
            user = self.run_ep("new", FLLIPER_PROFILE="mein-" + rel, FLLIPER_PROFILES_DIR=self.users)
            self.assertEqual(user["rc"], 0, user["log"][-600:])
            self.assertIn("Nutzerprofil", user["log"])
            self.assertEqual(user["launch"].replace(b"mein-" + rel.encode(), rel.encode()), base["launch"], rel)
            drop = lambda e: sorted(x for x in e.split(b"\0") if x and not x.startswith((b"HTSGLANG_PROFILE=", b"HTSGLANG_TAG=", b"HTSGLANG_PROFILES_DIR=", b"PROFILE_", b"OLDPWD", b"_=")))
            self.assertEqual(drop(user["env"]), drop(base["env"]), rel)

    def test_unknown_profile_stays_refused(self):
        a, b = self.both(HTSGLANG_PROFILE="nf-gibt-es-nicht")
        self.assertEqual((a["rc"], a["log"]), (b["rc"], b["log"]))
        self.assertEqual(b["rc"], 3)


if __name__ == "__main__":
    unittest.main()
