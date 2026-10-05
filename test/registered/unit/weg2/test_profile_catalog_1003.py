"""PROFIL-EDITOR S1 (Auftrag 930): der Wertkatalog -- jede Erklaerung hat eine Quelle, jede Kante zeigt auf etwas, das es gibt.

Gepinnt:
  * Jeder kuratierte Flag existiert im Launcher-Parser oder in ServerArgs, jede kuratierte Env in ``environ.py`` (Ausnahme: gelesene
    Envs ohne ``Envs``-Feld, hier ``SGLANG_MOE_SCRATCH_SLOTS``, belegt durch Fundstelle im Launcher).
  * Jede Kante ``depends[].to`` zeigt auf einen existierenden Flag/Env/Profilvariable/Ablehnungscode; ``rel`` ist aus dem Vokabular.
  * Abdeckung sinkt nicht: die Erntequellen liefern Mindestzahlen (Flags des Launchers mit Hilfetext, ServerArgs, Env-Kommentare).
  * Die AST-Ernte liest ``help=`` roh, auch mit Konkatenation und ``%``.
  * Das ausgelieferte ``catalog.json`` des Dashboards stammt aus diesem Generator (Schema, Statistik, Verdrahtungsliste gleich dem Launcher).
"""

import importlib.util
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "python"))
WEG2 = os.path.join(PY, "sglang", "srt", "weg2")
SRT = os.path.join(PY, "sglang", "srt")
SHIPPED = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "tools", "rig_dashboard", "rigdash", "profil_data", "catalog.json"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PC = _load("t_profile_catalog", os.path.join(WEG2, "profile_catalog.py"))
CU = _load("t_profile_catalog_curated", os.path.join(WEG2, "profile_catalog_curated.py"))
RF = _load("t_refusals_cat", os.path.join(WEG2, "refusals.py"))
RELS = {"tauscht", "braucht", "schliesst_aus", "abgeleitet_von", "skaliert_mit"}


class Harvest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.flags = PC.launcher_flags(os.path.join(WEG2, "launcher.py"))
        cls.server = PC.server_args_flags(os.path.join(SRT, "server_args.py"))
        cls.envs = PC.environ_fields(os.path.join(SRT, "environ.py"))

    def test_minimum_harvest(self):
        self.assertGreaterEqual(len(self.flags), 180)
        self.assertGreaterEqual(sum(1 for r in self.flags.values() if r["help"]), 170)
        self.assertGreaterEqual(len(self.server), 500)
        self.assertGreaterEqual(len(self.envs), 900)
        self.assertGreaterEqual(sum(1 for r in self.envs.values() if r["comment"]), 550)

    def test_help_is_read_raw(self):
        self.assertIn("OVERRIDE the solved layer cut", self.flags["--pp-stage-ratio"]["help"])
        self.assertTrue(self.flags["--p-hostgap"]["bare"])
        self.assertEqual(self.flags["--d-reshard"]["choices"], ["off", "wake", "wake-seg", "live"])
        self.assertIn("MoE", self.server["--rank-moe-ratio"]["help"])
        self.assertEqual(self.server["--chunked-prefill-size"]["help"][:20], "The maximum number o")

    def test_arg_specs_feed_the_tokenizer(self):
        sp = PC.arg_specs(self.flags)
        self.assertTrue(sp["--p-hostgap"]["bare"])
        self.assertFalse(sp["--pp-stage-ratio"]["bare"])

    def test_profile_comments_are_found_above_the_line(self):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "x.env")
        with open(p, "w") as fh:
            fh.write("# reason one\n# reason two\nPROFILE_X=1\nPROFILE_ARGS=(\n  # why this\n  --p-bs 2   # trailing note\n)\n_form SGLANG_Q 1  # q\n")
        c = PC.harvest_profile_comments(p)
        self.assertEqual(c["PROFILE_X"]["text"], "reason one reason two")
        self.assertIn("why this", c["--p-bs"]["text"])
        self.assertIn("trailing note", c["--p-bs"]["text"])
        self.assertEqual(c["SGLANG_Q"]["text"], "q")


class Curated(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.flags = PC.launcher_flags(os.path.join(WEG2, "launcher.py"))
        cls.server = PC.server_args_flags(os.path.join(SRT, "server_args.py"))
        cls.envs = PC.environ_fields(os.path.join(SRT, "environ.py"))
        with open(os.path.join(WEG2, "launcher.py"), encoding="utf-8") as fh:
            cls.launcher_src = fh.read()

    def test_every_curated_name_exists(self):
        miss = []
        for name, c in CU.CURATED.items():
            if c["kind"] == "flag" and name not in self.flags and name not in self.server:
                miss.append(name)
            if c["kind"] == "env" and name not in self.envs and name not in self.launcher_src:
                miss.append(name)
        self.assertEqual(miss, [])

    def test_every_edge_points_at_something_that_exists(self):
        codes = {r.code for r in RF.REGISTER}
        bad = []
        for name, c in CU.CURATED.items():
            for d in c["depends"]:
                self.assertIn(d["rel"], RELS, name)
                self.assertIn(d["calc"], ("text", "S4"), name)
                self.assertTrue(d["effect"], name)
                t = d["to"]
                ok = t in CU.CURATED or t in self.flags or t in self.server or t in self.envs or t in codes
                if not ok:
                    bad.append((name, t))
        self.assertEqual(bad, [])

    def test_every_curated_entry_is_explained_and_core_values_have_a_price(self):
        for name, c in CU.CURATED.items():
            self.assertGreater(len(c["text"]), 15, name)
            self.assertIn(c["level"], ("einfach", "experte"))
        for name in ("--pp-stage-ratio", "--rank-gpu-memory-mib", "--rank-moe-resident-fraction", "--p-chunk-policy"):
            self.assertTrue(CU.CURATED[name]["gain"] and CU.CURATED[name]["cost"], name)

    def test_the_users_trades_are_there(self):
        """Layer-Schnitt <-> KV/Kontext, Experten <-> KV, Chunk <-> Aktivierung (Nutzer-Order 03.10.)."""
        def edge(a, b, rel):
            return any(d["to"] == b and d["rel"] == rel for d in CU.CURATED[a]["depends"])
        self.assertTrue(edge("--pp-stage-ratio", "--rank-gpu-memory-mib", "tauscht"))
        self.assertTrue(edge("--rank-moe-resident-fraction", "--rank-gpu-memory-mib", "tauscht"))
        self.assertTrue(edge("--p-chunk-max", "--rank-gpu-memory-mib", "tauscht"))
        self.assertTrue(edge("--pp-stage-ratio", "--pp-attn-stage-ratio", "braucht"))


class Build(unittest.TestCase):
    def test_build_and_shipped_catalog_agree(self):
        cat = PC.build_catalog(os.path.join(WEG2, "launcher.py"), os.path.join(SRT, "environ.py"), CU.CURATED, "t",
                               os.path.join(SRT, "server_args.py"))
        st = cat["stats"]
        self.assertEqual(st["kuratiert"], len(CU.CURATED))
        self.assertGreater(st["geerntet"], 1000)
        self.assertEqual(cat["entries"]["--pp-stage-ratio"]["status"], "kuratiert")
        self.assertEqual(cat["entries"]["--p-hostgap"]["status"], "kuratiert")
        with open(os.path.join(WEG2, "launcher.py"), encoding="utf-8") as fh:
            wired = RF.wired_codes(fh.read())
        self.assertEqual(cat["register_wired"], wired)
        if os.path.isfile(SHIPPED):
            with open(SHIPPED, encoding="utf-8") as fh:
                shipped = json.load(fh)
            self.assertEqual(shipped["schema"], PC.SCHEMA)
            self.assertEqual(shipped["register_wired"], wired)             # regenerate: python -m sglang.srt.weg2.profile_catalog
            self.assertEqual(shipped["stats"]["kuratiert"], st["kuratiert"])
            self.assertEqual(set(shipped["entries"]), set(cat["entries"]))


if __name__ == "__main__":
    unittest.main()
