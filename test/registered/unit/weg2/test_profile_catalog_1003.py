"""PROFIL-EDITOR S1 (Auftrag 930): der Wertkatalog -- jede Erklaerung hat eine Quelle, jede Kante zeigt auf etwas, das es gibt.

Gepinnt:
  * Jeder kuratierte Flag existiert im Launcher-Parser oder in ServerArgs, jede kuratierte Env in ``environ.py`` (Ausnahme: gelesene
    Envs ohne ``Envs``-Feld, hier ``SGLANG_MOE_SCRATCH_SLOTS``, belegt durch Fundstelle im Launcher).
  * Jede Kante ``depends[].to`` zeigt auf einen existierenden Flag/Env/Profilvariable/Ablehnungscode; ``rel`` ist aus dem Vokabular.
  * Abdeckung sinkt nicht: die Erntequellen liefern Mindestzahlen (Flags des Launchers mit Hilfetext, ServerArgs, Env-Kommentare).
  * Die AST-Ernte liest ``help=`` roh, auch mit Konkatenation und ``%``.
  * Das ausgelieferte ``catalog.json`` des Dashboards stammt aus diesem Generator (Schema, Statistik, Verdrahtungsliste gleich dem Launcher).
  * Auftrag 990 W6 (Katalog ehrlich): nur gelesene Envs (``env_readers``) und die ``--dual-*``-Flags der 27B-Linie stehen mit der Herkunft
    ``nur-leser`` im Katalog; ohne Kommentar am Leser heisst die Erklaerung ``ohne Doku`` (Status bleibt ``unerklaert``), nichts wird erfunden.
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
                               os.path.join(SRT, "server_args.py"), python_dir=PY, extra_flags=PC.load_extra_flags())
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



def _tree(files):
    """A throw-away ``python/sglang/...`` tree: {relpath under python/: source}."""
    d = tempfile.mkdtemp()
    for rel, src in files.items():
        path = os.path.join(d, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(src)
    return d


class NurLeser(unittest.TestCase):
    """Auftrag 990 W6: the names code reads but nothing declares."""

    def test_reader_patterns(self):
        src = (
            "import os\n"
            "from os import environ\n"
            "X_ENV = 'SGLANG_VIA_CONST'\n"
            "# why this knob exists\n"
            "# and a second line\n"
            "a = os.environ.get('SGLANG_LITERAL', '7')\n"
            "b = os.getenv('HTSGLANG_GETENV')\n"
            "c = os.environ['SGLANG_SUBSCRIPT']\n"
            "d = 'SGLANG_MEMBER' in os.environ\n"
            "e = environ.get(X_ENV)\n"
            "f = _env_int('SGLANG_HELPER', 3)\n"
            "g = os.environ.get('NOT_OURS')\n"
            "h = os.environ.get('sglang_lower')\n"
            "os.environ['SGLANG_WRITE_ONLY'] = '1'\n"
        )
        d = _tree({"sglang/srt/mod.py": src,
                   "sglang/srt/environ.py": "import os\nz = os.environ.get('SGLANG_IN_ENVIRON_PY')\n",
                   "sglang/srt/tests/t.py": "import os\nos.environ.get('SGLANG_IN_TESTS')\n",
                   "sglang/srt/test_x.py": "import os\nos.environ.get('SGLANG_IN_TEST_FILE')\n"})
        r = PC.env_readers(d)
        self.assertEqual(sorted(r), ["HTSGLANG_GETENV", "SGLANG_HELPER", "SGLANG_LITERAL", "SGLANG_MEMBER", "SGLANG_SUBSCRIPT", "SGLANG_VIA_CONST"])
        self.assertEqual(r["SGLANG_LITERAL"]["default"], "7")
        self.assertEqual(r["SGLANG_LITERAL"]["comment"], "why this knob exists and a second line")
        self.assertEqual(r["SGLANG_LITERAL"]["line"], 6)
        self.assertEqual(r["SGLANG_HELPER"]["default"], "3")
        self.assertEqual(r["SGLANG_SUBSCRIPT"]["comment"], "")           # no comment above: nothing is made up

    def test_a_heading_is_no_explanation(self):
        d = _tree({"sglang/srt/mod.py": "import os\n# Global constants\nA = os.environ.get('SGLANG_HEAD')\n"})
        self.assertEqual(PC.env_readers(d)["SGLANG_HEAD"]["comment"], "")                # two words: a section heading, not a doc

    def test_entries_are_nur_leser_and_ohne_doku(self):
        d = _tree({"sglang/srt/mod.py": "import os\n# documented here in a full sentence about the knob\na = os.environ.get('SGLANG_DOCUMENTED')\nb = os.environ.get('SGLANG_BARE')\n",
                   "sglang/srt/weg2/launcher.py": "import argparse\nap = argparse.ArgumentParser()\nap.add_argument('--a', help='x')\n",
                   "sglang/srt/environ.py": "class Envs:\n    SGLANG_DECLARED = EnvBool(False)\n"})
        cat = PC.build_catalog(os.path.join(d, "sglang/srt/weg2/launcher.py"), os.path.join(d, "sglang/srt/environ.py"), {}, "t", python_dir=d,
                               extra_flags={"--dual-x": {"help": "", "bare": True, "default": False, "line": 5, "rev": "abc"}})
        e = cat["entries"]
        self.assertEqual(e["SGLANG_DOCUMENTED"]["source"]["kind"], "nur-leser")
        self.assertEqual(e["SGLANG_DOCUMENTED"]["status"], "geerntet")
        self.assertEqual(e["SGLANG_DOCUMENTED"]["help"], "documented here in a full sentence about the knob")
        self.assertEqual(e["SGLANG_BARE"]["status"], "unerklaert")
        self.assertEqual(e["SGLANG_BARE"]["help"], "")
        self.assertEqual(e["SGLANG_BARE"]["doc_note"], PC.NO_DOC)
        self.assertEqual(PC.NO_DOC, "ohne Doku")
        self.assertEqual(e["SGLANG_DECLARED"]["source"]["kind"], "environ")      # declared names are never nur-leser
        self.assertEqual(e["--dual-x"]["source"]["kind"], "nur-leser")
        self.assertEqual(e["--dual-x"]["doc_note"], "ohne Doku")
        self.assertEqual(cat["stats"]["by_source"], {"argparse/flag": 1, "environ/env": 1, "nur-leser/env": 2, "nur-leser/flag": 1})

    def test_multiline_environ_field_is_found(self):
        d = _tree({"sglang/srt/environ.py": "class Envs:\n    # two lines\n    SGLANG_MULTI = EnvBool(\n        _profile_default('SGLANG_MULTI', False))\n"
                                           "    SGLANG_ONE = EnvInt(3)  # trailing\n"})
        f = PC.environ_fields(os.path.join(d, "sglang/srt/environ.py"))
        self.assertEqual(sorted(f), ["SGLANG_MULTI", "SGLANG_ONE"])
        self.assertEqual(f["SGLANG_MULTI"]["comment"], "two lines")
        self.assertEqual(f["SGLANG_MULTI"]["kind"], "EnvBool")
        self.assertEqual(f["SGLANG_ONE"]["trailing"], "trailing")

    def test_real_tree_has_the_multiline_fields_and_the_dual_flags(self):
        envs = PC.environ_fields(os.path.join(SRT, "environ.py"))
        self.assertGreaterEqual(len(envs), 958)                                  # 942 one-line + 16 multi-line at 5444cf8cde
        for n in ("SGLANG_WEG2_VISION_FLIP_URGENT", "SGLANG_WEG2_CTL_KICK_AFTER_FLIP", "SGLANG_HICACHE_LOAD_ASYNC_INDEX"):
            self.assertIn(n, envs)
        snap = PC.load_extra_flags()
        self.assertEqual(len(snap), 17)
        self.assertTrue(all(k.startswith("--dual-") for k in snap))
        self.assertEqual({r["rev"] for r in snap.values()}, {"54a30199b1"})
        self.assertIn("--dual-share", snap)
        self.assertTrue(snap["--dual-share"]["help"])
        launcher = PC.launcher_flags(os.path.join(WEG2, "launcher.py"))
        self.assertFalse([k for k in snap if k in launcher])                     # only what this tree's launcher lacks

    def test_shipped_catalog_carries_the_honest_origins(self):
        if not os.path.isfile(SHIPPED):
            self.skipTest("no shipped catalog")
        with open(SHIPPED, encoding="utf-8") as fh:
            shipped = json.load(fh)
        st = shipped["stats"]["by_source"]
        self.assertEqual(sum(st.values()), shipped["stats"]["entries"])
        self.assertEqual(st["nur-leser/flag"], 17)
        self.assertGreater(st["nur-leser/env"], 500)
        self.assertGreaterEqual(st["environ/env"], 958)
        for n, e in shipped["entries"].items():
            if e["source"]["kind"] == "nur-leser":
                if e["status"] == "kuratiert":                                       # a curated text on top of the reader (SGLANG_MOE_SCRATCH_SLOTS)
                    self.assertTrue(e["text"], n)
                    continue
                self.assertTrue(e["help"] or e["doc_note"] == "ohne Doku", n)     # explained from code, or openly undocumented
                self.assertEqual(e["status"], "geerntet" if e["help"] else "unerklaert", n)
                self.assertTrue(e["help"] == "" or len(e["help"].split()) >= PC.MIN_DOC_WORDS, n)   # a heading is no explanation

    def test_the_editor_row_says_ohne_doku_but_stays_unexplained(self):
        spec = importlib.util.spec_from_file_location("t_profile_json_cat", os.path.join(WEG2, "profile_json.py"))
        pj = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = pj
        spec.loader.exec_module(pj)
        cat = {"SGLANG_BARE": {"help": "", "doc_note": "ohne Doku", "status": "unerklaert", "source": {"file": "a.py", "line": 1, "kind": "nur-leser"}}}
        ex = pj.explain_row({"name": "SGLANG_BARE"}, cat, None)
        self.assertEqual(ex["status"], "unerklaert")
        self.assertEqual(ex["note"], "ohne Doku")
        self.assertEqual(ex["parts"], [])


if __name__ == "__main__":
    unittest.main()
