"""PROFIL-EDITOR S1 (Auftrag 930): der Wertkatalog -- jede Erklaerung hat eine Quelle, jede Kante zeigt auf etwas, das es gibt.

Gepinnt:
  * Jeder kuratierte Flag existiert im Launcher-Parser oder in ServerArgs, jede kuratierte Env in ``environ.py`` (Ausnahme: gelesene
    Envs ohne ``Envs``-Feld, hier ``FLLIPER_MOE_SCRATCH_SLOTS``, belegt durch Fundstelle im Launcher).
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
PDFLIP = os.path.join(PY, "flliper", "srt", "pdflip")
SRT = os.path.join(PY, "flliper", "srt")
SHIPPED = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "tools", "rig_dashboard", "rigdash", "profil_data", "catalog.json"))


#: line probe (module exists, never a sha or a branch name): the Dual form (dual_green.py, --dual-*) is a 27B-line feature
DUAL_LINE = os.path.isfile(os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "python")),
                                        "flliper", "srt", "pdflip", "dual_green.py"))
TREE = "27b" if DUAL_LINE else "nf"
#: curated names the code of the NF line does not carry (measured 07.10. on 2e68b3f94b: curated 119 entries, these 26 are neither a flag of the
#: launcher / server_args nor an env of environ.py nor a literal of the launcher): the Dual form and four guard envs of the 27B line.  The
#: curated catalog is shared by both lines (the dashboard serves both, the shipped catalog is the union); on the NF line they stay curated
#: texts without a code behind them, and this test names them instead of passing them silently.
ABSENT_ON_NF_LINE = frozenset((
    "--dual-p-overhead-mib", "--dual-d-prefill-tokens", "--dual-p-duty", "--dual-p-sm-pct", "--dual-priority", "--dual-d-min-rate-tps",
    "--dual-p-min-share", "--dual-share-actuators", "--dual-green-ladder", "--dual-d-capture-prio", "--dual-p-mps-low-prio", "--dual-p-sleep",
    "--dual-unified-kv", "--dual-p-kv-max-tokens", "--dual-d-kv-max-tokens", "--dual-mps",
    "FLLIPER_PDFLIP_DUAL_SHARE_GREEN_TABLE", "FLLIPER_PDFLIP_DUAL_SHARE_STARVE_AGE_S", "FLLIPER_PDFLIP_DUAL_SHARE_STARVE_MAX_RUNG",
    "FLLIPER_PDFLIP_DUAL_GRANT_RETRY_MS", "FLLIPER_PDFLIP_DUAL_D_COMPACT", "FLLIPER_PDFLIP_DUAL_ARENA_AUX_SPILL_S",
    "FLLIPER_PDFLIP_HOST_GUARD_W22", "FLLIPER_PDFLIP_HOST_GUARD_W98", "FLLIPER_ADMISSION_WEDGE_MODE", "FLLIPER_PREFILL_LIVELOCK_MODE"))
#: curated names the code of the 27B line does not carry: the NF line's own curated entry (``baeume`` ["nf"] in the catalog; measured 07.10. on
#: 65fc0e2076: --pdflip-xchg-census-map is neither a launcher flag nor a server_args flag of the 27B tree).  It stays curated (user/27B seat 07.10.:
#: do not delete it) and this test names it.  The curated total is therefore 119 on both lines (118 + this entry).
ABSENT_ON_27B_LINE = frozenset(("--pdflip-xchg-census-map",))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PC = _load("t_profile_catalog", os.path.join(PDFLIP, "profile_catalog.py"))
CU = _load("t_profile_catalog_curated", os.path.join(PDFLIP, "profile_catalog_curated.py"))
RF = _load("t_refusals_cat", os.path.join(PDFLIP, "refusals.py"))
RELS = {"tauscht", "braucht", "schliesst_aus", "abgeleitet_von", "skaliert_mit"}


class Harvest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.flags = PC.launcher_flags(os.path.join(PDFLIP, "launcher.py"))
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
            fh.write("# reason one\n# reason two\nPROFILE_X=1\nPROFILE_ARGS=(\n  # why this\n  --p-bs 2   # trailing note\n)\n_form FLLIPER_Q 1  # q\n")
        c = PC.harvest_profile_comments(p)
        self.assertEqual(c["PROFILE_X"]["text"], "reason one reason two")
        self.assertIn("why this", c["--p-bs"]["text"])
        self.assertIn("trailing note", c["--p-bs"]["text"])
        self.assertEqual(c["FLLIPER_Q"]["text"], "q")


class Curated(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.flags = PC.launcher_flags(os.path.join(PDFLIP, "launcher.py"))
        cls.server = PC.server_args_flags(os.path.join(SRT, "server_args.py"))
        cls.envs = PC.environ_fields(os.path.join(SRT, "environ.py"))
        with open(os.path.join(PDFLIP, "launcher.py"), encoding="utf-8") as fh:
            cls.launcher_src = fh.read()
        # Envs ohne Envs-Feld, die ein Modul ueber os.environ liest (Konstante AUX_ENV usw.): dieselbe Ernte, die der Katalog-Generator nutzt.
        cls.env_constants = PC.environ_constants(SRT)

    def test_every_curated_name_exists(self):
        miss = []
        for name, c in CU.CURATED.items():
            if c["kind"] == "flag" and name not in self.flags and name not in self.server:
                miss.append(name)
            if c["kind"] == "env" and name not in self.envs and name not in self.launcher_src and not self._composed_env_read(name) \
                    and name not in self.env_constants:
                miss.append(name)
        if DUAL_LINE:       # 27B line: exactly the NF-only curated entry (the curated catalog is shared by both lines), no more and no fewer
            self.assertEqual(sorted(miss), sorted(ABSENT_ON_27B_LINE))
        else:       # NF line: exactly the named names, no more (a new miss is a defect) and no fewer (a stale entry hides one)
            self.assertEqual(sorted(miss), sorted(ABSENT_ON_NF_LINE))

    #: Envs, die der Code aus ``ENV_PREFIX + "<NAME>"`` zusammensetzt (der volle Name steht nirgends als Literal): Name -> (Datei im pdflip-Verzeichnis,
    #: Quelltext der Stelle, die das Suffix liest).  AP-H1 (Dual-ENV-Tabelle): dual_green.py / dual_share.py lesen sie ueber ``g("TABLE")`` bzw. die Namensliste.
    COMPOSED_ENV = {
        "FLLIPER_PDFLIP_DUAL_SHARE_GREEN_TABLE": ("dual_green.py", 'g("TABLE")'),
        "FLLIPER_PDFLIP_DUAL_SHARE_STARVE_AGE_S": ("dual_share.py", '("STARVE_AGE_S", "starve_age_s", float)'),
        "FLLIPER_PDFLIP_DUAL_SHARE_STARVE_MAX_RUNG": ("dual_share.py", '("STARVE_MAX_RUNG", "starve_max_rung", int)'),
    }

    def _composed_env_read(self, name):
        """Ein zusammengesetzter Env-Name gilt als belegt, wenn der Praefix im Code steht UND die Lesestelle des Suffixes im genannten Quelltext."""
        hit = self.COMPOSED_ENV.get(name)
        if hit is None:
            return False
        prefix, suffix_src = "FLLIPER_PDFLIP_DUAL_SHARE_", hit[1]
        if not os.path.isfile(os.path.join(PDFLIP, "dual_share.py")) or not os.path.isfile(os.path.join(PDFLIP, hit[0])):
            return False                                   # the Dual form is not on this line
        with open(os.path.join(PDFLIP, "dual_share.py"), encoding="utf-8") as fh:
            if 'ENV_PREFIX = "%s"' % prefix not in fh.read():
                return False
        with open(os.path.join(PDFLIP, hit[0]), encoding="utf-8") as fh:
            return suffix_src in fh.read() and name.startswith(prefix)

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
        cat = PC.build_catalog(os.path.join(PDFLIP, "launcher.py"), os.path.join(SRT, "environ.py"), CU.CURATED, "t",
                               os.path.join(SRT, "server_args.py"))
        st = cat["stats"]
        self.assertEqual(st["kuratiert"], len(CU.CURATED))
        self.assertGreater(st["geerntet"], 1000)
        self.assertEqual(cat["entries"]["--pp-stage-ratio"]["status"], "kuratiert")
        self.assertEqual(cat["entries"]["--p-hostgap"]["status"], "kuratiert")
        with open(os.path.join(PDFLIP, "launcher.py"), encoding="utf-8") as fh:
            wired = RF.wired_codes(fh.read())
        self.assertEqual(cat["register_wired"], wired)
        if os.path.isfile(SHIPPED):
            with open(SHIPPED, encoding="utf-8") as fh:
                shipped = json.load(fh)
            self.assertEqual(shipped["schema"], PC.SCHEMA)
            # The shipped file is the catalog over BOTH code trees (--tree-27b/--tree-nf, see test_profile_catalog_union_1005): it has more entries than
            # this tree alone.  What is tagged with THIS tree's label must be exactly this tree's harvest: when the tree moves (flag, env, os.environ
            # read site), this check goes red = rebuild catalog.json (command and tree revisions are in the commit message of the file).
            self.assertEqual(sorted(shipped["trees"]), ["27b", "nf"])
            self.assertEqual(shipped["stats"]["kuratiert"], st["kuratiert"])
            here = PC._harvest(os.path.join(PDFLIP, "launcher.py"), os.path.join(SRT, "environ.py"), os.path.join(SRT, "server_args.py"), SRT)
            tagged = {n for n, e in shipped["entries"].items() if TREE in e.get("baeume", [])}
            self.assertEqual(tagged, set(here))
            if DUAL_LINE:
                # the file's ``register_wired`` is the FIRST tree's (27B launcher): comparable only on the 27B line (the NF dashboard reads the wired
                # codes from the planner tree's own launcher: test_profil_force_katalog_2002)
                self.assertEqual(shipped["register_wired"], wired)             # regenerate: python -m flliper.srt.pdflip.profile_catalog
            # the edge anchors are checked in BOTH trees at build time and the file says so: no stale anchor in either tree
            self.assertEqual({lb: v["problem"] for lb, v in shipped["kanten"]["beleg_aufloesung_baeume"].items()}, {"27b": [], "nf": []})


if __name__ == "__main__":
    unittest.main()
