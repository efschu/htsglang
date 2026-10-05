"""Profil-Editor: EIN Katalog über zwei Code-Bäume (27B- und NF-Linie) und die os.environ-Ernte (Auftrag 27B-Sitz 18:58Z, 05.10.).

Das Release-Image trägt zwei Code-Stände nebeneinander; der Editor-Katalog kam aus genau einem Baum, Schalter der anderen Linie fehlten.

Gepinnt:
  * ``build_union_catalog``: jeder Eintrag nennt ``baeume`` (in welchen Bäumen er steht); ein Name nur im zweiten Baum ist ``["nf"]``, nicht verloren.
  * ``abweichung`` erscheint genau dann, wenn Default oder Beschreibung zwischen den Bäumen verschieden sind, und nennt dann jeden Baum.
  * ``source.baum`` sagt, zu welchem Baum Datei:Zeile gehört.
  * Die os.environ-Ernte (``environ_constants``): ``*ENV*``-Konstante + Lesestelle (os.environ.get, Alias.get mit der Konstante, os.getenv, os.environ[...]),
    Default nur wo der Aufruf ihn wörtlich nennt; ein Name aus ``environ.py`` wird nicht doppelt geführt; ein Name, der nur in Docstring/Kommentar
    oder als nie benutzte Konstante steht, ist KEIN Verbraucher; ``dict.get("SGLANG_X")`` auf einem fremden Dict zählt nicht.
  * ``build_catalog`` ohne ``srt_dir`` bleibt, was es war (keine os.environ-Einträge).
  * Die Kommandozeile verlangt beide Bäume zusammen.
"""

import importlib.util
import os
import sys
import tempfile
import textwrap
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "python"))
WEG2 = os.path.join(PY, "sglang", "srt", "weg2")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PC = _load("t_pc_union", os.path.join(WEG2, "profile_catalog.py"))


def _w(root, rel, text):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(textwrap.dedent(text))


def _tree(base, label, environ_lines, flags=(), extra_files=None):
    """A minimal ``<base>/<label>/python`` tree: environ.py, launcher.py (``add_argument``), extra files under srt/."""
    py = os.path.join(base, label, "python")
    srt = os.path.join(py, "sglang", "srt")
    _w(srt, "environ.py", "class Envs:\n" + "".join("    %s\n" % ln for ln in environ_lines))
    _w(srt, "weg2/launcher.py", "def parser(ap):\n" + "".join("    ap.add_argument(%r, help=%r)\n" % f for f in flags) + "    return ap\n")
    for rel, text in (extra_files or {}).items():
        _w(srt, rel, text)
    return py


class Harvest(unittest.TestCase):
    def test_constant_and_reads(self):
        with tempfile.TemporaryDirectory() as d:
            py = _tree(d, "a", ["SGLANG_DECLARED = EnvBool(False)  # in environ.py"], extra_files={"weg2/m.py": '''
                import os
                # carries the clock over a settle
                CARRY_ENV = "SGLANG_CARRY"
                ALIAS_ENV = "SGLANG_ALIAS"
                GETENV_ENV = "SGLANG_VIA_GETENV"
                SUB_ENV = "SGLANG_VIA_SUB"
                UNUSED_ENV = "SGLANG_NEVER_USED"
                DECL_ENV = "SGLANG_DECLARED"

                def f(env=None):
                    env = os.environ if env is None else env
                    a = os.environ.get(CARRY_ENV, "1")
                    b = env.get(ALIAS_ENV, "0")
                    c = os.getenv(GETENV_ENV)
                    d = os.environ[SUB_ENV]
                    e = os.environ.get("SGLANG_LITERAL", "x")
                    other = {}
                    g = other.get("SGLANG_NOT_AN_ENV", 1)
                    h = env.get(DECL_ENV)
                    """SGLANG_ONLY_IN_A_DOCSTRING"""
                    # SGLANG_ONLY_IN_A_COMMENT
                '''})
            got = PC.environ_constants(os.path.join(py, "sglang", "srt"))
        self.assertEqual(got["SGLANG_CARRY"]["default"], '"1"')
        self.assertEqual(got["SGLANG_CARRY"]["comment"], "carries the clock over a settle")
        self.assertEqual(got["SGLANG_CARRY"]["file"], "weg2/m.py")
        self.assertEqual(got["SGLANG_ALIAS"]["default"], '"0"')
        self.assertIn("SGLANG_VIA_GETENV", got)
        self.assertIn("SGLANG_VIA_SUB", got)
        self.assertEqual(got["SGLANG_LITERAL"]["default"], '"x"')
        for not_a_consumer in ("SGLANG_NEVER_USED", "SGLANG_NOT_AN_ENV", "SGLANG_ONLY_IN_A_DOCSTRING", "SGLANG_ONLY_IN_A_COMMENT"):
            self.assertNotIn(not_a_consumer, got)

    def test_declared_name_is_not_listed_twice_and_single_tree_is_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            py = _tree(d, "a", ["SGLANG_DECLARED = EnvBool(False)  # in environ.py"],
                       extra_files={"weg2/m.py": 'import os\nDECL_ENV = "SGLANG_DECLARED"\nx = os.environ.get(DECL_ENV)\ny = os.environ.get("SGLANG_EXTRA", "7")\n'})
            launcher, environ, sargs = PC.find_tree_files(py)
            plain = PC.build_catalog(launcher, environ, {}, "t", sargs, edges_root=PY)
            withsrt = PC.build_catalog(launcher, environ, {}, "t", sargs, edges_root=PY, srt_dir=os.path.join(py, "sglang", "srt"))
        self.assertNotIn("SGLANG_EXTRA", plain["entries"])
        self.assertEqual(withsrt["entries"]["SGLANG_EXTRA"]["source"]["kind"], "os.environ")
        self.assertEqual(withsrt["entries"]["SGLANG_DECLARED"]["source"]["kind"], "environ")


class Union(unittest.TestCase):
    def _build(self, d, a_env, b_env, a_flags=(), b_flags=()):
        a = _tree(d, "a", a_env, a_flags)
        b = _tree(d, "b", b_env, b_flags)
        return PC.build_union_catalog([("27b", a), ("nf", b)], {}, {"27b": "r27", "nf": "rnf"}, edges_root=PY)

    def test_origin_per_entry_and_stats(self):
        with tempfile.TemporaryDirectory() as d:
            cat = self._build(d, ["SGLANG_ONLY_A = EnvBool(False)", "SGLANG_BOTH = EnvInt(1)"], ["SGLANG_BOTH = EnvInt(1)", "SGLANG_ONLY_B = EnvBool(True)"])
        e = cat["entries"]
        self.assertEqual((e["SGLANG_ONLY_A"]["baeume"], e["SGLANG_ONLY_B"]["baeume"], e["SGLANG_BOTH"]["baeume"]), (["27b"], ["nf"], ["27b", "nf"]))
        self.assertEqual(cat["stats"]["baeume"]["nur_27b"], 1)
        self.assertEqual((cat["stats"]["baeume"]["nur_nf"], cat["stats"]["baeume"]["beide"]), (1, 1))
        self.assertEqual(e["SGLANG_ONLY_B"]["source"]["baum"], "nf")
        self.assertEqual(cat["trees"]["nf"]["rev"], "rnf")

    def test_abweichung_only_when_the_trees_disagree(self):
        with tempfile.TemporaryDirectory() as d:
            cat = self._build(d, ["SGLANG_SAME = EnvBool(False)  # same", "SGLANG_DEF = EnvBool(False)", "SGLANG_HELP = EnvInt(1)  # one"],
                              ["SGLANG_SAME = EnvBool(False)  # same", "SGLANG_DEF = EnvBool(True)", "SGLANG_HELP = EnvInt(1)  # other"])
        e = cat["entries"]
        self.assertNotIn("abweichung", e["SGLANG_SAME"])
        self.assertEqual({k: v["default"] for k, v in e["SGLANG_DEF"]["abweichung"].items()}, {"27b": "False", "nf": "True"})
        self.assertEqual({k: v["help"] for k, v in e["SGLANG_HELP"]["abweichung"].items()}, {"27b": "one", "nf": "other"})
        self.assertEqual(cat["stats"]["baeume"]["abweichung"], 2)

    def test_flag_only_in_the_second_tree_is_kept(self):
        with tempfile.TemporaryDirectory() as d:
            cat = self._build(d, [], [], a_flags=[("--both", "h")], b_flags=[("--both", "h"), ("--nf-only", "nf flag")])
        self.assertEqual(cat["entries"]["--nf-only"]["baeume"], ["nf"])
        self.assertEqual(cat["entries"]["--both"]["baeume"], ["27b", "nf"])

    def test_cli_wants_both_trees(self):
        with self.assertRaises(SystemExit):
            PC.main(["--tree-27b", "/nonexistent"])


if __name__ == "__main__":
    unittest.main()
