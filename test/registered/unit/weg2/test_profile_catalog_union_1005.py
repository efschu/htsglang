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

    def test_edge_anchors_are_checked_in_every_tree_and_baeume_marks_the_other_lines_code(self):
        """Zwei Linien (NF-Katalog 07.10.): der Katalog nennt je Baum, ob alle Kanten-Anker aufloesen (``kanten.beleg_aufloesung_baeume``).  Eine Kante mit
        ``baeume: [27b]`` ist im NF-Baum ``andere_linie`` (kein Problem, Datei und Anker dort nie gesucht), eine Kante OHNE ``baeume`` mit fehlendem Anker
        bleibt ein Problem dieses Baums -- ``baeume`` ist der einzige Ausweg, nie ein stilles Ueberspringen."""
        import json
        with tempfile.TemporaryDirectory() as d:
            a = _tree(d, "a", [], [("--x", "hx"), ("--y", "hy"), ("--dualonly", "hd")])
            b = _tree(d, "b", [], [("--x", "hx"), ("--y", "hy")])
            mk = lambda i, von, nach, anker, **kw: dict({"id": i, "von": von, "nach": nach, "rel": "braucht", "calc": "text", "wert": None, "satz": "s",
                                                         "beleg": {"datei": "python/sglang/srt/weg2/launcher.py", "zeile": 2, "anker": anker}}, **kw)
            edges = {"schema": PC.EDGES_SCHEMA, "kanten": [mk("K1", "--x", "--y", "add_argument('--x'"),
                                                           mk("K2", "--dualonly", "--x", "add_argument('--dualonly'", baeume=["27b"]),
                                                           mk("K3", "--y", "--x", "add_argument('--dualonly'")]}
            ep = os.path.join(d, "edges.json")
            with open(ep, "w", encoding="utf-8") as fh:
                json.dump(edges, fh)
            cat = PC.build_union_catalog([("27b", a), ("nf", b)], {}, {"27b": "r27", "nf": "rnf"}, edges_path=ep)
        per = cat["kanten"]["beleg_aufloesung_baeume"]
        self.assertEqual(per["27b"]["problem"], [])
        self.assertEqual(per["nf"]["problem"], ["K3"])                                   # K3: no baeume, anchor missing in the NF tree
        self.assertEqual(per["nf"]["status"], {"eindeutig": 1, "andere_linie": 1, "veraltet": 1})
        dep = {x["kante"]: x for e in cat["entries"].values() for x in e["depends"]}
        self.assertEqual(dep["K2"]["baeume"], ["27b"])
        self.assertNotIn("baeume", dep["K1"])
        self.assertEqual(dep["K1"]["beleg"]["aufloesung"], "eindeutig")                    # the shown line is the FIRST tree's (27b)

    def test_output_carries_no_build_path(self):
        """Reproduzierbarer Bau (Wunsch 27B-Sitz 05.10.): derselbe Quellstand ergibt dieselbe Datei, egal in welchem Verzeichnis die Bäume liegen.
        Ein Baum-Pfad in der Ausgabe (``trees.<baum>.python_dir``) machte sie von Maschine zu Maschine verschieden."""
        import json
        with tempfile.TemporaryDirectory() as d:
            cat = self._build(d, ["SGLANG_A = EnvBool(False)"], ["SGLANG_A = EnvBool(False)"])
            text = json.dumps(cat, default=str)
            self.assertNotIn(d, text)
        self.assertEqual(sorted(cat["trees"]["27b"]), ["entries", "rev"])

    def test_cli_wants_both_trees(self):
        with self.assertRaises(SystemExit):
            PC.main(["--tree-27b", "/nonexistent"])


REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
#: rename_rigdash.py lebt auf desk/rename-editor-1005 (docker/weg2-release/); RENAME_RIGDASH überschreibt den Pfad, das Kit liegt unter /spinning/flliper/tools
RENAME = os.environ.get("RENAME_RIGDASH") or os.path.join(REPO, "docker", "weg2-release", "rename_rigdash.py")
KIT = os.environ.get("RELEASE_KIT_TOOLS") or "/spinning/flliper/tools"


@unittest.skipUnless(os.path.isfile(RENAME) and os.path.isdir(KIT), "rename_rigdash.py oder das Release-Kit fehlt")
class RenameRoundtrip(unittest.TestCase):
    """Die Generator-Ausgabe muss durch die Editor-Umbenennung (rename_rigdash.py) laufen: kein Rest ``SGLANG_WEG2_``/``sglang/srt`` (Exit 3 sonst),
    gleiche Eintragszahl, ``baeume``/``abweichung`` bleiben."""

    def test_union_catalog_survives_the_rename(self):
        import json
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            a = _tree(d, "a", ["SGLANG_WEG2_FOO = EnvBool(False)  # foo", "SGLANG_WEG2_BOTH = EnvInt(1)"],
                      extra_files={"weg2/m.py": 'import os\nX_ENV = "SGLANG_WEG2_VIA_CONST"\nv = os.environ.get(X_ENV, "1")\n'})
            b = _tree(d, "b", ["SGLANG_WEG2_BOTH = EnvInt(2)", "SGLANG_WEG2_NF_ONLY = EnvBool(True)"])
            cat = PC.build_union_catalog([("27b", a), ("nf", b)], {}, edges_root=PY)
            pkg = os.path.join(d, "pkg")
            os.makedirs(os.path.join(pkg, "rigdash", "profil_data"))
            with open(os.path.join(pkg, "rigdash", "profil_data", "catalog.json"), "w", encoding="utf-8") as fh:
                json.dump(cat, fh, ensure_ascii=False)
            out = os.path.join(d, "out")
            res = subprocess.run([sys.executable, RENAME, pkg, "--out", out, "--kit", KIT], capture_output=True, text=True, timeout=240)
            self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
            with open(os.path.join(out, "rigdash", "profil_data", "catalog.json"), encoding="utf-8") as fh:
                renamed = json.load(fh)
        self.assertEqual(len(renamed["entries"]), len(cat["entries"]))
        self.assertFalse([k for k in renamed["entries"] if k.startswith("SGLANG_WEG2_")])
        both = next(e for k, e in renamed["entries"].items() if k.endswith("_BOTH"))
        self.assertEqual(both["baeume"], ["27b", "nf"])
        self.assertEqual(sorted(both["abweichung"]), ["27b", "nf"])


if __name__ == "__main__":
    unittest.main()
