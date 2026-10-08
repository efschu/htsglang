# SPDX-License-Identifier: Apache-2.0
"""F0-C: the compatibility layer survives the mechanical rename, and a profile with the OLD names plans like one with the NEW names.

PLAN-RENAME-FLLIPER-1007 F0-C, RENAME_PLAN 4.  Three proofs, all GPU-free:

1. ``TestRenamedCopy``: the shim files plus the census module are copied into a throwaway git repo, ``rename_to_flliper.py apply
   --weg2`` runs on that COPY, and the same self-check script (itself renamed by the tool) runs against the copy BEFORE and AFTER:
   the env mirror (legacy -> canonical, the renamed value wins, foreign readers keep both spellings), the announcement (one conflict
   warning, one deprecation line after the rename and none before it), the flag aliases of a parser, the census of both
   generations.  Skips where the rename tool is not on the box.
2. ``test_compat_layer_survives_rename`` (the sibling of ``test_name_compat_survives_rename``): the real tool leaves the shim files
   and ``_compat_boot.py`` byte-identical (the layer must not be rewritten into new <-> new).
3. ``TestDryRunOldNamesEqualNewNames``: the launcher dry run of a release profile with the OLD env/flag names and with the NEW
   names (through the one bridge the package import runs) gives the same plan modulo the name tokens.

Every name is built from the shim's own tokens: the tests mean the same before and after the mechanical rename.
"""

import copy
import importlib.util
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import pytest

from sglang import _compat_boot as boot
from sglang.srt import compat_shims as cs
from sglang.srt import name_compat as nc
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

TOOL = "/spinning/flliper/tools/rename_to_flliper.py"
SUB, GEN = nc.ENV_PREFIX_PAIRS[0], nc.ENV_PREFIX_PAIRS[-1]
TOK_OLD, TOK_NEW = cs.FLAG_TOKENS
PKG_OLD, PKG_NEW = cs._PKG_OLD, cs._PKG_NEW
HERE = os.path.dirname(os.path.abspath(__file__))
SRT = os.path.dirname(os.path.abspath(cs.__file__))
PKG_DIR = os.path.dirname(SRT)


def _tool():
    if not os.path.exists(TOOL):
        return None
    spec = importlib.util.spec_from_file_location("_rename_to_flliper_f0c", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# 2. fixed point of the real tool
# --------------------------------------------------------------------------


def test_compat_layer_survives_rename():
    """The mechanical rename (all rule sets on) leaves the whole layer byte-identical: the env mirror and its announcement
    (``name_compat``, ``_compat_boot``), the flag/route/cache/census shims (``compat_shims``)."""
    tool = _tool()
    if tool is None:
        pytest.skip("rename tool not on this box")
    for path in (nc.__file__, cs.__file__, boot.__file__):
        src = open(path).read()
        out, rep, _skip = tool.rewrite_all(src, True, True, {})
        assert out == src, (path, rep)


# --------------------------------------------------------------------------
# 1. the throwaway copy, renamed by the real tool
# --------------------------------------------------------------------------

#: runs INSIDE the copy (before and after the rename); written without a legacy name in one piece, so the tool leaves it alone
#: except for what it must rewrite (the import below names the running package from argv)
SELF_CHECK = r'''
import argparse, importlib, json, logging, os, sys

logging.basicConfig(stream=sys.stderr, format="%(message)s", level=logging.WARNING)   # BEFORE the import: the bridge logs at import
pkg, sub, other_sub = sys.argv[1], sys.argv[2], sys.argv[3]
importlib.import_module(pkg)                       # the package import runs the env bridge
cs = importlib.import_module(pkg + ".srt.compat_shims")
nc = importlib.import_module(pkg + ".srt.name_compat")
hc = importlib.import_module(pkg + ".srt." + sub + ".host_census")

ap = argparse.ArgumentParser()
ap.add_argument("--" + sub + "-vision", choices=("off", "transient"), default="off")
ap.add_argument("--" + sub + "-d-adopt", action="store_true")
n_aliases = cs.register_flag_aliases(ap)
a = ap.parse_args(["--" + sub + "-vision", "transient", "--" + sub + "-d-adopt"])
b = ap.parse_args(["--" + other_sub + "-vision=transient", "--" + other_sub + "-d-adopt"])

census = {}
for p, s in ((pkg, sub), (cs.other_package(), other_sub)):
    census[p] = [hc.classify_process(p + "::scheduler_PP0", ""), hc.classify_process("python", "python -m " + p + ".srt." + s + ".front"),
                 hc.classify_process("python", "python -m " + p + ".launch_server")]
keep = [k for k in os.environ if k.startswith(("SG" + "LANG_", "FLLIPER_", "SGL_"))]
print(json.dumps({"running": cs.running_package(), "canonical_side": nc.CANONICAL_SIDE,
                  "environ": {k: os.environ[k] for k in sorted(keep)}, "aliases": n_aliases,
                  "flags_equal": vars(a) == vars(b), "flag_value": sorted(vars(a).values(), key=str), "census": census}))
'''


def _mini_repo(root: pathlib.Path) -> None:
    (root / "python" / "sg" "lang" / "srt" / TOK_OLD).mkdir(parents=True)
    (root / "tools").mkdir()
    pkg = root / "python" / ("sg" "lang")
    shutil.copy(boot.__file__, pkg / "_compat_boot.py")
    for name in ("name_compat.py", "compat_shims.py"):
        shutil.copy(os.path.join(SRT, name), pkg / "srt" / name)
    shutil.copy(os.path.join(SRT, TOK_OLD, "host_census.py") if os.path.isdir(os.path.join(SRT, TOK_OLD))
                else os.path.join(SRT, TOK_NEW, "host_census.py"), pkg / "srt" / TOK_OLD / "host_census.py")
    (pkg / "srt" / "__init__.py").write_text("")
    (pkg / "srt" / TOK_OLD / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text(
        "from ._compat_boot import bridge_environ as _bridge_environ\n\n_bridge_environ()\ndel _bridge_environ\n")
    (root / "tools" / "selfcheck.py").write_text(SELF_CHECK)


def _git(root, *args):
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args], check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _names():
    return {"legacy_only_gen": "MIRROR_A", "legacy_only_sub": "MIRROR_B", "conflict": "MIRROR_C", "renamed_only": "MIRROR_D"}


def _run_selfcheck(root: pathlib.Path, pkg: str, sub: str, other_sub: str):
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(root / "python"), "HOME": str(root),
           "PYTHONDONTWRITEBYTECODE": "1",
           GEN[0] + "MIRROR_A": "a", SUB[0] + "MIRROR_B": "b",
           GEN[0] + "MIRROR_C": "old", GEN[1] + "MIRROR_C": "new", GEN[1] + "MIRROR_D": "d",
           GEN[0] + "RPF_N": "8", "SGL_ALIAS": "keep"}
    cp = subprocess.run([sys.executable, str(root / "tools" / "selfcheck.py"), pkg, sub, other_sub], env=env,
                        capture_output=True, text=True, timeout=120)
    assert cp.returncode == 0, cp.stderr[-2000:]
    return json.loads(cp.stdout.strip().splitlines()[-1]), cp.stderr


class TestRenamedCopy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(TOOL):
            raise unittest.SkipTest("rename tool not on this box")
        cls.tmp = pathlib.Path(tempfile.mkdtemp(prefix="f0c-mini-"))
        cls.root = cls.tmp / "copy"
        cls.root.mkdir()
        _mini_repo(cls.root)
        _git(cls.root, "init", "-q", ".")
        _git(cls.root, "add", "-A")
        _git(cls.root, "commit", "-qm", "base")
        cls.before = _run_selfcheck(cls.root, PKG_OLD, TOK_OLD, TOK_NEW)
        cp = subprocess.run([sys.executable, TOOL, "apply", "--root", str(cls.root), "--weg2", "--no-stage"],
                            capture_output=True, text=True, timeout=300)
        assert cp.returncode == 0, cp.stderr[-2000:] + cp.stdout[-2000:]
        cls.after = _run_selfcheck(cls.root, PKG_NEW, TOK_NEW, TOK_OLD)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_the_tool_really_renamed_the_copy(self):
        self.assertTrue((self.root / "python" / PKG_NEW / "srt" / TOK_NEW / "host_census.py").is_file())
        self.assertEqual(list((self.root / "python" / PKG_OLD).rglob("*.py")) if (self.root / "python" / PKG_OLD).exists() else [], [])
        # ... and left the shim files byte-identical
        for name in ("name_compat.py", "compat_shims.py"):
            self.assertEqual((self.root / "python" / PKG_NEW / "srt" / name).read_text(), open(os.path.join(SRT, name)).read())

    def test_before_the_rename_the_old_names_are_the_canonical_ones(self):
        out, err = self.before
        self.assertEqual((out["running"], out["canonical_side"]), (PKG_OLD, 0))
        self.assertEqual(out["environ"], {
            GEN[0] + "MIRROR_A": "a", SUB[0] + "MIRROR_B": "b", GEN[0] + "MIRROR_C": "new", GEN[0] + "MIRROR_D": "d",
            GEN[0] + "RPF_N": "8", "SGL_ALIAS": "keep"})
        self.assertNotIn("DEPRECATED", err)
        self.assertEqual(err.count("both set"), 1)
        self.assertIn(GEN[0] + "MIRROR_C", err)

    def test_after_the_rename_the_old_profile_reaches_the_new_names(self):
        out, err = self.after
        self.assertEqual((out["running"], out["canonical_side"]), (PKG_NEW, 1))
        self.assertEqual(out["environ"], {
            GEN[1] + "MIRROR_A": "a", SUB[1] + "MIRROR_B": "b", GEN[1] + "MIRROR_C": "new", GEN[1] + "MIRROR_D": "d",
            GEN[1] + "RPF_N": "8",     # a foreign reader (sgl-kernel) keeps the legacy spelling beside the new one
            GEN[0] + "RPF_N": "8", "SGL_ALIAS": "keep"})

    def test_after_the_rename_one_conflict_line_and_one_deprecation_line(self):
        _out, err = self.after
        self.assertEqual(err.count("both set"), 1)
        self.assertEqual(err.count("DEPRECATED"), 1)
        dep = [ln for ln in err.split("\n") if "DEPRECATED" in ln][0]
        for n in (GEN[0] + "MIRROR_A", SUB[0] + "MIRROR_B", GEN[0] + "MIRROR_C", GEN[0] + "RPF_N"):
            self.assertIn(n, dep)
        self.assertNotIn(GEN[1] + "MIRROR_D", dep)       # a new name is not deprecated
        self.assertNotIn("SGL_ALIAS", err)

    def test_flags_parse_in_both_spellings_on_both_sides(self):
        for out, _err in (self.before, self.after):
            self.assertEqual(out["aliases"], 2)
            self.assertTrue(out["flags_equal"])
            self.assertEqual(sorted(out["flag_value"], key=str), sorted([True, "transient"], key=str))

    def test_the_census_sees_both_generations_on_both_sides(self):
        for out, _err in (self.before, self.after):
            self.assertEqual(len(out["census"]), 2)
            for role in out["census"].values():
                self.assertEqual(role, ["rank", "front", "server_main"])


# --------------------------------------------------------------------------
# 3. dry run: old names == new names (modulo the name tokens)
# --------------------------------------------------------------------------

try:
    from sglang.srt.weg2 import launcher
    from sglang.srt.weg2 import propose_oracle as O
except Exception as exc:  # pragma: no cover - no weg2 launcher in this build
    launcher = O = None
    _IMPORT_ERR = str(exc)

FIX = os.path.join(HERE, "fixtures", "planer_1006")
REPLAY_REF = os.path.join(HERE, "fixtures", "xchg_launch_replay_0911", "nvml_devices_1378.json")
_CENSUS_27B = "/spinning/gpu-arb/weg2/census/xchg_census_weg2xsn246_27198a2711.json"
_NF_NAMES = ("Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist-abl-wxp", "Qwen3.8-Flash-Next-MTP-INT4-g32-albucino-abl-wxp")


def _snapshots():
    root = os.path.join(FIX, "checkpoints")
    out = {}
    if os.path.isdir(root):
        for n in sorted(os.listdir(root)):
            if os.path.isfile(os.path.join(root, n, "manifest.json")):
                out[O.read_snapshot_manifest(os.path.join(root, n))["name"]] = os.path.join(root, n)
    return out


def _spell(text: str, side: int) -> str:
    """``text`` with the env prefixes and flip flags of the other spelling written as ``side``'s (0 legacy, 1 renamed)."""
    for pair in nc.ENV_PREFIX_PAIRS:                       # the more specific family first
        text = text.replace(pair[1 - side], pair[side])
    return text.replace("--%s-" % (TOK_OLD, TOK_NEW)[1 - side], "--%s-" % (TOK_OLD, TOK_NEW)[side])


def _spelled_input(li, side: int):
    """A copy of the LaunchInput ``li`` with every env name and flip flag written in the spelling of ``side``."""
    out = copy.copy(li)
    env = {}
    for k, v in li.env.items():
        for pair in nc.ENV_PREFIX_PAIRS:
            if k.startswith(pair[1 - side]):
                k = pair[side] + k[len(pair[1 - side]):]
                break
        env[k] = v
    other, run = (TOK_OLD, TOK_NEW)[1 - side], (TOK_OLD, TOK_NEW)[side]
    out.env = env
    out.argv = [("--%s-%s" % (run, a[len(other) + 3:])) if isinstance(a, str) and a.startswith("--%s-" % other) else a
                for a in li.argv]
    return out


@unittest.skipIf(O is None, "weg2 launcher unavailable")
class TestDryRunOldNamesEqualNewNames(unittest.TestCase):
    def setUp(self):
        if O.launcher_line() == O.LINE_27B:
            self.profile, ok = "27b-base", os.path.exists(_CENSUS_27B)
        else:
            snaps = _snapshots()
            self.profile, ok = "nf-int4-h6-abl", all(n in snaps for n in _NF_NAMES)
        if not ok:
            self.skipTest("the reference profile's census / header snapshots are not on this box")
        self.path = os.path.join(FIX, "profiles", self.profile + ".env")
        self.li = O.profile_launch_input(self.path)
        if self.li.unresolved_paths:
            self.skipTest("rig asset dirs not on this box: %s" % self.li.unresolved_paths[:2])

    def _run(self, side: int):
        li = _spelled_input(self.li, side)
        # what the package import does to a process environment before anything reads it
        li.env = nc.canonical_env(dict(li.env))
        tree = str(pathlib.Path(launcher.__file__).resolve().parents[4])
        res = O.run_profile(self.path, O.read_replay(REPLAY_REF), tree=tree, snapshots=_snapshots() or None, launch_input=li)
        return res, li

    def test_the_two_spellings_really_differ_before_the_bridge(self):
        a, b = _spelled_input(self.li, 0), _spelled_input(self.li, 1)
        self.assertNotEqual(sorted(a.env), sorted(b.env))
        self.assertNotEqual(a.argv, b.argv)
        self.assertGreater(len([k for k in b.env if nc.canonical_env_name(k) != k]), 5)
        self.assertTrue(any(isinstance(t, str) and t.startswith("--%s-" % TOK_NEW) for t in b.argv) or nc.CANONICAL_SIDE == 1)

    def test_the_bridge_folds_the_new_spelling_onto_the_old_profile_environment(self):
        _r0, li0 = self._run(0)
        _r1, li1 = self._run(1)
        self.assertEqual(li1.env, li0.env)

    def test_the_launcher_parser_reads_the_profile_argv_in_both_spellings(self):
        base = ["--tree", "/t", "--tag", "x"]
        a = launcher.build_parser().parse_args(base + _spelled_input(self.li, 0).argv)
        b = launcher.build_parser().parse_args(base + _spelled_input(self.li, 1).argv)
        self.assertEqual(vars(a), vars(b))

    def test_dry_run_with_old_names_equals_dry_run_with_new_names(self):
        run0, _l0 = self._run(0)
        run1, _l1 = self._run(1)
        res0, res1 = run0.result, run1.result
        self.assertEqual((res1.rc, res1.exc_type), (res0.rc, res0.exc_type), res1.exc_msg)
        self.assertEqual(res1.forced, res0.forced)
        self.assertGreater(len(res0.text.splitlines()), 100, "the dry run produced no plan: the comparison would prove nothing")
        # modulo the name tokens: spell both runs' text the same way, then require 0 diff lines
        self.assertEqual(O.diff_lines(_spell(res0.text, 0), _spell(res1.text, 0)), [])


if __name__ == "__main__":
    unittest.main()
