"""F0-G (08.10.2026): profiles and container files of the renamed trees.

What is pinned here, without docker, GPU or network:

  * docker/flliper/profiles_release/*.env and profiles/nf-int4-h6-abl.env are the kit's conversion
    (tools/release/profconv.py, ``rename_to_flliper.rewrite_all``) of the old files next to them (``*.env.alt``; the old file with
    ONE kind of line changed: a sibling-source line names ``<sibling>.env.alt``, so a .alt run is a pure old-spelling run, see
    ``PC.alt_text`` / ``PC.from_alt``): converting the old file gives the new one, the conversion is idempotent, the new file carries no
    pre-rename name (SGLANG_*, --weg2-*, WEG2-*) and differs from the old one ONLY at names (line count equal, every
    line equal after both name families are folded to one token).  The five kit profiles are byte-equal to
    tools/release/profconv/.
  * NF finding of F0-D: the abliterated NF profile with the old names delivered no FLLIPER_MOE_RESIDENT_EXPERT_FRACTION to
    the renamed tree; the converted one does (the launcher input the planner oracle reads carries it in --env-p AND --env-d).
  * drift: the committed old files are what the live rig dirs held (when they are still unconverted) or what the live dirs hold
    converted; a live file that is neither is reported (skip with the name), as test_profile_snapshots_vs_live does for the planner snapshot.
  * Duo: ONE entrypoint (docker/pdflip-release/entrypoint.sh) takes every profile of both lines up to the model step:
    the profile family picks the code stand (27b* -> src-27b, nf* -> src-nf), the package is flliper, the subsystem pdflip.
  * Dockerfile.flliper: bake points, labels (source, release, revision), the legacy ENV set is exactly the five names the
    entrypoint reads, every RUN body parses; make_flat_ctx.sh / host_publish_flliper.sh bash syntax, ``--check`` usage, the
    pre-rename profile gate (B6); the publish gate's self-test (known red cases only).
"""

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

ROOT = pathlib.Path(__file__).resolve().parents[4]
D = ROOT / "docker" / "flliper"
REL = D / "profiles_release"
RIG = D / "profiles"
KIT = ROOT / "tools" / "release"
# The Duo image carries ONE entrypoint (make_flat_ctx.sh ships tools/entrypoint.sh); its tree master is docker/pdflip-release/entrypoint.sh of the
# 27B tree. The NF tree has none: there F0G_ENTRYPOINT=<27B tree>/docker/pdflip-release/entrypoint.sh runs the same script against the NF checkout.
EP = pathlib.Path(os.environ.get("F0G_ENTRYPOINT") or str(ROOT / "docker" / "pdflip-release" / "entrypoint.sh"))
DOCKERFILE = D / "Dockerfile.flliper"
MAKE = D / "make_flat_ctx.sh"
PUBLISH = D / "host_publish_flliper.sh"
SELFTEST = D / "host_publish_flliper_selftest.sh"

LIVE_REL = "/spinning/gpu-arb/docker/profiles_release"
LIVE_RIG = "/spinning/gpu-arb/docker/profiles"
KIT_FIVE = ("27b-base", "27b", "27b-nvfp4-dual", "nf", "nf-int4")
# the publish gate's self-test has seven red cases that are red in the live copy as well (mtime order of the stand-in verdict files)
KNOWN_RED = {
    "baseline, 27B id in verdict instead of facts",
    "L3 27B newest bound verdict FAIL",
    "L3 27B verdict OFFEN",
    "L3 27B verdict for another tree",
    "L3 27B verdict is not INT8 (NVFP4 heading)",
    "L3 NF verdict is NVFP4, not INT4",
    "L3 NF verdict without image id",
}


def _profconv():
    spec = importlib.util.spec_from_file_location("profconv_f0g", KIT / "profconv.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(KIT))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(KIT))
    return mod


PC = _profconv()
IMAP = PC.R._load_imap(str(KIT / "data" / "merged_0928.json"))
NAME = re.compile(r"WEG2|PDFLIP|Weg2|PdFlip|SGLANG|FLLIPER|weg2|pdflip|sglang|flliper")


def _read(p):
    return pathlib.Path(p).read_text(encoding="utf-8")


def _tree_profiles():
    return sorted(REL.glob("*.env")) + sorted(RIG.glob("*.env"))


def _old(p):
    """The old-spelling file of a tree profile: its .alt with the sibling-source lines mapped back (PC.from_alt) = the live file of 08.10."""
    return PC.from_alt(_read(str(p) + ".alt"))


class TestProfilesInTree(unittest.TestCase):
    def test_set_is_present(self):
        names = {p.name for p in _tree_profiles()}
        for need in ("27b.env", "27b-base.env", "nf.env", "nf-int4.env", "27b-nvfp4-dual.env", "nf-int4-h6-abl.env"):
            self.assertIn(need, names)
        for p in _tree_profiles():
            self.assertTrue(pathlib.Path(str(p) + ".alt").is_file(), "no .alt next to %s" % p.name)
        for j in ("27b-nvfp4.graphcal.json", "27b-nvfp4.pchunk.json", "27b.xcurves.json"):
            self.assertTrue((REL / j).is_file(), j)

    def test_new_is_the_conversion_of_alt_and_idempotent(self):
        for p in _tree_profiles():
            old, new = _old(p), _read(p)
            self.assertEqual(PC.convert(old, IMAP), new, p.name)
            self.assertEqual(PC.convert(new, IMAP), new, p.name + " (idempotent)")

    def test_no_pre_rename_name_left(self):
        for p in _tree_profiles():
            self.assertEqual(PC.old_name_lines(_read(p)), [], p.name)

    def test_alt_carries_old_names_where_the_new_carries_none(self):
        # at least the release profiles that the Duo image ships were old-named (the live files of 08.10.): the gate is not vacuous
        for n in ("27b-base", "27b", "nf", "nf-int4", "27b-nvfp4-dual"):
            self.assertGreater(len(PC.old_name_lines(_read(str(REL / (n + ".env.alt"))))), 0, n)

    def test_alt_vs_new_differ_only_at_names(self):
        for p in _tree_profiles():
            a, b = _old(p).splitlines(), _read(p).splitlines()
            self.assertEqual(len(a), len(b), p.name)
            for i, (x, y) in enumerate(zip(a, b), 1):
                # the HTSGLANG_* product env is not touched: fold it out first, then both name families to one token
                fx, fy = NAME.sub("N", x.replace("HTSGLANG_", "H_")), NAME.sub("N", y.replace("HTSGLANG_", "H_"))
                self.assertEqual(fx, fy, "%s:%d" % (p.name, i))

    def test_htsglang_product_env_is_unchanged(self):
        for p in _tree_profiles():
            old, new = _old(p), _read(p)
            self.assertEqual(re.findall(r"HTSGLANG_[A-Z0-9_]+", old), re.findall(r"HTSGLANG_[A-Z0-9_]+", new), p.name)

    def test_kit_five_equal_the_kit_dir(self):
        for n in KIT_FIVE + ("nf-int4-h6-abl",):
            here = REL / (n + ".env") if (REL / (n + ".env")).exists() else RIG / (n + ".env")
            self.assertEqual(_read(here), _read(KIT / "profconv" / (n + ".env")), n)

    def test_data_files_are_the_kit_copies_byte_for_byte(self):
        for j in ("27b-nvfp4.graphcal.json", "27b-nvfp4.pchunk.json"):
            self.assertEqual((REL / j).read_bytes(), (KIT / "profconv" / j).read_bytes(), j)

    def test_sourced_profiles_resolve_in_the_same_dir(self):
        for p in _tree_profiles():
            for dep in re.findall(r'^\s*source "\$\(dirname "\$\{BASH_SOURCE\[0\]\}"\)/([A-Za-z0-9._-]+\.env)"', _read(p), re.M):
                self.assertTrue((p.parent / dep).is_file(), "%s sources %s" % (p.name, dep))

    def test_alt_is_self_contained_and_maps_back_byte_for_byte(self):
        """Fix round 1 (finding 1): 10 of the 16 profiles source a sibling.  As .alt each must source the sibling's .alt (a pure old-spelling
        chain, not old loop + converted base); the only difference to the live file is that one line kind, and PC.from_alt gives it back."""
        n_sourcing = 0
        for p in _tree_profiles():
            alt = _read(str(p) + ".alt")
            deps = re.findall(r'^\s*source "\$\(dirname "\$\{BASH_SOURCE\[0\]\}"\)/([A-Za-z0-9._-]+)"', alt, re.M)
            for dep in deps:
                n_sourcing += 1
                self.assertTrue(dep.endswith(".env.alt"), "%s.alt sources %s" % (p.name, dep))
                self.assertTrue((p.parent / dep).is_file(), "%s.alt sources %s: missing" % (p.name, dep))
            self.assertEqual(PC.alt_text(PC.from_alt(alt)), alt, p.name)          # both directions are exact
            self.assertEqual(PC.from_alt(PC.alt_text(_read(p))), _read(p), p.name)   # a file without old sibling names would not move
        self.assertGreaterEqual(n_sourcing, 10)
        self.assertEqual(sum(1 for p in _tree_profiles() if PC.alt_text(_old(p)) != _old(p)), 10)

    def test_alt_chain_is_pure_old_and_new_chain_is_pure_new(self):
        """The chain a profile sources, executed (bash, `_form` stubbed like the dry-run does): the variables it sets carry NO new spelling for
        a .alt and NO old spelling for the converted file.  A mixed chain (the finding) would show both."""
        old_re, new_re = PC.OLD_NAME_RE, re.compile(r"(?<![A-Za-z])FLLIPER_|--pdflip-|PDFLIP-|flliper\.srt")
        checked = 0
        for p in _tree_profiles():
            for kind, f in (("alt", pathlib.Path(str(p) + ".alt")), ("new", p)):
                r = subprocess.run(["bash", "-c", '_form(){ :; }; set -eu; source "$1"; declare -p PROFILE_ARGS; declare -p | grep -E "^declare -[-a-zA-Z]+ [A-Z_0-9]+=" | grep -v -E " (BASH|SHLVL|PWD|_=|OLDPWD)"', "x", str(f)],
                                   capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent"}, timeout=60)
                self.assertEqual(r.returncode, 0, "%s: %s" % (f.name, r.stderr[-800:]))
                out = r.stdout
                self.assertIn("PROFILE_ARGS", out, f.name)
                if kind == "alt":
                    self.assertIsNone(new_re.search(out), "%s: new spelling in a .alt chain: %s" % (f.name, new_re.search(out)))
                else:
                    self.assertIsNone(old_re.search(out), "%s: old spelling in the converted chain: %s" % (f.name, old_re.search(out)))
                checked += 1
        self.assertEqual(checked, 2 * len(_tree_profiles()))

    def test_nf_abl_delivers_resident_expert_fraction(self):
        new, old = _read(RIG / "nf-int4-h6-abl.env"), _old(RIG / "nf-int4-h6-abl.env")
        self.assertIn("SGLANG_MOE_RESIDENT_EXPERT_FRACTION=", old)          # the F0-D finding: the old profile names the old spelling
        self.assertNotIn("SGLANG_MOE_RESIDENT_EXPERT_FRACTION", new)
        self.assertEqual(new.count("FLLIPER_MOE_RESIDENT_EXPERT_FRACTION="), 2)   # NF_ENV_P_FORM and NF_ENV_D_FORM

    def test_nf_abl_launcher_input_carries_the_renamed_variable(self):
        try:
            from flliper.srt.pdflip import propose_oracle as O
        except Exception as exc:  # pragma: no cover - no pdflip oracle in this build
            self.skipTest("planner oracle unavailable: %s" % exc)
        li = O.profile_launch_input(str(RIG / "nf-int4-h6-abl.env"))
        argv = list(li.argv)
        seen = {argv[i - 1]: t for i, t in enumerate(argv) if "RESIDENT_EXPERT_FRACTION" in t}
        self.assertEqual(sorted(seen), ["--env-d", "--env-p"])
        for t in seen.values():
            self.assertIn("FLLIPER_MOE_RESIDENT_EXPERT_FRACTION=", t)
            self.assertNotIn("SGLANG_MOE_RESIDENT_EXPERT_FRACTION", t)

    def test_profiles_vs_live(self):
        """REPORT, never silently follow: the live rig dirs are either still the committed .alt (unconverted) or the committed new
        file (converted by ``profconv.py --convert-live --apply``).  Anything else = the live profile moved: skip with the names."""
        moved = []
        for d, prof_dir in ((LIVE_REL, REL), (LIVE_RIG, RIG)):
            for p in sorted(prof_dir.glob("*.env")):
                live = pathlib.Path(d) / p.name
                if not live.is_file():
                    continue
                txt = _read(live)
                if txt not in (_old(p), _read(p)):
                    moved.append(str(live))
        if moved:
            self.skipTest("LIVE PROFILE MOVED under its committed snapshot (re-run profconv.py --tree-out): " + ", ".join(moved))

    def test_tree_out_check_is_clean_against_live(self):
        if not (os.path.isdir(LIVE_REL) and os.path.isdir(LIVE_RIG)):
            self.skipTest("live profile dirs not on this box")
        r = subprocess.run([sys.executable, str(KIT / "profconv.py"), "--tree-out", str(D), "--check"], capture_output=True, text=True)
        if r.returncode != 0:
            self.skipTest("LIVE PROFILE MOVED under the committed set:\n" + "\n".join(l for l in r.stdout.splitlines() if l.startswith("DIFFERS")))

    def test_convert_live_is_a_plan_without_apply_and_idempotent_with_it(self):
        with tempfile.TemporaryDirectory() as td:
            rel, rig = pathlib.Path(td, "rel"), pathlib.Path(td, "rig")
            rel.mkdir(); rig.mkdir()
            for p in sorted(REL.glob("27b*.env"))[:3]:
                (rel / p.name).write_text(_old(p), encoding="utf-8")
            (rig / "nf-int4-h6-abl.env").write_text(_old(RIG / "nf-int4-h6-abl.env"), encoding="utf-8")
            before = {p.name: p.read_bytes() for p in list(rel.iterdir()) + list(rig.iterdir())}
            cmd = [sys.executable, str(KIT / "profconv.py"), "--convert-live", "--src", str(rel), "--src2", str(rig)]
            plan = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(plan.returncode, 0, plan.stderr)
            self.assertIn("would convert", plan.stdout)
            self.assertEqual(before, {p.name: p.read_bytes() for p in list(rel.iterdir()) + list(rig.iterdir())})   # nothing written
            ap = subprocess.run(cmd + ["--apply"], capture_output=True, text=True)
            self.assertEqual(ap.returncode, 0, ap.stderr)
            for d in (rel, rig):
                for p in d.glob("*.env"):
                    self.assertEqual(p.read_text(), _read([x for x in _tree_profiles() if x.name == p.name][0]))   # = the committed new file
                    tree_p = [x for x in _tree_profiles() if x.name == p.name][0]
                    self.assertEqual((d / (p.name + ".alt")).read_text(), _read(str(tree_p) + ".alt"))               # = the committed .alt
                    self.assertEqual(PC.from_alt((d / (p.name + ".alt")).read_text()).encode(), before[p.name])      # old kept byte for byte
            again = subprocess.run(cmd + ["--apply"], capture_output=True, text=True)
            self.assertIn("0 to convert", again.stdout)
            # the way back: rollback-live restores the live files byte for byte (plan first, then --apply)
            rb = [sys.executable, str(KIT / "profconv.py"), "--rollback-live", "--src", str(rel), "--src2", str(rig)]
            conv = {p.name: p.read_bytes() for d in (rel, rig) for p in d.glob("*.env")}
            plan = subprocess.run(rb, capture_output=True, text=True)
            self.assertIn("would restore", plan.stdout)
            self.assertEqual(conv, {p.name: p.read_bytes() for d in (rel, rig) for p in d.glob("*.env")})
            done = subprocess.run(rb + ["--apply"], capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertEqual(before, {p.name: p.read_bytes() for d in (rel, rig) for p in d.glob("*.env")})
            self.assertIn("0 to restore", subprocess.run(rb + ["--apply"], capture_output=True, text=True).stdout)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestDryRunComparison(unittest.TestCase):
    """Fix round 1 (finding 2): the acceptance 'dry-run old vs new, 0 diff modulo names' has a record in the tree: one protocol per profile
    (docker/flliper/drycmp/protocols/<profile>.txt) and SUMMARY.txt, written by drycmp.py / summary.py from the logs of run_pair.sh."""

    PROT = D / "drycmp" / "protocols"

    @classmethod
    def setUpClass(cls):
        cls.dc = _load("drycmp_f0g", D / "drycmp" / "drycmp.py")
        cls.sm = _load("drycmp_summary_f0g", D / "drycmp" / "summary.py")

    def test_one_protocol_per_profile_all_ok(self):
        want = sorted(p.stem for p in _tree_profiles())
        got = sorted(p.stem for p in self.PROT.glob("*.txt") if p.stem != "SUMMARY")
        self.assertEqual(got, want)
        self.assertEqual(len(want), 16)
        for n in want:
            t = _read(self.PROT / (n + ".txt"))
            self.assertIn("VERDICT OK:", t, n)
            self.assertIn("PROFILE_ARGS folded: equal", t, n)
            self.assertIn("(pre-rename)   profile file %s.env.alt" % n, t, n)       # the OLD side is the pure .alt chain, not the converted sibling

    def test_summary_is_what_the_protocols_say_and_counts_are_exact(self):
        self.assertEqual(_read(self.PROT / "SUMMARY.txt"), self.sm.build(str(self.PROT)))
        t = _read(self.PROT / "SUMMARY.txt")
        # 13 stop at a refusal about model/draft files, 1 at the plan's W64, 2 complete: 13 + 1 + 2 = 16 (the commit message of F0-G had 12 + 4)
        self.assertIn("13 (config.json 7, W163 1, W128 5)", t)
        self.assertIn("(W64): 1. Plan complete (DRY-RUN complete, EXIT=0): 2.  13 + 1 + 2 = 16.", t)
        self.assertIn("VERDICT: 16 OK, 0 FAIL.", t)
        self.assertIn("LIMIT (stated, not a success)", t)

    def test_the_plan_reaching_profiles_have_equal_env_dicts(self):
        for n, groups in (("27b-nvfp4", {"D": 74, "P": 71}), ("27b-nvfp4-dual.diet", {"D": 68, "P": 68}), ("27b-nvfp4-dual", {"P": 80})):
            t = _read(self.PROT / (n + ".txt"))
            for g, k in groups.items():
                self.assertIn("ENV group %s: keys old=%d new=%d  only_old=[] only_new=[]  changed=0" % (g, k, k), t, n)

    def _mini(self, td, old_lines, new_lines):
        for side, lines in (("old", old_lines), ("new", new_lines)):
            pathlib.Path(td, "%s_p.log" % side).write_text("\n".join(lines) + "\n")
        ns = type("A", (), dict(old_tree="/o", new_tree="/n", work="/w", old_tree_sha="o", new_tree_sha="n"))()
        out = pathlib.Path(td, "out")
        out.mkdir()
        return self.dc.compare("p", td, ns, str(out))[0]

    def test_comparison_accepts_names_and_rejects_real_differences(self):
        args_old = "PROFILE_ARGS(p): --weg2-x 1 --foo 2"
        env_old = 'ENVDUMP P [["SGLANG_WEG2_A", "1"], ["SGLANG_B", "/o/x"]]'
        tail = ["WEG2-LAUNCH VRAM 100 MiB free", "EXIT=1"]
        base_new = ["PROFILE_ARGS(p): --pdflip-x 1 --foo 2", 'ENVDUMP P [["FLLIPER_PDFLIP_A", "1"], ["FLLIPER_B", "/n/x"]]',
                    "PDFLIP-LAUNCH VRAM 101 MiB free", "EXIT=1"]
        with tempfile.TemporaryDirectory() as td:
            self.assertTrue(self._mini(td, [args_old, env_old] + tail, base_new))            # names, a digit (reading) and the tree path differ: fine
        mutants = {
            "env value": [base_new[0], 'ENVDUMP P [["FLLIPER_PDFLIP_A", "2"], ["FLLIPER_B", "/n/x"]]'] + base_new[2:],
            "env key only on one side": [base_new[0], 'ENVDUMP P [["FLLIPER_PDFLIP_A", "1"], ["FLLIPER_B", "/n/x"], ["FLLIPER_C", "1"]]'] + base_new[2:],
            "flag": ["PROFILE_ARGS(p): --pdflip-x 1 --foo 3"] + base_new[1:],
            "unexplained line": base_new[:3] + ["PDFLIP-LAUNCH something new happened", "EXIT=1"],
        }
        for what, new in mutants.items():
            with tempfile.TemporaryDirectory() as td:
                self.assertFalse(self._mini(td, [args_old, env_old] + tail, new), what)


@unittest.skipUnless(EP.is_file() and shutil.which("bash"), "entrypoint not in this tree")
class TestDuoEntrypoint(unittest.TestCase):
    """ONE entrypoint, both lines: the real script (a copy) up to the model step, in a stand-in /opt/htsglang with src-27b and src-nf."""

    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.mkdtemp(prefix="f0g-ep-")
        o = pathlib.Path(cls.td, "opt")
        (o / "profiles").mkdir(parents=True)
        for stand in ("27b", "nf"):
            os.symlink(str(ROOT), str(o / ("src-" + stand)))
        for p in list(REL.glob("*.env")) + list(REL.glob("*.json")) + list(RIG.glob("*.env")):
            shutil.copy(p, o / "profiles" / p.name)
        src = EP.read_text()
        src = re.sub(r"^HOME_DIR=/opt/htsglang$", "HOME_DIR=%s" % o, src, flags=re.M).replace("/tmp/htsglang", cls.td + "/run")
        stop = 'echo "SANDBOX-STOP STAND=$STAND PKG=$PKG PDF=$PDF LINE=${PROFILE_LINE-}" >&2; exit 0\n'
        marker = "# --- 2. Modellformat"
        assert marker in src
        src = src.replace(marker, stop + marker, 1)
        cls.script = pathlib.Path(cls.td, "ep.sh")
        cls.script.write_text(src)
        pathlib.Path(cls.td, "home").mkdir()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.td, ignore_errors=True)

    def _run(self, profile):
        env = {"PATH": "/usr/bin:/bin", "HOME": self.td + "/home", "MODE": "pdflip", "HTSGLANG_EXPECT_GPUS": "0",
               "SGLANG_WEG2_TMS_OUT_DIR": self.td + "/tms", "FLLIPER_PROFILE": profile, "FLLIPER_ALLOW_EXPERIMENTAL": "1"}
        r = subprocess.run(["bash", str(self.script)], env=env, capture_output=True, text=True, timeout=120)
        return r.returncode, r.stdout + r.stderr

    def test_every_profile_of_both_lines_runs_to_the_model_step(self):
        names = sorted(p.stem for p in list(REL.glob("*.env")) + list(RIG.glob("*.env")))
        self.assertGreaterEqual(len(names), 16)
        for n in names:
            rc, out = self._run(n)
            want = "27b" if n.startswith("27b") else "nf"
            self.assertEqual(rc, 0, "%s:\n%s" % (n, out[-1500:]))
            self.assertIn("SANDBOX-STOP STAND=%s PKG=flliper PDF=pdflip LINE=%s" % (want, want), out, n)
            self.assertIn("Code-Stand %s:" % want, out, n)

    def test_unknown_family_is_refused_not_defaulted(self):
        rc, out = self._run("bogus")
        self.assertEqual(rc, 3)
        self.assertIn("REFUSED PROFILE", out)

    def test_the_old_profile_names_still_run(self):
        # the old-spelling chain in the same stand-in: <n>.env = the OLD file (PC.from_alt of the .alt) and its sibling profiles as the .alt
        # files they source (`27b-base.env.alt`): a pure old-spelling run, not an old file over the converted base.  The entrypoint is
        # agnostic about the spelling INSIDE a profile (the mirror folds it later).
        prof = pathlib.Path(self.td, "opt", "profiles")
        for a in REL.glob("*.env.alt"):
            shutil.copy(a, prof / a.name)
        for n in ("27b", "nf-int4", "27b-nvfp4"):
            (prof / (n + ".env")).write_text(_old(REL / (n + ".env")), encoding="utf-8")
            try:
                rc, out = self._run(n)
                self.assertEqual(rc, 0, out[-1500:])
                self.assertIn("SANDBOX-STOP STAND=%s" % ("27b" if n.startswith("27b") else "nf"), out)
            finally:
                shutil.copy(REL / (n + ".env"), prof / (n + ".env"))
        for a in REL.glob("*.env.alt"):
            (prof / a.name).unlink()


def _run_bodies(text):
    """The shell body of every RUN instruction (continuations joined, --mount options dropped)."""
    lines, bodies, cur = text.splitlines(), [], None
    for ln in lines:
        if cur is None:
            if ln.startswith("RUN "):
                cur = ln[4:]
                if not cur.rstrip().endswith("\\"):
                    bodies.append(cur)
                    cur = None
        else:
            if ln.lstrip().startswith("#"):
                continue
            cur += "\n" + ln
            if not ln.rstrip().endswith("\\"):
                bodies.append(cur)
                cur = None
    return [re.sub(r"^\s*(--mount=\S+\s+)+", "", b.replace("\\\n", "")) for b in bodies]


class TestDockerfile(unittest.TestCase):
    text = DOCKERFILE.read_text()

    def test_bake_points_and_labels(self):
        for ph in ("__FLLIPER_VERSION__", "__FLLIPER_SOURCE__", "__FLLIPER_PLACEHOLDERS__", "__FLLIPER_LOCK_SHA256__"):
            self.assertRegex(self.text, r"(?m)^ARG FLLIPER_\w+=%s$" % ph)
        self.assertIn("ARG ALLOW_DUO_TREES=0", self.text)
        for lab in ('org.opencontainers.image.title="fLLiper"', 'io.github.efschu.flliper.revision="${REV_27B}"',
                    'io.github.efschu.flliper.revision.27b="${REV_27B}"', 'io.github.efschu.flliper.revision.nf="${REV_NF}"',
                    'io.github.efschu.flliper.slots="27b nf"', 'io.github.efschu.flliper.build="flat"'):
            self.assertIn(lab, self.text)
        self.assertNotRegex(self.text, r"(?m)^\s*(LABEL\s+)?htsglang\.")        # no leftover htsglang.* label
        self.assertIn("ENV FLLIPER_VERSION=${FLLIPER_VERSION}", self.text)

    def test_legacy_env_set_is_exactly_what_the_entrypoint_reads(self):
        baked = set(re.findall(r"(?m)^(?:ENV\s+|\s{4})(SGLANG_[A-Z0-9_]+)=", self.text))
        self.assertEqual(baked, {"SGLANG_WEG2_VENV", "SGLANG_WEG2_DEVTOOLS_DIR", "SGLANG_WEG2_EVIDENCE_DIR", "SGLANG_WEG2_GPU_ARB",
                                 "SGLANG_WEG2_STORE_ROOT"})
        if EP.is_file():
            ep = EP.read_text()
            for n in baked:
                self.assertIn(n, ep, n)
        for n in ("FLLIPER_BARLINK_BAR1_NV_SOURCE", "FLLIPER_BARLINK_LAUNCH_DUMP", "FLLIPER_PLANNER_PROFILES", "FLLIPER_PLANNER_GRAPH_ANCHORS",
                  "FLLIPER_REVISION_27B", "FLLIPER_REVISION_NF", "FLLIPER_LINES"):
            self.assertRegex(self.text, r"(?m)(?:ENV\s+|\s{4})%s=" % n)
        # the state-dir defaults stay the entrypoint's "no choice of the caller" values
        self.assertIn("SGLANG_WEG2_EVIDENCE_DIR=/var/lib/htsglang/evidence", self.text)

    def test_workspace_link_for_both_names(self):
        self.assertIn("/sgl-workspace/sglang", self.text)
        self.assertIn("/sgl-workspace/flliper", self.text)

    def test_run_bodies_parse(self):
        sh = shutil.which("dash") or shutil.which("sh")
        bodies = _run_bodies(self.text)
        self.assertGreater(len(bodies), 15)
        for b in bodies:
            r = subprocess.run([sh, "-n", "-c", b], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr + "\n" + b[:300])


class TestScripts(unittest.TestCase):
    def test_bash_syntax(self):
        for f in (MAKE, PUBLISH, SELFTEST):
            r = subprocess.run(["bash", "-n", str(f)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, "%s: %s" % (f.name, r.stderr))

    def test_check_is_offline_and_listed(self):
        self.assertIn("--check", MAKE.read_text().split("set -uo pipefail")[0])
        r = subprocess.run(["bash", str(MAKE), "--check", "--host", "--rev", "HEAD"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn("offline", r.stderr)
        r = subprocess.run(["bash", str(MAKE), "--check", "--write", "--rev", "HEAD"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)       # exactly one of --plan/--write/--check

    def test_profile_name_gate_in_the_script(self):
        txt = MAKE.read_text()
        self.assertIn("B6: profile", txt)
        self.assertIn("--profiles-from-tree", txt)
        self.assertIn("PROFILE_RIG", txt)

    @unittest.skipUnless(os.path.isfile("/spinning/gpu-arb/docker/overrides_fi070.txt") and (ROOT / ".git").exists(),
                         "rig inputs (pins, wheels) or git checkout absent")
    def test_b6_gate_blocks_old_named_profiles_and_passes_converted_ones(self):
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if not re.fullmatch(r"[0-9a-f]{40}", head):
            self.skipTest("no HEAD")
        with tempfile.TemporaryDirectory() as td:
            for kind in ("new", "old"):
                base = pathlib.Path(td, kind, "docker", "flliper")
                (base / "profiles_release").mkdir(parents=True)
                (base / "profiles").mkdir()
                for p in REL.glob("*"):
                    if p.suffix in (".env", ".json"):
                        src = pathlib.Path(str(p) + ".alt") if (kind == "old" and p.suffix == ".env") else p
                        shutil.copy(src, base / "profiles_release" / p.name)
                for p in RIG.glob("*.env"):
                    shutil.copy(str(p) + ".alt" if kind == "old" else p, base / "profiles" / p.name)
            outs = {}
            for kind in ("new", "old"):
                r = subprocess.run(["bash", str(MAKE), "--plan", "--allow-unpushed", "--rev", head, "--profiles-from-tree", str(pathlib.Path(td, kind))],
                                   capture_output=True, text=True, timeout=600)
                outs[kind] = r.stdout
            if "renamed layout: python/flliper" not in outs["new"]:
                self.skipTest("HEAD is not a renamed tree for the script's own checks")
            self.assertNotIn("B6: profile", outs["new"])
            self.assertIn("B6: profile 27b-base", outs["old"])
            self.assertIn("pre-rename name lines=0", outs["new"])

    @unittest.skipUnless(shutil.which("git"), "git absent")
    def test_publish_gate_selftest_has_only_the_known_red_cases(self):
        r = subprocess.run(["bash", str(SELFTEST)], capture_output=True, text=True, timeout=900)
        out = r.stdout
        m = re.search(r"== (\d+) passed, (\d+) failed", out)
        self.assertIsNotNone(m, out[-800:])
        # the case names contain commas: take everything after the "N passed, M failed" line, drop the "failed:" markers and the known names;
        # nothing but separators may remain
        remaining = out[m.end():].replace("failed:", "")
        for k in sorted(KNOWN_RED, key=len, reverse=True):
            remaining = remaining.replace(k, "")
        self.assertEqual(remaining.replace(",", "").strip(), "", "unexpected red self-test case(s): " + out[-600:])
        self.assertGreaterEqual(int(m.group(1)), 49)
        self.assertLessEqual(int(m.group(2)), len(KNOWN_RED))


if __name__ == "__main__":
    unittest.main()
