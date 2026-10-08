"""F0-C compatibility layer (PLAN-RENAME-FLLIPER-1007 F0-C, RENAME_PLAN 4): what a user profile with the OLD names needs.

* env mirror: both spellings -> one, the RENAMED one wins, ONE warning per conflicting pair, ONE deprecation line per process;
* flag aliases: every flip flag of the launcher parser parses under both spellings, same ``dest`` (registration only);
* rig state READ across the rename: ``card_library.json``, its ``.by-uuid.json`` side file, ``hw_profile-*.json``,
  ``card_probe-*.json`` are read from the other name's cache dir when missing under the running name; WRITES stay;
* the host census counts the processes (titles, module names) of either package generation.

Every name here is built from the shim's own tokens, so the tests mean the same before and after the mechanical rename.
"""

import ast
import io
import json
import logging
import os
import tempfile
import unittest
from unittest import mock

from flliper import _compat_boot as boot
from flliper.srt import compat_shims as cs
from flliper.srt import name_compat as nc
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

SUB, GEN = nc.ENV_PREFIX_PAIRS[0], nc.ENV_PREFIX_PAIRS[-1]
C = nc.CANONICAL_SIDE
PKG_OLD, PKG_NEW = cs._PKG_OLD, cs._PKG_NEW


def _canon(pair, rest):
    return pair[C] + rest


def _other(pair, rest):
    return pair[1 - C] + rest


class _Quiet:
    """Collect what ``_compat_boot`` logs, from a clean 'announced' state."""

    def __enter__(self):
        boot.reset_announcements()
        self.buf = io.StringIO()
        self.h = logging.StreamHandler(self.buf)
        boot._logger.addHandler(self.h)
        self._lvl = boot._logger.level
        boot._logger.setLevel(logging.WARNING)
        return self

    def __exit__(self, *exc):
        boot._logger.removeHandler(self.h)
        boot._logger.setLevel(self._lvl)
        boot.reset_announcements()

    @property
    def lines(self):
        return [ln for ln in self.buf.getvalue().split("\n") if ln]


class TestEnvMirror(CustomTestCase):
    def test_the_renamed_value_wins_on_either_side_of_the_rename(self):
        for rest, pair in (("GROUP", SUB), ("HICACHE_X", GEN)):
            env = {pair[0] + rest: "old", pair[1] + rest: "new"}
            nc.canonical_env(env)
            self.assertEqual(env, {_canon(pair, rest): "new"})

    def test_one_side_only_moves_onto_the_canonical_name(self):
        env = {_other(SUB, "GROUP"): "P", _other(GEN, "HICACHE_X"): "1"}
        nc.canonical_env(env)
        self.assertEqual(env, {_canon(SUB, "GROUP"): "P", _canon(GEN, "HICACHE_X"): "1"})

    def test_the_report_names_conflicts_and_legacy_use(self):
        env = {GEN[0] + "A": "1", GEN[1] + "A": "2",          # conflict
               SUB[0] + "B": "x",                              # legacy only
               GEN[1] + "C": "z",                              # renamed only
               GEN[0] + "SAME": "q", GEN[1] + "SAME": "q",     # both, equal: no conflict
               "SGL_ALIAS": "keep", "HT" "SG" "LANG_PRODUCT": "keep"}
        rep = {}
        nc.canonical_env(env, report=rep)
        self.assertEqual(rep["conflicts"], [(GEN[0] + "A", GEN[1] + "A")])
        legacy = sorted(rep.get("legacy", []))
        if C == 1:    # after the rename the legacy names are read through the mirror
            self.assertEqual(legacy, sorted([GEN[0] + "A", SUB[0] + "B", GEN[0] + "SAME"]))
        else:         # before it they ARE the canonical names: nothing deprecated
            self.assertEqual(legacy, [])
        # untouched: upstream alias prefix and the product prefix belong to neither family
        self.assertEqual(env["SGL_ALIAS"], "keep")
        self.assertEqual(env["HT" "SG" "LANG_PRODUCT"], "keep")

    def test_a_foreign_reader_beside_its_equal_renamed_twin_is_not_a_use(self):
        """What a parent of the other generation hands down: both spellings, same value."""
        rest = "RPF_N"
        leg = GEN[0] + rest
        self.assertIn(leg, nc.FOREIGN_READERS)
        rep = {}
        nc.canonical_env({GEN[0] + rest: "8", GEN[1] + rest: "8"}, report=rep)
        self.assertEqual(rep.get("legacy", []), [])
        rep = {}
        nc.canonical_env({GEN[0] + rest: "8"}, report=rep)      # the user set only the old one
        self.assertEqual(rep.get("legacy", []), [] if C == 0 else [leg])

    def test_the_launchers_write_of_a_foreign_name_beats_the_twin_of_the_parent(self):
        """Review F0-C 1.1: a parent resolved a foreign name to both spellings; the launcher then writes the canonical one
        (the rename tool rewrites that literal along) into the child's env, and ``canonical_env`` must keep ITS value."""
        for leg in sorted(nc.FOREIGN_READERS):
            fam = nc._env_family(leg)
            self.assertIsNotNone(fam)
            _leg, new, canon = fam
            parent = {new: "0"}
            nc.canonical_env(parent)
            self.assertEqual(parent[_leg], "0")
            child = dict(parent)
            child[canon] = "1"                       # what the launcher writes
            rep = {}
            nc.canonical_env(child, report=rep)
            self.assertEqual(child[canon], "1", leg)
            self.assertEqual(child[_leg], "1", leg)  # what the foreign code reads
            self.assertEqual(child.get(new), "1", leg)
            self.assertNotIn((_leg, new), rep.get("conflicts", []), leg)

    def test_a_non_foreign_conflict_still_lets_the_renamed_spelling_win(self):
        rest = "ZZ_NO_FOREIGN"
        rep = {}
        env = {GEN[0] + rest: "old", GEN[1] + rest: "new"}
        nc.canonical_env(env, report=rep)
        self.assertEqual(env, {_canon(GEN, rest): "new"})
        self.assertEqual(rep["conflicts"], [(GEN[0] + rest, GEN[1] + rest)])

    def test_the_env_without_a_report_is_unchanged_behaviour(self):
        env = {_canon(GEN, "A"): "1", "PATH": "/bin"}
        self.assertIs(nc.canonical_env(env), env)
        self.assertEqual(env, {_canon(GEN, "A"): "1", "PATH": "/bin"})


class TestAnnouncement(CustomTestCase):
    def test_conflict_warning_names_both_variables_never_the_values(self):
        with _Quiet() as q:
            env = {GEN[0] + "SECRETISH": "hunter2", GEN[1] + "SECRETISH": "hunter3"}
            boot.bridge_environ(env)
        self.assertEqual(len([ln for ln in q.lines if "both set" in ln]), 1)
        text = "\n".join(q.lines)
        self.assertIn(GEN[0] + "SECRETISH", text)
        self.assertIn(GEN[1] + "SECRETISH", text)
        self.assertNotIn("hunter", text)
        self.assertEqual(env, {_canon(GEN, "SECRETISH"): "hunter3"})

    def test_deprecation_line_once_per_process_and_only_after_the_rename(self):
        with _Quiet() as q:
            boot.bridge_environ({GEN[0] + "ONE": "1", SUB[0] + "TWO": "2"})
            boot.bridge_environ({GEN[0] + "THREE": "3"})
        dep = [ln for ln in q.lines if "DEPRECATED" in ln]
        if C == 1:
            self.assertEqual(len(dep), 1)
            self.assertIn(GEN[0] + "ONE", dep[0])
            self.assertIn(SUB[0] + "TWO", dep[0])
            self.assertNotIn(GEN[0] + "THREE", dep[0])    # the second call adds no second line
        else:
            self.assertEqual(dep, [])

    def test_a_conflict_is_announced_once_per_pair(self):
        with _Quiet() as q:
            for _ in range(3):
                boot.bridge_environ({GEN[0] + "K": "1", GEN[1] + "K": "2"})
        self.assertEqual(len([ln for ln in q.lines if "both set" in ln]), 1)

    def test_a_child_of_a_folded_environment_is_silent(self):
        with _Quiet() as q:
            env = {GEN[0] + "ONE": "1", GEN[0] + "RPF_N": "8"}
            boot.bridge_environ(env)
            boot.reset_announcements()
            first = len(q.lines)
            boot.bridge_environ(env)            # what the child process sees: the folded env
        self.assertEqual(len(q.lines), first)

    def test_long_lists_are_cut(self):
        names = ["%sN%02d" % (GEN[0], i) for i in range(20)]
        with _Quiet() as q:
            out = boot.announce({"legacy": names})
        self.assertEqual(len(out), 1)
        self.assertIn("(+12 more)", out[0])

    def test_nothing_to_say_says_nothing(self):
        with _Quiet() as q:
            boot.bridge_environ({_canon(GEN, "A"): "1"})
            boot.announce({})
            boot.announce(None)
        self.assertEqual(q.lines, [])

    def test_the_package_init_runs_the_bridge_first(self):
        """An import must fold the environment before ANY other statement of the package runs."""
        import flliper

        tree = ast.parse(open(flliper.__file__, encoding="utf-8").read())
        body = [n for n in tree.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
        self.assertIsInstance(body[0], ast.ImportFrom)
        self.assertEqual(body[0].module, "_compat_boot")
        self.assertEqual(body[1].value.func.id, "_bridge_environ")


class TestFlagAliasRegistration(CustomTestCase):
    def _parser(self):
        import argparse

        run = cs.FLAG_TOKENS[0] if cs.running_package() == PKG_OLD else cs.FLAG_TOKENS[1]
        ap = argparse.ArgumentParser()
        ap.add_argument("--tag")
        ap.add_argument("--%s-vision" % run, choices=("off", "transient"), default="off", help="vision")
        ap.add_argument("--%s-d-adopt" % run, action="store_true")
        return ap, run

    def test_both_spellings_parse_to_the_same_dest(self):
        ap, run = self._parser()
        other = cs.FLAG_TOKENS[1] if run == cs.FLAG_TOKENS[0] else cs.FLAG_TOKENS[0]
        self.assertEqual(cs.register_flag_aliases(ap), 2)
        a = ap.parse_args(["--%s-vision" % run, "transient", "--%s-d-adopt" % run])
        b = ap.parse_args(["--%s-vision" % other, "transient", "--%s-d-adopt" % other])
        c = ap.parse_args(["--%s-vision=transient" % other, "--%s-d-adopt" % other])
        self.assertEqual(vars(a), vars(b))
        self.assertEqual(vars(a), vars(c))
        self.assertEqual(a.__dict__["%s_vision" % run], "transient")   # dest comes from the running spelling
        self.assertNotIn("%s_vision" % other, vars(b))

    def test_help_lists_both_and_registration_is_idempotent(self):
        ap, run = self._parser()
        other = cs.FLAG_TOKENS[1] if run == cs.FLAG_TOKENS[0] else cs.FLAG_TOKENS[0]
        cs.register_flag_aliases(ap)
        self.assertEqual(cs.register_flag_aliases(ap), 0)
        h = ap.format_help()
        self.assertIn("--%s-vision" % run, h)
        self.assertIn("--%s-vision" % other, h)

    def test_a_choice_error_still_refuses_under_the_alias(self):
        ap, run = self._parser()
        other = cs.FLAG_TOKENS[1] if run == cs.FLAG_TOKENS[0] else cs.FLAG_TOKENS[0]
        cs.register_flag_aliases(ap)
        with mock.patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
            ap.parse_args(["--%s-vision" % other, "nonsense"])

    def test_a_non_parser_is_left_alone(self):
        self.assertEqual(cs.register_flag_aliases(object()), 0)

    def test_the_launcher_parser_accepts_every_flip_flag_in_both_spellings(self):
        from flliper.srt.pdflip import launcher

        ap = launcher.build_parser()
        run = cs.FLAG_TOKENS[0] if cs.running_package() == PKG_OLD else cs.FLAG_TOKENS[1]
        other = cs.FLAG_TOKENS[1] if run == cs.FLAG_TOKENS[0] else cs.FLAG_TOKENS[0]
        mine = {o: a for a in ap._actions for o in a.option_strings if o.startswith("--%s-" % run)}
        self.assertGreaterEqual(len(mine), 10)
        for o, act in mine.items():
            alias = "--%s-%s" % (other, o[len(run) + 3:])
            self.assertIs(ap._option_string_actions.get(alias), act, o)    # the SAME action: dest, default, choices
            self.assertIn(alias, act.option_strings)

    def test_the_launcher_parser_parses_a_profile_in_the_other_spelling(self):
        from flliper.srt.pdflip import launcher

        run = cs.FLAG_TOKENS[0] if cs.running_package() == PKG_OLD else cs.FLAG_TOKENS[1]
        other = cs.FLAG_TOKENS[1] if run == cs.FLAG_TOKENS[0] else cs.FLAG_TOKENS[0]
        base = ["--tree", "/t", "--tag", "x"]
        a = launcher.build_parser().parse_args(base + ["--%s-xchg-legs" % run, "both", "--%s-xchg-inject" % run, "authoritative"])
        b = launcher.build_parser().parse_args(base + ["--%s-xchg-legs" % other, "both", "--%s-xchg-inject" % other, "authoritative"])
        self.assertEqual(vars(a), vars(b))

    def test_server_args_parser_has_the_hook_and_no_flip_flags_to_alias(self):
        """ServerArgs defines no flip-subsystem flag (they live in the launcher parser): the hook registers 0."""
        import argparse

        from flliper.srt.server_args import ServerArgs

        ap = argparse.ArgumentParser()
        ServerArgs.add_cli_args(ap)
        run = cs.FLAG_TOKENS[0] if cs.running_package() == PKG_OLD else cs.FLAG_TOKENS[1]
        self.assertEqual([o for a in ap._actions for o in a.option_strings if o.startswith("--%s-" % run)], [])
        self.assertEqual(cs.register_flag_aliases(ap), 0)


class TestCacheReadFallback(CustomTestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="f0c-home-")
        self.run = cs._cache_root(cs.running_package(), self.home)
        self.oth = cs._cache_root(cs.other_package(), self.home)
        os.makedirs(self.run)
        os.makedirs(self.oth)

    def _put(self, root, name, text="x"):
        with open(os.path.join(root, name), "w") as f:
            f.write(text)
        return os.path.join(root, name)

    def test_read_fallback_prefers_the_running_name_then_reads_the_other(self):
        f_run = os.path.join(self.run, "card_library.json")
        f_oth = self._put(self.oth, "card_library.json")
        self.assertEqual(cs.read_fallback(f_run, home=self.home), f_oth)
        self._put(self.run, "card_library.json")
        self.assertEqual(cs.read_fallback(f_run, home=self.home), f_run)

    def test_the_by_uuid_side_file_is_read_the_same_way(self):
        side = os.path.join(self.run, "card_library.json.by-uuid.json")
        oth = self._put(self.oth, "card_library.json.by-uuid.json")
        self.assertEqual(cs.read_fallback(side, home=self.home), oth)

    def test_nothing_anywhere_returns_the_primary_path_for_the_caller_to_refuse_on(self):
        p = os.path.join(self.run, "card_library.json")
        self.assertEqual(cs.read_fallback(p, home=self.home), p)

    def test_a_path_outside_the_rig_cache_is_never_redirected(self):
        tmp = self._put(tempfile.mkdtemp(), "card_library.json")
        os.remove(tmp)
        self._put(self.oth, "card_library.json")
        self.assertEqual(cs.read_fallback(tmp, home=self.home), tmp)
        self.assertEqual(cs.cache_dirs(os.path.dirname(tmp), home=self.home), (os.path.dirname(tmp),))

    def test_cache_dirs_lists_the_other_dir_and_dedups_a_symlink(self):
        self.assertEqual(cs.cache_dirs(self.run, home=self.home), (self.run, self.oth))
        os.rmdir(self.run)
        os.rmdir(self.oth)
        os.makedirs(self.oth)
        os.symlink(self.oth, self.run, target_is_directory=True)      # what link_legacy_cache_dir makes
        self.assertEqual(cs.cache_dirs(self.run, home=self.home), (self.run,))
        # a missing other dir
        import shutil

        os.unlink(self.run)
        shutil.rmtree(self.oth)
        os.makedirs(self.run)
        self.assertEqual(cs.cache_dirs(self.run, home=self.home), (self.run,))

    def test_subdirectories_map_too(self):
        os.makedirs(os.path.join(self.oth, "rigmon"))
        f = self._put(os.path.join(self.oth, "rigmon"), "state.json")
        self.assertEqual(cs.read_fallback(os.path.join(self.run, "rigmon", "state.json"), home=self.home), f)

    # -- the wired readers ---------------------------------------------------

    def test_load_measured_library_reads_the_library_of_the_other_name_and_does_not_write(self):
        from flliper.srt.planner import card_rate_pass as crp
        from flliper.srt.planner.card_library import CardLibrary
        from flliper.srt.rigmon import card_probe

        CardLibrary().save(os.path.join(self.oth, "card_library.json"))
        with mock.patch.dict(os.environ, {"HOME": self.home}), mock.patch.object(card_probe, "CACHE_DIR", self.run):
            for side in (0, 1):
                os.environ.pop(GEN[side] + "CARD_LIBRARY", None)
            lib = crp.load_measured_library()
            self.assertIsNotNone(lib, "the library of the other name must be read")
            self.assertEqual(crp.card_library_path(), os.path.join(self.run, "card_library.json"))   # WRITES stay
        self.assertEqual(os.listdir(self.run), [])

    def test_load_measured_library_still_refuses_when_neither_exists(self):
        from flliper.srt.planner import card_rate_pass as crp
        from flliper.srt.rigmon import card_probe

        with mock.patch.dict(os.environ, {"HOME": self.home}), mock.patch.object(card_probe, "CACHE_DIR", self.run):
            for side in (0, 1):
                os.environ.pop(GEN[side] + "CARD_LIBRARY", None)
            self.assertIsNone(crp.load_measured_library())

    def test_the_hardware_profile_of_the_other_name_is_found_by_the_collector(self):
        from flliper.srt.rigmon import collector

        self._put(self.oth, "hw_profile-aaaa.json", json.dumps({"created": "2026-10-01 00:00:00", "gpus": {}}))
        self._put(self.oth, "power_profile.json", json.dumps({"limit": 1}))
        with mock.patch.dict(os.environ, {"HOME": self.home}):
            hw, power = collector.load_cached_profiles(self.run)
        self.assertEqual(hw["created"], "2026-10-01 00:00:00")
        self.assertEqual(power, {"limit": 1})
        # the running name's own newer profile still wins
        self._put(self.run, "hw_profile-bbbb.json", json.dumps({"created": "2026-10-05 00:00:00", "gpus": {}}))
        with mock.patch.dict(os.environ, {"HOME": self.home}):
            hw, _ = collector.load_cached_profiles(self.run)
        self.assertEqual(hw["created"], "2026-10-05 00:00:00")

    def test_stage0_and_probes_are_listed_from_both_dirs_by_the_hardware_profile_module(self):
        from flliper.srt.rigmon import hardware_profile as hp

        for d, n in ((self.run, "hw_profile-run.json"), (self.oth, "hw_profile-oth.json")):
            self._put(d, n, json.dumps({"created": "2026-10-01 00:00:00", "driver": "1", "gpus": {"0": {}}}))
        self._put(self.oth, "hw_profile-run.json", json.dumps({"created": "2026-01-01 00:00:00", "gpus": {"0": {}}}))  # same name: first wins
        with mock.patch.dict(os.environ, {"HOME": self.home}):
            got = {p["file"]: p for p in hp.load_stage0(self.run)}
        self.assertEqual(sorted(got), ["hw_profile-oth.json", "hw_profile-run.json"])
        self.assertEqual(got["hw_profile-run.json"]["driver"], "1")      # the running name's file, not the other's
        with mock.patch.dict(os.environ, {"HOME": self.home}):
            self.assertEqual(hp.load_stage0(self.run + "-elsewhere"), [])   # not the rig cache: no redirect

    def test_the_profile_for_a_narrowed_view_is_found_in_the_other_dir(self):
        from flliper.srt import uneven_perf as up

        prof = {"driver": "595", "uuids": ["GPU-A", "GPU-B"], "version": up.PROFILE_VERSION}
        self._put(self.oth, "hw_profile-zz.json", json.dumps(prof))
        with mock.patch.dict(os.environ, {"HOME": self.home}), mock.patch.object(up, "PROFILE_CACHE_DIR", self.run):
            got = up._cached_profile_for_view(["GPU-A"], "595")
            self.assertEqual(got, prof)
            # _load_profile (the exact-key reader) reads through as well
            self.assertEqual(up._load_profile(os.path.join(self.run, "hw_profile-zz.json"), "595", ["GPU-A", "GPU-B"]), prof)
        self.assertEqual(os.listdir(self.run), [])      # nothing was written

    def test_a_tmp_cache_dir_is_read_as_before(self):
        from flliper.srt import uneven_perf as up

        d = tempfile.mkdtemp()
        prof = {"driver": "595", "uuids": ["GPU-A"], "version": up.PROFILE_VERSION}
        self._put(d, "hw_profile-t.json", json.dumps(prof))
        self._put(self.oth, "hw_profile-other.json", json.dumps({"driver": "595", "uuids": ["GPU-A", "GPU-Z"]}))
        with mock.patch.dict(os.environ, {"HOME": self.home}), mock.patch.object(up, "PROFILE_CACHE_DIR", d):
            self.assertEqual(up._cached_profile_for_view(["GPU-A"], "595"), prof)


class TestCensusSeesBothGenerations(CustomTestCase):
    def test_titles_and_module_names_of_either_generation(self):
        from flliper.srt.pdflip import host_census as hc

        for pkg, sub in ((PKG_OLD, cs.FLAG_TOKENS[0]), (PKG_NEW, cs.FLAG_TOKENS[1])):
            self.assertEqual(hc.classify_process("%s::scheduler_PP0" % pkg, ""), hc.RANK_ROLE, pkg)
            self.assertEqual(hc.classify_process("python", "%s::scheduler_TP1" % pkg), hc.RANK_ROLE, pkg)
            self.assertEqual(hc.classify_process("%s::detokenizer" % pkg, ""), "detokenizer", pkg)
            self.assertEqual(hc.classify_process("python", "python -m %s.srt.%s.front --x" % (pkg, sub)), "front", pkg)
            self.assertEqual(hc.classify_process("python", "python -m %s.srt.%s.launcher" % (pkg, sub)), "launcher", pkg)
            self.assertEqual(hc.classify_process("python", "python -m %s.launch_server" % pkg), "server_main", pkg)
        self.assertEqual(hc.classify_process("python", "python -m vllm.entrypoints"), "other")


if __name__ == "__main__":
    unittest.main()
