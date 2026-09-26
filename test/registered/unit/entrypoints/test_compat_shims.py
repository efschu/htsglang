"""Rename transition shims (RENAME_PLAN 8.7 step 2): routes, rig-state dir, operator dir, env bridge call.

The last test class is the one that matters most: the shim files must come out of the mechanical
rename pass byte-identical and still bridge old <-> new afterwards. The rename rules are restated
here (same regexes as rename_to_flliper.py MAIN / W_MAIN) because the tool lives outside the tree.
"""

import asyncio
import os
import re
import sys
import tempfile
import types
import unittest
from unittest import mock

from sglang.srt import compat_shims as cs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

OLD, NEW = cs.ROUTE_PREFIXES
PKG_OLD, PKG_NEW = "sg" + "lang", "fl" + "liper"


class TestCounterpart(CustomTestCase):
    def test_both_directions(self):
        self.assertEqual(cs.counterpart(OLD + "state"), NEW + "state")
        self.assertEqual(cs.counterpart(NEW + "flip"), OLD + "flip")

    def test_other_paths_are_untouched(self):
        for p in ("/health", "/v1/models", "/generate", OLD.rstrip("/"), "/x" + OLD):
            self.assertIsNone(cs.counterpart(p), p)


class TestAiohttpAliases(CustomTestCase):
    def _app(self):
        from aiohttp import web

        async def state(request):
            return web.json_response({"state": "serving", "path": request.path})

        async def flip(request):
            return web.json_response({"flipped": True})

        app = web.Application()
        app.router.add_get(OLD + "state", state)
        app.router.add_post(OLD + "flip", flip)
        app.router.add_get("/health", state)
        return app

    def test_the_new_prefix_answers_with_the_same_handler(self):
        from aiohttp.test_utils import TestClient, TestServer

        app = self._app()
        self.assertEqual(cs.alias_aiohttp_routes(app), 2)

        async def run():
            async with TestClient(TestServer(app)) as c:
                a = await (await c.get(OLD + "state")).json()
                b = await (await c.get(NEW + "state")).json()
                f = await c.post(NEW + "flip")
                h = await c.get(NEW + "health".lstrip("/"))
                return a, b, f.status, h.status

        a, b, fs, hs = asyncio.run(run())
        self.assertEqual((a["state"], b["state"]), ("serving", "serving"))
        self.assertEqual(b["path"], NEW + "state")
        self.assertEqual(fs, 200)
        self.assertEqual(hs, 404, "only flip-front routes are aliased")

    def test_idempotent_and_never_overwrites(self):
        from aiohttp import web

        app = self._app()

        async def own(request):
            return web.json_response({"own": True})

        app.router.add_get(NEW + "state", own)
        self.assertEqual(cs.alias_aiohttp_routes(app), 1)  # only the flip POST
        self.assertEqual(cs.alias_aiohttp_routes(app), 0)


class TestFastapiAliases(CustomTestCase):
    def test_post_route_answers_under_both_prefixes(self):
        from fastapi import FastAPI
        from starlette.testclient import TestClient

        app = FastAPI()

        @app.api_route(OLD + "ple_prefetch_hint", methods=["POST"])
        async def hint():
            return {"ok": True}

        self.assertEqual(cs.alias_fastapi_routes(app), 1)
        self.assertEqual(cs.alias_fastapi_routes(app), 0)
        c = TestClient(app)
        self.assertEqual(c.post(OLD + "ple_prefetch_hint").json(), {"ok": True})
        self.assertEqual(c.post(NEW + "ple_prefetch_hint").json(), {"ok": True})
        self.assertNotIn(NEW + "ple_prefetch_hint", str(app.openapi()["paths"]))


class TestStateDir(CustomTestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp()

    def _mk(self, pkg, name="card_library.json"):
        d = os.path.join(self.home, ".cache", pkg)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, name), "w") as f:
            f.write("{}")
        return d

    def test_after_the_rename_the_new_dir_links_to_the_old_one(self):
        old = self._mk(PKG_OLD)
        link = cs.link_legacy_cache_dir(home=self.home, name=PKG_NEW + ".srt.x")
        self.assertEqual(link, os.path.join(self.home, ".cache", PKG_NEW))
        self.assertEqual(os.path.realpath(link), os.path.realpath(old))
        self.assertTrue(os.path.exists(os.path.join(link, "card_library.json")))
        self.assertIsNone(cs.link_legacy_cache_dir(home=self.home, name=PKG_NEW + ".srt.x"))

    def test_before_the_rename_nothing_happens(self):
        self._mk(PKG_OLD)
        self.assertIsNone(cs.link_legacy_cache_dir(home=self.home, name=PKG_OLD + ".srt.x"))
        self.assertFalse(os.path.lexists(os.path.join(self.home, ".cache", PKG_NEW)))

    def test_an_existing_new_dir_is_never_replaced_but_read_through(self):
        self._mk(PKG_OLD)
        new = os.path.join(self.home, ".cache", PKG_NEW)
        os.makedirs(new)
        self.assertIsNone(cs.link_legacy_cache_dir(home=self.home, name=PKG_NEW + ".x"))
        got = cs.cache_file("card_library.json", home=self.home, name=PKG_NEW + ".x")
        self.assertEqual(got, os.path.join(self.home, ".cache", PKG_OLD, "card_library.json"))
        self.assertEqual(cs.cache_file("absent.json", home=self.home, name=PKG_NEW + ".x"),
                         os.path.join(new, "absent.json"), "writers get the primary path")

    def test_a_failing_link_is_logged_not_raised(self):
        self._mk(PKG_OLD)
        with mock.patch("os.symlink", side_effect=PermissionError("ro")):
            self.assertIsNone(cs.link_legacy_cache_dir(home=self.home, name=PKG_NEW + ".x"))


class TestOperatorDir(CustomTestCase):
    SUB = "we" + "g2"

    def test_host_paths_keep_the_operator_subdir(self):
        from sglang.srt.weg2 import admin_key, corridor_budget

        self.assertEqual(cs.operator_dir("/g", "a.json"), "/g/" + self.SUB + "/a.json")
        self.assertEqual(admin_key.key_path("/g", "t1"), "/g/" + self.SUB + "/boot_t1.adminkey")
        if not os.environ.get("SG" "LANG_" "WE" "G2_GPU_ARB"):
            self.assertTrue(corridor_budget.DEFAULT_SAMPLE_PATH.endswith(
                "/gpu-arb/" + self.SUB + "/corridor_budget_sample.json"), corridor_budget.DEFAULT_SAMPLE_PATH)


class TestShmNameFamilies(CustomTestCase):
    SUB_OLD, SUB_NEW = "we" + "g2", "pd" + "flip"

    def test_both_directions_and_foreign_names_untouched(self):
        got = cs.name_counterparts((self.SUB_OLD + "-seq-", "sem." + self.SUB_NEW + "-xchg-",
                                    PKG_OLD + "_loads_", "sem.mp-", "hicache-" + self.SUB_OLD + "-"))
        self.assertEqual(got, (self.SUB_NEW + "-seq-", "sem." + self.SUB_OLD + "-xchg-", PKG_NEW + "_loads_",
                               "hicache-" + self.SUB_NEW + "-"))

    def test_already_listed_counterparts_are_not_repeated(self):
        self.assertEqual(cs.name_counterparts((self.SUB_OLD + "-seq-", self.SUB_NEW + "-seq-")), ())

    def test_the_launcher_sweep_lists_both_spellings(self):
        from sglang.srt.weg2 import launcher

        own = launcher.SHM_OWN_PREFIXES
        self.assertEqual(len(own), len(set(own)))
        for p in own:
            for cp in cs.name_counterparts((p,)):
                self.assertIn(cp, own, p)
        self.assertIn(self.SUB_OLD + "-seq-", own)
        self.assertIn(self.SUB_NEW + "-seq-", own)

    def test_the_semaphore_and_credit_sweeps_see_both_spellings(self):
        from sglang.srt.weg2 import launcher

        d = tempfile.mkdtemp()
        for tok in (self.SUB_OLD, self.SUB_NEW):
            with open(os.path.join(d, "sem." + tok + "-xchg-123-0-1-0-empty"), "w"):
                pass
            with open(os.path.join(d, "." + tok + "-vram-credit-GPU-x.json"), "w") as f:
                f.write('{"epoch": "1.2", "publisher_pid": 999999999}')
        seen = []
        got = launcher.sweep_xchg_semaphores(lambda m: None, shm_dir=d, dry=False,
                                                   unlink=lambda n: seen.append(n) or 0)
        self.assertEqual(len(seen), 2, (got, seen))
        rows = launcher._credit_counter_rows(d)
        self.assertEqual(len(rows), 2, rows)


class TestLegacyFlags(CustomTestCase):
    T_OLD, T_NEW = "we" + "g2", "pd" + "flip"

    def test_both_directions_and_values_untouched(self):
        argv = ["--tag", "x", "--" + self.T_OLD + "-vision", "off", "--" + self.T_OLD + "-weight-source=exchange",
                "--p-extra", "--" + self.T_OLD + "-not-a-flag-here and more"]
        got = cs.canonical_flags(argv, name=PKG_NEW + ".srt.x")
        self.assertEqual(got[2:4], ["--" + self.T_NEW + "-vision", "off"])
        self.assertEqual(got[4], "--" + self.T_NEW + "-weight-source=exchange")
        self.assertEqual(got[:2], ["--tag", "x"])
        back = cs.canonical_flags(got, name=PKG_OLD + ".srt.x")
        self.assertEqual(back[:5], argv[:5])

    def test_every_flip_flag_of_the_launcher_parses_in_both_spellings(self):
        from sglang.srt.weg2 import launcher

        opts = [o for a in launcher.build_parser()._actions for o in a.option_strings]
        run = cs.FLAG_TOKENS[0] if cs.running_package() == PKG_OLD else cs.FLAG_TOKENS[1]
        other = cs.FLAG_TOKENS[1] if run == cs.FLAG_TOKENS[0] else cs.FLAG_TOKENS[0]
        mine = [o for o in opts if o.startswith("--" + run + "-")]
        self.assertGreaterEqual(len(mine), 10)
        for o in mine:
            self.assertIn(cs.canonical_flags(["--" + other + o[len(run) + 2:]])[0], opts, o)
        self.assertIn("_canonical_flags(", open(launcher.__file__, encoding="utf-8").read())


def _fake_proc(root, pid, argv, env=None, nvidia=False, ppid=1):
    d = os.path.join(root, str(pid))
    os.makedirs(os.path.join(d, "fd"), exist_ok=True)
    with open(os.path.join(d, "cmdline"), "wb") as f:
        f.write(b"\0".join(a.encode() for a in argv) + b"\0")
    with open(os.path.join(d, "stat"), "w") as f:
        f.write(f"{pid} (x) S {ppid} {pid} {pid} 0 -1\n")
    with open(os.path.join(d, "environ"), "wb") as f:
        f.write(b"\0".join(f"{k}={v}".encode() for k, v in (env or {}).items()) + b"\0")
    with open(os.path.join(d, "maps"), "w") as f:
        f.write("7f0000000000-7f0000001000 rw-p 00000000 00:00 0 \n")
    if nvidia:
        os.symlink("/dev/nvidiactl", os.path.join(d, "fd", "7"))


class TestOtherGenerationServers(CustomTestCase):
    """#1217 across the rename (FL5, RENAME_PLAN 8.14 item 1): the launcher's live-server census
    must see a server of the other package generation as a server, and read its boot token under
    the other env spelling. Written for whichever side runs: the test file is renamed with the tree."""

    SUB_OLD, SUB_NEW = "we" + "g2", "pd" + "flip"
    ENV_OLD = "SG" "LANG_" + "WE" "G2_" + "BOOT_TOKEN"
    ENV_NEW = "FL" "LIPER_" + "PD" "FLIP_" + "BOOT_TOKEN"

    def setUp(self):
        from sglang.srt.weg2 import launcher

        self.launcher = launcher
        run = cs.running_package()
        self.run, self.other = run, cs.other_package()
        self.env_run, self.env_other = ((self.ENV_OLD, self.ENV_NEW) if run == PKG_OLD
                                        else (self.ENV_NEW, self.ENV_OLD))
        self.proc = tempfile.mkdtemp()

    def _server(self, pkg):
        return ["/v/bin/python3", "-m", pkg + ".launch_server", "--model-path", "/m", "--port", "30031"]

    def test_variants_both_directions(self):
        self.assertEqual(cs.name_variants(PKG_OLD + ".launch_server"),
                         (PKG_OLD + ".launch_server", PKG_NEW + ".launch_server"))
        self.assertEqual(cs.name_variants(PKG_NEW + ".launch_server"),
                         (PKG_NEW + ".launch_server", PKG_OLD + ".launch_server"))
        self.assertEqual(cs.name_variants("vllm.entrypoints"), ("vllm.entrypoints",))
        self.assertEqual(cs.env_name_variants(self.ENV_OLD), (self.ENV_OLD, self.ENV_NEW))
        self.assertEqual(cs.env_name_variants(self.ENV_NEW), (self.ENV_NEW, self.ENV_OLD))
        self.assertEqual(cs.env_name_variants("SG" "LANG_X"), ("SG" "LANG_X", "FL" "LIPER_X"))
        self.assertEqual(cs.env_name_variants("FL" "LIPER_X"), ("FL" "LIPER_X", "SG" "LANG_X"))
        for plain in ("SGL_X", "HT" "SG" "LANG_X", "PATH", "SG" "LANG_"):
            self.assertEqual(cs.env_name_variants(plain), (plain,))

    def test_the_launcher_constants_carry_both_generations_running_first(self):
        L = self.launcher
        self.assertEqual(L.LAUNCH_SERVER_MODULES, (self.run + ".launch_server", self.other + ".launch_server"))
        self.assertEqual(L.BOOT_TOKEN_ENV_KEYS, (self.env_run.encode() + b"=", self.env_other.encode() + b"="))
        src = open(L.__file__, encoding="utf-8").read()
        self.assertIn('_env_name_variants("' + self.env_run + '")', src,
                      "the key the launcher reads must be the one build_env writes")
        self.assertIn('env["' + self.env_run + '"] = ', src)

    def test_server_argv_of_either_generation(self):
        L = self.launcher
        for pkg in (PKG_OLD, PKG_NEW):
            self.assertTrue(L.is_launch_server_argv(self._server(pkg)), pkg)
            self.assertTrue(L.is_launch_server_argv(["python", "-m" + pkg + ".launch_server"]), pkg)
            for sub in (self.SUB_OLD, self.SUB_NEW):
                self.assertFalse(L.is_launch_server_argv(["python", "-m", pkg + ".srt." + sub + ".launcher"]))
            self.assertFalse(L.is_launch_server_argv(
                ["/bin/bash", "-c", 'pgrep -fc "' + pkg + '.launch_server"; bash start_when_free.sh x']))
            self.assertFalse(L.is_launch_server_argv(["python", "-m", pkg + ".launch_server_x"]))

    def test_boot_token_of_either_generation(self):
        L = self.launcher
        _fake_proc(self.proc, 10, self._server(self.other), {self.env_other: "tagA:1.2:99", "X": "1"})
        _fake_proc(self.proc, 11, self._server(self.run), {self.env_run: "tagB:1.2:99"})
        _fake_proc(self.proc, 12, self._server(self.other), {"X": "1"})
        # both spellings present (never written that way, but defined): the running one wins
        _fake_proc(self.proc, 13, self._server(self.run), {self.env_other: "tagO:1:1", self.env_run: "tagR:1:1"})
        self.assertEqual(L._proc_boot_tag("10", self.proc), "tagA")
        self.assertEqual(L._proc_boot_tag("11", self.proc), "tagB")
        self.assertIsNone(L._proc_boot_tag("12", self.proc))
        self.assertEqual(L._proc_boot_tag("13", self.proc), "tagR")
        self.assertIsNone(L._proc_boot_tag("404", self.proc))

    def test_the_census_sees_the_other_generation(self):
        L = self.launcher
        # a server of the other generation, foreign tag, no CUDA yet -> LIVE (before: invisible)
        _fake_proc(self.proc, 20, self._server(self.other), {self.env_other: "old1:1:1"})
        # a server of the other generation, OUR tag, no CUDA -> skipped like one of ours
        _fake_proc(self.proc, 21, self._server(self.other), {self.env_other: "mine:1:1"})
        # a server of the other generation, our tag, holding CUDA -> LIVE
        _fake_proc(self.proc, 22, self._server(self.other), {self.env_other: "mine:1:1"}, nvidia=True)
        # a stock server of the other generation without a token -> LIVE (foreign)
        _fake_proc(self.proc, 23, self._server(self.other))
        # an agent shell that only MENTIONS the other module -> ignored, never live
        _fake_proc(self.proc, 24, ["/bin/bash", "-c", 'pgrep -fc "' + self.other + '.launch_server"'])
        # the running generation still works the same
        _fake_proc(self.proc, 25, self._server(self.run), {self.env_run: "mine:1:1"})
        _fake_proc(self.proc, 26, self._server(self.run), {self.env_run: "old2:1:1"})
        live, ignored = L.live_launch_servers("mine", self.proc, self_pid=999)
        self.assertEqual(sorted(r["pid"] for r in live), [20, 22, 23, 26])
        self.assertEqual({r["pid"]: r["tag"] for r in live}, {20: "old1", 22: "mine", 23: None, 26: "old2"})
        why = {r["pid"]: r["why"] for r in ignored}
        self.assertEqual(why, {21: "own tag, no CUDA context", 24: "not a server argv",
                               25: "own tag, no CUDA context"})

    def test_the_sweep_refuses_on_a_live_server_of_the_other_generation(self):
        L = self.launcher
        shm = tempfile.mkdtemp()
        with open(os.path.join(shm, PKG_OLD + "_loads_x.shm"), "wb") as f:
            f.write(b"\0" * 16)
        _fake_proc(self.proc, 30, self._server(self.other), {self.env_other: "old1:1:1"})
        lines = []
        with self.assertRaises(L.Weg2LaunchRefused) as cm:
            L.shm_residue_sweep(lines.append, "mine", "stamp", dry=True, shm_dir=shm, proc_root=self.proc,
                                archive_root=tempfile.mkdtemp())
        self.assertIn("pid(s)=[30]", str(cm.exception))

    def test_vram_hires_names_the_ranks_of_either_generation(self):
        from sglang.srt.weg2.tools import vram_hires as vh

        for pkg in (PKG_OLD, PKG_NEW):
            m = vh._RX_SCHED.search(pkg + "::scheduler_PP2")
            self.assertEqual((m.group(1), m.group(2)), ("PP", "2"), pkg)
        self.assertIsNone(vh._RX_SCHED.search("vllm::scheduler_PP2"))


class TestEnvBridgeCall(CustomTestCase):
    def test_the_package_hook_calls_the_one_helper_with_the_environment(self):
        from sglang import _compat_boot

        seen = []
        stub = types.ModuleType("name_compat_stub")
        stub.canonical_env = lambda env, **kw: seen.append(env) or env
        key = _compat_boot.__name__.rsplit(".", 1)[0] + ".srt.name_compat"
        with mock.patch.dict(sys.modules, {key: stub}):
            env = {"X": "1"}
            _compat_boot.bridge_environ(env)
            _compat_boot.bridge_environ()
        self.assertIs(seen[0], env)
        self.assertIs(seen[1], os.environ)

    def test_the_state_dir_is_linked_by_the_entry_points_not_at_import(self):
        from sglang import _compat_boot

        with mock.patch.object(cs, "link_legacy_cache_dir") as link:
            _compat_boot.link_state_dir()
        link.assert_called_once_with()
        with open(os.path.join(os.path.dirname(_compat_boot.__file__), "__init__.py"), encoding="utf-8") as f:
            self.assertNotIn("link_state_dir()", f.read(), "an import must not write into $HOME")

    def test_a_missing_helper_is_an_import_error_not_a_silent_skip(self):
        from sglang import _compat_boot

        key = _compat_boot.__name__.rsplit(".", 1)[0] + ".srt.name_compat"
        with mock.patch.dict(sys.modules, {key: None}):
            with self.assertRaises(ImportError):
                _compat_boot.bridge_environ({})


# the mechanical pass, restated (rename_to_flliper.py MAIN / W_MAIN). Every old/new word is built by
# concatenation: a literal pair in THIS file would itself be a collision for the tool.
_O, _N = "sg" + "lang", "fl" + "liper"
_WO, _WN = "we" + "g2", "pd" + "flip"
_MAIN_MAP = {_O: _N, _O.upper(): _N.upper(), "S" + _O[1:]: "F" + _N[1:]}
_MAIN = re.compile(r"(?<![Hh][Tt])(" + "|".join([_O, _O.upper(), "SGL" + "ang", "S" + _O[1:], "SGl" + "ang", "sGL" + "ang"]) + ")")
_W_MAP = {_WO: _WN, _WO.upper(): _WN.upper()}
_W = re.compile("(" + "|".join(["W" + _WO[1:] + "Flip", _WO, _WO.upper(), "W" + _WO[1:]]) + ")")


def _rename(text):
    text = _MAIN.sub(lambda m: _MAIN_MAP.get(m.group(1), "fLL" + "iper"), text)
    return _W.sub(lambda m: _W_MAP.get(m.group(1), "Pd" + "Flip"), text)


class TestSurvivesTheMechanicalRename(CustomTestCase):
    def _src(self, mod):
        with open(mod.__file__, encoding="utf-8") as f:
            return f.read()

    def test_shim_files_are_a_fixed_point_of_the_rename(self):
        from sglang import _compat_boot

        for mod in (cs, _compat_boot):
            src = self._src(mod)
            self.assertEqual(_rename(src), src, mod.__file__)

    def test_after_the_rename_the_shim_still_bridges_old_and_new(self):
        ns = {"__name__": PKG_NEW + ".srt.compat_shims"}
        exec(compile(_rename(self._src(cs)), "renamed_compat_shims.py", "exec"), ns)
        self.assertEqual(ns["running_package"](), PKG_NEW)
        self.assertEqual(ns["other_package"](), PKG_OLD)
        self.assertEqual(ns["counterpart"](NEW + "state"), OLD + "state")
        self.assertEqual(ns["operator_dir"]("/g"), "/g/" + "we" + "g2")
        self.assertEqual(ns["name_counterparts"](("pd" + "flip-seq-",)), ("we" + "g2-seq-",))
        self.assertEqual(ns["canonical_flags"](["--we" + "g2-vision"]), ["--pd" + "flip-vision"])

    def test_the_restated_rules_really_rename(self):
        self.assertEqual(_rename(PKG_OLD + ".srt." + _WO + ".front"), PKG_NEW + ".srt." + _WN + ".front")


if __name__ == "__main__":
    unittest.main()
