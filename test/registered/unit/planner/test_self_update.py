"""Hermetic CPU tests for the dashboard self-update machinery.

No GPU, no network, no real dashboard restart: the version store, the local
git source (against a throwaway git repo), the auto-rollback decision, the
downgrade write-guard, and the code/data separation gate are exercised as
plain functions, plus the /api/version routes over one in-process HTTP
round-trip.
"""

import hashlib
import json
import os
import subprocess
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from unittest import mock

from sglang.srt.planner import self_update as su
from sglang.srt.planner import webui
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def _git(repo, *args):
    subprocess.run(
        ["git", "-C", repo, *args],
        check=True,
        capture_output=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
        },
    )


def _make_repo(root):
    """A minimal fork checkout: python/ subtree + two dashboard-v* tags."""
    os.makedirs(os.path.join(root, "python", "pkg"))
    _git(root, "init", "-q")
    with open(os.path.join(root, "python", "pkg", "mod.py"), "w") as f:
        f.write("VERSION = 1\n")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "v1")
    _git(root, "tag", "-a", "dashboard-v0.0.1", "-m", "v0.0.1")
    with open(os.path.join(root, "python", "pkg", "mod.py"), "w") as f:
        f.write("VERSION = 2\n")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "v2")
    _git(root, "tag", "-a", "dashboard-v0.0.2", "-m", "v0.0.2")
    return root


def _fake_install(store, vid, installed_at="2026-01-01T00:00:00"):
    """Handcraft an installed version (no git needed)."""
    d = store.version_dir(vid)
    os.makedirs(os.path.join(d, "python"), exist_ok=True)
    with open(os.path.join(d, su.MANIFEST_NAME), "w") as f:
        json.dump(
            {"version": vid, "source": "test", "installed_at": installed_at}, f
        )


def _tree_digest(root):
    h = hashlib.sha256()
    for dirpath, dirnames, filenames in sorted(os.walk(root)):
        dirnames.sort()
        for name in sorted(filenames):
            p = os.path.join(dirpath, name)
            h.update(os.path.relpath(p, root).encode())
            with open(p, "rb") as f:
                h.update(f.read())
    return h.hexdigest()


class TestLocalGitSource(CustomTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = _make_repo(os.path.join(self._tmp.name, "repo"))
        self.home = os.path.join(self._tmp.name, "home")

    def tearDown(self):
        self._tmp.cleanup()

    def test_lists_tags_and_head(self):
        src = su.LocalGitSource(self.repo)
        self.assertTrue(src.configured)
        vers = {v.id: v for v in src.list_versions()}
        self.assertIn("0.0.1", vers)
        self.assertIn("0.0.2", vers)
        self.assertEqual(vers["0.0.1"].origin, "local-git")
        self.assertTrue(vers["0.0.1"].date)
        head = [v for v in vers.values() if v.ref == "HEAD"]
        self.assertEqual(len(head), 1)
        self.assertTrue(head[0].id.startswith("head-"))

    def test_install_fills_version_dir_and_pointer(self):
        src = su.LocalGitSource(self.repo)
        store = su.VersionStore(self.home)
        v1 = next(v for v in src.list_versions() if v.id == "0.0.1")
        manifest = store.install(src, v1)
        self.assertEqual(manifest["version"], "0.0.1")
        mod = os.path.join(store.version_dir("0.0.1"), "python", "pkg", "mod.py")
        with open(mod) as f:
            self.assertEqual(f.read(), "VERSION = 1\n")  # the TAGGED content
        self.assertTrue(store.is_installed("0.0.1"))
        self.assertIsNone(store.current_id())
        store.set_current("0.0.1")
        self.assertEqual(store.current_id(), "0.0.1")
        self.assertEqual(
            store.python_root("0.0.1"),
            os.path.join(store.version_dir("0.0.1"), "python"),
        )

    def test_set_current_refuses_uninstalled(self):
        store = su.VersionStore(self.home)
        with self.assertRaises(ValueError):
            store.set_current("9.9.9")

    def test_github_release_source_is_a_clean_stub(self):
        src = su.GitHubReleaseSource()
        self.assertFalse(src.configured)
        self.assertIn("no remote release source", src.note)
        self.assertEqual(src.list_versions(), [])
        with self.assertRaises(RuntimeError):
            src.install(
                su.VersionInfo(id="x", label="x", origin="github-release", ref="x"),
                self.home,
            )


class TestAutoRollback(CustomTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = su.VersionStore(os.path.join(self._tmp.name, "home"))
        _fake_install(self.store, "v1", "2026-01-01T00:00:00")
        _fake_install(self.store, "v2", "2026-01-02T00:00:00")

    def tearDown(self):
        self._tmp.cleanup()

    def test_health_pass_marks_good(self):
        self.store.set_current("v1")
        out = su.apply_health_result(self.store, "v1", healthy=True)
        self.assertEqual(out["action"], "none")
        self.assertTrue(self.store.is_good("v1"))

    def test_health_fail_rolls_back_to_last_good(self):
        self.store.set_current("v1")
        self.store.mark_good("v1")
        self.store.set_current("v2")  # the switch the supervisor performed
        out = su.apply_health_result(self.store, "v2", healthy=False)
        self.assertEqual(out["action"], "rollback")
        self.assertEqual(out["rolled_back_to"], "v1")
        self.assertEqual(self.store.current_id(), "v1")
        self.assertFalse(self.store.is_good("v2"))

    def test_health_fail_without_fallback_halts(self):
        self.store.set_current("v2")
        out = su.apply_health_result(self.store, "v2", healthy=False)
        self.assertEqual(out["action"], "halt")
        self.assertEqual(self.store.current_id(), "v2")  # pointer untouched

    def test_switch_request_roundtrip(self):
        self.store.write_switch_request("v2")
        req = self.store.take_switch_request()
        self.assertEqual(req["target"], "v2")
        self.assertIsNone(self.store.take_switch_request())  # consumed

    def test_cleanup_plan_protects_current_and_last_good(self):
        for i in range(3, 9):
            _fake_install(self.store, f"v{i}", f"2026-01-{i:02d}T00:00:00")
        self.store.set_current("v2")
        self.store.mark_good("v1")
        plan = self.store.cleanup_plan(keep=3)
        self.assertNotIn("v2", plan)  # current
        self.assertNotIn("v1", plan)  # last good
        # newest-installed three of the rest (v8, v7, v6) survive.
        for kept in ("v8", "v7", "v6"):
            self.assertNotIn(kept, plan)
        self.assertEqual(set(plan), {"v3", "v4", "v5"})
        removed = self.store.cleanup(keep=3)
        self.assertEqual(set(removed), {"v3", "v4", "v5"})
        self.assertFalse(self.store.is_installed("v3"))
        self.assertTrue(self.store.is_installed("v2"))


class TestDataSchemaGuard(CustomTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.data = os.path.join(self._tmp.name, "data")

    def tearDown(self):
        self._tmp.cleanup()

    def test_stamp_written_and_idempotent(self):
        s1 = su.stamp_data_schema(self.data)
        self.assertEqual(s1["schema_version"], su.DATA_SCHEMA_VERSION)
        stamp_file = os.path.join(self.data, su.SCHEMA_STAMP_NAME)
        with open(stamp_file, "rb") as f:
            raw1 = f.read()
        s2 = su.stamp_data_schema(self.data)
        self.assertEqual(s2["schema_version"], su.DATA_SCHEMA_VERSION)
        with open(stamp_file, "rb") as f:
            # Re-stamping the same generation is byte-neutral: restarts and
            # version switches never touch the data dir.
            self.assertEqual(f.read(), raw1)
        self.assertIsNone(su.data_write_guard(self.data))

    def test_newer_stamp_blocks_writes_and_is_never_downgraded(self):
        os.makedirs(self.data)
        newer = {"schema_version": su.DATA_SCHEMA_VERSION + 5, "written_by": "9.9.9"}
        with open(os.path.join(self.data, su.SCHEMA_STAMP_NAME), "w") as f:
            json.dump(newer, f)
        warning = su.data_write_guard(self.data)
        self.assertIsNotNone(warning)
        self.assertIn("read-only", warning)
        # stamping again must NOT downgrade the newer stamp.
        kept = su.stamp_data_schema(self.data)
        self.assertEqual(kept["schema_version"], su.DATA_SCHEMA_VERSION + 5)

    def test_webui_write_endpoints_refuse_on_newer_schema(self):
        os.makedirs(self.data)
        with open(os.path.join(self.data, su.SCHEMA_STAMP_NAME), "w") as f:
            json.dump({"schema_version": su.DATA_SCHEMA_VERSION + 1}, f)
        with mock.patch.dict(
            os.environ, {"SGLANG_PLANNER_DATA_DIR": self.data}
        ):
            d = webui.config_profiles_save(
                {"name": "x", "settings": {"model": "m"}}
            )
            self.assertFalse(d["ok"])
            self.assertIn("schema", d["error"])
            d = webui.quality_save_payload({"model": "m", "save": True})
            self.assertFalse(d["ok"])
            d = webui.hicache_saved_record({"model": "m", "recovered_tokens": 1})
            self.assertFalse(d["ok"])
            d = webui.config_profiles_delete({"name": "x"})
            self.assertFalse(d["ok"])


class TestCodeDataSeparation(CustomTestCase):
    """The hard gate: install + switch + rollback leave the data dir
    byte-identical."""

    def test_version_lifecycle_leaves_data_dir_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = os.path.join(tmp, "data")
            os.makedirs(os.path.join(data, "gguf_headers"))
            for name, content in [
                ("planner_profiles.json", '{"profiles": {"a": 1}}'),
                ("power_profile.json", '{"cards": []}'),
                ("quality_shots.jsonl", '{"ts": "x"}\n'),
                (os.path.join("gguf_headers", "h.json"), "{}"),
            ]:
                with open(os.path.join(data, name), "w") as f:
                    f.write(content)
            before = _tree_digest(data)

            repo = _make_repo(os.path.join(tmp, "repo"))
            store = su.VersionStore(os.path.join(tmp, "home"))
            src = su.LocalGitSource(repo)
            vers = {v.id: v for v in src.list_versions()}
            store.install(src, vers["0.0.1"])
            store.install(src, vers["0.0.2"])
            store.set_current("0.0.1")
            su.apply_health_result(store, "0.0.1", healthy=True)
            store.set_current("0.0.2")
            su.apply_health_result(store, "0.0.2", healthy=False)  # rollback
            self.assertEqual(store.current_id(), "0.0.1")
            store.cleanup()

            self.assertEqual(before, _tree_digest(data))

    def test_legacy_store_migration_is_copy_forward_and_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            legacy = os.path.join(tmp, "pkg", "store.json")
            os.makedirs(os.path.dirname(legacy))
            with open(legacy, "w") as f:
                f.write('{"rows": [1]}')
            data = os.path.join(tmp, "data")
            new = su.planner_data_path("store.json", legacy=legacy, data_dir=data)
            self.assertEqual(new, os.path.join(data, "store.json"))
            with open(new) as f:
                self.assertEqual(f.read(), '{"rows": [1]}')
            # idempotent: a second resolve never overwrites the migrated copy.
            with open(new, "w") as f:
                f.write('{"rows": [1, 2]}')
            again = su.planner_data_path("store.json", legacy=legacy, data_dir=data)
            self.assertEqual(again, new)
            with open(new) as f:
                self.assertEqual(f.read(), '{"rows": [1, 2]}')
            # the legacy copy stays in place as an inert backup.
            self.assertTrue(os.path.exists(legacy))

    def test_default_store_paths_live_outside_the_code_tree(self):
        import sglang.srt.planner as planner_pkg

        pkg_root = os.path.dirname(os.path.abspath(planner_pkg.__file__))
        from sglang.srt.planner.energy import DEFAULT_RESULTS_STORE
        from sglang.srt.planner.hicache_savings import DEFAULT_HICACHE_STORE

        for p in (DEFAULT_HICACHE_STORE, DEFAULT_RESULTS_STORE):
            self.assertFalse(
                os.path.abspath(p).startswith(pkg_root + os.sep),
                f"store default {p} lives inside the code tree",
            )


class TestVersionRoutes(CustomTestCase):
    """/api/version + /api/version/switch payloads and HTTP wiring."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.home = os.path.join(self._tmp.name, "home")
        self.env = mock.patch.dict(
            os.environ,
            {
                "SGLANG_DASHBOARD_HOME": self.home,
                "SGLANG_PLANNER_DATA_DIR": os.path.join(self._tmp.name, "data"),
            },
        )
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self._tmp.cleanup()

    def test_version_payload_shape(self):
        d = webui.version_payload()
        self.assertTrue(d["ok"])
        self.assertIn("version", d["current"])
        self.assertIn(d["current"]["origin"], ("checkout", "installed"))
        self.assertFalse(d["current"]["supervised"])
        self.assertIsInstance(d["versions"], list)
        names = {s["name"] for s in d["sources"]}
        self.assertEqual(names, {"local-git", "github-release"})
        gh = next(s for s in d["sources"] if s["name"] == "github-release")
        self.assertFalse(gh["configured"])
        self.assertTrue(gh["note"])

    def test_switch_requires_confirmation_and_supervisor(self):
        d = webui.version_switch_payload({"action": "switch", "version": "x"})
        self.assertFalse(d["ok"])
        self.assertIn("confirmation", d["error"])
        d = webui.version_switch_payload(
            {"action": "switch", "version": "x", "confirmed": True}
        )
        self.assertFalse(d["ok"])
        self.assertIn("--serve-supervised", d["error"])

    def test_supervised_switch_writes_request_and_restarts(self):
        store = su.VersionStore(self.home)
        _fake_install(store, "v1")
        _fake_install(store, "v2")
        store.set_current("v1")
        with mock.patch.dict(os.environ, {"SGLANG_DASHBOARD_SUPERVISED": "1"}):
            d = webui.version_switch_payload(
                {"action": "switch", "version": "v1", "confirmed": True}
            )
            self.assertFalse(d["ok"])  # already current
            with mock.patch.object(webui, "request_restart", return_value=True):
                d = webui.version_switch_payload(
                    {"action": "switch", "version": "v2", "confirmed": True}
                )
        self.assertTrue(d["ok"])
        self.assertTrue(d["restarting"])
        self.assertEqual(store.take_switch_request()["target"], "v2")
        # the pointer is NOT moved by the worker — that is the supervisor's job.
        self.assertEqual(store.current_id(), "v1")

    def test_install_via_route_from_local_git(self):
        repo = _make_repo(os.path.join(self._tmp.name, "repo"))
        with mock.patch.object(
            su, "default_sources", return_value=[su.LocalGitSource(repo)]
        ):
            d = webui.version_switch_payload(
                {"action": "install", "version": "0.0.2", "confirmed": True}
            )
        self.assertTrue(d["ok"])
        self.assertEqual(d["installed"], "0.0.2")
        self.assertTrue(su.VersionStore(self.home).is_installed("0.0.2"))

    def test_cleanup_route_previews_without_confirmation(self):
        store = su.VersionStore(self.home)
        for i in range(6):
            _fake_install(store, f"v{i}", f"2026-01-0{i + 1}T00:00:00")
        store.set_current("v5")
        d = webui.version_cleanup_payload({})
        self.assertTrue(d["ok"])
        self.assertEqual(len(d["would_remove"]), 2)
        self.assertEqual(d["removed"], [])
        d = webui.version_cleanup_payload({"confirmed": True})
        self.assertEqual(len(d["removed"]), 2)

    def test_http_round_trip(self):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), webui._Handler)
        port = srv.server_address[1]
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/version", timeout=10
            ) as r:
                d = json.loads(r.read())
            self.assertTrue(d["ok"])
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/version/switch",
                data=json.dumps(
                    {"action": "switch", "version": "x", "confirmed": True}
                ).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                d = json.loads(r.read())
            self.assertFalse(d["ok"])  # unsupervised: refused, not crashed
        finally:
            srv.shutdown()
            srv.server_close()
            thread.join(timeout=5)

    def test_index_has_about_tab_and_version_api(self):
        for marker in (
            "tab_about",
            "view_about",
            "/api/version",
            "loadVersionInfo",
            "aboutPollRestart",
        ):
            self.assertIn(marker, webui.INDEX_HTML)


class TestSupervisorHelpers(CustomTestCase):
    def test_wait_health_gives_up_when_proc_dies(self):
        class DeadProc:
            def poll(self):
                return 1

        ok = su.wait_health(
            "http://127.0.0.1:9/", timeout=5, interval=0.01, proc=DeadProc()
        )
        self.assertFalse(ok)

    def test_serve_supervised_flag_parses(self):
        from sglang.srt.planner.cli import build_parser

        args = build_parser().parse_args(["--serve-supervised", "--port", "1234"])
        self.assertTrue(args.serve_supervised)
        args = build_parser().parse_args(["--serve"])
        self.assertFalse(args.serve_supervised)


if __name__ == "__main__":
    unittest.main()
