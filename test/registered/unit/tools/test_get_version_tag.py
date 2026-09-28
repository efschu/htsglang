import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[4]
CI_REGISTER_PATH = REPO_ROOT / "python" / "sglang" / "test" / "ci" / "ci_register.py"
VERSION_HELPER_PATH = REPO_ROOT / "python" / "tools" / "get_version_tag.py"
PYPROJECT_PATHS = [
    REPO_ROOT / "python" / "pyproject.toml",
    REPO_ROOT / "python" / "pyproject_cpu.toml",
    REPO_ROOT / "python" / "pyproject_npu.toml",
    REPO_ROOT / "python" / "pyproject_other.toml",
    REPO_ROOT / "python" / "pyproject_xpu.toml",
    REPO_ROOT / "3rdparty" / "amd" / "wheel" / "sglang" / "pyproject.toml",
]
DESCRIBE_COMMAND = (
    'git_describe_command = ["python3", "python/tools/get_version_tag.py"]'
)
TAG_ONLY_DESCRIBE_COMMAND = (
    'git_describe_command = ["python3", "python/tools/get_version_tag.py", '
    '"--tag-only"]'
)
FALLBACK_VERSION = 'fallback_version = "0.0.0.dev0"'
#: fLLiper release item 3: the package we build (python/pyproject.toml) falls back to the release
#: being prepared and points at the fLLiper repo; the upstream platform variants stay as they were.
FLLIPER_PYPROJECT = REPO_ROOT / "python" / "pyproject.toml"
FLLIPER_FALLBACK_VERSION = 'fallback_version = "0.1.0.dev0"'


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


register_cpu_ci = _load_module("ci_register", CI_REGISTER_PATH).register_cpu_ci
register_cpu_ci(est_time=0, suite="base-a-test-cpu")


class TestGetVersionTag(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.version_helper = _load_module("get_version_tag", VERSION_HELPER_PATH)

    def test_parse_version_tuple_sorts_stable_above_rc_and_post_above_stable(self):
        tags = ["v0.5.10rc0", "v0.5.9", "v0.5.10.post1", "v0.5.10"]

        self.assertEqual(
            sorted(tags, key=self.version_helper.parse_version_tuple, reverse=True),
            ["v0.5.10.post1", "v0.5.10", "v0.5.10rc0", "v0.5.9"],
        )

    def test_exact_version_tag_takes_precedence_over_latest_tag(self):
        with (
            patch.object(
                self.version_helper, "get_exact_version_tag", return_value="v0.5.9"
            ),
            patch.object(
                self.version_helper, "get_latest_version_tag_describe"
            ) as latest_describe,
        ):
            self.assertEqual(self.version_helper.get_version_describe(), "v0.5.9")

        latest_describe.assert_not_called()

    def test_pyprojects_use_describe_mode_for_setuptools_scm(self):
        for path in PYPROJECT_PATHS:
            with self.subTest(path=path):
                content = path.read_text()
                self.assertIn(DESCRIBE_COMMAND, content)
                self.assertNotIn(TAG_ONLY_DESCRIBE_COMMAND, content)
                self.assertIn(
                    FLLIPER_FALLBACK_VERSION if path == FLLIPER_PYPROJECT else FALLBACK_VERSION,
                    content,
                )

    def test_the_built_package_points_at_the_flliper_repo(self):
        content = FLLIPER_PYPROJECT.read_text()
        self.assertIn('"Homepage" = "https://github.com/efschu/fLLiper"', content)
        self.assertIn('"Bug Tracker" = "https://github.com/efschu/fLLiper/issues"', content)
        self.assertNotIn("github.com/sgl-project/sglang", content.split("[project.urls]")[1].split("[")[0])

    def test_tag_only_cli_mode_remains_available_for_callers_that_need_latest_tag(self):
        with (
            patch.object(sys, "argv", ["get_version_tag.py", "--tag-only"]),
            patch.object(
                self.version_helper, "get_latest_version_tag", return_value="v0.5.10"
            ),
            patch.object(
                self.version_helper, "get_version_describe"
            ) as version_describe,
            patch("builtins.print") as print_mock,
        ):
            self.version_helper.main()

        version_describe.assert_not_called()
        print_mock.assert_called_once_with("v0.5.10")


class TestFlliperReleaseTags(unittest.TestCase):
    """fLLiper release item 3 (28.09.): a dev install of the fork read 0.5.21.dev6428 -- the
    version came from UPSTREAM sglang tags. It must come from fLLiper's own release tags
    (annotated, subject 'fLLiper ...'), and an untagged tree is <NEXT_RELEASE>.dev<N>+g<sha>."""

    @classmethod
    def setUpClass(cls):
        cls.version_helper = _load_module("get_version_tag_rel", VERSION_HELPER_PATH)

    def _repo(self):
        import os
        import subprocess
        import tempfile

        d = tempfile.mkdtemp(prefix="vtag_")
        env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
                   GIT_COMMITTER_EMAIL="t@t", GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_NOSYSTEM="1")

        def git(*a):
            return subprocess.run(["git", "-C", d, *a], check=True, capture_output=True, text=True, env=env).stdout.strip()

        git("init", "-q")
        return d, git

    def _run(self, d, *args):
        import subprocess

        return subprocess.run([sys.executable, str(VERSION_HELPER_PATH), *args], cwd=d,
                              capture_output=True, text=True)

    def test_upstream_tags_are_not_the_version(self):
        d, git = self._repo()
        git("commit", "-q", "--allow-empty", "-m", "upstream")
        git("tag", "v0.5.9")                                   # upstream: lightweight
        git("tag", "-a", "v0.5.10", "-m", "Release v0.5.10")   # upstream: annotated, other subject
        for _ in range(3):
            git("commit", "-q", "--allow-empty", "-m", "fork")
        sha = git("rev-parse", "--short", "HEAD")
        out = self._run(d)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), f"v{self.version_helper.NEXT_RELEASE}.dev0-3-g{sha}")
        self.assertNotEqual(self._run(d, "--tag-only").returncode, 0)

    def test_an_untagged_history_counts_every_commit(self):
        d, git = self._repo()
        for _ in range(2):
            git("commit", "-q", "--allow-empty", "-m", "c")
        sha = git("rev-parse", "--short", "HEAD")
        self.assertEqual(self._run(d).stdout.strip(), f"v0.1.0.dev0-2-g{sha}")

    def test_a_flliper_release_tag_is_the_version(self):
        d, git = self._repo()
        git("commit", "-q", "--allow-empty", "-m", "upstream")
        git("tag", "-a", "v0.5.10", "-m", "Release v0.5.10")
        git("commit", "-q", "--allow-empty", "-m", "release")
        git("tag", "-a", "v0.1.0", "-m", "fLLiper 0.1.0")
        self.assertEqual(self._run(d).stdout.strip(), "v0.1.0")
        self.assertEqual(self._run(d, "--tag-only").stdout.strip(), "v0.1.0")
        git("commit", "-q", "--allow-empty", "-m", "after")
        sha = git("rev-parse", "--short", "HEAD")
        self.assertEqual(self._run(d).stdout.strip(), f"v0.1.0-1-g{sha}")

    def test_setuptools_scm_reads_the_untagged_form_as_a_dev_release(self):
        try:
            from setuptools_scm import get_version
        except Exception:  # noqa: BLE001 - not in every venv (the build env has it)
            self.skipTest("setuptools_scm not importable here")
        d, git = self._repo()
        for _ in range(3):
            git("commit", "-q", "--allow-empty", "-m", "c")
        sha = git("rev-parse", "--short", "HEAD")
        v = get_version(root=d, git_describe_command=[sys.executable, str(VERSION_HELPER_PATH)])
        self.assertEqual(v, f"0.1.0.dev3+g{sha}")

if __name__ == "__main__":
    unittest.main()
