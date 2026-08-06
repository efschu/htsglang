"""Hermetic coverage for py-spy binary resolution in the crash/wedge handler.

Why this file exists. On 2026-08-06 a live 3-rank wedge was lost because the
automatic dump could not find py-spy:

    [TP0] Pyspy failed (py-spy dump  --pid 20955). Error: /bin/sh: 1: py-spy: not found
    [TP0] All pyspy dump attempts failed for PID 20955.

py-spy was installed the whole time, in the venv bin directory next to the
interpreter. The handler shelled out to a bare ``py-spy`` with ``shell=True``,
so the name was resolved against ``/bin/sh``'s PATH, and a server started as
``/path/to/venv/bin/python -m sglang.launch_server`` never puts that directory
on PATH. The failure is silent in the sense that matters: it costs you the
specimen, and you only find out afterwards.

These tests are all pure-Python: no CUDA, no subprocess is ever really
executed, no scheduler process is required. They run under
CUDA_VISIBLE_DEVICES=99.
"""

import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

from sglang.srt.utils.cudacore_pyspy_dump_utils import (
    pyspy_dump_schedulers,
    resolve_pyspy_binary,
)

MODULE = "sglang.srt.utils.cudacore_pyspy_dump_utils"


def _make_executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    """An interpreter in bindir/, and a PATH that does NOT contain bindir."""
    bindir = tmp_path / "venv" / "bin"
    bindir.mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    monkeypatch.setattr(f"{MODULE}.sys.executable", str(bindir / "python"))
    monkeypatch.setenv("PATH", str(elsewhere))
    monkeypatch.delenv("SGLANG_PYSPY_BIN", raising=False)
    return bindir, elsewhere


# --------------------------------------------------------------------------
# resolve_pyspy_binary
# --------------------------------------------------------------------------


def test_finds_pyspy_beside_the_interpreter_even_with_it_absent_from_path(fake_env):
    """The exact configuration that lost the specimen."""
    bindir, _ = fake_env
    expected = _make_executable(bindir / "py-spy")

    assert resolve_pyspy_binary() == str(expected)


def test_returns_none_when_absent_beside_interpreter_and_on_path(fake_env):
    assert resolve_pyspy_binary() is None


def test_falls_back_to_path_when_not_beside_the_interpreter(fake_env):
    _, elsewhere = fake_env
    on_path = _make_executable(elsewhere / "py-spy")

    assert resolve_pyspy_binary() == str(on_path)


def test_interpreter_bin_dir_wins_over_path(fake_env):
    bindir, elsewhere = fake_env
    beside = _make_executable(bindir / "py-spy")
    _make_executable(elsewhere / "py-spy")

    assert resolve_pyspy_binary() == str(beside)


def test_env_override_wins_over_everything(fake_env, monkeypatch):
    bindir, _ = fake_env
    _make_executable(bindir / "py-spy")
    monkeypatch.setenv("SGLANG_PYSPY_BIN", "/opt/custom/py-spy")

    assert resolve_pyspy_binary() == "/opt/custom/py-spy"


def test_non_executable_candidate_beside_interpreter_is_ignored(fake_env):
    """A stray non-executable file must not shadow a real one on PATH."""
    bindir, elsewhere = fake_env
    (bindir / "py-spy").write_text("not executable\n")
    (bindir / "py-spy").chmod(0o644)
    on_path = _make_executable(elsewhere / "py-spy")

    assert resolve_pyspy_binary() == str(on_path)


def test_a_directory_named_pyspy_is_not_mistaken_for_the_binary(fake_env):
    bindir, _ = fake_env
    (bindir / "py-spy").mkdir()

    assert resolve_pyspy_binary() is None


# --------------------------------------------------------------------------
# pyspy_dump_schedulers argv construction
# --------------------------------------------------------------------------


def test_dump_uses_absolute_path_and_never_an_empty_argument(fake_env):
    """The old code produced 'py-spy dump  --pid N' -- note the doubled space,
    the visible tell of an empty --native slot interpolated into an f-string.
    An argv list must never carry an empty element."""
    bindir, _ = fake_env
    pyspy = _make_executable(bindir / "py-spy")

    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="STACK", stderr="")

    with mock.patch(f"{MODULE}.subprocess.run", side_effect=fake_run):
        pyspy_dump_schedulers(scheduler_only=False)

    assert len(calls) == 1
    argv = calls[0]
    assert argv[0] == str(pyspy)
    assert os.path.isabs(argv[0])
    assert "" not in argv
    assert argv[1] == "dump"
    assert "--native" in argv
    assert "--pid" in argv


def test_dump_does_not_run_a_subprocess_when_the_binary_is_missing(fake_env):
    """No py-spy anywhere: report it, do not hand a bare name to the shell."""
    with mock.patch(f"{MODULE}.subprocess.run") as run:
        pyspy_dump_schedulers(scheduler_only=False)

    run.assert_not_called()


def test_dump_retries_without_native_then_reports_exhaustion(fake_env):
    bindir, _ = fake_env
    _make_executable(bindir / "py-spy")

    calls = []

    def always_fail(argv, **kwargs):
        calls.append(argv)
        raise subprocess.CalledProcessError(1, argv, stderr="nope")

    with mock.patch(f"{MODULE}.subprocess.run", side_effect=always_fail):
        pyspy_dump_schedulers(scheduler_only=False)

    assert len(calls) == 2
    assert "--native" in calls[0]
    assert "--native" not in calls[1]


def test_dump_stops_after_the_first_success(fake_env):
    bindir, _ = fake_env
    _make_executable(bindir / "py-spy")

    calls = []

    def ok(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="STACK", stderr="")

    with mock.patch(f"{MODULE}.subprocess.run", side_effect=ok):
        pyspy_dump_schedulers(scheduler_only=False)

    assert len(calls) == 1


def test_dump_survives_an_oserror_from_exec(fake_env):
    """A binary that resolves but cannot be exec'd must be reported, not raised."""
    bindir, _ = fake_env
    _make_executable(bindir / "py-spy")

    with mock.patch(f"{MODULE}.subprocess.run", side_effect=OSError("exec format")):
        pyspy_dump_schedulers(scheduler_only=False)  # must not raise


def test_dump_is_silent_about_schedulers_when_none_are_found(fake_env):
    bindir, _ = fake_env
    _make_executable(bindir / "py-spy")

    with mock.patch(f"{MODULE}.collect_scheduler_processes", return_value=[]):
        with mock.patch(f"{MODULE}.subprocess.run") as run:
            pyspy_dump_schedulers(scheduler_only=True)

    run.assert_not_called()


def test_resolution_happens_once_not_per_pid(fake_env):
    """Guard against reintroducing a per-pid PATH lookup in the hot crash path."""
    bindir, _ = fake_env
    _make_executable(bindir / "py-spy")

    fake_procs = [mock.Mock(pid=101), mock.Mock(pid=102), mock.Mock(pid=103)]

    with mock.patch(f"{MODULE}.collect_scheduler_processes", return_value=fake_procs):
        with mock.patch(f"{MODULE}.shutil.which") as which:
            with mock.patch(f"{MODULE}.subprocess.run") as run:
                run.return_value = subprocess.CompletedProcess([], 0, "S", "")
                pyspy_dump_schedulers(scheduler_only=True)

    assert run.call_count == 3
    which.assert_not_called()  # found beside the interpreter, no PATH search


# --------------------------------------------------------------------------
# The real environment this ships into
# --------------------------------------------------------------------------


def test_the_running_interpreter_can_actually_find_pyspy():
    """Not hermetic in the strict sense -- it reads the real venv -- but it is
    the check that would have caught the original defect on this rig, and it
    caught a second one: an early version of the fix called .resolve() on
    sys.executable, which follows the venv symlink to /usr/bin/python3.12 and
    lands in a directory that holds no venv console scripts. That version
    reproduced the exact bug it was meant to fix, and this test is what exposed
    it. Skipped rather than failed where py-spy is genuinely not installed."""
    beside = Path(sys.executable).parent / "py-spy"
    if not beside.is_file():
        pytest.skip("py-spy is not installed next to this interpreter")

    assert resolve_pyspy_binary() == str(beside)


def test_a_symlinked_venv_interpreter_still_finds_its_own_console_script(
    tmp_path, monkeypatch
):
    """The regression test for the .resolve() mistake, made hermetic.

    Layout: a venv whose bin/python is a symlink to a base interpreter living in
    a different directory. py-spy sits beside the SYMLINK, not beside its
    target. Resolving the symlink loses it.
    """
    base_bin = tmp_path / "usr" / "bin"
    base_bin.mkdir(parents=True)
    base_python = _make_executable(base_bin / "python3.12")

    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to(base_python)
    expected = _make_executable(venv_bin / "py-spy")

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(f"{MODULE}.sys.executable", str(venv_bin / "python"))
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.delenv("SGLANG_PYSPY_BIN", raising=False)

    assert resolve_pyspy_binary() == str(expected)


def test_resolved_interpreter_dir_is_still_searched_as_a_fallback(
    tmp_path, monkeypatch
):
    """Symmetric case: py-spy beside the symlink TARGET must still be found."""
    base_bin = tmp_path / "usr" / "bin"
    base_bin.mkdir(parents=True)
    base_python = _make_executable(base_bin / "python3.12")
    expected = _make_executable(base_bin / "py-spy")

    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to(base_python)

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(f"{MODULE}.sys.executable", str(venv_bin / "python"))
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.delenv("SGLANG_PYSPY_BIN", raising=False)

    assert resolve_pyspy_binary() == str(expected)
