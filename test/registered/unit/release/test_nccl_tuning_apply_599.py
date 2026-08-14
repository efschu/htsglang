"""#599: the NCCL tuning package must be applied by refusal, and must read back.

WHAT THIS PINS, AND WHY IT IS NOT A FORMALITY.

`deploy/release/nccl-tuning.env` is config plus evidence, not a file to source.
Almost everything in it is either [AUTO] -- the fork sets it at runtime from
the actual rank layout -- or [RIG] -- measured on a machine with no P2P and no
NVLink, where exporting it for a user who HAS NVLink is a performance
regression rather than a fix. The number of environment variables the package
is willing to ship to a stranger is therefore ZERO, and
`test_env_command_emits_no_assignments` pins exactly that. If someone later
"improves" the package by exporting a tuned-looking default, that test goes red
and they have to justify it with a measurement.

The second thing pinned is the read-back rule. Exporting a variable proves the
shell exported it; it proves nothing about whether NCCL honoured it. NCCL
ignores unknown names silently, and a value set after ncclCommInitRank never
takes effect at all. The only authority on what NCCL actually consumed is NCCL
itself, under NCCL_DEBUG=INFO:

    host:1:1 [0] NCCL INFO NCCL_MAX_CTAS set by environment to 4

`verify-log` parses those lines, and the fixtures below feed it logs that are
wrong in specific ways to show it notices.

The script also carries its own `selftest`, which drives every check into its
red state -- including mounting a real 64 MB tmpfs, Docker's actual default, in
a private mount namespace, because a plain temp directory inherits the host
filesystem's size and would not exercise the /dev/shm floor at all. A check
that has never been observed failing is not a check, so
`test_selftest_shows_every_check_going_red` runs it here rather than trusting
that someone ran it by hand.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "deploy" / "release" / "apply-nccl-tuning.sh"


def run(*args, env_extra=None, expect=None):
    """Invoke the script with a deliberately clean environment.

    The ambient environment of a test runner may legitimately carry NCCL_*
    variables; inheriting them would make these assertions depend on the
    machine the suite happens to run on.
    """
    env = {
        k: v for k, v in os.environ.items() if not k.startswith("NCCL_")
    }
    env.update(env_extra or {})
    proc = subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    if expect is not None:
        assert proc.returncode == expect, (
            f"expected exit {expect}, got {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


def test_script_exists_and_is_executable():
    assert SCRIPT.is_file(), f"{SCRIPT} is missing"
    assert os.access(SCRIPT, os.X_OK), f"{SCRIPT} is not executable"


def test_script_is_syntactically_valid():
    run_syntax = subprocess.run(
        ["bash", "-n", str(SCRIPT)], capture_output=True, text=True
    )
    assert run_syntax.returncode == 0, run_syntax.stderr


def test_env_command_emits_no_assignments():
    """The package ships no environment variables. That is a result, not a gap.

    Every candidate was [AUTO], [RIG], or unmeasured. If this goes red, someone
    added an export -- it needs a measurement behind it, not a plausible name.
    """
    out = run("env", expect=0).stdout
    assignment_lines = [
        ln
        for ln in out.splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    assert assignment_lines == [], (
        "the NCCL package emitted environment assignments:\n"
        + "\n".join(assignment_lines)
    )


def test_run_flags_carry_the_one_universal_requirement():
    """Docker's 64 MB /dev/shm makes NCCL abort at ncclGroupEnd()."""
    out = run("run-flags", expect=0).stdout
    assert "--ipc=host" in out
    assert "--shm-size=4g" in out


def test_auto_variable_set_by_hand_is_refused():
    """[AUTO] variables lose to, or fight with, the value the fork computes."""
    proc = run("check", env_extra={"NCCL_MAX_CTAS": "4"}, expect=1)
    assert "NCCL_MAX_CTAS" in proc.stdout
    assert "[AUTO]" in proc.stdout


def test_rig_variable_needs_explicit_consent():
    """[RIG] settings are conditioned on this rig having no P2P and no NVLink."""
    refused = run("check", env_extra={"NCCL_P2P_DISABLE": "1"}, expect=1)
    assert "NCCL_P2P_DISABLE" in refused.stdout

    # ...and the consent flag must actually work, or the check is just a wall.
    run(
        "check",
        "--allow-rig-conditioned",
        env_extra={"NCCL_P2P_DISABLE": "1"},
        expect=0,
    )


def test_unbacked_upstream_variable_is_refused():
    """Re-shipping an upstream recipe as a fork finding is the #251 defect."""
    proc = run("check", env_extra={"NCCL_IB_TC": "136"}, expect=1)
    assert "NCCL_IB_TC" in proc.stdout


def test_verify_log_reads_back_what_nccl_actually_consumed(tmp_path):
    log = tmp_path / "rig.log"
    log.write_text(
        "host:1:1 [0] NCCL INFO NCCL_P2P_DISABLE set by environment to 1\n"
    )
    proc = run("verify-log", str(log), expect=1)
    assert "NCCL_P2P_DISABLE" in proc.stdout


def test_verify_log_catches_a_version_below_the_floor(tmp_path):
    """2.28.9 REJECTS a co-located communicator; 2.30.7 builds it and serves."""
    log = tmp_path / "old.log"
    log.write_text("host:1:1 [0] NCCL INFO NCCL version 2.28.9+cuda13.0\n")
    proc = run("verify-log", str(log), expect=1)
    assert "2.28.9" in proc.stdout


def test_a_silent_log_is_a_skip_and_strict_escalates_it(tmp_path):
    """An unrunnable check must not read as a passing one."""
    log = tmp_path / "quiet.log"
    log.write_text("nothing to see here\n")

    lenient = run("verify-log", str(log), expect=0)
    assert "SKIP" in lenient.stdout

    run("verify-log", str(log), "--strict", expect=1)


@pytest.mark.skipif(
    shutil.which("unshare") is None,
    reason="selftest's /dev/shm case needs a private mount namespace",
)
def test_selftest_shows_every_check_going_red():
    """The script's own falsifier. Exit 0 means every case behaved as required."""
    proc = run("selftest", expect=0)
    assert "cases behaved as required" in proc.stdout
    assert "GREEN -- check is broken" not in proc.stdout
    assert "NOT EXERCISED" not in proc.stdout
