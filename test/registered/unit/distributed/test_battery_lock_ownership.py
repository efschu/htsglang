# SPDX-License-Identifier: Apache-2.0
"""#314: battery_release_locks must be FILE-BASED, not process-local.

Before this fix, held-lock bookkeeping lived only in BATTERY_HELD_LOCKS /
BATTERY_HEARTBEAT_PID -- bash variables of the process that ran
battery_acquire_locks. A shell that did not run that acquire (a fresh
terminal, an agent that reconnects, a retry after the original process died)
had no way to find out what it was supposed to release: the lock directories
and the heartbeat orphaned. This happened on the real rig and was cleaned up
by hand.

The fix moves ownership onto DISK: the lock's own info file carries
`step=` and (new) `heartbeat_pid=`, and battery_release_locks reads them
back instead of trusting in-process state. These tests exercise that
end-to-end across an ACTUAL process boundary: acquire in one subprocess,
release in a completely different one, exactly the scenario that broke.

Hermetic and CPU-only:
  * a fake `nvidia-smi` on PATH reports a fixed, fake GPU count -- no real
    card, no dependency on what this shared box's real GPUs are doing.
  * BATTERY_LOCK_ROOT points every lock at an isolated tmp_path directory,
    never at the real /tmp/gpu-card-N.lock the rest of the rig arbitrates
    through. Two hermetic runs, or a hermetic run and a real battery step,
    can never collide.
"""

from __future__ import annotations

import os
import stat
import subprocess
import time

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")
)
BATTERY = os.path.join(REPO_ROOT, "scripts", "gpu_battery")
COMMON = os.path.join(BATTERY, "battery_common.sh")

#: Fake nvidia-smi that answers exactly what battery_common.sh's lock
#: functions ask of it (`-L`, counted by `grep -c '^GPU'`) with a FIXED,
#: fake card count -- never the real hardware's.
FAKE_NVIDIA_SMI = """#!/usr/bin/env bash
if [ "$1" = "-L" ]; then
    for i in $(seq 0 $(({n} - 1))); do
        echo "GPU $i: FakeCard$i (UUID: GPU-fake-$i)"
    done
    exit 0
fi
exit 0
"""


def _fake_path(tmp_path, n_gpus: int) -> str:
    bindir = tmp_path / "fakebin"
    bindir.mkdir(exist_ok=True)
    smi = bindir / "nvidia-smi"
    smi.write_text(FAKE_NVIDIA_SMI.format(n=n_gpus))
    smi.chmod(smi.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(bindir)


def _run_bash(script: str, tmp_path, n_gpus: int = 2, timeout: int = 30):
    fakebin = _fake_path(tmp_path, n_gpus)
    env = dict(os.environ)
    env["PATH"] = fakebin + os.pathsep + env.get("PATH", "")
    env["BATTERY_LOCK_ROOT"] = str(tmp_path / "locks")
    env.pop("BATTERY_RUN", None)
    os.makedirs(env["BATTERY_LOCK_ROOT"], exist_ok=True)
    full = f'set -uo pipefail\nsource "{COMMON}"\n{script}\n'
    proc = subprocess.run(
        ["bash", "-c", full],
        capture_output=True, text=True, timeout=timeout, env=env,
    )
    return proc


def _lock_dirs(tmp_path):
    root = tmp_path / "locks"
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.name.startswith("gpu-card-"))


def _heartbeat_pid_of(lock_dir) -> str:
    info = (lock_dir / "info").read_text()
    for line in info.splitlines():
        if line.startswith("heartbeat_pid="):
            return line.split("=", 1)[1].strip()
    return ""


def _pid_alive(pid: str) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True
    return True


class TestAcquireWritesHeartbeatPidToDisk:
    def test_every_lock_carries_its_own_heartbeat_pid(self, tmp_path):
        proc = _run_bash(
            'battery_acquire_locks "probe-tag" && echo ACQUIRED',
            tmp_path, n_gpus=2,
        )
        assert "ACQUIRED" in proc.stdout, proc.stderr
        locks = _lock_dirs(tmp_path)
        assert len(locks) == 2
        pids = set()
        for lock in locks:
            info = (lock / "info").read_text()
            assert "step=probe-tag" in info
            hb = _heartbeat_pid_of(lock)
            assert hb, f"no heartbeat_pid recorded in {lock}/info:\n{info}"
            pids.add(hb)
        # One heartbeat loop covers every lock the SAME acquire call took.
        assert len(pids) == 1
        assert _pid_alive(next(iter(pids)))
        # cleanup: this process's own heartbeat, not a fixture the test
        # framework will otherwise leak.
        os.kill(int(next(iter(pids))), 15)


class TestCrossProcessRelease:
    """Acquire in process A, release in process B -- the scenario #314 is
    named for. The two subprocess.run calls below are genuinely separate
    processes with no shared bash state; only the info file on disk
    connects them.
    """

    def test_release_from_a_different_process_removes_locks_and_heartbeat(
        self, tmp_path
    ):
        acquire = _run_bash(
            'battery_acquire_locks "cross-proc-tag" && echo ACQUIRED '
            '&& cat "$BATTERY_LOCK_ROOT"/gpu-card-0.lock/info',
            tmp_path, n_gpus=2,
        )
        assert "ACQUIRED" in acquire.stdout, acquire.stderr
        locks = _lock_dirs(tmp_path)
        assert len(locks) == 2
        hb_pid = _heartbeat_pid_of(locks[0])
        assert _pid_alive(hb_pid), "heartbeat should still be running"

        # A brand-new bash process, no relation to the one that acquired.
        # It knows only the step tag -- exactly what a fresh shell has.
        release = _run_bash(
            'battery_release_locks "cross-proc-tag" && echo RELEASED',
            tmp_path, n_gpus=2,
        )
        assert "RELEASED" in release.stdout, release.stderr

        deadline = time.time() + 5
        while time.time() < deadline and _pid_alive(hb_pid):
            time.sleep(0.1)
        assert not _pid_alive(hb_pid), (
            "heartbeat process is still alive after a cross-process release"
        )
        assert _lock_dirs(tmp_path) == [], "lock directories were not removed"

    def test_a_foreign_lock_with_a_different_step_tag_is_untouched(self, tmp_path):
        acquire = _run_bash(
            'battery_acquire_locks "mine" && echo ACQUIRED',
            tmp_path, n_gpus=1,
        )
        assert "ACQUIRED" in acquire.stdout, acquire.stderr
        (locks_root_lock,) = _lock_dirs(tmp_path)

        # Simulate a foreign holder by hand-writing a SECOND lock directory
        # with a different step tag -- the shared-box rule: a lock this
        # release call did not ask for is never broken.
        foreign = tmp_path / "locks" / "gpu-card-9.lock"
        foreign.mkdir()
        (foreign / "info").write_text(
            "holder=someone_else\nstep=foreign-tag\npid=999999\n"
            "acquired=2026-01-01T00:00:00\nheartbeat=2026-01-01T00:00:00\n"
            "heartbeat_pid=999999\n"
        )

        release = _run_bash(
            'battery_release_locks "mine" && echo RELEASED',
            tmp_path, n_gpus=1,
        )
        assert "RELEASED" in release.stdout, release.stderr

        remaining = _lock_dirs(tmp_path)
        assert remaining == [foreign], (
            f"release touched a lock it was not asked to release: {remaining}"
        )
        # And the foreign info file itself is exactly as written -- release
        # must not so much as inspect-and-rewrite it.
        assert "step=foreign-tag" in (foreign / "info").read_text()

    def test_release_with_no_step_tag_touches_nothing(self, tmp_path):
        """No identity, no action. Guards against the empty-step-matches-
        empty-info accident: an info file with no readable `step=` line must
        never be swept up just because the caller also passed nothing."""
        acquire = _run_bash(
            'battery_acquire_locks "held-tag" && echo ACQUIRED',
            tmp_path, n_gpus=1,
        )
        assert "ACQUIRED" in acquire.stdout, acquire.stderr

        release = _run_bash(
            "BATTERY_STEP= battery_release_locks && echo RELEASED",
            tmp_path, n_gpus=1,
        )
        assert "RELEASED" in release.stdout, release.stderr
        assert len(_lock_dirs(tmp_path)) == 1, (
            "a step-tag-less release must not remove any lock"
        )
        # cleanup for real, now that the test has proven the no-op.
        cleanup = _run_bash('battery_release_locks "held-tag"', tmp_path, n_gpus=1)
        assert cleanup.returncode == 0, cleanup.stderr

    def test_release_is_idempotent(self, tmp_path):
        acquire = _run_bash(
            'battery_acquire_locks "idempotent-tag" && echo ACQUIRED',
            tmp_path, n_gpus=1,
        )
        assert "ACQUIRED" in acquire.stdout, acquire.stderr

        first = _run_bash(
            'battery_release_locks "idempotent-tag" && echo RELEASED',
            tmp_path, n_gpus=1,
        )
        assert "RELEASED" in first.stdout, first.stderr
        assert _lock_dirs(tmp_path) == []

        # Same call again, nothing left to find -- must not error.
        second = _run_bash(
            'battery_release_locks "idempotent-tag" && echo RELEASED',
            tmp_path, n_gpus=1,
        )
        assert second.returncode == 0, second.stderr
        assert "RELEASED" in second.stdout
        assert _lock_dirs(tmp_path) == []

    def test_release_defaults_to_battery_step_env_var(self, tmp_path):
        """run_step.sh's own call sites pass no argument -- BATTERY_STEP,
        which it already exports before acquiring, is the default."""
        acquire = _run_bash(
            'battery_acquire_locks "env-default-tag" && echo ACQUIRED',
            tmp_path, n_gpus=1,
        )
        assert "ACQUIRED" in acquire.stdout, acquire.stderr

        release = _run_bash(
            'export BATTERY_STEP="env-default-tag"\n'
            "battery_release_locks && echo RELEASED",
            tmp_path, n_gpus=1,
        )
        assert "RELEASED" in release.stdout, release.stderr
        assert _lock_dirs(tmp_path) == []


class TestHeartbeatIdentityCheck:
    """The cmdline counter-check release does before killing a pid it read
    off disk: existence is not enough, a pid the OS already recycled to an
    unrelated process must never be killed just because a stale lock's info
    file still names it."""

    def test_a_live_pid_that_is_not_the_heartbeat_is_not_killed(self, tmp_path):
        acquire = _run_bash(
            'battery_acquire_locks "recycled-pid-tag" && echo ACQUIRED',
            tmp_path, n_gpus=1,
        )
        assert "ACQUIRED" in acquire.stdout, acquire.stderr
        (lock,) = _lock_dirs(tmp_path)
        real_hb_pid = _heartbeat_pid_of(lock)
        assert _pid_alive(real_hb_pid)
        # This process's OWN pid is unmistakably real, alive, and NOT the
        # heartbeat -- the sharpest stand-in for "the number got recycled".
        bystander = subprocess.Popen(["sleep", "30"])
        try:
            info = (lock / "info").read_text()
            info = info.replace(
                f"heartbeat_pid={real_hb_pid}", f"heartbeat_pid={bystander.pid}"
            )
            (lock / "info").write_text(info)

            release = _run_bash(
                'battery_release_locks "recycled-pid-tag" && echo RELEASED',
                tmp_path, n_gpus=1,
            )
            assert "RELEASED" in release.stdout, release.stderr
            assert _lock_dirs(tmp_path) == []
            # The bystander is untouched -- release must never kill a live
            # pid that is not demonstrably the heartbeat.
            assert bystander.poll() is None, (
                "release killed a live process that was never its heartbeat"
            )
        finally:
            bystander.kill()
            bystander.wait(timeout=5)
            # The REAL heartbeat was never told to die by the release above
            # (it targeted the bystander's now-stale pid instead) -- clean
            # it up so the test does not leak a process.
            if _pid_alive(real_hb_pid):
                os.kill(int(real_hb_pid), 15)
