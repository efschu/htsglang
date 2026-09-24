# SPDX-License-Identifier: Apache-2.0
"""fnFL2 H26: the #1217 residue guard no longer catches its own start.

Metal 24.09. 08:56-08:57Z, boot fnFL2x140: three starts refused with
``#1217 shm residue: LIVE launch_server pid(s)=[3122930]`` (then 3157795,
3189930), each pid gone seconds later.  ``pgrep -f "sglang[.]launch_server"``
matched the operator's own Bash-tool shell -- ``bash -c '... pgrep -fc
"sglang.launch_server"; bash start_when_free.sh fnFL2x139 boot_nf_x140.sh
fnFL2x140'`` -- the ancestor of the launcher doing the check.  It only ran
because two dead 16 KB ``sglang_loads_*`` flags of boot weg2xsn414 sat in
/dev/shm.

Hermetic: fake ``/proc`` and ``/dev/shm`` trees per test, no pgrep, no GPU.
"""

import os
import tempfile
import unittest

from sglang.srt.weg2 import launcher
from sglang.test.test_utils import CustomTestCase

SELF = 5000
TAG = "fnFL2x140"

# The measured chain, outermost first: Claude Bash-tool shell -> start wrapper
# -> boot script -> arm -> timeout -> launcher (dry-run).
TOOL_SHELL = [
    "/bin/bash", "-c",
    'nvidia-smi --query-gpu=memory.used --format=csv,noheader; pgrep -fc "sglang.launch_server"; '
    "cd /root/.claude/jobs/aef87d47/tmp && bash start_when_free.sh fnFL2x139 boot_nf_x140.sh fnFL2x140",
]
LAUNCHER_ARGV = [
    "/spinning/htsglang-gpu/.venv/bin/python", "-m", "sglang.srt.weg2.launcher",
    "--tree", "/x", "--tag", TAG, "--extra-d", "--port 30032", "--dry-run",
]
SERVER_ARGV = [
    "/spinning/htsglang-gpu/.venv/bin/python", "-m", "sglang.launch_server",
    "--model-path", "/m", "--port", "30000",
]


class _Log:
    def __init__(self):
        self.lines = []

    def __call__(self, msg):
        self.lines.append(msg)


def _proc(root, pid, ppid, argv, env=None, nvidia=False):
    d = os.path.join(root, str(pid))
    os.makedirs(os.path.join(d, "fd"), exist_ok=True)
    with open(os.path.join(d, "cmdline"), "wb") as f:
        f.write(b"\0".join(a.encode() for a in argv) + b"\0")
    with open(os.path.join(d, "stat"), "w") as f:
        f.write(f"{pid} (some (odd) comm) S {ppid} {pid} {pid} 0 -1\n")
    with open(os.path.join(d, "environ"), "wb") as f:
        f.write(b"\0".join(f"{k}={v}".encode() for k, v in (env or {}).items()) + b"\0")
    with open(os.path.join(d, "maps"), "w") as f:
        f.write("7f0000000000-7f0000001000 rw-p 00000000 00:00 0 \n")
    if nvidia:
        os.symlink("/dev/nvidiactl", os.path.join(d, "fd", "7"))


def _self_chain(root):
    """The launcher (SELF) under the measured chain of ancestors."""
    _proc(root, 4000, 1, TOOL_SHELL)
    _proc(root, 4001, 4000, ["bash", "start_when_free.sh", "fnFL2x139", "boot_nf_x140.sh", "fnFL2x140"])
    _proc(root, 4002, 4001, ["bash", "boot_nf_x140.sh", "fnFL2x140"])
    _proc(root, 4003, 4002, ["bash", "/spinning/gpu-arb/weg2/arm_fnFL2_long.sh", "fnFL2x140", "--go"])
    _proc(root, 4004, 4003, ["timeout", "900", LAUNCHER_ARGV[0]] + LAUNCHER_ARGV[1:])
    _proc(root, SELF, 4004, LAUNCHER_ARGV)


def _shm_with_dead_flags(tmp):
    shm = os.path.join(tmp, "shm")
    os.makedirs(shm)
    for n in ("sglang_loads_734260e231b8_e876e294.shm", "sglang_loads_d7bb0de288f8_55645a8b.shm"):
        with open(os.path.join(shm, n), "wb") as f:
            f.write(b"z" * 16396)
    with open(os.path.join(shm, "sem.mp-foreign"), "wb") as f:
        f.write(b"y")
    return shm


class TestServerArgv(CustomTestCase):
    def test_only_a_python_dash_m_server_is_a_server(self):
        self.assertTrue(launcher.is_launch_server_argv(SERVER_ARGV))
        self.assertTrue(launcher.is_launch_server_argv(["python3", "-msglang.launch_server"]))
        self.assertFalse(launcher.is_launch_server_argv(TOOL_SHELL))
        self.assertFalse(launcher.is_launch_server_argv(LAUNCHER_ARGV))
        # the module as a piece of a longer argument is not the module
        self.assertFalse(launcher.is_launch_server_argv(["python", "-c", "import sglang.launch_server"]))
        self.assertFalse(launcher.is_launch_server_argv(["python", "-m", "sglang.launch_server_x"]))
        # a shell that runs the server is not the server
        self.assertFalse(launcher.is_launch_server_argv(["bash", "-c", "python -m sglang.launch_server"]))


class TestLiveLaunchServers(CustomTestCase):
    def test_the_measured_self_catch_is_not_a_live_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = os.path.join(tmp, "proc")
            _self_chain(proc)
            # a sibling shell polling with the same pattern, too
            _proc(proc, 4100, 4003, ["bash", "-c", "ps -eo args | grep -c 'sglang.launch_server'"])
            live, ignored = launcher.live_launch_servers(TAG, proc, self_pid=SELF)
            self.assertEqual(live, [])
            why = {r["pid"]: r["why"] for r in ignored}
            self.assertEqual(why[4000], "self/ancestor")
            self.assertEqual(why[4100], "not a server argv")

    def test_a_foreign_server_with_a_cuda_context_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = os.path.join(tmp, "proc")
            _self_chain(proc)
            _proc(proc, 7000, 1, SERVER_ARGV, env={"SGLANG_WEG2_BOOT_TOKEN": "weg2xsn414:1:2"}, nvidia=True)
            live, _ = launcher.live_launch_servers(TAG, proc, self_pid=SELF)
            self.assertEqual([r["pid"] for r in live], [7000])
            self.assertTrue(live[0]["cuda"])
            self.assertEqual(live[0]["tag"], "weg2xsn414")

    def test_a_stock_server_still_initialising_counts_as_foreign(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = os.path.join(tmp, "proc")
            _proc(proc, 7001, 1, SERVER_ARGV)  # no token, no CUDA yet
            live, _ = launcher.live_launch_servers(TAG, proc, self_pid=SELF)
            self.assertEqual([r["pid"] for r in live], [7001])
            self.assertIsNone(live[0]["tag"])

    def test_an_own_tag_server_without_cuda_is_named_not_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = os.path.join(tmp, "proc")
            _proc(proc, 7002, 1, SERVER_ARGV, env={"SGLANG_WEG2_BOOT_TOKEN": f"{TAG}:1:2"})
            live, ignored = launcher.live_launch_servers(TAG, proc, self_pid=SELF)
            self.assertEqual(live, [])
            self.assertEqual(ignored[0]["why"], "own tag, no CUDA context")

    def test_an_own_tag_server_with_cuda_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = os.path.join(tmp, "proc")
            _proc(proc, 7003, 1, SERVER_ARGV, env={"SGLANG_WEG2_BOOT_TOKEN": f"{TAG}:1:2"}, nvidia=True)
            live, _ = launcher.live_launch_servers(TAG, proc, self_pid=SELF)
            self.assertEqual([r["pid"] for r in live], [7003])


class TestTheSweepWithTheMeasuredBox(CustomTestCase):
    def _sweep(self, tmp, proc, dry=False):
        log = _Log()
        orig = os.getpid
        # the sweep asks os.getpid(); the fake chain's launcher is SELF
        launcher.os.getpid = lambda: SELF
        try:
            out = launcher.shm_residue_sweep(
                log, TAG, "0924_085631", dry, shm_dir=_shm_with_dead_flags(tmp), proc_root=proc,
                archive_root=os.path.join(tmp, "archive"),
            )
        finally:
            launcher.os.getpid = orig
        return out, log

    def test_self_catch_passes_and_the_holderless_rest_is_archived(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = os.path.join(tmp, "proc")
            _self_chain(proc)
            out, log = self._sweep(tmp, proc)
            self.assertEqual(
                sorted(out["swept"]),
                ["sglang_loads_734260e231b8_e876e294.shm", "sglang_loads_d7bb0de288f8_55645a8b.shm"],
            )
            archived = sorted(os.listdir(out["archive"]))
            self.assertIn("MANIFEST.json", archived)
            self.assertIn("sglang_loads_734260e231b8_e876e294.shm", archived)
            self.assertFalse(os.path.exists(os.path.join(tmp, "shm", "sglang_loads_734260e231b8_e876e294.shm")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "shm", "sem.mp-foreign")))
            self.assertTrue(any("self/ancestor" in ln for ln in log.lines))

    def test_a_real_foreign_server_still_refuses_and_nothing_moves(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = os.path.join(tmp, "proc")
            _self_chain(proc)
            _proc(proc, 7000, 1, SERVER_ARGV, env={"SGLANG_WEG2_BOOT_TOKEN": "weg2xsn414:1:2"}, nvidia=True)
            with self.assertRaises(launcher.Weg2LaunchRefused) as cm:
                self._sweep(tmp, proc)
            msg = str(cm.exception)
            self.assertIn("LIVE launch_server pid(s)=[7000]", msg)
            self.assertNotIn("4000", msg)
            self.assertTrue(os.path.exists(os.path.join(tmp, "shm", "sglang_loads_734260e231b8_e876e294.shm")))

    def test_a_live_holder_of_our_rest_still_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = os.path.join(tmp, "proc")
            _self_chain(proc)
            shm_file = os.path.join(tmp, "shm", "sglang_loads_734260e231b8_e876e294.shm")
            _proc(proc, 7100, 1, ["python3", "worker.py"])
            with open(os.path.join(proc, "7100", "maps"), "w") as f:
                f.write(f"7f0000000000-7f0000001000 rw-s 00000000 00:19 1 {shm_file}\n")
            with self.assertRaises(launcher.Weg2LaunchRefused) as cm:
                self._sweep(tmp, proc)
            self.assertIn("LIVE HOLDER", str(cm.exception))
            self.assertIn("7100", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
