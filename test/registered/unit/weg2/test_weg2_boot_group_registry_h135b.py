# SPDX-License-Identifier: Apache-2.0
"""H135b: the boot groups (P, D, front) are registered for the arm's owner guard.

MEASURED 26.09.: launcher, front, P and D run in their own sessions and outlive
an arm killed by SIGKILL / OOM / `timeout -k` -- a headless boot holding the
cards past its gpuq window. The arm's guard (boot_helpers.sh) needs the groups
by identity: ``{EVIDENCE_DIR}/helpers_{tag}/{name}.grp``, the SAME five-field
line as a helper's ``.pid`` entry, under its own suffix so that the helper
paths (bh_stop_all, bh_sweep_stale) never stop a serving group.

Pinned here:
  * the ``.grp`` entry is the boot_helpers.sh line and never a ``.pid`` entry;
  * the helper default (``.pid``, message WEG2-HELPER) is unchanged;
  * teardown() removes the ``.grp`` entries of every boot group (mutant: drop
    the loop -> red) and leaves an unrelated tag alone.
"""

import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.registry import nvml as nvml_registry
from sglang.srt.weg2 import launcher as L
from sglang.test.test_utils import CustomTestCase


def _state(tag):
    st = L.BootState(tag=tag, tip="deadbeef", tree="/nonexistent-h135b", stamp="2026-09-26T00:00:00Z")
    st.pids = {}
    st.helper_pids = []
    st.ring_dir = ""
    st.admin_key_file = ""
    return st


class _Tmp(CustomTestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="weg2-h135b-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        for attr in ("GPU_ARB", "EVIDENCE_DIR"):
            p = mock.patch.object(L, attr, self.tmp)
            p.start()
            self.addCleanup(p.stop)


class TestBootGroupEntry(_Tmp):
    def test_grp_entry_is_the_boot_helpers_line(self):
        msg = L.register_boot_group("h135bf", "front", os.getpid())
        d = f"{self.tmp}/helpers_h135bf"
        self.assertEqual(sorted(os.listdir(d)), ["front.grp"], "a boot group is never a .pid helper entry")
        with open(f"{d}/front.grp") as f:
            pgid, start, owner, ostart, name = f.read().split()
        self.assertEqual(int(pgid), os.getpid())
        self.assertEqual(int(start), L._proc_start_ticks(os.getpid()))
        self.assertEqual(int(owner), os.getppid())
        self.assertEqual(int(ostart), L._proc_start_ticks(os.getppid()))
        self.assertEqual(name, "front")
        self.assertIn("WEG2-BOOT-GROUP registered front", msg)

    def test_helper_default_unchanged(self):
        msg = L.register_helper("h135bh", "deadman_front", os.getpid())
        self.assertTrue(os.path.exists(f"{self.tmp}/helpers_h135bh/deadman_front.pid"))
        self.assertIn("WEG2-HELPER registered deadman_front", msg)

    def test_names_cover_every_group_the_guard_stops(self):
        self.assertEqual(set(L.BOOT_GROUP_NAMES), {"launcher", "P", "D", "front"})


class TestTeardownRemovesGroupEntries(_Tmp):
    def test_grp_entries_removed_other_tag_kept(self):
        st = _state("h135bt")
        path = L.state_path(st)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(st.__dict__, f, default=str)
        for name in ("P", "D", "front"):
            L.register_boot_group(st.tag, name, os.getpid())
        # the arm writes the launcher entry itself (bash bh_register) -- same file name
        with open(f"{self.tmp}/helpers_{st.tag}/launcher.grp", "w") as f:
            f.write(f"{os.getpid()} 1 1 1 launcher\n")
        L.register_boot_group("h135bother", "P", os.getpid())
        buf = io.StringIO()
        with (
            mock.patch("os.killpg"),
            mock.patch("os.kill"),
            mock.patch("time.sleep"),
            mock.patch("subprocess.run", return_value=mock.Mock(stdout="")),
            mock.patch.object(L, "_alive", return_value=False),
            mock.patch.object(nvml_registry, "memory_snapshot", return_value=[]),
            contextlib.redirect_stdout(buf),
        ):
            L.teardown(path)
        left = [n for n in os.listdir(f"{self.tmp}/helpers_{st.tag}") if n.endswith(".grp")] \
            if os.path.isdir(f"{self.tmp}/helpers_{st.tag}") else []
        self.assertEqual(left, [])
        self.assertTrue(os.path.exists(f"{self.tmp}/helpers_h135bother/P.grp"))


if __name__ == "__main__":
    unittest.main()
