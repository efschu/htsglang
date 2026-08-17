# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#407 cut 6: --hibernate-dir must resolve to a PERSISTENT tier.

THE HOLE, and it is a correctness hole rather than an optimisation.
`--enable-weights-disk-backup` promises that weights are parked on disk and can
be restored after a restart. `server_args` validates that the flag and
`--hibernate-dir` are passed together and that the checkpoint is GGUF, and then
accepts **any directory** -- including one on a tmpfs. Weights are written, the
run reports a successful park, and the backup is gone the moment the box
reboots. Nothing tells the operator, because from the process's point of view
every write succeeded.

`hibernate.py:432` even takes an `flock` on that directory to serialise
co-located ranks, which works perfectly on a tmpfs and so proves nothing about
durability.

THE REGISTRY ALREADY MODELS THIS, which is why cut 6 is a wiring cut and not a
new mechanism. `memtier/bootstrap.py` carries `NON_PERSISTENT_FS_TYPES`
(`tmpfs`, `ramfs`, `devtmpfs`), derives `persistent` from the fs type, and
`collect_fs_types` resolves a directory to its mount by LONGEST match -- its
docstring names `--hibernate-dir /dev/shm/img` as precisely the case that
matters, because that path is on a tmpfs while not being a mount point itself.
The design's cut table lists this cut as unblocked ("prerequisite shipped in
1b") and as a correctness fix that could be pulled forward at any point.

WHY UNKNOWN IS A WARNING AND NOT A REFUSAL. `bootstrap` treats an unresolvable
fs type as not-persistent, which is the right default for RANKING a tier: an
unknown medium should not win a placement. It is the wrong default for a
REFUSAL, because a container without a readable `/proc/mounts` would then be
unable to hibernate at all -- a working configuration broken by a gate that
cannot see. So this refuses only what it can positively identify as volatile,
and says so when it cannot tell.
"""

from __future__ import annotations

import os
import tempfile
import unittest

from sglang.srt.memtier.bootstrap import NON_PERSISTENT_FS_TYPES
from sglang.srt.memtier.hibernate_tier import (
    HibernateDirNotPersistent,
    hibernate_dir_verdict,
    refuse_volatile_hibernate_dir,
)
from sglang.test.test_utils import CustomTestCase


class TestTheVerdict(CustomTestCase):
    def test_a_tmpfs_directory_is_volatile(self):
        verdict = hibernate_dir_verdict("/dev/shm", fs_types={"/dev/shm": "tmpfs"})
        self.assertFalse(verdict.persistent)
        self.assertEqual(verdict.fs_type, "tmpfs")
        self.assertTrue(verdict.known)

    def test_a_subdirectory_inherits_its_mount(self):
        """The case the registry docstring names: not itself a mount point."""
        verdict = hibernate_dir_verdict(
            "/dev/shm/img", fs_types={"/dev/shm/img": "tmpfs"}
        )
        self.assertFalse(verdict.persistent)

    def test_a_real_filesystem_is_persistent(self):
        for fs in ("ext4", "xfs", "zfs", "btrfs"):
            with self.subTest(fs=fs):
                verdict = hibernate_dir_verdict("/var/hib", fs_types={"/var/hib": fs})
                self.assertTrue(verdict.persistent, fs)

    def test_every_declared_volatile_type_is_caught(self):
        for fs in NON_PERSISTENT_FS_TYPES:
            with self.subTest(fs=fs):
                self.assertFalse(
                    hibernate_dir_verdict("/x", fs_types={"/x": fs}).persistent
                )

    def test_an_unresolvable_mount_is_unknown_not_volatile(self):
        verdict = hibernate_dir_verdict("/x", fs_types={})
        self.assertFalse(verdict.known)
        # Not claimed persistent, but not claimed volatile either.
        self.assertIsNone(verdict.fs_type)


class TestTheRefusal(CustomTestCase):
    def test_a_tmpfs_dir_is_refused_by_name(self):
        with self.assertRaises(HibernateDirNotPersistent) as caught:
            refuse_volatile_hibernate_dir("/dev/shm/img", fs_types={"/dev/shm/img": "tmpfs"})
        message = str(caught.exception)
        self.assertIn("/dev/shm/img", message)
        self.assertIn("tmpfs", message)
        # The refusal must say what the operator loses, not just that it said no.
        self.assertIn("reboot", message.lower())

    def test_a_persistent_dir_passes(self):
        self.assertIsNone(
            refuse_volatile_hibernate_dir("/var/hib", fs_types={"/var/hib": "ext4"})
        )

    def test_an_unknown_mount_does_not_refuse(self):
        # A container without a readable /proc/mounts must still be able to
        # hibernate; a gate that cannot see must not veto.
        self.assertIsNone(refuse_volatile_hibernate_dir("/x", fs_types={}))

    def test_it_reads_real_mounts_when_not_injected(self):
        # The real path, exercised once so the injection is not the only shape
        # this function is ever run in.
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(refuse_volatile_hibernate_dir(tmp))

    def test_dev_shm_is_refused_for_real(self):
        """No injection: /dev/shm is a tmpfs on any normal Linux box."""
        if not os.path.isdir("/dev/shm"):
            self.skipTest("no /dev/shm on this host")
        with self.assertRaises(HibernateDirNotPersistent):
            refuse_volatile_hibernate_dir("/dev/shm")


class TestServerArgsCallsIt(CustomTestCase):
    """A gate nobody calls is the defect this whole family is about."""

    def test_the_hibernate_validation_refuses_a_volatile_dir(self):
        import inspect

        from sglang.srt.server_args import ServerArgs

        src = inspect.getsource(ServerArgs)
        self.assertIn("refuse_volatile_hibernate_dir", src)


if __name__ == "__main__":
    unittest.main()
