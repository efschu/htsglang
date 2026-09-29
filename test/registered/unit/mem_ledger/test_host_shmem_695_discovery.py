# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""#695: the metal recipe could never find a rank.

``host_shmem_695.py`` discovers ranks by reading ``/proc/<pid>/comm`` and
testing ``comm.startswith("flliper::scheduler")``. The kernel caps ``comm`` at
``TASK_COMM_LEN - 1`` = **15 characters**. ``"flliper::scheduler"`` is 17. So
the kernel stores ``"flliper::schedul"`` and the prefix test is False for every
process, always -- auto-discovery returned an empty list on a fully healthy
three-rank boot and the script exited with "no flliper::scheduler process
found. Boot the server first".

Measured on the live 2026-08-12 PP=3 instance (pids 2641744/5/6, one per PP
rank)::

    $ cat /proc/2641744/comm
    flliper::schedul          # 15 chars, not 17

This is the class of defect that only a metal run finds, and it would have
consumed the first GPU window the recipe was used in. The desk half of the
branch was hermetic and never called ``discover_scheduler_pids`` against a
real ``comm``.

The tests below pin the truncated form specifically. Asserting against the
UNtruncated string is what let this through, so the fixture values here are
literal 15-character kernel output, not the name the process asked for.
"""

import importlib.util
import os

from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_HERE = os.path.abspath(__file__)
_REPO_ROOT = _HERE
for _ in range(5):
    _REPO_ROOT = os.path.dirname(_REPO_ROOT)
_SCRIPT = os.path.join(_REPO_ROOT, "scripts", "vram_ledger", "host_shmem_695.py")


def _load():
    spec = importlib.util.spec_from_file_location("host_shmem_695", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestRankDiscoveryMatchesTheKernelsTruncatedComm(CustomTestCase):
    #: Exactly what the kernel stores for a process that asked to be called
    #: "flliper::scheduler_PP0". Copied from /proc on the live instance.
    KERNEL_COMM = "flliper::schedul"

    def setUp(self):
        self.mod = _load()

    def test_the_kernel_truncates_below_the_string_the_code_tested_for(self):
        """The premise, pinned. If TASK_COMM_LEN ever grows this test says so
        before the discovery test starts looking mysterious."""
        self.assertEqual(len(self.KERNEL_COMM), 15)
        self.assertFalse(self.KERNEL_COMM.startswith("flliper::scheduler"))

    def test_a_real_rank_comm_is_recognised(self):
        """THE fix test. Red before the fix: returns []."""
        self.assertTrue(
            self.mod.is_scheduler_comm(self.KERNEL_COMM),
            "a live rank's actual comm was not recognised as a scheduler")

    def test_discovery_finds_ranks_whose_comm_is_truncated(self, ):
        """End to end through the discovery function, with /proc faked."""
        procs = {101: self.KERNEL_COMM, 102: self.KERNEL_COMM,
                 103: self.KERNEL_COMM, 104: "flliper::detoken",
                 105: "python3", 106: "bash"}
        found = self.mod.discover_scheduler_pids(read_comm=procs.get,
                                                 pids=list(procs))
        self.assertEqual(sorted(found), [101, 102, 103])

    def test_the_detokenizer_is_not_a_rank(self):
        """``flliper::detoken`` is a sibling child of the same launcher and
        holds no flip images. Counting it would inflate the census."""
        self.assertFalse(self.mod.is_scheduler_comm("flliper::detoken"))

    def test_an_untruncated_comm_would_still_match(self):
        """Not every kernel/namespace need truncate identically, and the
        fix must not trade one exact-string assumption for another."""
        self.assertTrue(self.mod.is_scheduler_comm("flliper::scheduler_PP0"))

    def test_unrelated_processes_are_not_ranks(self):
        for comm in ("python", "bash", "flliper::router", "sgl", ""):
            self.assertFalse(self.mod.is_scheduler_comm(comm), comm)


if __name__ == "__main__":
    import unittest

    unittest.main()
