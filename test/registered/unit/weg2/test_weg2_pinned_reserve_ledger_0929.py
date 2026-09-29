# SPDX-License-Identifier: Apache-2.0
"""One pinned-host reserve for the host ledger and the ranks (z30x2-yarn2, 29.09.).

27B's three conditions, model-neutral (the HiCache pool sites are shared):
  1. an explicit SGLANG_PINNED_HOST_RESERVE_GIB always wins;
  2. otherwise, under a finite cgroup memory.max, the reserve is the ledger's
     own MEASURED margin (flip transient + run residual + idle drift from the
     records), exported to the ranks -- no flat constant; no cgroup = native;
  3. the ledger bounds its run peak by exactly the wall the pools enforce
     (memory.max - that reserve): ledger green => every pool admitted, and the
     yarn2 gap (ledger green, pool refused) cannot recur.
"""
from __future__ import annotations

import inspect
import os
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.mem_cache import pinned_host_budget as phb
from sglang.srt.weg2 import host_ledger as hl
from sglang.srt.weg2 import launcher
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

GIB = int(hl.GIB)
CAP = 84 * GIB
CG = "cgroup memory.max"
POOLS = int(0.67e9)
MARGIN = 1.47                          # the measured margin on the z30x2 ARM lines
ENVS = (phb.PINNED_HOST_RESERVE_ENV, phb.PINNED_HOST_RESERVE_LEDGER_ENV,
        phb.PINNED_HOST_RESERVE_LEDGER_SOURCE_ENV)


class _CleanEnv(CustomTestCase):
    def setUp(self):
        self._saved = {k: os.environ.pop(k, None) for k in ENVS}
        phb.clear_registered_posts()

    def tearDown(self):
        phb.clear_registered_posts()
        for k, v in self._saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def _export(self, gib, src):
        # what launcher.main does after the ledger chose
        os.environ[phb.PINNED_HOST_RESERVE_LEDGER_ENV] = f"{float(gib):.4f}"
        os.environ[phb.PINNED_HOST_RESERVE_LEDGER_SOURCE_ENV] = src


class TheExplicitEnvWins(_CleanEnv):
    def test_env_beats_the_ledger_on_both_sides(self):
        os.environ[phb.PINNED_HOST_RESERVE_ENV] = "2"
        gib, src = hl.pinned_reserve_for_ranks(MARGIN, CAP, CG)
        self.assertEqual(gib, 2.0)
        self.assertIn("SGLANG_PINNED_HOST_RESERVE_GIB=2", src)
        self._export(MARGIN, "ledger margin")       # even if something exported
        b, rsrc = phb.pinned_host_reserve()
        self.assertEqual(b, 2 * GIB)
        self.assertIn("env SGLANG_PINNED_HOST_RESERVE_GIB=2", rsrc)


class TheDefaultIsTheMeasuredMargin(_CleanEnv):
    def test_cgroup_without_env_takes_the_margin(self):
        gib, src = hl.pinned_reserve_for_ranks(MARGIN, CAP, CG)
        self.assertEqual(gib, MARGIN)
        self.assertIn("ledger margin 1.47", src)

    def test_no_cgroup_keeps_the_native_reserve_byte_identical(self):
        self.assertIsNone(hl.pinned_reserve_for_ranks(MARGIN, None, "")[0])
        self.assertIsNone(hl.pinned_reserve_for_ranks(MARGIN, 118 * GIB, "lxcfs MemTotal FALLBACK")[0])
        b, src = phb.pinned_host_reserve()
        self.assertEqual((b, src), (10 * (1 << 30), "default 10 GiB"))

    def test_the_ranks_read_back_exactly_the_ledger_number(self):
        gib, src = hl.pinned_reserve_for_ranks(MARGIN, CAP, CG)
        self._export(gib, src)
        b, rsrc = phb.pinned_host_reserve()
        self.assertEqual(b, int(round(gib, 4) * (1 << 30)))
        self.assertIn("ledger 1.4700 GiB", rsrc)


class TheLedgerBooksTheWallThePoolsEnforce(_CleanEnv):
    def _pool_admitted(self, nonreclaim_gib):
        with mock.patch.object(phb, "pinned_host_memory_bytes",
                               return_value=(CAP, CAP - int(nonreclaim_gib * GIB))):
            try:
                phb.check_and_register_pinned_post(
                    name="V4 paged host pool qsa_indexer", flag="--hicache-size",
                    requested_bytes=POOLS, reserve_bytes=None)
                return True
            except ValueError:
                return False
            finally:
                phb.clear_registered_posts()

    def _gap(self, env):
        for k, v in env.items():
            os.environ[k] = v
        gib, src = hl.pinned_reserve_for_ranks(MARGIN, CAP, CG)
        if not env:
            self._export(gib, src)
        wall, why = hl.pinned_wall(CAP, CG, gib, src)
        pools_gib = POOLS / GIB
        # ledger GREEN at the edge: run peak == wall. Every earlier moment holds
        # at most run peak - pools before the pools register -> admitted.
        self.assertIsNone(hl.pinned_wall_binding(wall, wall, why))
        self.assertTrue(self._pool_admitted(wall - pools_gib))
        # one step over the wall: the ledger refuses it by name
        self.assertIn("PINNED WALL", hl.pinned_wall_binding(wall + 0.01, wall, why))

    def test_no_gap_under_the_ledger_margin(self):
        self._gap({})

    def test_no_gap_under_an_explicit_env(self):
        self._gap({phb.PINNED_HOST_RESERVE_ENV: "2"})

    def test_the_yarn2_moment(self):
        """10:59:37Z: non-reclaimable 73.6 GiB (11.18 GB available of 84 GiB).
        Refused under the native 10 GiB, admitted under the ledger's margin."""
        at = 84 - 11.18e9 / GIB
        self.assertFalse(self._pool_admitted(at))
        self._export(*hl.pinned_reserve_for_ranks(MARGIN, CAP, CG))
        self.assertTrue(self._pool_admitted(at))


class TheLauncherLineNamesValueAndSource(_CleanEnv):
    def test_env(self):
        os.environ[phb.PINNED_HOST_RESERVE_ENV] = "2"
        line = launcher.pinned_reserve_line(2.0, "env SGLANG_PINNED_HOST_RESERVE_GIB=2")
        self.assertIn("effective=2.00 GiB source=env", line)

    def test_ledger(self):
        gib, src = hl.pinned_reserve_for_ranks(MARGIN, CAP, CG)
        self._export(gib, src)
        line = launcher.pinned_reserve_line(gib, src)
        self.assertIn("effective=1.47 GiB source=ledger", line)
        self.assertIn("ledger_wall_reserve=1.47 GiB", line)

    def test_native(self):
        line = launcher.pinned_reserve_line(None, "")
        self.assertIn("effective=10.00 GiB source=native", line)
        self.assertIn("ledger_wall_reserve=none", line)


class TheRenderCarriesNoFixedReserve(CustomTestCase):
    """27B 29.09.: no flat reserve in the container form -- the ledger's
    measured margin decides; an operator -e still wins."""

    def test_no_profile_renders_the_line(self):
        from sglang.srt.weg2 import profile_docker as PD

        for name, profile, fmt in (("27b", "qwen27b", "int8"), ("nf", "nextflash", "int4-mixed")):
            self.assertNotIn("PINNED_HOST_RESERVE", PD.render(name, profile, fmt), name)


class Wiring(CustomTestCase):
    def test_choose_gates_by_the_wall_and_hands_the_number_on(self):
        src = inspect.getsource(hl.choose)
        self.assertIn("pinned_reserve_for_ranks(", src)
        self.assertIn("and pinned_ok", src)
        self.assertIn("chosen.pinned_reserve = ", src)

    def test_the_launcher_exports_it_unless_the_env_is_explicit(self):
        src = inspect.getsource(launcher.main)
        self.assertIn("PINNED_HOST_RESERVE_LEDGER_ENV", src)
        self.assertIn("pinned_reserve_line(", src)


if __name__ == "__main__":
    unittest.main()
