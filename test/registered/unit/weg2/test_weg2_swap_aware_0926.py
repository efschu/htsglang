# SPDX-License-Identifier: Apache-2.0
"""SWAP READINESS 0926 -- the ledger's currencies against a cgroup that swaps.

/spinning/gpu-arb/docs/SWAP_READINESS_0926.md: every host currency assumes the
serving cgroup cannot swap (CT999 ``swap: 0``, Docker ``--memory-swap ==
--memory``). Swapped-out pages leave ``anon``/``shmem``/``memory.current``, so
a swapping cgroup reads optimistic. ``SGLANG_WEG2_SWAP_AWARE=1`` counts
``memory.swap.current`` back in; default off must stay bit-identical.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from sglang.srt.weg2 import host_ledger as hl

GIB = 1 << 30
STAT = (
    "anon {a}\nfile {f}\nshmem {s}\nslab_reclaimable 0\nslab_unreclaimable {su}\n"
    "inactive_file {inf}\nactive_file {af}\nunevictable 0\n"
)


def _cg(tmp: str, swap_current=None, zswap=None, swap_max=None, current=40 * GIB):
    with open(os.path.join(tmp, "memory.stat"), "w") as f:
        f.write(STAT.format(a=20 * GIB, f=12 * GIB, s=10 * GIB, su=GIB, inf=GIB, af=GIB))
    with open(os.path.join(tmp, "memory.current"), "w") as f:
        f.write(f"{current}\n")
    if swap_current is not None:
        with open(os.path.join(tmp, "memory.swap.current"), "w") as f:
            f.write(f"{swap_current}\n")
    if zswap is not None:
        with open(os.path.join(tmp, "memory.zswap.current"), "w") as f:
            f.write(f"{zswap}\n")
    if swap_max is not None:
        with open(os.path.join(tmp, "memory.swap.max"), "w") as f:
            f.write(f"{swap_max}\n")


def _meminfo(tmp: str, total_kb: int, free_kb: int) -> str:
    p = os.path.join(tmp, "meminfo")
    with open(p, "w") as f:
        f.write(f"MemTotal: 123 kB\nSwapTotal: {total_kb} kB\nSwapFree: {free_kb} kB\n")
    return p


class SwapAwareCurrency(unittest.TestCase):
    def _env(self, on: bool):
        env = {k: v for k, v in os.environ.items() if k != hl.SWAP_AWARE_ENV}
        if on:
            env[hl.SWAP_AWARE_ENV] = "1"
        return mock.patch.dict(os.environ, env, clear=True)

    def test_swap_zero_is_bit_identical_both_ways(self):
        with tempfile.TemporaryDirectory() as tmp:
            _cg(tmp, swap_current=0)
            with self._env(False):
                off_c = hl.read_flip_currency_gib(tmp)
                off_p = hl.read_cgroup_pressure(tmp)
            with self._env(True):
                on_c = hl.read_flip_currency_gib(tmp)
                on_p = hl.read_cgroup_pressure(tmp)
        self.assertEqual(off_c, (20 + 10 + 1) * GIB / GIB)
        self.assertEqual(off_c, on_c)
        self.assertEqual(off_p["nonreclaim_gib"], on_p["nonreclaim_gib"])
        self.assertEqual(off_p["nonreclaim_gib"], 38.0)
        self.assertEqual(off_p["swap_gib"], 0.0)

    def test_swapped_pages_count_only_with_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            _cg(tmp, swap_current=2 * GIB, zswap=GIB)
            with self._env(False):
                off_c = hl.read_flip_currency_gib(tmp)
                off_p = hl.read_cgroup_pressure(tmp)
            with self._env(True):
                on_c = hl.read_flip_currency_gib(tmp)
                on_p = hl.read_cgroup_pressure(tmp)
        self.assertEqual(off_c, 31.0)
        self.assertEqual(on_c, 33.0)  # swap.current added once, zswap NOT added
        self.assertEqual(off_p["nonreclaim_gib"], 38.0)
        self.assertEqual(on_p["nonreclaim_gib"], 40.0)
        self.assertIn("memory.swap.current", on_p["source"])
        self.assertNotIn("memory.swap.current", off_p["source"])
        self.assertEqual(off_p["swap_gib"], 2.0)

    def test_absent_swap_file_is_zero_contribution_and_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            _cg(tmp)  # kernel/cgroup without memory.swap.current
            with self._env(True):
                self.assertEqual(hl.read_flip_currency_gib(tmp), 31.0)
                p = hl.read_cgroup_pressure(tmp)
                line, warn = hl.swap_state_line(tmp, _meminfo(tmp, 0, 0))
        self.assertEqual(p["nonreclaim_gib"], 38.0)
        self.assertIsNone(p["swap_gib"])
        self.assertFalse(warn)
        self.assertIn("swap.current=absent", line)

    def test_visible_swap_max_is_not_evidence_and_warn_follows_swap_current(self):
        # CT999: visible swap.max "max", SwapTotal 0, nothing swapped -> no warn.
        with tempfile.TemporaryDirectory() as tmp:
            _cg(tmp, swap_current=0, swap_max="max")
            with self._env(False):
                line, warn = hl.swap_state_line(tmp, _meminfo(tmp, 0, 0))
        self.assertFalse(warn)
        self.assertIn("swap.max(visible)=max", line)
        self.assertIn("SwapTotal=0.00", line)
        # Host with swap and pages out -> warn, and the hint names the switch.
        with tempfile.TemporaryDirectory() as tmp:
            _cg(tmp, swap_current=GIB, swap_max="max")
            with self._env(False):
                line, warn = hl.swap_state_line(tmp, _meminfo(tmp, 32 << 20, 31 << 20))
        self.assertTrue(warn)
        self.assertIn("WARN", line)
        self.assertIn(hl.SWAP_AWARE_ENV + "=1 to count it", line)


if __name__ == "__main__":
    unittest.main()
