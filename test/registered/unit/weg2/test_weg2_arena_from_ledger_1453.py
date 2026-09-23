# SPDX-License-Identifier: Apache-2.0
"""#1453: the L2 KV arena is sized from the host ledger's headroom.

User order 2026-09-16 ('L2-Groesse aus Ledger-Spielraum').  The arena is the
largest host term; when the priced run peak sits above the riegel the arena
gives that excess back (rounded up to 0.25 GiB, never below 8 GiB = one 262k
request) and the ledger prices once more -- boot weg2xsn207 ran 93.15 GiB
against a riegel of 93.00 only under a named deviation.

Hermetic: the pure sizing function and source pins for the re-price wiring.
"""
import inspect
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import launcher
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class Sizing(CustomTestCase):
    def test_over_the_riegel_gives_the_excess_back_rounded_up(self):
        new, why = launcher.arena_from_ledger(22.0, 93.15, 93.0)
        self.assertEqual(new, 21.75)          # 0.15 over -> 0.25 back
        self.assertIn("re-pricing", why)
        self.assertIn("KV slots", why)
        new, _ = launcher.arena_from_ledger(22.0, 95.4, 93.0)
        self.assertEqual(new, 19.5)           # 2.4 over -> 2.5 back

    def test_under_the_riegel_stays(self):
        new, why = launcher.arena_from_ledger(22.0, 92.0, 93.0)
        self.assertIsNone(new)
        self.assertIn("headroom 1.00", why)

    def test_floor_and_missing_inputs(self):
        new, why = launcher.arena_from_ledger(8.0, 99.0, 93.0)
        self.assertIsNone(new)
        self.assertIn("floor", why)
        new, _ = launcher.arena_from_ledger(9.0, 99.0, 93.0)
        self.assertEqual(new, 8.0)            # never below the floor
        self.assertIsNone(launcher.arena_from_ledger(22.0, None, 93.0)[0])
        self.assertIsNone(launcher.arena_from_ledger(22.0, 99.0, None)[0])
        self.assertIsNone(launcher.arena_from_ledger(22.0, 99.0, 93.0, enabled=False)[0])
        self.assertEqual(launcher.ARENA_FROM_LEDGER_FLOOR_GIB, 8.0)


class Wiring(CustomTestCase):
    def test_main_prices_twice_when_the_arena_moves(self):
        src = inspect.getsource(launcher.main)
        self.assertIn("def _price_host_ledger():", src)
        self.assertEqual(src.count("= _price_host_ledger()"), 2)
        self.assertIn("arena_from_ledger(", src)
        self.assertIn('os.environ["SGLANG_HICACHE_ARENA_GIB"] = f"{_arena_new:g}"', src)
        # the env the groups inherit is the same one the ledger term reads
        self.assertLess(src.index('os.environ["SGLANG_HICACHE_ARENA_GIB"]'), src.index("= _price_host_ledger()", src.index("= _price_host_ledger()") + 1))


if __name__ == "__main__":
    unittest.main()
