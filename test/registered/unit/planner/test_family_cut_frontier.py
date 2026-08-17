# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The family-placement frontier is arithmetic, and it must stay pinned.

The whole family-placement verdict rests on one fact: pipeline stages are
CONTIGUOUS and full attention is interleaved at a fixed period, so a stage's FA
count and its layer count are proportional. If that ever stops holding -- a
checkpoint with a non-uniform interleave, or a non-contiguous stage assignment
-- the verdict must be re-derived rather than quoted.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_SOLVER = os.path.join(_REPO, "bench", "family_placement", "solve_family_cut.py")


def _load():
    sys.path.insert(0, os.path.dirname(_SOLVER))
    try:
        import solve_family_cut as s
    finally:
        sys.path.pop(0)
    return s


def _geometry():
    s = _load()
    lt = [
        "full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(64)
    ]
    return s, s.fa_positions(lt)


class TestSelfTestIsGreen(unittest.TestCase):
    def test_self_test_passes(self):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["PYTHONPATH"] = os.path.join(_REPO, "python")
        done = subprocess.run(
            [sys.executable, _SOLVER, "--self-test"],
            capture_output=True, text=True, timeout=300, env=env,
        )
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)


class TestTheFrontier(unittest.TestCase):
    def test_it_is_four_fa_plus_three(self):
        s, fa = _geometry()
        fr = s.frontier(64, fa)
        for k, v in fr.items():
            if k < 15:
                self.assertEqual(v, 4 * k + 3, f"FA0={k}")

    def test_the_incumbent_is_on_the_frontier(self):
        s, fa = _geometry()
        self.assertEqual(s.frontier(64, fa)[7], 31)

    def test_shedding_one_fa_costs_four_layers(self):
        s, fa = _geometry()
        fr = s.frontier(64, fa)
        self.assertEqual(fr[7] - fr[6], 4)


class TestTheRequestedDirectionIsWorseOnBothHalves(unittest.TestCase):
    """The load-bearing claim: fewer FA on stage 0 also means less GDN."""

    def _priced(self, s, fa, bounds):
        return s.build_cut(
            bounds, 64, fa,
            attn_mib=355.1, gdn_mib=476.1, kv_mib_per_fa_layer=852.1,
        )

    def test_fewer_fa_on_stage0_also_reduces_gdn_on_stage0(self):
        s, fa = _geometry()
        inc = self._priced(s, fa, (31, 48, 64))
        req = self._priced(s, fa, (27, 48, 64))
        self.assertLess(req.stages[0].fa, inc.stages[0].fa)
        self.assertLess(req.stages[0].gdn, inc.stages[0].gdn)

    def test_the_mass_lands_on_a_smaller_card(self):
        s, fa = _geometry()
        inc = self._priced(s, fa, (31, 48, 64))
        req = self._priced(s, fa, (27, 48, 64))
        moved = inc.stages[0].total_mib - req.stages[0].total_mib
        self.assertGreater(moved, 2500)
        self.assertAlmostEqual(
            req.stages[1].total_mib - inc.stages[1].total_mib, moved, delta=1.0
        )

    def test_the_more_gdn_direction_relieves_both_small_stages(self):
        s, fa = _geometry()
        inc = self._priced(s, fa, (31, 48, 64))
        alt = self._priced(s, fa, (35, 50, 64))
        self.assertGreater(alt.stages[0].gdn, inc.stages[0].gdn)
        self.assertLess(alt.stages[1].total_mib, inc.stages[1].total_mib)
        self.assertLess(alt.stages[2].total_mib, inc.stages[2].total_mib)


if __name__ == "__main__":
    unittest.main()
