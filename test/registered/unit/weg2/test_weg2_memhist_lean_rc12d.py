# SPDX-License-Identifier: Apache-2.0
"""rc12d (27.09. 02:06Z): SGLANG_WEG2_MEMHIST=1 in the container env armed the
allocation history with stacks from rank start in P AND D; P's memory.current
stood at 67.7 GiB instead of 62.3 (rc12c) and D was refused by its pinned-host
riegel. The history was never booked.

The lean form (SGLANG_WEG2_MEMHIST=sleep: D only, armed after the first sleep,
20000 entries) and the ledger post 'memhist' (weg2/memhist.py), tested here.
"""

import os
from unittest import mock

from sglang.test.test_utils import CustomTestCase

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import host_ledger as HL  # noqa: E402
from sglang.srt.weg2 import memhist as MH  # noqa: E402

RANKS = {"P": 3, "D": 3}


class ThePlan(CustomTestCase):
    def test_off(self):
        self.assertIsNone(MH.plan({}))
        self.assertIsNone(MH.plan({MH.MEMHIST_ENV: "0"}))

    def test_legacy_one_is_unchanged(self):
        p = MH.plan({MH.MEMHIST_ENV: "1"})
        self.assertEqual((p.mode, p.groups, p.max_entries), (MH.MODE_LOAD, ("P", "D"), 200000))
        self.assertFalse(p.run_moment_only)

    def test_lean(self):
        p = MH.plan({MH.MEMHIST_ENV: "sleep"})
        self.assertEqual((p.mode, p.groups, p.max_entries), (MH.MODE_SLEEP, ("D",), 20000))
        self.assertTrue(p.run_moment_only)
        self.assertTrue(p.armed_for("D"))
        self.assertFalse(p.armed_for("P"))

    def test_overrides(self):
        p = MH.plan({MH.MEMHIST_ENV: "sleep", MH.GROUPS_ENV: "p,d", MH.MAX_ENTRIES_ENV: "5000"})
        self.assertEqual((p.groups, p.max_entries), (("P", "D"), 5000))


class TheLedgerPost(CustomTestCase):
    def test_legacy_reproduces_the_rc12d_measurement(self):
        # the per-entry price is DERIVED from rc12d; the legacy form on P's
        # three ranks must give back the 5.4 GiB it was derived from
        gib, run_only, prov = MH.host_charge({MH.MEMHIST_ENV: "1"}, {"P": {}, "D": {}}, RANKS)
        self.assertAlmostEqual(gib, 2 * 5.4, places=6)
        self.assertFalse(run_only)
        self.assertIn("rc12d", prov)

    def test_lean_is_d_only_and_small(self):
        gib, run_only, prov = MH.host_charge({}, {"P": {}, "D": {MH.MEMHIST_ENV: "sleep"}}, RANKS)
        self.assertAlmostEqual(gib, 3 * 20000 * MH.HOST_KIB_PER_ENTRY / 1024 ** 2, places=9)
        self.assertLess(gib, 0.6)
        self.assertTrue(run_only)
        self.assertNotIn("P mode", prov)

    def test_container_env_lean_still_d_only(self):
        gib, _r, _p = MH.host_charge({MH.MEMHIST_ENV: "sleep"}, {"P": {}, "D": {}}, RANKS)
        self.assertAlmostEqual(gib, 3 * 20000 * MH.HOST_KIB_PER_ENTRY / 1024 ** 2, places=9)

    def test_off_is_zero(self):
        self.assertEqual(MH.host_charge({}, {"P": {}, "D": {}}, RANKS), (0.0, False, ""))

    def _terms(self, **kw):
        images = HL.ImageTerms(p_gib=1.0, d_gib=1.0, p_source="t", d_source="t", p_measured=True,
                               d_measured=True, extra_p_gib=0.0, extra_d_gib=0.0)
        return HL.charge_terms(1, 1024, 3, images, flip_ratchet_gib=0.5, **kw)

    def test_moments(self):
        base = self._terms()
        both = self._terms(memhist_gib=2.0, memhist_run_only=False)
        run = self._terms(memhist_gib=2.0, memhist_run_only=True)
        b0, r0 = HL._boot_charges_gib(base), HL._run_moment_charges_gib(base)
        self.assertAlmostEqual(HL._boot_charges_gib(both), b0 + 2.0)
        self.assertAlmostEqual(HL._run_moment_charges_gib(both), r0 + 2.0)
        self.assertAlmostEqual(HL._boot_charges_gib(run), b0)
        self.assertAlmostEqual(HL._run_moment_charges_gib(run), r0 + 2.0)
        self.assertEqual(base["memhist_gib"], 0.0)


class TheRankArmsOnlyWhereAndWhenPlanned(CustomTestCase):
    def _run(self, env, stage):
        from sglang.srt.managers import weg2_memory_saver as S

        calls = []
        with mock.patch.dict(os.environ, env, clear=False), \
                mock.patch.object(S, "_MEMHIST_ARMED", False), \
                mock.patch("torch.cuda.memory._record_memory_history",
                           lambda **kw: calls.append(kw)):
            S._arm_memory_history(stage)
            armed = S._MEMHIST_ARMED
        return calls, armed

    def test_lean_d_arms_after_sleep_not_at_load(self):
        env = {MH.MEMHIST_ENV: "sleep", "SGLANG_WEG2_GROUP": "D"}
        self.assertEqual(self._run(env, "load"), ([], False))
        self.assertEqual(self._run(env, "sleep"), ([{"max_entries": 20000}], True))

    def test_lean_p_never_arms(self):
        env = {MH.MEMHIST_ENV: "sleep", "SGLANG_WEG2_GROUP": "P"}
        self.assertEqual(self._run(env, "sleep"), ([], False))

    def test_legacy_arms_at_load_with_its_ring(self):
        env = {MH.MEMHIST_ENV: "1", "SGLANG_WEG2_GROUP": "P"}
        self.assertEqual(self._run(env, "load"), ([{"max_entries": 200000}], True))


if __name__ == "__main__":
    import unittest

    unittest.main()
