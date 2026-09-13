# SPDX-License-Identifier: Apache-2.0
"""#1358 [fix] -- the cut must REACH the ledger, and lanes only where they exist.

TWO BOOT BLOCKERS, one commit, and both are the same class as the `argv_p`
positional shift that preceded them: a parameter whose PRODUCTION caller
behaves differently from every desk test.

(1) `choose_host_ledger(stage_ratio: str = "")` grew the parameter with a
    DEFAULT, and its ONLY caller never passed it. The lane record was looked up
    under `'pp=;d=tp3;legs=both'` while the seed is
    `'pp=39,13,12;d=tp3;legs=both'`, so `resolve_xchg_lanes` raised W102 on
    every rung: rc=2, ARM=0, every boot. A DEFAULT PARAMETER IS THE SAME TRAP
    AS A POSITIONAL ONE, one step quieter -- the positional shift crashed,
    this silently looked up the wrong key. Desk tests passed `stage_ratio=`
    explicitly and were blind to it, exactly as they were to the argv_p shift.

(2) The lane resolution had no guard on the weight source. An arm that pins no
    host bounce creates NO assemble buffer, so demanding a lane record from it
    refused the DEFAULT arm -- the very boot that would have produced the
    record the exchange arm needs. A producer that locks out its own bootstrap
    is worse than the under-charge it fixes.

The under-charge itself is 5.64 GiB, not 4.88: term(1) = 1.409 + 0.751 = 2.160
and term(5) = 1.409x5 + 0.751 = 7.795. The 4.88 compared a five-lane BUFFER sum
against a total that includes staging -- a population error, corrected here and
in the three docstrings that carried it.
"""

import inspect
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import host_ledger as hl
from sglang.srt.weg2 import launcher as lc
from sglang.test.test_utils import CustomTestCase


class TheCutReachesTheLedger(CustomTestCase):
    def test_the_only_caller_passes_stage_ratio(self):
        """SOURCE PIN at the production call site, not at a keyword test.

        A behavioural test that passes `stage_ratio=` itself cannot see this;
        that is precisely how it shipped.
        """
        src = "\n".join(ln for ln in inspect.getsource(lc).splitlines()
                        if not ln.strip().startswith("#"))
        i = src.index("arm, reap_headroom_gib, lines, cg = choose_host_ledger(")
        window = src[i:i + 2500]
        self.assertIn("stage_ratio=", window,
                      "choose_host_ledger's only caller does not pass the cut; "
                      "the lane record is then looked up under an empty cut")
        self.assertIn("legs=", window)

    def test_an_empty_cut_is_not_the_seeded_cut(self):
        """The two keys that were being confused, as values."""
        self.assertNotEqual(hl.xchg_cut_key(""), hl.xchg_cut_key("39,13,12"))
        self.assertIn("pp=39,13,12", hl.xchg_cut_key("39,13,12"))
        # #1362 [bootstrap] SEMANTICS CHANGED DELIBERATELY: an unrecorded cut
        # is now the FIRST BOOT of that cut, priced at the region's worst case
        # and named, not refused -- refusing it made the boot that would
        # produce the record impossible. What must still differ is the KEY, and
        # what must still be measured is a cut that HAS a record.
        n_empty, prov = hl.resolve_xchg_lanes(hl.xchg_cut_key(""))
        self.assertIn("WORST-CASE", prov)
        n_seed, seed_prov = hl.resolve_xchg_lanes(hl.xchg_cut_key("39,13,12"))
        self.assertEqual(n_seed, 5)
        self.assertIn("source=measured", seed_prov)
        self.assertNotEqual(n_empty, n_seed,
                            "the empty cut must not silently inherit the seed")


class LanesAreResolvedOnlyWhereTheyExist(CustomTestCase):
    def test_the_guard_uses_the_one_arm_predicate(self):
        src = "\n".join(ln for ln in inspect.getsource(lc).splitlines()
                        if not ln.strip().startswith("#"))
        i = src.index("resolve_xchg_lanes(")
        self.assertIn("xchg_bounce_arm_pins_host(", src[max(0, i - 700):i],
                      "the lane lookup must be guarded by the ONE predicate "
                      "that answers whether this arm pins host bytes")

    def test_a_default_arm_pins_nothing_and_must_not_be_refused(self):
        """THE BOOTSTRAP: the arm that would MAKE the record must be bootable."""
        self.assertFalse(lc.xchg_bounce_arm_pins_host("ring", "ipc"))
        self.assertFalse(lc.xchg_bounce_arm_pins_host("ring", "host"))
        self.assertTrue(lc.xchg_bounce_arm_pins_host("exchange", "host"))

    def test_the_unarmed_line_says_n_a_rather_than_a_number(self):
        src = inspect.getsource(lc)
        i = src.index("lanes=n/a (no exchange)")
        self.assertIn("weight_source=", src[i:i + 200],
                      "the n/a line must name the arm it is reporting about")


class TheUnderchargeNumberIsCorrected(CustomTestCase):
    def test_the_arithmetic(self):
        from sglang.srt.weg2 import xchg_bounce as xb

        g = 1024 ** 3
        w = 756323776
        one = xb.bounce_terms(bytes_per_direction=29119878266, n_layers=64,
                              widest_layer_bytes=w, pairs=3, depth=1,
                              slot_bytes=134217728, n_lanes=1)
        five = xb.bounce_terms(bytes_per_direction=29119878266, n_layers=64,
                               widest_layer_bytes=w, pairs=3, depth=1,
                               slot_bytes=134217728, n_lanes=5)
        self.assertAlmostEqual(one.total_bytes / g, 2.160, places=2)
        self.assertAlmostEqual(five.total_bytes / g, 7.795, places=2)
        self.assertAlmostEqual((five.total_bytes - one.total_bytes) / g,
                               5.635, places=2)

    def test_no_docstring_still_says_4_88(self):
        """The wrong figure was carried into three files; none may keep it."""
        from sglang.srt.weg2 import xchg_bounce as xb

        for mod in (lc, hl, xb):
            with self.subTest(module=mod.__name__):
                self.assertNotIn("4.88", inspect.getsource(mod))


if __name__ == "__main__":
    unittest.main()
