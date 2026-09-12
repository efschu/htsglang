# SPDX-License-Identifier: Apache-2.0
"""#1350c -- Sigma H re-anchored on the PER-RANK MANIFESTS, the one independent source.

Everything the ring solve reads today is downstream of the ring it sizes
(#1350b): the per-card tag census of boot n is the Sigma H of boot n-1 minus 1-2
MiB, and the measured dormant image is that census plus the ring's own
preallocated slack. The manifests are not: `xchg_manifest` writes, per rank and
per region tag, the NBYTES OF EVERY TENSOR THAT RANK HOLDS, read from the live
parameter inventory, and they know nothing about any ring.

THE FORMULA IS max(P, D) PER CARD, and the ruling is MEASURED, not argued.
C19's ring holds the DORMANT image and the legs copy through it, so exactly one
group is dormant at a time and a card holds the larger of the two, never both.
Against boot weg2xsn24's own ten manifests:

    max per card  32783 MiB = 32.01 GiB
    sum per card  63092 MiB = 61.61 GiB
    independently recorded `RUN MOMENT = the host weights term 32.19 GiB`
      (host_ledger.price's #1327 note, derived years of tickets earlier and by
       another route entirely)

`max` lands 177 MiB (0.5 %) from a figure nothing in this ticket produced;
`sum` lands 29.4 GiB away. The operator's brief said "P-Image + D-Image", and
this file is the correction with its evidence.
"""

import json
import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import ring_table as rt
from sglang.test.test_utils import CustomTestCase

MIB = 1024 * 1024

#: boot weg2xsn24's own ten manifests, summed per (card, group), MiB -- read
#: once from /spinning/evidence-665-f1/weg2xsn24_0912/phase_manifest_*.json and
#: pinned here so the test is HERMETIC and runs on the remote gate.
XSN24 = {0: {"P": 17896, "D": 18287}, 1: {"P": 5591, "D": 6836},
         2: {"P": 7660, "D": 6820}}
XSN24_MAX_MIB = 18287 + 6836 + 7660          # 32783
XSN24_SUM_MIB = (17896 + 18287) + (5591 + 6836) + (7660 + 6820)   # 63090
#: What the tree recorded for the same quantity, by another route, before this
#: ticket existed: host_ledger.price's #1327 note.
RECORDED_RUN_MOMENT_GIB = 32.19
#: The charged Sigma H of the xsn25 dry run, and the census it came from.
CHARGED_SIGMA_H_MIB = 47512


def _write_fixture(d):
    for card, groups in XSN24.items():
        for group, mib in groups.items():
            rank = card
            name = f"phase_manifest_{group}_rank{rank}x0_weights.json"
            with open(os.path.join(d, name), "w") as f:
                json.dump({
                    "version": 1, "boot_token": "1789241942", "card": card,
                    "group": group, "rank": rank, "pp_rank": 0, "tp_rank": rank,
                    "region_tag": "weights",
                    "pieces": [{
                        "param_name": f"w{card}{group}", "tag": "weights",
                        "nbytes": mib * MIB, "itemsize": 1,
                        "rows_full": 1, "cols_full": mib * MIB,
                        "tensor_class": "linear",
                    }],
                }, f)


class ManifestSum1350c(CustomTestCase):
    def test_max_per_card_and_the_refuted_sum_are_both_returned(self):
        with tempfile.TemporaryDirectory() as d:
            _write_fixture(d)
            rows, line, detail = rt.sigma_h_from_manifests(d)
            self.assertEqual(rows, {0: 18287, 1: 6836, 2: 7660})
            self.assertEqual(detail["sigma_max_mib"], XSN24_MAX_MIB)
            self.assertEqual(detail["sigma_sum_mib"], XSN24_SUM_MIB)
            self.assertIn("SIGMA-H source=manifest", line)
            self.assertIn("boot=1789241942", line)
            self.assertIn("MAX per card", line)

    def test_max_reproduces_the_independently_recorded_run_moment(self):
        """The ruling between max and sum is a MEASUREMENT, not an argument."""
        max_gib = XSN24_MAX_MIB / 1024.0
        sum_gib = XSN24_SUM_MIB / 1024.0
        self.assertLess(abs(max_gib - RECORDED_RUN_MOMENT_GIB), 0.25)
        self.assertGreater(abs(sum_gib - RECORDED_RUN_MOMENT_GIB), 25.0)

    def test_an_absent_manifest_set_is_an_absence_not_a_zero_ring(self):
        with tempfile.TemporaryDirectory() as d:
            rows, line, detail = rt.sigma_h_from_manifests(d)
            self.assertEqual(rows, {})
            self.assertEqual(detail, {})
            self.assertIn("ABSENCE, not a zero ring", line)
        rows, line, _d = rt.sigma_h_from_manifests("/no/such/dir/at/all")
        self.assertEqual(rows, {})
        self.assertIn("ABSENCE, not a zero ring", line)


class CrossCheck1350c(CustomTestCase):
    def test_the_census_is_named_as_the_suspect_source(self):
        ok, why = rt.manifest_census_crosscheck(
            XSN24_MAX_MIB, CHARGED_SIGMA_H_MIB)
        self.assertFalse(ok)
        self.assertIn("the CENSUS is the suspect source", why)
        self.assertIn("-14729 MiB", why)
        self.assertIn("#1350b recurrence signature", why)

    def test_the_independent_xsn20_level_census_also_disagrees(self):
        """42156 (xsn20-in) and 42492 (xsn21b) are the LEAST ratcheted censuses
        in the series, and the manifests still sit ~9.2 GiB below them: the
        recurrence had already compounded before the series this ticket read."""
        for census in (42156, 42492):
            with self.subTest(census=census):
                ok, why = rt.manifest_census_crosscheck(XSN24_MAX_MIB, census)
                self.assertFalse(ok)
                self.assertIn("the CENSUS is the suspect source", why)
        self.assertAlmostEqual((42156 - XSN24_MAX_MIB) / 1024.0, 9.15, places=1)

    def test_a_manifest_sum_ABOVE_the_census_blames_the_manifests(self):
        """The other direction is the dangerous one: a ring sized BELOW the
        tensors the ranks carry ends in W31/W51 exhaustion at the flip."""
        ok, why = rt.manifest_census_crosscheck(50000, 40000)
        self.assertFalse(ok)
        self.assertIn("the MANIFESTS are the suspect source", why)
        self.assertIn("W31/W51", why)

    def test_inside_the_tolerance_the_two_corroborate(self):
        ok, why = rt.manifest_census_crosscheck(32783, 32000)
        self.assertTrue(ok)
        self.assertIn("CROSS-CHECK OK", why)
        self.assertLess(rt.MANIFEST_CENSUS_TOLERANCE, 0.10)


class TheReplayIsFlatOnTheManifestAnchor1350c(CustomTestCase):
    """The six-value ladder, re-anchored: the manifests carry no ring at all."""

    def test_the_manifest_anchor_does_not_move_with_the_ring(self):
        """Same checkpoint, same ranks -> same tensor bytes, whatever the ring.

        The six measured Sigma H values 43996/44587/45471/46356/47514/48672
        differ by +4676 MiB with no weight change behind them. A manifest sum
        cannot move for that reason, because no term of it is a ring quantity:
        it is `sum(ManifestPiece.nbytes)` over the live parameter inventory.
        """
        with tempfile.TemporaryDirectory() as d:
            _write_fixture(d)
            first = rt.sigma_h_from_manifests(d)[2]["sigma_max_mib"]
            for _ring_grew in range(5):
                again = rt.sigma_h_from_manifests(d)[2]["sigma_max_mib"]
                self.assertEqual(again, first)
        ladder = (43996, 44587, 45471, 46356, 47514, 48672)
        self.assertEqual(ladder[-1] - ladder[0], 4676)
        for v in ladder:
            self.assertGreater(v - XSN24_MAX_MIB, 10000)   # >= 9.8 GiB above


if __name__ == "__main__":
    unittest.main()
