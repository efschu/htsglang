"""NF1c (rc12z30y3c, 16:55:33Z): W87 CUSHION FLOOR 81.36 + 1.50 = 82.86 > 82.53.

The hard bound 82.53 = cgroup ceiling 84.00 - margin 1.47, and the margin's
residual 0.91 is the max over the five 0912 ratchet replays -- all
Qwen3.8-27B-INT8 boots (H87 names the xsn series as 27B). The run origin and
the ratchet already say FOREIGN-MODEL for NF; the margin charged the 27B
residual silently. NF's own two-sided replay (cf6f2108fb: predicted 81.36 vs
measured raw memory.peak 80.30 / 80.32 incl. teardown of 09291559, same arm)
is an OVER-prediction, residual <= -1.04 -> clamped 0.

The cushion floor is NOT the doubled term: the W98 latch measures against
min(reap mark, memory.max) = 84.00 (the 09291559 log: ``mark=84.00``), so
``run_peak + residual + drift + floor <= 84`` is exactly the conjunct -- one
wall, two different terms. What was wrong is which model's residual.
"""
from __future__ import annotations

import csv
import json
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import form as pdflip_form
from flliper.srt.pdflip import host_census, host_ledger
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GIB = int(host_ledger.GIB)
# 1539 06b: the evidence is a FROZEN COPY, not the live tree. The live
# /spinning/docker-acceptance/nf/evidence files are append-only and written by
# every boot: its host_census_record.json first entry now holds boots from 10.02.
# on and replays 84.06 GiB against the 81.55 bound, while the 29.09. 16:18:07Z
# snapshot of that record (the state the 16:34 z30y3b boot, i.e. this test, was
# written against) replays 81.364 -- the 81.36 test_nf1c_arm_funds pins to the
# cent. Provenance and sha256: fixtures/host_ledger_evidence_1539/nf_0929/PROVENANCE.json.
EV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures",
                  "host_ledger_evidence_1539", "nf_0929")
REC = f"{EV}/weg2_measured_record.json"
CENSUS = f"{EV}/host_census_record.json"
TAG_1559 = "dkrnfh91dprsavisadoptstcutvsyncodx2bswre2cutz30y2bar1dauer09291559"
MEMTS_1559 = f"{EV}/memts_weg2_09291559.csv.txt"
NF_NAME = "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"


def _margin(ref_ok, fp=""):
    return host_ledger.resolve_margin(flip_ratchet_charged_gib=1.84, window_min=90,
                                      reference_model_ok=ref_ok, model_footprint=fp)


class TestOwnModelResidual(CustomTestCase):
    def test_footprint_is_the_pinned_nf_one(self):
        self.assertEqual(host_ledger.NF_FOOTPRINT, pdflip_form.REFERENCE_FOOTPRINTS[NF_NAME])

    def test_reference_model_unchanged(self):
        for ok in (None, True):
            m = _margin(ok, host_ledger.NF_FOOTPRINT)
            self.assertAlmostEqual(m.residual_gib, 0.908 + 0.07 * 90 / 1024, places=6)
            self.assertAlmostEqual(84.0 - m.total_gib, 82.53, places=2)  # NF1c's bound, 27B

    def test_nf_charges_its_own_rows_not_the_27b_ones(self):
        m = _margin(False, host_ledger.NF_FOOTPRINT)
        self.assertAlmostEqual(m.residual_gib, 0.07 * 90 / 1024, places=6)  # -1.04 clamped + drift
        self.assertIn("OWN-MODEL", m.residual_source)
        self.assertIn(host_ledger.FOREIGN_REFERENCE_TAG, m.residual_source)
        self.assertAlmostEqual(84.0 - m.total_gib, 83.44, places=2)

    def test_unknown_foreign_model_keeps_27b_rows_loudly(self):
        m = _margin(False, "")
        self.assertAlmostEqual(m.residual_gib, 0.908 + 0.07 * 90 / 1024, places=6)
        self.assertIn("FALLBACK", m.residual_source)

    @unittest.skipUnless(os.path.exists(MEMTS_1559), "NF evidence absent")
    def test_own_row_is_the_measured_one(self):
        with open(MEMTS_1559) as f:
            peak = max(int(r["cg_peak_b"]) for r in csv.DictReader(f)) / GIB
        row = host_ledger.RUN_PEAK_RESIDUAL_RATCHET_OWN_MODEL_GIB[host_ledger.NF_FOOTPRINT][TAG_1559]
        self.assertAlmostEqual(row, round(peak - 81.36, 2), places=2)
        self.assertLess(row, 0.0)  # over-prediction: the NF ledger is >= the metal


class TestNfArmReplay(CustomTestCase):
    """choose() on NF1c's own inputs (the cf6f2108fb replay: real census, real
    09291559 record, ratchet 1.84), ladder restricted to S=1 M=600 like the pin."""

    def _choose(self, extra_origin_gib=0.0, fp=host_ledger.NF_FOOTPRINT):
        census = host_census.ledger_terms(next(iter(json.load(open(CENSUS)).values())))
        rec = host_ledger.read_measured_record(REC, boot_tag=TAG_1559)
        ratchet = host_ledger.FlipRatchet(per_flip_gib=1.84, flips_priced=1,
                                          source="09291559 FLIP", from_record=True)
        return host_ledger.choose(
            int(125.70 * GIB), int(100.50 * GIB), arms=[(1, 600)], s_gb_d=4,
            ring_absent_by_design=True, cg_current_bytes=int((1.09 + extra_origin_gib) * GIB),
            reclaimable_bytes=int(0.29 * GIB), cg_ceiling_bytes=84 * GIB, measured_record=rec,
            flip_ratchet=ratchet, arena_gib=5.75, staging_gb=0.0494, anchor_mib=101,
            d_draft_host_gib=1587 / 1024, l3_index_gib=0.10, cold_tier_shm_gib=38.97,
            census=census, reference_model_ok=False,
            reference_model_why=f"footprint 5d82a6f6b1f1... of {NF_NAME} != 28e1c5c3ec47...",
            model_footprint=fp,
        )

    @unittest.skipUnless(all(os.path.exists(p) for p in (REC, CENSUS)), "NF evidence absent")
    def test_nf1c_arm_funds(self):
        arm, _h, lines = self._choose()
        self.assertAlmostEqual(arm.predicted_run_peak_gib(), 81.36, places=2)
        self.assertIn("run_peak=81.36 GiB vs hard bound 83.44 GiB", "\n".join(lines))

    @unittest.skipUnless(all(os.path.exists(p) for p in (REC, CENSUS)), "NF evidence absent")
    def test_without_footprint_it_is_nf1cs_refusal(self):
        with self.assertRaises(host_ledger.PdFlipHostLedgerRefused) as cm:
            self._choose(fp="")
        self.assertIn("CUSHION FLOOR (81.36 + 1.50 floor = 82.86 > 82.53", str(cm.exception))

    @unittest.skipUnless(all(os.path.exists(p) for p in (REC, CENSUS)), "NF evidence absent")
    def test_just_above_the_honest_limit_still_refused(self):
        # honest limit: 84.00 - 0.563 margin - 1.50 floor = 81.94
        arm, _h, _l = self._choose(extra_origin_gib=0.55)  # 81.91 funds
        self.assertLess(arm.predicted_run_peak_gib(), 81.94)
        with self.assertRaises(host_ledger.PdFlipHostLedgerRefused) as cm:
            self._choose(extra_origin_gib=0.62)  # 81.98: under the bound, over bound - floor
        self.assertIn("binding: CUSHION FLOOR (81.98 + 1.50 floor = 83.48 > 83.44", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
