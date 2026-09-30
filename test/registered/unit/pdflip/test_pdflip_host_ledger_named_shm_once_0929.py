"""NF 09291634 (z30y3b c7b12134cb): W21 run peak 101.02 vs hard bound 82.53.

The origin 25.35 GiB came from the z30y2 09291559 D record (16:04:07Z, stored
39.65, -14.30 census posts). Not a wedge sample (first sleep, epoch 0, 51 live
pids, oom 0; the D-admission wedge came 14 min later) -- a sample in the NEW
image currency read by a reader that nets only half of what it holds:

1. eb73b011d1 switched the sampler's image from RssShmem (NF 140 GiB, residual
   -99: the floor never bound) to ``pss_anon_shared`` -- named tmpfs files
   (tmpfs expert store, arena file, L3 index) are no longer subtracted, so they
   sit inside the residual. The arm charges them again as posts
   (cold_tier_shm 38.97 + arena 5.75): 44.72 GiB twice.
2. ``price`` builds ``arm.terms`` as a fresh literal without 66c46ba763's two
   measured shmem posts (arena_census_excess + unposted_shm, NF 4.90), while
   ``census_now_gib`` netted them out of the floor: charged at both moments,
   missing from the run peak.

Fixing only (1) predicts 76.46 (under the metal); only (2), 106; both: 81.36.
"""
from __future__ import annotations

import csv
import json
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import host_census, host_ledger
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GIB = int(host_ledger.GIB)
EV = "/spinning/docker-acceptance/nf/evidence"
REC = f"{EV}/weg2_measured_record.json"
CENSUS = f"{EV}/host_census_record.json"
TAG_1559 = "dkrnfh91dprsavisadoptstcutvsyncodx2bswre2cutz30y2bar1dauer09291559"
MEMTS_1559 = f"{EV}/docker_{TAG_1559}/memts_weg2_{TAG_1559}.csv"
HARD_BOUND = 82.53  # 09291634 W21 line: reap 84.00 - margin 1.47


def _entry(resid, instrument=None, census=0.0):
    e = {"run_residual_gib": resid, "sampled_at_flip_epoch": 0, "boot_tag": "b", "pids": [1],
         "at": "t", "residual_census_gib": census}
    if instrument is not None:
        e["image_instrument"] = instrument
    return {"D": e}


class TestNamedShmOnce(CustomTestCase):
    def test_pss_record_gives_store_and_arena_back(self):
        o, src = host_ledger.run_origin_gib(0.79, _entry(39.65, "pss_anon_shared"),
                                            census_now_gib=14.30, named_shm_now_gib=44.72)
        self.assertAlmostEqual(o, 0.79, places=2)  # 39.65 - 14.30 - 44.72 < launch
        self.assertIn("44.72 store+arena+L3-index", src)

    def test_rssshmem_record_already_subtracted_them(self):
        o, _ = host_ledger.run_origin_gib(0.79, _entry(39.65), census_now_gib=14.30, named_shm_now_gib=44.72)
        self.assertAlmostEqual(o, 25.35, places=2)

    def test_the_sum_names_the_three_posts(self):
        self.assertEqual(host_ledger.named_shm_posts_gib(
            {"cold_tier_shm_gib": 38.97, "arena_gib": 5.75, "l3_index_gib": 0.10, "rings_gib": 9.0}), 44.82)

    def test_arm_terms_carry_the_measured_shm_posts(self):
        census = {"arena_measured_gib": 7.0, "other_tmpfs_gib": 1.0, "unbooked_shm_gib": 3.0}
        a = host_ledger.price(int(125.70 * GIB), int(100.50 * GIB), 1, 600, s_gb_d=4,
                              ring_absent_by_design=True, cg_current_bytes=int(1.09 * GIB),
                              reclaimable_bytes=int(0.29 * GIB), cg_ceiling_bytes=84 * GIB,
                              arena_gib=5.75, staging_gb=0.0494, anchor_mib=101, census=census)
        self.assertAlmostEqual(a.terms["arena_census_excess_gib"], 1.25, places=6)
        self.assertGreater(a.terms["unposted_shm_gib"], 0.0)
        self.assertIn("unposted_shm=", host_ledger.arm_terms_line(a))


class TestNfReplay(CustomTestCase):
    @unittest.skipUnless(all(os.path.exists(p) for p in (REC, CENSUS, MEMTS_1559)), "NF evidence absent")
    def test_nf_z30y3b_replay_brackets_the_measured_peak(self):
        """Operator 29.09., two-sided, REAL records: the 09291634 arm replayed
        from its TERMS/ARM/COLD-TIER/L3 lines (heaps 17.23, anchors 0.40, rings
        0.22, arena 5.75, d_draft_host 1587 MiB, cold_tier 38.97, l3 0.10,
        ratchet 1.84, Sigma H 0, launch 0.80), the real census record and the
        real 09291559 residual record. Measured: memory.peak 80.30 GiB of the
        SAME form (z30y2 09291559, arena 4 GiB, before its 16:18Z wedge) --
        raw memory.current, an UPPER bound of the non-reclaimable level, so
        ``>=`` is the conservative side. Margin 1.25 GiB = memory.peak spread of
        today's good boots with this store (z30x2 09291210..09291248:
        80.08..81.06 = 0.98) + one 20-s memts step (0.25). Red on c7b12134cb:
        101.02 (upper); with the floor fix alone 76.46 (lower)."""
        census = host_census.ledger_terms(next(iter(json.load(open(CENSUS)).values())))
        rec = host_ledger.read_measured_record(REC, boot_tag=TAG_1559)
        ratchet = host_ledger.FlipRatchet(per_flip_gib=1.84, flips_priced=1, source="09291559 FLIP",
                                          from_record=True)
        a = host_ledger.price(int(125.70 * GIB), int(100.50 * GIB), 1, 600, s_gb_d=4,
                              ring_absent_by_design=True, cg_current_bytes=int(1.09 * GIB),
                              reclaimable_bytes=int(0.29 * GIB), cg_ceiling_bytes=84 * GIB,
                              measured_record=rec, flip_ratchet=ratchet, arena_gib=5.75,
                              staging_gb=0.0494, anchor_mib=101, d_draft_host_gib=1587 / 1024,
                              l3_index_gib=0.10, cold_tier_shm_gib=38.97, census=census)
        self.assertAlmostEqual(a.terms["heaps_gib"], 17.23, places=1)  # the TERMS line
        self.assertAlmostEqual(a.terms["anchors_gib"], 0.40, places=2)
        self.assertAlmostEqual(a.terms["rings_gib"], 0.22, places=2)
        with open(MEMTS_1559) as f:
            peak = max(int(r["cg_peak_b"]) for r in csv.DictReader(f)
                       if r["ts_utc"] < "2026-09-29T16:18:30Z") / GIB
        self.assertAlmostEqual(peak, 80.30, places=1)
        predicted = a.predicted_run_peak_gib()
        self.assertGreaterEqual(predicted, peak, (predicted, a.terms["run_origin_source"]))
        self.assertLessEqual(predicted, peak + 1.25, (predicted, a.terms["run_origin_source"]))
        self.assertLess(predicted, HARD_BOUND)  # the known good form is fundable again


if __name__ == "__main__":
    unittest.main()
