"""NF1d (z30y3d @ 70288e7921, 17:13:36Z): W21 run peak 84.16 vs 83.44.

Where the +2.80 over NF1c's 81.36 came from: NOT 9fcf7bef97 (the footprint
moves only the margin, 82.53 -> 83.44) and NOT a changed record -- the
COLD-TIER RE-PRICE: the written expert map has 368 slots, the dry-built one
344, so cold_tier_shm 38.97 -> 41.69 (+2.72) plus origin 0.80 -> 0.87. NF1c
never got that far (W87 at the dry map). 09291559 ran THE SAME written map
(368 slots, store class 41.69), so cf6f2108fb's "81.36 in [80.30; 81.55]"
graded a 38.97 prediction against a 41.69 boot. Graded right: 84.09 vs 80.32.

The shmem half of that over-prediction, measured: the ledger claims 56.80 GiB
of shmem for 09291559's own arm (store 41.69 + arena 5.75 + 0.03 excess + l3
0.11 + seq_ring 1.50 + sidecar 0.26 + hand-off 0.23 + other_tmpfs/ungebucht
7.23), host Shmem never exceeded 54.11 in that boot (memts, 5 s, whole boot;
host >= cgroup). The census record max-merges each field ON ITS OWN: store
41.69 is 09291559's, ungebucht 6.26 is z30w's (09:12Z creep, store 39.20) --
a sum of maxima from instants with different stores (z30w cg shmem 53.36 ->
09291559 <= 54.11: +0.75 for +2.49 of store).
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

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

GIB = int(host_ledger.GIB)
EV = "/spinning/docker-acceptance/nf/evidence"
REC = f"{EV}/weg2_measured_record.json"
CENSUS = f"{EV}/host_census_record.json"
TAG_1559 = "dkrnfh91dprsavisadoptstcutvsyncodx2bswre2cutz30y2bar1dauer09291559"
MEMTS_1559 = f"{EV}/docker_{TAG_1559}/memts_weg2_{TAG_1559}.csv"
MODEL = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"
STORE_WRITTEN = 41.69   # COLD-TIER post, written map 368 x 48 x 2.417 MiB
ARENA_PRICED = 5.75     # 09291559's and NF1c's ARM line arena=
REFUSED = (host_ledger.PdFlipHostLedgerRefused, host_ledger.PdFlipHostRunPeakRefused)
EVID = all(os.path.exists(p) for p in (REC, CENSUS, MEMTS_1559))


def _memts_shmem_max_gib() -> float:
    with open(MEMTS_1559) as f:
        return max(int(r["shmem_kb"]) for r in csv.DictReader(f)) / (1024 * 1024)


def _entry(seeded: bool):
    e = dict(next(iter(json.load(open(CENSUS)).values())))
    for k in ("shm_total_max_gib", "shm_total_store_gib", "shm_total_arena_gib", "shm_total_source"):
        e.pop(k, None)   # the test decides the seed, not the file's current state
    if seeded:
        e["shm_total_max_gib"] = _memts_shmem_max_gib()
        e["shm_total_store_gib"] = float(e["shm_classes_gib"]["store"])
        e["shm_total_arena_gib"] = ARENA_PRICED
    return e


def _choose(seeded: bool, store=STORE_WRITTEN):
    """The launcher's path: reference_model_verdict + footprint_key on the real
    checkpoint, the real census record entry, the real 09291559 record."""
    ref_ok, ref_why = pdflip_form.reference_model_verdict(MODEL, host_ledger.REFERENCE_MODEL)
    fp = (pdflip_form.footprint_key(MODEL)[0] or "") if ref_ok is False else ""
    rec = host_ledger.read_measured_record(REC, boot_tag=TAG_1559)
    ratchet = host_ledger.FlipRatchet(per_flip_gib=1.84, flips_priced=1,
                                      source="09291559 FLIP", from_record=True)
    return host_ledger.choose(
        int(125.70 * GIB), int(99.11 * GIB), arms=[(1, 600)], s_gb_d=4,
        ring_absent_by_design=True, cg_current_bytes=int(1.26 * GIB),
        reclaimable_bytes=int(0.39 * GIB), cg_ceiling_bytes=84 * GIB, measured_record=rec,
        flip_ratchet=ratchet, arena_gib=ARENA_PRICED, staging_gb=0.0494, anchor_mib=101,
        d_draft_host_gib=1587 / 1024, l3_index_gib=0.10, cold_tier_shm_gib=store,
        census=host_census.ledger_terms(_entry(seeded)),
        reference_model_ok=ref_ok, reference_model_why=ref_why, model_footprint=fp,
    )


class TestMergeKeepsThePeakOfTheSum(CustomTestCase):
    def test_one_instant_not_per_field(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "r.json")
            host_census.merge_into_record(p, "k", {"cg_shmem_gib": 53.36, "unattributed_shm_gib": 6.26,
                                                   "shm_classes_gib": {"store": 39.20, "arena_booked": 5.0}})
            e = host_census.merge_into_record(p, "k", {"cg_shmem_gib": 54.00, "unattributed_shm_gib": 3.0,
                                                       "shm_classes_gib": {"store": 41.69, "arena_booked": 5.0}})
            self.assertEqual(e["unattributed_shm_gib"], 6.26)          # per field: max
            self.assertEqual(e["shm_classes_gib"]["store"], 41.69)
            self.assertEqual(e["shm_total_max_gib"], 54.00)            # the sum: one instant
            self.assertEqual(e["shm_total_store_gib"], 41.69)
            e = host_census.merge_into_record(p, "k", {"cg_shmem_gib": 50.0, "shm_classes_gib": {}})
            self.assertEqual(e["shm_total_max_gib"], 54.00)            # never lowered

    def test_no_field_is_byte_identical(self):
        t = {"cold_tier_shm_gib": 41.69, "arena_gib": 5.75, "anchors_gib": 0.4}
        c = {"arena_measured_gib": 5.78, "other_tmpfs_gib": 0.97, "unbooked_shm_gib": 6.26}
        a = host_ledger.census_shm_posts(t, c)
        self.assertNotIn("unposted_shm_trim_gib", a)
        self.assertAlmostEqual(a["unposted_shm_gib"], 0.97 + 6.26 - 0.4, places=6)


@unittest.skipUnless(EVID and os.path.isdir(MODEL), "NF evidence/checkpoint absent")
class TestNf1dReplay(CustomTestCase):
    def test_the_written_map_is_nf1ds_refusal(self):
        with self.assertRaises(REFUSED) as cm:
            _choose(seeded=False)
        # NF1c's cause-13 line: 84.16 at its live origin 0.87 (rounding here: 84.15)
        self.assertRegex(str(cm.exception), r"run_peak=84\.1[56] GiB vs hard bound 83\.44")
        self.assertIn("binding: RUN PEAK", str(cm.exception))

    def test_capped_at_the_measured_instant_funds_and_stays_above_the_metal(self):
        arm, _h, lines = _choose(seeded=True)
        peak = arm.predicted_run_peak_gib()
        self.assertLess(peak + host_ledger.RATE_LATCH_CUSHION_FLOOR_GIB, 83.44)
        self.assertGreaterEqual(peak, 80.32)                            # two-sided: >= the metal
        self.assertAlmostEqual(arm.terms["unposted_shm_trim_gib"], 56.80 - 54.11, delta=0.03)
        self.assertIn("unposted_shm_trim=-", host_ledger.arm_terms_line(arm))
        self.assertIn("run_peak=81.47 GiB vs hard bound 83.44", "\n".join(lines))

    def test_a_bigger_store_is_not_trimmed_away(self):
        # +3 GiB of store beyond the measured instant: the cap grows with it
        arm0, _h, _l = _choose(seeded=True)
        with self.assertRaises(REFUSED) as cm:
            _choose(seeded=True, store=STORE_WRITTEN + 3.0)
        self.assertIn("run_peak=84.47 GiB", str(cm.exception))  # 81.47 + 3.00, untrimmed
        self.assertAlmostEqual(arm0.terms["unposted_shm_trim_gib"], 2.69, delta=0.03)


if __name__ == "__main__":
    unittest.main()


R27B_CENSUS = "/spinning/docker-acceptance/27b/evidence/host_census_record.json"
MODEL_27B = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-gdncov-vocabembed"


def _fixture(d: str, with_memts: bool, record_src: str = CENSUS) -> str:
    """A tmp evidence dir: the real record (field stripped) + optionally the
    real 09291559 memts CSV at the path the boot writes it to."""
    import shutil

    data = json.load(open(record_src))
    for e in data.values():
        for k in ("shm_total_max_gib", "shm_total_store_gib", "shm_total_arena_gib", "shm_total_source"):
            e.pop(k, None)
    rec = os.path.join(d, host_census.RECORD_NAME)
    json.dump(data, open(rec, "w"), indent=1, sort_keys=True)
    if with_memts:
        os.makedirs(os.path.join(d, f"docker_{TAG_1559}"))
        shutil.copy(MEMTS_1559, os.path.join(d, f"docker_{TAG_1559}", os.path.basename(MEMTS_1559)))
    return rec


@unittest.skipUnless(EVID and os.path.isdir(MODEL), "NF evidence/checkpoint absent")
class TestLauncherBackfill(CustomTestCase):
    """Operator 29.09. ~18:00Z: no hand-seeded record. The launcher rebuilds the
    peak of the SUM from the key's own memts, writes it back with provenance and
    prices with it; no memts -> today's refusing sum; 27B untouched."""

    def _price(self, entry):
        ref_ok, ref_why = pdflip_form.reference_model_verdict(MODEL, host_ledger.REFERENCE_MODEL)
        rec = host_ledger.read_measured_record(REC, boot_tag=TAG_1559)
        ratchet = host_ledger.FlipRatchet(per_flip_gib=1.84, flips_priced=1,
                                          source="09291559 FLIP", from_record=True)
        return host_ledger.choose(
            int(125.70 * GIB), int(99.11 * GIB), arms=[(1, 600)], s_gb_d=4,
            ring_absent_by_design=True, cg_current_bytes=int(1.26 * GIB),
            reclaimable_bytes=int(0.39 * GIB), cg_ceiling_bytes=84 * GIB, measured_record=rec,
            flip_ratchet=ratchet, arena_gib=ARENA_PRICED, staging_gb=0.0494, anchor_mib=101,
            d_draft_host_gib=1587 / 1024, l3_index_gib=0.10, cold_tier_shm_gib=STORE_WRITTEN,
            census=host_census.ledger_terms(entry), reference_model_ok=ref_ok,
            reference_model_why=ref_why, model_footprint=pdflip_form.footprint_key(MODEL)[0] or "")

    def test_backfill_from_memts_funds_and_is_written_back(self):
        import tempfile

        from flliper.srt.pdflip import launcher
        key = next(iter(json.load(open(CENSUS))))
        with tempfile.TemporaryDirectory() as d:
            rec = _fixture(d, with_memts=True)
            entry, line = launcher.census_entry_for_pricing(rec, key, model=MODEL, evidence_dir=d)
            self.assertAlmostEqual(entry["shm_total_max_gib"], 54.11, places=2)
            self.assertIn("BACKFILLED", line)
            self.assertIn("backfill from memts", json.load(open(rec))[key]["shm_total_source"])
            arm, _h, lines = self._price(entry)
            self.assertLessEqual(arm.predicted_run_peak_gib() + host_ledger.RATE_LATCH_CUSHION_FLOOR_GIB, 83.44)
            self.assertIn("run_peak=81.47 GiB vs hard bound 83.44", "\n".join(lines))
            # second load reads the written field, no second backfill
            _e2, line2 = launcher.census_entry_for_pricing(rec, key, model=MODEL, evidence_dir=d)
            self.assertEqual(line2, "")

    def test_without_memts_the_refusing_sum_stays(self):
        import tempfile

        from flliper.srt.pdflip import launcher
        key = next(iter(json.load(open(CENSUS))))
        with tempfile.TemporaryDirectory() as d:
            rec = _fixture(d, with_memts=False)
            entry, line = launcher.census_entry_for_pricing(rec, key, model=MODEL, evidence_dir=d)
            self.assertIsNone(entry.get("shm_total_max_gib"))
            self.assertIn("NOT BACKFILLED", line)
            with self.assertRaises(REFUSED) as cm:
                self._price(entry)
            self.assertRegex(str(cm.exception), r"run_peak=84\.1[56] GiB vs hard bound 83\.44")

    @unittest.skipUnless(os.path.exists(R27B_CENSUS) and os.path.isdir(MODEL_27B), "27B record/checkpoint absent")
    def test_27b_record_is_never_touched(self):
        import tempfile

        from flliper.srt.pdflip import launcher
        key = next(iter(json.load(open(R27B_CENSUS))))
        with tempfile.TemporaryDirectory() as d:
            rec = _fixture(d, with_memts=True, record_src=R27B_CENSUS)
            before = open(rec, "rb").read()
            _e, line = launcher.census_entry_for_pricing(rec, key, model=MODEL_27B, evidence_dir=d)
            self.assertEqual(line, "")
            self.assertEqual(open(rec, "rb").read(), before)
