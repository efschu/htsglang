# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""Unit tests for the persistent HiCache energy-saved accumulator (#147).

No GPU, no server boot — pure offline accounting over already-collected metrics.
Covers the four hard user rules:

  * NO fetch/paging energy deducted (saved = recovered * J/prefill-token, exact),
  * a von–bis BAND (min/max measured J/prefill-token across buckets), never a
    point,
  * PERSISTENT per (model, config_label) — accumulates across a save/load cycle,
  * kWh = J / 3.6e6 ; ct = kWh * price,

plus the recovered-token metric parsing (HiCache RAM/disk tiers only), the
counter delta/reset guard, provenance discipline (measured band never clobbered
by an estimate), the grand-total band, and the webui GET/POST payloads.
"""

import json
import os
import tempfile
import unittest

from sglang.srt.planner.hicache_savings import (
    JOULES_PER_KWH,
    HiCacheSavingRecord,
    HiCacheSavingsStore,
    band_from_buckets,
    cached_tokens_by_source,
    hicache_recovered_from_metrics,
)
from sglang.srt.planner.webui import hicache_saved_read, hicache_saved_record


# ---------------------------------------------------------------------------
# Recovered-prefill-token metric (sglang:cached_tokens_total by cache_source).
# ---------------------------------------------------------------------------

_METRICS = (
    "# HELP sglang:cached_tokens_total foo\n"
    "# TYPE sglang:cached_tokens_total counter\n"
    'sglang:cached_tokens_total{model="m",cache_source="device"} 900\n'
    'sglang:cached_tokens_total{model="m",cache_source="host"} 100\n'
    'sglang:cached_tokens_total{model="m",cache_source="host"} 25\n'
    'sglang:cached_tokens_total{model="m",cache_source="storage_file"} 40\n'
    "sglang:prompt_tokens_total 5000\n"
)


class TestRecoveredTokenMetric(unittest.TestCase):
    def test_parse_sums_per_source(self):
        by = cached_tokens_by_source(_METRICS)
        self.assertAlmostEqual(by["device"], 900.0)
        self.assertAlmostEqual(by["host"], 125.0)   # two host label sets summed
        self.assertAlmostEqual(by["storage_file"], 40.0)
        self.assertNotIn("prompt", " ".join(by))    # only cached_tokens lines

    def test_hicache_recovery_excludes_device_tier(self):
        # HiCache = RAM(host) + disk(storage). The on-GPU radix (device) tier is
        # NOT the RAM/disk offload and must not be counted as saved.
        rec = hicache_recovered_from_metrics(_METRICS)
        self.assertAlmostEqual(rec, 125.0 + 40.0)   # host + storage only

    def test_bare_total_fallback_excluded(self):
        # When the per-source breakdown is unavailable the counter carries a
        # "total" label (or none) that lumps in the device tier -> excluded.
        text = "sglang:cached_tokens_total 777\n"
        self.assertEqual(cached_tokens_by_source(text), {"total": 777.0})
        self.assertEqual(hicache_recovered_from_metrics(text), 0.0)


# ---------------------------------------------------------------------------
# Band (min/max across buckets — never a scalar).
# ---------------------------------------------------------------------------


class TestBand(unittest.TestCase):
    def test_min_max_across_buckets(self):
        self.assertEqual(band_from_buckets({1: 2.0, 8: 3.5, 32: 2.9}), (2.0, 3.5))

    def test_empty_or_none_is_none(self):
        self.assertIsNone(band_from_buckets(None))
        self.assertIsNone(band_from_buckets({}))
        self.assertIsNone(band_from_buckets({1: None}))


# ---------------------------------------------------------------------------
# The accumulator record.
# ---------------------------------------------------------------------------


class TestSavingRecord(unittest.TestCase):
    def _rec(self):
        r = HiCacheSavingRecord(model="Qwen3.6-27B", config_label="MTP+adaptive")
        r.set_band(2.0, 3.0)   # J/prefill-token band
        return r

    def test_no_fetch_energy_deducted(self):
        # RULE 1: saved_J is EXACTLY recovered * J/tok on both band edges — no
        # subtraction for the RAM/disk fetch (that power is sunk cost).
        r = self._rec()
        r.add_tokens(1_000_000)
        j = r.saved_joules_band()
        self.assertEqual(j, (1_000_000 * 2.0, 1_000_000 * 3.0))
        kwh = r.saved_kwh_band()
        self.assertEqual(kwh, (2.0e6 / JOULES_PER_KWH, 3.0e6 / JOULES_PER_KWH))

    def test_kwh_and_ct_conversion(self):
        r = self._rec()
        r.add_tokens(3_600_000)  # 3.6e6 tok -> at 2 J/tok = 7.2e6 J = 2 kWh
        kwh = r.saved_kwh_band()
        self.assertAlmostEqual(kwh[0], 2.0)
        self.assertAlmostEqual(kwh[1], 3.0)
        ct = r.saved_ct_band(30.0)
        self.assertAlmostEqual(ct[0], 60.0)   # 2 kWh * 30 ct
        self.assertAlmostEqual(ct[1], 90.0)

    def test_band_absent_yields_none(self):
        r = HiCacheSavingRecord(model="m", config_label="c")
        r.add_tokens(1000)
        self.assertIsNone(r.saved_kwh_band())   # no measured band -> no fabrication
        self.assertIsNone(r.to_view(30.0)["saved_ct_band"])

    def test_counter_snapshot_delta(self):
        r = self._rec()
        self.assertEqual(r.record_counter_snapshot(100), 100)  # first snapshot
        self.assertEqual(r.record_counter_snapshot(160), 60)   # +60 delta
        self.assertEqual(r.recovered_prefill_tokens, 160)

    def test_counter_reset_accumulates_not_double_counts(self):
        r = self._rec()
        r.record_counter_snapshot(500)
        # server restart: counter drops -> the new value is the post-reset delta
        added = r.record_counter_snapshot(30)
        self.assertEqual(added, 30)
        self.assertEqual(r.recovered_prefill_tokens, 530)

    def test_negative_inputs_rejected(self):
        r = self._rec()
        with self.assertRaises(ValueError):
            r.add_tokens(-1)
        with self.assertRaises(ValueError):
            r.record_counter_snapshot(-5)


class TestProvenanceDiscipline(unittest.TestCase):
    def test_estimate_never_overwrites_measured(self):
        r = HiCacheSavingRecord(model="m", config_label="c")
        self.assertTrue(r.set_band(2.0, 3.0, provenance="measured"))
        self.assertFalse(r.set_band(1.0, 9.0, provenance="estimate"))
        self.assertEqual((r.j_per_prefill_token_lo, r.j_per_prefill_token_hi),
                         (2.0, 3.0))
        self.assertEqual(r.band_provenance, "measured")

    def test_measured_can_replace_measured(self):
        r = HiCacheSavingRecord(model="m", config_label="c")
        r.set_band(2.0, 3.0, provenance="measured")
        self.assertTrue(r.set_band(2.5, 4.0, provenance="measured"))
        self.assertEqual(r.j_per_prefill_token_hi, 4.0)

    def test_hi_below_lo_rejected(self):
        r = HiCacheSavingRecord(model="m", config_label="c")
        with self.assertRaises(ValueError):
            r.set_band(3.0, 2.0)


# ---------------------------------------------------------------------------
# The persistent store.
# ---------------------------------------------------------------------------


class TestStore(unittest.TestCase):
    def test_persistence_round_trip_accumulates(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "hicache_savings.json")
            s1 = HiCacheSavingsStore.load(path)          # empty (no file yet)
            r = s1.get_or_create("Qwen", "MTP")
            r.set_band(2.0, 3.0)
            r.add_tokens(1000)
            s1.save(path)
            # a later SESSION: load and add more -> total GROWS, not resets
            s2 = HiCacheSavingsStore.load(path)
            r2 = s2.get_or_create("Qwen", "MTP")
            self.assertEqual(r2.recovered_prefill_tokens, 1000)   # persisted
            self.assertEqual((r2.j_per_prefill_token_lo, r2.j_per_prefill_token_hi),
                             (2.0, 3.0))
            r2.add_tokens(500)
            s2.save(path)
            s3 = HiCacheSavingsStore.load(path)
            self.assertEqual(s3.get("Qwen", "MTP").recovered_prefill_tokens, 1500)

    def test_load_missing_file_is_empty(self):
        self.assertEqual(len(HiCacheSavingsStore.load("/no/such/file.json")), 0)

    def test_load_corrupt_file_is_empty(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bad.json")
            with open(path, "w") as f:
                f.write("{ not json ]")
            self.assertEqual(len(HiCacheSavingsStore.load(path)), 0)

    def test_grand_total_band_sums_only_records_with_a_band(self):
        s = HiCacheSavingsStore()
        a = s.get_or_create("A", "c")
        a.set_band(2.0, 3.0)
        a.add_tokens(3_600_000)         # -> 2..3 kWh
        b = s.get_or_create("B", "c")
        b.set_band(1.0, 1.0)
        b.add_tokens(3_600_000)         # -> 1..1 kWh
        c = s.get_or_create("C", "c")   # no band -> contributes nothing
        c.add_tokens(9_999)
        gt = s.grand_total_saved_kwh_band()
        self.assertAlmostEqual(gt[0], 3.0)   # 2 + 1
        self.assertAlmostEqual(gt[1], 4.0)   # 3 + 1
        ct = s.grand_total_saved_ct_band(30.0)
        self.assertAlmostEqual(ct[0], 90.0)
        self.assertAlmostEqual(ct[1], 120.0)

    def test_view_schema(self):
        s = HiCacheSavingsStore()
        r = s.get_or_create("A", "c")
        r.set_band(2.0, 3.0)
        r.add_tokens(3_600_000)
        v = s.to_view(30.0)
        self.assertIn("records", v)
        self.assertIn("grand_total", v)
        self.assertEqual(v["price_ct_per_kwh"], 30.0)
        rv = v["records"][0]
        self.assertEqual(rv["j_per_prefill_token_band"], [2.0, 3.0])
        self.assertEqual(len(rv["saved_kwh_band"]), 2)   # a band, not a scalar


# ---------------------------------------------------------------------------
# webui /api/hicache_saved GET (read) + POST (record).
# ---------------------------------------------------------------------------


class TestWebuiPayloads(unittest.TestCase):
    def test_record_manual_then_read(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "hicache_savings.json")
            # POST a manual delta (no server, no measured band available)
            out = hicache_saved_record({
                "hicache_store": path,
                "model": "ZZZ-Nonexistent-Model-99B",
                "config_label": "unit-test-cfg",
                "recovered_tokens": 1000,
                "results_store": "/no/such/store.jsonl",  # -> no band attached
            })
            self.assertTrue(out["ok"])
            self.assertEqual(out["recorded_delta_tokens"], 1000)
            # a SECOND POST accumulates (persisted across the load/save)
            out2 = hicache_saved_record({
                "hicache_store": path,
                "model": "ZZZ-Nonexistent-Model-99B",
                "config_label": "unit-test-cfg",
                "recovered_tokens": 500,
            })
            self.assertEqual(out2["record"]["recovered_prefill_tokens"], 1500)
            # GET reads the persisted total back
            got = hicache_saved_read({"hicache_store": path, "price_ct_per_kwh": 30})
            self.assertTrue(got["ok"])
            self.assertEqual(got["grand_total"]["recovered_prefill_tokens"], 1500)
            # no measured band -> saving band absent, not fabricated
            self.assertIsNone(got["records"][0]["saved_kwh_band"])
            # and the file really is on disk (persistent)
            with open(path) as f:
                self.assertEqual(len(json.load(f)), 1)

    def test_record_requires_model(self):
        out = hicache_saved_record({"recovered_tokens": 10})
        self.assertFalse(out["ok"])
        self.assertIn("model", out["error"])

    def test_record_requires_a_signal(self):
        with tempfile.TemporaryDirectory() as d:
            out = hicache_saved_record({
                "hicache_store": os.path.join(d, "s.json"),
                "model": "m",
            })
            self.assertFalse(out["ok"])
            self.assertIn("target", out["error"])

    def test_read_empty_store(self):
        got = hicache_saved_read({"hicache_store": "/no/such/file.json"})
        self.assertTrue(got["ok"])
        self.assertEqual(got["records"], [])
        self.assertIsNone(got["grand_total"]["saved_kwh_band"])


if __name__ == "__main__":
    unittest.main()
