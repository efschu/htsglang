# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""Unit tests for the persistent Joules-per-token counter (jtok_counter.py).

No GPU, no server boot -- pure offline accounting over already-computed
numbers (the module never touches NVML/network itself; callers hand it
pre-measured joules/tokens/rates). Covers:

  * persistence round-trip (save/load, JSON on disk, corrupt-file safety),
  * prefill/decode kept as two fully separate accumulators,
  * reset (one counter, and reset-all), identity survives a reset,
  * the toggle (default OFF; toggled-off recording paths cost nothing --
    they must not even build a dict or touch the store),
  * the lane-list key dimension (today one lane, but never hardcoded to
    exactly one or two -- two different lane lists must not collide),
  * mixed-window handling (both phases active in one live-poll tick is
    tracked separately, never apportioned into the pure phase totals),
  * the harness-result path (phase-pure by construction, no mixed_* writes),
  * the schema-stamp write-guard (mirrors test_self_update's pattern).
"""

import json
import os
import tempfile
import unittest
from unittest import mock

from sglang.srt.planner import jtok_counter as jc
from sglang.srt.planner import self_update as su


# ---------------------------------------------------------------------------
# Record-level accumulation.
# ---------------------------------------------------------------------------
class TestRecord(unittest.TestCase):
    def test_prefill_and_decode_are_independent(self):
        rec = jc.JtokCounterRecord(model="m", config_label="c", lanes=["local"])
        rec.add_prefill(1000.0, 500.0, source=jc.SOURCE_HARNESS)
        rec.add_decode(300.0, 100.0, source=jc.SOURCE_HARNESS)
        self.assertEqual(rec.j_per_prefill_token(), 2.0)
        self.assertEqual(rec.j_per_decode_token(), 3.0)
        # Accumulating more of one phase must not move the other's number.
        rec.add_prefill(1000.0, 500.0, source=jc.SOURCE_HARNESS)
        self.assertEqual(rec.j_per_prefill_token(), 2.0)  # weighted avg unchanged
        self.assertEqual(rec.j_per_decode_token(), 3.0)

    def test_no_tokens_yields_none_not_zero(self):
        rec = jc.JtokCounterRecord(model="m", config_label="c", lanes=["local"])
        self.assertIsNone(rec.j_per_prefill_token())
        self.assertIsNone(rec.j_per_decode_token())

    def test_zero_token_contribution_is_a_noop(self):
        rec = jc.JtokCounterRecord(model="m", config_label="c", lanes=["local"])
        rec.add_prefill(50.0, 0.0, source=jc.SOURCE_HARNESS)
        self.assertIsNone(rec.j_per_prefill_token())
        self.assertEqual(rec.prefill_joules, 0.0)

    def test_negative_inputs_rejected(self):
        rec = jc.JtokCounterRecord(model="m", config_label="c", lanes=["local"])
        with self.assertRaises(ValueError):
            rec.add_prefill(-1.0, 10.0, source=jc.SOURCE_HARNESS)
        with self.assertRaises(ValueError):
            rec.add_decode(1.0, -10.0, source=jc.SOURCE_HARNESS)

    def test_mixed_kept_separate_from_pure_totals(self):
        rec = jc.JtokCounterRecord(model="m", config_label="c", lanes=["local"])
        rec.add_prefill(100.0, 50.0, source=jc.SOURCE_LIVE_POLL)
        rec.add_mixed(90.0, 10.0, 20.0, source=jc.SOURCE_LIVE_POLL)
        # The pure prefill number is untouched by the mixed contribution.
        self.assertEqual(rec.j_per_prefill_token(), 2.0)
        self.assertIsNone(rec.j_per_decode_token())  # no PURE decode ever added
        self.assertEqual(rec.mixed_windows, 1)
        self.assertAlmostEqual(rec.mixed_j_per_token_blended(), 90.0 / 30.0)

    def test_reset_zeros_accumulators_keeps_identity(self):
        rec = jc.JtokCounterRecord(model="m", config_label="c", lanes=["local"])
        rec.add_prefill(100.0, 50.0, source=jc.SOURCE_HARNESS)
        rec.add_decode(60.0, 20.0, source=jc.SOURCE_HARNESS)
        rec.add_mixed(10.0, 1.0, 1.0, source=jc.SOURCE_LIVE_POLL)
        rec.reset()
        self.assertIsNone(rec.j_per_prefill_token())
        self.assertIsNone(rec.j_per_decode_token())
        self.assertEqual(rec.mixed_windows, 0)
        self.assertEqual(rec.mixed_j_per_token_blended(), None)
        self.assertEqual(rec.model, "m")
        self.assertEqual(rec.config_label, "c")
        self.assertEqual(rec.lanes, ["local"])

    def test_sources_tracked_without_duplicates(self):
        rec = jc.JtokCounterRecord(model="m", config_label="c", lanes=["local"])
        rec.add_prefill(1.0, 1.0, source=jc.SOURCE_HARNESS)
        rec.add_prefill(1.0, 1.0, source=jc.SOURCE_HARNESS)
        rec.add_decode(1.0, 1.0, source=jc.SOURCE_LIVE_POLL)
        self.assertEqual(sorted(rec.sources), sorted([jc.SOURCE_HARNESS, jc.SOURCE_LIVE_POLL]))

    def test_provenance_is_always_measured(self):
        rec = jc.JtokCounterRecord(model="m", config_label="c", lanes=["local"])
        rec.add_prefill(1.0, 1.0, source=jc.SOURCE_HARNESS)
        self.assertEqual(rec.to_view()["provenance"], jc.MEASURED_PROVENANCE)


# ---------------------------------------------------------------------------
# Lane-list key dimension (dual-group readiness).
# ---------------------------------------------------------------------------
class TestLaneKey(unittest.TestCase):
    def test_different_lanes_are_different_records(self):
        store = jc.JtokCounterStore(enabled=True)
        a = store.get_or_create("m", "c", ["rigA"])
        b = store.get_or_create("m", "c", ["rigB"])
        self.assertIsNot(a, b)
        self.assertEqual(len(store), 2)

    def test_multi_element_lane_list_is_one_key(self):
        # A lane can itself be a LIST spanning several cards/rigs (mirrors
        # rig_coupling.LaneCandidate: "a lane is a list"); the counter key
        # must treat the whole ordered list as one identity, not explode it.
        store = jc.JtokCounterStore(enabled=True)
        a = store.get_or_create("m", "c", ["rigA", "rigB"])
        b = store.get_or_create("m", "c", ["rigA", "rigB"])
        self.assertIs(a, b)
        c = store.get_or_create("m", "c", ["rigB", "rigA"])  # order matters
        self.assertIsNot(a, c)

    def test_default_single_lane_not_hardcoded_length(self):
        # A caller passing a single-element list today must not break if a
        # future caller passes more elements for the same (model, config).
        store = jc.JtokCounterStore(enabled=True)
        store.get_or_create("m", "c", ["local"])
        store.get_or_create("m", "c", ["local", "second"])
        self.assertEqual(len(store), 2)


# ---------------------------------------------------------------------------
# Store: reset-one / reset-all / grand total.
# ---------------------------------------------------------------------------
class TestStoreReset(unittest.TestCase):
    def setUp(self):
        self.store = jc.JtokCounterStore(enabled=True)
        r1 = self.store.get_or_create("m1", "cfgA", ["local"])
        r1.add_prefill(100.0, 50.0, source=jc.SOURCE_HARNESS)
        r2 = self.store.get_or_create("m2", "cfgB", ["local"])
        r2.add_decode(60.0, 20.0, source=jc.SOURCE_HARNESS)

    def test_reset_one(self):
        ok = self.store.reset_one("m1", "cfgA", ["local"])
        self.assertTrue(ok)
        self.assertIsNone(self.store.get("m1", "cfgA", ["local"]).j_per_prefill_token())
        # the other record is untouched
        self.assertEqual(self.store.get("m2", "cfgB", ["local"]).j_per_decode_token(), 3.0)

    def test_reset_one_missing_key(self):
        self.assertFalse(self.store.reset_one("nope", "nope", ["local"]))

    def test_reset_all(self):
        n = self.store.reset_all()
        self.assertEqual(n, 2)
        for rec in self.store.records():
            self.assertIsNone(rec.j_per_prefill_token())
            self.assertIsNone(rec.j_per_decode_token())

    def test_grand_total(self):
        gt = self.store.grand_total()
        self.assertEqual(gt["prefill_tokens"], 50.0)
        self.assertEqual(gt["j_per_prefill_token"], 2.0)
        self.assertEqual(gt["decode_tokens"], 20.0)
        self.assertEqual(gt["j_per_decode_token"], 3.0)


# ---------------------------------------------------------------------------
# Persistence round-trip.
# ---------------------------------------------------------------------------
class TestPersistence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="jtok_")
        self.path = os.path.join(self.tmp, "jtok_counter.json")

    def test_save_load_round_trip(self):
        store = jc.JtokCounterStore(enabled=True)
        rec = store.get_or_create("Qwen3.6-27B", "no-MTP baseline", ["local"])
        rec.add_prefill(500.0, 100.0, source=jc.SOURCE_HARNESS)
        rec.add_decode(90.0, 30.0, source=jc.SOURCE_HARNESS)
        store.save(self.path)

        loaded = jc.JtokCounterStore.load(self.path)
        self.assertTrue(loaded.enabled)
        got = loaded.get("Qwen3.6-27B", "no-MTP baseline", ["local"])
        self.assertIsNotNone(got)
        self.assertEqual(got.j_per_prefill_token(), 5.0)
        self.assertEqual(got.j_per_decode_token(), 3.0)
        self.assertEqual(got.lanes, ["local"])

    def test_load_missing_file_is_empty_disabled(self):
        store = jc.JtokCounterStore.load(os.path.join(self.tmp, "nope.json"))
        self.assertEqual(len(store), 0)
        self.assertFalse(store.enabled)

    def test_load_corrupt_file_is_empty_not_a_crash(self):
        with open(self.path, "w") as f:
            f.write("{not json")
        store = jc.JtokCounterStore.load(self.path)
        self.assertEqual(len(store), 0)
        self.assertFalse(store.enabled)

    def test_atomic_write_no_tmp_left_behind(self):
        store = jc.JtokCounterStore(enabled=True)
        store.save(self.path)
        self.assertTrue(os.path.exists(self.path))
        self.assertFalse(os.path.exists(self.path + ".tmp"))

    def test_enabled_flag_persists_independent_of_records(self):
        store = jc.JtokCounterStore(enabled=True)
        store.save(self.path)
        reloaded = jc.JtokCounterStore.load(self.path)
        self.assertTrue(reloaded.enabled)
        reloaded.enabled = False
        reloaded.save(self.path)
        self.assertFalse(jc.JtokCounterStore.load(self.path).enabled)


# ---------------------------------------------------------------------------
# Toggle-off = null overhead, on the two recording entry points.
# ---------------------------------------------------------------------------
class TestToggleOff(unittest.TestCase):
    def test_default_is_disabled(self):
        self.assertFalse(jc.JtokCounterStore().enabled)

    def test_live_tick_disabled_is_noop(self):
        store = jc.JtokCounterStore(enabled=False)
        out = jc.record_live_tick(
            store, model="m", config_label="c", lanes=["local"],
            dt=2.0, prefill_tok_s=100.0, decode_tok_s=0.0, total_watts=300.0,
        )
        self.assertIsNone(out)
        self.assertEqual(len(store), 0)  # no record was even created

    def test_harness_result_disabled_is_noop(self):
        store = jc.JtokCounterStore(enabled=False)
        fake = mock.Mock(prompt_tokens=100, decode_tokens=50, n_requests=4,
                         prefill_joules=10.0, decode_joules=5.0)
        out = jc.record_harness_result(
            store, model="m", config_label="c", lanes=["local"],
            measurements=[fake],
        )
        self.assertEqual(out, [])
        self.assertEqual(len(store), 0)

    def test_live_tick_does_not_touch_store_when_disabled(self):
        # A stronger null-overhead check: even get_or_create must not run.
        store = jc.JtokCounterStore(enabled=False)
        with mock.patch.object(store, "get_or_create") as m:
            jc.record_live_tick(
                store, model="m", config_label="c", lanes=["local"],
                dt=2.0, prefill_tok_s=100.0, decode_tok_s=0.0, total_watts=300.0,
            )
            m.assert_not_called()


# ---------------------------------------------------------------------------
# record_live_tick: phase-pure attribution + mixed-window honesty + guards.
# ---------------------------------------------------------------------------
class TestLiveTick(unittest.TestCase):
    def setUp(self):
        self.store = jc.JtokCounterStore(enabled=True)

    def test_pure_prefill_window(self):
        jc.record_live_tick(
            self.store, model="m", config_label="c", lanes=["local"],
            dt=2.0, prefill_tok_s=100.0, decode_tok_s=0.0, total_watts=300.0,
        )
        rec = self.store.get("m", "c", ["local"])
        # joules = 300W * 2s = 600J; tokens = 100tok/s * 2s = 200 tok
        self.assertEqual(rec.j_per_prefill_token(), 3.0)
        self.assertIsNone(rec.j_per_decode_token())

    def test_pure_decode_window(self):
        jc.record_live_tick(
            self.store, model="m", config_label="c", lanes=["local"],
            dt=2.0, prefill_tok_s=0.0, decode_tok_s=50.0, total_watts=300.0,
        )
        rec = self.store.get("m", "c", ["local"])
        self.assertIsNone(rec.j_per_prefill_token())
        self.assertEqual(rec.j_per_decode_token(), 6.0)

    def test_mixed_window_not_apportioned_to_pure_buckets(self):
        jc.record_live_tick(
            self.store, model="m", config_label="c", lanes=["local"],
            dt=2.0, prefill_tok_s=100.0, decode_tok_s=50.0, total_watts=300.0,
        )
        rec = self.store.get("m", "c", ["local"])
        self.assertIsNone(rec.j_per_prefill_token())
        self.assertIsNone(rec.j_per_decode_token())
        self.assertEqual(rec.mixed_windows, 1)

    def test_idle_window_is_a_noop(self):
        out = jc.record_live_tick(
            self.store, model="m", config_label="c", lanes=["local"],
            dt=2.0, prefill_tok_s=0.0, decode_tok_s=0.0, total_watts=300.0,
        )
        self.assertIsNone(out)
        self.assertEqual(len(self.store), 0)

    def test_none_or_nonpositive_dt_is_a_noop(self):
        for dt in (None, 0.0, -1.0):
            out = jc.record_live_tick(
                self.store, model="m", config_label="c", lanes=["local"],
                dt=dt, prefill_tok_s=100.0, decode_tok_s=0.0, total_watts=300.0,
            )
            self.assertIsNone(out)
        self.assertEqual(len(self.store), 0)

    def test_repeated_pure_ticks_accumulate(self):
        for _ in range(3):
            jc.record_live_tick(
                self.store, model="m", config_label="c", lanes=["local"],
                dt=2.0, prefill_tok_s=100.0, decode_tok_s=0.0, total_watts=300.0,
            )
        rec = self.store.get("m", "c", ["local"])
        self.assertEqual(rec.prefill_tokens, 600.0)
        self.assertEqual(rec.j_per_prefill_token(), 3.0)


# ---------------------------------------------------------------------------
# record_harness_result: phase-pure by construction, multi-bucket.
# ---------------------------------------------------------------------------
class TestHarnessResult(unittest.TestCase):
    def test_accumulates_across_buckets(self):
        store = jc.JtokCounterStore(enabled=True)
        m1 = mock.Mock(prompt_tokens=100, decode_tokens=20, n_requests=1,
                       prefill_joules=200.0, decode_joules=40.0)
        m2 = mock.Mock(prompt_tokens=200, decode_tokens=40, n_requests=2,
                       prefill_joules=1000.0, decode_joules=240.0)
        out = jc.record_harness_result(
            store, model="Qwen3.6-27B", config_label="no-MTP baseline",
            lanes=["local"], measurements=[m1, m2],
        )
        self.assertEqual(len(out), 2)
        rec = store.get("Qwen3.6-27B", "no-MTP baseline", ["local"])
        # total_prefill_tok = 100*1 + 200*2 = 500 ; total_prefill_j = 200+1000=1200
        self.assertEqual(rec.j_per_prefill_token(), 1200.0 / 500.0)
        # total_decode_tok = 20*1 + 40*2 = 100 ; total_decode_j = 40+240=280
        self.assertEqual(rec.j_per_decode_token(), 280.0 / 100.0)
        self.assertEqual(rec.mixed_windows, 0)
        self.assertEqual(rec.sources, [jc.SOURCE_HARNESS])

    def test_zero_token_bucket_skipped_cleanly(self):
        store = jc.JtokCounterStore(enabled=True)
        m = mock.Mock(prompt_tokens=0, decode_tokens=10, n_requests=1,
                     prefill_joules=0.0, decode_joules=30.0)
        jc.record_harness_result(store, model="m", config_label="c",
                                  lanes=["local"], measurements=[m])
        rec = store.get("m", "c", ["local"])
        self.assertIsNone(rec.j_per_prefill_token())
        self.assertEqual(rec.j_per_decode_token(), 3.0)


# ---------------------------------------------------------------------------
# Guarded read/write helpers + schema-stamp interaction (mirrors
# test_self_update.TestDataSchemaGuard.test_webui_write_endpoints_refuse_on_newer_schema).
# ---------------------------------------------------------------------------
class TestSchemaGuard(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = os.path.join(self._tmp.name, "data")
        self.path = os.path.join(self.data_dir, "jtok_counter.json")

    def tearDown(self):
        self._tmp.cleanup()

    def test_writes_allowed_on_fresh_data_dir(self):
        with mock.patch.dict(os.environ, {"SGLANG_PLANNER_DATA_DIR": self.data_dir}):
            d = jc.jtok_set_enabled(True, path=self.path)
            self.assertTrue(d["ok"])
            self.assertTrue(d["enabled"])

    def test_newer_schema_blocks_toggle_and_reset(self):
        os.makedirs(self.data_dir)
        with open(os.path.join(self.data_dir, su.SCHEMA_STAMP_NAME), "w") as f:
            json.dump({"schema_version": su.DATA_SCHEMA_VERSION + 1}, f)
        with mock.patch.dict(os.environ, {"SGLANG_PLANNER_DATA_DIR": self.data_dir}):
            d = jc.jtok_set_enabled(True, path=self.path)
            self.assertFalse(d["ok"])
            self.assertIn("schema", d["error"])
            d = jc.jtok_reset(path=self.path, reset_all=True)
            self.assertFalse(d["ok"])
            self.assertIn("schema", d["error"])
        # Nothing was written -- the guarded path never even opened the file.
        self.assertFalse(os.path.exists(self.path))

    def test_read_never_blocked_by_guard(self):
        os.makedirs(self.data_dir)
        with open(os.path.join(self.data_dir, su.SCHEMA_STAMP_NAME), "w") as f:
            json.dump({"schema_version": su.DATA_SCHEMA_VERSION + 1}, f)
        with mock.patch.dict(os.environ, {"SGLANG_PLANNER_DATA_DIR": self.data_dir}):
            # jtok_read must succeed (return an empty view) even though writes
            # to this same data dir are guarded off.
            view = jc.jtok_read(path=self.path)
            self.assertEqual(view["records"], [])
            self.assertFalse(view["enabled"])


if __name__ == "__main__":
    unittest.main()
