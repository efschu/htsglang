# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""Unit tests for the S2.5 energy MEASUREMENT harness (no GPU / no server boot).

Covers the pure logic: power integration (trapezoidal, per-card), the
measured ResultEntry build (must pass the results-store ingest guard), the
kWh/1M-token derivation, and the per-card energy structure. The end-to-end
boot+measure path is exercised by the live validation run, not here.
"""

import unittest

from sglang.srt.planner.energy import (
    BucketMeasurement,
    MeasurementConfig,
    MeasurementResult,
    PowerSampler,
    validation_config,
)
from sglang.srt.planner.results_store import ResultsStore


def _bm(workload, bucket, jpre, jdec, ncards=3):
    """A synthetic per-(workload,bucket) measurement."""
    per_pre = [jpre / ncards] * ncards
    per_dec = [jdec / ncards] * ncards
    return BucketMeasurement(
        workload=workload, bucket=bucket, prompt_tokens=50, decode_tokens=128,
        n_requests=bucket, prefill_seconds=1.0, decode_seconds=2.0,
        prefill_tok_s=100.0 * bucket, decode_tok_s=40.0 * bucket,
        prefill_joules=jpre * 50 * bucket, decode_joules=jdec * 128 * bucket,
        j_per_prefill_token=jpre, j_per_decode_token=jdec,
        avg_prefill_watts=300.0, avg_decode_watts=700.0,
        gpu_names=["RTX 5090", "RTX 3080", "RTX 3080"][:ncards],
        per_card_j_per_prefill_token=per_pre,
        per_card_j_per_decode_token=per_dec,
        per_card_avg_prefill_watts=[100.0] * ncards,
        per_card_avg_decode_watts=[233.0] * ncards,
    )


class TestPowerIntegration(unittest.TestCase):
    def test_trapezoidal_per_card_and_total(self):
        s = PowerSampler.__new__(PowerSampler)  # bypass NVML init
        s._handles = [object(), object()]  # two cards
        # constant 100 W and 200 W over 2 s -> 200 J and 400 J.
        s._samples = [
            (0.0, [100.0, 200.0]),
            (1.0, [100.0, 200.0]),
            (2.0, [100.0, 200.0]),
        ]
        pc_j, pc_w = s.integrate_per_card(0.0, 2.0)
        self.assertAlmostEqual(pc_j[0], 200.0, places=3)
        self.assertAlmostEqual(pc_j[1], 400.0, places=3)
        self.assertAlmostEqual(pc_w[0], 100.0, places=3)
        tot_j, tot_w = s.integrate(0.0, 2.0)
        self.assertAlmostEqual(tot_j, 600.0, places=3)   # SUM of the two cards
        self.assertAlmostEqual(tot_w, 300.0, places=3)

    def test_per_card_is_not_total_over_n(self):
        # Different per-card draw must NOT collapse to an even split.
        s = PowerSampler.__new__(PowerSampler)
        s._handles = [object(), object(), object()]
        s._samples = [(0.0, [222.0, 248.0, 236.0]), (1.0, [222.0, 248.0, 236.0])]
        pc_j, _ = s.integrate_per_card(0.0, 1.0)
        self.assertNotAlmostEqual(pc_j[0], pc_j[1], places=1)


class TestMeasuredEntries(unittest.TestCase):
    def _result(self):
        cfg = validation_config()
        ms = [_bm("code", 1, 0.6, 17.5), _bm("code", 16, 0.3, 1.6),
              _bm("prose", 1, 1.0, 18.0), _bm("prose", 16, 0.5, 1.7)]
        return MeasurementResult(
            config=cfg,
            hardware_cards=[(1, "RTX 5090", 32607), (2, "RTX 3080", 20480)],
            gpu_names_sampled=["RTX 5090", "RTX 3080", "RTX 3080"],
            measurements=ms, idle_watts=330.0,
            launch_flags=cfg.launch_command()[3:],
            compute_share_by_card=[0.46, 0.27, 0.27],
        )

    def test_entries_pass_ingest_guard(self):
        res = self._result()
        entries = res.result_entries()
        self.assertEqual({e.workload for e in entries}, {"code", "prose"})
        store = ResultsStore()
        for e in entries:
            self.assertIsNone(store.try_ingest(e))  # None == accepted
            self.assertEqual(e.provenance, "measured")
            self.assertTrue(e.has_measured_perf())

    def test_per_card_energy_stored_with_shares_and_label(self):
        e = self._result().result_entries()[0]
        pce = e.per_card_energy
        self.assertEqual(len(pce["gpu_names"]), 3)
        self.assertEqual(pce["compute_share"], [0.46, 0.27, 0.27])
        self.assertIn("excludes CPU/PSU", pce["source"])
        # per-card decode J/token present per bucket, aligned to the cards.
        self.assertEqual(len(pce["decode_j_per_token_by_bucket"][16]), 3)

    def test_kwh_per_1m_is_j_over_3p6(self):
        bm = _bm("code", 16, 0.3, 1.8)
        self.assertAlmostEqual(bm.kwh_per_1m_decode(), 1.8 / 3.6, places=6)
        self.assertAlmostEqual(bm.kwh_per_1m_prefill(), 0.3 / 3.6, places=6)

    def test_reload_under_strict_guard(self):
        import os
        import tempfile

        res = self._result()
        store = ResultsStore()
        for e in res.result_entries():
            store.ingest(e)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.jsonl")
            store.save(path)
            reloaded = ResultsStore.load(path, strict=True)
        self.assertEqual(len(reloaded), 2)
        self.assertIsNotNone(reloaded.entries()[0].per_card_energy)


class TestConfig(unittest.TestCase):
    def test_validation_launch_command_shape(self):
        cmd = validation_config().launch_command()
        joined = " ".join(cmd)
        self.assertIn("--rank-gpu-id 0,1,2", joined)
        self.assertIn("--rank-tp-ratio 28591,16464,16464", joined)
        self.assertIn("--kv-cache-dtype fp8_e4m3", joined)
        self.assertNotIn("--disable-cuda-graph", joined)  # graphs ON

    def test_cuda_graph_stays_on(self):
        cfg = MeasurementConfig(model_path="/x")
        self.assertFalse(cfg.disable_cuda_graph)


class TestMtpConfig(unittest.TestCase):
    def test_mtp_launch_command_has_spec_stack(self):
        from sglang.srt.planner.energy import mtp_config

        joined = " ".join(mtp_config().launch_command())
        self.assertIn("--speculative-algorithm NEXTN", joined)
        self.assertIn("--speculative-num-steps 5", joined)   # ceiling k=5
        self.assertIn("--speculative-eagle-topk 1", joined)  # chain
        self.assertIn("--speculative-num-draft-tokens 6", joined)  # steps+1
        self.assertIn("--speculative-adaptive", joined)
        self.assertIn("--speculative-adaptive-config high-accept", joined)
        self.assertNotIn("--disable-cuda-graph", joined)     # graphs ON
        self.assertEqual(mtp_config().label, "MTP+adaptive")

    def test_spec_entry_carries_label_and_accept(self):
        from sglang.srt.planner.energy import mtp_config

        cfg = mtp_config()
        bm = _bm("code", 1, 0.5, 6.8)
        bm.accept_length = 3.1
        bm.accept_rate = 0.72
        r = MeasurementResult(
            config=cfg,
            hardware_cards=[(1, "RTX 5090", 32607), (2, "RTX 3080", 20480)],
            gpu_names_sampled=["RTX 5090", "RTX 3080", "RTX 3080"],
            measurements=[bm], idle_watts=330.0,
            launch_flags=cfg.launch_command()[3:],
            label="MTP+adaptive",
            server_avg_accept_length=3.34,
            compute_share_by_card=[0.46, 0.27, 0.27],
        )
        e = r.result_entries()[0]
        store = ResultsStore()
        self.assertIsNone(store.try_ingest(e))  # passes the guard
        self.assertEqual(e.config_label, "MTP+adaptive")
        self.assertEqual(e.spec_accept_length_by_bucket, {1: 3.1})

    def test_compare_summary_reports_multiplier(self):
        from sglang.srt.planner.energy import compare_summary, validation_config

        def result(label, dec):
            bm = _bm("code", 1, 0.4, 4.0)
            bm.decode_tok_s = dec
            return MeasurementResult(
                config=validation_config(), hardware_cards=[(1, "X", 1)],
                gpu_names_sampled=["a", "b", "c"], measurements=[bm],
                idle_watts=1.0, launch_flags=[], label=label)
        txt = compare_summary(result("no-MTP baseline", 40.0),
                              result("MTP+adaptive", 100.0))
        self.assertIn("2.50x", txt)


if __name__ == "__main__":
    unittest.main()
