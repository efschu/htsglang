"""CPU tests for the S5 benchmark DB / crowdsourced landscape (design §5B).

No GPU, no network: the store ingest guard, the quant descriptor, the Mode-A /
Mode-B landscapes and the honesty structure (measured-only perf, bands, no
tok/s without measured provenance, composed != measured) are exercised over a
synthetic checkpoint + in-memory store.
"""

import dataclasses
import json
import os
import tempfile
import unittest

from sglang.srt.planner import landscape as landscape_mod
from sglang.srt.planner.landscape import (
    LandscapeCell,
    build_mode_a,
    build_mode_b,
    render_mode_a_text,
)
from sglang.srt.planner.card_library import compose_rig
from sglang.srt.planner.hardware import hardware_from_manual
from sglang.srt.planner.results_store import (
    Band,
    IngestRejected,
    QuantDescriptor,
    ResultEntry,
    ResultsStore,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=25, suite="base-a-test-cpu")

_CONFIG = {
    "architectures": ["Qwen3NextForCausalLM"],
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "num_hidden_layers": 48,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "vocab_size": 151936,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 12,
    "quantization_config": {"group_size": 32},
}


def _make_model(tmpdir):
    path = os.path.join(tmpdir, "m")
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(_CONFIG, f)
    with open(os.path.join(path, "m.safetensors"), "wb") as f:
        f.truncate(int(14 * 2**30))
    return path


def _measured_entry(**kw):
    base = dict(
        model="Qwen3.6-27B",
        quant=QuantDescriptor.parse("4b/AWQ/g128"),
        hardware_cards=[(1, "RTX 5090", 32607), (2, "RTX 3080 20GB", 20480)],
        reproduce_flags=["--tp-size 3", "--rank-gpu-id 0,1,2", "SGLANG_UNEVEN_DCP=1"],
        provenance="measured",
        fits=True,
        max_context_tokens=200000,
        batch=16,
        concurrency=8,
        tp_config="tp3-uneven",
        kv_cache_dtype="fp8_e4m3",
        peak_decode_tok_s=128.0,
        decode_tok_s_by_bucket={16: 120.0, 8: 95.0},
        j_per_decode_token_by_bucket={16: 0.42},
        kwh_saved=Band(1.8, 3.4, 2.6),
    )
    base.update(kw)
    return ResultEntry(**base)


# ---------------------------------------------------------------------------
# Quant descriptor (design §5B.3).
# ---------------------------------------------------------------------------


class TestQuantDescriptor(CustomTestCase):
    def test_parse_canonical_forms(self):
        cases = {
            "4b/AWQ/g128": (4.0, "AWQ", 128),
            "Q4_K_M": (4.0, "Q_K_M", None),
            "Q3_K_XL": (3.0, "Q_K_XL", None),
            "fp8": (8.0, "FP8", None),
            "fp8_e4m3": (8.0, "FP8", None),
            "GPTQ-Int4": (4.0, "GPTQ", None),
            "compressed-tensors g32": (16.0, "COMPRESSED", 32),
        }
        for text, key in cases.items():
            self.assertEqual(QuantDescriptor.parse(text).exact_key(), key, text)

    def test_exact_vs_similar_grouping(self):
        awq = QuantDescriptor.parse("4b/AWQ/g128")
        gptq = QuantDescriptor.parse("4b/GPTQ/g128")
        # Different exact keys (scheme differs) ...
        self.assertNotEqual(awq.exact_key(), gptq.exact_key())
        # ... but the same SIMILAR key (both 4-bit) — approximate grouping.
        self.assertEqual(awq.similar_key(), gptq.similar_key())


# ---------------------------------------------------------------------------
# Band: never a scalar (design §5A.4/§5A.5).
# ---------------------------------------------------------------------------


class TestBand(CustomTestCase):
    def test_scalar_rejected(self):
        with self.assertRaises(IngestRejected):
            Band.coerce(2.6)

    def test_band_coerces_and_orders(self):
        b = Band.coerce([1.8, 3.4])
        self.assertEqual(b.as_list(), [1.8, 3.4])
        with self.assertRaises(IngestRejected):
            Band(3.4, 1.8)  # hi < lo


# ---------------------------------------------------------------------------
# Ingest guard (design §5B.3a) — the core honesty gate.
# ---------------------------------------------------------------------------


class TestIngestGuard(CustomTestCase):
    def test_measured_accepted(self):
        store = ResultsStore()
        self.assertIsNone(store.try_ingest(_measured_entry()))
        self.assertEqual(len(store), 1)

    def test_unmeasured_perf_rejected(self):
        store = ResultsStore()
        bad = _measured_entry(provenance="planner-estimate")
        reason = store.try_ingest(bad)
        self.assertIsNotNone(reason)
        self.assertIn("not measured", reason)
        self.assertEqual(len(store), 0)  # rejected, not stored-with-caveat

    def test_measured_provenance_without_perf_rejected(self):
        # A measured-tagged entry that carries no measured field is refused.
        store = ResultsStore()
        entry = ResultEntry(
            model="X",
            quant=QuantDescriptor.parse("fp8"),
            hardware_cards=[(1, "RTX 5090", 32607)],
            reproduce_flags=[],
            provenance="measured",
        )
        self.assertIsNotNone(store.try_ingest(entry))

    def test_feasibility_record_not_stored(self):
        store = ResultsStore()
        entry = ResultEntry(
            model="X",
            quant=QuantDescriptor.parse("fp8"),
            hardware_cards=[(1, "RTX 5090", 32607)],
            reproduce_flags=[],
            provenance="planner-estimate",
            fits=True,
            max_context_tokens=90000,
        )
        reason = store.try_ingest(entry)
        self.assertIn("not stored here", reason)

    def test_jsonl_roundtrip_reapplies_guard(self):
        store = ResultsStore()
        store.ingest(_measured_entry())
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "results.jsonl")
            store.save(p)
            # Smuggle an unmeasured perf line in by hand.
            with open(p, "a") as f:
                bad = _measured_entry(provenance="planner-estimate").to_json()
                f.write(json.dumps(bad) + "\n")
            reloaded = ResultsStore.load(p)  # non-strict: skips the bad line
        self.assertEqual(len(reloaded), 1)  # only the measured one survives
        # kWh band survives the roundtrip as a band.
        self.assertIsInstance(reloaded.entries()[0].kwh_saved, Band)


# ---------------------------------------------------------------------------
# Mode A — same-model cross-rig (design §5B.3 Mode A).
# ---------------------------------------------------------------------------


class TestModeA(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.model = _make_model(cls._tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_measured_and_estimate_rows_separated(self):
        store = ResultsStore()
        store.ingest(_measured_entry())
        composed = compose_rig(["RTX 4090", "RTX 4090", "RTX 4090"])
        ls = build_mode_a(
            "Qwen3.6-27B",
            QuantDescriptor.parse("4b/AWQ/g128"),
            store=store,
            planner_rigs=[(self.model, composed)],
            bucket=16,
        )
        self.assertEqual(len(ls.rows), 2)
        measured = [r for r in ls.rows if r.is_measured]
        estimate = [r for r in ls.rows if not r.is_measured]
        self.assertEqual(len(measured), 1)
        self.assertEqual(len(estimate), 1)
        # Measured row carries the measured efficiency at the chosen bucket.
        self.assertEqual(measured[0].j_per_decode_token, 0.42)
        self.assertEqual(measured[0].efficiency_bucket, 16)
        self.assertEqual(measured[0].peak_decode.value, 128.0)
        # Estimate (composed) row has NO measured perf — absent, not invented.
        self.assertIsNone(estimate[0].j_per_decode_token)
        self.assertIsNone(estimate[0].peak_decode)
        self.assertEqual(estimate[0].provenance, "composed-estimate")

    def test_measured_row_preferred_over_planner_for_same_rig(self):
        # A measured entry for the 5090+2x3080 rig must suppress a planner
        # cell for the SAME rig class (measured-preferred, §5B).
        store = ResultsStore()
        store.ingest(_measured_entry())
        same_rig = hardware_from_manual(
            ["RTX 5090:32607", "RTX 3080 20GB:20480", "RTX 3080 20GB:20480"]
        )
        ls = build_mode_a(
            "Qwen3.6-27B",
            QuantDescriptor.parse("4b/AWQ/g128"),
            store=store,
            planner_rigs=[(self.model, same_rig)],
            bucket=16,
        )
        rigs = [r.rig for r in ls.rows]
        self.assertEqual(
            rigs.count("1x RTX 5090, 2x RTX 3080 20GB"), 1
        )
        self.assertTrue(ls.rows[0].is_measured)

    def test_efficiency_only_at_matched_bucket(self):
        # No data in the requested bucket -> efficiency blank, never
        # extrapolated (§5B.3 guard).
        store = ResultsStore()
        store.ingest(_measured_entry())
        ls = build_mode_a(
            "Qwen3.6-27B",
            QuantDescriptor.parse("4b/AWQ/g128"),
            store=store,
            bucket=64,  # no curve point here
        )
        self.assertIsNone(ls.rows[0].j_per_decode_token)
        self.assertIsNone(ls.rows[0].efficiency_bucket)

    def test_similar_quant_is_labelled_approximate(self):
        store = ResultsStore()
        store.ingest(_measured_entry(quant=QuantDescriptor.parse("4b/GPTQ/g128")))
        # Query AWQ but in similar mode -> the 4-bit GPTQ run matches.
        ls = build_mode_a(
            "Qwen3.6-27B",
            QuantDescriptor.parse("4b/AWQ/g128"),
            store=store,
            similar=True,
        )
        self.assertTrue(ls.approximate_quant)
        self.assertIn("similar", ls.quant.lower())
        self.assertIn("APPROXIMATE", ls.note)
        self.assertEqual(len(ls.rows), 1)  # matched by bits
        # Exact mode would NOT match (different scheme).
        ls2 = build_mode_a(
            "Qwen3.6-27B", QuantDescriptor.parse("4b/AWQ/g128"), store=store
        )
        self.assertEqual(len(ls2.rows), 0)

    def test_max_values_carry_operating_point(self):
        store = ResultsStore()
        store.ingest(_measured_entry())
        ls = build_mode_a(
            "Qwen3.6-27B", QuantDescriptor.parse("4b/AWQ/g128"), store=store,
            bucket=16,
        )
        self.assertIn("batch 16", ls.rows[0].peak_decode.render())

    def test_reproduce_block_present(self):
        store = ResultsStore()
        store.ingest(_measured_entry())
        ls = build_mode_a(
            "Qwen3.6-27B", QuantDescriptor.parse("4b/AWQ/g128"), store=store
        )
        self.assertIn("--rank-gpu-id 0,1,2", " ".join(ls.rows[0].config))

    def test_text_render_marks_estimate_and_measured_only_perf(self):
        store = ResultsStore()
        store.ingest(_measured_entry())
        composed = compose_rig(["RTX 4090", "RTX 4090", "RTX 4090"])
        ls = build_mode_a(
            "Qwen3.6-27B", QuantDescriptor.parse("4b/AWQ/g128"), store=store,
            planner_rigs=[(self.model, composed)], bucket=16,
        )
        txt = render_mode_a_text(ls)
        self.assertIn("MEASURED-only", txt)
        self.assertIn("*", txt)  # estimate rows starred


# ---------------------------------------------------------------------------
# Mode B — strict same-key reproducibility (design §5B.3 Mode B).
# ---------------------------------------------------------------------------


class TestModeB(CustomTestCase):
    def test_only_identical_key_entries_share_the_axis(self):
        store = ResultsStore()
        store.ingest(_measured_entry(batch=16))
        store.ingest(
            _measured_entry(
                batch=32,  # different batch -> different key
                hardware_cards=[(4, "RTX 4090", 24564)],
                decode_tok_s_by_bucket={32: 150.0},
                peak_decode_tok_s=150.0,
            )
        )
        ls = build_mode_b(
            "Qwen3.6-27B",
            QuantDescriptor.parse("4b/AWQ/g128"),
            tp_config="tp3-uneven",
            batch=16,
            concurrency=8,
            kv_cache_dtype="fp8_e4m3",
            store=store,
        )
        # Only the batch-16 run matches the key.
        self.assertEqual(len(ls.rows), 1)
        self.assertEqual(ls.mode, "B")


# ---------------------------------------------------------------------------
# Honesty structure.
# ---------------------------------------------------------------------------


class TestHonesty(CustomTestCase):
    def test_no_tok_s_without_measured_provenance(self):
        # A landscape built with NO store and only planner rigs must expose no
        # throughput number anywhere (perf columns absent).
        with tempfile.TemporaryDirectory() as d:
            model = _make_model(d)
            composed = compose_rig(["RTX 5090", "RTX 3080 20GB", "RTX 3080 20GB"])
            ls = build_mode_a(
                "Qwen3.6-27B",
                QuantDescriptor.parse("4b/AWQ/g128"),
                planner_rigs=[(model, composed)],
                bucket=16,
            )
            for r in ls.rows:
                self.assertFalse(r.is_measured)
                self.assertIsNone(r.j_per_decode_token)
                self.assertIsNone(r.j_per_prefill_token)
                self.assertIsNone(r.peak_decode)
                self.assertIsNone(r.peak_prefill)

    def test_landscape_cell_has_no_bare_scalar_saved_quantity(self):
        # Saved energy/time quantities only ever exist as Bands on the entry,
        # and the LandscapeCell surfaces measured curves/peaks with operating
        # points — never a bare kwh scalar field.
        names = {f.name for f in dataclasses.fields(LandscapeCell)}
        self.assertNotIn("kwh_saved", names)
        self.assertNotIn("kwh", names)

    def test_measured_flag_is_derived_not_supplied(self):
        # A composed/planner cell can never be flagged measured.
        with tempfile.TemporaryDirectory() as d:
            model = _make_model(d)
            composed = compose_rig(["RTX 5090"])
            ls = build_mode_a(
                "Qwen3.6-27B",
                QuantDescriptor.parse("4b/AWQ/g128"),
                planner_rigs=[(model, composed)],
            )
            self.assertTrue(all(not r.is_measured for r in ls.rows))
            self.assertTrue(
                all(r.provenance == "composed-estimate" for r in ls.rows)
            )


if __name__ == "__main__":
    unittest.main()
