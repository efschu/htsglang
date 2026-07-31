"""The INT8 W8A8 lane in the prefill objective (task #353).

Before this landed, `checkpoint_compute_format` returned the bare
`"compressed-tensors"` for an INT8 W8A8 checkpoint, `_FORMAT_LANES` had no
entry for it, and `--rank-tp-ratio auto-performance` scored every rank on the
DENSE BF16 probe with the generic "no lane table" warning. The #327 boot's
split was usable anyway only by coincidence: on that rig the bf16 ratios
(3.68 : 1.00 : 0.97) happen to sit near the true int8 ratios
(3.81 : 1.00 : 1.03), while the fp8 ratios of the same cards are
9.36 : 1.00 : 0.99 -- a rig with a different card mix has no such luck.

Card-measured on the reference rig 2026-07-31, one run, all lanes, probe shape
2048 x 5120 x 17408: int8_native 684.27 (5090, sm_120) / 186.92 / 189.03
(3080s, sm_86), against best-fp8 568.77 / 59.50 / 57.91 on the same cards in
the same run. Those are the numbers rigged below.

The lane carries a real probe, so unlike the nvfp4 entries it is measurable
today; and it deliberately carries only ONE lane, because INT8 W8A8 has no
dequant fallback in this tree.

All on CPU with rigged probe values.
"""

import json
import os
import tempfile
import unittest

from sglang.srt import uneven_perf
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

#: Measured on the reference rig 2026-07-31 (see module docstring).
_BF16 = [231.90, 62.81, 63.78]
_INT8 = [684.27, 186.92, 189.03]
_FP8_NATIVE_U0 = 568.77
_FP8_MARLIN = [None, 59.50, 57.91]

#: The Avesed/Qwen3.6-27B-INT8-W8A8 quantization_config, trimmed to the keys
#: the format detection reads.
_INT8_W8A8_CONFIG = {
    "quant_method": "compressed-tensors",
    "format": "int-quantized",
    "config_groups": {
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "num_bits": 8,
                "type": "int",
                "strategy": "channel",
                "symmetric": True,
                "dynamic": False,
            },
            "input_activations": {
                "num_bits": 8,
                "type": "int",
                "strategy": "token",
                "symmetric": True,
                "dynamic": True,
            },
        }
    },
}

#: TheHouseOfTheDude/Qwen3.6-27B-INT8 class: weight-only INT8, no activation
#: quantization at all. Runs through Marlin wNa16, NOT through int8_scaled_mm.
_INT8_W8A16_CONFIG = {
    "quant_method": "compressed-tensors",
    "format": "pack-quantized",
    "config_groups": {
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "num_bits": 8,
                "type": "int",
                "strategy": "group",
                "group_size": 128,
                "symmetric": True,
            },
            "input_activations": None,
        }
    },
}


def _entries(int8=True):
    out = []
    for i in range(3):
        lanes = {}
        notes = {}
        if i == 0:
            lanes[uneven_perf.LANE_FP8_NATIVE] = _FP8_NATIVE_U0
        else:
            lanes[uneven_perf.LANE_FP8_MARLIN] = _FP8_MARLIN[i]
        if int8:
            lanes[uneven_perf.LANE_INT8_NATIVE] = _INT8[i]
        else:
            notes[uneven_perf.LANE_INT8_NATIVE] = (
                "int8 GEMM did not run: NotImplementedError: No implemented "
                "int8_scaled_mm for compute capability sm120."
            )
        out.append(
            {
                "name": "NVIDIA GeForce RTX 5090"
                if i == 0
                else "NVIDIA GeForce RTX 3080",
                "cuda_index": i,
                "gemm_tflops": _BF16[i],
                "gemm_lanes": lanes,
                "gemm_lane_notes": notes,
            }
        )
    return out


def _write_config(tmp, quantization_config):
    with open(os.path.join(tmp, "config.json"), "w") as f:
        json.dump({"hidden_size": 5120, "quantization_config": quantization_config}, f)
    return tmp


class TestFormatDetection(CustomTestCase):
    def test_a_w8a8_checkpoint_reports_int8(self):
        with tempfile.TemporaryDirectory() as tmp:
            fmt, desc = uneven_perf.checkpoint_compute_format(
                _write_config(tmp, _INT8_W8A8_CONFIG)
            )
        self.assertEqual(fmt, "int8")
        self.assertIn("int8 W8A8", desc)
        self.assertIn("int-quantized", desc)

    def test_weight_only_int8_does_not_take_the_lane(self):
        # It has no int8_scaled_mm path; claiming the lane would price it on a
        # kernel it never runs. Falls through to the scheme's own name, which
        # has no lane table -> the loud bf16 fallback, unchanged.
        with tempfile.TemporaryDirectory() as tmp:
            fmt, _ = uneven_perf.checkpoint_compute_format(
                _write_config(tmp, _INT8_W8A16_CONFIG)
            )
        self.assertNotEqual(fmt, "int8")
        self.assertNotIn(fmt, uneven_perf._FORMAT_LANES)

    def test_the_standalone_w8a8_int8_method_declares_itself(self):
        self.assertTrue(uneven_perf._is_int8_w8a8_like("w8a8_int8", "", {}))

    def test_a_four_bit_int_group_is_not_int8(self):
        qc = {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "config_groups": {
                "g": {
                    "weights": {"num_bits": 4, "type": "int"},
                    "input_activations": {
                        "num_bits": 4,
                        "type": "int",
                        "dynamic": True,
                    },
                }
            },
        }
        self.assertFalse(
            uneven_perf._is_int8_w8a8_like("compressed-tensors", "pack-quantized", qc)
        )

    def test_fp8_is_still_fp8(self):
        # num_bits 8 with a FLOAT type must keep taking the fp8 branch, which
        # is tested ahead of the int8 one.
        qc = {
            "quant_method": "compressed-tensors",
            "format": "float-quantized",
            "config_groups": {
                "g": {
                    "weights": {"num_bits": 8, "type": "float"},
                    "input_activations": {"num_bits": 8, "type": "float"},
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            fmt, _ = uneven_perf.checkpoint_compute_format(_write_config(tmp, qc))
        self.assertEqual(fmt, "fp8")

    def test_the_per_family_key_separates_w8a8_from_w8a16(self):
        w8a8 = _INT8_W8A8_CONFIG["config_groups"]["group_0"]
        w8a16 = _INT8_W8A16_CONFIG["config_groups"]["group_0"]
        self.assertEqual(uneven_perf._ct_group_format(w8a8), "int8")
        self.assertEqual(uneven_perf._ct_group_format(w8a16), "int8_a16")
        self.assertIn("int8", uneven_perf._FORMAT_LANES)
        self.assertNotIn("int8_a16", uneven_perf._FORMAT_LANES)


class TestLaneTable(CustomTestCase):
    def test_int8_carries_exactly_one_lane_and_it_has_a_probe(self):
        self.assertEqual(
            uneven_perf._FORMAT_LANES["int8"], (uneven_perf.LANE_INT8_NATIVE,)
        )
        self.assertIn(uneven_perf.LANE_INT8_NATIVE, uneven_perf._LANE_PROBES)
        self.assertIn(uneven_perf.LANE_INT8_NATIVE, uneven_perf._LANE_LABELS)

    def test_the_probe_returns_a_reason_not_a_number_off_card(self):
        value, note = uneven_perf._bench_gemm_int8_native_tflops("cuda:0")
        self.assertIsNone(value)
        self.assertTrue(note.strip())

    def test_no_profile_version_bump(self):
        # The lane is a new KEY inside the already-declared v3 gemm_lanes /
        # gemm_lane_notes fields, so no cache key changes and no rig pays the
        # #303 link-matrix re-probe (600 s/boot) for it.
        self.assertEqual(uneven_perf.PROFILE_VERSION, 3)
        self.assertIn("gemm_lanes", uneven_perf._PROFILE_VERSION_FIELDS[3])
        self.assertIn("gemm_lane_notes", uneven_perf._PROFILE_VERSION_FIELDS[3])


class TestScoring(CustomTestCase):
    def test_every_rank_is_scored_on_the_int8_lane(self):
        scores, labels, warnings = uneven_perf.rank_gemm_scores(_entries(), "int8")
        self.assertEqual(scores, _INT8)
        self.assertEqual(warnings, [])
        for label in labels:
            self.assertEqual(
                label, uneven_perf._LANE_LABELS[uneven_perf.LANE_INT8_NATIVE]
            )

    def test_the_lane_changes_the_ratio_the_objective_sees(self):
        int8, _, _ = uneven_perf.rank_gemm_scores(_entries(), "int8")
        fp8, _, _ = uneven_perf.rank_gemm_scores(_entries(), "fp8")
        bf16, _, _ = uneven_perf.rank_gemm_scores(_entries(int8=False), "int8")
        self.assertEqual(bf16, _BF16)  # the pre-#353 answer, loudly

        def ratio(v):
            return round(v[0] / min(v), 2)

        # The number the plan is actually made of: 3.66 on the int8 lane,
        # 9.82 if the fp8 lane were used, 3.69 on the bf16 fallback. The
        # fallback is CLOSE here and wrong everywhere else.
        self.assertAlmostEqual(ratio(int8), 3.66, places=2)
        self.assertAlmostEqual(ratio(fp8), 9.82, places=2)
        self.assertAlmostEqual(ratio(bf16), 3.69, places=2)

    def test_an_unprobed_card_falls_back_loudly_and_names_the_reprobe(self):
        scores, labels, warnings = uneven_perf.rank_gemm_scores(
            _entries(int8=False), "int8"
        )
        self.assertEqual(scores, _BF16)
        self.assertEqual(len(warnings), 3)
        for warning in warnings:
            self.assertIn("no int8 GEMM lane measured on this card", warning)
            self.assertIn("SGLANG_PERF_REPROBE=1", warning)
        for label in labels:
            self.assertIn("fallback", label)

    def test_a_partly_probed_rig_mixes_lanes_per_card(self):
        # sm100/sm103 have no classic IMMA path: that rank falls back, the
        # others keep their measured lane. Per-card, not per-rig.
        entries = _entries()
        del entries[0]["gemm_lanes"][uneven_perf.LANE_INT8_NATIVE]
        entries[0]["gemm_lane_notes"][uneven_perf.LANE_INT8_NATIVE] = (
            "int8 GEMM did not run: NotImplementedError: No implemented "
            "int8_scaled_mm for compute capability sm100."
        )
        scores, labels, warnings = uneven_perf.rank_gemm_scores(entries, "int8")
        self.assertEqual(scores, [_BF16[0], _INT8[1], _INT8[2]])
        self.assertIn("fallback", labels[0])
        self.assertNotIn("fallback", labels[1])
        self.assertEqual(len(warnings), 1)
        self.assertIn("sm100", warnings[0])


if __name__ == "__main__":
    unittest.main()
