# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Unit tests for the per-card power CALIBRATION artifact + its consumption by
the roofline energy estimate. The GPU-measurement path itself is NOT exercised
here (no hardware in CI); it is guarded so the module imports GPU-free. These
tests cover: the parse/aggregation of the sampled table, its persistence
round-trip, and — the point of the feature — that a MEASURED power entry
OVERRIDES the TDP heuristic for the card it matches while an unmeasured card
still falls back, with the hetero efficiency ranking preserved."""

import json
import os
import tempfile
import unittest

from sglang.srt.planner.hardware import GpuDescriptor, HardwareSpec
from sglang.srt.planner.power_calibration import (
    CardPowerMeasurement,
    PowerCalibrationResult,
    load_power_profile,
    power_profile_by_arch,
    save_power_profile,
)
from sglang.srt.planner.profiles import GpuProfile, ProfileLibrary
from sglang.srt.planner.roofline import (
    IDLE_FRACTION_OF_TDP,
    estimate_roofline,
    roofline_energy,
)
from sglang.srt.uneven_perf import PlanInputs


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


def _m(uuid, name, arch, idle, membw_w, gemm_w):
    return CardPowerMeasurement(
        uuid=uuid, name=name, arch=arch, total_mib=32607,
        p_idle_w=idle, p_membw_w=membw_w, p_gemm_w=gemm_w,
        membw_gbs=1560.0, gemm_tflops=300.0,
    )


def _tiny_model_dir():
    d = tempfile.mkdtemp(prefix="powcal_model_")
    cfg = {
        "hidden_size": 2048, "num_hidden_layers": 24,
        "num_attention_heads": 16, "num_key_value_heads": 4,
        "intermediate_size": 8192, "vocab_size": 100000,
    }
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f)
    return d


def _profile(name, total_mib, tdp_w, membw, flops, arch):
    return GpuProfile(name=name, total_mib=total_mib, tdp_w=tdp_w, sm_arch=arch,
                      peak_membw_gbs=membw, peak_gemm_tflops_fp16=flops)


def _hetero_rig(source="manual", uuids=(None, None)):
    names = ["RTX 5090", "RTX 3080 20GB"]
    gpus = tuple(
        GpuDescriptor(index=i, name=n, total_mib=(32607 if i == 0 else 20480),
                      pcie_gen=4, pcie_width=16, uuid=uuids[i])
        for i, n in enumerate(names)
    )
    hw = HardwareSpec(gpus=gpus, source=source, host_ram_mib=256 * 1024)
    lib = ProfileLibrary({
        "RTX 5090": _profile("RTX 5090", 32607, 575, 1792.0, 419.0, "sm120"),
        "RTX 3080 20GB": _profile("RTX 3080 20GB", 20480, 320, 760.0, 119.0, "sm86"),
    })
    return hw, lib


def _plan_inputs(model_dir, rank_gpu_id, budgets, ratio=None):
    return PlanInputs(
        tp_size=len(rank_gpu_id), model_path=model_dir, kv_cache_dtype="auto",
        rank_gpu_id=list(rank_gpu_id), effective_vram_mib=list(budgets),
        rank_tp_ratio=ratio,
    )


# ---------------------------------------------------------------------------
# Parse / aggregate / persist.
# ---------------------------------------------------------------------------


class TestArtifactParse(unittest.TestCase):
    def test_measurement_roundtrip(self):
        m = _m("GPU-abc", "RTX 5090", "sm120", 30.0, 200.0, 300.0)
        m2 = CardPowerMeasurement.from_json(m.to_json())
        self.assertEqual(m, m2)
        self.assertEqual(m.provenance, "measured")

    def test_from_json_ignores_extra_keys(self):
        d = _m("GPU-x", "RTX 3080 20GB", "sm86", 40.0, 220.0, 260.0).to_json()
        d["some_future_field"] = 123
        m = CardPowerMeasurement.from_json(d)
        self.assertEqual(m.arch, "sm86")

    def test_result_aggregation_and_lookup(self):
        res = PowerCalibrationResult(
            cards=[
                _m("GPU-a", "RTX 5090", "sm120", 30.0, 200.0, 300.0),
                _m("GPU-b", "RTX 3080 20GB", "sm86", 40.0, 220.0, 260.0),
            ],
            skipped=[{"uuid": "GPU-c", "name": "RTX 3080 20GB", "reason": "busy"}],
        )
        by_uuid = res.by_uuid()
        self.assertEqual(set(by_uuid), {"GPU-a", "GPU-b"})
        by_arch = power_profile_by_arch(by_uuid)
        self.assertEqual(set(by_arch), {"sm120", "sm86"})
        self.assertEqual(by_arch["sm120"].p_membw_w, 200.0)
        # skipped rows are carried, not silently dropped.
        self.assertEqual(len(res.skipped), 1)

    def test_persist_roundtrip(self):
        res = PowerCalibrationResult(
            cards=[_m("GPU-a", "RTX 5090", "sm120", 30.0, 200.0, 300.0)],
            driver="580.00",
        )
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "power_profile.json")
            save_power_profile(res, path)
            loaded = load_power_profile(path)
            self.assertIn("GPU-a", loaded)
            self.assertEqual(loaded["GPU-a"].p_gemm_w, 300.0)
            self.assertEqual(loaded["GPU-a"].provenance, "measured")

    def test_load_missing_file_is_empty(self):
        self.assertEqual(load_power_profile("/nonexistent/nope.json"), {})

    def test_load_rejects_non_measured(self):
        # A row without the measured provenance is not admitted as measured.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "p.json")
            row = _m("GPU-a", "RTX 5090", "sm120", 30.0, 200.0, 300.0).to_json()
            row["provenance"] = "planner-estimate"
            with open(path, "w") as f:
                json.dump({"cards": [row]}, f)
            self.assertEqual(load_power_profile(path), {})


# ---------------------------------------------------------------------------
# Consumption by the roofline energy estimate.
# ---------------------------------------------------------------------------


class TestMeasuredOverridesHeuristic(unittest.TestCase):
    def setUp(self):
        self.model = _tiny_model_dir()

    def test_measured_entry_overrides_by_uuid(self):
        hw, lib = _hetero_rig(uuids=("GPU-5090", "GPU-3080"))
        inp = _plan_inputs(self.model, [0, 1], [30000, 18000], ratio=[2, 1])
        rf = estimate_roofline(inp, hw, None, None, library=lib)
        profile = {
            "GPU-5090": _m("GPU-5090", "RTX 5090", "sm120", 55.0, 300.0, 360.0),
            "GPU-3080": _m("GPU-3080", "RTX 3080 20GB", "sm86", 42.0, 250.0, 240.0),
        }
        en = roofline_energy(rf, inp, hw, library=lib, power_profile=profile)
        by_idx = {c.gpu_index: c for c in en.decode}
        # Both cards now anchored on MEASURED power, not TDP.
        self.assertEqual(by_idx[0].power_source, "measured")
        self.assertEqual(by_idx[0].idle_w, 55.0)          # measured idle
        self.assertEqual(by_idx[0].active_anchor_w, 300.0)  # measured membw draw
        self.assertEqual(by_idx[1].idle_w, 42.0)
        # Watts sit between the measured idle and the measured active ceiling.
        for c in en.decode:
            self.assertGreaterEqual(c.watts + 1e-9, c.idle_w)
            self.assertLessEqual(c.watts - 1e-9, c.active_anchor_w)

    def test_prefill_uses_gemm_anchor(self):
        hw, lib = _hetero_rig(uuids=("GPU-5090", "GPU-3080"))
        inp = _plan_inputs(self.model, [0, 1], [30000, 18000], ratio=[2, 1])
        rf = estimate_roofline(inp, hw, None, None, library=lib)
        profile = {
            "GPU-5090": _m("GPU-5090", "RTX 5090", "sm120", 55.0, 300.0, 360.0),
        }
        en = roofline_energy(rf, inp, hw, library=lib, power_profile=profile)
        by_idx = {c.gpu_index: c for c in en.prefill}
        self.assertEqual(by_idx[0].active_anchor_w, 360.0)  # gemm anchor

    def test_unmeasured_card_falls_back_to_heuristic(self):
        # Only the 5090 has a measurement; the 3080 keeps the TDP heuristic.
        hw, lib = _hetero_rig(uuids=("GPU-5090", "GPU-3080"))
        inp = _plan_inputs(self.model, [0, 1], [30000, 18000], ratio=[2, 1])
        rf = estimate_roofline(inp, hw, None, None, library=lib)
        profile = {
            "GPU-5090": _m("GPU-5090", "RTX 5090", "sm120", 55.0, 300.0, 360.0),
        }
        en = roofline_energy(rf, inp, hw, library=lib, power_profile=profile)
        by_idx = {c.gpu_index: c for c in en.decode}
        self.assertEqual(by_idx[0].power_source, "measured")
        self.assertEqual(by_idx[1].power_source, "estimate-tdp")
        # 3080 fell back: idle = fraction x TDP, ceiling = TDP.
        self.assertAlmostEqual(by_idx[1].idle_w,
                               IDLE_FRACTION_OF_TDP * by_idx[1].tdp_w, places=6)
        self.assertAlmostEqual(by_idx[1].active_anchor_w, by_idx[1].tdp_w, places=6)

    def test_empty_profile_forces_heuristic(self):
        hw, lib = _hetero_rig(uuids=("GPU-5090", "GPU-3080"))
        inp = _plan_inputs(self.model, [0, 1], [30000, 18000], ratio=[2, 1])
        rf = estimate_roofline(inp, hw, None, None, library=lib)
        en = roofline_energy(rf, inp, hw, library=lib, power_profile={})
        for c in en.decode + en.prefill:
            self.assertEqual(c.power_source, "estimate-tdp")

    def test_manual_source_ignores_arch_only_match(self):
        # No UUIDs on a MANUAL rig -> an arch-keyed measurement must NOT bleed in
        # (keeps unit tests hermetic w.r.t. whatever is cached on the box).
        hw, lib = _hetero_rig(source="manual", uuids=(None, None))
        inp = _plan_inputs(self.model, [0, 1], [30000, 18000], ratio=[2, 1])
        rf = estimate_roofline(inp, hw, None, None, library=lib)
        profile = {
            "GPU-5090": _m("GPU-5090", "RTX 5090", "sm120", 55.0, 300.0, 360.0),
        }
        en = roofline_energy(rf, inp, hw, library=lib, power_profile=profile)
        for c in en.decode:
            self.assertEqual(c.power_source, "estimate-tdp")

    def test_live_source_allows_arch_match(self):
        # A LIVE rig (pynvml) with no UUID still matches by sm-arch.
        hw, lib = _hetero_rig(source="pynvml", uuids=(None, None))
        inp = _plan_inputs(self.model, [0, 1], [30000, 18000], ratio=[2, 1])
        rf = estimate_roofline(inp, hw, None, None, library=lib)
        profile = {
            "GPU-any": _m("GPU-any", "RTX 5090", "sm120", 55.0, 300.0, 360.0),
        }
        en = roofline_energy(rf, inp, hw, library=lib, power_profile=profile)
        by_idx = {c.gpu_index: c for c in en.decode}
        self.assertEqual(by_idx[0].power_source, "measured")   # sm120 matched
        self.assertEqual(by_idx[1].power_source, "estimate-tdp")  # sm86 absent


class TestHeteroRankingPreserved(unittest.TestCase):
    """With MEASURED anchors the per-card sums still equal the total and the
    hetero work-per-watt ranking (5090 above 3080) is preserved."""

    def setUp(self):
        self.model = _tiny_model_dir()

    def test_sum_and_ranking(self):
        hw, lib = _hetero_rig(uuids=("GPU-5090", "GPU-3080"))
        inp = _plan_inputs(self.model, [0, 1], [30000, 18000], ratio=[2, 1])
        rf = estimate_roofline(inp, hw, None, None, library=lib)
        profile = {
            "GPU-5090": _m("GPU-5090", "RTX 5090", "sm120", 55.0, 300.0, 360.0),
            "GPU-3080": _m("GPU-3080", "RTX 3080 20GB", "sm86", 42.0, 250.0, 240.0),
        }
        en = roofline_energy(rf, inp, hw, library=lib, power_profile=profile)
        self.assertAlmostEqual(sum(c.j_per_token for c in en.decode),
                               en.j_per_decode_token_total, places=9)
        wpw = {c.gpu_index: c.compute_share / c.j_per_token for c in en.decode}
        self.assertGreater(wpw[0], wpw[1])  # 5090 more work per joule

    def test_measured_note_in_caveats(self):
        hw, lib = _hetero_rig(uuids=("GPU-5090", "GPU-3080"))
        inp = _plan_inputs(self.model, [0, 1], [30000, 18000], ratio=[2, 1])
        rf = estimate_roofline(inp, hw, None, None, library=lib)
        profile = {
            "GPU-5090": _m("GPU-5090", "RTX 5090", "sm120", 55.0, 300.0, 360.0),
        }
        en = roofline_energy(rf, inp, hw, library=lib, power_profile=profile)
        self.assertTrue(any("MEASURED power anchors" in c for c in en.caveats))


if __name__ == "__main__":
    unittest.main()
