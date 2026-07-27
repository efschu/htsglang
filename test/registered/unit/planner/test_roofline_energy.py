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
"""CPU unit tests for the roofline ENERGY estimate (design #148).

The estimate must: be produced from TDP + compute + membw + compute_share;
sum per-card J to the total; PREDICT the heterogeneous efficiency gap (a virtual
5090's work-per-watt > a virtual 3080's) before any measurement; keep the power
heuristic bounded to [idle, TDP]; carry the ``planner-estimate`` provenance; and
NEVER be admissible into the measured results store. No GPU."""

import dataclasses
import json
import os
import tempfile
import unittest

from sglang.srt.planner.hardware import GpuDescriptor, HardwareSpec
from sglang.srt.planner.card_library import CardSpec, CardLibrary
from sglang.srt.planner.roofline import (
    IDLE_FRACTION_OF_TDP,
    ROOFLINE_PROVENANCE,
    RooflineEnergyEstimate,
    estimate_roofline,
    roofline_energy,
)
from sglang.srt.uneven_perf import PlanInputs


def _tiny_model_dir() -> str:
    """A minimal HF config.json the cost model can size without weights."""
    d = tempfile.mkdtemp(prefix="rfenergy_model_")
    cfg = {
        "hidden_size": 2048,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "intermediate_size": 8192,
        "vocab_size": 100000,
    }
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f)
    return d


def _profile(name, total_mib, tdp_w, membw, flops):
    return CardSpec(
        name=name,
        total_mib=total_mib,
        tdp_w=tdp_w,
        peak_membw_gbs=membw,
        peak_gemm_tflops_fp16=flops,
    )


def _hetero_rig():
    """A virtual RTX 5090 + RTX 3080 rig (distinct TDP + compute + membw) with a
    matching profile library — the hetero case the feature is designed to
    predict. Physical GPU 0 = 5090, 1 = 3080."""
    names = ["RTX 5090", "RTX 3080 20GB"]
    gpus = tuple(
        GpuDescriptor(index=i, name=n, total_mib=(32607 if i == 0 else 20480),
                      pcie_gen=4, pcie_width=16)
        for i, n in enumerate(names)
    )
    hw = HardwareSpec(gpus=gpus, source="manual", host_ram_mib=256 * 1024)
    lib = CardLibrary({
        "RTX 5090": _profile("RTX 5090", 32607, 575, 1792.0, 419.0),
        "RTX 3080 20GB": _profile("RTX 3080 20GB", 20480, 320, 760.0, 119.0),
    })
    return hw, lib


def _plan_inputs(model_dir, rank_gpu_id, budgets, ratio=None):
    return PlanInputs(
        tp_size=len(rank_gpu_id),
        model_path=model_dir,
        kv_cache_dtype="auto",
        rank_gpu_id=list(rank_gpu_id),
        effective_vram_mib=list(budgets),
        rank_tp_ratio=ratio,
    )


class TestRooflineEnergyProduced(unittest.TestCase):
    def setUp(self):
        self.model = _tiny_model_dir()

    def test_produced_and_provenance(self):
        hw, lib = _hetero_rig()
        inp = _plan_inputs(self.model, [0, 1], [30000, 18000], ratio=[2, 1])
        rf = estimate_roofline(inp, hw, None, None, library=lib)
        en = roofline_energy(rf, inp, hw, library=lib)
        self.assertIsInstance(en, RooflineEnergyEstimate)
        self.assertEqual(en.provenance, ROOFLINE_PROVENANCE)
        self.assertEqual(en.provenance, "planner-estimate")
        self.assertGreater(en.j_per_decode_token_total, 0)
        self.assertGreater(en.j_per_prefill_token_total, 0)
        self.assertEqual(len(en.decode), 2)
        self.assertEqual(len(en.prefill), 2)

    def test_none_estimate_no_energy(self):
        hw, lib = _hetero_rig()
        inp = _plan_inputs(self.model, [0, 1], [30000, 18000], ratio=[2, 1])
        self.assertIsNone(roofline_energy(None, inp, hw, library=lib))

    def test_missing_tdp_no_energy(self):
        # A card whose profile carries no TDP -> no energy estimate (like an
        # unknown peak drops the throughput estimate).
        hw, _ = _hetero_rig()
        lib = CardLibrary({
            "RTX 5090": CardSpec("RTX 5090", 32607, peak_membw_gbs=1792.0,
                                   peak_gemm_tflops_fp16=419.0),  # tdp_w=None
            "RTX 3080 20GB": _profile("RTX 3080 20GB", 20480, 320, 760.0, 119.0),
        })
        inp = _plan_inputs(self.model, [0, 1], [30000, 18000], ratio=[2, 1])
        rf = estimate_roofline(inp, hw, None, None, library=lib)
        self.assertIsNone(roofline_energy(rf, inp, hw, library=lib))


class TestPerCardSumsToTotal(unittest.TestCase):
    def setUp(self):
        self.model = _tiny_model_dir()

    def test_sum_equals_total(self):
        hw, lib = _hetero_rig()
        inp = _plan_inputs(self.model, [0, 1], [30000, 18000], ratio=[2, 1])
        rf = estimate_roofline(inp, hw, None, None, library=lib)
        en = roofline_energy(rf, inp, hw, library=lib)
        self.assertAlmostEqual(
            sum(c.j_per_token for c in en.decode),
            en.j_per_decode_token_total, places=9)
        self.assertAlmostEqual(
            sum(c.j_per_token for c in en.prefill),
            en.j_per_prefill_token_total, places=9)

    def test_compute_share_sums_to_one(self):
        hw, lib = _hetero_rig()
        inp = _plan_inputs(self.model, [0, 1], [30000, 18000], ratio=[2, 1])
        rf = estimate_roofline(inp, hw, None, None, library=lib)
        en = roofline_energy(rf, inp, hw, library=lib)
        self.assertAlmostEqual(sum(c.compute_share for c in en.decode), 1.0, places=6)
        # ratio [2,1] -> 5090 gets 2/3 of the work.
        by_idx = {c.gpu_index: c for c in en.decode}
        self.assertAlmostEqual(by_idx[0].compute_share, 2 / 3, places=6)
        self.assertAlmostEqual(by_idx[1].compute_share, 1 / 3, places=6)


class TestHeteroEfficiencyGap(unittest.TestCase):
    """The predicted point of the feature: a virtual 5090 has higher work-per
    -watt than a virtual 3080 — the hetero efficiency gap PREDICTED, not
    measured. wpw = compute_share / J-per-token (the measured panel's metric)."""

    def setUp(self):
        self.model = _tiny_model_dir()

    def _wpw(self, cards):
        return {c.gpu_index: (c.compute_share / c.j_per_token) for c in cards}

    def test_5090_wpw_above_3080(self):
        hw, lib = _hetero_rig()
        inp = _plan_inputs(self.model, [0, 1], [30000, 18000], ratio=[2, 1])
        rf = estimate_roofline(inp, hw, None, None, library=lib)
        en = roofline_energy(rf, inp, hw, library=lib)
        for cards in (en.decode, en.prefill):
            wpw = self._wpw(cards)
            self.assertGreater(wpw[0], wpw[1])  # 5090 more work per watt

    def test_fast_card_lower_util_than_bottleneck(self):
        # The 3080 (slower membw/FLOPS) is the pace-setter -> higher utilization;
        # the 5090 finishes early and waits -> lower utilization -> draws less of
        # its (much larger) TDP. This is the physical cause of the gap.
        hw, lib = _hetero_rig()
        inp = _plan_inputs(self.model, [0, 1], [30000, 18000], ratio=[2, 1])
        rf = estimate_roofline(inp, hw, None, None, library=lib)
        en = roofline_energy(rf, inp, hw, library=lib)
        by_idx = {c.gpu_index: c for c in en.decode}
        self.assertLess(by_idx[0].util, by_idx[1].util)


class TestUtilBounded(unittest.TestCase):
    """The power heuristic is bounded: idle <= P_op <= TDP and 0 <= util <= 1
    for every card, every phase (design: 'P between idle and TDP')."""

    def setUp(self):
        self.model = _tiny_model_dir()

    def test_power_between_idle_and_tdp(self):
        hw, lib = _hetero_rig()
        inp = _plan_inputs(self.model, [0, 1], [30000, 18000], ratio=[2, 1])
        rf = estimate_roofline(inp, hw, None, None, library=lib)
        en = roofline_energy(rf, inp, hw, library=lib)
        self.assertEqual(en.idle_fraction_of_tdp, IDLE_FRACTION_OF_TDP)
        for cards in (en.decode, en.prefill):
            for c in cards:
                self.assertGreaterEqual(c.util, 0.0)
                self.assertLessEqual(c.util, 1.0)
                self.assertAlmostEqual(c.idle_w, IDLE_FRACTION_OF_TDP * c.tdp_w,
                                       places=6)
                self.assertGreaterEqual(c.watts + 1e-9, c.idle_w)
                self.assertLessEqual(c.watts - 1e-9, c.tdp_w)

    def test_bottleneck_util_is_efficiency_factor(self):
        # The pace-setting card runs at exactly eff (decode/prefill efficiency);
        # nothing exceeds it.
        hw, lib = _hetero_rig()
        inp = _plan_inputs(self.model, [0, 1], [30000, 18000], ratio=[2, 1])
        rf = estimate_roofline(inp, hw, None, None, library=lib)
        en = roofline_energy(rf, inp, hw, library=lib)
        self.assertAlmostEqual(max(c.util for c in en.decode), rf.eff_decode,
                               places=6)
        self.assertAlmostEqual(max(c.util for c in en.prefill), rf.eff_prefill,
                               places=6)


class TestColocatedRanks(unittest.TestCase):
    """Two ranks on ONE physical card aggregate onto a single board — their
    utilizations add (and clamp at TDP), one power draw, not two."""

    def setUp(self):
        self.model = _tiny_model_dir()

    def test_two_ranks_one_card(self):
        names = ["RTX 5090"]
        gpus = (GpuDescriptor(index=0, name="RTX 5090", total_mib=32607,
                              pcie_gen=4, pcie_width=16),)
        hw = HardwareSpec(gpus=gpus, source="manual", host_ram_mib=256 * 1024)
        lib = CardLibrary({"RTX 5090": _profile("RTX 5090", 32607, 575,
                                                   1792.0, 419.0)})
        inp = _plan_inputs(self.model, [0, 0], [15000, 15000], ratio=[1, 1])
        rf = estimate_roofline(inp, hw, None, None, library=lib)
        en = roofline_energy(rf, inp, hw, library=lib)
        # One physical card in the per-card table, holding the whole workload.
        self.assertEqual(len(en.decode), 1)
        self.assertAlmostEqual(en.decode[0].compute_share, 1.0, places=6)
        self.assertLessEqual(en.decode[0].watts, en.decode[0].tdp_w + 1e-9)


class TestEnergyNeverMeasured(unittest.TestCase):
    """The energy estimate carries planner-estimate provenance, which the
    measured results store REJECTS at ingest — an estimate can never overwrite
    or masquerade as a measurement."""

    def test_rejected_by_results_store(self):
        from sglang.srt.planner.results_store import (
            IngestRejected,
            QuantDescriptor,
            ResultEntry,
            ResultsStore,
        )

        model = _tiny_model_dir()
        hw, lib = _hetero_rig()
        inp = _plan_inputs(model, [0, 1], [30000, 18000], ratio=[2, 1])
        rf = estimate_roofline(inp, hw, None, None, library=lib)
        en = roofline_energy(rf, inp, hw, library=lib)
        self.assertEqual(en.provenance, "planner-estimate")

        entry = ResultEntry(
            model="tiny",
            quant=QuantDescriptor.parse("fp16"),
            hardware_cards=[(1, "RTX 5090", 32607), (1, "RTX 3080 20GB", 20480)],
            reproduce_flags=["--tp-size 2"],
            provenance=en.provenance,  # "planner-estimate"
            j_per_decode_token_by_bucket={1: en.j_per_decode_token_total},
        )
        store = ResultsStore()
        with self.assertRaises(IngestRejected):
            store.check(entry)
        self.assertEqual(len(store), 0)


if __name__ == "__main__":
    unittest.main()
