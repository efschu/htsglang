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
"""CPU unit tests for the roofline throughput ESTIMATE (design #145).

The estimate must: be produced; be monotonic (more bandwidth -> faster decode,
more FLOPS -> faster prefill, PCIe discount < NVLink); carry the
``planner-estimate`` provenance; NEVER be admissible into the measured results
store; and prefer MEASURED probe scores over the nameplate peaks. No GPU."""

import dataclasses
import json
import os
import tempfile
import unittest

from sglang.srt.planner.hardware import GpuDescriptor, HardwareSpec
from sglang.srt.planner.profiles import GpuProfile, ProfileLibrary
from sglang.srt.planner.roofline import (
    ROOFLINE_PROVENANCE,
    RooflineEstimate,
    estimate_roofline,
)
from sglang.srt.uneven_perf import PlanInputs


def _tiny_model_dir(moe: bool = False) -> str:
    """A minimal HF config.json the cost model can size without weights."""
    d = tempfile.mkdtemp(prefix="roofline_model_")
    cfg = {
        "hidden_size": 2048,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "intermediate_size": 8192,
        "vocab_size": 100000,
    }
    if moe:
        cfg.update(
            num_experts=64,
            num_experts_per_tok=8,
            moe_intermediate_size=768,
            shared_expert_intermediate_size=768,
        )
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f)
    return d


def _hardware(names, membw, gemm_fp16, nvlink=False, tp=None):
    """A HardwareSpec + a matching ProfileLibrary with the given peak specs."""
    gpus = tuple(
        GpuDescriptor(index=i, name=n, total_mib=40000, pcie_gen=4, pcie_width=16)
        for i, n in enumerate(names)
    )
    hw = HardwareSpec(gpus=gpus, source="manual", host_ram_mib=256 * 1024)
    profs = {
        n: GpuProfile(
            name=n,
            total_mib=40000,
            nvlink=nvlink,
            peak_membw_gbs=membw,
            peak_gemm_tflops_fp16=gemm_fp16,
        )
        for n in set(names)
    }
    return hw, ProfileLibrary(profs)


def _plan_inputs(model_dir, rank_gpu_id, budgets, ratio=None):
    return PlanInputs(
        tp_size=len(rank_gpu_id),
        model_path=model_dir,
        kv_cache_dtype="auto",
        rank_gpu_id=list(rank_gpu_id),
        effective_vram_mib=list(budgets),
        rank_tp_ratio=ratio,
    )


class TestRooflineProduced(unittest.TestCase):
    def setUp(self):
        self.model = _tiny_model_dir()

    def test_produced_and_provenance(self):
        hw, lib = _hardware(["ACard"], 1000.0, 200.0)
        inp = _plan_inputs(self.model, [0], [30000])
        rf = estimate_roofline(inp, hw, None, None, library=lib)
        self.assertIsInstance(rf, RooflineEstimate)
        self.assertEqual(rf.provenance, ROOFLINE_PROVENANCE)
        self.assertEqual(rf.provenance, "planner-estimate")
        self.assertGreater(rf.decode_tok_s_low, 0)
        self.assertGreater(rf.prefill_tok_s, 0)
        # short-context ceiling >= at-context value (KV only slows decode).
        self.assertGreaterEqual(rf.decode_tok_s_high, rf.decode_tok_s_low)

    def test_unknown_card_no_estimate(self):
        # A card absent from the library and with no measured probe -> None.
        hw = HardwareSpec(
            gpus=(GpuDescriptor(index=0, name="MysteryGPU", total_mib=40000),),
            source="manual",
        )
        inp = _plan_inputs(self.model, [0], [30000])
        rf = estimate_roofline(inp, hw, None, None, library=ProfileLibrary({}))
        self.assertIsNone(rf)


class TestRooflineMonotonic(unittest.TestCase):
    def setUp(self):
        self.model = _tiny_model_dir()

    def test_more_bandwidth_faster_decode(self):
        inp = _plan_inputs(self.model, [0], [30000])
        hw_lo, lib_lo = _hardware(["C"], 800.0, 200.0)
        hw_hi, lib_hi = _hardware(["C"], 1600.0, 200.0)
        lo = estimate_roofline(inp, hw_lo, None, None, library=lib_lo)
        hi = estimate_roofline(inp, hw_hi, None, None, library=lib_hi)
        self.assertGreater(hi.decode_tok_s_low, lo.decode_tok_s_low)

    def test_more_flops_faster_prefill(self):
        inp = _plan_inputs(self.model, [0], [30000])
        hw_lo, lib_lo = _hardware(["C"], 1000.0, 100.0)
        hw_hi, lib_hi = _hardware(["C"], 1000.0, 400.0)
        lo = estimate_roofline(inp, hw_lo, None, None, library=lib_lo)
        hi = estimate_roofline(inp, hw_hi, None, None, library=lib_hi)
        self.assertGreater(hi.prefill_tok_s, lo.prefill_tok_s)
        # membw unchanged -> decode ceiling essentially unchanged.
        self.assertAlmostEqual(
            hi.decode_tok_s_high, lo.decode_tok_s_high, delta=0.01
        )

    def test_pcie_discount_below_nvlink(self):
        # Same peaks + same TP=2 plan; only the interconnect differs.
        inp = _plan_inputs(self.model, [0, 1], [30000, 30000], ratio=[1, 1])
        hw_nv, lib_nv = _hardware(["N", "N"], 1000.0, 200.0, nvlink=True)
        hw_pc, lib_pc = _hardware(["P", "P"], 1000.0, 200.0, nvlink=False)
        nv = estimate_roofline(inp, hw_nv, None, None, library=lib_nv)
        pc = estimate_roofline(inp, hw_pc, None, None, library=lib_pc)
        self.assertEqual(nv.interconnect, "nvlink")
        self.assertEqual(pc.interconnect, "pcie-host-staging")
        self.assertGreater(nv.interconnect_discount, pc.interconnect_discount)
        self.assertGreater(nv.decode_tok_s_low, pc.decode_tok_s_low)


class TestRooflineMeasuredWins(unittest.TestCase):
    def setUp(self):
        self.model = _tiny_model_dir()

    def test_measured_scores_override_nameplate(self):
        @dataclasses.dataclass
        class Score:
            gpu_index: int
            name: str
            gemm_tflops: float
            membw_gbs: float

        inp = _plan_inputs(self.model, [0], [30000])
        hw, lib = _hardware(["C"], 1000.0, 200.0)  # nameplate membw 1000
        # A measured probe of 1700 GB/s must win over the 1000 nameplate.
        scores = (Score(0, "C", 250.0, 1700.0),)
        rf = estimate_roofline(inp, hw, None, None, library=lib, measured_scores=scores)
        self.assertEqual(rf.per_rank[0].membw_source, "measured")
        self.assertEqual(rf.per_rank[0].peak_membw_gbs, 1700.0)
        self.assertEqual(rf.per_rank[0].flops_source, "measured")
        # And the measured (faster) card yields a faster decode than nameplate.
        base = estimate_roofline(inp, hw, None, None, library=lib)
        self.assertGreater(rf.decode_tok_s_low, base.decode_tok_s_low)


class TestRooflineNeverMeasured(unittest.TestCase):
    """The estimate carries planner-estimate provenance, which the measured
    results store REJECTS at ingest (design §5B.3a) — an estimate can never be
    stored as a measurement."""

    def test_rejected_by_results_store(self):
        from sglang.srt.planner.results_store import (
            IngestRejected,
            QuantDescriptor,
            ResultEntry,
            ResultsStore,
        )

        model = _tiny_model_dir()
        hw, lib = _hardware(["C"], 1000.0, 200.0)
        inp = _plan_inputs(model, [0], [30000])
        rf = estimate_roofline(inp, hw, None, None, library=lib)
        self.assertEqual(rf.provenance, "planner-estimate")

        # Try to smuggle the estimate's decode number in as a measurement.
        entry = ResultEntry(
            model="tiny",
            quant=QuantDescriptor.parse("fp16"),
            hardware_cards=[(1, "C", 40000)],
            reproduce_flags=["--tp-size 1"],
            provenance=rf.provenance,  # "planner-estimate"
            peak_decode_tok_s=rf.decode_tok_s_low,
            peak_prefill_tok_s=rf.prefill_tok_s,
        )
        store = ResultsStore()
        with self.assertRaises(IngestRejected):
            store.check(entry)
        self.assertEqual(len(store), 0)


class TestRooflineMoEOffload(unittest.TestCase):
    """A MoE plan that fits only with host-RAM offload gains a PCIe-fetch term
    that lowers decode vs the same plan resident in VRAM."""

    def test_offload_lowers_decode(self):
        from sglang.srt.planner.feasibility import OffloadAssessment

        model = _tiny_model_dir(moe=True)
        hw, lib = _hardware(["C"], 1600.0, 300.0)
        inp = _plan_inputs(model, [0], [30000])
        vram = estimate_roofline(inp, hw, None, None, library=lib)
        offload = OffloadAssessment(
            status="ram_offload",
            offloaded_gib=8.0,
            host_ram_needed_mib=8192,
            host_ram_total_mib=256 * 1024,
            per_rank_offloaded_gib=[8.0],
            note="test",
        )
        off = estimate_roofline(inp, hw, None, offload, library=lib)
        self.assertIsNotNone(off.offload_note)
        self.assertLess(off.decode_tok_s_low, vram.decode_tok_s_low)


if __name__ == "__main__":
    unittest.main()
