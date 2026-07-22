"""Regression guard for planner weight-sizing on MoE + multi-GGUF checkpoints
(#144 PART 1).

Two classes of bug this pins:

  (a) A GGUF export directory that ALSO carries a BF16-shaped ``config.json``
      (Unsloth layout) must be sized from the GGUF header, never from the
      BF16 config (which over-counts ~4x -> spurious "no space"). A multi-gguf
      dir must let the caller pick the quant.

  (b) A sparse-MoE checkpoint (Qwen3.5-MoE: ``num_experts`` routed experts,
      no dense ``intermediate_size``) must count the expert mass. The old cost
      model KeyError'd on the missing key and, before that, ignored the
      experts entirely -> ~10x undercount / crash.

These run against the REAL local model cache and are SKIPPED wholesale when it
is not present (CI machines without the weights), so they never flake — but on
the rig they assert every listed ref sizes within +/-10 % of its on-disk
weight bytes AND yields a feasible plan on the real 3-card rig.
"""

import glob
import os
import unittest

from sglang.srt.planner.feasibility import plan
from sglang.srt.planner.hardware import hardware_from_manual
from sglang.srt.planner.model import list_gguf_options, resolve_model_ref
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

# Local model-zoo root (machine-specific). These tests read REAL checkpoints
# from disk, so they only run where HTSGLANG_TEST_MODEL_DIR points at a
# populated cache; everywhere else they skip cleanly.
_CACHE = os.environ.get("HTSGLANG_TEST_MODEL_DIR", "")

#: The reference rig (5090 32 GiB + 2x 3080 20 GiB), hand-declared.
_RIG3 = ("RTX 5090:32760", "RTX 3080:20480", "RTX 3080:20480")
_RIG2 = ("RTX 3080:20480", "RTX 3080:20480")

# (label, ref, gguf_choice, tp, rig, max_running_requests)
_REFS = [
    ("FP8-MoE", f"{_CACHE}/Qwen3.6-35B-A3B-FP8", None, 3, _RIG3, None),
    ("AWQ-MoE", f"{_CACHE}/Qwen3.6-35B-A3B-AWQ-4bit", None, 3, _RIG3, None),
    ("GPTQ-MoE", f"{_CACHE}/Qwen3.5-35B-A3B-GPTQ-Int4", None, 3, _RIG3, None),
    (
        "GGUF-MoE-35B",
        f"{_CACHE}/Qwen3.6-35B-A3B-MTP-UD-Q3_K_M-GGUF",
        None,
        3,
        _RIG3,
        None,
    ),
    (
        "unsloth-27B-Q4",
        f"{_CACHE}/unsloth/Qwen3.6-27B-MTP-GGUF",
        "Qwen3.6-27B-UD-Q4_K_XL.gguf",
        2,
        _RIG2,
        1,
    ),
    (
        "unsloth-27B-Q6",
        f"{_CACHE}/unsloth/Qwen3.6-27B-MTP-GGUF",
        "Qwen3.6-27B-UD-Q6_K_XL.gguf",
        2,
        _RIG2,
        1,
    ),
]


def _disk_weight_bytes(resolved: str) -> int:
    """On-disk weight bytes: the .gguf file, or the sum of the *.safetensors
    shards in a checkpoint directory."""
    if os.path.isfile(resolved):
        return os.path.getsize(resolved)
    total = sum(
        os.path.getsize(f)
        for f in glob.glob(os.path.join(resolved, "*.safetensors"))
    )
    return total


@unittest.skipUnless(
    _CACHE and os.path.isdir(_CACHE),
    "HTSGLANG_TEST_MODEL_DIR not set or not a directory",
)
class TestMoEGgufSizing(CustomTestCase):
    def test_multi_gguf_dir_lists_checkpoints_excluding_sidecars(self):
        d = f"{_CACHE}/unsloth/Qwen3.6-27B-MTP-GGUF"
        if not os.path.isdir(d):
            self.skipTest(f"{d} absent")
        opts = list_gguf_options(d)
        # The three UD quants, none of the mmproj/imatrix sidecars.
        self.assertTrue(all(o.endswith(".gguf") for o in opts))
        self.assertFalse(any("mmproj" in o for o in opts))
        self.assertFalse(any("imatrix" in o for o in opts))
        self.assertIn("Qwen3.6-27B-UD-Q4_K_XL.gguf", opts)

    def test_gguf_dir_with_config_json_sizes_from_gguf_not_bf16(self):
        # The Unsloth dir carries a BF16-shaped config.json beside the ggufs.
        # Resolving with a choice must land on the .gguf FILE (real quant
        # bytes), not the directory (BF16 config -> ~4x over-count).
        d = f"{_CACHE}/unsloth/Qwen3.6-27B-MTP-GGUF"
        if not os.path.isdir(d):
            self.skipTest(f"{d} absent")
        resolved = resolve_model_ref(d, gguf_choice="Qwen3.6-27B-UD-Q4_K_XL.gguf")
        self.assertTrue(resolved.endswith("Qwen3.6-27B-UD-Q4_K_XL.gguf"))

    def test_refs_size_within_10pct_and_fit_on_rig(self):
        checked = 0
        for label, ref, choice, tp, rig, mrr in _REFS:
            with self.subTest(model=label):
                if not os.path.exists(ref):
                    self.skipTest(f"{ref} absent")
                resolved = resolve_model_ref(ref, gguf_choice=choice)
                disk = _disk_weight_bytes(resolved)
                self.assertGreater(disk, 0, f"{label}: no on-disk weight bytes")
                hw = hardware_from_manual(rig)
                r = plan(
                    resolved,
                    hw,
                    tp_size=tp,
                    max_running_requests=mrr,
                    with_advantage=False,
                )
                self.assertIsNotNone(
                    r.capacity, f"{label}: cost model could not size the plan"
                )
                model_bytes = sum(
                    rc.weight_gib * 2**30 for rc in r.capacity.per_rank
                )
                rel = abs(model_bytes - disk) / disk
                self.assertLessEqual(
                    rel,
                    0.10,
                    f"{label}: sized {model_bytes/2**30:.1f} GiB vs on-disk "
                    f"{disk/2**30:.1f} GiB ({rel*100:.0f}% off, > 10%)",
                )
                self.assertTrue(
                    r.fits,
                    f"{label}: feasible plan expected on {rig} at TP={tp} "
                    f"(mrr={mrr}); got infeasible: {r.infeasible_reasons}",
                )
                checked += 1
        if checked == 0:
            self.skipTest("none of the reference models are present")

    def test_moe_ram_offload_when_vram_short(self):
        # A MoE that does not fit in VRAM on 2x3080 must report a host-RAM
        # offload path (its routed experts tier to host), not a flat "no fit".
        ref = f"{_CACHE}/Qwen3.6-35B-A3B-FP8"
        if not os.path.isdir(ref):
            self.skipTest(f"{ref} absent")
        hw = hardware_from_manual(_RIG2)
        r = plan(
            ref,
            hw,
            tp_size=2,
            host_ram_mib=128 * 1024,
            with_advantage=False,
        )
        self.assertFalse(r.fits, "expected NOT to fit purely in VRAM on 2x3080")
        self.assertIsNotNone(r.offload)
        self.assertEqual(r.offload.status, "ram_offload")
        self.assertGreater(r.offload.offloaded_gib, 0.0)

    def test_moe_ram_offload_needs_host_ram(self):
        # With a tiny host, the same plan cannot fit even with offload.
        ref = f"{_CACHE}/Qwen3.6-35B-A3B-FP8"
        if not os.path.isdir(ref):
            self.skipTest(f"{ref} absent")
        hw = hardware_from_manual(_RIG2)
        r = plan(ref, hw, tp_size=2, host_ram_mib=2048, with_advantage=False)
        self.assertEqual(r.offload.status, "cannot_fit")

    def test_dense_checkpoint_has_no_offloadable_class(self):
        # A dense GGUF that overflows VRAM cannot claim host-offload (only MoE
        # routed experts tier at serving time).
        ref = f"{_CACHE}/unsloth/Qwen3.6-27B-MTP-GGUF"
        choice = "Qwen3.6-27B-UD-Q6_K_XL.gguf"
        if not os.path.exists(os.path.join(ref, choice)):
            self.skipTest(f"{ref}/{choice} absent")
        resolved = resolve_model_ref(ref, gguf_choice=choice)
        hw = hardware_from_manual(_RIG2)
        # High concurrency inflates the (non-offloadable) mamba pool -> overflow.
        r = plan(
            resolved,
            hw,
            tp_size=2,
            max_running_requests=16,
            host_ram_mib=128 * 1024,
            with_advantage=False,
        )
        self.assertFalse(r.fits)
        self.assertEqual(r.offload.status, "cannot_fit")
        for rc in r.capacity.per_rank:
            self.assertEqual(rc.offloadable_weight_gib, 0.0)


if __name__ == "__main__":
    unittest.main()
