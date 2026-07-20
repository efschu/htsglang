"""CPU tests for the S4 hardware-profile library + combination explorer
(design §7-S4 / §2.7 / §8). No GPU, no network bind: pure library + matrix
math over synthetic checkpoints.
"""

import json
import os
import tempfile
import unittest

from sglang.srt.planner import explorer as explorer_mod
from sglang.srt.planner import webui
from sglang.srt.planner.explorer import (
    COMPOSED_ESTIMATE_NOTE,
    plan_matrix,
    provenance_of,
    render_matrix_text,
)
from sglang.srt.planner.hardware import HardwareSpec, hardware_from_manual
from sglang.srt.planner.issue_text import HardwareFingerprint
from sglang.srt.planner.profiles import (
    SEED_PROFILES,
    GpuProfile,
    ProfileLibrary,
    compose_rig,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

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


def _make_model(tmpdir, name="model", ckpt_gib=14.0):
    path = os.path.join(tmpdir, name)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(_CONFIG, f)
    with open(os.path.join(path, "m-00001.safetensors"), "wb") as f:
        f.truncate(int(ckpt_gib * 2**30))
    return path


# ---------------------------------------------------------------------------
# Profile library.
# ---------------------------------------------------------------------------


class TestProfileLibrary(CustomTestCase):
    def test_seed_has_the_rig_cards(self):
        lib = ProfileLibrary()
        self.assertTrue(lib.has("RTX 5090"))
        self.assertTrue(lib.has("RTX 3080 20GB"))
        self.assertEqual(lib.get("RTX 5090").total_mib, 32607)
        # case / vendor-prefix insensitive lookup.
        self.assertTrue(lib.has("nvidia rtx 5090"))

    def test_populate_from_fingerprint(self):
        lib = ProfileLibrary()
        n0 = len(lib.names())
        fp = HardwareFingerprint(
            cards=[(1, "RTX 7090", 49152), (2, "RTX 3080 20GB", 20480)]
        )
        added = lib.populate_from_fingerprint(fp)
        # New card added; the already-seeded one is not a duplicate add.
        self.assertEqual(added, 1)
        self.assertTrue(lib.has("RTX 7090"))
        self.assertEqual(len(lib.names()), n0 + 1)

    def test_populate_skips_unknown_vram(self):
        # A boot-log-only fingerprint has total_mib=0 -> must not poison the
        # library with a bogus total.
        lib = ProfileLibrary()
        fp = HardwareFingerprint(cards=[(1, "MysteryCard", 0)])
        self.assertEqual(lib.populate_from_fingerprint(fp), 0)
        self.assertFalse(lib.has("MysteryCard"))

    def test_measured_fields_fill_but_never_downgrade(self):
        lib = ProfileLibrary(profiles={})
        lib.add(GpuProfile("Card", 20480, gemm_tflops=100.0))
        # A newcomer without perf fields must not wipe the measured one.
        changed = lib.add(GpuProfile("Card", 20480))
        self.assertFalse(changed)
        self.assertEqual(lib.get("Card").gemm_tflops, 100.0)

    def test_save_load_roundtrip_keeps_seed(self):
        lib = ProfileLibrary()
        lib.add(GpuProfile("RTX 7090", 49152))
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "profiles.json")
            lib.save(p)
            reloaded = ProfileLibrary.load(p)
        self.assertTrue(reloaded.has("RTX 7090"))
        self.assertTrue(reloaded.has("RTX 5090"))  # seed preserved


# ---------------------------------------------------------------------------
# Composition.
# ---------------------------------------------------------------------------


class TestComposition(CustomTestCase):
    def test_compose_heterogeneous_rig(self):
        rig = compose_rig(["RTX 5090", "RTX 3080 20GB", "RTX 3080 20GB"])
        self.assertEqual(rig.source, "library-composition")
        self.assertEqual(len(rig.gpus), 3)
        self.assertEqual(rig.gpus[0].total_mib, 32607)
        self.assertEqual([g.index for g in rig.gpus], [0, 1, 2])
        # No live free-VRAM on a composed card (design §8).
        self.assertTrue(all(g.free_mib is None for g in rig.gpus))

    def test_compose_unknown_profile_fails_loud(self):
        with self.assertRaisesRegex(KeyError, "unknown GPU profile"):
            compose_rig(["RTX 5090", "RTX 9999"])

    def test_compose_accepts_profile_objects(self):
        rig = compose_rig([GpuProfile("Custom", 12345)])
        self.assertEqual(rig.gpus[0].total_mib, 12345)


# ---------------------------------------------------------------------------
# provenance: composed != measured, structural (design §8).
# ---------------------------------------------------------------------------


class TestProvenance(CustomTestCase):
    def test_composed_is_estimate(self):
        rig = compose_rig(["RTX 5090"])
        prov, est, note = provenance_of(rig)
        self.assertEqual(prov, "composed")
        self.assertTrue(est)
        self.assertIn("not measured", note)

    def test_manual_and_nvml_are_not_estimates(self):
        prov, est, note = provenance_of(hardware_from_manual(["RTX 5090:32607"]))
        self.assertEqual(prov, "declared")
        self.assertFalse(est)
        self.assertIsNone(note)
        prov2, est2, _ = provenance_of(
            HardwareSpec(gpus=(), source="nvml")
        )
        self.assertEqual(prov2, "live")
        self.assertFalse(est2)


# ---------------------------------------------------------------------------
# The matrix.
# ---------------------------------------------------------------------------


class TestMatrix(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.m27 = _make_model(cls._tmp.name, "m27", 14.0)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_matrix_over_models_and_rigs(self):
        models = [("27B", self.m27)]
        rigs = [
            compose_rig(["RTX 5090", "RTX 3080 20GB", "RTX 3080 20GB"]),
            compose_rig(["RTX 4090"]),  # 24GB single card
            hardware_from_manual(["RTX 5090:32607", "RTX 5090:32607"]),
        ]
        mx = plan_matrix(models, rigs)
        self.assertEqual(mx.models, ["27B"])
        self.assertEqual(len(mx.cells), 3)

        hetero = mx.cell("27B", "1x RTX 5090, 2x RTX 3080 20GB")
        self.assertTrue(hetero.fits)
        self.assertTrue(hetero.estimate)  # composed
        self.assertEqual(hetero.provenance, "composed")

        single = mx.cell("27B", "1x RTX 4090")
        self.assertFalse(single.fits)  # 27B does not fit one 24GB card
        self.assertTrue(single.estimate)

        real = mx.cell("27B", "2x RTX 5090")
        self.assertEqual(real.provenance, "declared")
        self.assertFalse(real.estimate)  # a declared real rig, not composed

    def test_every_composed_cell_carries_the_estimate_note(self):
        mx = plan_matrix(
            [("27B", self.m27)],
            [compose_rig(["RTX 5090", "RTX 3080 20GB", "RTX 3080 20GB"])],
        )
        for c in mx.cells:
            self.assertTrue(c.estimate)
            self.assertEqual(c.estimate_note, COMPOSED_ESTIMATE_NOTE)
            self.assertIn("not measured", c.estimate_note)

    def test_real_rig_cell_has_no_estimate_note(self):
        mx = plan_matrix(
            [("27B", self.m27)],
            [hardware_from_manual(["RTX 5090:32607", "RTX 5090:32607"])],
        )
        c = mx.cells[0]
        self.assertFalse(c.estimate)
        self.assertIsNone(c.estimate_note)

    def test_render_text_marks_composed_with_star(self):
        mx = plan_matrix(
            [("27B", self.m27)],
            [compose_rig(["RTX 5090", "RTX 3080 20GB", "RTX 3080 20GB"])],
        )
        txt = render_matrix_text(mx)
        self.assertIn("fit*", txt)  # composed -> starred
        self.assertIn("ESTIMATE", txt)
        self.assertIn("never tok/s", txt)

    def test_matrix_carries_no_throughput_field(self):
        # Honesty: a matrix cell is capacity/feasibility only.
        import dataclasses

        names = {f.name for f in dataclasses.fields(explorer_mod.MatrixCell)}
        # max_context_tokens is a KV/capacity (memory) quantity, not a rate —
        # same carve-out as the S1/S2/S3 honesty tests.
        names.discard("max_context_tokens")
        for bad in ("tok_s", "tokps", "tps", "throughput", "speed", "perf"):
            self.assertFalse(any(bad in n for n in names), bad)


# ---------------------------------------------------------------------------
# webui explorer endpoints (in-process).
# ---------------------------------------------------------------------------


class TestWebuiExplorer(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.m27 = _make_model(cls._tmp.name, "m27", 14.0)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_list_profiles(self):
        d = webui.list_profiles()
        names = {p["name"] for p in d["profiles"]}
        self.assertIn("RTX 5090", names)
        self.assertIn("RTX 3080 20GB", names)

    def test_matrix_from_payload_marks_composed_estimate(self):
        d = webui.matrix_from_payload(
            {
                "models": [{"label": "27B", "model": self.m27}],
                "rigs": [
                    {
                        "name": "hetero",
                        "profiles": ["RTX 5090", "RTX 3080 20GB", "RTX 3080 20GB"],
                    }
                ],
            }
        )
        self.assertTrue(d["ok"])
        self.assertTrue(all(c["estimate"] for c in d["cells"]))
        self.assertTrue(all(c["provenance"] == "composed" for c in d["cells"]))

    def test_matrix_unknown_profile_error(self):
        d = webui.matrix_from_payload(
            {
                "models": [{"label": "27B", "model": self.m27}],
                "rigs": [{"name": "bad", "profiles": ["RTX 9999"]}],
            }
        )
        self.assertFalse(d["ok"])
        self.assertIn("unknown GPU profile", d["error"])


if __name__ == "__main__":
    unittest.main()
