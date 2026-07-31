"""Per-(rank, family) GEMM scores for mixed-precision checkpoints (task #324).

``rank_gemm_scores`` returns ONE number per rank, and the checkpoint format
only selects which measured number that is. That is exactly right for fp8 --
one scheme, one lane, every family -- and wrong for a MIXED_PRECISION
checkpoint, where on the SAME card the MLP runs weight-only Marlin while
attention and the GDN in_proj run the native tensor path. The reference rig's
5090 measures 566.88 native against a 216 Marlin band: a 2.6x divergence
INSIDE one rank that a scalar cannot express (ANALYSE_321 sec. 8.3).

What is asserted here, all on CPU with rigged probe values:

* module paths map to compute families, experts before dense MLP;
* a single-scheme checkpoint reports NO per-family split, and its scores are
  byte-identical to the scalar path -- that is the migration story;
* a ModelOpt MIXED_PRECISION config and a two-group compressed-tensors config
  both produce family-different formats, and those resolve to family-different
  lanes per card;
* the enc objective's input vector -- and the candidate ladder it produces --
  differ from the scalar path's on a mixed checkpoint;
* a family with no datum falls back to the scalar and the fallback is NAMED;
* the v2 -> v3 profile migration path is untouched.
"""

import json
import os
import tempfile
import unittest

from sglang.srt import uneven_perf
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

#: NVML totals of the reference rig in rank order (5090, 3080, 3080).
_TOTALS = [32607, 20480, 20480]

#: Measured rates. bf16 and fp8-native are the #213 card probe's. The Marlin
#: band is ANALYSE_321's: 566.88 / 216.0 = 2.62x on the 5090, which is the
#: intra-card divergence this widening exists to represent. The 3080 figures
#: are the #298b lane reprobe's.
_BF16 = {"U0": 232.97, "U1": 62.72, "U2": 62.98}
_FP8_NATIVE = {"U0": 566.88}
_FP8_W8A16 = {"U1": 55.0, "U2": 55.2}
_NVFP4_MARLIN = {"U0": 216.0, "U1": 58.44, "U2": 59.15}


def _profile(nvfp4=True, lanes=True):
    gpus = {}
    for i, uuid in enumerate(("U0", "U1", "U2")):
        ent = {
            "name": "NVIDIA GeForce RTX 5090" if i == 0 else "NVIDIA GeForce RTX 3080",
            "cuda_index": i,
            "total_mib": _TOTALS[i],
            "gemm_tflops": _BF16[uuid],
            "membw_gbs": 1664.2 if i == 0 else 717.8,
            "membw_gemv_gbs": 1529.7 if i == 0 else 717.8,
        }
        if lanes:
            values = {}
            notes = {}
            if uuid in _FP8_NATIVE:
                values[uneven_perf.LANE_FP8_NATIVE] = _FP8_NATIVE[uuid]
            else:
                notes[uneven_perf.LANE_FP8_NATIVE] = (
                    "native fp8 GEMM did not run: RuntimeError: "
                    "compute capability 8.6"
                )
            if uuid in _FP8_W8A16:
                values[uneven_perf.LANE_FP8_W8A16] = _FP8_W8A16[uuid]
            if nvfp4:
                values[uneven_perf.LANE_NVFP4_MARLIN] = _NVFP4_MARLIN[uuid]
            ent["gemm_lanes"] = values
            ent["gemm_lane_notes"] = notes
        gpus[uuid] = ent
    return {
        "version": uneven_perf.PROFILE_VERSION,
        "driver": "595.58.03",
        "gpus": gpus,
        "links": {"__group__": {"ar_10kb_us": 32.4, "ar_1mb_us": 361.3}},
    }


def _entries(nvfp4=True, lanes=True):
    p = _profile(nvfp4=nvfp4, lanes=lanes)
    return [p["gpus"][u] for u in ("U0", "U1", "U2")]


def _write(tmp, quantization_config=None, hf_quant=None):
    """A checkpoint directory carrying just the quantization declaration."""
    cfg = {"hidden_size": 5120}
    if quantization_config is not None:
        cfg["quantization_config"] = quantization_config
    with open(os.path.join(tmp, "config.json"), "w") as f:
        json.dump(cfg, f)
    if hf_quant is not None:
        with open(os.path.join(tmp, "hf_quant_config.json"), "w") as f:
            json.dump(hf_quant, f)
    return tmp


def _layer(prefix, algo):
    return {f"model.layers.0.{prefix}": {"quant_algo": algo}}


#: nvidia/Qwen3.6-27B-NVFP4 (V1) in miniature: the MLP is weight-only NVFP4
#: (Marlin on EVERY architecture) while attention and the GDN in_proj stay fp8.
_MIXED_V1 = {
    "quant_method": "modelopt",
    "quant_algo": "MIXED_PRECISION",
    "kv_cache_quant_algo": "FP8",
    "quantized_layers": {
        **_layer("mlp.gate_proj", "W4A16_NVFP4"),
        **_layer("mlp.up_proj", "W4A16_NVFP4"),
        **_layer("mlp.down_proj", "W4A16_NVFP4"),
        **_layer("self_attn.qkv_proj", "FP8"),
        **_layer("self_attn.o_proj", "FP8"),
        **_layer("linear_attn.in_proj_qkv", "FP8"),
    },
}


class TestFamilyOfModule(CustomTestCase):
    """Module path -> compute family. Order is the load-bearing part."""

    def test_known_paths(self):
        cases = {
            "model.layers.0.mlp.gate_proj": uneven_perf.GEMM_FAMILY_MLP,
            "model.layers.0.feed_forward.w1": uneven_perf.GEMM_FAMILY_MLP,
            "model.layers.0.self_attn.q_proj": uneven_perf.GEMM_FAMILY_ATTN_GDN,
            "model.layers.0.linear_attn.in_proj_qkv": (
                uneven_perf.GEMM_FAMILY_ATTN_GDN
            ),
            "model.layers.0.self_attn.qkv_proj": uneven_perf.GEMM_FAMILY_ATTN_GDN,
            "lm_head": uneven_perf.GEMM_FAMILY_VOCAB,
            "model.embed_tokens": uneven_perf.GEMM_FAMILY_VOCAB,
            "re:.*mlp\\.experts.*": uneven_perf.GEMM_FAMILY_MOE,
            "model.layers.0.mlp.shared_expert.up_proj": uneven_perf.GEMM_FAMILY_MOE,
        }
        for path, family in cases.items():
            with self.subTest(path=path):
                self.assertEqual(uneven_perf.gemm_family_of_module(path), family)

    def test_routed_experts_are_moe_not_mlp(self):
        """They live UNDER ``mlp.experts``; the family order decides."""
        self.assertEqual(
            uneven_perf.gemm_family_of_module("model.layers.3.mlp.experts.7.w1"),
            uneven_perf.GEMM_FAMILY_MOE,
        )

    def test_unmatched_paths_are_none(self):
        for path in ("model.norm", "model.layers.0.input_layernorm", "router"):
            with self.subTest(path=path):
                self.assertIsNone(uneven_perf.gemm_family_of_module(path))


class TestPerFamilyFormatDetection(CustomTestCase):
    """The split comes from the checkpoint's own config, per module."""

    def test_single_scheme_declares_no_split(self):
        """Every non-mixed checkpoint: the third element stays EMPTY, so the
        scalar key is still the whole answer."""
        for qc in (
            None,
            {"quant_method": "fp8", "fmt": "e4m3"},
            {"quant_method": "gptq", "bits": 4},
            {
                "quant_method": "compressed-tensors",
                "format": "float-quantized",
                "config_groups": {
                    "group_0": {
                        "targets": ["Linear"],
                        "weights": {"num_bits": 8, "type": "float"},
                    }
                },
            },
        ):
            with self.subTest(qc=qc), tempfile.TemporaryDirectory() as tmp:
                _write(tmp, qc)
                key, _desc, families = uneven_perf.checkpoint_compute_format_families(
                    tmp
                )
                self.assertEqual(families, {})
                self.assertEqual(key, uneven_perf.checkpoint_compute_format(tmp)[0])

    def test_modelopt_mixed_precision_splits_mlp_from_attention(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, _MIXED_V1)
            key, desc, families = uneven_perf.checkpoint_compute_format_families(tmp)
        self.assertEqual(
            families,
            {
                uneven_perf.GEMM_FAMILY_MLP: "nvfp4_a16",
                uneven_perf.GEMM_FAMILY_ATTN_GDN: "fp8",
            },
        )
        # The checkpoint-wide key stays the FLOP-dominant family's, so a
        # scalar consumer is on the format most of the model runs in.
        self.assertEqual(key, "nvfp4_a16")
        self.assertIn("per-family", desc)

    def test_hf_quant_config_json_is_read_when_the_model_config_has_none(self):
        """ModelOpt exports keep the block in its own file."""
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, None, hf_quant={"quantization": _MIXED_V1})
            key, _desc, families = uneven_perf.checkpoint_compute_format_families(tmp)
        self.assertEqual(families[uneven_perf.GEMM_FAMILY_MLP], "nvfp4_a16")
        self.assertEqual(key, "nvfp4_a16")

    def test_compressed_tensors_two_groups_split_by_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write(
                tmp,
                {
                    "quant_method": "compressed-tensors",
                    "format": "nvfp4-pack-quantized",
                    "config_groups": {
                        "group_0": {
                            "targets": ["re:.*mlp\\.(gate|up|down)_proj"],
                            "weights": {
                                "num_bits": 4,
                                "type": "float",
                                "group_size": 16,
                            },
                            "input_activations": {"num_bits": 4, "type": "float"},
                        },
                        "group_1": {
                            "targets": ["re:.*self_attn.*"],
                            "weights": {"num_bits": 8, "type": "float"},
                        },
                    },
                },
            )
            _key, _desc, families = uneven_perf.checkpoint_compute_format_families(tmp)
        self.assertEqual(
            families,
            {
                uneven_perf.GEMM_FAMILY_MLP: "nvfp4_a4",
                uneven_perf.GEMM_FAMILY_ATTN_GDN: "fp8",
            },
        )

    def test_a_uniform_mixed_precision_map_is_not_a_split(self):
        """MIXED_PRECISION that happens to give every family the same algo has
        nothing to split, and must not grow a family dimension for show."""
        qc = dict(_MIXED_V1)
        qc["quantized_layers"] = {
            k: {"quant_algo": "FP8"} for k in _MIXED_V1["quantized_layers"]
        }
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, qc)
            key, _desc, families = uneven_perf.checkpoint_compute_format_families(tmp)
        self.assertEqual(families, {})
        self.assertEqual(key, "fp8")


class TestScalarPathUnchanged(CustomTestCase):
    """Regression pins on the #298a lane fixtures: nothing about a
    single-scheme checkpoint may move."""

    def test_fp8_scores_are_the_same_objects_scalar_path_produced(self):
        entries = _entries()
        scalar = uneven_perf.rank_gemm_scores(entries, "fp8")
        widened = uneven_perf.rank_gemm_family_scores(entries, "fp8", {})
        self.assertEqual(widened.scalar, scalar[0])
        self.assertEqual(widened.scalar, [566.88, 55.0, 55.2])
        self.assertEqual(widened.scalar_labels, scalar[1])
        self.assertEqual(widened.warnings, scalar[2])
        self.assertEqual(widened.families, {})
        self.assertFalse(widened.mixed)

    def test_every_family_lookup_returns_the_scalar_when_uniform(self):
        widened = uneven_perf.rank_gemm_family_scores(_entries(), "fp8", {})
        for family in uneven_perf.GEMM_FAMILIES:
            with self.subTest(family=family):
                self.assertEqual(widened.for_family(family), widened.scalar)

    def test_bf16_checkpoint_is_untouched(self):
        widened = uneven_perf.rank_gemm_family_scores(_entries(), "bf16", None)
        self.assertEqual(widened.scalar, [232.97, 62.72, 62.98])
        self.assertEqual(widened.warnings, [])
        self.assertFalse(widened.mixed)

    def test_a_family_declared_at_the_checkpoint_format_gets_no_own_vector(self):
        """Same format under a family key is the same numbers; storing them
        twice would make ``mixed`` lie."""
        widened = uneven_perf.rank_gemm_family_scores(
            _entries(),
            "fp8",
            {uneven_perf.GEMM_FAMILY_MLP: "fp8"},
        )
        self.assertEqual(widened.families, {})
        self.assertFalse(widened.mixed)


class TestMixedPrecisionScores(CustomTestCase):
    """Two families, two lanes, one card."""

    def _widened(self, **kw):
        return uneven_perf.rank_gemm_family_scores(
            _entries(**kw),
            "nvfp4_a16",
            {
                uneven_perf.GEMM_FAMILY_MLP: "nvfp4_a16",
                uneven_perf.GEMM_FAMILY_ATTN_GDN: "fp8",
            },
        )

    def test_the_two_families_land_on_different_lanes_on_the_same_card(self):
        widened = self._widened()
        mlp = widened.for_family(uneven_perf.GEMM_FAMILY_MLP)
        attn = widened.for_family(uneven_perf.GEMM_FAMILY_ATTN_GDN)
        # The MLP family IS the checkpoint-wide format here, so it has no own
        # vector and reads the scalar -- which is the Marlin lane.
        self.assertEqual(mlp, [216.0, 58.44, 59.15])
        self.assertEqual(attn, [566.88, 55.0, 55.2])
        self.assertIn("Marlin", widened.scalar_labels[0])
        self.assertIn(
            "native",
            widened.family_labels[uneven_perf.GEMM_FAMILY_ATTN_GDN][0],
        )
        self.assertTrue(widened.mixed)

    def test_the_intra_card_divergence_is_the_measured_2_6x(self):
        widened = self._widened()
        mlp = widened.for_family(uneven_perf.GEMM_FAMILY_MLP)
        attn = widened.for_family(uneven_perf.GEMM_FAMILY_ATTN_GDN)
        self.assertAlmostEqual(attn[0] / mlp[0], 2.62, places=2)
        # ... and it is INTRA-card: on the 3080s the two lanes are close,
        # so the divergence is not a rig-wide rescale the scalar could absorb.
        self.assertAlmostEqual(attn[1] / mlp[1], 0.94, places=2)

    def test_the_enc_input_vector_differs_from_the_scalar_path(self):
        widened = self._widened()
        attn, source = widened.resolve(uneven_perf.GEMM_FAMILY_ATTN_GDN)
        self.assertEqual(source, uneven_perf.GEMM_FAMILY_ATTN_GDN)
        self.assertNotEqual(attn, widened.scalar)
        # The ratio the prefill objective is made of moves by more than 2x.
        self.assertAlmostEqual(widened.scalar[0] / widened.scalar[1], 3.70, 2)
        self.assertAlmostEqual(attn[0] / attn[1], 10.31, 2)

    def test_family_warnings_name_the_format_they_came_from(self):
        """A card with no lane for a FAMILY's format must say which format."""
        widened = self._widened(lanes=False)
        self.assertTrue(widened.warnings)
        self.assertTrue(
            all("DENSE BF16" in w for w in widened.warnings),
            widened.warnings,
        )
        self.assertTrue(
            any(w.startswith("[fp8]") for w in widened.warnings),
            "a warning raised while resolving a FAMILY must carry its format",
        )

    def test_an_unprobed_lane_does_not_advise_a_reprobe(self):
        """``nvfp4_*`` carries dispatch order but no probe yet. Telling the
        reader to re-run the probe would be advice that cannot work."""
        widened = uneven_perf.rank_gemm_family_scores(
            _entries(nvfp4=False), "nvfp4_a16", {}
        )
        self.assertEqual(widened.scalar, [232.97, 62.72, 62.98])
        self.assertEqual(len(widened.warnings), 3)
        for w in widened.warnings:
            self.assertIn("no probe yet", w)
            self.assertNotIn("SGLANG_PERF_REPROBE", w)

    def test_the_fp8_reprobe_advice_survives(self):
        """The pre-existing hint must stay on the formats it applies to."""
        widened = uneven_perf.rank_gemm_family_scores(_entries(lanes=False), "fp8", {})
        self.assertEqual(len(widened.warnings), 3)
        for w in widened.warnings:
            self.assertIn("SGLANG_PERF_REPROBE=1", w)


class TestFallbackIsNamed(CustomTestCase):
    """Missing family data is the migration case, and it must be visible."""

    def test_resolve_falls_back_to_the_scalar_and_says_so(self):
        widened = uneven_perf.rank_gemm_family_scores(
            _entries(),
            "nvfp4_a16",
            {uneven_perf.GEMM_FAMILY_ATTN_GDN: "fp8"},
        )
        scores, source = widened.resolve(uneven_perf.GEMM_FAMILY_MOE)
        self.assertEqual(scores, widened.scalar)
        self.assertEqual(source, "scalar")

    def test_resolve_prefers_the_first_family_that_has_a_vector(self):
        widened = uneven_perf.rank_gemm_family_scores(
            _entries(),
            "nvfp4_a16",
            {uneven_perf.GEMM_FAMILY_ATTN_GDN: "fp8"},
        )
        scores, source = widened.resolve(
            uneven_perf.GEMM_FAMILY_MOE, uneven_perf.GEMM_FAMILY_ATTN_GDN
        )
        self.assertEqual(source, uneven_perf.GEMM_FAMILY_ATTN_GDN)
        self.assertEqual(scores, [566.88, 55.0, 55.2])


class _StubModel:
    """The three attributes ``_mlp_candidates`` reads. A stub, because the
    assertion is about the CANDIDATE LADDER reacting to the score vector, not
    about any particular checkpoint's geometry."""

    tp_size = 3
    families = {
        "mlp": uneven_perf._Family(17.113e9, 0.6, "mlp"),
        "attn": uneven_perf._Family(7.240e9, 1.0, "attn"),
    }

    def _shard_fractions(self, shard, mlp_vector):
        if shard == "mlp":
            total = float(sum(mlp_vector))
            return [v / total for v in mlp_vector]
        return [1.0 / self.tp_size] * self.tp_size


class TestTheObjectiveSeesTheDifference(CustomTestCase):
    """The point of the widening: a different score vector proposes a
    different split, not just a different log line."""

    def test_the_candidate_ladder_moves(self):
        widened = uneven_perf.rank_gemm_family_scores(
            _entries(),
            "nvfp4_a16",
            {
                uneven_perf.GEMM_FAMILY_MLP: "nvfp4_a16",
                uneven_perf.GEMM_FAMILY_ATTN_GDN: "fp8",
            },
        )
        model, base = _StubModel(), [1, 1, 1]
        from_scalar = uneven_perf._mlp_candidates(model, widened.scalar, base)
        from_attn = uneven_perf._mlp_candidates(
            model, widened.for_family(uneven_perf.GEMM_FAMILY_ATTN_GDN), base
        )
        self.assertNotEqual(from_scalar, from_attn)
        # The fp8 lane concentrates harder: its top candidate puts more units
        # on the compute-strong rank than the Marlin lane's does.
        self.assertGreater(max(c[0] for c in from_attn), max(c[0] for c in from_scalar))

    def test_a_uniform_checkpoint_produces_the_identical_ladder(self):
        widened = uneven_perf.rank_gemm_family_scores(_entries(), "fp8", {})
        model, base = _StubModel(), [1, 1, 1]
        scalar = uneven_perf.rank_gemm_scores(_entries(), "fp8")[0]
        self.assertEqual(
            uneven_perf._mlp_candidates(model, scalar, base),
            uneven_perf._mlp_candidates(
                model, widened.for_family(uneven_perf.GEMM_FAMILY_MLP), base
            ),
        )


class TestProfileShapeUntouched(CustomTestCase):
    """No PROFILE_VERSION bump, no new profile field -- new lane KEYS only."""

    def test_profile_version_and_fields_are_unchanged(self):
        self.assertEqual(uneven_perf.PROFILE_VERSION, 3)
        self.assertEqual(
            uneven_perf._PROFILE_VERSION_FIELDS[3],
            ("gemm_lanes", "gemm_lane_notes"),
        )

    def test_v2_to_v3_migration_still_lifts_a_lane_less_profile(self):
        old = _profile(lanes=False)
        old["version"] = 2
        for ent in old["gpus"].values():
            ent.pop("gemm_lanes", None)
            ent.pop("gemm_lane_notes", None)
        migrated, added = uneven_perf.migrate_profile(old)
        self.assertEqual(migrated["version"], uneven_perf.PROFILE_VERSION)
        # The lane fields are REPORTED missing for the lazy top-up, not
        # invented -- and the family widening did not add a field to that list.
        for fields in added.values():
            self.assertEqual(
                [f for f in fields if f.startswith("gemm")],
                ["gemm_lanes", "gemm_lane_notes"],
            )
        for uuid, ent in migrated["gpus"].items():
            self.assertNotIn("gemm_lanes", ent)
            self.assertEqual(ent["gemm_tflops"], _BF16[uuid])

    def test_every_new_lane_has_a_label(self):
        lanes = {uneven_perf.LANE_BF16}
        for group in uneven_perf._FORMAT_LANES.values():
            lanes.update(group)
        for lane in lanes:
            with self.subTest(lane=lane):
                self.assertIn(lane, uneven_perf._LANE_LABELS)

    def test_nvfp4_lane_order_mirrors_the_serving_dispatch(self):
        """``initialize_fp4_gemm_config`` resolves a native backend first and
        Marlin on sm_80..sm_89; a W4A16 checkpoint never reaches native."""
        self.assertEqual(
            uneven_perf._FORMAT_LANES["nvfp4_a4"],
            (uneven_perf.LANE_NVFP4_NATIVE, uneven_perf.LANE_NVFP4_MARLIN),
        )
        self.assertEqual(
            uneven_perf._FORMAT_LANES["nvfp4_a16"],
            (uneven_perf.LANE_NVFP4_MARLIN,),
        )


_CACHE = os.environ.get("HTSGLANG_TEST_MODEL_DIR", "")


@unittest.skipUnless(
    _CACHE and os.path.isdir(_CACHE),
    "HTSGLANG_TEST_MODEL_DIR not present",
)
class TestRealCheckpointsOnDisk(CustomTestCase):
    """Config reading against real checkpoints -- metadata only, no weights,
    no GPU. Pins that the detector's reach did not grow past NVFP4."""

    def _fmt(self, name):
        path = os.path.join(_CACHE, name)
        if not os.path.isdir(path):
            self.skipTest(f"{name} not present")
        return uneven_perf.checkpoint_compute_format_families(path)

    def test_compressed_tensors_nvfp4_is_recognised_as_w4a4(self):
        """Qwen3.6-27B-NVFP4 is nvfp4-pack-quantized with 4-bit FLOAT weights
        AND activations, one config group covering ``Linear``: the native lane
        is reachable in principle, and there is nothing to split per family."""
        key, _desc, families = self._fmt("Qwen3.6-27B-NVFP4")
        self.assertEqual(key, "nvfp4_a4")
        self.assertEqual(families, {})

    def test_the_other_schemes_keep_the_keys_they_had(self):
        for name, expected in (
            ("Qwen3.6-27B-FP8", "fp8"),
            ("Qwen3.6-27B-AWQ-BF16-INT4", "compressed-tensors"),
            ("Qwen3.5-122B-A10B-GPTQ-Int4", "gptq"),
            ("Huihui-Qwen3.6-27B-abliterated-AWQ-MTP", "awq"),
        ):
            with self.subTest(model=name):
                key, _desc, families = self._fmt(name)
                self.assertEqual(key, expected)
                self.assertEqual(families, {})


if __name__ == "__main__":
    unittest.main()
