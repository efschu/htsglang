"""The four open items #348b left behind, closed (task #359).

Each class below started as a FALSIFIER: it was written against the shipped
code, went red, and names the defect in its docstring. What it pins now is the
contract that replaced the defect.

1. ``lever_profiles`` divided two predicted times by a link rate nobody
   measured and compared the quotient against a move threshold. A user-visible
   throughput figure therefore depended on which of the three divergent
   fallback constants (1e-3 / 0.1 / 8.0 GB/s, 80x apart) happened to be in
   scope. The library now has no fallback constant at all: an absent pair
   matrix means the collective term is NOT PRICED, a compute-only figure may
   settle an argmax (the term is split-invariant, so omitting it cannot
   reorder) and may never settle a ratio against a threshold.
2. ``key_solver`` classified a checkpoint as ``"fp8"`` or ``"bf16"`` with its
   own reader and never called the #324 per-(rank, family) lane resolution, so
   an int8 / nvfp4 / W4A16 checkpoint was ranked on a measured but wrong-lane
   bf16 number.
3. The two on-disk pair-matrix shapes had a reconciler in the library and no
   caller: every consumer still read one shape with its own precedence, and a
   malformed row was dropped without a word.
4. A card the probe never scored entered ``RigRates`` as ``0.0`` -- clamped to
   ``1e-9`` downstream and raised to a fractional exponent, i.e. an absent
   measurement reading as an extremely slow but valid card.

All CPU, all on inlined fixtures.
"""

import json
import os
import tempfile
import unittest
from unittest import mock

from sglang.srt import uneven_perf
from sglang.srt.planner import cost_model as cm
from sglang.srt.planner import key_solver as ks
from sglang.srt.planner import lever_profiles as lp
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=25, suite="base-a-test-cpu")


# ---------------------------------------------------------------------------
# Fixtures: the reference rig, in both on-disk shapes
# ---------------------------------------------------------------------------

#: Measured GEMM lanes of the #354 window, the same numbers
#: ``test_phase_optimal_targets`` pins. The card probe cannot produce these --
#: it measures dense bf16 and native fp8 only -- so they are what the hardware
#: profile contributes to the solver's lane resolution.
_LANES = {
    "GPU-5090": {
        "fp8_native": 566.88,
        "fp8_marlin": 216.34,
        "fp8_w8a16": 181.43,
        "int8_native": 676.69,
    },
    "GPU-3080a": {"fp8_marlin": 58.44, "fp8_w8a16": 53.43, "int8_native": 183.78},
    "GPU-3080b": {"fp8_marlin": 59.15, "fp8_w8a16": 53.78, "int8_native": 164.77},
}

_PROBE_CARDS = [
    {
        "uuid": "GPU-5090",
        "name": "NVIDIA GeForce RTX 5090",
        "cuda_index": 0,
        "total_mib": 32607,
        "gemm_bf16_tflops": 231.97,
        "gemm_fp8_tflops": 566.88,
        "membw_read_gbs": 1660.4,
        "membw_gemv_gbs": 1533.8,
        "h2d_gbs": 14.41,
        "d2h_gbs": 14.26,
    },
    {
        "uuid": "GPU-3080a",
        "name": "NVIDIA GeForce RTX 3080",
        "cuda_index": 1,
        "total_mib": 20480,
        "gemm_bf16_tflops": 65.57,
        "gemm_fp8_tflops": None,
        "membw_read_gbs": 717.0,
        "membw_gemv_gbs": 717.4,
        "h2d_gbs": 6.47,
        "d2h_gbs": 6.58,
    },
    {
        "uuid": "GPU-3080b",
        "name": "NVIDIA GeForce RTX 3080",
        "cuda_index": 2,
        "total_mib": 20480,
        "gemm_bf16_tflops": 65.59,
        "gemm_fp8_tflops": None,
        "membw_read_gbs": 717.1,
        "membw_gemv_gbs": 717.8,
        "h2d_gbs": 13.4,
        "d2h_gbs": 13.16,
    },
]

_PAIR_ROWS = (
    ("GPU-5090", "GPU-3080a", 4.44, 22.4),
    ("GPU-5090", "GPU-3080b", 6.91, 19.8),
    ("GPU-3080a", "GPU-5090", 4.52, 22.1),
    ("GPU-3080a", "GPU-3080b", 4.41, 21.5),
    ("GPU-3080b", "GPU-5090", 6.88, 19.5),
    ("GPU-3080b", "GPU-3080a", 4.32, 21.6),
)

_PROBE = {
    "cards": _PROBE_CARDS,
    "pairs": [
        {
            "src_uuid": a,
            "dst_uuid": b,
            "bandwidth_gbs": bw,
            "latency_us": lat,
            "transport": "host staging (pinned)",
        }
        for a, b, bw, lat in _PAIR_ROWS
    ],
}


def _profile_with_lanes():
    """The hardware profile shape: per-card ``gemm_lanes`` plus the unordered
    link map. This is the artifact the #324 lane resolution reads."""
    gpus = {}
    for card in _PROBE_CARDS:
        gpus[card["uuid"]] = {
            "name": card["name"],
            "cuda_index": card["cuda_index"],
            "total_mib": card["total_mib"],
            "gemm_tflops": card["gemm_bf16_tflops"],
            "membw_gbs": card["membw_read_gbs"],
            "membw_gemv_gbs": card["membw_gemv_gbs"],
            "gemm_lanes": dict(_LANES[card["uuid"]]),
            "gemm_lane_notes": {},
        }
    return {
        "version": 3,
        "gpus": gpus,
        "links": {
            "GPU-3080a|GPU-5090": {"p2p_gbs": 4.48},
            "GPU-3080b|GPU-5090": {"p2p_gbs": 6.90},
            "GPU-3080a|GPU-3080b": {"p2p_gbs": 4.37},
            "__group__": {"ar_10kb_us": 88.1, "ar_1mb_us": 512.4},
        },
    }


#: Qwen3.6-27B geometry, the same text config the other planner fixtures use.
_QWEN36_27B_TEXT = dict(
    model_type="qwen3_5",
    hidden_size=5120,
    num_hidden_layers=64,
    num_attention_heads=24,
    num_key_value_heads=4,
    head_dim=256,
    intermediate_size=17408,
    vocab_size=248064,
    linear_num_key_heads=16,
    linear_num_value_heads=48,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
    full_attention_interval=4,
    max_position_embeddings=262144,
    layer_types=[
        "full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(64)
    ],
)

_FP8_QUANT = {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic"}

#: Avesed/Qwen3.6-27B-INT8-W8A8, trimmed to the keys the format detection reads.
_INT8_QUANT = {
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


def _write_checkpoint(tmpdir: str, quant) -> str:
    cfg = dict(
        architectures=["Qwen3_5ForConditionalGeneration"],
        model_type="qwen3_5",
        text_config=dict(_QWEN36_27B_TEXT),
    )
    if quant is not None:
        cfg["quantization_config"] = dict(quant)
    with open(os.path.join(tmpdir, "config.json"), "w") as f:
        json.dump(cfg, f)
    return tmpdir


class _Checkpoints(CustomTestCase):
    """Three checkpoints on disk, differing only in ``quantization_config``."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._tmp = tempfile.TemporaryDirectory()
        for name, quant in (
            ("bf16", None),
            ("fp8", _FP8_QUANT),
            ("int8", _INT8_QUANT),
        ):
            path = os.path.join(cls._tmp.name, name)
            os.makedirs(path, exist_ok=True)
            setattr(cls, name, _write_checkpoint(path, quant))
        cls._env = mock.patch.dict(
            os.environ, {"SGLANG_MAMBA_SSM_DTYPE": "bfloat16"}, clear=False
        )
        cls._env.start()

    @classmethod
    def tearDownClass(cls):
        cls._env.stop()
        cls._tmp.cleanup()
        super().tearDownClass()

    @staticmethod
    def _inputs(model_path):
        from sglang.srt.uneven_perf import PlanInputs

        return PlanInputs(
            tp_size=3,
            model_path=model_path,
            kv_cache_dtype="fp8_e4m3",
            max_running_requests=16,
            rank_gpu_id=[0, 1, 2],
            effective_vram_mib=[32607 - 3000, 20480 - 2700, 20480 - 2700],
        )


# ---------------------------------------------------------------------------
# Item 1 -- an unmeasured link must not drive a ratio or a threshold
# ---------------------------------------------------------------------------


class TestAbsentLinkNeverRanks(CustomTestCase):
    """FALSIFIER (red before #359).

    With no pair matrix on disk, ``lever_profiles`` fell back to
    ``_FALLBACK_LINK_GBS = 8.0``, divided two predicted prefill times by it and
    reported the quotient as a throughput figure. Swapping the constant for
    ``0.1`` -- the floor the SAME arithmetic used one module over -- moved that
    published number on identical measured inputs. A figure that moves when an
    unmeasured constant moves is not a measurement of anything.
    """

    def test_no_fallback_link_constant_survives_in_the_library(self):
        self.assertFalse(hasattr(cm, "ABSENT_LINK_ASSUMED_GBS"))
        self.assertFalse(hasattr(cm, "ABSENT_LINK_RANKING_PLACEHOLDER_GBS"))
        self.assertFalse(hasattr(lp, "_FALLBACK_LINK_GBS"))

    def test_the_prefill_model_refuses_a_non_positive_link_instead_of_flooring(self):
        """``max(min_link_gbs, 0.1)`` silently rescued any nonsense below it --
        including the 1e-3 placeholder the key solver passed, which therefore
        never reached the arithmetic it was documented to drive."""
        model = _tiny_cost_model()
        with self.assertRaises(ValueError):
            model.prefill_time_model([2, 1, 1], [200.0, 60.0, 60.0], 0.0)
        with self.assertRaises(ValueError):
            model.prefill_time_model([2, 1, 1], [200.0, 60.0, 60.0], -1.0)

    def test_an_absent_link_prices_compute_only_and_cannot_reorder(self):
        """``None`` = the collective is not priced. The omitted term is
        split-invariant, so the compute-only ORDER is the order at any link
        rate -- which is what makes an argmax still answerable."""
        model = _tiny_cost_model()
        gemm = [231.97, 65.57, 65.59]
        cands = [[2, 1, 1], [4, 1, 1], [6, 1, 1], [1, 1, 1]]
        compute_only = sorted(
            cands, key=lambda c: model.prefill_time_model(c, gemm, None)
        )
        for bw in (0.1, 4.32, 8.0, 1000.0):
            self.assertEqual(
                compute_only,
                sorted(cands, key=lambda c: model.prefill_time_model(c, gemm, bw)),
                bw,
            )

    def test_a_published_prefill_figure_no_longer_moves_with_an_unmeasured_number(self):
        """The end-to-end property: with no link measurement anywhere, every
        prefill throughput figure the lever surface publishes for a non-base
        vector is ABSENT with a reason, so there is nothing left for a
        fabricated constant to move."""
        report = _lever_report_without_a_link()
        for row in report["profiles"]:
            if not row.get("resolved") or row["is_base_split"]:
                continue
            cell = [m for m in row["metrics"] if m["key"] == "prefill_tok_s"][0]
            self.assertFalse(cell["available"], row["key"])
            self.assertIn("pair matrix", cell["reason"])

    def test_the_prefill_objective_reports_itself_unresolved_without_a_link(self):
        """The move threshold (1 % noise floor) is a comparison of a RATIO
        against a constant. A compute-only ratio overstates the move by exactly
        the collective term, so it may not settle that comparison either."""
        report = _lever_report_without_a_link()
        row = [p for p in report["profiles"] if p["key"] == "max_prefill"][0]
        self.assertTrue(row["objective_unresolved"])
        self.assertIn("pair matrix", row["selection_reason"])

    def test_the_link_basis_states_the_absence(self):
        report = _lever_report_without_a_link()
        self.assertEqual(report["basis"]["min_link_source"].split()[0], "absent")


def _tiny_cost_model():
    from sglang.srt.uneven_perf import PerfCostModel, PlanInputs

    tmp = tempfile.mkdtemp()
    path = _write_checkpoint(tmp, None)
    inputs = PlanInputs(
        tp_size=3,
        model_path=path,
        kv_cache_dtype="fp8_e4m3",
        max_running_requests=16,
        rank_gpu_id=[0, 1, 2],
        effective_vram_mib=[29607, 17780, 17780],
    )
    with mock.patch.dict(os.environ, {"SGLANG_MAMBA_SSM_DTYPE": "bfloat16"}):
        return PerfCostModel(inputs, [1, 1, 1], [29607, 17780, 17780])


def _lever_report_without_a_link():
    """The lever surface on a rig whose link matrix was never measured."""
    from sglang.srt.planner import webui

    tmp = tempfile.mkdtemp()
    model = _write_checkpoint(tmp, _FP8_QUANT)
    payload = {
        "model": model,
        "hardware": {
            "source": "manual",
            "gpus": ["RTX 5090:32760", "RTX 3080:20480", "RTX 3080:20480"],
        },
        "tp_size": 3,
        "kv_cache_dtype": "fp8_e4m3",
        "max_running_requests": 16,
    }
    with mock.patch.object(lp, "_load_pair_matrix", return_value=(None, [])):
        with mock.patch.dict(os.environ, {"SGLANG_MAMBA_SSM_DTYPE": "bfloat16"}):
            report = webui.lever_profiles_payload(payload)
    assert report.get("ok"), report.get("reasons")
    return report


# ---------------------------------------------------------------------------
# Item 2 -- the solver scores on the checkpoint's own GEMM lane
# ---------------------------------------------------------------------------


class TestSolverUsesTheLaneResolution(_Checkpoints):
    """FALSIFIER (red before #359).

    ``build_cost_model`` called ``rates.resolve_gemm_dtype(
    gemm_dtype_for_checkpoint(path))``, a binary fp8/bf16 classifier over its
    own ``config.json`` read. An INT8 W8A8 checkpoint answered ``"bf16"`` and
    was priced at 231.97 / 65.57 / 65.59 TFLOP/s -- measured numbers, wrong
    lane -- while the profile held the int8 lane it will actually dispatch on
    (676.69 / 183.78 / 164.77, a 3.68:1 ratio against 3.54:1).
    """

    def test_an_int8_checkpoint_is_priced_on_the_int8_lane(self):
        rates = ks.rates_from_probe(
            _PROBE, [0, 1, 2], hardware_profile=_profile_with_lanes()
        )
        model = ks.build_cost_model(
            self._inputs(self.int8), [1, 1, 1], [29607, 17780, 17780], rates
        )
        self.assertEqual(model.rates.gemm_format, "int8")
        self.assertEqual(
            [round(x, 2) for x in model.rates.gemm_tflops],
            [676.69, 183.78, 164.77],
        )

    def test_a_bf16_checkpoint_is_byte_identical_to_the_old_classifier(self):
        """The binary classifier was RIGHT for bf16 and fp8; only the formats
        it could not name were mispriced. Both must come through the new path
        unchanged, or this fix would be a re-tune wearing a bugfix's label."""
        rates = ks.rates_from_probe(_PROBE, [0, 1, 2])
        old = rates.resolve_gemm_dtype(ks.gemm_dtype_for_checkpoint(self.bf16))
        model = ks.build_cost_model(
            self._inputs(self.bf16), [1, 1, 1], [29607, 17780, 17780], rates
        )
        self.assertEqual(model.rates.gemm_tflops, old.gemm_tflops)
        self.assertEqual(model.rates.gemm_tflops, [231.97, 65.57, 65.59])

    def test_an_fp8_checkpoint_is_byte_identical_on_the_card_probe_alone(self):
        rates = ks.rates_from_probe(_PROBE, [0, 1, 2])
        old = rates.resolve_gemm_dtype(ks.gemm_dtype_for_checkpoint(self.fp8))
        model = ks.build_cost_model(
            self._inputs(self.fp8), [1, 1, 1], [29607, 17780, 17780], rates
        )
        self.assertEqual(model.rates.gemm_tflops, old.gemm_tflops)
        # 5090 on its native fp8 path, both 3080s on the loud dense fallback:
        # exactly what the binary classifier resolved, card for card.
        self.assertEqual(model.rates.gemm_tflops, [566.88, 65.57, 65.59])

    def test_the_fp8_marlin_lane_is_taken_when_the_profile_measured_it(self):
        """The deliberate delta: an Ampere card serving fp8 runs weight-only
        Marlin, not dense bf16. The card probe cannot measure that lane; the
        hardware profile can, and when it has, the solver scores on it."""
        rates = ks.rates_from_probe(
            _PROBE, [0, 1, 2], hardware_profile=_profile_with_lanes()
        )
        model = ks.build_cost_model(
            self._inputs(self.fp8), [1, 1, 1], [29607, 17780, 17780], rates
        )
        self.assertEqual(
            [round(x, 2) for x in model.rates.gemm_tflops], [566.88, 58.44, 59.15]
        )

    def _solve(self, model_path, goal, profile=None):
        rates = ks.rates_from_probe(_PROBE, [0, 1, 2], hardware_profile=profile)
        budgets = [32607 - 3000, 20480 - 2700, 20480 - 2700]
        answer = ks.solve(self._inputs(model_path), budgets, budgets, rates, goal=goal)
        best = answer.candidates[0]
        return list(best.units), round(best.raw["enc"], 6)

    def test_the_int8_lane_moves_the_prefill_key_and_this_is_the_delta(self):
        """The one plan #359 deliberately changes, with the lane that moved it.

        Dense bf16 reads the two 3080s as near-equal (65.57 and 65.59, 0.03 %
        apart). Their int8 lanes are not: 183.78 and 164.77, 10 % apart. The
        prefill key therefore stops splitting them evenly and moves four MLP
        units off the slower card. Nothing else in the solve changes -- the
        capacity, decode and session keys are the same vectors -- because the
        GEMM rate does not enter their objective.
        """
        profile = _profile_with_lanes()
        self.assertEqual(self._solve(self.int8, "enc"), ([104, 15, 17], 1.130001))
        self.assertEqual(
            self._solve(self.int8, "enc", profile), ([108, 15, 13], 1.074886)
        )
        for goal in ("maxkv", "dec", "sessions"):
            self.assertEqual(
                self._solve(self.int8, goal)[0],
                self._solve(self.int8, goal, profile)[0],
                goal,
            )

    def test_an_fp8_key_does_not_move_only_its_reported_ratio_does(self):
        """The fp8 Marlin lane widens the rank ratio from 8.64:1 to 9.70:1, so
        the predicted prefill gain of concentrating on rank 0 rises. The
        integer key at the optimum is the same vector either way, which is the
        difference between a re-tune and a better-measured prediction."""
        profile = _profile_with_lanes()
        self.assertEqual(self._solve(self.fp8, "enc"), ([136, 0, 0], 1.23466))
        self.assertEqual(self._solve(self.fp8, "enc", profile), ([136, 0, 0], 1.252042))

    def test_a_bf16_key_and_ratio_are_untouched_by_the_profile(self):
        profile = _profile_with_lanes()
        for goal in ("maxkv", "dec", "enc", "sessions"):
            self.assertEqual(
                self._solve(self.bf16, goal),
                self._solve(self.bf16, goal, profile),
                goal,
            )

    def test_a_wrong_lane_fallback_is_loud(self):
        """Without the profile there is no int8 lane to score on. The answer is
        the dense number WITH a warning naming the format -- not the silent
        substitution the binary classifier made."""
        rates = ks.rates_from_probe(_PROBE, [0, 1, 2])
        model = ks.build_cost_model(
            self._inputs(self.int8), [1, 1, 1], [29607, 17780, 17780], rates
        )
        self.assertEqual(model.rates.gemm_tflops, [231.97, 65.57, 65.59])
        self.assertTrue(model.rates.gemm_warnings)
        self.assertIn("int8", " ".join(model.rates.gemm_warnings))


# ---------------------------------------------------------------------------
# Item 3 -- one pair-matrix shape at the library boundary
# ---------------------------------------------------------------------------


class TestOnePairMatrixShape(CustomTestCase):
    """FALSIFIER (red before #359).

    ``reconcile_pair_matrices`` and ``pair_matrix_from_hardware_profile``
    existed and had zero production callers (grep). Every consumer still read
    one shape with its own precedence, so the ordered card-probe matrix and the
    unordered NCCL link map could disagree about the same wire with nothing
    comparing them. Malformed rows were dropped by a bare ``continue``.
    """

    def test_the_inlined_lane_key_matches_the_real_one(self):
        """``cost_model`` stays stdlib-only at import time, so it spells the
        fp8 lane key itself. This is the pin that keeps the copy honest."""
        self.assertEqual(cm._LANE_FP8_NATIVE, uneven_perf.LANE_FP8_NATIVE)

    def test_a_malformed_row_is_rejected_loudly_not_dropped(self):
        profile = _profile_with_lanes()
        profile["links"]["GPU-5090|GPU-3080a|GPU-3080b"] = {"p2p_gbs": 99.0}
        profile["links"]["not-a-pair-key"] = {"p2p_gbs": 99.0}
        matrix = cm.pair_matrix_from_hardware_profile(
            profile, ["GPU-5090", "GPU-3080a", "GPU-3080b"]
        )
        self.assertEqual(len(matrix.rejected), 2)
        self.assertTrue(all("malformed" in r for r in matrix.rejected))

    def test_a_probe_row_without_endpoints_is_rejected_loudly(self):
        probe = {"cards": _PROBE_CARDS, "pairs": list(_PROBE["pairs"])}
        probe["pairs"].append({"bandwidth_gbs": 99.0, "latency_us": 1.0})
        matrix = cm.pair_matrix_from_card_probe(
            probe, ["GPU-5090", "GPU-3080a", "GPU-3080b"]
        )
        self.assertEqual(len(matrix.rejected), 1)
        self.assertIn("malformed", matrix.rejected[0])

    def test_one_boundary_reads_both_shapes_and_names_the_disagreement(self):
        keys = ["GPU-5090", "GPU-3080a", "GPU-3080b"]
        matrix, divergences = cm.load_pair_matrix(
            keys, card_probe=_PROBE, hardware_profile=_profile_with_lanes()
        )
        # The ordered artifact wins: it measures the direction a collective
        # takes, and the 4.52 / 6.88 asymmetry the unordered shape cannot hold.
        self.assertAlmostEqual(
            matrix.hop("GPU-3080a", "GPU-5090").bandwidth_gbs.require("bw"), 4.52
        )
        self.assertAlmostEqual(
            matrix.hop("GPU-3080b", "GPU-5090").bandwidth_gbs.require("bw"), 6.88
        )
        self.assertEqual(divergences, [])

    def test_a_real_disagreement_is_reported_rather_than_averaged(self):
        profile = _profile_with_lanes()
        profile["links"]["GPU-3080a|GPU-5090"] = {"p2p_gbs": 12.0}
        keys = ["GPU-5090", "GPU-3080a", "GPU-3080b"]
        matrix, divergences = cm.load_pair_matrix(
            keys, card_probe=_PROBE, hardware_profile=profile
        )
        self.assertTrue(divergences)
        self.assertIn("GPU-3080a -> GPU-5090", " ".join(divergences))
        self.assertAlmostEqual(
            matrix.hop("GPU-3080a", "GPU-5090").bandwidth_gbs.require("bw"), 4.52
        )

    def test_the_lever_surface_reads_the_shared_boundary(self):
        """The adapter at the caller: ``lever_profiles`` no longer carries its
        own two-shape precedence."""
        import inspect

        src = inspect.getsource(lp)
        self.assertNotIn("links.items()", src)
        self.assertIn("_load_pair_matrix", src)


# ---------------------------------------------------------------------------
# Item 4 -- no silent 0.0
# ---------------------------------------------------------------------------


class TestNoSilentZero(CustomTestCase):
    """FALSIFIER (red before #359).

    ``rates_from_probe`` named a missing streaming bandwidth / GEMM rate in
    ``absent`` (#348b) but still put ``0.0`` into the rate vector. The name was
    for the reader; the arithmetic downstream got the zero, clamped it to
    ``1e-9`` and raised it to a fractional exponent -- an absent card reading
    as an extremely slow but perfectly valid one.
    """

    @staticmethod
    def _probe_missing(field):
        cards = [dict(c) for c in _PROBE_CARDS]
        for name in (field if isinstance(field, tuple) else (field,)):
            cards[1].pop(name, None)
        return {"cards": cards, "pairs": _PROBE["pairs"]}

    def test_a_card_without_a_streaming_rate_has_no_rate_at_all(self):
        rates = ks.rates_from_probe(
            self._probe_missing(("membw_read_gbs", "membw_gbs")), [0, 1, 2]
        )
        self.assertIsNone(rates.membw_gbs[1])
        self.assertTrue(any("streaming" in a for a in rates.absent))

    def test_a_card_without_a_gemm_rate_has_no_rate_at_all(self):
        rates = ks.rates_from_probe(
            self._probe_missing(("gemm_bf16_tflops", "gemm_tflops")), [0, 1, 2]
        )
        self.assertIsNone(rates.gemm_tflops[1])
        self.assertTrue(any("GEMM" in a for a in rates.absent))

    def test_a_consumer_of_an_absent_rate_fails_loudly(self):
        rates = ks.rates_from_probe(
            self._probe_missing(("membw_read_gbs", "membw_gbs")), [0, 1, 2]
        )
        with self.assertRaises(cm.AbsentRate):
            rates.require_membw_gbs()
        rates = ks.rates_from_probe(
            self._probe_missing(("gemm_bf16_tflops", "gemm_tflops")), [0, 1, 2]
        )
        with self.assertRaises(cm.AbsentRate):
            rates.require_gemm_tflops()

    def test_a_complete_probe_is_unchanged(self):
        rates = ks.rates_from_probe(_PROBE, [0, 1, 2])
        self.assertEqual(rates.absent, [])
        self.assertEqual(rates.require_membw_gbs(), [1660.4, 717.0, 717.1])
        self.assertEqual(rates.require_gemm_tflops(), [231.97, 65.57, 65.59])


if __name__ == "__main__":
    unittest.main()
