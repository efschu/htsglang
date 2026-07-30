"""Which GEMM the prefill objective scores the ranks on (task #298a).

The ``enc``/``both`` objective of ``--rank-tp-ratio auto-performance`` is a
compute RATIO between ranks. It read that ratio off a dense **bf16** matmul
while the checkpoint being planned was **fp8**, and the two are not the same
number: the #213 card probe measures 566.88 fp8 TFLOPS on the reference rig's
5090 (sm_120, native tensor path) against 231.97 bf16, and the sm_86 3080s have
no fp8 tensor path at all. The objective therefore saw 3.79 where the fp8
checkpoint runs at 8.64+, and proposed a 6,2,2-class split against a measured
optimum of 10,1,1 (#296, ``fp8_objective_audit.md``).

``null`` is not the right answer for the sm_86 cards either -- they serve the
same fp8 checkpoint, through the weight-only Marlin GEMM by default and through
the W8A16 dequant lane when Marlin is unreachable or switched off for
determinism (#190). Both are timed, so a card without fp8 tensor cores gets a
MEASURED score in the checkpoint's format rather than a bf16 stand-in.

What is asserted here, all on CPU with rigged probe values:

* the format is read from the checkpoint's own config, never from its path;
* an fp8 checkpoint picks the fp8 lanes, per card, in the order the serving
  path tries them;
* a bf16 checkpoint is byte-identically unchanged;
* every fallback -- format with no lane table, card with no measured lane --
  is loud, and returns the pre-existing bf16 score rather than a new one.
"""

import json
import os
import tempfile
import types
import unittest
from unittest import mock

from sglang.srt import uneven_perf
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

_CACHE = os.environ.get("HTSGLANG_TEST_MODEL_DIR", "")
_MODEL = os.path.join(_CACHE, "Qwen3.6-27B-FP8") if _CACHE else ""

#: NVML totals of the reference rig in rank order (5090, 3080, 3080).
_TOTALS = [32607, 20480, 20480]

#: Measured rates of the reference rig. The bf16 and fp8-native figures are the
#: #213 card probe's; the weight-only figure for the 3080s stands in for the
#: lane probe that has not run on a GPU yet, and is deliberately set BELOW that
#: card's bf16 rate -- a dequantising lane cannot outrun the dense matmul it
#: feeds into.
_BF16 = {"U0": 232.97, "U1": 62.72, "U2": 62.98}
_FP8_NATIVE = {"U0": 566.88}
_FP8_W8A16 = {"U1": 55.0, "U2": 55.2}


def _profile(lanes=True):
    """A PROFILE_VERSION-3 profile of the reference rig, with or without the
    quantized GEMM lanes (``lanes=False`` is what a v2 cache looks like)."""
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
            ent["gemm_lanes"] = values
            ent["gemm_lane_notes"] = notes
        gpus[uuid] = ent
    return {
        "version": uneven_perf.PROFILE_VERSION,
        "driver": "595.58.03",
        "gpus": gpus,
        "links": {
            "U0|U1": {"p2p_gbs": 5.1},
            "U0|U2": {"p2p_gbs": 9.06},
            "U1|U2": {"p2p_gbs": 5.83},
            "__group__": {"ar_10kb_us": 32.4, "ar_1mb_us": 361.3},
        },
    }


_GPUS = [
    {"uuid": "U0", "cuda_index": 0, "name": "RTX 5090", "total_mib": 32607},
    {"uuid": "U1", "cuda_index": 1, "name": "RTX 3080", "total_mib": 20480},
    {"uuid": "U2", "cuda_index": 2, "name": "RTX 3080", "total_mib": 20480},
]


def _entries(lanes=True):
    p = _profile(lanes=lanes)
    return [p["gpus"][u] for u in ("U0", "U1", "U2")]


def _write_config(tmp, quantization_config, nested=False):
    """A minimal checkpoint directory carrying just the quantization block."""
    cfg = {"hidden_size": 5120}
    if quantization_config is not None:
        if nested:
            cfg["text_config"] = {"quantization_config": quantization_config}
        else:
            cfg["quantization_config"] = quantization_config
    with open(os.path.join(tmp, "config.json"), "w") as f:
        json.dump(cfg, f)
    return tmp


class TestCheckpointComputeFormat(CustomTestCase):
    """The format key comes from the config, not from the directory name."""

    def test_fp8_block_scaled(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_config(
                tmp,
                {
                    "quant_method": "fp8",
                    "fmt": "e4m3",
                    "weight_block_size": [128, 128],
                },
            )
            key, desc = uneven_perf.checkpoint_compute_format(tmp)
        self.assertEqual(key, "fp8")
        self.assertIn("weight_block_size [128, 128]", desc)

    def test_fp8_declared_only_by_format_string(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_config(tmp, {"quant_method": "compressed-tensors", "format": "float-quantized"})
            key, _desc = uneven_perf.checkpoint_compute_format(tmp)
        self.assertEqual(key, "fp8")

    def test_vl_checkpoint_keeps_the_block_under_text_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_config(tmp, {"quant_method": "fp8", "fmt": "e4m3"}, nested=True)
            key, _desc = uneven_perf.checkpoint_compute_format(tmp)
        self.assertEqual(key, "fp8")

    def test_unquantized_checkpoint_is_bf16(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_config(tmp, None)
            key, desc = uneven_perf.checkpoint_compute_format(tmp)
        self.assertEqual(key, "bf16")
        self.assertIn("unquantized", desc)

    def test_int_schemes_keep_their_own_name(self):
        for method in ("gptq", "awq", "compressed-tensors"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp:
                _write_config(tmp, {"quant_method": method, "bits": 4})
                key, _desc = uneven_perf.checkpoint_compute_format(tmp)
                self.assertEqual(key, method)
                self.assertNotIn(key, uneven_perf._FORMAT_LANES)

    def test_a_directory_named_fp8_holding_int4_is_not_fp8(self):
        """The path name is not evidence. A repo called ``...-FP8`` that ships
        int4 weights would otherwise be scored on a lane it never runs."""
        with tempfile.TemporaryDirectory() as parent:
            tmp = os.path.join(parent, "Qwen3.6-27B-FP8")
            os.makedirs(tmp)
            _write_config(tmp, {"quant_method": "gptq", "bits": 4})
            key, _desc = uneven_perf.checkpoint_compute_format(tmp)
        self.assertEqual(key, "gptq")

    def test_unreadable_config_is_unknown_not_bf16(self):
        with tempfile.TemporaryDirectory() as tmp:
            key, desc = uneven_perf.checkpoint_compute_format(tmp)
        self.assertEqual(key, "unknown")
        self.assertIn("unreadable", desc)
        self.assertNotIn(key, uneven_perf._FORMAT_LANES)


class TestLaneDispatch(CustomTestCase):
    """``rank_gemm_scores``: one lane per card, in the serving path's order."""

    def test_fp8_checkpoint_scores_each_card_on_its_own_lane(self):
        scores, labels, warnings = uneven_perf.rank_gemm_scores(_entries(), "fp8")
        self.assertEqual(scores, [566.88, 55.0, 55.2])
        self.assertIn("native", labels[0])
        self.assertIn("dequant", labels[1])
        self.assertEqual(warnings, [])

    def test_the_ratio_the_objective_sees_is_the_fp8_ratio(self):
        """The whole point: bf16 compresses the spread by more than 2x."""
        bf16, _l, _w = uneven_perf.rank_gemm_scores(_entries(), "bf16")
        fp8, _l2, _w2 = uneven_perf.rank_gemm_scores(_entries(), "fp8")
        self.assertAlmostEqual(bf16[0] / bf16[1], 3.71, places=2)
        self.assertAlmostEqual(fp8[0] / fp8[1], 10.31, places=2)

    def test_lane_preference_follows_the_serving_order(self):
        """native > Marlin > dequant, per card, exactly as fp8_utils tries
        them (``fp8_needs_dequant_fallback``)."""
        entries = _entries()
        entries[1]["gemm_lanes"][uneven_perf.LANE_FP8_MARLIN] = 61.0
        scores, labels, _w = uneven_perf.rank_gemm_scores(entries, "fp8")
        self.assertEqual(scores[1], 61.0)
        self.assertIn("Marlin", labels[1])

        entries[0]["gemm_lanes"][uneven_perf.LANE_FP8_MARLIN] = 300.0
        scores, labels, _w = uneven_perf.rank_gemm_scores(entries, "fp8")
        self.assertEqual(scores[0], 566.88, "native must win over Marlin")

    def test_bf16_checkpoint_is_unchanged(self):
        """Backward compatibility: an unquantized checkpoint takes the dense
        probe, byte-identically, with no warning and no new number."""
        for lanes in (True, False):
            with self.subTest(profile_has_lanes=lanes):
                scores, labels, warnings = uneven_perf.rank_gemm_scores(
                    _entries(lanes=lanes), "bf16"
                )
                self.assertEqual(scores, [232.97, 62.72, 62.98])
                self.assertEqual(labels, ["dense bf16"] * 3)
                self.assertEqual(warnings, [])


class TestFallbacksAreLoud(CustomTestCase):
    """A bf16 number wearing a quantized label is the defect under repair, so
    every path back to bf16 says so and none of them changes a value."""

    def test_format_without_a_lane_table_warns_and_keeps_bf16(self):
        scores, labels, warnings = uneven_perf.rank_gemm_scores(_entries(), "gptq")
        self.assertEqual(scores, [232.97, 62.72, 62.98])
        self.assertEqual(labels, ["dense bf16"] * 3)
        self.assertEqual(len(warnings), 1)
        self.assertIn("gptq", warnings[0])
        self.assertIn("DENSE BF16", warnings[0])

    def test_profile_without_lanes_warns_per_card_and_keeps_bf16(self):
        """A cache written before the lane probes existed."""
        scores, labels, warnings = uneven_perf.rank_gemm_scores(
            _entries(lanes=False), "fp8"
        )
        self.assertEqual(scores, [232.97, 62.72, 62.98])
        self.assertTrue(all("fallback" in x for x in labels))
        self.assertEqual(len(warnings), 3)
        self.assertIn("SGLANG_PERF_REPROBE=1", warnings[0])

    def test_a_card_with_no_lane_reports_the_reason_it_has_none(self):
        entries = _entries()
        entries[1]["gemm_lanes"] = {}
        _s, _l, warnings = uneven_perf.rank_gemm_scores(entries, "fp8")
        self.assertEqual(len(warnings), 1)
        self.assertIn("RTX 3080", warnings[0])
        self.assertIn("compute capability 8.6", warnings[0])


class TestLaneTablesAgree(CustomTestCase):
    """The three tables that describe a lane must describe the same lanes."""

    def test_every_fp8_lane_has_a_probe(self):
        for lane in uneven_perf._FORMAT_LANES["fp8"]:
            with self.subTest(lane=lane):
                self.assertIn(lane, uneven_perf._LANE_PROBES)

    def test_every_lane_has_a_label(self):
        lanes = {uneven_perf.LANE_BF16}
        for group in uneven_perf._FORMAT_LANES.values():
            lanes.update(group)
        lanes.update(uneven_perf._LANE_PROBES)
        for lane in lanes:
            with self.subTest(lane=lane):
                self.assertIn(lane, uneven_perf._LANE_LABELS)

    def test_the_bf16_format_takes_the_dense_lane_alone(self):
        self.assertEqual(
            uneven_perf._FORMAT_LANES["bf16"], (uneven_perf.LANE_BF16,)
        )


class TestLaneProbesNeverSubstituteANumber(CustomTestCase):
    """Run on CPU: no lane can execute, so each probe must return its REASON.

    The contract the audit turns on -- an unmeasurable lane is absent, never
    filled in from a neighbouring measurement -- is exactly what this asserts.
    """

    def test_each_probe_returns_none_with_a_reason(self):
        for lane, probe in uneven_perf._LANE_PROBES.items():
            with self.subTest(lane=lane):
                value, note = probe("cuda:0")
                self.assertIsNone(value)
                self.assertTrue(note.strip(), "an absent lane must carry a reason")

    def test_bench_gemm_lanes_records_reasons_not_values(self):
        values, notes = uneven_perf._bench_gemm_lanes("cuda:0")
        self.assertEqual(values, {})
        self.assertEqual(set(notes), set(uneven_perf._LANE_PROBES))


def _args(tune="enc", loose=0.0, model=None):
    """A duck-typed ServerArgs carrying the fields the optimizer reads.

    Same shape as ``test_perf_tune_targets._args``; kept local so a change to
    that fixture's reserve/demand story cannot silently move this one.
    """
    reserve = (4500, 4200, 4200)
    budgets = [t - r for t, r in zip(_TOTALS, reserve)]
    demand = 4160
    sa = types.SimpleNamespace(
        model_path=model or _MODEL,
        tp_size=3,
        rank_gpu_id=[0, 1, 2],
        rank_gpu_memory_mib=list(budgets),
        rank_tp_ratio=list(budgets),
        rank_mlp_ratio=None,
        rank_vocab_ratio=None,
        rank_moe_ratio=None,
        rank_kv_ratio="coupled",
        rank_kv_capacity_seed=None,
        rank_auto_reserve_mib=",".join(map(str, reserve)),
        rank_perf_tune=tune,
        rank_perf_loose_ctx_percent=loose,
        kv_cache_dtype="fp8_e4m3",
        context_length=32768,
        page_size=1,
        quantization=None,
        max_running_requests=16,
        chunked_prefill_size=2048,
        mem_fraction_static=0.7435115625,
        speculative_algorithm="EAGLE",
        speculative_draft_model_path=None,
        speculative_num_draft_tokens=4,
        speculative_adaptive=False,
        speculative_adaptive_config=None,
        speculative_cross_algorithm=False,
        speculative_draft_placement="split",
        disable_cuda_graph=False,
        dcp_size=3,
        _derived_rank_auto_reserve_per_gpu={0: demand, 1: demand, 2: demand},
        _measured_kv_budget_registry_path="/nonexistent/registry.json",
        cuda_graph_config=types.SimpleNamespace(
            decode=types.SimpleNamespace(max_bs=24)
        ),
    )
    sa.uneven_kv_flag_active = lambda: sa.rank_kv_ratio != "coupled"
    sa.uneven_kv_capacity_mode = lambda: sa.rank_kv_ratio == "capacity"
    sa.uneven_kv_speed_mode = lambda: sa.rank_kv_ratio == "speed"
    sa.uneven_kv_derived_mode = lambda: (
        sa.uneven_kv_capacity_mode() or sa.uneven_kv_speed_mode()
    )
    return sa


_ENV = {"SGLANG_MAMBA_SSM_DTYPE": "bfloat16"}


def _ladder(log):
    """Every candidate MLP vector the objective built, as int tuples."""
    out = []
    for ln in log.splitlines():
        marker = "candidate MLP vector "
        if marker in ln:
            vec = ln.split(marker, 1)[1].split(":", 1)[0]
            out.append(tuple(int(x) for x in vec.split(",")))
    return out


def _rank0_shares(log):
    return [v[0] / sum(v) for v in _ladder(log)]


def _plan(lanes=True, **kw):
    sa = _args(**kw)
    captured = []
    with mock.patch.dict(os.environ, _ENV), mock.patch.object(
        uneven_perf,
        "get_hardware_profile",
        return_value=(_profile(lanes=lanes), "test fixture", _GPUS),
    ), mock.patch.object(
        uneven_perf.logger,
        "info",
        lambda *a, **k: captured.append(a[0] if a else ""),
    ), mock.patch.object(uneven_perf.logger, "warning", lambda *a, **k: None):
        uneven_perf.apply_auto_performance(sa)
    return sa, "\n".join(captured)


@unittest.skipUnless(
    _MODEL and os.path.isdir(_MODEL),
    "HTSGLANG_TEST_MODEL_DIR/Qwen3.6-27B-FP8 not present",
)
class TestTheObjectiveConsumesTheLanes(CustomTestCase):
    """End to end through ``apply_auto_performance`` on the fp8 checkpoint the
    #296 audit was written against."""

    def test_the_plan_log_names_the_format_and_the_lane_per_rank(self):
        _sa, log = _plan()
        self.assertIn("checkpoint weight format: fp8", log)
        self.assertIn("566.9 TFLOPS [fp8 native (_scaled_mm)]", log)
        self.assertIn("55.0 TFLOPS [fp8 W8A16 dequant]", log)

    def test_a_lane_less_profile_says_so_in_the_plan_log(self):
        _sa, log = _plan(lanes=False)
        self.assertIn("WARNING:", log)
        self.assertIn("no fp8 GEMM lane measured", log)
        self.assertIn("233.0 TFLOPS [dense bf16 (fallback)]", log)

    def test_the_fp8_ratio_concentrates_the_candidate_ladder(self):
        """The #296 finding, as a decision rather than a number.

        The ladder is score-proportional, so reading the ratio in the
        checkpoint's own format moves every candidate it proposes toward the
        compute-strong rank. Asserted on the LADDER rather than on the accepted
        vector because at this operating point the decode-knee guard rejects
        the whole class either way -- that guard is a separate question, and
        this test is about which candidates the objective builds."""
        _sa, fp8 = _plan(lanes=True)
        _sa2, bf16 = _plan(lanes=False)
        share_fp8 = max(_rank0_shares(fp8))
        share_bf16 = max(_rank0_shares(bf16))
        self.assertGreater(share_fp8, share_bf16)

    def test_the_measured_optimum_is_on_the_fp8_ladder_and_not_the_bf16_one(self):
        """#296 measured 10,1,1 as the prefill optimum on this rig. The bf16
        objective never proposed it -- it offered 10,1,2, spending two units on
        cards the fp8 checkpoint does not run fast on."""
        _sa, fp8 = _plan(lanes=True)
        _sa2, bf16 = _plan(lanes=False)
        self.assertIn((10, 1, 1), _ladder(fp8))
        self.assertNotIn((10, 1, 1), _ladder(bf16))

    def test_dec_stays_a_documented_no_op(self):
        """``dec`` has no compute-ratio objective to correct: it keeps the
        VRAM-auto split by design (M22), so the lanes must not move it."""
        sa, log = _plan(tune="dec")
        self.assertIsNone(sa.rank_mlp_ratio)
        self.assertIn("tune=dec:", log)

    def test_both_rides_the_same_corrected_objective_as_enc(self):
        sa_enc, _l1 = _plan(tune="enc")
        sa_both, _l2 = _plan(tune="both")
        self.assertEqual(sa_enc.rank_mlp_ratio, sa_both.rank_mlp_ratio)


if __name__ == "__main__":
    unittest.main()
