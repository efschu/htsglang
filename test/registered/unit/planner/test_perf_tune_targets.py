"""What ``--rank-perf-tune`` actually decides, and what it refuses (task #265).

Two defects found in the #264 A/B, both in the ``auto-performance`` candidate
ladder, both reproduced here against the numbers of the boots that found them
(TP=3 FP8 on the reference rig, ``--rank-auto-reserve-mib 3000,2700,2700``,
NVML totals 32607/20480/20480, derived reserve demand 4160 MiB per GPU).

**Bug 1 -- ``--rank-perf-tune enc`` was indistinguishable from ``both``.**
The optimizer branched only on ``tune == "dec"``; ``enc`` fell into the
``both`` arm without a line of its own, so a run in which enc had nothing to
do looked exactly like a run in which enc had optimized and found nothing.
Two things fix it: enc states its own objective and its own lever, and the
refusal at the end names the gate that actually bound instead of always
recommending ``--rank-perf-loose-ctx-percent``.

The same ladder also rejected every candidate through a floating-point
accident. Re-partitioning the MLP family CONSERVES total weight bytes, so
``sum_r P_r`` -- the predicted context whenever the sum is the binding term --
is the same number for every candidate in exact arithmetic. The #264 log shows
six candidates "REJECTED by floor" at a printed context identical to the
floor's own 492416: a bare ``>=`` on two differently-ordered summations of the
same quantity.

**Bug 2 -- the ladder had no notion of bootability.** ``6,1,1`` at the runbook
reserve does not boot: it OOMs in the first real prefill (measured, GPU 0 down
to 41.69 MiB free). The ladder printed "REJECTED by floor" for it -- a CONTEXT
verdict, whose documented remedy is to raise ``--rank-perf-loose-ctx-percent``,
which here buys an OOM instead of a slower server. The mechanism is in the
plan's own numbers: the KV pool follows the token vector scaled by the
TIGHTEST rank, so a rank that is not the tightest keeps its unused capacity as
free VRAM, and concentration moves the tight rank onto the card being
concentrated. Measured on the three #264 boots:

    arm                          rank-0 free VRAM   outcome
    2,1,1 @ reserve 3000         2.02 GB            boots
    6,1,1 @ reserve 3000         0.38 GB            OOM in the first prefill
    6,1,1 @ reserve 4500         1.97 GB            boots

and the plan-time residual model reproduces all three verdicts.
"""

import os
import types
import unittest
from unittest import mock

from sglang.srt import uneven_perf
from sglang.srt.distributed.utils import partition_units
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

_CACHE = os.environ.get("HTSGLANG_TEST_MODEL_DIR", "")
_MODEL = os.path.join(_CACHE, "Qwen3.6-27B-FP8") if _CACHE else ""

#: NVML totals of the reference rig in rank order (5090, 3080, 3080).
_TOTALS = [32607, 20480, 20480]
#: ``derived_rank_auto_reserve_mib`` for this boot shape, from the #264 logs.
_DEMAND = 4160

_PROFILE = {
    "version": 2,
    "driver": "595.58.03",
    "gpus": {
        "U0": {
            "name": "NVIDIA GeForce RTX 5090",
            "cuda_index": 0,
            "total_mib": 32607,
            "gemm_tflops": 232.97,
            "membw_gbs": 1664.2,
            "membw_gemv_gbs": 1529.7,
        },
        "U1": {
            "name": "NVIDIA GeForce RTX 3080",
            "cuda_index": 1,
            "total_mib": 20480,
            "gemm_tflops": 62.72,
            "membw_gbs": 717.8,
            "membw_gemv_gbs": 717.8,
        },
        "U2": {
            "name": "NVIDIA GeForce RTX 3080",
            "cuda_index": 2,
            "total_mib": 20480,
            "gemm_tflops": 62.98,
            "membw_gbs": 717.4,
            "membw_gemv_gbs": 717.8,
        },
    },
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


def _args(reserve=(3000, 2700, 2700), tune="enc", loose=0.0, demand=_DEMAND):
    """A duck-typed ServerArgs carrying exactly the fields the optimizer reads.

    Deliberately not a real ``ServerArgs``: constructing one resolves NVML,
    torch and the whole model-specific adjustment chain, none of which this
    decision depends on, and all of which would make the fixture a boot.
    """
    budgets = [t - r for t, r in zip(_TOTALS, reserve)]
    sa = types.SimpleNamespace(
        model_path=_MODEL,
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
        _derived_rank_auto_reserve_per_gpu=(
            {0: demand, 1: demand, 2: demand} if demand else {}
        ),
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


#: The #264 boots ran with the bf16 SSM state (halves the mamba pool, hence
#: the per-rank capacities); the model reads it from the environment.
_ENV = {"SGLANG_MAMBA_SSM_DTYPE": "bfloat16"}


def _plan(**kw):
    """Run the optimizer on the fixture; return ``(server_args, log text)``."""
    sa = _args(**kw)
    captured = []
    with mock.patch.dict(os.environ, _ENV), mock.patch.object(
        uneven_perf,
        "get_hardware_profile",
        return_value=(_PROFILE, "test fixture", _GPUS),
    ), mock.patch.object(
        uneven_perf.logger,
        "info",
        lambda *a, **k: captured.append(a[0] if a else ""),
    ):
        uneven_perf.apply_auto_performance(sa)
    return sa, "\n".join(captured)


def _cost_model(reserve):
    """The same ``PerfCostModel`` the optimizer builds for that reserve."""
    sa = _args(reserve=reserve)
    with mock.patch.dict(os.environ, _ENV):
        return uneven_perf.PerfCostModel(
            uneven_perf.PlanInputs.from_server_args(sa),
            list(sa.rank_tp_ratio),
            list(sa.rank_gpu_memory_mib),
        )


def _candidate_lines(log):
    return [ln.strip() for ln in log.splitlines() if "candidate MLP vector" in ln]


def _line_for(log, vector):
    for ln in _candidate_lines(log):
        if ln.startswith(f"candidate MLP vector {vector}:"):
            return ln
    return None


@unittest.skipUnless(
    _MODEL and os.path.isdir(_MODEL),
    "HTSGLANG_TEST_MODEL_DIR/Qwen3.6-27B-FP8 not present",
)
class TestFixtureReproducesTheBoot(CustomTestCase):
    """The fixture is the #264 boot, not a plausible-looking stand-in.

    Every verdict below is only worth something if the inputs are the ones the
    measured boots ran on, so the reproduction is asserted first.
    """

    def test_base_plan_capacity_matches_the_measured_plan_log(self):
        _sa, log = _plan()
        self.assertIn("[277031, 86160, 129224]", log)
        self.assertIn("predicted max context ~492416", log)
        self.assertIn("materialized MLP units [62, 37, 37]", log)

    def test_candidate_prefill_gains_match_the_measured_plan_log(self):
        _sa, log = _plan()
        for vector, gain in (
            ("16,1,2", "+18.5%"),
            ("10,1,2", "+16.6%"),
            ("8,1,1", "+16.3%"),
            ("6,1,1", "+13.9%"),
            ("5,1,1", "+11.5%"),
            ("4,1,1", "+9.3%"),
        ):
            with self.subTest(vector=vector):
                line = _line_for(log, vector)
                self.assertIsNotNone(line, f"{vector} missing from the ladder")
                self.assertIn(f"predicted prefill gain {gain}", line)


@unittest.skipUnless(
    _MODEL and os.path.isdir(_MODEL),
    "HTSGLANG_TEST_MODEL_DIR/Qwen3.6-27B-FP8 not present",
)
class TestEncIsItsOwnTarget(CustomTestCase):
    """Bug 1: ``enc`` must be distinguishable from ``both`` in the log.

    FALSIFIER: the shipped optimizer had exactly one branch here,
    ``if tune == "both"``, so an enc run was the both run MINUS that one line
    -- enc contributed nothing of its own to the log and could not be told
    apart from the arm it had silently fallen into.

    enc and both do share the objective, and deliberately so ('both' rides
    the same MLP-concentration lever, M22). What they must not share is the
    account they give of themselves: enc's lever is the only one it has, so
    "no candidate accepted" means something different for enc than for both.
    """

    def test_enc_is_not_the_unnamed_default_arm(self):
        _sa_enc, enc = _plan(tune="enc")
        _sa_both, both = _plan(tune="both")
        both_lines = set(both.splitlines())
        enc_only = [ln for ln in enc.splitlines() if ln not in both_lines]
        self.assertTrue(
            enc_only,
            "the enc plan log is a subset of the both plan log -- enc says "
            "nothing about itself, exactly the #264 finding",
        )

    def test_enc_states_its_objective_and_its_lever(self):
        _sa, log = _plan(tune="enc")
        self.assertIn("tune=enc:", log)
        self.assertIn("PREFILL", log)
        self.assertIn("MLP concentration", log)

    def test_enc_refuses_out_loud_when_it_has_no_lever(self):
        """The forbidden variant is a quiet fall-through to the base split."""
        sa, log = _plan(tune="enc")
        self.assertIsNone(sa.rank_mlp_ratio)
        self.assertIn("enc has no effective lever at this operating point", log)

    def test_both_does_not_claim_to_have_no_lever(self):
        """'both' is not enc: it must not borrow enc's refusal wording."""
        _sa, log = _plan(tune="both")
        self.assertNotIn("enc has no effective lever", log)
        self.assertIn("tune=both:", log)

    def test_the_refusal_tallies_the_gates_and_names_the_binding_one(self):
        _sa, log = _plan(tune="enc")
        self.assertIn("candidates evaluated, none accepted", log)
        self.assertIn("is rejected by:", log)

    def test_the_refusal_does_not_recommend_loose_ctx_when_it_cannot_help(self):
        """The old refusal always pointed at --rank-perf-loose-ctx-percent.

        At this operating point the binding gate is fundability, where that
        knob buys an OOM rather than context -- so the recommendation must be
        the reserve, not the floor."""
        _sa, log = _plan(tune="enc")
        refusal = [ln for ln in log.splitlines() if "binds:" in ln]
        self.assertTrue(refusal, "no binding-gate sentence in the refusal")
        self.assertIn("Fundability is what binds", refusal[0])
        self.assertIn("--rank-auto-reserve-mib", refusal[0])


@unittest.skipUnless(
    _MODEL and os.path.isdir(_MODEL),
    "HTSGLANG_TEST_MODEL_DIR/Qwen3.6-27B-FP8 not present",
)
class TestContextFloorTolerance(CustomTestCase):
    """Bug 1, second half: the floor rejected capacity-NEUTRAL candidates.

    FALSIFIER: at ``--rank-perf-loose-ctx-percent 0`` the floor IS the base
    prediction, and MLP re-partitioning conserves total weight bytes, so a
    candidate's predicted context is the same number -- reached by a different
    summation order. The shipped ``>=`` rejected six of them at a printed
    context equal to the floor.
    """

    def test_a_candidate_predicting_the_floor_is_not_rejected_by_the_floor(self):
        _sa, log = _plan(tune="enc", loose=0.0)
        floor = 492416
        for line in _candidate_lines(log):
            if f"predicted ctx ~{floor}" not in line:
                continue
            with self.subTest(line=line[:60]):
                self.assertNotIn(
                    "REJECTED by floor",
                    line,
                    "a candidate whose predicted context EQUALS the floor is "
                    "rejected by the floor (float-noise verdict)",
                )

    def test_the_floor_still_rejects_a_candidate_that_really_loses_context(self):
        """The tolerance must not turn the floor off. Forced by shaving 0.1 %
        off every concentrated candidate's predicted context -- three orders
        of magnitude above the summation noise the tolerance absorbs, and
        still far below anything a reader would call a context loss. Run at
        the reserve where the candidates are fundable, so the floor is the
        gate that gets to speak."""
        real = uneven_perf.PerfCostModel.predict_capacity

        def shaved(self, mlp_vector):
            out = dict(real(self, mlp_vector))
            if list(mlp_vector) != list(self.base_plan):
                out["ctx"] = out["ctx"] * 0.999
            return out

        with mock.patch.object(uneven_perf.PerfCostModel, "predict_capacity", shaved):
            _sa, log = _plan(tune="enc", loose=0.0, reserve=(4500, 2700, 2700))
        rejected = [ln for ln in _candidate_lines(log) if "REJECTED by floor" in ln]
        self.assertTrue(
            rejected,
            "no candidate is rejected by the floor even when every one of "
            "them predicts 0.1 % less context -- the tolerance disabled the "
            "gate instead of absorbing summation noise",
        )


@unittest.skipUnless(
    _MODEL and os.path.isdir(_MODEL),
    "HTSGLANG_TEST_MODEL_DIR/Qwen3.6-27B-FP8 not present",
)
class TestFundability(CustomTestCase):
    """Bug 2: a candidate that cannot boot is not a candidate that costs context.

    FALSIFIER: the shipped ladder had no bootability notion at all, so the
    measured-unbootable ``6,1,1`` was reported with a context verdict and its
    documented remedy pointed at the wrong knob.
    """

    def test_the_measured_unbootable_candidate_is_named_unbootable(self):
        """6,1,1 at reserve 3000: measured OOM in the first real prefill."""
        _sa, log = _plan(tune="enc", reserve=(3000, 2700, 2700))
        line = _line_for(log, "6,1,1")
        self.assertIsNotNone(line)
        self.assertIn("UNBOOTABLE", line)
        self.assertIn("rank 0", line)

    def test_the_unbootable_candidate_is_not_given_a_context_verdict(self):
        _sa, log = _plan(tune="enc", reserve=(3000, 2700, 2700))
        line = _line_for(log, "6,1,1")
        self.assertNotIn("REJECTED by floor", line)

    def test_the_same_candidate_is_fundable_at_the_reserve_that_booted(self):
        """6,1,1 at reserve 4500: measured boot, 1.97 GB free on rank 0."""
        _sa, log = _plan(tune="enc", reserve=(4500, 2700, 2700))
        line = _line_for(log, "6,1,1")
        self.assertIsNotNone(line)
        self.assertNotIn("UNBOOTABLE", line)

    def test_the_three_measured_boots_get_their_measured_verdicts(self):
        """The check against the campaign, one row per boot that was run.

        A gate that rejects everything is not a gate: arm A (2,1,1 at reserve
        3000) and arm B (6,1,1 at reserve 4500) both booted and must clear the
        demand, while 6,1,1 at reserve 3000 -- the boot that OOMed -- must
        not."""
        for reserve, vector, bootable in (
            ((3000, 2700, 2700), [2, 1, 1], True),
            ((3000, 2700, 2700), [6, 1, 1], False),
            ((4500, 2700, 2700), [6, 1, 1], True),
        ):
            with self.subTest(reserve=reserve[0], vector=vector):
                model = _cost_model(reserve)
                vec = partition_units(
                    uneven_perf._PREDICT_TOKEN_UNITS, list(model.base_plan)
                )
                res = model.residual_free_mib(vector, _TOTALS, [1, 1, 1], vec)
                base = model.residual_free_mib(
                    list(model.base_plan), _TOTALS, [1, 1, 1], vec
                )
                fell_through = any(res[r] < _DEMAND <= base[r] for r in range(3))
                self.assertEqual(not fell_through, bootable, f"{res} MiB")

    def test_an_unbootable_candidate_is_never_accepted_by_loosening_the_floor(
        self,
    ):
        """The whole point: loose-ctx trades context for speed, and an
        unbootable candidate has no context to trade."""
        for loose in (0.0, 25.0, 50.0, 90.0):
            with self.subTest(loose=loose):
                sa, log = _plan(tune="enc", loose=loose, reserve=(3000, 2700, 2700))
                chosen = sa.rank_mlp_ratio
                if chosen is None:
                    continue
                line = _line_for(log, ",".join(map(str, chosen)))
                self.assertNotIn("UNBOOTABLE", line or "")

    def test_the_reference_residuals_match_the_measured_boots(self):
        """The residual model, stated as numbers rather than as a verdict.

        Rank 0 keeps 7589 MiB at the base split (reserve 3000 + 4589 MiB of
        capacity the token vector does not use) and exactly the reserve at
        6,1,1, because the concentration makes it the tightest rank."""
        _sa, log = _plan(tune="enc", reserve=(3000, 2700, 2700))
        self.assertIn(
            "residual free VRAM at the VRAM-auto split " "[7589, 2700, 4288] MiB", log
        )
        self.assertIn("residual free [3000, 3765, 5392] MiB", _line_for(log, "6,1,1"))

    def test_no_demand_model_means_no_verdict_not_a_wrong_verdict(self):
        """When the derived reserve demand is unavailable the ladder must
        degrade to its pre-#265 behavior, not to a guess."""
        _sa, log = _plan(tune="enc", demand=None)
        self.assertNotIn("UNBOOTABLE", log)
        self.assertNotIn("fundability reference", log)


@unittest.skipUnless(
    _MODEL and os.path.isdir(_MODEL),
    "HTSGLANG_TEST_MODEL_DIR/Qwen3.6-27B-FP8 not present",
)
class TestDecIsStillADocumentedNoOp(CustomTestCase):
    """The one target that was ALREADY honest stays untouched: dec says it
    keeps the auto split and why, and it does."""

    def test_dec_keeps_the_auto_split_and_says_so(self):
        sa, log = _plan(tune="dec")
        self.assertIsNone(sa.rank_mlp_ratio)
        self.assertIn("tune=dec:", log)
        self.assertIn("documented no-op", log)

    def test_dec_evaluates_no_candidates(self):
        _sa, log = _plan(tune="dec")
        self.assertEqual(_candidate_lines(log), [])


if __name__ == "__main__":
    unittest.main()
