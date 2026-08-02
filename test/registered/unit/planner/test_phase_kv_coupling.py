"""The phase solve and the boot must agree on ONE KV token vector (#435).

``predict_capacity`` returns two coupled things for a candidate MLP vector:
the per-rank KV token capacity ``P_r``, and the token vector
``partition_units(64, P)`` that MATCHES it. Everything the optimizer prints
and gates on is computed with that matched vector -- ``ctx = min(sum P,
64 * min P)`` is exactly the pool capacity of it, and the context floor is a
comparison between two such numbers.

The boot resolved a different vector. ``--rank-kv-ratio coupled`` (the
default) falls through ``resolve_cp_token_ratios`` to the budget estimate,
which splits tokens proportionally to VRAM budget minus a flat weight share
-- i.e. proportionally to the BASE plan. That is the same vector as the match
only while the MLP vector is the base plan. The whole point of the
phase-prefill arm is that it is not.

#433 measured the consequence on a real boot (`/root/addendum_433.md`,
`docs/dev/NOTE_433_int8_prefill_vector.md`): solved MLP vector ``8,1,1``,
matched token vector ``[12,26,26]``, booted token vector ``[31,17,16]`` --
rank 0 asked to own 48 % of the tokens while holding 13 % of the capacity.
``max_total_num_tokens`` came out at 125 504 against the 358 693 the
admissibility gate had just accepted, and against 431 360 for the decode arm
of the same recipe.

Everything below runs on a SYNTHETIC FOREIGN rig (three cards that are not
this repo's reference rig) and a SYNTHETIC checkpoint written to a temp dir,
so nothing here depends on a local model cache or on the hardware the fork
was developed on. The expected vector is never a literal: it is recomputed
from the plan's own cost model, which is the property under test.

FALSIFIER: ``TestTheSolveAndTheBootAgree`` fails on the pre-#435 tree --
``resolve_cp_token_ratios`` returns the budget split there.
``test_the_two_vectors_genuinely_differ_on_this_profile`` is the anti-vacuity
guard for it: it asserts the mismatch this fixture reproduces is real and
costly, so the equality above cannot pass by the two vectors happening to
coincide.
"""

import json
import math
import os
import tempfile
import types
import unittest
from unittest import mock

from sglang.srt import uneven_perf
from sglang.srt.distributed.utils import (
    partition_units,
    resolve_cp_token_ratios,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

#: A dense checkpoint that exists nowhere: plain full attention, no MoE, no
#: quantization config, no hybrid layer types. Only the geometry the
#: parse-time cost model reads.
_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "num_hidden_layers": 48,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
    "max_position_embeddings": 262144,
    "torch_dtype": "bfloat16",
}
#: On-disk checkpoint size the sizing math reads (sparse file, no real bytes).
_CHECKPOINT_MIB = 26000

#: A FOREIGN rig: one fast card that is not the largest by much, and two
#: slower ones. The lane ratio (700 : 90 : 88) is what makes the phase arm
#: concentrate; the VRAM ratio is what makes the concentrated rank the
#: capacity-poor one, which is the shape #433 hit.
_CARDS = (
    # uuid, cuda_index, name, total_mib, tflops, membw_gbs, gemv_gbs
    ("SYNTH-UUID-0", 0, "SYNTH Accel X 32G", 32760, 700.0, 1800.0, 1600.0),
    ("SYNTH-UUID-1", 1, "SYNTH Accel S 24G", 24564, 90.0, 640.0, 620.0),
    ("SYNTH-UUID-2", 2, "SYNTH Accel S 24G", 24564, 88.0, 630.0, 610.0),
)
_RESERVE = (4600, 2600, 2600)
#: ``derived_rank_auto_reserve_mib`` for this synthetic boot shape.
_DEMAND = 4200
_BUDGETS = [total - res for (*_, total, _, _, _), res in zip(_CARDS, _RESERVE)]

_TOTALS = [card[3] for card in _CARDS]
_RANKS_ON_GPU = [1, 1, 1]


def _write_checkpoint(root):
    """A checkpoint directory the cost model can size, with no real weights."""
    path = os.path.join(root, "SynthDense-27B")
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w") as handle:
        json.dump(_CONFIG, handle)
    with open(os.path.join(path, "model-00001-of-00001.safetensors"), "wb") as f:
        f.truncate(_CHECKPOINT_MIB * 2**20)
    return path


def _profile():
    gpus = {}
    for uuid, idx, name, total, tflops, bw, gemv in _CARDS:
        gpus[uuid] = {
            "name": name,
            "cuda_index": idx,
            "total_mib": total,
            "gemm_tflops": tflops,
            "membw_gbs": bw,
            "membw_gemv_gbs": gemv,
            "gemm_lanes": {"bf16": tflops},
            "gemm_lane_notes": {},
        }
    profile = {
        "version": 3,
        "driver": "999.00.00",
        "gpus": gpus,
        "links": {
            "SYNTH-UUID-0|SYNTH-UUID-1": {"p2p_gbs": 6.0},
            "SYNTH-UUID-0|SYNTH-UUID-2": {"p2p_gbs": 6.0},
            "SYNTH-UUID-1|SYNTH-UUID-2": {"p2p_gbs": 6.0},
            "__group__": {"ar_10kb_us": 30.0, "ar_1mb_us": 350.0},
        },
    }
    inventory = [
        {
            "uuid": uuid,
            "cuda_index": g["cuda_index"],
            "name": g["name"],
            "total_mib": g["total_mib"],
        }
        for uuid, g in gpus.items()
    ]
    return profile, inventory


def _args(model_path, tune="phase-prefill", kv_ratio="coupled"):
    """Duck-typed ServerArgs with exactly the fields the optimizer reads."""
    sa = types.SimpleNamespace(
        model_path=model_path,
        tp_size=3,
        rank_gpu_id=[0, 1, 2],
        rank_gpu_memory_mib=list(_BUDGETS),
        rank_tp_ratio=list(_BUDGETS),
        rank_mlp_ratio=None,
        rank_vocab_ratio=None,
        rank_moe_ratio=None,
        rank_kv_ratio=kv_ratio,
        rank_kv_capacity_seed=None,
        rank_auto_reserve_mib=",".join(map(str, _RESERVE)),
        rank_perf_tune=tune,
        rank_perf_loose_ctx_percent=0.0,
        kv_cache_dtype="fp8_e4m3",
        context_length=131072,
        page_size=1,
        quantization=None,
        max_running_requests=16,
        chunked_prefill_size=2048,
        mem_fraction_static=0.74,
        speculative_algorithm=None,
        speculative_draft_model_path=None,
        speculative_num_draft_tokens=None,
        speculative_adaptive=False,
        speculative_adaptive_config=None,
        speculative_cross_algorithm=False,
        speculative_draft_placement="split",
        disable_cuda_graph=False,
        dcp_size=3,
        _derived_rank_auto_reserve_per_gpu={0: _DEMAND, 1: _DEMAND, 2: _DEMAND},
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


def _global_pool(capacity, vector):
    """The boot's own pool arithmetic: the token vector is scaled by the
    TIGHTEST rank, so the global pool is ``min_r(P_r // v_r) * sum(v)``
    (model_runner_kv_cache_mixin's min-reduced unit)."""
    unit = min(int(p) // int(v) for p, v in zip(capacity, vector))
    return unit * sum(int(v) for v in vector)


class PhaseKvCouplingTestCase(CustomTestCase):
    """One synthetic checkpoint per class; the plan itself is cheap."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._tmp = tempfile.TemporaryDirectory()
        cls.model_path = _write_checkpoint(cls._tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()
        super().tearDownClass()

    def plan(self, **kwargs):
        """Run the optimizer on the synthetic profile; return (args, log)."""
        sa = _args(self.model_path, **kwargs)
        profile, inventory = _profile()
        captured = []
        with mock.patch.object(
            uneven_perf,
            "get_hardware_profile",
            return_value=(profile, "synthetic fixture", inventory),
        ), mock.patch.object(
            uneven_perf.logger,
            "info",
            lambda *a, **k: captured.append(a[0] if a else ""),
        ):
            uneven_perf.apply_auto_performance(sa)
        return sa, "\n".join(captured)

    def cost_model(self, sa):
        """The same cost model the planner built, rebuilt from the args."""
        return uneven_perf.PerfCostModel(
            uneven_perf.PlanInputs.from_server_args(sa),
            list(_BUDGETS),
            list(_BUDGETS),
        )

    def derived_match(self, sa, mlp_vector):
        """The token vector the solve itself derived for a candidate."""
        return self.cost_model(sa).predict_capacity(list(mlp_vector))["token_vector"]

    def budget_split(self, sa):
        """What ``resolve_cp_token_ratios`` returns without any seed -- the
        vector the pre-#435 boot used."""
        clone = _args(self.model_path)
        clone.rank_tp_ratio = list(sa.rank_tp_ratio)
        return resolve_cp_token_ratios(clone)


class TestTheSolveAndTheBootAgree(PhaseKvCouplingTestCase):
    """The regression itself."""

    def test_the_boot_vector_is_the_solves_own_matched_vector(self):
        sa, log = self.plan(tune="phase-prefill")
        self.assertIsNotNone(
            sa.rank_mlp_ratio, "the synthetic profile did not solve a vector"
        )
        self.assertIn("CHOSEN MLP vector", log)
        match = self.derived_match(sa, sa.rank_mlp_ratio)
        self.assertEqual(
            resolve_cp_token_ratios(sa),
            match,
            "the boot resolves a KV token vector the solve did not derive",
        )

    def test_the_seed_carries_it_and_the_mode_string_survives(self):
        """The vector goes into the dedicated seed field, never into
        ``rank_kv_ratio``: that string gates the dcp_size auto-set and the
        weighted owner rule, both resolved after the planner runs, and
        matching a token vector must not switch a parallelism mode on."""
        sa, _log = self.plan(tune="phase-prefill")
        self.assertEqual(sa.rank_kv_ratio, "coupled")
        self.assertFalse(sa.uneven_kv_flag_active())
        self.assertEqual(
            sa.rank_kv_capacity_seed, self.derived_match(sa, sa.rank_mlp_ratio)
        )

    def test_the_log_names_the_match_it_installed(self):
        sa, log = self.plan(tune="phase-prefill")
        vector = ",".join(map(str, sa.rank_kv_capacity_seed))
        self.assertIn("coupled KV ratio MATCHED to the solve", log)
        self.assertIn(vector, log)

    def test_the_two_vectors_genuinely_differ_on_this_profile(self):
        """Anti-vacuity guard for the equality above, and the structural
        reproduction of #433: the budget split over-assigns tokens to the
        concentrated rank, and the min-reduced pool pays for it."""
        sa, _log = self.plan(tune="phase-prefill")
        capacity = self.cost_model(sa).predict_capacity(list(sa.rank_mlp_ratio))["p"]
        match = self.derived_match(sa, sa.rank_mlp_ratio)
        budget = self.budget_split(sa)
        self.assertNotEqual(
            match, budget, "the fixture does not reproduce the mismatch"
        )
        # Rank 0 is the concentrated one: the budget split hands it a token
        # share far above its capacity share, exactly the #433 shape.
        token_share = budget[0] / sum(budget)
        capacity_share = capacity[0] / sum(capacity)
        self.assertGreater(token_share, capacity_share * 1.5)
        # ... and the pool the boot would size is materially smaller.
        self.assertGreater(
            _global_pool(capacity, match),
            _global_pool(capacity, budget) * 1.2,
        )

    def test_the_installed_vector_is_gcd_reduced_and_well_formed(self):
        sa, _log = self.plan(tune="phase-prefill")
        vector = sa.rank_kv_capacity_seed
        self.assertEqual(len(vector), sa.tp_size)
        self.assertTrue(all(isinstance(v, int) and v > 0 for v in vector))
        self.assertEqual(math.gcd(*vector), 1)


class TestPrecedence(PhaseKvCouplingTestCase):
    """An explicit user opinion always wins (#88 family-flag convention)."""

    def test_an_explicit_kv_vector_wins(self):
        sa, _log = self.plan(tune="phase-prefill", kv_ratio=[5, 3, 3])
        self.assertIsNone(sa.rank_kv_capacity_seed)
        self.assertEqual(resolve_cp_token_ratios(sa), [5, 3, 3])

    def test_an_explicit_derived_mode_wins(self):
        """``capacity``/``speed`` are opinions too: they ask for the MEASURED
        vector installed after profiling, so the planner must not park a
        pre-boot prediction under them on the non-solo path."""
        for mode in ("capacity", "speed"):
            with self.subTest(mode=mode):
                sa, _log = self.plan(tune="phase-prefill", kv_ratio=mode)
                self.assertEqual(sa.rank_kv_ratio, mode)
                self.assertIsNone(sa.rank_kv_capacity_seed)

    def test_the_env_vector_wins_over_the_seed(self):
        sa, _log = self.plan(tune="phase-prefill")
        self.assertIsNotNone(sa.rank_kv_capacity_seed)
        with mock.patch.dict(os.environ, {"SGLANG_UNEVEN_TOKEN_VECTOR": "5,3,3"}):
            self.assertEqual(resolve_cp_token_ratios(sa), [5, 3, 3])


class TestTheOtherPathsAreUntouched(PhaseKvCouplingTestCase):
    """Behavior-neutrality pins. A fix that changes the default path is a
    different bug, not a fix."""

    def test_non_phase_targets_keep_the_budget_split(self):
        for tune in ("enc", "both", "dec", "maxkv"):
            with self.subTest(tune=tune):
                sa, _log = self.plan(tune=tune)
                self.assertIsNone(sa.rank_kv_capacity_seed)
                self.assertEqual(resolve_cp_token_ratios(sa), self.budget_split(sa))

    def test_phase_decode_is_unchanged(self):
        """The decode arm solves the plain VRAM-auto split and installs no MLP
        vector, so there is no divergence to correct and nothing to seed."""
        sa, log = self.plan(tune="phase-decode")
        self.assertIsNone(sa.rank_mlp_ratio)
        self.assertIsNone(sa.rank_kv_capacity_seed)
        self.assertIn("phase-optimal DECODE arm", log)
        self.assertNotIn("coupled KV ratio MATCHED", log)
        self.assertEqual(resolve_cp_token_ratios(sa), self.budget_split(sa))

    def test_a_pinned_mlp_vector_skips_the_whole_path(self):
        """The pin path returns before the solve, so there is no derived match
        to install -- an operator who pins the weight vector still owns the
        KV vector."""
        sa = _args(self.model_path, tune="phase-prefill")
        sa.rank_mlp_ratio = [8, 1, 1]
        profile, inventory = _profile()
        with mock.patch.object(
            uneven_perf,
            "get_hardware_profile",
            return_value=(profile, "synthetic fixture", inventory),
        ), mock.patch.object(uneven_perf.logger, "info", lambda *a, **k: None):
            uneven_perf.apply_auto_performance(sa)
        self.assertIsNone(sa.rank_kv_capacity_seed)


class TestTheFundabilityGateIsNotRebased(PhaseKvCouplingTestCase):
    """The gate that rejects UNBOOTABLE candidates keeps pricing them against
    the BASE-PLAN token vector, deliberately, and #435 does not touch it.

    Its residual term is ``(P_r - unit * v_r) * kv_cell`` -- capacity the
    token vector does not use. A MATCHED vector has none by construction, so
    re-basing the gate on the vector the phase arm now boots collapses every
    residual to the bare reserve and the gate stops discriminating. That is
    not hypothetical: ``--rank-kv-ratio capacity`` already prices
    per-candidate, and on the #264/#354 fixture it accepts ``16,1,1`` at
    reserve ``3000,2700,2700`` -- the configuration #264 booted into an OOM.
    Widening that hole to the phase arm's default path is a loosening of a
    measured OOM guard, and it is not what #435 was asked to do.

    What #435 does change is the direction of the gate's error: post-fix the
    non-binding ranks no longer keep the slack this term credits them with,
    so the reported residual is optimistic for them. Named in the code and in
    NOTE_433's arm spec rather than silently taken.
    """

    def test_the_reference_line_still_prices_the_base_plan_vector(self):
        sa, log = self.plan(tune="phase-prefill")
        model = self.cost_model(sa)
        base = list(_BUDGETS)
        on_base_plan = model.residual_free_mib(
            base, _TOTALS, _RANKS_ON_GPU, partition_units(64, base)
        )
        line = next(ln for ln in log.splitlines() if "fundability reference" in ln)
        self.assertIn(str([int(x) for x in on_base_plan]), line)

    def test_the_matched_pricing_would_be_a_different_number(self):
        """Anti-vacuity guard: the two references really do differ here, so
        the assertion above pins a choice rather than a coincidence."""
        sa, _log = self.plan(tune="phase-prefill")
        model = self.cost_model(sa)
        base = list(_BUDGETS)
        matched = model.residual_free_mib(
            base,
            _TOTALS,
            _RANKS_ON_GPU,
            model.predict_capacity(base)["token_vector"],
        )
        on_base_plan = model.residual_free_mib(
            base, _TOTALS, _RANKS_ON_GPU, partition_units(64, base)
        )
        self.assertNotEqual([int(x) for x in matched], [int(x) for x in on_base_plan])


class TestTheOwnershipPredicate(CustomTestCase):
    """``_phase_solve_owns_kv_ratio`` is read at two places that must agree
    (the fundability reference and the install). Pin its truth table."""

    def _model(self, solo=False):
        return types.SimpleNamespace(solo_active=solo, tp_size=3)

    def _sa(self, kv_ratio="coupled"):
        return types.SimpleNamespace(rank_kv_ratio=kv_ratio)

    def test_true_only_for_the_phase_arms_on_the_default_ratio(self):
        cases = (
            ("phase-prefill", "coupled", False, True),
            # True for the family, not just the prefill arm: phase-decode
            # never reaches the install (it clears the chosen vector), and
            # the two readers must not be able to drift apart if it ever
            # does.
            ("phase-decode", "coupled", False, True),
            ("enc", "coupled", False, False),
            ("both", "coupled", False, False),
            ("dec", "coupled", False, False),
            ("maxkv", "coupled", False, False),
            ("phase-prefill", "capacity", False, False),
            ("phase-prefill", "speed", False, False),
            ("phase-prefill", [3, 2, 2], False, False),
            # Draft-solo keeps its own, older seeding rule.
            ("phase-prefill", "coupled", True, False),
        )
        for tune, kv_ratio, solo, expected in cases:
            with self.subTest(tune=tune, kv_ratio=kv_ratio, solo=solo):
                self.assertEqual(
                    uneven_perf._phase_solve_owns_kv_ratio(
                        self._sa(kv_ratio), self._model(solo), tune
                    ),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
