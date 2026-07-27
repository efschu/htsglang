"""CPU unit tests for the free PD topology choice (#107).

Covers the structural flag validation, the pure per-card planning and the
boot-time VRAM feasibility check (itemized reject instead of a late OOM),
the probe-gated prerequisites of two processes on one physical GPU, the
prefill-server normalization onto the existing uneven-PP machinery, and the
inertness of the default path when no topology flag is set.
"""

import dataclasses
import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.topology import (
    TOPOLOGY_COLOCATED_CONGRUENT,
    TOPOLOGY_COLOCATED_PROCESS,
    TOPOLOGY_DISJOINT,
    apply_pd_topology,
    check_process_colocation_prerequisites,
    check_vram_feasibility,
    plan_pd_topology,
    validate_pd_topology_args,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# The rig this targets: one strong card (GPU 0) and two smaller ones.
TOTALS = {0: 32607, 1: 20096, 2: 20096}


def _args(**kw):
    base = dict(
        disaggregation_topology=None,
        disaggregation_prefill_gpus=None,
        disaggregation_prefill_layer_split=None,
        disaggregation_prefill_budget_mib=None,
        disaggregation_mode="null",
        tp_size=3,
        pp_size=1,
        dp_size=1,
        ep_size=1,
        rank_gpu_id=None,
        base_gpu_id=0,
        gpu_id_step=1,
        rank_gpu_memory_mib=None,
        mem_fraction_static=None,
        model_path="",
    )
    base.update(kw)
    return SimpleNamespace(**base)


class TestDefaultPathInert(CustomTestCase):
    def test_no_flags_is_a_no_op(self):
        env = {}
        result = apply_pd_topology(_args(), setenv=env)
        self.assertIsNone(result)
        self.assertEqual(env, {})

    def test_stray_companion_flags_rejected(self):
        for kw in (
            dict(disaggregation_prefill_gpus=[0]),
            dict(disaggregation_prefill_layer_split=[48]),
            dict(disaggregation_prefill_budget_mib=2000),
        ):
            with self.assertRaisesRegex(ValueError, "disaggregation-topology"):
                validate_pd_topology_args(_args(**kw))


class TestStructuralValidation(CustomTestCase):
    def test_topology_requires_prefill_gpus(self):
        with self.assertRaisesRegex(ValueError, "prefill-gpus"):
            validate_pd_topology_args(
                _args(
                    disaggregation_topology=TOPOLOGY_DISJOINT,
                    disaggregation_mode="prefill",
                    disaggregation_prefill_budget_mib=2000,
                )
            )

    def test_duplicate_and_negative_gpus_rejected(self):
        for gpus, pattern in (([0, 0], "duplicates"), ([-1], "negative")):
            with self.assertRaisesRegex(ValueError, pattern):
                validate_pd_topology_args(
                    _args(
                        disaggregation_topology=TOPOLOGY_DISJOINT,
                        disaggregation_mode="prefill",
                        disaggregation_prefill_gpus=gpus,
                        disaggregation_prefill_budget_mib=2000,
                    )
                )

    def test_budget_is_mandatory(self):
        with self.assertRaisesRegex(ValueError, "budget-mib"):
            validate_pd_topology_args(
                _args(
                    disaggregation_topology=TOPOLOGY_DISJOINT,
                    disaggregation_mode="prefill",
                    disaggregation_prefill_gpus=[0],
                )
            )

    def test_layer_split_length_must_match_gpus(self):
        with self.assertRaisesRegex(ValueError, "one layer count per card"):
            validate_pd_topology_args(
                _args(
                    disaggregation_topology=TOPOLOGY_DISJOINT,
                    disaggregation_mode="prefill",
                    disaggregation_prefill_gpus=[0, 1],
                    disaggregation_prefill_layer_split=[48],
                    disaggregation_prefill_budget_mib=2000,
                    tp_size=1,
                )
            )

    def test_disjoint_belongs_on_the_prefill_server(self):
        for mode in ("null", "decode"):
            with self.assertRaises(ValueError):
                validate_pd_topology_args(
                    _args(
                        disaggregation_topology=TOPOLOGY_DISJOINT,
                        disaggregation_mode=mode,
                        disaggregation_prefill_gpus=[0],
                        disaggregation_prefill_budget_mib=2000,
                    )
                )

    def test_congruent_requires_single_server(self):
        with self.assertRaisesRegex(ValueError, "must stay"):
            validate_pd_topology_args(
                _args(
                    disaggregation_topology=TOPOLOGY_COLOCATED_CONGRUENT,
                    disaggregation_mode="decode",
                    disaggregation_prefill_gpus=[0],
                    disaggregation_prefill_budget_mib=2000,
                )
            )

    def test_congruent_rejects_layer_split(self):
        with self.assertRaisesRegex(ValueError, "decode sharding"):
            validate_pd_topology_args(
                _args(
                    disaggregation_topology=TOPOLOGY_COLOCATED_CONGRUENT,
                    disaggregation_prefill_gpus=[0],
                    disaggregation_prefill_layer_split=[48],
                    disaggregation_prefill_budget_mib=2000,
                )
            )

    def test_congruent_prefill_card_must_host_a_decode_rank(self):
        # Decode ranks sit on GPUs 0,1,2 (base formula, tp_size=3); GPU 5
        # hosts none, so there is no shard for the lane to reuse.
        with self.assertRaisesRegex(ValueError, r"\[5\]"):
            validate_pd_topology_args(
                _args(
                    disaggregation_topology=TOPOLOGY_COLOCATED_CONGRUENT,
                    disaggregation_prefill_gpus=[5],
                    disaggregation_prefill_budget_mib=2000,
                )
            )

    def test_process_requires_layer_split(self):
        with self.assertRaisesRegex(ValueError, "layer-split"):
            validate_pd_topology_args(
                _args(
                    disaggregation_topology=TOPOLOGY_COLOCATED_PROCESS,
                    disaggregation_mode="decode",
                    disaggregation_prefill_gpus=[0],
                    disaggregation_prefill_budget_mib=2000,
                )
            )

    def test_process_without_shared_card_is_disjoint(self):
        with self.assertRaisesRegex(ValueError, "disjoint"):
            validate_pd_topology_args(
                _args(
                    disaggregation_topology=TOPOLOGY_COLOCATED_PROCESS,
                    disaggregation_mode="decode",
                    rank_gpu_id=[0, 1],
                    tp_size=2,
                    disaggregation_prefill_gpus=[3],
                    disaggregation_prefill_layer_split=[48],
                    disaggregation_prefill_budget_mib=2000,
                )
            )

    def test_layer_split_with_partial_tp_is_out_of_scope(self):
        # Full-width slices only: layer split x partial TP is the
        # hierarchical-parallelism track, not this flag.
        with self.assertRaisesRegex(ValueError, "hierarchical"):
            validate_pd_topology_args(
                _args(
                    disaggregation_topology=TOPOLOGY_DISJOINT,
                    disaggregation_mode="prefill",
                    tp_size=2,
                    disaggregation_prefill_gpus=[0, 1],
                    disaggregation_prefill_layer_split=[36, 12],
                    disaggregation_prefill_budget_mib=2000,
                )
            )

    def test_dp_rejected(self):
        with self.assertRaisesRegex(ValueError, "dp_size"):
            validate_pd_topology_args(
                _args(
                    disaggregation_topology=TOPOLOGY_DISJOINT,
                    disaggregation_mode="prefill",
                    dp_size=2,
                    disaggregation_prefill_gpus=[0],
                    disaggregation_prefill_budget_mib=2000,
                )
            )


class TestPlanAndFeasibility(CustomTestCase):
    def test_layer_split_must_sum_to_model_layers(self):
        with self.assertRaisesRegex(ValueError, "sums to 40 layers"):
            plan_pd_topology(
                topology=TOPOLOGY_DISJOINT,
                prefill_gpus=[0, 1],
                prefill_layer_split=[30, 10],
                prefill_budget_mib=2000,
                decode_rank_gpus=[],
                decode_rank_budgets_mib=None,
                card_totals_mib=TOTALS,
                model_weight_mib=14000,
                num_hidden_layers=48,
            )

    def test_infeasible_card_named_with_items_sum_and_capacity(self):
        # colocated-process on GPU 0: decode rank claims 26000 MiB, a full
        # 48-layer prefill copy (14000 MiB) plus 2000 MiB budget cannot fit
        # a 32607 MiB card. The reject must name all of it.
        plan = plan_pd_topology(
            topology=TOPOLOGY_COLOCATED_PROCESS,
            prefill_gpus=[0],
            prefill_layer_split=[48],
            prefill_budget_mib=2000,
            decode_rank_gpus=[0, 1, 2],
            decode_rank_budgets_mib=[26000, 17000, 17000],
            card_totals_mib=TOTALS,
            model_weight_mib=14000,
            num_hidden_layers=48,
        )
        with self.assertRaises(ValueError) as ctx:
            check_vram_feasibility(plan)
        msg = str(ctx.exception)
        for needle in (
            "GPU 0",
            "total 32607 MiB",
            "decode claim 26000 MiB (ranks 0)",
            "prefill weight slice 14000 MiB (48 layers)",
            "prefill budget 2000 MiB",
            "= 42000 MiB > 32607 MiB",
        ):
            self.assertIn(needle, msg)

    def test_congruent_sharing_costs_no_weight_bytes(self):
        plan = plan_pd_topology(
            topology=TOPOLOGY_COLOCATED_CONGRUENT,
            prefill_gpus=[0],
            prefill_layer_split=None,
            prefill_budget_mib=2000,
            decode_rank_gpus=[0, 1, 2],
            decode_rank_budgets_mib=[26000, 17000, 17000],
            card_totals_mib=TOTALS,
            model_weight_mib=14000,
            num_hidden_layers=48,
        )
        card = plan.card(0)
        self.assertTrue(card.weights_shared)
        self.assertEqual(card.prefill_weight_mib, 0)
        self.assertEqual(card.sum_mib(), 28000)
        self.assertEqual(check_vram_feasibility(plan), [])

    def test_uneven_layer_slices_are_proportional(self):
        plan = plan_pd_topology(
            topology=TOPOLOGY_DISJOINT,
            prefill_gpus=[0, 1],
            prefill_layer_split=[36, 12],
            prefill_budget_mib=2000,
            decode_rank_gpus=[],
            decode_rank_budgets_mib=None,
            card_totals_mib=TOTALS,
            model_weight_mib=14000,
            num_hidden_layers=48,
        )
        self.assertEqual(plan.card(0).prefill_weight_mib, 10500)
        self.assertEqual(plan.card(1).prefill_weight_mib, 3500)
        self.assertEqual(check_vram_feasibility(plan), [])

    def test_unknown_inputs_warn_instead_of_silently_passing(self):
        plan = plan_pd_topology(
            topology=TOPOLOGY_COLOCATED_PROCESS,
            prefill_gpus=[0],
            prefill_layer_split=[48],
            prefill_budget_mib=2000,
            decode_rank_gpus=[0],
            decode_rank_budgets_mib=None,
            card_totals_mib=TOTALS,
            model_weight_mib=None,
            num_hidden_layers=48,
        )
        warnings = check_vram_feasibility(plan)
        self.assertEqual(len(warnings), 1)
        self.assertIn("GPU 0", warnings[0])
        self.assertIn("not computable", warnings[0])


class TestProcessColocationGate(CustomTestCase):
    def _probe_env(self, nccl_raw, mps_present):
        from sglang.srt.rigmon.capabilities import ProbeEnv

        return ProbeEnv(
            exists=lambda p: mps_present,
            listdir=lambda p: [],
            read_text=lambda p: "",
            run=lambda cmd: (127, "not found"),
            import_module=lambda n: (_ for _ in ()).throw(ImportError("absent")),
            env={},
            nccl_version=lambda environ: (nccl_raw, "libnccl.so.2", None),
        )

    def test_low_nccl_runtime_rejected_with_measured_version(self):
        with self.assertRaises(ValueError) as ctx:
            check_process_colocation_prerequisites(self._probe_env(22809, True))
        self.assertIn("2.28.9", str(ctx.exception))
        self.assertIn("2.30", str(ctx.exception))

    def test_missing_mps_daemon_rejected_with_remedy(self):
        with self.assertRaises(ValueError) as ctx:
            check_process_colocation_prerequisites(self._probe_env(23007, False))
        self.assertIn("nvidia-cuda-mps-control", str(ctx.exception))

    def test_satisfied_host_passes(self):
        check_process_colocation_prerequisites(self._probe_env(23007, True))


class TestApplyNormalization(CustomTestCase):
    def test_prefill_disjoint_maps_onto_uneven_pp_and_cvd(self):
        args = _args(
            disaggregation_topology=TOPOLOGY_DISJOINT,
            disaggregation_mode="prefill",
            tp_size=1,
            disaggregation_prefill_gpus=[2, 0],
            disaggregation_prefill_layer_split=[36, 12],
            disaggregation_prefill_budget_mib=2000,
        )
        env = {}
        plan = apply_pd_topology(
            args,
            card_totals_mib=TOTALS,
            model_weight_mib=14000,
            num_hidden_layers=48,
            setenv=env,
        )
        self.assertIsNotNone(plan)
        self.assertEqual(args.pp_size, 2)
        self.assertEqual(env["SGLANG_PP_LAYER_PARTITION"], "36,12")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2,0")
        self.assertEqual(env["SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS"], "1")

    def test_existing_cvd_restriction_is_respected(self):
        args = _args(
            disaggregation_topology=TOPOLOGY_DISJOINT,
            disaggregation_mode="prefill",
            tp_size=1,
            disaggregation_prefill_gpus=[0],
            disaggregation_prefill_budget_mib=2000,
        )
        env = {"CUDA_VISIBLE_DEVICES": "1"}
        apply_pd_topology(
            args,
            card_totals_mib=TOTALS,
            model_weight_mib=14000,
            num_hidden_layers=48,
            setenv=env,
        )
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")

    def test_congruent_names_the_stage_boundary(self):
        args = _args(
            disaggregation_topology=TOPOLOGY_COLOCATED_CONGRUENT,
            disaggregation_prefill_gpus=[0],
            disaggregation_prefill_budget_mib=2000,
            rank_gpu_id=[0, 1, 2],
            rank_gpu_memory_mib=[26000, 17000, 17000],
        )
        with self.assertRaisesRegex(NotImplementedError, "colocated-congruent"):
            apply_pd_topology(
                args,
                card_totals_mib=TOTALS,
                model_weight_mib=14000,
                num_hidden_layers=48,
                setenv={},
            )

    def test_infeasible_topology_rejected_at_apply(self):
        args = _args(
            disaggregation_topology=TOPOLOGY_COLOCATED_PROCESS,
            disaggregation_mode="decode",
            rank_gpu_id=[0, 1, 2],
            rank_gpu_memory_mib=[26000, 17000, 17000],
            disaggregation_prefill_gpus=[0],
            disaggregation_prefill_layer_split=[48],
            disaggregation_prefill_budget_mib=2000,
        )
        with self.assertRaisesRegex(ValueError, "infeasible"):
            apply_pd_topology(
                args,
                probe_env=self._good_probe_env(),
                card_totals_mib=TOTALS,
                model_weight_mib=14000,
                num_hidden_layers=48,
                setenv={},
            )

    def _good_probe_env(self):
        from sglang.srt.rigmon.capabilities import ProbeEnv

        return ProbeEnv(
            exists=lambda p: True,
            listdir=lambda p: [],
            read_text=lambda p: "",
            run=lambda cmd: (127, "not found"),
            import_module=lambda n: (_ for _ in ()).throw(ImportError("absent")),
            env={},
            nccl_version=lambda environ: (23007, "libnccl.so.2", None),
        )


class TestServerArgsSurface(CustomTestCase):
    def test_flags_exist_in_the_disaggregation_block_and_default_off(self):
        from sglang.srt.server_args import ServerArgs

        fields = {f.name: f for f in dataclasses.fields(ServerArgs)}
        for name in (
            "disaggregation_topology",
            "disaggregation_prefill_gpus",
            "disaggregation_prefill_layer_split",
            "disaggregation_prefill_budget_mib",
        ):
            self.assertIn(name, fields)
            self.assertIsNone(fields[name].default)


if __name__ == "__main__":
    unittest.main()
