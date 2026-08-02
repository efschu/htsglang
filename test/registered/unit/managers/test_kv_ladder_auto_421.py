"""#421 F1 falsifier: ``--kv-pressure-ladder auto`` must actually build.

The flag's help text advertises auto as "the step table is computed once from
the rig/model profile by the #272 planner", and argument-time validation
accepts the value -- but the sole production construction site injected no
table source, so ``build_ladder_from_server_args`` hit its own deliberate
refusal and every boot with ``auto`` died inside the scheduler constructor.

These tests drive the REAL production entry point
(``build_kv_pressure_runtime``) with a stub scheduler and a stubbed NVML, so
the can-fail proof is exact: revert the ``table_fn=`` injection at
``kv_pressure_runtime.py`` and ``test_auto_spec_builds_a_runtime`` raises the
same ValueError an operator got.

Hermetic: no GPU, no torch.distributed, no NVML on the box. What is NOT
covered here, and is the reason the fix is reported BOOT-PENDING: that a rung
flip fires under real KV pressure on a real model. That lives in
``scripts/dev/428_boot_checks/``.
"""

import unittest
from unittest import mock

from sglang.srt.managers.kv_ladder_auto import (
    auto_ladder_table_fn,
    build_auto_ladder_profile,
    wired_relief_features,
)
from sglang.srt.managers.kv_pressure_runtime import build_kv_pressure_runtime
from sglang.srt.model_executor.kv_pressure_ladder import STEP_BASE
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class _Device:
    """Stand-in for ``registry.nvml.DeviceInfo``."""

    def __init__(self, index, uuid, name, total_mib):
        self.index = index
        self.uuid = uuid
        self.name = name
        self.total_bytes = total_mib * (1 << 20)
        self.pci_bus_id = f"0000:{index:02d}:00.0"

    @property
    def total_mib(self):
        return self.total_bytes // (1 << 20)


_HETERO = (
    _Device(0, "GPU-aaa", "NVIDIA GeForce RTX 5090", 32768),
    _Device(1, "GPU-bbb", "NVIDIA GeForce RTX 3080", 20480),
    _Device(2, "GPU-ccc", "NVIDIA GeForce RTX 3080", 20480),
)
_HOMOGENEOUS = (
    _Device(0, "GPU-xxx", "NVIDIA GeForce RTX 3080", 20480),
    _Device(1, "GPU-yyy", "NVIDIA GeForce RTX 3080", 20480),
)


class _ServerArgs:
    """Only the fields the bridge reads. Deliberately not a real ServerArgs:
    the point is that the bridge touches nothing rank-local."""

    def __init__(self, **kw):
        self.kv_pressure_ladder = "auto"
        self.kv_pressure_pre_stage = False
        self.kv_pressure_external_hysteresis_rounds = 512
        self.kv_pressure_consensus_interval = 8
        self.kv_pressure_ascend_threshold = 0.85
        self.kv_pressure_descend_threshold = 0.60
        self.tp_size = 2
        self.rank_gpu_id = None
        self.rank_gpu_memory_mib = None
        self.rank_tp_ratio = None
        self.mem_fraction_static = None
        self.base_gpu_id = 0
        self.gpu_id_step = 1
        self.max_running_requests_ceiling = None
        self.enable_kv_session_offload = False
        self.kv_reshard_vectors = None
        self.__dict__.update(kw)

    def gpu_id_for_rank(self, pp_rank, tp_rank, pp_size_per_node, tp_size_per_node):
        if self.rank_gpu_id is not None:
            return self.rank_gpu_id[pp_rank * self.tp_size + tp_rank]
        return self.base_gpu_id + tp_rank * self.gpu_id_step


class _Scheduler:
    def __init__(self, server_args):
        self.server_args = server_args
        self.kv_session_offload = None
        self.admission_limiter = None
        self.tp_cpu_group = None
        self.running_batch = None
        self.kv_reshard_runtime = None


def _patch_nvml(devices, uuids=None):
    """Patch the two identity sources the bridge is allowed to read."""
    by_uuid = {d.uuid: d for d in devices}
    return (
        mock.patch("sglang.srt.registry.nvml.list_devices", lambda: list(devices)),
        mock.patch("sglang.srt.registry.nvml.device_by_uuid", lambda u: by_uuid[u]),
        mock.patch(
            "sglang.srt.registry.rank_cards.rank_card_uuids",
            lambda world_size=None: uuids,
        ),
    )


class _NvmlContext:
    def __init__(self, devices, uuids=None):
        self._patches = _patch_nvml(devices, uuids)

    def __enter__(self):
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()
        return False


class TestAutoLadderReachesTheRuntime(CustomTestCase):
    def test_auto_spec_builds_a_runtime(self):
        """THE falsifier. Without the ``table_fn=`` injection this raises
        ValueError('--kv-pressure-ladder auto needs the planner's step-table
        source; none was supplied')."""
        sa = _ServerArgs(tp_size=2)
        with _NvmlContext(_HOMOGENEOUS):
            runtime = build_kv_pressure_runtime(_Scheduler(sa))
        self.assertIsNotNone(runtime)
        self.assertEqual(runtime.ladder.table.steps[0].step_type, STEP_BASE)

    def test_unset_flag_still_builds_nothing(self):
        """The default path must stay byte-identical: no ladder, and the
        thunk is never called, so NVML is never touched."""
        sa = _ServerArgs(kv_pressure_ladder=None)
        called = []

        def _boom():
            called.append(1)
            raise AssertionError("the table source must not be evaluated")

        with mock.patch(
            "sglang.srt.managers.kv_ladder_auto.auto_ladder_table_fn",
            lambda _sa: _boom,
        ):
            self.assertIsNone(build_kv_pressure_runtime(_Scheduler(sa)))
        self.assertEqual(called, [])

    def test_explicit_spec_does_not_evaluate_the_auto_source(self):
        sa = _ServerArgs(kv_pressure_ladder="relief:admission_cap")
        sa.max_running_requests_ceiling = 32

        class _Limiter:
            auto = True
            current = 32
            start = 32

            def throttle(self, *a, **k):
                return None

            def release(self, *a, **k):
                return None

        sched = _Scheduler(sa)
        sched.admission_limiter = _Limiter()
        # No NVML patch at all: if the auto source were evaluated on an
        # explicit spec, this would fail with an NVML error.
        runtime = build_kv_pressure_runtime(sched)
        self.assertIsNotNone(runtime)


class TestAutoProfileIsRankUniformAndHonest(CustomTestCase):
    def test_uuid_vector_wins_over_cuda_ordinals(self):
        """The launcher's UUID vector decides which card a rank sits on --
        the CUDA ordinal is process-local under CUDA_VISIBLE_DEVICES."""
        sa = _ServerArgs(
            tp_size=3,
            rank_gpu_id=[0, 1, 2],
            rank_gpu_memory_mib=15000,
        )
        with _NvmlContext(_HETERO, uuids=("GPU-ccc", "GPU-aaa", "GPU-bbb")):
            profile = build_auto_ladder_profile(sa)
        self.assertEqual([c.index for c in profile.cards], [0, 1, 2])
        # rank 0 -> GPU-ccc = nvml index 2, rank 1 -> aaa = 0, rank 2 -> bbb = 1
        self.assertEqual(profile.geometries[0].gpus, (2, 0, 1))
        self.assertEqual({c.budget_mib for c in profile.cards}, {15000})
        self.assertEqual({c.total_mib for c in profile.cards}, {32768, 20480})

    def test_per_rank_budget_list_is_indexed_per_rank(self):
        sa = _ServerArgs(
            tp_size=2,
            rank_gpu_id=[0, 1],
            rank_gpu_memory_mib=[26000, 17000],
            rank_tp_ratio=[3, 2],
        )
        with _NvmlContext(_HETERO, uuids=("GPU-aaa", "GPU-bbb")):
            profile = build_auto_ladder_profile(sa)
        budgets = {c.index: c.budget_mib for c in profile.cards}
        self.assertEqual(budgets, {0: 26000, 1: 17000})
        self.assertEqual(profile.geometries[0].ratio, (3, 2))

    def test_mixed_node_without_a_card_vector_refuses_by_name(self):
        """No guessing an enumeration order: refuse and name the remedies."""
        sa = _ServerArgs(tp_size=2)
        with _NvmlContext(_HETERO, uuids=None):
            with self.assertRaises(ValueError) as ctx:
                build_auto_ladder_profile(sa)
        msg = str(ctx.exception)
        self.assertIn("--rank-gpu-id", msg)
        self.assertIn("SGLANG_RANK_CARD_PROBE_CUDA", msg)

    def test_homogeneous_node_without_a_card_vector_is_allowed(self):
        """Indistinguishable cards make the mapping immaterial."""
        sa = _ServerArgs(tp_size=2)
        with _NvmlContext(_HOMOGENEOUS, uuids=None):
            profile = build_auto_ladder_profile(sa)
        self.assertEqual([c.index for c in profile.cards], [0, 1])
        self.assertEqual(profile.geometries[0].gpus, (0, 1))

    def test_co_located_ranks_with_conflicting_budgets_refuse(self):
        sa = _ServerArgs(
            tp_size=2,
            rank_gpu_id=[0, 0],
            rank_gpu_memory_mib=[15000, 14000],
        )
        with _NvmlContext(_HETERO, uuids=("GPU-aaa", "GPU-aaa")):
            with self.assertRaises(ValueError) as ctx:
                build_auto_ladder_profile(sa)
        self.assertIn("co-located", str(ctx.exception))

    def test_capacities_are_labelled_placeholders_not_guesses(self):
        """kv_bytes_per_token / weight_bytes_total are deliberately absent,
        so every rung must SAY its capacity is unmeasured."""
        sa = _ServerArgs(tp_size=2)
        with _NvmlContext(_HOMOGENEOUS):
            table = auto_ladder_table_fn(sa)()
        self.assertTrue(
            all(step.provenance == "placeholder" for step in table.steps),
            [s.provenance for s in table.steps],
        )


class TestOnlyWiredReliefsAreInventoried(CustomTestCase):
    def test_no_actuator_no_relief_rung(self):
        self.assertEqual(wired_relief_features(_ServerArgs()), ())

    def test_each_flag_adds_exactly_its_rung(self):
        self.assertEqual(
            wired_relief_features(_ServerArgs(max_running_requests_ceiling=64)),
            ("admission_cap",),
        )
        self.assertEqual(
            wired_relief_features(_ServerArgs(enable_kv_session_offload=True)),
            ("session_offload",),
        )
        self.assertEqual(
            wired_relief_features(_ServerArgs(kv_reshard_vectors="1,1")),
            ("dcp_ratio",),
        )

    def test_planned_only_features_are_never_inventoried(self):
        """``kv_spill`` and ``weightless_rank`` move nothing at this tip; an
        auto table naming them would advertise a rung that does nothing."""
        sa = _ServerArgs(
            max_running_requests_ceiling=64,
            enable_kv_session_offload=True,
            kv_reshard_vectors="1,1",
            weightless_kv_fastlane=True,
        )
        self.assertNotIn("kv_spill", wired_relief_features(sa))
        self.assertNotIn("weightless_rank", wired_relief_features(sa))

    def test_auto_table_never_names_a_rung_the_runtime_would_reject(self):
        """The runtime refuses an ``admission_cap`` rung without an armed
        limiter. With the relief gate in place, auto cannot produce one."""
        sa = _ServerArgs(tp_size=2, max_running_requests_ceiling=None)
        sched = _Scheduler(sa)
        self.assertIsNone(sched.admission_limiter)
        with _NvmlContext(_HOMOGENEOUS):
            runtime = build_kv_pressure_runtime(sched)
        names = [s.relief_feature for s in runtime.ladder.table.steps]
        self.assertNotIn("admission_cap", names)


if __name__ == "__main__":
    unittest.main()
