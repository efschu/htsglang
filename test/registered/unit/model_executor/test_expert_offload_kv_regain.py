"""CPU unit tests for #119: expert-offload weight VRAM -> KV pool.

The expert offload (#77/#123) parks cold MoE experts in a pinned host pool and
gives their VRAM back. The KV budget is profiled from a live free-memory
reading, so that VRAM flows into the pool by itself -- but only if the release
has happened, and happened on every rank, before anyone measures. These tests
pin down the four properties that make the reclaim real instead of incidental:

  * the released-bytes accounting is a pure, exact function of the split;
  * the reclaim lane is entered only with the offload on (default path is
    byte-identical, and takes no extra collective);
  * the "reclaim is present" verdict is a GROUP MINIMUM, never rank-local;
  * release strictly precedes measurement (gc -> empty_cache -> barrier ->
    read), and the reclaim lands 1:1 in the budget without touching the
    graph-capture reserve.

No GPU, no distributed init: collectives and the memory readings are patched,
and `sgl_kernel` is stubbed before the sglang imports.
"""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch


def _install_sgl_kernel_stub():
    def _make(name, pkg=False):
        mod = types.ModuleType(name)
        if pkg:
            mod.__path__ = []

        def _getattr(attr):
            if attr.startswith("__"):
                raise AttributeError(attr)
            return lambda *a, **k: None

        mod.__getattr__ = _getattr
        sys.modules.setdefault(name, mod)

    _make("sgl_kernel", pkg=True)
    _make("sgl_kernel.quantization")
    _make("sgl_kernel.kvcacheio")


_install_sgl_kernel_stub()

import torch  # noqa: E402

import sglang.srt.model_executor.model_runner_kv_cache_mixin as mixin  # noqa: E402
from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.layers.moe.expert_offload import (  # noqa: E402
    ExpertOffloadRelease,
    expert_offload_release_totals,
    expert_offload_released_device_bytes,
    record_expert_offload_release,
    reset_expert_offload_release,
)
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

GB = 1 << 30

MIXIN = mixin.ModelRunnerKVCacheMixin


def _fake_group(world_size=1):
    return SimpleNamespace(world_size=world_size, cpu_group=object())


def _bind_lane_methods(fake_self):
    """Attach the #119 helpers to a stub ModelRunner, as the mixin would."""
    import types as _types

    for name in (
        "_expert_offload_lane_active",
        "_assert_expert_offload_installed",
        "_expert_offload_reclaim_active",
        "_expert_offload_release_sync",
    ):
        setattr(
            fake_self, name, _types.MethodType(getattr(MIXIN, name), fake_self)
        )
    return fake_self


class TestReleasedBytesMath(CustomTestCase):
    """The pure sizing function: exact, and silent off the offload lane."""

    def test_released_bytes_is_the_dropped_rows(self):
        # 256 experts, 80 slots resident+scratch, 1 MiB per expert row.
        self.assertEqual(
            expert_offload_released_device_bytes(256, 80, 1 << 20), 176 << 20
        )

    def test_fully_resident_releases_nothing(self):
        # buffer_slots >= experts is the no-offload configuration.
        self.assertEqual(expert_offload_released_device_bytes(64, 64, 4096), 0)
        self.assertEqual(expert_offload_released_device_bytes(64, 999, 4096), 0)

    def test_degenerate_inputs_are_zero_not_negative(self):
        self.assertEqual(expert_offload_released_device_bytes(0, 0, 4096), 0)
        self.assertEqual(expert_offload_released_device_bytes(-8, 4, 4096), 0)
        self.assertEqual(expert_offload_released_device_bytes(64, -4, 4096), 64 * 4096)
        self.assertEqual(expert_offload_released_device_bytes(64, 4, -1), 0)


class TestReleaseTally(CustomTestCase):
    """The per-rank tally accumulates per layer and hands out copies."""

    def setUp(self):
        super().setUp()
        reset_expert_offload_release()
        self.addCleanup(reset_expert_offload_release)

    def test_starts_empty(self):
        self.assertEqual(expert_offload_release_totals(), ExpertOffloadRelease())

    def test_accumulates_per_layer(self):
        record_expert_offload_release(3 * GB, 4 * GB, tensors=4)
        record_expert_offload_release(2 * GB, 3 * GB, tensors=4)
        totals = expert_offload_release_totals()
        self.assertEqual(totals.device_bytes, 5 * GB)
        self.assertEqual(totals.host_bytes, 7 * GB)
        self.assertEqual(totals.layers, 2)
        self.assertEqual(totals.tensors, 8)

    def test_snapshot_is_a_copy(self):
        record_expert_offload_release(GB, GB)
        snap = expert_offload_release_totals()
        snap.device_bytes = 999
        self.assertEqual(expert_offload_release_totals().device_bytes, GB)

    def test_reset_clears(self):
        record_expert_offload_release(GB, GB)
        reset_expert_offload_release()
        self.assertEqual(expert_offload_release_totals(), ExpertOffloadRelease())


class TestLaneGate(CustomTestCase):
    """Both gate terms are env vars, hence world-uniform by construction."""

    def _active(self, *, regain, fraction):
        with envs.SGLANG_MOE_OFFLOAD_KV_REGAIN.override(regain), (
            envs.SGLANG_MOE_RESIDENT_EXPERT_FRACTION.override(fraction)
        ):
            return MIXIN._expert_offload_lane_active(SimpleNamespace())

    def test_default_no_offload_is_off_lane(self):
        self.assertFalse(self._active(regain=True, fraction=1.0))

    def test_offload_on_is_on_lane(self):
        self.assertTrue(self._active(regain=True, fraction=0.25))

    def test_switch_disables_the_lane(self):
        self.assertFalse(self._active(regain=False, fraction=0.25))


class TestReclaimVerdictIsGroupMinimum(CustomTestCase):
    """The pool decision is rank-uniform: MIN over the group, never local."""

    def setUp(self):
        super().setUp()
        reset_expert_offload_release()
        self.addCleanup(reset_expert_offload_release)

    def _run(self, *, world_size, local_layers, other_verdict):
        if local_layers:
            record_expert_offload_release(GB, GB)
        calls = []

        def fake_all_reduce(tensor, op=None, group=None):
            calls.append(op)
            tensor.copy_(torch.minimum(tensor, torch.tensor(other_verdict)))

        with patch.object(
            mixin, "get_world_group", return_value=_fake_group(world_size)
        ), patch("torch.distributed.all_reduce", side_effect=fake_all_reduce):
            verdict = MIXIN._expert_offload_reclaim_active(SimpleNamespace())
        return verdict, calls

    def test_single_process_uses_the_local_answer(self):
        verdict, calls = self._run(world_size=1, local_layers=2, other_verdict=0)
        self.assertTrue(verdict)
        self.assertEqual(calls, [])

    def test_all_ranks_released_is_active(self):
        verdict, calls = self._run(world_size=3, local_layers=2, other_verdict=1)
        self.assertTrue(verdict)
        self.assertEqual(calls, [torch.distributed.ReduceOp.MIN])

    def test_one_silent_rank_disables_the_group(self):
        # This rank released VRAM, a peer did not -> the whole group falls back.
        verdict, _ = self._run(world_size=3, local_layers=2, other_verdict=0)
        self.assertFalse(verdict)

    def test_local_miss_disables_even_if_peers_released(self):
        verdict, _ = self._run(world_size=3, local_layers=0, other_verdict=1)
        self.assertFalse(verdict)


class TestReleaseSyncOrdering(CustomTestCase):
    """collect -> empty_cache -> barrier, in that order, once."""

    def _run(self, world_size):
        order = []
        with patch.object(
            mixin, "get_world_group", return_value=_fake_group(world_size)
        ), patch("gc.collect", side_effect=lambda: order.append("collect")), patch.object(
            mixin, "empty_device_cache", side_effect=lambda _: order.append("empty")
        ), patch(
            "torch.distributed.barrier",
            side_effect=lambda group=None: order.append("barrier"),
        ):
            MIXIN._expert_offload_release_sync(SimpleNamespace())
        return order

    def test_group_release_is_ordered_and_synchronized(self):
        self.assertEqual(self._run(3), ["collect", "empty", "barrier"])

    def test_single_process_needs_no_barrier(self):
        self.assertEqual(self._run(1), ["collect", "empty"])


class _StubMoE:
    """Stands in for FusedMoE without importing the CUDA-flavoured layer."""

    def __init__(self, layer_id, enabled=True, installed=True, failed=False):
        self.layer_id = layer_id
        self._moe_offload_enabled = enabled
        self._expert_offload = object() if installed else None
        self._expert_offload_install_failed = failed


def _stub_fused_moe_module():
    mod = types.ModuleType("sglang.srt.layers.moe.fused_moe_triton.layer")
    mod.FusedMoE = _StubMoE
    return {"sglang.srt.layers.moe.fused_moe_triton.layer": mod}


class TestInstallOrderingInvariant(CustomTestCase):
    """A layer still waiting to install would release its VRAM after sizing."""

    def _check(self, modules, *, weightless=False):
        model = SimpleNamespace(modules=lambda: list(modules))
        fake_self = SimpleNamespace(
            model=model, tp_rank=0, is_weightless_worker=weightless
        )
        with patch.dict(sys.modules, _stub_fused_moe_module()):
            MIXIN._assert_expert_offload_installed(fake_self)

    def test_installed_layers_pass(self):
        self._check([_StubMoE(0), _StubMoE(1)])

    def test_pending_layer_fails_fast(self):
        with self.assertRaises(ValueError) as ctx:
            self._check([_StubMoE(0), _StubMoE(7, installed=False)])
        msg = str(ctx.exception)
        self.assertIn("[7]", msg)
        self.assertIn("before the KV pool is sized", msg)
        self.assertIn("SGLANG_MOE_OFFLOAD_KV_REGAIN=0", msg)

    def test_offload_disabled_layer_is_not_pending(self):
        self._check([_StubMoE(0, enabled=False, installed=False)])

    def test_failed_install_fell_back_and_is_not_pending(self):
        self._check([_StubMoE(0, installed=False, failed=True)])

    def test_weightless_worker_is_skipped(self):
        self._check([_StubMoE(0, installed=False)], weightless=True)

    def test_no_model_is_skipped(self):
        fake_self = SimpleNamespace(model=None, tp_rank=0)
        MIXIN._assert_expert_offload_installed(fake_self)


class TestProfileAvailableBytes(CustomTestCase):
    """End-to-end budget: reclaim lands 1:1, reserves untouched, order kept."""

    def setUp(self):
        super().setUp()
        reset_expert_offload_release()
        self.addCleanup(reset_expert_offload_release)

    def _profile(
        self,
        *,
        fraction,
        available_gb,
        pre_gb=30.0,
        mem_fraction_static=0.9,
        world_size=1,
        released_gb=0.0,
        regain=True,
    ):
        """Run _profile_available_bytes against mocked memory readings.

        Returns (budget_bytes, call_order).
        """
        if released_gb:
            record_expert_offload_release(int(released_gb * GB), int(released_gb * GB))
        order = []

        fake_self = _bind_lane_methods(
            SimpleNamespace(
                server_args=SimpleNamespace(
                    uneven_memory_budgets_active=lambda: False
                ),
                device="cuda",
                gpu_id=0,
                tp_rank=0,
                model=SimpleNamespace(modules=lambda: []),
                is_weightless_worker=False,
                mambaish_config=None,
                post_capture_kv_active=False,
                mem_fraction_static=mem_fraction_static,
                _colocated_sibling_reserved_gb=lambda: 0.0,
                _measured_kv_budget_correction_bytes=lambda: 0,
            )
        )

        def fake_available(device, gpu_id, distributed=False, cpu_group=None):
            order.append("read")
            return available_gb

        with envs.SGLANG_MOE_OFFLOAD_KV_REGAIN.override(regain), (
            envs.SGLANG_MOE_RESIDENT_EXPERT_FRACTION.override(fraction)
        ), patch.object(
            mixin, "get_world_group", return_value=_fake_group(world_size)
        ), patch.object(
            mixin, "get_available_gpu_memory", side_effect=fake_available
        ), patch.object(
            mixin, "empty_device_cache", side_effect=lambda _: order.append("empty")
        ), patch(
            "gc.collect", side_effect=lambda: order.append("collect")
        ), patch(
            "torch.distributed.barrier",
            side_effect=lambda group=None: order.append("barrier"),
        ), patch(
            "torch.distributed.all_reduce",
            side_effect=lambda t, op=None, group=None: order.append("all_reduce"),
        ), patch(
            "torch.cuda.memory_allocated", return_value=0
        ), patch(
            "torch.cuda.memory_reserved", return_value=0
        ), patch.dict(
            sys.modules, _stub_fused_moe_module()
        ):
            budget = MIXIN._profile_available_bytes(fake_self, pre_gb)
        return budget, order

    def test_default_path_untouched(self):
        # No offload: budget = available - pre * (1 - mem_fraction_static),
        # and NOT a single extra collective or cache operation.
        budget, order = self._profile(fraction=1.0, available_gb=12.0)
        self.assertAlmostEqual(budget / GB, 12.0 - 30.0 * 0.1, places=5)
        self.assertEqual(order, ["read"])

    def test_switch_off_restores_the_default_path(self):
        budget, order = self._profile(
            fraction=0.25, available_gb=12.0, released_gb=4.0, regain=False
        )
        self.assertAlmostEqual(budget / GB, 12.0 - 3.0, places=5)
        self.assertEqual(order, ["read"])

    def test_release_precedes_the_measurement(self):
        _, order = self._profile(
            fraction=0.25, available_gb=16.0, released_gb=4.0, world_size=3
        )
        # all_reduce = the group-minimum verdict; then the ordered release;
        # only afterwards may any rank read driver-level free memory.
        self.assertEqual(
            order, ["all_reduce", "collect", "empty", "barrier", "read"]
        )
        self.assertLess(order.index("barrier"), order.index("read"))

    def test_reclaim_lands_one_to_one_and_spares_the_graph_reserve(self):
        # Same rig, same reserves; the offload hands back 4 GiB of weight VRAM,
        # so the free reading is 4 GiB higher. The budget must grow by exactly
        # that -- no more (the reserve is not raided) and no less (the reclaim
        # is not swallowed).
        base, _ = self._profile(fraction=1.0, available_gb=12.0)
        with_offload, _ = self._profile(
            fraction=0.25, available_gb=16.0, released_gb=4.0
        )
        self.assertAlmostEqual((with_offload - base) / GB, 4.0, places=5)
        # The slack term is what carries the #68 graph-capture reserve; it is a
        # function of pre_model_load_memory and mem_fraction_static alone, so it
        # is identical on both runs.
        self.assertAlmostEqual(base / GB, 12.0 - 3.0, places=5)
        self.assertAlmostEqual(with_offload / GB, 16.0 - 3.0, places=5)

    def test_pending_install_aborts_before_any_measurement(self):
        order = []
        fake_self = _bind_lane_methods(
            SimpleNamespace(
                server_args=SimpleNamespace(
                    uneven_memory_budgets_active=lambda: False
                ),
                device="cuda",
                gpu_id=0,
                tp_rank=1,
                model=SimpleNamespace(
                    modules=lambda: [_StubMoE(3, installed=False)]
                ),
                is_weightless_worker=False,
                mambaish_config=None,
                post_capture_kv_active=False,
                mem_fraction_static=0.9,
                _colocated_sibling_reserved_gb=lambda: 0.0,
                _measured_kv_budget_correction_bytes=lambda: 0,
            )
        )
        with envs.SGLANG_MOE_RESIDENT_EXPERT_FRACTION.override(0.25), patch.object(
            mixin, "get_world_group", return_value=_fake_group(1)
        ), patch.object(
            mixin,
            "get_available_gpu_memory",
            side_effect=lambda *a, **k: order.append("read") or 12.0,
        ), patch.dict(
            sys.modules, _stub_fused_moe_module()
        ):
            with self.assertRaises(ValueError):
                MIXIN._profile_available_bytes(fake_self, 30.0)
        self.assertEqual(order, [])


if __name__ == "__main__":
    unittest.main()
