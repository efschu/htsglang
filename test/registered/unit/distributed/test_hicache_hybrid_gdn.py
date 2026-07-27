"""CPU unit tests for hybrid-model (GDN) HiCache compatibility.

Covers the two host-side fixes:
1. sync_fixed_hicache_size — the fixed --hicache-size token capacity is
   min-synced not only across PP stages but also across uneven-TP ranks
   (--rank-tp-ratio), whose bytes/token differ with their kv-head/GDN
   state share.
2. HiMambaRadixCache.normalize_hicache_args — hybrid-Mamba HiCache only
   works with the page_first_direct host layout + direct IO backend;
   the stock defaults (page_first + kernel) previously crashed deep in
   MambaPoolHost, now they are normalized with a warning up front.

No GPU, no distributed init: collectives and parallel groups are
patched; `sgl_kernel` is stubbed before the sglang imports.
"""

import importlib.util
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch


def _install_sgl_kernel_stub():
    if importlib.util.find_spec("sgl_kernel") is not None:
        # The real package is importable; stubbing it here would leave a
        # process-wide empty-__path__ package that breaks every later
        # ``import sgl_kernel.<submodule>`` in the same pytest run.
        return
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

import sglang.srt.distributed.parallel_state as parallel_state  # noqa: E402
from sglang.srt.distributed.utils import set_tp_partition_ratios  # noqa: E402
from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache  # noqa: E402
from sglang.srt.mem_cache.pool_host.base import sync_fixed_hicache_size  # noqa: E402
from sglang.srt.server_args import ServerArgs  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class HiCacheTestCase(CustomTestCase):
    def setUp(self):
        set_tp_partition_ratios(None)

    def tearDown(self):
        set_tp_partition_ratios(None)


class TestSyncFixedHiCacheSize(HiCacheTestCase):
    def _run(self, size, *, plan, world_size, pp_world_size, other_size):
        groups_used = []

        def fake_all_reduce(tensor, op=None, group=None):
            groups_used.append(group)
            tensor.copy_(torch.minimum(tensor, torch.tensor(other_size)))

        world_group = SimpleNamespace(world_size=world_size, cpu_group="world")
        pp_group = SimpleNamespace(world_size=pp_world_size, cpu_group="pp")
        set_tp_partition_ratios(plan)
        with patch.object(torch.distributed, "is_available", return_value=True), (
            patch.object(torch.distributed, "is_initialized", return_value=True)
        ), patch.object(
            parallel_state, "get_world_group", return_value=world_group
        ), patch.object(
            parallel_state, "get_pp_group", return_value=pp_group
        ), patch.object(
            torch.distributed, "all_reduce", side_effect=fake_all_reduce
        ):
            result = sync_fixed_hicache_size(size, host_size=10)
        return result, groups_used

    def test_ratio_based_sizing_skips_sync(self):
        # host_size <= 0 means ratio-based sizing: no collective at all.
        self.assertEqual(sync_fixed_hicache_size(1234, host_size=0), 1234)

    def test_not_initialized_returns_unchanged(self):
        set_tp_partition_ratios([2, 1, 1])
        # torch.distributed is not initialized in the CPU test process.
        self.assertEqual(sync_fixed_hicache_size(1234, host_size=10), 1234)

    def test_pp_min_sync_unchanged(self):
        # Classic PP behavior: min over the PP group.
        result, groups = self._run(
            1000, plan=None, world_size=6, pp_world_size=2, other_size=800
        )
        self.assertEqual(result, 800)
        self.assertEqual(groups, ["pp"])

    def test_no_plan_single_pp_stage_no_sync(self):
        result, groups = self._run(
            1000, plan=None, world_size=3, pp_world_size=1, other_size=1
        )
        self.assertEqual(result, 1000)
        self.assertEqual(groups, [])

    def test_uneven_tp_min_syncs_over_world_group(self):
        # Uneven TP (pure TP, pp=1): bytes/token differ per rank, so the
        # fixed-GB capacity must be min-synced over the WORLD group.
        result, groups = self._run(
            1000, plan=[2, 1, 1], world_size=3, pp_world_size=1, other_size=500
        )
        self.assertEqual(result, 500)
        self.assertEqual(groups, ["world"])


class TestHybridMambaHiCacheNormalization(HiCacheTestCase):
    def _args(self, **kwargs):
        return ServerArgs(model_path="dummy", **kwargs)

    def test_stock_defaults_are_normalized(self):
        # Defaults: page_first + kernel -> previously a late ValueError in
        # MambaPoolHost; now normalized to page_first_direct + direct.
        args = self._args()
        self.assertEqual(args.hicache_mem_layout, "page_first")
        self.assertEqual(args.hicache_io_backend, "kernel")
        HiMambaRadixCache.normalize_hicache_args(args)
        self.assertEqual(args.hicache_mem_layout, "page_first_direct")
        self.assertEqual(args.hicache_io_backend, "direct")

    def test_layer_first_direct_only_layout_switches(self):
        args = self._args(
            hicache_mem_layout="layer_first", hicache_io_backend="direct"
        )
        HiMambaRadixCache.normalize_hicache_args(args)
        self.assertEqual(args.hicache_mem_layout, "page_first_direct")
        self.assertEqual(args.hicache_io_backend, "direct")

    def test_supported_combination_is_untouched(self):
        args = self._args(
            hicache_mem_layout="page_first_direct", hicache_io_backend="direct"
        )
        HiMambaRadixCache.normalize_hicache_args(args)
        self.assertEqual(args.hicache_mem_layout, "page_first_direct")
        self.assertEqual(args.hicache_io_backend, "direct")

    def test_kernel_ascend_io_is_kept(self):
        # Unknown/NPU IO backend: enforce the layout, keep the backend.
        args = self._args(
            hicache_mem_layout="page_first", hicache_io_backend="kernel_ascend"
        )
        HiMambaRadixCache.normalize_hicache_args(args)
        self.assertEqual(args.hicache_mem_layout, "page_first_direct")
        self.assertEqual(args.hicache_io_backend, "kernel_ascend")


if __name__ == "__main__":
    unittest.main()
