# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#923: the HiCache controller's owner-ctx memo must not survive a cutover.

THE SPECIMEN, 2026-08-27, two of three boots (window acceptance-2d-0827):

    File ".../mem_cache/pool_host/mha.py", line 465, in
      backup_from_device_all_layer -> transfer_kv_direct(
    File ".../sgl_kernel/kvcacheio.py", line 190, in transfer_kv_direct
    RuntimeError: The size of tensor a (3) must match the size of tensor b (0)
      at non-singleton dimension 0

with tensor a taking 3, 621 and 2294 across the three occurrences and tensor b
0 in every one. Reached from ``process_batch_result_decode ->
_handle_finish_state_updated_req -> release_kv_cache -> cache_finished_req ->
insert -> _inc_hit_count -> write_backup``, i.e. whenever a request FINISHES
under ``--hicache-write-policy write_through``.

WHICH SIDE IS EMPTY, settled rather than assumed. ``transfer_kv_direct``
(sgl-kernel/csrc/kvcacheio/transfer.cu:742) ends in

    dst_buffer.slice(0, dst_index, dst_index + n).copy_(
        src_buffer.slice(0, src_index, src_index + n), true)

and ``copy_`` builds its TensorIterator output-first, so "a" is the
DESTINATION and "b" the SOURCE. a=3 / b=0 therefore reads: the host
destination had its three rows and the DEVICE SOURCE slice was empty --
because ``at::Tensor::slice`` CLAMPS an out-of-range start instead of raising.
The device row id was past the end of this rank's KV buffer.
:class:`TestTheCrashFormIsASourceSideClamp` pins that reading to torch itself.

THE ROOT. ``HiCacheController._dcp_owner_ctx`` memoized the weighted uneven-DCP
owner range for process life, on the documented reasoning "the token vector is
installed once at engine init, before the cache controller is constructed".
The phase flip falsifies the reasoning: ``phase_flip_runtime._cutover``
(phase_flip_runtime.py:2728) reinstalls the token vector on EVERY leg and
``dcp_size`` goes 1 -> 3 across the flip. The memo taken in the PP phase says
None, which makes ``_dcp_kv_transfer_pairs`` the identity, which hands the TP
phase's COMPACT device pool a GLOBAL allocator slot id. Below the pool's row
count that silently copies another token's KV under this request's key; above
it, the slice clamps and the copy dies as above.

It was a KNOWN, WRITTEN-DOWN, UNCLOSED site: ``docs/dev/631/
PROD_BRINGUP_BENCH.md`` §6a names it verbatim -- "the cutover does not
invalidate ``cache_controller._dcp_owner_ctx_cache`` (the #297 cutover does,
and ``dcp_size`` changes 1->3 across the flip)".

THE FIX IS THE REGISTRY, NOT A SECOND DELATTR. There was exactly one hand
-written invalidation, in ``kv_reshard._cutover_fn_for``. Adding a second one
in the flip would have been a third mover of the same payload. The controller
now registers with the #297 owner-bounds registry, so every installer of a
token vector drops the memo through the same call -- and the reshard's bespoke
line is gone.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.dcp.owner import refresh_all_owner_bounds
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_BOUNDS = "sglang.srt.distributed.utils.uneven_dcp_owner_bounds"

# The rig's TP-phase decode geometry: token vector 29,19,16 -> S = 64, and this
# rank owns [0, 29). A global context of C slots compacts to
# (C // S + 1) * ratio rows, so rank 0's pool is far smaller than C.
_S, _LO, _HI = 64, 0, 29


def _controller(rows: int = 4096) -> HiCacheController:
    """A controller with only the state #923 reads. ``__new__`` because the
    real constructor builds CUDA streams and a device pool."""
    ctrl = HiCacheController.__new__(HiCacheController)
    ctrl.mem_pool_device = SimpleNamespace(k_buffer=[torch.zeros((rows, 1, 1))])
    ctrl._register_owner_bounds_refresh()
    return ctrl


class TestTheCrashFormIsASourceSideClamp(unittest.TestCase):
    """torch itself, so the reading of the metal message is not an opinion."""

    def test_an_out_of_range_source_row_clamps_to_empty_and_reports_a_then_b(self):
        dst = torch.zeros(100, 8)
        src = torch.zeros(50, 8)
        # slice() does not raise on an out-of-range start; it returns nothing.
        self.assertEqual(tuple(src[60:63].shape), (0, 8))
        with self.assertRaises(RuntimeError) as caught:
            dst[10:13].copy_(src[60:63], True)
        self.assertIn(
            "The size of tensor a (3) must match the size of tensor b (0)",
            str(caught.exception),
        )

    def test_the_mirror_case_reports_the_numbers_the_other_way_round(self):
        # If the HOST destination had been the empty side, the metal would have
        # said "a (0) ... b (3)". It said a (3) / b (0) on all three specimens.
        dst = torch.zeros(100, 8)
        src = torch.zeros(50, 8)
        with self.assertRaises(RuntimeError) as caught:
            dst[120:123].copy_(src[10:13], True)
        self.assertIn(
            "The size of tensor a (0) must match the size of tensor b (3)",
            str(caught.exception),
        )


class TestTheMemoIsDroppedAtEveryCutover(unittest.TestCase):
    def test_a_cutover_re_derives_the_owner_ctx(self):
        ctrl = _controller()
        with mock.patch(_BOUNDS, return_value=None):
            # PP phase: dcp_size == 1, no owner rule.
            self.assertIsNone(ctrl._dcp_owner_ctx())
        with mock.patch(_BOUNDS, return_value=(_S, _LO, _HI)):
            # THE REGRESSION: without the refresh this still answered None for
            # the rest of the process, and every TP-phase transfer went through
            # the identity mapping.
            refresh_all_owner_bounds()
            self.assertEqual(ctrl._dcp_owner_ctx(), (_S, _LO, _HI))

    def test_the_return_leg_drops_it_again(self):
        ctrl = _controller()
        with mock.patch(_BOUNDS, return_value=(_S, _LO, _HI)):
            self.assertEqual(ctrl._dcp_owner_ctx(), (_S, _LO, _HI))
        with mock.patch(_BOUNDS, return_value=None):
            refresh_all_owner_bounds()
            self.assertIsNone(ctrl._dcp_owner_ctx())

    def test_the_hybrid_controller_is_registered_too(self):
        # The hybrid subclass IS the write path on this rig; inheriting the
        # method is not the same as being in the registry.
        ctrl = HybridCacheController.__new__(HybridCacheController)
        ctrl.mem_pool_device = SimpleNamespace(k_buffer=[torch.zeros((16, 1, 1))])
        ctrl._register_owner_bounds_refresh()
        with mock.patch(_BOUNDS, return_value=None):
            self.assertIsNone(ctrl._dcp_owner_ctx())
        with mock.patch(_BOUNDS, return_value=(_S, _LO, _HI)):
            refresh_all_owner_bounds()
            self.assertEqual(ctrl._dcp_owner_ctx(), (_S, _LO, _HI))

    def test_registration_happens_in_the_constructor(self):
        # PRESENT-BUT-UNWIRED is the expensive middle state: a refresh hook
        # nobody registers reads exactly like a fixed bug. The constructor call
        # is the wiring, so it is asserted rather than assumed.
        import inspect

        source = inspect.getsource(HiCacheController.__init__)
        self.assertIn("self._register_owner_bounds_refresh()", source)


class TestTheStaleMemoIsWhatReachedTheKernel(unittest.TestCase):
    """The mechanism, stated in indices rather than in prose."""

    def test_a_stale_none_hands_a_global_slot_to_a_compact_pool(self):
        ctrl = _controller(rows=4096)
        host = torch.arange(4, dtype=torch.int64)
        # Global slots this rank owns (L % 64 in [0, 29)), high in the id space.
        device = torch.tensor([200000, 200001, 200002, 200003], dtype=torch.int64)
        with mock.patch(_BOUNDS, return_value=None):
            _, rows = ctrl._dcp_kv_transfer_pairs(host, device)
        # Identity: the row ids are the global ids, far past a 4096-row pool.
        self.assertEqual(rows.tolist(), device.tolist())
        self.assertGreaterEqual(int(rows.max()), 4096)

    def test_the_fresh_ctx_maps_them_into_the_pool(self):
        ctrl = _controller(rows=4096)
        host = torch.arange(4, dtype=torch.int64)
        device = torch.tensor([200000, 200001, 200002, 200003], dtype=torch.int64)
        with mock.patch(_BOUNDS, return_value=(_S, _LO, _HI)):
            host_owned, rows = ctrl._dcp_kv_transfer_pairs(host, device)
        # 200000 % 64 == 32, which this rank does NOT own; 200001..200003 map to
        # offsets 33..35, also outside [0, 29). Nothing owned -> nothing copied,
        # and the host side is filtered by the SAME mask.
        self.assertEqual(rows.numel(), host_owned.numel())
        owned_offsets = [int(x) % _S for x in device if _LO <= int(x) % _S < _HI]
        self.assertEqual(rows.numel(), len(owned_offsets))

    def test_owned_slots_compact_below_the_pool_row_count(self):
        ctrl = _controller(rows=4096)
        # Slots this rank genuinely owns: offset 0..28 of block 100.
        device = torch.tensor([100 * _S + off for off in range(5)], dtype=torch.int64)
        host = torch.arange(device.numel(), dtype=torch.int64)
        with mock.patch(_BOUNDS, return_value=(_S, _LO, _HI)):
            host_owned, rows = ctrl._dcp_kv_transfer_pairs(host, device)
        self.assertEqual(rows.tolist(), [100 * (_HI - _LO) + off for off in range(5)])
        self.assertLess(int(rows.max()), 4096)
        self.assertEqual(host_owned.tolist(), host.tolist())


class TestUnaddressableRowsAreSaidAndCounted(unittest.TestCase):
    """A cache write may fail. It may not fail QUIETLY (#767 class)."""

    def test_an_in_range_transfer_is_not_refused(self):
        ctrl = _controller(rows=4096)
        with mock.patch(_BOUNDS, return_value=None):
            device = torch.arange(10, dtype=torch.int64)
            self.assertFalse(ctrl._refuse_unaddressable_kv_rows(device, "write"))
        self.assertEqual(getattr(ctrl, "_unaddressable_kv_rows_refused", 0), 0)

    def test_an_out_of_range_row_is_refused_and_counted(self):
        ctrl = _controller(rows=4096)
        with mock.patch(_BOUNDS, return_value=None):
            device = torch.tensor([4095, 4096], dtype=torch.int64)
            self.assertTrue(ctrl._refuse_unaddressable_kv_rows(device, "write"))
            self.assertTrue(ctrl._refuse_unaddressable_kv_rows(device, "write"))
        self.assertEqual(ctrl._unaddressable_kv_rows_refused, 2)

    def test_the_refusal_names_pool_rows_and_the_owner_ctx(self):
        ctrl = _controller(rows=4096)
        with mock.patch(_BOUNDS, return_value=None):
            device = torch.tensor([200000], dtype=torch.int64)
            with self.assertLogs(
                "sglang.srt.managers.cache_controller", level="ERROR"
            ) as logs:
                self.assertTrue(ctrl._refuse_unaddressable_kv_rows(device, "write"))
        line = "\n".join(logs.output)
        self.assertIn("#923", line)
        self.assertIn("200000", line)
        self.assertIn("4096", line)

    def test_an_empty_transfer_is_not_a_refusal(self):
        ctrl = _controller(rows=4096)
        with mock.patch(_BOUNDS, return_value=None):
            empty = torch.zeros(0, dtype=torch.int64)
            self.assertFalse(ctrl._refuse_unaddressable_kv_rows(empty, "write"))

    def test_an_unreadable_pool_fails_open(self):
        # Absence is not a mismatch: the same contract as the #760 seam guard.
        ctrl = HiCacheController.__new__(HiCacheController)
        ctrl.mem_pool_device = SimpleNamespace()
        ctrl._register_owner_bounds_refresh()
        with mock.patch(_BOUNDS, return_value=None):
            device = torch.tensor([10**9], dtype=torch.int64)
            self.assertFalse(ctrl._refuse_unaddressable_kv_rows(device, "write"))

    def test_only_owned_rows_are_judged(self):
        # A global slot this rank does not own is not copied at all, so it can
        # never be an out-of-range row here. Refusing on it would turn a normal
        # DCP transfer into a permanent HiCache outage.
        ctrl = _controller(rows=4096)
        with mock.patch(_BOUNDS, return_value=(_S, _LO, _HI)):
            device = torch.tensor([10**7 + 40], dtype=torch.int64)  # offset 40
            self.assertFalse(ctrl._refuse_unaddressable_kv_rows(device, "write"))


class TestTheWriteRefusalReachesTheCaller(unittest.TestCase):
    def _bind_write(self, cls, rows: int):
        ctrl = cls.__new__(cls)
        ctrl.mem_pool_device = SimpleNamespace(k_buffer=[torch.zeros((rows, 1, 1))])
        ctrl._register_owner_bounds_refresh()
        ctrl.write_queue = []
        allocated = []

        def _alloc(n):
            allocated.append(n)
            return torch.arange(n, dtype=torch.int64)

        ctrl.mem_pool_host = SimpleNamespace(alloc=_alloc)
        return ctrl, allocated

    def test_the_base_write_returns_none_without_allocating_a_host_page(self):
        ctrl, allocated = self._bind_write(HiCacheController, 4096)
        with (
            mock.patch(_BOUNDS, return_value=None),
            mock.patch(
                "sglang.srt.managers.cache_controller.device_tier_disarmed",
                return_value=False,
            ),
        ):
            out = ctrl.write(torch.tensor([200000, 200001], dtype=torch.int64))
        self.assertIsNone(out)
        # A refusal that had already taken host slots would strand them.
        self.assertEqual(allocated, [])
        self.assertEqual(ctrl.write_queue, [])

    def test_the_hybrid_write_returns_none_too(self):
        ctrl, allocated = self._bind_write(HybridCacheController, 4096)
        with (
            mock.patch(_BOUNDS, return_value=None),
            mock.patch(
                "sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller."
                "device_tier_disarmed",
                return_value=False,
            ),
        ):
            out = ctrl.write(torch.tensor([200000, 200001], dtype=torch.int64))
        self.assertIsNone(out)
        self.assertEqual(allocated, [])
        self.assertEqual(ctrl.write_queue, [])


class TestTheSeamGuardNowCarriesTheCapacities(unittest.TestCase):
    """The last line, only reachable if the enqueue refusal was bypassed."""

    def _pools(self, device_rows: int, host_rows: int):
        device_pool = SimpleNamespace(
            k_buffer=[torch.zeros((device_rows, 1, 1)) for _ in range(2)],
            k_data_ptrs=torch.zeros(2, dtype=torch.uint64),
            v_data_ptrs=torch.zeros(2, dtype=torch.uint64),
            token_stride_size=64,
            _binding_generation=1,
        )
        host_pool = SimpleNamespace(
            layout="layer_first",
            k_data_refs=[torch.zeros((host_rows, 1, 1)) for _ in range(2)],
            k_data_ptrs=torch.zeros(2, dtype=torch.uint64),
            v_data_ptrs=torch.zeros(2, dtype=torch.uint64),
            token_stride_size=64,
            _binding_generation=1,
        )
        return host_pool, device_pool

    def test_a_matched_transfer_still_passes(self):
        from sglang.srt.mem_cache.pool_host.mha import _guard_kv_transfer

        host_pool, device_pool = self._pools(4096, 4096)
        _guard_kv_transfer(
            host_pool,
            device_pool,
            torch.arange(3, dtype=torch.int64),
            torch.arange(3, dtype=torch.int64),
            where="unit",
        )

    def test_an_out_of_range_device_row_is_named_instead_of_clamped(self):
        from sglang.srt.mem_cache.kv_transfer_guard import KvTransferShapeMismatch
        from sglang.srt.mem_cache.pool_host.mha import _guard_kv_transfer

        host_pool, device_pool = self._pools(4096, 4096)
        with self.assertRaises(KvTransferShapeMismatch) as caught:
            _guard_kv_transfer(
                host_pool,
                device_pool,
                torch.arange(3, dtype=torch.int64),
                torch.tensor([10, 11, 200000], dtype=torch.int64),
                where="unit",
            )
        message = str(caught.exception)
        self.assertIn("out of bounds", message)
        self.assertIn("200000", message)
        self.assertIn("4096", message)

    def test_a_paged_host_layout_still_fails_open_on_the_host_side(self):
        from sglang.srt.mem_cache.pool_host.mha import _guard_kv_transfer

        host_pool, device_pool = self._pools(4096, 8)
        host_pool.layout = "page_first"
        # host_indices past k_data_refs' row count, which under page_first is a
        # page axis rather than a slot axis -- no capacity is asserted there.
        _guard_kv_transfer(
            host_pool,
            device_pool,
            torch.tensor([100, 101], dtype=torch.int64),
            torch.tensor([10, 11], dtype=torch.int64),
            where="unit-paged",
        )


if __name__ == "__main__":
    unittest.main()
