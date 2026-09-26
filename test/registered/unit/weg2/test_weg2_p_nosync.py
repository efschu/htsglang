"""P-NOSYNC: no host wait on a CUDA stream in group P's per-chunk cache path.

Measured reason (weg2xsn422, --p-host-overlap + #PGAP): the host sat 409-604 ms
per 4096 chunk in the ANCHOR; py-spy on PP0 put 1731 of 2357 samples on
``MambaSlotAllocator._do_alloc``'s ``self.slot_used[select_index] = True`` -- a
Python scalar assigned into a CUDA tensor is copied host->device BLOCKING
(cudaStreamSynchronize on the schedule stream, which carries the fence of the
running forward). These tests pin the three sync-free forms, that each is behind
``SGLANG_WEG2_P_NOSYNC`` (unset = the stock lines), and that they compute the same
answers. CPU only: the CUDA-only branches are pinned by source shape, their
semantics by the CPU equivalents.
"""
import inspect
import os
import types
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from sglang.srt.managers import cache_controller as cc_mod
from sglang.srt.managers import weg2_p_overlap as pov
from sglang.srt.mem_cache.allocator import mamba as mamba_alloc
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="stage-a-weg2-unit")


class _Env:
    def __init__(self, on):
        self.on = on

    def __enter__(self):
        self.old = os.environ.get(pov.P_NOSYNC_ENV)
        if self.on:
            os.environ[pov.P_NOSYNC_ENV] = "1"
        else:
            os.environ.pop(pov.P_NOSYNC_ENV, None)

    def __exit__(self, *a):
        if self.old is None:
            os.environ.pop(pov.P_NOSYNC_ENV, None)
        else:
            os.environ[pov.P_NOSYNC_ENV] = self.old


class TestSwitch(unittest.TestCase):
    def test_env_and_launcher_set(self):
        with _Env(False):
            self.assertFalse(pov.p_nosync_on())
        with _Env(True):
            self.assertTrue(pov.p_nosync_on())
        # --p-host-overlap carries it for group P
        self.assertEqual(pov.launcher_env_p_host_overlap().get(pov.P_NOSYNC_ENV), "1")


class TestMambaSlotAllocator(unittest.TestCase):
    def _run(self, on):
        with _Env(on):
            a = mamba_alloc.MambaSlotAllocator(8, device="cpu")
        self.assertEqual(a._nosync, on)
        s1 = a.alloc(1)
        s2 = a.alloc(2)
        used_after_alloc = a.slot_used.clone()
        a.free(s2)
        used_after_free = a.slot_used.clone()
        return s1, s2, used_after_alloc, used_after_free, a

    def test_index_fill_is_the_same_ledger_as_the_scalar_assignment(self):
        off = self._run(False)
        on = self._run(True)
        for x, y in zip(off[:4], on[:4]):
            self.assertTrue(torch.equal(x, y))

    def test_double_free_still_refused_under_nosync(self):
        _, s2, _, _, a = self._run(True)
        with self.assertRaises(Exception):
            a.free(s2)  # second release of the same slots

    def test_pinned_answer_is_what_the_drain_reads(self):
        with _Env(True):
            a = mamba_alloc.MambaSlotAllocator(4, device="cpu")
        refused = []
        a._refuse_double_free = lambda idx, stack=None: refused.append(idx.tolist())
        done = types.SimpleNamespace(query=lambda: True, synchronize=lambda: None)
        already = torch.tensor([False, True])
        safe = torch.tensor([1, 2])
        # the device answer says "hit"; the pinned copy is what counts
        a._1467_pending = [(done, already, safe, "", torch.tensor(False))]
        a._drain_double_free_checks()
        self.assertEqual(refused, [])
        a._1467_pending = [(done, already, safe, "", torch.tensor(True))]
        a._drain_double_free_checks()
        self.assertEqual(refused, [[2]])
        # a pending answer whose event is not complete is kept, not waited on
        busy = types.SimpleNamespace(query=lambda: False, synchronize=lambda: None)
        a._1467_pending = [(busy, already, safe, "", torch.tensor(True))]
        a._drain_double_free_checks()
        self.assertEqual(len(a._1467_pending), 1)

    def test_source_keeps_the_stock_lines_behind_the_flag(self):
        src = inspect.getsource(mamba_alloc.MambaSlotAllocator._do_alloc)
        self.assertIn("self.slot_used.index_fill_(0, select_index, True)", src)
        self.assertIn("self.slot_used[select_index] = True", src)
        src = inspect.getsource(mamba_alloc.MambaSlotAllocator.free)
        self.assertIn("self.slot_used.index_fill_(0, safe, False)", src)
        self.assertIn("self.slot_used[safe] = False", src)
        src = inspect.getsource(mamba_alloc.MambaSlotAllocator._defer_double_free_check)
        self.assertIn("pin_memory=True", src)
        self.assertIn("non_blocking=True", src)


class TestHiCacheController(unittest.TestCase):
    def _ctl(self):
        c = types.SimpleNamespace()
        c._923_drain_deferred = types.MethodType(
            cc_mod.HiCacheController._923_drain_deferred, c)
        return c

    def test_deferred_bound_check_raises_on_a_hit_only_once_complete(self):
        c = self._ctl()
        done = types.SimpleNamespace(query=lambda: True, synchronize=lambda: None)
        busy = types.SimpleNamespace(query=lambda: False, synchronize=lambda: None)
        c._923_deferred = [(done, torch.tensor(False), "write", 100)]
        self.assertEqual(c._923_drain_deferred(), 1)
        c._923_deferred = [(busy, torch.tensor(True), "write", 100)]
        self.assertEqual(c._923_drain_deferred(), 0)          # not complete: kept
        self.assertEqual(len(c._923_deferred), 1)
        c._923_deferred = [(done, torch.tensor(True), "write", 100)]
        with self.assertRaises(RuntimeError) as cm:
            c._923_drain_deferred()
        self.assertIn("#923", str(cm.exception))

    def test_rowcheck_defers_only_for_identity_rows_on_the_device(self):
        src = inspect.getsource(cc_mod.HiCacheController._refuse_unaddressable_kv_rows)
        i = src.index("_weg2_p_overlap.p_nosync_on()")
        gate = src[i:src.index("self._923_defer_bound_check", i)]
        self.assertIn("rows.is_cuda", gate)
        self.assertIn("self._dcp_owner_ctx() is None", gate)
        # the host read stays for everything else
        self.assertIn("row_max = int(rows.max())", src)

    def test_move_indices_keeps_device_indices_on_the_device(self):
        src = inspect.getsource(cc_mod.HiCacheController.move_indices)
        i = src.index("_weg2_p_overlap.p_nosync_on()")
        branch = src[i:src.index("device_indices = device_indices.cpu()", i)]
        self.assertIn("device_indices.is_cuda", branch)
        self.assertIn("non_blocking=True", branch)
        self.assertIn("device_indices.index_select(0, idx)", branch)

    def test_move_indices_cpu_path_unchanged(self):
        # CPU device indices never take the nosync branch: same pairs as stock
        c = types.SimpleNamespace(io_backend="direct",
                                  mem_pool_host=types.SimpleNamespace(layout="layer_first"))
        mv = types.MethodType(cc_mod.HiCacheController.move_indices, c)
        host = torch.tensor([7, 3, 5])
        dev = torch.tensor([70, 30, 50])
        for on in (False, True):
            with _Env(on):
                h, d = mv(host.clone(), dev.clone())
            self.assertEqual(h.tolist(), [3, 5, 7])
            self.assertEqual(d.tolist(), [30, 50, 70])


if __name__ == "__main__":
    unittest.main()
