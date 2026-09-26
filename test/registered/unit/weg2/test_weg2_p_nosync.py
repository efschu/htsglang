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


# ---------------------------------------------------------------------------
# SECOND SITE, in the PLAN (weg2xsn423, anchor 0 with the fixes above): py-spy
# PP0 1538 of 2263 samples on HybridReqToTokenPool.alloc's
# `mapping[select_index] = t` -- a Python row list as the index, moved to the
# device BLOCKING by index_put_. #PGAP plan 417-495 ms per 4096 chunk.
# ---------------------------------------------------------------------------
def _hybrid_pool(nosync):
    """A HybridReqToTokenPool on CPU, built with the switch in the given state."""
    from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
    from sglang.srt.environ import envs
    from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    sa = ServerArgs(model_path="dummy", page_size=1)
    sa._mamba_cache_chunk_size = CHUNK_SIZE
    set_global_server_args_for_scheduler(sa)
    layers = [0, 1, 2]
    with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
        shape = Mamba2StateShape.create(
            tp_world_size=1, intermediate_size=512, n_groups=4, num_heads=8,
            head_dim=64, state_size=32, conv_kernel=4,
        )
        params = Mamba2CacheParams(shape=shape, layers=layers)
    with _Env(nosync):
        return HybridReqToTokenPool(
            size=4, mamba_size=6, mamba_spec_state_size=4, max_context_len=64,
            device="cpu", enable_memory_saver=False, cache_params=params,
            mamba_layer_ids=layers, enable_mamba_extra_buffer=False,
            enable_linear_replayssm=False,
        )


def _req(rid, ids):
    from array import array

    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    return Req(rid=rid, origin_input_text="", origin_input_ids=array("q", ids),
               sampling_params=SamplingParams(temperature=0, max_new_tokens=1))


class TestPlanMappingRows(unittest.TestCase):
    def test_switch_is_read_at_construction(self):
        self.assertTrue(_hybrid_pool(True)._nosync)
        self.assertFalse(_hybrid_pool(False)._nosync)

    def test_same_mapping_either_way(self):
        got = []
        for on in (False, True):
            pool = _hybrid_pool(on)
            reqs = [_req("a", [1, 2, 3]), _req("b", [4, 5])]
            rows = pool.alloc(reqs)
            got.append([(r, int(pool.req_index_to_mamba_index_mapping[r]))
                        for r in rows])
            for r, req in zip(rows, reqs):
                self.assertEqual(int(pool.req_index_to_mamba_index_mapping[r]),
                                 int(req.mamba_pool_idx))
            # a continuing chunk (committed KV, row kept) rewrites the same row
            # with the same slot -- the per-chunk write py-spy caught
            reqs[0].kv_committed_len = 3
            self.assertEqual(pool.alloc(reqs[:1]), [rows[0]])
            self.assertEqual(int(pool.req_index_to_mamba_index_mapping[rows[0]]),
                             int(reqs[0].mamba_pool_idx))
        self.assertEqual(got[0], got[1])

    def test_nosync_alloc_writes_through_the_device_rows(self):
        pool = _hybrid_pool(True)
        calls = []

        def rows_as_tensor(select_index):  # the stand-in for the device copy
            calls.append(list(select_index))
            return torch.tensor(select_index, dtype=torch.int64)

        pool._nosync_mapping_rows = rows_as_tensor
        reqs = [_req("a", [1, 2, 3]), _req("b", [4, 5])]
        rows = pool.alloc(reqs)
        self.assertEqual(calls, [rows])
        self.assertEqual([int(pool.req_index_to_mamba_index_mapping[r]) for r in rows],
                         [int(q.mamba_pool_idx) for q in reqs])
        # the stock pool never asks for device rows
        stock = _hybrid_pool(False)
        stock._nosync_mapping_rows = lambda s: self.fail("stock path took the nosync rows")
        stock.alloc([_req("c", [6])])

    def test_cuda_mapping_rows_go_pinned_and_non_blocking(self):
        from sglang.srt.mem_cache import memory_pool as mp

        seen = []
        real = mp._pinned_device_rows
        mp._pinned_device_rows = lambda rows, device: seen.append((list(rows), device)) or "DEV"
        try:
            holder = types.SimpleNamespace(req_index_to_mamba_index_mapping=types.SimpleNamespace(
                is_cuda=True, device="cuda:3"))
            self.assertEqual(mp.HybridReqToTokenPool._nosync_mapping_rows(holder, [5, 2]), "DEV")
            self.assertEqual(seen, [([5, 2], "cuda:3")])
            # a CPU mapping keeps the list (nothing to copy, nothing to wait for)
            holder.req_index_to_mamba_index_mapping = torch.zeros(8, dtype=torch.int32)
            self.assertEqual(mp.HybridReqToTokenPool._nosync_mapping_rows(holder, [5, 2]), [5, 2])
        finally:
            mp._pinned_device_rows = real
        src = inspect.getsource(real)
        self.assertIn("pin_memory=True", src)
        self.assertIn("non_blocking=True", src)

    def test_source_keeps_the_stock_index_behind_the_flag(self):
        from sglang.srt.mem_cache import memory_pool as mp

        src = inspect.getsource(mp.HybridReqToTokenPool.alloc)
        i = src.index("rows = select_index")
        tail = src[i:]
        self.assertIn('if getattr(self, "_nosync", False):', tail)
        self.assertIn("rows = self._nosync_mapping_rows(select_index)", tail)
        self.assertIn("self.req_index_to_mamba_index_mapping[rows] = mamba_index_tensor", tail)


class _TolistProbe(torch.Tensor):
    calls = 0

    def tolist(self):
        _TolistProbe.calls += 1
        return super().tolist()


class TestCowNoteIsNotEvaluatedOff(unittest.TestCase):
    """The #924D first_state note's `extra=` text `.tolist()`s a CUDA tensor
    before note_924d can decline -- a blocking read in the plan of every
    prefix-hit request."""

    def _collect(self, nosync, trail):
        from sglang.srt.managers.schedule_batch import ScheduleBatch

        noted = []
        real_note, real_trail = mamba_alloc.note_924d, mamba_alloc._SLOT_TRAIL
        mamba_alloc.note_924d = lambda *a, **k: noted.append(k.get("extra"))
        mamba_alloc._SLOT_TRAIL = trail
        _TolistProbe.calls = 0
        try:
            req = types.SimpleNamespace(
                mamba_cow_src_index=torch.tensor([3]).as_subclass(_TolistProbe),
                mamba_pool_idx=torch.tensor(1), mamba_needs_clear=True,
                mamba_pingpong_clear_indices=None, rid="r", last_node=None)
            batch = types.SimpleNamespace()
            with _Env(nosync):
                ScheduleBatch._collect_deferred_mamba_cow_and_clear(batch, [req])
            calls = _TolistProbe.calls  # before this test's own reads below
        finally:
            mamba_alloc.note_924d, mamba_alloc._SLOT_TRAIL = real_note, real_trail
        # the COW itself is collected in every case
        self.assertEqual(batch.mamba_cow_src_indices.tolist(), [3])
        self.assertEqual(batch.mamba_cow_dst_indices.tolist(), [1])
        self.assertIsNone(req.mamba_cow_src_index)
        return calls, noted

    def test_stock_evaluates_it(self):
        calls, noted = self._collect(nosync=False, trail=False)
        self.assertGreaterEqual(calls, 1)
        self.assertEqual(noted, ["cow_src=[3]"])

    def test_nosync_with_the_trail_off_never_reads_the_tensor(self):
        calls, noted = self._collect(nosync=True, trail=False)
        self.assertEqual((calls, noted), (0, []))

    def test_nosync_with_the_trail_on_still_notes(self):
        calls, noted = self._collect(nosync=True, trail=True)
        self.assertGreaterEqual(calls, 1)
        self.assertEqual(noted, ["cow_src=[3]"])

    def test_slot_trail_on_reads_the_module_flag(self):
        real = mamba_alloc._SLOT_TRAIL
        try:
            for v in (True, False):
                mamba_alloc._SLOT_TRAIL = v
                self.assertIs(mamba_alloc.slot_trail_on(), v)
        finally:
            mamba_alloc._SLOT_TRAIL = real


if __name__ == "__main__":
    unittest.main()
