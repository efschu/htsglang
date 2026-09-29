"""#281 D-CACHE-NOSYNC + WEG2-PREPARE-PARTS: group D's prepare_for_extend.

Measured reason (z30u 29.09., rc12z30u -e2cut, 83 post-wake passes per rank):
D TP0 ``prepare_ms`` median 1421 ms, linear in the admitted prefix (~5.6 ms per
1000 hit tokens), TP1/TP2 of the same passes 3 / 79 ms. TP0 is the Form A rank
with the mamba pool, and its prepare_for_extend runs the P-NOSYNC cache-path
sites (hybrid mapping write, mamba slot write, #924D ``.tolist()``) -- each a
host wait on the schedule stream behind the park loadback just queued.

These tests pin: the D switch turns on exactly the cache-path sites (not the
forward-plan sites that stay P's), unset is the stock path, and the
PREPARE-PARTS instrument names the segments without touching the device unless
asked. CPU only.
"""
import inspect
import logging
import os
import types
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from sglang.srt.managers import cache_controller as cc_mod
from sglang.srt.managers import schedule_batch as sb_mod
from sglang.srt.managers import weg2_p_overlap as pov
from sglang.srt.mem_cache.allocator import mamba as mamba_alloc
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="stage-a-weg2-unit")


class _Env:
    """Set exactly the given switches; restore everything on exit."""

    KEYS = (pov.P_NOSYNC_ENV, pov.D_CACHE_NOSYNC_ENV, sb_mod.PREPARE_PARTS_ENV)

    def __init__(self, **on):
        self.on = on

    def __enter__(self):
        self.old = {k: os.environ.get(k) for k in self.KEYS}
        for k in self.KEYS:
            os.environ.pop(k, None)
        for k, v in self.on.items():
            os.environ[k] = v
        return self

    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


D_ON = {pov.D_CACHE_NOSYNC_ENV: "1"}


class TestSwitch(unittest.TestCase):
    def test_cache_path_follows_either_group_switch(self):
        with _Env():
            self.assertFalse(pov.cache_path_nosync_on())
        with _Env(**D_ON):
            self.assertTrue(pov.cache_path_nosync_on())
            # group P's own switch (and the forward-plan sites behind it) stay off
            self.assertFalse(pov.p_nosync_on())
        with _Env(**{pov.P_NOSYNC_ENV: "1"}):
            self.assertTrue(pov.cache_path_nosync_on())
        self.assertEqual(pov.launcher_env_d_cache_nosync(), {pov.D_CACHE_NOSYNC_ENV: "1"})

    def test_forward_plan_sites_still_ask_only_p(self):
        from sglang.srt.layers.attention import flashinfer_backend as fib
        from sglang.srt.layers.attention import hybrid_linear_attn_backend as hlab

        self.assertIn("_weg2_p_overlap.p_nosync_on()", inspect.getsource(fib._p_nosync_plan_ok))
        self.assertNotIn("cache_path_nosync_on", inspect.getsource(fib))
        self.assertNotIn("cache_path_nosync_on", inspect.getsource(hlab))


class TestCachePathSitesUnderD(unittest.TestCase):
    def test_mamba_slot_allocator_reads_the_d_switch(self):
        with _Env():
            self.assertFalse(mamba_alloc.MambaSlotAllocator(8, device="cpu")._nosync)
        with _Env(**D_ON):
            self.assertTrue(mamba_alloc.MambaSlotAllocator(8, device="cpu")._nosync)

    def test_hybrid_pool_reads_the_d_switch(self):
        from sglang.srt.mem_cache import memory_pool as mp

        src = inspect.getsource(mp.HybridReqToTokenPool.__init__)
        self.assertIn("self._nosync = cache_path_nosync_on()", src)

    def test_controller_rowcheck_and_move_indices_read_the_d_switch(self):
        src = inspect.getsource(cc_mod)
        self.assertEqual(src.count("_weg2_p_overlap.cache_path_nosync_on()"), 2)
        self.assertNotIn("_weg2_p_overlap.p_nosync_on()", src)


class _TolistProbe(torch.Tensor):
    calls = 0

    def tolist(self):
        _TolistProbe.calls += 1
        return super().tolist()


class TestCowNoteUnderD(unittest.TestCase):
    """RED on the base: the #924D note `.tolist()`s the COW source (a blocking
    device read on TP0) although group D set its switch."""

    def _calls(self, **env):
        noted = []
        real_note, real_trail = mamba_alloc.note_924d, mamba_alloc._SLOT_TRAIL
        mamba_alloc.note_924d = lambda *a, **k: noted.append(k.get("extra"))
        mamba_alloc._SLOT_TRAIL = False
        _TolistProbe.calls = 0
        try:
            req = types.SimpleNamespace(
                mamba_cow_src_index=torch.tensor([3]).as_subclass(_TolistProbe),
                mamba_pool_idx=torch.tensor(1), mamba_needs_clear=True,
                mamba_pingpong_clear_indices=None, rid="r", last_node=None)
            batch = types.SimpleNamespace()
            with _Env(**env):
                sb_mod.ScheduleBatch._collect_deferred_mamba_cow_and_clear(batch, [req])
            calls = _TolistProbe.calls
        finally:
            mamba_alloc.note_924d, mamba_alloc._SLOT_TRAIL = real_note, real_trail
        self.assertEqual(batch.mamba_cow_src_indices.tolist(), [3])
        self.assertEqual(batch.mamba_cow_dst_indices.tolist(), [1])
        return calls, noted

    def test_stock_d_still_reads_it(self):
        calls, noted = self._calls()
        self.assertGreaterEqual(calls, 1)
        self.assertEqual(noted, ["cow_src=[3]"])

    def test_d_switch_never_reads_the_tensor(self):
        self.assertEqual(self._calls(**D_ON), (0, []))


class TestPrepareParts(unittest.TestCase):
    def test_off_is_none_and_costs_nothing(self):
        with _Env():
            self.assertIsNone(sb_mod._PrepareParts.begin())

    def test_mode_1_names_the_segments_without_a_device_call(self):
        real = torch.cuda.current_stream
        torch.cuda.current_stream = lambda *a, **k: self.fail("mode 1 touched the device")
        try:
            with _Env(**{sb_mod.PREPARE_PARTS_ENV: "1"}):
                pp = sb_mod._PrepareParts.begin()
        finally:
            torch.cuda.current_stream = real
        for name in ("head", "alloc", "reqs", "mamba_cow"):
            pp.mark(name)
        with self.assertLogs(sb_mod.logger, level=logging.INFO) as cm:
            pp.finish(types.SimpleNamespace(extend_num_tokens=113), [51520, 56256])
        line = cm.output[-1]
        self.assertIn("WEG2-PREPARE-PARTS bs=2 prefix_tokens=107776 extend_tokens=113", line)
        self.assertIn("entry_sync_ms=-1.0", line)
        for name in ("head=", "alloc=", "reqs=", "mamba_cow=", "tail="):
            self.assertIn(name, line)

    def test_prepare_for_extend_marks_every_segment_in_order(self):
        src = inspect.getsource(sb_mod.ScheduleBatch.prepare_for_extend)
        order = [src.index(f'_pp.mark("{n}")') for n in ("head", "alloc", "reqs", "mamba_cow")]
        self.assertEqual(order, sorted(order))
        self.assertLess(src.index("_PrepareParts.begin()"), order[0])
        self.assertGreater(src.index("_pp.finish(self, prefix_lens)"), order[-1])


if __name__ == "__main__":
    unittest.main()
