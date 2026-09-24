"""P-NOSYNC in the LAUNCH: flashinfer's prefill plan and FLA's chunk tables.

Once the plan stopped waiting for the running forward (memory_pool
``_nosync_mapping_rows``), the next host waits sit inside the launch of the
next chunk, both device->host reads of values the host already knows:

* flashinfer prefill ``plan()`` (paged + ragged) opens with
  ``qo_indptr.to("cpu")`` -- py-spy weg2xsn423 PP0 97 samples on
  flashinfer/prefill.py:1963;
* FLA ``prepare_chunk_indices`` / ``prepare_chunk_offsets`` read the chunk
  counts back (``.tolist()``) and send the table with a blocking pageable copy,
  once per table size at the first GDN layer (fla/index.py).

These tests pin, CPU only: the host tables equal the stock ones, the host path
never reads the device tensor, the plan gets host indptrs only when every
mirror agrees, the plan runs ONE plan ahead of the card (flashinfer refills a
pinned staging buffer per plan), and all of it only under SGLANG_WEG2_P_NOSYNC.
"""
import inspect
import os
import random
import types
import unittest
from functools import partial
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from sglang.srt.layers.attention import flashinfer_backend as fb
from sglang.srt.layers.attention import hybrid_linear_attn_backend as hlab
from sglang.srt.layers.attention.fla import index as fidx
from sglang.srt.managers import weg2_p_overlap as pov
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="stage-a-weg2-unit")


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


def _cu(lens, dtype=torch.int32):
    acc = [0]
    for n in lens:
        acc.append(acc[-1] + n)
    return torch.tensor(acc, dtype=dtype)


class _ReadProbe(torch.Tensor):
    """A cu_seqlens whose host reads are counted (the stock path reads it)."""

    reads = 0

    def tolist(self):
        _ReadProbe.reads += 1
        return super().tolist()

    def new_tensor(self, *a, **k):
        _ReadProbe.reads += 1
        return super().new_tensor(*a, **k)


# ------------------------------------------------------------------ FLA
class TestFlaHostTables(unittest.TestCase):
    CASES = [
        [4096], [512], [1], [63], [64], [65], [3000, 1096], [1, 1, 1],
        [4096, 0, 17], [0, 5], [130, 64, 2, 700],
    ]

    def test_indices_and_offsets_equal_the_stock_tables(self):
        rng = random.Random(7)
        cases = self.CASES + [[rng.randint(0, 9000) for _ in range(rng.randint(1, 4))]
                              for _ in range(40)]
        for lens in cases:
            for cs in (16, 64, 128):
                for dt in (torch.int32, torch.int64):
                    stock_i = fidx._prepare_chunk_indices(_cu(lens, dt), cs)
                    stock_o = fidx._prepare_chunk_offsets(_cu(lens, dt), cs)
                    cu = _cu(lens, dt)
                    self.assertTrue(fidx.note_host_seq_lens(cu, lens))
                    host_i = fidx._prepare_chunk_indices(cu, cs)
                    cu2 = _cu(lens, dt)  # a fresh object: the offsets cache is per object
                    fidx.note_host_seq_lens(cu2, lens)
                    host_o = fidx._prepare_chunk_offsets(cu2, cs)
                    msg = f"lens={lens} cs={cs} dtype={dt}"
                    self.assertEqual(host_i.dtype, stock_i.dtype, msg)
                    self.assertEqual(tuple(host_i.shape), tuple(stock_i.shape), msg)
                    self.assertTrue(torch.equal(host_i, stock_i), msg)
                    self.assertEqual(host_o.dtype, stock_o.dtype, msg)
                    self.assertTrue(torch.equal(host_o, stock_o), msg)

    def test_host_path_never_reads_the_device_tensor(self):
        lens = [3000, 1096]
        cu = _cu(lens).as_subclass(_ReadProbe)
        fidx.note_host_seq_lens(cu, lens)
        _ReadProbe.reads = 0
        fidx._prepare_chunk_indices(cu, 64)
        fidx._prepare_chunk_offsets(cu, 64)
        self.assertEqual(_ReadProbe.reads, 0)
        # the stock path does read it (the waits this removes)
        cu2 = _cu(lens).as_subclass(_ReadProbe)
        fidx._prepare_chunk_indices(cu2, 64)
        fidx._prepare_chunk_offsets(cu2, 64)
        self.assertGreaterEqual(_ReadProbe.reads, 2)

    def test_only_the_registered_object_takes_the_host_path(self):
        lens = [100]
        fidx.note_host_seq_lens(_cu(lens), [7])  # a different object, other lengths
        cu = _cu(lens)
        self.assertIsNone(fidx._host_seq_lens(cu))
        self.assertEqual(fidx._prepare_chunk_offsets(cu, 64).tolist(), [0, 2])

    def test_registration_refuses_what_cannot_be_the_lengths(self):
        cu = _cu([5, 6])
        self.assertFalse(fidx.note_host_seq_lens(cu, [5]))           # wrong count
        self.assertFalse(fidx.note_host_seq_lens(cu, [5, -1]))       # negative
        self.assertFalse(fidx.note_host_seq_lens(cu, None))          # gpu_only mirror
        self.assertIsNone(fidx._host_seq_lens(cu))
        self.assertTrue(fidx.note_host_seq_lens(cu, [5, 6]))
        self.assertEqual(fidx._host_seq_lens(cu), (5, 6))

    def test_registry_is_bounded(self):
        keep = [_cu([1]) for _ in range(3 * fidx._HOST_SEQ_LENS_CAP)]
        for t in keep:
            fidx.note_host_seq_lens(t, [1])
        self.assertLessEqual(len(fidx._HOST_SEQ_LENS), fidx._HOST_SEQ_LENS_CAP)
        self.assertEqual(fidx._host_seq_lens(keep[-1]), (1,))
        self.assertIsNone(fidx._host_seq_lens(keep[0]))

    def test_graph_static_pin_still_served_first(self):
        # H's pin wrapper (P prefill graph) keeps its permanent table; an
        # unregistered pinned tensor computes it the stock way, once
        cu = _cu([512])
        fidx.pin_graph_static_cu_seqlens(cu)
        try:
            t1 = fidx.prepare_chunk_indices(cu, 64)
            t2 = fidx.prepare_chunk_indices(cu, 64)
            self.assertIs(t1, t2)
            self.assertIs(fidx.graph_static_tables(cu)[("chunk_indices", 64)], t1)
        finally:
            fidx._GRAPH_STATIC_CU_SEQLENS.pop(id(cu), None)

    def test_backend_registers_under_the_switch_only(self):
        src = inspect.getsource(hlab.MambaAttnBackendBase._forward_metadata)
        i = src.index("if _weg2_p_overlap.p_nosync_on():")
        block = src[i:i + 400]
        self.assertIn("note_host_seq_lens(", block)
        self.assertIn("query_start_loc, forward_batch.extend_seq_lens_cpu", block)
        # registered after the device cu_seqlens is complete
        self.assertLess(src.index("query_start_loc[bs] = ("), i)


# ------------------------------------------------------------ flashinfer
class TestHostIndptrs(unittest.TestCase):
    def test_ragged_extend(self):
        # use_ragged: paged = prefix lens, qo = seq - prefix
        qo, kv, last = fb._p_nosync_host_indptrs(
            2, torch.tensor([4596, 700]), [500, 0], [500, 0], 500)
        self.assertEqual(qo.tolist(), [0, 4096, 4796])
        self.assertEqual(kv.tolist(), [0, 500, 500])
        self.assertEqual(last.tolist(), [1, 1])
        for t in (qo, kv, last):
            self.assertEqual(t.dtype, torch.int32)
            self.assertEqual(t.device.type, "cpu")

    def test_paged_only_extend(self):
        # not ragged: paged = seq lens (the same tensor as seq_lens_cpu)
        seq = torch.tensor([4096 + 8192])
        qo, kv, _ = fb._p_nosync_host_indptrs(1, seq, [8192], seq, 12288)
        self.assertEqual(qo.tolist(), [0, 4096])
        self.assertEqual(kv.tolist(), [0, 12288])

    def test_disagreeing_or_missing_mirrors_keep_the_stock_plan(self):
        seq = torch.tensor([600])
        self.assertIsNone(fb._p_nosync_host_indptrs(1, None, [0], [0], 0))
        self.assertIsNone(fb._p_nosync_host_indptrs(1, seq, None, [0], 0))
        self.assertIsNone(fb._p_nosync_host_indptrs(1, seq, [0], None, 0))
        self.assertIsNone(fb._p_nosync_host_indptrs(1, seq, [100], [100], 99))   # sum
        self.assertIsNone(fb._p_nosync_host_indptrs(2, seq, [100], [100], 100))  # bs
        self.assertIsNone(fb._p_nosync_host_indptrs(1, seq, [600], [600], 600))  # no new token


class _FakeEvent:
    log = None
    n = 0

    def __init__(self, *a, **k):
        self.id = _FakeEvent.n
        _FakeEvent.n += 1

    def record(self, *a):
        _FakeEvent.log.append(("record", self.id))

    def synchronize(self):
        _FakeEvent.log.append(("sync", self.id))


class TestOnePlanAhead(unittest.TestCase):
    def test_second_plan_waits_for_the_first_plans_copy_first(self):
        _FakeEvent.log, _FakeEvent.n = [], 0
        planned = []
        w = types.SimpleNamespace(begin_forward=lambda *a, **k: (
            planned.append(a), _FakeEvent.log.append(("plan", a[0]))))
        with mock.patch.object(torch.cuda, "Event", _FakeEvent):
            fb._p_nosync_plan(w, "k1", x=1)
            fb._p_nosync_plan(w, "k2", x=1)
        self.assertEqual(planned, [("k1",), ("k2",)])
        # plan k1, fence it; before plan k2: wait for k1's fence, then plan, fence
        self.assertEqual(_FakeEvent.log, [("plan", "k1"), ("record", 0),
                                          ("sync", 0), ("plan", "k2"), ("record", 1)])

    def test_gate(self):
        fa2 = types.SimpleNamespace(_backend="fa2", begin_forward=lambda *a, **k: None)
        auto = types.SimpleNamespace(_backend="auto", begin_forward=lambda *a, **k: None)
        cutlass = types.SimpleNamespace(_backend="cutlass", begin_forward=lambda *a, **k: None)
        fast = types.SimpleNamespace(_backend="fa2", begin_forward=partial(print))
        with _Env(False):
            self.assertFalse(fb._p_nosync_plan_ok(fa2))
        with _Env(True):
            self.assertTrue(fb._p_nosync_plan_ok(fa2))
            self.assertTrue(fb._p_nosync_plan_ok(None, auto))
            self.assertFalse(fb._p_nosync_plan_ok(auto, cutlass))
            self.assertFalse(fb._p_nosync_plan_ok(fast))  # fast_prefill_plan partial


class TestCallBeginForwardWiring(unittest.TestCase):
    def _src(self):
        return inspect.getsource(fb.FlashInferIndicesUpdaterPrefill.call_begin_forward)

    def test_host_indptrs_only_from_the_normal_extend_branch(self):
        src = self._src()
        self.assertLess(src.index("nosync_host = None"),
                        src.index("if spec_info is None and self.attn_backend.uneven_dcp:"))
        start = src.index("# Normal extend")
        normal = src[start:src.index("assert isinstance(spec_info, SpecInput)", start)]
        # the device vectors (read by the kv-indices kernel) are still built
        self.assertIn("kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)", normal)
        self.assertIn("qo_indptr[1 : bs + 1] = torch.cumsum(seq_lens - prefix_lens, dim=0)", normal)
        self.assertIn("nosync_host = _p_nosync_host_indptrs(", normal)
        for cond in ("custom_mask is None", "not use_sliding_window_kv_pool",
                     "multi_item_params.is_enabled()", "_p_nosync_plan_ok("):
            self.assertIn(cond, normal)
        self.assertEqual(src.count("nosync_host = _p_nosync_host_indptrs("), 1)

    def test_both_plans_take_the_host_indptrs_one_plan_ahead(self):
        src = self._src()
        self.assertIn("_qo_r = qo_indptr if nosync_host is None else nosync_host[0]", src)
        self.assertIn("else partial(_p_nosync_plan, wrapper_ragged)", src)
        self.assertIn("_qo_p, _kv_p, _last_p = qo_indptr, kv_indptr, self.kv_last_page_len[:bs]", src)
        self.assertIn("_qo_p, _kv_p, _last_p = nosync_host", src)
        self.assertIn("_plan_p = partial(_p_nosync_plan, wrapper_paged)", src)
        # the stock calls are the default
        self.assertIn("wrapper_ragged.begin_forward\n                if nosync_host is None", src)
        self.assertIn("_plan_p = wrapper_paged.begin_forward", src)


if __name__ == "__main__":
    unittest.main()
